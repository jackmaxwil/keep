from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.export_glm45_air_cleanroom_cache import (
    _json_value_or_file,
    _rank_view_roots_json_from_dir,
    main,
)


def test_rank_view_roots_json_from_dir_sorts_numeric_rank_dirs(tmp_path: Path) -> None:
    view_dir = tmp_path / "views"
    view_dir.mkdir()
    (view_dir / "rank-10").mkdir()
    (view_dir / "rank-2").mkdir()
    (view_dir / "rank-0").mkdir()
    (view_dir / "notes.txt").write_text("ignored")

    roots = json.loads(_rank_view_roots_json_from_dir(view_dir))

    assert list(roots) == ["0", "2", "10"]
    assert roots == {
        "0": str(view_dir / "rank-0"),
        "2": str(view_dir / "rank-2"),
        "10": str(view_dir / "rank-10"),
    }


def test_rank_view_roots_json_from_dir_rejects_missing_rank_dirs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no rank view directories"):
        _rank_view_roots_json_from_dir(tmp_path)


def test_json_value_or_file_reads_file_paths(tmp_path: Path) -> None:
    payload = '{"lower-pre":{"1":"/views/lower-pre"}}'
    path = tmp_path / "stage-roots.json"
    path.write_text(payload + "\n")

    assert _json_value_or_file(str(path)) == payload
    assert _json_value_or_file(payload) == payload
    assert _json_value_or_file('[{"rank":0}]') == '[{"rank":0}]'


def test_cleanroom_adapter_forwards_peer_view_roots_and_remote_worker_flags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def fake_run(command: list[str], *, env: dict[str, str]) -> object:
        del command
        captured.update(env)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_glm45_air_cleanroom_cache.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--rank-view-roots-json",
            '{"0":"/rank0","1":"/rank1"}',
            "--peer-rank0-view-root",
            "artifacts/peer-rank0",
            "--peer-rank1-view-root",
            "artifacts/peer-rank1",
            "--peer-source-dir",
            "/peer/source",
            "--peer-source-stage-ssh",
            "jackmazac@192.168.10.2",
            "--stage-peer-source",
            "--peer-source-min-free-gb",
            "16",
            "--local-sequential-remote-workers",
            "head",
            "--local-sequential-remote-tmp-dir",
            "/tmp/glm-p2-remote-workers",
            "--local-sequential-remote-dirty-retries",
            "1",
            "--local-sequential-remote-retry-sleep-seconds",
            "30",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 0
    assert captured["GLM_PEER_RANK0_VIEW_ROOT"] == "artifacts/peer-rank0"
    assert captured["GLM_PEER_RANK1_VIEW_ROOT"] == "artifacts/peer-rank1"
    assert captured["GLM_PEER_SOURCE_DIR"] == "/peer/source"
    assert captured["GLM_PEER_SOURCE_STAGE_SSH"] == "jackmazac@192.168.10.2"
    assert captured["GLM_STAGE_PEER_SOURCE"] == "1"
    assert captured["GLM_PEER_SOURCE_MIN_FREE_GB"] == "16.0"
    assert captured["GLM_LOCAL_SEQUENTIAL_REMOTE_WORKERS"] == "head"
    assert captured["GLM_LOCAL_SEQUENTIAL_REMOTE_TMP_DIR"] == "/tmp/glm-p2-remote-workers"
    assert captured["GLM_LOCAL_SEQUENTIAL_REMOTE_DIRTY_RETRIES"] == "1"
    assert captured["GLM_LOCAL_SEQUENTIAL_REMOTE_RETRY_SLEEP_SECONDS"] == "30.0"
