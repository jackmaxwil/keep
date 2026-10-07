from __future__ import annotations

import json
from pathlib import Path

import pytest

from keep.build import get_model
from keep.build.cli import main as keep_main


def _write_snapshot(
    cache_root: Path,
    *,
    model_id: str = "Example/Model",
    revision: str = "abc123",
    shard_sizes: tuple[int, ...] = (1024, 2048),
    include_config: bool = True,
    include_index: bool = True,
    missing_shards: set[int] | None = None,
) -> Path:
    missing_shards = missing_shards or set()
    snapshot = (
        cache_root
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    if include_config:
        (snapshot / "config.json").write_text("{}\n")
    shard_names = [
        f"model-{index:05d}-of-{len(shard_sizes):05d}.safetensors"
        for index in range(1, len(shard_sizes) + 1)
    ]
    if include_index:
        weight_map = {
            f"layers.{index}.weight": shard_name
            for index, shard_name in enumerate(shard_names)
        }
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}) + "\n"
        )
    for index, (shard_name, size) in enumerate(zip(shard_names, shard_sizes), start=1):
        if index in missing_shards:
            continue
        (snapshot / shard_name).write_bytes(b"x" * size)
    return snapshot


def test_resolves_profile_name_to_pinned_hf_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_stage(request: get_model.ModelGetRequest) -> get_model.ModelReadiness:
        captured["request"] = request
        return get_model.ModelReadiness(
            model_id=request.model_id,
            revision=request.revision,
            snapshot_path=Path("/tmp/snapshot"),
            profile_name=request.profile_name,
            check_only=True,
            config_present=True,
            index_present=True,
            shards_present=1,
            shards_total=1,
            missing_shards=(),
            total_bytes=0,
            approx_bf16_gb=request.approx_bf16_gb,
            machine_ram_gb=128.0,
            downloaded=(),
        )

    monkeypatch.setattr(get_model, "stage_model", fake_stage)

    assert keep_main(["get", "qwen36-35b-a3b", "--check-only"]) == 0

    request = captured["request"]
    assert isinstance(request, get_model.ModelGetRequest)
    assert request.profile_name == "qwen36-35b-a3b"
    assert request.model_id == "Qwen/Qwen3.6-35B-A3B"
    assert request.revision == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert request.approx_bf16_gb == 70


def test_resolves_raw_hf_id_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_stage(request: get_model.ModelGetRequest) -> get_model.ModelReadiness:
        captured["request"] = request
        return get_model.ModelReadiness(
            model_id=request.model_id,
            revision=request.revision,
            snapshot_path=Path("/tmp/snapshot"),
            profile_name=None,
            check_only=True,
            config_present=True,
            index_present=True,
            shards_present=1,
            shards_total=1,
            missing_shards=(),
            total_bytes=0,
            approx_bf16_gb=None,
            machine_ram_gb=128.0,
            downloaded=(),
        )

    monkeypatch.setattr(get_model, "stage_model", fake_stage)

    assert keep_main(["get", "Org/Raw", "--revision", "rev1", "--check-only"]) == 0

    request = captured["request"]
    assert isinstance(request, get_model.ModelGetRequest)
    assert request.profile_name is None
    assert request.model_id == "Org/Raw"
    assert request.revision == "rev1"
    assert request.approx_bf16_gb is None


def test_check_only_present_snapshot_reports_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "hub"
    snapshot = _write_snapshot(cache_root)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(get_model, "machine_ram_gb", lambda: 128.0)

    rc = keep_main(["get", "Example/Model", "--revision", "abc123", "--check-only"])

    assert rc == 0
    output = capsys.readouterr().out
    assert f"snapshot: {snapshot}" in output
    assert "config.json: present" in output
    assert "model.safetensors.index.json: present" in output
    assert "shards: 2/2 present" in output
    assert "total: 0.000003 GB" in output


def test_check_only_raw_main_resolves_cached_ref_to_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "hub"
    revision = "a" * 40
    snapshot = _write_snapshot(cache_root, revision=revision)
    refs = cache_root / "models--Example--Model" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(revision + "\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(get_model, "machine_ram_gb", lambda: 128.0)

    rc = keep_main(["get", "Example/Model", "--check-only"])

    assert rc == 0
    output = capsys.readouterr().out
    assert f"revision: {revision}" in output
    assert f"snapshot: {snapshot}" in output


def test_downloaded_main_reports_resolved_snapshot_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "hub"
    revision = "b" * 40
    snapshot = _write_snapshot(cache_root, revision=revision)
    index_path = snapshot / "model.safetensors.index.json"
    index_contents = index_path.read_text()
    index_path.unlink()
    calls: list[tuple[str, str]] = []
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(get_model, "machine_ram_gb", lambda: 128.0)

    def fake_hf_hub_download(
        _model_id: str,
        filename: str,
        *,
        revision: str,
    ) -> str:
        calls.append((filename, revision))
        if filename == "model.safetensors.index.json":
            index_path.write_text(index_contents)
        return str(snapshot / filename)

    monkeypatch.setattr(
        get_model,
        "hf_hub_download",
        fake_hf_hub_download,
    )

    rc = keep_main(["get", "Example/Model"])

    assert rc == 0
    output = capsys.readouterr().out
    assert f"revision: {revision}" in output
    assert f"snapshot: {snapshot}" in output
    assert calls == [
        ("config.json", "main"),
        ("model.safetensors.index.json", revision),
    ]


def test_check_only_cached_main_ref_stays_offline_and_incomplete_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "hub"
    revision = "c" * 40
    _write_snapshot(cache_root, revision=revision, missing_shards={2})
    refs = cache_root / "models--Example--Model" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(revision + "\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(get_model, "machine_ram_gb", lambda: 128.0)
    monkeypatch.setattr(
        get_model,
        "hf_hub_download",
        lambda *_args, **_kwargs: pytest.fail("check-only must remain offline"),
    )
    monkeypatch.setattr(
        get_model,
        "snapshot_download",
        lambda *_args, **_kwargs: pytest.fail("check-only must remain offline"),
    )

    rc = keep_main(["get", "Example/Model", "--check-only"])

    assert rc == 1
    output = capsys.readouterr().out
    assert f"revision: {revision}" in output
    assert "shards: 1/2 present" in output
    assert "ready: no" in output


def test_explicit_commit_revision_cannot_be_remapped_by_cache_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "hub"
    pinned_revision = "d" * 40
    other_revision = "e" * 40
    pinned_snapshot = _write_snapshot(cache_root, revision=pinned_revision)
    _write_snapshot(cache_root, revision=other_revision)
    refs = cache_root / "models--Example--Model" / "refs"
    refs.mkdir(parents=True)
    (refs / pinned_revision).write_text(other_revision + "\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(get_model, "machine_ram_gb", lambda: 128.0)

    rc = keep_main(
        [
            "get",
            "Example/Model",
            "--revision",
            pinned_revision,
            "--check-only",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert f"revision: {pinned_revision}" in output
    assert f"snapshot: {pinned_snapshot}" in output


def test_download_mode_refreshes_symbolic_main_before_pinning_followup_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "hub"
    stale_revision = "1" * 40
    current_revision = "2" * 40
    _write_snapshot(cache_root, revision=stale_revision)
    current_snapshot = _write_snapshot(cache_root, revision=current_revision)
    current_index = current_snapshot / "model.safetensors.index.json"
    index_contents = current_index.read_text()
    current_index.unlink()
    refs = cache_root / "models--Example--Model" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(stale_revision + "\n")
    calls: list[tuple[str, str]] = []

    def fake_hf_hub_download(
        _model_id: str,
        filename: str,
        *,
        revision: str,
    ) -> str:
        calls.append((filename, revision))
        if filename == "model.safetensors.index.json":
            current_index.write_text(index_contents)
        return str(current_snapshot / filename)

    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(get_model, "machine_ram_gb", lambda: 128.0)
    monkeypatch.setattr(get_model, "hf_hub_download", fake_hf_hub_download)

    rc = keep_main(["get", "Example/Model"])

    assert rc == 0
    output = capsys.readouterr().out
    assert f"revision: {current_revision}" in output
    assert f"snapshot: {current_snapshot}" in output
    assert calls == [
        ("config.json", "main"),
        ("model.safetensors.index.json", current_revision),
    ]


def test_check_only_missing_snapshot_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "hub"
    _write_snapshot(cache_root, missing_shards={2})
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(get_model, "machine_ram_gb", lambda: 128.0)

    rc = keep_main(["get", "Example/Model", "--revision", "abc123", "--check-only"])

    assert rc == 1
    output = capsys.readouterr().out
    assert "shards: 1/2 present" in output
    assert "missing:" in output
    assert "model-00002-of-00002.safetensors" in output


def test_profile_readiness_summary_includes_fit_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "hub"
    _write_snapshot(
        cache_root,
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(get_model, "machine_ram_gb", lambda: 128.0)

    rc = keep_main(["get", "qwen36-35b-a3b", "--check-only"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "model: qwen36-35b-a3b (Qwen/Qwen3.6-35B-A3B)" in output
    assert "revision: 995ad96eacd98c81ed38be0c5b274b04031597b0" in output
    assert "teacher ~70GB, machine 128GB: fits single-host" in output


def test_download_mode_uses_staging_download_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "hub"
    snapshot = _write_snapshot(
        cache_root,
        include_config=False,
        include_index=False,
        missing_shards={2},
    )
    calls: list[tuple[str, str]] = []

    def fake_hf_hub_download(model_id: str, filename: str, *, revision: str | None = None) -> str:
        calls.append(("file", filename))
        path = snapshot / filename
        if filename == "config.json":
            path.write_text("{}\n")
        elif filename == "model.safetensors.index.json":
            path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layers.0.weight": "model-00001-of-00002.safetensors",
                            "layers.1.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                )
                + "\n"
            )
        return str(path)

    def fake_snapshot_download(
        model_id: str,
        *,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        local_dir: str | None = None,
        max_workers: int = 8,
    ) -> str:
        calls.append(("snapshot", ",".join(allow_patterns or [])))
        for filename in allow_patterns or []:
            (snapshot / filename).write_bytes(b"x")
        return str(snapshot)

    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(get_model, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(get_model, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(get_model, "machine_ram_gb", lambda: 128.0)

    rc = keep_main(["get", "Example/Model", "--revision", "abc123"])

    assert rc == 0
    assert ("file", "config.json") in calls
    assert ("file", "model.safetensors.index.json") in calls
    assert ("snapshot", "model-00002-of-00002.safetensors") in calls
