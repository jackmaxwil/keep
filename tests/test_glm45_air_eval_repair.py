from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


def _load_repair_cli():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "repair_glm45_air_eval_evidence.py"
    spec = importlib.util.spec_from_file_location("repair_glm45_air_eval_evidence_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(row_index: int, *, clean: bool, mean_kld: float | None = None) -> dict:
    return {
        "schema_version": 1,
        "prompt_id": f"report_math_{row_index:03d}",
        "row_index": row_index,
        "nll_delta": 0.01 * (row_index + 1),
        "ppl_ratio": 1.01,
        "mean_kld": 0.1 if mean_kld is None else mean_kld,
        "token_klds": [0.1 if mean_kld is None else mean_kld],
        "top1_agreement": 1.0,
        "pageouts_delta": 0 if clean else 3,
        "swapouts_delta": 0,
    }


def test_repair_eval_evidence_copies_complete_clean_existing_without_spawn(tmp_path: Path) -> None:
    module = _load_repair_cli()
    teacher_jsonl = tmp_path / "teacher.jsonl"
    existing_jsonl = tmp_path / "existing.jsonl"
    merged_jsonl = tmp_path / "merged.jsonl"
    _write_jsonl(teacher_jsonl, [{"prompt_id": "row0"}, {"prompt_id": "row1"}])
    _write_jsonl(existing_jsonl, [_record(0, clean=True), _record(1, clean=True)])

    def _no_spawn(*args, **kwargs):
        raise AssertionError("clean repair must not spawn eval")

    summary = module.repair_eval_evidence(
        existing_jsonl=existing_jsonl,
        teacher_jsonl=teacher_jsonl,
        append_jsonl=merged_jsonl,
        artifact_dir=tmp_path / "artifact",
        engine="vq_e1_routed_nax_e8p",
        max_rows=2,
        spawn=_no_spawn,
    )

    rows = [json.loads(line) for line in merged_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["row_index"] for row in rows] == [0, 1]
    assert summary["repair_row_indices"] == []
    assert summary["clean_record_count"] == 2


def test_repair_eval_evidence_reruns_only_dirty_or_missing_rows(tmp_path: Path) -> None:
    module = _load_repair_cli()
    teacher_jsonl = tmp_path / "teacher.jsonl"
    existing_jsonl = tmp_path / "existing.jsonl"
    merged_jsonl = tmp_path / "merged.jsonl"
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_jsonl(
        teacher_jsonl,
        [
            {"prompt_id": "row0"},
            {"prompt_id": "row1"},
            {"prompt_id": "row2"},
            {"prompt_id": "row3"},
        ],
    )
    _write_jsonl(
        existing_jsonl,
        [_record(0, clean=True), _record(1, clean=False), _record(3, clean=True)],
    )
    spawned: list[list[str]] = []

    def _spawn(argv: list[str], **kwargs):
        spawned.append(argv)
        repair_jsonl = Path(argv[argv.index("--append-jsonl") + 1])
        _write_jsonl(repair_jsonl, [_record(1, clean=True, mean_kld=0.01), _record(2, clean=True)])
        return subprocess.CompletedProcess(argv, 0)

    summary = module.repair_eval_evidence(
        existing_jsonl=existing_jsonl,
        teacher_jsonl=teacher_jsonl,
        append_jsonl=merged_jsonl,
        artifact_dir=artifact_dir,
        engine="vq_e1_routed_nax_e8p",
        max_rows=4,
        mlx_cache_limit_gb=0,
        mlx_clear_cache_before_load=True,
        spawn=_spawn,
    )

    assert len(spawned) == 1
    argv = spawned[0]
    assert argv[argv.index("--row-indices") + 1] == "1,2"
    assert argv[argv.index("--artifact-dir") + 1] == str(artifact_dir)
    assert argv[argv.index("--teacher-jsonl") + 1] == str(teacher_jsonl)
    assert argv[argv.index("--engine") + 1] == "vq_e1_routed_nax_e8p"
    assert argv[argv.index("--mlx-cache-limit-gb") + 1] == "0"
    assert "--mlx-clear-cache-before-load" in argv
    rows = [json.loads(line) for line in merged_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["row_index"] for row in rows] == [0, 1, 2, 3]
    assert rows[1]["mean_kld"] == 0.01
    assert summary["repair_row_indices"] == [1, 2]
    assert summary["clean_record_count"] == 4

