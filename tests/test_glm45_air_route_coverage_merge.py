from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_merge_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "merge_glm45_air_route_coverage_cache.py"
    )
    spec = importlib.util.spec_from_file_location("route_coverage_merge_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_cache(cache_dir: Path, rows: list[dict]) -> None:
    (cache_dir / "teacher_logits").mkdir(parents=True)
    with (cache_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            shard = row["shard"]
            (cache_dir / shard).write_bytes(b"tensor")
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _row(
    prompt_id: str,
    *,
    pageouts_delta: int = 0,
    route_coverage: dict | None = None,
    route_trace: dict | None = None,
) -> dict:
    row = {
        "prompt_id": prompt_id,
        "pageouts_delta": pageouts_delta,
        "swapouts_delta": 0,
        "shard": f"teacher_logits/{prompt_id}.safetensors",
    }
    if route_coverage is not None:
        row["route_coverage"] = route_coverage
    if route_trace is not None:
        row["route_trace"] = route_trace
    return row


def test_merge_route_coverage_cache_preserves_clean_base_and_tracks_source(tmp_path) -> None:
    cli = _load_merge_cli()
    clean = tmp_path / "clean"
    route_dirty = tmp_path / "route-dirty"
    route_clean = tmp_path / "route-clean"
    output = tmp_path / "merged"
    coverage = {
        "31": {"route_count": 1},
        "36": {"route_count": 1},
        "41": {"route_count": 1, "glu_code_route_count": 1},
    }
    _write_cache(clean, [_row("a"), _row("b")])
    _write_cache(route_dirty, [_row("a", pageouts_delta=4, route_coverage=coverage), _row("b", pageouts_delta=3, route_coverage=coverage)])
    _write_cache(route_clean, [_row("a", route_coverage=coverage)])

    summary = cli.merge_route_coverage_cache(
        clean_base_dir=clean,
        route_source_dirs=[route_dirty, route_clean],
        output_dir=output,
    )
    rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["all_memory_clean"] is True
    assert summary["route_coverage_row_count"] == 2
    assert summary["dirty_route_source_count"] == 1
    assert rows[0]["route_coverage_source"]["cache_root"] == str(route_clean)
    assert rows[0]["route_coverage_source"]["memory_clean"] is True
    assert rows[1]["route_coverage_source"]["cache_root"] == str(route_dirty)
    assert rows[1]["route_coverage_source"]["memory_clean"] is False
    assert (output / "teacher_logits" / "a.safetensors").read_bytes() == b"tensor"


def test_merge_route_coverage_cache_copies_route_trace_when_present(tmp_path) -> None:
    cli = _load_merge_cli()
    clean = tmp_path / "clean"
    route = tmp_path / "route"
    output = tmp_path / "merged"
    coverage = {
        "31": {"route_count": 1},
        "36": {"route_count": 1},
        "41": {"route_count": 1, "glu_code_route_count": 1},
    }
    trace = {"36": {"top_k": 2, "token_expert_indices": [[7, 3]]}}
    _write_cache(clean, [_row("a")])
    _write_cache(route, [_row("a", route_coverage=coverage, route_trace=trace)])

    summary = cli.merge_route_coverage_cache(
        clean_base_dir=clean,
        route_source_dirs=[route],
        output_dir=output,
    )
    rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["route_trace_row_count"] == 1
    assert rows[0]["route_trace"] == trace
    assert rows[0]["route_trace_source"]["cache_root"] == str(route)
