from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_memory_clean(row: dict[str, Any]) -> bool:
    return int(row.get("pageouts_delta", 0) or 0) == 0 and int(row.get("swapouts_delta", 0) or 0) == 0


def _memory_event_count(row: dict[str, Any]) -> int:
    return int(row.get("pageouts_delta", 0) or 0) + int(row.get("swapouts_delta", 0) or 0)


def _row_shard(row: dict[str, Any]) -> str:
    shard = row.get("logit_shard") or row.get("shard")
    if not isinstance(shard, str) or not shard:
        raise ValueError(f"row for {row.get('prompt_id')} does not contain a shard path")
    return shard


def _index_rows(cache_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_rows(cache_dir / "metadata.jsonl")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"row in {cache_dir} has invalid prompt_id: {prompt_id!r}")
        if prompt_id in indexed:
            raise ValueError(f"duplicate prompt_id {prompt_id!r} in {cache_dir}")
        indexed[prompt_id] = row
    return indexed


def _has_route_coverage(row: dict[str, Any]) -> bool:
    coverage = row.get("route_coverage")
    if not isinstance(coverage, dict) or not coverage:
        return False
    return all(isinstance(coverage.get(str(layer)), dict) and coverage.get(str(layer)) for layer in (31, 36, 41))


def _choose_route_row(
    prompt_id: str,
    route_rows: dict[str, list[tuple[int, Path, dict[str, Any]]]],
) -> tuple[int, Path, dict[str, Any]] | None:
    candidates = [
        (order, source_dir, row)
        for order, source_dir, row in route_rows.get(prompt_id, [])
        if _has_route_coverage(row)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (_memory_event_count(item[2]), item[0]))


def merge_route_coverage_cache(
    *,
    clean_base_dir: Path,
    route_source_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    base_rows = _read_rows(clean_base_dir / "metadata.jsonl")
    route_rows: dict[str, list[tuple[int, Path, dict[str, Any]]]] = {}
    for order, route_dir in enumerate(route_source_dirs):
        for prompt_id, row in _index_rows(route_dir).items():
            route_rows.setdefault(prompt_id, []).append((order, route_dir, row))

    missing_route_coverage: list[str] = []
    dirty_route_sources: list[dict[str, Any]] = []
    selected: list[tuple[dict[str, Any], Path]] = []
    for base_row in base_rows:
        prompt_id = base_row.get("prompt_id")
        if not isinstance(prompt_id, str):
            raise ValueError(f"base row has invalid prompt_id: {prompt_id!r}")
        if not _is_memory_clean(base_row):
            raise ValueError(f"clean base row {prompt_id!r} is not memory clean")
        route_choice = _choose_route_row(prompt_id, route_rows)
        if route_choice is None:
            missing_route_coverage.append(prompt_id)
            continue
        _order, route_dir, route_row = route_choice
        merged = dict(base_row)
        merged["route_coverage"] = route_row["route_coverage"]
        merged["route_coverage_source"] = {
            "cache_root": str(route_dir),
            "prompt_id": prompt_id,
            "memory_clean": _is_memory_clean(route_row),
            "pageouts_delta": int(route_row.get("pageouts_delta", 0) or 0),
            "swapouts_delta": int(route_row.get("swapouts_delta", 0) or 0),
        }
        route_trace = route_row.get("route_trace")
        if isinstance(route_trace, dict) and route_trace:
            merged["route_trace"] = route_trace
            merged["route_trace_source"] = dict(merged["route_coverage_source"])
        if not _is_memory_clean(route_row):
            dirty_route_sources.append(merged["route_coverage_source"])
        selected.append((merged, clean_base_dir))

    if missing_route_coverage:
        raise SystemExit(
            "missing route coverage for prompt ids: " + ", ".join(sorted(missing_route_coverage))
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "teacher_logits").mkdir(parents=True, exist_ok=True)

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row, source_dir in selected:
            shard = _row_shard(row)
            source = source_dir / shard
            destination = output_dir / shard
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "clean_base_dir": str(clean_base_dir),
        "output_dir": str(output_dir),
        "route_source_dirs": [str(path) for path in route_source_dirs],
        "row_count": len(selected),
        "all_memory_clean": all(_is_memory_clean(row) for row, _source_dir in selected),
        "route_coverage_row_count": sum(1 for row, _source_dir in selected if _has_route_coverage(row)),
        "route_trace_row_count": sum(
            1
            for row, _source_dir in selected
            if isinstance(row.get("route_trace"), dict) and row.get("route_trace")
        ),
        "dirty_route_source_count": len(dirty_route_sources),
        "dirty_route_sources": dirty_route_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach traced route coverage to a clean GLM-4.5-Air teacher cache."
    )
    parser.add_argument("--clean-base-dir", required=True)
    parser.add_argument("--route-source-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = merge_route_coverage_cache(
        clean_base_dir=Path(args.clean_base_dir),
        route_source_dirs=[Path(path) for path in args.route_source_dir],
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_memory_clean"]:
        raise SystemExit("merged cache is not memory clean")
    if summary["route_coverage_row_count"] != summary["row_count"]:
        raise SystemExit("merged cache is missing route coverage")


if __name__ == "__main__":
    main()
