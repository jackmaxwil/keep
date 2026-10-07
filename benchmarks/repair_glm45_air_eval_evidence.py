from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mlx_vq.quality.teacher_cache import read_teacher_cache_rows, summarize_teacher_cache_records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_row_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("--row-indices must be a comma-separated list of integers") from error
    if not indices:
        raise ValueError("--row-indices must include at least one row")
    if any(index < 0 for index in indices):
        raise ValueError("--row-indices values must be zero or greater")
    return indices


def _expected_row_indices(
    teacher_jsonl: Path,
    *,
    max_rows: int | None,
    row_indices: tuple[int, ...] | None,
) -> list[int]:
    rows = read_teacher_cache_rows(teacher_jsonl)
    if row_indices is not None:
        for index in row_indices:
            if index >= len(rows):
                raise IndexError(f"row index {index} is out of range for {len(rows)} rows")
        return list(row_indices)
    selected_count = min(max_rows, len(rows)) if max_rows is not None else len(rows)
    return list(range(selected_count))


def _record_row_index(record: dict[str, Any]) -> int | None:
    value = record.get("row_index")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def _memory_clean(record: dict[str, Any] | None) -> bool:
    return bool(
        record is not None
        and record.get("pageouts_delta") == 0
        and record.get("swapouts_delta") == 0
    )


def _best_records_by_index(
    records: list[dict[str, Any]],
    *,
    expected: set[int],
) -> dict[int, dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for record in records:
        row_index = _record_row_index(record)
        if row_index is None or row_index not in expected:
            continue
        previous = best.get(row_index)
        if previous is None or (not _memory_clean(previous)) or _memory_clean(record):
            best[row_index] = record
    return best


def _dirty_or_missing(expected: list[int], records_by_index: dict[int, dict[str, Any]]) -> list[int]:
    return [index for index in expected if not _memory_clean(records_by_index.get(index))]


def _append_optional(argv: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _append_switch(argv: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def _extend_repair_eval_argv(
    argv: list[str],
    *,
    teacher_jsonl: Path,
    cache_root: Path | None,
    artifact_dir: Path,
    engine: str,
    row_indices: list[int],
    append_jsonl: Path,
    eval_script: Path,
    model_id: str | None = None,
    revision: str | None = None,
    source_dir: Path | None = None,
    config_path: Path | None = None,
    index_path: Path | None = None,
    min_top_k: int | None = None,
    check_values: bool = False,
    watch_token_ids: str | None = None,
    logit_biases: list[str] | None = None,
    include_route_trace: bool = False,
    route_trace_layers: list[str] | None = None,
    clear_mlx_cache_between_rows: bool = False,
    mlx_cache_limit_gb: float | None = None,
    mlx_memory_limit_gb: float | None = None,
    mlx_wired_limit_gb: float | None = None,
    mlx_clear_cache_before_load: bool = False,
    truncate_input_to_selected_positions: bool = False,
) -> None:
    argv.extend(
        [
            str(eval_script),
            "--teacher-jsonl",
            str(teacher_jsonl),
            "--artifact-dir",
            str(artifact_dir),
            "--engine",
            engine,
            "--row-indices",
            ",".join(str(index) for index in row_indices),
            "--append-jsonl",
            str(append_jsonl),
        ]
    )
    _append_optional(argv, "--cache-root", cache_root)
    _append_optional(argv, "--model-id", model_id)
    _append_optional(argv, "--revision", revision)
    _append_optional(argv, "--source-dir", source_dir)
    _append_optional(argv, "--config-path", config_path)
    _append_optional(argv, "--index-path", index_path)
    _append_optional(argv, "--min-top-k", min_top_k)
    _append_optional(argv, "--watch-token-ids", watch_token_ids)
    for value in logit_biases or []:
        argv.extend(["--logit-bias", str(value)])
    for value in route_trace_layers or []:
        argv.extend(["--route-trace-layer", str(value)])
    _append_switch(argv, "--check-values", check_values)
    _append_switch(argv, "--include-route-trace", include_route_trace)
    _append_switch(argv, "--clear-mlx-cache-between-rows", clear_mlx_cache_between_rows)
    _append_optional(argv, "--mlx-cache-limit-gb", mlx_cache_limit_gb)
    _append_optional(argv, "--mlx-memory-limit-gb", mlx_memory_limit_gb)
    _append_optional(argv, "--mlx-wired-limit-gb", mlx_wired_limit_gb)
    _append_switch(argv, "--mlx-clear-cache-before-load", mlx_clear_cache_before_load)
    _append_switch(
        argv,
        "--truncate-input-to-selected-positions",
        truncate_input_to_selected_positions,
    )


def repair_eval_evidence(
    *,
    existing_jsonl: Path,
    teacher_jsonl: Path,
    append_jsonl: Path,
    artifact_dir: Path,
    engine: str,
    cache_root: Path | None = None,
    max_rows: int | None = None,
    row_indices: str | None = None,
    repair_attempts: int = 1,
    repair_work_dir: Path | None = None,
    eval_script: Path = Path("benchmarks/eval_glm45_air_teacher_cache.py"),
    model_id: str | None = None,
    revision: str | None = None,
    source_dir: Path | None = None,
    config_path: Path | None = None,
    index_path: Path | None = None,
    min_top_k: int | None = None,
    check_values: bool = False,
    watch_token_ids: str | None = None,
    logit_biases: list[str] | None = None,
    include_route_trace: bool = False,
    route_trace_layers: list[str] | None = None,
    clear_mlx_cache_between_rows: bool = False,
    mlx_cache_limit_gb: float | None = None,
    mlx_memory_limit_gb: float | None = None,
    mlx_wired_limit_gb: float | None = None,
    mlx_clear_cache_before_load: bool = False,
    truncate_input_to_selected_positions: bool = False,
    spawn=subprocess.run,
) -> dict[str, Any]:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")
    if repair_attempts < 0:
        raise ValueError("repair_attempts must be non-negative")

    parsed_row_indices = _parse_row_indices(row_indices)
    expected = _expected_row_indices(
        teacher_jsonl,
        max_rows=max_rows,
        row_indices=parsed_row_indices,
    )
    expected_set = set(expected)
    records_by_index = _best_records_by_index(
        _read_jsonl(existing_jsonl),
        expected=expected_set,
    )
    all_repair_rows: list[int] = []
    work_dir = repair_work_dir or append_jsonl.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, repair_attempts + 1):
        repair_rows = _dirty_or_missing(expected, records_by_index)
        if not repair_rows:
            break
        all_repair_rows.extend(index for index in repair_rows if index not in all_repair_rows)
        repair_jsonl = work_dir / f"repair-attempt-{attempt}.jsonl"
        if repair_jsonl.exists():
            repair_jsonl.unlink()
        argv = [sys.executable]
        _extend_repair_eval_argv(
            argv,
            teacher_jsonl=teacher_jsonl,
            cache_root=cache_root,
            artifact_dir=artifact_dir,
            engine=engine,
            row_indices=repair_rows,
            append_jsonl=repair_jsonl,
            eval_script=eval_script,
            model_id=model_id,
            revision=revision,
            source_dir=source_dir,
            config_path=config_path,
            index_path=index_path,
            min_top_k=min_top_k,
            check_values=check_values,
            watch_token_ids=watch_token_ids,
            logit_biases=logit_biases,
            include_route_trace=include_route_trace,
            route_trace_layers=route_trace_layers,
            clear_mlx_cache_between_rows=clear_mlx_cache_between_rows,
            mlx_cache_limit_gb=mlx_cache_limit_gb,
            mlx_memory_limit_gb=mlx_memory_limit_gb,
            mlx_wired_limit_gb=mlx_wired_limit_gb,
            mlx_clear_cache_before_load=mlx_clear_cache_before_load,
            truncate_input_to_selected_positions=truncate_input_to_selected_positions,
        )
        result = spawn(argv)
        if result.returncode != 0:
            raise RuntimeError(f"repair eval attempt {attempt} failed with exit {result.returncode}")
        for row_index, record in _best_records_by_index(
            _read_jsonl(repair_jsonl),
            expected=expected_set,
        ).items():
            previous = records_by_index.get(row_index)
            if previous is None or (not _memory_clean(previous)) or _memory_clean(record):
                records_by_index[row_index] = record

    merged = [records_by_index[index] for index in expected if index in records_by_index]
    _write_jsonl(append_jsonl, merged)
    summary = summarize_teacher_cache_records(merged) if merged else {}
    remaining_dirty = _dirty_or_missing(expected, records_by_index)
    summary.update(
        {
            "schema_version": 1,
            "expected_row_count": len(expected),
            "merged_record_count": len(merged),
            "repair_row_indices": all_repair_rows,
            "remaining_dirty_or_missing_row_indices": remaining_dirty,
            "existing_jsonl": str(existing_jsonl),
            "append_jsonl": str(append_jsonl),
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repair GLM-4.5-Air eval evidence by rerunning only dirty or missing rows."
    )
    parser.add_argument("--existing-jsonl", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--append-jsonl", required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--row-indices")
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--repair-work-dir")
    parser.add_argument("--eval-script", default="benchmarks/eval_glm45_air_teacher_cache.py")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--min-top-k", type=int)
    parser.add_argument("--check-values", action="store_true")
    parser.add_argument("--watch-token-ids")
    parser.add_argument("--logit-bias", action="append")
    parser.add_argument("--include-route-trace", action="store_true")
    parser.add_argument("--route-trace-layer", action="append")
    parser.add_argument("--clear-mlx-cache-between-rows", action="store_true")
    parser.add_argument("--mlx-cache-limit-gb", type=float)
    parser.add_argument("--mlx-memory-limit-gb", type=float)
    parser.add_argument("--mlx-wired-limit-gb", type=float)
    parser.add_argument("--mlx-clear-cache-before-load", action="store_true")
    parser.add_argument("--truncate-input-to-selected-positions", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = repair_eval_evidence(
            existing_jsonl=Path(args.existing_jsonl),
            teacher_jsonl=Path(args.teacher_jsonl),
            cache_root=Path(args.cache_root) if args.cache_root is not None else None,
            append_jsonl=Path(args.append_jsonl),
            artifact_dir=Path(args.artifact_dir),
            engine=args.engine,
            max_rows=args.max_rows,
            row_indices=args.row_indices,
            repair_attempts=args.repair_attempts,
            repair_work_dir=Path(args.repair_work_dir) if args.repair_work_dir else None,
            eval_script=Path(args.eval_script),
            model_id=args.model_id,
            revision=args.revision,
            source_dir=Path(args.source_dir) if args.source_dir else None,
            config_path=Path(args.config_path) if args.config_path else None,
            index_path=Path(args.index_path) if args.index_path else None,
            min_top_k=args.min_top_k,
            check_values=args.check_values,
            watch_token_ids=args.watch_token_ids,
            logit_biases=args.logit_bias,
            include_route_trace=args.include_route_trace,
            route_trace_layers=args.route_trace_layer,
            clear_mlx_cache_between_rows=args.clear_mlx_cache_between_rows,
            mlx_cache_limit_gb=args.mlx_cache_limit_gb,
            mlx_memory_limit_gb=args.mlx_memory_limit_gb,
            mlx_wired_limit_gb=args.mlx_wired_limit_gb,
            mlx_clear_cache_before_load=args.mlx_clear_cache_before_load,
            truncate_input_to_selected_positions=args.truncate_input_to_selected_positions,
        )
    except (IndexError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
