from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_vq.benchmark.glm45_air import append_jsonl, load_resident_air
from mlx_vq.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.models.glm45_air_vq_adapter import (
    has_dense_glm45_air_routed_expert_parameters,
    has_unbound_glm45_air_vq_experts,
)
from mlx_vq.quality.teacher_cache import (
    evaluate_teacher_cache_row,
    read_teacher_cache_rows,
    summarize_teacher_cache_records,
    validate_teacher_cache_metadata,
)

DEFAULT_VQ_ARTIFACT_DIR = "artifacts/glm-4.5-air-vq"


def _load_route_trace_exporter_symbols() -> tuple[tuple[int, ...], type]:
    exporter_path = Path(__file__).with_name("export_glm45_air_distributed_teacher_cache.py")
    spec = importlib.util.spec_from_file_location("glm45_air_route_trace_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {exporter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.DEFAULT_ROUTE_COVERAGE_LAYERS), module._RouteCoverageTracer


DEFAULT_ROUTE_COVERAGE_LAYERS, _RouteCoverageTracer = _load_route_trace_exporter_symbols()


def _select_teacher_rows(
    rows: list[dict],
    *,
    max_rows: int | None,
    row_index: int | None,
    row_indices: tuple[int, ...] | None,
) -> list[tuple[int, dict]]:
    if row_indices is not None:
        selected = []
        for index in row_indices:
            if index >= len(rows):
                raise IndexError(f"--row-indices includes {index}, out of range for {len(rows)} rows")
            selected.append((index, rows[index]))
        return selected
    if row_index is not None:
        if row_index >= len(rows):
            raise IndexError(f"--row-index {row_index} is out of range for {len(rows)} rows")
        return [(row_index, rows[row_index])]
    selected_rows = rows[:max_rows] if max_rows is not None else rows
    return list(enumerate(selected_rows))


def _parse_row_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("--row-indices must be a comma-separated list of integers") from error
    if not indices:
        raise ValueError("--row-indices must include at least one row")
    if any(index < 0 for index in indices):
        raise ValueError("--row-indices values must be zero or greater")
    return indices


def _parse_token_ids(value: str) -> tuple[int, ...]:
    try:
        token_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("--watch-token-ids must be a comma-separated list of integers") from error
    if not token_ids:
        raise ValueError("--watch-token-ids must include at least one token")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("--watch-token-ids values must be zero or greater")
    return token_ids


def _parse_logit_biases(values: list[str] | None) -> tuple[tuple[int, float], ...]:
    if not values:
        return ()
    biases: list[tuple[int, float]] = []
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if not stripped:
                continue
            if ":" not in stripped:
                raise ValueError("--logit-bias values must use TOKEN:BIAS")
            token_text, bias_text = stripped.split(":", 1)
            try:
                token_id = int(token_text.strip())
                bias = float(bias_text.strip())
            except ValueError as error:
                raise ValueError("--logit-bias values must use TOKEN:BIAS") from error
            if token_id < 0:
                raise ValueError("--logit-bias token ids must be zero or greater")
            biases.append((token_id, bias))
    if not biases:
        raise ValueError("--logit-bias must include at least one TOKEN:BIAS value")
    return tuple(biases)


def _apply_logit_biases(logits: mx.array, biases: tuple[tuple[int, float], ...]) -> mx.array:
    if not biases:
        return logits
    if logits.ndim != 2:
        raise ValueError(f"expected selected logits shape [positions, vocab], got {logits.shape}")
    vocab_size = int(logits.shape[1])
    bias_values = np.zeros((vocab_size,), dtype=np.float32)
    for token_id, bias in biases:
        if token_id >= vocab_size:
            raise ValueError(f"logit bias token id {token_id} is outside logits vocab size {vocab_size}")
        bias_values[token_id] += float(bias)
    return logits + mx.array(bias_values, dtype=logits.dtype)


def _clear_mlx_caches() -> None:
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
        return
    metal = getattr(mx, "metal", None)
    metal_clear_cache = getattr(metal, "clear_cache", None)
    if callable(metal_clear_cache):
        metal_clear_cache()


def _gb_to_bytes(value: float | None, *, name: str, allow_zero: bool = False) -> int | None:
    if value is None:
        return None
    numeric_value = float(value)
    if numeric_value < 0 or (numeric_value == 0 and not allow_zero):
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {requirement}")
    return int(numeric_value * 1024**3)


def _apply_mlx_memory_policy(args: argparse.Namespace) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "cache_limit_bytes": None,
        "memory_limit_bytes": None,
        "wired_limit_bytes": None,
        "previous_cache_limit_bytes": None,
        "previous_memory_limit_bytes": None,
        "previous_wired_limit_bytes": None,
        "clear_cache_before_load": bool(getattr(args, "mlx_clear_cache_before_load", False)),
        "cache_bytes_before_clear": None,
        "cache_bytes_after_clear": None,
    }
    cache_limit = _gb_to_bytes(
        getattr(args, "mlx_cache_limit_gb", None),
        name="--mlx-cache-limit-gb",
        allow_zero=True,
    )
    memory_limit = _gb_to_bytes(getattr(args, "mlx_memory_limit_gb", None), name="--mlx-memory-limit-gb")
    wired_limit = _gb_to_bytes(getattr(args, "mlx_wired_limit_gb", None), name="--mlx-wired-limit-gb")
    if cache_limit is not None:
        policy["cache_limit_bytes"] = cache_limit
        policy["previous_cache_limit_bytes"] = int(mx.set_cache_limit(cache_limit))
    if memory_limit is not None:
        policy["memory_limit_bytes"] = memory_limit
        policy["previous_memory_limit_bytes"] = int(mx.set_memory_limit(memory_limit))
    if wired_limit is not None:
        policy["wired_limit_bytes"] = wired_limit
        policy["previous_wired_limit_bytes"] = int(mx.set_wired_limit(wired_limit))
    if policy["clear_cache_before_load"]:
        policy["cache_bytes_before_clear"] = int(mx.get_cache_memory())
        _clear_mlx_caches()
        policy["cache_bytes_after_clear"] = int(mx.get_cache_memory())
    policy["enabled"] = any(
        value is not None
        for value in (
            policy["cache_limit_bytes"],
            policy["memory_limit_bytes"],
            policy["wired_limit_bytes"],
        )
    ) or policy["clear_cache_before_load"]
    return policy


def _append_memory_trace(
    trace_jsonl: str | None,
    *,
    stage: str,
    previous_vm_stat_counts: dict[str, int] | None,
    row_index: int | None = None,
    rows_completed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if trace_jsonl is None:
        return
    record: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "row_index": row_index,
        "rows_completed": rows_completed,
        "trace_time_seconds": time.time(),
        **collect_metric_snapshot(previous_vm_stat_counts=previous_vm_stat_counts),
    }
    if extra:
        record.update(extra)
    append_jsonl(trace_jsonl, record)


def _summarize_process_memory(previous_vm_stat_counts: dict[str, int] | None) -> dict[str, Any]:
    metrics = collect_metric_snapshot(previous_vm_stat_counts=previous_vm_stat_counts)
    pageouts_delta = int(metrics.get("pageouts_delta") or 0)
    swapouts_delta = int(metrics.get("swapouts_delta") or 0)
    return {
        "process_memory_clean": pageouts_delta == 0 and swapouts_delta == 0,
        "process_pageouts_delta": pageouts_delta,
        "process_swapouts_delta": swapouts_delta,
        "process_pageouts_total": metrics.get("pageouts_total"),
        "process_swapouts_total": metrics.get("swapouts_total"),
        "process_mlx_active_bytes": metrics.get("mlx_active_bytes"),
        "process_mlx_peak_bytes": metrics.get("mlx_peak_bytes"),
        "process_mlx_cache_bytes": metrics.get("mlx_cache_bytes"),
        "process_rss_bytes": metrics.get("rss_bytes"),
    }


def _parse_route_trace_layers(values: list[str] | None) -> tuple[int, ...]:
    if not values:
        return DEFAULT_ROUTE_COVERAGE_LAYERS
    layers: list[int] = []
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if not stripped:
                continue
            try:
                layer = int(stripped)
            except ValueError as error:
                raise ValueError("--route-trace-layer must contain integer layer indices") from error
            if layer < 0:
                raise ValueError("--route-trace-layer values must be zero or greater")
            layers.append(layer)
    if not layers:
        raise ValueError("--route-trace-layer must include at least one layer")
    return tuple(dict.fromkeys(layers))


def _select_positions(logits: mx.array, positions: list[int]) -> mx.array:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"expected logits shape [1, tokens, vocab], got {logits.shape}")
    if not positions:
        raise ValueError("teacher cache row has no positions")
    max_position = max(positions)
    if max_position >= logits.shape[1]:
        raise ValueError(
            f"teacher cache position {max_position} is outside VQ logits length {logits.shape[1]}"
        )
    return logits[0, mx.array(positions, dtype=mx.int32), :]


def _truncate_input_to_selected_position_prefix(
    input_token_ids: list[int],
    positions: list[int],
) -> list[int]:
    if not positions:
        return input_token_ids
    required_len = max(positions) + 2
    if required_len >= len(input_token_ids):
        return input_token_ids
    return input_token_ids[:required_len]


def _safe_logit_shard_stem(prompt_id: object, row_index: int) -> str:
    text = str(prompt_id) if prompt_id is not None else f"row_{row_index:05d}"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    if not text:
        text = f"row_{row_index:05d}"
    return f"row-{row_index:05d}-{text}"


def _save_vq_logits_shard(
    *,
    output_dir: str | Path,
    row_index: int,
    row: dict[str, Any],
    selected_logits: mx.array,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    shard_name = f"{_safe_logit_shard_stem(row.get('prompt_id'), row_index)}.safetensors"
    shard_path = output_root / shard_name
    tensor_name = "vq_logits"
    mx.save_safetensors(str(shard_path), {tensor_name: selected_logits.astype(mx.float16)})
    return {
        "vq_logit_shard": shard_name,
        "vq_logit_tensor": tensor_name,
        "vq_logit_shape": [int(dim) for dim in selected_logits.shape],
        "vq_logit_dtype": "float16",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GLM-4.5-Air VQ logits against an off-box teacher cache."
    )
    parser.add_argument("--teacher-jsonl", required=True, help="Teacher cache metadata JSONL.")
    parser.add_argument(
        "--cache-root",
        help="Root for tensor shards referenced by the teacher JSONL. Defaults to JSONL parent.",
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--artifact-dir", default=DEFAULT_VQ_ARTIFACT_DIR)
    parser.add_argument(
        "--engine",
        choices=["vq_e1_routed", "vq_e1_routed_vq_metal", "vq_e1_routed_nax_e8", "vq_e1_routed_nax_e8p"],
        default="vq_e1_routed",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--row-index",
        type=int,
        help="Evaluate exactly one zero-based row from the teacher JSONL.",
    )
    parser.add_argument(
        "--row-indices",
        help="Evaluate comma-separated zero-based rows from the teacher JSONL.",
    )
    parser.add_argument(
        "--min-top-k",
        type=int,
        default=128,
        help="Minimum accepted top-k width for rows without a smaller full-logit vocab.",
    )
    parser.add_argument(
        "--check-values",
        action="store_true",
        help="Load referenced tensors during prevalidation and check finite/id-range values.",
    )
    parser.add_argument(
        "--watch-token-ids",
        help="Comma-separated token ids whose VQ logprobs and margins should be recorded per position.",
    )
    parser.add_argument(
        "--logit-bias",
        action="append",
        help="Eval-only VQ logit bias as TOKEN:BIAS. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--include-route-trace",
        action="store_true",
        help="Append token-level student routed expert traces to each eval record.",
    )
    parser.add_argument(
        "--route-trace-layer",
        action="append",
        help="Layer index or comma-separated layer indices to trace. Defaults to route-coverage layers.",
    )
    parser.add_argument(
        "--append-jsonl",
        default="artifacts/quality/glm45-air-teacher-cache-local.jsonl",
    )
    parser.add_argument(
        "--save-vq-logits-dir",
        help=(
            "Optional directory for selected student/VQ logits. Records get "
            "artifact-relative vq_logit_* fields for exact offline replay."
        ),
    )
    parser.add_argument(
        "--clear-mlx-cache-between-rows",
        action="store_true",
        help="Also clear MLX caches after the default per-row Python reference release and gc.collect().",
    )
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=float,
        help="Set MLX cache limit before model load. Use 0 to disable persistent cache residency.",
    )
    parser.add_argument(
        "--mlx-memory-limit-gb",
        type=float,
        help="Set MLX memory limit before model load.",
    )
    parser.add_argument(
        "--mlx-wired-limit-gb",
        type=float,
        help="Set MLX wired limit before model load.",
    )
    parser.add_argument(
        "--mlx-clear-cache-before-load",
        action="store_true",
        help="Clear MLX caches immediately before loading the resident model.",
    )
    parser.add_argument(
        "--memory-trace-jsonl",
        help="Append stage-level memory snapshots for model load, row forward, row record, and cleanup.",
    )
    parser.add_argument(
        "--truncate-input-to-selected-positions",
        action="store_true",
        help="Evaluate only the prefix needed for selected positions, preserving next-token targets.",
    )
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive when provided")
    if args.row_index is not None and args.row_index < 0:
        parser.error("--row-index must be zero or greater")
    if args.row_index is not None and args.max_rows is not None:
        parser.error("--row-index cannot be combined with --max-rows")
    if args.row_indices is not None and args.max_rows is not None:
        parser.error("--row-indices cannot be combined with --max-rows")
    if args.row_indices is not None and args.row_index is not None:
        parser.error("--row-indices cannot be combined with --row-index")
    if args.min_top_k <= 0:
        parser.error("--min-top-k must be positive")
    try:
        row_indices = _parse_row_indices(args.row_indices) if args.row_indices is not None else None
    except ValueError as error:
        parser.error(str(error))
    try:
        watch_token_ids = _parse_token_ids(args.watch_token_ids) if args.watch_token_ids is not None else None
    except ValueError as error:
        parser.error(str(error))
    try:
        logit_biases = _parse_logit_biases(args.logit_bias)
    except ValueError as error:
        parser.error(str(error))
    try:
        route_trace_layers = _parse_route_trace_layers(args.route_trace_layer)
    except ValueError as error:
        parser.error(str(error))

    teacher_jsonl = Path(args.teacher_jsonl)
    cache_root = Path(args.cache_root) if args.cache_root is not None else teacher_jsonl.parent
    rows = read_teacher_cache_rows(teacher_jsonl)
    try:
        selected_rows = _select_teacher_rows(
            rows,
            max_rows=args.max_rows,
            row_index=args.row_index,
            row_indices=row_indices,
        )
    except IndexError as error:
        parser.error(str(error))
    validation_row_indices = tuple(index for index, _ in selected_rows)
    validate_all_rows = args.max_rows is None and args.row_index is None and row_indices is None
    validation = validate_teacher_cache_metadata(
        teacher_jsonl,
        cache_root=cache_root,
        min_top_k=args.min_top_k,
        check_values=args.check_values,
        row_indices=None if validate_all_rows else validation_row_indices,
    )
    if not validation["ok"]:
        print(json.dumps(validation, indent=2, sort_keys=True))
        raise SystemExit(1)

    prefill_engine = {
        "vq_e1_routed": "auto",
        "vq_e1_routed_vq_metal": "vq_metal",
        "vq_e1_routed_nax_e8": "nax_e8",
        "vq_e1_routed_nax_e8p": "nax_e8p",
    }[args.engine]
    resident_engine = {
        "vq_e1_routed": "vq_resident",
        "vq_e1_routed_vq_metal": "vq_resident_vq_metal",
        "vq_e1_routed_nax_e8": "vq_resident_nax_e8",
        "vq_e1_routed_nax_e8p": "vq_resident_nax_e8p",
    }[args.engine]
    try:
        mlx_memory_policy = _apply_mlx_memory_policy(args)
    except ValueError as error:
        parser.error(str(error))
    enabled_mlx_memory_policy = mlx_memory_policy if mlx_memory_policy["enabled"] else None
    trace_vm_start = collect_vm_stat_counts()
    _append_memory_trace(
        args.memory_trace_jsonl,
        stage="before_model_load",
        previous_vm_stat_counts=trace_vm_start,
        rows_completed=0,
        extra={
            "artifact_dir": args.artifact_dir,
            "engine": args.engine,
            "mlx_memory_policy": enabled_mlx_memory_policy,
        },
    )
    model, _, _, _ = load_resident_air(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
        config_path=args.config_path,
        index_path=args.index_path,
        artifact_dir=args.artifact_dir,
        prefill_engine=prefill_engine,
    )
    _append_memory_trace(
        args.memory_trace_jsonl,
        stage="after_model_load",
        previous_vm_stat_counts=trace_vm_start,
        rows_completed=0,
        extra={
            "artifact_dir": args.artifact_dir,
            "engine": args.engine,
            "mlx_memory_policy": enabled_mlx_memory_policy,
        },
    )
    route_tracer = None
    if args.include_route_trace:
        route_tracer = _RouteCoverageTracer(layers=route_trace_layers, include_trace=True)
        route_tracer.install(model.model)

    records = []
    for row_index, row in selected_rows:
        input_token_ids = row.get("input_token_ids")
        if not isinstance(input_token_ids, list) or not input_token_ids:
            raise ValueError(f"teacher row {row_index} must contain non-empty input_token_ids")
        positions = [int(position) for position in row.get("positions", [])]
        if not positions:
            target_ids = row.get("target_token_ids")
            if not isinstance(target_ids, list) or not target_ids:
                raise ValueError(f"teacher row {row_index} needs positions or target_token_ids")
            positions = list(range(len(target_ids)))
        eval_input_token_ids = (
            _truncate_input_to_selected_position_prefix(input_token_ids, positions)
            if args.truncate_input_to_selected_positions
            else input_token_ids
        )
        original_input_token_count = len(input_token_ids)
        eval_input_token_count = len(eval_input_token_ids)

        before_vm = collect_vm_stat_counts()
        reset_mlx_peak_memory()
        if route_tracer is not None:
            route_tracer.reset()
        cache = model.make_cache()
        _append_memory_trace(
            args.memory_trace_jsonl,
            stage="before_row_forward",
            previous_vm_stat_counts=before_vm,
            row_index=row_index,
            rows_completed=len(records),
            extra={
                "prompt_id": row.get("prompt_id"),
                "original_input_token_count": original_input_token_count,
                "eval_input_token_count": eval_input_token_count,
            },
        )
        start = time.perf_counter()
        logits = model(mx.array([eval_input_token_ids], dtype=mx.int32), cache=cache)
        selected_logits = _select_positions(logits, positions)
        selected_logits = _apply_logit_biases(selected_logits, logit_biases)
        mx.eval(selected_logits)
        elapsed_seconds = time.perf_counter() - start
        _append_memory_trace(
            args.memory_trace_jsonl,
            stage="after_row_forward",
            previous_vm_stat_counts=before_vm,
            row_index=row_index,
            rows_completed=len(records),
            extra={
                "prompt_id": row.get("prompt_id"),
                "elapsed_seconds": elapsed_seconds,
                "original_input_token_count": original_input_token_count,
                "eval_input_token_count": eval_input_token_count,
            },
        )
        metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
        saved_vq_logits = (
            _save_vq_logits_shard(
                output_dir=args.save_vq_logits_dir,
                row_index=row_index,
                row=row,
                selected_logits=selected_logits,
            )
            if args.save_vq_logits_dir is not None
            else {}
        )
        record = evaluate_teacher_cache_row(
            row,
            vq_logits=selected_logits,
            cache_root=cache_root,
            watch_token_ids=watch_token_ids,
            extra={
                "engine": resident_engine,
                "prefill_engine": prefill_engine,
                "artifact_dir": args.artifact_dir,
                "eval_logit_biases": [
                    {"token_id": int(token_id), "bias": float(bias)}
                    for token_id, bias in logit_biases
                ],
                "row_index": row_index,
                "elapsed_seconds": elapsed_seconds,
                "original_input_token_count": original_input_token_count,
                "eval_input_token_count": eval_input_token_count,
                "input_truncated_to_selected_positions": bool(
                    args.truncate_input_to_selected_positions
                    and eval_input_token_count < original_input_token_count
                ),
                "dense_expert_params": has_dense_glm45_air_routed_expert_parameters(model),
                "unbound_vq_experts": has_unbound_glm45_air_vq_experts(model),
                "mlx_memory_policy": enabled_mlx_memory_policy,
                "saved_vq_logits_dir": args.save_vq_logits_dir,
                **saved_vq_logits,
                **metrics,
            },
        )
        if route_tracer is not None:
            route_coverage = route_tracer.summary(row_intent=row.get("row_intent"))
            route_trace = {
                layer: coverage["route_trace"]
                for layer, coverage in route_coverage.items()
                if isinstance(coverage, dict) and "route_trace" in coverage
            }
            record["route_trace"] = route_trace
            record["route_trace_layers"] = sorted(route_trace, key=int)
        records.append(record)
        append_jsonl(args.append_jsonl, record)
        _append_memory_trace(
            args.memory_trace_jsonl,
            stage="after_row_record",
            previous_vm_stat_counts=before_vm,
            row_index=row_index,
            rows_completed=len(records),
            extra={
                "prompt_id": row.get("prompt_id"),
                "original_input_token_count": original_input_token_count,
                "eval_input_token_count": eval_input_token_count,
            },
        )
        del cache, logits, selected_logits
        gc.collect()
        if args.clear_mlx_cache_between_rows:
            _clear_mlx_caches()
        _append_memory_trace(
            args.memory_trace_jsonl,
            stage="after_row_cleanup",
            previous_vm_stat_counts=before_vm,
            row_index=row_index,
            rows_completed=len(records),
            extra={
                "prompt_id": row.get("prompt_id"),
                "original_input_token_count": original_input_token_count,
                "eval_input_token_count": eval_input_token_count,
                "mlx_cache_cleared": bool(args.clear_mlx_cache_between_rows),
            },
        )

    summary = summarize_teacher_cache_records(records)
    row_memory_clean = bool(summary.get("all_memory_clean"))
    process_memory_summary = _summarize_process_memory(previous_vm_stat_counts=trace_vm_start)
    summary.update(
        {
            "teacher_jsonl": str(teacher_jsonl),
            "cache_root": str(cache_root),
            "append_jsonl": args.append_jsonl,
            "engine": args.engine,
            "prefill_engine": prefill_engine,
            "artifact_dir": args.artifact_dir,
            "mlx_memory_policy": enabled_mlx_memory_policy,
            "row_memory_clean": row_memory_clean,
            **process_memory_summary,
            "acceptance_memory_clean": row_memory_clean and bool(process_memory_summary["process_memory_clean"]),
            "route_trace_row_count": sum(
                1 for record in records if isinstance(record.get("route_trace"), dict) and record.get("route_trace")
            ),
        }
    )
    _append_memory_trace(
        args.memory_trace_jsonl,
        stage="after_summary",
        previous_vm_stat_counts=trace_vm_start,
        rows_completed=len(records),
        extra={
            "record_count": len(records),
            "clean_record_count": summary.get("clean_record_count"),
            "row_memory_clean": row_memory_clean,
            "process_memory_clean": process_memory_summary["process_memory_clean"],
            "acceptance_memory_clean": summary["acceptance_memory_clean"],
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
