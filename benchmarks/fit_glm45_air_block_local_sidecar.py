from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from ramp.benchmark.glm45_air import append_jsonl, load_resident_air
from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID
from keep.convert.stream_convert import load_safetensors_index
from keep.io.continuous_sidecar import (
    copy_declared_continuous_sidecars,
    link_seed_artifact_groups,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)
from keep.io.source_safetensors import read_indexed_safetensors_tensor_mlx
from ramp.models.glm45_air_vq_adapter import GLM45AirVQMoE
from keep.quality.mlx_surrogate import (
    fit_low_rank_residual_sidecar_least_squares,
    fit_output_bias_sidecar_least_squares,
    fit_scale_delta_output_bias_sidecar_least_squares,
    fit_scale_delta_sidecar_least_squares,
)
from keep.quality.teacher_cache import read_teacher_cache_rows, validate_teacher_cache_metadata
from keep.validate.glm45_air_vq import capture_glm45_air_moe_input_states


def _target_projection(model, *, layer: int, projection: str):
    layer_module = model.model.layers[layer]
    if not isinstance(layer_module.mlp, GLM45AirVQMoE):
        raise ValueError(f"layer {layer} is not a sparse GLM-4.5-Air MoE layer")
    if layer_module.mlp.switch_mlp is None:
        raise RuntimeError(f"layer {layer} VQ switch_mlp is not bound")
    if projection == "gate_proj":
        return layer_module.mlp.switch_mlp.gate_proj
    if projection == "up_proj":
        return layer_module.mlp.switch_mlp.up_proj
    if projection == "down_proj":
        return layer_module.mlp.switch_mlp.down_proj
    raise ValueError("projection must be gate_proj, up_proj, or down_proj")


def _source_weight_name(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


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


def _resolve_source_dir_arg(
    *,
    source_dir: str | None,
    model_id: str,
    revision: str,
) -> Path:
    if source_dir:
        return Path(source_dir)
    return Path(snapshot_download(repo_id=model_id, revision=revision))


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int | None,
    row_indices: tuple[int, ...] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    if row_indices is not None:
        selected = []
        for index in row_indices:
            if index >= len(rows):
                raise IndexError(f"--row-indices includes {index}, out of range for {len(rows)} rows")
            selected.append((index, rows[index]))
        return selected
    selected_rows = rows if max_rows is None else rows[:max_rows]
    return list(enumerate(selected_rows))


def _selected_positions(row: dict[str, Any], *, max_positions: int | None) -> tuple[int, ...]:
    positions = row.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError(f"teacher row {row.get('prompt_id')} must contain positions")
    selected = positions if max_positions is None else positions[:max_positions]
    if not selected:
        raise ValueError(f"teacher row {row.get('prompt_id')} selected no positions")
    return tuple(int(position) for position in selected)


def _silu_np(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return values * (1.0 / (1.0 + np.exp(-clipped)))


def _collect_block_local_batch(
    *,
    model,
    teacher_rows: list[tuple[int, dict[str, Any]]],
    layer: int,
    projection: str,
    max_positions: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    input_rows: list[np.ndarray] = []
    source_input_rows: list[np.ndarray] = []
    route_rows: list[np.ndarray] = []
    row_records: list[dict[str, Any]] = []
    layer_module = model.model.layers[layer]
    for row_index, row in teacher_rows:
        input_ids = row.get("input_token_ids")
        if not isinstance(input_ids, list) or not input_ids:
            raise ValueError(f"teacher row {row_index} must contain input_token_ids")
        positions = _selected_positions(row, max_positions=max_positions)
        states = capture_glm45_air_moe_input_states(model, [int(token) for token in input_ids], layer=layer)
        max_position = max(positions)
        if max_position >= states.shape[0]:
            raise ValueError(
                f"row {row_index} position {max_position} is outside captured states length {states.shape[0]}"
            )
        x = states[list(positions)].astype(np.float32, copy=False)
        indices_mx, _ = layer_module.mlp.gate(mx.array(x))
        mx.eval(indices_mx)
        indices = np.asarray(indices_mx, dtype=np.int32)
        source_input_rows.append(x)
        if projection == "down_proj":
            if layer_module.mlp.switch_mlp is None:
                raise RuntimeError(f"layer {layer} VQ switch_mlp is not bound")
            x_mx = mx.array(x)
            gate_mx = layer_module.mlp.switch_mlp.gate_proj(x_mx, indices_mx)
            up_mx = layer_module.mlp.switch_mlp.up_proj(x_mx, indices_mx)
            down_input_mx = nn.silu(gate_mx) * up_mx
            mx.eval(down_input_mx)
            input_rows.append(np.asarray(down_input_mx.astype(mx.float32), dtype=np.float32))
        else:
            input_rows.append(x)
        route_rows.append(indices)
        row_records.append(
            {
                "row_index": int(row_index),
                "prompt_id": str(row.get("prompt_id", row_index)),
                "position_count": len(positions),
                "positions": list(positions),
                "selected_experts": sorted(int(value) for value in np.unique(indices)),
            }
        )
    return (
        np.concatenate(input_rows, axis=0),
        np.concatenate(route_rows, axis=0),
        np.concatenate(source_input_rows, axis=0),
        row_records,
    )


def _source_projection_targets(
    *,
    source_dir: Path,
    index,
    layer: int,
    projection: str,
    x: np.ndarray,
    source_x: np.ndarray,
    indices: np.ndarray,
) -> tuple[mx.array, int, int]:
    source_tensors_read = 0
    peak_source_tensor_bytes = 0
    target: np.ndarray | None = None
    if projection == "down_proj" and x.shape[:-1] != indices.shape:
        raise ValueError(
            "down_proj fitting expects per-route inputs with leading shape matching indices; "
            f"got x={x.shape}, indices={indices.shape}"
        )
    for expert in sorted(int(value) for value in np.unique(indices)):
        if projection == "down_proj":
            gate_weight = np.asarray(
                read_indexed_safetensors_tensor_mlx(
                    source_dir,
                    index,
                    _source_weight_name(layer, expert, "gate_proj"),
                ).astype(mx.float32),
                dtype=np.float32,
            )
            up_weight = np.asarray(
                read_indexed_safetensors_tensor_mlx(
                    source_dir,
                    index,
                    _source_weight_name(layer, expert, "up_proj"),
                ).astype(mx.float32),
                dtype=np.float32,
            )
            weight_np = np.asarray(
                read_indexed_safetensors_tensor_mlx(
                    source_dir,
                    index,
                    _source_weight_name(layer, expert, "down_proj"),
                ).astype(mx.float32),
                dtype=np.float32,
            )
            if gate_weight.ndim != 2 or gate_weight.shape[1] != source_x.shape[-1]:
                raise ValueError(
                    f"source gate weight for layer {layer} expert {expert} must have "
                    f"input dim {source_x.shape[-1]}, found {gate_weight.shape}"
                )
            if up_weight.shape != gate_weight.shape:
                raise ValueError(
                    f"source gate/up shapes for layer {layer} expert {expert} differ: "
                    f"{gate_weight.shape} vs {up_weight.shape}"
                )
            if weight_np.ndim != 2 or weight_np.shape[1] != gate_weight.shape[0]:
                raise ValueError(
                    f"source down weight for layer {layer} expert {expert} must have "
                    f"shape [output, {gate_weight.shape[0]}], found {weight_np.shape}"
                )
            source_tensors_read += 3
            peak_source_tensor_bytes = max(
                peak_source_tensor_bytes,
                int(gate_weight.nbytes),
                int(up_weight.nbytes),
                int(weight_np.nbytes),
            )
        else:
            weight = read_indexed_safetensors_tensor_mlx(
                source_dir,
                index,
                _source_weight_name(layer, expert, projection),
            )
            weight_np = np.asarray(weight.astype(mx.float32), dtype=np.float32)
            if weight_np.ndim != 2 or weight_np.shape[1] != x.shape[-1]:
                raise ValueError(
                    f"source weight for layer {layer} expert {expert} {projection} must have "
                    f"shape [output, {x.shape[-1]}], found {weight_np.shape}"
                )
            source_tensors_read += 1
            peak_source_tensor_bytes = max(peak_source_tensor_bytes, int(weight_np.nbytes))
        if target is None:
            target = np.empty((*indices.shape, weight_np.shape[0]), dtype=np.float32)
        elif target.shape[-1] != weight_np.shape[0]:
            raise ValueError(
                f"source output dimensions changed within one projection: "
                f"{target.shape[-1]} vs {weight_np.shape[0]}"
            )
        route_positions = np.argwhere(indices == expert)
        for token_idx, route_idx in route_positions:
            if projection == "down_proj":
                source_hidden = _silu_np(gate_weight @ source_x[token_idx]) * (up_weight @ source_x[token_idx])
                target[token_idx, route_idx] = weight_np @ source_hidden
            else:
                target[token_idx, route_idx] = weight_np @ x[token_idx]
    if target is None:
        raise ValueError("no routed experts selected for source projection target")
    return mx.array(target), source_tensors_read, peak_source_tensor_bytes


def _summarize_non_expert_bind_report(report) -> dict[str, Any]:
    payload = report.to_dict()
    return {
        "loaded_count": int(payload.get("loaded_count", 0) or 0),
        "missing_count": len(payload.get("missing_model_parameters", []) or []),
        "skipped_mtp_count": len(payload.get("skipped_mtp_tensors", []) or []),
        "skipped_expert_count": len(payload.get("skipped_expert_tensors", []) or []),
        "skipped_unmatched_count": len(payload.get("skipped_unmatched_tensors", []) or []),
    }


def _materialize_block_local_sidecar_artifact(
    *,
    seed_artifact_dir: Path,
    output_dir: Path,
    layer: int,
    projection: str,
    fitted_sidecar,
    run_manifest: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    linked_group_count = link_seed_artifact_groups(
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
    )
    preserved_sidecars = copy_declared_continuous_sidecars(
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
        exclude={(int(layer), projection)},
    )
    sidecar_entry = write_continuous_sidecar(
        output_dir=output_dir,
        layer=layer,
        projection=projection,
        scale_delta=fitted_sidecar.scale_delta,
        output_bias=fitted_sidecar.output_bias,
        low_rank_left=fitted_sidecar.low_rank_left,
        low_rank_right=fitted_sidecar.low_rank_right,
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
        sidecars=[*preserved_sidecars, sidecar_entry],
        run_manifest={
            **run_manifest,
            "preserved_sidecar_count": int(len(preserved_sidecars)),
        },
    )
    return sidecar_entry, linked_group_count, len(preserved_sidecars)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a block-local GLM-4.5-Air continuous sidecar against source projection outputs."
    )
    parser.add_argument("--teacher-jsonl", required=True, action="append")
    parser.add_argument("--teacher-cache-root", required=True, action="append")
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--projection", choices=["gate_proj", "up_proj", "down_proj"], required=True)
    parser.add_argument(
        "--trainable",
        choices=["output_bias", "scale_delta", "scale_delta_output_bias", "low_rank_residual"],
        default="output_bias",
    )
    parser.add_argument("--low-rank", type=int, default=4)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument(
        "--row-indices",
        help="Evaluate comma-separated zero-based rows from each teacher JSONL instead of a prefix.",
    )
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--min-top-k", type=int, default=128)
    parser.add_argument("--allow-dirty-cache", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--append-jsonl", required=True)
    args = parser.parse_args()

    if args.max_train_rows is not None and args.max_train_rows <= 0:
        parser.error("--max-train-rows must be positive")
    if args.row_indices is not None and args.max_train_rows is not None:
        parser.error("--row-indices cannot be combined with --max-train-rows")
    try:
        row_indices = _parse_row_indices(args.row_indices) if args.row_indices is not None else None
    except ValueError as error:
        parser.error(str(error))
    if row_indices is None and args.max_train_rows is None:
        args.max_train_rows = 4
    if args.max_positions is not None and args.max_positions <= 0:
        parser.error("--max-positions must be positive")
    if args.low_rank <= 0:
        parser.error("--low-rank must be positive")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and output_dir.exists() and not args.allow_existing:
        parser.error(f"--output-dir already exists: {output_dir}")

    source_dir = _resolve_source_dir_arg(
        source_dir=args.source_dir,
        model_id=args.model_id,
        revision=args.revision,
    )
    index_path = Path(args.index_path) if args.index_path else source_dir / "model.safetensors.index.json"
    seed_artifact_dir = Path(args.seed_artifact_dir)
    teacher_jsonls = [Path(value) for value in args.teacher_jsonl]
    teacher_cache_roots = [Path(value) for value in args.teacher_cache_root]
    if len(teacher_jsonls) != len(teacher_cache_roots):
        parser.error("--teacher-jsonl and --teacher-cache-root must be supplied the same number of times")

    validations = []
    rows = []
    for teacher_jsonl, teacher_cache_root in zip(teacher_jsonls, teacher_cache_roots, strict=True):
        validation = validate_teacher_cache_metadata(
            teacher_jsonl,
            cache_root=teacher_cache_root,
            min_top_k=args.min_top_k,
            check_values=False,
        )
        if not validation["ok"]:
            raise ValueError(f"teacher cache validation failed for {teacher_jsonl}")
        if not args.allow_dirty_cache and not validation["all_memory_clean"]:
            raise ValueError(f"teacher cache must be memory-clean: {teacher_jsonl}")
        validations.append(
            {
                "teacher_jsonl": str(teacher_jsonl),
                "teacher_cache_root": str(teacher_cache_root),
                "row_count": validation["row_count"],
                "all_memory_clean": validation["all_memory_clean"],
            }
        )
        try:
            rows.extend(
                _select_rows(
                    read_teacher_cache_rows(teacher_jsonl),
                    max_rows=args.max_train_rows,
                    row_indices=row_indices,
                )
            )
        except IndexError as error:
            parser.error(str(error))

    start = time.perf_counter()
    model, _, non_expert_report, bound_layers = load_resident_air(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=str(source_dir),
        config_path=args.config_path,
        index_path=str(index_path),
        artifact_dir=seed_artifact_dir,
    )
    projection = _target_projection(model, layer=args.layer, projection=args.projection)
    x, indices, source_x, row_records = _collect_block_local_batch(
        model=model,
        teacher_rows=rows,
        layer=args.layer,
        projection=args.projection,
        max_positions=args.max_positions,
    )
    index = load_safetensors_index(index_path)
    target, source_tensors_read, peak_source_tensor_bytes = _source_projection_targets(
        source_dir=source_dir,
        index=index,
        layer=args.layer,
        projection=args.projection,
        x=x,
        source_x=source_x,
        indices=indices,
    )
    fit_fns = {
        "output_bias": fit_output_bias_sidecar_least_squares,
        "scale_delta": fit_scale_delta_sidecar_least_squares,
        "scale_delta_output_bias": fit_scale_delta_output_bias_sidecar_least_squares,
    }
    if args.trainable == "low_rank_residual":
        result = fit_low_rank_residual_sidecar_least_squares(
            projection,
            mx.array(x),
            mx.array(indices),
            target,
            rank=args.low_rank,
        )
    else:
        fit_fn = fit_fns[args.trainable]
        result = fit_fn(projection, mx.array(x), mx.array(indices), target)
    fitted_sidecar = result["sidecar"]
    eval_targets = []
    if fitted_sidecar.scale_delta is not None:
        eval_targets.append(fitted_sidecar.scale_delta)
    if fitted_sidecar.output_bias is not None:
        eval_targets.append(fitted_sidecar.output_bias)
    if fitted_sidecar.low_rank_left is not None:
        eval_targets.append(fitted_sidecar.low_rank_left)
    if fitted_sidecar.low_rank_right is not None:
        eval_targets.append(fitted_sidecar.low_rank_right)
    mx.eval(*eval_targets)

    sidecar_entry = None
    linked_group_count = None
    preserved_sidecar_count = None
    if output_dir is not None:
        sidecar_entry, linked_group_count, preserved_sidecar_count = _materialize_block_local_sidecar_artifact(
            seed_artifact_dir=seed_artifact_dir,
            output_dir=output_dir,
            layer=args.layer,
            projection=args.projection,
            fitted_sidecar=fitted_sidecar,
            run_manifest={
                "kind": "block_local_sidecar_least_squares",
                "trainable": args.trainable,
                "low_rank": int(args.low_rank) if args.trainable == "low_rank_residual" else None,
                "layer": int(args.layer),
                "projection": args.projection,
                "teacher_jsonl": [str(path) for path in teacher_jsonls],
                "teacher_cache_root": [str(path) for path in teacher_cache_roots],
                "teacher_sources": validations,
                "train_row_count": len(rows),
                "train_row_indices": None if row_indices is None else list(row_indices),
                "route_count": int(indices.size),
                "baseline_mse": result["baseline_mse"],
                "scale_delta_mse": result.get("scale_delta_mse"),
                "fitted_mse": result["fitted_mse"],
                "improvement_ratio": result["improvement_ratio"],
                "actual_ranks": result.get("actual_ranks"),
            },
        )

    elapsed = time.perf_counter() - start
    record = {
        "schema_version": 1,
        "record_type": "air_vq_block_local_sidecar_fit",
        "evidence_scope": "block_local_source_projection_residual_only",
        "model_id": args.model_id,
        "revision": args.revision,
        "seed_artifact_dir": str(seed_artifact_dir),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "source_dir": str(source_dir),
        "teacher_jsonl": [str(path) for path in teacher_jsonls],
        "teacher_cache_root": [str(path) for path in teacher_cache_roots],
        "teacher_sources": validations,
        "teacher_cache_rows": sum(int(item["row_count"]) for item in validations),
        "teacher_cache_all_memory_clean": all(bool(item["all_memory_clean"]) for item in validations),
        "layer": int(args.layer),
        "projection": args.projection,
        "trainable": args.trainable,
        "low_rank": int(args.low_rank) if args.trainable == "low_rank_residual" else None,
        "train_row_count": len(rows),
        "train_row_indices": None if row_indices is None else list(row_indices),
        "token_count": int(x.shape[0]),
        "route_count": int(indices.size),
        "selected_experts": sorted(int(value) for value in np.unique(indices)),
        "source_tensors_read": int(source_tensors_read),
        "peak_source_tensor_bytes": int(peak_source_tensor_bytes),
        "baseline_mse": result["baseline_mse"],
        "scale_delta_mse": result.get("scale_delta_mse"),
        "fitted_mse": result["fitted_mse"],
        "improvement_ratio": result["improvement_ratio"],
        "actual_ranks": result.get("actual_ranks"),
        "expert_route_counts": list(result["expert_route_counts"]),
        "linked_group_count": linked_group_count,
        "preserved_sidecar_count": preserved_sidecar_count,
        "sidecar": sidecar_entry,
        "rows": row_records,
        "non_expert_bind_report": _summarize_non_expert_bind_report(non_expert_report),
        "bound_vq_layers": list(bound_layers),
        "elapsed_seconds": elapsed,
    }
    append_jsonl(args.append_jsonl, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
