from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID, summarize_config, validate_glm45_air
from keep.convert.stream_convert import load_safetensors_index
from ramp.models.glm4_moe_adapter import GLM4MoEGate
from keep.validate.glm45_air_hessian_probe import (
    _full_8bit_table_from_layer,
    _sample_output_rows,
    _select_experts,
    _select_or_random_input_states,
    HessianReassignedWeight,
    hessian_weighted_reassign_codes,
)
from keep.quality.imatrix import ProjectionImatrixEntry
from keep.validate.glm45_air_vq import (
    _expert_weight_name,
    _load_switch_glu,
    _read_named_tensor,
    _routing_config,
)


GLU_PROJECTIONS = ("gate_proj", "up_proj")


def _imatrix_importance_vector(
    imatrix_importance: ProjectionImatrixEntry | np.ndarray,
    *,
    input_dim: int,
    layer: int | None = None,
    projection: str | None = None,
    expert: int | None = None,
) -> np.ndarray:
    if isinstance(imatrix_importance, ProjectionImatrixEntry):
        if layer is not None and imatrix_importance.layer != layer:
            raise ValueError(f"imatrix entry layer {imatrix_importance.layer} does not match expected {layer}")
        if projection is not None and imatrix_importance.projection != projection:
            raise ValueError(
                f"imatrix entry projection {imatrix_importance.projection!r} "
                f"does not match expected {projection!r}"
            )
        if expert is not None and imatrix_importance.expert != expert:
            raise ValueError(f"imatrix entry expert {imatrix_importance.expert} does not match expected {expert}")
        importance = imatrix_importance.routing_weighted_importance
    else:
        importance = imatrix_importance

    vector = np.asarray(importance, dtype=np.float32)
    if vector.shape != (input_dim,):
        raise ValueError(f"imatrix_importance must have shape ({input_dim},), found {vector.shape}")
    if not np.isfinite(vector).all() or np.any(vector < 0.0):
        raise ValueError("imatrix_importance must be finite and non-negative")
    return vector


def imatrix_weighted_reassign_codes(
    source_weight: np.ndarray,
    current_codes: np.ndarray,
    scales: np.ndarray,
    imatrix_importance: ProjectionImatrixEntry | np.ndarray,
    *,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int = 8,
    row_indices: np.ndarray | tuple[int, ...] | list[int] | None = None,
    codeword_chunk_size: int = 4096,
    layer: int | None = None,
    projection: str | None = None,
    expert: int | None = None,
) -> HessianReassignedWeight:
    """Reassign VQ codes using a Step-2 imatrix vector as the diagonal weight."""

    source = np.asarray(source_weight)
    if source.ndim != 2:
        raise ValueError(f"source_weight must be 2D [out, in], found {source.shape}")
    importance = _imatrix_importance_vector(
        imatrix_importance,
        input_dim=int(source.shape[1]),
        layer=layer,
        projection=projection,
        expert=expert,
    )
    return hessian_weighted_reassign_codes(
        source_weight,
        current_codes,
        scales,
        importance,
        codebook=codebook,
        group_size=group_size,
        code_bits=code_bits,
        row_indices=row_indices,
        codeword_chunk_size=codeword_chunk_size,
    )


def _switch_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _group_filename(layer: int, projection: str) -> str:
    return f"layer-{layer:05d}-{projection}.safetensors"


def _relative_symlink_target(source_path: Path, target_parent: Path) -> str:
    return os.path.relpath(source_path.resolve(), start=target_parent.resolve())


def _link_seed_groups(*, seed_artifact_dir: Path, output_dir: Path, rewritten_filenames: set[str]) -> int:
    linked_count = 0
    for source_path in sorted(seed_artifact_dir.glob("layer-*.safetensors")):
        target_path = output_dir / source_path.name
        if source_path.name in rewritten_filenames:
            continue
        if target_path.exists() or target_path.is_symlink():
            linked_count += 1
            continue
        os.symlink(_relative_symlink_target(source_path, target_path.parent), target_path)
        linked_count += 1
    return linked_count


def _load_safetensors_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="np") as handle:
        return dict(handle.metadata() or {})


def _save_rewritten_group(
    *,
    seed_path: Path,
    target_path: Path,
    codes_name: str,
    codes: np.ndarray,
) -> None:
    arrays = dict(mx.load(str(seed_path)))
    metadata = _load_safetensors_metadata(seed_path)
    arrays[codes_name] = mx.array(codes)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    mx.save_safetensors(str(target_path), arrays, metadata=metadata)


def _projection_layer(switch_glu, projection: str):
    if projection == "gate_proj":
        return switch_glu.gate_proj
    if projection == "up_proj":
        return switch_glu.up_proj
    raise ValueError(f"unsupported projection {projection!r}")


def _select_rows(output_dim: int, max_output_rows: int | None) -> np.ndarray:
    if max_output_rows is None:
        return np.arange(output_dim, dtype=np.int64)
    return _sample_output_rows(output_dim, max_output_rows)


def _row_selection_summary(rows: np.ndarray, *, output_dim: int) -> dict[str, Any]:
    values = np.asarray(rows, dtype=np.int64)
    summary: dict[str, Any] = {
        "count": int(values.size),
        "output_dim": int(output_dim),
        "mode": "all" if values.size == output_dim else "sampled",
    }
    if values.size:
        summary["first"] = int(values[0])
        summary["last"] = int(values[-1])
    if values.size <= 256:
        summary["rows"] = [int(value) for value in values]
    return summary


def materialize_hessian_rounding_candidate(
    config: dict[str, Any],
    *,
    source_dir: str | Path,
    index_path: str | Path,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    model_id: str = GLM45_AIR_MODEL_ID,
    layer: int = 1,
    tokens: int = 2,
    seed: int = 20260624,
    input_scale: float = 1.0,
    input_states: np.ndarray | None = None,
    input_source: str | None = None,
    input_state_indices: tuple[int, ...] | None = None,
    projections: tuple[str, ...] = GLU_PROJECTIONS,
    max_experts: int = 8,
    max_output_rows: int | None = None,
    strict_config: bool = True,
    allow_existing: bool = False,
    manifest_name: str = "hessian-rounding-manifest.json",
) -> dict[str, Any]:
    """Create a linked candidate with selected 8-bit gate/up shards rewritten.

    This is a bounded Lane 4 experiment: diagonal Hessian, frozen scales, no RHT,
    and no full BlockLDLQ. It writes a new artifact directory and never mutates
    the seed artifact.
    """

    normalized_projections = tuple(str(projection) for projection in projections)
    if not normalized_projections:
        raise ValueError("at least one projection is required")
    unsupported = sorted(set(normalized_projections) - set(GLU_PROJECTIONS))
    if unsupported:
        raise ValueError(f"hessian materialization only supports gate_proj/up_proj; unsupported: {unsupported}")

    summary = summarize_config(config, model_id=model_id)
    if strict_config:
        failed_checks = validate_glm45_air(summary)
        if failed_checks:
            raise ValueError(f"config does not match GLM-4.5-Air: {', '.join(failed_checks)}")
    if summary.model_type != "glm4_moe":
        raise ValueError(f"expected model_type='glm4_moe', found {summary.model_type!r}")
    if layer < summary.dense_layers or layer >= summary.num_hidden_layers:
        raise ValueError(
            f"layer {layer} is not a sparse MoE layer; sparse range is "
            f"[{summary.dense_layers}, {summary.num_hidden_layers})"
        )

    source_root = Path(source_dir)
    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    if output_root.exists() and not allow_existing:
        raise FileExistsError(f"{output_root} already exists")
    output_root.mkdir(parents=True, exist_ok=True)

    rewritten_filenames = {_group_filename(layer, projection) for projection in normalized_projections}
    linked_count = _link_seed_groups(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
        rewritten_filenames=rewritten_filenames,
    )

    index = load_safetensors_index(index_path)
    routing = _routing_config(config, model_id=model_id)
    gate_weight_name = f"model.layers.{layer}.mlp.gate.weight"
    correction_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    gate_weight = _read_named_tensor(source_root, index, gate_weight_name)
    source_tensors_read = 1
    peak_source_bytes = int(gate_weight.nbytes)
    if correction_name in index.weight_map:
        correction = _read_named_tensor(source_root, index, correction_name)
        source_tensors_read += 1
        peak_source_bytes = max(peak_source_bytes, int(correction.nbytes))
    else:
        correction = np.zeros((summary.n_routed_experts,), dtype=np.float32)

    x, tokens, resolved_input_source, resolved_state_indices = _select_or_random_input_states(
        summary_hidden_size=summary.hidden_size,
        tokens=tokens,
        seed=seed,
        input_scale=input_scale,
        input_states=input_states,
        input_source=input_source,
        input_state_indices=input_state_indices,
    )
    gate = GLM4MoEGate(
        routing,
        weight=mx.array(gate_weight),
        e_score_correction_bias=mx.array(correction),
    )
    indices_mx, scores_mx = gate(mx.array(x))
    mx.eval(indices_mx, scores_mx)
    indices = np.asarray(indices_mx, dtype=np.int64)
    scores = np.asarray(scores_mx.astype(mx.float32), dtype=np.float32)
    selected_experts = _select_experts(indices, max_experts)
    selected_routes = int(np.count_nonzero(np.isin(indices, np.asarray(selected_experts, dtype=np.int64))))
    hessian_diag = np.mean(x.astype(np.float64) * x.astype(np.float64), axis=0).astype(np.float32)

    switch_glu = _load_switch_glu(seed_root, layer)
    projection_results: dict[str, dict[str, Any]] = {}
    rewritten_group_count = 0
    sampled_rows_by_projection: dict[str, tuple[int, ...]] = {}

    for projection in normalized_projections:
        projection_layer = _projection_layer(switch_glu, projection)
        if projection_layer.code_bits != 8:
            raise ValueError(
                f"layer {layer} {projection} has code_bits={projection_layer.code_bits}; "
                "materialized diagonal-Hessian reassignment currently supports 8-bit groups only"
            )
        sampled_rows = _select_rows(projection_layer.output_dims, max_output_rows)
        sampled_rows_by_projection[projection] = tuple(int(value) for value in sampled_rows)
        full_codebook = _full_8bit_table_from_layer(projection_layer)
        codes_name = f"{_switch_prefix(layer, projection)}.codes"
        scales_name = f"{_switch_prefix(layer, projection)}.scales"
        seed_path = seed_root / _group_filename(layer, projection)
        target_path = output_root / _group_filename(layer, projection)
        arrays = mx.load(str(seed_path))
        codes = np.asarray(arrays[codes_name]).copy()
        scales = np.asarray(arrays[scales_name], dtype=np.float32)

        current_error = 0.0
        hessian_error = 0.0
        changed_code_count = 0
        code_count = 0
        expert_stats: dict[str, dict[str, Any]] = {}
        for expert in selected_experts:
            source_weight = _read_named_tensor(
                source_root,
                index,
                _expert_weight_name(layer, expert, projection),
            ).astype(np.float32, copy=False)
            source_tensors_read += 1
            peak_source_bytes = max(peak_source_bytes, int(source_weight.nbytes))
            reassigned = hessian_weighted_reassign_codes(
                source_weight,
                codes[expert],
                scales[expert],
                hessian_diag,
                codebook=full_codebook,
                group_size=projection_layer.group_size,
                code_bits=projection_layer.code_bits,
                row_indices=sampled_rows,
            )
            codes[expert, sampled_rows] = reassigned.codes
            expert_stats[f"expert_{expert}"] = asdict(reassigned.stats)
            current_error += reassigned.stats.current_hessian_weighted_error
            hessian_error += reassigned.stats.hessian_weighted_error
            changed_code_count += reassigned.stats.changed_code_count
            code_count += reassigned.stats.code_count

        _save_rewritten_group(
            seed_path=seed_path,
            target_path=target_path,
            codes_name=codes_name,
            codes=codes,
        )
        rewritten_group_count += 1
        projection_results[projection] = {
            "projection": projection,
            "selected_experts": list(selected_experts),
            "output_row_selection": _row_selection_summary(
                sampled_rows,
                output_dim=projection_layer.output_dims,
            ),
            "current_hessian_weighted_error": current_error,
            "hessian_weighted_error": hessian_error,
            "hessian_weighted_error_ratio": (
                hessian_error / current_error if abs(current_error) > 1e-12 else 1.0
            ),
            "changed_code_count": int(changed_code_count),
            "code_count": int(code_count),
            "changed_code_fraction": float(changed_code_count / code_count) if code_count else 0.0,
            "expert_stats": expert_stats,
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": {
            "kind": "diagonal_hessian_code_reassignment",
            "candidate_artifact_mutated": False,
            "seed_artifact_mutated": False,
            "full_blockldlq": False,
            "rht": False,
            "frozen_scales": True,
            "notes": (
                "Bounded Lane 4 materialized experiment: selected 8-bit gate/up groups only, "
                "diagonal Hessian proxy, no training, no candidate acceptance."
            ),
        },
        "model_id": model_id,
        "layer": int(layer),
        "tokens": int(tokens),
        "input_source": resolved_input_source,
        "input_state_indices": list(resolved_state_indices),
        "seed_artifact_dir": str(seed_root),
        "output_dir": str(output_root),
        "source_dir": str(source_root),
        "projections": list(normalized_projections),
        "selected_experts": [int(value) for value in selected_experts],
        "routes": selected_routes,
        "routed_indices": [[int(value) for value in row] for row in indices],
        "router_scores": [[float(value) for value in row] for row in scores],
        "hessian": {
            "proxy": "mean_squared_moe_input_diagonal",
            "sample_count": int(x.shape[0]),
            "dimension": int(hessian_diag.shape[0]),
            "diag_min": float(np.min(hessian_diag)) if hessian_diag.size else 0.0,
            "diag_max": float(np.max(hessian_diag)) if hessian_diag.size else 0.0,
            "diag_mean": float(np.mean(hessian_diag)) if hessian_diag.size else 0.0,
            "diag_std": float(np.std(hessian_diag)) if hessian_diag.size else 0.0,
        },
        "linked_group_count": linked_count,
        "rewritten_group_count": rewritten_group_count,
        "source_tensors_read": source_tensors_read,
        "peak_source_tensor_bytes": peak_source_bytes,
        "projection_results": projection_results,
    }
    manifest_path = output_root / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


__all__ = ["imatrix_weighted_reassign_codes", "materialize_hessian_rounding_candidate"]
