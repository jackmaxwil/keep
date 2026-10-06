from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, summarize_config, validate_glm45_air
from mlx_vq.convert.stream_convert import (
    ConvertedVQGroup,
    convert_vq_group_from_safetensors,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
    vq_group_output_filename,
)
from mlx_vq.quant.rht import deterministic_rht_signs
from mlx_vq.quant.rtn import ScaleEstimator


SUPPORTED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


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


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _converted_group_payload(group: ConvertedVQGroup) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "converted",
        "output_path": str(group.output_path),
        "codes_name": group.codes_name,
        "scales_name": group.scales_name,
        "codes_shape": list(group.codes_shape),
        "scales_shape": list(group.scales_shape),
        "source_tensors_read": group.source_tensors_read,
        "peak_source_tensor_bytes": group.peak_source_tensor_bytes,
        "code_bits": group.code_bits,
        "group_size": group.group_size,
        "scale_estimator": group.scale_estimator,
        "elapsed_seconds": group.elapsed_seconds,
        "expert_workers": group.expert_workers,
    }
    if group.rht_seed is not None:
        payload["rht"] = "hadamard_signs_v1"
        payload["rht_seed"] = group.rht_seed
    return payload


def _select_group(plan, *, layer: int, projection: str):
    for group in plan.vq_groups:
        if group.layer == layer and group.projection == projection:
            return group
    raise ValueError(f"no planned VQ group for layer {layer} {projection}")


def _validate_sparse_layer(config: dict[str, Any], *, model_id: str, layer: int, strict_config: bool):
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
    return summary


def _normalize_target(target: dict[str, Any]) -> dict[str, Any]:
    layer = int(target["layer"])
    projection = str(target["projection"])
    if projection not in SUPPORTED_PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}; expected one of {SUPPORTED_PROJECTIONS}")
    rht_seed = target.get("rht_seed")
    if rht_seed is None:
        raise ValueError("rht_seed is required for every RHT target")
    return {"layer": layer, "projection": projection, "rht_seed": str(rht_seed)}


def materialize_rht_projection_candidates(
    config: dict[str, Any],
    *,
    source_dir: str | Path,
    index_path: str | Path,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    targets: list[dict[str, Any]],
    model_id: str = GLM45_AIR_MODEL_ID,
    group_size: int = 512,
    code_bits: int = 8,
    scale_estimator: ScaleEstimator = "max_abs",
    expert_workers: int = 1,
    strict_config: bool = True,
    allow_existing: bool = False,
    manifest_name: str = "rht-materialization-manifest.json",
) -> dict[str, Any]:
    """Create a linked candidate with multiple projections re-quantized after RHT."""

    if not targets:
        raise ValueError("at least one RHT target is required")
    if expert_workers <= 0:
        raise ValueError("expert_workers must be positive")

    normalized_targets = [_normalize_target(target) for target in targets]
    for target in normalized_targets:
        _validate_sparse_layer(
            config,
            model_id=model_id,
            layer=int(target["layer"]),
            strict_config=strict_config,
        )

    source_root = Path(source_dir)
    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    if output_root.exists() and not allow_existing:
        raise FileExistsError(f"{output_root} already exists")

    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id=model_id,
        group_size=group_size,
        code_bits=code_bits,
    )

    planned_targets: list[tuple[dict[str, Any], Any, str]] = []
    rewritten_filenames: set[str] = set()
    for target in normalized_targets:
        group = _select_group(
            plan,
            layer=int(target["layer"]),
            projection=str(target["projection"]),
        )
        deterministic_rht_signs(group.input_dims, seed=str(target["rht_seed"]))
        target_filename = vq_group_output_filename(group)
        if target_filename in rewritten_filenames:
            raise ValueError(f"duplicate RHT target for {target_filename}")
        target_path = output_root / target_filename
        if (target_path.exists() or target_path.is_symlink()) and not allow_existing:
            raise FileExistsError(f"{target_path} already exists")
        rewritten_filenames.add(target_filename)
        planned_targets.append((target, group, target_filename))

    output_root.mkdir(parents=True, exist_ok=True)
    linked_count = _link_seed_groups(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
        rewritten_filenames=rewritten_filenames,
    )

    group_payloads: list[dict[str, Any]] = []
    projection_payloads: list[dict[str, Any]] = []
    total_source_tensors_read = 0
    peak_source_tensor_bytes = 0
    for target, group, target_filename in planned_targets:
        target_path = output_root / target_filename
        converted = convert_vq_group_from_safetensors(
            source_dir=source_root,
            index=index,
            group=group,
            output_path=target_path,
            scale_estimator=scale_estimator,
            expert_workers=expert_workers,
            rht_seed=str(target["rht_seed"]),
        )
        group_payload = _converted_group_payload(converted)
        projection_payload = {
            "layer": int(target["layer"]),
            "projection": str(target["projection"]),
            "output_path": str(target_path),
            "input_dims": int(group.input_dims),
            "output_dims": int(group.output_dims),
            "experts": [int(expert) for expert in group.experts],
            "code_bits": int(group.code_bits),
            "group_size": int(group.group_size),
            "scale_estimator": scale_estimator,
            "rht": "hadamard_signs_v1",
            "rht_seed": str(target["rht_seed"]),
            "source_shards": list(group.source_shards),
        }
        group_payloads.append(group_payload)
        projection_payloads.append(projection_payload)
        total_source_tensors_read += converted.source_tensors_read
        peak_source_tensor_bytes = max(peak_source_tensor_bytes, converted.peak_source_tensor_bytes)

    conversion_manifest: dict[str, Any] = {
        "model_id": model_id,
        "output_dir": str(output_root),
        "planned_vq_groups": len(plan.vq_groups),
        "converted_vq_groups": len(group_payloads),
        "existing_vq_groups": 0,
        "ready_vq_groups": len(group_payloads) + linked_count,
        "skipped_vq_groups": len(plan.vq_groups) - len(group_payloads),
        "skipped_existing_outputs": 0,
        "total_source_tensors_read": total_source_tensors_read,
        "peak_source_tensor_bytes": peak_source_tensor_bytes,
        "scale_estimator": scale_estimator,
        "code_bits_policy": dict(plan.code_bits_policy),
        "groups": group_payloads,
        "rht_materialization": {"projections": projection_payloads},
    }
    _write_json(output_root / "conversion-manifest.json", conversion_manifest)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "air_vq_rht_materialization_manifest",
        "method": {
            "kind": "quip_rht_rtn_batch",
            "rht": "hadamard_signs_v1",
            "full_quip": False,
            "seed_artifact_mutated": False,
            "candidate_artifact_mutated": False,
            "notes": "Multi-projection RHT RTN candidate; not a full QuIP# implementation.",
        },
        "model_id": model_id,
        "source_dir": str(source_root),
        "index_path": str(index_path),
        "seed_artifact_dir": str(seed_root),
        "output_dir": str(output_root),
        "linked_group_count": linked_count,
        "rewritten_group_count": len(group_payloads),
        "source_tensors_read": total_source_tensors_read,
        "peak_source_tensor_bytes": peak_source_tensor_bytes,
        "projections": projection_payloads,
        "converted_groups": group_payloads,
    }
    _write_json(output_root / manifest_name, manifest)
    return manifest


def materialize_rht_projection_candidate(
    config: dict[str, Any],
    *,
    source_dir: str | Path,
    index_path: str | Path,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    model_id: str = GLM45_AIR_MODEL_ID,
    layer: int,
    projection: str,
    rht_seed: int | str,
    group_size: int = 512,
    code_bits: int = 8,
    scale_estimator: ScaleEstimator = "max_abs",
    expert_workers: int = 1,
    strict_config: bool = True,
    allow_existing: bool = False,
    manifest_name: str = "rht-materialization-manifest.json",
) -> dict[str, Any]:
    """Create a linked candidate with one projection re-quantized after RHT.

    The seed artifact is never mutated. The output artifact symlinks every
    untouched ``layer-*.safetensors`` group and stores a fresh RHT-converted
    shard for the selected layer/projection.
    """

    normalized_projection = str(projection)
    if normalized_projection not in SUPPORTED_PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}; expected one of {SUPPORTED_PROJECTIONS}")
    if rht_seed is None:
        raise ValueError("rht_seed is required")
    if expert_workers <= 0:
        raise ValueError("expert_workers must be positive")

    _validate_sparse_layer(config, model_id=model_id, layer=layer, strict_config=strict_config)
    source_root = Path(source_dir)
    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    if output_root.exists() and not allow_existing:
        raise FileExistsError(f"{output_root} already exists")

    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id=model_id,
        group_size=group_size,
        code_bits=code_bits,
    )
    group = _select_group(plan, layer=layer, projection=normalized_projection)
    deterministic_rht_signs(group.input_dims, seed=rht_seed)

    target_filename = vq_group_output_filename(group)
    target_path = output_root / target_filename
    if (target_path.exists() or target_path.is_symlink()) and not allow_existing:
        raise FileExistsError(f"{target_path} already exists")

    output_root.mkdir(parents=True, exist_ok=True)
    linked_count = _link_seed_groups(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
        rewritten_filenames={target_filename},
    )
    converted = convert_vq_group_from_safetensors(
        source_dir=source_root,
        index=index,
        group=group,
        output_path=target_path,
        scale_estimator=scale_estimator,
        expert_workers=expert_workers,
        rht_seed=rht_seed,
    )
    group_payload = _converted_group_payload(converted)
    projection_payload = {
        "layer": int(layer),
        "projection": normalized_projection,
        "output_path": str(target_path),
        "input_dims": int(group.input_dims),
        "output_dims": int(group.output_dims),
        "experts": [int(expert) for expert in group.experts],
        "code_bits": int(group.code_bits),
        "group_size": int(group.group_size),
        "scale_estimator": scale_estimator,
        "rht": "hadamard_signs_v1",
        "rht_seed": str(rht_seed),
        "source_shards": list(group.source_shards),
    }
    conversion_manifest: dict[str, Any] = {
        "model_id": model_id,
        "output_dir": str(output_root),
        "planned_vq_groups": len(plan.vq_groups),
        "converted_vq_groups": 1,
        "existing_vq_groups": 0,
        "ready_vq_groups": 1 + linked_count,
        "skipped_vq_groups": len(plan.vq_groups) - 1,
        "skipped_existing_outputs": 0,
        "total_source_tensors_read": converted.source_tensors_read,
        "peak_source_tensor_bytes": converted.peak_source_tensor_bytes,
        "scale_estimator": scale_estimator,
        "code_bits_policy": dict(plan.code_bits_policy),
        "groups": [group_payload],
        "rht_materialization": projection_payload,
    }
    _write_json(output_root / "conversion-manifest.json", conversion_manifest)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "air_vq_rht_materialization_manifest",
        "method": {
            "kind": "quip_rht_rtn",
            "rht": "hadamard_signs_v1",
            "full_quip": False,
            "seed_artifact_mutated": False,
            "candidate_artifact_mutated": False,
            "notes": "One-projection RHT RTN candidate; not a full QuIP# implementation.",
        },
        "model_id": model_id,
        "source_dir": str(source_root),
        "index_path": str(index_path),
        "seed_artifact_dir": str(seed_root),
        "output_dir": str(output_root),
        "linked_group_count": linked_count,
        "rewritten_group_count": 1,
        "source_tensors_read": converted.source_tensors_read,
        "peak_source_tensor_bytes": converted.peak_source_tensor_bytes,
        "projection": projection_payload,
        "converted_group": group_payload,
    }
    _write_json(output_root / manifest_name, manifest)
    return manifest


__all__ = ["materialize_rht_projection_candidate", "materialize_rht_projection_candidates"]
