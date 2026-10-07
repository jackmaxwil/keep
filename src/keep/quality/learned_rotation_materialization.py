from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID, summarize_config, validate_glm45_air
from keep.convert.stream_convert import (
    ConvertedVQGroup,
    convert_vq_group_from_safetensors,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
    vq_group_output_filename,
)
from keep.quant.rotation import validate_rotation_matrix_np
from keep.quant.rtn import ScaleEstimator

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
    if group.rotation_id is not None:
        payload["rotation"] = "learned_orthogonal_v1"
        payload["rotation_id"] = group.rotation_id
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


def _load_rotation_matrix(path: str | Path, *, dim: int) -> np.ndarray:
    matrix_path = Path(path)
    if matrix_path.suffix == ".npy":
        values = np.load(matrix_path)
    else:
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        values = np.asarray(payload["rotation_matrix"], dtype=np.float32)
    return validate_rotation_matrix_np(values, dim=dim)


def _load_rht_signs(path: str | Path, *, dim: int) -> np.ndarray:
    signs_path = Path(path)
    if signs_path.suffix == ".npy":
        values = np.load(signs_path)
    else:
        payload = json.loads(signs_path.read_text(encoding="utf-8"))
        values = np.asarray(payload["rht_signs"], dtype=np.int8)
    signs = np.asarray(values, dtype=np.int8)
    if signs.shape != (dim,):
        raise ValueError(f"rht_signs must have shape ({dim},), found {signs.shape}")
    if not np.all((signs == 1) | (signs == -1)):
        raise ValueError("rht_signs values must be -1 or 1")
    return signs


def _manifest_targets(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rotation_id = str(payload.get("rotation_id") or manifest_path.stem)
    targets = payload.get("projections")
    if not isinstance(targets, list) or not targets:
        raise ValueError("rotation manifest must contain a non-empty projections list")
    normalized: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("rotation manifest projections must be objects")
        layer = int(target["layer"])
        projection = str(target["projection"])
        if projection not in SUPPORTED_PROJECTIONS:
            raise ValueError(f"unsupported projection {projection!r}; expected one of {SUPPORTED_PROJECTIONS}")
        matrix_raw = target.get("rotation_matrix_path")
        signs_raw = target.get("rht_signs_path")
        if (matrix_raw is None) == (signs_raw is None):
            raise ValueError("each projection must contain exactly one of rotation_matrix_path or rht_signs_path")
        matrix_path = None
        signs_path = None
        if matrix_raw is not None:
            matrix_path = Path(str(matrix_raw))
            if not matrix_path.is_absolute():
                matrix_path = manifest_path.parent / matrix_path
        if signs_raw is not None:
            signs_path = Path(str(signs_raw))
            if not signs_path.is_absolute():
                signs_path = manifest_path.parent / signs_path
        target_rotation_id = str(target.get("rotation_id") or f"{rotation_id}:{layer}:{projection}")
        payload = {
            "layer": layer,
            "projection": projection,
            "rotation_id": target_rotation_id,
        }
        if matrix_path is not None:
            payload["rotation_matrix_path"] = matrix_path
        if signs_path is not None:
            payload["rht_signs_path"] = signs_path
        normalized.append(payload)
    return rotation_id, normalized


def materialize_learned_rotation_projection_candidates(
    config: dict[str, Any],
    *,
    source_dir: str | Path,
    index_path: str | Path,
    seed_artifact_dir: str | Path,
    rotation_manifest_path: str | Path,
    output_dir: str | Path,
    model_id: str = GLM45_AIR_MODEL_ID,
    group_size: int = 512,
    code_bits: int = 8,
    scale_estimator: ScaleEstimator = "max_abs",
    expert_workers: int = 1,
    strict_config: bool = True,
    allow_existing: bool = False,
    manifest_name: str = "learned-rotation-materialization-manifest.json",
) -> dict[str, Any]:
    if expert_workers <= 0:
        raise ValueError("expert_workers must be positive")

    rotation_id, targets = _manifest_targets(rotation_manifest_path)
    for target in targets:
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

    planned_targets: list[tuple[dict[str, Any], Any, str, np.ndarray, str]] = []
    rewritten_filenames: set[str] = set()
    for target in targets:
        group = _select_group(
            plan,
            layer=int(target["layer"]),
            projection=str(target["projection"]),
        )
        if "rotation_matrix_path" in target:
            transform_kind = "rotation_matrix"
            transform = _load_rotation_matrix(target["rotation_matrix_path"], dim=group.input_dims)
        else:
            transform_kind = "rht_signs"
            transform = _load_rht_signs(target["rht_signs_path"], dim=group.input_dims)
        target_filename = vq_group_output_filename(group)
        if target_filename in rewritten_filenames:
            raise ValueError(f"duplicate learned-rotation target for {target_filename}")
        target_path = output_root / target_filename
        if (target_path.exists() or target_path.is_symlink()) and not allow_existing:
            raise FileExistsError(f"{target_path} already exists")
        rewritten_filenames.add(target_filename)
        planned_targets.append((target, group, transform_kind, transform, target_filename))

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
    for target, group, transform_kind, transform, target_filename in planned_targets:
        target_path = output_root / target_filename
        converted = convert_vq_group_from_safetensors(
            source_dir=source_root,
            index=index,
            group=group,
            output_path=target_path,
            scale_estimator=scale_estimator,
            expert_workers=expert_workers,
            rotation_matrix=transform if transform_kind == "rotation_matrix" else None,
            rotation_id=str(target["rotation_id"]) if transform_kind == "rotation_matrix" else None,
            rht_signs=transform if transform_kind == "rht_signs" else None,
            rht_id=str(target["rotation_id"]) if transform_kind == "rht_signs" else None,
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
            "rotation_id": str(target["rotation_id"]),
            "source_shards": list(group.source_shards),
        }
        if transform_kind == "rotation_matrix":
            projection_payload["rotation"] = "learned_orthogonal_v1"
            projection_payload["rotation_matrix_path"] = str(target["rotation_matrix_path"])
        else:
            projection_payload["rotation"] = "learned_hadamard_signs_v1"
            projection_payload["rht_signs_path"] = str(target["rht_signs_path"])
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
        "learned_rotation_materialization": {"projections": projection_payloads},
    }
    _write_json(output_root / "conversion-manifest.json", conversion_manifest)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "air_vq_learned_rotation_materialization_manifest",
        "method": {
            "kind": "spinquant_cayley_learned_rotation_rtn_batch",
            "rotation": "learned_orthogonal_v1",
            "rotation_id": rotation_id,
            "seed_artifact_mutated": False,
            "candidate_artifact_mutated": False,
            "notes": "Multi-projection learned orthogonal rotation RTN candidate.",
        },
        "model_id": model_id,
        "source_dir": str(source_root),
        "index_path": str(index_path),
        "seed_artifact_dir": str(seed_root),
        "rotation_manifest_path": str(rotation_manifest_path),
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


__all__ = ["materialize_learned_rotation_projection_candidates"]
