from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx


SPARSE_RESIDUAL_MANIFEST_KEY = "sparse_residuals"
SPARSE_RESIDUAL_DIR = "sparse_residuals"
SPARSE_RESIDUAL_SCHEMA_VERSION = 1
SUPPORTED_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}


@dataclass(frozen=True)
class SparseSwitchLinearResidualRows:
    expert_indices: mx.array
    output_indices: mx.array
    values: mx.array


def sparse_residual_relpath(layer: int, projection: str) -> str:
    _validate_projection(projection)
    return f"{SPARSE_RESIDUAL_DIR}/layer-{layer:05d}-{projection}.safetensors"


def load_conversion_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(artifact_dir) / "conversion-manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return payload


def load_switch_linear_sparse_residual_rows(
    artifact_dir: str | Path,
    *,
    layer: int,
    projection: str,
    input_dims: int,
    output_dims: int,
    num_experts: int,
) -> SparseSwitchLinearResidualRows | None:
    _validate_projection(projection)
    artifact_root = Path(artifact_dir)
    manifest = load_conversion_manifest(artifact_root)
    sparse = manifest.get(SPARSE_RESIDUAL_MANIFEST_KEY)
    if not isinstance(sparse, dict) or sparse.get("enabled") is not True:
        return None
    if int(sparse.get("schema_version", 0)) != SPARSE_RESIDUAL_SCHEMA_VERSION:
        raise ValueError(
            f"{SPARSE_RESIDUAL_MANIFEST_KEY}.schema_version must be {SPARSE_RESIDUAL_SCHEMA_VERSION}"
        )
    entry = _find_residual_entry(sparse, layer=layer, projection=projection)
    if entry is None:
        return None
    relpath = entry.get("path")
    if not isinstance(relpath, str) or not relpath:
        raise ValueError(f"sparse residual entry for layer {layer} {projection} needs a path")
    residual_path = _resolve_artifact_relative_path(artifact_root, relpath)
    arrays = mx.load(str(residual_path))
    expert_indices = arrays.get("expert_indices")
    output_indices = arrays.get("output_indices")
    values = arrays.get("values")
    if expert_indices is None or output_indices is None or values is None:
        raise ValueError(f"{residual_path} must contain expert_indices, output_indices, and values")
    _validate_residual_tensors(
        expert_indices=expert_indices,
        output_indices=output_indices,
        values=values,
        input_dims=input_dims,
        output_dims=output_dims,
        num_experts=num_experts,
    )
    return SparseSwitchLinearResidualRows(
        expert_indices=expert_indices,
        output_indices=output_indices,
        values=values,
    )


def write_sparse_residual_rows(
    *,
    output_dir: str | Path,
    layer: int,
    projection: str,
    expert_indices: mx.array,
    output_indices: mx.array,
    values: mx.array,
    input_dims: int,
    output_dims: int,
    num_experts: int,
) -> dict[str, Any]:
    _validate_projection(projection)
    _validate_residual_tensors(
        expert_indices=expert_indices,
        output_indices=output_indices,
        values=values,
        input_dims=input_dims,
        output_dims=output_dims,
        num_experts=num_experts,
    )
    output_root = Path(output_dir)
    relpath = sparse_residual_relpath(layer, projection)
    residual_path = output_root / relpath
    residual_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(residual_path),
        {
            "expert_indices": expert_indices.astype(mx.int32),
            "output_indices": output_indices.astype(mx.int32),
            "values": values.astype(mx.float16),
        },
        metadata={
            "sparse_residual_config": json.dumps(
                {
                    "schema_version": SPARSE_RESIDUAL_SCHEMA_VERSION,
                    "layer": int(layer),
                    "projection": projection,
                    "format": "row_residuals",
                    "row_count": int(values.shape[0]),
                },
                sort_keys=True,
            )
        },
    )
    return {
        "layer": int(layer),
        "projection": projection,
        "path": relpath,
        "format": "row_residuals",
        "row_count": int(values.shape[0]),
        "input_dims": int(input_dims),
        "output_dims": int(output_dims),
        "num_experts": int(num_experts),
    }


def write_sparse_residual_artifact_manifest(
    *,
    seed_manifest: dict[str, Any],
    output_dir: str | Path,
    residuals: list[dict[str, Any]],
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = dict(seed_manifest)
    payload: dict[str, Any] = {
        "schema_version": SPARSE_RESIDUAL_SCHEMA_VERSION,
        "enabled": True,
        "format": "switch_linear_row_residuals",
        "residuals": residuals,
    }
    if run_manifest is not None:
        payload["run"] = run_manifest
    manifest[SPARSE_RESIDUAL_MANIFEST_KEY] = payload
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _find_residual_entry(sparse: dict[str, Any], *, layer: int, projection: str) -> dict[str, Any] | None:
    entries = sparse.get("residuals", [])
    if not isinstance(entries, list):
        raise ValueError(f"{SPARSE_RESIDUAL_MANIFEST_KEY}.residuals must be a list")
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("sparse residual entries must be objects")
        if int(item.get("layer", -1)) == int(layer) and item.get("projection") == projection:
            return item
    return None


def _validate_residual_tensors(
    *,
    expert_indices: mx.array,
    output_indices: mx.array,
    values: mx.array,
    input_dims: int,
    output_dims: int,
    num_experts: int,
) -> None:
    if values.ndim != 2 or values.shape[1] != input_dims:
        raise ValueError(f"sparse residual values must have shape [rows, {input_dims}], found {values.shape}")
    row_count = values.shape[0]
    if expert_indices.shape != (row_count,):
        raise ValueError(f"expert_indices must have shape ({row_count},), found {expert_indices.shape}")
    if output_indices.shape != (row_count,):
        raise ValueError(f"output_indices must have shape ({row_count},), found {output_indices.shape}")
    if row_count == 0:
        return
    expert_min = int(mx.min(expert_indices).item())
    expert_max = int(mx.max(expert_indices).item())
    output_min = int(mx.min(output_indices).item())
    output_max = int(mx.max(output_indices).item())
    if expert_min < 0 or expert_max >= num_experts:
        raise ValueError("expert_indices out of range")
    if output_min < 0 or output_max >= output_dims:
        raise ValueError("output_indices out of range")


def _resolve_artifact_relative_path(artifact_root: Path, relpath: str) -> Path:
    residual_path = Path(relpath)
    if residual_path.is_absolute():
        raise ValueError(f"sparse residual path must be artifact-relative, got {relpath!r}")
    root = artifact_root.resolve(strict=False)
    candidate = (root / residual_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"sparse residual path escapes artifact root: {relpath!r}") from error
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def _validate_projection(projection: str) -> None:
    if projection not in SUPPORTED_PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}; expected one of {sorted(SUPPORTED_PROJECTIONS)}")
