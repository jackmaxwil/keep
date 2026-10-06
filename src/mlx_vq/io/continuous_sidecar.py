from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx


CONTINUOUS_MANIFEST_KEY = "continuous_parameters"
CONTINUOUS_SIDECAR_DIR = "continuous_params"
CONTINUOUS_SCHEMA_VERSION = 1
SUPPORTED_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}


@dataclass(frozen=True)
class SwitchLinearSidecar:
    scale_delta: mx.array | None = None
    output_bias: mx.array | None = None
    low_rank_left: mx.array | None = None
    low_rank_right: mx.array | None = None
    enabled: bool = True


def continuous_sidecar_relpath(layer: int, projection: str) -> str:
    _validate_projection(projection)
    return f"{CONTINUOUS_SIDECAR_DIR}/layer-{layer:05d}-{projection}.safetensors"


def load_conversion_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(artifact_dir) / "conversion-manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return payload


def continuous_parameters_enabled(artifact_dir: str | Path) -> bool:
    config = load_conversion_manifest(artifact_dir).get(CONTINUOUS_MANIFEST_KEY)
    return isinstance(config, dict) and config.get("enabled") is True


def load_switch_linear_continuous_sidecar(
    artifact_dir: str | Path,
    *,
    layer: int,
    projection: str,
    scale_shape: tuple[int, ...],
    output_bias_shape: tuple[int, ...],
) -> SwitchLinearSidecar | None:
    """Load an optional switch-linear sidecar declared by conversion-manifest.json."""

    _validate_projection(projection)
    artifact_root = Path(artifact_dir)
    manifest = load_conversion_manifest(artifact_root)
    continuous = manifest.get(CONTINUOUS_MANIFEST_KEY)
    if not isinstance(continuous, dict) or continuous.get("enabled") is not True:
        return None
    if int(continuous.get("schema_version", 0)) != CONTINUOUS_SCHEMA_VERSION:
        raise ValueError(
            f"{CONTINUOUS_MANIFEST_KEY}.schema_version must be {CONTINUOUS_SCHEMA_VERSION}"
        )
    sidecar_entry = _find_sidecar_entry(continuous, layer=layer, projection=projection)
    if sidecar_entry is None:
        return None
    relpath = sidecar_entry.get("path")
    if not isinstance(relpath, str) or not relpath:
        raise ValueError(f"continuous sidecar entry for layer {layer} {projection} needs a path")
    sidecar_path = _resolve_artifact_relative_path(artifact_root, relpath)
    arrays = mx.load(str(sidecar_path))
    scale_delta = arrays.get("scale_delta")
    output_bias = arrays.get("output_bias")
    low_rank_left = arrays.get("low_rank_left")
    low_rank_right = arrays.get("low_rank_right")
    if scale_delta is not None and scale_delta.shape != scale_shape:
        raise ValueError(
            f"{sidecar_path} scale_delta must have shape {scale_shape}, found {scale_delta.shape}"
        )
    if output_bias is not None and output_bias.shape != output_bias_shape:
        raise ValueError(
            f"{sidecar_path} output_bias must have shape {output_bias_shape}, found {output_bias.shape}"
        )
    if low_rank_left is not None or low_rank_right is not None:
        if low_rank_left is None or low_rank_right is None:
            raise ValueError(f"{sidecar_path} low-rank sidecar requires both low_rank_left and low_rank_right")
        _validate_low_rank_shapes(
            low_rank_left,
            low_rank_right,
            output_bias_shape=output_bias_shape,
            input_dim=None,
        )
    if scale_delta is None and output_bias is None and low_rank_left is None:
        raise ValueError(f"{sidecar_path} must contain scale_delta, output_bias, or low-rank tensors")
    return SwitchLinearSidecar(
        scale_delta=scale_delta,
        output_bias=output_bias,
        low_rank_left=low_rank_left,
        low_rank_right=low_rank_right,
    )


def link_seed_artifact_groups(*, seed_artifact_dir: str | Path, output_dir: str | Path) -> int:
    """Symlink existing layer shards into a derived sidecar artifact directory."""

    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    linked_count = 0
    for source_path in sorted(seed_root.glob("layer-*.safetensors")):
        target_path = output_root / source_path.name
        if target_path.exists() or target_path.is_symlink():
            linked_count += 1
            continue
        os.symlink(_relative_symlink_target(source_path, target_path.parent), target_path)
        linked_count += 1
    return linked_count


def copy_declared_continuous_sidecars(
    *,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    exclude: set[tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    """Copy sidecars declared by a seed artifact into a derived artifact."""

    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    continuous = load_conversion_manifest(seed_root).get(CONTINUOUS_MANIFEST_KEY)
    if not isinstance(continuous, dict) or continuous.get("enabled") is not True:
        return []
    sidecars = continuous.get("sidecars", [])
    if not isinstance(sidecars, list):
        raise ValueError(f"{CONTINUOUS_MANIFEST_KEY}.sidecars must be a list")
    excluded = exclude or set()
    copied: list[dict[str, Any]] = []
    for item in sidecars:
        if not isinstance(item, dict):
            raise ValueError("continuous sidecar entries must be objects")
        layer = int(item.get("layer", -1))
        projection = str(item.get("projection", ""))
        _validate_projection(projection)
        if (layer, projection) in excluded:
            continue
        relpath = item.get("path")
        if not isinstance(relpath, str) or not relpath:
            raise ValueError(f"continuous sidecar entry for layer {layer} {projection} needs a path")
        source_path = _resolve_artifact_relative_path(seed_root, relpath)
        target_path = output_root / relpath
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied.append(dict(item))
    return copied


def write_continuous_sidecar(
    *,
    output_dir: str | Path,
    layer: int,
    projection: str,
    scale_delta: mx.array | None = None,
    output_bias: mx.array | None = None,
    low_rank_left: mx.array | None = None,
    low_rank_right: mx.array | None = None,
) -> dict[str, Any]:
    _validate_projection(projection)
    if scale_delta is None and output_bias is None and low_rank_left is None and low_rank_right is None:
        raise ValueError("sidecar must contain scale_delta, output_bias, or low-rank tensors")
    if (low_rank_left is None) != (low_rank_right is None):
        raise ValueError("low-rank sidecar requires both low_rank_left and low_rank_right")
    output_root = Path(output_dir)
    relpath = continuous_sidecar_relpath(layer, projection)
    sidecar_path = output_root / relpath
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, mx.array] = {}
    if scale_delta is not None:
        tensors["scale_delta"] = scale_delta.astype(mx.float32)
    if output_bias is not None:
        tensors["output_bias"] = output_bias.astype(mx.float32)
    if low_rank_left is not None and low_rank_right is not None:
        _validate_low_rank_shapes(
            low_rank_left,
            low_rank_right,
            output_bias_shape=None,
            input_dim=None,
        )
        tensors["low_rank_left"] = low_rank_left.astype(mx.float32)
        tensors["low_rank_right"] = low_rank_right.astype(mx.float32)
    mx.save_safetensors(
        str(sidecar_path),
        tensors,
        metadata={
            "continuous_sidecar_config": json.dumps(
                {
                    "schema_version": CONTINUOUS_SCHEMA_VERSION,
                    "layer": int(layer),
                    "projection": projection,
                    "tensors": sorted(tensors),
                },
                sort_keys=True,
            )
        },
    )
    entry: dict[str, Any] = {
        "layer": int(layer),
        "projection": projection,
        "path": relpath,
        "tensors": sorted(tensors),
    }
    if scale_delta is not None:
        entry["scale_delta_shape"] = [int(dim) for dim in scale_delta.shape]
    if output_bias is not None:
        entry["output_bias_shape"] = [int(dim) for dim in output_bias.shape]
    if low_rank_left is not None and low_rank_right is not None:
        entry["low_rank_left_shape"] = [int(dim) for dim in low_rank_left.shape]
        entry["low_rank_right_shape"] = [int(dim) for dim in low_rank_right.shape]
        entry["low_rank_rank"] = int(low_rank_left.shape[2])
    return entry


def write_continuous_artifact_manifest(
    *,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    sidecars: list[dict[str, Any]],
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    manifest = dict(load_conversion_manifest(seed_artifact_dir))
    continuous = {
        "schema_version": CONTINUOUS_SCHEMA_VERSION,
        "enabled": True,
        "format": "switch_linear_scale_delta_output_bias",
        "seed_artifact_dir": str(seed_artifact_dir),
        "sidecars": sidecars,
    }
    if run_manifest is not None:
        continuous["run"] = run_manifest
    manifest[CONTINUOUS_MANIFEST_KEY] = continuous
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _find_sidecar_entry(
    continuous: dict[str, Any],
    *,
    layer: int,
    projection: str,
) -> dict[str, Any] | None:
    sidecars = continuous.get("sidecars", [])
    if not isinstance(sidecars, list):
        raise ValueError(f"{CONTINUOUS_MANIFEST_KEY}.sidecars must be a list")
    for item in sidecars:
        if not isinstance(item, dict):
            raise ValueError("continuous sidecar entries must be objects")
        if int(item.get("layer", -1)) == int(layer) and item.get("projection") == projection:
            return item
    return None


def _resolve_artifact_relative_path(artifact_root: Path, relpath: str) -> Path:
    sidecar_path = Path(relpath)
    if sidecar_path.is_absolute():
        raise ValueError(f"continuous sidecar path must be artifact-relative, got {relpath!r}")
    root = artifact_root.resolve(strict=False)
    candidate = (root / sidecar_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"continuous sidecar path escapes artifact root: {relpath!r}") from error
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def _relative_symlink_target(source_path: Path, target_parent: Path) -> str:
    return os.path.relpath(source_path.resolve(), start=target_parent.resolve())


def _validate_projection(projection: str) -> None:
    if projection not in SUPPORTED_PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}; expected one of {sorted(SUPPORTED_PROJECTIONS)}")


def _validate_low_rank_shapes(
    low_rank_left: mx.array,
    low_rank_right: mx.array,
    *,
    output_bias_shape: tuple[int, ...] | None,
    input_dim: int | None,
) -> None:
    if low_rank_left.ndim != 3:
        raise ValueError(f"low_rank_left must be 3D [experts, out, rank], found {low_rank_left.shape}")
    if low_rank_right.ndim != 3:
        raise ValueError(f"low_rank_right must be 3D [experts, rank, in], found {low_rank_right.shape}")
    experts, out_dim, rank = low_rank_left.shape
    right_experts, right_rank, right_input = low_rank_right.shape
    if rank <= 0:
        raise ValueError("low-rank rank must be positive")
    if (right_experts, right_rank) != (experts, rank):
        raise ValueError(
            "low_rank_right must have shape [experts, rank, in] matching low_rank_left; "
            f"found {low_rank_right.shape} for left {low_rank_left.shape}"
        )
    if output_bias_shape is not None and (experts, out_dim) != output_bias_shape:
        raise ValueError(
            f"low_rank_left leading dimensions must match output shape {output_bias_shape}, "
            f"found {(experts, out_dim)}"
        )
    if input_dim is not None and right_input != input_dim:
        raise ValueError(f"low_rank_right input dimension must be {input_dim}, found {right_input}")
