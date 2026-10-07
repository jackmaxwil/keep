from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx


ROUTER_CORRECTION_MANIFEST_KEY = "router_corrections"
ROUTER_CORRECTION_DIR = "router_corrections"
ROUTER_CORRECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RouterCorrectionSidecar:
    expert_bias_delta: mx.array | None = None
    temperature: mx.array | None = None


def router_correction_relpath(layer: int) -> str:
    return f"{ROUTER_CORRECTION_DIR}/layer-{layer:05d}.safetensors"


def load_conversion_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(artifact_dir) / "conversion-manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return payload


def router_corrections_enabled(artifact_dir: str | Path) -> bool:
    config = load_conversion_manifest(artifact_dir).get(ROUTER_CORRECTION_MANIFEST_KEY)
    return isinstance(config, dict) and config.get("enabled") is True


def load_router_correction_sidecar(
    artifact_dir: str | Path,
    *,
    layer: int,
    num_experts: int,
) -> RouterCorrectionSidecar | None:
    artifact_root = Path(artifact_dir)
    manifest = load_conversion_manifest(artifact_root)
    corrections = manifest.get(ROUTER_CORRECTION_MANIFEST_KEY)
    if not isinstance(corrections, dict) or corrections.get("enabled") is not True:
        return None
    if int(corrections.get("schema_version", 0)) != ROUTER_CORRECTION_SCHEMA_VERSION:
        raise ValueError(
            f"{ROUTER_CORRECTION_MANIFEST_KEY}.schema_version must be "
            f"{ROUTER_CORRECTION_SCHEMA_VERSION}"
        )
    entry = _find_router_entry(corrections, layer=layer)
    if entry is None:
        return None
    relpath = entry.get("path")
    if not isinstance(relpath, str) or not relpath:
        raise ValueError(f"router correction entry for layer {layer} needs a path")
    sidecar_path = _resolve_artifact_relative_path(artifact_root, relpath)
    arrays = mx.load(str(sidecar_path))
    expert_bias_delta = arrays.get("expert_bias_delta")
    temperature = arrays.get("temperature")
    _validate_router_tensors(
        expert_bias_delta=expert_bias_delta,
        temperature=temperature,
        num_experts=num_experts,
        path=sidecar_path,
    )
    if expert_bias_delta is None and temperature is None:
        raise ValueError(f"{sidecar_path} must contain expert_bias_delta or temperature")
    return RouterCorrectionSidecar(
        expert_bias_delta=expert_bias_delta,
        temperature=temperature,
    )


def write_router_correction_sidecar(
    *,
    output_dir: str | Path,
    layer: int,
    num_experts: int,
    expert_bias_delta: mx.array | None = None,
    temperature: mx.array | None = None,
) -> dict[str, Any]:
    _validate_router_tensors(
        expert_bias_delta=expert_bias_delta,
        temperature=temperature,
        num_experts=num_experts,
        path=Path(output_dir) / router_correction_relpath(layer),
    )
    if expert_bias_delta is None and temperature is None:
        raise ValueError("router correction must contain expert_bias_delta or temperature")

    output_root = Path(output_dir)
    relpath = router_correction_relpath(layer)
    sidecar_path = output_root / relpath
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, mx.array] = {}
    if expert_bias_delta is not None:
        tensors["expert_bias_delta"] = expert_bias_delta.astype(mx.float32)
    if temperature is not None:
        tensors["temperature"] = temperature.astype(mx.float32)
    mx.save_safetensors(
        str(sidecar_path),
        tensors,
        metadata={
            "router_correction_config": json.dumps(
                {
                    "schema_version": ROUTER_CORRECTION_SCHEMA_VERSION,
                    "layer": int(layer),
                    "num_experts": int(num_experts),
                    "tensors": sorted(tensors),
                },
                sort_keys=True,
            )
        },
    )
    return {
        "layer": int(layer),
        "path": relpath,
        "format": "expert_bias_delta_temperature",
        "num_experts": int(num_experts),
        "tensors": sorted(tensors),
    }


def write_router_correction_artifact_manifest(
    *,
    seed_manifest: dict[str, Any],
    output_dir: str | Path,
    corrections: list[dict[str, Any]],
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = dict(seed_manifest)
    payload: dict[str, Any] = {
        "schema_version": ROUTER_CORRECTION_SCHEMA_VERSION,
        "enabled": True,
        "format": "moe_router_expert_bias_temperature",
        "corrections": corrections,
    }
    if run_manifest is not None:
        payload["run"] = run_manifest
    manifest[ROUTER_CORRECTION_MANIFEST_KEY] = payload
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _find_router_entry(corrections: dict[str, Any], *, layer: int) -> dict[str, Any] | None:
    entries = corrections.get("corrections", [])
    if not isinstance(entries, list):
        raise ValueError(f"{ROUTER_CORRECTION_MANIFEST_KEY}.corrections must be a list")
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("router correction entries must be objects")
        if int(item.get("layer", -1)) == int(layer):
            return item
    return None


def _validate_router_tensors(
    *,
    expert_bias_delta: mx.array | None,
    temperature: mx.array | None,
    num_experts: int,
    path: Path,
) -> None:
    if expert_bias_delta is not None and expert_bias_delta.shape != (num_experts,):
        raise ValueError(
            f"{path} expert_bias_delta must have shape ({num_experts},), "
            f"found {expert_bias_delta.shape}"
        )
    if temperature is not None:
        if temperature.shape not in {(), (1,)}:
            raise ValueError(f"{path} temperature must be scalar or shape (1,), found {temperature.shape}")
        value = float(temperature.item())
        if not value > 0.0:
            raise ValueError(f"{path} temperature must be positive")


def _resolve_artifact_relative_path(artifact_root: Path, relpath: str) -> Path:
    sidecar_path = Path(relpath)
    if sidecar_path.is_absolute():
        raise ValueError(f"router correction path must be artifact-relative, got {relpath!r}")
    root = artifact_root.resolve(strict=False)
    candidate = (root / sidecar_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"router correction path escapes artifact root: {relpath!r}") from error
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate
