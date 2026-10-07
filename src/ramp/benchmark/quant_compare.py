from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from keep.io.load import (
    infer_high_precision_switch_linear_dims,
    infer_vq_switch_linear_dims,
    inspect_safetensors,
)


_ARTIFACT_PROJECTION_RE = re.compile(r"^layer-(?P<layer>\d+)-(?P<projection>gate_proj|up_proj|down_proj)\.safetensors$")


_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
}


def _dtype_name(dtype: object) -> str:
    name = getattr(dtype, "name", str(dtype))
    return name.rsplit(".", maxsplit=1)[-1]


def tensor_nbytes(array: mx.array) -> int:
    count = 1
    for dim in array.shape:
        count *= int(dim)
    dtype_name = _dtype_name(array.dtype)
    if dtype_name not in _DTYPE_BYTES:
        raise ValueError(f"unsupported dtype for byte accounting: {array.dtype}")
    return count * _DTYPE_BYTES[dtype_name]


def mlx_affine_bit_support(
    *,
    bits: Iterable[int] = (1, 2, 3, 4),
    group_size: int = 64,
) -> dict[int, dict[str, Any]]:
    support: dict[int, dict[str, Any]] = {}
    for bit in bits:
        try:
            layer = nn.Linear(group_size, 16, bias=False)
            quantized = layer.to_quantized(group_size=group_size, bits=int(bit), mode="affine")
            mx.eval(quantized.parameters())
            support[int(bit)] = {"supported": True, "error": None}
        except Exception as exc:  # noqa: BLE001 - preflight records exact local failure.
            support[int(bit)] = {"supported": False, "error": str(exc)}
    return support


def validate_mlx_routed_group_size(
    group_size: int,
    *,
    hidden_size: int = 4096,
    moe_intermediate_size: int = 1408,
) -> dict[str, Any]:
    invalid = [
        dim
        for dim in (hidden_size, moe_intermediate_size)
        if dim % group_size != 0
    ]
    return {
        "group_size": int(group_size),
        "valid": not invalid,
        "invalid_dimensions": invalid,
    }


def quantized_parameter_bytes(params: Mapping[str, mx.array]) -> dict[str, int]:
    code_bytes = 0
    scale_bytes = 0
    bias_bytes = 0
    for name, value in params.items():
        size = tensor_nbytes(value)
        leaf = name.rsplit(".", maxsplit=1)[-1]
        if leaf in {"codes", "weight"}:
            code_bytes += size
        elif leaf == "scales":
            scale_bytes += size
        elif leaf in {"bias", "biases"}:
            bias_bytes += size
    return {
        "code_bytes": code_bytes,
        "scale_bytes": scale_bytes,
        "bias_bytes": bias_bytes,
        "total_quantized_bytes": code_bytes + scale_bytes + bias_bytes,
    }


def artifact_quantized_parameter_bytes(artifact_dir: str | Path) -> dict[str, int]:
    code_bytes = 0
    scale_bytes = 0
    bias_bytes = 0
    for path in sorted(Path(artifact_dir).glob("*.safetensors")):
        inspection = inspect_safetensors(path)
        for name, info in inspection.tensors.items():
            leaf = name.rsplit(".", maxsplit=1)[-1]
            if leaf in {"codes", "weight"}:
                code_bytes += info.nbytes
            elif leaf == "scales":
                scale_bytes += info.nbytes
            elif leaf in {"bias", "biases"}:
                bias_bytes += info.nbytes
    return {
        "code_bytes": code_bytes,
        "scale_bytes": scale_bytes,
        "bias_bytes": bias_bytes,
        "total_quantized_bytes": code_bytes + scale_bytes + bias_bytes,
    }


def _projection_prefix(tensor_names: Iterable[str], suffix: str) -> str:
    matches = [name[: -len(suffix)] for name in tensor_names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one tensor ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def audit_vq_artifact_prefill_compatibility(artifact_dir: str | Path) -> dict[str, Any]:
    """Inspect a GLM-4.5-Air VQ artifact for the resident auto-prefill fast path.

    The runtime chooses a NAX fast path when a layer's gate, up, and down switch
    projections are either all 8-bit E8 or all 16-bit E8P with 8-aligned group
    sizes. Mixed-bit layers intentionally fall back to Metal. This audit is
    metadata-only and follows symlinked shard files.
    """

    root = Path(artifact_dir)
    projection_rows: list[dict[str, Any]] = []
    code_bits_counts: dict[str, int] = {}
    group_size_counts: dict[str, int] = {}
    symlink_projection_count = 0
    high_precision_projection_count = 0

    for path in sorted(root.glob("layer-*-*.safetensors")):
        match = _ARTIFACT_PROJECTION_RE.match(path.name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        projection = match.group("projection")
        if path.is_symlink():
            symlink_projection_count += 1
        inspection = inspect_safetensors(path)
        tensor_names = tuple(inspection.tensors)
        row: dict[str, Any] = {
            "layer": layer,
            "projection": projection,
            "path": str(path),
            "is_symlink": path.is_symlink(),
            "resolved_path": str(path.resolve(strict=False)) if path.is_symlink() else str(path),
        }
        if any(name.endswith(".weight") for name in tensor_names):
            prefix = _projection_prefix(tensor_names, ".weight")
            in_dim, out_dim, num_experts = infer_high_precision_switch_linear_dims(inspection, prefix)
            row.update(
                {
                    "storage": "high_precision",
                    "input_dims": in_dim,
                    "output_dims": out_dim,
                    "num_experts": num_experts,
                    "group_size": None,
                    "code_bits": None,
                    "nax_e8_projection_compatible": False,
                    "nax_e8_projection_blocker": "high_precision_projection",
                    "nax_e8p_projection_compatible": False,
                    "nax_e8p_projection_blocker": "high_precision_projection",
                    "nax_fast_projection_compatible": False,
                    "nax_fast_projection_implementation": None,
                    "nax_fast_projection_blocker": "high_precision_projection",
                }
            )
            high_precision_projection_count += 1
        else:
            prefix = _projection_prefix(tensor_names, ".codes")
            in_dim, out_dim, num_experts, group_size, code_bits = infer_vq_switch_linear_dims(inspection, prefix)
            e8_compatible = code_bits == 8 and group_size % 8 == 0
            e8p_compatible = code_bits == 16 and group_size % 8 == 0
            e8_blocker = None
            if code_bits != 8:
                e8_blocker = f"code_bits={code_bits}"
            elif group_size % 8 != 0:
                e8_blocker = f"group_size={group_size}"
            e8p_blocker = None
            if code_bits != 16:
                e8p_blocker = f"code_bits={code_bits}"
            elif group_size % 8 != 0:
                e8p_blocker = f"group_size={group_size}"
            fast_impl = "nax_e8" if e8_compatible else "nax_e8p" if e8p_compatible else None
            if fast_impl is not None:
                fast_blocker = None
            elif code_bits in (8, 16) and group_size % 8 != 0:
                fast_blocker = f"group_size={group_size}"
            else:
                fast_blocker = f"code_bits={code_bits}"
            row.update(
                {
                    "storage": "vq",
                    "input_dims": in_dim,
                    "output_dims": out_dim,
                    "num_experts": num_experts,
                    "group_size": group_size,
                    "code_bits": code_bits,
                    "nax_e8_projection_compatible": e8_compatible,
                    "nax_e8_projection_blocker": e8_blocker,
                    "nax_e8p_projection_compatible": e8p_compatible,
                    "nax_e8p_projection_blocker": e8p_blocker,
                    "nax_fast_projection_compatible": fast_impl is not None,
                    "nax_fast_projection_implementation": fast_impl,
                    "nax_fast_projection_blocker": fast_blocker,
                }
            )
            code_bits_counts[str(code_bits)] = code_bits_counts.get(str(code_bits), 0) + 1
            group_size_counts[str(group_size)] = group_size_counts.get(str(group_size), 0) + 1
        projection_rows.append(row)

    rows_by_layer: dict[int, dict[str, dict[str, Any]]] = {}
    for row in projection_rows:
        rows_by_layer.setdefault(int(row["layer"]), {})[str(row["projection"])] = row

    layer_rows: list[dict[str, Any]] = []
    required = ("gate_proj", "up_proj", "down_proj")
    for layer in sorted(rows_by_layer):
        projections = rows_by_layer[layer]
        missing = [projection for projection in required if projection not in projections]
        e8_blockers = []
        e8p_blockers = []
        fast_blockers = []
        fast_implementations = set()
        for projection in required:
            row = projections.get(projection)
            if row is None:
                e8_blockers.append(f"{projection}:missing")
                e8p_blockers.append(f"{projection}:missing")
                fast_blockers.append(f"{projection}:missing")
                continue
            e8_blocker = row.get("nax_e8_projection_blocker")
            if e8_blocker is not None:
                e8_blockers.append(f"{projection}:{e8_blocker}")
            e8p_blocker = row.get("nax_e8p_projection_blocker")
            if e8p_blocker is not None:
                e8p_blockers.append(f"{projection}:{e8p_blocker}")
            fast_blocker = row.get("nax_fast_projection_blocker")
            if fast_blocker is not None:
                fast_blockers.append(f"{projection}:{fast_blocker}")
            fast_impl = row.get("nax_fast_projection_implementation")
            if fast_impl is not None:
                fast_implementations.add(str(fast_impl))
        fast_implementation = None
        if not fast_blockers and len(fast_implementations) == 1:
            fast_implementation = next(iter(fast_implementations))
        elif not fast_blockers:
            fast_blockers.append(
                "mixed_prefill_implementations:" + ",".join(sorted(fast_implementations))
            )
        layer_rows.append(
            {
                "layer": layer,
                "missing_projections": missing,
                "nax_e8_compatible": not e8_blockers,
                "nax_e8p_compatible": not e8p_blockers,
                "nax_fast_compatible": fast_implementation is not None,
                "auto_prefill_implementation": fast_implementation or "metal",
                "blockers": e8_blockers,
                "nax_e8_blockers": e8_blockers,
                "nax_e8p_blockers": e8p_blockers,
                "nax_fast_blockers": fast_blockers,
                "projections": {projection: projections.get(projection) for projection in required},
            }
        )

    compatible_layer_count = sum(1 for row in layer_rows if row["nax_e8_compatible"])
    e8p_compatible_layer_count = sum(1 for row in layer_rows if row["nax_e8p_compatible"])
    fast_compatible_layer_count = sum(1 for row in layer_rows if row["nax_fast_compatible"])
    manifest_path = root / "conversion-manifest.json"
    continuous_sidecar_count = 0
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            continuous = manifest.get("continuous_parameters")
            if isinstance(continuous, Mapping):
                sidecars = continuous.get("sidecars")
                continuous_sidecar_count = len(sidecars) if isinstance(sidecars, list) else 0
        except json.JSONDecodeError:
            continuous_sidecar_count = 0

    return {
        "record_type": "vq_artifact_prefill_compatibility_audit",
        "artifact_dir": str(root),
        "projection_count": len(projection_rows),
        "layer_count": len(layer_rows),
        "code_bits_counts": dict(sorted(code_bits_counts.items())),
        "group_size_counts": dict(sorted(group_size_counts.items())),
        "symlink_projection_count": symlink_projection_count,
        "high_precision_projection_count": high_precision_projection_count,
        "continuous_sidecar_count": continuous_sidecar_count,
        "nax_e8_compatible_layer_count": compatible_layer_count,
        "nax_e8p_compatible_layer_count": e8p_compatible_layer_count,
        "nax_fast_compatible_layer_count": fast_compatible_layer_count,
        "metal_fallback_layer_count": len(layer_rows) - fast_compatible_layer_count,
        "auto_prefill_all_layers_nax_e8_compatible": compatible_layer_count == len(layer_rows) and bool(layer_rows),
        "auto_prefill_all_layers_nax_e8p_compatible": e8p_compatible_layer_count == len(layer_rows) and bool(layer_rows),
        "auto_prefill_all_layers_nax_fast_compatible": fast_compatible_layer_count == len(layer_rows) and bool(layer_rows),
        "pinned_nax_e8_will_fail": any(
            row.get("storage") != "vq" or row.get("code_bits") != 8 for row in projection_rows
        ),
        "pinned_nax_e8p_will_fail": any(
            row.get("storage") != "vq" or row.get("code_bits") != 16 for row in projection_rows
        ),
        "projection_rows": projection_rows,
        "layer_rows": layer_rows,
    }


def glm45_air_routed_expert_weight_count(config: Mapping[str, Any]) -> int:
    sparse_layers = int(config["num_hidden_layers"]) - int(config.get("first_k_dense_replace", 0))
    experts = int(config["n_routed_experts"])
    hidden_size = int(config["hidden_size"])
    moe_intermediate_size = int(config["moe_intermediate_size"])
    per_expert = (
        hidden_size * moe_intermediate_size
        + hidden_size * moe_intermediate_size
        + moe_intermediate_size * hidden_size
    )
    return sparse_layers * experts * per_expert


def compute_effective_bits_per_weight(*, total_bytes: int, weights: int) -> float | None:
    if weights <= 0:
        return None
    return total_bytes * 8.0 / weights


def official_conversion_memory_preflight(
    *,
    dense_index_total_bytes: int,
    system_memory_bytes: int,
    safety_factor: float = 2.0,
) -> dict[str, Any]:
    required = int(dense_index_total_bytes * safety_factor)
    return {
        "dense_index_total_bytes": int(dense_index_total_bytes),
        "system_memory_bytes": int(system_memory_bytes),
        "safety_factor": float(safety_factor),
        "required_bytes": required,
        "allowed": required <= int(system_memory_bytes),
    }


def has_memory_pressure(record: Mapping[str, Any]) -> bool:
    return int(record.get("pageouts_delta") or 0) != 0 or int(record.get("swapouts_delta") or 0) != 0


def add_comparison_metadata(
    record: Mapping[str, Any],
    *,
    engine: str,
    quant_family: str,
    quantized_scope: str,
    artifact_dir: str,
    code_bytes: int,
    scale_bytes: int,
    bias_bytes: int,
    quantized_weights: int,
    mlx_bits: int | None = None,
    mlx_group_size: int | None = None,
    mlx_mode: str | None = None,
    vq_code_bits: int | None = None,
    vq_group_size: int | None = None,
    non_expert_dtype_policy: str = "source_bfloat16",
    non_expert_dtype_verified: bool = False,
    run_index: int = 1,
    repetition_count: int = 1,
    fresh_process: bool = True,
) -> dict[str, Any]:
    total = int(code_bytes) + int(scale_bytes) + int(bias_bytes)
    decorated = dict(record)
    decorated.update(
        {
            "engine": engine,
            "quant_family": quant_family,
            "quantized_scope": quantized_scope,
            "artifact_dir": artifact_dir,
            "mlx_bits": mlx_bits,
            "mlx_group_size": mlx_group_size,
            "mlx_mode": mlx_mode,
            "vq_code_bits": vq_code_bits,
            "vq_group_size": vq_group_size,
            "non_expert_dtype_policy": non_expert_dtype_policy,
            "non_expert_dtype_verified": non_expert_dtype_verified,
            "code_bytes": int(code_bytes),
            "scale_bytes": int(scale_bytes),
            "bias_bytes": int(bias_bytes),
            "total_quantized_bytes": total,
            "effective_bits_per_weight": compute_effective_bits_per_weight(
                total_bytes=total,
                weights=int(quantized_weights),
            ),
            "run_index": int(run_index),
            "repetition_count": int(repetition_count),
            "fresh_process": bool(fresh_process),
            "invalid_memory_pressure": has_memory_pressure(record),
        }
    )
    return decorated


def aggregate_repetition_records(
    records: Iterable[Mapping[str, Any]],
    *,
    timing_key: str,
    max_relative_spread: float = 0.5,
) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    valid = [record for record in rows if not has_memory_pressure(record)]
    values = [float(record[timing_key]) for record in valid if record.get(timing_key) is not None]
    invalid_memory_pressure = len(valid) != len(rows)
    invalid_repetition_reasons = [
        {
            "run_index": record.get("run_index"),
            "pageouts_delta": int(record.get("pageouts_delta") or 0),
            "swapouts_delta": int(record.get("swapouts_delta") or 0),
            "reason": "memory_pressure",
        }
        for record in rows
        if has_memory_pressure(record)
    ]
    if not values:
        return {
            "repetition_count": len(rows),
            "valid_repetition_count": 0,
            "invalid_repetition_count": len(rows),
            "invalid_repetition_reasons": invalid_repetition_reasons,
            "invalid_memory_pressure": invalid_memory_pressure,
            "timing_median_seconds": None,
            "timing_min_seconds": None,
            "timing_max_seconds": None,
            "timing_p90_seconds": None,
            "timing_stddev_seconds": None,
            "timing_relative_spread": None,
            "timing_stability_max_relative_spread": float(max_relative_spread),
            "timing_stable": False,
            "invalid_timing_instability": False,
        }
    median = float(statistics.median(values))
    min_value = float(min(values))
    max_value = float(max(values))
    p90_index = max(0, min(len(values) - 1, math.ceil(len(values) * 0.9) - 1))
    p90 = float(sorted(values)[p90_index])
    stddev = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    spread = None if median == 0 else (max_value - min_value) / median
    stable = spread is not None and spread <= float(max_relative_spread)
    return {
        "repetition_count": len(rows),
        "valid_repetition_count": len(values),
        "invalid_repetition_count": len(rows) - len(values),
        "invalid_repetition_reasons": invalid_repetition_reasons,
        "invalid_memory_pressure": invalid_memory_pressure,
        "timing_median_seconds": median,
        "timing_min_seconds": min_value,
        "timing_max_seconds": max_value,
        "timing_p90_seconds": p90,
        "timing_stddev_seconds": stddev,
        "timing_relative_spread": spread,
        "timing_stability_max_relative_spread": float(max_relative_spread),
        "timing_stable": stable,
        "invalid_timing_instability": not stable,
    }
