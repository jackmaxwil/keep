from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import numpy as np

from keep.vq.e8 import e8_1bit_packed, e8p_packed_abs_grid
from keep.convert.stream_convert import _read_tensor_from_shard
from keep.io.schema import QuantizationConfig, codebook_metadata_for_bits
from keep.quant.rtn import ScaleEstimator, quantize_weight_rtn


QWEN36_35B_A3B_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
QWEN36_35B_A3B_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
QWEN36_35B_A3B_MODEL_TYPE = "qwen3_5_moe"

QWEN36_35B_A3B_ARCHITECTURE: dict[str, object] = {
    "source": "Qwen/Qwen3.6-35B-A3B Hugging Face model card checked 2026-07-02",
    "total_parameters": 35_000_000_000,
    "active_parameters": 3_000_000_000,
    "language_layers": 40,
    "hidden_size": 2048,
    "experts": 256,
    "routed_experts_per_token": 8,
    "shared_experts": 1,
    "expert_intermediate_size": 512,
    "native_context": 262_144,
    "max_extended_context": 1_010_000,
    "model_type": QWEN36_35B_A3B_MODEL_TYPE,
}

QWEN_MOE_EXPECTED_LANGUAGE_PROJECTIONS = ("down_proj", "gate_up_proj")
QWEN_MOE_SOURCE_GROUP_ORDER = ("gate_up_proj", "down_proj")

_LANGUAGE_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<projection>[^.]+)$"
)
_AUXILIARY_EXPERT_RE = re.compile(
    r"^(?P<prefix>mtp)\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<projection>[^.]+)$"
)
_LANGUAGE_LAYER_NON_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.(?P<module>[^.]+)\."
)


@dataclass(frozen=True)
class QwenMoeConversionGroup:
    layer: int
    source_projection: str
    source_tensor: str
    source_shards: tuple[str, ...]
    target_projections: tuple[str, ...]
    experts: int
    input_dims: int
    output_dims: int
    logical_output_dims: tuple[int, ...]
    expected_source_shape: tuple[int, int, int]
    source_shape: tuple[int, ...] | None
    source_dtype: str | None
    source_parameter_count: int | None
    source_shape_matches: bool | None
    code_bits: int
    group_size: int

    @property
    def code_bytes(self) -> int:
        return self.experts * self.output_dims * (self.input_dims // 8) * (
            self.code_bits // 8
        )

    @property
    def scale_bytes(self) -> int:
        return self.experts * self.output_dims * (self.input_dims // self.group_size) * 2

    @property
    def dense_source_bytes(self) -> int:
        return self.experts * self.output_dims * self.input_dims * 2

    def to_json_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "source_projection": self.source_projection,
            "source_tensor": self.source_tensor,
            "source_shards": list(self.source_shards),
            "target_projections": list(self.target_projections),
            "experts": self.experts,
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "logical_output_dims": list(self.logical_output_dims),
            "expected_source_shape": list(self.expected_source_shape),
            "source_shape": None if self.source_shape is None else list(self.source_shape),
            "source_dtype": self.source_dtype,
            "source_parameter_count": self.source_parameter_count,
            "source_shape_matches": self.source_shape_matches,
            "code_bits": self.code_bits,
            "group_size": self.group_size,
            "code_bytes": self.code_bytes,
            "scale_bytes": self.scale_bytes,
            "dense_source_bytes": self.dense_source_bytes,
        }


@dataclass(frozen=True)
class ConvertedQwenMoeGroup:
    output_path: Path
    source_tensor: str
    target_projections: tuple[str, ...]
    codes_names: Mapping[str, str]
    scales_names: Mapping[str, str]
    codes_shapes: Mapping[str, tuple[int, int, int]]
    scales_shapes: Mapping[str, tuple[int, int, int]]
    source_tensors_read: int
    peak_source_tensor_bytes: int
    code_bits: int
    group_size: int
    scale_estimator: ScaleEstimator = "max_abs"
    elapsed_seconds: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "source_tensor": self.source_tensor,
            "target_projections": list(self.target_projections),
            "codes_names": dict(self.codes_names),
            "scales_names": dict(self.scales_names),
            "codes_shapes": {
                projection: list(shape) for projection, shape in self.codes_shapes.items()
            },
            "scales_shapes": {
                projection: list(shape) for projection, shape in self.scales_shapes.items()
            },
            "source_tensors_read": self.source_tensors_read,
            "peak_source_tensor_bytes": self.peak_source_tensor_bytes,
            "code_bits": self.code_bits,
            "group_size": self.group_size,
            "scale_estimator": self.scale_estimator,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class QwenMoeMaterializationManifest:
    output_dir: Path
    manifest_path: Path
    model_id: str
    revision: str
    materialization_status: str
    planned_groups: int
    converted_groups: int
    skipped_existing_groups: int
    source_tensors_read: int
    peak_source_tensor_bytes: int
    code_bits: int
    group_size: int
    scale_estimator: ScaleEstimator
    groups: tuple[Mapping[str, object], ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "model_id": self.model_id,
            "revision": self.revision,
            "materialization_status": self.materialization_status,
            "planned_groups": self.planned_groups,
            "converted_groups": self.converted_groups,
            "skipped_existing_groups": self.skipped_existing_groups,
            "source_tensors_read": self.source_tensors_read,
            "peak_source_tensor_bytes": self.peak_source_tensor_bytes,
            "code_bits": self.code_bits,
            "group_size": self.group_size,
            "scale_estimator": self.scale_estimator,
            "groups": [dict(group) for group in self.groups],
        }


@dataclass(frozen=True)
class QwenMoeConversionPlan:
    model_id: str
    revision: str
    groups: tuple[QwenMoeConversionGroup, ...]
    missing_language_projection_pairs: tuple[str, ...]
    source_shape_checks_present: bool
    conversion_status: str
    conversion_blockers: tuple[str, ...]

    @property
    def source_projection_groups(self) -> int:
        return len(self.groups)

    @property
    def target_projection_groups(self) -> int:
        return sum(len(group.target_projections) for group in self.groups)

    @property
    def vq_code_bytes(self) -> int:
        return sum(group.code_bytes for group in self.groups)

    @property
    def vq_scale_bytes(self) -> int:
        return sum(group.scale_bytes for group in self.groups)

    @property
    def peak_source_projection_bytes(self) -> int:
        return max((group.dense_source_bytes for group in self.groups), default=0)

    @property
    def missing_source_shapes(self) -> tuple[str, ...]:
        if not self.source_shape_checks_present:
            return ()
        return tuple(group.source_tensor for group in self.groups if group.source_shape is None)

    @property
    def mismatched_source_shapes(self) -> tuple[str, ...]:
        if not self.source_shape_checks_present:
            return ()
        mismatches: list[str] = []
        for group in self.groups:
            if group.source_shape is None or group.source_shape_matches is not False:
                continue
            mismatches.append(
                f"{group.source_tensor}: expected {list(group.expected_source_shape)}, "
                f"found {list(group.source_shape)}"
            )
        return tuple(mismatches)

    @property
    def source_shapes_verified(self) -> bool:
        if not self.source_shape_checks_present:
            return False
        return not self.missing_source_shapes and not self.mismatched_source_shapes

    def to_json_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "source_projection_groups": self.source_projection_groups,
            "target_projection_groups": self.target_projection_groups,
            "missing_language_projection_pairs": list(
                self.missing_language_projection_pairs
            ),
            "vq_code_bytes": self.vq_code_bytes,
            "vq_scale_bytes": self.vq_scale_bytes,
            "peak_source_projection_bytes": self.peak_source_projection_bytes,
            "source_shape_checks_present": self.source_shape_checks_present,
            "source_shapes_verified": self.source_shapes_verified,
            "missing_source_shapes": list(self.missing_source_shapes),
            "mismatched_source_shapes": list(self.mismatched_source_shapes),
            "conversion_status": self.conversion_status,
            "conversion_blockers": list(self.conversion_blockers),
            "groups": [group.to_json_dict() for group in self.groups],
        }


@dataclass(frozen=True)
class QwenMoeSourceAudit:
    model_id: str
    revision: str
    model_type: str | None
    source_index_tensors: int
    source_shards: int
    expert_tensors_total: int
    language_expert_tensors: int
    auxiliary_expert_tensors: int
    non_expert_tensors_total: int
    non_expert_tensors_by_family: Mapping[str, int]
    language_non_expert_tensors_by_module: Mapping[str, int]
    language_non_expert_layers_by_module: Mapping[str, tuple[int, ...]]
    language_sparse_layers: int
    language_layer_range: tuple[int, int] | None
    projections_per_language_layer: tuple[tuple[str, ...], ...]
    missing_language_projection_pairs: tuple[str, ...]
    unexpected_language_projections: tuple[str, ...]
    has_fused_gate_up_proj: bool
    has_unfused_gate_or_up_proj: bool
    expert_tensor_shape: Mapping[str, str]
    checks: Mapping[str, bool]
    conversion_status: str
    conversion_blockers: tuple[str, ...]
    architecture: Mapping[str, object]
    conversion_plan: QwenMoeConversionPlan | None = None

    @property
    def source_checks_pass(self) -> bool:
        return all(self.checks.values()) and (
            self.conversion_plan is None
            or not self.conversion_plan.source_shape_checks_present
            or self.conversion_plan.source_shapes_verified
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "model_type": self.model_type,
            "source_index_tensors": self.source_index_tensors,
            "source_shards": self.source_shards,
            "expert_tensors_total": self.expert_tensors_total,
            "language_expert_tensors": self.language_expert_tensors,
            "auxiliary_expert_tensors": self.auxiliary_expert_tensors,
            "non_expert_tensors_total": self.non_expert_tensors_total,
            "non_expert_tensors_by_family": dict(self.non_expert_tensors_by_family),
            "language_non_expert_tensors_by_module": dict(
                self.language_non_expert_tensors_by_module
            ),
            "language_non_expert_layers_by_module": {
                module: list(layers)
                for module, layers in self.language_non_expert_layers_by_module.items()
            },
            "language_sparse_layers": self.language_sparse_layers,
            "language_layer_range": (
                None if self.language_layer_range is None else list(self.language_layer_range)
            ),
            "projections_per_language_layer": [
                list(projections) for projections in self.projections_per_language_layer
            ],
            "missing_language_projection_pairs": list(self.missing_language_projection_pairs),
            "unexpected_language_projections": list(self.unexpected_language_projections),
            "has_fused_gate_up_proj": self.has_fused_gate_up_proj,
            "has_unfused_gate_or_up_proj": self.has_unfused_gate_or_up_proj,
            "expert_tensor_shape": dict(self.expert_tensor_shape),
            "checks": dict(self.checks),
            "source_checks_pass": self.source_checks_pass,
            "conversion_status": self.conversion_status,
            "conversion_blockers": list(self.conversion_blockers),
            "architecture": dict(self.architecture),
            "conversion_plan": (
                None if self.conversion_plan is None else self.conversion_plan.to_json_dict()
            ),
        }


def _weight_map_from_index(index: Mapping[str, Any]) -> Mapping[str, str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError("safetensors index must contain a weight_map mapping")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def _metadata_shape(entry: Any) -> tuple[int, ...] | None:
    value = entry.get("shape") if isinstance(entry, Mapping) else getattr(entry, "shape", None)
    if value is None:
        return None
    return tuple(int(dim) for dim in value)


def _metadata_dtype(entry: Any) -> str | None:
    value = entry.get("dtype") if isinstance(entry, Mapping) else getattr(entry, "dtype", None)
    return None if value is None else str(value)


def _metadata_parameter_count(entry: Any) -> int | None:
    value = (
        entry.get("parameter_count")
        if isinstance(entry, Mapping)
        else getattr(entry, "parameter_count", None)
    )
    return None if value is None else int(value)


def _language_expert_tensors(
    weight_map: Mapping[str, str],
) -> dict[int, dict[str, tuple[str, str]]]:
    by_layer: dict[int, dict[str, tuple[str, str]]] = {}
    for tensor_name, shard in weight_map.items():
        language_match = _LANGUAGE_EXPERT_RE.match(tensor_name)
        if language_match is None:
            continue
        layer = int(language_match.group("layer"))
        projection = language_match.group("projection")
        by_layer.setdefault(layer, {})[projection] = (tensor_name, shard)
    return by_layer


def _qwen_non_expert_inventory(
    weight_map: Mapping[str, str],
) -> tuple[int, dict[str, int], dict[str, int], dict[str, tuple[int, ...]]]:
    by_family: dict[str, int] = {}
    language_by_module: dict[str, int] = {}
    language_layers_by_module: dict[str, set[int]] = {}

    def bump(mapping: dict[str, int], key: str) -> None:
        mapping[key] = mapping.get(key, 0) + 1

    for tensor_name in weight_map:
        if (
            _LANGUAGE_EXPERT_RE.match(tensor_name) is not None
            or _AUXILIARY_EXPERT_RE.match(tensor_name) is not None
        ):
            continue

        if tensor_name.startswith("model.language_model."):
            family = "language_model"
            layer_match = _LANGUAGE_LAYER_NON_EXPERT_RE.match(tensor_name)
            if layer_match is not None:
                module = layer_match.group("module")
                layer = int(layer_match.group("layer"))
                language_layers_by_module.setdefault(module, set()).add(layer)
            elif tensor_name.startswith("model.language_model.embed_tokens."):
                module = "embed_tokens"
            elif tensor_name.startswith("model.language_model.norm."):
                module = "norm"
            else:
                module = "other"
            bump(language_by_module, module)
        elif tensor_name.startswith("model.visual."):
            family = "visual"
        elif tensor_name.startswith("mtp."):
            family = "mtp"
        elif tensor_name.startswith("lm_head."):
            family = "lm_head"
        else:
            family = "other"

        bump(by_family, family)

    layers_by_module = {
        module: tuple(sorted(layers))
        for module, layers in sorted(language_layers_by_module.items())
    }
    return (
        sum(by_family.values()),
        dict(sorted(by_family.items())),
        dict(sorted(language_by_module.items())),
        layers_by_module,
    )


def _projection_contract(source_projection: str) -> tuple[tuple[str, ...], int, int, tuple[int, ...]]:
    hidden_size = int(QWEN36_35B_A3B_ARCHITECTURE["hidden_size"])
    intermediate_size = int(QWEN36_35B_A3B_ARCHITECTURE["expert_intermediate_size"])
    if source_projection == "gate_up_proj":
        return (
            ("gate_proj", "up_proj"),
            hidden_size,
            intermediate_size * 2,
            (intermediate_size, intermediate_size),
        )
    if source_projection == "down_proj":
        return (("down_proj",), intermediate_size, hidden_size, (hidden_size,))
    raise ValueError(f"unknown Qwen MoE source projection {source_projection!r}")


def _effective_group_size(input_dims: int, preferred_group_size: int) -> int:
    if input_dims % preferred_group_size == 0:
        return preferred_group_size
    limit = min(input_dims, preferred_group_size)
    for candidate in range(limit, 0, -8):
        if input_dims % candidate == 0:
            return candidate
    raise AssertionError("Qwen MoE projection dims should be 8D aligned")


def plan_qwen_moe_conversion_from_index(
    index: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
    model_id: str,
    revision: str,
    expected_language_layers: int | None = None,
    code_bits: int = 8,
    group_size: int = 512,
    tensor_metadata: Mapping[str, Any] | None = None,
) -> QwenMoeConversionPlan:
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    weight_map = _weight_map_from_index(index)
    by_layer = _language_expert_tensors(weight_map)
    expected_layer_ids = (
        range(expected_language_layers)
        if expected_language_layers is not None
        else sorted(by_layer)
    )
    experts = int(QWEN36_35B_A3B_ARCHITECTURE["experts"])
    groups: list[QwenMoeConversionGroup] = []
    missing_pairs: list[str] = []

    for layer in expected_layer_ids:
        projections = by_layer.get(layer, {})
        for source_projection in QWEN_MOE_SOURCE_GROUP_ORDER:
            entry = projections.get(source_projection)
            if entry is None:
                missing_pairs.append(f"{layer}:{source_projection}")
                continue
            source_tensor, shard = entry
            target_projections, input_dims, output_dims, logical_dims = (
                _projection_contract(source_projection)
            )
            expected_source_shape = (experts, output_dims, input_dims)
            metadata = None if tensor_metadata is None else tensor_metadata.get(source_tensor)
            source_shape = None if metadata is None else _metadata_shape(metadata)
            groups.append(
                QwenMoeConversionGroup(
                    layer=layer,
                    source_projection=source_projection,
                    source_tensor=source_tensor,
                    source_shards=(shard,),
                    target_projections=target_projections,
                    experts=experts,
                    input_dims=input_dims,
                    output_dims=output_dims,
                    logical_output_dims=logical_dims,
                    expected_source_shape=expected_source_shape,
                    source_shape=source_shape,
                    source_dtype=None if metadata is None else _metadata_dtype(metadata),
                    source_parameter_count=(
                        None if metadata is None else _metadata_parameter_count(metadata)
                    ),
                    source_shape_matches=(
                        None if tensor_metadata is None else source_shape == expected_source_shape
                    ),
                    code_bits=code_bits,
                    group_size=_effective_group_size(input_dims, group_size),
                )
            )

    return QwenMoeConversionPlan(
        model_id=model_id,
        revision=revision,
        groups=tuple(groups),
        missing_language_projection_pairs=tuple(missing_pairs),
        source_shape_checks_present=tensor_metadata is not None,
        conversion_status="qwen_moe_plan_ready__materializer_available",
        conversion_blockers=(
            "qwen3_5_moe_runtime_adapter",
            "family_specific_eval_and_benchmark_gates",
        ),
    )


def qwen_moe_target_prefix(group: QwenMoeConversionGroup, projection: str) -> str:
    if projection not in group.target_projections:
        raise ValueError(
            f"projection {projection!r} is not a target for source {group.source_projection!r}"
        )
    return f"model.language_model.layers.{group.layer}.mlp.switch_mlp.{projection}"


def qwen_moe_group_output_filename(group: QwenMoeConversionGroup) -> str:
    return f"layer-{group.layer:05d}-{group.source_projection}.safetensors"


def audit_qwen_moe_source_payloads(
    *,
    source_dir: str | Path,
    plan: QwenMoeConversionPlan,
    max_groups: int | None = None,
) -> dict[str, object]:
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive when provided")

    source_root = Path(source_dir)
    selected_groups = plan.groups[:max_groups]
    required_shards: list[str] = []
    group_records: list[dict[str, object]] = []
    for group in selected_groups:
        missing_for_group = [
            shard for shard in group.source_shards if not (source_root / shard).is_file()
        ]
        for shard in group.source_shards:
            if shard not in required_shards:
                required_shards.append(shard)
        group_records.append(
            {
                "layer": group.layer,
                "source_projection": group.source_projection,
                "source_tensor": group.source_tensor,
                "source_shards": list(group.source_shards),
                "missing_shards": missing_for_group,
                "payload_ready": not missing_for_group,
            }
        )

    present_shards = [
        shard for shard in required_shards if (source_root / shard).is_file()
    ]
    missing_shards = [
        shard for shard in required_shards if not (source_root / shard).is_file()
    ]
    materialization_blocked = bool(missing_shards)
    return {
        "record_type": "qwen_moe_source_payload_audit",
        "model_id": plan.model_id,
        "revision": plan.revision,
        "source_dir": str(source_root),
        "planned_groups": len(selected_groups),
        "required_shards": required_shards,
        "present_shards": present_shards,
        "missing_shards": missing_shards,
        "payload_status": (
            "qwen_moe_source_payloads_missing"
            if materialization_blocked
            else "qwen_moe_source_payloads_ready"
        ),
        "materialization_blocked": materialization_blocked,
        "materialization_blockers": (
            [
                "download_payload_shards",
                "rerun_with_payload_bearing_source_dir",
            ]
            if materialization_blocked
            else []
        ),
        "groups": group_records,
    }


def _validate_qwen_group_for_materialization(group: QwenMoeConversionGroup) -> None:
    if group.code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if group.group_size <= 0:
        raise ValueError("group_size must be positive")
    if group.input_dims % 8 != 0:
        raise ValueError("input_dims must be divisible by 8")
    if group.input_dims % group.group_size != 0:
        raise ValueError("input_dims must be divisible by group_size")
    if sum(group.logical_output_dims) != group.output_dims:
        raise ValueError(
            f"logical output dims {group.logical_output_dims} do not sum to {group.output_dims}"
        )
    if len(group.target_projections) != len(group.logical_output_dims):
        raise ValueError("target_projections and logical_output_dims must have the same length")
    if len(group.source_shards) != 1:
        raise ValueError("Qwen MoE fused source groups must live in exactly one source shard")


def _quantize_stacked_experts(
    values: np.ndarray,
    *,
    group_size: int,
    code_bits: int,
    scale_estimator: ScaleEstimator,
) -> tuple[np.ndarray, np.ndarray]:
    codes = []
    scales = []
    for expert in range(values.shape[0]):
        quantized = quantize_weight_rtn(
            values[expert],
            group_size=group_size,
            code_bits=code_bits,
            scale_estimator=scale_estimator,
        )
        codes.append(quantized.codes)
        scales.append(quantized.scales)
    return np.stack(codes, axis=0), np.stack(scales, axis=0)


def convert_qwen_moe_group_from_safetensors(
    *,
    source_dir: str | Path,
    group: QwenMoeConversionGroup,
    output_path: str | Path,
    scale_estimator: ScaleEstimator = "max_abs",
) -> ConvertedQwenMoeGroup:
    _validate_qwen_group_for_materialization(group)
    source_root = Path(source_dir)
    output = Path(output_path)
    start = time.perf_counter()
    source = _read_tensor_from_shard(source_root, group.source_shards[0], group.source_tensor)
    if source.shape != group.expected_source_shape:
        raise ValueError(
            f"{group.source_tensor} must have shape {group.expected_source_shape}, found {source.shape}"
        )

    arrays: dict[str, mx.array] = {}
    codes_names: dict[str, str] = {}
    scales_names: dict[str, str] = {}
    codes_shapes: dict[str, tuple[int, int, int]] = {}
    scales_shapes: dict[str, tuple[int, int, int]] = {}
    offset = 0
    for projection, output_dim in zip(group.target_projections, group.logical_output_dims):
        chunk = source[:, offset : offset + output_dim, :]
        offset += output_dim
        codes, scales = _quantize_stacked_experts(
            chunk,
            group_size=group.group_size,
            code_bits=group.code_bits,
            scale_estimator=scale_estimator,
        )
        prefix = qwen_moe_target_prefix(group, projection)
        codes_name = f"{prefix}.codes"
        scales_name = f"{prefix}.scales"
        arrays[codes_name] = mx.array(codes)
        arrays[scales_name] = mx.array(scales)
        codes_names[projection] = codes_name
        scales_names[projection] = scales_name
        codes_shapes[projection] = tuple(int(dim) for dim in codes.shape)
        scales_shapes[projection] = tuple(int(dim) for dim in scales.shape)

    codebook_name, codebook_sha256 = codebook_metadata_for_bits(group.code_bits)
    codebook = e8_1bit_packed() if group.code_bits == 8 else e8p_packed_abs_grid()
    arrays["model.vq_codebook.e8"] = mx.array(codebook)
    metadata = {
        "quantization_config": json.dumps(
            QuantizationConfig(
                default_code_bits=group.code_bits,
                default_group_size=group.group_size,
                codebook_name=codebook_name,
                codebook_sha256=codebook_sha256,
                policy={
                    "source_model_family": "qwen3_5_moe",
                    "source_projection": group.source_projection,
                    "scale_estimator": scale_estimator,
                },
            ).to_json_dict()
        )
    }
    mx.save_safetensors(str(output), arrays, metadata=metadata)
    return ConvertedQwenMoeGroup(
        output_path=output,
        source_tensor=group.source_tensor,
        target_projections=group.target_projections,
        codes_names=codes_names,
        scales_names=scales_names,
        codes_shapes=codes_shapes,
        scales_shapes=scales_shapes,
        source_tensors_read=1,
        peak_source_tensor_bytes=source.nbytes,
        code_bits=group.code_bits,
        group_size=group.group_size,
        scale_estimator=scale_estimator,
        elapsed_seconds=time.perf_counter() - start,
    )


def convert_qwen_moe_groups_from_safetensors(
    *,
    source_dir: str | Path,
    plan: QwenMoeConversionPlan,
    output_dir: str | Path,
    max_groups: int | None = None,
    skip_existing: bool = False,
    overwrite: bool = False,
    scale_estimator: ScaleEstimator = "max_abs",
) -> QwenMoeMaterializationManifest:
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive when provided")
    if skip_existing and overwrite:
        raise ValueError("skip_existing and overwrite cannot both be true")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected_groups = plan.groups[:max_groups]
    records: list[Mapping[str, object]] = []
    converted_count = 0
    skipped_count = 0
    source_tensors_read = 0
    peak_source_tensor_bytes = 0

    for group in selected_groups:
        output_path = destination / qwen_moe_group_output_filename(group)
        if output_path.exists():
            if skip_existing:
                skipped_count += 1
                records.append(
                    {
                        "status": "skipped_existing",
                        "output_path": str(output_path),
                        "source_tensor": group.source_tensor,
                        "source_projection": group.source_projection,
                        "target_projections": list(group.target_projections),
                        "layer": group.layer,
                    }
                )
                continue
            if not overwrite:
                raise FileExistsError(f"{output_path} already exists")
            output_path.unlink()

        converted = convert_qwen_moe_group_from_safetensors(
            source_dir=source_dir,
            group=group,
            output_path=output_path,
            scale_estimator=scale_estimator,
        )
        converted_count += 1
        source_tensors_read += converted.source_tensors_read
        peak_source_tensor_bytes = max(
            peak_source_tensor_bytes,
            converted.peak_source_tensor_bytes,
        )
        record = converted.to_json_dict()
        record.update(
            {
                "status": "converted",
                "layer": group.layer,
                "source_projection": group.source_projection,
            }
        )
        records.append(record)

    manifest_path = destination / "qwen-moe-materialization-manifest.json"
    status = (
        "qwen_moe_groups_materialized"
        if converted_count > 0
        else "qwen_moe_groups_all_existing"
    )
    manifest = QwenMoeMaterializationManifest(
        output_dir=destination,
        manifest_path=manifest_path,
        model_id=plan.model_id,
        revision=plan.revision,
        materialization_status=status,
        planned_groups=len(selected_groups),
        converted_groups=converted_count,
        skipped_existing_groups=skipped_count,
        source_tensors_read=source_tensors_read,
        peak_source_tensor_bytes=peak_source_tensor_bytes,
        code_bits=selected_groups[0].code_bits if selected_groups else 0,
        group_size=selected_groups[0].group_size if selected_groups else 0,
        scale_estimator=scale_estimator,
        groups=tuple(records),
    )
    manifest_path.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def audit_qwen_moe_source_index(
    index: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
    model_id: str,
    revision: str,
    expected_model_type: str = QWEN36_35B_A3B_MODEL_TYPE,
    expected_language_layers: int | None = None,
    expected_expert_tensors: int | None = None,
    tensor_metadata: Mapping[str, Any] | None = None,
) -> QwenMoeSourceAudit:
    weight_map = _weight_map_from_index(index)
    conversion_plan = plan_qwen_moe_conversion_from_index(
        index,
        config=config,
        model_id=model_id,
        revision=revision,
        expected_language_layers=expected_language_layers,
        tensor_metadata=tensor_metadata,
    )
    config = config or {}
    model_type = None if config.get("model_type") is None else str(config["model_type"])

    language_layers: dict[int, set[str]] = {}
    auxiliary_expert_tensors = 0
    has_fused_gate_up_proj = False
    has_unfused_gate_or_up_proj = False
    unexpected_projection_pairs: list[str] = []

    for tensor_name in weight_map:
        language_match = _LANGUAGE_EXPERT_RE.match(tensor_name)
        if language_match is not None:
            layer = int(language_match.group("layer"))
            projection = language_match.group("projection")
            language_layers.setdefault(layer, set()).add(projection)
            if projection == "gate_up_proj":
                has_fused_gate_up_proj = True
            if projection in {"gate_proj", "up_proj"}:
                has_unfused_gate_or_up_proj = True
            if projection not in QWEN_MOE_EXPECTED_LANGUAGE_PROJECTIONS:
                unexpected_projection_pairs.append(f"{layer}:{projection}")
            continue

        auxiliary_match = _AUXILIARY_EXPERT_RE.match(tensor_name)
        if auxiliary_match is not None:
            auxiliary_expert_tensors += 1
            projection = auxiliary_match.group("projection")
            if projection == "gate_up_proj":
                has_fused_gate_up_proj = True
            if projection in {"gate_proj", "up_proj"}:
                has_unfused_gate_or_up_proj = True

    sorted_layers = sorted(language_layers)
    layer_range = None if not sorted_layers else (sorted_layers[0], sorted_layers[-1])
    projections_per_layer = tuple(
        sorted({tuple(sorted(projections)) for projections in language_layers.values()})
    )

    expected_layer_ids = (
        range(expected_language_layers)
        if expected_language_layers is not None
        else sorted_layers
    )
    missing_pairs: list[str] = []
    for layer in expected_layer_ids:
        projections = language_layers.get(layer, set())
        for projection in QWEN_MOE_EXPECTED_LANGUAGE_PROJECTIONS:
            if projection not in projections:
                missing_pairs.append(f"{layer}:{projection}")

    language_expert_tensors = sum(len(projections) for projections in language_layers.values())
    expert_tensors_total = language_expert_tensors + auxiliary_expert_tensors
    (
        non_expert_tensors_total,
        non_expert_tensors_by_family,
        language_non_expert_tensors_by_module,
        language_non_expert_layers_by_module,
    ) = _qwen_non_expert_inventory(weight_map)
    expected_expert_tensors = (
        expected_expert_tensors
        if expected_expert_tensors is not None
        else (
            expected_language_layers * len(QWEN_MOE_EXPECTED_LANGUAGE_PROJECTIONS)
            if expected_language_layers is not None
            else None
        )
    )

    checks = {
        "model_type": model_type == expected_model_type,
        "expected_language_layers": (
            True
            if expected_language_layers is None
            else len(language_layers) == expected_language_layers
        ),
        "expected_expert_tensors": (
            True
            if expected_expert_tensors is None
            else language_expert_tensors == expected_expert_tensors
        ),
        "complete_language_projection_pairs": not missing_pairs,
        "known_language_projection_names": not unexpected_projection_pairs,
        "fused_gate_up_proj_present": has_fused_gate_up_proj,
        "no_unfused_gate_or_up_proj": not has_unfused_gate_or_up_proj,
    }

    return QwenMoeSourceAudit(
        model_id=model_id,
        revision=revision,
        model_type=model_type,
        source_index_tensors=len(weight_map),
        source_shards=len(set(weight_map.values())),
        expert_tensors_total=expert_tensors_total,
        language_expert_tensors=language_expert_tensors,
        auxiliary_expert_tensors=auxiliary_expert_tensors,
        non_expert_tensors_total=non_expert_tensors_total,
        non_expert_tensors_by_family=non_expert_tensors_by_family,
        language_non_expert_tensors_by_module=language_non_expert_tensors_by_module,
        language_non_expert_layers_by_module=language_non_expert_layers_by_module,
        language_sparse_layers=len(language_layers),
        language_layer_range=layer_range,
        projections_per_language_layer=projections_per_layer,
        missing_language_projection_pairs=tuple(missing_pairs),
        unexpected_language_projections=tuple(sorted(unexpected_projection_pairs)),
        has_fused_gate_up_proj=has_fused_gate_up_proj,
        has_unfused_gate_or_up_proj=has_unfused_gate_or_up_proj,
        expert_tensor_shape={
            "gate_up_proj": "fused" if has_fused_gate_up_proj else "absent",
            "down_proj": "per_layer_tensor",
        },
        checks=checks,
        conversion_status="source_mapping_audited__materializer_available",
        conversion_blockers=(
            "qwen3_5_moe_runtime_adapter",
            "family_specific_eval_and_benchmark_gates",
        ),
        architecture=QWEN36_35B_A3B_ARCHITECTURE,
        conversion_plan=conversion_plan,
    )
