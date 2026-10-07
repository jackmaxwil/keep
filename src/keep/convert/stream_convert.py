from __future__ import annotations

import json
import re
import struct
import time
from contextlib import ExitStack
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from keep.vq.e8 import e8_1bit_packed, e8p_packed_abs_grid
from keep.convert.inspect_hf import ConfigSummary, summarize_config
from keep.convert.nvfp4 import (
    ModelOptNvfp4WeightBundle,
    parse_modelopt_nvfp4_weight_spec,
    read_modelopt_nvfp4_weight,
    resolve_modelopt_nvfp4_weight_bundle,
)
from keep.io.schema import DEFAULT_GROUP_SIZE, QuantizationConfig, codebook_metadata_for_bits
from ramp.models.glm52_policy import GLM52Precision, classify_glm52_parameter
from keep.quant.rotation import validate_rotation_matrix_np
from keep.quant.rht import apply_rht_np, deterministic_rht_signs
from keep.quant.rtn import ScaleEstimator, quantize_weight_rtn


@dataclass(frozen=True)
class VQStorageEstimate:
    weights: int
    code_bits: int
    group_size: int
    code_bytes: int
    scale_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.code_bytes + self.scale_bytes


@dataclass(frozen=True)
class ModelConversionPlan:
    summary: ConfigSummary
    routed_expert_params: int
    vq1_storage: VQStorageEstimate
    vq2_storage: VQStorageEstimate


class TensorAction(str, Enum):
    VQ = "vq"
    COPY = "copy"
    SKIP = "skip"


class SourceWeightEncoding(str, Enum):
    DENSE = "dense"
    MODELOPT_NVFP4 = "modelopt_nvfp4"

    @property
    def decoder(self) -> str | None:
        if self == SourceWeightEncoding.MODELOPT_NVFP4:
            return "modelopt_nvfp4_v1"
        return None


@dataclass(frozen=True)
class SafetensorsIndex:
    metadata: dict[str, object]
    weight_map: dict[str, str]

    @property
    def shards(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.weight_map.values())))


@dataclass(frozen=True)
class PlannedTensor:
    name: str
    source_shard: str
    action: TensorAction
    precision: GLM52Precision


@dataclass(frozen=True)
class ExpertProjection:
    layer: int
    expert: int
    projection: str


@dataclass(frozen=True)
class VQExpertGroup:
    layer: int
    projection: str
    experts: tuple[int, ...]
    source_shards: tuple[str, ...]
    input_dims: int
    output_dims: int
    code_bits: int
    group_size: int
    source_weight_encoding: SourceWeightEncoding = SourceWeightEncoding.DENSE
    source_model_id: str | None = None
    source_revision: str | None = None
    source_config_sha256: str | None = None
    source_index_sha256: str | None = None
    source_profile: str | None = None

    @property
    def source_decoder(self) -> str | None:
        return self.source_weight_encoding.decoder

    @property
    def codes_name(self) -> str:
        return f"model.layers.{self.layer}.mlp.switch_mlp.{self.projection}.codes"

    @property
    def scales_name(self) -> str:
        return f"model.layers.{self.layer}.mlp.switch_mlp.{self.projection}.scales"

    @property
    def rht_signs_name(self) -> str:
        return f"model.layers.{self.layer}.mlp.switch_mlp.{self.projection}.rht_signs"

    @property
    def rotation_matrix_name(self) -> str:
        return f"model.layers.{self.layer}.mlp.switch_mlp.{self.projection}.rotation_matrix"

    @property
    def codes_shape(self) -> tuple[int, int, int]:
        return (len(self.experts), self.output_dims, self.input_dims // 8)

    @property
    def scales_shape(self) -> tuple[int, int, int]:
        return (len(self.experts), self.output_dims, self.input_dims // self.group_size)

    @property
    def code_bytes(self) -> int:
        return len(self.experts) * self.output_dims * (self.input_dims // 8) * (self.code_bits // 8)

    @property
    def scale_bytes(self) -> int:
        return len(self.experts) * self.output_dims * (self.input_dims // self.group_size) * 2

    @property
    def dense_source_bytes(self) -> int:
        bytes_per_value = 4 if self.source_weight_encoding == SourceWeightEncoding.MODELOPT_NVFP4 else 2
        return len(self.experts) * self.output_dims * self.input_dims * bytes_per_value


@dataclass(frozen=True)
class StreamingConversionPlan:
    model_id: str
    summary: ConfigSummary
    source_shards: tuple[str, ...]
    tensors: tuple[PlannedTensor, ...]
    vq_groups: tuple[VQExpertGroup, ...]
    missing_vq_groups: tuple[str, ...]
    code_bits_policy: tuple[tuple[str, int], ...] = ()
    source_weight_encoding: SourceWeightEncoding = SourceWeightEncoding.DENSE
    source_revision: str | None = None
    source_config_sha256: str | None = None
    source_index_sha256: str | None = None
    source_profile: str | None = None

    @property
    def source_decoder(self) -> str | None:
        return self.source_weight_encoding.decoder

    @property
    def vq_code_bytes(self) -> int:
        return sum(group.code_bytes for group in self.vq_groups)

    @property
    def vq_scale_bytes(self) -> int:
        return sum(group.scale_bytes for group in self.vq_groups)

    @property
    def peak_source_projection_bytes(self) -> int:
        return max((group.dense_source_bytes // max(1, len(group.experts)) for group in self.vq_groups), default=0)


@dataclass(frozen=True)
class ConvertedVQGroup:
    output_path: Path
    codes_name: str
    scales_name: str
    codes_shape: tuple[int, int, int]
    scales_shape: tuple[int, int, int]
    source_tensors_read: int
    peak_source_tensor_bytes: int
    code_bits: int = 8
    group_size: int = DEFAULT_GROUP_SIZE
    scale_estimator: ScaleEstimator = "max_abs"
    elapsed_seconds: float | None = None
    expert_workers: int = 1
    rht_seed: str | None = None
    rht_id: str | None = None
    rht_kind: str | None = None
    rotation_id: str | None = None
    source_weight_encoding: SourceWeightEncoding = SourceWeightEncoding.DENSE
    source_decoder: str | None = None
    source_model_id: str | None = None
    source_revision: str | None = None
    source_config_sha256: str | None = None
    source_index_sha256: str | None = None
    source_profile: str | None = None


@dataclass(frozen=True)
class _ExpertConversionTask:
    expert: int
    source_dir: str
    shard_name: str
    tensor_name: str
    expected_shape: tuple[int, int]
    group_size: int
    code_bits: int
    scale_estimator: ScaleEstimator
    source_weight_encoding: SourceWeightEncoding = SourceWeightEncoding.DENSE
    source_bundle: ModelOptNvfp4WeightBundle | None = None
    rht_signs: np.ndarray | None = None
    rotation_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class _ExpertConversionResult:
    expert: int
    codes: np.ndarray
    scales: np.ndarray
    source_tensor_bytes: int


@dataclass(frozen=True)
class ConversionManifest:
    output_dir: Path
    manifest_path: Path
    model_id: str
    converted_groups: tuple[ConvertedVQGroup, ...]
    existing_groups: tuple[ConvertedVQGroup, ...]
    planned_vq_groups: int
    selected_vq_groups: int
    skipped_vq_groups: int
    skipped_existing_outputs: int
    total_source_tensors_read: int
    peak_source_tensor_bytes: int
    scale_estimator: ScaleEstimator = "max_abs"
    code_bits_policy: tuple[tuple[str, int], ...] = ()
    source_weight_encoding: SourceWeightEncoding = SourceWeightEncoding.DENSE
    source_decoder: str | None = None
    source_revision: str | None = None
    source_config_sha256: str | None = None
    source_index_sha256: str | None = None
    source_profile: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        group_records = []
        for group in self.converted_groups:
            group_records.append(_converted_group_json(group, status="converted"))
        for group in self.existing_groups:
            group_records.append(_converted_group_json(group, status="existing"))
        return {
            "model_id": self.model_id,
            "source_revision": self.source_revision,
            "source_config_sha256": self.source_config_sha256,
            "source_index_sha256": self.source_index_sha256,
            "source_profile": self.source_profile,
            "output_dir": str(self.output_dir),
            "planned_vq_groups": self.planned_vq_groups,
            "selected_vq_groups": self.selected_vq_groups,
            "converted_vq_groups": len(self.converted_groups),
            "existing_vq_groups": len(self.existing_groups),
            "ready_vq_groups": len(self.converted_groups) + len(self.existing_groups),
            "skipped_vq_groups": self.skipped_vq_groups,
            "skipped_existing_outputs": self.skipped_existing_outputs,
            "total_source_tensors_read": self.total_source_tensors_read,
            "peak_source_tensor_bytes": self.peak_source_tensor_bytes,
            "scale_estimator": self.scale_estimator,
            "code_bits_policy": dict(self.code_bits_policy),
            "source_weight_encoding": self.source_weight_encoding.value,
            "source_decoder": self.source_decoder,
            "groups": group_records,
        }


def _converted_group_json(group: ConvertedVQGroup, *, status: str) -> dict[str, object]:
    payload = {
        "status": status,
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
        "source_weight_encoding": group.source_weight_encoding.value,
        "source_decoder": group.source_decoder,
        "source_model_id": group.source_model_id,
        "source_revision": group.source_revision,
        "source_config_sha256": group.source_config_sha256,
        "source_index_sha256": group.source_index_sha256,
        "source_profile": group.source_profile,
    }
    if group.rht_seed is not None:
        payload["rht"] = "hadamard_signs_v1"
        payload["rht_seed"] = group.rht_seed
    if group.rht_id is not None:
        payload["rht"] = group.rht_kind or "learned_hadamard_signs_v1"
        payload["rht_id"] = group.rht_id
    if group.rotation_id is not None:
        payload["rotation"] = "learned_orthogonal_v1"
        payload["rotation_id"] = group.rotation_id
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{uuid4().hex}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


@dataclass(frozen=True)
class SourceShardInventory:
    source_dir: Path
    required_shards: int
    present_shards: int
    missing_shards: tuple[Path, ...]
    present_bytes: int

    @property
    def missing_shard_count(self) -> int:
        return len(self.missing_shards)

    def to_json_dict(self, *, max_missing_examples: int = 10) -> dict[str, object]:
        missing_examples = [str(path) for path in self.missing_shards[:max_missing_examples]]
        return {
            "scope": "planned_vq_groups",
            "source_dir": str(self.source_dir),
            "required_shards": self.required_shards,
            "present_shards": self.present_shards,
            "missing_shards": self.missing_shard_count,
            "missing_examples": missing_examples,
            "present_bytes": self.present_bytes,
        }


_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def _code_bits_policy_items(policy: Mapping[str, int] | None) -> tuple[tuple[str, int], ...]:
    if not policy:
        return ()
    normalized = []
    for key, value in sorted(policy.items()):
        if value not in (8, 16):
            raise ValueError("code bits policy values must be 8 or 16")
        normalized.append((str(key), int(value)))
    return tuple(normalized)


def _resolve_group_code_bits(
    *,
    default_code_bits: int,
    code_bits_policy: Mapping[str, int] | None,
    layer: int,
    projection: str,
) -> int:
    if default_code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if not code_bits_policy:
        return default_code_bits
    keys = (
        f"{layer}:{projection}",
        f"*:{projection}",
        f"{layer}:*",
        "*",
    )
    for key in keys:
        if key in code_bits_policy:
            value = int(code_bits_policy[key])
            if value not in (8, 16):
                raise ValueError("code bits policy values must be 8 or 16")
            return value
    return default_code_bits


def estimate_vq_storage(weights: int, *, code_bits: int, group_size: int = 512) -> VQStorageEstimate:
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if weights <= 0:
        raise ValueError("weights must be positive")
    if weights % 8 != 0:
        raise ValueError("VQ weights must be divisible into 8D codewords")
    code_bytes = (weights // 8) * (code_bits // 8)
    scale_groups = (weights + group_size - 1) // group_size
    scale_bytes = scale_groups * 2
    return VQStorageEstimate(
        weights=weights,
        code_bits=code_bits,
        group_size=group_size,
        code_bytes=code_bytes,
        scale_bytes=scale_bytes,
    )


def routed_expert_param_count(summary: ConfigSummary) -> int:
    params_per_expert = 3 * summary.hidden_size * summary.moe_intermediate_size
    return params_per_expert * summary.n_routed_experts * summary.sparse_layers


def effective_group_size(in_dim: int, preferred_group_size: int = 512) -> int:
    if in_dim <= 0:
        raise ValueError("in_dim must be positive")
    if in_dim % 8 != 0:
        raise ValueError("in_dim must be divisible by 8")
    if preferred_group_size <= 0:
        raise ValueError("preferred_group_size must be positive")
    if in_dim % preferred_group_size == 0:
        return preferred_group_size
    limit = min(preferred_group_size, in_dim)
    for candidate in range(limit, 0, -8):
        if in_dim % candidate == 0:
            return candidate
    raise AssertionError("8 should always divide an 8D-aligned input dimension")


def estimate_routed_vq_storage(summary: ConfigSummary, *, code_bits: int, group_size: int = 512) -> VQStorageEstimate:
    weights = routed_expert_param_count(summary)
    code_bytes = 0
    scale_bytes = 0
    for projection in ("gate_proj", "up_proj", "down_proj"):
        input_dims, output_dims = _projection_dims(summary, projection)
        effective_group = effective_group_size(input_dims, group_size)
        per_projection_rows = summary.sparse_layers * summary.n_routed_experts * output_dims
        code_bytes += per_projection_rows * (input_dims // 8) * (code_bits // 8)
        scale_bytes += per_projection_rows * (input_dims // effective_group) * 2
    return VQStorageEstimate(
        weights=weights,
        code_bits=code_bits,
        group_size=group_size,
        code_bytes=code_bytes,
        scale_bytes=scale_bytes,
    )


def plan_from_config(config: dict, *, model_id: str, group_size: int = 512) -> ModelConversionPlan:
    summary = summarize_config(config, model_id=model_id)
    routed_params = routed_expert_param_count(summary)
    return ModelConversionPlan(
        summary=summary,
        routed_expert_params=routed_params,
        vq1_storage=estimate_routed_vq_storage(summary, code_bits=8, group_size=group_size),
        vq2_storage=estimate_routed_vq_storage(summary, code_bits=16, group_size=group_size),
    )


def load_safetensors_index(path: str | Path) -> SafetensorsIndex:
    payload = json.loads(Path(path).read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index must contain a weight_map object")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("safetensors index metadata must be an object when present")
    return SafetensorsIndex(metadata=metadata, weight_map={str(k): str(v) for k, v in weight_map.items()})


def parse_expert_projection(name: str) -> ExpertProjection | None:
    match = _EXPERT_RE.match(name)
    if match is None:
        return None
    return ExpertProjection(
        layer=int(match.group("layer")),
        expert=int(match.group("expert")),
        projection=match.group("projection"),
    )


def _projection_dims(summary: ConfigSummary, projection: str) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return summary.hidden_size, summary.moe_intermediate_size
    if projection == "down_proj":
        return summary.moe_intermediate_size, summary.hidden_size
    raise ValueError(f"unknown expert projection {projection!r}")


def _expected_sparse_layers(summary: ConfigSummary) -> range:
    return range(summary.dense_layers, summary.num_hidden_layers)


def _detect_source_weight_encoding(config: Mapping[str, object]) -> SourceWeightEncoding:
    if "quantization_config" not in config:
        return SourceWeightEncoding.DENSE
    quantization_config = config["quantization_config"]
    if not isinstance(quantization_config, Mapping) or quantization_config.get("quant_method") != "modelopt":
        raise ValueError("unsupported quantization_config for streaming conversion")
    try:
        parse_modelopt_nvfp4_weight_spec(config)
    except ValueError as exc:
        raise ValueError(f"unsupported quantization_config for streaming conversion: {exc}") from exc
    return SourceWeightEncoding.MODELOPT_NVFP4


def plan_streaming_conversion_from_index(
    config: dict,
    index: SafetensorsIndex,
    *,
    model_id: str,
    revision: str | None = None,
    config_sha256: str | None = None,
    index_sha256: str | None = None,
    source_profile: str | None = None,
    group_size: int = 512,
    code_bits: int = 8,
    code_bits_policy: Mapping[str, int] | None = None,
) -> StreamingConversionPlan:
    summary = summarize_config(config, model_id=model_id)
    source_weight_encoding = _detect_source_weight_encoding(config)
    tensors: list[PlannedTensor] = []
    grouped: dict[tuple[int, str], list[tuple[int, str]]] = {}

    for name, shard in sorted(index.weight_map.items()):
        precision = classify_glm52_parameter(name)
        action = TensorAction.VQ if precision == GLM52Precision.VQ_ROUTED_EXPERT else TensorAction.COPY
        tensors.append(PlannedTensor(name=name, source_shard=shard, action=action, precision=precision))

        expert = parse_expert_projection(name)
        if expert is not None:
            grouped.setdefault((expert.layer, expert.projection), []).append((expert.expert, shard))

    vq_groups: list[VQExpertGroup] = []
    missing: list[str] = []
    expected_experts = set(range(summary.n_routed_experts))
    for layer in _expected_sparse_layers(summary):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            entries = grouped.get((layer, projection), [])
            experts = tuple(sorted(expert for expert, _ in entries))
            if set(experts) != expected_experts:
                missing_count = summary.n_routed_experts - len(set(experts))
                missing.append(f"layer {layer} {projection}: missing {missing_count} experts")
                continue
            source_shards = {shard for _, shard in entries}
            if source_weight_encoding == SourceWeightEncoding.MODELOPT_NVFP4:
                companion_errors = []
                for expert in experts:
                    weight_name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                    try:
                        bundle = resolve_modelopt_nvfp4_weight_bundle(index.weight_map, weight_name)
                    except (KeyError, ValueError) as exc:
                        companion_errors.append(f"expert {expert}: {exc}")
                        continue
                    source_shards.update(
                        (bundle.weight_shard, bundle.block_scale_shard, bundle.global_scale_shard)
                    )
                if companion_errors:
                    missing.append(
                        f"layer {layer} {projection}: missing ModelOpt NVFP4 companion(s): "
                        + "; ".join(companion_errors)
                    )
                    continue
            input_dims, output_dims = _projection_dims(summary, projection)
            effective_group = effective_group_size(input_dims, group_size)
            group_code_bits = _resolve_group_code_bits(
                default_code_bits=code_bits,
                code_bits_policy=code_bits_policy,
                layer=layer,
                projection=projection,
            )
            vq_groups.append(
                VQExpertGroup(
                    layer=layer,
                    projection=projection,
                    experts=experts,
                    source_shards=tuple(sorted(source_shards)),
                    input_dims=input_dims,
                    output_dims=output_dims,
                    code_bits=group_code_bits,
                    group_size=effective_group,
                    source_weight_encoding=source_weight_encoding,
                    source_model_id=model_id if revision is not None else None,
                    source_revision=revision,
                    source_config_sha256=config_sha256,
                    source_index_sha256=index_sha256,
                    source_profile=source_profile,
                )
            )

    return StreamingConversionPlan(
        model_id=model_id,
        summary=summary,
        source_shards=index.shards,
        tensors=tuple(tensors),
        vq_groups=tuple(vq_groups),
        missing_vq_groups=tuple(missing),
        code_bits_policy=_code_bits_policy_items(code_bits_policy),
        source_weight_encoding=source_weight_encoding,
        source_revision=revision,
        source_config_sha256=config_sha256,
        source_index_sha256=index_sha256,
        source_profile=source_profile,
    )


def _read_tensor_from_shard(source_dir: Path, shard_name: str, tensor_name: str) -> np.ndarray:
    path = source_dir / shard_name
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"{path} is not a valid safetensors file")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size))
        tensor_info = header.get(tensor_name)
        if tensor_info is None:
            raise KeyError(f"{tensor_name!r} not found in {path}")
        dtype_name = tensor_info["dtype"]
        shape = tuple(int(dim) for dim in tensor_info["shape"])
        start, end = (int(offset) for offset in tensor_info["data_offsets"])
        data_start = 8 + header_size
        handle.seek(data_start + start)
        raw = handle.read(end - start)

    if dtype_name == "BF16":
        values = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return values.view(np.float32).reshape(shape).copy()

    dtype = {
        "F32": np.dtype("<f4"),
        "F16": np.dtype("<f2"),
        "U8": np.dtype("u1"),
        "I8": np.dtype("i1"),
        "U16": np.dtype("<u2"),
        "I16": np.dtype("<i2"),
        "I32": np.dtype("<i4"),
        "I64": np.dtype("<i8"),
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported safetensors dtype {dtype_name!r} for {tensor_name!r}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32, copy=True)


def vq_group_output_filename(group: VQExpertGroup) -> str:
    return f"layer-{group.layer:05d}-{group.projection}.safetensors"


def source_shard_inventory(source_dir: str | Path, groups: tuple[VQExpertGroup, ...]) -> SourceShardInventory:
    source_root = Path(source_dir)
    shard_names = sorted({shard for group in groups for shard in group.source_shards})
    missing = []
    present = 0
    present_bytes = 0
    for shard_name in shard_names:
        path = source_root / shard_name
        if path.exists():
            present += 1
            present_bytes += path.stat().st_size
        else:
            missing.append(path)
    return SourceShardInventory(
        source_dir=source_root,
        required_shards=len(shard_names),
        present_shards=present,
        missing_shards=tuple(missing),
        present_bytes=present_bytes,
    )


def missing_source_shards(source_dir: str | Path, groups: tuple[VQExpertGroup, ...]) -> tuple[Path, ...]:
    return source_shard_inventory(source_dir, groups).missing_shards


def _convert_expert_projection(task: _ExpertConversionTask) -> _ExpertConversionResult:
    if task.source_weight_encoding == SourceWeightEncoding.MODELOPT_NVFP4:
        if task.source_bundle is None:
            raise ValueError(f"{task.tensor_name} is missing its ModelOpt NVFP4 source bundle")
        weight = read_modelopt_nvfp4_weight(Path(task.source_dir), task.source_bundle)
    else:
        weight = _read_tensor_from_shard(Path(task.source_dir), task.shard_name, task.tensor_name)
    if weight.shape != task.expected_shape:
        raise ValueError(f"{task.tensor_name} must have shape {task.expected_shape}, found {weight.shape}")
    source_tensor_bytes = weight.nbytes
    if task.rht_signs is not None:
        weight = apply_rht_np(weight, task.rht_signs)
    if task.rotation_matrix is not None:
        weight = weight @ task.rotation_matrix
    quantized = quantize_weight_rtn(
        weight,
        group_size=task.group_size,
        code_bits=task.code_bits,
        scale_estimator=task.scale_estimator,
    )
    return _ExpertConversionResult(
        expert=task.expert,
        codes=quantized.codes,
        scales=quantized.scales,
        source_tensor_bytes=source_tensor_bytes,
    )


def convert_vq_group_from_safetensors(
    *,
    source_dir: str | Path,
    index: SafetensorsIndex,
    group: VQExpertGroup,
    output_path: str | Path,
    scale_estimator: ScaleEstimator = "max_abs",
    expert_workers: int = 1,
    expert_executor: Executor | None = None,
    rht_seed: int | str | None = None,
    rht_signs: np.ndarray | None = None,
    rht_id: str | None = None,
    rotation_matrix: np.ndarray | None = None,
    rotation_id: str | None = None,
) -> ConvertedVQGroup:
    if expert_workers <= 0:
        raise ValueError("expert_workers must be positive")
    if expert_executor is not None and expert_workers == 1:
        raise ValueError("expert_executor requires expert_workers greater than 1")
    source_root = Path(source_dir)
    output = Path(output_path)
    start = time.perf_counter()
    if rht_seed is not None and rht_signs is not None:
        raise ValueError("rht_seed and rht_signs are mutually exclusive")
    if (rht_seed is not None or rht_signs is not None) and rotation_matrix is not None:
        raise ValueError("rht and rotation_matrix are mutually exclusive")
    if rht_seed is not None:
        rht_values = deterministic_rht_signs(group.input_dims, seed=rht_seed)
    elif rht_signs is not None:
        rht_values = np.asarray(rht_signs, dtype=np.int8)
        if rht_values.shape != (group.input_dims,):
            raise ValueError(f"rht_signs must have shape ({group.input_dims},), found {rht_values.shape}")
        if not np.all((rht_values == 1) | (rht_values == -1)):
            raise ValueError("rht_signs values must be -1 or 1")
    else:
        rht_values = None
    rotation_values = (
        validate_rotation_matrix_np(rotation_matrix, dim=group.input_dims)
        if rotation_matrix is not None
        else None
    )
    tasks: list[_ExpertConversionTask] = []
    for expert in group.experts:
        source_name = f"model.layers.{group.layer}.mlp.experts.{expert}.{group.projection}.weight"
        shard_name = index.weight_map[source_name]
        source_bundle = None
        if group.source_weight_encoding == SourceWeightEncoding.MODELOPT_NVFP4:
            source_bundle = resolve_modelopt_nvfp4_weight_bundle(index.weight_map, source_name)
        tasks.append(
            _ExpertConversionTask(
                expert=expert,
                source_dir=str(source_root),
                shard_name=shard_name,
                tensor_name=source_name,
                expected_shape=(group.output_dims, group.input_dims),
                group_size=group.group_size,
                code_bits=group.code_bits,
                scale_estimator=scale_estimator,
                source_weight_encoding=group.source_weight_encoding,
                source_bundle=source_bundle,
                rht_signs=rht_values,
                rotation_matrix=rotation_values,
            )
        )

    results_by_expert: dict[int, _ExpertConversionResult] = {}
    if expert_workers == 1:
        for task in tasks:
            result = _convert_expert_projection(task)
            results_by_expert[result.expert] = result
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal"):
                mx.metal.clear_cache()
    else:
        if expert_executor is None:
            with ProcessPoolExecutor(max_workers=expert_workers) as executor:
                futures = [executor.submit(_convert_expert_projection, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    results_by_expert[result.expert] = result
        else:
            futures = [expert_executor.submit(_convert_expert_projection, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results_by_expert[result.expert] = result

    ordered_results = [results_by_expert[expert] for expert in group.experts]
    peak_source_bytes = max((result.source_tensor_bytes for result in ordered_results), default=0)
    codes = [result.codes for result in ordered_results]
    scales = [result.scales for result in ordered_results]

    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal"):
        mx.metal.clear_cache()

    codebook_name, codebook_sha256 = codebook_metadata_for_bits(group.code_bits)
    codebook = e8_1bit_packed() if group.code_bits == 8 else e8p_packed_abs_grid()
    arrays = {
        group.codes_name: mx.array(np.stack(codes, axis=0)),
        group.scales_name: mx.array(np.stack(scales, axis=0)),
        "model.vq_codebook.e8": mx.array(codebook),
    }
    if rht_values is not None:
        arrays[group.rht_signs_name] = mx.array(rht_values)
    if rotation_values is not None:
        arrays[group.rotation_matrix_name] = mx.array(rotation_values)
    policy = {"scale_estimator": scale_estimator}
    if group.source_decoder is not None:
        policy["source_weight_encoding"] = group.source_weight_encoding.value
        policy["source_decoder"] = group.source_decoder
    if group.source_model_id is not None:
        policy["source_model_id"] = group.source_model_id
    if group.source_revision is not None:
        policy["source_revision"] = group.source_revision
    if group.source_config_sha256 is not None:
        policy["source_config_sha256"] = group.source_config_sha256
    if group.source_index_sha256 is not None:
        policy["source_index_sha256"] = group.source_index_sha256
    if group.source_profile is not None:
        policy["source_profile"] = group.source_profile
        policy["decoded_expert_working_set"] = "one_per_worker"
    if rht_seed is not None:
        policy["rht"] = "hadamard_signs_v1"
        policy["rht_seed"] = str(rht_seed)
    if rht_signs is not None:
        policy["rht"] = "learned_hadamard_signs_v1"
        if rht_id is not None:
            policy["rht_id"] = str(rht_id)
    if rotation_values is not None:
        policy["rotation"] = "learned_orthogonal_v1"
        if rotation_id is not None:
            policy["rotation_id"] = str(rotation_id)
    metadata = {
        "quantization_config": json.dumps(
            QuantizationConfig(
                default_code_bits=group.code_bits,
                default_group_size=group.group_size,
                codebook_name=codebook_name,
                codebook_sha256=codebook_sha256,
                policy=policy,
            ).to_json_dict()
        )
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_name(
        f"{output.stem}.partial-{uuid4().hex}{output.suffix}"
    )
    try:
        mx.save_safetensors(str(partial_output), arrays, metadata=metadata)
        partial_output.replace(output)
    finally:
        partial_output.unlink(missing_ok=True)
    return ConvertedVQGroup(
        output_path=output,
        codes_name=group.codes_name,
        scales_name=group.scales_name,
        codes_shape=group.codes_shape,
        scales_shape=group.scales_shape,
        source_tensors_read=len(group.experts),
        peak_source_tensor_bytes=peak_source_bytes,
        code_bits=group.code_bits,
        group_size=group.group_size,
        scale_estimator=scale_estimator,
        elapsed_seconds=time.perf_counter() - start,
        expert_workers=expert_workers,
        rht_seed=str(rht_seed) if rht_seed is not None else None,
        rht_id=str(rht_id) if rht_signs is not None and rht_id is not None else None,
        rht_kind="learned_hadamard_signs_v1" if rht_signs is not None else None,
        rotation_id=str(rotation_id) if rotation_values is not None and rotation_id is not None else None,
        source_weight_encoding=group.source_weight_encoding,
        source_decoder=group.source_decoder,
        source_model_id=group.source_model_id,
        source_revision=group.source_revision,
        source_config_sha256=group.source_config_sha256,
        source_index_sha256=group.source_index_sha256,
        source_profile=group.source_profile,
    )


def _validate_existing_vq_group_artifact(
    path: Path,
    *,
    group: VQExpertGroup,
    scale_estimator: ScaleEstimator,
) -> None:
    with safe_open(path, framework="np") as handle:
        metadata = dict(handle.metadata() or {})
        raw_config = metadata.get("quantization_config")
        if raw_config is None:
            raise ValueError(f"existing VQ group {path} is missing quantization_config metadata")
        try:
            quantization_config = json.loads(raw_config)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"existing VQ group {path} has invalid quantization_config metadata"
            ) from exc
        if not isinstance(quantization_config, Mapping):
            raise ValueError(f"existing VQ group {path} quantization_config must be an object")

        expected_codebook_name, expected_codebook_sha256 = codebook_metadata_for_bits(group.code_bits)
        expected_metadata = {
            "default_code_bits": group.code_bits,
            "default_group_size": group.group_size,
        }
        for name, expected in expected_metadata.items():
            actual = quantization_config.get(name)
            if actual != expected:
                raise ValueError(
                    f"existing VQ group {path} {name} must be {expected!r}, found {actual!r}"
                )

        codebook = quantization_config.get("codebook")
        if not isinstance(codebook, Mapping):
            raise ValueError(f"existing VQ group {path} codebook metadata must be an object")
        expected_codebook = {
            "name": expected_codebook_name,
            "sha256": expected_codebook_sha256,
        }
        for name, expected in expected_codebook.items():
            actual = codebook.get(name)
            if actual != expected:
                raise ValueError(
                    f"existing VQ group {path} codebook {name} must be {expected!r}, found {actual!r}"
                )

        policy = quantization_config.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError(f"existing VQ group {path} policy metadata must be an object")
        expected_policy = {
            "scale_estimator": scale_estimator,
            "source_weight_encoding": group.source_weight_encoding.value,
            "source_decoder": group.source_decoder,
        }
        if group.source_model_id is not None:
            expected_policy["source_model_id"] = group.source_model_id
        if group.source_revision is not None:
            expected_policy["source_revision"] = group.source_revision
        if group.source_config_sha256 is not None:
            expected_policy["source_config_sha256"] = group.source_config_sha256
        if group.source_index_sha256 is not None:
            expected_policy["source_index_sha256"] = group.source_index_sha256
        if group.source_profile is not None:
            expected_policy["source_profile"] = group.source_profile
            expected_policy["decoded_expert_working_set"] = "one_per_worker"
        actual_policy = {
            "scale_estimator": policy.get("scale_estimator"),
            "source_weight_encoding": policy.get("source_weight_encoding", "dense"),
            "source_decoder": policy.get("source_decoder"),
            "source_model_id": policy.get("source_model_id"),
            "source_revision": policy.get("source_revision"),
            "source_config_sha256": policy.get("source_config_sha256"),
            "source_index_sha256": policy.get("source_index_sha256"),
            "source_profile": policy.get("source_profile"),
            "decoded_expert_working_set": policy.get("decoded_expert_working_set"),
        }
        for name, expected in expected_policy.items():
            actual = actual_policy[name]
            if actual != expected:
                raise ValueError(
                    f"existing VQ group {path} {name} must be {expected!r}, found {actual!r}"
                )

        keys = set(handle.keys())
        codebook_tensor_name = "model.vq_codebook.e8"
        expected_codebook_values = (
            e8_1bit_packed() if group.code_bits == 8 else e8p_packed_abs_grid()
        )
        required_tensors = {
            codebook_tensor_name: (expected_codebook_values.shape, "U32"),
            group.codes_name: (group.codes_shape, "U8" if group.code_bits == 8 else "U16"),
            group.scales_name: (group.scales_shape, "F16"),
        }
        unexpected_tensors = sorted(keys - set(required_tensors))
        if unexpected_tensors:
            raise ValueError(
                f"existing VQ group {path} contains unexpected tensor(s): "
                f"{unexpected_tensors[:10]}"
            )
        for name, (expected_shape, expected_dtype) in required_tensors.items():
            if name not in keys:
                raise ValueError(f"existing VQ group {path} is missing tensor {name!r}")
            tensor_slice = handle.get_slice(name)
            actual_shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            actual_dtype = str(tensor_slice.get_dtype()).upper()
            if actual_shape != expected_shape:
                raise ValueError(
                    f"existing VQ group {path} tensor {name!r} must have shape "
                    f"{expected_shape}, found {actual_shape}"
                )
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"existing VQ group {path} tensor {name!r} must have dtype "
                    f"{expected_dtype}, found {actual_dtype}"
                )
        actual_codebook_values = handle.get_tensor(codebook_tensor_name)
        if not np.array_equal(actual_codebook_values, expected_codebook_values):
            raise ValueError(
                f"existing VQ group {path} tensor {codebook_tensor_name!r} does not match "
                f"the {expected_codebook_name} codebook"
            )


def convert_vq_groups_from_safetensors(
    *,
    source_dir: str | Path,
    index: SafetensorsIndex,
    plan: StreamingConversionPlan,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    max_groups: int | None = None,
    selected_groups: tuple[VQExpertGroup, ...] | None = None,
    skip_existing: bool = False,
    overwrite: bool = False,
    scale_estimator: ScaleEstimator = "max_abs",
    expert_workers: int = 1,
) -> ConversionManifest:
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive when provided")
    if max_groups is not None and selected_groups is not None:
        raise ValueError("max_groups and selected_groups are mutually exclusive")
    if expert_workers <= 0:
        raise ValueError("expert_workers must be positive")
    if skip_existing and overwrite:
        raise ValueError("skip_existing and overwrite are mutually exclusive")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_output = Path(manifest_path) if manifest_path is not None else output_root / "conversion-manifest.json"
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    planned_groups = plan.vq_groups
    groups_to_convert = (
        selected_groups
        if selected_groups is not None
        else planned_groups[:max_groups]
        if max_groups is not None
        else planned_groups
    )
    planned_by_key = {
        (group.layer, group.projection): group for group in planned_groups
    }
    planned_group_keys = set(planned_by_key)
    selected_group_keys = [(group.layer, group.projection) for group in groups_to_convert]
    if len(set(selected_group_keys)) != len(selected_group_keys):
        raise ValueError("selected_groups contains duplicate layer/projection groups")
    unknown_groups = [key for key in selected_group_keys if key not in planned_group_keys]
    if unknown_groups:
        raise ValueError(f"selected_groups contains unplanned groups: {unknown_groups}")
    incompatible_groups = [
        key
        for key, group in zip(selected_group_keys, groups_to_convert, strict=True)
        if planned_by_key.get(key) != group
    ]
    if incompatible_groups:
        raise ValueError(
            "selected_groups must contain exact members of plan.vq_groups: "
            f"{incompatible_groups}"
        )
    missing_shards = missing_source_shards(source_dir, groups_to_convert)
    if missing_shards:
        missing_display = ", ".join(str(path) for path in missing_shards[:5])
        if len(missing_shards) > 5:
            missing_display += f", ... ({len(missing_shards)} total)"
        raise FileNotFoundError(f"missing source shard(s): {missing_display}")

    converted: list[ConvertedVQGroup] = []
    existing: list[ConvertedVQGroup] = []
    skipped_existing = 0
    with ExitStack() as stack:
        expert_executor = None
        for group in groups_to_convert:
            output_path = output_root / vq_group_output_filename(group)
            if output_path.exists() and not overwrite:
                if skip_existing:
                    _validate_existing_vq_group_artifact(
                        output_path,
                        group=group,
                        scale_estimator=scale_estimator,
                    )
                    skipped_existing += 1
                    existing.append(
                        ConvertedVQGroup(
                            output_path=output_path,
                            codes_name=group.codes_name,
                            scales_name=group.scales_name,
                            codes_shape=group.codes_shape,
                            scales_shape=group.scales_shape,
                            source_tensors_read=0,
                            peak_source_tensor_bytes=0,
                            code_bits=group.code_bits,
                            group_size=group.group_size,
                            scale_estimator=scale_estimator,
                            expert_workers=expert_workers,
                            source_weight_encoding=group.source_weight_encoding,
                            source_decoder=group.source_decoder,
                            source_model_id=group.source_model_id,
                            source_revision=group.source_revision,
                            source_config_sha256=group.source_config_sha256,
                            source_index_sha256=group.source_index_sha256,
                            source_profile=group.source_profile,
                        )
                    )
                    continue
                raise FileExistsError(f"{output_path} already exists; use skip_existing=True or overwrite=True")
            if expert_workers > 1 and expert_executor is None:
                expert_executor = stack.enter_context(ProcessPoolExecutor(max_workers=expert_workers))
            converted.append(
                convert_vq_group_from_safetensors(
                    source_dir=source_dir,
                    index=index,
                    group=group,
                    output_path=output_path,
                    scale_estimator=scale_estimator,
                    expert_workers=expert_workers,
                    expert_executor=expert_executor,
                )
            )

    manifest = ConversionManifest(
        output_dir=output_root,
        manifest_path=manifest_output,
        model_id=plan.model_id,
        converted_groups=tuple(converted),
        existing_groups=tuple(existing),
        planned_vq_groups=len(planned_groups),
        selected_vq_groups=len(groups_to_convert),
        skipped_vq_groups=len(planned_groups) - len(converted) - len(existing),
        skipped_existing_outputs=skipped_existing,
        total_source_tensors_read=sum(group.source_tensors_read for group in converted),
        peak_source_tensor_bytes=max((group.peak_source_tensor_bytes for group in converted), default=0),
        scale_estimator=scale_estimator,
        code_bits_policy=plan.code_bits_policy,
        source_weight_encoding=plan.source_weight_encoding,
        source_decoder=plan.source_decoder,
        source_revision=plan.source_revision,
        source_config_sha256=plan.source_config_sha256,
        source_index_sha256=plan.source_index_sha256,
        source_profile=plan.source_profile,
    )
    _write_json_atomic(manifest_output, manifest.to_json_dict())
    return manifest
