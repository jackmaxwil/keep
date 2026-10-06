from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx_vq.io.source_safetensors import read_safetensors_tensor_bytes


_E2M1_VALUES = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)
_SUPPORTED_WEIGHT_CONFIG = {
    "dynamic": False,
    "num_bits": 4,
    "type": "float",
    "group_size": 16,
}


@dataclass(frozen=True)
class ModelOptNvfp4Spec:
    quant_method: str
    quant_algo: str
    producer_name: str
    producer_version: str | None
    dynamic: bool
    num_bits: int
    weight_type: str
    group_size: int


@dataclass(frozen=True)
class ModelOptNvfp4WeightBundle:
    weight_name: str
    weight_shard: str
    block_scale_name: str
    block_scale_shard: str
    global_scale_name: str
    global_scale_shard: str


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def parse_modelopt_nvfp4_weight_spec(config: Mapping[str, Any]) -> ModelOptNvfp4Spec:
    """Validate and describe the static ModelOpt NVFP4 weight contract."""

    root = _require_mapping(config, "model config")
    quantization = _require_mapping(root.get("quantization_config"), "quantization_config")

    quant_method = quantization.get("quant_method")
    if quant_method != "modelopt":
        raise ValueError("quantization_config.quant_method must be 'modelopt'")
    quant_algo = quantization.get("quant_algo")
    if quant_algo != "NVFP4":
        raise ValueError("quantization_config.quant_algo must be 'NVFP4'")

    producer = _require_mapping(quantization.get("producer"), "quantization_config.producer")
    producer_name = producer.get("name")
    if producer_name != "modelopt":
        raise ValueError("quantization_config.producer.name must be 'modelopt'")
    producer_version = producer.get("version")
    if producer_version is not None and not isinstance(producer_version, str):
        raise ValueError("quantization_config.producer.version must be a string when present")

    config_groups = _require_mapping(
        quantization.get("config_groups"),
        "quantization_config.config_groups",
    )
    if len(config_groups) != 1:
        raise ValueError("quantization_config.config_groups must contain exactly one entry")
    group = _require_mapping(
        next(iter(config_groups.values())),
        "quantization_config.config_groups entry",
    )
    weights = _require_mapping(group.get("weights"), "ModelOpt NVFP4 weights config")
    if set(weights) != set(_SUPPORTED_WEIGHT_CONFIG):
        raise ValueError("ModelOpt NVFP4 weights config must contain exactly the supported fields")
    if weights.get("dynamic") is not False:
        raise ValueError("ModelOpt NVFP4 weights must be static")
    if type(weights.get("num_bits")) is not int or weights["num_bits"] != 4:
        raise ValueError("ModelOpt NVFP4 weights must use 4 bits")
    if weights.get("type") != "float":
        raise ValueError("ModelOpt NVFP4 weights must use float codes")
    if type(weights.get("group_size")) is not int or weights["group_size"] != 16:
        raise ValueError("ModelOpt NVFP4 weights must use group_size 16")

    return ModelOptNvfp4Spec(
        quant_method=quant_method,
        quant_algo=quant_algo,
        producer_name=producer_name,
        producer_version=producer_version,
        dynamic=False,
        num_bits=4,
        weight_type="float",
        group_size=16,
    )


def _indexed_shard(weight_map: Mapping[str, str], tensor_name: str) -> str:
    if tensor_name not in weight_map:
        raise KeyError(f"{tensor_name!r} is missing from the safetensors index")
    shard = weight_map[tensor_name]
    if not isinstance(shard, str) or not shard:
        raise ValueError(f"safetensors index shard for {tensor_name!r} must be a non-empty string")
    return shard


def resolve_modelopt_nvfp4_weight_bundle(
    weight_map: Mapping[str, str],
    weight_name: str,
) -> ModelOptNvfp4WeightBundle:
    """Resolve a packed weight and both independently indexed scale tensors."""

    if not isinstance(weight_map, Mapping):
        raise ValueError("weight_map must be a mapping")
    if not isinstance(weight_name, str) or not weight_name.endswith(".weight"):
        raise ValueError("ModelOpt NVFP4 weight name must end with '.weight'")
    base_name = weight_name[: -len(".weight")]
    if not base_name:
        raise ValueError("ModelOpt NVFP4 weight name must have a non-empty base")
    block_scale_name = f"{base_name}.weight_scale"
    global_scale_name = f"{base_name}.weight_scale_2"

    return ModelOptNvfp4WeightBundle(
        weight_name=weight_name,
        weight_shard=_indexed_shard(weight_map, weight_name),
        block_scale_name=block_scale_name,
        block_scale_shard=_indexed_shard(weight_map, block_scale_name),
        global_scale_name=global_scale_name,
        global_scale_shard=_indexed_shard(weight_map, global_scale_name),
    )


def _decode_e4m3fn_nonnegative(codes: np.ndarray) -> np.ndarray:
    if np.any(codes & np.uint8(0x80)):
        raise ValueError("ModelOpt NVFP4 block scale code has its sign bit set")

    exponent = (codes >> np.uint8(3)) & np.uint8(0x0F)
    mantissa = codes & np.uint8(0x07)
    if np.any((exponent == 0x0F) & (mantissa == 0x07)):
        raise ValueError("ModelOpt NVFP4 block scales must not contain E4M3FN NaN")

    exponent_i32 = exponent.astype(np.int32)
    mantissa_f32 = mantissa.astype(np.float32)
    values = np.empty(codes.shape, dtype=np.float32)
    subnormal = exponent == 0
    values[subnormal] = mantissa_f32[subnormal] * np.float32(2.0**-9)
    normal = ~subnormal
    values[normal] = (np.float32(1.0) + mantissa_f32[normal] / np.float32(8.0)) * np.exp2(
        exponent_i32[normal] - 7,
    ).astype(np.float32)
    return values


def _validated_global_scale(global_scale: np.ndarray | np.float32 | float) -> np.float32:
    scalar = np.asarray(global_scale)
    if scalar.shape != ():
        raise ValueError("ModelOpt NVFP4 global scale must be scalar")
    try:
        value = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ModelOpt NVFP4 global scale must be numeric") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("ModelOpt NVFP4 global scale must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        value_f32 = np.float32(value)
    if not np.isfinite(value_f32) or value_f32 <= np.float32(0.0):
        raise ValueError("ModelOpt NVFP4 global scale must be finite and positive in float32")
    return value_f32


def decode_modelopt_nvfp4_weight(
    packed_weight: np.ndarray,
    block_scale_codes: np.ndarray,
    global_scale: np.ndarray | np.float32 | float,
    *,
    row_chunk_size: int = 64,
) -> np.ndarray:
    """Decode ModelOpt's packed E2M1 weights with E4M3FN block scales."""

    if not isinstance(packed_weight, np.ndarray) or packed_weight.dtype != np.dtype(np.uint8):
        raise ValueError("packed ModelOpt NVFP4 weight must be a uint8 NumPy array")
    if packed_weight.ndim != 2:
        raise ValueError("packed ModelOpt NVFP4 weight must have rank 2")
    rows, packed_columns = packed_weight.shape
    if rows <= 0 or packed_columns <= 0 or packed_columns % 8 != 0:
        raise ValueError("packed ModelOpt NVFP4 weight shape must encode a positive multiple of 16 inputs")

    if not isinstance(block_scale_codes, np.ndarray) or block_scale_codes.dtype != np.dtype(np.uint8):
        raise ValueError("ModelOpt NVFP4 block scale codes must be a uint8 NumPy array")
    if block_scale_codes.ndim != 2:
        raise ValueError("ModelOpt NVFP4 block scale codes must have rank 2")
    expected_scale_shape = (rows, packed_columns // 8)
    if block_scale_codes.shape != expected_scale_shape:
        raise ValueError(
            "ModelOpt NVFP4 block scale shape must be "
            f"{expected_scale_shape}, got {block_scale_codes.shape}"
        )
    if type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be a positive integer")

    global_scale_f32 = _validated_global_scale(global_scale)
    input_columns = packed_columns * 2
    output = np.empty((rows, input_columns), dtype=np.float32, order="C")

    for row_start in range(0, rows, row_chunk_size):
        row_end = min(row_start + row_chunk_size, rows)
        packed_chunk = packed_weight[row_start:row_end]
        decoded_chunk = np.empty((row_end - row_start, input_columns), dtype=np.float32)
        decoded_chunk[:, 0::2] = _E2M1_VALUES[packed_chunk & np.uint8(0x0F)]
        decoded_chunk[:, 1::2] = _E2M1_VALUES[packed_chunk >> np.uint8(4)]

        block_scales = _decode_e4m3fn_nonnegative(block_scale_codes[row_start:row_end])
        with np.errstate(over="ignore", invalid="ignore"):
            block_scales *= global_scale_f32
            decoded_chunk *= np.repeat(block_scales, 16, axis=1)
        if not np.all(np.isfinite(decoded_chunk)):
            raise ValueError("decoded ModelOpt NVFP4 weight must contain only finite values")
        output[row_start:row_end] = decoded_chunk

    return output


def _require_exact_payload_length(raw: bytes, expected: int, tensor_name: str) -> None:
    if len(raw) != expected:
        raise ValueError(
            f"safetensors payload for {tensor_name!r} must be exactly {expected} bytes, got {len(raw)}"
        )


def read_modelopt_nvfp4_weight(
    source_dir: str | Path,
    bundle: ModelOptNvfp4WeightBundle,
) -> np.ndarray:
    """Read and decode one independently sharded ModelOpt NVFP4 weight bundle."""

    if not isinstance(bundle, ModelOptNvfp4WeightBundle):
        raise ValueError("bundle must be a ModelOptNvfp4WeightBundle")
    source_root = Path(source_dir)
    weight_header, weight_raw = read_safetensors_tensor_bytes(
        source_root / bundle.weight_shard,
        bundle.weight_name,
    )
    scale_header, scale_raw = read_safetensors_tensor_bytes(
        source_root / bundle.block_scale_shard,
        bundle.block_scale_name,
    )
    global_header, global_raw = read_safetensors_tensor_bytes(
        source_root / bundle.global_scale_shard,
        bundle.global_scale_name,
    )

    if weight_header.dtype != "U8":
        raise ValueError(f"{bundle.weight_name!r} must use safetensors dtype 'U8'")
    if len(weight_header.shape) != 2:
        raise ValueError(f"{bundle.weight_name!r} must have rank 2")
    rows, packed_columns = weight_header.shape
    if rows <= 0 or packed_columns <= 0 or packed_columns % 8 != 0:
        raise ValueError(f"{bundle.weight_name!r} must encode a positive multiple of 16 inputs")
    _require_exact_payload_length(weight_raw, rows * packed_columns, bundle.weight_name)

    if scale_header.dtype != "F8_E4M3":
        raise ValueError(f"{bundle.block_scale_name!r} must use safetensors dtype 'F8_E4M3'")
    if len(scale_header.shape) != 2:
        raise ValueError(f"{bundle.block_scale_name!r} must have rank 2")
    expected_scale_shape = (rows, packed_columns // 8)
    if scale_header.shape != expected_scale_shape:
        raise ValueError(
            f"{bundle.block_scale_name!r} must have shape {expected_scale_shape}, got {scale_header.shape}"
        )
    _require_exact_payload_length(
        scale_raw,
        expected_scale_shape[0] * expected_scale_shape[1],
        bundle.block_scale_name,
    )

    if global_header.dtype != "F32":
        raise ValueError(f"{bundle.global_scale_name!r} must use safetensors dtype 'F32'")
    if global_header.shape != ():
        raise ValueError(f"{bundle.global_scale_name!r} must be a scalar")
    _require_exact_payload_length(global_raw, 4, bundle.global_scale_name)

    packed_weight = np.frombuffer(weight_raw, dtype=np.uint8).reshape(weight_header.shape)
    block_scale_codes = np.frombuffer(scale_raw, dtype=np.uint8).reshape(scale_header.shape)
    global_scale = np.frombuffer(global_raw, dtype="<f4", count=1)[0]
    return decode_modelopt_nvfp4_weight(packed_weight, block_scale_codes, global_scale)
