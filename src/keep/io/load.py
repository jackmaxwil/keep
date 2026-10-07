from __future__ import annotations

import json
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

from keep.vq.e8 import (
    E8_1BIT_PACKED_SHA256,
    E8P_PACKED_ABS_SHA256,
    e8_1bit_packed,
    e8p_packed_abs_grid,
)
from ramp.nn.linear import QuantizedVQLinear
from ramp.nn.switch_linear import HighPrecisionSwitchLinear, QuantizedVQSwitchLinear


DTYPE_NBYTES = {
    "U8": 1,
    "UINT8": 1,
    "U16": 2,
    "UINT16": 2,
    "U32": 4,
    "UINT32": 4,
    "F16": 2,
    "FLOAT16": 2,
    "BF16": 2,
    "BFLOAT16": 2,
    "F32": 4,
    "FLOAT32": 4,
    "I8": 1,
    "INT8": 1,
    "I16": 2,
    "INT16": 2,
    "I32": 4,
    "INT32": 4,
    "I64": 8,
    "INT64": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def nbytes(self) -> int:
        if self.dtype not in DTYPE_NBYTES:
            raise ValueError(f"unsupported safetensors dtype {self.dtype!r}")
        count = 1
        for dim in self.shape:
            count *= dim
        return count * DTYPE_NBYTES[self.dtype]


@dataclass(frozen=True)
class SafetensorsInspection:
    path: Path
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]

    @property
    def stored_nbytes(self) -> int:
        return sum(info.nbytes for info in self.tensors.values())


def inspect_safetensors(path: str | Path) -> SafetensorsInspection:
    checkpoint_path = Path(path)
    tensors: dict[str, TensorInfo] = {}
    with safe_open(checkpoint_path, framework="np") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensor_slice = handle.get_slice(key)
            tensors[key] = TensorInfo(
                name=key,
                shape=tuple(tensor_slice.get_shape()),
                dtype=str(tensor_slice.get_dtype()).upper(),
            )
    return SafetensorsInspection(path=checkpoint_path, tensors=tensors, metadata=metadata)


def dense_equivalent_bytes(*, in_dim: int, out_dim: int, dtype_bytes: int = 2) -> int:
    return in_dim * out_dim * dtype_bytes


def switch_dense_equivalent_bytes(
    *,
    in_dim: int,
    out_dim: int,
    num_experts: int,
    dtype_bytes: int = 2,
) -> int:
    return num_experts * dense_equivalent_bytes(in_dim=in_dim, out_dim=out_dim, dtype_bytes=dtype_bytes)


def _metadata_quantization_config(inspection: SafetensorsInspection) -> dict[str, object]:
    raw = inspection.metadata.get("quantization_config")
    if raw is None:
        return {}
    return json.loads(raw)


def _validate_codebook_hash(inspection: SafetensorsInspection) -> None:
    config = _metadata_quantization_config(inspection)
    codebook = config.get("codebook") if isinstance(config, dict) else None
    expected = codebook.get("sha256") if isinstance(codebook, dict) else None
    if expected is not None and expected not in {E8_1BIT_PACKED_SHA256, E8P_PACKED_ABS_SHA256}:
        raise ValueError(f"checkpoint E8 codebook hash {expected!r} is not supported by this runtime")


def _default_codebook_for_bits(code_bits: int) -> mx.array:
    if code_bits == 8:
        return mx.array(e8_1bit_packed())
    if code_bits == 16:
        return mx.array(e8p_packed_abs_grid())
    raise ValueError("code_bits must be 8 or 16")


def infer_vq_linear_dims(inspection: SafetensorsInspection, prefix: str) -> tuple[int, int, int, int]:
    codes_name = f"{prefix}.codes"
    scales_name = f"{prefix}.scales"
    if codes_name not in inspection.tensors:
        raise KeyError(f"missing tensor {codes_name!r}")
    if scales_name not in inspection.tensors:
        raise KeyError(f"missing tensor {scales_name!r}")

    codes = inspection.tensors[codes_name]
    scales = inspection.tensors[scales_name]
    if len(codes.shape) != 2:
        raise ValueError(f"{codes_name} must be 2D, found {codes.shape}")
    if len(scales.shape) != 2:
        raise ValueError(f"{scales_name} must be 2D, found {scales.shape}")
    if codes.shape[0] != scales.shape[0]:
        raise ValueError("codes and scales must have the same out_dim")

    code_bits = 8 if codes.dtype in {"U8", "UINT8"} else 16 if codes.dtype in {"U16", "UINT16"} else None
    if code_bits is None:
        raise ValueError(f"{codes_name} must be uint8 or uint16, found {codes.dtype}")

    out_dim = codes.shape[0]
    in_dim = codes.shape[1] * 8
    if scales.shape[1] <= 0 or in_dim % scales.shape[1] != 0:
        raise ValueError("scale count must divide inferred in_dim")
    group_size = in_dim // scales.shape[1]
    return in_dim, out_dim, group_size, code_bits


def infer_vq_switch_linear_dims(inspection: SafetensorsInspection, prefix: str) -> tuple[int, int, int, int, int]:
    codes_name = f"{prefix}.codes"
    scales_name = f"{prefix}.scales"
    if codes_name not in inspection.tensors:
        raise KeyError(f"missing tensor {codes_name!r}")
    if scales_name not in inspection.tensors:
        raise KeyError(f"missing tensor {scales_name!r}")

    codes = inspection.tensors[codes_name]
    scales = inspection.tensors[scales_name]
    if len(codes.shape) != 3:
        raise ValueError(f"{codes_name} must be 3D [experts, out, in/8], found {codes.shape}")
    if len(scales.shape) != 3:
        raise ValueError(f"{scales_name} must be 3D [experts, out, in/group], found {scales.shape}")
    if codes.shape[:2] != scales.shape[:2]:
        raise ValueError("switch codes and scales must have the same expert and out dimensions")

    code_bits = 8 if codes.dtype in {"U8", "UINT8"} else 16 if codes.dtype in {"U16", "UINT16"} else None
    if code_bits is None:
        raise ValueError(f"{codes_name} must be uint8 or uint16, found {codes.dtype}")

    num_experts, out_dim, codewords = codes.shape
    in_dim = codewords * 8
    if scales.shape[2] <= 0 or in_dim % scales.shape[2] != 0:
        raise ValueError("switch scale count must divide inferred in_dim")
    group_size = in_dim // scales.shape[2]
    return in_dim, out_dim, num_experts, group_size, code_bits


def infer_high_precision_switch_linear_dims(inspection: SafetensorsInspection, prefix: str) -> tuple[int, int, int]:
    weight_name = f"{prefix}.weight"
    if weight_name not in inspection.tensors:
        raise KeyError(f"missing tensor {weight_name!r}")

    weight = inspection.tensors[weight_name]
    if len(weight.shape) != 3:
        raise ValueError(f"{weight_name} must be 3D [experts, out, in], found {weight.shape}")
    if weight.dtype not in {"F16", "FLOAT16", "BF16", "BFLOAT16", "F32", "FLOAT32"}:
        raise ValueError(f"{weight_name} must be a floating point tensor, found {weight.dtype}")
    num_experts, out_dim, in_dim = weight.shape
    if num_experts <= 0 or out_dim <= 0 or in_dim <= 0:
        raise ValueError(f"{weight_name} dimensions must be positive, found {weight.shape}")
    bias_name = f"{prefix}.bias"
    if bias_name in inspection.tensors:
        bias = inspection.tensors[bias_name]
        if bias.shape != (num_experts, out_dim):
            raise ValueError(f"{bias_name} must have shape ({num_experts}, {out_dim}), found {bias.shape}")
    return in_dim, out_dim, num_experts


def current_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def load_quantized_vq_linear(path: str | Path, prefix: str) -> QuantizedVQLinear:
    inspection = inspect_safetensors(path)
    _validate_codebook_hash(inspection)
    in_dim, out_dim, group_size, code_bits = infer_vq_linear_dims(inspection, prefix)
    arrays = mx.load(str(path), format="safetensors")

    codes = arrays[f"{prefix}.codes"]
    scales = arrays[f"{prefix}.scales"]
    bias = arrays.get(f"{prefix}.bias")
    rht_signs = arrays.get(f"{prefix}.rht_signs")
    rotation_matrix = arrays.get(f"{prefix}.rotation_matrix")
    codebook = arrays.get("model.vq_codebook.e8", _default_codebook_for_bits(code_bits))
    return QuantizedVQLinear(
        input_dims=in_dim,
        output_dims=out_dim,
        codes=codes,
        scales=scales,
        codebook=codebook,
        bias=bias,
        group_size=group_size,
        code_bits=code_bits,
        rht_signs=rht_signs,
        rotation_matrix=rotation_matrix,
    )


def load_quantized_vq_switch_linear(path: str | Path, prefix: str) -> QuantizedVQSwitchLinear:
    inspection = inspect_safetensors(path)
    _validate_codebook_hash(inspection)
    in_dim, out_dim, num_experts, group_size, code_bits = infer_vq_switch_linear_dims(inspection, prefix)
    arrays = mx.load(str(path), format="safetensors")

    codes = arrays[f"{prefix}.codes"]
    scales = arrays[f"{prefix}.scales"]
    bias = arrays.get(f"{prefix}.bias")
    rht_signs = arrays.get(f"{prefix}.rht_signs")
    rotation_matrix = arrays.get(f"{prefix}.rotation_matrix")
    codebook = arrays.get("model.vq_codebook.e8", _default_codebook_for_bits(code_bits))
    return QuantizedVQSwitchLinear(
        input_dims=in_dim,
        output_dims=out_dim,
        num_experts=num_experts,
        codes=codes,
        scales=scales,
        codebook=codebook,
        bias=bias,
        group_size=group_size,
        code_bits=code_bits,
        rht_signs=rht_signs,
        rotation_matrix=rotation_matrix,
    )


def _high_precision_tier_from_metadata(inspection: SafetensorsInspection) -> str:
    config = _metadata_quantization_config(inspection)
    policy = config.get("policy") if isinstance(config, dict) else None
    tier = policy.get("dynamic_precision_tier") if isinstance(policy, dict) else None
    return str(tier) if tier is not None else "high"


def load_high_precision_switch_linear(path: str | Path, prefix: str) -> HighPrecisionSwitchLinear:
    inspection = inspect_safetensors(path)
    infer_high_precision_switch_linear_dims(inspection, prefix)
    arrays = mx.load(str(path))

    weight = arrays[f"{prefix}.weight"]
    bias = arrays.get(f"{prefix}.bias")
    return HighPrecisionSwitchLinear(
        weight=weight,
        bias=bias,
        tier=_high_precision_tier_from_metadata(inspection),
    )


def load_switch_linear_projection(
    path: str | Path,
    prefix: str,
) -> QuantizedVQSwitchLinear | HighPrecisionSwitchLinear:
    inspection = inspect_safetensors(path)
    if f"{prefix}.weight" in inspection.tensors:
        return load_high_precision_switch_linear(path, prefix)
    return load_quantized_vq_switch_linear(path, prefix)
