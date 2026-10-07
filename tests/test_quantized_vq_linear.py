from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import e8_1bit_packed
from mlx_vq.io.load import (
    current_rss_bytes,
    dense_equivalent_bytes,
    infer_vq_linear_dims,
    inspect_safetensors,
    load_quantized_vq_linear,
)
from mlx_vq.io.schema import QuantizationConfig
from mlx_vq.kernels.vq_qmv import vq_qmv_reference_np


def _dtype_name(dtype: object) -> str:
    return getattr(dtype, "name", str(dtype)).rsplit(".", maxsplit=1)[-1]


def _write_checkpoint(
    path,
    *,
    prefix: str = "layers.0.mlp.down_proj",
    out_dim: int = 64,
    in_dim: int = 512,
    group_size: int = 128,
    seed: int = 20260623,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, size=(out_dim, in_dim // 8), dtype=np.uint8)
    scales = rng.uniform(0.9, 1.1, size=(out_dim, in_dim // group_size)).astype(np.float16)
    bias = rng.normal(scale=0.01, size=(out_dim,)).astype(np.float32)
    arrays = {
        f"{prefix}.codes": mx.array(codes),
        f"{prefix}.scales": mx.array(scales),
        f"{prefix}.bias": mx.array(bias),
        "model.vq_codebook.e8": mx.array(e8_1bit_packed()),
    }
    metadata = {"quantization_config": json.dumps(QuantizationConfig().to_json_dict())}
    mx.save_safetensors(str(path), arrays, metadata=metadata)
    return codes, scales, bias


def test_inspect_safetensors_estimates_vq_storage_without_dense_weight(tmp_path) -> None:
    prefix = "layers.0.mlp.down_proj"
    path = tmp_path / "synthetic-vq.safetensors"
    codes, scales, bias = _write_checkpoint(path, prefix=prefix)

    inspection = inspect_safetensors(path)
    assert set(inspection.tensors) == {
        f"{prefix}.codes",
        f"{prefix}.scales",
        f"{prefix}.bias",
        "model.vq_codebook.e8",
    }
    assert inspection.tensors[f"{prefix}.codes"].shape == codes.shape
    assert inspection.tensors[f"{prefix}.codes"].dtype in {"U8", "UINT8"}
    assert inspection.tensors[f"{prefix}.scales"].shape == scales.shape
    assert inspection.tensors[f"{prefix}.bias"].shape == bias.shape

    in_dim, out_dim, group_size, code_bits = infer_vq_linear_dims(inspection, prefix)
    assert (in_dim, out_dim, group_size, code_bits) == (512, 64, 128, 8)
    assert inspection.stored_nbytes < dense_equivalent_bytes(in_dim=in_dim, out_dim=out_dim)


def test_load_quantized_vq_linear_preserves_integer_tensors_and_runs_kernel(tmp_path) -> None:
    prefix = "layers.0.mlp.down_proj"
    path = tmp_path / "synthetic-vq.safetensors"
    codes, scales, bias = _write_checkpoint(path, prefix=prefix, out_dim=32, in_dim=256, group_size=64)
    rng = np.random.default_rng(123)
    x = rng.normal(size=(256,)).astype(np.float32)

    layer = load_quantized_vq_linear(path, prefix)
    assert layer.input_dims == 256
    assert layer.output_dims == 32
    assert _dtype_name(layer.codes.dtype) == "uint8"
    assert _dtype_name(layer.codebook.dtype) == "uint32"

    actual = layer(mx.array(x))
    expected = vq_qmv_reference_np(
        x,
        codes,
        scales,
        e8_1bit_packed(),
        in_dim=256,
        out_dim=32,
        group_size=64,
        code_bits=8,
    ) + bias
    mx.eval(actual)

    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-4)


def test_load_quantized_vq_linear_loads_rht_signs(tmp_path) -> None:
    prefix = "layers.0.mlp.gate_proj"
    path = tmp_path / "synthetic-vq-rht.safetensors"
    rng = np.random.default_rng(919)
    codes = rng.integers(0, 256, size=(8, 4), dtype=np.uint8)
    scales = rng.uniform(0.9, 1.1, size=(8, 4)).astype(np.float16)
    signs = np.where(np.arange(32) % 2 == 0, 1, -1).astype(np.int8)
    arrays = {
        f"{prefix}.codes": mx.array(codes),
        f"{prefix}.scales": mx.array(scales),
        f"{prefix}.rht_signs": mx.array(signs),
        "model.vq_codebook.e8": mx.array(e8_1bit_packed()),
    }
    metadata = {"quantization_config": json.dumps(QuantizationConfig().to_json_dict())}
    mx.save_safetensors(str(path), arrays, metadata=metadata)

    layer = load_quantized_vq_linear(path, prefix)

    assert layer.get("rht_signs") is not None
    np.testing.assert_array_equal(np.array(layer.rht_signs), signs.astype(np.float32))


def test_load_quantized_vq_linear_loads_rotation_matrix(tmp_path) -> None:
    prefix = "layers.0.mlp.gate_proj"
    path = tmp_path / "synthetic-vq-rotation.safetensors"
    rng = np.random.default_rng(920)
    codes = rng.integers(0, 256, size=(8, 4), dtype=np.uint8)
    scales = rng.uniform(0.9, 1.1, size=(8, 4)).astype(np.float16)
    rotation = np.eye(32, dtype=np.float32)
    rotation[[0, 1]] = rotation[[1, 0]]
    arrays = {
        f"{prefix}.codes": mx.array(codes),
        f"{prefix}.scales": mx.array(scales),
        f"{prefix}.rotation_matrix": mx.array(rotation),
        "model.vq_codebook.e8": mx.array(e8_1bit_packed()),
    }
    metadata = {"quantization_config": json.dumps(QuantizationConfig().to_json_dict())}
    mx.save_safetensors(str(path), arrays, metadata=metadata)

    layer = load_quantized_vq_linear(path, prefix)

    assert layer.get("rotation_matrix") is not None
    np.testing.assert_array_equal(np.array(layer.rotation_matrix), rotation)


def test_synthetic_checkpoint_rss_does_not_jump_by_dense_weight_size(tmp_path) -> None:
    prefix = "layers.0.mlp.down_proj"
    path = tmp_path / "synthetic-vq-large.safetensors"
    _write_checkpoint(path, prefix=prefix, out_dim=2048, in_dim=2048, group_size=512)

    inspection = inspect_safetensors(path)
    in_dim, out_dim, _, _ = infer_vq_linear_dims(inspection, prefix)
    dense_bytes = dense_equivalent_bytes(in_dim=in_dim, out_dim=out_dim)
    assert inspection.stored_nbytes < dense_bytes

    before = current_rss_bytes()
    layer = load_quantized_vq_linear(path, prefix)
    mx.eval(layer.parameters())
    after = current_rss_bytes()

    assert _dtype_name(layer.codes.dtype) == "uint8"
    assert after - before < dense_bytes
