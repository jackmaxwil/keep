from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import e8_1bit_packed
from mlx_vq.io.load import (
    infer_high_precision_switch_linear_dims,
    infer_vq_switch_linear_dims,
    inspect_safetensors,
    load_high_precision_switch_linear,
    load_quantized_vq_switch_linear,
    load_switch_linear_projection,
    switch_dense_equivalent_bytes,
)
from mlx_vq.io.schema import QuantizationConfig
from mlx_vq.kernels.vq_qmv import vq_qmv_reference_np
from mlx_vq.nn.switch_linear import HighPrecisionSwitchLinear, QuantizedVQSwitchLinear


def _write_switch_checkpoint(
    path,
    *,
    prefix: str = "layers.1.mlp.switch_mlp.gate_proj",
    num_experts: int = 4,
    out_dim: int = 16,
    in_dim: int = 64,
    group_size: int = 16,
    seed: int = 909,
    policy: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, size=(num_experts, out_dim, in_dim // 8), dtype=np.uint8)
    scales = rng.uniform(0.4, 0.9, size=(num_experts, out_dim, in_dim // group_size)).astype(np.float16)
    bias = rng.normal(scale=0.01, size=(num_experts, out_dim)).astype(np.float32)
    arrays = {
        f"{prefix}.codes": mx.array(codes),
        f"{prefix}.scales": mx.array(scales),
        f"{prefix}.bias": mx.array(bias),
        "model.vq_codebook.e8": mx.array(e8_1bit_packed()),
    }
    metadata = {"quantization_config": json.dumps(QuantizationConfig(policy=policy or {}).to_json_dict())}
    mx.save_safetensors(str(path), arrays, metadata=metadata)
    return codes, scales, bias


def test_inspect_safetensors_infers_stacked_switch_dims_without_dense_weight(tmp_path) -> None:
    prefix = "layers.1.mlp.switch_mlp.gate_proj"
    path = tmp_path / "switch-vq.safetensors"
    codes, scales, _ = _write_switch_checkpoint(path, prefix=prefix)

    inspection = inspect_safetensors(path)
    assert inspection.tensors[f"{prefix}.codes"].shape == codes.shape
    assert inspection.tensors[f"{prefix}.scales"].shape == scales.shape
    assert infer_vq_switch_linear_dims(inspection, prefix) == (64, 16, 4, 16, 8)
    assert inspection.stored_nbytes < switch_dense_equivalent_bytes(in_dim=64, out_dim=16, num_experts=4)


def test_load_quantized_vq_switch_linear_runs_selected_experts(tmp_path) -> None:
    prefix = "layers.1.mlp.switch_mlp.up_proj"
    path = tmp_path / "switch-vq.safetensors"
    codes, scales, bias = _write_switch_checkpoint(path, prefix=prefix, num_experts=3, out_dim=8, in_dim=32, group_size=8)
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2, 32)).astype(np.float32)
    indices = np.array([[0, 2], [1, 1]], dtype=np.int32)

    layer = load_quantized_vq_switch_linear(path, prefix)
    actual = layer(mx.array(x), mx.array(indices))
    mx.eval(actual)

    expected = np.empty((2, 2, 8), dtype=np.float32)
    for token_idx in range(x.shape[0]):
        for route_idx in range(indices.shape[1]):
            expert_idx = indices[token_idx, route_idx]
            expected[token_idx, route_idx] = (
                vq_qmv_reference_np(
                    x[token_idx],
                    codes[expert_idx],
                    scales[expert_idx],
                    e8_1bit_packed(),
                    in_dim=32,
                    out_dim=8,
                    group_size=8,
                    code_bits=8,
                )
                + bias[expert_idx]
            )

    assert layer.num_experts == 3
    assert layer.input_dims == 32
    assert layer.output_dims == 8
    assert layer.codes.dtype == mx.uint8
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-4)


def test_load_high_precision_switch_linear_runs_selected_experts(tmp_path) -> None:
    prefix = "layers.1.mlp.switch_mlp.gate_proj"
    path = tmp_path / "switch-high.safetensors"
    rng = np.random.default_rng(20260629)
    weight = rng.normal(scale=0.2, size=(3, 5, 4)).astype(np.float32)
    bias = rng.normal(scale=0.01, size=(3, 5)).astype(np.float32)
    mx.save_safetensors(
        str(path),
        {
            f"{prefix}.weight": mx.array(weight).astype(mx.bfloat16),
            f"{prefix}.bias": mx.array(bias).astype(mx.float32),
        },
        metadata={
            "quantization_config": json.dumps(
                QuantizationConfig(policy={"dynamic_precision_tier": "high"}).to_json_dict()
            )
        },
    )
    x = rng.normal(size=(2, 4)).astype(np.float32)
    indices = np.array([[0, 2], [1, 1]], dtype=np.int32)

    inspection = inspect_safetensors(path)
    layer = load_high_precision_switch_linear(path, prefix)
    generic_layer = load_switch_linear_projection(path, prefix)
    actual = layer(mx.array(x), mx.array(indices))
    generic_actual = generic_layer(mx.array(x), mx.array(indices))
    mx.eval(actual, generic_actual)

    expected = np.empty((2, 2, 5), dtype=np.float32)
    stored_weight = np.array(mx.array(weight).astype(mx.bfloat16).astype(mx.float32))
    for token_idx in range(x.shape[0]):
        for route_idx in range(indices.shape[1]):
            expert_idx = indices[token_idx, route_idx]
            expected[token_idx, route_idx] = stored_weight[expert_idx] @ x[token_idx] + bias[expert_idx]

    assert infer_high_precision_switch_linear_dims(inspection, prefix) == (4, 5, 3)
    assert isinstance(layer, HighPrecisionSwitchLinear)
    assert isinstance(generic_layer, HighPrecisionSwitchLinear)
    assert layer.code_bits is None
    assert layer.tier == "high"
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.array(generic_actual), expected, rtol=1e-5, atol=1e-4)


def test_load_quantized_vq_switch_linear_loads_rht_signs(tmp_path) -> None:
    prefix = "layers.1.mlp.switch_mlp.gate_proj"
    path = tmp_path / "switch-vq-rht.safetensors"
    rng = np.random.default_rng(810)
    codes = rng.integers(0, 256, size=(3, 8, 4), dtype=np.uint8)
    scales = rng.uniform(0.4, 0.9, size=(3, 8, 4)).astype(np.float16)
    signs = np.where(np.arange(32) % 3 == 0, -1, 1).astype(np.int8)
    arrays = {
        f"{prefix}.codes": mx.array(codes),
        f"{prefix}.scales": mx.array(scales),
        f"{prefix}.rht_signs": mx.array(signs),
        "model.vq_codebook.e8": mx.array(e8_1bit_packed()),
    }
    metadata = {"quantization_config": json.dumps(QuantizationConfig().to_json_dict())}
    mx.save_safetensors(str(path), arrays, metadata=metadata)

    layer = load_quantized_vq_switch_linear(path, prefix)

    assert layer.get("rht_signs") is not None
    np.testing.assert_array_equal(np.array(layer.rht_signs), signs.astype(np.float32))


def test_load_quantized_vq_switch_linear_loads_rotation_matrix(tmp_path) -> None:
    prefix = "layers.1.mlp.switch_mlp.gate_proj"
    path = tmp_path / "switch-vq-rotation.safetensors"
    rng = np.random.default_rng(811)
    codes = rng.integers(0, 256, size=(3, 8, 4), dtype=np.uint8)
    scales = rng.uniform(0.4, 0.9, size=(3, 8, 4)).astype(np.float16)
    rotation = np.eye(32, dtype=np.float32)
    rotation[[2, 3]] = rotation[[3, 2]]
    arrays = {
        f"{prefix}.codes": mx.array(codes),
        f"{prefix}.scales": mx.array(scales),
        f"{prefix}.rotation_matrix": mx.array(rotation),
        "model.vq_codebook.e8": mx.array(e8_1bit_packed()),
    }
    metadata = {"quantization_config": json.dumps(QuantizationConfig().to_json_dict())}
    mx.save_safetensors(str(path), arrays, metadata=metadata)

    layer = load_quantized_vq_switch_linear(path, prefix)

    assert layer.get("rotation_matrix") is not None
    np.testing.assert_array_equal(np.array(layer.rotation_matrix), rotation)


def test_m1_per_route_down_gather_matches_scalar_qmv() -> None:
    rng = np.random.default_rng(20260624)
    weight = mx.array(rng.normal(scale=0.04, size=(6, 32, 16)).astype(np.float32))
    gather_layer = QuantizedVQSwitchLinear.from_weights(
        weight,
        group_size=8,
        use_gather_vqmm=True,
        route_strategy="auto",
    )
    scalar_layer = QuantizedVQSwitchLinear.from_weights(
        weight,
        group_size=8,
        use_gather_vqmm=False,
        route_strategy="auto",
    )
    x = mx.array(rng.normal(size=(1, 4, 16)).astype(np.float32))
    indices = mx.array(np.array([[0, 3, 5, 2]], dtype=np.int32))

    actual = gather_layer(x, indices)
    expected = scalar_layer(x, indices)
    mx.eval(actual, expected)

    actual_np = np.array(actual)
    expected_np = np.array(expected)
    dot = float(np.sum(actual_np * expected_np))
    norm = float(np.sqrt(np.sum(actual_np * actual_np) * np.sum(expected_np * expected_np)))
    assert actual.shape == (1, 4, 32)
    assert float(np.max(np.abs(actual_np - expected_np))) < 2e-4
    assert dot / norm >= 0.99999


def test_loader_tolerates_selective_precision_policy_metadata(tmp_path) -> None:
    prefix = "layers.1.mlp.switch_mlp.gate_proj"
    path = tmp_path / "switch-vq-policy.safetensors"
    _write_switch_checkpoint(
        path,
        prefix=prefix,
        policy={"code_bits_policy": "1:gate_proj=16", "scale_estimator": "max_abs"},
    )

    inspection = inspect_safetensors(path)
    config = json.loads(inspection.metadata["quantization_config"])
    layer = load_quantized_vq_switch_linear(path, prefix)

    assert config["policy"]["code_bits_policy"] == "1:gate_proj=16"
    assert layer.code_bits == 8
    assert layer.codes.dtype == mx.uint8
