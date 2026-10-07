from __future__ import annotations

import mlx.core as mx
import numpy as np

from keep.vq.e8 import e8_1bit_packed
from ramp.kernels.vq_qmv import vq_qmv_reference_np
from ramp.nn.linear import QuantizedVQLinear
from ramp.nn.switch_linear import QuantizedVQSwitchLinear
from keep.quant.rht import apply_inverse_rht_np, apply_rht_np, deterministic_rht_signs
from keep.quant.rtn import quantize_weight_rtn


def test_rht_round_trips_power_of_two_vectors() -> None:
    rng = np.random.default_rng(20260627)
    x = rng.normal(size=(5, 32)).astype(np.float32)
    signs = deterministic_rht_signs(32, seed=41)

    transformed = apply_rht_np(x, signs)
    recovered = apply_inverse_rht_np(transformed, signs)

    np.testing.assert_allclose(recovered, x, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(transformed, axis=-1),
        np.linalg.norm(x, axis=-1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_quantized_vq_linear_applies_rht_to_runtime_inputs() -> None:
    rng = np.random.default_rng(12345)
    weight = rng.normal(scale=0.08, size=(12, 32)).astype(np.float32)
    x = rng.normal(size=(3, 32)).astype(np.float32)
    signs = deterministic_rht_signs(32, seed=7)

    layer = QuantizedVQLinear.from_weights(
        mx.array(weight),
        group_size=8,
        rht_signs=mx.array(signs),
    )
    actual = layer(mx.array(x))
    mx.eval(actual)

    transformed_weight = apply_rht_np(weight, signs)
    quantized = quantize_weight_rtn(transformed_weight, group_size=8)
    transformed_x = apply_rht_np(x, signs)
    expected = np.stack(
        [
            vq_qmv_reference_np(
                row,
                quantized.codes,
                quantized.scales,
                e8_1bit_packed(),
                in_dim=32,
                out_dim=12,
                group_size=8,
                code_bits=8,
            )
            for row in transformed_x
        ],
        axis=0,
    )

    assert layer.get("rht_signs") is not None
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=2e-4)


def test_quantized_vq_switch_linear_applies_rht_before_selected_experts() -> None:
    rng = np.random.default_rng(90210)
    weight = rng.normal(scale=0.05, size=(3, 10, 32)).astype(np.float32)
    x = rng.normal(size=(2, 32)).astype(np.float32)
    indices = np.array([[0, 2], [1, 0]], dtype=np.int32)
    signs = deterministic_rht_signs(32, seed=99)

    layer = QuantizedVQSwitchLinear.from_weights(
        mx.array(weight),
        group_size=8,
        use_gather_vqmm=False,
        rht_signs=mx.array(signs),
    )
    actual = layer(mx.array(x), mx.array(indices))
    mx.eval(actual)

    transformed_weight = apply_rht_np(weight, signs)
    quantized = [
        quantize_weight_rtn(expert_weight, group_size=8)
        for expert_weight in transformed_weight
    ]
    transformed_x = apply_rht_np(x, signs)
    expected = np.empty((2, 2, 10), dtype=np.float32)
    for token_idx in range(x.shape[0]):
        for route_idx in range(indices.shape[1]):
            expert_idx = indices[token_idx, route_idx]
            expected[token_idx, route_idx] = vq_qmv_reference_np(
                transformed_x[token_idx],
                quantized[expert_idx].codes,
                quantized[expert_idx].scales,
                e8_1bit_packed(),
                in_dim=32,
                out_dim=10,
                group_size=8,
                code_bits=8,
            )

    assert layer.get("rht_signs") is not None
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=2e-4)


def test_gate_up_only_rht_is_dense_glu_identity_before_quantization() -> None:
    rng = np.random.default_rng(20260701)
    num_experts = 3
    input_dims = 32
    hidden_dims = 12
    route_tokens = 6
    top_k = 2
    gate_weight = rng.normal(scale=0.04, size=(num_experts, hidden_dims, input_dims)).astype(np.float32)
    up_weight = rng.normal(scale=0.04, size=(num_experts, hidden_dims, input_dims)).astype(np.float32)
    down_weight = rng.normal(scale=0.04, size=(num_experts, input_dims, hidden_dims)).astype(np.float32)
    x = rng.normal(size=(route_tokens, input_dims)).astype(np.float32)
    indices = rng.integers(0, num_experts, size=(route_tokens, top_k), dtype=np.int32)
    gate_signs = deterministic_rht_signs(input_dims, seed="dense-gate-rht")
    up_signs = deterministic_rht_signs(input_dims, seed="dense-up-rht")

    original = np.empty((route_tokens, top_k, input_dims), dtype=np.float32)
    rotated = np.empty_like(original)
    rotated_gate_weight = apply_rht_np(gate_weight, gate_signs)
    rotated_up_weight = apply_rht_np(up_weight, up_signs)
    rotated_gate_input = apply_rht_np(x, gate_signs)
    rotated_up_input = apply_rht_np(x, up_signs)
    for token_idx in range(route_tokens):
        for route_idx in range(top_k):
            expert = int(indices[token_idx, route_idx])
            gate = gate_weight[expert] @ x[token_idx]
            up = up_weight[expert] @ x[token_idx]
            hidden = (gate / (1.0 + np.exp(-gate))) * up
            original[token_idx, route_idx] = down_weight[expert] @ hidden

            rotated_gate = rotated_gate_weight[expert] @ rotated_gate_input[token_idx]
            rotated_up = rotated_up_weight[expert] @ rotated_up_input[token_idx]
            rotated_hidden = (rotated_gate / (1.0 + np.exp(-rotated_gate))) * rotated_up
            rotated[token_idx, route_idx] = down_weight[expert] @ rotated_hidden

    np.testing.assert_allclose(rotated, original, rtol=2e-6, atol=2e-6)
