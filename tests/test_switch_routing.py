from __future__ import annotations

import inspect

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import ramp.models.glm4_moe_adapter as glm4_moe_adapter
from keep.vq.e8 import cosine_similarity, decode_weight_matrix
from ramp.models.glm4_moe_adapter import (
    GLM4MoeRoutingConfig,
    QuantizedVQGLM4MoE,
    QuantizedVQSwitchGLU,
    group_expert_select,
)
from ramp.nn.switch_linear import HighPrecisionSwitchLinear, QuantizedVQSwitchLinear
from ramp.ops.vq_switch import vq_switch_qmv
from keep.quant.rht import apply_rht_np, deterministic_rht_signs


def _dense_expert_outputs(
    x: np.ndarray,
    indices: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    bias: np.ndarray | None,
) -> np.ndarray:
    flat_x = x.reshape(-1, x.shape[-1])
    per_route_input = x.shape[:-1] == indices.shape
    flat_indices = indices.reshape(-1, 1) if per_route_input else indices.reshape(-1, indices.shape[-1])
    out_dim = codes.shape[1]
    out = np.empty((flat_x.shape[0], flat_indices.shape[1], out_dim), dtype=np.float32)
    for token_idx, token_x in enumerate(flat_x):
        for route_idx, expert_idx in enumerate(flat_indices[token_idx]):
            weight = decode_weight_matrix(codes[expert_idx], scales[expert_idx])
            out[token_idx, route_idx] = weight @ token_x
            if bias is not None:
                out[token_idx, route_idx] += bias[expert_idx]
    if per_route_input:
        return out[:, 0, :].reshape((*x.shape[:-1], out_dim))
    return out.reshape((*x.shape[:-1], indices.shape[-1], out_dim))


def _dense_switch_glu(
    x: np.ndarray,
    indices: np.ndarray,
    switch_mlp: QuantizedVQSwitchGLU,
) -> np.ndarray:
    gate_bias = switch_mlp.gate_proj.get("bias")
    up_bias = switch_mlp.up_proj.get("bias")
    down_bias = switch_mlp.down_proj.get("bias")
    gate_input = _rht_input_for_projection(switch_mlp.gate_proj, x)
    up_input = _rht_input_for_projection(switch_mlp.up_proj, x)
    gate = _dense_expert_outputs(
        gate_input,
        indices,
        np.array(switch_mlp.gate_proj.codes, copy=False),
        np.array(switch_mlp.gate_proj.scales, copy=False),
        None if gate_bias is None else np.array(gate_bias, copy=False),
    )
    up = _dense_expert_outputs(
        up_input,
        indices,
        np.array(switch_mlp.up_proj.codes, copy=False),
        np.array(switch_mlp.up_proj.scales, copy=False),
        None if up_bias is None else np.array(up_bias, copy=False),
    )
    hidden = (1 / (1 + np.exp(-gate))) * gate * up
    down_input = _rht_input_for_projection(switch_mlp.down_proj, hidden)
    return _dense_expert_outputs(
        down_input,
        indices,
        np.array(switch_mlp.down_proj.codes, copy=False),
        np.array(switch_mlp.down_proj.scales, copy=False),
        None if down_bias is None else np.array(down_bias, copy=False),
    )


def _rht_input_for_projection(projection: QuantizedVQSwitchLinear, x: np.ndarray) -> np.ndarray:
    signs = projection.get("rht_signs")
    if signs is None:
        return x
    return apply_rht_np(x, np.array(signs, dtype=np.int8))


def test_quantized_vq_switch_linear_matches_dense_dequant_selected_experts() -> None:
    rng = np.random.default_rng(20260624)
    weight = rng.normal(scale=0.04, size=(3, 5, 16)).astype(np.float32)
    bias = rng.normal(scale=0.01, size=(3, 5)).astype(np.float32)
    x = rng.normal(size=(4, 16)).astype(np.float32)
    indices = np.array([[0, 2], [1, 1], [2, 0], [0, 1]], dtype=np.int32)

    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), mx.array(bias), group_size=8)
    actual = layer(mx.array(x), mx.array(indices))
    mx.eval(actual)
    expected = _dense_expert_outputs(
        x,
        indices,
        np.array(layer.codes, copy=False),
        np.array(layer.scales, copy=False),
        bias,
    )

    assert actual.shape == (4, 2, 5)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)
    assert cosine_similarity(np.array(actual), expected) >= 0.99999


def test_quantized_vq_switch_linear_preserves_batch_leading_dims() -> None:
    rng = np.random.default_rng(7)
    weight = rng.normal(scale=0.03, size=(4, 7, 16)).astype(np.float32)
    x = rng.normal(size=(2, 3, 16)).astype(np.float32)
    indices = np.array([[[0, 1], [2, 3], [1, 1]], [[3, 0], [2, 0], [0, 2]]], dtype=np.int32)

    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    actual = layer(mx.array(x), mx.array(indices))
    mx.eval(actual)

    assert actual.shape == (2, 3, 2, 7)


def test_quantized_vq_switch_glu_matches_dense_dequant_reference() -> None:
    rng = np.random.default_rng(55)
    switch_mlp = QuantizedVQSwitchGLU.from_weights(
        gate_weight=mx.array(rng.normal(scale=0.03, size=(3, 8, 16)).astype(np.float32)),
        up_weight=mx.array(rng.normal(scale=0.03, size=(3, 8, 16)).astype(np.float32)),
        down_weight=mx.array(rng.normal(scale=0.03, size=(3, 16, 8)).astype(np.float32)),
        group_size=8,
    )
    x = rng.normal(size=(5, 16)).astype(np.float32)
    indices = np.array([[0, 1], [2, 0], [1, 2], [2, 2], [0, 0]], dtype=np.int32)

    actual = switch_mlp(mx.array(x), mx.array(indices))
    expected = _dense_switch_glu(x, indices, switch_mlp)
    mx.eval(actual)

    assert actual.shape == (5, 2, 16)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)
    assert cosine_similarity(np.array(actual), expected) >= 0.99999


def test_switch_glu_accepts_high_precision_projection_tier() -> None:
    rng = np.random.default_rng(20260629)
    gate_weight = rng.normal(scale=0.03, size=(3, 8, 16)).astype(np.float32)
    up_weight = rng.normal(scale=0.03, size=(3, 8, 16)).astype(np.float32)
    down_weight = rng.normal(scale=0.03, size=(3, 16, 8)).astype(np.float32)
    switch_mlp = QuantizedVQSwitchGLU(
        gate_proj=HighPrecisionSwitchLinear(weight=mx.array(gate_weight), tier="high"),
        up_proj=HighPrecisionSwitchLinear(weight=mx.array(up_weight), tier="high"),
        down_proj=HighPrecisionSwitchLinear(weight=mx.array(down_weight), tier="high"),
    )
    x = rng.normal(size=(5, 16)).astype(np.float32)
    indices = np.array([[0, 1], [2, 0], [1, 2], [2, 2], [0, 0]], dtype=np.int32)

    actual = switch_mlp(mx.array(x), mx.array(indices))
    mx.eval(actual)

    gate = np.empty((5, 2, 8), dtype=np.float32)
    up = np.empty((5, 2, 8), dtype=np.float32)
    for token_idx in range(x.shape[0]):
        for route_idx, expert_idx in enumerate(indices[token_idx]):
            gate[token_idx, route_idx] = gate_weight[expert_idx] @ x[token_idx]
            up[token_idx, route_idx] = up_weight[expert_idx] @ x[token_idx]
    hidden = (1 / (1 + np.exp(-gate))) * gate * up
    expected = np.empty((5, 2, 16), dtype=np.float32)
    for token_idx in range(x.shape[0]):
        for route_idx, expert_idx in enumerate(indices[token_idx]):
            expected[token_idx, route_idx] = down_weight[expert_idx] @ hidden[token_idx, route_idx]

    assert switch_mlp._can_use_shared_sorted_route_prefill(mx.array(x), mx.array(indices)) is False
    assert switch_mlp.gate_proj.code_bits is None
    assert actual.shape == (5, 2, 16)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)


def test_quantized_vq_switch_glu_large_route_path_reuses_sorted_helper(monkeypatch) -> None:
    rng = np.random.default_rng(20260628)
    num_experts = 4
    input_dims = 16
    hidden_dims = 8
    route_tokens = 32
    top_k = 2
    gate_bias = rng.normal(scale=0.01, size=(num_experts, hidden_dims)).astype(np.float32)
    up_bias = rng.normal(scale=0.01, size=(num_experts, hidden_dims)).astype(np.float32)
    down_bias = rng.normal(scale=0.01, size=(num_experts, input_dims)).astype(np.float32)
    switch_mlp = QuantizedVQSwitchGLU(
        gate_proj=QuantizedVQSwitchLinear.from_weights(
            mx.array(rng.normal(scale=0.03, size=(num_experts, hidden_dims, input_dims)).astype(np.float32)),
            mx.array(gate_bias),
            group_size=8,
        ),
        up_proj=QuantizedVQSwitchLinear.from_weights(
            mx.array(rng.normal(scale=0.03, size=(num_experts, hidden_dims, input_dims)).astype(np.float32)),
            mx.array(up_bias),
            group_size=8,
        ),
        down_proj=QuantizedVQSwitchLinear.from_weights(
            mx.array(rng.normal(scale=0.03, size=(num_experts, input_dims, hidden_dims)).astype(np.float32)),
            mx.array(down_bias),
            group_size=8,
        ),
        prefill_engine="vq_metal",
    )
    x = rng.normal(size=(route_tokens, input_dims)).astype(np.float32)
    indices = rng.integers(0, num_experts, size=(route_tokens, top_k), dtype=np.int32)
    real_helper = getattr(glm4_moe_adapter, "gather_vqmm_sorted_routes")
    calls: list[dict[str, object]] = []

    def recording_helper(x_arg, codes, scales, codebook, sorted_rhs, sorted_lhs, **kwargs):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "rhs": np.array(sorted_rhs, copy=False),
                "lhs": np.array(sorted_lhs, copy=False),
                "input_dims": kwargs["input_dims"],
                "output_dims": kwargs["output_dims"],
                "projection": kwargs["projection"],
            }
        )
        return real_helper(x_arg, codes, scales, codebook, sorted_rhs, sorted_lhs, **kwargs)

    monkeypatch.setattr(glm4_moe_adapter, "gather_vqmm_sorted_routes", recording_helper)

    actual = switch_mlp(mx.array(x), mx.array(indices))
    expected = _dense_switch_glu(x, indices, switch_mlp)
    mx.eval(actual)

    route_count = indices.size
    assert actual.shape == (route_tokens, top_k, input_dims)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)
    assert cosine_similarity(np.array(actual), expected) >= 0.99999
    assert [call["projection"] for call in calls] == ["gate_up", "gate_up", "down"]
    assert [call["x_shape"] for call in calls] == [
        (route_tokens, input_dims),
        (route_tokens, input_dims),
        (route_count, hidden_dims),
    ]
    assert all(np.all(call["rhs"][:-1] <= call["rhs"][1:]) for call in calls)
    np.testing.assert_array_equal(calls[0]["rhs"], calls[1]["rhs"])
    np.testing.assert_array_equal(calls[0]["rhs"], calls[2]["rhs"])
    np.testing.assert_array_equal(calls[0]["lhs"], calls[1]["lhs"])
    np.testing.assert_array_equal(calls[2]["lhs"], np.arange(route_count, dtype=np.int32))


def test_quantized_vq_switch_glu_large_route_path_applies_rht_signs() -> None:
    rng = np.random.default_rng(20260701)
    num_experts = 4
    input_dims = 32
    hidden_dims = 16
    route_tokens = 32
    top_k = 2
    gate_signs = mx.array(deterministic_rht_signs(input_dims, seed="gate-shared-rht"))
    up_signs = mx.array(deterministic_rht_signs(input_dims, seed="up-shared-rht"))
    switch_mlp = QuantizedVQSwitchGLU(
        gate_proj=QuantizedVQSwitchLinear.from_weights(
            mx.array(rng.normal(scale=0.03, size=(num_experts, hidden_dims, input_dims)).astype(np.float32)),
            group_size=8,
            rht_signs=gate_signs,
        ),
        up_proj=QuantizedVQSwitchLinear.from_weights(
            mx.array(rng.normal(scale=0.03, size=(num_experts, hidden_dims, input_dims)).astype(np.float32)),
            group_size=8,
            rht_signs=up_signs,
        ),
        down_proj=QuantizedVQSwitchLinear.from_weights(
            mx.array(rng.normal(scale=0.03, size=(num_experts, input_dims, hidden_dims)).astype(np.float32)),
            group_size=8,
        ),
        prefill_engine="vq_metal",
    )
    x = rng.normal(size=(route_tokens, input_dims)).astype(np.float32)
    indices = rng.integers(0, num_experts, size=(route_tokens, top_k), dtype=np.int32)

    assert switch_mlp._can_use_shared_sorted_route_prefill(mx.array(x), mx.array(indices))
    actual = switch_mlp(mx.array(x), mx.array(indices))
    expected = _dense_switch_glu(x, indices, switch_mlp)
    mx.eval(actual)

    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)
    assert cosine_similarity(np.array(actual), expected) >= 0.99999


def test_quantized_vq_switch_glu_low_rank_sidecar_keeps_shared_sorted_prefill_for_zero_residual() -> None:
    rng = np.random.default_rng(20260701)
    num_experts = 4
    input_dims = 32
    hidden_dims = 16
    route_tokens = 32
    top_k = 2
    gate_proj = QuantizedVQSwitchLinear.from_weights(
        mx.array(rng.normal(scale=0.03, size=(num_experts, hidden_dims, input_dims)).astype(np.float32)),
        group_size=8,
    )
    gate_proj.set_continuous_sidecar(
        low_rank_left=mx.zeros((num_experts, hidden_dims, 2), dtype=mx.float32),
        low_rank_right=mx.zeros((num_experts, 2, input_dims), dtype=mx.float32),
    )
    switch_mlp = QuantizedVQSwitchGLU(
        gate_proj=gate_proj,
        up_proj=QuantizedVQSwitchLinear.from_weights(
            mx.array(rng.normal(scale=0.03, size=(num_experts, hidden_dims, input_dims)).astype(np.float32)),
            group_size=8,
        ),
        down_proj=QuantizedVQSwitchLinear.from_weights(
            mx.array(rng.normal(scale=0.03, size=(num_experts, input_dims, hidden_dims)).astype(np.float32)),
            group_size=8,
        ),
        prefill_engine="vq_metal",
    )
    x = mx.array(rng.normal(size=(route_tokens, input_dims)).astype(np.float32))
    indices = mx.array(rng.integers(0, num_experts, size=(route_tokens, top_k), dtype=np.int32))

    assert switch_mlp._can_use_shared_sorted_route_prefill(x, indices) is True
    actual = switch_mlp(x, indices)
    expected = _dense_switch_glu(np.array(x), np.array(indices), switch_mlp)
    mx.eval(actual)

    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)


def test_quantized_vq_switch_glu_auto_prefill_uses_nax_when_available(monkeypatch) -> None:
    rng = np.random.default_rng(20260630)
    switch_mlp = QuantizedVQSwitchGLU.from_weights(
        gate_weight=mx.array(rng.normal(scale=0.03, size=(4, 8, 16)).astype(np.float32)),
        up_weight=mx.array(rng.normal(scale=0.03, size=(4, 8, 16)).astype(np.float32)),
        down_weight=mx.array(rng.normal(scale=0.03, size=(4, 16, 8)).astype(np.float32)),
        group_size=8,
    )
    prefill = {"route_count": 64, "activation_dtype": mx.bfloat16}

    assert switch_mlp.prefill_engine == "auto"
    monkeypatch.setattr(glm4_moe_adapter.nax, "is_available", lambda: True)
    assert switch_mlp._sorted_prefill_implementation(**prefill) == "nax_e8"

    monkeypatch.setattr(glm4_moe_adapter.nax, "is_available", lambda: False)
    assert switch_mlp._sorted_prefill_implementation(**prefill) == "metal"

    switch_mlp.prefill_engine = "vq_metal"
    assert switch_mlp._sorted_prefill_implementation(**prefill) == "metal"

    switch_mlp.prefill_engine = "nax_e8"
    assert switch_mlp._sorted_prefill_implementation(**prefill) == "nax_e8"

    switch_mlp_e8p = QuantizedVQSwitchGLU.from_weights(
        gate_weight=mx.array(rng.normal(scale=0.03, size=(4, 8, 16)).astype(np.float32)),
        up_weight=mx.array(rng.normal(scale=0.03, size=(4, 8, 16)).astype(np.float32)),
        down_weight=mx.array(rng.normal(scale=0.03, size=(4, 16, 8)).astype(np.float32)),
        group_size=8,
        code_bits=16,
    )
    monkeypatch.setattr(glm4_moe_adapter.nax, "is_available", lambda: True)
    assert switch_mlp_e8p._sorted_prefill_implementation(**prefill) == "nax_e8p_m32n64"

    switch_mlp_e8p.up_proj.code_bits = 8
    assert switch_mlp_e8p._sorted_prefill_implementation(**prefill) == "metal"

    switch_mlp_e8p.prefill_engine = "nax_e8p"
    assert switch_mlp_e8p._sorted_prefill_implementation(**prefill) == "nax_e8p"


def _e8p_prefill_switch(rng: np.random.Generator) -> QuantizedVQSwitchGLU:
    return QuantizedVQSwitchGLU.from_weights(
        gate_weight=mx.array(rng.normal(scale=0.03, size=(4, 8, 16)).astype(np.float32)),
        up_weight=mx.array(rng.normal(scale=0.03, size=(4, 8, 16)).astype(np.float32)),
        down_weight=mx.array(rng.normal(scale=0.03, size=(4, 16, 8)).astype(np.float32)),
        group_size=8,
        code_bits=16,
    )


def test_e8p_auto_prefill_production_path_selects_m32n64(monkeypatch) -> None:
    rng = np.random.default_rng(20260819)
    switch_mlp = _e8p_prefill_switch(rng)
    x = mx.array(rng.normal(size=(32, 16)).astype(np.float32)).astype(mx.bfloat16)
    indices = mx.zeros((32, 2), dtype=mx.int32)
    calls: list[tuple[str, int]] = []

    def recording_helper(x_arg, codes, scales, codebook, sorted_rhs, sorted_lhs, **kwargs):
        del codes, scales, codebook, sorted_lhs
        descriptors = kwargs["tile_descriptors"]
        calls.append(
            (
                kwargs["implementation"],
                -1 if descriptors is None else int(mx.max(descriptors[2]).item()),
            )
        )
        return mx.zeros((sorted_rhs.shape[0], kwargs["output_dims"]), dtype=x_arg.dtype)

    monkeypatch.setattr(glm4_moe_adapter.nax, "is_available", lambda: True)
    monkeypatch.setattr(glm4_moe_adapter, "gather_vqmm_sorted_routes", recording_helper)

    actual = switch_mlp(x, indices)
    mx.eval(actual)

    assert actual.shape == (32, 2, 16)
    assert calls == [("nax_e8p_m32n64", 32)] * 3


def test_e8p_auto_prefill_production_path_warns_and_falls_back_without_native(
    monkeypatch,
) -> None:
    rng = np.random.default_rng(20260820)
    switch_mlp = _e8p_prefill_switch(rng)
    x = mx.array(rng.normal(size=(32, 16)).astype(np.float32)).astype(mx.bfloat16)
    indices = mx.array(rng.integers(0, 4, size=(32, 2), dtype=np.int32))
    implementations: list[str] = []

    def recording_helper(x_arg, codes, scales, codebook, sorted_rhs, sorted_lhs, **kwargs):
        del codes, scales, codebook, sorted_lhs
        implementations.append(kwargs["implementation"])
        return mx.zeros((sorted_rhs.shape[0], kwargs["output_dims"]), dtype=x_arg.dtype)

    monkeypatch.setattr(glm4_moe_adapter.nax, "is_available", lambda: False)
    monkeypatch.setattr(glm4_moe_adapter, "gather_vqmm_sorted_routes", recording_helper)

    with pytest.warns(
        RuntimeWarning,
        match="native VQ NAX extension is unavailable.*E8P fast path is not being used",
    ):
        actual = switch_mlp(x, indices)
    mx.eval(actual)

    assert actual.shape == (32, 2, 16)
    assert implementations == ["metal"] * 3


def test_quantized_vq_switch_glu_honors_direct_route_strategy(monkeypatch) -> None:
    rng = np.random.default_rng(20260629)
    num_experts = 4
    input_dims = 16
    hidden_dims = 8

    def direct_layer(weight_shape: tuple[int, int, int]) -> QuantizedVQSwitchLinear:
        return QuantizedVQSwitchLinear.from_weights(
            mx.array(rng.normal(scale=0.03, size=weight_shape).astype(np.float32)),
            group_size=8,
            route_strategy="direct",
        )

    switch_mlp = QuantizedVQSwitchGLU(
        gate_proj=direct_layer((num_experts, hidden_dims, input_dims)),
        up_proj=direct_layer((num_experts, hidden_dims, input_dims)),
        down_proj=direct_layer((num_experts, input_dims, hidden_dims)),
    )
    x = rng.normal(size=(32, input_dims)).astype(np.float32)
    indices = rng.integers(0, num_experts, size=(32, 2), dtype=np.int32)

    def fail_shared_helper(*_args, **_kwargs):
        raise AssertionError("direct route_strategy must not use shared sorted GLU path")

    monkeypatch.setattr(glm4_moe_adapter, "gather_vqmm_sorted_routes", fail_shared_helper)

    actual = switch_mlp(mx.array(x), mx.array(indices))
    expected = _dense_switch_glu(x, indices, switch_mlp)
    mx.eval(actual)

    assert actual.shape == (32, 2, input_dims)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)


def test_glm4_moe_router_matches_installed_mlx_lm_group_selection() -> None:
    rng = np.random.default_rng(88)
    gates = mx.array(rng.normal(size=(6, 4)).astype(np.float32))
    correction = mx.array([0.0, 0.1, -0.2, 0.3], dtype=mx.float32)

    actual_inds, actual_scores = group_expert_select(gates, correction, 2, 2, 1, 1.7, True)

    from mlx_lm.models.glm4_moe import group_expert_select as mlx_lm_group_expert_select

    expected_inds, expected_scores = mlx_lm_group_expert_select(gates, correction, 2, 2, 1, 1.7, True)
    mx.eval(actual_inds, actual_scores, expected_inds, expected_scores)

    np.testing.assert_array_equal(np.array(actual_inds), np.array(expected_inds))
    np.testing.assert_allclose(np.array(actual_scores), np.array(expected_scores), rtol=1e-6, atol=1e-6)


def test_quantized_vq_glm4_moe_matches_dense_dequant_adapter() -> None:
    rng = np.random.default_rng(99)
    config = GLM4MoeRoutingConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.25,
    )
    switch_mlp = QuantizedVQSwitchGLU.from_weights(
        gate_weight=mx.array(rng.normal(scale=0.03, size=(4, 8, 16)).astype(np.float32)),
        up_weight=mx.array(rng.normal(scale=0.03, size=(4, 8, 16)).astype(np.float32)),
        down_weight=mx.array(rng.normal(scale=0.03, size=(4, 16, 8)).astype(np.float32)),
        group_size=8,
    )
    gate_weight = mx.array(rng.normal(scale=0.05, size=(4, 16)).astype(np.float32))
    correction = mx.array(rng.normal(scale=0.02, size=(4,)).astype(np.float32))
    x = rng.normal(size=(2, 3, 16)).astype(np.float32)

    moe = QuantizedVQGLM4MoE(
        config,
        switch_mlp=switch_mlp,
        gate_weight=gate_weight,
        e_score_correction_bias=correction,
    )
    inds, scores = moe.gate(mx.array(x))
    expected_routes = _dense_switch_glu(x, np.array(inds), switch_mlp)
    expected = (expected_routes * np.array(scores)[..., None]).sum(axis=-2)

    actual = moe(mx.array(x))
    mx.eval(actual, inds, scores)

    assert actual.shape == (2, 3, 16)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)
    assert cosine_similarity(np.array(actual), expected) >= 0.99999


def test_switch_runtime_path_does_not_call_dense_decode() -> None:
    assert "decode_weight_matrix" not in inspect.getsource(vq_switch_qmv)
    assert "decode_weight_matrix" not in inspect.getsource(QuantizedVQSwitchLinear.__call__)


def test_token_level_switch_linear_calls_fused_gather(monkeypatch) -> None:
    rng = np.random.default_rng(20260624)
    weight = mx.array(rng.normal(scale=0.04, size=(3, 8, 16)).astype(np.float32))
    layer = QuantizedVQSwitchLinear.from_weights(weight, group_size=8, use_gather_vqmm=True)
    x = mx.array(rng.normal(size=(4, 16)).astype(np.float32))
    indices = mx.array(np.array([[0, 2], [1, 1], [2, 0], [0, 1]], dtype=np.int32))
    calls: list[dict[str, object]] = []

    def fake_gather_vqmm(x_arg, codes, scales, codebook, rhs_indices, **kwargs):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "rhs_shape": rhs_indices.shape,
                "kwargs": kwargs,
                "codes_shape": codes.shape,
                "scales_shape": scales.shape,
                "codebook_shape": codebook.shape,
            }
        )
        return mx.zeros((x_arg.shape[0], rhs_indices.shape[1], layer.output_dims), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.nn.switch_linear.gather_vqmm", fake_gather_vqmm)

    actual = layer(x, indices)
    mx.eval(actual)

    assert actual.shape == (4, 2, 8)
    assert calls == [
        {
            "x_shape": (4, 16),
            "rhs_shape": (4, 2),
            "kwargs": {
                "input_dims": 16,
                "output_dims": 8,
                "group_size": 8,
                "code_bits": 8,
                "implementation": "metal",
                "sorted_indices": False,
                "route_strategy": "auto",
                "codebook_duplication": 8,
            },
            "codes_shape": layer.codes.shape,
            "scales_shape": layer.scales.shape,
            "codebook_shape": layer.codebook.shape,
        }
    ]


def test_batched_token_level_switch_linear_flattens_into_fused_gather(monkeypatch) -> None:
    rng = np.random.default_rng(20260625)
    weight = mx.array(rng.normal(scale=0.04, size=(4, 8, 16)).astype(np.float32))
    layer = QuantizedVQSwitchLinear.from_weights(weight, group_size=8, use_gather_vqmm=True)
    x = mx.array(rng.normal(size=(2, 3, 16)).astype(np.float32))
    indices = mx.array(
        np.array([[[0, 1], [2, 3], [1, 1]], [[3, 0], [2, 0], [0, 2]]], dtype=np.int32)
    )
    calls: list[dict[str, object]] = []

    def fake_gather_vqmm(x_arg, codes, scales, codebook, rhs_indices, **kwargs):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "rhs_shape": rhs_indices.shape,
                "kwargs": kwargs,
            }
        )
        return mx.zeros((x_arg.shape[0], rhs_indices.shape[1], layer.output_dims), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.nn.switch_linear.gather_vqmm", fake_gather_vqmm)

    actual = layer(x, indices)
    mx.eval(actual)

    assert actual.shape == (2, 3, 2, 8)
    assert calls == [
        {
            "x_shape": (6, 16),
            "rhs_shape": (6, 2),
            "kwargs": {
                "input_dims": 16,
                "output_dims": 8,
                "group_size": 8,
                "code_bits": 8,
                "implementation": "metal",
                "sorted_indices": False,
                "route_strategy": "auto",
                "codebook_duplication": 8,
            },
        }
    ]


def test_per_route_switch_linear_uses_fused_lhs_when_enabled(monkeypatch) -> None:
    rng = np.random.default_rng(20260626)
    weight = mx.array(rng.normal(scale=0.04, size=(4, 8, 16)).astype(np.float32))
    layer = QuantizedVQSwitchLinear.from_weights(weight, group_size=8, use_gather_vqmm=True)
    x = mx.array(rng.normal(size=(2, 3, 16)).astype(np.float32))
    indices = mx.array(np.array([[0, 1, 2], [3, 2, 0]], dtype=np.int32))
    calls: list[dict[str, object]] = []

    def fake_gather_vqmm(x_arg, codes, scales, codebook, rhs_indices, **kwargs):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "rhs_shape": rhs_indices.shape,
                "lhs_shape": kwargs["lhs_indices"].shape,
                "input_dims": kwargs["input_dims"],
                "output_dims": kwargs["output_dims"],
            }
        )
        return mx.zeros((x_arg.shape[0], layer.output_dims), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.nn.switch_linear.gather_vqmm", fake_gather_vqmm)

    actual = layer(x, indices)
    mx.eval(actual)

    assert actual.shape == (2, 3, 8)
    assert calls == [
        {
            "x_shape": (6, 16),
            "rhs_shape": (6,),
            "lhs_shape": (6,),
            "input_dims": 16,
            "output_dims": 8,
        }
    ]


def test_m1_per_route_switch_linear_uses_fast_decode(monkeypatch) -> None:
    rng = np.random.default_rng(20260627)
    weight = mx.array(rng.normal(scale=0.04, size=(4, 8, 16)).astype(np.float32))
    layer = QuantizedVQSwitchLinear.from_weights(weight, group_size=8, use_gather_vqmm=True)
    x = mx.array(rng.normal(size=(1, 3, 16)).astype(np.float32))
    indices = mx.array(np.array([[0, 1, 2]], dtype=np.int32))
    calls: list[dict[str, object]] = []

    def fake_m1_per_route(x_arg, codes, scales, codebook, rhs_indices, **kwargs):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "rhs_shape": rhs_indices.shape,
                "output_dims": kwargs["output_dims"],
                "rows_per_threadgroup": kwargs["rows_per_threadgroup"],
            }
        )
        return mx.zeros((x_arg.shape[0], layer.output_dims), dtype=x_arg.dtype)

    monkeypatch.setattr(
        "ramp.nn.switch_linear.gather_vqmm_m1_per_route_kernel_unchecked",
        fake_m1_per_route,
    )

    actual = layer(x, indices)
    mx.eval(actual)

    assert actual.shape == (1, 3, 8)
    assert calls == [
        {
            "x_shape": (3, 16),
            "rhs_shape": (3,),
            "output_dims": 8,
            "rows_per_threadgroup": 32,
        }
    ]


def test_switch_path_reports_token_gather_backend_by_default() -> None:
    rng = np.random.default_rng(20260624)
    weight = mx.array(rng.normal(scale=0.04, size=(2, 8, 16)).astype(np.float32))
    layer = QuantizedVQSwitchLinear.from_weights(weight, group_size=8)

    assert layer.route_backend == "gather_vqmm_auto"


def test_switch_path_reports_scalar_backend_when_disabled() -> None:
    rng = np.random.default_rng(20260624)
    weight = mx.array(rng.normal(scale=0.04, size=(2, 8, 16)).astype(np.float32))
    layer = QuantizedVQSwitchLinear.from_weights(weight, group_size=8, use_gather_vqmm=False)

    assert layer.route_backend == "scalar_qmv"


def test_switch_linear_accepts_explicit_route_strategy() -> None:
    rng = np.random.default_rng(20260627)
    weight = mx.array(rng.normal(scale=0.04, size=(2, 8, 16)).astype(np.float32))
    layer = QuantizedVQSwitchLinear.from_weights(weight, group_size=8, route_strategy="sorted_tiled")

    assert layer.route_backend == "gather_vqmm_sorted_tiled"
