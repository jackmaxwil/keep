from __future__ import annotations

import mlx.core as mx
import numpy as np
import importlib.util
from pathlib import Path

from mlx_vq.codebook.e8 import decode_weight_matrix, e8_1bit_packed
from mlx_vq.nn.switch_linear import QuantizedVQSwitchLinear
from mlx_vq.quant.rht import apply_rht_np, deterministic_rht_signs
from mlx_vq.quality.mlx_surrogate import (
    RouteLocalSwitchLinearSurrogate,
    SwitchLinearSidecar,
    SwitchLinearSurrogate,
    call_switch_linear_with_optional_sidecar,
    fit_low_rank_residual_sidecar_least_squares,
    fit_scale_delta_output_bias_sidecar_least_squares,
    fit_output_bias_sidecar_least_squares,
    fit_scale_delta_sidecar_least_squares,
)


def _load_probe_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "probe_glm45_air_mlx_surrogate_parity.py"
    )
    spec = importlib.util.spec_from_file_location("mlx_surrogate_probe_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_switch_layer() -> QuantizedVQSwitchLinear:
    rng = np.random.default_rng(20260626)
    codes = rng.integers(0, 256, size=(3, 5, 4), dtype=np.uint8)
    scales = rng.uniform(0.25, 0.75, size=(3, 5, 2)).astype(np.float32)
    bias = rng.normal(scale=0.01, size=(3, 5)).astype(np.float32)
    return QuantizedVQSwitchLinear(
        input_dims=32,
        output_dims=5,
        num_experts=3,
        codes=mx.array(codes),
        scales=mx.array(scales),
        codebook=mx.array(e8_1bit_packed()),
        bias=mx.array(bias),
        group_size=16,
        use_gather_vqmm=False,
    )


def _tiny_rht_switch_layer() -> QuantizedVQSwitchLinear:
    rng = np.random.default_rng(20260701)
    weight = rng.normal(scale=0.04, size=(3, 5, 32)).astype(np.float32)
    signs = deterministic_rht_signs(32, seed="surrogate-rht")
    return QuantizedVQSwitchLinear.from_weights(
        mx.array(weight),
        group_size=8,
        use_gather_vqmm=False,
        rht_signs=mx.array(signs),
    )


def _reference_switch_output(
    layer: QuantizedVQSwitchLinear,
    x: np.ndarray,
    indices: np.ndarray,
    *,
    scale_delta: np.ndarray | None = None,
    output_bias: np.ndarray | None = None,
) -> np.ndarray:
    signs = layer.get("rht_signs")
    if signs is not None:
        x = apply_rht_np(x, np.array(signs, dtype=np.int8))
    scales = np.array(layer.scales, dtype=np.float32)
    if scale_delta is not None:
        scales = scales * np.exp(scale_delta.astype(np.float32))
    weights = np.stack(
        [
            decode_weight_matrix(
                np.array(layer.codes[expert]),
                scales[expert],
                code_bits=layer.code_bits,
                codebook=np.array(layer.codebook),
            )
            for expert in range(layer.num_experts)
        ],
        axis=0,
    )
    bias_value = layer.get("bias")
    bias = None if bias_value is None else np.array(bias_value, dtype=np.float32)
    bias_delta = None if output_bias is None else np.asarray(output_bias, dtype=np.float32)
    out = np.empty((*indices.shape, layer.output_dims), dtype=np.float32)
    for token_idx in range(indices.shape[0]):
        for route_idx in range(indices.shape[1]):
            expert = int(indices[token_idx, route_idx])
            out[token_idx, route_idx] = x[token_idx] @ weights[expert].T
            if bias is not None:
                out[token_idx, route_idx] += bias[expert]
            if bias_delta is not None:
                out[token_idx, route_idx] += bias_delta[expert]
    return out


def test_no_sidecar_surrogate_equals_decoded_reference() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, layer.input_dims)).astype(np.float32)
    indices = np.array([[0, 1], [2, 0], [1, 2], [0, 0]], dtype=np.int32)

    surrogate = SwitchLinearSurrogate.from_layer(layer)
    actual = surrogate(mx.array(x), mx.array(indices))
    mx.eval(actual)

    np.testing.assert_allclose(
        np.array(actual),
        _reference_switch_output(layer, x, indices),
        rtol=1e-4,
        atol=1e-5,
    )


def test_route_local_surrogate_matches_full_surrogate_with_chunks() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(20260707)
    x = rng.normal(size=(5, layer.input_dims)).astype(np.float32)
    indices = np.array([[0, 1], [2, 0], [1, 2], [0, 0], [2, 1]], dtype=np.int32)
    output_bias = rng.normal(scale=0.02, size=(layer.num_experts, layer.output_dims)).astype(np.float32)
    scale_delta = rng.normal(scale=0.01, size=layer.scales.shape).astype(np.float32)
    sidecar = SwitchLinearSidecar(
        scale_delta=mx.array(scale_delta),
        output_bias=mx.array(output_bias),
    )

    expected = SwitchLinearSurrogate.from_layer(layer, sidecar=sidecar)(mx.array(x), mx.array(indices))
    actual = RouteLocalSwitchLinearSurrogate.from_layer(
        layer,
        sidecar=sidecar,
        output_chunk_size=2,
    )(mx.array(x), mx.array(indices))
    mx.eval(expected, actual)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-4, atol=1e-5)


def test_route_local_surrogate_supports_per_route_inputs() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(20260708)
    x = rng.normal(size=(5, 2, layer.input_dims)).astype(np.float32)
    indices = np.array([[0, 1], [2, 0], [1, 2], [0, 0], [2, 1]], dtype=np.int32)

    expected = SwitchLinearSurrogate.from_layer(layer)(mx.array(x), mx.array(indices))
    actual = RouteLocalSwitchLinearSurrogate.from_layer(layer, output_chunk_size=3)(mx.array(x), mx.array(indices))
    mx.eval(expected, actual)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-4, atol=1e-5)


def test_rht_surrogate_equals_decoded_reference_and_runtime() -> None:
    layer = _tiny_rht_switch_layer()
    rng = np.random.default_rng(20260702)
    x = rng.normal(size=(4, layer.input_dims)).astype(np.float32)
    indices = np.array([[0, 1], [2, 0], [1, 2], [0, 0]], dtype=np.int32)

    surrogate = SwitchLinearSurrogate.from_layer(layer)
    surrogate_output = surrogate(mx.array(x), mx.array(indices))
    runtime_output = layer(mx.array(x), mx.array(indices))
    mx.eval(surrogate_output, runtime_output)
    reference = _reference_switch_output(layer, x, indices)

    np.testing.assert_allclose(np.array(surrogate_output), reference, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.array(runtime_output), reference, rtol=1e-5, atol=2e-4)


def test_fit_output_bias_sidecar_least_squares_reduces_block_local_residual() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(20260703)
    x = rng.normal(size=(6, layer.input_dims)).astype(np.float32)
    indices = np.array([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1]], dtype=np.int32)
    true_bias = np.array(
        [
            [0.3, -0.1, 0.05, 0.0, 0.2],
            [-0.2, 0.15, 0.0, 0.25, -0.05],
            [0.1, 0.0, -0.15, -0.2, 0.1],
        ],
        dtype=np.float32,
    )
    baseline = SwitchLinearSurrogate.from_layer(layer)(mx.array(x), mx.array(indices))
    target = baseline + mx.array(true_bias[indices])

    result = fit_output_bias_sidecar_least_squares(layer, mx.array(x), mx.array(indices), target)

    assert result["ok"] is True
    assert result["baseline_mse"] > 0.0
    assert result["fitted_mse"] < 1.0e-10
    assert result["improvement_ratio"] < 1.0e-6
    np.testing.assert_allclose(np.array(result["sidecar"].output_bias), true_bias, rtol=1e-5, atol=1e-5)


def test_fit_low_rank_residual_sidecar_least_squares_recovers_rank1_residual() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(20260709)
    x = rng.normal(size=(18, layer.input_dims)).astype(np.float32)
    indices = np.array([[0], [1], [2]] * 6, dtype=np.int32)
    true_left = rng.normal(scale=0.2, size=(layer.num_experts, layer.output_dims, 1)).astype(np.float32)
    true_right = rng.normal(scale=0.2, size=(layer.num_experts, 1, layer.input_dims)).astype(np.float32)
    target = SwitchLinearSurrogate.from_layer(
        layer,
        sidecar=SwitchLinearSidecar(
            low_rank_left=mx.array(true_left),
            low_rank_right=mx.array(true_right),
        ),
    )(mx.array(x), mx.array(indices))

    result = fit_low_rank_residual_sidecar_least_squares(
        layer,
        mx.array(x),
        mx.array(indices),
        target,
        rank=1,
    )
    fitted = SwitchLinearSurrogate.from_layer(layer, sidecar=result["sidecar"])(mx.array(x), mx.array(indices))
    mx.eval(fitted, target)

    assert result["ok"] is True
    assert result["baseline_mse"] > 0.0
    assert result["fitted_mse"] < 1.0e-10
    assert result["improvement_ratio"] < 1.0e-6
    assert result["sidecar"].low_rank_left.shape == true_left.shape
    assert result["sidecar"].low_rank_right.shape == true_right.shape
    np.testing.assert_allclose(np.array(fitted), np.array(target), rtol=1e-5, atol=1e-5)


def test_fit_scale_delta_sidecar_least_squares_recovers_positive_group_scales() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(20260704)
    x = rng.normal(size=(18, layer.input_dims)).astype(np.float32)
    indices = np.array([[0], [1], [2]] * 6, dtype=np.int32)
    true_delta = np.zeros(layer.scales.shape, dtype=np.float32)
    true_delta[:, :, 0] = 0.20
    true_delta[:, :, 1] = -0.15
    target = SwitchLinearSurrogate.from_layer(
        layer,
        sidecar=SwitchLinearSidecar(scale_delta=mx.array(true_delta)),
    )(mx.array(x), mx.array(indices))

    result = fit_scale_delta_sidecar_least_squares(layer, mx.array(x), mx.array(indices), target)

    assert result["ok"] is True
    assert result["baseline_mse"] > 0.0
    assert result["fitted_mse"] < 1.0e-10
    assert result["improvement_ratio"] < 1.0e-6
    np.testing.assert_allclose(np.array(result["sidecar"].scale_delta), true_delta, rtol=1e-4, atol=1e-4)


def test_fit_scale_delta_sidecar_least_squares_handles_multiple_routes_per_token() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(20260705)
    x = rng.normal(size=(12, layer.input_dims)).astype(np.float32)
    indices = np.array([[0, 1], [1, 2], [2, 0]] * 4, dtype=np.int32)
    true_delta = np.zeros(layer.scales.shape, dtype=np.float32)
    true_delta[:, :, 0] = 0.05
    true_delta[:, :, 1] = 0.10
    target = SwitchLinearSurrogate.from_layer(
        layer,
        sidecar=SwitchLinearSidecar(scale_delta=mx.array(true_delta)),
    )(mx.array(x), mx.array(indices))

    result = fit_scale_delta_sidecar_least_squares(layer, mx.array(x), mx.array(indices), target)

    assert result["ok"] is True
    assert result["fitted_mse"] < 1.0e-10
    assert result["improvement_ratio"] < 1.0e-6


def test_fit_scale_delta_output_bias_sidecar_least_squares_recovers_combined_residual() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(20260706)
    x = rng.normal(size=(18, layer.input_dims)).astype(np.float32)
    indices = np.array([[0, 1], [1, 2], [2, 0]] * 6, dtype=np.int32)
    true_delta = np.zeros(layer.scales.shape, dtype=np.float32)
    true_delta[:, :, 0] = 0.06
    true_delta[:, :, 1] = -0.04
    true_bias = np.array(
        [
            [0.03, -0.01, 0.02, 0.00, 0.04],
            [-0.02, 0.01, 0.00, 0.05, -0.01],
            [0.01, 0.00, -0.03, -0.02, 0.02],
        ],
        dtype=np.float32,
    )
    target = SwitchLinearSurrogate.from_layer(
        layer,
        sidecar=SwitchLinearSidecar(
            scale_delta=mx.array(true_delta),
            output_bias=mx.array(true_bias),
        ),
    )(mx.array(x), mx.array(indices))

    result = fit_scale_delta_output_bias_sidecar_least_squares(layer, mx.array(x), mx.array(indices), target)

    assert result["ok"] is True
    assert result["baseline_mse"] > 0.0
    assert result["scale_delta_mse"] < result["baseline_mse"]
    assert result["fitted_mse"] < result["scale_delta_mse"]
    assert result["improvement_ratio"] < 1.0
    assert result["sidecar"].scale_delta.shape == true_delta.shape
    assert result["sidecar"].output_bias.shape == true_bias.shape


def test_no_sidecar_runtime_call_remains_current_layer_path() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(8)
    x = mx.array(rng.normal(size=(2, layer.input_dims)).astype(np.float32))
    indices = mx.array(np.array([[0, 1], [2, 2]], dtype=np.int32))

    current = layer(x, indices)
    optional = call_switch_linear_with_optional_sidecar(layer, x, indices, sidecar=None)
    mx.eval(current, optional)

    np.testing.assert_allclose(np.array(optional), np.array(current), rtol=0.0, atol=0.0)


def test_sidecar_scale_delta_matches_decoded_reference() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3, layer.input_dims)).astype(np.float32)
    indices = np.array([[0, 2], [2, 1], [1, 0]], dtype=np.int32)
    scale_delta = np.zeros(layer.scales.shape, dtype=np.float32)
    scale_delta[2, :, 1] = 0.125
    sidecar = SwitchLinearSidecar(scale_delta=mx.array(scale_delta))

    surrogate = SwitchLinearSurrogate.from_layer(layer, sidecar=sidecar)
    actual = surrogate(mx.array(x), mx.array(indices))
    mx.eval(actual)

    np.testing.assert_allclose(
        np.array(actual),
        _reference_switch_output(layer, x, indices, scale_delta=scale_delta),
        rtol=1e-4,
        atol=1e-5,
    )


def test_runtime_output_bias_sidecar_matches_reference() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(11)
    x = rng.normal(size=(3, layer.input_dims)).astype(np.float32)
    indices = np.array([[0, 2], [2, 1], [1, 0]], dtype=np.int32)
    output_bias = rng.normal(scale=0.01, size=(layer.num_experts, layer.output_dims)).astype(np.float32)

    layer.set_continuous_sidecar(output_bias=mx.array(output_bias))
    actual = layer(mx.array(x), mx.array(indices))
    mx.eval(actual)

    np.testing.assert_allclose(
        np.array(actual),
        _reference_switch_output(layer, x, indices, output_bias=output_bias),
        rtol=1e-4,
        atol=1e-5,
    )


def test_scale_delta_has_mlx_gradients() -> None:
    layer = _tiny_switch_layer()
    rng = np.random.default_rng(10)
    x = mx.array(rng.normal(size=(2, layer.input_dims)).astype(np.float32))
    indices = mx.array(np.array([[0, 1], [2, 0]], dtype=np.int32))

    def loss_fn(scale_delta: mx.array) -> mx.array:
        sidecar = SwitchLinearSidecar(scale_delta=scale_delta)
        output = SwitchLinearSurrogate.from_layer(layer, sidecar=sidecar)(x, indices)
        return mx.sum(output * output)

    loss_value, grad = mx.value_and_grad(loss_fn)(mx.zeros(layer.scales.shape, dtype=mx.float32))
    mx.eval(loss_value, grad)

    assert grad.shape == layer.scales.shape
    assert float(mx.sum(mx.abs(grad)).item()) > 0.0


def test_probe_tolerance_uses_allclose_semantics_for_near_zero_values() -> None:
    probe_cli = _load_probe_cli()
    actual = np.array([1.0e-7, 1.0], dtype=np.float32)
    expected = np.array([0.0, 1.0], dtype=np.float32)

    assert probe_cli._within_tolerance(actual, expected, atol=1.0e-6, rtol=1.0e-4) is True
