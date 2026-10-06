from __future__ import annotations

import numpy as np
import pytest

from mlx_vq.codebook.e8 import cosine_similarity, e8p_full_grid
from mlx_vq.quant.rtn import (
    dequantize_weight_np,
    nearest_codebook_indices_diagonal_hessian,
    quantize_weight_rtn,
)


def _require_mlx_runtime():
    try:
        import mlx.core as mx
        from mlx_vq.nn.linear import QuantizedVQLinear

        probe = mx.array([0.0], dtype=mx.float32)
        mx.eval(probe)
    except RuntimeError as error:
        if "No Metal device available" not in str(error):
            raise
        pytest.skip("MLX runtime is unavailable: no Metal device")
    return mx, QuantizedVQLinear


def test_quantize_weight_rtn_shapes_and_zero_group() -> None:
    weight = np.zeros((3, 16), dtype=np.float32)
    quantized = quantize_weight_rtn(weight, group_size=8)

    assert quantized.codes.shape == (3, 2)
    assert quantized.codes.dtype == np.uint8
    assert quantized.scales.shape == (3, 2)
    assert np.all(quantized.scales == 0)
    np.testing.assert_array_equal(dequantize_weight_np(quantized), np.zeros_like(weight))


def test_quantize_weight_rtn_default_scale_estimator_matches_max_abs_formula() -> None:
    rng = np.random.default_rng(1234)
    weight = rng.normal(scale=0.2, size=(4, 32)).astype(np.float32)

    default = quantize_weight_rtn(weight, group_size=16)
    explicit = quantize_weight_rtn(weight, group_size=16, scale_estimator="max_abs")
    expected_scales = (np.max(np.abs(weight.reshape(4, 2, 16)), axis=2) / 2).astype(np.float16)

    np.testing.assert_array_equal(default.codes, explicit.codes)
    np.testing.assert_array_equal(default.scales, explicit.scales)
    np.testing.assert_array_equal(default.scales, expected_scales)
    assert default.scale_estimator == "max_abs"


def test_quantize_weight_rtn_percentile_scale_estimator_is_explicit_variant() -> None:
    weight = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(2, 16)
    weight[0, 0] = 10.0

    baseline = quantize_weight_rtn(weight, group_size=16)
    variant = quantize_weight_rtn(weight, group_size=16, scale_estimator="percentile_99")

    assert variant.scale_estimator == "percentile_99"
    assert variant.codes.shape == baseline.codes.shape
    assert variant.scales.shape == baseline.scales.shape
    assert variant.scales[0, 0] < baseline.scales[0, 0]


def test_quantize_weight_rtn_rejects_unknown_scale_estimator() -> None:
    weight = np.zeros((1, 8), dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        quantize_weight_rtn(weight, group_size=8, scale_estimator="not-a-policy")  # type: ignore[arg-type]


def test_quantize_weight_rtn_supports_e8p_code_bits_16() -> None:
    rng = np.random.default_rng(20260625)
    weight = rng.normal(scale=0.2, size=(3, 16)).astype(np.float32)

    quantized = quantize_weight_rtn(weight, group_size=8, code_bits=16, codeword_chunk_size=4)
    dequantized = dequantize_weight_np(quantized)

    assert quantized.codes.shape == (3, 2)
    assert quantized.codes.dtype == np.uint16
    assert quantized.scales.shape == (3, 2)
    assert quantized.codebook.shape == (256,)
    assert quantized.code_bits == 16
    assert np.all(np.isfinite(dequantized))
    assert dequantized.shape == weight.shape


def test_diagonal_hessian_e8p_encoding_is_chunk_invariant_and_weighted() -> None:
    vectors = np.array(
        [[0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, 3.75]],
        dtype=np.float32,
    )
    diagonal = np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 0.001], dtype=np.float32)
    table = e8p_full_grid()

    small_chunks = nearest_codebook_indices_diagonal_hessian(
        vectors,
        diagonal,
        codebook=table,
        index_dtype=np.dtype(np.uint16),
        vector_chunk_size=1,
        codebook_chunk_size=17,
    )
    large_chunks = nearest_codebook_indices_diagonal_hessian(
        vectors,
        diagonal,
        codebook=table,
        index_dtype=np.dtype(np.uint16),
        vector_chunk_size=8,
        codebook_chunk_size=8192,
    )
    unweighted = np.argmin(np.sum((table - vectors[0]) ** 2, axis=1))

    np.testing.assert_array_equal(small_chunks, large_chunks)
    weighted_error = np.sum((table[int(small_chunks[0])] - vectors[0]) ** 2 * diagonal)
    unweighted_error = np.sum((table[int(unweighted)] - vectors[0]) ** 2 * diagonal)
    assert weighted_error < unweighted_error


def test_quantized_vq_linear_from_weights_matches_dense_dequant_reference() -> None:
    mx, QuantizedVQLinear = _require_mlx_runtime()
    rng = np.random.default_rng(20260624)
    weight = rng.normal(scale=0.05, size=(5, 32)).astype(np.float32)
    bias = rng.normal(scale=0.01, size=(5,)).astype(np.float32)
    x = rng.normal(size=(32,)).astype(np.float32)

    quantized = quantize_weight_rtn(weight, group_size=16)
    dense_dequant = dequantize_weight_np(quantized)
    layer = QuantizedVQLinear.from_weights(mx.array(weight), mx.array(bias), group_size=16)
    actual = layer(mx.array(x))
    expected = dense_dequant @ x + bias
    mx.eval(actual)

    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)
    assert cosine_similarity(np.array(actual), expected) >= 0.99999


def test_quantized_vq_linear_from_weights_accepts_half_precision_mlx_weight() -> None:
    mx, QuantizedVQLinear = _require_mlx_runtime()
    rng = np.random.default_rng(77)
    weight = mx.array(rng.normal(scale=0.05, size=(4, 16)).astype(np.float16))
    layer = QuantizedVQLinear.from_weights(weight, group_size=8)

    assert layer.input_dims == 16
    assert layer.output_dims == 4
    assert layer.codes.dtype == mx.uint8


def test_quantized_vq_linear_from_weights_accepts_e8p_code_bits_16() -> None:
    mx, QuantizedVQLinear = _require_mlx_runtime()
    rng = np.random.default_rng(20260625)
    weight = mx.array(rng.normal(scale=0.05, size=(4, 16)).astype(np.float16))
    x = mx.array(rng.normal(size=(16,)).astype(np.float16))

    layer = QuantizedVQLinear.from_weights(weight, group_size=8, code_bits=16)
    actual = layer(x)
    mx.eval(actual)

    assert layer.input_dims == 16
    assert layer.output_dims == 4
    assert layer.codes.dtype == mx.uint16
    assert layer.code_bits == 16
    assert bool(mx.all(mx.isfinite(actual.astype(mx.float32))).item())
