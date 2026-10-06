from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.codebook.e8 import e8_1bit_packed
from mlx_vq.kernels.vq_qmv import vq_qmv_reference_np
from mlx_vq.nn.linear import QuantizedVQLinear
from mlx_vq.nn.switch_linear import QuantizedVQSwitchLinear
from mlx_vq.quant.rotation import (
    apply_rotation_np,
    cayley_update_np,
    validate_rotation_matrix_np,
)
from mlx_vq.quant.rtn import quantize_weight_rtn
from mlx_vq.quant.rht import deterministic_rht_signs
from mlx_vq.quality.learned_rotation_training import train_learned_rht_signs_np, train_learned_rotation_np


def _orthogonal_matrix(dim: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)).astype(np.float32))
    signs = np.sign(np.diag(r)).astype(np.float32)
    signs[signs == 0] = 1.0
    return (q * signs[None, :]).astype(np.float32)


def test_cayley_update_preserves_orthogonality() -> None:
    rng = np.random.default_rng(20260707)
    rotation = _orthogonal_matrix(16, seed=7)
    gradient = rng.normal(size=(16, 16)).astype(np.float32)

    updated = cayley_update_np(rotation, gradient, learning_rate=0.05)

    assert updated.dtype == np.float32
    np.testing.assert_allclose(updated.T @ updated, np.eye(16), rtol=2e-5, atol=2e-5)
    assert np.linalg.norm(updated - rotation) > 0.0


def test_validate_rotation_matrix_rejects_non_orthogonal_input() -> None:
    bad = np.eye(8, dtype=np.float32)
    bad[0, 0] = 2.0

    with pytest.raises(ValueError, match="orthogonal"):
        validate_rotation_matrix_np(bad, dim=8)


def test_quantized_vq_linear_applies_learned_rotation_to_runtime_inputs() -> None:
    rng = np.random.default_rng(12345)
    weight = rng.normal(scale=0.08, size=(12, 32)).astype(np.float32)
    x = rng.normal(size=(3, 32)).astype(np.float32)
    rotation = _orthogonal_matrix(32, seed=11)

    layer = QuantizedVQLinear.from_weights(
        mx.array(weight),
        group_size=8,
        rotation_matrix=mx.array(rotation),
    )
    actual = layer(mx.array(x))
    mx.eval(actual)

    transformed_weight = apply_rotation_np(weight, rotation)
    quantized = quantize_weight_rtn(transformed_weight, group_size=8)
    transformed_x = apply_rotation_np(x, rotation)
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

    assert layer.get("rotation_matrix") is not None
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=8e-4)


def test_quantized_vq_switch_linear_applies_learned_rotation_before_selected_experts() -> None:
    rng = np.random.default_rng(90210)
    weight = rng.normal(scale=0.05, size=(3, 10, 32)).astype(np.float32)
    x = rng.normal(size=(2, 32)).astype(np.float32)
    indices = np.array([[0, 2], [1, 0]], dtype=np.int32)
    rotation = _orthogonal_matrix(32, seed=13)

    layer = QuantizedVQSwitchLinear.from_weights(
        mx.array(weight),
        group_size=8,
        use_gather_vqmm=False,
        rotation_matrix=mx.array(rotation),
    )
    actual = layer(mx.array(x), mx.array(indices))
    mx.eval(actual)

    transformed_weight = apply_rotation_np(weight, rotation)
    quantized = [
        quantize_weight_rtn(expert_weight, group_size=8)
        for expert_weight in transformed_weight
    ]
    transformed_x = apply_rotation_np(x, rotation)
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

    assert layer.get("rotation_matrix") is not None
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=8e-4)


def test_dense_learned_rotation_is_mergeable_before_quantization() -> None:
    rng = np.random.default_rng(20260708)
    weight = rng.normal(scale=0.04, size=(3, 12, 32)).astype(np.float32)
    x = rng.normal(size=(5, 32)).astype(np.float32)
    rotation = _orthogonal_matrix(32, seed=17)

    original = np.einsum("ti,eoi->teo", x, weight)
    rotated = np.einsum(
        "ti,eoi->teo",
        apply_rotation_np(x, rotation),
        apply_rotation_np(weight, rotation),
    )

    np.testing.assert_allclose(rotated, original, rtol=2e-5, atol=2e-5)


def test_train_learned_rotation_reduces_quantized_reconstruction_loss() -> None:
    rng = np.random.default_rng(20260709)
    weight = rng.normal(scale=0.4, size=(96, 32)).astype(np.float32)

    result = train_learned_rotation_np(
        weight,
        group_size=8,
        steps=12,
        learning_rate=0.2,
    )

    np.testing.assert_allclose(result.rotation_matrix.T @ result.rotation_matrix, np.eye(32), atol=4e-5)
    assert result.final_loss < result.initial_loss


def test_train_learned_rht_signs_reduces_quantized_reconstruction_loss() -> None:
    rng = np.random.default_rng(20260710)
    weight = rng.normal(scale=0.4, size=(48, 32)).astype(np.float32)
    initial_signs = deterministic_rht_signs(32, seed="structured-learned-rotation-test")

    result = train_learned_rht_signs_np(
        weight,
        group_size=8,
        steps=6,
        candidates_per_step=12,
        initial_signs=initial_signs,
        rng=rng,
    )

    assert set(np.unique(result.rht_signs)).issubset({-1, 1})
    assert result.final_loss <= result.initial_loss
    assert result.accepted_flips > 0
