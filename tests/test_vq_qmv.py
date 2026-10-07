from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.codebook.e8 import cosine_similarity, decode_weight_matrix, e8_1bit_packed, e8p_packed_abs_grid
from mlx_vq.kernels.vq_qmv import vq_qmv, vq_qmv_reference_np


def _case(seed: int, *, out_dim: int, in_dim: int, group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(in_dim,)).astype(np.float32)
    codes = rng.integers(0, 256, size=(out_dim, in_dim // 8), dtype=np.uint8)
    scales = rng.uniform(0.85, 1.15, size=(out_dim, in_dim // group_size)).astype(np.float32)
    return x, codes, scales


@pytest.mark.parametrize(
    ("out_dim", "in_dim", "group_size"),
    [
        (3, 16, 8),
        (7, 64, 16),
        (8, 6144, 512),
    ],
)
def test_vq_qmv_reference_matches_dense_dequant(out_dim: int, in_dim: int, group_size: int) -> None:
    x, codes, scales = _case(1000 + out_dim + in_dim, out_dim=out_dim, in_dim=in_dim, group_size=group_size)

    actual = vq_qmv_reference_np(
        x,
        codes,
        scales,
        e8_1bit_packed(),
        in_dim=in_dim,
        out_dim=out_dim,
        group_size=group_size,
        code_bits=8,
    )
    expected = decode_weight_matrix(codes, scales, codebook=e8_1bit_packed()) @ x

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
    assert cosine_similarity(actual, expected) >= 0.99999


def test_vq_qmv_accepts_mlx_arrays_and_returns_mlx_array() -> None:
    x, codes, scales = _case(20260623, out_dim=5, in_dim=128, group_size=32)
    expected = vq_qmv_reference_np(
        x,
        codes,
        scales,
        in_dim=128,
        out_dim=5,
        group_size=32,
        code_bits=8,
    )

    actual = vq_qmv(
        mx.array(x),
        mx.array(codes),
        mx.array(scales),
        mx.array(e8_1bit_packed()),
        in_dim=128,
        out_dim=5,
        group_size=32,
        code_bits=8,
    )
    mx.eval(actual)

    assert isinstance(actual, mx.array)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-6, atol=1e-5)


def test_vq_qmv_metal_matches_reference_for_e8p_16bit_codes() -> None:
    rng = np.random.default_rng(4242)
    in_dim = 32
    out_dim = 4
    group_size = 16
    x = rng.normal(size=(in_dim,)).astype(np.float32)
    codes = rng.integers(0, 65536, size=(out_dim, in_dim // 8), dtype=np.uint16)
    scales = rng.uniform(0.9, 1.1, size=(out_dim, in_dim // group_size)).astype(np.float32)

    actual = vq_qmv(
        mx.array(x),
        mx.array(codes),
        mx.array(scales),
        mx.array(e8p_packed_abs_grid()),
        in_dim=in_dim,
        out_dim=out_dim,
        group_size=group_size,
        code_bits=16,
    )
    expected = decode_weight_matrix(codes, scales, code_bits=16, codebook=e8p_packed_abs_grid()) @ x
    mx.eval(actual)

    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-6, atol=1e-5)
    assert cosine_similarity(np.array(actual), expected) >= 0.99999


def test_vq_qmv_writes_float16_for_float16_activation() -> None:
    x, codes, scales = _case(888, out_dim=6, in_dim=256, group_size=64)
    x16 = x.astype(np.float16)
    scales16 = scales.astype(np.float16)

    actual = vq_qmv(
        mx.array(x16),
        mx.array(codes),
        mx.array(scales16),
        in_dim=256,
        out_dim=6,
        group_size=64,
        code_bits=8,
    )
    expected = vq_qmv_reference_np(
        x16,
        codes,
        scales16,
        in_dim=256,
        out_dim=6,
        group_size=64,
        code_bits=8,
    ).astype(np.float16)
    mx.eval(actual)

    assert actual.dtype == mx.float16
    np.testing.assert_allclose(np.array(actual), expected, rtol=0.0, atol=0.03125)


def test_vq_qmv_writes_bfloat16_for_bfloat16_activation() -> None:
    x, codes, scales = _case(889, out_dim=4, in_dim=128, group_size=64)
    actual = vq_qmv(
        mx.array(x).astype(mx.bfloat16),
        mx.array(codes),
        mx.array(scales).astype(mx.bfloat16),
        in_dim=128,
        out_dim=4,
        group_size=64,
        code_bits=8,
    )
    mx.eval(actual)
    assert actual.dtype == mx.bfloat16


def test_vq_qmv_rejects_bad_shapes() -> None:
    x, codes, scales = _case(7, out_dim=2, in_dim=32, group_size=16)
    with pytest.raises(ValueError, match="codes must have shape"):
        vq_qmv_reference_np(
            x,
            codes[:, :-1],
            scales,
            in_dim=32,
            out_dim=2,
            group_size=16,
            code_bits=8,
        )
