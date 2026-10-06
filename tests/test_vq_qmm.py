from __future__ import annotations

import mlx.core as mx
import numpy as np

import mlx_vq.nn.linear as linear_module
from mlx_vq.codebook.e8 import cosine_similarity, e8_1bit_packed
from mlx_vq.kernels.vq_qmm import vq_qmm, vq_qmm_reference_np
from mlx_vq.nn.linear import QuantizedVQLinear


def _fixture(*, m: int, in_dim: int = 64, out_dim: int = 32, group_size: int = 16, seed: int = 123):
    rng = np.random.default_rng(seed + m)
    x = rng.normal(size=(m, in_dim)).astype(np.float32)
    codes = rng.integers(0, 256, size=(out_dim, in_dim // 8), dtype=np.uint8)
    scales = rng.uniform(0.25, 0.75, size=(out_dim, in_dim // group_size)).astype(np.float16)
    return x, codes, scales, e8_1bit_packed()


def test_vq_qmm_prefill_matches_dense_dequant_reference_for_required_m_values() -> None:
    for m in (8, 32, 128, 512):
        x, codes, scales, codebook = _fixture(m=m)
        actual = vq_qmm(
            mx.array(x),
            mx.array(codes),
            mx.array(scales),
            mx.array(codebook),
            in_dim=64,
            out_dim=32,
            group_size=16,
            code_bits=8,
        )
        expected = vq_qmm_reference_np(
            x,
            codes,
            scales,
            codebook,
            in_dim=64,
            out_dim=32,
            group_size=16,
            code_bits=8,
        )
        mx.eval(actual)

        np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-4)
        assert cosine_similarity(np.array(actual), expected) >= 0.99999


def test_vq_qmm_preserves_half_precision_output_dtype() -> None:
    x, codes, scales, codebook = _fixture(m=8)
    actual = vq_qmm(
        mx.array(x.astype(np.float16)),
        mx.array(codes),
        mx.array(scales),
        mx.array(codebook),
        in_dim=64,
        out_dim=32,
        group_size=16,
        code_bits=8,
    )
    mx.eval(actual)

    assert actual.dtype == mx.float16


def test_quantized_vq_linear_dispatches_qmm_for_batched_prefill(monkeypatch) -> None:
    x, codes, scales, codebook = _fixture(m=2, in_dim=16, out_dim=4, group_size=8)
    calls = []

    def fake_qmm(*args, **kwargs):
        calls.append((args, kwargs))
        return mx.zeros((2, 4), dtype=mx.float32)

    monkeypatch.setattr(linear_module, "vq_qmm", fake_qmm)
    layer = QuantizedVQLinear(
        input_dims=16,
        output_dims=4,
        codes=mx.array(codes[:4, :2]),
        scales=mx.array(scales[:4, :2]),
        codebook=mx.array(codebook),
        group_size=8,
        code_bits=8,
    )
    out = layer(mx.array(x))

    assert out.shape == (2, 4)
    assert len(calls) == 1
