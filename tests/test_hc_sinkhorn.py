"""The opt-in one-dispatch hyper-connection sinkhorn and its reference."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from ramp.kernels.hc_sinkhorn import (  # noqa: E402
    HC_SINKHORN_ENV,
    fused_hc_sinkhorn,
    fused_hc_sinkhorn_enabled,
    hc_sinkhorn_reference,
)
from ramp.models.deepseek_v4_flash_adapter import (  # noqa: E402
    _hc_split_logits_ops,
    _hc_split_sinkhorn_ops,
)

HC, ITERS, EPS = 4, 20, 1e-6


def _inputs(rows, seed=0):
    rng = np.random.default_rng(seed)
    mix = (2 + HC) * HC
    mixes = mx.array(rng.normal(size=(1, rows, mix)).astype(np.float32) * 3)
    scale = mx.array(rng.uniform(0.5, 2.0, size=(3,)).astype(np.float32))
    base = mx.array(rng.normal(size=(mix,)).astype(np.float32))
    return mixes, scale, base


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return bool(metal is not None and metal.is_available())


@pytest.mark.parametrize("rows", [1, 7, 256])
def test_reference_reproduces_the_models_compiled_sinkhorn(rows):
    mixes, scale, base = _inputs(rows)
    pre, post, comb = _hc_split_sinkhorn_ops(mixes, scale, base, HC, ITERS, EPS)
    pre2, post2, logits = _hc_split_logits_ops(mixes, scale, base, HC, EPS)
    # pre/post are computed identically on both paths.
    assert mx.array_equal(pre, pre2) and mx.array_equal(post, post2)
    want = np.array(comb)
    got = hc_sinkhorn_reference(np.array(logits), iters=ITERS, eps=EPS)
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-7)
    # The last step is a column normalisation, so columns sum to one; rows
    # only approach it (20 rounds can leave them ~0.3% off for some inputs).
    np.testing.assert_allclose(got.sum(axis=-2), 1.0, atol=1e-5)


def test_backend_flag(monkeypatch):
    monkeypatch.delenv(HC_SINKHORN_ENV, raising=False)
    assert fused_hc_sinkhorn_enabled() is False
    monkeypatch.setenv(HC_SINKHORN_ENV, "metal")
    assert fused_hc_sinkhorn_enabled() is True
    monkeypatch.setenv(HC_SINKHORN_ENV, "cuda")
    with pytest.raises(ValueError, match=HC_SINKHORN_ENV):
        fused_hc_sinkhorn_enabled()


def test_fused_kernel_validates_its_input_shape():
    with pytest.raises(ValueError, match="hc, hc"):
        fused_hc_sinkhorn(mx.zeros((3, 4, 5)), iters=ITERS, eps=EPS)
    with pytest.raises(ValueError, match="iters"):
        fused_hc_sinkhorn(mx.zeros((3, 4, 4)), iters=0, eps=EPS)


@pytest.mark.skipif(not _metal_available(), reason="needs a Metal GPU")
@pytest.mark.parametrize("rows", [1, 7, 1024])
def test_fused_kernel_matches_the_compiled_path(rows):
    mixes, scale, base = _inputs(rows, seed=rows)
    _, _, comb = _hc_split_sinkhorn_ops(mixes, scale, base, HC, ITERS, EPS)
    _, _, logits = _hc_split_logits_ops(mixes, scale, base, HC, EPS)
    fused = fused_hc_sinkhorn(logits, iters=ITERS, eps=EPS)
    assert fused.shape == comb.shape and fused.dtype == mx.float32
    np.testing.assert_allclose(np.array(fused), np.array(comb), rtol=2e-6, atol=1e-7)


@pytest.mark.skipif(not _metal_available(), reason="needs a Metal GPU")
def test_hyper_connection_output_with_the_flag(monkeypatch):
    from ramp.models.deepseek_v4_flash_adapter import (
        DeepseekV4FlashVQModelArgs,
        HyperConnection,
    )

    args = DeepseekV4FlashVQModelArgs(hidden_size=64, hc_mult=HC, num_hidden_layers=1, num_hash_layers=0, compress_ratios=[0])
    hc = HyperConnection(args)
    rng = np.random.default_rng(1)
    hc.fn = mx.array(rng.normal(size=hc.fn.shape).astype(np.float32) * 0.05)
    hc.base = mx.array(rng.normal(size=hc.base.shape).astype(np.float32))
    x = mx.array(rng.normal(size=(1, 3, HC, 64)).astype(np.float32)).astype(mx.bfloat16)
    monkeypatch.delenv(HC_SINKHORN_ENV, raising=False)
    want = hc(x)
    monkeypatch.setenv(HC_SINKHORN_ENV, "metal")
    got = hc(x)
    for a, b in zip(got, want):
        np.testing.assert_allclose(
            np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)), rtol=1e-2, atol=1e-6
        )
