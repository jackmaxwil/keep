"""The FP8/FP4 stand-ins in torch_kernel.py against block-loop references written like the kernels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "runpod" / "v41_reap"))
sys.path.insert(0, str(REPO / "src"))

import torch_kernel as k  # noqa: E402


@pytest.fixture(autouse=True)
def _keep_default_dtype():
    """The GEMMs return the default dtype; set fp32 here and leave other tests' default alone."""
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(before)


def _e8m0(shape, lo=-6, hi=2):
    return (2.0 ** torch.randint(lo, hi, shape).float()).to(torch.float8_e8m0fnu)


def test_decode_e2m1_matches_numpy_decoder():
    from keep.convert.fp4_expert import decode_fp4_e2m1

    codes = torch.arange(256, dtype=torch.uint8).view(16, 16)
    want = decode_fp4_e2m1(codes.numpy().view(np.int8))
    got = k.decode_e2m1(codes.view(torch.float4_e2m1fn_x2))
    np.testing.assert_array_equal(got.numpy(), np.asarray(want, dtype=np.float32))


def test_act_quant_scales_are_pow2_and_inplace_is_quant_dequant():
    torch.manual_seed(0)
    x = torch.randn(5, 96, dtype=torch.bfloat16) * 7
    y, s = k.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    sf = s.float()
    assert torch.equal(sf, 2.0 ** torch.log2(sf).round())  # powers of two
    amax = x.float().unflatten(-1, (-1, 32)).abs().amax(-1)
    assert (amax / sf <= 448).all() and (amax / sf > 224).all()  # the smallest pow2 scale that fits
    k.exact_quant = True
    try:
        z = x.clone()
        k.act_quant(z, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    finally:
        k.exact_quant = False
    deq = (y.float().unflatten(-1, (-1, 32)) * sf[..., None]).flatten(-2).to(torch.bfloat16)
    assert torch.equal(z, deq)


def _block_loop(A, sa, B, sb_of, K, kb):
    """sum over K blocks of (A_k @ B_k^T) * sa[:, k] * sb[:, k], as the tilelang kernels accumulate."""
    out = torch.zeros(A.shape[0], B.shape[0])
    for i in range(K // kb):
        part = A[:, i * kb : (i + 1) * kb] @ B[:, i * kb : (i + 1) * kb].T
        out += part * sa[:, i : i + 1] * sb_of(i)[None, :]
    return out


def test_fp8_gemm_matches_block_loop():
    torch.manual_seed(1)
    M, N, K, bs = 7, 70, 96, 32
    a = (torch.randn(M, K) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
    a_s, b_s = _e8m0((M, K // bs)), _e8m0(((N + bs - 1) // bs, K // bs))
    got = k.fp8_gemm(a, a_s, b, b_s, torch.float8_e8m0fnu, bs)
    rows = torch.arange(N) // bs
    want = _block_loop(a.float(), a_s.float(), b.float(), lambda i: b_s.float()[rows, i], K, bs)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-3)


def test_fp4_gemm_matches_block_loop():
    torch.manual_seed(2)
    M, N, K, act_bs = 6, 40, 128, 32
    a = (torch.randn(M, K) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    a_s, b_s = _e8m0((M, K // act_bs)), _e8m0((N, K // 32))
    got = k.fp4_gemm(a, a_s, b, b_s, torch.float8_e8m0fnu, act_bs)
    B = k.decode_e2m1(b)
    want = _block_loop(a.float(), a_s.float().repeat_interleave(act_bs // 32, 1), B, lambda i: b_s.float()[:, i], K, 32)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-3)


def test_e2m1_round_ties_to_even():
    v = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -2.5, 6.0, 0.3])
    want = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -0.0, -2.0, 6.0, 0.5]
    assert k._e2m1_round(v).tolist() == want
