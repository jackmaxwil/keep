"""The 8-bit E8 residual kernels against a float64 decode of the same codes."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from keep.vq.e8 import e8_1bit_grid, e8_1bit_packed
from ramp.kernels import e8_1bit_m1

pytestmark = pytest.mark.skipif(not (getattr(mx, "metal", None) and mx.metal.is_available()), reason="needs a Metal GPU")
NONE = e8_1bit_m1.NO_RESIDUAL


def _layer(experts, out, inp, group_size, seed):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, (experts, out, inp // 8), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.2, (experts, out, inp // group_size)).astype(np.float16)
    dense = e8_1bit_grid().astype(np.float64)[codes].reshape(experts, out, inp)
    dense *= np.repeat(scales.astype(np.float64), group_size, axis=-1)
    return mx.array(codes), mx.array(scales), dense


def _close(got, want, tol):
    got = np.array(got.astype(mx.float32)).astype(np.float64)
    assert got.shape == want.shape, (got.shape, want.shape)
    err = np.abs(got - want).max() / np.abs(want).max()
    assert err <= tol, err


def test_pair_packed_keeps_every_value():
    pairs = e8_1bit_m1.pair_packed(e8_1bit_packed())
    nib = lambda p, n: ((p >> (4 * n)) & 0xF).astype(np.float64)
    back = np.stack([(nib(pairs, k if d % 2 == 0 else k + 4) - 8) / 2 for d in range(8) for k in [d // 2]], axis=1)
    np.testing.assert_array_equal(back, e8_1bit_grid())


# group 32 -> 4 codewords per group (4-byte loads); 64 -> 8. inp=768 makes lanes take more than
# one step; out=93 leaves a ragged last tile.
@pytest.mark.parametrize("group_size,inp,out", [(32, 128, 96), (64, 768, 93), (256, 4096, 64)])
@pytest.mark.parametrize("rows_per_simdgroup,simdgroups", [(1, 1), (8, 1), (4, 4)])
@pytest.mark.parametrize("dtype,tol", [(mx.float32, 1e-5), (mx.bfloat16, 2e-2)])
def test_both_layouts_match_the_dense_decode(group_size, inp, out, rows_per_simdgroup, simdgroups, dtype, tol):
    codes, scales, dense = _layer(6, out, inp, group_size, seed=3)
    book = mx.array(e8_1bit_m1.pair_packed(e8_1bit_packed()))
    kw = {"output_dims": out, "rows_per_simdgroup": rows_per_simdgroup, "simdgroups": simdgroups}
    rng = np.random.default_rng(4)
    rhs = np.array([[5, 0, 3], [NONE, 2, 2], [1, NONE, 4]], dtype=np.uint32)

    def rows(n):  # values as the kernel sees them, after rounding to dtype
        x = mx.array(rng.normal(size=(n, inp)).astype(np.float32)).astype(dtype)
        return x, np.array(x.astype(mx.float32)).astype(np.float64)

    def ref(x_row, e):
        return np.zeros(out) if e == NONE else x_row @ dense[e].T

    x, xn = rows(3)
    got = e8_1bit_m1.e8_1bit_m1_token_layout(x, codes, scales, book, mx.array(rhs), **kw)
    assert got.dtype == dtype
    _close(got, np.stack([[ref(xn[t], e) for e in rhs[t]] for t in range(3)]), tol)

    x, xn = rows(9)
    got = e8_1bit_m1.e8_1bit_m1_per_route(x, codes, scales, book, mx.array(rhs.reshape(-1)), **kw)
    _close(got, np.stack([ref(xn[r], e) for r, e in enumerate(rhs.reshape(-1))]), tol)
