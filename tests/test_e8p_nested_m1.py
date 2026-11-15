"""The fused base + residual kernels against the two separate kernels (bit for bit in float32)
and against a float64 decode of the same codes."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from keep.vq.e8 import decode_weight_matrix, e8_1bit_grid, e8_1bit_packed, e8p_packed_abs_grid
from ramp.kernels import e8_1bit_m1, e8p_m1, e8p_nested_m1

pytestmark = pytest.mark.skipif(not (getattr(mx, "metal", None) and mx.metal.is_available()), reason="needs a Metal GPU")
NONE = e8_1bit_m1.NO_RESIDUAL
HOT_OF_SLOT = np.array([2, NONE, 0, NONE, NONE, 1], dtype=np.uint32)  # six slots, three residual rows


def _nested(out, inp, group_size, res_group_size, seed):
    rng = np.random.default_rng(seed)
    slots, rows = len(HOT_OF_SLOT), int((HOT_OF_SLOT != NONE).sum())
    codes = rng.integers(0, 1 << 16, (slots, out, inp // 8), dtype=np.uint16)
    scales = rng.uniform(0.01, 0.2, (slots, out, inp // group_size)).astype(np.float16)
    res_codes = rng.integers(0, 256, (rows, out, inp // 8), dtype=np.uint8)
    res_scales = rng.uniform(0.005, 0.1, (rows, out, inp // res_group_size)).astype(np.float16)
    dense = np.stack([decode_weight_matrix(codes[s], scales[s].astype(np.float32), code_bits=16)
                      for s in range(slots)]).astype(np.float64)
    res = e8_1bit_grid().astype(np.float64)[res_codes].reshape(rows, out, inp)
    res *= np.repeat(res_scales.astype(np.float64), res_group_size, axis=-1)
    for slot, h in enumerate(HOT_OF_SLOT):
        if h != NONE:
            dense[slot] += res[h]
    base = (mx.array(codes), mx.array(scales), mx.array(e8p_packed_abs_grid()))
    res = (mx.array(res_codes), mx.array(res_scales))
    return base, (*res, mx.array(e8p_nested_m1.half2_pairs(e8_1bit_grid()))), (*res, mx.array(e8_1bit_m1.pair_packed(e8_1bit_packed()))), dense


# Base group 32 -> 4-codeword loads; 64 and up -> 8. The residual groups differ from the base's in
# two cases. inp=768 makes lanes take more than one step; out=93 leaves a ragged last tile.
@pytest.mark.parametrize("group_size,res_group_size,inp,out", [(32, 32, 128, 96), (64, 128, 768, 93), (512, 64, 4096, 64)])
@pytest.mark.parametrize("rows_per_simdgroup,simdgroups", [(1, 1), (8, 1), (4, 4)])
@pytest.mark.parametrize("dtype,tol", [(mx.float32, 1e-5), (mx.bfloat16, 2e-2)])
def test_both_layouts(group_size, res_group_size, inp, out, rows_per_simdgroup, simdgroups, dtype, tol):
    base, residual, separate_residual, dense = _nested(out, inp, group_size, res_group_size, seed=3)
    hot_of_slot = mx.array(HOT_OF_SLOT)
    kw = {"output_dims": out, "rows_per_simdgroup": rows_per_simdgroup, "simdgroups": simdgroups}
    rng = np.random.default_rng(4)
    rhs = np.array([[5, 0, 3], [1, 2, 2], [4, 5, 0]], dtype=np.uint32)  # hot and base-tier slots in one token

    def rows(n):  # values as the kernel sees them, after rounding to dtype
        x = mx.array(rng.normal(size=(n, inp)).astype(np.float32)).astype(dtype)
        return x, np.array(x.astype(mx.float32)).astype(np.float64)

    def check(got, want, separate):
        assert got.dtype == dtype
        got = np.array(got.astype(mx.float32))
        assert got.shape == want.shape, (got.shape, want.shape)
        err = np.abs(got - want).max() / np.abs(want).max()
        assert err <= tol, err
        if dtype == mx.float32:  # the two kernels it replaces, summed in float32
            np.testing.assert_array_equal(got, np.array(separate))

    x, xn = rows(3)
    got = e8p_nested_m1.e8p_nested_m1_token_layout(x, base, mx.array(rhs), residual, hot_of_slot, **kw)
    separate = (e8p_m1.e8p_m1_token_layout(x, *base, mx.array(rhs), **kw)
                + e8_1bit_m1.e8_1bit_m1_token_layout(x, *separate_residual, mx.array(HOT_OF_SLOT[rhs]), **kw))
    check(got, np.stack([[xn[t] @ dense[s].T for s in rhs[t]] for t in range(3)]), separate)

    x, xn = rows(9)
    flat = rhs.reshape(-1)
    got = e8p_nested_m1.e8p_nested_m1_per_route(x, base, mx.array(flat), residual, hot_of_slot, **kw)
    separate = (e8p_m1.e8p_m1_per_route(x, *base, mx.array(flat), **kw)
                + e8_1bit_m1.e8_1bit_m1_per_route(x, *separate_residual, mx.array(HOT_OF_SLOT[flat]), **kw))
    check(got, np.stack([xn[r] @ dense[s].T for r, s in enumerate(flat)]), separate)
