"""Few-token kernels for a nested tier: 16-bit E8P base plus an 8-bit E8 residual, in one pass.

``e8p_m1`` followed by ``e8_1bit_m1`` costs a second kernel launch per projection and the ops that
add the two outputs. This kernel reads both code arrays in the same lane loop and looks up, per
route, whether the slot has a residual: ``hot_of_slot[slot]`` is the slot's row in the residual
arrays or ``NO_RESIDUAL``. A float32 output equals the float32 sum of the two separate kernels bit
for bit; a 16-bit output is rounded once, after the sum.

M4 Max, gate shape (2304 x 5120, 6 routes, bf16 rows), 2026-11-15: the base kernel 0.077 ms; this
kernel 0.077 ms with no route hot and 0.12 ms with all six hot; the two separate kernels and the
add 0.100 and 0.151 ms.

- :func:`e8p_nested_m1_token_layout` for gate/up (one activation row, top-k slots).
- :func:`e8p_nested_m1_per_route` for down (one activation row per route).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from ramp.kernels.e8p_m1 import _chunk

__all__ = ["e8p_nested_m1_per_route", "e8p_nested_m1_token_layout", "half2_pairs"]


def half2_pairs(grid: np.ndarray) -> np.ndarray:
    """A [256, 8] codebook as the kernel reads it: [256, 4] uint32, entry k the float16 bits of
    dims 2k (low half) and 2k+1. Every value must be exact in float16."""

    half = np.asarray(grid).astype(np.float16)
    assert half.shape == (256, 8) and np.array_equal(half.astype(np.float64), grid), "codebook not exact in float16"
    bits = half.view(np.uint16).astype(np.uint32)
    return np.ascontiguousarray(bits[:, 0::2] | (bits[:, 1::2] << 16))


@cache
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="ramp_e8p_nested_m1",
        input_names=["x", "codes", "scales", "codebook", "rhs_indices",
                     "res_codes", "res_scales", "res_table", "hot_of_slot"],
        output_names=["out"],
        source=(Path(__file__).resolve().parent / "e8p_nested_m1.metal").read_text(),
    )


def _launch(x, base, rhs_indices, residual, hot_of_slot, *, output_dims, rows_per_simdgroup, simdgroups, x_div):
    import mlx.core as mx

    from ramp.kernels.vq_qmv import _qmv_output_dtype

    (codes, scales, codebook), (res_codes, res_scales, res_table) = base, residual
    if codes.dtype != mx.uint16 or res_codes.dtype != mx.uint8:
        raise ValueError(f"nested codes must be uint16 base and uint8 residual, got {codes.dtype} and {res_codes.dtype}")
    if codes.shape[1:] != res_codes.shape[1:]:
        raise ValueError(f"base and residual rows differ: {codes.shape[1:]} and {res_codes.shape[1:]}")
    if res_table.shape != (256, 4) or res_table.dtype != mx.uint32:
        raise ValueError("the residual table must be half2_pairs(grid): [256, 4] uint32")
    if hot_of_slot.size != codes.shape[0]:
        raise ValueError(f"hot_of_slot needs one entry per slot, got {hot_of_slot.size} for {codes.shape[0]}")
    if rows_per_simdgroup < 1 or simdgroups < 1 or simdgroups > 32:
        raise ValueError("rows_per_simdgroup must be >= 1 and simdgroups in 1..32")
    routes, rows_per_tile = int(rhs_indices.size), rows_per_simdgroup * simdgroups
    codewords = int(codes.shape[2])
    chunk = min(_chunk(codewords, int(scales.shape[2])), _chunk(codewords, int(res_scales.shape[2])))
    output_dtype = _qmv_output_dtype(x.dtype)
    (out,) = _kernel()(
        inputs=[mx.contiguous(x), codes, scales, codebook, rhs_indices.reshape(-1),
                res_codes, res_scales, res_table, hot_of_slot],
        template=[
            ("ROWS_PER_SG", rows_per_simdgroup),
            ("SIMDGROUPS", simdgroups),
            ("CHUNK", chunk),
            ("X_DIV", int(x_div)),
            ("OUT_T", output_dtype),
        ],
        grid=(32 * simdgroups, (output_dims + rows_per_tile - 1) // rows_per_tile, routes),
        threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[(1, routes, output_dims)],
        output_dtypes=[output_dtype],
    )
    return out


def e8p_nested_m1_token_layout(x, base, rhs_indices, residual, hot_of_slot, *, output_dims: int,
                               rows_per_simdgroup: int = 4, simdgroups: int = 1):
    """Token ``t``'s row against its ``top_k`` slots: ``[tokens, top_k, out]``. ``base`` and
    ``residual`` are (codes, scales, table): the base's packed abs grid, the residual's ``half2_pairs``."""

    if x.ndim != 2 or rhs_indices.ndim != 2 or rhs_indices.shape[0] != x.shape[0]:
        raise ValueError("token-layout nested E8P needs x [tokens, in] and rhs_indices [tokens, top_k]")
    tokens, top_k = int(rhs_indices.shape[0]), int(rhs_indices.shape[1])
    out = _launch(x, base, rhs_indices, residual, hot_of_slot, output_dims=output_dims,
                  rows_per_simdgroup=rows_per_simdgroup, simdgroups=simdgroups, x_div=top_k)
    return out.reshape(tokens, top_k, output_dims)


def e8p_nested_m1_per_route(x, base, rhs_indices, residual, hot_of_slot, *, output_dims: int,
                            rows_per_simdgroup: int = 4, simdgroups: int = 1):
    """Route ``r``'s activation row against slot ``rhs_indices[r]``: ``[routes, out]``."""

    routes = int(rhs_indices.size)
    if x.ndim != 2 or x.shape[0] != routes:
        raise ValueError(f"per-route nested E8P needs x [{routes}, in], got {tuple(x.shape)}")
    out = _launch(x, base, rhs_indices, residual, hot_of_slot, output_dims=output_dims,
                  rows_per_simdgroup=rows_per_simdgroup, simdgroups=simdgroups, x_div=1)
    return out.reshape(routes, output_dims)
