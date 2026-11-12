"""Few-token kernels for 8-bit E8 codes, the 1-bit residual of a nested tier.

The companion of ``e8p_m1``: the same two layouts over ``uint8`` codes, for a residual that is
added to the output of the 16-bit base. A route whose index is ``NO_RESIDUAL`` writes zeros, so
one call serves a token whose routes are partly at the higher tier.

- :func:`e8_1bit_m1_token_layout` for gate/up (one activation row, top-k experts).
- :func:`e8_1bit_m1_per_route` for down (one activation row per route).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from ramp.kernels.e8p_m1 import _chunk

__all__ = ["NO_RESIDUAL", "e8_1bit_m1_per_route", "e8_1bit_m1_token_layout", "pair_packed"]

NO_RESIDUAL = 0xFFFFFFFF


def pair_packed(packed: np.ndarray) -> np.ndarray:
    """The packed 1-bit table (nibble d is dim d) with dims 2k and 2k+1 in nibbles k and k+4,
    the layout the kernel decodes into half2 pairs."""

    p, out = np.asarray(packed, dtype=np.uint32), np.zeros(256, np.uint32)
    for k in range(4):
        out |= ((p >> np.uint32(8 * k)) & np.uint32(0xF)) << np.uint32(4 * k)
        out |= ((p >> np.uint32(8 * k + 4)) & np.uint32(0xF)) << np.uint32(16 + 4 * k)
    return out


@cache
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="ramp_e8_1bit_m1",
        input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
        output_names=["out"],
        source=(Path(__file__).resolve().parent / "e8_1bit_m1.metal").read_text(),
    )


def _launch(x, codes, scales, codebook, rhs_indices, *, output_dims, rows_per_simdgroup, simdgroups, x_div):
    import mlx.core as mx

    from ramp.kernels.vq_qmv import _qmv_output_dtype

    if codes.dtype != mx.uint8:
        raise ValueError(f"8-bit E8 codes must be uint8, got {codes.dtype}")
    if rows_per_simdgroup < 1 or simdgroups < 1 or simdgroups > 32:
        raise ValueError("rows_per_simdgroup must be >= 1 and simdgroups in 1..32")
    routes, rows_per_tile = int(rhs_indices.size), rows_per_simdgroup * simdgroups
    output_dtype = _qmv_output_dtype(x.dtype)
    (out,) = _kernel()(
        inputs=[mx.contiguous(x), codes, scales, codebook, rhs_indices.reshape(-1)],
        template=[
            ("ROWS_PER_SG", rows_per_simdgroup),
            ("SIMDGROUPS", simdgroups),
            ("CHUNK", _chunk(int(codes.shape[2]), int(scales.shape[2]))),
            ("X_DIV", int(x_div)),
            ("OUT_T", output_dtype),
        ],
        grid=(32 * simdgroups, (output_dims + rows_per_tile - 1) // rows_per_tile, routes),
        threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[(1, routes, output_dims)],
        output_dtypes=[output_dtype],
    )
    return out


def e8_1bit_m1_token_layout(x, codes, scales, codebook, rhs_indices, *, output_dims: int,
                            rows_per_simdgroup: int = 4, simdgroups: int = 1):
    """Token ``t``'s row against its ``top_k`` experts: ``[tokens, top_k, out]``."""

    if x.ndim != 2 or rhs_indices.ndim != 2 or rhs_indices.shape[0] != x.shape[0]:
        raise ValueError("token-layout E8 needs x [tokens, in] and rhs_indices [tokens, top_k]")
    tokens, top_k = int(rhs_indices.shape[0]), int(rhs_indices.shape[1])
    out = _launch(x, codes, scales, codebook, rhs_indices, output_dims=output_dims,
                  rows_per_simdgroup=rows_per_simdgroup, simdgroups=simdgroups, x_div=top_k)
    return out.reshape(tokens, top_k, output_dims)


def e8_1bit_m1_per_route(x, codes, scales, codebook, rhs_indices, *, output_dims: int,
                         rows_per_simdgroup: int = 4, simdgroups: int = 1):
    """Route ``r``'s activation row against expert ``rhs_indices[r]``: ``[routes, out]``."""

    routes = int(rhs_indices.size)
    if x.ndim != 2 or x.shape[0] != routes:
        raise ValueError(f"per-route E8 needs x [{routes}, in], got {tuple(x.shape)}")
    out = _launch(x, codes, scales, codebook, rhs_indices, output_dims=output_dims,
                  rows_per_simdgroup=rows_per_simdgroup, simdgroups=simdgroups, x_div=1)
    return out.reshape(routes, output_dims)
