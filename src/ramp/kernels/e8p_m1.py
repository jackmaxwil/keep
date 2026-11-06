"""Opt-in single-token (M=1) kernels for 16-bit E8P codes on GPUs without NAX.

The shipped DeepSeek-V4-Flash artifact uses 16-bit E8P codes. Every M=1 fast
path in ``QuantizedVQSwitchLinear`` is gated on 8-bit codes, and the E8P
tensor-unit kernels are M5-only, so on an M1-M4 GPU a decoded token's routed
projections fall through to the scalar kernel, which spends a whole 256-thread
threadgroup per output element. ``gather_vqmm_m1.metal`` already carries an
E8P branch; nothing dispatched to it.

With ``RAMP_E8P_M1=1`` (opt-in until ``benchmarks/bench_e8p_m1.py`` passes on
the target Mac), M=1 calls with 16-bit codes use ``e8p_m1.metal``:

- :func:`e8p_m1_token_layout` for gate/up (one activation row, top-k
  experts).
- :func:`e8p_m1_per_route` for down (one activation row *per route*).

The kernel gives each simdgroup several rows, loads 4-8 codewords per lane at
once, and decodes the abs grid once per threadgroup. On the M4 Max that is
1.6-2.0x the generic M=1 kernel's E8P branch, which it replaces here; the
production 8-bit M=1 kernel is untouched. A loads-only stub reaches ~410 GB/s,
so decode ALU is still the limit.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

__all__ = [
    "E8P_M1_ENV",
    "e8p_m1_enabled",
    "e8p_m1_per_route",
    "e8p_m1_token_layout",
    "should_use_e8p_m1",
]

E8P_M1_ENV = "RAMP_E8P_M1"
_KERNEL_DIR = Path(__file__).resolve().parent
_ON = {"1", "on", "true", "yes"}
_OFF = {"", "0", "off", "false", "no"}


def e8p_m1_enabled() -> bool:
    value = os.environ.get(E8P_M1_ENV, "").strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise ValueError(f"{E8P_M1_ENV} must be on/off (1/0), got {value!r}")


def should_use_e8p_m1(
    *, code_bits: int, tokens: int, sorted_indices: bool, route_strategy: str
) -> bool:
    """The opt-in decision, kept pure so the dispatch table is testable."""

    return (
        code_bits == 16
        and tokens == 1
        and not sorted_indices
        and route_strategy in ("auto", "direct")
        and e8p_m1_enabled()
    )


@cache
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="ramp_e8p_m1",
        input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
        output_names=["out"],
        source=(_KERNEL_DIR / "e8p_m1.metal").read_text(),
    )


def _chunk(codewords: int, groups: int) -> int:
    """Codewords per wide load: a chunk must not straddle a row or scale group."""

    for chunk in (8, 4):
        if codewords % groups == 0 and (codewords // groups) % chunk == 0:
            return chunk
    raise ValueError(f"E8P M=1 needs codewords per scale group divisible by 4, got {codewords}/{groups}")


def _launch(x, codes, scales, codebook, rhs_indices, *, output_dims, rows_per_simdgroup, simdgroups, per_route):
    import mlx.core as mx

    from ramp.kernels.gather_vqmm import _qmv_output_dtype

    if rows_per_simdgroup < 1 or simdgroups < 1 or simdgroups > 32:
        raise ValueError("rows_per_simdgroup must be >= 1 and simdgroups in 1..32")
    routes = int(rhs_indices.size)
    rows_per_tile = rows_per_simdgroup * simdgroups
    output_dtype = _qmv_output_dtype(x.dtype)
    (out,) = _kernel()(
        inputs=[mx.contiguous(x), codes, scales, codebook, rhs_indices.reshape(1, routes)],
        template=[
            ("ROWS_PER_SG", rows_per_simdgroup),
            ("SIMDGROUPS", simdgroups),
            ("CHUNK", _chunk(int(codes.shape[2]), int(scales.shape[2]))),
            ("PER_ROUTE", int(per_route)),
            ("OUT_T", output_dtype),
        ],
        grid=(32 * simdgroups, (output_dims + rows_per_tile - 1) // rows_per_tile, routes),
        threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[(1, routes, output_dims)],
        output_dtypes=[output_dtype],
    )
    return out


def e8p_m1_token_layout(
    x,
    codes,
    scales,
    codebook,
    rhs_indices,
    *,
    output_dims: int,
    rows_per_simdgroup: int = 4,
    simdgroups: int = 1,
):
    """One activation row against ``top_k`` experts: ``[1, top_k, out]``."""

    if x.shape[0] != 1 or rhs_indices.ndim != 2 or rhs_indices.shape[0] != 1:
        raise ValueError("token-layout E8P M=1 needs x [1, in] and rhs_indices [1, top_k]")
    return _launch(
        x, codes, scales, codebook, rhs_indices,
        output_dims=output_dims,
        rows_per_simdgroup=rows_per_simdgroup,
        simdgroups=simdgroups,
        per_route=False,
    )


def e8p_m1_per_route(
    x,
    codes,
    scales,
    codebook,
    rhs_indices,
    *,
    output_dims: int,
    rows_per_simdgroup: int = 4,
    simdgroups: int = 1,
):
    """Route ``r``'s activation row against expert ``rhs_indices[r]``: ``[routes, out]``."""

    routes = int(rhs_indices.size)
    if x.ndim != 2 or x.shape[0] != routes:
        raise ValueError(f"per-route E8P M=1 needs x [{routes}, in], got {tuple(x.shape)}")
    out = _launch(
        x, codes, scales, codebook, rhs_indices,
        output_dims=output_dims,
        rows_per_simdgroup=rows_per_simdgroup,
        simdgroups=simdgroups,
        per_route=True,
    )
    return out.reshape(routes, output_dims)
