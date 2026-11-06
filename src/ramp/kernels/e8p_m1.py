"""Opt-in single-token (M=1) kernels for 16-bit E8P codes on GPUs without NAX.

The shipped DeepSeek-V4-Flash artifact uses 16-bit E8P codes. Every M=1 fast
path in ``QuantizedVQSwitchLinear`` is gated on 8-bit codes, and the E8P
tensor-unit kernels are M5-only, so on an M1-M4 GPU a decoded token's routed
projections fall through to the scalar kernel, which spends a whole 256-thread
threadgroup per output element. ``gather_vqmm_m1.metal`` already carries an
E8P branch; nothing dispatched to it.

On a Metal GPU without the NAX kernels (M1-M4), M=1 calls with 16-bit codes
use ``e8p_m1.metal`` by default, since ``benchmarks/bench_e8p_m1.py`` passed
on the M4 Max (2026-11-05). ``RAMP_E8P_M1=1`` forces it on, ``=0`` off:

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
#: Token rows the kernel serves: M=1 decode and MTP verify windows. Wider
#: calls (prefill) keep the sorted-route kernels, which share expert decode.
E8P_MAX_TOKENS = 32
_KERNEL_DIR = Path(__file__).resolve().parent
_ON = {"1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}


def default_enabled() -> bool:
    """On for a Metal GPU without NAX; NAX GPUs keep their tensor-unit kernels.

    Not cached: ``nax.is_available`` already is, and tests patch it.
    """

    import mlx.core as mx

    from ramp.kernels import nax

    return bool(mx.metal.is_available()) and not nax.is_available()


def e8p_m1_enabled() -> bool:
    value = os.environ.get(E8P_M1_ENV, "").strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    if value in ("", "auto"):
        return default_enabled()
    raise ValueError(f"{E8P_M1_ENV} must be on/off/auto (1/0), got {value!r}")


def should_use_e8p_m1(
    *, code_bits: int, tokens: int, sorted_indices: bool, route_strategy: str
) -> bool:
    """The opt-in decision, kept pure so the dispatch table is testable."""

    return (
        code_bits == 16
        and 1 <= tokens <= E8P_MAX_TOKENS
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


def _launch(x, codes, scales, codebook, rhs_indices, *, output_dims, rows_per_simdgroup, simdgroups, x_div):
    import mlx.core as mx

    from ramp.kernels.gather_vqmm import _qmv_output_dtype

    if rows_per_simdgroup < 1 or simdgroups < 1 or simdgroups > 32:
        raise ValueError("rows_per_simdgroup must be >= 1 and simdgroups in 1..32")
    routes = int(rhs_indices.size)
    rows_per_tile = rows_per_simdgroup * simdgroups
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
    """Token ``t``'s row against its ``top_k`` experts: ``[tokens, top_k, out]``."""

    if x.ndim != 2 or rhs_indices.ndim != 2 or rhs_indices.shape[0] != x.shape[0]:
        raise ValueError("token-layout E8P needs x [tokens, in] and rhs_indices [tokens, top_k]")
    tokens, top_k = int(rhs_indices.shape[0]), int(rhs_indices.shape[1])
    out = _launch(
        x, codes, scales, codebook, rhs_indices,
        output_dims=output_dims,
        rows_per_simdgroup=rows_per_simdgroup,
        simdgroups=simdgroups,
        x_div=top_k,
    )
    return out.reshape(tokens, top_k, output_dims)


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
        x_div=1,
    )
    return out.reshape(routes, output_dims)
