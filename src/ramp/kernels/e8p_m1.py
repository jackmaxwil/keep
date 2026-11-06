"""Opt-in single-token (M=1) kernels for 16-bit E8P codes on GPUs without NAX.

The shipped DeepSeek-V4-Flash artifact uses 16-bit E8P codes. Every M=1 fast
path in ``QuantizedVQSwitchLinear`` is gated on 8-bit codes, and the E8P
tensor-unit kernels are M5-only, so on an M1-M4 GPU a decoded token's routed
projections fall through to the scalar kernel, which spends a whole 256-thread
threadgroup per output element. ``gather_vqmm_m1.metal`` already carries an
E8P branch; nothing dispatched to it.

With ``RAMP_E8P_M1=1`` (opt-in until ``benchmarks/bench_e8p_m1.py`` passes on
the target Mac), M=1 calls with 16-bit codes use:

- :func:`e8p_m1_token_layout` for gate/up (one activation row, top-k
  experts): the existing M=1 kernel with ``CODE_BITS=16``.
- :func:`e8p_m1_per_route` for down (one activation row *per route*): the
  same kernel source with the activation read offset by route. It is built as
  a separate kernel from a text substitution, so the production 8-bit kernel
  stays byte-identical.

The E8P branch decodes each codeword once per dimension, but the compiler lifts
the per-codeword work out of that loop: a decode-once rewrite measured slower
on the M4 Max (113 vs 154 GB/s, 2026-11-05).
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
    "per_route_x_source",
    "should_use_e8p_m1",
]

E8P_M1_ENV = "RAMP_E8P_M1"
_KERNEL_DIR = Path(__file__).resolve().parent
_ON = {"1", "on", "true", "yes"}
_OFF = {"", "0", "off", "false", "no"}
_X_INDEX_REWRITES = (
    ("uint x_base = cw * 8u;", "uint x_base = route * (codewords * 8u) + cw * 8u;"),
    ("uint x_idx = cw * 8u + dim;", "uint x_idx = route * (codewords * 8u) + cw * 8u + dim;"),
)


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


def per_route_x_source() -> str:
    """``gather_vqmm_m1.metal`` with the activation row chosen by route."""

    source = (_KERNEL_DIR / "gather_vqmm_m1.metal").read_text()
    for old, new in _X_INDEX_REWRITES:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"gather_vqmm_m1.metal changed: expected one {old!r}, found {count}"
            )
        source = source.replace(old, new)
    return source


@cache
def _per_route_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="ramp_e8p_m1_per_route_x",
        input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
        output_names=["out"],
        source=per_route_x_source(),
        header=(_KERNEL_DIR / "common_e8.metal").read_text(),
    )


def _launch_shape(output_dims: int, rows_per_threadgroup: int) -> tuple[int, int]:
    lanes_per_row = 256 // rows_per_threadgroup
    if rows_per_threadgroup * lanes_per_row != 256 or lanes_per_row < 1:
        raise ValueError("rows_per_threadgroup must divide 256")
    return lanes_per_row, (output_dims + rows_per_threadgroup - 1) // rows_per_threadgroup


def e8p_m1_token_layout(
    x,
    codes,
    scales,
    codebook,
    rhs_indices,
    *,
    output_dims: int,
    rows_per_threadgroup: int = 32,
    use_threadgroup_codebook: bool = True,
):
    """One activation row against ``top_k`` experts: ``[1, top_k, out]``."""

    from ramp.kernels.gather_vqmm import gather_vqmm_m1_kernel_unchecked

    if x.shape[0] != 1 or rhs_indices.ndim != 2 or rhs_indices.shape[0] != 1:
        raise ValueError("token-layout E8P M=1 needs x [1, in] and rhs_indices [1, top_k]")
    return gather_vqmm_m1_kernel_unchecked(
        x,
        codes,
        scales,
        codebook,
        rhs_indices,
        output_dims=output_dims,
        code_bits=16,
        rows_per_threadgroup=rows_per_threadgroup,
        use_threadgroup_codebook=use_threadgroup_codebook,
        use_decoded_codebook=False,
    )


def e8p_m1_per_route(
    x,
    codes,
    scales,
    codebook,
    rhs_indices,
    *,
    output_dims: int,
    rows_per_threadgroup: int = 32,
    use_threadgroup_codebook: bool = True,
):
    """Route ``r``'s activation row against expert ``rhs_indices[r]``: ``[routes, out]``."""

    import mlx.core as mx

    from ramp.kernels.gather_vqmm import _qmv_output_dtype

    routes = int(rhs_indices.size)
    if x.ndim != 2 or x.shape[0] != routes:
        raise ValueError(f"per-route E8P M=1 needs x [{routes}, in], got {tuple(x.shape)}")
    lanes_per_row, row_tiles = _launch_shape(output_dims, rows_per_threadgroup)
    output_dtype = _qmv_output_dtype(x.dtype)
    (out,) = _per_route_kernel()(
        inputs=[mx.contiguous(x), codes, scales, codebook, rhs_indices.reshape(1, routes)],
        template=[
            ("ROWS_PER_TG", rows_per_threadgroup),
            ("LANES_PER_ROW", lanes_per_row),
            ("OUT_T", output_dtype),
            ("CODE_BITS", 16),
            ("USE_THREADGROUP_CODEBOOK", int(use_threadgroup_codebook)),
            ("USE_DECODED_CODEBOOK", 0),
        ],
        grid=(256, row_tiles, routes),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, routes, output_dims)],
        output_dtypes=[output_dtype],
    )
    return out.reshape(routes, output_dims)
