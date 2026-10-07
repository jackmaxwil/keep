from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Literal

import mlx.core as mx
import numpy as np

from keep.vq.e8 import CODEWORD_DIM, e8_1bit_packed, e8p_packed_abs_grid
from ramp.kernels import nax
from ramp.kernels.gather_vqmm import (
    VERIFY_ROWS_PER_THREADGROUP_VALUES,
    gather_vqmm_kernel,
    gather_vqmm_lhs_kernel,
    gather_vqmm_m1_kernel,
    gather_vqmm_mma4x4_blocks_kernel,
    gather_vqmm_mma8x4_blocks_kernel,
    gather_vqmm_mma_cwdecode_blocks_kernel,
    gather_vqmm_mma_k32_cwdecode_blocks_kernel,
    gather_vqmm_mma_k64_cwdecode_blocks_kernel,
    gather_vqmm_tiled_down_kernel,
    gather_vqmm_verify_mrows_kernel,
    m1_rows_per_threadgroup,
    m1_use_decoded_codebook,
    m1_use_threadgroup_codebook,
    verify_m_rows,
)
from ramp.kernels.vq_qmv import vq_qmv

RouteStrategy = Literal["auto", "direct", "sorted_tiled", "per_route_decoded"]
SortedRouteProjection = Literal["auto", "gate_up", "down"]
VerifyGrouping = Literal["expert", "flat"]
_VERIFY_DEFAULT_M_ROWS = 4
# _verify_row_tile_descriptors builds routes x routes device matrices. That is
# fine for verify widths (M <= 8, top-6 => 48 routes => 2304 elements) and
# quadratically wrong for prefill; sorted_tiled/block dispatch owns those.
_VERIFY_MAX_ROUTES = 256
# route_count at or above this goes to sorted_tiled; below it, auto picks
# between the M=1 fast path, per_route_decoded, and the scalar kernel.
_SORTED_TILED_ROUTE_THRESHOLD = 128
# Activation dtypes "auto" will move onto the decoded verify kernel.
#
# Deliberately float16 only, and the reason is precision, not capability. The
# verify kernel stages activations as half4. At float16 that is a *reinterpret*
# of the same bytes -- activations enter the dot at full input precision, and
# the only difference from the scalar kernel is that the 4-term dot runs in half
# instead of as four float32 FMAs. At float32 the kernel instead narrows each
# activation with half(x[i]), discarding ~13 mantissa bits of input data before
# any arithmetic happens. That is a precision regression rather than a
# reassociation, so float32 callers keep the scalar float32 path bit-for-bit.
# bfloat16 is excluded for a stronger version of the same reason: its exponent
# range exceeds half's, so narrowing can overflow finite values to inf.
#
# The production MTP verify path this increment targets runs float16
# activations, so this restriction costs none of the measured win. Note that the
# M=1 decode fast path *does* already narrow float32 activations this way; that
# asymmetry predates this increment and is recorded in the increment-2 report.
_VERIFY_ACTIVATION_DTYPES = (mx.float16,)
_GATE_UP_MMA_ROUTE_THRESHOLD = 8_192
_GATE_UP_MMA_ROUTE_TILE = 32
_DOWN_BLOCK_MMA_ROUTE_THRESHOLD = 128
_DOWN_BLOCK_MMA_ROUTE_TILE = 32
_LARGE_BLOCK_MMA_ROUTE_THRESHOLD = 16_384
_LARGE_BLOCK_MMA_ROUTE_TILE = 64
_NAX_E8P_AIR_DOWN_SMALL_ROUTE_THRESHOLD = 8_192
_GATHER_VQMM_TRACE: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "gather_vqmm_trace", default=None
)


@contextmanager
def trace_gather_vqmm_calls() -> Iterator[list[dict[str, Any]]]:
    """Record the concrete routed-kernel dispatches inside one target verify."""

    records: list[dict[str, Any]] = []
    token = _GATHER_VQMM_TRACE.set(records)
    try:
        yield records
    finally:
        _GATHER_VQMM_TRACE.reset(token)


def _as_mlx(array: mx.array | np.ndarray) -> mx.array:
    if isinstance(array, mx.array):
        return array
    return mx.array(array)


def _indices_to_numpy(indices: mx.array | np.ndarray) -> np.ndarray:
    values = np.array(indices, copy=False)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"indices must use an integer dtype, found {values.dtype}")
    return values.astype(np.int64, copy=False)


def _inverse_permutation(order: mx.array) -> mx.array:
    positions = mx.arange(order.shape[0], dtype=order.dtype)
    return mx.zeros_like(order).at[order].add(positions)


def _default_codebook(code_bits: Literal[8, 16]) -> np.ndarray:
    if code_bits == 8:
        return e8_1bit_packed()
    return e8p_packed_abs_grid()


def _infer_air_projection(input_dims: int, output_dims: int) -> Literal["gate_up", "down"]:
    # GLM-4.5-Air routed experts use a reduced intermediate dimension:
    # gate/up are 4096 -> 1408 and down expands 1408 -> 4096.
    return "down" if output_dims >= input_dims else "gate_up"


def _expert_block_tile_descriptors(
    sorted_rhs: mx.array,
    *,
    num_experts: int,
    route_tile: int = _GATE_UP_MMA_ROUTE_TILE,
) -> tuple[mx.array, mx.array, mx.array]:
    route_count = sorted_rhs.shape[0]
    expert_ids = mx.arange(num_experts, dtype=mx.int32)
    counts = mx.sum(sorted_rhs[:, None] == expert_ids[None, :], axis=0).astype(mx.int32)
    route_offsets = mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.cumsum(counts)])[:-1]
    tile_counts_per_expert = (counts + route_tile - 1) // route_tile
    tile_prefix = mx.concatenate(
        [mx.zeros((1,), dtype=mx.int32), mx.cumsum(tile_counts_per_expert)]
    )

    max_tiles = (route_count + route_tile - 1) // route_tile + num_experts
    tile_pos = mx.arange(max_tiles, dtype=mx.int32)
    tile_active_by_expert = (tile_pos[:, None] >= tile_prefix[:-1][None, :]) & (
        tile_pos[:, None] < tile_prefix[1:][None, :]
    )
    tile_active_i = tile_active_by_expert.astype(mx.int32)
    tile_experts = mx.sum(tile_active_i * expert_ids[None, :], axis=1).astype(mx.int32)
    active = mx.sum(tile_active_i, axis=1).astype(mx.bool_)

    local_tile = tile_pos - tile_prefix[tile_experts]
    remaining = counts[tile_experts] - local_tile * route_tile
    tile_counts = mx.where(
        active,
        mx.minimum(
            mx.maximum(remaining, mx.array(0, dtype=mx.int32)),
            mx.array(route_tile, dtype=mx.int32),
        ),
        mx.zeros((max_tiles,), dtype=mx.int32),
    )
    tile_offsets = route_offsets[tile_experts] + local_tile * route_tile
    return tile_experts, tile_offsets.astype(mx.int32), tile_counts.astype(mx.int32)


@lru_cache(maxsize=_VERIFY_MAX_ROUTES)
def _flat_route_descriptors(route_count: int) -> tuple[mx.array, mx.array]:
    """Cached ``(tile_offsets, tile_counts)`` for one-route-per-tile dispatch.

    Flat verify dispatch needs a descriptor triple whose offsets are ``arange``
    and whose counts are all ones. Building those per call adds two device
    allocations to a path whose whole kernel is ~50-250 us, so they are cached
    by route count.

    The cache is sized to ``_VERIFY_MAX_ROUTES`` rather than a smaller number
    because ``auto`` now selects this path for every route count below
    ``_SORTED_TILED_ROUTE_THRESHOLD``: a 64-entry cache would thrash across the
    1..127 band that real routing walks through. Overflow would only cost the
    two small allocations back, not correctness.
    """

    offsets = mx.arange(route_count, dtype=mx.int32)
    counts = mx.ones((route_count,), dtype=mx.int32)
    mx.eval(offsets, counts)
    return offsets, counts


@lru_cache(maxsize=64)
def _token_route_lhs(route_count: int, top_k: int) -> mx.array:
    """Cached route -> token map for the ``[verify_rows, top_k]`` layout."""

    lhs = (mx.arange(route_count, dtype=mx.int32) // top_k).astype(mx.int32)
    mx.eval(lhs)
    return lhs


def _verify_row_tile_descriptors(
    sorted_rhs: mx.array,
    *,
    m_rows: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Expert-grouped route tiles of at most ``m_rows`` routes each.

    ``sorted_rhs`` holds expert ids sorted nondecreasing. Unlike
    ``_expert_block_tile_descriptors`` the tile bound here is the route count
    instead of ``ceil(routes / tile) + num_experts``, which keeps the launched
    tile grid tight for verify-sized route counts (``M * top_k <= 64``) where
    the 256-expert term would otherwise dominate the grid by ~10x. Fully
    on-device: no host synchronization, O(routes^2) elementwise work on a
    matrix that is at most 64x64 for the verify widths this path serves.

    Raises above ``_VERIFY_MAX_ROUTES`` rather than quietly allocating a
    quadratic matrix: past that point the caller wants ``sorted_tiled``.
    """

    route_count = sorted_rhs.shape[0]
    if route_count > _VERIFY_MAX_ROUTES:
        raise ValueError(
            f"expert-grouped verify tiles support at most {_VERIFY_MAX_ROUTES} routes, "
            f"found {route_count}; this builder is O(routes^2) on device, so larger "
            "route counts belong on route_strategy='sorted_tiled'"
        )
    positions = mx.arange(route_count, dtype=mx.int32)
    # In a sorted array the count of strictly smaller entries is exactly the
    # index of the first occurrence of this route's expert.
    first_index = mx.sum((sorted_rhs[None, :] < sorted_rhs[:, None]).astype(mx.int32), axis=1)
    local_rank = positions - first_index.astype(mx.int32)
    starts = (local_rank % m_rows) == 0
    tile_id = (mx.cumsum(starts.astype(mx.int32)) - 1).astype(mx.int32)
    tile_counts = mx.sum((tile_id[None, :] == positions[:, None]).astype(mx.int32), axis=1)
    tile_offsets = mx.sum((tile_id[None, :] < positions[:, None]).astype(mx.int32), axis=1)
    # Tile slots past the last real tile carry count 0 and are early-returned by
    # the kernel; clamp their offset so the expert gather stays in bounds.
    safe_offsets = mx.minimum(tile_offsets, mx.array(route_count - 1, dtype=mx.int32)).astype(mx.int32)
    tile_experts = sorted_rhs[safe_offsets].astype(mx.int32)
    return tile_experts, safe_offsets, tile_counts.astype(mx.int32)


def _verify_rows_routed(
    x_mx: mx.array,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    flat_rhs: mx.array,
    flat_lhs: mx.array,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int,
    grouping: VerifyGrouping,
    m_rows: int,
    rows_per_threadgroup: int,
) -> mx.array:
    """Run the wide-M verify kernel and return ``[routes, output_dims]``."""

    route_count = flat_rhs.shape[0]
    if grouping == "flat":
        return gather_vqmm_verify_mrows_kernel(
            x_mx,
            codes,
            scales,
            codebook,
            flat_lhs,
            flat_rhs,
            *_flat_route_descriptors(route_count),
            route_count=route_count,
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            m_rows=1,
            rows_per_threadgroup=rows_per_threadgroup,
        )

    tile_width = m_rows
    order = mx.argsort(flat_rhs)
    inverse_order = _inverse_permutation(order)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_descriptors = _verify_row_tile_descriptors(sorted_rhs, m_rows=tile_width)
    routed = gather_vqmm_verify_mrows_kernel(
        x_mx,
        codes,
        scales,
        codebook,
        sorted_lhs,
        *tile_descriptors,
        route_count=route_count,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        m_rows=tile_width,
        rows_per_threadgroup=rows_per_threadgroup,
    )
    return routed[inverse_order]


def gather_vqmm_verify_rows(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    rhs_indices: mx.array | np.ndarray,
    *,
    lhs_indices: mx.array | np.ndarray | None = None,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    grouping: VerifyGrouping = "flat",
    m_rows: int | Literal["auto"] = "auto",
    rows_per_threadgroup: int | Literal["auto"] = "auto",
) -> mx.array:
    """Wide-M MTP verify pass over VQ-quantized routed experts.

    ``x`` is ``[verify_rows, input_dims]`` with ``rhs_indices``
    ``[verify_rows, top_k]``, and the result is ``[verify_rows, top_k,
    output_dims]`` — the same contract ``gather_vqmm`` uses at M=1, so a verify
    pass is one call instead of M decode-shaped calls. Pass ``lhs_indices`` for
    the per-route activation layout the down projection uses, in which case
    ``rhs_indices`` is flat and the result is ``[routes, output_dims]``.

    ``grouping="expert"`` sorts routes by expert and folds up to ``m_rows``
    same-expert verify rows into one threadgroup, reusing code bytes, decoded
    codebook entries, and group scales across them. ``grouping="flat"`` skips
    the sort and fuses the M decode launches into one without weight reuse.
    Both are byte-exact against the M=1 decoded kernel run per verify row.

    ``rows_per_threadgroup="auto"`` takes the row packing from
    ``m1_rows_per_threadgroup(input_dims, output_dims)`` — the same helper the
    M=1 decode path uses — so verify cannot drift off decode's tuning and
    diverge bit-for-bit while a parity gate pinned to a literal keeps passing.
    """

    if grouping not in ("expert", "flat"):
        raise ValueError("grouping must be 'expert' or 'flat'")
    x_mx = _as_mlx(x)
    rhs_mx = _as_mlx(rhs_indices)
    if x_mx.ndim != 2:
        raise ValueError(f"x must be 2D [rows, input_dims], found {x_mx.shape}")

    if lhs_indices is not None:
        flat_rhs = rhs_mx.reshape((-1,)).astype(mx.int32)
        flat_lhs = _as_mlx(lhs_indices).reshape((-1,)).astype(mx.int32)
        if flat_lhs.shape != flat_rhs.shape:
            raise ValueError("lhs_indices and rhs_indices must have the same flattened shape")
        output_shape = (flat_rhs.shape[0], output_dims)
    else:
        if rhs_mx.ndim != 2:
            raise ValueError("rhs_indices must be 2D [verify_rows, top_k] when lhs_indices is not provided")
        if rhs_mx.shape[0] != x_mx.shape[0]:
            raise ValueError(
                f"rhs_indices row dimension {rhs_mx.shape[0]} must match x row dimension {x_mx.shape[0]}"
            )
        top_k = rhs_mx.shape[1]
        flat_rhs = rhs_mx.reshape((-1,)).astype(mx.int32)
        flat_lhs = _token_route_lhs(flat_rhs.shape[0], top_k)
        output_shape = (x_mx.shape[0], top_k, output_dims)

    if flat_rhs.shape[0] == 0:
        return mx.zeros(output_shape, dtype=x_mx.dtype)

    if m_rows == "auto":
        # With the token layout the verify width is routes / top_k; with an
        # explicit per-route layout it is not recoverable, so cap the tile at 4.
        resolved_m_rows = (
            verify_m_rows(flat_rhs.shape[0], output_shape[1]) if lhs_indices is None else _VERIFY_DEFAULT_M_ROWS
        )
    else:
        resolved_m_rows = m_rows
    resolved_rows_per_threadgroup = (
        m1_rows_per_threadgroup(input_dims, output_dims)
        if rows_per_threadgroup == "auto"
        else rows_per_threadgroup
    )
    routed = _verify_rows_routed(
        x_mx,
        codes,
        scales,
        codebook,
        flat_rhs,
        flat_lhs,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        grouping=grouping,
        m_rows=resolved_m_rows,
        rows_per_threadgroup=resolved_rows_per_threadgroup,
    )
    return routed.reshape(output_shape)


def auto_selects_per_route_decoded(
    *,
    tokens: int,
    has_lhs_indices: bool,
    implementation: str,
    code_bits: int,
    input_dims: int,
    output_dims: int,
    group_size: int,
    activation_dtype: mx.Dtype,
) -> bool:
    """Whether ``route_strategy="auto"`` routes this call to per_route_decoded.

    Called only after ``auto`` has already declined ``sorted_tiled``, so the
    alternative here is the scalar kernel (``gather_vqmm.metal`` for the token
    layout, ``gather_vqmm_lhs.metal`` for the per-route layout). Both spend one
    256-thread threadgroup per output element per route; Wave 4 increment 1
    measured that structure at 3.2-4.5x slower than the per-route decoded
    kernel on the DeepSeek-V4-Flash routed-expert shapes, which is why this
    predicate exists. See ``docs/deepseek-v4-flash/research/wave4-wide-m-vqmm-survey.md`` and
    ``docs/deepseek-v4-flash/research/wave4-increment2-report.md``.

    The window this opens is defined by route count, not by token count: with
    top-6 routing ``route_count < 128`` means ``tokens <= 21``, so the affected
    band is the 2..21-token MTP verify window, but at other ``top_k`` values the
    same rule lands on a different token range. Nothing here hardcodes 21.

    Each guard is a decision to leave an existing dispatch alone:

    ``implementation``/``code_bits``
        The literal preconditions the ``per_route_decoded`` branch enforces.
        code_bits=16 (E8P) has no decoded verify kernel, so it stays scalar.
    ``tokens <= 1`` in the token layout
        The M=1 fast path is already a decoded kernel and is the measured best
        option at M=1. Flipping it would be a pure regression risk for no gain.
        The per-route layout has no M=1 fast path inside this op -- it lives in
        ``QuantizedVQSwitchLinear.__call__`` and runs before this op is called
        -- so there is nothing to preserve there.
    ``activation_dtype``
        float16 only -- see ``_VERIFY_ACTIVATION_DTYPES``. float32 and bfloat16
        activations keep the scalar path and are bit-for-bit unaffected by this
        increment, because the decoded kernel would have to narrow them to half
        before multiplying.
    ``group_size``/``input_dims`` divisibility and the row packing
        The verify kernel *raises* on 8D-unaligned group sizes and on row
        packings outside ``VERIFY_ROWS_PER_THREADGROUP_VALUES``, while the
        scalar kernel accepts them. A dispatch change must never turn a working
        call into an exception, so anything the verify kernel would reject stays
        on the path that already serves it.
    """

    if implementation != "metal" or code_bits != 8:
        return False
    if not has_lhs_indices and tokens <= 1:
        return False
    if activation_dtype not in _VERIFY_ACTIVATION_DTYPES:
        return False
    if input_dims <= 0 or output_dims <= 0:
        return False
    if input_dims % CODEWORD_DIM != 0 or group_size <= 0:
        return False
    if input_dims % group_size != 0 or group_size % CODEWORD_DIM != 0:
        return False
    if m1_rows_per_threadgroup(input_dims, output_dims) not in VERIFY_ROWS_PER_THREADGROUP_VALUES:
        return False
    return True


def auto_selects_nax_e8p(
    *,
    route_count: int,
    implementation: str,
    code_bits: int,
    input_dims: int,
    output_dims: int,
    group_size: int,
    activation_dtype: mx.Dtype,
) -> bool:
    """Whether an auto call satisfies the native E8P kernel contract."""

    return (
        implementation == "metal"
        and code_bits == 16
        and activation_dtype in (mx.float16, mx.bfloat16)
        and route_count > 0
        and input_dims > 0
        and output_dims > 0
        and input_dims % CODEWORD_DIM == 0
        and group_size > 0
        and group_size % CODEWORD_DIM == 0
        and input_dims % group_size == 0
    )


def _nax_e8p_sorted_steel_route_plan(
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    group_size: int,
) -> tuple[int, Literal["steel", "m32n64"]]:
    if (
        route_count <= _NAX_E8P_AIR_DOWN_SMALL_ROUTE_THRESHOLD
        and input_dims == 1_408
        and output_dims == 4_096
        and group_size == 352
    ):
        return _DOWN_BLOCK_MMA_ROUTE_TILE, "m32n64"
    return _LARGE_BLOCK_MMA_ROUTE_TILE, "steel"


def _validate_switch_metadata(
    x: mx.array,
    codes: mx.array,
    scales: mx.array,
    codebook: mx.array,
    indices: mx.array,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int,
    code_bits: Literal[8, 16],
) -> tuple[int, int]:
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by group_size")
    if x.shape[-1] != input_dims:
        raise ValueError(f"x trailing dimension must be {input_dims}, found {x.shape[-1]}")
    if indices.ndim < 1:
        raise ValueError("indices must have at least one dimension")
    if x.shape[:-1] != indices.shape[:-1] and x.shape[:-1] != indices.shape:
        raise ValueError(
            "x leading shape must either match indices leading shape for "
            f"token routing or the full indices shape for per-route routing; found x {x.shape} and indices {indices.shape}"
        )

    expected_code_dtype = "uint8" if code_bits == 8 else "uint16"
    dtype_name = getattr(codes.dtype, "name", str(codes.dtype)).rsplit(".", maxsplit=1)[-1]
    if dtype_name != expected_code_dtype:
        raise ValueError(f"codes dtype must be {expected_code_dtype}, found {codes.dtype}")
    codebook_dtype = getattr(codebook.dtype, "name", str(codebook.dtype)).rsplit(".", maxsplit=1)[-1]
    if codebook_dtype != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook.dtype}")
    if codebook.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook.shape}")
    if codes.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes.shape}")
    if scales.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales.shape}")

    num_experts = codes.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes.shape}")
    if scales.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales.shape}")
    return num_experts, indices.shape[-1]


def vq_switch_qmv(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    indices: mx.array | np.ndarray,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    codebook_duplication: Literal[1, 4, 8] = 1,
    implementation: Literal["metal", "reference"] = "metal",
    sorted_indices: bool = False,
) -> mx.array:
    """Run selected VQ experts for each token/top-k route.

    The A5 implementation is intentionally scalar over routed tokens and calls
    the proven qmv kernel for each selected expert. A6 replaces this with a
    sorted batching kernel while preserving this public output contract.
    """

    del sorted_indices
    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    indices_mx = _as_mlx(indices)
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    num_experts, top_k = _validate_switch_metadata(
        x_mx,
        codes_mx,
        scales_mx,
        codebook_mx,
        indices_mx,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        code_bits=code_bits,
    )
    flat_x = x_mx.reshape((-1, input_dims))
    indices_np = _indices_to_numpy(indices_mx)
    per_route_input = x_mx.shape[:-1] == indices_mx.shape
    flat_indices_np = indices_np.reshape((-1, 1)) if per_route_input else indices_np.reshape((-1, top_k))
    route_count = 1 if per_route_input else top_k
    if flat_indices_np.size and (flat_indices_np.min() < 0 or flat_indices_np.max() >= num_experts):
        raise ValueError(f"indices must be in [0, {num_experts}), found range [{flat_indices_np.min()}, {flat_indices_np.max()}]")

    token_outputs = []
    for token_idx in range(flat_x.shape[0]):
        route_outputs = []
        for route_idx in range(route_count):
            expert_idx = int(flat_indices_np[token_idx, route_idx])
            route_outputs.append(
                vq_qmv(
                    flat_x[token_idx],
                    codes_mx[expert_idx],
                    scales_mx[expert_idx],
                    codebook_mx,
                    in_dim=input_dims,
                    out_dim=output_dims,
                    group_size=group_size,
                    code_bits=code_bits,
                    implementation=implementation,
                )
            )
        token_outputs.append(mx.stack(route_outputs, axis=0))

    if per_route_input:
        return mx.stack([routes[0] for routes in token_outputs], axis=0).reshape((*x_mx.shape[:-1], output_dims))
    return mx.stack(token_outputs, axis=0).reshape((*x_mx.shape[:-1], top_k, output_dims))


def gather_vqmm_sorted_routes(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    sorted_rhs: mx.array | np.ndarray,
    sorted_lhs: mx.array | np.ndarray,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    codebook_duplication: Literal[1, 4, 8] = 8,
    implementation: Literal[
        "metal", "reference", "nax_e8", "nax_e8p", "nax_e8p_m32n64"
    ] = "metal",
    projection: SortedRouteProjection = "auto",
    validate_indices: bool = False,
    tile_descriptors: tuple[mx.array, mx.array, mx.array] | None = None,
    x_pre_sorted: bool = False,
) -> mx.array:
    """Evaluate already-sorted routed VQ matmuls without sorting or scattering.

    ``sorted_rhs`` contains expert ids sorted by expert, and ``sorted_lhs``
    contains activation row ids into ``x`` for each route. The returned tensor is
    ``[routes, output_dims]`` in that same sorted route order. Set
    ``validate_indices=True`` for external or test callers that need clean
    Python errors for bad explicit metadata; it materializes the index vectors
    on the host and is therefore left off for the hot resident path.
    """

    if projection not in ("auto", "gate_up", "down"):
        raise ValueError("projection must be 'auto', 'gate_up', or 'down'")
    x_mx = _as_mlx(x)
    rhs_mx = _as_mlx(sorted_rhs).reshape((-1,)).astype(mx.int32)
    lhs_mx = _as_mlx(sorted_lhs).reshape((-1,)).astype(mx.int32)
    if x_mx.ndim != 2:
        raise ValueError(f"x must be 2D [tokens_or_routes, input_dims], found {x_mx.shape}")
    if x_mx.shape[1] != input_dims:
        raise ValueError(f"x trailing dimension must be {input_dims}, found {x_mx.shape[1]}")
    if rhs_mx.shape != lhs_mx.shape:
        raise ValueError("sorted_rhs and sorted_lhs must have the same flattened shape")
    if codebook_duplication not in (1, 4, 8):
        raise ValueError("codebook_duplication must be 1, 4, or 8")

    route_count = rhs_mx.shape[0]
    if route_count == 0:
        return mx.zeros((0, output_dims), dtype=x_mx.dtype)
    num_experts = _as_mlx(codes).shape[0]
    if validate_indices:
        rhs_np = _indices_to_numpy(rhs_mx)
        lhs_np = _indices_to_numpy(lhs_mx)
        if rhs_np.size and (rhs_np.min() < 0 or rhs_np.max() >= num_experts):
            raise ValueError(
                f"sorted_rhs must be in [0, {num_experts}), "
                f"found range [{rhs_np.min()}, {rhs_np.max()}]"
            )
        if lhs_np.size and (lhs_np.min() < 0 or lhs_np.max() >= x_mx.shape[0]):
            raise ValueError(
                f"sorted_lhs must be in [0, {x_mx.shape[0]}), "
                f"found range [{lhs_np.min()}, {lhs_np.max()}]"
            )
        if rhs_np.size > 1 and np.any(rhs_np[1:] < rhs_np[:-1]):
            raise ValueError("sorted_rhs must be sorted nondecreasing by expert id")

    projection_role = _infer_air_projection(input_dims, output_dims) if projection == "auto" else projection
    is_down_projection = projection_role == "down"
    # Skipping the activation gather requires ``x`` to already be in sorted-route
    # order. Only the caller knows that: the shared prefill path computes the down
    # input from sorted gate/up outputs and passes an ``arange`` lhs, while
    # ``gather_vqmm`` passes ``flat_lhs[argsort(rhs)]`` with unsorted ``x``. This
    # used to be inferred from ``x.shape[0] == route_count``, which is true in both
    # cases and only valid in the first, so the three-chain M=1 decode path was
    # computing the down projection against permuted activations.
    skip_activation_gather = is_down_projection and x_pre_sorted and x_mx.shape[0] == route_count
    if implementation == "nax_e8":
        if code_bits != 8:
            raise ValueError("nax_e8 sorted routes require code_bits=8")
        if not nax.is_available():
            raise RuntimeError("native VQ NAX extension is not built")
        if group_size % CODEWORD_DIM != 0:
            raise ValueError("nax_e8 sorted routes require group_size divisible by 8")
        codes_mx = _as_mlx(codes)
        scales_mx = _as_mlx(scales)
        codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)
        route_tile = _LARGE_BLOCK_MMA_ROUTE_TILE
        active_tile_descriptors = tile_descriptors or _expert_block_tile_descriptors(
            rhs_mx,
            num_experts=num_experts,
            route_tile=route_tile,
        )
        if skip_activation_gather:
            sorted_x = x_mx.astype(mx.float16)
        else:
            sorted_x = x_mx[lhs_mx].astype(mx.float16)
        return nax.nax_e8_fp16_sorted_steel_matmul(
            sorted_x,
            codes_mx,
            scales_mx,
            *active_tile_descriptors,
            codebook_mx,
            group_size=group_size,
        )
    if implementation in ("nax_e8p", "nax_e8p_m32n64"):
        if code_bits != 16:
            raise ValueError("nax_e8p sorted routes require code_bits=16")
        if not nax.is_available():
            raise RuntimeError("native VQ NAX extension is not built")
        if group_size % CODEWORD_DIM != 0:
            raise ValueError("nax_e8p sorted routes require group_size divisible by 8")
        codes_mx = _as_mlx(codes)
        scales_mx = _as_mlx(scales)
        codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)
        if implementation == "nax_e8p_m32n64":
            route_tile, kernel_name = _DOWN_BLOCK_MMA_ROUTE_TILE, "m32n64"
            active_tile_descriptors = tile_descriptors or _expert_block_tile_descriptors(
                rhs_mx, num_experts=num_experts, route_tile=route_tile
            )
        elif tile_descriptors is None:
            route_tile, kernel_name = _nax_e8p_sorted_steel_route_plan(
                route_count=route_count,
                input_dims=input_dims,
                output_dims=output_dims,
                group_size=group_size,
            )
            # Kernel/tile selection only, so the shape test is the right premise
            # here: it asks whether the activation rows *are* the routes. Skipping
            # the gather needs the stronger ``x_pre_sorted`` fact instead.
            if is_down_projection and x_mx.shape[0] == route_count:
                route_tile, kernel_name = _LARGE_BLOCK_MMA_ROUTE_TILE, "steel"
            active_tile_descriptors = _expert_block_tile_descriptors(
                rhs_mx,
                num_experts=num_experts,
                route_tile=route_tile,
            )
        else:
            kernel_name = "steel"
            active_tile_descriptors = tile_descriptors
        if skip_activation_gather:
            sorted_x = x_mx.astype(mx.float16)
        else:
            sorted_x = x_mx[lhs_mx].astype(mx.float16)
        if kernel_name == "m32n64":
            return nax.nax_e8p_fp16_sorted_steel_m32n64_matmul(
                sorted_x,
                codes_mx,
                scales_mx,
                *active_tile_descriptors,
                codebook_mx,
                group_size=group_size,
            )
        return nax.nax_e8p_fp16_sorted_steel_matmul(
            sorted_x,
            codes_mx,
            scales_mx,
            *active_tile_descriptors,
            codebook_mx,
            group_size=group_size,
        )
    if implementation == "metal":
        if is_down_projection and route_count >= _DOWN_BLOCK_MMA_ROUTE_THRESHOLD:
            block_kernel = gather_vqmm_mma4x4_blocks_kernel
            route_tile = _DOWN_BLOCK_MMA_ROUTE_TILE
            if route_count >= _LARGE_BLOCK_MMA_ROUTE_THRESHOLD:
                block_kernel = gather_vqmm_mma8x4_blocks_kernel
                route_tile = _LARGE_BLOCK_MMA_ROUTE_TILE
            tile_descriptors = _expert_block_tile_descriptors(
                rhs_mx,
                num_experts=num_experts,
                route_tile=route_tile,
            )
            if code_bits == 8 and group_size % CODEWORD_DIM == 0:
                cwdecode_kernel = (
                    gather_vqmm_mma_k64_cwdecode_blocks_kernel
                    if input_dims % 64 == 0
                    else gather_vqmm_mma_k32_cwdecode_blocks_kernel
                    if input_dims % 32 == 0
                    else gather_vqmm_mma_cwdecode_blocks_kernel
                )
                cwdecode_kwargs = {
                    "route_count": route_count,
                    "input_dims": input_dims,
                    "output_dims": output_dims,
                    "group_size": group_size,
                    "route_tile": route_tile,
                }
                if cwdecode_kernel is gather_vqmm_mma_k64_cwdecode_blocks_kernel:
                    cwdecode_kwargs["row_groups"] = 8 if route_tile == 32 else 4
                return cwdecode_kernel(
                    x_mx,
                    codes,
                    scales,
                    codebook,
                    lhs_mx,
                    *tile_descriptors,
                    **cwdecode_kwargs,
                )
            return block_kernel(
                x_mx,
                codes,
                scales,
                codebook,
                lhs_mx,
                *tile_descriptors,
                route_count=route_count,
                input_dims=input_dims,
                output_dims=output_dims,
                group_size=group_size,
                code_bits=code_bits,
            )
        if not is_down_projection and route_count >= _GATE_UP_MMA_ROUTE_THRESHOLD:
            block_kernel = gather_vqmm_mma4x4_blocks_kernel
            route_tile = _GATE_UP_MMA_ROUTE_TILE
            if route_count >= _LARGE_BLOCK_MMA_ROUTE_THRESHOLD:
                block_kernel = gather_vqmm_mma8x4_blocks_kernel
                route_tile = _LARGE_BLOCK_MMA_ROUTE_TILE
            tile_descriptors = _expert_block_tile_descriptors(
                rhs_mx,
                num_experts=num_experts,
                route_tile=route_tile,
            )
            if code_bits == 8 and group_size % CODEWORD_DIM == 0:
                cwdecode_kernel = (
                    gather_vqmm_mma_k64_cwdecode_blocks_kernel
                    if input_dims % 64 == 0
                    else gather_vqmm_mma_k32_cwdecode_blocks_kernel
                    if input_dims % 32 == 0
                    else gather_vqmm_mma_cwdecode_blocks_kernel
                )
                cwdecode_kwargs = {
                    "route_count": route_count,
                    "input_dims": input_dims,
                    "output_dims": output_dims,
                    "group_size": group_size,
                    "route_tile": route_tile,
                }
                if cwdecode_kernel is gather_vqmm_mma_k64_cwdecode_blocks_kernel:
                    cwdecode_kwargs["row_groups"] = 8 if route_tile == 32 else 4
                return cwdecode_kernel(
                    x_mx,
                    codes,
                    scales,
                    codebook,
                    lhs_mx,
                    *tile_descriptors,
                    **cwdecode_kwargs,
                )
            return block_kernel(
                x_mx,
                codes,
                scales,
                codebook,
                lhs_mx,
                *tile_descriptors,
                route_count=route_count,
                input_dims=input_dims,
                output_dims=output_dims,
                group_size=group_size,
                code_bits=code_bits,
            )

        m_tile = 64 if is_down_projection else 16
        n_tile = 4 if is_down_projection else 8
        tiled_codebook_duplication = 1 if is_down_projection else codebook_duplication
        return gather_vqmm_tiled_down_kernel(
            x_mx,
            codes,
            scales,
            codebook,
            rhs_mx,
            lhs_mx,
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            code_bits=code_bits,
            codebook_duplication=tiled_codebook_duplication,
            m_tile=m_tile,
            n_tile=n_tile,
            k_tile_dims=group_size,
        )

    return vq_switch_qmv(
        x_mx[lhs_mx],
        codes,
        scales,
        codebook,
        rhs_mx,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        code_bits=code_bits,
        implementation=implementation,
        sorted_indices=True,
    )


def gather_vqmm(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    rhs_indices: mx.array | np.ndarray,
    *,
    lhs_indices: mx.array | np.ndarray | None = None,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    codebook_duplication: Literal[1, 4, 8] = 1,
    implementation: Literal["metal", "reference"] = "metal",
    sorted_indices: bool = False,
    route_strategy: RouteStrategy = "auto",
) -> mx.array:
    """VQ analog of gather-qmm for routed expert matrices.

    This contract implementation keeps the public shape expected by GLM MoE:
    ``x`` is ``[tokens, hidden]``, ``rhs_indices`` is ``[tokens, top_k]``, and
    the result is ``[tokens, top_k, out]``. ``route_strategy="sorted_tiled"``
    sorts routes by expert id on-device, passes ``lhs_indices`` to the kernel,
    and scatters back to the original token/top-k layout without materializing a
    sorted activation tensor.

    ``route_strategy="auto"`` resolves in four steps, in this order:

    1. Eligible E8P calls use the native sorted M32/N64 kernel. If the native
       extension is unavailable, a warning is emitted and the old selection
       below remains in force.
    2. ``sorted_indices`` or ``route_count >= _SORTED_TILED_ROUTE_THRESHOLD``
       selects ``sorted_tiled`` -- prefill and long batches.
    3. Otherwise, if ``auto_selects_per_route_decoded`` holds, select
       ``per_route_decoded``. This is the MTP verify window (2..21 tokens at
       top-6) and it is a **kernel change with a numeric consequence**: the
       scalar kernel it replaces accumulates each 8D codeword as eight float32
       FMAs, while the decoded kernel evaluates it as two ``dot(half4, half4)``
       products scaled in float32. Neither is a rounding of the other; each is
       exact with respect to its own accumulation order. The flip is taken
       because it is 3.2-4.5x faster *and* because it puts this window on the
       same kernel family as the M=1 decode path, which is the agreement MTP
       verify actually needs. Deltas are quantified in
       ``tests/test_gather_vqmm_verify_rows.py`` and the increment-2 report.
    4. Otherwise ``direct`` -- the M=1 fast path, or the scalar kernel for
       shapes and dtypes the decoded verify kernel does not serve.
    """

    x_mx = _as_mlx(x)
    rhs_mx = _as_mlx(rhs_indices)
    if route_strategy not in ("auto", "direct", "sorted_tiled", "per_route_decoded"):
        raise ValueError(
            "route_strategy must be 'auto', 'direct', 'sorted_tiled', or 'per_route_decoded'"
        )
    if x_mx.ndim != 2:
        raise ValueError(f"x must be 2D [tokens, hidden], found {x_mx.shape}")
    if rhs_mx.ndim not in (1, 2):
        raise ValueError(f"rhs_indices must be 1D or 2D, found {rhs_mx.shape}")

    if lhs_indices is not None:
        lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
        flat_rhs = rhs_mx.reshape((-1,))
        if lhs_mx.shape != flat_rhs.shape:
            raise ValueError("lhs_indices and rhs_indices must have the same flattened shape")
        output_shape = (flat_rhs.shape[0], output_dims)
        flat_lhs = lhs_mx.astype(mx.int32)
    else:
        if rhs_mx.ndim != 2:
            raise ValueError("rhs_indices must be 2D [tokens, top_k] when lhs_indices is not provided")
        if rhs_mx.shape[0] != x_mx.shape[0]:
            raise ValueError(
                f"rhs_indices token dimension {rhs_mx.shape[0]} must match x token dimension {x_mx.shape[0]}"
            )
        top_k = rhs_mx.shape[1]
        flat_rhs = rhs_mx.reshape((-1,))
        flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // top_k).astype(mx.int32)
        output_shape = (x_mx.shape[0], top_k, output_dims)

    route_count = flat_rhs.shape[0]
    selected_strategy = route_strategy
    selected_implementation = implementation
    if selected_strategy == "auto":
        e8p_eligible = auto_selects_nax_e8p(
            route_count=route_count,
            implementation=implementation,
            code_bits=code_bits,
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            activation_dtype=x_mx.dtype,
        )
        if e8p_eligible:
            if nax.is_available():
                selected_strategy = "sorted_tiled"
                selected_implementation = "nax_e8p_m32n64"
            else:
                warnings.warn(
                    "native VQ NAX extension is unavailable; the E8P fast path is not being used. "
                    "Rebuild native/vq_nax_ext for this Python environment.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if selected_strategy == "auto" and (
            sorted_indices or route_count >= _SORTED_TILED_ROUTE_THRESHOLD
        ):
            selected_strategy = "sorted_tiled"
        elif selected_strategy == "auto" and auto_selects_per_route_decoded(
            tokens=x_mx.shape[0],
            has_lhs_indices=lhs_indices is not None,
            implementation=implementation,
            code_bits=code_bits,
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            activation_dtype=x_mx.dtype,
        ):
            selected_strategy = "per_route_decoded"
        elif selected_strategy == "auto":
            selected_strategy = "direct"

    if selected_strategy == "per_route_decoded":
        trace = _GATHER_VQMM_TRACE.get()
        if trace is not None:
            trace.append(
                {
                    "token_rows": int(x_mx.shape[0]),
                    "route_count": int(route_count),
                    "has_lhs_indices": lhs_indices is not None,
                    "input_dims": int(input_dims),
                    "output_dims": int(output_dims),
                    "group_size": int(group_size),
                    "code_bits": int(code_bits),
                    "strategy": selected_strategy,
                    "implementation": selected_implementation,
                }
            )
        if implementation != "metal":
            raise ValueError("route_strategy='per_route_decoded' requires implementation='metal'")
        if code_bits != 8:
            raise ValueError("route_strategy='per_route_decoded' requires code_bits=8")
        # Flat grouping (one route per tile, m_rows=1) is the measured default:
        # at verify widths (M <= 8, top-6 of 256 experts) same-expert routes are
        # rare, so folding the M decode launches into one per-route launch beats
        # sorting routes by expert. This strategy therefore does not widen M at
        # all -- hence the name. See docs/deepseek-v4-flash/research/wave4-wide-m-vqmm-survey.md.
        #
        # Reached both explicitly and from "auto" via
        # auto_selects_per_route_decoded. The two guards above are kept as hard
        # errors rather than folded into that predicate: an explicit caller
        # asking for this strategy with code_bits=16 has made a mistake and
        # should hear about it, whereas "auto" must silently decline.
        routed = _verify_rows_routed(
            x_mx,
            codes,
            scales,
            codebook,
            flat_rhs.astype(mx.int32),
            flat_lhs,
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            grouping="flat",
            m_rows=1,
            rows_per_threadgroup=m1_rows_per_threadgroup(input_dims, output_dims),
        )
        return routed.reshape(output_shape)

    trace = _GATHER_VQMM_TRACE.get()
    if trace is not None:
        trace.append(
            {
                "token_rows": int(x_mx.shape[0]),
                "route_count": int(route_count),
                "has_lhs_indices": lhs_indices is not None,
                "input_dims": int(input_dims),
                "output_dims": int(output_dims),
                "group_size": int(group_size),
                "code_bits": int(code_bits),
                "strategy": selected_strategy,
                "implementation": selected_implementation,
            }
        )

    if implementation == "metal":
        if selected_strategy == "direct" and lhs_indices is None:
            if x_mx.shape[0] == 1 and code_bits == 8:
                return gather_vqmm_m1_kernel(
                    x_mx,
                    codes,
                    scales,
                    codebook,
                    rhs_mx,
                    input_dims=input_dims,
                    output_dims=output_dims,
                    group_size=group_size,
                    code_bits=code_bits,
                    rows_per_threadgroup=m1_rows_per_threadgroup(input_dims, output_dims),
                    use_threadgroup_codebook=m1_use_threadgroup_codebook(input_dims, output_dims),
                    use_decoded_codebook=m1_use_decoded_codebook(input_dims, output_dims),
                )
            return gather_vqmm_kernel(
                x_mx,
                codes,
                scales,
                codebook,
                rhs_mx,
                input_dims=input_dims,
                output_dims=output_dims,
                group_size=group_size,
                code_bits=code_bits,
                codebook_duplication=codebook_duplication,
            )

        rhs_for_kernel = flat_rhs.astype(mx.int32)
        lhs_for_kernel = flat_lhs
        inverse_order = None
        if selected_strategy == "sorted_tiled":
            order = mx.argsort(rhs_for_kernel)
            inverse_order = _inverse_permutation(order)
            rhs_for_kernel = rhs_for_kernel[order]
            lhs_for_kernel = lhs_for_kernel[order]

        if selected_strategy == "sorted_tiled":
            routed = gather_vqmm_sorted_routes(
                x_mx,
                codes,
                scales,
                codebook,
                rhs_for_kernel,
                lhs_for_kernel,
                input_dims=input_dims,
                output_dims=output_dims,
                group_size=group_size,
                code_bits=code_bits,
                codebook_duplication=codebook_duplication,
                implementation=selected_implementation,
                projection=_infer_air_projection(input_dims, output_dims),
            )
        else:
            routed = gather_vqmm_lhs_kernel(
                x_mx,
                codes,
                scales,
                codebook,
                rhs_for_kernel,
                lhs_for_kernel,
                input_dims=input_dims,
                output_dims=output_dims,
                group_size=group_size,
                code_bits=code_bits,
                codebook_duplication=codebook_duplication,
            )
        if inverse_order is not None:
            routed = routed[inverse_order]
        return routed.reshape(output_shape)

    if selected_strategy == "direct" and lhs_indices is None:
        return vq_switch_qmv(
            x_mx,
            codes,
            scales,
            codebook,
            rhs_mx,
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            code_bits=code_bits,
            implementation=implementation,
            sorted_indices=False,
        )

    rhs_np = _indices_to_numpy(flat_rhs)
    lhs_np = _indices_to_numpy(flat_lhs)
    if selected_strategy == "direct":
        selected_y = vq_switch_qmv(
            x_mx[mx.array(lhs_np)],
            codes,
            scales,
            codebook,
            mx.array(rhs_np),
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            code_bits=code_bits,
            implementation=implementation,
            sorted_indices=False,
        )
        return selected_y.reshape(output_shape)

    order = np.argsort(rhs_np, kind="stable")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.shape[0])
    sorted_y = vq_switch_qmv(
        x_mx[mx.array(lhs_np[order])],
        codes,
        scales,
        codebook,
        mx.array(rhs_np[order]),
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        code_bits=code_bits,
        implementation=implementation,
        sorted_indices=True,
    )
    return sorted_y[mx.array(inv_order)].reshape(output_shape)
