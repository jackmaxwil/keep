"""Parity gate for the wide-M-capable VQ verify kernels.

MTP verify pushes M=2-8 rows through the same expert-gathered projections that
decode pushes at M=1. Two parity contracts are exercised here, and they are not
the same contract:

**Byte-exact vs looped M=1** (the gate). Every verify variant must produce
bit-identical output to ``gather_vqmm_m1_kernel_unchecked`` at
``use_decoded_codebook=True`` and the **same** ``rows_per_threadgroup``, run
once per verify row. The row packing is always sourced from
``m1_rows_per_threadgroup(input_dims, output_dims)`` on both sides rather than
from a literal, so if that helper is ever tuned off 32 this gate moves with it
instead of pinning a value production no longer uses. The reference is exact
because the verify kernel is a literal row-tiling of the M=1 lane discipline:
same lane-to-codeword partition, same float32 accumulator, same shuffle-down
reduction tree. Reassociation would break this, so it is asserted on raw bytes,
not with a tolerance.

Only row packings in ``VERIFY_ROWS_PER_THREADGROUP_VALUES`` are exact:
``rows_per_threadgroup=8`` gives 32 lanes per row, where the M=1 reference
switches to ``simd_sum`` instead of the shuffle-down tree. That is a genuinely
different accumulation order, so the wrapper rejects it and
``test_verify_rows_rejects_row_packings_that_break_exactness`` pins the
rejection. Note that the small fixture below cannot expose reduction-order
bugs at all (its 2 codewords per group spread over 8 lanes leaves most lanes
idle), which is why the row-packing sweep runs at the real V4 shapes.

**Approximate vs float32 dequantized matmul** (a sanity bound, not the gate).
The kernel evaluates each 8D codeword as ``dot(half4, half4)``, which is a
half-precision 4-term dot, then scales in float32. A float32 dequantize-then-
matmul reference therefore cannot be matched bit-for-bit by construction; it is
checked with a tolerance to catch layout and indexing errors that a
self-consistent byte-exact comparison would miss.
"""

from __future__ import annotations

import zlib

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.codebook.e8 import decode_weight_matrix, e8_1bit_packed, e8p_packed_abs_grid
from mlx_vq.kernels.gather_vqmm import (
    VERIFY_MROWS_VALUES,
    VERIFY_ROWS_PER_THREADGROUP_VALUES,
    gather_vqmm_m1_kernel_unchecked,
    gather_vqmm_verify_mrows_kernel,
    m1_rows_per_threadgroup,
    verify_m_rows,
)
from mlx_vq.nn.switch_linear import QuantizedVQSwitchLinear
import mlx_vq.ops.vq_switch as vq_switch
from mlx_vq.ops.vq_switch import gather_vqmm, gather_vqmm_verify_rows

# DeepSeek-V4-Flash routed-expert shapes (measured 2026-08-11): gate/up are
# K=4096 -> N=2048 and down expands K=2048 -> N=4096. The expert count is cut
# from the model's 256 to 8 so the fixtures stay test-sized; expert count does
# not enter the kernel's accumulation order, only the gather address.
V4_SHAPES = (
    ("gate", 4096, 2048),
    ("up", 4096, 2048),
    ("down", 2048, 4096),
)
V4_TOP_K = 6
V4_GROUP_SIZE = 512
V4_EXPERTS = 8

# Small proportional shape for the exhaustive sweeps, so M x pattern x m_rows
# combinatorics do not each allocate megabytes of codes.
SMALL_IN, SMALL_OUT, SMALL_GROUP, SMALL_EXPERTS, SMALL_TOP_K = 64, 32, 16, 6, 3


def _seed(*parts: object) -> int:
    """Deterministic seed from a canonical string.

    ``hash()`` on str is randomized per process unless PYTHONHASHSEED is set, so
    seeding from it makes failures unreproducible. CRC32 is stable across runs,
    interpreters, and platforms.
    """

    return zlib.crc32("|".join(str(part) for part in parts).encode()) & 0x7FFFFFFF


def _fixture(
    *,
    experts: int,
    out_dim: int,
    in_dim: int,
    group_size: int,
    seed: int,
) -> QuantizedVQSwitchLinear:
    rng = np.random.default_rng(seed)
    weight = rng.normal(scale=0.04, size=(experts, out_dim, in_dim)).astype(np.float32)
    return QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=group_size)


def _routes(
    *,
    pattern: str,
    verify_rows: int,
    top_k: int,
    experts: int,
    seed: int,
) -> np.ndarray:
    """Verify-row routing patterns in increasing order of same-expert overlap."""

    rng = np.random.default_rng(seed)
    if pattern == "uniform":
        return rng.integers(0, experts, size=(verify_rows, top_k), dtype=np.int32)
    if pattern == "skewed":
        hot = max(2, experts // 3)
        return rng.integers(0, hot, size=(verify_rows, top_k), dtype=np.int32)
    if pattern == "duplicate":
        # Every verify row selects the same experts: maximum overlap, so every
        # route tile in expert grouping is full.
        base = rng.choice(experts, size=(top_k,), replace=False).astype(np.int32)
        return np.tile(base, (verify_rows, 1))
    if pattern == "single_expert":
        # All routes collapse onto one expert: exercises tiles that fill and
        # then spill into a partial tail tile.
        return np.zeros((verify_rows, top_k), dtype=np.int32)
    raise ValueError(f"unknown routing pattern {pattern!r}")


def _looped_m1_reference(
    layer: QuantizedVQSwitchLinear,
    x: mx.array,
    rhs: mx.array,
    *,
    out_dim: int,
    rows_per_threadgroup: int | None = None,
) -> mx.array:
    """The M=1 production decode kernel, run once per verify row.

    The row packing defaults to ``m1_rows_per_threadgroup(in_dim, out_dim)``,
    the same helper the production M=1 decode path and the verify path both
    consult, so this reference tracks decode's tuning rather than pinning a
    literal that could silently drift away from it.
    """

    packing = (
        m1_rows_per_threadgroup(x.shape[1], out_dim)
        if rows_per_threadgroup is None
        else rows_per_threadgroup
    )
    rows = [
        gather_vqmm_m1_kernel_unchecked(
            x[i : i + 1],
            layer.codes,
            layer.scales,
            layer.codebook,
            rhs[i : i + 1],
            output_dims=out_dim,
            code_bits=8,
            rows_per_threadgroup=packing,
            use_threadgroup_codebook=False,
            use_decoded_codebook=True,
        )
        for i in range(x.shape[0])
    ]
    return mx.concatenate(rows, axis=0)


def _assert_byte_exact(actual: mx.array, expected: mx.array, context: str) -> None:
    mx.eval(actual, expected)
    actual_np = np.array(actual)
    expected_np = np.array(expected)
    assert actual_np.dtype == expected_np.dtype, f"{context}: dtype {actual_np.dtype} != {expected_np.dtype}"
    assert actual_np.shape == expected_np.shape, f"{context}: shape {actual_np.shape} != {expected_np.shape}"
    if actual_np.tobytes() == expected_np.tobytes():
        return
    mismatch = int(np.count_nonzero(actual_np != expected_np))
    worst = float(np.max(np.abs(actual_np.astype(np.float64) - expected_np.astype(np.float64))))
    raise AssertionError(
        f"{context}: verify output is not byte-exact against looped M=1; "
        f"{mismatch}/{actual_np.size} elements differ, max abs delta {worst:.3e}"
    )


def _float32_dequant_reference(
    layer: QuantizedVQSwitchLinear,
    x: mx.array,
    rhs: np.ndarray,
    *,
    in_dim: int,
) -> np.ndarray:
    """Dequantize codes to float32 and matmul, in float32 throughout."""

    codes = np.array(layer.codes)
    scales = np.array(layer.scales)
    codebook = np.array(layer.codebook)
    x_np = np.array(x).astype(np.float32)
    verify_rows, top_k = rhs.shape
    out = np.zeros((verify_rows, top_k, codes.shape[1]), dtype=np.float32)
    for row in range(verify_rows):
        for slot in range(top_k):
            expert = int(rhs[row, slot])
            weight = decode_weight_matrix(
                codes[expert], scales[expert], code_bits=8, codebook=codebook
            )
            assert weight.shape[1] == in_dim
            out[row, slot] = weight.astype(np.float32) @ x_np[row]
    return out


# --------------------------------------------------------------------------
# Byte-exact gate: the three V4 shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("proj,in_dim,out_dim", V4_SHAPES, ids=[s[0] for s in V4_SHAPES])
@pytest.mark.parametrize("verify_rows", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("pattern", ["uniform", "skewed", "duplicate"])
def test_verify_rows_byte_exact_against_looped_m1_at_v4_shapes(
    proj: str, in_dim: int, out_dim: int, verify_rows: int, pattern: str
) -> None:
    layer = _fixture(
        experts=V4_EXPERTS,
        out_dim=out_dim,
        in_dim=in_dim,
        group_size=V4_GROUP_SIZE,
        seed=4096 + verify_rows,
    )
    rhs_np = _routes(
        pattern=pattern,
        verify_rows=verify_rows,
        top_k=V4_TOP_K,
        experts=V4_EXPERTS,
        seed=_seed(proj, verify_rows, pattern),
    )
    rng = np.random.default_rng(7 + verify_rows)
    x = mx.array(rng.normal(size=(verify_rows, in_dim)).astype(np.float16))
    rhs = mx.array(rhs_np)
    mx.eval(x, rhs)

    expected = _looped_m1_reference(layer, x, rhs, out_dim=out_dim)
    for grouping in ("flat", "expert"):
        actual = gather_vqmm_verify_rows(
            x,
            layer.codes,
            layer.scales,
            layer.codebook,
            rhs,
            input_dims=in_dim,
            output_dims=out_dim,
            group_size=V4_GROUP_SIZE,
            grouping=grouping,
        )
        assert actual.shape == (verify_rows, V4_TOP_K, out_dim)
        _assert_byte_exact(
            actual, expected, f"{proj} M={verify_rows} {pattern} grouping={grouping}"
        )


@pytest.mark.parametrize("proj,in_dim,out_dim", V4_SHAPES, ids=[s[0] for s in V4_SHAPES])
@pytest.mark.parametrize("rows_per_threadgroup", VERIFY_ROWS_PER_THREADGROUP_VALUES)
def test_verify_rows_byte_exact_across_row_packings_at_v4_shapes(
    proj: str, in_dim: int, out_dim: int, rows_per_threadgroup: int
) -> None:
    """Every accepted row packing must be exact against the *same* packing.

    Runs at the real V4 shapes on purpose: the small fixture spreads 2 codewords
    per scale group over 8 lanes, so most lanes contribute nothing and a
    reduction-order divergence cannot show up. At K=4096/group=512 each lane
    accumulates 8 codewords per group, which is what exposed the
    rows_per_threadgroup=8 simd_sum divergence.
    """

    layer = _fixture(
        experts=V4_EXPERTS,
        out_dim=out_dim,
        in_dim=in_dim,
        group_size=V4_GROUP_SIZE,
        seed=_seed("packing", proj, rows_per_threadgroup),
    )
    verify_rows = 4
    rhs_np = _routes(
        pattern="uniform",
        verify_rows=verify_rows,
        top_k=V4_TOP_K,
        experts=V4_EXPERTS,
        seed=_seed("packing-routes", proj, rows_per_threadgroup),
    )
    rng = np.random.default_rng(_seed("packing-x", proj, rows_per_threadgroup))
    x = mx.array(rng.normal(size=(verify_rows, in_dim)).astype(np.float16))
    rhs = mx.array(rhs_np)
    mx.eval(x, rhs)

    expected = _looped_m1_reference(
        layer, x, rhs, out_dim=out_dim, rows_per_threadgroup=rows_per_threadgroup
    )
    for grouping in ("flat", "expert"):
        actual = gather_vqmm_verify_rows(
            x,
            layer.codes,
            layer.scales,
            layer.codebook,
            rhs,
            input_dims=in_dim,
            output_dims=out_dim,
            group_size=V4_GROUP_SIZE,
            grouping=grouping,
            rows_per_threadgroup=rows_per_threadgroup,
        )
        _assert_byte_exact(
            actual, expected, f"{proj} rptg={rows_per_threadgroup} grouping={grouping}"
        )


def test_verify_rows_rejects_row_packings_that_break_exactness() -> None:
    """rows_per_threadgroup=8 gives 32 lanes/row, where the M=1 reference uses
    simd_sum rather than the shuffle-down tree this kernel implements. Measured
    divergence at the V4 gate/up shape: 46/49152 elements, max delta ~4.9e-4.
    It must raise rather than silently return non-exact output."""

    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=_seed("reject-packing"),
    )
    x = mx.zeros((2, SMALL_IN), dtype=mx.float16)
    rhs = mx.zeros((2, SMALL_TOP_K), dtype=mx.int32)

    assert 8 not in VERIFY_ROWS_PER_THREADGROUP_VALUES
    for bad in (8, 1, 128, 256):
        with pytest.raises(ValueError, match="rows_per_threadgroup must be one of"):
            gather_vqmm_verify_rows(
                x,
                layer.codes,
                layer.scales,
                layer.codebook,
                rhs,
                input_dims=SMALL_IN,
                output_dims=SMALL_OUT,
                group_size=SMALL_GROUP,
                rows_per_threadgroup=bad,
            )


def test_verify_row_packing_defaults_to_the_m1_decode_helper() -> None:
    """The verify path must not hardcode a row packing.

    If ``m1_rows_per_threadgroup`` is ever tuned off 32, verify has to move with
    it or decode and verify diverge bit-for-bit while a literal-pinned gate keeps
    passing. This asserts the wiring, not the current value.
    """

    for _proj, in_dim, out_dim in V4_SHAPES:
        helper_value = m1_rows_per_threadgroup(in_dim, out_dim)
        assert helper_value in VERIFY_ROWS_PER_THREADGROUP_VALUES, (
            "m1_rows_per_threadgroup returns a packing the verify kernel rejects; "
            "verify must be extended before decode is tuned to it"
        )

    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=_seed("auto-packing"),
    )
    rng = np.random.default_rng(_seed("auto-packing-x"))
    x = mx.array(rng.normal(size=(4, SMALL_IN)).astype(np.float16))
    rhs = mx.array(rng.integers(0, SMALL_EXPERTS, size=(4, SMALL_TOP_K), dtype=np.int32))
    mx.eval(x, rhs)

    kwargs = dict(
        input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP, grouping="expert"
    )
    auto = gather_vqmm_verify_rows(x, layer.codes, layer.scales, layer.codebook, rhs, **kwargs)
    explicit = gather_vqmm_verify_rows(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        rhs,
        rows_per_threadgroup=m1_rows_per_threadgroup(SMALL_IN, SMALL_OUT),
        **kwargs,
    )
    _assert_byte_exact(auto, explicit, "rows_per_threadgroup='auto'")


def test_measured_negative_result_defaults_stay_pinned() -> None:
    """Guard the two defaults chosen by measurement, so a silent flip fails here.

    ``codeword_unroll`` was measured slower at every setting above 1, and the
    per-route dispatch strategy was measured faster with flat grouping than with
    expert grouping at verify widths. Both are recorded in
    docs/research/wave4-wide-m-vqmm-survey.md; changing either should require
    updating this test and the survey together.
    """

    import inspect

    signature = inspect.signature(gather_vqmm_verify_mrows_kernel)
    assert signature.parameters["codeword_unroll"].default == 1
    assert signature.parameters["rows_per_threadgroup"].default == 32
    assert inspect.signature(gather_vqmm_verify_rows).parameters["grouping"].default == "flat"

    captured: dict[str, object] = {}
    original = vq_switch._verify_rows_routed

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=_seed("pin-defaults"),
    )
    x = mx.zeros((4, SMALL_IN), dtype=mx.float16)
    rhs = mx.zeros((4, SMALL_TOP_K), dtype=mx.int32)
    vq_switch._verify_rows_routed = spy
    try:
        gather_vqmm(
            x,
            layer.codes,
            layer.scales,
            layer.codebook,
            rhs,
            input_dims=SMALL_IN,
            output_dims=SMALL_OUT,
            group_size=SMALL_GROUP,
            code_bits=8,
            route_strategy="per_route_decoded",
        )
    finally:
        vq_switch._verify_rows_routed = original

    assert captured["grouping"] == "flat", "per_route_decoded must use flat grouping"
    assert captured["m_rows"] == 1, "per_route_decoded must not widen M"
    assert captured["rows_per_threadgroup"] == m1_rows_per_threadgroup(SMALL_IN, SMALL_OUT)


# --------------------------------------------------------------------------
# Byte-exact gate: m_rows tile widths, routing patterns, small shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verify_rows", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("m_rows", VERIFY_MROWS_VALUES)
@pytest.mark.parametrize("pattern", ["uniform", "skewed", "duplicate", "single_expert"])
def test_verify_rows_byte_exact_across_tile_widths_and_patterns(
    verify_rows: int, m_rows: int, pattern: str
) -> None:
    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=1000 + m_rows,
    )
    rhs_np = _routes(
        pattern=pattern,
        verify_rows=verify_rows,
        top_k=SMALL_TOP_K,
        experts=SMALL_EXPERTS,
        seed=_seed(verify_rows, m_rows, pattern),
    )
    rng = np.random.default_rng(200 + verify_rows)
    x = mx.array(rng.normal(size=(verify_rows, SMALL_IN)).astype(np.float16))
    rhs = mx.array(rhs_np)
    mx.eval(x, rhs)

    expected = _looped_m1_reference(layer, x, rhs, out_dim=SMALL_OUT)
    actual = gather_vqmm_verify_rows(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        rhs,
        input_dims=SMALL_IN,
        output_dims=SMALL_OUT,
        group_size=SMALL_GROUP,
        grouping="expert",
        m_rows=m_rows,
    )
    _assert_byte_exact(actual, expected, f"M={verify_rows} m_rows={m_rows} {pattern}")


@pytest.mark.parametrize("codeword_unroll", [1, 2, 4, 8])
@pytest.mark.parametrize("m_rows", [1, 4])
def test_verify_rows_byte_exact_across_codeword_unroll(codeword_unroll: int, m_rows: int) -> None:
    """Unrolling the codeword walk must not reassociate the accumulation.

    ``codeword_unroll`` is a measured-slower knob kept for re-testing, but every
    setting must remain bit-identical: unrolling only changes issue order, and
    each accumulator still advances in ascending codeword order.
    """

    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=6161,
    )
    verify_rows, top_k = 4, SMALL_TOP_K
    route_count = verify_rows * top_k
    rng = np.random.default_rng(6162)
    x = mx.array(rng.normal(size=(verify_rows, SMALL_IN)).astype(np.float16))
    rhs = mx.array(rng.integers(0, SMALL_EXPERTS, size=(verify_rows, top_k), dtype=np.int32))
    mx.eval(x, rhs)

    expected = _looped_m1_reference(layer, x, rhs, out_dim=SMALL_OUT).reshape(
        (route_count, SMALL_OUT)
    )
    flat_rhs = rhs.reshape((-1,))
    sorted_order = mx.argsort(flat_rhs)
    sorted_rhs = flat_rhs[sorted_order]
    sorted_lhs = vq_switch._token_route_lhs(route_count, top_k)[sorted_order]
    descriptors = vq_switch._verify_row_tile_descriptors(sorted_rhs, m_rows=m_rows)
    actual = gather_vqmm_verify_mrows_kernel(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        sorted_lhs,
        *descriptors,
        route_count=route_count,
        input_dims=SMALL_IN,
        output_dims=SMALL_OUT,
        group_size=SMALL_GROUP,
        m_rows=m_rows,
        codeword_unroll=codeword_unroll,
    )[vq_switch._inverse_permutation(sorted_order)]
    _assert_byte_exact(actual, expected, f"codeword_unroll={codeword_unroll} m_rows={m_rows}")


@pytest.mark.parametrize("verify_rows", [2, 4, 8])
def test_verify_rows_byte_exact_with_float32_activations(verify_rows: int) -> None:
    """The vectorized activation load is float16-only; float32 x must still match."""

    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=555,
    )
    rng = np.random.default_rng(556 + verify_rows)
    x = mx.array(rng.normal(size=(verify_rows, SMALL_IN)).astype(np.float32))
    rhs = mx.array(
        _routes(
            pattern="uniform",
            verify_rows=verify_rows,
            top_k=SMALL_TOP_K,
            experts=SMALL_EXPERTS,
            seed=557,
        )
    )
    mx.eval(x, rhs)

    expected = _looped_m1_reference(layer, x, rhs, out_dim=SMALL_OUT)
    for grouping in ("flat", "expert"):
        actual = gather_vqmm_verify_rows(
            x,
            layer.codes,
            layer.scales,
            layer.codebook,
            rhs,
            input_dims=SMALL_IN,
            output_dims=SMALL_OUT,
            group_size=SMALL_GROUP,
            grouping=grouping,
        )
        _assert_byte_exact(actual, expected, f"float32 x M={verify_rows} {grouping}")


def test_verify_rows_byte_exact_for_per_route_activation_layout() -> None:
    """The down projection feeds per-route activations plus explicit lhs indices."""

    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=880,
    )
    verify_rows, top_k = 4, SMALL_TOP_K
    route_count = verify_rows * top_k
    rng = np.random.default_rng(881)
    route_x = mx.array(rng.normal(size=(route_count, SMALL_IN)).astype(np.float16))
    flat_rhs = mx.array(rng.integers(0, SMALL_EXPERTS, size=(route_count,), dtype=np.int32))
    flat_lhs = mx.arange(route_count, dtype=mx.int32)
    mx.eval(route_x, flat_rhs, flat_lhs)

    # One route per activation row, so the M=1 reference is one call per route.
    expected = mx.concatenate(
        [
            gather_vqmm_m1_kernel_unchecked(
                route_x[i : i + 1],
                layer.codes,
                layer.scales,
                layer.codebook,
                flat_rhs[i : i + 1].reshape((1, 1)),
                output_dims=SMALL_OUT,
                code_bits=8,
                rows_per_threadgroup=32,
                use_threadgroup_codebook=False,
                use_decoded_codebook=True,
            )
            for i in range(route_count)
        ],
        axis=0,
    ).reshape((route_count, SMALL_OUT))

    for grouping in ("flat", "expert"):
        actual = gather_vqmm_verify_rows(
            route_x,
            layer.codes,
            layer.scales,
            layer.codebook,
            flat_rhs,
            lhs_indices=flat_lhs,
            input_dims=SMALL_IN,
            output_dims=SMALL_OUT,
            group_size=SMALL_GROUP,
            grouping=grouping,
        )
        assert actual.shape == (route_count, SMALL_OUT)
        _assert_byte_exact(actual, expected, f"per-route layout grouping={grouping}")


def test_gather_vqmm_per_route_decoded_strategy_matches_looped_m1() -> None:
    """``route_strategy='per_route_decoded'`` is byte-exact through the public contract."""

    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=1313,
    )
    rng = np.random.default_rng(1314)
    x = mx.array(rng.normal(size=(4, SMALL_IN)).astype(np.float16))
    rhs = mx.array(rng.integers(0, SMALL_EXPERTS, size=(4, SMALL_TOP_K), dtype=np.int32))
    mx.eval(x, rhs)

    expected = _looped_m1_reference(layer, x, rhs, out_dim=SMALL_OUT)
    actual = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        rhs,
        input_dims=SMALL_IN,
        output_dims=SMALL_OUT,
        group_size=SMALL_GROUP,
        code_bits=8,
        route_strategy="per_route_decoded",
    )
    assert actual.shape == (4, SMALL_TOP_K, SMALL_OUT)
    _assert_byte_exact(actual, expected, "route_strategy='per_route_decoded'")


# --------------------------------------------------------------------------
# Independent reference: float32 dequantize-then-matmul
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verify_rows", [1, 2, 4])
def test_verify_rows_approximates_float32_dequant_matmul(verify_rows: int) -> None:
    """Catches layout/indexing errors a self-consistent byte comparison cannot.

    Not a byte-exact contract: the kernel's per-codeword ``dot(half4, half4)``
    is a half-precision 4-term dot, so it cannot reproduce a float32
    dequantize-then-matmul bit-for-bit.
    """

    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=2024,
    )
    rhs_np = _routes(
        pattern="uniform",
        verify_rows=verify_rows,
        top_k=SMALL_TOP_K,
        experts=SMALL_EXPERTS,
        seed=2025,
    )
    rng = np.random.default_rng(2026)
    x = mx.array(rng.normal(size=(verify_rows, SMALL_IN)).astype(np.float16))
    rhs = mx.array(rhs_np)
    mx.eval(x, rhs)

    expected = _float32_dequant_reference(layer, x, rhs_np, in_dim=SMALL_IN)
    actual = np.array(
        gather_vqmm_verify_rows(
            x,
            layer.codes,
            layer.scales,
            layer.codebook,
            rhs,
            input_dims=SMALL_IN,
            output_dims=SMALL_OUT,
            group_size=SMALL_GROUP,
            grouping="expert",
        )
    ).astype(np.float32)

    scale = float(np.max(np.abs(expected))) or 1.0
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2 * scale)
    dot = float(np.sum(actual * expected))
    norm = float(np.sqrt(np.sum(actual * actual) * np.sum(expected * expected)))
    assert dot / norm >= 0.9999


# --------------------------------------------------------------------------
# Route-tile descriptors
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m_rows", [1, 2, 4, 8])
@pytest.mark.parametrize("pattern", ["uniform", "skewed", "duplicate", "single_expert"])
def test_verify_row_tile_descriptors_partition_routes_by_expert(m_rows: int, pattern: str) -> None:
    rhs_np = _routes(
        pattern=pattern,
        verify_rows=8,
        top_k=SMALL_TOP_K,
        experts=SMALL_EXPERTS,
        seed=_seed(m_rows, pattern),
    ).reshape(-1)
    flat_rhs = mx.array(rhs_np)
    sorted_rhs = flat_rhs[mx.argsort(flat_rhs)]
    experts, offsets, counts = vq_switch._verify_row_tile_descriptors(sorted_rhs, m_rows=m_rows)
    mx.eval(sorted_rhs, experts, offsets, counts)

    sorted_np = np.array(sorted_rhs)
    experts_np, offsets_np, counts_np = np.array(experts), np.array(offsets), np.array(counts)
    route_count = sorted_np.shape[0]

    assert int(counts_np.sum()) == route_count, "tiles must cover every route exactly once"
    assert counts_np.max() <= m_rows, "no tile may exceed the m_rows width"

    covered = np.zeros(route_count, dtype=np.int32)
    for tile in range(counts_np.shape[0]):
        count = int(counts_np[tile])
        if count == 0:
            continue
        start = int(offsets_np[tile])
        span = sorted_np[start : start + count]
        assert np.all(span == experts_np[tile]), "every route in a tile shares the tile's expert"
        covered[start : start + count] += 1
    assert np.all(covered == 1), "tiles must not overlap or leave gaps"


def test_verify_row_tile_descriptors_reject_prefill_sized_route_counts() -> None:
    """The builder is O(routes^2) on device; it must refuse prefill route counts."""

    limit = vq_switch._VERIFY_MAX_ROUTES
    ok = mx.zeros((limit,), dtype=mx.int32)
    mx.eval(ok)
    experts, offsets, counts = vq_switch._verify_row_tile_descriptors(ok, m_rows=4)
    mx.eval(experts, offsets, counts)
    assert int(np.array(counts).sum()) == limit

    too_many = mx.zeros((limit + 1,), dtype=mx.int32)
    mx.eval(too_many)
    with pytest.raises(ValueError, match=f"at most {limit} routes"):
        vq_switch._verify_row_tile_descriptors(too_many, m_rows=4)


def test_verify_m_rows_caps_tile_width_at_verify_width() -> None:
    assert verify_m_rows(6, 6) == 1
    assert verify_m_rows(12, 6) == 2
    assert verify_m_rows(18, 6) == 3
    assert verify_m_rows(24, 6) == 4
    assert verify_m_rows(30, 6) == 8
    assert verify_m_rows(48, 6) == 8
    # Widths past the largest compiled tile clamp rather than raise.
    assert verify_m_rows(96, 6) == 8


# --------------------------------------------------------------------------
# Contract validation
# --------------------------------------------------------------------------


def test_verify_rows_rejects_unsupported_configurations() -> None:
    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=3131,
    )
    x = mx.zeros((2, SMALL_IN), dtype=mx.float16)
    rhs = mx.zeros((2, SMALL_TOP_K), dtype=mx.int32)

    with pytest.raises(ValueError, match="grouping must be"):
        gather_vqmm_verify_rows(
            x, layer.codes, layer.scales, layer.codebook, rhs,
            input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP,
            grouping="bogus",
        )
    with pytest.raises(ValueError, match="rhs_indices must be 2D"):
        gather_vqmm_verify_rows(
            x, layer.codes, layer.scales, layer.codebook, mx.zeros((4,), dtype=mx.int32),
            input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP,
        )
    with pytest.raises(ValueError, match="must match x row dimension"):
        gather_vqmm_verify_rows(
            x, layer.codes, layer.scales, layer.codebook,
            mx.zeros((5, SMALL_TOP_K), dtype=mx.int32),
            input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP,
        )
    with pytest.raises(ValueError, match="m_rows must be one of"):
        gather_vqmm_verify_mrows_kernel(
            x, layer.codes, layer.scales, layer.codebook,
            mx.zeros((2,), dtype=mx.int32), mx.zeros((2,), dtype=mx.int32),
            mx.zeros((2,), dtype=mx.int32), mx.ones((2,), dtype=mx.int32),
            route_count=2, input_dims=SMALL_IN, output_dims=SMALL_OUT,
            group_size=SMALL_GROUP, m_rows=5,
        )
    with pytest.raises(ValueError, match="requires code_bits=8"):
        gather_vqmm(
            x, layer.codes, layer.scales, layer.codebook, rhs,
            input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP,
            code_bits=16, route_strategy="per_route_decoded",
        )
    with pytest.raises(ValueError, match="route_strategy must be"):
        gather_vqmm(
            x, layer.codes, layer.scales, layer.codebook, rhs,
            input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP,
            code_bits=8, route_strategy="bogus",
        )


def test_verify_rows_default_codebook_matches_explicit() -> None:
    layer = _fixture(
        experts=SMALL_EXPERTS,
        out_dim=SMALL_OUT,
        in_dim=SMALL_IN,
        group_size=SMALL_GROUP,
        seed=4141,
    )
    rng = np.random.default_rng(4142)
    x = mx.array(rng.normal(size=(4, SMALL_IN)).astype(np.float16))
    rhs = mx.array(rng.integers(0, SMALL_EXPERTS, size=(4, SMALL_TOP_K), dtype=np.int32))
    mx.eval(x, rhs)
    assert np.array_equal(np.array(layer.codebook), e8_1bit_packed())

    kwargs = dict(
        input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP, grouping="expert"
    )
    explicit = gather_vqmm_verify_rows(
        x, layer.codes, layer.scales, layer.codebook, rhs, **kwargs
    )
    defaulted = gather_vqmm_verify_rows(x, layer.codes, layer.scales, None, rhs, **kwargs)
    _assert_byte_exact(defaulted, explicit, "default codebook")


# --------------------------------------------------------------------------
# Wave 4 increment 2: the auto-dispatch decision table
#
# Increment 1 measured that route_strategy="auto" sent the MTP verify window to
# gather_vqmm.metal, a scalar kernel 3.2-4.5x slower than the per-route decoded
# kernel. Increment 2 flips that decision. These tests pin the new table so the
# flip cannot silently drift, and quantify its numeric consequence -- which is
# NOT byte-exact against the kernel it replaces.
# --------------------------------------------------------------------------


_DISPATCH_PATHS = {
    # attribute on mlx_vq.ops.vq_switch -> the dispatch decision it witnesses
    "gather_vqmm_m1_kernel": "m1_fast_path",
    "gather_vqmm_kernel": "scalar_token",
    "gather_vqmm_lhs_kernel": "scalar_per_route",
    "_verify_rows_routed": "per_route_decoded",
    "gather_vqmm_tiled_down_kernel": "sorted_tiled",
    "gather_vqmm_mma_cwdecode_blocks_kernel": "sorted_tiled",
    "gather_vqmm_mma_k32_cwdecode_blocks_kernel": "sorted_tiled",
    "gather_vqmm_mma_k64_cwdecode_blocks_kernel": "sorted_tiled",
}


def _observe_dispatch(monkeypatch, call, *, out_dims: int) -> str:
    """Return the name of the dispatch path ``call()`` takes.

    Every kernel entry point ``gather_vqmm`` can reach is replaced with a
    recorder, so the assertion lands on the branch actually taken rather than on
    output values -- which several branches share by construction. Output shapes
    are reconstructed loosely because nothing downstream inspects them beyond
    the final reshape.
    """

    seen: list[str] = []

    def recorder(name: str):
        def fake(*args, **kwargs):
            seen.append(name)
            if name == "per_route_decoded":
                return mx.zeros((args[4].shape[0], out_dims), dtype=mx.float16)
            route_count = kwargs.get("route_count")
            if route_count is None:
                rhs = args[4]
                route_count = rhs.shape[0] if rhs.ndim == 1 else rhs.shape[0] * rhs.shape[1]
            return mx.zeros((route_count, out_dims), dtype=mx.float16)

        return fake

    for attr, path in _DISPATCH_PATHS.items():
        monkeypatch.setattr(f"mlx_vq.ops.vq_switch.{attr}", recorder(path))
    result = call()
    if isinstance(result, mx.array):
        mx.eval(result)
    assert len(seen) == 1, f"expected exactly one dispatch, observed {seen}"
    return seen[0]


def _dispatch_fixture(seed: int = 909) -> QuantizedVQSwitchLinear:
    return _fixture(
        experts=V4_EXPERTS, out_dim=SMALL_OUT, in_dim=SMALL_IN, group_size=SMALL_GROUP, seed=seed
    )


@pytest.mark.parametrize(
    "tokens,top_k,expected",
    [
        # top-6, the DeepSeek-V4-Flash routing width. route_count = 6 * tokens.
        (1, 6, "m1_fast_path"),        # 6 routes   -- unchanged, already decoded
        (2, 6, "per_route_decoded"),   # 12 routes  -- lower edge of the flip
        (21, 6, "per_route_decoded"),  # 126 routes -- upper edge of the flip
        (22, 6, "sorted_tiled"),       # 132 routes -- unchanged, prefill rule wins
        # top-2, to land exactly on the 126/128 route-count boundary. The window
        # is defined by route count, not token count: 2..21 is a top-6
        # consequence, and nothing in the dispatch hardcodes 21.
        (63, 2, "per_route_decoded"),  # 126 routes
        (64, 2, "sorted_tiled"),       # 128 routes
    ],
)
def test_auto_dispatch_table_pins_the_verify_window(monkeypatch, tokens, top_k, expected) -> None:
    """``auto`` -> path across the token-layout boundaries of the flip.

    Both edges matter, for different reasons. ``tokens=1`` must stay on the M=1
    fast path, which is already a decoded kernel and measured best at M=1.
    ``route_count >= 128`` must stay on ``sorted_tiled``, which serves prefill
    and is untouched by this increment.
    """

    layer = _dispatch_fixture()
    rng = np.random.default_rng(_seed("auto-table", tokens, top_k))
    x = mx.array(rng.normal(size=(tokens, SMALL_IN)).astype(np.float16))
    rhs = mx.array(rng.integers(0, V4_EXPERTS, size=(tokens, top_k), dtype=np.int32))
    mx.eval(x, rhs)

    path = _observe_dispatch(
        monkeypatch,
        lambda: gather_vqmm(
            x, layer.codes, layer.scales, layer.codebook, rhs,
            input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP,
            code_bits=8, route_strategy="auto",
        ),
        out_dims=SMALL_OUT,
    )
    assert path == expected


def test_auto_dispatch_flips_the_per_route_activation_layout(monkeypatch) -> None:
    """The verify down projection uses the per-route layout, and it flips too.

    ``gather_vqmm_lhs.metal`` has the same one-threadgroup-per-output-element
    structure as ``gather_vqmm.metal``, so it is the same slow path under a
    different activation indexing scheme. There is no M=1 fast path to preserve
    inside this op for that layout -- ``QuantizedVQSwitchLinear`` handles the
    single-route case before ``gather_vqmm`` is reached -- so every eligible
    per-route call moves.
    """

    layer = _dispatch_fixture(seed=911)
    rng = np.random.default_rng(_seed("auto-per-route"))
    routes = 12
    x = mx.array(rng.normal(size=(routes, SMALL_IN)).astype(np.float16))
    flat_rhs = mx.array(rng.integers(0, V4_EXPERTS, size=(routes,), dtype=np.int32))
    flat_lhs = mx.arange(routes, dtype=mx.int32)
    mx.eval(x, flat_rhs, flat_lhs)

    def call(strategy: str):
        return lambda: gather_vqmm(
            x, layer.codes, layer.scales, layer.codebook, flat_rhs,
            lhs_indices=flat_lhs, input_dims=SMALL_IN, output_dims=SMALL_OUT,
            group_size=SMALL_GROUP, code_bits=8, route_strategy=strategy,
        )

    assert _observe_dispatch(monkeypatch, call("auto"), out_dims=SMALL_OUT) == "per_route_decoded"
    # "direct" still reaches the scalar per-route kernel, so the path this flip
    # steps off keeps its coverage and stays available to callers that want it.
    assert _observe_dispatch(monkeypatch, call("direct"), out_dims=SMALL_OUT) == "scalar_per_route"


def test_auto_dispatch_leaves_ineligible_calls_on_their_existing_paths(monkeypatch) -> None:
    """Everything ``auto`` used to do that this increment does not claim.

    ``sorted_indices`` still short-circuits to ``sorted_tiled`` regardless of
    route count, and float32 activations stay on the scalar float32 kernel --
    the decoded kernel would narrow them to half before multiplying, which is a
    precision regression rather than a reassociation.
    """

    layer = _dispatch_fixture(seed=913)
    rng = np.random.default_rng(_seed("auto-ineligible"))
    rhs = mx.array(rng.integers(0, V4_EXPERTS, size=(4, 6), dtype=np.int32))
    kwargs = dict(
        input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP,
        code_bits=8, route_strategy="auto",
    )

    x16 = mx.array(rng.normal(size=(4, SMALL_IN)).astype(np.float16))
    x32 = mx.array(rng.normal(size=(4, SMALL_IN)).astype(np.float32))
    mx.eval(rhs, x16, x32)

    assert _observe_dispatch(
        monkeypatch,
        lambda: gather_vqmm(
            x16, layer.codes, layer.scales, layer.codebook, rhs, sorted_indices=True, **kwargs
        ),
        out_dims=SMALL_OUT,
    ) == "sorted_tiled"
    assert _observe_dispatch(
        monkeypatch,
        lambda: gather_vqmm(x32, layer.codes, layer.scales, layer.codebook, rhs, **kwargs),
        out_dims=SMALL_OUT,
    ) == "scalar_token"


def test_auto_selects_per_route_decoded_guard_matrix() -> None:
    """The predicate's guards, unit-tested away from fixture-validity concerns.

    Each ``False`` here is a config the decoded verify kernel would either
    reject outright or serve at reduced precision, so ``auto`` must decline it
    and leave the call on the kernel that already handles it. Asserted against
    the predicate directly because several of these configs cannot be built as a
    valid quantized layer at all -- an 8D-unaligned group size, for instance, is
    reachable as a predicate input but not as a working ``gather_vqmm`` call.
    """

    base = dict(
        tokens=4, has_lhs_indices=False, implementation="metal", code_bits=8,
        input_dims=4096, output_dims=2048, group_size=512, activation_dtype=mx.float16,
    )
    assert vq_switch.auto_selects_per_route_decoded(**base)

    # The M=1 token-layout fast path is preserved; the per-route layout has no
    # such fast path inside this op, so a single row there still flips.
    assert not vq_switch.auto_selects_per_route_decoded(**{**base, "tokens": 1})
    assert vq_switch.auto_selects_per_route_decoded(
        **{**base, "tokens": 1, "has_lhs_indices": True}
    )

    assert not vq_switch.auto_selects_per_route_decoded(**{**base, "code_bits": 16})
    assert not vq_switch.auto_selects_per_route_decoded(**{**base, "implementation": "reference"})
    assert not vq_switch.auto_selects_per_route_decoded(**{**base, "activation_dtype": mx.float32})
    assert not vq_switch.auto_selects_per_route_decoded(**{**base, "activation_dtype": mx.bfloat16})
    # group_size must be 8D-aligned and must divide input_dims; the verify
    # kernel raises on both, while the scalar kernel accepts them.
    assert not vq_switch.auto_selects_per_route_decoded(**{**base, "group_size": 12})
    assert not vq_switch.auto_selects_per_route_decoded(**{**base, "group_size": 3000})
    assert not vq_switch.auto_selects_per_route_decoded(**{**base, "input_dims": 4100})

    # The row packing the M=1 helper hands out must be one the verify kernel
    # accepts, or the flip would convert a working call into an exception.
    assert m1_rows_per_threadgroup(4096, 2048) in VERIFY_ROWS_PER_THREADGROUP_VALUES


def test_auto_dispatch_in_the_window_equals_explicit_per_route_decoded() -> None:
    """``auto`` in the window is byte-identical to asking for the strategy."""

    layer = _dispatch_fixture(seed=917)
    rng = np.random.default_rng(_seed("auto-equals-explicit"))
    x = mx.array(rng.normal(size=(4, SMALL_IN)).astype(np.float16))
    rhs = mx.array(rng.integers(0, V4_EXPERTS, size=(4, 6), dtype=np.int32))
    mx.eval(x, rhs)

    kwargs = dict(
        input_dims=SMALL_IN, output_dims=SMALL_OUT, group_size=SMALL_GROUP, code_bits=8
    )
    auto = gather_vqmm(
        x, layer.codes, layer.scales, layer.codebook, rhs, route_strategy="auto", **kwargs
    )
    explicit = gather_vqmm(
        x, layer.codes, layer.scales, layer.codebook, rhs,
        route_strategy="per_route_decoded", **kwargs
    )
    _assert_byte_exact(auto, explicit, "auto vs explicit per_route_decoded")


# --------------------------------------------------------------------------
# Wave 4 increment 2: what the dispatch flip does to the numbers
# --------------------------------------------------------------------------

# Tolerances below are gates on a *measured* delta, not aspirations. Observed
# maxima on this machine (Apple M5 Max, MLX 0.31.2) over the three V4 shapes at
# M in {2,4,8,16,21}, float16 activations, uniform routing:
#
#   max abs delta            0.0078125  = 2 float16 spacings at the peak output
#   relative RMS delta       3.7e-4
#   err-vs-float32 ratio     1.61x  (decoded 3.35e-4 vs scalar 2.09e-4, rel RMS)
#
# Each gate carries roughly 2-3x headroom over those, which is enough to absorb
# routing-pattern variation without being loose enough to hide a real change of
# kernel behavior.
_FLIP_MAX_SPACINGS = 4.0
_FLIP_MAX_RELATIVE_RMS = 1e-3
_FLIP_MAX_ERROR_RATIO = 3.0


def _flip_metrics(scalar: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    delta = np.abs(scalar.astype(np.float64) - decoded.astype(np.float64))
    magnitude = float(np.abs(scalar).max())
    # float16 spacing at the output's peak magnitude: the natural unit for "how
    # far apart are two float16 results", and shape-independent.
    spacing = float(np.spacing(np.float16(magnitude)))
    reference_rms = float(np.sqrt((scalar.astype(np.float64) ** 2).mean()))
    return {
        "max_abs": float(delta.max()),
        "spacings": float(delta.max()) / spacing,
        "relative_rms": float(np.sqrt((delta**2).mean())) / reference_rms,
        "mismatch_frac": float(np.count_nonzero(scalar != decoded)) / scalar.size,
    }


@pytest.mark.parametrize("proj,in_dim,out_dim", V4_SHAPES, ids=[s[0] for s in V4_SHAPES])
@pytest.mark.parametrize("verify_rows", [2, 8, 21])
def test_dispatch_flip_numeric_delta_stays_within_documented_tolerance(
    proj: str, in_dim: int, out_dim: int, verify_rows: int
) -> None:
    """The dispatch flip changes kernels, so it changes numbers. This bounds it.

    **This is not a byte-exactness gate, and it cannot be.** Increment 2 moves
    the 2..21-token verify window from ``gather_vqmm.metal`` to the per-route
    decoded kernel, and the two evaluate an 8D codeword differently by
    construction:

    * the scalar kernel widens each half weight and activation to float32 and
      does eight float32 fused multiply-adds per codeword;
    * the decoded kernel evaluates the codeword as two ``dot(half4, half4)``
      products -- a half-precision 4-term dot -- and accumulates the scaled
      results in float32.

    Neither result is a rounding of the other. Each is exact with respect to its
    own accumulation order, and neither is the "true" answer: a float32
    dequantize-then-matmul is a third value again. So the honest contract is a
    bounded, measured, documented delta rather than equality, and the bound is
    expressed in float16 spacings at the output's own magnitude so it does not
    silently mean something different on a shape with a different output scale.

    The flip is taken anyway for two reasons, and the second is why the delta is
    acceptable rather than merely small: it is 3.2-4.5x faster, and it moves
    this window onto the same kernel family the M=1 decode path already uses.
    MTP verify exists to confirm what decode would have emitted, so agreeing
    with decode is worth more here than agreeing with the scalar kernel --
    which, note, no decode step has ever used at any width.
    See ``test_dispatch_flip_agrees_with_the_shipped_m1_decode_path``.
    """

    layer = _fixture(
        experts=V4_EXPERTS, out_dim=out_dim, in_dim=in_dim, group_size=V4_GROUP_SIZE,
        seed=_seed("flip-delta", proj, verify_rows),
    )
    rhs_np = _routes(
        pattern="uniform", verify_rows=verify_rows, top_k=V4_TOP_K,
        experts=V4_EXPERTS, seed=_seed("flip-delta-routes", proj, verify_rows),
    )
    rng = np.random.default_rng(_seed("flip-delta-x", proj, verify_rows))
    x = mx.array(rng.normal(size=(verify_rows, in_dim)).astype(np.float16))
    rhs = mx.array(rhs_np)
    mx.eval(x, rhs)

    kwargs = dict(
        input_dims=in_dim, output_dims=out_dim, group_size=V4_GROUP_SIZE, code_bits=8
    )
    # "direct" is exactly what "auto" resolved to for this window before the
    # flip, so it is a faithful stand-in for the old dispatch decision.
    old = gather_vqmm(
        x, layer.codes, layer.scales, layer.codebook, rhs, route_strategy="direct", **kwargs
    )
    new = gather_vqmm(
        x, layer.codes, layer.scales, layer.codebook, rhs, route_strategy="auto", **kwargs
    )
    mx.eval(old, new)
    old_np, new_np = np.array(old), np.array(new)

    metrics = _flip_metrics(old_np, new_np)
    context = f"{proj} M={verify_rows}"
    assert metrics["spacings"] <= _FLIP_MAX_SPACINGS, (
        f"{context}: dispatch flip moved an output by {metrics['spacings']:.2f} float16 "
        f"spacings (max abs {metrics['max_abs']:.3e}), over the documented "
        f"{_FLIP_MAX_SPACINGS} -- the kernels have diverged further than a "
        "difference in accumulation order explains"
    )
    assert metrics["relative_rms"] <= _FLIP_MAX_RELATIVE_RMS, (
        f"{context}: relative RMS delta {metrics['relative_rms']:.3e} exceeds "
        f"{_FLIP_MAX_RELATIVE_RMS}"
    )


@pytest.mark.parametrize("proj,in_dim,out_dim", V4_SHAPES, ids=[s[0] for s in V4_SHAPES])
def test_dispatch_flip_does_not_regress_accuracy_against_float32_matmul(
    proj: str, in_dim: int, out_dim: int
) -> None:
    """Neither kernel is exact, so check the flip does not move much further off.

    The float32 dequantize-then-matmul reference is the only third party here
    that is not one of the two kernels under comparison. The decoded kernel is
    expected to be somewhat *less* accurate against it than the scalar kernel --
    a half-precision 4-term dot carries less mantissa than four float32 FMAs --
    and this gate pins how much less: measured 1.61x on relative RMS error,
    gated at ``_FLIP_MAX_ERROR_RATIO``. A regression well past that would mean
    the flip degraded the result rather than merely reassociating it.
    """

    verify_rows = 2
    layer = _fixture(
        experts=V4_EXPERTS, out_dim=out_dim, in_dim=in_dim, group_size=V4_GROUP_SIZE,
        seed=_seed("flip-accuracy", proj),
    )
    rhs_np = _routes(
        pattern="uniform", verify_rows=verify_rows, top_k=V4_TOP_K,
        experts=V4_EXPERTS, seed=_seed("flip-accuracy-routes", proj),
    )
    rng = np.random.default_rng(_seed("flip-accuracy-x", proj))
    x = mx.array(rng.normal(size=(verify_rows, in_dim)).astype(np.float16))
    rhs = mx.array(rhs_np)
    mx.eval(x, rhs)

    kwargs = dict(
        input_dims=in_dim, output_dims=out_dim, group_size=V4_GROUP_SIZE, code_bits=8
    )
    old = np.array(
        gather_vqmm(
            x, layer.codes, layer.scales, layer.codebook, rhs, route_strategy="direct", **kwargs
        )
    ).astype(np.float64)
    new = np.array(
        gather_vqmm(
            x, layer.codes, layer.scales, layer.codebook, rhs, route_strategy="auto", **kwargs
        )
    ).astype(np.float64)
    reference = _float32_dequant_reference(layer, x, rhs_np, in_dim=in_dim).astype(np.float64)

    old_err = float(np.sqrt(((old - reference) ** 2).mean()))
    new_err = float(np.sqrt(((new - reference) ** 2).mean()))
    ratio = new_err / old_err
    assert ratio <= _FLIP_MAX_ERROR_RATIO, (
        f"{proj}: post-flip RMS error vs float32 matmul is {ratio:.2f}x the pre-flip "
        f"error ({new_err:.3e} vs {old_err:.3e}), over the documented "
        f"{_FLIP_MAX_ERROR_RATIO}x"
    )


@pytest.mark.parametrize("proj,in_dim,out_dim", V4_SHAPES, ids=[s[0] for s in V4_SHAPES])
@pytest.mark.parametrize("verify_rows", [2, 8, 21])
def test_dispatch_flip_agrees_with_the_shipped_m1_decode_path(
    proj: str, in_dim: int, out_dim: int, verify_rows: int
) -> None:
    """The property that justifies the flip: agreement with M=1 decode.

    ``gather_vqmm(route_strategy="auto")`` at one token is the kernel a decode
    step actually runs, and this increment leaves it untouched. Looping it per
    verify row therefore gives the values decode would emit for the same
    (token, expert) pairs -- which is exactly what an MTP verify pass is trying
    to reproduce.

    Two outcomes, both asserted, because the M=1 fast path is not one kernel:
    ``gather_vqmm_m1_kernel`` dispatches a **rowpair** variant when
    ``input_dims > output_dims`` and the plain decoded kernel otherwise.

    * ``down`` (2048 -> 4096) takes the plain decoded kernel, so the flipped
      window is **byte-identical** to looped M=1 decode. Increment 1's parity
      contract lands exactly on the shipped path here.
    * ``gate``/``up`` (4096 -> 2048) take the rowpair variant, so a small
      residual remains. It is bounded here rather than eliminated: the residual
      is the rowpair-vs-plain-decoded difference itself, which is present at M=1
      too (measured 0.23% of elements, max 1 float16 spacing) and is not
      something a dispatch change can close. Closing it needs a wide-M rowpair
      kernel, recorded as a Wave 5 item in the increment-2 report.

    For contrast, the pre-flip scalar kernel disagreed with M=1 decode on ~48%
    of elements by up to ~2 spacings, so the flip is a large net improvement in
    decode agreement on every shape.
    """

    layer = _fixture(
        experts=V4_EXPERTS, out_dim=out_dim, in_dim=in_dim, group_size=V4_GROUP_SIZE,
        seed=_seed("flip-decode-agreement", proj, verify_rows),
    )
    rhs_np = _routes(
        pattern="uniform", verify_rows=verify_rows, top_k=V4_TOP_K,
        experts=V4_EXPERTS, seed=_seed("flip-decode-routes", proj, verify_rows),
    )
    rng = np.random.default_rng(_seed("flip-decode-x", proj, verify_rows))
    x = mx.array(rng.normal(size=(verify_rows, in_dim)).astype(np.float16))
    rhs = mx.array(rhs_np)
    mx.eval(x, rhs)

    kwargs = dict(
        input_dims=in_dim, output_dims=out_dim, group_size=V4_GROUP_SIZE, code_bits=8
    )
    flipped = gather_vqmm(
        x, layer.codes, layer.scales, layer.codebook, rhs, route_strategy="auto", **kwargs
    )
    decode_loop = mx.concatenate(
        [
            gather_vqmm(
                x[i:i + 1], layer.codes, layer.scales, layer.codebook, rhs[i:i + 1],
                route_strategy="auto", **kwargs,
            )
            for i in range(verify_rows)
        ],
        axis=0,
    )
    mx.eval(flipped, decode_loop)

    scalar_baseline = gather_vqmm(
        x, layer.codes, layer.scales, layer.codebook, rhs, route_strategy="direct", **kwargs
    )
    mx.eval(scalar_baseline)

    flipped_np = np.array(flipped)
    decode_np = np.array(decode_loop)
    scalar_np = np.array(scalar_baseline)
    assert flipped_np.shape == decode_np.shape

    uses_rowpair = in_dim > out_dim
    flip_agreement = _flip_metrics(decode_np, flipped_np)
    scalar_agreement = _flip_metrics(decode_np, scalar_np)

    if not uses_rowpair:
        _assert_byte_exact(
            flipped, decode_loop,
            f"{proj} M={verify_rows}: flipped dispatch vs looped M=1 decode",
        )
    else:
        assert flip_agreement["spacings"] <= _FLIP_MAX_SPACINGS, (
            f"{proj} M={verify_rows}: flipped dispatch is {flip_agreement['spacings']:.2f} "
            "float16 spacings from looped M=1 decode; the rowpair residual should be "
            "within one or two"
        )
        assert flip_agreement["mismatch_frac"] <= 0.05, (
            f"{proj} M={verify_rows}: {flip_agreement['mismatch_frac']:.3%} of elements "
            "disagree with looped M=1 decode, far above the measured rowpair residual"
        )

    # The point of the flip, asserted rather than asserted-in-prose: the new path
    # agrees with decode on strictly more elements than the path it replaced.
    assert flip_agreement["mismatch_frac"] < scalar_agreement["mismatch_frac"], (
        f"{proj} M={verify_rows}: flipped dispatch agrees with M=1 decode on "
        f"{1 - flip_agreement['mismatch_frac']:.3%} of elements versus the scalar "
        f"kernel's {1 - scalar_agreement['mismatch_frac']:.3%} -- the flip's central "
        "justification does not hold on this shape"
    )
