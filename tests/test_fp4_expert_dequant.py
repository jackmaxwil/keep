"""Bit-exact FP4 (e2m1) routed-expert decode and group-32 E8M0 dequantization.

The format facts asserted here are pinned against three independent sources
(see ``src/keep/convert/fp4_expert.py`` for the full citation block):

* ``~/models/DeepSeek-V4-Flash-0731/inference/convert.py:11-14`` (value table)
  and ``:30-33`` (nibble order),
* MLX ``mode="mxfp4"`` dequantize (``test_matches_mlx_mxfp4_dequantize``),
* ``.repos/omlx`` passing the raw checkpoint bytes into that MLX path
  unchanged (``omlx/patches/deepseek_v4/deepseek_v4_model.py:2219-2226``).
"""

from __future__ import annotations

import json
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from keep.convert.fp4_expert import decode_fp4_e2m1, dequantize_fp4_expert
from keep.convert.fp8_block import decode_ue8m0

# The e2m1 value for each 4-bit code, transcribed from DeepSeek's own
# FP4_TABLE (inference/convert.py:11-14). Code 0x8 is written as a bare 0.0
# there; we decode it as -0.0 (OCP e2m1 / MLX), which is numerically identical
# and is pinned separately by test_code_eight_is_negative_zero.
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# Bytes whose (low, high) nibbles spell codes 0,1 / 2,3 / ... / 14,15 in
# logical order, i.e. low nibble first.
CODES_0_TO_15 = np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
                         dtype=np.uint8)

CHECKPOINT = Path.home() / "models" / "DeepSeek-V4-Flash-0731"


# --------------------------------------------------------------------------
# value table
# --------------------------------------------------------------------------


def test_value_table_covers_all_sixteen_codes():
    values = decode_fp4_e2m1(CODES_0_TO_15)
    assert values.shape == (16,)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, np.array(E2M1_VALUES, dtype=np.float32))


def test_value_table_magnitudes_are_sign_symmetric():
    # e2m1 is sign-magnitude: code c and code c|0x8 differ only in sign.
    values = decode_fp4_e2m1(CODES_0_TO_15)
    np.testing.assert_array_equal(values[:8], -values[8:])


def test_value_table_has_no_nan_or_inf():
    # e2m1fn is "finite": unlike e4m3fn there is no NaN encoding and no
    # infinities, so every one of the 16 codes is a real number.
    assert np.isfinite(decode_fp4_e2m1(np.arange(256, dtype=np.uint8))).all()


def test_code_eight_is_negative_zero():
    # 0x8 is the e2m1 negative-zero encoding (sign=1, exp=0, mantissa=0).
    # DeepSeek's FP4_TABLE writes it as +0.0; MLX's mxfp4 kernel and the OCP
    # spec both produce -0.0. We follow OCP/MLX so the sign bit survives the
    # decode, matching fp8_block's signed-zero fidelity. == cannot see this.
    values = decode_fp4_e2m1(np.array([0x80], dtype=np.uint8))
    assert values[0] == 0.0 and not np.signbit(values[0]), "low nibble 0x0 is +0.0"
    assert values[1] == 0.0 and np.signbit(values[1]), "high nibble 0x8 is -0.0"


# --------------------------------------------------------------------------
# nibble order
# --------------------------------------------------------------------------


def test_nibble_order_is_low_nibble_first():
    # 0x71: low=0x1 -> 0.5, high=0x7 -> 6.0. If the order were reversed this
    # would decode to [6.0, 0.5]. Deliberately asymmetric in both magnitude
    # and position so no symmetry can mask a swap.
    np.testing.assert_array_equal(
        decode_fp4_e2m1(np.array([0x71], dtype=np.uint8)),
        np.array([0.5, 6.0], dtype=np.float32),
    )


def test_nibble_order_across_a_row_is_strictly_increasing():
    # A whole row of strictly increasing magnitudes: any nibble swap turns
    # this into a sawtooth, and any byte-order confusion reverses it.
    values = decode_fp4_e2m1(np.array([0x21, 0x43, 0x65, 0x07], dtype=np.uint8))
    np.testing.assert_array_equal(
        values, np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0], dtype=np.float32)
    )
    assert (np.diff(values[:-1]) > 0).all()


def test_nibble_order_matches_deepseek_reference_expression():
    # Re-derives DeepSeek convert.py:31-33 in numpy on random bytes:
    #   low = x & 0x0F; high = (x >> 4) & 0x0F
    #   stack([TABLE[low], TABLE[high]], dim=-1).flatten()
    rng = np.random.default_rng(20260811)
    codes = rng.integers(0, 256, size=(7, 32), dtype=np.uint8)
    table = np.array(E2M1_VALUES, dtype=np.float32)
    reference = np.stack(
        [table[codes & 0x0F], table[(codes >> 4) & 0x0F]], axis=-1
    ).reshape(7, 64)
    np.testing.assert_array_equal(decode_fp4_e2m1(codes), reference)


def test_decode_doubles_only_the_last_axis():
    codes = np.zeros((3, 5, 8), dtype=np.uint8)
    assert decode_fp4_e2m1(codes).shape == (3, 5, 16)
    assert decode_fp4_e2m1(np.zeros(0, dtype=np.uint8)).shape == (0,)


# --------------------------------------------------------------------------
# decode dtype handling
# --------------------------------------------------------------------------


def test_decode_reinterprets_int8_rather_than_converting():
    # safetensors declares the packed expert bytes as I8, which numpy loads as
    # int8; byte 0xFE arrives as -2. Reinterpretation must give codes
    # (0xE, 0xF) = (-4.0, -6.0), not anything derived from the value -2.
    signed = np.array([-2], dtype=np.int8)
    assert signed.view(np.uint8)[0] == 0xFE
    np.testing.assert_array_equal(
        decode_fp4_e2m1(signed), np.array([-4.0, -6.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        decode_fp4_e2m1(signed), decode_fp4_e2m1(np.array([0xFE], dtype=np.uint8))
    )


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.int32, np.uint16, np.bool_])
def test_decode_rejects_other_dtypes(dtype):
    with pytest.raises(ValueError, match=np.dtype(dtype).name):
        decode_fp4_e2m1(np.zeros(4, dtype=dtype))


def test_decode_rejects_float8_dtypes():
    # ml_dtypes float8 arrays must be .view(np.uint8)'d by the caller; numpy
    # would otherwise *convert* them and decode the wrong bit patterns.
    ml_dtypes = pytest.importorskip("ml_dtypes")
    with pytest.raises(ValueError):
        decode_fp4_e2m1(np.zeros(4, dtype=ml_dtypes.float8_e4m3fn))


def test_decode_rejects_non_ndarray():
    with pytest.raises(ValueError):
        decode_fp4_e2m1([0x71])  # type: ignore[arg-type]


def test_decode_does_not_mutate_input():
    codes = np.array([0x71, 0x8F], dtype=np.uint8)
    before = codes.copy()
    decode_fp4_e2m1(codes)
    np.testing.assert_array_equal(codes, before)


def test_decode_accepts_read_only_input():
    # np.frombuffer of a mmapped shard is read-only.
    codes = np.frombuffer(bytes([0x71]), dtype=np.uint8)
    assert not codes.flags.writeable
    np.testing.assert_array_equal(
        decode_fp4_e2m1(codes), np.array([0.5, 6.0], dtype=np.float32)
    )


# --------------------------------------------------------------------------
# scale application
# --------------------------------------------------------------------------


def _all_ones_codes(rows: int, logical_cols: int) -> np.ndarray:
    """[rows, logical_cols/2] packed bytes whose every fp4 value is 1.0."""
    return np.full((rows, logical_cols // 2), 0x22, dtype=np.uint8)


def test_scale_is_group32_along_the_logical_input_dim():
    # 2 rows x 64 logical columns = 2 groups of 32; every code decodes to 1.0
    # so the output must equal the group's power-of-two scale exactly.
    codes = _all_ones_codes(2, 64)
    scales = np.array([[127, 128], [126, 130]], dtype=np.uint8)
    weight = dequantize_fp4_expert(codes, scales)
    assert weight.shape == (2, 64)
    assert weight.dtype == np.float32
    assert (weight[0, :32] == 1.0).all() and (weight[0, 32:] == 2.0).all()
    assert (weight[1, :32] == 0.5).all() and (weight[1, 32:] == 8.0).all()


def test_scale_boundary_lands_between_logical_elements_31_and_32():
    # Pins the group boundary to the *logical* dim, not the packed byte dim.
    # If group_size were applied to the 32 stored bytes the switch would fall
    # at logical element 64 instead.
    codes = _all_ones_codes(1, 128)
    scales = np.array([[127, 129, 127, 127]], dtype=np.uint8)
    weight = dequantize_fp4_expert(codes, scales)
    assert weight[0, 31] == 1.0
    assert weight[0, 32] == 4.0
    assert weight[0, 63] == 4.0
    assert weight[0, 64] == 1.0


def test_scale_multiplies_the_decoded_value_not_the_code():
    codes = np.array([[0x71, 0xF9] * 8], dtype=np.uint8)  # 32 logical values
    scales = np.array([[128]], dtype=np.uint8)  # 2.0
    weight = dequantize_fp4_expert(codes, scales)
    expected = decode_fp4_e2m1(codes[0]) * 2.0
    np.testing.assert_array_equal(weight[0], expected)


def test_scale_preserves_negative_zero_sign():
    codes = np.full((1, 16), 0x88, dtype=np.uint8)  # every value -0.0
    weight = dequantize_fp4_expert(codes, np.array([[130]], dtype=np.uint8))
    assert (weight == 0.0).all()
    assert np.signbit(weight).all()


def test_ue8m0_scale_decode_is_shared_with_fp8_block():
    # Same bias-127 exponent encoding and the same OCP NaN choice for 0xFF,
    # so the two decoders must not drift apart.
    codes = _all_ones_codes(1, 96)
    scales = np.array([[0, 127, 254]], dtype=np.uint8)
    weight = dequantize_fp4_expert(codes, scales)
    expected = decode_ue8m0(scales[0])
    np.testing.assert_array_equal(weight[0, ::32], expected)


def test_scale_byte_255_is_nan_without_warning():
    # fp8_block decodes E8M0 0xFF as NaN (OCP) rather than 2**128; the FP4
    # path must agree, and must not emit an overflow RuntimeWarning.
    codes = _all_ones_codes(1, 64)
    scales = np.array([[255, 127]], dtype=np.uint8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        weight = dequantize_fp4_expert(codes, scales)
    assert caught == []
    assert np.isnan(weight[0, :32]).all()
    assert (weight[0, 32:] == 1.0).all()


def test_accepts_float_scales_as_literal_multipliers():
    codes = _all_ones_codes(2, 64)
    scales = np.array([[3.0, 0.25], [1.0, -2.0]], dtype=np.float32)
    weight = dequantize_fp4_expert(codes, scales)
    assert (weight[0, :32] == 3.0).all() and (weight[0, 32:] == 0.25).all()
    assert (weight[1, 32:] == -2.0).all()


def test_accepts_float64_scales_but_returns_float32():
    codes = _all_ones_codes(1, 32)
    weight = dequantize_fp4_expert(codes, np.array([[0.5]], dtype=np.float64))
    assert weight.dtype == np.float32
    assert (weight == 0.5).all()


def test_accepts_int8_codes_like_the_checkpoint_declares():
    packed = np.full((1, 16), 0x22, dtype=np.uint8).view(np.int8)
    assert packed.dtype == np.int8
    weight = dequantize_fp4_expert(packed, np.array([[128]], dtype=np.uint8))
    assert (weight == 2.0).all()


def test_does_not_mutate_inputs():
    codes = np.full((2, 16), 0x35, dtype=np.uint8)
    scales = np.array([[128], [129]], dtype=np.uint8)
    codes_before, scales_before = codes.copy(), scales.copy()
    dequantize_fp4_expert(codes, scales)
    np.testing.assert_array_equal(codes, codes_before)
    np.testing.assert_array_equal(scales, scales_before)


def test_accepts_read_only_shard_views():
    codes = np.frombuffer(bytes([0x22]) * 16, dtype=np.uint8).reshape(1, 16)
    scales = np.frombuffer(bytes([128]), dtype=np.uint8).reshape(1, 1)
    assert not codes.flags.writeable and not scales.flags.writeable
    assert (dequantize_fp4_expert(codes, scales) == 2.0).all()


# --------------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------------


def test_rejects_ragged_group_when_logical_k_is_not_a_multiple_of_group_size():
    # 17 packed bytes -> logical K=34, which straddles a 32-wide group. A
    # partial trailing group has no well-defined scale, so reject rather than
    # silently ceil-divide.
    codes = np.zeros((1, 17), dtype=np.uint8)
    scales = np.zeros((1, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="34"):
        dequantize_fp4_expert(codes, scales)


def test_rejects_odd_group_size_that_would_split_a_byte():
    codes = np.zeros((1, 16), dtype=np.uint8)
    scales = np.zeros((1, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="even"):
        dequantize_fp4_expert(codes, scales, group_size=17)


@pytest.mark.parametrize("group_size", [0, -32, 1.5, True, None])
def test_rejects_invalid_group_size(group_size):
    codes = np.zeros((1, 16), dtype=np.uint8)
    scales = np.zeros((1, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="group_size"):
        dequantize_fp4_expert(codes, scales, group_size=group_size)


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (2, 2), (1, 2, 1), (2,)])
def test_rejects_wrong_scale_shape(shape):
    codes = np.zeros((1, 32), dtype=np.uint8)  # logical K=64 -> expects (1, 2)
    with pytest.raises(ValueError, match="scales shape"):
        dequantize_fp4_expert(codes, np.zeros(shape, dtype=np.uint8))


def test_rejects_transposed_scale_shape():
    # w1 is [2048, 4096] logical -> scales [2048, 128]; the transpose has the
    # same element count, so only a shape check catches it.
    codes = np.zeros((64, 64), dtype=np.uint8)  # logical [64, 128] -> (64, 4)
    with pytest.raises(ValueError, match="scales shape"):
        dequantize_fp4_expert(codes, np.zeros((4, 64), dtype=np.uint8))


@pytest.mark.parametrize("ndim_shape", [(16,), (1, 2, 16), ()])
def test_rejects_non_2d_codes(ndim_shape):
    with pytest.raises(ValueError, match="2-D"):
        dequantize_fp4_expert(
            np.zeros(ndim_shape, dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)
        )


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint16, np.bool_])
def test_rejects_int_scale_dtypes(dtype):
    # An integer scale array is neither a literal multiplier nor ue8m0; taking
    # either branch would mis-scale by up to 2**127, so name the dtype and
    # refuse.
    codes = np.zeros((1, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match=np.dtype(dtype).name):
        dequantize_fp4_expert(codes, np.ones((1, 1), dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_rejects_non_byte_code_dtypes(dtype):
    with pytest.raises(ValueError, match=np.dtype(dtype).name):
        dequantize_fp4_expert(
            np.zeros((1, 16), dtype=dtype), np.zeros((1, 1), dtype=np.uint8)
        )


def test_rejects_non_ndarray_inputs():
    codes = np.zeros((1, 16), dtype=np.uint8)
    scales = np.zeros((1, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        dequantize_fp4_expert([[0x22] * 16], scales)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dequantize_fp4_expert(codes, [[1.0]])  # type: ignore[arg-type]


def test_rejects_empty_packed_width():
    # Zero logical columns cannot carry a scale group; catch it here rather
    # than producing a (rows, 0) array a caller would silently accept.
    with pytest.raises(ValueError):
        dequantize_fp4_expert(
            np.zeros((4, 0), dtype=np.uint8), np.zeros((4, 0), dtype=np.uint8)
        )


# --------------------------------------------------------------------------
# release shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proj,packed,logical,scale_shape",
    [
        ("w1", (2048, 2048), (2048, 4096), (2048, 128)),
        ("w3", (2048, 2048), (2048, 4096), (2048, 128)),
        ("w2", (4096, 1024), (4096, 2048), (4096, 64)),
    ],
)
def test_release_expert_shapes_round_trip(proj, packed, logical, scale_shape):
    # Shapes measured on DeepSeek-V4-Flash-0731 rev 7872f01b.
    codes = np.full(packed, 0x22, dtype=np.uint8)
    scales = np.full(scale_shape, 127, dtype=np.uint8)
    weight = dequantize_fp4_expert(codes, scales)
    assert weight.shape == logical, proj
    assert weight.dtype == np.float32
    assert (weight == 1.0).all()


# --------------------------------------------------------------------------
# MLX cross-check (opt-in: needs mlx)
# --------------------------------------------------------------------------


@pytest.mark.mlx
def test_matches_mlx_mxfp4_dequantize():
    """MLX ``mode="mxfp4"`` is bit-exactly this format.

    This is the load-bearing cross-check for the nibble order: ``.repos/omlx``
    hands the raw checkpoint bytes to MLX unchanged
    (``omlx/patches/deepseek_v4/deepseek_v4_model.py:2219-2226``: ``scale`` ->
    ``scales``, ``weight.view(mx.uint32)``) and declares
    ``{"group_size": 32, "bits": 4, "mode": "mxfp4"}``
    (``deepseek_v4_model.py:191``), so MLX's unpack order *is* the release's
    unpack order. Agreement here therefore pins low-nibble-first
    independently of DeepSeek's convert.py.
    """
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(7872)
    rows, packed = 8, 64  # logical K = 128 -> 4 groups of 32
    codes = rng.integers(0, 256, size=(rows, packed), dtype=np.uint8)
    # Exclude 0xFF: MLX evaluates it as +inf, we follow OCP/ml_dtypes and
    # fp8_block and return NaN. Real shards contain no 0xFF scale bytes.
    scales = rng.integers(100, 150, size=(rows, packed * 2 // 32), dtype=np.uint8)

    ours = dequantize_fp4_expert(codes, scales)
    theirs = np.array(
        mx.dequantize(
            mx.array(codes).view(mx.uint32),
            mx.array(scales),
            group_size=32,
            bits=4,
            mode="mxfp4",
        ).astype(mx.float32)
    )
    assert theirs.shape == ours.shape
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.mlx
def test_mxfp4_gather_qmm_agrees_with_dequantise_then_gather_mm():
    """The *matmul* path over raw FP4 bytes, not just the decode.

    A teacher run on this Mac never dequantises the routed experts: the shipped
    ``I8`` code bytes are reinterpreted as ``uint32`` and handed to
    ``mx.gather_qmm(..., mode="mxfp4", group_size=32, bits=4)``, which is 3.42 GB
    per layer against 12.9 GB dequantised to bf16. The whole argument for running
    the teacher locally is that it holds the released weights *exactly*, so the
    kernel that consumes those bytes has to agree with decoding them and doing an
    ordinary gathered matmul -- ``test_matches_mlx_mxfp4_dequantize`` pins the
    decode and says nothing about the GEMM.

    fp32 accumulation order differs between the two paths, so this is a tight
    relative tolerance rather than bit-equality.
    """
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(1192)
    experts, out_dim, in_dim, group = 4, 16, 128, 32
    codes = rng.integers(0, 256, size=(experts, out_dim, in_dim // 2), dtype=np.uint8)
    # Keep the scale exponents near 2**0 so the products stay in a range where a
    # relative tolerance is meaningful, and exclude 0xFF (see the test above).
    scales = rng.integers(
        120, 134, size=(experts, out_dim, in_dim // group), dtype=np.uint8
    )
    packed = mx.array(codes).view(mx.uint32)
    scale_arr = mx.array(scales)

    tokens, top_k = 6, 2
    x = mx.array(rng.normal(size=(tokens, 1, 1, in_dim)).astype(np.float32))
    indices = mx.array(
        rng.integers(0, experts, size=(tokens, top_k)).astype(np.uint32)
    )

    quantised = mx.gather_qmm(
        x, packed, scale_arr, None,
        rhs_indices=indices, transpose=True,
        group_size=group, bits=4, mode="mxfp4",
    )
    dense = mx.dequantize(
        packed, scale_arr, group_size=group, bits=4, mode="mxfp4"
    ).astype(mx.float32)
    reference = mx.gather_mm(x, dense.swapaxes(-1, -2), rhs_indices=indices)

    assert quantised.shape == reference.shape
    gap = float(mx.max(mx.abs(quantised - reference)))
    scale = float(mx.max(mx.abs(reference)))
    assert scale > 0.0
    assert gap <= 1e-5 * scale, f"gather_qmm diverged: {gap} vs scale {scale}"

    # And the decoded weights are exactly the ones this module produces, so the
    # chain checkpoint bytes -> gather_qmm -> logits has no unverified link.
    for expert in range(experts):
        np.testing.assert_array_equal(
            dequantize_fp4_expert(codes[expert], scales[expert]),
            np.array(dense[expert]),
        )


@pytest.mark.mlx
def test_mlx_and_ocp_disagree_only_on_scale_byte_255():
    """Documents the single divergence, so it can never regress silently."""
    mx = pytest.importorskip("mlx.core")
    codes = np.full((1, 16), 0x22, dtype=np.uint8)
    scales = np.array([[255]], dtype=np.uint8)
    theirs = np.array(
        mx.dequantize(
            mx.array(codes).view(mx.uint32), mx.array(scales),
            group_size=32, bits=4, mode="mxfp4",
        ).astype(mx.float32)
    )
    assert np.isinf(theirs).all(), "MLX treats E8M0 0xFF as 2**128 -> inf"
    assert np.isnan(dequantize_fp4_expert(codes, scales)).all(), "we follow OCP"


# --------------------------------------------------------------------------
# real checkpoint (opt-in: needs the local shards)
# --------------------------------------------------------------------------


def _read_raw_tensor(path: Path, name: str) -> tuple[str, tuple[int, ...], np.ndarray]:
    """Read one tensor's raw bytes, ignoring the declared dtype.

    The safetensors header-fallback trick: F8_E8M0 is not a dtype numpy or
    ``safetensors.numpy`` knows, so read the header ourselves and take the
    payload as uint8. Mirrors ``omlx``'s F8_E8M0 -> U8 fallback
    (``omlx/patches/deepseek_v4/utils_patch.py:40``).
    """
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        info = header[name]
        start, end = info["data_offsets"]
        handle.seek(8 + header_len + start)
        payload = handle.read(end - start)
    shape = tuple(info["shape"])
    raw = np.frombuffer(payload, dtype=np.uint8)
    assert raw.size == int(np.prod(shape)), f"{name}: {raw.size} bytes for {shape}"
    return info["dtype"], shape, raw.reshape(shape)


def _load_expert(layer: int, expert: int, proj: str):
    index_path = CHECKPOINT / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip(f"DeepSeek-V4-Flash-0731 shards not present at {CHECKPOINT}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    base = f"layers.{layer}.ffn.experts.{expert}.{proj}"
    wname, sname = f"{base}.weight", f"{base}.scale"
    if wname not in weight_map or sname not in weight_map:
        pytest.skip(f"{base} not in the local index")
    for name in (wname, sname):
        if not (CHECKPOINT / weight_map[name]).exists():
            pytest.skip(f"shard {weight_map[name]} not downloaded")
    wdtype, wshape, codes = _read_raw_tensor(CHECKPOINT / weight_map[wname], wname)
    sdtype, sshape, scales = _read_raw_tensor(CHECKPOINT / weight_map[sname], sname)
    return (wdtype, wshape, codes), (sdtype, sshape, scales)


@pytest.mark.checkpoint
@pytest.mark.parametrize(
    "layer,expert,proj,logical",
    [
        (0, 0, "w1", (2048, 4096)),
        (0, 0, "w2", (4096, 2048)),
        (0, 0, "w3", (2048, 4096)),
        (0, 7, "w1", (2048, 4096)),
        (5, 3, "w2", (4096, 2048)),
    ],
)
def test_real_expert_tensor_decodes_to_plausible_weights(layer, expert, proj, logical):
    (wdtype, wshape, codes), (sdtype, sshape, scales) = _load_expert(layer, expert, proj)

    # The release contract, straight from the shard header.
    assert wdtype == "I8", f"expected packed FP4 in I8, got {wdtype}"
    assert sdtype == "F8_E8M0", f"expected ue8m0 scales, got {sdtype}"
    assert wshape == (logical[0], logical[1] // 2)
    assert sshape == (logical[0], logical[1] // 32)

    weight = dequantize_fp4_expert(codes, scales)
    assert weight.shape == logical
    assert weight.dtype == np.float32

    assert np.isfinite(weight).all(), "decoded expert contains NaN/inf"
    absmax = float(np.abs(weight).max())
    assert 0.01 <= absmax <= 10.0, f"absmax {absmax} outside plausible weight range"
    rms = float(np.sqrt(np.mean(weight.astype(np.float64) ** 2)))
    assert 1e-4 <= rms <= 1.0, f"rms {rms} outside plausible weight range"

    # Trained weights are near-symmetric about zero. A nibble-order or
    # value-table error would skew the sign balance or inflate the mean.
    nonzero = weight != 0
    frac_neg = float((weight < 0).sum()) / float(nonzero.sum())
    assert 0.45 <= frac_neg <= 0.55, f"sign balance {frac_neg} is not symmetric"
    assert abs(float(weight.mean())) < 0.02 * rms, "mean is far from zero"

    # Every value must land on the 16-entry e2m1 grid times a power of two.
    sample = weight[:8].ravel()
    ratio = np.abs(sample) / np.exp2(np.floor(np.log2(np.abs(sample) + 1e-30)))
    on_grid = np.isin(np.round(ratio * 4).astype(np.int64), [4, 6]) | (sample == 0)
    assert on_grid.all(), "decoded values are not on the e2m1 x 2**k grid"


@pytest.mark.checkpoint
def test_real_expert_projections_have_consistent_rms():
    """w1/w2/w3 of one expert are trained together; their rms should track.

    A per-projection decode bug (wrong scale grouping for w2's K=2048, say)
    would show up as one projection's rms drifting from its siblings.
    """
    rms = {}
    for proj in ("w1", "w2", "w3"):
        (_, _, codes), (_, _, scales) = _load_expert(0, 0, proj)
        weight = dequantize_fp4_expert(codes, scales)
        rms[proj] = float(np.sqrt(np.mean(weight.astype(np.float64) ** 2)))
    spread = max(rms.values()) / min(rms.values())
    assert spread < 2.0, f"per-projection rms spread {spread:.3f} too wide: {rms}"


@pytest.mark.checkpoint
def test_real_scale_bytes_stay_in_the_finite_e8m0_range():
    # No 0xFF (E8M0 NaN) anywhere in a real expert scale tensor, which is why
    # the OCP-vs-MLX divergence on 0xFF is documentation rather than a hazard.
    _, (sdtype, _, scales) = _load_expert(0, 0, "w1")
    assert sdtype == "F8_E8M0"
    assert not (scales == 255).any(), "unexpected E8M0 NaN scale in the release"
    assert scales.min() > 0, "unexpected E8M0 subnormal-floor scale in the release"


@pytest.mark.checkpoint
def test_real_expert_matches_deepseek_reference_expression():
    """Real tensor vs a literal transcription of ``inference/convert.py:28-33``.

    The statistical gate above cannot see a nibble swap -- swapping is a
    permutation *within* each 32-wide scale group, so absmax, rms, sign
    balance and the e2m1 grid are all invariant under it. This test and the
    MLX cross-check are what actually pin the order on real data.
    """
    (_, _, codes), (_, _, scales) = _load_expert(0, 0, "w2")
    table = np.array(E2M1_VALUES, dtype=np.float32)
    packed = np.asarray(codes)
    low, high = packed & 0x0F, (packed >> 4) & 0x0F
    reference = np.stack([table[low], table[high]], axis=-1).reshape(
        packed.shape[0], packed.shape[1] * 2
    )
    exponent = scales.astype(np.int64) - 127
    reference = reference * np.repeat(np.exp2(exponent).astype(np.float32), 32, axis=1)
    np.testing.assert_array_equal(dequantize_fp4_expert(codes, scales), reference)


@pytest.mark.checkpoint
def test_real_expert_matches_mlx_mxfp4_end_to_end():
    """Full real tensor, our numpy decode vs the MLX kernel omlx runs."""
    mx = pytest.importorskip("mlx.core")
    (_, _, codes), (_, _, scales) = _load_expert(0, 0, "w1")
    ours = dequantize_fp4_expert(codes, scales)
    theirs = np.array(
        mx.dequantize(
            mx.array(np.ascontiguousarray(codes)).view(mx.uint32),
            mx.array(np.ascontiguousarray(scales)),
            group_size=32,
            bits=4,
            mode="mxfp4",
        ).astype(mx.float32)
    )
    np.testing.assert_array_equal(ours, theirs)
