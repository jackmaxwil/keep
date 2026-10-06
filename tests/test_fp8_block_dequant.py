"""Bit-exact FP8 e4m3 decode and 128x128 block dequantization."""

import warnings

import numpy as np
import pytest

from keep.convert.fp8_block import (
    decode_e4m3,
    decode_ue8m0,
    dequantize_fp8_block,
)


def test_e4m3_known_values():
    # 0x00 = +0.0; 0x38 = 1.0 (exp 0b0111=bias, mantissa 0)
    # 0x3C = 1.5; 0xB8 = -1.0; 0x7E = 448.0 (max normal)
    codes = np.array([0x00, 0x38, 0x3C, 0xB8, 0x7E], dtype=np.uint8)
    expected = np.array([0.0, 1.0, 1.5, -1.0, 448.0], dtype=np.float32)
    np.testing.assert_array_equal(decode_e4m3(codes), expected)


def test_e4m3_signed_zero():
    # 0x00 = +0.0, 0x80 = -0.0: distinct bit patterns, both compare equal to
    # 0.0 under IEEE 754, so this must check the sign bit explicitly.
    values = decode_e4m3(np.array([0x00, 0x80], dtype=np.uint8))
    assert values[0] == 0.0 and not np.signbit(values[0])
    assert values[1] == 0.0 and np.signbit(values[1])


def test_e4m3_subnormals():
    # exponent bits 0000, mantissa m: value = m/8 * 2**-6
    codes = np.array([0x01, 0x07], dtype=np.uint8)
    expected = np.array([2.0**-9, 7.0 / 8.0 * 2.0**-6], dtype=np.float32)
    np.testing.assert_allclose(decode_e4m3(codes), expected, rtol=0)


def test_e4m3_nan():
    # S.1111.111 is NaN in e4m3 (no infinities)
    values = decode_e4m3(np.array([0x7F, 0xFF], dtype=np.uint8))
    assert np.isnan(values).all()


def test_e4m3_roundtrip_against_ml_dtypes():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    codes = np.arange(256, dtype=np.uint8)
    reference = codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    ours = decode_e4m3(codes)
    both_nan = np.isnan(reference) & np.isnan(ours)
    np.testing.assert_array_equal(ours[~both_nan], reference[~both_nan])


def test_e4m3_rejects_non_ndarray():
    with pytest.raises(ValueError):
        decode_e4m3([0x38, 0x3C])  # type: ignore[arg-type]


def test_ue8m0_scale_decode():
    scales = np.array([127, 128, 126, 0], dtype=np.uint8)
    expected = np.array([1.0, 2.0, 0.5, 2.0**-127], dtype=np.float32)
    np.testing.assert_array_equal(decode_ue8m0(scales), expected)


def test_ue8m0_255_is_nan_without_warning():
    # 0xFF is the OCP E8M0 NaN encoding, not 2**128 (which would overflow
    # float32). Decoding it must not raise/emit any RuntimeWarning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = decode_ue8m0(np.array([255], dtype=np.uint8))
    assert caught == []
    assert np.isnan(result).all()


def test_ue8m0_roundtrip_against_ml_dtypes():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    scales = np.arange(256, dtype=np.uint8)
    reference = scales.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    ours = decode_ue8m0(scales)
    both_nan = np.isnan(reference) & np.isnan(ours)
    np.testing.assert_array_equal(ours[~both_nan], reference[~both_nan])


def test_ue8m0_rejects_non_ndarray():
    with pytest.raises(ValueError):
        decode_ue8m0([127, 128])  # type: ignore[arg-type]


def test_block_dequant_applies_per_block_scale():
    # 256x256 weight, 2x2 blocks of 128: every code is 1.0 (0x38),
    # scales double per block; output must equal the block scale.
    codes = np.full((256, 256), 0x38, dtype=np.uint8)
    scales = np.array([[127, 128], [129, 130]], dtype=np.uint8)
    weight = dequantize_fp8_block(codes, scales)
    assert weight.shape == (256, 256)
    assert weight.dtype == np.float32
    assert weight[0, 0] == 1.0
    assert weight[0, 255] == 2.0
    assert weight[255, 0] == 4.0
    assert weight[255, 255] == 8.0


def test_block_dequant_ragged_edge():
    # O and I not multiples of 128: last blocks are partial.
    codes = np.full((130, 200), 0x38, dtype=np.uint8)
    scales = np.full((2, 2), 128, dtype=np.uint8)
    weight = dequantize_fp8_block(codes, scales)
    assert weight.shape == (130, 200)
    assert (weight == 2.0).all()


def test_block_dequant_accepts_float_scales():
    codes = np.full((128, 128), 0x38, dtype=np.uint8)
    scales = np.array([[3.0]], dtype=np.float32)
    weight = dequantize_fp8_block(codes, scales)
    assert (weight == 3.0).all()


def test_block_dequant_rejects_wrong_scale_shape():
    codes = np.full((256, 256), 0x38, dtype=np.uint8)
    scales = np.full((1, 2), 127, dtype=np.uint8)
    with pytest.raises(ValueError):
        dequantize_fp8_block(codes, scales)


def test_block_dequant_rejects_int_scale_dtype():
    # An int32/int64 scale array used to fall into the "literal multiplier"
    # branch and get silently mis-scaled (e.g. up to 127x); it must now be
    # rejected outright, naming the offending dtype.
    codes = np.full((128, 128), 0x38, dtype=np.uint8)
    scales = np.array([[3]], dtype=np.int32)
    with pytest.raises(ValueError, match="int32"):
        dequantize_fp8_block(codes, scales)


def test_block_dequant_custom_block_size():
    # Non-default, non-square block size.
    codes = np.full((64, 96), 0x38, dtype=np.uint8)
    scales = np.array([[127, 128, 129]], dtype=np.uint8)
    weight = dequantize_fp8_block(codes, scales, block_size=(64, 32))
    assert weight.shape == (64, 96)
    assert (weight[:, 0:32] == 1.0).all()
    assert (weight[:, 32:64] == 2.0).all()
    assert (weight[:, 64:96] == 4.0).all()


def test_block_dequant_rejects_non_positive_block_size():
    codes = np.full((128, 128), 0x38, dtype=np.uint8)
    scales = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        dequantize_fp8_block(codes, scales, block_size=(0, 128))
    with pytest.raises(ValueError):
        dequantize_fp8_block(codes, scales, block_size=(128, -1))


def test_block_dequant_nan_survives():
    # A NaN e4m3 code must propagate as NaN into the dequantized output,
    # not get silently clobbered by the scale multiply.
    codes = np.full((128, 128), 0x38, dtype=np.uint8)
    codes[0, 0] = 0x7F  # e4m3 NaN
    scales = np.array([[2.0]], dtype=np.float32)
    weight = dequantize_fp8_block(codes, scales)
    assert np.isnan(weight[0, 0])
    assert weight[0, 1] == 2.0


def test_block_dequant_rejects_non_ndarray_inputs():
    scales = np.array([[1.0]], dtype=np.float32)
    codes = np.full((128, 128), 0x38, dtype=np.uint8)
    with pytest.raises(ValueError):
        dequantize_fp8_block([[0x38]], scales)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dequantize_fp8_block(codes, [[1.0]])  # type: ignore[arg-type]
