"""Decode FP8 e4m3 block-quantized weights (DeepSeek-V4 release format).

Format per config.json: quant_method fp8, fmt e4m3, scale_fmt ue8m0,
weight_block_size [128, 128], dynamic activation scheme. Weights are uint8
e4m3fn codes; each 128x128 block has one power-of-two scale stored as an
unsigned 8-bit exponent (ue8m0, bias 127).

This decodes *resident* tensors only -- attention and shared experts. Routed
experts in the same release are FP4 packed two-per-byte in I8 with F8_E8M0
group-32 scales; see the sibling :mod:`keep.convert.fp4_expert`, which reuses
:func:`decode_ue8m0` for its scales so the two paths cannot drift on the
ue8m0 encoding or on the OCP NaN choice for byte 0xFF.

Callers pass raw code bytes: a safetensors/``ml_dtypes`` float8 array (e.g.
``float8_e4m3fn``, ``float8_e8m0fnu``) must be reinterpreted with
``.view(np.uint8)`` first. Arrays still carrying those dtypes are rejected by
design, because numpy would otherwise convert rather than reinterpret them and
the decode would silently run on the wrong bit patterns.

Pure numpy so it runs headless; the MLX fast path can come later if decode
throughput matters.

:func:`repack_fp8_block_as_mxfp8` is the lossless alternative to decoding: it
re-lays the same e4m3 bytes and ue8m0 exponents out in MLX's ``mxfp8`` layout
(group 32 along the input axis), so a resident can stay at 8.25 bits per weight
instead of expanding to bf16, with every dequantized value bit-identical.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MXFP8_GROUP_SIZE",
    "decode_e4m3",
    "decode_ue8m0",
    "dequantize_fp8_block",
    "repack_fp8_block_as_mxfp8",
]

#: MLX ``mxfp8`` supports exactly one group size: 32 elements per E8M0 scale.
MXFP8_GROUP_SIZE = 32

_E4M3_TABLE: np.ndarray | None = None


def _e4m3_table() -> np.ndarray:
    global _E4M3_TABLE
    if _E4M3_TABLE is None:
        codes = np.arange(256, dtype=np.uint16)
        sign = np.where(codes & 0x80, -1.0, 1.0).astype(np.float64)
        exponent = ((codes >> 3) & 0x0F).astype(np.int64)
        mantissa = (codes & 0x07).astype(np.float64)
        normal = sign * (1.0 + mantissa / 8.0) * np.exp2(exponent - 7.0)
        subnormal = sign * (mantissa / 8.0) * np.exp2(-6.0)
        values = np.where(exponent == 0, subnormal, normal)
        # e4m3fn: S.1111.111 is NaN; there are no infinities.
        nan_mask = (exponent == 0x0F) & ((codes & 0x07) == 0x07)
        values = np.where(nan_mask, np.nan, values)
        values = values.astype(np.float32)
        values.flags.writeable = False
        _E4M3_TABLE = values
    return _E4M3_TABLE


def decode_e4m3(codes: np.ndarray) -> np.ndarray:
    if not isinstance(codes, np.ndarray):
        raise ValueError(f"codes must be a numpy ndarray, got {type(codes).__name__}")
    if codes.dtype != np.uint8:
        raise ValueError(f"e4m3 codes must be uint8, got {codes.dtype}")
    return _e4m3_table()[codes]


def decode_ue8m0(scales: np.ndarray) -> np.ndarray:
    """Decode OCP E8M0 (ue8m0) exponent bytes into power-of-two float32 scales.

    Bias-127 encoding: value = 2**(byte - 127) for byte in 0..254. Per the
    OCP Microscaling (MX) E8M0 spec -- matching ml_dtypes.float8_e8m0fnu --
    byte 0xFF (255) decodes to NaN rather than the mathematical 2**128, which
    would silently overflow float32 (max ~3.4028235e38, just under 2**128).
    The NaN path never evaluates 2**128, so no overflow RuntimeWarning is
    raised.
    """
    if not isinstance(scales, np.ndarray):
        raise ValueError(f"scales must be a numpy ndarray, got {type(scales).__name__}")
    if scales.dtype != np.uint8:
        raise ValueError(f"ue8m0 scales must be uint8, got {scales.dtype}")
    is_nan = scales == 255
    # Clamp the exponent for NaN slots to 0 before exp2 so we never compute
    # 2**128 (which would overflow float32 and emit a RuntimeWarning); the
    # clamped result is discarded below in favor of NaN.
    safe_exponent = np.where(is_nan, 0, scales.astype(np.int64) - 127).astype(np.float64)
    powers = np.exp2(safe_exponent)
    return np.where(is_nan, np.float32(np.nan), powers).astype(np.float32)


def dequantize_fp8_block(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    if not isinstance(codes, np.ndarray):
        raise ValueError(f"codes must be a numpy ndarray, got {type(codes).__name__}")
    if not isinstance(scales, np.ndarray):
        raise ValueError(f"scales must be a numpy ndarray, got {type(scales).__name__}")
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got shape {codes.shape}")
    block_rows, block_cols = block_size
    for label, value in (("block_size[0]", block_rows), ("block_size[1]", block_cols)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{label} must be a positive int, got {value!r}")
    rows, cols = codes.shape
    expected = (-(-rows // block_rows), -(-cols // block_cols))
    if scales.shape != expected:
        raise ValueError(
            f"scales shape {scales.shape} does not match "
            f"{expected} blocks for weight {codes.shape} "
            f"at block size {block_size}"
        )
    if np.issubdtype(scales.dtype, np.floating):
        scale_values = scales.astype(np.float32)
    elif scales.dtype == np.uint8:
        scale_values = decode_ue8m0(scales)
    else:
        raise ValueError(
            f"scales must be a floating dtype (literal multipliers) or uint8 "
            f"(ue8m0-encoded), got dtype {scales.dtype}"
        )
    decoded = decode_e4m3(codes)
    nb_r, nb_c = scale_values.shape
    # Broadcast-multiply each block by its scale in place on the freshly
    # decoded (and therefore exclusively owned) buffer, rather than
    # materializing a full-size expanded scale array via np.repeat twice --
    # this keeps peak memory at ~1x the output instead of ~3x.
    for i in range(nb_r):
        r0, r1 = i * block_rows, min((i + 1) * block_rows, rows)
        row_scales = scale_values[i]
        for j in range(nb_c):
            c0, c1 = j * block_cols, min((j + 1) * block_cols, cols)
            decoded[r0:r1, c0:c1] *= row_scales[j]
    return decoded


def _power_of_two_scales_as_ue8m0(scales: np.ndarray) -> np.ndarray:
    """Encode literal power-of-two multipliers as ue8m0 bytes, or refuse.

    Only exact powers of two in ``2**-127 .. 2**127`` have an E8M0 encoding.
    Anything else would need rounding, and rounding is exactly what a lossless
    repack must never do, so it raises instead.
    """

    values = scales.astype(np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("power-of-two block scales must be finite and positive")
    mantissa, exponent = np.frexp(values)
    if np.any(mantissa != 0.5):
        raise ValueError(
            "block scales are not all exact powers of two; an mxfp8 repack "
            "would have to round them, so it is refused"
        )
    unbiased = exponent.astype(np.int64) - 1
    if np.any(unbiased < -127) or np.any(unbiased > 127):
        raise ValueError("power-of-two block scales are outside the E8M0 range")
    return (unbiased + 127).astype(np.uint8)


def repack_fp8_block_as_mxfp8(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    block_size: tuple[int, int] = (128, 128),
    group_size: int = MXFP8_GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-lay block-scaled e4m3 weights out as MLX ``mxfp8`` without changing a bit.

    Returns ``(packed, group_scales)``:

    - ``packed``: ``uint32`` of shape ``(rows, cols // 4)``, the unchanged e4m3
      code bytes read little-endian four to a word, which is the element order
      ``mx.dequantize(..., mode="mxfp8")`` expects.
    - ``group_scales``: ``uint8`` of shape ``(rows, cols // 32)``, each block's
      ue8m0 exponent repeated over the rows and the 32-wide column groups the
      block covers.

    Dequantizing the result reproduces :func:`dequantize_fp8_block` exactly:
    e4m3 times a power of two is representable in float32 and bf16 alike.
    Building the same weights with ``mx.quantize(..., mode="mxfp8")`` instead is
    *not* exact -- it re-derives each group's exponent by rounding -- so this
    function is the only supported way to produce an mxfp8 resident.

    Requires ``cols % 32 == 0`` and ``block_size[1] % 32 == 0`` so every
    32-wide group sits inside one source block. ue8m0 byte 255 (NaN) is
    refused rather than bound.
    """

    if not isinstance(codes, np.ndarray):
        raise ValueError(f"codes must be a numpy ndarray, got {type(codes).__name__}")
    if not isinstance(scales, np.ndarray):
        raise ValueError(f"scales must be a numpy ndarray, got {type(scales).__name__}")
    if codes.dtype != np.uint8:
        raise ValueError(f"e4m3 codes must be uint8, got {codes.dtype}")
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got shape {codes.shape}")
    if group_size != MXFP8_GROUP_SIZE:
        raise ValueError(
            f"mxfp8 has a fixed group size of {MXFP8_GROUP_SIZE}, got {group_size!r}"
        )
    block_rows, block_cols = block_size
    for label, value in (("block_size[0]", block_rows), ("block_size[1]", block_cols)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{label} must be a positive int, got {value!r}")
    rows, cols = codes.shape
    if cols % group_size != 0:
        raise ValueError(
            f"input dimension {cols} is not a multiple of the mxfp8 group size {group_size}"
        )
    if block_cols % group_size != 0:
        raise ValueError(
            f"block width {block_cols} is not a multiple of the mxfp8 group size "
            f"{group_size}, so a group would straddle two source scales"
        )
    expected = (-(-rows // block_rows), -(-cols // block_cols))
    if scales.shape != expected:
        raise ValueError(
            f"scales shape {scales.shape} does not match "
            f"{expected} blocks for weight {codes.shape} "
            f"at block size {block_size}"
        )
    if scales.dtype == np.uint8:
        if np.any(scales == 255):
            raise ValueError("ue8m0 byte 255 is NaN; refusing to repack a NaN block scale")
        exponents = scales
    elif np.issubdtype(scales.dtype, np.floating):
        exponents = _power_of_two_scales_as_ue8m0(scales)
    else:
        raise ValueError(
            f"scales must be uint8 (ue8m0-encoded) or a floating dtype holding "
            f"powers of two, got dtype {scales.dtype}"
        )

    # ``astype`` copies, so the result never aliases the caller's code buffer.
    packed = np.ascontiguousarray(codes).view("<u4").astype(np.uint32)
    groups_per_block = block_cols // group_size
    group_scales = np.repeat(np.repeat(exponents, block_rows, axis=0)[:rows], groups_per_block, axis=1)
    group_scales = np.ascontiguousarray(group_scales[:, : cols // group_size], dtype=np.uint8)
    return packed, group_scales
