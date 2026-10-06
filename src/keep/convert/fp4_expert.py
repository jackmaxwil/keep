"""Decode FP4 e2m1 routed-expert weights (DeepSeek-V4-Flash release format).

Sibling of :mod:`keep.convert.fp8_block`, which handles the *resident* tensors
(attention, shared experts) of the same release. This module handles the
routed experts -- the tensors KEEP actually compresses -- which ship in a
different, denser encoding that no numpy dtype can express.

Determined format
=================

Measured on ``deepseek-ai/DeepSeek-V4-Flash-0731`` revision ``7872f01b``,
whose ``config.json`` declares ``expert_dtype: "fp4"``.

Storage
    ``layers.<L>.ffn.experts.<E>.<proj>.weight`` is safetensors ``I8`` of
    shape ``[O, K/2]`` for a logical ``[O, K]`` weight -- two FP4 values per
    stored byte, packed along the input (``K``) dimension. ``w1``/``w3`` are
    ``I8 [2048, 2048]`` for logical ``[2048, 4096]``; ``w2`` is
    ``I8 [4096, 1024]`` for logical ``[4096, 2048]``. ``I8`` is a stand-in for
    "raw bytes": the values are unsigned 4-bit code pairs, not int8 numbers.
    Confirmed by ``inference/model.py:137-143``, which allocates the parameter
    as ``torch.empty(out_features, in_features // 2,
    dtype=torch.float4_e2m1fn_x2)``, and by ``inference/convert.py:135``,
    which turns the shipped ``I8`` tensor into that dtype with a bare
    ``.view()`` -- a reinterpret, no data movement.

Nibble order
    **The low nibble is the first logical element**; the high nibble is the
    second. Logical column ``2*j`` comes from ``byte[j] & 0x0F`` and column
    ``2*j + 1`` from ``byte[j] >> 4``. From ``inference/convert.py:30-33``::

        x = x.view(torch.uint8)
        low  = x & 0x0F
        high = (x >> 4) & 0x0F
        x = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1).flatten(2)

    ``stack(..., dim=-1)`` puts ``low`` at even and ``high`` at odd positions
    of the trailing axis, and the subsequent
    ``x.view(bOut, 128, bIn, 128)`` (``convert.py:42``) reads that contiguous
    buffer as the logical ``[O, K]`` row. Independently confirmed against MLX:
    ``.repos/omlx`` feeds the shipped bytes to MLX's ``mxfp4`` path *without
    reordering them* -- ``omlx/patches/deepseek_v4/deepseek_v4_model.py:2219-2226``
    passes ``weight.view(mx.uint32)`` and the raw ``.scale`` bytes as
    ``scales``, under ``{"group_size": 32, "bits": 4, "mode": "mxfp4"}``
    (``deepseek_v4_model.py:191``) -- so MLX's unpack order is the release's
    unpack order. ``mx.dequantize(..., mode="mxfp4")`` agrees with this module
    bit-for-bit on random bytes and on real expert tensors
    (``tests/test_fp4_expert_dequant.py``).

Value table
    Standard OCP e2m1 (sign, 2-bit exponent, 1-bit mantissa), no NaN and no
    infinities -- all 16 codes are finite. Transcribed from
    ``inference/convert.py:11-14``::

        FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    One deliberate divergence: code ``0x8`` (sign=1, exponent=0, mantissa=0)
    is e2m1 negative zero. DeepSeek's table writes it as a bare ``0.0``; MLX's
    ``mxfp4`` kernel and the OCP spec both produce ``-0.0``. We follow
    OCP/MLX, so the sign bit survives the decode -- numerically identical, and
    consistent with :func:`keep.convert.fp8_block.decode_e4m3`, which likewise
    preserves signed zero.

Scales
    ``<proj>.scale`` is safetensors ``F8_E8M0`` of shape ``[O, K/32]``: one
    unsigned bias-127 power-of-two exponent per **32 logical input elements**
    (16 stored bytes). ``4096/128 == 2048/64 == 32``. Declared at
    ``inference/model.py:139-143`` ("1 scale per 32 fp4 elements along K",
    ``dtype=torch.float8_e8m0fnu``) and asserted at ``inference/convert.py:28``
    (``scale.size(1) == in_dim // fp4_block_size`` with ``fp4_block_size = 32``
    from ``convert.py:26`` / ``model.py:18``).

Scale rule
    ``logical[o, g*32:(g+1)*32] = e2m1_value(codes) * 2**(scale[o, g] - 127)``
    -- a plain per-group multiply, no offset and no second level of scaling.
    Derived from ``inference/convert.py:44-52``, which recasts FP4 to FP8 by
    factoring each 128x128 block's scales into a shared block exponent
    ``s_blk = amax(scale)/2**6`` and a per-group residue
    ``offset = scale/s_blk``, then emits ``x_fp4 * offset`` with block scale
    ``s_blk``; the product ``offset * s_blk`` telescopes back to ``scale``, so
    the dequantized value is exactly ``x_fp4 * scale``. Matches the GEMM,
    which multiplies the accumulator by ``scales_b[n, k]`` for the ``k``-th
    32-wide slice (``inference/kernel.py:498-509``).

    E8M0 byte ``0xFF`` decodes to NaN, not ``2**128``, via the shared
    :func:`keep.convert.fp8_block.decode_ue8m0` -- the OCP Microscaling
    choice, which also avoids a float32 overflow warning. MLX's ``mxfp4``
    yields ``+inf`` there instead; real release shards contain no ``0xFF``
    scale bytes, so the divergence is documentation rather than a hazard.

Measured sanity (revision ``7872f01b``, layer 0 expert 0)
    ``absmax`` 0.125-0.25, ``rms`` ~0.0246 for all three projections, mean
    ~1e-5, 43.5% of values negative and 13% exactly zero -- a symmetric,
    plausibly-scaled weight distribution. Scale exponents span ``2**-8`` to
    ``2**-4``, consistent with ``absmax = 6.0 * 2**-5``.

Pure numpy so it runs headless; an MLX fast path can come later if decode
throughput matters. Callers pass raw bytes: ``uint8``, or the ``int8`` numpy
loads ``I8`` as (reinterpreted, never converted). Arrays carrying an
``ml_dtypes`` float8 dtype are rejected by design, because numpy would
convert rather than reinterpret them.
"""

from __future__ import annotations

import numpy as np

from keep.convert.fp8_block import decode_ue8m0

__all__ = ["decode_fp4_e2m1", "dequantize_fp4_expert"]

# OCP e2m1 by 4-bit code: bit 3 sign, bits 2-1 exponent, bit 0 mantissa.
# Index 8 is -0.0 (see the module docstring on the DeepSeek divergence).
_E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
_E2M1_TABLE.flags.writeable = False


def _as_code_bytes(codes: np.ndarray, label: str = "codes") -> np.ndarray:
    """Validate a packed-FP4 array and return it as uint8 without converting."""
    if not isinstance(codes, np.ndarray):
        raise ValueError(
            f"{label} must be a numpy ndarray, got {type(codes).__name__}"
        )
    if codes.dtype == np.uint8:
        return codes
    if codes.dtype == np.int8:
        # safetensors declares the packed nibbles as I8; reinterpret the sign
        # bit as code bit 7 rather than letting numpy convert the value.
        return codes.view(np.uint8)
    raise ValueError(
        f"packed FP4 {label} must be uint8 or int8 (raw bytes, two e2m1 "
        f"values each), got dtype {codes.dtype}"
    )


def decode_fp4_e2m1(codes: np.ndarray) -> np.ndarray:
    """Unpack FP4 e2m1 code bytes into float32, two values per byte.

    The last axis doubles: element ``2*j`` of the output is the *low* nibble
    of byte ``j`` and element ``2*j + 1`` is the high nibble, matching the
    release's packing order (see the module docstring).

    Args:
        codes: uint8 or int8 array of packed nibble pairs, any rank.

    Returns:
        float32 array with the same leading axes and the last axis doubled.
        Every value is finite -- e2m1fn has no NaN and no infinities.
    """
    codes = _as_code_bytes(codes)
    table = _E2M1_TABLE
    out = np.empty((*codes.shape[:-1], codes.shape[-1] * 2), dtype=np.float32)
    # Strided writes into a single output buffer: no full-size temporaries.
    out[..., 0::2] = table[codes & 0x0F]
    out[..., 1::2] = table[codes >> 4]
    return out


def dequantize_fp4_expert(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int = 32,
) -> np.ndarray:
    """Dequantize one packed FP4 routed-expert weight to float32.

    Args:
        codes: uint8/int8 packed weight of shape ``[O, K/2]`` -- logically
            ``[O, K]`` FP4 values, two per byte along the input dimension.
        scales: shape ``[O, K/group_size]``. ``uint8`` is read as ue8m0
            (bias-127 power-of-two exponents, ``0xFF`` -> NaN); a floating
            dtype is used as literal multipliers.
        group_size: logical input elements sharing one scale. The release uses
            32; must be positive and even so a group starts on a byte
            boundary.

    Returns:
        float32 weight of shape ``[O, K]``.

    Raises:
        ValueError: on a non-ndarray input, non-2-D ``codes``, an unsupported
            dtype, a ``K`` that is not a whole multiple of ``group_size``, or a
            ``scales`` shape that does not match the implied group grid.
    """
    if not isinstance(scales, np.ndarray):
        raise ValueError(
            f"scales must be a numpy ndarray, got {type(scales).__name__}"
        )
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, (int, np.integer))
        or group_size <= 0
    ):
        raise ValueError(f"group_size must be a positive int, got {group_size!r}")
    if group_size % 2:
        raise ValueError(
            f"group_size must be even so each scale group starts on a packed "
            f"byte boundary, got {group_size}"
        )

    code_bytes = _as_code_bytes(codes)
    if code_bytes.ndim != 2:
        raise ValueError(f"codes must be 2-D [O, K/2], got shape {code_bytes.shape}")
    rows, packed_cols = code_bytes.shape
    logical_cols = packed_cols * 2
    if logical_cols == 0:
        raise ValueError("codes has zero packed columns; nothing to dequantize")
    if logical_cols % group_size:
        raise ValueError(
            f"logical input dim {logical_cols} (from {packed_cols} packed "
            f"bytes) is not a multiple of group_size {group_size}; a partial "
            f"trailing group has no well-defined scale"
        )
    n_groups = logical_cols // group_size
    if scales.shape != (rows, n_groups):
        raise ValueError(
            f"scales shape {scales.shape} does not match the expected "
            f"{(rows, n_groups)} for a logical {(rows, logical_cols)} weight "
            f"at group_size {group_size}"
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

    decoded = decode_fp4_e2m1(code_bytes)
    # Multiply in place through a reshaped *view* of the freshly decoded (and
    # therefore exclusively owned) buffer: no np.repeat of the scale grid, so
    # peak memory stays at ~1x the output instead of ~2-3x.
    decoded.reshape(rows, n_groups, group_size)[...] *= scale_values[:, :, None]
    return decoded
