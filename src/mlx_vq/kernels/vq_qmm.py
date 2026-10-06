from __future__ import annotations

from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import CODEWORD_DIM, decode_weight_matrix, e8_1bit_packed, e8p_packed_abs_grid
from mlx_vq.kernels.vq_qmv import _as_mlx, _as_numpy, _dtype_name, _qmv_output_dtype


_KERNEL_DIR = Path(__file__).resolve().parent
_VQ_QMM_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_qmm",
    input_names=["x", "codes", "scales", "codebook"],
    output_names=["out"],
    source=(_KERNEL_DIR / "vq_qmm.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)


def _default_codebook(code_bits: Literal[8, 16]) -> np.ndarray:
    if code_bits == 8:
        return e8_1bit_packed()
    return e8p_packed_abs_grid()


def _validate_qmm_metadata(
    x: mx.array,
    codes: mx.array,
    scales: mx.array,
    codebook: mx.array,
    *,
    in_dim: int,
    out_dim: int,
    group_size: int,
    code_bits: Literal[8, 16],
    codebook_duplication: Literal[1, 4, 8],
) -> None:
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if in_dim <= 0 or out_dim <= 0:
        raise ValueError("in_dim and out_dim must be positive")
    if in_dim % CODEWORD_DIM != 0:
        raise ValueError("in_dim must be divisible by 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if in_dim % group_size != 0:
        raise ValueError("in_dim must be divisible by group_size")
    if x.ndim != 2 or x.shape[1] != in_dim:
        raise ValueError(f"x must have shape [M, {in_dim}], found {x.shape}")
    if codes.shape != (out_dim, in_dim // CODEWORD_DIM):
        raise ValueError(f"codes must have shape ({out_dim}, {in_dim // CODEWORD_DIM}), found {codes.shape}")
    if scales.shape != (out_dim, in_dim // group_size):
        raise ValueError(f"scales must have shape ({out_dim}, {in_dim // group_size}), found {scales.shape}")
    if codebook.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook.shape}")
    if codebook_duplication not in (1, 4, 8):
        raise ValueError("codebook_duplication must be 1, 4, or 8")

    expected_code_dtype = "uint8" if code_bits == 8 else "uint16"
    if _dtype_name(codes.dtype) != expected_code_dtype:
        raise ValueError(f"codes dtype must be {expected_code_dtype}, found {codes.dtype}")
    if _dtype_name(codebook.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook.dtype}")


def vq_qmm_reference_np(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None = None,
    *,
    in_dim: int,
    out_dim: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
) -> np.ndarray:
    x_np = _as_numpy(x, np.float32)
    if x_np.ndim != 2 or x_np.shape[1] != in_dim:
        raise ValueError(f"x must have shape [M, {in_dim}], found {x_np.shape}")
    codes_dtype = np.uint8 if code_bits == 8 else np.uint16
    codes_np = _as_numpy(codes, codes_dtype)
    scales_np = _as_numpy(scales, np.float32)
    codebook_np = _default_codebook(code_bits) if codebook is None else _as_numpy(codebook, np.uint32)
    weights = decode_weight_matrix(codes_np, scales_np, code_bits=code_bits, codebook=codebook_np)
    if weights.shape != (out_dim, in_dim):
        raise ValueError(f"decoded weights must have shape ({out_dim}, {in_dim}), found {weights.shape}")
    return (x_np @ weights.T).astype(np.float32, copy=False)


def vq_qmm(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None = None,
    *,
    in_dim: int,
    out_dim: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    implementation: Literal["metal", "reference"] = "metal",
    codebook_duplication: Literal[1, 4, 8] = 1,
) -> mx.array:
    if implementation == "reference":
        return mx.array(
            vq_qmm_reference_np(
                x,
                codes,
                scales,
                codebook,
                in_dim=in_dim,
                out_dim=out_dim,
                group_size=group_size,
                code_bits=code_bits,
            )
        )
    if implementation != "metal":
        raise ValueError("implementation must be 'metal' or 'reference'")

    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    _validate_qmm_metadata(
        x_mx,
        codes_mx,
        scales_mx,
        codebook_mx,
        in_dim=in_dim,
        out_dim=out_dim,
        group_size=group_size,
        code_bits=code_bits,
        codebook_duplication=codebook_duplication,
    )

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    outputs = _VQ_QMM_KERNEL(
        inputs=[x_mx, codes_mx, scales_mx, codebook_mx],
        template=[
            ("CODE_BITS", code_bits),
            ("CODEBOOK_DUP", codebook_duplication),
            ("OUT_T", output_dtype),
        ],
        grid=(256, out_dim, x_mx.shape[0]),
        threadgroup=(256, 1, 1),
        output_shapes=[(x_mx.shape[0], out_dim)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def vq_qmm_reference(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None = None,
    *,
    in_dim: int,
    out_dim: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
) -> mx.array:
    return mx.array(
        vq_qmm_reference_np(
            x,
            codes,
            scales,
            codebook,
            in_dim=in_dim,
            out_dim=out_dim,
            group_size=group_size,
            code_bits=code_bits,
        )
    )
