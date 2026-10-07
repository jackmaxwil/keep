from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from keep.vq.e8 import (
    CODEWORD_DIM,
    decode_weight_matrix,
    e8_1bit_grid,
    e8_1bit_packed,
    e8p_abs_grid,
    e8p_full_grid,
    e8p_packed_abs_grid,
    encode_e8p_rtn,
    encode_e8p_rtn_diagonal_hessian,
)

ScaleEstimator = Literal["max_abs", "percentile_99"]


@dataclass(frozen=True)
class QuantizedWeight:
    codes: np.ndarray
    scales: np.ndarray
    codebook: np.ndarray
    group_size: int
    code_bits: int
    scale_estimator: ScaleEstimator = "max_abs"


def _validate_weight(weight: np.ndarray, group_size: int) -> np.ndarray:
    values = np.asarray(weight, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in], found shape {values.shape}")
    if values.shape[1] % CODEWORD_DIM != 0:
        raise ValueError("weight input dimension must be divisible by 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if values.shape[1] % group_size != 0:
        raise ValueError("weight input dimension must be divisible by group_size")
    if group_size % CODEWORD_DIM != 0:
        raise ValueError("group_size must be divisible by 8")
    return values


def _codebook_for_bits(code_bits: Literal[8, 16]) -> tuple[np.ndarray, np.ndarray]:
    if code_bits == 8:
        return e8_1bit_grid().astype(np.float32), e8_1bit_packed()
    if code_bits == 16:
        scale_grid = np.array([np.max(np.abs(e8p_abs_grid())) + np.float32(0.25)], dtype=np.float32)
        return scale_grid, e8p_packed_abs_grid()
    raise ValueError("code_bits must be 8 or 16")


def _nearest_codebook_indices(
    vectors: np.ndarray,
    *,
    codebook: np.ndarray,
    index_dtype: np.dtype,
    vector_chunk_size: int,
    codebook_chunk_size: int = 8192,
) -> np.ndarray:
    table = np.asarray(codebook, dtype=np.float32)
    table_norm = np.sum(table * table, axis=1)
    flat = vectors.reshape(-1, CODEWORD_DIM).astype(np.float32, copy=False)
    result = np.empty(flat.shape[0], dtype=index_dtype)

    for start in range(0, flat.shape[0], vector_chunk_size):
        chunk = flat[start : start + vector_chunk_size]
        chunk_norm = np.sum(chunk * chunk, axis=1)
        best_idx = np.zeros(chunk.shape[0], dtype=np.int64)
        best_dist = np.full(chunk.shape[0], np.inf, dtype=np.float32)
        for table_start in range(0, table.shape[0], codebook_chunk_size):
            table_chunk = table[table_start : table_start + codebook_chunk_size]
            distances = (
                chunk_norm[:, None]
                + table_norm[table_start : table_start + table_chunk.shape[0]][None, :]
                - 2 * chunk @ table_chunk.T
            )
            local_idx = np.argmin(distances, axis=1)
            local_dist = distances[np.arange(chunk.shape[0]), local_idx]
            improved = local_dist < best_dist
            best_dist[improved] = local_dist[improved]
            best_idx[improved] = table_start + local_idx[improved]
        result[start : start + chunk.shape[0]] = best_idx.astype(index_dtype)

    return result.reshape(vectors.shape[:-1])


def nearest_codebook_indices_diagonal_hessian(
    vectors: np.ndarray,
    diagonal: np.ndarray,
    *,
    codebook: np.ndarray,
    index_dtype: np.dtype,
    vector_chunk_size: int,
    codebook_chunk_size: int = 8192,
) -> np.ndarray:
    """Return deterministic nearest codes under a shared diagonal metric."""

    values = np.asarray(vectors, dtype=np.float32)
    weights = np.asarray(diagonal, dtype=np.float32)
    table = np.asarray(codebook, dtype=np.float32)
    if values.shape[-1] != CODEWORD_DIM or table.ndim != 2 or table.shape[1] != CODEWORD_DIM:
        raise ValueError("vectors and codebook must have trailing dimension 8")
    if weights.shape != (CODEWORD_DIM,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("diagonal must be finite, non-negative, and have shape (8,)")
    if not np.any(weights > 0):
        raise ValueError("diagonal must contain positive weight")
    if vector_chunk_size <= 0 or codebook_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")

    flat = values.reshape(-1, CODEWORD_DIM)
    result = np.empty(flat.shape[0], dtype=index_dtype)
    weighted_table = table * weights[None, :]
    table_norm = np.sum(weighted_table * table, axis=1)
    for start in range(0, flat.shape[0], vector_chunk_size):
        chunk = flat[start : start + vector_chunk_size]
        vector_norm = np.sum(chunk * chunk * weights[None, :], axis=1)
        best_idx = np.zeros(chunk.shape[0], dtype=np.int64)
        best_dist = np.full(chunk.shape[0], np.inf, dtype=np.float32)
        for table_start in range(0, table.shape[0], codebook_chunk_size):
            table_stop = min(table_start + codebook_chunk_size, table.shape[0])
            distances = (
                vector_norm[:, None]
                + table_norm[table_start:table_stop][None, :]
                - 2.0 * chunk @ weighted_table[table_start:table_stop].T
            )
            local_idx = np.argmin(distances, axis=1)
            local_dist = distances[np.arange(chunk.shape[0]), local_idx]
            improved = local_dist < best_dist
            best_dist[improved] = local_dist[improved]
            best_idx[improved] = table_start + local_idx[improved]
        result[start : start + chunk.shape[0]] = best_idx.astype(index_dtype)
    return result.reshape(values.shape[:-1])


def nearest_e8_codes_diagonal_hessian(
    vectors: np.ndarray,
    diagonal: np.ndarray,
    *,
    backend: str = "numpy",
    index_dtype: np.dtype = np.dtype(np.uint8),
) -> np.ndarray:
    """Return nearest codes from the 256-row E8 grid under a diagonal metric."""

    values = np.asarray(vectors)
    weights = np.asarray(diagonal, dtype=np.float32)
    codebook = e8_1bit_grid().astype(np.float32)

    if backend == "metal":
        try:
            from keep.quant.e8p_metal import (
                nearest_codebook_diagonal_hessian_fused_mlx,
            )
        except ImportError:
            backend = "numpy"
        else:
            return nearest_codebook_diagonal_hessian_fused_mlx(
                values,
                weights,
                codebook,
                index_dtype=index_dtype,
            )

    if backend == "numpy":
        if weights.ndim == 1:
            return nearest_codebook_indices_diagonal_hessian(
                values,
                weights,
                codebook=codebook,
                index_dtype=index_dtype,
                vector_chunk_size=4096,
                codebook_chunk_size=8192,
            )

        flat = values.reshape(-1, CODEWORD_DIM)
        if weights.shape != flat.shape:
            raise ValueError("per-row diagonal must have shape (N, 8)")
        result = np.empty(flat.shape[0], dtype=index_dtype)
        unique_weights, inverse = np.unique(weights, axis=0, return_inverse=True)
        for group, shared_weights in enumerate(unique_weights):
            rows = np.flatnonzero(inverse == group)
            result[rows] = nearest_codebook_indices_diagonal_hessian(
                flat[rows],
                shared_weights,
                codebook=codebook,
                index_dtype=index_dtype,
                vector_chunk_size=4096,
                codebook_chunk_size=8192,
            )
        return result.reshape(values.shape[:-1])

    raise ValueError(f"unknown backend {backend!r}")


def nearest_e8p_codes_diagonal_hessian(
    vectors: np.ndarray,
    diagonal: np.ndarray,
    *,
    backend: str = "numpy",
) -> np.ndarray:
    """Fast factored E8P code search; byte-exact vs the exhaustive full grid.

    Default backend is "numpy": the deterministic, byte-identity-preserving
    reference. "metal" is an explicit opt-in accelerator; it falls back to the
    numpy reference only when the optional MLX module is unimportable, and lets
    any runtime error from the Metal encoder propagate rather than silently
    masking a broken backend.
    """

    if backend == "numpy":
        return encode_e8p_rtn_diagonal_hessian(vectors, diagonal)
    if backend == "metal":
        try:
            from keep.quant.e8p_metal import encode_e8p_diagonal_hessian_fused_mlx
        except ImportError:
            return encode_e8p_rtn_diagonal_hessian(vectors, diagonal)
        return encode_e8p_diagonal_hessian_fused_mlx(vectors, diagonal)
    if backend == "exhaustive":
        values = np.asarray(vectors)
        weights = np.asarray(diagonal, dtype=np.float32)
        grid = e8p_full_grid().astype(np.float32)
        if weights.ndim > 1:
            # Per-row diagonals: loop each row (test-only oracle, slow).
            flat = values.reshape(-1, CODEWORD_DIM)
            if weights.shape != flat.shape:
                raise ValueError("per-row diagonal must have shape (N, 8)")
            result = np.empty(flat.shape[0], dtype=np.uint16)
            for row in range(flat.shape[0]):
                result[row] = nearest_codebook_indices_diagonal_hessian(
                    flat[row : row + 1],
                    weights[row],
                    codebook=grid,
                    index_dtype=np.dtype(np.uint16),
                    vector_chunk_size=1,
                    codebook_chunk_size=8192,
                )[0]
            return result.reshape(values.shape[:-1])
        return nearest_codebook_indices_diagonal_hessian(
            values,
            diagonal,
            codebook=grid,
            index_dtype=np.dtype(np.uint16),
            vector_chunk_size=max(1, values.shape[0]) if values.ndim > 1 else 4096,
            codebook_chunk_size=8192,
        )
    raise ValueError(f"unknown backend {backend!r}")


def _estimate_group_scales(
    group_values: np.ndarray,
    *,
    scale_estimator: ScaleEstimator,
    codebook: np.ndarray,
) -> np.ndarray:
    abs_values = np.abs(group_values)
    divisor = np.max(np.abs(codebook)).astype(np.float32)
    if divisor <= 0:
        raise ValueError("codebook scale divisor must be positive")
    if scale_estimator == "max_abs":
        return np.max(abs_values, axis=2) / divisor
    if scale_estimator == "percentile_99":
        return np.percentile(abs_values, 99.0, axis=2).astype(np.float32) / divisor
    raise ValueError(f"unknown scale_estimator {scale_estimator!r}")


def quantize_weight_rtn(
    weight: np.ndarray,
    *,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    codeword_chunk_size: int = 65536,
    scale_estimator: ScaleEstimator = "max_abs",
) -> QuantizedWeight:
    """Round a dense weight matrix to a frozen E8-family VQ table.

    This is a data-free RTN path: each row/group receives a scale, then
    each 8D vector is rounded to the nearest VQ codeword.
    """

    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    values = _validate_weight(weight, group_size)
    full_codebook, packed_codebook = _codebook_for_bits(code_bits)

    out_dim, in_dim = values.shape
    groups = in_dim // group_size
    group_values = values.reshape(out_dim, groups, group_size)
    scales = _estimate_group_scales(
        group_values,
        scale_estimator=scale_estimator,
        codebook=full_codebook,
    ).astype(np.float32)
    safe_scales = np.where(scales > 0, scales, np.float32(1.0)).astype(np.float32)

    vectors = group_values.reshape(out_dim, groups, group_size // CODEWORD_DIM, CODEWORD_DIM)
    normalized = vectors / safe_scales[:, :, None, None]
    if code_bits == 8:
        codes_by_group = _nearest_codebook_indices(
            normalized,
            codebook=full_codebook,
            index_dtype=np.dtype(np.uint8),
            vector_chunk_size=codeword_chunk_size,
        )
    else:
        codes_by_group = encode_e8p_rtn(normalized, chunk_size=min(codeword_chunk_size, 8192))
    codes = codes_by_group.reshape(out_dim, in_dim // CODEWORD_DIM)
    return QuantizedWeight(
        codes=codes.astype(np.uint8 if code_bits == 8 else np.uint16, copy=False),
        scales=scales.astype(np.float16),
        codebook=packed_codebook,
        group_size=group_size,
        code_bits=code_bits,
        scale_estimator=scale_estimator,
    )


def dequantize_weight_np(quantized: QuantizedWeight) -> np.ndarray:
    return decode_weight_matrix(
        quantized.codes,
        quantized.scales,
        code_bits=quantized.code_bits,
        codebook=quantized.codebook,
    )
