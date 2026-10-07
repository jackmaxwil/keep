from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keep.vq.e8 import CODEWORD_DIM, e8_1bit_grid, encode_e8p_rtn
from keep.quality.kronecker_hessian import (
    blockldlq_hin_only_reassign_codes,
    hin_weighted_error,
)


LEVER_NAME = "selection_hin_blockldlq_feedback_fp32_v1"


@dataclass(frozen=True)
class LDLQStats:
    group_size: int
    code_bits: int
    row_count: int
    codewords_per_row: int
    code_count: int
    changed_code_count: int
    changed_code_fraction: float
    seed_hin_weighted_error: float
    ldlq_hin_weighted_error: float
    refined_hin_weighted_error: float | None
    hin_weighted_error: float
    hin_weighted_error_ratio: float
    ldlq_accepted: bool
    sweeps: int


@dataclass(frozen=True)
class LDLQReassignedWeight:
    codes: np.ndarray
    weight: np.ndarray
    stats: LDLQStats


def _block_ldl_feedback(h_in: np.ndarray) -> np.ndarray:
    """Return the strict lower transpose of the Babai feedback factor.

    Backward block elimination gives ``H = U D U.T`` with unit-block-upper
    ``U``.  Local nearest-plane errors obey ``E U = Z``, hence decoded errors
    are ``E = Z inv(U)`` and the left-to-right feedback is ``inv(U) - I``.
    Its transpose is the conventional strictly-block-lower feedback factor.
    """

    h = np.asarray(h_in, dtype=np.float64)
    dim = h.shape[0]
    blocks = dim // CODEWORD_DIM
    upper = np.eye(dim, dtype=np.float64)
    diagonal: list[np.ndarray | None] = [None] * blocks

    for block_j in range(blocks - 1, -1, -1):
        j0 = block_j * CODEWORD_DIM
        j1 = j0 + CODEWORD_DIM
        d_j = h[j0:j1, j0:j1].copy()
        for block_k in range(block_j + 1, blocks):
            k0 = block_k * CODEWORD_DIM
            k1 = k0 + CODEWORD_DIM
            d_k = diagonal[block_k]
            assert d_k is not None
            u_jk = upper[j0:j1, k0:k1]
            d_j -= u_jk @ d_k @ u_jk.T
        d_j = 0.5 * (d_j + d_j.T)
        try:
            np.linalg.cholesky(d_j)
        except np.linalg.LinAlgError as exc:
            raise ValueError("h_in must be positive definite for block LDLQ") from exc
        diagonal[block_j] = d_j

        for block_i in range(block_j):
            i0 = block_i * CODEWORD_DIM
            i1 = i0 + CODEWORD_DIM
            residual = h[i0:i1, j0:j1].copy()
            for block_k in range(block_j + 1, blocks):
                k0 = block_k * CODEWORD_DIM
                k1 = k0 + CODEWORD_DIM
                d_k = diagonal[block_k]
                assert d_k is not None
                residual -= upper[i0:i1, k0:k1] @ d_k @ upper[j0:j1, k0:k1].T
            upper[i0:i1, j0:j1] = np.linalg.solve(d_j, residual.T).T

    inverse_upper = np.linalg.solve(upper, np.eye(dim, dtype=np.float64))
    lower_feedback = np.triu(inverse_upper, k=CODEWORD_DIM).T
    return lower_feedback.astype(np.float32)


def _encode_8bit(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    flat = values.reshape(-1, CODEWORD_DIM)
    table = e8_1bit_grid().astype(np.float32)
    vector_norm = np.sum(flat * flat, axis=1)
    distances = vector_norm[:, None] + np.sum(table * table, axis=1)[None, :] - 2 * flat @ table.T
    return np.argmin(distances, axis=1).reshape(values.shape[:-1]).astype(np.uint8)


def _encode(vectors: np.ndarray, *, code_bits: int) -> np.ndarray:
    if code_bits == 8:
        return _encode_8bit(vectors)
    return encode_e8p_rtn(vectors).astype(np.uint16, copy=False)


def _decode(codes: np.ndarray, scales: np.ndarray, codebook: np.ndarray, *, group_size: int) -> np.ndarray:
    words_per_scale = group_size // CODEWORD_DIM
    expanded_scales = np.repeat(scales, words_per_scale, axis=1)[..., None]
    decoded = codebook[codes.astype(np.int64)] * expanded_scales
    return decoded.reshape(codes.shape[0], codes.shape[1] * CODEWORD_DIM).astype(np.float32)


def _validate_inputs(
    source_weight: np.ndarray,
    scales: np.ndarray,
    h_in: np.ndarray,
    codebook: np.ndarray,
    *,
    group_size: int,
    code_bits: int,
    sweeps: int,
    row_indices: np.ndarray | tuple[int, ...] | list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if group_size <= 0 or group_size % CODEWORD_DIM != 0:
        raise ValueError("group_size must be a positive multiple of 8")
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if not isinstance(sweeps, (int, np.integer)) or sweeps < 0:
        raise ValueError("sweeps must be a non-negative integer")

    source = np.asarray(source_weight, dtype=np.float32)
    scale_array = np.asarray(scales, dtype=np.float32)
    h = np.asarray(h_in, dtype=np.float64)
    table = np.asarray(codebook, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"source_weight must be 2D [out, in], found {source.shape}")
    if source.shape[1] == 0 or source.shape[1] % group_size != 0:
        raise ValueError("source input dimension must be a positive multiple of group_size")
    if h.shape != (source.shape[1], source.shape[1]):
        raise ValueError(f"h_in must have shape ({source.shape[1]}, {source.shape[1]}), found {h.shape}")
    if scale_array.shape != (source.shape[0], source.shape[1] // group_size):
        raise ValueError("scales shape does not match source weight")
    expected_entries = 1 << code_bits
    if table.shape != (expected_entries, CODEWORD_DIM):
        raise ValueError(
            f"codebook must have shape ({expected_entries}, {CODEWORD_DIM}) for code_bits={code_bits}, "
            f"found {table.shape}"
        )
    if not np.isfinite(source).all() or not np.isfinite(scale_array).all():
        raise ValueError("source_weight and scales must be finite")
    if not np.isfinite(h).all() or not np.isfinite(table).all():
        raise ValueError("h_in and codebook must be finite")
    if np.any(scale_array <= 0.0):
        raise ValueError("scales must be positive")

    if row_indices is None:
        selected_rows = np.arange(source.shape[0], dtype=np.int64)
    else:
        selected_rows = np.asarray(row_indices, dtype=np.int64)
    if selected_rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    if selected_rows.size and (
        int(np.min(selected_rows)) < 0 or int(np.max(selected_rows)) >= source.shape[0]
    ):
        raise ValueError("row_indices contain rows outside source_weight")
    return source, scale_array, h, table, selected_rows


def ldlq_reassign_codes(
    source_weight: np.ndarray,
    scales: np.ndarray,
    h_in: np.ndarray,
    *,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int,
    sweeps: int = 0,
    row_indices: np.ndarray | tuple[int, ...] | list[int] | None = None,
) -> LDLQReassignedWeight:
    """Reassign E8 codewords with forward fp32 block-LDLQ error feedback."""

    source, scale_array, h, table, selected_rows = _validate_inputs(
        source_weight,
        scales,
        h_in,
        codebook,
        group_size=group_size,
        code_bits=code_bits,
        sweeps=sweeps,
        row_indices=row_indices,
    )
    selected_source = source[selected_rows]
    selected_scales = scale_array[selected_rows]
    rows = selected_source.shape[0]
    codewords = selected_source.shape[1] // CODEWORD_DIM
    words_per_scale = group_size // CODEWORD_DIM

    normalized_source = selected_source.reshape(rows, codewords, CODEWORD_DIM) / np.repeat(
        selected_scales, words_per_scale, axis=1
    )[..., None]
    seed_codes = _encode(normalized_source, code_bits=code_bits)
    seed_weight = _decode(seed_codes, selected_scales, table, group_size=group_size)
    seed_error = hin_weighted_error(seed_weight, selected_source, h)

    lower_feedback = _block_ldl_feedback(h)
    upper_feedback = lower_feedback.T
    delta = np.zeros_like(selected_source, dtype=np.float32)
    ldlq_codes = np.empty((rows, codewords), dtype=np.uint8 if code_bits == 8 else np.uint16)
    ldlq_weight = np.empty_like(selected_source, dtype=np.float32)

    for codeword in range(codewords):
        start = codeword * CODEWORD_DIM
        end = start + CODEWORD_DIM
        feedback = delta[:, :start] @ upper_feedback[:start, start:end]
        adjusted_block = selected_source[:, start:end] + feedback
        scale = selected_scales[:, codeword // words_per_scale, None]
        codes_j = _encode(adjusted_block / scale, code_bits=code_bits)
        decoded_j = table[codes_j.astype(np.int64)] * scale
        ldlq_codes[:, codeword] = codes_j
        ldlq_weight[:, start:end] = decoded_j.astype(np.float32)
        delta[:, start:end] = ldlq_weight[:, start:end] - adjusted_block

    raw_ldlq_error = hin_weighted_error(ldlq_weight, selected_source, h)
    ldlq_accepted = raw_ldlq_error <= seed_error
    if ldlq_accepted:
        accepted_codes = ldlq_codes
        accepted_weight = ldlq_weight
        accepted_error = raw_ldlq_error
    else:
        accepted_codes = seed_codes.copy()
        accepted_weight = seed_weight.copy()
        accepted_error = seed_error

    refined_error: float | None = None
    final_codes = accepted_codes
    final_weight = accepted_weight
    final_error = accepted_error
    if sweeps:
        refined = blockldlq_hin_only_reassign_codes(
            selected_source,
            accepted_codes,
            selected_scales,
            h,
            codebook=table,
            group_size=group_size,
            code_bits=code_bits,
            sweeps=int(sweeps),
        )
        refined_error = hin_weighted_error(refined.weight, selected_source, h)
        if refined_error <= accepted_error:
            final_codes = refined.codes
            final_weight = refined.weight
            final_error = refined_error

    changed = int(np.count_nonzero(final_codes != seed_codes))
    code_count = int(final_codes.size)
    stats = LDLQStats(
        group_size=int(group_size),
        code_bits=int(code_bits),
        row_count=int(rows),
        codewords_per_row=int(codewords),
        code_count=code_count,
        changed_code_count=changed,
        changed_code_fraction=float(changed / code_count) if code_count else 0.0,
        seed_hin_weighted_error=seed_error,
        ldlq_hin_weighted_error=raw_ldlq_error,
        refined_hin_weighted_error=refined_error,
        hin_weighted_error=final_error,
        hin_weighted_error_ratio=float(final_error / seed_error) if abs(seed_error) > 1.0e-12 else 1.0,
        ldlq_accepted=ldlq_accepted,
        sweeps=int(sweeps),
    )
    return LDLQReassignedWeight(codes=final_codes, weight=final_weight, stats=stats)


__all__ = ["LEVER_NAME", "LDLQReassignedWeight", "LDLQStats", "ldlq_reassign_codes"]
