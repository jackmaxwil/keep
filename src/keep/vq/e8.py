from __future__ import annotations

import hashlib
import itertools
from functools import cache
from typing import Literal

import numpy as np


QUIP_SHARP_SOURCE_COMMIT = "1d8f873e9a2a8b86b12bb1064c312c5689b77d98"
QUIP_SHARP_E8P_SOURCE = (
    "https://github.com/Cornell-RelaxML/quip-sharp/blob/"
    f"{QUIP_SHARP_SOURCE_COMMIT}/lib/codebook/latticee8_padded12.py"
)
QUIP_SHARP_E8_1BIT_SOURCE = (
    "https://github.com/Cornell-RelaxML/quip-sharp/blob/"
    f"{QUIP_SHARP_SOURCE_COMMIT}/lib/codebook/latticee8_padded12_rvq3bit.py"
)

CODEWORD_DIM = 8
E8P_SHUFFLE_MAP = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.uint32)

# QuIP# Appendix C.1 / get_norm12(): 29 padding elements, stored in units of 1/2.
E8P_NORM12_PAD_INT2 = np.array(
    [
        [3, 1, 1, 1, 3, 3, 3, 3],
        [1, 3, 1, 1, 3, 3, 3, 3],
        [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3],
        [3, 3, 3, 1, 3, 3, 1, 1],
        [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1],
        [3, 3, 3, 1, 3, 1, 1, 3],
        [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3],
        [3, 3, 1, 3, 3, 3, 1, 1],
        [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1],
        [3, 3, 1, 3, 3, 1, 1, 3],
        [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 3, 1, 1],
        [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1],
        [3, 1, 3, 3, 3, 1, 1, 3],
        [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [1, 3, 3, 3, 3, 3, 1, 1],
        [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1],
        [1, 3, 3, 3, 3, 1, 1, 3],
        [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ],
    dtype=np.int8,
)

# QuIP# get_e81bgrid() appends 15 norm-4 E8 points to make the residual table
# exactly 256 entries. Stored in real-value units, then converted to int2 below.
E8_1BIT_NORM4_PAD_INT2 = np.array(
    [
        [4, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 4],
        [-4, 0, 0, 0, 0, 0, 0, 0],
        [0, -4, 0, 0, 0, 0, 0, 0],
        [0, 0, -4, 0, 0, 0, 0, 0],
        [0, 0, 0, -4, 0, 0, 0, 0],
        [0, 0, 0, 0, -4, 0, 0, 0],
        [0, 0, 0, 0, 0, -4, 0, 0],
        [0, 0, 0, 0, 0, 0, -4, 0],
    ],
    dtype=np.int8,
)


def _sha256_u32(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<u4").tobytes()).hexdigest()


def _pack_int2_rows(values_int2: np.ndarray) -> np.ndarray:
    values = np.asarray(values_int2, dtype=np.int16)
    if values.ndim != 2 or values.shape[1] != CODEWORD_DIM:
        raise ValueError(f"expected [N, {CODEWORD_DIM}] values")

    nibbles = values + 8
    if np.any(nibbles < 0) or np.any(nibbles > 15):
        raise ValueError("packed E8 values must fit signed 4-bit int2 range")

    packed = np.zeros(values.shape[0], dtype=np.uint32)
    for dim in range(CODEWORD_DIM):
        packed |= nibbles[:, dim].astype(np.uint32) << np.uint32(4 * dim)
    return packed


def _unpack_int2_rows(packed: np.ndarray) -> np.ndarray:
    packed_u32 = np.asarray(packed, dtype=np.uint32)
    out = np.empty((*packed_u32.shape, CODEWORD_DIM), dtype=np.int16)
    for dim in range(CODEWORD_DIM):
        out[..., dim] = ((packed_u32 >> np.uint32(4 * dim)) & np.uint32(0xF)).astype(np.int16) - 8
    return out


def _e8p_abs_int2_rows() -> np.ndarray:
    rows: list[tuple[int, ...]] = []
    values = (1, 3, 5, 7)
    for row in itertools.product(values, repeat=CODEWORD_DIM):
        if sum(component * component for component in row) <= 40:
            rows.append(row)

    if len(rows) != 227:
        raise AssertionError(f"expected 227 D8-hat absolute rows, found {len(rows)}")

    return np.concatenate(
        [np.array(rows, dtype=np.int8), E8P_NORM12_PAD_INT2],
        axis=0,
    )


@cache
def e8p_packed_abs_grid() -> np.ndarray:
    """Return the QuIP# E8P packed absolute table, shape [256], dtype uint32.

    This mirrors QuIP# ``get_packed_abs_grid()``: 227 low-norm absolute
    D8-hat rows plus 29 norm-12 padding rows, column-shuffled, parity-fixed,
    and packed as eight 4-bit ``value * 2 + 8`` nibbles.
    """

    cba = _e8p_abs_int2_rows()[:, [0, 2, 4, 6, 1, 3, 5, 7]].astype(np.int16)
    odd_half_sum = (cba.sum(axis=1) % 4) == 2
    cba[odd_half_sum, 7] *= -1
    return _pack_int2_rows(cba)


E8P_PACKED_ABS_SHA256 = _sha256_u32(e8p_packed_abs_grid())


def _e8_1bit_int2_rows() -> np.ndarray:
    integer_rows: list[tuple[int, ...]] = []
    for row in itertools.product((-1, 0, 1), repeat=CODEWORD_DIM):
        if sum(component * component for component in row) <= 2 and sum(row) % 2 == 0:
            integer_rows.append(tuple(2 * component for component in row))

    half_rows: list[tuple[int, ...]] = []
    for row in itertools.product((-1, 1), repeat=CODEWORD_DIM):
        if sum(row) % 4 == 0:
            half_rows.append(row)

    rows = np.concatenate(
        [
            np.array(integer_rows, dtype=np.int8),
            np.array(half_rows, dtype=np.int8),
            E8_1BIT_NORM4_PAD_INT2,
        ],
        axis=0,
    )
    if rows.shape != (256, CODEWORD_DIM):
        raise AssertionError(f"expected E8 1-bit table shape (256, 8), found {rows.shape}")
    return rows


@cache
def e8_1bit_packed() -> np.ndarray:
    """Return QuIP#'s residual 1-bit E8 table packed as uint32[256].

    This is the 8-bit VQ-1.0 storage table used by this project: one byte
    selects one full signed 8D vector. It is distinct from the 16-bit E8P
    storage contract, where a byte selects an absolute row and the other byte
    carries sign/shift state.
    """

    return _pack_int2_rows(_e8_1bit_int2_rows())


E8_1BIT_PACKED_SHA256 = _sha256_u32(e8_1bit_packed())


@cache
def e8_1bit_grid() -> np.ndarray:
    """Return the unpacked QuIP# residual 1-bit E8 table, shape [256, 8]."""

    return _unpack_int2_rows(e8_1bit_packed()).astype(np.float32) * np.float32(0.5)


def decode_e8_1bit(codes: np.ndarray, packed_codebook: np.ndarray | None = None) -> np.ndarray:
    """Decode VQ-1.0 uint8 codewords to float32 vectors with trailing dim 8."""

    table = e8_1bit_packed() if packed_codebook is None else np.asarray(packed_codebook, dtype=np.uint32)
    if table.shape != (256,):
        raise ValueError("E8 1-bit packed codebook must have shape [256]")

    code_array = np.asarray(codes, dtype=np.uint8)
    return _unpack_int2_rows(table[code_array]).astype(np.float32) * np.float32(0.5)


def _sign_parity(signs: np.ndarray) -> np.ndarray:
    parity = np.zeros_like(signs, dtype=np.uint32)
    for bit in range(8):
        parity ^= (signs >> np.uint32(bit)) & np.uint32(1)
    return parity


def decode_e8p(codewords: np.ndarray, packed_abs_grid: np.ndarray | None = None) -> np.ndarray:
    """Decode QuIP# 16-bit E8P codewords using get_full_grid parity behavior."""

    packed_abs = e8p_packed_abs_grid() if packed_abs_grid is None else np.asarray(packed_abs_grid, dtype=np.uint32)
    if packed_abs.shape != (256,):
        raise ValueError("E8P packed absolute grid must have shape [256]")

    codes = np.asarray(codewords, dtype=np.uint16)
    flat = codes.reshape(-1).astype(np.uint32)
    signs = flat & np.uint32(0xFF)
    abs_idx = flat >> np.uint32(8)
    parity = _sign_parity(signs)
    effective_signs = signs ^ parity
    abs_code = packed_abs[abs_idx]

    decoded_int2 = np.empty((flat.shape[0], CODEWORD_DIM), dtype=np.int16)
    for out_dim, packed_dim in enumerate(E8P_SHUFFLE_MAP):
        value = ((abs_code >> (np.uint32(4) * packed_dim)) & np.uint32(0xF)).astype(np.int16) - 8
        sign_bit = ((effective_signs >> packed_dim) & np.uint32(1)).astype(bool)
        value[sign_bit] *= -1
        decoded_int2[:, out_dim] = value

    decoded = decoded_int2.astype(np.float32) * np.float32(0.5)
    decoded += np.where(parity.astype(bool), np.float32(-0.25), np.float32(0.25))[:, None]
    return decoded.reshape((*codes.shape, CODEWORD_DIM))


@cache
def e8p_full_grid() -> np.ndarray:
    """Return the full QuIP# E8P grid, shape [65536, 8]."""

    return decode_e8p(np.arange(1 << 16, dtype=np.uint16))


@cache
def e8p_abs_grid() -> np.ndarray:
    """Return E8P packed-abs rows in decoded output-dimension order.

    QuIP# parity-fixes one stored component before packing, so these rows are
    the signed base rows used by ``decode_e8p`` rather than pure magnitudes.
    """

    return _unpack_int2_rows(e8p_packed_abs_grid()).astype(np.float32)[:, E8P_SHUFFLE_MAP] * np.float32(0.5)


def _nearest_codebook_indices(
    vectors: np.ndarray,
    codebook: np.ndarray,
    *,
    chunk_size: int = 8192,
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[-1] != CODEWORD_DIM:
        raise ValueError(f"vectors must have trailing dimension {CODEWORD_DIM}")

    flat = values.reshape(-1, CODEWORD_DIM)
    table = np.asarray(codebook, dtype=np.float32)
    best_idx = np.zeros(flat.shape[0], dtype=np.int64)
    best_dist = np.full(flat.shape[0], np.inf, dtype=np.float32)
    vector_norm = np.sum(flat * flat, axis=1)

    for start in range(0, table.shape[0], chunk_size):
        chunk = table[start : start + chunk_size]
        distances = vector_norm[:, None] + np.sum(chunk * chunk, axis=1)[None, :] - 2 * flat @ chunk.T
        local_idx = np.argmin(distances, axis=1)
        local_dist = distances[np.arange(flat.shape[0]), local_idx]
        improved = local_dist < best_dist
        best_dist[improved] = local_dist[improved]
        best_idx[improved] = start + local_idx[improved]

    return best_idx.reshape(values.shape[:-1])


def encode_e8_1bit_rtn(vectors: np.ndarray) -> np.ndarray:
    """Round vectors to nearest entries in the 256-entry E8 1-bit table."""

    return _nearest_codebook_indices(vectors, e8_1bit_grid()).astype(np.uint8)


def encode_e8p_rtn(vectors: np.ndarray, *, chunk_size: int = 8192) -> np.ndarray:
    """Round vectors to nearest entries in the 65,536-entry E8P grid.

    E8P codewords factor into 256 absolute rows, an even-parity sign pattern,
    and a global +/-0.25 shift. Exploiting that structure keeps real artifact
    conversion on a 256-row search instead of a direct 65,536-row nearest-table
    scan, while preserving the same RTN objective as the full grid.
    """

    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[-1] != CODEWORD_DIM:
        raise ValueError(f"vectors must have trailing dimension {CODEWORD_DIM}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    flat = values.reshape(-1, CODEWORD_DIM)
    base_rows = e8p_abs_grid().astype(np.float32)
    abs_rows = np.abs(base_rows)
    abs_norm = np.sum(abs_rows * abs_rows, axis=1)
    packed_dim_by_out_dim = E8P_SHUFFLE_MAP.astype(np.uint16)
    base_signs = np.zeros(abs_rows.shape[0], dtype=np.uint16)
    base_negative = base_rows < 0
    for out_dim, packed_dim in enumerate(packed_dim_by_out_dim):
        base_signs |= base_negative[:, out_dim].astype(np.uint16) << packed_dim

    best_codes = np.zeros(flat.shape[0], dtype=np.uint16)

    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        local_best_dist = np.full(chunk.shape[0], np.inf, dtype=np.float32)
        local_best_codes = np.zeros(chunk.shape[0], dtype=np.uint16)

        for parity, shift in ((0, np.float32(0.25)), (1, np.float32(-0.25))):
            target = chunk - shift
            abs_target = np.abs(target)
            distances = (
                np.sum(abs_target * abs_target, axis=1)[:, None]
                + abs_norm[None, :]
                - 2 * abs_target @ abs_rows.T
            )

            negative = target < 0
            target_signs = np.zeros(chunk.shape[0], dtype=np.uint16)
            for out_dim, packed_dim in enumerate(packed_dim_by_out_dim):
                target_signs |= negative[:, out_dim].astype(np.uint16) << packed_dim

            effective_signs = (target_signs[:, None] ^ base_signs[None, :]).astype(np.uint16)
            sign_parity = _sign_parity(effective_signs.astype(np.uint32)).astype(bool)
            flip_dim = None
            if np.any(sign_parity):
                flip_penalty = 4 * abs_target[:, None, :] * abs_rows[None, :, :]
                flip_dim = np.argmin(flip_penalty, axis=2).astype(np.uint8)
                correction = np.min(flip_penalty, axis=2)
                distances = np.where(sign_parity, distances + correction, distances)

            abs_idx = np.argmin(distances, axis=1).astype(np.uint16)
            row_dist = distances[np.arange(chunk.shape[0]), abs_idx]
            candidate_signs = effective_signs[np.arange(chunk.shape[0]), abs_idx].copy()
            if flip_dim is not None and np.any(sign_parity):
                selected_flip_dim = flip_dim[np.arange(chunk.shape[0]), abs_idx]
                flip_masks = (np.uint16(1) << packed_dim_by_out_dim[selected_flip_dim]).astype(np.uint16)
                selected_needs_flip = sign_parity[np.arange(chunk.shape[0]), abs_idx]
                candidate_signs = np.where(
                    selected_needs_flip,
                    candidate_signs ^ flip_masks,
                    candidate_signs,
                ).astype(np.uint16)

            stored_signs = (candidate_signs ^ np.uint16(parity)).astype(np.uint16)
            codes = ((abs_idx.astype(np.uint16) << np.uint16(8)) | stored_signs).astype(np.uint16)
            improved = row_dist < local_best_dist
            local_best_dist[improved] = row_dist[improved]
            local_best_codes[improved] = codes[improved]

        best_codes[start : start + chunk.shape[0]] = local_best_codes

    return best_codes.reshape(values.shape[:-1])


def _encode_e8p_rtn_per_row_diagonal_hessian(
    vectors: np.ndarray, diagonal: np.ndarray, *, chunk_size: int
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[-1] != CODEWORD_DIM:
        raise ValueError(f"vectors must have trailing dimension {CODEWORD_DIM}")
    flat = values.reshape(-1, CODEWORD_DIM)
    w = np.asarray(diagonal, dtype=np.float32)
    if (
        w.shape != flat.shape
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or not np.all(np.any(w > 0, axis=1))
    ):
        raise ValueError(
            "per-row diagonal must be finite, non-negative, shape (N, 8), "
            "and nonzero per row"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    base_rows = e8p_abs_grid().astype(np.float32)
    abs_rows = np.abs(base_rows)
    abs_rows_squared = abs_rows * abs_rows
    packed = E8P_SHUFFLE_MAP.astype(np.uint16)
    base_signs = np.zeros(abs_rows.shape[0], dtype=np.uint16)
    base_negative = base_rows < 0
    for out_dim, packed_dim in enumerate(packed):
        base_signs |= base_negative[:, out_dim].astype(np.uint16) << packed_dim

    best_codes = np.zeros(flat.shape[0], dtype=np.uint16)
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start:start + chunk_size]
        chunk_w = w[start:start + chunk.shape[0]]
        abs_norm_w = np.sum(
            abs_rows_squared[None, :, :] * chunk_w[:, None, :], axis=2
        )
        local_best_dist = np.full(chunk.shape[0], np.inf, dtype=np.float32)
        local_best_codes = np.zeros(chunk.shape[0], dtype=np.uint16)
        for parity, shift in ((0, np.float32(0.25)), (1, np.float32(-0.25))):
            target = chunk - shift
            abs_target = np.abs(target)
            weighted_abs_target = abs_target * chunk_w
            vec_norm_w = np.sum(abs_target * weighted_abs_target, axis=1)
            distances = (
                vec_norm_w[:, None] + abs_norm_w
                - 2.0 * weighted_abs_target @ abs_rows.T
            )
            negative = target < 0
            target_signs = np.zeros(chunk.shape[0], dtype=np.uint16)
            for out_dim, packed_dim in enumerate(packed):
                target_signs |= negative[:, out_dim].astype(np.uint16) << packed_dim
            effective_signs = (target_signs[:, None] ^ base_signs[None, :]).astype(np.uint16)
            sign_parity = _sign_parity(effective_signs.astype(np.uint32)).astype(bool)
            flip_dim = None
            if np.any(sign_parity):
                flip_penalty = 4 * weighted_abs_target[:, None, :] * abs_rows[None, :, :]
                correction = np.min(flip_penalty, axis=2)
                flip_masks = np.uint16(1) << packed
                flipped_signs = effective_signs[:, :, None] ^ flip_masks[None, None, :]
                tied_for_minimum = flip_penalty == correction[:, :, None]
                tie_codes = np.where(tied_for_minimum, flipped_signs, np.iinfo(np.uint16).max)
                flip_dim = np.argmin(tie_codes, axis=2).astype(np.uint8)
                distances = np.where(sign_parity, distances + correction, distances)
            abs_idx = np.argmin(distances, axis=1).astype(np.uint16)
            row_dist = distances[np.arange(chunk.shape[0]), abs_idx]
            candidate_signs = effective_signs[np.arange(chunk.shape[0]), abs_idx].copy()
            if flip_dim is not None and np.any(sign_parity):
                selected_flip_dim = flip_dim[np.arange(chunk.shape[0]), abs_idx]
                flip_masks = (np.uint16(1) << packed[selected_flip_dim]).astype(np.uint16)
                selected_needs_flip = sign_parity[np.arange(chunk.shape[0]), abs_idx]
                candidate_signs = np.where(
                    selected_needs_flip, candidate_signs ^ flip_masks, candidate_signs
                ).astype(np.uint16)
            stored_signs = (candidate_signs ^ np.uint16(parity)).astype(np.uint16)
            codes = ((abs_idx.astype(np.uint16) << np.uint16(8)) | stored_signs).astype(np.uint16)
            better = row_dist < local_best_dist
            tie = row_dist == local_best_dist
            lower_code = codes < local_best_codes
            take = better | (tie & lower_code)
            local_best_dist = np.where(take, row_dist, local_best_dist)
            local_best_codes = np.where(take, codes, local_best_codes).astype(np.uint16)
        best_codes[start:start + chunk.shape[0]] = local_best_codes
    return best_codes.reshape(values.shape[:-1])


def encode_e8p_rtn_diagonal_hessian(
    vectors: np.ndarray, diagonal: np.ndarray, *, chunk_size: int = 65536
) -> np.ndarray:
    """Weighted 256-row factored E8P search under a shared or per-row metric.

    Byte-identical to an exhaustive weighted scan of e8p_full_grid(): each wᵢ≥0
    keeps the per-dim optimal sign = sign(targetᵢ), so the weighted distance is
    Σᵢ wᵢ(|targetᵢ| − abs_rowᵢ)², and the even-parity fix flips the dim of least
    weighted penalty 4·wᵢ·|targetᵢ|·abs_rowᵢ.
    """
    if np.ndim(diagonal) == 1:
        pass
    else:
        return _encode_e8p_rtn_per_row_diagonal_hessian(
            vectors, diagonal, chunk_size=chunk_size
        )

    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[-1] != CODEWORD_DIM:
        raise ValueError(f"vectors must have trailing dimension {CODEWORD_DIM}")
    w = np.asarray(diagonal, dtype=np.float32)
    if w.shape != (CODEWORD_DIM,) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("diagonal must be finite, non-negative, shape (8,), and nonzero")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    flat = values.reshape(-1, CODEWORD_DIM)
    base_rows = e8p_abs_grid().astype(np.float32)
    abs_rows = np.abs(base_rows)
    abs_norm_w = (abs_rows * abs_rows) @ w
    packed = E8P_SHUFFLE_MAP.astype(np.uint16)
    base_signs = np.zeros(abs_rows.shape[0], dtype=np.uint16)
    base_negative = base_rows < 0
    for out_dim, packed_dim in enumerate(packed):
        base_signs |= base_negative[:, out_dim].astype(np.uint16) << packed_dim

    best_codes = np.zeros(flat.shape[0], dtype=np.uint16)
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start:start + chunk_size]
        local_best_dist = np.full(chunk.shape[0], np.inf, dtype=np.float32)
        local_best_codes = np.zeros(chunk.shape[0], dtype=np.uint16)
        for parity, shift in ((0, np.float32(0.25)), (1, np.float32(-0.25))):
            target = chunk - shift
            abs_target = np.abs(target)
            vec_norm_w = (abs_target * abs_target) @ w
            distances = (
                vec_norm_w[:, None] + abs_norm_w[None, :]
                - 2.0 * (abs_target * w[None, :]) @ abs_rows.T
            )
            negative = target < 0
            target_signs = np.zeros(chunk.shape[0], dtype=np.uint16)
            for out_dim, packed_dim in enumerate(packed):
                target_signs |= negative[:, out_dim].astype(np.uint16) << packed_dim
            effective_signs = (target_signs[:, None] ^ base_signs[None, :]).astype(np.uint16)
            sign_parity = _sign_parity(effective_signs.astype(np.uint32)).astype(bool)
            flip_dim = None
            if np.any(sign_parity):
                flip_penalty = 4 * (abs_target * w[None, :])[:, None, :] * abs_rows[None, :, :]
                correction = np.min(flip_penalty, axis=2)
                flip_masks = np.uint16(1) << packed
                flipped_signs = effective_signs[:, :, None] ^ flip_masks[None, None, :]
                tied_for_minimum = flip_penalty == correction[:, :, None]
                tie_codes = np.where(tied_for_minimum, flipped_signs, np.iinfo(np.uint16).max)
                flip_dim = np.argmin(tie_codes, axis=2).astype(np.uint8)
                distances = np.where(sign_parity, distances + correction, distances)
            abs_idx = np.argmin(distances, axis=1).astype(np.uint16)
            row_dist = distances[np.arange(chunk.shape[0]), abs_idx]
            candidate_signs = effective_signs[np.arange(chunk.shape[0]), abs_idx].copy()
            if flip_dim is not None and np.any(sign_parity):
                selected_flip_dim = flip_dim[np.arange(chunk.shape[0]), abs_idx]
                flip_masks = (np.uint16(1) << packed[selected_flip_dim]).astype(np.uint16)
                selected_needs_flip = sign_parity[np.arange(chunk.shape[0]), abs_idx]
                candidate_signs = np.where(
                    selected_needs_flip, candidate_signs ^ flip_masks, candidate_signs
                ).astype(np.uint16)
            stored_signs = (candidate_signs ^ np.uint16(parity)).astype(np.uint16)
            codes = ((abs_idx.astype(np.uint16) << np.uint16(8)) | stored_signs).astype(np.uint16)
            # replace the `improved` block with tie-aware selection:
            better = row_dist < local_best_dist
            tie = row_dist == local_best_dist
            lower_code = codes < local_best_codes
            take = better | (tie & lower_code)
            local_best_dist = np.where(take, row_dist, local_best_dist)
            local_best_codes = np.where(take, codes, local_best_codes).astype(np.uint16)
        best_codes[start:start + chunk.shape[0]] = local_best_codes
    return best_codes.reshape(values.shape[:-1])


def decode_weight_matrix(
    codes: np.ndarray,
    scales: np.ndarray | None = None,
    *,
    code_bits: Literal[8, 16] = 8,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a matrix of 8D VQ codewords to row-major float32 weights.

    ``codes`` is shaped ``[..., in_dim / 8]``. If ``scales`` is provided, its
    trailing dimension partitions the codeword axis into equal-sized groups.
    """

    if code_bits == 8:
        decoded = decode_e8_1bit(codes, codebook)
    elif code_bits == 16:
        decoded = decode_e8p(codes, codebook)
    else:
        raise ValueError("code_bits must be 8 or 16")

    if scales is not None:
        scale_array = np.asarray(scales, dtype=np.float32)
        codewords = decoded.shape[-2]
        if codewords % scale_array.shape[-1] != 0:
            raise ValueError("codeword count must be divisible by scale count")
        words_per_scale = codewords // scale_array.shape[-1]
        expanded = np.repeat(scale_array, words_per_scale, axis=-1)[..., None]
        decoded = decoded * expanded

    return decoded.reshape((*decoded.shape[:-2], decoded.shape[-2] * CODEWORD_DIM))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(left @ right / denom)
