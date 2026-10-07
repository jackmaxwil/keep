from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np

from keep.vq.e8 import CODEWORD_DIM, decode_e8_1bit, decode_e8p


@dataclass(frozen=True)
class RotationGain:
    unrotated_objective: float
    rotated_objective: float
    objective_gain: float

    @property
    def relative_gain(self) -> float:
        if self.unrotated_objective == 0.0:
            return 0.0
        return self.objective_gain / self.unrotated_objective


@dataclass(frozen=True)
class ProjectionRotationSelection:
    selected: bool
    selected_signs: np.ndarray | None
    unrotated_objective: float
    rotated_objective: float
    objective_gain: float

    @property
    def keep_rotation(self) -> bool:
        return self.selected


def _validate_power_of_two(dim: int) -> None:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("RHT dimension must be a positive power of two")


def _validate_signs(signs: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(signs, dtype=np.int8)
    if values.shape != (dim,):
        raise ValueError(f"signs must have shape ({dim},), found {values.shape}")
    if not np.all((values == 1) | (values == -1)):
        raise ValueError("signs values must be -1 or 1")
    return values


def _hadamard(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    dim = source.shape[-1]
    _validate_power_of_two(dim)
    out = source.copy()
    width = 1
    while width < dim:
        shaped = out.reshape((*out.shape[:-1], dim // (2 * width), 2, width))
        left = shaped[..., 0, :].copy()
        right = shaped[..., 1, :].copy()
        shaped[..., 0, :] = left + right
        shaped[..., 1, :] = left - right
        out = shaped.reshape((*source.shape[:-1], dim))
        width *= 2
    return out * np.float32(1.0 / math.sqrt(dim))


def generate_rht_signs(dim: int, seed: int | str) -> np.ndarray:
    """Generate the runtime-compatible deterministic RHT sign vector."""

    _validate_power_of_two(dim)
    bits = np.frombuffer(
        hashlib.shake_256(str(seed).encode("utf-8")).digest((dim + 7) // 8),
        dtype=np.uint8,
    )
    unpacked = np.unpackbits(bits, bitorder="little")[:dim]
    return np.where(unpacked == 0, np.int8(-1), np.int8(1))


def rotate_weight_rht(weight: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Return ``W @ (D @ H)`` matching the activation-side runtime RHT."""

    values = np.asarray(weight, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in], found {values.shape}")
    sign_values = _validate_signs(signs, values.shape[1]).astype(np.float32)
    return _hadamard(values * sign_values)


def _inverse_rht(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return _hadamard(values) * np.asarray(signs, dtype=np.float32)


def _validate_inputs(
    weight: np.ndarray,
    diagonal_importance: np.ndarray,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(weight, dtype=np.float32)
    importance = np.asarray(diagonal_importance, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % CODEWORD_DIM:
        raise ValueError("weight must have shape [out, in] with in divisible by 8")
    if importance.shape != (values.shape[1],):
        raise ValueError(f"diagonal_importance must have shape ({values.shape[1]},), found {importance.shape}")
    if not np.isfinite(values).all():
        raise ValueError("weight must be finite")
    if not np.isfinite(importance).all() or np.any(importance < 0) or not np.any(importance > 0):
        raise ValueError("diagonal_importance must be finite, non-negative, and contain a positive value")
    if group_size <= 0 or group_size % CODEWORD_DIM or values.shape[1] % group_size:
        raise ValueError("group_size must be positive, divisible by 8, and divide the input dimension")
    return values, importance


def _decoded_codebook(codebook: np.ndarray, code_bits: int) -> np.ndarray:
    supplied = np.asarray(codebook)
    if supplied.ndim == 2:
        if supplied.shape[1] != CODEWORD_DIM or not np.isfinite(supplied).all():
            raise ValueError("decoded codebook must be finite with shape [codes, 8]")
        return supplied.astype(np.float32, copy=False)
    if supplied.shape != (256,):
        raise ValueError("packed E8-family codebook must have shape (256,)")
    if code_bits == 8:
        return decode_e8_1bit(np.arange(256, dtype=np.uint8), supplied).astype(np.float32)
    if code_bits == 16:
        return decode_e8p(np.arange(1 << 16, dtype=np.uint16), supplied).astype(np.float32)
    raise ValueError("code_bits must be 8 or 16")


def _nearest_diagonal_encoder():
    """Load the existing NumPy RTN encoder without importing MLX-backed quant.__init__."""

    module_name = "_mlx_vq_rotation_search_rtn_numpy"
    loaded = sys.modules.get(module_name)
    if loaded is None:
        module_path = Path(__file__).parents[1] / "quant/rtn.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load NumPy RTN encoders from {module_path}")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        spec.loader.exec_module(loaded)
    return loaded.nearest_codebook_indices_diagonal_hessian


def _quantize_importance_aware(
    weight: np.ndarray,
    importance: np.ndarray,
    codebook: np.ndarray,
    *,
    group_size: int,
    code_bits: int,
) -> np.ndarray:
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    table = _decoded_codebook(codebook, code_bits)
    divisor = float(np.max(np.abs(table)))
    if divisor <= 0.0:
        raise ValueError("codebook must contain a non-zero value")

    out_dim, in_dim = weight.shape
    group_count = in_dim // group_size
    words_per_group = group_size // CODEWORD_DIM
    grouped = weight.reshape(out_dim, group_count, words_per_group, CODEWORD_DIM)
    grouped_importance = importance.reshape(group_count, words_per_group, CODEWORD_DIM)
    scales = np.max(np.abs(grouped), axis=(2, 3)) / np.float32(divisor)
    codes = np.empty((out_dim, group_count, words_per_group), dtype=np.int64)
    nearest = _nearest_diagonal_encoder()

    for _ in range(3):
        safe_scales = np.where(scales > 0, scales, np.float32(1.0))
        normalized = grouped / safe_scales[:, :, None, None]
        per_row_importance = np.broadcast_to(grouped_importance[None, ...], normalized.shape)
        flat_vectors = normalized.reshape(-1, CODEWORD_DIM)
        flat_importance = per_row_importance.reshape(-1, CODEWORD_DIM)
        flat_codes = np.empty(flat_vectors.shape[0], dtype=np.int64)
        unique_importance, inverse = np.unique(flat_importance, axis=0, return_inverse=True)
        for importance_index, shared_importance in enumerate(unique_importance):
            rows = np.flatnonzero(inverse == importance_index)
            flat_codes[rows] = nearest(
                flat_vectors[rows],
                shared_importance,
                codebook=table,
                index_dtype=np.dtype(np.uint8 if table.shape[0] <= 256 else np.uint16),
                vector_chunk_size=4096,
                codebook_chunk_size=8192,
            )
        codes[...] = flat_codes.reshape(codes.shape)
        decoded = table[codes]
        numerator = np.sum(grouped_importance[None, ...] * grouped * decoded, axis=(2, 3), dtype=np.float64)
        denominator = np.sum(grouped_importance[None, ...] * decoded * decoded, axis=(2, 3), dtype=np.float64)
        scales = np.divide(
            numerator,
            denominator,
            out=scales.astype(np.float64),
            where=(denominator > 0) & (numerator > 0),
        ).astype(np.float32)

    return (table[codes] * scales[:, :, None, None]).reshape(weight.shape).astype(np.float32)


def _weighted_objective(reference: np.ndarray, candidate: np.ndarray, importance: np.ndarray) -> float:
    residual = reference.astype(np.float64) - candidate.astype(np.float64)
    return float(np.sum(residual * residual * importance.astype(np.float64)))


def evaluate_rotation_gain(
    weight: np.ndarray,
    diagonal_importance: np.ndarray,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int,
    signs: np.ndarray,
) -> RotationGain:
    """Compare diagonal-Hessian RTN objectives before and after an RHT."""

    values, importance = _validate_inputs(weight, diagonal_importance, group_size)
    sign_values = _validate_signs(signs, values.shape[1])
    baseline = _quantize_importance_aware(
        values, importance, codebook, group_size=group_size, code_bits=code_bits
    )
    rotated_source = rotate_weight_rht(values, sign_values)
    # diag(R.T @ diag(h) @ R) is mean(h) for a normalized Hadamard R.
    rotated_importance = np.full_like(importance, np.mean(importance, dtype=np.float64))
    rotated_quantized = _quantize_importance_aware(
        rotated_source,
        rotated_importance,
        codebook,
        group_size=group_size,
        code_bits=code_bits,
    )
    equivalent_original = _inverse_rht(rotated_quantized, sign_values)
    unrotated_objective = _weighted_objective(values, baseline, importance)
    rotated_objective = _weighted_objective(values, equivalent_original, importance)
    return RotationGain(
        unrotated_objective=unrotated_objective,
        rotated_objective=rotated_objective,
        objective_gain=unrotated_objective - rotated_objective,
    )


def select_projection_rotation(
    weight: np.ndarray,
    diagonal_importance: np.ndarray,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int,
    signs: np.ndarray,
) -> ProjectionRotationSelection:
    """Keep a projection RHT only for a strict objective improvement."""

    gain = evaluate_rotation_gain(
        weight, diagonal_importance, codebook, group_size, code_bits, signs
    )
    selected = gain.rotated_objective < gain.unrotated_objective
    return ProjectionRotationSelection(
        selected=selected,
        selected_signs=np.asarray(signs, dtype=np.int8).copy() if selected else None,
        unrotated_objective=gain.unrotated_objective,
        rotated_objective=gain.rotated_objective,
        objective_gain=gain.objective_gain,
    )


__all__ = [
    "ProjectionRotationSelection",
    "RotationGain",
    "evaluate_rotation_gain",
    "generate_rht_signs",
    "rotate_weight_rht",
    "select_projection_rotation",
]
