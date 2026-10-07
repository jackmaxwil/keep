from __future__ import annotations

import hashlib
import math

import mlx.core as mx
import numpy as np


def _validate_power_of_two_dim(dim: int) -> None:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("RHT dimension must be a positive power of two")


def _validate_signs_np(signs: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(signs, dtype=np.int8)
    if values.shape != (dim,):
        raise ValueError(f"rht_signs must have shape ({dim},), found {values.shape}")
    if not np.all((values == 1) | (values == -1)):
        raise ValueError("rht_signs values must be -1 or 1")
    return values


def deterministic_rht_signs(dim: int, *, seed: int | str) -> np.ndarray:
    """Return deterministic +/-1 signs for a randomized Hadamard transform."""

    _validate_power_of_two_dim(dim)
    seed_bytes = str(seed).encode("utf-8")
    bits = np.frombuffer(hashlib.shake_256(seed_bytes).digest((dim + 7) // 8), dtype=np.uint8)
    unpacked = np.unpackbits(bits, bitorder="little")[:dim]
    return np.where(unpacked == 0, np.int8(-1), np.int8(1))


def _hadamard_np(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    dim = values.shape[-1]
    _validate_power_of_two_dim(dim)
    out = values.copy()
    width = 1
    while width < dim:
        out = out.reshape((*out.shape[:-1], dim // (2 * width), 2, width))
        left = out[..., 0, :].copy()
        right = out[..., 1, :].copy()
        out[..., 0, :] = left + right
        out[..., 1, :] = left - right
        out = out.reshape((*values.shape[:-1], dim))
        width *= 2
    return out * np.float32(1.0 / math.sqrt(dim))


def apply_rht_np(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply row-vector RHT: ``x @ (D @ H)`` over the trailing dimension."""

    values = np.asarray(x, dtype=np.float32)
    dim = values.shape[-1]
    sign_values = _validate_signs_np(signs, dim).astype(np.float32)
    return _hadamard_np(values * sign_values)


def apply_inverse_rht_np(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply the inverse row-vector RHT: ``x @ (H @ D)``."""

    values = np.asarray(x, dtype=np.float32)
    dim = values.shape[-1]
    sign_values = _validate_signs_np(signs, dim).astype(np.float32)
    return _hadamard_np(values) * sign_values


def _hadamard_mx(x: mx.array) -> mx.array:
    dim = x.shape[-1]
    _validate_power_of_two_dim(dim)
    out = x
    width = 1
    while width < dim:
        out = out.reshape((*out.shape[:-1], dim // (2 * width), 2, width))
        left = out[..., 0, :]
        right = out[..., 1, :]
        out = mx.stack((left + right, left - right), axis=-2)
        out = out.reshape((*x.shape[:-1], dim))
        width *= 2
    return out * (1.0 / math.sqrt(dim))


def apply_rht_mx(x: mx.array, signs: mx.array) -> mx.array:
    """Apply row-vector RHT in MLX over the trailing dimension."""

    dim = x.shape[-1]
    _validate_power_of_two_dim(dim)
    if signs.shape != (dim,):
        raise ValueError(f"rht_signs must have shape ({dim},), found {signs.shape}")
    return _hadamard_mx(x * signs.astype(x.dtype))


def apply_inverse_rht_mx(x: mx.array, signs: mx.array) -> mx.array:
    """Apply inverse row-vector RHT in MLX over the trailing dimension."""

    dim = x.shape[-1]
    _validate_power_of_two_dim(dim)
    if signs.shape != (dim,):
        raise ValueError(f"rht_signs must have shape ({dim},), found {signs.shape}")
    return _hadamard_mx(x) * signs.astype(x.dtype)


__all__ = [
    "apply_inverse_rht_mx",
    "apply_inverse_rht_np",
    "apply_rht_mx",
    "apply_rht_np",
    "deterministic_rht_signs",
]
