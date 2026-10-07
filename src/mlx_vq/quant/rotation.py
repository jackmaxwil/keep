from __future__ import annotations

import mlx.core as mx
import numpy as np


def validate_rotation_matrix_np(
    rotation: np.ndarray,
    *,
    dim: int,
    atol: float = 1e-4,
) -> np.ndarray:
    values = np.asarray(rotation, dtype=np.float32)
    if values.shape != (dim, dim):
        raise ValueError(f"rotation_matrix must have shape ({dim}, {dim}), found {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("rotation_matrix must contain only finite values")
    identity = np.eye(dim, dtype=np.float32)
    gram = values.T @ values
    if not np.allclose(gram, identity, rtol=atol, atol=atol):
        raise ValueError("rotation_matrix must be orthogonal")
    return values


def validate_rotation_matrix_mx(rotation: mx.array, *, dim: int) -> mx.array:
    if rotation.shape != (dim, dim):
        raise ValueError(f"rotation_matrix must have shape ({dim}, {dim}), found {rotation.shape}")
    return rotation.astype(mx.float32)


def apply_rotation_np(x: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a row-vector input rotation over the trailing dimension."""

    values = np.asarray(x, dtype=np.float32)
    matrix = validate_rotation_matrix_np(rotation, dim=values.shape[-1])
    return np.matmul(values, matrix).astype(np.float32, copy=False)


def apply_rotation_mx(x: mx.array, rotation: mx.array) -> mx.array:
    """Apply a row-vector input rotation in MLX over the trailing dimension."""

    matrix = validate_rotation_matrix_mx(rotation, dim=x.shape[-1])
    return x @ matrix.astype(x.dtype)


def cayley_update_np(
    rotation: np.ndarray,
    gradient: np.ndarray,
    *,
    learning_rate: float,
) -> np.ndarray:
    """Perform one Cayley-style Stiefel update for an orthogonal rotation matrix."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    r = validate_rotation_matrix_np(rotation, dim=np.asarray(rotation).shape[0]).astype(np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    if g.shape != r.shape:
        raise ValueError(f"gradient must have shape {r.shape}, found {g.shape}")
    if not np.all(np.isfinite(g)):
        raise ValueError("gradient must contain only finite values")

    g_hat = g @ r.T - 0.5 * (r @ r.T @ g @ r.T)
    skew = g_hat - g_hat.T
    identity = np.eye(r.shape[0], dtype=np.float64)
    step = 0.5 * float(learning_rate) * skew
    updated = np.linalg.solve(identity - step, (identity + step) @ r)
    return updated.astype(np.float32)


__all__ = [
    "apply_rotation_mx",
    "apply_rotation_np",
    "cayley_update_np",
    "validate_rotation_matrix_mx",
    "validate_rotation_matrix_np",
]
