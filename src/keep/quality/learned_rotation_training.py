from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keep.quant.rht import apply_inverse_rht_np, apply_rht_np, deterministic_rht_signs
from keep.quant.rtn import dequantize_weight_np, quantize_weight_rtn
from keep.quant.rotation import cayley_update_np, validate_rotation_matrix_np


@dataclass(frozen=True)
class LearnedRotationTrainResult:
    rotation_matrix: np.ndarray
    initial_loss: float
    final_loss: float
    losses: tuple[float, ...]
    steps: int
    learning_rate: float
    init_kind: str


@dataclass(frozen=True)
class LearnedRHTSignsTrainResult:
    rht_signs: np.ndarray
    initial_loss: float
    final_loss: float
    losses: tuple[float, ...]
    accepted_flips: int
    steps: int
    candidates_per_step: int
    init_kind: str


def dense_rht_rotation_matrix(dim: int, *, seed: int | str) -> np.ndarray:
    signs = deterministic_rht_signs(dim, seed=seed)
    return apply_rht_np(np.eye(dim, dtype=np.float32), signs)


def quantized_reconstruction_loss(
    weight: np.ndarray,
    rotation: np.ndarray,
    *,
    group_size: int,
    code_bits: int = 8,
) -> tuple[float, np.ndarray]:
    w = np.asarray(weight, dtype=np.float32)
    r = validate_rotation_matrix_np(rotation, dim=w.shape[-1])
    quantized = quantize_weight_rtn(w @ r, group_size=group_size, code_bits=code_bits)
    q = dequantize_weight_np(quantized)
    equivalent = q @ r.T
    residual = equivalent - w
    return float(np.mean(residual * residual)), q.astype(np.float32, copy=False)


def rht_quantized_reconstruction_loss(
    weight: np.ndarray,
    signs: np.ndarray,
    *,
    group_size: int,
    code_bits: int = 8,
) -> float:
    w = np.asarray(weight, dtype=np.float32)
    sign_values = np.asarray(signs, dtype=np.int8)
    if sign_values.shape != (w.shape[-1],):
        raise ValueError(f"signs must have shape ({w.shape[-1]},), found {sign_values.shape}")
    if not np.all((sign_values == 1) | (sign_values == -1)):
        raise ValueError("signs values must be -1 or 1")
    quantized = quantize_weight_rtn(
        apply_rht_np(w, sign_values),
        group_size=group_size,
        code_bits=code_bits,
    )
    equivalent = apply_inverse_rht_np(dequantize_weight_np(quantized), sign_values)
    residual = equivalent - w
    return float(np.mean(residual * residual))


def train_learned_rotation_np(
    weight: np.ndarray,
    *,
    group_size: int,
    code_bits: int = 8,
    steps: int = 8,
    learning_rate: float = 0.05,
    init_rotation: np.ndarray | None = None,
    init_kind: str = "identity",
) -> LearnedRotationTrainResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2D [rows, input_dims], found {w.shape}")
    if init_rotation is None:
        rotation = np.eye(w.shape[-1], dtype=np.float32)
    else:
        rotation = validate_rotation_matrix_np(init_rotation, dim=w.shape[-1])

    losses: list[float] = []
    loss, q = quantized_reconstruction_loss(
        w,
        rotation,
        group_size=group_size,
        code_bits=code_bits,
    )
    losses.append(loss)
    for _ in range(steps):
        residual = q @ rotation.T - w
        gradient = 2.0 * (residual.T @ q) / float(w.size)
        rotation = cayley_update_np(rotation, -gradient, learning_rate=learning_rate)
        loss, q = quantized_reconstruction_loss(
            w,
            rotation,
            group_size=group_size,
            code_bits=code_bits,
        )
        losses.append(loss)

    return LearnedRotationTrainResult(
        rotation_matrix=rotation,
        initial_loss=losses[0],
        final_loss=losses[-1],
        losses=tuple(losses),
        steps=steps,
        learning_rate=learning_rate,
        init_kind=init_kind,
    )


def train_learned_rht_signs_np(
    weight: np.ndarray,
    *,
    group_size: int,
    code_bits: int = 8,
    steps: int = 8,
    candidates_per_step: int = 32,
    initial_signs: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    init_kind: str = "rht",
) -> LearnedRHTSignsTrainResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if candidates_per_step <= 0:
        raise ValueError("candidates_per_step must be positive")
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2D [rows, input_dims], found {w.shape}")
    if initial_signs is None:
        signs = np.ones((w.shape[-1],), dtype=np.int8)
    else:
        signs = np.asarray(initial_signs, dtype=np.int8).copy()
    if signs.shape != (w.shape[-1],):
        raise ValueError(f"initial_signs must have shape ({w.shape[-1]},), found {signs.shape}")
    if not np.all((signs == 1) | (signs == -1)):
        raise ValueError("initial_signs values must be -1 or 1")
    rng = rng or np.random.default_rng(0)

    current_loss = rht_quantized_reconstruction_loss(
        w,
        signs,
        group_size=group_size,
        code_bits=code_bits,
    )
    losses = [current_loss]
    accepted = 0
    dim = signs.shape[0]
    for _ in range(steps):
        candidate_indices = rng.choice(
            dim,
            size=min(candidates_per_step, dim),
            replace=False,
        )
        best_loss = current_loss
        best_index: int | None = None
        for index in candidate_indices:
            trial = signs.copy()
            trial[int(index)] *= np.int8(-1)
            trial_loss = rht_quantized_reconstruction_loss(
                w,
                trial,
                group_size=group_size,
                code_bits=code_bits,
            )
            if trial_loss < best_loss:
                best_loss = trial_loss
                best_index = int(index)
        if best_index is not None:
            signs[best_index] *= np.int8(-1)
            current_loss = best_loss
            accepted += 1
        losses.append(current_loss)

    return LearnedRHTSignsTrainResult(
        rht_signs=signs,
        initial_loss=losses[0],
        final_loss=losses[-1],
        losses=tuple(losses),
        accepted_flips=accepted,
        steps=steps,
        candidates_per_step=candidates_per_step,
        init_kind=init_kind,
    )


__all__ = [
    "LearnedRotationTrainResult",
    "LearnedRHTSignsTrainResult",
    "dense_rht_rotation_matrix",
    "quantized_reconstruction_loss",
    "rht_quantized_reconstruction_loss",
    "train_learned_rht_signs_np",
    "train_learned_rotation_np",
]
