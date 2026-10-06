"""Import-light NumPy oracles for GLM-5.2 distillation losses.

The training implementation mirrors these equations with MLX.  This module
intentionally has no MLX imports so contract and numerical tests run in the
headless/no-Metal sandbox.
"""

from __future__ import annotations

import numpy as np


_EPS = np.float32(1.0e-7)


def _as_float32_2d(value: np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"{label} must have shape [positions, features]")
    return result


def _log_softmax(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    maximum = np.max(value, axis=-1, keepdims=True)
    return value - (maximum + np.log(np.sum(np.exp(value - maximum), axis=-1, keepdims=True)))


def full_vocab_kl_numpy(student_logits: np.ndarray, teacher_logits: np.ndarray) -> np.float32:
    """Current dense full-vocabulary KL, evaluated in float32."""

    student = _as_float32_2d(student_logits, label="student_logits")
    teacher = _as_float32_2d(teacher_logits, label="teacher_logits")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher logits must have matching shapes")
    teacher_logp = _log_softmax(teacher)
    student_logp = _log_softmax(student)
    per_position = np.sum(
        np.exp(teacher_logp) * (teacher_logp - student_logp),
        axis=-1,
        dtype=np.float32,
    )
    return np.mean(per_position, dtype=np.float32)


def _full_vocab_teacher_logits(
    topk_ids: np.ndarray,
    topk_values: np.ndarray,
    vocab_size: int,
) -> np.ndarray | None:
    if topk_ids.shape[-1] != vocab_size:
        return None
    expected = np.arange(vocab_size, dtype=np.int32)
    if any(not np.array_equal(np.sort(row), expected) for row in topk_ids):
        return None
    dense = np.empty_like(topk_values, dtype=np.float32)
    np.put_along_axis(dense, topk_ids, topk_values.astype(np.float32), axis=-1)
    return dense


def topk_aggregated_tail_kl_numpy(
    student_logits: np.ndarray,
    topk_logit_ids: np.ndarray,
    topk_logit_values: np.ndarray,
    tail_mass: np.ndarray,
) -> np.float32:
    """KL over teacher top-K tokens plus one aggregated tail event.

    The cached logits determine the conditional distribution within top-K;
    ``tail_mass`` supplies the teacher probability outside top-K.  A complete
    vocabulary with exactly zero tail dispatches to the dense implementation,
    preserving its float32 operation order and bytes.
    """

    student = _as_float32_2d(student_logits, label="student_logits")
    ids = np.asarray(topk_logit_ids)
    values = np.asarray(topk_logit_values)
    tail = np.asarray(tail_mass)
    if ids.ndim != 2 or ids.shape != values.shape or ids.shape[0] != student.shape[0]:
        raise ValueError("top-K IDs and values must have matching [positions, K] shapes")
    if tail.shape != (student.shape[0],):
        raise ValueError("tail_mass must have shape [positions]")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("top-K IDs must be integers")
    ids = ids.astype(np.int32, copy=False)
    if np.any(ids < 0) or np.any(ids >= student.shape[-1]):
        raise ValueError("top-K token ID is outside the student vocabulary")
    if any(np.unique(row).size != row.size for row in ids):
        raise ValueError("top-K token IDs must be unique per position")
    values32 = values.astype(np.float32)
    tail32 = tail.astype(np.float32)
    if np.any(tail32 < 0.0) or np.any(tail32 >= 1.0):
        raise ValueError("tail_mass must be in [0, 1)")

    if np.all(tail32 == 0.0):
        dense_teacher = _full_vocab_teacher_logits(ids, values32, student.shape[-1])
        if dense_teacher is not None:
            return full_vocab_kl_numpy(student, dense_teacher)

    top_conditional_logp = _log_softmax(values32)
    retained_mass = np.float32(1.0) - tail32
    teacher_top_p = np.exp(top_conditional_logp) * retained_mass[:, None]
    teacher_top_logp = np.log(np.maximum(teacher_top_p, _EPS))
    student_logp = _log_softmax(student)
    rows = np.arange(student.shape[0])[:, None]
    selected_student_logp = student_logp[rows, ids]
    student_tail_p = np.maximum(
        np.float32(1.0) - np.sum(np.exp(selected_student_logp), axis=-1, dtype=np.float32),
        _EPS,
    )
    tail_term = np.where(
        tail32 > 0.0,
        tail32 * (np.log(np.maximum(tail32, _EPS)) - np.log(student_tail_p)),
        np.float32(0.0),
    )
    per_position = np.sum(
        teacher_top_p * (teacher_top_logp - selected_student_logp),
        axis=-1,
        dtype=np.float32,
    ) + tail_term
    return np.mean(per_position, dtype=np.float32)


def feature_space_linear_cka_numpy(
    teacher_hidden: np.ndarray,
    student_hidden: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> np.float32:
    """Linear CKA via feature-space covariance products, never N x N Grams."""

    teacher = _as_float32_2d(teacher_hidden, label="teacher_hidden")
    student = _as_float32_2d(student_hidden, label="student_hidden")
    if teacher.shape[0] != student.shape[0]:
        raise ValueError("teacher and student hidden states must share positions")
    teacher = teacher - np.mean(teacher, axis=0, keepdims=True, dtype=np.float32)
    student = student - np.mean(student, axis=0, keepdims=True, dtype=np.float32)
    cross = teacher.T @ student
    teacher_cov = teacher.T @ teacher
    student_cov = student.T @ student
    numerator = np.sum(cross * cross, dtype=np.float32)
    denominator = np.sqrt(
        np.sum(teacher_cov * teacher_cov, dtype=np.float32)
        * np.sum(student_cov * student_cov, dtype=np.float32)
    )
    if denominator <= np.float32(eps):
        return np.float32(0.0)
    similarity = np.float32(numerator / denominator)
    if np.array_equal(teacher_hidden, student_hidden):
        return np.float32(1.0)
    return np.clip(similarity, np.float32(0.0), np.float32(1.0)).astype(np.float32)


def seeded_expert_exploration_schedule(num_experts: int, *, seed: int) -> np.ndarray:
    """Return a deterministic permutation that covers every expert once."""

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    return np.random.default_rng(seed).permutation(num_experts).astype(np.int32)


def router_distillation_numpy(
    student_router_logits: np.ndarray,
    teacher_topk_ids: np.ndarray,
    teacher_topk_weights: np.ndarray,
    *,
    entropy_beta: float,
    mc_expert_explore: bool,
    seed: int,
) -> np.float32:
    """Sparse-teacher router KL, entropy reward, and seeded full-expert exploration."""

    logits = _as_float32_2d(student_router_logits, label="student_router_logits")
    ids = np.asarray(teacher_topk_ids)
    weights = np.asarray(teacher_topk_weights, dtype=np.float32)
    if ids.ndim != 2 or ids.shape != weights.shape or ids.shape[0] != logits.shape[0]:
        raise ValueError("teacher router IDs and weights must share [positions, K]")
    if entropy_beta < 0:
        raise ValueError("router entropy beta must be non-negative")
    ids = ids.astype(np.int32, copy=False)
    if np.any(ids < 0) or np.any(ids >= logits.shape[-1]):
        raise ValueError("teacher router expert ID is outside the student router")
    if np.any(weights < 0) or np.any(np.sum(weights, axis=-1) <= 0):
        raise ValueError("teacher router weights must be non-negative with positive row sums")
    teacher = weights / np.sum(weights, axis=-1, keepdims=True, dtype=np.float32)
    logp = _log_softmax(logits)
    rows = np.arange(logits.shape[0])[:, None]
    kl = np.mean(
        np.sum(
            teacher * (np.log(np.maximum(teacher, _EPS)) - logp[rows, ids]),
            axis=-1,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    probs = np.exp(logp)
    entropy = np.mean(-np.sum(probs * logp, axis=-1, dtype=np.float32), dtype=np.float32)
    result = np.float32(kl - np.float32(entropy_beta) * entropy)
    if mc_expert_explore > 0:
        schedule = seeded_expert_exploration_schedule(logits.shape[-1], seed=seed)
        rng = np.random.default_rng(seed)
        positive_weights = rng.uniform(0.5, 1.5, size=logits.shape[-1]).astype(np.float32)
        exploration_target = np.empty_like(positive_weights)
        exploration_target[schedule] = positive_weights
        exploration_target /= np.sum(exploration_target, dtype=np.float32)
        exploration = np.mean(
            -np.sum(logp * exploration_target[None, :], axis=-1, dtype=np.float32),
            dtype=np.float32,
        )
        result = np.float32(result + exploration)
    return result


__all__ = [
    "feature_space_linear_cka_numpy",
    "full_vocab_kl_numpy",
    "router_distillation_numpy",
    "seeded_expert_exploration_schedule",
    "topk_aggregated_tail_kl_numpy",
]
