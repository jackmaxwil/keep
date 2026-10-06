from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1] / "src/mlx_vq/quality/glm52_distill_loss.py"
)
SPEC = importlib.util.spec_from_file_location("glm52_distill_loss", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
distill_loss = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distill_loss)

feature_space_linear_cka_numpy = distill_loss.feature_space_linear_cka_numpy
full_vocab_kl_numpy = distill_loss.full_vocab_kl_numpy
router_distillation_numpy = distill_loss.router_distillation_numpy
seeded_expert_exploration_schedule = distill_loss.seeded_expert_exploration_schedule
topk_aggregated_tail_kl_numpy = distill_loss.topk_aggregated_tail_kl_numpy


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    maximum = np.max(x, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(x - maximum), axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


def test_topk_full_vocab_zero_tail_is_bit_identical_to_dense_kl():
    teacher = np.array(
        [[-1.25, 0.5, 2.0, -0.75], [0.125, -0.5, 0.75, 1.5]], dtype=np.float32
    )
    student = np.array(
        [[0.25, -0.125, 1.25, -1.0], [-0.75, 0.5, 1.0, 0.25]], dtype=np.float32
    )
    ids = np.broadcast_to(np.arange(teacher.shape[-1], dtype=np.int32), teacher.shape)

    dense = full_vocab_kl_numpy(student, teacher)
    topk = topk_aggregated_tail_kl_numpy(
        student,
        ids,
        teacher,
        np.zeros((teacher.shape[0],), dtype=np.float16),
    )

    assert dense.dtype == np.float32
    assert topk.dtype == np.float32
    assert dense.tobytes() == topk.tobytes()


def test_topk_aggregated_tail_matches_explicit_numpy_oracle():
    student = np.array(
        [[0.4, -0.3, 1.2, -0.8, 0.1], [-0.2, 0.9, 0.3, -0.5, 1.1]], dtype=np.float32
    )
    ids = np.array([[2, 0], [4, 1]], dtype=np.int32)
    values = np.array([[1.5, 0.25], [2.0, 0.5]], dtype=np.float16)
    tail = np.array([0.2, 0.125], dtype=np.float16)

    actual = topk_aggregated_tail_kl_numpy(student, ids, values, tail)

    student_logp = student - _logsumexp(student, keepdims=True)
    rows = np.arange(student.shape[0])[:, None]
    top_conditional_logp = values.astype(np.float32) - _logsumexp(
        values.astype(np.float32), keepdims=True
    )
    top_teacher_p = np.exp(top_conditional_logp) * (1.0 - tail.astype(np.float32))[:, None]
    top_teacher_logp = np.log(top_teacher_p)
    top_student_logp = student_logp[rows, ids]
    student_tail_p = 1.0 - np.sum(np.exp(top_student_logp), axis=-1)
    per_position = np.sum(
        top_teacher_p * (top_teacher_logp - top_student_logp), axis=-1
    ) + tail.astype(np.float32) * (
        np.log(tail.astype(np.float32)) - np.log(student_tail_p)
    )
    expected = np.mean(per_position, dtype=np.float32)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_feature_space_cka_matches_explicit_gram_oracle_and_invariances():
    teacher = np.array(
        [[1.0, 2.0, -1.0], [0.0, -1.0, 2.0], [2.0, 0.5, 1.0], [-1.0, 1.5, 0.0]],
        dtype=np.float32,
    )
    student = np.array(
        [[0.5, -1.0, 2.0], [1.5, 0.0, -0.5], [-1.0, 2.0, 0.5], [2.0, 1.0, 1.5]],
        dtype=np.float32,
    )
    actual = feature_space_linear_cka_numpy(teacher, student)

    tx = teacher - teacher.mean(axis=0, keepdims=True)
    sx = student - student.mean(axis=0, keepdims=True)
    kt = tx @ tx.T
    ks = sx @ sx.T
    expected = np.sum(kt * ks) / np.sqrt(np.sum(kt * kt) * np.sum(ks * ks))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    q, _ = np.linalg.qr(
        np.array([[1.0, 2.0, 0.0], [-2.0, 1.0, 0.5], [1.0, -1.0, 2.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        feature_space_linear_cka_numpy(teacher, student * np.float32(3.25)),
        actual,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        feature_space_linear_cka_numpy(teacher, student @ q),
        actual,
        rtol=1e-5,
        atol=1e-6,
    )
    assert feature_space_linear_cka_numpy(teacher, teacher) == np.float32(1.0)


def test_router_kl_entropy_matches_numpy_and_mc_is_seeded_with_full_coverage():
    logits = np.array(
        [[0.4, -0.2, 1.0, 0.1, -0.5], [-0.1, 0.8, 0.25, -0.4, 0.5]], dtype=np.float32
    )
    teacher_ids = np.array([[2, 0], [1, 4]], dtype=np.int32)
    teacher_weights = np.array([[0.75, 0.25], [0.6, 0.4]], dtype=np.float16)
    beta = 0.2

    actual = router_distillation_numpy(
        logits,
        teacher_ids,
        teacher_weights,
        entropy_beta=beta,
        mc_expert_explore=False,
        seed=17,
    )
    logp = logits - _logsumexp(logits, keepdims=True)
    teacher = teacher_weights.astype(np.float32)
    teacher /= teacher.sum(axis=-1, keepdims=True)
    rows = np.arange(logits.shape[0])[:, None]
    kl = np.mean(np.sum(teacher * (np.log(teacher) - logp[rows, teacher_ids]), axis=-1))
    entropy = np.mean(-np.sum(np.exp(logp) * logp, axis=-1))
    np.testing.assert_allclose(actual, kl - beta * entropy, rtol=1e-6, atol=1e-7)

    schedule_a = seeded_expert_exploration_schedule(11, seed=20260712)
    schedule_b = seeded_expert_exploration_schedule(11, seed=20260712)
    assert schedule_a.tobytes() == schedule_b.tobytes()
    assert sorted(schedule_a.tolist()) == list(range(11))

    explored_a = router_distillation_numpy(
        logits,
        teacher_ids,
        teacher_weights,
        entropy_beta=beta,
        mc_expert_explore=True,
        seed=20260712,
    )
    explored_b = router_distillation_numpy(
        logits,
        teacher_ids,
        teacher_weights,
        entropy_beta=beta,
        mc_expert_explore=True,
        seed=20260712,
    )
    assert np.asarray(explored_a).tobytes() == np.asarray(explored_b).tobytes()
