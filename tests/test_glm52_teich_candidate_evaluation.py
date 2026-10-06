from __future__ import annotations

from mlx_vq.quality.glm52_teich_candidate_evaluation import (
    evaluate_candidate_gate,
    paired_session_bootstrap_ci,
)


def test_paired_bootstrap_is_deterministic_and_session_resampled() -> None:
    baseline = {"a": [False] * 100, "b": [True, False] * 50}
    candidate = {"a": [True] * 10 + [False] * 90, "b": [True, False] * 50}
    assert paired_session_bootstrap_ci(baseline, candidate, samples=1000) == (
        0.0,
        0.1,
    )


def test_candidate_gate_enforces_all_five_acceptance_rails() -> None:
    baseline = {str(index): [False] * 100 for index in range(20)}
    candidate = {str(index): [True] * 2 + [False] * 98 for index in range(20)}
    result = evaluate_candidate_gate(
        baseline_correct=baseline,
        candidate_correct=candidate,
        baseline_topk_kl=0.4,
        candidate_topk_kl=0.3,
        frozen_baseline_top1=0.8,
        frozen_candidate_top1=0.798,
        frozen_baseline_mean_kld=0.2,
        frozen_candidate_mean_kld=0.215,
    )
    assert result.promising is True
    assert result.heldout_top1_improvement == 0.02

    failed = evaluate_candidate_gate(
        baseline_correct=baseline,
        candidate_correct=baseline,
        baseline_topk_kl=0.4,
        candidate_topk_kl=0.4,
        frozen_baseline_top1=0.8,
        frozen_candidate_top1=0.79,
        frozen_baseline_mean_kld=0.2,
        frozen_candidate_mean_kld=0.23,
    )
    assert set(failed.failures) == {
        "heldout_top1_improvement",
        "heldout_top1_bootstrap_ci",
        "heldout_topk_kl",
        "frozen66_top1_regression",
        "frozen66_mean_kld_regression",
    }
