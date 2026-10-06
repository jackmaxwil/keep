"""Deterministic acceptance metrics for the Teich canonical adapter candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateGateResult:
    promising: bool
    heldout_top1_improvement: float
    heldout_top1_ci95: tuple[float, float]
    heldout_baseline_topk_kl: float
    heldout_candidate_topk_kl: float
    frozen_top1_regression: float
    frozen_mean_kld_regression: float
    failures: tuple[str, ...]


def paired_session_bootstrap_ci(
    baseline_correct: Mapping[str, Sequence[bool]],
    candidate_correct: Mapping[str, Sequence[bool]],
    *,
    seed: int = 20260712,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Position-weighted paired CI, resampling whole sessions with replacement."""

    if set(baseline_correct) != set(candidate_correct) or not baseline_correct:
        raise ValueError("paired bootstrap requires identical non-empty session inventories")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    sessions = sorted(baseline_correct)
    baseline_arrays = []
    candidate_arrays = []
    for session in sessions:
        baseline = np.asarray(baseline_correct[session], dtype=np.bool_)
        candidate = np.asarray(candidate_correct[session], dtype=np.bool_)
        if baseline.ndim != 1 or candidate.shape != baseline.shape or baseline.size == 0:
            raise ValueError("paired bootstrap session arrays must be aligned non-empty vectors")
        baseline_arrays.append(baseline)
        candidate_arrays.append(candidate)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected = rng.integers(0, len(sessions), size=len(sessions))
        numerator = 0
        denominator = 0
        for index in selected:
            baseline = baseline_arrays[int(index)]
            candidate = candidate_arrays[int(index)]
            numerator += int(candidate.sum()) - int(baseline.sum())
            denominator += baseline.size
        deltas[sample] = numerator / denominator
    low, high = np.quantile(deltas, [0.025, 0.975], method="linear")
    return float(low), float(high)


def evaluate_candidate_gate(
    *,
    baseline_correct: Mapping[str, Sequence[bool]],
    candidate_correct: Mapping[str, Sequence[bool]],
    baseline_topk_kl: float,
    candidate_topk_kl: float,
    frozen_baseline_top1: float,
    frozen_candidate_top1: float,
    frozen_baseline_mean_kld: float,
    frozen_candidate_mean_kld: float,
    bootstrap_seed: int = 20260712,
) -> CandidateGateResult:
    if any(
        not np.isfinite(value)
        for value in (
            baseline_topk_kl,
            candidate_topk_kl,
            frozen_baseline_top1,
            frozen_candidate_top1,
            frozen_baseline_mean_kld,
            frozen_candidate_mean_kld,
        )
    ):
        raise ValueError("candidate gate metrics must be finite")
    baseline_total = sum(np.asarray(value, dtype=np.bool_).sum() for value in baseline_correct.values())
    candidate_total = sum(np.asarray(value, dtype=np.bool_).sum() for value in candidate_correct.values())
    position_count = sum(len(value) for value in baseline_correct.values())
    if position_count <= 0:
        raise ValueError("candidate gate requires held-out positions")
    improvement = float(candidate_total - baseline_total) / position_count
    ci = paired_session_bootstrap_ci(
        baseline_correct,
        candidate_correct,
        seed=bootstrap_seed,
    )
    frozen_top1_regression = frozen_candidate_top1 - frozen_baseline_top1
    frozen_kld_regression = frozen_candidate_mean_kld - frozen_baseline_mean_kld
    failures = []
    if improvement < 0.01:
        failures.append("heldout_top1_improvement")
    if ci[0] <= 0.0:
        failures.append("heldout_top1_bootstrap_ci")
    if candidate_topk_kl >= baseline_topk_kl:
        failures.append("heldout_topk_kl")
    if frozen_top1_regression < -0.005:
        failures.append("frozen66_top1_regression")
    if frozen_kld_regression > 0.02:
        failures.append("frozen66_mean_kld_regression")
    return CandidateGateResult(
        promising=not failures,
        heldout_top1_improvement=improvement,
        heldout_top1_ci95=ci,
        heldout_baseline_topk_kl=float(baseline_topk_kl),
        heldout_candidate_topk_kl=float(candidate_topk_kl),
        frozen_top1_regression=float(frozen_top1_regression),
        frozen_mean_kld_regression=float(frozen_kld_regression),
        failures=tuple(failures),
    )


__all__ = [
    "CandidateGateResult",
    "evaluate_candidate_gate",
    "paired_session_bootstrap_ci",
]
