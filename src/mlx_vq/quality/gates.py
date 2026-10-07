from __future__ import annotations

from typing import Any, Iterable

from mlx_vq.quality.teacher_cache import summarize_teacher_cache_records

QUALITY_VERDICTS = ("ACCEPT", "RECOVER", "FALL BACK", "ESCALATE")

DEFAULT_ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "mean_ppl_ratio_max": 1.15,
    "max_ppl_ratio_max": 1.30,
    "mean_kld_max": 0.25,
    "p999_kld_max": 2.0,
    "mean_top1_agreement_min": 0.85,
    "hard_prompt_ppl_ratio_max": 1.30,
    "hard_prompt_top1_agreement_min": 0.70,
}

DEFAULT_REGRESSION_LIMITS: dict[str, float] = {
    "max_ppl_ratio_delta": 0.02,
    "p999_kld_delta": 0.05,
    "hard_prompt_ppl_ratio_delta": 0.02,
    "hard_prompt_top1_drop": 0.02,
}

DEFAULT_HARD_ROW_INTENTS = frozenset({
    "code_debug",
    "hard_arithmetic",
    "instruction_following",
    "layer41_glu_code_route",
})


def validate_ladder_cache_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for index, record in enumerate(records):
        for key in ("prompt_set", "eval_split", "suite_version", "row_intent", "prompt_text_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or not value:
                errors.append(f"row {index} missing non-empty {key}")
        split = record.get("eval_split")
        prompt_id = record.get("prompt_id")
        prompt_set = record.get("prompt_set")
        if split not in {"selection", "report"}:
            errors.append(f"row {index} has invalid eval_split={split!r}")
        if split == "selection" and isinstance(prompt_id, str) and not prompt_id.startswith("select_"):
            errors.append(f"row {index} selection prompt_id must start with select_")
        if split == "report" and isinstance(prompt_id, str) and not prompt_id.startswith("report_"):
            errors.append(f"row {index} report prompt_id must start with report_")
        if split == "selection" and prompt_set != "air_vq_ladder_select_v1":
            errors.append(f"row {index} selection prompt_set mismatch")
        if split == "report" and prompt_set != "air_vq_ladder_report_v1":
            errors.append(f"row {index} report prompt_set mismatch")
    return {
        "schema_version": 1,
        "record_type": "air_vq_ladder_cache_metadata_validation",
        "row_count": len(records),
        "ok": not errors,
        "errors": errors,
    }


def validate_ladder_route_coverage(
    records: list[dict[str, Any]],
    *,
    required_layers: Iterable[int] = (31, 36, 41),
) -> dict[str, Any]:
    required = {str(layer) for layer in required_layers}
    present: set[str] = set()
    layer41_glu_code_represented = False
    for record in records:
        coverage = record.get("route_coverage")
        if not isinstance(coverage, dict):
            continue
        for layer in required:
            layer_payload = coverage.get(layer)
            if isinstance(layer_payload, dict) and layer_payload:
                present.add(layer)
        layer41 = coverage.get("41")
        if (
            isinstance(layer41, dict)
            and int(layer41.get("glu_code_route_count", 0) or 0) > 0
            and record.get("row_intent") == "layer41_glu_code_route"
        ):
            layer41_glu_code_represented = True
    missing = sorted(required - present, key=int)
    errors = [f"missing route coverage for layer {layer}" for layer in missing]
    if not layer41_glu_code_represented:
        errors.append("missing layer 41 GLU/code route representation")
    return {
        "schema_version": 1,
        "record_type": "air_vq_ladder_route_coverage_validation",
        "required_layers": sorted(required, key=int),
        "covered_layers": sorted(present, key=int),
        "missing_layers": missing,
        "layer41_glu_code_represented": layer41_glu_code_represented,
        "ok": not errors,
        "errors": errors,
    }


def evaluate_quality_gate(
    *,
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
    regression_limits: dict[str, float] | None = None,
    hard_row_intents: Iterable[str] = DEFAULT_HARD_ROW_INTENTS,
    require_exact_kld: bool = True,
) -> dict[str, Any]:
    active_thresholds = dict(DEFAULT_ACCEPTANCE_THRESHOLDS)
    if thresholds:
        active_thresholds.update(thresholds)
    active_limits = dict(DEFAULT_REGRESSION_LIMITS)
    if regression_limits:
        active_limits.update(regression_limits)
    reasons: list[str] = []
    if not baseline_records or not candidate_records:
        return _gate_report(
            "ESCALATE",
            reasons=["missing_baseline_or_candidate_records"],
            baseline_summary=None,
            candidate_summary=None,
        )

    if require_exact_kld and not _all_exact_full_logits(baseline_records + candidate_records):
        reasons.append("exact_full_logits_required")

    baseline_summary = summarize_teacher_cache_records(baseline_records)
    candidate_summary = summarize_teacher_cache_records(candidate_records)
    if not baseline_summary["all_memory_clean"] or not candidate_summary["all_memory_clean"]:
        reasons.append("memory_not_clean")

    if reasons:
        return _gate_report(
            "ESCALATE",
            reasons=reasons,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
        )

    tail_regression = _tail_regression(
        baseline_records=baseline_records,
        candidate_records=candidate_records,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        regression_limits=active_limits,
        hard_row_intents=set(hard_row_intents),
    )
    mean_improved = _mean_improved(baseline_summary, candidate_summary)
    accepted = _meets_acceptance(candidate_summary, active_thresholds)
    if accepted and not tail_regression["regressed"]:
        verdict = "ACCEPT"
    elif mean_improved and tail_regression["regressed"]:
        verdict = "FALL BACK"
        reasons.append("mean_improves_tail_worsens")
    elif mean_improved:
        verdict = "RECOVER"
    else:
        verdict = "FALL BACK"
        reasons.append("quality_not_improved")
    if tail_regression["regressed"] and "tail_regression" not in reasons:
        reasons.append("tail_regression")
    return _gate_report(
        verdict,
        reasons=reasons,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        tail_regression=tail_regression,
        thresholds=active_thresholds,
        regression_limits=active_limits,
    )


def _gate_report(
    verdict: str,
    *,
    reasons: list[str],
    baseline_summary: dict[str, Any] | None,
    candidate_summary: dict[str, Any] | None,
    tail_regression: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
    regression_limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    if verdict not in QUALITY_VERDICTS:
        raise ValueError(f"unknown quality verdict {verdict!r}")
    return {
        "schema_version": 1,
        "record_type": "air_vq_quality_gate",
        "verdict": verdict,
        "reasons": reasons,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "tail_regression": tail_regression or {
            "regressed": False,
            "split_half_confirmed": False,
            "hard_row_regressions": [],
        },
        "thresholds": thresholds or {},
        "regression_limits": regression_limits or {},
    }


def _all_exact_full_logits(records: list[dict[str, Any]]) -> bool:
    return all(record.get("kld_mode", "exact_full_logits") == "exact_full_logits" for record in records)


def _mean_improved(baseline_summary: dict[str, Any], candidate_summary: dict[str, Any]) -> bool:
    return (
        candidate_summary["mean_ppl_ratio"] < baseline_summary["mean_ppl_ratio"]
        and _number(candidate_summary.get("mean_kld"), float("inf"))
        < _number(baseline_summary.get("mean_kld"), float("inf"))
    )


def _meets_acceptance(summary: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return (
        summary["mean_ppl_ratio"] <= thresholds["mean_ppl_ratio_max"]
        and summary["max_ppl_ratio"] <= thresholds["max_ppl_ratio_max"]
        and _number(summary.get("mean_kld"), float("inf")) <= thresholds["mean_kld_max"]
        and _number(summary.get("p999_kld"), float("inf")) <= thresholds["p999_kld_max"]
        and _number(summary.get("mean_top1_agreement"), -float("inf"))
        >= thresholds["mean_top1_agreement_min"]
    )


def _tail_regression(
    *,
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    regression_limits: dict[str, float],
    hard_row_intents: set[str],
) -> dict[str, Any]:
    max_ppl_delta = candidate_summary["max_ppl_ratio"] - baseline_summary["max_ppl_ratio"]
    p999_delta = _number(candidate_summary.get("p999_kld"), 0.0) - _number(
        baseline_summary.get("p999_kld"),
        0.0,
    )
    hard_row_regressions = _hard_row_regressions(
        baseline_records,
        candidate_records,
        hard_row_intents=hard_row_intents,
        regression_limits=regression_limits,
    )
    split_half = _split_half_tail_regression(
        baseline_records,
        candidate_records,
        regression_limits=regression_limits,
    )
    regressed = (
        (
            max_ppl_delta > regression_limits["max_ppl_ratio_delta"]
            or p999_delta > regression_limits["p999_kld_delta"]
        )
        and split_half
    ) or bool(hard_row_regressions)
    return {
        "regressed": regressed,
        "max_ppl_delta": max_ppl_delta,
        "p999_kld_delta": p999_delta,
        "split_half_confirmed": split_half,
        "hard_row_regressions": hard_row_regressions,
    }


def _split_half_tail_regression(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    regression_limits: dict[str, float],
) -> bool:
    if len(baseline_records) < 2 or len(candidate_records) < 2:
        return False
    by_prompt = _records_by_prompt(baseline_records)
    paired = [
        (base, candidate)
        for candidate in candidate_records
        if (base := by_prompt.get(str(candidate.get("prompt_id")))) is not None
    ]
    if len(paired) < 2:
        return False
    confirmations = []
    for parity in (0, 1):
        half = [pair for index, pair in enumerate(paired) if index % 2 == parity]
        if not half:
            confirmations.append(False)
            continue
        base_max_ppl = max(float(base["ppl_ratio"]) for base, _ in half)
        cand_max_ppl = max(float(candidate["ppl_ratio"]) for _, candidate in half)
        base_p999 = max(_record_p999(base) for base, _ in half)
        cand_p999 = max(_record_p999(candidate) for _, candidate in half)
        confirmations.append(
            cand_max_ppl - base_max_ppl > regression_limits["max_ppl_ratio_delta"]
            or cand_p999 - base_p999 > regression_limits["p999_kld_delta"]
        )
    return all(confirmations)


def _hard_row_regressions(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    hard_row_intents: set[str],
    regression_limits: dict[str, float],
) -> list[dict[str, Any]]:
    by_prompt = _records_by_prompt(baseline_records)
    regressions: list[dict[str, Any]] = []
    for candidate in candidate_records:
        if candidate.get("row_intent") not in hard_row_intents:
            continue
        baseline = by_prompt.get(str(candidate.get("prompt_id")))
        if baseline is None:
            continue
        ppl_delta = float(candidate["ppl_ratio"]) - float(baseline["ppl_ratio"])
        top1_delta = _number(candidate.get("top1_agreement"), 0.0) - _number(
            baseline.get("top1_agreement"),
            0.0,
        )
        if (
            ppl_delta > regression_limits["hard_prompt_ppl_ratio_delta"]
            or -top1_delta > regression_limits["hard_prompt_top1_drop"]
        ):
            regressions.append(
                {
                    "prompt_id": candidate.get("prompt_id"),
                    "row_intent": candidate.get("row_intent"),
                    "ppl_delta": ppl_delta,
                    "top1_delta": top1_delta,
                }
            )
    return regressions


def _records_by_prompt(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("prompt_id")): record for record in records}


def _record_p999(record: dict[str, Any]) -> float:
    if record.get("p999_kld") is not None:
        return float(record["p999_kld"])
    token_klds = record.get("token_klds") or []
    return max((float(value) for value in token_klds), default=0.0)


def _number(value: Any, default: float) -> float:
    return default if value is None else float(value)
