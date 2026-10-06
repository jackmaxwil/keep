from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            row["_source_jsonl"] = str(path)
            rows.append(row)
    return rows


def _select_eval_row(
    rows: list[dict[str, Any]],
    *,
    prompt_id: str | None,
    row_index: int | None,
    label: str,
) -> dict[str, Any]:
    matches = []
    for row in rows:
        if prompt_id is not None and row.get("prompt_id") != prompt_id:
            continue
        if row_index is not None and int(row.get("row_index", -1)) != int(row_index):
            continue
        matches.append(row)
    if len(matches) != 1:
        target = f"prompt_id={prompt_id!r} row_index={row_index!r}"
        raise ValueError(f"expected exactly one {label} row for {target}, found {len(matches)}")
    return matches[0]


def _memory_clean(row: dict[str, Any]) -> bool:
    return int(row.get("pageouts_delta") or 0) == 0 and int(row.get("swapouts_delta") or 0) == 0


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return float(candidate[key]) - float(baseline[key])


def _list_item(row: dict[str, Any], key: str, index: int) -> Any:
    values = row.get(key)
    if not isinstance(values, list):
        raise ValueError(f"row is missing list field {key!r}")
    if index < 0 or index >= len(values):
        raise ValueError(f"focus_position {index} out of range for {key!r} length {len(values)}")
    return values[index]


def _optional_float_list_item(row: dict[str, Any], key: str, index: int) -> float | None:
    if key not in row:
        return None
    value = _list_item(row, key, index)
    if value is None:
        return None
    return float(value)


def _focus_position_summary(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    focus_position: int,
) -> dict[str, Any]:
    teacher_top1 = int(_list_item(baseline, "teacher_top1_ids", focus_position))
    baseline_vq_top1 = int(_list_item(baseline, "vq_top1_ids", focus_position))
    candidate_vq_top1 = int(_list_item(candidate, "vq_top1_ids", focus_position))
    baseline_token_kld = float(_list_item(baseline, "token_klds", focus_position))
    candidate_token_kld = float(_list_item(candidate, "token_klds", focus_position))
    baseline_target_logprob = float(_list_item(baseline, "vq_target_logprobs", focus_position))
    candidate_target_logprob = float(_list_item(candidate, "vq_target_logprobs", focus_position))
    baseline_teacher_margin = _optional_float_list_item(
        baseline,
        "vq_teacher_top1_margin_vs_vq_top1",
        focus_position,
    )
    candidate_teacher_margin = _optional_float_list_item(
        candidate,
        "vq_teacher_top1_margin_vs_vq_top1",
        focus_position,
    )
    teacher_margin_delta = (
        None
        if baseline_teacher_margin is None or candidate_teacher_margin is None
        else candidate_teacher_margin - baseline_teacher_margin
    )
    focus = {
        "position": int(_list_item(baseline, "positions", focus_position)),
        "target_token_id": int(_list_item(baseline, "target_token_ids", focus_position)),
        "teacher_top1_id": teacher_top1,
        "baseline_vq_top1_id": baseline_vq_top1,
        "candidate_vq_top1_id": candidate_vq_top1,
        "baseline_top1_match": baseline_vq_top1 == teacher_top1,
        "candidate_top1_match": candidate_vq_top1 == teacher_top1,
        "top1_flipped_to_teacher": baseline_vq_top1 != teacher_top1 and candidate_vq_top1 == teacher_top1,
        "baseline_token_kld": baseline_token_kld,
        "candidate_token_kld": candidate_token_kld,
        "token_kld_delta": candidate_token_kld - baseline_token_kld,
        "baseline_vq_target_logprob": baseline_target_logprob,
        "candidate_vq_target_logprob": candidate_target_logprob,
        "vq_target_logprob_delta": candidate_target_logprob - baseline_target_logprob,
    }
    if teacher_margin_delta is not None:
        focus.update(
            {
                "baseline_vq_teacher_top1_logprob": _optional_float_list_item(
                    baseline,
                    "vq_teacher_top1_logprobs",
                    focus_position,
                ),
                "candidate_vq_teacher_top1_logprob": _optional_float_list_item(
                    candidate,
                    "vq_teacher_top1_logprobs",
                    focus_position,
                ),
                "baseline_vq_top1_logprob": _optional_float_list_item(
                    baseline,
                    "vq_top1_logprobs",
                    focus_position,
                ),
                "candidate_vq_top1_logprob": _optional_float_list_item(
                    candidate,
                    "vq_top1_logprobs",
                    focus_position,
                ),
                "baseline_vq_teacher_top1_margin_vs_vq_top1": baseline_teacher_margin,
                "candidate_vq_teacher_top1_margin_vs_vq_top1": candidate_teacher_margin,
                "vq_teacher_top1_margin_delta": teacher_margin_delta,
                "vq_teacher_top1_margin_improved": teacher_margin_delta > 0.0,
            }
        )
    return focus


def _reject_reasons(
    *,
    candidate_memory_clean: bool,
    deltas: dict[str, float],
    focus: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not candidate_memory_clean:
        reasons.append("candidate_memory_dirty")
    if deltas["top1_agreement"] <= 0.0:
        reasons.append("no_top1_gain")
    if deltas["p999_kld"] >= 0.0:
        reasons.append("p999_kld_regressed")
    if deltas["mean_kld"] >= 0.0:
        reasons.append("mean_kld_regressed")
    if deltas["ppl_ratio"] >= 0.0:
        reasons.append("ppl_ratio_regressed")
    if not bool(focus["top1_flipped_to_teacher"]):
        reasons.append("focus_top1_not_flipped")
    if float(focus["token_kld_delta"]) >= 0.0:
        reasons.append("focus_token_kld_regressed")
    if float(focus["vq_target_logprob_delta"]) <= 0.0:
        reasons.append("focus_target_logprob_not_improved")
    return reasons


def _combined_margin_guard(
    *,
    candidate_memory_clean: bool,
    baseline_memory_clean: bool,
    deltas: dict[str, float],
    focus: dict[str, Any],
) -> dict[str, Any]:
    margin_delta = focus.get("vq_teacher_top1_margin_delta")
    margin_available = margin_delta is not None
    teacher_top1_margin_improved = margin_available and float(margin_delta) > 0.0
    target_logprob_preserved = float(focus["vq_target_logprob_delta"]) >= 0.0
    token_kld_improved = float(focus["token_kld_delta"]) <= 0.0
    row_top1_non_regressed = float(deltas["top1_agreement"]) >= 0.0
    row_nll_preserved = float(deltas["nll_delta"]) <= 0.0
    row_ppl_preserved = float(deltas["ppl_ratio"]) <= 0.0
    row_mean_kld_preserved = float(deltas["mean_kld"]) <= 0.0
    row_p999_kld_preserved = float(deltas["p999_kld"]) <= 0.0
    focus_top1_flipped = bool(focus["top1_flipped_to_teacher"])

    failed_checks: list[str] = []
    if not baseline_memory_clean:
        failed_checks.append("baseline_memory_dirty")
    if not candidate_memory_clean:
        failed_checks.append("candidate_memory_dirty")
    if not margin_available:
        failed_checks.append("teacher_top1_margin_missing")
    elif not teacher_top1_margin_improved:
        failed_checks.append("teacher_top1_margin_not_improved")
    if not target_logprob_preserved:
        failed_checks.append("target_logprob_not_preserved")
    if not token_kld_improved:
        failed_checks.append("focus_token_kld_regressed")
    if not row_top1_non_regressed:
        failed_checks.append("row_top1_regressed")
    if not row_nll_preserved:
        failed_checks.append("row_nll_regressed")
    if not row_ppl_preserved:
        failed_checks.append("row_ppl_regressed")
    if not row_mean_kld_preserved:
        failed_checks.append("row_mean_kld_regressed")
    if not row_p999_kld_preserved:
        failed_checks.append("row_p999_kld_regressed")
    if not focus_top1_flipped:
        failed_checks.append("focus_top1_not_flipped")

    required_training_checks = (
        candidate_memory_clean
        and baseline_memory_clean
        and teacher_top1_margin_improved
        and target_logprob_preserved
        and token_kld_improved
        and row_top1_non_regressed
        and row_nll_preserved
        and row_ppl_preserved
        and row_mean_kld_preserved
        and row_p999_kld_preserved
    )
    ready_for_broader_eval = bool(required_training_checks and focus_top1_flipped)
    training_signal = bool(required_training_checks and not focus_top1_flipped)

    if ready_for_broader_eval:
        decision = "candidate_ready_for_cross_row_guard"
    elif training_signal:
        decision = "margin_signal_needs_top1_flip_or_cross_row_guard"
    elif (
        teacher_top1_margin_improved
        and target_logprob_preserved
        and token_kld_improved
        and row_top1_non_regressed
    ):
        decision = "partial_margin_signal_needs_row_preservation_or_top1_guard"
    elif teacher_top1_margin_improved and (
        not target_logprob_preserved
        or not row_top1_non_regressed
        or not row_nll_preserved
        or not row_ppl_preserved
    ):
        decision = "reject_margin_overfit"
    elif margin_available:
        decision = "reject_no_guarded_margin_signal"
    else:
        decision = "margin_guard_unavailable"

    margin_score = 0.0 if margin_delta is None else float(margin_delta)
    score = (
        margin_score
        + max(float(focus["vq_target_logprob_delta"]), 0.0)
        + max(-float(focus["token_kld_delta"]), 0.0)
        + max(-float(deltas["nll_delta"]), 0.0)
        + max(float(deltas["top1_agreement"]), 0.0)
        - max(float(deltas["ppl_ratio"]), 0.0)
        - max(float(deltas["mean_kld"]), 0.0)
        - max(float(deltas["p999_kld"]), 0.0)
    )

    return {
        "decision": decision,
        "score": score,
        "margin_available": margin_available,
        "teacher_top1_margin_improved": bool(teacher_top1_margin_improved),
        "target_logprob_preserved": target_logprob_preserved,
        "token_kld_improved": token_kld_improved,
        "row_top1_non_regressed": row_top1_non_regressed,
        "row_nll_preserved": row_nll_preserved,
        "row_ppl_preserved": row_ppl_preserved,
        "row_mean_kld_preserved": row_mean_kld_preserved,
        "row_p999_kld_preserved": row_p999_kld_preserved,
        "focus_top1_flipped": focus_top1_flipped,
        "training_signal": training_signal,
        "ready_for_broader_eval": ready_for_broader_eval,
        "failed_checks": failed_checks,
    }


def analyze_sparse_plan_eval(
    *,
    candidate_jsonl: Path,
    baseline_jsonl: Path,
    prompt_id: str | None = None,
    row_index: int | None = None,
    focus_position: int = 0,
) -> dict[str, Any]:
    if prompt_id is None and row_index is None:
        raise ValueError("prompt_id or row_index is required")
    candidate = _select_eval_row(
        _read_jsonl(candidate_jsonl),
        prompt_id=prompt_id,
        row_index=row_index,
        label="candidate",
    )
    baseline = _select_eval_row(
        _read_jsonl(baseline_jsonl),
        prompt_id=prompt_id,
        row_index=row_index,
        label="baseline",
    )
    deltas = {
        "mean_kld": _delta(candidate, baseline, "mean_kld"),
        "top1_agreement": _delta(candidate, baseline, "top1_agreement"),
        "ppl_ratio": _delta(candidate, baseline, "ppl_ratio"),
        "p999_kld": _delta(candidate, baseline, "p999_kld"),
        "nll_delta": _delta(candidate, baseline, "nll_delta") if "nll_delta" in candidate and "nll_delta" in baseline else 0.0,
    }
    focus = _focus_position_summary(candidate=candidate, baseline=baseline, focus_position=focus_position)
    candidate_memory_clean = _memory_clean(candidate)
    baseline_memory_clean = _memory_clean(baseline)
    reasons = _reject_reasons(
        candidate_memory_clean=candidate_memory_clean,
        deltas=deltas,
        focus=focus,
    )
    combined_margin_guard = _combined_margin_guard(
        candidate_memory_clean=candidate_memory_clean,
        baseline_memory_clean=baseline_memory_clean,
        deltas=deltas,
        focus=focus,
    )
    if reasons:
        decision = "reject_direct_route_output_transplant"
        next_action = "learn_logit_objective_or_abandon_direct_transplant"
    else:
        decision = "promising_requires_cross_row_guard"
        next_action = "run_cross_row_report_selection_holdout_guards_before_broadening"
    return {
        "schema_version": 1,
        "record_type": "glm45_air_sparse_plan_eval_analysis",
        "decision": decision,
        "next_action": next_action,
        "candidate_jsonl": str(candidate_jsonl),
        "baseline_jsonl": str(baseline_jsonl),
        "candidate_artifact_dir": candidate.get("artifact_dir"),
        "baseline_artifact_dir": baseline.get("artifact_dir"),
        "prompt_id": candidate.get("prompt_id"),
        "row_index": candidate.get("row_index"),
        "candidate_memory_clean": candidate_memory_clean,
        "baseline_memory_clean": baseline_memory_clean,
        "candidate_pageouts_delta": int(candidate.get("pageouts_delta") or 0),
        "candidate_swapouts_delta": int(candidate.get("swapouts_delta") or 0),
        "baseline_pageouts_delta": int(baseline.get("pageouts_delta") or 0),
        "baseline_swapouts_delta": int(baseline.get("swapouts_delta") or 0),
        "candidate_metrics": {
            "mean_kld": float(candidate["mean_kld"]),
            "top1_agreement": float(candidate["top1_agreement"]),
            "ppl_ratio": float(candidate["ppl_ratio"]),
            "p999_kld": float(candidate["p999_kld"]),
        },
        "baseline_metrics": {
            "mean_kld": float(baseline["mean_kld"]),
            "top1_agreement": float(baseline["top1_agreement"]),
            "ppl_ratio": float(baseline["ppl_ratio"]),
            "p999_kld": float(baseline["p999_kld"]),
        },
        "deltas": deltas,
        "focus_position": focus,
        "combined_margin_guard": combined_margin_guard,
        "reject_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a GLM-4.5-Air sparse-plan candidate eval against an accepted baseline row."
    )
    parser.add_argument("--candidate-jsonl", required=True, type=Path)
    parser.add_argument("--baseline-jsonl", required=True, type=Path)
    parser.add_argument("--prompt-id")
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--focus-position", type=int, default=0)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    report = analyze_sparse_plan_eval(
        candidate_jsonl=args.candidate_jsonl,
        baseline_jsonl=args.baseline_jsonl,
        prompt_id=args.prompt_id,
        row_index=args.row_index,
        focus_position=args.focus_position,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
