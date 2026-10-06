from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    payload["_source_json"] = str(path)
    return payload


def _guard(analysis: dict[str, Any]) -> dict[str, Any]:
    guard = analysis.get("combined_margin_guard")
    if not isinstance(guard, dict):
        raise ValueError(f"{analysis.get('_source_json', '<analysis>')}: missing combined_margin_guard")
    return guard


def _classify_candidate(analysis: dict[str, Any]) -> str:
    guard = _guard(analysis)
    decision = str(guard.get("decision") or "")
    if bool(guard.get("ready_for_broader_eval")) or decision == "candidate_ready_for_cross_row_guard":
        return "ready"
    if bool(guard.get("training_signal")) or decision in {
        "margin_signal_needs_top1_flip_or_cross_row_guard",
        "partial_margin_signal_needs_row_preservation_or_top1_guard",
    }:
        return "partial"
    return "reject"


def _rank_key(candidate: dict[str, Any]) -> tuple[int, float, str]:
    tier = {"ready": 0, "partial": 1, "reject": 2}[candidate["selector_status"]]
    return (tier, -float(candidate["guard_score"]), str(candidate.get("candidate_artifact_dir") or ""))


def _candidate_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    guard = _guard(analysis)
    focus = analysis.get("focus_position") if isinstance(analysis.get("focus_position"), dict) else {}
    deltas = analysis.get("deltas") if isinstance(analysis.get("deltas"), dict) else {}
    failed_checks = list(guard.get("failed_checks") or [])
    status = _classify_candidate(analysis)
    repair_checks = failed_checks if status == "partial" else []
    return {
        "selector_status": status,
        "source_json": analysis.get("_source_json"),
        "candidate_artifact_dir": analysis.get("candidate_artifact_dir"),
        "candidate_jsonl": analysis.get("candidate_jsonl"),
        "baseline_jsonl": analysis.get("baseline_jsonl"),
        "prompt_id": analysis.get("prompt_id"),
        "row_index": analysis.get("row_index"),
        "focus_position": focus.get("position"),
        "guard_decision": guard.get("decision"),
        "guard_score": float(guard.get("score") or 0.0),
        "ready_for_broader_eval": bool(guard.get("ready_for_broader_eval")),
        "training_signal": bool(guard.get("training_signal")),
        "failed_checks": failed_checks,
        "repair_checks": repair_checks,
        "margin_delta": focus.get("vq_teacher_top1_margin_delta"),
        "target_logprob_delta": focus.get("vq_target_logprob_delta"),
        "token_kld_delta": focus.get("token_kld_delta"),
        "top1_delta": deltas.get("top1_agreement"),
        "nll_delta": deltas.get("nll_delta"),
        "ppl_delta": deltas.get("ppl_ratio"),
        "mean_kld_delta": deltas.get("mean_kld"),
        "p999_kld_delta": deltas.get("p999_kld"),
    }


def _artifact_status(candidates: list[dict[str, Any]]) -> str:
    if any(candidate["selector_status"] == "ready" for candidate in candidates):
        return "ready"
    if any(candidate["selector_status"] == "partial" for candidate in candidates):
        return "partial"
    return "reject"


def _artifact_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    artifact = str(candidates[0].get("candidate_artifact_dir") or "")
    status = _artifact_status(candidates)
    failed_checks = sorted({check for candidate in candidates for check in candidate.get("failed_checks", [])})
    blocking_checks = [check for check in failed_checks if check != "focus_top1_not_flipped"]
    guard_scores = [float(candidate.get("guard_score") or 0.0) for candidate in candidates]
    return {
        "candidate_artifact_dir": artifact,
        "selector_status": status,
        "split_count": len(candidates),
        "artifact_failed_checks": failed_checks,
        "artifact_blocking_checks": blocking_checks,
        "min_guard_score": min(guard_scores),
        "max_guard_score": max(guard_scores),
        "mean_guard_score": sum(guard_scores) / len(guard_scores),
        "ready_for_broader_eval": status == "ready" and not blocking_checks,
        "training_signal": any(bool(candidate.get("training_signal")) for candidate in candidates),
        "candidate_jsonls": [candidate.get("candidate_jsonl") for candidate in candidates],
        "source_jsons": [candidate.get("source_json") for candidate in candidates],
        "prompt_ids": [candidate.get("prompt_id") for candidate in candidates],
        "row_indices": [candidate.get("row_index") for candidate in candidates],
    }


def _artifact_rank_key(candidate: dict[str, Any]) -> tuple[int, int, float, str]:
    tier = {"ready": 0, "partial": 1, "reject": 2}[candidate["selector_status"]]
    return (
        tier,
        len(candidate.get("artifact_blocking_checks") or []),
        -float(candidate.get("min_guard_score") or 0.0),
        str(candidate.get("candidate_artifact_dir") or ""),
    )


def _rank_artifact_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_artifact: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        artifact = str(candidate.get("candidate_artifact_dir") or "")
        by_artifact.setdefault(artifact, []).append(candidate)
    return sorted((_artifact_summary(rows) for rows in by_artifact.values()), key=_artifact_rank_key)


def select_sparse_plan_candidates(analysis_jsons: list[Path]) -> dict[str, Any]:
    if not analysis_jsons:
        raise ValueError("at least one analysis JSON is required")
    candidates = [_candidate_summary(_read_json(path)) for path in analysis_jsons]
    ranked = sorted(candidates, key=_rank_key)
    status_counts = {"ready": 0, "partial": 0, "reject": 0}
    for candidate in ranked:
        status_counts[candidate["selector_status"]] += 1
    artifact_ranked = _rank_artifact_candidates(ranked)

    selected = next((candidate for candidate in ranked if candidate["selector_status"] != "reject"), None)
    selected_artifact = next(
        (candidate for candidate in artifact_ranked if candidate["selector_status"] != "reject"),
        None,
    )
    if selected is None:
        decision = "all_candidates_rejected"
        next_action = "change_objective_or_candidate_family"
    elif selected["selector_status"] == "ready":
        decision = "candidate_ready_for_cross_row_guard"
        next_action = "run_clean_cross_row_report_selection_holdout_guards"
    else:
        decision = "partial_signal_requires_row_preservation"
        next_action = "refine_partial_candidate_before_broad_eval"

    return {
        "schema_version": 1,
        "record_type": "glm45_air_sparse_plan_candidate_selection",
        "decision": decision,
        "next_action": next_action,
        "analysis_jsons": [str(path) for path in analysis_jsons],
        "status_counts": status_counts,
        "selected_candidate": selected,
        "selected_artifact_candidate": selected_artifact,
        "ranked_candidates": ranked,
        "ranked_artifact_candidates": artifact_ranked,
        "rejected_candidates": [candidate for candidate in ranked if candidate["selector_status"] == "reject"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank GLM-4.5-Air sparse-plan candidate analyses by combined margin guard status."
    )
    parser.add_argument("--analysis-json", required=True, type=Path, action="append")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    report = select_sparse_plan_candidates(args.analysis_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
