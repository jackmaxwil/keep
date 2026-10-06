from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_selector():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "select_glm45_air_sparse_plan_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("select_glm45_air_sparse_plan_candidates", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _analysis(
    *,
    artifact: str,
    guard_decision: str,
    score: float,
    failed_checks: list[str],
    prompt_id: str = "report_route_000",
    ready: bool = False,
    training_signal: bool = False,
    row_index: int = 120,
    margin_delta: float = 1.0,
    target_logprob_delta: float = 0.1,
    token_kld_delta: float = -0.2,
    top1_delta: float = 0.0,
    nll_delta: float = 0.0,
    ppl_delta: float = 0.0,
) -> dict:
    return {
        "schema_version": 1,
        "record_type": "glm45_air_sparse_plan_eval_analysis",
        "candidate_artifact_dir": artifact,
        "candidate_jsonl": f"artifacts/quality/{artifact}.jsonl",
        "baseline_jsonl": "artifacts/quality/baseline.jsonl",
        "prompt_id": prompt_id,
        "row_index": row_index,
        "candidate_memory_clean": True,
        "baseline_memory_clean": True,
        "deltas": {
            "mean_kld": -0.01,
            "nll_delta": nll_delta,
            "p999_kld": -0.3,
            "ppl_ratio": ppl_delta,
            "top1_agreement": top1_delta,
        },
        "focus_position": {
            "position": 0,
            "teacher_top1_id": 565,
            "baseline_vq_top1_id": 82,
            "candidate_vq_top1_id": 82,
            "top1_flipped_to_teacher": ready,
            "vq_teacher_top1_margin_delta": margin_delta,
            "vq_target_logprob_delta": target_logprob_delta,
            "token_kld_delta": token_kld_delta,
        },
        "combined_margin_guard": {
            "decision": guard_decision,
            "score": score,
            "teacher_top1_margin_improved": margin_delta > 0.0,
            "target_logprob_preserved": target_logprob_delta >= 0.0,
            "token_kld_improved": token_kld_delta <= 0.0,
            "row_top1_non_regressed": top1_delta >= 0.0,
            "row_nll_preserved": nll_delta <= 0.0,
            "row_ppl_preserved": ppl_delta <= 0.0,
            "row_mean_kld_preserved": True,
            "row_p999_kld_preserved": True,
            "focus_top1_flipped": ready,
            "training_signal": training_signal,
            "ready_for_broader_eval": ready,
            "failed_checks": failed_checks,
        },
    }


def test_sparse_plan_candidate_selector_ranks_ready_partial_then_reject(tmp_path) -> None:
    selector = _load_selector()
    ready_path = tmp_path / "ready.json"
    partial_path = tmp_path / "partial.json"
    reject_path = tmp_path / "reject.json"
    _write_json(
        ready_path,
        _analysis(
            artifact="ready",
            guard_decision="candidate_ready_for_cross_row_guard",
            ready=True,
            score=2.0,
            failed_checks=[],
            top1_delta=0.03,
        ),
    )
    _write_json(
        partial_path,
        _analysis(
            artifact="partial",
            guard_decision="partial_margin_signal_needs_row_preservation_or_top1_guard",
            score=3.0,
            failed_checks=["row_nll_regressed", "focus_top1_not_flipped"],
            nll_delta=0.01,
        ),
    )
    _write_json(
        reject_path,
        _analysis(
            artifact="reject",
            guard_decision="reject_margin_overfit",
            score=10.0,
            failed_checks=["target_logprob_not_preserved", "row_top1_regressed"],
            target_logprob_delta=-0.5,
            top1_delta=-0.02,
        ),
    )

    report = selector.select_sparse_plan_candidates([reject_path, partial_path, ready_path])

    assert report["record_type"] == "glm45_air_sparse_plan_candidate_selection"
    assert report["decision"] == "candidate_ready_for_cross_row_guard"
    assert [row["candidate_artifact_dir"] for row in report["ranked_candidates"]] == [
        "ready",
        "partial",
        "reject",
    ]
    assert report["selected_candidate"]["candidate_artifact_dir"] == "ready"
    assert report["status_counts"] == {"ready": 1, "partial": 1, "reject": 1}


def test_sparse_plan_candidate_selector_keeps_partial_signal_but_rejects_overfit(tmp_path) -> None:
    selector = _load_selector()
    partial_path = tmp_path / "s2.json"
    overfit_path = tmp_path / "s4.json"
    _write_json(
        partial_path,
        _analysis(
            artifact="s2",
            guard_decision="partial_margin_signal_needs_row_preservation_or_top1_guard",
            score=1.8,
            failed_checks=["row_nll_regressed", "row_ppl_regressed", "focus_top1_not_flipped"],
            margin_delta=1.3462705612182617,
            target_logprob_delta=0.10800868148201026,
            token_kld_delta=-0.36480869884067246,
            nll_delta=0.0027529508830053118,
            ppl_delta=0.002527008181499646,
        ),
    )
    _write_json(
        overfit_path,
        _analysis(
            artifact="s4",
            guard_decision="reject_margin_overfit",
            score=2.1,
            failed_checks=[
                "target_logprob_not_preserved",
                "row_top1_regressed",
                "row_nll_regressed",
                "row_ppl_regressed",
                "focus_top1_not_flipped",
            ],
            margin_delta=1.9515542984008785,
            target_logprob_delta=-1.287880684024337,
            token_kld_delta=-0.24292856577676147,
            top1_delta=-0.023255813953488302,
            nll_delta=0.03883457803513224,
            ppl_delta=0.03629852304764214,
        ),
    )

    report = selector.select_sparse_plan_candidates([overfit_path, partial_path])

    assert report["decision"] == "partial_signal_requires_row_preservation"
    assert report["selected_candidate"]["candidate_artifact_dir"] == "s2"
    assert report["selected_candidate"]["selector_status"] == "partial"
    assert report["selected_candidate"]["repair_checks"] == [
        "row_nll_regressed",
        "row_ppl_regressed",
        "focus_top1_not_flipped",
    ]
    assert report["rejected_candidates"][0]["candidate_artifact_dir"] == "s4"
    assert report["next_action"] == "refine_partial_candidate_before_broad_eval"


def test_sparse_plan_candidate_selector_reports_safer_artifact_level_partial(tmp_path) -> None:
    selector = _load_selector()
    safe_paths = []
    for split, score in [("report", 2.0), ("select", 2.2), ("holdout", 2.1)]:
        path = tmp_path / f"safe-{split}.json"
        _write_json(
            path,
            _analysis(
                artifact="safe",
                prompt_id=f"{split}_route_000",
                guard_decision="margin_signal_needs_top1_flip_or_cross_row_guard",
                score=score,
                failed_checks=["focus_top1_not_flipped"],
                training_signal=True,
            ),
        )
        safe_paths.append(path)
    risky_paths = []
    for split, failed_checks, score, nll_delta, ppl_delta in [
        ("report", ["focus_top1_not_flipped"], 2.3, -0.001, -0.001),
        ("select", ["focus_top1_not_flipped"], 2.4, -0.002, -0.002),
        ("holdout", ["row_nll_regressed", "row_ppl_regressed", "focus_top1_not_flipped"], 2.5, 0.006, 0.005),
    ]:
        path = tmp_path / f"risky-{split}.json"
        _write_json(
            path,
            _analysis(
                artifact="risky",
                prompt_id=f"{split}_route_000",
                guard_decision="partial_margin_signal_needs_row_preservation_or_top1_guard",
                score=score,
                failed_checks=failed_checks,
                training_signal="row_nll_regressed" not in failed_checks,
                nll_delta=nll_delta,
                ppl_delta=ppl_delta,
            ),
        )
        risky_paths.append(path)

    report = selector.select_sparse_plan_candidates([*risky_paths, *safe_paths])

    assert report["selected_candidate"]["candidate_artifact_dir"] == "risky"
    assert report["selected_artifact_candidate"]["candidate_artifact_dir"] == "safe"
    assert report["selected_artifact_candidate"]["artifact_failed_checks"] == ["focus_top1_not_flipped"]
    assert report["selected_artifact_candidate"]["split_count"] == 3
