from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_analyzer():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "analyze_glm45_air_sparse_plan_eval.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_glm45_air_sparse_plan_eval", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(**overrides) -> dict:
    row = {
        "row_index": 120,
        "prompt_id": "report_route_000",
        "artifact_dir": "artifact",
        "mean_kld": 0.23,
        "top1_agreement": 0.79,
        "ppl_ratio": 0.92,
        "p999_kld": 4.45,
        "pageouts_delta": 0,
        "swapouts_delta": 0,
        "positions": [0, 1],
        "target_token_ids": [279, 5588],
        "teacher_top1_ids": [565, 2701],
        "vq_top1_ids": [82, 2701],
        "token_klds": [4.61, 0.22],
        "teacher_target_logprobs": [-4.4, -11.1],
        "vq_target_logprobs": [-8.65, -11.2],
        "vq_teacher_top1_logprobs": [-9.0, -0.5],
        "vq_top1_logprobs": [-4.0, -0.5],
        "vq_teacher_top1_margin_vs_vq_top1": [-5.0, 0.0],
    }
    row.update(overrides)
    return row


def test_sparse_plan_eval_analysis_rejects_dirty_worse_no_flip_candidate(tmp_path) -> None:
    analyzer = _load_analyzer()
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(
        baseline_path,
        [
            _row(
                artifact_dir="accepted",
                mean_kld=0.22954924968181412,
                top1_agreement=0.7906976744186046,
                ppl_ratio=0.9166641614773223,
                p999_kld=4.449225386999793,
                token_klds=[4.612546822034927, 0.22172624829347443],
                vq_target_logprobs=[-8.65489019572885, -11.211641025805955],
            )
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            _row(
                artifact_dir="sparse-plan",
                mean_kld=0.2302011944580066,
                top1_agreement=0.7906976744186046,
                ppl_ratio=0.9189750939955827,
                p999_kld=4.471762060612188,
                pageouts_delta=5,
                token_klds=[4.635989215567724, 0.22141845047420317],
                vq_target_logprobs=[-8.660662088729282, -11.209657678377422],
            )
        ],
    )

    report = analyzer.analyze_sparse_plan_eval(
        candidate_jsonl=candidate_path,
        baseline_jsonl=baseline_path,
        prompt_id="report_route_000",
        focus_position=0,
    )

    assert report["record_type"] == "glm45_air_sparse_plan_eval_analysis"
    assert report["decision"] == "reject_direct_route_output_transplant"
    assert report["candidate_memory_clean"] is False
    assert report["deltas"]["top1_agreement"] == 0.0
    assert report["deltas"]["p999_kld"] > 0.0
    assert report["focus_position"]["teacher_top1_id"] == 565
    assert report["focus_position"]["baseline_vq_top1_id"] == 82
    assert report["focus_position"]["candidate_vq_top1_id"] == 82
    assert report["focus_position"]["top1_flipped_to_teacher"] is False
    assert report["focus_position"]["token_kld_delta"] > 0.0
    assert "candidate_memory_dirty" in report["reject_reasons"]
    assert "focus_token_kld_regressed" in report["reject_reasons"]


def test_sparse_plan_eval_analysis_reports_teacher_top1_margin_delta(tmp_path) -> None:
    analyzer = _load_analyzer()
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(
        baseline_path,
        [
            _row(
                vq_teacher_top1_logprobs=[-9.0, -0.5],
                vq_top1_logprobs=[-4.0, -0.5],
                vq_teacher_top1_margin_vs_vq_top1=[-5.0, 0.0],
            )
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            _row(
                mean_kld=0.22,
                p999_kld=4.1,
                vq_teacher_top1_logprobs=[-7.5, -0.5],
                vq_top1_logprobs=[-4.2, -0.5],
                vq_teacher_top1_margin_vs_vq_top1=[-3.3, 0.0],
            )
        ],
    )

    report = analyzer.analyze_sparse_plan_eval(
        candidate_jsonl=candidate_path,
        baseline_jsonl=baseline_path,
        prompt_id="report_route_000",
        focus_position=0,
    )

    focus = report["focus_position"]
    assert focus["baseline_vq_teacher_top1_logprob"] == -9.0
    assert focus["candidate_vq_teacher_top1_logprob"] == -7.5
    assert focus["baseline_vq_top1_logprob"] == -4.0
    assert focus["candidate_vq_top1_logprob"] == -4.2
    assert focus["baseline_vq_teacher_top1_margin_vs_vq_top1"] == -5.0
    assert focus["candidate_vq_teacher_top1_margin_vs_vq_top1"] == -3.3
    assert abs(focus["vq_teacher_top1_margin_delta"] - 1.7) < 1e-12
    assert focus["vq_teacher_top1_margin_improved"] is True


def test_sparse_plan_eval_analysis_marks_guarded_margin_signal_without_flip(tmp_path) -> None:
    analyzer = _load_analyzer()
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(
        baseline_path,
        [
            _row(
                nll_delta=0.10,
                vq_teacher_top1_logprobs=[-9.0, -0.5],
                vq_top1_logprobs=[-4.0, -0.5],
                vq_teacher_top1_margin_vs_vq_top1=[-5.0, 0.0],
            )
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            _row(
                mean_kld=0.22,
                top1_agreement=0.79,
                ppl_ratio=0.91,
                p999_kld=4.1,
                nll_delta=0.08,
                token_klds=[4.0, 0.20],
                vq_target_logprobs=[-8.0, -11.0],
                vq_teacher_top1_logprobs=[-7.5, -0.5],
                vq_top1_logprobs=[-4.2, -0.5],
                vq_teacher_top1_margin_vs_vq_top1=[-3.3, 0.0],
            )
        ],
    )

    report = analyzer.analyze_sparse_plan_eval(
        candidate_jsonl=candidate_path,
        baseline_jsonl=baseline_path,
        prompt_id="report_route_000",
        focus_position=0,
    )

    guard = report["combined_margin_guard"]
    assert guard["decision"] == "margin_signal_needs_top1_flip_or_cross_row_guard"
    assert guard["training_signal"] is True
    assert guard["ready_for_broader_eval"] is False
    assert guard["teacher_top1_margin_improved"] is True
    assert guard["target_logprob_preserved"] is True
    assert guard["row_nll_preserved"] is True
    assert guard["row_top1_non_regressed"] is True
    assert guard["failed_checks"] == ["focus_top1_not_flipped"]
    assert guard["score"] > 0.0


def test_sparse_plan_eval_analysis_rejects_margin_overfit_regressions(tmp_path) -> None:
    analyzer = _load_analyzer()
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(
        baseline_path,
        [
            _row(
                nll_delta=0.10,
                vq_teacher_top1_logprobs=[-9.0, -0.5],
                vq_top1_logprobs=[-4.0, -0.5],
                vq_teacher_top1_margin_vs_vq_top1=[-5.0, 0.0],
            )
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            _row(
                mean_kld=0.24,
                top1_agreement=0.76,
                ppl_ratio=0.95,
                p999_kld=4.8,
                nll_delta=0.14,
                vq_top1_ids=[102376, 2701],
                token_klds=[4.0, 0.20],
                vq_target_logprobs=[-9.5, -11.0],
                vq_teacher_top1_logprobs=[-7.2, -0.5],
                vq_top1_logprobs=[-4.1, -0.5],
                vq_teacher_top1_margin_vs_vq_top1=[-3.1, 0.0],
            )
        ],
    )

    report = analyzer.analyze_sparse_plan_eval(
        candidate_jsonl=candidate_path,
        baseline_jsonl=baseline_path,
        prompt_id="report_route_000",
        focus_position=0,
    )

    guard = report["combined_margin_guard"]
    assert guard["decision"] == "reject_margin_overfit"
    assert guard["training_signal"] is False
    assert guard["ready_for_broader_eval"] is False
    assert guard["teacher_top1_margin_improved"] is True
    assert "target_logprob_not_preserved" in guard["failed_checks"]
    assert "row_top1_regressed" in guard["failed_checks"]
    assert "row_nll_regressed" in guard["failed_checks"]


def test_sparse_plan_eval_analysis_marks_clean_flip_candidate_as_promising(tmp_path) -> None:
    analyzer = _load_analyzer()
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(baseline_path, [_row()])
    _write_jsonl(
        candidate_path,
        [
            _row(
                top1_agreement=0.82,
                p999_kld=3.9,
                mean_kld=0.20,
                ppl_ratio=0.90,
                vq_top1_ids=[565, 2701],
                token_klds=[3.2, 0.20],
                vq_target_logprobs=[-7.1, -11.0],
            )
        ],
    )

    report = analyzer.analyze_sparse_plan_eval(
        candidate_jsonl=candidate_path,
        baseline_jsonl=baseline_path,
        row_index=120,
        focus_position=0,
    )

    assert report["decision"] == "promising_requires_cross_row_guard"
    assert report["candidate_memory_clean"] is True
    assert report["focus_position"]["top1_flipped_to_teacher"] is True
    assert report["reject_reasons"] == []
