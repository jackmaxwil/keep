from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_analyzer():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "analyze_glm45_air_top1_gap.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_glm45_air_top1_gap", module_path)
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


def _row(prompt_id: str, split: str) -> dict:
    return {
        "artifact_dir": f"{split}-artifact",
        "mean_kld": 0.22,
        "nll_delta": -0.08,
        "pageouts_delta": 0,
        "ppl_ratio": 0.92,
        "p999_kld": 4.23,
        "positions": [0, 1],
        "prompt_id": prompt_id,
        "row_index": 120,
        "swapouts_delta": 0,
        "target_token_ids": [279, 5588],
        "teacher_top1_ids": [565, 2701],
        "token_klds": [4.23, 0.2],
        "top1_agreement": 0.72,
        "vq_target_logprobs": [-8.42, -11.2],
        "vq_teacher_top1_logprobs": [-4.93, -1.0],
        "vq_teacher_top1_margin_vs_vq_top1": [-2.59, 0.0],
        "vq_top1_ids": [220, 2701],
        "vq_top1_logprobs": [-2.34, -1.0],
        "vq_watch_token_ids": [565],
        "vq_watch_token_logprobs": {"565": [-4.93, -2.0]},
        "vq_watch_token_margin_vs_vq_top1": {"565": [-2.59, -1.0]},
        "vq_watch_token_required_bias_to_vq_top1": {"565": [2.59, 1.0]},
    }


def test_top1_gap_analysis_detects_stable_wrong_attractor(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    report = tmp_path / "report.jsonl"
    select = tmp_path / "select.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    _write_jsonl(report, [_row("report_route_000", "report"), _row("report_route_001", "report")])
    _write_jsonl(select, [_row("select_route_000", "select"), _row("select_route_001", "select")])
    _write_jsonl(holdout, [_row("holdout_route_000", "holdout"), _row("holdout_route_001", "holdout")])

    result = analyzer.analyze_top1_gap(
        jsonls=[report, select, holdout],
        focus_position=0,
        target_rows=[120],
        label="route4-froms6",
    )

    assert result["record_type"] == "glm45_air_top1_gap_analysis"
    assert result["label"] == "route4-froms6"
    assert result["row_count"] == 6
    assert result["memory_clean"] is True
    assert result["teacher_top1_id"] == 565
    assert result["stable_wrong_top1_id"] == 220
    assert result["stable_wrong_top1_count"] == 6
    assert result["top1_flip_count"] == 0
    assert result["mean_teacher_margin"] == -2.59
    assert result["max_teacher_margin"] == -2.59
    assert result["decision"] == "stable_wrong_attractor_needs_new_mechanism"
    assert result["next_action"] == "target_teacher_vs_stable_wrong_gap_directly"


def test_top1_gap_analysis_marks_ready_when_focus_flips(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    candidate = tmp_path / "candidate.jsonl"
    row = _row("report_route_000", "report")
    row["vq_top1_ids"] = [565, 2701]
    row["vq_teacher_top1_margin_vs_vq_top1"] = [0.0, 0.0]
    _write_jsonl(candidate, [row])

    result = analyzer.analyze_top1_gap(jsonls=[candidate], focus_position=0)

    assert result["top1_flip_count"] == 1
    assert result["stable_wrong_top1_id"] is None
    assert result["decision"] == "focus_top1_flip_observed"
    assert result["next_action"] == "run_cross_split_guard"


def test_top1_gap_analysis_reports_watch_token_bias_collateral(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate, [_row("report_route_000", "report")])

    result = analyzer.analyze_top1_gap(
        jsonls=[candidate],
        focus_position=0,
        watch_token_id=565,
    )

    assert result["watch_token_id"] == 565
    assert result["watch_token_focus_required_bias_max"] == 2.59
    assert result["watch_token_same_row_collateral_count"] == 1
    assert result["watch_token_same_row_collateral_positions"][0]["position"] == 1
    assert result["watch_token_same_row_collateral_positions"][0]["required_bias_to_vq_top1"] == 1.0
    assert result["watch_token_decision"] == "watch_token_global_bias_collateral_risk"
    assert result["watch_token_next_action"] == "prefer_position_or_mechanism_specific_bias"


def test_watch_token_full_split_collateral_compares_baseline_candidate_positions(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_rows = [
        {
            **_row("report_route_000", "baseline"),
            "row_index": 0,
            "positions": [0, 1, 2],
            "teacher_top1_ids": [565, 20, 30],
            "vq_top1_ids": [111, 20, 30],
            "token_klds": [2.0, 0.1, 0.1],
            "top1_agreement": 2 / 3,
            "mean_kld": 0.7333333333333334,
            "p999_kld": 2.0,
            "ppl_ratio": 1.0,
        },
        {
            **_row("report_math_001", "baseline"),
            "row_index": 1,
            "positions": [0, 1],
            "teacher_top1_ids": [40, 50],
            "vq_top1_ids": [40, 51],
            "token_klds": [0.1, 1.0],
            "top1_agreement": 0.5,
            "mean_kld": 0.55,
            "p999_kld": 1.0,
            "ppl_ratio": 1.0,
        },
        {"record_type": "air_vq_teacher_cache_eval_summary", "record_count": 2},
    ]
    candidate_rows = [
        {
            **baseline_rows[0],
            "artifact_dir": "candidate-artifact",
            "vq_top1_ids": [565, 565, 30],
            "token_klds": [0.5, 1.2, 0.1],
            "top1_agreement": 2 / 3,
            "mean_kld": 0.6,
            "p999_kld": 1.2,
            "vq_watch_token_ids": [565],
        },
        {
            **baseline_rows[1],
            "artifact_dir": "candidate-artifact",
            "vq_top1_ids": [565, 50],
            "token_klds": [2.5, 0.2],
            "top1_agreement": 0.5,
            "mean_kld": 1.35,
            "p999_kld": 2.5,
            "pageouts_delta": 9,
            "vq_watch_token_ids": [565],
        },
    ]
    _write_jsonl(baseline_path, baseline_rows)
    _write_jsonl(candidate_path, candidate_rows)

    result = analyzer.analyze_watch_token_full_split(
        baseline_jsonls=[baseline_path],
        candidate_jsonls=[candidate_path],
        watch_token_id=565,
        label="bias565",
    )

    assert result["record_type"] == "glm45_air_watch_token_full_split_collateral"
    assert result["label"] == "bias565"
    assert result["paired_row_count"] == 2
    assert result["memory_clean_paired_row_count"] == 1
    assert result["all_rows"]["position_top1_delta"] == 0.0
    assert result["all_rows"]["mean_kld_delta"] > 0.0
    assert result["all_rows"]["watch_teacher_top1_gain_count"] == 1
    assert result["all_rows"]["non_watch_teacher_top1_regression_count"] == 2
    assert result["all_rows"]["false_watch_top1_takeover_count"] == 2
    assert result["memory_clean_rows"]["non_watch_teacher_top1_regression_count"] == 1
    assert result["memory_clean_rows"]["false_watch_top1_takeover_count"] == 1
    assert result["decision"] == "watch_token_global_bias_collateral_risk"
    assert result["next_action"] == "scope_bias_before_selection_holdout"


def test_watch_token_full_split_collateral_can_filter_candidate_bias_positions(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_rows = [
        {
            **_row("report_route_000", "baseline"),
            "row_index": 0,
            "positions": [0, 1],
            "teacher_top1_ids": [565, 20],
            "vq_top1_ids": [111, 20],
            "token_klds": [2.0, 0.1],
            "top1_agreement": 0.5,
            "mean_kld": 1.05,
            "p999_kld": 2.0,
            "ppl_ratio": 1.0,
        }
    ]
    candidate_rows = [
        {
            **baseline_rows[0],
            "artifact_dir": "candidate-artifact",
            "vq_top1_ids": [565, 565],
            "token_klds": [0.5, 1.2],
            "top1_agreement": 0.5,
            "mean_kld": 0.85,
            "p999_kld": 1.2,
            "vq_watch_token_ids": [565],
        }
    ]
    _write_jsonl(baseline_path, baseline_rows)
    _write_jsonl(candidate_path, candidate_rows)

    result = analyzer.analyze_watch_token_full_split(
        baseline_jsonls=[baseline_path],
        candidate_jsonls=[candidate_path],
        watch_token_id=565,
        candidate_bias_position_indices=[0],
        label="bias565-position0",
    )

    assert result["candidate_bias_position_indices"] == [0]
    assert result["paired_row_count"] == 1
    assert result["all_rows"]["position_count"] == 1
    assert result["all_rows"]["watch_teacher_top1_gain_count"] == 1
    assert result["all_rows"]["non_watch_teacher_top1_regression_count"] == 0
    assert result["decision"] == "watch_token_global_bias_full_split_viable_probe"
    assert result["next_action"] == "run_selection_holdout_guard"


def test_select_tail_cleanup_targets_keeps_holdout_validation_only(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    report_path = tmp_path / "report.jsonl"
    selection_path = tmp_path / "selection.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    report_rows = []
    selection_rows = []
    holdout_rows = []
    for row_index, domain, kld in [(73, "instruction", 4.2), (44, "math", 4.1), (120, "route", 4.05), (75, "instruction", 4.0)]:
        row = {
            **_row(f"report_{domain}_{row_index}", "report"),
            "row_index": row_index,
            "teacher_top1_ids": [565, 20],
            "vq_top1_ids": [261, 20],
            "token_klds": [kld, 0.2],
            "target_token_ids": [16812, 1],
            "vq_target_logprobs": [-20.0, -0.5],
            "vq_teacher_top1_margin_vs_vq_top1": [-0.4, 0.0],
        }
        report_rows.append(row)
    for row_index, domain, kld in [(73, "instruction", 4.2), (44, "math", 4.1)]:
        row = {
            **_row(f"selection_{domain}_{row_index}", "selection"),
            "row_index": row_index,
            "teacher_top1_ids": [565, 20],
            "vq_top1_ids": [261, 20],
            "token_klds": [kld, 0.2],
            "target_token_ids": [16812, 1],
            "vq_target_logprobs": [-20.0, -0.5],
            "vq_teacher_top1_margin_vs_vq_top1": [-0.4, 0.0],
        }
        selection_rows.append(row)
    for row_index, domain, kld in [(73, "instruction", 4.2), (44, "math", 4.1)]:
        row = {
            **_row(f"holdout_{domain}_{row_index}", "holdout"),
            "row_index": row_index,
            "teacher_top1_ids": [565, 20],
            "vq_top1_ids": [261, 20],
            "token_klds": [kld, 0.2],
            "target_token_ids": [16812, 1],
            "vq_target_logprobs": [-20.0, -0.5],
            "vq_teacher_top1_margin_vs_vq_top1": [-0.4, 0.0],
        }
        holdout_rows.append(row)
    _write_jsonl(report_path, report_rows)
    _write_jsonl(selection_path, selection_rows)
    _write_jsonl(holdout_path, holdout_rows)

    result = analyzer.select_tail_cleanup_targets(
        jsonls=[report_path, selection_path, holdout_path],
        teacher_token_id=565,
        focus_position=0,
        max_train_rows=3,
        max_validation_rows=3,
        top_tail_records=8,
        label="route4-mix-tail",
    )

    assert result["record_type"] == "glm45_air_tail_cleanup_target_selection"
    assert result["label"] == "route4-mix-tail"
    assert result["decision"] == "tail_cleanup_report_train_packet_ready"
    assert result["recommended_training"]["train_cache"] == "validation"
    assert result["recommended_training"]["max_positions"] == 1
    assert result["recommended_training"]["aux_loss_position_indices"] == [0]
    assert result["recommended_training"]["train_row_indices"] == [73, 44, 120]
    assert all(record["split"] == "report" for record in result["recommended_training"]["train_rows"])
    assert {record["domain"] for record in result["recommended_training"]["train_rows"]} == {
        "instruction",
        "math",
        "route",
    }
    assert all(record["split"] in {"selection", "holdout"} for record in result["validation_focus_rows"])
    assert any(record["split"] == "holdout" for record in result["validation_focus_rows"])
    assert result["top_tail_domain_counts"]["holdout:instruction"] == 1
    assert result["top_tail_vq_counts"]["261"] == 8


def test_simulate_watch_token_bias_delta_reports_top1_and_kld_bounds(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    candidate_path = tmp_path / "candidate.jsonl"
    row = {
        **_row("report_instruction_001", "candidate"),
        "row_index": 73,
        "positions": [0, 1],
        "teacher_top1_ids": [565, 20],
        "vq_top1_ids": [261, 20],
        "token_klds": [4.2, 0.2],
        "vq_top1_logprobs": [-3.16, -0.5],
        "vq_watch_token_logprobs": {"565": [-3.54, -10.0]},
        "vq_watch_token_margin_vs_vq_top1": {"565": [-0.38, -9.5]},
        "top1_agreement": 0.5,
    }
    _write_jsonl(candidate_path, [row])

    result = analyzer.simulate_watch_token_bias_delta(
        jsonls=[candidate_path],
        watch_token_id=565,
        extra_bias=0.39,
        bias_position_indices=[0],
        label="extra565",
    )

    assert result["record_type"] == "glm45_air_watch_token_bias_delta_simulation"
    assert result["label"] == "extra565"
    assert result["kl_assumption"] == "bounds_only_teacher_watch_probability_unavailable"
    assert result["position_count"] == 2
    assert result["biased_position_count"] == 1
    assert result["teacher_watch_position_count"] == 1
    assert result["current_top1"] == 0.5
    assert result["simulated_top1"] == 1.0
    assert result["watch_top1_gain_count"] == 1
    assert result["false_watch_top1_takeover_count"] == 0
    assert result["best_case_p999_kld"] < result["current_p999_kld"]
    assert result["worst_case_p999_kld"] > result["best_case_p999_kld"]
    assert result["decision"] == "extra_bias_cannot_clear_p999_even_best_case"
    assert result["next_action"] == "use_non_bias_tail_cleanup_or_training_signal"


def test_simulate_watch_token_bias_delta_rejects_false_takeover(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    candidate_path = tmp_path / "candidate.jsonl"
    row = {
        **_row("report_control_000", "candidate"),
        "row_index": 0,
        "positions": [0],
        "teacher_top1_ids": [42],
        "vq_top1_ids": [42],
        "token_klds": [0.1],
        "vq_top1_logprobs": [-1.0],
        "vq_watch_token_logprobs": {"565": [-1.1]},
        "vq_watch_token_margin_vs_vq_top1": {"565": [-0.1]},
        "top1_agreement": 1.0,
    }
    _write_jsonl(candidate_path, [row])

    result = analyzer.simulate_watch_token_bias_delta(
        jsonls=[candidate_path],
        watch_token_id=565,
        extra_bias=0.2,
        bias_position_indices=[0],
    )

    assert result["simulated_top1"] == 0.0
    assert result["false_watch_top1_takeover_count"] == 1
    assert result["non_watch_teacher_top1_regression_count"] == 1
    assert result["decision"] == "extra_bias_collateral_risk"
    assert result["next_action"] == "do_not_materialize_without_tighter_scope"
