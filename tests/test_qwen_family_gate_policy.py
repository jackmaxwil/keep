from __future__ import annotations

from benchmarks.define_qwen_family_gate_policy import define_qwen_family_gate_policy


def test_qwen_family_gate_policy_defines_eval_and_benchmark_without_air_thresholds() -> None:
    payload = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
    )

    assert payload["record_type"] == "qwen_family_gate_policy"
    assert payload["policy_status"] == "qwen_family_policy_defined"
    assert payload["air_thresholds_reused"] is False
    assert payload["eval_gate_defined"] is True
    assert payload["benchmark_gate_defined"] is True
    assert payload["eval_gate"]["required_splits"] == ["report", "selection", "holdout"]
    assert payload["eval_gate"]["minimum_clean_rows_per_split"] == 22
    assert payload["eval_gate"]["minimum_total_clean_rows"] == 64
    assert payload["eval_gate"]["hard_requirements"] == [
        "all_rows_memory_clean",
        "finite_nll_ppl_kld_and_top1_metrics",
        "real_prompt_pack_minimum_64_rows",
        "teacher_cache_model_id_matches_qwen_source",
        "teacher_logits_from_source_model",
        "summary_recomputes_metrics_from_jsonl",
    ]
    assert payload["benchmark_gate"]["required_scenarios"] == [
        "prefill_1k",
        "decode_128",
    ]
    assert payload["benchmark_gate"]["minimum_repetitions_per_scenario"] == 2
    assert (
        payload["benchmark_gate"]["comparison_baseline"]
        == "same_machine_qwen_source_switch_projection_control"
    )
    assert payload["benchmark_gate"]["same_machine_reference"] is True
    assert payload["benchmark_gate"]["maximum_candidate_to_reference_ratio"] == 3.0
    assert payload["benchmark_gate"]["hard_requirements"] == [
        "candidate_and_control_same_scenario",
        "candidate_and_control_clean_pageouts_and_swapouts",
        "finite_latency_rows",
        "comparison_baseline_is_same_machine_reference",
        "summary_reports_candidate_reference_ratio",
    ]
