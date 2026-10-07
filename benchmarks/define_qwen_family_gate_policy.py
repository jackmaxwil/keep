from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def define_qwen_family_gate_policy(
    *,
    model_id: str,
    revision: str,
    minimum_clean_rows_per_split: int = 22,
    minimum_total_clean_rows: int = 64,
    minimum_repetitions_per_scenario: int = 2,
    comparison_baseline: str = "same_machine_qwen_source_switch_projection_control",
    maximum_candidate_to_reference_ratio: float = 3.0,
) -> dict[str, Any]:
    if minimum_clean_rows_per_split <= 0:
        raise ValueError("minimum_clean_rows_per_split must be positive")
    if minimum_total_clean_rows <= 0:
        raise ValueError("minimum_total_clean_rows must be positive")
    if minimum_repetitions_per_scenario <= 0:
        raise ValueError("minimum_repetitions_per_scenario must be positive")
    if not math.isfinite(maximum_candidate_to_reference_ratio) or (
        maximum_candidate_to_reference_ratio <= 0.0
    ):
        raise ValueError("maximum_candidate_to_reference_ratio must be finite and positive")
    comparison_baseline = comparison_baseline.strip()
    if not comparison_baseline:
        raise ValueError("comparison_baseline must be non-empty")
    same_machine_reference = "same_machine" in comparison_baseline.replace("-", "_").lower()
    return {
        "record_type": "qwen_family_gate_policy",
        "policy_status": "qwen_family_policy_defined",
        "model_id": model_id,
        "revision": revision,
        "model_type": "qwen3_5_moe",
        "template_source": "docs/new-model-family.md",
        "air_thresholds_reused": False,
        "eval_gate_defined": True,
        "benchmark_gate_defined": True,
        "eval_gate": {
            "gate_id": "qwen3_5_moe_teacher_cache_eval_v1",
            "required_splits": ["report", "selection", "holdout"],
            "minimum_clean_rows_per_split": minimum_clean_rows_per_split,
            "minimum_total_clean_rows": minimum_total_clean_rows,
            "hard_requirements": [
                "all_rows_memory_clean",
                "finite_nll_ppl_kld_and_top1_metrics",
                "real_prompt_pack_minimum_64_rows",
                "teacher_cache_model_id_matches_qwen_source",
                "teacher_logits_from_source_model",
                "summary_recomputes_metrics_from_jsonl",
            ],
            "quality_thresholds": {
                "absolute_qwen_thresholds_deferred_until_source_baseline_exists": True,
                "dirty_rows_accepted": False,
            },
        },
        "benchmark_gate": {
            "gate_id": "qwen3_5_moe_quant_compare_v1",
            "required_scenarios": ["prefill_1k", "decode_128"],
            "minimum_repetitions_per_scenario": minimum_repetitions_per_scenario,
            "hard_requirements": [
                "candidate_and_control_same_scenario",
                "candidate_and_control_clean_pageouts_and_swapouts",
                "finite_latency_rows",
                "comparison_baseline_is_same_machine_reference",
                "summary_reports_candidate_reference_ratio",
            ],
            "comparison_baseline": comparison_baseline,
            "same_machine_reference": same_machine_reference,
            "ratio_metric": (
                "candidate_median_ms / same_machine_source_control_median_ms"
            ),
            "maximum_candidate_to_reference_ratio": maximum_candidate_to_reference_ratio,
        },
        "required_evidence": [
            "qwen_report_eval_jsonl",
            "qwen_selection_eval_jsonl",
            "qwen_holdout_eval_jsonl",
            "qwen_quant_compare_jsonl",
            "qwen_eval_summary_json",
            "qwen_benchmark_summary_json",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Define Qwen family-specific eval and benchmark gate policy."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--minimum-clean-rows-per-split", type=int, default=22)
    parser.add_argument("--minimum-total-clean-rows", type=int, default=64)
    parser.add_argument("--minimum-repetitions-per-scenario", type=int, default=2)
    parser.add_argument(
        "--comparison-baseline",
        default="same_machine_qwen_source_switch_projection_control",
    )
    parser.add_argument("--maximum-candidate-to-reference-ratio", type=float, default=3.0)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = define_qwen_family_gate_policy(
        model_id=args.model_id,
        revision=args.revision,
        minimum_clean_rows_per_split=args.minimum_clean_rows_per_split,
        minimum_total_clean_rows=args.minimum_total_clean_rows,
        minimum_repetitions_per_scenario=args.minimum_repetitions_per_scenario,
        comparison_baseline=args.comparison_baseline,
        maximum_candidate_to_reference_ratio=args.maximum_candidate_to_reference_ratio,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
