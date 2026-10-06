from __future__ import annotations

import numpy as np

from mlx_vq.quality.gates import (
    evaluate_quality_gate,
    validate_ladder_cache_metadata,
    validate_ladder_route_coverage,
)
from mlx_vq.quality.prompts import (
    get_quality_prompts,
    prompt_text_hash,
    validate_quality_prompt_splits,
)
from mlx_vq.quality.teacher_cache import summarize_teacher_cache_records


def _record(
    *,
    prompt_id: str,
    prompt_set: str,
    eval_split: str,
    row_intent: str,
    ppl_ratio: float,
    mean_kld: float,
    token_klds: list[float],
    top1_agreement: float,
    route_coverage: dict[str, dict] | None = None,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt_set": prompt_set,
        "eval_split": eval_split,
        "suite_version": "air_vq_ladder_v1",
        "row_intent": row_intent,
        "prompt_text_sha256": prompt_text_hash(f"prompt {prompt_id}"),
        "nll_delta": float(np.log(ppl_ratio)),
        "ppl_ratio": ppl_ratio,
        "mean_kld": mean_kld,
        "p999_kld": max(token_klds),
        "token_klds": token_klds,
        "top1_agreement": top1_agreement,
        "pageouts_delta": 0,
        "swapouts_delta": 0,
        "route_coverage": route_coverage or {},
    }


def test_air_vq_ladder_prompt_splits_are_large_and_non_leaking() -> None:
    select = get_quality_prompts(prompt_set="air_vq_ladder_select_v1")
    report = get_quality_prompts(prompt_set="air_vq_ladder_report_v1")

    split_report = validate_quality_prompt_splits()

    assert len(select) == 128
    assert len(report) == 128
    assert all(prompt.prompt_id.startswith("select_") for prompt in select)
    assert all(prompt.prompt_id.startswith("report_") for prompt in report)
    assert split_report["ok"] is True
    assert split_report["duplicate_normalized_text_across_splits"] == []
    assert any(prompt.row_intent == "layer41_glu_code_route" for prompt in select)
    assert any(prompt.row_intent == "layer41_glu_code_route" for prompt in report)
    assert sum(1 for prompt in select if prompt.row_intent == "code_debug") >= 40
    assert sum(1 for prompt in report if prompt.row_intent == "code_debug") >= 40
    assert {prompt.eval_split for prompt in select} == {"selection"}
    assert {prompt.eval_split for prompt in report} == {"report"}


def test_ladder_cache_metadata_requires_split_fields_and_hashes() -> None:
    records = [
        _record(
            prompt_id="select_code_000",
            prompt_set="air_vq_ladder_select_v1",
            eval_split="selection",
            row_intent="code_debug",
            ppl_ratio=1.0,
            mean_kld=0.1,
            token_klds=[0.1],
            top1_agreement=1.0,
        ),
        _record(
            prompt_id="report_code_000",
            prompt_set="air_vq_ladder_report_v1",
            eval_split="report",
            row_intent="code_debug",
            ppl_ratio=1.0,
            mean_kld=0.1,
            token_klds=[0.1],
            top1_agreement=1.0,
        ),
    ]

    assert validate_ladder_cache_metadata(records)["ok"] is True
    broken = [dict(records[0])]
    broken[0].pop("prompt_text_sha256")
    report = validate_ladder_cache_metadata(broken)
    assert report["ok"] is False
    assert "prompt_text_sha256" in report["errors"][0]


def test_teacher_cache_summary_includes_row_distributions() -> None:
    records = [
        _record(
            prompt_id="report_a",
            prompt_set="air_vq_ladder_report_v1",
            eval_split="report",
            row_intent="hard_arithmetic",
            ppl_ratio=1.0,
            mean_kld=0.10,
            token_klds=[0.10],
            top1_agreement=1.0,
        ),
        _record(
            prompt_id="report_b",
            prompt_set="air_vq_ladder_report_v1",
            eval_split="report",
            row_intent="code_debug",
            ppl_ratio=2.0,
            mean_kld=0.30,
            token_klds=[0.30, 0.40],
            top1_agreement=0.5,
        ),
        _record(
            prompt_id="report_c",
            prompt_set="air_vq_ladder_report_v1",
            eval_split="report",
            row_intent="long_recall",
            ppl_ratio=4.0,
            mean_kld=0.90,
            token_klds=[0.90],
            top1_agreement=0.25,
        ),
    ]

    summary = summarize_teacher_cache_records(records)

    ppl = summary["row_distributions"]["ppl_ratio"]
    assert ppl["median"] == 2.0
    assert ppl["max"] == 4.0
    assert summary["row_distributions"]["mean_kld"]["p90"] > 0.3
    assert summary["row_distributions"]["top1_agreement"]["p99"] <= 1.0
    assert summary["row_distributions"]["nll_delta"]["max"] == float(np.log(4.0))


def test_quality_gate_rejects_mean_improves_tail_worsens_reproducibly() -> None:
    baseline = [
        _record(
            prompt_id=f"report_row_{idx}",
            prompt_set="air_vq_ladder_report_v1",
            eval_split="report",
            row_intent="code_debug" if idx % 2 else "hard_arithmetic",
            ppl_ratio=1.20,
            mean_kld=0.40,
            token_klds=[0.40, 0.45],
            top1_agreement=0.80,
        )
        for idx in range(8)
    ]
    candidate = [
        _record(
            prompt_id=f"report_row_{idx}",
            prompt_set="air_vq_ladder_report_v1",
            eval_split="report",
            row_intent="code_debug" if idx % 2 else "hard_arithmetic",
            ppl_ratio=1.05 if idx < 6 else 1.60,
            mean_kld=0.20 if idx < 6 else 0.55,
            token_klds=[0.20, 0.25] if idx < 6 else [1.40, 1.70],
            top1_agreement=0.90 if idx < 6 else 0.50,
        )
        for idx in range(8)
    ]

    verdict = evaluate_quality_gate(
        baseline_records=baseline,
        candidate_records=candidate,
        require_exact_kld=False,
    )

    assert verdict["verdict"] == "FALL BACK"
    assert "mean_improves_tail_worsens" in verdict["reasons"]
    assert verdict["tail_regression"]["split_half_confirmed"] is True


def test_quality_gate_escalates_when_exact_kld_is_required_but_topk_is_used() -> None:
    baseline = [
        _record(
            prompt_id="report_exact",
            prompt_set="air_vq_ladder_report_v1",
            eval_split="report",
            row_intent="code_debug",
            ppl_ratio=1.10,
            mean_kld=0.20,
            token_klds=[0.20],
            top1_agreement=0.90,
        )
    ]
    candidate = [dict(baseline[0], kld_mode="teacher_topk_lower_bound")]
    baseline[0]["kld_mode"] = "exact_full_logits"

    verdict = evaluate_quality_gate(
        baseline_records=baseline,
        candidate_records=candidate,
        require_exact_kld=True,
    )

    assert verdict["verdict"] == "ESCALATE"
    assert "exact_full_logits_required" in verdict["reasons"]


def test_route_coverage_gate_requires_report_layers_and_layer41_glu_code() -> None:
    complete_row = _record(
        prompt_id="report_route",
        prompt_set="air_vq_ladder_report_v1",
        eval_split="report",
        row_intent="layer41_glu_code_route",
        ppl_ratio=1.0,
        mean_kld=0.1,
        token_klds=[0.1],
        top1_agreement=1.0,
        route_coverage={
            "31": {"covered_expert_count": 8},
            "36": {"covered_expert_count": 8},
            "41": {"covered_expert_count": 8, "glu_code_route_count": 2},
        },
    )

    assert validate_ladder_route_coverage([complete_row])["ok"] is True
    missing = validate_ladder_route_coverage([{**complete_row, "route_coverage": {"31": {}}}])
    assert missing["ok"] is False
    assert "36" in missing["missing_layers"]
    assert missing["layer41_glu_code_represented"] is False
