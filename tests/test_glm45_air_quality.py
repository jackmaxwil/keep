from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from keep.quality.glm45_air import (
    build_quality_record,
    compute_nll_metrics,
    compute_topk_summary,
)
from keep.quality.prompts import (
    QualityPrompt,
    get_quality_prompt_by_id,
    get_quality_prompt_ids,
    get_quality_prompt_set_names,
    get_quality_prompts,
    summarize_quality_prompt_corpus,
    validate_imatrix_calibration_prompt_set,
)


def test_quality_prompt_registry_is_stable() -> None:
    prompts = get_quality_prompts()

    assert [prompt.prompt_id for prompt in prompts] == [
        "capital_france",
        "short_math",
        "code_completion",
        "instruction_following",
        "long_recall_1k",
    ]
    assert prompts[-1].context_tokens == 1024


def test_lane0_widened_quality_prompt_set_extends_base() -> None:
    base_ids = [prompt.prompt_id for prompt in get_quality_prompts()]
    widened = get_quality_prompts(prompt_set="lane0_widened")
    widened_ids = [prompt.prompt_id for prompt in widened]

    assert get_quality_prompt_set_names() == (
        "base",
        "lane0_widened",
        "air_vq_ladder_select_v1",
        "air_vq_ladder_report_v1",
        "air_vq_ladder_holdout_v1",
        "air_imatrix_calib_v1",
    )
    assert widened_ids[: len(base_ids)] == base_ids
    assert len(widened_ids) == 17
    assert {
        "short_math_addition_carry",
        "short_math_multi_step",
        "short_math_half",
        "code_completion_loop",
        "code_completion_branch",
        "code_completion_list",
        "instruction_following_six_words",
        "instruction_following_json",
        "instruction_following_avoid_word",
        "capital_japan",
        "basic_color",
        "long_recall_shape_1k",
    }.issubset(widened_ids)
    assert set(widened_ids).issubset(set(get_quality_prompt_ids(prompt_set=None)))
    assert get_quality_prompt_by_id("short_math_multi_step").text.startswith("Compute")


def test_air_imatrix_calibration_prompt_set_is_chat_templated_code_heavy_and_non_leaking() -> None:
    prompts = get_quality_prompts(prompt_set="air_imatrix_calib_v1")
    validation = validate_imatrix_calibration_prompt_set()
    coverage = summarize_quality_prompt_corpus("air_imatrix_calib_v1")

    assert len(prompts) == 96
    assert all(prompt.prompt_id.startswith("imatrix_calib_") for prompt in prompts)
    assert all(prompt.eval_split is None for prompt in prompts)
    assert {prompt.suite_version for prompt in prompts} == {"air_imatrix_calib_v1"}
    assert all(
        "<|system|>" in prompt.text
        and "<|user|>" in prompt.text
        and "<|assistant|>" in prompt.text
        for prompt in prompts
    )
    assert validation["ok"] is True
    assert validation["overlap_prompt_pairs"] == []
    assert coverage["prompt_count"] == 96
    assert coverage["row_intent_counts"]["imatrix_code_debug"] >= 16
    assert coverage["row_intent_counts"]["imatrix_code_completion"] >= 16
    assert coverage["row_intent_counts"]["imatrix_code_tests"] >= 16
    assert coverage["code_prompt_fraction"] >= 0.60
    assert coverage["estimated_text_token_count"] >= 5000
    assert "air_vq_ladder_report_v1" in coverage["dedupe_checked_against"]
    assert "air_vq_ladder_holdout_v1" in coverage["dedupe_checked_against"]


def test_air_vq_ladder_holdout_is_blind_third_split() -> None:
    from keep.quality.prompts import (
        prompt_text_hash,
        summarize_quality_prompt_corpus,
        validate_quality_prompt_splits,
    )

    holdout = get_quality_prompts(prompt_set="air_vq_ladder_holdout_v1")
    report = get_quality_prompts(prompt_set="air_vq_ladder_report_v1")
    selection = get_quality_prompts(prompt_set="air_vq_ladder_select_v1")
    validation = validate_quality_prompt_splits()
    coverage = summarize_quality_prompt_corpus("air_vq_ladder_holdout_v1")

    assert len(holdout) == 128
    assert validation["ok"] is True
    assert validation["holdout_count"] == 128
    assert all(prompt.prompt_id.startswith("holdout_") for prompt in holdout)
    assert {prompt.eval_split for prompt in holdout} == {"holdout"}
    assert {prompt.suite_version for prompt in holdout} == {"air_vq_ladder_v1"}
    assert {prompt.prompt_id for prompt in holdout}.isdisjoint(
        {prompt.prompt_id for prompt in selection + report}
    )
    assert {prompt_text_hash(prompt.text) for prompt in holdout}.isdisjoint(
        {prompt_text_hash(prompt.text) for prompt in selection + report}
    )
    assert coverage["prompt_count"] == 128
    assert set(coverage["dedupe_checked_against"]) == {
        "air_vq_ladder_select_v1",
        "air_vq_ladder_report_v1",
    }


def test_quality_metric_helpers_report_finite_topk_and_nll() -> None:
    logits = mx.array(
        np.array(
            [
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [3.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 4.0, 1.0],
                ]
            ],
            dtype=np.float32,
        )
    )

    topk = compute_topk_summary(logits, top_k=2)
    nll = compute_nll_metrics(logits, [0, 3, 0], max_tokens=2)

    assert topk["finite_logits"] is True
    assert topk["top_token_ids"] == [2, 3]
    assert len(topk["top_probabilities"]) == 2
    assert topk["entropy"] > 0
    assert nll["nll_token_count"] == 2
    assert nll["perplexity"] is not None


def test_quality_record_schema_is_json_serializable() -> None:
    prompt = QualityPrompt(
        prompt_id="capital_france",
        text="The capital of France is",
        max_new_tokens=8,
    )
    record = build_quality_record(
        model_id="zai-org/GLM-4.5-Air",
        revision="main",
        artifact_dir="artifacts/glm-4.5-air-vq",
        prompt=prompt,
        input_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5],
        generated_text=" Paris",
        prefill_seconds=0.5,
        decode_seconds=0.25,
        metrics={
            "mlx_active_bytes": 10,
            "mlx_peak_bytes": 20,
            "mlx_cache_bytes": 5,
            "rss_bytes": 30,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
        topk_summary={
            "finite_logits": True,
            "entropy": 1.0,
            "max_probability": 0.5,
            "top_token_ids": [4, 5],
            "top_probabilities": [0.5, 0.25],
        },
        nll_metrics={"nll": 1.25, "perplexity": 3.49, "nll_token_count": 2},
        dense_expert_params=False,
        unbound_vq_experts=False,
        extra={"engine": "vq_resident", "quantized_scope": "routed_experts"},
    )

    assert record["prompt_id"] == "capital_france"
    assert record["engine"] == "vq_resident"
    assert record["quantized_scope"] == "routed_experts"
    assert record["decode_tokens_per_second"] == 8.0
    assert record["dense_expert_params"] is False
    assert json.dumps(record, sort_keys=True)
