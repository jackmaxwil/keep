from __future__ import annotations

import json

import numpy as np

from mlx_vq.quality.calibration import (
    build_activation_record,
    compare_routing_records,
    per_expert_kld_contribution,
    summarize_activation,
    summarize_route_records,
)
from mlx_vq.quality.prompts import QualityPrompt


def test_summarize_activation_reports_finite_magnitude_stats() -> None:
    values = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)

    summary = summarize_activation(values)

    assert summary.abs_max == 4.0
    assert summary.mean_abs == 2.5
    assert summary.rms > 2.7
    assert summary.finite is True
    assert json.dumps(summary.to_dict(), sort_keys=True)


def test_build_activation_record_reports_expert_coverage_and_shapes() -> None:
    prompt = QualityPrompt(
        prompt_id="capital_france",
        text="The capital of France is",
        max_new_tokens=8,
    )
    indices = np.array([[3, 1], [3, 7]], dtype=np.int64)
    scores = np.array([[0.6, 0.4], [0.55, 0.45]], dtype=np.float32)
    moe_input = np.ones((2, 4), dtype=np.float32)
    down_input = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)

    record = build_activation_record(
        model_id="zai-org/GLM-4.5-Air",
        artifact_dir="artifacts/glm-4.5-air-vq",
        prompt=prompt,
        input_token_ids=[1, 2, 3, 4],
        layer=1,
        num_experts=8,
        indices=indices,
        scores=scores,
        moe_input=moe_input,
        down_input=down_input,
        dense_expert_params=False,
        unbound_vq_experts=False,
    )

    assert record["prompt_id"] == "capital_france"
    assert record["layer"] == 1
    assert record["route_count"] == 4
    assert record["selected_experts"] == [1, 3, 7]
    assert record["expert_counts"] == {"1": 1, "3": 2, "7": 1}
    assert record["expert_coverage_fraction"] == 3 / 8
    assert record["route_indices"] == [[3, 1], [3, 7]]
    assert record["router_scores"] == [[0.6000000238418579, 0.4000000059604645], [0.550000011920929, 0.44999998807907104]]
    assert record["projection_inputs"] == {
        "gate_proj": "moe_input",
        "up_proj": "moe_input",
        "down_proj": "down_input",
    }
    assert record["moe_input"]["finite"] is True
    assert record["down_input"]["finite"] is True
    assert record["dense_expert_params"] is False
    assert json.dumps(record, sort_keys=True)


def test_summarize_route_records_aggregates_layer_coverage() -> None:
    records = [
        {
            "prompt_id": "a",
            "layer": 41,
            "num_experts": 8,
            "route_count": 4,
            "expert_counts": {"1": 1, "3": 2, "7": 1},
            "router_scores": [[0.6, 0.4], [0.55, 0.45]],
        },
        {
            "prompt_id": "b",
            "layer": 41,
            "num_experts": 8,
            "route_count": 2,
            "expert_counts": {"3": 1, "4": 1},
            "router_scores": [[0.7, 0.3]],
        },
    ]

    summary = summarize_route_records(records)
    layer = summary["layers"]["41"]

    assert layer["record_count"] == 2
    assert layer["route_count"] == 6
    assert layer["selected_expert_count"] == 4
    assert layer["cold_expert_count"] == 4
    assert layer["expert_counts"]["3"] == 3
    assert layer["top_experts"][0] == {"expert": 3, "route_count": 3}


def test_compare_routing_records_reports_set_order_and_score_metrics() -> None:
    reference = {
        "prompt_id": "row",
        "layer": 41,
        "route_indices": [[1, 2], [3, 4]],
        "router_scores": [[0.7, 0.3], [0.6, 0.4]],
    }
    candidate = {
        "prompt_id": "row",
        "layer": 41,
        "route_indices": [[2, 1], [3, 5]],
        "router_scores": [[0.3, 0.7], [0.55, 0.45]],
    }

    comparison = compare_routing_records(reference, candidate)

    assert comparison["route_topk_set_agreement"] == 0.5
    assert comparison["route_order_agreement"] == 0.25
    assert comparison["router_score_mse"] > 0
    assert comparison["router_score_kl"] > 0


def test_per_expert_kld_contribution_uses_router_score_weights() -> None:
    records = [
        {
            "route_indices": [[1, 2], [2, 3]],
            "router_scores": [[0.75, 0.25], [0.5, 0.5]],
            "token_klds": [2.0, 4.0],
        }
    ]

    contribution = per_expert_kld_contribution(records)
    by_expert = {item["expert"]: item for item in contribution["experts"]}

    assert by_expert[1]["weighted_kld_contribution"] == 1.5
    assert by_expert[2]["weighted_kld_contribution"] == 2.5
    assert by_expert[3]["weighted_kld_contribution"] == 2.0
