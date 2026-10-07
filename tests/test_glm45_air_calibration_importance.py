from __future__ import annotations

from keep.quality.calibration_importance import build_calibration_importance_report


def _calibration_record(
    *,
    layer: int,
    prompt_id: str,
    context_tokens: int,
    moe_rms: float,
    down_rms: float,
    selected_experts: list[int],
    num_experts: int = 8,
) -> dict:
    return {
        "layer": layer,
        "prompt_id": prompt_id,
        "context_tokens": context_tokens,
        "route_count": context_tokens * 8,
        "num_experts": num_experts,
        "selected_experts": selected_experts,
        "selected_expert_count": len(selected_experts),
        "expert_coverage_fraction": len(selected_experts) / num_experts,
        "moe_input": {"rms": moe_rms, "mean_abs": moe_rms / 2, "abs_max": moe_rms * 3, "finite": True},
        "down_input": {"rms": down_rms, "mean_abs": down_rms / 2, "abs_max": down_rms * 3, "finite": True},
    }


def _source_probe_report() -> dict:
    return {
        "summary": {
            "layer_projection_rankings": [
                {
                    "layer": 1,
                    "metric": "source_routed_glu",
                    "mean_rel_l2": 0.9,
                    "max_rel_l2": 1.0,
                    "record_count": 2,
                },
                {
                    "layer": 2,
                    "metric": "source_routed_glu",
                    "mean_rel_l2": 0.8,
                    "max_rel_l2": 0.9,
                    "record_count": 2,
                },
                {
                    "layer": 2,
                    "metric": "source_down_proj",
                    "mean_rel_l2": 0.7,
                    "max_rel_l2": 0.8,
                    "record_count": 2,
                },
            ]
        }
    }


def test_calibration_importance_weights_source_errors_by_activation_and_coverage() -> None:
    records = [
        _calibration_record(
            layer=1,
            prompt_id="short",
            context_tokens=4,
            moe_rms=0.5,
            down_rms=0.25,
            selected_experts=[0, 1],
        ),
        _calibration_record(
            layer=2,
            prompt_id="short",
            context_tokens=4,
            moe_rms=2.0,
            down_rms=0.1,
            selected_experts=[0, 1, 2, 3, 4, 5],
        ),
        _calibration_record(
            layer=2,
            prompt_id="long",
            context_tokens=12,
            moe_rms=4.0,
            down_rms=0.2,
            selected_experts=[2, 3, 4, 5, 6, 7],
        ),
    ]

    report = build_calibration_importance_report(
        source_probe_report=_source_probe_report(),
        calibration_records=records,
        top_candidate_layers=1,
    )

    top = report["group_rankings"][0]
    assert top["layer"] == 2
    assert top["metric"] == "source_routed_glu"
    assert top["precision_groups"] == ["2:gate_proj", "2:up_proj"]
    assert top["calibration"]["coverage_fraction"] == 1.0
    assert top["score"] > report["group_rankings"][1]["score"]

    candidate = report["candidates"][0]
    assert candidate["name"] == "lane3-calib-top1-glu"
    assert candidate["code_bits_policy"] == {
        "2:gate_proj": 16,
        "2:up_proj": 16,
    }


def test_calibration_importance_marks_missing_layer_calibration_as_zero_signal() -> None:
    report = build_calibration_importance_report(
        source_probe_report=_source_probe_report(),
        calibration_records=[
            _calibration_record(
                layer=1,
                prompt_id="short",
                context_tokens=4,
                moe_rms=1.0,
                down_rms=1.0,
                selected_experts=[0, 1],
            )
        ],
    )

    by_layer = {row["layer"]: row for row in report["group_rankings"] if row["metric"] == "source_routed_glu"}

    assert by_layer[2]["score"] == 0.0
    assert by_layer[2]["calibration"]["record_count"] == 0
    assert by_layer[2]["calibration"]["coverage_fraction"] == 0.0
