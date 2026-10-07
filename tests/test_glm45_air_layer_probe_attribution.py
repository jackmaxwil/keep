from __future__ import annotations

import json

import numpy as np
import pytest

from mlx_vq.quality.layer_probe_attribution import (
    build_sparse_residual_rows_from_plan_report,
    load_layer_probe_state_bundle,
    parse_layer_spec,
    parse_probe_target,
    write_layer_probe_state_bundle,
    summarize_layer_probe_records,
)


def _metric(rel_l2: float, cosine: float = 0.9) -> dict:
    return {
        "rel_l2": rel_l2,
        "cosine": cosine,
        "max_abs": rel_l2,
        "mean_abs": rel_l2 / 10,
        "actual_finite": True,
        "expected_finite": True,
    }


def _record(layer: int, target_key: str, *, gate: float, up: float, routed: float, weighted: float) -> dict:
    prompt_id, position = target_key.split(":")
    return {
        "layer": layer,
        "selected_experts": [1, 3],
        "probe_target": {
            "key": target_key,
            "prompt_id": prompt_id,
            "position": int(position),
        },
        "metrics": {
            "source_gate_proj": _metric(gate),
            "source_up_proj": _metric(up),
            "source_routed_glu": _metric(routed),
            "source_weighted_routed": _metric(weighted),
        },
    }


def _route_record(layer: int, target_key: str) -> dict:
    record = _record(layer, target_key, gate=0.1, up=0.2, routed=0.3, weighted=0.4)
    record["route_source_metrics"] = [
        {
            "token_index": 0,
            "route_rank": 0,
            "expert": 93,
            "router_score": 0.55,
            "metrics": {
                "source_routed_glu": _metric(0.7, cosine=0.6) | {"mean_abs": 0.6},
                "source_weighted_route_contribution": _metric(0.8, cosine=0.5) | {"mean_abs": 0.4},
            },
        },
        {
            "token_index": 0,
            "route_rank": 1,
            "expert": 73,
            "router_score": 0.07,
            "metrics": {
                "source_routed_glu": _metric(1.2, cosine=0.4) | {"mean_abs": 0.07},
                "source_weighted_route_contribution": _metric(1.2, cosine=0.4) | {"mean_abs": 0.005},
            },
        },
    ]
    record["route_source_residual_topk"] = [
        {
            "token_index": 0,
            "route_rank": 0,
            "expert": 93,
            "router_score": 0.55,
            "top_abs_residuals": {
                "source_weighted_route_contribution": [
                    {
                        "index": 11,
                        "actual": -0.2,
                        "expected": 0.9,
                        "source_minus_actual": 1.1,
                        "abs_source_minus_actual": 1.1,
                    },
                    {
                        "index": 4,
                        "actual": 0.3,
                        "expected": -0.1,
                        "source_minus_actual": -0.4,
                        "abs_source_minus_actual": 0.4,
                    },
                ],
                "source_routed_glu": [
                    {
                        "index": 7,
                        "actual": 0.0,
                        "expected": -1.5,
                        "source_minus_actual": -1.5,
                        "abs_source_minus_actual": 1.5,
                    }
                ],
            },
        }
    ]
    record["route_source_sparse_residual_plans"] = [
        {
            "token_index": 0,
            "route_rank": 0,
            "expert": 93,
            "router_score": 0.55,
            "projection": "down_proj",
            "source_metric": "source_weighted_route_contribution",
            "rows": [
                {
                    "output_index": 11,
                    "desired_weighted_correction": 1.1,
                    "desired_unweighted_correction": 2.0,
                    "input_norm_sq": 4.0,
                    "residual_value_norm": 1.0,
                    "residual_values": [0.5, 0.5],
                    "reconstructed_unweighted_correction": 2.0,
                    "reconstruction_abs_error": 0.0,
                },
                {
                    "output_index": 4,
                    "desired_weighted_correction": -0.4,
                    "desired_unweighted_correction": -0.8,
                    "input_norm_sq": 4.0,
                    "residual_value_norm": 0.4,
                    "residual_values": [-0.2, -0.2],
                    "reconstructed_unweighted_correction": -0.8,
                    "reconstruction_abs_error": 0.0,
                },
            ],
        }
    ]
    return record


def test_parse_layer_spec_accepts_ranges_and_dedupes() -> None:
    assert parse_layer_spec("1,3-5,3,8") == (1, 3, 4, 5, 8)


def test_parse_probe_target_requires_prompt_and_position() -> None:
    target = parse_probe_target("short_math:0")

    assert target.prompt_id == "short_math"
    assert target.position == 0
    assert target.key == "short_math:0"


def test_summarize_layer_probe_records_ranks_layers_and_projections() -> None:
    records = [
        _record(1, "short_math:0", gate=0.1, up=0.2, routed=0.3, weighted=0.4),
        _record(1, "code_completion:0", gate=0.2, up=0.1, routed=0.2, weighted=0.2),
        _record(6, "short_math:0", gate=0.7, up=0.3, routed=0.6, weighted=0.8),
        _record(6, "code_completion:0", gate=0.6, up=0.2, routed=0.5, weighted=0.7),
    ]

    summary = summarize_layer_probe_records(records)

    assert summary["record_count"] == 4
    assert summary["target_count"] == 2
    assert summary["layer_rankings"][0]["layer"] == 6
    assert summary["layer_rankings"][0]["rank"] == 1
    assert summary["layer_projection_rankings"][0]["layer"] == 6
    assert summary["layer_projection_rankings"][0]["metric"] == "source_weighted_routed"
    assert summary["target_rankings"][0]["target_key"] == "short_math:0"
    assert summary["records"][2]["derived"]["dominant_projection"] == "source_gate_proj"
    json.dumps(summary, sort_keys=True)


def test_summarize_layer_probe_records_ranks_route_source_metrics() -> None:
    summary = summarize_layer_probe_records([_route_record(41, "report_route_000:0")])

    assert summary["route_source_rankings"][0]["metric"] == "source_weighted_route_contribution"
    assert summary["route_source_rankings"][0]["rank"] == 1
    assert summary["route_source_rankings"][0]["expert"] == 93
    assert summary["route_source_rankings"][0]["mean_abs"] == 0.4
    routed_glu_rows = [
        row for row in summary["route_source_rankings"]
        if row["metric"] == "source_routed_glu"
    ]
    assert routed_glu_rows[0]["expert"] == 73
    assert routed_glu_rows[0]["rel_l2"] == 1.2


def test_summarize_layer_probe_records_ranks_route_source_residual_coordinates() -> None:
    summary = summarize_layer_probe_records([_route_record(41, "report_route_000:0")])

    assert summary["route_source_residual_rankings"][0]["metric"] == "source_weighted_route_contribution"
    assert summary["route_source_residual_rankings"][0]["expert"] == 93
    assert summary["route_source_residual_rankings"][0]["coordinate_index"] == 11
    assert summary["route_source_residual_rankings"][0]["source_minus_actual"] == 1.1
    assert summary["route_source_residual_rankings"][0]["abs_source_minus_actual"] == 1.1
    routed_glu_rows = [
        row for row in summary["route_source_residual_rankings"]
        if row["metric"] == "source_routed_glu"
    ]
    assert routed_glu_rows[0]["coordinate_index"] == 7
    assert routed_glu_rows[0]["abs_source_minus_actual"] == 1.5


def test_summarize_layer_probe_records_ranks_sparse_residual_plan_rows() -> None:
    summary = summarize_layer_probe_records([_route_record(41, "report_route_000:0")])

    assert summary["route_source_sparse_residual_plan_rankings"][0]["projection"] == "down_proj"
    assert summary["route_source_sparse_residual_plan_rankings"][0]["expert"] == 93
    assert summary["route_source_sparse_residual_plan_rankings"][0]["output_index"] == 11
    assert summary["route_source_sparse_residual_plan_rankings"][0]["desired_weighted_correction"] == 1.1
    assert summary["route_source_sparse_residual_plan_rankings"][0]["residual_value_norm"] == 1.0


def test_build_sparse_residual_rows_from_plan_report_selects_target_route_rows() -> None:
    report = {
        "summary": {
            "records": [_route_record(41, "report_route_000:0")],
        },
    }

    rows = build_sparse_residual_rows_from_plan_report(
        report,
        target_key="report_route_000:0",
        expert=93,
        route_rank=0,
        max_rows=1,
    )

    assert rows["layer"] == 41
    assert rows["projection"] == "down_proj"
    np.testing.assert_array_equal(rows["expert_indices"], np.array([93], dtype=np.int32))
    np.testing.assert_array_equal(rows["output_indices"], np.array([11], dtype=np.int32))
    np.testing.assert_allclose(rows["values"], np.array([[0.5, 0.5]], dtype=np.float32))
    assert rows["selected_rows"][0]["desired_weighted_correction"] == 1.1


def test_layer_probe_state_bundle_roundtrips_requested_layers(tmp_path) -> None:
    path = tmp_path / "states.npz"
    layer36 = np.arange(12, dtype=np.float32).reshape(3, 4)
    layer41 = (np.arange(12, dtype=np.float32).reshape(3, 4) + 100.0)

    write_layer_probe_state_bundle(
        path,
        prompt_id="report_route_000",
        input_token_ids=[101, 202, 303],
        captures={36: layer36, 41: layer41},
    )

    captures = load_layer_probe_state_bundle(
        path,
        prompt_id="report_route_000",
        input_token_ids=[101, 202, 303],
        layers=(41,),
    )

    assert sorted(captures) == [41]
    np.testing.assert_array_equal(captures[41], layer41)


def test_layer_probe_state_bundle_rejects_prompt_token_mismatch(tmp_path) -> None:
    path = tmp_path / "states.npz"
    write_layer_probe_state_bundle(
        path,
        prompt_id="report_route_000",
        input_token_ids=[101, 202, 303],
        captures={41: np.zeros((3, 4), dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="input_token_ids"):
        load_layer_probe_state_bundle(
            path,
            prompt_id="report_route_000",
            input_token_ids=[101, 999, 303],
            layers=(41,),
        )
