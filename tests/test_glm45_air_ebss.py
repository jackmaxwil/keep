from __future__ import annotations

import json

import pytest

from benchmarks.select_glm45_air_ebss_prompts import write_ebss_selection_manifest
from mlx_vq.quality.ebss import build_ebss_prompt_selection, build_route_record


def _route_record(
    *,
    prompt_id: str,
    layer: int = 41,
    counts: dict[str, int],
    num_experts: int = 2,
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "layer": layer,
        "num_experts": num_experts,
        "route_count": sum(counts.values()),
        "expert_counts": counts,
    }


def test_ebss_prompt_selection_prefers_lower_expert_imbalance() -> None:
    records = [
        _route_record(prompt_id="imatrix_calib_hot_a_000", counts={"0": 8}),
        _route_record(prompt_id="imatrix_calib_hot_a_000", layer=42, counts={"0": 8}),
        _route_record(prompt_id="imatrix_calib_hot_b_001", counts={"0": 8}),
        _route_record(prompt_id="imatrix_calib_hot_b_001", layer=42, counts={"0": 8}),
        _route_record(prompt_id="imatrix_calib_balanced_002", counts={"0": 4, "1": 4}),
        _route_record(prompt_id="imatrix_calib_balanced_002", layer=42, counts={"0": 4, "1": 4}),
    ]

    selection = build_ebss_prompt_selection(records, max_prompts=1)

    assert selection["selection_mode"] == "ebss_rebalanced_existing_pool"
    assert selection["selected_prompt_ids"] == ["imatrix_calib_balanced_002"]
    assert selection["selected_imbalance"] < selection["full_pool_imbalance"]
    assert selection["selected_imbalance"] < selection["source_prefix_imbalance"]
    assert selection["selected_layer_summaries"]["41"]["expert_frequencies"] == {"0": 0.5, "1": 0.5}
    assert selection["full_pool_layer_summaries"]["41"]["expert_frequencies"]["0"] > 0.8


def test_ebss_prompt_selection_rejects_records_without_expert_counts() -> None:
    records = [{"prompt_id": "imatrix_calib_missing_000", "layer": 41, "num_experts": 2}]

    with pytest.raises(ValueError, match="expert_counts"):
        build_ebss_prompt_selection(records, max_prompts=1)


def test_ebss_manifest_writer_refuses_existing_output(tmp_path) -> None:
    records_path = tmp_path / "activation-records.jsonl"
    out_path = tmp_path / "ebss-selection.json"
    records = [_route_record(prompt_id="imatrix_calib_balanced_002", counts={"0": 2, "1": 2})]
    records_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    out_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_ebss_selection_manifest(records_path=records_path, out_path=out_path, max_prompts=1)


def test_build_route_record_counts_experts_and_router_score_mass() -> None:
    record = build_route_record(
        prompt_id="imatrix_calib_code_debug_000",
        layer=12,
        route_indices=[[2, 1], [2, 3]],
        router_scores=[[0.7, 0.3], [0.4, 0.6]],
        num_experts=4,
    )

    assert record["record_type"] == "air_route_record"
    assert record["prompt_id"] == "imatrix_calib_code_debug_000"
    assert record["layer"] == 12
    assert record["route_count"] == 4
    assert record["expert_counts"] == {"0": 0, "1": 1, "2": 2, "3": 1}
    assert record["expert_router_score_sums"] == {
        "0": 0.0,
        "1": 0.3,
        "2": 1.1,
        "3": 0.6,
    }
