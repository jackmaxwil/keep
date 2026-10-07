from __future__ import annotations

import json
import sys

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.quality.calibration_importance import (
    ProjectionImatrixEntry,
    accumulate_routed_projection_imatrix,
    build_air_imatrix_collection_plan,
    imatrix_entries_from_activation_records,
    write_projection_imatrix_sidecars,
)
from mlx_vq.quality.calibration import build_activation_record
from mlx_vq.quality.imatrix_collection import (
    finalize_entries,
    merge_entries,
    parse_layer_selection,
    parse_projection_selection,
)
from mlx_vq.quality.prompts import QualityPrompt


imatrix_module = sys.modules[accumulate_routed_projection_imatrix.__module__]


def _reference_accumulate_routed_projection_imatrix(
    *,
    layer: int,
    projection: str,
    inputs: np.ndarray,
    route_indices: np.ndarray,
    router_scores: np.ndarray | None,
    num_experts: int,
) -> tuple[ProjectionImatrixEntry, ...]:
    """Original per-expert masked implementation, retained for byte-identity tests."""

    values = np.asarray(inputs, dtype=np.float32)
    routes = np.asarray(route_indices, dtype=np.int64)
    routed_inputs = (
        np.repeat(values[:, None, :], routes.shape[1], axis=1)
        if values.ndim == 2
        else values
    ).reshape(-1, values.shape[-1])
    routed_experts = routes.reshape(-1)
    flat_scores = (
        np.asarray(router_scores, dtype=np.float32).reshape(-1).astype(np.float64, copy=False)
        if router_scores is not None
        else None
    )
    total_route_count = int(routed_experts.size)
    input_dim = int(routed_inputs.shape[-1])
    sums = np.zeros((num_experts, input_dim), dtype=np.float64)
    affinity_sums = (
        np.zeros((num_experts, input_dim), dtype=np.float64) if flat_scores is not None else None
    )
    affinity_score_sums = np.zeros(num_experts, dtype=np.float64) if flat_scores is not None else None
    counts = np.zeros(num_experts, dtype=np.int64)
    squared_inputs = routed_inputs.astype(np.float64) * routed_inputs.astype(np.float64)
    for expert in range(num_experts):
        mask = routed_experts == expert
        counts[expert] = int(np.count_nonzero(mask))
        if counts[expert]:
            sums[expert] = np.sum(squared_inputs[mask], axis=0)
            if affinity_sums is not None and flat_scores is not None:
                affinity_sums[expert] = np.sum(
                    squared_inputs[mask] * flat_scores[mask, None], axis=0
                )
                assert affinity_score_sums is not None
                affinity_score_sums[expert] = float(np.sum(flat_scores[mask]))

    entries = []
    for expert in range(num_experts):
        route_count = int(counts[expert])
        entries.append(
            ProjectionImatrixEntry(
                layer=layer,
                projection=projection,
                expert=expert,
                importance_sum=sums[expert].astype(np.float32),
                mean_importance=(
                    (sums[expert] / route_count).astype(np.float32)
                    if route_count
                    else np.zeros(input_dim, dtype=np.float32)
                ),
                routing_weighted_importance=(
                    sums[expert] / total_route_count
                ).astype(np.float32),
                route_count=route_count,
                total_route_count=total_route_count,
                route_frequency=route_count / total_route_count,
                affinity_weighted_importance=(
                    affinity_sums[expert].astype(np.float32)
                    if affinity_sums is not None
                    else None
                ),
                affinity_score_sum=(
                    float(affinity_score_sums[expert])
                    if affinity_score_sums is not None
                    else None
                ),
            )
        )
    return tuple(entries)


def _assert_imatrix_entries_byte_identical(
    actual: tuple[ProjectionImatrixEntry, ...],
    expected: tuple[ProjectionImatrixEntry, ...],
) -> None:
    assert len(actual) == len(expected)
    for actual_entry, expected_entry in zip(actual, expected, strict=True):
        assert actual_entry.layer == expected_entry.layer
        assert actual_entry.projection == expected_entry.projection
        assert actual_entry.expert == expected_entry.expert
        np.testing.assert_array_equal(actual_entry.importance_sum, expected_entry.importance_sum)
        np.testing.assert_array_equal(actual_entry.mean_importance, expected_entry.mean_importance)
        np.testing.assert_array_equal(
            actual_entry.routing_weighted_importance,
            expected_entry.routing_weighted_importance,
        )
        assert actual_entry.route_count == expected_entry.route_count
        assert actual_entry.total_route_count == expected_entry.total_route_count
        assert actual_entry.route_frequency == expected_entry.route_frequency
        assert actual_entry.prompt_ids == expected_entry.prompt_ids
        if expected_entry.affinity_weighted_importance is None:
            assert actual_entry.affinity_weighted_importance is None
        else:
            np.testing.assert_array_equal(
                actual_entry.affinity_weighted_importance,
                expected_entry.affinity_weighted_importance,
            )
        assert actual_entry.affinity_score_sum == expected_entry.affinity_score_sum


def _prompt(prompt_id: str) -> QualityPrompt:
    return QualityPrompt(prompt_id=prompt_id, text=f"Prompt {prompt_id}", max_new_tokens=8)


def _record(
    *,
    prompt_id: str,
    indices: np.ndarray,
    moe_input: np.ndarray,
    down_input: np.ndarray | None,
    include_activation_rows: bool,
) -> dict[str, object]:
    return build_activation_record(
        model_id="zai-org/GLM-4.5-Air",
        artifact_dir="artifacts/glm-4.5-air-vq",
        prompt=_prompt(prompt_id),
        input_token_ids=list(range(1, int(indices.shape[0]) + 1)),
        layer=3,
        num_experts=2,
        indices=indices,
        scores=np.ones(indices.shape, dtype=np.float32) / indices.shape[1],
        moe_input=moe_input,
        down_input=down_input,
        dense_expert_params=False,
        unbound_vq_experts=False,
        include_activation_rows=include_activation_rows,
    )


def test_accumulate_routed_projection_imatrix_sums_squared_columns_and_route_weights() -> None:
    inputs = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    route_indices = np.array(
        [
            [0, 1],
            [1, 1],
        ],
        dtype=np.int32,
    )

    entries = accumulate_routed_projection_imatrix(
        layer=7,
        projection="gate_proj",
        inputs=inputs,
        route_indices=route_indices,
        num_experts=3,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    by_expert = {entry.expert: entry for entry in entries}

    np.testing.assert_allclose(by_expert[0].importance_sum, np.array([1.0, 4.0, 9.0], dtype=np.float32))
    np.testing.assert_allclose(by_expert[1].importance_sum, np.array([33.0, 54.0, 81.0], dtype=np.float32))
    np.testing.assert_allclose(by_expert[2].importance_sum, np.zeros(3, dtype=np.float32))
    assert by_expert[0].route_count == 1
    assert by_expert[1].route_count == 3
    assert by_expert[2].route_count == 0
    assert by_expert[1].total_route_count == 4
    assert by_expert[1].route_frequency == 0.75
    np.testing.assert_allclose(
        by_expert[1].routing_weighted_importance,
        np.array([8.25, 13.5, 20.25], dtype=np.float32),
    )


def test_accumulate_routed_projection_imatrix_tracks_affinity_weighted_importance() -> None:
    inputs = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    route_indices = np.array(
        [
            [0, 1],
            [0, 1],
        ],
        dtype=np.int64,
    )
    router_scores = np.array(
        [
            [0.75, 0.25],
            [0.10, 0.90],
        ],
        dtype=np.float32,
    )

    entries = accumulate_routed_projection_imatrix(
        layer=4,
        projection="gate_proj",
        inputs=inputs,
        route_indices=route_indices,
        router_scores=router_scores,
        num_experts=2,
    )
    by_expert = {entry.expert: entry for entry in entries}

    np.testing.assert_allclose(
        by_expert[0].importance_sum,
        np.array([10.0, 20.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        by_expert[0].affinity_weighted_importance,
        np.array([1.65, 4.6], dtype=np.float32),
    )
    assert by_expert[0].affinity_score_sum == np.float32(0.85)
    np.testing.assert_allclose(
        by_expert[0].affinity_weighted_mean_importance,
        np.array([1.65, 4.6], dtype=np.float32) / np.float32(0.85),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        by_expert[1].affinity_weighted_importance,
        np.array([8.35, 15.4], dtype=np.float32),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        by_expert[1].affinity_weighted_mean_importance,
        np.array([8.35, 15.4], dtype=np.float32) / np.float32(1.15),
        rtol=1.0e-6,
    )


def test_accumulate_routed_projection_imatrix_affinity_weights_per_route_down_inputs() -> None:
    down_inputs = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    route_indices = np.array([[0, 1], [0, 1]], dtype=np.int64)
    router_scores = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)

    entries = accumulate_routed_projection_imatrix(
        layer=4,
        projection="down_proj",
        inputs=down_inputs,
        route_indices=route_indices,
        router_scores=router_scores,
        num_experts=2,
    )
    by_expert = {entry.expert: entry for entry in entries}

    np.testing.assert_allclose(
        by_expert[0].affinity_weighted_importance,
        np.array([15.2, 22.4], dtype=np.float32),
    )
    np.testing.assert_allclose(
        by_expert[0].affinity_weighted_mean_importance,
        np.array([19.0, 28.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        by_expert[1].affinity_weighted_importance,
        np.array([26.8, 38.4], dtype=np.float32),
    )
    np.testing.assert_allclose(
        by_expert[1].affinity_weighted_mean_importance,
        np.array([26.8, 38.4], dtype=np.float32) / np.float32(1.2),
        rtol=1.0e-6,
    )


@pytest.mark.parametrize(
    ("projection", "with_scores"),
    (("gate_proj", True), ("down_proj", True), ("gate_proj", False)),
)
def test_accumulate_routed_projection_imatrix_stable_segments_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
    with_scores: bool,
) -> None:
    rng = np.random.default_rng(20260711)
    tokens, top_k, input_dim, num_experts = 257, 4, 19, 13
    # Expert 12 intentionally has no routes.
    route_indices = rng.integers(0, num_experts - 1, size=(tokens, top_k), dtype=np.int64)
    inputs = rng.standard_normal(
        (tokens, top_k, input_dim) if projection == "down_proj" else (tokens, input_dim),
        dtype=np.float32,
    )
    router_scores = (
        rng.random((tokens, top_k), dtype=np.float32) if with_scores else None
    )
    expected = _reference_accumulate_routed_projection_imatrix(
        layer=17,
        projection=projection,
        inputs=inputs,
        route_indices=route_indices,
        router_scores=router_scores,
        num_experts=num_experts,
    )

    original_argsort = imatrix_module.np.argsort
    argsort_kinds: list[str | None] = []

    def recording_argsort(values: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        argsort_kinds.append(kwargs.get("kind"))
        return original_argsort(values, *args, **kwargs)

    monkeypatch.setattr(imatrix_module.np, "argsort", recording_argsort)
    actual = accumulate_routed_projection_imatrix(
        layer=17,
        projection=projection,
        inputs=inputs,
        route_indices=route_indices,
        router_scores=router_scores,
        num_experts=num_experts,
    )

    assert argsort_kinds == ["stable"]
    _assert_imatrix_entries_byte_identical(actual, expected)


def test_collect_imatrix_cli_parses_layers_and_projection_filters() -> None:
    assert parse_layer_selection("1-3,5,7-8") == (1, 2, 3, 5, 7, 8)
    assert parse_projection_selection("gate_proj,down_proj") == ("gate_proj", "down_proj")

    for bad_layers in ("", "3-1", "-1"):
        try:
            parse_layer_selection(bad_layers)
        except Exception:
            pass
        else:
            raise AssertionError(f"expected {bad_layers!r} to fail")

    try:
        parse_projection_selection("gate_proj,bogus_proj")
    except Exception as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected unsupported projection to fail")


def test_collect_imatrix_cli_accumulator_merges_entries_without_activation_json() -> None:
    first = accumulate_routed_projection_imatrix(
        layer=2,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        route_indices=np.array([[0], [1]], dtype=np.int64),
        num_experts=2,
        prompt_ids=("prompt-a",),
    )
    second = accumulate_routed_projection_imatrix(
        layer=2,
        projection="gate_proj",
        inputs=np.array([[5.0, 6.0]], dtype=np.float32),
        route_indices=np.array([[1]], dtype=np.int64),
        num_experts=2,
        prompt_ids=("prompt-b",),
    )

    accumulators = {}
    merge_entries(accumulators, first)
    merge_entries(accumulators, second)
    by_expert = {entry.expert: entry for entry in finalize_entries(accumulators)}

    np.testing.assert_allclose(by_expert[0].importance_sum, np.array([1.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(by_expert[1].importance_sum, np.array([34.0, 52.0], dtype=np.float32))
    assert by_expert[0].route_count == 1
    assert by_expert[1].route_count == 2
    assert by_expert[1].total_route_count == 3
    assert by_expert[1].prompt_ids == ("prompt-a", "prompt-b")


def test_accumulate_routed_projection_imatrix_accepts_per_route_down_inputs() -> None:
    down_inputs = np.array(
        [
            [[1.0, 0.0], [2.0, 1.0]],
            [[3.0, 1.0], [4.0, 2.0]],
        ],
        dtype=np.float32,
    )
    route_indices = np.array(
        [
            [0, 1],
            [0, 1],
        ],
        dtype=np.int32,
    )

    entries = accumulate_routed_projection_imatrix(
        layer=8,
        projection="down_proj",
        inputs=down_inputs,
        route_indices=route_indices,
        num_experts=2,
    )
    by_expert = {entry.expert: entry for entry in entries}

    np.testing.assert_allclose(by_expert[0].importance_sum, np.array([10.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(by_expert[1].importance_sum, np.array([20.0, 5.0], dtype=np.float32))
    assert by_expert[0].route_count == 2
    assert by_expert[1].route_count == 2


def test_activation_records_store_imatrix_rows_only_when_requested() -> None:
    indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    moe_input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    down_input = np.array([[[1.0, 0.0], [2.0, 0.0]], [[0.0, 3.0], [0.0, 4.0]]], dtype=np.float32)

    summary_only = _record(
        prompt_id="prompt-a",
        indices=indices,
        moe_input=moe_input,
        down_input=down_input,
        include_activation_rows=False,
    )
    with_rows = _record(
        prompt_id="prompt-a",
        indices=indices,
        moe_input=moe_input,
        down_input=down_input,
        include_activation_rows=True,
    )

    assert "moe_input_rows" not in summary_only
    assert "down_input_rows" not in summary_only
    assert with_rows["moe_input"]["rms"] == summary_only["moe_input"]["rms"]
    assert with_rows["down_input"]["rms"] == summary_only["down_input"]["rms"]
    assert with_rows["moe_input_rows"] == moe_input.tolist()
    assert with_rows["down_input_rows"] == down_input.tolist()


def test_imatrix_entries_from_activation_records_merges_projection_inputs() -> None:
    first = _record(
        prompt_id="prompt-a",
        indices=np.array([[0, 1], [1, 0]], dtype=np.int64),
        moe_input=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        down_input=np.array([[[1.0, 0.0], [2.0, 0.0]], [[0.0, 3.0], [0.0, 4.0]]], dtype=np.float32),
        include_activation_rows=True,
    )
    second = _record(
        prompt_id="prompt-b",
        indices=np.array([[1, 1]], dtype=np.int64),
        moe_input=np.array([[5.0, 6.0]], dtype=np.float32),
        down_input=np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=np.float32),
        include_activation_rows=True,
    )

    entries = imatrix_entries_from_activation_records([first, second])
    by_projection_expert = {(entry.projection, entry.expert): entry for entry in entries}

    assert len(entries) == 6
    np.testing.assert_allclose(
        by_projection_expert[("gate_proj", 0)].importance_sum,
        np.array([10.0, 20.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        by_projection_expert[("gate_proj", 1)].importance_sum,
        np.array([60.0, 92.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        by_projection_expert[("up_proj", 1)].importance_sum,
        by_projection_expert[("gate_proj", 1)].importance_sum,
    )
    np.testing.assert_allclose(
        by_projection_expert[("down_proj", 0)].importance_sum,
        np.array([1.0, 16.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        by_projection_expert[("down_proj", 1)].importance_sum,
        np.array([9.0, 14.0], dtype=np.float32),
    )
    assert by_projection_expert[("gate_proj", 0)].route_count == 2
    assert by_projection_expert[("gate_proj", 1)].route_count == 4
    assert by_projection_expert[("gate_proj", 1)].total_route_count == 6
    assert by_projection_expert[("gate_proj", 1)].prompt_ids == ("prompt-a", "prompt-b")
    np.testing.assert_allclose(
        by_projection_expert[("gate_proj", 1)].routing_weighted_importance,
        np.array([10.0, 92.0 / 6.0], dtype=np.float32),
    )


def test_imatrix_entries_from_activation_records_requires_raw_activation_rows() -> None:
    record = _record(
        prompt_id="prompt-a",
        indices=np.array([[0, 1]], dtype=np.int64),
        moe_input=np.array([[1.0, 2.0]], dtype=np.float32),
        down_input=np.array([[[1.0, 0.0], [2.0, 0.0]]], dtype=np.float32),
        include_activation_rows=False,
    )

    try:
        imatrix_entries_from_activation_records([record])
    except ValueError as exc:
        assert "include_activation_rows" in str(exc)
    else:
        raise AssertionError("expected missing activation rows to fail")


def test_write_projection_imatrix_sidecars_writes_manifest_and_safetensors(tmp_path) -> None:
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    route_indices = np.array([[0], [1]], dtype=np.int32)
    entries = accumulate_routed_projection_imatrix(
        layer=2,
        projection="up_proj",
        inputs=inputs,
        route_indices=route_indices,
        num_experts=2,
        prompt_ids=("imatrix_calib_code_completion_000", "imatrix_calib_code_completion_001"),
    )

    manifest = write_projection_imatrix_sidecars(
        entries,
        output_dir=tmp_path,
        prompt_set="air_imatrix_calib_v1",
    )

    manifest_path = tmp_path / "imatrix-manifest.json"
    first_path = tmp_path / "imatrix" / "layer-00002-up_proj-expert-00000.safetensors"
    first_arrays = mx.load(str(first_path))

    assert manifest_path.exists()
    assert first_path.exists()
    assert manifest["record_type"] == "air_projection_imatrix_manifest"
    assert manifest["prompt_set"] == "air_imatrix_calib_v1"
    assert manifest["entry_count"] == 2
    assert manifest["entries"][0]["path"] == "imatrix/layer-00002-up_proj-expert-00000.safetensors"
    assert manifest["entries"][0]["route_count"] == 1
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["entry_count"] == 2
    np.testing.assert_allclose(np.asarray(first_arrays["importance_sum"]), np.array([1.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(
        np.asarray(first_arrays["routing_weighted_importance"]),
        np.array([0.5, 2.0], dtype=np.float32),
    )


def test_write_projection_imatrix_sidecars_writes_normalized_affinity_tensor(tmp_path) -> None:
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    route_indices = np.array([[0, 1], [0, 1]], dtype=np.int64)
    router_scores = np.array([[0.75, 0.25], [0.10, 0.90]], dtype=np.float32)
    entries = accumulate_routed_projection_imatrix(
        layer=2,
        projection="up_proj",
        inputs=inputs,
        route_indices=route_indices,
        router_scores=router_scores,
        num_experts=2,
        prompt_ids=("imatrix_calib_code_completion_000",),
    )

    manifest = write_projection_imatrix_sidecars(
        entries,
        output_dir=tmp_path,
        prompt_set="air_imatrix_calib_v1",
    )

    first_path = tmp_path / "imatrix" / "layer-00002-up_proj-expert-00000.safetensors"
    first_arrays = mx.load(str(first_path))

    assert manifest["entries"][0]["affinity_score_sum"] == np.float32(0.85)
    assert "affinity_weighted_mean_importance" in manifest["entries"][0]["tensors"]
    np.testing.assert_allclose(
        np.asarray(first_arrays["affinity_weighted_importance"]),
        np.array([1.65, 4.6], dtype=np.float32),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(first_arrays["affinity_weighted_mean_importance"]),
        np.array([1.65, 4.6], dtype=np.float32) / np.float32(0.85),
        rtol=1.0e-6,
    )


def test_air_imatrix_collection_plan_validates_prompt_split_without_writes(tmp_path) -> None:
    output_dir = tmp_path / "planned-imatrix"

    plan = build_air_imatrix_collection_plan(
        output_dir=output_dir,
        layers=(1, 2, 45),
        projections=("gate_proj", "up_proj"),
        num_experts=128,
        collection_source="resident_vq_model",
    )

    assert json.loads(json.dumps(plan, sort_keys=True)) == plan
    assert plan["schema"] == "air_projection_imatrix_collection_plan"
    assert plan["schema_version"] == 1
    assert plan["ok"] is True
    assert plan["collection_allowed"] is False
    assert plan["sidecar_writes_allowed"] is False
    assert plan["prompt_validation"]["ok"] is True
    assert plan["prompt_validation"]["prompt_set"] == "air_imatrix_calib_v1"
    assert plan["prompt_validation"]["prompt_count"] == 96
    assert plan["summary"]["planned_layer_count"] == 3
    assert plan["summary"]["planned_projection_count"] == 2
    assert plan["summary"]["planned_expert_count"] == 128
    assert plan["summary"]["planned_sidecar_count"] == 3 * 2 * 128
    assert plan["collection"]["include_activation_rows"] is True
    assert plan["collection"]["use_chat_template"] is True
    assert plan["filesystem"]["writes_performed"] is False
    assert plan["filesystem"]["output_dir_exists"] is False
    assert plan["filesystem"]["manifest_written"] is False
    assert plan["targets"]["manifest_path"] == str(output_dir / "imatrix-manifest.json")
    assert plan["targets"]["sidecar_path_template"] == "imatrix/layer-{layer:05d}-{projection}-expert-{expert:05d}.safetensors"
    assert not output_dir.exists()

    output_dir.mkdir()
    blocked = build_air_imatrix_collection_plan(
        output_dir=output_dir,
        layers=(1,),
        projections=("gate_proj",),
        num_experts=1,
        collection_source="resident_vq_model",
    )

    assert blocked["ok"] is False
    assert "output_dir already exists" in blocked["errors"]
    assert blocked["filesystem"]["writes_performed"] is False
