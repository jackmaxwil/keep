from __future__ import annotations

import json

import numpy as np
from safetensors.numpy import save_file

import keep.quality.dynamic_precision as dynamic_precision_module
from keep.quality.dynamic_precision import (
    DynamicTensorProfile,
    attach_high_precision_source_manifest_to_materialization_plan,
    attach_imatrix_manifest_to_materialization_plan,
    allocate_dynamic_precision_tiers,
    build_high_precision_source_manifest_from_index,
    build_dynamic_precision_approval_batch_plan,
    build_dynamic_precision_budget_sweep_approval_bundle,
    build_dynamic_precision_eval_frontier_contract,
    build_dynamic_precision_eval_frontier_report,
    build_dynamic_precision_human_approval_request,
    build_dynamic_precision_materialized_eval_plan,
    build_dynamic_precision_materialization_approval_bundle,
    build_dynamic_precision_materialization_preflight,
    build_dynamic_precision_materialization_plan,
    build_dynamic_precision_materialization_dry_run,
    build_dynamic_precision_tier_map_approval_bundle,
    build_dynamic_precision_tier_map_report,
    build_air_dynamic_precision_prior_summary,
    build_air_layer_type_prior_floors,
    default_air_dynamic_precision_tiers,
    imatrix_weighted_squared_error,
    validate_dynamic_precision_approval_batch_objective,
    validate_dynamic_precision_tier_map_report,
)
from keep.quality.imatrix import (
    accumulate_routed_projection_imatrix,
    build_air_imatrix_collection_plan,
    load_projection_imatrix_manifest,
    write_projection_imatrix_sidecars,
)


def _profile(
    key: str,
    *,
    layer: int,
    projection: str,
    expert: int | None = None,
    weight_count: int = 100,
    low_error: float = 10.0,
    mid_error: float = 5.0,
    high_error: float = 1.0,
) -> DynamicTensorProfile:
    return DynamicTensorProfile(
        key=key,
        layer=layer,
        projection=projection,
        weight_count=weight_count,
        weighted_errors={
            "low": low_error,
            "mid": mid_error,
            "high": high_error,
        },
        expert=expert,
    )


def test_default_dynamic_precision_tiers_include_real_high_escape_tier() -> None:
    tiers = default_air_dynamic_precision_tiers()

    assert tuple(tiers) == ("low", "mid", "high")
    assert tiers["low"].artifact_kind == "vq_e8"
    assert tiers["low"].code_bits == 8
    assert tiers["mid"].artifact_kind == "vq_e8p"
    assert tiers["mid"].code_bits == 16
    assert tiers["high"].artifact_kind == "bf16"
    assert tiers["high"].code_bits is None
    assert tiers["low"].effective_bits_per_weight < tiers["mid"].effective_bits_per_weight
    assert tiers["mid"].effective_bits_per_weight < tiers["high"].effective_bits_per_weight


def test_air_layer_type_priors_keep_fragile_glu_down_and_edge_layers_higher() -> None:
    profiles = [
        _profile("1:gate_proj", layer=1, projection="gate_proj"),
        _profile("14:down_proj", layer=14, projection="down_proj"),
        _profile("20:gate_proj", layer=20, projection="gate_proj"),
        _profile("31:up_proj", layer=31, projection="up_proj"),
        _profile("36:gate_proj", layer=36, projection="gate_proj"),
        _profile("41:up_proj", layer=41, projection="up_proj"),
        _profile("45:gate_proj", layer=45, projection="gate_proj"),
    ]

    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    by_key = {floor.key: floor for floor in floors}

    assert by_key["1:gate_proj"].minimum_tier == "mid"
    assert by_key["14:down_proj"].minimum_tier == "mid"
    assert "ffn_down_proj" in by_key["14:down_proj"].rationale
    assert by_key["20:gate_proj"].minimum_tier == "low"
    assert by_key["31:up_proj"].minimum_tier == "high"
    assert by_key["36:gate_proj"].minimum_tier == "high"
    assert by_key["41:up_proj"].minimum_tier == "high"
    assert by_key["45:gate_proj"].minimum_tier == "mid"


def test_air_dynamic_precision_prior_summary_keeps_non_routed_surfaces_fixed_high() -> None:
    profiles = [
        _profile("1:gate_proj", layer=1, projection="gate_proj"),
        _profile("14:down_proj", layer=14, projection="down_proj"),
        _profile("20:gate_proj", layer=20, projection="gate_proj"),
        _profile("41:up_proj", layer=41, projection="up_proj"),
        _profile("45:gate_proj", layer=45, projection="gate_proj"),
    ]

    summary = build_air_dynamic_precision_prior_summary(profiles, total_layers=46)

    assert json.loads(json.dumps(summary, sort_keys=True)) == summary
    assert summary["schema"] == "air_dynamic_precision_prior_summary"
    assert summary["schema_version"] == 1
    assert summary["ok"] is True
    assert summary["materialization_allowed"] is False
    assert summary["allocation_scope"]["profile_scope"] == "routed_expert_gate_up_down"
    assert summary["allocation_scope"]["fixed_high_precision_action"] == "keep_on_existing_dense_source_path"
    routed_floors = {row["key"]: row for row in summary["routed_prior_floors"]}
    assert routed_floors["1:gate_proj"]["minimum_tier"] == "mid"
    assert routed_floors["14:down_proj"]["minimum_tier"] == "mid"
    assert routed_floors["41:up_proj"]["minimum_tier"] == "high"
    assert routed_floors["45:gate_proj"]["minimum_tier"] == "mid"
    fixed_rules = {rule["name"]: rule for rule in summary["fixed_high_precision_priors"]}
    assert set(fixed_rules) == {
        "token_embeddings",
        "output_lm_head",
        "attention_value_output",
        "shared_or_always_on_expert",
        "normalization_and_router",
    }
    assert fixed_rules["token_embeddings"]["tensor_patterns"] == ["model.embed_tokens.weight"]
    assert fixed_rules["output_lm_head"]["tensor_patterns"] == ["lm_head.weight"]
    assert fixed_rules["attention_value_output"]["tensor_patterns"] == [
        "model.layers.*.self_attn.v_proj.weight",
        "model.layers.*.self_attn.o_proj.weight",
    ]
    assert fixed_rules["shared_or_always_on_expert"]["minimum_tier"] == "high"
    assert all(rule["candidate_materialization_action"] == "none_fixed_source_precision" for rule in fixed_rules.values())


def test_greedy_allocator_respects_budget_floors_and_imatrix_weighted_gain() -> None:
    profiles = [
        _profile(
            "41:gate_proj",
            layer=41,
            projection="gate_proj",
            weight_count=10,
            low_error=100.0,
            mid_error=70.0,
            high_error=10.0,
        ),
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            weight_count=100,
            low_error=1000.0,
            mid_error=100.0,
            high_error=90.0,
        ),
        _profile(
            "21:up_proj",
            layer=21,
            projection="up_proj",
            weight_count=100,
            low_error=500.0,
            mid_error=400.0,
            high_error=390.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)

    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=3.2,
        prior_floors=floors,
    )

    assert plan.tier_by_key == {
        "20:gate_proj": "mid",
        "21:up_proj": "low",
        "41:gate_proj": "high",
    }
    assert plan.total_effective_bits <= plan.budget_effective_bits
    assert plan.effective_bits_per_weight <= 3.2
    assert plan.upgrades[0]["key"] == "41:gate_proj"
    assert plan.upgrades[0]["reason"] == "prior_floor"
    assert plan.upgrades[1]["key"] == "20:gate_proj"
    assert plan.upgrades[1]["reason"] == "greedy_gain_per_bit"


def test_tier_map_report_is_json_ready_and_marks_materialization_gated() -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )

    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )

    assert json.loads(json.dumps(report, sort_keys=True)) == report
    assert report["schema"] == "air_dynamic_precision_tier_map_report"
    assert report["source"] == "synthetic_unit_test"
    assert report["materialization"] == {
        "candidate_materialized": False,
        "artifact_written": False,
        "approval_required": True,
        "approval_gate": "real_imatrix_sidecar_collection_and_candidate_materialization",
    }
    assert report["tier_map"] == {
        "20:gate_proj": "mid",
        "41:up_proj": "high",
    }
    assert report["budget"]["total_weight_count"] == 110
    assert report["budget"]["effective_bits_per_weight"] == plan.effective_bits_per_weight
    assert report["tiers"]["high"]["artifact_kind"] == "bf16"
    assert report["tiers"]["high"]["code_bits"] is None
    assert report["tier_counts"] == {"low": 0, "mid": 1, "high": 1}
    assert report["upgrades"][0]["reason"] == "prior_floor"

    tensor_rows = {row["key"]: row for row in report["tensors"]}
    assert tensor_rows["20:gate_proj"]["selected_tier"] == "mid"
    assert tensor_rows["20:gate_proj"]["selected_weighted_error"] == 10.0
    assert tensor_rows["20:gate_proj"]["expert"] == 7
    assert tensor_rows["20:gate_proj"]["prior_floor"]["minimum_tier"] == "low"
    assert tensor_rows["41:up_proj"]["selected_tier"] == "high"
    assert tensor_rows["41:up_proj"]["prior_floor"]["minimum_tier"] == "high"
    assert "fragile_glu_layer" in tensor_rows["41:up_proj"]["prior_floor"]["rationale"]


def test_tier_map_report_validator_rejects_materialization_and_floor_mismatches() -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )
    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )

    valid = validate_dynamic_precision_tier_map_report(report)

    assert valid["ok"] is True
    assert valid["tensor_count"] == 2
    assert valid["errors"] == []

    corrupted = json.loads(json.dumps(report))
    corrupted["materialization"]["approval_required"] = False
    corrupted["tier_map"]["41:up_proj"] = "mid"
    for row in corrupted["tensors"]:
        if row["key"] == "41:up_proj":
            row["selected_tier"] = "mid"
            row["selected_weighted_error"] = 40.0
            row["selected_artifact_kind"] = "vq_e8p"
            row["selected_code_bits"] = 16
            row["selected_effective_bits_per_weight"] = 3.0
    corrupted["tier_counts"] = {"low": 0, "mid": 2, "high": 0}

    invalid = validate_dynamic_precision_tier_map_report(corrupted)

    assert invalid["ok"] is False
    assert "materialization.approval_required must be true" in invalid["errors"]
    assert "41:up_proj: selected tier mid is below prior floor high" in invalid["errors"]


def test_materialization_plan_is_in_memory_and_classifies_tier_actions() -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )
    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )

    materialization_plan = build_dynamic_precision_materialization_plan(
        report,
        candidate_name="synthetic-dynamic-v1",
    )

    assert json.loads(json.dumps(materialization_plan, sort_keys=True)) == materialization_plan
    assert materialization_plan["schema"] == "air_dynamic_precision_materialization_plan"
    assert materialization_plan["candidate"] == {
        "name": "synthetic-dynamic-v1",
        "candidate_materialized": False,
        "artifact_written": False,
        "approval_required": True,
        "approval_gate": "real_imatrix_sidecar_collection_and_candidate_materialization",
    }
    assert materialization_plan["source_report"] == {
        "schema": "air_dynamic_precision_tier_map_report",
        "schema_version": 1,
        "source": "synthetic_unit_test",
    }
    assert materialization_plan["summary"] == {
        "action_count": 2,
        "vq_reencode_action_count": 1,
        "high_precision_action_count": 1,
        "total_weight_count": 110,
        "effective_bits_per_weight": report["budget"]["effective_bits_per_weight"],
        "weighted_error": report["budget"]["weighted_error"],
    }

    actions = {row["key"]: row for row in materialization_plan["actions"]}
    assert actions["20:gate_proj"]["materialization_action"] == "vq_reencode_with_imatrix"
    assert actions["20:gate_proj"]["requires_imatrix_sidecar"] is True
    assert actions["20:gate_proj"]["requires_high_precision_source"] is False
    assert actions["20:gate_proj"]["artifact_kind"] == "vq_e8p"
    assert actions["20:gate_proj"]["code_bits"] == 16
    assert actions["41:up_proj"]["materialization_action"] == "materialize_high_precision_projection"
    assert actions["41:up_proj"]["requires_imatrix_sidecar"] is False
    assert actions["41:up_proj"]["requires_high_precision_source"] is True
    assert actions["41:up_proj"]["approval_required"] is True

    corrupted = json.loads(json.dumps(report))
    corrupted["materialization"]["approval_required"] = False

    invalid_plan = build_dynamic_precision_materialization_plan(
        corrupted,
        candidate_name="bad",
    )

    assert invalid_plan["ok"] is False
    assert "materialization.approval_required must be true" in invalid_plan["errors"]
    assert invalid_plan["actions"] == []


def test_materialization_plan_joins_synthetic_imatrix_manifest_without_materializing(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )
    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )
    materialization_plan = build_dynamic_precision_materialization_plan(
        report,
        candidate_name="synthetic-dynamic-v1",
    )
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=tmp_path,
        prompt_set="air_imatrix_calib_v1",
    )
    manifest = load_projection_imatrix_manifest(tmp_path / "imatrix-manifest.json")

    joined = attach_imatrix_manifest_to_materialization_plan(
        materialization_plan,
        manifest,
        manifest_root=tmp_path,
        require_sidecar_files=True,
    )

    assert json.loads(json.dumps(joined, sort_keys=True)) == joined
    assert joined["ok"] is True
    assert joined["candidate"]["candidate_materialized"] is False
    assert joined["imatrix_manifest"] == {
        "record_type": "air_projection_imatrix_manifest",
        "prompt_set": "air_imatrix_calib_v1",
        "entry_count": 8,
        "covered_action_count": 1,
        "missing_action_count": 0,
        "sidecar_files_checked": True,
    }

    actions = {row["key"]: row for row in joined["actions"]}
    assert actions["20:gate_proj"]["imatrix_sidecar_available"] is True
    assert actions["20:gate_proj"]["imatrix_sidecar"] == {
        "path": "imatrix/layer-00020-gate_proj-expert-00007.safetensors",
        "input_dim": 3,
        "route_count": 1,
        "total_route_count": 1,
        "route_frequency": 1.0,
        "prompt_ids": ["imatrix_calib_code_debug_000"],
    }
    assert actions["41:up_proj"]["imatrix_sidecar_available"] is False
    assert actions["41:up_proj"]["imatrix_sidecar"] is None
    assert actions["41:up_proj"]["requires_high_precision_source"] is True

    missing_manifest = json.loads(json.dumps(manifest))
    missing_manifest["entries"] = [
        row
        for row in missing_manifest["entries"]
        if not (row["layer"] == 20 and row["projection"] == "gate_proj" and row["expert"] == 7)
    ]
    missing_manifest["entry_count"] = len(missing_manifest["entries"])

    missing = attach_imatrix_manifest_to_materialization_plan(
        materialization_plan,
        missing_manifest,
        manifest_root=tmp_path,
        require_sidecar_files=True,
    )

    assert missing["ok"] is False
    assert "missing imatrix sidecar for 20:gate_proj layer=20 projection=gate_proj expert=7" in missing["errors"]
    assert missing["imatrix_manifest"]["missing_action_count"] == 1


def test_materialization_dry_run_plans_paths_without_creating_artifacts(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )
    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )
    materialization_plan = build_dynamic_precision_materialization_plan(
        report,
        candidate_name="synthetic-dynamic-v1",
    )
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=tmp_path,
        prompt_set="air_imatrix_calib_v1",
    )
    joined_plan = attach_imatrix_manifest_to_materialization_plan(
        materialization_plan,
        load_projection_imatrix_manifest(tmp_path / "imatrix-manifest.json"),
        manifest_root=tmp_path,
        require_sidecar_files=True,
    )
    source_dir = tmp_path / "source"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"synthetic high precision source placeholder")
    joined_plan = attach_high_precision_source_manifest_to_materialization_plan(
        joined_plan,
        {
            "schema_version": 1,
            "record_type": "air_dynamic_precision_high_source_manifest",
            "source_kind": "synthetic_high_precision_source",
            "entry_count": 1,
            "entries": [
                {
                    "layer": 41,
                    "projection": "up_proj",
                    "expert": 11,
                    "path": "model-00001-of-00001.safetensors",
                    "tensor_name": "model.layers.41.mlp.experts.11.up_proj.weight",
                    "dtype": "bf16",
                    "shape": [4, 3],
                }
            ],
        },
        source_root=source_dir,
        require_source_files=True,
    )
    seed_dir = tmp_path / "seed"
    output_dir = tmp_path / "candidate"

    dry_run = build_dynamic_precision_materialization_dry_run(
        joined_plan,
        source_dir=source_dir,
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
    )

    assert json.loads(json.dumps(dry_run, sort_keys=True)) == dry_run
    assert dry_run["schema"] == "air_dynamic_precision_materialization_dry_run"
    assert dry_run["ok"] is True
    assert dry_run["mode"] == "dry_run"
    assert dry_run["materialization_allowed"] is False
    assert dry_run["candidate"] == {
        "name": "synthetic-dynamic-v1",
        "candidate_materialized": False,
        "artifact_written": False,
        "approval_required": True,
        "approval_gate": "real_imatrix_sidecar_collection_and_candidate_materialization",
    }
    assert dry_run["filesystem"] == {
        "writes_performed": False,
        "candidate_dir_created": False,
        "seed_artifact_mutated": False,
        "candidate_artifact_mutated": False,
        "source_dir": str(source_dir),
        "seed_artifact_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "source_dir_exists": True,
        "seed_artifact_dir_exists": False,
        "output_dir_exists": False,
    }
    assert dry_run["summary"]["planned_action_count"] == 2
    assert dry_run["summary"]["planned_vq_reencode_count"] == 1
    assert dry_run["summary"]["planned_high_precision_count"] == 1
    assert dry_run["summary"]["required_imatrix_sidecar_count"] == 1
    assert dry_run["summary"]["available_imatrix_sidecar_count"] == 1
    assert dry_run["summary"]["required_high_precision_source_count"] == 1
    assert dry_run["summary"]["available_high_precision_source_count"] == 1

    planned = {row["key"]: row for row in dry_run["planned_outputs"]}
    assert planned["20:gate_proj"]["would_write"] is True
    assert planned["20:gate_proj"]["output_path"] == str(output_dir / "layer-00020-gate_proj.safetensors")
    assert planned["20:gate_proj"]["source"] == "seed_artifact_plus_imatrix_sidecar"
    assert planned["20:gate_proj"]["imatrix_sidecar_path"] == "imatrix/layer-00020-gate_proj-expert-00007.safetensors"
    assert planned["41:up_proj"]["output_path"] == str(output_dir / "layer-00041-up_proj-high.safetensors")
    assert planned["41:up_proj"]["source"] == "high_precision_source"
    assert planned["41:up_proj"]["imatrix_sidecar_path"] is None
    assert planned["41:up_proj"]["high_precision_source_path"] == "model-00001-of-00001.safetensors"
    assert planned["41:up_proj"]["high_precision_source_tensor"] == "model.layers.41.mlp.experts.11.up_proj.weight"
    assert source_dir.exists()
    assert not seed_dir.exists()
    assert not output_dir.exists()

    blocked = build_dynamic_precision_materialization_dry_run(
        {**joined_plan, "ok": False, "errors": ["missing imatrix sidecar for x"]},
        source_dir=source_dir,
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
    )

    assert blocked["ok"] is False
    assert blocked["planned_outputs"] == []
    assert "materialization_plan must be ok before dry-run path planning" in blocked["errors"]


def test_materialization_preflight_requests_approval_without_writes(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )
    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )
    materialization_plan = build_dynamic_precision_materialization_plan(
        report,
        candidate_name="synthetic-dynamic-v1",
    )
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=tmp_path,
        prompt_set="air_imatrix_calib_v1",
    )
    joined_plan = attach_imatrix_manifest_to_materialization_plan(
        materialization_plan,
        load_projection_imatrix_manifest(tmp_path / "imatrix-manifest.json"),
        manifest_root=tmp_path,
        require_sidecar_files=True,
    )
    source_dir = tmp_path / "source"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"synthetic high precision source placeholder")
    joined_plan = attach_high_precision_source_manifest_to_materialization_plan(
        joined_plan,
        {
            "schema_version": 1,
            "record_type": "air_dynamic_precision_high_source_manifest",
            "source_kind": "synthetic_high_precision_source",
            "entry_count": 1,
            "entries": [
                {
                    "layer": 41,
                    "projection": "up_proj",
                    "expert": 11,
                    "path": "model-00001-of-00001.safetensors",
                    "tensor_name": "model.layers.41.mlp.experts.11.up_proj.weight",
                    "dtype": "bf16",
                    "shape": [4, 3],
                }
            ],
        },
        source_root=source_dir,
        require_source_files=True,
    )
    dry_run = build_dynamic_precision_materialization_dry_run(
        joined_plan,
        source_dir=source_dir,
        seed_artifact_dir=tmp_path / "seed",
        output_dir=tmp_path / "candidate",
    )

    preflight = build_dynamic_precision_materialization_preflight(joined_plan, dry_run)

    assert json.loads(json.dumps(preflight, sort_keys=True)) == preflight
    assert preflight["schema"] == "air_dynamic_precision_materialization_preflight"
    assert preflight["schema_version"] == 1
    assert preflight["ok"] is True
    assert preflight["ready_for_approval_request"] is True
    assert preflight["materialization_allowed"] is False
    assert preflight["materialization_state"] == {
        "candidate_materialized": False,
        "artifact_written": False,
        "writes_performed": False,
        "candidate_dir_created": False,
        "seed_artifact_mutated": False,
        "candidate_artifact_mutated": False,
    }
    assert preflight["approval_request"] == {
        "approval_required": True,
        "approval_gate": "real_imatrix_sidecar_collection_and_candidate_materialization",
        "ready": True,
        "requested_operations": [
            "collect_real_air_imatrix_calib_v1_imatrix_sidecars",
            "build_real_high_precision_source_manifest_from_source_index",
            "run_real_metadata_validation_join_and_dry_run",
            "materialize_dynamic_mixed_precision_candidate_in_fresh_artifact_dir",
            "evaluate_report_and_selection_caches_and_budget_sweep",
        ],
    }
    assert preflight["blocked_operations"] == [
        "real_imatrix_sidecar_collection",
        "real_high_precision_source_tensor_reads",
        "persisted_tier_map_or_report_artifact_writes",
        "high_tier_shard_generation",
        "candidate_directory_creation",
        "report_selection_cache_evaluation_and_budget_sweep",
    ]
    assert preflight["summary"]["planned_action_count"] == 2
    assert preflight["summary"]["planned_vq_reencode_count"] == 1
    assert preflight["summary"]["planned_high_precision_count"] == 1

    blocked = build_dynamic_precision_materialization_preflight(
        joined_plan,
        {**dry_run, "ok": False, "errors": ["output_dir already exists"]},
    )

    assert blocked["ok"] is False
    assert blocked["ready_for_approval_request"] is False
    assert blocked["approval_request"]["ready"] is False
    assert "dry_run must be ok before preflight" in blocked["errors"]
    assert "output_dir already exists" in blocked["errors"]


def test_high_precision_source_manifest_builder_uses_source_index_headers(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )
    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )
    materialization_plan = build_dynamic_precision_materialization_plan(
        report,
        candidate_name="synthetic-dynamic-v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)

    manifest = build_high_precision_source_manifest_from_index(
        materialization_plan,
        {
            "metadata": {"total_size": 48},
            "weight_map": {
                source_tensor: source_file.name,
                "model.layers.41.input_layernorm.weight": source_file.name,
            },
        },
        source_root=source_dir,
        source_kind="synthetic_source_index",
        inspect_source_headers=True,
    )

    assert json.loads(json.dumps(manifest, sort_keys=True)) == manifest
    assert manifest["record_type"] == "air_dynamic_precision_high_source_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["ok"] is True
    assert manifest["source_kind"] == "synthetic_source_index"
    assert manifest["source_headers_checked"] is True
    assert manifest["required_action_count"] == 1
    assert manifest["covered_action_count"] == 1
    assert manifest["missing_action_count"] == 0
    assert manifest["entry_count"] == 1
    assert manifest["entries"] == [
        {
            "layer": 41,
            "projection": "up_proj",
            "expert": 11,
            "path": "model-00001-of-00001.safetensors",
            "tensor_name": source_tensor,
            "dtype": "f32",
            "shape": [4, 3],
        }
    ]

    joined = attach_high_precision_source_manifest_to_materialization_plan(
        materialization_plan,
        manifest,
        source_root=source_dir,
        require_source_files=True,
    )

    assert joined["ok"] is True
    assert joined["high_precision_source_manifest"]["covered_action_count"] == 1

    missing = build_high_precision_source_manifest_from_index(
        materialization_plan,
        {"metadata": {}, "weight_map": {}},
        source_root=source_dir,
        source_kind="synthetic_source_index",
        inspect_source_headers=True,
    )

    assert missing["ok"] is False
    assert missing["entry_count"] == 0
    assert missing["missing_action_count"] == 1
    assert (
        "missing high-precision source tensor model.layers.41.mlp.experts.11.up_proj.weight"
        in missing["errors"]
    )


def test_materialization_approval_bundle_composes_no_write_preflight(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )
    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )
    materialization_plan = build_dynamic_precision_materialization_plan(
        report,
        candidate_name="synthetic-dynamic-v1",
    )
    imatrix_dir = tmp_path / "imatrix"
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=imatrix_dir,
        prompt_set="air_imatrix_calib_v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)
    seed_dir = tmp_path / "seed"
    output_dir = tmp_path / "candidate"

    bundle = build_dynamic_precision_materialization_approval_bundle(
        materialization_plan,
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )

    assert json.loads(json.dumps(bundle, sort_keys=True)) == bundle
    assert bundle["schema"] == "air_dynamic_precision_materialization_approval_bundle"
    assert bundle["schema_version"] == 1
    assert bundle["ok"] is True
    assert bundle["materialization_allowed"] is False
    assert bundle["imatrix_join"]["ok"] is True
    assert bundle["high_precision_source_manifest"]["ok"] is True
    assert bundle["source_join"]["ok"] is True
    assert bundle["dry_run"]["ok"] is True
    assert bundle["preflight"]["ready_for_approval_request"] is True
    assert bundle["approval_request"] == bundle["preflight"]["approval_request"]
    assert bundle["summary"]["planned_action_count"] == 2
    assert bundle["summary"]["planned_vq_reencode_count"] == 1
    assert bundle["summary"]["planned_high_precision_count"] == 1
    assert not seed_dir.exists()
    assert not output_dir.exists()

    blocked = build_dynamic_precision_materialization_approval_bundle(
        materialization_plan,
        imatrix_manifest={
            "schema_version": 1,
            "record_type": "air_projection_imatrix_manifest",
            "prompt_set": "air_imatrix_calib_v1",
            "entry_count": 0,
            "entries": [],
        },
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )

    assert blocked["ok"] is False
    assert blocked["preflight"]["ready_for_approval_request"] is False
    assert blocked["dry_run"]["planned_outputs"] == []
    assert "missing imatrix sidecar for 20:gate_proj layer=20 projection=gate_proj expert=7" in blocked["errors"]


def test_tier_map_report_approval_bundle_validates_report_before_preflight(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )
    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )
    imatrix_dir = tmp_path / "imatrix"
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=imatrix_dir,
        prompt_set="air_imatrix_calib_v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)
    seed_dir = tmp_path / "seed"
    output_dir = tmp_path / "candidate"

    bundle = build_dynamic_precision_tier_map_approval_bundle(
        report,
        candidate_name="synthetic-dynamic-v1",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )

    assert json.loads(json.dumps(bundle, sort_keys=True)) == bundle
    assert bundle["schema"] == "air_dynamic_precision_tier_map_approval_bundle"
    assert bundle["schema_version"] == 1
    assert bundle["ok"] is True
    assert bundle["report_validation"]["ok"] is True
    assert bundle["materialization_plan"]["ok"] is True
    assert bundle["materialization_bundle"]["ok"] is True
    assert bundle["approval_request"] == bundle["materialization_bundle"]["approval_request"]
    assert bundle["summary"]["planned_action_count"] == 2
    assert bundle["materialization_allowed"] is False
    assert not seed_dir.exists()
    assert not output_dir.exists()

    invalid_report = json.loads(json.dumps(report))
    invalid_report["materialization"]["candidate_materialized"] = True
    blocked = build_dynamic_precision_tier_map_approval_bundle(
        invalid_report,
        candidate_name="synthetic-dynamic-v1",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )

    assert blocked["ok"] is False
    assert blocked["report_validation"]["ok"] is False
    assert blocked["materialization_plan"]["ok"] is False
    assert blocked["materialization_bundle"]["dry_run"]["planned_outputs"] == []
    assert "materialization.candidate_materialized must be false" in blocked["errors"]


def test_budget_sweep_approval_bundle_batches_budget_frontier_without_writes(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    reports = [
        build_dynamic_precision_tier_map_report(
            profiles,
            allocate_dynamic_precision_tiers(
                profiles,
                budget_bits_per_weight=budget,
                prior_floors=floors,
            ),
            prior_floors=floors,
            source=f"synthetic_budget_{budget}",
        )
        for budget in (3.3, 4.2)
    ]
    imatrix_dir = tmp_path / "imatrix"
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=imatrix_dir,
        prompt_set="air_imatrix_calib_v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)
    seed_dir = tmp_path / "seed"
    output_root = tmp_path / "candidates"

    bundle = build_dynamic_precision_budget_sweep_approval_bundle(
        reports,
        candidate_name_prefix="synthetic-dynamic-sweep",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=seed_dir,
        output_root=output_root,
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )

    assert json.loads(json.dumps(bundle, sort_keys=True)) == bundle
    assert bundle["schema"] == "air_dynamic_precision_budget_sweep_approval_bundle"
    assert bundle["schema_version"] == 1
    assert bundle["ok"] is True
    assert bundle["materialization_allowed"] is False
    assert bundle["summary"]["budget_count"] == 2
    assert bundle["summary"]["ready_bundle_count"] == 2
    assert [row["requested_budget_bits_per_weight"] for row in bundle["budget_sweep"]] == [3.3, 4.2]
    assert bundle["budget_sweep"][0]["candidate_name"] != bundle["budget_sweep"][1]["candidate_name"]
    assert bundle["budget_sweep"][0]["effective_bits_per_weight"] < bundle["budget_sweep"][1]["effective_bits_per_weight"]
    assert bundle["budget_sweep"][0]["weighted_error"] > bundle["budget_sweep"][1]["weighted_error"]
    assert all(entry["bundle_ok"] for entry in bundle["budget_sweep"])
    assert len(bundle["bundles"]) == 2
    assert bundle["approval_request"]["ready"] is True
    assert not seed_dir.exists()
    assert not output_root.exists()

    invalid_reports = json.loads(json.dumps(reports))
    invalid_reports[1]["materialization"]["candidate_materialized"] = True
    blocked = build_dynamic_precision_budget_sweep_approval_bundle(
        invalid_reports,
        candidate_name_prefix="synthetic-dynamic-sweep",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=seed_dir,
        output_root=output_root,
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )

    assert blocked["ok"] is False
    assert blocked["summary"]["ready_bundle_count"] == 1
    assert blocked["budget_sweep"][1]["bundle_ok"] is False
    assert blocked["bundles"][1]["materialization_plan"]["ok"] is False
    assert blocked["bundles"][1]["materialization_bundle"]["dry_run"]["planned_outputs"] == []
    assert "materialization.candidate_materialized must be false" in blocked["errors"]
    assert not output_root.exists()


def test_materialized_eval_plan_requires_report_selection_metrics_without_running_eval(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    reports = [
        build_dynamic_precision_tier_map_report(
            profiles,
            allocate_dynamic_precision_tiers(
                profiles,
                budget_bits_per_weight=budget,
                prior_floors=floors,
            ),
            prior_floors=floors,
            source=f"synthetic_budget_{budget}",
        )
        for budget in (3.3, 4.2)
    ]
    imatrix_dir = tmp_path / "imatrix"
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=imatrix_dir,
        prompt_set="air_imatrix_calib_v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)
    sweep = build_dynamic_precision_budget_sweep_approval_bundle(
        reports,
        candidate_name_prefix="synthetic-dynamic-sweep",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=tmp_path / "seed",
        output_root=tmp_path / "candidates",
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )

    plan = build_dynamic_precision_materialized_eval_plan(
        sweep,
        report_cache_path=tmp_path / "report-cache.jsonl",
        selection_cache_path=tmp_path / "selection-cache.jsonl",
        lane_s_speed_gate={"scenario": "prefill_1k", "max_ratio_vs_q2": 1.5},
    )

    assert json.loads(json.dumps(plan, sort_keys=True)) == plan
    assert plan["schema"] == "air_dynamic_precision_materialized_eval_plan"
    assert plan["schema_version"] == 1
    assert plan["ok"] is True
    assert plan["evaluation_allowed"] is False
    assert plan["materialization_allowed"] is False
    assert plan["metric_contract"]["primary_metric"] == "mean_kld"
    assert plan["metric_contract"]["required_metrics"] == [
        "mean_kld",
        "p999_kld",
        "mean_ppl_ratio",
        "max_ppl_ratio",
        "mean_top1_agreement",
        "clean_row_count",
        "effective_bits_per_weight",
        "lane_s_speed",
    ]
    assert plan["acceptance"]["top1_agreement_min"] == 0.75
    assert plan["budget"]["target_effective_bits_per_weight"] == 2.4
    assert plan["budget"]["max_effective_bits_per_weight"] == 3.0
    assert plan["caches"]["report"]["prompt_set"] == "air_vq_ladder_report_v1"
    assert plan["caches"]["report"]["expected_row_count"] == 128
    assert plan["caches"]["selection"]["prompt_set"] == "air_vq_ladder_select_v1"
    assert plan["caches"]["selection"]["expected_row_count"] == 128
    assert plan["caches"]["report"]["payload_read"] is False
    assert plan["summary"]["candidate_count"] == 2
    assert len(plan["candidate_eval_plan"]) == 2
    assert all(row["splits"]["report"]["primary"] is True for row in plan["candidate_eval_plan"])
    assert all(row["eval_records_written"] is False for row in plan["candidate_eval_plan"])
    assert plan["selection_rule"]["rank_by"] == ["mean_kld", "p999_kld", "effective_bits_per_weight"]
    assert "pick_smallest_budget_clearing_top1_and_kld" in plan["selection_rule"]["decision"]
    assert not (tmp_path / "report-cache.jsonl").exists()
    assert not (tmp_path / "selection-cache.jsonl").exists()

    blocked = build_dynamic_precision_materialized_eval_plan(
        {**sweep, "ok": False, "errors": ["synthetic sweep failure"]},
        report_cache_path=tmp_path / "report-cache.jsonl",
        selection_cache_path=tmp_path / "selection-cache.jsonl",
    )

    assert blocked["ok"] is False
    assert blocked["candidate_eval_plan"] == []
    assert "sweep_bundle must be ok before materialized eval planning" in blocked["errors"]
    assert "synthetic sweep failure" in blocked["errors"]


def test_eval_frontier_report_ranks_by_report_kld_after_top1_budget_and_speed_gates(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    reports = [
        build_dynamic_precision_tier_map_report(
            profiles,
            allocate_dynamic_precision_tiers(
                profiles,
                budget_bits_per_weight=budget,
                prior_floors=floors,
            ),
            prior_floors=floors,
            source=f"synthetic_budget_{budget}",
        )
        for budget in (3.3, 4.2, 4.4)
    ]
    imatrix_dir = tmp_path / "imatrix"
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=imatrix_dir,
        prompt_set="air_imatrix_calib_v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)
    sweep = build_dynamic_precision_budget_sweep_approval_bundle(
        reports,
        candidate_name_prefix="synthetic-dynamic-sweep",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=tmp_path / "seed",
        output_root=tmp_path / "candidates",
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )
    eval_plan = build_dynamic_precision_materialized_eval_plan(
        sweep,
        report_cache_path=tmp_path / "report-cache.jsonl",
        selection_cache_path=tmp_path / "selection-cache.jsonl",
        max_effective_bits_per_weight=4.5,
        lane_s_speed_gate={"scenario": "prefill_1k", "max_ratio_vs_q2": 1.5},
    )
    candidate_names = [row["candidate_name"] for row in eval_plan["candidate_eval_plan"]]

    def split_metrics(*, mean_kld: float, p999_kld: float, top1: float, bits: float) -> dict[str, object]:
        return {
            "mean_kld": mean_kld,
            "p999_kld": p999_kld,
            "mean_ppl_ratio": 1.2,
            "max_ppl_ratio": 1.6,
            "mean_top1_agreement": top1,
            "clean_row_count": 128,
            "effective_bits_per_weight": bits,
            "lane_s_speed": {"ok": True, "ratio_vs_q2": 1.35},
        }

    results = [
        {
            "candidate_name": candidate_names[0],
            "splits": {
                "report": split_metrics(mean_kld=0.35, p999_kld=2.1, top1=0.74, bits=3.3),
                "selection": split_metrics(mean_kld=0.37, p999_kld=2.2, top1=0.80, bits=3.3),
            },
        },
        {
            "candidate_name": candidate_names[1],
            "splits": {
                "report": split_metrics(mean_kld=0.41, p999_kld=2.0, top1=0.78, bits=4.2),
                "selection": split_metrics(mean_kld=0.43, p999_kld=2.1, top1=0.79, bits=4.2),
            },
        },
        {
            "candidate_name": candidate_names[2],
            "splits": {
                "report": split_metrics(mean_kld=0.50, p999_kld=1.9, top1=0.82, bits=4.4),
                "selection": split_metrics(mean_kld=0.51, p999_kld=2.0, top1=0.83, bits=4.4),
            },
        },
    ]

    frontier = build_dynamic_precision_eval_frontier_report(eval_plan, results)

    assert json.loads(json.dumps(frontier, sort_keys=True)) == frontier
    assert frontier["schema"] == "air_dynamic_precision_eval_frontier_report"
    assert frontier["schema_version"] == 1
    assert frontier["ok"] is True
    assert frontier["candidate_acceptance_claimed"] is False
    assert frontier["decision"]["status"] == "candidate_selected"
    assert frontier["decision"]["selected_candidate_name"] == candidate_names[1]
    assert frontier["decision"]["rank_by"] == ["report.mean_kld", "report.p999_kld", "effective_bits_per_weight"]
    assert [row["candidate_name"] for row in frontier["frontier"]] == candidate_names
    assert frontier["frontier"][0]["eligible"] is False
    assert "report.mean_top1_agreement below threshold" in frontier["frontier"][0]["failures"]
    assert frontier["frontier"][1]["eligible"] is True
    assert frontier["frontier"][2]["eligible"] is True
    assert frontier["summary"]["eligible_candidate_count"] == 2

    all_blocked = build_dynamic_precision_eval_frontier_report(
        eval_plan,
        [
            {
                **row,
                "splits": {
                    split: {**metrics, "mean_top1_agreement": 0.70}
                    for split, metrics in row["splits"].items()
                },
            }
            for row in results
        ],
    )

    assert all_blocked["ok"] is True
    assert all_blocked["decision"]["status"] == "escalate_no_candidate"
    assert all_blocked["decision"]["selected_candidate_name"] is None
    assert all_blocked["summary"]["eligible_candidate_count"] == 0


def test_approval_batch_carries_post_eval_frontier_contract_without_result_rows(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    reports = [
        build_dynamic_precision_tier_map_report(
            profiles,
            allocate_dynamic_precision_tiers(
                profiles,
                budget_bits_per_weight=budget,
                prior_floors=floors,
            ),
            prior_floors=floors,
            source=f"synthetic_budget_{budget}",
        )
        for budget in (3.3, 4.2)
    ]
    imatrix_dir = tmp_path / "imatrix"
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=imatrix_dir,
        prompt_set="air_imatrix_calib_v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)
    sweep = build_dynamic_precision_budget_sweep_approval_bundle(
        reports,
        candidate_name_prefix="synthetic-dynamic-sweep",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=tmp_path / "seed",
        output_root=tmp_path / "candidates",
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )
    eval_plan = build_dynamic_precision_materialized_eval_plan(
        sweep,
        report_cache_path=tmp_path / "report-cache.jsonl",
        selection_cache_path=tmp_path / "selection-cache.jsonl",
        lane_s_speed_gate={"scenario": "prefill_1k", "max_ratio_vs_q2": 1.5},
    )
    collection_plan = build_air_imatrix_collection_plan(
        output_dir=tmp_path / "planned-imatrix",
        layers=range(1, 46),
        projections=("gate_proj", "up_proj", "down_proj"),
        num_experts=128,
        collection_source="resident_vq_model",
    )

    contract = build_dynamic_precision_eval_frontier_contract(eval_plan)
    batch = build_dynamic_precision_approval_batch_plan(
        imatrix_collection_plan=collection_plan,
        sweep_bundle=sweep,
        eval_plan=eval_plan,
    )

    assert json.loads(json.dumps(contract, sort_keys=True)) == contract
    assert contract["schema"] == "air_dynamic_precision_eval_frontier_contract"
    assert contract["schema_version"] == 1
    assert contract["ok"] is True
    assert contract["awaits_eval_results"] is True
    assert contract["result_artifact_written"] is False
    assert contract["candidate_acceptance_claimed"] is False
    assert contract["candidate_count"] == 2
    assert contract["required_splits"] == ["report", "selection"]
    assert contract["rank_by"] == ["report.mean_kld", "report.p999_kld", "effective_bits_per_weight"]
    assert contract["gates"]["top1_agreement_min"] == 0.75
    assert contract["gates"]["lane_s_speed_required"] is True
    assert contract["gates"]["budget"]["max_effective_bits_per_weight"] == 3.0
    assert "candidate_acceptance_or_ranking_claim" in contract["blocked_operations"]
    assert contract["decision_statuses"] == ["candidate_selected", "escalate_no_candidate", "invalid"]

    assert batch["plans"]["eval_frontier_contract"] == contract
    assert batch["summary"]["frontier_contract_ready"] is True
    assert batch["state"]["candidate_acceptance_allowed"] is False


def test_approval_batch_plan_composes_collection_materialization_and_eval_without_writes(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    reports = [
        build_dynamic_precision_tier_map_report(
            profiles,
            allocate_dynamic_precision_tiers(
                profiles,
                budget_bits_per_weight=budget,
                prior_floors=floors,
            ),
            prior_floors=floors,
            source=f"synthetic_budget_{budget}",
        )
        for budget in (3.3, 4.2)
    ]
    imatrix_dir = tmp_path / "imatrix"
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=imatrix_dir,
        prompt_set="air_imatrix_calib_v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)
    sweep = build_dynamic_precision_budget_sweep_approval_bundle(
        reports,
        candidate_name_prefix="synthetic-dynamic-sweep",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=tmp_path / "seed",
        output_root=tmp_path / "candidates",
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )
    eval_plan = build_dynamic_precision_materialized_eval_plan(
        sweep,
        report_cache_path=tmp_path / "report-cache.jsonl",
        selection_cache_path=tmp_path / "selection-cache.jsonl",
        lane_s_speed_gate={"scenario": "prefill_1k", "max_ratio_vs_q2": 1.5},
    )
    collection_plan = build_air_imatrix_collection_plan(
        output_dir=tmp_path / "planned-imatrix",
        layers=range(1, 46),
        projections=("gate_proj", "up_proj", "down_proj"),
        num_experts=128,
        collection_source="resident_vq_model",
    )

    batch = build_dynamic_precision_approval_batch_plan(
        imatrix_collection_plan=collection_plan,
        sweep_bundle=sweep,
        eval_plan=eval_plan,
    )

    assert json.loads(json.dumps(batch, sort_keys=True)) == batch
    assert batch["schema"] == "air_dynamic_precision_approval_batch_plan"
    assert batch["schema_version"] == 1
    assert batch["ok"] is True
    assert batch["collection_allowed"] is False
    assert batch["materialization_allowed"] is False
    assert batch["evaluation_allowed"] is False
    assert batch["state"] == {
        "collection_allowed": False,
        "sidecar_writes_allowed": False,
        "materialization_allowed": False,
        "evaluation_allowed": False,
        "candidate_acceptance_allowed": False,
        "writes_performed": False,
    }
    assert batch["summary"]["planned_sidecar_count"] == 45 * 3 * 128
    assert batch["summary"]["budget_count"] == 2
    assert batch["summary"]["candidate_eval_count"] == 2
    assert batch["summary"]["report_expected_row_count"] == 128
    assert batch["summary"]["selection_expected_row_count"] == 128
    assert batch["approval_request"]["ready"] is True
    assert "run_air_imatrix_calib_v1_activation_capture" in batch["approval_request"]["requested_operations"]
    assert "materialize_dynamic_mixed_precision_candidate_in_fresh_artifact_dir" in batch["approval_request"]["requested_operations"]
    assert "pick_smallest_budget_clearing_top1_and_kld" in batch["approval_request"]["requested_operations"]
    assert "real_activation_capture" in batch["blocked_operations"]
    assert "candidate_acceptance_or_ranking_claim" in batch["blocked_operations"]
    assert batch["plans"]["imatrix_collection"]["summary"]["planned_sidecar_count"] == 45 * 3 * 128
    assert batch["plans"]["materialization_sweep"]["summary"]["budget_count"] == 2
    assert batch["plans"]["materialized_eval"]["summary"]["candidate_count"] == 2
    assert not (tmp_path / "planned-imatrix").exists()
    assert not (tmp_path / "candidates").exists()

    blocked_collection = json.loads(json.dumps(collection_plan))
    blocked_collection["ok"] = False
    blocked_collection["errors"] = ["output_dir already exists"]
    blocked = build_dynamic_precision_approval_batch_plan(
        imatrix_collection_plan=blocked_collection,
        sweep_bundle=sweep,
        eval_plan=eval_plan,
    )

    assert blocked["ok"] is False
    assert blocked["approval_request"]["ready"] is False
    assert "imatrix_collection_plan must be ok before approval batch planning" in blocked["errors"]
    assert "output_dir already exists" in blocked["errors"]


def test_objective_audit_rejects_non_single_box_or_relaxed_dynamic_batch(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    reports = [
        build_dynamic_precision_tier_map_report(
            profiles,
            allocate_dynamic_precision_tiers(
                profiles,
                budget_bits_per_weight=budget,
                prior_floors=floors,
            ),
            prior_floors=floors,
            source=f"synthetic_budget_{budget}",
        )
        for budget in (3.3, 4.2)
    ]
    imatrix_dir = tmp_path / "imatrix"
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=imatrix_dir,
        prompt_set="air_imatrix_calib_v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)
    sweep = build_dynamic_precision_budget_sweep_approval_bundle(
        reports,
        candidate_name_prefix="synthetic-dynamic-sweep",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=tmp_path / "seed",
        output_root=tmp_path / "candidates",
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )
    eval_plan = build_dynamic_precision_materialized_eval_plan(
        sweep,
        report_cache_path=tmp_path / "report-cache.jsonl",
        selection_cache_path=tmp_path / "selection-cache.jsonl",
        lane_s_speed_gate={"scenario": "prefill_1k", "max_ratio_vs_q2": 1.5},
    )
    collection_plan = build_air_imatrix_collection_plan(
        output_dir=tmp_path / "planned-imatrix",
        layers=range(1, 46),
        projections=("gate_proj", "up_proj", "down_proj"),
        num_experts=128,
        collection_source="resident_vq_model",
    )
    batch = build_dynamic_precision_approval_batch_plan(
        imatrix_collection_plan=collection_plan,
        sweep_bundle=sweep,
        eval_plan=eval_plan,
    )

    audit = validate_dynamic_precision_approval_batch_objective(batch)

    assert json.loads(json.dumps(audit, sort_keys=True)) == audit
    assert audit["schema"] == "air_dynamic_precision_objective_audit"
    assert audit["schema_version"] == 1
    assert audit["ok"] is True
    assert audit["objective"]["single_m5_only"] is True
    assert audit["objective"]["implementation"] == "MLX-only"
    assert audit["requirements"]["collection_source"] == "resident_vq_model"
    assert audit["requirements"]["approved_collection_sources"] == [
        "resident_vq_model",
        "streamed_source_weights",
    ]
    assert audit["requirements"]["top1_agreement_min"] == 0.75
    assert audit["requirements"]["budget"] == {
        "min_effective_bits_per_weight": 2.0,
        "target_effective_bits_per_weight": 2.4,
        "max_effective_bits_per_weight": 3.0,
    }
    assert audit["requirements"]["primary_metric"] == "report.mean_kld"
    assert audit["requirements"]["rank_by"] == [
        "report.mean_kld",
        "report.p999_kld",
        "effective_bits_per_weight",
    ]
    assert audit["requirements"]["frontier_contract_ready"] is True
    assert audit["requirements"]["report_expected_row_count"] == 128
    assert audit["requirements"]["selection_expected_row_count"] == 128
    assert audit["state"]["all_execution_flags_closed"] is True

    corrupted = json.loads(json.dumps(batch))
    corrupted["plans"]["imatrix_collection"]["collection"]["collection_source"] = "distributed_bf16_teacher"
    corrupted["state"]["materialization_allowed"] = True
    corrupted["plans"]["materialized_eval"]["acceptance"]["top1_agreement_min"] = 0.7

    blocked = validate_dynamic_precision_approval_batch_objective(corrupted)

    assert blocked["ok"] is False
    assert "collection_source must stay single-box resident_vq_model or streamed_source_weights" in blocked["errors"]
    assert "batch.state.materialization_allowed must be false before approval" in blocked["errors"]
    assert "top1_agreement_min must remain 0.75" in blocked["errors"]


def test_human_approval_request_summarizes_audited_batch_without_opening_gate(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    reports = [
        build_dynamic_precision_tier_map_report(
            profiles,
            allocate_dynamic_precision_tiers(
                profiles,
                budget_bits_per_weight=budget,
                prior_floors=floors,
            ),
            prior_floors=floors,
            source=f"synthetic_budget_{budget}",
        )
        for budget in (3.3, 4.2)
    ]
    imatrix_dir = tmp_path / "imatrix"
    entries = accumulate_routed_projection_imatrix(
        layer=20,
        projection="gate_proj",
        inputs=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        route_indices=np.array([[7]], dtype=np.int64),
        num_experts=8,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )
    write_projection_imatrix_sidecars(
        entries,
        output_dir=imatrix_dir,
        prompt_set="air_imatrix_calib_v1",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensor = "model.layers.41.mlp.experts.11.up_proj.weight"
    source_file = source_dir / "model-00001-of-00001.safetensors"
    save_file({source_tensor: np.zeros((4, 3), dtype=np.float32)}, source_file)
    sweep = build_dynamic_precision_budget_sweep_approval_bundle(
        reports,
        candidate_name_prefix="synthetic-dynamic-sweep",
        imatrix_manifest=load_projection_imatrix_manifest(imatrix_dir / "imatrix-manifest.json"),
        source_index={
            "metadata": {"total_size": 48},
            "weight_map": {source_tensor: source_file.name},
        },
        imatrix_manifest_root=imatrix_dir,
        source_dir=source_dir,
        seed_artifact_dir=tmp_path / "seed",
        output_root=tmp_path / "candidates",
        source_kind="synthetic_source_index",
        require_imatrix_sidecar_files=True,
        require_source_files=True,
        inspect_source_headers=True,
    )
    eval_plan = build_dynamic_precision_materialized_eval_plan(
        sweep,
        report_cache_path=tmp_path / "report-cache.jsonl",
        selection_cache_path=tmp_path / "selection-cache.jsonl",
        lane_s_speed_gate={"scenario": "prefill_1k", "max_ratio_vs_q2": 1.5},
    )
    collection_plan = build_air_imatrix_collection_plan(
        output_dir=tmp_path / "planned-imatrix",
        layers=range(1, 46),
        projections=("gate_proj", "up_proj", "down_proj"),
        num_experts=128,
        collection_source="streamed_source_weights",
    )
    batch = build_dynamic_precision_approval_batch_plan(
        imatrix_collection_plan=collection_plan,
        sweep_bundle=sweep,
        eval_plan=eval_plan,
    )
    audit = validate_dynamic_precision_approval_batch_objective(batch)

    request = build_dynamic_precision_human_approval_request(
        approval_batch=batch,
        objective_audit=audit,
    )

    assert json.loads(json.dumps(request, sort_keys=True)) == request
    assert request["schema"] == "air_dynamic_precision_human_approval_request"
    assert request["schema_version"] == 1
    assert request["ready_for_human_approval"] is True
    assert request["approval_required"] is True
    assert request["approval_gate"] == "real_imatrix_sidecar_collection_and_candidate_materialization"
    assert request["execution_flags"] == {
        "collection_allowed": False,
        "materialization_allowed": False,
        "evaluation_allowed": False,
        "candidate_acceptance_allowed": False,
    }
    assert request["single_m5_only"] is True
    assert request["implementation"] == "MLX-only"
    assert request["requested_scope"]["candidate_count"] == 2
    assert request["requested_scope"]["collection_source"] == "streamed_source_weights"
    assert request["requested_scope"]["top1_agreement_min"] == 0.75
    assert request["requested_scope"]["rank_by"] == [
        "report.mean_kld",
        "report.p999_kld",
        "effective_bits_per_weight",
    ]
    assert "collect_real_air_imatrix_calib_v1_imatrix_sidecars" in request["requested_operations"]
    assert "pick_smallest_budget_clearing_top1_and_kld" in request["requested_operations"]
    assert "candidate_acceptance_or_ranking_claim" in request["forbidden_without_approval"]
    assert "top-1 >= 0.75" in request["approval_text"]
    assert "report mean KLD" in request["approval_text"]
    assert "single-M5 MLX-only" in request["approval_text"]
    assert not (tmp_path / "planned-imatrix").exists()
    assert not (tmp_path / "candidates").exists()

    blocked_audit = json.loads(json.dumps(audit))
    blocked_audit["ok"] = False
    blocked_audit["errors"] = ["top1_agreement_min must remain 0.75"]
    blocked = build_dynamic_precision_human_approval_request(
        approval_batch=batch,
        objective_audit=blocked_audit,
    )

    assert blocked["ready_for_human_approval"] is False
    assert blocked["requested_operations"] == []
    assert "objective_audit must be ok before human approval request" in blocked["errors"]
    assert "top1_agreement_min must remain 0.75" in blocked["errors"]


def test_human_approval_markdown_is_paste_ready_without_granting_approval() -> None:
    request = {
        "schema": "air_dynamic_precision_human_approval_request",
        "schema_version": 1,
        "ready_for_human_approval": True,
        "errors": [],
        "approval_required": True,
        "approval_gate": "real_imatrix_sidecar_collection_and_candidate_materialization",
        "single_m5_only": True,
        "implementation": "MLX-only",
        "execution_flags": {
            "collection_allowed": False,
            "materialization_allowed": False,
            "evaluation_allowed": False,
            "candidate_acceptance_allowed": False,
        },
        "requested_scope": {
            "candidate_count": 3,
            "collection_source": "streamed_source_weights",
            "report_expected_row_count": 128,
            "selection_expected_row_count": 128,
            "top1_agreement_min": 0.75,
            "budget": {
                "min_effective_bits_per_weight": 2.0,
                "target_effective_bits_per_weight": 2.4,
                "max_effective_bits_per_weight": 3.0,
            },
            "primary_metric": "report.mean_kld",
            "rank_by": [
                "report.mean_kld",
                "report.p999_kld",
                "effective_bits_per_weight",
            ],
        },
        "requested_operations": [
            "collect_real_air_imatrix_calib_v1_imatrix_sidecars",
            "materialize_dynamic_mixed_precision_candidate_in_fresh_artifact_dir",
            "pick_smallest_budget_clearing_top1_and_kld",
        ],
        "forbidden_without_approval": [
            "candidate_acceptance_or_ranking_claim",
            "budget_frontier_result_writes",
        ],
        "approval_text": (
            "Approve the GLM-4.5-Air dynamic mixed precision real-data batch: "
            "single-M5 MLX-only imatrix sidecar collection, high-source metadata validation, "
            "fresh candidate materialization, 128-row report/selection eval, and KLD frontier "
            "selection with top-1 >= 0.75 and report mean KLD as primary."
        ),
        "source_schemas": {
            "approval_batch": "air_dynamic_precision_approval_batch_plan",
            "objective_audit": "air_dynamic_precision_objective_audit",
        },
    }

    renderer = getattr(dynamic_precision_module, "build_dynamic_precision_human_approval_markdown", None)
    assert renderer is not None

    summary = renderer(request)

    assert json.loads(json.dumps(summary, sort_keys=True)) == summary
    assert summary["schema"] == "air_dynamic_precision_human_approval_markdown"
    assert summary["schema_version"] == 1
    assert summary["ok"] is True
    assert summary["ready_to_present"] is True
    assert summary["approval_required"] is True
    assert summary["approval_granted"] is False
    assert summary["approval_gate"] == "real_imatrix_sidecar_collection_and_candidate_materialization"
    assert summary["requested_operations"] == request["requested_operations"]
    assert "candidate_acceptance_or_ranking_claim" in summary["forbidden_without_approval"]
    markdown = summary["markdown"]
    assert markdown.startswith("# GLM-4.5-Air Dynamic Mixed Precision Approval Request")
    assert "Scope: single-M5 MLX-only" in markdown
    assert "Budget: 2.0 -> 2.4 -> 3.0 effective bits/weight" in markdown
    assert "Acceptance: top-1 >= 0.75" in markdown
    assert "Rank by: report.mean_kld, report.p999_kld, effective_bits_per_weight" in markdown
    assert "- collection_allowed=false" in markdown
    assert "- materialization_allowed=false" in markdown
    assert "- evaluation_allowed=false" in markdown
    assert "- candidate_acceptance_allowed=false" in markdown
    assert "Approve the GLM-4.5-Air dynamic mixed precision real-data batch" in markdown

    blocked = renderer(
        {**request, "ready_for_human_approval": False, "errors": ["objective audit failed"]}
    )

    assert blocked["ok"] is False
    assert blocked["ready_to_present"] is False
    assert blocked["markdown"] == ""
    assert blocked["requested_operations"] == []
    assert "approval_request must be ready before rendering approval markdown" in blocked["errors"]
    assert "objective audit failed" in blocked["errors"]


def test_human_approval_response_audit_requires_exact_text_without_writes() -> None:
    request = {
        "schema": "air_dynamic_precision_human_approval_request",
        "schema_version": 1,
        "ready_for_human_approval": True,
        "errors": [],
        "approval_required": True,
        "approval_gate": "real_imatrix_sidecar_collection_and_candidate_materialization",
        "single_m5_only": True,
        "implementation": "MLX-only",
        "execution_flags": {
            "collection_allowed": False,
            "materialization_allowed": False,
            "evaluation_allowed": False,
            "candidate_acceptance_allowed": False,
        },
        "requested_scope": {
            "candidate_count": 3,
            "collection_source": "streamed_source_weights",
            "report_expected_row_count": 128,
            "selection_expected_row_count": 128,
            "top1_agreement_min": 0.75,
            "budget": {
                "min_effective_bits_per_weight": 2.0,
                "target_effective_bits_per_weight": 2.4,
                "max_effective_bits_per_weight": 3.0,
            },
            "primary_metric": "report.mean_kld",
            "rank_by": [
                "report.mean_kld",
                "report.p999_kld",
                "effective_bits_per_weight",
            ],
        },
        "requested_operations": [
            "collect_real_air_imatrix_calib_v1_imatrix_sidecars",
            "materialize_dynamic_mixed_precision_candidate_in_fresh_artifact_dir",
            "pick_smallest_budget_clearing_top1_and_kld",
        ],
        "forbidden_without_approval": [
            "candidate_acceptance_or_ranking_claim",
            "budget_frontier_result_writes",
        ],
        "approval_text": (
            "Approve the GLM-4.5-Air dynamic mixed precision real-data batch: "
            "single-M5 MLX-only imatrix sidecar collection, high-source metadata validation, "
            "fresh candidate materialization, 128-row report/selection eval, and KLD frontier "
            "selection with top-1 >= 0.75 and report mean KLD as primary."
        ),
        "source_schemas": {
            "approval_batch": "air_dynamic_precision_approval_batch_plan",
            "objective_audit": "air_dynamic_precision_objective_audit",
        },
    }
    renderer = getattr(dynamic_precision_module, "build_dynamic_precision_human_approval_markdown", None)
    assert renderer is not None
    markdown = renderer(request)
    auditor = getattr(dynamic_precision_module, "validate_dynamic_precision_human_approval_response", None)
    assert auditor is not None

    audit = auditor(approval_markdown=markdown, response_text=request["approval_text"])

    assert json.loads(json.dumps(audit, sort_keys=True)) == audit
    assert audit["schema"] == "air_dynamic_precision_human_approval_response_audit"
    assert audit["schema_version"] == 1
    assert audit["ok"] is True
    assert audit["approval_required"] is True
    assert audit["approval_gate"] == "real_imatrix_sidecar_collection_and_candidate_materialization"
    assert audit["approval_phrase_matches"] is True
    assert audit["ready_for_operator_to_start_approved_batch"] is True
    assert audit["requested_operations"] == request["requested_operations"]
    assert "candidate_acceptance_or_ranking_claim" in audit["forbidden_without_approval"]
    assert audit["execution_flags_after_audit"] == {
        "collection_allowed": False,
        "materialization_allowed": False,
        "evaluation_allowed": False,
        "candidate_acceptance_allowed": False,
    }
    assert audit["filesystem"] == {
        "writes_performed": False,
        "artifact_created": False,
        "candidate_dir_created": False,
        "result_artifact_written": False,
    }

    near_miss = request["approval_text"].replace("top-1 >= 0.75", "top-1 >= 0.70")
    blocked = auditor(approval_markdown=markdown, response_text=near_miss)

    assert blocked["ok"] is False
    assert blocked["approval_phrase_matches"] is False
    assert blocked["ready_for_operator_to_start_approved_batch"] is False
    assert blocked["requested_operations"] == []
    assert "approval response must exactly match approval_text" in blocked["errors"]


def test_high_precision_source_manifest_joiner_blocks_missing_high_tier_sources(tmp_path) -> None:
    profiles = [
        _profile(
            "20:gate_proj",
            layer=20,
            projection="gate_proj",
            expert=7,
            weight_count=100,
            low_error=50.0,
            mid_error=10.0,
            high_error=8.0,
        ),
        _profile(
            "41:up_proj",
            layer=41,
            projection="up_proj",
            expert=11,
            weight_count=10,
            low_error=80.0,
            mid_error=40.0,
            high_error=5.0,
        ),
    ]
    floors = build_air_layer_type_prior_floors(profiles, total_layers=46)
    plan = allocate_dynamic_precision_tiers(
        profiles,
        budget_bits_per_weight=4.2,
        prior_floors=floors,
    )
    report = build_dynamic_precision_tier_map_report(
        profiles,
        plan,
        prior_floors=floors,
        source="synthetic_unit_test",
    )
    materialization_plan = build_dynamic_precision_materialization_plan(
        report,
        candidate_name="synthetic-dynamic-v1",
    )
    source_file = tmp_path / "source" / "model-00001-of-00001.safetensors"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"synthetic high precision source placeholder")
    manifest = {
        "schema_version": 1,
        "record_type": "air_dynamic_precision_high_source_manifest",
        "source_kind": "synthetic_high_precision_source",
        "entry_count": 1,
        "entries": [
            {
                "layer": 41,
                "projection": "up_proj",
                "expert": 11,
                "path": "model-00001-of-00001.safetensors",
                "tensor_name": "model.layers.41.mlp.experts.11.up_proj.weight",
                "dtype": "bf16",
                "shape": [4, 3],
            }
        ],
    }

    joined = attach_high_precision_source_manifest_to_materialization_plan(
        materialization_plan,
        manifest,
        source_root=source_file.parent,
        require_source_files=True,
    )

    assert json.loads(json.dumps(joined, sort_keys=True)) == joined
    assert joined["ok"] is True
    assert joined["high_precision_source_manifest"] == {
        "record_type": "air_dynamic_precision_high_source_manifest",
        "source_kind": "synthetic_high_precision_source",
        "entry_count": 1,
        "covered_action_count": 1,
        "missing_action_count": 0,
        "source_files_checked": True,
    }
    actions = {row["key"]: row for row in joined["actions"]}
    assert actions["20:gate_proj"]["high_precision_source_available"] is False
    assert actions["20:gate_proj"]["high_precision_source"] is None
    assert actions["41:up_proj"]["high_precision_source_available"] is True
    assert actions["41:up_proj"]["high_precision_source"] == {
        "path": "model-00001-of-00001.safetensors",
        "tensor_name": "model.layers.41.mlp.experts.11.up_proj.weight",
        "dtype": "bf16",
        "shape": [4, 3],
        "source_kind": "synthetic_high_precision_source",
    }

    missing = attach_high_precision_source_manifest_to_materialization_plan(
        materialization_plan,
        {**manifest, "entry_count": 0, "entries": []},
        source_root=source_file.parent,
        require_source_files=True,
    )

    assert missing["ok"] is False
    assert "missing high-precision source for 41:up_proj layer=41 projection=up_proj expert=11" in missing["errors"]
    assert missing["high_precision_source_manifest"]["missing_action_count"] == 1


def test_imatrix_weighted_squared_error_weights_input_columns() -> None:
    reference = np.array([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]], dtype=np.float32)
    candidate = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 3.0]], dtype=np.float32)
    importance = np.array([1.0, 0.5, 2.0], dtype=np.float32)

    error = imatrix_weighted_squared_error(reference, candidate, importance)

    assert error == 1.0 + 0.5 + 8.0 + 1.0 + 0.0 + 8.0
