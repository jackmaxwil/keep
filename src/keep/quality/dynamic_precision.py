from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


AIR_DYNAMIC_TIER_ORDER = ("low", "mid", "high")
AIR_FRAGILE_GLU_LAYERS = (31, 36, 41)
AIR_ROUTED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE = "real_imatrix_sidecar_collection_and_candidate_materialization"
AIR_DYNAMIC_MATERIALIZATION_APPROVAL_OPERATIONS = (
    "collect_real_air_imatrix_calib_v1_imatrix_sidecars",
    "build_real_high_precision_source_manifest_from_source_index",
    "run_real_metadata_validation_join_and_dry_run",
    "materialize_dynamic_mixed_precision_candidate_in_fresh_artifact_dir",
    "evaluate_report_and_selection_caches_and_budget_sweep",
)
AIR_DYNAMIC_MATERIALIZATION_BLOCKED_OPERATIONS = (
    "real_imatrix_sidecar_collection",
    "real_high_precision_source_tensor_reads",
    "persisted_tier_map_or_report_artifact_writes",
    "high_tier_shard_generation",
    "candidate_directory_creation",
    "report_selection_cache_evaluation_and_budget_sweep",
)
AIR_DYNAMIC_REPORT_PROMPT_SET = "air_vq_ladder_report_v1"
AIR_DYNAMIC_SELECTION_PROMPT_SET = "air_vq_ladder_select_v1"
AIR_DYNAMIC_EVAL_REQUIRED_METRICS = (
    "mean_kld",
    "p999_kld",
    "mean_ppl_ratio",
    "max_ppl_ratio",
    "mean_top1_agreement",
    "clean_row_count",
    "effective_bits_per_weight",
    "lane_s_speed",
)
AIR_DYNAMIC_EVAL_APPROVAL_OPERATIONS = (
    "evaluate_materialized_candidates_on_128_row_report_cache",
    "evaluate_materialized_candidates_on_128_row_selection_cache",
    "record_kld_vs_size_budget_frontier",
    "pick_smallest_budget_clearing_top1_and_kld",
)
AIR_DYNAMIC_EVAL_BLOCKED_OPERATIONS = (
    "materialized_report_cache_evaluation",
    "materialized_selection_cache_evaluation",
    "budget_frontier_result_writes",
    "candidate_acceptance_or_ranking_claim",
)
AIR_DYNAMIC_APPROVED_IMATRIX_COLLECTION_SOURCES = (
    "resident_vq_model",
    "streamed_source_weights",
)
AIR_FIXED_HIGH_PRECISION_PRIOR_RULES = (
    {
        "name": "token_embeddings",
        "tensor_patterns": ("model.embed_tokens.weight",),
        "rationale": (
            "llama_cpp_token_embedding_high_precision_prior",
            "not_part_of_routed_expert_vq_artifact",
        ),
    },
    {
        "name": "output_lm_head",
        "tensor_patterns": ("lm_head.weight",),
        "rationale": (
            "llama_cpp_output_or_tied_embedding_high_precision_prior",
            "not_part_of_routed_expert_vq_artifact",
        ),
    },
    {
        "name": "attention_value_output",
        "tensor_patterns": (
            "model.layers.*.self_attn.v_proj.weight",
            "model.layers.*.self_attn.o_proj.weight",
        ),
        "rationale": (
            "llama_cpp_attention_value_output_high_precision_prior",
            "dense_non_expert_source_path",
        ),
    },
    {
        "name": "shared_or_always_on_expert",
        "tensor_patterns": (
            "model.layers.*.mlp.shared_expert.*",
            "model.layers.*.mlp.shared_experts.*",
        ),
        "rationale": (
            "llama_cpp_moe_shared_expert_high_precision_prior",
            "always_on_expert_is_not_routed_sparse_vq_group",
        ),
    },
    {
        "name": "normalization_and_router",
        "tensor_patterns": (
            "model.layers.*.input_layernorm.weight",
            "model.layers.*.post_attention_layernorm.weight",
            "model.norm.weight",
            "model.layers.*.mlp.gate.weight",
            "model.layers.*.mlp.router.*",
        ),
        "rationale": (
            "control_tensors_stay_dense",
            "not_candidate_for_imatrix_vq_reencoding",
        ),
    },
)


@dataclass(frozen=True)
class DynamicPrecisionTier:
    name: str
    artifact_kind: str
    effective_bits_per_weight: float
    code_bits: int | None
    description: str


@dataclass(frozen=True)
class DynamicTensorProfile:
    key: str
    layer: int
    projection: str
    weight_count: int
    weighted_errors: Mapping[str, float]
    expert: int | None = None


@dataclass(frozen=True)
class DynamicPrecisionPriorFloor:
    key: str
    minimum_tier: str
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class DynamicPrecisionPlan:
    tier_by_key: dict[str, str]
    total_effective_bits: float
    budget_effective_bits: float
    effective_bits_per_weight: float
    weighted_error: float
    upgrades: tuple[dict[str, object], ...]


def default_air_dynamic_precision_tiers() -> dict[str, DynamicPrecisionTier]:
    """Return the GLM-4.5-Air tier menu for dynamic allocation planning."""

    return {
        "low": DynamicPrecisionTier(
            name="low",
            artifact_kind="vq_e8",
            effective_bits_per_weight=2.0,
            code_bits=8,
            description="Bulk routed-expert E8 VQ tier; roughly the low/2-bit allocation bucket.",
        ),
        "mid": DynamicPrecisionTier(
            name="mid",
            artifact_kind="vq_e8p",
            effective_bits_per_weight=3.0,
            code_bits=16,
            description="E8P VQ tier for tensors that need more codebook resolution.",
        ),
        "high": DynamicPrecisionTier(
            name="high",
            artifact_kind="bf16",
            effective_bits_per_weight=16.0,
            code_bits=None,
            description="High-precision escape tier for sensitive tensors; materialization is gated.",
        ),
    }


def build_air_layer_type_prior_floors(
    profiles: Iterable[DynamicTensorProfile],
    *,
    total_layers: int,
    edge_layers: int = 2,
    fragile_glu_layers: Iterable[int] = AIR_FRAGILE_GLU_LAYERS,
) -> tuple[DynamicPrecisionPriorFloor, ...]:
    """Seed llama.cpp-style precision floors for routed GLM-4.5-Air tensors."""

    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if edge_layers < 0:
        raise ValueError("edge_layers must be non-negative")

    fragile_layers = {int(layer) for layer in fragile_glu_layers}
    floors: list[DynamicPrecisionPriorFloor] = []
    for profile in sorted(profiles, key=lambda item: item.key):
        _validate_profile_shape(profile)
        minimum_tier = "low"
        rationale: list[str] = []
        if profile.projection == "down_proj":
            minimum_tier = _max_tier(minimum_tier, "mid")
            rationale.append("ffn_down_proj")
        if edge_layers and (
            profile.layer <= edge_layers
            or profile.layer >= max(0, total_layers - edge_layers)
        ):
            minimum_tier = _max_tier(minimum_tier, "mid")
            rationale.append("first_last_layers")
        if profile.layer in fragile_layers and profile.projection in ("gate_proj", "up_proj"):
            minimum_tier = _max_tier(minimum_tier, "high")
            rationale.append("fragile_glu_layer")
        if not rationale:
            rationale.append("default_low_tier")
        floors.append(
            DynamicPrecisionPriorFloor(
                key=profile.key,
                minimum_tier=minimum_tier,
                rationale=tuple(rationale),
            )
        )
    return tuple(floors)


def build_air_dynamic_precision_prior_summary(
    profiles: Iterable[DynamicTensorProfile],
    *,
    total_layers: int,
    edge_layers: int = 2,
    fragile_glu_layers: Iterable[int] = AIR_FRAGILE_GLU_LAYERS,
) -> dict[str, Any]:
    """Summarize routed allocator floors plus fixed high-precision Air surfaces."""

    floors = build_air_layer_type_prior_floors(
        profiles,
        total_layers=total_layers,
        edge_layers=edge_layers,
        fragile_glu_layers=fragile_glu_layers,
    )
    return {
        "schema": "air_dynamic_precision_prior_summary",
        "schema_version": 1,
        "ok": True,
        "materialization_allowed": False,
        "artifact_written": False,
        "allocation_scope": {
            "profile_scope": "routed_expert_gate_up_down",
            "allocator_consumes": "routed_prior_floors",
            "fixed_high_precision_action": "keep_on_existing_dense_source_path",
            "total_layers": int(total_layers),
            "edge_layers": int(edge_layers),
        },
        "routed_prior_floors": [
            {
                "key": floor.key,
                "minimum_tier": floor.minimum_tier,
                "rationale": list(floor.rationale),
            }
            for floor in floors
        ],
        "fixed_high_precision_priors": [
            {
                "name": str(rule["name"]),
                "minimum_tier": "high",
                "candidate_materialization_action": "none_fixed_source_precision",
                "tensor_patterns": [str(pattern) for pattern in rule["tensor_patterns"]],
                "rationale": [str(reason) for reason in rule["rationale"]],
            }
            for rule in AIR_FIXED_HIGH_PRECISION_PRIOR_RULES
        ],
        "notes": [
            "Routed expert gate/up/down tensors are the current dynamic allocation scope.",
            "Dense non-expert Air tensors stay on the existing source/dense path and are recorded as fixed high precision.",
        ],
    }


def imatrix_weighted_squared_error(
    reference: np.ndarray,
    candidate: np.ndarray,
    importance: np.ndarray,
) -> float:
    """Compute sum((reference - candidate)^2 * input_column_importance)."""

    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    importance_array = np.asarray(importance, dtype=np.float64)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            f"reference and candidate shapes must match, found {reference_array.shape} and {candidate_array.shape}"
        )
    if reference_array.ndim < 1:
        raise ValueError("reference and candidate must have at least one dimension")
    if importance_array.shape != (reference_array.shape[-1],):
        raise ValueError(
            f"importance must have shape ({reference_array.shape[-1]},), found {importance_array.shape}"
        )
    if not np.isfinite(reference_array).all() or not np.isfinite(candidate_array).all():
        raise ValueError("reference and candidate must be finite")
    if not np.isfinite(importance_array).all() or np.any(importance_array < 0):
        raise ValueError("importance must be finite and non-negative")
    squared = (reference_array - candidate_array) * (reference_array - candidate_array)
    return float(np.sum(squared * importance_array))


def allocate_dynamic_precision_tiers(
    profiles: Iterable[DynamicTensorProfile],
    *,
    budget_bits_per_weight: float,
    prior_floors: Iterable[DynamicPrecisionPriorFloor] = (),
    tiers: Mapping[str, DynamicPrecisionTier] | None = None,
) -> DynamicPrecisionPlan:
    """Greedily allocate tiers by imatrix-weighted error reduction per added bit."""

    tier_map = dict(tiers or default_air_dynamic_precision_tiers())
    _validate_tiers(tier_map)
    profile_list = sorted(list(profiles), key=lambda item: item.key)
    if not profile_list:
        raise ValueError("at least one tensor profile is required")
    if budget_bits_per_weight <= 0:
        raise ValueError("budget_bits_per_weight must be positive")

    by_key: dict[str, DynamicTensorProfile] = {}
    for profile in profile_list:
        _validate_profile(profile, tier_map)
        if profile.key in by_key:
            raise ValueError(f"duplicate tensor profile key {profile.key!r}")
        by_key[profile.key] = profile

    tier_by_key = {profile.key: AIR_DYNAMIC_TIER_ORDER[0] for profile in profile_list}
    floor_by_key = {floor.key: floor for floor in prior_floors}
    unknown_floors = sorted(set(floor_by_key) - set(by_key))
    if unknown_floors:
        raise ValueError(f"prior floors reference unknown tensor keys: {unknown_floors}")

    upgrades: list[dict[str, object]] = []
    for key in sorted(floor_by_key):
        floor = floor_by_key[key]
        _validate_tier_name(floor.minimum_tier, tier_map)
        current_tier = tier_by_key[key]
        if _tier_index(floor.minimum_tier) > _tier_index(current_tier):
            tier_by_key[key] = floor.minimum_tier
            upgrades.append(
                {
                    "key": key,
                    "from": current_tier,
                    "to": floor.minimum_tier,
                    "reason": "prior_floor",
                    "rationale": list(floor.rationale),
                }
            )

    total_weights = sum(profile.weight_count for profile in profile_list)
    budget_bits = float(budget_bits_per_weight * total_weights)
    total_bits = _total_effective_bits(profile_list, tier_by_key, tier_map)
    if total_bits > budget_bits + 1.0e-9:
        raise ValueError(
            f"prior floors require {total_bits:.6f} effective bits, exceeding budget {budget_bits:.6f}"
        )

    while True:
        candidate = _best_upgrade(profile_list, tier_by_key, tier_map, total_bits, budget_bits)
        if candidate is None:
            break
        profile, next_tier, added_bits, gain, score = candidate
        old_tier = tier_by_key[profile.key]
        tier_by_key[profile.key] = next_tier
        total_bits += added_bits
        upgrades.append(
            {
                "key": profile.key,
                "from": old_tier,
                "to": next_tier,
                "reason": "greedy_gain_per_bit",
                "weighted_error_reduction": gain,
                "added_effective_bits": added_bits,
                "gain_per_added_bit": score,
            }
        )

    weighted_error = sum(
        float(profile.weighted_errors[tier_by_key[profile.key]])
        for profile in profile_list
    )
    return DynamicPrecisionPlan(
        tier_by_key=dict(sorted(tier_by_key.items())),
        total_effective_bits=total_bits,
        budget_effective_bits=budget_bits,
        effective_bits_per_weight=total_bits / total_weights,
        weighted_error=weighted_error,
        upgrades=tuple(upgrades),
    )


def build_dynamic_precision_tier_map_report(
    profiles: Iterable[DynamicTensorProfile],
    plan: DynamicPrecisionPlan,
    *,
    prior_floors: Iterable[DynamicPrecisionPriorFloor] = (),
    tiers: Mapping[str, DynamicPrecisionTier] | None = None,
    source: str = "synthetic_or_approved_inputs",
) -> dict[str, Any]:
    """Build a deterministic JSON-ready tier-map report without writing artifacts."""

    tier_map = dict(tiers or default_air_dynamic_precision_tiers())
    _validate_tiers(tier_map)
    profile_list = sorted(list(profiles), key=lambda item: item.key)
    if not profile_list:
        raise ValueError("at least one tensor profile is required")

    profile_by_key: dict[str, DynamicTensorProfile] = {}
    for profile in profile_list:
        _validate_profile(profile, tier_map)
        if profile.key in profile_by_key:
            raise ValueError(f"duplicate tensor profile key {profile.key!r}")
        profile_by_key[profile.key] = profile

    if set(plan.tier_by_key) != set(profile_by_key):
        missing = sorted(set(profile_by_key) - set(plan.tier_by_key))
        extra = sorted(set(plan.tier_by_key) - set(profile_by_key))
        raise ValueError(f"plan/profile key mismatch; missing={missing}, extra={extra}")
    for key, tier_name in plan.tier_by_key.items():
        _validate_tier_name(tier_name, tier_map)

    floor_by_key = {floor.key: floor for floor in prior_floors}
    unknown_floors = sorted(set(floor_by_key) - set(profile_by_key))
    if unknown_floors:
        raise ValueError(f"prior floors reference unknown tensor keys: {unknown_floors}")
    for floor in floor_by_key.values():
        _validate_tier_name(floor.minimum_tier, tier_map)

    total_weights = sum(profile.weight_count for profile in profile_list)
    tier_counts = {
        tier_name: sum(1 for selected in plan.tier_by_key.values() if selected == tier_name)
        for tier_name in AIR_DYNAMIC_TIER_ORDER
    }
    return {
        "schema": "air_dynamic_precision_tier_map_report",
        "schema_version": 1,
        "source": str(source),
        "materialization": {
            "candidate_materialized": False,
            "artifact_written": False,
            "approval_required": True,
            "approval_gate": "real_imatrix_sidecar_collection_and_candidate_materialization",
        },
        "budget": {
            "total_weight_count": int(total_weights),
            "total_effective_bits": float(plan.total_effective_bits),
            "budget_effective_bits": float(plan.budget_effective_bits),
            "effective_bits_per_weight": float(plan.effective_bits_per_weight),
            "weighted_error": float(plan.weighted_error),
        },
        "tiers": {
            tier_name: _tier_to_report_row(tier_map[tier_name])
            for tier_name in AIR_DYNAMIC_TIER_ORDER
        },
        "tier_counts": tier_counts,
        "tier_map": dict(sorted(plan.tier_by_key.items())),
        "upgrades": [_json_ready_upgrade(upgrade) for upgrade in plan.upgrades],
        "tensors": [
            _profile_to_report_row(
                profile,
                selected_tier=plan.tier_by_key[profile.key],
                tier=tier_map[plan.tier_by_key[profile.key]],
                prior_floor=floor_by_key.get(profile.key),
            )
            for profile in profile_list
        ],
    }


def validate_dynamic_precision_tier_map_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an in-memory dynamic tier-map report before any materialization."""

    errors: list[str] = []
    if not isinstance(report, Mapping):
        return {
            "ok": False,
            "errors": ["report must be a mapping"],
            "tensor_count": 0,
        }

    if report.get("schema") != "air_dynamic_precision_tier_map_report":
        errors.append("schema must be air_dynamic_precision_tier_map_report")
    if report.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    materialization = _mapping_value(report, "materialization", errors)
    if materialization:
        if materialization.get("candidate_materialized") is not False:
            errors.append("materialization.candidate_materialized must be false")
        if materialization.get("artifact_written") is not False:
            errors.append("materialization.artifact_written must be false")
        if materialization.get("approval_required") is not True:
            errors.append("materialization.approval_required must be true")
        if materialization.get("approval_gate") != "real_imatrix_sidecar_collection_and_candidate_materialization":
            errors.append("materialization.approval_gate mismatch")

    tiers = _mapping_value(report, "tiers", errors)
    tier_bits: dict[str, float] = {}
    if tiers:
        if tuple(tiers) != AIR_DYNAMIC_TIER_ORDER:
            errors.append(f"tiers must be ordered as {AIR_DYNAMIC_TIER_ORDER}")
        previous_bits = -1.0
        for tier_name in AIR_DYNAMIC_TIER_ORDER:
            tier = tiers.get(tier_name)
            if not isinstance(tier, Mapping):
                errors.append(f"tiers.{tier_name} must be a mapping")
                continue
            if tier.get("name") != tier_name:
                errors.append(f"tiers.{tier_name}.name mismatch")
            bits = _finite_number(tier.get("effective_bits_per_weight"))
            if bits is None:
                errors.append(f"tiers.{tier_name}.effective_bits_per_weight must be finite")
                continue
            if bits <= previous_bits:
                errors.append("tier effective_bits_per_weight must increase monotonically")
            previous_bits = bits
            tier_bits[tier_name] = bits

    tier_map = _mapping_value(report, "tier_map", errors)
    tensors_value = report.get("tensors")
    tensors = tensors_value if isinstance(tensors_value, list) else []
    if not isinstance(tensors_value, list):
        errors.append("tensors must be a list")

    tensor_keys: set[str] = set()
    computed_tier_counts = {tier_name: 0 for tier_name in AIR_DYNAMIC_TIER_ORDER}
    total_weight_count = 0
    total_effective_bits = 0.0
    weighted_error = 0.0
    for index, row_value in enumerate(tensors):
        if not isinstance(row_value, Mapping):
            errors.append(f"tensors[{index}] must be a mapping")
            continue
        key = row_value.get("key")
        if not isinstance(key, str) or not key:
            errors.append(f"tensors[{index}].key must be non-empty")
            continue
        if key in tensor_keys:
            errors.append(f"duplicate tensor key {key}")
        tensor_keys.add(key)

        selected_tier = row_value.get("selected_tier")
        mapped_tier = tier_map.get(key) if tier_map else None
        if selected_tier not in AIR_DYNAMIC_TIER_ORDER:
            errors.append(f"{key}: selected tier {selected_tier!r} is unknown")
            continue
        if mapped_tier != selected_tier:
            errors.append(f"{key}: tier_map {mapped_tier!r} does not match selected tier {selected_tier!r}")
        computed_tier_counts[str(selected_tier)] += 1

        prior_floor = row_value.get("prior_floor")
        if not isinstance(prior_floor, Mapping):
            errors.append(f"{key}: prior_floor must be a mapping")
        else:
            minimum_tier = prior_floor.get("minimum_tier")
            if minimum_tier not in AIR_DYNAMIC_TIER_ORDER:
                errors.append(f"{key}: prior floor {minimum_tier!r} is unknown")
            elif _tier_index(str(selected_tier)) < _tier_index(str(minimum_tier)):
                errors.append(f"{key}: selected tier {selected_tier} is below prior floor {minimum_tier}")

        weight_count = _positive_int(row_value.get("weight_count"))
        if weight_count is None:
            errors.append(f"{key}: weight_count must be positive")
            continue
        selected_bits = tier_bits.get(str(selected_tier))
        selected_error = _finite_number(row_value.get("selected_weighted_error"))
        if selected_error is None or selected_error < 0.0:
            errors.append(f"{key}: selected_weighted_error must be finite and non-negative")
            selected_error = 0.0
        total_weight_count += weight_count
        if selected_bits is not None:
            total_effective_bits += weight_count * selected_bits
        weighted_error += selected_error

    if tier_map and set(tier_map) != tensor_keys:
        missing = sorted(tensor_keys - set(tier_map))
        extra = sorted(set(tier_map) - tensor_keys)
        errors.append(f"tier_map/tensor key mismatch; missing={missing}, extra={extra}")

    tier_counts = report.get("tier_counts")
    if tier_counts != computed_tier_counts:
        errors.append(f"tier_counts mismatch; expected {computed_tier_counts}")

    budget = _mapping_value(report, "budget", errors)
    if budget:
        _expect_close(
            budget.get("total_weight_count"),
            total_weight_count,
            errors,
            "budget.total_weight_count",
        )
        _expect_close(
            budget.get("total_effective_bits"),
            total_effective_bits,
            errors,
            "budget.total_effective_bits",
        )
        _expect_close(
            budget.get("weighted_error"),
            weighted_error,
            errors,
            "budget.weighted_error",
        )
        effective_bpw = total_effective_bits / total_weight_count if total_weight_count else 0.0
        _expect_close(
            budget.get("effective_bits_per_weight"),
            effective_bpw,
            errors,
            "budget.effective_bits_per_weight",
        )
        budget_bits = _finite_number(budget.get("budget_effective_bits"))
        if budget_bits is None or budget_bits <= 0.0:
            errors.append("budget.budget_effective_bits must be positive")
        elif total_effective_bits > budget_bits + 1.0e-6:
            errors.append("budget.total_effective_bits exceeds budget_effective_bits")

    return {
        "ok": not errors,
        "errors": errors,
        "tensor_count": len(tensor_keys),
    }


def build_dynamic_precision_materialization_plan(
    report: Mapping[str, Any],
    *,
    candidate_name: str,
) -> dict[str, Any]:
    """Translate a validated tier-map report into an in-memory materializer plan."""

    candidate = _candidate_gate(candidate_name)
    validation = validate_dynamic_precision_tier_map_report(report)
    if not validation["ok"]:
        return {
            "schema": "air_dynamic_precision_materialization_plan",
            "schema_version": 1,
            "ok": False,
            "errors": list(validation["errors"]),
            "candidate": candidate,
            "actions": [],
        }

    tensors = list(report["tensors"])
    actions = [_materialization_action(row, report["materialization"]) for row in tensors]
    budget = report["budget"]
    return {
        "schema": "air_dynamic_precision_materialization_plan",
        "schema_version": 1,
        "ok": True,
        "errors": [],
        "candidate": candidate,
        "source_report": {
            "schema": report["schema"],
            "schema_version": int(report["schema_version"]),
            "source": str(report.get("source", "")),
        },
        "summary": {
            "action_count": len(actions),
            "vq_reencode_action_count": sum(
                1 for action in actions if action["materialization_action"] == "vq_reencode_with_imatrix"
            ),
            "high_precision_action_count": sum(
                1
                for action in actions
                if action["materialization_action"] == "materialize_high_precision_projection"
            ),
            "total_weight_count": int(budget["total_weight_count"]),
            "effective_bits_per_weight": float(budget["effective_bits_per_weight"]),
            "weighted_error": float(budget["weighted_error"]),
        },
        "actions": actions,
    }


def attach_imatrix_manifest_to_materialization_plan(
    materialization_plan: Mapping[str, Any],
    imatrix_manifest: Mapping[str, Any],
    *,
    manifest_root: str | Path | None = None,
    require_sidecar_files: bool = False,
) -> dict[str, Any]:
    """Join imatrix sidecar metadata onto VQ actions without materializing artifacts."""

    errors: list[str] = []
    if not isinstance(materialization_plan, Mapping):
        return {
            "schema": "air_dynamic_precision_materialization_plan",
            "schema_version": 1,
            "ok": False,
            "errors": ["materialization_plan must be a mapping"],
            "actions": [],
        }
    if not isinstance(imatrix_manifest, Mapping):
        errors.append("imatrix_manifest must be a mapping")
        imatrix_manifest = {}

    if materialization_plan.get("schema") != "air_dynamic_precision_materialization_plan":
        errors.append("materialization_plan.schema must be air_dynamic_precision_materialization_plan")
    if materialization_plan.get("schema_version") != 1:
        errors.append("materialization_plan.schema_version must be 1")

    plan_errors = list(materialization_plan.get("errors", []))
    if materialization_plan.get("ok") is not True:
        errors.append("materialization_plan must be ok before joining imatrix sidecars")

    entries = _imatrix_manifest_entries(imatrix_manifest, errors)
    entry_index = _index_imatrix_manifest_entries(
        entries,
        errors,
        manifest_root=manifest_root,
        require_sidecar_files=require_sidecar_files,
    )

    actions_value = materialization_plan.get("actions")
    original_actions = actions_value if isinstance(actions_value, list) else []
    if not isinstance(actions_value, list):
        errors.append("materialization_plan.actions must be a list")

    joined_actions: list[dict[str, Any]] = []
    covered_action_count = 0
    missing_action_count = 0
    for action_value in original_actions:
        if not isinstance(action_value, Mapping):
            errors.append("materialization_plan.actions entries must be mappings")
            continue
        action = dict(action_value)
        if bool(action.get("requires_imatrix_sidecar")):
            lookup_key = (
                int(action["layer"]),
                str(action["projection"]),
                int(action["expert"]) if action.get("expert") is not None else None,
            )
            sidecar = entry_index.get(lookup_key)
            if sidecar is None:
                missing_action_count += 1
                errors.append(
                    "missing imatrix sidecar for "
                    f"{action.get('key')} layer={lookup_key[0]} "
                    f"projection={lookup_key[1]} expert={lookup_key[2]}"
                )
                action["imatrix_sidecar_available"] = False
                action["imatrix_sidecar"] = None
            else:
                covered_action_count += 1
                action["imatrix_sidecar_available"] = True
                action["imatrix_sidecar"] = sidecar
        else:
            action["imatrix_sidecar_available"] = False
            action["imatrix_sidecar"] = None
        joined_actions.append(action)

    result = dict(materialization_plan)
    result["ok"] = materialization_plan.get("ok") is True and not errors
    result["errors"] = plan_errors + errors
    result["imatrix_manifest"] = {
        "record_type": str(imatrix_manifest.get("record_type", "")),
        "prompt_set": str(imatrix_manifest.get("prompt_set", "")),
        "entry_count": int(imatrix_manifest.get("entry_count", len(entries))) if entries else int(imatrix_manifest.get("entry_count", 0) or 0),
        "covered_action_count": covered_action_count,
        "missing_action_count": missing_action_count,
        "sidecar_files_checked": bool(require_sidecar_files),
    }
    result["actions"] = joined_actions
    return result


def attach_high_precision_source_manifest_to_materialization_plan(
    materialization_plan: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
    require_source_files: bool = False,
) -> dict[str, Any]:
    """Join high-precision source metadata onto high-tier actions without reading weights."""

    errors: list[str] = []
    if not isinstance(materialization_plan, Mapping):
        return {
            "schema": "air_dynamic_precision_materialization_plan",
            "schema_version": 1,
            "ok": False,
            "errors": ["materialization_plan must be a mapping"],
            "actions": [],
        }
    if not isinstance(source_manifest, Mapping):
        errors.append("source_manifest must be a mapping")
        source_manifest = {}

    if materialization_plan.get("schema") != "air_dynamic_precision_materialization_plan":
        errors.append("materialization_plan.schema must be air_dynamic_precision_materialization_plan")
    if materialization_plan.get("schema_version") != 1:
        errors.append("materialization_plan.schema_version must be 1")

    plan_errors = list(materialization_plan.get("errors", []))
    if materialization_plan.get("ok") is not True:
        errors.append("materialization_plan must be ok before joining high-precision source")

    entries = _high_source_manifest_entries(source_manifest, errors)
    source_kind = str(source_manifest.get("source_kind", ""))
    entry_index = _index_high_source_manifest_entries(
        entries,
        errors,
        source_root=source_root,
        source_kind=source_kind,
        require_source_files=require_source_files,
    )

    actions_value = materialization_plan.get("actions")
    original_actions = actions_value if isinstance(actions_value, list) else []
    if not isinstance(actions_value, list):
        errors.append("materialization_plan.actions must be a list")

    joined_actions: list[dict[str, Any]] = []
    covered_action_count = 0
    missing_action_count = 0
    for action_value in original_actions:
        if not isinstance(action_value, Mapping):
            errors.append("materialization_plan.actions entries must be mappings")
            continue
        action = dict(action_value)
        if bool(action.get("requires_high_precision_source")):
            lookup_key = (
                int(action["layer"]),
                str(action["projection"]),
                int(action["expert"]) if action.get("expert") is not None else None,
            )
            source = entry_index.get(lookup_key)
            if source is None:
                missing_action_count += 1
                errors.append(
                    "missing high-precision source for "
                    f"{action.get('key')} layer={lookup_key[0]} "
                    f"projection={lookup_key[1]} expert={lookup_key[2]}"
                )
                action["high_precision_source_available"] = False
                action["high_precision_source"] = None
            else:
                covered_action_count += 1
                action["high_precision_source_available"] = True
                action["high_precision_source"] = source
        else:
            action["high_precision_source_available"] = False
            action["high_precision_source"] = None
        joined_actions.append(action)

    result = dict(materialization_plan)
    result["ok"] = materialization_plan.get("ok") is True and not errors
    result["errors"] = plan_errors + errors
    result["high_precision_source_manifest"] = {
        "record_type": str(source_manifest.get("record_type", "")),
        "source_kind": source_kind,
        "entry_count": int(source_manifest.get("entry_count", len(entries))) if entries else int(source_manifest.get("entry_count", 0) or 0),
        "covered_action_count": covered_action_count,
        "missing_action_count": missing_action_count,
        "source_files_checked": bool(require_source_files),
    }
    result["actions"] = joined_actions
    return result


def build_high_precision_source_manifest_from_index(
    materialization_plan: Mapping[str, Any],
    source_index: object,
    *,
    source_root: str | Path | None = None,
    source_kind: str = "safetensors_source_index",
    source_tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    inspect_source_headers: bool = False,
) -> dict[str, Any]:
    """Build high-tier source metadata from safetensors index/header metadata only."""

    errors: list[str] = []
    if not isinstance(materialization_plan, Mapping):
        errors.append("materialization_plan must be a mapping")
        materialization_plan = {}
    if materialization_plan.get("schema") != "air_dynamic_precision_materialization_plan":
        errors.append("materialization_plan.schema must be air_dynamic_precision_materialization_plan")
    if materialization_plan.get("schema_version") != 1:
        errors.append("materialization_plan.schema_version must be 1")
    plan_errors = [str(error) for error in materialization_plan.get("errors", [])]
    if materialization_plan.get("ok") is not True:
        errors.append("materialization_plan must be ok before building high-precision source manifest")

    source_kind = str(source_kind).strip()
    if not source_kind:
        errors.append("source_kind must be a non-empty string")
        source_kind = "safetensors_source_index"
    root = Path(source_root) if source_root is not None else None
    if inspect_source_headers and root is None:
        errors.append("source_root is required when inspect_source_headers is true")

    weight_map = _source_index_weight_map(source_index, errors)
    metadata_by_tensor = _source_tensor_metadata_map(source_tensor_metadata, errors)
    actions_value = materialization_plan.get("actions", [])
    actions = actions_value if isinstance(actions_value, list) else []
    if not isinstance(actions_value, list):
        errors.append("materialization_plan.actions must be a list")

    entries: list[dict[str, Any]] = []
    required_action_count = 0
    covered_action_count = 0
    missing_action_count = 0
    for position, action_value in enumerate(actions):
        if not isinstance(action_value, Mapping):
            errors.append(f"materialization_plan.actions[{position}] must be a mapping")
            continue
        if not bool(action_value.get("requires_high_precision_source")):
            continue
        required_action_count += 1
        try:
            layer = int(action_value["layer"])
            projection = str(action_value["projection"])
            expert_value = action_value.get("expert")
            expert = int(expert_value) if expert_value is not None else None
        except (KeyError, TypeError, ValueError) as exc:
            missing_action_count += 1
            errors.append(f"invalid high-precision source action at index {position}: {exc}")
            continue
        if projection not in AIR_ROUTED_PROJECTIONS:
            missing_action_count += 1
            errors.append(f"invalid high-precision source action projection {projection!r} at index {position}")
            continue
        if expert is None:
            missing_action_count += 1
            errors.append(f"{action_value.get('key')}: high-precision source action must include expert")
            continue
        tensor_name = _air_expert_source_tensor_name(layer=layer, projection=projection, expert=expert)
        shard_name = weight_map.get(tensor_name)
        if shard_name is None:
            missing_action_count += 1
            errors.append(f"missing high-precision source tensor {tensor_name}")
            continue
        source_path = Path(str(shard_name))
        if source_path.is_absolute() or ".." in source_path.parts:
            missing_action_count += 1
            errors.append(f"high-precision source path must be relative and contained: {shard_name}")
            continue

        tensor_metadata = (
            _inspect_source_tensor_header(root / source_path, tensor_name, errors)
            if inspect_source_headers and root is not None
            else metadata_by_tensor.get(tensor_name)
        )
        if tensor_metadata is None:
            missing_action_count += 1
            errors.append(f"missing high-precision source metadata for {tensor_name}")
            continue

        entries.append(
            {
                "layer": layer,
                "projection": projection,
                "expert": expert,
                "path": str(source_path),
                "tensor_name": tensor_name,
                "dtype": str(tensor_metadata["dtype"]),
                "shape": [int(dim) for dim in tensor_metadata["shape"]],
            }
        )
        covered_action_count += 1

    return {
        "schema_version": 1,
        "record_type": "air_dynamic_precision_high_source_manifest",
        "ok": not (plan_errors or errors),
        "errors": plan_errors + errors,
        "source_kind": source_kind,
        "source_headers_checked": bool(inspect_source_headers),
        "source_index_weight_count": len(weight_map),
        "required_action_count": required_action_count,
        "covered_action_count": covered_action_count,
        "missing_action_count": missing_action_count,
        "entry_count": len(entries),
        "entries": entries,
    }


def build_dynamic_precision_materialization_approval_bundle(
    materialization_plan: Mapping[str, Any],
    *,
    imatrix_manifest: Mapping[str, Any],
    source_index: object,
    source_dir: str | Path,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    imatrix_manifest_root: str | Path | None = None,
    source_root: str | Path | None = None,
    source_kind: str = "safetensors_source_index",
    source_tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    require_imatrix_sidecar_files: bool = False,
    require_source_files: bool = False,
    inspect_source_headers: bool = False,
) -> dict[str, Any]:
    """Compose all no-write materialization preflight checks into one packet."""

    resolved_source_root = Path(source_root) if source_root is not None else Path(source_dir)
    imatrix_joined = attach_imatrix_manifest_to_materialization_plan(
        materialization_plan,
        imatrix_manifest,
        manifest_root=imatrix_manifest_root,
        require_sidecar_files=require_imatrix_sidecar_files,
    )
    source_manifest = build_high_precision_source_manifest_from_index(
        materialization_plan,
        source_index,
        source_root=resolved_source_root,
        source_kind=source_kind,
        source_tensor_metadata=source_tensor_metadata,
        inspect_source_headers=inspect_source_headers,
    )
    source_joined = attach_high_precision_source_manifest_to_materialization_plan(
        imatrix_joined,
        source_manifest,
        source_root=resolved_source_root,
        require_source_files=require_source_files,
    )
    dry_run = build_dynamic_precision_materialization_dry_run(
        source_joined,
        source_dir=source_dir,
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
    )
    preflight = build_dynamic_precision_materialization_preflight(source_joined, dry_run)
    errors = _bundle_errors(imatrix_joined, source_manifest, source_joined, dry_run, preflight)
    return {
        "schema": "air_dynamic_precision_materialization_approval_bundle",
        "schema_version": 1,
        "ok": not errors and preflight.get("ok") is True,
        "errors": errors,
        "materialization_allowed": False,
        "imatrix_join": {
            "ok": bool(imatrix_joined.get("ok")),
            "manifest": dict(imatrix_joined.get("imatrix_manifest", {}))
            if isinstance(imatrix_joined.get("imatrix_manifest"), Mapping)
            else {},
        },
        "high_precision_source_manifest": source_manifest,
        "source_join": {
            "ok": bool(source_joined.get("ok")),
            "manifest": dict(source_joined.get("high_precision_source_manifest", {}))
            if isinstance(source_joined.get("high_precision_source_manifest"), Mapping)
            else {},
        },
        "dry_run": dry_run,
        "preflight": preflight,
        "summary": dict(preflight.get("summary", {})) if isinstance(preflight.get("summary"), Mapping) else {},
        "approval_request": dict(preflight.get("approval_request", {}))
        if isinstance(preflight.get("approval_request"), Mapping)
        else {},
        "blocked_operations": list(preflight.get("blocked_operations", []))
        if isinstance(preflight.get("blocked_operations"), list)
        else [],
    }


def build_dynamic_precision_tier_map_approval_bundle(
    report: Mapping[str, Any],
    *,
    candidate_name: str,
    imatrix_manifest: Mapping[str, Any],
    source_index: object,
    source_dir: str | Path,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    imatrix_manifest_root: str | Path | None = None,
    source_root: str | Path | None = None,
    source_kind: str = "safetensors_source_index",
    source_tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    require_imatrix_sidecar_files: bool = False,
    require_source_files: bool = False,
    inspect_source_headers: bool = False,
) -> dict[str, Any]:
    """Build a no-write approval bundle directly from a tier-map report."""

    report_validation = validate_dynamic_precision_tier_map_report(report)
    materialization_plan = build_dynamic_precision_materialization_plan(
        report,
        candidate_name=candidate_name,
    )
    materialization_bundle = build_dynamic_precision_materialization_approval_bundle(
        materialization_plan,
        imatrix_manifest=imatrix_manifest,
        source_index=source_index,
        source_dir=source_dir,
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
        imatrix_manifest_root=imatrix_manifest_root,
        source_root=source_root,
        source_kind=source_kind,
        source_tensor_metadata=source_tensor_metadata,
        require_imatrix_sidecar_files=require_imatrix_sidecar_files,
        require_source_files=require_source_files,
        inspect_source_headers=inspect_source_headers,
    )
    errors = _bundle_errors(report_validation, materialization_plan, materialization_bundle)
    return {
        "schema": "air_dynamic_precision_tier_map_approval_bundle",
        "schema_version": 1,
        "ok": not errors and materialization_bundle.get("ok") is True,
        "errors": errors,
        "materialization_allowed": False,
        "report_validation": report_validation,
        "materialization_plan": _materialization_plan_summary(materialization_plan),
        "materialization_bundle": materialization_bundle,
        "summary": dict(materialization_bundle.get("summary", {}))
        if isinstance(materialization_bundle.get("summary"), Mapping)
        else {},
        "approval_request": dict(materialization_bundle.get("approval_request", {}))
        if isinstance(materialization_bundle.get("approval_request"), Mapping)
        else {},
        "blocked_operations": list(materialization_bundle.get("blocked_operations", []))
        if isinstance(materialization_bundle.get("blocked_operations"), list)
        else [],
    }


def build_dynamic_precision_budget_sweep_approval_bundle(
    reports: Iterable[Mapping[str, Any]],
    *,
    candidate_name_prefix: str,
    imatrix_manifest: Mapping[str, Any],
    source_index: object,
    source_dir: str | Path,
    seed_artifact_dir: str | Path,
    output_root: str | Path,
    imatrix_manifest_root: str | Path | None = None,
    source_root: str | Path | None = None,
    source_kind: str = "safetensors_source_index",
    source_tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    require_imatrix_sidecar_files: bool = False,
    require_source_files: bool = False,
    inspect_source_headers: bool = False,
) -> dict[str, Any]:
    """Build no-write approval bundles for a planned budget sweep."""

    report_list = list(reports)
    output_base = Path(output_root)
    errors: list[str] = []
    if not report_list:
        errors.append("at least one budget report is required")

    candidate_prefix = _safe_candidate_prefix(candidate_name_prefix)
    reference_tensor_keys: tuple[str, ...] | None = None
    previous_requested_budget: float | None = None
    bundles: list[dict[str, Any]] = []
    budget_sweep: list[dict[str, Any]] = []

    for index, report in enumerate(report_list):
        report_errors: list[str] = []
        budget = report.get("budget") if isinstance(report, Mapping) else None
        if not isinstance(budget, Mapping):
            report_errors.append(f"reports[{index}].budget must be a mapping")
            budget = {}
        tensor_keys = _report_tensor_keys(report)
        if reference_tensor_keys is None and tensor_keys is not None:
            reference_tensor_keys = tensor_keys
        elif tensor_keys is not None and reference_tensor_keys is not None and tensor_keys != reference_tensor_keys:
            report_errors.append(f"reports[{index}] tensor keys do not match the first budget report")

        requested_budget_bpw = _requested_budget_bits_per_weight(budget, report_errors, index)
        if requested_budget_bpw is not None:
            if previous_requested_budget is not None and requested_budget_bpw < previous_requested_budget - 1.0e-9:
                report_errors.append("budget reports must be sorted by non-decreasing requested bits/weight")
            previous_requested_budget = requested_budget_bpw

        candidate_name = _budget_sweep_candidate_name(
            candidate_prefix,
            index=index,
            requested_budget_bits_per_weight=requested_budget_bpw,
        )
        bundle = build_dynamic_precision_tier_map_approval_bundle(
            report,
            candidate_name=candidate_name,
            imatrix_manifest=imatrix_manifest,
            source_index=source_index,
            source_dir=source_dir,
            seed_artifact_dir=seed_artifact_dir,
            output_dir=output_base / candidate_name,
            imatrix_manifest_root=imatrix_manifest_root,
            source_root=source_root,
            source_kind=source_kind,
            source_tensor_metadata=source_tensor_metadata,
            require_imatrix_sidecar_files=require_imatrix_sidecar_files,
            require_source_files=require_source_files,
            inspect_source_headers=inspect_source_headers,
        )
        if report_errors:
            bundle = dict(bundle)
            bundle["ok"] = False
            bundle["errors"] = _dedupe_errors([*report_errors, *bundle.get("errors", [])])
        bundles.append(bundle)
        errors.extend(report_errors)
        errors.extend(str(error) for error in bundle.get("errors", []))

        budget_sweep.append(
            {
                "index": index,
                "candidate_name": candidate_name,
                "bundle_ok": bool(bundle.get("ok")),
                "requested_budget_bits_per_weight": float(requested_budget_bpw)
                if requested_budget_bpw is not None
                else None,
                "effective_bits_per_weight": _finite_number(budget.get("effective_bits_per_weight")),
                "total_effective_bits": _finite_number(budget.get("total_effective_bits")),
                "weighted_error": _finite_number(budget.get("weighted_error")),
                "tier_counts": dict(report.get("tier_counts", {}))
                if isinstance(report, Mapping) and isinstance(report.get("tier_counts"), Mapping)
                else {},
                "summary": dict(bundle.get("summary", {}))
                if isinstance(bundle.get("summary"), Mapping)
                else {},
            }
        )

    errors = _dedupe_errors(errors)
    ready = not errors and all(bundle.get("ok") is True for bundle in bundles)
    return {
        "schema": "air_dynamic_precision_budget_sweep_approval_bundle",
        "schema_version": 1,
        "ok": ready,
        "errors": errors,
        "materialization_allowed": False,
        "summary": _budget_sweep_summary(budget_sweep, bundles),
        "budget_sweep": budget_sweep,
        "bundles": bundles,
        "approval_request": {
            "approval_required": True,
            "approval_gate": AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE,
            "ready": ready,
            "requested_operations": list(AIR_DYNAMIC_MATERIALIZATION_APPROVAL_OPERATIONS),
        },
        "blocked_operations": list(AIR_DYNAMIC_MATERIALIZATION_BLOCKED_OPERATIONS),
    }


def build_dynamic_precision_materialized_eval_plan(
    sweep_bundle: Mapping[str, Any],
    *,
    report_cache_path: str | Path,
    selection_cache_path: str | Path,
    baseline_report_cache_path: str | Path | None = None,
    baseline_selection_cache_path: str | Path | None = None,
    expected_report_rows: int = 128,
    expected_selection_rows: int = 128,
    top1_agreement_min: float = 0.75,
    min_effective_bits_per_weight: float = 2.0,
    target_effective_bits_per_weight: float = 2.4,
    max_effective_bits_per_weight: float = 3.0,
    lane_s_speed_gate: Mapping[str, Any] | None = None,
    require_cache_files: bool = False,
) -> dict[str, Any]:
    """Build the no-write report/selection eval contract for materialized candidates."""

    errors: list[str] = []
    if not isinstance(sweep_bundle, Mapping):
        errors.append("sweep_bundle must be a mapping")
        sweep_bundle = {}
    if sweep_bundle.get("schema") != "air_dynamic_precision_budget_sweep_approval_bundle":
        errors.append("sweep_bundle.schema must be air_dynamic_precision_budget_sweep_approval_bundle")
    if sweep_bundle.get("schema_version") != 1:
        errors.append("sweep_bundle.schema_version must be 1")
    sweep_errors = [str(error) for error in sweep_bundle.get("errors", [])]
    if sweep_bundle.get("ok") is not True:
        errors.append("sweep_bundle must be ok before materialized eval planning")

    if expected_report_rows <= 0:
        errors.append("expected_report_rows must be positive")
    if expected_selection_rows <= 0:
        errors.append("expected_selection_rows must be positive")
    if not (0.0 < float(top1_agreement_min) <= 1.0):
        errors.append("top1_agreement_min must be in (0, 1]")
    if min_effective_bits_per_weight <= 0.0:
        errors.append("min_effective_bits_per_weight must be positive")
    if target_effective_bits_per_weight <= 0.0:
        errors.append("target_effective_bits_per_weight must be positive")
    if max_effective_bits_per_weight <= 0.0:
        errors.append("max_effective_bits_per_weight must be positive")
    if not (
        min_effective_bits_per_weight
        <= target_effective_bits_per_weight
        <= max_effective_bits_per_weight
    ):
        errors.append("budget bits/weight must satisfy min <= target <= max")

    caches = {
        "report": _eval_cache_plan(
            split="report",
            prompt_set=AIR_DYNAMIC_REPORT_PROMPT_SET,
            expected_row_count=expected_report_rows,
            path=report_cache_path,
            baseline_path=baseline_report_cache_path,
            primary=True,
            require_cache_files=require_cache_files,
            errors=errors,
        ),
        "selection": _eval_cache_plan(
            split="selection",
            prompt_set=AIR_DYNAMIC_SELECTION_PROMPT_SET,
            expected_row_count=expected_selection_rows,
            path=selection_cache_path,
            baseline_path=baseline_selection_cache_path,
            primary=False,
            require_cache_files=require_cache_files,
            errors=errors,
        ),
    }

    budget_rows_value = sweep_bundle.get("budget_sweep", [])
    budget_rows = budget_rows_value if isinstance(budget_rows_value, list) else []
    if not isinstance(budget_rows_value, list):
        errors.append("sweep_bundle.budget_sweep must be a list")
    bundle_rows_value = sweep_bundle.get("bundles", [])
    bundle_rows = bundle_rows_value if isinstance(bundle_rows_value, list) else []
    if not isinstance(bundle_rows_value, list):
        errors.append("sweep_bundle.bundles must be a list")

    candidate_eval_plan: list[dict[str, Any]] = []
    if not errors:
        for index, row_value in enumerate(budget_rows):
            if not isinstance(row_value, Mapping):
                errors.append(f"sweep_bundle.budget_sweep[{index}] must be a mapping")
                candidate_eval_plan = []
                break
            if row_value.get("bundle_ok") is not True:
                errors.append(f"sweep budget {index} must be ok before eval planning")
                candidate_eval_plan = []
                break
            candidate_name = str(row_value.get("candidate_name", "")).strip()
            if not candidate_name:
                errors.append(f"sweep budget {index} candidate_name must be non-empty")
                candidate_eval_plan = []
                break
            materialized_artifact_dir = _planned_candidate_output_dir(bundle_rows, index)
            candidate_eval_plan.append(
                {
                    "index": index,
                    "candidate_name": candidate_name,
                    "materialized_artifact_dir": materialized_artifact_dir,
                    "candidate_materialized": False,
                    "eval_records_written": False,
                    "requested_budget_bits_per_weight": row_value.get("requested_budget_bits_per_weight"),
                    "effective_bits_per_weight": row_value.get("effective_bits_per_weight"),
                    "weighted_error": row_value.get("weighted_error"),
                    "splits": {
                        "report": _candidate_eval_split(caches["report"]),
                        "selection": _candidate_eval_split(caches["selection"]),
                    },
                }
            )

    errors = _dedupe_errors([*sweep_errors, *errors])
    ready = not errors
    if not ready:
        candidate_eval_plan = []
    return {
        "schema": "air_dynamic_precision_materialized_eval_plan",
        "schema_version": 1,
        "ok": ready,
        "errors": errors,
        "evaluation_allowed": False,
        "materialization_allowed": False,
        "metric_contract": {
            "primary_split": "report",
            "primary_metric": "mean_kld",
            "required_metrics": list(AIR_DYNAMIC_EVAL_REQUIRED_METRICS),
            "exact_full_logits_required": True,
        },
        "acceptance": {
            "top1_agreement_min": float(top1_agreement_min),
            "mean_kld_primary_split": "report",
            "kld_tail_metric": "p999_kld",
            "lane_s_speed_required": True,
        },
        "budget": {
            "min_effective_bits_per_weight": float(min_effective_bits_per_weight),
            "target_effective_bits_per_weight": float(target_effective_bits_per_weight),
            "max_effective_bits_per_weight": float(max_effective_bits_per_weight),
        },
        "selection_rule": {
            "rank_by": ["mean_kld", "p999_kld", "effective_bits_per_weight"],
            "decision": "pick_smallest_budget_clearing_top1_and_kld",
        },
        "caches": caches,
        "lane_s_speed_gate": dict(lane_s_speed_gate or {}),
        "summary": {
            "candidate_count": len(candidate_eval_plan),
            "report_expected_row_count": int(expected_report_rows),
            "selection_expected_row_count": int(expected_selection_rows),
            "evaluation_allowed": False,
        },
        "candidate_eval_plan": candidate_eval_plan,
        "approval_request": {
            "approval_required": True,
            "approval_gate": AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE,
            "ready": ready,
            "requested_operations": list(AIR_DYNAMIC_EVAL_APPROVAL_OPERATIONS),
        },
        "blocked_operations": list(AIR_DYNAMIC_EVAL_BLOCKED_OPERATIONS),
    }


def build_dynamic_precision_eval_frontier_report(
    eval_plan: Mapping[str, Any],
    eval_results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rank materialized eval metrics KLD-first without writing acceptance artifacts."""

    errors: list[str] = []
    if not isinstance(eval_plan, Mapping):
        errors.append("eval_plan must be a mapping")
        eval_plan = {}
    plan_errors = [str(error) for error in eval_plan.get("errors", [])]
    if eval_plan.get("schema") != "air_dynamic_precision_materialized_eval_plan":
        errors.append("eval_plan.schema must be air_dynamic_precision_materialized_eval_plan")
    if eval_plan.get("schema_version") != 1:
        errors.append("eval_plan.schema_version must be 1")
    if eval_plan.get("ok") is not True:
        errors.append("eval_plan must be ok before frontier reporting")

    metric_contract = _plain_mapping(eval_plan.get("metric_contract"))
    required_metrics = _sequence_strings(metric_contract.get("required_metrics"))
    if not required_metrics:
        required_metrics = list(AIR_DYNAMIC_EVAL_REQUIRED_METRICS)
    acceptance = _plain_mapping(eval_plan.get("acceptance"))
    budget = _plain_mapping(eval_plan.get("budget"))
    top1_min = _finite_number(acceptance.get("top1_agreement_min"))
    if top1_min is None:
        top1_min = 0.75
    min_bits = _finite_number(budget.get("min_effective_bits_per_weight"))
    max_bits = _finite_number(budget.get("max_effective_bits_per_weight"))
    caches = _plain_mapping(eval_plan.get("caches"))
    report_expected_rows = _optional_int(_plain_mapping(caches.get("report")).get("expected_row_count")) or 0
    selection_expected_rows = _optional_int(_plain_mapping(caches.get("selection")).get("expected_row_count")) or 0

    planned_candidates = {
        str(row.get("candidate_name")): row
        for row in eval_plan.get("candidate_eval_plan", [])
        if isinstance(row, Mapping) and str(row.get("candidate_name", "")).strip()
    }
    if not planned_candidates:
        errors.append("eval_plan.candidate_eval_plan must contain candidates")

    result_rows = list(eval_results)
    frontier: list[dict[str, Any]] = []
    seen_results: set[str] = set()
    for index, result_value in enumerate(result_rows):
        row_errors: list[str] = []
        failures: list[str] = []
        if not isinstance(result_value, Mapping):
            errors.append(f"eval_results[{index}] must be a mapping")
            continue
        candidate_name = str(result_value.get("candidate_name", "")).strip()
        if not candidate_name:
            errors.append(f"eval_results[{index}].candidate_name must be non-empty")
            continue
        if candidate_name in seen_results:
            errors.append(f"duplicate eval result for candidate {candidate_name}")
            continue
        seen_results.add(candidate_name)
        if candidate_name not in planned_candidates:
            row_errors.append(f"{candidate_name}: candidate not present in eval_plan")

        splits = _plain_mapping(result_value.get("splits"))
        report = _frontier_split_metrics(
            splits.get("report"),
            split="report",
            required_metrics=required_metrics,
            expected_rows=report_expected_rows,
            top1_min=top1_min,
            errors=row_errors,
        )
        selection = _frontier_split_metrics(
            splits.get("selection"),
            split="selection",
            required_metrics=required_metrics,
            expected_rows=selection_expected_rows,
            top1_min=top1_min,
            errors=row_errors,
        )
        for split_name, metrics in (("report", report), ("selection", selection)):
            if metrics.get("top1_pass") is False:
                failures.append(f"{split_name}.mean_top1_agreement below threshold")
            if metrics.get("clean_rows_pass") is False:
                failures.append(f"{split_name}.clean_row_count below expected rows")
            if metrics.get("lane_s_speed_pass") is False:
                failures.append(f"{split_name}.lane_s_speed failed")

        effective_bits = report.get("effective_bits_per_weight")
        if effective_bits is None:
            row_errors.append(f"{candidate_name}: report.effective_bits_per_weight must be finite")
        else:
            if min_bits is not None and float(effective_bits) < min_bits:
                failures.append("effective_bits_per_weight below minimum budget")
            if max_bits is not None and float(effective_bits) > max_bits:
                failures.append("effective_bits_per_weight above maximum budget")

        errors.extend(row_errors)
        frontier.append(
            {
                "candidate_name": candidate_name,
                "eligible": not row_errors and not failures,
                "failures": _dedupe_errors(failures),
                "metrics": {
                    "report": _frontier_metric_payload(report),
                    "selection": _frontier_metric_payload(selection),
                },
                "effective_bits_per_weight": effective_bits,
            }
        )

    missing_results = sorted(set(planned_candidates) - seen_results)
    for candidate_name in missing_results:
        errors.append(f"missing eval result for candidate {candidate_name}")

    eligible_rows = [row for row in frontier if row["eligible"]]
    selected = None
    if not errors and eligible_rows:
        selected = min(
            eligible_rows,
            key=lambda row: (
                float(row["metrics"]["report"]["mean_kld"]),
                float(row["metrics"]["report"]["p999_kld"]),
                float(row["effective_bits_per_weight"]),
                str(row["candidate_name"]),
            ),
        )
    status = "invalid"
    if not errors:
        status = "candidate_selected" if selected is not None else "escalate_no_candidate"

    all_errors = _dedupe_errors([*plan_errors, *errors])
    return {
        "schema": "air_dynamic_precision_eval_frontier_report",
        "schema_version": 1,
        "ok": not all_errors,
        "errors": all_errors,
        "candidate_acceptance_claimed": False,
        "result_artifact_written": False,
        "frontier": frontier,
        "decision": {
            "status": status,
            "selected_candidate_name": selected["candidate_name"] if selected is not None else None,
            "rank_by": ["report.mean_kld", "report.p999_kld", "effective_bits_per_weight"],
            "primary_split": "report",
            "primary_metric": "mean_kld",
        },
        "summary": {
            "result_count": len(result_rows),
            "frontier_count": len(frontier),
            "eligible_candidate_count": len(eligible_rows),
            "top1_agreement_min": float(top1_min),
            "candidate_acceptance_claimed": False,
        },
    }


def build_dynamic_precision_eval_frontier_contract(eval_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the post-eval KLD-first decision contract before result rows exist."""

    errors: list[str] = []
    if not isinstance(eval_plan, Mapping):
        errors.append("eval_plan must be a mapping")
        eval_plan = {}
    plan_errors = [str(error) for error in eval_plan.get("errors", [])]
    if eval_plan.get("schema") != "air_dynamic_precision_materialized_eval_plan":
        errors.append("eval_plan.schema must be air_dynamic_precision_materialized_eval_plan")
    if eval_plan.get("schema_version") != 1:
        errors.append("eval_plan.schema_version must be 1")
    if eval_plan.get("ok") is not True:
        errors.append("eval_plan must be ok before frontier contract planning")
    if eval_plan.get("evaluation_allowed") is not False:
        errors.append("eval_plan.evaluation_allowed must be false")
    if eval_plan.get("materialization_allowed") is not False:
        errors.append("eval_plan.materialization_allowed must be false")

    metric_contract = _plain_mapping(eval_plan.get("metric_contract"))
    required_metrics = _sequence_strings(metric_contract.get("required_metrics"))
    if not required_metrics:
        required_metrics = list(AIR_DYNAMIC_EVAL_REQUIRED_METRICS)
    acceptance = _plain_mapping(eval_plan.get("acceptance"))
    budget = _plain_mapping(eval_plan.get("budget"))
    selection_rule = _plain_mapping(eval_plan.get("selection_rule"))
    summary = _plain_mapping(eval_plan.get("summary"))

    candidate_rows_value = eval_plan.get("candidate_eval_plan", [])
    if not isinstance(candidate_rows_value, list):
        errors.append("eval_plan.candidate_eval_plan must be a list")
        candidate_rows: list[Any] = []
    else:
        candidate_rows = candidate_rows_value
    candidate_names: list[str] = []
    seen_candidates: set[str] = set()
    for index, row_value in enumerate(candidate_rows):
        if not isinstance(row_value, Mapping):
            errors.append(f"eval_plan.candidate_eval_plan[{index}] must be a mapping")
            continue
        candidate_name = str(row_value.get("candidate_name", "")).strip()
        if not candidate_name:
            errors.append(f"eval_plan.candidate_eval_plan[{index}].candidate_name must be non-empty")
            continue
        if candidate_name in seen_candidates:
            errors.append(f"duplicate eval_plan candidate {candidate_name}")
            continue
        seen_candidates.add(candidate_name)
        candidate_names.append(candidate_name)

    rank_by_value = _sequence_strings(selection_rule.get("rank_by"))
    rank_by = (
        ["report.mean_kld", "report.p999_kld", "effective_bits_per_weight"]
        if rank_by_value == ["mean_kld", "p999_kld", "effective_bits_per_weight"]
        else rank_by_value
    )
    if not rank_by:
        rank_by = ["report.mean_kld", "report.p999_kld", "effective_bits_per_weight"]

    all_errors = _dedupe_errors([*plan_errors, *errors])
    return {
        "schema": "air_dynamic_precision_eval_frontier_contract",
        "schema_version": 1,
        "ok": not all_errors,
        "errors": all_errors,
        "awaits_eval_results": True,
        "evaluation_allowed": False,
        "result_artifact_written": False,
        "candidate_acceptance_claimed": False,
        "frontier_report_schema": "air_dynamic_precision_eval_frontier_report",
        "candidate_count": _optional_int(summary.get("candidate_count")) or len(candidate_names),
        "candidate_names": candidate_names,
        "required_splits": ["report", "selection"],
        "required_metrics": required_metrics,
        "rank_by": rank_by,
        "gates": {
            "top1_agreement_min": _finite_number(acceptance.get("top1_agreement_min")),
            "lane_s_speed_required": acceptance.get("lane_s_speed_required") is True,
            "clean_report_rows": _optional_int(summary.get("report_expected_row_count")),
            "clean_selection_rows": _optional_int(summary.get("selection_expected_row_count")),
            "budget": {
                "min_effective_bits_per_weight": _finite_number(
                    budget.get("min_effective_bits_per_weight")
                ),
                "target_effective_bits_per_weight": _finite_number(
                    budget.get("target_effective_bits_per_weight")
                ),
                "max_effective_bits_per_weight": _finite_number(
                    budget.get("max_effective_bits_per_weight")
                ),
            },
        },
        "decision_statuses": ["candidate_selected", "escalate_no_candidate", "invalid"],
        "blocked_operations": list(AIR_DYNAMIC_EVAL_BLOCKED_OPERATIONS),
    }


def build_dynamic_precision_approval_batch_plan(
    *,
    imatrix_collection_plan: Mapping[str, Any],
    sweep_bundle: Mapping[str, Any],
    eval_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the full no-write approval gate for real dynamic precision work."""

    errors: list[str] = []
    if not isinstance(imatrix_collection_plan, Mapping):
        errors.append("imatrix_collection_plan must be a mapping")
        imatrix_collection_plan = {}
    if not isinstance(sweep_bundle, Mapping):
        errors.append("sweep_bundle must be a mapping")
        sweep_bundle = {}
    if not isinstance(eval_plan, Mapping):
        errors.append("eval_plan must be a mapping")
        eval_plan = {}

    collection_errors = [str(error) for error in imatrix_collection_plan.get("errors", [])]
    sweep_errors = [str(error) for error in sweep_bundle.get("errors", [])]
    eval_errors = [str(error) for error in eval_plan.get("errors", [])]

    if imatrix_collection_plan.get("schema") != "air_projection_imatrix_collection_plan":
        errors.append("imatrix_collection_plan.schema must be air_projection_imatrix_collection_plan")
    if imatrix_collection_plan.get("schema_version") != 1:
        errors.append("imatrix_collection_plan.schema_version must be 1")
    if imatrix_collection_plan.get("ok") is not True:
        errors.append("imatrix_collection_plan must be ok before approval batch planning")
    if imatrix_collection_plan.get("collection_allowed") is not False:
        errors.append("imatrix_collection_plan.collection_allowed must be false")
    if imatrix_collection_plan.get("sidecar_writes_allowed") is not False:
        errors.append("imatrix_collection_plan.sidecar_writes_allowed must be false")

    if sweep_bundle.get("schema") != "air_dynamic_precision_budget_sweep_approval_bundle":
        errors.append("sweep_bundle.schema must be air_dynamic_precision_budget_sweep_approval_bundle")
    if sweep_bundle.get("schema_version") != 1:
        errors.append("sweep_bundle.schema_version must be 1")
    if sweep_bundle.get("ok") is not True:
        errors.append("sweep_bundle must be ok before approval batch planning")
    if sweep_bundle.get("materialization_allowed") is not False:
        errors.append("sweep_bundle.materialization_allowed must be false")

    if eval_plan.get("schema") != "air_dynamic_precision_materialized_eval_plan":
        errors.append("eval_plan.schema must be air_dynamic_precision_materialized_eval_plan")
    if eval_plan.get("schema_version") != 1:
        errors.append("eval_plan.schema_version must be 1")
    if eval_plan.get("ok") is not True:
        errors.append("eval_plan must be ok before approval batch planning")
    if eval_plan.get("evaluation_allowed") is not False:
        errors.append("eval_plan.evaluation_allowed must be false")
    if eval_plan.get("materialization_allowed") is not False:
        errors.append("eval_plan.materialization_allowed must be false")

    collection_filesystem = _plain_mapping(imatrix_collection_plan.get("filesystem"))
    if collection_filesystem.get("writes_performed") is not False:
        errors.append("imatrix_collection_plan.filesystem.writes_performed must be false")
    if collection_filesystem.get("manifest_written") is not False:
        errors.append("imatrix_collection_plan.filesystem.manifest_written must be false")

    collection_summary = _plain_mapping(imatrix_collection_plan.get("summary"))
    sweep_summary = _plain_mapping(sweep_bundle.get("summary"))
    eval_summary = _plain_mapping(eval_plan.get("summary"))
    budget_count = _optional_int(sweep_summary.get("budget_count"))
    ready_bundle_count = _optional_int(sweep_summary.get("ready_bundle_count"))
    candidate_eval_count = _optional_int(eval_summary.get("candidate_count"))
    if budget_count is not None and candidate_eval_count is not None and budget_count != candidate_eval_count:
        errors.append("eval_plan candidate_count must match sweep budget_count")
    if ready_bundle_count is not None and candidate_eval_count is not None and ready_bundle_count != candidate_eval_count:
        errors.append("eval_plan candidate_count must match ready sweep bundles")

    frontier_contract = build_dynamic_precision_eval_frontier_contract(eval_plan)
    frontier_errors = [str(error) for error in frontier_contract.get("errors", [])]

    errors = _dedupe_errors([*collection_errors, *sweep_errors, *eval_errors, *frontier_errors, *errors])
    ready = not errors
    requested_operations = _dedupe_errors(
        [
            *_approval_request_operations(imatrix_collection_plan),
            *_approval_request_operations(sweep_bundle),
            *_approval_request_operations(eval_plan),
        ]
    )
    blocked_operations = _dedupe_errors(
        [
            *_sequence_strings(imatrix_collection_plan.get("blocked_operations")),
            *_sequence_strings(sweep_bundle.get("blocked_operations")),
            *_sequence_strings(eval_plan.get("blocked_operations")),
        ]
    )
    state = {
        "collection_allowed": False,
        "sidecar_writes_allowed": False,
        "materialization_allowed": False,
        "evaluation_allowed": False,
        "candidate_acceptance_allowed": False,
        "writes_performed": False,
    }
    return {
        "schema": "air_dynamic_precision_approval_batch_plan",
        "schema_version": 1,
        "ok": ready,
        "errors": errors,
        "collection_allowed": False,
        "materialization_allowed": False,
        "evaluation_allowed": False,
        "state": state,
        "summary": {
            "planned_sidecar_count": _optional_int(collection_summary.get("planned_sidecar_count")),
            "imatrix_prompt_count": _optional_int(
                _plain_mapping(imatrix_collection_plan.get("prompt_validation")).get("prompt_count")
            ),
            "budget_count": budget_count,
            "ready_bundle_count": ready_bundle_count,
            "candidate_eval_count": candidate_eval_count,
            "frontier_contract_ready": frontier_contract.get("ok") is True,
            "report_expected_row_count": _optional_int(eval_summary.get("report_expected_row_count")),
            "selection_expected_row_count": _optional_int(eval_summary.get("selection_expected_row_count")),
        },
        "approval_request": {
            "approval_required": True,
            "approval_gate": AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE,
            "ready": ready,
            "requested_operations": requested_operations,
        },
        "blocked_operations": blocked_operations,
        "plans": {
            "imatrix_collection": dict(imatrix_collection_plan),
            "materialization_sweep": dict(sweep_bundle),
            "materialized_eval": dict(eval_plan),
            "eval_frontier_contract": dict(frontier_contract),
        },
    }


def validate_dynamic_precision_approval_batch_objective(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Audit the no-write approval batch against the dynamic-precision objective."""

    errors: list[str] = []
    if not isinstance(batch, Mapping):
        errors.append("batch must be a mapping")
        batch = {}
    batch_errors = [str(error) for error in batch.get("errors", [])]
    if batch.get("schema") != "air_dynamic_precision_approval_batch_plan":
        errors.append("batch.schema must be air_dynamic_precision_approval_batch_plan")
    if batch.get("schema_version") != 1:
        errors.append("batch.schema_version must be 1")
    if batch.get("ok") is not True:
        errors.append("batch must be ok before objective audit")

    state = _plain_mapping(batch.get("state"))
    closed_flag_names = (
        "collection_allowed",
        "sidecar_writes_allowed",
        "materialization_allowed",
        "evaluation_allowed",
        "candidate_acceptance_allowed",
        "writes_performed",
    )
    all_execution_flags_closed = True
    for flag_name in closed_flag_names:
        if state.get(flag_name) is not False:
            errors.append(f"batch.state.{flag_name} must be false before approval")
            all_execution_flags_closed = False
    for flag_name in ("collection_allowed", "materialization_allowed", "evaluation_allowed"):
        if batch.get(flag_name) is not False:
            errors.append(f"batch.{flag_name} must be false before approval")
            all_execution_flags_closed = False

    plans = _plain_mapping(batch.get("plans"))
    imatrix_plan = _plain_mapping(plans.get("imatrix_collection"))
    eval_plan = _plain_mapping(plans.get("materialized_eval"))
    frontier_contract = _plain_mapping(plans.get("eval_frontier_contract"))

    collection = _plain_mapping(imatrix_plan.get("collection"))
    collection_source = str(collection.get("collection_source", ""))
    if collection_source not in AIR_DYNAMIC_APPROVED_IMATRIX_COLLECTION_SOURCES:
        errors.append("collection_source must stay single-box resident_vq_model or streamed_source_weights")
    if collection.get("prompt_set") != "air_imatrix_calib_v1":
        errors.append("imatrix collection prompt_set must be air_imatrix_calib_v1")
    if collection.get("use_chat_template") is not True:
        errors.append("imatrix collection must use chat template")
    if collection.get("include_activation_rows") is not True:
        errors.append("imatrix collection must include activation rows")
    collection_filesystem = _plain_mapping(imatrix_plan.get("filesystem"))
    if collection_filesystem.get("writes_performed") is not False:
        errors.append("imatrix collection filesystem writes_performed must be false")
    if collection_filesystem.get("manifest_written") is not False:
        errors.append("imatrix collection manifest_written must be false")

    metric_contract = _plain_mapping(eval_plan.get("metric_contract"))
    primary_split = str(metric_contract.get("primary_split", ""))
    primary_metric = str(metric_contract.get("primary_metric", ""))
    if primary_split != "report" or primary_metric != "mean_kld":
        errors.append("primary metric must remain report.mean_kld")

    acceptance = _plain_mapping(eval_plan.get("acceptance"))
    top1_min = _finite_number(acceptance.get("top1_agreement_min"))
    if top1_min != 0.75:
        errors.append("top1_agreement_min must remain 0.75")
    if acceptance.get("lane_s_speed_required") is not True:
        errors.append("lane_s_speed_required must remain true")

    budget = _plain_mapping(eval_plan.get("budget"))
    min_bits = _finite_number(budget.get("min_effective_bits_per_weight"))
    target_bits = _finite_number(budget.get("target_effective_bits_per_weight"))
    max_bits = _finite_number(budget.get("max_effective_bits_per_weight"))
    if min_bits != 2.0:
        errors.append("min_effective_bits_per_weight must remain 2.0")
    if target_bits != 2.4:
        errors.append("target_effective_bits_per_weight must remain 2.4")
    if max_bits != 3.0:
        errors.append("max_effective_bits_per_weight must remain 3.0")

    caches = _plain_mapping(eval_plan.get("caches"))
    report_cache = _plain_mapping(caches.get("report"))
    selection_cache = _plain_mapping(caches.get("selection"))
    report_expected_rows = _optional_int(report_cache.get("expected_row_count"))
    selection_expected_rows = _optional_int(selection_cache.get("expected_row_count"))
    if report_expected_rows != 128:
        errors.append("report expected_row_count must remain 128")
    if selection_expected_rows != 128:
        errors.append("selection expected_row_count must remain 128")
    if report_cache.get("prompt_set") != AIR_DYNAMIC_REPORT_PROMPT_SET:
        errors.append("report prompt_set must remain air_vq_ladder_report_v1")
    if selection_cache.get("prompt_set") != AIR_DYNAMIC_SELECTION_PROMPT_SET:
        errors.append("selection prompt_set must remain air_vq_ladder_select_v1")

    summary = _plain_mapping(batch.get("summary"))
    frontier_ready = bool(summary.get("frontier_contract_ready") is True and frontier_contract.get("ok") is True)
    if not frontier_ready:
        errors.append("eval frontier contract must be ready in approval batch")
    rank_by = _sequence_strings(frontier_contract.get("rank_by"))
    expected_rank_by = ["report.mean_kld", "report.p999_kld", "effective_bits_per_weight"]
    if rank_by != expected_rank_by:
        errors.append("frontier rank_by must remain KLD-first")
    if frontier_contract.get("candidate_acceptance_claimed") is not False:
        errors.append("frontier contract candidate_acceptance_claimed must be false")
    if frontier_contract.get("result_artifact_written") is not False:
        errors.append("frontier contract result_artifact_written must be false")

    blocked_operations = _sequence_strings(batch.get("blocked_operations"))
    if "candidate_acceptance_or_ranking_claim" not in blocked_operations:
        errors.append("candidate acceptance/ranking must remain blocked")
    if "real_activation_capture" not in blocked_operations:
        errors.append("real activation capture must remain blocked")
    if "report_selection_cache_evaluation_and_budget_sweep" not in blocked_operations:
        errors.append("report/selection eval and budget sweep must remain blocked")

    all_errors = _dedupe_errors([*batch_errors, *errors])
    return {
        "schema": "air_dynamic_precision_objective_audit",
        "schema_version": 1,
        "ok": not all_errors,
        "errors": all_errors,
        "objective": {
            "single_m5_only": True,
            "implementation": "MLX-only",
            "materialization_approval_gated": True,
        },
        "requirements": {
            "collection_source": collection_source,
            "approved_collection_sources": list(AIR_DYNAMIC_APPROVED_IMATRIX_COLLECTION_SOURCES),
            "top1_agreement_min": top1_min,
            "budget": {
                "min_effective_bits_per_weight": min_bits,
                "target_effective_bits_per_weight": target_bits,
                "max_effective_bits_per_weight": max_bits,
            },
            "primary_metric": f"{primary_split}.{primary_metric}" if primary_split and primary_metric else "",
            "rank_by": rank_by,
            "frontier_contract_ready": frontier_ready,
            "report_expected_row_count": report_expected_rows,
            "selection_expected_row_count": selection_expected_rows,
        },
        "state": {
            "all_execution_flags_closed": all_execution_flags_closed,
            "blocked_operations": blocked_operations,
        },
    }


def build_dynamic_precision_human_approval_request(
    *,
    approval_batch: Mapping[str, Any],
    objective_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a no-write, human-readable approval request from an audited batch."""

    errors: list[str] = []
    if not isinstance(approval_batch, Mapping):
        errors.append("approval_batch must be a mapping")
        approval_batch = {}
    if not isinstance(objective_audit, Mapping):
        errors.append("objective_audit must be a mapping")
        objective_audit = {}

    batch_errors = [str(error) for error in approval_batch.get("errors", [])]
    audit_errors = [str(error) for error in objective_audit.get("errors", [])]
    if approval_batch.get("schema") != "air_dynamic_precision_approval_batch_plan":
        errors.append("approval_batch.schema must be air_dynamic_precision_approval_batch_plan")
    if approval_batch.get("schema_version") != 1:
        errors.append("approval_batch.schema_version must be 1")
    if approval_batch.get("ok") is not True:
        errors.append("approval_batch must be ok before human approval request")
    if objective_audit.get("schema") != "air_dynamic_precision_objective_audit":
        errors.append("objective_audit.schema must be air_dynamic_precision_objective_audit")
    if objective_audit.get("schema_version") != 1:
        errors.append("objective_audit.schema_version must be 1")
    if objective_audit.get("ok") is not True:
        errors.append("objective_audit must be ok before human approval request")

    state = _plain_mapping(approval_batch.get("state"))
    for flag_name in (
        "collection_allowed",
        "materialization_allowed",
        "evaluation_allowed",
        "candidate_acceptance_allowed",
    ):
        if state.get(flag_name) is not False:
            errors.append(f"approval_batch.state.{flag_name} must be false before human approval")

    approval_request = _plain_mapping(approval_batch.get("approval_request"))
    summary = _plain_mapping(approval_batch.get("summary"))
    audit_requirements = _plain_mapping(objective_audit.get("requirements"))
    audit_objective = _plain_mapping(objective_audit.get("objective"))
    ready = not _dedupe_errors([*batch_errors, *audit_errors, *errors])
    requested_operations = (
        _sequence_strings(approval_request.get("requested_operations"))
        if ready
        else []
    )
    forbidden_without_approval = _dedupe_errors(
        [
            *_sequence_strings(approval_batch.get("blocked_operations")),
            "candidate_acceptance_or_ranking_claim",
        ]
    )
    top1_min = _finite_number(audit_requirements.get("top1_agreement_min"))
    budget = _plain_mapping(audit_requirements.get("budget"))
    approval_text = (
        "Approve the GLM-4.5-Air dynamic mixed precision real-data batch: "
        "single-M5 MLX-only imatrix sidecar collection, high-source metadata validation, "
        "fresh candidate materialization, 128-row report/selection eval, and KLD frontier "
        f"selection with top-1 >= {top1_min} and report mean KLD as primary."
    )
    all_errors = _dedupe_errors([*batch_errors, *audit_errors, *errors])
    return {
        "schema": "air_dynamic_precision_human_approval_request",
        "schema_version": 1,
        "ready_for_human_approval": not all_errors,
        "errors": all_errors,
        "approval_required": True,
        "approval_gate": AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE,
        "single_m5_only": audit_objective.get("single_m5_only") is True,
        "implementation": str(audit_objective.get("implementation", "MLX-only")),
        "execution_flags": {
            "collection_allowed": False,
            "materialization_allowed": False,
            "evaluation_allowed": False,
            "candidate_acceptance_allowed": False,
        },
        "requested_scope": {
            "candidate_count": _optional_int(summary.get("candidate_eval_count")),
            "collection_source": str(audit_requirements.get("collection_source", "")),
            "report_expected_row_count": _optional_int(
                audit_requirements.get("report_expected_row_count")
            ),
            "selection_expected_row_count": _optional_int(
                audit_requirements.get("selection_expected_row_count")
            ),
            "top1_agreement_min": top1_min,
            "budget": {
                "min_effective_bits_per_weight": _finite_number(
                    budget.get("min_effective_bits_per_weight")
                ),
                "target_effective_bits_per_weight": _finite_number(
                    budget.get("target_effective_bits_per_weight")
                ),
                "max_effective_bits_per_weight": _finite_number(
                    budget.get("max_effective_bits_per_weight")
                ),
            },
            "primary_metric": str(audit_requirements.get("primary_metric", "")),
            "rank_by": _sequence_strings(audit_requirements.get("rank_by")),
        },
        "requested_operations": requested_operations,
        "forbidden_without_approval": forbidden_without_approval,
        "approval_text": approval_text,
        "source_schemas": {
            "approval_batch": str(approval_batch.get("schema", "")),
            "objective_audit": str(objective_audit.get("schema", "")),
        },
    }


def build_dynamic_precision_human_approval_markdown(approval_request: Mapping[str, Any]) -> dict[str, Any]:
    """Render the audited approval request as paste-ready markdown without opening the gate."""

    errors: list[str] = []
    if not isinstance(approval_request, Mapping):
        errors.append("approval_request must be a mapping")
        approval_request = {}

    request_errors = [str(error) for error in approval_request.get("errors", [])]
    if approval_request.get("schema") != "air_dynamic_precision_human_approval_request":
        errors.append("approval_request.schema must be air_dynamic_precision_human_approval_request")
    if approval_request.get("schema_version") != 1:
        errors.append("approval_request.schema_version must be 1")
    ready_request = approval_request.get("ready_for_human_approval") is True
    if not ready_request:
        errors.append("approval_request must be ready before rendering approval markdown")
    if approval_request.get("approval_required") is not True:
        errors.append("approval_request.approval_required must be true")
    approval_gate = str(approval_request.get("approval_gate", ""))
    if approval_gate != AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE:
        errors.append("approval_request.approval_gate mismatch")
    if approval_request.get("single_m5_only") is not True:
        errors.append("approval_request.single_m5_only must be true")
    implementation = str(approval_request.get("implementation", ""))
    if implementation != "MLX-only":
        errors.append("approval_request.implementation must be MLX-only")

    execution_flags = _plain_mapping(approval_request.get("execution_flags"))
    for flag_name in (
        "collection_allowed",
        "materialization_allowed",
        "evaluation_allowed",
        "candidate_acceptance_allowed",
    ):
        if execution_flags.get(flag_name) is not False:
            errors.append(f"approval_request.execution_flags.{flag_name} must be false")

    requested_scope = _plain_mapping(approval_request.get("requested_scope"))
    budget = _plain_mapping(requested_scope.get("budget"))
    requested_operations_raw = _sequence_strings(approval_request.get("requested_operations"))
    forbidden_raw = _sequence_strings(approval_request.get("forbidden_without_approval"))
    if ready_request and not requested_operations_raw:
        errors.append("approval_request.requested_operations must be non-empty")
    approval_text = str(approval_request.get("approval_text", ""))
    if ready_request and not approval_text.strip():
        errors.append("approval_request.approval_text must be non-empty")

    all_errors = _dedupe_errors([*request_errors, *errors])
    requested_operations = requested_operations_raw if not all_errors else []
    forbidden_without_approval = forbidden_raw if not all_errors else _dedupe_errors(forbidden_raw)

    def _fmt(value: object) -> str:
        number = _finite_number(value)
        if number is not None:
            return str(number)
        return str(value) if value is not None else ""

    rank_by = _sequence_strings(requested_scope.get("rank_by"))
    markdown = ""
    if not all_errors:
        markdown_lines = [
            "# GLM-4.5-Air Dynamic Mixed Precision Approval Request",
            "",
            f"Approval gate: {approval_gate}",
            f"Scope: single-M5 {implementation}",
            f"Collection source: {requested_scope.get('collection_source', '')}",
            f"Candidates: {_fmt(requested_scope.get('candidate_count'))}",
            (
                "Budget: "
                f"{_fmt(budget.get('min_effective_bits_per_weight'))} -> "
                f"{_fmt(budget.get('target_effective_bits_per_weight'))} -> "
                f"{_fmt(budget.get('max_effective_bits_per_weight'))} effective bits/weight"
            ),
            f"Acceptance: top-1 >= {_fmt(requested_scope.get('top1_agreement_min'))}",
            f"Primary metric: {requested_scope.get('primary_metric', '')}",
            f"Rank by: {', '.join(rank_by)}",
            (
                "Caches: "
                f"report {_fmt(requested_scope.get('report_expected_row_count'))} rows, "
                f"selection {_fmt(requested_scope.get('selection_expected_row_count'))} rows"
            ),
            "",
            "Requested approval:",
            approval_text,
            "",
            "Requested operations:",
            *[f"- {operation}" for operation in requested_operations],
            "",
            "Still forbidden without approval:",
            *[f"- {operation}" for operation in forbidden_without_approval],
            "",
            "Execution flags before approval:",
            "- collection_allowed=false",
            "- materialization_allowed=false",
            "- evaluation_allowed=false",
            "- candidate_acceptance_allowed=false",
        ]
        markdown = "\n".join(markdown_lines)

    return {
        "schema": "air_dynamic_precision_human_approval_markdown",
        "schema_version": 1,
        "ok": not all_errors,
        "errors": all_errors,
        "ready_to_present": not all_errors,
        "approval_required": True,
        "approval_granted": False,
        "approval_gate": approval_gate,
        "requested_operations": requested_operations,
        "forbidden_without_approval": forbidden_without_approval,
        "approval_text": approval_text if not all_errors else "",
        "markdown": markdown,
    }


def validate_dynamic_precision_human_approval_response(
    *,
    approval_markdown: Mapping[str, Any],
    response_text: str,
) -> dict[str, Any]:
    """Audit a human response against the exact no-write approval text."""

    errors: list[str] = []
    if not isinstance(approval_markdown, Mapping):
        errors.append("approval_markdown must be a mapping")
        approval_markdown = {}
    if not isinstance(response_text, str):
        errors.append("response_text must be a string")
        response_text = ""

    markdown_errors = [str(error) for error in approval_markdown.get("errors", [])]
    if approval_markdown.get("schema") != "air_dynamic_precision_human_approval_markdown":
        errors.append("approval_markdown.schema must be air_dynamic_precision_human_approval_markdown")
    if approval_markdown.get("schema_version") != 1:
        errors.append("approval_markdown.schema_version must be 1")
    if approval_markdown.get("ok") is not True:
        errors.append("approval_markdown must be ok before response audit")
    if approval_markdown.get("ready_to_present") is not True:
        errors.append("approval_markdown.ready_to_present must be true")
    if approval_markdown.get("approval_required") is not True:
        errors.append("approval_markdown.approval_required must be true")
    if approval_markdown.get("approval_granted") is not False:
        errors.append("approval_markdown.approval_granted must remain false before response audit")

    approval_gate = str(approval_markdown.get("approval_gate", ""))
    if approval_gate != AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE:
        errors.append("approval_markdown.approval_gate mismatch")

    approval_text = str(approval_markdown.get("approval_text", ""))
    response_matches = bool(response_text.strip() and response_text.strip() == approval_text.strip())
    if not response_matches:
        errors.append("approval response must exactly match approval_text")

    all_errors = _dedupe_errors([*markdown_errors, *errors])
    ready = not all_errors and response_matches
    requested_operations = (
        _sequence_strings(approval_markdown.get("requested_operations"))
        if ready
        else []
    )
    return {
        "schema": "air_dynamic_precision_human_approval_response_audit",
        "schema_version": 1,
        "ok": ready,
        "errors": all_errors,
        "approval_required": True,
        "approval_gate": approval_gate,
        "approval_phrase_matches": response_matches,
        "ready_for_operator_to_start_approved_batch": ready,
        "requested_operations": requested_operations,
        "forbidden_without_approval": _sequence_strings(
            approval_markdown.get("forbidden_without_approval")
        ),
        "execution_flags_after_audit": {
            "collection_allowed": False,
            "materialization_allowed": False,
            "evaluation_allowed": False,
            "candidate_acceptance_allowed": False,
        },
        "filesystem": {
            "writes_performed": False,
            "artifact_created": False,
            "candidate_dir_created": False,
            "result_artifact_written": False,
        },
    }


def build_dynamic_precision_materialization_dry_run(
    materialization_plan: Mapping[str, Any],
    *,
    source_dir: str | Path,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Plan dynamic materialization paths without creating or mutating artifacts."""

    source_root = Path(source_dir)
    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    filesystem = {
        "writes_performed": False,
        "candidate_dir_created": False,
        "seed_artifact_mutated": False,
        "candidate_artifact_mutated": False,
        "source_dir": str(source_root),
        "seed_artifact_dir": str(seed_root),
        "output_dir": str(output_root),
        "source_dir_exists": source_root.exists(),
        "seed_artifact_dir_exists": seed_root.exists(),
        "output_dir_exists": output_root.exists(),
    }
    errors: list[str] = []
    if not isinstance(materialization_plan, Mapping):
        errors.append("materialization_plan must be a mapping")
        materialization_plan = {}
    plan_errors = [str(error) for error in materialization_plan.get("errors", [])]
    if materialization_plan.get("schema") != "air_dynamic_precision_materialization_plan":
        errors.append("materialization_plan.schema must be air_dynamic_precision_materialization_plan")
    if materialization_plan.get("schema_version") != 1:
        errors.append("materialization_plan.schema_version must be 1")
    if materialization_plan.get("ok") is not True:
        errors.append("materialization_plan must be ok before dry-run path planning")
    if source_root.resolve(strict=False) == output_root.resolve(strict=False):
        errors.append("output_dir must be separate from source_dir")
    if seed_root.resolve(strict=False) == output_root.resolve(strict=False):
        errors.append("output_dir must be separate from seed_artifact_dir")
    if output_root.exists():
        errors.append("output_dir already exists")

    actions_value = materialization_plan.get("actions", [])
    actions = actions_value if isinstance(actions_value, list) else []
    if not isinstance(actions_value, list):
        errors.append("materialization_plan.actions must be a list")
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        if bool(action.get("requires_imatrix_sidecar")) and not bool(action.get("imatrix_sidecar_available")):
            errors.append(f"{action.get('key')}: missing imatrix sidecar coverage")
        if bool(action.get("requires_high_precision_source")) and not bool(
            action.get("high_precision_source_available")
        ):
            errors.append(f"{action.get('key')}: missing high-precision source coverage")

    planned_outputs: list[dict[str, Any]] = []
    if not errors:
        for action_value in actions:
            if not isinstance(action_value, Mapping):
                errors.append("materialization_plan.actions entries must be mappings")
                planned_outputs = []
                break
            planned_outputs.append(_dry_run_output_for_action(action_value, output_root=output_root))

    candidate = dict(materialization_plan.get("candidate", {})) if isinstance(materialization_plan.get("candidate"), Mapping) else {}
    return {
        "schema": "air_dynamic_precision_materialization_dry_run",
        "schema_version": 1,
        "ok": not errors,
        "errors": plan_errors + errors,
        "mode": "dry_run",
        "materialization_allowed": False,
        "candidate": candidate,
        "filesystem": filesystem,
        "summary": _dry_run_summary(actions),
        "planned_outputs": planned_outputs if not errors else [],
    }


def build_dynamic_precision_materialization_preflight(
    materialization_plan: Mapping[str, Any],
    dry_run: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize the no-write approval gate before real materialization work."""

    errors: list[str] = []
    if not isinstance(materialization_plan, Mapping):
        errors.append("materialization_plan must be a mapping")
        materialization_plan = {}
    if not isinstance(dry_run, Mapping):
        errors.append("dry_run must be a mapping")
        dry_run = {}

    plan_errors = [str(error) for error in materialization_plan.get("errors", [])]
    dry_run_errors = [str(error) for error in dry_run.get("errors", [])]
    if materialization_plan.get("schema") != "air_dynamic_precision_materialization_plan":
        errors.append("materialization_plan.schema must be air_dynamic_precision_materialization_plan")
    if materialization_plan.get("schema_version") != 1:
        errors.append("materialization_plan.schema_version must be 1")
    if materialization_plan.get("ok") is not True:
        errors.append("materialization_plan must be ok before preflight")
    if dry_run.get("schema") != "air_dynamic_precision_materialization_dry_run":
        errors.append("dry_run.schema must be air_dynamic_precision_materialization_dry_run")
    if dry_run.get("schema_version") != 1:
        errors.append("dry_run.schema_version must be 1")
    if dry_run.get("ok") is not True:
        errors.append("dry_run must be ok before preflight")
    if dry_run.get("materialization_allowed") is not False:
        errors.append("dry_run.materialization_allowed must be false")

    candidate = _preflight_mapping(dry_run.get("candidate"), "dry_run.candidate", errors)
    if not candidate:
        candidate = _preflight_mapping(materialization_plan.get("candidate"), "materialization_plan.candidate", errors)
    filesystem = _preflight_mapping(dry_run.get("filesystem"), "dry_run.filesystem", errors)
    summary = _preflight_mapping(dry_run.get("summary"), "dry_run.summary", errors)

    materialization_state = {
        "candidate_materialized": candidate.get("candidate_materialized"),
        "artifact_written": candidate.get("artifact_written"),
        "writes_performed": filesystem.get("writes_performed"),
        "candidate_dir_created": filesystem.get("candidate_dir_created"),
        "seed_artifact_mutated": filesystem.get("seed_artifact_mutated"),
        "candidate_artifact_mutated": filesystem.get("candidate_artifact_mutated"),
    }
    for name, value in materialization_state.items():
        if value is not False:
            errors.append(f"{name} must be false before approval preflight")

    approval_gate = str(candidate.get("approval_gate", AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE))
    if approval_gate != AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE:
        errors.append("candidate.approval_gate mismatch")
    if candidate.get("approval_required") is not True:
        errors.append("candidate.approval_required must be true")

    ready = not errors
    return {
        "schema": "air_dynamic_precision_materialization_preflight",
        "schema_version": 1,
        "ok": ready,
        "errors": plan_errors + dry_run_errors + errors,
        "ready_for_approval_request": ready,
        "materialization_allowed": False,
        "candidate": dict(candidate),
        "materialization_state": materialization_state,
        "summary": dict(summary),
        "approval_request": {
            "approval_required": True,
            "approval_gate": AIR_DYNAMIC_MATERIALIZATION_APPROVAL_GATE,
            "ready": ready,
            "requested_operations": list(AIR_DYNAMIC_MATERIALIZATION_APPROVAL_OPERATIONS),
        },
        "blocked_operations": list(AIR_DYNAMIC_MATERIALIZATION_BLOCKED_OPERATIONS),
    }


def _best_upgrade(
    profiles: list[DynamicTensorProfile],
    tier_by_key: Mapping[str, str],
    tiers: Mapping[str, DynamicPrecisionTier],
    total_bits: float,
    budget_bits: float,
) -> tuple[DynamicTensorProfile, str, float, float, float] | None:
    choices: list[tuple[float, float, str, DynamicTensorProfile, str, float]] = []
    for profile in profiles:
        current_tier = tier_by_key[profile.key]
        current_index = _tier_index(current_tier)
        if current_index >= len(AIR_DYNAMIC_TIER_ORDER) - 1:
            continue
        next_tier = AIR_DYNAMIC_TIER_ORDER[current_index + 1]
        added_bits = (
            tiers[next_tier].effective_bits_per_weight
            - tiers[current_tier].effective_bits_per_weight
        ) * profile.weight_count
        if added_bits <= 0 or total_bits + added_bits > budget_bits + 1.0e-9:
            continue
        gain = float(profile.weighted_errors[current_tier]) - float(profile.weighted_errors[next_tier])
        if gain <= 0:
            continue
        score = gain / added_bits
        choices.append((-score, -gain, profile.key, profile, next_tier, added_bits))
    if not choices:
        return None
    score_key, gain_key, _key, profile, next_tier, added_bits = min(choices)
    gain = -gain_key
    score = -score_key
    return profile, next_tier, added_bits, gain, score


def _total_effective_bits(
    profiles: Iterable[DynamicTensorProfile],
    tier_by_key: Mapping[str, str],
    tiers: Mapping[str, DynamicPrecisionTier],
) -> float:
    return sum(
        profile.weight_count * tiers[tier_by_key[profile.key]].effective_bits_per_weight
        for profile in profiles
    )


def _tier_to_report_row(tier: DynamicPrecisionTier) -> dict[str, Any]:
    return {
        "name": tier.name,
        "artifact_kind": tier.artifact_kind,
        "effective_bits_per_weight": float(tier.effective_bits_per_weight),
        "code_bits": int(tier.code_bits) if tier.code_bits is not None else None,
        "description": tier.description,
    }


def _profile_to_report_row(
    profile: DynamicTensorProfile,
    *,
    selected_tier: str,
    tier: DynamicPrecisionTier,
    prior_floor: DynamicPrecisionPriorFloor | None,
) -> dict[str, Any]:
    weighted_errors = {
        tier_name: float(profile.weighted_errors[tier_name])
        for tier_name in AIR_DYNAMIC_TIER_ORDER
    }
    floor = prior_floor or DynamicPrecisionPriorFloor(
        key=profile.key,
        minimum_tier=AIR_DYNAMIC_TIER_ORDER[0],
        rationale=(),
    )
    return {
        "key": profile.key,
        "layer": int(profile.layer),
        "projection": profile.projection,
        "expert": int(profile.expert) if profile.expert is not None else None,
        "weight_count": int(profile.weight_count),
        "selected_tier": selected_tier,
        "selected_artifact_kind": tier.artifact_kind,
        "selected_code_bits": int(tier.code_bits) if tier.code_bits is not None else None,
        "selected_effective_bits_per_weight": float(tier.effective_bits_per_weight),
        "selected_weighted_error": float(profile.weighted_errors[selected_tier]),
        "weighted_errors": weighted_errors,
        "prior_floor": {
            "minimum_tier": floor.minimum_tier,
            "rationale": list(floor.rationale),
        },
    }


def _json_ready_upgrade(upgrade: Mapping[str, object]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in upgrade.items():
        if isinstance(value, tuple):
            ready[key] = list(value)
        elif isinstance(value, np.generic):
            ready[key] = value.item()
        else:
            ready[key] = value
    return ready


def _candidate_gate(candidate_name: str) -> dict[str, Any]:
    name = str(candidate_name).strip()
    return {
        "name": name,
        "candidate_materialized": False,
        "artifact_written": False,
        "approval_required": True,
        "approval_gate": "real_imatrix_sidecar_collection_and_candidate_materialization",
    }


def _safe_candidate_prefix(candidate_name_prefix: str) -> str:
    raw = str(candidate_name_prefix).strip()
    if not raw:
        return "dynamic-sweep"
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in raw)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "dynamic-sweep"


def _budget_sweep_candidate_name(
    candidate_name_prefix: str,
    *,
    index: int,
    requested_budget_bits_per_weight: float | None,
) -> str:
    if requested_budget_bits_per_weight is None:
        budget_suffix = "unknown"
    else:
        budget_suffix = f"{requested_budget_bits_per_weight:.3f}".replace(".", "p")
    return f"{candidate_name_prefix}-b{index:02d}-bpw-{budget_suffix}"


def _report_tensor_keys(report: object) -> tuple[str, ...] | None:
    if not isinstance(report, Mapping):
        return None
    tier_map = report.get("tier_map")
    if isinstance(tier_map, Mapping):
        return tuple(sorted(str(key) for key in tier_map))
    tensors = report.get("tensors")
    if not isinstance(tensors, list):
        return None
    keys = []
    for row in tensors:
        if not isinstance(row, Mapping) or not isinstance(row.get("key"), str):
            return None
        keys.append(str(row["key"]))
    return tuple(sorted(keys))


def _requested_budget_bits_per_weight(
    budget: Mapping[str, Any],
    errors: list[str],
    index: int,
) -> float | None:
    total_weight_count = _finite_number(budget.get("total_weight_count"))
    budget_effective_bits = _finite_number(budget.get("budget_effective_bits"))
    if total_weight_count is None or total_weight_count <= 0:
        errors.append(f"reports[{index}].budget.total_weight_count must be positive")
        return None
    if budget_effective_bits is None or budget_effective_bits <= 0:
        errors.append(f"reports[{index}].budget.budget_effective_bits must be positive")
        return None
    return budget_effective_bits / total_weight_count


def _budget_sweep_summary(
    budget_sweep: Iterable[Mapping[str, Any]],
    bundles: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in budget_sweep]
    bundle_list = list(bundles)
    requested = [
        float(row["requested_budget_bits_per_weight"])
        for row in rows
        if row.get("requested_budget_bits_per_weight") is not None
    ]
    effective = [
        float(row["effective_bits_per_weight"])
        for row in rows
        if row.get("effective_bits_per_weight") is not None
    ]
    weighted_errors = [
        float(row["weighted_error"])
        for row in rows
        if row.get("weighted_error") is not None
    ]
    return {
        "budget_count": len(rows),
        "ready_bundle_count": sum(1 for bundle in bundle_list if bundle.get("ok") is True),
        "materialization_allowed": False,
        "min_requested_budget_bits_per_weight": min(requested) if requested else None,
        "max_requested_budget_bits_per_weight": max(requested) if requested else None,
        "min_effective_bits_per_weight": min(effective) if effective else None,
        "max_effective_bits_per_weight": max(effective) if effective else None,
        "best_weighted_error": min(weighted_errors) if weighted_errors else None,
        "frontier": [
            {
                "index": int(row["index"]),
                "candidate_name": str(row["candidate_name"]),
                "bundle_ok": bool(row["bundle_ok"]),
                "requested_budget_bits_per_weight": row.get("requested_budget_bits_per_weight"),
                "effective_bits_per_weight": row.get("effective_bits_per_weight"),
                "weighted_error": row.get("weighted_error"),
            }
            for row in rows
        ],
    }


def _eval_cache_plan(
    *,
    split: str,
    prompt_set: str,
    expected_row_count: int,
    path: str | Path,
    baseline_path: str | Path | None,
    primary: bool,
    require_cache_files: bool,
    errors: list[str],
) -> dict[str, Any]:
    cache_path = Path(path)
    baseline = Path(baseline_path) if baseline_path is not None else None
    if require_cache_files and not cache_path.exists():
        errors.append(f"{split} cache path does not exist: {cache_path}")
    if require_cache_files and baseline is not None and not baseline.exists():
        errors.append(f"{split} baseline cache path does not exist: {baseline}")
    return {
        "split": split,
        "prompt_set": prompt_set,
        "expected_row_count": int(expected_row_count),
        "path": str(cache_path),
        "exists": cache_path.exists(),
        "baseline_path": str(baseline) if baseline is not None else None,
        "baseline_exists": baseline.exists() if baseline is not None else None,
        "payload_read": False,
        "primary": primary,
    }


def _candidate_eval_split(cache_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": str(cache_plan["split"]),
        "prompt_set": str(cache_plan["prompt_set"]),
        "cache_path": str(cache_plan["path"]),
        "expected_row_count": int(cache_plan["expected_row_count"]),
        "primary": bool(cache_plan["primary"]),
        "required_metrics": list(AIR_DYNAMIC_EVAL_REQUIRED_METRICS),
        "eval_records_written": False,
    }


def _planned_candidate_output_dir(
    bundles: list[Any],
    index: int,
) -> str | None:
    if index >= len(bundles) or not isinstance(bundles[index], Mapping):
        return None
    dry_run = bundles[index].get("materialization_bundle", {})
    if isinstance(dry_run, Mapping):
        dry_run = dry_run.get("dry_run", {})
    if not isinstance(dry_run, Mapping):
        return None
    filesystem = dry_run.get("filesystem", {})
    if not isinstance(filesystem, Mapping):
        return None
    output_dir = filesystem.get("output_dir")
    return str(output_dir) if output_dir is not None else None


def _materialization_action(
    row: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_kind = str(row["selected_artifact_kind"])
    is_vq = artifact_kind.startswith("vq_")
    action = "vq_reencode_with_imatrix" if is_vq else "materialize_high_precision_projection"
    return {
        "key": str(row["key"]),
        "layer": int(row["layer"]),
        "projection": str(row["projection"]),
        "expert": int(row["expert"]) if row.get("expert") is not None else None,
        "weight_count": int(row["weight_count"]),
        "selected_tier": str(row["selected_tier"]),
        "artifact_kind": artifact_kind,
        "code_bits": int(row["selected_code_bits"]) if row.get("selected_code_bits") is not None else None,
        "materialization_action": action,
        "requires_imatrix_sidecar": is_vq,
        "requires_high_precision_source": not is_vq,
        "selected_weighted_error": float(row["selected_weighted_error"]),
        "approval_required": bool(materialization["approval_required"]),
        "approval_gate": str(materialization["approval_gate"]),
    }


def _imatrix_manifest_entries(
    manifest: Mapping[str, Any],
    errors: list[str],
) -> list[Mapping[str, Any]]:
    if manifest.get("record_type") != "air_projection_imatrix_manifest":
        errors.append("imatrix_manifest.record_type must be air_projection_imatrix_manifest")
    if manifest.get("schema_version") != 1:
        errors.append("imatrix_manifest.schema_version must be 1")
    entries_value = manifest.get("entries")
    if not isinstance(entries_value, list):
        errors.append("imatrix_manifest.entries must be a list")
        return []
    if manifest.get("entry_count") != len(entries_value):
        errors.append("imatrix_manifest.entry_count must match entries length")
    entries: list[Mapping[str, Any]] = []
    for index, entry in enumerate(entries_value):
        if not isinstance(entry, Mapping):
            errors.append(f"imatrix_manifest.entries[{index}] must be a mapping")
            continue
        entries.append(entry)
    return entries


def _index_imatrix_manifest_entries(
    entries: Iterable[Mapping[str, Any]],
    errors: list[str],
    *,
    manifest_root: str | Path | None,
    require_sidecar_files: bool,
) -> dict[tuple[int, str, int], dict[str, Any]]:
    if require_sidecar_files and manifest_root is None:
        errors.append("manifest_root is required when require_sidecar_files is true")
    root = Path(manifest_root) if manifest_root is not None else None
    index: dict[tuple[int, str, int], dict[str, Any]] = {}
    for position, entry in enumerate(entries):
        try:
            layer = int(entry["layer"])
            projection = str(entry["projection"])
            expert = int(entry["expert"])
            path = str(entry["path"])
            input_dim = int(entry["input_dim"])
            route_count = int(entry["route_count"])
            total_route_count = int(entry["total_route_count"])
            route_frequency = float(entry["route_frequency"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid imatrix manifest entry at index {position}: {exc}")
            continue
        if projection not in AIR_ROUTED_PROJECTIONS:
            errors.append(f"invalid imatrix projection {projection!r} at index {position}")
            continue
        if layer < 0 or expert < 0 or input_dim <= 0 or route_count < 0 or total_route_count <= 0:
            errors.append(f"invalid imatrix counts at index {position}")
            continue
        if not np.isfinite(route_frequency) or route_frequency < 0:
            errors.append(f"invalid imatrix route_frequency at index {position}")
            continue
        if require_sidecar_files:
            sidecar_path = Path(path)
            if sidecar_path.is_absolute() or ".." in sidecar_path.parts:
                errors.append(f"imatrix sidecar path must be relative and contained: {path}")
                continue
            if root is not None and not (root / sidecar_path).exists():
                errors.append(f"imatrix sidecar file does not exist: {path}")
                continue
        key = (layer, projection, expert)
        if key in index:
            errors.append(f"duplicate imatrix sidecar entry for layer={layer} projection={projection} expert={expert}")
            continue
        prompt_ids_value = entry.get("prompt_ids", [])
        prompt_ids = [str(prompt_id) for prompt_id in prompt_ids_value] if isinstance(prompt_ids_value, list) else []
        index[key] = {
            "path": path,
            "input_dim": input_dim,
            "route_count": route_count,
            "total_route_count": total_route_count,
            "route_frequency": route_frequency,
            "prompt_ids": prompt_ids,
        }
    return index


def _high_source_manifest_entries(
    manifest: Mapping[str, Any],
    errors: list[str],
) -> list[Mapping[str, Any]]:
    if manifest.get("record_type") != "air_dynamic_precision_high_source_manifest":
        errors.append("source_manifest.record_type must be air_dynamic_precision_high_source_manifest")
    if manifest.get("schema_version") != 1:
        errors.append("source_manifest.schema_version must be 1")
    if not isinstance(manifest.get("source_kind"), str) or not str(manifest.get("source_kind")).strip():
        errors.append("source_manifest.source_kind must be a non-empty string")
    entries_value = manifest.get("entries")
    if not isinstance(entries_value, list):
        errors.append("source_manifest.entries must be a list")
        return []
    if manifest.get("entry_count") != len(entries_value):
        errors.append("source_manifest.entry_count must match entries length")
    entries: list[Mapping[str, Any]] = []
    for index, entry in enumerate(entries_value):
        if not isinstance(entry, Mapping):
            errors.append(f"source_manifest.entries[{index}] must be a mapping")
            continue
        entries.append(entry)
    return entries


def _index_high_source_manifest_entries(
    entries: Iterable[Mapping[str, Any]],
    errors: list[str],
    *,
    source_root: str | Path | None,
    source_kind: str,
    require_source_files: bool,
) -> dict[tuple[int, str, int | None], dict[str, Any]]:
    if require_source_files and source_root is None:
        errors.append("source_root is required when require_source_files is true")
    root = Path(source_root) if source_root is not None else None
    index: dict[tuple[int, str, int | None], dict[str, Any]] = {}
    for position, entry in enumerate(entries):
        try:
            layer = int(entry["layer"])
            projection = str(entry["projection"])
            expert_value = entry.get("expert")
            expert = int(expert_value) if expert_value is not None else None
            path = str(entry["path"])
            tensor_name = str(entry["tensor_name"])
            dtype = str(entry["dtype"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid high-precision source manifest entry at index {position}: {exc}")
            continue
        if projection not in AIR_ROUTED_PROJECTIONS:
            errors.append(f"invalid high-precision source projection {projection!r} at index {position}")
            continue
        if layer < 0 or (expert is not None and expert < 0):
            errors.append(f"invalid high-precision source layer/expert at index {position}")
            continue
        if not tensor_name:
            errors.append(f"high-precision source tensor_name must be non-empty at index {position}")
            continue
        if not dtype:
            errors.append(f"high-precision source dtype must be non-empty at index {position}")
            continue
        shape_value = entry.get("shape")
        if (
            not isinstance(shape_value, list)
            or not shape_value
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape_value)
        ):
            errors.append(f"high-precision source shape must be positive integer list at index {position}")
            continue
        if require_source_files:
            source_path = Path(path)
            if source_path.is_absolute() or ".." in source_path.parts:
                errors.append(f"high-precision source path must be relative and contained: {path}")
                continue
            if root is not None and not (root / source_path).exists():
                errors.append(f"high-precision source file does not exist: {path}")
                continue
        key = (layer, projection, expert)
        if key in index:
            errors.append(
                f"duplicate high-precision source entry for layer={layer} projection={projection} expert={expert}"
            )
            continue
        index[key] = {
            "path": path,
            "tensor_name": tensor_name,
            "dtype": dtype,
            "shape": [int(dim) for dim in shape_value],
            "source_kind": source_kind,
        }
    return index


def _source_index_weight_map(source_index: object, errors: list[str]) -> dict[str, str]:
    if isinstance(source_index, Mapping):
        weight_map_value = source_index.get("weight_map", source_index)
    else:
        weight_map_value = getattr(source_index, "weight_map", None)
    if not isinstance(weight_map_value, Mapping):
        errors.append("source_index must contain a weight_map mapping")
        return {}
    return {str(name): str(shard) for name, shard in weight_map_value.items()}


def _source_tensor_metadata_map(
    metadata: Mapping[str, Mapping[str, Any]] | None,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        errors.append("source_tensor_metadata must be a mapping")
        return {}
    parsed: dict[str, dict[str, Any]] = {}
    for tensor_name, value in metadata.items():
        if not isinstance(value, Mapping):
            errors.append(f"source_tensor_metadata[{tensor_name!r}] must be a mapping")
            continue
        parsed_metadata = _parse_source_tensor_metadata(str(tensor_name), value, errors)
        if parsed_metadata is not None:
            parsed[str(tensor_name)] = parsed_metadata
    return parsed


def _parse_source_tensor_metadata(
    tensor_name: str,
    metadata: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    dtype = str(metadata.get("dtype", "")).strip()
    shape_value = metadata.get("shape")
    if not dtype:
        errors.append(f"source_tensor_metadata[{tensor_name!r}].dtype must be non-empty")
        return None
    if (
        not isinstance(shape_value, (list, tuple))
        or not shape_value
        or any(not isinstance(dim, int) or dim <= 0 for dim in shape_value)
    ):
        errors.append(f"source_tensor_metadata[{tensor_name!r}].shape must be a positive integer list")
        return None
    return {
        "dtype": _normalize_source_dtype(dtype),
        "shape": [int(dim) for dim in shape_value],
    }


def _inspect_source_tensor_header(
    path: Path,
    tensor_name: str,
    errors: list[str],
) -> dict[str, Any] | None:
    from keep.io.source_safetensors import read_safetensors_tensor_header

    try:
        header = read_safetensors_tensor_header(path, tensor_name)
    except (KeyError, OSError, ValueError) as exc:
        errors.append(f"unable to inspect high-precision source header for {tensor_name}: {exc}")
        return None
    return {
        "dtype": _normalize_source_dtype(header.dtype),
        "shape": [int(dim) for dim in header.shape],
    }


def _normalize_source_dtype(dtype: str) -> str:
    return str(dtype).strip().lower()


def _air_expert_source_tensor_name(*, layer: int, projection: str, expert: int) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def _plain_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: object) -> int | None:
    number = _finite_number(value)
    return int(number) if number is not None else None


def _sequence_strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _approval_request_operations(plan: Mapping[str, Any]) -> list[str]:
    request = _plain_mapping(plan.get("approval_request"))
    return _sequence_strings(request.get("requested_operations"))


def _frontier_split_metrics(
    value: object,
    *,
    split: str,
    required_metrics: Iterable[str],
    expected_rows: int,
    top1_min: float,
    errors: list[str],
) -> dict[str, Any]:
    metrics = _plain_mapping(value)
    if not metrics:
        errors.append(f"{split} metrics must be a mapping")
    parsed: dict[str, Any] = {}
    for metric in required_metrics:
        if metric == "lane_s_speed":
            parsed[metric] = metrics.get(metric)
            continue
        number = _finite_number(metrics.get(metric))
        if number is None:
            errors.append(f"{split}.{metric} must be finite")
        parsed[metric] = number

    top1 = parsed.get("mean_top1_agreement")
    clean_rows = parsed.get("clean_row_count")
    parsed["top1_pass"] = bool(top1 is not None and float(top1) >= top1_min)
    parsed["clean_rows_pass"] = bool(
        clean_rows is not None and int(clean_rows) >= int(expected_rows)
    )
    parsed["lane_s_speed_pass"] = _frontier_lane_s_speed_pass(metrics.get("lane_s_speed"))
    return parsed


def _frontier_metric_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean_kld": metrics.get("mean_kld"),
        "p999_kld": metrics.get("p999_kld"),
        "mean_ppl_ratio": metrics.get("mean_ppl_ratio"),
        "max_ppl_ratio": metrics.get("max_ppl_ratio"),
        "mean_top1_agreement": metrics.get("mean_top1_agreement"),
        "clean_row_count": int(metrics["clean_row_count"])
        if metrics.get("clean_row_count") is not None
        else None,
        "effective_bits_per_weight": metrics.get("effective_bits_per_weight"),
        "lane_s_speed_pass": metrics.get("lane_s_speed_pass"),
    }


def _frontier_lane_s_speed_pass(value: object) -> bool:
    if isinstance(value, Mapping):
        return value.get("ok") is True
    if isinstance(value, bool):
        return value
    number = _finite_number(value)
    return bool(number is not None and number > 0.0)


def _bundle_errors(*parts: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for error in part.get("errors", []):
            text = str(error)
            if text not in seen:
                errors.append(text)
                seen.add(text)
    return errors


def _dedupe_errors(errors: Iterable[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for error in errors:
        text = str(error)
        if text not in seen:
            deduped.append(text)
            seen.add(text)
    return deduped


def _materialization_plan_summary(materialization_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": str(materialization_plan.get("schema", "")),
        "schema_version": int(materialization_plan.get("schema_version", 0) or 0),
        "ok": bool(materialization_plan.get("ok")),
        "errors": [str(error) for error in materialization_plan.get("errors", [])],
        "candidate": dict(materialization_plan.get("candidate", {}))
        if isinstance(materialization_plan.get("candidate"), Mapping)
        else {},
        "summary": dict(materialization_plan.get("summary", {}))
        if isinstance(materialization_plan.get("summary"), Mapping)
        else {},
    }


def _dry_run_summary(actions: Iterable[Any]) -> dict[str, int]:
    action_list = [action for action in actions if isinstance(action, Mapping)]
    required_imatrix = [
        action for action in action_list if bool(action.get("requires_imatrix_sidecar"))
    ]
    required_high_source = [
        action for action in action_list if bool(action.get("requires_high_precision_source"))
    ]
    return {
        "planned_action_count": len(action_list),
        "planned_vq_reencode_count": sum(
            1
            for action in action_list
            if action.get("materialization_action") == "vq_reencode_with_imatrix"
        ),
        "planned_high_precision_count": sum(
            1
            for action in action_list
            if action.get("materialization_action") == "materialize_high_precision_projection"
        ),
        "required_imatrix_sidecar_count": len(required_imatrix),
        "available_imatrix_sidecar_count": sum(
            1 for action in required_imatrix if bool(action.get("imatrix_sidecar_available"))
        ),
        "required_high_precision_source_count": len(required_high_source),
        "available_high_precision_source_count": sum(
            1 for action in required_high_source if bool(action.get("high_precision_source_available"))
        ),
    }


def _dry_run_output_for_action(action: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    layer = int(action["layer"])
    projection = str(action["projection"])
    materialization_action = str(action["materialization_action"])
    filename = (
        f"layer-{layer:05d}-{projection}.safetensors"
        if materialization_action == "vq_reencode_with_imatrix"
        else f"layer-{layer:05d}-{projection}-high.safetensors"
    )
    sidecar = action.get("imatrix_sidecar")
    sidecar_path = sidecar.get("path") if isinstance(sidecar, Mapping) else None
    high_source = action.get("high_precision_source")
    high_source_path = high_source.get("path") if isinstance(high_source, Mapping) else None
    high_source_tensor = high_source.get("tensor_name") if isinstance(high_source, Mapping) else None
    source = (
        "seed_artifact_plus_imatrix_sidecar"
        if materialization_action == "vq_reencode_with_imatrix"
        else "high_precision_source"
    )
    return {
        "key": str(action["key"]),
        "layer": layer,
        "projection": projection,
        "expert": int(action["expert"]) if action.get("expert") is not None else None,
        "materialization_action": materialization_action,
        "would_write": True,
        "output_path": str(output_root / filename),
        "artifact_kind": str(action["artifact_kind"]),
        "code_bits": int(action["code_bits"]) if action.get("code_bits") is not None else None,
        "source": source,
        "imatrix_sidecar_path": str(sidecar_path) if sidecar_path is not None else None,
        "high_precision_source_path": str(high_source_path) if high_source_path is not None else None,
        "high_precision_source_tensor": str(high_source_tensor) if high_source_tensor is not None else None,
        "requires_high_precision_source": bool(action.get("requires_high_precision_source")),
    }


def _preflight_mapping(value: object, name: str, errors: list[str]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    errors.append(f"{name} must be a mapping")
    return {}


def _mapping_value(report: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any] | None:
    value = report.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be a mapping")
        return None
    return value


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 and float(number) == float(value) else None


def _expect_close(
    actual: object,
    expected: float,
    errors: list[str],
    name: str,
    *,
    atol: float = 1.0e-6,
) -> None:
    actual_number = _finite_number(actual)
    if actual_number is None or abs(actual_number - float(expected)) > atol:
        errors.append(f"{name} mismatch; expected {expected}")


def _validate_tiers(tiers: Mapping[str, DynamicPrecisionTier]) -> None:
    if tuple(tiers) != AIR_DYNAMIC_TIER_ORDER:
        raise ValueError(f"tiers must be ordered as {AIR_DYNAMIC_TIER_ORDER}")
    previous_bits = -1.0
    for name in AIR_DYNAMIC_TIER_ORDER:
        tier = tiers.get(name)
        if tier is None or tier.name != name:
            raise ValueError(f"missing tier {name!r}")
        if tier.effective_bits_per_weight <= previous_bits:
            raise ValueError("tier effective_bits_per_weight must increase monotonically")
        previous_bits = tier.effective_bits_per_weight


def _validate_profile(profile: DynamicTensorProfile, tiers: Mapping[str, DynamicPrecisionTier]) -> None:
    _validate_profile_shape(profile)
    if profile.weight_count <= 0:
        raise ValueError(f"{profile.key}: weight_count must be positive")
    for tier in tiers:
        if tier not in profile.weighted_errors:
            raise ValueError(f"{profile.key}: missing weighted error for tier {tier!r}")
        value = float(profile.weighted_errors[tier])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{profile.key}: weighted errors must be finite and non-negative")


def _validate_profile_shape(profile: DynamicTensorProfile) -> None:
    if profile.layer < 0:
        raise ValueError(f"{profile.key}: layer must be non-negative")
    if profile.projection not in AIR_ROUTED_PROJECTIONS:
        raise ValueError(f"{profile.key}: unsupported projection {profile.projection!r}")


def _validate_tier_name(name: str, tiers: Mapping[str, DynamicPrecisionTier]) -> None:
    if name not in tiers:
        raise ValueError(f"unknown tier {name!r}")


def _tier_index(name: str) -> int:
    return AIR_DYNAMIC_TIER_ORDER.index(name)


def _max_tier(left: str, right: str) -> str:
    return left if _tier_index(left) >= _tier_index(right) else right


__all__ = [
    "AIR_DYNAMIC_TIER_ORDER",
    "AIR_FRAGILE_GLU_LAYERS",
    "DynamicPrecisionPlan",
    "DynamicPrecisionPriorFloor",
    "DynamicPrecisionTier",
    "DynamicTensorProfile",
    "allocate_dynamic_precision_tiers",
    "attach_high_precision_source_manifest_to_materialization_plan",
    "attach_imatrix_manifest_to_materialization_plan",
    "build_dynamic_precision_approval_batch_plan",
    "build_dynamic_precision_budget_sweep_approval_bundle",
    "build_dynamic_precision_eval_frontier_contract",
    "build_dynamic_precision_eval_frontier_report",
    "build_dynamic_precision_human_approval_markdown",
    "build_dynamic_precision_human_approval_request",
    "build_dynamic_precision_materialized_eval_plan",
    "build_high_precision_source_manifest_from_index",
    "build_dynamic_precision_materialization_approval_bundle",
    "build_dynamic_precision_materialization_dry_run",
    "build_dynamic_precision_materialization_preflight",
    "build_dynamic_precision_materialization_plan",
    "build_dynamic_precision_tier_map_approval_bundle",
    "build_dynamic_precision_tier_map_report",
    "build_air_dynamic_precision_prior_summary",
    "build_air_layer_type_prior_floors",
    "default_air_dynamic_precision_tiers",
    "imatrix_weighted_squared_error",
    "validate_dynamic_precision_approval_batch_objective",
    "validate_dynamic_precision_human_approval_response",
    "validate_dynamic_precision_tier_map_report",
]
