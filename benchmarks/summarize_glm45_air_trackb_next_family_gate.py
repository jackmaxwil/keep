from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_FRONTIER_PATH = Path(
    "artifacts/quality/glm45-air-e8p-trackb-frontier-summary-20260703.json"
)
DEFAULT_PLAN_PATH = Path("docs/plans/TRACK_B_E8P16_NAX_KERNEL_PLAN.md")

STALE_WORKING_NAMES = (
    "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2",
)

STALE_PLAN_REJECTIONS = [
    "do_not_implement_nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_unchanged",
    "do_not_benchmark_near_duplicate_expert_kblock_v2",
    "do_not_retime_current_expert_kblock_v2_plan_route",
]

STALE_PLAN_REQUIRED_FEATURES = [
    "supersede_stale_expert_kblock_v2_plan_route",
    "name_materially_new_rhs_storage_or_kernel_family_before_source_work",
    "prove_new_family_is_not_a_current_frontier_rejected_family",
]


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def _transport_violations(frontier: Mapping[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for key in ("peer2_used", "rdma_jaccl_touched", "speed_claim", "native_parity_claim"):
        value = frontier.get(key)
        if value is not False:
            violations.append({"claim": key, "value": value})
    return violations


def build_next_family_gate(frontier_path: Path, plan_path: Path) -> dict[str, Any]:
    frontier = _load_json(frontier_path)
    plan_text = plan_path.read_text(encoding="utf-8")

    frontier_decision = frontier.get("frontier_decision")
    rejected_families = frontier.get("rejected_families", [])
    if not isinstance(rejected_families, list):
        rejected_families = []

    blocked_working_names = [
        name for name in STALE_WORKING_NAMES if name in plan_text
    ]
    stale_plan_variant_blocked = bool(blocked_working_names)
    requires_new_family = (
        frontier_decision == "track_b_current_frontier_requires_new_rhs_or_kernel_family"
    )
    transport_violations = _transport_violations(frontier)

    required_next_features = list(frontier.get("next_required_features") or [])
    rejected_next_steps = list(frontier.get("rejected_next_steps") or [])
    if stale_plan_variant_blocked:
        required_next_features.extend(STALE_PLAN_REQUIRED_FEATURES)
        rejected_next_steps.extend(STALE_PLAN_REJECTIONS)

    if transport_violations:
        decision = "track_b_next_family_gate_incomplete_transport_claims_present"
    elif not requires_new_family:
        decision = "track_b_next_family_gate_incomplete_frontier_not_current"
    elif stale_plan_variant_blocked:
        decision = "block_stale_trackb_plan_variant_requires_materially_new_family_design"
    else:
        decision = "track_b_next_family_design_required"

    return {
        "schema_version": 1,
        "record_type": "glm45_air_e8p_trackb_next_family_gate",
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "speed_claim": False,
        "native_parity_claim": False,
        "frontier_artifact": str(frontier_path),
        "plan_path": str(plan_path),
        "frontier_decision": frontier_decision,
        "rejected_family_count": len(rejected_families),
        "blocked_working_names": blocked_working_names,
        "stale_plan_variant_blocked": stale_plan_variant_blocked,
        "transport_violations": transport_violations,
        "decision": decision,
        "next_track_b_action": (
            "write_materially_new_rhs_or_kernel_family_design_before_source_or_speed_work"
        ),
        "required_next_features": _unique([str(value) for value in required_next_features]),
        "rejected_next_steps": _unique([str(value) for value in rejected_next_steps]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate the next Track B slice against the current rejected frontier."
    )
    parser.add_argument("--frontier-json", type=Path, default=DEFAULT_FRONTIER_PATH)
    parser.add_argument("--plan-md", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_next_family_gate(args.frontier_json, args.plan_md)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
