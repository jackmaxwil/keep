from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EvidenceSpec:
    key: str
    record_type: str
    decision_path: tuple[str, ...]
    expected_decision: str
    family_group: str


DEFAULT_EVIDENCE_PATHS = {
    "current_family": Path(
        "artifacts/quality/glm45-air-e8p-trackb-current-family-component-inclusive-20260703.json"
    ),
    "route_codeword": Path(
        "artifacts/quality/glm45-air-e8p-route-codeword-projection-feasibility-20260703.json"
    ),
    "route_active_codeword": Path(
        "artifacts/quality/glm45-air-e8p-route-active-codeword-projection-feasibility-20260703.json"
    ),
    "route_abs_index": Path(
        "artifacts/quality/glm45-air-e8p-route-abs-index-projection-feasibility-20260703.json"
    ),
    "route_abs_component": Path(
        "artifacts/quality/glm45-air-e8p-route-abs-component-projection-feasibility-20260703.json"
    ),
    "route_abs_component_tile_sweep": Path(
        "artifacts/quality/glm45-air-e8p-route-abs-component-tile-sweep-20260703.json"
    ),
    "route_abs_component_token_reuse": Path(
        "artifacts/quality/glm45-air-e8p-route-abs-component-token-reuse-feasibility-20260703.json"
    ),
    "route_abs_component_token_active_expert": Path(
        "artifacts/quality/glm45-air-e8p-route-abs-component-token-active-expert-feasibility-20260703.json"
    ),
    "route_abs_component_lower_bound": Path(
        "artifacts/quality/glm45-air-e8p-route-abs-component-lower-bound-20260703.json"
    ),
    "component_shared_decode": Path(
        "artifacts/quality/glm45-air-e8p-component-stream-shared-decode-feasibility-20260703.json"
    ),
    "expert_kblock_predecode": Path(
        "artifacts/quality/glm45-air-e8p-expert-kblock-decode-reuse-feasibility-20260703.json"
    ),
}

EVIDENCE_SPECS = (
    EvidenceSpec(
        key="current_family",
        record_type="glm45_air_e8p_kernel_structure_analysis",
        decision_path=("track_b_current_family_verdict", "decision"),
        expected_decision="current_track_b_family_not_benchmarkable",
        family_group="current_source_families",
    ),
    EvidenceSpec(
        key="route_codeword",
        record_type="glm45_air_e8p_route_codeword_projection_feasibility",
        decision_path=("feasibility", "decision"),
        expected_decision="reject_route_codeword_projection_cache_too_large",
        family_group="route_projection",
    ),
    EvidenceSpec(
        key="route_active_codeword",
        record_type="glm45_air_e8p_route_active_codeword_projection_feasibility",
        decision_path=("feasibility", "decision"),
        expected_decision="reject_route_active_codeword_projection_cache_too_large",
        family_group="route_projection",
    ),
    EvidenceSpec(
        key="route_abs_index",
        record_type="glm45_air_e8p_route_abs_index_projection_feasibility",
        decision_path=("feasibility", "decision"),
        expected_decision="reject_route_abs_index_projection_requires_sign_axis",
        family_group="route_projection",
    ),
    EvidenceSpec(
        key="route_abs_component",
        record_type="glm45_air_e8p_route_abs_component_projection_feasibility",
        decision_path=("feasibility", "decision"),
        expected_decision="reject_route_abs_component_projection_cache_too_large",
        family_group="route_projection",
    ),
    EvidenceSpec(
        key="route_abs_component_tile_sweep",
        record_type="glm45_air_e8p_route_abs_component_tile_sweep",
        decision_path=("summary", "frontier_decision"),
        expected_decision="reject_route_abs_component_tile_cache_missing_route_slot_axis",
        family_group="route_projection",
    ),
    EvidenceSpec(
        key="route_abs_component_token_reuse",
        record_type="glm45_air_e8p_route_abs_component_token_reuse_feasibility",
        decision_path=("feasibility", "decision"),
        expected_decision="reject_route_abs_component_token_reuse_cache_too_large",
        family_group="route_projection",
    ),
    EvidenceSpec(
        key="route_abs_component_token_active_expert",
        record_type="glm45_air_e8p_route_abs_component_token_active_expert_feasibility",
        decision_path=("feasibility", "decision"),
        expected_decision="reject_route_abs_component_token_active_expert_cache_too_large",
        family_group="route_projection",
    ),
    EvidenceSpec(
        key="route_abs_component_lower_bound",
        record_type="glm45_air_e8p_route_abs_component_lower_bound",
        decision_path=("feasibility", "decision"),
        expected_decision="reject_exact_route_abs_component_projection_lower_bound_over_cap",
        family_group="route_projection",
    ),
    EvidenceSpec(
        key="component_shared_decode",
        record_type="glm45_air_e8p_component_stream_shared_decode_feasibility",
        decision_path=("feasibility", "decision"),
        expected_decision="reject_component_stream_shared_decode_cross_ntile_cache",
        family_group="decode_reuse",
    ),
    EvidenceSpec(
        key="expert_kblock_predecode",
        record_type="glm45_air_e8p_expert_kblock_decode_reuse_feasibility",
        decision_path=("feasibility", "decision"),
        expected_decision="reject_expert_kblock_predecode_cache_decoded_rhs_equivalent",
        family_group="decode_reuse",
    ),
)

NEXT_REQUIRED_FEATURES = [
    "materially_different_rhs_layout_or_kernel_family",
    "bounded_cardinality_mechanism_below_cache_cap",
    "preserve_compressed_rhs_storage",
    "preserve_codeword_scale_slots",
    "avoid_decoded_rhs_equivalent_cache",
    "preserve_e8p_sign_axis_for_route_projection",
    "preserve_route_slot_axis_or_prove_equivalent_rowwise_reconstruction",
    "bounded_rowwise_cardinality_mechanism_below_cache_cap",
    "exact_token_reuse_still_requires_new_bounded_mechanism",
    "active_expert_token_reuse_still_requires_stronger_bounded_mechanism",
    "exact_abs_component_lower_bound_requires_new_rhs_or_kernel_family",
]

REJECTED_NEXT_STEPS = [
    "do_not_time_current_e8p16_families",
    "do_not_benchmark_route_codeword_projection_cache",
    "do_not_benchmark_route_active_codeword_projection_cache",
    "do_not_benchmark_route_abs_index_projection_cache",
    "do_not_benchmark_route_abs_component_projection_cache",
    "do_not_benchmark_route_abs_component_tile_cache_without_route_slot_axis",
    "do_not_benchmark_token_reuse_abs_component_cache",
    "do_not_benchmark_token_active_expert_abs_component_cache",
    "do_not_benchmark_exact_route_abs_component_projection_family",
    "do_not_claim_exactness_from_route_tile_level_cache_bytes",
    "do_not_claim_speed_from_token_reuse_cardinality_probe",
    "do_not_claim_speed_from_token_active_expert_cardinality_probe",
    "do_not_claim_speed_from_lower_bound_cardinality_probe",
    "do_not_benchmark_scalar_shared_decode_cache_scaffold",
    "do_not_benchmark_expert_kblock_predecode_cache",
    "do_not_claim_speed_from_structure_or_cardinality_probe",
]


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _get_path(data: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _transport_claims(record: Mapping[str, Any]) -> dict[str, Any]:
    claims: dict[str, Any] = {}
    for key in ("peer2_used", "rdma_jaccl_touched", "speed_claim", "native_parity_claim"):
        if key in record and record[key] is not None:
            claims[key] = record[key]
    return claims


def build_summary(evidence_paths: Mapping[str, Path]) -> dict[str, Any]:
    missing_evidence: list[str] = []
    mismatched_evidence: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    rejected_families: list[dict[str, Any]] = []
    transport_violations: list[dict[str, Any]] = []

    for spec in EVIDENCE_SPECS:
        path = evidence_paths[spec.key]
        if not path.exists():
            missing_evidence.append(spec.key)
            continue

        record = _load_json(path)
        record_type = record.get("record_type")
        decision = _get_path(record, spec.decision_path)
        claims = _transport_claims(record)
        for claim_key in ("peer2_used", "rdma_jaccl_touched", "speed_claim"):
            if claim_key in claims and claims[claim_key] is not False:
                transport_violations.append(
                    {"evidence": spec.key, "claim": claim_key, "value": claims[claim_key]}
                )
        if "native_parity_claim" in claims and claims["native_parity_claim"] is not False:
            transport_violations.append(
                {
                    "evidence": spec.key,
                    "claim": "native_parity_claim",
                    "value": claims["native_parity_claim"],
                }
            )

        if record_type != spec.record_type or decision != spec.expected_decision:
            mismatched_evidence.append(
                {
                    "evidence": spec.key,
                    "expected_record_type": spec.record_type,
                    "actual_record_type": record_type,
                    "decision_path": list(spec.decision_path),
                    "expected_decision": spec.expected_decision,
                    "actual_decision": decision,
                }
            )

        rejected_families.append(
            {
                "evidence": spec.key,
                "family_group": spec.family_group,
                "decision": decision,
                "path": str(path),
            }
        )
        evidence_records.append(
            {
                "evidence": spec.key,
                "record_type": record_type,
                "decision": decision,
                "transport_claims": claims,
                "path": str(path),
            }
        )

    complete = not missing_evidence and not mismatched_evidence and not transport_violations
    return {
        "schema_version": 1,
        "record_type": "glm45_air_e8p_trackb_frontier_summary",
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "speed_claim": False,
        "native_parity_claim": False,
        "ready_for_native_speed_work": False,
        "frontier_decision": (
            "track_b_current_frontier_requires_new_rhs_or_kernel_family"
            if complete
            else "track_b_frontier_summary_incomplete"
        ),
        "all_required_evidence_present": not missing_evidence,
        "all_required_evidence_matches_expected_decisions": not mismatched_evidence,
        "all_transport_claims_local_and_non_speed": not transport_violations,
        "missing_evidence": missing_evidence,
        "mismatched_evidence": mismatched_evidence,
        "transport_violations": transport_violations,
        "rejected_families": rejected_families,
        "next_required_features": NEXT_REQUIRED_FEATURES,
        "rejected_next_steps": REJECTED_NEXT_STEPS,
        "evidence": evidence_records,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the current GLM-4.5-Air E8P Track B frontier."
    )
    for key, default in DEFAULT_EVIDENCE_PATHS.items():
        parser.add_argument(f"--{key.replace('_', '-')}-json", type=Path, default=default)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evidence_paths = {
        key: getattr(args, f"{key}_json") for key in DEFAULT_EVIDENCE_PATHS
    }
    report = build_summary(evidence_paths)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
