from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlx_vq.kernels.e8p_rhs_layout import (
    E8PNextKernelFamilyCandidate,
    evaluate_e8p_next_kernel_family_candidate,
)


DEFAULT_KBLOCK_SPEED_PACKET_PATH = Path(
    "artifacts/quality/glm45-air-e8p-kblock-wavefront-codeword-scan-speed-packet-20260704.json"
)


def build_token_route_output_stripe_pipeline_candidate() -> E8PNextKernelFamilyCandidate:
    return E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_token_route_output_stripe_pipeline"
        ),
        storage_constraint="compressed_e8p_token_route_output_stripe_pipelines",
        dispatch_grid="tokens_x_route_slots_x_output_stripes_x_kblock_stages",
        decode_reuse_scope=(
            "token_route_output_stripes_accumulate_full_output_without_route_expansion"
        ),
        tensorops_rhs_source=(
            "compressed_e8p_codeword_tiles_streamed_by_token_route_output_stripe"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def build_design_report(*, kblock_speed_packet_path: Path) -> dict[str, Any]:
    kblock_speed = _load_json(kblock_speed_packet_path)
    candidate = build_token_route_output_stripe_pipeline_candidate()
    selector_verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    kblock_speed_rejected = (
        kblock_speed.get("decision") == "reject_kblock_wavefront_codeword_scan_speed_path"
        and kblock_speed.get("same_window_q2_speed_packet") is True
    )
    selector_ready = selector_verdict.get("candidate_ready_for_source_probe") is True
    decision = (
        "token_route_output_stripe_pipeline_ready_for_source_structure_probe"
        if kblock_speed_rejected and selector_ready
        else "token_route_output_stripe_pipeline_design_incomplete"
    )

    return {
        "schema_version": 1,
        "record_type": (
            "glm45_air_e8p_trackb_token_route_output_stripe_pipeline_design"
        ),
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "speed_claim": False,
        "native_parity_claim": False,
        "artifact_parity_claim": False,
        "kblock_speed_packet_artifact": str(kblock_speed_packet_path),
        "kblock_speed_packet_decision": kblock_speed.get("decision"),
        "kblock_same_window_q2_speed_packet": bool(
            kblock_speed.get("same_window_q2_speed_packet")
        ),
        "kblock_worst_ratio": kblock_speed.get("worst_ratio"),
        "decision": decision,
        "candidate": {
            "target_kernel_family": candidate.target_kernel_family,
            "storage_constraint": candidate.storage_constraint,
            "dispatch_grid": candidate.dispatch_grid,
            "decode_reuse_scope": candidate.decode_reuse_scope,
            "tensorops_rhs_source": candidate.tensorops_rhs_source,
            "preserves_compressed_rhs_storage": candidate.preserves_compressed_rhs_storage,
            "preserves_codeword_scale_slots": candidate.preserves_codeword_scale_slots,
            "decoded_dense_weight_bytes": candidate.decoded_dense_weight_bytes,
            "reconstructs_e8p_values_per_fragment": (
                candidate.reconstructs_e8p_values_per_fragment
            ),
            "uses_decoded_b_threadgroup_staging": (
                candidate.uses_decoded_b_threadgroup_staging
            ),
            "uses_lane_local_fragment_buffer": candidate.uses_lane_local_fragment_buffer,
        },
        "selector_verdict": selector_verdict,
        "rejected_family_inputs": [
            {
                "family": "kblock_wavefront_codeword_scan",
                "artifact": str(kblock_speed_packet_path),
                "decision": kblock_speed.get("decision"),
            }
        ],
        "material_difference_from_rejected_families": [
            "token_route_slot_major_dispatch_not_kblock_route_microtile_major",
            "preserves_token_route_slot_output_contract",
            "accumulates_output_stripes_without_expanding_route_microtiles",
            "streams_kblock_stages_inside_token_route_output_stripes",
            "keeps_compressed_rhs_codeword_tiles_and_scale_slots",
            "does_not_materialize_decoded_rhs_or_decoded_b_threadgroup_cache",
            "does_not_use_lane_local_fragment_buffer",
        ],
        "design_constraints": [
            "preserve_compressed_rhs_storage_and_codeword_scale_slots",
            "preserve_full_e8p_codeword_sign_axis",
            "preserve_token_route_slot_output_contract",
            "avoid_route_microtile_output_expansion",
            "make_output_stripe_accumulation_the_visible_contract",
            "stream_kblock_stages_inside_each_token_route_output_stripe",
            "avoid_decoded_rhs_equivalent_cache",
            "avoid_threadgroup_decoded_b_staging",
            "avoid_lane_local_fragment_buffer",
            "prove_source_structure_before_native_or_speed_work",
        ],
        "required_before_benchmark": [
            "source_structure_guardrail_for_token_route_output_stripe_pipeline",
            "native_token_route_output_stripe_pipeline_parity",
            "air_down_group_size_352_artifact_parity",
            "same_window_q2_speed_packet_after_parity_only",
        ],
        "rejected_next_steps": [
            "do_not_retime_kblock_wavefront_codeword_scan",
            "do_not_claim_speed_from_design_artifact",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
            "do_not_set_custom_memory_limit_override_for_this_design_slice",
            "do_not_run_transport_or_peer_scripts_for_this_design_slice",
        ],
        "next_track_b_action": (
            "implement_source_structure_guardrail_for_token_route_output_stripe_pipeline"
            if decision
            == "token_route_output_stripe_pipeline_ready_for_source_structure_probe"
            else "design_materially_new_trackb_rhs_or_kernel_family"
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Design-gate the post-k-block-rejection Track B "
            "token-route/output-stripe pipeline family."
        )
    )
    parser.add_argument(
        "--kblock-speed-packet-json",
        type=Path,
        default=DEFAULT_KBLOCK_SPEED_PACKET_PATH,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_design_report(kblock_speed_packet_path=args.kblock_speed_packet_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
