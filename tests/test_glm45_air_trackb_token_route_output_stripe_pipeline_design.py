from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "design_glm45_air_trackb_token_route_output_stripe_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "trackb_token_route_output_stripe_pipeline_design_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_token_route_output_stripe_pipeline_rejected_after_selector_learning(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    kblock_speed = _write_json(
        tmp_path / "kblock-speed.json",
        {
            "decision": "reject_kblock_wavefront_codeword_scan_speed_path",
            "same_window_q2_speed_packet": True,
            "worst_ratio": {
                "ratio_to_q2": 1570.7417385984265,
                "candidate_output_shape": [8192, 4096],
                "q2_output_shape": [1024, 8, 4096],
            },
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "speed_claim": False,
            "lane_s_claim": False,
        },
    )

    report = cli.build_design_report(kblock_speed_packet_path=kblock_speed)

    assert report["record_type"] == (
        "glm45_air_e8p_trackb_token_route_output_stripe_pipeline_design"
    )
    assert report["decision"] == "token_route_output_stripe_pipeline_design_incomplete"
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert report["speed_claim"] is False
    assert report["native_parity_claim"] is False
    assert report["artifact_parity_claim"] is False
    assert report["kblock_speed_packet_decision"] == (
        "reject_kblock_wavefront_codeword_scan_speed_path"
    )
    assert report["candidate"]["target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_route_output_stripe_pipeline"
    )
    assert report["candidate"]["storage_constraint"] == (
        "compressed_e8p_token_route_output_stripe_pipelines"
    )
    assert report["candidate"]["dispatch_grid"] == (
        "tokens_x_route_slots_x_output_stripes_x_kblock_stages"
    )
    assert report["candidate"]["decoded_dense_weight_bytes"] == 0
    assert report["selector_verdict"]["decision"] == (
        "reject_token_route_output_stripe_pipeline_speed_path"
    )
    assert report["selector_verdict"]["matched_rejected_families"] == [
        "token_route_output_stripe_pipeline"
    ]
    assert report["selector_verdict"]["candidate_ready_for_source_probe"] is False
    assert "avoid_token_route_output_stripe_pipeline_schedule" in (
        report["selector_verdict"]["required_next_features"]
    )
    assert "preserves_token_route_slot_output_contract" in (
        report["material_difference_from_rejected_families"]
    )
    assert "source_structure_guardrail_for_token_route_output_stripe_pipeline" in (
        report["required_before_benchmark"]
    )
    assert report["next_track_b_action"] == (
        "design_materially_new_trackb_rhs_or_kernel_family"
    )


def test_token_route_output_stripe_pipeline_requires_kblock_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    kblock_speed = _write_json(
        tmp_path / "kblock-speed.json",
        {
            "decision": "kblock_wavefront_codeword_scan_speed_packet_lane_s_pass",
            "same_window_q2_speed_packet": True,
        },
    )

    report = cli.build_design_report(kblock_speed_packet_path=kblock_speed)

    assert report["decision"] == (
        "token_route_output_stripe_pipeline_design_incomplete"
    )
    assert report["selector_verdict"]["candidate_ready_for_source_probe"] is False
