from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "summarize_keep_five_front_status.py"
    spec = importlib.util.spec_from_file_location("keep_five_front_status_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_qwen_status_report(tmp_path: Path, qwen_payload: dict) -> dict:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "rejected_families": [],
        },
    )
    qwen = _write_json(tmp_path / "qwen.json", qwen_payload)
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": True, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    return cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_route_microtile_codeword_block_reduce_native_parity_path=None,
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=None,
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=None,
        trackb_kblock_wavefront_codeword_scan_design_path=None,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=None,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=None,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )


def test_five_front_status_default_single_host_cache_attempt_uses_latest_prefix() -> None:
    cli = _load_cli()

    assert cli.DEFAULT_SINGLE_HOST_CACHE_ATTEMPT == Path(
        "artifacts/quality/"
        "glm45-air-cleanroom-single-host-local-sequential-prefix-split23-lower2-4-6-8-10-12-14-16-18-20-22-upper25-27-29-31-33-35-37-39-41-43-45-stageviews-headproc-systemlimits-20260704.json"
    )
    assert cli.DEFAULT_WORKSTREAM1 == Path(
        "artifacts/quality/"
        "glm45-air-workstream1-free-levers-fullsplit-verdict-20260703.json"
    )
    assert cli.DEFAULT_WORKSTREAM1_FIXED_BUMP == Path(
        "artifacts/quality/"
        "glm45-air-workstream1-bump-fixed-loader-verdict-20260704.json"
    )
    assert cli.DEFAULT_LAYER_SPLIT_RECOMMENDATION == Path(
        "artifacts/quality/"
        "glm45-air-rank0-logits-layer-split-recommendation-20260704.json"
    )
    assert cli.DEFAULT_RDMA_TOPOLOGY_AUDIT == Path(
        "artifacts/quality/"
        "glm45-air-rdma-topology-audit-20260704-goal-continuation5-readonly.json"
    )
    assert cli.DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_DESIGN == Path(
        "artifacts/quality/"
        "glm45-air-e8p-kblock-wavefront-codeword-scan-design-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_NATIVE_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-kblock-wavefront-codeword-scan-native-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_ARTIFACT_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-kblock-wavefront-codeword-scan-artifact-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_SPEED_PACKET == Path(
        "artifacts/quality/"
        "glm45-air-e8p-kblock-wavefront-codeword-scan-speed-packet-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_DESIGN == Path(
        "artifacts/quality/"
        "glm45-air-e8p-token-route-output-stripe-pipeline-design-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_NATIVE_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-token-route-output-stripe-pipeline-native-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_ARTIFACT_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-token-route-output-stripe-pipeline-artifact-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_DESIGN == Path(
        "artifacts/quality/"
        "glm45-air-e8p-expert-kblock-scale-slot-stream-design-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_NATIVE_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-expert-kblock-scale-slot-stream-native-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_ARTIFACT_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-expert-kblock-scale-slot-stream-artifact-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_SPEED_PACKET == Path(
        "artifacts/quality/"
        "glm45-air-e8p-expert-kblock-scale-slot-stream-speed-packet-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_DESIGN == Path(
        "artifacts/quality/"
        "glm45-air-e8p-scale-group-route-block-reduce-design-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_NATIVE_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-scale-group-route-block-reduce-native-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_ARTIFACT_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-scale-group-route-block-reduce-artifact-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_SPEED_PACKET == Path(
        "artifacts/quality/"
        "glm45-air-e8p-scale-group-route-block-reduce-speed-packet-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_DESIGN == Path(
        "artifacts/quality/"
        "glm45-air-e8p-route-block-output-group-stream-design-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_NATIVE_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-route-block-output-group-stream-native-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY == Path(
        "artifacts/quality/"
        "glm45-air-e8p-route-block-output-group-stream-artifact-parity-20260704.json"
    )
    assert cli.DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_SPEED_PACKET == Path(
        "artifacts/quality/"
        "glm45-air-e8p-route-block-output-group-stream-speed-packet-20260704.json"
    )
    assert (
        cli.DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_NATIVE_PARITY
        == Path(
            "artifacts/quality/"
            "glm45-air-e8p-token-pair-slot-topk-output-group-stream-native-parity-20260704.json"
        )
    )
    assert (
        cli.DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_ARTIFACT_PARITY
        == Path(
            "artifacts/quality/"
            "glm45-air-e8p-token-pair-slot-topk-codeword-group-pipeline-artifact-parity-20260704.json"
        )
    )
    assert (
        cli.DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_SPEED_PACKET
        == Path(
            "artifacts/quality/"
            "glm45-air-e8p-token-pair-slot-topk-codeword-group-pipeline-speed-packet-20260704.json"
        )
    )
    assert (
        cli.DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_DESIGN
        == Path(
            "artifacts/quality/"
            "glm45-air-e8p-token-pair-slot-topk-scale-slot-broadcast-stream-design-20260704.json"
        )
    )
    assert (
        cli.DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_NATIVE_PARITY
        == Path(
            "artifacts/quality/"
            "glm45-air-e8p-token-pair-slot-topk-route-bucket-codeword-reduce-native-parity-20260704.json"
        )
    )
    assert (
        cli.DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_ARTIFACT_PARITY
        == Path(
            "artifacts/quality/"
            "glm45-air-e8p-token-pair-slot-topk-route-bucket-codeword-reduce-artifact-parity-20260704.json"
        )
    )
    assert (
        cli.DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_SPEED_PACKET
        == Path(
            "artifacts/quality/"
            "glm45-air-e8p-token-pair-slot-topk-route-bucket-codeword-reduce-speed-packet-20260704.json"
        )
    )
    assert cli.DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_DESIGN == Path(
        "artifacts/quality/"
        "glm45-air-e8p-token-pair-slot-topk-kblock-microtile-stream-design-20260704.json"
    )


def test_five_front_status_routes_kblock_wavefront_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "rejected_families": [{"evidence": str(index)} for index in range(11)],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "missing_kblock_wavefront_codeword_scan_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    route_microtile_design = _write_json(
        tmp_path / "route_microtile_design.json",
        {
            **base,
            "decision": (
                "route_microtile_codeword_block_reduce_ready_for_source_structure_probe"
            ),
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    output_tile_speed = _write_json(
        tmp_path / "output_tile_speed.json",
        {
            **base,
            "decision": "reject_output_tile_local_codeword_lut_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
        },
    )
    route_microtile_native = _write_json(
        tmp_path / "route_microtile_native.json",
        {
            **base,
            "decision": "route_microtile_codeword_block_reduce_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
        },
    )
    route_microtile_artifact = _write_json(
        tmp_path / "route_microtile_artifact.json",
        {
            **base,
            "decision": "route_microtile_codeword_block_reduce_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
        },
    )
    route_microtile_speed = _write_json(
        tmp_path / "route_microtile_speed.json",
        {
            **base,
            "decision": "reject_route_microtile_codeword_block_reduce_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
        },
    )
    kblock_design = _write_json(
        tmp_path / "kblock_design.json",
        {
            **base,
            "decision": "kblock_wavefront_codeword_scan_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_kblock_wavefront_codeword_scan"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_layer3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            )
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_layer77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            )
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": True, "cache_rows_generated": False},
    )
    model_card = tmp_path / "model-card.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=None,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=None,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "kblock_wavefront_codeword_scan_source_guardrail_missing"
    )
    assert trackb_front["kblock_wavefront_codeword_scan_design_decision"] == (
        "kblock_wavefront_codeword_scan_ready_for_source_structure_probe"
    )
    assert trackb_front[
        "kblock_wavefront_codeword_scan_source_guardrail_decision"
    ] == "missing_kblock_wavefront_codeword_scan_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_kblock_wavefront_codeword_scan"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_kblock_wavefront_codeword_scan_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    _write_json(
        source_guard,
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=None,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=None,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "kblock_wavefront_codeword_scan_source_guardrail_present"
    )
    assert trackb_front[
        "kblock_wavefront_codeword_scan_source_guardrail_decision"
    ] == "kblock_wavefront_codeword_scan_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_kblock_wavefront_codeword_scan_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_kblock_wavefront_codeword_scan_native_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    kblock_native = _write_json(
        tmp_path / "kblock_native.json",
        {
            **base,
            "decision": "kblock_wavefront_codeword_scan_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=None,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "kblock_wavefront_codeword_scan_native_parity_present"
    )
    assert trackb_front[
        "kblock_wavefront_codeword_scan_native_parity_decision"
    ] == "kblock_wavefront_codeword_scan_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_kblock_wavefront_codeword_scan_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_kblock_wavefront_codeword_scan_air_artifact_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    kblock_artifact = _write_json(
        tmp_path / "kblock_artifact.json",
        {
            **base,
            "decision": "kblock_wavefront_codeword_scan_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "kblock_wavefront_codeword_scan_artifact_parity_present"
    )
    assert trackb_front[
        "kblock_wavefront_codeword_scan_artifact_parity_decision"
    ] == "kblock_wavefront_codeword_scan_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_kblock_wavefront_codeword_scan_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_kblock_wavefront_codeword_scan_same_window_q2_speed_packet"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    kblock_speed = _write_json(
        tmp_path / "kblock_speed.json",
        {
            **base,
            "decision": "reject_kblock_wavefront_codeword_scan_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=None,
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=None,
        trackb_token_route_output_stripe_pipeline_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "kblock_wavefront_codeword_scan_speed_rejected"
    assert trackb_front[
        "kblock_wavefront_codeword_scan_speed_packet_decision"
    ] == "reject_kblock_wavefront_codeword_scan_speed_path"
    assert trackb_front["next"] == (
        "change_kblock_wavefront_codeword_scan_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_kblock_wavefront_codeword_scan_layout_or_kernel_family"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_route_design = _write_json(
        tmp_path / "token_route_output_stripe_pipeline_design.json",
        {
            **base,
            "decision": (
                "token_route_output_stripe_pipeline_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_route_output_stripe_pipeline"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=None,
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=None,
        trackb_token_route_output_stripe_pipeline_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_route_output_stripe_pipeline_design_ready"
    )
    assert trackb_front["token_route_output_stripe_pipeline_design_decision"] == (
        "token_route_output_stripe_pipeline_ready_for_source_structure_probe"
    )
    assert trackb_front[
        "token_route_output_stripe_pipeline_target_kernel_family"
    ] == "sorted_gather_qmm_rhs_nax_token_route_output_stripe_pipeline"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_route_output_stripe_pipeline"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_route_output_stripe_pipeline_source_guardrail"
    )
    assert str(token_route_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    _write_json(
        source_guard,
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": "missing_token_route_output_stripe_pipeline_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=None,
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=None,
        trackb_token_route_output_stripe_pipeline_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_route_output_stripe_pipeline_source_guardrail_missing"
    )
    assert trackb_front[
        "token_route_output_stripe_pipeline_source_guardrail_decision"
    ] == "missing_token_route_output_stripe_pipeline_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_route_output_stripe_pipeline"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_route_output_stripe_pipeline_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    _write_json(
        source_guard,
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": (
                    "token_route_output_stripe_pipeline_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=None,
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=None,
        trackb_token_route_output_stripe_pipeline_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_route_output_stripe_pipeline_source_guardrail_present"
    )
    assert trackb_front[
        "token_route_output_stripe_pipeline_source_guardrail_decision"
    ] == "token_route_output_stripe_pipeline_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_route_output_stripe_pipeline_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_route_output_stripe_pipeline_native_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_route_native = _write_json(
        tmp_path / "token_route_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_route_output_stripe_pipeline_native_parity"
            ),
            "decision": "token_route_output_stripe_pipeline_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=None,
        trackb_token_route_output_stripe_pipeline_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_route_output_stripe_pipeline_native_parity_present"
    )
    assert trackb_front[
        "token_route_output_stripe_pipeline_native_parity_decision"
    ] == "token_route_output_stripe_pipeline_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_route_output_stripe_pipeline_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_route_output_stripe_pipeline_air_artifact_parity"
    )
    assert str(token_route_native) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_route_artifact = _write_json(
        tmp_path / "token_route_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_route_output_stripe_pipeline_artifact_parity"
            ),
            "decision": "token_route_output_stripe_pipeline_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_route_output_stripe_pipeline_artifact_parity_present"
    )
    assert trackb_front[
        "token_route_output_stripe_pipeline_artifact_parity_decision"
    ] == "token_route_output_stripe_pipeline_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_route_output_stripe_pipeline_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_route_output_stripe_pipeline_same_window_q2_speed_packet"
    )
    assert str(token_route_artifact) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_route_speed = _write_json(
        tmp_path / "token_route_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_route_output_stripe_pipeline_speed_packet"
            ),
            "decision": "reject_token_route_output_stripe_pipeline_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "all_parity_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=None,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=None,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=None,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_route_output_stripe_pipeline_speed_rejected"
    )
    assert trackb_front[
        "token_route_output_stripe_pipeline_speed_packet_decision"
    ] == "reject_token_route_output_stripe_pipeline_speed_path"
    assert trackb_front["next"] == (
        "change_token_route_output_stripe_pipeline_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_route_output_stripe_pipeline_layout_or_kernel_family"
    )
    assert str(token_route_speed) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    expert_kblock_design = _write_json(
        tmp_path / "expert_kblock_scale_slot_stream_design.json",
        {
            **base,
            "decision": (
                "expert_kblock_scale_slot_stream_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_kblock_scale_slot_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=None,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=None,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "expert_kblock_scale_slot_stream_design_ready"
    assert trackb_front["expert_kblock_scale_slot_stream_design_decision"] == (
        "expert_kblock_scale_slot_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["expert_kblock_scale_slot_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_expert_kblock_scale_slot_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_expert_kblock_scale_slot_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_expert_kblock_scale_slot_stream_source_guardrail"
    )
    assert str(expert_kblock_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    _write_json(
        source_guard,
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": "token_route_output_stripe_pipeline_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "expert_kblock_scale_slot_stream_guardrail": {
                "decision": "missing_expert_kblock_scale_slot_stream_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=None,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=None,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "expert_kblock_scale_slot_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "expert_kblock_scale_slot_stream_source_guardrail_decision"
    ] == "missing_expert_kblock_scale_slot_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_expert_kblock_scale_slot_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_expert_kblock_scale_slot_stream_source_guardrail"
    )
    assert str(expert_kblock_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    native_parity = _write_json(
        tmp_path / "expert_kblock_scale_slot_stream_native_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_expert_kblock_scale_slot_stream_native_parity",
            "decision": "expert_kblock_scale_slot_stream_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "expert_kblock_scale_slot_stream_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_expert_kblock_scale_slot_stream_artifact_parity"
            ),
            "decision": "expert_kblock_scale_slot_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": True,
        },
    )
    speed_packet = _write_json(
        tmp_path / "expert_kblock_scale_slot_stream_speed_packet.json",
        {
            **base,
            "record_type": "glm45_air_e8p_expert_kblock_scale_slot_stream_speed_packet",
            "decision": "reject_expert_kblock_scale_slot_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    _write_json(
        source_guard,
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": "token_route_output_stripe_pipeline_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "expert_kblock_scale_slot_stream_guardrail": {
                "decision": "expert_kblock_scale_slot_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=None,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "expert_kblock_scale_slot_stream_native_parity_present"
    )
    assert trackb_front[
        "expert_kblock_scale_slot_stream_native_parity_decision"
    ] == "expert_kblock_scale_slot_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_expert_kblock_scale_slot_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_expert_kblock_scale_slot_stream_air_artifact_parity"
    )
    assert str(native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "expert_kblock_scale_slot_stream_artifact_parity_present"
    )
    assert trackb_front[
        "expert_kblock_scale_slot_stream_artifact_parity_decision"
    ] == "expert_kblock_scale_slot_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_expert_kblock_scale_slot_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_expert_kblock_scale_slot_stream_same_window_q2_speed_packet"
    )
    assert str(artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "expert_kblock_scale_slot_stream_speed_rejected"
    assert trackb_front[
        "expert_kblock_scale_slot_stream_speed_packet_decision"
    ] == "reject_expert_kblock_scale_slot_stream_speed_path"
    assert trackb_front["next"] == (
        "change_expert_kblock_scale_slot_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_expert_kblock_scale_slot_stream_layout_or_kernel_family"
    )
    assert str(speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    scale_group_design = _write_json(
        tmp_path / "scale_group_route_block_reduce_design.json",
        {
            **base,
            "decision": (
                "scale_group_route_block_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_scale_group_route_block_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "scale_group_route_block_reduce_design_ready"
    assert trackb_front["scale_group_route_block_reduce_design_decision"] == (
        "scale_group_route_block_reduce_ready_for_source_structure_probe"
    )
    assert trackb_front["scale_group_route_block_reduce_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_scale_group_route_block_reduce"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_scale_group_route_block_reduce"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_scale_group_route_block_reduce_source_guardrail"
    )
    assert str(scale_group_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    scale_group_missing_source_guard = _write_json(
        tmp_path / "scale_group_route_block_reduce_source_guard.json",
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": "token_route_output_stripe_pipeline_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "expert_kblock_scale_slot_stream_guardrail": {
                "decision": "expert_kblock_scale_slot_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "scale_group_route_block_reduce_guardrail": {
                "decision": "missing_scale_group_route_block_reduce_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=scale_group_missing_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "scale_group_route_block_reduce_source_guardrail_missing"
    )
    assert trackb_front[
        "scale_group_route_block_reduce_source_guardrail_decision"
    ] == "missing_scale_group_route_block_reduce_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_scale_group_route_block_reduce"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_scale_group_route_block_reduce_source_guardrail"
    )
    assert str(scale_group_design) in trackb_front["evidence"]
    assert str(scale_group_missing_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    scale_group_present_source_guard = _write_json(
        tmp_path / "scale_group_route_block_reduce_present_source_guard.json",
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": (
                    "token_route_output_stripe_pipeline_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "expert_kblock_scale_slot_stream_guardrail": {
                "decision": "expert_kblock_scale_slot_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "scale_group_route_block_reduce_guardrail": {
                "decision": "scale_group_route_block_reduce_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=scale_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=None,
        trackb_scale_group_route_block_reduce_artifact_parity_path=None,
        trackb_scale_group_route_block_reduce_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "scale_group_route_block_reduce_source_guardrail_present"
    )
    assert trackb_front[
        "scale_group_route_block_reduce_source_guardrail_decision"
    ] == "scale_group_route_block_reduce_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_scale_group_route_block_reduce_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_scale_group_route_block_reduce_native_parity"
    )
    assert str(scale_group_present_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    scale_group_native_parity = _write_json(
        tmp_path / "scale_group_route_block_reduce_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_scale_group_route_block_reduce_native_parity"
            ),
            "decision": "scale_group_route_block_reduce_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=scale_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=None,
        trackb_scale_group_route_block_reduce_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "scale_group_route_block_reduce_native_parity_present"
    )
    assert trackb_front[
        "scale_group_route_block_reduce_native_parity_decision"
    ] == "scale_group_route_block_reduce_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_scale_group_route_block_reduce_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_scale_group_route_block_reduce_air_artifact_parity"
    )
    assert str(scale_group_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    scale_group_artifact_parity = _write_json(
        tmp_path / "scale_group_route_block_reduce_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_scale_group_route_block_reduce_artifact_parity"
            ),
            "decision": "scale_group_route_block_reduce_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=scale_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "scale_group_route_block_reduce_artifact_parity_present"
    )
    assert trackb_front[
        "scale_group_route_block_reduce_artifact_parity_decision"
    ] == "scale_group_route_block_reduce_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_scale_group_route_block_reduce_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_scale_group_route_block_reduce_same_window_q2_speed_packet"
    )
    assert str(scale_group_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    scale_group_speed_packet = _write_json(
        tmp_path / "scale_group_route_block_reduce_speed_packet.json",
        {
            **base,
            "record_type": "glm45_air_e8p_scale_group_route_block_reduce_speed_packet",
            "decision": "reject_scale_group_route_block_reduce_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "all_parity_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=scale_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "scale_group_route_block_reduce_speed_rejected"
    assert trackb_front["scale_group_route_block_reduce_speed_packet_decision"] == (
        "reject_scale_group_route_block_reduce_speed_path"
    )
    assert trackb_front["next"] == (
        "change_scale_group_route_block_reduce_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_scale_group_route_block_reduce_layout_or_kernel_family"
    )
    assert str(scale_group_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_block_output_group_design = _write_json(
        tmp_path / "route_block_output_group_stream_design.json",
        {
            **base,
            "record_type": "glm45_air_e8p_trackb_route_block_output_group_stream_design",
            "decision": "route_block_output_group_stream_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_block_output_group_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=scale_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "route_block_output_group_stream_design_ready"
    assert trackb_front["route_block_output_group_stream_design_decision"] == (
        "route_block_output_group_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["route_block_output_group_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_route_block_output_group_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_block_output_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_block_output_group_stream_source_guardrail"
    )
    assert str(route_block_output_group_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_block_output_group_missing_source_guard = _write_json(
        tmp_path / "route_block_output_group_stream_source_guard.json",
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": "token_route_output_stripe_pipeline_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "expert_kblock_scale_slot_stream_guardrail": {
                "decision": "expert_kblock_scale_slot_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "scale_group_route_block_reduce_guardrail": {
                "decision": "scale_group_route_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "route_block_output_group_stream_guardrail": {
                "decision": "missing_route_block_output_group_stream_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=route_block_output_group_missing_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert (
        trackb_front["status"]
        == "route_block_output_group_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "route_block_output_group_stream_source_guardrail_decision"
    ] == "missing_route_block_output_group_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_block_output_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_block_output_group_stream_source_guardrail"
    )
    assert str(route_block_output_group_missing_source_guard) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_block_output_group_present_source_guard = _write_json(
        tmp_path / "route_block_output_group_stream_present_source_guard.json",
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": "token_route_output_stripe_pipeline_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "expert_kblock_scale_slot_stream_guardrail": {
                "decision": "expert_kblock_scale_slot_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "scale_group_route_block_reduce_guardrail": {
                "decision": "scale_group_route_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "route_block_output_group_stream_guardrail": {
                "decision": "route_block_output_group_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=route_block_output_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert (
        trackb_front["status"]
        == "route_block_output_group_stream_source_guardrail_present"
    )
    assert trackb_front[
        "route_block_output_group_stream_source_guardrail_decision"
    ] == "route_block_output_group_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_route_block_output_group_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_block_output_group_stream_native_parity"
    )
    assert str(route_block_output_group_present_source_guard) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_block_output_group_native_parity = _write_json(
        tmp_path / "route_block_output_group_stream_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_block_output_group_stream_native_parity"
            ),
            "decision": "route_block_output_group_stream_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=route_block_output_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert (
        trackb_front["status"]
        == "route_block_output_group_stream_native_parity_present"
    )
    assert trackb_front[
        "route_block_output_group_stream_native_parity_decision"
    ] == "route_block_output_group_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_route_block_output_group_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_block_output_group_stream_air_artifact_parity"
    )
    assert str(route_block_output_group_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_block_output_group_artifact_parity = _write_json(
        tmp_path / "route_block_output_group_stream_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_block_output_group_stream_artifact_parity"
            ),
            "decision": "route_block_output_group_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=route_block_output_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            route_block_output_group_artifact_parity
        ),
        trackb_route_block_output_group_stream_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert (
        trackb_front["status"]
        == "route_block_output_group_stream_artifact_parity_present"
    )
    assert trackb_front[
        "route_block_output_group_stream_artifact_parity_decision"
    ] == "route_block_output_group_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_route_block_output_group_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_route_block_output_group_stream_same_window_q2_speed_packet"
    )
    assert str(route_block_output_group_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_block_output_group_speed_packet = _write_json(
        tmp_path / "route_block_output_group_stream_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_block_output_group_stream_speed_packet"
            ),
            "decision": "reject_route_block_output_group_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "all_parity_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=route_block_output_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            route_block_output_group_artifact_parity
        ),
        trackb_route_block_output_group_stream_speed_packet_path=(
            route_block_output_group_speed_packet
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "route_block_output_group_stream_speed_rejected"
    assert trackb_front[
        "route_block_output_group_stream_speed_packet_decision"
    ] == "reject_route_block_output_group_stream_speed_path"
    assert trackb_front["next"] == (
        "change_route_block_output_group_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_route_block_output_group_stream_layout_or_kernel_family"
    )
    assert str(route_block_output_group_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_group_pretransposed_design = _write_json(
        tmp_path / "output_group_pretransposed_codeword_stream_design.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_trackb_output_group_pretransposed_codeword_stream_design"
            ),
            "decision": (
                "output_group_pretransposed_codeword_stream_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_output_group_pretransposed_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=route_block_output_group_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            route_block_output_group_artifact_parity
        ),
        trackb_route_block_output_group_stream_speed_packet_path=(
            route_block_output_group_speed_packet
        ),
        trackb_output_group_pretransposed_codeword_stream_design_path=(
            output_group_pretransposed_design
        ),
        trackb_output_group_pretransposed_codeword_stream_native_parity_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "output_group_pretransposed_codeword_stream_design_ready"
    )
    assert trackb_front[
        "output_group_pretransposed_codeword_stream_design_decision"
    ] == "output_group_pretransposed_codeword_stream_ready_for_source_structure_probe"
    assert trackb_front[
        "output_group_pretransposed_codeword_stream_target_kernel_family"
    ] == "sorted_gather_qmm_rhs_nax_output_group_pretransposed_codeword_stream"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_output_group_pretransposed_codeword_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_output_group_pretransposed_codeword_stream_source_guardrail"
    )
    assert str(output_group_pretransposed_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_group_pretransposed_missing_source_guard = _write_json(
        tmp_path / "output_group_pretransposed_source_guard.json",
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": "token_route_output_stripe_pipeline_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "expert_kblock_scale_slot_stream_guardrail": {
                "decision": "expert_kblock_scale_slot_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "scale_group_route_block_reduce_guardrail": {
                "decision": "scale_group_route_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "route_block_output_group_stream_guardrail": {
                "decision": "route_block_output_group_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "output_group_pretransposed_codeword_stream_guardrail": {
                "decision": (
                    "missing_output_group_pretransposed_codeword_stream_source"
                ),
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=output_group_pretransposed_missing_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            route_block_output_group_artifact_parity
        ),
        trackb_route_block_output_group_stream_speed_packet_path=(
            route_block_output_group_speed_packet
        ),
        trackb_output_group_pretransposed_codeword_stream_design_path=(
            output_group_pretransposed_design
        ),
        trackb_output_group_pretransposed_codeword_stream_native_parity_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "output_group_pretransposed_codeword_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "output_group_pretransposed_codeword_stream_source_guardrail_decision"
    ] == "missing_output_group_pretransposed_codeword_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_output_group_pretransposed_codeword_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_output_group_pretransposed_codeword_stream_source_guardrail"
    )
    assert str(output_group_pretransposed_missing_source_guard) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_group_pretransposed_present_source_guard = _write_json(
        tmp_path / "output_group_pretransposed_present_source_guard.json",
        {
            **base,
            "route_microtile_codeword_block_reduce_guardrail": {
                "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "kblock_wavefront_codeword_scan_guardrail": {
                "decision": "kblock_wavefront_codeword_scan_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_route_output_stripe_pipeline_guardrail": {
                "decision": "token_route_output_stripe_pipeline_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "expert_kblock_scale_slot_stream_guardrail": {
                "decision": "expert_kblock_scale_slot_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "scale_group_route_block_reduce_guardrail": {
                "decision": "scale_group_route_block_reduce_source_guardrail_present",
                "passes_contract": True,
            },
            "route_block_output_group_stream_guardrail": {
                "decision": "route_block_output_group_stream_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "output_group_pretransposed_codeword_stream_guardrail": {
                "decision": (
                    "output_group_pretransposed_codeword_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=output_group_pretransposed_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            route_block_output_group_artifact_parity
        ),
        trackb_route_block_output_group_stream_speed_packet_path=(
            route_block_output_group_speed_packet
        ),
        trackb_output_group_pretransposed_codeword_stream_design_path=(
            output_group_pretransposed_design
        ),
        trackb_output_group_pretransposed_codeword_stream_native_parity_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "output_group_pretransposed_codeword_stream_source_guardrail_present"
    )
    assert trackb_front[
        "output_group_pretransposed_codeword_stream_source_guardrail_decision"
    ] == "output_group_pretransposed_codeword_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_output_group_pretransposed_codeword_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_output_group_pretransposed_codeword_stream_native_parity"
    )
    assert str(output_group_pretransposed_present_source_guard) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_group_pretransposed_native_parity = _write_json(
        tmp_path / "output_group_pretransposed_native_parity.json",
        {
            **base,
            "decision": (
                "output_group_pretransposed_codeword_stream_native_parity_pass"
            ),
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=output_group_pretransposed_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            route_block_output_group_artifact_parity
        ),
        trackb_route_block_output_group_stream_speed_packet_path=(
            route_block_output_group_speed_packet
        ),
        trackb_output_group_pretransposed_codeword_stream_design_path=(
            output_group_pretransposed_design
        ),
        trackb_output_group_pretransposed_codeword_stream_native_parity_path=(
            output_group_pretransposed_native_parity
        ),
        trackb_output_group_pretransposed_codeword_stream_artifact_parity_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "output_group_pretransposed_codeword_stream_native_parity_present"
    )
    assert trackb_front[
        "output_group_pretransposed_codeword_stream_native_parity_decision"
    ] == "output_group_pretransposed_codeword_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_output_group_pretransposed_codeword_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_output_group_pretransposed_codeword_stream_air_artifact_parity"
    )
    assert str(output_group_pretransposed_native_parity) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_group_pretransposed_artifact_parity = _write_json(
        tmp_path / "output_group_pretransposed_artifact_parity.json",
        {
            **base,
            "decision": (
                "output_group_pretransposed_codeword_stream_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=output_group_pretransposed_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            route_block_output_group_artifact_parity
        ),
        trackb_route_block_output_group_stream_speed_packet_path=(
            route_block_output_group_speed_packet
        ),
        trackb_output_group_pretransposed_codeword_stream_design_path=(
            output_group_pretransposed_design
        ),
        trackb_output_group_pretransposed_codeword_stream_native_parity_path=(
            output_group_pretransposed_native_parity
        ),
        trackb_output_group_pretransposed_codeword_stream_artifact_parity_path=(
            output_group_pretransposed_artifact_parity
        ),
        trackb_output_group_pretransposed_codeword_stream_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "output_group_pretransposed_codeword_stream_artifact_parity_present"
    )
    assert trackb_front[
        "output_group_pretransposed_codeword_stream_artifact_parity_decision"
    ] == "output_group_pretransposed_codeword_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_output_group_pretransposed_codeword_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_output_group_pretransposed_codeword_stream_same_window_q2_speed_packet"
    )
    assert str(output_group_pretransposed_artifact_parity) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_group_pretransposed_speed_packet = _write_json(
        tmp_path / "output_group_pretransposed_speed_packet.json",
        {
            **base,
            "decision": "reject_output_group_pretransposed_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=output_group_pretransposed_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            route_block_output_group_artifact_parity
        ),
        trackb_route_block_output_group_stream_speed_packet_path=(
            route_block_output_group_speed_packet
        ),
        trackb_output_group_pretransposed_codeword_stream_design_path=(
            output_group_pretransposed_design
        ),
        trackb_output_group_pretransposed_codeword_stream_native_parity_path=(
            output_group_pretransposed_native_parity
        ),
        trackb_output_group_pretransposed_codeword_stream_artifact_parity_path=(
            output_group_pretransposed_artifact_parity
        ),
        trackb_output_group_pretransposed_codeword_stream_speed_packet_path=(
            output_group_pretransposed_speed_packet
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "output_group_pretransposed_codeword_stream_speed_rejected"
    )
    assert trackb_front[
        "output_group_pretransposed_codeword_stream_speed_packet_decision"
    ] == "reject_output_group_pretransposed_codeword_stream_speed_path"
    assert trackb_front["next"] == (
        "change_output_group_pretransposed_codeword_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_output_group_pretransposed_codeword_stream_layout_or_kernel_family"
    )
    assert str(output_group_pretransposed_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    kblock_output_group_route_fused_design = _write_json(
        tmp_path / "kblock_output_group_route_fused_design.json",
        {
            **base,
            "decision": (
                "kblock_output_group_route_fused_stream_ready_for_source_structure_probe"
            ),
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_kblock_output_group_route_fused_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_kblock_output_group_route_fused_streams"
                ),
            },
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=output_group_pretransposed_present_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_output_tile_local_codeword_lut_speed_packet_path=output_tile_speed,
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            route_microtile_speed
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
        trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=kblock_artifact,
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
        trackb_token_route_output_stripe_pipeline_design_path=token_route_design,
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            token_route_native
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            token_route_artifact
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=token_route_speed,
        trackb_expert_kblock_scale_slot_stream_design_path=expert_kblock_design,
        trackb_expert_kblock_scale_slot_stream_native_parity_path=native_parity,
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=artifact_parity,
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
        trackb_scale_group_route_block_reduce_design_path=scale_group_design,
        trackb_scale_group_route_block_reduce_native_parity_path=(
            scale_group_native_parity
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            scale_group_artifact_parity
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            scale_group_speed_packet
        ),
        trackb_route_block_output_group_stream_design_path=(
            route_block_output_group_design
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            route_block_output_group_native_parity
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            route_block_output_group_artifact_parity
        ),
        trackb_route_block_output_group_stream_speed_packet_path=(
            route_block_output_group_speed_packet
        ),
        trackb_output_group_pretransposed_codeword_stream_design_path=(
            output_group_pretransposed_design
        ),
        trackb_output_group_pretransposed_codeword_stream_native_parity_path=(
            output_group_pretransposed_native_parity
        ),
        trackb_output_group_pretransposed_codeword_stream_artifact_parity_path=(
            output_group_pretransposed_artifact_parity
        ),
        trackb_output_group_pretransposed_codeword_stream_speed_packet_path=(
            output_group_pretransposed_speed_packet
        ),
        trackb_kblock_output_group_route_fused_stream_design_path=(
            kblock_output_group_route_fused_design
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "kblock_output_group_route_fused_stream_design_ready"
    assert trackb_front["kblock_output_group_route_fused_stream_design_decision"] == (
        "kblock_output_group_route_fused_stream_ready_for_source_structure_probe"
    )
    assert trackb_front[
        "kblock_output_group_route_fused_stream_target_kernel_family"
    ] == "sorted_gather_qmm_rhs_nax_kblock_output_group_route_fused_stream"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_kblock_output_group_route_fused_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_kblock_output_group_route_fused_stream_source_guardrail"
    )
    assert str(kblock_output_group_route_fused_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    def _build_kblock_route_fused_status(
        source_guard_path: Path,
        native_parity_path: Path | None = None,
        artifact_parity_path: Path | None = None,
        speed_packet_path: Path | None = None,
        route_tile_output_swizzle_design_path: Path | None = None,
        route_tile_output_swizzle_native_parity_path: Path | None = None,
        route_tile_output_swizzle_artifact_parity_path: Path | None = None,
        route_tile_output_swizzle_speed_packet_path: Path | None = None,
        token_topk_output_tile_design_path: Path | None = None,
        token_topk_output_tile_native_parity_path: Path | None = None,
        token_topk_output_tile_artifact_parity_path: Path | None = None,
        token_topk_output_tile_speed_packet_path: Path | None = None,
        token_block_output_group_design_path: Path | None = None,
        token_block_output_group_native_parity_path: Path | None = None,
        token_block_output_group_artifact_parity_path: Path | None = None,
        token_block_output_group_speed_packet_path: Path | None = None,
        token_output_stripe_group_design_path: Path | None = None,
        token_output_stripe_group_native_parity_path: Path | None = None,
        token_output_stripe_group_artifact_parity_path: Path | None = None,
        token_output_stripe_group_speed_packet_path: Path | None = None,
        token_expert_output_block_stream_design_path: Path | None = None,
        token_expert_output_block_stream_native_parity_path: Path | None = None,
        token_expert_output_block_stream_artifact_parity_path: Path | None = None,
        token_expert_output_block_stream_speed_packet_path: Path | None = None,
        token_pair_kblock_accumulator_stream_design_path: Path | None = None,
        token_pair_kblock_accumulator_stream_native_parity_path: Path | None = None,
        token_pair_kblock_accumulator_stream_artifact_parity_path: Path | None = None,
        token_pair_kblock_accumulator_stream_speed_packet_path: Path | None = None,
        token_pair_output_group_stream_design_path: Path | None = None,
        token_pair_output_group_stream_native_parity_path: Path | None = None,
        token_pair_output_group_stream_artifact_parity_path: Path | None = None,
        token_pair_output_group_stream_speed_packet_path: Path | None = None,
        token_pair_slot_topk_output_group_stream_design_path: Path | None = None,
        token_pair_slot_topk_output_group_stream_native_parity_path: Path
        | None = None,
        token_pair_slot_topk_output_group_stream_artifact_parity_path: Path
        | None = None,
        token_pair_slot_topk_output_group_stream_speed_packet_path: Path
        | None = None,
        token_pair_slot_topk_codeword_group_pipeline_design_path: Path
        | None = None,
        token_pair_slot_topk_codeword_group_pipeline_native_parity_path: Path
        | None = None,
        token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path: Path
        | None = None,
        token_pair_slot_topk_codeword_group_pipeline_speed_packet_path: Path
        | None = None,
        token_pair_slot_topk_scale_slot_broadcast_stream_design_path: Path
        | None = None,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path: Path
        | None = None,
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path: Path
        | None = None,
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path: Path
        | None = None,
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path: Path
        | None = None,
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path: Path
        | None = None,
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path: Path
        | None = None,
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path: Path
        | None = None,
        token_pair_slot_topk_kblock_microtile_stream_design_path: Path
        | None = None,
        token_pair_slot_topk_kblock_microtile_stream_native_parity_path: Path
        | None = None,
        token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path: Path
        | None = None,
        token_pair_slot_topk_kblock_microtile_stream_speed_packet_path: Path
        | None = None,
        token_pair_slot_topk_output_tile_fused_stream_design_path: Path
        | None = None,
        token_pair_slot_topk_output_tile_fused_stream_native_parity_path: Path
        | None = None,
        token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path: Path
        | None = None,
        token_pair_slot_topk_output_tile_fused_stream_speed_packet_path: Path
        | None = None,
    ) -> dict:
        assert (
            "trackb_kblock_output_group_route_fused_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_route_tile_output_swizzle_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_route_tile_output_swizzle_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_route_tile_output_swizzle_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_route_tile_output_swizzle_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_topk_output_tile_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_topk_output_tile_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_topk_output_tile_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_topk_output_tile_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_block_output_group_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_block_output_group_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_block_output_group_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_block_output_group_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_output_stripe_group_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_output_stripe_group_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_output_stripe_group_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_output_stripe_group_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_expert_output_block_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_expert_output_block_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_expert_output_block_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_kblock_accumulator_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_kblock_accumulator_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_kblock_accumulator_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_kblock_accumulator_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_output_group_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_output_group_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_output_group_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_output_group_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_output_group_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_output_group_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_output_group_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_output_group_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_codeword_group_pipeline_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_kblock_microtile_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_output_tile_fused_stream_design_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path"
            in inspect.signature(cli.build_status).parameters
        )
        assert (
            "trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_path"
            in inspect.signature(cli.build_status).parameters
        )
        status_kwargs = dict(
            workstream1_path=workstream1,
            trackb_path=trackb,
            trackb_design_path=None,
            trackb_source_guard_path=source_guard_path,
            trackb_native_parity_path=None,
            trackb_artifact_parity_path=None,
            trackb_speed_packet_path=None,
            trackb_successor_design_path=None,
            trackb_successor_native_parity_path=None,
            trackb_successor_artifact_parity_path=None,
            trackb_successor_speed_packet_path=None,
            trackb_output_tile_local_codeword_lut_speed_packet_path=(
                output_tile_speed
            ),
            trackb_route_microtile_codeword_block_reduce_design_path=(
                route_microtile_design
            ),
            trackb_route_microtile_codeword_block_reduce_native_parity_path=(
                route_microtile_native
            ),
            trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
                route_microtile_artifact
            ),
            trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
                route_microtile_speed
            ),
            trackb_kblock_wavefront_codeword_scan_design_path=kblock_design,
            trackb_kblock_wavefront_codeword_scan_native_parity_path=kblock_native,
            trackb_kblock_wavefront_codeword_scan_artifact_parity_path=(
                kblock_artifact
            ),
            trackb_kblock_wavefront_codeword_scan_speed_packet_path=kblock_speed,
            trackb_token_route_output_stripe_pipeline_design_path=(
                token_route_design
            ),
            trackb_token_route_output_stripe_pipeline_native_parity_path=(
                token_route_native
            ),
            trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
                token_route_artifact
            ),
            trackb_token_route_output_stripe_pipeline_speed_packet_path=(
                token_route_speed
            ),
            trackb_expert_kblock_scale_slot_stream_design_path=(
                expert_kblock_design
            ),
            trackb_expert_kblock_scale_slot_stream_native_parity_path=(
                native_parity
            ),
            trackb_expert_kblock_scale_slot_stream_artifact_parity_path=(
                artifact_parity
            ),
            trackb_expert_kblock_scale_slot_stream_speed_packet_path=speed_packet,
            trackb_scale_group_route_block_reduce_design_path=scale_group_design,
            trackb_scale_group_route_block_reduce_native_parity_path=(
                scale_group_native_parity
            ),
            trackb_scale_group_route_block_reduce_artifact_parity_path=(
                scale_group_artifact_parity
            ),
            trackb_scale_group_route_block_reduce_speed_packet_path=(
                scale_group_speed_packet
            ),
            trackb_route_block_output_group_stream_design_path=(
                route_block_output_group_design
            ),
            trackb_route_block_output_group_stream_native_parity_path=(
                route_block_output_group_native_parity
            ),
            trackb_route_block_output_group_stream_artifact_parity_path=(
                route_block_output_group_artifact_parity
            ),
            trackb_route_block_output_group_stream_speed_packet_path=(
                route_block_output_group_speed_packet
            ),
            trackb_output_group_pretransposed_codeword_stream_design_path=(
                output_group_pretransposed_design
            ),
            trackb_output_group_pretransposed_codeword_stream_native_parity_path=(
                output_group_pretransposed_native_parity
            ),
            trackb_output_group_pretransposed_codeword_stream_artifact_parity_path=(
                output_group_pretransposed_artifact_parity
            ),
            trackb_output_group_pretransposed_codeword_stream_speed_packet_path=(
                output_group_pretransposed_speed_packet
            ),
            trackb_kblock_output_group_route_fused_stream_design_path=(
                kblock_output_group_route_fused_design
            ),
            trackb_kblock_output_group_route_fused_stream_native_parity_path=(
                native_parity_path
            ),
            trackb_kblock_output_group_route_fused_stream_artifact_parity_path=(
                artifact_parity_path
            ),
            trackb_kblock_output_group_route_fused_stream_speed_packet_path=(
                speed_packet_path
            ),
            trackb_route_tile_output_swizzle_stream_design_path=(
                route_tile_output_swizzle_design_path
            ),
            trackb_route_tile_output_swizzle_stream_native_parity_path=(
                route_tile_output_swizzle_native_parity_path
            ),
            trackb_route_tile_output_swizzle_stream_artifact_parity_path=(
                route_tile_output_swizzle_artifact_parity_path
            ),
            trackb_route_tile_output_swizzle_stream_speed_packet_path=(
                route_tile_output_swizzle_speed_packet_path
            ),
            trackb_token_topk_output_tile_stream_design_path=(
                token_topk_output_tile_design_path
            ),
            trackb_token_topk_output_tile_stream_native_parity_path=(
                token_topk_output_tile_native_parity_path
            ),
            trackb_token_topk_output_tile_stream_artifact_parity_path=(
                token_topk_output_tile_artifact_parity_path
            ),
            trackb_token_topk_output_tile_stream_speed_packet_path=(
                token_topk_output_tile_speed_packet_path
            ),
            trackb_token_block_output_group_stream_design_path=(
                token_block_output_group_design_path
            ),
            trackb_token_block_output_group_stream_native_parity_path=(
                token_block_output_group_native_parity_path
            ),
            trackb_token_block_output_group_stream_artifact_parity_path=(
                token_block_output_group_artifact_parity_path
            ),
            trackb_token_block_output_group_stream_speed_packet_path=(
                token_block_output_group_speed_packet_path
            ),
            trackb_token_output_stripe_group_stream_design_path=(
                token_output_stripe_group_design_path
            ),
            trackb_token_output_stripe_group_stream_native_parity_path=(
                token_output_stripe_group_native_parity_path
            ),
            trackb_token_output_stripe_group_stream_artifact_parity_path=(
                token_output_stripe_group_artifact_parity_path
            ),
            trackb_token_output_stripe_group_stream_speed_packet_path=(
                token_output_stripe_group_speed_packet_path
            ),
            trackb_token_expert_output_block_stream_design_path=(
                token_expert_output_block_stream_design_path
            ),
            trackb_token_expert_output_block_stream_native_parity_path=(
                token_expert_output_block_stream_native_parity_path
            ),
            trackb_token_expert_output_block_stream_artifact_parity_path=(
                token_expert_output_block_stream_artifact_parity_path
            ),
            trackb_token_expert_output_block_stream_speed_packet_path=(
                token_expert_output_block_stream_speed_packet_path
            ),
            trackb_token_pair_kblock_accumulator_stream_design_path=(
                token_pair_kblock_accumulator_stream_design_path
            ),
            trackb_token_pair_kblock_accumulator_stream_native_parity_path=(
                token_pair_kblock_accumulator_stream_native_parity_path
            ),
            trackb_token_pair_kblock_accumulator_stream_artifact_parity_path=(
                token_pair_kblock_accumulator_stream_artifact_parity_path
            ),
            trackb_token_pair_kblock_accumulator_stream_speed_packet_path=(
                token_pair_kblock_accumulator_stream_speed_packet_path
            ),
            trackb_token_pair_output_group_stream_design_path=(
                token_pair_output_group_stream_design_path
            ),
            trackb_token_pair_output_group_stream_native_parity_path=(
                token_pair_output_group_stream_native_parity_path
            ),
            trackb_token_pair_output_group_stream_artifact_parity_path=(
                token_pair_output_group_stream_artifact_parity_path
            ),
            trackb_token_pair_output_group_stream_speed_packet_path=(
                token_pair_output_group_stream_speed_packet_path
            ),
            trackb_token_pair_slot_topk_output_group_stream_design_path=(
                token_pair_slot_topk_output_group_stream_design_path
            ),
            trackb_token_pair_slot_topk_output_group_stream_native_parity_path=(
                token_pair_slot_topk_output_group_stream_native_parity_path
            ),
            trackb_token_pair_slot_topk_output_group_stream_artifact_parity_path=(
                token_pair_slot_topk_output_group_stream_artifact_parity_path
            ),
            trackb_token_pair_slot_topk_output_group_stream_speed_packet_path=(
                token_pair_slot_topk_output_group_stream_speed_packet_path
            ),
            trackb_token_pair_slot_topk_codeword_group_pipeline_design_path=(
                token_pair_slot_topk_codeword_group_pipeline_design_path
            ),
            trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_path=(
                token_pair_slot_topk_codeword_group_pipeline_native_parity_path
            ),
            trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path=(
                token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path
            ),
            trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_path=(
                token_pair_slot_topk_codeword_group_pipeline_speed_packet_path
            ),
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_path=(
                token_pair_slot_topk_scale_slot_broadcast_stream_design_path
            ),
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
                token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path
            ),
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
                token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path
            ),
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
                token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path
            ),
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
                token_pair_slot_topk_route_bucket_codeword_reduce_design_path
            ),
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
                token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path
            ),
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
                token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path
            ),
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
                token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path
            ),
            trackb_token_pair_slot_topk_kblock_microtile_stream_design_path=(
                token_pair_slot_topk_kblock_microtile_stream_design_path
            ),
            trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
                token_pair_slot_topk_kblock_microtile_stream_native_parity_path
            ),
            trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path=(
                token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path
            ),
            trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_path=(
                token_pair_slot_topk_kblock_microtile_stream_speed_packet_path
            ),
            trackb_token_pair_slot_topk_output_tile_fused_stream_design_path=(
                token_pair_slot_topk_output_tile_fused_stream_design_path
            ),
            trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_path=(
                token_pair_slot_topk_output_tile_fused_stream_native_parity_path
            ),
            trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path=(
                token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path
            ),
            trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_path=(
                token_pair_slot_topk_output_tile_fused_stream_speed_packet_path
            ),
            qwen_gate_path=qwen,
            glm52_layer3_path=glm52_layer3,
            glm52_layer77_path=glm52_layer77,
            cache_source_scan_path=cache,
            model_card_path=model_card,
        )
        return cli.build_status(**status_kwargs)

    output_group_present_payload = json.loads(
        output_group_pretransposed_present_source_guard.read_text(encoding="utf-8")
    )
    kblock_route_fused_missing_source_guard = _write_json(
        tmp_path / "kblock_route_fused_missing_source_guard.json",
        {
            **output_group_present_payload,
            "kblock_output_group_route_fused_stream_guardrail": {
                "decision": "missing_kblock_output_group_route_fused_stream_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = _build_kblock_route_fused_status(kblock_route_fused_missing_source_guard)
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "kblock_output_group_route_fused_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "kblock_output_group_route_fused_stream_source_guardrail_decision"
    ] == "missing_kblock_output_group_route_fused_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_kblock_output_group_route_fused_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_kblock_output_group_route_fused_stream_source_guardrail"
    )
    assert str(kblock_route_fused_missing_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    kblock_route_fused_present_source_guard = _write_json(
        tmp_path / "kblock_route_fused_present_source_guard.json",
        {
            **output_group_present_payload,
            "kblock_output_group_route_fused_stream_guardrail": {
                "decision": (
                    "kblock_output_group_route_fused_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = _build_kblock_route_fused_status(kblock_route_fused_present_source_guard)
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "kblock_output_group_route_fused_stream_source_guardrail_present"
    )
    assert trackb_front[
        "kblock_output_group_route_fused_stream_source_guardrail_decision"
    ] == "kblock_output_group_route_fused_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_kblock_output_group_route_fused_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_kblock_output_group_route_fused_stream_native_parity"
    )
    assert str(kblock_route_fused_present_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    kblock_route_fused_native_parity = _write_json(
        tmp_path / "kblock_route_fused_native_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_kblock_output_group_route_fused_stream_native_parity"
            ),
            "decision": "kblock_output_group_route_fused_stream_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        kblock_route_fused_present_source_guard,
        kblock_route_fused_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "kblock_output_group_route_fused_stream_native_parity_present"
    )
    assert trackb_front[
        "kblock_output_group_route_fused_stream_native_parity_decision"
    ] == "kblock_output_group_route_fused_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_kblock_output_group_route_fused_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_kblock_output_group_route_fused_stream_air_artifact_parity"
    )
    assert str(kblock_route_fused_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    kblock_route_fused_artifact_parity = _write_json(
        tmp_path / "kblock_route_fused_artifact_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_kblock_output_group_route_fused_stream_artifact_parity"
            ),
            "decision": (
                "kblock_output_group_route_fused_stream_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        kblock_route_fused_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "kblock_output_group_route_fused_stream_artifact_parity_present"
    )
    assert trackb_front[
        "kblock_output_group_route_fused_stream_artifact_parity_decision"
    ] == "kblock_output_group_route_fused_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_kblock_output_group_route_fused_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_kblock_output_group_route_fused_stream_same_window_q2_speed_packet"
    )
    assert str(kblock_route_fused_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    kblock_route_fused_speed_packet = _write_json(
        tmp_path / "kblock_route_fused_speed_packet.json",
        {
            "record_type": (
                "glm45_air_e8p_kblock_output_group_route_fused_stream_speed_packet"
            ),
            "decision": "reject_kblock_output_group_route_fused_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        kblock_route_fused_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "kblock_output_group_route_fused_stream_speed_rejected"
    )
    assert trackb_front[
        "kblock_output_group_route_fused_stream_speed_packet_decision"
    ] == "reject_kblock_output_group_route_fused_stream_speed_path"
    assert trackb_front["next"] == (
        "change_kblock_output_group_route_fused_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_kblock_output_group_route_fused_stream_layout_or_kernel_family"
    )
    assert str(kblock_route_fused_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_tile_output_swizzle_design = _write_json(
        tmp_path / "route_tile_output_swizzle_design.json",
        {
            **base,
            "decision": (
                "route_tile_output_swizzle_stream_ready_for_source_structure_probe"
            ),
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_tile_output_swizzle_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_route_tile_output_swizzle_streams"
                ),
            },
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        kblock_route_fused_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "route_tile_output_swizzle_stream_design_ready"
    assert trackb_front["route_tile_output_swizzle_stream_design_decision"] == (
        "route_tile_output_swizzle_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["route_tile_output_swizzle_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_route_tile_output_swizzle_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_tile_output_swizzle_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_tile_output_swizzle_stream_source_guardrail"
    )
    assert str(route_tile_output_swizzle_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_tile_output_swizzle_missing_source_guard = _write_json(
        tmp_path / "route_tile_output_swizzle_missing_source_guard.json",
        {
            **output_group_present_payload,
            "kblock_output_group_route_fused_stream_guardrail": {
                "decision": (
                    "kblock_output_group_route_fused_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "route_tile_output_swizzle_stream_guardrail": {
                "decision": "missing_route_tile_output_swizzle_stream_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = _build_kblock_route_fused_status(
        route_tile_output_swizzle_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "route_tile_output_swizzle_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "route_tile_output_swizzle_stream_source_guardrail_decision"
    ] == "missing_route_tile_output_swizzle_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_tile_output_swizzle_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_tile_output_swizzle_stream_source_guardrail"
    )
    assert str(route_tile_output_swizzle_missing_source_guard) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_tile_output_swizzle_present_source_guard = _write_json(
        tmp_path / "route_tile_output_swizzle_present_source_guard.json",
        {
            **output_group_present_payload,
            "kblock_output_group_route_fused_stream_guardrail": {
                "decision": (
                    "kblock_output_group_route_fused_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "route_tile_output_swizzle_stream_guardrail": {
                "decision": (
                    "route_tile_output_swizzle_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    route_tile_output_swizzle_native_parity = _write_json(
        tmp_path / "route_tile_output_swizzle_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_tile_output_swizzle_stream_native_parity"
            ),
            "decision": "route_tile_output_swizzle_stream_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        route_tile_output_swizzle_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "route_tile_output_swizzle_stream_native_parity_present"
    )
    assert trackb_front[
        "route_tile_output_swizzle_stream_native_parity_decision"
    ] == "route_tile_output_swizzle_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_route_tile_output_swizzle_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_tile_output_swizzle_stream_air_artifact_parity"
    )
    assert str(route_tile_output_swizzle_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_tile_output_swizzle_artifact_parity = _write_json(
        tmp_path / "route_tile_output_swizzle_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_tile_output_swizzle_stream_artifact_parity"
            ),
            "decision": (
                "route_tile_output_swizzle_stream_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        route_tile_output_swizzle_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "route_tile_output_swizzle_stream_artifact_parity_present"
    )
    assert trackb_front[
        "route_tile_output_swizzle_stream_artifact_parity_decision"
    ] == "route_tile_output_swizzle_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_route_tile_output_swizzle_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_route_tile_output_swizzle_stream_same_window_q2_speed_packet"
    )
    assert str(route_tile_output_swizzle_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_tile_output_swizzle_speed_packet = _write_json(
        tmp_path / "route_tile_output_swizzle_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_tile_output_swizzle_stream_speed_packet"
            ),
            "decision": "reject_route_tile_output_swizzle_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        route_tile_output_swizzle_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "route_tile_output_swizzle_stream_speed_rejected"
    assert trackb_front["route_tile_output_swizzle_stream_speed_packet_decision"] == (
        "reject_route_tile_output_swizzle_stream_speed_path"
    )
    assert trackb_front["next"] == (
        "change_route_tile_output_swizzle_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_route_tile_output_swizzle_stream_layout_or_kernel_family"
    )
    assert str(route_tile_output_swizzle_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_topk_output_tile_design = _write_json(
        tmp_path / "token_topk_output_tile_design.json",
        {
            **base,
            "decision": (
                "token_topk_output_tile_stream_ready_for_source_structure_probe"
            ),
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_topk_output_tile_stream"
                ),
                "storage_constraint": "compressed_e8p_token_topk_output_tile_streams",
            },
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        route_tile_output_swizzle_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "token_topk_output_tile_stream_design_ready"
    assert trackb_front["token_topk_output_tile_stream_design_decision"] == (
        "token_topk_output_tile_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["token_topk_output_tile_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_topk_output_tile_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_topk_output_tile_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_topk_output_tile_stream_source_guardrail"
    )
    assert str(token_topk_output_tile_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_topk_output_tile_missing_source_guard = _write_json(
        tmp_path / "token_topk_output_tile_missing_source_guard.json",
        {
            **output_group_present_payload,
            "kblock_output_group_route_fused_stream_guardrail": {
                "decision": (
                    "kblock_output_group_route_fused_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "route_tile_output_swizzle_stream_guardrail": {
                "decision": (
                    "route_tile_output_swizzle_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_topk_output_tile_stream_guardrail": {
                "decision": "missing_token_topk_output_tile_stream_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = _build_kblock_route_fused_status(
        token_topk_output_tile_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_topk_output_tile_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_topk_output_tile_stream_source_guardrail_decision"
    ] == "missing_token_topk_output_tile_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_topk_output_tile_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_topk_output_tile_stream_source_guardrail"
    )
    assert str(token_topk_output_tile_missing_source_guard) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_topk_output_tile_present_source_guard = _write_json(
        tmp_path / "token_topk_output_tile_present_source_guard.json",
        {
            **output_group_present_payload,
            "kblock_output_group_route_fused_stream_guardrail": {
                "decision": (
                    "kblock_output_group_route_fused_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "route_tile_output_swizzle_stream_guardrail": {
                "decision": (
                    "route_tile_output_swizzle_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_topk_output_tile_stream_guardrail": {
                "decision": (
                    "token_topk_output_tile_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    token_topk_output_tile_native_parity = _write_json(
        tmp_path / "token_topk_output_tile_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_topk_output_tile_stream_native_parity"
            ),
            "decision": "token_topk_output_tile_stream_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_topk_output_tile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_topk_output_tile_stream_native_parity_present"
    )
    assert trackb_front[
        "token_topk_output_tile_stream_native_parity_decision"
    ] == "token_topk_output_tile_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_topk_output_tile_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_topk_output_tile_stream_air_artifact_parity"
    )
    assert str(token_topk_output_tile_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_topk_output_tile_artifact_parity = _write_json(
        tmp_path / "token_topk_output_tile_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_topk_output_tile_stream_artifact_parity"
            ),
            "decision": (
                "token_topk_output_tile_stream_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_topk_output_tile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_topk_output_tile_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_topk_output_tile_stream_artifact_parity_decision"
    ] == "token_topk_output_tile_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_topk_output_tile_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_topk_output_tile_stream_same_window_q2_speed_packet"
    )
    assert str(token_topk_output_tile_artifact_parity) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_topk_output_tile_speed_packet = _write_json(
        tmp_path / "token_topk_output_tile_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_topk_output_tile_stream_speed_packet"
            ),
            "decision": "reject_token_topk_output_tile_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_topk_output_tile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "token_topk_output_tile_stream_speed_rejected"
    assert trackb_front["token_topk_output_tile_stream_speed_packet_decision"] == (
        "reject_token_topk_output_tile_stream_speed_path"
    )
    assert trackb_front["next"] == (
        "change_token_topk_output_tile_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_topk_output_tile_stream_layout_or_kernel_family"
    )
    assert str(token_topk_output_tile_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_block_output_group_design = _write_json(
        tmp_path / "token_block_output_group_design.json",
        {
            **base,
            "decision": (
                "token_block_output_group_stream_ready_for_source_structure_probe"
            ),
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_block_output_group_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_block_output_group_streams"
                ),
            },
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_topk_output_tile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "token_block_output_group_stream_design_ready"
    assert trackb_front["token_block_output_group_stream_design_decision"] == (
        "token_block_output_group_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["token_block_output_group_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_block_output_group_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_block_output_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_block_output_group_stream_source_guardrail"
    )
    assert str(token_block_output_group_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_block_output_group_missing_source_guard = _write_json(
        tmp_path / "token_block_output_group_missing_source_guard.json",
        {
            **output_group_present_payload,
            "kblock_output_group_route_fused_stream_guardrail": {
                "decision": (
                    "kblock_output_group_route_fused_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "route_tile_output_swizzle_stream_guardrail": {
                "decision": (
                    "route_tile_output_swizzle_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_topk_output_tile_stream_guardrail": {
                "decision": (
                    "token_topk_output_tile_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_block_output_group_stream_guardrail": {
                "decision": "missing_token_block_output_group_stream_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = _build_kblock_route_fused_status(
        token_block_output_group_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_block_output_group_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_block_output_group_stream_source_guardrail_decision"
    ] == "missing_token_block_output_group_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_block_output_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_block_output_group_stream_source_guardrail"
    )
    assert str(token_block_output_group_missing_source_guard) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_block_output_group_present_source_guard = _write_json(
        tmp_path / "token_block_output_group_present_source_guard.json",
        {
            **output_group_present_payload,
            "kblock_output_group_route_fused_stream_guardrail": {
                "decision": (
                    "kblock_output_group_route_fused_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "route_tile_output_swizzle_stream_guardrail": {
                "decision": (
                    "route_tile_output_swizzle_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_topk_output_tile_stream_guardrail": {
                "decision": (
                    "token_topk_output_tile_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "token_block_output_group_stream_guardrail": {
                "decision": (
                    "token_block_output_group_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    report = _build_kblock_route_fused_status(
        token_block_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_block_output_group_stream_source_guardrail_present"
    )
    assert trackb_front[
        "token_block_output_group_stream_source_guardrail_decision"
    ] == "token_block_output_group_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_block_output_group_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_block_output_group_stream_native_parity"
    )

    token_block_output_group_native_parity = _write_json(
        tmp_path / "token_block_output_group_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_block_output_group_stream_native_parity"
            ),
            "decision": "token_block_output_group_stream_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_block_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_block_output_group_stream_native_parity_present"
    )
    assert trackb_front[
        "token_block_output_group_stream_native_parity_decision"
    ] == "token_block_output_group_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_block_output_group_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_block_output_group_stream_air_artifact_parity"
    )
    assert str(token_block_output_group_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_block_output_group_artifact_parity = _write_json(
        tmp_path / "token_block_output_group_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_block_output_group_stream_artifact_parity"
            ),
            "decision": "token_block_output_group_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_block_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_block_output_group_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_block_output_group_stream_artifact_parity_decision"
    ] == "token_block_output_group_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_block_output_group_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_block_output_group_stream_same_window_q2_speed_packet"
    )
    assert str(token_block_output_group_artifact_parity) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_block_output_group_speed_packet = _write_json(
        tmp_path / "token_block_output_group_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_block_output_group_stream_speed_packet"
            ),
            "decision": "reject_token_block_output_group_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_block_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "token_block_output_group_stream_speed_rejected"
    assert trackb_front["token_block_output_group_stream_speed_packet_decision"] == (
        "reject_token_block_output_group_stream_speed_path"
    )
    assert trackb_front["next"] == (
        "change_token_block_output_group_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_block_output_group_stream_layout_or_kernel_family"
    )
    assert str(token_block_output_group_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_output_stripe_group_design = _write_json(
        tmp_path / "token_output_stripe_group_design.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_trackb_token_output_stripe_group_stream_design"
            ),
            "decision": (
                "token_output_stripe_group_stream_ready_for_source_structure_probe"
            ),
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_output_stripe_group_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_output_stripe_group_streams"
                ),
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    report = _build_kblock_route_fused_status(
        token_block_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "token_output_stripe_group_stream_design_ready"
    assert trackb_front["token_output_stripe_group_stream_design_decision"] == (
        "token_output_stripe_group_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["token_output_stripe_group_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_output_stripe_group_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_output_stripe_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_output_stripe_group_stream_source_guardrail"
    )
    assert str(token_output_stripe_group_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_block_output_group_present_payload = json.loads(
        token_block_output_group_present_source_guard.read_text(encoding="utf-8")
    )
    token_output_stripe_missing_source_guard = _write_json(
        tmp_path / "token_output_stripe_missing_source_guard.json",
        {
            **token_block_output_group_present_payload,
            "token_output_stripe_group_stream_guardrail": {
                "decision": "missing_token_output_stripe_group_stream_source",
                "passes_contract": False,
            },
        },
    )
    report = _build_kblock_route_fused_status(
        token_output_stripe_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_output_stripe_group_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_output_stripe_group_stream_source_guardrail_decision"
    ] == "missing_token_output_stripe_group_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_output_stripe_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_output_stripe_group_stream_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_output_stripe_present_source_guard = _write_json(
        tmp_path / "token_output_stripe_present_source_guard.json",
        {
            **token_block_output_group_present_payload,
            "token_output_stripe_group_stream_guardrail": {
                "decision": (
                    "token_output_stripe_group_stream_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    token_output_stripe_native_parity = _write_json(
        tmp_path / "token_output_stripe_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_output_stripe_group_stream_native_parity"
            ),
            "decision": "token_output_stripe_group_stream_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_output_stripe_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_output_stripe_group_stream_native_parity_present"
    )
    assert trackb_front[
        "token_output_stripe_group_stream_native_parity_decision"
    ] == "token_output_stripe_group_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_output_stripe_group_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_output_stripe_group_stream_air_artifact_parity"
    )
    assert str(token_output_stripe_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_output_stripe_artifact_parity = _write_json(
        tmp_path / "token_output_stripe_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_output_stripe_group_stream_artifact_parity"
            ),
            "decision": "token_output_stripe_group_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_output_stripe_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_output_stripe_group_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_output_stripe_group_stream_artifact_parity_decision"
    ] == "token_output_stripe_group_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_output_stripe_group_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_output_stripe_group_stream_same_window_q2_speed_packet"
    )
    assert str(token_output_stripe_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_output_stripe_speed_packet = _write_json(
        tmp_path / "token_output_stripe_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_output_stripe_group_stream_speed_packet"
            ),
            "decision": "reject_token_output_stripe_group_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_output_stripe_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_output_stripe_group_stream_speed_rejected"
    )
    assert trackb_front[
        "token_output_stripe_group_stream_speed_packet_decision"
    ] == "reject_token_output_stripe_group_stream_speed_path"
    assert trackb_front["next"] == (
        "change_token_output_stripe_group_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_output_stripe_group_stream_layout_or_kernel_family"
    )
    assert str(token_output_stripe_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_expert_output_block_stream_design = _write_json(
        tmp_path / "token_expert_output_block_stream_design.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_trackb_token_expert_output_block_stream_design"
            ),
            "decision": (
                "token_expert_output_block_stream_ready_for_source_structure_probe"
            ),
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_expert_output_block_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_expert_output_block_streams"
                ),
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    report = _build_kblock_route_fused_status(
        token_output_stripe_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "token_expert_output_block_stream_design_ready"
    assert trackb_front["token_expert_output_block_stream_design_decision"] == (
        "token_expert_output_block_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["token_expert_output_block_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_expert_output_block_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_expert_output_block_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_expert_output_block_stream_source_guardrail"
    )
    assert str(token_expert_output_block_stream_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_expert_missing_source_guard = _write_json(
        tmp_path / "token_expert_missing_source_guard.json",
        {
            **token_block_output_group_present_payload,
            "record_type": "glm45_air_e8p_kernel_structure_analysis",
            "token_output_stripe_group_stream_guardrail": {
                "decision": "token_output_stripe_group_stream_source_guardrail_present",
                "passes_contract": True,
            },
            "token_expert_output_block_stream_guardrail": {
                "decision": "missing_token_expert_output_block_stream_source",
                "passes_contract": False,
            },
        },
    )
    report = _build_kblock_route_fused_status(
        token_expert_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_expert_output_block_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_expert_output_block_stream_source_guardrail_decision"
    ] == "missing_token_expert_output_block_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_expert_output_block_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_expert_output_block_stream_source_guardrail"
    )
    assert str(token_expert_missing_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_expert_present_source_guard = _write_json(
        tmp_path / "token_expert_present_source_guard.json",
        {
            **token_block_output_group_present_payload,
            "record_type": "glm45_air_e8p_kernel_structure_analysis",
            "token_output_stripe_group_stream_guardrail": {
                "decision": "token_output_stripe_group_stream_source_guardrail_present",
                "passes_contract": True,
            },
            "token_expert_output_block_stream_guardrail": {
                "decision": (
                    "token_expert_output_block_stream_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    report = _build_kblock_route_fused_status(
        token_expert_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_expert_output_block_stream_source_guardrail_present"
    )
    assert trackb_front[
        "token_expert_output_block_stream_source_guardrail_decision"
    ] == "token_expert_output_block_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_expert_output_block_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_expert_output_block_stream_native_parity"
    )
    assert str(token_expert_present_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_expert_native_parity = _write_json(
        tmp_path / "token_expert_native_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_token_expert_output_block_stream_native_parity"
            ),
            "decision": "token_expert_output_block_stream_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_expert_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_expert_output_block_stream_native_parity_present"
    )
    assert trackb_front[
        "token_expert_output_block_stream_native_parity_decision"
    ] == "token_expert_output_block_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_expert_output_block_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_expert_output_block_stream_air_artifact_parity"
    )
    assert str(token_expert_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_expert_artifact_parity = _write_json(
        tmp_path / "token_expert_artifact_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_token_expert_output_block_stream_artifact_parity"
            ),
            "decision": "token_expert_output_block_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_expert_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_expert_output_block_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_expert_output_block_stream_artifact_parity_decision"
    ] == "token_expert_output_block_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_expert_output_block_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_expert_output_block_stream_same_window_q2_speed_packet"
    )
    assert str(token_expert_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_expert_artifact_parity = _write_json(
        tmp_path / "token_expert_artifact_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_token_expert_output_block_stream_artifact_parity"
            ),
            "decision": "token_expert_output_block_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    token_expert_speed_packet = _write_json(
        tmp_path / "token_expert_speed_packet.json",
        {
            "record_type": (
                "glm45_air_e8p_token_expert_output_block_stream_speed_packet"
            ),
            "decision": "reject_token_expert_output_block_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_expert_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_expert_output_block_stream_speed_rejected"
    )
    assert trackb_front[
        "token_expert_output_block_stream_speed_packet_decision"
    ] == "reject_token_expert_output_block_stream_speed_path"
    assert trackb_front["next"] == (
        "change_token_expert_output_block_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_expert_output_block_stream_layout_or_kernel_family"
    )
    assert str(token_expert_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_kblock_design = _write_json(
        tmp_path / "token_pair_kblock_design.json",
        {
            "record_type": (
                "glm45_air_e8p_trackb_token_pair_kblock_accumulator_stream_design"
            ),
            "decision": (
                "token_pair_kblock_accumulator_stream_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_pair_kblock_accumulator_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_pair_kblock_accumulator_streams"
                ),
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "custom_memory_limit_override": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_expert_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_kblock_accumulator_stream_design_ready"
    )
    assert trackb_front["token_pair_kblock_accumulator_stream_design_decision"] == (
        "token_pair_kblock_accumulator_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["token_pair_kblock_accumulator_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_pair_kblock_accumulator_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_kblock_accumulator_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_kblock_accumulator_stream_source_guardrail"
    )
    assert str(token_pair_kblock_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_missing_source_guard = _write_json(
        tmp_path / "token_pair_missing_source_guard.json",
        {
            **json.loads(token_expert_present_source_guard.read_text(encoding="utf-8")),
            "token_pair_kblock_accumulator_stream_guardrail": {
                "decision": "missing_token_pair_kblock_accumulator_stream_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "implement_token_pair_kblock_accumulator_stream_source_guardrail"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_kblock_accumulator_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_pair_kblock_accumulator_stream_source_guardrail_decision"
    ] == "missing_token_pair_kblock_accumulator_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_kblock_accumulator_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_kblock_accumulator_stream_source_guardrail"
    )
    assert str(token_pair_missing_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_present_source_guard = _write_json(
        tmp_path / "token_pair_present_source_guard.json",
        {
            **json.loads(token_expert_present_source_guard.read_text(encoding="utf-8")),
            "token_pair_kblock_accumulator_stream_guardrail": {
                "decision": (
                    "token_pair_kblock_accumulator_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "prove_token_pair_kblock_accumulator_stream_native_parity"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_kblock_accumulator_stream_source_guardrail_present"
    )
    assert trackb_front[
        "token_pair_kblock_accumulator_stream_source_guardrail_decision"
    ] == "token_pair_kblock_accumulator_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_pair_kblock_accumulator_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_kblock_accumulator_stream_native_parity"
    )
    assert str(token_pair_present_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_native_parity = _write_json(
        tmp_path / "token_pair_native_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_token_pair_kblock_accumulator_stream_native_parity"
            ),
            "decision": "token_pair_kblock_accumulator_stream_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_kblock_accumulator_stream_native_parity_present"
    )
    assert trackb_front[
        "token_pair_kblock_accumulator_stream_native_parity_decision"
    ] == "token_pair_kblock_accumulator_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_pair_kblock_accumulator_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_kblock_accumulator_stream_air_artifact_parity"
    )
    assert str(token_pair_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_artifact_parity = _write_json(
        tmp_path / "token_pair_artifact_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_token_pair_kblock_accumulator_stream_artifact_parity"
            ),
            "decision": "token_pair_kblock_accumulator_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_kblock_accumulator_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_pair_kblock_accumulator_stream_artifact_parity_decision"
    ] == "token_pair_kblock_accumulator_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_pair_kblock_accumulator_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_pair_kblock_accumulator_stream_same_window_q2_speed_packet"
    )
    assert str(token_pair_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_speed_packet = _write_json(
        tmp_path / "token_pair_speed_packet.json",
        {
            "record_type": (
                "glm45_air_e8p_token_pair_kblock_accumulator_stream_speed_packet"
            ),
            "decision": "reject_token_pair_kblock_accumulator_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "custom_memory_limit_override": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_kblock_accumulator_stream_speed_rejected"
    )
    assert trackb_front[
        "token_pair_kblock_accumulator_stream_speed_packet_decision"
    ] == "reject_token_pair_kblock_accumulator_stream_speed_path"
    assert trackb_front["next"] == (
        "change_token_pair_kblock_accumulator_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_pair_kblock_accumulator_stream_layout_or_kernel_family"
    )
    assert str(token_pair_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_output_group_design = _write_json(
        tmp_path / "token_pair_output_group_design.json",
        {
            "record_type": (
                "glm45_air_e8p_trackb_token_pair_output_group_stream_design"
            ),
            "decision": (
                "token_pair_output_group_stream_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_pair_output_group_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_pair_output_group_streams"
                ),
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "custom_memory_limit_override": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == "token_pair_output_group_stream_design_ready"
    assert trackb_front["token_pair_output_group_stream_design_decision"] == (
        "token_pair_output_group_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["token_pair_output_group_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_pair_output_group_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_output_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_output_group_stream_source_guardrail"
    )
    assert str(token_pair_output_group_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_output_group_missing_source_guard = _write_json(
        tmp_path / "token_pair_output_group_missing_source_guard.json",
        {
            **json.loads(token_pair_present_source_guard.read_text(encoding="utf-8")),
            "token_pair_output_group_stream_guardrail": {
                "decision": "missing_token_pair_output_group_stream_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "implement_token_pair_output_group_stream_source_guardrail"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_output_group_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_output_group_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_pair_output_group_stream_source_guardrail_decision"
    ] == "missing_token_pair_output_group_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_output_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_output_group_stream_source_guardrail"
    )
    assert str(token_pair_output_group_missing_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_output_group_present_source_guard = _write_json(
        tmp_path / "token_pair_output_group_present_source_guard.json",
        {
            **json.loads(token_pair_present_source_guard.read_text(encoding="utf-8")),
            "token_pair_output_group_stream_guardrail": {
                "decision": (
                    "token_pair_output_group_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "prove_token_pair_output_group_stream_native_parity"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_output_group_stream_source_guardrail_present"
    )
    assert trackb_front[
        "token_pair_output_group_stream_source_guardrail_decision"
    ] == "token_pair_output_group_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_pair_output_group_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_output_group_stream_native_parity"
    )
    assert str(token_pair_output_group_present_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_output_group_native_parity = _write_json(
        tmp_path / "token_pair_output_group_native_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_token_pair_output_group_stream_native_parity"
            ),
            "decision": "token_pair_output_group_stream_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_output_group_stream_native_parity_present"
    )
    assert trackb_front[
        "token_pair_output_group_stream_native_parity_decision"
    ] == "token_pair_output_group_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_pair_output_group_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_output_group_stream_air_artifact_parity"
    )
    assert str(token_pair_output_group_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_output_group_artifact_parity = _write_json(
        tmp_path / "token_pair_output_group_artifact_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_token_pair_output_group_stream_artifact_parity"
            ),
            "decision": "token_pair_output_group_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_output_group_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_pair_output_group_stream_artifact_parity_decision"
    ] == "token_pair_output_group_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_pair_output_group_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_pair_output_group_stream_same_window_q2_speed_packet"
    )
    assert str(token_pair_output_group_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_output_group_speed_packet = _write_json(
        tmp_path / "token_pair_output_group_speed_packet.json",
        {
            "record_type": (
                "glm45_air_e8p_token_pair_output_group_stream_speed_packet"
            ),
            "decision": "reject_token_pair_output_group_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "custom_memory_limit_override": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_output_group_stream_speed_rejected"
    )
    assert trackb_front[
        "token_pair_output_group_stream_speed_packet_decision"
    ] == "reject_token_pair_output_group_stream_speed_path"
    assert trackb_front["next"] == (
        "change_token_pair_output_group_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_pair_output_group_stream_layout_or_kernel_family"
    )
    assert str(token_pair_output_group_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_output_group_design = _write_json(
        tmp_path / "token_pair_slot_topk_output_group_design.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_trackb_token_pair_slot_topk_output_group_stream_design"
            ),
            "decision": (
                "token_pair_slot_topk_output_group_stream_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_output_group_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_pair_slot_topk_output_group_streams"
                ),
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "custom_memory_limit_override": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_group_stream_design_ready"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_group_stream_design_decision"
    ] == "token_pair_slot_topk_output_group_stream_ready_for_source_structure_probe"
    assert trackb_front[
        "token_pair_slot_topk_output_group_stream_target_kernel_family"
    ] == "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_output_group_stream"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_output_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_output_group_stream_source_guardrail"
    )
    assert str(token_pair_slot_topk_output_group_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_output_group_missing_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_output_group_missing_source_guard.json",
        {
            **json.loads(
                token_pair_output_group_present_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_output_group_stream_guardrail": {
                "decision": (
                    "missing_token_pair_slot_topk_output_group_stream_source"
                ),
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "implement_token_pair_slot_topk_output_group_stream_source_guardrail"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_output_group_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_group_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_group_stream_source_guardrail_decision"
    ] == "missing_token_pair_slot_topk_output_group_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_output_group_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_output_group_stream_source_guardrail"
    )
    assert (
        str(token_pair_slot_topk_output_group_missing_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_output_group_present_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_output_group_present_source_guard.json",
        {
            **json.loads(
                token_pair_output_group_present_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_output_group_stream_guardrail": {
                "decision": (
                    "token_pair_slot_topk_output_group_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "prove_token_pair_slot_topk_output_group_stream_native_parity"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_group_stream_source_guardrail_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_group_stream_source_guardrail_decision"
    ] == "token_pair_slot_topk_output_group_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_output_group_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_output_group_stream_native_parity"
    )
    assert (
        str(token_pair_slot_topk_output_group_present_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_output_group_native_parity = _write_json(
        tmp_path / "token_pair_slot_topk_output_group_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_output_group_stream_native_parity"
            ),
            "decision": (
                "token_pair_slot_topk_output_group_stream_native_parity_pass"
            ),
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
            "custom_memory_limit_override": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_group_stream_native_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_group_stream_native_parity_decision"
    ] == "token_pair_slot_topk_output_group_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_output_group_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_output_group_stream_air_artifact_parity"
    )
    assert (
        str(token_pair_slot_topk_output_group_native_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_output_group_artifact_parity = _write_json(
        tmp_path / "token_pair_slot_topk_output_group_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_output_group_stream_artifact_parity"
            ),
            "decision": (
                "token_pair_slot_topk_output_group_stream_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
            "custom_memory_limit_override": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_group_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_group_stream_artifact_parity_decision"
    ] == "token_pair_slot_topk_output_group_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_pair_slot_topk_output_group_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_pair_slot_topk_output_group_stream_same_window_q2_speed_packet"
    )
    assert (
        str(token_pair_slot_topk_output_group_artifact_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_output_group_speed_packet = _write_json(
        tmp_path / "token_pair_slot_topk_output_group_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_output_group_stream_speed_packet"
            ),
            "decision": (
                "reject_token_pair_slot_topk_output_group_stream_speed_path"
            ),
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "custom_memory_limit_override": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_group_stream_speed_rejected"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_group_stream_speed_packet_decision"
    ] == "reject_token_pair_slot_topk_output_group_stream_speed_path"
    assert trackb_front["next"] == (
        "change_token_pair_slot_topk_output_group_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_pair_slot_topk_output_group_stream_layout_or_kernel_family"
    )
    assert str(token_pair_slot_topk_output_group_speed_packet) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_codeword_group_pipeline_design = _write_json(
        tmp_path / "token_pair_slot_topk_codeword_group_pipeline_design.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_trackb_token_pair_slot_topk_codeword_group_pipeline_design"
            ),
            "decision": (
                "token_pair_slot_topk_codeword_group_pipeline_ready_for_source_structure_probe"
            ),
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "custom_memory_limit_override": False,
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_codeword_group_pipeline"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_pair_slot_topk_codeword_group_pipelines"
                ),
                "dispatch_grid": (
                    "token_pairs_x_pair_slots_x_topk_slots_x_codeword_groups_x_output_stripes_x_k_blocks"
                ),
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_output_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_codeword_group_pipeline_design_ready"
    )
    assert trackb_front[
        "token_pair_slot_topk_codeword_group_pipeline_design_decision"
    ] == (
        "token_pair_slot_topk_codeword_group_pipeline_ready_for_source_structure_probe"
    )
    assert trackb_front[
        "token_pair_slot_topk_codeword_group_pipeline_target_kernel_family"
    ] == (
        "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_codeword_group_pipeline"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_codeword_group_pipeline"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_codeword_group_pipeline_source_guardrail"
    )
    assert str(token_pair_slot_topk_codeword_group_pipeline_design) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_codeword_group_missing_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_codeword_group_missing_source_guard.json",
        {
            **json.loads(
                token_pair_slot_topk_output_group_present_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_codeword_group_pipeline_guardrail": {
                "decision": (
                    "missing_token_pair_slot_topk_codeword_group_pipeline_source"
                ),
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "implement_token_pair_slot_topk_codeword_group_pipeline_source_guardrail"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_codeword_group_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_missing"
    )
    assert trackb_front[
        "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_decision"
    ] == "missing_token_pair_slot_topk_codeword_group_pipeline_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_codeword_group_pipeline"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_codeword_group_pipeline_source_guardrail"
    )
    assert (
        str(token_pair_slot_topk_codeword_group_missing_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_codeword_group_present_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_codeword_group_present_source_guard.json",
        {
            **json.loads(
                token_pair_slot_topk_codeword_group_missing_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_codeword_group_pipeline_guardrail": {
                "decision": (
                    "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "prove_token_pair_slot_topk_codeword_group_pipeline_native_parity"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_codeword_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_decision"
    ] == "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_codeword_group_pipeline_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity"
    )
    assert (
        str(token_pair_slot_topk_codeword_group_present_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_codeword_group_native_parity = _write_json(
        tmp_path / "token_pair_slot_topk_codeword_group_native_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_codeword_group_pipeline_native_parity"
            ),
            "decision": (
                "token_pair_slot_topk_codeword_group_pipeline_native_parity_pass"
            ),
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_codeword_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_codeword_group_pipeline_native_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_codeword_group_pipeline_native_parity_decision"
    ] == "token_pair_slot_topk_codeword_group_pipeline_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_codeword_group_pipeline_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_codeword_group_pipeline_air_artifact_parity"
    )
    assert (
        str(token_pair_slot_topk_codeword_group_native_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_codeword_group_artifact_parity = _write_json(
        tmp_path / "token_pair_slot_topk_codeword_group_artifact_parity.json",
        {
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_codeword_group_pipeline_artifact_parity"
            ),
            "decision": (
                "token_pair_slot_topk_codeword_group_pipeline_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_codeword_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_codeword_group_pipeline_artifact_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_codeword_group_pipeline_artifact_parity_decision"
    ] == "token_pair_slot_topk_codeword_group_pipeline_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_pair_slot_topk_codeword_group_pipeline_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_pair_slot_topk_codeword_group_pipeline_same_window_q2_speed_packet"
    )
    assert (
        str(token_pair_slot_topk_codeword_group_artifact_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_codeword_group_speed_packet = _write_json(
        tmp_path / "token_pair_slot_topk_codeword_group_speed_packet.json",
        {
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_codeword_group_pipeline_speed_packet"
            ),
            "decision": (
                "reject_token_pair_slot_topk_codeword_group_pipeline_speed_path"
            ),
            "same_window_q2_speed_packet": True,
            "all_output_shape_equivalent": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "custom_memory_limit_override": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_codeword_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_codeword_group_pipeline_speed_rejected"
    )
    assert trackb_front[
        "token_pair_slot_topk_codeword_group_pipeline_speed_packet_decision"
    ] == "reject_token_pair_slot_topk_codeword_group_pipeline_speed_path"
    assert trackb_front["next"] == (
        "change_token_pair_slot_topk_codeword_group_pipeline_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_pair_slot_topk_codeword_group_pipeline_layout_or_kernel_family"
    )
    assert str(token_pair_slot_topk_codeword_group_speed_packet) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_scale_slot_broadcast_design = _write_json(
        tmp_path / "token_pair_slot_topk_scale_slot_broadcast_design.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design"
            ),
            "decision": (
                "token_pair_slot_topk_scale_slot_broadcast_stream_ready_for_source_structure_probe"
            ),
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "custom_memory_limit_override": False,
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_scale_slot_broadcast_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_pair_slot_topk_scale_slot_broadcast_streams"
                ),
                "dispatch_grid": (
                    "token_pairs_x_pair_slots_x_topk_slots_x_scale_slots_x_output_stripes_x_k_blocks_x_codewords"
                ),
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_codeword_group_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_scale_slot_broadcast_stream_design_ready"
    )
    assert trackb_front[
        "token_pair_slot_topk_scale_slot_broadcast_stream_design_decision"
    ] == (
        "token_pair_slot_topk_scale_slot_broadcast_stream_ready_for_source_structure_probe"
    )
    assert trackb_front[
        "token_pair_slot_topk_scale_slot_broadcast_stream_target_kernel_family"
    ] == (
        "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_scale_slot_broadcast_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_scale_slot_broadcast_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail"
    )
    assert str(token_pair_slot_topk_scale_slot_broadcast_design) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_scale_slot_broadcast_missing_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_scale_slot_broadcast_missing_source_guard.json",
        {
            **json.loads(
                token_pair_slot_topk_codeword_group_present_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_scale_slot_broadcast_stream_guardrail": {
                "decision": (
                    "missing_token_pair_slot_topk_scale_slot_broadcast_stream_source"
                ),
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "implement_token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_scale_slot_broadcast_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_decision"
    ] == "missing_token_pair_slot_topk_scale_slot_broadcast_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_scale_slot_broadcast_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail"
    )
    assert (
        str(token_pair_slot_topk_scale_slot_broadcast_missing_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_scale_slot_broadcast_present_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_scale_slot_broadcast_present_source_guard.json",
        {
            **json.loads(
                token_pair_slot_topk_codeword_group_present_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_scale_slot_broadcast_stream_guardrail": {
                "decision": (
                    "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "prove_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_scale_slot_broadcast_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_decision"
    ] == "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity"
    )
    assert (
        str(token_pair_slot_topk_scale_slot_broadcast_present_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_scale_slot_broadcast_native_parity = _write_json(
        tmp_path / "token_pair_slot_topk_scale_slot_broadcast_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity"
            ),
            "decision": (
                "token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_pass"
            ),
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_scale_slot_broadcast_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_decision"
    ] == "token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_scale_slot_broadcast_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_air_artifact_parity"
    )
    assert (
        str(token_pair_slot_topk_scale_slot_broadcast_native_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_scale_slot_broadcast_artifact_parity = _write_json(
        tmp_path / "token_pair_slot_topk_scale_slot_broadcast_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity"
            ),
            "decision": (
                "token_pair_slot_topk_scale_slot_broadcast_stream_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_scale_slot_broadcast_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_decision"
    ] == "token_pair_slot_topk_scale_slot_broadcast_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_pair_slot_topk_scale_slot_broadcast_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_same_window_q2_speed_packet"
    )
    assert (
        str(token_pair_slot_topk_scale_slot_broadcast_artifact_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_scale_slot_broadcast_speed_packet = _write_json(
        tmp_path / "token_pair_slot_topk_scale_slot_broadcast_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet"
            ),
            "decision": (
                "reject_token_pair_slot_topk_scale_slot_broadcast_stream_speed_path"
            ),
            "same_window_q2_speed_packet": True,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_scale_slot_broadcast_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_scale_slot_broadcast_stream_speed_rejected"
    )
    assert trackb_front[
        "token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_decision"
    ] == "reject_token_pair_slot_topk_scale_slot_broadcast_stream_speed_path"
    assert trackb_front["next"] == (
        "change_token_pair_slot_topk_scale_slot_broadcast_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_layout_or_kernel_family"
    )
    assert (
        str(token_pair_slot_topk_scale_slot_broadcast_speed_packet)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_route_bucket_design = _write_json(
        tmp_path / "token_pair_slot_topk_route_bucket_codeword_reduce_design.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design"
            ),
            "decision": (
                "token_pair_slot_topk_route_bucket_codeword_reduce_ready_for_source_structure_probe"
            ),
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "custom_memory_limit_override": False,
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_route_bucket_codeword_reduce"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_pair_slot_topk_route_bucket_codeword_reductions"
                ),
                "dispatch_grid": (
                    "route_buckets_x_token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles_x_codeword_tiles"
                ),
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_scale_slot_broadcast_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_route_bucket_codeword_reduce_design_ready"
    )
    assert trackb_front[
        "token_pair_slot_topk_route_bucket_codeword_reduce_design_decision"
    ] == (
        "token_pair_slot_topk_route_bucket_codeword_reduce_ready_for_source_structure_probe"
    )
    assert trackb_front[
        "token_pair_slot_topk_route_bucket_codeword_reduce_target_kernel_family"
    ] == (
        "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_route_bucket_codeword_reduce"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_route_bucket_codeword_reduce"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail"
    )
    assert (
        str(token_pair_slot_topk_route_bucket_design) in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_route_bucket_missing_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_route_bucket_missing_source_guard.json",
        {
            **json.loads(
                token_pair_slot_topk_scale_slot_broadcast_present_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_route_bucket_codeword_reduce_guardrail": {
                "decision": (
                    "missing_token_pair_slot_topk_route_bucket_codeword_reduce_source"
                ),
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "implement_token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_route_bucket_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_missing"
    )
    assert trackb_front[
        "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_decision"
    ] == "missing_token_pair_slot_topk_route_bucket_codeword_reduce_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_route_bucket_codeword_reduce"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail"
    )
    assert (
        str(token_pair_slot_topk_route_bucket_missing_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_route_bucket_present_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_route_bucket_present_source_guard.json",
        {
            **json.loads(
                token_pair_slot_topk_route_bucket_missing_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_route_bucket_codeword_reduce_guardrail": {
                "decision": (
                    "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "next_track_b_hypothesis": (
                "prove_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_route_bucket_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_decision"
    ] == "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity"
    )
    assert (
        str(token_pair_slot_topk_route_bucket_present_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_route_bucket_native_parity = _write_json(
        tmp_path / "token_pair_slot_topk_route_bucket_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity"
            ),
            "decision": (
                "token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_pass"
            ),
            "passes_native_parity": True,
            "native_parity_claim": True,
            "artifact_parity_claim": False,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_route_bucket_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_decision"
    ] == "token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_route_bucket_codeword_reduce_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_air_artifact_parity"
    )
    assert (
        str(token_pair_slot_topk_route_bucket_native_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_route_bucket_artifact_parity = _write_json(
        tmp_path / "token_pair_slot_topk_route_bucket_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity"
            ),
            "decision": (
                "token_pair_slot_topk_route_bucket_codeword_reduce_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_route_bucket_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_decision"
    ] == "token_pair_slot_topk_route_bucket_codeword_reduce_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_pair_slot_topk_route_bucket_codeword_reduce_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_same_window_q2_speed_packet"
    )
    assert (
        str(token_pair_slot_topk_route_bucket_artifact_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_route_bucket_speed_packet = _write_json(
        tmp_path / "token_pair_slot_topk_route_bucket_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet"
            ),
            "decision": (
                "reject_token_pair_slot_topk_route_bucket_codeword_reduce_speed_path"
            ),
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "all_lane_s_pass": False,
            "all_parity_pass": False,
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_route_bucket_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_route_bucket_codeword_reduce_speed_rejected"
    )
    assert trackb_front[
        "token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_decision"
    ] == "reject_token_pair_slot_topk_route_bucket_codeword_reduce_speed_path"
    assert trackb_front["next"] == (
        "change_token_pair_slot_topk_route_bucket_codeword_reduce_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_layout_or_kernel_family"
    )
    assert str(token_pair_slot_topk_route_bucket_speed_packet) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_kblock_microtile_design = _write_json(
        tmp_path / "token_pair_slot_topk_kblock_microtile_stream_design.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_trackb_token_pair_slot_topk_kblock_microtile_stream_design"
            ),
            "decision": (
                "token_pair_slot_topk_kblock_microtile_stream_ready_for_source_structure_probe"
            ),
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "custom_memory_limit_override": False,
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_kblock_microtile_stream"
                ),
                "storage_constraint": (
                    "compressed_e8p_token_pair_slot_topk_kblock_microtile_streams"
                ),
                "dispatch_grid": (
                    "token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles"
                ),
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_route_bucket_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_kblock_microtile_stream_design_ready"
    )
    assert trackb_front[
        "token_pair_slot_topk_kblock_microtile_stream_design_decision"
    ] == "token_pair_slot_topk_kblock_microtile_stream_ready_for_source_structure_probe"
    assert trackb_front[
        "token_pair_slot_topk_kblock_microtile_stream_target_kernel_family"
    ] == "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_kblock_microtile_stream"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_kblock_microtile_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_kblock_microtile_stream_source_guardrail"
    )
    assert str(token_pair_slot_topk_kblock_microtile_design) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_kblock_microtile_missing_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_kblock_microtile_missing_source_guard.json",
        {
            **json.loads(
                token_pair_slot_topk_route_bucket_present_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_kblock_microtile_stream_guardrail": {
                "decision": (
                    "missing_token_pair_slot_topk_kblock_microtile_stream_source"
                ),
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
                "required_next_features": [
                    "add_token_pair_slot_topk_kblock_microtile_stream_native_source_guardrail"
                ],
                "rejected_next_steps": [
                    "do_not_benchmark_token_pair_slot_topk_kblock_microtile_stream_before_source_guardrail"
                ],
            },
            "next_track_b_hypothesis": (
                "implement_token_pair_slot_topk_kblock_microtile_stream_source_guardrail"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_kblock_microtile_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_decision"
    ] == "missing_token_pair_slot_topk_kblock_microtile_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_kblock_microtile_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_kblock_microtile_stream_source_guardrail"
    )
    assert (
        str(token_pair_slot_topk_kblock_microtile_missing_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_kblock_microtile_present_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_kblock_microtile_present_source_guard.json",
        {
            **json.loads(
                token_pair_slot_topk_kblock_microtile_missing_source_guard.read_text(
                    encoding="utf-8"
                )
            ),
            "token_pair_slot_topk_kblock_microtile_stream_guardrail": {
                "decision": (
                    "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
                "required_next_features": [
                    "prove_token_pair_slot_topk_kblock_microtile_stream_native_parity"
                ],
                "rejected_next_steps": [
                    "do_not_claim_speed_from_source_guardrail"
                ],
            },
            "next_track_b_hypothesis": (
                "prove_token_pair_slot_topk_kblock_microtile_stream_native_parity"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_kblock_microtile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_decision"
    ] == "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_present"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_kblock_microtile_stream_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity"
    )
    assert (
        str(token_pair_slot_topk_kblock_microtile_present_source_guard)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_kblock_microtile_native_parity = _write_json(
        tmp_path / "token_pair_slot_topk_kblock_microtile_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_kblock_microtile_stream_native_parity"
            ),
            "decision": (
                "token_pair_slot_topk_kblock_microtile_stream_native_parity_pass"
            ),
            "passes_native_parity": True,
            "native_parity_claim": True,
            "artifact_parity_claim": False,
            "speed_claim": False,
            "custom_memory_limit_override": False,
            "next_track_b_hypothesis": (
                "prove_token_pair_slot_topk_kblock_microtile_stream_air_artifact_parity"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_kblock_microtile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
        token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
            token_pair_slot_topk_kblock_microtile_native_parity
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_kblock_microtile_stream_native_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_kblock_microtile_stream_native_parity_decision"
    ] == "token_pair_slot_topk_kblock_microtile_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_kblock_microtile_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_kblock_microtile_stream_air_artifact_parity"
    )
    assert (
        str(token_pair_slot_topk_kblock_microtile_native_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_kblock_microtile_artifact_parity = _write_json(
        tmp_path / "token_pair_slot_topk_kblock_microtile_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_kblock_microtile_stream_artifact_parity"
            ),
            "decision": (
                "token_pair_slot_topk_kblock_microtile_stream_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
            "custom_memory_limit_override": False,
            "next_track_b_hypothesis": (
                "run_token_pair_slot_topk_kblock_microtile_stream_same_window_q2_speed_packet"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_kblock_microtile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
        token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
            token_pair_slot_topk_kblock_microtile_native_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path=(
            token_pair_slot_topk_kblock_microtile_artifact_parity
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_kblock_microtile_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_kblock_microtile_stream_artifact_parity_decision"
    ] == "token_pair_slot_topk_kblock_microtile_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_pair_slot_topk_kblock_microtile_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_pair_slot_topk_kblock_microtile_stream_same_window_q2_speed_packet"
    )
    assert (
        str(token_pair_slot_topk_kblock_microtile_artifact_parity)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_kblock_microtile_speed_packet = _write_json(
        tmp_path / "token_pair_slot_topk_kblock_microtile_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_kblock_microtile_stream_speed_packet"
            ),
            "decision": (
                "reject_token_pair_slot_topk_kblock_microtile_stream_speed_path"
            ),
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "all_parity_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "custom_memory_limit_override": False,
            "next_track_b_hypothesis": (
                "change_token_pair_slot_topk_kblock_microtile_stream_layout_or_kernel_family"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_kblock_microtile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
        token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
            token_pair_slot_topk_kblock_microtile_native_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path=(
            token_pair_slot_topk_kblock_microtile_artifact_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_speed_packet_path=(
            token_pair_slot_topk_kblock_microtile_speed_packet
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_kblock_microtile_stream_speed_rejected"
    )
    assert trackb_front[
        "token_pair_slot_topk_kblock_microtile_stream_speed_packet_decision"
    ] == "reject_token_pair_slot_topk_kblock_microtile_stream_speed_path"
    assert trackb_front["next"] == (
        "change_token_pair_slot_topk_kblock_microtile_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_pair_slot_topk_kblock_microtile_stream_layout_or_kernel_family"
    )
    assert (
        str(token_pair_slot_topk_kblock_microtile_speed_packet)
        in trackb_front["evidence"]
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    token_pair_slot_topk_output_tile_fused_design = _write_json(
        tmp_path / "token_pair_slot_topk_output_tile_fused_design.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_trackb_token_pair_slot_topk_output_tile_fused_stream_design"
            ),
            "decision": (
                "token_pair_slot_topk_output_tile_fused_stream_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_output_tile_fused_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
            "custom_memory_limit_override": False,
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_token_pair_slot_topk_output_tile_fused_stream"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        token_pair_slot_topk_kblock_microtile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
        token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
            token_pair_slot_topk_kblock_microtile_native_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path=(
            token_pair_slot_topk_kblock_microtile_artifact_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_speed_packet_path=(
            token_pair_slot_topk_kblock_microtile_speed_packet
        ),
        token_pair_slot_topk_output_tile_fused_stream_design_path=(
            token_pair_slot_topk_output_tile_fused_design
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_tile_fused_stream_design_ready"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_tile_fused_stream_design_decision"
    ] == "token_pair_slot_topk_output_tile_fused_stream_ready_for_source_structure_probe"
    assert trackb_front["token_pair_slot_topk_output_tile_fused_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_output_tile_fused_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_output_tile_fused_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_output_tile_fused_stream_source_guardrail"
    )
    assert str(token_pair_slot_topk_output_tile_fused_design) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_tile_missing_source_payload = json.loads(
        token_pair_slot_topk_kblock_microtile_present_source_guard.read_text(
            encoding="utf-8"
        )
    )
    output_tile_missing_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_output_tile_fused_missing_source_guard.json",
        {
            **output_tile_missing_source_payload,
            "token_pair_slot_topk_output_tile_fused_stream_guardrail": {
                "decision": (
                    "missing_token_pair_slot_topk_output_tile_fused_stream_source"
                ),
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
                "required_next_features": [
                    "add_token_pair_slot_topk_output_tile_fused_stream_native_source_guardrail"
                ],
                "rejected_next_steps": [
                    "do_not_benchmark_token_pair_slot_topk_output_tile_fused_stream_before_source_guardrail"
                ],
            },
            "next_track_b_hypothesis": (
                "implement_token_pair_slot_topk_output_tile_fused_stream_source_guardrail"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        output_tile_missing_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
        token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
            token_pair_slot_topk_kblock_microtile_native_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path=(
            token_pair_slot_topk_kblock_microtile_artifact_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_speed_packet_path=(
            token_pair_slot_topk_kblock_microtile_speed_packet
        ),
        token_pair_slot_topk_output_tile_fused_stream_design_path=(
            token_pair_slot_topk_output_tile_fused_design
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_missing"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_decision"
    ] == "missing_token_pair_slot_topk_output_tile_fused_stream_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_pair_slot_topk_output_tile_fused_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_pair_slot_topk_output_tile_fused_stream_source_guardrail"
    )
    assert str(token_pair_slot_topk_output_tile_fused_design) in trackb_front[
        "evidence"
    ]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_tile_present_source_guard = _write_json(
        tmp_path / "token_pair_slot_topk_output_tile_fused_present_source_guard.json",
        {
            **output_tile_missing_source_payload,
            "token_pair_slot_topk_output_tile_fused_stream_guardrail": {
                "decision": (
                    "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_present"
                ),
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
                "required_next_features": [
                    "prove_token_pair_slot_topk_output_tile_fused_stream_native_parity"
                ],
                "rejected_next_steps": [],
            },
            "next_track_b_hypothesis": (
                "prove_token_pair_slot_topk_output_tile_fused_stream_native_parity"
            ),
        },
    )
    output_tile_native_parity = _write_json(
        tmp_path / "token_pair_slot_topk_output_tile_fused_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_output_tile_fused_stream_native_parity"
            ),
            "decision": (
                "token_pair_slot_topk_output_tile_fused_stream_native_parity_pass"
            ),
            "passes_native_parity": True,
            "native_parity_claim": True,
            "artifact_parity_claim": False,
            "speed_claim": False,
            "custom_memory_limit_override": False,
            "next_track_b_hypothesis": (
                "prove_token_pair_slot_topk_output_tile_fused_stream_air_artifact_parity"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        output_tile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
        token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
            token_pair_slot_topk_kblock_microtile_native_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path=(
            token_pair_slot_topk_kblock_microtile_artifact_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_speed_packet_path=(
            token_pair_slot_topk_kblock_microtile_speed_packet
        ),
        token_pair_slot_topk_output_tile_fused_stream_design_path=(
            token_pair_slot_topk_output_tile_fused_design
        ),
        token_pair_slot_topk_output_tile_fused_stream_native_parity_path=(
            output_tile_native_parity
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_tile_fused_stream_native_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_tile_fused_stream_native_parity_decision"
    ] == "token_pair_slot_topk_output_tile_fused_stream_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_token_pair_slot_topk_output_tile_fused_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_pair_slot_topk_output_tile_fused_stream_air_artifact_parity"
    )
    assert str(output_tile_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_tile_artifact_parity = _write_json(
        tmp_path / "token_pair_slot_topk_output_tile_fused_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_output_tile_fused_stream_artifact_parity"
            ),
            "decision": (
                "token_pair_slot_topk_output_tile_fused_stream_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
            "custom_memory_limit_override": False,
            "next_track_b_hypothesis": (
                "run_token_pair_slot_topk_output_tile_fused_stream_same_window_q2_speed_packet"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        output_tile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
        token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
            token_pair_slot_topk_kblock_microtile_native_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path=(
            token_pair_slot_topk_kblock_microtile_artifact_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_speed_packet_path=(
            token_pair_slot_topk_kblock_microtile_speed_packet
        ),
        token_pair_slot_topk_output_tile_fused_stream_design_path=(
            token_pair_slot_topk_output_tile_fused_design
        ),
        token_pair_slot_topk_output_tile_fused_stream_native_parity_path=(
            output_tile_native_parity
        ),
        token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path=(
            output_tile_artifact_parity
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_tile_fused_stream_artifact_parity_present"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_tile_fused_stream_artifact_parity_decision"
    ] == "token_pair_slot_topk_output_tile_fused_stream_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_token_pair_slot_topk_output_tile_fused_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_pair_slot_topk_output_tile_fused_stream_same_window_q2_speed_packet"
    )
    assert str(output_tile_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    output_tile_speed_packet = _write_json(
        tmp_path / "token_pair_slot_topk_output_tile_fused_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_token_pair_slot_topk_output_tile_fused_stream_speed_packet"
            ),
            "decision": (
                "reject_token_pair_slot_topk_output_tile_fused_stream_speed_path"
            ),
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "all_parity_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
            "custom_memory_limit_override": False,
            "next_track_b_hypothesis": (
                "change_token_pair_slot_topk_output_tile_fused_stream_layout_or_kernel_family"
            ),
        },
    )
    report = _build_kblock_route_fused_status(
        output_tile_present_source_guard,
        kblock_route_fused_native_parity,
        kblock_route_fused_artifact_parity,
        kblock_route_fused_speed_packet,
        route_tile_output_swizzle_design,
        route_tile_output_swizzle_native_parity,
        route_tile_output_swizzle_artifact_parity,
        route_tile_output_swizzle_speed_packet,
        token_topk_output_tile_design,
        token_topk_output_tile_native_parity,
        token_topk_output_tile_artifact_parity,
        token_topk_output_tile_speed_packet,
        token_block_output_group_design,
        token_block_output_group_native_parity,
        token_block_output_group_artifact_parity,
        token_block_output_group_speed_packet,
        token_output_stripe_group_design,
        token_output_stripe_native_parity,
        token_output_stripe_artifact_parity,
        token_output_stripe_speed_packet,
        token_expert_output_block_stream_design,
        token_expert_native_parity,
        token_expert_artifact_parity,
        token_expert_speed_packet,
        token_pair_kblock_design,
        token_pair_native_parity,
        token_pair_artifact_parity,
        token_pair_speed_packet,
        token_pair_output_group_design,
        token_pair_output_group_native_parity,
        token_pair_output_group_artifact_parity,
        token_pair_output_group_speed_packet,
        token_pair_slot_topk_output_group_design,
        token_pair_slot_topk_output_group_native_parity,
        token_pair_slot_topk_output_group_artifact_parity,
        token_pair_slot_topk_output_group_speed_packet,
        token_pair_slot_topk_codeword_group_pipeline_design,
        token_pair_slot_topk_codeword_group_native_parity,
        token_pair_slot_topk_codeword_group_artifact_parity,
        token_pair_slot_topk_codeword_group_speed_packet,
        token_pair_slot_topk_scale_slot_broadcast_design,
        token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_native_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            token_pair_slot_topk_scale_slot_broadcast_artifact_parity
        ),
        token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            token_pair_slot_topk_scale_slot_broadcast_speed_packet
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            token_pair_slot_topk_route_bucket_design
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            token_pair_slot_topk_route_bucket_native_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            token_pair_slot_topk_route_bucket_artifact_parity
        ),
        token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            token_pair_slot_topk_route_bucket_speed_packet
        ),
        token_pair_slot_topk_kblock_microtile_stream_design_path=(
            token_pair_slot_topk_kblock_microtile_design
        ),
        token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
            token_pair_slot_topk_kblock_microtile_native_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path=(
            token_pair_slot_topk_kblock_microtile_artifact_parity
        ),
        token_pair_slot_topk_kblock_microtile_stream_speed_packet_path=(
            token_pair_slot_topk_kblock_microtile_speed_packet
        ),
        token_pair_slot_topk_output_tile_fused_stream_design_path=(
            token_pair_slot_topk_output_tile_fused_design
        ),
        token_pair_slot_topk_output_tile_fused_stream_native_parity_path=(
            output_tile_native_parity
        ),
        token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path=(
            output_tile_artifact_parity
        ),
        token_pair_slot_topk_output_tile_fused_stream_speed_packet_path=(
            output_tile_speed_packet
        ),
    )
    trackb_front = report["fronts"]["workstream2_trackb"]

    assert trackb_front["status"] == (
        "token_pair_slot_topk_output_tile_fused_stream_speed_rejected"
    )
    assert trackb_front[
        "token_pair_slot_topk_output_tile_fused_stream_speed_packet_decision"
    ] == "reject_token_pair_slot_topk_output_tile_fused_stream_speed_path"
    assert trackb_front["next"] == (
        "change_token_pair_slot_topk_output_tile_fused_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_pair_slot_topk_output_tile_fused_stream_layout_or_kernel_family"
    )
    assert str(output_tile_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def _build_token_cohort_mma_status_report(
    tmp_path: Path,
    mma_guardrail: dict,
) -> tuple[dict, Path, Path]:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "token_cohort_codeword_stream_guardrail": {
                "decision": "token_cohort_codeword_stream_source_guardrail_present",
                "passes_contract": True,
            },
            "token_cohort_mma_codeword_tile_guardrail": mma_guardrail,
        },
    )
    token_cohort_design = _write_json(
        tmp_path / "token_cohort_design.json",
        {
            **base,
            "decision": "token_cohort_codeword_stream_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_cohort_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    token_cohort_native_parity = _write_json(
        tmp_path / "token_cohort_native_parity.json",
        {
            **base,
            "decision": "token_cohort_codeword_stream_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    token_cohort_artifact_parity = _write_json(
        tmp_path / "token_cohort_artifact_parity.json",
        {
            **base,
            "decision": "token_cohort_codeword_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    token_cohort_speed = _write_json(
        tmp_path / "token_cohort_speed.json",
        {
            **base,
            "decision": "reject_token_cohort_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    token_cohort_mma_design = _write_json(
        tmp_path / "token_cohort_mma_design.json",
        {
            **base,
            "decision": "token_cohort_mma_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_token_cohort_mma_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_cohort_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=token_cohort_design,
        trackb_token_cohort_codeword_stream_native_parity_path=(
            token_cohort_native_parity
        ),
        trackb_token_cohort_codeword_stream_artifact_parity_path=(
            token_cohort_artifact_parity
        ),
        trackb_token_cohort_codeword_stream_speed_packet_path=token_cohort_speed,
        trackb_token_cohort_mma_codeword_tile_design_path=(
            token_cohort_mma_design
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    return report, source_guard, token_cohort_mma_design


def test_five_front_status_routes_from_current_local_evidence(tmp_path: Path) -> None:
    cli = _load_cli()
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {
            "record_type": "glm45_air_workstream1_free_levers_fullsplit_verdict",
            "decision": "reject_free_p1_levers_metric_backed",
            "heavy_slice_rule_satisfied": True,
            "full_split_record_count_satisfied": True,
            "train_router_kd_1": {
                "decision": "reject_metric_flat_fullsplit_router_kd",
                "compose_triggered": False,
                "global_top1_compose_triggered": False,
                "route_top1_compose_triggered": False,
                "max_global_top1_delta": -0.000037202380952283676,
                "max_route_domain_top1_delta": 0.0,
            },
            "next_recommendation": "continue P1 with a different top1 mechanism",
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    fixed_bump = _write_json(
        tmp_path / "fixed_bump.json",
        {
            "record_type": "glm45_air_workstream1_bump_fixed_loader_verdict",
            "decision": (
                "bump_lm_head_embed_fixed_loader_metric_backed_not_promotable"
            ),
            "implementation_bug_fixed": True,
            "non_expert_precision_exhausted": False,
            "quality_promotable": False,
            "residency_promotable": False,
            "lane_s_valid": False,
            "next_recommendation": "do_not_compose_bump",
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            "record_type": "glm45_air_e8p_trackb_frontier_summary",
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}, {"evidence": "lower_bound"}],
            "next_required_features": ["materially_different_rhs_layout_or_kernel_family"],
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    trackb_design = _write_json(
        tmp_path / "trackb_design.json",
        {
            "record_type": "glm45_air_e8p_trackb_route_slot_codeword_stream_design",
            "decision": "route_slot_codeword_stream_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_slot_codeword_stream"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "trackb_source_guard.json",
        {
            "record_type": "glm45_air_e8p_kernel_structure_analysis",
            "route_slot_codeword_stream_guardrail": {
                "decision": "missing_route_slot_codeword_stream_source",
                "passes_contract": False,
            },
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {
            "record_type": "qwen_family_gate_check",
            "family_gate_pass": True,
            "family_gate_status": "qwen_family_gate_ready",
            "missing_requirements": [],
        },
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "record_type": "glm52_vq_validation",
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "record_type": "glm52_vq_validation",
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {
            "record_type": "glm45_air_single_host_teacher_source_audit",
            "source_ready_for_single_host_export": False,
            "local_export_candidate_ready": False,
            "recommended_model_path": None,
            "missing_shards": ["model-00047-of-00047.safetensors"],
            "cache_rows_generated": False,
            "peer2_used": False,
            "rdma_jaccl_touched": False,
        },
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        workstream1_fixed_bump_path=fixed_bump,
        trackb_path=trackb,
        trackb_design_path=trackb_design,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    assert report["record_type"] == "keep_five_front_current_status"
    assert report["mission_complete"] is False
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert report["transport_boundary"] == "peer2_approved_not_used_in_this_slice"
    assert report["fronts"]["workstream1_p1_free_levers"]["status"] == (
        "metric_backed_free_levers_rejected"
    )
    assert report["fronts"]["workstream1_p1_free_levers"][
        "full_split_record_count_satisfied"
    ] is True
    assert report["fronts"]["workstream1_p1_free_levers"][
        "implementation_bug_fixed"
    ] is True
    assert report["fronts"]["workstream1_p1_free_levers"]["router_kd_decision"] == (
        "reject_metric_flat_fullsplit_router_kd"
    )
    assert report["fronts"]["workstream1_p1_free_levers"][
        "router_kd_compose_triggered"
    ] is False
    assert report["fronts"]["workstream1_p1_free_levers"][
        "router_kd_max_global_top1_delta"
    ] == pytest.approx(-0.000037202380952283676)
    assert report["fronts"]["workstream1_p1_free_levers"][
        "router_kd_max_route_domain_top1_delta"
    ] == pytest.approx(0.0)
    assert report["fronts"]["workstream1_p1_free_levers"][
        "fixed_loader_bump_evidence"
    ] == str(fixed_bump)
    assert report["fronts"]["workstream1_p1_free_levers"][
        "fixed_loader_bump_decision"
    ] == "bump_lm_head_embed_fixed_loader_metric_backed_not_promotable"
    assert report["fronts"]["workstream1_p1_free_levers"]["next"] == (
        "do_not_compose_bump"
    )
    assert report["fronts"]["workstream2_trackb"]["status"] == (
        "source_structure_guardrail_missing"
    )
    assert report["fronts"]["workstream2_trackb"]["source_guardrail_decision"] == (
        "missing_route_slot_codeword_stream_source"
    )
    assert report["fronts"]["workstream2_trackb"]["candidate_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_route_slot_codeword_stream"
    )
    assert report["fronts"]["workstream3_qwen"]["status"] == "first_family_gate_green"
    assert report["fronts"]["workstream3_glm52"]["status"] == (
        "source_artifact_gap_diagnosed"
    )
    assert report["fronts"]["workstream4_cache"]["status"] == (
        "blocked_on_local_high_bit_source"
    )
    assert report["fronts"]["workstream5_hygiene"]["status"] == (
        "publication_packet_draft_present"
    )
    assert "repair_qwen_first_family_gate" not in report["next_local_slices"]
    assert "implement_trackb_route_slot_codeword_stream_source_guardrail" in report[
        "next_local_slices"
    ]


def test_five_front_status_routes_corrected_w1_bump_verdict(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {
            **base,
            "record_type": "glm45_air_workstream1_bump_fixed_loader_verdict",
            "decision": (
                "bump_lm_head_embed_fixed_loader_metric_backed_not_promotable"
            ),
            "heavy_slice_rule_satisfied": True,
            "implementation_bug_fixed": True,
            "non_expert_precision_exhausted": False,
            "quality_promotable": False,
            "residency_promotable": False,
            "lane_s_valid": False,
            "next_recommendation": "do_not_compose_bump",
        },
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": (
                "track_b_current_frontier_requires_new_rhs_or_kernel_family"
            ),
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    w1 = report["fronts"]["workstream1_p1_free_levers"]
    assert w1["status"] == "bump_lm_head_embed_fixed_loader_measured_not_promotable"
    assert w1["decision"] == (
        "bump_lm_head_embed_fixed_loader_metric_backed_not_promotable"
    )
    assert w1["implementation_bug_fixed"] is True
    assert w1["non_expert_precision_exhausted"] is False
    assert w1["quality_promotable"] is False
    assert w1["residency_promotable"] is False
    assert w1["lane_s_valid"] is False


def test_five_front_status_routes_trackb_token_cohort_design_after_gate(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {
            **base,
            "decision": "reject_free_p1_levers_metric_backed",
            "heavy_slice_rule_satisfied": True,
        },
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
            "next_required_features": [
                "materially_different_rhs_layout_or_kernel_family"
            ],
        },
    )
    next_family_gate = _write_json(
        tmp_path / "next_family_gate.json",
        {
            **base,
            "decision": "block_stale_trackb_plan_variant_requires_materially_new_family_design",
            "stale_plan_variant_blocked": True,
            "next_track_b_action": (
                "write_materially_new_rhs_or_kernel_family_design_before_source_or_speed_work"
            ),
        },
    )
    token_cohort_design = _write_json(
        tmp_path / "token_cohort_design.json",
        {
            **base,
            "record_type": "glm45_air_e8p_trackb_token_cohort_codeword_stream_design",
            "decision": "token_cohort_codeword_stream_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_token_cohort_codeword_stream"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_cohort_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=next_family_gate,
        trackb_token_cohort_codeword_stream_design_path=token_cohort_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_codeword_stream_design_ready"
    assert trackb_front["token_cohort_codeword_stream_design_decision"] == (
        "token_cohort_codeword_stream_ready_for_source_structure_probe"
    )
    assert trackb_front["token_cohort_codeword_stream_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_cohort_codeword_stream"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_cohort_codeword_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_cohort_codeword_stream_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_token_cohort_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {
            **base,
            "decision": "reject_free_p1_levers_metric_backed",
            "heavy_slice_rule_satisfied": True,
        },
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "token_cohort_codeword_stream_guardrail": {
                "decision": "missing_token_cohort_codeword_stream_source",
                "passes_contract": False,
            },
        },
    )
    next_family_gate = _write_json(
        tmp_path / "next_family_gate.json",
        {
            **base,
            "decision": "block_stale_trackb_plan_variant_requires_materially_new_family_design",
            "stale_plan_variant_blocked": True,
        },
    )
    token_cohort_design = _write_json(
        tmp_path / "token_cohort_design.json",
        {
            **base,
            "decision": "token_cohort_codeword_stream_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_token_cohort_codeword_stream"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_cohort_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=next_family_gate,
        trackb_token_cohort_codeword_stream_design_path=token_cohort_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_source_guardrail_missing"
    assert trackb_front["token_cohort_codeword_stream_source_guardrail_decision"] == (
        "missing_token_cohort_codeword_stream_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_cohort_codeword_stream"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_cohort_codeword_stream_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_token_cohort_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {
            **base,
            "decision": "reject_free_p1_levers_metric_backed",
            "heavy_slice_rule_satisfied": True,
        },
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "token_cohort_codeword_stream_guardrail": {
                "decision": "token_cohort_codeword_stream_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    token_cohort_design = _write_json(
        tmp_path / "token_cohort_design.json",
        {
            **base,
            "decision": "token_cohort_codeword_stream_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_cohort_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    token_cohort_native_parity = _write_json(
        tmp_path / "token_cohort_native_parity.json",
        {
            **base,
            "decision": "token_cohort_codeword_stream_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=token_cohort_design,
        trackb_token_cohort_codeword_stream_native_parity_path=(
            token_cohort_native_parity
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_native_parity_present"
    assert trackb_front["token_cohort_codeword_stream_native_parity_decision"] == (
        "token_cohort_codeword_stream_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_token_cohort_codeword_stream_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_cohort_codeword_stream_air_artifact_parity"
    )
    assert str(token_cohort_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_token_cohort_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {
            **base,
            "decision": "reject_free_p1_levers_metric_backed",
            "heavy_slice_rule_satisfied": True,
        },
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "token_cohort_codeword_stream_guardrail": {
                "decision": "token_cohort_codeword_stream_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    token_cohort_design = _write_json(
        tmp_path / "token_cohort_design.json",
        {
            **base,
            "decision": "token_cohort_codeword_stream_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_cohort_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    token_cohort_native_parity = _write_json(
        tmp_path / "token_cohort_native_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_token_cohort_codeword_stream_native_parity",
            "decision": "token_cohort_codeword_stream_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    token_cohort_artifact_parity = _write_json(
        tmp_path / "token_cohort_artifact_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_token_cohort_codeword_stream_artifact_parity",
            "decision": "token_cohort_codeword_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=token_cohort_design,
        trackb_token_cohort_codeword_stream_native_parity_path=(
            token_cohort_native_parity
        ),
        trackb_token_cohort_codeword_stream_artifact_parity_path=(
            token_cohort_artifact_parity
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_artifact_parity_present"
    assert trackb_front["token_cohort_codeword_stream_artifact_parity_decision"] == (
        "token_cohort_codeword_stream_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_token_cohort_codeword_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_cohort_codeword_stream_same_window_q2_speed_packet"
    )
    assert str(token_cohort_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_token_cohort_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "token_cohort_codeword_stream_guardrail": {
                "decision": "token_cohort_codeword_stream_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    token_cohort_design = _write_json(
        tmp_path / "token_cohort_design.json",
        {
            **base,
            "decision": "token_cohort_codeword_stream_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_cohort_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    token_cohort_native_parity = _write_json(
        tmp_path / "token_cohort_native_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_token_cohort_codeword_stream_native_parity",
            "decision": "token_cohort_codeword_stream_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    token_cohort_artifact_parity = _write_json(
        tmp_path / "token_cohort_artifact_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_token_cohort_codeword_stream_artifact_parity",
            "decision": "token_cohort_codeword_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    token_cohort_speed = _write_json(
        tmp_path / "token_cohort_speed.json",
        {
            **base,
            "record_type": "glm45_air_e8p_token_cohort_codeword_stream_speed_packet",
            "decision": "reject_token_cohort_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=token_cohort_design,
        trackb_token_cohort_codeword_stream_native_parity_path=(
            token_cohort_native_parity
        ),
        trackb_token_cohort_codeword_stream_artifact_parity_path=(
            token_cohort_artifact_parity
        ),
        trackb_token_cohort_codeword_stream_speed_packet_path=token_cohort_speed,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_speed_rejected"
    assert trackb_front["token_cohort_codeword_stream_speed_packet_decision"] == (
        "reject_token_cohort_codeword_stream_speed_path"
    )
    assert trackb_front["next"] == (
        "change_token_cohort_codeword_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_cohort_codeword_stream_layout_or_kernel_family"
    )
    assert str(token_cohort_speed) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_token_cohort_mma_design(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "token_cohort_codeword_stream_guardrail": {
                "decision": "token_cohort_codeword_stream_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    token_cohort_design = _write_json(
        tmp_path / "token_cohort_design.json",
        {
            **base,
            "decision": "token_cohort_codeword_stream_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_cohort_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    token_cohort_native_parity = _write_json(
        tmp_path / "token_cohort_native_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_token_cohort_codeword_stream_native_parity",
            "decision": "token_cohort_codeword_stream_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    token_cohort_artifact_parity = _write_json(
        tmp_path / "token_cohort_artifact_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_token_cohort_codeword_stream_artifact_parity",
            "decision": "token_cohort_codeword_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    token_cohort_speed = _write_json(
        tmp_path / "token_cohort_speed.json",
        {
            **base,
            "record_type": "glm45_air_e8p_token_cohort_codeword_stream_speed_packet",
            "decision": "reject_token_cohort_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    token_cohort_mma_design = _write_json(
        tmp_path / "token_cohort_mma_design.json",
        {
            **base,
            "record_type": "glm45_air_e8p_trackb_token_cohort_mma_codeword_tile_design",
            "decision": "token_cohort_mma_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_token_cohort_mma_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_token_cohort_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=token_cohort_design,
        trackb_token_cohort_codeword_stream_native_parity_path=(
            token_cohort_native_parity
        ),
        trackb_token_cohort_codeword_stream_artifact_parity_path=(
            token_cohort_artifact_parity
        ),
        trackb_token_cohort_codeword_stream_speed_packet_path=token_cohort_speed,
        trackb_token_cohort_mma_codeword_tile_design_path=(
            token_cohort_mma_design
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_mma_design_ready"
    assert trackb_front["token_cohort_mma_design_decision"] == (
        "token_cohort_mma_codeword_tile_ready_for_source_structure_probe"
    )
    assert trackb_front["token_cohort_mma_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_token_cohort_mma_codeword_tile"
    )
    assert trackb_front["token_cohort_codeword_stream_speed_packet_decision"] == (
        "reject_token_cohort_codeword_stream_speed_path"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_cohort_mma_codeword_tile"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_cohort_mma_codeword_tile_source_guardrail"
    )
    assert str(token_cohort_mma_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_token_cohort_mma_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    report, source_guard, token_cohort_mma_design = _build_token_cohort_mma_status_report(
        tmp_path,
        {
            "decision": "missing_token_cohort_mma_codeword_tile_source",
            "passes_contract": False,
        },
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_mma_source_guardrail_missing"
    assert trackb_front["token_cohort_mma_design_decision"] == (
        "token_cohort_mma_codeword_tile_ready_for_source_structure_probe"
    )
    assert trackb_front["token_cohort_mma_source_guardrail_decision"] == (
        "missing_token_cohort_mma_codeword_tile_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_token_cohort_mma_codeword_tile"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_token_cohort_mma_codeword_tile_source_guardrail"
    )
    assert str(source_guard) in trackb_front["evidence"]
    assert str(token_cohort_mma_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_token_cohort_mma_native_parity_after_source_guardrail(
    tmp_path: Path,
) -> None:
    report, source_guard, _token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_mma_source_guardrail_present"
    assert trackb_front["token_cohort_mma_source_guardrail_decision"] == (
        "token_cohort_mma_codeword_tile_source_guardrail_present"
    )
    assert trackb_front["next"] == (
        "prove_token_cohort_mma_codeword_tile_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_cohort_mma_codeword_tile_native_parity"
    )
    assert str(source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_token_cohort_mma_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    native_parity = _write_json(
        tmp_path / "token_cohort_mma_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_token_cohort_mma_codeword_tile_native_parity",
            "decision": "token_cohort_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=native_parity,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_mma_native_parity_present"
    assert trackb_front["token_cohort_mma_native_parity_decision"] == (
        "token_cohort_mma_codeword_tile_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_token_cohort_mma_codeword_tile_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_token_cohort_mma_codeword_tile_air_artifact_parity"
    )
    assert str(native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_token_cohort_mma_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    native_parity = _write_json(
        tmp_path / "token_cohort_mma_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_token_cohort_mma_codeword_tile_native_parity",
            "decision": "token_cohort_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "token_cohort_mma_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_token_cohort_mma_codeword_tile_artifact_parity",
            "decision": "token_cohort_mma_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=native_parity,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=artifact_parity,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_mma_artifact_parity_present"
    assert trackb_front["token_cohort_mma_artifact_parity_decision"] == (
        "token_cohort_mma_codeword_tile_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_token_cohort_mma_codeword_tile_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_token_cohort_mma_codeword_tile_same_window_q2_speed_packet"
    )
    assert str(artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_token_cohort_mma_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    native_parity = _write_json(
        tmp_path / "token_cohort_mma_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_token_cohort_mma_codeword_tile_native_parity",
            "decision": "token_cohort_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "token_cohort_mma_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_token_cohort_mma_codeword_tile_artifact_parity",
            "decision": "token_cohort_mma_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    speed_packet = _write_json(
        tmp_path / "token_cohort_mma_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_token_cohort_mma_codeword_tile_speed_packet",
            "decision": "reject_token_cohort_mma_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=native_parity,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=artifact_parity,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=speed_packet,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "token_cohort_mma_speed_rejected"
    assert trackb_front["token_cohort_mma_speed_packet_decision"] == (
        "reject_token_cohort_mma_codeword_tile_speed_path"
    )
    assert trackb_front["next"] == (
        "change_token_cohort_mma_codeword_tile_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_token_cohort_mma_codeword_tile_layout_or_kernel_family"
    )
    assert str(speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_output_stationary_design_after_token_cohort_mma_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    output_stationary_design = _write_json(
        tmp_path / "output_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_output_stationary_codeword_tile_design",
            "decision": "output_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_output_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_output_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    native_parity = _write_json(
        tmp_path / "token_cohort_mma_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "token_cohort_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "token_cohort_mma_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "token_cohort_mma_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
        },
    )
    speed_packet = _write_json(
        tmp_path / "token_cohort_mma_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "reject_token_cohort_mma_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=native_parity,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=artifact_parity,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=speed_packet,
        trackb_output_stationary_codeword_tile_design_path=output_stationary_design,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "output_stationary_codeword_tile_design_ready"
    assert trackb_front["output_stationary_codeword_tile_design_decision"] == (
        "output_stationary_codeword_tile_ready_for_source_structure_probe"
    )
    assert trackb_front["output_stationary_codeword_tile_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_output_stationary_codeword_tile"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_output_stationary_codeword_tile"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_output_stationary_codeword_tile_source_guardrail"
    )
    assert str(output_stationary_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_output_stationary_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    output_stationary_design = _write_json(
        tmp_path / "output_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_output_stationary_codeword_tile_design",
            "decision": "output_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_output_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_output_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["output_stationary_codeword_tile_guardrail"] = {
        "decision": "missing_output_stationary_codeword_tile_source",
        "passes_contract": False,
        "kernel_present": False,
        "primitive_binding_present": False,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")
    native_parity = _write_json(
        tmp_path / "token_cohort_mma_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "token_cohort_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "token_cohort_mma_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "token_cohort_mma_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
        },
    )
    speed_packet = _write_json(
        tmp_path / "token_cohort_mma_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "reject_token_cohort_mma_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=native_parity,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=artifact_parity,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=speed_packet,
        trackb_output_stationary_codeword_tile_design_path=output_stationary_design,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "output_stationary_source_guardrail_missing"
    assert trackb_front["output_stationary_codeword_tile_source_guardrail_decision"] == (
        "missing_output_stationary_codeword_tile_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_output_stationary_codeword_tile"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_output_stationary_codeword_tile_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_output_stationary_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    output_stationary_design = _write_json(
        tmp_path / "output_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_output_stationary_codeword_tile_design",
            "decision": "output_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_output_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_output_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "output_stationary_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_native_parity",
            "decision": "output_stationary_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["output_stationary_codeword_tile_guardrail"] = {
        "decision": "output_stationary_codeword_tile_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")
    token_cohort_mma_native_parity = _write_json(
        tmp_path / "token_cohort_mma_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "token_cohort_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=(
            token_cohort_mma_native_parity
        ),
        trackb_output_stationary_codeword_tile_design_path=output_stationary_design,
        trackb_output_stationary_codeword_tile_native_parity_path=native_parity,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "output_stationary_native_parity_present"
    assert trackb_front["output_stationary_codeword_tile_native_parity_decision"] == (
        "output_stationary_codeword_tile_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_output_stationary_codeword_tile_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_output_stationary_codeword_tile_air_artifact_parity"
    )
    assert str(native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_output_stationary_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    output_stationary_design = _write_json(
        tmp_path / "output_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_output_stationary_codeword_tile_design",
            "decision": "output_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_output_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_output_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "output_stationary_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_native_parity",
            "decision": "output_stationary_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "output_stationary_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_artifact_parity",
            "decision": "output_stationary_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["output_stationary_codeword_tile_guardrail"] = {
        "decision": "output_stationary_codeword_tile_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")
    token_cohort_mma_native_parity = _write_json(
        tmp_path / "token_cohort_mma_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "token_cohort_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=(
            token_cohort_mma_native_parity
        ),
        trackb_output_stationary_codeword_tile_design_path=output_stationary_design,
        trackb_output_stationary_codeword_tile_native_parity_path=native_parity,
        trackb_output_stationary_codeword_tile_artifact_parity_path=artifact_parity,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "output_stationary_artifact_parity_present"
    assert trackb_front["output_stationary_codeword_tile_artifact_parity_decision"] == (
        "output_stationary_codeword_tile_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_output_stationary_codeword_tile_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_output_stationary_codeword_tile_same_window_q2_speed_packet"
    )
    assert str(artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_output_stationary_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    output_stationary_design = _write_json(
        tmp_path / "output_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_output_stationary_codeword_tile_design",
            "decision": "output_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_output_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_output_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "output_stationary_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_native_parity",
            "decision": "output_stationary_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "output_stationary_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_artifact_parity",
            "decision": "output_stationary_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    speed_packet = _write_json(
        tmp_path / "output_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_speed_packet",
            "decision": "reject_output_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["output_stationary_codeword_tile_guardrail"] = {
        "decision": "output_stationary_codeword_tile_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")
    token_cohort_mma_native_parity = _write_json(
        tmp_path / "token_cohort_mma_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "token_cohort_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=(
            token_cohort_mma_native_parity
        ),
        trackb_output_stationary_codeword_tile_design_path=output_stationary_design,
        trackb_output_stationary_codeword_tile_native_parity_path=native_parity,
        trackb_output_stationary_codeword_tile_artifact_parity_path=artifact_parity,
        trackb_output_stationary_codeword_tile_speed_packet_path=speed_packet,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "output_stationary_speed_rejected"
    assert trackb_front["output_stationary_codeword_tile_speed_packet_decision"] == (
        "reject_output_stationary_codeword_tile_speed_path"
    )
    assert trackb_front["next"] == (
        "change_output_stationary_codeword_tile_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_output_stationary_codeword_tile_layout_or_kernel_family"
    )
    assert str(speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_input_stationary_design_after_output_stationary_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    input_stationary_design = _write_json(
        tmp_path / "input_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_input_stationary_codeword_tile_design",
            "decision": "input_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_input_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_input_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    output_stationary_design = _write_json(
        tmp_path / "output_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_output_stationary_codeword_tile_design",
            "decision": "output_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_output_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_output_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "output_stationary_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_native_parity",
            "decision": "output_stationary_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "output_stationary_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_artifact_parity",
            "decision": "output_stationary_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    speed_packet = _write_json(
        tmp_path / "output_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_speed_packet",
            "decision": "reject_output_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": False,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["output_stationary_codeword_tile_guardrail"] = {
        "decision": "output_stationary_codeword_tile_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")
    token_cohort_mma_native_parity = _write_json(
        tmp_path / "token_cohort_mma_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "decision": "token_cohort_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=(
            token_cohort_mma_native_parity
        ),
        trackb_output_stationary_codeword_tile_design_path=output_stationary_design,
        trackb_output_stationary_codeword_tile_native_parity_path=native_parity,
        trackb_output_stationary_codeword_tile_artifact_parity_path=artifact_parity,
        trackb_output_stationary_codeword_tile_speed_packet_path=speed_packet,
        trackb_input_stationary_codeword_tile_design_path=input_stationary_design,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "input_stationary_codeword_tile_design_ready"
    assert trackb_front["input_stationary_codeword_tile_design_decision"] == (
        "input_stationary_codeword_tile_ready_for_source_structure_probe"
    )
    assert trackb_front["input_stationary_codeword_tile_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_input_stationary_codeword_tile"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_input_stationary_codeword_tile"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_input_stationary_codeword_tile_source_guardrail"
    )
    assert str(input_stationary_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_input_stationary_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    input_stationary_design = _write_json(
        tmp_path / "input_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_input_stationary_codeword_tile_design",
            "decision": "input_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_input_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_input_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    speed_packet = _write_json(
        tmp_path / "output_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_speed_packet",
            "decision": "reject_output_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": False,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["input_stationary_codeword_tile_guardrail"] = {
        "decision": "missing_input_stationary_codeword_tile_source",
        "passes_contract": False,
        "kernel_present": False,
        "primitive_binding_present": False,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_output_stationary_codeword_tile_speed_packet_path=speed_packet,
        trackb_input_stationary_codeword_tile_design_path=input_stationary_design,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "input_stationary_source_guardrail_missing"
    assert trackb_front["input_stationary_codeword_tile_source_guardrail_decision"] == (
        "missing_input_stationary_codeword_tile_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_input_stationary_codeword_tile"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_input_stationary_codeword_tile_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_input_stationary_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    input_stationary_design = _write_json(
        tmp_path / "input_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_input_stationary_codeword_tile_design",
            "decision": "input_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_input_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_input_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "input_stationary_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_native_parity",
            "decision": "input_stationary_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    speed_packet = _write_json(
        tmp_path / "output_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_speed_packet",
            "decision": "reject_output_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": False,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["input_stationary_codeword_tile_guardrail"] = {
        "decision": "input_stationary_codeword_tile_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_output_stationary_codeword_tile_speed_packet_path=speed_packet,
        trackb_input_stationary_codeword_tile_design_path=input_stationary_design,
        trackb_input_stationary_codeword_tile_native_parity_path=native_parity,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "input_stationary_native_parity_present"
    assert trackb_front["input_stationary_codeword_tile_native_parity_decision"] == (
        "input_stationary_codeword_tile_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_input_stationary_codeword_tile_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_input_stationary_codeword_tile_air_artifact_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_input_stationary_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    input_stationary_design = _write_json(
        tmp_path / "input_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_input_stationary_codeword_tile_design",
            "decision": "input_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_input_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_input_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "input_stationary_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_native_parity",
            "decision": "input_stationary_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "input_stationary_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_artifact_parity",
            "decision": "input_stationary_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    speed_packet = _write_json(
        tmp_path / "output_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_speed_packet",
            "decision": "reject_output_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": False,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["input_stationary_codeword_tile_guardrail"] = {
        "decision": "input_stationary_codeword_tile_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_output_stationary_codeword_tile_speed_packet_path=speed_packet,
        trackb_input_stationary_codeword_tile_design_path=input_stationary_design,
        trackb_input_stationary_codeword_tile_native_parity_path=native_parity,
        trackb_input_stationary_codeword_tile_artifact_parity_path=artifact_parity,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "input_stationary_artifact_parity_present"
    assert trackb_front["input_stationary_codeword_tile_artifact_parity_decision"] == (
        "input_stationary_codeword_tile_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_input_stationary_codeword_tile_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_input_stationary_codeword_tile_same_window_q2_speed_packet"
    )
    assert str(artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_input_stationary_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    input_stationary_design = _write_json(
        tmp_path / "input_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_input_stationary_codeword_tile_design",
            "decision": "input_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_input_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_input_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "input_stationary_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_native_parity",
            "decision": "input_stationary_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "input_stationary_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_artifact_parity",
            "decision": "input_stationary_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    output_speed_packet = _write_json(
        tmp_path / "output_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_output_stationary_codeword_tile_speed_packet",
            "decision": "reject_output_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": False,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    input_speed_packet = _write_json(
        tmp_path / "input_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_speed_packet",
            "decision": "reject_input_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["input_stationary_codeword_tile_guardrail"] = {
        "decision": "input_stationary_codeword_tile_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_output_stationary_codeword_tile_speed_packet_path=output_speed_packet,
        trackb_input_stationary_codeword_tile_design_path=input_stationary_design,
        trackb_input_stationary_codeword_tile_native_parity_path=native_parity,
        trackb_input_stationary_codeword_tile_artifact_parity_path=artifact_parity,
        trackb_input_stationary_codeword_tile_speed_packet_path=input_speed_packet,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "input_stationary_speed_rejected"
    assert trackb_front["input_stationary_codeword_tile_speed_packet_decision"] == (
        "reject_input_stationary_codeword_tile_speed_path"
    )
    assert trackb_front["next"] == (
        "change_input_stationary_codeword_tile_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_input_stationary_codeword_tile_layout_or_kernel_family"
    )
    assert str(input_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_expert_kblock_codeword_factor_reuse_design_after_input_stationary_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    expert_kblock_design = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_expert_kblock_codeword_factor_reuse_design",
            "decision": (
                "expert_kblock_codeword_factor_reuse_ready_for_source_structure_probe"
            ),
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_expert_kblock_codeword_factor_reuse"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_kblock_codeword_factor_reuse"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    input_stationary_design = _write_json(
        tmp_path / "input_stationary_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_input_stationary_codeword_tile_design",
            "decision": "input_stationary_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_input_stationary_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_input_stationary_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    input_speed_packet = _write_json(
        tmp_path / "input_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_speed_packet",
            "decision": "reject_input_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_input_stationary_codeword_tile_design_path=input_stationary_design,
        trackb_input_stationary_codeword_tile_speed_packet_path=input_speed_packet,
        trackb_expert_kblock_codeword_factor_reuse_design_path=expert_kblock_design,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "expert_kblock_codeword_factor_reuse_design_ready"
    assert trackb_front["expert_kblock_codeword_factor_reuse_design_decision"] == (
        "expert_kblock_codeword_factor_reuse_ready_for_source_structure_probe"
    )
    assert trackb_front["expert_kblock_codeword_factor_reuse_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_expert_kblock_codeword_factor_reuse"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_expert_kblock_codeword_factor_reuse"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_expert_kblock_codeword_factor_reuse_source_guardrail"
    )
    assert str(expert_kblock_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_expert_kblock_codeword_factor_reuse_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    expert_kblock_design = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_expert_kblock_codeword_factor_reuse_design",
            "decision": (
                "expert_kblock_codeword_factor_reuse_ready_for_source_structure_probe"
            ),
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_expert_kblock_codeword_factor_reuse"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_kblock_codeword_factor_reuse"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    input_speed_packet = _write_json(
        tmp_path / "input_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_speed_packet",
            "decision": "reject_input_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["expert_kblock_codeword_factor_reuse_guardrail"] = {
        "decision": "missing_expert_kblock_codeword_factor_reuse_source",
        "passes_contract": False,
        "kernel_present": False,
        "primitive_binding_present": False,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_input_stationary_codeword_tile_speed_packet_path=input_speed_packet,
        trackb_expert_kblock_codeword_factor_reuse_design_path=expert_kblock_design,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert (
        trackb_front["status"]
        == "expert_kblock_codeword_factor_reuse_source_guardrail_missing"
    )
    assert trackb_front[
        "expert_kblock_codeword_factor_reuse_source_guardrail_decision"
    ] == "missing_expert_kblock_codeword_factor_reuse_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_expert_kblock_codeword_factor_reuse"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_expert_kblock_codeword_factor_reuse_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_expert_kblock_codeword_factor_reuse_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    expert_kblock_design = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_expert_kblock_codeword_factor_reuse_design",
            "decision": (
                "expert_kblock_codeword_factor_reuse_ready_for_source_structure_probe"
            ),
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_expert_kblock_codeword_factor_reuse"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_kblock_codeword_factor_reuse"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    input_speed_packet = _write_json(
        tmp_path / "input_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_speed_packet",
            "decision": "reject_input_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": (
                "glm45_air_e8p_expert_kblock_codeword_factor_reuse_native_parity"
            ),
            "decision": "expert_kblock_codeword_factor_reuse_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["expert_kblock_codeword_factor_reuse_guardrail"] = {
        "decision": "expert_kblock_codeword_factor_reuse_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_input_stationary_codeword_tile_speed_packet_path=input_speed_packet,
        trackb_expert_kblock_codeword_factor_reuse_design_path=expert_kblock_design,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=native_parity,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert (
        trackb_front["status"]
        == "expert_kblock_codeword_factor_reuse_native_parity_present"
    )
    assert trackb_front[
        "expert_kblock_codeword_factor_reuse_native_parity_decision"
    ] == "expert_kblock_codeword_factor_reuse_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_expert_kblock_codeword_factor_reuse_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_expert_kblock_codeword_factor_reuse_air_artifact_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_expert_kblock_codeword_factor_reuse_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    expert_kblock_design = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_expert_kblock_codeword_factor_reuse_design",
            "decision": (
                "expert_kblock_codeword_factor_reuse_ready_for_source_structure_probe"
            ),
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_expert_kblock_codeword_factor_reuse"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_kblock_codeword_factor_reuse"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    input_speed_packet = _write_json(
        tmp_path / "input_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_speed_packet",
            "decision": "reject_input_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": (
                "glm45_air_e8p_expert_kblock_codeword_factor_reuse_native_parity"
            ),
            "decision": "expert_kblock_codeword_factor_reuse_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": (
                "glm45_air_e8p_expert_kblock_codeword_factor_reuse_artifact_parity"
            ),
            "decision": "expert_kblock_codeword_factor_reuse_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": True,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["expert_kblock_codeword_factor_reuse_guardrail"] = {
        "decision": "expert_kblock_codeword_factor_reuse_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_input_stationary_codeword_tile_speed_packet_path=input_speed_packet,
        trackb_expert_kblock_codeword_factor_reuse_design_path=expert_kblock_design,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=native_parity,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=(
            artifact_parity
        ),
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert (
        trackb_front["status"]
        == "expert_kblock_codeword_factor_reuse_artifact_parity_present"
    )
    assert trackb_front[
        "expert_kblock_codeword_factor_reuse_artifact_parity_decision"
    ] == "expert_kblock_codeword_factor_reuse_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_expert_kblock_codeword_factor_reuse_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_expert_kblock_codeword_factor_reuse_same_window_q2_speed_packet"
    )
    assert str(artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_expert_kblock_codeword_factor_reuse_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    expert_kblock_design = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_design.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_trackb_expert_kblock_codeword_factor_reuse_design",
            "decision": (
                "expert_kblock_codeword_factor_reuse_ready_for_source_structure_probe"
            ),
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_expert_kblock_codeword_factor_reuse"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_kblock_codeword_factor_reuse"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    input_speed_packet = _write_json(
        tmp_path / "input_stationary_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": "glm45_air_e8p_input_stationary_codeword_tile_speed_packet",
            "decision": "reject_input_stationary_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_native_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": (
                "glm45_air_e8p_expert_kblock_codeword_factor_reuse_native_parity"
            ),
            "decision": "expert_kblock_codeword_factor_reuse_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_artifact_parity.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": (
                "glm45_air_e8p_expert_kblock_codeword_factor_reuse_artifact_parity"
            ),
            "decision": "expert_kblock_codeword_factor_reuse_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": True,
        },
    )
    speed_packet = _write_json(
        tmp_path / "expert_kblock_codeword_factor_reuse_speed_packet.json",
        {
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "record_type": (
                "glm45_air_e8p_expert_kblock_codeword_factor_reuse_speed_packet"
            ),
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "all_memory_clean": False,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    report, source_guard, token_cohort_mma_design = (
        _build_token_cohort_mma_status_report(
            tmp_path,
            {
                "decision": "token_cohort_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        )
    )
    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["expert_kblock_codeword_factor_reuse_guardrail"] = {
        "decision": "expert_kblock_codeword_factor_reuse_source_guardrail_present",
        "passes_contract": True,
        "kernel_present": True,
        "primitive_binding_present": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=Path(report["fronts"]["workstream1_p1_free_levers"]["evidence"]),
        trackb_path=tmp_path / "trackb.json",
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=tmp_path
        / "token_cohort_design.json",
        trackb_token_cohort_codeword_stream_native_parity_path=tmp_path
        / "token_cohort_native_parity.json",
        trackb_token_cohort_codeword_stream_artifact_parity_path=tmp_path
        / "token_cohort_artifact_parity.json",
        trackb_token_cohort_codeword_stream_speed_packet_path=tmp_path
        / "token_cohort_speed.json",
        trackb_token_cohort_mma_codeword_tile_design_path=token_cohort_mma_design,
        trackb_input_stationary_codeword_tile_speed_packet_path=input_speed_packet,
        trackb_expert_kblock_codeword_factor_reuse_design_path=expert_kblock_design,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=native_parity,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=(
            artifact_parity
        ),
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=speed_packet,
        qwen_gate_path=tmp_path / "qwen.json",
        glm52_layer3_path=tmp_path / "glm52_l3.json",
        glm52_layer77_path=tmp_path / "glm52_l77.json",
        cache_source_scan_path=tmp_path / "cache.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert (
        trackb_front["status"]
        == "expert_kblock_codeword_factor_reuse_speed_rejected"
    )
    assert trackb_front[
        "expert_kblock_codeword_factor_reuse_speed_packet_decision"
    ] == "reject_expert_kblock_codeword_factor_reuse_speed_path"
    assert trackb_front["next"] == (
        "change_expert_kblock_codeword_factor_reuse_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_expert_kblock_codeword_factor_reuse_layout_or_kernel_family"
    )
    assert str(speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_trackb_after_route_slot_native_parity(tmp_path: Path) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {
            **base,
            "decision": "reject_free_p1_levers_metric_backed",
            "heavy_slice_rule_satisfied": True,
        },
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_design = _write_json(
        tmp_path / "trackb_design.json",
        {
            **base,
            "decision": "route_slot_codeword_stream_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "trackb_source_guard.json",
        {
            **base,
            "route_slot_codeword_stream_guardrail": {
                "decision": "route_slot_codeword_stream_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    trackb_native_parity = _write_json(
        tmp_path / "trackb_native_parity.json",
        {
            **base,
            "decision": "route_slot_codeword_stream_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=trackb_design,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=trackb_native_parity,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "native_parity_present"
    assert trackb_front["native_parity_decision"] == (
        "route_slot_codeword_stream_native_parity_pass"
    )
    assert trackb_front["next"] == "prove_route_slot_codeword_stream_air_artifact_parity"
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_slot_codeword_stream_air_artifact_parity"
    )
    assert str(trackb_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_trackb_after_route_slot_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_design = _write_json(
        tmp_path / "trackb_design.json",
        {
            **base,
            "decision": "route_slot_codeword_stream_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "trackb_source_guard.json",
        {
            **base,
            "route_slot_codeword_stream_guardrail": {
                "decision": "route_slot_codeword_stream_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    trackb_native_parity = _write_json(
        tmp_path / "trackb_native_parity.json",
        {
            **base,
            "decision": "route_slot_codeword_stream_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    trackb_artifact_parity = _write_json(
        tmp_path / "trackb_artifact_parity.json",
        {
            **base,
            "decision": "route_slot_codeword_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=trackb_design,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=trackb_native_parity,
        trackb_artifact_parity_path=trackb_artifact_parity,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "artifact_parity_present"
    assert trackb_front["artifact_parity_decision"] == (
        "route_slot_codeword_stream_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_route_slot_codeword_stream_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_route_slot_codeword_stream_same_window_q2_speed_packet"
    )
    assert str(trackb_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_trackb_after_route_slot_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_design = _write_json(
        tmp_path / "trackb_design.json",
        {
            **base,
            "decision": "route_slot_codeword_stream_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_codeword_stream"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "trackb_source_guard.json",
        {
            **base,
            "route_slot_codeword_stream_guardrail": {
                "decision": "route_slot_codeword_stream_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    trackb_native_parity = _write_json(
        tmp_path / "trackb_native_parity.json",
        {
            **base,
            "decision": "route_slot_codeword_stream_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    trackb_artifact_parity = _write_json(
        tmp_path / "trackb_artifact_parity.json",
        {
            **base,
            "decision": "route_slot_codeword_stream_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=trackb_design,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=trackb_native_parity,
        trackb_artifact_parity_path=trackb_artifact_parity,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "speed_rejected"
    assert trackb_front["speed_packet_decision"] == (
        "reject_route_slot_codeword_stream_speed_path"
    )
    assert trackb_front["next"] == (
        "change_route_slot_codeword_stream_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_route_slot_codeword_stream_layout_or_kernel_family"
    )
    assert str(trackb_speed) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_trackb_after_successor_design(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    successor_design = _write_json(
        tmp_path / "successor.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=successor_design,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "successor_design_ready"
    assert trackb_front["successor_design_decision"] == (
        "route_slot_mma_codeword_tile_ready_for_source_structure_probe"
    )
    assert trackb_front["successor_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_slot_mma_codeword_tile_source_guardrail"
    )
    assert str(successor_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_trackb_after_successor_source_guard_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_slot_mma_codeword_tile_guardrail": {
                "decision": "missing_route_slot_mma_codeword_tile_source",
                "passes_contract": False,
            },
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    successor_design = _write_json(
        tmp_path / "successor.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=successor_design,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "successor_source_guardrail_missing"
    assert trackb_front["successor_source_guardrail_decision"] == (
        "missing_route_slot_mma_codeword_tile_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_slot_mma_codeword_tile_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_trackb_after_successor_source_guard_present(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_slot_mma_codeword_tile_guardrail": {
                "decision": "route_slot_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    successor_design = _write_json(
        tmp_path / "successor.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=successor_design,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "successor_source_guardrail_present"
    assert trackb_front["successor_source_guardrail_decision"] == (
        "route_slot_mma_codeword_tile_source_guardrail_present"
    )
    assert trackb_front["next"] == "prove_route_slot_mma_codeword_tile_native_parity"
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_slot_mma_codeword_tile_native_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_trackb_after_successor_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_slot_mma_codeword_tile_guardrail": {
                "decision": "route_slot_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    trackb_successor_native_parity = _write_json(
        tmp_path / "successor_native_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    successor_design = _write_json(
        tmp_path / "successor.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=successor_design,
        trackb_successor_native_parity_path=trackb_successor_native_parity,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "successor_native_parity_present"
    assert trackb_front["successor_native_parity_decision"] == (
        "route_slot_mma_codeword_tile_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_route_slot_mma_codeword_tile_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_slot_mma_codeword_tile_air_artifact_parity"
    )
    assert str(trackb_successor_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_trackb_after_successor_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_slot_mma_codeword_tile_guardrail": {
                "decision": "route_slot_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    trackb_successor_native_parity = _write_json(
        tmp_path / "successor_native_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    trackb_successor_artifact_parity = _write_json(
        tmp_path / "successor_artifact_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    successor_design = _write_json(
        tmp_path / "successor.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=successor_design,
        trackb_successor_native_parity_path=trackb_successor_native_parity,
        trackb_successor_artifact_parity_path=trackb_successor_artifact_parity,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "successor_artifact_parity_present"
    assert trackb_front["successor_artifact_parity_decision"] == (
        "route_slot_mma_codeword_tile_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_route_slot_mma_codeword_tile_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_route_slot_mma_codeword_tile_same_window_q2_speed_packet"
    )
    assert str(trackb_successor_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_trackb_after_successor_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_slot_mma_codeword_tile_guardrail": {
                "decision": "route_slot_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    trackb_successor_native_parity = _write_json(
        tmp_path / "successor_native_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    trackb_successor_artifact_parity = _write_json(
        tmp_path / "successor_artifact_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )
    trackb_successor_speed = _write_json(
        tmp_path / "successor_speed.json",
        {
            **base,
            "decision": "reject_route_slot_mma_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    successor_design = _write_json(
        tmp_path / "successor.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=successor_design,
        trackb_successor_native_parity_path=trackb_successor_native_parity,
        trackb_successor_artifact_parity_path=trackb_successor_artifact_parity,
        trackb_successor_speed_packet_path=trackb_successor_speed,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "successor_speed_rejected"
    assert trackb_front["successor_speed_packet_decision"] == (
        "reject_route_slot_mma_codeword_tile_speed_path"
    )
    assert trackb_front["next"] == (
        "change_route_slot_mma_codeword_tile_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_route_slot_mma_codeword_tile_layout_or_kernel_family"
    )
    assert str(trackb_successor_speed) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_active_route_tile_design_after_mma_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_slot_mma_codeword_tile_guardrail": {
                "decision": "route_slot_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    trackb_successor_native_parity = _write_json(
        tmp_path / "successor_native_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    trackb_successor_artifact_parity = _write_json(
        tmp_path / "successor_artifact_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    trackb_successor_speed = _write_json(
        tmp_path / "successor_speed.json",
        {
            **base,
            "decision": "reject_route_slot_mma_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    successor_design = _write_json(
        tmp_path / "successor.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_ready_for_source_structure_probe",
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    next_design = _write_json(
        tmp_path / "active_route_tile_design.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
            ),
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_active_route_tile_codeword_outer_product"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=successor_design,
        trackb_successor_native_parity_path=trackb_successor_native_parity,
        trackb_successor_artifact_parity_path=trackb_successor_artifact_parity,
        trackb_successor_speed_packet_path=trackb_successor_speed,
        trackb_next_design_path=next_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "active_route_tile_design_ready"
    assert trackb_front["next_design_decision"] == (
        "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
    )
    assert trackb_front["next_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_active_route_tile_codeword_outer_product"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_active_route_tile_codeword_outer_product_source_guardrail"
    )
    assert str(next_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_active_route_tile_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_slot_mma_codeword_tile_guardrail": {
                "decision": "route_slot_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
            "active_route_tile_codeword_outer_product_guardrail": {
                "decision": "missing_active_route_tile_codeword_outer_product_source",
                "passes_contract": False,
            },
        },
    )
    trackb_successor_native_parity = _write_json(
        tmp_path / "successor_native_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    trackb_successor_artifact_parity = _write_json(
        tmp_path / "successor_artifact_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    trackb_successor_speed = _write_json(
        tmp_path / "successor_speed.json",
        {
            **base,
            "decision": "reject_route_slot_mma_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    successor_design = _write_json(
        tmp_path / "successor.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    next_design = _write_json(
        tmp_path / "active_route_tile_design.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
            ),
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_active_route_tile_codeword_outer_product"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=successor_design,
        trackb_successor_native_parity_path=trackb_successor_native_parity,
        trackb_successor_artifact_parity_path=trackb_successor_artifact_parity,
        trackb_successor_speed_packet_path=trackb_successor_speed,
        trackb_next_design_path=next_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "active_route_tile_source_guardrail_missing"
    assert trackb_front["next_source_guardrail_decision"] == (
        "missing_active_route_tile_codeword_outer_product_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_active_route_tile_codeword_outer_product"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_active_route_tile_codeword_outer_product_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_active_route_tile_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_slot_mma_codeword_tile_guardrail": {
                "decision": "route_slot_mma_codeword_tile_source_guardrail_present",
                "passes_contract": True,
            },
            "active_route_tile_codeword_outer_product_guardrail": {
                "decision": (
                    "active_route_tile_codeword_outer_product_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    active_route_tile_native_parity = _write_json(
        tmp_path / "active_route_tile_native_parity.json",
        {
            **base,
            "decision": "active_route_tile_codeword_outer_product_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    trackb_successor_native_parity = _write_json(
        tmp_path / "successor_native_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    trackb_successor_artifact_parity = _write_json(
        tmp_path / "successor_artifact_parity.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    trackb_successor_speed = _write_json(
        tmp_path / "successor_speed.json",
        {
            **base,
            "decision": "reject_route_slot_mma_codeword_tile_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    trackb_speed = _write_json(
        tmp_path / "trackb_speed.json",
        {
            **base,
            "decision": "reject_route_slot_codeword_stream_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    successor_design = _write_json(
        tmp_path / "successor.json",
        {
            **base,
            "decision": "route_slot_mma_codeword_tile_ready_for_source_structure_probe",
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    next_design = _write_json(
        tmp_path / "active_route_tile_design.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
            ),
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_active_route_tile_codeword_outer_product"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=trackb_speed,
        trackb_successor_design_path=successor_design,
        trackb_successor_native_parity_path=trackb_successor_native_parity,
        trackb_successor_artifact_parity_path=trackb_successor_artifact_parity,
        trackb_successor_speed_packet_path=trackb_successor_speed,
        trackb_next_design_path=next_design,
        trackb_active_route_tile_native_parity_path=active_route_tile_native_parity,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "active_route_tile_native_parity_present"
    assert trackb_front["active_route_tile_native_parity_decision"] == (
        "active_route_tile_codeword_outer_product_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_active_route_tile_codeword_outer_product_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_active_route_tile_codeword_outer_product_air_artifact_parity"
    )
    assert str(active_route_tile_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_active_route_tile_artifact_parity_speed_packet(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "active_route_tile_codeword_outer_product_guardrail": {
                "decision": (
                    "active_route_tile_codeword_outer_product_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    active_route_tile_native_parity = _write_json(
        tmp_path / "active_route_tile_native_parity.json",
        {
            **base,
            "decision": "active_route_tile_codeword_outer_product_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    active_route_tile_artifact_parity = _write_json(
        tmp_path / "active_route_tile_artifact_parity.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
        },
    )
    next_design = _write_json(
        tmp_path / "active_route_tile_design.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
            ),
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_active_route_tile_codeword_outer_product"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=next_design,
        trackb_active_route_tile_native_parity_path=active_route_tile_native_parity,
        trackb_active_route_tile_artifact_parity_path=active_route_tile_artifact_parity,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "active_route_tile_artifact_parity_present"
    assert trackb_front["active_route_tile_artifact_parity_decision"] == (
        "active_route_tile_codeword_outer_product_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_active_route_tile_codeword_outer_product_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_active_route_tile_codeword_outer_product_same_window_q2_speed_packet"
    )
    assert str(active_route_tile_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_active_route_tile_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "active_route_tile_codeword_outer_product_guardrail": {
                "decision": (
                    "active_route_tile_codeword_outer_product_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    active_route_tile_native_parity = _write_json(
        tmp_path / "active_route_tile_native_parity.json",
        {
            **base,
            "decision": "active_route_tile_codeword_outer_product_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    active_route_tile_artifact_parity = _write_json(
        tmp_path / "active_route_tile_artifact_parity.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "speed_claim": False,
        },
    )
    active_route_tile_speed = _write_json(
        tmp_path / "active_route_tile_speed.json",
        {
            **base,
            "decision": "reject_active_route_tile_codeword_outer_product_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    next_design = _write_json(
        tmp_path / "active_route_tile_design.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=next_design,
        trackb_active_route_tile_native_parity_path=active_route_tile_native_parity,
        trackb_active_route_tile_artifact_parity_path=active_route_tile_artifact_parity,
        trackb_active_route_tile_speed_packet_path=active_route_tile_speed,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "active_route_tile_speed_rejected"
    assert trackb_front["active_route_tile_speed_packet_decision"] == (
        "reject_active_route_tile_codeword_outer_product_speed_path"
    )
    assert trackb_front["next"] == (
        "change_active_route_tile_codeword_outer_product_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_active_route_tile_codeword_outer_product_layout_or_kernel_family"
    )
    assert str(active_route_tile_speed) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_expert_cohort_design_after_active_route_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "active_route_tile_codeword_outer_product_guardrail": {
                "decision": (
                    "active_route_tile_codeword_outer_product_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    active_route_tile_artifact_parity = _write_json(
        tmp_path / "active_route_tile_artifact_parity.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "speed_claim": False,
        },
    )
    active_route_tile_speed = _write_json(
        tmp_path / "active_route_tile_speed.json",
        {
            **base,
            "decision": "reject_active_route_tile_codeword_outer_product_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    next_design = _write_json(
        tmp_path / "active_route_tile_design.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=next_design,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=active_route_tile_artifact_parity,
        trackb_active_route_tile_speed_packet_path=active_route_tile_speed,
        trackb_expert_cohort_design_path=expert_cohort_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "expert_cohort_design_ready"
    assert trackb_front["expert_cohort_design_decision"] == (
        "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
    )
    assert trackb_front["expert_cohort_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_expert_cohort_codeword_broadcast"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_expert_cohort_codeword_broadcast_source_guardrail"
    )
    assert str(expert_cohort_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_expert_cohort_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "active_route_tile_codeword_outer_product_guardrail": {
                "decision": (
                    "active_route_tile_codeword_outer_product_source_guardrail_present"
                ),
                "passes_contract": True,
            },
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": "missing_expert_cohort_codeword_broadcast_source",
                "passes_contract": False,
            },
        },
    )
    active_route_tile_artifact_parity = _write_json(
        tmp_path / "active_route_tile_artifact_parity.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "speed_claim": False,
        },
    )
    active_route_tile_speed = _write_json(
        tmp_path / "active_route_tile_speed.json",
        {
            **base,
            "decision": "reject_active_route_tile_codeword_outer_product_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    next_design = _write_json(
        tmp_path / "active_route_tile_design.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=next_design,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=active_route_tile_artifact_parity,
        trackb_active_route_tile_speed_packet_path=active_route_tile_speed,
        trackb_expert_cohort_design_path=expert_cohort_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "expert_cohort_source_guardrail_missing"
    assert trackb_front["expert_cohort_source_guardrail_decision"] == (
        "missing_expert_cohort_codeword_broadcast_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_expert_cohort_codeword_broadcast"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_expert_cohort_codeword_broadcast_source_guardrail"
    )
    assert str(trackb_source_guard) in trackb_front["evidence"]
    assert str(expert_cohort_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_expert_cohort_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "active_route_tile_codeword_outer_product_guardrail": {
                "decision": (
                    "active_route_tile_codeword_outer_product_source_guardrail_present"
                ),
                "passes_contract": True,
            },
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": (
                    "expert_cohort_codeword_broadcast_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    active_route_tile_artifact_parity = _write_json(
        tmp_path / "active_route_tile_artifact_parity.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "speed_claim": False,
        },
    )
    active_route_tile_speed = _write_json(
        tmp_path / "active_route_tile_speed.json",
        {
            **base,
            "decision": "reject_active_route_tile_codeword_outer_product_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    expert_cohort_native_parity = _write_json(
        tmp_path / "expert_cohort_native_parity.json",
        {
            **base,
            "decision": "expert_cohort_codeword_broadcast_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    next_design = _write_json(
        tmp_path / "active_route_tile_design.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=next_design,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=active_route_tile_artifact_parity,
        trackb_active_route_tile_speed_packet_path=active_route_tile_speed,
        trackb_expert_cohort_design_path=expert_cohort_design,
        trackb_expert_cohort_native_parity_path=expert_cohort_native_parity,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "expert_cohort_native_parity_present"
    assert trackb_front["expert_cohort_native_parity_decision"] == (
        "expert_cohort_codeword_broadcast_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_expert_cohort_codeword_broadcast_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_expert_cohort_codeword_broadcast_air_artifact_parity"
    )
    assert str(expert_cohort_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_expert_cohort_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": (
                    "expert_cohort_codeword_broadcast_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    active_route_tile_artifact_parity = _write_json(
        tmp_path / "active_route_tile_artifact_parity.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "speed_claim": False,
        },
    )
    active_route_tile_speed = _write_json(
        tmp_path / "active_route_tile_speed.json",
        {
            **base,
            "decision": "reject_active_route_tile_codeword_outer_product_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    expert_cohort_native_parity = _write_json(
        tmp_path / "expert_cohort_native_parity.json",
        {
            **base,
            "decision": "expert_cohort_codeword_broadcast_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    expert_cohort_artifact_parity = _write_json(
        tmp_path / "expert_cohort_artifact_parity.json",
        {
            **base,
            "decision": "expert_cohort_codeword_broadcast_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=active_route_tile_artifact_parity,
        trackb_active_route_tile_speed_packet_path=active_route_tile_speed,
        trackb_expert_cohort_design_path=expert_cohort_design,
        trackb_expert_cohort_native_parity_path=expert_cohort_native_parity,
        trackb_expert_cohort_artifact_parity_path=expert_cohort_artifact_parity,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "expert_cohort_artifact_parity_present"
    assert trackb_front["expert_cohort_artifact_parity_decision"] == (
        "expert_cohort_codeword_broadcast_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_expert_cohort_codeword_broadcast_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_expert_cohort_codeword_broadcast_same_window_q2_speed_packet"
    )
    assert str(expert_cohort_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_expert_cohort_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": (
                    "expert_cohort_codeword_broadcast_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    active_route_tile_artifact_parity = _write_json(
        tmp_path / "active_route_tile_artifact_parity.json",
        {
            **base,
            "decision": (
                "active_route_tile_codeword_outer_product_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "speed_claim": False,
        },
    )
    active_route_tile_speed = _write_json(
        tmp_path / "active_route_tile_speed.json",
        {
            **base,
            "decision": "reject_active_route_tile_codeword_outer_product_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    expert_cohort_native_parity = _write_json(
        tmp_path / "expert_cohort_native_parity.json",
        {
            **base,
            "decision": "expert_cohort_codeword_broadcast_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    expert_cohort_artifact_parity = _write_json(
        tmp_path / "expert_cohort_artifact_parity.json",
        {
            **base,
            "decision": "expert_cohort_codeword_broadcast_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
        },
    )
    expert_cohort_speed = _write_json(
        tmp_path / "expert_cohort_speed.json",
        {
            **base,
            "decision": "reject_expert_cohort_codeword_broadcast_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=active_route_tile_artifact_parity,
        trackb_active_route_tile_speed_packet_path=active_route_tile_speed,
        trackb_expert_cohort_design_path=expert_cohort_design,
        trackb_expert_cohort_native_parity_path=expert_cohort_native_parity,
        trackb_expert_cohort_artifact_parity_path=expert_cohort_artifact_parity,
        trackb_expert_cohort_speed_packet_path=expert_cohort_speed,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "expert_cohort_speed_rejected"
    assert trackb_front["expert_cohort_speed_packet_decision"] == (
        "reject_expert_cohort_codeword_broadcast_speed_path"
    )
    assert trackb_front["next"] == (
        "change_expert_cohort_codeword_broadcast_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_expert_cohort_codeword_broadcast_layout_or_kernel_family"
    )
    assert str(expert_cohort_speed) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_route_batch_segmented_design(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": (
                    "expert_cohort_codeword_broadcast_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    expert_cohort_speed = _write_json(
        tmp_path / "expert_cohort_speed.json",
        {
            **base,
            "decision": "reject_expert_cohort_codeword_broadcast_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    route_batch_design = _write_json(
        tmp_path / "route_batch_design.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=expert_cohort_design,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=expert_cohort_speed,
        trackb_route_batch_segmented_design_path=route_batch_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "route_batch_segmented_design_ready"
    assert trackb_front["route_batch_segmented_design_decision"] == (
        "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
    )
    assert trackb_front["route_batch_segmented_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_batch_segmented_codeword_reduce"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_batch_segmented_codeword_reduce_source_guardrail"
    )
    assert str(route_batch_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_route_batch_segmented_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": (
                    "expert_cohort_codeword_broadcast_source_guardrail_present"
                ),
                "passes_contract": True,
            },
            "route_batch_segmented_codeword_reduce_guardrail": {
                "decision": "missing_route_batch_segmented_codeword_reduce_source",
                "passes_contract": False,
            },
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    expert_cohort_speed = _write_json(
        tmp_path / "expert_cohort_speed.json",
        {
            **base,
            "decision": "reject_expert_cohort_codeword_broadcast_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_batch_design = _write_json(
        tmp_path / "route_batch_design.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=expert_cohort_design,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=expert_cohort_speed,
        trackb_route_batch_segmented_design_path=route_batch_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "route_batch_segmented_source_guardrail_missing"
    )
    assert trackb_front["route_batch_segmented_source_guardrail_decision"] == (
        "missing_route_batch_segmented_codeword_reduce_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_batch_segmented_codeword_reduce"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_batch_segmented_codeword_reduce_source_guardrail"
    )
    assert str(trackb_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_route_batch_segmented_native_parity_after_source_guardrail(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": (
                    "expert_cohort_codeword_broadcast_source_guardrail_present"
                ),
                "passes_contract": True,
            },
            "route_batch_segmented_codeword_reduce_guardrail": {
                "decision": (
                    "route_batch_segmented_codeword_reduce_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    expert_cohort_speed = _write_json(
        tmp_path / "expert_cohort_speed.json",
        {
            **base,
            "decision": "reject_expert_cohort_codeword_broadcast_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_batch_design = _write_json(
        tmp_path / "route_batch_design.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=expert_cohort_design,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=expert_cohort_speed,
        trackb_route_batch_segmented_design_path=route_batch_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "route_batch_segmented_source_guardrail_present"
    )
    assert trackb_front["route_batch_segmented_source_guardrail_decision"] == (
        "route_batch_segmented_codeword_reduce_source_guardrail_present"
    )
    assert trackb_front["next"] == (
        "prove_route_batch_segmented_codeword_reduce_native_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_batch_segmented_codeword_reduce_native_parity"
    )
    assert str(trackb_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_route_batch_segmented_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": (
                    "expert_cohort_codeword_broadcast_source_guardrail_present"
                ),
                "passes_contract": True,
            },
            "route_batch_segmented_codeword_reduce_guardrail": {
                "decision": (
                    "route_batch_segmented_codeword_reduce_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    expert_cohort_speed = _write_json(
        tmp_path / "expert_cohort_speed.json",
        {
            **base,
            "decision": "reject_expert_cohort_codeword_broadcast_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_batch_design = _write_json(
        tmp_path / "route_batch_design.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    route_batch_native_parity = _write_json(
        tmp_path / "route_batch_native_parity.json",
        {
            **base,
            "decision": "route_batch_segmented_codeword_reduce_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=expert_cohort_design,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=expert_cohort_speed,
        trackb_route_batch_segmented_design_path=route_batch_design,
        trackb_route_batch_segmented_native_parity_path=route_batch_native_parity,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "route_batch_segmented_native_parity_present"
    assert trackb_front["route_batch_segmented_native_parity_decision"] == (
        "route_batch_segmented_codeword_reduce_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_route_batch_segmented_codeword_reduce_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_batch_segmented_codeword_reduce_air_artifact_parity"
    )
    assert str(route_batch_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_route_batch_segmented_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": (
                    "expert_cohort_codeword_broadcast_source_guardrail_present"
                ),
                "passes_contract": True,
            },
            "route_batch_segmented_codeword_reduce_guardrail": {
                "decision": (
                    "route_batch_segmented_codeword_reduce_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    expert_cohort_speed = _write_json(
        tmp_path / "expert_cohort_speed.json",
        {
            **base,
            "decision": "reject_expert_cohort_codeword_broadcast_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_batch_design = _write_json(
        tmp_path / "route_batch_design.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    route_batch_native_parity = _write_json(
        tmp_path / "route_batch_native_parity.json",
        {
            **base,
            "decision": "route_batch_segmented_codeword_reduce_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    route_batch_artifact_parity = _write_json(
        tmp_path / "route_batch_artifact_parity.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=expert_cohort_design,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=expert_cohort_speed,
        trackb_route_batch_segmented_design_path=route_batch_design,
        trackb_route_batch_segmented_native_parity_path=route_batch_native_parity,
        trackb_route_batch_segmented_artifact_parity_path=(
            route_batch_artifact_parity
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "route_batch_segmented_artifact_parity_present"
    assert trackb_front["route_batch_segmented_artifact_parity_decision"] == (
        "route_batch_segmented_codeword_reduce_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_route_batch_segmented_codeword_reduce_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_route_batch_segmented_codeword_reduce_same_window_q2_speed_packet"
    )
    assert str(route_batch_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_route_batch_segmented_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "expert_cohort_codeword_broadcast_guardrail": {
                "decision": (
                    "expert_cohort_codeword_broadcast_source_guardrail_present"
                ),
                "passes_contract": True,
            },
            "route_batch_segmented_codeword_reduce_guardrail": {
                "decision": (
                    "route_batch_segmented_codeword_reduce_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    expert_cohort_design = _write_json(
        tmp_path / "expert_cohort_design.json",
        {
            **base,
            "decision": (
                "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    expert_cohort_speed = _write_json(
        tmp_path / "expert_cohort_speed.json",
        {
            **base,
            "decision": "reject_expert_cohort_codeword_broadcast_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_batch_design = _write_json(
        tmp_path / "route_batch_design.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    route_batch_native_parity = _write_json(
        tmp_path / "route_batch_native_parity.json",
        {
            **base,
            "decision": "route_batch_segmented_codeword_reduce_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    route_batch_artifact_parity = _write_json(
        tmp_path / "route_batch_artifact_parity.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    route_batch_speed = _write_json(
        tmp_path / "route_batch_speed.json",
        {
            **base,
            "decision": "reject_route_batch_segmented_codeword_reduce_speed_path",
            "same_window_q2_speed_packet": True,
            "all_parity_pass": False,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=expert_cohort_design,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=expert_cohort_speed,
        trackb_route_batch_segmented_design_path=route_batch_design,
        trackb_route_batch_segmented_native_parity_path=route_batch_native_parity,
        trackb_route_batch_segmented_artifact_parity_path=(
            route_batch_artifact_parity
        ),
        trackb_route_batch_segmented_speed_packet_path=route_batch_speed,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "route_batch_segmented_speed_rejected"
    assert trackb_front["route_batch_segmented_speed_packet_decision"] == (
        "reject_route_batch_segmented_codeword_reduce_speed_path"
    )
    assert trackb_front["next"] == (
        "change_route_batch_segmented_codeword_reduce_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_route_batch_segmented_codeword_reduce_layout_or_kernel_family"
    )
    assert str(route_batch_speed) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_component_stream_partial_reduction_design(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_batch_segmented_codeword_reduce_guardrail": {
                "decision": (
                    "route_batch_segmented_codeword_reduce_source_guardrail_present"
                ),
                "passes_contract": True,
            },
        },
    )
    route_batch_design = _write_json(
        tmp_path / "route_batch_design.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    route_batch_native_parity = _write_json(
        tmp_path / "route_batch_native_parity.json",
        {
            **base,
            "decision": "route_batch_segmented_codeword_reduce_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    route_batch_artifact_parity = _write_json(
        tmp_path / "route_batch_artifact_parity.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
        },
    )
    route_batch_speed = _write_json(
        tmp_path / "route_batch_speed.json",
        {
            **base,
            "decision": "reject_route_batch_segmented_codeword_reduce_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    component_stream_design = _write_json(
        tmp_path / "component_stream_design.json",
        {
            **base,
            "decision": (
                "component_stream_partial_reduction_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_component_stream_partial_reduction"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=route_batch_design,
        trackb_route_batch_segmented_native_parity_path=route_batch_native_parity,
        trackb_route_batch_segmented_artifact_parity_path=(
            route_batch_artifact_parity
        ),
        trackb_route_batch_segmented_speed_packet_path=route_batch_speed,
        trackb_component_stream_partial_reduction_design_path=component_stream_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "component_stream_partial_reduction_design_ready"
    assert trackb_front["component_stream_partial_reduction_design_decision"] == (
        "component_stream_partial_reduction_ready_for_source_structure_probe"
    )
    assert trackb_front["component_stream_partial_reduction_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_component_stream_partial_reduction"
    )
    assert trackb_front["route_batch_segmented_speed_packet_decision"] == (
        "reject_route_batch_segmented_codeword_reduce_speed_path"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_component_stream_partial_reduction"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_component_stream_partial_reduction_source_guardrail"
    )
    assert str(component_stream_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


@pytest.mark.parametrize(
    (
        "guard_decision",
        "partial_body_parallel_schedule_present",
        "expected_next",
        "expected_next_slice",
    ),
    [
        (
            "component_stream_partial_reduction_scaffold_scalar_body",
            False,
            "replace_scalar_component_stream_partial_body_with_valid_parallel_schedule",
            "replace_trackb_component_stream_partial_reduction_scalar_body",
        ),
        (
            "component_stream_partial_reduction_parallel_body_present",
            True,
            "prove_component_stream_partial_reduction_native_parity",
            "prove_trackb_component_stream_partial_reduction_native_parity",
        ),
    ],
)
def test_five_front_status_routes_to_component_stream_partial_reduction_source_guardrail(
    tmp_path: Path,
    guard_decision: str,
    partial_body_parallel_schedule_present: bool,
    expected_next: str,
    expected_next_slice: str,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_batch_segmented_codeword_reduce_guardrail": {
                "decision": (
                    "route_batch_segmented_codeword_reduce_source_guardrail_present"
                ),
                "passes_contract": True,
            },
            "component_stream_partial_reduction_guardrail": {
                "decision": guard_decision,
                "passes_contract": True,
                "partial_body_parallel_schedule_present": (
                    partial_body_parallel_schedule_present
                ),
                "speed_claim": False,
            },
        },
    )
    route_batch_design = _write_json(
        tmp_path / "route_batch_design.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    route_batch_speed = _write_json(
        tmp_path / "route_batch_speed.json",
        {
            **base,
            "decision": "reject_route_batch_segmented_codeword_reduce_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    component_stream_design = _write_json(
        tmp_path / "component_stream_design.json",
        {
            **base,
            "decision": (
                "component_stream_partial_reduction_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_component_stream_partial_reduction"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=route_batch_design,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=route_batch_speed,
        trackb_component_stream_partial_reduction_design_path=component_stream_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "component_stream_partial_reduction_source_guardrail_present"
    )
    assert trackb_front[
        "component_stream_partial_reduction_source_guardrail_decision"
    ] == guard_decision
    assert trackb_front["next"] == expected_next
    assert report["next_local_slices"][0] == expected_next_slice
    assert str(trackb_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_component_stream_partial_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_batch_segmented_codeword_reduce_guardrail": {
                "decision": (
                    "route_batch_segmented_codeword_reduce_source_guardrail_present"
                ),
                "passes_contract": True,
            },
            "component_stream_partial_reduction_guardrail": {
                "decision": "component_stream_partial_reduction_parallel_body_present",
                "passes_contract": True,
                "partial_body_parallel_schedule_present": True,
                "component_pair_parallel_reduction_present": True,
                "speed_claim": False,
            },
        },
    )
    route_batch_design = _write_json(
        tmp_path / "route_batch_design.json",
        {
            **base,
            "decision": (
                "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    route_batch_speed = _write_json(
        tmp_path / "route_batch_speed.json",
        {
            **base,
            "decision": "reject_route_batch_segmented_codeword_reduce_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    component_stream_design = _write_json(
        tmp_path / "component_stream_design.json",
        {
            **base,
            "decision": (
                "component_stream_partial_reduction_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_component_stream_partial_reduction"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    component_stream_native_parity = _write_json(
        tmp_path / "component_stream_native_parity.json",
        {
            **base,
            "decision": "component_stream_partial_reduction_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=route_batch_design,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=route_batch_speed,
        trackb_component_stream_partial_reduction_design_path=component_stream_design,
        trackb_component_stream_partial_reduction_native_parity_path=(
            component_stream_native_parity
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "component_stream_partial_reduction_native_parity_present"
    assert trackb_front[
        "component_stream_partial_reduction_native_parity_decision"
    ] == "component_stream_partial_reduction_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_component_stream_partial_reduction_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_component_stream_partial_reduction_air_artifact_parity"
    )
    assert str(component_stream_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_component_stream_tensorops_rejection_to_new_family_gate(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "component_stream_partial_reduction_guardrail": {
                "decision": "component_stream_partial_reduction_parallel_body_present",
                "passes_contract": True,
                "speed_claim": False,
            },
        },
    )
    component_stream_design = _write_json(
        tmp_path / "component_stream_design.json",
        {
            **base,
            "decision": (
                "component_stream_partial_reduction_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_component_stream_partial_reduction"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    component_stream_native_parity = _write_json(
        tmp_path / "component_stream_native_parity.json",
        {
            **base,
            "decision": "component_stream_partial_reduction_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
        },
    )
    component_stream_tensorops_rejection = _write_json(
        tmp_path / "component_stream_tensorops_rejection.json",
        {
            **base,
            "component_stream_speed_path_guardrail": {
                "decision": "reject_component_stream_tensorops_speed_path",
                "speed_claim": False,
                "component_stream_tensorops_best_ratio_to_q2": 7.78,
            },
            "next_track_b_hypothesis": "change_component_stream_tensorops_layout_or_kernel_family",
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    next_family_gate = _write_json(
        tmp_path / "next_family_gate.json",
        {
            **base,
            "record_type": "glm45_air_e8p_trackb_next_family_gate",
            "decision": "block_stale_trackb_plan_variant_requires_materially_new_family_design",
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "next_track_b_action": (
                "write_materially_new_rhs_or_kernel_family_design_before_source_or_speed_work"
            ),
            "stale_plan_variant_blocked": True,
            "speed_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=component_stream_design,
        trackb_component_stream_partial_reduction_native_parity_path=(
            component_stream_native_parity
        ),
        trackb_component_stream_tensorops_rejection_path=(
            component_stream_tensorops_rejection
        ),
        trackb_next_family_gate_path=next_family_gate,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "requires_new_rhs_or_kernel_family"
    assert trackb_front["next_family_gate_decision"] == (
        "block_stale_trackb_plan_variant_requires_materially_new_family_design"
    )
    assert trackb_front["component_stream_tensorops_rejection_decision"] == (
        "reject_component_stream_tensorops_speed_path"
    )
    assert trackb_front["next"] == (
        "write_materially_new_rhs_or_kernel_family_design_before_source_or_speed_work"
    )
    assert report["next_local_slices"][0] == (
        "write_materially_new_trackb_rhs_or_kernel_family_design"
    )
    assert str(component_stream_tensorops_rejection) in trackb_front["evidence"]
    assert str(next_family_gate) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_route_codeword_lut_design_after_expert_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {**base},
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "route_codeword_lut_accumulate_design_ready"
    assert trackb_front["route_codeword_lut_accumulate_design_decision"] == (
        "route_codeword_lut_accumulate_ready_for_source_structure_probe"
    )
    assert trackb_front["route_codeword_lut_accumulate_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_codeword_lut_accumulate_source_guardrail"
    )
    assert str(route_codeword_lut_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_route_codeword_lut_source_guardrail_missing(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "missing_route_codeword_lut_accumulate_source",
                "passes_contract": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "route_codeword_lut_accumulate_source_guardrail_missing"
    )
    assert trackb_front["route_codeword_lut_accumulate_source_guardrail_decision"] == (
        "missing_route_codeword_lut_accumulate_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_codeword_lut_accumulate_source_guardrail"
    )
    assert str(trackb_source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_route_codeword_lut_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_codeword_lut_accumulate_native_parity"
            ),
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=native_parity,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "route_codeword_lut_accumulate_native_parity_present"
    )
    assert trackb_front["route_codeword_lut_accumulate_native_parity_decision"] == (
        "route_codeword_lut_accumulate_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_route_codeword_lut_accumulate_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_codeword_lut_accumulate_air_artifact_parity"
    )
    assert str(native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_route_codeword_lut_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_codeword_lut_accumulate_native_parity"
            ),
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "route_codeword_lut_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_codeword_lut_accumulate_artifact_parity"
            ),
            "decision": "route_codeword_lut_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=native_parity,
        trackb_route_codeword_lut_accumulate_artifact_parity_path=artifact_parity,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "route_codeword_lut_accumulate_artifact_parity_present"
    )
    assert trackb_front["route_codeword_lut_accumulate_artifact_parity_decision"] == (
        "route_codeword_lut_accumulate_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_route_codeword_lut_accumulate_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_route_codeword_lut_accumulate_same_window_q2_speed_packet"
    )
    assert str(artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_route_codeword_lut_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_codeword_lut_accumulate_native_parity"
            ),
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "route_codeword_lut_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_codeword_lut_accumulate_artifact_parity"
            ),
            "decision": "route_codeword_lut_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    speed_packet = _write_json(
        tmp_path / "route_codeword_lut_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_codeword_lut_accumulate_speed_packet"
            ),
            "decision": "reject_route_codeword_lut_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=native_parity,
        trackb_route_codeword_lut_accumulate_artifact_parity_path=artifact_parity,
        trackb_route_codeword_lut_accumulate_speed_packet_path=speed_packet,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "route_codeword_lut_accumulate_speed_rejected"
    assert trackb_front["route_codeword_lut_accumulate_speed_packet_decision"] == (
        "reject_route_codeword_lut_accumulate_speed_path"
    )
    assert trackb_front["next"] == (
        "change_route_codeword_lut_accumulate_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_route_codeword_lut_accumulate_layout_or_kernel_family"
    )
    assert str(speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_rowwise_codeword_tile_design_after_route_lut_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    trackb_source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_codeword_lut_accumulate_native_parity"
            ),
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "route_codeword_lut_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_codeword_lut_accumulate_artifact_parity"
            ),
            "decision": "route_codeword_lut_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    speed_packet = _write_json(
        tmp_path / "route_codeword_lut_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_codeword_lut_accumulate_speed_packet"
            ),
            "decision": "reject_route_codeword_lut_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
            "candidate_preflight_rejected": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    rowwise_design = _write_json(
        tmp_path / "rowwise_design.json",
        {
            **base,
            "decision": (
                "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_rowwise_codeword_tile_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_rowwise_codeword_tile_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=trackb_source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=native_parity,
        trackb_route_codeword_lut_accumulate_artifact_parity_path=artifact_parity,
        trackb_route_codeword_lut_accumulate_speed_packet_path=speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "rowwise_codeword_tile_accumulate_design_ready"
    assert trackb_front["rowwise_codeword_tile_accumulate_design_decision"] == (
        "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
    )
    assert trackb_front["rowwise_codeword_tile_accumulate_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_rowwise_codeword_tile_accumulate"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_rowwise_codeword_tile_accumulate"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_rowwise_codeword_tile_accumulate_source_guardrail"
    )
    assert str(rowwise_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_missing_rowwise_codeword_tile_source_guardrail(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "rowwise_codeword_tile_accumulate_guardrail": {
                "decision": "missing_rowwise_codeword_tile_accumulate_source",
                "passes_contract": False,
                "kernel_present": False,
                "primitive_binding_present": False,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    artifact_parity = _write_json(
        tmp_path / "route_codeword_lut_artifact_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    speed_packet = _write_json(
        tmp_path / "route_codeword_lut_speed_packet.json",
        {
            **base,
            "decision": "reject_route_codeword_lut_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
            "candidate_preflight_rejected": True,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    rowwise_design = _write_json(
        tmp_path / "rowwise_design.json",
        {
            **base,
            "decision": (
                "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_rowwise_codeword_tile_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_rowwise_codeword_tile_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_source_guard_path=source_guard,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=native_parity,
        trackb_route_codeword_lut_accumulate_artifact_parity_path=artifact_parity,
        trackb_route_codeword_lut_accumulate_speed_packet_path=speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "rowwise_codeword_tile_accumulate_source_guardrail_missing"
    )
    assert trackb_front["rowwise_codeword_tile_accumulate_source_guardrail_decision"] == (
        "missing_rowwise_codeword_tile_accumulate_source"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_rowwise_codeword_tile_accumulate"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_rowwise_codeword_tile_accumulate_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_rowwise_codeword_tile_native_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "rowwise_codeword_tile_accumulate_guardrail": {
                "decision": "rowwise_codeword_tile_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    route_lut_native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    route_lut_artifact_parity = _write_json(
        tmp_path / "route_codeword_lut_artifact_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    route_lut_speed_packet = _write_json(
        tmp_path / "route_codeword_lut_speed_packet.json",
        {
            **base,
            "decision": "reject_route_codeword_lut_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
            "candidate_preflight_rejected": True,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    rowwise_design = _write_json(
        tmp_path / "rowwise_design.json",
        {
            **base,
            "decision": (
                "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_rowwise_codeword_tile_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_rowwise_codeword_tile_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    rowwise_native_parity = _write_json(
        tmp_path / "rowwise_native_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_rowwise_codeword_tile_accumulate_native_parity",
            "decision": "rowwise_codeword_tile_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_source_guard_path=source_guard,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "rowwise_codeword_tile_accumulate_native_parity_present"
    )
    assert trackb_front["rowwise_codeword_tile_accumulate_native_parity_decision"] == (
        "rowwise_codeword_tile_accumulate_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_rowwise_codeword_tile_accumulate_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_rowwise_codeword_tile_accumulate_air_artifact_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_rowwise_codeword_tile_artifact_parity(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "rowwise_codeword_tile_accumulate_guardrail": {
                "decision": "rowwise_codeword_tile_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    route_lut_native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    route_lut_artifact_parity = _write_json(
        tmp_path / "route_codeword_lut_artifact_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    route_lut_speed_packet = _write_json(
        tmp_path / "route_codeword_lut_speed_packet.json",
        {
            **base,
            "decision": "reject_route_codeword_lut_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
            "candidate_preflight_rejected": True,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    rowwise_design = _write_json(
        tmp_path / "rowwise_design.json",
        {
            **base,
            "decision": (
                "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_rowwise_codeword_tile_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_rowwise_codeword_tile_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    rowwise_native_parity = _write_json(
        tmp_path / "rowwise_native_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_rowwise_codeword_tile_accumulate_native_parity",
            "decision": "rowwise_codeword_tile_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    rowwise_artifact_parity = _write_json(
        tmp_path / "rowwise_artifact_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_rowwise_codeword_tile_accumulate_artifact_parity",
            "decision": "rowwise_codeword_tile_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_source_guard_path=source_guard,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "rowwise_codeword_tile_accumulate_artifact_parity_present"
    )
    assert trackb_front["rowwise_codeword_tile_accumulate_artifact_parity_decision"] == (
        "rowwise_codeword_tile_accumulate_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_rowwise_codeword_tile_accumulate_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_rowwise_codeword_tile_accumulate_same_window_q2_speed_packet"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_after_rowwise_codeword_tile_speed_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
            "rowwise_codeword_tile_accumulate_guardrail": {
                "decision": "rowwise_codeword_tile_accumulate_source_guardrail_present",
                "passes_contract": True,
                "kernel_present": True,
                "primitive_binding_present": True,
                "speed_claim": False,
                "native_parity_claim": False,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    route_lut_native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    route_lut_artifact_parity = _write_json(
        tmp_path / "route_codeword_lut_artifact_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    route_lut_speed_packet = _write_json(
        tmp_path / "route_codeword_lut_speed_packet.json",
        {
            **base,
            "decision": "reject_route_codeword_lut_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
            "candidate_preflight_rejected": True,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    rowwise_design = _write_json(
        tmp_path / "rowwise_design.json",
        {
            **base,
            "decision": (
                "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_rowwise_codeword_tile_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    rowwise_native_parity = _write_json(
        tmp_path / "rowwise_native_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_rowwise_codeword_tile_accumulate_native_parity",
            "decision": "rowwise_codeword_tile_accumulate_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )
    rowwise_artifact_parity = _write_json(
        tmp_path / "rowwise_artifact_parity.json",
        {
            **base,
            "record_type": "glm45_air_e8p_rowwise_codeword_tile_accumulate_artifact_parity",
            "decision": "rowwise_codeword_tile_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "speed_claim": False,
            "artifact_parity_claim": True,
        },
    )
    rowwise_speed_packet = _write_json(
        tmp_path / "rowwise_speed_packet.json",
        {
            **base,
            "record_type": "glm45_air_e8p_rowwise_codeword_tile_accumulate_speed_packet",
            "decision": "reject_rowwise_codeword_tile_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_source_guard_path=source_guard,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "rowwise_codeword_tile_accumulate_speed_rejected"
    assert trackb_front["rowwise_codeword_tile_accumulate_speed_packet_decision"] == (
        "reject_rowwise_codeword_tile_accumulate_speed_path"
    )
    assert trackb_front["next"] == (
        "change_rowwise_codeword_tile_accumulate_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_rowwise_codeword_tile_accumulate_layout_or_kernel_family"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_routes_to_output_tile_local_codeword_lut_design_after_rowwise_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [{"evidence": "current"}],
        },
    )
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
            },
            "rowwise_codeword_tile_accumulate_guardrail": {
                "decision": "rowwise_codeword_tile_accumulate_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
            "speed_claim": False,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    route_lut_native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    route_lut_artifact_parity = _write_json(
        tmp_path / "route_codeword_lut_artifact_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    route_lut_speed_packet = _write_json(
        tmp_path / "route_codeword_lut_speed_packet.json",
        {
            **base,
            "decision": "reject_route_codeword_lut_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
        },
    )
    rowwise_design = _write_json(
        tmp_path / "rowwise_design.json",
        {
            **base,
            "decision": (
                "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_rowwise_codeword_tile_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
        },
    )
    rowwise_native_parity = _write_json(
        tmp_path / "rowwise_native_parity.json",
        {
            **base,
            "decision": "rowwise_codeword_tile_accumulate_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    rowwise_artifact_parity = _write_json(
        tmp_path / "rowwise_artifact_parity.json",
        {
            **base,
            "decision": "rowwise_codeword_tile_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    rowwise_speed_packet = _write_json(
        tmp_path / "rowwise_speed_packet.json",
        {
            **base,
            "decision": "reject_rowwise_codeword_tile_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )
    output_tile_design = _write_json(
        tmp_path / "output_tile_local_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "output_tile_local_codeword_lut_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_output_tile_local_codeword_lut"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_output_tile_local_codeword_lut"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
            "artifact_parity_claim": False,
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": False, "cache_rows_generated": False},
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_source_guard_path=source_guard,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "output_tile_local_codeword_lut_design_ready"
    assert trackb_front["output_tile_local_codeword_lut_design_decision"] == (
        "output_tile_local_codeword_lut_ready_for_source_structure_probe"
    )
    assert trackb_front["output_tile_local_codeword_lut_target_kernel_family"] == (
        "sorted_gather_qmm_rhs_nax_output_tile_local_codeword_lut"
    )
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_output_tile_local_codeword_lut"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_output_tile_local_codeword_lut_source_guardrail"
    )
    assert str(output_tile_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["output_tile_local_codeword_lut_guardrail"] = {
        "decision": "missing_output_tile_local_codeword_lut_source",
        "passes_contract": False,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_source_guard_path=source_guard,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert (
        trackb_front["status"]
        == "output_tile_local_codeword_lut_source_guardrail_missing"
    )
    assert trackb_front[
        "output_tile_local_codeword_lut_source_guardrail_decision"
    ] == "missing_output_tile_local_codeword_lut_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_output_tile_local_codeword_lut"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_output_tile_local_codeword_lut_source_guardrail"
    )
    assert str(source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["output_tile_local_codeword_lut_guardrail"] = {
        "decision": "output_tile_local_codeword_lut_source_guardrail_present",
        "passes_contract": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")
    output_tile_native_parity = _write_json(
        tmp_path / "output_tile_local_native_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_output_tile_local_codeword_lut_native_parity"
            ),
            "decision": "output_tile_local_codeword_lut_native_parity_pass",
            "passes_native_parity": True,
            "native_parity_claim": True,
            "speed_claim": False,
        },
    )
    output_tile_artifact_parity = _write_json(
        tmp_path / "output_tile_local_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_output_tile_local_codeword_lut_artifact_parity"
            ),
            "decision": "output_tile_local_codeword_lut_air_artifact_parity_pass",
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "native_parity_claim": False,
            "speed_claim": False,
        },
    )
    output_tile_speed_packet = _write_json(
        tmp_path / "output_tile_local_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_output_tile_local_codeword_lut_speed_packet"
            ),
            "decision": "reject_output_tile_local_codeword_lut_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_source_guard_path=source_guard,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        trackb_output_tile_local_codeword_lut_native_parity_path=(
            output_tile_native_parity
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert (
        trackb_front["status"]
        == "output_tile_local_codeword_lut_native_parity_present"
    )
    assert trackb_front["output_tile_local_codeword_lut_native_parity_decision"] == (
        "output_tile_local_codeword_lut_native_parity_pass"
    )
    assert trackb_front["next"] == (
        "prove_output_tile_local_codeword_lut_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_output_tile_local_codeword_lut_air_artifact_parity"
    )
    assert str(output_tile_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_source_guard_path=source_guard,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        trackb_output_tile_local_codeword_lut_native_parity_path=(
            output_tile_native_parity
        ),
        trackb_output_tile_local_codeword_lut_artifact_parity_path=(
            output_tile_artifact_parity
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]
    assert (
        trackb_front["status"]
        == "output_tile_local_codeword_lut_artifact_parity_present"
    )
    assert trackb_front["output_tile_local_codeword_lut_artifact_parity_decision"] == (
        "output_tile_local_codeword_lut_air_artifact_parity_pass"
    )
    assert trackb_front["next"] == (
        "run_output_tile_local_codeword_lut_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_output_tile_local_codeword_lut_same_window_q2_speed_packet"
    )
    assert str(output_tile_artifact_parity) in trackb_front["evidence"]

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_source_guard_path=source_guard,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        trackb_output_tile_local_codeword_lut_native_parity_path=(
            output_tile_native_parity
        ),
        trackb_output_tile_local_codeword_lut_artifact_parity_path=(
            output_tile_artifact_parity
        ),
        trackb_output_tile_local_codeword_lut_speed_packet_path=(
            output_tile_speed_packet
        ),
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )
    trackb_front = report["fronts"]["workstream2_trackb"]
    assert (
        trackb_front["status"]
        == "output_tile_local_codeword_lut_speed_rejected"
    )
    assert trackb_front["output_tile_local_codeword_lut_speed_packet_decision"] == (
        "reject_output_tile_local_codeword_lut_speed_path"
    )
    assert trackb_front["next"] == (
        "change_output_tile_local_codeword_lut_layout_or_kernel_family"
    )
    assert report["next_local_slices"][0] == (
        "change_trackb_output_tile_local_codeword_lut_layout_or_kernel_family"
    )
    assert str(output_tile_speed_packet) in trackb_front["evidence"]


def test_five_front_status_routes_to_route_microtile_codeword_block_reduce_after_output_tile_rejection(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {"peer2_used": False, "rdma_jaccl_touched": False}
    workstream1 = _write_json(
        tmp_path / "workstream1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "ready_for_native_speed_work": False,
            "rejected_families": [
                {"evidence": f"rejected-{index}"}
                for index in range(11)
            ],
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_layer3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            )
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_layer77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            )
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": True, "cache_rows_generated": False},
    )
    model_card = tmp_path / "model_card.md"
    model_card.write_text("draft", encoding="utf-8")
    source_guard = _write_json(
        tmp_path / "source_guard.json",
        {
            **base,
            "route_codeword_lut_accumulate_guardrail": {
                "decision": "route_codeword_lut_accumulate_source_guardrail_present",
                "passes_contract": True,
            },
            "rowwise_codeword_tile_accumulate_guardrail": {
                "decision": "rowwise_codeword_tile_accumulate_source_guardrail_present",
                "passes_contract": True,
            },
            "output_tile_local_codeword_lut_guardrail": {
                "decision": "output_tile_local_codeword_lut_source_guardrail_present",
                "passes_contract": True,
            },
        },
    )
    expert_kblock_speed = _write_json(
        tmp_path / "expert_kblock_speed.json",
        {
            **base,
            "decision": "reject_expert_kblock_codeword_factor_reuse_speed_path",
            "same_window_q2_speed_packet": True,
        },
    )
    route_codeword_lut_design = _write_json(
        tmp_path / "route_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "route_codeword_lut_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    route_lut_native_parity = _write_json(
        tmp_path / "route_codeword_lut_native_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    route_lut_artifact_parity = _write_json(
        tmp_path / "route_codeword_lut_artifact_parity.json",
        {
            **base,
            "decision": "route_codeword_lut_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    route_lut_speed_packet = _write_json(
        tmp_path / "route_codeword_lut_speed_packet.json",
        {
            **base,
            "decision": "reject_route_codeword_lut_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
        },
    )
    rowwise_design = _write_json(
        tmp_path / "rowwise_design.json",
        {
            **base,
            "decision": (
                "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_rowwise_codeword_tile_accumulate"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_rowwise_codeword_tile_accumulate"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    rowwise_native_parity = _write_json(
        tmp_path / "rowwise_native_parity.json",
        {
            **base,
            "decision": "rowwise_codeword_tile_accumulate_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    rowwise_artifact_parity = _write_json(
        tmp_path / "rowwise_artifact_parity.json",
        {
            **base,
            "decision": "rowwise_codeword_tile_accumulate_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    rowwise_speed_packet = _write_json(
        tmp_path / "rowwise_speed_packet.json",
        {
            **base,
            "decision": "reject_rowwise_codeword_tile_accumulate_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    output_tile_design = _write_json(
        tmp_path / "output_tile_local_codeword_lut_design.json",
        {
            **base,
            "decision": (
                "output_tile_local_codeword_lut_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_output_tile_local_codeword_lut"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_output_tile_local_codeword_lut"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )
    output_tile_native_parity = _write_json(
        tmp_path / "output_tile_local_native_parity.json",
        {
            **base,
            "decision": "output_tile_local_codeword_lut_native_parity_pass",
            "passes_native_parity": True,
        },
    )
    output_tile_artifact_parity = _write_json(
        tmp_path / "output_tile_local_artifact_parity.json",
        {
            **base,
            "decision": "output_tile_local_codeword_lut_air_artifact_parity_pass",
            "passes_artifact_parity": True,
        },
    )
    output_tile_speed_packet = _write_json(
        tmp_path / "output_tile_local_speed_packet.json",
        {
            **base,
            "decision": "reject_output_tile_local_codeword_lut_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
        },
    )
    route_microtile_design = _write_json(
        tmp_path / "route_microtile_design.json",
        {
            **base,
            "decision": (
                "route_microtile_codeword_block_reduce_ready_for_source_structure_probe"
            ),
            "candidate": {
                "target_kernel_family": (
                    "sorted_gather_qmm_rhs_nax_route_microtile_codeword_block_reduce"
                )
            },
            "selector_verdict": {"candidate_ready_for_source_probe": True},
            "next_track_b_action": (
                "implement_source_structure_guardrail_for_route_microtile_codeword_block_reduce"
            ),
            "speed_claim": False,
            "native_parity_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        trackb_output_tile_local_codeword_lut_native_parity_path=(
            output_tile_native_parity
        ),
        trackb_output_tile_local_codeword_lut_artifact_parity_path=(
            output_tile_artifact_parity
        ),
        trackb_output_tile_local_codeword_lut_speed_packet_path=(
            output_tile_speed_packet
        ),
            trackb_route_microtile_codeword_block_reduce_design_path=(
                route_microtile_design
            ),
            trackb_route_microtile_codeword_block_reduce_artifact_parity_path=None,
            trackb_route_microtile_codeword_block_reduce_speed_packet_path=None,
            trackb_kblock_wavefront_codeword_scan_design_path=None,
            trackb_kblock_wavefront_codeword_scan_native_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_artifact_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
            qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "route_microtile_codeword_block_reduce_design_ready"
    )
    assert trackb_front[
        "route_microtile_codeword_block_reduce_design_decision"
    ] == "route_microtile_codeword_block_reduce_ready_for_source_structure_probe"
    assert trackb_front[
        "route_microtile_codeword_block_reduce_target_kernel_family"
    ] == "sorted_gather_qmm_rhs_nax_route_microtile_codeword_block_reduce"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_microtile_codeword_block_reduce"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_microtile_codeword_block_reduce_source_guardrail"
    )
    assert str(route_microtile_design) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    source_guard_payload = json.loads(source_guard.read_text(encoding="utf-8"))
    source_guard_payload["route_microtile_codeword_block_reduce_guardrail"] = {
        "decision": "missing_route_microtile_codeword_block_reduce_source",
        "passes_contract": False,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        trackb_output_tile_local_codeword_lut_native_parity_path=(
            output_tile_native_parity
        ),
        trackb_output_tile_local_codeword_lut_artifact_parity_path=(
            output_tile_artifact_parity
        ),
        trackb_output_tile_local_codeword_lut_speed_packet_path=(
            output_tile_speed_packet
        ),
            trackb_route_microtile_codeword_block_reduce_design_path=(
                route_microtile_design
            ),
            trackb_route_microtile_codeword_block_reduce_artifact_parity_path=None,
            trackb_route_microtile_codeword_block_reduce_speed_packet_path=None,
            trackb_kblock_wavefront_codeword_scan_design_path=None,
            trackb_kblock_wavefront_codeword_scan_native_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_artifact_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
            qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "route_microtile_codeword_block_reduce_source_guardrail_missing"
    )
    assert trackb_front[
        "route_microtile_codeword_block_reduce_source_guardrail_decision"
    ] == "missing_route_microtile_codeword_block_reduce_source"
    assert trackb_front["next"] == (
        "implement_source_structure_guardrail_for_route_microtile_codeword_block_reduce"
    )
    assert report["next_local_slices"][0] == (
        "implement_trackb_route_microtile_codeword_block_reduce_source_guardrail"
    )
    assert str(source_guard) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    source_guard_payload["route_microtile_codeword_block_reduce_guardrail"] = {
        "decision": "route_microtile_codeword_block_reduce_source_guardrail_present",
        "passes_contract": True,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    source_guard.write_text(json.dumps(source_guard_payload), encoding="utf-8")
    route_microtile_native_parity = _write_json(
        tmp_path / "route_microtile_native_parity.json",
        {
            **base,
            "decision": "route_microtile_codeword_block_reduce_native_parity_pass",
            "passes_native_parity": True,
            "speed_claim": False,
            "native_parity_claim": True,
        },
    )

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        trackb_output_tile_local_codeword_lut_native_parity_path=(
            output_tile_native_parity
        ),
        trackb_output_tile_local_codeword_lut_artifact_parity_path=(
            output_tile_artifact_parity
        ),
        trackb_output_tile_local_codeword_lut_speed_packet_path=(
            output_tile_speed_packet
        ),
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
            trackb_route_microtile_codeword_block_reduce_native_parity_path=(
                route_microtile_native_parity
            ),
            trackb_route_microtile_codeword_block_reduce_artifact_parity_path=None,
            trackb_route_microtile_codeword_block_reduce_speed_packet_path=None,
            trackb_kblock_wavefront_codeword_scan_design_path=None,
            trackb_kblock_wavefront_codeword_scan_native_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_artifact_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
            qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "route_microtile_codeword_block_reduce_native_parity_present"
    )
    assert trackb_front[
        "route_microtile_codeword_block_reduce_native_parity_decision"
    ] == "route_microtile_codeword_block_reduce_native_parity_pass"
    assert trackb_front["next"] == (
        "prove_route_microtile_codeword_block_reduce_air_artifact_parity"
    )
    assert report["next_local_slices"][0] == (
        "prove_trackb_route_microtile_codeword_block_reduce_air_artifact_parity"
    )
    assert str(route_microtile_native_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_microtile_artifact_parity = _write_json(
        tmp_path / "route_microtile_artifact_parity.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_microtile_codeword_block_reduce_artifact_parity"
            ),
            "decision": (
                "route_microtile_codeword_block_reduce_air_artifact_parity_pass"
            ),
            "passes_artifact_parity": True,
            "artifact_parity_claim": True,
            "speed_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        trackb_output_tile_local_codeword_lut_native_parity_path=(
            output_tile_native_parity
        ),
        trackb_output_tile_local_codeword_lut_artifact_parity_path=(
            output_tile_artifact_parity
        ),
        trackb_output_tile_local_codeword_lut_speed_packet_path=(
            output_tile_speed_packet
        ),
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native_parity
        ),
            trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
                route_microtile_artifact_parity
            ),
            trackb_route_microtile_codeword_block_reduce_speed_packet_path=None,
            trackb_kblock_wavefront_codeword_scan_design_path=None,
            trackb_kblock_wavefront_codeword_scan_native_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_artifact_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
            qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == (
        "route_microtile_codeword_block_reduce_artifact_parity_present"
    )
    assert trackb_front[
        "route_microtile_codeword_block_reduce_artifact_parity_decision"
    ] == "route_microtile_codeword_block_reduce_air_artifact_parity_pass"
    assert trackb_front["next"] == (
        "run_route_microtile_codeword_block_reduce_same_window_q2_speed_packet"
    )
    assert report["next_local_slices"][0] == (
        "run_trackb_route_microtile_codeword_block_reduce_same_window_q2_speed_packet"
    )
    assert str(route_microtile_artifact_parity) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False

    route_microtile_speed_packet = _write_json(
        tmp_path / "route_microtile_speed_packet.json",
        {
            **base,
            "record_type": (
                "glm45_air_e8p_route_microtile_codeword_block_reduce_speed_packet"
            ),
            "decision": "reject_route_microtile_codeword_block_reduce_speed_path",
            "same_window_q2_speed_packet": True,
            "all_lane_s_pass": False,
            "speed_claim": False,
            "lane_s_claim": False,
            "resident_auto_claim": False,
        },
    )

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=source_guard,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        trackb_next_design_path=None,
        trackb_active_route_tile_native_parity_path=None,
        trackb_active_route_tile_artifact_parity_path=None,
        trackb_active_route_tile_speed_packet_path=None,
        trackb_expert_cohort_design_path=None,
        trackb_expert_cohort_native_parity_path=None,
        trackb_expert_cohort_artifact_parity_path=None,
        trackb_expert_cohort_speed_packet_path=None,
        trackb_route_batch_segmented_design_path=None,
        trackb_route_batch_segmented_native_parity_path=None,
        trackb_route_batch_segmented_artifact_parity_path=None,
        trackb_route_batch_segmented_speed_packet_path=None,
        trackb_component_stream_partial_reduction_design_path=None,
        trackb_component_stream_partial_reduction_native_parity_path=None,
        trackb_component_stream_tensorops_rejection_path=None,
        trackb_next_family_gate_path=None,
        trackb_token_cohort_codeword_stream_design_path=None,
        trackb_token_cohort_codeword_stream_native_parity_path=None,
        trackb_token_cohort_codeword_stream_artifact_parity_path=None,
        trackb_token_cohort_codeword_stream_speed_packet_path=None,
        trackb_token_cohort_mma_codeword_tile_design_path=None,
        trackb_token_cohort_mma_codeword_tile_native_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=None,
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=None,
        trackb_output_stationary_codeword_tile_design_path=None,
        trackb_output_stationary_codeword_tile_native_parity_path=None,
        trackb_output_stationary_codeword_tile_artifact_parity_path=None,
        trackb_output_stationary_codeword_tile_speed_packet_path=None,
        trackb_input_stationary_codeword_tile_design_path=None,
        trackb_input_stationary_codeword_tile_native_parity_path=None,
        trackb_input_stationary_codeword_tile_artifact_parity_path=None,
        trackb_input_stationary_codeword_tile_speed_packet_path=None,
        trackb_expert_kblock_codeword_factor_reuse_design_path=None,
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=None,
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            expert_kblock_speed
        ),
        trackb_route_codeword_lut_accumulate_design_path=route_codeword_lut_design,
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            route_lut_native_parity
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            route_lut_artifact_parity
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=route_lut_speed_packet,
        trackb_rowwise_codeword_tile_accumulate_design_path=rowwise_design,
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            rowwise_native_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            rowwise_artifact_parity
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            rowwise_speed_packet
        ),
        trackb_output_tile_local_codeword_lut_design_path=output_tile_design,
        trackb_output_tile_local_codeword_lut_native_parity_path=(
            output_tile_native_parity
        ),
        trackb_output_tile_local_codeword_lut_artifact_parity_path=(
            output_tile_artifact_parity
        ),
        trackb_output_tile_local_codeword_lut_speed_packet_path=(
            output_tile_speed_packet
        ),
        trackb_route_microtile_codeword_block_reduce_design_path=(
            route_microtile_design
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            route_microtile_native_parity
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            route_microtile_artifact_parity
        ),
            trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
                route_microtile_speed_packet
            ),
            trackb_kblock_wavefront_codeword_scan_design_path=None,
            trackb_kblock_wavefront_codeword_scan_native_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_artifact_parity_path=None,
            trackb_kblock_wavefront_codeword_scan_speed_packet_path=None,
            qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    trackb_front = report["fronts"]["workstream2_trackb"]
    assert trackb_front["status"] == "trackb_native_speed_circuit_broken"
    assert trackb_front["circuit_breaker"] == {
        "track": "trackb_native_speed",
        "status": "circuit_broken",
        "rejected_family_count": 11,
        "switch_to": "workstream4_cache",
        "requires": "materially_different_rhs_or_kernel_family",
    }
    assert trackb_front[
        "route_microtile_codeword_block_reduce_speed_packet_decision"
    ] == "reject_route_microtile_codeword_block_reduce_speed_path"
    assert trackb_front["next"] == (
        "trackb_circuit_broken_switch_to_w4_cache_cleanup_until_new_family_exists"
    )
    assert report["next_local_slices"][0] == "run_single_host_cache_prefix"
    assert (
        "change_trackb_route_microtile_codeword_block_reduce_layout_or_kernel_family"
        not in report["next_local_slices"]
    )
    assert str(route_microtile_speed_packet) in trackb_front["evidence"]
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_five_front_status_surfaces_qwen_and_glm52_attention(tmp_path: Path) -> None:
    cli = _load_cli()
    base = {
        "peer2_used": False,
        "rdma_jaccl_touched": False,
    }
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "rejected_families": [],
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {
            "family_gate_pass": False,
            "family_gate_status": "qwen_family_gate_not_ready",
            "missing_requirements": ["qwen_family_eval_gate"],
        },
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {"source_artifact_diagnosis": "unknown"},
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {"source_artifact_diagnosis": "unknown"},
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {**base, "local_export_candidate_ready": True, "cache_rows_generated": False},
    )
    model_card = tmp_path / "missing.md"

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    assert report["fronts"]["workstream3_qwen"]["status"] == "needs_attention"
    assert report["fronts"]["workstream3_glm52"]["status"] == "needs_attention"
    assert report["fronts"]["workstream4_cache"]["status"] == (
        "local_single_host_source_ready"
    )
    assert report["fronts"]["workstream5_hygiene"]["status"] == (
        "publication_packet_missing"
    )
    assert report["next_local_slices"][:2] == [
        "finish_glm52_source_artifact_root_cause",
        "repair_qwen_first_family_gate",
    ]


def test_five_front_status_demotes_qwen_placeholder_gate_to_scaffolding(
    tmp_path: Path,
) -> None:
    report = _build_qwen_status_report(
        tmp_path,
        {
            "record_type": "qwen_family_gate_check",
            "family_gate_pass": True,
            "family_gate_status": "qwen_family_gate_ready",
            "qwen_family_thresholds": {
                "eval_gate": {
                    "minimum_clean_rows_per_split": 16,
                    "required_splits": ["report", "selection", "holdout"],
                },
                "benchmark_gate": {
                    "comparison_baseline": "qwen_source_or_qwen_control_runtime",
                },
            },
            "missing_requirements": [],
        },
    )

    qwen_front = report["fronts"]["workstream3_qwen"]
    assert qwen_front["status"] == "first_family_gate_scaffolding"
    assert qwen_front["family_gate_pass"] is True
    assert qwen_front["hardened_family_gate_pass"] is False
    assert qwen_front["public_family_gate_ready"] is False
    assert qwen_front["raw_family_gate_status"] == "qwen_family_gate_ready"
    assert (
        qwen_front["family_gate_status"]
        == "qwen_family_gate_scaffolding_not_public_ready"
    )
    assert qwen_front["qwen_eval_required_row_count"] == 48
    assert (
        qwen_front["qwen_benchmark_reference"]
        == "qwen_source_or_qwen_control_runtime"
    )
    assert qwen_front["hardening_requirements"] == [
        "qwen_eval_prompt_pack_at_least_64_rows",
        "qwen_benchmark_same_machine_reference_ratio",
    ]
    assert report["next_local_slices"][0] == (
        "harden_qwen_family_gate_64row_source_teacher_eval_and_reference_benchmark"
    )


def test_five_front_status_accepts_qwen_hardened_gate_policy(
    tmp_path: Path,
) -> None:
    report = _build_qwen_status_report(
        tmp_path,
        {
            "record_type": "qwen_family_gate_check",
            "family_gate_pass": True,
            "family_gate_status": "qwen_family_gate_ready",
            "qwen_family_thresholds": {
                "eval_gate": {
                    "minimum_clean_rows_per_split": 32,
                    "required_splits": ["report", "selection", "holdout"],
                },
                "benchmark_gate": {
                    "comparison_baseline": "same_machine_qwen_source_runtime",
                },
            },
            "missing_requirements": [],
        },
    )

    qwen_front = report["fronts"]["workstream3_qwen"]
    assert qwen_front["status"] == "hardened_family_gate_green"
    assert qwen_front["family_gate_pass"] is True
    assert qwen_front["hardened_family_gate_pass"] is True
    assert qwen_front["public_family_gate_ready"] is True
    assert qwen_front["raw_family_gate_status"] == "qwen_family_gate_ready"
    assert qwen_front["family_gate_status"] == "qwen_family_gate_ready"
    assert qwen_front["qwen_eval_required_row_count"] == 96
    assert qwen_front["qwen_benchmark_reference"] == "same_machine_qwen_source_runtime"
    assert qwen_front["hardening_requirements"] == []
    assert (
        "harden_qwen_family_gate_64row_source_teacher_eval_and_reference_benchmark"
        not in report["next_local_slices"]
    )


def test_five_front_status_surfaces_runtime_source_ready_memory_risky_cache(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {
        "peer2_used": False,
        "rdma_jaccl_touched": False,
    }
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "rejected_families": [],
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {
            **base,
            "decision": (
                "single_host_teacher_runtime_source_complete_full_source_incomplete"
            ),
            "source_ready_for_single_host_export": True,
            "complete_source": False,
            "complete_runtime_source": True,
            "local_export_candidate_ready": True,
            "cache_rows_generated": False,
            "missing_shards": ["model-00047-of-00047.safetensors"],
            "missing_required_runtime_shards": [],
            "missing_extra_shards": ["model-00047-of-00047.safetensors"],
            "memory_observation": {
                "source_bytes_exceed_physical_memory": True,
                "expected_source_to_physical_memory_ratio": 1.57,
            },
        },
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        model_card_path=model_card,
    )

    cache_front = report["fronts"]["workstream4_cache"]
    assert cache_front["status"] == "local_runtime_source_ready_memory_risky"
    assert cache_front["complete_runtime_source"] is True
    assert cache_front["complete_source"] is False
    assert cache_front["missing_required_runtime_shards"] == []
    assert cache_front["missing_extra_shards"] == [
        "model-00047-of-00047.safetensors"
    ]
    assert cache_front["source_bytes_exceed_physical_memory"] is True
    assert report["next_local_slices"][0] == (
        "run_bounded_single_host_cache_prefix_with_lazy_skip_consume"
    )


def test_five_front_status_surfaces_single_host_cache_prefix_memory_kill(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {
        "peer2_used": False,
        "rdma_jaccl_touched": False,
    }
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "rejected_families": [],
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {
            **base,
            "source_ready_for_single_host_export": True,
            "complete_source": False,
            "complete_runtime_source": True,
            "local_export_candidate_ready": True,
            "cache_rows_generated": False,
            "memory_observation": {"source_bytes_exceed_physical_memory": True},
        },
    )
    cache_attempt = _write_json(
        tmp_path / "single_host_cache_attempt.json",
        {
            **base,
            "decision": "single_host_cache_prefix_killed_exit137",
            "exit_code": 137,
            "cache_rows_generated": False,
            "validation_ran": False,
            "bounded_single_host_attempt": True,
        },
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        single_host_cache_attempt_path=cache_attempt,
        model_card_path=model_card,
    )

    cache_front = report["fronts"]["workstream4_cache"]
    assert cache_front["status"] == "single_host_cache_prefix_memory_killed"
    assert cache_front["single_host_cache_attempt_decision"] == (
        "single_host_cache_prefix_killed_exit137"
    )
    assert cache_front["single_host_cache_attempt_exit_code"] == 137
    assert cache_front["cache_rows_generated"] is False
    assert report["next_local_slices"][0] == (
        "reduce_single_host_cache_loader_peak_or_restore_distributed_cache_path"
    )


def test_five_front_status_surfaces_single_host_source_memory_guard_block(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {
        "peer2_used": False,
        "rdma_jaccl_touched": False,
    }
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "rejected_families": [],
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {
            **base,
            "source_ready_for_single_host_export": True,
            "complete_source": False,
            "complete_runtime_source": True,
            "local_export_candidate_ready": True,
            "cache_rows_generated": False,
            "memory_observation": {"source_bytes_exceed_physical_memory": True},
        },
    )
    cache_attempt = _write_json(
        tmp_path / "single_host_cache_attempt.json",
        {
            **base,
            "decision": "single_host_source_memory_guard_blocked",
            "exit_code": 23,
            "source_memory_guard_pass": False,
            "cache_rows_generated": False,
            "validation_ran": False,
            "bounded_single_host_attempt": True,
        },
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        single_host_cache_attempt_path=cache_attempt,
        model_card_path=model_card,
    )

    cache_front = report["fronts"]["workstream4_cache"]
    assert cache_front["status"] == "single_host_source_memory_guard_blocked"
    assert cache_front["single_host_cache_attempt_decision"] == (
        "single_host_source_memory_guard_blocked"
    )
    assert cache_front["single_host_cache_attempt_exit_code"] == 23
    assert cache_front["source_memory_guard_pass"] is False
    assert report["next_local_slices"][0] == (
        "restore_distributed_cache_path_or_implement_streaming_single_host_source_loader"
    )


def test_five_front_status_surfaces_single_host_cache_prefix_generated_dirty(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {
        "peer2_used": False,
        "rdma_jaccl_touched": False,
    }
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "rejected_families": [],
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {
            **base,
            "source_ready_for_single_host_export": True,
            "complete_source": False,
            "complete_runtime_source": True,
            "local_export_candidate_ready": True,
            "cache_rows_generated": False,
            "memory_observation": {"source_bytes_exceed_physical_memory": True},
        },
    )
    cache_attempt = _write_json(
        tmp_path / "single_host_cache_attempt.json",
        {
            **base,
            "decision": "single_host_cache_prefix_generated_dirty_memory",
            "exit_code": 0,
            "cache_rows_generated": True,
            "validation_ran": True,
            "validation_ok": True,
            "all_memory_clean": False,
            "bounded_single_host_attempt": True,
            "lm_head_chunk_rows": 4096,
            "local_sequential_stage_processes": True,
            "local_sequential_head_process": True,
            "local_sequential_stage_view_plan_json": "stage-plan.json",
            "local_sequential_stage_view_roots_json": "stage-roots.json",
            "local_sequential_stage_view_roots": {
                "head": {"0": "stage-views/head"},
                "lower-pre": {"1": "stage-views/lower-pre"},
            },
            "stage_view_source_visibility": [
                {
                    "stage": "head",
                    "rank": 0,
                    "required_present_tensor_bytes": 1241513984,
                    "visible_shard_file_bytes": 7166072280,
                }
            ],
            "local_sequential_lower_split_layer": 12,
            "local_sequential_upper_split_layer": 34,
            "local_sequential_lower_split_layers": [6, 12, 18],
            "local_sequential_upper_split_layers": [29, 34, 40],
            "local_sequential_lower_window_count": 4,
            "local_sequential_upper_window_count": 4,
            "local_sequential_lower_pre_hidden_host_spill": True,
            "local_sequential_lower_pre_hidden_host_dtype": "bfloat16",
            "local_sequential_lower_pre_hidden_host_shape": [1, 5, 4096],
            "local_sequential_lower_pre_hidden_host_nbytes": 81920,
            "local_sequential_lower_hidden_host_spill": True,
            "local_sequential_lower_hidden_host_dtype": "bfloat16",
            "local_sequential_lower_hidden_host_shape": [1, 5, 4096],
            "local_sequential_lower_hidden_host_nbytes": 81920,
            "local_sequential_upper_pre_hidden_host_spill": True,
            "local_sequential_upper_pre_hidden_host_dtype": "bfloat16",
            "local_sequential_upper_pre_hidden_host_shape": [1, 5, 4096],
            "local_sequential_upper_pre_hidden_host_nbytes": 81920,
            "local_sequential_upper_hidden_host_spill": True,
            "local_sequential_upper_hidden_host_dtype": "bfloat16",
            "local_sequential_upper_hidden_host_shape": [1, 5, 4096],
            "local_sequential_upper_hidden_host_nbytes": 81920,
            "mlx_peak_bytes": 116535330997,
            "pageouts_delta": 1807,
            "swapouts_delta": 3025562,
            "memory_delta_vs_previous_prefix": {
                "previous_attempt": "previous.json",
                "mlx_peak_bytes_delta": 0,
                "pageouts_delta_delta": 233,
                "swapouts_delta_delta": 1062430,
            },
        },
    )
    layer_split_recommendation = _write_json(
        tmp_path / "layer_split_recommendation.json",
        {
            "recommended_layer_split": 23,
            "candidate_count": 45,
            "optimization_target": (
                "minimize_max_required_present_tensor_bytes_then_visible_shard_file_bytes"
            ),
            "previous_legacy_recipe_layer_split": 24,
            "updated_legacy_recipe_layer_split": 23,
        },
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        single_host_cache_attempt_path=cache_attempt,
        layer_split_recommendation_path=layer_split_recommendation,
        model_card_path=model_card,
    )

    cache_front = report["fronts"]["workstream4_cache"]
    assert cache_front["status"] == "single_host_cache_prefix_generated_dirty_memory"
    assert cache_front["cache_rows_generated"] is True
    assert cache_front["single_host_cache_attempt_decision"] == (
        "single_host_cache_prefix_generated_dirty_memory"
    )
    assert cache_front["single_host_cache_all_memory_clean"] is False
    assert cache_front["single_host_cache_lm_head_chunk_rows"] == 4096
    assert cache_front["single_host_cache_stage_processes"] is True
    assert cache_front["single_host_cache_head_process"] is True
    assert cache_front["single_host_cache_stage_view_plan_json"] == "stage-plan.json"
    assert cache_front["single_host_cache_stage_view_roots_json"] == "stage-roots.json"
    assert cache_front["single_host_cache_stage_view_roots"] == {
        "head": {"0": "stage-views/head"},
        "lower-pre": {"1": "stage-views/lower-pre"},
    }
    assert cache_front["single_host_cache_stage_view_source_visibility"] == [
        {
            "stage": "head",
            "rank": 0,
            "required_present_tensor_bytes": 1241513984,
            "visible_shard_file_bytes": 7166072280,
        }
    ]
    assert cache_front["single_host_cache_lower_split_layer"] == 12
    assert cache_front["single_host_cache_upper_split_layer"] == 34
    assert cache_front["single_host_cache_lower_split_layers"] == [6, 12, 18]
    assert cache_front["single_host_cache_upper_split_layers"] == [29, 34, 40]
    assert cache_front["single_host_cache_lower_window_count"] == 4
    assert cache_front["single_host_cache_upper_window_count"] == 4
    assert cache_front["single_host_cache_lower_pre_hidden_host_spill"] is True
    assert cache_front["single_host_cache_lower_pre_hidden_host_dtype"] == "bfloat16"
    assert cache_front["single_host_cache_lower_pre_hidden_host_shape"] == [1, 5, 4096]
    assert cache_front["single_host_cache_lower_pre_hidden_host_nbytes"] == 81920
    assert cache_front["single_host_cache_lower_hidden_host_spill"] is True
    assert cache_front["single_host_cache_lower_hidden_host_dtype"] == "bfloat16"
    assert cache_front["single_host_cache_lower_hidden_host_shape"] == [1, 5, 4096]
    assert cache_front["single_host_cache_lower_hidden_host_nbytes"] == 81920
    assert cache_front["single_host_cache_upper_pre_hidden_host_spill"] is True
    assert cache_front["single_host_cache_upper_pre_hidden_host_dtype"] == "bfloat16"
    assert cache_front["single_host_cache_upper_pre_hidden_host_shape"] == [1, 5, 4096]
    assert cache_front["single_host_cache_upper_pre_hidden_host_nbytes"] == 81920
    assert cache_front["single_host_cache_upper_hidden_host_spill"] is True
    assert cache_front["single_host_cache_upper_hidden_host_dtype"] == "bfloat16"
    assert cache_front["single_host_cache_upper_hidden_host_shape"] == [1, 5, 4096]
    assert cache_front["single_host_cache_upper_hidden_host_nbytes"] == 81920
    assert cache_front["single_host_cache_mlx_peak_bytes"] == 116535330997
    assert cache_front["single_host_cache_pageouts_delta"] == 1807
    assert cache_front["single_host_cache_swapouts_delta"] == 3025562
    assert cache_front["single_host_cache_memory_delta_vs_previous_prefix"] == {
        "previous_attempt": "previous.json",
        "mlx_peak_bytes_delta": 0,
        "pageouts_delta_delta": 233,
        "swapouts_delta_delta": 1062430,
    }
    assert cache_front["rank0_logits_layer_split_recommendation"] == {
        "recommended_layer_split": 23,
        "candidate_count": 45,
        "optimization_target": (
            "minimize_max_required_present_tensor_bytes_then_visible_shard_file_bytes"
        ),
        "previous_legacy_recipe_layer_split": 24,
        "updated_legacy_recipe_layer_split": 23,
    }
    assert str(layer_split_recommendation) in cache_front["evidence"]
    assert "single_host_teacher_cache_prefix_memory_dirty" in report[
        "completion_blockers"
    ]
    assert "distributed_or_single_host_teacher_cache_not_generated" not in report[
        "completion_blockers"
    ]
    assert report["next_local_slices"][0] != (
        "restore_distributed_cache_path_or_implement_streaming_single_host_source_loader"
    )


def test_five_front_status_surfaces_rdma_topology_down_gate(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    base = {
        "peer2_used": False,
        "rdma_jaccl_touched": False,
    }
    workstream1 = _write_json(
        tmp_path / "w1.json",
        {**base, "decision": "reject_free_p1_levers_metric_backed"},
    )
    trackb = _write_json(
        tmp_path / "trackb.json",
        {
            **base,
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "rejected_families": [],
        },
    )
    qwen = _write_json(
        tmp_path / "qwen.json",
        {"family_gate_pass": True, "family_gate_status": "qwen_family_gate_ready"},
    )
    glm52_layer3 = _write_json(
        tmp_path / "glm52_l3.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    glm52_layer77 = _write_json(
        tmp_path / "glm52_l77.json",
        {
            "source_artifact_diagnosis": (
                "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
            ),
        },
    )
    cache = _write_json(
        tmp_path / "cache.json",
        {
            **base,
            "source_ready_for_single_host_export": True,
            "complete_runtime_source": True,
            "local_export_candidate_ready": True,
            "cache_rows_generated": False,
        },
    )
    topology = _write_json(
        tmp_path / "topology.json",
        {
            "decision": "rdma_link_down",
            "direct_path_ready": False,
            "link_layer_ready": False,
            "direct_ip_ready": False,
            "direct_route_ready": False,
            "direct_arp_ready": False,
            "rdma_ipv4_gid_ready": False,
            "ready_for_uc_pingpong": False,
            "ready_for_jaccl_preflight": False,
            "local": {
                "interface": {"status": "inactive", "has_expected_ip": False},
                "route": {"interface": "en0", "uses_expected_interface": False},
                "rdma": {"port_active": False, "has_ipv4_gid": False},
            },
            "peer": {
                "interface": {"status": "inactive", "has_expected_ip": False},
                "route": {"interface": "utun13", "uses_expected_interface": False},
                "rdma": {"port_active": False, "has_ipv4_gid": False},
            },
        },
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("draft", encoding="utf-8")

    report = cli.build_status(
        workstream1_path=workstream1,
        trackb_path=trackb,
        trackb_design_path=None,
        trackb_source_guard_path=None,
        trackb_native_parity_path=None,
        trackb_artifact_parity_path=None,
        trackb_speed_packet_path=None,
        trackb_successor_design_path=None,
        trackb_successor_native_parity_path=None,
        trackb_successor_artifact_parity_path=None,
        trackb_successor_speed_packet_path=None,
        qwen_gate_path=qwen,
        glm52_layer3_path=glm52_layer3,
        glm52_layer77_path=glm52_layer77,
        cache_source_scan_path=cache,
        rdma_topology_audit_path=topology,
        model_card_path=model_card,
    )

    cache_front = report["fronts"]["workstream4_cache"]
    assert cache_front["rdma_topology"] == {
        "decision": "rdma_link_down",
        "direct_path_ready": False,
        "link_layer_ready": False,
        "direct_ip_ready": False,
        "direct_route_ready": False,
        "direct_arp_ready": False,
        "rdma_ipv4_gid_ready": False,
        "ready_for_uc_pingpong": False,
        "ready_for_jaccl_preflight": False,
        "local_interface_status": "inactive",
        "peer_interface_status": "inactive",
        "local_route_interface": "en0",
        "peer_route_interface": "utun13",
        "local_rdma_port_active": False,
        "peer_rdma_port_active": False,
    }
    assert str(topology) in cache_front["evidence"]
    assert report["transport_boundary"] == (
        "peer2_read_only_topology_checked_direct_path_down"
    )
