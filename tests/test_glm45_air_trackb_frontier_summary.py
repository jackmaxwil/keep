from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_summarizer():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "summarize_glm45_air_trackb_frontier.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_glm45_air_trackb_frontier", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n")


def _record(record_type: str, decision: str, *path: str) -> dict:
    record: dict = {
        "record_type": record_type,
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "speed_claim": False,
        "native_parity_claim": False,
    }
    cursor = record
    for key in path[:-1]:
        cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = decision
    return record


def test_build_summary_accepts_complete_local_no_go_frontier(tmp_path: Path) -> None:
    summarizer = _load_summarizer()
    paths = {}
    for spec in summarizer.EVIDENCE_SPECS:
        path = tmp_path / f"{spec.key}.json"
        _write(path, _record(spec.record_type, spec.expected_decision, *spec.decision_path))
        paths[spec.key] = path

    summary = summarizer.build_summary(paths)

    assert summary["frontier_decision"] == (
        "track_b_current_frontier_requires_new_rhs_or_kernel_family"
    )
    assert summary["peer2_used"] is False
    assert summary["rdma_jaccl_touched"] is False
    assert summary["speed_claim"] is False
    assert summary["native_parity_claim"] is False
    assert summary["ready_for_native_speed_work"] is False
    assert summary["missing_evidence"] == []
    assert summary["mismatched_evidence"] == []
    assert summary["transport_violations"] == []
    assert {
        item["decision"] for item in summary["rejected_families"]
    } == {spec.expected_decision for spec in summarizer.EVIDENCE_SPECS}
    assert {
        item["evidence"] for item in summary["rejected_families"]
    } >= {
        "route_abs_component_tile_sweep",
        "route_abs_component_token_reuse",
        "route_abs_component_token_active_expert",
        "route_abs_component_lower_bound",
    }
    assert "do_not_claim_exactness_from_route_tile_level_cache_bytes" in summary[
        "rejected_next_steps"
    ]
    assert (
        "preserve_route_slot_axis_or_prove_equivalent_rowwise_reconstruction"
        in summary["next_required_features"]
    )
    assert "do_not_benchmark_token_reuse_abs_component_cache" in summary[
        "rejected_next_steps"
    ]
    assert "do_not_benchmark_token_active_expert_abs_component_cache" in summary[
        "rejected_next_steps"
    ]
    assert "do_not_benchmark_exact_route_abs_component_projection_family" in summary[
        "rejected_next_steps"
    ]


def test_build_summary_flags_mismatch_and_transport_claim(tmp_path: Path) -> None:
    summarizer = _load_summarizer()
    paths = {}
    for spec in summarizer.EVIDENCE_SPECS:
        path = tmp_path / f"{spec.key}.json"
        decision = spec.expected_decision
        if spec.key == "route_abs_index":
            decision = "candidate_family_ready_for_source_structure_probe"
        record = _record(spec.record_type, decision, *spec.decision_path)
        if spec.key == "route_codeword":
            record["peer2_used"] = True
        _write(path, record)
        paths[spec.key] = path

    summary = summarizer.build_summary(paths)

    assert summary["frontier_decision"] == "track_b_frontier_summary_incomplete"
    assert summary["missing_evidence"] == []
    assert summary["mismatched_evidence"] == [
        {
            "evidence": "route_abs_index",
            "expected_record_type": "glm45_air_e8p_route_abs_index_projection_feasibility",
            "actual_record_type": "glm45_air_e8p_route_abs_index_projection_feasibility",
            "decision_path": ["feasibility", "decision"],
            "expected_decision": "reject_route_abs_index_projection_requires_sign_axis",
            "actual_decision": "candidate_family_ready_for_source_structure_probe",
        }
    ]
    assert summary["transport_violations"] == [
        {"evidence": "route_codeword", "claim": "peer2_used", "value": True}
    ]
