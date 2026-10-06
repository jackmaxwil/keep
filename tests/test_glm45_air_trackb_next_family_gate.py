from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "summarize_glm45_air_trackb_next_family_gate.py"
    )
    spec = importlib.util.spec_from_file_location("trackb_next_family_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_next_family_gate_blocks_stale_expert_kblock_v2_plan_route(tmp_path: Path) -> None:
    cli = _load_cli()
    frontier = _write_json(
        tmp_path / "frontier.json",
        {
            "record_type": "glm45_air_e8p_trackb_frontier_summary",
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "peer2_used": False,
            "rdma_jaccl_touched": False,
            "speed_claim": False,
            "native_parity_claim": False,
            "rejected_families": [
                {"evidence": "current_family"},
                {"evidence": "route_abs_component_lower_bound"},
            ],
            "next_required_features": [
                "materially_different_rhs_layout_or_kernel_family"
            ],
            "rejected_next_steps": ["do_not_time_current_e8p16_families"],
        },
    )
    plan = tmp_path / "TRACK_B.md"
    plan.write_text(
        "Working name:\n"
        "`nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2`\n",
        encoding="utf-8",
    )

    report = cli.build_next_family_gate(frontier, plan)

    assert report["record_type"] == "glm45_air_e8p_trackb_next_family_gate"
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert report["decision"] == (
        "block_stale_trackb_plan_variant_requires_materially_new_family_design"
    )
    assert report["rejected_family_count"] == 2
    assert report["stale_plan_variant_blocked"] is True
    assert report["blocked_working_names"] == [
        "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2"
    ]
    assert "supersede_stale_expert_kblock_v2_plan_route" in report[
        "required_next_features"
    ]
    assert (
        "do_not_implement_nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_unchanged"
        in report["rejected_next_steps"]
    )


def test_next_family_gate_stays_local_and_flags_transport_claims(tmp_path: Path) -> None:
    cli = _load_cli()
    frontier = _write_json(
        tmp_path / "frontier.json",
        {
            "frontier_decision": "track_b_current_frontier_requires_new_rhs_or_kernel_family",
            "peer2_used": True,
            "rdma_jaccl_touched": False,
            "speed_claim": False,
            "native_parity_claim": False,
            "rejected_families": [],
            "next_required_features": [],
            "rejected_next_steps": [],
        },
    )
    plan = tmp_path / "TRACK_B.md"
    plan.write_text("No stale working name here.\n", encoding="utf-8")

    report = cli.build_next_family_gate(frontier, plan)

    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert report["decision"] == (
        "track_b_next_family_gate_incomplete_transport_claims_present"
    )
    assert report["transport_violations"] == [
        {"claim": "peer2_used", "value": True}
    ]
