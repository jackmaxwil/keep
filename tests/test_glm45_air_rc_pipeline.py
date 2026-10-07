from __future__ import annotations

import json
import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_vq.quality import rc_gates


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_glm45_air_rc_pipeline.py"
_REPO_ROOT = _SCRIPT_PATH.parents[1]
_SPEC = importlib.util.spec_from_file_location("run_glm45_air_rc_pipeline", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
rc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rc)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_source_snapshot(path: Path, *, missing_runtime: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"num_hidden_layers": 2}) + "\n")
    weight_map = {
        "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
        "model.layers.1.mlp.experts.0.gate_proj.weight": "model-00002-of-00003.safetensors",
        "model.layers.2.mlp.experts.0.gate_proj.weight": "model-00003-of-00003.safetensors",
    }
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}) + "\n")
    (path / "model-00001-of-00003.safetensors").write_bytes(b"runtime-a")
    if not missing_runtime:
        (path / "model-00002-of-00003.safetensors").write_bytes(b"runtime-b")
    return path


def _write_quality_compare(
    path: Path,
    *,
    mean_kld: float,
    top1: float,
    p999: float,
    delta_kld: float = -0.05,
    delta_top1: float = 0.05,
    delta_p999: float = -1.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "candidate_summary": {
                    "record_count": 128,
                    "clean_record_count": 128,
                    "all_memory_clean": True,
                    "mean_kld": mean_kld,
                    "mean_top1_agreement": top1,
                    "mean_ppl_ratio": 0.97,
                    "max_ppl_ratio": 1.60,
                    "p999_kld": p999,
                },
                "baseline_summary": {
                    "record_count": 128,
                    "clean_record_count": 128,
                    "all_memory_clean": True,
                    "mean_kld": mean_kld - delta_kld,
                    "mean_top1_agreement": top1 - delta_top1,
                    "mean_ppl_ratio": 1.04,
                    "max_ppl_ratio": 1.74,
                    "p999_kld": p999 - delta_p999,
                },
                "summary_delta": {
                    "mean_kld": delta_kld,
                    "mean_top1_agreement": delta_top1,
                    "mean_ppl_ratio": -0.07,
                    "max_ppl_ratio": -0.14,
                    "p999_kld": delta_p999,
                },
                "decision": {
                    "label": "RECOVER",
                    "quality_label_without_speed": "RECOVER",
                },
            }
        )
        + "\n"
    )


def _run_rc_shell(*args: str, dry_run: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if dry_run:
        env["GLM45_AIR_RC_DRY_RUN"] = "1"
    return subprocess.run(
        ["bash", "scripts/glm45_air_rc.sh", *args],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_assert_engine_tier_compatible_rejects_e8p_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _audit(_artifact_dir: Path) -> dict:
        return {
            "auto_prefill_all_layers_nax_e8_compatible": False,
            "projection_rows": [
                {
                    "layer": 45,
                    "projection": "gate_proj",
                    "storage": "vq",
                    "code_bits": 16,
                },
                {
                    "layer": 45,
                    "projection": "up_proj",
                    "storage": "vq",
                    "code_bits": 8,
                },
            ],
        }

    monkeypatch.setattr(rc_gates, "audit_vq_artifact_prefill_compatibility", _audit)

    with pytest.raises(ValueError, match="layer 45 gate_proj .*code_bits=16"):
        rc_gates.assert_engine_tier_compatible(tmp_path, "vq_e1_routed_nax_e8")


def test_assert_engine_tier_compatible_allows_e8p_engine_without_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _audit(_artifact_dir: Path) -> dict:
        raise AssertionError("nax_e8p engines should not require the E8 audit guard")

    monkeypatch.setattr(rc_gates, "audit_vq_artifact_prefill_compatibility", _audit)

    rc_gates.assert_engine_tier_compatible(tmp_path, "vq_e1_routed_nax_e8p")


def test_summarize_benchmark_requires_clean_rows(tmp_path: Path) -> None:
    path = tmp_path / "bench.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prefill_seconds": 1.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "effective_bits_per_weight": 2.0,
                "dense_expert_params": False,
                "unbound_vq_experts": False,
                "non_expert_dtype_verified": True,
            },
            {
                "prefill_seconds": 3.0,
                "pageouts_delta": 1,
                "swapouts_delta": 0,
                "effective_bits_per_weight": 2.0,
                "dense_expert_params": False,
                "unbound_vq_experts": False,
                "non_expert_dtype_verified": True,
            },
        ],
    )

    summary = rc._summarize_benchmark(path)

    assert summary["record_count"] == 2
    assert summary["clean_record_count"] == 1
    assert summary["all_memory_clean"] is False
    assert summary["median_seconds"] == 1.0
    assert summary["clean_timing"]["seconds"] == [1.0]
    assert summary["attempted_timing"]["seconds"] == [1.0, 3.0]
    assert summary["attempted_timing"]["median_seconds"] == 2.0
    assert summary["memory_pressure_summary"]["dirty_record_count"] == 1
    assert summary["memory_pressure_summary"]["pageouts_delta_sum"] == 1


def test_lane_s_rejects_timing_unstable_clean_rows(tmp_path: Path) -> None:
    path = tmp_path / "bench.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prefill_seconds": 1.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            },
            {
                "prefill_seconds": 1.9,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            },
            {
                "prefill_seconds": 2.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            },
        ],
    )

    summary = rc._summarize_benchmark(path)
    checks = {
        "candidate_clean": bool(summary["all_memory_clean"]),
        "candidate_timing_stable": bool(summary["timing_stable"]),
    }

    assert summary["all_memory_clean"] is True
    assert summary["median_seconds"] == 1.9
    assert summary["relative_spread"] > rc.LANE_S_TIMING_RELATIVE_SPREAD_MAX
    assert summary["timing_stable"] is False
    assert rc._lane_s_failure_reasons(checks) == ["candidate_timing_unstable_or_unavailable"]


def test_summarize_benchmark_compacts_parent_vm_stat_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "bench.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prefill_seconds": 1.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "parent_vm_stat": {
                    "available": True,
                    "deltas": {
                        "pageouts": 0,
                        "swapouts": 0,
                        "pages_free": -10,
                        "pages_active": 20,
                    },
                },
            },
            {
                "prefill_seconds": 3.0,
                "pageouts_delta": 5,
                "swapouts_delta": 0,
                "invalid_memory_pressure": True,
                "parent_vm_stat": {
                    "available": True,
                    "deltas": {
                        "pageouts": 7,
                        "swapouts": 0,
                        "pages_free": -30,
                        "pages_occupied_by_compressor": 40,
                    },
                },
            },
        ],
    )

    summary = rc._summarize_benchmark(path)
    parent_vm = summary["parent_vm_stat_summary"]

    assert parent_vm["available_record_count"] == 2
    assert parent_vm["dirty_available_record_count"] == 1
    assert parent_vm["selected_delta_sums"]["pageouts"] == 7
    assert parent_vm["selected_delta_sums"]["pages_free"] == -40
    assert parent_vm["dirty_selected_delta_sums"]["pageouts"] == 7
    assert parent_vm["dirty_selected_delta_sums"]["pages_occupied_by_compressor"] == 40


def test_quality_frontier_records_rejected_quality_attempt(tmp_path: Path, monkeypatch) -> None:
    compare_jsons = {
        "report": tmp_path / "report-compare.json",
        "selection": tmp_path / "selection-compare.json",
        "holdout": tmp_path / "holdout-compare.json",
    }
    _write_quality_compare(compare_jsons["report"], mean_kld=0.314, top1=0.852, p999=3.58)
    _write_quality_compare(compare_jsons["selection"], mean_kld=0.319, top1=0.861, p999=3.58)
    _write_quality_compare(compare_jsons["holdout"], mean_kld=0.305, top1=0.850, p999=3.58)
    candidate_lane_s = tmp_path / "candidate-lane-s.jsonl"
    q2_lane_s = tmp_path / "q2-lane-s.jsonl"
    _write_jsonl(
        candidate_lane_s,
        [
            {
                "prefill_seconds": 1.62,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "effective_bits_per_weight": 2.04,
                "dense_expert_params": False,
                "unbound_vq_experts": False,
                "non_expert_dtype_verified": True,
            },
            {
                "prefill_seconds": 1.63,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "effective_bits_per_weight": 2.04,
                "dense_expert_params": False,
                "unbound_vq_experts": False,
                "non_expert_dtype_verified": True,
            },
        ],
    )
    _write_jsonl(
        q2_lane_s,
        [
            {
                "prefill_seconds": 0.94,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "effective_bits_per_weight": 2.25,
                "dense_expert_params": False,
                "unbound_vq_experts": False,
                "non_expert_dtype_verified": True,
            },
            {
                "prefill_seconds": 0.95,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "effective_bits_per_weight": 2.25,
                "dense_expert_params": False,
                "unbound_vq_experts": False,
                "non_expert_dtype_verified": True,
            },
        ],
    )

    # Hermetic: this test exercises a genuine lane_s ratio rejection, independent of
    # any real golden-q2 baseline that may exist on disk. Disable the golden guard so
    # the suspect-window reclassification does not fire on the synthetic fixture q2.
    monkeypatch.setattr(rc, "_load_golden_q2_baseline", lambda: None)

    frontier = rc._build_quality_frontier(
        compare_jsons=compare_jsons,
        lane_s_jsonl=candidate_lane_s,
        q2_control_jsonl=q2_lane_s,
        artifact_dir="quality-artifact",
    )

    assert frontier["promoted"] == "balanced"
    attempt = frontier["attempts"][0]
    assert attempt["promotion_status"] == "quality_candidate_rejected"
    assert "lane_s_speed_or_invariant_gate_failed" in attempt["rejection_reasons"]
    assert "community_wow_quality_targets_still_missed" in attempt["rejection_reasons"]
    assert attempt["split_compares"]["report"]["candidate_summary"]["mean_kld"] == 0.314
    assert attempt["split_compares"]["report"]["summary_delta"]["mean_top1_agreement"] == 0.05
    assert attempt["lane_s"]["checks"]["candidate_clean"] is True
    assert attempt["lane_s"]["checks"]["candidate_timing_stable"] is True
    assert attempt["lane_s"]["checks"]["q2_control_timing_stable"] is True
    assert attempt["lane_s"]["checks"]["ratio_le_1p15"] is False
    assert attempt["lane_s"]["failure_reasons"] == ["lane_s_ratio_above_target_or_unavailable"]


def test_quality_frontier_reports_missing_evidence(tmp_path: Path) -> None:
    frontier = rc._build_quality_frontier(
        compare_jsons={"report": tmp_path / "missing-report.json"},
        lane_s_jsonl=tmp_path / "missing-candidate.jsonl",
        q2_control_jsonl=tmp_path / "missing-q2.jsonl",
    )

    assert frontier["status"] == "quality_frontier_evidence_unavailable"
    assert frontier["promoted"] == "balanced"
    assert len(frontier["missing_paths"]) == 3


def test_quality_checks_split_hard_and_wow_targets() -> None:
    summary = {
        "clean_record_count": 128,
        "all_memory_clean": True,
        "mean_ppl_ratio": 1.04,
        "mean_kld": 0.29,
        "p999_kld": 2.9,
        "mean_top1_agreement": 0.86,
    }

    checks = rc._quality_checks(summary)

    assert checks == {
        "clean_128": True,
        "mean_ppl_ratio_le_1p05": True,
        "mean_kld_le_0p30": True,
        "p999_kld_le_3p0": True,
        "top1_ge_0p85": True,
    }


def test_domain_summaries_parse_prompt_ids_and_check_top1() -> None:
    records = [
        {
            "prompt_id": "report_code_000",
            "nll_delta": 0.0,
            "ppl_ratio": 1.0,
            "mean_kld": 0.2,
            "token_klds": [0.1, 0.2],
            "top1_agreement": 0.9,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
        {
            "prompt_id": "report_route_000",
            "nll_delta": 0.0,
            "ppl_ratio": 1.1,
            "mean_kld": 0.4,
            "token_klds": [0.3, 0.4],
            "top1_agreement": 0.7,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    ]

    domains = rc._summarize_eval_domains(records)

    assert sorted(domains) == ["code", "route"]
    assert domains["code"]["record_count"] == 1
    assert domains["code"]["top1_ge_0p80"] is True
    assert domains["route"]["top1_ge_0p80"] is False
    assert rc._domain_from_record({"prompt_id": "holdout_math_012"}) == "math"


def test_quality_focus_ranks_top1_and_token_tails() -> None:
    records = [
        {
            "prompt_id": "report_math_000",
            "row_index": 0,
            "positions": [10, 11],
            "target_token_ids": [100, 101],
            "nll_delta": 0.0,
            "ppl_ratio": 1.2,
            "mean_kld": 0.5,
            "token_klds": [0.2, 9.0],
            "top1_agreement": 0.4,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
        {
            "prompt_id": "report_code_000",
            "row_index": 1,
            "positions": [20, 21],
            "target_token_ids": [200, 201],
            "nll_delta": 0.0,
            "ppl_ratio": 1.0,
            "mean_kld": 0.9,
            "token_klds": [2.0, 3.0],
            "top1_agreement": 0.8,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    ]

    focus = rc._build_quality_focus(records, limit=1)

    assert focus["lowest_top1_rows"][0]["prompt_id"] == "report_math_000"
    assert focus["highest_mean_kld_rows"][0]["prompt_id"] == "report_code_000"
    token_tail = focus["highest_token_kld_rows"][0]
    assert token_tail["prompt_id"] == "report_math_000"
    assert token_tail["max_token_kld"] == 9.0
    assert token_tail["max_token_position"] == 11
    assert token_tail["max_token_target_id"] == 101


def test_focus_export_contains_domains_and_focus_rows() -> None:
    report = {
        "schema_version": 1,
        "artifact_dir": "artifact",
        "overall_status": "balanced_rc_pass_community_wow_miss",
        "targets": {
            "community_wow": {
                "domain_top1_min": 0.8,
                "mean_kld_max": 0.3,
                "p999_kld_max": 3.0,
                "top1_min": 0.85,
            }
        },
        "eval_splits": {
            "report": {
                "domains": {"math": {"mean_top1_agreement": 0.7}},
                "quality_focus": {"lowest_top1_rows": [{"prompt_id": "report_math_000"}]},
            }
        },
    }

    export = rc._build_focus_export(report)

    assert export["artifact_dir"] == "artifact"
    assert export["targets"]["domain_top1_min"] == 0.8
    assert export["splits"]["report"]["domains"]["math"]["mean_top1_agreement"] == 0.7
    assert export["splits"]["report"]["quality_focus"]["lowest_top1_rows"][0]["prompt_id"] == "report_math_000"


def test_quality_frontier_export_is_compact_and_promoted_aware() -> None:
    report = {
        "schema_version": 1,
        "artifact_dir": "artifact",
        "overall_status": "balanced_rc_pass_community_wow_miss",
        "targets": {
            "balanced_hard": {"lane_s_ratio_max": 1.15},
            "community_wow": {"mean_kld_max": 0.30},
        },
        "quality_frontier": {
            "schema_version": 1,
            "status": "balanced_promoted_quality_attempt_recorded",
            "promoted": "balanced",
            "attempts": [
                {
                    "label": "quality-r4-route-math-plan1",
                    "decision": "RECOVER_quality_rejected_speed",
                }
            ],
        },
    }

    export = rc._build_quality_frontier_export(report)

    assert export["schema_version"] == 1
    assert export["artifact_dir"] == "artifact"
    assert export["overall_status"] == "balanced_rc_pass_community_wow_miss"
    assert export["promoted"] == "balanced"
    assert export["targets"]["balanced_hard"]["lane_s_ratio_max"] == 1.15
    assert export["targets"]["community_wow"]["mean_kld_max"] == 0.30
    assert export["quality_frontier"]["attempts"][0]["decision"] == "RECOVER_quality_rejected_speed"


def test_rc_shell_help_lists_public_commands() -> None:
    result = _run_rc_shell("help", dry_run=False)

    assert "env-preflight" in result.stdout
    assert "packet" in result.stdout
    assert "reproduce" in result.stdout
    assert "verify" in result.stdout
    assert "memory-preflight" in result.stdout
    assert "attribute" in result.stdout
    assert "attribute-report" in result.stdout
    assert "attribute-selection" in result.stdout
    assert "attribute-holdout" in result.stdout
    assert "benchmark-candidate" in result.stdout
    assert "benchmark-candidate-cache1" in result.stdout
    assert "benchmark-candidate-cache-sweep" in result.stdout
    assert "collect-imatrix" in result.stdout
    assert "materialize-sweep" in result.stdout
    assert "train-low-rank" in result.stdout
    assert "new-model-template" in result.stdout
    assert "scripts/glm45_air_rc.sh preflight" in result.stdout


def test_rc_shell_script_is_directly_executable() -> None:
    path = _REPO_ROOT / "scripts" / "glm45_air_rc.sh"

    assert path.is_file()
    assert os.access(path, os.X_OK)


def test_path_check_can_require_executable_entrypoint(tmp_path: Path) -> None:
    path = tmp_path / "wrapper.sh"
    path.write_text("#!/usr/bin/env bash\nexit 0\n")
    path.chmod(0o644)

    check = rc._path_check("entrypoint:wrapper", path, kind="file", executable=True)

    assert check["exists"] is True
    assert check["type_ok"] is True
    assert check["executable_required"] is True
    assert check["executable_ok"] is False
    assert check["ok"] is False

    path.chmod(0o755)
    assert rc._path_check("entrypoint:wrapper", path, kind="file", executable=True)["ok"] is True


def test_public_doc_check_reports_missing_required_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = tmp_path / "README.md"
    doc.write_text("# KEEP\n\n## Quick Start\n")
    monkeypatch.setitem(
        rc.REQUIRED_PUBLIC_DOC_PATTERNS,
        str(doc),
        ("# KEEP", "## Quick Start", "## Current Limitations"),
    )

    check = rc._public_doc_check(doc)

    assert check["exists"] is True
    assert check["content_ok"] is False
    assert check["ok"] is False
    assert check["missing_patterns"] == ["## Current Limitations"]


def test_rc_shell_packet_dry_run_writes_all_public_exports() -> None:
    result = _run_rc_shell("packet", "--overwrite", "--output-dir", "out")

    assert "uv run python benchmarks/run_glm45_air_rc_pipeline.py" in result.stdout
    assert "--overwrite" in result.stdout
    assert "--output-dir out" in result.stdout
    assert "--write-focus-json out/quality-focus.json" in result.stdout
    assert "--write-quality-frontier-json out/quality-frontier.json" in result.stdout
    assert "--write-quality-plan-json out/quality-plan.json" in result.stdout


def test_rc_shell_env_preflight_runs_environment_verifier() -> None:
    result = _run_rc_shell("env-preflight")

    assert "uv run python scripts/verify_env.py" in result.stdout


def test_rc_shell_verify_dry_run_executes_full_public_rc_path() -> None:
    result = _run_rc_shell("verify", "--overwrite", "--output-dir", "out")

    assert "uv run python benchmarks/run_glm45_air_rc_pipeline.py" in result.stdout
    assert "--execute audit" in result.stdout
    assert "--execute eval_report" in result.stdout
    assert "--execute eval_selection" in result.stdout
    assert "--execute eval_holdout" in result.stdout
    assert "--execute lane_s_q2_control" in result.stdout
    assert "--execute lane_s_candidate" in result.stdout
    assert "--overwrite" in result.stdout
    assert "--output-dir out" in result.stdout
    assert "--write-focus-json out/quality-focus.json" in result.stdout
    assert "--write-quality-frontier-json out/quality-frontier.json" in result.stdout
    assert "--write-quality-plan-json out/quality-plan.json" in result.stdout


def test_rc_shell_reproduce_dry_run_trains_then_verifies_reproduced_artifact() -> None:
    result = _run_rc_shell("reproduce", "--overwrite", "--output-dir", "out")
    lines = [line for line in result.stdout.splitlines() if line.strip()]

    assert len(lines) == 2
    assert "benchmarks/finetune_glm45_air_vq_continuous.py" in lines[0]
    assert "--trainable low_rank_residual" in lines[0]
    assert "--output-dir artifacts/glm45-air-public-reproduction-low-rank-residual-r4" in lines[0]
    assert "benchmarks/run_glm45_air_rc_pipeline.py" in lines[1]
    assert "--execute audit" in lines[1]
    assert "--execute eval_report" in lines[1]
    assert "--execute eval_selection" in lines[1]
    assert "--execute eval_holdout" in lines[1]
    assert "--execute lane_s_q2_control" in lines[1]
    assert "--execute lane_s_candidate" in lines[1]
    assert "--artifact-dir artifacts/glm45-air-public-reproduction-low-rank-residual-r4" in lines[1]
    assert "--output-dir out" in lines[1]
    assert "--write-quality-plan-json out/quality-plan.json" in lines[1]


def test_rc_shell_benchmark_candidate_dry_run_maps_to_execute_flag() -> None:
    result = _run_rc_shell("benchmark-candidate", "--overwrite")

    assert "--execute lane_s_candidate" in result.stdout
    assert "--overwrite" in result.stdout


def test_rc_shell_benchmark_lane_s_dry_run_executes_q2_and_candidate() -> None:
    result = _run_rc_shell("benchmark-lane-s", "--overwrite", "--output-dir", "out")

    assert "--execute lane_s_q2_control" in result.stdout
    assert "--execute lane_s_candidate" in result.stdout
    assert "--overwrite" in result.stdout
    assert "--output-dir out" in result.stdout


def test_rc_shell_attribute_dry_run_exports_all_accepted_splits() -> None:
    result = _run_rc_shell("attribute", "--overwrite", "--output-dir", "out")
    lines = [line for line in result.stdout.splitlines() if line.strip()]

    assert len(lines) == 3
    assert all("benchmarks/analyze_glm45_air_teacher_cache_attribution.py" in line for line in lines)
    assert "--baseline-label balanced_rc_report" in lines[0]
    assert "--output-json out/report-attribution.json" in lines[0]
    assert "--baseline-label balanced_rc_selection" in lines[1]
    assert "--output-json out/selection-attribution.json" in lines[1]
    assert "--baseline-label balanced_rc_holdout" in lines[2]
    assert "--output-json out/holdout-attribution.json" in lines[2]
    assert "--output-dir" not in result.stdout
    assert "--overwrite" not in result.stdout


def test_rc_shell_attribute_split_dry_run_forwards_candidate_args() -> None:
    result = _run_rc_shell(
        "attribute-report",
        "--overwrite",
        "--output-dir",
        "out",
        "--candidate-jsonl",
        "candidate=artifacts/quality/candidate-report.jsonl",
        "--top-tokens",
        "4",
    )

    assert "benchmarks/analyze_glm45_air_teacher_cache_attribution.py" in result.stdout
    assert "--baseline-label balanced_rc_report" in result.stdout
    assert "--output-json out/report-attribution.json" in result.stdout
    assert "--candidate-jsonl candidate=artifacts/quality/candidate-report.jsonl" in result.stdout
    assert "--top-tokens 4" in result.stdout
    assert "--output-dir" not in result.stdout
    assert "--overwrite" not in result.stdout


def test_rc_shell_attribute_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "attr"
    output_dir.mkdir()
    (output_dir / "report-attribution.json").write_text("{}\n")

    result = subprocess.run(
        [
            "bash",
            "scripts/glm45_air_rc.sh",
            "attribute-report",
            "--output-dir",
            str(output_dir),
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "attribution output already exists" in result.stderr
    assert "pass --overwrite" in result.stderr


def test_rc_shell_memory_preflight_runs_quant_compare_without_model() -> None:
    result = _run_rc_shell("memory-preflight")

    assert "benchmarks/bench_glm45_air_quant_compare.py" in result.stdout
    assert "--memory-quiet-preflight" in result.stdout
    assert "--memory-quiet-window-seconds 3" in result.stdout
    assert "--memory-quiet-max-attempts 5" in result.stdout


def test_rc_shell_benchmark_candidate_cache_diagnostic_maps_to_execute_flag() -> None:
    result = _run_rc_shell("benchmark-candidate-cache1", "--overwrite")

    assert "--execute lane_s_candidate_cache1_diagnostic" in result.stdout
    assert "--overwrite" in result.stdout


def test_rc_shell_benchmark_candidate_cache_sweep_maps_to_execute_flags() -> None:
    result = _run_rc_shell("benchmark-candidate-cache-sweep", "--overwrite")

    assert "--execute lane_s_candidate_cache1_diagnostic" in result.stdout
    assert "--execute lane_s_candidate_cache2_diagnostic" in result.stdout
    assert "--execute lane_s_candidate_cache4_diagnostic" in result.stdout
    assert "--overwrite" in result.stdout


def test_rc_shell_public_rerun_commands_map_to_execute_flags() -> None:
    cases = {
        "audit": "audit",
        "eval-report": "eval_report",
        "eval-selection": "eval_selection",
        "eval-holdout": "eval_holdout",
        "benchmark-q2": "lane_s_q2_control",
    }

    for command_name, execute_name in cases.items():
        result = _run_rc_shell(command_name, "--overwrite", "--output-dir", "out")
        assert f"--execute {execute_name}" in result.stdout
        assert "--overwrite" in result.stdout
        assert "--output-dir out" in result.stdout


def test_rc_shell_public_training_commands_dry_run_default_recipes() -> None:
    collect = _run_rc_shell("collect-imatrix")
    assert "benchmarks/collect_glm45_air_imatrix.py" in collect.stdout
    assert "--output-dir artifacts/imatrix/glm45-air-public-calibration" in collect.stdout
    assert "--prompt-set air_vq_ladder_select_v1" in collect.stdout
    assert "--layers 31\\,36\\,41\\,45" in collect.stdout

    materialize = _run_rc_shell("materialize-sweep")
    assert "benchmarks/materialize_glm45_air_dynamic_imatrix_sweep.py" in materialize.stdout
    assert "--imatrix-manifest artifacts/imatrix/glm45-air-public-calibration/imatrix-manifest.json" in materialize.stdout
    assert "--budget 2.0 --budget 2.4" in materialize.stdout
    assert "--candidate-prefix public-dynamic-imatrix" in materialize.stdout

    train = _run_rc_shell("train-low-rank")
    assert "benchmarks/finetune_glm45_air_vq_continuous.py" in train.stdout
    assert "--trainable low_rank_residual" in train.stdout
    assert "--low-rank 4" in train.stdout
    assert "--train-row-indices 44\\,45\\,46\\,47\\,48\\,49\\,50\\,51" in train.stdout


def test_rc_shell_materialize_sweep_accepts_custom_imatrix_manifest() -> None:
    result = _run_rc_shell(
        "materialize-sweep",
        "--imatrix-manifest",
        "artifacts/imatrix/custom/imatrix-manifest.json",
        "--budget",
        "2.5",
    )

    assert "--imatrix-manifest artifacts/imatrix/custom/imatrix-manifest.json" in result.stdout
    assert result.stdout.count("--imatrix-manifest") == 1
    assert "--budget 2.0 --budget 2.4 --budget 2.5" in result.stdout


def test_rc_shell_materialize_sweep_missing_manifest_has_actionable_error() -> None:
    env = os.environ.copy()
    env.pop("GLM45_AIR_RC_DRY_RUN", None)
    result = subprocess.run(
        [
            "bash",
            "scripts/glm45_air_rc.sh",
            "materialize-sweep",
            "--imatrix-manifest",
            "artifacts/imatrix/does-not-exist/imatrix-manifest.json",
        ],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "required file is missing: artifacts/imatrix/does-not-exist/imatrix-manifest.json" in result.stderr
    assert "Run scripts/glm45_air_rc.sh collect-imatrix first" in result.stderr


def test_rc_shell_new_model_template_prints_checklist() -> None:
    result = _run_rc_shell("new-model-template", dry_run=False)

    assert "# New Model Family Template For KEEP" in result.stdout
    assert "## Adapter Contract" in result.stdout
    assert "## Required Tests" in result.stdout


def test_preflight_records_environment_report(tmp_path: Path) -> None:
    source_dir = _write_source_snapshot(tmp_path / "source")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "enabled": True,
                    "seed_artifact_dir": str(seed_dir),
                    "run": {"trainable": "low_rank_residual"},
                    "sidecars": [{"path": "continuous_params/layer-00045-gate_proj.safetensors"}],
                }
            }
        )
    )
    existing = tmp_path / "existing.jsonl"
    existing.write_text("{}\n")
    args = SimpleNamespace(
        artifact_dir=str(artifact_dir),
        seed_artifact_dir=str(seed_dir),
        model_id="test-model",
        revision="test",
        source_dir=str(source_dir),
        report_jsonl=str(existing),
        selection_jsonl=str(existing),
        holdout_jsonl=str(existing),
        lane_s_jsonl=str(existing),
        q2_control_jsonl=str(existing),
        report_teacher_jsonl=str(existing),
        selection_teacher_jsonl=str(existing),
        holdout_teacher_jsonl=str(existing),
        output_dir=str(tmp_path / "out"),
        overwrite=False,
    )

    preflight = rc._build_preflight(args)
    markdown = rc._render_preflight_markdown(preflight)

    assert preflight["summary"]["environment_checks_pass"] is True
    assert preflight["checks"]["environment"]["ok"] is True
    assert "## Environment" in markdown
    assert "MLX version" in markdown


def test_summary_cli_errors_include_environment_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeReport:
        def to_dict(self) -> dict:
            return {
                "ok": False,
                "mlx_version": "0.0.0",
                "mlx_lm_available": False,
                "metal_kernel_available": False,
                "hadamard_sizes_ok": {4096: False},
                "safetensors_integer_roundtrip": False,
                "failures": ["mlx.core version is too old"],
            }

    monkeypatch.setattr(rc, "verify_environment", lambda: FakeReport())
    source_dir = _write_source_snapshot(tmp_path / "source")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "enabled": True,
                    "seed_artifact_dir": str(seed_dir),
                    "run": {"trainable": "low_rank_residual"},
                    "sidecars": [{"path": "continuous_params/layer-00045-gate_proj.safetensors"}],
                }
            }
        )
    )
    existing = tmp_path / "existing.jsonl"
    existing.write_text("{}\n")
    args = SimpleNamespace(
        artifact_dir=str(artifact_dir),
        seed_artifact_dir=str(seed_dir),
        model_id="test-model",
        revision="test",
        source_dir=str(source_dir),
        report_jsonl=str(existing),
        selection_jsonl=str(existing),
        holdout_jsonl=str(existing),
        lane_s_jsonl=str(existing),
        q2_control_jsonl=str(existing),
        report_teacher_jsonl=str(existing),
        selection_teacher_jsonl=str(existing),
        holdout_teacher_jsonl=str(existing),
        output_dir=str(tmp_path / "out"),
        overwrite=True,
        write_focus_json=None,
        write_quality_plan_json=None,
        write_quality_frontier_json=None,
    )

    errors = rc._summary_cli_errors(args)

    assert any("environment check failed: mlx.core version is too old" in error for error in errors)


def test_rerun_commands_include_artifact_and_output_dir() -> None:
    commands = rc._build_rerun_commands(
        artifact_dir="artifact",
        output_dir="out",
        report_teacher_jsonl="report-meta",
        selection_teacher_jsonl="select-meta",
        holdout_teacher_jsonl="holdout-meta",
    )

    assert "--artifact-dir artifact" in commands["eval_report"]
    assert "--teacher-jsonl report-meta" in commands["eval_report"]
    assert "--append-jsonl out/selection128.jsonl" in commands["eval_selection"]
    assert "--warmup-repetitions 1" in commands["lane_s_q2_control"]
    assert "--repetitions 3" in commands["lane_s_q2_control"]
    assert "--timing-stability-max-relative-spread 0.2" in commands["lane_s_q2_control"]
    assert "--require-memory-quiet-preflight" in commands["lane_s_q2_control"]
    assert "--parent-vm-stat-diagnostics" in commands["lane_s_q2_control"]
    assert "--warmup-repetitions 1" in commands["lane_s_candidate"]
    assert "--repetitions 3" in commands["lane_s_candidate"]
    assert "--timing-stability-max-relative-spread 0.2" in commands["lane_s_candidate"]
    assert "--require-memory-quiet-preflight" in commands["lane_s_candidate"]
    assert "--parent-vm-stat-diagnostics" in commands["lane_s_candidate"]
    assert "--append-jsonl out/lane-s-candidate.jsonl" in commands["lane_s_candidate"]
    assert "--mlx-cache-limit-gb 1" in commands["lane_s_candidate_cache1_diagnostic"]
    assert "--require-memory-quiet-preflight" in commands["lane_s_candidate_cache1_diagnostic"]
    assert "--mlx-clear-cache-before-run" in commands["lane_s_candidate_cache1_diagnostic"]
    assert "--repetitions 1" in commands["lane_s_candidate_cache1_diagnostic"]
    assert "--append-jsonl out/lane-s-candidate-cache1-diagnostic.jsonl" in commands[
        "lane_s_candidate_cache1_diagnostic"
    ]
    assert "--mlx-cache-limit-gb 2" in commands["lane_s_candidate_cache2_diagnostic"]
    assert "--append-jsonl out/lane-s-candidate-cache2-diagnostic.jsonl" in commands[
        "lane_s_candidate_cache2_diagnostic"
    ]
    assert "--mlx-cache-limit-gb 4" in commands["lane_s_candidate_cache4_diagnostic"]
    assert "--append-jsonl out/lane-s-candidate-cache4-diagnostic.jsonl" in commands[
        "lane_s_candidate_cache4_diagnostic"
    ]


def test_execute_outputs_remap_summary_inputs_and_overwrite_jsonl(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    old_candidate = output_dir / "lane-s-candidate.jsonl"
    old_candidate.write_text('{"old": true}\n')
    old_cache = output_dir / "lane-s-candidate-cache1-diagnostic.jsonl"
    old_cache.write_text('{"old": true}\n')
    args = SimpleNamespace(
        output_dir=str(output_dir),
        overwrite=False,
        lane_s_jsonl="historical-lane-s.jsonl",
        q2_control_jsonl="historical-q2.jsonl",
        report_jsonl="historical-report.jsonl",
        selection_jsonl="historical-selection.jsonl",
        holdout_jsonl="historical-holdout.jsonl",
    )

    errors = rc._execute_output_errors(
        args,
        ["lane_s_candidate", "lane_s_candidate_cache1_diagnostic"],
    )

    assert any("lane-s-candidate.jsonl" in error for error in errors)
    assert any("lane-s-candidate-cache1-diagnostic.jsonl" in error for error in errors)

    args.overwrite = True
    rc._prepare_execute_outputs(
        args,
        ["lane_s_candidate", "lane_s_candidate_cache1_diagnostic"],
    )
    remapped = rc._args_with_execute_outputs(args, ["lane_s_candidate"])

    assert old_candidate.exists() is False
    assert old_cache.exists() is False
    assert remapped.lane_s_jsonl == str(output_dir / "lane-s-candidate.jsonl")
    assert remapped.q2_control_jsonl == "historical-q2.jsonl"


def test_source_model_check_requires_runtime_shards_only(tmp_path: Path) -> None:
    source_dir = _write_source_snapshot(tmp_path / "source")

    check = rc._source_model_check(
        model_id="test-model",
        revision="test",
        source_dir=str(source_dir),
    )

    assert check["ok"] is True
    assert check["num_hidden_layers"] == 2
    assert check["required_shards"] == 2
    assert check["present_shards"] == 2
    assert check["missing_shards"] == []
    assert check["ignored_optional_shards"] == ["model-00003-of-00003.safetensors"]


def test_source_model_check_fails_when_runtime_shard_is_missing(tmp_path: Path) -> None:
    source_dir = _write_source_snapshot(tmp_path / "source", missing_runtime=True)

    check = rc._source_model_check(
        model_id="test-model",
        revision="test",
        source_dir=str(source_dir),
    )

    assert check["ok"] is False
    assert check["required_shards"] == 2
    assert check["present_shards"] == 1
    assert any("missing source safetensors shards" in error for error in check["errors"])


def test_workflow_commands_document_train_and_materialize_paths() -> None:
    rerun_commands = {
        "audit": "audit-command",
        "eval_report": "eval-report-command",
        "eval_selection": "eval-selection-command",
        "eval_holdout": "eval-holdout-command",
        "lane_s_q2_control": "bench-q2-command",
        "lane_s_candidate": "bench-candidate-command",
    }

    commands = rc._build_workflow_commands(
        seed_artifact_dir="seed-artifact",
        output_dir="out",
        report_teacher_jsonl="report-cache/metadata.jsonl",
        selection_teacher_jsonl="selection-cache/metadata.jsonl",
        holdout_teacher_jsonl="holdout-cache/metadata.jsonl",
        rerun_commands=rerun_commands,
    )

    audit = commands["artifact_audit"]
    assert audit["status"] == "available_via_execute_not_rerun_in_this_slice"
    assert audit["command"] == "scripts/glm45_air_rc.sh audit --overwrite --output-dir out"

    environment = commands["environment_preflight"]
    assert environment["status"] == "verified_this_goal"
    assert environment["command"] == "scripts/glm45_air_rc.sh env-preflight"

    reproduce = commands["reproduce_rc_from_seed"]
    assert reproduce["status"] == "heavy_not_rerun_in_this_slice"
    assert reproduce["command"] == "scripts/glm45_air_rc.sh reproduce --overwrite --output-dir out"
    assert "protected seed" in reproduce["purpose"]

    verify = commands["verify_rc"]
    assert verify["status"] == "heavy_not_rerun_in_this_slice"
    assert verify["command"] == "scripts/glm45_air_rc.sh verify --overwrite --output-dir out"
    assert "fresh outputs" in verify["purpose"]

    eval_report = commands["eval_report"]
    assert eval_report["status"] == "heavy_not_rerun_in_this_slice"
    assert eval_report["command"] == "scripts/glm45_air_rc.sh eval-report --overwrite --output-dir out"

    eval_selection = commands["eval_selection"]
    assert eval_selection["command"] == "scripts/glm45_air_rc.sh eval-selection --overwrite --output-dir out"

    eval_holdout = commands["eval_holdout"]
    assert eval_holdout["command"] == "scripts/glm45_air_rc.sh eval-holdout --overwrite --output-dir out"

    benchmark_q2 = commands["benchmark_q2_control"]
    assert benchmark_q2["command"] == "scripts/glm45_air_rc.sh benchmark-q2 --overwrite --output-dir out"
    assert "diagnosis" in benchmark_q2["purpose"]

    benchmark_lane_s = commands["benchmark_lane_s"]
    assert benchmark_lane_s["command"] == "scripts/glm45_air_rc.sh benchmark-lane-s --overwrite --output-dir out"
    assert "both fresh files" in benchmark_lane_s["purpose"]

    benchmark_candidate = commands["benchmark_candidate"]
    assert benchmark_candidate["command"] == "scripts/glm45_air_rc.sh benchmark-candidate --overwrite --output-dir out"
    assert "diagnosis" in benchmark_candidate["purpose"]

    collect = commands["collect_calibration_imatrix"]
    assert collect["command"] == "scripts/glm45_air_rc.sh collect-imatrix"

    materialize = commands["materialize_dynamic_imatrix_sweep"]
    assert materialize["command"] == "scripts/glm45_air_rc.sh materialize-sweep"

    train = commands["train_low_rank_residual_sidecars"]
    assert train["status"] == "heavy_not_rerun_in_this_slice"
    assert train["command"] == "scripts/glm45_air_rc.sh train-low-rank"

    new_model = commands["new_model_family_template"]
    assert new_model["status"] == "verified_this_goal"
    assert new_model["command"] == "scripts/glm45_air_rc.sh new-model-template"

    preflight = commands["rc_preflight"]
    assert preflight["status"] == "verified_this_goal"
    assert preflight["command"] == "scripts/glm45_air_rc.sh preflight"

    memory_preflight = commands["host_memory_preflight"]
    assert memory_preflight["status"] == "verified_this_goal"
    assert memory_preflight["command"] == "scripts/glm45_air_rc.sh memory-preflight"

    inspect = commands["inspect_tail_metrics"]
    assert inspect["status"] == "verified_this_goal"
    assert inspect["command"] == "scripts/glm45_air_rc.sh focus --overwrite --output-dir out"
    assert "analyze_glm45_air_teacher_cache_attribution.py" not in inspect["command"]

    attribution = commands["inspect_token_attribution"]
    assert attribution["status"] == "verified_this_goal"
    assert attribution["command"] == "scripts/glm45_air_rc.sh attribute --overwrite --output-dir out"
    assert "prompt/token attribution" in attribution["purpose"]

    quality_frontier = commands["export_quality_frontier"]
    assert quality_frontier["status"] == "verified_this_goal"
    assert quality_frontier["command"] == "scripts/glm45_air_rc.sh frontier --overwrite --output-dir out"

    quality_plan = commands["plan_next_quality_slice"]
    assert quality_plan["status"] == "verified_this_goal"
    assert quality_plan["command"] == "scripts/glm45_air_rc.sh plan --overwrite --output-dir out"


def test_quality_plan_uses_report_selection_rows_and_keeps_holdout_for_validation() -> None:
    report = {
        "schema_version": 1,
        "artifact_dir": "accepted-artifact",
        "seed_artifact_dir": "seed-artifact",
        "overall_status": "balanced_rc_pass_community_wow_miss",
        "eval_splits": {
            "report": {
                "quality_focus": {
                    "highest_token_kld_rows": [
                        {
                            "prompt_id": "report_route_007",
                            "domain": "route",
                            "row_index": 127,
                            "top1_agreement": 0.67,
                            "mean_kld": 0.23,
                            "max_token_kld": 4.6,
                        }
                    ],
                    "lowest_top1_rows": [
                        {
                            "prompt_id": "report_math_015",
                            "domain": "math",
                            "row_index": 59,
                            "top1_agreement": 0.50,
                            "mean_kld": 0.55,
                            "max_token_kld": 4.4,
                        }
                    ],
                }
            },
            "selection": {
                "quality_focus": {
                    "highest_token_kld_rows": [
                        {
                            "prompt_id": "select_route_007",
                            "domain": "route",
                            "row_index": 127,
                            "top1_agreement": 0.73,
                            "mean_kld": 0.25,
                            "max_token_kld": 4.6,
                        }
                    ],
                    "lowest_top1_rows": [
                        {
                            "prompt_id": "select_instruction_002",
                            "domain": "instruction",
                            "row_index": 74,
                            "top1_agreement": 0.66,
                            "mean_kld": 0.39,
                            "max_token_kld": 4.1,
                        }
                    ],
                }
            },
            "holdout": {
                "quality_focus": {
                    "lowest_top1_rows": [
                        {
                            "prompt_id": "holdout_math_024",
                            "domain": "math",
                            "row_index": 68,
                            "top1_agreement": 0.55,
                            "mean_kld": 0.39,
                            "max_token_kld": 4.4,
                        }
                    ]
                }
            },
        },
    }
    args = type(
        "Args",
        (),
        {
            "output_dir": "out",
            "report_teacher_jsonl": "report-cache/metadata.jsonl",
            "selection_teacher_jsonl": "selection-cache/metadata.jsonl",
            "holdout_teacher_jsonl": "holdout-cache/metadata.jsonl",
        },
    )()

    plan = rc._build_quality_plan(report, args)

    assert plan["policy"]["training_splits"] == ["report", "selection"]
    assert plan["policy"]["holdout_usage"] == "validation_focus_only_not_training"
    assert 68 not in plan["train_row_indices"]
    assert any(row["row_index"] == 68 for row in plan["holdout_validation_focus"])
    command = plan["commands"]["train_quality_sidecar"]["command"]
    assert "--seed-artifact-dir accepted-artifact" in command
    assert "--trainable low_rank_residual" in command
    assert "--low-rank 4" in command
    assert "--loss-scope final_layer_selected" in command
    assert "--surrogate-projections gate_proj up_proj down_proj" in command


def test_preflight_reports_missing_required_paths(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "enabled": True,
                    "seed_artifact_dir": str(tmp_path / "seed"),
                    "run": {"trainable": "low_rank_residual"},
                    "sidecars": [{"path": "continuous_params/layer-00045-gate_proj.safetensors"}],
                }
            }
        )
    )
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    report_jsonl = tmp_path / "report.jsonl"
    report_jsonl.write_text("{}\n")
    source_dir = _write_source_snapshot(tmp_path / "source")
    args = type(
        "Args",
        (),
        {
            "artifact_dir": str(artifact_dir),
            "seed_artifact_dir": str(seed_dir),
            "model_id": "test-model",
            "revision": "test",
            "source_dir": str(source_dir),
            "report_jsonl": str(report_jsonl),
            "selection_jsonl": str(tmp_path / "missing-selection.jsonl"),
            "holdout_jsonl": str(tmp_path / "missing-holdout.jsonl"),
            "lane_s_jsonl": str(tmp_path / "missing-lane-s.jsonl"),
            "q2_control_jsonl": str(tmp_path / "missing-q2.jsonl"),
            "report_teacher_jsonl": str(tmp_path / "report-cache" / "metadata.jsonl"),
            "selection_teacher_jsonl": str(tmp_path / "selection-cache" / "metadata.jsonl"),
            "holdout_teacher_jsonl": str(tmp_path / "holdout-cache" / "metadata.jsonl"),
            "output_dir": str(tmp_path / "out"),
            "overwrite": False,
            "write_focus_json": None,
            "write_quality_plan_json": None,
            "write_quality_frontier_json": None,
        },
    )()

    preflight = rc._build_preflight(args)

    assert preflight["status"] == "fail"
    assert preflight["summary"]["manifest_checks_pass"] is True
    assert preflight["summary"]["public_doc_checks_pass"] is True
    assert preflight["summary"]["source_model_checks_pass"] is True
    assert all(item["ok"] for item in preflight["checks"]["public_docs"])
    failed_paths = {
        item["name"]
        for item in preflight["checks"]["paths"]
        if item["ok"] is False
    }
    assert "selection_jsonl" in failed_paths
    assert "holdout_teacher_jsonl" in failed_paths


def test_summary_cli_errors_are_actionable(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "enabled": True,
                    "seed_artifact_dir": str(seed_dir),
                    "run": {"trainable": "low_rank_residual"},
                    "sidecars": [{"path": "continuous_params/layer-00045-gate_proj.safetensors"}],
                }
            }
        )
    )
    existing = tmp_path / "existing.jsonl"
    existing.write_text("{}\n")
    output_dir = tmp_path / "rc-output"
    output_dir.mkdir()
    (output_dir / "rc-summary.json").write_text("{}\n")
    (output_dir / "RC_SUMMARY.md").write_text("# old\n")
    focus_json = output_dir / "quality-focus.json"
    focus_json.write_text("{}\n")
    quality_plan_json = output_dir / "quality-plan.json"
    quality_plan_json.write_text("{}\n")
    quality_frontier_json = output_dir / "quality-frontier.json"
    quality_frontier_json.write_text("{}\n")
    source_dir = _write_source_snapshot(tmp_path / "source")
    args = type(
        "Args",
        (),
        {
            "artifact_dir": str(artifact_dir),
            "seed_artifact_dir": str(seed_dir),
            "model_id": "test-model",
            "revision": "test",
            "source_dir": str(source_dir),
            "report_jsonl": str(existing),
            "selection_jsonl": str(tmp_path / "missing-selection.jsonl"),
            "holdout_jsonl": str(existing),
            "lane_s_jsonl": str(existing),
            "q2_control_jsonl": str(existing),
            "report_teacher_jsonl": str(existing),
            "selection_teacher_jsonl": str(existing),
            "holdout_teacher_jsonl": str(existing),
            "output_dir": str(output_dir),
            "overwrite": False,
            "write_focus_json": str(focus_json),
            "write_quality_plan_json": str(quality_plan_json),
            "write_quality_frontier_json": str(quality_frontier_json),
        },
    )()

    errors = rc._summary_cli_errors(args)

    assert any("missing required file for selection_jsonl" in error for error in errors)
    assert any("summary outputs already exist" in error for error in errors)
    assert any("focus JSON already exists" in error for error in errors)
    assert any("quality plan JSON already exists" in error for error in errors)
    assert any("quality frontier JSON already exists" in error for error in errors)
