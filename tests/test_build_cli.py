from __future__ import annotations

import json
from pathlib import Path

import pytest

from keep.build.cli import (
    _resolve_repair_from_step,
    _run_optional_rc_diff,
    build_parser,
)
from keep.build.recipe import load_recipe, parse_recipe


def _accepted_summary() -> dict:
    return {
        "overall_status": "balanced_rc_pass_community_wow_miss",
        "checks": {
            "split_hard": {
                split: {"clean_128": True, "mean_ppl_ratio_le_1p05": True}
                for split in ("report", "selection", "holdout")
            },
            "split_wow": {
                split: {
                    "mean_kld_le_0p30": False,
                    "p999_kld_le_3p0": False,
                    "top1_ge_0p85": False,
                    "all_domain_top1_ge_0p80": False,
                }
                for split in ("report", "selection", "holdout")
            },
            "lane_s": {
                "candidate_clean": True,
                "candidate_timing_stable": True,
                "effective_bpw_le_2p1": True,
                "no_dense_routed_experts": True,
                "no_unbound_vq_experts": True,
                "non_expert_dtype_verified": True,
                "q2_control_clean": True,
                "q2_control_timing_stable": True,
                "ratio_le_1p15": True,
            },
        },
    }


def _promotion_result() -> dict:
    split_checks = {
        "clean_rows": True,
        "mean_ppl_ratio": True,
        "mean_kld": False,
        "p999_kld": False,
        "top1": False,
        "domain_top1": False,
    }
    return {
        "status": "balanced_rc_pass_community_wow_miss",
        "gates": {
            "eval_report": {"profile": "community_wow", "checks": split_checks},
            "eval_selection": {"profile": "community_wow", "checks": split_checks},
            "eval_holdout": {"profile": "community_wow", "checks": split_checks},
            "lane_s_candidate": {
                "profile": "lane_s",
                "checks": {
                    "candidate_clean": True,
                    "candidate_timing_stable": True,
                    "effective_bpw": True,
                    "no_dense_routed_experts": True,
                    "no_unbound_vq_experts": True,
                    "non_expert_dtype_verified": True,
                    "q2_control_clean": True,
                    "q2_control_timing_stable": True,
                    "ratio": True,
                },
            },
        },
    }


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def test_build_parser_exposes_accepted_rc_summary_diff_options() -> None:
    args = build_parser().parse_args(
        [
            "build",
            "recipe.yaml",
            "--accepted-rc-summary",
            "accepted.json",
            "--rc-diff-out",
            "diff.json",
        ]
    )

    assert args.accepted_rc_summary == "accepted.json"
    assert args.rc_diff_out == "diff.json"


def test_build_parser_exposes_repair_from_split_option() -> None:
    args = build_parser().parse_args(
        ["build", "recipe.yaml", "--repair-from", "eval_report"]
    )

    assert args.repair_from == "eval_report"


def test_resolve_repair_from_step_maps_target_raw_split_to_repair_step() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )

    assert _resolve_repair_from_step(recipe, "eval_report") == "eval_report_repair"
    assert _resolve_repair_from_step(recipe, "eval_selection") == "eval_selection_repair"
    assert _resolve_repair_from_step(recipe, "eval_holdout") == "eval_holdout_repair"
    assert _resolve_repair_from_step(recipe, "eval_report_repair") == "eval_report_repair"


def test_resolve_repair_from_step_rejects_split_without_repair() -> None:
    recipe = parse_recipe(
        {
            "schema_version": 1,
            "name": "fixture",
            "external_inputs": {},
            "steps": [
                {
                    "id": "eval_report",
                    "op": "eval",
                    "class": "verify",
                    "params": {"engine": "vq_e1_routed_nax_e8p"},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="no eval-repair step"):
        _resolve_repair_from_step(recipe, "eval_report")


def test_run_optional_rc_diff_writes_default_build_diff(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    _write_json(build_root / "promotion-result.json", _promotion_result())
    accepted = _write_json(tmp_path / "accepted.json", _accepted_summary())

    rc = _run_optional_rc_diff(
        build_root=build_root,
        accepted_rc_summary=accepted,
        rc_diff_out=None,
    )

    assert rc == 0
    diff = json.loads((build_root / "rc-diff.json").read_text())
    assert diff["passed"] is True
    assert diff["mismatches"] == []


def test_run_optional_rc_diff_reports_mismatch(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    promotion = _promotion_result()
    promotion["gates"]["eval_report"]["checks"]["clean_rows"] = False
    _write_json(build_root / "promotion-result.json", promotion)
    accepted = _write_json(tmp_path / "accepted.json", _accepted_summary())

    rc = _run_optional_rc_diff(
        build_root=build_root,
        accepted_rc_summary=accepted,
        rc_diff_out=tmp_path / "custom-diff.json",
    )

    assert rc == 1
    diff = json.loads((tmp_path / "custom-diff.json").read_text())
    assert {
        "path": "checks.split_hard.report.clean_128",
        "accepted": True,
        "promotion": False,
    } in diff["mismatches"]
