from __future__ import annotations

from keep.build.rc_diff import compare_promotion_to_rc


def _accepted_summary() -> dict:
    return {
        "overall_status": "balanced_rc_pass_community_wow_miss",
        "checks": {
            "split_hard": {
                "report": {
                    "clean_128": True,
                    "mean_ppl_ratio_le_1p05": True,
                },
                "selection": {
                    "clean_128": True,
                    "mean_ppl_ratio_le_1p05": True,
                },
                "holdout": {
                    "clean_128": True,
                    "mean_ppl_ratio_le_1p05": True,
                },
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


def _promotion_result(*, report_clean: bool = True) -> dict:
    eval_checks = {
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
            "eval_report": {
                "profile": "community_wow",
                "checks": {**eval_checks, "clean_rows": report_clean},
            },
            "eval_selection": {
                "profile": "community_wow",
                "checks": eval_checks,
            },
            "eval_holdout": {
                "profile": "community_wow",
                "checks": eval_checks,
            },
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


def _promotion_result_with_repaired_split_steps() -> dict:
    promotion = _promotion_result()
    gates = promotion["gates"]
    gates["eval_report_repair"] = gates.pop("eval_report")
    gates["eval_selection_repair"] = gates.pop("eval_selection")
    gates["eval_holdout_repair"] = gates.pop("eval_holdout")
    return promotion


def test_compare_promotion_to_rc_passes_when_gate_checks_match() -> None:
    diff = compare_promotion_to_rc(_promotion_result(), _accepted_summary())

    assert diff["passed"] is True
    assert diff["status_match"] is True
    assert diff["mismatches"] == []
    assert diff["promotion_checks"]["split_hard"]["report"]["clean_128"] is True


def test_compare_promotion_to_rc_reports_check_mismatches() -> None:
    diff = compare_promotion_to_rc(
        _promotion_result(report_clean=False),
        _accepted_summary(),
    )

    assert diff["passed"] is False
    assert {
        "path": "checks.split_hard.report.clean_128",
        "accepted": True,
        "promotion": False,
    } in diff["mismatches"]


def test_compare_promotion_to_rc_maps_repaired_split_step_names() -> None:
    diff = compare_promotion_to_rc(
        _promotion_result_with_repaired_split_steps(),
        _accepted_summary(),
    )

    assert diff["passed"] is True
    assert diff["promotion_checks"]["split_hard"]["report"]["clean_128"] is True
    assert diff["promotion_checks"]["split_hard"]["selection"]["clean_128"] is True
    assert diff["promotion_checks"]["split_hard"]["holdout"]["clean_128"] is True
