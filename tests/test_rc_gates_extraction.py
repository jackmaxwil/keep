from __future__ import annotations

import importlib.util
from pathlib import Path

from keep.quality import rc_gates


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_glm45_air_rc_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("run_glm45_air_rc_pipeline_extraction", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
rc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rc)


_EXTRACTED_NAMES = (
    "BALANCED_HARD_TARGETS",
    "COMMUNITY_WOW_TARGETS",
    "LANE_S_TIMING_RELATIVE_SPREAD_MAX",
    "PARENT_VM_DELTA_KEYS",
    "QUALITY_PLAN_CATEGORY_WEIGHTS",
    "QUALITY_PLAN_DOMAIN_QUOTAS",
    "QUALITY_PLAN_DOMAIN_WEIGHTS",
    "SPLIT_PREFIXES",
    "_all",
    "_build_quality_focus",
    "_check_ge",
    "_check_le",
    "_domain_from_record",
    "_float_or_none",
    "_focus_row",
    "_is_clean_benchmark_row",
    "_lane_s_failure_reasons",
    "_load_manifest",
    "_quality_checks",
    "_read_json_object",
    "_read_jsonl",
    "_selected_parent_vm_deltas",
    "_sum_delta_maps",
    "_summarize_benchmark",
    "_summarize_benchmark_memory",
    "_summarize_eval_domains",
    "_summarize_eval_split",
    "_summarize_parent_vm_stats",
    "_take_ranked",
    "_timing_summary",
    "_top_token_kld",
)


def test_rc_pipeline_rebinds_extracted_gate_objects() -> None:
    for name in _EXTRACTED_NAMES:
        assert getattr(rc, name) is getattr(rc_gates, name), name


def test_gate_constants_hold_documented_thresholds() -> None:
    assert rc_gates.BALANCED_HARD_TARGETS == {
        "clean_rows": 128,
        "effective_bpw_max": 2.1,
        "lane_s_ratio_max": 1.15,
        "mean_ppl_ratio_max": 1.05,
    }
    assert rc_gates.COMMUNITY_WOW_TARGETS == {
        "mean_kld_max": 0.30,
        "p999_kld_max": 3.0,
        "top1_min": 0.85,
        "domain_top1_min": 0.80,
    }
    assert rc_gates.LANE_S_TIMING_RELATIVE_SPREAD_MAX == 0.2


def test_quality_checks_boundary_values() -> None:
    passing = {
        "clean_record_count": 128,
        "all_memory_clean": True,
        "mean_ppl_ratio": 1.05,
        "mean_kld": 0.30,
        "p999_kld": 3.0,
        "mean_top1_agreement": 0.85,
    }
    assert rc_gates._all(rc_gates._quality_checks(passing))

    dirty = dict(passing, clean_record_count=127)
    assert not rc_gates._quality_checks(dirty)["clean_128"]

    missed_top1 = dict(passing, mean_top1_agreement=0.8499)
    assert not rc_gates._quality_checks(missed_top1)["top1_ge_0p85"]


def test_keep_alias_exposes_rc_gates() -> None:
    from keep.quality import rc_gates as keep_rc_gates

    assert keep_rc_gates is rc_gates
