"""Named gate profiles for recipe steps.

Profiles resolve to the extracted RC gate constants in
``keep.quality.rc_gates`` so a recipe references thresholds by name and a
threshold change is a recipe-visible, re-gateable event. Evaluation of each
profile against step evidence lives in the runner; this module owns the
names and resolved thresholds only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from keep.quality.rc_gates import (
    BALANCED_HARD_TARGETS,
    COMMUNITY_WOW_TARGETS,
    LANE_S_TIMING_RELATIVE_SPREAD_MAX,
)

GATE_PROFILES: dict[str, dict[str, Any]] = {
    # Composite GLM-5.2 release evidence. Thresholds are frozen inside the
    # authenticated family-policy/gate payload; this profile validates verdict
    # structure and makes a valid blocked result a first-class build state.
    "glm52_family": {},
    # Cheap structural check after training: loss finite and declared
    # sidecars present in the output manifest.
    "train_sane": {},
    # Artifact audit payload reports prefill compatibility and no dense
    # routed / unbound VQ experts.
    "audit_ok": {},
    # Per-split balanced hard gate: 128 clean rows, PPL ratio bound.
    "balanced_rc_split": {
        "allow_dirty_rows": False,
        "clean_rows": BALANCED_HARD_TARGETS["clean_rows"],
        "mean_ppl_ratio_max": BALANCED_HARD_TARGETS["mean_ppl_ratio_max"],
    },
    # Community-wow targets per split (includes the balanced-split checks).
    "community_wow": {
        "allow_dirty_rows": False,
        "clean_rows": BALANCED_HARD_TARGETS["clean_rows"],
        "mean_ppl_ratio_max": BALANCED_HARD_TARGETS["mean_ppl_ratio_max"],
        "mean_kld_max": COMMUNITY_WOW_TARGETS["mean_kld_max"],
        "p999_kld_max": COMMUNITY_WOW_TARGETS["p999_kld_max"],
        "top1_min": COMMUNITY_WOW_TARGETS["top1_min"],
        "domain_top1_min": COMMUNITY_WOW_TARGETS["domain_top1_min"],
    },
    # Standalone benchmark cleanliness (used by the q2 control step).
    "bench_clean": {
        "allow_dirty_rows": False,
        "timing_relative_spread_max": LANE_S_TIMING_RELATIVE_SPREAD_MAX,
    },
    # Paired Lane S gate: candidate vs q2 control.
    "lane_s": {
        "allow_dirty_rows": False,
        "lane_s_ratio_max": BALANCED_HARD_TARGETS["lane_s_ratio_max"],
        "effective_bpw_max": BALANCED_HARD_TARGETS["effective_bpw_max"],
        "timing_relative_spread_max": LANE_S_TIMING_RELATIVE_SPREAD_MAX,
    },
}


def resolve_gate_thresholds(
    profile: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if profile not in GATE_PROFILES:
        raise KeyError(f"unknown gate profile: {profile}")
    thresholds = dict(GATE_PROFILES[profile])
    for key, value in (overrides or {}).items():
        if key not in thresholds:
            raise KeyError(f"gate profile {profile!r} has no threshold {key!r}")
        thresholds[key] = value
    return thresholds
