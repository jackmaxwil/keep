"""Compare recipe promotion gate results with an accepted RC summary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_SPLIT_STEP_NAMES = {
    "eval_report": "report",
    "report": "report",
    "eval_selection": "selection",
    "selection": "selection",
    "eval_select": "selection",
    "select": "selection",
    "eval_holdout": "holdout",
    "holdout": "holdout",
}


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _split_name(step_id: str) -> str | None:
    if step_id in _SPLIT_STEP_NAMES:
        return _SPLIT_STEP_NAMES[step_id]
    for token, split in _SPLIT_STEP_NAMES.items():
        if step_id.endswith("_" + token) or token in step_id.split("_"):
            return split
    return None


def _normalize_promotion_checks(promotion: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "split_hard": {},
        "split_wow": {},
        "lane_s": {},
    }
    gates = promotion.get("gates") or {}
    if not isinstance(gates, Mapping):
        return checks

    for step_id, gate in gates.items():
        if not isinstance(step_id, str) or not isinstance(gate, Mapping):
            continue
        raw_checks = gate.get("checks") or {}
        if not isinstance(raw_checks, Mapping):
            continue
        profile = gate.get("profile")
        split = _split_name(step_id)
        if profile in {"balanced_rc_split", "community_wow"} and split is not None:
            checks["split_hard"][split] = {
                "clean_128": _bool_or_none(raw_checks.get("clean_rows")),
                "mean_ppl_ratio_le_1p05": _bool_or_none(
                    raw_checks.get("mean_ppl_ratio")
                ),
            }
            if profile == "community_wow":
                checks["split_wow"][split] = {
                    "mean_kld_le_0p30": _bool_or_none(raw_checks.get("mean_kld")),
                    "p999_kld_le_3p0": _bool_or_none(raw_checks.get("p999_kld")),
                    "top1_ge_0p85": _bool_or_none(raw_checks.get("top1")),
                    "all_domain_top1_ge_0p80": _bool_or_none(
                        raw_checks.get("domain_top1")
                    ),
                }
            continue
        if profile == "lane_s" or "lane_s" in step_id:
            checks["lane_s"] = {
                "candidate_clean": _bool_or_none(raw_checks.get("candidate_clean")),
                "candidate_timing_stable": _bool_or_none(
                    raw_checks.get("candidate_timing_stable")
                ),
                "effective_bpw_le_2p1": _bool_or_none(raw_checks.get("effective_bpw")),
                "no_dense_routed_experts": _bool_or_none(
                    raw_checks.get("no_dense_routed_experts")
                ),
                "no_unbound_vq_experts": _bool_or_none(
                    raw_checks.get("no_unbound_vq_experts")
                ),
                "non_expert_dtype_verified": _bool_or_none(
                    raw_checks.get("non_expert_dtype_verified")
                ),
                "q2_control_clean": _bool_or_none(raw_checks.get("q2_control_clean")),
                "q2_control_timing_stable": _bool_or_none(
                    raw_checks.get("q2_control_timing_stable")
                ),
                "ratio_le_1p15": _bool_or_none(raw_checks.get("ratio")),
            }
    return checks


def _iter_gate_paths(accepted_checks: Mapping[str, Any]) -> list[tuple[str, Any]]:
    paths: list[tuple[str, Any]] = []
    for group in ("split_hard", "split_wow"):
        splits = accepted_checks.get(group) or {}
        if not isinstance(splits, Mapping):
            continue
        for split, checks in splits.items():
            if not isinstance(split, str) or not isinstance(checks, Mapping):
                continue
            for name, value in checks.items():
                paths.append((f"{group}.{split}.{name}", value))
    lane_s = accepted_checks.get("lane_s") or {}
    if isinstance(lane_s, Mapping):
        for name, value in lane_s.items():
            paths.append((f"lane_s.{name}", value))
    return paths


def _get_path(root: Mapping[str, Any], path: str) -> Any:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def compare_promotion_to_rc(
    promotion: Mapping[str, Any],
    accepted_rc_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable gate-check diff.

    The recipe promotion result stores gate checks by recipe step id, while the
    RC summary stores them by public split/check names. This comparator maps the
    recipe result into the RC summary shape and compares only gate-level split
    and Lane S checks. Artifact audit checks are intentionally outside the
    promotion gate result and are not claimed here.
    """

    accepted_checks = accepted_rc_summary.get("checks") or {}
    if not isinstance(accepted_checks, Mapping):
        accepted_checks = {}
    promotion_checks = _normalize_promotion_checks(promotion)
    accepted_status = accepted_rc_summary.get("overall_status")
    promotion_status = promotion.get("status")
    mismatches: list[dict[str, Any]] = []

    if accepted_status != promotion_status:
        mismatches.append(
            {
                "path": "overall_status",
                "accepted": accepted_status,
                "promotion": promotion_status,
            }
        )

    for path, accepted_value in _iter_gate_paths(accepted_checks):
        promotion_value = _get_path(promotion_checks, path)
        if promotion_value != accepted_value:
            mismatches.append(
                {
                    "path": "checks." + path,
                    "accepted": accepted_value,
                    "promotion": promotion_value,
                }
            )

    status_match = accepted_status == promotion_status
    checks_match = not [item for item in mismatches if item["path"] != "overall_status"]
    return {
        "schema_version": 1,
        "passed": status_match and checks_match,
        "status_match": status_match,
        "checks_match": checks_match,
        "accepted_status": accepted_status,
        "promotion_status": promotion_status,
        "mismatches": mismatches,
        "promotion_checks": promotion_checks,
    }


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value
