"""RC hard-gate and community-wow gate math for the GLM-4.5-Air pipeline.

Extracted verbatim from ``benchmarks/run_glm45_air_rc_pipeline.py`` so the
gate constants and pass/fail math have one importable home (also aliased as
``keep.quality.rc_gates``). This module is distinct from
``keep.quality.gates``, which implements the older teacher-cache ladder
verdicts (ACCEPT/RECOVER/FALL BACK/ESCALATE) with different thresholds; the
two must not be merged.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ramp.benchmark.quant_compare import audit_vq_artifact_prefill_compatibility
from keep.quality.teacher_cache import summarize_teacher_cache_records


BALANCED_HARD_TARGETS = {
    "clean_rows": 128,
    "effective_bpw_max": 2.1,
    "lane_s_ratio_max": 1.15,
    "mean_ppl_ratio_max": 1.05,
}
LANE_S_TIMING_RELATIVE_SPREAD_MAX = 0.2
COMMUNITY_WOW_TARGETS = {
    "mean_kld_max": 0.30,
    "p999_kld_max": 3.0,
    "top1_min": 0.85,
    "domain_top1_min": 0.80,
}
QUALITY_PLAN_DOMAIN_QUOTAS = {
    "route": 4,
    "math": 3,
    "instruction": 1,
}
QUALITY_PLAN_CATEGORY_WEIGHTS = {
    "lowest_top1_rows": 6.0,
    "highest_token_kld_rows": 4.0,
    "highest_mean_kld_rows": 2.0,
}
QUALITY_PLAN_DOMAIN_WEIGHTS = {
    "route": 3.0,
    "math": 2.0,
    "instruction": 1.0,
}
SPLIT_PREFIXES = {"report", "select", "selection", "holdout"}
PARENT_VM_DELTA_KEYS = (
    "pageouts",
    "swapouts",
    "pages_free",
    "pages_active",
    "pages_inactive",
    "pages_occupied_by_compressor",
    "pages_stored_in_compressor",
)


def _engine_requires_nax_e8(engine: str) -> bool:
    return "nax_e8" in engine and "nax_e8p" not in engine


def assert_engine_tier_compatible(artifact_dir: str | Path, engine: str) -> None:
    """Fail before eval if a pinned E8 engine is pointed at E8P groups."""

    if not _engine_requires_nax_e8(engine):
        return
    audit = audit_vq_artifact_prefill_compatibility(artifact_dir)
    if audit.get("auto_prefill_all_layers_nax_e8_compatible") is True:
        return

    offenders: list[str] = []
    for row in audit.get("projection_rows") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("storage") == "vq" and row.get("code_bits") == 8:
            continue
        layer = row.get("layer", "?")
        projection = row.get("projection", "?")
        storage = row.get("storage", "unknown")
        code_bits = row.get("code_bits")
        if code_bits is None:
            detail = f"storage={storage}"
        else:
            detail = f"code_bits={code_bits}"
        offenders.append(f"layer {layer} {projection} ({detail})")

    if not offenders:
        for row in audit.get("layer_rows") or []:
            if not isinstance(row, Mapping):
                continue
            if row.get("nax_e8_compatible") is True:
                continue
            blockers = row.get("nax_e8_blockers") or row.get("blockers") or []
            offenders.append(f"layer {row.get('layer', '?')} ({', '.join(map(str, blockers))})")

    detail = "; ".join(offenders[:12]) if offenders else "no E8-compatible routed projections found"
    if len(offenders) > 12:
        detail += f"; ... {len(offenders) - 12} more"
    raise ValueError(
        f"engine {engine!r} requires code_bits=8 for every routed projection, "
        f"but {artifact_dir} is not NAX E8 compatible: {detail}"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _load_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "conversion-manifest.json"
    if not manifest_path.exists():
        return {"manifest_path": str(manifest_path), "exists": False}
    manifest = json.loads(manifest_path.read_text())
    continuous = manifest.get("continuous_parameters") or {}
    if not isinstance(continuous, Mapping):
        continuous = {}
    run = continuous.get("run") or {}
    sidecars = continuous.get("sidecars") or []
    return {
        "manifest_path": str(manifest_path),
        "exists": True,
        "seed_artifact_dir": continuous.get("seed_artifact_dir"),
        "continuous_enabled": bool(continuous.get("enabled")),
        "continuous_format": continuous.get("format"),
        "sidecar_count": len(sidecars) if isinstance(sidecars, list) else 0,
        "trainable": run.get("trainable") if isinstance(run, Mapping) else None,
        "low_rank": run.get("low_rank") if isinstance(run, Mapping) else None,
        "layer": run.get("layer") if isinstance(run, Mapping) else None,
        "projections": run.get("projections") if isinstance(run, Mapping) else None,
        "train_cache": run.get("train_cache") if isinstance(run, Mapping) else None,
        "steps": run.get("steps") if isinstance(run, Mapping) else None,
        "learning_rate": run.get("learning_rate") if isinstance(run, Mapping) else None,
    }


def _summarize_eval_split(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    summary = summarize_teacher_cache_records(records)
    summary["domains"] = _summarize_eval_domains(records)
    summary["quality_focus"] = _build_quality_focus(records)
    summary.update({"path": str(path)})
    return summary


def _domain_from_record(record: Mapping[str, Any]) -> str:
    prompt_id = record.get("prompt_id")
    if isinstance(prompt_id, str) and prompt_id:
        parts = [part for part in prompt_id.split("_") if part]
        if len(parts) >= 3 and parts[0] in SPLIT_PREFIXES:
            return parts[1]
        if len(parts) >= 2:
            return parts[0]
    row_intent = record.get("row_intent")
    if isinstance(row_intent, str) and row_intent:
        return row_intent.split("_", 1)[0]
    return "unknown"


def _summarize_eval_domains(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_domain.setdefault(_domain_from_record(record), []).append(dict(record))
    domains: dict[str, dict[str, Any]] = {}
    for domain_name in sorted(by_domain):
        summary = summarize_teacher_cache_records(by_domain[domain_name])
        summary["top1_ge_0p80"] = _check_ge(
            summary.get("mean_top1_agreement"),
            COMMUNITY_WOW_TARGETS["domain_top1_min"],
        )
        domains[domain_name] = summary
    return domains


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _top_token_kld(record: Mapping[str, Any]) -> dict[str, Any]:
    token_klds = record.get("token_klds") or []
    if not isinstance(token_klds, Sequence) or isinstance(token_klds, (str, bytes)):
        token_klds = []
    best_index: int | None = None
    best_value: float | None = None
    for index, value in enumerate(token_klds):
        numeric = _float_or_none(value)
        if numeric is None:
            continue
        if best_value is None or numeric > best_value:
            best_index = index
            best_value = numeric
    positions = record.get("positions") if isinstance(record.get("positions"), list) else []
    target_ids = record.get("target_token_ids") if isinstance(record.get("target_token_ids"), list) else []
    return {
        "max_token_kld": best_value,
        "max_token_kld_index": best_index,
        "max_token_position": positions[best_index] if best_index is not None and best_index < len(positions) else None,
        "max_token_target_id": target_ids[best_index] if best_index is not None and best_index < len(target_ids) else None,
    }


def _focus_row(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "prompt_id": record.get("prompt_id"),
        "domain": _domain_from_record(record),
        "row_index": record.get("row_index"),
        "top1_agreement": _float_or_none(record.get("top1_agreement")),
        "mean_kld": _float_or_none(record.get("mean_kld")),
        "ppl_ratio": _float_or_none(record.get("ppl_ratio")),
        "pageouts_delta": record.get("pageouts_delta"),
        "swapouts_delta": record.get("swapouts_delta"),
    }
    row.update(_top_token_kld(record))
    return row


def _take_ranked(
    records: Sequence[Mapping[str, Any]],
    *,
    key: str,
    reverse: bool,
    limit: int,
) -> list[dict[str, Any]]:
    rows = [_focus_row(record) for record in records]
    rows = [row for row in rows if row.get(key) is not None]
    rows.sort(
        key=lambda row: (
            float(row[key]),
            str(row.get("prompt_id") or ""),
        ),
        reverse=reverse,
    )
    return rows[:limit]


def _build_quality_focus(records: Sequence[Mapping[str, Any]], *, limit: int = 8) -> dict[str, Any]:
    return {
        "lowest_top1_rows": _take_ranked(records, key="top1_agreement", reverse=False, limit=limit),
        "highest_mean_kld_rows": _take_ranked(records, key="mean_kld", reverse=True, limit=limit),
        "highest_token_kld_rows": _take_ranked(records, key="max_token_kld", reverse=True, limit=limit),
    }


def _is_clean_benchmark_row(row: Mapping[str, Any]) -> bool:
    return (
        int(row.get("pageouts_delta") or 0) == 0
        and int(row.get("swapouts_delta") or 0) == 0
        and not bool(row.get("invalid_memory_pressure"))
    )


def _selected_parent_vm_deltas(record: Mapping[str, Any]) -> dict[str, int]:
    parent = record.get("parent_vm_stat")
    if not isinstance(parent, Mapping) or not parent.get("available"):
        return {}
    deltas = parent.get("deltas")
    if not isinstance(deltas, Mapping):
        return {}
    selected: dict[str, int] = {}
    for key in PARENT_VM_DELTA_KEYS:
        value = deltas.get(key)
        if isinstance(value, int):
            selected[key] = int(value)
    return selected


def _sum_delta_maps(delta_maps: Sequence[Mapping[str, int]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for deltas in delta_maps:
        for key, value in deltas.items():
            summary[key] = summary.get(key, 0) + int(value)
    return summary


def _summarize_parent_vm_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available_rows = [row for row in records if _selected_parent_vm_deltas(row)]
    dirty_rows = [row for row in available_rows if not _is_clean_benchmark_row(row)]
    all_deltas = [_selected_parent_vm_deltas(row) for row in available_rows]
    dirty_deltas = [_selected_parent_vm_deltas(row) for row in dirty_rows]
    return {
        "available_record_count": len(available_rows),
        "dirty_available_record_count": len(dirty_rows),
        "selected_delta_sums": _sum_delta_maps(all_deltas),
        "dirty_selected_delta_sums": _sum_delta_maps(dirty_deltas),
    }


def _timing_summary(records: Sequence[Mapping[str, Any]], *, timing_key: str) -> dict[str, Any]:
    timings = [float(row[timing_key]) for row in records if row.get(timing_key) is not None]
    median = statistics.median(timings) if timings else None
    relative_spread = None
    if median and len(timings) >= 2:
        relative_spread = (max(timings) - min(timings)) / median
    stability_required = len(timings) >= 3
    return {
        "seconds": timings,
        "median_seconds": median,
        "relative_spread": relative_spread,
        "stability_required": stability_required,
        "stable": (
            bool(timings)
            if not stability_required
            else relative_spread is not None
            and relative_spread <= LANE_S_TIMING_RELATIVE_SPREAD_MAX
        ),
    }


def _summarize_benchmark_memory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pageouts = [
        int(row["pageouts_delta"])
        for row in records
        if isinstance(row.get("pageouts_delta"), int)
    ]
    swapouts = [
        int(row["swapouts_delta"])
        for row in records
        if isinstance(row.get("swapouts_delta"), int)
    ]
    dirty_records = [row for row in records if not _is_clean_benchmark_row(row)]
    invalid_pressure_records = [
        row for row in records if bool(row.get("invalid_memory_pressure"))
    ]
    return {
        "dirty_record_count": len(dirty_records),
        "invalid_memory_pressure_count": len(invalid_pressure_records),
        "pageouts_delta_sum": sum(pageouts),
        "swapouts_delta_sum": sum(swapouts),
        "max_pageouts_delta": max(pageouts) if pageouts else None,
        "max_swapouts_delta": max(swapouts) if swapouts else None,
    }


def _summarize_benchmark(path: Path, *, timing_key: str = "prefill_seconds") -> dict[str, Any]:
    records = _read_jsonl(path)
    clean_records = [row for row in records if _is_clean_benchmark_row(row)]
    clean_timing = _timing_summary(clean_records, timing_key=timing_key)
    attempted_timing = _timing_summary(records, timing_key=timing_key)
    first = records[0] if records else {}
    return {
        "path": str(path),
        "record_count": len(records),
        "clean_record_count": len(clean_records),
        "all_memory_clean": len(clean_records) == len(records) and bool(records),
        "median_seconds": clean_timing["median_seconds"],
        "relative_spread": clean_timing["relative_spread"],
        "timing_stable": clean_timing["stable"],
        "clean_timing": clean_timing,
        "attempted_timing": attempted_timing,
        "memory_pressure_summary": _summarize_benchmark_memory(records),
        "effective_bits_per_weight": first.get("effective_bits_per_weight"),
        "dense_expert_params": first.get("dense_expert_params"),
        "unbound_vq_experts": first.get("unbound_vq_experts"),
        "non_expert_dtype_verified": first.get("non_expert_dtype_verified"),
        "prefill_compatibility": first.get("prefill_compatibility"),
        "parent_vm_stat_summary": _summarize_parent_vm_stats(records),
    }


def _check_le(actual: float | int | None, expected: float | int) -> bool:
    return actual is not None and float(actual) <= float(expected)


def _check_ge(actual: float | int | None, expected: float | int) -> bool:
    return actual is not None and float(actual) >= float(expected)


def _quality_checks(split_summary: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "clean_128": int(split_summary.get("clean_record_count") or 0) >= BALANCED_HARD_TARGETS["clean_rows"]
        and bool(split_summary.get("all_memory_clean")),
        "mean_ppl_ratio_le_1p05": _check_le(
            split_summary.get("mean_ppl_ratio"), BALANCED_HARD_TARGETS["mean_ppl_ratio_max"]
        ),
        "mean_kld_le_0p30": _check_le(
            split_summary.get("mean_kld"), COMMUNITY_WOW_TARGETS["mean_kld_max"]
        ),
        "p999_kld_le_3p0": _check_le(
            split_summary.get("p999_kld"), COMMUNITY_WOW_TARGETS["p999_kld_max"]
        ),
        "top1_ge_0p85": _check_ge(
            split_summary.get("mean_top1_agreement"), COMMUNITY_WOW_TARGETS["top1_min"]
        ),
    }


def _all(values: Mapping[str, bool]) -> bool:
    return all(bool(value) for value in values.values())


def _lane_s_failure_reasons(checks: Mapping[str, bool]) -> list[str]:
    labels = {
        "candidate_clean": "candidate_memory_dirty",
        "q2_control_clean": "q2_control_memory_dirty",
        "candidate_timing_stable": "candidate_timing_unstable_or_unavailable",
        "q2_control_timing_stable": "q2_control_timing_unstable_or_unavailable",
        "ratio_le_1p15": "lane_s_ratio_above_target_or_unavailable",
        "effective_bpw_le_2p1": "effective_bpw_above_2p1_or_unavailable",
        "effective_bpw_le_2p5": "effective_bpw_above_2p5_or_unavailable",
        "no_dense_routed_experts": "dense_routed_experts_present_or_unknown",
        "no_unbound_vq_experts": "unbound_vq_experts_present_or_unknown",
        "non_expert_dtype_verified": "non_expert_dtype_not_verified",
    }
    return [labels.get(name, name) for name, passed in checks.items() if not passed]
