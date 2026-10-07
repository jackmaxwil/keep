from __future__ import annotations

from typing import Any

from keep.quality.gates import evaluate_quality_gate
from keep.quality.teacher_cache import summarize_teacher_cache_records


DEFAULT_GATE3_TARGETS = {
    "mean_kld_max": 0.25,
    "mean_top1_agreement_min": 0.85,
    "mean_ppl_ratio_max": 1.15,
    "max_ppl_ratio_max": 1.30,
    "p999_kld_max": 2.0,
}


def merge_repaired_teacher_cache_records(
    records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_row: dict[int, dict[str, Any]] = {}
    for record in records:
        row_index = _row_index(record)
        if row_index in by_row:
            raise ValueError(f"duplicate row_index {row_index} in primary records")
        by_row[row_index] = dict(record)
    for record in repair_records or []:
        row_index = _row_index(record)
        if row_index not in by_row:
            raise ValueError(f"repair row_index {row_index} is absent from primary records")
        by_row[row_index] = dict(record)
    return [by_row[row_index] for row_index in sorted(by_row)]


def merge_clean_repaired_teacher_cache_records(
    records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_row: dict[int, dict[str, Any]] = {}
    for record in records:
        row_index = _row_index(record)
        if row_index in by_row:
            raise ValueError(f"duplicate row_index {row_index} in primary records")
        by_row[row_index] = dict(record)
    for record in repair_records or []:
        row_index = _row_index(record)
        if row_index not in by_row:
            raise ValueError(f"repair row_index {row_index} is absent from primary records")
        if _memory_clean(record):
            by_row[row_index] = dict(record)
    return [by_row[row_index] for row_index in sorted(by_row)]


def summarize_clean_repaired_teacher_cache_records(
    records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    merged = merge_clean_repaired_teacher_cache_records(records, repair_records)
    summary = summarize_teacher_cache_records(merged)
    dirty_rows = [
        _row_memory_report(record)
        for record in merged
        if not _memory_clean(record)
    ]
    original_dirty = {
        _row_index(record)
        for record in records
        if not _memory_clean(record)
    }
    remaining_dirty = {_row_index(row) for row in merged if not _memory_clean(row)}
    clean_repaired = sorted(original_dirty - remaining_dirty)
    return {
        "schema_version": 1,
        "record_type": "air_vq_teacher_cache_clean_repair_summary",
        "row_count": len(merged),
        "clean_record_count": summary["clean_record_count"],
        "all_memory_clean": summary["all_memory_clean"],
        "clean_repaired_row_indices": clean_repaired,
        "clean_repaired_row_count": len(clean_repaired),
        "dirty_rows": dirty_rows,
        "dirty_row_indices": [row["row_index"] for row in dirty_rows],
        "dirty_row_count": len(dirty_rows),
        "summary": summary,
    }


def compare_teacher_cache_rows(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    candidate_label: str,
    top_rows: int = 10,
    speed_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if top_rows <= 0:
        raise ValueError("top_rows must be positive")
    baseline_by_row = _records_by_row_index(baseline_records, label="baseline")
    candidate_by_row = _records_by_row_index(candidate_records, label="candidate")
    if set(baseline_by_row) != set(candidate_by_row):
        missing = sorted(set(baseline_by_row) - set(candidate_by_row))
        extra = sorted(set(candidate_by_row) - set(baseline_by_row))
        raise ValueError(f"row_index mismatch: missing={missing}, extra={extra}")

    paired_baseline = [baseline_by_row[row_index] for row_index in sorted(baseline_by_row)]
    paired_candidate = [candidate_by_row[row_index] for row_index in sorted(candidate_by_row)]
    row_deltas = [
        _row_delta(baseline, candidate)
        for baseline, candidate in zip(paired_baseline, paired_candidate, strict=True)
    ]
    quality_gate = evaluate_quality_gate(
        baseline_records=paired_baseline,
        candidate_records=paired_candidate,
    )
    baseline_summary = summarize_teacher_cache_records(paired_baseline)
    candidate_summary = summarize_teacher_cache_records(paired_candidate)
    speed_failed = _speed_gate_failed(speed_gate)
    quality_label = quality_gate["verdict"]
    decision_label = "SPEED_FAIL" if speed_failed else quality_label

    return {
        "schema_version": 1,
        "record_type": "air_vq_teacher_cache_row_compare",
        "candidate_label": candidate_label,
        "row_count": len(row_deltas),
        "clean_candidate_row_count": sum(1 for row in paired_candidate if _memory_clean(row)),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "summary_delta": {
            key: _delta(candidate_summary.get(key), baseline_summary.get(key))
            for key in (
                "mean_ppl_ratio",
                "max_ppl_ratio",
                "mean_kld",
                "p999_kld",
                "mean_top1_agreement",
            )
        },
        "decision": {
            "label": decision_label,
            "quality_label_without_speed": quality_label,
            "speed_gate_failed": speed_failed,
            "quality_gate": quality_gate,
        },
        "gap_closure": _gap_closure(baseline_summary, candidate_summary),
        "max_ppl_row": max(row_deltas, key=lambda row: row["candidate_ppl_ratio"]),
        "top_mean_kld_improved_rows": sorted(
            row_deltas,
            key=lambda row: (
                row["delta_mean_kld"],
                -row["delta_ppl_ratio"],
                row["row_index"],
            ),
        )[:top_rows],
        "top_mean_kld_worsened_rows": sorted(
            row_deltas,
            key=lambda row: (
                -row["delta_mean_kld"],
                -row["delta_ppl_ratio"],
                row["row_index"],
            ),
        )[:top_rows],
        "top_ppl_worsened_rows": sorted(
            row_deltas,
            key=lambda row: (
                -row["delta_ppl_ratio"],
                -row["delta_mean_kld"],
                row["row_index"],
            ),
        )[:top_rows],
        "top_p999_worsened_rows": sorted(
            row_deltas,
            key=lambda row: (
                -row["delta_p999_kld"],
                -row["delta_mean_kld"],
                row["row_index"],
            ),
        )[:top_rows],
        "kld_improved_ppl_worsened_rows": [
            row for row in sorted(
                row_deltas,
                key=lambda item: (-item["delta_ppl_ratio"], item["delta_mean_kld"], item["row_index"]),
            )
            if row["delta_mean_kld"] < 0.0 and row["delta_ppl_ratio"] > 0.0
        ][:top_rows],
        "row_deltas": row_deltas,
    }


def _records_by_row_index(records: list[dict[str, Any]], *, label: str) -> dict[int, dict[str, Any]]:
    by_row: dict[int, dict[str, Any]] = {}
    for record in records:
        row_index = _row_index(record)
        if row_index in by_row:
            raise ValueError(f"duplicate row_index {row_index} in {label} records")
        by_row[row_index] = record
    return by_row


def _row_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    row_index = _row_index(baseline)
    candidate_row_index = _row_index(candidate)
    if row_index != candidate_row_index:
        raise ValueError(f"paired row_index mismatch: {row_index} != {candidate_row_index}")
    baseline_prompt = str(baseline.get("prompt_id", ""))
    candidate_prompt = str(candidate.get("prompt_id", ""))
    if baseline_prompt != candidate_prompt:
        raise ValueError(
            f"row {row_index} prompt_id mismatch: {baseline_prompt!r} != {candidate_prompt!r}"
        )
    baseline_p999 = _record_p999(baseline)
    candidate_p999 = _record_p999(candidate)
    return {
        "row_index": row_index,
        "prompt_id": candidate_prompt,
        "baseline_ppl_ratio": float(baseline["ppl_ratio"]),
        "candidate_ppl_ratio": float(candidate["ppl_ratio"]),
        "delta_ppl_ratio": float(candidate["ppl_ratio"]) - float(baseline["ppl_ratio"]),
        "baseline_mean_kld": float(baseline["mean_kld"]),
        "candidate_mean_kld": float(candidate["mean_kld"]),
        "delta_mean_kld": float(candidate["mean_kld"]) - float(baseline["mean_kld"]),
        "baseline_p999_kld": baseline_p999,
        "candidate_p999_kld": candidate_p999,
        "delta_p999_kld": candidate_p999 - baseline_p999,
        "baseline_top1_agreement": _float_or_none(baseline.get("top1_agreement")),
        "candidate_top1_agreement": _float_or_none(candidate.get("top1_agreement")),
        "delta_top1_agreement": _delta(
            candidate.get("top1_agreement"),
            baseline.get("top1_agreement"),
        ),
        "candidate_memory_clean": _memory_clean(candidate),
    }


def _gap_closure(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    targets: dict[str, float] = DEFAULT_GATE3_TARGETS,
) -> dict[str, dict[str, float | None]]:
    return {
        "mean_kld": _lower_is_better_gap(
            baseline_summary.get("mean_kld"),
            candidate_summary.get("mean_kld"),
            targets["mean_kld_max"],
        ),
        "mean_ppl_ratio": _lower_is_better_gap(
            baseline_summary.get("mean_ppl_ratio"),
            candidate_summary.get("mean_ppl_ratio"),
            targets["mean_ppl_ratio_max"],
        ),
        "max_ppl_ratio": _lower_is_better_gap(
            baseline_summary.get("max_ppl_ratio"),
            candidate_summary.get("max_ppl_ratio"),
            targets["max_ppl_ratio_max"],
        ),
        "p999_kld": _lower_is_better_gap(
            baseline_summary.get("p999_kld"),
            candidate_summary.get("p999_kld"),
            targets["p999_kld_max"],
        ),
        "mean_top1_agreement": _higher_is_better_gap(
            baseline_summary.get("mean_top1_agreement"),
            candidate_summary.get("mean_top1_agreement"),
            targets["mean_top1_agreement_min"],
        ),
    }


def _lower_is_better_gap(value: Any, candidate_value: Any, target: float) -> dict[str, float | None]:
    baseline = _float_or_none(value)
    candidate = _float_or_none(candidate_value)
    if baseline is None or candidate is None:
        return {"target": target, "remaining_gap": None, "closed_fraction": None}
    initial_gap = baseline - target
    remaining_gap = candidate - target
    return {
        "target": target,
        "remaining_gap": remaining_gap,
        "closed_fraction": (baseline - candidate) / initial_gap if initial_gap else None,
    }


def _higher_is_better_gap(value: Any, candidate_value: Any, target: float) -> dict[str, float | None]:
    baseline = _float_or_none(value)
    candidate = _float_or_none(candidate_value)
    if baseline is None or candidate is None:
        return {"target": target, "remaining_gap": None, "closed_fraction": None}
    initial_gap = target - baseline
    remaining_gap = target - candidate
    return {
        "target": target,
        "remaining_gap": remaining_gap,
        "closed_fraction": (candidate - baseline) / initial_gap if initial_gap else None,
    }


def _record_p999(record: dict[str, Any]) -> float:
    if record.get("p999_kld") is not None:
        return float(record["p999_kld"])
    values = sorted(float(value) for value in record.get("token_klds") or [])
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = 0.999 * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _memory_clean(record: dict[str, Any]) -> bool:
    return record.get("pageouts_delta") == 0 and record.get("swapouts_delta") == 0


def _row_memory_report(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": _row_index(record),
        "prompt_id": record.get("prompt_id"),
        "pageouts_delta": record.get("pageouts_delta"),
        "swapouts_delta": record.get("swapouts_delta"),
    }


def _speed_gate_failed(speed_gate: dict[str, Any] | None) -> bool:
    if not speed_gate:
        return False
    verdict = str(speed_gate.get("speed_verdict") or speed_gate.get("verdict") or "").upper()
    return verdict in {"FAIL", "SPEED_FAIL", "REJECT_SPEED_FAIL"}


def _row_index(record: dict[str, Any]) -> int:
    if "row_index" not in record:
        raise ValueError("teacher-cache record is missing row_index")
    return int(record["row_index"])


def _delta(candidate_value: Any, baseline_value: Any) -> float | None:
    candidate = _float_or_none(candidate_value)
    baseline = _float_or_none(baseline_value)
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
