from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlx_vq.quality.rc_gates import (
    COMMUNITY_WOW_TARGETS,
    _read_jsonl,
    _summarize_eval_split,
)


DEFAULT_RESIDENT_AUDIT = Path(
    "artifacts/quality/glm45-air-r4-bump-lm-head-embed-fixed-loader-resident-byte-audit-20260704.json"
)
DEFAULT_REPORT = Path(
    "artifacts/quality/glm45-air-r4-bump-lm-head-embed-fixed-loader-report-full-repair2-20260704.jsonl"
)
DEFAULT_SELECTION = Path(
    "artifacts/quality/glm45-air-r4-bump-lm-head-embed-fixed-loader-selection-full-repair-20260704.jsonl"
)
DEFAULT_LANE_S = Path(
    "artifacts/benchmarks/glm45-air-r4-bump-lm-head-embed-fixed-loader-lane-s-prefill1k-20260704.jsonl"
)
DEFAULT_OUTPUT = Path(
    "artifacts/quality/glm45-air-workstream1-bump-fixed-loader-verdict-20260704.json"
)


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _metric_le(summary: Mapping[str, Any], key: str, ceiling: float) -> bool:
    value = summary.get(key)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value <= ceiling


def _metric_ge(summary: Mapping[str, Any], key: str, floor: float) -> bool:
    value = summary.get(key)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= floor


def _quality_promotable(summary: Mapping[str, Any]) -> bool:
    return (
        bool(summary.get("all_memory_clean"))
        and _metric_ge(
            summary,
            "mean_top1_agreement",
            COMMUNITY_WOW_TARGETS["top1_min"],
        )
        and _metric_le(summary, "mean_kld", COMMUNITY_WOW_TARGETS["mean_kld_max"])
        and _metric_le(summary, "p999_kld", COMMUNITY_WOW_TARGETS["p999_kld_max"])
    )


def _lane_s_summary(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    invalid_rows = [
        row
        for row in rows
        if bool(row.get("invalid_memory_pressure"))
        or int(row.get("pageouts_delta") or 0) != 0
        or int(row.get("swapouts_delta") or 0) != 0
    ]
    valid_rows = [row for row in rows if row not in invalid_rows]
    return {
        "path": str(path),
        "row_count": len(rows),
        "valid_row_count": len(valid_rows),
        "invalid_memory_pressure_row_count": len(invalid_rows),
        "pageouts_delta_values": [
            int(row.get("pageouts_delta") or 0) for row in invalid_rows
        ],
        "swapouts_delta_values": [
            int(row.get("swapouts_delta") or 0) for row in invalid_rows
        ],
        "speed_claim": any(bool(row.get("speed_claim")) for row in rows),
        "lane_s_claim": any(bool(row.get("lane_s_claim")) for row in rows),
        "valid_for_lane_s": bool(rows) and not invalid_rows,
    }


def build_verdict(
    *,
    resident_audit_path: Path = DEFAULT_RESIDENT_AUDIT,
    report_jsonl_path: Path = DEFAULT_REPORT,
    selection_jsonl_path: Path = DEFAULT_SELECTION,
    lane_s_jsonl_path: Path = DEFAULT_LANE_S,
) -> dict[str, Any]:
    audit = _read_json_object(resident_audit_path)
    report = _summarize_eval_split(report_jsonl_path)
    selection = _summarize_eval_split(selection_jsonl_path)
    lane_s = _lane_s_summary(lane_s_jsonl_path)
    full_split_record_count_satisfied = (
        int(report.get("record_count") or 0) >= 128
        and int(selection.get("record_count") or 0) >= 128
    )
    resident_byte_reduction = int(audit.get("resident_byte_reduction_bytes") or 0)
    source_non_expert_delta = int(audit.get("source_non_expert_delta_bytes") or 0)
    source_skipped_delta = int(audit.get("source_skipped_by_surface_total_bytes") or 0)
    residency_promotable = (
        resident_byte_reduction > 0
        and source_non_expert_delta < 0
        and source_skipped_delta == 0
        and bool(audit.get("source_non_expert_complete"))
    )
    quality_promotable = _quality_promotable(report) and _quality_promotable(selection)
    return {
        "schema_version": 1,
        "record_type": "glm45_air_workstream1_bump_fixed_loader_verdict",
        "created_date": "2026-07-04",
        "decision": "bump_lm_head_embed_fixed_loader_metric_backed_not_promotable",
        "mechanism": "bump_non_expert_precision_lm_head_embed_fixed_loader",
        "implementation_bug_fixed": True,
        "old_top1_zero_result_superseded_by_bug_signal": True,
        "non_expert_precision_exhausted": False,
        "full_split_record_count_satisfied": full_split_record_count_satisfied,
        "heavy_slice_rule_satisfied": full_split_record_count_satisfied,
        "quality_promotable": quality_promotable,
        "residency_promotable": residency_promotable,
        "lane_s_valid": bool(lane_s["valid_for_lane_s"]),
        "do_not_compose": True,
        "next_recommendation": (
            "do_not_compose_bump; continue P1 with a materially different top1 "
            "mechanism; rerun Lane S only for a future residency-changing bump"
        ),
        "resident_audit": {
            "path": str(resident_audit_path),
            "artifact_dir": audit.get("artifact_dir"),
            "counted_resident_total_bytes": audit.get("counted_resident_total_bytes"),
            "resident_byte_reduction_bytes": resident_byte_reduction,
            "source_non_expert_delta_bytes": source_non_expert_delta,
            "source_skipped_by_surface_total_bytes": source_skipped_delta,
            "source_non_expert_complete": bool(audit.get("source_non_expert_complete")),
        },
        "report": report,
        "selection": selection,
        "lane_s": lane_s,
        "evidence": {
            "resident_audit": str(resident_audit_path),
            "report_jsonl": str(report_jsonl_path),
            "selection_jsonl": str(selection_jsonl_path),
            "lane_s_jsonl": str(lane_s_jsonl_path),
        },
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "transport_scope": "local_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize corrected W1 fixed-loader bump evidence."
    )
    parser.add_argument("--resident-audit-json", type=Path, default=DEFAULT_RESIDENT_AUDIT)
    parser.add_argument("--report-jsonl", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--selection-jsonl", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--lane-s-jsonl", type=Path, default=DEFAULT_LANE_S)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    verdict = build_verdict(
        resident_audit_path=args.resident_audit_json,
        report_jsonl_path=args.report_jsonl,
        selection_jsonl_path=args.selection_jsonl,
        lane_s_jsonl_path=args.lane_s_jsonl,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
