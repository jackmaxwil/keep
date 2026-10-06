from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            row["_source_jsonl"] = str(path)
            rows.append(row)
    return rows


def _is_clean_benchmark_row(row: dict[str, Any]) -> bool:
    return (
        row.get("finite_output") is True
        and int(row.get("pageouts_delta") or 0) == 0
        and int(row.get("swapouts_delta") or 0) == 0
    )


def _row_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["tokens"]), str(row["projection"])


def _baseline_key(row: dict[str, Any]) -> tuple[int, str, str]:
    projection = str(row["projection"])
    artifact_projection = str(row.get("artifact_projection") or projection)
    return int(row["tokens"]), projection, artifact_projection


def _best_q2_by_shape(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not _is_clean_benchmark_row(row):
            continue
        if "tokens" not in row or "projection" not in row or "ms_per_iter" not in row:
            continue
        key = _row_key(row)
        current = best.get(key)
        if current is None or float(row["ms_per_iter"]) < float(current["ms_per_iter"]):
            best[key] = row
    return best


def _best_baseline_by_shape(
    rows: list[dict[str, Any]],
    baseline_variant: str,
) -> dict[tuple[int, str, str], dict[str, Any]]:
    best: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("variant") != baseline_variant or not _is_clean_benchmark_row(row):
            continue
        if "tokens" not in row or "projection" not in row or "ms_per_iter" not in row:
            continue
        key = _baseline_key(row)
        current = best.get(key)
        if current is None or float(row["ms_per_iter"]) < float(current["ms_per_iter"]):
            best[key] = row
    return best


def _projection_summaries(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in comparisons:
        projection = str(row["projection"])
        summary = summaries.setdefault(
            projection,
            {
                "comparison_count": 0,
                "clean_comparison_count": 0,
                "best_ratio_to_q2": None,
                "worst_ratio_to_q2": None,
                "best_variant": None,
                "worst_variant": None,
            },
        )
        summary["comparison_count"] += 1
        if row["candidate_memory_clean"]:
            summary["clean_comparison_count"] += 1
        ratio = float(row["ratio_to_q2"])
        if summary["best_ratio_to_q2"] is None or ratio < float(summary["best_ratio_to_q2"]):
            summary["best_ratio_to_q2"] = ratio
            summary["best_variant"] = row["candidate_variant"]
        if summary["worst_ratio_to_q2"] is None or ratio > float(summary["worst_ratio_to_q2"]):
            summary["worst_ratio_to_q2"] = ratio
            summary["worst_variant"] = row["candidate_variant"]
    return summaries


def _baseline_comparisons(
    rows: list[dict[str, Any]],
    *,
    baseline_variant: str | None,
) -> list[dict[str, Any]]:
    if baseline_variant is None:
        return []
    baseline_by_shape = _best_baseline_by_shape(rows, baseline_variant)
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if (
            row.get("variant") == baseline_variant
            or "tokens" not in row
            or "projection" not in row
            or "ms_per_iter" not in row
        ):
            continue
        baseline = baseline_by_shape.get(_baseline_key(row))
        if baseline is None:
            continue
        candidate_ms = float(row["ms_per_iter"])
        baseline_ms = float(baseline["ms_per_iter"])
        comparisons.append(
            {
                "tokens": int(row["tokens"]),
                "projection": str(row["projection"]),
                "artifact_projection": row.get("artifact_projection") or row["projection"],
                "candidate_variant": row.get("variant"),
                "baseline_variant": baseline_variant,
                "candidate_ms_per_iter": candidate_ms,
                "baseline_ms_per_iter": baseline_ms,
                "ratio_to_baseline": candidate_ms / baseline_ms,
                "beats_baseline": candidate_ms < baseline_ms,
                "candidate_memory_clean": _is_clean_benchmark_row(row),
                "baseline_source_jsonl": baseline.get("_source_jsonl"),
                "candidate_source_jsonl": row.get("_source_jsonl"),
                "candidate_diagnostic_phase": row.get("diagnostic_phase"),
                "candidate_vq_group_size": row.get("vq_group_size"),
                "candidate_input_dims": row.get("input_dims"),
                "candidate_output_dims": row.get("output_dims"),
            }
        )
    comparisons.sort(
        key=lambda row: (
            str(row["artifact_projection"]),
            int(row["tokens"]),
            float(row["ratio_to_baseline"]),
        )
    )
    return comparisons


def analyze_benchmark_files(
    *,
    e8p_files: list[Path],
    q2_files: list[Path],
    parity_ratio: float = 1.25,
    lane_s_ratio: float = 1.15,
    baseline_variant: str | None = None,
) -> dict[str, Any]:
    candidate_rows = [
        row
        for path in e8p_files
        for row in _read_jsonl(path)
        if row.get("variant") != "mlx_q2"
    ]
    q2_rows = [
        row
        for path in q2_files
        for row in _read_jsonl(path)
        if row.get("variant") == "mlx_q2"
    ]
    q2_by_shape = _best_q2_by_shape(q2_rows)
    comparisons: list[dict[str, Any]] = []
    unmatched_candidates: list[dict[str, Any]] = []
    for row in candidate_rows:
        if "tokens" not in row or "projection" not in row or "ms_per_iter" not in row:
            continue
        key = _row_key(row)
        q2_row = q2_by_shape.get(key)
        if q2_row is None:
            unmatched_candidates.append(
                {
                    "candidate_source_jsonl": row.get("_source_jsonl"),
                    "candidate_variant": row.get("variant"),
                    "tokens": int(row["tokens"]),
                    "projection": str(row["projection"]),
                    "reason": "no clean q2 row for tokens/projection",
                }
            )
            continue
        candidate_ms = float(row["ms_per_iter"])
        q2_ms = float(q2_row["ms_per_iter"])
        ratio = candidate_ms / q2_ms
        comparisons.append(
            {
                "tokens": int(row["tokens"]),
                "projection": str(row["projection"]),
                "candidate_variant": row.get("variant"),
                "candidate_ms_per_iter": candidate_ms,
                "q2_ms_per_iter": q2_ms,
                "ratio_to_q2": ratio,
                "parity_ratio": float(parity_ratio),
                "lane_s_ratio": float(lane_s_ratio),
                "parity_pass": ratio <= parity_ratio,
                "lane_s_pass": ratio <= lane_s_ratio,
                "candidate_memory_clean": _is_clean_benchmark_row(row),
                "q2_source_jsonl": q2_row.get("_source_jsonl"),
                "candidate_source_jsonl": row.get("_source_jsonl"),
                "candidate_diagnostic_phase": row.get("diagnostic_phase"),
                "candidate_vq_group_size": row.get("vq_group_size"),
                "candidate_input_dims": row.get("input_dims"),
                "candidate_output_dims": row.get("output_dims"),
                "candidate_mlx_peak_bytes": row.get("mlx_peak_bytes"),
            }
        )
    comparisons.sort(key=lambda row: (float(row["ratio_to_q2"]), row["tokens"], row["projection"]))
    worst = max(comparisons, key=lambda row: float(row["ratio_to_q2"])) if comparisons else None
    closest = min(comparisons, key=lambda row: float(row["ratio_to_q2"])) if comparisons else None
    return {
        "schema_version": 1,
        "record_type": "glm45_air_e8p_benchmark_analysis",
        "candidate_files": [str(path) for path in e8p_files],
        "q2_files": [str(path) for path in q2_files],
        "candidate_row_count": len(candidate_rows),
        "q2_row_count": len(q2_rows),
        "clean_candidate_row_count": sum(1 for row in candidate_rows if _is_clean_benchmark_row(row)),
        "clean_q2_row_count": sum(1 for row in q2_rows if _is_clean_benchmark_row(row)),
        "comparison_count": len(comparisons),
        "parity_ratio": float(parity_ratio),
        "lane_s_ratio": float(lane_s_ratio),
        "all_parity_pass": bool(comparisons) and all(row["parity_pass"] for row in comparisons),
        "all_lane_s_pass": bool(comparisons) and all(row["lane_s_pass"] for row in comparisons),
        "closest_ratio": closest,
        "worst_ratio": worst,
        "projection_summaries": _projection_summaries(comparisons),
        "baseline_variant": baseline_variant,
        "baseline_comparisons": _baseline_comparisons(
            candidate_rows,
            baseline_variant=baseline_variant,
        ),
        "comparisons": comparisons,
        "unmatched_candidates": unmatched_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GLM-4.5-Air E8P NAX projection benchmark JSONLs against q2 controls."
    )
    parser.add_argument("--candidate-jsonl", action="append", required=True)
    parser.add_argument("--q2-jsonl", action="append", required=True)
    parser.add_argument("--parity-ratio", type=float, default=1.25)
    parser.add_argument("--lane-s-ratio", type=float, default=1.15)
    parser.add_argument("--baseline-variant")
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()
    summary = analyze_benchmark_files(
        e8p_files=[Path(path) for path in args.candidate_jsonl],
        q2_files=[Path(path) for path in args.q2_jsonl],
        parity_ratio=args.parity_ratio,
        lane_s_ratio=args.lane_s_ratio,
        baseline_variant=args.baseline_variant,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    if args.append_jsonl:
        with Path(args.append_jsonl).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
