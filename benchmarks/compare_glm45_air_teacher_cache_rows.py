from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_vq.quality.teacher_cache import read_teacher_cache_rows
from mlx_vq.quality.teacher_cache_row_compare import (
    compare_teacher_cache_rows,
    merge_repaired_teacher_cache_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GLM-4.5-Air teacher-cache eval JSONLs by row_index."
    )
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--candidate-repair-jsonl", action="append", default=[])
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--speed-gate-json")
    parser.add_argument("--top-rows", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    baseline_records = read_teacher_cache_rows(args.baseline_jsonl)
    candidate_records = read_teacher_cache_rows(args.candidate_jsonl)
    repair_records = []
    for path in args.candidate_repair_jsonl:
        repair_records.extend(read_teacher_cache_rows(path))
    merged_candidate = merge_repaired_teacher_cache_records(candidate_records, repair_records)
    speed_gate = None
    if args.speed_gate_json:
        speed_gate = json.loads(Path(args.speed_gate_json).read_text(encoding="utf-8"))

    report = compare_teacher_cache_rows(
        baseline_records,
        merged_candidate,
        candidate_label=args.candidate_label,
        top_rows=args.top_rows,
        speed_gate=speed_gate,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_path),
        "candidate_label": args.candidate_label,
        "decision": report["decision"]["label"],
        "row_count": report["row_count"],
        "max_ppl_row": report["max_ppl_row"]["prompt_id"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
