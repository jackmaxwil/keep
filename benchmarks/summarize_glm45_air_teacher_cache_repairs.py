from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlx_vq.quality.teacher_cache import read_teacher_cache_rows
from mlx_vq.quality.teacher_cache_row_compare import summarize_clean_repaired_teacher_cache_records


def _read_eval_records(path: str | Path) -> list[dict[str, Any]]:
    return [row for row in read_teacher_cache_rows(path) if "row_index" in row]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize clean row coverage after GLM-4.5-Air teacher-cache repair evals."
    )
    parser.add_argument("--base-jsonl", required=True)
    parser.add_argument("--repair-jsonl", action="append", default=[])
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    base_records = _read_eval_records(args.base_jsonl)
    repair_records: list[dict[str, Any]] = []
    for path in args.repair_jsonl:
        repair_records.extend(_read_eval_records(path))

    report = summarize_clean_repaired_teacher_cache_records(base_records, repair_records)
    report.update(
        {
            "label": args.label,
            "base_jsonl": args.base_jsonl,
            "repair_jsonls": list(args.repair_jsonl),
        }
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "output_json": str(output_path),
            "label": args.label,
            "clean_record_count": report["clean_record_count"],
            "dirty_row_count": report["dirty_row_count"],
            "all_memory_clean": report["all_memory_clean"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
