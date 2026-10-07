from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from keep.quality.teacher_cache import validate_teacher_cache_metadata


def _append_jsonl(path: str | Path, record: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a GLM-4.5-Air off-box teacher-cache before resident VQ eval."
    )
    parser.add_argument("--teacher-jsonl", required=True, help="Teacher cache metadata JSONL.")
    parser.add_argument(
        "--cache-root",
        help="Root for tensor shards referenced by the teacher JSONL. Defaults to JSONL parent.",
    )
    parser.add_argument("--append-jsonl", help="Optional JSONL summary artifact path.")
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Print and append invalid summaries but return success.",
    )
    parser.add_argument(
        "--min-top-k",
        type=int,
        default=128,
        help="Minimum accepted top-k width for rows without a smaller full-logit vocab.",
    )
    parser.add_argument(
        "--check-values",
        action="store_true",
        help="Load referenced tensors and check finite values plus token-id ranges.",
    )
    args = parser.parse_args()
    if args.min_top_k <= 0:
        parser.error("--min-top-k must be positive")

    teacher_jsonl = Path(args.teacher_jsonl)
    cache_root = Path(args.cache_root) if args.cache_root is not None else teacher_jsonl.parent
    summary = validate_teacher_cache_metadata(
        teacher_jsonl,
        cache_root=cache_root,
        min_top_k=args.min_top_k,
        check_values=args.check_values,
    )
    if args.append_jsonl:
        _append_jsonl(args.append_jsonl, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["ok"] and not args.allow_invalid:
        sys.exit(1)


if __name__ == "__main__":
    main()
