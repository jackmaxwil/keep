from __future__ import annotations

import argparse
import json
from pathlib import Path

from keep.quality.teacher_cache_attribution import (
    build_teacher_cache_attribution_report,
    load_teacher_cache_run,
)


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--candidate-jsonl must use LABEL=PATH, for example e8p=artifacts/run.jsonl"
        )
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("candidate label must be non-empty")
    if not path:
        raise argparse.ArgumentTypeError("candidate path must be non-empty")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank GLM-4.5-Air teacher-cache prompt/token failures and compare "
            "candidate eval JSONL runs."
        )
    )
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument(
        "--candidate-jsonl",
        action="append",
        default=[],
        type=_parse_candidate,
        help="Candidate eval JSONL as LABEL=PATH. May be supplied more than once.",
    )
    parser.add_argument("--top-tokens", type=int, default=20)
    parser.add_argument("--top-prompts", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    baseline = load_teacher_cache_run(args.baseline_label, args.baseline_jsonl)
    candidates = [
        load_teacher_cache_run(label, path)
        for label, path in args.candidate_jsonl
    ]
    report = build_teacher_cache_attribution_report(
        baseline,
        candidates=candidates,
        top_tokens=args.top_tokens,
        top_prompts=args.top_prompts,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_json": str(output_path),
        "baseline": args.baseline_jsonl,
        "candidate_count": len(candidates),
        "top_kld_prompt": report["top_kld_tokens"][0]["prompt_id"]
        if report["top_kld_tokens"]
        else None,
        "quality_floor_violation_count": len(report["quality_floor_violations"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
