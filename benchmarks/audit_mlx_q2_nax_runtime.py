from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from mlx_vq.benchmark.glm45_air import append_jsonl
from mlx_vq.benchmark.nax_audit import (
    ProjectionName,
    q2_nax_capture_plan_records,
    q2_nax_runtime_cases,
    run_sorted_q2_case,
    summarize_gputrace,
    summarize_xctrace,
)


def _projections(value: str) -> Iterable[ProjectionName]:
    if value == "all":
        return ("gate_up", "down")
    return (value,)  # type: ignore[return-value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or run GLM-4.5-Air sorted MLX q2 NAX runtime kernel captures."
    )
    parser.add_argument("--projection", choices=["gate_up", "down", "all"], default="all")
    parser.add_argument("--tokens", type=int, action="append", default=None)
    parser.add_argument("--trace-dir", default="artifacts/traces/mlx-q2-nax")
    parser.add_argument("--trace-path")
    parser.add_argument("--run", action="store_true", help="Run the sorted q2 projection without capture.")
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Run under mx.metal.start_capture(). Requires MTL_CAPTURE_ENABLED=1.",
    )
    parser.add_argument("--summarize-trace", help="Summarize an existing .gputrace directory.")
    parser.add_argument("--append-jsonl")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.summarize_trace:
        trace_path = Path(args.summarize_trace)
        if trace_path.suffix == ".trace":
            cases = q2_nax_runtime_cases(tokens=args.tokens or [1024], projections=tuple(_projections(args.projection)))
            allowed_regexes = cases[0].allowed_kernel_regexes if cases else ()
            record = summarize_xctrace(trace_path, allowed_kernel_regexes=allowed_regexes)
        else:
            record = summarize_gputrace(args.summarize_trace)
        if args.append_jsonl:
            append_jsonl(args.append_jsonl, record)
        print(json.dumps(record, indent=2, sort_keys=True))
        return

    tokens = args.tokens or [1024, 2048, 4096]
    projections = tuple(_projections(args.projection))
    cases = q2_nax_runtime_cases(tokens=tokens, projections=projections)

    records = []
    if not args.run and not args.capture:
        records = q2_nax_capture_plan_records(
            trace_dir=args.trace_dir,
            tokens=tokens,
            projections=projections,
        )
    else:
        trace_dir = Path(args.trace_dir)
        for case in cases:
            trace_path = None
            if args.capture:
                trace_path = Path(args.trace_path) if args.trace_path else trace_dir / case.trace_name
            records.append(run_sorted_q2_case(case, capture_path=trace_path))

    for record in records:
        if args.append_jsonl:
            append_jsonl(args.append_jsonl, record)
        print(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
