from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_vq.quality.selective_precision import (
    build_custom_selective_precision_candidate,
    build_selective_precision_candidates,
    candidate_to_json,
    discover_artifact_group_files,
    materialize_linked_candidate,
)


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _custom_candidate_from_arg(spec: str):
    name, separator, policy = spec.partition("=")
    if separator != "=" or not name.strip() or not policy.strip():
        raise SystemExit(
            "--custom-candidate must use NAME=layer:projection,layer:projection"
        )
    try:
        return build_custom_selective_precision_candidate(
            name=name,
            policy=policy,
            description=(
                "Custom Lane 3 selective precision candidate supplied on the "
                "planner command line."
            ),
            rationale=("Custom policy supplied after fixed Lane 3 candidate review.",),
        )
    except ValueError as exc:
        raise SystemExit(f"invalid --custom-candidate {spec!r}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan GLM-4.5-Air Lane 3 selective-precision candidates from Lane 2 reports."
    )
    parser.add_argument("--hard-report", required=True)
    parser.add_argument("--tail-report")
    parser.add_argument("--baseline-artifact-dir", default="artifacts/glm-4.5-air-vq")
    parser.add_argument(
        "--high-bit-artifact-dir",
        default="artifacts/glm-4.5-air-vq2-e8p-rtn-uniform-parallel8",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--materialize-root",
        help="Optional root directory where linked candidate artifact directories are created.",
    )
    parser.add_argument(
        "--allow-existing-materialized",
        action="store_true",
        help="Reuse existing linked candidate directories instead of failing.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        help="Optional candidate name to include/materialize; repeatable. Defaults to all.",
    )
    parser.add_argument(
        "--custom-candidate",
        action="append",
        help=(
            "Optional custom candidate in NAME=layer:projection,layer:projection "
            "format; repeatable."
        ),
    )
    parser.add_argument("--secondary-hard-layers", type=int, default=3)
    parser.add_argument("--tail-layers", type=int, default=4)
    args = parser.parse_args()

    hard_report = _read_json(args.hard_report)
    tail_report = _read_json(args.tail_report) if args.tail_report else None
    groups = discover_artifact_group_files(
        baseline_dir=args.baseline_artifact_dir,
        high_bit_dir=args.high_bit_artifact_dir,
    )
    candidates = build_selective_precision_candidates(
        hard_report=hard_report,
        tail_report=tail_report,
        secondary_hard_layers=args.secondary_hard_layers,
        tail_layers=args.tail_layers,
    )
    if args.custom_candidate:
        custom_candidates = tuple(
            _custom_candidate_from_arg(spec) for spec in args.custom_candidate
        )
        duplicate_names = sorted(
            {
                candidate.name
                for candidate in candidates
                if candidate.name in {custom.name for custom in custom_candidates}
            }
        )
        if duplicate_names:
            raise SystemExit(f"duplicate custom candidate name(s): {', '.join(duplicate_names)}")
        candidates = (*candidates, *custom_candidates)
    if args.candidate:
        requested = set(args.candidate)
        unknown = sorted(requested - {candidate.name for candidate in candidates})
        if unknown:
            raise SystemExit(f"unknown candidate(s): {', '.join(unknown)}")
        candidates = tuple(candidate for candidate in candidates if candidate.name in requested)

    materialized = []
    if args.materialize_root:
        materialize_root = Path(args.materialize_root)
        for candidate in candidates:
            output_dir = materialize_root / candidate.name
            materialized.append(
                {
                    "candidate": candidate.name,
                    **materialize_linked_candidate(
                        candidate=candidate,
                        baseline_dir=args.baseline_artifact_dir,
                        high_bit_dir=args.high_bit_artifact_dir,
                        output_dir=output_dir,
                        groups=groups,
                        allow_existing=args.allow_existing_materialized,
                    ),
                }
            )

    report = {
        "schema_version": 1,
        "evidence_scope": "lane3_selective_precision_allocation_and_linked_artifacts",
        "hard_report": args.hard_report,
        "tail_report": args.tail_report,
        "baseline_artifact_dir": args.baseline_artifact_dir,
        "high_bit_artifact_dir": args.high_bit_artifact_dir,
        "candidate_count": len(candidates),
        "total_group_count": len(groups),
        "projection_granularity": "layer_projection",
        "expert_selectivity_supported": False,
        "notes": [
            "Candidates reuse existing 8-bit baseline shards and existing uniform E8P shards; no source conversion is performed.",
            "Current artifact format is whole layer/projection granular, so Lane 2 selected experts are rationale only.",
            "This is n=5 directional allocation evidence until Lane 0 widened authority exists.",
        ],
        "candidates": [
            candidate_to_json(candidate, groups)
            for candidate in candidates
        ],
        "materialized": materialized,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_path),
        "candidate_count": report["candidate_count"],
        "materialized_count": len(materialized),
        "candidate_names": [candidate["name"] for candidate in report["candidates"]],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
