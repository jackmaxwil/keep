from __future__ import annotations

import argparse
import json
from pathlib import Path

from keep.quality.calibration_importance import (
    build_calibration_importance_report,
    read_jsonl_records,
)


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank GLM-4.5-Air selective-precision targets by combining source-error "
            "probe rows with activation calibration and expert coverage."
        )
    )
    parser.add_argument("--source-probe-report", required=True)
    parser.add_argument("--calibration-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-candidate-layers", type=int, default=3)
    args = parser.parse_args()

    report = build_calibration_importance_report(
        source_probe_report=_read_json(args.source_probe_report),
        calibration_records=read_jsonl_records(args.calibration_jsonl),
        top_candidate_layers=args.top_candidate_layers,
    )
    report.update(
        {
            "source_probe_report": args.source_probe_report,
            "calibration_jsonl": args.calibration_jsonl,
        }
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "group_ranking_count": len(report["group_rankings"]),
                "candidate_count": len(report["candidates"]),
                "top_group": report["group_rankings"][0] if report["group_rankings"] else None,
                "candidate_names": [candidate["name"] for candidate in report["candidates"]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
