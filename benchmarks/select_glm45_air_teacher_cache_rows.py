from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_row_indices(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    try:
        indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("row indices must be a comma-separated list of integers") from error
    if any(index < 0 for index in indices):
        raise ValueError("row indices must be zero or greater")
    return indices


def _read_eval_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "top1_agreement" not in record or "mean_kld" not in record:
            continue
        row_index = record.get("row_index", len(rows))
        if not isinstance(row_index, int):
            raise ValueError(f"{path}:{line_number} row_index must be an integer")
        rows.append(
            {
                "row_index": row_index,
                "prompt_id": record.get("prompt_id"),
                "top1_agreement": float(record["top1_agreement"]),
                "mean_kld": float(record["mean_kld"]),
                "ppl_ratio": float(record.get("ppl_ratio", 0.0)),
                "memory_clean": bool(
                    record.get("pageouts_delta") == 0 and record.get("swapouts_delta") == 0
                ),
            }
        )
    if not rows:
        raise ValueError(f"{path} contained no eval rows")
    return rows


def _ranked_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    reverse: bool,
    pool_size: int,
    avoided: set[int],
) -> list[tuple[int, dict[str, Any]]]:
    eligible = [row for row in rows if row["row_index"] not in avoided]
    if reverse:
        ordered = sorted(eligible, key=lambda row: (-row[metric], row["row_index"]))
    else:
        ordered = sorted(eligible, key=lambda row: (row[metric], row["row_index"]))
    return list(
        enumerate(
            ordered[:pool_size],
            start=1,
        )
    )


def select_target_rows(
    report_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    *,
    count: int,
    pool_size: int,
    avoided_row_indices: tuple[int, ...] = (),
    top1_weight: float = 1.0,
    kld_weight: float = 0.5,
    shared_signal_only: bool = False,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if top1_weight < 0 or kld_weight < 0:
        raise ValueError("weights must be zero or greater")
    avoided = set(avoided_row_indices)
    scores: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "row_index": None,
            "score": 0.0,
            "sources": [],
            "report": None,
            "selection": None,
        }
    )
    split_rows = {"report": report_rows, "selection": selection_rows}
    for split, rows in split_rows.items():
        by_index = {row["row_index"]: row for row in rows}
        for rank, row in _ranked_rows(
            rows,
            metric="top1_agreement",
            reverse=False,
            pool_size=pool_size,
            avoided=avoided,
        ):
            row_index = row["row_index"]
            entry = scores[row_index]
            entry["row_index"] = row_index
            entry[split] = row
            points = top1_weight * (pool_size - rank + 1)
            entry["score"] += points
            entry["sources"].append({"split": split, "metric": "low_top1", "rank": rank, "points": points})
        for rank, row in _ranked_rows(
            rows,
            metric="mean_kld",
            reverse=True,
            pool_size=pool_size,
            avoided=avoided,
        ):
            row_index = row["row_index"]
            entry = scores[row_index]
            entry["row_index"] = row_index
            entry[split] = row
            points = kld_weight * (pool_size - rank + 1)
            entry["score"] += points
            entry["sources"].append({"split": split, "metric": "high_kld", "rank": rank, "points": points})
        for row_index, entry in list(scores.items()):
            if entry[split] is None and row_index in by_index:
                entry[split] = by_index[row_index]

    candidates = list(scores.values())
    if shared_signal_only:
        candidates = [
            row
            for row in candidates
            if {source["split"] for source in row["sources"]} == {"report", "selection"}
        ]

    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["score"]),
            row["report"] is None,
            row["selection"] is None,
            row["row_index"],
        ),
    )
    selected = ranked[:count]
    row_indices = [int(row["row_index"]) for row in selected]
    return {
        "row_indices": row_indices,
        "row_indices_csv": ",".join(str(index) for index in row_indices),
        "count": len(row_indices),
        "requested_count": count,
        "pool_size": pool_size,
        "avoided_row_indices": sorted(avoided),
        "top1_weight": top1_weight,
        "kld_weight": kld_weight,
        "shared_signal_only": shared_signal_only,
        "rows": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select GLM-4.5-Air teacher-cache rows for the next hard-row curriculum slice."
    )
    parser.add_argument("--report-eval-jsonl", required=True)
    parser.add_argument("--selection-eval-jsonl", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--pool-size", type=int, default=40)
    parser.add_argument("--avoid-row-indices", default="")
    parser.add_argument("--top1-weight", type=float, default=1.0)
    parser.add_argument("--kld-weight", type=float, default=0.5)
    parser.add_argument(
        "--shared-signal-only",
        action="store_true",
        help="Only select rows with low-top1/high-KLD rank signals in both report and selection evals.",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()
    try:
        avoided = _parse_row_indices(args.avoid_row_indices)
        result = select_target_rows(
            _read_eval_rows(Path(args.report_eval_jsonl)),
            _read_eval_rows(Path(args.selection_eval_jsonl)),
            count=args.count,
            pool_size=args.pool_size,
            avoided_row_indices=avoided,
            top1_weight=args.top1_weight,
            kld_weight=args.kld_weight,
            shared_signal_only=args.shared_signal_only,
        )
    except ValueError as error:
        parser.error(str(error))

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
