from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _read_jsonl_paths(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(row)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _split_counts(prompt_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"holdout": 0, "report": 0, "selection": 0}
    for row in prompt_rows:
        split = row.get("split")
        if split in counts:
            counts[str(split)] += 1
    return counts


def _prompt_rows(eval_prompt_pack: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        eval_prompt_pack.get("record_type") != "qwen_family_eval_prompt_probe"
        or eval_prompt_pack.get("prompt_pack_ready") is not True
        or eval_prompt_pack.get("missing_requirements")
    ):
        return []
    return [
        row
        for row in eval_prompt_pack.get("prompt_rows") or []
        if isinstance(row, dict)
        and isinstance(row.get("prompt_id"), str)
        and row.get("split") in {"holdout", "report", "selection"}
    ]


def _metric_row_by_prompt(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id not in result:
            result[prompt_id] = row
    return result


def _teacher_metadata_by_prompt(
    rows: Iterable[dict[str, Any]],
    *,
    model_id: str | None,
    revision: str | None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id in result:
            continue
        if (
            row.get("teacher_cache_metadata_verified") is True
            and row.get("teacher_model_id") == model_id
            and row.get("teacher_revision") == revision
        ):
            result[prompt_id] = row
    return result


def _required_metric_fields_present(row: dict[str, Any]) -> bool:
    return all(
        field in row
        for field in (
            "nll",
            "ppl",
            "mean_kld",
            "top1_match",
            "pageouts_delta",
            "swapouts_delta",
        )
    )


def _missing_counts_by_split(
    prompt_rows: Iterable[dict[str, Any]],
    available_prompt_ids: set[str],
) -> dict[str, int]:
    counts = {"holdout": 0, "report": 0, "selection": 0}
    for row in prompt_rows:
        split = row.get("split")
        prompt_id = row.get("prompt_id")
        if split in counts and prompt_id not in available_prompt_ids:
            counts[str(split)] += 1
    return counts


def _nonzero_counts(counts: dict[str, int]) -> bool:
    return any(value > 0 for value in counts.values())


def prepare_qwen_family_eval_rows(
    *,
    family_policy: dict[str, Any],
    eval_prompt_pack: dict[str, Any],
    eval_metric_rows: Iterable[dict[str, Any]] = (),
    teacher_cache_metadata_rows: Iterable[dict[str, Any]] = (),
    output_jsonl: str | Path,
) -> dict[str, Any]:
    model_id = family_policy.get("model_id")
    revision = family_policy.get("revision")
    prompt_rows = _prompt_rows(eval_prompt_pack)
    metrics_by_prompt = _metric_row_by_prompt(eval_metric_rows)
    metadata_by_prompt = _teacher_metadata_by_prompt(
        teacher_cache_metadata_rows,
        model_id=model_id,
        revision=revision,
    )

    emitted_rows: list[dict[str, Any]] = []
    for prompt_row in prompt_rows:
        prompt_id = str(prompt_row["prompt_id"])
        metric = metrics_by_prompt.get(prompt_id)
        metadata = metadata_by_prompt.get(prompt_id)
        if metric is None or metadata is None or not _required_metric_fields_present(metric):
            continue
        emitted_rows.append(
            {
                "record_type": "qwen_family_eval_row",
                "model_id": model_id,
                "revision": revision,
                "split": prompt_row["split"],
                "prompt_id": prompt_id,
                "teacher_model_id": metadata["teacher_model_id"],
                "teacher_revision": metadata["teacher_revision"],
                "teacher_cache_metadata_verified": True,
                "nll": metric["nll"],
                "ppl": metric["ppl"],
                "mean_kld": metric["mean_kld"],
                "top1_match": metric["top1_match"],
                "pageouts_delta": metric["pageouts_delta"],
                "swapouts_delta": metric["swapouts_delta"],
            }
        )

    _write_jsonl(output_jsonl, emitted_rows)
    metric_prompt_ids = {
        prompt_id
        for prompt_id, row in metrics_by_prompt.items()
        if _required_metric_fields_present(row)
    }
    metadata_prompt_ids = set(metadata_by_prompt)
    missing_metric_counts = _missing_counts_by_split(prompt_rows, metric_prompt_ids)
    missing_metadata_counts = _missing_counts_by_split(prompt_rows, metadata_prompt_ids)
    missing_requirements: list[str] = []
    if not prompt_rows:
        missing_requirements.append("qwen_eval_prompt_pack")
    if _nonzero_counts(missing_metric_counts):
        missing_requirements.append("qwen_eval_metric_rows")
    if _nonzero_counts(missing_metadata_counts):
        missing_requirements.append("qwen_eval_teacher_cache_metadata")

    return {
        "record_type": "qwen_family_eval_row_probe",
        "model_id": model_id,
        "revision": revision,
        "output_jsonl": str(Path(output_jsonl)),
        "required_prompt_counts": _split_counts(prompt_rows),
        "eval_row_count": len(emitted_rows),
        "missing_metric_counts": missing_metric_counts,
        "missing_teacher_metadata_counts": missing_metadata_counts,
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Qwen family eval rows from metrics plus teacher metadata."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--eval-prompt-pack-json", required=True)
    parser.add_argument("--eval-metric-jsonl", action="append", default=[])
    parser.add_argument("--teacher-cache-metadata-jsonl", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_jsonl = Path(args.output_dir) / "qwen_family_eval_rows.jsonl"
    payload = prepare_qwen_family_eval_rows(
        family_policy=_load_json(args.family_policy_json),
        eval_prompt_pack=_load_json(args.eval_prompt_pack_json),
        eval_metric_rows=_read_jsonl_paths(args.eval_metric_jsonl),
        teacher_cache_metadata_rows=_read_jsonl_paths(args.teacher_cache_metadata_jsonl),
        output_jsonl=output_jsonl,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
