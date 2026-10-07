from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _row_memory_clean(row: dict[str, Any]) -> bool:
    return row.get("pageouts_delta") == 0 and row.get("swapouts_delta") == 0


def _row_metrics_finite(row: dict[str, Any]) -> bool:
    top1 = row.get("top1_match", row.get("top1_agreement"))
    return (
        _finite_number(row.get("nll"))
        and _finite_number(row.get("ppl"))
        and _finite_number(row.get("mean_kld"))
        and _finite_number(top1)
        and 0.0 <= float(top1) <= 1.0
    )


def _row_teacher_cache_metadata_verified(
    row: dict[str, Any],
    *,
    model_id: str | None,
    revision: str | None,
) -> bool:
    return (
        row.get("teacher_cache_metadata_verified") is True
        and row.get("teacher_model_id") == model_id
        and row.get("teacher_revision") == revision
    )


def _probe_missing_requirements(
    probe: dict[str, Any] | None,
    *,
    record_type: str,
) -> list[str]:
    if probe is None or probe.get("record_type") != record_type:
        return []
    missing = probe.get("missing_requirements")
    if not isinstance(missing, list | tuple):
        return []
    return [item for item in missing if isinstance(item, str)]


def _append_missing_once(missing_requirements: list[str], requirement: str) -> None:
    if requirement not in missing_requirements:
        missing_requirements.append(requirement)


def _missing_split_name(split: str) -> str:
    return f"qwen_{split}_eval_jsonl"


def _missing_prompt_rows_name(split: str) -> str:
    return f"qwen_{split}_eval_prompt_rows"


def _prompt_ids_by_split(
    eval_prompt_pack: dict[str, Any] | None,
    *,
    required_splits: tuple[str, ...],
) -> dict[str, set[str]]:
    if eval_prompt_pack is None:
        return {}
    if (
        eval_prompt_pack.get("record_type") != "qwen_family_eval_prompt_probe"
        or eval_prompt_pack.get("prompt_pack_ready") is not True
        or eval_prompt_pack.get("missing_requirements")
    ):
        return {}
    prompt_ids: dict[str, set[str]] = {split: set() for split in required_splits}
    for row in eval_prompt_pack.get("prompt_rows") or []:
        if not isinstance(row, dict):
            continue
        split = row.get("split")
        prompt_id = row.get("prompt_id")
        if split in prompt_ids and isinstance(prompt_id, str) and prompt_id:
            prompt_ids[str(split)].add(prompt_id)
    return prompt_ids


def check_qwen_family_eval_gate(
    *,
    family_policy: dict[str, Any],
    eval_jsonl_paths: Iterable[str | Path] = (),
    eval_prompt_pack: dict[str, Any] | None = None,
    eval_row_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_gate = family_policy.get("eval_gate") or {}
    required_splits = tuple(eval_gate.get("required_splits") or ())
    minimum_clean_rows = int(eval_gate.get("minimum_clean_rows_per_split") or 0)
    model_id = family_policy.get("model_id")
    revision = family_policy.get("revision")
    rows: list[dict[str, Any]] = []
    input_paths = [str(Path(path)) for path in eval_jsonl_paths]
    for path in eval_jsonl_paths:
        rows.extend(_read_jsonl(path))

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in required_splits}
    required_prompt_ids = _prompt_ids_by_split(
        eval_prompt_pack,
        required_splits=required_splits,
    )
    row_errors: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        split = row.get("split")
        if split in split_rows:
            split_rows[str(split)].append(row)
        if not _row_memory_clean(row):
            row_errors.append({"row_index": row_index, "error": "row_memory_not_clean"})
        if not _row_metrics_finite(row):
            row_errors.append({"row_index": row_index, "error": "row_metrics_not_finite"})
        if not _row_teacher_cache_metadata_verified(
            row,
            model_id=model_id,
            revision=revision,
        ):
            row_errors.append(
                {"row_index": row_index, "error": "teacher_cache_metadata_not_verified"}
            )
        if required_prompt_ids and split in required_prompt_ids:
            prompt_id = row.get("prompt_id")
            if prompt_id not in required_prompt_ids[str(split)]:
                row_errors.append({"row_index": row_index, "error": "prompt_id_not_in_prompt_pack"})

    clean_split_counts = {
        split: sum(
            1
            for row in split_rows[split]
            if _row_memory_clean(row)
            and _row_metrics_finite(row)
            and _row_teacher_cache_metadata_verified(
                row,
                model_id=model_id,
                revision=revision,
            )
        )
        for split in required_splits
    }
    missing_requirements = [
        _missing_split_name(split)
        for split in required_splits
        if clean_split_counts[split] < minimum_clean_rows
    ]
    clean_prompt_ids_by_split = {
        split: {
            str(row.get("prompt_id"))
            for row in split_rows[split]
            if isinstance(row.get("prompt_id"), str)
            and _row_memory_clean(row)
            and _row_metrics_finite(row)
            and _row_teacher_cache_metadata_verified(
                row,
                model_id=model_id,
                revision=revision,
            )
            and (
                not required_prompt_ids
                or row.get("prompt_id") in required_prompt_ids.get(split, set())
            )
        }
        for split in required_splits
    }
    missing_prompt_ids_by_split = {
        split: sorted(required_prompt_ids[split] - clean_prompt_ids_by_split[split])
        for split in required_prompt_ids
    }
    for split, missing_ids in missing_prompt_ids_by_split.items():
        if missing_ids:
            missing_requirements.append(_missing_prompt_rows_name(split))
    if row_errors:
        missing_requirements.append("qwen_eval_rows_clean_finite_and_matching_teacher")
    clean_rows = [
        row
        for split in required_splits
        for row in split_rows[split]
        if _row_memory_clean(row)
        and _row_metrics_finite(row)
        and _row_teacher_cache_metadata_verified(
            row,
            model_id=model_id,
            revision=revision,
        )
    ]
    if any(
        error.get("error") == "teacher_cache_metadata_not_verified"
        for error in row_errors
    ):
        _append_missing_once(
            missing_requirements,
            "qwen_eval_teacher_cache_metadata",
        )
    eval_row_probe_missing_requirements = _probe_missing_requirements(
        eval_row_probe,
        record_type="qwen_family_eval_row_probe",
    )
    for requirement in eval_row_probe_missing_requirements:
        _append_missing_once(missing_requirements, requirement)
    top1_values = [float(row.get("top1_match", row.get("top1_agreement"))) for row in clean_rows]
    family_eval_gate_pass = not missing_requirements
    return {
        "record_type": "qwen_family_eval_gate_check",
        "model_id": model_id,
        "input_paths": input_paths,
        "required_splits": list(required_splits),
        "minimum_clean_rows_per_split": minimum_clean_rows,
        "prompt_pack_required": bool(required_prompt_ids),
        "required_prompt_counts": {
            split: len(required_prompt_ids.get(split, set()))
            for split in required_splits
        },
        "missing_prompt_ids_by_split": missing_prompt_ids_by_split,
        "split_counts": clean_split_counts,
        "row_error_count": len(row_errors),
        "row_errors": row_errors[:20],
        "eval_row_probe_missing_requirements": eval_row_probe_missing_requirements,
        "metric_summary": {
            "clean_row_count": len(clean_rows),
            "top1_mean": (sum(top1_values) / len(top1_values)) if top1_values else None,
        },
        "family_eval_gate_pass": family_eval_gate_pass,
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Qwen family eval gate evidence.")
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--eval-prompt-pack-json")
    parser.add_argument("--eval-row-probe-json")
    parser.add_argument("--eval-jsonl", action="append", default=[])
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = check_qwen_family_eval_gate(
        family_policy=_load_json(args.family_policy_json),
        eval_jsonl_paths=args.eval_jsonl,
        eval_prompt_pack=(
            _load_json(args.eval_prompt_pack_json)
            if args.eval_prompt_pack_json is not None
            else None
        ),
        eval_row_probe=(
            _load_json(args.eval_row_probe_json)
            if args.eval_row_probe_json is not None
            else None
        ),
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
