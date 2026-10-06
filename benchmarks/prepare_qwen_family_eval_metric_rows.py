from __future__ import annotations

import argparse
import json
import math
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


def _rows_by_prompt(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id not in result:
            result[prompt_id] = row
    return result


def _float_logits(row: dict[str, Any]) -> list[float] | None:
    logits = row.get("logits")
    if not isinstance(logits, list) or not logits:
        return None
    result: list[float] = []
    for value in logits:
        if not isinstance(value, int | float) or isinstance(value, bool):
            return None
        float_value = float(value)
        if not math.isfinite(float_value):
            return None
        result.append(float_value)
    return result


def _logit_token_ids(row: dict[str, Any], logit_count: int) -> list[int] | None:
    token_ids = row.get("logit_token_ids")
    if token_ids is None:
        return list(range(logit_count))
    if not isinstance(token_ids, list) or len(token_ids) != logit_count:
        return None
    result: list[int] = []
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            return None
        result.append(token_id)
    return result


def _target_token_id(candidate: dict[str, Any], teacher: dict[str, Any]) -> int | None:
    value = teacher.get("target_token_id", candidate.get("target_token_id"))
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _log_softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    denom = sum(math.exp(value - max_logit) for value in logits)
    log_denom = max_logit + math.log(denom)
    return [value - log_denom for value in logits]


def _split_counts(prompt_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"holdout": 0, "report": 0, "selection": 0}
    for row in prompt_rows:
        split = row.get("split")
        if split in counts:
            counts[str(split)] += 1
    return counts


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


def _metric_row(
    *,
    prompt_row: dict[str, Any],
    candidate: dict[str, Any],
    teacher: dict[str, Any],
    model_id: str | None,
    revision: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    candidate_logits = _float_logits(candidate)
    teacher_logits = _float_logits(teacher)
    if candidate_logits is None:
        return None, "candidate_logits_missing_or_nonfinite"
    if teacher_logits is None:
        return None, "teacher_logits_missing_or_nonfinite"
    if len(candidate_logits) != len(teacher_logits):
        return None, "candidate_teacher_logit_shape_mismatch"
    candidate_token_ids = _logit_token_ids(candidate, len(candidate_logits))
    teacher_token_ids = _logit_token_ids(teacher, len(teacher_logits))
    if candidate_token_ids is None:
        return None, "candidate_logit_token_ids_invalid"
    if teacher_token_ids is None:
        return None, "teacher_logit_token_ids_invalid"
    if candidate_token_ids != teacher_token_ids:
        return None, "candidate_teacher_logit_token_ids_mismatch"
    target_token_id = _target_token_id(candidate, teacher)
    if target_token_id is None or target_token_id not in candidate_token_ids:
        return None, "target_token_id_missing_or_out_of_range"
    target_index = candidate_token_ids.index(target_token_id)

    candidate_log_probs = _log_softmax(candidate_logits)
    teacher_log_probs = _log_softmax(teacher_logits)
    nll = -candidate_log_probs[target_index]
    teacher_probs = [math.exp(value) for value in teacher_log_probs]
    kld = sum(
        prob * (teacher_log_prob - candidate_log_prob)
        for prob, teacher_log_prob, candidate_log_prob in zip(
            teacher_probs,
            teacher_log_probs,
            candidate_log_probs,
            strict=True,
        )
    )
    top1_match = float(
        max(range(len(candidate_logits)), key=candidate_logits.__getitem__)
        == max(range(len(teacher_logits)), key=teacher_logits.__getitem__)
    )
    return (
        {
            "record_type": "qwen_family_eval_metric_row",
            "model_id": model_id,
            "revision": revision,
            "split": prompt_row["split"],
            "prompt_id": prompt_row["prompt_id"],
            "target_token_id": target_token_id,
            "metric_scope": "candidate_vs_source_teacher_logits",
            "logit_count": len(candidate_logits),
            "logit_token_ids": candidate_token_ids,
            "nll": nll,
            "ppl": math.exp(nll),
            "mean_kld": kld,
            "top1_match": top1_match,
            "pageouts_delta": int(candidate.get("pageouts_delta", 0)),
            "swapouts_delta": int(candidate.get("swapouts_delta", 0)),
        },
        None,
    )


def prepare_qwen_family_eval_metric_rows(
    *,
    family_policy: dict[str, Any],
    eval_prompt_pack: dict[str, Any],
    candidate_logits_rows: Iterable[dict[str, Any]] = (),
    teacher_logits_rows: Iterable[dict[str, Any]] = (),
    output_jsonl: str | Path,
) -> dict[str, Any]:
    model_id = family_policy.get("model_id")
    revision = family_policy.get("revision")
    prompt_rows = _prompt_rows(eval_prompt_pack)
    candidates_by_prompt = _rows_by_prompt(candidate_logits_rows)
    teachers_by_prompt = _rows_by_prompt(teacher_logits_rows)

    emitted_rows: list[dict[str, Any]] = []
    row_errors: list[dict[str, Any]] = []
    for prompt_row in prompt_rows:
        prompt_id = str(prompt_row["prompt_id"])
        candidate = candidates_by_prompt.get(prompt_id)
        teacher = teachers_by_prompt.get(prompt_id)
        if candidate is None or teacher is None:
            continue
        metric, error = _metric_row(
            prompt_row=prompt_row,
            candidate=candidate,
            teacher=teacher,
            model_id=model_id,
            revision=revision,
        )
        if metric is None:
            row_errors.append({"prompt_id": prompt_id, "error": error})
        else:
            emitted_rows.append(metric)

    _write_jsonl(output_jsonl, emitted_rows)
    candidate_prompt_ids = set(candidates_by_prompt)
    teacher_prompt_ids = set(teachers_by_prompt)
    emitted_prompt_ids = {str(row["prompt_id"]) for row in emitted_rows}
    missing_candidate_counts = _missing_counts_by_split(prompt_rows, candidate_prompt_ids)
    missing_teacher_counts = _missing_counts_by_split(prompt_rows, teacher_prompt_ids)
    missing_metric_counts = _missing_counts_by_split(prompt_rows, emitted_prompt_ids)
    missing_requirements: list[str] = []
    if not prompt_rows:
        missing_requirements.append("qwen_eval_prompt_pack")
    if _nonzero_counts(missing_candidate_counts):
        missing_requirements.append("qwen_candidate_eval_logits")
    if _nonzero_counts(missing_teacher_counts):
        missing_requirements.append("qwen_teacher_eval_logits")
    if _nonzero_counts(missing_metric_counts) or row_errors:
        missing_requirements.append("qwen_eval_metric_rows")
    return {
        "record_type": "qwen_family_eval_metric_row_probe",
        "model_id": model_id,
        "revision": revision,
        "output_jsonl": str(Path(output_jsonl)),
        "required_prompt_counts": _split_counts(prompt_rows),
        "metric_row_count": len(emitted_rows),
        "missing_candidate_logit_counts": missing_candidate_counts,
        "missing_teacher_logit_counts": missing_teacher_counts,
        "missing_metric_counts": missing_metric_counts,
        "row_error_count": len(row_errors),
        "row_errors": row_errors[:20],
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Qwen family eval metric rows from candidate and teacher logits."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--eval-prompt-pack-json", required=True)
    parser.add_argument("--candidate-logits-jsonl", action="append", default=[])
    parser.add_argument("--teacher-logits-jsonl", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_jsonl = Path(args.output_dir) / "qwen_family_eval_metric_rows.jsonl"
    payload = prepare_qwen_family_eval_metric_rows(
        family_policy=_load_json(args.family_policy_json),
        eval_prompt_pack=_load_json(args.eval_prompt_pack_json),
        candidate_logits_rows=_read_jsonl_paths(args.candidate_logits_jsonl),
        teacher_logits_rows=_read_jsonl_paths(args.teacher_logits_jsonl),
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
