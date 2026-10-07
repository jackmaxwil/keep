from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from ramp.benchmark.glm45_air import append_jsonl
from ramp.benchmark.metrics import collect_metric_snapshot, collect_vm_stat_counts
from keep.io.logit_bias import load_logit_bias_sidecar
from keep.quality.teacher_cache import (
    evaluate_teacher_cache_row,
    read_teacher_cache_rows,
    summarize_teacher_cache_records,
    validate_teacher_cache_metadata,
)


def _parse_token_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        token_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("--watch-token-ids must be a comma-separated list of integers") from error
    if not token_ids:
        raise ValueError("--watch-token-ids must include at least one token")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("--watch-token-ids values must be zero or greater")
    return token_ids


def _read_student_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"student logits row {line_number} is not an object")
        if payload.get("record_type") == "air_vq_teacher_cache_eval_summary":
            continue
        if "vq_logit_shard" not in payload:
            continue
        rows.append(payload)
    if not rows:
        raise ValueError(f"{path} contained no saved VQ-logit rows")
    return rows


def _read_student_summary(path: str | Path) -> dict[str, Any] | None:
    summary: dict[str, Any] | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and payload.get("record_type") == "air_vq_teacher_cache_eval_summary":
            summary = payload
    return summary


def _memory_clean_bool(row: dict[str, Any]) -> bool:
    pageouts_delta = row.get("pageouts_delta")
    swapouts_delta = row.get("swapouts_delta")
    if pageouts_delta is None or swapouts_delta is None:
        return False
    return int(pageouts_delta) == 0 and int(swapouts_delta) == 0


def _student_vq_logits_memory_summary(
    rows: list[dict[str, Any]],
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    clean_rows = sum(1 for row in rows if _memory_clean_bool(row))
    row_memory_clean = clean_rows == len(rows)
    process_memory_clean = summary.get("process_memory_clean") if summary is not None else None
    acceptance_memory_clean = summary.get("acceptance_memory_clean") if summary is not None else None
    summary_acceptance_clean = bool(acceptance_memory_clean) if acceptance_memory_clean is not None else False
    return {
        "source_student_row_memory_clean": row_memory_clean,
        "source_student_clean_record_count": clean_rows,
        "source_student_dirty_record_count": len(rows) - clean_rows,
        "source_student_process_memory_clean": process_memory_clean,
        "source_student_acceptance_memory_clean": acceptance_memory_clean,
        "source_vq_logits_memory_clean": row_memory_clean and summary_acceptance_clean,
    }


def _load_saved_vq_logits(row: dict[str, Any], *, logits_root: str | Path) -> mx.array:
    shard = row.get("vq_logit_shard")
    tensor = row.get("vq_logit_tensor", "vq_logits")
    if not isinstance(shard, str) or not shard:
        raise ValueError("student row needs vq_logit_shard")
    if not isinstance(tensor, str) or not tensor:
        raise ValueError("student row needs vq_logit_tensor")
    shard_path = Path(shard)
    if shard_path.is_absolute():
        raise ValueError(f"vq_logit_shard must be relative to logits root, got {shard!r}")
    root = Path(logits_root).resolve(strict=False)
    resolved = (root / shard_path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"vq_logit_shard escapes logits root: {shard!r}") from error
    arrays = mx.load(str(resolved))
    logits = arrays.get(tensor)
    if logits is None:
        raise ValueError(f"{resolved} does not contain tensor {tensor!r}")
    if logits.ndim != 2:
        raise ValueError(f"saved VQ logits must have shape [positions, vocab], got {logits.shape}")
    return logits


def _apply_logit_bias_sidecar(
    logits: mx.array,
    *,
    positions: list[int],
    artifact_dir: str | Path | None,
) -> tuple[mx.array, dict[str, Any] | None]:
    if artifact_dir is None:
        return logits, None
    sidecar = load_logit_bias_sidecar(artifact_dir, vocab_size=int(logits.shape[1]))
    if sidecar is None:
        return logits, None
    ids = [int(token_id) for token_id in sidecar.token_ids.tolist()]
    biases = [float(value) for value in sidecar.biases.tolist()]
    if sidecar.position_indices is None:
        row_indices = list(range(int(logits.shape[0])))
    else:
        scoped = set(int(position) for position in sidecar.position_indices)
        row_indices = [index for index, position in enumerate(positions) if int(position) in scoped]
    if row_indices:
        row_index_array = mx.array(row_indices, dtype=mx.int32)
        for token_id, bias in zip(ids, biases, strict=True):
            logits = logits.at[row_index_array, token_id].add(bias)
    return logits, {
        "artifact_dir": str(artifact_dir),
        "token_ids": ids,
        "biases": biases,
        "position_indices": list(sidecar.position_indices) if sidecar.position_indices is not None else None,
        "applied_selected_position_count": len(row_indices),
    }


def _teacher_row_for_student(
    teacher_rows: list[dict[str, Any]],
    student_row: dict[str, Any],
    fallback_index: int,
) -> tuple[int, dict[str, Any]]:
    row_index = int(student_row.get("row_index", fallback_index))
    if row_index < 0 or row_index >= len(teacher_rows):
        raise IndexError(f"student row_index {row_index} is out of range for {len(teacher_rows)} teacher rows")
    teacher_row = teacher_rows[row_index]
    teacher_prompt = teacher_row.get("prompt_id")
    student_prompt = student_row.get("prompt_id")
    if teacher_prompt is not None and student_prompt is not None and teacher_prompt != student_prompt:
        raise ValueError(
            f"student row_index {row_index} prompt_id {student_prompt!r} does not match teacher {teacher_prompt!r}"
        )
    return row_index, teacher_row


def replay_saved_vq_logits(
    *,
    teacher_jsonl: str | Path,
    student_jsonl: str | Path,
    append_jsonl_path: str | Path,
    cache_root: str | Path | None = None,
    student_logits_root: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    watch_token_ids: tuple[int, ...] | None = None,
    min_top_k: int = 128,
    check_values: bool = False,
) -> dict[str, Any]:
    teacher_path = Path(teacher_jsonl)
    teacher_cache_root = Path(cache_root) if cache_root is not None else teacher_path.parent
    logits_root = Path(student_logits_root) if student_logits_root is not None else Path(student_jsonl).parent
    teacher_rows = read_teacher_cache_rows(teacher_path)
    student_rows = _read_student_rows(student_jsonl)
    student_summary = _read_student_summary(student_jsonl)
    source_memory_summary = _student_vq_logits_memory_summary(student_rows, student_summary)
    row_indices = tuple(
        int(row.get("row_index", index))
        for index, row in enumerate(student_rows)
    )
    validation = validate_teacher_cache_metadata(
        teacher_path,
        cache_root=teacher_cache_root,
        min_top_k=min_top_k,
        check_values=check_values,
        row_indices=row_indices,
    )
    if not validation["ok"]:
        raise ValueError(json.dumps(validation, indent=2, sort_keys=True))

    process_vm_start = collect_vm_stat_counts()
    records: list[dict[str, Any]] = []
    for fallback_index, student_row in enumerate(student_rows):
        row_index, teacher_row = _teacher_row_for_student(teacher_rows, student_row, fallback_index)
        before_vm = collect_vm_stat_counts()
        logits = _load_saved_vq_logits(student_row, logits_root=logits_root)
        positions = [int(position) for position in teacher_row.get("positions", list(range(int(logits.shape[0]))))]
        logits, logit_bias_report = _apply_logit_bias_sidecar(
            logits,
            positions=positions,
            artifact_dir=artifact_dir,
        )
        mx.eval(logits)
        metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
        record = evaluate_teacher_cache_row(
            teacher_row,
            vq_logits=logits,
            cache_root=teacher_cache_root,
            watch_token_ids=watch_token_ids,
            extra={
                "engine": "saved_vq_logits_replay",
                "prefill_engine": "saved_vq_logits_replay",
                "artifact_dir": str(artifact_dir) if artifact_dir is not None else student_row.get("artifact_dir"),
                "row_index": row_index,
                "source_student_jsonl": str(student_jsonl),
                "student_vq_logit_shard": student_row.get("vq_logit_shard"),
                "student_vq_logit_tensor": student_row.get("vq_logit_tensor", "vq_logits"),
                "source_student_row_memory_clean": _memory_clean_bool(student_row),
                "saved_vq_logits_replay": True,
                "logit_bias_sidecar": logit_bias_report,
                **metrics,
            },
        )
        records.append(record)
        append_jsonl(append_jsonl_path, record)

    summary = summarize_teacher_cache_records(records)
    process_metrics = collect_metric_snapshot(previous_vm_stat_counts=process_vm_start)
    process_pageouts_delta = int(process_metrics.get("pageouts_delta") or 0)
    process_swapouts_delta = int(process_metrics.get("swapouts_delta") or 0)
    summary.update(
        {
            "teacher_jsonl": str(teacher_jsonl),
            "student_jsonl": str(student_jsonl),
            "cache_root": str(teacher_cache_root),
            "student_logits_root": str(logits_root),
            "append_jsonl": str(append_jsonl_path),
            "engine": "saved_vq_logits_replay",
            "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
            **source_memory_summary,
            "row_memory_clean": bool(summary.get("all_memory_clean")),
            "process_memory_clean": process_pageouts_delta == 0 and process_swapouts_delta == 0,
            "process_pageouts_delta": process_pageouts_delta,
            "process_swapouts_delta": process_swapouts_delta,
            "acceptance_memory_clean": bool(summary.get("all_memory_clean"))
            and process_pageouts_delta == 0
            and process_swapouts_delta == 0
            and bool(source_memory_summary["source_vq_logits_memory_clean"]),
        }
    )
    append_jsonl(append_jsonl_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay saved GLM-4.5-Air selected VQ logits against a teacher cache."
    )
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--student-jsonl", required=True)
    parser.add_argument("--append-jsonl", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--student-logits-root")
    parser.add_argument("--artifact-dir", help="Optional artifact with a sparse final-logit bias sidecar to apply.")
    parser.add_argument("--watch-token-ids")
    parser.add_argument("--min-top-k", type=int, default=128)
    parser.add_argument("--check-values", action="store_true")
    args = parser.parse_args()
    if args.min_top_k <= 0:
        parser.error("--min-top-k must be positive")
    try:
        watch_token_ids = _parse_token_ids(args.watch_token_ids)
    except ValueError as error:
        parser.error(str(error))
    summary = replay_saved_vq_logits(
        teacher_jsonl=args.teacher_jsonl,
        student_jsonl=args.student_jsonl,
        append_jsonl_path=args.append_jsonl,
        cache_root=args.cache_root,
        student_logits_root=args.student_logits_root,
        artifact_dir=args.artifact_dir,
        watch_token_ids=watch_token_ids,
        min_top_k=args.min_top_k,
        check_values=args.check_values,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
