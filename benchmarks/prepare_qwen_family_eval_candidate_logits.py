from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import numpy as np

from keep.convert.stream_convert import load_safetensors_index
from keep.io.source_safetensors import read_indexed_safetensors_tensor_mlx


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


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


def _split_counts(prompt_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"holdout": 0, "report": 0, "selection": 0}
    for row in prompt_rows:
        split = row.get("split")
        if split in counts:
            counts[str(split)] += 1
    return counts


def _rows_by_prompt(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id not in result:
            result[prompt_id] = row
    return result


def _selected_token_id(prompt_row: dict[str, Any], position: str) -> int:
    token_ids = [int(token_id) for token_id in prompt_row["encoded_token_ids"]]
    if position == "first":
        return token_ids[0]
    if position == "last":
        return token_ids[-1]
    raise ValueError("selected_token_position must be 'first' or 'last'")


def _compact_logits(
    logits: np.ndarray,
    *,
    target_token_id: int,
    top_k: int,
) -> tuple[list[int], list[float]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    bounded_top_k = min(top_k, int(logits.shape[0]))
    top_indices = np.argsort(-logits, kind="stable")[:bounded_top_k]
    token_ids: list[int] = []
    for token_id in [int(item) for item in top_indices] + [target_token_id]:
        if token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids, [float(logits[token_id]) for token_id in token_ids]


def _teacher_logit_token_ids(teacher_row: dict[str, Any] | None) -> list[int] | None:
    if teacher_row is None:
        return None
    token_ids = teacher_row.get("logit_token_ids")
    logits = teacher_row.get("logits")
    if not isinstance(token_ids, list) or not token_ids:
        return None
    if not isinstance(logits, list) or len(logits) != len(token_ids):
        return None
    result: list[int] = []
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            return None
        result.append(token_id)
    return result


def _candidate_logits_for_tokens(
    *,
    embeddings: mx.array,
    lm_head: mx.array,
    target_token_id: int,
    logit_token_ids: list[int],
) -> list[float]:
    vocab_size = int(lm_head.shape[0])
    invalid = [token_id for token_id in logit_token_ids if token_id >= vocab_size]
    if invalid:
        raise ValueError(f"logit token ids outside lm_head vocab: {invalid[:5]}")
    if target_token_id < 0 or target_token_id >= int(embeddings.shape[0]):
        raise ValueError(
            f"target_token_id={target_token_id} outside embedding vocab {embeddings.shape[0]}"
        )
    hidden = embeddings[target_token_id].astype(mx.float32)
    rows = lm_head[mx.array(logit_token_ids, dtype=mx.int32)].astype(mx.float32)
    logits = mx.matmul(rows, hidden)
    mx.eval(logits)
    logits_np = np.array(logits)
    if not np.isfinite(logits_np).all():
        raise ValueError("candidate logits contain non-finite values")
    return [float(value) for value in logits_np]


def prepare_qwen_family_eval_candidate_logits(
    *,
    family_policy: dict[str, Any],
    eval_prompt_pack: dict[str, Any],
    output_jsonl: str | Path,
    source_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    teacher_logits_rows: Iterable[dict[str, Any]] = (),
    selected_token_position: str = "last",
    top_k: int = 32,
) -> dict[str, Any]:
    prompt_rows = _prompt_rows(eval_prompt_pack)
    missing_requirements: list[str] = []
    if not prompt_rows:
        missing_requirements.append("qwen_eval_prompt_pack")
    if source_dir is None or index_path is None:
        missing_requirements.extend(
            [
                "qwen_candidate_tokenizer_to_logits_runtime",
                "qwen_candidate_eval_logits",
            ]
        )
        _write_jsonl(output_jsonl, [])
        return {
            "record_type": "qwen_family_eval_candidate_logits_probe",
            "model_id": family_policy.get("model_id"),
            "revision": family_policy.get("revision"),
            "output_jsonl": str(Path(output_jsonl)),
            "required_prompt_counts": _split_counts(prompt_rows),
            "candidate_logit_row_count": 0,
            "candidate_logit_counts_by_split": {"holdout": 0, "report": 0, "selection": 0},
            "candidate_runtime_available": False,
            "selected_token_position": selected_token_position,
            "top_k": top_k,
            "row_error_count": 0,
            "row_errors": [],
            "missing_requirements": missing_requirements,
        }

    source_path = Path(source_dir)
    index = load_safetensors_index(index_path)
    embeddings = read_indexed_safetensors_tensor_mlx(
        source_path,
        index,
        "model.language_model.embed_tokens.weight",
    )
    lm_head = read_indexed_safetensors_tensor_mlx(source_path, index, "lm_head.weight")
    if int(embeddings.shape[1]) != int(lm_head.shape[1]):
        raise ValueError(
            f"embedding hidden dims {embeddings.shape[1]} do not match lm_head {lm_head.shape[1]}"
        )

    teachers_by_prompt = _rows_by_prompt(teacher_logits_rows)
    row_errors: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for prompt_row in prompt_rows:
        prompt_id = str(prompt_row["prompt_id"])
        try:
            target_token_id = _selected_token_id(prompt_row, selected_token_position)
            logit_token_ids = _teacher_logit_token_ids(teachers_by_prompt.get(prompt_id))
            if logit_token_ids is None:
                hidden = embeddings[target_token_id].astype(mx.float32)
                logits = mx.matmul(hidden, lm_head.astype(mx.float32).T)
                mx.eval(logits)
                logits_np = np.array(logits)
                if not np.isfinite(logits_np).all():
                    raise ValueError("candidate logits contain non-finite values")
                logit_token_ids, compact_logits = _compact_logits(
                    logits_np,
                    target_token_id=target_token_id,
                    top_k=top_k,
                )
            else:
                compact_logits = _candidate_logits_for_tokens(
                    embeddings=embeddings,
                    lm_head=lm_head,
                    target_token_id=target_token_id,
                    logit_token_ids=logit_token_ids,
                )
            output_rows.append(
                {
                    "record_type": "qwen_candidate_eval_logits",
                    "model_id": family_policy.get("model_id"),
                    "revision": family_policy.get("revision"),
                    "split": prompt_row["split"],
                    "prompt_id": prompt_id,
                    "target_token_id": target_token_id,
                    "selected_token_position": selected_token_position,
                    "logit_scope": "qwen_candidate_non_expert_embed_lm_head_compact",
                    "full_vocab_size": int(lm_head.shape[0]),
                    "logit_token_ids": logit_token_ids,
                    "logits": compact_logits,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                }
            )
        except Exception as exc:
            row_errors.append({"prompt_id": prompt_id, "error": str(exc)})

    if row_errors:
        missing_requirements.append("qwen_candidate_eval_logits")
    _write_jsonl(output_jsonl, output_rows)
    split_counts = {"holdout": 0, "report": 0, "selection": 0}
    for row in output_rows:
        split = row.get("split")
        if split in split_counts:
            split_counts[str(split)] += 1
    return {
        "record_type": "qwen_family_eval_candidate_logits_probe",
        "model_id": family_policy.get("model_id"),
        "revision": family_policy.get("revision"),
        "source_dir": str(source_path),
        "index_path": str(Path(index_path)),
        "output_jsonl": str(Path(output_jsonl)),
        "required_prompt_counts": _split_counts(prompt_rows),
        "candidate_logit_row_count": len(output_rows),
        "candidate_logit_counts_by_split": split_counts,
        "candidate_runtime_available": True,
        "candidate_runtime_scope": "qwen_candidate_non_expert_embed_lm_head",
        "selected_token_position": selected_token_position,
        "top_k": top_k,
        "teacher_aligned_prompt_count": len(teachers_by_prompt),
        "row_error_count": len(row_errors),
        "row_errors": row_errors[:20],
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Qwen candidate eval logits, or surface the missing runtime blocker."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--eval-prompt-pack-json", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--index-path")
    parser.add_argument("--teacher-logits-jsonl", action="append", default=[])
    parser.add_argument("--selected-token-position", choices=("first", "last"), default="last")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_jsonl = Path(args.output_dir) / "qwen_candidate_eval_logits.jsonl"
    payload = prepare_qwen_family_eval_candidate_logits(
        family_policy=_load_json(args.family_policy_json),
        eval_prompt_pack=_load_json(args.eval_prompt_pack_json),
        output_jsonl=output_jsonl,
        source_dir=args.source_dir,
        index_path=args.index_path,
        teacher_logits_rows=_read_jsonl_paths(args.teacher_logits_jsonl),
        selected_token_position=args.selected_token_position,
        top_k=args.top_k,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["missing_requirements"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
