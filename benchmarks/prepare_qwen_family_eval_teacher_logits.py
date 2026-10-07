from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.io.source_safetensors import read_indexed_safetensors_tensor_mlx


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


def _prompt_rows(eval_prompt_pack: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        eval_prompt_pack.get("record_type") != "qwen_family_eval_prompt_probe"
        or eval_prompt_pack.get("prompt_pack_ready") is not True
        or eval_prompt_pack.get("missing_requirements")
    ):
        return []
    rows: list[dict[str, Any]] = []
    for row in eval_prompt_pack.get("prompt_rows") or []:
        if (
            isinstance(row, dict)
            and isinstance(row.get("prompt_id"), str)
            and row.get("split") in {"holdout", "report", "selection"}
            and isinstance(row.get("encoded_token_ids"), list)
            and row.get("encoded_token_ids")
        ):
            rows.append(row)
    return rows


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


def prepare_qwen_family_eval_teacher_logits(
    *,
    family_policy: dict[str, Any],
    eval_prompt_pack: dict[str, Any],
    source_dir: str | Path,
    index_path: str | Path,
    output_jsonl: str | Path,
    selected_token_position: str = "last",
    top_k: int = 32,
) -> dict[str, Any]:
    source_path = Path(source_dir)
    index = load_safetensors_index(index_path)
    prompt_rows = _prompt_rows(eval_prompt_pack)
    missing_requirements: list[str] = []
    if not prompt_rows:
        missing_requirements.append("qwen_eval_prompt_pack")

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
    selected_token_ids = [
        _selected_token_id(row, selected_token_position) for row in prompt_rows
    ]
    hidden = embeddings[mx.array(selected_token_ids, dtype=mx.int32)].astype(mx.float32)
    logits = mx.matmul(hidden, lm_head.astype(mx.float32).T)
    mx.eval(logits)
    logits_np = np.array(logits)
    row_errors: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for row_index, prompt_row in enumerate(prompt_rows):
        prompt_id = str(prompt_row["prompt_id"])
        row_logits = logits_np[row_index]
        if not np.isfinite(row_logits).all():
            row_errors.append({"prompt_id": prompt_id, "error": "teacher_logits_nonfinite"})
            continue
        target_token_id = int(selected_token_ids[row_index])
        logit_token_ids, compact = _compact_logits(
            row_logits,
            target_token_id=target_token_id,
            top_k=top_k,
        )
        output_rows.append(
            {
                "record_type": "qwen_teacher_eval_logits",
                "model_id": family_policy.get("model_id"),
                "revision": family_policy.get("revision"),
                "split": prompt_row["split"],
                "prompt_id": prompt_id,
                "target_token_id": target_token_id,
                "selected_token_position": selected_token_position,
                "logit_scope": "qwen_source_embed_lm_head_compact",
                "full_vocab_size": int(row_logits.shape[0]),
                "logit_token_ids": logit_token_ids,
                "logits": compact,
            }
        )

    if row_errors:
        missing_requirements.append("qwen_teacher_eval_logits")
    _write_jsonl(output_jsonl, output_rows)
    split_counts = {"holdout": 0, "report": 0, "selection": 0}
    for row in output_rows:
        split = row.get("split")
        if split in split_counts:
            split_counts[str(split)] += 1
    return {
        "record_type": "qwen_family_eval_teacher_logits_probe",
        "model_id": family_policy.get("model_id"),
        "revision": family_policy.get("revision"),
        "source_dir": str(source_path),
        "index_path": str(Path(index_path)),
        "output_jsonl": str(Path(output_jsonl)),
        "teacher_logit_row_count": len(output_rows),
        "teacher_logit_counts_by_split": split_counts,
        "selected_token_position": selected_token_position,
        "top_k": top_k,
        "row_error_count": len(row_errors),
        "row_errors": row_errors[:20],
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare compact Qwen source-teacher eval logits for prompt rows."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--eval-prompt-pack-json", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--selected-token-position", choices=("first", "last"), default="last")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_jsonl = Path(args.output_dir) / "qwen_teacher_eval_logits.jsonl"
    payload = prepare_qwen_family_eval_teacher_logits(
        family_policy=_load_json(args.family_policy_json),
        eval_prompt_pack=_load_json(args.eval_prompt_pack_json),
        source_dir=args.source_dir,
        index_path=args.index_path,
        output_jsonl=output_jsonl,
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
