from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


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
            and isinstance(row.get("prompt"), str)
        ):
            rows.append(row)
    return rows


def _encoded_ids(tokenizer: Any, prompt: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]


def prepare_qwen_family_eval_teacher_metadata(
    *,
    family_policy: dict[str, Any],
    eval_prompt_pack: dict[str, Any],
    tokenizer_dir: str | Path,
    source_dir: str | Path,
    output_jsonl: str | Path,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    model_id = family_policy.get("model_id")
    revision = family_policy.get("revision")
    tokenizer_path = Path(tokenizer_dir)
    source_path = Path(source_dir)
    missing_requirements: list[str] = []
    if not source_path.exists():
        missing_requirements.append("qwen_source_dir")
    prompt_rows = _prompt_rows(eval_prompt_pack)
    if not prompt_rows:
        missing_requirements.append("qwen_eval_prompt_pack")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    metadata_rows: list[dict[str, Any]] = []
    row_errors: list[dict[str, Any]] = []
    for prompt_row in prompt_rows:
        prompt_id = str(prompt_row["prompt_id"])
        prompt = str(prompt_row["prompt"])
        encoded = _encoded_ids(tokenizer, prompt)
        expected_prefix = prompt_row.get("encoded_token_ids") or []
        encoded_matches_prompt_pack = encoded[: len(expected_prefix)] == [
            int(item) for item in expected_prefix
        ]
        if not encoded or not encoded_matches_prompt_pack:
            row_errors.append(
                {
                    "prompt_id": prompt_id,
                    "error": "prompt_tokenization_mismatch",
                }
            )
            continue
        metadata_rows.append(
            {
                "record_type": "qwen_teacher_cache_metadata",
                "prompt_id": prompt_id,
                "split": prompt_row["split"],
                "teacher_model_id": model_id,
                "teacher_revision": revision,
                "teacher_cache_metadata_verified": True,
                "teacher_cache_payload_present": False,
                "metadata_scope": "qwen_source_prompt_pack_tokenization",
                "source_dir": str(source_path),
                "tokenizer_dir": str(tokenizer_path),
                "token_count": len(encoded),
                "selected_token_id": int(encoded[-1]),
                "logit_mode": "source_next_token_logits_pending_metric_rows",
            }
        )

    if row_errors:
        missing_requirements.append("qwen_teacher_metadata_prompt_tokenization")
    _write_jsonl(output_jsonl, metadata_rows)
    split_counts = {"holdout": 0, "report": 0, "selection": 0}
    for row in metadata_rows:
        split = row.get("split")
        if split in split_counts:
            split_counts[str(split)] += 1
    return {
        "record_type": "qwen_family_eval_teacher_metadata_probe",
        "model_id": model_id,
        "revision": revision,
        "source_dir": str(source_path),
        "tokenizer_dir": str(tokenizer_path),
        "output_jsonl": str(Path(output_jsonl)),
        "metadata_row_count": len(metadata_rows),
        "metadata_counts_by_split": split_counts,
        "teacher_cache_payload_present": False,
        "row_error_count": len(row_errors),
        "row_errors": row_errors[:20],
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare verified Qwen eval teacher metadata rows for a prompt pack."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--eval-prompt-pack-json", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_jsonl = Path(args.output_dir) / "qwen_teacher_cache_metadata.jsonl"
    payload = prepare_qwen_family_eval_teacher_metadata(
        family_policy=_load_json(args.family_policy_json),
        eval_prompt_pack=_load_json(args.eval_prompt_pack_json),
        tokenizer_dir=args.tokenizer_dir,
        source_dir=args.source_dir,
        output_jsonl=output_jsonl,
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
