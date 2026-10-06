from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer


DEFAULT_QWEN_EVAL_PROMPTS: dict[str, tuple[str, ...]] = {
    "report": (
        "explain matrix multiplication in simple terms",
        "write a python function to add two numbers",
        "solve twelve plus thirty four",
        "summarize why routers choose experts",
        "name three planets in order",
        "translate hello world into spanish",
        "classify this sentence as positive",
        "complete the pattern two four six eight",
        "state one safety rule for batteries",
        "describe how rain forms",
        "answer what is the capital of france",
        "write a short sql select query",
        "explain what latency means",
        "give one example of recursion",
        "compare apples and oranges briefly",
        "rewrite this sentence more formally",
        "explain why caching can reduce repeated work",
        "write a tiny javascript array example",
        "solve twenty one plus twenty two",
        "name one reason benchmarks need controls",
        "summarize a simple bug report",
        "describe one use for a tokenizer",
    ),
    "selection": (
        "solve seven times six",
        "write pseudocode for binary search",
        "summarize the role of attention",
        "name two properties of water",
        "translate good morning into french",
        "explain why tests catch regressions",
        "complete the sequence one three five seven",
        "classify this as math or language",
        "describe a cache in computing",
        "answer who wrote hamlet",
        "write a json object with name age",
        "explain the difference between cpu and gpu",
        "give a concise meeting summary",
        "state one reason compression helps",
        "solve one hundred minus forty",
        "write a polite reminder message",
        "explain one difference between training and inference",
        "write a shell command that prints hello",
        "solve eighteen divided by three",
        "name two kinds of software tests",
        "summarize a pull request change",
        "describe why reproducibility matters",
    ),
    "holdout": (
        "explain gradient descent briefly",
        "write a bash command to list files",
        "solve nine squared",
        "summarize why memory bandwidth matters",
        "name two primary colors",
        "translate thank you into german",
        "complete the analogy hot cold up down",
        "classify the tone as neutral",
        "describe a database index",
        "answer what gas do plants absorb",
        "write a regular expression for digits",
        "explain what top k routing means",
        "give one example of a loop",
        "state one benefit of documentation",
        "solve fifty divided by five",
        "rewrite a sentence in active voice",
        "explain one cause of numerical overflow",
        "write a small dictionary literal in python",
        "solve thirteen plus twenty nine",
        "name one property of a reliable benchmark",
        "summarize a failed experiment briefly",
        "describe a model checkpoint",
    ),
}


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _encoded_ids(tokenizer: Any, prompt: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]


def build_qwen_family_eval_prompt_probe(
    tokenizer: Any,
    *,
    model_id: str,
    revision: str,
    prompts_by_split: Mapping[str, Sequence[str]] | None = None,
    min_prompts_per_split: int,
    min_tokens_per_prompt: int,
    expected_vocab_size: int | None = None,
) -> dict[str, object]:
    if min_prompts_per_split < 0:
        raise ValueError("min_prompts_per_split must be non-negative")
    if min_tokens_per_prompt < 0:
        raise ValueError("min_tokens_per_prompt must be non-negative")
    split_order = ("report", "selection", "holdout")
    prompt_source = prompts_by_split or DEFAULT_QWEN_EVAL_PROMPTS
    rows: list[dict[str, object]] = []
    missing_requirements: list[str] = []
    split_counts: dict[str, int] = {}
    tokenizer_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    tokenizer_length = int(len(tokenizer)) if hasattr(tokenizer, "__len__") else tokenizer_vocab_size

    for split in split_order:
        prompts = tuple(prompt_source.get(split, ()))
        split_counts[split] = len(prompts)
        if len(prompts) < min_prompts_per_split:
            missing_requirements.append(f"qwen_{split}_eval_prompts")
        for row_index, prompt in enumerate(prompts):
            token_ids = _encoded_ids(tokenizer, prompt)
            max_token_id = max(token_ids) if token_ids else None
            row_requirements: list[str] = []
            if len(token_ids) < min_tokens_per_prompt:
                row_requirements.append("prompt_encodes_enough_tokens")
            if expected_vocab_size is not None and max_token_id is not None and max_token_id >= expected_vocab_size:
                row_requirements.append("token_ids_within_expected_vocab")
            if row_requirements and "qwen_eval_prompt_rows_tokenized" not in missing_requirements:
                missing_requirements.append("qwen_eval_prompt_rows_tokenized")
            rows.append(
                {
                    "record_type": "qwen_family_eval_prompt",
                    "model_id": model_id,
                    "revision": revision,
                    "split": split,
                    "prompt_id": f"qwen_{split}_{row_index:03d}",
                    "prompt": prompt,
                    "token_count": len(token_ids),
                    "encoded_token_ids": token_ids[:32],
                    "max_token_id": max_token_id,
                    "missing_requirements": row_requirements,
                }
            )

    return {
        "record_type": "qwen_family_eval_prompt_probe",
        "model_id": model_id,
        "revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "tokenizer_length": tokenizer_length,
        "expected_vocab_size": expected_vocab_size,
        "required_splits": list(split_order),
        "min_prompts_per_split": min_prompts_per_split,
        "min_tokens_per_prompt": min_tokens_per_prompt,
        "split_counts": split_counts,
        "encoded_prompt_count": len(rows),
        "prompt_pack_ready": not missing_requirements,
        "missing_requirements": missing_requirements,
        "prompt_rows": rows,
    }


def probe_qwen_family_eval_prompts(
    *,
    tokenizer_dir: str | Path,
    model_id: str,
    revision: str,
    min_prompts_per_split: int,
    min_tokens_per_prompt: int,
    expected_vocab_size: int | None = None,
) -> dict[str, object]:
    tokenizer_path = Path(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    payload = build_qwen_family_eval_prompt_probe(
        tokenizer,
        model_id=model_id,
        revision=revision,
        min_prompts_per_split=min_prompts_per_split,
        min_tokens_per_prompt=min_tokens_per_prompt,
        expected_vocab_size=expected_vocab_size,
    )
    payload["tokenizer_dir"] = str(tokenizer_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and tokenize the Qwen family eval prompt pack."
    )
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--min-prompts-per-split", type=int, default=22)
    parser.add_argument("--min-tokens-per-prompt", type=int, default=2)
    parser.add_argument("--expected-vocab-size", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = probe_qwen_family_eval_prompts(
        tokenizer_dir=args.tokenizer_dir,
        model_id=args.model_id,
        revision=args.revision,
        min_prompts_per_split=args.min_prompts_per_split,
        min_tokens_per_prompt=args.min_tokens_per_prompt,
        expected_vocab_size=args.expected_vocab_size,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
