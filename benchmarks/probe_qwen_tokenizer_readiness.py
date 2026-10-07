from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def evaluate_qwen_tokenizer_readiness(
    tokenizer: Any,
    *,
    prompt: str,
    min_token_count: int,
    expected_vocab_size: int | None = None,
) -> dict[str, object]:
    if min_token_count < 0:
        raise ValueError("min_token_count must be non-negative")
    token_ids = [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    tokenizer_length = int(len(tokenizer)) if hasattr(tokenizer, "__len__") else vocab_size
    max_token_id = max(token_ids) if token_ids else None
    decoded = tokenizer.decode(token_ids) if token_ids else ""
    missing_requirements: list[str] = []
    if len(token_ids) < min_token_count:
        missing_requirements.append("prompt_encodes_tokens")
    if expected_vocab_size is not None and (
        max_token_id is not None and max_token_id >= expected_vocab_size
    ):
        missing_requirements.append("token_ids_within_expected_vocab")
    return {
        "record_type": "qwen_tokenizer_readiness_probe",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_load_pass": True,
        "tokenizer_payload_ready": not missing_requirements,
        "prompt": prompt,
        "min_token_count": min_token_count,
        "token_count": len(token_ids),
        "encoded_token_ids": token_ids[:32],
        "decoded_text": decoded,
        "vocab_size": vocab_size,
        "tokenizer_length": tokenizer_length,
        "expected_vocab_size": expected_vocab_size,
        "max_token_id": max_token_id,
        "missing_requirements": missing_requirements,
    }


def probe_qwen_tokenizer_readiness(
    *,
    tokenizer_dir: str | Path,
    prompt: str,
    min_token_count: int,
    expected_vocab_size: int | None = None,
) -> dict[str, object]:
    tokenizer_path = Path(tokenizer_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as exc:  # pragma: no cover - surfaced in JSON evidence.
        return {
            "record_type": "qwen_tokenizer_readiness_probe",
            "tokenizer_dir": str(tokenizer_path),
            "tokenizer_load_pass": False,
            "tokenizer_payload_ready": False,
            "prompt": prompt,
            "min_token_count": min_token_count,
            "token_count": 0,
            "encoded_token_ids": [],
            "decoded_text": "",
            "vocab_size": 0,
            "tokenizer_length": 0,
            "expected_vocab_size": expected_vocab_size,
            "max_token_id": None,
            "missing_requirements": ["tokenizer_load"],
            "load_error": str(exc),
        }
    payload = evaluate_qwen_tokenizer_readiness(
        tokenizer,
        prompt=prompt,
        min_token_count=min_token_count,
        expected_vocab_size=expected_vocab_size,
    )
    payload["tokenizer_dir"] = str(tokenizer_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe local Qwen tokenizer payload readiness."
    )
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--min-token-count", type=int, default=1)
    parser.add_argument("--expected-vocab-size", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = probe_qwen_tokenizer_readiness(
        tokenizer_dir=args.tokenizer_dir,
        prompt=args.prompt,
        min_token_count=args.min_token_count,
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
