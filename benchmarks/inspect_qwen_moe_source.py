from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, hf_hub_download

from keep.convert.inspect_hf import fetch_hf_config
from keep.convert.qwen_moe import (
    QWEN36_35B_A3B_MODEL_ID,
    QWEN36_35B_A3B_MODEL_TYPE,
    QWEN36_35B_A3B_REVISION,
    audit_qwen_moe_source_index,
)


def _load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _load_index(args: argparse.Namespace) -> dict:
    if args.index_path is not None:
        return _load_json(args.index_path)
    path = hf_hub_download(
        args.model_id,
        "model.safetensors.index.json",
        revision=args.revision,
    )
    return _load_json(path)


def _load_config(args: argparse.Namespace) -> dict:
    if args.config_path is not None:
        return _load_json(args.config_path)
    return fetch_hf_config(args.model_id, revision=args.revision)


def _load_tensor_metadata(args: argparse.Namespace) -> dict[str, dict[str, object]] | None:
    if not args.verify_hf_shapes:
        return None
    metadata = get_safetensors_metadata(args.model_id, revision=args.revision)
    tensors: dict[str, dict[str, object]] = {}
    for file_metadata in metadata.files_metadata.values():
        for tensor_name, info in file_metadata.tensors.items():
            tensors[tensor_name] = {
                "dtype": info.dtype,
                "shape": list(info.shape),
                "parameter_count": int(info.parameter_count),
            }
    return tensors


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Qwen MoE source tensor naming before KEEP conversion onboarding."
    )
    parser.add_argument("--model-id", default=QWEN36_35B_A3B_MODEL_ID)
    parser.add_argument("--revision", default=QWEN36_35B_A3B_REVISION)
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--expected-model-type", default=QWEN36_35B_A3B_MODEL_TYPE)
    parser.add_argument("--expected-language-layers", type=int)
    parser.add_argument("--expected-expert-tensors", type=int)
    parser.add_argument(
        "--verify-hf-shapes",
        action="store_true",
        help="Fetch HF safetensors shard metadata and verify planned source tensor shapes.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    audit = audit_qwen_moe_source_index(
        _load_index(args),
        config=_load_config(args),
        model_id=args.model_id,
        revision=args.revision,
        expected_model_type=args.expected_model_type,
        expected_language_layers=args.expected_language_layers,
        expected_expert_tensors=args.expected_expert_tensors,
        tensor_metadata=_load_tensor_metadata(args),
    )
    payload = audit.to_json_dict()
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not audit.source_checks_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
