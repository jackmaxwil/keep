from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlx_vq.convert.inspect_hf import fetch_hf_config
from mlx_vq.convert.qwen_moe import (
    QWEN36_35B_A3B_MODEL_ID,
    QWEN36_35B_A3B_REVISION,
    audit_qwen_moe_source_payloads,
    plan_qwen_moe_conversion_from_index,
)


def _load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _load_config(args: argparse.Namespace) -> dict:
    if args.config_path is not None:
        return _load_json(args.config_path)
    return fetch_hf_config(args.model_id, revision=args.revision)


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
        description="Audit local Qwen MoE safetensors payload shards before materialization."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--model-id", default=QWEN36_35B_A3B_MODEL_ID)
    parser.add_argument("--revision", default=QWEN36_35B_A3B_REVISION)
    parser.add_argument("--expected-language-layers", type=int)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--code-bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    plan = plan_qwen_moe_conversion_from_index(
        _load_json(args.index_path),
        config=_load_config(args),
        model_id=args.model_id,
        revision=args.revision,
        expected_language_layers=args.expected_language_layers,
        code_bits=args.code_bits,
        group_size=args.group_size,
    )
    payload = audit_qwen_moe_source_payloads(
        source_dir=args.source_dir,
        plan=plan,
        max_groups=args.max_groups,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["materialization_blocked"]:
        print(
            "qwen_moe_source_payloads_missing: "
            + ", ".join(str(shard) for shard in payload["missing_shards"]),
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
