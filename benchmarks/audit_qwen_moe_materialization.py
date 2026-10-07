from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_vq.validate.qwen_vq import audit_qwen_moe_materialization_manifest


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
        description="Audit a Qwen MoE materialization manifest without loading a model."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--expected-source-projection-groups", type=int)
    parser.add_argument("--expected-target-projection-groups", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    audit = audit_qwen_moe_materialization_manifest(
        args.artifact_dir,
        expected_source_projection_groups=args.expected_source_projection_groups,
        expected_target_projection_groups=args.expected_target_projection_groups,
    )
    payload = audit.to_dict()
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not audit.audit_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
