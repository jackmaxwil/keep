from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from keep.convert.inspect_hf import fetch_hf_config
from keep.convert.qwen_moe import (
    QWEN36_35B_A3B_MODEL_ID,
    QWEN36_35B_A3B_REVISION,
    audit_qwen_moe_source_payloads,
    convert_qwen_moe_group_from_safetensors,
    convert_qwen_moe_groups_from_safetensors,
    plan_qwen_moe_conversion_from_index,
    qwen_moe_group_output_filename,
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


def _block_on_missing_source_payloads(
    payload: dict[str, object],
    *,
    output_json: str | None,
    append_jsonl: str | None,
) -> None:
    if output_json is not None:
        _write_json(output_json, payload)
    if append_jsonl is not None:
        _append_jsonl(append_jsonl, payload)
    print(
        "qwen_moe_source_payloads_missing: "
        + ", ".join(str(shard) for shard in payload.get("missing_shards", [])),
        file=sys.stderr,
    )
    raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize Qwen fused MoE source groups into KEEP/RAMP VQ tensors."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--model-id", default=QWEN36_35B_A3B_MODEL_ID)
    parser.add_argument("--revision", default=QWEN36_35B_A3B_REVISION)
    parser.add_argument("--expected-language-layers", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--source-projection", choices=("gate_up_proj", "down_proj"))
    parser.add_argument("--all-groups", action="store_true")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--code-bits", type=int, default=8, choices=(8, 16))
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument(
        "--scale-estimator",
        choices=("max_abs", "percentile_99"),
        default="max_abs",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.all_groups:
        if args.layer is not None or args.source_projection is not None:
            raise ValueError("--layer/--source-projection cannot be used with --all-groups")
    elif args.layer is None or args.source_projection is None:
        raise ValueError("--layer and --source-projection are required without --all-groups")

    plan = plan_qwen_moe_conversion_from_index(
        _load_json(args.index_path),
        config=_load_config(args),
        model_id=args.model_id,
        revision=args.revision,
        expected_language_layers=args.expected_language_layers,
        code_bits=args.code_bits,
        group_size=args.group_size,
    )
    if args.all_groups:
        payload_audit = audit_qwen_moe_source_payloads(
            source_dir=args.source_dir,
            plan=plan,
            max_groups=args.max_groups,
        )
        if payload_audit["materialization_blocked"]:
            _block_on_missing_source_payloads(
                payload_audit,
                output_json=args.output_json,
                append_jsonl=args.append_jsonl,
            )
        manifest = convert_qwen_moe_groups_from_safetensors(
            source_dir=args.source_dir,
            plan=plan,
            output_dir=output_dir,
            max_groups=args.max_groups,
            skip_existing=args.skip_existing,
            scale_estimator=args.scale_estimator,
        )
        payload = manifest.to_json_dict()
        if args.output_json is not None:
            _write_json(args.output_json, payload)
        if args.append_jsonl is not None:
            _append_jsonl(args.append_jsonl, payload)
        if args.output_json is None and args.append_jsonl is None:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return

    group = next(
        (
            candidate
            for candidate in plan.groups
            if candidate.layer == args.layer
            and candidate.source_projection == args.source_projection
        ),
        None,
    )
    if group is None:
        raise ValueError(
            f"no Qwen MoE group found for layer {args.layer} "
            f"projection {args.source_projection!r}"
        )
    payload_audit = audit_qwen_moe_source_payloads(
        source_dir=args.source_dir,
        plan=replace(plan, groups=(group,)),
    )
    if payload_audit["materialization_blocked"]:
        _block_on_missing_source_payloads(
            payload_audit,
            output_json=args.output_json,
            append_jsonl=args.append_jsonl,
        )
    output_path = output_dir / qwen_moe_group_output_filename(group)
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists")

    converted = convert_qwen_moe_group_from_safetensors(
        source_dir=args.source_dir,
        group=group,
        output_path=output_path,
        scale_estimator=args.scale_estimator,
    )
    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "conversion_status": "qwen_moe_group_materialized",
        "source_projection_groups": plan.source_projection_groups,
        "target_projection_groups": plan.target_projection_groups,
        "converted_group": converted.to_json_dict(),
    }
    _write_json(output_dir / "qwen-moe-materialization.json", payload)
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
