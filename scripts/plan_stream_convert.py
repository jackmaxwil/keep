from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from mlx_vq.convert.inspect_hf import fetch_hf_config
from mlx_vq.convert.stream_convert import (
    convert_vq_group_from_safetensors,
    convert_vq_groups_from_safetensors,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
    source_shard_inventory,
)


def _load_config(args) -> dict:
    if args.config_path is not None:
        return json.loads(Path(args.config_path).read_text())
    return fetch_hf_config(args.model_id, revision=args.revision)


def _load_index_path(args) -> Path:
    if args.index_path is not None:
        return Path(args.index_path)
    return Path(hf_hub_download(args.model_id, "model.safetensors.index.json", revision=args.revision))


def _select_group(plan, spec: str):
    try:
        layer_raw, projection = spec.split(":", maxsplit=1)
        layer = int(layer_raw)
    except ValueError as exc:
        raise SystemExit("--convert-group must use '<layer>:<projection>', e.g. '3:gate_proj'") from exc
    for group in plan.vq_groups:
        if group.layer == layer and group.projection == projection:
            return group
    raise SystemExit(f"no planned VQ group for {spec!r}")


def _inventory_groups(plan, max_groups: int | None):
    return plan.vq_groups[:max_groups] if max_groups is not None else plan.vq_groups


def _parse_code_bits_policy(entries: list[str] | None) -> dict[str, int]:
    policy: dict[str, int] = {}
    for entry in entries or []:
        try:
            key, raw_value = entry.split("=", maxsplit=1)
            value = int(raw_value)
        except ValueError as exc:
            raise SystemExit("--code-bits-policy entries must use '<layer|*>:<projection|*>=<8|16>'") from exc
        if ":" not in key:
            raise SystemExit("--code-bits-policy keys must include a ':' separator")
        if value not in (8, 16):
            raise SystemExit("--code-bits-policy values must be 8 or 16")
        policy[key] = value
    return policy


def _download_missing_source_shards(args, *, index_path: Path, source_dir: Path, inventory) -> dict:
    missing = list(inventory.missing_shards)
    if args.max_download_shards is not None:
        missing = missing[: args.max_download_shards]
    filenames = [path.name for path in missing]
    result = {
        "dry_run": args.download_dry_run,
        "requested_shards": len(filenames),
        "requested_filenames": filenames[:10],
        "downloaded_shards": 0,
        "downloaded_paths": [],
        "snapshot_path": None,
    }
    if args.download_dry_run or not filenames:
        return result

    use_default_cache = source_dir.resolve() == index_path.parent.resolve()
    snapshot_path = Path(
        snapshot_download(
            args.model_id,
            revision=args.revision,
            allow_patterns=filenames,
            local_dir=None if use_default_cache else source_dir,
            max_workers=args.download_workers,
        )
    )
    downloaded_paths = [snapshot_path / filename for filename in filenames]
    result["downloaded_shards"] = len(filenames)
    result["downloaded_paths"] = [str(path) for path in downloaded_paths[:10]]
    result["snapshot_path"] = str(snapshot_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run streaming VQ conversion from HF safetensors indexes.")
    parser.add_argument("--model-id", default="zai-org/GLM-5.2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--code-bits", type=int, default=8)
    parser.add_argument(
        "--code-bits-policy",
        action="append",
        help="Optional per-group policy '<layer|*>:<projection|*>=<8|16>'; repeatable.",
    )
    parser.add_argument("--scale-estimator", choices=("max_abs", "percentile_99"), default="max_abs")
    parser.add_argument("--convert-group", help="Optional '<layer>:<projection>' group to convert from local shards.")
    parser.add_argument("--convert-all", action="store_true", help="Convert planned VQ groups from local shards.")
    parser.add_argument("--source-dir", help="Directory containing local safetensors shards for preflight or conversion.")
    parser.add_argument("--output-path", help="Output safetensors path for --convert-group.")
    parser.add_argument("--output-dir", help="Output directory for --convert-all group shards.")
    parser.add_argument("--manifest-path", help="Optional manifest JSON path for --convert-all.")
    parser.add_argument("--max-groups", type=int, help="Optional conversion cap for staged --convert-all runs.")
    parser.add_argument(
        "--download-missing-source-shards",
        action="store_true",
        help="Download missing source shards for planned VQ groups before conversion.",
    )
    parser.add_argument("--download-dry-run", action="store_true", help="Report missing source shards without downloading.")
    parser.add_argument("--max-download-shards", type=int, help="Optional cap for --download-missing-source-shards.")
    parser.add_argument("--download-workers", type=int, default=8, help="Parallel workers for source-shard downloads.")
    parser.add_argument("--skip-existing", action="store_true", help="For --convert-all, skip group shards already present.")
    parser.add_argument("--overwrite", action="store_true", help="For --convert-all, overwrite group shards already present.")
    parser.add_argument("--expert-workers", type=int, default=1, help="Expert-level process workers for VQ group conversion.")
    args = parser.parse_args()

    if args.convert_group is not None and args.convert_all:
        raise SystemExit("--convert-group and --convert-all are mutually exclusive")
    if args.max_download_shards is not None and args.max_download_shards <= 0:
        raise SystemExit("--max-download-shards must be positive")
    if args.download_workers <= 0:
        raise SystemExit("--download-workers must be positive")
    if args.expert_workers <= 0:
        raise SystemExit("--expert-workers must be positive")

    config = _load_config(args)
    index_path = _load_index_path(args)
    index = load_safetensors_index(index_path)
    code_bits_policy = _parse_code_bits_policy(args.code_bits_policy)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id=args.model_id,
        group_size=args.group_size,
        code_bits=args.code_bits,
        code_bits_policy=code_bits_policy,
    )

    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "index_path": str(index_path),
        "source_shards": len(plan.source_shards),
        "tensors": len(plan.tensors),
        "vq_groups": len(plan.vq_groups),
        "missing_vq_groups": list(plan.missing_vq_groups),
        "vq_code_bytes": plan.vq_code_bytes,
        "vq_scale_bytes": plan.vq_scale_bytes,
        "peak_source_projection_bytes": plan.peak_source_projection_bytes,
        "scale_estimator": args.scale_estimator,
        "expert_workers": args.expert_workers,
        "code_bits_policy": dict(plan.code_bits_policy),
        "source_weight_encoding": plan.source_weight_encoding.value,
        "source_decoder": plan.source_decoder,
    }

    if args.source_dir is not None:
        inventory_groups = _inventory_groups(plan, args.max_groups)
        result["source_shards_local"] = source_shard_inventory(args.source_dir, inventory_groups).to_json_dict()

    if args.download_missing_source_shards:
        source_dir = Path(args.source_dir) if args.source_dir is not None else index_path.parent
        inventory_groups = _inventory_groups(plan, args.max_groups)
        before = source_shard_inventory(source_dir, inventory_groups)
        download = _download_missing_source_shards(args, index_path=index_path, source_dir=source_dir, inventory=before)
        after = source_shard_inventory(source_dir, inventory_groups)
        result["source_shard_download"] = {
            **download,
            "source_dir": str(source_dir),
            "before": before.to_json_dict(),
            "after": after.to_json_dict(),
        }
        result["source_shards_local"] = after.to_json_dict()

    if args.convert_group is not None:
        if args.source_dir is None or args.output_path is None:
            raise SystemExit("--convert-group requires --source-dir and --output-path")
        group = _select_group(plan, args.convert_group)
        converted = convert_vq_group_from_safetensors(
            source_dir=args.source_dir,
            index=index,
            group=group,
            output_path=args.output_path,
            scale_estimator=args.scale_estimator,
            expert_workers=args.expert_workers,
        )
        result["converted"] = {
            "output_path": str(converted.output_path),
            "codes_name": converted.codes_name,
            "scales_name": converted.scales_name,
            "codes_shape": converted.codes_shape,
            "scales_shape": converted.scales_shape,
            "source_tensors_read": converted.source_tensors_read,
            "peak_source_tensor_bytes": converted.peak_source_tensor_bytes,
            "code_bits": converted.code_bits,
            "group_size": converted.group_size,
            "scale_estimator": converted.scale_estimator,
            "elapsed_seconds": converted.elapsed_seconds,
            "expert_workers": converted.expert_workers,
            "source_weight_encoding": converted.source_weight_encoding.value,
            "source_decoder": converted.source_decoder,
        }

    if args.convert_all:
        if args.output_dir is None:
            raise SystemExit("--convert-all requires --output-dir")
        source_dir = Path(args.source_dir) if args.source_dir is not None else index_path.parent
        inventory_groups = _inventory_groups(plan, args.max_groups)
        result["source_dir"] = str(source_dir)
        result["source_shards_local"] = source_shard_inventory(
            source_dir,
            inventory_groups,
        ).to_json_dict()
        manifest = convert_vq_groups_from_safetensors(
            source_dir=source_dir,
            index=index,
            plan=plan,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            max_groups=args.max_groups,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
            scale_estimator=args.scale_estimator,
            expert_workers=args.expert_workers,
        )
        result["manifest"] = {
            "path": str(manifest.manifest_path),
            "converted_vq_groups": len(manifest.converted_groups),
            "existing_vq_groups": len(manifest.existing_groups),
            "ready_vq_groups": len(manifest.converted_groups) + len(manifest.existing_groups),
            "planned_vq_groups": manifest.planned_vq_groups,
            "skipped_vq_groups": manifest.skipped_vq_groups,
            "skipped_existing_outputs": manifest.skipped_existing_outputs,
            "total_source_tensors_read": manifest.total_source_tensors_read,
            "peak_source_tensor_bytes": manifest.peak_source_tensor_bytes,
            "scale_estimator": manifest.scale_estimator,
            "code_bits_policy": dict(manifest.code_bits_policy),
            "source_weight_encoding": manifest.source_weight_encoding.value,
            "source_decoder": manifest.source_decoder,
            "source_revision": manifest.source_revision,
        }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
