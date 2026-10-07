from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from keep.io.continuous_sidecar import (
    link_seed_artifact_groups,
    load_conversion_manifest,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)


SUPPORTED_TENSORS = {"scale_delta", "output_bias", "low_rank_left", "low_rank_right"}


def _source_sidecars(source_root: Path) -> list[dict[str, Any]]:
    manifest = load_conversion_manifest(source_root)
    continuous = manifest.get("continuous_parameters")
    if not isinstance(continuous, dict) or continuous.get("enabled") is not True:
        raise ValueError(f"{source_root} must declare enabled continuous_parameters")
    sidecars = continuous.get("sidecars")
    if not isinstance(sidecars, list) or not sidecars:
        raise ValueError(f"{source_root} must declare at least one continuous sidecar")
    result: list[dict[str, Any]] = []
    for item in sidecars:
        if not isinstance(item, dict):
            raise ValueError(f"{source_root} continuous sidecar entries must be objects")
        result.append(dict(item))
    return result


def _load_sidecar_arrays(source_root: Path, sidecar: dict[str, Any]) -> dict[str, mx.array]:
    relpath = sidecar.get("path")
    if not isinstance(relpath, str) or not relpath:
        raise ValueError("continuous sidecar entry needs a path")
    path = Path(relpath)
    if path.is_absolute():
        raise ValueError(f"continuous sidecar path must be artifact-relative: {relpath!r}")
    arrays = dict(mx.load(str(source_root / path)))
    unknown = set(arrays) - SUPPORTED_TENSORS
    if unknown:
        raise ValueError(f"unsupported continuous sidecar tensors in {source_root / path}: {sorted(unknown)}")
    if not arrays:
        raise ValueError(f"continuous sidecar cannot be empty: {source_root / path}")
    return arrays


def _run_fields_from_sources(source_roots: list[Path]) -> dict[str, Any]:
    source_runs: list[dict[str, Any]] = []
    source_seeds: list[str] = []
    for source_root in source_roots:
        manifest = load_conversion_manifest(source_root)
        continuous = manifest.get("continuous_parameters") or {}
        if isinstance(continuous, dict):
            seed = continuous.get("seed_artifact_dir")
            if isinstance(seed, str) and seed:
                source_seeds.append(seed)
            run = continuous.get("run")
            if isinstance(run, dict):
                source_runs.append(dict(run))
    fields: dict[str, Any] = {}
    if source_seeds:
        fields["source_seed_artifacts"] = source_seeds
    if source_runs:
        fields["source_run_kinds"] = [
            run.get("kind") for run in source_runs if isinstance(run.get("kind"), str)
        ]
    return fields


def materialize_merged_sidecars(
    *,
    seed_artifact_dir: str | Path,
    source_artifact_dirs: list[str | Path],
    output_dir: str | Path,
    run_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed_root = Path(seed_artifact_dir)
    source_roots = [Path(path) for path in source_artifact_dirs]
    output_root = Path(output_dir)
    if output_root.exists():
        raise FileExistsError(f"{output_root} already exists")
    if not source_roots:
        raise ValueError("at least one source artifact is required")

    linked_group_count = link_seed_artifact_groups(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
    )
    merged_sidecars: list[dict[str, Any]] = []
    materialized_layers: list[int] = []
    seen: set[tuple[int, str]] = set()
    for source_root in source_roots:
        for sidecar in _source_sidecars(source_root):
            layer = int(sidecar.get("layer", -1))
            projection = str(sidecar.get("projection", ""))
            key = (layer, projection)
            if key in seen:
                raise ValueError(
                    f"duplicate continuous sidecar for layer {layer} {projection}"
                )
            seen.add(key)
            arrays = _load_sidecar_arrays(source_root, sidecar)
            merged_sidecars.append(
                write_continuous_sidecar(
                    output_dir=output_root,
                    layer=layer,
                    projection=projection,
                    scale_delta=arrays.get("scale_delta"),
                    output_bias=arrays.get("output_bias"),
                    low_rank_left=arrays.get("low_rank_left"),
                    low_rank_right=arrays.get("low_rank_right"),
                )
            )
            materialized_layers.append(layer)

    run_manifest = {
        "kind": "merged_block_local_sidecars",
        "seed_artifact_dir": str(seed_root),
        "source_artifacts": [str(path) for path in source_roots],
        "sidecar_count": int(len(merged_sidecars)),
        "linked_group_count": int(linked_group_count),
        "layers": sorted(set(materialized_layers)),
        **_run_fields_from_sources(source_roots),
    }
    for key, value in (run_overrides or {}).items():
        if value is not None:
            run_manifest[str(key)] = value
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
        sidecars=merged_sidecars,
        run_manifest=run_manifest,
    )

    return {
        "record_type": "air_merged_sidecar_materialization",
        "schema_version": 1,
        "seed_artifact_dir": str(seed_root),
        "source_artifact_dirs": [str(path) for path in source_roots],
        "output_dir": str(output_root),
        "linked_group_count": int(linked_group_count),
        "sidecar_count": len(merged_sidecars),
        "layers": sorted(set(materialized_layers)),
    }


def _append_many(parser: argparse.ArgumentParser, values: list[str] | None, flag: str) -> list[str] | None:
    if values is None:
        return None
    if not values:
        parser.error(f"{flag} cannot be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge GLM-4.5-Air continuous sidecars from source artifacts."
    )
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--source-artifact-dir", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-split")
    parser.add_argument("--train-row-count", type=int)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--trainable")
    parser.add_argument("--teacher-jsonl", action="append")
    parser.add_argument("--teacher-cache-root", action="append")
    parser.add_argument("--append-jsonl", help="Optional JSONL ledger to append the materialization summary to.")
    args = parser.parse_args()

    run_overrides = {
        "train_split": args.train_split,
        "train_row_count": args.train_row_count,
        "max_positions": args.max_positions,
        "trainable": args.trainable,
        "teacher_jsonl": _append_many(parser, args.teacher_jsonl, "--teacher-jsonl"),
        "teacher_cache_root": _append_many(parser, args.teacher_cache_root, "--teacher-cache-root"),
    }
    try:
        summary = materialize_merged_sidecars(
            seed_artifact_dir=args.seed_artifact_dir,
            source_artifact_dirs=[Path(value) for value in args.source_artifact_dir],
            output_dir=args.output_dir,
            run_overrides=run_overrides,
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.append_jsonl:
        with Path(args.append_jsonl).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(text, end="")


if __name__ == "__main__":
    main()
