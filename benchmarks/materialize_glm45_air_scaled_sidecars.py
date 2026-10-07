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


def _parse_scale_policy_json(value: str) -> dict[str, float]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--scale-policy-json must be a JSON object")
    policy: dict[str, float] = {}
    for key, raw in payload.items():
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"scale policy {key!r} must be numeric")
        policy[str(key)] = float(raw)
    return policy


def _scale_for_layer(policy: dict[str, float], layer: int) -> float:
    if f"layer_{layer}" in policy:
        return policy[f"layer_{layer}"]
    if str(layer) in policy:
        return policy[str(layer)]
    return 1.0


def _source_sidecars(source_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    continuous = source_manifest.get("continuous_parameters")
    if not isinstance(continuous, dict) or continuous.get("enabled") is not True:
        raise ValueError("source artifact must declare enabled continuous_parameters")
    sidecars = continuous.get("sidecars")
    if not isinstance(sidecars, list) or not sidecars:
        raise ValueError("source artifact must declare at least one continuous sidecar")
    result: list[dict[str, Any]] = []
    for item in sidecars:
        if not isinstance(item, dict):
            raise ValueError("continuous sidecar entries must be objects")
        result.append(dict(item))
    return result


def _scale_sidecar_tensors(
    arrays: dict[str, mx.array],
    *,
    scale: float,
) -> dict[str, mx.array | None]:
    unknown = set(arrays) - SUPPORTED_TENSORS
    if unknown:
        raise ValueError(f"unsupported continuous sidecar tensors: {sorted(unknown)}")
    if not arrays:
        raise ValueError("continuous sidecar cannot be empty")
    low_rank_left = arrays.get("low_rank_left")
    low_rank_right = arrays.get("low_rank_right")
    if (low_rank_left is None) != (low_rank_right is None):
        raise ValueError("low-rank sidecar requires both low_rank_left and low_rank_right")
    return {
        "scale_delta": arrays.get("scale_delta") * scale if arrays.get("scale_delta") is not None else None,
        "output_bias": arrays.get("output_bias") * scale if arrays.get("output_bias") is not None else None,
        "low_rank_left": low_rank_left * scale if low_rank_left is not None else None,
        "low_rank_right": low_rank_right,
    }


def _base_train_fields(source_run: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if "train_split" in source_run:
        fields["base_train_split"] = source_run["train_split"]
    if "train_row_count" in source_run:
        fields["base_train_row_count"] = source_run["train_row_count"]
    if "max_positions" in source_run:
        fields["base_max_positions"] = source_run["max_positions"]
    if "trainable" in source_run:
        fields["trainable"] = source_run["trainable"]
    return fields


def materialize_scaled_sidecars(
    *,
    seed_artifact_dir: str | Path,
    source_artifact_dir: str | Path,
    output_dir: str | Path,
    scale_policy: dict[str, float],
) -> dict[str, Any]:
    seed_root = Path(seed_artifact_dir)
    source_root = Path(source_artifact_dir)
    output_root = Path(output_dir)
    if output_root.exists():
        raise FileExistsError(f"{output_root} already exists")
    source_manifest = load_conversion_manifest(source_root)
    source_continuous = source_manifest.get("continuous_parameters") or {}
    source_run = source_continuous.get("run")
    if not isinstance(source_run, dict):
        source_run = {}

    linked_group_count = link_seed_artifact_groups(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
    )
    scaled_sidecars: list[dict[str, Any]] = []
    materialized_layers: list[int] = []
    for sidecar in _source_sidecars(source_manifest):
        layer = int(sidecar.get("layer", -1))
        projection = str(sidecar.get("projection", ""))
        relpath = sidecar.get("path")
        if not isinstance(relpath, str) or not relpath:
            raise ValueError(f"continuous sidecar entry for layer {layer} {projection} needs a path")
        scale = _scale_for_layer(scale_policy, layer)
        arrays = mx.load(str(source_root / relpath))
        tensors = _scale_sidecar_tensors(arrays, scale=scale)
        scaled_sidecars.append(
            write_continuous_sidecar(
                output_dir=output_root,
                layer=layer,
                projection=projection,
                scale_delta=tensors["scale_delta"],
                output_bias=tensors["output_bias"],
                low_rank_left=tensors["low_rank_left"],
                low_rank_right=tensors["low_rank_right"],
            )
        )
        materialized_layers.append(layer)

    run_manifest = {
        "kind": "scaled_block_local_sidecar_alpha_sweep",
        "seed_artifact_dir": str(seed_root),
        "source_artifact": str(source_root),
        "source_run_kind": source_run.get("kind"),
        "scale_policy": dict(sorted(scale_policy.items())),
        "sidecar_count": int(len(scaled_sidecars)),
        "linked_group_count": int(linked_group_count),
        "layers": sorted(set(materialized_layers)),
        **_base_train_fields(source_run),
    }
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
        sidecars=scaled_sidecars,
        run_manifest=run_manifest,
    )

    return {
        "record_type": "air_scaled_sidecar_materialization",
        "schema_version": 1,
        "seed_artifact_dir": str(seed_root),
        "source_artifact_dir": str(source_root),
        "output_dir": str(output_root),
        "scale_policy": dict(sorted(scale_policy.items())),
        "linked_group_count": int(linked_group_count),
        "sidecar_count": len(scaled_sidecars),
        "layers": sorted(set(materialized_layers)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize GLM-4.5-Air scaled continuous sidecars from a source sidecar artifact."
    )
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--source-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale-policy-json", required=True)
    parser.add_argument("--append-jsonl", help="Optional JSONL ledger to append the materialization summary to.")
    args = parser.parse_args()

    try:
        summary = materialize_scaled_sidecars(
            seed_artifact_dir=args.seed_artifact_dir,
            source_artifact_dir=args.source_artifact_dir,
            output_dir=args.output_dir,
            scale_policy=_parse_scale_policy_json(args.scale_policy_json),
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
