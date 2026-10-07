from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from keep.io.continuous_sidecar import (
    copy_declared_continuous_sidecars,
    link_seed_artifact_groups,
)
from keep.io.router_correction import (
    load_conversion_manifest,
    write_router_correction_artifact_manifest,
    write_router_correction_sidecar,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sorted_layer_items(layers: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    parsed: list[tuple[int, dict[str, Any]]] = []
    for key, value in layers.items():
        if not isinstance(value, dict):
            raise ValueError(f"layer {key!r} must contain a JSON object")
        parsed.append((int(key), value))
    return sorted(parsed)


def _delta_vector(
    entries: list[Any],
    *,
    num_experts: int,
    scale: float,
    max_abs_delta: float | None,
    layer: int,
) -> mx.array:
    values = [0.0] * num_experts
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(f"layer {layer} expert_bias_delta entries must be objects")
        expert = int(item.get("expert", -1))
        if expert < 0 or expert >= num_experts:
            raise ValueError(f"layer {layer} expert index {expert} outside num_experts={num_experts}")
        delta = float(item.get("delta", 0.0)) * scale
        if max_abs_delta is not None:
            limit = abs(float(max_abs_delta))
            delta = max(-limit, min(limit, delta))
        values[expert] = delta
    return mx.array(values, dtype=mx.float32)


def _layer_delta_entries(layer_record: dict[str, Any], *, layer: int) -> list[Any]:
    entries = layer_record.get("expert_bias_delta")
    if entries is None:
        raise ValueError(f"layer {layer} is missing expert_bias_delta")
    if not isinstance(entries, list):
        raise ValueError(f"layer {layer} expert_bias_delta must be a list")
    return entries


def materialize_router_correction(
    *,
    disagreement_json: str | Path,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    num_experts: int = 128,
    scale: float = 1.0,
    max_abs_delta: float | None = None,
) -> dict[str, Any]:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    disagreement_path = Path(disagreement_json)
    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    summary = _read_json(disagreement_path)
    if summary.get("record_type") != "air_route_trace_disagreement_summary":
        raise ValueError("disagreement JSON must be an air_route_trace_disagreement_summary")
    layers = summary.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("disagreement JSON must contain non-empty layers")

    linked_group_count = link_seed_artifact_groups(seed_artifact_dir=seed_root, output_dir=output_root)
    preserved_sidecars = copy_declared_continuous_sidecars(seed_artifact_dir=seed_root, output_dir=output_root)

    correction_entries: list[dict[str, Any]] = []
    materialized_layers: list[int] = []
    for layer, layer_record in _sorted_layer_items(layers):
        vector = _delta_vector(
            _layer_delta_entries(layer_record, layer=layer),
            num_experts=num_experts,
            scale=scale,
            max_abs_delta=max_abs_delta,
            layer=layer,
        )
        correction_entries.append(
            write_router_correction_sidecar(
                output_dir=output_root,
                layer=layer,
                num_experts=num_experts,
                expert_bias_delta=vector,
            )
        )
        materialized_layers.append(layer)

    if not correction_entries:
        raise ValueError("no router corrections were materialized")

    seed_manifest = load_conversion_manifest(seed_root)
    run_manifest = {
        "kind": "route_trace_expert_bias_delta",
        "source_disagreement_json": str(disagreement_path),
        "source_record_type": summary.get("record_type"),
        "num_experts": int(num_experts),
        "scale": float(scale),
        "max_abs_delta": None if max_abs_delta is None else float(max_abs_delta),
        "layers": materialized_layers,
        "linked_group_count": int(linked_group_count),
        "preserved_sidecar_count": len(preserved_sidecars),
    }
    write_router_correction_artifact_manifest(
        seed_manifest=seed_manifest,
        output_dir=output_root,
        corrections=correction_entries,
        run_manifest=run_manifest,
    )

    return {
        "record_type": "air_router_correction_materialization",
        "schema_version": 1,
        "seed_artifact_dir": str(seed_root),
        "output_dir": str(output_root),
        "source_disagreement_json": str(disagreement_path),
        "num_experts": int(num_experts),
        "scale": float(scale),
        "max_abs_delta": None if max_abs_delta is None else float(max_abs_delta),
        "layers": materialized_layers,
        "linked_group_count": int(linked_group_count),
        "preserved_sidecar_count": len(preserved_sidecars),
        "correction_count": len(correction_entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize GLM-4.5-Air router correction sidecars from route-trace disagreement JSON."
    )
    parser.add_argument("--disagreement-json", required=True)
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-abs-delta", type=float)
    parser.add_argument("--append-jsonl", help="Optional JSONL ledger to append the materialization summary to.")
    args = parser.parse_args()

    summary = materialize_router_correction(
        disagreement_json=args.disagreement_json,
        seed_artifact_dir=args.seed_artifact_dir,
        output_dir=args.output_dir,
        num_experts=args.num_experts,
        scale=args.scale,
        max_abs_delta=args.max_abs_delta,
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.append_jsonl:
        with Path(args.append_jsonl).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(text, end="")


if __name__ == "__main__":
    main()
