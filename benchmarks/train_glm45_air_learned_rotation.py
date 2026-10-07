#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID
from keep.convert.stream_convert import (
    _read_tensor_from_shard,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
)
from keep.quant.rht import deterministic_rht_signs
from keep.quality.learned_rotation_training import (
    dense_rht_rotation_matrix,
    train_learned_rht_signs_np,
    train_learned_rotation_np,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train learned orthogonal rotations for GLM-4.5-Air VQ projections.",
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", action="append", required=True, help="Layer/projection as L:projection")
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--code-bits", type=int, choices=(8, 16), default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-experts", type=int, default=2)
    parser.add_argument("--max-rows-per-expert", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260707)
    parser.add_argument("--init", choices=("rht", "identity"), default="rht")
    parser.add_argument("--structured-rht-signs", action="store_true")
    parser.add_argument("--sign-candidates-per-step", type=int, default=32)
    parser.add_argument("--rht-seed-prefix", default="learned-rotation-rht")
    parser.add_argument("--manifest-name", default="rotation-manifest.json")
    parser.add_argument("--evidence-jsonl", default=None)
    return parser


def _parse_target(raw: str) -> tuple[int, str]:
    if ":" not in raw:
        raise ValueError("--target must use L:projection format")
    layer_raw, projection = raw.split(":", maxsplit=1)
    if projection not in {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError(f"unsupported projection {projection!r}")
    return int(layer_raw), projection


def _select_group(plan, *, layer: int, projection: str):
    for group in plan.vq_groups:
        if group.layer == layer and group.projection == projection:
            return group
    raise ValueError(f"no planned VQ group for layer {layer} {projection}")


def _sample_group_weight_rows(
    *,
    source_dir: Path,
    index,
    group,
    max_experts: int,
    max_rows_per_expert: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if max_experts <= 0:
        raise ValueError("max_experts must be positive")
    if max_rows_per_expert <= 0:
        raise ValueError("max_rows_per_expert must be positive")
    sampled: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for expert in group.experts[:max_experts]:
        tensor_name = f"model.layers.{group.layer}.mlp.experts.{expert}.{group.projection}.weight"
        shard_name = index.weight_map[tensor_name]
        weight = _read_tensor_from_shard(source_dir, shard_name, tensor_name)
        if weight.shape != (group.output_dims, group.input_dims):
            raise ValueError(f"{tensor_name} must have shape {(group.output_dims, group.input_dims)}, found {weight.shape}")
        row_count = min(max_rows_per_expert, weight.shape[0])
        if row_count == weight.shape[0]:
            row_indices = np.arange(row_count, dtype=np.int32)
        else:
            row_indices = np.sort(rng.choice(weight.shape[0], size=row_count, replace=False)).astype(np.int32)
        sampled.append(weight[row_indices])
        records.append(
            {
                "expert": int(expert),
                "source_shard": shard_name,
                "row_count": int(row_count),
                "row_indices": row_indices.tolist(),
            }
        )
    return np.concatenate(sampled, axis=0).astype(np.float32, copy=False), records


def main() -> int:
    args = _parser().parse_args()
    config_path = Path(args.config_path)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = load_safetensors_index(args.index_path)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id=args.model_id,
        group_size=args.group_size,
        code_bits=args.code_bits,
    )
    rng = np.random.default_rng(args.sample_seed)
    targets = [_parse_target(raw) for raw in args.target]
    projection_payloads: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for layer, projection in targets:
        group = _select_group(plan, layer=layer, projection=projection)
        sample_weight, sample_records = _sample_group_weight_rows(
            source_dir=source_dir,
            index=index,
            group=group,
            max_experts=args.max_experts,
            max_rows_per_expert=args.max_rows_per_expert,
            rng=rng,
        )
        rotation_id = f"learned-rotation-l{layer}-{projection}-s{args.steps}-lr{args.learning_rate:g}"
        if args.structured_rht_signs:
            initial_signs = deterministic_rht_signs(
                group.input_dims,
                seed=f"{args.rht_seed_prefix}:{layer}:{projection}",
            )
            result = train_learned_rht_signs_np(
                sample_weight,
                group_size=group.group_size,
                code_bits=group.code_bits,
                steps=args.steps,
                candidates_per_step=args.sign_candidates_per_step,
                initial_signs=initial_signs,
                rng=rng,
            )
            signs_name = f"layer-{layer:05d}-{projection}-rht-signs.npy"
            np.save(output_dir / signs_name, result.rht_signs)
            projection_payloads.append(
                {
                    "layer": int(layer),
                    "projection": projection,
                    "rotation_id": rotation_id,
                    "rht_signs_path": signs_name,
                    "input_dims": int(group.input_dims),
                    "output_dims": int(group.output_dims),
                    "sample_rows": int(sample_weight.shape[0]),
                    "sample_records": sample_records,
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "accepted_flips": int(result.accepted_flips),
                }
            )
            evidence_rows.append(
                {
                    "record_type": "learned_rht_signs_training",
                    "layer": int(layer),
                    "projection": projection,
                    "rotation_id": rotation_id,
                    "steps": int(result.steps),
                    "candidates_per_step": int(result.candidates_per_step),
                    "init_kind": result.init_kind,
                    "group_size": int(group.group_size),
                    "code_bits": int(group.code_bits),
                    "sample_rows": int(sample_weight.shape[0]),
                    "accepted_flips": int(result.accepted_flips),
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "losses": list(result.losses),
                }
            )
        else:
            init_rotation = (
                dense_rht_rotation_matrix(group.input_dims, seed=f"{args.rht_seed_prefix}:{layer}:{projection}")
                if args.init == "rht"
                else np.eye(group.input_dims, dtype=np.float32)
            )
            result = train_learned_rotation_np(
                sample_weight,
                group_size=group.group_size,
                code_bits=group.code_bits,
                steps=args.steps,
                learning_rate=args.learning_rate,
                init_rotation=init_rotation,
                init_kind=args.init,
            )
            matrix_name = f"layer-{layer:05d}-{projection}-rotation.npy"
            np.save(output_dir / matrix_name, result.rotation_matrix)
            projection_payloads.append(
                {
                    "layer": int(layer),
                    "projection": projection,
                    "rotation_id": rotation_id,
                    "rotation_matrix_path": matrix_name,
                    "input_dims": int(group.input_dims),
                    "output_dims": int(group.output_dims),
                    "sample_rows": int(sample_weight.shape[0]),
                    "sample_records": sample_records,
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                }
            )
            evidence_rows.append(
                {
                    "record_type": "learned_rotation_training",
                    "layer": int(layer),
                    "projection": projection,
                    "rotation_id": rotation_id,
                    "steps": int(result.steps),
                    "learning_rate": float(result.learning_rate),
                    "init_kind": result.init_kind,
                    "group_size": int(group.group_size),
                    "code_bits": int(group.code_bits),
                    "sample_rows": int(sample_weight.shape[0]),
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "losses": list(result.losses),
                }
            )

    manifest = {
        "schema_version": 1,
        "record_type": "air_vq_learned_rotation_manifest",
        "rotation_id": f"learned-rotation-s{args.steps}-lr{args.learning_rate:g}",
        "method": {
            "kind": "cayley_sgd_quantized_reconstruction_surrogate",
            "init": args.init,
            "structured_rht_signs": bool(args.structured_rht_signs),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "notes": "Orthogonal rotations trained with refreshed RTN codes on sampled source projection rows.",
        },
        "model_id": args.model_id,
        "source_dir": str(source_dir),
        "index_path": str(args.index_path),
        "group_size": int(args.group_size),
        "code_bits": int(args.code_bits),
        "projections": projection_payloads,
    }
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_path = Path(args.evidence_jsonl) if args.evidence_jsonl else output_dir / "training-evidence.jsonl"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("w", encoding="utf-8") as handle:
        for row in evidence_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "manifest_path": str(manifest_path), "evidence_path": str(evidence_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
