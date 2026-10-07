from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from ramp.benchmark.glm45_air import append_jsonl
from keep.vq.e8 import decode_weight_matrix
from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID, fetch_hf_config
from keep.convert.stream_convert import load_safetensors_index
from keep.io.continuous_sidecar import (
    copy_declared_continuous_sidecars,
    link_seed_artifact_groups,
    load_conversion_manifest,
)
from keep.io.sparse_residual import (
    write_sparse_residual_artifact_manifest,
    write_sparse_residual_rows,
)
from keep.quality.layer_probe_attribution import build_sparse_residual_rows_from_plan_report
from keep.validate.glm45_air_vq import (
    _expert_weight_name,
    _load_switch_glu,
    _read_named_tensor,
)


def _load_config(*, model_id: str, revision: str, config_path: str | None) -> dict:
    if config_path is not None:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        cached_config = hf_hub_download(
            model_id,
            "config.json",
            revision=revision,
            local_files_only=True,
        )
        return json.loads(Path(cached_config).read_text(encoding="utf-8"))
    except Exception:
        return fetch_hf_config(model_id, revision=revision)


def _resolve_source_dir(*, model_id: str, revision: str, source_dir: str | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=["model.safetensors.index.json"],
            local_files_only=True,
        )
    )


def _parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _projection_layer(switch_glu, projection: str):
    if projection == "gate_proj":
        return switch_glu.gate_proj
    if projection == "up_proj":
        return switch_glu.up_proj
    if projection == "down_proj":
        return switch_glu.down_proj
    raise ValueError("projection must be gate_proj, up_proj, or down_proj")


def _top_residual_rows(residual: np.ndarray, max_rows: int) -> np.ndarray:
    scores = np.linalg.norm(residual.astype(np.float64), axis=1)
    count = min(max_rows, scores.shape[0])
    if count <= 0:
        raise ValueError("max rows must select at least one row")
    selected = np.argpartition(-scores, count - 1)[:count]
    return selected[np.argsort(-scores[selected])].astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a GLM-4.5-Air VQ sparse row-residual sidecar candidate."
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument(
        "--projection",
        choices=["gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--experts", type=_parse_int_csv)
    parser.add_argument("--max-rows-per-expert", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--plan-json")
    parser.add_argument("--plan-target")
    parser.add_argument("--plan-expert", type=int)
    parser.add_argument("--plan-route-rank", type=int, default=0)
    parser.add_argument("--plan-max-rows", type=int)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    if args.max_rows_per_expert <= 0:
        parser.error("--max-rows-per-expert must be positive")
    if args.plan_max_rows is not None and args.plan_max_rows <= 0:
        parser.error("--plan-max-rows must be positive")
    if args.residual_scale == 0.0:
        parser.error("--residual-scale must be nonzero")
    if args.plan_json:
        missing = [
            name for name, value in {
                "--plan-target": args.plan_target,
                "--plan-expert": args.plan_expert,
            }.items()
            if value is None
        ]
        if missing:
            parser.error(f"--plan-json requires {', '.join(missing)}")
    elif args.layer is None or args.projection is None or args.experts is None:
        parser.error("--layer, --projection, and --experts are required without --plan-json")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.allow_existing:
        parser.error(f"--output-dir already exists: {output_dir}")

    seed_artifact_dir = Path(args.seed_artifact_dir)
    plan_rows = None
    if args.plan_json:
        plan_report = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
        assert args.plan_target is not None
        assert args.plan_expert is not None
        plan_rows = build_sparse_residual_rows_from_plan_report(
            plan_report,
            target_key=args.plan_target,
            expert=args.plan_expert,
            route_rank=args.plan_route_rank,
            projection=args.projection or "down_proj",
            max_rows=args.plan_max_rows,
        )
        args.layer = int(plan_rows["layer"])
        args.projection = str(plan_rows["projection"])
        args.experts = (int(plan_rows["expert"]),)

    assert args.layer is not None
    assert args.projection is not None
    assert args.experts is not None
    switch_glu = _load_switch_glu(seed_artifact_dir, args.layer)
    projection_layer = _projection_layer(switch_glu, args.projection)

    expert_indices: list[int] = []
    output_indices: list[int] = []
    residual_rows: list[np.ndarray] = []
    selected_summary: dict[str, list[int]] = {}
    residual_norms: dict[str, list[float]] = {}
    source_tensors_read = 0
    peak_source_tensor_bytes = 0

    if plan_rows is not None:
        plan_expert_indices = np.asarray(plan_rows["expert_indices"], dtype=np.int32)
        plan_output_indices = np.asarray(plan_rows["output_indices"], dtype=np.int32)
        plan_values = np.asarray(plan_rows["values"], dtype=np.float32)
        if plan_values.ndim != 2 or plan_values.shape[1] != projection_layer.input_dims:
            raise ValueError(
                f"plan residual values must have shape [rows, {projection_layer.input_dims}], "
                f"found {plan_values.shape}"
            )
        expert_indices.extend(int(value) for value in plan_expert_indices.tolist())
        output_indices.extend(int(value) for value in plan_output_indices.tolist())
        residual_rows.extend(plan_values[row].astype(np.float32, copy=False) for row in range(plan_values.shape[0]))
        selected_summary[str(plan_rows["expert"])] = [int(value) for value in plan_output_indices.tolist()]
        residual_norms[str(plan_rows["expert"])] = [
            float(np.linalg.norm(row.astype(np.float64))) for row in plan_values
        ]
        source_dir = None
        index_path = None
    else:
        source_dir = _resolve_source_dir(
            model_id=args.model_id,
            revision=args.revision,
            source_dir=args.source_dir,
        )
        index_path = Path(args.index_path) if args.index_path is not None else source_dir / "model.safetensors.index.json"
        _load_config(model_id=args.model_id, revision=args.revision, config_path=args.config_path)
        index = load_safetensors_index(index_path)
        for expert in args.experts:
            source_weight = _read_named_tensor(
                source_dir,
                index,
                _expert_weight_name(args.layer, expert, args.projection),
            ).astype(np.float32, copy=False)
            source_tensors_read += 1
            peak_source_tensor_bytes = max(peak_source_tensor_bytes, int(source_weight.nbytes))
            decoded = decode_weight_matrix(
                np.asarray(projection_layer.codes[expert]),
                np.asarray(projection_layer.scales[expert], dtype=np.float32),
                code_bits=projection_layer.code_bits,
                codebook=np.asarray(projection_layer.codebook),
            ).astype(np.float32, copy=False)
            residual = (source_weight - decoded) * float(args.residual_scale)
            rows = _top_residual_rows(residual, args.max_rows_per_expert)
            selected_summary[str(expert)] = [int(row) for row in rows.tolist()]
            residual_norms[str(expert)] = [float(np.linalg.norm(residual[row].astype(np.float64))) for row in rows]
            for row in rows:
                expert_indices.append(int(expert))
                output_indices.append(int(row))
                residual_rows.append(residual[row].astype(np.float32, copy=False))

    linked_count = link_seed_artifact_groups(
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
    )
    preserved_sidecars = copy_declared_continuous_sidecars(
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
    )
    values = np.stack(residual_rows, axis=0).astype(np.float32, copy=False)
    residual_entry = write_sparse_residual_rows(
        output_dir=output_dir,
        layer=args.layer,
        projection=args.projection,
        expert_indices=mx.array(np.asarray(expert_indices, dtype=np.int32)),
        output_indices=mx.array(np.asarray(output_indices, dtype=np.int32)),
        values=mx.array(values),
        input_dims=projection_layer.input_dims,
        output_dims=projection_layer.output_dims,
        num_experts=projection_layer.num_experts,
    )
    run_manifest = {
        "kind": "layer_probe_sparse_residual_plan" if plan_rows is not None else "rung3_sparse_row_residual_probe",
        "model_id": args.model_id,
        "revision": args.revision,
        "seed_artifact_dir": str(seed_artifact_dir),
        "source_dir": str(source_dir) if source_dir is not None else None,
        "index_path": str(index_path) if index_path is not None else None,
        "layer": int(args.layer),
        "projection": args.projection,
        "experts": [int(expert) for expert in args.experts],
        "max_rows_per_expert": int(args.max_rows_per_expert),
        "residual_scale": float(args.residual_scale),
        "plan_json": args.plan_json,
        "plan_target": args.plan_target,
        "plan_expert": args.plan_expert,
        "plan_route_rank": int(args.plan_route_rank),
        "plan_max_rows": args.plan_max_rows,
        "plan_selected_rows": plan_rows.get("selected_rows") if plan_rows is not None else None,
        "selected_rows": selected_summary,
        "residual_norms": residual_norms,
        "source_tensors_read": int(source_tensors_read),
        "peak_source_tensor_bytes": int(peak_source_tensor_bytes),
        "linked_group_count": int(linked_count),
        "preserved_continuous_sidecar_count": len(preserved_sidecars),
    }
    manifest = write_sparse_residual_artifact_manifest(
        seed_manifest=load_conversion_manifest(seed_artifact_dir),
        output_dir=output_dir,
        residuals=[residual_entry],
        run_manifest=run_manifest,
    )
    summary = {
        "schema_version": 1,
        "record_type": "air_vq_sparse_residual_materialization_summary",
        "ok": True,
        "output_dir": str(output_dir),
        "residual": residual_entry,
        "run": run_manifest,
        "manifest_keys": sorted(manifest),
    }
    if args.append_jsonl:
        append_jsonl(args.append_jsonl, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
