from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID
from keep.convert.stream_convert import load_safetensors_index
from keep.io.schema import QuantizationConfig
from keep.validate.glm45_air_vq import _expert_weight_name, _read_named_tensor


ROUTED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_EXPERT_COUNT = 128


def _switch_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _group_filename(layer: int, projection: str) -> str:
    return f"layer-{layer:05d}-{projection}.safetensors"


def _relative_symlink_target(source_path: Path, target_parent: Path) -> str:
    return os.path.relpath(source_path.resolve(), start=target_parent.resolve())


def _resolve_source_root(
    *,
    model_id: str,
    revision: str,
    source_dir: str | None,
) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(snapshot_download(repo_id=model_id, revision=revision, local_files_only=True))


def _link_seed_artifact(seed_artifact_dir: Path, output_dir: Path, *, allow_existing: bool) -> dict[str, Any]:
    if output_dir.exists() and not allow_existing:
        raise FileExistsError(f"{output_dir} already exists")
    output_dir.mkdir(parents=True, exist_ok=allow_existing)
    link_count = 0
    for source_path in sorted(seed_artifact_dir.glob("layer-*.safetensors")):
        target_path = output_dir / source_path.name
        if target_path.exists() or target_path.is_symlink():
            if not allow_existing:
                raise FileExistsError(f"{target_path} already exists")
            target_path.unlink()
        os.symlink(_relative_symlink_target(source_path, target_path.parent), target_path)
        link_count += 1

    sidecar_source = seed_artifact_dir / "continuous_params"
    sidecar_target = output_dir / "continuous_params"
    if sidecar_source.exists():
        if sidecar_target.exists() and allow_existing:
            shutil.rmtree(sidecar_target)
        if sidecar_target.exists():
            raise FileExistsError(f"{sidecar_target} already exists")
        shutil.copytree(sidecar_source, sidecar_target, symlinks=True)
    return {
        "linked_group_count": link_count,
        "continuous_params_copied": sidecar_source.exists(),
    }


def _stack_projection_weight(
    *,
    source_root: Path,
    index: Any,
    layer: int,
    projection: str,
    expert_count: int,
) -> np.ndarray:
    weights = [
        _read_named_tensor(source_root, index, _expert_weight_name(layer, expert, projection))
        for expert in range(expert_count)
    ]
    return np.stack(weights, axis=0)


def _write_high_precision_projection(
    *,
    output_dir: Path,
    source_root: Path,
    index: Any,
    layer: int,
    projection: str,
    expert_count: int,
) -> dict[str, Any]:
    if projection not in ROUTED_PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}")
    weight = _stack_projection_weight(
        source_root=source_root,
        index=index,
        layer=layer,
        projection=projection,
        expert_count=expert_count,
    )
    prefix = _switch_prefix(layer, projection)
    target_path = output_dir / _group_filename(layer, projection)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    metadata = {
        "quantization_config": json.dumps(
            QuantizationConfig(policy={"dynamic_precision_tier": "high"}).to_json_dict()
        )
    }
    mx.save_safetensors(
        str(target_path),
        {f"{prefix}.weight": mx.array(weight).astype(mx.bfloat16)},
        metadata=metadata,
    )
    stored_bytes = int(target_path.stat().st_size)
    return {
        "layer": int(layer),
        "projection": projection,
        "path": str(target_path),
        "source_tensors_read": int(expert_count),
        "weight_shape": list(weight.shape),
        "stored_bytes": stored_bytes,
    }


def materialize_high_precision_projection_candidate(
    *,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    projections: list[tuple[int, str]],
    source_dir: str | Path,
    index_path: str | Path,
    expert_count: int = DEFAULT_EXPERT_COUNT,
    allow_existing: bool = False,
) -> dict[str, Any]:
    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    source_root = Path(source_dir)
    index = load_safetensors_index(Path(index_path))
    link_summary = _link_seed_artifact(seed_root, output_root, allow_existing=allow_existing)
    actions = [
        _write_high_precision_projection(
            output_dir=output_root,
            source_root=source_root,
            index=index,
            layer=layer,
            projection=projection,
            expert_count=expert_count,
        )
        for layer, projection in projections
    ]
    manifest = {
        "schema_version": 1,
        "record_type": "air_high_precision_projection_candidate_materialization",
        "seed_artifact_dir": str(seed_root),
        "output_dir": str(output_root),
        "seed_artifact_mutated": False,
        "projection_count": len(actions),
        "high_precision_projections": actions,
        "link_summary": link_summary,
        "source_dir": str(source_root),
        "index_path": str(index_path),
    }
    (output_root / "high-precision-projection-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_projection(value: str) -> tuple[int, str]:
    try:
        layer_text, projection = value.split(":", 1)
        layer = int(layer_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("projection must be LAYER:projection") from exc
    if projection not in ROUTED_PROJECTIONS:
        raise argparse.ArgumentTypeError(f"projection must be one of {ROUTED_PROJECTIONS}")
    if layer <= 0:
        raise argparse.ArgumentTypeError("layer must be positive")
    return layer, projection


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize linked GLM-4.5-Air candidates with selected BF16/source routed projections."
    )
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--projection", action="append", type=_parse_projection, required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--index-path")
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--expert-count", type=int, default=DEFAULT_EXPERT_COUNT)
    parser.add_argument("--allow-existing", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.expert_count <= 0:
        parser.error("--expert-count must be positive")
    source_root = _resolve_source_root(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    index_path = Path(args.index_path) if args.index_path is not None else source_root / "model.safetensors.index.json"
    if not index_path.exists():
        parser.error(f"source index does not exist: {index_path}")
    manifest = materialize_high_precision_projection_candidate(
        seed_artifact_dir=args.seed_artifact_dir,
        output_dir=args.output_dir,
        projections=list(args.projection),
        source_dir=source_root,
        index_path=index_path,
        expert_count=args.expert_count,
        allow_existing=args.allow_existing,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
