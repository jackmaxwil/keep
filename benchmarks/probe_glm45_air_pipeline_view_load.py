from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model


def _load_materializer_module():
    path = Path(__file__).with_name("materialize_glm45_air_pipeline_views.py")
    spec = importlib.util.spec_from_file_location("glm45_air_pipeline_view_materializer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_materializer = _load_materializer_module()
GLM45_AIR_MODEL_ID = _materializer.GLM45_AIR_MODEL_ID
GLM45_AIR_REVISION = _materializer.GLM45_AIR_REVISION
_PipelineGroup = _materializer._PipelineGroup
_resolve_source_dir = _materializer._resolve_source_dir
build_pipeline_view_plan = _materializer.build_pipeline_view_plan
materialize_pipeline_views = _materializer.materialize_pipeline_views
_apply_pipeline_split = _materializer._apply_pipeline_split


def _layer_summary(parameter_keys: list[str]) -> dict[str, Any]:
    layers = sorted(
        {int(key.split(".")[2]) for key in parameter_keys if key.startswith("model.layers.")}
    )
    if not layers:
        return {"layers": [], "first_layer": None, "last_layer": None, "layer_count": 0}
    return {
        "layers": layers,
        "first_layer": layers[0],
        "last_layer": layers[-1],
        "layer_count": len(layers),
    }


def _probe_rank_view(
    *,
    rank_dir: Path,
    rank_plan: dict[str, Any],
    pipeline_size: int,
    layer_split: int | None,
) -> dict[str, Any]:
    mx.reset_peak_memory()
    start = time.perf_counter()
    model, _config = load_model(rank_dir, lazy=True, strict=False)
    load_seconds = time.perf_counter() - start
    before_keys = [key for key, _ in tree_flatten(model.parameters())]
    _apply_pipeline_split(
        model.model,
        rank=int(rank_plan["rank"]),
        pipeline_size=pipeline_size,
        layer_split=layer_split,
    )
    after_keys = [key for key, _ in tree_flatten(model.parameters())]

    expected_keys = set(rank_plan["parameter_keys"])
    after_key_set = set(after_keys)
    missing_after_keys = sorted(expected_keys - after_key_set)
    unexpected_after_keys = sorted(after_key_set - expected_keys)
    result = {
        "rank": rank_plan["rank"],
        "rank_dir": str(rank_dir),
        "lazy_load_seconds": load_seconds,
        "did_eval_weights": False,
        "before_pipeline_parameter_key_count": len(before_keys),
        "after_pipeline_parameter_key_count": len(after_keys),
        "expected_parameter_key_count": len(expected_keys),
        "parameter_keys_match_plan": not missing_after_keys and not unexpected_after_keys,
        "missing_after_key_count": len(missing_after_keys),
        "unexpected_after_key_count": len(unexpected_after_keys),
        "missing_after_key_examples": missing_after_keys[:20],
        "unexpected_after_key_examples": unexpected_after_keys[:20],
        "has_embed": "model.embed_tokens.weight" in after_key_set,
        "has_norm": "model.norm.weight" in after_key_set,
        "has_lm_head": "lm_head.weight" in after_key_set,
        "required_present_tensor_bytes": rank_plan["required_present_tensor_bytes"],
        "visible_shard_file_bytes": rank_plan["visible_shard_file_bytes"],
        "mlx_active_bytes_after_probe": mx.get_active_memory(),
        "mlx_peak_bytes_after_probe": mx.get_peak_memory(),
        "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **_layer_summary(after_keys),
    }
    del model
    mx.clear_cache()
    return result


def probe_pipeline_views(
    *,
    source_dir: str | Path,
    index_path: str | Path,
    output_dir: str | Path,
    rank_budget_bytes: int,
    pipeline_size: int = 2,
    layer_split: int | None = None,
    reuse_existing: bool = False,
    copy_files: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    plan = build_pipeline_view_plan(
        source_dir=source_dir,
        index_path=index_path,
        rank_budget_bytes=rank_budget_bytes,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
    )
    output_root = Path(output_dir)
    if not reuse_existing:
        materialize_pipeline_views(
            plan,
            output_dir=output_root,
            copy_files=copy_files,
            overwrite=overwrite,
        )

    rank_results = [
        _probe_rank_view(
            rank_dir=output_root / f"rank-{rank_plan['rank']}",
            rank_plan=rank_plan,
            pipeline_size=pipeline_size,
            layer_split=layer_split,
        )
        for rank_plan in plan["rank_plans"]
    ]
    return {
        "source_dir": str(source_dir),
        "output_dir": str(output_root),
        "pipeline_size": pipeline_size,
        "layer_split": layer_split,
        "complete_required_source": plan["complete_required_source"],
        "missing_source_tensor_count": plan["missing_source_tensor_count"],
        "all_ranks_match_plan": all(result["parameter_keys_match_plan"] for result in rank_results),
        "did_eval_weights": False,
        "rank_results": rank_results,
        "notes": [
            "This probe calls mlx_lm.utils.load_model(..., lazy=True, strict=False).",
            "It applies the same upstream PipelineMixin rank split and compares the remaining parameter keys to the source-view plan.",
            "It intentionally does not call mx.eval on model weights; this proves lazy bind/prune compatibility, not full BF16 residency or teacher-cache generation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe lazy upstream MLX loading against GLM-4.5-Air pipeline source views."
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default=GLM45_AIR_REVISION)
    parser.add_argument("--source-dir")
    parser.add_argument("--index-path")
    parser.add_argument("--pipeline-size", type=int, default=2)
    parser.add_argument(
        "--layer-split",
        type=int,
        help=(
            "Optional two-rank split boundary. Rank 1 owns layers below this "
            "index and rank 0 owns this layer through the final layer."
        ),
    )
    parser.add_argument("--rank-budget-gb", type=float, default=115.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--copy", action="store_true", help="copy files instead of creating symlinks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    if args.rank_budget_gb <= 0:
        parser.error("--rank-budget-gb must be positive")
    if args.pipeline_size < 2:
        parser.error("--pipeline-size must be at least 2")
    if args.layer_split is not None and args.layer_split <= 0:
        parser.error("--layer-split must be positive")
    if args.layer_split is not None and args.pipeline_size != 2:
        parser.error("--layer-split is currently supported only with --pipeline-size 2")
    if args.copy and args.reuse_existing:
        parser.error("--copy cannot be used with --reuse-existing")
    if args.overwrite and args.reuse_existing:
        parser.error("--overwrite cannot be used with --reuse-existing")

    source_dir = _resolve_source_dir(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    index_path = Path(args.index_path) if args.index_path else source_dir / "model.safetensors.index.json"
    result = probe_pipeline_views(
        source_dir=source_dir,
        index_path=index_path,
        output_dir=args.output_dir,
        rank_budget_bytes=int(args.rank_budget_gb * 1024**3),
        pipeline_size=args.pipeline_size,
        layer_split=args.layer_split,
        reuse_existing=args.reuse_existing,
        copy_files=args.copy,
        overwrite=args.overwrite,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
