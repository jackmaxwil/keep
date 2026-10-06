from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask

from mlx_vq.benchmark.glm45_air import append_jsonl, load_resident_air
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.models.glm45_air_vq_adapter import GLM45AirVQMoE
from mlx_vq.quality.calibration import prompt_token_ids
from mlx_vq.quality.imatrix import (
    ROUTED_PROJECTIONS,
    accumulate_routed_projection_imatrix,
    build_air_imatrix_collection_plan,
    load_projection_imatrix_manifest,
    write_projection_imatrix_sidecars,
)
from mlx_vq.quality.imatrix_collection import (
    ImatrixAccumulator,
    finalize_entries,
    merge_entries,
    parse_layer_selection,
    parse_projection_selection,
    selected_prompts,
)
from mlx_vq.quality.prompts import QualityPrompt, get_quality_prompts


DEFAULT_LAYERS = "1-45"
DEFAULT_ARTIFACT_DIR = "artifacts/glm-4.5-air-vq"
DEFAULT_PROMPT_SET = "air_imatrix_calib_v1"


def _route_indices_array(indices_mx: mx.array) -> np.ndarray:
    indices = np.asarray(indices_mx, dtype=np.int64)
    if indices.ndim == 3 and indices.shape[0] == 1:
        indices = indices[0]
    if indices.ndim != 2:
        raise ValueError(f"expected route indices [tokens, top_k], found {indices.shape}")
    return indices


def _router_scores_array(scores_mx: mx.array) -> np.ndarray:
    scores = np.asarray(scores_mx.astype(mx.float32), dtype=np.float32)
    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 2:
        raise ValueError(f"expected router scores [tokens, top_k], found {scores.shape}")
    return scores


def _collect_prompt_imatrix(
    *,
    model,
    prompt: QualityPrompt,
    input_token_ids: list[int],
    layers: tuple[int, ...],
    projections: tuple[str, ...],
    accumulators: dict[tuple[int, str, int], ImatrixAccumulator],
) -> dict[str, object]:
    if not input_token_ids:
        raise ValueError(f"prompt {prompt.prompt_id} has no tokens")
    target_layers = set(layers)
    if not target_layers:
        raise ValueError("layers must not be empty")

    h = model.model.embed_tokens(mx.array([input_token_ids], dtype=mx.int32))
    cache = [None] * len(model.layers)
    mask = create_attention_mask(h, cache[0])
    collect_down_inputs = "down_proj" in projections
    layer_summaries: dict[str, object] = {}

    for layer_idx, layer_module in enumerate(model.layers):
        if layer_idx in target_layers:
            if not isinstance(layer_module.mlp, GLM45AirVQMoE):
                raise ValueError(f"layer {layer_idx} is not a sparse GLM-4.5-Air MoE layer")
            attention = layer_module.self_attn(layer_module.input_layernorm(h), mask, cache[layer_idx])
            h_after_attention = h + attention
            moe_input_mx = layer_module.post_attention_layernorm(h_after_attention)
            indices_mx, scores_mx = layer_module.mlp.route(moe_input_mx)
            down_input_mx = None
            if collect_down_inputs:
                if layer_module.mlp.switch_mlp is None:
                    raise RuntimeError(f"layer {layer_idx} VQ switch_mlp is not bound")
                gate_mx = layer_module.mlp.switch_mlp.gate_proj(moe_input_mx, indices_mx)
                up_mx = layer_module.mlp.switch_mlp.up_proj(moe_input_mx, indices_mx)
                down_input_mx = nn.silu(gate_mx) * up_mx
                mx.eval(down_input_mx)
            mx.eval(moe_input_mx, indices_mx)

            route_indices = _route_indices_array(indices_mx)
            router_scores = _router_scores_array(scores_mx)
            if router_scores.shape != route_indices.shape:
                raise ValueError(
                    f"router scores shape {router_scores.shape} does not match route indices {route_indices.shape}"
                )
            moe_input = np.asarray(moe_input_mx[0].astype(mx.float32), dtype=np.float32)
            down_input = (
                np.asarray(down_input_mx[0].astype(mx.float32), dtype=np.float32)
                if down_input_mx is not None
                else None
            )
            num_experts = int(layer_module.mlp.config.n_routed_experts)
            for projection in projections:
                inputs = down_input if projection == "down_proj" else moe_input
                if inputs is None:
                    raise RuntimeError(f"projection {projection} was requested without input rows")
                merge_entries(
                    accumulators,
                    accumulate_routed_projection_imatrix(
                        layer=layer_idx,
                        projection=projection,
                        inputs=inputs,
                        route_indices=route_indices,
                        router_scores=router_scores,
                        num_experts=num_experts,
                        prompt_ids=(prompt.prompt_id,),
                    ),
                )

            selected_experts = sorted({int(expert) for expert in route_indices.reshape(-1).tolist()})
            layer_summaries[str(layer_idx)] = {
                "layer": layer_idx,
                "context_tokens": len(input_token_ids),
                "route_count": int(route_indices.size),
                "selected_expert_count": len(selected_experts),
                "expert_coverage_fraction": len(selected_experts) / max(1, num_experts),
            }

        h = layer_module(h, mask, cache[layer_idx])

    missing_layers = target_layers - {int(layer) for layer in layer_summaries}
    if missing_layers:
        raise ValueError(f"requested layer(s) not reached: {sorted(missing_layers)}")

    return {
        "schema_version": 1,
        "record_type": "air_projection_imatrix_prompt_progress",
        "prompt_id": prompt.prompt_id,
        "context_tokens": len(input_token_ids),
        "layer_count": len(layer_summaries),
        "layers": layer_summaries,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prompt_ids_from_selection(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    value = payload.get("selected_prompt_ids")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain a non-empty selected_prompt_ids list")
    prompt_ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{path} selected_prompt_ids contains invalid prompt id {item!r}")
        prompt_ids.append(item)
    return prompt_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect GLM-4.5-Air routed projection imatrix sidecars from a resident VQ model."
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET)
    parser.add_argument("--prompt-id", action="append")
    parser.add_argument("--prompt-selection-json", type=Path)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--layers", type=parse_layer_selection, default=parse_layer_selection(DEFAULT_LAYERS))
    parser.add_argument(
        "--projections",
        type=parse_projection_selection,
        default=ROUTED_PROJECTIONS,
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.max_prompts is not None and args.max_prompts <= 0:
        parser.error("--max-prompts must be positive")

    output_dir = Path(args.output_dir)
    layers = tuple(args.layers)
    projections = tuple(args.projections)
    plan = build_air_imatrix_collection_plan(
        output_dir=output_dir,
        layers=layers,
        projections=projections,
        prompt_set=args.prompt_set,
        model_id=args.model_id,
        artifact_dir=args.artifact_dir,
        collection_source="resident_vq_model",
        require_output_absent=not args.allow_existing_output,
    )
    if not plan["ok"]:
        print(json.dumps(plan, indent=2, sort_keys=True))
        raise SystemExit(1)

    try:
        prompt_ids = list(args.prompt_id or [])
        selection_prompt_ids: list[str] = []
        if args.prompt_selection_json is not None:
            selection_prompt_ids = _prompt_ids_from_selection(args.prompt_selection_json)
            prompt_ids.extend(selection_prompt_ids)
        prompts = selected_prompts(
            prompt_set=args.prompt_set,
            prompt_ids=prompt_ids or None,
            max_prompts=args.max_prompts,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=args.allow_existing_output)
    plan_path = output_dir / "collection-plan.json"
    progress_path = output_dir / "collection-progress.jsonl"
    _write_json(plan_path, plan)

    model, tokenizer, _, _ = load_resident_air(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
        config_path=args.config_path,
        index_path=args.index_path,
        artifact_dir=args.artifact_dir,
    )

    accumulators: dict[tuple[int, str, int], ImatrixAccumulator] = {}
    progress_records: list[dict[str, object]] = []
    for index, prompt in enumerate(prompts):
        token_ids = prompt_token_ids(tokenizer, prompt)
        record = _collect_prompt_imatrix(
            model=model,
            prompt=prompt,
            input_token_ids=token_ids,
            layers=layers,
            projections=projections,
            accumulators=accumulators,
        )
        record["prompt_index"] = index
        record["prompt_count"] = len(prompts)
        append_jsonl(progress_path, record)
        progress_records.append(record)
        try:
            mx.clear_cache()
            mx.metal.clear_cache()
        except Exception:
            pass

    entries = finalize_entries(accumulators)
    manifest = write_projection_imatrix_sidecars(
        entries,
        output_dir=output_dir,
        prompt_set=args.prompt_set,
    )
    load_projection_imatrix_manifest(output_dir / "imatrix-manifest.json")
    summary = {
        "schema_version": 1,
        "record_type": "air_projection_imatrix_collection_summary",
        "model_id": args.model_id,
        "revision": args.revision,
        "artifact_dir": args.artifact_dir,
        "prompt_set": args.prompt_set,
        "prompt_selection_json": str(args.prompt_selection_json) if args.prompt_selection_json else None,
        "selection_prompt_ids": selection_prompt_ids,
        "prompt_count": len(prompts),
        "prompt_ids": [prompt.prompt_id for prompt in prompts],
        "layers": list(layers),
        "projections": list(projections),
        "entry_count": len(entries),
        "manifest_entry_count": manifest["entry_count"],
        "plan_path": str(plan_path),
        "progress_path": str(progress_path),
        "manifest_path": str(output_dir / "imatrix-manifest.json"),
        "sidecar_root": str(output_dir / "imatrix"),
        "progress_record_count": len(progress_records),
    }
    summary_path = output_dir / "collection-summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
