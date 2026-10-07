from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

from mlx_vq.benchmark.glm45_air import append_jsonl, load_resident_air
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.models.glm45_air_vq_adapter import GLM45AirVQMoE
from mlx_vq.quality.calibration import prompt_token_ids
from mlx_vq.quality.ebss import build_route_record
from mlx_vq.quality.imatrix_collection import parse_layer_selection, selected_prompts

DEFAULT_LAYERS = "1-45"
DEFAULT_ARTIFACT_DIR = "artifacts/glm-4.5-air-vq"
DEFAULT_PROMPT_SET = "air_imatrix_calib_v1"


def _to_2d_list(array: mx.array) -> list[list[float | int]]:
    value = array
    if len(value.shape) == 3 and value.shape[0] == 1:
        value = value[0]
    return value.tolist()


def collect_route_records_for_prompt(
    *,
    model,
    prompt_id: str,
    input_token_ids: list[int],
    layers: tuple[int, ...],
) -> list[dict[str, object]]:
    if not input_token_ids:
        raise ValueError(f"prompt {prompt_id} has no tokens")
    target_layers = set(layers)
    h = model.model.embed_tokens(mx.array([input_token_ids], dtype=mx.int32))
    cache = [None] * len(model.layers)
    mask = create_attention_mask(h, cache[0])
    records: list[dict[str, object]] = []

    for layer_idx, layer_module in enumerate(model.layers):
        if layer_idx in target_layers:
            if not isinstance(layer_module.mlp, GLM45AirVQMoE):
                raise ValueError(f"layer {layer_idx} is not a sparse GLM-4.5-Air MoE layer")
            attention = layer_module.self_attn(layer_module.input_layernorm(h), mask, cache[layer_idx])
            h_after_attention = h + attention
            moe_input = layer_module.post_attention_layernorm(h_after_attention)
            indices_mx, scores_mx = layer_module.mlp.route(moe_input)
            mx.eval(indices_mx, scores_mx)
            records.append(
                build_route_record(
                    prompt_id=prompt_id,
                    layer=layer_idx,
                    route_indices=_to_2d_list(indices_mx),
                    router_scores=_to_2d_list(scores_mx),
                    num_experts=int(layer_module.mlp.config.n_routed_experts),
                    context_tokens=len(input_token_ids),
                )
            )
        h = layer_module(h, mask, cache[layer_idx])

    reached_layers = {int(record["layer"]) for record in records}
    missing_layers = target_layers - reached_layers
    if missing_layers:
        raise ValueError(f"requested layer(s) not reached: {sorted(missing_layers)}")
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect GLM-4.5-Air routed expert-count records for EBSS prompt selection."
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET)
    parser.add_argument("--prompt-id", action="append")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--layers", type=parse_layer_selection, default=parse_layer_selection(DEFAULT_LAYERS))
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.max_prompts is not None and args.max_prompts <= 0:
        parser.error("--max-prompts must be positive")
    if args.output_jsonl.exists():
        parser.error(f"{args.output_jsonl} already exists")

    try:
        prompts = selected_prompts(
            prompt_set=args.prompt_set,
            prompt_ids=args.prompt_id,
            max_prompts=args.max_prompts,
        )
    except ValueError as exc:
        parser.error(str(exc))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer, _, _ = load_resident_air(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
        config_path=args.config_path,
        index_path=args.index_path,
        artifact_dir=args.artifact_dir,
    )

    record_count = 0
    for prompt_index, prompt in enumerate(prompts):
        token_ids = prompt_token_ids(tokenizer, prompt)
        for record in collect_route_records_for_prompt(
            model=model,
            prompt_id=prompt.prompt_id,
            input_token_ids=token_ids,
            layers=tuple(args.layers),
        ):
            record["prompt_index"] = prompt_index
            record["prompt_count"] = len(prompts)
            append_jsonl(args.output_jsonl, record)
            record_count += 1
        try:
            mx.clear_cache()
            mx.metal.clear_cache()
        except Exception:
            pass

    summary = {
        "record_type": "air_route_records_collection_summary",
        "prompt_set": args.prompt_set,
        "prompt_count": len(prompts),
        "prompt_ids": [prompt.prompt_id for prompt in prompts],
        "layers": list(args.layers),
        "record_count": record_count,
        "output_jsonl": str(args.output_jsonl),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
