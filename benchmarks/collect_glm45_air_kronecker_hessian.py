from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask

from mlx_vq.benchmark.glm45_air import load_resident_air
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.models.glm45_air_vq_adapter import GLM45AirVQMoE
from mlx_vq.quality.calibration import prompt_token_ids
from mlx_vq.quality.kronecker_hessian import (
    activation_covariance,
    save_hessian_factors,
    write_json,
    yaqa_sketch_a_hessian_factors,
)
from mlx_vq.quality.prompts import QualityPrompt, get_quality_prompts
from mlx_vq.validate.glm45_air_vq import (
    _artifact_input_for_layer,
    _decode_expert_weight,
    _expert_weight_name,
    _load_switch_glu,
    _read_named_tensor,
    _silu_np,
)


SUPPORTED_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}
FULL_YAQA_RESIDUAL_METHOD = "full_yaqa_projection_residual"


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one value")
    unsupported = sorted(set(result) - SUPPORTED_PROJECTIONS)
    if unsupported:
        raise argparse.ArgumentTypeError(f"unsupported projection(s): {unsupported}")
    return result


def _parse_layer_expert_filters(values: list[str] | None) -> dict[int, set[int]] | None:
    if not values:
        return None
    filters: dict[int, set[int]] = {}
    for value in values:
        try:
            layer_text, expert_text = value.split(":", 1)
            layer = int(layer_text)
            expert = int(expert_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"layer expert filters must have form LAYER:EXPERT, found {value!r}"
            ) from error
        if layer < 0 or expert < 0:
            raise argparse.ArgumentTypeError(
                f"layer and expert ids must be non-negative, found {value!r}"
            )
        filters.setdefault(layer, set()).add(expert)
    return filters


def _allowed_experts_for_layer(
    layer: int,
    allowed_experts: set[int] | None,
    allowed_experts_by_layer: dict[int, set[int]] | None,
) -> set[int] | None:
    if allowed_experts_by_layer is None:
        return allowed_experts
    return set(allowed_experts_by_layer.get(layer, set()))


def _selected_prompts(prompt_set: str, prompt_ids: list[str] | None, max_prompts: int | None) -> list[QualityPrompt]:
    prompts = list(get_quality_prompts(prompt_set))
    if prompt_ids:
        wanted = set(prompt_ids)
        prompts = [prompt for prompt in prompts if prompt.prompt_id in wanted]
        missing = sorted(wanted - {prompt.prompt_id for prompt in prompts})
        if missing:
            raise ValueError(f"unknown prompt ids for {prompt_set}: {missing}")
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError("no prompts selected")
    return prompts


def _append_samples(
    buckets: dict[tuple[int, str, int], list[np.ndarray]],
    *,
    layer: int,
    projections: tuple[str, ...],
    indices: np.ndarray,
    moe_input: np.ndarray,
    down_input: np.ndarray | None,
    allowed_experts: set[int] | None,
    max_samples_per_expert: int,
) -> None:
    flat_indices = indices.reshape(-1)
    flat_moe = np.repeat(moe_input.reshape(-1, moe_input.shape[-1]), indices.shape[-1], axis=0)
    flat_down = down_input.reshape(-1, down_input.shape[-1]) if down_input is not None else None
    for route_idx, expert in enumerate(flat_indices.tolist()):
        expert_id = int(expert)
        if allowed_experts is not None and expert_id not in allowed_experts:
            continue
        for projection in projections:
            key = (layer, projection, expert_id)
            if len(buckets[key]) >= max_samples_per_expert:
                continue
            if projection in {"gate_proj", "up_proj"}:
                buckets[key].append(flat_moe[route_idx].astype(np.float32, copy=False))
            else:
                if flat_down is None:
                    continue
                buckets[key].append(flat_down[route_idx].astype(np.float32, copy=False))


def _projection_layer(switch_glu, projection: str):
    if projection == "gate_proj":
        return switch_glu.gate_proj
    if projection == "up_proj":
        return switch_glu.up_proj
    if projection == "down_proj":
        return switch_glu.down_proj
    raise ValueError(f"unsupported projection {projection!r}")


def _projection_residual_output_gradients(
    source_weight: np.ndarray,
    artifact_weight: np.ndarray,
    artifact_input_rows: np.ndarray,
    *,
    source_input_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Return d(0.5 || artifact_output - source_output ||^2) / d artifact_output."""

    source = np.asarray(source_weight, dtype=np.float32)
    artifact = np.asarray(artifact_weight, dtype=np.float32)
    artifact_inputs = np.asarray(artifact_input_rows, dtype=np.float32)
    source_inputs = (
        artifact_inputs
        if source_input_rows is None
        else np.asarray(source_input_rows, dtype=np.float32)
    )
    if source.ndim != 2 or artifact.ndim != 2:
        raise ValueError("source_weight and artifact_weight must be 2D")
    if source.shape[0] != artifact.shape[0]:
        raise ValueError(f"source/artifact output dimensions differ: {source.shape} vs {artifact.shape}")
    if artifact_inputs.ndim != 2 or artifact_inputs.shape[1] != artifact.shape[1]:
        raise ValueError(
            f"artifact_input_rows must have shape [samples, {artifact.shape[1]}], "
            f"found {artifact_inputs.shape}"
        )
    if source_inputs.ndim != 2 or source_inputs.shape[1] != source.shape[1]:
        raise ValueError(
            f"source_input_rows must have shape [samples, {source.shape[1]}], "
            f"found {source_inputs.shape}"
        )
    if source_inputs.shape[0] != artifact_inputs.shape[0]:
        raise ValueError("source_input_rows and artifact_input_rows must have the same sample count")
    artifact_output = artifact_inputs @ artifact.T
    source_output = source_inputs @ source.T
    return (artifact_output - source_output).astype(np.float32)


def _append_full_yaqa_projection_residual_samples(
    input_buckets: dict[tuple[int, str, int], list[np.ndarray]],
    output_grad_buckets: dict[tuple[int, str, int], list[np.ndarray]],
    *,
    layer: int,
    projections: tuple[str, ...],
    indices: np.ndarray,
    moe_input: np.ndarray,
    switch_glu,
    source_weights: dict[tuple[int, str, int], np.ndarray],
    artifact_weights: dict[tuple[int, str, int], np.ndarray],
    allowed_experts: set[int] | None,
    max_samples_per_expert: int,
) -> None:
    flat_indices = indices.reshape(-1)
    flat_moe = np.repeat(moe_input.reshape(-1, moe_input.shape[-1]), indices.shape[-1], axis=0)
    for route_idx, expert in enumerate(flat_indices.tolist()):
        expert_id = int(expert)
        if allowed_experts is not None and expert_id not in allowed_experts:
            continue
        token = flat_moe[route_idx].astype(np.float32, copy=False)

        cached_gate: tuple[np.ndarray, np.ndarray] | None = None
        cached_up: tuple[np.ndarray, np.ndarray] | None = None
        for projection in projections:
            key = (layer, projection, expert_id)
            if len(input_buckets[key]) >= max_samples_per_expert:
                continue
            if projection in {"gate_proj", "up_proj"}:
                projection_layer = _projection_layer(switch_glu, projection)
                artifact_input = _artifact_input_for_layer(projection_layer, token)
                output_grad = _projection_residual_output_gradients(
                    source_weights[key],
                    artifact_weights[key],
                    artifact_input[None, :],
                    source_input_rows=token[None, :],
                )[0]
                input_buckets[key].append(artifact_input.astype(np.float32, copy=False))
                output_grad_buckets[key].append(output_grad.astype(np.float32, copy=False))
                continue

            gate_key = (layer, "gate_proj", expert_id)
            up_key = (layer, "up_proj", expert_id)
            down_key = (layer, "down_proj", expert_id)
            if cached_gate is None:
                gate_input = _artifact_input_for_layer(switch_glu.gate_proj, token)
                cached_gate = (
                    source_weights[gate_key] @ token,
                    artifact_weights[gate_key] @ gate_input,
                )
            if cached_up is None:
                up_input = _artifact_input_for_layer(switch_glu.up_proj, token)
                cached_up = (
                    source_weights[up_key] @ token,
                    artifact_weights[up_key] @ up_input,
                )
            source_hidden = _silu_np(cached_gate[0]) * cached_up[0]
            artifact_hidden = _silu_np(cached_gate[1]) * cached_up[1]
            artifact_down_input = _artifact_input_for_layer(switch_glu.down_proj, artifact_hidden)
            source_output = source_weights[down_key] @ source_hidden
            artifact_output = artifact_weights[down_key] @ artifact_down_input
            input_buckets[key].append(artifact_down_input.astype(np.float32, copy=False))
            output_grad_buckets[key].append((artifact_output - source_output).astype(np.float32, copy=False))


def collect_hin_samples(
    model,
    tokenizer,
    *,
    prompts: list[QualityPrompt],
    layers: tuple[int, ...],
    projections: tuple[str, ...],
    allowed_experts: set[int] | None,
    allowed_experts_by_layer: dict[int, set[int]] | None = None,
    max_samples_per_expert: int,
) -> tuple[dict[tuple[int, str, int], list[np.ndarray]], dict[str, Any]]:
    buckets: dict[tuple[int, str, int], list[np.ndarray]] = defaultdict(list)
    target_layers = set(layers)
    prompt_summaries: list[dict[str, Any]] = []
    collect_down = "down_proj" in projections

    for prompt in prompts:
        token_ids = prompt_token_ids(tokenizer, prompt)
        h = model.model.embed_tokens(mx.array([token_ids], dtype=mx.int32))
        cache = [None] * len(model.layers)
        mask = create_attention_mask(h, cache[0])
        prompt_layer_routes: dict[int, int] = {}
        for layer_idx, layer_module in enumerate(model.layers):
            if layer_idx in target_layers:
                layer_allowed_experts = _allowed_experts_for_layer(
                    layer_idx,
                    allowed_experts,
                    allowed_experts_by_layer,
                )
                if not isinstance(layer_module.mlp, GLM45AirVQMoE):
                    raise ValueError(f"layer {layer_idx} is not a sparse GLM-4.5-Air MoE layer")
                attention = layer_module.self_attn(layer_module.input_layernorm(h), mask, cache[layer_idx])
                h_after_attention = h + attention
                moe_input_mx = layer_module.post_attention_layernorm(h_after_attention)
                indices_mx, scores_mx = layer_module.mlp.gate(moe_input_mx)
                down_input_mx = None
                if collect_down:
                    if layer_module.mlp.switch_mlp is None:
                        raise RuntimeError(f"layer {layer_idx} VQ switch_mlp is not bound")
                    gate_mx = layer_module.mlp.switch_mlp.gate_proj(moe_input_mx, indices_mx)
                    up_mx = layer_module.mlp.switch_mlp.up_proj(moe_input_mx, indices_mx)
                    down_input_mx = nn.silu(gate_mx) * up_mx
                    mx.eval(down_input_mx)
                mx.eval(moe_input_mx, indices_mx, scores_mx)
                indices = np.asarray(indices_mx, dtype=np.int64)[0]
                moe_input = np.asarray(moe_input_mx[0].astype(mx.float32), dtype=np.float32)
                down_input = (
                    np.asarray(down_input_mx[0].astype(mx.float32), dtype=np.float32)
                    if down_input_mx is not None
                    else None
                )
                _append_samples(
                    buckets,
                    layer=layer_idx,
                    projections=projections,
                    indices=indices,
                    moe_input=moe_input,
                    down_input=down_input,
                    allowed_experts=layer_allowed_experts,
                    max_samples_per_expert=max_samples_per_expert,
                )
                prompt_layer_routes[layer_idx] = int(indices.size)
            h = layer_module(h, mask, cache[layer_idx])
        prompt_summaries.append(
            {
                "prompt_id": prompt.prompt_id,
                "token_count": len(token_ids),
                "layer_route_counts": {str(layer): count for layer, count in sorted(prompt_layer_routes.items())},
            }
        )
    return buckets, {"prompts": prompt_summaries}


def collect_full_yaqa_projection_residual_samples(
    model,
    tokenizer,
    *,
    prompts: list[QualityPrompt],
    layers: tuple[int, ...],
    projections: tuple[str, ...],
    allowed_experts: set[int] | None,
    allowed_experts_by_layer: dict[int, set[int]] | None = None,
    max_samples_per_expert: int,
    source_dir: Path,
    index_path: Path,
    artifact_dir: Path,
) -> tuple[
    dict[tuple[int, str, int], list[np.ndarray]],
    dict[tuple[int, str, int], list[np.ndarray]],
    dict[str, Any],
]:
    input_buckets: dict[tuple[int, str, int], list[np.ndarray]] = defaultdict(list)
    output_grad_buckets: dict[tuple[int, str, int], list[np.ndarray]] = defaultdict(list)
    target_layers = set(layers)
    prompt_summaries: list[dict[str, Any]] = []
    index = load_safetensors_index(index_path)
    source_weights: dict[tuple[int, str, int], np.ndarray] = {}
    artifact_weights: dict[tuple[int, str, int], np.ndarray] = {}
    switch_glu_by_layer = {layer: _load_switch_glu(artifact_dir, layer) for layer in layers}
    source_tensors_read = 0
    peak_source_tensor_bytes = 0

    def ensure_weights(layer: int, projection: str, expert: int) -> None:
        nonlocal source_tensors_read, peak_source_tensor_bytes
        key = (layer, projection, expert)
        if key in source_weights:
            return
        source_weight = _read_named_tensor(
            source_dir,
            index,
            _expert_weight_name(layer, expert, projection),
        ).astype(np.float32, copy=False)
        source_weights[key] = source_weight
        artifact_weights[key] = _decode_expert_weight(
            _projection_layer(switch_glu_by_layer[layer], projection),
            expert,
        ).astype(np.float32, copy=False)
        source_tensors_read += 1
        peak_source_tensor_bytes = max(peak_source_tensor_bytes, int(source_weight.nbytes))

    for prompt in prompts:
        token_ids = prompt_token_ids(tokenizer, prompt)
        h = model.model.embed_tokens(mx.array([token_ids], dtype=mx.int32))
        cache = [None] * len(model.layers)
        mask = create_attention_mask(h, cache[0])
        prompt_layer_routes: dict[int, int] = {}
        for layer_idx, layer_module in enumerate(model.layers):
            if layer_idx in target_layers:
                layer_allowed_experts = _allowed_experts_for_layer(
                    layer_idx,
                    allowed_experts,
                    allowed_experts_by_layer,
                )
                if not isinstance(layer_module.mlp, GLM45AirVQMoE):
                    raise ValueError(f"layer {layer_idx} is not a sparse GLM-4.5-Air MoE layer")
                attention = layer_module.self_attn(layer_module.input_layernorm(h), mask, cache[layer_idx])
                h_after_attention = h + attention
                moe_input_mx = layer_module.post_attention_layernorm(h_after_attention)
                indices_mx, scores_mx = layer_module.mlp.gate(moe_input_mx)
                mx.eval(moe_input_mx, indices_mx, scores_mx)
                indices = np.asarray(indices_mx, dtype=np.int64)[0]
                moe_input = np.asarray(moe_input_mx[0].astype(mx.float32), dtype=np.float32)
                needed_experts = {
                    int(expert)
                    for expert in indices.reshape(-1).tolist()
                    if layer_allowed_experts is None or int(expert) in layer_allowed_experts
                }
                needed_projections = set(projections)
                if "down_proj" in needed_projections:
                    needed_projections.update({"gate_proj", "up_proj"})
                for expert in needed_experts:
                    for projection in needed_projections:
                        ensure_weights(layer_idx, projection, expert)
                _append_full_yaqa_projection_residual_samples(
                    input_buckets,
                    output_grad_buckets,
                    layer=layer_idx,
                    projections=projections,
                    indices=indices,
                    moe_input=moe_input,
                    switch_glu=switch_glu_by_layer[layer_idx],
                    source_weights=source_weights,
                    artifact_weights=artifact_weights,
                    allowed_experts=layer_allowed_experts,
                    max_samples_per_expert=max_samples_per_expert,
                )
                prompt_layer_routes[layer_idx] = int(indices.size)
            h = layer_module(h, mask, cache[layer_idx])
        prompt_summaries.append(
            {
                "prompt_id": prompt.prompt_id,
                "token_count": len(token_ids),
                "layer_route_counts": {str(layer): count for layer, count in sorted(prompt_layer_routes.items())},
            }
        )
    return (
        input_buckets,
        output_grad_buckets,
        {
            "prompts": prompt_summaries,
            "source_tensors_read": source_tensors_read,
            "peak_source_tensor_bytes": peak_source_tensor_bytes,
            "h_out_source": "projection_residual_squared_loss",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GLM-4.5-Air Kronecker Hessian factors.")
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--artifact-dir", default="artifacts/glm-4.5-air-vq")
    parser.add_argument("--prompt-set", default="air_vq_ladder_select_v1")
    parser.add_argument("--prompt-id", action="append", dest="prompt_ids")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--layers", type=_parse_csv_ints, required=True)
    parser.add_argument("--projections", type=_parse_csv_strings, required=True)
    parser.add_argument("--expert", action="append", type=int, dest="experts")
    parser.add_argument(
        "--layer-expert",
        action="append",
        dest="layer_experts",
        help="Restrict collection to one expert for one layer, formatted LAYER:EXPERT. May repeat.",
    )
    parser.add_argument("--max-samples-per-expert", type=int, default=256)
    parser.add_argument("--min-samples-per-expert", type=int, default=1)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--sketch", choices=("A", "B"), default="A")
    parser.add_argument(
        "--method",
        choices=("blockldlq_hin_only", FULL_YAQA_RESIDUAL_METHOD),
        default="blockldlq_hin_only",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.max_samples_per_expert <= 0:
        raise SystemExit("--max-samples-per-expert must be positive")
    if args.min_samples_per_expert <= 0:
        raise SystemExit("--min-samples-per-expert must be positive")

    prompts = _selected_prompts(args.prompt_set, args.prompt_ids, args.max_prompts)
    allowed_experts = set(args.experts) if args.experts else None
    allowed_experts_by_layer = _parse_layer_expert_filters(args.layer_experts)
    if allowed_experts is not None and allowed_experts_by_layer is not None:
        raise SystemExit("--expert cannot be combined with --layer-expert")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, _, _ = load_resident_air(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
        config_path=args.config_path,
        index_path=args.index_path,
        artifact_dir=args.artifact_dir,
    )
    output_grad_buckets: dict[tuple[int, str, int], list[np.ndarray]] | None = None
    if args.method == FULL_YAQA_RESIDUAL_METHOD:
        if args.source_dir is None:
            raise SystemExit("--source-dir is required for full_yaqa projection-residual collection")
        source_root = Path(args.source_dir)
        index_path = Path(args.index_path) if args.index_path is not None else source_root / "model.safetensors.index.json"
        buckets, output_grad_buckets, collection_summary = collect_full_yaqa_projection_residual_samples(
            model,
            tokenizer,
            prompts=prompts,
            layers=args.layers,
            projections=args.projections,
            allowed_experts=allowed_experts,
            allowed_experts_by_layer=allowed_experts_by_layer,
            max_samples_per_expert=args.max_samples_per_expert,
            source_dir=source_root,
            index_path=index_path,
            artifact_dir=Path(args.artifact_dir),
        )
    else:
        buckets, collection_summary = collect_hin_samples(
            model,
            tokenizer,
            prompts=prompts,
            layers=args.layers,
            projections=args.projections,
            allowed_experts=allowed_experts,
            allowed_experts_by_layer=allowed_experts_by_layer,
            max_samples_per_expert=args.max_samples_per_expert,
        )

    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for (layer, projection, expert), sample_rows in sorted(buckets.items()):
        sample_count = len(sample_rows)
        if sample_count < args.min_samples_per_expert:
            skipped.append(
                {
                    "layer": layer,
                    "projection": projection,
                    "expert": expert,
                    "sample_count": sample_count,
                    "reason": "below_min_samples",
                }
            )
            continue
        samples = np.stack(sample_rows, axis=0).astype(np.float32, copy=False)
        h_out: np.ndarray | None = None
        if output_grad_buckets is None:
            h_in = activation_covariance(samples)
        else:
            output_grad_rows = output_grad_buckets[(layer, projection, expert)]
            if len(output_grad_rows) != sample_count:
                raise ValueError(
                    f"input/output-gradient sample count mismatch for layer {layer} "
                    f"{projection} expert {expert}: {sample_count} vs {len(output_grad_rows)}"
                )
            h_in, h_out = yaqa_sketch_a_hessian_factors(
                samples,
                np.stack(output_grad_rows, axis=0).astype(np.float32, copy=False),
            )
        filename = f"layer-{layer:05d}-{projection}-expert-{expert:05d}.safetensors"
        is_full_yaqa = h_out is not None
        metadata = {
            "method": "full_yaqa" if is_full_yaqa else "blockldlq_hin_only",
            "model_id": args.model_id,
            "revision": args.revision,
            "artifact_dir": args.artifact_dir,
            "prompt_set": args.prompt_set,
            "layer": layer,
            "projection": projection,
            "expert": expert,
            "sample_count": sample_count,
            "sketch": args.sketch,
            "input_source": "down_input" if projection == "down_proj" else "moe_input",
            "hout_status": "projection_residual_squared_loss" if is_full_yaqa else "not_collected_single_host_fallback",
        }
        save_hessian_factors(output_dir / filename, h_in=h_in, h_out=h_out, metadata=metadata, damping=args.damping)
        entry = {
            **metadata,
            "path": filename,
            "h_in_dim": int(h_in.shape[0]),
        }
        if h_out is not None:
            entry["h_out_dim"] = int(h_out.shape[0])
        entries.append(entry)

    is_full_manifest = args.method == FULL_YAQA_RESIDUAL_METHOD

    manifest = {
        "schema_version": 1,
        "record_type": "air_vq_kronecker_hessian_manifest",
        "method": {
            "kind": "full_yaqa" if is_full_manifest else "blockldlq_hin_only",
            "full_yaqa": is_full_manifest,
            "h_out_collected": is_full_manifest,
            "h_out_source": "projection_residual_squared_loss" if is_full_manifest else None,
            "notes": (
                "Full-YAQA Sketch-A factors using H_in from projection inputs and H_out from "
                "projection residual output gradients."
                if is_full_manifest
                else "Single-host Rung 4 fallback: activation covariance H_in only; do not call this YAQA."
            ),
        },
        "model_id": args.model_id,
        "revision": args.revision,
        "artifact_dir": args.artifact_dir,
        "prompt_set": args.prompt_set,
        "layers": list(args.layers),
        "projections": list(args.projections),
        "requested_experts": sorted(args.experts or []),
        "requested_layer_experts": {
            str(layer): sorted(experts)
            for layer, experts in sorted((allowed_experts_by_layer or {}).items())
        },
        "max_samples_per_expert": args.max_samples_per_expert,
        "min_samples_per_expert": args.min_samples_per_expert,
        "damping": args.damping,
        "entries": entries,
        "skipped": skipped,
        "collection": collection_summary,
    }
    write_json(output_dir / "kronecker-hessian-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
