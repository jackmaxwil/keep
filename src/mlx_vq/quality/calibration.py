from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask

from mlx_vq.benchmark.glm45_air import append_jsonl, load_resident_air
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.models.glm45_air_vq_adapter import (
    GLM45AirVQMoE,
    has_dense_glm45_air_routed_expert_parameters,
    has_unbound_glm45_air_vq_experts,
)
from mlx_vq.quality.prompts import QualityPrompt, get_quality_prompts


@dataclass(frozen=True)
class ActivationSummary:
    abs_max: float
    mean_abs: float
    rms: float
    finite: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "abs_max": self.abs_max,
            "mean_abs": self.mean_abs,
            "rms": self.rms,
            "finite": self.finite,
        }


def summarize_activation(values: np.ndarray) -> ActivationSummary:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return ActivationSummary(abs_max=0.0, mean_abs=0.0, rms=0.0, finite=True)
    return ActivationSummary(
        abs_max=float(np.max(np.abs(array))),
        mean_abs=float(np.mean(np.abs(array))),
        rms=float(np.sqrt(np.mean(array * array))),
        finite=bool(np.isfinite(array).all()),
    )


def _prompt_by_id(prompt_id: str, *, prompt_set: str = "base") -> QualityPrompt:
    for prompt in get_quality_prompts(prompt_set):
        if prompt.prompt_id == prompt_id:
            return prompt
    choices = ", ".join(prompt.prompt_id for prompt in get_quality_prompts(prompt_set))
    raise ValueError(f"unknown quality prompt {prompt_id!r}; choices: {choices}")


def prompt_token_ids(tokenizer, prompt: QualityPrompt) -> list[int]:
    tokens = list(tokenizer.encode(prompt.text, add_special_tokens=False))
    if prompt.context_tokens is None:
        return tokens
    if not tokens:
        raise ValueError(f"quality prompt {prompt.prompt_id!r} tokenized to zero tokens")
    expanded = list(tokens)
    while len(expanded) < prompt.context_tokens:
        expanded.extend(tokens)
    return expanded[: prompt.context_tokens]


def build_activation_record(
    *,
    model_id: str,
    artifact_dir: str,
    prompt: QualityPrompt,
    input_token_ids: list[int],
    layer: int,
    num_experts: int,
    indices: np.ndarray,
    scores: np.ndarray,
    moe_input: np.ndarray,
    down_input: np.ndarray | None,
    dense_expert_params: bool,
    unbound_vq_experts: bool,
    include_activation_rows: bool = False,
) -> dict[str, Any]:
    expert_ids, counts = np.unique(indices.astype(np.int64), return_counts=True)
    expert_counts = {str(int(expert)): int(count) for expert, count in zip(expert_ids, counts, strict=True)}
    coverage_fraction = float(len(expert_ids) / max(1, num_experts))
    record: dict[str, Any] = {
        "model_id": model_id,
        "artifact_dir": artifact_dir,
        "prompt_id": prompt.prompt_id,
        "prompt_text": prompt.text,
        "input_token_ids": input_token_ids,
        "context_tokens": len(input_token_ids),
        "layer": layer,
        "num_experts": int(num_experts),
        "route_count": int(indices.size),
        "top_k": int(indices.shape[1]),
        "selected_experts": [int(expert) for expert in expert_ids],
        "selected_expert_count": int(len(expert_ids)),
        "expert_counts": expert_counts,
        "expert_coverage_fraction": coverage_fraction,
        "route_indices": indices.astype(np.int64).tolist(),
        "router_scores": scores.astype(np.float32).tolist(),
        "router_score_min": float(np.min(scores)),
        "router_score_max": float(np.max(scores)),
        "router_score_mean": float(np.mean(scores)),
        "projection_inputs": {
            "gate_proj": "moe_input",
            "up_proj": "moe_input",
            "down_proj": "down_input" if down_input is not None else None,
        },
        "moe_input": summarize_activation(moe_input).to_dict(),
        "down_input": summarize_activation(down_input).to_dict() if down_input is not None else None,
        "dense_expert_params": dense_expert_params,
        "unbound_vq_experts": unbound_vq_experts,
    }
    if include_activation_rows:
        record["moe_input_rows"] = np.asarray(moe_input, dtype=np.float32).tolist()
        record["down_input_rows"] = (
            np.asarray(down_input, dtype=np.float32).tolist()
            if down_input is not None
            else None
        )
    return record


def collect_activation_records_for_tokens(
    model,
    *,
    model_id: str,
    artifact_dir: str,
    prompt: QualityPrompt,
    input_token_ids: list[int],
    layers: tuple[int, ...],
    collect_down_inputs: bool = True,
    include_activation_rows: bool = False,
) -> list[dict[str, Any]]:
    if not input_token_ids:
        raise ValueError("input_token_ids must not be empty")
    target_layers = set(layers)
    if not target_layers:
        raise ValueError("layers must not be empty")

    h = model.model.embed_tokens(mx.array([input_token_ids], dtype=mx.int32))
    cache = [None] * len(model.layers)
    mask = create_attention_mask(h, cache[0])
    records: list[dict[str, Any]] = []

    dense_expert_params = has_dense_glm45_air_routed_expert_parameters(model)
    unbound_vq_experts = has_unbound_glm45_air_vq_experts(model)

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
            mx.eval(moe_input_mx, indices_mx, scores_mx)
            records.append(
                build_activation_record(
                    model_id=model_id,
                    artifact_dir=artifact_dir,
                    prompt=prompt,
                    input_token_ids=input_token_ids,
                    layer=layer_idx,
                    num_experts=int(layer_module.mlp.config.n_routed_experts),
                    indices=np.asarray(indices_mx, dtype=np.int64),
                    scores=np.asarray(scores_mx.astype(mx.float32), dtype=np.float32),
                    moe_input=np.asarray(moe_input_mx[0].astype(mx.float32), dtype=np.float32),
                    down_input=(
                        np.asarray(down_input_mx[0].astype(mx.float32), dtype=np.float32)
                        if down_input_mx is not None
                        else None
                    ),
                    dense_expert_params=dense_expert_params,
                    unbound_vq_experts=unbound_vq_experts,
                    include_activation_rows=include_activation_rows,
                )
            )
        h = layer_module(h, mask, cache[layer_idx])

    missing_layers = target_layers - {int(record["layer"]) for record in records}
    if missing_layers:
        raise ValueError(f"requested layer(s) not reached: {sorted(missing_layers)}")
    return records


def summarize_route_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_layer.setdefault(int(record["layer"]), []).append(record)
    layer_summaries: dict[str, Any] = {}
    for layer, layer_records in sorted(by_layer.items()):
        num_experts = max(int(record["num_experts"]) for record in layer_records)
        total_routes = sum(int(record["route_count"]) for record in layer_records)
        expert_counts: dict[int, int] = {}
        score_min = float("inf")
        score_max = -float("inf")
        score_sum = 0.0
        score_count = 0
        for record in layer_records:
            for expert, count in record.get("expert_counts", {}).items():
                expert_counts[int(expert)] = expert_counts.get(int(expert), 0) + int(count)
            scores = np.asarray(record.get("router_scores", []), dtype=np.float64)
            if scores.size:
                score_min = min(score_min, float(np.min(scores)))
                score_max = max(score_max, float(np.max(scores)))
                score_sum += float(np.sum(scores))
                score_count += int(scores.size)
        selected = sorted(expert_counts)
        layer_summaries[str(layer)] = {
            "layer": layer,
            "record_count": len(layer_records),
            "prompt_ids": sorted({str(record["prompt_id"]) for record in layer_records}),
            "num_experts": num_experts,
            "route_count": total_routes,
            "selected_expert_count": len(selected),
            "expert_coverage_fraction": float(len(selected) / max(1, num_experts)),
            "cold_expert_count": int(max(0, num_experts - len(selected))),
            "expert_counts": {str(expert): expert_counts[expert] for expert in selected},
            "top_experts": [
                {"expert": expert, "route_count": count}
                for expert, count in sorted(expert_counts.items(), key=lambda item: (-item[1], item[0]))[:16]
            ],
            "router_score_min": None if score_count == 0 else score_min,
            "router_score_max": None if score_count == 0 else score_max,
            "router_score_mean": None if score_count == 0 else score_sum / score_count,
        }
    return {
        "schema_version": 1,
        "record_type": "air_vq_route_calibration_summary",
        "record_count": len(records),
        "layers": layer_summaries,
    }


def compare_routing_records(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_indices = np.asarray(reference.get("route_indices"), dtype=np.int64)
    candidate_indices = np.asarray(candidate.get("route_indices"), dtype=np.int64)
    reference_scores = np.asarray(reference.get("router_scores"), dtype=np.float64)
    candidate_scores = np.asarray(candidate.get("router_scores"), dtype=np.float64)
    if reference_indices.shape != candidate_indices.shape:
        raise ValueError(
            "route_indices shape mismatch: "
            f"{reference_indices.shape} vs {candidate_indices.shape}"
        )
    if reference_scores.shape != candidate_scores.shape or reference_scores.shape != reference_indices.shape:
        raise ValueError("router_scores must match route_indices shape for both records")
    if reference_indices.ndim != 2:
        raise ValueError(f"route_indices must be 2D [tokens, top_k], found {reference_indices.shape}")

    set_matches = []
    token_kls = []
    union_mses = []
    for ref_row, cand_row, ref_scores, cand_scores in zip(
        reference_indices,
        candidate_indices,
        reference_scores,
        candidate_scores,
        strict=True,
    ):
        set_matches.append(set(ref_row.tolist()) == set(cand_row.tolist()))
        ref_map = _normalized_score_map(ref_row, ref_scores)
        cand_map = _normalized_score_map(cand_row, cand_scores)
        experts = sorted(set(ref_map) | set(cand_map))
        ref_values = np.asarray([ref_map.get(expert, 0.0) for expert in experts], dtype=np.float64)
        cand_values = np.asarray([cand_map.get(expert, 0.0) for expert in experts], dtype=np.float64)
        token_kls.append(_safe_kl(ref_values, cand_values))
        union_mses.append(float(np.mean((ref_values - cand_values) ** 2)))

    order_agreement = float(np.mean(reference_indices == candidate_indices))
    set_agreement = float(np.mean(np.asarray(set_matches, dtype=np.float64)))
    direct_score_mse = float(np.mean((reference_scores - candidate_scores) ** 2))
    return {
        "schema_version": 1,
        "record_type": "air_vq_route_trace_comparison",
        "prompt_id": candidate.get("prompt_id"),
        "layer": candidate.get("layer"),
        "route_topk_set_agreement": set_agreement,
        "route_order_agreement": order_agreement,
        "router_score_mse": direct_score_mse,
        "router_score_union_mse": float(np.mean(union_mses)),
        "router_score_kl": float(np.mean(token_kls)),
        "token_count": int(reference_indices.shape[0]),
        "top_k": int(reference_indices.shape[1]),
    }


def per_expert_kld_contribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    contributions: dict[int, float] = {}
    counts: dict[int, int] = {}
    for record in records:
        route_indices = np.asarray(record.get("route_indices"), dtype=np.int64)
        router_scores = np.asarray(record.get("router_scores"), dtype=np.float64)
        token_klds = np.asarray(record.get("token_klds") or [], dtype=np.float64)
        if route_indices.ndim != 2 or router_scores.shape != route_indices.shape or token_klds.size == 0:
            continue
        usable = min(route_indices.shape[0], token_klds.size)
        for token_idx in range(usable):
            scores = router_scores[token_idx]
            weights = scores / max(float(np.sum(scores)), 1.0e-12)
            for expert, weight in zip(route_indices[token_idx], weights, strict=True):
                expert_id = int(expert)
                contributions[expert_id] = contributions.get(expert_id, 0.0) + float(token_klds[token_idx] * weight)
                counts[expert_id] = counts.get(expert_id, 0) + 1
    ranked = sorted(contributions.items(), key=lambda item: (-item[1], item[0]))
    return {
        "schema_version": 1,
        "record_type": "air_vq_per_expert_kld_contribution",
        "record_count": len(records),
        "experts": [
            {
                "expert": expert,
                "weighted_kld_contribution": value,
                "route_count": counts.get(expert, 0),
            }
            for expert, value in ranked
        ],
    }


def _normalized_score_map(experts: np.ndarray, scores: np.ndarray) -> dict[int, float]:
    total = max(float(np.sum(scores)), 1.0e-12)
    return {int(expert): float(score / total) for expert, score in zip(experts, scores, strict=True)}


def _safe_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    eps = 1.0e-12
    p = np.clip(reference, eps, None)
    q = np.clip(candidate, eps, None)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def run_activation_calibration(
    *,
    prompt_ids: set[str] | None = None,
    layers: tuple[int, ...] = (1,),
    append_path: str | Path | None = None,
    model_id: str = GLM45_AIR_MODEL_ID,
    revision: str = "main",
    source_dir: str | None = None,
    config_path: str | None = None,
    index_path: str | None = None,
    artifact_dir: str | Path = "artifacts/glm-4.5-air-vq",
    collect_down_inputs: bool = True,
    include_activation_rows: bool = False,
    prompt_set: str = "base",
) -> list[dict[str, Any]]:
    selected_prompts = [
        prompt for prompt in get_quality_prompts(prompt_set)
        if prompt_ids is None or prompt.prompt_id in prompt_ids
    ]
    if not selected_prompts:
        raise ValueError("no calibration prompts selected")

    model, tokenizer, _, _ = load_resident_air(
        model_id=model_id,
        revision=revision,
        source_dir=source_dir,
        config_path=config_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
    )
    records: list[dict[str, Any]] = []
    for prompt in selected_prompts:
        token_ids = prompt_token_ids(tokenizer, prompt)
        prompt_records = collect_activation_records_for_tokens(
            model,
            model_id=model_id,
            artifact_dir=str(artifact_dir),
            prompt=prompt,
            input_token_ids=token_ids,
            layers=layers,
            collect_down_inputs=collect_down_inputs,
            include_activation_rows=include_activation_rows,
        )
        records.extend(prompt_records)
        if append_path is not None:
            for record in prompt_records:
                append_jsonl(append_path, record)
    return records


__all__ = [
    "ActivationSummary",
    "build_activation_record",
    "compare_routing_records",
    "collect_activation_records_for_tokens",
    "per_expert_kld_contribution",
    "prompt_token_ids",
    "run_activation_calibration",
    "summarize_route_records",
    "summarize_activation",
]
