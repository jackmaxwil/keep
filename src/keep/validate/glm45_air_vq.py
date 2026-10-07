from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask

from keep.vq.e8 import cosine_similarity, decode_weight_matrix
from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID, summarize_config, validate_glm45_air
from keep.convert.stream_convert import (
    SafetensorsIndex,
    _read_tensor_from_shard,
    load_safetensors_index,
)
from keep.io.load import load_quantized_vq_switch_linear
from ramp.models.glm4_moe_adapter import (
    GLM4MoEGate,
    GLM4MoeRoutingConfig,
    QuantizedVQSwitchGLU,
)
from ramp.models.glm45_air_vq_adapter import GLM45AirVQMoE, GLM45AirVQModel
from keep.quant.rht import apply_rht_np


@dataclass(frozen=True)
class ArrayComparisonMetrics:
    cosine: float
    max_abs: float
    mean_abs: float
    rel_l2: float
    actual_finite: bool
    expected_finite: bool


@dataclass(frozen=True)
class GLM45AirVQValidationResult:
    model_id: str
    layer: int
    tokens: int
    input_source: str
    prompt_text: str | None
    prompt_token_ids: tuple[int, ...]
    input_state_indices: tuple[int, ...]
    hidden_size: int
    moe_intermediate_size: int
    top_k: int
    selected_experts: tuple[int, ...]
    routes: int
    source_tensors_read: int
    peak_source_tensor_bytes: int
    source_dir: str
    artifact_dir: str
    routed_indices: tuple[tuple[int, ...], ...]
    router_scores: tuple[tuple[float, ...], ...]
    route_source_metrics: tuple[dict[str, Any], ...]
    route_source_residual_topk: tuple[dict[str, Any], ...]
    route_source_sparse_residual_plans: tuple[dict[str, Any], ...]
    metrics: dict[str, ArrayComparisonMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "layer": self.layer,
            "tokens": self.tokens,
            "input_source": self.input_source,
            "prompt_text": self.prompt_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "input_state_indices": list(self.input_state_indices),
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "top_k": self.top_k,
            "selected_experts": list(self.selected_experts),
            "routes": self.routes,
            "source_tensors_read": self.source_tensors_read,
            "peak_source_tensor_bytes": self.peak_source_tensor_bytes,
            "source_dir": self.source_dir,
            "artifact_dir": self.artifact_dir,
            "routed_indices": [list(row) for row in self.routed_indices],
            "router_scores": [list(row) for row in self.router_scores],
            "route_source_metrics": list(self.route_source_metrics),
            "route_source_residual_topk": list(self.route_source_residual_topk),
            "route_source_sparse_residual_plans": list(self.route_source_sparse_residual_plans),
            "metrics": {name: asdict(metrics) for name, metrics in self.metrics.items()},
        }


def _array_metrics(actual: np.ndarray, expected: np.ndarray) -> ArrayComparisonMetrics:
    actual_values = np.asarray(actual, dtype=np.float32)
    expected_values = np.asarray(expected, dtype=np.float32)
    if actual_values.shape != expected_values.shape:
        raise ValueError(f"metric shapes must match, found {actual_values.shape} and {expected_values.shape}")
    diff = actual_values - expected_values
    expected_norm = np.linalg.norm(expected_values.reshape(-1))
    return ArrayComparisonMetrics(
        cosine=cosine_similarity(actual_values, expected_values),
        max_abs=float(np.max(np.abs(diff))) if diff.size else 0.0,
        mean_abs=float(np.mean(np.abs(diff))) if diff.size else 0.0,
        rel_l2=float(np.linalg.norm(diff.reshape(-1)) / max(float(expected_norm), 1e-12)),
        actual_finite=bool(np.isfinite(actual_values).all()),
        expected_finite=bool(np.isfinite(expected_values).all()),
    )


def _route_source_metric_records(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    vq_gate: np.ndarray,
    vq_up: np.ndarray,
    vq_routes: np.ndarray,
    source_gate: np.ndarray,
    source_up: np.ndarray,
    source_routes: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    token_count, top_k = indices.shape
    for token_idx in range(token_count):
        for route_idx in range(top_k):
            score = float(scores[token_idx, route_idx])
            weighted_actual = vq_routes[token_idx, route_idx] * score
            weighted_expected = source_routes[token_idx, route_idx] * score
            records.append(
                {
                    "token_index": int(token_idx),
                    "route_rank": int(route_idx),
                    "expert": int(indices[token_idx, route_idx]),
                    "router_score": score,
                    "metrics": {
                        "source_gate_proj": asdict(
                            _array_metrics(vq_gate[token_idx, route_idx], source_gate[token_idx, route_idx])
                        ),
                        "source_up_proj": asdict(
                            _array_metrics(vq_up[token_idx, route_idx], source_up[token_idx, route_idx])
                        ),
                        "source_routed_glu": asdict(
                            _array_metrics(vq_routes[token_idx, route_idx], source_routes[token_idx, route_idx])
                        ),
                        "source_weighted_route_contribution": asdict(
                            _array_metrics(weighted_actual, weighted_expected)
                        ),
                    },
                }
            )
    return tuple(records)


def _top_abs_residual_entries(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    limit: int = 8,
) -> list[dict[str, float | int]]:
    actual_values = np.asarray(actual, dtype=np.float32).reshape(-1)
    expected_values = np.asarray(expected, dtype=np.float32).reshape(-1)
    if actual_values.shape != expected_values.shape:
        raise ValueError(
            f"residual shapes must match, found {actual_values.shape} and {expected_values.shape}"
        )
    if actual_values.size == 0:
        return []
    residual = expected_values - actual_values
    abs_residual = np.abs(residual)
    top_count = min(int(limit), int(abs_residual.size))
    if top_count <= 0:
        return []
    candidate_indices = np.argpartition(-abs_residual, top_count - 1)[:top_count]
    ordered_indices = sorted(
        (int(index) for index in candidate_indices),
        key=lambda index: (-float(abs_residual[index]), index),
    )
    return [
        {
            "index": index,
            "actual": float(actual_values[index]),
            "expected": float(expected_values[index]),
            "source_minus_actual": float(residual[index]),
            "abs_source_minus_actual": float(abs_residual[index]),
        }
        for index in ordered_indices
    ]


def _route_source_residual_topk_records(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    vq_gate: np.ndarray,
    vq_up: np.ndarray,
    vq_routes: np.ndarray,
    source_gate: np.ndarray,
    source_up: np.ndarray,
    source_routes: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    token_count, top_k = indices.shape
    for token_idx in range(token_count):
        for route_idx in range(top_k):
            score = float(scores[token_idx, route_idx])
            weighted_actual = vq_routes[token_idx, route_idx] * score
            weighted_expected = source_routes[token_idx, route_idx] * score
            records.append(
                {
                    "token_index": int(token_idx),
                    "route_rank": int(route_idx),
                    "expert": int(indices[token_idx, route_idx]),
                    "router_score": score,
                    "top_abs_residuals": {
                        "source_gate_proj": _top_abs_residual_entries(
                            vq_gate[token_idx, route_idx],
                            source_gate[token_idx, route_idx],
                        ),
                        "source_up_proj": _top_abs_residual_entries(
                            vq_up[token_idx, route_idx],
                            source_up[token_idx, route_idx],
                        ),
                        "source_routed_glu": _top_abs_residual_entries(
                            vq_routes[token_idx, route_idx],
                            source_routes[token_idx, route_idx],
                        ),
                        "source_weighted_route_contribution": _top_abs_residual_entries(
                            weighted_actual,
                            weighted_expected,
                        ),
                    },
                }
            )
    return tuple(records)


def _route_source_sparse_residual_plan_records(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    artifact_down_inputs: np.ndarray,
    vq_routes: np.ndarray,
    source_routes: np.ndarray,
    limit: int = 8,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    token_count, top_k = indices.shape
    for token_idx in range(token_count):
        for route_idx in range(top_k):
            score = float(scores[token_idx, route_idx])
            route_input = np.asarray(artifact_down_inputs[token_idx, route_idx], dtype=np.float32).reshape(-1)
            input_norm_sq = float(np.dot(route_input.astype(np.float64), route_input.astype(np.float64)))
            desired_unweighted = (
                np.asarray(source_routes[token_idx, route_idx], dtype=np.float32)
                - np.asarray(vq_routes[token_idx, route_idx], dtype=np.float32)
            ).reshape(-1)
            desired_weighted = desired_unweighted * score
            top_count = min(int(limit), int(desired_weighted.size))
            if top_count <= 0:
                selected_indices: list[int] = []
            else:
                abs_weighted = np.abs(desired_weighted)
                candidate_indices = np.argpartition(-abs_weighted, top_count - 1)[:top_count]
                selected_indices = sorted(
                    (int(index) for index in candidate_indices),
                    key=lambda index: (-float(abs_weighted[index]), index),
                )

            rows: list[dict[str, Any]] = []
            for output_index in selected_indices:
                correction = float(desired_unweighted[output_index])
                if input_norm_sq <= 1.0e-20:
                    residual_values = np.zeros_like(route_input, dtype=np.float32)
                else:
                    residual_values = (correction / input_norm_sq * route_input).astype(np.float32, copy=False)
                reconstructed = float(np.dot(route_input.astype(np.float64), residual_values.astype(np.float64)))
                rows.append(
                    {
                        "output_index": int(output_index),
                        "desired_weighted_correction": float(desired_weighted[output_index]),
                        "desired_unweighted_correction": correction,
                        "input_norm_sq": input_norm_sq,
                        "residual_value_norm": float(np.linalg.norm(residual_values.astype(np.float64))),
                        "residual_values": [float(value) for value in residual_values.tolist()],
                        "reconstructed_unweighted_correction": reconstructed,
                        "reconstruction_abs_error": float(abs(reconstructed - correction)),
                    }
                )
            records.append(
                {
                    "token_index": int(token_idx),
                    "route_rank": int(route_idx),
                    "expert": int(indices[token_idx, route_idx]),
                    "router_score": score,
                    "projection": "down_proj",
                    "source_metric": "source_weighted_route_contribution",
                    "rows": rows,
                }
            )
    return tuple(records)


def _routing_config(config: dict[str, Any], *, model_id: str) -> GLM4MoeRoutingConfig:
    summary = summarize_config(config, model_id=model_id)
    return GLM4MoeRoutingConfig(
        hidden_size=summary.hidden_size,
        moe_intermediate_size=summary.moe_intermediate_size,
        n_routed_experts=summary.n_routed_experts,
        num_experts_per_tok=summary.num_experts_per_tok,
        norm_topk_prob=bool(config.get("norm_topk_prob", False)),
        n_group=int(config.get("n_group", 1)),
        topk_group=int(config.get("topk_group", 1)),
        routed_scaling_factor=float(config.get("routed_scaling_factor", 1.0)),
        scoring_func=str(config.get("scoring_func", "sigmoid")),
        topk_method=str(config.get("topk_method", "noaux_tc")),
        n_shared_experts=summary.n_shared_experts,
    )


def _read_named_tensor(
    source_dir: Path,
    index: SafetensorsIndex,
    name: str,
) -> np.ndarray:
    shard_name = index.weight_map.get(name)
    if shard_name is None:
        raise KeyError(f"{name!r} is missing from the safetensors index")
    return _read_tensor_from_shard(source_dir, shard_name, name)


def _silu_np(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return values * (1.0 / (1.0 + np.exp(-clipped)))


def _switch_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _expert_weight_name(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def _load_switch_glu(artifact_dir: Path, layer: int) -> QuantizedVQSwitchGLU:
    return QuantizedVQSwitchGLU(
        gate_proj=load_quantized_vq_switch_linear(
            artifact_dir / f"layer-{layer:05d}-gate_proj.safetensors",
            _switch_prefix(layer, "gate_proj"),
        ),
        up_proj=load_quantized_vq_switch_linear(
            artifact_dir / f"layer-{layer:05d}-up_proj.safetensors",
            _switch_prefix(layer, "up_proj"),
        ),
        down_proj=load_quantized_vq_switch_linear(
            artifact_dir / f"layer-{layer:05d}-down_proj.safetensors",
            _switch_prefix(layer, "down_proj"),
        ),
    )


def _decode_expert_weight(layer, expert: int) -> np.ndarray:
    return decode_weight_matrix(
        np.asarray(layer.codes[expert]),
        np.asarray(layer.scales[expert]),
        code_bits=layer.code_bits,
        codebook=np.asarray(layer.codebook),
    )


def _artifact_input_for_layer(layer, values: np.ndarray) -> np.ndarray:
    signs = getattr(layer, "rht_signs", None)
    if signs is None:
        return values
    return apply_rht_np(values, np.asarray(signs, dtype=np.int8))


@dataclass(frozen=True)
class _OracleOutputs:
    artifact_gate: np.ndarray
    artifact_up: np.ndarray
    artifact_down_inputs: np.ndarray
    artifact_routes: np.ndarray
    source_gate: np.ndarray
    source_up: np.ndarray
    source_routes: np.ndarray
    source_tensors_read: int
    peak_source_tensor_bytes: int


def select_input_state_rows(
    states: np.ndarray | mx.array,
    *,
    tokens: int,
    position: str = "tail",
) -> tuple[np.ndarray, tuple[int, ...]]:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    values = np.asarray(states, dtype=np.float32)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError(f"batched states must have batch size 1, found {values.shape[0]}")
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"states must have shape [tokens, hidden] or [1, tokens, hidden], found {values.shape}")
    row_count = values.shape[0]
    selected_count = min(tokens, row_count)
    if position == "head":
        start = 0
    elif position == "tail":
        start = row_count - selected_count
    else:
        raise ValueError("position must be 'head' or 'tail'")
    end = start + selected_count
    indices = tuple(range(start, end))
    return values[start:end].astype(np.float32, copy=False), indices


def capture_glm45_air_moe_input_states(
    model: GLM45AirVQModel,
    input_token_ids: list[int] | tuple[int, ...],
    *,
    layer: int,
) -> np.ndarray:
    if not input_token_ids:
        raise ValueError("input_token_ids must not be empty")
    if layer < 0 or layer >= len(model.layers):
        raise ValueError(f"layer {layer} is outside model layer range [0, {len(model.layers)})")
    target_layer = model.layers[layer]
    if not isinstance(target_layer.mlp, GLM45AirVQMoE):
        raise ValueError(f"layer {layer} is not a sparse GLM-4.5-Air MoE layer")

    inputs = mx.array([list(input_token_ids)], dtype=mx.int32)
    h = model.model.embed_tokens(inputs)
    cache = [None] * len(model.layers)
    mask = create_attention_mask(h, cache[0])

    for layer_idx, layer_module in enumerate(model.layers):
        if layer_idx < layer:
            h = layer_module(h, mask, cache[layer_idx])
            continue
        attention = layer_module.self_attn(layer_module.input_layernorm(h), mask, cache[layer_idx])
        h = h + attention
        moe_input = layer_module.post_attention_layernorm(h)
        mx.eval(moe_input)
        return np.asarray(moe_input[0].astype(mx.float32), dtype=np.float32)

    raise RuntimeError("target layer was not reached")


def _compute_oracles(
    *,
    source_dir: Path,
    index: SafetensorsIndex,
    switch_glu: QuantizedVQSwitchGLU,
    layer: int,
    x: np.ndarray,
    indices: np.ndarray,
    source_tensors_already_read: int,
    peak_source_tensor_bytes: int,
) -> _OracleOutputs:
    tokens, top_k = indices.shape
    artifact_gate = np.empty((tokens, top_k, switch_glu.hidden_dims), dtype=np.float32)
    artifact_up = np.empty_like(artifact_gate)
    artifact_down_inputs = np.empty((tokens, top_k, switch_glu.down_proj.input_dims), dtype=np.float32)
    artifact_routes = np.empty((tokens, top_k, switch_glu.input_dims), dtype=np.float32)
    source_gate = np.empty_like(artifact_gate)
    source_up = np.empty_like(artifact_gate)
    source_routes = np.empty_like(artifact_routes)
    source_tensors_read = source_tensors_already_read
    peak_source_bytes = peak_source_tensor_bytes

    for expert in sorted(int(value) for value in np.unique(indices)):
        route_positions = np.argwhere(indices == expert)
        if route_positions.size == 0:
            continue

        source_gate_weight = _read_named_tensor(
            source_dir,
            index,
            _expert_weight_name(layer, expert, "gate_proj"),
        )
        source_up_weight = _read_named_tensor(
            source_dir,
            index,
            _expert_weight_name(layer, expert, "up_proj"),
        )
        source_down_weight = _read_named_tensor(
            source_dir,
            index,
            _expert_weight_name(layer, expert, "down_proj"),
        )
        source_tensors_read += 3
        peak_source_bytes = max(
            peak_source_bytes,
            source_gate_weight.nbytes,
            source_up_weight.nbytes,
            source_down_weight.nbytes,
        )

        artifact_gate_weight = _decode_expert_weight(switch_glu.gate_proj, expert)
        artifact_up_weight = _decode_expert_weight(switch_glu.up_proj, expert)
        artifact_down_weight = _decode_expert_weight(switch_glu.down_proj, expert)

        for token_idx, route_idx in route_positions:
            token = x[token_idx]

            src_gate = source_gate_weight @ token
            src_up = source_up_weight @ token
            source_gate[token_idx, route_idx] = src_gate
            source_up[token_idx, route_idx] = src_up
            source_routes[token_idx, route_idx] = source_down_weight @ (_silu_np(src_gate) * src_up)

            art_gate = artifact_gate_weight @ _artifact_input_for_layer(switch_glu.gate_proj, token)
            art_up = artifact_up_weight @ _artifact_input_for_layer(switch_glu.up_proj, token)
            artifact_gate[token_idx, route_idx] = art_gate
            artifact_up[token_idx, route_idx] = art_up
            artifact_hidden = _artifact_input_for_layer(switch_glu.down_proj, _silu_np(art_gate) * art_up)
            artifact_down_inputs[token_idx, route_idx] = artifact_hidden
            artifact_routes[token_idx, route_idx] = artifact_down_weight @ artifact_hidden

    return _OracleOutputs(
        artifact_gate=artifact_gate,
        artifact_up=artifact_up,
        artifact_down_inputs=artifact_down_inputs,
        artifact_routes=artifact_routes,
        source_gate=source_gate,
        source_up=source_up,
        source_routes=source_routes,
        source_tensors_read=source_tensors_read,
        peak_source_tensor_bytes=peak_source_bytes,
    )


def validate_glm45_air_vq(
    config: dict[str, Any],
    *,
    source_dir: str | Path,
    index_path: str | Path,
    artifact_dir: str | Path,
    model_id: str = GLM45_AIR_MODEL_ID,
    layer: int = 1,
    tokens: int = 2,
    seed: int = 20260624,
    input_scale: float = 1.0,
    input_states: np.ndarray | None = None,
    input_source: str | None = None,
    prompt_text: str | None = None,
    prompt_token_ids: tuple[int, ...] = (),
    input_state_indices: tuple[int, ...] | None = None,
    strict_config: bool = True,
) -> GLM45AirVQValidationResult:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if input_scale <= 0:
        raise ValueError("input_scale must be positive")

    summary = summarize_config(config, model_id=model_id)
    if strict_config:
        failed_checks = validate_glm45_air(summary)
        if failed_checks:
            raise ValueError(f"config does not match GLM-4.5-Air: {', '.join(failed_checks)}")
    if summary.model_type != "glm4_moe":
        raise ValueError(f"expected model_type='glm4_moe', found {summary.model_type!r}")
    if layer < summary.dense_layers or layer >= summary.num_hidden_layers:
        raise ValueError(
            f"layer {layer} is not a sparse MoE layer; sparse range is "
            f"[{summary.dense_layers}, {summary.num_hidden_layers})"
        )

    source_root = Path(source_dir)
    artifact_root = Path(artifact_dir)
    index = load_safetensors_index(index_path)
    routing = _routing_config(config, model_id=model_id)

    gate_weight_name = f"model.layers.{layer}.mlp.gate.weight"
    correction_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    gate_weight = _read_named_tensor(source_root, index, gate_weight_name)
    source_tensors_read = 1
    peak_source_bytes = gate_weight.nbytes
    if gate_weight.shape != (summary.n_routed_experts, summary.hidden_size):
        raise ValueError(
            f"{gate_weight_name} must have shape "
            f"({summary.n_routed_experts}, {summary.hidden_size}), found {gate_weight.shape}"
        )

    if correction_name in index.weight_map:
        correction = _read_named_tensor(source_root, index, correction_name)
        source_tensors_read += 1
        peak_source_bytes = max(peak_source_bytes, correction.nbytes)
    else:
        correction = np.zeros((summary.n_routed_experts,), dtype=np.float32)
    if correction.shape != (summary.n_routed_experts,):
        raise ValueError(f"{correction_name} must have shape ({summary.n_routed_experts},), found {correction.shape}")

    if input_states is None:
        rng = np.random.default_rng(seed)
        x = rng.normal(loc=0.0, scale=input_scale, size=(tokens, summary.hidden_size)).astype(np.float32)
        resolved_input_source = input_source or "random_normal"
        resolved_state_indices = tuple(range(tokens))
    else:
        x = np.asarray(input_states, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"input_states must have shape [tokens, hidden], found {x.shape}")
        if x.shape[0] <= 0:
            raise ValueError("input_states must include at least one token row")
        if x.shape[1] != summary.hidden_size:
            raise ValueError(
                f"input_states hidden dimension must be {summary.hidden_size}, found {x.shape[1]}"
            )
        tokens = int(x.shape[0])
        resolved_input_source = input_source or "provided"
        resolved_state_indices = input_state_indices or tuple(range(tokens))
        if len(resolved_state_indices) != tokens:
            raise ValueError(
                "input_state_indices length must match input_states token rows, "
                f"found {len(resolved_state_indices)} and {tokens}"
            )
    x_mx = mx.array(x)

    gate = GLM4MoEGate(
        routing,
        weight=mx.array(gate_weight),
        e_score_correction_bias=mx.array(correction),
    )
    indices_mx, scores_mx = gate(x_mx)
    mx.eval(indices_mx, scores_mx)
    indices = np.asarray(indices_mx, dtype=np.int64)
    scores = np.asarray(scores_mx.astype(mx.float32), dtype=np.float32)
    if indices.ndim != 2 or indices.shape[0] != tokens:
        raise ValueError(f"router indices must have shape [tokens, top_k], found {indices.shape}")

    switch_glu = _load_switch_glu(artifact_root, layer)
    route_indices_mx = mx.array(indices.astype(np.int32, copy=False))
    vq_gate_mx = switch_glu.gate_proj(x_mx, route_indices_mx)
    vq_up_mx = switch_glu.up_proj(x_mx, route_indices_mx)
    vq_routes_mx = switch_glu.down_proj(nn.silu(vq_gate_mx) * vq_up_mx, route_indices_mx)
    vq_weighted_mx = (vq_routes_mx * mx.array(scores)[..., None]).sum(axis=-2)
    mx.eval(vq_gate_mx, vq_up_mx, vq_routes_mx, vq_weighted_mx)

    vq_gate = np.asarray(vq_gate_mx.astype(mx.float32), dtype=np.float32)
    vq_up = np.asarray(vq_up_mx.astype(mx.float32), dtype=np.float32)
    vq_routes = np.asarray(vq_routes_mx.astype(mx.float32), dtype=np.float32)
    vq_weighted = np.asarray(vq_weighted_mx.astype(mx.float32), dtype=np.float32)

    oracles = _compute_oracles(
        source_dir=source_root,
        index=index,
        switch_glu=switch_glu,
        layer=layer,
        x=x,
        indices=indices,
        source_tensors_already_read=source_tensors_read,
        peak_source_tensor_bytes=peak_source_bytes,
    )
    artifact_weighted = (oracles.artifact_routes * scores[..., None]).sum(axis=-2)
    source_weighted = (oracles.source_routes * scores[..., None]).sum(axis=-2)

    metrics = {
        "artifact_gate_proj": _array_metrics(vq_gate, oracles.artifact_gate),
        "artifact_up_proj": _array_metrics(vq_up, oracles.artifact_up),
        "artifact_routed_glu": _array_metrics(vq_routes, oracles.artifact_routes),
        "artifact_weighted_routed": _array_metrics(vq_weighted, artifact_weighted),
        "source_gate_proj": _array_metrics(vq_gate, oracles.source_gate),
        "source_up_proj": _array_metrics(vq_up, oracles.source_up),
        "source_routed_glu": _array_metrics(vq_routes, oracles.source_routes),
        "source_weighted_routed": _array_metrics(vq_weighted, source_weighted),
    }
    route_source_metrics = _route_source_metric_records(
        indices=indices,
        scores=scores,
        vq_gate=vq_gate,
        vq_up=vq_up,
        vq_routes=vq_routes,
        source_gate=oracles.source_gate,
        source_up=oracles.source_up,
        source_routes=oracles.source_routes,
    )
    route_source_residual_topk = _route_source_residual_topk_records(
        indices=indices,
        scores=scores,
        vq_gate=vq_gate,
        vq_up=vq_up,
        vq_routes=vq_routes,
        source_gate=oracles.source_gate,
        source_up=oracles.source_up,
        source_routes=oracles.source_routes,
    )
    route_source_sparse_residual_plans = _route_source_sparse_residual_plan_records(
        indices=indices,
        scores=scores,
        artifact_down_inputs=oracles.artifact_down_inputs,
        vq_routes=vq_routes,
        source_routes=oracles.source_routes,
    )

    return GLM45AirVQValidationResult(
        model_id=model_id,
        layer=layer,
        tokens=tokens,
        input_source=resolved_input_source,
        prompt_text=prompt_text,
        prompt_token_ids=tuple(int(token) for token in prompt_token_ids),
        input_state_indices=tuple(int(index) for index in resolved_state_indices),
        hidden_size=summary.hidden_size,
        moe_intermediate_size=summary.moe_intermediate_size,
        top_k=indices.shape[1],
        selected_experts=tuple(sorted(int(value) for value in np.unique(indices))),
        routes=int(indices.size),
        source_tensors_read=oracles.source_tensors_read,
        peak_source_tensor_bytes=oracles.peak_source_tensor_bytes,
        source_dir=str(source_root),
        artifact_dir=str(artifact_root),
        routed_indices=tuple(tuple(int(value) for value in row) for row in indices),
        router_scores=tuple(tuple(float(value) for value in row) for row in scores),
        route_source_metrics=route_source_metrics,
        route_source_residual_topk=route_source_residual_topk,
        route_source_sparse_residual_plans=route_source_sparse_residual_plans,
        metrics=metrics,
    )
