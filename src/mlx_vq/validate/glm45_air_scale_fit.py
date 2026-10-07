from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import CODEWORD_DIM, decode_weight_matrix
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, summarize_config, validate_glm45_air
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.models.glm4_moe_adapter import GLM4MoEGate
from mlx_vq.validate.glm45_air_vq import (
    ArrayComparisonMetrics,
    _array_metrics,
    _expert_weight_name,
    _load_switch_glu,
    _read_named_tensor,
    _routing_config,
    _silu_np,
)


@dataclass(frozen=True)
class FittedScaleStats:
    group_size: int
    groups: int
    scale_min: float
    scale_max: float
    scale_mean: float
    scale_std: float
    negative_fraction: float
    zero_denominator_count: int
    weight_cosine: float
    weight_rel_l2: float


@dataclass(frozen=True)
class FittedScaleWeight:
    scales: np.ndarray
    weight: np.ndarray
    stats: FittedScaleStats


@dataclass(frozen=True)
class GLM45AirScaleFitResult:
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
    allow_negative_scales: bool
    source_tensors_read: int
    peak_source_tensor_bytes: int
    source_dir: str
    artifact_dir: str
    routed_indices: tuple[tuple[int, ...], ...]
    router_scores: tuple[tuple[float, ...], ...]
    fit_stats: dict[str, FittedScaleStats]
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
            "allow_negative_scales": self.allow_negative_scales,
            "source_tensors_read": self.source_tensors_read,
            "peak_source_tensor_bytes": self.peak_source_tensor_bytes,
            "source_dir": self.source_dir,
            "artifact_dir": self.artifact_dir,
            "routed_indices": [list(row) for row in self.routed_indices],
            "router_scores": [list(row) for row in self.router_scores],
            "fit_stats": {name: asdict(stats) for name, stats in self.fit_stats.items()},
            "metrics": {name: asdict(metrics) for name, metrics in self.metrics.items()},
        }


def fit_e8_group_scales(
    source_weight: np.ndarray,
    codes: np.ndarray,
    codebook: np.ndarray,
    *,
    group_size: int,
    code_bits: int = 8,
    allow_negative_scales: bool = True,
    eps: float = 1e-12,
) -> FittedScaleWeight:
    """Fit per-row/per-group scales while freezing the existing E8 codes.

    This is an optimistic diagnostic for the quality backlog. With
    ``allow_negative_scales=True`` it is a least-squares upper bound for a
    scale-only P-step, not a proposed artifact policy.
    """

    source = np.asarray(source_weight, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"source_weight must be 2D [out, in], found {source.shape}")
    if source.shape[1] % group_size != 0:
        raise ValueError("source_weight input dimension must be divisible by group_size")
    if group_size % CODEWORD_DIM != 0:
        raise ValueError("group_size must be divisible by 8")
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")

    base = decode_weight_matrix(
        np.asarray(codes),
        scales=None,
        code_bits=code_bits,
        codebook=np.asarray(codebook),
    ).astype(np.float32, copy=False)
    if base.shape != source.shape:
        raise ValueError(f"decoded code shape {base.shape} does not match source shape {source.shape}")

    out_dim, in_dim = source.shape
    groups = in_dim // group_size
    base_groups = base.reshape(out_dim, groups, group_size).astype(np.float64)
    source_groups = source.reshape(out_dim, groups, group_size).astype(np.float64)

    numerators = np.sum(base_groups * source_groups, axis=2)
    denominators = np.sum(base_groups * base_groups, axis=2)
    scales = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=np.float64),
        where=denominators > eps,
    )
    if not allow_negative_scales:
        scales = np.maximum(scales, 0.0)

    fitted = (base_groups * scales[:, :, None]).reshape(out_dim, in_dim).astype(np.float32)
    diff = fitted.astype(np.float64) - source.astype(np.float64)
    source_norm = np.linalg.norm(source.astype(np.float64).reshape(-1))
    stats = FittedScaleStats(
        group_size=int(group_size),
        groups=int(groups),
        scale_min=float(np.min(scales)) if scales.size else 0.0,
        scale_max=float(np.max(scales)) if scales.size else 0.0,
        scale_mean=float(np.mean(scales)) if scales.size else 0.0,
        scale_std=float(np.std(scales)) if scales.size else 0.0,
        negative_fraction=float(np.mean(scales < 0.0)) if scales.size else 0.0,
        zero_denominator_count=int(np.count_nonzero(denominators <= eps)),
        weight_cosine=_array_metrics(fitted, source).cosine,
        weight_rel_l2=float(np.linalg.norm(diff.reshape(-1)) / max(float(source_norm), eps)),
    )
    return FittedScaleWeight(
        scales=scales.astype(np.float32),
        weight=fitted,
        stats=stats,
    )


def _select_or_random_input_states(
    *,
    summary_hidden_size: int,
    tokens: int,
    seed: int,
    input_scale: float,
    input_states: np.ndarray | None,
    input_source: str | None,
    input_state_indices: tuple[int, ...] | None,
) -> tuple[np.ndarray, int, str, tuple[int, ...]]:
    if input_states is None:
        rng = np.random.default_rng(seed)
        x = rng.normal(loc=0.0, scale=input_scale, size=(tokens, summary_hidden_size)).astype(np.float32)
        return x, tokens, input_source or "random_normal", tuple(range(tokens))

    x = np.asarray(input_states, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"input_states must have shape [tokens, hidden], found {x.shape}")
    if x.shape[0] <= 0:
        raise ValueError("input_states must include at least one token row")
    if x.shape[1] != summary_hidden_size:
        raise ValueError(f"input_states hidden dimension must be {summary_hidden_size}, found {x.shape[1]}")
    resolved_indices = input_state_indices or tuple(range(x.shape[0]))
    if len(resolved_indices) != x.shape[0]:
        raise ValueError(
            "input_state_indices length must match input_states token rows, "
            f"found {len(resolved_indices)} and {x.shape[0]}"
        )
    return x, int(x.shape[0]), input_source or "provided", tuple(int(value) for value in resolved_indices)


def _fit_projection_for_expert(layer, source_weight: np.ndarray, expert: int, *, allow_negative_scales: bool):
    return fit_e8_group_scales(
        source_weight,
        np.asarray(layer.codes[expert]),
        np.asarray(layer.codebook),
        group_size=layer.group_size,
        code_bits=layer.code_bits,
        allow_negative_scales=allow_negative_scales,
    )


def evaluate_glm45_air_scale_fit(
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
    allow_negative_scales: bool = True,
    strict_config: bool = True,
) -> GLM45AirScaleFitResult:
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
    if correction_name in index.weight_map:
        correction = _read_named_tensor(source_root, index, correction_name)
        source_tensors_read += 1
        peak_source_bytes = max(peak_source_bytes, correction.nbytes)
    else:
        correction = np.zeros((summary.n_routed_experts,), dtype=np.float32)

    x, tokens, resolved_input_source, resolved_state_indices = _select_or_random_input_states(
        summary_hidden_size=summary.hidden_size,
        tokens=tokens,
        seed=seed,
        input_scale=input_scale,
        input_states=input_states,
        input_source=input_source,
        input_state_indices=input_state_indices,
    )

    gate = GLM4MoEGate(
        routing,
        weight=mx.array(gate_weight),
        e_score_correction_bias=mx.array(correction),
    )
    indices_mx, scores_mx = gate(mx.array(x))
    mx.eval(indices_mx, scores_mx)
    indices = np.asarray(indices_mx, dtype=np.int64)
    scores = np.asarray(scores_mx.astype(mx.float32), dtype=np.float32)
    if indices.ndim != 2 or indices.shape[0] != tokens:
        raise ValueError(f"router indices must have shape [tokens, top_k], found {indices.shape}")

    switch_glu = _load_switch_glu(artifact_root, layer)
    top_k = indices.shape[1]
    current_gate = np.empty((tokens, top_k, switch_glu.hidden_dims), dtype=np.float32)
    current_up = np.empty_like(current_gate)
    current_routes = np.empty((tokens, top_k, switch_glu.input_dims), dtype=np.float32)
    fitted_gate = np.empty_like(current_gate)
    fitted_up = np.empty_like(current_gate)
    fitted_routes = np.empty_like(current_routes)
    source_gate = np.empty_like(current_gate)
    source_up = np.empty_like(current_gate)
    source_routes = np.empty_like(current_routes)
    fit_stats: dict[str, FittedScaleStats] = {}

    for expert in sorted(int(value) for value in np.unique(indices)):
        route_positions = np.argwhere(indices == expert)
        if route_positions.size == 0:
            continue

        source_gate_weight = _read_named_tensor(source_root, index, _expert_weight_name(layer, expert, "gate_proj"))
        source_up_weight = _read_named_tensor(source_root, index, _expert_weight_name(layer, expert, "up_proj"))
        source_down_weight = _read_named_tensor(source_root, index, _expert_weight_name(layer, expert, "down_proj"))
        source_tensors_read += 3
        peak_source_bytes = max(
            peak_source_bytes,
            source_gate_weight.nbytes,
            source_up_weight.nbytes,
            source_down_weight.nbytes,
        )

        current_gate_weight = decode_weight_matrix(
            np.asarray(switch_glu.gate_proj.codes[expert]),
            np.asarray(switch_glu.gate_proj.scales[expert]),
            code_bits=switch_glu.gate_proj.code_bits,
            codebook=np.asarray(switch_glu.gate_proj.codebook),
        )
        current_up_weight = decode_weight_matrix(
            np.asarray(switch_glu.up_proj.codes[expert]),
            np.asarray(switch_glu.up_proj.scales[expert]),
            code_bits=switch_glu.up_proj.code_bits,
            codebook=np.asarray(switch_glu.up_proj.codebook),
        )
        current_down_weight = decode_weight_matrix(
            np.asarray(switch_glu.down_proj.codes[expert]),
            np.asarray(switch_glu.down_proj.scales[expert]),
            code_bits=switch_glu.down_proj.code_bits,
            codebook=np.asarray(switch_glu.down_proj.codebook),
        )
        fitted_gate_weight = _fit_projection_for_expert(
            switch_glu.gate_proj,
            source_gate_weight,
            expert,
            allow_negative_scales=allow_negative_scales,
        )
        fitted_up_weight = _fit_projection_for_expert(
            switch_glu.up_proj,
            source_up_weight,
            expert,
            allow_negative_scales=allow_negative_scales,
        )
        fitted_down_weight = _fit_projection_for_expert(
            switch_glu.down_proj,
            source_down_weight,
            expert,
            allow_negative_scales=allow_negative_scales,
        )
        fit_stats[f"expert_{expert}.gate_proj"] = fitted_gate_weight.stats
        fit_stats[f"expert_{expert}.up_proj"] = fitted_up_weight.stats
        fit_stats[f"expert_{expert}.down_proj"] = fitted_down_weight.stats

        for token_idx, route_idx in route_positions:
            token = x[token_idx]

            src_gate = source_gate_weight @ token
            src_up = source_up_weight @ token
            source_gate[token_idx, route_idx] = src_gate
            source_up[token_idx, route_idx] = src_up
            source_routes[token_idx, route_idx] = source_down_weight @ (_silu_np(src_gate) * src_up)

            cur_gate = current_gate_weight @ token
            cur_up = current_up_weight @ token
            current_gate[token_idx, route_idx] = cur_gate
            current_up[token_idx, route_idx] = cur_up
            current_routes[token_idx, route_idx] = current_down_weight @ (_silu_np(cur_gate) * cur_up)

            fit_gate = fitted_gate_weight.weight @ token
            fit_up = fitted_up_weight.weight @ token
            fitted_gate[token_idx, route_idx] = fit_gate
            fitted_up[token_idx, route_idx] = fit_up
            fitted_routes[token_idx, route_idx] = fitted_down_weight.weight @ (_silu_np(fit_gate) * fit_up)

    current_weighted = (current_routes * scores[..., None]).sum(axis=-2)
    fitted_weighted = (fitted_routes * scores[..., None]).sum(axis=-2)
    source_weighted = (source_routes * scores[..., None]).sum(axis=-2)

    metrics = {
        "current_gate_proj": _array_metrics(current_gate, source_gate),
        "fitted_gate_proj": _array_metrics(fitted_gate, source_gate),
        "current_up_proj": _array_metrics(current_up, source_up),
        "fitted_up_proj": _array_metrics(fitted_up, source_up),
        "current_routed_glu": _array_metrics(current_routes, source_routes),
        "fitted_routed_glu": _array_metrics(fitted_routes, source_routes),
        "current_weighted_routed": _array_metrics(current_weighted, source_weighted),
        "fitted_weighted_routed": _array_metrics(fitted_weighted, source_weighted),
        "fitted_vs_current_weighted_routed": _array_metrics(fitted_weighted, current_weighted),
    }

    return GLM45AirScaleFitResult(
        model_id=model_id,
        layer=layer,
        tokens=tokens,
        input_source=resolved_input_source,
        prompt_text=prompt_text,
        prompt_token_ids=tuple(int(token) for token in prompt_token_ids),
        input_state_indices=tuple(int(index) for index in resolved_state_indices),
        hidden_size=summary.hidden_size,
        moe_intermediate_size=summary.moe_intermediate_size,
        top_k=top_k,
        selected_experts=tuple(sorted(int(value) for value in np.unique(indices))),
        routes=int(indices.size),
        allow_negative_scales=allow_negative_scales,
        source_tensors_read=source_tensors_read,
        peak_source_tensor_bytes=peak_source_bytes,
        source_dir=str(source_root),
        artifact_dir=str(artifact_root),
        routed_indices=tuple(tuple(int(value) for value in row) for row in indices),
        router_scores=tuple(tuple(float(value) for value in row) for row in scores),
        fit_stats=fit_stats,
        metrics=metrics,
    )
