from __future__ import annotations

import json
import re
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.layers.distributed import sum_gradients
from mlx.utils import tree_flatten

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.glm4_moe import Attention, MLP, ModelArgs, MoEGate, group_expert_select
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)

from mlx_vq.io.continuous_sidecar import load_switch_linear_continuous_sidecar
from mlx_vq.io.load import inspect_safetensors, load_switch_linear_projection
from mlx_vq.io.logit_bias import load_logit_bias_sidecar
from mlx_vq.io.router_correction import load_router_correction_sidecar
from mlx_vq.io.sparse_residual import load_switch_linear_sparse_residual_rows
from mlx_vq.io.source_safetensors import read_indexed_safetensors_tensor_mlx
from mlx_vq.models.glm4_moe_adapter import QuantizedVQSwitchGLU
from mlx_vq.ops.vq_switch import _inverse_permutation, gather_vqmm_sorted_routes


_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")
_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class GLM45AirNonExpertBindReport:
    loaded_model_parameters: tuple[str, ...]
    skipped_routed_expert_tensors: tuple[str, ...]
    skipped_mtp_tensors: tuple[str, ...]
    skipped_unmatched_tensors: tuple[str, ...]
    missing_model_parameters: tuple[str, ...]
    requested_surfaces: tuple[str, ...] = ()
    skipped_by_surface_tensors: tuple[str, ...] = ()

    @property
    def loaded_count(self) -> int:
        return len(self.loaded_model_parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "loaded_model_parameters": list(self.loaded_model_parameters),
            "loaded_count": self.loaded_count,
            "skipped_routed_expert_tensors": list(self.skipped_routed_expert_tensors),
            "skipped_mtp_tensors": list(self.skipped_mtp_tensors),
            "skipped_unmatched_tensors": list(self.skipped_unmatched_tensors),
            "missing_model_parameters": list(self.missing_model_parameters),
            "requested_surfaces": list(self.requested_surfaces),
            "skipped_by_surface_tensors": list(self.skipped_by_surface_tensors),
        }


@dataclass(frozen=True)
class _GLM45AirBindEntry:
    source_name: str
    target_name: str


def glm45_air_args_from_config(config: dict[str, Any]) -> ModelArgs:
    field_names = {field.name for field in fields(ModelArgs)}
    kwargs = {name: config[name] for name in field_names if name in config}
    return ModelArgs(**kwargs)


def is_glm45_air_sparse_layer(config: ModelArgs, layer_idx: int) -> bool:
    return config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.match(name)
    return None if match is None else int(match.group("layer"))


def _is_mtp_tensor(name: str, args: ModelArgs) -> bool:
    layer_idx = _layer_index(name)
    return layer_idx is not None and layer_idx >= args.num_hidden_layers


def _is_routed_expert_weight(name: str) -> bool:
    return _EXPERT_WEIGHT_RE.match(name) is not None


def _is_vq_runtime_param(name: str) -> bool:
    return ".mlp.switch_mlp." in name


def _profile_section(recorder: Any, name: str):
    if recorder is None:
        return nullcontext()
    return recorder.time(name)


def _profile_eval(recorder: Any, *values: Any) -> None:
    if recorder is not None and getattr(recorder, "eval_outputs", False):
        mx.eval(*values)


class GLM45AirVQMoE(nn.Module):
    def __init__(self, config: ModelArgs, *, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp: QuantizedVQSwitchGLU | None = None
        self.profile_recorder = None
        self.gate = MoEGate(config)
        self.router_expert_bias_delta: mx.array | None = None
        self.router_temperature: mx.array | None = None
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP(config=config, intermediate_size=intermediate_size)
        self.sharding_group = None

    def bind_switch_mlp(self, switch_mlp: QuantizedVQSwitchGLU) -> None:
        if switch_mlp.input_dims != self.config.hidden_size:
            raise ValueError("switch_mlp input dimension must match config.hidden_size")
        if switch_mlp.hidden_dims != self.config.moe_intermediate_size:
            raise ValueError("switch_mlp hidden dimension must match config.moe_intermediate_size")
        if switch_mlp.num_experts != self.config.n_routed_experts:
            raise ValueError("switch_mlp expert count must match config.n_routed_experts")
        self.switch_mlp = switch_mlp

    @property
    def has_router_correction(self) -> bool:
        return self.router_expert_bias_delta is not None or self.router_temperature is not None

    def set_router_correction(
        self,
        *,
        expert_bias_delta: mx.array | None = None,
        temperature: mx.array | None = None,
    ) -> None:
        if expert_bias_delta is not None and expert_bias_delta.shape != (self.config.n_routed_experts,):
            raise ValueError(
                "router expert_bias_delta must have shape "
                f"({self.config.n_routed_experts},), found {expert_bias_delta.shape}"
            )
        if temperature is not None:
            if temperature.shape not in {(), (1,)}:
                raise ValueError(f"router temperature must be scalar or shape (1,), found {temperature.shape}")
            if float(temperature.item()) <= 0.0:
                raise ValueError("router temperature must be positive")
        self.router_expert_bias_delta = (
            None if expert_bias_delta is None else expert_bias_delta.astype(mx.float32)
        )
        self.router_temperature = None if temperature is None else temperature.astype(mx.float32)

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        if not self.has_router_correction:
            return self.gate(x)
        gates = x @ self.gate.weight.T
        if self.router_temperature is not None:
            gates = gates / self.router_temperature
        correction_bias = self.gate.e_score_correction_bias
        if self.router_expert_bias_delta is not None:
            correction_bias = correction_bias + self.router_expert_bias_delta
        return group_expert_select(
            gates,
            correction_bias,
            self.gate.top_k,
            self.gate.n_group,
            self.gate.topk_group,
            self.gate.routed_scaling_factor,
            self.gate.norm_topk_prob,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        if self.switch_mlp is None:
            raise RuntimeError("VQ switch_mlp must be bound before running a sparse GLM-4.5-Air MoE layer")

        recorder = self.profile_recorder
        prefix = f"layer.{self.layer_idx}.moe"
        with _profile_section(recorder, f"{prefix}.router"):
            inds, scores = self.route(x)
            _profile_eval(recorder, inds, scores)
        if recorder is None:
            y = self.switch_mlp(x, inds)
        elif isinstance(self.switch_mlp, QuantizedVQSwitchGLU):
            if self.switch_mlp._can_use_shared_sorted_route_prefill(x, inds):
                with _profile_section(recorder, f"{prefix}.route_sort"):
                    leading_shape = x.shape[:-1]
                    top_k = inds.shape[-1]
                    flat_x = x.reshape((-1, self.switch_mlp.input_dims))
                    flat_rhs = inds.reshape((-1,)).astype(mx.int32)
                    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // top_k).astype(mx.int32)
                    order = mx.argsort(flat_rhs)
                    inverse_order = _inverse_permutation(order)
                    sorted_rhs = flat_rhs[order]
                    sorted_lhs = flat_lhs[order]
                    implementation = self.switch_mlp._sorted_prefill_implementation(
                        route_count=flat_rhs.shape[0],
                        activation_dtype=flat_x.dtype,
                    )
                    tile_descriptors = self.switch_mlp._shared_sorted_tile_descriptors(
                        sorted_rhs,
                        implementation,
                    )
                    _profile_eval(recorder, sorted_rhs, sorted_lhs, inverse_order)
                with _profile_section(recorder, f"{prefix}.gate_proj"):
                    gate_input = self.switch_mlp._rht_input_for_projection(self.switch_mlp.gate_proj, flat_x)
                    gate = gather_vqmm_sorted_routes(
                        gate_input,
                        self.switch_mlp.gate_proj.codes,
                        self.switch_mlp.gate_proj.effective_scales(),
                        self.switch_mlp.gate_proj.codebook,
                        sorted_rhs,
                        sorted_lhs,
                        input_dims=self.switch_mlp.gate_proj.input_dims,
                        output_dims=self.switch_mlp.gate_proj.output_dims,
                        group_size=self.switch_mlp.gate_proj.group_size,
                        code_bits=self.switch_mlp.gate_proj.code_bits,
                        codebook_duplication=self.switch_mlp.gate_proj.gather_codebook_duplication,
                        implementation=implementation,
                        projection="gate_up",
                        tile_descriptors=tile_descriptors,
                    )
                    gate = self.switch_mlp._add_sorted_bias(gate, self.switch_mlp.gate_proj, sorted_rhs)
                    _profile_eval(recorder, gate)
                with _profile_section(recorder, f"{prefix}.up_proj"):
                    up_input = self.switch_mlp._rht_input_for_projection(self.switch_mlp.up_proj, flat_x)
                    up = gather_vqmm_sorted_routes(
                        up_input,
                        self.switch_mlp.up_proj.codes,
                        self.switch_mlp.up_proj.effective_scales(),
                        self.switch_mlp.up_proj.codebook,
                        sorted_rhs,
                        sorted_lhs,
                        input_dims=self.switch_mlp.up_proj.input_dims,
                        output_dims=self.switch_mlp.up_proj.output_dims,
                        group_size=self.switch_mlp.up_proj.group_size,
                        code_bits=self.switch_mlp.up_proj.code_bits,
                        codebook_duplication=self.switch_mlp.up_proj.gather_codebook_duplication,
                        implementation=implementation,
                        projection="gate_up",
                        tile_descriptors=tile_descriptors,
                    )
                    up = self.switch_mlp._add_sorted_bias(up, self.switch_mlp.up_proj, sorted_rhs)
                    _profile_eval(recorder, up)
                hidden = nn.silu(gate) * up
                with _profile_section(recorder, f"{prefix}.down_proj"):
                    down_input = self.switch_mlp._rht_input_for_projection(self.switch_mlp.down_proj, hidden)
                    y = gather_vqmm_sorted_routes(
                        down_input,
                        self.switch_mlp.down_proj.codes,
                        self.switch_mlp.down_proj.effective_scales(),
                        self.switch_mlp.down_proj.codebook,
                        sorted_rhs,
                        mx.arange(flat_rhs.shape[0], dtype=mx.int32),
                        input_dims=self.switch_mlp.down_proj.input_dims,
                        output_dims=self.switch_mlp.down_proj.output_dims,
                        group_size=self.switch_mlp.down_proj.group_size,
                        code_bits=self.switch_mlp.down_proj.code_bits,
                        codebook_duplication=self.switch_mlp.down_proj.gather_codebook_duplication,
                        implementation=implementation,
                        projection="down",
                        tile_descriptors=tile_descriptors,
                        # Mirrors QuantizedVQSwitchGLU._shared_sorted_route_prefill:
                        # ``hidden`` is already in sorted-route order and the lhs is an
                        # explicit arange.
                        x_pre_sorted=True,
                    )
                    y = self.switch_mlp._add_sorted_bias(y, self.switch_mlp.down_proj, sorted_rhs)
                    y = y[inverse_order].reshape((*leading_shape, top_k, self.switch_mlp.down_proj.output_dims))
                    _profile_eval(recorder, y)
            else:
                with _profile_section(recorder, f"{prefix}.gate_proj"):
                    gate = self.switch_mlp.gate_proj(x, inds)
                    _profile_eval(recorder, gate)
                with _profile_section(recorder, f"{prefix}.up_proj"):
                    up = self.switch_mlp.up_proj(x, inds)
                    _profile_eval(recorder, up)
                hidden = nn.silu(gate) * up
                with _profile_section(recorder, f"{prefix}.down_proj"):
                    y = self.switch_mlp.down_proj(hidden, inds)
                    _profile_eval(recorder, y)
        elif isinstance(self.switch_mlp, GLM45AirMLXQuantizedSwitchGLU):
            expanded = mx.expand_dims(x, (-2, -3))
            do_sort = inds.size >= 64
            idx = inds
            inv_order = None
            if do_sort:
                expanded, idx, inv_order = _gather_sort(expanded, inds)
            with _profile_section(recorder, f"{prefix}.gate_proj"):
                gate = self.switch_mlp.gate_proj(expanded, idx, sorted_indices=do_sort)
                _profile_eval(recorder, gate)
            with _profile_section(recorder, f"{prefix}.up_proj"):
                up = self.switch_mlp.up_proj(expanded, idx, sorted_indices=do_sort)
                _profile_eval(recorder, up)
            hidden = nn.silu(gate) * up
            with _profile_section(recorder, f"{prefix}.down_proj"):
                y = self.switch_mlp.down_proj(hidden, idx, sorted_indices=do_sort)
                if do_sort:
                    y = _scatter_unsort(y, inv_order, inds.shape)
                y = y.squeeze(-2)
                _profile_eval(recorder, y)
        else:
            with _profile_section(recorder, f"{prefix}.switch_mlp"):
                y = self.switch_mlp(x, inds)
                _profile_eval(recorder, y)
        with _profile_section(recorder, f"{prefix}.weighted_reduce"):
            y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
            _profile_eval(recorder, y)
        shared_experts = self.get("shared_experts")
        if shared_experts is not None:
            with _profile_section(recorder, f"{prefix}.shared_experts"):
                y = y + shared_experts(x)
                _profile_eval(recorder, y)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class GLM45AirVQDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Attention(config)
        self.mlp = GLM45AirVQMoE(config, layer_idx=layer_idx) if is_glm45_air_sparse_layer(config, layer_idx) else MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.profile_recorder = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        recorder = self.profile_recorder
        with _profile_section(recorder, f"layer.{self.layer_idx}.attention"):
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            _profile_eval(recorder, r)
        h = x + r
        if isinstance(self.mlp, GLM45AirVQMoE):
            self.mlp.profile_recorder = recorder
            r = self.mlp(self.post_attention_layernorm(h))
        else:
            with _profile_section(recorder, f"layer.{self.layer_idx}.mlp"):
                r = self.mlp(self.post_attention_layernorm(h))
                _profile_eval(recorder, r)
        return h + r


class GLM45AirVQBackbone(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GLM45AirVQDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.profile_recorder = None

    @property
    def pipeline_layers(self):
        return self.layers

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        mask = create_attention_mask(h, cache[0])

        if self.pipeline_rank < self.pipeline_size - 1:
            h = mx.distributed.recv_like(h, (self.pipeline_rank + 1))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            layer.profile_recorder = self.profile_recorder
            h = layer(h, mask, layer_cache)

        if self.pipeline_rank != 0:
            h = mx.distributed.send(h, (self.pipeline_rank - 1) % self.pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        if self.pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class GLM45AirVQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = GLM45AirVQBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_logit_bias_token_ids: mx.array | None = None
        self.final_logit_bias_values: mx.array | None = None
        self.final_logit_bias_position_indices: tuple[int, ...] | None = None
        self.final_logit_bias_token_position_indices: tuple[tuple[int, ...] | None, ...] | None = None

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(key):
            return "e_score_correction_bias" not in key

        return predicate

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def set_profile_recorder(self, recorder: Any) -> None:
        self.model.profile_recorder = recorder
        for layer in self.layers:
            layer.profile_recorder = recorder
            if isinstance(layer.mlp, GLM45AirVQMoE):
                layer.mlp.profile_recorder = recorder

    def set_final_logit_bias(
        self,
        token_ids: mx.array,
        biases: mx.array,
        position_indices: tuple[int, ...] | None = None,
        token_position_indices: tuple[tuple[int, ...] | None, ...] | None = None,
    ) -> None:
        if token_ids.ndim != 1:
            raise ValueError(f"logit bias token_ids must be 1D, found {token_ids.shape}")
        if biases.ndim != 1:
            raise ValueError(f"logit bias biases must be 1D, found {biases.shape}")
        if token_ids.shape != biases.shape:
            raise ValueError("logit bias token_ids and biases must have matching shapes")
        if position_indices is not None and token_position_indices is not None:
            raise ValueError("logit bias position_indices and token_position_indices cannot both be set")
        ids = np.array(token_ids, dtype=np.int64)
        values = np.array(biases, dtype=np.float32)
        if np.any(ids < 0) or np.any(ids >= int(self.args.vocab_size)):
            raise ValueError(f"logit bias token ids must be within vocab_size={self.args.vocab_size}")
        if position_indices is not None:
            positions = tuple(int(position) for position in position_indices)
            if not positions:
                raise ValueError("logit bias position_indices must include at least one position")
            if len(set(positions)) != len(positions):
                raise ValueError("logit bias position_indices must be unique")
            if any(position < 0 for position in positions):
                raise ValueError("logit bias position_indices must be non-negative")
            self.final_logit_bias_position_indices = positions
        else:
            self.final_logit_bias_position_indices = None
        if token_position_indices is not None:
            if len(token_position_indices) != int(token_ids.shape[0]):
                raise ValueError("logit bias token_position_indices must match token_ids length")
            normalized_token_positions: list[tuple[int, ...] | None] = []
            for positions in token_position_indices:
                if positions is None:
                    normalized_token_positions.append(None)
                    continue
                normalized_positions = tuple(int(position) for position in positions)
                if not normalized_positions:
                    raise ValueError("logit bias token_position_indices values must include at least one position")
                if len(set(normalized_positions)) != len(normalized_positions):
                    raise ValueError("logit bias token_position_indices values must be unique per token")
                if any(position < 0 for position in normalized_positions):
                    raise ValueError("logit bias token_position_indices values must be non-negative")
                normalized_token_positions.append(normalized_positions)
            self.final_logit_bias_token_position_indices = tuple(normalized_token_positions)
        else:
            self.final_logit_bias_token_position_indices = None
        self.final_logit_bias_token_ids = mx.array(ids, dtype=mx.int32)
        self.final_logit_bias_values = mx.array(values, dtype=self.lm_head.weight.dtype)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        out = self.model(inputs, cache)
        logits = self.lm_head(out)
        if self.final_logit_bias_token_ids is not None and self.final_logit_bias_values is not None:
            if self.final_logit_bias_token_position_indices is not None:
                seq_len = int(logits.shape[1])
                for token_offset, positions in enumerate(self.final_logit_bias_token_position_indices):
                    token_id = self.final_logit_bias_token_ids[token_offset]
                    bias = self.final_logit_bias_values[token_offset]
                    if positions is None:
                        logits = logits.at[..., token_id].add(bias)
                        continue
                    for position in positions:
                        if position < seq_len:
                            logits = logits.at[:, position, token_id].add(bias)
            elif self.final_logit_bias_position_indices is None:
                logits = logits.at[..., self.final_logit_bias_token_ids].add(self.final_logit_bias_values)
            else:
                seq_len = int(logits.shape[1])
                for position in self.final_logit_bias_position_indices:
                    if position < seq_len:
                        logits = logits.at[:, position, self.final_logit_bias_token_ids].add(
                            self.final_logit_bias_values
                        )
        return logits


def load_glm45_air_vq_switch_glu(artifact_dir: str | Path, layer: int) -> QuantizedVQSwitchGLU:
    artifact_root = Path(artifact_dir)
    prefix = f"model.layers.{layer}.mlp.switch_mlp"
    gate_proj = load_switch_linear_projection(
        artifact_root / f"layer-{layer:05d}-gate_proj.safetensors",
        f"{prefix}.gate_proj",
    )
    up_proj = load_switch_linear_projection(
        artifact_root / f"layer-{layer:05d}-up_proj.safetensors",
        f"{prefix}.up_proj",
    )
    down_proj = load_switch_linear_projection(
        artifact_root / f"layer-{layer:05d}-down_proj.safetensors",
        f"{prefix}.down_proj",
    )
    for projection_name, projection in (
        ("gate_proj", gate_proj),
        ("up_proj", up_proj),
        ("down_proj", down_proj),
    ):
        if hasattr(projection, "scales") and hasattr(projection, "set_continuous_sidecar"):
            sidecar = load_switch_linear_continuous_sidecar(
                artifact_root,
                layer=layer,
                projection=projection_name,
                scale_shape=projection.scales.shape,
                output_bias_shape=(projection.num_experts, projection.output_dims),
            )
            if sidecar is not None:
                projection.set_continuous_sidecar(
                    scale_delta=sidecar.scale_delta,
                    output_bias=sidecar.output_bias,
                    low_rank_left=sidecar.low_rank_left,
                    low_rank_right=sidecar.low_rank_right,
                )
        if hasattr(projection, "set_sparse_residual_rows"):
            sparse_residual = load_switch_linear_sparse_residual_rows(
                artifact_root,
                layer=layer,
                projection=projection_name,
                input_dims=projection.input_dims,
                output_dims=projection.output_dims,
                num_experts=projection.num_experts,
            )
            if sparse_residual is not None:
                projection.set_sparse_residual_rows(
                    expert_indices=sparse_residual.expert_indices,
                    output_indices=sparse_residual.output_indices,
                    values=sparse_residual.values,
                )
    return QuantizedVQSwitchGLU(
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
    )


class GLM45AirMLXQuantizedSwitchGLU(nn.Module):
    def __init__(
        self,
        *,
        gate_proj: QuantizedSwitchLinear,
        up_proj: QuantizedSwitchLinear,
        down_proj: QuantizedSwitchLinear,
    ):
        super().__init__()
        if gate_proj.input_dims != up_proj.input_dims:
            raise ValueError("gate_proj and up_proj input dimensions must match")
        if gate_proj.output_dims != up_proj.output_dims:
            raise ValueError("gate_proj and up_proj output dimensions must match")
        if down_proj.input_dims != gate_proj.output_dims:
            raise ValueError("down_proj input dimension must match routed hidden dimension")
        if down_proj.output_dims != gate_proj.input_dims:
            raise ValueError("down_proj output dimension must match model hidden dimension")
        if len({gate_proj.num_experts, up_proj.num_experts, down_proj.num_experts}) != 1:
            raise ValueError("all switch projections must have the same expert count")
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    @property
    def input_dims(self) -> int:
        return self.gate_proj.input_dims

    @property
    def hidden_dims(self) -> int:
        return self.gate_proj.output_dims

    @property
    def num_experts(self) -> int:
        return self.gate_proj.num_experts

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        expanded = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            expanded, idx, inv_order = _gather_sort(expanded, indices)
        gate = self.gate_proj(expanded, idx, sorted_indices=do_sort)
        up = self.up_proj(expanded, idx, sorted_indices=do_sort)
        y = self.down_proj(nn.silu(gate) * up, idx, sorted_indices=do_sort)
        if do_sort:
            y = _scatter_unsort(y, inv_order, indices.shape)
        return y.squeeze(-2)

    @classmethod
    def from_weights(
        cls,
        *,
        gate_weight: mx.array,
        up_weight: mx.array,
        down_weight: mx.array,
        group_size: int = 128,
        bits: int = 2,
        mode: str = "affine",
    ) -> "GLM45AirMLXQuantizedSwitchGLU":
        return cls(
            gate_proj=_mlx_quantized_switch_linear_from_weight(
                gate_weight,
                group_size=group_size,
                bits=bits,
                mode=mode,
            ),
            up_proj=_mlx_quantized_switch_linear_from_weight(
                up_weight,
                group_size=group_size,
                bits=bits,
                mode=mode,
            ),
            down_proj=_mlx_quantized_switch_linear_from_weight(
                down_weight,
                group_size=group_size,
                bits=bits,
                mode=mode,
            ),
        )


def _mlx_quantized_switch_linear_from_weight(
    weight: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str,
) -> QuantizedSwitchLinear:
    if weight.ndim != 3:
        raise ValueError(f"weight must be 3D [experts, out, in], found {weight.shape}")
    num_experts, output_dims, input_dims = weight.shape
    dense = SwitchLinear(input_dims, output_dims, num_experts, bias=False)
    dense.weight = weight
    quantized = dense.to_quantized(group_size=group_size, bits=bits, mode=mode)
    mx.eval(quantized.parameters())
    return quantized


def _load_mlx_quantized_switch_linear(path: str | Path, prefix: str) -> QuantizedSwitchLinear:
    inspection = inspect_safetensors(path)
    raw_config = inspection.metadata.get("mlx_quantization_config")
    if raw_config is None:
        raise ValueError(f"{path} is missing mlx_quantization_config metadata")
    config = json.loads(raw_config)
    group_size = int(config["group_size"])
    bits = int(config["bits"])
    mode = str(config.get("mode", "affine"))
    arrays = mx.load(str(path))
    weight = arrays[f"{prefix}.weight"]
    scales = arrays[f"{prefix}.scales"]
    biases = arrays.get(f"{prefix}.biases")
    bias = arrays.get(f"{prefix}.bias")
    if scales.ndim != 3:
        raise ValueError(f"{prefix}.scales must be 3D [experts, out, in/group], found {scales.shape}")
    num_experts, output_dims = weight.shape[:2]
    input_dims = scales.shape[2] * group_size
    quantized = QuantizedSwitchLinear(
        input_dims,
        output_dims,
        num_experts,
        bias=False,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    quantized.weight = weight
    quantized.scales = scales
    if biases is not None:
        quantized.biases = biases
    if bias is not None:
        quantized.bias = bias
    mx.eval(quantized.parameters())
    return quantized


def load_glm45_air_mlx_quantized_switch_glu(
    artifact_dir: str | Path,
    layer: int,
) -> GLM45AirMLXQuantizedSwitchGLU:
    artifact_root = Path(artifact_dir)
    prefix = f"model.layers.{layer}.mlp.switch_mlp"
    return GLM45AirMLXQuantizedSwitchGLU(
        gate_proj=_load_mlx_quantized_switch_linear(
            artifact_root / f"layer-{layer:05d}-gate_proj.safetensors",
            f"{prefix}.gate_proj",
        ),
        up_proj=_load_mlx_quantized_switch_linear(
            artifact_root / f"layer-{layer:05d}-up_proj.safetensors",
            f"{prefix}.up_proj",
        ),
        down_proj=_load_mlx_quantized_switch_linear(
            artifact_root / f"layer-{layer:05d}-down_proj.safetensors",
            f"{prefix}.down_proj",
        ),
    )


def bind_glm45_air_vq_experts(
    model: GLM45AirVQModel,
    artifact_dir: str | Path,
    *,
    layers: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    target_layers = set(layers) if layers is not None else None
    logit_bias = load_logit_bias_sidecar(artifact_dir, vocab_size=int(model.args.vocab_size))
    if logit_bias is not None:
        model.set_final_logit_bias(
            token_ids=logit_bias.token_ids,
            biases=logit_bias.biases,
            position_indices=logit_bias.position_indices,
            token_position_indices=logit_bias.token_position_indices,
        )
    bound = []
    for layer_idx, layer in enumerate(model.model.layers):
        if target_layers is not None and layer_idx not in target_layers:
            continue
        if isinstance(layer.mlp, GLM45AirVQMoE):
            layer.mlp.bind_switch_mlp(load_glm45_air_vq_switch_glu(artifact_dir, layer_idx))
            router_correction = load_router_correction_sidecar(
                artifact_dir,
                layer=layer_idx,
                num_experts=int(model.args.n_routed_experts),
            )
            if router_correction is not None:
                layer.mlp.set_router_correction(
                    expert_bias_delta=router_correction.expert_bias_delta,
                    temperature=router_correction.temperature,
                )
            bound.append(layer_idx)
    return tuple(bound)


def bind_glm45_air_mlx_quantized_routed_experts(
    model: GLM45AirVQModel,
    artifact_dir: str | Path,
    *,
    layers: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    target_layers = set(layers) if layers is not None else None
    bound = []
    for layer_idx, layer in enumerate(model.model.layers):
        if target_layers is not None and layer_idx not in target_layers:
            continue
        if isinstance(layer.mlp, GLM45AirVQMoE):
            layer.mlp.bind_switch_mlp(load_glm45_air_mlx_quantized_switch_glu(artifact_dir, layer_idx))
            bound.append(layer_idx)
    return tuple(bound)


def set_glm45_air_switch_gather_vqmm(model: GLM45AirVQModel, enabled: bool) -> None:
    for layer in model.model.layers:
        if not isinstance(layer.mlp, GLM45AirVQMoE) or layer.mlp.switch_mlp is None:
            continue
        layer.mlp.switch_mlp.gate_proj.use_gather_vqmm = enabled
        layer.mlp.switch_mlp.up_proj.use_gather_vqmm = enabled
        layer.mlp.switch_mlp.down_proj.use_gather_vqmm = enabled


def set_glm45_air_prefill_engine(
    model: GLM45AirVQModel,
    engine: Literal["auto", "vq_metal", "nax_e8", "nax_e8p"],
) -> None:
    if engine not in ("auto", "vq_metal", "nax_e8", "nax_e8p"):
        raise ValueError("prefill engine must be 'auto', 'vq_metal', 'nax_e8', or 'nax_e8p'")
    for layer in model.model.layers:
        if not isinstance(layer.mlp, GLM45AirVQMoE) or layer.mlp.switch_mlp is None:
            continue
        layer.mlp.switch_mlp.prefill_engine = engine


def has_unbound_glm45_air_vq_experts(model: GLM45AirVQModel) -> bool:
    return any(isinstance(layer.mlp, GLM45AirVQMoE) and layer.mlp.switch_mlp is None for layer in model.model.layers)


def has_dense_glm45_air_routed_expert_parameters(model: GLM45AirVQModel) -> bool:
    return any(".mlp.experts." in key for key, _ in tree_flatten(model.parameters()))


def _dtype_name(dtype: object) -> str:
    name = getattr(dtype, "name", str(dtype))
    return name.rsplit(".", maxsplit=1)[-1]


def glm45_air_non_expert_dtype_report(
    model: GLM45AirVQModel,
    *,
    expected_dtype_name: str = "bfloat16",
) -> dict[str, object]:
    allowed_float32_suffixes = ("e_score_correction_bias",)
    allowed_non_expected = []
    unexpected = []
    checked = 0
    for key, value in tree_flatten(model.parameters()):
        if _is_vq_runtime_param(key):
            continue
        dtype_name = _dtype_name(value.dtype)
        if dtype_name not in {"float16", "bfloat16", "float32"}:
            continue
        checked += 1
        if dtype_name != expected_dtype_name:
            if dtype_name == "float32" and key.endswith(allowed_float32_suffixes):
                allowed_non_expected.append({"name": key, "dtype": dtype_name})
                continue
            unexpected.append({"name": key, "dtype": dtype_name})
    return {
        "expected_dtype_name": expected_dtype_name,
        "allowed_float32_suffixes": allowed_float32_suffixes,
        "checked": checked,
        "allowed_non_expected": allowed_non_expected,
        "unexpected": unexpected,
        "verified": not unexpected,
    }


def _target_non_expert_params(model: GLM45AirVQModel) -> dict[str, mx.array]:
    return {
        key: value
        for key, value in tree_flatten(model.parameters())
        if not _is_vq_runtime_param(key)
    }


def _normalize_non_expert_surfaces(surfaces: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if surfaces is None:
        return ()
    normalized: list[str] = []
    for surface in surfaces:
        value = str(surface).strip()
        if value and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _plan_glm45_air_non_expert_bind(
    model: GLM45AirVQModel,
    index,
    *,
    surfaces: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[_GLM45AirBindEntry], GLM45AirNonExpertBindReport]:
    requested_surfaces = _normalize_non_expert_surfaces(surfaces)
    all_target_params = _target_non_expert_params(model)
    # Requested surfaces describe the experiment; correctness requires a full source bind.
    target_params = all_target_params
    entries: list[_GLM45AirBindEntry] = []
    loaded: set[str] = set()
    skipped_experts: list[str] = []
    skipped_mtp: list[str] = []
    skipped_unmatched: list[str] = []

    for name in sorted(index.weight_map):
        if _is_mtp_tensor(name, model.args):
            skipped_mtp.append(name)
            continue
        if _is_routed_expert_weight(name):
            skipped_experts.append(name)
            continue

        if name in target_params:
            entries.append(_GLM45AirBindEntry(name, name))
            loaded.add(name)
        else:
            skipped_unmatched.append(name)

    missing = tuple(sorted(set(target_params) - loaded))
    return entries, GLM45AirNonExpertBindReport(
        loaded_model_parameters=tuple(sorted(loaded)),
        skipped_routed_expert_tensors=tuple(skipped_experts),
        skipped_mtp_tensors=tuple(skipped_mtp),
        skipped_unmatched_tensors=tuple(skipped_unmatched),
        missing_model_parameters=missing,
        requested_surfaces=requested_surfaces,
        skipped_by_surface_tensors=(),
    )


def bind_glm45_air_non_expert_weights(
    model: GLM45AirVQModel,
    source_dir: str | Path,
    index,
    *,
    strict: bool = True,
    surfaces: tuple[str, ...] | list[str] | None = None,
) -> GLM45AirNonExpertBindReport:
    entries, report = _plan_glm45_air_non_expert_bind(model, index, surfaces=surfaces)
    target_params = _target_non_expert_params(model)
    if strict and report.missing_model_parameters:
        missing = ",\n".join(report.missing_model_parameters)
        raise ValueError(f"Missing {len(report.missing_model_parameters)} non-expert parameters: \n{missing}.")

    for entry in entries:
        value = read_indexed_safetensors_tensor_mlx(source_dir, index, entry.source_name)
        expected = target_params.get(entry.target_name)
        if expected is None:
            raise ValueError(f"Received non-expert parameter not in model: {entry.target_name}")
        if value.shape != expected.shape:
            raise ValueError(
                f"Expected shape {expected.shape} but received shape {value.shape} for parameter {entry.target_name}"
            )
        model.load_weights([(entry.target_name, value)], strict=False)
    return report


def bind_glm45_air_non_expert_precision_policy(
    model: GLM45AirVQModel,
    artifact_dir: str | Path,
    source_dir: str | Path,
    index,
    *,
    strict: bool = True,
) -> GLM45AirNonExpertBindReport:
    manifest_path = Path(artifact_dir) / "conversion-manifest.json"
    if not manifest_path.exists():
        return bind_glm45_air_non_expert_weights(
            model, source_dir, index, strict=strict
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = manifest.get("non_expert_precision")
    if not isinstance(policy, dict) or policy.get("enabled") is not True:
        return bind_glm45_air_non_expert_weights(
            model, source_dir, index, strict=strict
        )
    surfaces = policy.get("surfaces")
    if not isinstance(surfaces, list) or not all(isinstance(item, str) for item in surfaces):
        raise ValueError("non_expert_precision.surfaces must be a list of strings")
    return bind_glm45_air_non_expert_weights(
        model,
        source_dir,
        index,
        strict=strict,
        surfaces=tuple(surfaces),
    )
