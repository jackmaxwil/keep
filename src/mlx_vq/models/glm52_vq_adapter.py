from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from mlx.utils import tree_flatten

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32MLP,
    MoEGate,
)

from mlx_vq.io.load import load_quantized_vq_switch_linear
from mlx_vq.io.source_safetensors import read_indexed_safetensors_tensor_mlx
from mlx_vq.models.glm4_moe_adapter import QuantizedVQSwitchGLU
from mlx_vq.models.profiles import ModelProfile


_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")
_EXPERT_TENSOR_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
)
_INDEXER_TENSOR_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.self_attn\.indexer\.(?P<suffix>.+)$"
)
_GLM52_INDEXER_TENSOR_SUFFIXES = (
    "k_norm.bias",
    "k_norm.weight",
    "weights_proj.weight",
    "wk.weight",
    "wq_b.weight",
)


@dataclass
class GLM52VQModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict
    attention_bias: bool
    rope_scaling: Dict | None = None
    rope_theta: Optional[float] = None
    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[Any] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2
    mlp_layer_types: Optional[List[str]] = None
    rope_interleave: bool = True
    indexer_rope_interleave: bool = True

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]

        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                if isinstance(pattern, str):
                    self.indexer_types = [{"F": "full", "S": "shared"}[char] for char in pattern]
                else:
                    self.indexer_types = list(pattern)
            else:
                freq = max(self.index_topk_freq, 1)
                offset = self.index_skip_topk_offset
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]
        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError("indexer_types length must match num_hidden_layers")
        if any(kind not in {"full", "shared"} for kind in self.indexer_types):
            raise ValueError("indexer_types entries must be 'full' or 'shared'")
        if self.indexer_types and self.indexer_types[0] != "full":
            raise ValueError("indexer_types must start with 'full'")
        if self.mlp_layer_types is not None and len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("mlp_layer_types length must match num_hidden_layers")
        if self.rope_interleave is not True:
            raise ValueError("rope_interleave must be true for the GLM52 runtime")
        if self.indexer_rope_interleave is not True:
            raise ValueError(
                "indexer_rope_interleave must be true for the GLM52 runtime"
            )


@dataclass(frozen=True)
class GLM52IndexShareStaticAudit:
    full_layers: tuple[int, ...]
    shared_layers: tuple[int, ...]
    main_indexer_tensors: tuple[str, ...]
    mtp_indexer_tensors: tuple[str, ...]
    audit_pass: bool = True
    production_long_context_proven: bool = False

    @property
    def main_indexer_tensor_count(self) -> int:
        return len(self.main_indexer_tensors)


def audit_glm52_indexshare_static_contract(
    config: Mapping[str, object],
    index: Mapping[str, object],
) -> GLM52IndexShareStaticAudit:
    """Validate the pinned production IndexShare schedule and tensor inventory.

    This is a nonresident static audit. It deliberately does not claim that
    production long-context IndexShare execution has run.
    """

    failures: list[str] = []
    try:
        num_hidden_layers = int(config["num_hidden_layers"])
    except (KeyError, TypeError, ValueError):
        num_hidden_layers = -1
        failures.append("num_hidden_layers must be 78")
    if num_hidden_layers != 78 and not failures:
        failures.append(f"num_hidden_layers={num_hidden_layers}, expected=78")

    index_topk_pattern = config.get("index_topk_pattern")
    if index_topk_pattern is not None:
        failures.append("index_topk_pattern must be null")
    try:
        index_topk_freq = int(config["index_topk_freq"])
    except (KeyError, TypeError, ValueError):
        index_topk_freq = -1
    if index_topk_freq != 4:
        failures.append(f"index_topk_freq={index_topk_freq}, expected=4")
    try:
        index_skip_topk_offset = int(config["index_skip_topk_offset"])
    except (KeyError, TypeError, ValueError):
        index_skip_topk_offset = -1
    if index_skip_topk_offset != 3:
        failures.append(
            f"index_skip_topk_offset={index_skip_topk_offset}, expected=3"
        )
    for field_name in ("rope_interleave", "indexer_rope_interleave"):
        if config.get(field_name) is not True:
            failures.append(f"{field_name} must be true")

    raw_schedule = config.get("indexer_types")
    if not isinstance(raw_schedule, (list, tuple)):
        schedule: tuple[str, ...] = ()
        failures.append("indexer_types must be a 78-entry sequence")
    else:
        schedule = tuple(str(kind) for kind in raw_schedule)
    expected_schedule = tuple(
        "full" if (max(layer - 3 + 1, 0) % 4) == 0 else "shared"
        for layer in range(78)
    )
    if schedule != expected_schedule:
        failures.append(
            "indexer_types does not match the pinned 21-full/57-shared schedule"
        )

    full_layers = tuple(
        layer for layer, kind in enumerate(expected_schedule) if kind == "full"
    )
    shared_layers = tuple(
        layer for layer, kind in enumerate(expected_schedule) if kind == "shared"
    )
    expected_main = {
        f"model.layers.{layer}.self_attn.indexer.{suffix}"
        for layer in full_layers
        for suffix in _GLM52_INDEXER_TENSOR_SUFFIXES
    }
    expected_mtp = {
        f"model.layers.78.self_attn.indexer.{suffix}"
        for suffix in _GLM52_INDEXER_TENSOR_SUFFIXES
    }

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        actual_indexer: set[str] = set()
        failures.append("index.weight_map must be a mapping")
    else:
        actual_indexer = {
            str(name)
            for name in weight_map
            if _INDEXER_TENSOR_RE.match(str(name)) is not None
        }
    missing = sorted((expected_main | expected_mtp) - actual_indexer)
    unexpected = sorted(actual_indexer - (expected_main | expected_mtp))
    if missing:
        failures.append(f"missing_indexer_tensors={missing}")
    if unexpected:
        failures.append(f"unexpected_indexer_tensors={unexpected}")
    if failures:
        raise ValueError(
            "GLM52 IndexShare static contract failed: " + "; ".join(failures)
        )

    return GLM52IndexShareStaticAudit(
        full_layers=full_layers,
        shared_layers=shared_layers,
        main_indexer_tensors=tuple(sorted(expected_main)),
        mtp_indexer_tensors=tuple(sorted(expected_mtp)),
    )


@dataclass(frozen=True)
class GLM52NonVQBindReport:
    loaded_model_parameters: tuple[str, ...]
    transformed_kv_b_tensors: tuple[str, ...]
    skipped_routed_expert_tensors: tuple[str, ...]
    skipped_mtp_tensors: tuple[str, ...]
    skipped_unmatched_tensors: tuple[str, ...]
    missing_model_parameters: tuple[str, ...]

    @property
    def loaded_count(self) -> int:
        return len(self.loaded_model_parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "loaded_model_parameters": list(self.loaded_model_parameters),
            "loaded_count": self.loaded_count,
            "transformed_kv_b_tensors": list(self.transformed_kv_b_tensors),
            "skipped_routed_expert_tensors": list(self.skipped_routed_expert_tensors),
            "skipped_mtp_tensors": list(self.skipped_mtp_tensors),
            "skipped_unmatched_tensors": list(self.skipped_unmatched_tensors),
            "missing_model_parameters": list(self.missing_model_parameters),
        }


@dataclass(frozen=True)
class _GLM52NonVQBindEntry:
    source_name: str
    target_names: tuple[str, ...]


def is_glm52_sparse_layer(config: GLM52VQModelArgs, layer_idx: int) -> bool:
    if config.mlp_layer_types is not None:
        return config.mlp_layer_types[layer_idx] == "sparse"
    return (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


def glm52_vq_args_from_config(config: dict[str, Any]) -> GLM52VQModelArgs:
    field_names = {field.name for field in fields(GLM52VQModelArgs)}
    kwargs = {name: config[name] for name in field_names if name in config}
    return GLM52VQModelArgs(**kwargs)


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.match(name)
    return None if match is None else int(match.group("layer"))


def _is_mtp_tensor(name: str, args: GLM52VQModelArgs) -> bool:
    layer_idx = _layer_index(name)
    return layer_idx is not None and layer_idx >= args.num_hidden_layers


def _is_routed_expert_tensor(name: str) -> bool:
    return _EXPERT_TENSOR_RE.match(name) is not None


def _is_vq_runtime_param(name: str) -> bool:
    return name.startswith("mlp.switch_mlp.") or ".mlp.switch_mlp." in name


def _split_kv_b_proj(weight: mx.array, args: GLM52VQModelArgs) -> tuple[mx.array, mx.array]:
    head_dim = args.qk_nope_head_dim + args.v_head_dim
    expected_shape = (args.num_attention_heads * head_dim, args.kv_lora_rank)
    if weight.shape != expected_shape:
        raise ValueError(f"kv_b_proj.weight must have shape {expected_shape}, found {weight.shape}")
    reshaped = weight.reshape(args.num_attention_heads, head_dim, -1)
    embed_q = mx.contiguous(
        reshaped[:, : args.qk_nope_head_dim, :].swapaxes(-1, -2)
    )
    unembed_out = mx.contiguous(reshaped[:, args.qk_nope_head_dim :, :])
    return embed_q, unembed_out


class Glm52IndexShareAttention(DeepseekV32Attention):
    def __init__(self, config: GLM52VQModelArgs, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ) -> tuple[mx.array, Optional[mx.array]]:
        B, L, _ = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        if (
            self.indexer is None
            and prev_topk_indices is None
            and offset + L > self.config.index_topk
        ):
            raise ValueError(
                f"shared IndexShare layer {self.layer_idx} requires top-k "
                "indices from a previous full layer"
            )
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        if self.indexer is not None:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        else:
            topk_indices = prev_topk_indices

        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask,
                    topk_indices,
                    mx.array(True),
                    axis=-1,
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        if self.indexer is not None and cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=pe_scores,
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), topk_indices


class Glm52VQMoE(nn.Module):
    def __init__(self, config: GLM52VQModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp: QuantizedVQSwitchGLU | None = None
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV32MLP(
                config=config,
                intermediate_size=intermediate_size,
            )
        self.sharding_group = None

    def bind_switch_mlp(self, switch_mlp: QuantizedVQSwitchGLU) -> None:
        if switch_mlp.input_dims != self.config.hidden_size:
            raise ValueError("switch_mlp input dimension must match config.hidden_size")
        if switch_mlp.hidden_dims != self.config.moe_intermediate_size:
            raise ValueError("switch_mlp hidden dimension must match config.moe_intermediate_size")
        if switch_mlp.num_experts != self.config.n_routed_experts:
            raise ValueError("switch_mlp expert count must match config.n_routed_experts")
        self.switch_mlp = switch_mlp

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        if self.switch_mlp is None:
            raise RuntimeError("VQ switch_mlp must be bound before running a sparse GLM-5.2 MoE layer")

        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        shared_experts = self.get("shared_experts")
        if shared_experts is not None:
            y = y + shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class Glm52VQDecoderLayer(nn.Module):
    def __init__(self, config: GLM52VQModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Glm52IndexShareAttention(config, layer_idx)
        self.mlp = Glm52VQMoE(config) if is_glm52_sparse_layer(config, layer_idx) else DeepseekV32MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ) -> tuple[mx.array, Optional[mx.array]]:
        r, topk_indices = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache,
            prev_topk_indices,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk_indices


class Glm52VQBackbone(nn.Module):
    def __init__(self, config: GLM52VQModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Glm52VQDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pipeline_rank = 0
        self.pipeline_size = 1

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h,
            cache[0][0] if cache[0] else None,
            return_array=True,
        )

        if self.pipeline_rank < self.pipeline_size - 1:
            h = mx.distributed.recv_like(h, (self.pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h,
                mask,
                cache[i],
                prev_topk_indices,
            )

        if self.pipeline_rank != 0:
            h = mx.distributed.send(h, (self.pipeline_rank - 1) % self.pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        if self.pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class GLM52VQModel(nn.Module):
    def __init__(self, config: GLM52VQModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = Glm52VQBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    @property
    def cast_predicate(self):
        def predicate(key):
            return "e_score_correction_bias" not in key

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        out = self.model(inputs, cache)
        return self.lm_head(out)


def load_glm52_vq_switch_glu(artifact_dir: str | Path, layer: int) -> QuantizedVQSwitchGLU:
    artifact_root = Path(artifact_dir)
    prefix = f"model.layers.{layer}.mlp.switch_mlp"
    return QuantizedVQSwitchGLU(
        gate_proj=load_quantized_vq_switch_linear(
            artifact_root / f"layer-{layer:05d}-gate_proj.safetensors",
            f"{prefix}.gate_proj",
        ),
        up_proj=load_quantized_vq_switch_linear(
            artifact_root / f"layer-{layer:05d}-up_proj.safetensors",
            f"{prefix}.up_proj",
        ),
        down_proj=load_quantized_vq_switch_linear(
            artifact_root / f"layer-{layer:05d}-down_proj.safetensors",
            f"{prefix}.down_proj",
        ),
    )


def load_glm52_vq_switch_glu_from_paths(
    artifact_paths: Mapping[str, str | Path],
    layer: int,
) -> QuantizedVQSwitchGLU:
    prefix = f"model.layers.{layer}.mlp.switch_mlp"

    def path_for(projection: str) -> Path:
        filename = f"layer-{layer:05d}-{projection}.safetensors"
        try:
            return Path(artifact_paths[filename])
        except KeyError as error:
            raise ValueError(
                f"authenticated GLM52 artifact map is missing {filename}"
            ) from error

    return QuantizedVQSwitchGLU(
        gate_proj=load_quantized_vq_switch_linear(
            path_for("gate_proj"),
            f"{prefix}.gate_proj",
        ),
        up_proj=load_quantized_vq_switch_linear(
            path_for("up_proj"),
            f"{prefix}.up_proj",
        ),
        down_proj=load_quantized_vq_switch_linear(
            path_for("down_proj"),
            f"{prefix}.down_proj",
        ),
    )


def bind_glm52_decoder_layer_vq_experts(
    layer: Glm52VQDecoderLayer,
    artifact_dir: str | Path,
    *,
    profile: ModelProfile,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> int:
    """Bind one audited sparse layer's gate/up/down VQ projections."""

    layer_idx = layer.layer_idx
    if not isinstance(layer.mlp, Glm52VQMoE):
        raise ValueError(f"GLM52 decoder layer {layer_idx} is not a sparse MoE layer")
    config = layer.mlp.config
    expected = {
        "profile layer count": (profile.num_layers, config.num_hidden_layers),
        "profile hidden size": (profile.hidden_size, config.hidden_size),
        "profile MoE intermediate size": (
            profile.moe_intermediate_size,
            config.moe_intermediate_size,
        ),
        "profile expert count": (profile.num_experts, config.n_routed_experts),
        "profile experts per token": (
            profile.experts_per_tok,
            config.num_experts_per_tok,
        ),
    }
    mismatches = [
        f"{label}={profile_value} runtime={runtime_value}"
        for label, (profile_value, runtime_value) in expected.items()
        if profile_value != runtime_value
    ]
    if mismatches:
        raise ValueError(
            f"GLM52 decoder layer {layer_idx} profile/runtime mismatch: "
            + "; ".join(mismatches)
        )
    first_sparse = profile.first_sparse_layer
    if first_sparse is None or not first_sparse <= layer_idx < profile.num_layers:
        raise ValueError(
            f"GLM52 decoder layer {layer_idx} is outside the profile sparse-layer range"
        )

    switch_mlp = (
        load_glm52_vq_switch_glu(artifact_dir, layer_idx)
        if artifact_paths is None
        else load_glm52_vq_switch_glu_from_paths(artifact_paths, layer_idx)
    )
    if switch_mlp.num_experts != profile.num_experts:
        raise ValueError(
            f"GLM52 decoder layer {layer_idx} artifact expert count "
            f"{switch_mlp.num_experts} does not match profile expert count "
            f"{profile.num_experts}"
        )
    if switch_mlp.input_dims != profile.hidden_size:
        raise ValueError(
            f"GLM52 decoder layer {layer_idx} artifact input dimension does not "
            "match profile hidden size"
        )
    if switch_mlp.hidden_dims != profile.moe_intermediate_size:
        raise ValueError(
            f"GLM52 decoder layer {layer_idx} artifact hidden dimension does not "
            "match profile MoE intermediate size"
        )
    layer.mlp.bind_switch_mlp(switch_mlp)
    return layer_idx


def bind_glm52_vq_experts(
    model: GLM52VQModel,
    artifact_dir: str | Path,
    *,
    layers: tuple[int, ...] | None = None,
    profile: ModelProfile | None = None,
    strict: bool = False,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> tuple[int, ...]:
    if artifact_paths is not None and (profile is None or strict is not True):
        raise ValueError(
            "explicit authenticated GLM52 paths require strict mode and a profile"
        )
    expected_sparse_layers = tuple(
        layer_idx
        for layer_idx in range(model.args.num_hidden_layers)
        if is_glm52_sparse_layer(model.args, layer_idx)
    )
    if layers is None:
        requested_layers = expected_sparse_layers
    else:
        requested_layers = tuple(layers)
        if len(set(requested_layers)) != len(requested_layers):
            raise ValueError("requested sparse layers contain duplicates")
    if strict and requested_layers != expected_sparse_layers:
        raise ValueError(
            "requested sparse layers must exactly match the runtime sparse-layer set: "
            f"requested={requested_layers} expected={expected_sparse_layers}"
        )
    if strict and profile is None:
        raise ValueError("strict GLM52 VQ binding requires a model profile")
    if profile is not None:
        profile_sparse_layers = tuple(
            range(
                profile.first_sparse_layer or profile.num_layers,
                profile.num_layers,
            )
        )
        if (
            profile.num_layers != model.args.num_hidden_layers
            or profile.num_sparse_layers != len(expected_sparse_layers)
            or profile_sparse_layers != expected_sparse_layers
        ):
            raise ValueError(
                "GLM52 profile sparse layers do not match the runtime sparse-layer set"
            )

    target_layers = set(requested_layers)
    bound = []
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in target_layers:
            continue
        if isinstance(layer.mlp, Glm52VQMoE):
            if profile is None:
                layer.mlp.bind_switch_mlp(
                    load_glm52_vq_switch_glu(artifact_dir, layer_idx)
                )
            else:
                bind_glm52_decoder_layer_vq_experts(
                    layer,
                    artifact_dir,
                    profile=profile,
                    artifact_paths=artifact_paths,
                )
            bound.append(layer_idx)
    if strict and tuple(bound) != expected_sparse_layers:
        raise ValueError(
            "bound sparse layers do not match the exact runtime sparse-layer set"
        )
    return tuple(bound)


def bind_glm52_vq_experts_from_paths(
    model: GLM52VQModel,
    artifact_paths: Mapping[str, str | Path],
    *,
    layers: tuple[int, ...] | None = None,
    profile: ModelProfile | None = None,
    strict: bool = False,
) -> tuple[int, ...]:
    """Bind routed groups from exact authenticated file identities."""

    if profile is None or strict is not True:
        raise ValueError(
            "explicit authenticated GLM52 paths require strict mode and a profile"
        )
    return bind_glm52_vq_experts(
        model,
        Path("."),
        layers=layers,
        profile=profile,
        strict=strict,
        artifact_paths=artifact_paths,
    )


def has_unbound_vq_experts(model: GLM52VQModel) -> bool:
    return any(isinstance(layer.mlp, Glm52VQMoE) and layer.mlp.switch_mlp is None for layer in model.model.layers)


def dense_glm52_routed_parameter_names(model: GLM52VQModel) -> tuple[str, ...]:
    return tuple(
        sorted(
            key
            for key, _value in tree_flatten(model.parameters())
            if ".mlp.experts." in key
            or (
                ".mlp.switch_mlp." in key
                and key.endswith(".weight")
            )
        )
    )


@dataclass(frozen=True)
class _GLM52TargetParamContract:
    shape: tuple[int, ...]
    dtype: str


def _target_non_vq_contracts(
    model: GLM52VQModel,
) -> dict[str, _GLM52TargetParamContract]:
    return {
        key: _GLM52TargetParamContract(
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(value.dtype),
        )
        for key, value in tree_flatten(model.parameters())
        if not _is_vq_runtime_param(key)
    }


def _plan_glm52_non_vq_bind(
    model: GLM52VQModel,
    index,
) -> tuple[list[_GLM52NonVQBindEntry], GLM52NonVQBindReport]:
    target_params = _target_non_vq_contracts(model)
    entries: list[_GLM52NonVQBindEntry] = []
    loaded: set[str] = set()
    transformed_kv_b: list[str] = []
    skipped_experts: list[str] = []
    skipped_mtp: list[str] = []
    skipped_unmatched: list[str] = []

    for name in sorted(index.weight_map):
        if _is_mtp_tensor(name, model.args):
            skipped_mtp.append(name)
            continue
        if _is_routed_expert_tensor(name):
            skipped_experts.append(name)
            continue

        if name.endswith(".self_attn.kv_b_proj.weight"):
            prefix = name[: -len(".kv_b_proj.weight")]
            embed_q_name = f"{prefix}.embed_q.weight"
            unembed_out_name = f"{prefix}.unembed_out.weight"
            if embed_q_name in target_params and unembed_out_name in target_params:
                entries.append(_GLM52NonVQBindEntry(name, (embed_q_name, unembed_out_name)))
                loaded.update((embed_q_name, unembed_out_name))
                transformed_kv_b.append(name)
            else:
                skipped_unmatched.append(name)
            continue

        if name in target_params:
            entries.append(_GLM52NonVQBindEntry(name, (name,)))
            loaded.add(name)
        else:
            skipped_unmatched.append(name)

    missing = tuple(sorted(set(target_params) - loaded))
    return entries, GLM52NonVQBindReport(
        loaded_model_parameters=tuple(sorted(loaded)),
        transformed_kv_b_tensors=tuple(transformed_kv_b),
        skipped_routed_expert_tensors=tuple(skipped_experts),
        skipped_mtp_tensors=tuple(skipped_mtp),
        skipped_unmatched_tensors=tuple(skipped_unmatched),
        missing_model_parameters=missing,
    )


def _read_glm52_non_vq_entry(
    source_dir: str | Path,
    index,
    entry: _GLM52NonVQBindEntry,
    args: GLM52VQModelArgs,
) -> list[tuple[str, mx.array]]:
    source_weight = read_indexed_safetensors_tensor_mlx(source_dir, index, entry.source_name)
    if entry.source_name.endswith(".self_attn.kv_b_proj.weight"):
        embed_q, unembed_out = _split_kv_b_proj(source_weight, args)
        return [
            (entry.target_names[0], embed_q),
            (entry.target_names[1], unembed_out),
        ]
    return [(entry.target_names[0], source_weight)]


def bind_glm52_decoder_layer_non_vq_weights(
    layer: Glm52VQDecoderLayer,
    source_dir: str | Path,
    index,
    *,
    model_args: GLM52VQModelArgs,
    strict: bool = True,
) -> GLM52NonVQBindReport:
    """Strictly bind one decoder layer from a full non-routed source index."""

    layer_idx = layer.layer_idx
    if not 0 <= layer_idx < model_args.num_hidden_layers:
        raise ValueError(
            f"GLM52 decoder layer {layer_idx} is outside model_args.num_hidden_layers"
        )
    source_prefix = f"model.layers.{layer_idx}."
    relative_targets = {
        name: value
        for name, value in tree_flatten(layer.parameters())
        if not _is_vq_runtime_param(name)
    }
    target_params = {
        f"{source_prefix}{name}": value for name, value in relative_targets.items()
    }

    entries: list[_GLM52NonVQBindEntry] = []
    loaded: set[str] = set()
    transformed_kv_b: list[str] = []
    skipped_experts: list[str] = []
    unexpected: list[str] = []
    for name in sorted(index.weight_map):
        if _layer_index(name) != layer_idx:
            continue
        if _is_routed_expert_tensor(name):
            skipped_experts.append(name)
            continue
        if name.endswith(".self_attn.kv_b_proj.weight"):
            prefix = name[: -len(".kv_b_proj.weight")]
            embed_q_name = f"{prefix}.embed_q.weight"
            unembed_out_name = f"{prefix}.unembed_out.weight"
            if embed_q_name in target_params and unembed_out_name in target_params:
                entries.append(
                    _GLM52NonVQBindEntry(
                        name,
                        (embed_q_name, unembed_out_name),
                    )
                )
                loaded.update((embed_q_name, unembed_out_name))
                transformed_kv_b.append(name)
            else:
                unexpected.append(name)
            continue
        if name in target_params:
            entries.append(_GLM52NonVQBindEntry(name, (name,)))
            loaded.add(name)
        else:
            unexpected.append(name)

    missing = tuple(sorted(set(target_params) - loaded))
    report = GLM52NonVQBindReport(
        loaded_model_parameters=tuple(sorted(loaded)),
        transformed_kv_b_tensors=tuple(transformed_kv_b),
        skipped_routed_expert_tensors=tuple(skipped_experts),
        skipped_mtp_tensors=(),
        skipped_unmatched_tensors=tuple(unexpected),
        missing_model_parameters=missing,
    )
    if strict and (missing or unexpected):
        raise ValueError(
            f"GLM52 decoder layer {layer_idx} strict non-VQ bind failed: "
            f"missing_model_parameters={list(missing)}; "
            f"unexpected_source_tensors={unexpected}"
        )

    # Read and validate the selected layer completely before mutating it. This
    # keeps strict failure all-or-nothing while bounding retained arrays to one
    # decoder layer rather than the 37 GB package.
    global_weights: list[tuple[str, mx.array]] = []
    for entry in entries:
        global_weights.extend(
            _read_glm52_non_vq_entry(source_dir, index, entry, model_args)
        )
    for key, value in global_weights:
        expected = target_params.get(key)
        if expected is None:
            raise ValueError(f"Received decoder-layer parameter not in model: {key}")
        if value.shape != expected.shape:
            raise ValueError(
                f"Expected shape {expected.shape} but received shape {value.shape} "
                f"for decoder-layer parameter {key}"
            )
    local_weights = [
        (key[len(source_prefix) :], value) for key, value in global_weights
    ]
    layer.load_weights(local_weights, strict=False)
    return report


def collect_glm52_non_vq_weights(
    model: GLM52VQModel,
    source_dir: str | Path,
    index,
) -> tuple[list[tuple[str, mx.array]], GLM52NonVQBindReport]:
    entries, report = _plan_glm52_non_vq_bind(model, index)
    weights: list[tuple[str, mx.array]] = []
    for entry in entries:
        weights.extend(_read_glm52_non_vq_entry(source_dir, index, entry, model.args))
    return weights, report


def bind_glm52_non_vq_weights(
    model: GLM52VQModel,
    source_dir: str | Path,
    index,
    *,
    strict: bool = True,
) -> GLM52NonVQBindReport:
    entries, report = _plan_glm52_non_vq_bind(model, index)
    target_params = _target_non_vq_contracts(model)
    if strict and report.missing_model_parameters:
        missing = ",\n".join(report.missing_model_parameters)
        raise ValueError(
            f"Missing {len(report.missing_model_parameters)} non-VQ parameters: "
            f"\n{missing}."
        )
    if strict and report.skipped_unmatched_tensors:
        unexpected = ",\n".join(report.skipped_unmatched_tensors)
        raise ValueError(
            f"Unexpected {len(report.skipped_unmatched_tensors)} non-VQ source "
            f"tensors: \n{unexpected}."
        )

    for entry in entries:
        weights = _read_glm52_non_vq_entry(source_dir, index, entry, model.args)
        for key, value in weights:
            expected = target_params.get(key)
            if expected is None:
                raise ValueError(f"Received non-VQ parameter not in model: {key}")
            if tuple(value.shape) != expected.shape:
                raise ValueError(
                    f"Expected shape {expected.shape} but received shape "
                    f"{value.shape} for parameter {key}"
                )
        model.load_weights(weights, strict=False)
    return report
