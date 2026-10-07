"""KEEP VQ adapter for ``zai-org/GLM-5.3-Flash`` (``glm5_next``), text path.

Reference
=========

``transformers`` v5.16.1, ``models/glm5_next/modeling_glm5_next.py``, the
first release that ships the family. Line numbers below refer to that file.
``tests/test_glm5_next_adapter.py`` checks this module against outputs of the
reference model saved by ``scripts/make_glm53_reference_fixtures.py``.

What is reused
--------------
* mHC: :class:`HyperConnection` and :func:`hc_expand` from the DeepSeek-V4
  adapter. The math is identical (reference 219 to 295). GLM's final collapse
  is a plain mean over streams (reference 298 to 302), not DeepSeek's learned
  head.
* Routing and the clamped SwiGLU: ``_expert_select`` and
  :class:`LimitedSwiGLU` from the same adapter. With ``n_group 1`` the
  reference router (145 to 183) is exactly that selection.
* KDA recurrence: mlx-lm's gated delta Metal kernel with per-channel decay.
* Routed experts: :class:`QuantizedVQSwitchGLU`, bound after construction, so
  no dense routed tensor is ever allocated.

What is new
-----------
* :class:`Glm5NextKDA`: the safe-gate decay
  ``g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias))`` (305 to 335), FLA
  ``l2norm`` on q and k, causal depthwise convs with SiLU, and an output
  RMSNorm gated by ``sigmoid`` (339 to 358).
* :class:`Glm5NextMLA`: DeepSeek-V3 MLA with no rotary part (NoPE).
* :class:`Glm5NextIndexer`: DSA top-k over pools of ``index_kpool`` keys with
  the open tail appended (736 to 1024).

Not implemented yet: the vision tower (skipped by design for v1), the NextN
drafter at layer 45, padding masks (batch 1 only), and ``shared`` indexer
layers (the release uses none; the policy rejects configs that do).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops

from ramp.models.glm4_moe_adapter import QuantizedVQSwitchGLU
from ramp.models.deepseek_v4_flash_adapter import (
    DeepseekV4FlashMLP,
    HyperConnection,
    LimitedSwiGLU,
    _expert_select,
    _stable_topk_indices,
    hc_expand,
    load_deepseek_v4_flash_vq_switch_glu,
)
from ramp.models.glm5_next_policy import (
    DSA,
    KDA,
    Glm5NextPrecision,
    classify_glm5_next_parameter,
    layer_schedule,
    text_config_of,
)

_CHECKPOINT_PREFIX = "model.language_model."


@dataclass
class Glm5NextVQModelArgs(BaseModelArgs):
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    layer_types: list
    mlp_layer_types: list
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    index_kpool: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    swiglu_limit: float
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    linear_attn_config: dict = field(default_factory=dict)
    model_type: str = "glm5_next_text"

    @property
    def linear_num_heads(self) -> int:
        return int(self.linear_attn_config["num_heads"])

    @property
    def linear_head_dim(self) -> int:
        return int(self.linear_attn_config["head_dim"])

    @property
    def linear_conv_kernel(self) -> int:
        return int(self.linear_attn_config.get("short_conv_kernel_size", 4))

    @property
    def linear_lower_bound(self) -> float:
        return float(self.linear_attn_config.get("gate_lower_bound", -5.0))


def glm5_next_args_from_config(config: Mapping[str, Any]) -> Glm5NextVQModelArgs:
    return Glm5NextVQModelArgs.from_dict(dict(text_config_of(config)))


def _l2norm(x: mx.array) -> mx.array:
    # FLA's l2norm: sqrt(sum + eps), not max(norm, eps) (reference 416 to 424).
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + 1e-6)


class Glm5NextKDA(nn.Module):
    def __init__(self, args: Glm5NextVQModelArgs):
        super().__init__()
        heads, dim = args.linear_num_heads, args.linear_head_dim
        width = heads * dim
        hidden = args.hidden_size
        self.num_heads, self.head_dim, self.kernel = heads, dim, args.linear_conv_kernel
        self.lower_bound = args.linear_lower_bound
        self.eps = args.rms_norm_eps
        self.q_proj = nn.Linear(hidden, width, bias=False)
        self.k_proj = nn.Linear(hidden, width, bias=False)
        self.v_proj = nn.Linear(hidden, width, bias=False)
        # Depthwise causal conv weights, [channels, kernel]. The checkpoint
        # stores [channels, 1, kernel]; the binder squeezes the middle axis.
        self.q_conv1d = mx.zeros((width, self.kernel), dtype=mx.float32)
        self.k_conv1d = mx.zeros((width, self.kernel), dtype=mx.float32)
        self.v_conv1d = mx.zeros((width, self.kernel), dtype=mx.float32)
        self.f_a_proj = nn.Linear(hidden, dim, bias=False)
        self.f_b_proj = nn.Linear(dim, width, bias=False)
        self.dt_bias = mx.zeros((width,), dtype=mx.float32)
        self.A_log = mx.zeros((heads,), dtype=mx.float32)
        self.b_proj = nn.Linear(hidden, heads, bias=False)
        self.g_a_proj = nn.Linear(hidden, dim, bias=False)
        self.g_b_proj = nn.Linear(dim, width, bias=False)
        self.o_norm = mx.ones((dim,))
        self.o_proj = nn.Linear(width, hidden, bias=False)

    def _conv(self, x: mx.array, weight: mx.array, state: mx.array | None):
        # out[t] = sum_j w[:, j] * x[t + j - (K - 1)], in float32 like the
        # reference, which keeps conv weights in fp32 (``_keep_in_fp32_modules``).
        batch, steps, width = x.shape
        if state is None:
            state = mx.zeros((batch, self.kernel - 1, width), dtype=x.dtype)
        padded = mx.concatenate([state, x], axis=1)
        wide = padded.astype(mx.float32)
        out = sum(wide[:, j : j + steps] * weight[:, j] for j in range(self.kernel))
        return nn.silu(out).astype(x.dtype), padded[:, -(self.kernel - 1) :]

    def __call__(self, x: mx.array, cache: ArraysCache | None = None) -> mx.array:
        batch, steps, _ = x.shape
        shape = (batch, steps, self.num_heads, self.head_dim)
        states = [None] * 4 if cache is None else [cache[i] for i in range(4)]
        q, states[0] = self._conv(self.q_proj(x), self.q_conv1d, states[0])
        k, states[1] = self._conv(self.k_proj(x), self.k_conv1d, states[1])
        v, states[2] = self._conv(self.v_proj(x), self.v_conv1d, states[2])

        q = _l2norm(q.reshape(shape).astype(mx.float32)) * self.head_dim**-0.5
        k = _l2norm(k.reshape(shape).astype(mx.float32))
        forget = self.f_b_proj(self.f_a_proj(x)).astype(mx.float32) + self.dt_bias
        rate = mx.exp(self.A_log.astype(mx.float32))[:, None]
        decay = mx.exp(self.lower_bound * mx.sigmoid(rate * forget.reshape(shape)))
        beta = mx.sigmoid(self.b_proj(x)).astype(mx.float32)

        state = states[3]
        if state is None:
            state = mx.zeros((batch, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32)
        values = v.reshape(shape).astype(mx.float32)
        if mx.default_device() == mx.gpu and mx.metal.is_available():
            out, state = gated_delta_kernel(q, k, values, decay, beta, state)
        else:
            out, state = gated_delta_ops(q, k, values, decay, beta, state)
        if cache is not None:
            for i, value in enumerate(states[:3] + [state]):
                cache[i] = value

        gate = self.g_b_proj(self.g_a_proj(x)).reshape(shape).astype(mx.float32)
        out = mx.fast.rms_norm(out, self.o_norm.astype(mx.float32), self.eps) * mx.sigmoid(gate)
        return self.o_proj(out.astype(x.dtype).reshape(batch, steps, -1))


class Glm5NextIndexer(nn.Module):
    def __init__(self, args: Glm5NextVQModelArgs):
        super().__init__()
        self.n_heads, self.head_dim = args.index_n_heads, args.index_head_dim
        self.topk, self.kpool = args.index_topk, args.index_kpool
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.index_kpool_compress_ape = mx.zeros((self.kpool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, args.hidden_size))

    def packed_keys(self, x: mx.array) -> mx.array:
        """``[B, T, 2 * head_dim]``: normalized key and its pooling gate."""

        gate = x @ self.index_kpool_compress_gate.T
        return mx.concatenate([self.k_norm(self.wk(x)), gate], axis=-1)

    def __call__(self, x: mx.array, q_resid: mx.array, packed: mx.array, offset: int) -> mx.array | None:
        """Boolean ``[B, 1, T, S]`` visibility, or ``None`` when it is plain causal.

        Every complete pool is selected while there are at most
        ``topk // kpool`` of them, and the open tail is always appended, so the
        layer is exact dense causal attention until the cache outgrows that.
        """

        batch, steps, _ = x.shape
        total = packed.shape[1]
        pools = total // self.kpool
        select = min(self.topk // self.kpool, pools)
        if pools <= self.topk // self.kpool:
            return None

        keys, gates = mx.split(packed[:, : pools * self.kpool], 2, axis=-1)
        keys = keys.reshape(batch, pools, self.kpool, self.head_dim)
        logits = gates.reshape(keys.shape).astype(mx.float32) + self.index_kpool_compress_ape.astype(mx.float32)
        weights = mx.softmax(logits, axis=2).astype(keys.dtype)
        pool_keys = (weights * keys).sum(axis=2).astype(mx.float32)

        q = self.wq_b(q_resid).reshape(batch, steps, self.n_heads, self.head_dim).astype(mx.float32)
        scores = nn.relu((q @ pool_keys[:, None].swapaxes(-1, -2)) * self.head_dim**-0.5)
        head_weights = self.weights_proj(x).astype(mx.float32) * self.n_heads**-0.5
        index = (head_weights[..., None, :] @ scores).squeeze(-2)

        positions = offset + mx.arange(steps)
        pool_end = mx.arange(pools) * self.kpool + self.kpool - 1
        visible = pool_end[None, :] <= positions[:, None]
        index = mx.where(visible, index, -mx.inf)
        # ReLU makes exact-zero scores common, so ties at the cut are real.
        # Lowest pool index wins, which is what the reference's torch.topk
        # does on CPU (the fixture pins it).
        chosen = _stable_topk_indices(index, select)
        hit = mx.take_along_axis(mx.broadcast_to(visible, index.shape), chosen, axis=-1)
        pool_mask = mx.put_along_axis(mx.zeros(index.shape, dtype=mx.bool_), chosen, hit, axis=-1)

        tokens = mx.arange(total)
        token_pool = mx.minimum(tokens // self.kpool, pools - 1)
        in_pool = mx.take(pool_mask, token_pool, axis=-1) & (tokens < pools * self.kpool)
        seen = (positions + 1)[:, None]
        in_tail = (tokens >= seen - seen % self.kpool) & (tokens < seen)
        return (in_pool | in_tail)[:, None]


class Glm5NextMLA(nn.Module):
    """DeepSeek-V3 MLA with no rotary part, plus the k-pool DSA indexer.

    ponytail: keys and values are expanded from the 512-wide latent on every
    call, which is exact but O(context) work per decoded token; absorbing
    ``kv_b_proj`` into q and the output is the decode optimization.
    """

    def __init__(self, args: Glm5NextVQModelArgs):
        super().__init__()
        hidden = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.nope, self.v_dim, self.rank = args.qk_nope_head_dim, args.v_head_dim, args.kv_lora_rank
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(args.q_lora_rank, self.num_heads * self.nope, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden, self.rank, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.rank, self.num_heads * (self.nope + self.v_dim), bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_dim, hidden, bias=False)
        self.indexer = Glm5NextIndexer(args)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        batch, steps, _ = x.shape
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).reshape(batch, steps, self.num_heads, self.nope).transpose(0, 2, 1, 3)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))[:, None]
        packed = self.indexer.packed_keys(x)[:, None]
        offset = 0 if cache is None else cache.offset
        if cache is not None:
            latent, packed = cache.update_and_fetch(latent, packed)

        mask = self.indexer(x, q_resid, packed[:, 0], offset)
        if mask is None:
            total = latent.shape[2]
            mask = mx.arange(total)[None, :] <= (offset + mx.arange(steps))[:, None]

        w = self.kv_b_proj.weight.reshape(self.num_heads, self.nope + self.v_dim, self.rank)
        keys = latent @ w[:, : self.nope].swapaxes(-1, -2)
        values = latent @ w[:, self.nope :].swapaxes(-1, -2)
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.nope**-0.5, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, steps, -1))


class Glm5NextGate(nn.Module):
    def __init__(self, args: Glm5NextVQModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # The reference scores in float32 regardless of activation dtype.
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return _expert_select(
            logits, self.e_score_correction_bias, self.top_k, self.scale, self.norm_topk_prob, "sigmoid"
        )


class Glm5NextMoE(nn.Module):
    """Routed experts arrive later via :meth:`bind_switch_mlp`."""

    def __init__(self, args: Glm5NextVQModelArgs):
        super().__init__()
        self.args = args
        self.gate = Glm5NextGate(args)
        self.switch_mlp: Any = None
        self.shared_experts = DeepseekV4FlashMLP(
            args, intermediate_size=args.moe_intermediate_size * args.n_shared_experts, swiglu_limit=args.swiglu_limit
        )

    def bind_switch_mlp(self, switch_mlp: Any) -> None:
        a = self.args
        dims = (switch_mlp.gate_proj.input_dims, switch_mlp.gate_proj.output_dims, switch_mlp.gate_proj.num_experts)
        if dims != (a.hidden_size, a.moe_intermediate_size, a.n_routed_experts):
            raise ValueError(f"routed experts {dims} do not match the config")
        activation = switch_mlp.activation
        # Plain SwiGLU binds, runs and stays finite while computing a
        # different function wherever an activation passes the clamp.
        if not isinstance(activation, LimitedSwiGLU) or activation.limit != a.swiglu_limit or activation.fp32:
            raise ValueError(f"routed experts need LimitedSwiGLU({a.swiglu_limit}), got {activation!r}")
        self.switch_mlp = switch_mlp

    def __call__(self, x: mx.array) -> mx.array:
        if self.switch_mlp is None:
            raise RuntimeError("routed experts must be bound before running a GLM-5.3 MoE layer")
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, args: Glm5NextVQModelArgs, attention: str, mlp: str):
        super().__init__()
        self.is_linear = attention == KDA
        self.self_attn = Glm5NextKDA(args) if self.is_linear else Glm5NextMLA(args)
        self.mlp = (
            Glm5NextMoE(args)
            if mlp == "sparse"
            else DeepseekV4FlashMLP(args, intermediate_size=args.intermediate_size, swiglu_limit=args.swiglu_limit)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)

    def __call__(self, h: mx.array, cache: Any = None) -> mx.array:
        x, post, comb = self.attn_hc(h)
        h = hc_expand(self.self_attn(self.input_layernorm(x), cache), h, post, comb)
        x, post, comb = self.ffn_hc(h)
        return hc_expand(self.mlp(self.post_attention_layernorm(x)), h, post, comb)


class Glm5NextTextModel(nn.Module):
    def __init__(self, args: Glm5NextVQModelArgs):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Glm5NextDecoderLayer(args, a, m) for a, m in layer_schedule(vars(args))]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, input_ids: mx.array, cache: list | None = None, *, hidden_states: list | None = None):
        h = self.embed_tokens(input_ids)
        h = mx.broadcast_to(h[:, :, None], (*h.shape[:2], self.hc_mult, h.shape[-1]))
        for i, layer in enumerate(self.layers):
            if hidden_states is not None:
                hidden_states.append(h)
            h = layer(h, None if cache is None else cache[i])
        return self.norm(h.mean(axis=2))


class Glm5NextVQModel(nn.Module):
    def __init__(self, args: Glm5NextVQModelArgs):
        super().__init__()
        self.args = args
        self.model = Glm5NextTextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> list:
        return [ArraysCache(size=4) if layer.is_linear else KVCache() for layer in self.layers]

    def __call__(self, input_ids: mx.array, cache: list | None = None) -> mx.array:
        return self.lm_head(self.model(input_ids, cache))


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def _module_name(checkpoint_name: str) -> str:
    name = checkpoint_name.removeprefix(_CHECKPOINT_PREFIX)
    if not name.startswith("lm_head"):
        name = "model." + name
    for site in ("attn", "ffn"):
        for leaf in ("fn", "base", "scale"):
            name = name.replace(f".hc_{site}_{leaf}", f".{site}_hc.{leaf}")
    return name


def bind_glm5_next_non_vq_weights(
    model: Glm5NextVQModel, weights: Mapping[str, mx.array]
) -> dict[str, int]:
    """Bind every non-routed text tensor, strictly, from checkpoint names.

    Routed experts, the MTP layer and the vision tower are counted and skipped.
    FP8 tensors must already be decoded: a stray ``weight_scale_inv`` is an
    error, not something to skip. Anything missing, unexpected or misshapen
    fails through ``load_weights(strict=True)`` before the model is touched.
    """

    report = {"bound": 0, "routed_skipped": 0, "mtp_skipped": 0, "vision_skipped": 0}
    layers = model.args.num_hidden_layers
    bound: list[tuple[str, mx.array]] = []
    for name, value in weights.items():
        cls = classify_glm5_next_parameter(name, num_hidden_layers=layers)
        if cls is Glm5NextPrecision.VQ_ROUTED_EXPERT:
            report["routed_skipped"] += 1
            continue
        if cls is Glm5NextPrecision.MTP:
            report["mtp_skipped"] += 1
            continue
        if cls is Glm5NextPrecision.VISION_SKIP:
            report["vision_skipped"] += 1
            continue
        if name.endswith("weight_scale_inv"):
            raise ValueError(f"{name}: FP8 tensors must be decoded before binding")
        target = _module_name(name)
        if name.endswith("_conv1d.weight"):
            # Stored as plain arrays [channels, kernel] on the KDA module.
            value = value.reshape(value.shape[0], value.shape[-1]).astype(mx.float32)
            target = target.removesuffix(".weight")
        elif name.endswith("self_attn.o_norm.weight"):
            target = target.removesuffix(".weight")
        elif name.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
            value = value.astype(mx.float32)
        bound.append((target, value))
    model.load_weights(bound, strict=True)
    report["bound"] = len(bound)
    return report


def bind_glm5_next_vq_experts(
    model: Glm5NextVQModel, artifact_dir: str | Path, layers: list[int] | None = None
) -> list[int]:
    """Bind KEEP VQ artifacts ``layer-NNNNN-{gate,up,down}_proj.safetensors``."""

    sparse = [i for i, layer in enumerate(model.layers) if isinstance(layer.mlp, Glm5NextMoE)]
    for i in sparse if layers is None else layers:
        switch = load_deepseek_v4_flash_vq_switch_glu(
            artifact_dir, i, swiglu_limit=model.args.swiglu_limit, prefix=f"model.layers.{i}.mlp.switch_mlp"
        )
        model.layers[i].mlp.bind_switch_mlp(switch)
    return sparse if layers is None else list(layers)


def has_unbound_glm5_next_vq_experts(model: Glm5NextVQModel) -> bool:
    return any(isinstance(layer.mlp, Glm5NextMoE) and layer.mlp.switch_mlp is None for layer in model.layers)


__all__ = [
    "DSA",
    "KDA",
    "Glm5NextIndexer",
    "Glm5NextKDA",
    "Glm5NextMLA",
    "Glm5NextMoE",
    "Glm5NextVQModel",
    "Glm5NextVQModelArgs",
    "QuantizedVQSwitchGLU",
    "bind_glm5_next_non_vq_weights",
    "bind_glm5_next_vq_experts",
    "glm5_next_args_from_config",
    "has_unbound_glm5_next_vq_experts",
]
