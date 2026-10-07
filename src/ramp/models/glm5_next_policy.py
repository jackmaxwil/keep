"""Config validation, layer schedule and tensor policy for GLM-5.3-Flash.

Headless on purpose: conversion and census tooling import this without MLX.

Every fact here was measured from ``zai-org/GLM-5.3-Flash`` at revision
``eb9eb208``: ``config.json`` and all 62 shard headers, recorded in
``artifacts/census/glm53-flash-source-census-eb9eb208.json``. The plan that
uses it is ``docs/glm-5.3-flash/2026-10-06-plan.md``.

The checkpoint prefixes the text model with ``model.language_model.`` and
the vision tower with ``model.visual.``. Routed experts ship per expert
(``mlp.experts.E.{gate,up,down}_proj``) as ``F8_E4M3`` with an F32 128x128
``weight_scale_inv`` companion, which classifies with its projection. The MLX
module tree stacks them under ``mlp.switch_mlp``, and both forms classify the
same way.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum

GLM5_NEXT_MODEL_ID = "zai-org/GLM-5.3-Flash"
GLM5_NEXT_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
GLM5_NEXT_NUM_HIDDEN_LAYERS = 45

KDA = "linear_attention"
DSA = "deepseek_sparse_attention"


class Glm5NextPrecision(str, Enum):
    VQ_ROUTED_EXPERT = "vq_routed_expert"
    # Large resident projections: kept at source precision in v1, and the
    # candidates for the resident quantization study (they set decode speed).
    RESIDENT_QUANT_CANDIDATE = "resident_quant_candidate"
    # Norms, mHC, KDA decay and conv parameters, the DSA indexer, the
    # embedding: small or selection-critical, never quantized.
    SOURCE_PRECISION = "source_precision"
    ROUTER = "router"
    # The NextN drafter at layer ``num_hidden_layers``; bound by its own path.
    MTP = "mtp"
    VISION_SKIP = "vision_skip"


REQUIRED_TEXT_CONFIG_FIELDS = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "vocab_size",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "first_k_dense_replace",
    "layer_types",
    "mlp_layer_types",
    "indexer_types",
    "linear_attn_config",
    "num_attention_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "index_topk",
    "index_n_heads",
    "index_head_dim",
    "index_kpool",
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    "routed_scaling_factor",
    "swiglu_limit",
    "num_nextn_predict_layers",
    "rms_norm_eps",
)

_EXPECTED_TEXT_SCALARS = {
    "model_type": "glm5_next_text",
    "num_hidden_layers": GLM5_NEXT_NUM_HIDDEN_LAYERS,
    "hidden_size": 4096,
    "moe_intermediate_size": 2048,
    "vocab_size": 154880,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    # The adapter implements NoPE MLA only. A rotary part would need RoPE in
    # both MLA and the indexer, which this checkpoint does not have.
    "qk_rope_head_dim": 0,
    "scoring_func": "sigmoid",
    "n_group": 1,
    "num_nextn_predict_layers": 1,
}

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Attention tensors that stay at source precision. In KDA that is every
# decay, beta and gate parameter (all small, all feeding the recurrence) and
# the convs. In DSA it is the norms and the whole indexer, which decides what
# attention can see. Only the large q/k/v/o and MLA projections are
# quantization candidates.
_ATTENTION_HIGH_PRECISION = frozenset(
    {
        "A_log", "dt_bias", "b_proj", "f_a_proj", "f_b_proj", "g_a_proj",
        "g_b_proj", "q_conv1d", "k_conv1d", "v_conv1d", "o_norm",
        "q_a_layernorm", "kv_a_layernorm", "indexer",
    }
)


def layer_schedule(text_config: Mapping[str, object]) -> list[tuple[str, str]]:
    """Per decoder layer ``(attention_kind, mlp_kind)``, MTP excluded."""

    layers = int(text_config["num_hidden_layers"])  # type: ignore[arg-type]
    attention = list(text_config["layer_types"])  # type: ignore[arg-type]
    mlp = list(text_config["mlp_layer_types"])  # type: ignore[arg-type]
    return list(zip(attention[:layers], mlp[:layers], strict=True))


def classify_glm5_next_parameter(
    path: str,
    *,
    num_hidden_layers: int = GLM5_NEXT_NUM_HIDDEN_LAYERS,
) -> Glm5NextPrecision:
    if path.startswith(("model.visual.", "visual.")):
        return Glm5NextPrecision.VISION_SKIP
    match = _LAYER.search(path)
    if match and int(match.group(1)) >= num_hidden_layers:
        return Glm5NextPrecision.MTP

    segments = path.split(".")
    if "mlp" in segments:
        child = segments[segments.index("mlp") + 1]
        if child in ("experts", "switch_mlp"):
            return Glm5NextPrecision.VQ_ROUTED_EXPERT
        if child == "gate":
            return Glm5NextPrecision.ROUTER
        return Glm5NextPrecision.RESIDENT_QUANT_CANDIDATE
    if "self_attn" in segments:
        if _ATTENTION_HIGH_PRECISION.intersection(segments):
            return Glm5NextPrecision.SOURCE_PRECISION
        return Glm5NextPrecision.RESIDENT_QUANT_CANDIDATE
    if segments[0] == "lm_head":
        return Glm5NextPrecision.RESIDENT_QUANT_CANDIDATE
    return Glm5NextPrecision.SOURCE_PRECISION


def text_config_of(config: Mapping[str, object]) -> Mapping[str, object]:
    """The text config of a full ``Glm5NextForConditionalGeneration`` config."""

    text = config.get("text_config")
    return text if isinstance(text, Mapping) else config


def validate_glm5_next_config(config: Mapping[str, object]) -> list[str]:
    """Failure tags for anything the adapter would silently mis-handle."""

    failures: list[str] = []
    if config.get("model_type") not in (None, "glm5_next"):
        failures.append("model_type")
    text = text_config_of(config)
    for field in REQUIRED_TEXT_CONFIG_FIELDS:
        if text.get(field) is None:
            failures.append(f"missing_{field}")
    for field, expected in _EXPECTED_TEXT_SCALARS.items():
        value = text.get(field)
        if value is not None and value != expected:
            failures.append(field)
    failures.extend(_schedule_failures(text))
    quantization = config.get("quantization_config", text.get("quantization_config"))
    if not (
        isinstance(quantization, Mapping)
        and quantization.get("quant_method") == "fp8"
        and quantization.get("fmt") == "e4m3"
        and list(quantization.get("weight_block_size") or ()) == [128, 128]
    ):
        failures.append("quantization_config")
    return list(dict.fromkeys(failures))


def _schedule_failures(text: Mapping[str, object]) -> list[str]:
    layers = text.get("num_hidden_layers")
    attention = text.get("layer_types")
    mlp = text.get("mlp_layer_types")
    linear = text.get("linear_attn_config")
    if not isinstance(layers, int) or not all(
        isinstance(v, Sequence) and not isinstance(v, str) for v in (attention, mlp)
    ):
        return []
    failures: list[str] = []
    if len(attention) < layers or set(attention[:layers]) - {KDA, DSA}:  # type: ignore[arg-type]
        failures.append("layer_types")
    elif isinstance(linear, Mapping):
        kda = [i for i in range(layers) if attention[i] == KDA]  # type: ignore[index]
        dsa = [i for i in range(layers) if attention[i] == DSA]  # type: ignore[index]
        if list(linear.get("kda_layers") or ()) != kda:
            failures.append("kda_layers")
        if list(linear.get("full_attn_layers") or ()) != dsa:
            failures.append("full_attn_layers")
    dense = text.get("first_k_dense_replace")
    if len(mlp) < layers or (  # type: ignore[arg-type]
        isinstance(dense, int)
        and list(mlp[:layers]) != ["dense"] * dense + ["sparse"] * (layers - dense)  # type: ignore[index]
    ):
        failures.append("mlp_layer_types")
    indexer = text.get("indexer_types")
    # ``shared`` layers reuse the previous indexer's top-k. Not implemented,
    # and not used by this checkpoint.
    if isinstance(indexer, Sequence) and any(v != "full" for v in indexer):
        failures.append("indexer_types")
    return failures
