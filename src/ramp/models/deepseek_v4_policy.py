"""Parameter precision policy and config validation for DeepSeek-V4-Flash.

Classification is derived from the tensor names actually present in
``deepseek-ai/DeepSeek-V4-Flash-0731`` (the 24 of 48 shards downloaded at scan
time, revision ``7872f01b``), not from a sibling family's naming. Two naming
forms are supported, because the checkpoint and the loaded MLX module tree
differ:

* the released checkpoint form, which is unprefixed and uses DeepSeek's own
  names -- ``layers.N.attn.wq_b.weight``, ``layers.N.ffn.experts.N.w1.weight``
  (``w1``=gate, ``w2``=down, ``w3``=up), ``layers.N.ffn.gate.{weight,bias,
  tid2eid}``, ``layers.N.hc_attn_scale``, ``embed.weight``. Block-scaled
  tensors ship a ``.scale`` companion next to ``.weight``; a companion always
  classifies with the projection it belongs to.
* the post-sanitize MLX module form, where experts are stacked and renamed --
  ``model.layers.N.ffn.switch_mlp.{gate,up,down}_proj``,
  ``model.layers.N.ffn.shared_experts.*``, ``model.layers.N.attn.*``,
  ``model.embed_tokens.weight``, ``lm_head.weight``. Leaf *module* paths carry
  no parameter suffix, mirroring the reference quantization predicate's
  ``endswith("_proj")`` tolerance.

Config facts come from the real ``config.json`` of the same revision, recorded
in keep-deepseek-v4-flash-pivot-20260811.md.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum

DEEPSEEK_V4_FLASH_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_V4_NUM_HIDDEN_LAYERS = 43


class DeepseekV4Precision(str, Enum):
    VQ_ROUTED_EXPERT = "vq_routed_expert"
    AFFINE_OR_FP16 = "affine_or_fp16"
    FP16_OR_FP32 = "fp16_or_fp32"
    ROUTER = "router_fp16_or_affine_8_bit"
    MTP_SKIP = "mtp_skip"


REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS = (
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "vocab_size",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "q_lora_rank",
    "qk_rope_head_dim",
    "max_position_embeddings",
    # Legacy MTP field; still set on V4-Flash, but the DSpark fields below are
    # the real architecture discriminator.
    "num_nextn_predict_layers",
    "index_topk",
    "index_n_heads",
    "index_head_dim",
    # V4-specific structure the loader has to honour. ``kv_lora_rank`` is
    # deliberately absent: V4-Flash attention is ``wkv``/``kv_norm`` with
    # ``o_lora_rank``/``o_groups``, and the real config has no such key.
    "compress_ratios",
    "dspark_block_size",
    "dspark_target_layer_ids",
    "num_hash_layers",
    "sliding_window",
    "o_lora_rank",
    "o_groups",
    "quantization_config",
    # ``quantization_config`` describes the *resident* tensors only (FP8 e4m3,
    # 128x128 blocks). ``expert_dtype`` is the discriminator for routed
    # experts, which ship FP4 packed two-per-byte in I8 with F8_E8M0 group-32
    # scales -- a different decoder entirely. A checkpoint that omits it, or
    # claims fp8 experts, is not the release this policy was measured against.
    "expert_dtype",
)

_EXPECTED_SCALARS = {
    "model_type": "deepseek_v4",
    "expert_dtype": "fp4",
    "num_hidden_layers": 43,
    "hidden_size": 4096,
    "vocab_size": 129280,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
}

_COMPRESS_RATIO_VALUES = frozenset({0, 4, 128})

# Anchored so the unprefixed released form (``layers.7.attn...``) matches as
# well as the sanitized form (``model.layers.7.attn...``).
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# The embedded DSpark drafter lives under a top-level ``mtp`` module; nothing
# below it is part of the backbone the VQ policy governs.
_MTP_PREFIXES = ("mtp.", "model.mtp.")

# FFN container: ``ffn`` in the released/reference tree, ``mlp`` in the
# HF-style tree.
_FFN_SEGMENTS = frozenset({"ffn", "mlp"})
# Routed-expert containers: per-expert in the checkpoint, stacked after
# sanitize.
_ROUTED_CONTAINERS = frozenset({"experts", "switch_mlp"})
# Released expert projections plus their sanitized names. Matched as whole
# path segments, so ``...switch_mlp.down_proj`` (a leaf module, no parameter
# suffix) and ``...experts.13.w1.scale`` (a block-scale companion) both hit.
_ROUTED_PROJECTIONS = frozenset(
    {"w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"}
)
_ATTENTION_SEGMENTS = frozenset({"attn", "self_attn"})
# Per-block norms and the model-level embedding/head/norm tensors. Note this
# does *not* include the ``norm`` inside ``attn.compressor``: that one is an
# attention tensor and is classified by the attention rule first.
_HIGH_PRECISION_SEGMENTS = frozenset(
    {"attn_norm", "ffn_norm", "norm", "embed", "embed_tokens", "head", "lm_head"}
)


def _layer_index(path: str) -> int | None:
    match = _LAYER_PATTERN.search(path)
    return int(match.group(1)) if match else None


def _is_hyper_connection(segments: Sequence[str]) -> bool:
    """Hyper-connection scalars: ``hc_ffn_fn`` released, ``ffn_hc.fn`` after
    sanitize (plus the model-level ``hc_head`` group)."""

    return any(
        segment.startswith("hc_") or segment.endswith("_hc") for segment in segments
    )


def classify_deepseek_v4_parameter(
    path: str,
    *,
    num_hidden_layers: int = DEEPSEEK_V4_NUM_HIDDEN_LAYERS,
) -> DeepseekV4Precision:
    # Drafter tensors first: an ``mtp.`` subtree mirrors the backbone module
    # names, so every module-path rule below would otherwise claim them.
    if path.startswith(_MTP_PREFIXES):
        return DeepseekV4Precision.MTP_SKIP
    layer = _layer_index(path)
    if layer is not None and layer >= num_hidden_layers:
        return DeepseekV4Precision.MTP_SKIP

    segments = path.split(".")
    if _is_hyper_connection(segments):
        return DeepseekV4Precision.FP16_OR_FP32
    if "attn_norm" in segments or "ffn_norm" in segments:
        return DeepseekV4Precision.FP16_OR_FP32

    ffn = next((i for i, seg in enumerate(segments) if seg in _FFN_SEGMENTS), None)
    if ffn is not None:
        child = segments[ffn + 1] if ffn + 1 < len(segments) else ""
        if child in _ROUTED_CONTAINERS and any(
            seg in _ROUTED_PROJECTIONS for seg in segments[ffn + 2 :]
        ):
            return DeepseekV4Precision.VQ_ROUTED_EXPERT
        if child == "gate":
            return DeepseekV4Precision.ROUTER
        if child == "shared_experts":
            return DeepseekV4Precision.AFFINE_OR_FP16

    # Everything under attention -- projections, indexer, compressor,
    # attention sink, and the norms that live inside them.
    if any(seg in _ATTENTION_SEGMENTS for seg in segments):
        return DeepseekV4Precision.AFFINE_OR_FP16
    if any(seg in _HIGH_PRECISION_SEGMENTS for seg in segments):
        return DeepseekV4Precision.FP16_OR_FP32
    if ffn is not None:
        return DeepseekV4Precision.AFFINE_OR_FP16
    return DeepseekV4Precision.FP16_OR_FP32


def is_vq_routed_expert(
    path: str,
    *,
    num_hidden_layers: int = DEEPSEEK_V4_NUM_HIDDEN_LAYERS,
) -> bool:
    return (
        classify_deepseek_v4_parameter(path, num_hidden_layers=num_hidden_layers)
        is DeepseekV4Precision.VQ_ROUTED_EXPERT
    )


def validate_deepseek_v4_config(config: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    for field in REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS:
        # A field present with an explicit ``null`` carries no more information
        # than an absent one -- JSON round-trips and hand-edited configs both
        # produce it -- so it fails as missing rather than sliding past the
        # membership check into the value checks below, which skip ``None``.
        if config.get(field) is None:
            failures.append(f"missing_{field}")
    # ``None`` is already reported as ``missing_<field>`` above; re-reporting it
    # as a bad value here would tag the same defect twice.
    for field, expected in _EXPECTED_SCALARS.items():
        value = config.get(field)
        if value is not None and value != expected:
            failures.append(field)
    failures.extend(_compress_ratio_failures(config))
    quantization = config.get("quantization_config")
    if isinstance(quantization, Mapping):
        if quantization.get("quant_method") != "fp8":
            failures.append("quantization_config")
        elif quantization.get("fmt") != "e4m3":
            failures.append("quantization_config")
    elif quantization is not None:
        failures.append("quantization_config")
    return list(dict.fromkeys(failures))


def _compress_ratio_failures(config: Mapping[str, object]) -> list[str]:
    """``compress_ratios`` selects per-layer KV compression (0/4/128).

    The released config ships 46 entries for 43 hidden layers -- the trailing
    entries cover the embedded drafter and the loader truncates to
    ``num_hidden_layers`` -- so the gate requires *at least* one entry per
    hidden layer and only validates the entries the backbone actually uses.
    """

    ratios = config.get("compress_ratios")
    layers = config.get("num_hidden_layers")
    if ratios is None or not isinstance(layers, int):
        return []
    if isinstance(ratios, (str, bytes)) or not isinstance(ratios, Sequence):
        return ["compress_ratios"]
    if len(ratios) < layers:
        return ["compress_ratios"]
    if any(ratio not in _COMPRESS_RATIO_VALUES for ratio in ratios[:layers]):
        return ["compress_ratios"]
    return []
