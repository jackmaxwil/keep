from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from mlx_vq.convert.inspect_hf import GLM52_MODEL_ID, summarize_config, validate_glm52
from mlx_vq.models.profiles import (
    ModelProfile,
    validate_profile_against_hf_config_data,
)


class GLM52Precision(str, Enum):
    VQ_ROUTED_EXPERT = "vq_routed_expert"
    AFFINE_OR_FP16 = "affine_or_fp16"
    FP16_OR_FP32 = "fp16_or_fp32"
    ROUTER = "router_fp16_or_affine_8_bit"


@dataclass(frozen=True)
class UpstreamGLM52Support:
    available: bool
    module: str
    missing_model_args: tuple[str, ...]
    indexshare_available: bool = False
    missing_indexshare_features: tuple[str, ...] = ()


REQUIRED_GLM52_MODEL_ARGS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "n_shared_experts",
    "n_routed_experts",
    "routed_scaling_factor",
    "kv_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "v_head_dim",
    "qk_nope_head_dim",
    "num_experts_per_tok",
    "first_k_dense_replace",
    "rope_parameters",
)


REQUIRED_GLM52_INDEXSHARE_CONFIG_FIELDS = (
    "indexer_types",
    "index_topk_pattern",
    "index_topk_freq",
    "index_skip_topk_offset",
)


REQUIRED_GLM52_INDEXSHARE_MODEL_ARGS = (
    "indexer_types",
    "index_topk_pattern",
    "index_topk_freq",
    "index_skip_topk_offset",
)


def validate_glm52_config(
    config: Mapping[str, object],
    *,
    model_id: str = GLM52_MODEL_ID,
    revision: str | None = None,
    profile: ModelProfile | None = None,
) -> list[str]:
    summary = summarize_config(dict(config), model_id=model_id)
    failures = validate_glm52(
        summary,
        expected_n_routed_experts=(
            256 if profile is None else profile.num_experts
        ),
    )
    if profile is not None:
        if profile.converter != "glm52_vq_groups":
            failures.append("profile_converter")
        if model_id != profile.hf_model_id:
            failures.append("model_id")
        if profile.revision is not None and revision != profile.revision:
            failures.append("revision")
        failures.extend(validate_profile_against_hf_config_data(profile, config))
    for field in REQUIRED_GLM52_MODEL_ARGS + REQUIRED_GLM52_INDEXSHARE_CONFIG_FIELDS:
        if field not in config:
            failures.append(f"missing_{field}")
    return list(dict.fromkeys(failures))


def installed_glm52_support() -> UpstreamGLM52Support:
    module_name = "mlx_lm.models.glm_moe_dsa"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return UpstreamGLM52Support(False, module_name, REQUIRED_GLM52_MODEL_ARGS)

    model_args = getattr(module, "ModelArgs", None)
    annotations = getattr(model_args, "__annotations__", {}) if model_args is not None else {}
    missing = tuple(field for field in REQUIRED_GLM52_MODEL_ARGS if field not in annotations)
    available = hasattr(module, "Model") and model_args is not None and not missing

    missing_indexshare = []
    for field in REQUIRED_GLM52_INDEXSHARE_MODEL_ARGS:
        if field not in annotations:
            missing_indexshare.append(f"ModelArgs.{field}")
    source_parts = []
    for candidate in (module, importlib.import_module("mlx_lm.models.deepseek_v32")):
        try:
            source_parts.append(inspect.getsource(candidate))
        except OSError:
            pass
    runtime_source = "\n".join(source_parts)
    if "prev_topk_indices" not in runtime_source:
        missing_indexshare.append("runtime.prev_topk_indices")
    if "indexer_types" not in runtime_source:
        missing_indexshare.append("runtime.indexer_types_schedule")
    if "make_cache" not in runtime_source or "indexer_types" not in runtime_source:
        missing_indexshare.append("runtime.indexshare_cache")

    missing_indexshare_tuple = tuple(missing_indexshare)
    return UpstreamGLM52Support(
        available,
        module_name,
        missing,
        indexshare_available=available and not missing_indexshare_tuple,
        missing_indexshare_features=missing_indexshare_tuple,
    )


def classify_glm52_parameter(path: str) -> GLM52Precision:
    if ".mlp.switch_mlp." in path and any(
        marker in path for marker in (".gate_proj.", ".up_proj.", ".down_proj.")
    ):
        return GLM52Precision.VQ_ROUTED_EXPERT
    if ".mlp.experts." in path and any(
        marker in path for marker in (".gate_proj.", ".up_proj.", ".down_proj.")
    ):
        return GLM52Precision.VQ_ROUTED_EXPERT
    if ".mlp.gate." in path:
        return GLM52Precision.ROUTER
    if ".self_attn.indexer." in path or ".self_attn." in path:
        return GLM52Precision.AFFINE_OR_FP16
    if ".mlp.shared_experts." in path:
        return GLM52Precision.AFFINE_OR_FP16
    if any(marker in path for marker in ("embed_tokens", "lm_head", "norm", "layernorm")):
        return GLM52Precision.FP16_OR_FP32
    if ".mlp." in path:
        return GLM52Precision.AFFINE_OR_FP16
    return GLM52Precision.FP16_OR_FP32


def is_vq_routed_expert(path: str) -> bool:
    return classify_glm52_parameter(path) == GLM52Precision.VQ_ROUTED_EXPERT
