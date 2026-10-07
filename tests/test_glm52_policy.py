from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from ramp.models.glm52_policy import (
    GLM52Precision,
    classify_glm52_parameter,
    installed_glm52_support,
    is_vq_routed_expert,
    validate_glm52_config,
)
from ramp.models.profiles import get_profile


GLM52_CONFIG = {
    "model_type": "glm_moe_dsa",
    "vocab_size": 256,
    "hidden_size": 6144,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 2048,
    "intermediate_size": 12288,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 78,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "n_shared_experts": 1,
    "n_routed_experts": 256,
    "routed_scaling_factor": 2.5,
    "kv_lora_rank": 512,
    "q_lora_rank": 2048,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "qk_nope_head_dim": 192,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 8,
    "moe_layer_freq": 1,
    "first_k_dense_replace": 3,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
    "rope_interleave": True,
    "indexer_rope_interleave": True,
    "index_topk_pattern": None,
    "index_topk_freq": 4,
    "index_skip_topk_offset": 3,
    "attention_bias": False,
    "mlp_layer_types": ["dense", "dense", "dense"] + ["sparse"] * 75,
    "indexer_types": ["full", "full", "full"]
    + [
        "full" if layer in {6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74} else "shared"
        for layer in range(3, 78)
    ],
}


def _tiny_args():
    from mlx_lm.models.glm_moe_dsa import ModelArgs

    return ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=32,
        hidden_size=16,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=2,
        v_head_dim=4,
        qk_nope_head_dim=4,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        attention_bias=False,
    )


def test_installed_mlx_lm_has_glm_moe_dsa_indexshare_surface() -> None:
    status = installed_glm52_support()

    assert status.available
    assert status.module == "mlx_lm.models.glm_moe_dsa"
    assert status.missing_model_args == ()
    if not status.indexshare_available:
        assert "ModelArgs.indexer_types" in status.missing_indexshare_features
        assert "runtime.prev_topk_indices" in status.missing_indexshare_features


def test_glm52_config_policy_enforces_expected_contract() -> None:
    assert validate_glm52_config(GLM52_CONFIG) == []

    broken = dict(GLM52_CONFIG)
    broken["indexer_types"] = ["shared"] * 78
    assert "indexer_full_layers" in validate_glm52_config(broken)


def test_glm52_config_policy_accepts_only_the_pinned_reap_identity() -> None:
    profile = get_profile("glm52-reap-504b-v2")
    reap_config = {**GLM52_CONFIG, "n_routed_experts": 168, "vocab_size": 154880}

    assert validate_glm52_config(
        reap_config,
        model_id=profile.hf_model_id,
        revision=profile.revision,
        profile=profile,
    ) == []
    assert "n_routed_experts" in validate_glm52_config(reap_config)
    assert "n_routed_experts" in validate_glm52_config(
        GLM52_CONFIG,
        model_id=profile.hf_model_id,
        revision=profile.revision,
        profile=profile,
    )


@pytest.mark.parametrize(
    ("model_id", "revision", "profile_name", "expected_failure"),
    [
        (
            "wrong/model",
            "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
            "glm52-reap-504b-v2",
            "model_id",
        ),
        ("0xSero/glm-5.2-reap-504B-v2", "wrong", "glm52-reap-504b-v2", "revision"),
        ("0xSero/glm-5.2-reap-504B-v2", None, "glm52-reap-504b-v2", "revision"),
        (
            "Qwen/Qwen3.6-35B-A3B",
            "995ad96eacd98c81ed38be0c5b274b04031597b0",
            "qwen36-35b-a3b",
            "profile_converter",
        ),
    ],
)
def test_glm52_config_policy_rejects_wrong_profile_identity(
    model_id: str,
    revision: str | None,
    profile_name: str,
    expected_failure: str,
) -> None:
    reap_config = {**GLM52_CONFIG, "n_routed_experts": 168, "vocab_size": 154880}

    failures = validate_glm52_config(
        reap_config,
        model_id=model_id,
        revision=revision,
        profile=get_profile(profile_name),
    )

    assert expected_failure in failures


def test_glm52_mixed_precision_policy_only_vqs_routed_experts() -> None:
    assert is_vq_routed_expert("model.layers.4.mlp.switch_mlp.gate_proj.weight")
    assert is_vq_routed_expert("model.layers.4.mlp.experts.7.down_proj.weight")
    assert classify_glm52_parameter("model.layers.4.mlp.gate.weight") == GLM52Precision.ROUTER
    assert classify_glm52_parameter("model.layers.4.mlp.shared_experts.up_proj.weight") == GLM52Precision.AFFINE_OR_FP16
    assert classify_glm52_parameter("model.layers.4.self_attn.indexer.wk.weight") == GLM52Precision.AFFINE_OR_FP16
    assert classify_glm52_parameter("model.embed_tokens.weight") == GLM52Precision.FP16_OR_FP32
    assert classify_glm52_parameter("lm_head.weight") == GLM52Precision.FP16_OR_FP32


def test_tiny_glm_moe_dsa_sanitizes_raw_experts_and_binds_strictly() -> None:
    from mlx_lm.models.glm_moe_dsa import Model

    model = Model(_tiny_args())
    params = dict(tree_flatten(model.parameters()))
    raw_weights = {}
    for key, value in params.items():
        if ".mlp.switch_mlp." in key and key.endswith(".weight"):
            prefix, rest = key.split(".mlp.switch_mlp.")
            proj = rest.split(".")[0]
            for expert_idx in range(value.shape[0]):
                raw_weights[f"{prefix}.mlp.experts.{expert_idx}.{proj}.weight"] = value[expert_idx]
        else:
            raw_weights[key] = value

    sanitized = model.sanitize(dict(raw_weights))

    assert set(sanitized) == set(params)
    assert sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape == (2, 8, 16)
    assert sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"].shape == (2, 16, 8)
    model.load_weights(list(sanitized.items()), strict=True)
    mx.eval(model.parameters())
