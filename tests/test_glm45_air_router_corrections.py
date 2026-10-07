from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest
from mlx_lm.models.glm4_moe import ModelArgs

from keep.io.router_correction import (
    load_router_correction_sidecar,
    router_corrections_enabled,
    write_router_correction_artifact_manifest,
    write_router_correction_sidecar,
)
from ramp.models.glm45_air_vq_adapter import GLM45AirVQMoE


def _tiny_config() -> ModelArgs:
    return ModelArgs(
        model_type="glm4_moe",
        vocab_size=16,
        hidden_size=2,
        intermediate_size=4,
        max_position_embeddings=16,
        moe_intermediate_size=2,
        norm_topk_prob=True,
        num_attention_heads=1,
        n_group=1,
        head_dim=2,
        topk_group=1,
        n_shared_experts=None,
        n_routed_experts=3,
        routed_scaling_factor=1.0,
        num_experts_per_tok=1,
        first_k_dense_replace=0,
        num_hidden_layers=1,
        num_key_value_heads=1,
        rms_norm_eps=1.0e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        use_qk_norm=False,
        tie_word_embeddings=False,
        attention_bias=False,
        partial_rotary_factor=1.0,
    )


def test_loose_router_correction_is_ignored_without_manifest(tmp_path) -> None:
    write_router_correction_sidecar(
        output_dir=tmp_path,
        layer=41,
        num_experts=3,
        expert_bias_delta=mx.zeros((3,), dtype=mx.float32),
    )

    assert router_corrections_enabled(tmp_path) is False
    assert load_router_correction_sidecar(tmp_path, layer=41, num_experts=3) is None


def test_manifest_declared_router_correction_loads(tmp_path) -> None:
    entry = write_router_correction_sidecar(
        output_dir=tmp_path,
        layer=41,
        num_experts=3,
        expert_bias_delta=mx.array([0.0, 0.5, -0.25], dtype=mx.float32),
        temperature=mx.array([1.25], dtype=mx.float32),
    )
    write_router_correction_artifact_manifest(
        seed_manifest={},
        output_dir=tmp_path,
        corrections=[entry],
    )

    assert router_corrections_enabled(tmp_path) is True
    loaded = load_router_correction_sidecar(tmp_path, layer=41, num_experts=3)

    assert loaded is not None
    assert loaded.expert_bias_delta is not None
    assert loaded.temperature is not None
    np.testing.assert_allclose(np.array(loaded.expert_bias_delta), [0.0, 0.5, -0.25])
    np.testing.assert_allclose(np.array(loaded.temperature), [1.25])


def test_router_correction_rejects_shape_mismatch(tmp_path) -> None:
    entry = write_router_correction_sidecar(
        output_dir=tmp_path,
        layer=41,
        num_experts=3,
        expert_bias_delta=mx.zeros((3,), dtype=mx.float32),
    )
    write_router_correction_artifact_manifest(
        seed_manifest={},
        output_dir=tmp_path,
        corrections=[entry],
    )

    with pytest.raises(ValueError, match="expert_bias_delta must have shape"):
        load_router_correction_sidecar(tmp_path, layer=41, num_experts=4)


def test_noop_router_correction_matches_original_gate() -> None:
    moe = GLM45AirVQMoE(_tiny_config(), layer_idx=0)
    moe.gate.weight = mx.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=mx.float32)
    x = mx.array([[[0.2, 1.0], [1.0, 0.1]]], dtype=mx.float32)

    base_indices, base_scores = moe.route(x)
    moe.set_router_correction(
        expert_bias_delta=mx.zeros((3,), dtype=mx.float32),
        temperature=mx.array(1.0, dtype=mx.float32),
    )
    corrected_indices, corrected_scores = moe.route(x)
    mx.eval(base_indices, base_scores, corrected_indices, corrected_scores)

    np.testing.assert_array_equal(np.array(corrected_indices), np.array(base_indices))
    np.testing.assert_allclose(np.array(corrected_scores), np.array(base_scores), atol=1.0e-7)


def test_expert_bias_delta_can_move_top_route() -> None:
    moe = GLM45AirVQMoE(_tiny_config(), layer_idx=0)
    moe.gate.weight = mx.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=mx.float32)
    x = mx.array([[[0.2, 1.0]]], dtype=mx.float32)

    base_indices, _ = moe.route(x)
    moe.set_router_correction(expert_bias_delta=mx.array([2.0, 0.0, 0.0], dtype=mx.float32))
    corrected_indices, _ = moe.route(x)
    mx.eval(base_indices, corrected_indices)

    assert int(np.array(base_indices)[0, 0, 0]) == 1
    assert int(np.array(corrected_indices)[0, 0, 0]) == 0
