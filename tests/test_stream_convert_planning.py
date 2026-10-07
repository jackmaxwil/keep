from __future__ import annotations

from keep.convert.inspect_hf import summarize_config, validate_glm45_air, validate_glm52
from keep.convert.stream_convert import estimate_vq_storage, plan_from_config


GLM52_CONFIG = {
    "model_type": "glm_moe_dsa",
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
    "mlp_layer_types": ["dense", "dense", "dense"] + ["sparse"] * 75,
    "indexer_types": ["full", "full", "full"]
    + [
        "full" if layer in {6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74} else "shared"
        for layer in range(3, 78)
    ],
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "hidden_size": 6144,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "max_position_embeddings": 1048576,
}


GLM45_AIR_CONFIG = {
    "model_type": "glm4_moe",
    "num_hidden_layers": 46,
    "first_k_dense_replace": 1,
    "n_routed_experts": 128,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "hidden_size": 4096,
    "moe_intermediate_size": 1408,
    "intermediate_size": 10944,
    "max_position_embeddings": 131072,
}


def test_glm52_summary_is_derived_from_config_metadata() -> None:
    summary = summarize_config(GLM52_CONFIG, model_id="zai-org/GLM-5.2")
    assert summary.dense_layers == 3
    assert summary.sparse_layers == 75
    assert summary.indexer_full_layers == 21
    assert summary.indexer_shared_layers == 57
    assert validate_glm52(summary) == []


def test_stock_glm52_validation_does_not_accept_reap_expert_count() -> None:
    summary = summarize_config(
        {**GLM52_CONFIG, "n_routed_experts": 168},
        model_id="0xSero/glm-5.2-reap-504B-v2",
    )

    assert "n_routed_experts" in validate_glm52(summary)


def test_glm45_air_summary_uses_first_dense_replace_when_layer_types_absent() -> None:
    summary = summarize_config(GLM45_AIR_CONFIG, model_id="zai-org/GLM-4.5-Air")
    assert summary.dense_layers == 1
    assert summary.sparse_layers == 45
    assert validate_glm45_air(summary) == []


def test_routed_expert_storage_estimates_match_plan_budget() -> None:
    plan = plan_from_config(GLM52_CONFIG, model_id="zai-org/GLM-5.2")
    assert plan.routed_expert_params == 724_775_731_200
    assert plan.vq1_storage.code_bytes == 90_596_966_400
    assert plan.vq1_storage.scale_bytes == 2_831_155_200
    assert plan.vq2_storage.code_bytes == 181_193_932_800
    assert plan.vq2_storage.scale_bytes == 2_831_155_200


def test_glm45_air_fixture_storage_is_small_enough_for_phase_a() -> None:
    plan = plan_from_config(GLM45_AIR_CONFIG, model_id="zai-org/GLM-4.5-Air")
    assert plan.routed_expert_params == 99_656_663_040
    assert plan.vq1_storage.total_bytes == 12_905_349_120
    assert plan.vq2_storage.total_bytes == 25_362_432_000


def test_estimate_vq_storage_rejects_non_codeword_aligned_weights() -> None:
    try:
        estimate_vq_storage(10, code_bits=8)
    except ValueError as exc:
        assert "8D codewords" in str(exc)
    else:
        raise AssertionError("expected non-codeword-aligned weights to fail")
