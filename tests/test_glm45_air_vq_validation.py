from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import decode_weight_matrix, e8_1bit_packed
from mlx_vq.convert.stream_convert import (
    convert_vq_groups_from_safetensors,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
)
from mlx_vq.validate.glm45_air_scale_fit import (
    evaluate_glm45_air_scale_fit,
    fit_e8_group_scales,
)
from mlx_vq.validate.glm45_air_vq import validate_glm45_air_vq
from mlx_vq.validate.glm45_air_vq import select_input_state_rows


def _write_tiny_glm4_sparse_checkpoint(tmp_path):
    config = {
        "model_type": "glm4_moe",
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
        "norm_topk_prob": True,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 1.25,
    }
    rng = np.random.default_rng(31415)
    weight_map = {}
    arrays = {}
    shard_name = "model-00001-of-00001.safetensors"

    gate_weight_name = "model.layers.1.mlp.gate.weight"
    correction_name = "model.layers.1.mlp.gate.e_score_correction_bias"
    arrays[gate_weight_name] = mx.array(rng.normal(scale=0.08, size=(4, 16)).astype(np.float32))
    arrays[correction_name] = mx.array(np.array([0.0, 0.05, -0.03, 0.02], dtype=np.float32))
    weight_map[gate_weight_name] = shard_name
    weight_map[correction_name] = shard_name

    for expert in range(4):
        for projection, shape in {
            "gate_proj": (8, 16),
            "up_proj": (8, 16),
            "down_proj": (16, 8),
        }.items():
            name = f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
            arrays[name] = mx.array(rng.normal(scale=0.04, size=shape).astype(np.float32))
            weight_map[name] = shard_name

    mx.save_safetensors(str(tmp_path / shard_name), arrays)
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    config_path.write_text(json.dumps(config))
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return config, config_path, index_path


def _write_tiny_vq_artifact(tmp_path):
    config, config_path, index_path = _write_tiny_glm4_sparse_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)
    artifact_dir = tmp_path / "artifact"
    convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=artifact_dir,
    )
    return config, config_path, index_path, artifact_dir


def test_validate_glm45_air_vq_compares_artifact_and_source_oracles(tmp_path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)

    result = validate_glm45_air_vq(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        model_id="tiny-glm4",
        layer=1,
        tokens=3,
        seed=9001,
        strict_config=False,
    )

    assert result.routes == 6
    assert result.top_k == 2
    assert result.source_tensors_read == 2 + 3 * len(result.selected_experts)
    assert result.metrics["artifact_weighted_routed"].actual_finite
    assert result.metrics["artifact_weighted_routed"].expected_finite
    assert result.metrics["artifact_weighted_routed"].cosine >= 0.999
    assert result.metrics["source_weighted_routed"].actual_finite
    assert result.metrics["source_weighted_routed"].expected_finite
    json.dumps(result.to_dict())


def test_validate_glm45_air_vq_emits_route_level_source_metrics(tmp_path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)

    result = validate_glm45_air_vq(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        model_id="tiny-glm4",
        layer=1,
        tokens=2,
        seed=9001,
        strict_config=False,
    )
    payload = result.to_dict()

    assert len(payload["route_source_metrics"]) == result.routes
    first = payload["route_source_metrics"][0]
    assert set(first) == {
        "token_index",
        "route_rank",
        "expert",
        "router_score",
        "metrics",
    }
    assert set(first["metrics"]) == {
        "source_gate_proj",
        "source_up_proj",
        "source_routed_glu",
        "source_weighted_route_contribution",
    }
    assert first["expert"] == payload["routed_indices"][0][0]
    assert first["router_score"] == payload["router_scores"][0][0]
    assert first["metrics"]["source_routed_glu"]["actual_finite"]


def test_validate_glm45_air_vq_emits_route_level_residual_topk(tmp_path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)

    result = validate_glm45_air_vq(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        model_id="tiny-glm4",
        layer=1,
        tokens=2,
        seed=9001,
        strict_config=False,
    )
    payload = result.to_dict()

    assert len(payload["route_source_residual_topk"]) == result.routes
    first = payload["route_source_residual_topk"][0]
    assert first["expert"] == payload["routed_indices"][0][0]
    assert first["router_score"] == payload["router_scores"][0][0]
    assert set(first["top_abs_residuals"]) == {
        "source_gate_proj",
        "source_up_proj",
        "source_routed_glu",
        "source_weighted_route_contribution",
    }
    weighted = first["top_abs_residuals"]["source_weighted_route_contribution"]
    assert 0 < len(weighted) <= 8
    assert set(weighted[0]) == {
        "index",
        "actual",
        "expected",
        "source_minus_actual",
        "abs_source_minus_actual",
    }
    assert weighted[0]["abs_source_minus_actual"] >= weighted[-1]["abs_source_minus_actual"]


def test_validate_glm45_air_vq_emits_down_proj_sparse_residual_plan(tmp_path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)

    result = validate_glm45_air_vq(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        model_id="tiny-glm4",
        layer=1,
        tokens=2,
        seed=9001,
        strict_config=False,
    )
    payload = result.to_dict()

    assert len(payload["route_source_sparse_residual_plans"]) == result.routes
    first = payload["route_source_sparse_residual_plans"][0]
    assert first["projection"] == "down_proj"
    assert first["expert"] == payload["routed_indices"][0][0]
    assert first["router_score"] == payload["router_scores"][0][0]
    assert 0 < len(first["rows"]) <= 8
    row = first["rows"][0]
    assert set(row) == {
        "output_index",
        "desired_weighted_correction",
        "desired_unweighted_correction",
        "input_norm_sq",
        "residual_value_norm",
        "residual_values",
        "reconstructed_unweighted_correction",
        "reconstruction_abs_error",
    }
    assert len(row["residual_values"]) == config["moe_intermediate_size"]
    assert row["reconstruction_abs_error"] <= 1e-4


def test_select_input_state_rows_accepts_batch_and_tail_window() -> None:
    states = np.arange(1 * 5 * 3, dtype=np.float32).reshape(1, 5, 3)

    selected, indices = select_input_state_rows(states, tokens=2, position="tail")

    assert selected.shape == (2, 3)
    assert indices == (3, 4)
    np.testing.assert_array_equal(selected, states[0, 3:5])


def test_validate_glm45_air_vq_accepts_prompt_derived_input_states(tmp_path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)
    rng = np.random.default_rng(20260624)
    input_states = rng.normal(size=(4, 16)).astype(np.float32)

    result = validate_glm45_air_vq(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        model_id="tiny-glm4",
        layer=1,
        tokens=99,
        input_states=input_states,
        input_source="unit_prompt:tail",
        prompt_text="Tiny prompt",
        prompt_token_ids=(1, 2, 3, 4, 5),
        input_state_indices=(1, 2, 3, 4),
        strict_config=False,
    )

    assert result.tokens == 4
    assert result.input_source == "unit_prompt:tail"
    assert result.prompt_text == "Tiny prompt"
    assert result.prompt_token_ids == (1, 2, 3, 4, 5)
    assert result.input_state_indices == (1, 2, 3, 4)
    payload = result.to_dict()
    assert payload["input_source"] == "unit_prompt:tail"
    assert payload["input_state_indices"] == [1, 2, 3, 4]
    assert payload["metrics"]["source_weighted_routed"]["actual_finite"]


def test_validate_glm45_air_vq_cli_emits_json_and_thresholds(tmp_path) -> None:
    _, config_path, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_glm45_air_vq.py",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--source-dir",
            str(tmp_path),
            "--artifact-dir",
            str(artifact_dir),
            "--layer",
            "1",
            "--tokens",
            "2",
            "--no-strict-config",
            "--min-artifact-cosine",
            "0.999",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["model_id"] == "tiny-glm4"
    assert payload["routes"] == 4
    assert payload["metrics"]["artifact_weighted_routed"]["cosine"] >= 0.999
    assert payload["thresholds"]["min_artifact_cosine"] == 0.999


def test_fit_e8_group_scales_recovers_known_scales() -> None:
    codebook = e8_1bit_packed()
    codes = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=np.uint8,
    )
    base = decode_weight_matrix(codes, scales=None, code_bits=8, codebook=codebook)
    known_scales = np.array(
        [
            [1.25, -0.75],
            [0.5, 2.0],
        ],
        dtype=np.float32,
    )
    source = (base.reshape(2, 2, 16) * known_scales[:, :, None]).reshape(2, 32)

    fitted = fit_e8_group_scales(
        source,
        codes,
        codebook,
        group_size=16,
        allow_negative_scales=True,
    )

    np.testing.assert_allclose(fitted.scales, known_scales, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fitted.weight, source, rtol=1e-6, atol=1e-6)
    assert fitted.stats.negative_fraction == 0.25
    assert fitted.stats.zero_denominator_count == 0


def test_evaluate_glm45_air_scale_fit_emits_tiny_schema(tmp_path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)

    result = evaluate_glm45_air_scale_fit(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        model_id="tiny-glm4",
        layer=1,
        tokens=3,
        seed=20260625,
        strict_config=False,
    )
    payload = result.to_dict()

    assert payload["model_id"] == "tiny-glm4"
    assert payload["routes"] == 6
    assert payload["allow_negative_scales"] is True
    assert payload["metrics"]["current_weighted_routed"]["actual_finite"]
    assert payload["metrics"]["fitted_weighted_routed"]["actual_finite"]
    assert payload["metrics"]["fitted_vs_current_weighted_routed"]["expected_finite"]
    assert payload["fit_stats"]
    assert result.source_tensors_read == 2 + 3 * len(result.selected_experts)
    json.dumps(payload)
