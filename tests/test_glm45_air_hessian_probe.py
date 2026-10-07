from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import e8p_packed_abs_grid, encode_e8p_rtn
from mlx_vq.convert.stream_convert import (
    convert_vq_groups_from_safetensors,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
)
from mlx_vq.quality.hessian_rounding import (
    imatrix_weighted_reassign_codes,
    materialize_hessian_rounding_candidate,
)
from mlx_vq.quality.imatrix import ProjectionImatrixEntry
from mlx_vq.validate.glm45_air_hessian_probe import (
    evaluate_glm45_air_hessian_probe,
    hessian_weighted_reassign_codes,
)


def _write_tiny_glm4_sparse_checkpoint(tmp_path: Path):
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
    rng = np.random.default_rng(2718)
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


def _write_tiny_vq_artifact(tmp_path: Path):
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


def test_hessian_weighted_reassignment_prefers_important_dimensions() -> None:
    codebook = np.zeros((2, 8), dtype=np.float32)
    codebook[0] = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    codebook[1] = np.array([0, 8, 8, 8, 8, 8, 8, 8], dtype=np.float32)
    source = np.array([[1, 8, 8, 8, 8, 8, 8, 8]], dtype=np.float32)
    current_codes = np.array([[1]], dtype=np.uint8)
    scales = np.array([[1.0]], dtype=np.float32)
    hessian_diag = np.array([1000.0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)

    reassigned = hessian_weighted_reassign_codes(
        source,
        current_codes,
        scales,
        hessian_diag,
        codebook=codebook,
        group_size=8,
        code_bits=8,
    )
    current_weight = (codebook[current_codes.astype(np.int64)] * scales[..., None]).reshape(1, 8)

    assert reassigned.codes.tolist() == [[0]]
    assert reassigned.stats.changed_code_fraction == 1.0
    assert reassigned.stats.hessian_weighted_error < reassigned.stats.current_hessian_weighted_error
    assert np.linalg.norm(reassigned.weight - source) > np.linalg.norm(current_weight - source)


def test_imatrix_weighted_reassignment_uses_routed_column_importance() -> None:
    codebook = np.zeros((2, 8), dtype=np.float32)
    codebook[0] = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    codebook[1] = np.array([0, 8, 8, 8, 8, 8, 8, 8], dtype=np.float32)
    source = np.array([[1, 8, 8, 8, 8, 8, 8, 8]], dtype=np.float32)
    current_codes = np.array([[1]], dtype=np.uint8)
    scales = np.array([[1.0]], dtype=np.float32)
    imatrix = ProjectionImatrixEntry(
        layer=18,
        projection="gate_proj",
        expert=2,
        importance_sum=np.ones(8, dtype=np.float32),
        mean_importance=np.ones(8, dtype=np.float32),
        routing_weighted_importance=np.array([1000.0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        route_count=1,
        total_route_count=8,
        route_frequency=0.125,
        prompt_ids=("imatrix_calib_code_debug_000",),
    )

    reassigned = imatrix_weighted_reassign_codes(
        source,
        current_codes,
        scales,
        imatrix,
        codebook=codebook,
        group_size=8,
        code_bits=8,
    )
    current_weight = (codebook[current_codes.astype(np.int64)] * scales[..., None]).reshape(1, 8)

    assert reassigned.codes.tolist() == [[0]]
    assert reassigned.stats.hessian_weighted_error < reassigned.stats.current_hessian_weighted_error
    assert np.linalg.norm(reassigned.weight - source) > np.linalg.norm(current_weight - source)


def test_hessian_weighted_reassignment_supports_e8p_unit_weight_rtn() -> None:
    source = np.array([[0.25, -0.75, 1.25, -1.25, 0.5, -0.5, 0.0, 1.0]], dtype=np.float32)
    current_codes = np.array([[0]], dtype=np.uint16)
    scales = np.array([[1.0]], dtype=np.float32)
    hessian_diag = np.ones(8, dtype=np.float32)
    expected = encode_e8p_rtn(source.reshape(1, 1, 8), chunk_size=8).reshape(1, 1)

    reassigned = hessian_weighted_reassign_codes(
        source,
        current_codes,
        scales,
        hessian_diag,
        codebook=e8p_packed_abs_grid(),
        group_size=8,
        code_bits=16,
        codeword_chunk_size=1,
    )

    assert reassigned.codes.dtype == np.uint16
    np.testing.assert_array_equal(reassigned.codes, expected)
    assert reassigned.stats.code_bits == 16
    assert reassigned.stats.hessian_weighted_error <= reassigned.stats.current_hessian_weighted_error


def test_evaluate_glm45_air_hessian_probe_emits_tiny_schema(tmp_path: Path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)
    rng = np.random.default_rng(20260626)
    input_states = rng.normal(size=(5, 16)).astype(np.float32)

    result = evaluate_glm45_air_hessian_probe(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        model_id="tiny-glm4",
        layer=1,
        input_states=input_states,
        projections=("gate_proj", "up_proj"),
        max_experts=2,
        max_output_rows=4,
        strict_config=False,
    )
    payload = result.to_dict()

    assert payload["model_id"] == "tiny-glm4"
    assert payload["layer"] == 1
    assert payload["hessian"]["sample_count"] == 5
    assert payload["sampled_output_rows"] == [0, 2, 4, 7]
    assert payload["selected_experts"]
    assert set(payload["projection_results"]) == {"gate_proj", "up_proj"}
    assert payload["projection_results"]["gate_proj"]["hessian_weighted_error_ratio"] <= 1.0
    assert payload["projection_results"]["up_proj"]["current_output"]["actual_finite"]
    json.dumps(payload)


def test_materialize_hessian_rounding_candidate_rewrites_selected_group(tmp_path: Path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)
    rng = np.random.default_rng(20260626)
    input_states = rng.normal(size=(5, 16)).astype(np.float32)
    output_dir = tmp_path / "candidate"

    result = materialize_hessian_rounding_candidate(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        seed_artifact_dir=artifact_dir,
        output_dir=output_dir,
        model_id="tiny-glm4",
        layer=1,
        input_states=input_states,
        projections=("gate_proj",),
        max_experts=1,
        max_output_rows=4,
        strict_config=False,
    )

    gate_path = output_dir / "layer-00001-gate_proj.safetensors"
    up_path = output_dir / "layer-00001-up_proj.safetensors"
    manifest_path = output_dir / "hessian-rounding-manifest.json"
    seed_gate = mx.load(str(artifact_dir / "layer-00001-gate_proj.safetensors"))
    candidate_gate = mx.load(str(gate_path))
    codes_name = "model.layers.1.mlp.switch_mlp.gate_proj.codes"
    scales_name = "model.layers.1.mlp.switch_mlp.gate_proj.scales"

    assert gate_path.exists()
    assert not gate_path.is_symlink()
    assert up_path.is_symlink()
    assert manifest_path.exists()
    assert result["rewritten_group_count"] == 1
    assert result["projection_results"]["gate_proj"]["changed_code_count"] > 0
    assert not np.array_equal(np.asarray(seed_gate[codes_name]), np.asarray(candidate_gate[codes_name]))
    assert np.array_equal(np.asarray(seed_gate[scales_name]), np.asarray(candidate_gate[scales_name]))
    assert json.loads(manifest_path.read_text())["method"]["kind"] == "diagonal_hessian_code_reassignment"
