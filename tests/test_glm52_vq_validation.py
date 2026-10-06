from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.convert.stream_convert import (
    convert_vq_groups_from_safetensors,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
)
from mlx_vq.validate.glm52_vq import validate_glm52_vq


def _write_tiny_glm52_sparse_checkpoint(tmp_path):
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 4,
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "indexer_types": ["full", "full", "full", "shared"],
        "first_k_dense_replace": 3,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
    }
    rng = np.random.default_rng(14152)
    weight_map = {}
    arrays = {}
    shard_name = "model-00001-of-00001.safetensors"

    gate_weight_name = "model.layers.3.mlp.gate.weight"
    correction_name = "model.layers.3.mlp.gate.e_score_correction_bias"
    arrays[gate_weight_name] = mx.array(rng.normal(scale=0.08, size=(4, 16)).astype(np.float32))
    arrays[correction_name] = mx.array(np.array([0.0, 0.04, -0.02, 0.03], dtype=np.float32))
    weight_map[gate_weight_name] = shard_name
    weight_map[correction_name] = shard_name

    for expert in range(4):
        for projection, shape in {
            "gate_proj": (8, 16),
            "up_proj": (8, 16),
            "down_proj": (16, 8),
        }.items():
            name = f"model.layers.3.mlp.experts.{expert}.{projection}.weight"
            arrays[name] = mx.array(rng.normal(scale=0.04, size=shape).astype(np.float32))
            weight_map[name] = shard_name

    mx.save_safetensors(str(tmp_path / shard_name), arrays)
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    config_path.write_text(json.dumps(config))
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return config, config_path, index_path


def _write_tiny_vq_artifact(tmp_path):
    config, config_path, index_path = _write_tiny_glm52_sparse_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm52", group_size=8)
    artifact_dir = tmp_path / "artifact"
    convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=artifact_dir,
    )
    return config, config_path, index_path, artifact_dir


def test_validate_glm52_vq_compares_artifact_and_source_oracles(tmp_path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)

    result = validate_glm52_vq(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        model_id="tiny-glm52",
        layer=3,
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
    assert result.source_artifact_metrics["weighted_routed"].actual_finite
    assert result.source_artifact_metrics["weighted_routed"].expected_finite
    assert result.source_artifact_metrics["weighted_routed"].cosine < result.metrics["artifact_weighted_routed"].cosine
    assert result.artifact_quantization["gate_proj"].code_bits == 8
    assert result.artifact_quantization["gate_proj"].group_size == 8
    assert result.artifact_quantization["gate_proj"].codebook_name == "quip_e8"
    assert result.source_artifact_diagnosis == "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
    json.dumps(result.to_dict())


def test_validate_glm52_vq_rejects_dense_layers(tmp_path) -> None:
    config, _, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)

    try:
        validate_glm52_vq(
            config,
            source_dir=tmp_path,
            index_path=index_path,
            artifact_dir=artifact_dir,
            model_id="tiny-glm52",
            layer=0,
            strict_config=False,
        )
    except ValueError as exc:
        assert "not a sparse MoE layer" in str(exc)
    else:
        raise AssertionError("expected dense-layer validation to fail")


def test_validate_glm52_vq_cli_emits_json_and_thresholds(tmp_path) -> None:
    _, config_path, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_glm52_vq.py",
            "--model-id",
            "tiny-glm52",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--source-dir",
            str(tmp_path),
            "--artifact-dir",
            str(artifact_dir),
            "--layer",
            "3",
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

    assert payload["model_id"] == "tiny-glm52"
    assert payload["routes"] == 4
    assert payload["metrics"]["artifact_weighted_routed"]["cosine"] >= 0.999
    assert payload["source_artifact_metrics"]["weighted_routed"]["actual_finite"]
    assert payload["source_artifact_metrics"]["weighted_routed"]["expected_finite"]
    assert payload["artifact_quantization"]["gate_proj"]["code_bits"] == 8
    assert payload["artifact_quantization"]["down_proj"]["group_size"] == 8
    assert payload["source_artifact_diagnosis"] == (
        "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
    )
    assert payload["thresholds"]["min_artifact_cosine"] == 0.999
