from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.convert.stream_convert import (
    convert_vq_groups_from_safetensors,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
)
from mlx_vq.io.load import load_quantized_vq_switch_linear
from mlx_vq.quant.rht import apply_rht_np, deterministic_rht_signs
from mlx_vq.quant.rtn import quantize_weight_rtn
from mlx_vq.quality.rht_materialization import (
    materialize_rht_projection_candidate,
    materialize_rht_projection_candidates,
)
from mlx_vq.validate.glm45_air_vq import validate_glm45_air_vq


def _write_tiny_glm4_sparse_checkpoint(
    tmp_path: Path,
    *,
    hidden_size: int = 16,
    moe_intermediate_size: int = 8,
) -> tuple[dict, Path, Path]:
    config = {
        "model_type": "glm4_moe",
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": hidden_size,
        "moe_intermediate_size": moe_intermediate_size,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
    }
    rng = np.random.default_rng(20260627)
    arrays = {}
    weight_map = {}
    shard_name = "model-00001-of-00001.safetensors"
    gate_weight_name = "model.layers.1.mlp.gate.weight"
    correction_name = "model.layers.1.mlp.gate.e_score_correction_bias"
    arrays[gate_weight_name] = mx.array(rng.normal(scale=0.01, size=(2, hidden_size)).astype(np.float32))
    arrays[correction_name] = mx.array(np.zeros((2,), dtype=np.float32))
    weight_map[gate_weight_name] = shard_name
    weight_map[correction_name] = shard_name
    for expert in range(2):
        for projection, shape in {
            "gate_proj": (moe_intermediate_size, hidden_size),
            "up_proj": (moe_intermediate_size, hidden_size),
            "down_proj": (hidden_size, moe_intermediate_size),
        }.items():
            name = f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
            arrays[name] = mx.array(rng.normal(scale=0.03, size=shape).astype(np.float32))
            weight_map[name] = shard_name
    mx.save_safetensors(str(tmp_path / shard_name), arrays)
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    return config, config_path, index_path


def _write_tiny_vq_artifact(
    tmp_path: Path,
    *,
    hidden_size: int = 16,
    moe_intermediate_size: int = 8,
) -> tuple[dict, Path, Path, Path]:
    config, config_path, index_path = _write_tiny_glm4_sparse_checkpoint(
        tmp_path,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
    )
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


def test_materialize_rht_projection_candidate_links_seed_groups_and_rewrites_one_group(tmp_path: Path) -> None:
    config, _config_path, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)
    output_dir = tmp_path / "rht-candidate"

    manifest = materialize_rht_projection_candidate(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        seed_artifact_dir=artifact_dir,
        output_dir=output_dir,
        model_id="tiny-glm4",
        layer=1,
        projection="gate_proj",
        rht_seed="tiny-rht",
        group_size=8,
        strict_config=False,
    )

    target_path = output_dir / "layer-00001-gate_proj.safetensors"
    assert target_path.exists()
    assert not target_path.is_symlink()
    assert (output_dir / "layer-00001-up_proj.safetensors").is_symlink()
    assert (output_dir / "layer-00001-down_proj.safetensors").is_symlink()
    assert manifest["rewritten_group_count"] == 1
    assert manifest["linked_group_count"] == 2
    assert manifest["projection"]["rht"] == "hadamard_signs_v1"
    assert manifest["projection"]["rht_seed"] == "tiny-rht"

    stored_manifest = json.loads((output_dir / "rht-materialization-manifest.json").read_text(encoding="utf-8"))
    assert stored_manifest == manifest
    conversion_manifest = json.loads((output_dir / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert conversion_manifest["rht_materialization"]["projection"] == "gate_proj"

    prefix = "model.layers.1.mlp.switch_mlp.gate_proj"
    loaded = load_quantized_vq_switch_linear(target_path, prefix)
    signs = deterministic_rht_signs(16, seed="tiny-rht")
    np.testing.assert_array_equal(np.asarray(loaded.rht_signs), signs)

    source_arrays = mx.load(str(tmp_path / "model-00001-of-00001.safetensors"))
    source_weight = np.asarray(
        source_arrays["model.layers.1.mlp.experts.0.gate_proj.weight"],
        dtype=np.float32,
    )
    expected = quantize_weight_rtn(
        apply_rht_np(source_weight, signs),
        group_size=8,
        code_bits=8,
    )
    stored = mx.load(str(target_path))[f"{prefix}.codes"]
    np.testing.assert_array_equal(np.asarray(stored)[0], expected.codes)

    validation = validate_glm45_air_vq(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=output_dir,
        model_id="tiny-glm4",
        layer=1,
        tokens=2,
        strict_config=False,
    )
    assert validation.metrics["artifact_gate_proj"].cosine >= 0.999
    assert validation.metrics["artifact_up_proj"].cosine >= 0.999
    assert validation.metrics["artifact_weighted_routed"].cosine >= 0.999


def test_materialize_rht_projection_candidates_rewrites_multiple_groups_without_mutating_seed(
    tmp_path: Path,
) -> None:
    config, _config_path, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)
    output_dir = tmp_path / "rht-candidate"

    manifest = materialize_rht_projection_candidates(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        seed_artifact_dir=artifact_dir,
        output_dir=output_dir,
        model_id="tiny-glm4",
        targets=[
            {"layer": 1, "projection": "gate_proj", "rht_seed": "tiny-gate-rht"},
            {"layer": 1, "projection": "up_proj", "rht_seed": "tiny-up-rht"},
        ],
        group_size=8,
        strict_config=False,
    )

    assert manifest["rewritten_group_count"] == 2
    assert manifest["linked_group_count"] == 1
    assert [target["projection"] for target in manifest["projections"]] == ["gate_proj", "up_proj"]
    assert (output_dir / "layer-00001-gate_proj.safetensors").exists()
    assert not (output_dir / "layer-00001-gate_proj.safetensors").is_symlink()
    assert (output_dir / "layer-00001-up_proj.safetensors").exists()
    assert not (output_dir / "layer-00001-up_proj.safetensors").is_symlink()
    assert (output_dir / "layer-00001-down_proj.safetensors").is_symlink()

    stored_manifest = json.loads((output_dir / "rht-materialization-manifest.json").read_text(encoding="utf-8"))
    assert stored_manifest == manifest
    conversion_manifest = json.loads((output_dir / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert len(conversion_manifest["groups"]) == 2
    assert conversion_manifest["converted_vq_groups"] == 2

    seed_gate = mx.load(str(artifact_dir / "layer-00001-gate_proj.safetensors"))
    candidate_gate = mx.load(str(output_dir / "layer-00001-gate_proj.safetensors"))
    prefix = "model.layers.1.mlp.switch_mlp.gate_proj"
    assert f"{prefix}.rht_signs" not in seed_gate
    assert f"{prefix}.rht_signs" in candidate_gate

    validation = validate_glm45_air_vq(
        config,
        source_dir=tmp_path,
        index_path=index_path,
        artifact_dir=output_dir,
        model_id="tiny-glm4",
        layer=1,
        tokens=2,
        strict_config=False,
    )
    assert validation.metrics["artifact_gate_proj"].cosine >= 0.999
    assert validation.metrics["artifact_up_proj"].cosine >= 0.999
    assert validation.metrics["artifact_weighted_routed"].cosine >= 0.999


def test_materialize_rht_projection_candidate_rejects_non_power_of_two_input_before_output(
    tmp_path: Path,
) -> None:
    config, _config_path, index_path, artifact_dir = _write_tiny_vq_artifact(
        tmp_path,
        hidden_size=32,
        moe_intermediate_size=24,
    )
    output_dir = tmp_path / "rht-candidate"

    with pytest.raises(ValueError, match="positive power of two"):
        materialize_rht_projection_candidate(
            config,
            source_dir=tmp_path,
            index_path=index_path,
            seed_artifact_dir=artifact_dir,
            output_dir=output_dir,
            model_id="tiny-glm4",
            layer=1,
            projection="down_proj",
            rht_seed="tiny-rht",
            group_size=8,
            strict_config=False,
        )

    assert not output_dir.exists()


def test_materialize_rht_projection_cli_emits_json_and_candidate(tmp_path: Path) -> None:
    _config, config_path, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)
    output_dir = tmp_path / "rht-candidate"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/materialize_glm45_air_rht_projection.py",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(config_path),
            "--source-dir",
            str(tmp_path),
            "--index-path",
            str(index_path),
            "--seed-artifact-dir",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
            "--layer",
            "1",
            "--projection",
            "up_proj",
            "--rht-seed",
            "tiny-rht-cli",
            "--group-size",
            "8",
            "--no-strict-config",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["projection"]["projection"] == "up_proj"
    assert payload["projection"]["rht_seed"] == "tiny-rht-cli"
    assert (output_dir / "layer-00001-up_proj.safetensors").exists()
    assert (output_dir / "layer-00001-gate_proj.safetensors").is_symlink()
