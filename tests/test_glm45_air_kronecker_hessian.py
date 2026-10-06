from __future__ import annotations

import importlib.util
import json
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
from mlx_vq.quant.rht import apply_rht_np, deterministic_rht_signs
from mlx_vq.quality.kronecker_hessian import (
    activation_covariance,
    blockldlq_full_reassign_codes,
    blockldlq_hin_only_reassign_codes,
    kronecker_weighted_error,
    ldl_factor,
    reconstruct_ldl,
    regularize_hessian,
    require_full_yaqa_hessian_manifest,
    save_hessian_factors,
    yaqa_sketch_a_hessian_factors,
)


def _write_tiny_glm4_sparse_checkpoint(tmp_path: Path):
    config = {
        "model_type": "glm4_moe",
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
    }
    rng = np.random.default_rng(77)
    arrays = {}
    weight_map = {}
    shard_name = "model-00001-of-00001.safetensors"
    arrays["model.layers.1.mlp.gate.weight"] = mx.array(rng.normal(scale=0.01, size=(2, 16)).astype(np.float32))
    arrays["model.layers.1.mlp.gate.e_score_correction_bias"] = mx.array(np.zeros((2,), dtype=np.float32))
    weight_map["model.layers.1.mlp.gate.weight"] = shard_name
    weight_map["model.layers.1.mlp.gate.e_score_correction_bias"] = shard_name
    for expert in range(2):
        for projection, shape in {
            "gate_proj": (8, 16),
            "up_proj": (8, 16),
            "down_proj": (16, 8),
        }.items():
            name = f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
            arrays[name] = mx.array(rng.normal(scale=0.03, size=shape).astype(np.float32))
            weight_map[name] = shard_name
    mx.save_safetensors(str(tmp_path / shard_name), arrays)
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    return config, index_path


def _write_tiny_vq_artifact(tmp_path: Path, *, code_bits_policy: dict[str, int] | None = None):
    config, index_path = _write_tiny_glm4_sparse_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id="tiny-glm4",
        group_size=8,
        code_bits_policy=code_bits_policy,
    )
    artifact_dir = tmp_path / "artifact"
    convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=artifact_dir,
    )
    return config, index_path, artifact_dir


def test_regularized_ldl_reconstructs_dense_hessian() -> None:
    samples = np.array(
        [
            [1.0, 2.0, -1.0],
            [0.5, -0.25, 2.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    covariance = activation_covariance(samples)
    regularized, regularization = regularize_hessian(covariance, damping=0.05)
    factor = ldl_factor(regularized)
    reconstructed = reconstruct_ldl(factor)

    assert regularization > 0.0
    assert np.allclose(reconstructed, regularized, atol=1.0e-5)


def test_blockldlq_hin_only_matches_direct_single_codeword_enumeration() -> None:
    codebook = np.zeros((3, 8), dtype=np.float32)
    codebook[0] = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    codebook[1] = np.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    codebook[2] = np.array([2, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    source = np.array([[1.7, 0.15, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    codes = np.array([[0]], dtype=np.uint8)
    scales = np.array([[1.0]], dtype=np.float32)
    h_in = np.eye(8, dtype=np.float32)
    h_in[0, 1] = h_in[1, 0] = 0.35

    reassigned = blockldlq_hin_only_reassign_codes(
        source,
        codes,
        scales,
        h_in,
        codebook=codebook,
        group_size=8,
    )
    direct_errors = []
    for candidate in codebook:
        diff = candidate - source[0]
        direct_errors.append(float(diff @ h_in @ diff))

    assert reassigned.codes.tolist() == [[int(np.argmin(direct_errors))]]
    assert np.isclose(reassigned.stats.hin_weighted_error, min(direct_errors), atol=1.0e-8)


def test_blockldlq_hin_only_accepts_16bit_code_tables() -> None:
    codebook = np.zeros((4, 8), dtype=np.float32)
    codebook[1] = np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    codebook[2] = np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    codebook[3] = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    source = np.array([[0.95, 0.20, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    codes = np.array([[0]], dtype=np.uint16)
    scales = np.ones((1, 1), dtype=np.float32)
    h_in = np.eye(8, dtype=np.float32)

    reassigned = blockldlq_hin_only_reassign_codes(
        source,
        codes,
        scales,
        h_in,
        codebook=codebook,
        group_size=8,
        code_bits=16,
    )

    assert reassigned.codes.dtype == np.uint16
    assert reassigned.codes.tolist() == [[1]]
    assert reassigned.stats.code_bits == 16


def test_kronecker_weighted_error_matches_direct_vectorized_objective() -> None:
    weight = np.array(
        [
            [1.0, -0.25, 0.5],
            [0.75, 0.5, -1.0],
        ],
        dtype=np.float32,
    )
    source = np.array(
        [
            [0.5, 0.25, -0.5],
            [1.25, -0.25, 0.0],
        ],
        dtype=np.float32,
    )
    h_in = np.array(
        [
            [1.5, 0.25, 0.0],
            [0.25, 2.0, -0.5],
            [0.0, -0.5, 1.25],
        ],
        dtype=np.float32,
    )
    h_out = np.array([[1.0, 0.4], [0.4, 1.75]], dtype=np.float32)
    diff = weight - source
    expected = float(np.einsum("oi,op,pj,ij->", diff, h_out, diff, h_in))

    assert np.isclose(kronecker_weighted_error(weight, source, h_in, h_out), expected)


def test_blockldlq_full_matches_bruteforce_coupled_output_enumeration() -> None:
    codebook = np.zeros((4, 8), dtype=np.float32)
    codebook[1] = np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    codebook[2] = np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    codebook[3] = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    source = np.array(
        [
            [0.90, 0.20, 0, 0, 0, 0, 0, 0],
            [0.10, 0.95, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    codes = np.array([[0], [0]], dtype=np.uint8)
    scales = np.ones((2, 1), dtype=np.float32)
    h_in = np.eye(8, dtype=np.float32)
    h_in[0, 1] = h_in[1, 0] = 0.2
    h_out = np.array([[1.0, 0.35], [0.35, 1.25]], dtype=np.float32)

    reassigned = blockldlq_full_reassign_codes(
        source,
        codes,
        scales,
        h_in,
        h_out,
        codebook=codebook,
        group_size=8,
        sweeps=4,
    )

    brute_force_errors: dict[tuple[int, int], float] = {}
    for first_code in range(codebook.shape[0]):
        for second_code in range(codebook.shape[0]):
            candidate = np.stack([codebook[first_code], codebook[second_code]], axis=0)
            brute_force_errors[(first_code, second_code)] = kronecker_weighted_error(
                candidate,
                source,
                h_in,
                h_out,
            )
    best_codes, best_error = min(brute_force_errors.items(), key=lambda item: item[1])

    assert reassigned.codes.tolist() == [[best_codes[0]], [best_codes[1]]]
    assert np.isclose(reassigned.stats.kronecker_weighted_error, best_error, atol=1.0e-8)
    assert reassigned.stats.kronecker_weighted_error_ratio <= 1.0


def test_quality_package_exports_full_yaqa_helpers() -> None:
    import mlx_vq.quality as quality

    assert quality.blockldlq_full_reassign_codes is blockldlq_full_reassign_codes
    assert quality.kronecker_weighted_error is kronecker_weighted_error
    assert quality.yaqa_sketch_a_hessian_factors is yaqa_sketch_a_hessian_factors


def test_full_yaqa_manifest_gate_accepts_hin_and_hout(tmp_path: Path) -> None:
    hessian_dir = tmp_path / "hessian"
    hessian_dir.mkdir()
    filename = "layer-00001-gate_proj-expert-00000.safetensors"
    h_in = np.eye(16, dtype=np.float32)
    h_out = np.eye(8, dtype=np.float32)
    save_hessian_factors(
        hessian_dir / filename,
        h_in=h_in,
        h_out=h_out,
        metadata={
            "method": "full_yaqa",
            "layer": 1,
            "projection": "gate_proj",
            "expert": 0,
            "sample_count": 4,
        },
    )
    manifest = {
        "schema_version": 1,
        "method": {"kind": "full_yaqa", "full_yaqa": True, "h_out_collected": True},
        "entries": [
            {
                "layer": 1,
                "projection": "gate_proj",
                "expert": 0,
                "path": filename,
                "sample_count": 4,
                "h_in_dim": 16,
                "h_out_dim": 8,
            }
        ],
    }
    (hessian_dir / "kronecker-hessian-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    payload = require_full_yaqa_hessian_manifest(hessian_dir)

    assert payload["ok"] is True
    assert payload["entry_count"] == 1
    assert payload["entries"][0]["h_in_dim"] == 16
    assert payload["entries"][0]["h_out_dim"] == 8


def test_yaqa_sketch_a_hessian_factors_use_input_and_output_gradients() -> None:
    input_samples = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.5, -1.0, 3.0],
        ],
        dtype=np.float32,
    )
    output_grad_samples = np.array(
        [
            [0.25, -0.5],
            [1.5, 2.0],
        ],
        dtype=np.float32,
    )

    h_in, h_out = yaqa_sketch_a_hessian_factors(input_samples, output_grad_samples)

    np.testing.assert_allclose(h_in, input_samples.T @ input_samples / 2.0)
    np.testing.assert_allclose(h_out, output_grad_samples.T @ output_grad_samples / 2.0)


def test_yaqa_sketch_a_hessian_factors_reject_sample_mismatch() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        yaqa_sketch_a_hessian_factors(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
        )


def test_full_yaqa_manifest_gate_rejects_hin_only_method(tmp_path: Path) -> None:
    hessian_dir = tmp_path / "hessian"
    hessian_dir.mkdir()
    filename = "layer-00001-gate_proj-expert-00000.safetensors"
    save_hessian_factors(
        hessian_dir / filename,
        h_in=np.eye(16, dtype=np.float32),
        metadata={"method": "blockldlq_hin_only"},
    )
    manifest = {
        "schema_version": 1,
        "method": {"kind": "blockldlq_hin_only", "full_yaqa": False, "h_out_collected": False},
        "entries": [
            {
                "layer": 1,
                "projection": "gate_proj",
                "expert": 0,
                "path": filename,
                "sample_count": 4,
            }
        ],
    }

    with pytest.raises(ValueError, match="method.kind must be 'full_yaqa'"):
        require_full_yaqa_hessian_manifest(hessian_dir, manifest=manifest)


def test_full_yaqa_manifest_gate_rejects_missing_hout_tensor(tmp_path: Path) -> None:
    hessian_dir = tmp_path / "hessian"
    hessian_dir.mkdir()
    filename = "layer-00001-gate_proj-expert-00000.safetensors"
    save_hessian_factors(
        hessian_dir / filename,
        h_in=np.eye(16, dtype=np.float32),
        metadata={"method": "full_yaqa"},
    )
    manifest = {
        "schema_version": 1,
        "method": {"kind": "full_yaqa", "full_yaqa": True, "h_out_collected": True},
        "entries": [
            {
                "layer": 1,
                "projection": "gate_proj",
                "expert": 0,
                "path": filename,
                "sample_count": 4,
            }
        ],
    }

    with pytest.raises(ValueError, match="missing required tensor 'h_out'"):
        require_full_yaqa_hessian_manifest(hessian_dir, manifest=manifest)


def test_save_hessian_factors_and_materializer_cli_tiny(tmp_path: Path) -> None:
    _config, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)
    hessian_dir = tmp_path / "hessian"
    hessian_dir.mkdir()
    h_in = np.eye(16, dtype=np.float32)
    save_hessian_factors(
        hessian_dir / "layer-00001-gate_proj-expert-00000.safetensors",
        h_in=h_in,
        metadata={
            "method": "blockldlq_hin_only",
            "model_id": "tiny-glm4",
            "layer": 1,
            "projection": "gate_proj",
            "expert": 0,
            "sample_count": 3,
        },
    )
    manifest = {
        "schema_version": 1,
        "method": {"kind": "blockldlq_hin_only", "full_yaqa": False, "h_out_collected": False},
        "entries": [
            {
                "layer": 1,
                "projection": "gate_proj",
                "expert": 0,
                "path": "layer-00001-gate_proj-expert-00000.safetensors",
                "sample_count": 3,
            }
        ],
    }
    (hessian_dir / "kronecker-hessian-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    output_dir = tmp_path / "candidate"

    module_path = Path(__file__).parents[1] / "benchmarks" / "materialize_glm45_air_yaqa_rounding.py"
    spec = importlib.util.spec_from_file_location("materialize_glm45_air_yaqa_rounding", module_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    old_argv = sys.argv
    try:
        sys.argv = [
            "materialize",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(tmp_path / "missing-config.json"),
            "--source-dir",
            str(tmp_path),
            "--index-path",
            str(index_path),
            "--hessian-dir",
            str(hessian_dir),
            "--seed-artifact-dir",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
            "--max-output-rows",
            "4",
        ]
        # The tiny materializer does not need the config payload yet, but the CLI
        # validates the option by reading it. Provide it immediately before call.
        (tmp_path / "missing-config.json").write_text("{}", encoding="utf-8")
        cli.main()
    finally:
        sys.argv = old_argv

    payload = json.loads((output_dir / "yaqa-rounding-manifest.json").read_text(encoding="utf-8"))
    assert payload["method"]["kind"] == "blockldlq_hin_only"
    assert payload["method"]["full_yaqa"] is False
    assert (output_dir / "layer-00001-gate_proj.safetensors").exists()
    assert (output_dir / "layer-00001-up_proj.safetensors").is_symlink()


def test_materializer_cli_tiny_accepts_full_yaqa_hout_manifest(tmp_path: Path) -> None:
    _config, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path)
    hessian_dir = tmp_path / "hessian"
    hessian_dir.mkdir()
    h_in = np.eye(16, dtype=np.float32)
    h_out = np.eye(8, dtype=np.float32)
    save_hessian_factors(
        hessian_dir / "layer-00001-gate_proj-expert-00000.safetensors",
        h_in=h_in,
        h_out=h_out,
        metadata={
            "method": "full_yaqa",
            "model_id": "tiny-glm4",
            "layer": 1,
            "projection": "gate_proj",
            "expert": 0,
            "sample_count": 3,
        },
    )
    manifest = {
        "schema_version": 1,
        "method": {"kind": "full_yaqa", "full_yaqa": True, "h_out_collected": True},
        "entries": [
            {
                "layer": 1,
                "projection": "gate_proj",
                "expert": 0,
                "path": "layer-00001-gate_proj-expert-00000.safetensors",
                "sample_count": 3,
                "h_in_dim": 16,
                "h_out_dim": 8,
            }
        ],
    }
    (hessian_dir / "kronecker-hessian-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    output_dir = tmp_path / "candidate"

    module_path = Path(__file__).parents[1] / "benchmarks" / "materialize_glm45_air_yaqa_rounding.py"
    spec = importlib.util.spec_from_file_location("materialize_glm45_air_yaqa_rounding", module_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    old_argv = sys.argv
    try:
        sys.argv = [
            "materialize",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(tmp_path / "config.json"),
            "--source-dir",
            str(tmp_path),
            "--index-path",
            str(index_path),
            "--hessian-dir",
            str(hessian_dir),
            "--seed-artifact-dir",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
            "--max-output-rows",
            "4",
        ]
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cli.main()
    finally:
        sys.argv = old_argv

    payload = json.loads((output_dir / "yaqa-rounding-manifest.json").read_text(encoding="utf-8"))
    assert payload["method"]["kind"] == "full_yaqa"
    assert payload["method"]["full_yaqa"] is True
    assert payload["method"]["h_out_collected"] is True
    result = payload["projection_results"]["layer_1.gate_proj"]
    assert "kronecker_weighted_error" in result
    assert "kronecker_weighted_error_ratio" in result
    assert "hin_weighted_error" not in result
    assert (output_dir / "layer-00001-gate_proj.safetensors").exists()
    assert (output_dir / "layer-00001-up_proj.safetensors").is_symlink()


def test_materializer_cli_tiny_accepts_16bit_seed_groups(tmp_path: Path) -> None:
    _config, index_path, artifact_dir = _write_tiny_vq_artifact(tmp_path, code_bits_policy={"*:gate_proj": 16})
    hessian_dir = tmp_path / "hessian"
    hessian_dir.mkdir()
    h_in = np.eye(16, dtype=np.float32)
    h_out = np.eye(8, dtype=np.float32)
    save_hessian_factors(
        hessian_dir / "layer-00001-gate_proj-expert-00000.safetensors",
        h_in=h_in,
        h_out=h_out,
        metadata={
            "method": "full_yaqa",
            "model_id": "tiny-glm4",
            "layer": 1,
            "projection": "gate_proj",
            "expert": 0,
            "sample_count": 3,
        },
    )
    manifest = {
        "schema_version": 1,
        "method": {"kind": "full_yaqa", "full_yaqa": True, "h_out_collected": True},
        "entries": [
            {
                "layer": 1,
                "projection": "gate_proj",
                "expert": 0,
                "path": "layer-00001-gate_proj-expert-00000.safetensors",
                "sample_count": 3,
                "h_in_dim": 16,
                "h_out_dim": 8,
            }
        ],
    }
    (hessian_dir / "kronecker-hessian-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "candidate"

    module_path = Path(__file__).parents[1] / "benchmarks" / "materialize_glm45_air_yaqa_rounding.py"
    spec = importlib.util.spec_from_file_location("materialize_glm45_air_yaqa_rounding", module_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    old_argv = sys.argv
    try:
        sys.argv = [
            "materialize",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(tmp_path / "config.json"),
            "--source-dir",
            str(tmp_path),
            "--index-path",
            str(index_path),
            "--hessian-dir",
            str(hessian_dir),
            "--seed-artifact-dir",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
            "--max-output-rows",
            "4",
        ]
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cli.main()
    finally:
        sys.argv = old_argv

    payload = json.loads((output_dir / "yaqa-rounding-manifest.json").read_text(encoding="utf-8"))
    stats = payload["projection_results"]["layer_1.gate_proj"]["expert_stats"]["expert_0"]
    assert stats["code_bits"] == 16
    assert (output_dir / "layer-00001-gate_proj.safetensors").exists()


def test_kronecker_collector_projection_residual_gradients_are_output_error_signal() -> None:
    module_path = Path(__file__).parents[1] / "benchmarks" / "collect_glm45_air_kronecker_hessian.py"
    spec = importlib.util.spec_from_file_location("collect_glm45_air_kronecker_hessian", module_path)
    assert spec is not None and spec.loader is not None
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)

    source_weight = np.array(
        [
            [1.0, -0.5, 0.25],
            [0.25, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    artifact_weight = np.array(
        [
            [0.75, -0.25, 0.5],
            [0.5, 0.0, 0.75],
        ],
        dtype=np.float32,
    )
    input_rows = np.array(
        [
            [1.0, 2.0, -1.0],
            [0.5, -1.0, 3.0],
        ],
        dtype=np.float32,
    )

    output_grad_rows = collector._projection_residual_output_gradients(
        source_weight,
        artifact_weight,
        input_rows,
    )

    expected = input_rows @ (artifact_weight - source_weight).T
    np.testing.assert_allclose(output_grad_rows, expected, atol=1.0e-6)
    h_in, h_out = yaqa_sketch_a_hessian_factors(input_rows, output_grad_rows)
    assert h_in.shape == (3, 3)
    assert h_out.shape == (2, 2)


def test_kronecker_collector_projection_residual_accepts_separate_source_and_artifact_inputs() -> None:
    module_path = Path(__file__).parents[1] / "benchmarks" / "collect_glm45_air_kronecker_hessian.py"
    spec = importlib.util.spec_from_file_location("collect_glm45_air_kronecker_hessian", module_path)
    assert spec is not None and spec.loader is not None
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)

    source_weight = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    artifact_weight = np.array([[0.5, 0.25], [1.5, -0.5]], dtype=np.float32)
    raw_input_rows = np.array([[2.0, -1.0]], dtype=np.float32)
    artifact_input_rows = np.array([[-1.0, 2.0]], dtype=np.float32)

    output_grad_rows = collector._projection_residual_output_gradients(
        source_weight,
        artifact_weight,
        artifact_input_rows,
        source_input_rows=raw_input_rows,
    )

    expected = artifact_input_rows @ artifact_weight.T - raw_input_rows @ source_weight.T
    np.testing.assert_allclose(output_grad_rows, expected, atol=1.0e-6)


def test_kronecker_collector_parses_layer_specific_expert_filters() -> None:
    module_path = Path(__file__).parents[1] / "benchmarks" / "collect_glm45_air_kronecker_hessian.py"
    spec = importlib.util.spec_from_file_location("collect_glm45_air_kronecker_hessian", module_path)
    assert spec is not None and spec.loader is not None
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)

    filters = collector._parse_layer_expert_filters(["14:75", "16:32", "16:66"])

    assert filters == {14: {75}, 16: {32, 66}}
    assert collector._allowed_experts_for_layer(14, None, filters) == {75}
    assert collector._allowed_experts_for_layer(15, {1, 2}, filters) == set()
    assert collector._allowed_experts_for_layer(15, {1, 2}, None) == {1, 2}


def test_yaqa_materializer_transforms_source_weight_into_rht_artifact_basis() -> None:
    module_path = Path(__file__).parents[1] / "benchmarks" / "materialize_glm45_air_yaqa_rounding.py"
    spec = importlib.util.spec_from_file_location("materialize_glm45_air_yaqa_rounding", module_path)
    assert spec is not None and spec.loader is not None
    materializer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(materializer)

    source_weight = np.arange(32, dtype=np.float32).reshape(2, 16)
    signs = deterministic_rht_signs(16, seed="tiny-rht-basis")

    class ProjectionLayer:
        rht_signs = signs

    transformed = materializer._source_weight_for_artifact_basis(source_weight, ProjectionLayer())

    np.testing.assert_allclose(transformed, apply_rht_np(source_weight, signs), atol=1.0e-6)
