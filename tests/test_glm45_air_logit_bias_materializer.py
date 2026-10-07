from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.io.continuous_sidecar import (
    continuous_sidecar_relpath,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)
from mlx_vq.io.logit_bias import load_logit_bias_sidecar, logit_bias_enabled
from mlx_vq.io.logit_bias import copy_declared_logit_bias_sidecar
from mlx_vq.io.logit_bias import write_logit_bias_artifact_manifest
from mlx_vq.io.logit_bias import write_logit_bias_sidecar


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "materialize_glm45_air_logit_bias.py"
    )
    spec = importlib.util.spec_from_file_location("logit_bias_materializer_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_materialize_logit_bias_preserves_seed_sidecars(tmp_path) -> None:
    cli = _load_cli()
    seed_dir = tmp_path / "seed"
    output_dir = tmp_path / "logit-biased"
    seed_dir.mkdir()
    (seed_dir / "layer-00041-up_proj.safetensors").write_bytes(b"seed-layer-shard")
    continuous_entry = write_continuous_sidecar(
        output_dir=seed_dir,
        layer=41,
        projection="up_proj",
        output_bias=mx.array([[0.125, -0.25]], dtype=mx.float32),
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed_dir,
        output_dir=seed_dir,
        sidecars=[continuous_entry],
        run_manifest={"kind": "test-seed"},
    )

    summary = cli.materialize_logit_bias(
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
        token_biases={565: 2.589, 220: -0.25},
    )

    assert summary["record_type"] == "air_logit_bias_materialization"
    assert summary["linked_group_count"] == 1
    assert summary["preserved_sidecar_count"] == 1
    assert summary["token_count"] == 2
    assert (output_dir / "layer-00041-up_proj.safetensors").is_symlink()
    assert (output_dir / continuous_sidecar_relpath(41, "up_proj")).exists()
    assert logit_bias_enabled(output_dir) is True

    sidecar = load_logit_bias_sidecar(output_dir, vocab_size=1024)
    assert sidecar is not None
    np.testing.assert_array_equal(np.array(sidecar.token_ids), [565, 220])
    np.testing.assert_allclose(np.array(sidecar.biases), [2.589, -0.25], rtol=1e-6)

    manifest = json.loads((output_dir / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert manifest["continuous_parameters"]["sidecars"][0]["path"] == continuous_sidecar_relpath(41, "up_proj")
    assert manifest["logit_bias"]["run"]["kind"] == "sparse_token_logit_bias"


def test_materialize_logit_bias_records_position_scope(tmp_path) -> None:
    cli = _load_cli()
    seed_dir = tmp_path / "seed"
    output_dir = tmp_path / "logit-biased-position0"
    seed_dir.mkdir()
    (seed_dir / "layer-00041-up_proj.safetensors").write_bytes(b"seed-layer-shard")

    summary = cli.materialize_logit_bias(
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
        token_biases={565: 2.589},
        position_indices=(0,),
    )

    assert summary["position_indices"] == [0]

    sidecar = load_logit_bias_sidecar(output_dir, vocab_size=1024)
    assert sidecar is not None
    assert sidecar.position_indices == (0,)

    manifest = json.loads((output_dir / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert manifest["logit_bias"]["scope"] == {"position_indices": [0]}
    assert manifest["logit_bias"]["run"]["scope"] == {"position_indices": [0]}


def test_materialize_logit_bias_records_token_position_scope(tmp_path) -> None:
    cli = _load_cli()
    seed_dir = tmp_path / "seed"
    output_dir = tmp_path / "logit-biased-token-positions"
    seed_dir.mkdir()
    (seed_dir / "layer-00041-up_proj.safetensors").write_bytes(b"seed-layer-shard")

    summary = cli.materialize_logit_bias(
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
        token_biases={304: 0.048, 18: 0.109, 2829: 0.179},
        token_position_indices={304: (4,), 18: (2,), 2829: (5,)},
    )

    assert summary["position_indices"] is None
    assert summary["token_position_indices"] == [
        {"token_id": 304, "position_indices": [4]},
        {"token_id": 18, "position_indices": [2]},
        {"token_id": 2829, "position_indices": [5]},
    ]

    sidecar = load_logit_bias_sidecar(output_dir, vocab_size=4096)
    assert sidecar is not None
    assert sidecar.position_indices is None
    assert sidecar.token_position_indices == ((4,), (2,), (5,))

    manifest = json.loads((output_dir / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert manifest["logit_bias"]["scope"] == {"token_position_indices": [[4], [2], [5]]}
    assert manifest["logit_bias"]["tokens"] == [
        {"token_id": 304, "bias": 0.048, "position_indices": [4]},
        {"token_id": 18, "bias": 0.109, "position_indices": [2]},
        {"token_id": 2829, "bias": 0.179, "position_indices": [5]},
    ]
    assert manifest["logit_bias"]["run"]["scope"] == {"token_position_indices": [[4], [2], [5]]}


def test_copy_declared_logit_bias_sidecar_preserves_continuation_manifest(tmp_path) -> None:
    seed_dir = tmp_path / "seed"
    output_dir = tmp_path / "continued"
    seed_dir.mkdir()
    sidecar = write_logit_bias_sidecar(
        output_dir=seed_dir,
        token_biases={565: 2.589},
        token_position_indices={565: (0,)},
    )
    write_logit_bias_artifact_manifest(
        seed_artifact_dir=seed_dir,
        output_dir=seed_dir,
        sidecar=sidecar,
        run_manifest={"kind": "seed-logit-bias"},
    )

    copied = copy_declared_logit_bias_sidecar(
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
        sidecars=[],
        run_manifest={"kind": "derived-continuation"},
    )

    assert copied is not None
    assert (output_dir / "logit_bias" / "token-bias.safetensors").exists()
    assert logit_bias_enabled(output_dir) is True
    copied_sidecar = load_logit_bias_sidecar(output_dir, vocab_size=1024)
    assert copied_sidecar is not None
    np.testing.assert_array_equal(np.array(copied_sidecar.token_ids), [565])
    np.testing.assert_allclose(np.array(copied_sidecar.biases), [2.589], rtol=1e-6)
    assert copied_sidecar.token_position_indices == ((0,),)
