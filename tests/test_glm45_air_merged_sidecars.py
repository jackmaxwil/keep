from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.io.continuous_sidecar import (
    load_conversion_manifest,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)


def _load_materializer_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "materialize_glm45_air_merged_sidecars.py"
    spec = importlib.util.spec_from_file_location("materialize_glm45_air_merged_sidecars_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


materializer = _load_materializer_module()


def _write_source_sidecar(
    artifact_dir: Path,
    *,
    seed: Path,
    layer: int,
    projection: str,
    value: float,
) -> None:
    sidecar = write_continuous_sidecar(
        output_dir=artifact_dir,
        layer=layer,
        projection=projection,
        output_bias=mx.array([[value, value + 1.0]], dtype=mx.float32),
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed,
        output_dir=artifact_dir,
        sidecars=[sidecar],
        run_manifest={
            "kind": "block_local_sidecar_least_squares",
            "layer": layer,
            "projection": projection,
            "trainable": "output_bias",
            "train_row_count": 8,
        },
    )


def test_materialize_merged_sidecars_copies_declared_source_sidecars(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    source_l18 = tmp_path / "l18-gate"
    source_l41 = tmp_path / "l41-up"
    output = tmp_path / "merged"
    (seed / "layer-00018-gate_proj.safetensors").parent.mkdir(parents=True)
    (seed / "layer-00018-gate_proj.safetensors").write_bytes(b"seed-shard")
    _write_source_sidecar(
        source_l18,
        seed=seed,
        layer=18,
        projection="gate_proj",
        value=1.0,
    )
    _write_source_sidecar(
        source_l41,
        seed=seed,
        layer=41,
        projection="up_proj",
        value=4.0,
    )

    result = materializer.materialize_merged_sidecars(
        seed_artifact_dir=seed,
        source_artifact_dirs=[source_l18, source_l41],
        output_dir=output,
        run_overrides={
            "train_split": "selection",
            "max_positions": 8,
            "train_row_count": 8,
            "trainable": "output_bias",
        },
    )

    assert result["sidecar_count"] == 2
    assert result["linked_group_count"] == 1
    assert result["layers"] == [18, 41]
    assert (output / "layer-00018-gate_proj.safetensors").is_symlink()
    l18_arrays = mx.load(str(output / "continuous_params/layer-00018-gate_proj.safetensors"))
    l41_arrays = mx.load(str(output / "continuous_params/layer-00041-up_proj.safetensors"))
    np.testing.assert_allclose(np.array(l18_arrays["output_bias"]), [[1.0, 2.0]])
    np.testing.assert_allclose(np.array(l41_arrays["output_bias"]), [[4.0, 5.0]])

    manifest = load_conversion_manifest(output)
    continuous = manifest["continuous_parameters"]
    assert continuous["seed_artifact_dir"] == str(seed)
    assert continuous["run"]["kind"] == "merged_block_local_sidecars"
    assert continuous["run"]["source_artifacts"] == [str(source_l18), str(source_l41)]
    assert continuous["run"]["train_split"] == "selection"
    assert continuous["run"]["max_positions"] == 8
