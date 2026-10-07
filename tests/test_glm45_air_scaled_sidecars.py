from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np

from keep.io.continuous_sidecar import (
    load_conversion_manifest,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)


def _load_materializer_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "materialize_glm45_air_scaled_sidecars.py"
    spec = importlib.util.spec_from_file_location("materialize_glm45_air_scaled_sidecars_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


materializer = _load_materializer_module()


def test_materialize_scaled_sidecars_scales_declared_source_sidecars(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    source = tmp_path / "source"
    output = tmp_path / "output"
    (seed / "layer-00018-gate_proj.safetensors").parent.mkdir(parents=True)
    (seed / "layer-00018-gate_proj.safetensors").write_bytes(b"seed-shard")

    l18 = write_continuous_sidecar(
        output_dir=source,
        layer=18,
        projection="gate_proj",
        output_bias=mx.array([[1.0, -2.0], [3.0, 4.0]], dtype=mx.float32),
    )
    l41 = write_continuous_sidecar(
        output_dir=source,
        layer=41,
        projection="up_proj",
        output_bias=mx.array([[2.0, 8.0]], dtype=mx.float32),
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed,
        output_dir=source,
        sidecars=[l18, l41],
        run_manifest={
            "kind": "merged_block_local_sidecars",
            "train_split": "selection",
            "train_row_count": 8,
            "max_positions": 8,
            "trainable": "output_bias",
        },
    )

    result = materializer.materialize_scaled_sidecars(
        seed_artifact_dir=seed,
        source_artifact_dir=source,
        output_dir=output,
        scale_policy={"layer_18": 1.0, "layer_41": 0.75},
    )

    l18_arrays = mx.load(str(output / l18["path"]))
    l41_arrays = mx.load(str(output / l41["path"]))
    np.testing.assert_allclose(np.array(l18_arrays["output_bias"]), [[1.0, -2.0], [3.0, 4.0]])
    np.testing.assert_allclose(np.array(l41_arrays["output_bias"]), [[1.5, 6.0]])
    assert result["sidecar_count"] == 2
    assert result["linked_group_count"] == 1
    assert (output / "layer-00018-gate_proj.safetensors").is_symlink()

    manifest = load_conversion_manifest(output)
    continuous = manifest["continuous_parameters"]
    assert continuous["seed_artifact_dir"] == str(seed)
    assert continuous["run"]["kind"] == "scaled_block_local_sidecar_alpha_sweep"
    assert continuous["run"]["source_artifact"] == str(source)
    assert continuous["run"]["scale_policy"] == {"layer_18": 1.0, "layer_41": 0.75}
    assert continuous["run"]["base_train_row_count"] == 8
