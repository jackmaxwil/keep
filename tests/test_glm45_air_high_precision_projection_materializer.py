from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

from keep.io.load import inspect_safetensors, load_high_precision_switch_linear


def _load_materializer_module():
    module_path = Path(__file__).parents[1] / "benchmarks" / "materialize_glm45_air_high_precision_projection.py"
    spec = importlib.util.spec_from_file_location("materialize_glm45_air_high_precision_projection", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_seed_group(path: Path, *, layer: int, projection: str) -> None:
    prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
    codes = np.zeros((2, 3, 1), dtype=np.uint8)
    scales = np.ones((2, 3, 1), dtype=np.float16)
    mx.save_safetensors(
        str(path),
        {
            f"{prefix}.codes": mx.array(codes),
            f"{prefix}.scales": mx.array(scales),
        },
    )


def test_high_precision_projection_materializer_links_seed_and_rewrites_selected_projection(tmp_path: Path) -> None:
    materializer = _load_materializer_module()
    seed = tmp_path / "seed"
    seed.mkdir()
    for projection in ("gate_proj", "up_proj", "down_proj"):
        _write_seed_group(seed / f"layer-00041-{projection}.safetensors", layer=41, projection=projection)
    sidecar_dir = seed / "continuous_params"
    sidecar_dir.mkdir()
    (sidecar_dir / "manifest.json").write_text('{"sidecars":[]}\n', encoding="utf-8")

    source = tmp_path / "source"
    source.mkdir()
    shard = source / "model-00001-of-00001.safetensors"
    arrays = {
        "model.layers.41.mlp.experts.0.gate_proj.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "model.layers.41.mlp.experts.1.gate_proj.weight": (100 + np.arange(12, dtype=np.float32)).reshape(3, 4),
    }
    save_file(arrays, shard)
    index = {
        "metadata": {"total_size": int(sum(value.nbytes for value in arrays.values()))},
        "weight_map": {name: shard.name for name in arrays},
    }
    index_path = source / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    output = tmp_path / "candidate"
    manifest = materializer.materialize_high_precision_projection_candidate(
        seed_artifact_dir=seed,
        output_dir=output,
        projections=[(41, "gate_proj")],
        source_dir=source,
        index_path=index_path,
        expert_count=2,
    )

    target = output / "layer-00041-gate_proj.safetensors"
    linked = output / "layer-00041-up_proj.safetensors"
    layer = load_high_precision_switch_linear(target, "model.layers.41.mlp.switch_mlp.gate_proj")
    inspection = inspect_safetensors(target)
    stored = np.asarray(layer.weight.astype(mx.float32))

    assert manifest["projection_count"] == 1
    assert manifest["seed_artifact_mutated"] is False
    assert manifest["high_precision_projections"][0]["weight_shape"] == [2, 3, 4]
    assert inspection.tensors["model.layers.41.mlp.switch_mlp.gate_proj.weight"].dtype in {"BF16", "BFLOAT16"}
    assert layer.tier == "high"
    np.testing.assert_allclose(stored, np.stack([arrays[name] for name in arrays], axis=0), rtol=0, atol=0.25)
    assert os.path.islink(linked)
    assert (output / "continuous_params" / "manifest.json").exists()
