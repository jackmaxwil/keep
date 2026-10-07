from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest


def _load_materializer_module():
    module_path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "materialize_glm45_air_non_expert_precision.py"
    )
    spec = importlib.util.spec_from_file_location(
        "materialize_glm45_air_non_expert_precision", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_expert_precision_materializer_links_seed_and_records_policy(tmp_path: Path) -> None:
    materializer = _load_materializer_module()
    seed = tmp_path / "seed"
    seed.mkdir()
    mx.save_safetensors(
        str(seed / "layer-00045-gate_proj.safetensors"),
        {"model.layers.45.mlp.switch_mlp.gate_proj.codes": mx.zeros((1, 1, 1), dtype=mx.uint8)},
    )
    sidecar_dir = seed / "continuous_params"
    sidecar_dir.mkdir()
    mx.save_safetensors(str(sidecar_dir / "layer-00045-gate_proj.safetensors"), {"output_bias": mx.zeros((1, 1))})
    (seed / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "schema_version": 1,
                    "enabled": True,
                    "sidecars": [
                        {
                            "layer": 45,
                            "projection": "gate_proj",
                            "path": "continuous_params/layer-00045-gate_proj.safetensors",
                            "tensors": ["output_bias"],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "candidate"
    manifest = materializer.materialize_non_expert_precision_candidate(
        seed_artifact_dir=seed,
        output_dir=output,
        surfaces=["embed_tokens", "lm_head"],
        dtype="bf16",
        reason="top1 gap attributed to non-expert surfaces",
    )

    assert manifest["non_expert_precision"]["enabled"] is True
    assert manifest["non_expert_precision"]["surfaces"] == ["embed_tokens", "lm_head"]
    assert manifest["non_expert_precision"]["dtype"] == "bf16"
    assert os.path.islink(output / "layer-00045-gate_proj.safetensors")
    assert (output / "continuous_params" / "layer-00045-gate_proj.safetensors").exists()


def test_non_expert_precision_materializer_rejects_attention_surface(tmp_path: Path) -> None:
    materializer = _load_materializer_module()
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "conversion-manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported non-expert surface 'attention'"):
        materializer.materialize_non_expert_precision_candidate(
            seed_artifact_dir=seed,
            output_dir=tmp_path / "candidate",
            surfaces=["attention"],
            dtype="bf16",
        )
