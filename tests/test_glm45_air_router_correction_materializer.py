from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from keep.io.continuous_sidecar import (
    continuous_sidecar_relpath,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)
from keep.io.router_correction import load_router_correction_sidecar, router_corrections_enabled


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "materialize_glm45_air_router_correction.py"
    )
    spec = importlib.util.spec_from_file_location("router_correction_materializer_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_materialize_router_correction_preserves_seed_sidecars(tmp_path) -> None:
    cli = _load_cli()
    seed_dir = tmp_path / "seed"
    output_dir = tmp_path / "router-corrected"
    seed_dir.mkdir()
    (seed_dir / "layer-00036-up_proj.safetensors").write_bytes(b"seed-layer-shard")
    continuous_entry = write_continuous_sidecar(
        output_dir=seed_dir,
        layer=36,
        projection="up_proj",
        output_bias=mx.array([[0.125, -0.25]], dtype=mx.float32),
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed_dir,
        output_dir=seed_dir,
        sidecars=[continuous_entry],
        run_manifest={"kind": "test-seed"},
    )
    disagreement_json = tmp_path / "disagreement.json"
    disagreement_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "air_route_trace_disagreement_summary",
                "layers": {
                    "36": {
                        "token_count": 8,
                        "expert_bias_delta": [
                            {"expert": 4, "delta": 0.5},
                            {"expert": 5, "delta": -0.25},
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary = cli.materialize_router_correction(
        disagreement_json=disagreement_json,
        seed_artifact_dir=seed_dir,
        output_dir=output_dir,
        num_experts=8,
        scale=2.0,
        max_abs_delta=0.75,
    )

    assert summary["layers"] == [36]
    assert summary["linked_group_count"] == 1
    assert summary["preserved_sidecar_count"] == 1
    assert (output_dir / "layer-00036-up_proj.safetensors").is_symlink()
    assert (output_dir / continuous_sidecar_relpath(36, "up_proj")).exists()
    assert router_corrections_enabled(output_dir) is True
    loaded = load_router_correction_sidecar(output_dir, layer=36, num_experts=8)

    assert loaded is not None
    assert loaded.expert_bias_delta is not None
    np.testing.assert_allclose(
        np.array(loaded.expert_bias_delta),
        [0.0, 0.0, 0.0, 0.0, 0.75, -0.5, 0.0, 0.0],
    )

    manifest = json.loads((output_dir / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert manifest["continuous_parameters"]["sidecars"][0]["path"] == continuous_sidecar_relpath(36, "up_proj")
    assert manifest["router_corrections"]["run"]["kind"] == "route_trace_expert_bias_delta"
