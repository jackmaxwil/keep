from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mlx.core as mx

from keep.io.continuous_sidecar import (
    continuous_sidecar_relpath,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)
from keep.vq.e8 import e8_1bit_packed
from keep.io.sparse_residual import (
    load_switch_linear_sparse_residual_rows,
    write_sparse_residual_artifact_manifest,
    write_sparse_residual_rows,
)
from ramp.nn.switch_linear import QuantizedVQSwitchLinear


def _tiny_layer() -> QuantizedVQSwitchLinear:
    rng = np.random.default_rng(20260627)
    return QuantizedVQSwitchLinear(
        input_dims=16,
        output_dims=4,
        num_experts=3,
        codes=mx.array(rng.integers(0, 256, size=(3, 4, 2), dtype=np.uint8)),
        scales=mx.array(rng.uniform(0.25, 0.75, size=(3, 4, 1)).astype(np.float32)),
        codebook=mx.array(e8_1bit_packed()),
        group_size=16,
        use_gather_vqmm=False,
    )


def _load_sparse_materializer_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "materialize_glm45_air_vq_sparse_residual.py"
    )
    spec = importlib.util.spec_from_file_location("sparse_residual_materializer_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sparse_residual_rows_add_expected_outputs() -> None:
    layer = _tiny_layer()
    x = mx.array(np.arange(32, dtype=np.float32).reshape(2, 16) / 10.0)
    indices = mx.array(np.array([[1, 2], [0, 1]], dtype=np.int32))
    baseline = layer(x, indices)
    expert_indices = mx.array(np.array([1, 2], dtype=np.int32))
    output_indices = mx.array(np.array([3, 0], dtype=np.int32))
    residual_values = np.zeros((2, 16), dtype=np.float32)
    residual_values[0, 5] = 0.5
    residual_values[1, 7] = -0.25
    values = mx.array(residual_values)

    layer.set_sparse_residual_rows(
        expert_indices=expert_indices,
        output_indices=output_indices,
        values=values,
    )
    actual = layer(x, indices)
    mx.eval(baseline, actual)

    expected = np.array(baseline)
    expected[0, 0, 3] += float(np.array(x)[0, 5] * 0.5)
    expected[0, 1, 0] += float(np.array(x)[0, 7] * -0.25)
    expected[1, 1, 3] += float(np.array(x)[1, 5] * 0.5)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-3, atol=3e-4)


def test_sparse_residual_manifest_loader_requires_enabled_manifest(tmp_path) -> None:
    residual_entry = write_sparse_residual_rows(
        output_dir=tmp_path,
        layer=41,
        projection="gate_proj",
        expert_indices=mx.array(np.array([1], dtype=np.int32)),
        output_indices=mx.array(np.array([2], dtype=np.int32)),
        values=mx.ones((1, 16), dtype=mx.float32),
        input_dims=16,
        output_dims=4,
        num_experts=3,
    )
    assert (
        load_switch_linear_sparse_residual_rows(
            tmp_path,
            layer=41,
            projection="gate_proj",
            input_dims=16,
            output_dims=4,
            num_experts=3,
        )
        is None
    )
    write_sparse_residual_artifact_manifest(
        seed_manifest={},
        output_dir=tmp_path,
        residuals=[residual_entry],
    )
    loaded = load_switch_linear_sparse_residual_rows(
        tmp_path,
        layer=41,
        projection="gate_proj",
        input_dims=16,
        output_dims=4,
        num_experts=3,
    )
    assert loaded is not None
    np.testing.assert_array_equal(np.array(loaded.expert_indices), np.array([1], dtype=np.int32))


def test_sparse_residual_materializer_preserves_seed_continuous_sidecars(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cli = _load_sparse_materializer_cli()
    seed_dir = tmp_path / "seed"
    output_dir = tmp_path / "sparse"
    plan_json = tmp_path / "plan.json"
    seed_dir.mkdir()
    (seed_dir / "layer-00041-down_proj.safetensors").write_bytes(b"seed-layer-shard")
    continuous_entry = write_continuous_sidecar(
        output_dir=seed_dir,
        layer=18,
        projection="gate_proj",
        output_bias=mx.array([[0.125, -0.25]], dtype=mx.float32),
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed_dir,
        output_dir=seed_dir,
        sidecars=[continuous_entry],
        run_manifest={"kind": "test-seed"},
    )
    plan_json.write_text(
        json.dumps(
            {
                "summary": {
                    "records": [
                        {
                            "layer": 41,
                            "probe_target": {"key": "report_route_000:0"},
                            "route_source_sparse_residual_plans": [
                                {
                                    "token_index": 0,
                                    "route_rank": 0,
                                    "expert": 93,
                                    "projection": "down_proj",
                                    "source_metric": "source_weighted_route_contribution",
                                    "rows": [
                                        {
                                            "output_index": 11,
                                            "desired_weighted_correction": 1.1,
                                            "residual_values": [0.5, 0.5],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    fake_projection = SimpleNamespace(input_dims=2, output_dims=16, num_experts=128)
    fake_switch_glu = SimpleNamespace(down_proj=fake_projection)
    monkeypatch.setattr(cli, "_load_switch_glu", lambda _seed, _layer: fake_switch_glu)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_glm45_air_vq_sparse_residual.py",
            "--seed-artifact-dir",
            str(seed_dir),
            "--output-dir",
            str(output_dir),
            "--plan-json",
            str(plan_json),
            "--plan-target",
            "report_route_000:0",
            "--plan-expert",
            "93",
            "--plan-route-rank",
            "0",
            "--plan-max-rows",
            "1",
        ],
    )

    cli.main()
    capsys.readouterr()

    assert (output_dir / "layer-00041-down_proj.safetensors").is_symlink()
    assert (output_dir / continuous_sidecar_relpath(18, "gate_proj")).exists()
    manifest = json.loads((output_dir / "conversion-manifest.json").read_text(encoding="utf-8"))
    assert manifest["continuous_parameters"]["sidecars"][0]["path"] == continuous_sidecar_relpath(18, "gate_proj")
