from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from keep.io.continuous_sidecar import (
    load_conversion_manifest,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)
from ramp.nn.switch_linear import QuantizedVQSwitchLinear
from keep.quality.mlx_surrogate import SwitchLinearSidecar, SwitchLinearSurrogate


def _load_fitter_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "fit_glm45_air_block_local_sidecar.py"
    spec = importlib.util.spec_from_file_location("fit_glm45_air_block_local_sidecar_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fitter = _load_fitter_module()


def test_select_rows_accepts_explicit_row_indices() -> None:
    rows = [{"prompt_id": f"row_{idx}"} for idx in range(5)]

    selected = fitter._select_rows(rows, max_rows=None, row_indices=(3, 1))

    assert selected == [
        (3, {"prompt_id": "row_3"}),
        (1, {"prompt_id": "row_1"}),
    ]


def test_select_rows_rejects_out_of_range_row_indices() -> None:
    rows = [{"prompt_id": "row_0"}]

    try:
        fitter._select_rows(rows, max_rows=None, row_indices=(1,))
    except IndexError as error:
        assert "out of range" in str(error)
    else:
        raise AssertionError("out-of-range row index should raise")


def test_parse_row_indices_validates_input() -> None:
    assert fitter._parse_row_indices("3, 1,0") == (3, 1, 0)
    for value in ("", "-1", "abc"):
        try:
            fitter._parse_row_indices(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{value!r} should raise")


def test_resolve_source_dir_downloads_full_snapshot_when_omitted(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "snapshots" / "abc123"

    def fake_snapshot_download(*, repo_id, revision):
        assert repo_id == "zai-org/GLM-4.5-Air"
        assert revision == "abc123"
        return str(snapshot)

    monkeypatch.setattr(fitter, "snapshot_download", fake_snapshot_download)

    assert fitter._resolve_source_dir_arg(
        source_dir=None,
        model_id="zai-org/GLM-4.5-Air",
        revision="abc123",
    ) == snapshot


def test_resolve_source_dir_arg_prefers_explicit_path(monkeypatch, tmp_path) -> None:
    explicit = tmp_path / "source"

    def fail_download(**_kwargs):
        raise AssertionError("explicit source dir should not download")

    monkeypatch.setattr(fitter, "snapshot_download", fail_download)

    assert fitter._resolve_source_dir_arg(
        source_dir=str(explicit),
        model_id="zai-org/GLM-4.5-Air",
        revision="abc123",
    ) == explicit


def test_source_projection_targets_down_uses_source_gate_up_hidden(monkeypatch, tmp_path) -> None:
    layer = 41
    source_x = np.array(
        [
            [0.5, -1.0, 0.25, 2.0],
            [1.5, 0.0, -0.5, 1.0],
        ],
        dtype=np.float32,
    )
    artifact_down_input = np.zeros((2, 2, 3), dtype=np.float32)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)

    weights: dict[str, np.ndarray] = {}
    for expert in (0, 1):
        gate = np.array(
            [
                [1.0 + expert, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.25 + expert, 0.0],
                [0.5, 0.0, 1.0, -0.25],
            ],
            dtype=np.float32,
        )
        up = np.array(
            [
                [0.25, 0.0, 1.0, 0.0],
                [0.0, -0.5 - expert, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.25],
            ],
            dtype=np.float32,
        )
        down = np.array(
            [
                [1.0, 0.5, 0.0],
                [0.0, -1.0, 0.25 + expert],
            ],
            dtype=np.float32,
        )
        weights[f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"] = gate
        weights[f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"] = up
        weights[f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"] = down

    def fake_read(_source_dir, _index, name):
        return mx.array(weights[name])

    monkeypatch.setattr(fitter, "read_indexed_safetensors_tensor_mlx", fake_read)

    target, source_reads, peak_bytes = fitter._source_projection_targets(
        source_dir=tmp_path,
        index=SimpleNamespace(),
        layer=layer,
        projection="down_proj",
        x=artifact_down_input,
        source_x=source_x,
        indices=indices,
    )
    mx.eval(target)

    expected = np.empty((2, 2, 2), dtype=np.float32)
    for token_idx in range(indices.shape[0]):
        for route_idx in range(indices.shape[1]):
            expert = int(indices[token_idx, route_idx])
            gate = weights[f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"]
            up = weights[f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"]
            down = weights[f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"]
            hidden = fitter._silu_np(gate @ source_x[token_idx]) * (up @ source_x[token_idx])
            expected[token_idx, route_idx] = down @ hidden

    np.testing.assert_allclose(np.array(target), expected, rtol=1e-6, atol=1e-6)
    assert source_reads == 6
    assert peak_bytes == max(value.nbytes for value in weights.values())


def test_low_rank_block_local_fit_writes_ranked_sidecar(tmp_path) -> None:
    layer = QuantizedVQSwitchLinear(
        input_dims=8,
        output_dims=3,
        num_experts=2,
        codes=mx.zeros((2, 3, 1), dtype=mx.uint8),
        scales=mx.zeros((2, 3, 1), dtype=mx.float32),
        group_size=8,
        use_gather_vqmm=False,
    )
    x = mx.array(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    indices = mx.array(np.array([[0], [0], [1], [1]], dtype=np.int32))
    left = mx.array(
        np.array(
            [
                [[1.0], [2.0], [3.0]],
                [[-1.0], [0.5], [1.5]],
            ],
            dtype=np.float32,
        )
    )
    right = mx.array(
        np.array(
            [
                [[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    )
    target = SwitchLinearSurrogate.from_layer(
        layer,
        sidecar=SwitchLinearSidecar(low_rank_left=left, low_rank_right=right),
    )(x, indices)

    result = fitter.fit_low_rank_residual_sidecar_least_squares(
        layer,
        x,
        indices,
        target,
        rank=1,
    )
    sidecar = write_continuous_sidecar(
        output_dir=tmp_path,
        layer=16,
        projection="up_proj",
        low_rank_left=result["sidecar"].low_rank_left,
        low_rank_right=result["sidecar"].low_rank_right,
    )

    assert sidecar["layer"] == 16
    assert sidecar["projection"] == "up_proj"
    assert sidecar["low_rank_rank"] == 1
    assert result["actual_ranks"] == (1, 1)
    assert (tmp_path / sidecar["path"]).exists()


def test_block_local_materialization_preserves_seed_sidecars_except_replaced(tmp_path) -> None:
    seed = tmp_path / "seed"
    output = tmp_path / "output"
    preserved_seed_sidecar = write_continuous_sidecar(
        output_dir=seed,
        layer=45,
        projection="gate_proj",
        output_bias=mx.ones((2, 3), dtype=mx.float32),
    )
    replaced_seed_sidecar = write_continuous_sidecar(
        output_dir=seed,
        layer=16,
        projection="up_proj",
        output_bias=mx.full((2, 4), 7.0, dtype=mx.float32),
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=tmp_path / "base",
        output_dir=seed,
        sidecars=[preserved_seed_sidecar, replaced_seed_sidecar],
    )

    sidecar_entry, linked_group_count, preserved_sidecar_count = fitter._materialize_block_local_sidecar_artifact(
        seed_artifact_dir=seed,
        output_dir=output,
        layer=16,
        projection="up_proj",
        fitted_sidecar=SwitchLinearSidecar(
            low_rank_left=mx.zeros((2, 4, 1), dtype=mx.float32),
            low_rank_right=mx.zeros((2, 1, 8), dtype=mx.float32),
        ),
        run_manifest={"kind": "unit_test"},
    )

    manifest = load_conversion_manifest(output)
    continuous = manifest["continuous_parameters"]
    sidecars = continuous["sidecars"]
    sidecar_keys = {(int(item["layer"]), item["projection"]) for item in sidecars}

    assert linked_group_count == 0
    assert preserved_sidecar_count == 1
    assert sidecar_entry["low_rank_rank"] == 1
    assert sidecar_keys == {(45, "gate_proj"), (16, "up_proj")}
    assert (output / preserved_seed_sidecar["path"]).exists()
    assert continuous["run"]["preserved_sidecar_count"] == 1
