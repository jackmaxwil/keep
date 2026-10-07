from __future__ import annotations

from typing import Literal

import mlx.core as mx
import numpy as np
import pytest

from ramp.nn.switch_linear import QuantizedVQSwitchLinear
from ramp.kernels.gather_vqmm import (
    gather_vqmm_m1_kernel,
    gather_vqmm_mma_cwdecode_blocks_kernel,
    gather_vqmm_mma_k32_cwdecode_blocks_kernel,
    gather_vqmm_mma_k64_cwdecode_blocks_kernel,
    gather_vqmm_mma_k128_cwdecode_blocks_kernel,
)
import ramp.ops.vq_switch as vq_switch
from ramp.ops.vq_switch import gather_vqmm, vq_switch_qmv


def _switch_fixture(seed: int = 314) -> tuple[QuantizedVQSwitchLinear, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weight = rng.normal(scale=0.04, size=(5, 6, 16)).astype(np.float32)
    x = rng.normal(size=(7, 16)).astype(np.float32)
    indices = np.array(
        [
            [0, 2, 2],
            [4, 1, 0],
            [3, 3, 1],
            [2, 4, 4],
            [1, 0, 3],
            [4, 4, 2],
            [0, 1, 3],
        ],
        dtype=np.int32,
    )
    return QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8), x, indices


def test_gather_vqmm_matches_unsorted_switch_contract() -> None:
    layer, x, indices = _switch_fixture()

    actual = gather_vqmm(
        mx.array(x),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(indices),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
    )
    expected = vq_switch_qmv(
        mx.array(x),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(indices),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
    )
    mx.eval(actual, expected)

    assert actual.shape == (7, 3, 6)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)


def test_gather_vqmm_sorted_path_scatters_back_to_unsorted_layout() -> None:
    layer, x, indices = _switch_fixture(seed=2718)

    unsorted = gather_vqmm(
        mx.array(x),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(indices),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        route_strategy="direct",
    )
    sorted_round_trip = gather_vqmm(
        mx.array(x),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(indices),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        route_strategy="sorted_tiled",
    )
    mx.eval(unsorted, sorted_round_trip)

    np.testing.assert_allclose(np.array(sorted_round_trip), np.array(unsorted), rtol=1e-5, atol=1e-5)


def test_sorted_route_helper_matches_gather_vqmm_after_manual_scatter() -> None:
    layer, x, indices = _switch_fixture(seed=2721)
    top_k = indices.shape[1]
    flat_rhs = indices.reshape(-1)
    flat_lhs = np.arange(flat_rhs.shape[0], dtype=np.int32) // top_k
    order = np.argsort(flat_rhs, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.shape[0])

    sorted_routes = vq_switch.gather_vqmm_sorted_routes(
        mx.array(x),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(flat_rhs[order]),
        mx.array(flat_lhs[order]),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        projection="gate_up",
    )
    scattered = sorted_routes[mx.array(inverse)].reshape((*indices.shape, 6))
    expected = gather_vqmm(
        mx.array(x),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(indices),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        route_strategy="sorted_tiled",
    )
    mx.eval(scattered, expected)

    assert sorted_routes.shape == (flat_rhs.shape[0], 6)
    np.testing.assert_allclose(np.array(scattered), np.array(expected), rtol=1e-5, atol=1e-5)


def test_sorted_route_helper_validates_explicit_metadata_when_requested() -> None:
    layer, x, _indices = _switch_fixture(seed=2721)
    x_mx = mx.array(x)
    sorted_lhs = mx.array([0, 1, 2], dtype=mx.int32)

    with pytest.raises(ValueError, match="sorted_rhs must be in"):
        vq_switch.gather_vqmm_sorted_routes(
            x_mx,
            layer.codes,
            layer.scales,
            layer.codebook,
            mx.array([0, 5, 5], dtype=mx.int32),
            sorted_lhs,
            input_dims=16,
            output_dims=6,
            group_size=8,
            code_bits=8,
            validate_indices=True,
        )

    with pytest.raises(ValueError, match="sorted_lhs must be in"):
        vq_switch.gather_vqmm_sorted_routes(
            x_mx,
            layer.codes,
            layer.scales,
            layer.codebook,
            mx.array([0, 1, 1], dtype=mx.int32),
            mx.array([0, x_mx.shape[0], 2], dtype=mx.int32),
            input_dims=16,
            output_dims=6,
            group_size=8,
            code_bits=8,
            validate_indices=True,
        )

    with pytest.raises(ValueError, match="sorted_rhs must be sorted"):
        vq_switch.gather_vqmm_sorted_routes(
            x_mx,
            layer.codes,
            layer.scales,
            layer.codebook,
            mx.array([1, 0, 1], dtype=mx.int32),
            sorted_lhs,
            input_dims=16,
            output_dims=6,
            group_size=8,
            code_bits=8,
            validate_indices=True,
        )


def test_sorted_route_helper_auto_uses_air_projection_roles(monkeypatch) -> None:
    rng = np.random.default_rng(2722)
    gate_layer = QuantizedVQSwitchLinear.from_weights(
        mx.array(rng.normal(scale=0.04, size=(5, 8, 16)).astype(np.float32)),
        group_size=8,
    )
    down_layer = QuantizedVQSwitchLinear.from_weights(
        mx.array(rng.normal(scale=0.04, size=(5, 16, 8)).astype(np.float32)),
        group_size=8,
    )
    calls: list[str] = []

    def fake_gate_up_blocks(
        x_arg,
        codes,
        scales,
        codebook,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        **kwargs,
    ):
        calls.append("down" if kwargs["output_dims"] > kwargs["input_dims"] else "gate_up")
        return mx.zeros((kwargs["route_count"], kwargs["output_dims"]), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_mma_cwdecode_blocks_kernel", fake_gate_up_blocks)

    gate_routes = 8192
    down_routes = 16384
    gate = vq_switch.gather_vqmm_sorted_routes(
        mx.zeros((gate_routes, 16), dtype=mx.float16),
        gate_layer.codes,
        gate_layer.scales,
        gate_layer.codebook,
        mx.zeros((gate_routes,), dtype=mx.int32),
        mx.arange(gate_routes, dtype=mx.int32),
        input_dims=16,
        output_dims=8,
        group_size=8,
        code_bits=8,
        projection="auto",
    )
    down = vq_switch.gather_vqmm_sorted_routes(
        mx.zeros((down_routes, 8), dtype=mx.float16),
        down_layer.codes,
        down_layer.scales,
        down_layer.codebook,
        mx.zeros((down_routes,), dtype=mx.int32),
        mx.arange(down_routes, dtype=mx.int32),
        input_dims=8,
        output_dims=16,
        group_size=8,
        code_bits=8,
        projection="auto",
    )
    mx.eval(gate, down)

    assert gate.shape == (gate_routes, 8)
    assert down.shape == (down_routes, 16)
    assert calls == ["gate_up", "down"]


def test_gather_vqmm_sorted_tiled_handles_mixed_expert_boundary_tile() -> None:
    layer, x, _indices = _switch_fixture(seed=2720)
    boundary_indices = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )

    direct = gather_vqmm(
        mx.array(x[: boundary_indices.shape[0]]),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(boundary_indices),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        route_strategy="direct",
    )
    sorted_round_trip = gather_vqmm(
        mx.array(x[: boundary_indices.shape[0]]),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(boundary_indices),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        route_strategy="sorted_tiled",
    )
    mx.eval(direct, sorted_round_trip)

    assert boundary_indices.size % 16 != 0
    np.testing.assert_allclose(np.array(sorted_round_trip), np.array(direct), rtol=1e-5, atol=1e-5)


def test_gather_vqmm_accepts_explicit_lhs_route_indices() -> None:
    layer, x, indices = _switch_fixture(seed=111)
    top_k = indices.shape[1]
    flat_rhs = indices.reshape(-1)
    flat_lhs = np.arange(flat_rhs.shape[0], dtype=np.int32) // top_k

    flat = gather_vqmm(
        mx.array(x),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(flat_rhs),
        lhs_indices=mx.array(flat_lhs),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
    )
    grouped = gather_vqmm(
        mx.array(x),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(indices),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
    )
    mx.eval(flat, grouped)

    np.testing.assert_allclose(
        np.array(flat).reshape(indices.shape[0], top_k, 6),
        np.array(grouped),
        rtol=1e-5,
        atol=1e-5,
    )


def test_gather_vqmm_explicit_lhs_uses_in_kernel_activation_indexing(monkeypatch) -> None:
    layer, x, indices = _switch_fixture(seed=112)
    top_k = indices.shape[1]
    flat_rhs = indices.reshape(-1)
    flat_lhs = np.arange(flat_rhs.shape[0], dtype=np.int32) // top_k
    calls: list[dict[str, object]] = []

    def fake_gather_vqmm_lhs_kernel(x_arg, codes, scales, codebook, rhs_indices, lhs_indices, **kwargs):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "rhs_shape": rhs_indices.shape,
                "lhs_shape": lhs_indices.shape,
                "kwargs": kwargs,
            }
        )
        return mx.zeros((rhs_indices.shape[0], 6), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_lhs_kernel", fake_gather_vqmm_lhs_kernel)

    actual = gather_vqmm(
        mx.array(x),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(flat_rhs),
        lhs_indices=mx.array(flat_lhs),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        codebook_duplication=4,
        implementation="metal",
    )
    mx.eval(actual)

    assert actual.shape == (flat_rhs.shape[0], 6)
    assert calls == [
        {
            "x_shape": (x.shape[0], 16),
            "rhs_shape": (flat_rhs.shape[0],),
            "lhs_shape": (flat_lhs.shape[0],),
            "kwargs": {
                "input_dims": 16,
                "output_dims": 6,
                "group_size": 8,
                "code_bits": 8,
                "codebook_duplication": 4,
            },
        }
    ]


def test_gather_vqmm_auto_routes_small_batches_direct_and_large_batches_sorted(monkeypatch) -> None:
    layer, x, indices = _switch_fixture(seed=114)
    small_calls: list[dict[str, object]] = []
    large_calls: list[dict[str, object]] = []

    def fake_m1_direct(x_arg, codes, scales, codebook, rhs_indices, **kwargs):
        small_calls.append(
            {
                "x_shape": x_arg.shape,
                "rhs_shape": rhs_indices.shape,
                "rows_per_threadgroup": kwargs["rows_per_threadgroup"],
                "use_threadgroup_codebook": kwargs["use_threadgroup_codebook"],
                "use_decoded_codebook": kwargs["use_decoded_codebook"],
            }
        )
        return mx.zeros((x_arg.shape[0], rhs_indices.shape[1], 6), dtype=x_arg.dtype)

    def fake_sorted(x_arg, codes, scales, codebook, rhs_indices, lhs_indices, **kwargs):
        large_calls.append(
            {
                "x_shape": x_arg.shape,
                "rhs_shape": rhs_indices.shape,
                "lhs_shape": lhs_indices.shape,
                "tile": (kwargs["m_tile"], kwargs["n_tile"], kwargs["k_tile_dims"]),
            }
        )
        return mx.zeros((rhs_indices.shape[0], 6), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_m1_kernel", fake_m1_direct)
    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_tiled_down_kernel", fake_sorted)

    small = gather_vqmm(
        mx.array(x[:1]),
        layer.codes,
        layer.scales,
        layer.codebook,
        mx.array(indices[:1]),
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    large_x = mx.array(np.tile(x[:1], (64, 1)))
    large_indices = mx.array(np.tile(indices[:1], (64, 1)))
    large = gather_vqmm(
        large_x,
        layer.codes,
        layer.scales,
        layer.codebook,
        large_indices,
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    mx.eval(small, large)

    assert small_calls == [
        {
            "x_shape": (1, 16),
            "rhs_shape": (1, 3),
            "rows_per_threadgroup": 32,
            "use_threadgroup_codebook": False,
            "use_decoded_codebook": True,
        }
    ]
    route_count = large_indices.shape[0] * large_indices.shape[1]
    assert large_calls == [
        {
            "x_shape": large_x.shape,
            "rhs_shape": (route_count,),
            "lhs_shape": (route_count,),
            "tile": (16, 8, 8),
        }
    ]


def test_gather_vqmm_direct_strategy_wins_over_sorted_indices(monkeypatch) -> None:
    layer, x, indices = _switch_fixture(seed=118)
    calls: list[dict[str, object]] = []

    def fake_direct(x_arg, codes, scales, codebook, rhs_indices, **kwargs):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "rhs_shape": rhs_indices.shape,
                "kwargs": kwargs,
            }
        )
        return mx.zeros((x_arg.shape[0], rhs_indices.shape[1], 6), dtype=x_arg.dtype)

    def fail_sorted(*_args, **_kwargs):
        raise AssertionError("route_strategy='direct' must not dispatch sorted_tiled")

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_kernel", fake_direct)
    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_tiled_down_kernel", fail_sorted)

    tiled_x = mx.array(np.tile(x[:1], (64, 1)))
    tiled_indices = mx.array(np.tile(indices[:1], (64, 1)))
    actual = gather_vqmm(
        tiled_x,
        layer.codes,
        layer.scales,
        layer.codebook,
        tiled_indices,
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        sorted_indices=True,
        route_strategy="direct",
    )
    mx.eval(actual)

    assert actual.shape == (64, 3, 6)
    assert calls == [
        {
            "x_shape": (64, 16),
            "rhs_shape": (64, 3),
            "kwargs": {
                "input_dims": 16,
                "output_dims": 6,
                "group_size": 8,
                "code_bits": 8,
                "codebook_duplication": 1,
            },
        }
    ]


def test_gather_vqmm_auto_uses_block_mma_for_large_down_projection(monkeypatch) -> None:
    rng = np.random.default_rng(115)
    weight = rng.normal(scale=0.04, size=(5, 32, 16)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(4096, 16)).astype(np.float32))
    indices = mx.array(np.tile(np.array([[0, 1]], dtype=np.int32), (4096, 1)))
    calls: list[dict[str, object]] = []

    def fake_mma_blocks(
        x_arg,
        codes,
        scales,
        codebook,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        **kwargs,
    ):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "lhs_shape": lhs_indices.shape,
                "tile_shapes": (tile_experts.shape, tile_offsets.shape, tile_counts.shape),
                "kwargs": kwargs,
            }
        )
        return mx.zeros((kwargs["route_count"], 32), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_mma_cwdecode_blocks_kernel", fake_mma_blocks)

    actual = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=32,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    mx.eval(actual)

    route_count = 4096 * 2
    max_tiles = (route_count + 31) // 32 + 5
    assert actual.shape == (4096, 2, 32)
    assert calls == [
        {
            "x_shape": x.shape,
            "lhs_shape": (route_count,),
            "tile_shapes": ((max_tiles,), (max_tiles,), (max_tiles,)),
            "kwargs": {
                "route_count": route_count,
                "input_dims": 16,
                "output_dims": 32,
                "group_size": 8,
                "route_tile": 32,
            },
        }
    ]


def test_gather_vqmm_auto_uses_large_block_mma_for_very_large_down_projection(monkeypatch) -> None:
    rng = np.random.default_rng(121)
    weight = rng.normal(scale=0.04, size=(5, 32, 16)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(8192, 16)).astype(np.float32))
    indices = mx.array(np.tile(np.array([[0, 1]], dtype=np.int32), (8192, 1)))
    calls: list[dict[str, object]] = []

    def fake_large_blocks(
        x_arg,
        codes,
        scales,
        codebook,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        **kwargs,
    ):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "lhs_shape": lhs_indices.shape,
                "tile_shapes": (tile_experts.shape, tile_offsets.shape, tile_counts.shape),
                "kwargs": kwargs,
            }
        )
        return mx.zeros((kwargs["route_count"], 32), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_mma_cwdecode_blocks_kernel", fake_large_blocks)

    actual = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=32,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    mx.eval(actual)

    route_count = 8192 * 2
    max_tiles = (route_count + 63) // 64 + 5
    assert actual.shape == (8192, 2, 32)
    assert calls == [
        {
            "x_shape": x.shape,
            "lhs_shape": (route_count,),
            "tile_shapes": ((max_tiles,), (max_tiles,), (max_tiles,)),
            "kwargs": {
                "route_count": route_count,
                "input_dims": 16,
                "output_dims": 32,
                "group_size": 8,
                "route_tile": 64,
            },
        }
    ]


def test_gather_vqmm_auto_uses_block_mma_for_large_gate_up_projection(monkeypatch) -> None:
    rng = np.random.default_rng(116)
    weight = rng.normal(scale=0.04, size=(5, 8, 16)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(8192, 16)).astype(np.float32))
    indices = mx.array(np.tile(np.array([[0]], dtype=np.int32), (8192, 1)))
    calls: list[dict[str, object]] = []

    def fake_mma_blocks(
        x_arg,
        codes,
        scales,
        codebook,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        **kwargs,
    ):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "lhs_shape": lhs_indices.shape,
                "tile_shapes": (tile_experts.shape, tile_offsets.shape, tile_counts.shape),
                "kwargs": kwargs,
            }
        )
        return mx.zeros((kwargs["route_count"], 8), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_mma_cwdecode_blocks_kernel", fake_mma_blocks)

    actual = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=8,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    mx.eval(actual)

    route_count = 8192
    max_tiles = (route_count + 31) // 32 + 5
    assert actual.shape == (8192, 1, 8)
    assert calls == [
        {
            "x_shape": x.shape,
            "lhs_shape": (route_count,),
            "tile_shapes": ((max_tiles,), (max_tiles,), (max_tiles,)),
            "kwargs": {
                "route_count": route_count,
                "input_dims": 16,
                "output_dims": 8,
                "group_size": 8,
                "route_tile": 32,
            },
        }
    ]


def test_gather_vqmm_auto_uses_k32_cwdecode_for_aligned_gate_up_projection(monkeypatch) -> None:
    rng = np.random.default_rng(124)
    weight = rng.normal(scale=0.04, size=(5, 8, 32)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(8192, 32)).astype(np.float32))
    indices = mx.array(np.tile(np.array([[0]], dtype=np.int32), (8192, 1)))
    calls: list[dict[str, object]] = []

    def fake_k32_blocks(
        x_arg,
        codes,
        scales,
        codebook,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        **kwargs,
    ):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "lhs_shape": lhs_indices.shape,
                "tile_shapes": (tile_experts.shape, tile_offsets.shape, tile_counts.shape),
                "kwargs": kwargs,
            }
        )
        return mx.zeros((kwargs["route_count"], 8), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_mma_k32_cwdecode_blocks_kernel", fake_k32_blocks)

    actual = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=32,
        output_dims=8,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    mx.eval(actual)

    route_count = 8192
    max_tiles = (route_count + 31) // 32 + 5
    assert actual.shape == (8192, 1, 8)
    assert calls == [
        {
            "x_shape": x.shape,
            "lhs_shape": (route_count,),
            "tile_shapes": ((max_tiles,), (max_tiles,), (max_tiles,)),
            "kwargs": {
                "route_count": route_count,
                "input_dims": 32,
                "output_dims": 8,
                "group_size": 8,
                "route_tile": 32,
            },
        }
    ]


def test_gather_vqmm_auto_uses_k64_cwdecode_for_64_aligned_gate_up_projection(monkeypatch) -> None:
    rng = np.random.default_rng(126)
    weight = rng.normal(scale=0.04, size=(5, 8, 64)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(8192, 64)).astype(np.float32))
    indices = mx.array(np.tile(np.array([[0]], dtype=np.int32), (8192, 1)))
    calls: list[dict[str, object]] = []

    def fake_k64_blocks(
        x_arg,
        codes,
        scales,
        codebook,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        **kwargs,
    ):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "lhs_shape": lhs_indices.shape,
                "tile_shapes": (tile_experts.shape, tile_offsets.shape, tile_counts.shape),
                "kwargs": kwargs,
            }
        )
        return mx.zeros((kwargs["route_count"], 8), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_mma_k64_cwdecode_blocks_kernel", fake_k64_blocks)

    actual = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=64,
        output_dims=8,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    mx.eval(actual)

    route_count = 8192
    max_tiles = (route_count + 31) // 32 + 5
    assert actual.shape == (8192, 1, 8)
    assert calls == [
        {
            "x_shape": x.shape,
            "lhs_shape": (route_count,),
            "tile_shapes": ((max_tiles,), (max_tiles,), (max_tiles,)),
            "kwargs": {
                "route_count": route_count,
                "input_dims": 64,
                "output_dims": 8,
                "group_size": 8,
                "route_tile": 32,
                "row_groups": 8,
            },
        }
    ]


def test_gather_vqmm_block_mma_matches_direct_when_enabled(monkeypatch) -> None:
    rng = np.random.default_rng(117)
    weight = rng.normal(scale=0.04, size=(5, 8, 16)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(64, 16)).astype(np.float32))
    indices = mx.array(
        rng.integers(0, 5, size=(64, 3), dtype=np.int32),
    )
    monkeypatch.setattr("ramp.ops.vq_switch._GATE_UP_MMA_ROUTE_THRESHOLD", 16)

    actual = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=8,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    expected = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=8,
        group_size=8,
        code_bits=8,
        route_strategy="direct",
    )
    mx.eval(actual, expected)

    assert actual.shape == (64, 3, 8)
    actual_np = np.array(actual)
    expected_np = np.array(expected)
    dot = float(np.sum(actual_np * expected_np))
    norm = float(np.sqrt(np.sum(actual_np * actual_np) * np.sum(expected_np * expected_np)))
    assert float(np.max(np.abs(actual_np - expected_np))) < 2e-4
    assert dot / norm >= 0.99999


def test_gather_vqmm_down_block_mma_matches_direct_when_enabled(monkeypatch) -> None:
    rng = np.random.default_rng(119)
    weight = rng.normal(scale=0.04, size=(5, 32, 16)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(64, 16)).astype(np.float32))
    indices = mx.array(
        rng.integers(0, 5, size=(64, 3), dtype=np.int32),
    )
    monkeypatch.setattr("ramp.ops.vq_switch._DOWN_BLOCK_MMA_ROUTE_THRESHOLD", 16)

    actual = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=32,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    expected = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=32,
        group_size=8,
        code_bits=8,
        route_strategy="direct",
    )
    mx.eval(actual, expected)

    assert actual.shape == (64, 3, 32)
    actual_np = np.array(actual)
    expected_np = np.array(expected)
    dot = float(np.sum(actual_np * expected_np))
    norm = float(np.sqrt(np.sum(actual_np * actual_np) * np.sum(expected_np * expected_np)))
    assert float(np.max(np.abs(actual_np - expected_np))) < 2e-4
    assert dot / norm >= 0.99999


def test_gather_vqmm_large_block_mma_matches_direct_when_enabled(monkeypatch) -> None:
    rng = np.random.default_rng(120)
    weight = rng.normal(scale=0.04, size=(5, 32, 16)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(64, 16)).astype(np.float32))
    indices = mx.array(
        rng.integers(0, 5, size=(64, 3), dtype=np.int32),
    )
    monkeypatch.setattr("ramp.ops.vq_switch._DOWN_BLOCK_MMA_ROUTE_THRESHOLD", 16)
    monkeypatch.setattr("ramp.ops.vq_switch._LARGE_BLOCK_MMA_ROUTE_THRESHOLD", 16)

    actual = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=32,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    expected = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=32,
        group_size=8,
        code_bits=8,
        route_strategy="direct",
    )
    mx.eval(actual, expected)

    assert actual.shape == (64, 3, 32)
    actual_np = np.array(actual)
    expected_np = np.array(expected)
    dot = float(np.sum(actual_np * expected_np))
    norm = float(np.sqrt(np.sum(actual_np * actual_np) * np.sum(expected_np * expected_np)))
    assert float(np.max(np.abs(actual_np - expected_np))) < 2e-4
    assert dot / norm >= 0.99999


def test_gather_vqmm_cwdecode_block_probe_matches_direct() -> None:
    rng = np.random.default_rng(122)
    weight = rng.normal(scale=0.04, size=(5, 32, 16)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(64, 16)).astype(np.float32))
    indices = mx.array(
        rng.integers(0, 5, size=(64, 3), dtype=np.int32),
    )
    flat_rhs = indices.reshape((-1,))
    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // indices.shape[1]).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=5,
        route_tile=32,
    )

    sorted_actual = gather_vqmm_mma_cwdecode_blocks_kernel(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        sorted_lhs,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_count=flat_rhs.shape[0],
        input_dims=16,
        output_dims=32,
        group_size=8,
        route_tile=32,
    )
    inverse = mx.argsort(order)
    actual = sorted_actual[inverse].reshape((64, 3, 32))
    expected = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=16,
        output_dims=32,
        group_size=8,
        code_bits=8,
        route_strategy="direct",
    )
    mx.eval(actual, expected)

    actual_np = np.array(actual)
    expected_np = np.array(expected)
    dot = float(np.sum(actual_np * expected_np))
    norm = float(np.sqrt(np.sum(actual_np * actual_np) * np.sum(expected_np * expected_np)))
    assert float(np.max(np.abs(actual_np - expected_np))) < 2e-4
    assert dot / norm >= 0.99999


def test_gather_vqmm_k32_cwdecode_block_probe_matches_direct() -> None:
    rng = np.random.default_rng(123)
    weight = rng.normal(scale=0.04, size=(5, 32, 32)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=8)
    x = mx.array(rng.normal(size=(64, 32)).astype(np.float32))
    indices = mx.array(
        rng.integers(0, 5, size=(64, 3), dtype=np.int32),
    )
    flat_rhs = indices.reshape((-1,))
    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // indices.shape[1]).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=5,
        route_tile=32,
    )

    sorted_actual = gather_vqmm_mma_k32_cwdecode_blocks_kernel(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        sorted_lhs,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_count=flat_rhs.shape[0],
        input_dims=32,
        output_dims=32,
        group_size=8,
        route_tile=32,
    )
    inverse = mx.argsort(order)
    actual = sorted_actual[inverse].reshape((64, 3, 32))
    expected = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=32,
        output_dims=32,
        group_size=8,
        code_bits=8,
        route_strategy="direct",
    )
    mx.eval(actual, expected)

    actual_np = np.array(actual)
    expected_np = np.array(expected)
    dot = float(np.sum(actual_np * expected_np))
    norm = float(np.sqrt(np.sum(actual_np * actual_np) * np.sum(expected_np * expected_np)))
    assert float(np.max(np.abs(actual_np - expected_np))) < 2e-4
    assert dot / norm >= 0.99999


def test_gather_vqmm_k64_cwdecode_block_probe_matches_direct() -> None:
    rng = np.random.default_rng(125)
    weight = rng.normal(scale=0.04, size=(5, 17, 64)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=16)
    x = mx.array(rng.normal(size=(64, 64)).astype(np.float32))
    indices = mx.array(
        rng.integers(0, 5, size=(64, 3), dtype=np.int32),
    )
    flat_rhs = indices.reshape((-1,))
    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // indices.shape[1]).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=5,
        route_tile=64,
    )

    sorted_actual = gather_vqmm_mma_k64_cwdecode_blocks_kernel(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        sorted_lhs,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_count=flat_rhs.shape[0],
        input_dims=64,
        output_dims=17,
        group_size=16,
        route_tile=64,
    )
    inverse = mx.argsort(order)
    actual = sorted_actual[inverse].reshape((64, 3, 17))
    expected = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=64,
        output_dims=17,
        group_size=16,
        code_bits=8,
        route_strategy="direct",
    )
    mx.eval(actual, expected)

    actual_np = np.array(actual)
    expected_np = np.array(expected)
    dot = float(np.sum(actual_np * expected_np))
    norm = float(np.sqrt(np.sum(actual_np * actual_np) * np.sum(expected_np * expected_np)))
    assert float(np.max(np.abs(actual_np - expected_np))) < 2e-4
    assert dot / norm >= 0.99999


def _run_sorted_k64_cwdecode_blocks(
    x: mx.array,
    codes: mx.array,
    scales: mx.array,
    indices: mx.array,
    *,
    num_experts: int,
    input_dims: int,
    output_dims: int,
    group_size: int,
    route_tile: Literal[32, 64],
    row_groups: Literal[4, 8],
) -> mx.array:
    flat_rhs = indices.reshape((-1,))
    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // indices.shape[1]).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=num_experts,
        route_tile=route_tile,
    )
    sorted_actual = gather_vqmm_mma_k64_cwdecode_blocks_kernel(
        x,
        codes,
        scales,
        None,
        sorted_lhs,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_count=flat_rhs.shape[0],
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        route_tile=route_tile,
        row_groups=row_groups,
    )
    inverse = mx.argsort(order)
    return sorted_actual[inverse].reshape((*indices.shape, output_dims))


def test_gather_vqmm_k64_row_groups8_probe_matches_existing_k64_and_direct() -> None:
    rng = np.random.default_rng(128)
    num_experts = 5
    input_dims = 64
    output_dims = 65
    group_size = 16
    x = mx.array(rng.normal(size=(17, input_dims)).astype(np.float32))
    codes = mx.array(
        rng.integers(0, 256, size=(num_experts, output_dims, input_dims // 8), dtype=np.uint8),
    )
    scales = mx.array(
        rng.normal(loc=0.02, scale=0.004, size=(num_experts, output_dims, input_dims // group_size)).astype(
            np.float16
        ),
    )
    indices = mx.array(rng.integers(0, num_experts, size=(17, 3), dtype=np.int32))

    row_groups4 = _run_sorted_k64_cwdecode_blocks(
        x,
        codes,
        scales,
        indices,
        num_experts=num_experts,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        route_tile=32,
        row_groups=4,
    )
    row_groups8 = _run_sorted_k64_cwdecode_blocks(
        x,
        codes,
        scales,
        indices,
        num_experts=num_experts,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        route_tile=32,
        row_groups=8,
    )
    direct = gather_vqmm(
        x,
        codes,
        scales,
        None,
        indices,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        code_bits=8,
        route_strategy="direct",
    )
    mx.eval(row_groups4, row_groups8, direct)

    row_groups8_np = np.array(row_groups8)
    row_groups4_np = np.array(row_groups4)
    direct_np = np.array(direct)
    dot = float(np.sum(row_groups8_np * direct_np))
    norm = float(np.sqrt(np.sum(row_groups8_np * row_groups8_np) * np.sum(direct_np * direct_np)))
    np.testing.assert_allclose(row_groups8_np, row_groups4_np, rtol=0, atol=0)
    assert float(np.max(np.abs(row_groups8_np - direct_np))) < 3e-4
    assert dot / norm >= 0.99999


def test_gather_vqmm_k64_row_groups8_probe_matches_existing_k64_at_air_gate_up_shape() -> None:
    rng = np.random.default_rng(129)
    num_experts = 3
    input_dims = 4096
    output_dims = 1408
    group_size = 512
    x = mx.array(rng.normal(size=(4, input_dims)).astype(np.float16))
    codes = mx.array(
        rng.integers(0, 256, size=(num_experts, output_dims, input_dims // 8), dtype=np.uint8),
    )
    scales = mx.array(
        rng.normal(loc=0.015, scale=0.003, size=(num_experts, output_dims, input_dims // group_size)).astype(
            np.float16
        ),
    )
    indices = mx.array(rng.integers(0, num_experts, size=(4, 8), dtype=np.int32))

    row_groups4 = _run_sorted_k64_cwdecode_blocks(
        x,
        codes,
        scales,
        indices,
        num_experts=num_experts,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        route_tile=32,
        row_groups=4,
    )
    row_groups8 = _run_sorted_k64_cwdecode_blocks(
        x,
        codes,
        scales,
        indices,
        num_experts=num_experts,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        route_tile=32,
        row_groups=8,
    )
    mx.eval(row_groups4, row_groups8)

    row_groups8_np = np.array(row_groups8, dtype=np.float32)
    row_groups4_np = np.array(row_groups4, dtype=np.float32)
    dot = float(np.sum(row_groups8_np * row_groups4_np))
    norm = float(np.sqrt(np.sum(row_groups8_np * row_groups8_np) * np.sum(row_groups4_np * row_groups4_np)))
    np.testing.assert_allclose(row_groups8_np, row_groups4_np, rtol=0, atol=0)
    assert dot / norm >= 0.99999


def test_gather_vqmm_k64_row_groups8_rejects_route_tile64() -> None:
    x = mx.zeros((1, 64), dtype=mx.float16)
    codes = mx.zeros((1, 8, 8), dtype=mx.uint8)
    scales = mx.ones((1, 8, 4), dtype=mx.float16)

    with pytest.raises(ValueError, match="row_groups=8 requires route_tile=32"):
        gather_vqmm_mma_k64_cwdecode_blocks_kernel(
            x,
            codes,
            scales,
            None,
            mx.array([0], dtype=mx.int32),
            mx.array([0], dtype=mx.int32),
            mx.array([0], dtype=mx.int32),
            mx.array([1], dtype=mx.int32),
            route_count=1,
            input_dims=64,
            output_dims=8,
            group_size=16,
            route_tile=64,
            row_groups=8,
        )


def test_gather_vqmm_air_down_352_dispatches_k64_cwdecode(monkeypatch) -> None:
    num_experts = 3
    input_dims = 1408
    output_dims = 4096
    group_size = 352
    tokens = 2048
    top_k = 8
    x = mx.zeros((tokens, input_dims), dtype=mx.float16)
    codes = mx.zeros((num_experts, output_dims, input_dims // 8), dtype=mx.uint8)
    scales = mx.ones((num_experts, output_dims, input_dims // group_size), dtype=mx.float16)
    indices = mx.array(np.tile(np.arange(top_k, dtype=np.int32) % num_experts, (tokens, 1)))
    calls: list[dict[str, object]] = []

    def fake_k64_blocks(
        x_arg,
        codes_arg,
        scales_arg,
        codebook_arg,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        **kwargs,
    ):
        calls.append(
            {
                "x_shape": x_arg.shape,
                "codes_shape": codes_arg.shape,
                "scales_shape": scales_arg.shape,
                "lhs_shape": lhs_indices.shape,
                "tile_shapes": (tile_experts.shape, tile_offsets.shape, tile_counts.shape),
                "kwargs": kwargs,
            }
        )
        return mx.zeros((kwargs["route_count"], output_dims), dtype=x_arg.dtype)

    monkeypatch.setattr("ramp.ops.vq_switch.gather_vqmm_mma_k64_cwdecode_blocks_kernel", fake_k64_blocks)

    actual = gather_vqmm(
        x,
        codes,
        scales,
        None,
        indices,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        code_bits=8,
        route_strategy="auto",
    )
    mx.eval(actual)

    route_count = tokens * top_k
    max_tiles = (route_count + 63) // 64 + num_experts
    assert actual.shape == (tokens, top_k, output_dims)
    assert calls == [
        {
            "x_shape": x.shape,
            "codes_shape": codes.shape,
            "scales_shape": scales.shape,
            "lhs_shape": (route_count,),
            "tile_shapes": ((max_tiles,), (max_tiles,), (max_tiles,)),
            "kwargs": {
                "route_count": route_count,
                "input_dims": input_dims,
                "output_dims": output_dims,
                "group_size": group_size,
                "route_tile": 64,
                "row_groups": 4,
            },
        }
    ]


def test_gather_vqmm_k64_cwdecode_matches_direct_at_air_down_group352() -> None:
    rng = np.random.default_rng(130)
    num_experts = 4
    input_dims = 1408
    output_dims = 4096
    group_size = 352
    tokens = 2
    top_k = 8
    x = mx.array(rng.normal(size=(tokens, input_dims)).astype(np.float16))
    codes = mx.array(
        rng.integers(0, 256, size=(num_experts, output_dims, input_dims // 8), dtype=np.uint8),
    )
    scale_groups = input_dims // group_size
    assert scale_groups == 4
    assert group_size % 64 != 0
    scale_ramp = np.array([0.01, 0.04, 0.09, 0.17], dtype=np.float16)
    scales_np = np.broadcast_to(scale_ramp, (num_experts, output_dims, scale_groups)).copy()
    scales_np *= rng.uniform(0.85, 1.15, size=(num_experts, output_dims, 1)).astype(np.float16)
    scales = mx.array(scales_np)
    indices = mx.array(rng.integers(0, num_experts, size=(tokens, top_k), dtype=np.int32))

    row_groups4 = _run_sorted_k64_cwdecode_blocks(
        x,
        codes,
        scales,
        indices,
        num_experts=num_experts,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        route_tile=64,
        row_groups=4,
    )
    direct = gather_vqmm(
        x,
        codes,
        scales,
        None,
        indices,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        code_bits=8,
        route_strategy="direct",
    )
    mx.eval(row_groups4, direct)

    actual_np = np.array(row_groups4, dtype=np.float32)
    direct_np = np.array(direct, dtype=np.float32)
    dot = float(np.sum(actual_np * direct_np))
    norm = float(np.sqrt(np.sum(actual_np * actual_np) * np.sum(direct_np * direct_np)))
    assert float(np.max(np.abs(actual_np - direct_np))) < 3e-3
    assert dot / norm >= 0.99999


def test_gather_vqmm_k128_cwdecode_block_probe_matches_direct() -> None:
    rng = np.random.default_rng(127)
    weight = rng.normal(scale=0.04, size=(5, 17, 128)).astype(np.float32)
    layer = QuantizedVQSwitchLinear.from_weights(mx.array(weight), group_size=16)
    x = mx.array(rng.normal(size=(64, 128)).astype(np.float32))
    indices = mx.array(
        rng.integers(0, 5, size=(64, 3), dtype=np.int32),
    )
    flat_rhs = indices.reshape((-1,))
    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // indices.shape[1]).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=5,
        route_tile=32,
    )

    sorted_actual = gather_vqmm_mma_k128_cwdecode_blocks_kernel(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        sorted_lhs,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_count=flat_rhs.shape[0],
        input_dims=128,
        output_dims=17,
        group_size=16,
    )
    inverse = mx.argsort(order)
    actual = sorted_actual[inverse].reshape((64, 3, 17))
    expected = gather_vqmm(
        x,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices,
        input_dims=128,
        output_dims=17,
        group_size=16,
        code_bits=8,
        route_strategy="direct",
    )
    mx.eval(actual, expected)

    actual_np = np.array(actual)
    expected_np = np.array(expected)
    dot = float(np.sum(actual_np * expected_np))
    norm = float(np.sqrt(np.sum(actual_np * actual_np) * np.sum(expected_np * expected_np)))
    assert float(np.max(np.abs(actual_np - expected_np))) < 3e-4
    assert dot / norm >= 0.99999


def test_gather_vqmm_m1_kernel_matches_gather_contract() -> None:
    layer, x, indices = _switch_fixture(seed=113)
    x_m1 = mx.array(x[:1])
    indices_m1 = mx.array(indices[:1])

    expected = vq_switch_qmv(
        x_m1,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices_m1,
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
    )
    actual = gather_vqmm_m1_kernel(
        x_m1,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices_m1,
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        rows_per_threadgroup=4,
    )
    mx.eval(actual, expected)

    assert actual.shape == (1, 3, 6)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)


def test_gather_vqmm_m1_auto_accepts_default_codebook() -> None:
    layer, x, indices = _switch_fixture(seed=116)
    x_m1 = mx.array(x[:1])
    indices_m1 = mx.array(indices[:1])

    explicit = gather_vqmm(
        x_m1,
        layer.codes,
        layer.scales,
        layer.codebook,
        indices_m1,
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    defaulted = gather_vqmm(
        x_m1,
        layer.codes,
        layer.scales,
        None,
        indices_m1,
        input_dims=16,
        output_dims=6,
        group_size=8,
        code_bits=8,
        route_strategy="auto",
    )
    mx.eval(explicit, defaulted)

    np.testing.assert_allclose(np.array(defaulted), np.array(explicit), rtol=1e-5, atol=1e-5)
