from __future__ import annotations

import numpy as np
import mlx.core as mx

import mlx_vq.ops.vq_switch as vq_switch


def test_nax_e8p_sorted_steel_route_plan_uses_m32_for_air_down_small_window() -> None:
    assert vq_switch._nax_e8p_sorted_steel_route_plan(
        route_count=8_192,
        input_dims=1_408,
        output_dims=4_096,
        group_size=352,
    ) == (32, "m32n64")

    assert vq_switch._nax_e8p_sorted_steel_route_plan(
        route_count=32_768,
        input_dims=1_408,
        output_dims=4_096,
        group_size=352,
    ) == (64, "steel")

    assert vq_switch._nax_e8p_sorted_steel_route_plan(
        route_count=8_192,
        input_dims=4_096,
        output_dims=1_408,
        group_size=512,
    ) == (64, "steel")


def test_inverse_permutation_matches_argsort_inverse() -> None:
    order = mx.array([3, 0, 4, 1, 2], dtype=mx.uint32)

    inverse = vq_switch._inverse_permutation(order)
    mx.eval(inverse)

    assert np.array(inverse).astype(int).tolist() == [1, 3, 4, 0, 2]


def test_gather_nax_e8p_sorted_routes_dispatches_steel_for_air_down_no_gather_small_window(
    monkeypatch,
) -> None:
    calls: list[tuple[str, list[int]]] = []

    monkeypatch.setattr(vq_switch.nax, "is_available", lambda: True)

    def fake_steel(
        sorted_x: mx.array,
        codes: mx.array,
        scales: mx.array,
        tile_experts: mx.array,
        tile_offsets: mx.array,
        tile_counts: mx.array,
        codebook: mx.array | None = None,
        *,
        group_size: int,
        stream=None,
    ) -> mx.array:
        del scales, tile_experts, tile_offsets, codebook, group_size, stream
        calls.append(("steel", np.array(tile_counts).astype(int).tolist()))
        return mx.zeros((sorted_x.shape[0], codes.shape[1]), dtype=mx.float16)

    def fake_m32n64(
        sorted_x: mx.array,
        codes: mx.array,
        scales: mx.array,
        tile_experts: mx.array,
        tile_offsets: mx.array,
        tile_counts: mx.array,
        codebook: mx.array | None = None,
        *,
        group_size: int,
        stream=None,
    ) -> mx.array:
        del scales, tile_experts, tile_offsets, codebook, group_size, stream
        calls.append(("m32n64", np.array(tile_counts).astype(int).tolist()))
        return mx.zeros((sorted_x.shape[0], codes.shape[1]), dtype=mx.float16)

    monkeypatch.setattr(vq_switch.nax, "nax_e8p_fp16_sorted_steel_matmul", fake_steel)
    monkeypatch.setattr(
        vq_switch.nax,
        "nax_e8p_fp16_sorted_steel_m32n64_matmul",
        fake_m32n64,
    )

    route_count = 65
    x = mx.zeros((route_count, 1_408), dtype=mx.float16)
    codes = mx.zeros((1, 4_096, 176), dtype=mx.uint16)
    scales = mx.ones((1, 4_096, 4), dtype=mx.float16)
    sorted_rhs = mx.zeros((route_count,), dtype=mx.int32)
    sorted_lhs = mx.arange(route_count, dtype=mx.int32)

    out = vq_switch.gather_vqmm_sorted_routes(
        x,
        codes,
        scales,
        None,
        sorted_rhs,
        sorted_lhs,
        input_dims=1_408,
        output_dims=4_096,
        group_size=352,
        code_bits=16,
        implementation="nax_e8p",
        projection="down",
    )
    mx.eval(out)

    assert out.shape == (route_count, 4_096)
    assert calls == [("steel", [64, 1, 0])]


def test_gather_nax_e8p_sorted_routes_dispatches_m32_for_air_down_token_gather_small_window(
    monkeypatch,
) -> None:
    calls: list[tuple[str, list[int]]] = []

    monkeypatch.setattr(vq_switch.nax, "is_available", lambda: True)

    def fake_steel(
        sorted_x: mx.array,
        codes: mx.array,
        scales: mx.array,
        tile_experts: mx.array,
        tile_offsets: mx.array,
        tile_counts: mx.array,
        codebook: mx.array | None = None,
        *,
        group_size: int,
        stream=None,
    ) -> mx.array:
        del scales, tile_experts, tile_offsets, codebook, group_size, stream
        calls.append(("steel", np.array(tile_counts).astype(int).tolist()))
        return mx.zeros((sorted_x.shape[0], codes.shape[1]), dtype=mx.float16)

    def fake_m32n64(
        sorted_x: mx.array,
        codes: mx.array,
        scales: mx.array,
        tile_experts: mx.array,
        tile_offsets: mx.array,
        tile_counts: mx.array,
        codebook: mx.array | None = None,
        *,
        group_size: int,
        stream=None,
    ) -> mx.array:
        del scales, tile_experts, tile_offsets, codebook, group_size, stream
        calls.append(("m32n64", np.array(tile_counts).astype(int).tolist()))
        return mx.zeros((sorted_x.shape[0], codes.shape[1]), dtype=mx.float16)

    monkeypatch.setattr(vq_switch.nax, "nax_e8p_fp16_sorted_steel_matmul", fake_steel)
    monkeypatch.setattr(
        vq_switch.nax,
        "nax_e8p_fp16_sorted_steel_m32n64_matmul",
        fake_m32n64,
    )

    route_count = 65
    x = mx.zeros((9, 1_408), dtype=mx.float16)
    codes = mx.zeros((1, 4_096, 176), dtype=mx.uint16)
    scales = mx.ones((1, 4_096, 4), dtype=mx.float16)
    sorted_rhs = mx.zeros((route_count,), dtype=mx.int32)
    sorted_lhs = mx.arange(route_count, dtype=mx.int32) % 9

    out = vq_switch.gather_vqmm_sorted_routes(
        x,
        codes,
        scales,
        None,
        sorted_rhs,
        sorted_lhs,
        input_dims=1_408,
        output_dims=4_096,
        group_size=352,
        code_bits=16,
        implementation="nax_e8p",
        projection="down",
    )
    mx.eval(out)

    assert out.shape == (route_count, 4_096)
    assert calls == [("m32n64", [32, 32, 1, 0])]
