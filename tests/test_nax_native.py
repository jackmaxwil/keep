from __future__ import annotations

import pytest

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import decode_weight_matrix, e8_1bit_packed, e8p_packed_abs_grid
from mlx_vq.kernels import nax
from mlx_vq.kernels.e8p_rhs_layout import (
    decode_e8p_expert_kblock_factor_reuse_rhs_tile,
    component_stream_sorted_matmul_oracle,
    decode_e8p_rhs_tile,
    decode_e8p_sign_plane_abs_index_rhs_tile,
    decode_e8p_sign_nibble_abs_index_rhs_tile,
    decode_e8p_sign_nibble_micro_lut_rhs_tile,
    decode_e8p_split_byte_factor_reuse_rhs_tile,
    decode_e8p_split_byte_rhs_tile,
    pack_e8p_component_stream_rhs_tiles,
    pack_e8p_expert_kblock_factor_reuse_rhs_tiles,
    pack_e8p_sign_plane_abs_index_rhs_tiles,
    pack_e8p_sign_nibble_abs_index_rhs_tiles,
    pack_e8p_sign_nibble_micro_lut_rhs_tiles,
    pack_e8p_split_byte_factor_reuse_rhs_tiles,
    pack_e8p_rhs_tiles,
    pack_e8p_split_byte_rhs_tiles,
)
import mlx_vq.ops.vq_switch as vq_switch


def test_nax_native_wrapper_reports_unavailable_before_build_or_smoke_matches_matmul() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    x = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float16)
    weight_t = mx.array([[0.5, 1.0], [1.5, -1.0]], dtype=mx.float16)

    observed = nax.predecoded_fp16_matmul(x, weight_t)
    expected = mx.matmul(x, weight_t)
    mx.eval(observed, expected)

    assert nax.build_info().startswith("vq_nax_ext scaffold linked against MLX")
    assert mx.allclose(observed, expected)


def test_nax_native_gather_mm_smoke_matches_mlx_gather_mm() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    x = mx.array(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[-1.0, 0.5]],
        ],
        dtype=mx.float16,
    )
    weight_t = mx.array(
        [
            [[0.5, 1.0], [1.5, -1.0]],
            [[-0.5, 2.0], [0.25, 0.75]],
        ],
        dtype=mx.float16,
    )
    rhs_indices = mx.array([0, 1, 1], dtype=mx.int32)

    observed = nax.predecoded_fp16_gather_mm(x, weight_t, rhs_indices)
    expected = mx.gather_mm(x, weight_t, rhs_indices=rhs_indices, sorted_indices=True)
    mx.eval(observed, expected)

    assert observed.shape == expected.shape
    assert mx.allclose(observed, expected)


def test_nax_native_runtime_tensorops_tile_matches_mlx_matmul() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    x = (mx.arange(32 * 16, dtype=mx.float32).reshape(32, 16) % 7).astype(mx.float16)
    weight_t = (
        (mx.arange(16 * 16, dtype=mx.float32).reshape(16, 16) * 3 + 1) % 11
    ).astype(mx.float16)

    observed = nax.nax_fp16_matmul_tile(x, weight_t)
    expected = mx.matmul(x, weight_t)
    mx.eval(observed, expected)

    assert observed.shape == expected.shape
    assert mx.allclose(observed, expected, rtol=1e-3, atol=1e-3)


def test_nax_native_fused_e8_tensorops_tile_matches_dequant_matmul() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260625)
    x_np = rng.normal(size=(32, 16)).astype(np.float16)
    codes_np = rng.integers(0, 256, size=(16, 2), dtype=np.uint8)
    scales_np = rng.uniform(0.01, 0.08, size=(16, 2)).astype(np.float16)
    codebook_np = e8_1bit_packed()
    weight = decode_weight_matrix(codes_np, scales_np, code_bits=8, codebook=codebook_np)

    x = mx.array(x_np, dtype=mx.float16)
    codes = mx.array(codes_np)
    scales = mx.array(scales_np)
    codebook = mx.array(codebook_np, dtype=mx.uint32)

    observed = nax.nax_e8_fp16_matmul_tile(
        x,
        codes,
        scales,
        codebook,
        group_size=8,
    )
    expected = mx.matmul(x, mx.array(weight.T, dtype=mx.float16))
    mx.eval(observed, expected)

    assert observed.shape == expected.shape
    assert mx.allclose(observed, expected, rtol=1e-3, atol=1e-3)


def test_nax_native_fused_e8_tensorops_dense_matches_dequant_matmul_with_tails() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260626)
    x_np = rng.normal(size=(33, 32)).astype(np.float16)
    codes_np = rng.integers(0, 256, size=(17, 4), dtype=np.uint8)
    scales_np = rng.uniform(0.01, 0.08, size=(17, 4)).astype(np.float16)
    codebook_np = e8_1bit_packed()
    weight = decode_weight_matrix(codes_np, scales_np, code_bits=8, codebook=codebook_np)

    x = mx.array(x_np, dtype=mx.float16)
    codes = mx.array(codes_np)
    scales = mx.array(scales_np)
    codebook = mx.array(codebook_np, dtype=mx.uint32)

    observed = nax.nax_e8_fp16_matmul(
        x,
        codes,
        scales,
        codebook,
        group_size=8,
    )
    expected = mx.matmul(x, mx.array(weight.T, dtype=mx.float16))
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8_tensorops_routed_matches_dequant_reference() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260627)
    experts = 4
    tokens = 11
    routes = 50
    output_dims = 17
    input_dims = 32
    group_size = 8
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 256, size=(experts, output_dims, input_dims // 8), dtype=np.uint8)
    scales_np = rng.uniform(0.01, 0.08, size=(experts, output_dims, input_dims // group_size)).astype(
        np.float16
    )
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    codebook_np = e8_1bit_packed()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8_fp16_routed_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        mx.array(sorted_lhs_np, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=8,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8_steel_routed_matches_dequant_reference() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260630)
    experts = 4
    tokens = 11
    routes = 50
    output_dims = 17
    input_dims = 64
    group_size = 8
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 256, size=(experts, output_dims, input_dims // 8), dtype=np.uint8)
    scales_np = rng.uniform(0.01, 0.08, size=(experts, output_dims, input_dims // group_size)).astype(
        np.float16
    )
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    codebook_np = e8_1bit_packed()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8_fp16_routed_steel_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        mx.array(sorted_lhs_np, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=8,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8_sorted_steel_matches_dequant_reference() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260702)
    experts = 4
    tokens = 11
    routes = 50
    output_dims = 17
    input_dims = 64
    group_size = 8
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 256, size=(experts, output_dims, input_dims // 8), dtype=np.uint8)
    scales_np = rng.uniform(0.01, 0.08, size=(experts, output_dims, input_dims // group_size)).astype(
        np.float16
    )
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8_1bit_packed()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8_fp16_sorted_steel_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=8,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_matches_dequant_reference() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260630)
    experts = 4
    tokens = 11
    routes = 50
    output_dims = 17
    input_dims = 64
    group_size = 8
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(0.01, 0.08, size=(experts, output_dims, input_dims // group_size)).astype(
        np.float16
    )
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_packed_rhs_tile_matmul_matches_layout_decode_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260702)
    m = 9
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    bn = 64
    bk = 64
    expert = 1
    n_tile = 1
    k_block = 5
    x_np = rng.normal(size=(m, bk)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    codebook_np = e8p_packed_abs_grid()
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=bn, bk=bk)

    observed = nax.nax_e8p_packed_rhs_tile_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(packed.code_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_group_indices[k_block], dtype=mx.int32),
        mx.array(packed.codeword_scale_slots[k_block], dtype=mx.int32),
        mx.array(codebook_np, dtype=mx.uint32),
        output_count=output_dims - n_tile * bn,
    )

    decoded_tile = decode_e8p_rhs_tile(
        packed,
        expert=expert,
        n_tile=n_tile,
        k_block=k_block,
        codebook=codebook_np,
    )
    expected = mx.array(x_np.astype(np.float32) @ decoded_tile.T.astype(np.float32), dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.scale_group_indices[k_block].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_split_byte_rhs_tile_matmul_matches_layout_decode_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260705)
    m = 9
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    bn = 64
    bk = 64
    expert = 1
    n_tile = 1
    k_block = 5
    x_np = rng.normal(size=(m, bk)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    codebook_np = e8p_packed_abs_grid()
    packed = pack_e8p_split_byte_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )

    observed = nax.nax_e8p_split_byte_rhs_tile_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(packed.sign_tiles[expert, n_tile, k_block]),
        mx.array(packed.abs_index_tiles[expert, n_tile, k_block]),
        mx.array(packed.parity_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_group_indices[k_block], dtype=mx.int32),
        mx.array(packed.codeword_scale_slots[k_block], dtype=mx.int32),
        mx.array(codebook_np, dtype=mx.uint32),
        output_count=output_dims - n_tile * bn,
    )

    decoded_tile = decode_e8p_split_byte_rhs_tile(
        packed,
        expert=expert,
        n_tile=n_tile,
        k_block=k_block,
        codebook=codebook_np,
    )
    expected = mx.array(x_np.astype(np.float32) @ decoded_tile.T.astype(np.float32), dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_tiles.dtype == np.uint8
    assert packed.abs_index_tiles.dtype == np.uint8
    assert packed.parity_tiles.dtype == np.uint8
    assert packed.scale_group_indices[k_block].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_split_byte_factor_reuse_rhs_tile_matmul_matches_layout_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260706)
    m = 9
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    bn = 64
    bk = 64
    expert = 1
    n_tile = 1
    k_block = 5
    x_np = rng.normal(size=(m, bk)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    codebook_np = e8p_packed_abs_grid()
    packed = pack_e8p_split_byte_factor_reuse_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )

    observed = nax.nax_e8p_split_byte_factor_reuse_rhs_tile_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(packed.sign_byte_lut[expert, n_tile, k_block]),
        mx.array(packed.sign_byte_slots[expert, n_tile, k_block]),
        mx.array(packed.abs_index_lut[expert, n_tile, k_block]),
        mx.array(packed.abs_index_slots[expert, n_tile, k_block]),
        mx.array(packed.scale_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_group_indices[k_block], dtype=mx.int32),
        mx.array(packed.codeword_scale_slots[k_block], dtype=mx.int32),
        mx.array(codebook_np, dtype=mx.uint32),
        output_count=output_dims - n_tile * bn,
    )

    decoded_tile = decode_e8p_split_byte_factor_reuse_rhs_tile(
        packed,
        expert=expert,
        n_tile=n_tile,
        k_block=k_block,
        codebook=codebook_np,
    )
    expected = mx.array(x_np.astype(np.float32) @ decoded_tile.T.astype(np.float32), dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_byte_lut.dtype == np.uint8
    assert packed.sign_byte_slots.dtype == np.uint8
    assert packed.abs_index_lut.dtype == np.uint8
    assert packed.abs_index_slots.dtype == np.uint8
    assert packed.scale_group_indices[k_block].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_sign_nibble_abs_index_rhs_tile_matmul_matches_layout_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260707)
    m = 9
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    bn = 64
    bk = 64
    expert = 1
    n_tile = 1
    k_block = 5
    x_np = rng.normal(size=(m, bk)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    codebook_np = e8p_packed_abs_grid()
    packed = pack_e8p_sign_nibble_abs_index_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )

    observed = nax.nax_e8p_sign_nibble_abs_index_rhs_tile_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(packed.sign_low_nibble_tiles[expert, n_tile, k_block]),
        mx.array(packed.sign_high_nibble_tiles[expert, n_tile, k_block]),
        mx.array(packed.abs_index_tiles[expert, n_tile, k_block]),
        mx.array(packed.parity_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_group_indices[k_block], dtype=mx.int32),
        mx.array(packed.codeword_scale_slots[k_block], dtype=mx.int32),
        mx.array(codebook_np, dtype=mx.uint32),
        output_count=output_dims - n_tile * bn,
    )

    decoded_tile = decode_e8p_sign_nibble_abs_index_rhs_tile(
        packed,
        expert=expert,
        n_tile=n_tile,
        k_block=k_block,
        codebook=codebook_np,
    )
    expected = mx.array(x_np.astype(np.float32) @ decoded_tile.T.astype(np.float32), dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_low_nibble_tiles.dtype == np.uint8
    assert packed.sign_high_nibble_tiles.dtype == np.uint8
    assert int(packed.sign_low_nibble_tiles.max()) <= 15
    assert int(packed.sign_high_nibble_tiles.max()) <= 15
    assert packed.abs_index_tiles.dtype == np.uint8
    assert packed.parity_tiles.dtype == np.uint8
    assert packed.scale_group_indices[k_block].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_sign_plane_abs_index_rhs_tile_matmul_matches_layout_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260709)
    m = 9
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    bn = 64
    bk = 64
    expert = 1
    n_tile = 1
    k_block = 5
    x_np = rng.normal(size=(m, bk)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    codebook_np = e8p_packed_abs_grid()
    packed = pack_e8p_sign_plane_abs_index_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )

    observed = nax.nax_e8p_sign_plane_abs_index_rhs_tile_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(packed.sign_bit_planes[expert, n_tile, k_block], dtype=mx.uint64),
        mx.array(packed.abs_index_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_group_indices[k_block], dtype=mx.int32),
        mx.array(packed.codeword_scale_slots[k_block], dtype=mx.int32),
        mx.array(codebook_np, dtype=mx.uint32),
        output_count=output_dims - n_tile * bn,
    )

    decoded_tile = decode_e8p_sign_plane_abs_index_rhs_tile(
        packed,
        expert=expert,
        n_tile=n_tile,
        k_block=k_block,
        codebook=codebook_np,
    )
    expected = mx.array(x_np.astype(np.float32) @ decoded_tile.T.astype(np.float32), dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_bit_planes.dtype == np.uint64
    assert packed.sign_bit_planes.shape[-2:] == (8, 8)
    assert packed.abs_index_tiles.dtype == np.uint8
    assert packed.scale_group_indices[k_block].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_sign_nibble_micro_lut_rhs_tile_matmul_matches_layout_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260708)
    m = 9
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    bn = 64
    bk = 64
    expert = 1
    n_tile = 1
    k_block = 5
    x_np = rng.normal(size=(m, bk)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    codebook_np = e8p_packed_abs_grid()
    packed = pack_e8p_sign_nibble_micro_lut_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )

    observed = nax.nax_e8p_sign_nibble_micro_lut_rhs_tile_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(packed.sign_low_nibble_lut),
        mx.array(packed.sign_low_nibble_slots[expert, n_tile, k_block]),
        mx.array(packed.sign_high_nibble_lut),
        mx.array(packed.sign_high_nibble_slots[expert, n_tile, k_block]),
        mx.array(packed.abs_index_lut[expert, n_tile, k_block]),
        mx.array(packed.abs_index_slots[expert, n_tile, k_block]),
        mx.array(packed.scale_tiles[expert, n_tile, k_block]),
        mx.array(packed.scale_group_indices[k_block], dtype=mx.int32),
        mx.array(packed.codeword_scale_slots[k_block], dtype=mx.int32),
        mx.array(codebook_np, dtype=mx.uint32),
        output_count=output_dims - n_tile * bn,
    )

    decoded_tile = decode_e8p_sign_nibble_micro_lut_rhs_tile(
        packed,
        expert=expert,
        n_tile=n_tile,
        k_block=k_block,
        codebook=codebook_np,
    )
    expected = mx.array(x_np.astype(np.float32) @ decoded_tile.T.astype(np.float32), dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_low_nibble_lut.tolist() == list(range(16))
    assert packed.sign_high_nibble_lut.tolist() == list(range(16))
    assert packed.sign_low_nibble_slots.dtype == np.uint8
    assert packed.sign_high_nibble_slots.dtype == np.uint8
    assert packed.abs_index_lut.dtype == np.uint8
    assert packed.abs_index_slots.dtype == np.uint8
    assert packed.scale_group_indices[k_block].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 2e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_sign_nibble_micro_lut_rhs_sorted_native_matmul_matches_layout_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260709)
    experts = 3
    tokens = 11
    routes = 29
    output_dims = 73
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_sign_nibble_micro_lut_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_low_nibble_lut),
        mx.array(packed.sign_low_nibble_slots),
        mx.array(packed.sign_high_nibble_lut),
        mx.array(packed.sign_high_nibble_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_sign_nibble_micro_lut_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))
    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_rhs_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[int(expert)].T.astype(
            np.float32
        )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_low_nibble_lut.tolist() == list(range(16))
    assert packed.sign_high_nibble_lut.tolist() == list(range(16))
    assert packed.sign_low_nibble_slots.dtype == np.uint8
    assert packed.sign_high_nibble_slots.dtype == np.uint8
    assert packed.abs_index_lut.dtype == np.uint8
    assert packed.abs_index_slots.dtype == np.uint8
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert packed.codeword_scale_slots[5].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_packed_rhs_expert_matmul_composes_packed_tiles() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260703)
    m = 7
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    expert = 1
    x_np = rng.normal(size=(m, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    codebook_np = e8p_packed_abs_grid()
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)

    observed = nax.nax_e8p_packed_rhs_expert_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(packed.code_tiles[expert]),
        mx.array(packed.scale_tiles[expert]),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        mx.array(codebook_np, dtype=mx.uint32),
        output_dims=output_dims,
    )

    decoded_tiles: list[np.ndarray] = []
    for n_tile in range(packed.layout.n_tiles):
        k_tiles = [
            decode_e8p_rhs_tile(
                packed,
                expert=expert,
                n_tile=n_tile,
                k_block=k_block,
                codebook=codebook_np,
            )
            for k_block in range(packed.layout.k_blocks)
        ]
        decoded_tiles.append(np.concatenate(k_tiles, axis=1))
    decoded_weight = np.concatenate(decoded_tiles, axis=0)
    expected = mx.array(x_np.astype(np.float32) @ decoded_weight.T.astype(np.float32), dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_packed_rhs_sorted_matmul_matches_expert_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260704)
    experts = 3
    tokens = 9
    routes = 23
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_packed_rhs_sorted_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))
    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_rhs_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[int(expert)].T.astype(
            np.float32
        )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_packed_rhs_sorted_native_matmul_matches_bridge() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260705)
    experts = 3
    tokens = 11
    routes = 29
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_packed_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_packed_rhs_sorted_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_route_slot_codeword_stream_rhs_sorted_matmul_matches_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260716)
    experts = 3
    tokens = 13
    routes = 41
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=8,
    )

    observed = nax.nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))
    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_rhs_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[int(expert)].T.astype(
            np.float32
        )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert int(np.max(np.array(tile_counts))) <= 8
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul_matches_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260717)
    experts = 3
    tokens = 13
    routes = 41
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=8,
    )

    observed = nax.nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))
    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_rhs_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[int(expert)].T.astype(
            np.float32
        )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert int(np.max(np.array(tile_counts))) <= 8
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_exposes_active_route_tile_codeword_outer_product_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_expert_cohort_codeword_broadcast_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_route_batch_segmented_codeword_reduce_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_token_cohort_codeword_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_token_cohort_mma_codeword_tile_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_output_stationary_codeword_tile_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_input_stationary_codeword_tile_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_expert_kblock_codeword_factor_reuse_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_route_codeword_lut_accumulate_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_rowwise_codeword_tile_accumulate_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_output_tile_local_codeword_lut_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_route_microtile_codeword_block_reduce_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_kblock_wavefront_codeword_scan_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_token_route_output_stripe_pipeline_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_expert_kblock_scale_slot_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_scale_group_route_block_reduce_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_route_block_output_group_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_route_block_output_group_stream_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_output_group_pretransposed_codeword_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_kblock_output_group_route_fused_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_route_tile_output_swizzle_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul",
        )
    )


def test_nax_native_exposes_token_topk_output_tile_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_block_output_group_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_block_output_group_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_block_output_group_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_output_stripe_group_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_expert_output_block_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_pair_kblock_accumulator_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_pair_output_group_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_pair_slot_topk_output_group_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_pair_slot_topk_codeword_group_pipeline_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_pair_slot_topk_scale_slot_broadcast_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_pair_slot_topk_route_bucket_codeword_reduce_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_pair_slot_topk_kblock_microtile_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_exposes_token_pair_slot_topk_output_tile_fused_stream_wrapper() -> None:
    assert callable(
        getattr(
            nax,
            "nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul",
        )
    )

    if nax.is_available():
        assert hasattr(
            nax.load_native(),
            "nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul_into",
        )


def test_nax_native_e8p_split_byte_rhs_sorted_native_matmul_matches_split_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260706)
    experts = 3
    tokens = 11
    routes = 29
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_split_byte_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_split_byte_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_tiles),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.parity_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_split_byte_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))
    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_rhs_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[int(expert)].T.astype(
            np.float32
        )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_tiles.dtype == np.uint8
    assert packed.abs_index_tiles.dtype == np.uint8
    assert packed.parity_tiles.dtype == np.uint8
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_sign_nibble_rhs_sorted_native_matmul_matches_layout_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260708)
    experts = 3
    tokens = 11
    routes = 29
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_sign_nibble_abs_index_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_sign_nibble_abs_index_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_low_nibble_tiles),
        mx.array(packed.sign_high_nibble_tiles),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.parity_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_sign_nibble_abs_index_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))
    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_rhs_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[int(expert)].T.astype(
            np.float32
        )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_low_nibble_tiles.dtype == np.uint8
    assert packed.sign_high_nibble_tiles.dtype == np.uint8
    assert np.max(packed.sign_low_nibble_tiles) <= 15
    assert np.max(packed.sign_high_nibble_tiles) <= 15
    assert packed.abs_index_tiles.dtype == np.uint8
    assert packed.parity_tiles.dtype == np.uint8
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert packed.codeword_scale_slots[5].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_sign_plane_rhs_sorted_native_matmul_matches_layout_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260711)
    experts = 3
    tokens = 11
    routes = 29
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_sign_plane_abs_index_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_sign_plane_abs_index_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_bit_planes, dtype=mx.uint64),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_sign_plane_abs_index_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))
    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_rhs_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[int(expert)].T.astype(
            np.float32
        )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_bit_planes.dtype == np.uint64
    assert packed.sign_bit_planes.shape[-2:] == (8, 8)
    assert packed.abs_index_tiles.dtype == np.uint8
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert packed.codeword_scale_slots[5].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_sign_nibble_rhs_sorted_tensorops_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260710)
    experts = 3
    tokens = 11
    routes = 35
    output_dims = 73
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_sign_nibble_abs_index_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul(
        sorted_x,
        mx.array(packed.sign_low_nibble_tiles),
        mx.array(packed.sign_high_nibble_tiles),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.parity_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_sign_nibble_abs_index_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_low_nibble_tiles),
        mx.array(packed.sign_high_nibble_tiles),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.parity_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert packed.codeword_scale_slots[5].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_sign_plane_rhs_sorted_tensorops_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native NAX extension is not built")
    rng = np.random.default_rng(623)
    experts = 3
    tokens = 11
    routes = 35
    output_dims = 73
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_sign_plane_abs_index_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul(
        sorted_x,
        mx.array(packed.sign_bit_planes, dtype=mx.uint64),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_sign_plane_abs_index_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_bit_planes, dtype=mx.uint64),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert packed.codeword_scale_slots[5].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260711)
    experts = 3
    tokens = 11
    routes = 35
    output_dims = 73
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_sign_nibble_micro_lut_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul(
        sorted_x,
        mx.array(packed.sign_low_nibble_lut),
        mx.array(packed.sign_low_nibble_slots),
        mx.array(packed.sign_high_nibble_lut),
        mx.array(packed.sign_high_nibble_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_low_nibble_lut),
        mx.array(packed.sign_low_nibble_slots),
        mx.array(packed.sign_high_nibble_lut),
        mx.array(packed.sign_high_nibble_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert packed.codeword_scale_slots[5].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_split_byte_factor_reuse_rhs_sorted_native_matmul_matches_layout_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260707)
    experts = 3
    tokens = 11
    routes = 29
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_split_byte_factor_reuse_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_split_byte_factor_reuse_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))
    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_rhs_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[int(expert)].T.astype(
            np.float32
        )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_byte_lut.dtype == np.uint8
    assert packed.sign_byte_slots.dtype == np.uint8
    assert packed.abs_index_lut.dtype == np.uint8
    assert packed.abs_index_slots.dtype == np.uint8
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_expert_kblock_factor_reuse_rhs_sorted_native_matmul_matches_layout_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260711)
    experts = 3
    tokens = 11
    routes = 29
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_expert_kblock_factor_reuse_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_expert_kblock_factor_reuse_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))
    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_rhs_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[int(expert)].T.astype(
            np.float32
        )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_byte_lut.shape == (experts, packed.layout.k_blocks, 256)
    assert packed.abs_index_lut.shape == packed.sign_byte_lut.shape
    assert packed.sign_byte_slots.shape == (
        experts,
        packed.layout.n_tiles,
        packed.layout.k_blocks,
        packed.layout.bn,
        packed.layout.codewords_per_bk,
    )
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_component_stream_rhs_sorted_scalar_matmul_matches_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260712)
    experts = 3
    tokens = 11
    routes = 29
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_component_stream_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_component_stream_rhs_sorted_scalar_matmul(
        sorted_x,
        mx.array(packed.sign_component_bits),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        mx.array(packed.component_scale_slots, dtype=mx.int32),
        mx.array(packed.component_codeword_indices, dtype=mx.int32),
        mx.array(packed.component_offsets, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expected_np = component_stream_sorted_matmul_oracle(
        packed,
        sorted_x_np,
        sorted_rhs_np,
        codebook=codebook_np,
    )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_component_bits.dtype == np.uint8
    assert packed.component_scale_slots.shape[-1] == packed.layout.codewords_per_bk * 8
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_component_stream_rhs_sorted_partial_matmul_matches_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260703)
    experts = 3
    tokens = 11
    routes = 29
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_component_stream_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_component_stream_rhs_sorted_partial_matmul(
        sorted_x,
        mx.array(packed.sign_component_bits),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        mx.array(packed.component_scale_slots, dtype=mx.int32),
        mx.array(packed.component_codeword_indices, dtype=mx.int32),
        mx.array(packed.component_offsets, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expected_np = component_stream_sorted_matmul_oracle(
        packed,
        sorted_x_np,
        sorted_rhs_np,
        codebook=codebook_np,
    )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3


def test_nax_native_e8p_component_stream_rhs_sorted_tensorops_matmul_matches_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260704)
    experts = 3
    tokens = 9
    routes = 23
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_component_stream_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_component_stream_rhs_sorted_tensorops_matmul(
        sorted_x,
        mx.array(packed.sign_component_bits),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        mx.array(packed.component_scale_slots, dtype=mx.int32),
        mx.array(packed.component_codeword_indices, dtype=mx.int32),
        mx.array(packed.component_offsets, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expected_np = component_stream_sorted_matmul_oracle(
        packed,
        sorted_x_np,
        sorted_rhs_np,
        codebook=codebook_np,
    )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999
    assert dot / norm >= 0.99999


def test_nax_native_e8p_component_stream_rhs_sorted_shared_decode_matmul_matches_oracle() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260711)
    experts = 3
    tokens = 9
    routes = 23
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    sorted_x = mx.array(sorted_x_np, dtype=mx.float16)
    codebook_np = e8p_packed_abs_grid()
    codebook = mx.array(codebook_np, dtype=mx.uint32)
    packed = pack_e8p_component_stream_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_component_stream_rhs_sorted_shared_decode_matmul(
        sorted_x,
        mx.array(packed.sign_component_bits),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        mx.array(packed.component_scale_slots, dtype=mx.int32),
        mx.array(packed.component_codeword_indices, dtype=mx.int32),
        mx.array(packed.component_offsets, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )

    expected_np = component_stream_sorted_matmul_oracle(
        packed,
        sorted_x_np,
        sorted_rhs_np,
        codebook=codebook_np,
    )
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260712)
    experts = 3
    tokens = 11
    routes = 35
    output_dims = 73
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_expert_kblock_factor_reuse_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert packed.sign_byte_lut.shape == (experts, packed.layout.k_blocks, 256)
    assert packed.abs_index_lut.shape == (experts, packed.layout.k_blocks, 256)
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert packed.codeword_scale_slots[5].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260713)
    experts = 3
    tokens = 11
    routes = 35
    output_dims = 73
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_expert_kblock_factor_reuse_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert packed.codeword_scale_slots[5].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_packed_rhs_sorted_tiled_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260706)
    experts = 3
    tokens = 13
    routes = 37
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_packed_rhs_sorted_tiled_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_packed_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_split_byte_rhs_sorted_tiled_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260707)
    experts = 3
    tokens = 13
    routes = 37
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_split_byte_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_split_byte_rhs_sorted_tiled_matmul(
        sorted_x,
        mx.array(packed.sign_tiles),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.parity_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_split_byte_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_tiles),
        mx.array(packed.abs_index_tiles),
        mx.array(packed.parity_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 6e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260708)
    experts = 3
    tokens = 13
    routes = 37
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_split_byte_factor_reuse_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.sign_byte_lut.dtype == np.uint8
    assert packed.sign_byte_slots.dtype == np.uint8
    assert packed.abs_index_lut.dtype == np.uint8
    assert packed.abs_index_slots.dtype == np.uint8
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260709)
    experts = 3
    tokens = 11
    routes = 35
    output_dims = 73
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_split_byte_factor_reuse_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260710)
    experts = 3
    tokens = 11
    routes = 35
    output_dims = 73
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_split_byte_factor_reuse_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.sign_byte_lut),
        mx.array(packed.sign_byte_slots),
        mx.array(packed.abs_index_lut),
        mx.array(packed.abs_index_slots),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert packed.scale_group_indices[5].tolist() == [0, 1]
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_packed_rhs_sorted_tiled_m128_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260711)
    experts = 4
    tokens = 29
    routes = 149
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=128,
    )

    observed = nax.nax_e8p_packed_rhs_sorted_tiled_m128_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_packed_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_e8p_packed_rhs_sorted_tiled_k128_matmul_matches_native_scalar() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260712)
    experts = 4
    tokens = 31
    routes = 83
    output_dims = 70
    input_dims = 704
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x = mx.array(x_np[sorted_lhs_np], dtype=mx.float16)
    codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=64, bk=64)
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )

    observed = nax.nax_e8p_packed_rhs_sorted_tiled_k128_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    expected = nax.nax_e8p_packed_rhs_sorted_native_matmul(
        sorted_x,
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        codebook,
        output_dims=output_dims,
    )
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_direct_reduce_matches_dequant_reference() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260714)
    experts = 3
    tokens = 13
    routes = 29
    output_dims = 11
    input_dims = 64
    group_size = 8
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(0.01, 0.08, size=(experts, output_dims, input_dims // group_size)).astype(
        np.float16
    )
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_direct_reduce_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_inline_b_matches_dequant_reference() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260715)
    experts = 3
    tokens = 13
    routes = 71
    output_dims = 37
    input_dims = 64
    group_size = 8
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(0.01, 0.08, size=(experts, output_dims, input_dims // group_size)).astype(
        np.float16
    )
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_inline_b_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260704)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_bk128_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260705)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_bk128_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_m128n32_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260706)
    experts = 3
    tokens = 23
    routes = 197
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=128,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_m128n32_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_m64n128_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260712)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 129
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_m64n128_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_m32n64_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260716)
    experts = 3
    tokens = 19
    routes = 83
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=32,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_m32n64_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_m64n64t64_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260719)
    experts = 3
    tokens = 19
    routes = 83
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_m64n64t64_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_m32n64t128_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260718)
    experts = 3
    tokens = 19
    routes = 83
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=32,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_m32n64t128_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_m32n128_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260717)
    experts = 3
    tokens = 19
    routes = 83
    output_dims = 129
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=32,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_m32n128_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_gs352_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260713)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_gs352_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_lut_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260707)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_lut_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_tgcb_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260708)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_tgcb_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_tgcb_hoist_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260709)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_tgcb_hoist_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_tgscale_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260710)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_tgscale_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8p_sorted_steel_tgcb_tgscale_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260711)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 65536, size=(experts, output_dims, input_dims // 8), dtype=np.uint16)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8p_packed_abs_grid()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8p_fp16_sorted_steel_tgcb_tgscale_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=16,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8_tensorops_routed_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260628)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 256, size=(experts, output_dims, input_dims // 8), dtype=np.uint8)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    codebook_np = e8_1bit_packed()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8_fp16_routed_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        mx.array(sorted_lhs_np, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=8,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8_steel_routed_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260701)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 256, size=(experts, output_dims, input_dims // 8), dtype=np.uint8)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    codebook_np = e8_1bit_packed()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8_fp16_routed_steel_matmul(
        mx.array(x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        mx.array(sorted_lhs_np, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=8,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_fused_e8_sorted_steel_handles_air_down_group352() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260703)
    experts = 3
    tokens = 17
    routes = 73
    output_dims = 65
    input_dims = 1408
    group_size = 352
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(0, 256, size=(experts, output_dims, input_dims // 8), dtype=np.uint8)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    sorted_x_np = x_np[sorted_lhs_np]
    codebook_np = e8_1bit_packed()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8_fp16_sorted_steel_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=8,
            codebook=codebook_np,
        )
        expected_np[route] = x_np[token].astype(np.float32) @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999


def test_nax_native_int8_routed_matches_quantized_activation_reference() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260629)
    experts = 3
    tokens = 13
    routes = 71
    output_dims = 33
    input_dims = 128
    group_size = 32
    groups = input_dims // group_size
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float32)
    x_grouped = x_np.reshape(tokens, groups, group_size)
    x_scales_np = np.maximum(np.max(np.abs(x_grouped), axis=-1) / 127.0, 1e-6).astype(
        np.float16
    )
    x_q_np = np.clip(
        np.rint(x_grouped / x_scales_np.astype(np.float32)[..., None]),
        -127,
        127,
    ).astype(np.int8)
    x_q_np = x_q_np.reshape(tokens, input_dims)
    x_dequant_np = (
        x_q_np.reshape(tokens, groups, group_size).astype(np.float32)
        * x_scales_np.astype(np.float32)[..., None]
    ).reshape(tokens, input_dims)
    codes_np = rng.integers(0, 256, size=(experts, output_dims, input_dims // 8), dtype=np.uint8)
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, groups),
    ).astype(np.float16)
    sorted_rhs_np = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_lhs_np = rng.integers(0, tokens, size=(routes,), dtype=np.int32)
    codebook_np = e8_1bit_packed()

    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_rhs_np),
        num_experts=experts,
        route_tile=64,
    )
    observed = nax.nax_e8_int8_routed_matmul(
        mx.array(x_q_np, dtype=mx.int8),
        mx.array(x_scales_np, dtype=mx.float16),
        mx.array(codes_np),
        mx.array(scales_np),
        mx.array(sorted_lhs_np, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(codebook_np, dtype=mx.uint32),
        group_size=group_size,
    )

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, (expert, token) in enumerate(zip(sorted_rhs_np, sorted_lhs_np, strict=True)):
        weight = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=8,
            codebook=codebook_np,
        )
        expected_np[route] = x_dequant_np[token] @ weight.T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)

    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert observed.shape == expected.shape
    assert float(np.max(np.abs(observed_np - expected_np))) < 8e-3
    assert dot / norm >= 0.99999
