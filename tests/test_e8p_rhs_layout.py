from __future__ import annotations

import numpy as np

from mlx_vq.codebook.e8 import decode_e8p, e8p_packed_abs_grid
from mlx_vq.kernels.e8p_rhs_layout import (
    E8PNextKernelFamilyCandidate,
    build_e8p_component_stream_contract,
    build_e8p_component_stream_partial_reduction_contract,
    build_e8p_component_stream_shared_decode_contract,
    evaluate_e8p_component_stream_shared_decode_feasibility,
    build_e8p_expert_kblock_dispatch_contract,
    evaluate_e8p_expert_kblock_decode_reuse_feasibility,
    evaluate_e8p_route_abs_component_projection_feasibility,
    evaluate_e8p_route_abs_component_lower_bound_feasibility,
    evaluate_e8p_route_abs_component_token_active_expert_feasibility,
    evaluate_e8p_route_abs_component_token_reuse_feasibility,
    build_e8p_expert_kblock_partial_reduction_contract,
    evaluate_e8p_route_abs_index_projection_feasibility,
    evaluate_e8p_route_active_codeword_projection_feasibility,
    evaluate_e8p_route_codeword_projection_feasibility,
    component_stream_sorted_matmul_oracle,
    decode_e8p_component_stream_rhs_tile,
    decode_e8p_rhs_tile,
    decode_e8p_sign_plane_abs_index_rhs_tile,
    decode_e8p_sign_nibble_abs_index_rhs_tile,
    decode_e8p_sign_nibble_micro_lut_rhs_tile,
    decode_e8p_split_byte_factor_reuse_rhs_tile,
    decode_e8p_split_byte_rhs_tile,
    evaluate_e8p_next_kernel_family_candidate,
    pack_e8p_component_stream_rhs_tiles,
    pack_e8p_sign_plane_abs_index_rhs_tiles,
    pack_e8p_sign_nibble_abs_index_rhs_tiles,
    pack_e8p_sign_nibble_micro_lut_rhs_tiles,
    pack_e8p_expert_kblock_factor_reuse_rhs_tiles,
    pack_e8p_split_byte_factor_reuse_rhs_tiles,
    pack_e8p_rhs_tiles,
    pack_e8p_split_byte_rhs_tiles,
)


def test_e8p_rhs_layout_is_exposed_through_ramp_kernels() -> None:
    from ramp.kernels.e8p_rhs_layout import pack_e8p_rhs_tiles as ramp_pack_e8p_rhs_tiles

    codes = np.zeros((1, 1, 8), dtype=np.uint16)
    scales = np.ones((1, 1, 1), dtype=np.float16)

    packed = ramp_pack_e8p_rhs_tiles(codes, scales, group_size=64)

    assert packed.layout.target_kernel_family == "sorted_gather_qmm_rhs_nax"


def test_pack_e8p_rhs_tiles_preserves_compressed_codes_and_crossing_scale_groups() -> None:
    rng = np.random.default_rng(20260702)
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)

    packed = pack_e8p_rhs_tiles(codes, scales, group_size=group_size, bn=64, bk=64)

    assert packed.code_tiles.shape == (2, 2, 11, 64, 8)
    assert packed.code_tiles.dtype == np.uint16
    assert packed.scale_tiles.shape == (2, 2, 11, 64, 2)
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert packed.layout.target_kernel_family == "sorted_gather_qmm_rhs_nax"
    assert packed.layout.storage_constraint == "compressed_e8p_codes_scales"

    # K block 5 covers dimensions 320..383 and crosses the down-proj scale
    # boundary at 352, so the first four codewords use scale group 0 and the
    # last four use scale group 1.
    np.testing.assert_array_equal(packed.scale_group_indices[5], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        packed.codeword_scale_slots[5],
        np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(packed.code_tiles[1, 1, 5, 3], codes[1, 67, 40:48])
    np.testing.assert_array_equal(packed.scale_tiles[1, 1, 5, 3], scales[1, 67, [0, 1]])

    decoded = decode_e8p_rhs_tile(
        packed,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )
    code_vectors = decode_e8p(codes[1, 64:70, 40:48], e8p_packed_abs_grid())
    scale_slots = packed.codeword_scale_slots[5]
    expected_scales = np.take(packed.scale_tiles[1, 1, 5, :6, :], scale_slots, axis=1)
    expected = (
        code_vectors
        * expected_scales[:, :, None].astype(np.float32)
    ).reshape(6, 64)

    assert decoded.shape == (6, 64)
    np.testing.assert_allclose(decoded, expected, rtol=0, atol=0)


def test_selector_rejects_output_tile_local_codeword_lut_after_speed_rejection() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_output_tile_local_codeword_lut",
        storage_constraint="compressed_e8p_output_tile_local_codeword_lut_scales",
        dispatch_grid=(
            "route_microtiles_x_output_tiles_x_k_blocks_x_unique_codewords_then_output_rows"
        ),
        decode_reuse_scope="output_tile_local_unique_codeword_lut_bounded_by_tile",
        tensorops_rhs_source=(
            "compressed_codeword_tiles_with_output_tile_local_activation_dot_lut"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_output_tile_local_codeword_lut_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert "output_tile_local_codeword_lut" in verdict["matched_rejected_families"]
    assert "candidate reopens the speed-rejected output-tile-local codeword LUT schedule" in (
        verdict["structural_reasons"]
    )
    assert "materially_change_output_tile_local_codeword_lut_layout_or_kernel_family" in (
        verdict["required_next_features"]
    )
    assert "do_not_retime_output_tile_local_codeword_lut" in (
        verdict["rejected_next_steps"]
    )


def test_selector_rejects_expert_kblock_scale_slot_stream_after_speed_rejection() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_expert_kblock_scale_slot_stream",
        storage_constraint="compressed_e8p_expert_kblock_scale_slot_streams",
        dispatch_grid="experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles",
        decode_reuse_scope=(
            "expert_kblock_scale_slot_streams_reuse_codeword_scale_slots_across_output_tiles"
        ),
        tensorops_rhs_source=(
            "compressed_e8p_codeword_tiles_with_expert_kblock_scale_slot_streams"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(
        candidate,
        speed_rejected_kernel_families=("expert_kblock_scale_slot_stream",),
    )

    assert verdict["decision"] == "reject_expert_kblock_scale_slot_stream_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert "expert_kblock_scale_slot_stream" in verdict["matched_rejected_families"]
    assert (
        "candidate reopens the speed-rejected expert/K-block scale-slot stream schedule"
        in verdict["structural_reasons"]
    )
    assert (
        "materially_change_expert_kblock_scale_slot_stream_layout_or_kernel_family"
        in verdict["required_next_features"]
    )
    assert "do_not_retime_expert_kblock_scale_slot_stream" in (
        verdict["rejected_next_steps"]
    )


def test_pack_e8p_split_byte_rhs_tiles_preserves_factors_and_scale_maps() -> None:
    rng = np.random.default_rng(20260703)
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)

    packed = pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=group_size, bn=64, bk=64)

    assert packed.layout.target_kernel_family == "sorted_gather_qmm_rhs_nax_split_byte"
    assert packed.layout.storage_constraint == "compressed_e8p_split_bytes_scales"
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert packed.sign_tiles.shape == (2, 2, 11, 64, 8)
    assert packed.sign_tiles.dtype == np.uint8
    assert packed.abs_index_tiles.shape == packed.sign_tiles.shape
    assert packed.abs_index_tiles.dtype == np.uint8
    assert packed.parity_tiles.shape == packed.sign_tiles.shape
    assert packed.parity_tiles.dtype == np.uint8
    np.testing.assert_array_equal(packed.scale_group_indices[5], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        packed.codeword_scale_slots[5],
        np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
    )

    expected_codes = codes[1, 67, 40:48]
    np.testing.assert_array_equal(packed.sign_tiles[1, 1, 5, 3], (expected_codes & 0xFF).astype(np.uint8))
    np.testing.assert_array_equal(packed.abs_index_tiles[1, 1, 5, 3], (expected_codes >> 8).astype(np.uint8))
    expected_parity = np.bitwise_xor.reduce(
        ((expected_codes[:, None] & 0xFF) >> np.arange(8, dtype=np.uint16)) & 1,
        axis=1,
    ).astype(np.uint8)
    np.testing.assert_array_equal(packed.parity_tiles[1, 1, 5, 3], expected_parity)


def test_decode_e8p_split_byte_rhs_tile_matches_uint16_packed_oracle() -> None:
    rng = np.random.default_rng(20260704)
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    packed_u16 = pack_e8p_rhs_tiles(codes, scales, group_size=group_size, bn=64, bk=64)
    packed_split = pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=group_size, bn=64, bk=64)

    observed = decode_e8p_split_byte_rhs_tile(
        packed_split,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )
    expected = decode_e8p_rhs_tile(
        packed_u16,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )

    assert observed.shape == (6, 64)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)


def test_pack_e8p_split_byte_factor_reuse_tiles_reconstructs_split_byte_oracle() -> None:
    rows = np.arange(64, dtype=np.uint16)[:, None]
    words = np.arange(8, dtype=np.uint16)[None, :]
    signs = ((rows % 8) * 17 + words) & np.uint16(0xFF)
    abs_indices = (words % 4) + np.uint16(11)
    codes = (signs | (abs_indices << np.uint16(8))).reshape(1, 64, 8)
    scales = np.ones((1, 64, 1), dtype=np.float16)

    packed = pack_e8p_split_byte_factor_reuse_rhs_tiles(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
    )

    assert packed.layout.target_kernel_family == "sorted_gather_qmm_rhs_nax_split_byte_factor_reuse"
    assert packed.layout.storage_constraint == "compressed_e8p_split_byte_factor_lut_scales"
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert packed.abs_index_lut.shape == (1, 1, 1, 256)
    assert packed.abs_index_slots.shape == (1, 1, 1, 64, 8)
    assert packed.abs_index_slots.dtype == np.uint8
    assert packed.sign_byte_lut.shape == (1, 1, 1, 256)
    assert packed.sign_byte_slots.shape == packed.abs_index_slots.shape
    assert int(packed.abs_index_counts[0, 0, 0]) == 4
    assert int(packed.sign_byte_counts[0, 0, 0]) == 64
    np.testing.assert_array_equal(
        packed.abs_index_lut[0, 0, 0, :4],
        np.array([11, 12, 13, 14], dtype=np.uint8),
    )

    reconstructed_abs = packed.abs_index_lut[0, 0, 0, packed.abs_index_slots[0, 0, 0]]
    reconstructed_sign = packed.sign_byte_lut[0, 0, 0, packed.sign_byte_slots[0, 0, 0]]
    np.testing.assert_array_equal(reconstructed_abs, abs_indices.astype(np.uint8).repeat(64, axis=0))
    np.testing.assert_array_equal(reconstructed_sign, signs.astype(np.uint8))

    expected = decode_e8p_split_byte_rhs_tile(
        pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=64, bn=64, bk=64),
        expert=0,
        n_tile=0,
        k_block=0,
        codebook=e8p_packed_abs_grid(),
    )
    observed = decode_e8p_split_byte_factor_reuse_rhs_tile(
        packed,
        expert=0,
        n_tile=0,
        k_block=0,
        codebook=e8p_packed_abs_grid(),
    )

    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)


def test_pack_e8p_expert_kblock_factor_reuse_tiles_share_luts_across_output_tiles() -> None:
    from mlx_vq.kernels import e8p_rhs_layout as layout

    rows = np.arange(128, dtype=np.uint16)[:, None]
    words = np.arange(8, dtype=np.uint16)[None, :]
    signs = ((rows % 64) + words * 3) & np.uint16(0xFF)
    abs_indices = (words % 4) + np.uint16(19)
    codes = (signs | (abs_indices << np.uint16(8))).reshape(1, 128, 8)
    scales = np.linspace(0.5, 1.5, num=128, dtype=np.float32).reshape(1, 128, 1).astype(
        np.float16
    )

    packed = layout.pack_e8p_expert_kblock_factor_reuse_rhs_tiles(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
    )

    assert packed.layout.target_kernel_family == (
        "sorted_gather_qmm_rhs_nax_expert_kblock_factor_reuse"
    )
    assert packed.layout.storage_constraint == "compressed_e8p_expert_kblock_factor_lut_scales"
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert packed.sign_byte_lut.shape == (1, 1, 256)
    assert packed.abs_index_lut.shape == (1, 1, 256)
    assert packed.sign_byte_slots.shape == (1, 2, 1, 64, 8)
    assert packed.abs_index_slots.shape == packed.sign_byte_slots.shape
    assert int(packed.sign_byte_counts[0, 0]) == np.unique(signs).size
    assert int(packed.abs_index_counts[0, 0]) == 4

    reconstructed_sign = packed.sign_byte_lut[0, 0, packed.sign_byte_slots[0, :, 0]]
    reconstructed_abs = packed.abs_index_lut[0, 0, packed.abs_index_slots[0, :, 0]]
    np.testing.assert_array_equal(reconstructed_sign.reshape(128, 8), signs.astype(np.uint8))
    np.testing.assert_array_equal(
        reconstructed_abs.reshape(128, 8),
        abs_indices.astype(np.uint8).repeat(128, axis=0),
    )

    split = layout.pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=64, bn=64, bk=64)
    for n_tile in range(2):
        observed = layout.decode_e8p_expert_kblock_factor_reuse_rhs_tile(
            packed,
            expert=0,
            n_tile=n_tile,
            k_block=0,
            codebook=e8p_packed_abs_grid(),
        )
        expected = decode_e8p_split_byte_rhs_tile(
            split,
            expert=0,
            n_tile=n_tile,
            k_block=0,
            codebook=e8p_packed_abs_grid(),
        )
        np.testing.assert_allclose(observed, expected, rtol=0, atol=0)


def test_route_abs_component_token_reuse_reduces_exact_rowwise_cache_by_topk() -> None:
    rng = np.random.default_rng(20260704)
    codes = rng.integers(0, 32, size=(2, 16, 8), dtype=np.uint16)
    scales = np.ones((2, 16, 1), dtype=np.float16)

    feasibility = evaluate_e8p_route_abs_component_token_reuse_feasibility(
        codes,
        scales,
        token_count=4,
        top_k=2,
        group_size=64,
        bn=16,
        bk=64,
        projection_cache_byte_cap=4096,
    )

    assert feasibility.decision == "route_abs_component_token_reuse_ready_for_source_probe"
    assert feasibility.token_reuse_exact_for_topk_routes is True
    assert feasibility.route_slot_axis_preserved_by_indirection is True
    assert feasibility.route_slot_reuse_factor == 2
    assert feasibility.token_projection_cache_shape == (4, 1, 1, 8)
    assert feasibility.rowwise_projection_cache_shape == (8, 1, 1, 8)
    assert feasibility.cache_reduction_vs_rowwise == 2.0
    assert feasibility.token_projection_cache_bytes * 2 == feasibility.rowwise_projection_cache_bytes
    assert feasibility.candidate_ready_for_source_probe is True
    assert feasibility.preserves_compressed_rhs_storage is True
    assert feasibility.preserves_codeword_scale_slots is True


def test_next_kernel_selector_rejects_route_codeword_lut_accumulate_after_speed_no_go() -> None:
    verdict = evaluate_e8p_next_kernel_family_candidate(
        E8PNextKernelFamilyCandidate(
            target_kernel_family="sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate",
            storage_constraint="compressed_e8p_route_codeword_lut_accumulate_scales",
            dispatch_grid="route_slots_x_k_blocks_x_codewords_then_output_tiles",
            decode_reuse_scope="activation_side_route_codeword_lut_preserves_route_slots",
            tensorops_rhs_source=(
                "compressed_codewords_scales_with_route_local_codeword_dot_lut"
            ),
            preserves_compressed_rhs_storage=True,
            preserves_codeword_scale_slots=True,
            decoded_dense_weight_bytes=0,
        )
    )

    assert verdict["decision"] == "reject_route_codeword_lut_accumulate_speed_path"
    assert verdict["matched_rejected_families"] == ["route_codeword_lut_accumulate"]
    assert verdict["candidate_ready_for_source_probe"] is False
    assert "avoid_route_local_full_codeword_lut" in verdict["required_next_features"]
    assert "do_not_retime_route_codeword_lut_accumulate" in verdict["rejected_next_steps"]


def test_route_abs_component_token_reuse_rejects_air_shaped_cache_over_cap() -> None:
    rng = np.random.default_rng(20260705)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(1, 256, 176),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(1, 256, 4)).astype(np.float16)

    feasibility = evaluate_e8p_route_abs_component_token_reuse_feasibility(
        codes,
        scales,
        token_count=1024,
        top_k=8,
        group_size=352,
        bn=64,
        bk=64,
        projection_cache_byte_cap=8 * 1024 * 1024,
    )

    assert feasibility.decision == "reject_route_abs_component_token_reuse_cache_too_large"
    assert feasibility.token_projection_cache_shape == (1024, 22, 256, 8)
    assert feasibility.token_projection_cache_bytes == 92_274_688
    assert feasibility.rowwise_projection_cache_bytes == 738_197_504
    assert feasibility.cache_reduction_vs_rowwise == 8.0
    assert feasibility.candidate_ready_for_source_probe is False
    assert "do_not_benchmark_token_reuse_abs_component_cache" in feasibility.rejected_next_steps


def test_route_abs_component_token_active_expert_reuse_reduces_to_active_abs_union() -> None:
    codes = np.zeros((3, 16, 8), dtype=np.uint16)
    for expert, base in enumerate((3, 17, 29)):
        codes[expert, :, :] = np.uint16(base << 8)
    scales = np.ones((3, 16, 1), dtype=np.float16)
    route_experts = np.array([[0, 1], [1, 2], [2, 2], [0, 2]], dtype=np.int32)

    feasibility = evaluate_e8p_route_abs_component_token_active_expert_feasibility(
        codes,
        scales,
        route_experts,
        token_count=4,
        top_k=2,
        group_size=64,
        bn=16,
        bk=64,
        projection_cache_byte_cap=4096,
    )

    assert feasibility.decision == (
        "route_abs_component_token_active_expert_reuse_ready_for_source_probe"
    )
    assert feasibility.token_active_expert_reuse_exact_for_topk_routes is True
    assert feasibility.route_slot_axis_preserved_by_indirection is True
    assert feasibility.active_expert_abs_union_scoped_to_token_topk is True
    assert feasibility.token_kblock_abs_index_count_shape == (4, 1)
    assert feasibility.min_abs_indices_per_token_kblock == 1
    assert feasibility.max_abs_indices_per_token_kblock == 2
    assert feasibility.ragged_token_active_projection_cache_elements == 7 * 8
    assert feasibility.ragged_token_active_projection_cache_bytes == 112
    assert feasibility.full_token_projection_cache_bytes == 16_384
    assert feasibility.rowwise_projection_cache_bytes == 32_768
    assert feasibility.candidate_ready_for_source_probe is True


def test_route_abs_component_token_active_expert_reuse_rejects_air_shaped_union() -> None:
    rng = np.random.default_rng(20260706)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(8, 256, 176),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(8, 256, 4)).astype(np.float16)
    route_experts = np.tile(np.arange(8, dtype=np.int32), (1024, 1))

    feasibility = evaluate_e8p_route_abs_component_token_active_expert_feasibility(
        codes,
        scales,
        route_experts,
        token_count=1024,
        top_k=8,
        group_size=352,
        bn=64,
        bk=64,
        projection_cache_byte_cap=8 * 1024 * 1024,
    )

    assert feasibility.decision == (
        "reject_route_abs_component_token_active_expert_cache_too_large"
    )
    assert feasibility.token_kblock_abs_index_count_shape == (1024, 22)
    assert feasibility.max_abs_indices_per_token_kblock == 256
    assert feasibility.ragged_token_active_projection_cache_bytes == 92_274_688
    assert feasibility.full_token_projection_cache_bytes == 92_274_688
    assert feasibility.rowwise_projection_cache_bytes == 738_197_504
    assert feasibility.cache_reduction_vs_rowwise == 8.0
    assert feasibility.candidate_ready_for_source_probe is False
    assert (
        "do_not_benchmark_token_active_expert_abs_component_cache"
        in feasibility.rejected_next_steps
    )


def test_route_abs_component_lower_bound_tracks_active_expert_exact_cache() -> None:
    codes = np.zeros((3, 16, 8), dtype=np.uint16)
    for expert, base in enumerate((3, 17, 29)):
        codes[expert, :, :] = np.uint16(base << 8)
    scales = np.ones((3, 16, 1), dtype=np.float16)
    route_experts = np.array([[0, 1], [1, 2], [2, 2], [0, 2]], dtype=np.int32)

    feasibility = evaluate_e8p_route_abs_component_lower_bound_feasibility(
        codes,
        scales,
        route_experts,
        token_count=4,
        top_k=2,
        group_size=64,
        bn=16,
        bk=64,
        projection_cache_byte_cap=4096,
    )

    assert feasibility.decision == "route_abs_component_exact_lower_bound_under_cache_cap"
    assert feasibility.exact_lower_bound_cache_elements == 7 * 8
    assert feasibility.exact_lower_bound_cache_bytes == 112
    assert feasibility.lower_bound_ratio_to_cap == 112 / 4096
    assert feasibility.lower_bound_preserves_route_slot_exactness is True
    assert feasibility.lower_bound_preserves_e8p_sign_axis is True
    assert feasibility.candidate_ready_for_source_probe is True
    assert "route_abs_component_token_active_expert_reuse" in (
        feasibility.families_blocked_by_lower_bound
    )


def test_route_abs_component_lower_bound_rejects_air_shaped_exact_family() -> None:
    rng = np.random.default_rng(20260707)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(8, 256, 176),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(8, 256, 4)).astype(np.float16)
    route_experts = np.tile(np.arange(8, dtype=np.int32), (1024, 1))

    feasibility = evaluate_e8p_route_abs_component_lower_bound_feasibility(
        codes,
        scales,
        route_experts,
        token_count=1024,
        top_k=8,
        group_size=352,
        bn=64,
        bk=64,
        projection_cache_byte_cap=8 * 1024 * 1024,
    )

    assert feasibility.decision == (
        "reject_exact_route_abs_component_projection_lower_bound_over_cap"
    )
    assert feasibility.max_abs_indices_per_token_kblock == 256
    assert feasibility.exact_lower_bound_cache_bytes == 92_274_688
    assert feasibility.projection_cache_byte_cap == 8_388_608
    assert feasibility.lower_bound_ratio_to_cap == 11.0
    assert feasibility.rowwise_projection_cache_bytes == 738_197_504
    assert feasibility.reduction_vs_rowwise == 8.0
    assert feasibility.candidate_ready_for_source_probe is False
    assert (
        "do_not_benchmark_exact_route_abs_component_projection_family"
        in feasibility.rejected_next_steps
    )


def test_expert_kblock_v2_dispatch_contract_groups_output_tiles_under_reuse_scope() -> None:
    rng = np.random.default_rng(20260703)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(2, 130, 16),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(2, 130, 2)).astype(np.float16)
    packed = pack_e8p_expert_kblock_factor_reuse_rhs_tiles(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
    )

    contract = build_e8p_expert_kblock_dispatch_contract(packed, route_tile_count=4)

    assert contract.target_kernel_family == (
        "sorted_gather_qmm_rhs_nax_expert_kblock_factor_reuse_v2"
    )
    assert contract.dispatch_grid == "experts_x_k_blocks_x_route_tiles"
    assert contract.incompatible_primitive_grid == "n_tiles_x_num_route_tiles"
    assert contract.reuse_scope == "expert_kblock"
    assert contract.slot_layout == "expert_n_tile_k_block_row_codeword"
    assert contract.decoded_dense_weight_bytes == 0
    assert contract.required_next_features == (
        "dispatch_over_reuse_scope_before_output_tile",
        "broaden_decode_reuse_scope_before_tensorops_fragment_fill",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    )
    assert len(contract.workgroups) == 2 * 2 * 4

    first = contract.workgroups[0]
    assert first.expert == 0
    assert first.k_block == 0
    assert first.route_tile == 0
    assert first.n_tiles == (0, 1, 2)
    assert first.sign_lut_index == (0, 0)
    assert first.abs_index_lut_index == (0, 0)
    assert first.sign_count == int(packed.sign_byte_counts[0, 0])
    assert first.abs_index_count == int(packed.abs_index_counts[0, 0])
    assert first.scale_group_indices == (0,)
    assert first.codeword_scale_slots == (0, 0, 0, 0, 0, 0, 0, 0)


def test_expert_kblock_v2_partial_reduction_contract_requires_per_kblock_partials() -> None:
    rng = np.random.default_rng(20260703)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(2, 130, 16),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(2, 130, 2)).astype(np.float16)
    packed = pack_e8p_expert_kblock_factor_reuse_rhs_tiles(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
    )
    dispatch = build_e8p_expert_kblock_dispatch_contract(packed, route_tile_count=3)

    contract = build_e8p_expert_kblock_partial_reduction_contract(
        dispatch,
        route_tile_size=64,
    )

    assert contract.target_kernel_family == (
        "sorted_gather_qmm_rhs_nax_expert_kblock_factor_reuse_v2"
    )
    assert contract.partial_accumulation_grid == "route_tiles_x_k_blocks_x_n_tiles"
    assert contract.final_reduction_grid == "route_tiles_x_n_tiles"
    assert contract.partial_accumulation_shape == (3, 2, 3, 64, 64)
    assert contract.partial_accumulation_dtype == "float32"
    assert contract.partial_accumulation_element_count == 3 * 2 * 3 * 64 * 64
    assert contract.partial_accumulation_bytes == 3 * 2 * 3 * 64 * 64 * 4
    assert contract.reduction_axis == "k_blocks"
    assert contract.output_shape == "route_count_x_output_dims"
    assert contract.valid_routes_source == "route_tile_counts"
    assert contract.expert_source == "route_tile_experts"
    assert contract.decoded_dense_weight_bytes == 0
    assert contract.required_next_features == (
        "allocate_partial_accumulation_scratch",
        "write_one_partial_per_kblock_workgroup",
        "reduce_kblock_partials_before_final_output",
        "keep_final_output_write_out_of_kblock_zero_only",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    )
    assert contract.rejected_next_steps == (
        "do_not_benchmark_single_kblock_completion_v2",
        "do_not_write_complete_output_from_one_kblock",
        "do_not_promote_without_final_reduction_pass",
    )


def test_next_kernel_family_guardrail_rejects_expert_kblock_near_duplicate() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_expert_kblock_factor_reuse_v3",
        storage_constraint="compressed_e8p_expert_kblock_factor_lut_scales",
        dispatch_grid="experts_x_k_blocks_x_route_tiles",
        decode_reuse_scope="expert_kblock",
        tensorops_rhs_source="cooperative_b_fragment_fill",
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=True,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_current_family_near_duplicate"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == [
        "expert_kblock_factor_reuse",
    ]
    assert verdict["rejected_next_steps"] == [
        "do_not_benchmark_near_duplicate_expert_kblock_factor_reuse",
        "do_not_retime_current_tensorops_fragment_decode_shape",
    ]
    assert verdict["required_next_features"] == [
        "materially_change_rhs_storage_contract_or_kernel_family",
        "avoid_per_fragment_e8p_reconstruction",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    ]


def test_next_kernel_family_guardrail_allows_materially_new_streaming_contract() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_component_stream",
        storage_constraint="compressed_e8p_component_stream_scales",
        dispatch_grid="route_tiles_x_experts_x_k_blocks_x_component_pairs",
        decode_reuse_scope="component_stream_before_accumulation",
        tensorops_rhs_source="streamed_component_outer_product_no_b_fragment",
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "candidate_family_ready_for_source_structure_probe"
    assert verdict["candidate_ready_for_source_probe"] is True
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False
    assert verdict["matched_rejected_families"] == []
    assert verdict["required_before_benchmark"] == [
        "source_structure_guardrail",
        "native_scalar_oracle_parity",
        "air_down_group_size_352_parity",
        "same_window_q2_speed_packet",
    ]


def test_next_kernel_family_guardrail_rejects_component_stream_shared_decode_cross_ntile_reuse() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_component_stream_shared_decode",
        storage_constraint="compressed_e8p_component_stream_scales",
        dispatch_grid="route_tiles_x_experts_x_k_blocks_x_component_pairs",
        decode_reuse_scope="component_pair_kblock_shared_across_n_tiles",
        tensorops_rhs_source="component_decode_cache_reused_across_n_tiles",
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_component_stream_shared_decode_cross_ntile_cache"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["component_stream_shared_decode"]
    assert "candidate depends on invalid component-stream cross-N-tile decode reuse" in verdict[
        "structural_reasons"
    ]
    assert verdict["required_next_features"] == [
        "materially_change_rhs_storage_contract_or_kernel_family",
        "avoid_cross_ntile_component_stream_decode_cache",
        "avoid_decoded_rhs_equivalent_cache",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    ]
    assert verdict["rejected_next_steps"] == [
        "do_not_benchmark_scalar_shared_decode_cache_scaffold",
        "do_not_claim_cross_ntile_shared_decode_from_component_stream_storage",
        "do_not_retime_component_stream_tensorops_unchanged",
    ]


def test_next_kernel_family_guardrail_rejects_expert_kblock_predecode_cache() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_expert_kblock_predecode_cache",
        storage_constraint="compressed_e8p_expert_kblock_factor_lut_scales",
        dispatch_grid="experts_x_k_blocks_x_route_tiles",
        decode_reuse_scope="expert_kblock_predecode_cache",
        tensorops_rhs_source="predecoded_expert_kblock_b_cache",
        preserves_compressed_rhs_storage=False,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=180224,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_expert_kblock_predecode_cache_decoded_rhs_equivalent"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["expert_kblock_factor_reuse"]
    assert (
        "candidate depends on decoded-RHS-equivalent expert/K-block predecode cache"
        in verdict["structural_reasons"]
    )
    assert verdict["required_next_features"] == [
        "materially_change_rhs_storage_contract_or_kernel_family",
        "avoid_expert_kblock_predecoded_rhs_cache",
        "avoid_decoded_rhs_equivalent_cache",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    ]
    assert verdict["rejected_next_steps"] == [
        "do_not_benchmark_expert_kblock_predecode_cache",
        "do_not_materialize_decoded_rhs_equivalent_expert_kblock_cache",
        "do_not_retime_near_duplicate_expert_kblock_v2",
    ]


def test_route_codeword_projection_feasibility_allows_bounded_unique_codewords() -> None:
    rows = np.arange(128, dtype=np.uint16)[:, None]
    words = np.arange(16, dtype=np.uint16)[None, :]
    codes = ((rows % 4) + words).reshape(1, 128, 16).astype(np.uint16)
    scales = np.ones((1, 128, 2), dtype=np.float16)

    feasibility = evaluate_e8p_route_codeword_projection_feasibility(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
        route_tile_count=3,
        route_tile_size=64,
        unique_codeword_cap=32,
        projection_cache_byte_cap=4096,
    )

    assert feasibility.target_kernel_family == (
        "sorted_gather_qmm_rhs_nax_route_codeword_projection"
    )
    assert feasibility.storage_constraint == "compressed_e8p_route_codeword_projection_scales"
    assert feasibility.dispatch_grid == "route_tiles_x_experts_x_k_blocks"
    assert feasibility.decision == "route_codeword_projection_ready_for_source_probe"
    assert feasibility.candidate_ready_for_source_probe is True
    assert feasibility.decoded_dense_weight_bytes == 0
    assert feasibility.preserves_compressed_rhs_storage is True
    assert feasibility.preserves_codeword_scale_slots is True
    assert feasibility.unique_codeword_counts_shape == (1, 2)
    assert feasibility.max_unique_codewords_per_expert_kblock == 11
    assert feasibility.projection_cache_shape == (3, 2, 11)
    assert feasibility.projection_cache_bytes == 3 * 2 * 11 * 2
    assert feasibility.required_next_features == (
        "source_structure_guardrail",
        "activation_codeword_dot_cache_parity",
        "air_down_group_size_352_parity",
        "same_window_q2_speed_packet_after_native_parity",
    )


def test_route_codeword_projection_feasibility_rejects_unbounded_cache() -> None:
    codes = np.arange(2 * 128 * 16, dtype=np.uint16).reshape(2, 128, 16)
    scales = np.ones((2, 128, 2), dtype=np.float16)

    feasibility = evaluate_e8p_route_codeword_projection_feasibility(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
        route_tile_count=4,
        route_tile_size=64,
        unique_codeword_cap=128,
        projection_cache_byte_cap=1024,
    )

    assert feasibility.decision == "reject_route_codeword_projection_cache_too_large"
    assert feasibility.candidate_ready_for_source_probe is False
    assert feasibility.max_unique_codewords_per_expert_kblock == 1024
    assert feasibility.projection_cache_shape == (4, 2, 1024)
    assert feasibility.projection_cache_bytes == 4 * 2 * 1024 * 2
    assert feasibility.rejected_next_steps == (
        "do_not_benchmark_route_codeword_projection_cache",
        "do_not_materialize_unbounded_activation_codeword_dot_cache",
        "do_not_claim_speed_from_unique_count_probe",
    )


def test_next_kernel_family_guardrail_allows_route_codeword_projection_family() -> None:
    verdict = evaluate_e8p_next_kernel_family_candidate(
        E8PNextKernelFamilyCandidate(
            target_kernel_family="sorted_gather_qmm_rhs_nax_route_codeword_projection",
            storage_constraint="compressed_e8p_route_codeword_projection_scales",
            dispatch_grid="route_tiles_x_experts_x_k_blocks",
            decode_reuse_scope="route_activation_codeword_dot_cache",
            tensorops_rhs_source="activation_side_codeword_dot_lookup_no_b_fragment",
            preserves_compressed_rhs_storage=True,
            preserves_codeword_scale_slots=True,
            decoded_dense_weight_bytes=0,
        )
    )

    assert verdict["decision"] == "candidate_family_ready_for_source_structure_probe"
    assert verdict["candidate_ready_for_source_probe"] is True
    assert verdict["matched_rejected_families"] == []
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_route_active_codeword_projection_feasibility_uses_actual_route_tiles() -> None:
    codes = np.zeros((2, 128, 16), dtype=np.uint16)
    codes[0, :, :8] = (np.arange(128, dtype=np.uint16)[:, None] % 4)
    codes[0, :, 8:16] = (np.arange(128, dtype=np.uint16)[:, None] % 8) + np.uint16(20)
    codes[1, :, :8] = (np.arange(128, dtype=np.uint16)[:, None] % 3) + np.uint16(40)
    codes[1, :, 8:16] = (np.arange(128, dtype=np.uint16)[:, None] % 5) + np.uint16(80)
    scales = np.ones((2, 128, 2), dtype=np.float16)
    route_experts = np.array([0] * 65 + [1], dtype=np.int32)

    feasibility = evaluate_e8p_route_active_codeword_projection_feasibility(
        codes,
        scales,
        route_experts,
        group_size=64,
        bn=64,
        bk=64,
        route_tile_size=64,
        unique_codeword_cap=16,
        projection_cache_byte_cap=1024,
    )

    assert feasibility.decision == "route_active_codeword_projection_ready_for_source_probe"
    assert feasibility.route_count == 66
    assert feasibility.active_expert_count == 2
    assert feasibility.active_route_tile_count == 3
    assert feasibility.active_route_tile_counts_shape == (2,)
    assert feasibility.active_route_tile_counts == (2, 1)
    assert feasibility.active_projection_cache_elements == (2 * (4 + 8)) + (1 * (3 + 5))
    assert feasibility.active_projection_cache_bytes == 32 * 2
    assert feasibility.global_max_projection_cache_bytes == 3 * 2 * 8 * 2
    assert feasibility.candidate_ready_for_source_probe is True
    assert feasibility.required_next_features == (
        "source_structure_guardrail",
        "active_route_tile_codeword_dot_cache_parity",
        "air_down_group_size_352_parity",
        "same_window_q2_speed_packet_after_native_parity",
    )


def test_route_active_codeword_projection_feasibility_rejects_large_active_cache() -> None:
    codes = np.arange(2 * 128 * 16, dtype=np.uint16).reshape(2, 128, 16)
    scales = np.ones((2, 128, 2), dtype=np.float16)
    route_experts = np.array([0] * 128 + [1] * 64, dtype=np.int32)

    feasibility = evaluate_e8p_route_active_codeword_projection_feasibility(
        codes,
        scales,
        route_experts,
        group_size=64,
        bn=64,
        bk=64,
        route_tile_size=64,
        unique_codeword_cap=2048,
        projection_cache_byte_cap=4096,
    )

    assert feasibility.decision == "reject_route_active_codeword_projection_cache_too_large"
    assert feasibility.active_route_tile_count == 3
    assert feasibility.active_route_tile_counts == (2, 1)
    assert feasibility.max_unique_codewords_per_active_expert_kblock == 1024
    assert feasibility.active_projection_cache_elements == 3 * 2 * 1024
    assert feasibility.active_projection_cache_bytes == 3 * 2 * 1024 * 2
    assert feasibility.candidate_ready_for_source_probe is False
    assert feasibility.rejected_next_steps == (
        "do_not_benchmark_route_active_codeword_projection_cache",
        "do_not_materialize_unbounded_active_route_codeword_dot_cache",
        "do_not_claim_speed_from_active_route_cardinality_probe",
    )


def test_route_abs_index_projection_feasibility_rejects_missing_sign_axis() -> None:
    rows = np.arange(128, dtype=np.uint16)[:, None]
    words = np.arange(16, dtype=np.uint16)[None, :]
    signs = ((rows * 17 + words * 3) & np.uint16(0xFF)).reshape(1, 128, 16)
    abs_indices = np.full((1, 128, 16), 23, dtype=np.uint16)
    codes = signs | (abs_indices << np.uint16(8))
    scales = np.ones((1, 128, 2), dtype=np.float16)

    feasibility = evaluate_e8p_route_abs_index_projection_feasibility(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
        route_tile_count=4,
        route_tile_size=64,
        projection_cache_byte_cap=1024,
    )

    assert feasibility.target_kernel_family == (
        "sorted_gather_qmm_rhs_nax_route_abs_index_projection"
    )
    assert feasibility.decision == "reject_route_abs_index_projection_requires_sign_axis"
    assert feasibility.abs_only_projection_exact is False
    assert feasibility.sign_values_depend_on_output_column is True
    assert feasibility.max_abs_indices_per_expert_kblock == 1
    assert feasibility.max_signed_abs_pairs_per_expert_kblock == 256
    assert feasibility.abs_projection_cache_shape == (4, 2, 1)
    assert feasibility.correct_signed_projection_cache_shape == (4, 2, 256)
    assert feasibility.correct_cache_is_full_codeword_equivalent is True
    assert feasibility.candidate_ready_for_source_probe is False
    assert feasibility.rejected_next_steps == (
        "do_not_benchmark_route_abs_index_projection_cache",
        "do_not_ignore_sign_axis_for_e8p_route_projection",
        "do_not_claim_speed_from_abs_index_cardinality",
    )


def test_next_kernel_family_guardrail_rejects_route_abs_index_projection_without_sign_axis() -> None:
    verdict = evaluate_e8p_next_kernel_family_candidate(
        E8PNextKernelFamilyCandidate(
            target_kernel_family="sorted_gather_qmm_rhs_nax_route_abs_index_projection",
            storage_constraint="compressed_e8p_route_abs_index_projection_scales",
            dispatch_grid="route_tiles_x_experts_x_k_blocks",
            decode_reuse_scope="route_activation_abs_index_dot_cache",
            tensorops_rhs_source="activation_side_abs_index_dot_lookup_no_sign_axis",
            preserves_compressed_rhs_storage=True,
            preserves_codeword_scale_slots=True,
            decoded_dense_weight_bytes=0,
        )
    )

    assert verdict["decision"] == "reject_route_abs_index_projection_missing_sign_axis"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["route_abs_index_projection"]
    assert "candidate drops the E8P sign axis from route projection" in verdict[
        "structural_reasons"
    ]
    assert verdict["required_next_features"] == [
        "materially_change_rhs_storage_contract_or_kernel_family",
        "preserve_e8p_sign_axis_or_use_full_codeword_projection",
        "avoid_full_codeword_equivalent_route_cache",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    ]


def test_route_abs_component_projection_feasibility_keeps_sign_axis_out_of_cache() -> None:
    rows = np.arange(128, dtype=np.uint16)[:, None]
    words = np.arange(16, dtype=np.uint16)[None, :]
    signs = ((rows * 17 + words * 3) & np.uint16(0xFF)).reshape(1, 128, 16)
    abs_indices = (words % 11).reshape(1, 1, 16).astype(np.uint16)
    codes = signs | (abs_indices << np.uint16(8))
    scales = np.ones((1, 128, 2), dtype=np.float16)

    feasibility = evaluate_e8p_route_abs_component_projection_feasibility(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
        route_tile_count=3,
        route_tile_size=64,
        abs_index_cap=16,
        projection_cache_byte_cap=4096,
    )

    assert feasibility.target_kernel_family == (
        "sorted_gather_qmm_rhs_nax_route_abs_component_projection"
    )
    assert feasibility.decision == "route_abs_component_projection_ready_for_source_probe"
    assert feasibility.abs_component_projection_exact is True
    assert feasibility.sign_values_apply_after_cache_lookup is True
    assert feasibility.scale_values_apply_after_cache_lookup is True
    assert feasibility.max_abs_indices_per_expert_kblock == 8
    assert feasibility.component_axis == 8
    assert feasibility.projection_cache_shape == (3, 2, 8, 8)
    assert feasibility.projection_cache_bytes == 3 * 2 * 8 * 8 * 2
    assert feasibility.full_codeword_cache_bytes == 3 * 2 * 1024 * 2
    assert feasibility.decoded_dense_weight_bytes == 0
    assert feasibility.candidate_ready_for_source_probe is True
    assert feasibility.required_next_features == (
        "source_structure_guardrail",
        "activation_abs_component_dot_cache_parity",
        "apply_sign_and_scale_after_cache_lookup",
        "air_down_group_size_352_parity",
        "same_window_q2_speed_packet_after_native_parity",
    )


def test_route_abs_component_projection_feasibility_rejects_large_cache() -> None:
    rows = np.arange(128, dtype=np.uint16)[:, None]
    words = np.arange(16, dtype=np.uint16)[None, :]
    signs = ((rows * 17 + words * 3) & np.uint16(0xFF)).reshape(1, 128, 16)
    abs_indices = ((rows * 16 + words) % np.uint16(256)).reshape(1, 128, 16)
    codes = signs | (abs_indices << np.uint16(8))
    scales = np.ones((1, 128, 2), dtype=np.float16)

    feasibility = evaluate_e8p_route_abs_component_projection_feasibility(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
        route_tile_count=4,
        route_tile_size=64,
        abs_index_cap=128,
        projection_cache_byte_cap=1024,
    )

    assert feasibility.decision == "reject_route_abs_component_projection_cache_too_large"
    assert feasibility.candidate_ready_for_source_probe is False
    assert feasibility.max_abs_indices_per_expert_kblock == 128
    assert feasibility.projection_cache_shape == (4, 2, 128, 8)
    assert feasibility.projection_cache_bytes == 4 * 2 * 128 * 8 * 2
    assert feasibility.rejected_next_steps == (
        "do_not_benchmark_route_abs_component_projection_cache",
        "do_not_materialize_unbounded_abs_component_dot_cache",
        "do_not_claim_speed_from_abs_component_cardinality_probe",
    )


def test_next_kernel_family_guardrail_rejects_route_abs_component_projection_family() -> None:
    verdict = evaluate_e8p_next_kernel_family_candidate(
        E8PNextKernelFamilyCandidate(
            target_kernel_family="sorted_gather_qmm_rhs_nax_route_abs_component_projection",
            storage_constraint="compressed_e8p_route_abs_component_projection_scales",
            dispatch_grid="route_tiles_x_experts_x_k_blocks_x_abs_components",
            decode_reuse_scope="route_activation_abs_component_dot_cache",
            tensorops_rhs_source="activation_side_abs_component_dot_lookup_sign_scale_after_lookup",
            preserves_compressed_rhs_storage=True,
            preserves_codeword_scale_slots=True,
            decoded_dense_weight_bytes=0,
        )
    )

    assert verdict["decision"] == "reject_route_abs_component_projection_lower_bound_over_cap"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["route_abs_component_projection"]
    assert "prove_new_exact_compression_below_abs_component_lower_bound" in (
        verdict["required_next_features"]
    )
    assert "do_not_benchmark_exact_route_abs_component_projection_family" in (
        verdict["rejected_next_steps"]
    )
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_pack_e8p_component_stream_tiles_preserves_component_bits_and_scale_slots() -> None:
    rng = np.random.default_rng(20260708)
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)

    packed = pack_e8p_component_stream_rhs_tiles(
        codes,
        scales,
        group_size=group_size,
        bn=64,
        bk=64,
    )

    assert packed.layout.target_kernel_family == "sorted_gather_qmm_rhs_nax_component_stream"
    assert packed.layout.storage_constraint == "compressed_e8p_component_stream_scales"
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert packed.sign_component_bits.shape == (2, 2, 11, 64, 8, 8)
    assert packed.sign_component_bits.dtype == np.uint8
    assert packed.abs_index_tiles.shape == (2, 2, 11, 64, 8)
    assert packed.component_scale_slots.shape == (11, 64)
    assert packed.component_codeword_indices.shape == (11, 64)
    assert packed.component_offsets.shape == (11, 64)
    np.testing.assert_array_equal(packed.scale_group_indices[5], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(packed.component_scale_slots[5, :32], np.zeros(32, dtype=np.int32))
    np.testing.assert_array_equal(packed.component_scale_slots[5, 32:], np.ones(32, dtype=np.int32))
    np.testing.assert_array_equal(
        packed.component_codeword_indices[5],
        np.repeat(np.arange(8, dtype=np.int32), 8),
    )
    np.testing.assert_array_equal(
        packed.component_offsets[5],
        np.tile(np.arange(8, dtype=np.int32), 8),
    )

    expected_codes = codes[1, 67, 40:48]
    expected_sign_bits = (
        ((expected_codes[:, None] & np.uint16(0xFF)) >> np.arange(8, dtype=np.uint16)) & 1
    ).astype(np.uint8)
    np.testing.assert_array_equal(packed.sign_component_bits[1, 1, 5, 3], expected_sign_bits)
    np.testing.assert_array_equal(
        packed.abs_index_tiles[1, 1, 5, 3],
        (expected_codes >> np.uint16(8)).astype(np.uint8),
    )

    observed = decode_e8p_component_stream_rhs_tile(
        packed,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )
    expected = decode_e8p_rhs_tile(
        pack_e8p_rhs_tiles(codes, scales, group_size=group_size, bn=64, bk=64),
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)


def test_component_stream_contract_dispatches_component_pairs_without_b_fragment() -> None:
    rng = np.random.default_rng(20260708)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(2, 130, 16),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(2, 130, 2)).astype(np.float16)
    packed = pack_e8p_component_stream_rhs_tiles(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
    )

    contract = build_e8p_component_stream_contract(
        packed,
        route_tile_count=3,
        component_pair_size=2,
    )

    assert contract.target_kernel_family == "sorted_gather_qmm_rhs_nax_component_stream"
    assert contract.storage_constraint == "compressed_e8p_component_stream_scales"
    assert contract.dispatch_grid == "route_tiles_x_experts_x_k_blocks_x_component_pairs"
    assert contract.decode_reuse_scope == "component_stream_before_accumulation"
    assert contract.rhs_source == "streamed_component_outer_product_no_b_fragment"
    assert contract.decoded_dense_weight_bytes == 0
    assert contract.component_count_per_kblock == 64
    assert contract.component_pair_size == 2
    assert contract.component_pair_count == 32
    assert contract.required_before_native_parity == (
        "native_component_stream_scalar_oracle",
        "air_down_group_size_352_component_scale_parity",
        "source_structure_guardrail",
    )
    assert contract.rejected_next_steps == (
        "do_not_fill_tensorops_b_fragment_from_e8p_values",
        "do_not_materialize_decoded_dense_rhs",
        "do_not_claim_speed_before_same_window_q2_packet",
    )
    assert len(contract.workgroups) == 3 * 2 * 2 * 32

    first = contract.workgroups[0]
    assert first.route_tile == 0
    assert first.expert == 0
    assert first.k_block == 0
    assert first.component_pair == 0
    assert first.component_offsets == (0, 1)
    assert first.codeword_indices == (0, 0)
    assert first.codeword_component_offsets == (0, 1)
    assert first.scale_slots == (0, 0)
    assert first.n_tiles == (0, 1, 2)


def test_component_stream_partial_reduction_contract_splits_component_pairs_before_speed() -> None:
    rng = np.random.default_rng(20260710)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(2, 130, 88),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(2, 130, 2)).astype(np.float16)
    packed = pack_e8p_component_stream_rhs_tiles(
        codes,
        scales,
        group_size=352,
        bn=64,
        bk=64,
    )
    dispatch = build_e8p_component_stream_contract(
        packed,
        route_tile_count=3,
        component_pair_size=4,
    )

    contract = build_e8p_component_stream_partial_reduction_contract(
        dispatch,
        route_tile_size=64,
        accumulation_dtype="float32",
    )

    assert contract.target_kernel_family == "sorted_gather_qmm_rhs_nax_component_stream"
    assert contract.partial_accumulation_grid == (
        "route_tiles_x_k_blocks_x_component_pairs_x_n_tiles"
    )
    assert contract.final_reduction_grid == "route_tiles_x_n_tiles"
    assert contract.partial_accumulation_shape == (3, 11, 16, 3, 64, 64)
    assert contract.partial_accumulation_dtype == "float32"
    assert contract.partial_accumulation_bytes == 3 * 11 * 16 * 3 * 64 * 64 * 4
    assert contract.reduction_axes == ("k_blocks", "component_pairs")
    assert contract.valid_routes_source == "route_tile_counts"
    assert contract.expert_source == "route_tile_experts"
    assert contract.decoded_dense_weight_bytes == 0
    assert contract.required_next_features == (
        "allocate_component_pair_partial_scratch",
        "write_one_partial_per_component_pair_workgroup",
        "reduce_kblock_and_component_pair_partials_before_final_output",
        "preserve_compressed_component_stream_storage",
        "prove_speed_path_against_decoded_sorted_reference",
    )
    assert contract.rejected_next_steps == (
        "do_not_benchmark_scalar_component_stream_oracle",
        "do_not_sum_component_pairs_inside_single_output_thread",
        "do_not_promote_without_same_window_q2_speed_packet",
    )


def test_component_stream_shared_decode_contract_changes_reuse_scope_after_tensorops_rejection() -> None:
    rng = np.random.default_rng(20260711)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(2, 130, 88),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(2, 130, 2)).astype(np.float16)
    packed = pack_e8p_component_stream_rhs_tiles(
        codes,
        scales,
        group_size=352,
        bn=64,
        bk=64,
    )
    dispatch = build_e8p_component_stream_contract(
        packed,
        route_tile_count=3,
        component_pair_size=4,
    )

    contract = build_e8p_component_stream_shared_decode_contract(
        dispatch,
        route_tile_size=64,
        component_decode_dtype="float16",
    )

    assert contract.target_kernel_family == "sorted_gather_qmm_rhs_nax_shared_decode"
    assert contract.dispatch_grid == "route_tiles_x_experts_x_k_blocks_x_component_pairs"
    assert contract.decode_reuse_scope == "component_pair_kblock_shared_across_n_tiles"
    assert contract.shared_decode_shape == (3, 2, 11, 16, 64, 4)
    assert contract.shared_decode_dtype == "float16"
    assert contract.shared_decode_bytes == 3 * 2 * 11 * 16 * 64 * 4 * 2
    assert contract.output_accumulation_grid == "route_tiles_x_n_tiles"
    assert contract.decoded_dense_weight_bytes == 0
    assert contract.preserved_storage_contract == "compressed_e8p_component_stream_scales"
    assert contract.required_next_features == (
        "native_shared_decode_cache_source_guardrail",
        "prove_shared_decode_cache_matches_component_stream_oracle",
        "prove_artifact_shared_decode_cache_parity",
        "same_window_q2_speed_packet_after_parity_only",
    )
    assert contract.rejected_next_steps == (
        "do_not_retime_component_stream_tensorops_unchanged",
        "do_not_fill_tensorops_b_fragment_from_e8p_values",
        "do_not_use_scalar_component_stream_partial_body_as_speed_path",
        "do_not_materialize_decoded_dense_rhs",
    )


def test_component_stream_shared_decode_feasibility_rejects_cross_ntile_cache() -> None:
    rng = np.random.default_rng(20260712)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(2, 130, 88),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(2, 130, 2)).astype(np.float16)
    packed = pack_e8p_component_stream_rhs_tiles(
        codes,
        scales,
        group_size=352,
        bn=64,
        bk=64,
    )
    dispatch = build_e8p_component_stream_contract(
        packed,
        route_tile_count=3,
        component_pair_size=4,
    )

    feasibility = evaluate_e8p_component_stream_shared_decode_feasibility(
        dispatch,
        route_tile_size=64,
        component_decode_dtype="float16",
    )

    assert feasibility.decision == "reject_component_stream_shared_decode_cross_ntile_cache"
    assert feasibility.n_tile_reuse_feasible is False
    assert feasibility.component_values_depend_on_output_column is True
    assert feasibility.scale_values_depend_on_output_column is True
    assert feasibility.correct_cache_shape == (2, 3, 11, 64, 64)
    assert feasibility.correct_cache_bytes == 2 * 3 * 11 * 64 * 64 * 2
    assert feasibility.correct_cache_is_decoded_rhs_equivalent is True
    assert feasibility.next_track_b_hypothesis == "change_component_stream_rhs_layout_or_kernel_family"
    assert "do_not_benchmark_scalar_shared_decode_cache_scaffold" in feasibility.rejected_next_steps


def test_expert_kblock_decode_reuse_feasibility_rejects_predecode_cache() -> None:
    rng = np.random.default_rng(20260713)
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(1, 128, 88),
        dtype=np.uint16,
    )
    scales = rng.uniform(0.01, 0.05, size=(1, 128, 2)).astype(np.float16)
    packed = pack_e8p_expert_kblock_factor_reuse_rhs_tiles(
        codes,
        scales,
        group_size=352,
        bn=64,
        bk=64,
    )

    feasibility = evaluate_e8p_expert_kblock_decode_reuse_feasibility(
        packed,
        decode_cache_dtype="float16",
    )

    assert feasibility.decision == "reject_expert_kblock_predecode_cache_decoded_rhs_equivalent"
    assert feasibility.output_tile_reuse_feasible_without_decoded_cache is False
    assert feasibility.factor_lut_shared_across_output_tiles is True
    assert feasibility.slot_values_depend_on_output_column is True
    assert feasibility.scale_values_depend_on_output_column is True
    assert feasibility.correct_cache_shape == (1, 11, 2, 64, 64)
    assert feasibility.correct_cache_bytes == 1 * 11 * 2 * 64 * 64 * 2
    assert feasibility.correct_cache_bytes == 1 * 128 * 704 * 2
    assert feasibility.correct_cache_is_decoded_rhs_equivalent is True
    assert (
        feasibility.next_track_b_hypothesis
        == "change_expert_kblock_rhs_layout_or_kernel_family"
    )
    assert "do_not_benchmark_expert_kblock_predecode_cache" in feasibility.rejected_next_steps


def test_next_family_selector_rejects_speed_rejected_route_slot_codeword_stream() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_slot_codeword_stream",
        storage_constraint="compressed_e8p_route_slot_codeword_stream_scales",
        dispatch_grid="route_tiles_x_route_slots_x_k_blocks_x_codewords",
        decode_reuse_scope="route_slot_streaming_no_cross_ntile_cache",
        tensorops_rhs_source="stream_route_slot_codewords_direct_from_compressed_uint16",
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_route_slot_codeword_stream_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["route_slot_codeword_stream"]
    assert "do_not_retime_scalar_route_slot_codeword_stream" in verdict[
        "rejected_next_steps"
    ]


def test_next_family_selector_rejects_route_slot_mma_codeword_tile_successor() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile",
        storage_constraint="compressed_e8p_route_slot_mma_codeword_tiles",
        dispatch_grid="route_tiles_x_output_tiles_x_k_blocks_x_codeword_tiles",
        decode_reuse_scope="route_slot_output_tile_mma_codeword_tile_accumulation",
        tensorops_rhs_source="mma_codeword_basis_dot_accumulation_from_compressed_uint16",
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_route_slot_mma_codeword_tile_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["route_slot_mma_codeword_tile"]
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_next_family_selector_rejects_speed_rejected_route_slot_mma_codeword_tile() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_slot_mma_codeword_tile",
        storage_constraint="compressed_e8p_route_slot_mma_codeword_tiles",
        dispatch_grid="route_tiles_x_output_tiles_x_k_blocks_x_codeword_tiles",
        decode_reuse_scope="route_slot_output_tile_mma_codeword_tile_accumulation",
        tensorops_rhs_source="mma_codeword_basis_dot_accumulation_from_compressed_uint16",
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_route_slot_mma_codeword_tile_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["route_slot_mma_codeword_tile"]
    assert "do_not_retime_route_slot_mma_codeword_tile" in verdict[
        "rejected_next_steps"
    ]


def test_next_family_selector_rejects_active_route_tile_after_speed_no_go() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_active_route_tile_codeword_outer_product"
        ),
        storage_constraint="compressed_e8p_active_route_tile_codeword_outer_products",
        dispatch_grid="active_route_tiles_x_k_blocks_x_codewords_x_output_microtiles",
        decode_reuse_scope="active_route_tile_codeword_outer_product_accumulation",
        tensorops_rhs_source=(
            "outer_product_codeword_basis_accumulation_from_compressed_uint16"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_active_route_tile_codeword_outer_product_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == [
        "active_route_tile_codeword_outer_product"
    ]
    assert "do_not_retime_active_route_tile_codeword_outer_product" in verdict[
        "rejected_next_steps"
    ]


def test_next_family_selector_rejects_expert_cohort_after_speed_no_go() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_expert_cohort_codeword_broadcast"
        ),
        storage_constraint="compressed_e8p_expert_cohort_codeword_broadcasts",
        dispatch_grid="experts_x_route_cohorts_x_output_microtiles_x_k_blocks",
        decode_reuse_scope="expert_cohort_codeword_broadcast_accumulation",
        tensorops_rhs_source=(
            "expert_scoped_codeword_broadcast_from_compressed_uint16"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_expert_cohort_codeword_broadcast_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["expert_cohort_codeword_broadcast"]
    assert "do_not_retime_expert_cohort_codeword_broadcast" in verdict[
        "rejected_next_steps"
    ]
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_next_family_selector_rejects_token_cohort_mma_after_speed_no_go() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_token_cohort_mma_codeword_tile"
        ),
        storage_constraint="compressed_e8p_token_cohort_mma_codeword_tiles",
        dispatch_grid=(
            "token_cohorts_x_output_tiles_x_active_experts_x_k_blocks_x_codeword_tiles"
        ),
        decode_reuse_scope="token_cohort_output_tile_mma_codeword_accumulation",
        tensorops_rhs_source=(
            "compressed_codeword_tiles_with_token_cohort_route_descriptors"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_token_cohort_mma_codeword_tile_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["token_cohort_mma_codeword_tile"]
    assert "do_not_retime_token_cohort_mma_codeword_tile" in verdict[
        "rejected_next_steps"
    ]
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_next_family_selector_rejects_output_stationary_after_speed_no_go() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_output_stationary_codeword_tile",
        storage_constraint="compressed_e8p_output_stationary_codeword_tiles",
        dispatch_grid=(
            "output_tiles_x_route_batches_x_active_experts_x_k_blocks_x_codeword_tiles"
        ),
        decode_reuse_scope="output_stationary_route_batch_codeword_tile_accumulation",
        tensorops_rhs_source=(
            "compressed_codeword_tiles_with_output_stationary_route_batches"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_output_stationary_codeword_tile_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["output_stationary_codeword_tile"]
    assert "do_not_retime_output_stationary_codeword_tile" in verdict[
        "rejected_next_steps"
    ]


def test_next_family_selector_rejects_input_stationary_after_speed_no_go() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_input_stationary_codeword_tile",
        storage_constraint="compressed_e8p_input_stationary_codeword_tiles",
        dispatch_grid=(
            "input_tiles_x_route_batches_x_active_experts_x_output_tiles_x_codeword_tiles"
        ),
        decode_reuse_scope="input_stationary_kblock_codeword_tile_accumulation",
        tensorops_rhs_source="compressed_codeword_tiles_with_input_stationary_kblock_reuse",
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_input_stationary_codeword_tile_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["input_stationary_codeword_tile"]
    assert "do_not_retime_input_stationary_codeword_tile" in verdict[
        "rejected_next_steps"
    ]


def test_next_family_selector_allows_expert_kblock_codeword_factor_reuse_successor() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_expert_kblock_codeword_factor_reuse"
        ),
        storage_constraint="compressed_e8p_expert_kblock_codeword_factor_tiles",
        dispatch_grid="experts_x_k_blocks_x_output_tiles_x_route_tiles_x_codeword_tiles",
        decode_reuse_scope="expert_kblock_factor_decode_reused_across_output_tiles",
        tensorops_rhs_source=(
            "compressed_codeword_factor_tiles_with_expert_kblock_output_tile_reuse"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "candidate_family_ready_for_source_structure_probe"
    assert verdict["candidate_ready_for_source_probe"] is True
    assert verdict["matched_rejected_families"] == []
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_next_family_selector_accepts_route_batch_segmented_codeword_reduce() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_route_batch_segmented_codeword_reduce"
        ),
        storage_constraint="compressed_e8p_route_batch_segmented_codeword_reductions",
        dispatch_grid="route_batches_x_k_blocks_x_output_microtiles_x_codewords",
        decode_reuse_scope="route_batch_segmented_codeword_reduce_accumulation",
        tensorops_rhs_source=(
            "route_batch_segmented_reduce_from_compressed_uint16_codewords"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "candidate_family_ready_for_source_structure_probe"
    assert verdict["candidate_ready_for_source_probe"] is True
    assert verdict["matched_rejected_families"] == []
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_next_family_selector_rejects_route_codeword_lut_accumulate_family() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_codeword_lut_accumulate",
        storage_constraint="compressed_e8p_route_codeword_lut_accumulate_scales",
        dispatch_grid="route_slots_x_k_blocks_x_codewords_then_output_tiles",
        decode_reuse_scope="activation_side_route_codeword_lut_preserves_route_slots",
        tensorops_rhs_source=(
            "compressed_codewords_scales_with_route_local_codeword_dot_lut"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_route_codeword_lut_accumulate_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["route_codeword_lut_accumulate"]
    assert "avoid_route_local_full_codeword_lut" in verdict["required_next_features"]
    assert "do_not_retime_route_codeword_lut_accumulate" in (
        verdict["rejected_next_steps"]
    )
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_next_family_selector_rejects_rowwise_codeword_tile_accumulate_family() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_rowwise_codeword_tile_accumulate"
        ),
        storage_constraint="compressed_e8p_rowwise_codeword_tile_accumulate_scales",
        dispatch_grid="route_microtiles_x_output_tiles_x_k_blocks_x_codeword_tiles",
        decode_reuse_scope="rowwise_online_codeword_tile_accumulation_no_route_local_lut",
        tensorops_rhs_source="compressed_codeword_tiles_with_online_activation_dot_products",
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_rowwise_codeword_tile_accumulate_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == [
        "rowwise_codeword_tile_accumulate"
    ]
    assert "avoid_rowwise_per_output_row_codeword_dot_recompute" in (
        verdict["required_next_features"]
    )
    assert "do_not_retime_rowwise_codeword_tile_accumulate" in (
        verdict["rejected_next_steps"]
    )
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_next_family_selector_rejects_output_tile_local_codeword_lut_successor() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_output_tile_local_codeword_lut"
        ),
        storage_constraint="compressed_e8p_output_tile_local_codeword_lut_scales",
        dispatch_grid=(
            "route_microtiles_x_output_tiles_x_k_blocks_x_unique_codewords_then_output_rows"
        ),
        decode_reuse_scope="output_tile_local_unique_codeword_lut_bounded_by_tile",
        tensorops_rhs_source=(
            "compressed_codeword_tiles_with_output_tile_local_activation_dot_lut"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_output_tile_local_codeword_lut_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == ["output_tile_local_codeword_lut"]
    assert "avoid_output_tile_local_full_codeword_pass" in (
        verdict["required_next_features"]
    )
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_next_family_selector_rejects_token_route_output_stripe_pipeline_successor() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_token_route_output_stripe_pipeline"
        ),
        storage_constraint="compressed_e8p_token_route_output_stripe_pipelines",
        dispatch_grid="tokens_x_route_slots_x_output_stripes_x_kblock_stages",
        decode_reuse_scope=(
            "token_route_output_stripes_accumulate_full_output_without_route_expansion"
        ),
        tensorops_rhs_source=(
            "compressed_e8p_codeword_tiles_streamed_by_token_route_output_stripe"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(candidate)

    assert verdict["decision"] == "reject_token_route_output_stripe_pipeline_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == [
        "token_route_output_stripe_pipeline"
    ]
    assert "avoid_token_route_output_stripe_pipeline_schedule" in (
        verdict["required_next_features"]
    )
    assert "do_not_retime_token_route_output_stripe_pipeline" in (
        verdict["rejected_next_steps"]
    )
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_next_family_selector_rejects_route_block_output_group_stream_successor() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_route_block_output_group_stream"
        ),
        storage_constraint="compressed_e8p_route_block_output_group_streams",
        dispatch_grid="route_blocks_x_output_groups_x_k_blocks_x_codeword_groups",
        decode_reuse_scope=(
            "route_block_output_group_accumulators_written_directly_to_route_slots"
        ),
        tensorops_rhs_source=(
            "compressed_e8p_codeword_tiles_streamed_by_route_block_output_group"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(
        candidate,
        speed_rejected_kernel_families=("route_block_output_group_stream",),
    )

    assert verdict["decision"] == "reject_route_block_output_group_stream_speed_path"
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == [
        "route_block_output_group_stream"
    ]
    assert "avoid_route_block_output_group_stream_schedule" in (
        verdict["required_next_features"]
    )
    assert "do_not_retime_route_block_output_group_stream" in (
        verdict["rejected_next_steps"]
    )
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False


def test_component_stream_sorted_matmul_oracle_matches_packed_u16_tiles_with_down_scale_crossing() -> None:
    rng = np.random.default_rng(20260709)
    experts = 3
    routes = 17
    output_dims = 70
    input_dims = 704
    group_size = 352
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_experts = np.sort(rng.integers(0, experts, size=(routes,), dtype=np.int32), kind="stable")
    sorted_x = rng.normal(size=(routes, input_dims)).astype(np.float16)
    component = pack_e8p_component_stream_rhs_tiles(
        codes,
        scales,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    packed_u16 = pack_e8p_rhs_tiles(codes, scales, group_size=group_size, bn=64, bk=64)

    observed = component_stream_sorted_matmul_oracle(
        component,
        sorted_x,
        sorted_experts,
        codebook=e8p_packed_abs_grid(),
    )

    expected = np.zeros((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_experts):
        for n_tile in range(packed_u16.layout.n_tiles):
            out_start = n_tile * packed_u16.layout.bn
            out_stop = min(output_dims, out_start + packed_u16.layout.bn)
            for k_block in range(packed_u16.layout.k_blocks):
                k_start = k_block * packed_u16.layout.bk
                k_stop = k_start + packed_u16.layout.bk
                tile = decode_e8p_rhs_tile(
                    packed_u16,
                    expert=int(expert),
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=e8p_packed_abs_grid(),
                )
                expected[route, out_start:out_stop] += (
                    sorted_x[route, k_start:k_stop].astype(np.float32) @ tile.T.astype(np.float32)
                )

    assert observed.shape == (routes, output_dims)
    np.testing.assert_array_equal(component.scale_group_indices[5], np.array([0, 1], dtype=np.int32))
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_pack_e8p_sign_nibble_abs_index_tiles_preserves_nibbles_and_scale_maps() -> None:
    rng = np.random.default_rng(20260705)
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)

    packed = pack_e8p_sign_nibble_abs_index_rhs_tiles(
        codes,
        scales,
        group_size=group_size,
        bn=64,
        bk=64,
    )

    assert packed.layout.target_kernel_family == "sorted_gather_qmm_rhs_nax_sign_nibble_abs_index"
    assert packed.layout.storage_constraint == "compressed_e8p_sign_nibbles_abs_index_scales"
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert packed.sign_low_nibble_tiles.shape == (2, 2, 11, 64, 8)
    assert packed.sign_low_nibble_tiles.dtype == np.uint8
    assert packed.sign_high_nibble_tiles.shape == packed.sign_low_nibble_tiles.shape
    assert packed.sign_high_nibble_tiles.dtype == np.uint8
    assert packed.abs_index_tiles.shape == packed.sign_low_nibble_tiles.shape
    assert packed.abs_index_tiles.dtype == np.uint8
    assert packed.parity_tiles.shape == packed.sign_low_nibble_tiles.shape
    assert packed.parity_tiles.dtype == np.uint8
    np.testing.assert_array_equal(packed.scale_group_indices[5], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        packed.codeword_scale_slots[5],
        np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
    )

    expected_codes = codes[1, 67, 40:48]
    expected_signs = (expected_codes & np.uint16(0xFF)).astype(np.uint8)
    np.testing.assert_array_equal(
        packed.sign_low_nibble_tiles[1, 1, 5, 3],
        expected_signs & np.uint8(0x0F),
    )
    np.testing.assert_array_equal(
        packed.sign_high_nibble_tiles[1, 1, 5, 3],
        expected_signs >> np.uint8(4),
    )
    np.testing.assert_array_equal(
        packed.abs_index_tiles[1, 1, 5, 3],
        (expected_codes >> np.uint16(8)).astype(np.uint8),
    )


def test_decode_e8p_sign_nibble_abs_index_rhs_tile_matches_split_byte_oracle() -> None:
    rng = np.random.default_rng(20260706)
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    packed_split = pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=group_size, bn=64, bk=64)
    packed_nibbles = pack_e8p_sign_nibble_abs_index_rhs_tiles(
        codes,
        scales,
        group_size=group_size,
        bn=64,
        bk=64,
    )

    observed = decode_e8p_sign_nibble_abs_index_rhs_tile(
        packed_nibbles,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )
    expected = decode_e8p_split_byte_rhs_tile(
        packed_split,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )

    assert observed.shape == (6, 64)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)


def test_pack_e8p_sign_nibble_micro_lut_tiles_exposes_fixed_nibble_and_abs_slots() -> None:
    rows = np.arange(64, dtype=np.uint16)[:, None]
    words = np.arange(8, dtype=np.uint16)[None, :]
    low_nibbles = (rows + words) & np.uint16(0x0F)
    high_nibbles = ((rows // np.uint16(4)) + words) & np.uint16(0x0F)
    signs = low_nibbles | (high_nibbles << np.uint16(4))
    abs_indices = (words % np.uint16(5)) + np.uint16(31)
    codes = (signs | (abs_indices << np.uint16(8))).reshape(1, 64, 8)
    scales = np.ones((1, 64, 1), dtype=np.float16)

    packed = pack_e8p_sign_nibble_micro_lut_rhs_tiles(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
    )

    assert packed.layout.target_kernel_family == "sorted_gather_qmm_rhs_nax_sign_nibble_micro_lut"
    assert packed.layout.storage_constraint == "compressed_e8p_sign_nibble_micro_lut_scales"
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert packed.sign_low_nibble_lut.shape == (16,)
    assert packed.sign_high_nibble_lut.shape == (16,)
    np.testing.assert_array_equal(packed.sign_low_nibble_lut, np.arange(16, dtype=np.uint8))
    np.testing.assert_array_equal(packed.sign_high_nibble_lut, np.arange(16, dtype=np.uint8))
    assert packed.sign_low_nibble_slots.shape == (1, 1, 1, 64, 8)
    assert packed.sign_high_nibble_slots.shape == packed.sign_low_nibble_slots.shape
    assert packed.sign_low_nibble_slots.dtype == np.uint8
    assert int(packed.sign_low_nibble_slots.max()) <= 15
    assert int(packed.sign_high_nibble_slots.max()) <= 15
    assert packed.abs_index_lut.shape == (1, 1, 1, 256)
    assert packed.abs_index_slots.shape == packed.sign_low_nibble_slots.shape
    assert packed.abs_index_slots.dtype == np.uint8
    assert int(packed.abs_index_counts[0, 0, 0]) == 5
    np.testing.assert_array_equal(
        packed.abs_index_lut[0, 0, 0, :5],
        np.array([31, 32, 33, 34, 35], dtype=np.uint8),
    )
    np.testing.assert_array_equal(packed.sign_low_nibble_slots[0, 0, 0], low_nibbles.astype(np.uint8))
    np.testing.assert_array_equal(packed.sign_high_nibble_slots[0, 0, 0], high_nibbles.astype(np.uint8))
    np.testing.assert_array_equal(
        packed.abs_index_lut[0, 0, 0, packed.abs_index_slots[0, 0, 0]],
        abs_indices.astype(np.uint8).repeat(64, axis=0),
    )


def test_decode_e8p_sign_nibble_micro_lut_rhs_tile_matches_sign_nibble_oracle() -> None:
    rng = np.random.default_rng(20260707)
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    packed_nibbles = pack_e8p_sign_nibble_abs_index_rhs_tiles(
        codes,
        scales,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    packed_micro = pack_e8p_sign_nibble_micro_lut_rhs_tiles(
        codes,
        scales,
        group_size=group_size,
        bn=64,
        bk=64,
    )

    np.testing.assert_array_equal(packed_micro.scale_group_indices[5], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        packed_micro.codeword_scale_slots[5],
        np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
    )
    observed = decode_e8p_sign_nibble_micro_lut_rhs_tile(
        packed_micro,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )
    expected = decode_e8p_sign_nibble_abs_index_rhs_tile(
        packed_nibbles,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )

    assert observed.shape == (6, 64)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)


def test_pack_e8p_sign_plane_abs_index_tiles_exposes_column_sign_masks() -> None:
    rows = np.arange(64, dtype=np.uint16)[:, None]
    words = np.arange(8, dtype=np.uint16)[None, :]
    signs = ((rows * np.uint16(13) + words * np.uint16(29)) ^ (rows >> np.uint16(1))) & np.uint16(0xFF)
    abs_indices = (words % np.uint16(7)) + np.uint16(41)
    codes = (signs | (abs_indices << np.uint16(8))).reshape(1, 64, 8)
    scales = np.ones((1, 64, 1), dtype=np.float16)

    packed = pack_e8p_sign_plane_abs_index_rhs_tiles(
        codes,
        scales,
        group_size=64,
        bn=64,
        bk=64,
    )

    assert packed.layout.target_kernel_family == "sorted_gather_qmm_rhs_nax_sign_plane_abs_index"
    assert packed.layout.storage_constraint == "compressed_e8p_sign_planes_abs_index_scales"
    assert packed.layout.decoded_dense_weight_bytes == 0
    assert packed.sign_bit_planes.shape == (1, 1, 1, 8, 8)
    assert packed.sign_bit_planes.dtype == np.uint64
    assert packed.abs_index_tiles.shape == (1, 1, 1, 64, 8)
    assert packed.abs_index_tiles.dtype == np.uint8

    reconstructed = np.zeros((64, 8), dtype=np.uint8)
    for word in range(8):
        for bit in range(8):
            mask = int(packed.sign_bit_planes[0, 0, 0, word, bit])
            for n in range(64):
                reconstructed[n, word] |= ((mask >> n) & 1) << bit

    np.testing.assert_array_equal(reconstructed, signs.astype(np.uint8))
    np.testing.assert_array_equal(
        packed.abs_index_tiles[0, 0, 0],
        abs_indices.astype(np.uint8).repeat(64, axis=0),
    )


def test_decode_e8p_sign_plane_abs_index_rhs_tile_matches_sign_nibble_oracle() -> None:
    rng = np.random.default_rng(20260708)
    experts = 2
    output_dims = 70
    input_dims = 704
    group_size = 352
    codes = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    packed_plane = pack_e8p_sign_plane_abs_index_rhs_tiles(
        codes,
        scales,
        group_size=group_size,
        bn=64,
        bk=64,
    )
    packed_nibbles = pack_e8p_sign_nibble_abs_index_rhs_tiles(
        codes,
        scales,
        group_size=group_size,
        bn=64,
        bk=64,
    )

    np.testing.assert_array_equal(packed_plane.scale_group_indices[5], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        packed_plane.codeword_scale_slots[5],
        np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
    )
    observed = decode_e8p_sign_plane_abs_index_rhs_tile(
        packed_plane,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )
    expected = decode_e8p_sign_nibble_abs_index_rhs_tile(
        packed_nibbles,
        expert=1,
        n_tile=1,
        k_block=5,
        codebook=e8p_packed_abs_grid(),
    )

    assert observed.shape == (6, 64)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)


def test_next_family_selector_rejects_speed_rejected_token_pair_slot_topk_output_group_stream() -> None:
    candidate = E8PNextKernelFamilyCandidate(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_token_pair_slot_topk_output_group_stream"
        ),
        storage_constraint=(
            "compressed_e8p_token_pair_slot_topk_output_group_streams"
        ),
        dispatch_grid=(
            "token_pairs_x_pair_slots_x_topk_slots_x_output_groups_x_k_blocks_x_codeword_groups"
        ),
        decode_reuse_scope=(
            "pair_slot_topk_output_group_accumulators_preserve_q2_shape_without_full_expert_axis"
        ),
        tensorops_rhs_source=(
            "compressed_e8p_codeword_tiles_streamed_by_token_pair_slot_topk_output_group"
        ),
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        decoded_dense_weight_bytes=0,
        reconstructs_e8p_values_per_fragment=False,
        uses_decoded_b_threadgroup_staging=False,
        uses_lane_local_fragment_buffer=False,
    )

    verdict = evaluate_e8p_next_kernel_family_candidate(
        candidate,
        speed_rejected_kernel_families=(
            "token_pair_slot_topk_output_group_stream",
        ),
    )

    assert verdict["decision"] == (
        "reject_token_pair_slot_topk_output_group_stream_speed_path"
    )
    assert verdict["candidate_ready_for_source_probe"] is False
    assert verdict["matched_rejected_families"] == [
        "token_pair_slot_topk_output_group_stream"
    ]
    assert "avoid_token_pair_slot_topk_output_group_stream_schedule" in (
        verdict["required_next_features"]
    )
    assert "do_not_retime_token_pair_slot_topk_output_group_stream" in (
        verdict["rejected_next_steps"]
    )
    assert verdict["speed_claim"] is False
    assert verdict["native_parity_claim"] is False
