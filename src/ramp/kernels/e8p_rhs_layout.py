from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keep.vq.e8 import CODEWORD_DIM, decode_e8p


@dataclass(frozen=True)
class E8PPackedRHSLayout:
    target_kernel_family: str
    storage_constraint: str
    experts: int
    output_dims: int
    input_dims: int
    group_size: int
    bn: int
    bk: int
    n_tiles: int
    k_blocks: int
    codewords_per_bk: int
    max_scale_groups_per_bk: int
    decoded_dense_weight_bytes: int = 0


@dataclass(frozen=True)
class E8PPackedRHSTiles:
    layout: E8PPackedRHSLayout
    code_tiles: np.ndarray
    scale_tiles: np.ndarray
    scale_group_indices: np.ndarray
    codeword_scale_slots: np.ndarray


@dataclass(frozen=True)
class E8PSplitByteRHSTiles:
    layout: E8PPackedRHSLayout
    sign_tiles: np.ndarray
    abs_index_tiles: np.ndarray
    parity_tiles: np.ndarray
    scale_tiles: np.ndarray
    scale_group_indices: np.ndarray
    codeword_scale_slots: np.ndarray


@dataclass(frozen=True)
class E8PSplitByteFactorReuseRHSTiles:
    layout: E8PPackedRHSLayout
    sign_byte_lut: np.ndarray
    sign_byte_slots: np.ndarray
    sign_byte_counts: np.ndarray
    abs_index_lut: np.ndarray
    abs_index_slots: np.ndarray
    abs_index_counts: np.ndarray
    scale_tiles: np.ndarray
    scale_group_indices: np.ndarray
    codeword_scale_slots: np.ndarray


@dataclass(frozen=True)
class E8PExpertKBlockFactorReuseRHSTiles:
    layout: E8PPackedRHSLayout
    sign_byte_lut: np.ndarray
    sign_byte_slots: np.ndarray
    sign_byte_counts: np.ndarray
    abs_index_lut: np.ndarray
    abs_index_slots: np.ndarray
    abs_index_counts: np.ndarray
    scale_tiles: np.ndarray
    scale_group_indices: np.ndarray
    codeword_scale_slots: np.ndarray


@dataclass(frozen=True)
class E8PExpertKBlockDispatchWorkgroup:
    expert: int
    k_block: int
    route_tile: int
    n_tiles: tuple[int, ...]
    sign_lut_index: tuple[int, int]
    abs_index_lut_index: tuple[int, int]
    sign_count: int
    abs_index_count: int
    scale_group_indices: tuple[int, ...]
    codeword_scale_slots: tuple[int, ...]


@dataclass(frozen=True)
class E8PExpertKBlockDispatchContract:
    target_kernel_family: str
    storage_constraint: str
    dispatch_grid: str
    incompatible_primitive_grid: str
    reuse_scope: str
    slot_layout: str
    route_tile_count: int
    k_blocks: int
    n_tiles: int
    output_tile_size: int
    decoded_dense_weight_bytes: int
    required_next_features: tuple[str, ...]
    workgroups: tuple[E8PExpertKBlockDispatchWorkgroup, ...]


@dataclass(frozen=True)
class E8PExpertKBlockPartialReductionContract:
    target_kernel_family: str
    partial_accumulation_grid: str
    final_reduction_grid: str
    partial_accumulation_shape: tuple[int, int, int, int, int]
    partial_accumulation_dtype: str
    partial_accumulation_element_count: int
    partial_accumulation_bytes: int
    route_tile_size: int
    reduction_axis: str
    output_shape: str
    valid_routes_source: str
    expert_source: str
    decoded_dense_weight_bytes: int
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PExpertKBlockDecodeReuseFeasibility:
    target_kernel_family: str
    decision: str
    output_tile_reuse_feasible_without_decoded_cache: bool
    factor_lut_shared_across_output_tiles: bool
    slot_values_depend_on_output_column: bool
    scale_values_depend_on_output_column: bool
    correct_cache_shape: tuple[int, int, int, int, int]
    correct_cache_dtype: str
    correct_cache_bytes: int
    correct_cache_is_decoded_rhs_equivalent: bool
    structural_risk: str
    next_track_b_hypothesis: str
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PRouteCodewordProjectionFeasibility:
    target_kernel_family: str
    decision: str
    storage_constraint: str
    dispatch_grid: str
    activation_projection_scope: str
    experts: int
    output_dims: int
    input_dims: int
    group_size: int
    bn: int
    bk: int
    n_tiles: int
    k_blocks: int
    codewords_per_bk: int
    route_tile_count: int
    route_tile_size: int
    unique_codeword_counts_shape: tuple[int, int]
    min_unique_codewords_per_expert_kblock: int
    median_unique_codewords_per_expert_kblock: float
    p95_unique_codewords_per_expert_kblock: float
    max_unique_codewords_per_expert_kblock: int
    max_unique_ratio_per_expert_kblock: float
    unique_codeword_cap: int
    projection_cache_dtype: str
    projection_cache_shape: tuple[int, int, int]
    projection_cache_bytes: int
    projection_cache_byte_cap: int
    decoded_dense_weight_bytes: int
    preserves_compressed_rhs_storage: bool
    preserves_codeword_scale_slots: bool
    candidate_ready_for_source_probe: bool
    structural_risk: str
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PRouteActiveCodewordProjectionFeasibility:
    target_kernel_family: str
    decision: str
    storage_constraint: str
    dispatch_grid: str
    activation_projection_scope: str
    experts: int
    output_dims: int
    input_dims: int
    group_size: int
    bn: int
    bk: int
    n_tiles: int
    k_blocks: int
    codewords_per_bk: int
    route_count: int
    route_tile_size: int
    active_expert_count: int
    active_route_tile_count: int
    active_route_tile_counts_shape: tuple[int]
    active_route_tile_counts: tuple[int, ...]
    unique_codeword_counts_shape: tuple[int, int]
    min_unique_codewords_per_active_expert_kblock: int
    median_unique_codewords_per_active_expert_kblock: float
    max_unique_codewords_per_active_expert_kblock: int
    unique_codeword_cap: int
    projection_cache_dtype: str
    active_projection_cache_elements: int
    active_projection_cache_bytes: int
    global_max_projection_cache_bytes: int
    projection_cache_byte_cap: int
    decoded_dense_weight_bytes: int
    preserves_compressed_rhs_storage: bool
    preserves_codeword_scale_slots: bool
    candidate_ready_for_source_probe: bool
    structural_risk: str
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PRouteAbsIndexProjectionFeasibility:
    target_kernel_family: str
    decision: str
    storage_constraint: str
    dispatch_grid: str
    activation_projection_scope: str
    experts: int
    output_dims: int
    input_dims: int
    group_size: int
    bn: int
    bk: int
    n_tiles: int
    k_blocks: int
    codewords_per_bk: int
    route_tile_count: int
    route_tile_size: int
    abs_index_counts_shape: tuple[int, int]
    signed_abs_pair_counts_shape: tuple[int, int]
    min_abs_indices_per_expert_kblock: int
    median_abs_indices_per_expert_kblock: float
    max_abs_indices_per_expert_kblock: int
    min_signed_abs_pairs_per_expert_kblock: int
    median_signed_abs_pairs_per_expert_kblock: float
    max_signed_abs_pairs_per_expert_kblock: int
    projection_cache_dtype: str
    abs_projection_cache_shape: tuple[int, int, int]
    abs_projection_cache_bytes: int
    correct_signed_projection_cache_shape: tuple[int, int, int]
    correct_signed_projection_cache_bytes: int
    projection_cache_byte_cap: int
    abs_only_projection_exact: bool
    sign_values_depend_on_output_column: bool
    correct_cache_is_full_codeword_equivalent: bool
    decoded_dense_weight_bytes: int
    preserves_compressed_rhs_storage: bool
    preserves_codeword_scale_slots: bool
    candidate_ready_for_source_probe: bool
    structural_risk: str
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PRouteAbsComponentProjectionFeasibility:
    target_kernel_family: str
    decision: str
    storage_constraint: str
    dispatch_grid: str
    activation_projection_scope: str
    experts: int
    output_dims: int
    input_dims: int
    group_size: int
    bn: int
    bk: int
    n_tiles: int
    k_blocks: int
    codewords_per_bk: int
    route_tile_count: int
    route_tile_size: int
    abs_index_counts_shape: tuple[int, int]
    full_codeword_counts_shape: tuple[int, int]
    min_abs_indices_per_expert_kblock: int
    median_abs_indices_per_expert_kblock: float
    max_abs_indices_per_expert_kblock: int
    max_full_codewords_per_expert_kblock: int
    abs_index_cap: int
    component_axis: int
    projection_cache_dtype: str
    projection_cache_shape: tuple[int, int, int, int]
    projection_cache_bytes: int
    full_codeword_cache_bytes: int
    projection_cache_byte_cap: int
    abs_component_projection_exact: bool
    sign_values_apply_after_cache_lookup: bool
    scale_values_apply_after_cache_lookup: bool
    decoded_dense_weight_bytes: int
    preserves_compressed_rhs_storage: bool
    preserves_codeword_scale_slots: bool
    candidate_ready_for_source_probe: bool
    structural_risk: str
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PRouteAbsComponentTokenReuseFeasibility:
    target_kernel_family: str
    decision: str
    storage_constraint: str
    dispatch_grid: str
    activation_projection_scope: str
    experts: int
    output_dims: int
    input_dims: int
    group_size: int
    bn: int
    bk: int
    n_tiles: int
    k_blocks: int
    codewords_per_bk: int
    token_count: int
    top_k: int
    route_count: int
    route_slot_reuse_factor: int
    abs_index_counts_shape: tuple[int, int]
    min_abs_indices_per_expert_kblock: int
    median_abs_indices_per_expert_kblock: float
    max_abs_indices_per_expert_kblock: int
    abs_index_cap: int
    component_axis: int
    projection_cache_dtype: str
    token_projection_cache_shape: tuple[int, int, int, int]
    token_projection_cache_bytes: int
    rowwise_projection_cache_shape: tuple[int, int, int, int]
    rowwise_projection_cache_bytes: int
    cache_reduction_vs_rowwise: float
    projection_cache_byte_cap: int
    token_reuse_exact_for_topk_routes: bool
    route_slot_axis_preserved_by_indirection: bool
    decoded_dense_weight_bytes: int
    preserves_compressed_rhs_storage: bool
    preserves_codeword_scale_slots: bool
    candidate_ready_for_source_probe: bool
    structural_risk: str
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PRouteAbsComponentTokenActiveExpertFeasibility:
    target_kernel_family: str
    decision: str
    storage_constraint: str
    dispatch_grid: str
    activation_projection_scope: str
    experts: int
    output_dims: int
    input_dims: int
    group_size: int
    bn: int
    bk: int
    n_tiles: int
    k_blocks: int
    codewords_per_bk: int
    token_count: int
    top_k: int
    route_count: int
    active_expert_ids_shape: tuple[int, int]
    token_kblock_abs_index_count_shape: tuple[int, int]
    min_abs_indices_per_token_kblock: int
    median_abs_indices_per_token_kblock: float
    p95_abs_indices_per_token_kblock: float
    max_abs_indices_per_token_kblock: int
    abs_index_cap: int
    component_axis: int
    projection_cache_dtype: str
    ragged_token_active_projection_cache_elements: int
    ragged_token_active_projection_cache_bytes: int
    max_token_active_projection_cache_shape: tuple[int, int, int, int]
    max_token_active_projection_cache_bytes: int
    full_token_projection_cache_shape: tuple[int, int, int, int]
    full_token_projection_cache_bytes: int
    rowwise_projection_cache_shape: tuple[int, int, int, int]
    rowwise_projection_cache_bytes: int
    cache_reduction_vs_full_token_cache: float
    cache_reduction_vs_rowwise: float
    projection_cache_byte_cap: int
    token_active_expert_reuse_exact_for_topk_routes: bool
    route_slot_axis_preserved_by_indirection: bool
    active_expert_abs_union_scoped_to_token_topk: bool
    decoded_dense_weight_bytes: int
    preserves_compressed_rhs_storage: bool
    preserves_codeword_scale_slots: bool
    candidate_ready_for_source_probe: bool
    structural_risk: str
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PRouteAbsComponentLowerBoundFeasibility:
    target_kernel_family: str
    decision: str
    storage_constraint: str
    lower_bound_scope: str
    families_blocked_by_lower_bound: tuple[str, ...]
    experts: int
    output_dims: int
    input_dims: int
    group_size: int
    bn: int
    bk: int
    n_tiles: int
    k_blocks: int
    codewords_per_bk: int
    token_count: int
    top_k: int
    route_count: int
    token_kblock_abs_index_count_shape: tuple[int, int]
    min_abs_indices_per_token_kblock: int
    median_abs_indices_per_token_kblock: float
    p95_abs_indices_per_token_kblock: float
    max_abs_indices_per_token_kblock: int
    abs_index_cap: int
    component_axis: int
    projection_cache_dtype: str
    exact_lower_bound_cache_elements: int
    exact_lower_bound_cache_bytes: int
    projection_cache_byte_cap: int
    lower_bound_ratio_to_cap: float
    rowwise_projection_cache_bytes: int
    reduction_vs_rowwise: float
    lower_bound_preserves_route_slot_exactness: bool
    lower_bound_preserves_e8p_sign_axis: bool
    lower_bound_requires_no_decoded_rhs_storage: bool
    decoded_dense_weight_bytes: int
    preserves_compressed_rhs_storage: bool
    preserves_codeword_scale_slots: bool
    candidate_ready_for_source_probe: bool
    structural_risk: str
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PComponentStreamRHSTiles:
    layout: E8PPackedRHSLayout
    sign_component_bits: np.ndarray
    abs_index_tiles: np.ndarray
    scale_tiles: np.ndarray
    scale_group_indices: np.ndarray
    codeword_scale_slots: np.ndarray
    component_scale_slots: np.ndarray
    component_codeword_indices: np.ndarray
    component_offsets: np.ndarray


@dataclass(frozen=True)
class E8PComponentStreamWorkgroup:
    route_tile: int
    expert: int
    k_block: int
    component_pair: int
    n_tiles: tuple[int, ...]
    component_offsets: tuple[int, ...]
    codeword_indices: tuple[int, ...]
    codeword_component_offsets: tuple[int, ...]
    scale_slots: tuple[int, ...]


@dataclass(frozen=True)
class E8PComponentStreamContract:
    target_kernel_family: str
    storage_constraint: str
    dispatch_grid: str
    decode_reuse_scope: str
    rhs_source: str
    route_tile_count: int
    k_blocks: int
    n_tiles: int
    component_count_per_kblock: int
    component_pair_size: int
    component_pair_count: int
    decoded_dense_weight_bytes: int
    required_before_native_parity: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]
    workgroups: tuple[E8PComponentStreamWorkgroup, ...]


@dataclass(frozen=True)
class E8PComponentStreamPartialReductionContract:
    target_kernel_family: str
    partial_accumulation_grid: str
    final_reduction_grid: str
    partial_accumulation_shape: tuple[int, int, int, int, int, int]
    partial_accumulation_dtype: str
    partial_accumulation_element_count: int
    partial_accumulation_bytes: int
    route_tile_size: int
    reduction_axes: tuple[str, ...]
    output_shape: str
    valid_routes_source: str
    expert_source: str
    decoded_dense_weight_bytes: int
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PComponentStreamSharedDecodeContract:
    target_kernel_family: str
    dispatch_grid: str
    decode_reuse_scope: str
    shared_decode_shape: tuple[int, int, int, int, int, int]
    shared_decode_dtype: str
    shared_decode_bytes: int
    output_accumulation_grid: str
    decoded_dense_weight_bytes: int
    preserved_storage_contract: str
    required_next_features: tuple[str, ...]
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PComponentStreamSharedDecodeFeasibility:
    target_kernel_family: str
    decision: str
    n_tile_reuse_feasible: bool
    component_values_depend_on_output_column: bool
    scale_values_depend_on_output_column: bool
    correct_cache_shape: tuple[int, int, int, int, int]
    correct_cache_dtype: str
    correct_cache_bytes: int
    correct_cache_is_decoded_rhs_equivalent: bool
    structural_risk: str
    next_track_b_hypothesis: str
    rejected_next_steps: tuple[str, ...]


@dataclass(frozen=True)
class E8PNextKernelFamilyCandidate:
    target_kernel_family: str
    storage_constraint: str
    dispatch_grid: str
    decode_reuse_scope: str
    tensorops_rhs_source: str
    preserves_compressed_rhs_storage: bool
    preserves_codeword_scale_slots: bool
    decoded_dense_weight_bytes: int
    reconstructs_e8p_values_per_fragment: bool = False
    uses_decoded_b_threadgroup_staging: bool = False
    uses_lane_local_fragment_buffer: bool = False


@dataclass(frozen=True)
class E8PSignNibbleAbsIndexRHSTiles:
    layout: E8PPackedRHSLayout
    sign_low_nibble_tiles: np.ndarray
    sign_high_nibble_tiles: np.ndarray
    abs_index_tiles: np.ndarray
    parity_tiles: np.ndarray
    scale_tiles: np.ndarray
    scale_group_indices: np.ndarray
    codeword_scale_slots: np.ndarray


@dataclass(frozen=True)
class E8PSignNibbleMicroLUTRHSTiles:
    layout: E8PPackedRHSLayout
    sign_low_nibble_lut: np.ndarray
    sign_low_nibble_slots: np.ndarray
    sign_high_nibble_lut: np.ndarray
    sign_high_nibble_slots: np.ndarray
    abs_index_lut: np.ndarray
    abs_index_slots: np.ndarray
    abs_index_counts: np.ndarray
    scale_tiles: np.ndarray
    scale_group_indices: np.ndarray
    codeword_scale_slots: np.ndarray


@dataclass(frozen=True)
class E8PSignPlaneAbsIndexRHSTiles:
    layout: E8PPackedRHSLayout
    sign_bit_planes: np.ndarray
    abs_index_tiles: np.ndarray
    scale_tiles: np.ndarray
    scale_group_indices: np.ndarray
    codeword_scale_slots: np.ndarray


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_e8p_inputs(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int,
    bk: int,
) -> tuple[int, int, int]:
    if group_size <= 0 or bn <= 0 or bk <= 0:
        raise ValueError("group_size, bn, and bk must be positive")
    if bk % CODEWORD_DIM != 0:
        raise ValueError("bk must be divisible by 8")
    if codes.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, output, input/8], found {codes.shape}")
    if scales.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, output, input/group], found {scales.shape}")
    if codes.dtype != np.uint16:
        raise ValueError(f"E8P codes must be uint16, found {codes.dtype}")
    experts, output_dims, codewords = codes.shape
    input_dims = codewords * CODEWORD_DIM
    expected_scales = (experts, output_dims, input_dims // group_size)
    if input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by group_size")
    if scales.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales.shape}")
    if input_dims % bk != 0:
        raise ValueError("input_dims must be divisible by bk")
    return experts, output_dims, input_dims


def _scale_maps(*, input_dims: int, group_size: int, bk: int) -> tuple[np.ndarray, np.ndarray, int]:
    codewords_per_bk = bk // CODEWORD_DIM
    k_blocks = input_dims // bk
    per_block_groups: list[list[int]] = []
    per_block_slots = np.zeros((k_blocks, codewords_per_bk), dtype=np.int32)
    max_groups = 0
    for k_block in range(k_blocks):
        groups: list[int] = []
        for word in range(codewords_per_bk):
            absolute_k = k_block * bk + word * CODEWORD_DIM
            group = absolute_k // group_size
            if group not in groups:
                groups.append(group)
            per_block_slots[k_block, word] = groups.index(group)
        max_groups = max(max_groups, len(groups))
        per_block_groups.append(groups)

    group_indices = np.full((k_blocks, max_groups), -1, dtype=np.int32)
    for k_block, groups in enumerate(per_block_groups):
        group_indices[k_block, : len(groups)] = np.asarray(groups, dtype=np.int32)
    return group_indices, per_block_slots, max_groups


def _sign_parity8(signs: np.ndarray) -> np.ndarray:
    signs_u16 = np.asarray(signs, dtype=np.uint16)
    parity = np.zeros(signs_u16.shape, dtype=np.uint16)
    for bit in range(8):
        parity ^= (signs_u16 >> np.uint16(bit)) & np.uint16(1)
    return parity.astype(np.uint8)


def _factor_lut_and_slots(factor_tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    experts, n_tiles, k_blocks, bn, codewords_per_bk = factor_tiles.shape
    lut = np.zeros((experts, n_tiles, k_blocks, 256), dtype=np.uint8)
    slots = np.zeros((experts, n_tiles, k_blocks, bn, codewords_per_bk), dtype=np.uint8)
    counts = np.zeros((experts, n_tiles, k_blocks), dtype=np.int16)
    for expert in range(experts):
        for n_tile in range(n_tiles):
            for k_block in range(k_blocks):
                tile = factor_tiles[expert, n_tile, k_block]
                unique, inverse = np.unique(tile.reshape(-1), return_inverse=True)
                if unique.size > 256:
                    raise ValueError("split-byte factor tiles cannot contain more than 256 unique values")
                lut[expert, n_tile, k_block, : unique.size] = unique.astype(np.uint8)
                slots[expert, n_tile, k_block] = inverse.reshape(tile.shape).astype(np.uint8)
                counts[expert, n_tile, k_block] = int(unique.size)
    return lut, slots, counts


def _expert_kblock_factor_lut_and_slots(
    factor_tiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    experts, n_tiles, k_blocks, bn, codewords_per_bk = factor_tiles.shape
    lut = np.zeros((experts, k_blocks, 256), dtype=np.uint8)
    slots = np.zeros((experts, n_tiles, k_blocks, bn, codewords_per_bk), dtype=np.uint8)
    counts = np.zeros((experts, k_blocks), dtype=np.int16)
    for expert in range(experts):
        for k_block in range(k_blocks):
            block = factor_tiles[expert, :, k_block]
            unique, inverse = np.unique(block.reshape(-1), return_inverse=True)
            if unique.size > 256:
                raise ValueError("expert/K-block factor groups cannot contain more than 256 unique values")
            lut[expert, k_block, : unique.size] = unique.astype(np.uint8)
            slots[expert, :, k_block] = inverse.reshape(block.shape).astype(np.uint8)
            counts[expert, k_block] = int(unique.size)
    return lut, slots, counts


def _sign_bit_planes_from_sign_tiles(sign_tiles: np.ndarray) -> np.ndarray:
    experts, n_tiles, k_blocks, bn, codewords_per_bk = sign_tiles.shape
    if bn > 64:
        raise ValueError("sign-plane layout requires bn <= 64")
    planes = np.zeros((experts, n_tiles, k_blocks, codewords_per_bk, 8), dtype=np.uint64)
    signs_u64 = sign_tiles.astype(np.uint64, copy=False)
    for n in range(bn):
        row_signs = signs_u64[:, :, :, n, :]
        row_mask = np.uint64(1) << np.uint64(n)
        for bit in range(8):
            planes[:, :, :, :, bit] |= ((row_signs >> np.uint64(bit)) & np.uint64(1)) * row_mask
    return planes


def _matched_rejected_kernel_families(
    candidate: E8PNextKernelFamilyCandidate,
) -> tuple[str, ...]:
    matches: list[str] = []
    family = candidate.target_kernel_family
    storage = candidate.storage_constraint
    if (
        "expert_kblock_factor_reuse" in family
        or storage == "compressed_e8p_expert_kblock_factor_lut_scales"
    ):
        matches.append("expert_kblock_factor_reuse")
    if (
        "sign_nibble" in family
        or storage
        in {
            "compressed_e8p_sign_nibbles_abs_index_scales",
            "compressed_e8p_sign_nibble_micro_lut_scales",
        }
    ):
        matches.append("sign_nibble_factor_reuse")
    if "sign_plane" in family or storage == "compressed_e8p_sign_planes_abs_index_scales":
        matches.append("sign_plane_factor_reuse")
    if (
        "split_byte" in family
        or storage == "compressed_e8p_split_byte_factor_lut_scales"
    ):
        matches.append("split_byte_factor_reuse")
    if "packed_rhs" in family or storage == "compressed_e8p_codes_scales":
        matches.append("packed_rhs_tiled")
    if (
        "component_stream_shared_decode" in family
        or "shared_across_n_tiles" in candidate.decode_reuse_scope
    ):
        matches.append("component_stream_shared_decode")
    if (
        "route_abs_index_projection" in family
        or storage == "compressed_e8p_route_abs_index_projection_scales"
    ):
        matches.append("route_abs_index_projection")
    if (
        "route_abs_component_projection" in family
        or storage
        in {
            "compressed_e8p_route_abs_component_projection_scales",
            "compressed_e8p_route_abs_component_exact_lower_bound",
        }
    ):
        matches.append("route_abs_component_projection")
    if (
        "route_slot_codeword_stream" in family
        or storage == "compressed_e8p_route_slot_codeword_stream_scales"
    ):
        matches.append("route_slot_codeword_stream")
    if (
        "route_slot_mma_codeword_tile" in family
        or storage == "compressed_e8p_route_slot_mma_codeword_tiles"
    ):
        matches.append("route_slot_mma_codeword_tile")
    if (
        "active_route_tile_codeword_outer_product" in family
        or storage == "compressed_e8p_active_route_tile_codeword_outer_products"
    ):
        matches.append("active_route_tile_codeword_outer_product")
    if (
        "expert_cohort_codeword_broadcast" in family
        or storage == "compressed_e8p_expert_cohort_codeword_broadcasts"
    ):
        matches.append("expert_cohort_codeword_broadcast")
    if (
        "token_cohort_codeword_stream" in family
        or storage == "compressed_e8p_token_cohort_codeword_stream_scales"
    ):
        matches.append("token_cohort_codeword_stream")
    if (
        "token_cohort_mma_codeword_tile" in family
        or storage == "compressed_e8p_token_cohort_mma_codeword_tiles"
    ):
        matches.append("token_cohort_mma_codeword_tile")
    if (
        "output_stationary_codeword_tile" in family
        or storage == "compressed_e8p_output_stationary_codeword_tiles"
    ):
        matches.append("output_stationary_codeword_tile")
    if (
        "input_stationary_codeword_tile" in family
        or storage == "compressed_e8p_input_stationary_codeword_tiles"
    ):
        matches.append("input_stationary_codeword_tile")
    if (
        "route_codeword_lut_accumulate" in family
        or storage == "compressed_e8p_route_codeword_lut_accumulate_scales"
    ):
        matches.append("route_codeword_lut_accumulate")
    if (
        "rowwise_codeword_tile_accumulate" in family
        or storage == "compressed_e8p_rowwise_codeword_tile_accumulate_scales"
    ):
        matches.append("rowwise_codeword_tile_accumulate")
    if (
        "output_tile_local_codeword_lut" in family
        or storage == "compressed_e8p_output_tile_local_codeword_lut_scales"
    ):
        matches.append("output_tile_local_codeword_lut")
    if (
        "kblock_wavefront_codeword_scan" in family
        or storage == "compressed_e8p_kblock_wavefront_codeword_scans"
    ):
        matches.append("kblock_wavefront_codeword_scan")
    if (
        "token_route_output_stripe_pipeline" in family
        or storage == "compressed_e8p_token_route_output_stripe_pipelines"
    ):
        matches.append("token_route_output_stripe_pipeline")
    if (
        "scale_group_route_block_reduce" in family
        or storage == "compressed_e8p_scale_group_route_block_reductions"
    ):
        matches.append("scale_group_route_block_reduce")
    return tuple(dict.fromkeys(matches))


def evaluate_e8p_next_kernel_family_candidate(
    candidate: E8PNextKernelFamilyCandidate,
    *,
    speed_rejected_kernel_families: tuple[str, ...] = (),
) -> dict[str, object]:
    """Gate Track B proposals before another near-duplicate timing run.

    This is intentionally a source-structure guardrail, not a performance or
    native-correctness claim. A candidate may proceed only when it preserves the
    compressed E8P contract while avoiding every rejected RHS decode family.
    """

    matched_list = list(_matched_rejected_kernel_families(candidate))
    if (
        "expert_kblock_scale_slot_stream" in speed_rejected_kernel_families
        and (
            "expert_kblock_scale_slot_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_expert_kblock_scale_slot_streams"
        )
    ):
        matched_list.append("expert_kblock_scale_slot_stream")
    if (
        "route_block_output_group_stream" in speed_rejected_kernel_families
        and (
            "route_block_output_group_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_route_block_output_group_streams"
        )
    ):
        matched_list.append("route_block_output_group_stream")
    if (
        "output_group_pretransposed_codeword_stream" in speed_rejected_kernel_families
        and (
            "output_group_pretransposed_codeword_stream"
            in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_output_group_pretransposed_codeword_streams"
        )
    ):
        matched_list.append("output_group_pretransposed_codeword_stream")
    if (
        "kblock_output_group_route_fused_stream" in speed_rejected_kernel_families
        and (
            "kblock_output_group_route_fused_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_kblock_output_group_route_fused_streams"
        )
    ):
        matched_list.append("kblock_output_group_route_fused_stream")
    if (
        "route_tile_output_swizzle_stream" in speed_rejected_kernel_families
        and (
            "route_tile_output_swizzle_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_route_tile_output_swizzle_streams"
        )
    ):
        matched_list.append("route_tile_output_swizzle_stream")
    if (
        "token_topk_output_tile_stream" in speed_rejected_kernel_families
        and (
            "token_topk_output_tile_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_topk_output_tile_streams"
        )
    ):
        matched_list.append("token_topk_output_tile_stream")
    if (
        "token_block_output_group_stream" in speed_rejected_kernel_families
        and (
            "token_block_output_group_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_block_output_group_streams"
        )
    ):
        matched_list.append("token_block_output_group_stream")
    if (
        "token_output_stripe_group_stream" in speed_rejected_kernel_families
        and (
            "token_output_stripe_group_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_output_stripe_group_streams"
        )
    ):
        matched_list.append("token_output_stripe_group_stream")
    if (
        "token_expert_output_block_stream" in speed_rejected_kernel_families
        and (
            "token_expert_output_block_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_expert_output_block_streams"
        )
    ):
        matched_list.append("token_expert_output_block_stream")
    if (
        "token_pair_kblock_accumulator_stream" in speed_rejected_kernel_families
        and (
            "token_pair_kblock_accumulator_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_pair_kblock_accumulator_streams"
        )
    ):
        matched_list.append("token_pair_kblock_accumulator_stream")
    if (
        "token_pair_output_group_stream" in speed_rejected_kernel_families
        and (
            "token_pair_output_group_stream" in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_pair_output_group_streams"
        )
    ):
        matched_list.append("token_pair_output_group_stream")
    if (
        "token_pair_slot_topk_output_group_stream" in speed_rejected_kernel_families
        and (
            "token_pair_slot_topk_output_group_stream"
            in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_pair_slot_topk_output_group_streams"
        )
    ):
        matched_list.append("token_pair_slot_topk_output_group_stream")
    if (
        "token_pair_slot_topk_codeword_group_pipeline" in speed_rejected_kernel_families
        and (
            "token_pair_slot_topk_codeword_group_pipeline"
            in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_pair_slot_topk_codeword_group_pipelines"
        )
    ):
        matched_list.append("token_pair_slot_topk_codeword_group_pipeline")
    if (
        "token_pair_slot_topk_scale_slot_broadcast_stream"
        in speed_rejected_kernel_families
        and (
            "token_pair_slot_topk_scale_slot_broadcast_stream"
            in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_pair_slot_topk_scale_slot_broadcast_streams"
        )
    ):
        matched_list.append("token_pair_slot_topk_scale_slot_broadcast_stream")
    if (
        "token_pair_slot_topk_route_bucket_codeword_reduce"
        in speed_rejected_kernel_families
        and (
            "token_pair_slot_topk_route_bucket_codeword_reduce"
            in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_pair_slot_topk_route_bucket_codeword_reductions"
        )
    ):
        matched_list.append("token_pair_slot_topk_route_bucket_codeword_reduce")
    if (
        "token_pair_slot_topk_kblock_microtile_stream"
        in speed_rejected_kernel_families
        and (
            "token_pair_slot_topk_kblock_microtile_stream"
            in candidate.target_kernel_family
            or candidate.storage_constraint
            == "compressed_e8p_token_pair_slot_topk_kblock_microtile_streams"
        )
    ):
        matched_list.append("token_pair_slot_topk_kblock_microtile_stream")
    matched = tuple(dict.fromkeys(matched_list))
    structural_reasons: list[str] = []
    if matched:
        structural_reasons.append("candidate overlaps rejected Track B RHS family")
    invalid_component_stream_shared_decode = "component_stream_shared_decode" in matched
    invalid_expert_kblock_predecode_cache = (
        "expert_kblock_factor_reuse" in matched
        and (
            "predecode" in candidate.decode_reuse_scope
            or "decode_cache" in candidate.decode_reuse_scope
            or "decode_cache" in candidate.tensorops_rhs_source
            or "predecoded" in candidate.tensorops_rhs_source
        )
    )
    invalid_route_abs_index_projection = "route_abs_index_projection" in matched
    invalid_route_abs_component_projection = "route_abs_component_projection" in matched
    invalid_route_slot_codeword_stream = "route_slot_codeword_stream" in matched
    invalid_route_slot_mma_codeword_tile = "route_slot_mma_codeword_tile" in matched
    invalid_active_route_tile_codeword_outer_product = (
        "active_route_tile_codeword_outer_product" in matched
    )
    invalid_expert_cohort_codeword_broadcast = (
        "expert_cohort_codeword_broadcast" in matched
    )
    invalid_token_cohort_codeword_stream = "token_cohort_codeword_stream" in matched
    invalid_token_cohort_mma_codeword_tile = (
        "token_cohort_mma_codeword_tile" in matched
    )
    invalid_output_stationary_codeword_tile = (
        "output_stationary_codeword_tile" in matched
    )
    invalid_input_stationary_codeword_tile = (
        "input_stationary_codeword_tile" in matched
    )
    invalid_route_codeword_lut_accumulate = "route_codeword_lut_accumulate" in matched
    invalid_rowwise_codeword_tile_accumulate = (
        "rowwise_codeword_tile_accumulate" in matched
    )
    invalid_output_tile_local_codeword_lut = "output_tile_local_codeword_lut" in matched
    invalid_kblock_wavefront_codeword_scan = (
        "kblock_wavefront_codeword_scan" in matched
    )
    invalid_token_route_output_stripe_pipeline = (
        "token_route_output_stripe_pipeline" in matched
    )
    invalid_expert_kblock_scale_slot_stream = (
        "expert_kblock_scale_slot_stream" in matched
    )
    invalid_scale_group_route_block_reduce = (
        "scale_group_route_block_reduce" in matched
    )
    invalid_route_block_output_group_stream = (
        "route_block_output_group_stream" in matched
    )
    invalid_output_group_pretransposed_codeword_stream = (
        "output_group_pretransposed_codeword_stream" in matched
    )
    invalid_kblock_output_group_route_fused_stream = (
        "kblock_output_group_route_fused_stream" in matched
    )
    invalid_route_tile_output_swizzle_stream = (
        "route_tile_output_swizzle_stream" in matched
    )
    invalid_token_topk_output_tile_stream = (
        "token_topk_output_tile_stream" in matched
    )
    invalid_token_block_output_group_stream = (
        "token_block_output_group_stream" in matched
    )
    invalid_token_output_stripe_group_stream = (
        "token_output_stripe_group_stream" in matched
    )
    invalid_token_expert_output_block_stream = (
        "token_expert_output_block_stream" in matched
    )
    invalid_token_pair_kblock_accumulator_stream = (
        "token_pair_kblock_accumulator_stream" in matched
    )
    invalid_token_pair_output_group_stream = (
        "token_pair_output_group_stream" in matched
    )
    invalid_token_pair_slot_topk_output_group_stream = (
        "token_pair_slot_topk_output_group_stream" in matched
    )
    invalid_token_pair_slot_topk_codeword_group_pipeline = (
        "token_pair_slot_topk_codeword_group_pipeline" in matched
    )
    invalid_token_pair_slot_topk_scale_slot_broadcast_stream = (
        "token_pair_slot_topk_scale_slot_broadcast_stream" in matched
    )
    invalid_token_pair_slot_topk_route_bucket_codeword_reduce = (
        "token_pair_slot_topk_route_bucket_codeword_reduce" in matched
    )
    invalid_token_pair_slot_topk_kblock_microtile_stream = (
        "token_pair_slot_topk_kblock_microtile_stream" in matched
    )
    if invalid_component_stream_shared_decode:
        structural_reasons.append(
            "candidate depends on invalid component-stream cross-N-tile decode reuse"
        )
    if invalid_expert_kblock_predecode_cache:
        structural_reasons.append(
            "candidate depends on decoded-RHS-equivalent expert/K-block predecode cache"
        )
    if invalid_route_abs_index_projection:
        structural_reasons.append("candidate drops the E8P sign axis from route projection")
    if invalid_route_abs_component_projection:
        structural_reasons.append(
            "candidate is closed by exact abs-component route-projection lower bound"
        )
    if invalid_route_slot_codeword_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected scalar route-slot codeword stream"
        )
    if invalid_route_slot_mma_codeword_tile:
        structural_reasons.append(
            "candidate reopens the speed-rejected route-slot MMA codeword-tile schedule"
        )
    if invalid_active_route_tile_codeword_outer_product:
        structural_reasons.append(
            "candidate reopens the speed-rejected active-route-tile codeword outer-product schedule"
        )
    if invalid_expert_cohort_codeword_broadcast:
        structural_reasons.append(
            "candidate reopens the speed-rejected expert-cohort codeword-broadcast schedule"
        )
    if invalid_token_cohort_codeword_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-cohort codeword-stream schedule"
        )
    if invalid_token_cohort_mma_codeword_tile:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-cohort MMA codeword-tile schedule"
        )
    if invalid_output_stationary_codeword_tile:
        structural_reasons.append(
            "candidate reopens the speed-rejected output-stationary codeword-tile schedule"
        )
    if invalid_input_stationary_codeword_tile:
        structural_reasons.append(
            "candidate reopens the speed-rejected input-stationary codeword-tile schedule"
        )
    if invalid_route_codeword_lut_accumulate:
        structural_reasons.append(
            "candidate reopens the speed-rejected route-codeword LUT accumulate schedule"
        )
    if invalid_rowwise_codeword_tile_accumulate:
        structural_reasons.append(
            "candidate reopens the speed-rejected rowwise codeword-tile accumulate schedule"
        )
    if invalid_output_tile_local_codeword_lut:
        structural_reasons.append(
            "candidate reopens the speed-rejected output-tile-local codeword LUT schedule"
        )
    if invalid_kblock_wavefront_codeword_scan:
        structural_reasons.append(
            "candidate reopens the speed-rejected k-block wavefront codeword-scan schedule"
        )
    if invalid_token_route_output_stripe_pipeline:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-route output-stripe pipeline schedule"
        )
    if invalid_expert_kblock_scale_slot_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected expert/K-block scale-slot stream schedule"
        )
    if invalid_scale_group_route_block_reduce:
        structural_reasons.append(
            "candidate reopens the speed-rejected scale-group route-block reduce schedule"
        )
    if invalid_route_block_output_group_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected route-block output-group stream schedule"
        )
    if invalid_output_group_pretransposed_codeword_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected output-group-pretransposed codeword-stream schedule"
        )
    if invalid_kblock_output_group_route_fused_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected K-block output-group route-fused stream schedule"
        )
    if invalid_route_tile_output_swizzle_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected route-tile output-swizzle stream schedule"
        )
    if invalid_token_topk_output_tile_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-topk output-tile stream schedule"
        )
    if invalid_token_block_output_group_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-block output-group stream schedule"
        )
    if invalid_token_output_stripe_group_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-output-stripe group-stream schedule"
        )
    if candidate.reconstructs_e8p_values_per_fragment:
        structural_reasons.append("candidate reconstructs E8P values per TensorOps fragment")
    if invalid_token_pair_kblock_accumulator_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-pair K-block accumulator schedule"
        )
    if invalid_token_pair_output_group_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-pair output-group stream schedule"
        )
    if invalid_token_pair_slot_topk_output_group_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-pair slot/top-k output-group stream schedule"
        )
    if invalid_token_pair_slot_topk_codeword_group_pipeline:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-pair slot/top-k codeword-group pipeline schedule"
        )
    if invalid_token_pair_slot_topk_scale_slot_broadcast_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-pair slot/top-k scale-slot broadcast stream schedule"
        )
    if invalid_token_pair_slot_topk_route_bucket_codeword_reduce:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-pair slot/top-k route-bucket codeword-reduce schedule"
        )
    if invalid_token_pair_slot_topk_kblock_microtile_stream:
        structural_reasons.append(
            "candidate reopens the speed-rejected token-pair slot/top-k K-block microtile stream schedule"
        )
    if candidate.uses_decoded_b_threadgroup_staging:
        structural_reasons.append("candidate stages decoded B in threadgroup memory")
    if candidate.uses_lane_local_fragment_buffer:
        structural_reasons.append("candidate uses lane-local fragment buffering")
    if not candidate.preserves_compressed_rhs_storage:
        structural_reasons.append("candidate does not preserve compressed RHS storage")
    if not candidate.preserves_codeword_scale_slots:
        structural_reasons.append("candidate does not preserve codeword scale slots")
    if candidate.decoded_dense_weight_bytes != 0:
        structural_reasons.append("candidate materializes decoded dense RHS bytes")

    if structural_reasons:
        if invalid_component_stream_shared_decode:
            rejected_next_steps = [
                "do_not_benchmark_scalar_shared_decode_cache_scaffold",
                "do_not_claim_cross_ntile_shared_decode_from_component_stream_storage",
                "do_not_retime_component_stream_tensorops_unchanged",
            ]
        elif invalid_expert_kblock_predecode_cache:
            rejected_next_steps = [
                "do_not_benchmark_expert_kblock_predecode_cache",
                "do_not_materialize_decoded_rhs_equivalent_expert_kblock_cache",
                "do_not_retime_near_duplicate_expert_kblock_v2",
            ]
        elif invalid_route_abs_index_projection:
            rejected_next_steps = [
                "do_not_benchmark_route_abs_index_projection_cache",
                "do_not_ignore_sign_axis_for_e8p_route_projection",
                "do_not_claim_speed_from_abs_index_cardinality",
            ]
        elif invalid_route_abs_component_projection:
            rejected_next_steps = [
                "do_not_benchmark_exact_route_abs_component_projection_family",
                "do_not_add_more_route_abs_component_cache_accounting_variants",
                "do_not_claim_speed_from_lower_bound_cardinality_probe",
            ]
        elif invalid_route_slot_codeword_stream:
            rejected_next_steps = [
                "do_not_retime_scalar_route_slot_codeword_stream",
                "do_not_claim_lane_s_from_rejected_route_slot_speed_packet",
                "change_route_slot_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_route_slot_mma_codeword_tile:
            rejected_next_steps = [
                "do_not_retime_route_slot_mma_codeword_tile",
                "do_not_claim_lane_s_from_rejected_route_slot_mma_speed_packet",
                "change_route_slot_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_active_route_tile_codeword_outer_product:
            rejected_next_steps = [
                "do_not_retime_active_route_tile_codeword_outer_product",
                "do_not_claim_lane_s_from_rejected_active_route_tile_speed_packet",
                "change_active_route_tile_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_expert_cohort_codeword_broadcast:
            rejected_next_steps = [
                "do_not_retime_expert_cohort_codeword_broadcast",
                "do_not_claim_lane_s_from_rejected_expert_cohort_speed_packet",
                "change_expert_cohort_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_cohort_codeword_stream:
            rejected_next_steps = [
                "do_not_retime_token_cohort_codeword_stream",
                "do_not_claim_lane_s_from_rejected_token_cohort_speed_packet",
                "change_token_cohort_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_cohort_mma_codeword_tile:
            rejected_next_steps = [
                "do_not_retime_token_cohort_mma_codeword_tile",
                "do_not_claim_lane_s_from_rejected_token_cohort_mma_speed_packet",
                "change_token_cohort_mma_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_output_stationary_codeword_tile:
            rejected_next_steps = [
                "do_not_retime_output_stationary_codeword_tile",
                "do_not_claim_lane_s_from_rejected_output_stationary_speed_packet",
                "change_output_stationary_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_input_stationary_codeword_tile:
            rejected_next_steps = [
                "do_not_retime_input_stationary_codeword_tile",
                "do_not_claim_lane_s_from_rejected_input_stationary_speed_packet",
                "change_input_stationary_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_route_codeword_lut_accumulate:
            rejected_next_steps = [
                "do_not_retime_route_codeword_lut_accumulate",
                "do_not_claim_lane_s_from_rejected_route_codeword_lut_speed_packet",
                "change_route_codeword_lut_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_rowwise_codeword_tile_accumulate:
            rejected_next_steps = [
                "do_not_retime_rowwise_codeword_tile_accumulate",
                "do_not_claim_lane_s_from_rejected_rowwise_codeword_tile_speed_packet",
                "change_rowwise_codeword_tile_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_output_tile_local_codeword_lut:
            rejected_next_steps = [
                "do_not_retime_output_tile_local_codeword_lut",
                "do_not_claim_lane_s_from_rejected_output_tile_local_speed_packet",
                "change_output_tile_local_codeword_lut_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_kblock_wavefront_codeword_scan:
            rejected_next_steps = [
                "do_not_retime_kblock_wavefront_codeword_scan",
                "do_not_claim_lane_s_from_rejected_kblock_wavefront_speed_packet",
                "preserve_token_route_slot_output_contract_before_next_speed_packet",
            ]
        elif invalid_token_route_output_stripe_pipeline:
            rejected_next_steps = [
                "do_not_retime_token_route_output_stripe_pipeline",
                "do_not_claim_lane_s_from_rejected_token_route_output_stripe_speed_packet",
                "change_token_route_output_stripe_pipeline_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_expert_kblock_scale_slot_stream:
            rejected_next_steps = [
                "do_not_retime_expert_kblock_scale_slot_stream",
                "do_not_claim_lane_s_from_rejected_expert_kblock_scale_slot_stream_speed_packet",
                "change_expert_kblock_scale_slot_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_scale_group_route_block_reduce:
            rejected_next_steps = [
                "do_not_retime_scale_group_route_block_reduce",
                "do_not_claim_lane_s_from_rejected_scale_group_route_block_reduce_speed_packet",
                "change_scale_group_route_block_reduce_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_route_block_output_group_stream:
            rejected_next_steps = [
                "do_not_retime_route_block_output_group_stream",
                "do_not_claim_lane_s_from_rejected_route_block_output_group_stream_speed_packet",
                "change_route_block_output_group_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_output_group_pretransposed_codeword_stream:
            rejected_next_steps = [
                "do_not_retime_output_group_pretransposed_codeword_stream",
                "do_not_claim_lane_s_from_rejected_output_group_pretransposed_codeword_stream_speed_packet",
                "change_output_group_pretransposed_codeword_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_kblock_output_group_route_fused_stream:
            rejected_next_steps = [
                "do_not_retime_kblock_output_group_route_fused_stream",
                "do_not_claim_lane_s_from_rejected_kblock_output_group_route_fused_stream_speed_packet",
                "change_kblock_output_group_route_fused_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_route_tile_output_swizzle_stream:
            rejected_next_steps = [
                "do_not_retime_route_tile_output_swizzle_stream",
                "do_not_claim_lane_s_from_rejected_route_tile_output_swizzle_stream_speed_packet",
                "change_route_tile_output_swizzle_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_topk_output_tile_stream:
            rejected_next_steps = [
                "do_not_retime_token_topk_output_tile_stream",
                "do_not_claim_lane_s_from_rejected_token_topk_output_tile_stream_speed_packet",
                "change_token_topk_output_tile_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_block_output_group_stream:
            rejected_next_steps = [
                "do_not_retime_token_block_output_group_stream",
                "do_not_claim_lane_s_from_rejected_token_block_output_group_stream_speed_packet",
                "change_token_block_output_group_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_output_stripe_group_stream:
            rejected_next_steps = [
                "do_not_retime_token_output_stripe_group_stream",
                "do_not_claim_lane_s_from_rejected_token_output_stripe_group_stream_speed_packet",
                "change_token_output_stripe_group_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_pair_kblock_accumulator_stream:
            rejected_next_steps = [
                "do_not_retime_token_pair_kblock_accumulator_stream",
                "do_not_claim_lane_s_from_rejected_token_pair_kblock_accumulator_stream_speed_packet",
                "change_token_pair_kblock_accumulator_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_pair_output_group_stream:
            rejected_next_steps = [
                "do_not_retime_token_pair_output_group_stream",
                "do_not_claim_lane_s_from_rejected_token_pair_output_group_stream_speed_packet",
                "change_token_pair_output_group_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_pair_slot_topk_output_group_stream:
            rejected_next_steps = [
                "do_not_retime_token_pair_slot_topk_output_group_stream",
                "do_not_claim_lane_s_from_rejected_token_pair_slot_topk_output_group_stream_speed_packet",
                "change_token_pair_slot_topk_output_group_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_pair_slot_topk_codeword_group_pipeline:
            rejected_next_steps = [
                "do_not_retime_token_pair_slot_topk_codeword_group_pipeline",
                "do_not_claim_lane_s_from_rejected_token_pair_slot_topk_codeword_group_pipeline_speed_packet",
                "change_token_pair_slot_topk_codeword_group_pipeline_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_pair_slot_topk_scale_slot_broadcast_stream:
            rejected_next_steps = [
                "do_not_retime_token_pair_slot_topk_scale_slot_broadcast_stream",
                "do_not_claim_lane_s_from_rejected_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet",
                "change_token_pair_slot_topk_scale_slot_broadcast_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_pair_slot_topk_route_bucket_codeword_reduce:
            rejected_next_steps = [
                "do_not_retime_token_pair_slot_topk_route_bucket_codeword_reduce",
                "do_not_claim_lane_s_from_rejected_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet",
                "change_token_pair_slot_topk_route_bucket_codeword_reduce_layout_or_kernel_family_before_next_speed_packet",
            ]
        elif invalid_token_pair_slot_topk_kblock_microtile_stream:
            rejected_next_steps = [
                "do_not_retime_token_pair_slot_topk_kblock_microtile_stream",
                "do_not_claim_lane_s_from_rejected_token_pair_slot_topk_kblock_microtile_stream_speed_packet",
                "change_token_pair_slot_topk_kblock_microtile_stream_layout_or_kernel_family_before_next_speed_packet",
            ]
        else:
            rejected_next_steps = [
                f"do_not_benchmark_near_duplicate_{matched[0]}"
                if matched
                else "do_not_benchmark_structurally_invalid_candidate",
            ]
        if candidate.reconstructs_e8p_values_per_fragment:
            rejected_next_steps.append("do_not_retime_current_tensorops_fragment_decode_shape")
        if candidate.uses_decoded_b_threadgroup_staging:
            rejected_next_steps.append("do_not_retime_decoded_b_threadgroup_staging")
        if candidate.uses_lane_local_fragment_buffer:
            rejected_next_steps.append("do_not_retime_lane_local_fragment_buffer")
        return {
            "target_kernel_family": candidate.target_kernel_family,
            "storage_constraint": candidate.storage_constraint,
            "dispatch_grid": candidate.dispatch_grid,
            "decode_reuse_scope": candidate.decode_reuse_scope,
            "tensorops_rhs_source": candidate.tensorops_rhs_source,
            "matched_rejected_families": list(matched),
            "decision": (
                "reject_component_stream_shared_decode_cross_ntile_cache"
                if invalid_component_stream_shared_decode
                else "reject_expert_kblock_predecode_cache_decoded_rhs_equivalent"
                if invalid_expert_kblock_predecode_cache
                else "reject_route_abs_index_projection_missing_sign_axis"
                if invalid_route_abs_index_projection
                else "reject_route_abs_component_projection_lower_bound_over_cap"
                if invalid_route_abs_component_projection
                else "reject_route_slot_codeword_stream_speed_path"
                if invalid_route_slot_codeword_stream
                else "reject_route_slot_mma_codeword_tile_speed_path"
                if invalid_route_slot_mma_codeword_tile
                else "reject_active_route_tile_codeword_outer_product_speed_path"
                if invalid_active_route_tile_codeword_outer_product
                else "reject_expert_cohort_codeword_broadcast_speed_path"
                if invalid_expert_cohort_codeword_broadcast
                else "reject_token_cohort_codeword_stream_speed_path"
                if invalid_token_cohort_codeword_stream
                else "reject_token_cohort_mma_codeword_tile_speed_path"
                if invalid_token_cohort_mma_codeword_tile
                else "reject_output_stationary_codeword_tile_speed_path"
                if invalid_output_stationary_codeword_tile
                else "reject_input_stationary_codeword_tile_speed_path"
                if invalid_input_stationary_codeword_tile
                else "reject_route_codeword_lut_accumulate_speed_path"
                if invalid_route_codeword_lut_accumulate
                else "reject_rowwise_codeword_tile_accumulate_speed_path"
                if invalid_rowwise_codeword_tile_accumulate
                else "reject_output_tile_local_codeword_lut_speed_path"
                if invalid_output_tile_local_codeword_lut
                else "reject_kblock_wavefront_codeword_scan_speed_path"
                if invalid_kblock_wavefront_codeword_scan
                else "reject_token_route_output_stripe_pipeline_speed_path"
                if invalid_token_route_output_stripe_pipeline
                else "reject_expert_kblock_scale_slot_stream_speed_path"
                if invalid_expert_kblock_scale_slot_stream
                else "reject_scale_group_route_block_reduce_speed_path"
                if invalid_scale_group_route_block_reduce
                else "reject_route_block_output_group_stream_speed_path"
                if invalid_route_block_output_group_stream
                else "reject_output_group_pretransposed_codeword_stream_speed_path"
                if invalid_output_group_pretransposed_codeword_stream
                else "reject_kblock_output_group_route_fused_stream_speed_path"
                if invalid_kblock_output_group_route_fused_stream
                else "reject_route_tile_output_swizzle_stream_speed_path"
                if invalid_route_tile_output_swizzle_stream
                else "reject_token_topk_output_tile_stream_speed_path"
                if invalid_token_topk_output_tile_stream
                else "reject_token_block_output_group_stream_speed_path"
                if invalid_token_block_output_group_stream
                else "reject_token_output_stripe_group_stream_speed_path"
                if invalid_token_output_stripe_group_stream
                else "reject_token_expert_output_block_stream_speed_path"
                if invalid_token_expert_output_block_stream
                else "reject_token_pair_kblock_accumulator_stream_speed_path"
                if invalid_token_pair_kblock_accumulator_stream
                else "reject_token_pair_output_group_stream_speed_path"
                if invalid_token_pair_output_group_stream
                else "reject_token_pair_slot_topk_output_group_stream_speed_path"
                if invalid_token_pair_slot_topk_output_group_stream
                else "reject_token_pair_slot_topk_codeword_group_pipeline_speed_path"
                if invalid_token_pair_slot_topk_codeword_group_pipeline
                else "reject_token_pair_slot_topk_scale_slot_broadcast_stream_speed_path"
                if invalid_token_pair_slot_topk_scale_slot_broadcast_stream
                else "reject_token_pair_slot_topk_route_bucket_codeword_reduce_speed_path"
                if invalid_token_pair_slot_topk_route_bucket_codeword_reduce
                else "reject_token_pair_slot_topk_kblock_microtile_stream_speed_path"
                if invalid_token_pair_slot_topk_kblock_microtile_stream
                else "reject_current_family_near_duplicate"
            ),
            "candidate_ready_for_source_probe": False,
            "speed_claim": False,
            "native_parity_claim": False,
            "structural_reasons": structural_reasons,
            "required_next_features": (
                [
                    "materially_change_rhs_storage_contract_or_kernel_family",
                    "avoid_cross_ntile_component_stream_decode_cache",
                    "avoid_decoded_rhs_equivalent_cache",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                ]
                if invalid_component_stream_shared_decode
                else [
                    "materially_change_rhs_storage_contract_or_kernel_family",
                    "avoid_expert_kblock_predecoded_rhs_cache",
                    "avoid_decoded_rhs_equivalent_cache",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                ]
                if invalid_expert_kblock_predecode_cache
                else [
                    "materially_change_rhs_storage_contract_or_kernel_family",
                    "preserve_e8p_sign_axis_or_use_full_codeword_projection",
                    "avoid_full_codeword_equivalent_route_cache",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                ]
                if invalid_route_abs_index_projection
                else [
                    "materially_change_rhs_storage_contract_or_kernel_family",
                    "prove_new_exact_compression_below_abs_component_lower_bound",
                    "preserve_route_slot_axis_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                ]
                if invalid_route_abs_component_projection
                else [
                    "materially_change_route_slot_layout_or_kernel_family",
                    "avoid_scalar_route_slot_per_output_column_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_route_slot_codeword_stream
                else [
                    "materially_change_route_slot_layout_or_kernel_family",
                    "avoid_output_tile_mma_codeword_tile_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_route_slot_mma_codeword_tile
                else [
                    "materially_change_active_route_tile_layout_or_kernel_family",
                    "avoid_active_route_tile_output_microtile_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_active_route_tile_codeword_outer_product
                else [
                    "materially_change_expert_cohort_layout_or_kernel_family",
                    "avoid_expert_major_codeword_broadcast_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_expert_cohort_codeword_broadcast
                else [
                    "materially_change_token_cohort_layout_or_kernel_family",
                    "avoid_token_cohort_streaming_codeword_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_cohort_codeword_stream
                else [
                    "materially_change_token_cohort_mma_layout_or_kernel_family",
                    "avoid_token_cohort_output_tile_mma_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_cohort_mma_codeword_tile
                else [
                    "materially_change_output_stationary_layout_or_kernel_family",
                    "avoid_output_tile_stationary_route_batch_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_output_stationary_codeword_tile
                else [
                    "materially_change_input_stationary_layout_or_kernel_family",
                    "avoid_input_tile_stationary_route_batch_sweep",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_input_stationary_codeword_tile
                else [
                    "materially_change_route_codeword_lut_layout_or_kernel_family",
                    "avoid_route_local_full_codeword_lut",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_route_codeword_lut_accumulate
                else [
                    "materially_change_rowwise_codeword_tile_layout_or_kernel_family",
                    "avoid_rowwise_per_output_row_codeword_dot_recompute",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_rowwise_codeword_tile_accumulate
                else [
                    "materially_change_output_tile_local_codeword_lut_layout_or_kernel_family",
                    "avoid_output_tile_local_full_codeword_pass",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_output_tile_local_codeword_lut
                else [
                    "materially_change_kblock_wavefront_layout_or_kernel_family",
                    "preserve_token_route_slot_output_contract",
                    "avoid_route_microtile_output_expansion",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_kblock_wavefront_codeword_scan
                else [
                    "materially_change_token_route_output_stripe_pipeline_layout_or_kernel_family",
                    "avoid_token_route_output_stripe_pipeline_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_route_output_stripe_pipeline
                else [
                    "materially_change_expert_kblock_scale_slot_stream_layout_or_kernel_family",
                    "avoid_expert_kblock_scale_slot_stream_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_expert_kblock_scale_slot_stream
                else [
                    "materially_change_scale_group_route_block_reduce_layout_or_kernel_family",
                    "avoid_scale_group_route_block_reduce_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_scale_group_route_block_reduce
                else [
                    "materially_change_route_block_output_group_stream_layout_or_kernel_family",
                    "avoid_route_block_output_group_stream_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_route_block_output_group_stream
                else [
                    "materially_change_output_group_pretransposed_codeword_stream_layout_or_kernel_family",
                    "avoid_output_group_pretransposed_codeword_stream_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_output_group_pretransposed_codeword_stream
                else [
                    "materially_change_kblock_output_group_route_fused_stream_layout_or_kernel_family",
                    "avoid_kblock_output_group_route_fused_stream_schedule",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_kblock_output_group_route_fused_stream
                else [
                    "materially_change_route_tile_output_swizzle_stream_layout_or_kernel_family",
                    "avoid_route_tile_output_swizzle_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_route_tile_output_swizzle_stream
                else [
                    "materially_change_token_topk_output_tile_stream_layout_or_kernel_family",
                    "avoid_token_topk_output_tile_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_topk_output_tile_stream
                else [
                    "materially_change_token_block_output_group_stream_layout_or_kernel_family",
                    "avoid_token_block_output_group_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_block_output_group_stream
                else [
                    "materially_change_token_output_stripe_group_stream_layout_or_kernel_family",
                    "avoid_token_output_stripe_group_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_output_stripe_group_stream
                else [
                    "materially_change_token_expert_output_block_stream_layout_or_kernel_family",
                    "avoid_token_expert_output_block_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_expert_output_block_stream
                else [
                    "materially_change_token_pair_kblock_accumulator_stream_layout_or_kernel_family",
                    "avoid_token_pair_kblock_accumulator_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_pair_kblock_accumulator_stream
                else [
                    "materially_change_token_pair_output_group_stream_layout_or_kernel_family",
                    "avoid_token_pair_output_group_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_pair_output_group_stream
                else [
                    "materially_change_token_pair_slot_topk_output_group_stream_layout_or_kernel_family",
                    "avoid_token_pair_slot_topk_output_group_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_pair_slot_topk_output_group_stream
                else [
                    "materially_change_token_pair_slot_topk_codeword_group_pipeline_layout_or_kernel_family",
                    "avoid_token_pair_slot_topk_codeword_group_pipeline_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_pair_slot_topk_codeword_group_pipeline
                else [
                    "materially_change_token_pair_slot_topk_scale_slot_broadcast_stream_layout_or_kernel_family",
                    "avoid_token_pair_slot_topk_scale_slot_broadcast_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_route_slot_axis_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_pair_slot_topk_scale_slot_broadcast_stream
                else [
                    "materially_change_token_pair_slot_topk_route_bucket_codeword_reduce_layout_or_kernel_family",
                    "avoid_route_bucket_codeword_reduce_route_slot_expansion",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_pair_slot_topk_route_bucket_codeword_reduce
                else [
                    "materially_change_token_pair_slot_topk_kblock_microtile_stream_layout_or_kernel_family",
                    "avoid_token_pair_slot_topk_kblock_microtile_stream_schedule",
                    "preserve_tokens_x_topk_output_shape_or_exact_token_indirection",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                    "prove_source_structure_before_native_or_speed_work",
                ]
                if invalid_token_pair_slot_topk_kblock_microtile_stream
                else [
                    "materially_change_rhs_storage_contract_or_kernel_family",
                    "avoid_per_fragment_e8p_reconstruction",
                    "preserve_compressed_rhs_storage",
                    "preserve_codeword_scale_slots",
                ]
            ),
            "rejected_next_steps": rejected_next_steps,
        }

    return {
        "target_kernel_family": candidate.target_kernel_family,
        "storage_constraint": candidate.storage_constraint,
        "dispatch_grid": candidate.dispatch_grid,
        "decode_reuse_scope": candidate.decode_reuse_scope,
        "tensorops_rhs_source": candidate.tensorops_rhs_source,
        "matched_rejected_families": [],
        "decision": "candidate_family_ready_for_source_structure_probe",
        "candidate_ready_for_source_probe": True,
        "speed_claim": False,
        "native_parity_claim": False,
        "structural_reasons": [],
        "required_before_benchmark": [
            "source_structure_guardrail",
            "native_scalar_oracle_parity",
            "air_down_group_size_352_parity",
            "same_window_q2_speed_packet",
        ],
        "rejected_next_steps": [
            "do_not_make_speed_claim_from_contract_only",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        ],
    }


def pack_e8p_rhs_tiles(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
) -> E8PPackedRHSTiles:
    """Pack E8P RHS weights into q2-like compressed NAX tiles.

    The output is a layout contract for a future sorted gather-qmm RHS NAX
    kernel: codes stay uint16 E8P codewords and scales stay compact per touched
    scale group. No dense FP16 RHS is materialized by this packer.
    """

    codes_np = np.asarray(codes)
    scales_np = np.asarray(scales)
    experts, output_dims, input_dims = _validate_e8p_inputs(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )
    n_tiles = _ceil_div(output_dims, bn)
    k_blocks = input_dims // bk
    codewords_per_bk = bk // CODEWORD_DIM
    scale_group_indices, codeword_scale_slots, max_scale_groups = _scale_maps(
        input_dims=input_dims,
        group_size=group_size,
        bk=bk,
    )

    code_tiles = np.zeros(
        (experts, n_tiles, k_blocks, bn, codewords_per_bk),
        dtype=np.uint16,
    )
    scale_tiles = np.zeros(
        (experts, n_tiles, k_blocks, bn, max_scale_groups),
        dtype=scales_np.dtype,
    )
    for n_tile in range(n_tiles):
        out_start = n_tile * bn
        out_stop = min(output_dims, out_start + bn)
        out_count = out_stop - out_start
        for k_block in range(k_blocks):
            code_start = k_block * codewords_per_bk
            code_stop = code_start + codewords_per_bk
            groups = scale_group_indices[k_block]
            valid_groups = groups[groups >= 0]
            code_tiles[:, n_tile, k_block, :out_count, :] = codes_np[
                :, out_start:out_stop, code_start:code_stop
            ]
            scale_tiles[:, n_tile, k_block, :out_count, : valid_groups.shape[0]] = scales_np[
                :, out_start:out_stop, valid_groups
            ]

    layout = E8PPackedRHSLayout(
        target_kernel_family="sorted_gather_qmm_rhs_nax",
        storage_constraint="compressed_e8p_codes_scales",
        experts=experts,
        output_dims=output_dims,
        input_dims=input_dims,
        group_size=group_size,
        bn=bn,
        bk=bk,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        codewords_per_bk=codewords_per_bk,
        max_scale_groups_per_bk=max_scale_groups,
    )
    return E8PPackedRHSTiles(
        layout=layout,
        code_tiles=code_tiles,
        scale_tiles=scale_tiles,
        scale_group_indices=scale_group_indices,
        codeword_scale_slots=codeword_scale_slots,
    )


def pack_e8p_split_byte_rhs_tiles(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
) -> E8PSplitByteRHSTiles:
    """Pack E8P RHS tiles as split sign/absolute-index/parity byte factors.

    This layout keeps the same q2-like RHS tile geometry as
    ``pack_e8p_rhs_tiles`` but avoids storing whole uint16 codewords. It is a
    native-kernel contract for split-byte E8P experiments where sign bytes and
    absolute-row indices can be handled as separate reusable factors.
    """

    packed = pack_e8p_rhs_tiles(codes, scales, group_size=group_size, bn=bn, bk=bk)
    sign_tiles = (packed.code_tiles & np.uint16(0xFF)).astype(np.uint8)
    abs_index_tiles = (packed.code_tiles >> np.uint16(8)).astype(np.uint8)
    parity_tiles = _sign_parity8(sign_tiles)
    layout = E8PPackedRHSLayout(
        target_kernel_family="sorted_gather_qmm_rhs_nax_split_byte",
        storage_constraint="compressed_e8p_split_bytes_scales",
        experts=packed.layout.experts,
        output_dims=packed.layout.output_dims,
        input_dims=packed.layout.input_dims,
        group_size=packed.layout.group_size,
        bn=packed.layout.bn,
        bk=packed.layout.bk,
        n_tiles=packed.layout.n_tiles,
        k_blocks=packed.layout.k_blocks,
        codewords_per_bk=packed.layout.codewords_per_bk,
        max_scale_groups_per_bk=packed.layout.max_scale_groups_per_bk,
        decoded_dense_weight_bytes=0,
    )
    return E8PSplitByteRHSTiles(
        layout=layout,
        sign_tiles=sign_tiles,
        abs_index_tiles=abs_index_tiles,
        parity_tiles=parity_tiles,
        scale_tiles=packed.scale_tiles,
        scale_group_indices=packed.scale_group_indices,
        codeword_scale_slots=packed.codeword_scale_slots,
    )


def pack_e8p_split_byte_factor_reuse_rhs_tiles(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
) -> E8PSplitByteFactorReuseRHSTiles:
    """Pack split-byte E8P factors as per-tile LUTs plus slot maps.

    This is a native-kernel contract for schedules that reuse absolute-row
    decode and sign-mask work outside the decoded-B staged tile. It keeps the
    same compact scale maps as the split-byte tile layout, but replaces full
    sign/abs factor grids with per-tile unique-value tables and uint8 slots.
    """

    split = pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=group_size, bn=bn, bk=bk)
    sign_lut, sign_slots, sign_counts = _factor_lut_and_slots(split.sign_tiles)
    abs_lut, abs_slots, abs_counts = _factor_lut_and_slots(split.abs_index_tiles)
    layout = E8PPackedRHSLayout(
        target_kernel_family="sorted_gather_qmm_rhs_nax_split_byte_factor_reuse",
        storage_constraint="compressed_e8p_split_byte_factor_lut_scales",
        experts=split.layout.experts,
        output_dims=split.layout.output_dims,
        input_dims=split.layout.input_dims,
        group_size=split.layout.group_size,
        bn=split.layout.bn,
        bk=split.layout.bk,
        n_tiles=split.layout.n_tiles,
        k_blocks=split.layout.k_blocks,
        codewords_per_bk=split.layout.codewords_per_bk,
        max_scale_groups_per_bk=split.layout.max_scale_groups_per_bk,
        decoded_dense_weight_bytes=0,
    )
    return E8PSplitByteFactorReuseRHSTiles(
        layout=layout,
        sign_byte_lut=sign_lut,
        sign_byte_slots=sign_slots,
        sign_byte_counts=sign_counts,
        abs_index_lut=abs_lut,
        abs_index_slots=abs_slots,
        abs_index_counts=abs_counts,
        scale_tiles=split.scale_tiles,
        scale_group_indices=split.scale_group_indices,
        codeword_scale_slots=split.codeword_scale_slots,
    )


def pack_e8p_expert_kblock_factor_reuse_rhs_tiles(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
) -> E8PExpertKBlockFactorReuseRHSTiles:
    """Pack split-byte E8P factors with one LUT per expert and K block.

    This is the native-facing contract for the broader reuse signal measured in
    the expert/K-block analyzer: sign and abs-index factors are decoded from a
    shared `(expert, k_block)` LUT across all output tiles, while per-output
    scale maps and codeword scale slots remain compact and tile-local.
    """

    split = pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=group_size, bn=bn, bk=bk)
    sign_lut, sign_slots, sign_counts = _expert_kblock_factor_lut_and_slots(split.sign_tiles)
    abs_lut, abs_slots, abs_counts = _expert_kblock_factor_lut_and_slots(split.abs_index_tiles)
    layout = E8PPackedRHSLayout(
        target_kernel_family="sorted_gather_qmm_rhs_nax_expert_kblock_factor_reuse",
        storage_constraint="compressed_e8p_expert_kblock_factor_lut_scales",
        experts=split.layout.experts,
        output_dims=split.layout.output_dims,
        input_dims=split.layout.input_dims,
        group_size=split.layout.group_size,
        bn=split.layout.bn,
        bk=split.layout.bk,
        n_tiles=split.layout.n_tiles,
        k_blocks=split.layout.k_blocks,
        codewords_per_bk=split.layout.codewords_per_bk,
        max_scale_groups_per_bk=split.layout.max_scale_groups_per_bk,
        decoded_dense_weight_bytes=0,
    )
    return E8PExpertKBlockFactorReuseRHSTiles(
        layout=layout,
        sign_byte_lut=sign_lut,
        sign_byte_slots=sign_slots,
        sign_byte_counts=sign_counts,
        abs_index_lut=abs_lut,
        abs_index_slots=abs_slots,
        abs_index_counts=abs_counts,
        scale_tiles=split.scale_tiles,
        scale_group_indices=split.scale_group_indices,
        codeword_scale_slots=split.codeword_scale_slots,
    )


def build_e8p_expert_kblock_dispatch_contract(
    packed: E8PExpertKBlockFactorReuseRHSTiles,
    *,
    route_tile_count: int,
) -> E8PExpertKBlockDispatchContract:
    """Describe the v2 dispatch shape needed to reuse one expert/K-block LUT.

    The current native primitive dispatches over output tiles and route tiles.
    This contract deliberately moves the reuse scope to the outer workgroup key:
    one `(expert, k_block, route_tile)` group owns all output tiles for that
    expert/K block while preserving the existing compressed slot/scales layout.
    """

    if route_tile_count <= 0:
        raise ValueError("route_tile_count must be positive")

    layout = packed.layout
    n_tiles = tuple(range(layout.n_tiles))
    workgroups: list[E8PExpertKBlockDispatchWorkgroup] = []
    for expert in range(layout.experts):
        for k_block in range(layout.k_blocks):
            scale_groups = tuple(
                int(group)
                for group in packed.scale_group_indices[k_block]
                if int(group) >= 0
            )
            codeword_slots = tuple(
                int(slot)
                for slot in packed.codeword_scale_slots[
                    k_block, : layout.codewords_per_bk
                ]
            )
            for route_tile in range(route_tile_count):
                workgroups.append(
                    E8PExpertKBlockDispatchWorkgroup(
                        expert=expert,
                        k_block=k_block,
                        route_tile=route_tile,
                        n_tiles=n_tiles,
                        sign_lut_index=(expert, k_block),
                        abs_index_lut_index=(expert, k_block),
                        sign_count=int(packed.sign_byte_counts[expert, k_block]),
                        abs_index_count=int(packed.abs_index_counts[expert, k_block]),
                        scale_group_indices=scale_groups,
                        codeword_scale_slots=codeword_slots,
                    )
                )

    return E8PExpertKBlockDispatchContract(
        target_kernel_family="sorted_gather_qmm_rhs_nax_expert_kblock_factor_reuse_v2",
        storage_constraint=layout.storage_constraint,
        dispatch_grid="experts_x_k_blocks_x_route_tiles",
        incompatible_primitive_grid="n_tiles_x_num_route_tiles",
        reuse_scope="expert_kblock",
        slot_layout="expert_n_tile_k_block_row_codeword",
        route_tile_count=route_tile_count,
        k_blocks=layout.k_blocks,
        n_tiles=layout.n_tiles,
        output_tile_size=layout.bn,
        decoded_dense_weight_bytes=layout.decoded_dense_weight_bytes,
        required_next_features=(
            "dispatch_over_reuse_scope_before_output_tile",
            "broaden_decode_reuse_scope_before_tensorops_fragment_fill",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        ),
        workgroups=tuple(workgroups),
    )


def build_e8p_expert_kblock_partial_reduction_contract(
    dispatch: E8PExpertKBlockDispatchContract,
    *,
    route_tile_size: int,
    accumulation_dtype: str = "float32",
) -> E8PExpertKBlockPartialReductionContract:
    """Describe the scratch and final reduction required after v2 K-block dispatch."""

    if route_tile_size <= 0:
        raise ValueError("route_tile_size must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(accumulation_dtype)
    if dtype_bytes is None:
        raise ValueError("accumulation_dtype must be float16 or float32")

    shape = (
        dispatch.route_tile_count,
        dispatch.k_blocks,
        dispatch.n_tiles,
        route_tile_size,
        dispatch.output_tile_size,
    )
    element_count = 1
    for dim in shape:
        element_count *= dim

    return E8PExpertKBlockPartialReductionContract(
        target_kernel_family=dispatch.target_kernel_family,
        partial_accumulation_grid="route_tiles_x_k_blocks_x_n_tiles",
        final_reduction_grid="route_tiles_x_n_tiles",
        partial_accumulation_shape=shape,
        partial_accumulation_dtype=accumulation_dtype,
        partial_accumulation_element_count=element_count,
        partial_accumulation_bytes=element_count * dtype_bytes,
        route_tile_size=route_tile_size,
        reduction_axis="k_blocks",
        output_shape="route_count_x_output_dims",
        valid_routes_source="route_tile_counts",
        expert_source="route_tile_experts",
        decoded_dense_weight_bytes=dispatch.decoded_dense_weight_bytes,
        required_next_features=(
            "allocate_partial_accumulation_scratch",
            "write_one_partial_per_kblock_workgroup",
            "reduce_kblock_partials_before_final_output",
            "keep_final_output_write_out_of_kblock_zero_only",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        ),
        rejected_next_steps=(
            "do_not_benchmark_single_kblock_completion_v2",
            "do_not_write_complete_output_from_one_kblock",
            "do_not_promote_without_final_reduction_pass",
        ),
    )


def pack_e8p_component_stream_rhs_tiles(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
) -> E8PComponentStreamRHSTiles:
    """Pack E8P RHS factors as per-codeword component streams.

    Unlike rejected B-fragment schedules, this contract exposes each signed E8P
    component as stream metadata before accumulation. It preserves compressed
    signs, abs-index rows, compact scale groups, and per-component scale slots
    without materializing decoded dense RHS bytes.
    """

    split = pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=group_size, bn=bn, bk=bk)
    bit_offsets = np.arange(CODEWORD_DIM, dtype=np.uint16)
    sign_component_bits = (
        (split.sign_tiles[..., None].astype(np.uint16) >> bit_offsets) & np.uint16(1)
    ).astype(np.uint8)
    component_count = split.layout.codewords_per_bk * CODEWORD_DIM
    component_codeword_indices = np.zeros((split.layout.k_blocks, component_count), dtype=np.int32)
    component_offsets = np.zeros_like(component_codeword_indices)
    component_scale_slots = np.zeros_like(component_codeword_indices)
    for k_block in range(split.layout.k_blocks):
        codeword_indices = np.repeat(
            np.arange(split.layout.codewords_per_bk, dtype=np.int32),
            CODEWORD_DIM,
        )
        offsets = np.tile(np.arange(CODEWORD_DIM, dtype=np.int32), split.layout.codewords_per_bk)
        component_codeword_indices[k_block] = codeword_indices
        component_offsets[k_block] = offsets
        component_scale_slots[k_block] = split.codeword_scale_slots[k_block, codeword_indices]

    layout = E8PPackedRHSLayout(
        target_kernel_family="sorted_gather_qmm_rhs_nax_component_stream",
        storage_constraint="compressed_e8p_component_stream_scales",
        experts=split.layout.experts,
        output_dims=split.layout.output_dims,
        input_dims=split.layout.input_dims,
        group_size=split.layout.group_size,
        bn=split.layout.bn,
        bk=split.layout.bk,
        n_tiles=split.layout.n_tiles,
        k_blocks=split.layout.k_blocks,
        codewords_per_bk=split.layout.codewords_per_bk,
        max_scale_groups_per_bk=split.layout.max_scale_groups_per_bk,
        decoded_dense_weight_bytes=0,
    )
    return E8PComponentStreamRHSTiles(
        layout=layout,
        sign_component_bits=sign_component_bits,
        abs_index_tiles=split.abs_index_tiles,
        scale_tiles=split.scale_tiles,
        scale_group_indices=split.scale_group_indices,
        codeword_scale_slots=split.codeword_scale_slots,
        component_scale_slots=component_scale_slots,
        component_codeword_indices=component_codeword_indices,
        component_offsets=component_offsets,
    )


def evaluate_e8p_expert_kblock_decode_reuse_feasibility(
    packed: E8PExpertKBlockFactorReuseRHSTiles,
    *,
    decode_cache_dtype: str = "float16",
) -> E8PExpertKBlockDecodeReuseFeasibility:
    """Explain why the obvious v2 predecode cache is not a valid speed family."""

    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(decode_cache_dtype)
    if dtype_bytes is None:
        raise ValueError("decode_cache_dtype must be float16 or float32")

    layout = packed.layout
    correct_cache_shape = (
        layout.experts,
        layout.k_blocks,
        layout.n_tiles,
        layout.bn,
        layout.bk,
    )
    element_count = 1
    for dim in correct_cache_shape:
        element_count *= dim

    return E8PExpertKBlockDecodeReuseFeasibility(
        target_kernel_family="sorted_gather_qmm_rhs_nax_expert_kblock_predecode_cache",
        decision="reject_expert_kblock_predecode_cache_decoded_rhs_equivalent",
        output_tile_reuse_feasible_without_decoded_cache=False,
        factor_lut_shared_across_output_tiles=True,
        slot_values_depend_on_output_column=True,
        scale_values_depend_on_output_column=True,
        correct_cache_shape=correct_cache_shape,
        correct_cache_dtype=decode_cache_dtype,
        correct_cache_bytes=element_count * dtype_bytes,
        correct_cache_is_decoded_rhs_equivalent=True,
        structural_risk=(
            "expert/K-block factor LUTs are shared across output tiles, but "
            "sign/abs slots and scale slots are keyed by expert/n_tile/k_block/"
            "output_column/codeword. Avoiding per-fragment reconstruction by "
            "predecoding the missing axes requires an expert/k_block/n_tile/"
            "output_column/input-column cache, which is decoded-RHS-equivalent "
            "storage."
        ),
        next_track_b_hypothesis="change_expert_kblock_rhs_layout_or_kernel_family",
        required_next_features=(
            "materially_change_rhs_storage_contract_or_kernel_family",
            "avoid_expert_kblock_predecoded_rhs_cache",
            "avoid_per_fragment_e8p_reconstruction",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        ),
        rejected_next_steps=(
            "do_not_benchmark_expert_kblock_predecode_cache",
            "do_not_materialize_decoded_rhs_equivalent_expert_kblock_cache",
            "do_not_retime_near_duplicate_expert_kblock_v2",
        ),
    )


def evaluate_e8p_route_codeword_projection_feasibility(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
    route_tile_count: int = 1,
    route_tile_size: int = 64,
    unique_codeword_cap: int = 4096,
    projection_cache_dtype: str = "float16",
    projection_cache_byte_cap: int = 8 * 1024 * 1024,
) -> E8PRouteCodewordProjectionFeasibility:
    """Measure a new activation-side codeword projection family.

    The hypothesis is to cache route-tile activation dot products against the
    unique full uint16 E8P codewords for each expert/K block, then apply compact
    output-column scale slots. This changes the RHS family without predecoding
    dense B tiles, but it is only plausible if full-codeword cardinality stays
    bounded on real artifacts.
    """

    if route_tile_count <= 0:
        raise ValueError("route_tile_count must be positive")
    if route_tile_size <= 0:
        raise ValueError("route_tile_size must be positive")
    if unique_codeword_cap <= 0:
        raise ValueError("unique_codeword_cap must be positive")
    if projection_cache_byte_cap <= 0:
        raise ValueError("projection_cache_byte_cap must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(projection_cache_dtype)
    if dtype_bytes is None:
        raise ValueError("projection_cache_dtype must be float16 or float32")

    codes_np = np.asarray(codes)
    scales_np = np.asarray(scales)
    experts, output_dims, input_dims = _validate_e8p_inputs(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )
    k_blocks = input_dims // bk
    n_tiles = _ceil_div(output_dims, bn)
    codewords_per_bk = bk // CODEWORD_DIM
    unique_counts = np.zeros((experts, k_blocks), dtype=np.int32)
    for expert in range(experts):
        for k_block in range(k_blocks):
            start = k_block * codewords_per_bk
            stop = start + codewords_per_bk
            unique_counts[expert, k_block] = int(
                np.unique(codes_np[expert, :, start:stop].reshape(-1)).size
            )

    max_unique = int(unique_counts.max(initial=0))
    max_possible = max(1, output_dims * codewords_per_bk)
    projection_cache_shape = (route_tile_count, k_blocks, max_unique)
    element_count = route_tile_count * k_blocks * max_unique
    projection_cache_bytes = element_count * dtype_bytes
    ready = (
        max_unique <= unique_codeword_cap
        and projection_cache_bytes <= projection_cache_byte_cap
    )
    if ready:
        decision = "route_codeword_projection_ready_for_source_probe"
        risk = (
            "full-codeword cardinality is bounded enough for an activation-side "
            "route-tile dot cache; native work still needs source and parity gates."
        )
        required_next_features = (
            "source_structure_guardrail",
            "activation_codeword_dot_cache_parity",
            "air_down_group_size_352_parity",
            "same_window_q2_speed_packet_after_native_parity",
        )
        rejected_next_steps = (
            "do_not_claim_speed_from_unique_count_probe",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        )
    else:
        decision = "reject_route_codeword_projection_cache_too_large"
        risk = (
            "full-codeword cardinality makes the activation-side route cache too "
            "large under the configured cap; this avoids spending a native timing "
            "slice on another unbounded RHS-derived cache."
        )
        required_next_features = (
            "materially_change_rhs_storage_contract_or_kernel_family",
            "reduce_full_codeword_cardinality_before_projection_cache",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        )
        rejected_next_steps = (
            "do_not_benchmark_route_codeword_projection_cache",
            "do_not_materialize_unbounded_activation_codeword_dot_cache",
            "do_not_claim_speed_from_unique_count_probe",
        )

    return E8PRouteCodewordProjectionFeasibility(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_codeword_projection",
        decision=decision,
        storage_constraint="compressed_e8p_route_codeword_projection_scales",
        dispatch_grid="route_tiles_x_experts_x_k_blocks",
        activation_projection_scope="route_tile_x_expert_kblock_full_codeword",
        experts=experts,
        output_dims=output_dims,
        input_dims=input_dims,
        group_size=group_size,
        bn=bn,
        bk=bk,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        codewords_per_bk=codewords_per_bk,
        route_tile_count=route_tile_count,
        route_tile_size=route_tile_size,
        unique_codeword_counts_shape=tuple(int(dim) for dim in unique_counts.shape),
        min_unique_codewords_per_expert_kblock=int(unique_counts.min()),
        median_unique_codewords_per_expert_kblock=float(np.median(unique_counts)),
        p95_unique_codewords_per_expert_kblock=float(np.percentile(unique_counts, 95)),
        max_unique_codewords_per_expert_kblock=max_unique,
        max_unique_ratio_per_expert_kblock=float(max_unique / max_possible),
        unique_codeword_cap=unique_codeword_cap,
        projection_cache_dtype=projection_cache_dtype,
        projection_cache_shape=projection_cache_shape,
        projection_cache_bytes=projection_cache_bytes,
        projection_cache_byte_cap=projection_cache_byte_cap,
        decoded_dense_weight_bytes=0,
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        candidate_ready_for_source_probe=ready,
        structural_risk=risk,
        required_next_features=required_next_features,
        rejected_next_steps=rejected_next_steps,
    )


def evaluate_e8p_route_active_codeword_projection_feasibility(
    codes: np.ndarray,
    scales: np.ndarray,
    route_experts: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
    route_tile_size: int = 64,
    unique_codeword_cap: int = 4096,
    projection_cache_dtype: str = "float16",
    projection_cache_byte_cap: int = 8 * 1024 * 1024,
) -> E8PRouteActiveCodewordProjectionFeasibility:
    """Measure full-codeword route projection using actual active route tiles."""

    if route_tile_size <= 0:
        raise ValueError("route_tile_size must be positive")
    if unique_codeword_cap <= 0:
        raise ValueError("unique_codeword_cap must be positive")
    if projection_cache_byte_cap <= 0:
        raise ValueError("projection_cache_byte_cap must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(projection_cache_dtype)
    if dtype_bytes is None:
        raise ValueError("projection_cache_dtype must be float16 or float32")

    codes_np = np.asarray(codes)
    scales_np = np.asarray(scales)
    experts, output_dims, input_dims = _validate_e8p_inputs(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )
    route_experts_np = np.asarray(route_experts)
    if route_experts_np.ndim != 1:
        raise ValueError("route_experts must be 1D")
    if route_experts_np.size and (
        int(route_experts_np.min()) < 0 or int(route_experts_np.max()) >= experts
    ):
        raise ValueError("route_experts contains an expert id out of range")

    k_blocks = input_dims // bk
    n_tiles = _ceil_div(output_dims, bn)
    codewords_per_bk = bk // CODEWORD_DIM
    unique_counts = np.zeros((experts, k_blocks), dtype=np.int32)
    for expert in range(experts):
        for k_block in range(k_blocks):
            start = k_block * codewords_per_bk
            stop = start + codewords_per_bk
            unique_counts[expert, k_block] = int(
                np.unique(codes_np[expert, :, start:stop].reshape(-1)).size
            )

    route_counts = np.bincount(route_experts_np.astype(np.int64), minlength=experts)
    active_route_tile_counts_np = np.asarray(
        [
            _ceil_div(int(count), route_tile_size) if int(count) > 0 else 0
            for count in route_counts
        ],
        dtype=np.int32,
    )
    active_mask = active_route_tile_counts_np > 0
    active_unique_counts = unique_counts[active_mask]
    if active_unique_counts.size == 0:
        min_unique = 0
        median_unique = 0.0
        max_unique = 0
    else:
        min_unique = int(active_unique_counts.min())
        median_unique = float(np.median(active_unique_counts))
        max_unique = int(active_unique_counts.max())

    active_projection_cache_elements = int(
        np.sum(active_route_tile_counts_np[:, None] * unique_counts)
    )
    active_projection_cache_bytes = active_projection_cache_elements * dtype_bytes
    active_route_tile_count = int(active_route_tile_counts_np.sum())
    global_max_projection_cache_bytes = (
        active_route_tile_count * k_blocks * max_unique * dtype_bytes
    )
    ready = (
        max_unique <= unique_codeword_cap
        and active_projection_cache_bytes <= projection_cache_byte_cap
    )
    if ready:
        decision = "route_active_codeword_projection_ready_for_source_probe"
        risk = (
            "actual routed expert tiles keep the full-codeword activation-side "
            "cache within the configured cap; native work still needs source and "
            "artifact parity gates."
        )
        required_next_features = (
            "source_structure_guardrail",
            "active_route_tile_codeword_dot_cache_parity",
            "air_down_group_size_352_parity",
            "same_window_q2_speed_packet_after_native_parity",
        )
        rejected_next_steps = (
            "do_not_claim_speed_from_active_route_cardinality_probe",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        )
    else:
        decision = "reject_route_active_codeword_projection_cache_too_large"
        risk = (
            "actual routed expert tiles do not reduce the full-codeword "
            "activation-side cache enough under the configured cap; this avoids "
            "spending native work on a route-projection cache that remains too "
            "large for the Air artifact."
        )
        required_next_features = (
            "materially_change_rhs_storage_contract_or_kernel_family",
            "reduce_full_codeword_cardinality_before_projection_cache",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        )
        rejected_next_steps = (
            "do_not_benchmark_route_active_codeword_projection_cache",
            "do_not_materialize_unbounded_active_route_codeword_dot_cache",
            "do_not_claim_speed_from_active_route_cardinality_probe",
        )

    return E8PRouteActiveCodewordProjectionFeasibility(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_active_codeword_projection",
        decision=decision,
        storage_constraint="compressed_e8p_route_active_codeword_projection_scales",
        dispatch_grid="active_route_tiles_x_experts_x_k_blocks",
        activation_projection_scope="active_route_tile_x_expert_kblock_full_codeword",
        experts=experts,
        output_dims=output_dims,
        input_dims=input_dims,
        group_size=group_size,
        bn=bn,
        bk=bk,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        codewords_per_bk=codewords_per_bk,
        route_count=int(route_experts_np.size),
        route_tile_size=route_tile_size,
        active_expert_count=int(np.count_nonzero(active_mask)),
        active_route_tile_count=active_route_tile_count,
        active_route_tile_counts_shape=tuple(int(dim) for dim in active_route_tile_counts_np.shape),
        active_route_tile_counts=tuple(int(value) for value in active_route_tile_counts_np),
        unique_codeword_counts_shape=tuple(int(dim) for dim in unique_counts.shape),
        min_unique_codewords_per_active_expert_kblock=min_unique,
        median_unique_codewords_per_active_expert_kblock=median_unique,
        max_unique_codewords_per_active_expert_kblock=max_unique,
        unique_codeword_cap=unique_codeword_cap,
        projection_cache_dtype=projection_cache_dtype,
        active_projection_cache_elements=active_projection_cache_elements,
        active_projection_cache_bytes=active_projection_cache_bytes,
        global_max_projection_cache_bytes=global_max_projection_cache_bytes,
        projection_cache_byte_cap=projection_cache_byte_cap,
        decoded_dense_weight_bytes=0,
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        candidate_ready_for_source_probe=ready,
        structural_risk=risk,
        required_next_features=required_next_features,
        rejected_next_steps=rejected_next_steps,
    )


def evaluate_e8p_route_abs_index_projection_feasibility(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
    route_tile_count: int = 1,
    route_tile_size: int = 64,
    projection_cache_dtype: str = "float16",
    projection_cache_byte_cap: int = 8 * 1024 * 1024,
) -> E8PRouteAbsIndexProjectionFeasibility:
    """Reject the tempting abs-index route projection without a sign axis.

    E8P route-dot reuse cannot be keyed only by the absolute row: the sign byte
    changes the signed dot product for the same abs index and depends on output
    column/codeword. Adding the missing sign axis collapses this family into the
    already-measured full-codeword projection cache.
    """

    if route_tile_count <= 0:
        raise ValueError("route_tile_count must be positive")
    if route_tile_size <= 0:
        raise ValueError("route_tile_size must be positive")
    if projection_cache_byte_cap <= 0:
        raise ValueError("projection_cache_byte_cap must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(projection_cache_dtype)
    if dtype_bytes is None:
        raise ValueError("projection_cache_dtype must be float16 or float32")

    codes_np = np.asarray(codes)
    scales_np = np.asarray(scales)
    experts, output_dims, input_dims = _validate_e8p_inputs(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )
    k_blocks = input_dims // bk
    n_tiles = _ceil_div(output_dims, bn)
    codewords_per_bk = bk // CODEWORD_DIM
    abs_indices = (codes_np >> np.uint16(8)).astype(np.uint8, copy=False)

    abs_counts = np.zeros((experts, k_blocks), dtype=np.int32)
    signed_abs_pair_counts = np.zeros_like(abs_counts)
    for expert in range(experts):
        for k_block in range(k_blocks):
            start = k_block * codewords_per_bk
            stop = start + codewords_per_bk
            block_abs = abs_indices[expert, :, start:stop].reshape(-1)
            block_codes = codes_np[expert, :, start:stop].reshape(-1)
            abs_counts[expert, k_block] = int(np.unique(block_abs).size)
            signed_abs_pair_counts[expert, k_block] = int(np.unique(block_codes).size)

    max_abs = int(abs_counts.max(initial=0))
    max_signed = int(signed_abs_pair_counts.max(initial=0))
    abs_element_count = route_tile_count * k_blocks * max_abs
    signed_element_count = route_tile_count * k_blocks * max_signed

    return E8PRouteAbsIndexProjectionFeasibility(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_abs_index_projection",
        decision="reject_route_abs_index_projection_requires_sign_axis",
        storage_constraint="compressed_e8p_route_abs_index_projection_scales",
        dispatch_grid="route_tiles_x_experts_x_k_blocks",
        activation_projection_scope="route_tile_x_expert_kblock_abs_index_without_sign_axis",
        experts=experts,
        output_dims=output_dims,
        input_dims=input_dims,
        group_size=group_size,
        bn=bn,
        bk=bk,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        codewords_per_bk=codewords_per_bk,
        route_tile_count=route_tile_count,
        route_tile_size=route_tile_size,
        abs_index_counts_shape=tuple(int(dim) for dim in abs_counts.shape),
        signed_abs_pair_counts_shape=tuple(int(dim) for dim in signed_abs_pair_counts.shape),
        min_abs_indices_per_expert_kblock=int(abs_counts.min()),
        median_abs_indices_per_expert_kblock=float(np.median(abs_counts)),
        max_abs_indices_per_expert_kblock=max_abs,
        min_signed_abs_pairs_per_expert_kblock=int(signed_abs_pair_counts.min()),
        median_signed_abs_pairs_per_expert_kblock=float(np.median(signed_abs_pair_counts)),
        max_signed_abs_pairs_per_expert_kblock=max_signed,
        projection_cache_dtype=projection_cache_dtype,
        abs_projection_cache_shape=(route_tile_count, k_blocks, max_abs),
        abs_projection_cache_bytes=abs_element_count * dtype_bytes,
        correct_signed_projection_cache_shape=(route_tile_count, k_blocks, max_signed),
        correct_signed_projection_cache_bytes=signed_element_count * dtype_bytes,
        projection_cache_byte_cap=projection_cache_byte_cap,
        abs_only_projection_exact=False,
        sign_values_depend_on_output_column=True,
        correct_cache_is_full_codeword_equivalent=True,
        decoded_dense_weight_bytes=0,
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        candidate_ready_for_source_probe=False,
        structural_risk=(
            "Abs-index route projection has bounded cardinality, but it drops "
            "the E8P sign byte. Exact signed route-dot reuse must add the sign "
            "axis, which is equivalent to the already rejected full-codeword "
            "projection cache family."
        ),
        required_next_features=(
            "materially_change_rhs_storage_contract_or_kernel_family",
            "preserve_e8p_sign_axis_or_use_full_codeword_projection",
            "avoid_full_codeword_equivalent_route_cache",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        ),
        rejected_next_steps=(
            "do_not_benchmark_route_abs_index_projection_cache",
            "do_not_ignore_sign_axis_for_e8p_route_projection",
            "do_not_claim_speed_from_abs_index_cardinality",
        ),
    )


def evaluate_e8p_route_abs_component_projection_feasibility(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
    route_tile_count: int = 1,
    route_tile_size: int = 64,
    abs_index_cap: int = 256,
    projection_cache_dtype: str = "float16",
    projection_cache_byte_cap: int = 8 * 1024 * 1024,
) -> E8PRouteAbsComponentProjectionFeasibility:
    """Measure an exact abs-index component-dot cache family.

    This keeps the output-column sign byte and scale values out of the cache:
    the cache stores activation dot products for each absolute E8P row and
    component, then a native kernel would apply signs and scales at lookup time.
    It is exact, unlike abs-index-only projection, but it is only useful if the
    component cache stays bounded on the real artifact.
    """

    if route_tile_count <= 0:
        raise ValueError("route_tile_count must be positive")
    if route_tile_size <= 0:
        raise ValueError("route_tile_size must be positive")
    if abs_index_cap <= 0:
        raise ValueError("abs_index_cap must be positive")
    if projection_cache_byte_cap <= 0:
        raise ValueError("projection_cache_byte_cap must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(projection_cache_dtype)
    if dtype_bytes is None:
        raise ValueError("projection_cache_dtype must be float16 or float32")

    codes_np = np.asarray(codes)
    scales_np = np.asarray(scales)
    experts, output_dims, input_dims = _validate_e8p_inputs(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )
    k_blocks = input_dims // bk
    n_tiles = _ceil_div(output_dims, bn)
    codewords_per_bk = bk // CODEWORD_DIM
    abs_indices = (codes_np >> np.uint16(8)).astype(np.uint8, copy=False)

    abs_counts = np.zeros((experts, k_blocks), dtype=np.int32)
    full_codeword_counts = np.zeros_like(abs_counts)
    for expert in range(experts):
        for k_block in range(k_blocks):
            start = k_block * codewords_per_bk
            stop = start + codewords_per_bk
            block_abs = abs_indices[expert, :, start:stop].reshape(-1)
            block_codes = codes_np[expert, :, start:stop].reshape(-1)
            abs_counts[expert, k_block] = int(np.unique(block_abs).size)
            full_codeword_counts[expert, k_block] = int(np.unique(block_codes).size)

    max_abs = int(abs_counts.max(initial=0))
    max_full_codewords = int(full_codeword_counts.max(initial=0))
    component_axis = CODEWORD_DIM
    projection_cache_shape = (route_tile_count, k_blocks, max_abs, component_axis)
    projection_cache_elements = route_tile_count * k_blocks * max_abs * component_axis
    projection_cache_bytes = projection_cache_elements * dtype_bytes
    full_codeword_cache_bytes = (
        route_tile_count * k_blocks * max_full_codewords * dtype_bytes
    )
    ready = (
        max_abs <= abs_index_cap
        and projection_cache_bytes <= projection_cache_byte_cap
    )
    if ready:
        decision = "route_abs_component_projection_ready_for_source_probe"
        risk = (
            "abs-index component-dot cardinality is bounded and exact because "
            "sign bytes and scales are applied after cache lookup; native work "
            "still needs source and parity gates before any timing claim."
        )
        required_next_features = (
            "source_structure_guardrail",
            "activation_abs_component_dot_cache_parity",
            "apply_sign_and_scale_after_cache_lookup",
            "air_down_group_size_352_parity",
            "same_window_q2_speed_packet_after_native_parity",
        )
        rejected_next_steps = (
            "do_not_claim_speed_from_abs_component_cardinality_probe",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        )
    else:
        decision = "reject_route_abs_component_projection_cache_too_large"
        risk = (
            "the exact abs-index component-dot cache keeps signs and scales out "
            "of storage, but its activation-side cache is still too large under "
            "the configured cap for this artifact shape."
        )
        required_next_features = (
            "materially_change_rhs_storage_contract_or_kernel_family",
            "reduce_abs_component_cardinality_before_projection_cache",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        )
        rejected_next_steps = (
            "do_not_benchmark_route_abs_component_projection_cache",
            "do_not_materialize_unbounded_abs_component_dot_cache",
            "do_not_claim_speed_from_abs_component_cardinality_probe",
        )

    return E8PRouteAbsComponentProjectionFeasibility(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_abs_component_projection",
        decision=decision,
        storage_constraint="compressed_e8p_route_abs_component_projection_scales",
        dispatch_grid="route_tiles_x_experts_x_k_blocks_x_abs_components",
        activation_projection_scope="route_tile_x_expert_kblock_abs_index_component",
        experts=experts,
        output_dims=output_dims,
        input_dims=input_dims,
        group_size=group_size,
        bn=bn,
        bk=bk,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        codewords_per_bk=codewords_per_bk,
        route_tile_count=route_tile_count,
        route_tile_size=route_tile_size,
        abs_index_counts_shape=tuple(int(dim) for dim in abs_counts.shape),
        full_codeword_counts_shape=tuple(int(dim) for dim in full_codeword_counts.shape),
        min_abs_indices_per_expert_kblock=int(abs_counts.min()),
        median_abs_indices_per_expert_kblock=float(np.median(abs_counts)),
        max_abs_indices_per_expert_kblock=max_abs,
        max_full_codewords_per_expert_kblock=max_full_codewords,
        abs_index_cap=abs_index_cap,
        component_axis=component_axis,
        projection_cache_dtype=projection_cache_dtype,
        projection_cache_shape=projection_cache_shape,
        projection_cache_bytes=projection_cache_bytes,
        full_codeword_cache_bytes=full_codeword_cache_bytes,
        projection_cache_byte_cap=projection_cache_byte_cap,
        abs_component_projection_exact=True,
        sign_values_apply_after_cache_lookup=True,
        scale_values_apply_after_cache_lookup=True,
        decoded_dense_weight_bytes=0,
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        candidate_ready_for_source_probe=ready,
        structural_risk=risk,
        required_next_features=required_next_features,
        rejected_next_steps=rejected_next_steps,
    )


def evaluate_e8p_route_abs_component_token_reuse_feasibility(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    token_count: int,
    top_k: int,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
    abs_index_cap: int = 256,
    projection_cache_dtype: str = "float16",
    projection_cache_byte_cap: int = 8 * 1024 * 1024,
) -> E8PRouteAbsComponentTokenReuseFeasibility:
    """Measure exact top-k token reuse for abs-component route projection.

    In routed MoE projection, all top-k expert routes for one token consume the
    same activation vector. A token-keyed abs-component cache preserves exact
    route-slot reconstruction through token->route indirection and reduces the
    rowwise route cache by ``top_k``. This evaluator tests whether that exact
    reuse is enough to fit the current cache budget before native work.
    """

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if abs_index_cap <= 0:
        raise ValueError("abs_index_cap must be positive")
    if projection_cache_byte_cap <= 0:
        raise ValueError("projection_cache_byte_cap must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(projection_cache_dtype)
    if dtype_bytes is None:
        raise ValueError("projection_cache_dtype must be float16 or float32")

    codes_np = np.asarray(codes)
    scales_np = np.asarray(scales)
    experts, output_dims, input_dims = _validate_e8p_inputs(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )
    k_blocks = input_dims // bk
    n_tiles = _ceil_div(output_dims, bn)
    codewords_per_bk = bk // CODEWORD_DIM
    abs_indices = (codes_np >> np.uint16(8)).astype(np.uint8, copy=False)

    abs_counts = np.zeros((experts, k_blocks), dtype=np.int32)
    for expert in range(experts):
        for k_block in range(k_blocks):
            start = k_block * codewords_per_bk
            stop = start + codewords_per_bk
            block_abs = abs_indices[expert, :, start:stop].reshape(-1)
            abs_counts[expert, k_block] = int(np.unique(block_abs).size)

    max_abs = int(abs_counts.max(initial=0))
    component_axis = CODEWORD_DIM
    route_count = token_count * top_k
    token_projection_cache_shape = (token_count, k_blocks, max_abs, component_axis)
    token_projection_cache_elements = token_count * k_blocks * max_abs * component_axis
    token_projection_cache_bytes = token_projection_cache_elements * dtype_bytes
    rowwise_projection_cache_shape = (route_count, k_blocks, max_abs, component_axis)
    rowwise_projection_cache_elements = route_count * k_blocks * max_abs * component_axis
    rowwise_projection_cache_bytes = rowwise_projection_cache_elements * dtype_bytes
    cache_reduction = (
        float(rowwise_projection_cache_bytes / token_projection_cache_bytes)
        if token_projection_cache_bytes > 0
        else 0.0
    )
    ready = (
        max_abs <= abs_index_cap
        and token_projection_cache_bytes <= projection_cache_byte_cap
    )
    if ready:
        decision = "route_abs_component_token_reuse_ready_for_source_probe"
        risk = (
            "token-keyed route projection preserves exact top-k route-slot "
            "outputs by indirection and fits the configured cache cap; native "
            "work still needs source and parity gates."
        )
        required_next_features = (
            "source_structure_guardrail",
            "token_keyed_activation_abs_component_dot_cache_parity",
            "route_slot_indirection_parity",
            "air_down_group_size_352_parity",
            "same_window_q2_speed_packet_after_native_parity",
        )
        rejected_next_steps = (
            "do_not_claim_speed_from_token_reuse_cardinality_probe",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        )
    else:
        decision = "reject_route_abs_component_token_reuse_cache_too_large"
        risk = (
            "top-k token reuse is exact and reduces the rowwise abs-component "
            "cache by the route reuse factor, but the token-keyed cache still "
            "exceeds the configured cap on this artifact shape."
        )
        required_next_features = (
            "materially_change_rhs_storage_contract_or_kernel_family",
            "find_bounded_rowwise_cardinality_mechanism_below_cache_cap",
            "preserve_route_slot_axis_or_exact_token_indirection",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        )
        rejected_next_steps = (
            "do_not_benchmark_token_reuse_abs_component_cache",
            "do_not_claim_exactness_without_route_slot_indirection",
            "do_not_claim_speed_from_token_reuse_cardinality_probe",
        )

    return E8PRouteAbsComponentTokenReuseFeasibility(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_abs_component_token_reuse",
        decision=decision,
        storage_constraint="compressed_e8p_route_abs_component_token_reuse_scales",
        dispatch_grid="tokens_x_experts_x_k_blocks_x_abs_components_plus_route_slots",
        activation_projection_scope="token_x_expert_kblock_abs_index_component_with_route_slot_indirection",
        experts=experts,
        output_dims=output_dims,
        input_dims=input_dims,
        group_size=group_size,
        bn=bn,
        bk=bk,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        codewords_per_bk=codewords_per_bk,
        token_count=token_count,
        top_k=top_k,
        route_count=route_count,
        route_slot_reuse_factor=top_k,
        abs_index_counts_shape=tuple(int(dim) for dim in abs_counts.shape),
        min_abs_indices_per_expert_kblock=int(abs_counts.min()),
        median_abs_indices_per_expert_kblock=float(np.median(abs_counts)),
        max_abs_indices_per_expert_kblock=max_abs,
        abs_index_cap=abs_index_cap,
        component_axis=component_axis,
        projection_cache_dtype=projection_cache_dtype,
        token_projection_cache_shape=token_projection_cache_shape,
        token_projection_cache_bytes=token_projection_cache_bytes,
        rowwise_projection_cache_shape=rowwise_projection_cache_shape,
        rowwise_projection_cache_bytes=rowwise_projection_cache_bytes,
        cache_reduction_vs_rowwise=cache_reduction,
        projection_cache_byte_cap=projection_cache_byte_cap,
        token_reuse_exact_for_topk_routes=True,
        route_slot_axis_preserved_by_indirection=True,
        decoded_dense_weight_bytes=0,
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        candidate_ready_for_source_probe=ready,
        structural_risk=risk,
        required_next_features=required_next_features,
        rejected_next_steps=rejected_next_steps,
    )


def evaluate_e8p_route_abs_component_token_active_expert_feasibility(
    codes: np.ndarray,
    scales: np.ndarray,
    route_experts: np.ndarray,
    *,
    token_count: int,
    top_k: int,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
    abs_index_cap: int = 256,
    projection_cache_dtype: str = "float16",
    projection_cache_byte_cap: int = 8 * 1024 * 1024,
) -> E8PRouteAbsComponentTokenActiveExpertFeasibility:
    """Measure an exact token/top-k-active-expert abs-component cache.

    This is stricter than the token-reuse probe: for each token and K block,
    cache activation dot products only for absolute E8P rows that are actually
    needed by that token's routed top-k experts. Route slots are still
    reconstructed exactly through token->route indirection and per-output sign
    and scale lookup, but the cache becomes ragged by token/K-block.
    """

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if abs_index_cap <= 0:
        raise ValueError("abs_index_cap must be positive")
    if projection_cache_byte_cap <= 0:
        raise ValueError("projection_cache_byte_cap must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(projection_cache_dtype)
    if dtype_bytes is None:
        raise ValueError("projection_cache_dtype must be float16 or float32")

    codes_np = np.asarray(codes)
    scales_np = np.asarray(scales)
    experts, output_dims, input_dims = _validate_e8p_inputs(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )
    route_experts_np = np.asarray(route_experts, dtype=np.int32)
    if route_experts_np.size != token_count * top_k:
        raise ValueError("route_experts must contain token_count * top_k entries")
    route_experts_np = route_experts_np.reshape(token_count, top_k)
    if np.any(route_experts_np < 0) or np.any(route_experts_np >= experts):
        raise ValueError("route_experts contains an expert id outside the RHS shape")

    k_blocks = input_dims // bk
    n_tiles = _ceil_div(output_dims, bn)
    codewords_per_bk = bk // CODEWORD_DIM
    abs_indices = (codes_np >> np.uint16(8)).astype(np.uint8, copy=False)

    token_kblock_counts = np.zeros((token_count, k_blocks), dtype=np.int32)
    for token in range(token_count):
        active_experts = np.unique(route_experts_np[token])
        for k_block in range(k_blocks):
            start = k_block * codewords_per_bk
            stop = start + codewords_per_bk
            block_abs = abs_indices[active_experts, :, start:stop].reshape(-1)
            token_kblock_counts[token, k_block] = int(np.unique(block_abs).size)

    max_abs = int(token_kblock_counts.max(initial=0))
    component_axis = CODEWORD_DIM
    route_count = token_count * top_k
    ragged_elements = int(token_kblock_counts.sum()) * component_axis
    ragged_bytes = ragged_elements * dtype_bytes
    max_shape = (token_count, k_blocks, max_abs, component_axis)
    max_elements = token_count * k_blocks * max_abs * component_axis
    max_bytes = max_elements * dtype_bytes
    full_shape = (token_count, k_blocks, abs_index_cap, component_axis)
    full_elements = token_count * k_blocks * abs_index_cap * component_axis
    full_bytes = full_elements * dtype_bytes
    rowwise_shape = (route_count, k_blocks, abs_index_cap, component_axis)
    rowwise_elements = route_count * k_blocks * abs_index_cap * component_axis
    rowwise_bytes = rowwise_elements * dtype_bytes
    reduction_vs_full = float(full_bytes / ragged_bytes) if ragged_bytes > 0 else 0.0
    reduction_vs_rowwise = float(rowwise_bytes / ragged_bytes) if ragged_bytes > 0 else 0.0
    ready = max_abs <= abs_index_cap and ragged_bytes <= projection_cache_byte_cap

    if ready:
        decision = "route_abs_component_token_active_expert_reuse_ready_for_source_probe"
        risk = (
            "token/top-k-active-expert abs-component caching preserves exact "
            "route-slot outputs by indirection and fits the configured cache "
            "cap; native work still needs ragged lookup/source/parity gates."
        )
        required_next_features = (
            "source_structure_guardrail",
            "ragged_token_active_abs_component_dot_cache_parity",
            "route_slot_indirection_parity",
            "air_down_group_size_352_parity",
            "same_window_q2_speed_packet_after_native_parity",
        )
        rejected_next_steps = (
            "do_not_claim_speed_from_token_active_expert_cardinality_probe",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        )
    else:
        decision = "reject_route_abs_component_token_active_expert_cache_too_large"
        risk = (
            "scoping the exact abs-component cache to each token's active top-k "
            "experts reduces cardinality versus the full token cache, but the "
            "ragged cache still exceeds the configured cap on this artifact shape."
        )
        required_next_features = (
            "materially_change_rhs_storage_contract_or_kernel_family",
            "find_stronger_bounded_rowwise_cardinality_mechanism_below_cache_cap",
            "preserve_route_slot_axis_or_exact_token_indirection",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        )
        rejected_next_steps = (
            "do_not_benchmark_token_active_expert_abs_component_cache",
            "do_not_claim_exactness_without_route_slot_indirection",
            "do_not_claim_speed_from_token_active_expert_cardinality_probe",
        )

    return E8PRouteAbsComponentTokenActiveExpertFeasibility(
        target_kernel_family=(
            "sorted_gather_qmm_rhs_nax_route_abs_component_token_active_expert_reuse"
        ),
        decision=decision,
        storage_constraint=(
            "compressed_e8p_route_abs_component_token_active_expert_reuse_scales"
        ),
        dispatch_grid="tokens_x_topk_active_experts_x_k_blocks_x_abs_components_plus_route_slots",
        activation_projection_scope=(
            "token_x_active_expert_union_kblock_abs_index_component_with_route_slot_indirection"
        ),
        experts=experts,
        output_dims=output_dims,
        input_dims=input_dims,
        group_size=group_size,
        bn=bn,
        bk=bk,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        codewords_per_bk=codewords_per_bk,
        token_count=token_count,
        top_k=top_k,
        route_count=route_count,
        active_expert_ids_shape=tuple(int(dim) for dim in route_experts_np.shape),
        token_kblock_abs_index_count_shape=tuple(int(dim) for dim in token_kblock_counts.shape),
        min_abs_indices_per_token_kblock=int(token_kblock_counts.min()),
        median_abs_indices_per_token_kblock=float(np.median(token_kblock_counts)),
        p95_abs_indices_per_token_kblock=float(np.percentile(token_kblock_counts, 95)),
        max_abs_indices_per_token_kblock=max_abs,
        abs_index_cap=abs_index_cap,
        component_axis=component_axis,
        projection_cache_dtype=projection_cache_dtype,
        ragged_token_active_projection_cache_elements=ragged_elements,
        ragged_token_active_projection_cache_bytes=ragged_bytes,
        max_token_active_projection_cache_shape=max_shape,
        max_token_active_projection_cache_bytes=max_bytes,
        full_token_projection_cache_shape=full_shape,
        full_token_projection_cache_bytes=full_bytes,
        rowwise_projection_cache_shape=rowwise_shape,
        rowwise_projection_cache_bytes=rowwise_bytes,
        cache_reduction_vs_full_token_cache=reduction_vs_full,
        cache_reduction_vs_rowwise=reduction_vs_rowwise,
        projection_cache_byte_cap=projection_cache_byte_cap,
        token_active_expert_reuse_exact_for_topk_routes=True,
        route_slot_axis_preserved_by_indirection=True,
        active_expert_abs_union_scoped_to_token_topk=True,
        decoded_dense_weight_bytes=0,
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        candidate_ready_for_source_probe=ready,
        structural_risk=risk,
        required_next_features=required_next_features,
        rejected_next_steps=rejected_next_steps,
    )


def evaluate_e8p_route_abs_component_lower_bound_feasibility(
    codes: np.ndarray,
    scales: np.ndarray,
    route_experts: np.ndarray,
    *,
    token_count: int,
    top_k: int,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
    abs_index_cap: int = 256,
    projection_cache_dtype: str = "float16",
    projection_cache_byte_cap: int = 8 * 1024 * 1024,
) -> E8PRouteAbsComponentLowerBoundFeasibility:
    """Compute the exact lower bound for abs-component route projection caches.

    Any exact abs-component route-projection family that applies E8P signs and
    scales after activation-dot lookup still needs every abs row used by each
    token's top-k active experts for every K block. This lower bound is
    independent of route tile size and is the cheapest exact cache before
    native source work.
    """

    active = evaluate_e8p_route_abs_component_token_active_expert_feasibility(
        codes,
        scales,
        route_experts,
        token_count=token_count,
        top_k=top_k,
        group_size=group_size,
        bn=bn,
        bk=bk,
        abs_index_cap=abs_index_cap,
        projection_cache_dtype=projection_cache_dtype,
        projection_cache_byte_cap=projection_cache_byte_cap,
    )
    ratio_to_cap = active.ragged_token_active_projection_cache_bytes / projection_cache_byte_cap
    ready = active.ragged_token_active_projection_cache_bytes <= projection_cache_byte_cap
    if ready:
        decision = "route_abs_component_exact_lower_bound_under_cache_cap"
        risk = (
            "the exact abs-component cache lower bound fits the configured cap; "
            "native source, ragged lookup, and q2 speed gates are still required."
        )
        required_next_features = (
            "source_structure_guardrail",
            "exact_lower_bound_abs_component_cache_parity",
            "route_slot_indirection_parity",
            "air_down_group_size_352_parity",
            "same_window_q2_speed_packet_after_native_parity",
        )
        rejected_next_steps = (
            "do_not_claim_speed_from_lower_bound_cardinality_probe",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        )
    else:
        decision = "reject_exact_route_abs_component_projection_lower_bound_over_cap"
        risk = (
            "the cheapest exact abs-component route-projection cache exceeds "
            "the configured cap, so further variants in this family need a "
            "different RHS layout/kernel family or a new exact compression "
            "argument rather than another timing run."
        )
        required_next_features = (
            "materially_change_rhs_storage_contract_or_kernel_family",
            "prove_new_exact_compression_below_abs_component_lower_bound",
            "preserve_route_slot_axis_or_exact_token_indirection",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        )
        rejected_next_steps = (
            "do_not_benchmark_exact_route_abs_component_projection_family",
            "do_not_add_more_route_abs_component_cache_accounting_variants",
            "do_not_claim_speed_from_lower_bound_cardinality_probe",
        )

    return E8PRouteAbsComponentLowerBoundFeasibility(
        target_kernel_family="sorted_gather_qmm_rhs_nax_route_abs_component_exact_lower_bound",
        decision=decision,
        storage_constraint="compressed_e8p_route_abs_component_exact_lower_bound",
        lower_bound_scope="token_topk_active_expert_kblock_abs_index_component_union",
        families_blocked_by_lower_bound=(
            "route_abs_component_projection",
            "route_abs_component_tile_cache",
            "route_abs_component_token_reuse",
            "route_abs_component_token_active_expert_reuse",
        ),
        experts=active.experts,
        output_dims=active.output_dims,
        input_dims=active.input_dims,
        group_size=active.group_size,
        bn=active.bn,
        bk=active.bk,
        n_tiles=active.n_tiles,
        k_blocks=active.k_blocks,
        codewords_per_bk=active.codewords_per_bk,
        token_count=active.token_count,
        top_k=active.top_k,
        route_count=active.route_count,
        token_kblock_abs_index_count_shape=active.token_kblock_abs_index_count_shape,
        min_abs_indices_per_token_kblock=active.min_abs_indices_per_token_kblock,
        median_abs_indices_per_token_kblock=active.median_abs_indices_per_token_kblock,
        p95_abs_indices_per_token_kblock=active.p95_abs_indices_per_token_kblock,
        max_abs_indices_per_token_kblock=active.max_abs_indices_per_token_kblock,
        abs_index_cap=active.abs_index_cap,
        component_axis=active.component_axis,
        projection_cache_dtype=active.projection_cache_dtype,
        exact_lower_bound_cache_elements=active.ragged_token_active_projection_cache_elements,
        exact_lower_bound_cache_bytes=active.ragged_token_active_projection_cache_bytes,
        projection_cache_byte_cap=active.projection_cache_byte_cap,
        lower_bound_ratio_to_cap=ratio_to_cap,
        rowwise_projection_cache_bytes=active.rowwise_projection_cache_bytes,
        reduction_vs_rowwise=active.cache_reduction_vs_rowwise,
        lower_bound_preserves_route_slot_exactness=True,
        lower_bound_preserves_e8p_sign_axis=True,
        lower_bound_requires_no_decoded_rhs_storage=True,
        decoded_dense_weight_bytes=0,
        preserves_compressed_rhs_storage=True,
        preserves_codeword_scale_slots=True,
        candidate_ready_for_source_probe=ready,
        structural_risk=risk,
        required_next_features=required_next_features,
        rejected_next_steps=rejected_next_steps,
    )


def build_e8p_component_stream_contract(
    packed: E8PComponentStreamRHSTiles,
    *,
    route_tile_count: int,
    component_pair_size: int,
) -> E8PComponentStreamContract:
    """Describe the dispatch grid for a material next-family component stream."""

    if route_tile_count <= 0:
        raise ValueError("route_tile_count must be positive")
    if component_pair_size <= 0:
        raise ValueError("component_pair_size must be positive")

    layout = packed.layout
    component_count = layout.codewords_per_bk * CODEWORD_DIM
    pair_count = _ceil_div(component_count, component_pair_size)
    n_tiles = tuple(range(layout.n_tiles))
    workgroups: list[E8PComponentStreamWorkgroup] = []
    for route_tile in range(route_tile_count):
        for expert in range(layout.experts):
            for k_block in range(layout.k_blocks):
                for component_pair in range(pair_count):
                    start = component_pair * component_pair_size
                    stop = min(component_count, start + component_pair_size)
                    component_offsets = tuple(range(start, stop))
                    workgroups.append(
                        E8PComponentStreamWorkgroup(
                            route_tile=route_tile,
                            expert=expert,
                            k_block=k_block,
                            component_pair=component_pair,
                            n_tiles=n_tiles,
                            component_offsets=component_offsets,
                            codeword_indices=tuple(
                                int(value)
                                for value in packed.component_codeword_indices[
                                    k_block, start:stop
                                ]
                            ),
                            codeword_component_offsets=tuple(
                                int(value)
                                for value in packed.component_offsets[k_block, start:stop]
                            ),
                            scale_slots=tuple(
                                int(value)
                                for value in packed.component_scale_slots[k_block, start:stop]
                            ),
                        )
                    )

    return E8PComponentStreamContract(
        target_kernel_family=layout.target_kernel_family,
        storage_constraint=layout.storage_constraint,
        dispatch_grid="route_tiles_x_experts_x_k_blocks_x_component_pairs",
        decode_reuse_scope="component_stream_before_accumulation",
        rhs_source="streamed_component_outer_product_no_b_fragment",
        route_tile_count=route_tile_count,
        k_blocks=layout.k_blocks,
        n_tiles=layout.n_tiles,
        component_count_per_kblock=component_count,
        component_pair_size=component_pair_size,
        component_pair_count=pair_count,
        decoded_dense_weight_bytes=layout.decoded_dense_weight_bytes,
        required_before_native_parity=(
            "native_component_stream_scalar_oracle",
            "air_down_group_size_352_component_scale_parity",
            "source_structure_guardrail",
        ),
        rejected_next_steps=(
            "do_not_fill_tensorops_b_fragment_from_e8p_values",
            "do_not_materialize_decoded_dense_rhs",
            "do_not_claim_speed_before_same_window_q2_packet",
        ),
        workgroups=tuple(workgroups),
    )


def build_e8p_component_stream_partial_reduction_contract(
    dispatch: E8PComponentStreamContract,
    *,
    route_tile_size: int,
    accumulation_dtype: str = "float32",
) -> E8PComponentStreamPartialReductionContract:
    if route_tile_size <= 0:
        raise ValueError("route_tile_size must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(accumulation_dtype)
    if dtype_bytes is None:
        raise ValueError("accumulation_dtype must be float16 or float32")

    shape = (
        dispatch.route_tile_count,
        dispatch.k_blocks,
        dispatch.component_pair_count,
        dispatch.n_tiles,
        route_tile_size,
        64,
    )
    element_count = 1
    for dim in shape:
        element_count *= dim

    return E8PComponentStreamPartialReductionContract(
        target_kernel_family=dispatch.target_kernel_family,
        partial_accumulation_grid="route_tiles_x_k_blocks_x_component_pairs_x_n_tiles",
        final_reduction_grid="route_tiles_x_n_tiles",
        partial_accumulation_shape=shape,
        partial_accumulation_dtype=accumulation_dtype,
        partial_accumulation_element_count=element_count,
        partial_accumulation_bytes=element_count * dtype_bytes,
        route_tile_size=route_tile_size,
        reduction_axes=("k_blocks", "component_pairs"),
        output_shape="route_count_x_output_dims",
        valid_routes_source="route_tile_counts",
        expert_source="route_tile_experts",
        decoded_dense_weight_bytes=dispatch.decoded_dense_weight_bytes,
        required_next_features=(
            "allocate_component_pair_partial_scratch",
            "write_one_partial_per_component_pair_workgroup",
            "reduce_kblock_and_component_pair_partials_before_final_output",
            "preserve_compressed_component_stream_storage",
            "prove_speed_path_against_decoded_sorted_reference",
        ),
        rejected_next_steps=(
            "do_not_benchmark_scalar_component_stream_oracle",
            "do_not_sum_component_pairs_inside_single_output_thread",
            "do_not_promote_without_same_window_q2_speed_packet",
        ),
    )


def build_e8p_component_stream_shared_decode_contract(
    dispatch: E8PComponentStreamContract,
    *,
    route_tile_size: int,
    component_decode_dtype: str = "float16",
) -> E8PComponentStreamSharedDecodeContract:
    """Describe the post-rejection component-stream shared decode family."""

    if route_tile_size <= 0:
        raise ValueError("route_tile_size must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(component_decode_dtype)
    if dtype_bytes is None:
        raise ValueError("component_decode_dtype must be float16 or float32")

    expert_count = 0
    for workgroup in dispatch.workgroups:
        expert_count = max(expert_count, workgroup.expert + 1)
    shape = (
        dispatch.route_tile_count,
        expert_count,
        dispatch.k_blocks,
        dispatch.component_pair_count,
        route_tile_size,
        dispatch.component_pair_size,
    )
    element_count = 1
    for dim in shape:
        element_count *= dim

    return E8PComponentStreamSharedDecodeContract(
        target_kernel_family="sorted_gather_qmm_rhs_nax_shared_decode",
        dispatch_grid=dispatch.dispatch_grid,
        decode_reuse_scope="component_pair_kblock_shared_across_n_tiles",
        shared_decode_shape=shape,
        shared_decode_dtype=component_decode_dtype,
        shared_decode_bytes=element_count * dtype_bytes,
        output_accumulation_grid="route_tiles_x_n_tiles",
        decoded_dense_weight_bytes=dispatch.decoded_dense_weight_bytes,
        preserved_storage_contract=dispatch.storage_constraint,
        required_next_features=(
            "native_shared_decode_cache_source_guardrail",
            "prove_shared_decode_cache_matches_component_stream_oracle",
            "prove_artifact_shared_decode_cache_parity",
            "same_window_q2_speed_packet_after_parity_only",
        ),
        rejected_next_steps=(
            "do_not_retime_component_stream_tensorops_unchanged",
            "do_not_fill_tensorops_b_fragment_from_e8p_values",
            "do_not_use_scalar_component_stream_partial_body_as_speed_path",
            "do_not_materialize_decoded_dense_rhs",
        ),
    )


def evaluate_e8p_component_stream_shared_decode_feasibility(
    dispatch: E8PComponentStreamContract,
    *,
    route_tile_size: int,
    component_decode_dtype: str = "float16",
) -> E8PComponentStreamSharedDecodeFeasibility:
    """Explain why component-stream decode cannot be shared across N tiles."""

    if route_tile_size <= 0:
        raise ValueError("route_tile_size must be positive")
    dtype_bytes = {
        "float16": 2,
        "float32": 4,
    }.get(component_decode_dtype)
    if dtype_bytes is None:
        raise ValueError("component_decode_dtype must be float16 or float32")

    expert_count = 0
    for workgroup in dispatch.workgroups:
        expert_count = max(expert_count, workgroup.expert + 1)
    correct_cache_shape = (
        expert_count,
        dispatch.n_tiles,
        dispatch.k_blocks,
        route_tile_size,
        dispatch.component_count_per_kblock,
    )
    element_count = 1
    for dim in correct_cache_shape:
        element_count *= dim

    return E8PComponentStreamSharedDecodeFeasibility(
        target_kernel_family="sorted_gather_qmm_rhs_nax_shared_decode",
        decision="reject_component_stream_shared_decode_cross_ntile_cache",
        n_tile_reuse_feasible=False,
        component_values_depend_on_output_column=True,
        scale_values_depend_on_output_column=True,
        correct_cache_shape=correct_cache_shape,
        correct_cache_dtype=component_decode_dtype,
        correct_cache_bytes=element_count * dtype_bytes,
        correct_cache_is_decoded_rhs_equivalent=True,
        structural_risk=(
            "component-stream signs, abs indices, and scales are keyed by "
            "expert/n_tile/k_block/output_column/codeword, so a cache reused "
            "across N tiles is not correct; adding the missing n_tile and "
            "output-column axes produces decoded-RHS-equivalent storage"
        ),
        next_track_b_hypothesis="change_component_stream_rhs_layout_or_kernel_family",
        rejected_next_steps=(
            "do_not_benchmark_scalar_shared_decode_cache_scaffold",
            "do_not_claim_cross_ntile_shared_decode_from_component_stream_storage",
            "do_not_materialize_decoded_rhs_equivalent_cache",
            "do_not_retime_component_stream_tensorops_unchanged",
        ),
    )


def pack_e8p_sign_nibble_abs_index_rhs_tiles(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
) -> E8PSignNibbleAbsIndexRHSTiles:
    """Pack E8P RHS tiles as low/high sign nibbles plus abs-index bytes.

    This keeps the q2-shaped RHS tile geometry while replacing whole sign bytes
    with two 4-bit factors. It is a native-facing contract for sign-mask decode
    reuse experiments after whole-sign-byte factor reuse was rejected.
    """

    split = pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=group_size, bn=bn, bk=bk)
    layout = E8PPackedRHSLayout(
        target_kernel_family="sorted_gather_qmm_rhs_nax_sign_nibble_abs_index",
        storage_constraint="compressed_e8p_sign_nibbles_abs_index_scales",
        experts=split.layout.experts,
        output_dims=split.layout.output_dims,
        input_dims=split.layout.input_dims,
        group_size=split.layout.group_size,
        bn=split.layout.bn,
        bk=split.layout.bk,
        n_tiles=split.layout.n_tiles,
        k_blocks=split.layout.k_blocks,
        codewords_per_bk=split.layout.codewords_per_bk,
        max_scale_groups_per_bk=split.layout.max_scale_groups_per_bk,
        decoded_dense_weight_bytes=0,
    )
    return E8PSignNibbleAbsIndexRHSTiles(
        layout=layout,
        sign_low_nibble_tiles=split.sign_tiles & np.uint8(0x0F),
        sign_high_nibble_tiles=split.sign_tiles >> np.uint8(4),
        abs_index_tiles=split.abs_index_tiles,
        parity_tiles=split.parity_tiles,
        scale_tiles=split.scale_tiles,
        scale_group_indices=split.scale_group_indices,
        codeword_scale_slots=split.codeword_scale_slots,
    )


def pack_e8p_sign_nibble_micro_lut_rhs_tiles(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
) -> E8PSignNibbleMicroLUTRHSTiles:
    """Pack sign-nibble RHS tiles as fixed nibble LUTs plus abs-index slots.

    This is the next native-facing contract after the raw sign-nibble TensorOps
    path was rejected for per-fragment nibble/abs reconstruction. Low/high sign
    nibble factors are represented as fixed 16-entry LUTs with uint8 slot maps;
    abs indices use per-tile LUTs and slot maps. The compact scale map and
    per-codeword scale slots are preserved.
    """

    packed = pack_e8p_sign_nibble_abs_index_rhs_tiles(
        codes,
        scales,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )
    abs_lut, abs_slots, abs_counts = _factor_lut_and_slots(packed.abs_index_tiles)
    layout = E8PPackedRHSLayout(
        target_kernel_family="sorted_gather_qmm_rhs_nax_sign_nibble_micro_lut",
        storage_constraint="compressed_e8p_sign_nibble_micro_lut_scales",
        experts=packed.layout.experts,
        output_dims=packed.layout.output_dims,
        input_dims=packed.layout.input_dims,
        group_size=packed.layout.group_size,
        bn=packed.layout.bn,
        bk=packed.layout.bk,
        n_tiles=packed.layout.n_tiles,
        k_blocks=packed.layout.k_blocks,
        codewords_per_bk=packed.layout.codewords_per_bk,
        max_scale_groups_per_bk=packed.layout.max_scale_groups_per_bk,
        decoded_dense_weight_bytes=0,
    )
    return E8PSignNibbleMicroLUTRHSTiles(
        layout=layout,
        sign_low_nibble_lut=np.arange(16, dtype=np.uint8),
        sign_low_nibble_slots=packed.sign_low_nibble_tiles.astype(np.uint8, copy=True),
        sign_high_nibble_lut=np.arange(16, dtype=np.uint8),
        sign_high_nibble_slots=packed.sign_high_nibble_tiles.astype(np.uint8, copy=True),
        abs_index_lut=abs_lut,
        abs_index_slots=abs_slots,
        abs_index_counts=abs_counts,
        scale_tiles=packed.scale_tiles,
        scale_group_indices=packed.scale_group_indices,
        codeword_scale_slots=packed.codeword_scale_slots,
    )


def pack_e8p_sign_plane_abs_index_rhs_tiles(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
) -> E8PSignPlaneAbsIndexRHSTiles:
    """Pack E8P RHS tiles as per-codeword sign-bit column masks plus abs indices."""

    if bn > 64:
        raise ValueError("sign-plane layout requires bn <= 64")
    split = pack_e8p_split_byte_rhs_tiles(codes, scales, group_size=group_size, bn=bn, bk=bk)
    layout = E8PPackedRHSLayout(
        target_kernel_family="sorted_gather_qmm_rhs_nax_sign_plane_abs_index",
        storage_constraint="compressed_e8p_sign_planes_abs_index_scales",
        experts=split.layout.experts,
        output_dims=split.layout.output_dims,
        input_dims=split.layout.input_dims,
        group_size=split.layout.group_size,
        bn=split.layout.bn,
        bk=split.layout.bk,
        n_tiles=split.layout.n_tiles,
        k_blocks=split.layout.k_blocks,
        codewords_per_bk=split.layout.codewords_per_bk,
        max_scale_groups_per_bk=split.layout.max_scale_groups_per_bk,
        decoded_dense_weight_bytes=0,
    )
    return E8PSignPlaneAbsIndexRHSTiles(
        layout=layout,
        sign_bit_planes=_sign_bit_planes_from_sign_tiles(split.sign_tiles),
        abs_index_tiles=split.abs_index_tiles,
        scale_tiles=split.scale_tiles,
        scale_group_indices=split.scale_group_indices,
        codeword_scale_slots=split.codeword_scale_slots,
    )


def decode_e8p_rhs_tile(
    packed: E8PPackedRHSTiles,
    *,
    expert: int,
    n_tile: int,
    k_block: int,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one packed RHS tile for parity tests and native-kernel oracles."""

    layout = packed.layout
    if not (0 <= expert < layout.experts):
        raise ValueError("expert out of range")
    if not (0 <= n_tile < layout.n_tiles):
        raise ValueError("n_tile out of range")
    if not (0 <= k_block < layout.k_blocks):
        raise ValueError("k_block out of range")
    out_start = n_tile * layout.bn
    out_count = min(layout.bn, layout.output_dims - out_start)
    codeword_count = min(
        layout.codewords_per_bk,
        (layout.input_dims - k_block * layout.bk) // CODEWORD_DIM,
    )
    codes = packed.code_tiles[expert, n_tile, k_block, :out_count, :codeword_count]
    code_vectors = decode_e8p(codes, codebook)
    scale_slots = packed.codeword_scale_slots[k_block, :codeword_count]
    tile_scales = packed.scale_tiles[expert, n_tile, k_block, :out_count, :]
    scales = np.take(tile_scales, scale_slots, axis=1).astype(np.float32)
    return (code_vectors * scales[:, :, None]).reshape(out_count, codeword_count * CODEWORD_DIM)


def decode_e8p_split_byte_rhs_tile(
    packed: E8PSplitByteRHSTiles,
    *,
    expert: int,
    n_tile: int,
    k_block: int,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one split-byte RHS tile for parity tests and native-kernel oracles."""

    layout = packed.layout
    if not (0 <= expert < layout.experts):
        raise ValueError("expert out of range")
    if not (0 <= n_tile < layout.n_tiles):
        raise ValueError("n_tile out of range")
    if not (0 <= k_block < layout.k_blocks):
        raise ValueError("k_block out of range")
    out_start = n_tile * layout.bn
    out_count = min(layout.bn, layout.output_dims - out_start)
    codeword_count = min(
        layout.codewords_per_bk,
        (layout.input_dims - k_block * layout.bk) // CODEWORD_DIM,
    )
    signs = packed.sign_tiles[expert, n_tile, k_block, :out_count, :codeword_count].astype(np.uint16)
    abs_indices = packed.abs_index_tiles[
        expert, n_tile, k_block, :out_count, :codeword_count
    ].astype(np.uint16)
    codes = signs | (abs_indices << np.uint16(8))
    code_vectors = decode_e8p(codes, codebook)
    scale_slots = packed.codeword_scale_slots[k_block, :codeword_count]
    tile_scales = packed.scale_tiles[expert, n_tile, k_block, :out_count, :]
    scales = np.take(tile_scales, scale_slots, axis=1).astype(np.float32)
    return (code_vectors * scales[:, :, None]).reshape(out_count, codeword_count * CODEWORD_DIM)


def decode_e8p_split_byte_factor_reuse_rhs_tile(
    packed: E8PSplitByteFactorReuseRHSTiles,
    *,
    expert: int,
    n_tile: int,
    k_block: int,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one factor-reuse RHS tile for parity tests and native-kernel oracles."""

    layout = packed.layout
    if not (0 <= expert < layout.experts):
        raise ValueError("expert out of range")
    if not (0 <= n_tile < layout.n_tiles):
        raise ValueError("n_tile out of range")
    if not (0 <= k_block < layout.k_blocks):
        raise ValueError("k_block out of range")
    out_start = n_tile * layout.bn
    out_count = min(layout.bn, layout.output_dims - out_start)
    codeword_count = min(
        layout.codewords_per_bk,
        (layout.input_dims - k_block * layout.bk) // CODEWORD_DIM,
    )
    sign_slots = packed.sign_byte_slots[expert, n_tile, k_block, :out_count, :codeword_count]
    abs_slots = packed.abs_index_slots[expert, n_tile, k_block, :out_count, :codeword_count]
    signs = packed.sign_byte_lut[expert, n_tile, k_block, sign_slots].astype(np.uint16)
    abs_indices = packed.abs_index_lut[expert, n_tile, k_block, abs_slots].astype(np.uint16)
    codes = signs | (abs_indices << np.uint16(8))
    code_vectors = decode_e8p(codes, codebook)
    scale_slots = packed.codeword_scale_slots[k_block, :codeword_count]
    tile_scales = packed.scale_tiles[expert, n_tile, k_block, :out_count, :]
    scales = np.take(tile_scales, scale_slots, axis=1).astype(np.float32)
    return (code_vectors * scales[:, :, None]).reshape(out_count, codeword_count * CODEWORD_DIM)


def decode_e8p_expert_kblock_factor_reuse_rhs_tile(
    packed: E8PExpertKBlockFactorReuseRHSTiles,
    *,
    expert: int,
    n_tile: int,
    k_block: int,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one expert/K-block factor-reuse RHS tile for parity tests."""

    layout = packed.layout
    if not (0 <= expert < layout.experts):
        raise ValueError("expert out of range")
    if not (0 <= n_tile < layout.n_tiles):
        raise ValueError("n_tile out of range")
    if not (0 <= k_block < layout.k_blocks):
        raise ValueError("k_block out of range")
    out_start = n_tile * layout.bn
    out_count = min(layout.bn, layout.output_dims - out_start)
    codeword_count = min(
        layout.codewords_per_bk,
        (layout.input_dims - k_block * layout.bk) // CODEWORD_DIM,
    )
    sign_slots = packed.sign_byte_slots[expert, n_tile, k_block, :out_count, :codeword_count]
    abs_slots = packed.abs_index_slots[expert, n_tile, k_block, :out_count, :codeword_count]
    signs = packed.sign_byte_lut[expert, k_block, sign_slots].astype(np.uint16)
    abs_indices = packed.abs_index_lut[expert, k_block, abs_slots].astype(np.uint16)
    codes = signs | (abs_indices << np.uint16(8))
    code_vectors = decode_e8p(codes, codebook)
    scale_slots = packed.codeword_scale_slots[k_block, :codeword_count]
    tile_scales = packed.scale_tiles[expert, n_tile, k_block, :out_count, :]
    scales = np.take(tile_scales, scale_slots, axis=1).astype(np.float32)
    return (code_vectors * scales[:, :, None]).reshape(out_count, codeword_count * CODEWORD_DIM)


def decode_e8p_component_stream_rhs_tile(
    packed: E8PComponentStreamRHSTiles,
    *,
    expert: int,
    n_tile: int,
    k_block: int,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one component-stream RHS tile for parity tests."""

    layout = packed.layout
    if not (0 <= expert < layout.experts):
        raise ValueError("expert out of range")
    if not (0 <= n_tile < layout.n_tiles):
        raise ValueError("n_tile out of range")
    if not (0 <= k_block < layout.k_blocks):
        raise ValueError("k_block out of range")
    out_start = n_tile * layout.bn
    out_count = min(layout.bn, layout.output_dims - out_start)
    codeword_count = min(
        layout.codewords_per_bk,
        (layout.input_dims - k_block * layout.bk) // CODEWORD_DIM,
    )
    bits = packed.sign_component_bits[
        expert, n_tile, k_block, :out_count, :codeword_count, :
    ].astype(np.uint16)
    bit_weights = (np.uint16(1) << np.arange(CODEWORD_DIM, dtype=np.uint16)).reshape(
        1,
        1,
        CODEWORD_DIM,
    )
    signs = np.sum(bits * bit_weights, axis=-1, dtype=np.uint16)
    abs_indices = packed.abs_index_tiles[
        expert, n_tile, k_block, :out_count, :codeword_count
    ].astype(np.uint16)
    codes = signs | (abs_indices << np.uint16(8))
    code_vectors = decode_e8p(codes, codebook)
    scale_slots = packed.codeword_scale_slots[k_block, :codeword_count]
    tile_scales = packed.scale_tiles[expert, n_tile, k_block, :out_count, :]
    scales = np.take(tile_scales, scale_slots, axis=1).astype(np.float32)
    return (code_vectors * scales[:, :, None]).reshape(out_count, codeword_count * CODEWORD_DIM)


def component_stream_sorted_matmul_oracle(
    packed: E8PComponentStreamRHSTiles,
    sorted_x: np.ndarray,
    sorted_experts: np.ndarray,
    *,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Route-level scalar oracle for the component-stream RHS contract.

    The implementation deliberately accumulates one compressed RHS tile at a
    time. It is a native-source oracle for future component-stream kernels, not
    a fast path and not a dense-RHS materializer.
    """

    layout = packed.layout
    x_np = np.asarray(sorted_x)
    experts_np = np.asarray(sorted_experts)
    if x_np.ndim != 2:
        raise ValueError("sorted_x must be 2D [routes, input_dims]")
    if x_np.shape[1] != layout.input_dims:
        raise ValueError(f"sorted_x input dimension must be {layout.input_dims}")
    if experts_np.ndim != 1:
        raise ValueError("sorted_experts must be 1D [routes]")
    if experts_np.shape[0] != x_np.shape[0]:
        raise ValueError("sorted_x and sorted_experts must have the same route count")
    if experts_np.size and (int(experts_np.min()) < 0 or int(experts_np.max()) >= layout.experts):
        raise ValueError("sorted_experts contains an expert id out of range")

    out = np.zeros((x_np.shape[0], layout.output_dims), dtype=np.float32)
    for route, expert_value in enumerate(experts_np):
        expert = int(expert_value)
        for n_tile in range(layout.n_tiles):
            out_start = n_tile * layout.bn
            out_stop = min(layout.output_dims, out_start + layout.bn)
            for k_block in range(layout.k_blocks):
                k_start = k_block * layout.bk
                k_stop = min(layout.input_dims, k_start + layout.bk)
                tile = decode_e8p_component_stream_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook,
                )
                out[route, out_start:out_stop] += (
                    x_np[route, k_start:k_stop].astype(np.float32)
                    @ tile.T.astype(np.float32)
                )
    return out


def decode_e8p_sign_nibble_abs_index_rhs_tile(
    packed: E8PSignNibbleAbsIndexRHSTiles,
    *,
    expert: int,
    n_tile: int,
    k_block: int,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one sign-nibble RHS tile for parity tests and native oracles."""

    layout = packed.layout
    if not (0 <= expert < layout.experts):
        raise ValueError("expert out of range")
    if not (0 <= n_tile < layout.n_tiles):
        raise ValueError("n_tile out of range")
    if not (0 <= k_block < layout.k_blocks):
        raise ValueError("k_block out of range")
    out_start = n_tile * layout.bn
    out_count = min(layout.bn, layout.output_dims - out_start)
    codeword_count = min(
        layout.codewords_per_bk,
        (layout.input_dims - k_block * layout.bk) // CODEWORD_DIM,
    )
    low = packed.sign_low_nibble_tiles[
        expert, n_tile, k_block, :out_count, :codeword_count
    ].astype(np.uint16)
    high = packed.sign_high_nibble_tiles[
        expert, n_tile, k_block, :out_count, :codeword_count
    ].astype(np.uint16)
    abs_indices = packed.abs_index_tiles[
        expert, n_tile, k_block, :out_count, :codeword_count
    ].astype(np.uint16)
    signs = low | (high << np.uint16(4))
    codes = signs | (abs_indices << np.uint16(8))
    code_vectors = decode_e8p(codes, codebook)
    scale_slots = packed.codeword_scale_slots[k_block, :codeword_count]
    tile_scales = packed.scale_tiles[expert, n_tile, k_block, :out_count, :]
    scales = np.take(tile_scales, scale_slots, axis=1).astype(np.float32)
    return (code_vectors * scales[:, :, None]).reshape(out_count, codeword_count * CODEWORD_DIM)


def decode_e8p_sign_nibble_micro_lut_rhs_tile(
    packed: E8PSignNibbleMicroLUTRHSTiles,
    *,
    expert: int,
    n_tile: int,
    k_block: int,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one sign-nibble micro-LUT RHS tile for parity tests."""

    layout = packed.layout
    if not (0 <= expert < layout.experts):
        raise ValueError("expert out of range")
    if not (0 <= n_tile < layout.n_tiles):
        raise ValueError("n_tile out of range")
    if not (0 <= k_block < layout.k_blocks):
        raise ValueError("k_block out of range")
    out_start = n_tile * layout.bn
    out_count = min(layout.bn, layout.output_dims - out_start)
    codeword_count = min(
        layout.codewords_per_bk,
        (layout.input_dims - k_block * layout.bk) // CODEWORD_DIM,
    )
    low_slots = packed.sign_low_nibble_slots[
        expert, n_tile, k_block, :out_count, :codeword_count
    ]
    high_slots = packed.sign_high_nibble_slots[
        expert, n_tile, k_block, :out_count, :codeword_count
    ]
    abs_slots = packed.abs_index_slots[
        expert, n_tile, k_block, :out_count, :codeword_count
    ]
    low = packed.sign_low_nibble_lut[low_slots].astype(np.uint16)
    high = packed.sign_high_nibble_lut[high_slots].astype(np.uint16)
    abs_indices = packed.abs_index_lut[expert, n_tile, k_block, abs_slots].astype(np.uint16)
    signs = low | (high << np.uint16(4))
    codes = signs | (abs_indices << np.uint16(8))
    code_vectors = decode_e8p(codes, codebook)
    scale_slots = packed.codeword_scale_slots[k_block, :codeword_count]
    tile_scales = packed.scale_tiles[expert, n_tile, k_block, :out_count, :]
    scales = np.take(tile_scales, scale_slots, axis=1).astype(np.float32)
    return (code_vectors * scales[:, :, None]).reshape(out_count, codeword_count * CODEWORD_DIM)


def decode_e8p_sign_plane_abs_index_rhs_tile(
    packed: E8PSignPlaneAbsIndexRHSTiles,
    *,
    expert: int,
    n_tile: int,
    k_block: int,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one sign-plane RHS tile for parity tests and native oracles."""

    layout = packed.layout
    if not (0 <= expert < layout.experts):
        raise ValueError("expert out of range")
    if not (0 <= n_tile < layout.n_tiles):
        raise ValueError("n_tile out of range")
    if not (0 <= k_block < layout.k_blocks):
        raise ValueError("k_block out of range")
    out_start = n_tile * layout.bn
    out_count = min(layout.bn, layout.output_dims - out_start)
    codeword_count = min(
        layout.codewords_per_bk,
        (layout.input_dims - k_block * layout.bk) // CODEWORD_DIM,
    )
    signs = np.zeros((out_count, codeword_count), dtype=np.uint16)
    masks = packed.sign_bit_planes[expert, n_tile, k_block, :codeword_count, :]
    for word in range(codeword_count):
        for bit in range(8):
            mask = int(masks[word, bit])
            for n in range(out_count):
                signs[n, word] |= np.uint16(((mask >> n) & 1) << bit)
    abs_indices = packed.abs_index_tiles[
        expert, n_tile, k_block, :out_count, :codeword_count
    ].astype(np.uint16)
    codes = signs | (abs_indices << np.uint16(8))
    code_vectors = decode_e8p(codes, codebook)
    scale_slots = packed.codeword_scale_slots[k_block, :codeword_count]
    tile_scales = packed.scale_tiles[expert, n_tile, k_block, :out_count, :]
    scales = np.take(tile_scales, scale_slots, axis=1).astype(np.float32)
    return (code_vectors * scales[:, :, None]).reshape(out_count, codeword_count * CODEWORD_DIM)


__all__ = [
    "E8PComponentStreamContract",
    "E8PComponentStreamSharedDecodeContract",
    "E8PComponentStreamSharedDecodeFeasibility",
    "E8PComponentStreamRHSTiles",
    "E8PComponentStreamWorkgroup",
    "E8PPackedRHSLayout",
    "E8PPackedRHSTiles",
    "E8PExpertKBlockDecodeReuseFeasibility",
    "E8PExpertKBlockDispatchContract",
    "E8PExpertKBlockDispatchWorkgroup",
    "E8PExpertKBlockFactorReuseRHSTiles",
    "E8PExpertKBlockPartialReductionContract",
    "E8PRouteAbsIndexProjectionFeasibility",
    "E8PRouteActiveCodewordProjectionFeasibility",
    "E8PRouteCodewordProjectionFeasibility",
    "E8PSignPlaneAbsIndexRHSTiles",
    "E8PSignNibbleAbsIndexRHSTiles",
    "E8PSignNibbleMicroLUTRHSTiles",
    "E8PSplitByteFactorReuseRHSTiles",
    "E8PSplitByteRHSTiles",
    "build_e8p_component_stream_contract",
    "build_e8p_component_stream_partial_reduction_contract",
    "build_e8p_component_stream_shared_decode_contract",
    "evaluate_e8p_component_stream_shared_decode_feasibility",
    "build_e8p_expert_kblock_dispatch_contract",
    "evaluate_e8p_expert_kblock_decode_reuse_feasibility",
    "evaluate_e8p_route_abs_component_projection_feasibility",
    "evaluate_e8p_route_abs_index_projection_feasibility",
    "evaluate_e8p_route_active_codeword_projection_feasibility",
    "build_e8p_expert_kblock_partial_reduction_contract",
    "evaluate_e8p_route_codeword_projection_feasibility",
    "component_stream_sorted_matmul_oracle",
    "decode_e8p_component_stream_rhs_tile",
    "decode_e8p_expert_kblock_factor_reuse_rhs_tile",
    "decode_e8p_sign_plane_abs_index_rhs_tile",
    "decode_e8p_sign_nibble_abs_index_rhs_tile",
    "decode_e8p_sign_nibble_micro_lut_rhs_tile",
    "decode_e8p_rhs_tile",
    "decode_e8p_split_byte_factor_reuse_rhs_tile",
    "decode_e8p_split_byte_rhs_tile",
    "pack_e8p_component_stream_rhs_tiles",
    "pack_e8p_expert_kblock_factor_reuse_rhs_tiles",
    "pack_e8p_sign_plane_abs_index_rhs_tiles",
    "pack_e8p_sign_nibble_abs_index_rhs_tiles",
    "pack_e8p_sign_nibble_micro_lut_rhs_tiles",
    "pack_e8p_split_byte_factor_reuse_rhs_tiles",
    "pack_e8p_rhs_tiles",
    "pack_e8p_split_byte_rhs_tiles",
]
