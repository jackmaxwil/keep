uint lane = thread_position_in_threadgroup.x;
uint simdgroup_id = lane >> 5;
uint route_group = simdgroup_id % uint(ROUTE_GROUPS);
uint row_group = simdgroup_id / uint(ROUTE_GROUPS);
uint tile_id = threadgroup_position_in_grid.z;
uint row_tile = threadgroup_position_in_grid.y;

uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint input_dim = codewords * 8u;
uint group_size = input_dim / groups;

uint expert = uint(tile_experts[tile_id]);
uint route_base = uint(tile_offsets[tile_id]);
uint routes_in_tile = uint(tile_counts[tile_id]);
uint row_base = row_tile * 32u;

if (routes_in_tile == 0u) {
    return;
}

threadgroup uint local_codebook[256];
threadgroup half a_tile[A_TILE_SIZE];
threadgroup half b_tile[256];
threadgroup float c_tile[C_TILE_SIZE];

for (uint idx = lane; idx < 256u; idx += threads_per_threadgroup.x) {
    local_codebook[idx] = codebook[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_float8x8 acc_frag(0.0f);

for (uint k_base = 0u; k_base < input_dim; k_base += 8u) {
    for (uint idx = lane; idx < uint(A_TILE_SIZE); idx += threads_per_threadgroup.x) {
        uint route_slot = idx / 8u;
        uint k_local = idx - route_slot * 8u;
        uint route = route_base + route_slot;
        half value = half(0.0h);
        if (route_slot < routes_in_tile) {
            uint token = uint(lhs_indices[route]);
            value = half(float(x[token * input_dim + k_base + k_local]));
        }
        a_tile[idx] = value;
    }

    for (uint idx = lane; idx < 32u; idx += threads_per_threadgroup.x) {
        uint row_block = idx >> 3;
        uint row_slot_in_block = idx & 7u;
        uint out_row = row_base + row_block * 8u + row_slot_in_block;
        uint packed = 0u;
        float scale = 0.0f;
        if (out_row < out_dim) {
            uint cw = k_base >> 3;
            uint code_base = (expert * out_dim + out_row) * codewords;
            uint scale_base = (expert * out_dim + out_row) * groups;
            packed = local_codebook[uint(codes[code_base + cw])];
            scale = float(scales[scale_base + (k_base / group_size)]);
        }

        uint base = row_block * 64u + row_slot_in_block;
        b_tile[base] = half(mlx_vq_decode_packed_int2_value(packed, 0u) * scale);
        b_tile[base + 8u] = half(mlx_vq_decode_packed_int2_value(packed, 1u) * scale);
        b_tile[base + 16u] = half(mlx_vq_decode_packed_int2_value(packed, 2u) * scale);
        b_tile[base + 24u] = half(mlx_vq_decode_packed_int2_value(packed, 3u) * scale);
        b_tile[base + 32u] = half(mlx_vq_decode_packed_int2_value(packed, 4u) * scale);
        b_tile[base + 40u] = half(mlx_vq_decode_packed_int2_value(packed, 5u) * scale);
        b_tile[base + 48u] = half(mlx_vq_decode_packed_int2_value(packed, 6u) * scale);
        b_tile[base + 56u] = half(mlx_vq_decode_packed_int2_value(packed, 7u) * scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 a_frag;
    simdgroup_half8x8 b_frag;
    simdgroup_load(a_frag, a_tile + route_group * 64u, 8);
    simdgroup_load(b_frag, b_tile + row_group * 64u, 8);
    simdgroup_multiply_accumulate(acc_frag, a_frag, b_frag, acc_frag);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

simdgroup_store(acc_frag, c_tile + simdgroup_id * 64u, 8);
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint idx = lane; idx < uint(C_TILE_SIZE); idx += threads_per_threadgroup.x) {
    uint sg = idx / 64u;
    uint local = idx - sg * 64u;
    uint rg = sg % uint(ROUTE_GROUPS);
    uint cg = sg / uint(ROUTE_GROUPS);
    uint route_slot = rg * 8u + local / 8u;
    uint row_slot = cg * 8u + (local & 7u);
    uint route = route_base + route_slot;
    uint row = row_base + row_slot;
    if (route_slot < routes_in_tile && row < out_dim) {
        out[route * out_dim + row] = OUT_T(c_tile[idx]);
    }
}
