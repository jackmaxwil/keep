uint lane = thread_position_in_threadgroup.x;
uint simdgroup_id = lane >> 5;
uint route_group = simdgroup_id & 7u;
uint row_group = simdgroup_id >> 3;
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
threadgroup half a_tile[512];
threadgroup half b_tile[256];
threadgroup float c_tile[2048];

for (uint idx = lane; idx < 256u; idx += threads_per_threadgroup.x) {
    local_codebook[idx] = codebook[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_float8x8 acc_frag(0.0f);

for (uint k_base = 0u; k_base < input_dim; k_base += 8u) {
    for (uint idx = lane; idx < 512u; idx += threads_per_threadgroup.x) {
        uint route_slot = idx / 8u;
        uint k_local = idx - route_slot * 8u;
        uint route = route_base + route_slot;
        half value = half(0.0h);
        if (route_slot < routes_in_tile && (k_base + k_local) < input_dim) {
            uint token = uint(lhs_indices[route]);
            value = half(float(x[token * input_dim + k_base + k_local]));
        }
        a_tile[idx] = value;
    }

    for (uint idx = lane; idx < 256u; idx += threads_per_threadgroup.x) {
        uint row_block = idx / 64u;
        uint local = idx - row_block * 64u;
        uint k_local = local / 8u;
        uint row_slot_in_block = local - k_local * 8u;
        uint out_row = row_base + row_block * 8u + row_slot_in_block;
        uint k_global = k_base + k_local;
        half value = half(0.0h);
        if (out_row < out_dim && k_global < input_dim) {
            uint cw = k_global >> 3;
            uint dim = k_global & 7u;
            uint code_base = (expert * out_dim + out_row) * codewords;
            uint scale_base = (expert * out_dim + out_row) * groups;
            uint codeword = uint(codes[code_base + cw]);
            float scale = float(scales[scale_base + (k_global / group_size)]);
            float decoded;
            if (CODE_BITS == 8) {
                decoded = mlx_vq_decode_packed_int2_value(local_codebook[codeword], dim);
            } else {
                decoded = mlx_vq_decode_e8p_value_tg(codeword, local_codebook, 0u, dim);
            }
            value = half(decoded * scale);
        }
        b_tile[idx] = value;
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

for (uint idx = lane; idx < 2048u; idx += threads_per_threadgroup.x) {
    uint sg = idx / 64u;
    uint local = idx - sg * 64u;
    uint rg = sg & 7u;
    uint cg = sg >> 3;
    uint route_slot = rg * 8u + local / 8u;
    uint row_slot = cg * 8u + (local & 7u);
    uint route = route_base + route_slot;
    uint row = row_base + row_slot;
    if (route_slot < routes_in_tile && row < out_dim) {
        out[route * out_dim + row] = OUT_T(c_tile[idx]);
    }
}
