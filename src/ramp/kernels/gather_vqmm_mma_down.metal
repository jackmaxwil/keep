uint lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint route_tile = threadgroup_position_in_grid.z;

uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint input_dim = codewords * 8u;
uint route_count = uint(rhs_indices_shape[0]);
uint group_size = input_dim / groups;

uint route_base = route_tile * 8u;
uint row_base = row_tile * 8u;
uint routes_in_tile = route_base < route_count ? min(8u, route_count - route_base) : 0u;

threadgroup uint local_codebook[256];
threadgroup half a_tile[64];
threadgroup half b_tile[64];
threadgroup float c_tile[64];

for (uint idx = lane; idx < 256u; idx += threads_per_threadgroup.x) {
    local_codebook[idx] = codebook[idx];
}
for (uint idx = lane; idx < 64u; idx += threads_per_threadgroup.x) {
    a_tile[idx] = half(0.0h);
    b_tile[idx] = half(0.0h);
    c_tile[idx] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint first_expert = 0u;
bool single_expert_tile = route_base < route_count;
if (single_expert_tile) {
    first_expert = uint(rhs_indices[route_base]);
    for (uint r = 1u; r < routes_in_tile; ++r) {
        if (uint(rhs_indices[route_base + r]) != first_expert) {
            single_expert_tile = false;
        }
    }
}

if (single_expert_tile) {
    simdgroup_float8x8 acc_frag(0.0f);

    for (uint k_base = 0u; k_base < input_dim; k_base += 8u) {
        for (uint idx = lane; idx < 64u; idx += threads_per_threadgroup.x) {
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

        for (uint idx = lane; idx < 64u; idx += threads_per_threadgroup.x) {
            uint k_local = idx / 8u;
            uint row_slot = idx - k_local * 8u;
            uint out_row = row_base + row_slot;
            uint k_global = k_base + k_local;
            half value = half(0.0h);
            if (out_row < out_dim && k_global < input_dim) {
                uint cw = k_global >> 3;
                uint dim = k_global & 7u;
                uint code_base = (first_expert * out_dim + out_row) * codewords;
                uint scale_base = (first_expert * out_dim + out_row) * groups;
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
        simdgroup_load(a_frag, a_tile, 8);
        simdgroup_load(b_frag, b_tile, 8);
        simdgroup_multiply_accumulate(acc_frag, a_frag, b_frag, acc_frag);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc_frag, c_tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = lane; idx < 64u; idx += threads_per_threadgroup.x) {
        uint route_slot = idx / 8u;
        uint row_slot = idx - route_slot * 8u;
        uint route = route_base + route_slot;
        uint row = row_base + row_slot;
        if (route_slot < routes_in_tile && row < out_dim) {
            out[route * out_dim + row] = OUT_T(c_tile[idx]);
        }
    }
} else {
    for (uint idx = lane; idx < 64u; idx += threads_per_threadgroup.x) {
        uint route_slot = idx / 8u;
        uint row_slot = idx - route_slot * 8u;
        uint route = route_base + route_slot;
        uint row = row_base + row_slot;
        if (route_slot < routes_in_tile && row < out_dim) {
            uint token = uint(lhs_indices[route]);
            uint expert = uint(rhs_indices[route]);
            uint code_base = (expert * out_dim + row) * codewords;
            uint scale_base = (expert * out_dim + row) * groups;
            uint x_base = token * input_dim;
            float acc = 0.0f;

            for (uint cw = 0u; cw < codewords; ++cw) {
                uint codeword = uint(codes[code_base + cw]);
                float scale = float(scales[scale_base + ((cw * 8u) / group_size)]);

                for (uint dim = 0u; dim < 8u; ++dim) {
                    float weight;
                    if (CODE_BITS == 8) {
                        weight = mlx_vq_decode_packed_int2_value(local_codebook[codeword], dim);
                    } else {
                        weight = mlx_vq_decode_e8p_value_tg(codeword, local_codebook, 0u, dim);
                    }
                    acc += weight * scale * float(x[x_base + cw * 8u + dim]);
                }
            }
            out[route * out_dim + row] = OUT_T(acc);
        }
    }
}
