uint lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint route_tile = threadgroup_position_in_grid.z;

uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint input_dim = codewords * 8u;
uint route_count = uint(rhs_indices_shape[0]);
uint group_size = input_dim / groups;
uint duplicate_base = (lane % uint(CODEBOOK_DUP)) * 256u;

uint route_base = route_tile * uint(M_TILE);
uint row_base = row_tile * uint(N_TILE);
uint lane_route = lane / uint(N_TILE);
uint lane_row = lane - lane_route * uint(N_TILE);
uint route = route_base + lane_route;
uint row = row_base + lane_row;
uint elements_per_tile = uint(M_TILE) * uint(N_TILE);

threadgroup uint local_codebook[2048];
threadgroup float weight_tile[N_TILE * K_TILE_DIMS];

for (uint idx = lane; idx < uint(CODEBOOK_DUP) * 256u; idx += threads_per_threadgroup.x) {
    local_codebook[idx] = codebook[idx & 255u];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

bool valid_element = lane < elements_per_tile && route < route_count && row < out_dim;
uint routes_in_tile = route_base < route_count ? min(uint(M_TILE), route_count - route_base) : 0u;

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

float acc = 0.0f;
if (single_expert_tile) {
    for (uint k_base = 0u; k_base < input_dim; k_base += uint(K_TILE_DIMS)) {
        for (uint idx = lane; idx < uint(N_TILE) * uint(K_TILE_DIMS); idx += threads_per_threadgroup.x) {
            uint tile_row = idx / uint(K_TILE_DIMS);
            uint k_local = idx - tile_row * uint(K_TILE_DIMS);
            uint out_row = row_base + tile_row;
            uint k_global = k_base + k_local;
            float decoded = 0.0f;

            if (out_row < out_dim && k_global < input_dim) {
                uint cw = k_global >> 3;
                uint dim = k_global & 7u;
                uint code_base = (first_expert * out_dim + out_row) * codewords;
                uint scale_base = (first_expert * out_dim + out_row) * groups;
                uint codeword = uint(codes[code_base + cw]);
                float scale = float(scales[scale_base + (k_global / group_size)]);

                if (CODE_BITS == 8) {
                    uint packed = local_codebook[duplicate_base + codeword];
                    decoded = mlx_vq_decode_packed_int2_value(packed, dim) * scale;
                } else {
                    decoded = mlx_vq_decode_e8p_value_tg(codeword, local_codebook, duplicate_base, dim) * scale;
                }
            }
            weight_tile[idx] = decoded;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_element) {
            uint token = uint(lhs_indices[route]);
            uint x_base = token * input_dim;
            for (uint k_local = 0u; k_local < uint(K_TILE_DIMS) && (k_base + k_local) < input_dim; ++k_local) {
                acc += weight_tile[lane_row * uint(K_TILE_DIMS) + k_local] * float(x[x_base + k_base + k_local]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
} else if (valid_element) {
    uint token = uint(lhs_indices[route]);
    uint expert = uint(rhs_indices[route]);
    uint code_base = (expert * out_dim + row) * codewords;
    uint scale_base = (expert * out_dim + row) * groups;
    uint x_base = token * input_dim;

    for (uint cw = 0u; cw < codewords; ++cw) {
        uint codeword = uint(codes[code_base + cw]);
        float scale = float(scales[scale_base + ((cw * 8u) / group_size)]);

        for (uint dim = 0u; dim < 8u; ++dim) {
            float weight;
            if (CODE_BITS == 8) {
                uint packed = local_codebook[duplicate_base + codeword];
                weight = mlx_vq_decode_packed_int2_value(packed, dim);
            } else {
                weight = mlx_vq_decode_e8p_value_tg(codeword, local_codebook, duplicate_base, dim);
            }
            acc += weight * scale * float(x[x_base + cw * 8u + dim]);
        }
    }
}

if (valid_element) {
    out[route * out_dim + row] = OUT_T(acc);
}
