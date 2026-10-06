uint lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;

uint top_k = uint(rhs_indices_shape[1]);
uint expert = uint(rhs_indices[route % top_k]);

uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint codewords_per_group = codewords / groups;

uint row_slot = lane / uint(LANES_PER_ROW);
uint row_lane = lane - row_slot * uint(LANES_PER_ROW);
uint row = row_tile * uint(ROWS_PER_TG) + row_slot;

threadgroup uint local_codebook[256];
threadgroup half4 local_decoded_codebook[512];
if (CODE_BITS == 8 && uint(USE_DECODED_CODEBOOK) != 0u) {
    for (uint idx = lane; idx < 512u; idx += threads_per_threadgroup.x) {
        uint packed = codebook[idx >> 1u];
        uint dim_base = (idx & 1u) * 4u;
        uint nibble0 = (packed >> (4u * dim_base)) & 0xFu;
        uint nibble1 = (packed >> (4u * (dim_base + 1u))) & 0xFu;
        uint nibble2 = (packed >> (4u * (dim_base + 2u))) & 0xFu;
        uint nibble3 = (packed >> (4u * (dim_base + 3u))) & 0xFu;
        local_decoded_codebook[idx] = half4(
            half((float(nibble0) - 8.0f) * 0.5f),
            half((float(nibble1) - 8.0f) * 0.5f),
            half((float(nibble2) - 8.0f) * 0.5f),
            half((float(nibble3) - 8.0f) * 0.5f)
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
} else if (uint(USE_THREADGROUP_CODEBOOK) != 0u) {
    for (uint idx = lane; idx < 256u; idx += threads_per_threadgroup.x) {
        local_codebook[idx] = codebook[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float acc = 0.0f;
if (row_slot < uint(ROWS_PER_TG) && row < out_dim) {
    uint code_base = (expert * out_dim + row) * codewords;
    uint scale_base = (expert * out_dim + row) * groups;

    for (uint group = 0; group < groups; ++group) {
        float scale = float(scales[scale_base + group]);
        uint group_start = group * codewords_per_group;
        uint group_end = group_start + codewords_per_group;
        for (uint cw = group_start + row_lane; cw < group_end; cw += uint(LANES_PER_ROW)) {
            uint codeword = uint(codes[code_base + cw]);
            if (CODE_BITS == 8) {
                uint x_base = cw * 8u;
                float x0 = float(x[x_base]);
                float x1 = float(x[x_base + 1u]);
                float x2 = float(x[x_base + 2u]);
                float x3 = float(x[x_base + 3u]);
                float x4 = float(x[x_base + 4u]);
                float x5 = float(x[x_base + 5u]);
                float x6 = float(x[x_base + 6u]);
                float x7 = float(x[x_base + 7u]);
                if (uint(USE_DECODED_CODEBOOK) != 0u) {
                    uint decoded_base = codeword * 2u;
                    float4 w0 = float4(local_decoded_codebook[decoded_base]);
                    float4 w1 = float4(local_decoded_codebook[decoded_base + 1u]);
                    acc += dot(w0, float4(x0, x1, x2, x3)) * scale;
                    acc += dot(w1, float4(x4, x5, x6, x7)) * scale;
                } else {
                    uint packed = uint(USE_THREADGROUP_CODEBOOK) != 0u ? local_codebook[codeword] : codebook[codeword];
                    float scaled = scale * 0.5f;
                    acc += (float(packed & 0xFu) - 8.0f) * scaled * x0;
                    acc += (float((packed >> 4u) & 0xFu) - 8.0f) * scaled * x1;
                    acc += (float((packed >> 8u) & 0xFu) - 8.0f) * scaled * x2;
                    acc += (float((packed >> 12u) & 0xFu) - 8.0f) * scaled * x3;
                    acc += (float((packed >> 16u) & 0xFu) - 8.0f) * scaled * x4;
                    acc += (float((packed >> 20u) & 0xFu) - 8.0f) * scaled * x5;
                    acc += (float((packed >> 24u) & 0xFu) - 8.0f) * scaled * x6;
                    acc += (float((packed >> 28u) & 0xFu) - 8.0f) * scaled * x7;
                }
            } else {
                for (uint dim = 0; dim < 8; ++dim) {
                    float weight;
                    if (uint(USE_THREADGROUP_CODEBOOK) != 0u) {
                        weight = mlx_vq_decode_e8p_value_tg(codeword, local_codebook, 0u, dim);
                    } else {
                        weight = mlx_vq_decode_e8p_value(codeword, codebook, dim);
                    }
                    uint x_idx = cw * 8u + dim;
                    acc += weight * scale * float(x[x_idx]);
                }
            }
        }
    }
}

if (uint(LANES_PER_ROW) == 32u) {
    float total = simd_sum(acc);
    if (row_lane == 0 && row_slot < uint(ROWS_PER_TG) && row < out_dim) {
        out[(route * out_dim) + row] = OUT_T(total);
    }
} else if (uint(LANES_PER_ROW) <= 16u) {
    float total = acc;
    for (uint stride = uint(LANES_PER_ROW) >> 1; stride > 0; stride >>= 1) {
        if (row_lane < stride) {
            total += simd_shuffle_down(total, stride);
        }
    }
    if (row_lane == 0 && row_slot < uint(ROWS_PER_TG) && row < out_dim) {
        out[(route * out_dim) + row] = OUT_T(total);
    }
} else {
    threadgroup float partials[256];
    partials[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = uint(LANES_PER_ROW) >> 1; stride > 0; stride >>= 1) {
        if (row_lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_lane == 0 && row_slot < uint(ROWS_PER_TG) && row < out_dim) {
        out[(route * out_dim) + row] = OUT_T(partials[lane]);
    }
}
