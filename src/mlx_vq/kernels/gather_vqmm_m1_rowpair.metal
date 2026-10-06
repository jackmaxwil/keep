uint lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;

uint expert = uint(rhs_indices[route]);

uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint codewords_per_group = codewords / groups;

uint row_pair_slot = lane / 16u;
uint row_lane = lane & 15u;
uint row0 = row_tile * 32u + row_pair_slot * 2u;
uint row1 = row0 + 1u;

threadgroup half4 local_decoded_codebook[512];
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

float acc0 = 0.0f;
float acc1 = 0.0f;
if (row0 < out_dim) {
    uint code_base0 = (expert * out_dim + row0) * codewords;
    uint scale_base0 = (expert * out_dim + row0) * groups;
    uint code_base1 = (expert * out_dim + row1) * codewords;
    uint scale_base1 = (expert * out_dim + row1) * groups;

    for (uint group = 0; group < groups; ++group) {
        float scale0_lane = row_lane == 0u ? float(scales[scale_base0 + group]) : 0.0f;
        float scale1_lane = (row_lane == 0u && row1 < out_dim) ? float(scales[scale_base1 + group]) : 0.0f;
        uint scale_source_lane = (row_pair_slot & 1u) * 16u;
        float scale0 = simd_broadcast(scale0_lane, scale_source_lane);
        float scale1 = simd_broadcast(scale1_lane, scale_source_lane);
        uint group_start = group * codewords_per_group;
        uint group_end = group_start + codewords_per_group;
        for (uint cw = group_start + row_lane; cw < group_end; cw += 16u) {
            uint x_base = cw * 8u;
            half4 x0 = half4(
                half(x[x_base]),
                half(x[x_base + 1u]),
                half(x[x_base + 2u]),
                half(x[x_base + 3u])
            );
            half4 x1 = half4(
                half(x[x_base + 4u]),
                half(x[x_base + 5u]),
                half(x[x_base + 6u]),
                half(x[x_base + 7u])
            );

            uint codeword0 = uint(codes[code_base0 + cw]);
            uint decoded_base0 = codeword0 * 2u;
            half4 w00 = local_decoded_codebook[decoded_base0];
            half4 w01 = local_decoded_codebook[decoded_base0 + 1u];
            acc0 += float(dot(w00, x0)) * scale0;
            acc0 += float(dot(w01, x1)) * scale0;

            if (row1 < out_dim) {
                uint codeword1 = uint(codes[code_base1 + cw]);
                uint decoded_base1 = codeword1 * 2u;
                half4 w10 = local_decoded_codebook[decoded_base1];
                half4 w11 = local_decoded_codebook[decoded_base1 + 1u];
                acc1 += float(dot(w10, x0)) * scale1;
                acc1 += float(dot(w11, x1)) * scale1;
            }
        }
    }
}

float total0 = acc0;
float total1 = acc1;
for (uint stride = 8u; stride > 0u; stride >>= 1u) {
    if (row_lane < stride) {
        total0 += simd_shuffle_down(total0, stride);
        total1 += simd_shuffle_down(total1, stride);
    }
}

if (row_lane == 0u && row0 < out_dim) {
    out[(route * out_dim) + row0] = OUT_T(total0);
    if (row1 < out_dim) {
        out[(route * out_dim) + row1] = OUT_T(total1);
    }
}
