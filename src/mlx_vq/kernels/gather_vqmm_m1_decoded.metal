uint lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;

uint expert = uint(rhs_indices[route]);

uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint codewords_per_group = codewords / groups;

uint row_slot = lane / uint(LANES_PER_ROW);
uint row_lane = lane - row_slot * uint(LANES_PER_ROW);
uint row = row_tile * uint(ROWS_PER_TG) + row_slot;

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

float acc = 0.0f;
if (row_slot < uint(ROWS_PER_TG) && row < out_dim) {
    uint code_base = (expert * out_dim + row) * codewords;
    uint scale_base = (expert * out_dim + row) * groups;

    for (uint group = 0; group < groups; ++group) {
        float scale_lane = row_lane == 0u ? float(scales[scale_base + group]) : 0.0f;
        uint rows_per_simd = 32u / uint(LANES_PER_ROW);
        uint scale_source_lane = (row_slot % rows_per_simd) * uint(LANES_PER_ROW);
        float scale = simd_broadcast(scale_lane, scale_source_lane);
        uint group_start = group * codewords_per_group;
        uint group_end = group_start + codewords_per_group;
        for (uint cw = group_start + row_lane; cw < group_end; cw += uint(LANES_PER_ROW)) {
            uint codeword = uint(codes[code_base + cw]);
            uint decoded_base = codeword * 2u;
            uint x_base = cw * 8u;

            half4 w0 = local_decoded_codebook[decoded_base];
            half4 w1 = local_decoded_codebook[decoded_base + 1u];
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
            acc += float(dot(w0, x0)) * scale;
            acc += float(dot(w1, x1)) * scale;
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
