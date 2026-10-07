// Wide-M VQ verify kernel: M_ROWS verify rows per expert-grouped route tile.
//
// Lane discipline is identical to gather_vqmm_m1_decoded.metal (ROWS_PER_TG
// output rows per threadgroup, LANES_PER_ROW lanes per output row, strided
// codeword walk, float32 accumulate, shuffle-down reduction). The only
// generalization is that each lane keeps M_ROWS accumulators and reuses the
// code byte, the decoded codebook entry, and the group scale across every
// verify row in the tile. That makes the per-output-element accumulation
// sequence bit-identical to the M=1 decoded kernel, so the wide-M result is
// byte-exact against the M=1 kernel looped over verify rows.
//
// Route tiles come from the same (tile_experts, tile_offsets, tile_counts)
// descriptor triple the simdgroup-MMA block family uses, so "flat" dispatch
// (one route per tile, M_ROWS=1) and "expert-grouped" dispatch (up to M_ROWS
// routes sharing one expert) are the same kernel with different descriptors.
// One codeword's 8 activations. When x is float16 the offset is always a
// multiple of 8, so this is two aligned half4 loads instead of eight scalar
// loads; the values and the subsequent dots are identical either way, so
// byte-exactness against the M=1 decoded kernel is preserved.
#define MLX_VQ_VERIFY_LOAD_X(BASE, DST0, DST1)                                     \
    do {                                                                           \
        uint mlx_vq_xb = (BASE);                                                   \
        if (uint(X_HALF) != 0u) {                                                   \
            const device half4 *mlx_vq_xv = (const device half4 *)(x + mlx_vq_xb); \
            DST0 = mlx_vq_xv[0];                                                   \
            DST1 = mlx_vq_xv[1];                                                   \
        } else {                                                                   \
            DST0 = half4(half(x[mlx_vq_xb]), half(x[mlx_vq_xb + 1u]),              \
                         half(x[mlx_vq_xb + 2u]), half(x[mlx_vq_xb + 3u]));        \
            DST1 = half4(half(x[mlx_vq_xb + 4u]), half(x[mlx_vq_xb + 5u]),         \
                         half(x[mlx_vq_xb + 6u]), half(x[mlx_vq_xb + 7u]));        \
        }                                                                          \
    } while (0)

uint lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint tile_id = threadgroup_position_in_grid.z;

uint routes_in_tile = uint(tile_counts[tile_id]);
if (routes_in_tile == 0u) {
    return;
}
uint expert = uint(tile_experts[tile_id]);
uint route_base = uint(tile_offsets[tile_id]);

uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint codewords_per_group = codewords / groups;
uint input_dim = codewords * 8u;

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

// Inactive verify rows in a partial tile are clamped onto row 0 rather than
// branch-guarded, so the accumulation loop below stays completely branch-free.
// Their results are simply not written out. Guarding inside the loop instead
// costs ~6x on this shape family: the per-row x loads stop being independent.
uint x_row_base[M_ROWS];
#pragma clang loop unroll(full)
for (uint m = 0; m < uint(M_ROWS); ++m) {
    uint slot = m < routes_in_tile ? m : 0u;
    x_row_base[m] = uint(lhs_indices[route_base + slot]) * input_dim;
}

float acc[M_ROWS];
#pragma clang loop unroll(full)
for (uint m = 0; m < uint(M_ROWS); ++m) {
    acc[m] = 0.0f;
}

if (row_slot < uint(ROWS_PER_TG) && row < out_dim) {
    uint code_base = (expert * out_dim + row) * codewords;
    uint scale_base = (expert * out_dim + row) * groups;

    for (uint group = 0; group < groups; ++group) {
        float scale = float(scales[scale_base + group]);
        uint group_start = group * codewords_per_group;
        uint group_end = group_start + codewords_per_group;
        uint lane_stride = uint(LANES_PER_ROW);
        uint block_stride = uint(CW_UNROLL) * lane_stride;
        uint cw = group_start + row_lane;

        // The inner loop is latency-bound, not bandwidth-bound: the codebook
        // lookup depends on the code byte that was just loaded, and one
        // codeword per iteration leaves that ~100-cycle chain exposed.
        // Processing CW_UNROLL codewords per iteration issues their code-byte
        // loads back to back and overlaps the dependent table lookups.
        // Accumulation stays strictly ascending in cw for every acc[m], so this
        // does not reassociate anything and byte-exactness is preserved.
        for (; cw + block_stride <= group_end; cw += block_stride) {
            uint codeword[CW_UNROLL];
            #pragma clang loop unroll(full)
            for (uint u = 0; u < uint(CW_UNROLL); ++u) {
                codeword[u] = uint(codes[code_base + cw + u * lane_stride]);
            }
            half4 w0[CW_UNROLL];
            half4 w1[CW_UNROLL];
            #pragma clang loop unroll(full)
            for (uint u = 0; u < uint(CW_UNROLL); ++u) {
                w0[u] = local_decoded_codebook[codeword[u] * 2u];
                w1[u] = local_decoded_codebook[codeword[u] * 2u + 1u];
            }
            #pragma clang loop unroll(full)
            for (uint u = 0; u < uint(CW_UNROLL); ++u) {
                uint x_offset = (cw + u * lane_stride) * 8u;
                #pragma clang loop unroll(full)
                for (uint m = 0; m < uint(M_ROWS); ++m) {
                    half4 x0;
                    half4 x1;
                    MLX_VQ_VERIFY_LOAD_X(x_row_base[m] + x_offset, x0, x1);
                    acc[m] += float(dot(w0[u], x0)) * scale;
                    acc[m] += float(dot(w1[u], x1)) * scale;
                }
            }
        }
        for (; cw < group_end; cw += lane_stride) {
            uint codeword = uint(codes[code_base + cw]);
            half4 w0 = local_decoded_codebook[codeword * 2u];
            half4 w1 = local_decoded_codebook[codeword * 2u + 1u];
            uint x_offset = cw * 8u;
            #pragma clang loop unroll(full)
            for (uint m = 0; m < uint(M_ROWS); ++m) {
                half4 x0;
                half4 x1;
                MLX_VQ_VERIFY_LOAD_X(x_row_base[m] + x_offset, x0, x1);
                acc[m] += float(dot(w0, x0)) * scale;
                acc[m] += float(dot(w1, x1)) * scale;
            }
        }
    }
}

#pragma clang loop unroll(full)
for (uint m = 0; m < uint(M_ROWS); ++m) {
    float total = acc[m];
    for (uint stride = uint(LANES_PER_ROW) >> 1; stride > 0; stride >>= 1) {
        if (row_lane < stride) {
            total += simd_shuffle_down(total, stride);
        }
    }
    if (row_lane == 0u && row_slot < uint(ROWS_PER_TG) && row < out_dim && m < routes_in_tile) {
        out[((route_base + m) * out_dim) + row] = OUT_T(total);
    }
}

#undef MLX_VQ_VERIFY_LOAD_X
