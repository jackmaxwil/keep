// Few-token gather matvec for 8-bit E8 codes (the 1-bit residual of a nested
// tier), the companion of e8p_m1.metal: one threadgroup column per route,
// route r multiplies activation row r / X_DIV by expert rhs_indices[r]. A
// route whose index is 0xFFFFFFFF has no residual and writes zeros.
//
// A codeword is one byte that indexes a 256-entry table of 8 values. The
// table arrives with nibbles k and k+4 holding dims 2k and 2k+1 (see
// pair_packed in e8_1bit_m1.py), so each threadgroup decodes it into half2
// pairs once and a codeword then costs one threadgroup read. A lane takes
// CHUNK (4 or 8) consecutive codewords with one 4- or 8-byte load per row.
uint simd_lane = thread_index_in_simdgroup;
uint simd_id = simdgroup_index_in_threadgroup;
uint tg_lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;

uint expert = uint(rhs_indices[route]);
uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint codewords_per_group = codewords / groups;
uint x_row = (route / uint(X_DIV)) * codewords * 8u;
uint row0 = (row_tile * uint(SIMDGROUPS) + simd_id) * uint(ROWS_PER_SG);

if (expert == 0xFFFFFFFFu) {
    for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
        if (simd_lane == 0u && row0 + r < out_dim) {
            out[route * out_dim + row0 + r] = OUT_T(0.0f);
        }
    }
    return;
}

// (1024 + n) * 0.5 - 516 = (n - 8) / 2, exact in half.
threadgroup uint4 decoded[256];
for (uint idx = tg_lane; idx < 256u; idx += threads_per_threadgroup.x) {
    uint packed = codebook[idx];
    uint4 entry;
    for (uint k = 0; k < 4u; ++k) {
        half2 h = as_type<half2>(((packed >> (4u * k)) & 0x000F000Fu) | 0x64006400u);
        entry[k] = as_type<uint>(fma(h, half2(0.5h), half2(-516.0h)));
    }
    decoded[idx] = entry;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float acc[ROWS_PER_SG];
for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
    acc[r] = 0.0f;
}

for (uint cw0 = simd_lane * uint(CHUNK); cw0 < codewords; cw0 += 32u * uint(CHUNK)) {
    float2 xv[CHUNK * 4];
    for (uint i = 0; i < uint(CHUNK); ++i) {
        for (uint k = 0; k < 4u; ++k) {
            uint xi = x_row + (cw0 + i) * 8u + 2u * k;
            xv[i * 4u + k] = float2(float(x[xi]), float(x[xi + 1u]));
        }
    }
    uint group = cw0 / codewords_per_group;
    for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
        uint row = row0 + r;
        if (row >= out_dim) {
            break;
        }
        uint base = expert * out_dim + row;
        uint2 words = uint2(0u);
        if (uint(CHUNK) == 8u) {
            words = ((const device uint2*)(codes + base * codewords))[cw0 / 8u];
        } else {
            words.x = ((const device uint*)(codes + base * codewords))[cw0 / 4u];
        }
        float part = 0.0f;
        for (uint i = 0; i < uint(CHUNK); ++i) {
            uint4 entry = decoded[(words[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu];
            for (uint k = 0; k < 4u; ++k) {
                part += dot(float2(as_type<half2>(entry[k])), xv[i * 4u + k]);
            }
        }
        acc[r] += part * float(scales[base * groups + group]);
    }
}

for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
    uint row = row0 + r;
    float total = simd_sum(acc[r]);
    if (simd_lane == 0u && row < out_dim) {
        out[route * out_dim + row] = OUT_T(total);
    }
}
