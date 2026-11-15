// Few-token gather matvec for a nested tier: 16-bit E8P base codes plus, for
// the slots that hold a hot expert, an 8-bit E8 residual. One pass does what
// e8p_m1.metal followed by e8_1bit_m1.metal do: the same threadgroup layout
// (one column per route, route r multiplies activation row r / X_DIV by slot
// rhs_indices[r]), the same lane chunks, one load of the activations.
//
// hot_of_slot[slot] is the slot's row in the residual arrays, or 0xFFFFFFFF
// for a slot at the base tier. The test is uniform over a threadgroup, so a
// base-tier route runs the e8p_m1 loop and nothing else.
//
// The base's abs grid is decoded into threadgroup memory as in e8p_m1.metal.
// The residual's table arrives decoded (res_table: 256 entries of four half2
// pairs, see half2_pairs in e8p_nested_m1.py) and is read from device memory:
// a second 4 KB threadgroup array made every route 1.5x slower on an M4 Max,
// the base-tier routes included.
//
// Base and residual keep separate accumulators, so a float32 output equals
// the float32 sum of the two separate kernels' outputs bit for bit.
uint simd_lane = thread_index_in_simdgroup;
uint simd_id = simdgroup_index_in_threadgroup;
uint tg_lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;

uint expert = uint(rhs_indices[route]);
uint hot = uint(hot_of_slot[expert]);
bool nested = hot != 0xFFFFFFFFu;
uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint res_groups = uint(res_scales_shape[2]);
uint codewords_per_group = codewords / groups;
uint codewords_per_res_group = codewords / res_groups;
uint x_row = (route / uint(X_DIV)) * codewords * 8u;

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

uint row0 = (row_tile * uint(SIMDGROUPS) + simd_id) * uint(ROWS_PER_SG);
float acc[ROWS_PER_SG];
float res_acc[ROWS_PER_SG];
for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
    acc[r] = 0.0f;
    res_acc[r] = 0.0f;
}

for (uint cw0 = simd_lane * uint(CHUNK); cw0 < codewords; cw0 += 32u * uint(CHUNK)) {
    float2 xv[CHUNK * 4];
    float xs[CHUNK];
    for (uint i = 0; i < uint(CHUNK); ++i) {
        float sum = 0.0f;
        for (uint k = 0; k < 4u; ++k) {
            uint xi = x_row + (cw0 + i) * 8u + 2u * k;
            xv[i * 4u + k] = float2(float(x[xi]), float(x[xi + 1u]));
            sum += xv[i * 4u + k].x + xv[i * 4u + k].y;
        }
        xs[i] = sum;
    }
    uint group = cw0 / codewords_per_group;
    uint res_group = cw0 / codewords_per_res_group;
    for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
        uint row = row0 + r;
        if (row >= out_dim) {
            break;
        }
        uint base = expert * out_dim + row;
        uint4 words;
        if (uint(CHUNK) == 8u) {
            words = ((const device uint4*)(codes + base * codewords))[cw0 / 8u];
        } else {
            words.xy = ((const device uint2*)(codes + base * codewords))[cw0 / 4u];
        }
        float part = 0.0f;
        for (uint i = 0; i < uint(CHUNK); ++i) {
            uint codeword = (words[i >> 1u] >> ((i & 1u) * 16u)) & 0xFFFFu;
            uint signs = codeword & 0xFFu;
            uint parity = popcount(signs) & 1u;
            uint effective = signs ^ parity;
            uint4 entry = decoded[codeword >> 8u];
            // Sign bits k and k+4 move to bits 15 and 31 of pair k.
            uint spread = (effective & 0x0Fu) | ((effective & 0xF0u) << 12u);
            for (uint k = 0; k < 4u; ++k) {
                half2 v = as_type<half2>(entry[k] ^ ((spread << (15u - k)) & 0x80008000u));
                part += dot(float2(v), xv[i * 4u + k]);
            }
            part += (parity != 0u ? -0.25f : 0.25f) * xs[i];
        }
        acc[r] += part * float(scales[base * groups + group]);
        if (nested) {
            uint res_base = hot * out_dim + row;
            uint2 res_words = uint2(0u);
            if (uint(CHUNK) == 8u) {
                res_words = ((const device uint2*)(res_codes + res_base * codewords))[cw0 / 8u];
            } else {
                res_words.x = ((const device uint*)(res_codes + res_base * codewords))[cw0 / 4u];
            }
            float res_part = 0.0f;
            for (uint i = 0; i < uint(CHUNK); ++i) {
                uint4 entry = ((const device uint4*)res_table)[(res_words[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu];
                for (uint k = 0; k < 4u; ++k) {
                    res_part += dot(float2(as_type<half2>(entry[k])), xv[i * 4u + k]);
                }
            }
            res_acc[r] += res_part * float(res_scales[res_base * res_groups + res_group]);
        }
    }
}

for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
    uint row = row0 + r;
    float total = simd_sum(acc[r]);
    if (nested) {
        total += simd_sum(res_acc[r]);
    }
    if (simd_lane == 0u && row < out_dim) {
        out[route * out_dim + row] = OUT_T(total);
    }
}
