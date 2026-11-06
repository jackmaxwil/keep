// Single-token (M=1) gather matvec for 16-bit E8P codes.
//
// Each simdgroup owns ROWS_PER_SG output rows. A lane takes CHUNK consecutive
// codewords (one 8- or 16-byte load per row), so a simdgroup reads one
// contiguous 32 * CHUNK * 2-byte span per row per step, and the lane's
// CHUNK * 8 activations are loaded once and reused for every row.
//
// E8P layout (see mlx_vq_decode_e8p_value in common_e8.metal): codeword bits
// 0-7 are signs, bits 8-15 index the 256-entry abs grid of packed nibbles.
// Dimension d reads nibble {0,4,1,5,2,6,3,7}[d]; bit p of the effective signs
// negates nibble p, and every value gets +-0.25 by parity.
//
// PER_ROUTE=0: one activation row for every route (gate/up).
// PER_ROUTE=1: route r reads activation row r (down).
uint simd_lane = thread_index_in_simdgroup;
uint simd_id = simdgroup_index_in_threadgroup;
uint tg_lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;

uint top_k = uint(rhs_indices_shape[1]);
uint expert = uint(rhs_indices[route % top_k]);
uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint codewords_per_group = codewords / groups;
uint x_row = uint(PER_ROUTE) != 0u ? route * codewords * 8u : 0u;

threadgroup uint local_codebook[256];
if (uint(TG_CODEBOOK) != 0u) {
    for (uint idx = tg_lane; idx < 256u; idx += threads_per_threadgroup.x) {
        local_codebook[idx] = codebook[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

uint row0 = (row_tile * uint(SIMDGROUPS) + simd_id) * uint(ROWS_PER_SG);
float acc[ROWS_PER_SG];
for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
    acc[r] = 0.0f;
}

for (uint cw0 = simd_lane * uint(CHUNK); cw0 < codewords; cw0 += 32u * uint(CHUNK)) {
    float2 xv[CHUNK * 4];
    half2 xh[CHUNK * 4];
    float xs[CHUNK];
    for (uint i = 0; i < uint(CHUNK); ++i) {
        float sum = 0.0f;
        for (uint k = 0; k < 4u; ++k) {
            uint xi = x_row + (cw0 + i) * 8u + 2u * k;
            xv[i * 4u + k] = float2(float(x[xi]), float(x[xi + 1u]));
            xh[i * 4u + k] = half2(xv[i * 4u + k]);
            sum += xv[i * 4u + k].x + xv[i * 4u + k].y;
        }
        xs[i] = sum;
    }
    uint group = cw0 / codewords_per_group;
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
            uint abs_code = uint(TG_CODEBOOK) != 0u ? local_codebook[codeword >> 8u] : codebook[codeword >> 8u];
            // Nibbles k and k+4 are dims 2k and 2k+1: one half2 per pair.
            // (1024 + n) * 0.5 - 516 = (n - 8) / 2, exact in half.
            uint spread = (effective & 0x0Fu) | ((effective & 0xF0u) << 12u);
            half2 hacc = half2(0.0h);
            for (uint k = 0; k < 4u; ++k) {
                half2 h = as_type<half2>(((abs_code >> (4u * k)) & 0x000F000Fu) | 0x64006400u);
                half2 v = fma(h, half2(0.5h), half2(-516.0h));
                v = as_type<half2>(as_type<uint>(v) ^ ((spread << (15u - k)) & 0x80008000u));
                hacc = fma(v, xh[i * 4u + k], hacc);
            }
            part += float(hacc.x) + float(hacc.y);
            part += (parity != 0u ? -0.25f : 0.25f) * xs[i];
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
