// Few-token (M=1 decode, M<=32 verify) gather matvec for 16-bit E8P codes on
// GPUs without NAX. One threadgroup column per route: route r multiplies
// activation row r / X_DIV by expert rhs_indices[r].
//
// Each simdgroup owns ROWS_PER_SG output rows. A lane takes CHUNK (4 or 8)
// consecutive codewords: one 8- or 16-byte load per row, so a simdgroup reads
// a contiguous 32 * CHUNK * 2-byte span per row per step, and the lane's
// activations are loaded once and reused for every row.
//
// E8P layout (see mlx_vq_decode_e8p_value in common_e8.metal): codeword bits
// 0-7 are signs, bits 8-15 index the 256-entry abs grid of packed nibbles.
// Dimension d reads nibble {0,4,1,5,2,6,3,7}[d], so nibbles k and k+4 are
// dims 2k and 2k+1 and decode together as one half2. Bit p of the effective
// signs negates nibble p, and every value gets +-0.25 by parity.
//
// Decode is the bottleneck (loads alone run at ~410 GB/s on an M4 Max), so
// each threadgroup first decodes the abs grid into half2 pairs, and a
// codeword then costs one threadgroup read plus one XOR per pair for signs.
//
// X_DIV=top_k: token t's routes share activation row t (gate/up).
// X_DIV=1: every route has its own activation row (down).
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
for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
    acc[r] = 0.0f;
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
    }
}

for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
    uint row = row0 + r;
    float total = simd_sum(acc[r]);
    if (simd_lane == 0u && row < out_dim) {
        out[route * out_dim + row] = OUT_T(total);
    }
}
