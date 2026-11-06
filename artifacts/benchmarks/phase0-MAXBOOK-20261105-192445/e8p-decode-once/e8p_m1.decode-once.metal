// Single-token (M=1) gather matvec for 16-bit E8P codes, decoding each
// codeword once into eight values (the generic M=1 kernel decodes it once per
// dimension, with an 8-step parity loop each time).
//
// E8P layout (see mlx_vq_decode_e8p_value in common_e8.metal): codeword bits
// 0-7 are signs, bits 8-15 index the 256-entry abs grid of packed nibbles.
// Dimension d reads nibble shuffle(d) = {0,4,1,5,2,6,3,7}[d], so nibbles 0-3
// pair with the even activations and nibbles 4-7 with the odd ones. Bit p of
// the effective signs negates nibble p, and every value gets +-0.25 by parity.
//
// Each lane loads CHUNK (4 or 8) codewords at once; 2-byte loads leave the
// matvec latency-bound.
//
// PER_ROUTE=0: one activation row for every route (gate/up).
// PER_ROUTE=1: route r reads activation row r (down).
uint lane = thread_position_in_threadgroup.x;
uint row_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;

uint top_k = uint(rhs_indices_shape[1]);
uint expert = uint(rhs_indices[route % top_k]);

uint out_dim = uint(codes_shape[1]);
uint codewords = uint(codes_shape[2]);
uint groups = uint(scales_shape[2]);
uint codewords_per_group = codewords / groups;
uint x_row = uint(PER_ROUTE) != 0u ? route * codewords * 8u : 0u;

uint row_slot = lane / uint(LANES_PER_ROW);
uint row_lane = lane - row_slot * uint(LANES_PER_ROW);
uint row = row_tile * uint(ROWS_PER_TG) + row_slot;

threadgroup uint local_codebook[256];
if (uint(USE_THREADGROUP_CODEBOOK) != 0u) {
    for (uint idx = lane; idx < 256u; idx += threads_per_threadgroup.x) {
        local_codebook[idx] = codebook[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float acc = 0.0f;
if (row_slot < uint(ROWS_PER_TG) && row < out_dim) {
    uint code_base = (expert * out_dim + row) * codewords;
    uint scale_base = (expert * out_dim + row) * groups;
    // One wide load of CHUNK codewords per lane per step; the host checks
    // that codewords per group is a multiple of CHUNK.
    uint chunks = codewords / uint(CHUNK);
    for (uint chunk = row_lane; chunk < chunks; chunk += uint(LANES_PER_ROW)) {
        uint cw0 = chunk * uint(CHUNK);
        uint4 words;
        if (uint(CHUNK) == 8u) {
            words = ((const device uint4*)(codes + code_base))[chunk];
        } else {
            words.xy = ((const device uint2*)(codes + code_base))[chunk];
        }
        float chunk_acc = 0.0f;
        for (uint i = 0; i < uint(CHUNK); ++i) {
            uint codeword = (words[i >> 1u] >> ((i & 1u) * 16u)) & 0xFFFFu;
            uint signs = codeword & 0xFFu;
            uint parity = popcount(signs) & 1u;
            uint effective = signs ^ parity;
            uint abs_code = uint(USE_THREADGROUP_CODEBOOK) != 0u
                ? local_codebook[codeword >> 8u]
                : codebook[codeword >> 8u];
            float4 v_lo = (float4((uint4(abs_code) >> uint4(0u, 4u, 8u, 12u)) & 0xFu) - 8.0f) * 0.5f;
            float4 v_hi = (float4((uint4(abs_code) >> uint4(16u, 20u, 24u, 28u)) & 0xFu) - 8.0f) * 0.5f;
            v_lo *= 1.0f - 2.0f * float4((uint4(effective) >> uint4(0u, 1u, 2u, 3u)) & 1u);
            v_hi *= 1.0f - 2.0f * float4((uint4(effective) >> uint4(4u, 5u, 6u, 7u)) & 1u);
            float offset = parity != 0u ? -0.25f : 0.25f;
            uint xb = x_row + (cw0 + i) * 8u;
            float4 x_even = float4(float(x[xb]), float(x[xb + 2u]), float(x[xb + 4u]), float(x[xb + 6u]));
            float4 x_odd = float4(float(x[xb + 1u]), float(x[xb + 3u]), float(x[xb + 5u]), float(x[xb + 7u]));
            chunk_acc += dot(v_lo + offset, x_even) + dot(v_hi + offset, x_odd);
        }
        acc += chunk_acc * float(scales[scale_base + cw0 / codewords_per_group]);
    }
}

if (uint(LANES_PER_ROW) <= 32u) {
    float total = acc;
    for (uint stride = uint(LANES_PER_ROW) >> 1; stride > 0; stride >>= 1) {
        total += simd_shuffle_down(total, stride);
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
