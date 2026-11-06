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

const device half4* abs4 = (const device half4*)abs_tab;
const device ushort4* sign4 = (const device ushort4*)sign_tab;
threadgroup half4 tg_abs[512];
threadgroup ushort4 tg_sign[512];
if (uint(TG_CODEBOOK) != 0u) {
    for (uint idx = tg_lane; idx < 512u; idx += threads_per_threadgroup.x) {
        tg_abs[idx] = abs4[idx];
        tg_sign[idx] = sign4[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

uint row0 = (row_tile * uint(SIMDGROUPS) + simd_id) * uint(ROWS_PER_SG);
float acc[ROWS_PER_SG];
for (uint r = 0; r < uint(ROWS_PER_SG); ++r) {
    acc[r] = 0.0f;
}

for (uint cw0 = simd_lane * uint(CHUNK); cw0 < codewords; cw0 += 32u * uint(CHUNK)) {
    float4 xe[CHUNK * 2];
    float xs[CHUNK];
    for (uint i = 0; i < uint(CHUNK); ++i) {
        uint xb = x_row + (cw0 + i) * 8u;
        xe[2 * i] = float4(float(x[xb]), float(x[xb + 1u]), float(x[xb + 2u]), float(x[xb + 3u]));
        xe[2 * i + 1] = float4(float(x[xb + 4u]), float(x[xb + 5u]), float(x[xb + 6u]), float(x[xb + 7u]));
        xs[i] = dot(xe[2 * i] + xe[2 * i + 1], float4(1.0f));
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
            uint a = (codeword >> 8u) * 2u;
            uint sg = (codeword & 0xFFu) * 2u;
            half4 a0, a1;
            ushort4 s0, s1;
            if (uint(TG_CODEBOOK) != 0u) {
                a0 = tg_abs[a]; a1 = tg_abs[a + 1u]; s0 = tg_sign[sg]; s1 = tg_sign[sg + 1u];
            } else {
                a0 = abs4[a]; a1 = abs4[a + 1u]; s0 = sign4[sg]; s1 = sign4[sg + 1u];
            }
            float4 w0 = float4(as_type<half4>(as_type<ushort4>(a0) ^ s0));
            float4 w1 = float4(as_type<half4>(as_type<ushort4>(a1) ^ s1));
            float offset = (popcount(codeword & 0xFFu) & 1u) != 0u ? -0.25f : 0.25f;
            part += dot(w0, xe[2 * i]) + dot(w1, xe[2 * i + 1]) + offset * xs[i];
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
