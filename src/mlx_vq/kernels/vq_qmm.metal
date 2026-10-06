uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint batch = threadgroup_position_in_grid.z;

uint codewords = uint(codes_shape[1]);
uint groups = uint(scales_shape[1]);
uint codewords_per_group = codewords / groups;
uint duplicate_base = (lane % uint(CODEBOOK_DUP)) * 256u;
float acc = 0.0f;

threadgroup uint local_codebook[2048];
for (uint idx = lane; idx < uint(CODEBOOK_DUP) * 256u; idx += threads_per_threadgroup.x) {
    local_codebook[idx] = codebook[idx & 255u];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint x_base = batch * codewords * 8u;

for (uint cw = lane; cw < codewords; cw += threads_per_threadgroup.x) {
    uint codeword = uint(codes[row * codewords + cw]);
    uint scale_idx = row * groups + (cw / codewords_per_group);
    float scale = float(scales[scale_idx]);

    for (uint dim = 0; dim < 8; ++dim) {
        float weight;
        if (CODE_BITS == 8) {
            uint packed = local_codebook[duplicate_base + codeword];
            weight = mlx_vq_decode_packed_int2_value(packed, dim);
        } else {
            weight = mlx_vq_decode_e8p_value_tg(codeword, local_codebook, duplicate_base, dim);
        }
        acc += weight * scale * float(x[x_base + cw * 8u + dim]);
    }
}

threadgroup float partials[256];
partials[lane] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = threads_per_threadgroup.x >> 1; stride > 0; stride >>= 1) {
    if (lane < stride) {
        partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (lane == 0) {
    out[batch * uint(codes_shape[0]) + row] = OUT_T(partials[0]);
}
