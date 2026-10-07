inline float mlx_vq_decode_packed_int2_value(uint packed, uint dim) {
    uint nibble = (packed >> (4 * dim)) & 0xFu;
    return (float(nibble) - 8.0f) * 0.5f;
}

inline uint mlx_vq_e8p_shuffle_dim(uint dim) {
    switch (dim) {
        case 0: return 0;
        case 1: return 4;
        case 2: return 1;
        case 3: return 5;
        case 4: return 2;
        case 5: return 6;
        case 6: return 3;
        default: return 7;
    }
}

inline uint mlx_vq_sign_parity8(uint signs) {
    uint parity = 0;
    for (uint bit = 0; bit < 8; ++bit) {
        parity ^= (signs >> bit) & 1u;
    }
    return parity;
}

inline float mlx_vq_decode_e8p_value(uint codeword, const device uint* packed_abs_grid, uint dim) {
    uint signs = codeword & 0xFFu;
    uint abs_idx = codeword >> 8;
    uint parity = mlx_vq_sign_parity8(signs);
    uint effective_signs = signs ^ parity;
    uint packed_dim = mlx_vq_e8p_shuffle_dim(dim);
    uint abs_code = packed_abs_grid[abs_idx];
    float value = mlx_vq_decode_packed_int2_value(abs_code, packed_dim);
    if (((effective_signs >> packed_dim) & 1u) != 0) {
        value = -value;
    }
    return value + (parity != 0 ? -0.25f : 0.25f);
}

inline float mlx_vq_decode_e8p_value_tg(
    uint codeword,
    threadgroup const uint* packed_abs_grid,
    uint duplicate_base,
    uint dim
) {
    uint signs = codeword & 0xFFu;
    uint abs_idx = codeword >> 8;
    uint parity = mlx_vq_sign_parity8(signs);
    uint effective_signs = signs ^ parity;
    uint packed_dim = mlx_vq_e8p_shuffle_dim(dim);
    uint abs_code = packed_abs_grid[duplicate_base + abs_idx];
    float value = mlx_vq_decode_packed_int2_value(abs_code, packed_dim);
    if (((effective_signs >> packed_dim) & 1u) != 0) {
        value = -value;
    }
    return value + (parity != 0 ? -0.25f : 0.25f);
}
