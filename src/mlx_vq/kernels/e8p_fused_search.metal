// Fused E8P diagonal-Hessian nearest-code search.
//
// One thread per codeword row. Each thread runs the full 256-abs-row factored
// search entirely in registers (no [N,256,8] materialization), so the kernel is
// compute-bound rather than memory-bandwidth-bound. Byte-identical to
// encode_e8p_rtn_diagonal_hessian:
//   - weighted distance  sum_i w_i * (|t_i| - a_ji)^2
//   - even-parity fix: add the minimum flip penalty 4 * (|t_i|*w_i) * a_ji,
//     tie-broken to the lowest resulting packed sign code
//   - within a parity branch, argmin keeps the lowest abs-row index on a tie
//   - across the two +/-0.25 parity branches, ties break to the lower code
//
// Inputs:
//   x         : float32 [N, 8]  target vectors (normalized weights)
//   w         : float32 [N, 8]  per-row diagonal metric (shared metric is
//                               broadcast to per-row by the Python wrapper)
//   abs_rows  : float32 [256, 8] |base grid values| in output-dim order
//   base_signs: uint32  [256]    sign bits at packed positions for the base grid
// Output:
//   out       : uint32  [N]      packed E8P codeword (fits in uint16)

uint n = thread_position_in_grid.x;
uint total = uint(x_shape[0]);
if (n >= total) {
    return;
}

float t[8];
float wv[8];
for (uint i = 0; i < 8; ++i) {
    t[i] = float(x[n * 8 + i]);
    wv[i] = float(w[n * 8 + i]);
}

float best_dist = 1.0e30f;
uint best_code = 0u;

for (uint p = 0; p < 2u; ++p) {
    float shift = (p == 0u) ? 0.25f : -0.25f;
    float ta[8];
    uint target_signs = 0u;
    for (uint i = 0; i < 8; ++i) {
        float d = t[i] - shift;
        ta[i] = fabs(d);
        if (d < 0.0f) {
            target_signs |= (1u << mlx_vq_e8p_shuffle_dim(i));
        }
    }

    float bp_dist = 1.0e30f;
    uint bp_code = 0u;
    for (uint j = 0; j < 256u; ++j) {
        float dist = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            float e = ta[i] - abs_rows[j * 8 + i];
            dist += wv[i] * (e * e);
        }
        uint eff = target_signs ^ base_signs[j];
        uint parity = mlx_vq_sign_parity8(eff & 0xFFu);
        uint cand_signs = eff;
        if (parity != 0u) {
            float best_pen = 1.0e30f;
            uint best_flip = eff;
            for (uint i = 0; i < 8; ++i) {
                // Match the reference FP order exactly: (4 * (|t|*w)) * a.
                float tw = ta[i] * wv[i];
                float pen = (4.0f * tw) * abs_rows[j * 8 + i];
                uint flipped = eff ^ (1u << mlx_vq_e8p_shuffle_dim(i));
                if (pen < best_pen || (pen == best_pen && flipped < best_flip)) {
                    best_pen = pen;
                    best_flip = flipped;
                }
            }
            dist += best_pen;
            cand_signs = best_flip;
        }
        uint stored = (cand_signs ^ p) & 0xFFu;
        uint code = (j << 8) | stored;
        // Within a parity branch: strict '<' keeps the lowest abs-row on a tie
        // (matches numpy argmin first-min behavior).
        if (dist < bp_dist) {
            bp_dist = dist;
            bp_code = code;
        }
    }

    // Across parity branches: lower distance wins, ties break to the lower code.
    if (bp_dist < best_dist || (bp_dist == best_dist && bp_code < best_code)) {
        best_dist = bp_dist;
        best_code = bp_code;
    }
}

out[n] = best_code;
