// Fused weighted nearest-codebook search under a diagonal metric.
//
// One thread per input row. Each thread scans all K codebook rows in registers
// and keeps the argmin of the diagonal-weighted squared distance
//   dist_j = sum_i w_i * (v_i - codebook_ji)^2
// No [N,K,8] scratch tensor, so the kernel is compute-bound. Byte-compatible
// with nearest_codebook_indices_diagonal_hessian (argmin keeps the lowest
// codebook index on a tie).
//
// Inputs:
//   x        : float32 [N, 8]  input vectors
//   w        : float32 [N, 8]  per-row diagonal metric (shared metric is
//                              broadcast to per-row by the Python wrapper)
//   codebook : float32 [K, 8]  codebook rows
// Output:
//   out      : uint32  [N]     argmin codebook index in [0, K)

uint n = thread_position_in_grid.x;
uint total = uint(x_shape[0]);
if (n >= total) {
    return;
}
uint k_rows = uint(codebook_shape[0]);

float v[8];
float wv[8];
for (uint i = 0; i < 8; ++i) {
    v[i] = float(x[n * 8 + i]);
    wv[i] = float(w[n * 8 + i]);
}

float best_dist = 1.0e30f;
uint best_j = 0u;
for (uint j = 0; j < k_rows; ++j) {
    float dist = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        float e = v[i] - codebook[j * 8 + i];
        dist += wv[i] * (e * e);
    }
    // Strict '<' keeps the lowest index on a tie (matches numpy argmin).
    if (dist < best_dist) {
        best_dist = dist;
        best_j = j;
    }
}

out[n] = best_j;
