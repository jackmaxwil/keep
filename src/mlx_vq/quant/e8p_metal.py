from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import CODEWORD_DIM, E8P_SHUFFLE_MAP, e8p_abs_grid

_KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"
_FUSED_SEARCH_KERNEL = None


def _fused_search_kernel():
    """Lazily compile the fused per-codeword E8P search kernel."""
    global _FUSED_SEARCH_KERNEL
    if _FUSED_SEARCH_KERNEL is None:
        _FUSED_SEARCH_KERNEL = mx.fast.metal_kernel(
            name="mlx_vq_e8p_fused_search",
            input_names=["x", "w", "abs_rows", "base_signs"],
            output_names=["out"],
            source=(_KERNEL_DIR / "e8p_fused_search.metal").read_text(),
            header=(_KERNEL_DIR / "common_e8.metal").read_text(),
        )
    return _FUSED_SEARCH_KERNEL


_NEAREST_DIAG_KERNEL = None


def _nearest_diag_kernel():
    """Lazily compile the fused weighted nearest-codebook search kernel."""
    global _NEAREST_DIAG_KERNEL
    if _NEAREST_DIAG_KERNEL is None:
        _NEAREST_DIAG_KERNEL = mx.fast.metal_kernel(
            name="mlx_vq_nearest_diag_search",
            input_names=["x", "w", "codebook"],
            output_names=["out"],
            source=(_KERNEL_DIR / "nearest_diag_search.metal").read_text(),
        )
    return _NEAREST_DIAG_KERNEL


def nearest_codebook_diagonal_hessian_fused_mlx(
    vectors: np.ndarray,
    diagonal: np.ndarray,
    codebook: np.ndarray,
    *,
    index_dtype: np.dtype = np.dtype(np.int64),
) -> np.ndarray:
    """Fused weighted nearest-codebook argmin over an arbitrary [K,8] codebook.

    Compute-bound drop-in for nearest_codebook_indices_diagonal_hessian on the
    8-bit / general conversion path. `diagonal` may be shared (8,) or per-row
    (n_rows, 8). Returns argmin indices (lowest index wins ties).
    """
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[-1] != CODEWORD_DIM:
        raise ValueError(f"vectors must have trailing dimension {CODEWORD_DIM}")
    cb = np.ascontiguousarray(np.asarray(codebook, dtype=np.float32))
    if cb.ndim != 2 or cb.shape[1] != CODEWORD_DIM:
        raise ValueError(f"codebook must have shape (K, {CODEWORD_DIM})")
    flat = np.ascontiguousarray(values.reshape(-1, CODEWORD_DIM))
    n_rows = flat.shape[0]
    if n_rows == 0:
        return np.zeros(values.shape[:-1], dtype=index_dtype)

    diagonal_np = np.asarray(diagonal, dtype=np.float32)
    if diagonal_np.shape == (CODEWORD_DIM,):
        w_all = np.ascontiguousarray(np.broadcast_to(diagonal_np, (n_rows, CODEWORD_DIM)))
    elif diagonal_np.shape == (n_rows, CODEWORD_DIM):
        w_all = np.ascontiguousarray(diagonal_np)
    else:
        raise ValueError("diagonal must have shape (8,) or (n_rows, 8)")
    if not np.isfinite(w_all).all() or np.any(w_all < 0):
        raise ValueError("diagonal must be finite and non-negative")

    threadgroup = 256
    grid_x = ((n_rows + threadgroup - 1) // threadgroup) * threadgroup
    out = _nearest_diag_kernel()(
        inputs=[mx.array(flat), mx.array(w_all), mx.array(cb)],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n_rows,)],
        output_dtypes=[mx.uint32],
    )[0]
    mx.eval(out)
    return np.asarray(out).astype(index_dtype).reshape(values.shape[:-1])


def _e8p_base_grid_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return (abs_rows[256,8] float32, base_signs[256] uint32) for the kernel."""
    base = e8p_abs_grid().astype(np.float32)
    abs_rows = np.ascontiguousarray(np.abs(base), dtype=np.float32)
    base_signs = np.zeros(base.shape[0], dtype=np.uint32)
    negative = base < 0
    for out_dim, packed_dim in enumerate(E8P_SHUFFLE_MAP.astype(np.uint32)):
        base_signs |= negative[:, out_dim].astype(np.uint32) << packed_dim
    return abs_rows, base_signs


def encode_e8p_diagonal_hessian_fused_mx(x: mx.array, w: mx.array) -> mx.array:
    """Device-resident twin of :func:`encode_e8p_diagonal_hessian_fused_mlx`.

    Both arguments stay on the GPU and the ``uint32`` codes stay there too, so a
    caller that already holds its vectors on device (the DeepSeek-V4 MLX fit
    loop) never pays a host round-trip per iteration. Exactly the same kernel,
    so the codes are the same bytes; the only thing dropped relative to the
    NumPy entry point is the finite/non-negative validation of ``w``, which
    would force a device synchronisation on every call. Callers must have
    validated the importance vector once, up front -- which is what
    ``_validate_importance`` does before the fit starts.

    ``x`` and ``w`` must both be contiguous ``[N, 8]`` float32 arrays.
    """

    if x.ndim != 2 or x.shape[1] != CODEWORD_DIM:
        raise ValueError(f"x must have shape (N, {CODEWORD_DIM}), found {x.shape}")
    if w.shape != x.shape:
        raise ValueError("w must have the same shape as x")
    if x.dtype != mx.float32 or w.dtype != mx.float32:
        raise ValueError("x and w must be float32")
    n_rows = x.shape[0]
    if n_rows == 0:
        return mx.zeros((0,), dtype=mx.uint32)

    abs_rows, base_signs = _e8p_base_grid_arrays()
    threadgroup = 256
    grid_x = ((n_rows + threadgroup - 1) // threadgroup) * threadgroup
    return _fused_search_kernel()(
        inputs=[x, w, mx.array(abs_rows), mx.array(base_signs)],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n_rows,)],
        output_dtypes=[mx.uint32],
    )[0]


def encode_e8p_diagonal_hessian_fused_mlx(
    vectors: np.ndarray, diagonal: np.ndarray
) -> np.ndarray:
    """Compute-bound fused E8P search: one GPU thread per codeword row.

    Byte-identical to encode_e8p_rtn_diagonal_hessian; avoids the [N,256,8]
    scratch tensor of the broadcast path so it is compute-bound and scales with
    batch size. `diagonal` may be a shared (8,) metric or per-row (n_rows, 8).
    """
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[-1] != CODEWORD_DIM:
        raise ValueError(f"vectors must have trailing dimension {CODEWORD_DIM}")
    flat = np.ascontiguousarray(values.reshape(-1, CODEWORD_DIM))
    n_rows = flat.shape[0]
    if n_rows == 0:
        return np.zeros(values.shape[:-1], dtype=np.uint16)

    diagonal_np = np.asarray(diagonal, dtype=np.float32)
    if diagonal_np.shape == (CODEWORD_DIM,):
        w_all = np.ascontiguousarray(np.broadcast_to(diagonal_np, (n_rows, CODEWORD_DIM)))
    elif diagonal_np.shape == (n_rows, CODEWORD_DIM):
        w_all = np.ascontiguousarray(diagonal_np)
    else:
        raise ValueError("diagonal must have shape (8,) or (n_rows, 8)")
    if not np.isfinite(w_all).all() or np.any(w_all < 0):
        raise ValueError("diagonal must be finite and non-negative")

    out = encode_e8p_diagonal_hessian_fused_mx(mx.array(flat), mx.array(w_all))
    mx.eval(out)
    return np.asarray(out).astype(np.uint16).reshape(values.shape[:-1])


def encode_e8p_diagonal_hessian_mlx(
    vectors: np.ndarray, diagonal: np.ndarray, *, row_block: int = 131_072
) -> np.ndarray:
    """Encode E8P vectors with an MLX factored search under a diagonal metric."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[-1] != CODEWORD_DIM:
        raise ValueError(f"vectors must have trailing dimension {CODEWORD_DIM}")
    diagonal_np = np.asarray(diagonal, dtype=np.float32)
    if (
        diagonal_np.shape != (CODEWORD_DIM,)
        or not np.isfinite(diagonal_np).all()
        or np.any(diagonal_np < 0)
        or not np.any(diagonal_np > 0)
    ):
        raise ValueError("diagonal must be finite, non-negative, shape (8,), and nonzero")
    if row_block <= 0:
        raise ValueError("row_block must be positive")

    flat_np = values.reshape(-1, CODEWORD_DIM)
    base_rows_np = e8p_abs_grid().astype(np.float32)
    abs_rows = mx.array(np.abs(base_rows_np))
    w = mx.array(diagonal_np)

    packed = E8P_SHUFFLE_MAP.astype(np.uint16)
    base_signs_np = np.zeros(base_rows_np.shape[0], dtype=np.uint16)
    for out_dim, packed_dim in enumerate(packed):
        base_signs_np |= (base_rows_np[:, out_dim] < 0).astype(np.uint16) << packed_dim
    base_signs = mx.array(base_signs_np.astype(np.uint32))

    out = np.empty(flat_np.shape[0], dtype=np.uint16)
    for start in range(0, flat_np.shape[0], row_block):
        chunk = mx.array(flat_np[start : start + row_block])
        local_best_dist = mx.full((chunk.shape[0],), float("inf"), dtype=mx.float32)
        local_best_codes = mx.zeros((chunk.shape[0],), dtype=mx.uint32)

        for parity, shift in ((0, np.float32(0.25)), (1, np.float32(-0.25))):
            target = chunk - shift
            abs_target = mx.abs(target)
            # Distance via fp32 elementwise broadcast + sum over the 8-dim, NOT a
            # matmul: MLX's Metal float32 matmul uses reduced-precision
            # accumulation (~1e-2 abs error) that flips argmin on near-tie rows.
            # sum_i w_i (|t_i| - a_ji)^2 in fp32 matches the NumPy reference to
            # ULP, so argmin agrees byte-for-byte. [chunk,256,8] is bounded by
            # row_block.
            diff = abs_target[:, None, :] - abs_rows[None, :, :]
            distances = mx.sum(w[None, None, :] * diff * diff, axis=2)

            negative = target < 0
            target_signs = mx.zeros((chunk.shape[0],), dtype=mx.uint32)
            for out_dim, packed_dim in enumerate(packed):
                target_signs = target_signs | (
                    negative[:, out_dim].astype(mx.uint32) << int(packed_dim)
                )

            effective_signs = target_signs[:, None] ^ base_signs[None, :]
            sign_parity = mx.zeros(effective_signs.shape, dtype=mx.uint32)
            for bit in range(CODEWORD_DIM):
                sign_parity = sign_parity ^ ((effective_signs >> bit) & 1)
            needs_flip = sign_parity.astype(mx.bool_)

            weighted_target = abs_target * w[None, :]
            correction = None
            selected_flip_code = None
            selected_flip_mask = None
            for out_dim, packed_dim in enumerate(packed):
                penalty = (
                    4.0
                    * weighted_target[:, out_dim][:, None]
                    * abs_rows[:, out_dim][None, :]
                )
                flip_mask = np.uint32(1 << int(packed_dim))
                flipped_code = effective_signs ^ flip_mask
                if correction is None:
                    correction = penalty
                    selected_flip_code = flipped_code
                    selected_flip_mask = mx.full(
                        effective_signs.shape, flip_mask, dtype=mx.uint32
                    )
                else:
                    assert selected_flip_code is not None
                    assert selected_flip_mask is not None
                    take = (penalty < correction) | (
                        (penalty == correction) & (flipped_code < selected_flip_code)
                    )
                    correction = mx.where(take, penalty, correction)
                    selected_flip_code = mx.where(take, flipped_code, selected_flip_code)
                    selected_flip_mask = mx.where(take, flip_mask, selected_flip_mask)

            assert correction is not None
            assert selected_flip_mask is not None
            distances = mx.where(needs_flip, distances + correction, distances)
            abs_idx = mx.argmin(distances, axis=1)
            row_dist = mx.take_along_axis(distances, abs_idx[:, None], axis=1)[:, 0]
            candidate_signs = mx.take_along_axis(
                effective_signs, abs_idx[:, None], axis=1
            )[:, 0]
            flip_masks = mx.take_along_axis(
                selected_flip_mask, abs_idx[:, None], axis=1
            )[:, 0]
            selected_needs_flip = mx.take_along_axis(
                needs_flip, abs_idx[:, None], axis=1
            )[:, 0]
            candidate_signs = mx.where(
                selected_needs_flip, candidate_signs ^ flip_masks, candidate_signs
            )

            stored_signs = candidate_signs ^ np.uint32(parity)
            codes = (abs_idx.astype(mx.uint32) << 8) | stored_signs
            better = row_dist < local_best_dist
            tie = row_dist == local_best_dist
            lower_code = codes < local_best_codes
            take = better | (tie & lower_code)
            local_best_dist = mx.where(take, row_dist, local_best_dist)
            local_best_codes = mx.where(take, codes, local_best_codes)

        mx.eval(local_best_codes)
        out[start : start + int(chunk.shape[0])] = np.asarray(local_best_codes).astype(
            np.uint16
        )

    return out.reshape(values.shape[:-1])
