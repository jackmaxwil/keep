"""One-dispatch Sinkhorn normalisation for DeepSeek-V4 hyper-connections.

Every hyper-connection (two per block: 86 calls per decoded token on
DeepSeek-V4-Flash) turns ``hc_mult x hc_mult`` mixing logits into a doubly
stochastic matrix: a row softmax, a column normalisation, then
``sinkhorn_iters - 1`` alternating row and column normalisations. The
compiled MLX path still launches a few kernels per iteration because each
normalisation is a reduction, so a 4x4 matrix costs dozens of dispatches per
call. The roadmap measured 30 to 70 ms of sinkhorn per decoded token.

:func:`fused_hc_sinkhorn` does the whole thing in one Metal dispatch, one
thread per matrix, holding the 16 values in registers. It follows the MLX
path's operation order, but exp and summation order are not guaranteed
bit-identical, so it is **opt-in** (``RAMP_HC_SINKHORN=metal``) until the
parity and speed check in ``benchmarks/bench_hc_sinkhorn.py`` passes on the
target Mac.

:func:`hc_sinkhorn_reference` is a float32 NumPy statement of the same
algorithm; tests pin it against the model's compiled path, and the Metal
kernel against both.
"""

from __future__ import annotations

import os
from functools import cache

import numpy as np

__all__ = [
    "HC_SINKHORN_ENV",
    "fused_hc_sinkhorn",
    "fused_hc_sinkhorn_enabled",
    "hc_sinkhorn_reference",
]

HC_SINKHORN_ENV = "RAMP_HC_SINKHORN"
_BACKENDS = ("mlx", "metal")

_SOURCE = """
    uint row = thread_position_in_grid.x;
    if (row >= rows[0]) {
        return;
    }
    const uint n = HC * HC;
    const uint base = row * n;
    const float e = eps[0];
    float m[HC * HC];
    for (uint k = 0; k < n; ++k) {
        m[k] = static_cast<float>(logits[base + k]);
    }
    // Row softmax (max-subtracted), then + eps.
    for (uint i = 0; i < HC; ++i) {
        float peak = m[i * HC];
        for (uint j = 1; j < HC; ++j) {
            peak = metal::max(peak, m[i * HC + j]);
        }
        float total = 0.0f;
        for (uint j = 0; j < HC; ++j) {
            float v = metal::precise::exp(m[i * HC + j] - peak);
            m[i * HC + j] = v;
            total += v;
        }
        for (uint j = 0; j < HC; ++j) {
            m[i * HC + j] = m[i * HC + j] / total + e;
        }
    }
    // Column normalisation.
    for (uint j = 0; j < HC; ++j) {
        float total = 0.0f;
        for (uint i = 0; i < HC; ++i) {
            total += m[i * HC + j];
        }
        for (uint i = 0; i < HC; ++i) {
            m[i * HC + j] = m[i * HC + j] / (total + e);
        }
    }
    // ITERS - 1 alternating row / column normalisations.
    for (uint it = 1; it < ITERS; ++it) {
        for (uint i = 0; i < HC; ++i) {
            float total = 0.0f;
            for (uint j = 0; j < HC; ++j) {
                total += m[i * HC + j];
            }
            for (uint j = 0; j < HC; ++j) {
                m[i * HC + j] = m[i * HC + j] / (total + e);
            }
        }
        for (uint j = 0; j < HC; ++j) {
            float total = 0.0f;
            for (uint i = 0; i < HC; ++i) {
                total += m[i * HC + j];
            }
            for (uint i = 0; i < HC; ++i) {
                m[i * HC + j] = m[i * HC + j] / (total + e);
            }
        }
    }
    for (uint k = 0; k < n; ++k) {
        comb[base + k] = m[k];
    }
"""


def fused_hc_sinkhorn_enabled() -> bool:
    """Whether ``RAMP_HC_SINKHORN=metal`` selects the fused kernel."""

    value = os.environ.get(HC_SINKHORN_ENV, "mlx").strip().lower() or "mlx"
    if value not in _BACKENDS:
        raise ValueError(f"{HC_SINKHORN_ENV} must be one of {_BACKENDS}, got {value!r}")
    return value == "metal"


@cache
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="ramp_hc_sinkhorn",
        input_names=["logits", "eps", "rows"],
        output_names=["comb"],
        source=_SOURCE,
    )


def fused_hc_sinkhorn(logits, *, iters: int, eps: float):
    """Doubly-stochastic ``[..., hc, hc]`` mixing matrices in one dispatch.

    ``logits`` are the scaled, biased comb logits (float32); ``iters`` is the
    model's ``hc_sinkhorn_iters``. Returns float32 of the same shape.
    """

    import mlx.core as mx

    if logits.ndim < 2 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError(f"expected [..., hc, hc] logits, got {tuple(logits.shape)}")
    if iters < 1:
        raise ValueError("iters must be at least 1")
    hc = int(logits.shape[-1])
    rows = int(np.prod(logits.shape[:-2])) if logits.ndim > 2 else 1
    flat = mx.contiguous(logits.astype(mx.float32).reshape(rows, hc * hc))
    (out,) = _kernel()(
        inputs=[flat, mx.array([eps], dtype=mx.float32), mx.array([rows], dtype=mx.uint32)],
        template=[("HC", hc), ("ITERS", int(iters))],
        grid=(rows, 1, 1),
        threadgroup=(min(rows, 64), 1, 1),
        output_shapes=[(rows, hc * hc)],
        output_dtypes=[mx.float32],
    )
    return out.reshape(logits.shape)


def hc_sinkhorn_reference(logits: np.ndarray, *, iters: int, eps: float) -> np.ndarray:
    """The model's sinkhorn in float32 NumPy, step for step."""

    m = np.asarray(logits, dtype=np.float32)
    m = np.exp(m - m.max(axis=-1, keepdims=True), dtype=np.float32)
    m = m / m.sum(axis=-1, keepdims=True, dtype=np.float32) + np.float32(eps)
    m = m / (m.sum(axis=-2, keepdims=True, dtype=np.float32) + np.float32(eps))
    for _ in range(max(int(iters) - 1, 0)):
        m = m / (m.sum(axis=-1, keepdims=True, dtype=np.float32) + np.float32(eps))
        m = m / (m.sum(axis=-2, keepdims=True, dtype=np.float32) + np.float32(eps))
    return m.astype(np.float32)
