from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vq.codebook.e8 import e8_1bit_grid
from mlx_vq.quant.rtn import nearest_codebook_indices_diagonal_hessian


def _metal_ok() -> bool:
    metal = getattr(mx, "metal", None)
    try:
        return metal is not None and bool(metal.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
def test_fused_nearest_matches_reference_on_e8_grid() -> None:
    from mlx_vq.quant.e8p_metal import nearest_codebook_diagonal_hessian_fused_mlx as fused

    grid = e8_1bit_grid().astype(np.float32)
    rng = np.random.default_rng(11)
    for _ in range(3):
        v = (rng.standard_normal((5000, 8)) * 0.6).astype(np.float32)
        d = (np.abs(rng.standard_normal(8)) + 0.05).astype(np.float32)
        ref = nearest_codebook_indices_diagonal_hessian(
            v, d, codebook=grid, index_dtype=np.dtype(np.uint8),
            vector_chunk_size=8192, codebook_chunk_size=8192,
        )
        got = fused(v, d, grid, index_dtype=np.dtype(np.uint8))
        np.testing.assert_array_equal(got, ref)


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
def test_fused_nearest_per_row_diagonal() -> None:
    from mlx_vq.quant.e8p_metal import nearest_codebook_diagonal_hessian_fused_mlx as fused

    grid = e8_1bit_grid().astype(np.float32)
    rng = np.random.default_rng(12)
    v = (rng.standard_normal((400, 8)) * 0.6).astype(np.float32)
    diagonals = (np.abs(rng.standard_normal((400, 8))) + 0.05).astype(np.float32)
    ref = np.array(
        [
            nearest_codebook_indices_diagonal_hessian(
                v[i : i + 1], diagonals[i], codebook=grid,
                index_dtype=np.dtype(np.uint8), vector_chunk_size=1, codebook_chunk_size=8192,
            )[0]
            for i in range(v.shape[0])
        ],
        dtype=np.uint8,
    )
    got = fused(v, diagonals, grid, index_dtype=np.dtype(np.uint8))
    np.testing.assert_array_equal(got, ref)
