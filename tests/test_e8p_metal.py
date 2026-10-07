from __future__ import annotations

import numpy as np
import pytest

from mlx_vq.codebook.e8 import encode_e8p_rtn_diagonal_hessian

mx = pytest.importorskip("mlx.core")


def _metal_ok() -> bool:
    metal = getattr(mx, "metal", None)
    try:
        if metal is None or not bool(metal.is_available()):
            return False
        mx.eval(mx.zeros((), stream=mx.gpu))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
def test_mlx_matches_numpy_reference_bit_for_bit() -> None:
    from mlx_vq.quant.e8p_metal import encode_e8p_diagonal_hessian_mlx

    rng = np.random.default_rng(20260711)
    for _ in range(3):
        vecs = (rng.standard_normal((5000, 8)) * 0.6).astype(np.float32)
        diag = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
        ref = encode_e8p_rtn_diagonal_hessian(vecs, diag)
        got = encode_e8p_diagonal_hessian_mlx(vecs, diag)
        np.testing.assert_array_equal(got, ref)
