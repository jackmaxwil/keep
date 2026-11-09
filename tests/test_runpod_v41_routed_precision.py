"""Routed-precision pieces of the V4.1 REAP recovery kit: the residual cache, the reduced-rank
sidecar fit, the static precision allocator and the nested-code decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "runpod" / "v41_reap"))
sys.path.insert(0, str(REPO / "src"))

import local_eval  # noqa: E402
import precision_alloc  # noqa: E402


def test_residual_cache_hits_only_after_async_admission():
    ind = np.array([[0, 1], [0, 1], [2, 0]])
    wts = np.array([[0.5, 0.1], [0.5, 0.1], [0.5, 0.5]])
    counts = np.zeros((3, 5))
    hi = local_eval.residual_cache(ind, wts, k=1, tau=0.3, bounds=[3], warm=None, counts=counts)
    # token 0: cold; expert 0 admitted. token 1: expert 0 resident, expert 1 never admitted (gate < tau).
    # token 2: expert 2 not yet resident, expert 0 still is. Then 2 is admitted (evicting 0, k=1) and 0
    # is admitted again (evicting 2): one slot of cache thrashes, and each admission is a fetch.
    assert hi.tolist() == [[False, False], [True, False], [False, True]]
    assert counts[:, 4].tolist() == [2, 0, 1]


def test_residual_cache_resets_per_record_and_warm_start():
    ind = np.array([[0], [0], [0]])
    wts = np.ones((3, 1))
    counts = np.zeros((1, 5))
    hi = local_eval.residual_cache(ind, wts, k=1, tau=0.3, bounds=[2, 1], warm=None, counts=counts)
    assert hi[:, 0].tolist() == [False, True, False]  # the second record starts cold again
    assert counts[0, 4] == 2
    hi = local_eval.residual_cache(ind, wts, k=1, tau=0.3, bounds=[3], warm=np.array([0]), counts=np.zeros((1, 5)))
    assert hi.all()


def test_precision_allocator():
    precision_alloc._check()


def test_sidecar_reduced_rank():
    pytest.importorskip("torch")
    import sidecar_fit

    sidecar_fit._check()


def test_nested_decoder_matches_local_eval():
    import keep_quant
    from keep.vq.e8 import e8_1bit_grid

    rng = np.random.default_rng(0)
    z = {"code_bits": np.asarray(8), "w1_codes": rng.integers(0, 256, (2, 4, 64), dtype=np.uint16),
         "w1_scales": rng.random((2, 4, 1)).astype(np.float16)}
    got = keep_quant.decode_codes(z, "w1", 1)
    grid = e8_1bit_grid().astype(np.float32)
    want = (grid[z["w1_codes"][1].astype(np.int64)] * z["w1_scales"][1].astype(np.float32)[..., None]).reshape(4, -1)
    assert np.allclose(got, want)
