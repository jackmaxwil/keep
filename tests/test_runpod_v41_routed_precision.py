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
import routed_report  # noqa: E402
import precision_alloc  # noqa: E402
import fit_tau


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


def test_residual_cache_prompt_warm_start():
    # Two prompt tokens route to expert 1, which is not warm and never admitted (tau too high);
    # from token 2 the residents are the prompt's own top k.
    ind = np.array([[1], [1], [1], [0]])
    counts = np.zeros((2, 5))
    hi = local_eval.residual_cache(ind, np.ones((4, 1)), k=1, tau=9.0, bounds=[4], warm=np.array([0]), counts=counts,
                                   policy="prompt:2")
    assert hi[:, 0].tolist() == [False, False, True, False]
    assert counts[:, 4].tolist() == [0, 1]
    lru = local_eval.residual_cache(ind, np.ones((4, 1)), k=1, tau=9.0, bounds=[4], warm=np.array([0]),
                                    counts=np.zeros((2, 5)))
    assert lru[:, 0].tolist() == [False, False, False, True]


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


def test_residual_cache_oracle_and_trace():
    ind = np.array([[0, 1], [2, 1], [2, 0], [3, 3]])
    wts = np.array([[0.9, 0.1], [0.5, 0.1], [0.5, 0.2], [0.3, 0.3]])
    counts = np.zeros((4, 5))
    # record 0 = tokens 0-2: expert 0 has mass 1.1, expert 2 has 1.0, expert 1 has 0.2 -> hot {0, 2}
    hi = local_eval.residual_cache(ind, wts, k=2, tau=0.3, bounds=[3, 1], warm=None, counts=counts, policy="oracle")
    assert hi.tolist() == [[True, False], [True, False], [True, True], [True, True]]
    assert counts[:, 4].sum() == 0  # the oracle never fetches
    trace: list = []
    local_eval.residual_cache(ind, wts, k=2, tau=0.3, bounds=[3, 1], warm=None, counts=counts, trace=trace)
    assert trace == [(0, 0, 0), (0, 1, 2), (1, 0, 3)]  # (record, token in record, expert); gate 0.2 never admits


def test_compare_bootstrap(tmp_path):
    import json

    rng = np.random.default_rng(1)
    sessions = {f"s{i}": {"domain": f"d{i % 2}"} for i in range(4)}
    for name, shift in (("a", 0.1), ("b", 0.0)):
        (tmp_path / name).mkdir()
        for s in sessions:
            np.savez(tmp_path / name / f"{s}.npz", kl=np.abs(np.random.default_rng(hash(s) % 97).normal(1, 0.2, 500)) + shift)
    (tmp_path / "summary.json").write_text(json.dumps({"variants": {"a": sessions, "b": sessions}}))
    out = routed_report.compare(f"{tmp_path}:a", f"{tmp_path}:b", reps=200)
    assert abs(out["kl_diff"] - 0.1) < 1e-9 and out["ci95"][0] <= 0.1 <= out["ci95"][1]


def test_residual_cache_reports_the_resident_set():
    ind = np.array([[0, 1], [2, 1], [2, 0]])
    wts = np.array([[0.9, 0.1], [0.5, 0.1], [0.5, 0.2]])
    resident = np.zeros((3, 4), dtype=bool)
    local_eval.residual_cache(ind, wts, k=2, tau=0.3, bounds=[3], warm=np.array([3, 1]), counts=np.zeros((4, 5)),
                              resident=resident)
    # token 0 sees the warm set {3, 1}; expert 0 is admitted and evicts 3; then 2 evicts 1
    assert resident.tolist() == [[False, True, False, True], [True, True, False, False], [True, False, True, False]]
    resident[:] = False
    local_eval.residual_cache(ind, wts, k=2, tau=0.3, bounds=[3], warm=None, counts=np.zeros((4, 5)),
                              policy="oracle", resident=resident)
    assert resident.all(axis=0).tolist() == [True, False, True, False]  # the record's top two by mass: 0 and 2


def test_gate_masked_routes_among_allowed_experts():
    torch = pytest.importorskip("torch")
    import types

    torch.manual_seed(0)
    gate = types.SimpleNamespace(weight=torch.randn(8, 5), bias=torch.zeros(8), gate_temp=1.0, score_func="sqrtsoftplus",
                                 topk=3, norm_topk_prob=True, route_scale=1.5)
    x = torch.randn(4, 5)
    scores = torch.nn.functional.softplus(x @ gate.weight.T).sqrt()
    w, i = local_eval.gate_masked(gate, x, torch.ones(4, 8, dtype=torch.bool), torch.nn.functional.linear)
    assert torch.equal(i, scores.topk(3, dim=-1)[1])  # nothing masked: the reference router
    assert torch.allclose(w.sum(-1), torch.full((4,), 1.5))
    allowed = torch.ones(4, 8, dtype=torch.bool)
    allowed[:, i[0, 0]] = False
    w2, i2 = local_eval.gate_masked(gate, x, allowed, torch.nn.functional.linear)
    assert not (i2 == i[0, 0]).any() and torch.allclose(w2.sum(-1), torch.full((4,), 1.5))  # re-routed, renormalized
    few = torch.zeros(4, 8, dtype=torch.bool)
    few[:, :2] = True  # fewer allowed than top-k: the surplus slot carries no weight
    w3, i3 = local_eval.gate_masked(gate, x, few, torch.nn.functional.linear)
    assert ((w3 > 0).sum(-1) == 2).all() and (i3[w3 > 0] < 2).all()



def test_hot_set_takes_the_hottest_experts_the_checkpoint_holds():
    assert local_eval.hot_set(np.array([7, 2, 9, 4, 1]), 2, [1, 2, 4]) == {2, 4}
    assert local_eval.hot_set(np.array([7, 2]), 3, [2, 7, 8]) == {2, 7}
    assert routed_report.FP4_BYTES == 18_800_640


def test_residual_cache_takes_one_threshold_per_expert():
    ind = np.array([[0, 1], [0, 1], [0, 1]])
    wts = np.array([[0.4, 0.2], [0.4, 0.2], [0.4, 0.2]], dtype=np.float32)
    same = [local_eval.residual_cache(ind, wts, k=2, tau=tau, bounds=[3], warm=None, counts=np.zeros((2, 5)))
            for tau in (0.3, np.array([0.3, 0.3]))]
    np.testing.assert_array_equal(*same)
    assert same[0].tolist() == [[False, False], [True, False], [True, False]]
    counts = np.zeros((2, 5))
    hi = local_eval.residual_cache(ind, wts, k=2, tau=np.array([0.5, 0.2]), bounds=[3], warm=None, counts=counts)
    assert hi.tolist() == [[False, False], [False, True], [False, True]] and counts[:, 4].tolist() == [0, 1]


def test_fit_tau_admits_early_what_turns_strong_and_never_what_stays_weak():
    # 6 experts, 3 residents (0, 1, 2). Expert 4: a weak slot, then strong ones. Expert 5: one weak slot per record.
    ids = np.array([[0, 4], [0, 5], [0, 4], [0, 4], [0, 4]])[:, None, :]
    gates = np.array([[0.6, 0.2], [0.6, 0.2], [0.3, 0.6], [0.3, 0.6], [0.3, 0.6]], dtype=np.float32)[:, None, :]
    ids, gates, bounds = np.concatenate([ids, ids]), np.concatenate([gates, gates]), [5, 5]
    warm = np.arange(6)[None]
    miss, inst = fit_tau.cost_tables(ids, gates, bounds, warm, 3, 6)
    grid = fit_tau.GRID.tolist()
    assert miss[0, 4, grid.index(0.2)] == 0 and inst[0, 4, grid.index(0.2)] == 2  # admitted before its strong slots
    np.testing.assert_allclose(miss[0, 4, grid.index(0.3)], 2 * 0.35, rtol=1e-6)  # the first strong slot is missed
    np.testing.assert_allclose(miss[0, 4, -1], 2 * 3 * 0.35, rtol=1e-6)  # never admitted: all three
    tau = fit_tau.fit(miss, inst, warm, 3, lam=0.1, prior=0.0, default=0.3)
    assert tau[0, 4] <= 0.2 < tau[0, 5] and tau[0, 0] == tau[0, 1] == 0.3
    learned = fit_tau.replay(ids, gates, bounds, warm, 3, tau, 6)
    single = fit_tau.replay(ids, gates, bounds, warm, 3, 0.3, 6)
    assert learned[0] == 0 < single[0] and learned[1] == single[1]  # no strong slot missed, as many installs


def test_miss_cost_recovers_the_price_and_fit_tau_reads_it(tmp_path):
    import miss_cost

    rng = np.random.default_rng(0)
    ids = np.stack([np.stack([rng.permutation(8)[:2] for _ in range(4000)]) for _ in range(2)])  # [2 layers, N, top 2]
    missed, cost = rng.random(ids.shape) < 0.4, np.stack([np.full(8, 0.5), np.full(8, 2.0)])
    cost[1, 3] = 8.0  # one expert whose miss costs four times its neighbours'
    kl = miss_cost.predict(ids, missed, cost)
    got = miss_cost.fit_cost(ids, missed, kl, 8, prior=0.01)
    np.testing.assert_allclose(got, cost, rtol=0.05)
    flat = miss_cost.fit_cost(ids, missed, kl, 8, prior=None)
    assert (flat == flat[0, 0]).all() and 0.5 < flat[0, 0] < 8
    np.savez(tmp_path / "cost.npz", cost=got)
    price = fit_tau.priced(tmp_path / "cost.npz")
    np.testing.assert_allclose(price(1, np.array([3, 2]), np.array([0.5, 0.12])), [8.0, 2.0], rtol=0.05)
    assert price(1, 3, np.array([0.5, 0.12])).shape == (2,)  # one expert, its slots in time order
