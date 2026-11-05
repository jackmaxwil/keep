"""Replay of routing through an in-RAM hot set: the SSD-tier kill test."""

from __future__ import annotations

import json

import numpy as np
import pytest

from keep.quality.expert_residency import (
    RouteRecorder,
    decode_tokens_per_second,
    residency_report,
    simulate_residency,
    simulate_sessions,
)


def _routes(tokens, layers, top_k, experts, seed=0, skew=1.2):
    """Zipf-ish routing so a hot set is meaningful."""

    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, experts + 1) ** skew
    weights /= weights.sum()
    ids = np.empty((tokens, layers, top_k), dtype=np.int64)
    for t in range(tokens):
        for layer in range(layers):
            ids[t, layer] = rng.choice(experts, size=top_k, replace=False, p=weights)
    gates = rng.dirichlet(np.ones(top_k), size=(tokens, layers))
    return ids, gates


def test_everything_resident_never_misses_once_warm():
    ids, _ = _routes(50, 2, 3, 8)
    order = np.zeros((2, 8))
    warm = simulate_residency(ids, num_experts=8, hot_per_layer=8, policy="decayed", initial_order=order)
    assert warm.misses == 0 and warm.miss_rate == 0.0
    # A cold start pays exactly one compulsory miss per distinct expert.
    cold = simulate_residency(ids, num_experts=8, hot_per_layer=8, policy="decayed")
    assert cold.misses == sum(len(np.unique(ids[:, layer])) for layer in range(2))


def test_hand_checked_lru_and_static_counts():
    # One layer, top-1, two slots: A B A C A  (experts 0 1 0 2 0)
    ids = np.array([0, 1, 0, 2, 0]).reshape(5, 1, 1)
    lru = simulate_residency(ids, num_experts=3, hot_per_layer=2, policy="lru")
    assert lru.misses == 3  # cold 0, cold 1, then 2 evicts 1 -> 0 stays a hit
    static = simulate_residency(
        ids, num_experts=3, hot_per_layer=2, policy="static", initial_order=np.array([[3.0, 2.0, 1.0]])
    )
    assert static.misses == 1  # only expert 2 is outside the fixed set {0, 1}


def test_decayed_counts_keep_the_frequent_expert_through_one_off_scans():
    # Expert 0 is used often; pairs of one-off experts arrive between its uses.
    # LRU evicts 0 on every scan; decayed counts keep it resident.
    seq = [0] * 4 + [1, 2, 0, 3, 4, 0, 5, 6, 0]
    ids = np.array(seq).reshape(-1, 1, 1)
    lru = simulate_residency(ids, num_experts=7, hot_per_layer=2, policy="lru")
    decayed = simulate_residency(ids, num_experts=7, hot_per_layer=2, policy="decayed", half_life=64)
    assert decayed.misses == 7  # the first 0 and the six one-offs
    assert lru.misses == 10  # plus three re-fetches of 0


def test_reap_ordered_warm_start_beats_a_cold_start():
    ids, _ = _routes(300, 3, 4, 32, seed=1)
    order = -np.tile(np.arange(32, dtype=float), (3, 1))  # expert 0 most salient
    cold = simulate_residency(ids, num_experts=32, hot_per_layer=12, policy="decayed")
    warm = simulate_residency(
        ids, num_experts=32, hot_per_layer=12, policy="decayed", initial_order=order
    )
    assert warm.misses < cold.misses


def test_fetch_threshold_drops_light_misses_and_reports_their_mass():
    ids, gates = _routes(200, 2, 4, 32, seed=2)
    order = -np.tile(np.arange(32, dtype=float), (2, 1))
    exact = simulate_residency(
        ids, num_experts=32, hot_per_layer=8, policy="static", initial_order=order, gate_weights=gates
    )
    hybrid = simulate_residency(
        ids,
        num_experts=32,
        hot_per_layer=8,
        policy="static",
        initial_order=order,
        gate_weights=gates,
        fetch_threshold=0.2,
    )
    assert hybrid.misses + hybrid.dropped == exact.misses
    assert hybrid.misses < exact.misses
    assert 0.0 < hybrid.dropped_gate_mass_fraction < 1.0
    assert exact.dropped == 0 and exact.dropped_gate_mass_fraction == 0.0


def test_warmup_tokens_are_replayed_but_not_counted():
    ids, _ = _routes(100, 1, 2, 16, seed=3)
    result = simulate_residency(ids, num_experts=16, hot_per_layer=4, policy="lru", warmup_tokens=40)
    assert result.tokens == 60 and result.routed_slots == 120


def test_sessions_reset_and_pool_by_token_count():
    a, ga = _routes(80, 2, 3, 16, seed=4)
    b, gb = _routes(40, 2, 3, 16, seed=5)
    pooled = simulate_sessions([(a, ga), (b, gb)], num_experts=16, hot_per_layer=6, policy="lru", expert_bytes=1e6)
    one = simulate_residency(a, num_experts=16, hot_per_layer=6, policy="lru")
    two = simulate_residency(b, num_experts=16, hot_per_layer=6, policy="lru")
    assert pooled.misses == one.misses + two.misses
    assert pooled.tokens == 120
    assert pooled.ssd_bytes_per_token == pytest.approx(pooled.misses_per_token * 1e6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hot_per_layer": 0},
        {"hot_per_layer": 2},  # fewer than top-k
        {"policy": "fifo"},
        {"policy": "static"},  # needs an order
        {"fetch_threshold": 0.1},  # needs gates
    ],
)
def test_invalid_replays_are_refused(kwargs):
    ids, _ = _routes(10, 1, 3, 8)
    params = {"num_experts": 8, "hot_per_layer": 4, "policy": "lru", **kwargs}
    with pytest.raises(ValueError):
        simulate_residency(ids, **params)


def test_report_applies_the_kill_rule_per_hot_size():
    ids, _ = _routes(200, 2, 4, 32, seed=6)
    small = simulate_residency(ids, num_experts=32, hot_per_layer=6, policy="lru")
    full = simulate_residency(ids, num_experts=32, hot_per_layer=32, policy="lru")
    report = residency_report([small, full], kill_miss_rate=0.05)
    assert report["by_hot_per_layer"]["32"]["verdict"] == "PASS"
    assert report["by_hot_per_layer"]["6"]["verdict"] == "KILL"


def test_decode_rate_estimate():
    # 30 ms resident, 15 misses x 9 MB at 5 GB/s = 27 ms more
    rate = decode_tokens_per_second(
        resident_ms_per_token=30.0, misses_per_token=15, expert_bytes=9e6, ssd_bytes_per_second=5e9
    )
    assert rate == pytest.approx(1e3 / 57.0)


def test_route_recorder_keeps_the_first_tokens_in_order():
    recorder = RouteRecorder(layers=[3, 7], max_tokens=5)
    for layer in (3, 7):
        recorder.add_chunk(layer=layer, expert_ids=np.arange(6).reshape(3, 2), gate_weights=np.full((3, 2), 0.5))
        recorder.add_chunk(layer=layer, expert_ids=np.arange(6, 12).reshape(3, 2), gate_weights=np.full((3, 2), 0.25))
    arrays = recorder.as_arrays()
    ids = arrays["routes__expert_ids"]
    assert ids.shape == (5, 2, 2) and ids.dtype == np.int16
    np.testing.assert_array_equal(ids[:, 0].ravel(), np.arange(10))
    assert arrays["routes__gate_weights"][4, 1, 0] == np.float16(0.25)


def test_replay_cli_on_recorded_calibration_sessions(tmp_path, capsys):
    from benchmarks import replay_dsv4_expert_residency as cli
    from keep.quality.reap import ReapStats

    sessions = tmp_path / "cal" / "sessions"
    sessions.mkdir(parents=True)
    counts = np.zeros((2, 16), dtype=np.int64)
    for index in range(2):
        ids, gates = _routes(60, 2, 2, 16, seed=10 + index)
        np.savez(
            sessions / f"s{index}.npz",
            routes__layers=np.array([0, 1]),
            routes__expert_ids=ids.astype(np.int16),
            routes__gate_weights=gates.astype(np.float16),
        )
        for layer in range(2):
            counts[layer] += np.bincount(ids[:, layer].ravel(), minlength=16)
    stats = ReapStats(
        layers=[0, 1],
        num_experts=16,
        route_count=counts,
        gate_sum=counts * 0.5,
        gate_norm_sum=counts * 1.0,
        norm_sum=counts * 2.0,
        norm_max=np.ones((2, 16)),
        abs_max=np.ones((2, 16)),
        routed_slots=counts.sum(axis=1),
    )
    stats.save(tmp_path / "reap-stats.npz")
    out = tmp_path / "residency.json"
    assert (
        cli.main(
            [
                "--calibration-dir", str(tmp_path / "cal"),
                "--reap-stats", str(tmp_path / "reap-stats.npz"),
                "--hot", "4,16",
                "--fetch-threshold", "0.3",
                "--resident-ms", "30",
                "--output", str(out),
            ]
        )
        == 0
    )
    report = json.loads(out.read_text())
    assert report["sessions"] == 2
    assert report["by_hot_per_layer"]["16"]["verdict"] == "PASS"
    assert {r["policy"] for r in report["results"]} == {"static", "lru", "decayed"}
    assert any(key.endswith("+drop<0.3") for key in report["tokens_per_second_estimate"])
