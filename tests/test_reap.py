"""REAP saliency from the real routed forward, survivor choice, and the prune plan."""

from __future__ import annotations

import numpy as np
import pytest

from keep.quality.reap import (
    ReapStats,
    expert_saliency,
    prune_checkpoint_plan,
    pruned_set_jaccard,
    select_survivors,
    super_expert_candidates,
)


def _stats(route_count, gate_norm_sum, *, abs_max=None, gate_sum=None, norm_sum=None):
    route_count = np.asarray(route_count, dtype=np.int64)
    gate_norm_sum = np.asarray(gate_norm_sum, dtype=np.float64)
    shape = route_count.shape
    return ReapStats(
        layers=np.arange(shape[0]),
        num_experts=shape[1],
        route_count=route_count,
        gate_sum=np.zeros(shape) if gate_sum is None else gate_sum,
        gate_norm_sum=gate_norm_sum,
        norm_sum=np.zeros(shape) if norm_sum is None else norm_sum,
        norm_max=np.zeros(shape),
        abs_max=np.zeros(shape) if abs_max is None else abs_max,
        routed_slots=route_count.sum(axis=1),
    )


# -- saliency -----------------------------------------------------------------


def test_reap_is_the_mean_gate_weighted_norm_over_routed_slots():
    stats = _stats([[4, 1, 0]], [[2.0, 3.0, 0.0]])
    np.testing.assert_allclose(expert_saliency(stats, "reap"), [[0.5, 3.0, 0.0]])
    # A rarely routed expert with large outputs outranks a frequent quiet one,
    # which is exactly where REAP and frequency pruning disagree.
    np.testing.assert_array_equal(expert_saliency(stats, "frequency"), [[4, 1, 0]])


def test_never_routed_experts_score_zero_under_every_criterion():
    stats = _stats([[3, 0]], [[1.5, 0.0]], gate_sum=[[0.9, 0.0]], norm_sum=[[6.0, 0.0]])
    for criterion in ("reap", "frequency", "weighted_frequency", "ean_mean"):
        assert expert_saliency(stats, criterion)[0, 1] == 0.0


def test_prior_shrinks_rare_experts_toward_the_layer_mean():
    stats = _stats([[1000, 1]], [[1000.0, 50.0]])  # layer mean ~1.05
    raw = expert_saliency(stats, "reap")
    shrunk = expert_saliency(stats, "reap", prior_slots=100.0)
    assert raw[0, 1] == pytest.approx(50.0)
    assert 1.0 < shrunk[0, 1] < 2.0  # one noisy slot no longer dominates
    assert shrunk[0, 0] == pytest.approx(raw[0, 0], rel=0.01)


def test_unknown_criterion_and_negative_prior_are_refused():
    stats = _stats([[1]], [[1.0]])
    with pytest.raises(ValueError):
        expert_saliency(stats, "softmax")
    with pytest.raises(ValueError):
        expert_saliency(stats, "reap", prior_slots=-1)


# -- survivors and comparisons ---------------------------------------------------


def test_survivors_are_the_most_salient_in_ascending_id_order():
    saliency = np.array([[0.1, 0.9, 0.5, 0.7], [1.0, 0.0, 0.0, 0.2]])
    np.testing.assert_array_equal(select_survivors(saliency, 2), [[1, 3], [0, 3]])


def test_ties_break_toward_the_lower_expert_id():
    np.testing.assert_array_equal(select_survivors(np.array([[0.5, 0.5, 0.5]]), 2), [[0, 1]])


@pytest.mark.parametrize("keep", [0, 5])
def test_keep_must_fit_the_expert_count(keep):
    with pytest.raises(ValueError):
        select_survivors(np.zeros((1, 4)), keep)


def test_pruned_set_jaccard():
    a = np.array([[0, 1], [0, 1]])
    b = np.array([[0, 1], [2, 3]])
    np.testing.assert_allclose(pruned_set_jaccard(a, b, num_experts=4), [1.0, 0.0])


def test_super_expert_screen_sees_negative_outliers_through_abs_max():
    # Enough experts that the 99.5th percentile describes the bulk, as it does
    # at 40 x 384 (it would be the outliers themselves at 4 x 8).
    abs_max = np.full((8, 64), 1.0)
    abs_max[1, 5] = 900.0  # e.g. a large negative output, invisible to a signed max
    abs_max[7, 2] = 950.0  # last quarter of layers is excluded, as upstream does
    stats = _stats(np.ones((8, 64), dtype=np.int64), np.ones((8, 64)), abs_max=abs_max)
    assert super_expert_candidates(stats) == [(1, 5)]


# -- stats container ----------------------------------------------------------------


def test_stats_validate_their_invariants():
    with pytest.raises(ValueError, match="sum to the routed slots"):
        ReapStats(
            layers=[0],
            num_experts=2,
            route_count=[[1, 1]],
            gate_sum=np.zeros((1, 2)),
            gate_norm_sum=np.zeros((1, 2)),
            norm_sum=np.zeros((1, 2)),
            norm_max=np.zeros((1, 2)),
            abs_max=np.zeros((1, 2)),
            routed_slots=[3],
        )
    with pytest.raises(ValueError, match="no routed slots"):
        _stats([[0, 2]], [[1.0, 1.0]])
    with pytest.raises(ValueError, match="non-negative"):
        _stats([[1, 1]], [[-1.0, 1.0]])


def test_stats_round_trip_and_merge(tmp_path):
    a = _stats([[2, 1]], [[1.0, 2.0]], abs_max=np.array([[3.0, 1.0]]))
    a.sessions = ("s0",)
    b = _stats([[1, 3]], [[0.5, 1.0]], abs_max=np.array([[1.0, 4.0]]))
    b.sessions = ("s1",)
    merged = a.merged(b)
    np.testing.assert_array_equal(merged.route_count, [[3, 4]])
    np.testing.assert_allclose(merged.gate_norm_sum, [[1.5, 3.0]])
    np.testing.assert_allclose(merged.abs_max, [[3.0, 4.0]])
    assert merged.sessions == ("s0", "s1")
    with pytest.raises(ValueError, match="double-count"):
        merged.merged(a)

    path = tmp_path / "reap-stats.npz"
    merged.save(path)
    loaded = ReapStats.load(path)
    assert loaded.sessions == ("s0", "s1")
    np.testing.assert_array_equal(loaded.route_count, merged.route_count)
    np.testing.assert_allclose(loaded.gate_norm_sum, merged.gate_norm_sum)


# -- device accumulator ---------------------------------------------------------------


@pytest.mark.mlx
def test_accumulator_matches_a_float64_reference():
    mx = pytest.importorskip("mlx.core")
    from keep.quality.reap import ReapAccumulator

    rng = np.random.default_rng(3)
    experts, top_k, hidden = 6, 3, 16
    acc = ReapAccumulator(layers=[4, 9], num_experts=experts)
    ref = {name: np.zeros((2, experts)) for name in ("count", "g", "gn", "n", "nmax", "amax")}
    for layer_row, layer in enumerate((4, 9)):
        for rows in (5, 11):  # two chunks per layer
            ids = rng.integers(0, experts, size=(rows, top_k))
            gates = rng.random((rows, top_k)).astype(np.float32)
            outs = rng.normal(size=(rows, top_k, hidden)).astype(np.float32) * 3
            acc.add_chunk(
                layer=layer,
                expert_ids=mx.array(ids),
                gate_weights=mx.array(gates),
                expert_outputs=mx.array(outs),
            )
            acc.eval_pending()
            norms = np.linalg.norm(outs.astype(np.float64), axis=-1)
            peaks = np.abs(outs).max(axis=-1)
            for e, g, n, p in zip(ids.ravel(), gates.ravel(), norms.ravel(), peaks.ravel()):
                ref["count"][layer_row, e] += 1
                ref["g"][layer_row, e] += g
                ref["gn"][layer_row, e] += g * n
                ref["n"][layer_row, e] += n
                ref["nmax"][layer_row, e] = max(ref["nmax"][layer_row, e], n)
                ref["amax"][layer_row, e] = max(ref["amax"][layer_row, e], p)
        acc.flush_device()
    stats = acc.stats(session="sess")
    np.testing.assert_array_equal(stats.route_count, ref["count"])
    np.testing.assert_allclose(stats.gate_sum, ref["g"], rtol=2e-6)
    np.testing.assert_allclose(stats.gate_norm_sum, ref["gn"], rtol=2e-6)
    np.testing.assert_allclose(stats.norm_sum, ref["n"], rtol=2e-6)
    np.testing.assert_allclose(stats.norm_max, ref["nmax"], rtol=2e-6)
    np.testing.assert_allclose(stats.abs_max, ref["amax"], rtol=1e-7)
    assert stats.sessions == ("sess",)
    assert stats.routed_slots.tolist() == [48, 48]


@pytest.mark.mlx
def test_accumulator_refuses_mismatched_shapes_and_untracked_layers():
    mx = pytest.importorskip("mlx.core")
    from keep.quality.reap import ReapAccumulator

    acc = ReapAccumulator(layers=[0], num_experts=4)
    ids = mx.zeros((2, 2), dtype=mx.int32)
    with pytest.raises(KeyError):
        acc.add_chunk(layer=1, expert_ids=ids, gate_weights=ids, expert_outputs=mx.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="do not match"):
        acc.add_chunk(layer=0, expert_ids=ids, gate_weights=ids, expert_outputs=mx.zeros((2, 3, 3)))


# -- slicing the router is the same as masking it ---------------------------------------


@pytest.mark.mlx
def test_sliced_gate_equals_the_full_gate_restricted_to_survivors():
    """The prune applies by slicing router rows; check it against masking.

    Masking pruned experts out of *selection* (correction bias -> -inf) is
    what "drop the expert and re-select top-k among survivors" means. The
    sliced gate must pick the same experts with the same renormalised
    weights, mapped back through the survivor table.
    """

    mx = pytest.importorskip("mlx.core")
    from ramp.models.deepseek_v4_flash_adapter import DeepseekV4FlashVQModelArgs, MoEGate

    rng = np.random.default_rng(11)
    experts, keep, hidden = 12, 7, 32
    args = DeepseekV4FlashVQModelArgs(
        hidden_size=hidden,
        n_routed_experts=experts,
        num_experts_per_tok=3,
        num_hash_layers=0,
        num_hidden_layers=1,
        compress_ratios=[0],
    )
    weight = rng.normal(size=(experts, hidden)).astype(np.float32)
    bias = rng.normal(size=(experts,)).astype(np.float32) * 0.1
    survivors = np.sort(rng.choice(experts, size=keep, replace=False))

    full = MoEGate(args, 0)
    full.weight = mx.array(weight)
    masked_bias = bias.copy()
    masked_bias[np.setdiff1d(np.arange(experts), survivors)] = -1e30
    full.e_score_correction_bias = mx.array(masked_bias)

    sliced_args = DeepseekV4FlashVQModelArgs(
        hidden_size=hidden,
        n_routed_experts=keep,
        num_experts_per_tok=3,
        num_hash_layers=0,
        num_hidden_layers=1,
        compress_ratios=[0],
    )
    sliced = MoEGate(sliced_args, 0)
    sliced.weight = mx.array(weight[survivors])
    sliced.e_score_correction_bias = mx.array(bias[survivors])

    x = mx.array(rng.normal(size=(1, 64, hidden)).astype(np.float32))
    full_ids, full_scores = full(x)
    sliced_ids, sliced_scores = sliced(x)
    mapped = survivors[np.array(sliced_ids)]
    np.testing.assert_array_equal(np.sort(mapped, axis=-1), np.sort(np.array(full_ids), axis=-1))
    order_full = np.argsort(np.array(full_ids), axis=-1)
    order_sliced = np.argsort(mapped, axis=-1)
    np.testing.assert_allclose(
        np.take_along_axis(np.array(sliced_scores), order_sliced, axis=-1),
        np.take_along_axis(np.array(full_scores), order_full, axis=-1),
        rtol=1e-6,
    )


# -- checkpoint prune plan ------------------------------------------------------------


def _names(layers=2, experts=4, *, hash_layer=None, with_bias_vl=True):
    names = ["embed.weight", "head.weight", "mtp.0.ffn.gate.weight", "mtp.0.ffn.experts.0.w1.weight"]
    for layer in range(layers):
        p = f"layers.{layer}"
        names += [f"{p}.attn.wq_a.weight", f"{p}.ffn.gate.weight", f"{p}.ffn.shared_experts.w1.weight"]
        if layer == hash_layer:
            names.append(f"{p}.ffn.gate.tid2eid")
        else:
            names.append(f"{p}.ffn.gate.bias")
            if with_bias_vl:
                names.append(f"{p}.ffn.gate.bias_vl")
        for e in range(experts):
            for proj in ("w1", "w2", "w3"):
                names += [f"{p}.ffn.experts.{e}.{proj}.weight", f"{p}.ffn.experts.{e}.{proj}.scale"]
    return names


def test_prune_plan_renumbers_slices_and_drops():
    plan = prune_checkpoint_plan(_names(), {0: [1, 3], 1: [0, 2]})
    assert plan["keep"] == 2
    assert plan["rename"]["layers.0.ffn.experts.3.w2.scale"] == "layers.0.ffn.experts.1.w2.scale"
    assert plan["rename"]["layers.1.ffn.experts.0.w1.weight"] == "layers.1.ffn.experts.0.w1.weight"
    assert "layers.0.ffn.experts.0.w1.weight" in plan["drop"]
    assert len(plan["rename"]) == 2 * 2 * 3 * 2
    assert len(plan["drop"]) == 2 * 2 * 3 * 2
    # bias_vl is sliced with the gate, never left at the old width.
    for leaf in ("weight", "bias", "bias_vl"):
        assert plan["slice_rows"][f"layers.0.ffn.gate.{leaf}"] == [1, 3]
    # Attention, shared experts, embeddings and the drafter are untouched.
    for name in ("layers.0.attn.wq_a.weight", "layers.1.ffn.shared_experts.w1.weight",
                 "embed.weight", "mtp.0.ffn.gate.weight", "mtp.0.ffn.experts.0.w1.weight"):
        assert name in plan["untouched"]


def test_prune_plan_refuses_hash_layers_and_ragged_or_phantom_survivors():
    with pytest.raises(NotImplementedError, match="hash-routed"):
        prune_checkpoint_plan(_names(hash_layer=0), {0: [0, 1], 1: [0, 1]})
    with pytest.raises(ValueError, match="same expert count"):
        prune_checkpoint_plan(_names(), {0: [0, 1], 1: [0, 1, 2]})
    with pytest.raises(ValueError, match="no tensors"):
        prune_checkpoint_plan(_names(experts=4), {0: [0, 9], 1: [0, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        prune_checkpoint_plan(_names(), {0: [1, 1], 1: [0, 1]})


def test_reap_select_command_reports_the_kill_test_and_writes_survivors(tmp_path, capsys):
    import json

    from benchmarks import produce_dsv4_teacher_cache as cli

    rng = np.random.default_rng(2)
    counts = rng.integers(0, 50, size=(3, 16))
    stats = _stats(counts, counts * rng.random((3, 16)) * 2)
    stats.sessions = ("a", "b")
    stats_path = tmp_path / "reap-stats.npz"
    stats.save(stats_path)
    out = tmp_path / "survivors.json"

    assert cli.main(["reap-select", "--stats", str(stats_path), "--keep", "8", "--output", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["keep"] == 8 and report["compare"] == "frequency"
    assert set(report["pruned_set_jaccard_vs_compare"]) == {"0", "1", "2"}
    saved = json.loads(out.read_text())
    assert saved["record_type"] == "keep_reap_survivors_v1"
    assert all(len(ids) == 8 and ids == sorted(ids) for ids in saved["survivors"].values())
