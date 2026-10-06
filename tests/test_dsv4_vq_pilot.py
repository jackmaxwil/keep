"""Headless tests for the Wave 5 DeepSeek-V4-Flash VQ materialization pilot.

Everything here runs without the 163 GB checkpoint, without the 20 GB
calibration artifact, and without a GPU: the rate model, the error metrics, the
route weighting, the expert stratification, the schedule arithmetic, and the
calibration reader (against a synthetic capture written in the real schema).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mlx_vq.convert.dsv4_vq_pilot import (
    PROJECTION_INPUT_SPACE,
    RATE_LADDER,
    GroupSizePolicy,
    LayerFitTiming,
    aggregate_calibration_importance,
    artifact_bytes,
    estimate_source_fp4_step,
    expert_weight_counts,
    extrapolate_schedule,
    load_profile_geometry,
    policy_bpw,
    projection_error,
    projection_importance,
    rate_bpw,
    reconstruct_quantized,
    route_weighted_mean,
    stratified_experts,
    write_json,
)

# --------------------------------------------------------------------------
# rate model
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code_bits", "group_size", "expected"),
    [
        (8, 512, 1.03125),
        (16, 512, 2.03125),
        (16, 256, 2.0625),
        (16, 128, 2.125),
        (16, 64, 2.25),
        (16, 32, 2.5),
        (16, 16, 3.0),
    ],
)
def test_rate_bpw_matches_stream_convert_storage_model(code_bits, group_size, expected):
    assert rate_bpw(code_bits, group_size) == pytest.approx(expected)


def test_rate_bpw_agrees_with_estimate_vq_storage():
    """The pilot's rate model must be the shipped one, not a parallel guess."""

    from mlx_vq.convert.stream_convert import estimate_vq_storage

    weights = 2048 * 4096
    for code_bits, group_size in RATE_LADDER:
        estimate = estimate_vq_storage(
            weights, code_bits=code_bits, group_size=group_size
        )
        stored = (estimate.code_bytes + estimate.scale_bytes) * 8 / weights
        assert stored == pytest.approx(rate_bpw(code_bits, group_size))


@pytest.mark.parametrize(
    ("code_bits", "group_size"),
    [(4, 512), (16, 0), (16, 12), (32, 512)],
)
def test_rate_bpw_rejects_unsupported_points(code_bits, group_size):
    with pytest.raises(ValueError):
        rate_bpw(code_bits, group_size)


def test_policy_bpw_is_the_weighted_mean_over_projections():
    policy = GroupSizePolicy(gate=(16, 512), up=(16, 512), down=(16, 32))
    expected = (2.03125 + 2.03125 + 2.5) / 3
    assert policy_bpw(policy) == pytest.approx(expected)


def test_policy_bpw_weights_by_actual_weight_counts():
    """All three V4-Flash projections hold the same count; assert it explicitly."""

    counts = expert_weight_counts()
    assert set(counts.values()) == {2048 * 4096}


def test_policy_profile_block_is_yaml_shaped():
    policy = GroupSizePolicy(gate=(16, 512), up=(16, 512), down=(16, 128))
    block = policy.as_profile_block()
    assert block["group_size_policy"] == {"gate": 512, "up": 512, "down": 128}
    assert block["code_bits_policy"] == {"gate": 16, "up": 16, "down": 16}


def test_policy_rejects_unknown_projection():
    policy = GroupSizePolicy(gate=(16, 512), up=(16, 512), down=(16, 512))
    with pytest.raises(ValueError):
        policy.for_projection("shared")


# --------------------------------------------------------------------------
# error metrics
# --------------------------------------------------------------------------


def test_projection_error_is_zero_for_an_exact_reconstruction():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((16, 32)).astype(np.float32)
    importance = rng.random(32).astype(np.float32) + 0.1
    metrics = projection_error(weight, weight, importance)
    assert metrics.weighted_relative_mse == pytest.approx(0.0)
    assert metrics.expected_cosine == pytest.approx(1.0)
    assert metrics.max_abs_error == pytest.approx(0.0)


def test_weighted_relative_mse_equals_the_expected_output_mse_ratio():
    """The proxy's whole claim: it is E||dW x||^2 / E||W x||^2 for diagonal cov.

    Sampled directly rather than asserted algebraically -- if the closed form
    and the Monte Carlo disagree, the proxy's headline sentence is wrong.
    """

    rng = np.random.default_rng(7)
    out_dim, in_dim = 24, 48
    weight = rng.standard_normal((out_dim, in_dim)).astype(np.float64)
    perturbed = weight + 0.05 * rng.standard_normal((out_dim, in_dim))
    sigma = rng.random(in_dim) + 0.25  # per-column standard deviation

    metrics = projection_error(weight, perturbed, sigma**2)

    samples = rng.standard_normal((200_000, in_dim)) * sigma
    reference = samples @ weight.T
    candidate = samples @ perturbed.T
    empirical = float(
        np.sum((reference - candidate) ** 2) / np.sum(reference**2)
    )
    assert metrics.weighted_relative_mse == pytest.approx(empirical, rel=0.02)

    empirical_cosine = float(
        np.sum(reference * candidate)
        / np.sqrt(np.sum(reference**2) * np.sum(candidate**2))
    )
    assert metrics.expected_cosine == pytest.approx(empirical_cosine, rel=1e-3)


def test_projection_error_rejects_a_mismatched_importance_length():
    weight = np.ones((4, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="importance must have shape"):
        projection_error(weight, weight, np.ones(4, dtype=np.float32))


def test_projection_error_rejects_negative_importance():
    weight = np.ones((4, 8), dtype=np.float32)
    bad = np.ones(8, dtype=np.float32)
    bad[2] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        projection_error(weight, weight, bad)


def test_projection_error_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must match"):
        projection_error(
            np.ones((4, 8), dtype=np.float32),
            np.ones((4, 16), dtype=np.float32),
            np.ones(8, dtype=np.float32),
        )


def test_route_weighted_mean_ignores_never_routed_experts():
    assert route_weighted_mean([0.1, 99.0], [1000, 0]) == pytest.approx(0.1)


def test_route_weighted_mean_rejects_an_empty_route_total():
    with pytest.raises(ValueError, match="nothing was routed"):
        route_weighted_mean([0.1, 0.2], [0, 0])


def test_route_weighted_mean_rejects_ragged_input():
    with pytest.raises(ValueError, match="same length"):
        route_weighted_mean([0.1, 0.2], [1])


# --------------------------------------------------------------------------
# reconstruction round-trip
# --------------------------------------------------------------------------


def test_reconstruct_quantized_round_trips_the_shipped_quantizer():
    """Decode must invert the packing the GLM lineage's quantizer produces."""

    from mlx_vq.convert.glm52_recovery_materialize import (
        quantize_weight_importance_aware,
    )

    rng = np.random.default_rng(11)
    weight = rng.standard_normal((32, 128)).astype(np.float32)
    importance = (rng.random(128) + 0.1).astype(np.float32)
    quantized = quantize_weight_importance_aware(
        weight, importance, group_size=64, code_bits=16, iterations=2
    )
    decoded = reconstruct_quantized(quantized)
    assert decoded.shape == weight.shape
    # A 2 bpw fit of Gaussian data is lossy but must stay far inside "random".
    metrics = projection_error(weight, decoded, importance)
    assert 0.0 < metrics.weighted_relative_mse < 0.5


def test_reconstruct_quantized_handles_the_eight_bit_codebook():
    from mlx_vq.convert.glm52_recovery_materialize import (
        quantize_weight_importance_aware,
    )

    rng = np.random.default_rng(12)
    weight = rng.standard_normal((16, 64)).astype(np.float32)
    importance = np.ones(64, dtype=np.float32)
    quantized = quantize_weight_importance_aware(
        weight, importance, group_size=32, code_bits=8, iterations=2
    )
    assert reconstruct_quantized(quantized).shape == weight.shape


# --------------------------------------------------------------------------
# FP4 source reference
# --------------------------------------------------------------------------


def test_fp4_grid_values_round_trip_exactly():
    """The released weight is already on the grid, so re-encoding is a no-op.

    This is the assertion behind the report's claim that there is no separate
    "bf16 reference" hiding on this machine.
    """

    from keep.convert.fp4_expert import decode_fp4_e2m1

    rng = np.random.default_rng(3)
    codes = rng.integers(0, 256, size=(8, 32), dtype=np.uint8)
    dense = decode_fp4_e2m1(codes)
    estimate = estimate_source_fp4_step(dense, group_size=32)
    assert estimate["roundtrip_relative_mse"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < estimate["synthetic_gaussian_relative_mse"] < 0.2


def test_fp4_values_survive_a_bfloat16_cast():
    """e2m1 has one mantissa bit; bf16 has eight. No information is lost."""

    import mlx.core as mx

    from keep.convert.fp4_expert import decode_fp4_e2m1

    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    dense = decode_fp4_e2m1(codes)
    scaled = dense * np.float32(2.0**-7)
    back = np.asarray(
        mx.array(scaled).astype(mx.bfloat16).astype(mx.float32), dtype=np.float32
    )
    assert np.array_equal(back, scaled)


def test_estimate_source_fp4_step_rejects_a_ragged_group():
    with pytest.raises(ValueError, match="multiple of group_size"):
        estimate_source_fp4_step(np.ones((4, 30), dtype=np.float32), group_size=32)


# --------------------------------------------------------------------------
# block proxy
# --------------------------------------------------------------------------


def test_limited_swiglu_matches_the_adapter_implementation():
    """The proxy must apply the release's own clamp, not a plain SwiGLU."""

    import mlx.core as mx

    from mlx_vq.convert.dsv4_vq_pilot import limited_swiglu
    from ramp.models.deepseek_v4_flash_adapter import _limited_swiglu

    rng = np.random.default_rng(21)
    gate = (rng.standard_normal((7, 11)) * 8.0).astype(np.float32)
    up = (rng.standard_normal((7, 11)) * 8.0).astype(np.float32)
    expected = np.asarray(
        _limited_swiglu(mx.array(gate), mx.array(up), 10.0), dtype=np.float64
    )
    np.testing.assert_allclose(limited_swiglu(gate, up, 10.0), expected, rtol=1e-5)


def test_limited_swiglu_clamps_gate_above_and_up_on_both_sides():
    from mlx_vq.convert.dsv4_vq_pilot import limited_swiglu

    # The clamp is deliberately asymmetric on the gate: min(gate, limit) only.
    got = limited_swiglu(np.array([[50.0, -50.0]]), np.array([[50.0, -50.0]]), 10.0)
    # gate 50 -> 10, up 50 -> 10, so silu(10) * 10.
    assert got[0, 0] == pytest.approx(10.0 * 10.0 / (1 + np.exp(-10.0)), rel=1e-9)
    # gate -50 is NOT clamped from below; silu(-50) ~ 0 kills the term.
    assert got[0, 1] == pytest.approx(0.0, abs=1e-15)

    # The up branch is clamped on both sides, so a huge |up| cannot run away.
    bounded = limited_swiglu(np.array([[2.0]]), np.array([[1e6]]), 10.0)
    assert bounded[0, 0] == pytest.approx(10.0 * 2.0 / (1 + np.exp(-2.0)), rel=1e-9)


def test_expert_block_proxy_is_zero_for_an_identical_reconstruction():
    from mlx_vq.convert.dsv4_vq_pilot import expert_block_proxy

    rng = np.random.default_rng(31)
    weights = {
        "gate": rng.standard_normal((12, 20)).astype(np.float32) * 0.05,
        "up": rng.standard_normal((12, 20)).astype(np.float32) * 0.05,
        "down": rng.standard_normal((20, 12)).astype(np.float32) * 0.05,
    }
    probe = expert_block_proxy(
        weights, weights, column_sigma=np.ones(20), tokens=64
    )
    assert probe["block_relative_mse"] == pytest.approx(0.0, abs=1e-24)
    assert probe["block_cosine"] == pytest.approx(1.0)


def test_expert_block_proxy_grows_with_the_perturbation():
    from mlx_vq.convert.dsv4_vq_pilot import expert_block_proxy

    rng = np.random.default_rng(32)
    weights = {
        "gate": rng.standard_normal((12, 20)).astype(np.float32) * 0.05,
        "up": rng.standard_normal((12, 20)).astype(np.float32) * 0.05,
        "down": rng.standard_normal((20, 12)).astype(np.float32) * 0.05,
    }
    previous = 0.0
    for scale in (0.001, 0.01, 0.05):
        noisy = {
            name: value + scale * rng.standard_normal(value.shape).astype(np.float32)
            for name, value in weights.items()
        }
        current = expert_block_proxy(
            weights, noisy, column_sigma=np.ones(20), tokens=128
        )["block_relative_mse"]
        assert current > previous
        previous = current


def test_expert_block_proxy_rejects_a_missing_projection():
    from mlx_vq.convert.dsv4_vq_pilot import expert_block_proxy

    weights = {
        "gate": np.ones((2, 4), dtype=np.float32),
        "up": np.ones((2, 4), dtype=np.float32),
        "down": np.ones((4, 2), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="missing"):
        expert_block_proxy(
            weights, {"gate": weights["gate"]}, column_sigma=np.ones(4), tokens=8
        )


def test_expert_block_proxy_rejects_a_negative_sigma():
    from mlx_vq.convert.dsv4_vq_pilot import expert_block_proxy

    weights = {
        "gate": np.ones((2, 4), dtype=np.float32),
        "up": np.ones((2, 4), dtype=np.float32),
        "down": np.ones((4, 2), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="non-negative"):
        expert_block_proxy(
            weights, weights, column_sigma=np.array([1.0, -1.0, 1.0, 1.0]), tokens=8
        )


# --------------------------------------------------------------------------
# expert sampling
# --------------------------------------------------------------------------


def test_stratified_experts_spans_the_route_count_range():
    counts = np.array([5, 900, 1, 40, 700, 2, 33, 1200], dtype=np.int64)
    picked = stratified_experts(counts, 4)
    assert len(picked) == 4
    assert int(np.argmin(counts)) in picked
    assert int(np.argmax(counts)) in picked


def test_stratified_experts_is_deterministic():
    counts = np.arange(64)[::-1].copy()
    assert stratified_experts(counts, 8) == stratified_experts(counts, 8)


def test_stratified_experts_returns_everything_when_asked_for_everything():
    counts = np.array([3, 1, 2], dtype=np.int64)
    assert stratified_experts(counts, 3) == [0, 1, 2]


@pytest.mark.parametrize("count", [0, -1, 9])
def test_stratified_experts_rejects_an_impossible_count(count):
    with pytest.raises(ValueError):
        stratified_experts(np.arange(8), count)


# --------------------------------------------------------------------------
# schedule arithmetic
# --------------------------------------------------------------------------


def _timing(**overrides) -> LayerFitTiming:
    base = {
        "layer": 20,
        "projections_fitted": 48,
        "read_seconds": 4.0,
        "decode_seconds": 1.0,
        "fit_seconds": 6.0,
        "peak_rss_gb": 5.0,
    }
    base.update(overrides)
    return LayerFitTiming(**base)


def test_scaled_to_full_layer_leaves_the_whole_layer_read_alone():
    """The read is one coalesced span whatever the expert sample size."""

    scaled = _timing().scaled_to_full_layer(768)
    assert scaled.read_seconds == pytest.approx(4.0)
    assert scaled.decode_seconds == pytest.approx(16.0)
    assert scaled.fit_seconds == pytest.approx(96.0)
    assert scaled.total_seconds == pytest.approx(116.0)


def test_scaled_to_full_layer_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="projections_fitted must be positive"):
        _timing(projections_fitted=0).scaled_to_full_layer(768)


def test_extrapolate_schedule_multiplies_the_mean_layer_by_the_layer_count():
    estimate = extrapolate_schedule(
        [_timing(fit_seconds=6.0), _timing(layer=1, fit_seconds=10.0)],
        label="probe",
        bpw=2.5,
        layers=46,
        projections_per_layer=768,
        routed_weight_count=1_000_000_000,
        resident_weight_count=100_000_000,
        resident_bpw=8.0,
    )
    # mean fit 8 s over 48 projections -> 128 s over 768, + 16 s decode + 4 s read
    assert estimate.seconds_per_layer == pytest.approx(4.0 + 16.0 + 128.0)
    assert estimate.total_seconds == pytest.approx(46 * 148.0)
    assert estimate.routed_bytes == pytest.approx(1e9 * 2.5 / 8)
    assert estimate.resident_bytes == pytest.approx(1e8)
    assert estimate.artifact_bytes == pytest.approx(1e9 * 2.5 / 8 + 1e8)


def test_extrapolate_schedule_requires_a_measurement():
    with pytest.raises(ValueError, match="at least one measured layer"):
        extrapolate_schedule(
            [],
            label="empty",
            bpw=2.5,
            layers=43,
            projections_per_layer=768,
            routed_weight_count=1,
            resident_weight_count=0,
            resident_bpw=8.0,
        )


def test_extrapolate_schedule_records_its_assumptions_verbatim():
    estimate = extrapolate_schedule(
        [_timing()],
        label="probe",
        bpw=2.0,
        layers=43,
        projections_per_layer=768,
        routed_weight_count=1000,
        resident_weight_count=0,
        resident_bpw=8.0,
        assumptions=("uniform layer cost",),
    )
    assert estimate.as_dict()["assumptions"] == ["uniform layer cost"]


def test_artifact_bytes_splits_routed_resident_and_codebook():
    sizes = artifact_bytes(
        bpw=2.0,
        routed_weight_count=8,
        resident_weight_count=8,
        resident_bpw=8.0,
        codebook_bytes=1024,
    )
    assert sizes["routed_bytes"] == pytest.approx(2.0)
    assert sizes["resident_bytes"] == pytest.approx(8.0)
    assert sizes["total_bytes"] == pytest.approx(2.0 + 8.0 + 1024)


def test_artifact_bytes_rejects_a_zero_rate():
    with pytest.raises(ValueError, match="bit rates must be positive"):
        artifact_bytes(
            bpw=0.0,
            routed_weight_count=8,
            resident_weight_count=0,
            resident_bpw=8.0,
        )


def test_campaign_envelope_arithmetic_reproduces_the_published_table():
    """The campaign's ~120 GB row is only consistent if MTP blocks are counted.

    43 sparse layers + 3 DSpark drafter blocks x 256 experts x 3 x 2048 x 4096
    at 3.0 bpw plus ~7.83 G resident weights at 8 bpw lands on the campaign
    doc's "~120 GB" figure; 43 layers alone does not. Pinning it here so a
    later change to the block count fails loudly instead of quietly shifting
    the deliverable envelope.
    """

    routed = 46 * 256 * 3 * 2048 * 4096
    sizes = artifact_bytes(
        bpw=3.0,
        routed_weight_count=routed,
        resident_weight_count=7_830_000_000,
        resident_bpw=8.0,
    )
    assert sizes["total_bytes"] / 1e9 == pytest.approx(119.5, abs=1.0)


# --------------------------------------------------------------------------
# calibration reader
# --------------------------------------------------------------------------


def _write_calibration_session(path, *, prompt_id, layers, experts, hidden, down, seed):
    rng = np.random.default_rng(seed)
    arrays = {
        "record_type": np.array("dsv4_teacher_calibration_v1"),
        "schema_version": np.array(1, dtype=np.int32),
        "prompt_id": np.array(prompt_id),
        "layers": np.asarray(layers, dtype=np.int32),
        "num_experts": np.array(experts, dtype=np.int32),
    }
    for space, dim in (("hidden", hidden), ("down", down)):
        arrays[f"importance_sum__{space}"] = rng.random(
            (len(layers), experts, dim)
        ).astype(np.float32)
        arrays[f"affinity_weighted_importance__{space}"] = rng.random(
            (len(layers), experts, dim)
        ).astype(np.float32)
        arrays[f"route_count__{space}"] = rng.integers(
            0, 100, size=(len(layers), experts)
        ).astype(np.int64)
        arrays[f"affinity_score_sum__{space}"] = rng.random(
            (len(layers), experts)
        ).astype(np.float64)
        arrays[f"total_route_count__{space}"] = np.full(len(layers), 600, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return arrays


def test_aggregate_calibration_importance_sums_sessions_for_chosen_layers(tmp_path):
    layers = [0, 1, 2]
    first = _write_calibration_session(
        tmp_path / "sessions" / "a.npz",
        prompt_id="a",
        layers=layers,
        experts=4,
        hidden=16,
        down=8,
        seed=1,
    )
    second = _write_calibration_session(
        tmp_path / "sessions" / "b.npz",
        prompt_id="b",
        layers=layers,
        experts=4,
        hidden=16,
        down=8,
        seed=2,
    )
    stats = aggregate_calibration_importance(tmp_path, layers=[2])
    assert stats["sessions"] == 2
    assert stats["num_experts"] == 4
    assert stats["prompt_ids"] == ("a", "b")
    expected = (
        first["importance_sum__hidden"][2].astype(np.float64)
        + second["importance_sum__hidden"][2].astype(np.float64)
    )
    np.testing.assert_allclose(stats["importance"]["hidden"][0], expected)
    np.testing.assert_array_equal(
        stats["route_count"]["hidden"][0],
        first["route_count__hidden"][2] + second["route_count__hidden"][2],
    )


def test_projection_importance_picks_the_right_input_space(tmp_path):
    _write_calibration_session(
        tmp_path / "sessions" / "a.npz",
        prompt_id="a",
        layers=[0, 5],
        experts=3,
        hidden=16,
        down=8,
        seed=4,
    )
    stats = aggregate_calibration_importance(tmp_path, layers=[5])
    assert projection_importance(stats, layer=5, projection="gate", expert=1).shape == (16,)
    assert projection_importance(stats, layer=5, projection="up", expert=1).shape == (16,)
    assert projection_importance(stats, layer=5, projection="down", expert=1).shape == (8,)
    assert PROJECTION_INPUT_SPACE == {
        "gate": "hidden",
        "up": "hidden",
        "down": "down",
    }


def test_aggregate_calibration_importance_rejects_a_missing_layer(tmp_path):
    _write_calibration_session(
        tmp_path / "sessions" / "a.npz",
        prompt_id="a",
        layers=[0, 1],
        experts=2,
        hidden=8,
        down=4,
        seed=5,
    )
    with pytest.raises(ValueError, match=r"no layers \[9\]"):
        aggregate_calibration_importance(tmp_path, layers=[9])


def test_aggregate_calibration_importance_rejects_an_empty_directory(tmp_path):
    (tmp_path / "sessions").mkdir(parents=True)
    with pytest.raises(ValueError, match="no session outputs"):
        aggregate_calibration_importance(tmp_path, layers=[0])


def test_aggregate_calibration_importance_rejects_a_foreign_record(tmp_path):
    path = tmp_path / "sessions" / "a.npz"
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        record_type=np.array("dsv4_teacher_logits_v1"),
        layers=np.array([0], dtype=np.int32),
        num_experts=np.array(1, dtype=np.int32),
        prompt_id=np.array("a"),
    )
    with pytest.raises(ValueError, match="not a calibration capture"):
        aggregate_calibration_importance(tmp_path, layers=[0])


# --------------------------------------------------------------------------
# profile geometry + artifacts
# --------------------------------------------------------------------------


def test_load_profile_geometry_reads_the_shipped_v4_flash_profile():
    from pathlib import Path

    profile = Path(__file__).resolve().parent.parent / "models" / "deepseek-v4-flash-0731.yaml"
    geometry = load_profile_geometry(profile)
    assert geometry == {
        "num_layers": 43,
        "num_sparse_layers": 43,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "num_experts": 256,
        "experts_per_tok": 6,
    }


def test_load_profile_geometry_rejects_an_incomplete_profile(tmp_path):
    path = tmp_path / "partial.yaml"
    path.write_text("num_layers: 43\nhidden_size: 4096\n")
    with pytest.raises(ValueError, match="is missing"):
        load_profile_geometry(path)


def test_write_json_is_atomic_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "nested" / "out.json"
    write_json(target, {"b": 1, "a": 2})
    assert json.loads(target.read_text()) == {"a": 2, "b": 1}
    assert list(target.parent.iterdir()) == [target]
