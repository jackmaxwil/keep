from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from keep.quant.rht import apply_inverse_rht_np, apply_rht_np

MODULE_PATH = Path(__file__).parents[1] / "src/keep/quality/rotation_search.py"
SPEC = importlib.util.spec_from_file_location("rotation_search", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
rotation_search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rotation_search
SPEC.loader.exec_module(rotation_search)

evaluate_rotation_gain = rotation_search.evaluate_rotation_gain
generate_rht_signs = rotation_search.generate_rht_signs
rotate_weight_rht = rotation_search.rotate_weight_rht
select_projection_rotation = rotation_search.select_projection_rotation


def test_rotated_weight_matches_runtime_activation_rht() -> None:
    rng = np.random.default_rng(20260711)
    weight = rng.normal(size=(7, 16)).astype(np.float32)
    activations = rng.normal(size=(5, 16)).astype(np.float32)
    signs = generate_rht_signs(16, seed=91)

    rotated_weight = rotate_weight_rht(weight, signs)
    rotated_activations = apply_rht_np(activations, signs)

    np.testing.assert_allclose(
        rotated_activations @ rotated_weight.T,
        activations @ weight.T,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def _metal_ok() -> bool:
    import mlx.core as mx

    metal = getattr(mx, "metal", None)
    try:
        if metal is None or not bool(metal.is_available()):
            return False
        mx.eval(mx.zeros((), stream=mx.gpu))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
def test_quantized_switch_linear_rht_matches_equivalent_unrotated_weight() -> None:
    import mlx.core as mx

    from keep.vq.e8 import decode_weight_matrix, e8_1bit_packed
    from ramp.nn.switch_linear import QuantizedVQSwitchLinear

    rng = np.random.default_rng(20260712)
    codes = rng.integers(0, 256, size=(1, 3, 1), dtype=np.uint8)
    scales = rng.uniform(0.25, 0.75, size=(1, 3, 1)).astype(np.float32)
    signs = generate_rht_signs(8, seed="switch-linear-runtime")
    layer = QuantizedVQSwitchLinear(
        input_dims=8,
        output_dims=3,
        num_experts=1,
        codes=mx.array(codes),
        scales=mx.array(scales),
        codebook=mx.array(e8_1bit_packed()),
        group_size=8,
        use_gather_vqmm=False,
        rht_signs=mx.array(signs),
    )
    activations = rng.normal(size=(4, 8)).astype(np.float32)
    indices = np.zeros((4,), dtype=np.int32)

    rotated_weight = decode_weight_matrix(
        codes[0], scales[0], code_bits=8, codebook=e8_1bit_packed()
    )
    equivalent_unrotated_weight = apply_inverse_rht_np(rotated_weight, signs)
    expected = activations @ equivalent_unrotated_weight.T
    actual = np.asarray(layer(mx.array(activations), mx.array(indices)))

    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_sign_generation_and_gain_evaluation_are_deterministic() -> None:
    rng = np.random.default_rng(7)
    weight = rng.normal(size=(3, 8)).astype(np.float32)
    importance = np.linspace(0.25, 2.0, 8, dtype=np.float32)
    codebook = np.stack(
        (np.ones(8, dtype=np.float32), np.array([1, -1] * 4, dtype=np.float32))
    )

    first_signs = generate_rht_signs(8, seed="projection-3-down")
    second_signs = generate_rht_signs(8, seed="projection-3-down")
    first_gain = evaluate_rotation_gain(weight, importance, codebook, 8, 8, first_signs)
    second_gain = evaluate_rotation_gain(weight, importance, codebook, 8, 8, second_signs)

    np.testing.assert_array_equal(first_signs, second_signs)
    assert first_signs.dtype == np.int8
    assert set(np.unique(first_signs)) <= {-1, 1}
    assert first_gain == second_gain


def test_gain_evaluator_matches_hand_computed_single_codeword_case() -> None:
    weight = np.arange(1, 9, dtype=np.float32)[None, :]
    importance = np.arange(1, 9, dtype=np.float32)
    codebook = np.ones((1, 8), dtype=np.float32)
    signs = np.ones(8, dtype=np.int8)

    result = evaluate_rotation_gain(weight, importance, codebook, 8, 8, signs)

    baseline_scale = float(np.sum(importance * weight[0]) / np.sum(importance))
    baseline_expected = float(np.sum(importance * (weight[0] - baseline_scale) ** 2))
    rotated_source = apply_rht_np(weight, signs)
    rotated_scale = float(np.mean(rotated_source))
    rotated_quantized = np.full_like(rotated_source, rotated_scale)
    equivalent_original = apply_rht_np(rotated_quantized, np.ones(8, dtype=np.int8))
    rotated_expected = float(np.sum(importance * (weight - equivalent_original) ** 2))

    assert np.isclose(result.unrotated_objective, baseline_expected, rtol=1.0e-6)
    assert np.isclose(result.rotated_objective, rotated_expected, rtol=1.0e-6)
    assert np.isclose(result.objective_gain, baseline_expected - rotated_expected, rtol=1.0e-6)


def test_selection_rejects_tie() -> None:
    weight = np.zeros((2, 8), dtype=np.float32)
    importance = np.ones(8, dtype=np.float32)
    codebook = np.ones((1, 8), dtype=np.float32)
    signs = generate_rht_signs(8, seed=0)

    decision = select_projection_rotation(weight, importance, codebook, 8, 8, signs)

    assert decision.selected is False
    assert decision.selected_signs is None
    assert decision.objective_gain == 0.0


def test_non_power_of_two_width_is_rejected_to_match_runtime() -> None:
    with pytest.raises(ValueError, match="positive power of two"):
        generate_rht_signs(6144, seed=0)
