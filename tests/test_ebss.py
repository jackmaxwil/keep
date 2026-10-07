from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import pytest
import sys

from keep.vq.e8 import e8_1bit_grid, e8p_full_grid

_MODULE_PATH = Path(__file__).parents[1] / "src/keep/quality/ebss.py"
_SPEC = importlib.util.spec_from_file_location("_ebss_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_EBSS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _EBSS
_SPEC.loader.exec_module(_EBSS)

DEFAULT_MULTIPLIERS = _EBSS.DEFAULT_MULTIPLIERS
canonical_multiplier_string = _EBSS.canonical_multiplier_string
search_group_scales = _EBSS.search_group_scales


def _weighted_error(
    weight: np.ndarray,
    diagonal: np.ndarray,
    scales: np.ndarray,
    codes: np.ndarray,
    codebook: np.ndarray,
    group_size: int,
) -> np.ndarray:
    out_dim, in_dim = weight.shape
    groups = in_dim // group_size
    words = group_size // 8
    decoded = codebook[codes.reshape(out_dim, groups, words)]
    reconstructed = scales[:, :, None, None] * decoded
    residual = weight.reshape(out_dim, groups, words, 8) - reconstructed
    hessian = diagonal.reshape(groups, words, 8)
    return np.sum(hessian[None, ...] * residual * residual, axis=(2, 3))


@pytest.mark.parametrize(
    ("code_bits", "codebook"),
    [(8, e8_1bit_grid()), (16, e8p_full_grid())],
)
def test_search_never_worsens_baseline_and_reports_choices(code_bits, codebook):
    rng = np.random.default_rng(20260711 + code_bits)
    weight = rng.normal(size=(3, 16)).astype(np.float32)
    diagonal = rng.uniform(0.1, 2.0, size=16).astype(np.float32)

    scales, codes, stats = search_group_scales(
        weight,
        diagonal,
        codebook=codebook,
        group_size=8,
        code_bits=code_bits,
        multipliers=DEFAULT_MULTIPLIERS,
    )

    baseline_scales, baseline_codes, _ = search_group_scales(
        weight,
        diagonal,
        codebook=codebook,
        group_size=8,
        code_bits=code_bits,
        multipliers=(1.0,),
    )
    objective = _weighted_error(weight, diagonal, scales, codes, codebook, 8)
    baseline = _weighted_error(
        weight, diagonal, baseline_scales, baseline_codes, codebook, 8
    )
    assert np.all(objective <= baseline + 1e-6)
    assert stats["chosen_multipliers"].shape == (3, 2)
    np.testing.assert_allclose(stats["selected_objective"], objective, rtol=1e-5)
    np.testing.assert_allclose(stats["baseline_objective"], baseline, rtol=1e-5)
    assert codes.dtype == (np.uint8 if code_bits == 8 else np.uint16)
    assert scales.dtype == np.float32


def test_search_is_deterministic_and_ties_use_canonical_multiplier_order():
    weight = np.zeros((2, 8), dtype=np.float32)
    diagonal = np.ones(8, dtype=np.float32)
    kwargs = dict(
        codebook=e8_1bit_grid(),
        group_size=8,
        code_bits=8,
        multipliers=(1.1, 0.9, 1.0),
    )

    first = search_group_scales(weight, diagonal, **kwargs)
    second = search_group_scales(weight, diagonal, **kwargs)

    for first_array, second_array in zip(first[:2], second[:2], strict=True):
        np.testing.assert_array_equal(first_array, second_array)
    np.testing.assert_array_equal(
        first[2]["chosen_multipliers"], second[2]["chosen_multipliers"]
    )
    np.testing.assert_array_equal(first[2]["chosen_multipliers"], 1.0)


def test_non_baseline_multiplier_strictly_wins_constructed_case():
    weight = np.array(
        [[-0.9096654, 0.2652688, -1.0223621, 1.1899315,
          0.61309457, 0.7068286, 0.27457637, 1.1620569]],
        dtype=np.float32,
    )
    diagonal = np.array(
        [0.25, 0.5, 1.0, 2.0, 4.0, 1.5, 0.75, 3.0], dtype=np.float32
    )

    _, _, stats = search_group_scales(
        weight,
        diagonal,
        codebook=e8_1bit_grid(),
        group_size=8,
        code_bits=8,
        multipliers=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1),
    )

    assert stats["chosen_multipliers"][0, 0] != 1.0
    assert stats["selected_objective"][0, 0] < stats["baseline_objective"][0, 0]


def test_multiplier_policy_has_canonical_string_and_requires_baseline():
    assert canonical_multiplier_string((0.85, 0.9, 1.0, 1.10)) == "0.85,0.9,1.0,1.1"
    with pytest.raises(ValueError, match="include 1.0"):
        search_group_scales(
            np.ones((1, 8), dtype=np.float32),
            np.ones(8, dtype=np.float32),
            codebook=e8_1bit_grid(),
            group_size=8,
            code_bits=8,
            multipliers=(0.9, 1.1),
        )


@pytest.mark.parametrize(
    ("code_bits", "codebook"),
    [(8, e8_1bit_grid()), (16, e8p_full_grid())],
)
def test_search_rejects_permuted_canonical_codebook(code_bits, codebook):
    permuted = codebook.copy()
    permuted[[0, 1]] = permuted[[1, 0]]

    with pytest.raises(ValueError, match="canonical"):
        search_group_scales(
            np.ones((1, 8), dtype=np.float32),
            np.ones(8, dtype=np.float32),
            codebook=permuted,
            group_size=8,
            code_bits=code_bits,
            multipliers=(1.0,),
        )


@pytest.mark.parametrize(
    ("code_bits", "codebook"),
    [(8, e8p_full_grid()), (16, e8_1bit_grid())],
)
def test_search_rejects_codebook_for_wrong_code_bits(code_bits, codebook):
    with pytest.raises(ValueError):
        search_group_scales(
            np.ones((1, 8), dtype=np.float32),
            np.ones(8, dtype=np.float32),
            codebook=codebook,
            group_size=8,
            code_bits=code_bits,
            multipliers=(1.0,),
        )
