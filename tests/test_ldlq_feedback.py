from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

from mlx_vq.codebook.e8 import e8_1bit_grid, e8p_full_grid, encode_e8_1bit_rtn, encode_e8p_rtn


def _load_quality_modules() -> tuple[ModuleType, ModuleType]:
    root = Path(__file__).parents[1]
    quality_package = ModuleType("mlx_vq.quality")
    quality_package.__path__ = [str(root / "src" / "mlx_vq" / "quality")]
    mlx_module = ModuleType("mlx")
    mlx_core_module = ModuleType("mlx.core")
    mlx_module.core = mlx_core_module
    modules = {
        "mlx": mlx_module,
        "mlx.core": mlx_core_module,
        "mlx_vq.quality": quality_package,
    }
    with patch.dict(sys.modules, modules):
        kh_name = "mlx_vq.quality.kronecker_hessian"
        kh_spec = importlib.util.spec_from_file_location(
            kh_name, root / "src" / "mlx_vq" / "quality" / "kronecker_hessian.py"
        )
        assert kh_spec is not None and kh_spec.loader is not None
        kh_module = importlib.util.module_from_spec(kh_spec)
        sys.modules[kh_name] = kh_module
        kh_spec.loader.exec_module(kh_module)

        feedback_name = "mlx_vq.quality.ldlq_feedback"
        feedback_spec = importlib.util.spec_from_file_location(
            feedback_name, root / "src" / "mlx_vq" / "quality" / "ldlq_feedback.py"
        )
        assert feedback_spec is not None and feedback_spec.loader is not None
        feedback_module = importlib.util.module_from_spec(feedback_spec)
        sys.modules[feedback_name] = feedback_module
        feedback_spec.loader.exec_module(feedback_module)
    return kh_module, feedback_module


kronecker_hessian, ldlq_feedback = _load_quality_modules()
blockldlq_hin_only_reassign_codes = kronecker_hessian.blockldlq_hin_only_reassign_codes
hin_weighted_error = kronecker_hessian.hin_weighted_error
LEVER_NAME = ldlq_feedback.LEVER_NAME
ldlq_reassign_codes = ldlq_feedback.ldlq_reassign_codes


CASES = (
    (32, 256, 128, 8, 101),
    (64, 256, 256, 8, 202),
    (32, 512, 128, 8, 303),
    (32, 256, 128, 16, 404),
    (64, 512, 256, 16, 505),
)


@lru_cache(maxsize=2)
def _codebook(code_bits: int) -> np.ndarray:
    if code_bits == 8:
        return e8_1bit_grid().astype(np.float32)
    return e8p_full_grid().astype(np.float32)


@lru_cache(maxsize=None)
def _case(
    output_dim: int,
    input_dim: int,
    group_size: int,
    code_bits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.65, 1.35, size=(output_dim, input_dim // group_size)).astype(np.float32)
    expanded = np.repeat(scales, group_size // 8, axis=1)[..., None]
    source = (rng.normal(0.0, 1.2, size=(output_dim, input_dim // 8, 8)).astype(np.float32) * expanded)
    source = source.reshape(output_dim, input_dim)

    # Dense, well-conditioned PSD Hessian with correlations large enough to
    # exercise feedback without making the acceptance cases pathological.
    rank = min(24, input_dim)
    factor = rng.normal(0.0, 1.0 / np.sqrt(rank), size=(input_dim, rank))
    h_in = (factor @ factor.T + np.eye(input_dim) * 1.5).astype(np.float32)
    return source, scales, h_in, _codebook(code_bits)


def _rtn_seed(
    source: np.ndarray,
    scales: np.ndarray,
    *,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    words_per_scale = group_size // 8
    expanded = np.repeat(scales, words_per_scale, axis=1)[..., None]
    normalized = source.reshape(source.shape[0], -1, 8) / expanded
    if code_bits == 8:
        codes = encode_e8_1bit_rtn(normalized)
    else:
        codes = encode_e8p_rtn(normalized)
    weight = (codebook[codes.astype(np.int64)] * expanded).reshape(source.shape).astype(np.float32)
    return codes, weight


@pytest.mark.parametrize("case", (CASES[0], CASES[3]))
def test_ldlq_feedback_is_bit_deterministic(case: tuple[int, int, int, int, int]) -> None:
    source, scales, h_in, codebook = _case(*case)
    output_dim, input_dim, group_size, code_bits, _seed = case
    assert source.shape == (output_dim, input_dim)
    kwargs = dict(codebook=codebook, group_size=group_size, code_bits=code_bits)

    first = ldlq_reassign_codes(source, scales, h_in, **kwargs)
    second = ldlq_reassign_codes(source, scales, h_in, **kwargs)

    assert LEVER_NAME == "selection_hin_blockldlq_feedback_fp32_v1"
    assert np.array_equal(first.codes, second.codes)
    assert np.array_equal(first.weight, second.weight)


@pytest.mark.parametrize("case", CASES)
def test_ldlq_feedback_dominates_rtn_seed(case: tuple[int, int, int, int, int]) -> None:
    source, scales, h_in, codebook = _case(*case)
    _output_dim, _input_dim, group_size, code_bits, _seed = case
    _seed_codes, seed_weight = _rtn_seed(
        source,
        scales,
        codebook=codebook,
        group_size=group_size,
        code_bits=code_bits,
    )

    result = ldlq_reassign_codes(
        source,
        scales,
        h_in,
        codebook=codebook,
        group_size=group_size,
        code_bits=code_bits,
    )

    seed_error = hin_weighted_error(seed_weight, source, h_in)
    result_error = hin_weighted_error(result.weight, source, h_in)
    assert result_error <= seed_error * (1.0 + 1.0e-12)
    assert result.stats.seed_hin_weighted_error == seed_error
    assert result.stats.ldlq_hin_weighted_error <= seed_error * (1.0 + 1.0e-12)
    assert result.stats.hin_weighted_error == result_error


@pytest.mark.parametrize("case", CASES[:3])
def test_ldlq_plus_one_sweep_tracks_cd_and_is_monotone(
    case: tuple[int, int, int, int, int],
) -> None:
    source, scales, h_in, codebook = _case(*case)
    _output_dim, _input_dim, group_size, code_bits, seed = case
    seed_codes, _seed_weight = _rtn_seed(
        source,
        scales,
        codebook=codebook,
        group_size=group_size,
        code_bits=code_bits,
    )
    ldlq = ldlq_reassign_codes(
        source,
        scales,
        h_in,
        codebook=codebook,
        group_size=group_size,
        code_bits=code_bits,
    )
    refined = ldlq_reassign_codes(
        source,
        scales,
        h_in,
        codebook=codebook,
        group_size=group_size,
        code_bits=code_bits,
        sweeps=1,
    )
    cd = blockldlq_hin_only_reassign_codes(
        source,
        seed_codes,
        scales,
        h_in,
        codebook=codebook,
        group_size=group_size,
        code_bits=code_bits,
        sweeps=1,
    )

    ldlq_error = hin_weighted_error(ldlq.weight, source, h_in)
    refined_error = hin_weighted_error(refined.weight, source, h_in)
    cd_error = hin_weighted_error(cd.weight, source, h_in)
    print(
        f"T3 seed={seed} bits={code_bits} shape={source.shape} "
        f"LDLQ/CD={ldlq_error / cd_error:.6f} "
        f"LDLQ+1/CD={refined_error / cd_error:.6f}"
    )
    assert refined_error <= ldlq_error * (1.0 + 1.0e-12)
    assert refined_error <= 1.02 * cd_error
    assert refined.stats.refined_hin_weighted_error == refined_error


@pytest.mark.parametrize("code_bits", (8, 16))
def test_ldlq_feedback_row_subset_scale_and_dtype_contract(code_bits: int) -> None:
    source, scales, h_in, codebook = _case(32, 256, 128, code_bits, 900 + code_bits)
    rows = np.array([7, 1, 19, 3], dtype=np.int64)
    result = ldlq_reassign_codes(
        source,
        scales,
        h_in,
        codebook=codebook,
        group_size=128,
        code_bits=code_bits,
        row_indices=rows,
    )

    expected_dtype = np.uint8 if code_bits == 8 else np.uint16
    assert result.codes.shape == (rows.size, source.shape[1] // 8)
    assert result.weight.shape == (rows.size, source.shape[1])
    assert result.codes.dtype == expected_dtype
    words_per_scale = 128 // 8
    expanded = np.repeat(scales[rows], words_per_scale, axis=1)[..., None]
    decoded = (codebook[result.codes.astype(np.int64)] * expanded).reshape(result.weight.shape)
    assert np.array_equal(result.weight, decoded.astype(np.float32))
    assert result.stats.row_count == rows.size
