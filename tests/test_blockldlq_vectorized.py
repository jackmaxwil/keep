from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest


def _load_kronecker_hessian_module() -> ModuleType:
    module_name = "_test_blockldlq_kronecker_hessian"
    module_path = Path(__file__).parents[1] / "src" / "mlx_vq" / "quality" / "kronecker_hessian.py"
    mlx_module = ModuleType("mlx")
    mlx_core_module = ModuleType("mlx.core")
    mlx_module.core = mlx_core_module
    mlx_vq_module = ModuleType("mlx_vq")
    codebook_module = ModuleType("mlx_vq.codebook")
    e8_module = ModuleType("mlx_vq.codebook.e8")
    e8_module.CODEWORD_DIM = 8
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            module_name: module,
            "mlx": mlx_module,
            "mlx.core": mlx_core_module,
            "mlx_vq": mlx_vq_module,
            "mlx_vq.codebook": codebook_module,
            "mlx_vq.codebook.e8": e8_module,
        },
    ):
        spec.loader.exec_module(module)
    return module


kronecker_hessian = _load_kronecker_hessian_module()


@pytest.mark.parametrize(
    ("output_dim", "input_dim", "group_size", "sweeps", "row_indices"),
    [
        (3, 16, 8, 1, None),
        (7, 32, 16, 2, [0, 2, 6]),
        (11, 64, 32, 1, [9, 1, 5, 3]),
        (520, 16, 16, 2, None),
    ],
)
def test_blockldlq_hin_only_vectorized_is_byte_identical_to_scalar_reference(
    output_dim: int,
    input_dim: int,
    group_size: int,
    sweeps: int,
    row_indices: list[int] | None,
) -> None:
    assert hasattr(kronecker_hessian, "_blockldlq_hin_only_reassign_codes_reference")
    reference = kronecker_hessian._blockldlq_hin_only_reassign_codes_reference
    rng = np.random.default_rng(10_000 + output_dim * 100 + input_dim + sweeps)
    codebook = rng.normal(size=(256, 8)).astype(np.float32)
    source = rng.normal(size=(output_dim, input_dim)).astype(np.float32)
    codes = rng.integers(0, 256, size=(output_dim, input_dim // 8), dtype=np.uint8)
    scales = rng.uniform(0.1, 1.5, size=(output_dim, input_dim // group_size)).astype(np.float32)
    h_factor = rng.normal(size=(input_dim, input_dim))
    h_in = h_factor.T @ h_factor + np.eye(input_dim, dtype=np.float64) * 0.25
    kwargs = {
        "codebook": codebook,
        "group_size": group_size,
        "row_indices": row_indices,
        "sweeps": sweeps,
    }

    expected = reference(source, codes, scales, h_in, **kwargs)
    actual = kronecker_hessian.blockldlq_hin_only_reassign_codes(
        source,
        codes,
        scales,
        h_in,
        **kwargs,
    )

    assert np.array_equal(actual.codes, expected.codes)
    assert np.array_equal(actual.weight, expected.weight)
    assert asdict(actual.stats) == asdict(expected.stats)


def test_blockldlq_hin_only_vectorized_preserves_unchanged_coordinate_skips() -> None:
    reference = kronecker_hessian._blockldlq_hin_only_reassign_codes_reference
    rng = np.random.default_rng(4242)
    codebook = rng.normal(size=(256, 8)).astype(np.float32)
    codes = rng.integers(0, 256, size=(5, 4), dtype=np.uint8)
    scales = rng.uniform(0.1, 1.5, size=(5, 2)).astype(np.float32)
    expanded_scales = np.repeat(scales, 2, axis=1)[..., None]
    source = (codebook[codes.astype(np.int64)] * expanded_scales).reshape(5, 32).astype(np.float32)
    h_in = np.eye(32, dtype=np.float64)
    kwargs = {"codebook": codebook, "group_size": 16, "sweeps": 2}

    expected = reference(source, codes, scales, h_in, **kwargs)
    actual = kronecker_hessian.blockldlq_hin_only_reassign_codes(
        source,
        codes,
        scales,
        h_in,
        **kwargs,
    )

    assert expected.stats.changed_code_count == 0
    assert np.array_equal(actual.codes, expected.codes)
    assert np.array_equal(actual.weight, expected.weight)
    assert asdict(actual.stats) == asdict(expected.stats)


def test_blockldlq_hin_only_vectorized_argmin_keeps_lowest_code_on_tie() -> None:
    codebook = np.zeros((3, 8), dtype=np.float32)
    codebook[2] = 1.0
    source = np.zeros((2, 8), dtype=np.float32)
    codes = np.full((2, 1), 2, dtype=np.uint8)
    scales = np.ones((2, 1), dtype=np.float32)

    actual = kronecker_hessian.blockldlq_hin_only_reassign_codes(
        source,
        codes,
        scales,
        np.eye(8, dtype=np.float64),
        codebook=codebook,
        group_size=8,
    )

    assert np.array_equal(actual.codes, np.zeros((2, 1), dtype=np.uint8))
