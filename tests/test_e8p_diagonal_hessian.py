from __future__ import annotations

import numpy as np

from mlx_vq.codebook.e8 import (
    e8_1bit_grid,
    e8p_full_grid,
    encode_e8p_rtn,
    encode_e8p_rtn_diagonal_hessian,
)
from mlx_vq.quant.rtn import nearest_codebook_indices_diagonal_hessian


def _exhaustive(vectors, diagonal):
    return nearest_codebook_indices_diagonal_hessian(
        vectors, diagonal, codebook=e8p_full_grid().astype(np.float32),
        index_dtype=np.dtype(np.uint16), vector_chunk_size=4096, codebook_chunk_size=8192,
    )


def test_factored_matches_exhaustive_bit_for_bit() -> None:
    rng = np.random.default_rng(20260711)
    for _ in range(5):
        vecs = (rng.standard_normal((3000, 8)) * 0.6).astype(np.float32)
        diag = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
        exact = _exhaustive(vecs, diag)
        fast = encode_e8p_rtn_diagonal_hessian(vecs, diag)
        np.testing.assert_array_equal(fast, exact)


def test_per_row_diagonal_matches_exhaustive_bit_for_bit() -> None:
    rng = np.random.default_rng(20260712)
    vecs = (rng.standard_normal((32, 8)) * 0.6).astype(np.float32)
    diagonals = np.abs(rng.standard_normal((32, 8))).astype(np.float32) + 0.05
    exact = np.concatenate(
        [_exhaustive(vecs[row : row + 1], diagonals[row]) for row in range(32)]
    )
    fast = encode_e8p_rtn_diagonal_hessian(vecs, diagonals)
    np.testing.assert_array_equal(fast, exact)


def test_per_row_shared_diagonal_matches_shared_diagonal_path() -> None:
    rng = np.random.default_rng(20260713)
    vecs = (rng.standard_normal((256, 8)) * 0.6).astype(np.float32)
    diagonal = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
    diagonals = np.broadcast_to(diagonal, vecs.shape)
    np.testing.assert_array_equal(
        encode_e8p_rtn_diagonal_hessian(vecs, diagonals),
        encode_e8p_rtn_diagonal_hessian(vecs, diagonal),
    )


def test_unit_diagonal_reduces_to_unweighted_encoder() -> None:
    rng = np.random.default_rng(7)
    vecs = (rng.standard_normal((500, 8)) * 0.6).astype(np.float32)
    ones = np.ones(8, dtype=np.float32)
    np.testing.assert_array_equal(
        encode_e8p_rtn_diagonal_hessian(vecs, ones),
        encode_e8p_rtn(vecs),
    )


def test_rejects_bad_diagonal() -> None:
    import pytest

    with pytest.raises(ValueError):
        encode_e8p_rtn_diagonal_hessian(np.zeros((1, 8), np.float32), np.zeros(8, np.float32))


def test_exact_tie_breaks_to_lowest_code() -> None:
    # A target equidistant to code A and code B under the metric must resolve to
    # min(A, B), matching the exhaustive full-grid argmin.
    rng = np.random.default_rng(101)
    grid = e8p_full_grid().astype(np.float32)
    diag = np.ones(8, dtype=np.float32)
    # Construct targets at the midpoint of adjacent grid rows so ties occur.
    lo = grid[:2000]
    hi = grid[1:2001]
    midpoints = ((lo + hi) / 2.0).astype(np.float32)
    exact = _exhaustive(midpoints, diag)
    fast = encode_e8p_rtn_diagonal_hessian(midpoints, diag)
    np.testing.assert_array_equal(fast, exact)


def test_dispatcher_numpy_backend_matches_exhaustive() -> None:
    from mlx_vq.quant.rtn import nearest_e8p_codes_diagonal_hessian

    rng = np.random.default_rng(55)
    vecs = (rng.standard_normal((4000, 8)) * 0.6).astype(np.float32)
    diag = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
    np.testing.assert_array_equal(
        nearest_e8p_codes_diagonal_hessian(vecs, diag, backend="numpy"),
        _exhaustive(vecs, diag),
    )


def test_materialize_fast_path_matches_exhaustive_end_to_end() -> None:
    from mlx_vq.convert.glm52_recovery_materialize import (
        quantize_weight_importance_aware,
    )

    rng = np.random.default_rng(2026)
    weight = (rng.standard_normal((16, 64)) * 0.3).astype(np.float32)
    diagonal = np.abs(rng.standard_normal(64)).astype(np.float32) + 0.05
    fast = quantize_weight_importance_aware(
        weight,
        diagonal,
        group_size=64,
        code_bits=16,
        iterations=3,
        e8p_search_backend="numpy",
    )
    exhaustive = quantize_weight_importance_aware(
        weight,
        diagonal,
        group_size=64,
        code_bits=16,
        iterations=3,
        e8p_search_backend="exhaustive",
    )
    np.testing.assert_array_equal(fast.codes, exhaustive.codes)
    np.testing.assert_array_equal(fast.scales, exhaustive.scales)


def test_materialize_batched_e8_matches_per_codeword_numpy(monkeypatch) -> None:
    import mlx_vq.convert.glm52_recovery_materialize as materialize
    from mlx_vq.quant.rtn import nearest_e8_codes_diagonal_hessian

    rng = np.random.default_rng(20260714)
    weight = (rng.standard_normal((5, 32)) * 0.3).astype(np.float32)
    diagonal = np.abs(rng.standard_normal(32)).astype(np.float32) + 0.05
    group_size = 16
    iterations = 3

    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def recording_search(vectors, diagonals, *, backend, index_dtype):
        calls.append((vectors.shape, diagonals.shape))
        return nearest_e8_codes_diagonal_hessian(
            vectors,
            diagonals,
            backend=backend,
            index_dtype=index_dtype,
        )

    monkeypatch.setattr(
        materialize, "nearest_e8_codes_diagonal_hessian", recording_search
    )
    batched = materialize.quantize_weight_importance_aware(
        weight,
        diagonal,
        group_size=group_size,
        code_bits=8,
        iterations=iterations,
        e8p_search_backend="numpy",
    )

    table = e8_1bit_grid().astype(np.float32)
    out_dim, in_dim = weight.shape
    group_count = in_dim // group_size
    codewords_per_group = group_size // 8
    grouped = weight.reshape(out_dim, group_count, codewords_per_group, 8)
    grouped_hessian = diagonal.reshape(group_count, codewords_per_group, 8)
    scales = np.max(np.abs(grouped), axis=(2, 3)) / np.float32(
        np.max(np.abs(table))
    )
    scales = np.where(scales > 0, scales, np.float32(1.0)).astype(np.float32)
    expected_codes = np.empty(
        (out_dim, group_count, codewords_per_group), dtype=np.uint8
    )

    def assign_per_codeword() -> None:
        for group in range(group_count):
            for codeword in range(codewords_per_group):
                hessian = grouped_hessian[group, codeword]
                normalized = grouped[:, group, codeword, :] / scales[:, group, None]
                vector_norm = np.sum(normalized * normalized * hessian, axis=1)
                table_norm = np.sum(table * table * hessian[None, :], axis=1)
                distances = (
                    vector_norm[:, None]
                    + table_norm[None, :]
                    - 2.0 * (normalized * hessian[None, :]) @ table.T
                )
                expected_codes[:, group, codeword] = np.argmin(
                    distances, axis=1
                ).astype(np.uint8)

    for _ in range(iterations):
        assign_per_codeword()
        decoded = table[expected_codes]
        numerator = np.sum(
            grouped_hessian[None, ...] * grouped * decoded,
            axis=(2, 3),
            dtype=np.float64,
        )
        denominator = np.sum(
            grouped_hessian[None, ...] * decoded * decoded,
            axis=(2, 3),
            dtype=np.float64,
        )
        scales = np.divide(
            numerator,
            denominator,
            out=scales.astype(np.float64),
            where=(denominator > 0) & (numerator > 0),
        ).astype(np.float32)
    assign_per_codeword()

    expected_shape = (out_dim * group_count * codewords_per_group, 8)
    assert calls == [(expected_shape, expected_shape)] * (iterations + 1)
    np.testing.assert_array_equal(batched.codes, expected_codes.reshape(out_dim, -1))
    np.testing.assert_array_equal(batched.scales, scales.astype(np.float16))


def test_dispatcher_default_backend_is_byte_identical_numpy() -> None:
    # The default (no backend argument) must be the deterministic, byte-identity
    # NumPy path — never the Metal accelerator. Guards the reviewed regression.
    from mlx_vq.quant.rtn import nearest_e8p_codes_diagonal_hessian

    rng = np.random.default_rng(999)
    vecs = (rng.standard_normal((4000, 8)) * 0.6).astype(np.float32)
    diag = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
    np.testing.assert_array_equal(
        nearest_e8p_codes_diagonal_hessian(vecs, diag),
        _exhaustive(vecs, diag),
    )
