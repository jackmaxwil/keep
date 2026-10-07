from __future__ import annotations

import numpy as np

from keep.vq.e8 import (
    E8_1BIT_PACKED_SHA256,
    E8P_PACKED_ABS_SHA256,
    QUIP_SHARP_SOURCE_COMMIT,
    cosine_similarity,
    decode_e8_1bit,
    decode_e8p,
    decode_weight_matrix,
    e8_1bit_grid,
    e8_1bit_packed,
    e8p_abs_grid,
    e8p_full_grid,
    e8p_packed_abs_grid,
    encode_e8_1bit_rtn,
    encode_e8p_rtn,
)


def test_quip_sharp_codebook_sources_are_pinned() -> None:
    assert QUIP_SHARP_SOURCE_COMMIT == "1d8f873e9a2a8b86b12bb1064c312c5689b77d98"
    assert E8P_PACKED_ABS_SHA256 == "efc2c03c60acd955dc812ff2bade9ec6cc31259807e48473506161f3a9a230ce"
    assert E8_1BIT_PACKED_SHA256 == "b78586631069e239d29c146538722d763f4a0040d59a800e8c918ecef03e2180"


def test_e8p_packed_abs_grid_contract() -> None:
    packed = e8p_packed_abs_grid()
    assert packed.shape == (256,)
    assert packed.dtype == np.uint32
    assert packed[:5].tolist() == [2576980377, 1503238553, 3650722201, 2040117657, 3113859481]
    assert packed[-5:].tolist() == [2075900345, 1505467321, 1539021241, 1538898361, 2073803675]


def test_e8p_decode_matches_quip_full_grid_golden_vectors() -> None:
    codes = np.array([0, 1, 255, 256, 0x1234, 0xFFFF], dtype=np.uint16)
    expected = np.array(
        [
            [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
            [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
            [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
            [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.25],
            [-0.75, -0.75, 0.25, -0.75, -1.75, 0.25, 0.25, 1.25],
            [-1.25, -1.25, -0.25, -0.25, -1.25, -1.25, -1.25, 0.75],
        ],
        dtype=np.float32,
    )
    decoded = decode_e8p(codes)
    np.testing.assert_array_equal(decoded, expected)
    np.testing.assert_array_equal(e8p_full_grid()[codes], expected)


def test_e8p_abs_grid_uses_decoded_signed_base_order() -> None:
    base_grid = e8p_abs_grid()
    assert base_grid.shape == (256, 8)
    np.testing.assert_array_equal(base_grid[0], np.full((8,), 0.5, dtype=np.float32))
    np.testing.assert_array_equal(
        base_grid[1],
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1.5], dtype=np.float32),
    )


def test_e8_1bit_vq1_table_contract_is_not_e8p_low_byte_decode() -> None:
    packed = e8_1bit_packed()
    assert packed.shape == (256,)
    assert packed.dtype == np.uint32

    codes = np.array([0, 1, 112, 113, 240, 241, 255], dtype=np.uint8)
    expected = np.array(
        [
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(decode_e8_1bit(codes), expected)
    assert not np.array_equal(decode_e8_1bit(np.arange(256, dtype=np.uint8)), decode_e8p(np.arange(256, dtype=np.uint16)))


def test_reference_encode_paths_round_codebook_vectors_back_to_indices() -> None:
    e8_codes = np.array([0, 56, 112, 113, 240, 255], dtype=np.uint8)
    np.testing.assert_array_equal(encode_e8_1bit_rtn(e8_1bit_grid()[e8_codes]), e8_codes)

    e8p_codes = np.array([0, 1, 255, 256, 0x1234, 0xFFFF], dtype=np.uint16)
    np.testing.assert_array_equal(encode_e8p_rtn(decode_e8p(e8p_codes), chunk_size=4096), e8p_codes)


def test_e8p_structured_encoder_matches_full_grid_nearest_distance() -> None:
    rng = np.random.default_rng(20260626)
    vectors = rng.normal(scale=1.25, size=(11, 8)).astype(np.float32)

    fast_codes = encode_e8p_rtn(vectors, chunk_size=3)
    full_grid = e8p_full_grid()
    distances = np.sum((vectors[:, None, :] - full_grid[None, :, :]) ** 2, axis=2)
    brute_codes = np.argmin(distances, axis=1).astype(np.uint16)

    fast_dist = np.sum((vectors - decode_e8p(fast_codes)) ** 2, axis=1)
    brute_dist = np.sum((vectors - decode_e8p(brute_codes)) ** 2, axis=1)
    np.testing.assert_allclose(fast_dist, brute_dist, rtol=1e-5, atol=1e-5)


def test_decode_weight_matrix_matches_independent_numpy_table_reference() -> None:
    rng = np.random.default_rng(20260623)
    codes = rng.integers(0, 256, size=(4, 16), dtype=np.uint8)
    scales = rng.normal(loc=1.0, scale=0.05, size=(4, 2)).astype(np.float32)

    decoded = decode_weight_matrix(codes, scales, code_bits=8)
    manual = e8_1bit_grid()[codes]
    manual = manual * np.repeat(scales, 8, axis=-1)[..., None]
    manual = manual.reshape(4, 128)

    np.testing.assert_allclose(decoded, manual, rtol=0.0, atol=0.0)
    assert cosine_similarity(decoded, manual) >= 0.999999
