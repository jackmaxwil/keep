from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest


E2M1_VALUES = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def _nvfp4() -> Any:
    return importlib.import_module("keep.convert.nvfp4")


def _valid_config() -> dict[str, Any]:
    return {
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "producer": {"name": "modelopt", "version": "0.29.0"},
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": False, "num_bits": 4, "type": "float"},
                    "weights": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                }
            },
        }
    }


def _write_raw_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    header: dict[str, Any] = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header_raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_raw += b" " * (-len(header_raw) % 8)
    path.write_bytes(struct.pack("<Q", len(header_raw)) + header_raw + payload)


def _write_bundle(
    root: Path,
    *,
    weight_dtype: str = "U8",
    weight_shape: list[int] | None = None,
    weight_raw: bytes | None = None,
    scale_dtype: str = "F8_E4M3",
    scale_shape: list[int] | None = None,
    scale_raw: bytes | None = None,
    global_dtype: str = "F32",
    global_shape: list[int] | None = None,
    global_raw: bytes | None = None,
) -> tuple[dict[str, str], str]:
    weight_name = "model.layers.0.mlp.up_proj.weight"
    block_scale_name = "model.layers.0.mlp.up_proj.weight_scale"
    global_scale_name = "model.layers.0.mlp.up_proj.weight_scale_2"
    weight_shard = "model-00001-of-00003.safetensors"
    block_scale_shard = "model-00002-of-00003.safetensors"
    global_scale_shard = "model-00003-of-00003.safetensors"

    _write_raw_safetensors(
        root / weight_shard,
        [
            (
                weight_name,
                weight_dtype,
                weight_shape if weight_shape is not None else [2, 8],
                weight_raw if weight_raw is not None else bytes([0x22] * 8 + [0xBB] * 8),
            )
        ],
    )
    _write_raw_safetensors(
        root / block_scale_shard,
        [
            (
                block_scale_name,
                scale_dtype,
                scale_shape if scale_shape is not None else [2, 1],
                scale_raw if scale_raw is not None else bytes([0x38, 0x40]),
            )
        ],
    )
    _write_raw_safetensors(
        root / global_scale_shard,
        [
            (
                global_scale_name,
                global_dtype,
                global_shape if global_shape is not None else [],
                global_raw if global_raw is not None else struct.pack("<f", 0.25),
            )
        ],
    )
    return (
        {
            weight_name: weight_shard,
            block_scale_name: block_scale_shard,
            global_scale_name: global_scale_shard,
            "model.layers.0.mlp.up_proj.input_scale": "ignored.safetensors",
        },
        weight_name,
    )


def test_parse_modelopt_nvfp4_weight_spec_accepts_only_supported_static_contract() -> None:
    api = _nvfp4()

    actual = api.parse_modelopt_nvfp4_weight_spec(_valid_config())

    assert actual == api.ModelOptNvfp4Spec(
        quant_method="modelopt",
        quant_algo="NVFP4",
        producer_name="modelopt",
        producer_version="0.29.0",
        dynamic=False,
        num_bits=4,
        weight_type="float",
        group_size=16,
    )

    config_without_version = _valid_config()
    del config_without_version["quantization_config"]["producer"]["version"]
    assert api.parse_modelopt_nvfp4_weight_spec(config_without_version).producer_version is None


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"quantization_config": None},
        {"quantization_config": []},
        {**_valid_config(), "quantization_config": {**_valid_config()["quantization_config"], "quant_method": "compressed-tensors"}},
        {**_valid_config(), "quantization_config": {**_valid_config()["quantization_config"], "quant_algo": "FP8"}},
        {**_valid_config(), "quantization_config": {**_valid_config()["quantization_config"], "producer": None}},
        {**_valid_config(), "quantization_config": {**_valid_config()["quantization_config"], "producer": {"name": "other"}}},
        {**_valid_config(), "quantization_config": {**_valid_config()["quantization_config"], "producer": {"name": "modelopt", "version": 29}}},
    ],
    ids=[
        "missing-quantization-config",
        "null-quantization-config",
        "non-mapping-quantization-config",
        "wrong-quant-method",
        "wrong-quant-algo",
        "missing-producer",
        "wrong-producer",
        "non-string-producer-version",
    ],
)
def test_parse_modelopt_nvfp4_weight_spec_rejects_missing_or_wrong_top_level_contract(
    config: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _nvfp4().parse_modelopt_nvfp4_weight_spec(config)


def test_parse_modelopt_nvfp4_weight_spec_requires_exactly_one_mapping_group() -> None:
    api = _nvfp4()
    invalid_groups: list[Any] = [
        None,
        [],
        {},
        {
            "group_0": _valid_config()["quantization_config"]["config_groups"]["group_0"],
            "group_1": _valid_config()["quantization_config"]["config_groups"]["group_0"],
        },
        {"group_0": None},
        {"group_0": {}},
    ]

    for config_groups in invalid_groups:
        config = _valid_config()
        config["quantization_config"]["config_groups"] = config_groups
        with pytest.raises(ValueError):
            api.parse_modelopt_nvfp4_weight_spec(config)


@pytest.mark.parametrize(
    "weights",
    [
        {"dynamic": True, "num_bits": 4, "type": "float", "group_size": 16},
        {"dynamic": 0, "num_bits": 4, "type": "float", "group_size": 16},
        {"dynamic": False, "num_bits": 8, "type": "float", "group_size": 16},
        {"dynamic": False, "num_bits": True, "type": "float", "group_size": 16},
        {"dynamic": False, "num_bits": 4, "type": "int", "group_size": 16},
        {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 32},
        {"dynamic": False, "num_bits": 4, "type": "float"},
        {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16, "strategy": "group"},
    ],
)
def test_parse_modelopt_nvfp4_weight_spec_rejects_non_exact_weight_contract(
    weights: dict[str, Any],
) -> None:
    config = _valid_config()
    config["quantization_config"]["config_groups"]["group_0"]["weights"] = weights

    with pytest.raises(ValueError):
        _nvfp4().parse_modelopt_nvfp4_weight_spec(config)


def test_resolve_modelopt_nvfp4_weight_bundle_resolves_each_companion_independently() -> None:
    api = _nvfp4()
    weight_name = "model.layers.2.mlp.down_proj.weight"
    weight_map = {
        weight_name: "weights.safetensors",
        "model.layers.2.mlp.down_proj.weight_scale": "block-scales.safetensors",
        "model.layers.2.mlp.down_proj.weight_scale_2": "global-scales.safetensors",
    }

    actual = api.resolve_modelopt_nvfp4_weight_bundle(weight_map, weight_name)

    assert actual == api.ModelOptNvfp4WeightBundle(
        weight_name=weight_name,
        weight_shard="weights.safetensors",
        block_scale_name="model.layers.2.mlp.down_proj.weight_scale",
        block_scale_shard="block-scales.safetensors",
        global_scale_name="model.layers.2.mlp.down_proj.weight_scale_2",
        global_scale_shard="global-scales.safetensors",
    )


@pytest.mark.parametrize("missing_suffix", [".weight", ".weight_scale", ".weight_scale_2"])
def test_resolve_modelopt_nvfp4_weight_bundle_rejects_missing_index_entries(
    missing_suffix: str,
) -> None:
    api = _nvfp4()
    base = "model.layers.2.mlp.down_proj"
    weight_name = f"{base}.weight"
    weight_map = {
        weight_name: "weights.safetensors",
        f"{base}.weight_scale": "block-scales.safetensors",
        f"{base}.weight_scale_2": "global-scales.safetensors",
    }
    del weight_map[f"{base}{missing_suffix}"]

    with pytest.raises(KeyError):
        api.resolve_modelopt_nvfp4_weight_bundle(weight_map, weight_name)


def test_resolve_modelopt_nvfp4_weight_bundle_requires_weight_suffix() -> None:
    with pytest.raises(ValueError):
        _nvfp4().resolve_modelopt_nvfp4_weight_bundle({}, "model.layers.2.mlp.down_proj")


def test_decode_modelopt_nvfp4_weight_covers_all_e2m1_values_in_nibble_order() -> None:
    packed = np.array(
        [[(high << 4) | low for low, high in zip(range(0, 16, 2), range(1, 16, 2), strict=True)]],
        dtype=np.uint8,
    )

    actual = _nvfp4().decode_modelopt_nvfp4_weight(
        packed,
        np.array([[0x38]], dtype=np.uint8),
        np.float32(1.0),
    )

    np.testing.assert_array_equal(actual, E2M1_VALUES.reshape(1, 16))
    assert actual[0, 0].view(np.uint32) == np.uint32(0x00000000)
    assert actual[0, 8].view(np.uint32) == np.uint32(0x00000000)


def test_decode_modelopt_nvfp4_weight_places_low_nibble_at_even_columns() -> None:
    actual = _nvfp4().decode_modelopt_nvfp4_weight(
        np.full((1, 8), 0xE1, dtype=np.uint8),
        np.array([[0x38]], dtype=np.uint8),
        1.0,
    )

    np.testing.assert_array_equal(actual[0, 0::2], np.full(8, 0.5, dtype=np.float32))
    np.testing.assert_array_equal(actual[0, 1::2], np.full(8, -4.0, dtype=np.float32))


def test_decode_modelopt_nvfp4_weight_decodes_e4m3fn_boundaries_and_subnormals() -> None:
    scale_codes = np.array([[0x00], [0x01], [0x07], [0x08], [0x77], [0x78], [0x7E]], dtype=np.uint8)
    packed = np.full((scale_codes.shape[0], 8), 0x22, dtype=np.uint8)

    actual = _nvfp4().decode_modelopt_nvfp4_weight(packed, scale_codes, 1.0)

    expected_scales = np.array([0.0, 2.0**-9, 7.0 * 2.0**-9, 2.0**-6, 240.0, 256.0, 448.0])
    np.testing.assert_array_equal(actual[:, 0], expected_scales.astype(np.float32))


@pytest.mark.parametrize("code", [0x7F, 0xFF])
def test_decode_modelopt_nvfp4_weight_rejects_e4m3fn_nan(code: int) -> None:
    with pytest.raises(ValueError, match="NaN|sign"):
        _nvfp4().decode_modelopt_nvfp4_weight(
            np.full((1, 8), 0x22, dtype=np.uint8),
            np.array([[code]], dtype=np.uint8),
            1.0,
        )


@pytest.mark.parametrize("code", [0x80, 0xB8])
def test_decode_modelopt_nvfp4_weight_rejects_signed_block_scales_including_negative_zero(
    code: int,
) -> None:
    with pytest.raises(ValueError, match="sign"):
        _nvfp4().decode_modelopt_nvfp4_weight(
            np.full((1, 8), 0x22, dtype=np.uint8),
            np.array([[code]], dtype=np.uint8),
            1.0,
        )


def test_decode_modelopt_nvfp4_weight_broadcasts_scales_and_chunks_rows() -> None:
    packed = np.full((2, 16), 0x22, dtype=np.uint8)
    block_scales = np.array([[0x38, 0x40], [0x48, 0x50]], dtype=np.uint8)

    actual = _nvfp4().decode_modelopt_nvfp4_weight(
        packed,
        block_scales,
        np.array(0.5, dtype=np.float32),
        row_chunk_size=1,
    )

    expected = np.array(
        [
            [0.5] * 16 + [1.0] * 16,
            [2.0] * 16 + [4.0] * 16,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


def test_decode_modelopt_nvfp4_weight_matches_producer_float32_multiplication_order() -> None:
    actual = _nvfp4().decode_modelopt_nvfp4_weight(
        np.full((1, 8), 0x33, dtype=np.uint8),
        np.array([[0x03]], dtype=np.uint8),
        np.float32(79.736565),
    )

    expected = np.array([0x3F336842], dtype=np.uint32).view(np.float32)[0]
    alternate_rounding = np.array([0x3F336843], dtype=np.uint32).view(np.float32)[0]
    np.testing.assert_array_equal(actual, np.full((1, 16), expected, dtype=np.float32))
    assert expected != alternate_rounding


def test_decode_modelopt_nvfp4_weight_rejects_finite_inputs_that_overflow_output() -> None:
    with pytest.raises(ValueError, match="finite"):
        _nvfp4().decode_modelopt_nvfp4_weight(
            np.full((1, 8), 0x77, dtype=np.uint8),
            np.array([[0x7E]], dtype=np.uint8),
            np.finfo(np.float32).max,
        )


def test_decode_modelopt_nvfp4_weight_allows_large_finite_output_for_small_codes() -> None:
    global_scale = np.float32(np.finfo(np.float32).max / 2.0)

    actual = _nvfp4().decode_modelopt_nvfp4_weight(
        np.full((1, 8), 0x11, dtype=np.uint8),
        np.array([[0x38]], dtype=np.uint8),
        global_scale,
    )

    np.testing.assert_array_equal(actual, np.full((1, 16), global_scale * np.float32(0.5)))
    assert np.isfinite(actual).all()


@pytest.mark.parametrize(
    ("packed", "scales", "global_scale", "row_chunk_size"),
    [
        (np.zeros((1, 8), dtype=np.int8), np.zeros((1, 1), dtype=np.uint8), 1.0, 64),
        (np.zeros(8, dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), 1.0, 64),
        (np.zeros((1, 7), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), 1.0, 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 1), dtype=np.int8), 1.0, 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros(1, dtype=np.uint8), 1.0, 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8), 1.0, 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((2, 1), dtype=np.uint8), 1.0, 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), np.array([1.0]), 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), 0.0, 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), -1.0, 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), np.inf, 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), np.nan, 64),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), 1.0, 0),
    ],
    ids=[
        "weight-dtype",
        "weight-rank",
        "weight-shape",
        "scale-dtype",
        "scale-rank",
        "scale-columns",
        "scale-rows",
        "global-nonscalar",
        "global-zero",
        "global-negative",
        "global-infinite",
        "global-nan",
        "row-chunk-size",
    ],
)
def test_decode_modelopt_nvfp4_weight_rejects_invalid_contract(
    packed: np.ndarray,
    scales: np.ndarray,
    global_scale: Any,
    row_chunk_size: int,
) -> None:
    with pytest.raises(ValueError):
        _nvfp4().decode_modelopt_nvfp4_weight(
            packed,
            scales,
            global_scale,
            row_chunk_size=row_chunk_size,
        )


def test_read_modelopt_nvfp4_weight_reads_cross_shard_manual_safetensors_fixture(tmp_path) -> None:
    api = _nvfp4()
    weight_map, weight_name = _write_bundle(tmp_path)
    bundle = api.resolve_modelopt_nvfp4_weight_bundle(weight_map, weight_name)

    actual = api.read_modelopt_nvfp4_weight(tmp_path, bundle)

    expected = np.array([[0.25] * 16, [-0.75] * 16], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


def test_read_modelopt_nvfp4_weight_raises_for_missing_shard(tmp_path) -> None:
    api = _nvfp4()
    weight_map, weight_name = _write_bundle(tmp_path)
    bundle = api.resolve_modelopt_nvfp4_weight_bundle(weight_map, weight_name)
    (tmp_path / bundle.block_scale_shard).unlink()

    with pytest.raises(FileNotFoundError):
        api.read_modelopt_nvfp4_weight(tmp_path, bundle)


@pytest.mark.parametrize(
    "overrides",
    [
        {"weight_dtype": "I8"},
        {"weight_shape": [16]},
        {"weight_shape": [2, 8], "weight_raw": bytes([0x22] * 15)},
        {"scale_dtype": "U8"},
        {"scale_shape": [2]},
        {"scale_shape": [2, 2]},
        {"scale_shape": [2, 1], "scale_raw": bytes([0x38])},
        {"global_dtype": "F16", "global_raw": struct.pack("<e", 0.25)},
        {"global_shape": [1]},
        {"global_shape": [], "global_raw": b"\x00\x00\x80"},
        {"global_raw": struct.pack("<f", float("inf"))},
        {"global_raw": struct.pack("<f", 0.0)},
    ],
    ids=[
        "weight-dtype",
        "weight-rank",
        "weight-payload-length",
        "scale-dtype",
        "scale-rank",
        "scale-shape",
        "scale-payload-length",
        "global-dtype",
        "global-rank",
        "global-payload-length",
        "global-nonfinite",
        "global-nonpositive",
    ],
)
def test_read_modelopt_nvfp4_weight_rejects_invalid_safetensors_contract(
    tmp_path,
    overrides: dict[str, Any],
) -> None:
    api = _nvfp4()
    weight_map, weight_name = _write_bundle(tmp_path, **overrides)
    bundle = api.resolve_modelopt_nvfp4_weight_bundle(weight_map, weight_name)

    with pytest.raises(ValueError):
        api.read_modelopt_nvfp4_weight(tmp_path, bundle)
