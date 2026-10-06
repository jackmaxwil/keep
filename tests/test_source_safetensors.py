from __future__ import annotations

import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.io import source_safetensors
from mlx_vq.io.source_safetensors import (
    read_indexed_safetensors_tensor_mlx,
    read_safetensors_tensor_header,
    read_safetensors_tensor_mlx,
)


def test_read_safetensors_tensor_mlx_preserves_bfloat16(tmp_path) -> None:
    name = "model.layers.0.self_attn.q_b_proj.weight"
    expected = (np.arange(12, dtype=np.float32).reshape(3, 4) / 8).astype(np.float32)
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(str(shard), {name: mx.array(expected).astype(mx.bfloat16)})

    header = read_safetensors_tensor_header(shard, name)
    actual = read_safetensors_tensor_mlx(shard, name)
    mx.eval(actual)

    assert header.dtype == "BF16"
    assert actual.dtype == mx.bfloat16
    assert actual.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected, rtol=1e-2, atol=1e-2)


def test_read_safetensors_tensor_mlx_preserves_common_dtypes(tmp_path) -> None:
    tensors = {
        "f16": mx.array(np.arange(4, dtype=np.float32)).astype(mx.float16),
        "f32": mx.array(np.arange(4, dtype=np.float32)),
        "u8": mx.array(np.arange(4, dtype=np.uint8)),
        "i32": mx.array(np.arange(4, dtype=np.int32)),
    }
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(str(shard), tensors)

    assert read_safetensors_tensor_mlx(shard, "f16").dtype == mx.float16
    assert read_safetensors_tensor_mlx(shard, "f32").dtype == mx.float32
    assert read_safetensors_tensor_mlx(shard, "u8").dtype == mx.uint8
    assert read_safetensors_tensor_mlx(shard, "i32").dtype == mx.int32


def test_read_safetensors_tensor_bytes_exposes_header_and_exact_payload(tmp_path) -> None:
    name = "packed.weight"
    expected = np.array([[0x10, 0x32], [0x54, 0x76]], dtype=np.uint8)
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(str(shard), {name: mx.array(expected)})

    reader = getattr(source_safetensors, "read_safetensors_tensor_bytes")
    header, raw = reader(shard, name)

    assert header.dtype == "U8"
    assert header.shape == (2, 2)
    assert raw == expected.tobytes(order="C")


def test_read_indexed_safetensors_tensor_mlx_uses_weight_map(tmp_path) -> None:
    name = "model.norm.weight"
    shard_name = "model-00002-of-00002.safetensors"
    mx.save_safetensors(
        str(tmp_path / shard_name),
        {name: mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)).astype(mx.bfloat16)},
    )
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": {name: shard_name}}))

    actual = read_indexed_safetensors_tensor_mlx(tmp_path, load_safetensors_index(index_path), name)
    mx.eval(actual)

    assert actual.dtype == mx.bfloat16
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), [1.0, 2.0, 3.0])


def test_file_header_rejects_oversized_or_truncated_header_lengths(tmp_path) -> None:
    oversized = tmp_path / "oversized.safetensors"
    oversized.write_bytes(struct.pack("<Q", 2**64 - 1))
    truncated = tmp_path / "truncated-header.safetensors"
    truncated.write_bytes(struct.pack("<Q", 64) + b"{}")

    with pytest.raises(ValueError, match="header size"):
        source_safetensors.read_safetensors_file_header(oversized)
    with pytest.raises(ValueError, match="truncated safetensors header"):
        source_safetensors.read_safetensors_file_header(truncated)


@pytest.mark.parametrize(
    "descriptor",
    [
        {"dtype": 123, "shape": [1], "data_offsets": [0, 1]},
        {"dtype": "U8", "shape": [1.5], "data_offsets": [0, 1]},
        {"dtype": "U8", "shape": [1], "data_offsets": ["0", 1]},
    ],
)
def test_file_header_rejects_coerced_descriptor_types(tmp_path, descriptor) -> None:
    encoded = json.dumps({"tensor": descriptor}).encode()
    encoded += b" " * (-len(encoded) % 8)
    shard = tmp_path / "invalid-descriptor.safetensors"
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0")

    with pytest.raises(ValueError, match="invalid tensor header"):
        source_safetensors.read_safetensors_file_header(shard)
