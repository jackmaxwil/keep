from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np


class TensorIndex(Protocol):
    weight_map: dict[str, str]


MAX_SAFETENSORS_HEADER_BYTES = 100_000_000


@dataclass(frozen=True)
class SafetensorsTensorHeader:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count


@dataclass(frozen=True)
class SafetensorsFileHeader:
    header_size: int
    tensors: dict[str, SafetensorsTensorHeader]

    @property
    def payload_offset(self) -> int:
        return 8 + self.header_size

    @property
    def declared_payload_bytes(self) -> int:
        return max((header.data_offsets[1] for header in self.tensors.values()), default=0)


_MLX_DTYPE_MAP = {
    "F32": (np.dtype("<f4"), None),
    "F16": (np.dtype("<f2"), None),
    "U8": (np.dtype("u1"), None),
    "I8": (np.dtype("i1"), None),
    "U16": (np.dtype("<u2"), None),
    "I16": (np.dtype("<i2"), None),
    "I32": (np.dtype("<i4"), None),
    "I64": (np.dtype("<i8"), None),
}


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"{path} is not a valid safetensors file")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        if header_size <= 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"{path} safetensors header size must be between 1 and "
                f"{MAX_SAFETENSORS_HEADER_BYTES} bytes, got {header_size}"
            )
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(
                f"{path} has a truncated safetensors header: expected "
                f"{header_size} bytes, got {len(raw_header)}"
            )
        return header_size, json.loads(raw_header)


def read_safetensors_file_header(path: str | Path) -> SafetensorsFileHeader:
    """Read and validate every tensor descriptor without loading tensor payloads."""

    checkpoint_path = Path(path)
    header_size, raw_header = _read_safetensors_header(checkpoint_path)
    if not isinstance(raw_header, dict):
        raise ValueError(f"{checkpoint_path} safetensors header must be an object")
    tensors: dict[str, SafetensorsTensorHeader] = {}
    for tensor_name, tensor_info in raw_header.items():
        if tensor_name == "__metadata__":
            continue
        if not isinstance(tensor_info, dict):
            raise ValueError(
                f"{checkpoint_path} tensor header for {tensor_name!r} must be an object"
            )
        try:
            dtype_raw = tensor_info["dtype"]
            shape_raw = tensor_info["shape"]
            data_offsets_raw = tensor_info["data_offsets"]
        except KeyError as error:
            raise ValueError(
                f"{checkpoint_path} has an invalid tensor header for {tensor_name!r}"
            ) from error
        if (
            not isinstance(dtype_raw, str)
            or not dtype_raw
            or not isinstance(shape_raw, list)
            or any(type(dim) is not int or dim < 0 for dim in shape_raw)
            or not isinstance(data_offsets_raw, list)
            or len(data_offsets_raw) != 2
            or any(type(offset) is not int for offset in data_offsets_raw)
        ):
            raise ValueError(
                f"{checkpoint_path} has an invalid tensor header for {tensor_name!r}"
            )
        tensors[str(tensor_name)] = SafetensorsTensorHeader(
            dtype=dtype_raw,
            shape=tuple(shape_raw),
            data_offsets=(data_offsets_raw[0], data_offsets_raw[1]),
        )
    return SafetensorsFileHeader(header_size=header_size, tensors=tensors)


def read_safetensors_tensor_header(path: str | Path, tensor_name: str) -> SafetensorsTensorHeader:
    checkpoint_path = Path(path)
    file_header = read_safetensors_file_header(checkpoint_path)
    tensor_header = file_header.tensors.get(tensor_name)
    if tensor_header is None:
        raise KeyError(f"{tensor_name!r} not found in {checkpoint_path}")
    return tensor_header


def read_safetensors_tensor_bytes(
    path: str | Path,
    tensor_name: str,
) -> tuple[SafetensorsTensorHeader, bytes]:
    checkpoint_path = Path(path)
    header_size, header = _read_safetensors_header(checkpoint_path)
    tensor_info = header.get(tensor_name)
    if tensor_info is None:
        raise KeyError(f"{tensor_name!r} not found in {checkpoint_path}")
    tensor_header = SafetensorsTensorHeader(
        dtype=str(tensor_info["dtype"]),
        shape=tuple(int(dim) for dim in tensor_info["shape"]),
        data_offsets=tuple(int(offset) for offset in tensor_info["data_offsets"]),
    )
    start, end = tensor_header.data_offsets
    with checkpoint_path.open("rb") as handle:
        handle.seek(8 + header_size + start)
        raw = handle.read(end - start)
    return tensor_header, raw


def read_safetensors_tensor_mlx(path: str | Path, tensor_name: str) -> mx.array:
    """Read one tensor from a safetensors shard without changing its logical dtype."""

    checkpoint_path = Path(path)
    tensor_header, raw = read_safetensors_tensor_bytes(checkpoint_path, tensor_name)
    if tensor_header.dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2").copy()
        return mx.array(words).view(mx.bfloat16).reshape(tensor_header.shape)

    dtype_info = _MLX_DTYPE_MAP.get(tensor_header.dtype)
    if dtype_info is None:
        raise ValueError(f"unsupported safetensors dtype {tensor_header.dtype!r} for {tensor_name!r}")
    dtype, _ = dtype_info
    values = np.frombuffer(raw, dtype=dtype).reshape(tensor_header.shape).copy()
    return mx.array(values)


def read_indexed_safetensors_tensor_mlx(
    source_dir: str | Path,
    index: TensorIndex,
    tensor_name: str,
) -> mx.array:
    shard_name = index.weight_map.get(tensor_name)
    if shard_name is None:
        raise KeyError(f"{tensor_name!r} is missing from the safetensors index")
    return read_safetensors_tensor_mlx(Path(source_dir) / shard_name, tensor_name)
