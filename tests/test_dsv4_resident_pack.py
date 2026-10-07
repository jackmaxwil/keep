from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

import mlx_vq.convert.dsv4_resident as resident_module
from mlx_vq.convert.dsv4_resident import (
    audit_dsv4_resident_package,
    pack_dsv4_resident_safetensors,
)
from mlx_vq.convert.stream_convert import load_safetensors_index


def _raw_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in sorted(tensors.items()):
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _fixture_source(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "resident"
    source.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    tensors = {
        "embed.weight": ("BF16", (2, 2), b"\x01\x00" * 4),
        "head.weight": ("BF16", (2, 2), b"\x02\x00" * 4),
        "layers.0.attn.wq_a.scale": ("F8_E8M0", (1, 1), b"\x7f"),
        "layers.0.attn.wq_a.weight": ("F8_E4M3", (2, 2), b"\x38" * 4),
        "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", (1, 1), b"\x7f"),
        "layers.0.ffn.experts.0.w1.weight": ("I8", (2, 1), b"\x12\x34"),
        "layers.0.ffn.gate.tid2eid": ("I64", (1, 1), b"\x00" * 8),
        "mtp.0.main_proj.scale": ("F8_E8M0", (1, 1), b"\x7f"),
        "mtp.0.main_proj.weight": ("F8_E4M3", (2, 2), b"\x3c" * 4),
    }
    _raw_safetensors(source / shard_name, tensors)
    index_path = source / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(len(raw) for _, _, raw in tensors.values())
                },
                "weight_map": {name: shard_name for name in tensors},
            }
        )
    )
    config_path = source / "config.json"
    config_path.write_text("{}\n")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    index = load_safetensors_index(index_path)
    return source, output, index, config_sha, index_sha


def test_dsv4_resident_pack_preserves_release_precision_and_excludes_routed(tmp_path):
    source, output, index, config_sha, index_sha = _fixture_source(tmp_path)

    result = pack_dsv4_resident_safetensors(
        source,
        index,
        output,
        model_id="fixture/dsv4",
        revision="fixture-revision",
        config_sha256=config_sha,
        index_sha256=index_sha,
        max_shard_payload_bytes=16,
        enforce_pinned_source=False,
    )
    audit = audit_dsv4_resident_package(
        output,
        source_dir=source,
        source_index=index,
        expected_model_id="fixture/dsv4",
        expected_revision="fixture-revision",
        expected_config_sha256=config_sha,
        expected_index_sha256=index_sha,
        enforce_pinned_source=False,
    )

    manifest = result.manifest
    assert audit.audit_pass is True
    assert manifest["retained_tensor_count"] == 7
    assert manifest["logical_parameter_tensor_count"] == 5
    assert manifest["scale_tensor_count"] == 2
    assert manifest["excluded_routed_tensor_count"] == 2
    assert manifest["duplicate_resident_tensor_count"] == 0
    assert manifest["missing_resident_tensor_count"] == 0
    assert manifest["routed_resident_tensor_count"] == 0
    assert manifest["dtype_tensor_counts"] == {
        "BF16": 2,
        "F8_E4M3": 2,
        "F8_E8M0": 2,
        "I64": 1,
    }
    assert manifest["quantization"] == {
        "affine_weight_dtype": "F8_E4M3",
        "block_geometry": [128, 128],
        "block_scale_dtype": "F8_E8M0",
        "high_precision_policy": "preserve_source_dtype",
    }
    assert not any(".ffn.experts." in tensor["name"] for tensor in manifest["tensors"])


def test_failed_mandatory_audit_never_publishes_official_output(tmp_path, monkeypatch):
    source, output, index, config_sha, index_sha = _fixture_source(tmp_path)

    def fail_audit(*args, **kwargs):
        raise ValueError("forced mandatory audit failure")

    monkeypatch.setattr(resident_module, "audit_dsv4_resident_package", fail_audit)

    with pytest.raises(ValueError, match="forced mandatory audit failure"):
        pack_dsv4_resident_safetensors(
            source,
            index,
            output,
            model_id="fixture/dsv4",
            revision="fixture-revision",
            config_sha256=config_sha,
            index_sha256=index_sha,
            max_shard_payload_bytes=16,
            enforce_pinned_source=False,
        )

    assert not output.exists()
    assert list(tmp_path.glob("resident.partial-*")) == []
