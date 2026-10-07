from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

from ramp.models import dsv4_composite_loader as loader
from ramp.models.dsv4_composite_loader import validate_dsv4_vq_manifest_structure


def _manifest(backbone=2, mtp=3):
    blocks = []
    for kind, count in (("backbone", backbone), ("mtp", mtp)):
        for index in range(count):
            stem = f"layer-{index:05d}" if kind == "backbone" else f"mtp-{index:05d}"
            prefix = (
                f"model.layers.{index}.ffn.switch_mlp"
                if kind == "backbone"
                else f"mtp_drafter.blocks.{index}.ffn.switch_mlp"
            )
            blocks.append(
                {
                    "kind": kind,
                    "index": index,
                    "key": f"layers.{index}" if kind == "backbone" else f"mtp.{index}",
                    "file_stem": stem,
                    "tensor_prefix": prefix,
                    "files": [
                        {
                            "path": f"{stem}-{projection}.safetensors",
                            "projection": projection,
                            "bytes": 100 + projection_index,
                            "sha256": str(projection_index + 1) * 64,
                        }
                        for projection_index, projection in enumerate(
                            ("gate_proj", "up_proj", "down_proj")
                        )
                    ],
                }
            )
    return {
        "record_type": "dsv4_vq_artifact_manifest_v1",
        "family": "deepseek-v4-flash",
        "source": {
            "hf_model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            "config_sha256": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
            "index_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
        },
        "geometry": {
            "backbone_layers": backbone,
            "mtp_blocks": mtp,
            "moe_blocks": backbone + mtp,
        },
        "blocks": blocks,
    }


def test_vq_manifest_structure_partitions_every_backbone_and_mtp_file_once():
    result = validate_dsv4_vq_manifest_structure(
        _manifest(), expected_backbone_layers=2, expected_mtp_blocks=3
    )
    assert result.backbone_indices == (0, 1)
    assert result.mtp_indices == (0, 1, 2)
    assert len(result.file_records) == 15
    assert len(set(result.file_records)) == 15
    assert len(result.inventory_sha256) == 64


@pytest.mark.parametrize(
    "mutation", ["duplicate_block", "duplicate_file", "missing_mtp"]
)
def test_vq_manifest_structure_fails_closed_on_ambiguous_or_partial_inventory(mutation):
    manifest = _manifest()
    if mutation == "duplicate_block":
        manifest["blocks"].append(copy.deepcopy(manifest["blocks"][0]))
    elif mutation == "duplicate_file":
        manifest["blocks"][1]["files"][0]["path"] = manifest["blocks"][0]["files"][0][
            "path"
        ]
    else:
        manifest["blocks"] = [
            block for block in manifest["blocks"] if block["key"] != "mtp.2"
        ]

    with pytest.raises(ValueError, match="duplicate|exact|inventory"):
        validate_dsv4_vq_manifest_structure(
            manifest, expected_backbone_layers=2, expected_mtp_blocks=3
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_fixture(tmp_path: Path, *, extra_vq_tensor: bool = False):
    resident_dir = tmp_path / "resident"
    vq_dir = tmp_path / "vq"
    resident_dir.mkdir()
    vq_dir.mkdir()

    resident_name = "model-00001-of-00001.safetensors"
    resident_path = resident_dir / resident_name
    mx.save_safetensors(
        str(resident_path), {"embed.weight": mx.zeros((2, 2), dtype=mx.float32)}
    )
    index_path = resident_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"embed.weight": resident_name}}) + "\n"
    )
    resident_manifest = {
        "shards": [
            {
                "filename": resident_name,
                "file_bytes": resident_path.stat().st_size,
                "file_sha256": _sha256(resident_path),
                "tensor_count": 1,
            }
        ],
        "tensors": [
            {
                "name": "embed.weight",
                "output_shard": resident_name,
                "dtype": "F32",
                "shape": [2, 2],
            }
        ],
        "package_index_sha256": _sha256(index_path),
    }

    prefix = "model.layers.0.ffn.switch_mlp.gate_proj"
    vq_name = "layer-00000-gate_proj.safetensors"
    vq_path = vq_dir / vq_name
    tensors = {
        "model.vq_codebook.e8": mx.zeros((256,), dtype=mx.uint32),
        f"{prefix}.codes": mx.zeros((1, 1, 1), dtype=mx.uint16),
        f"{prefix}.scales": mx.ones((1, 1, 1), dtype=mx.float16),
    }
    if extra_vq_tensor:
        tensors["unexpected.tensor"] = mx.zeros((1,), dtype=mx.float32)
    mx.save_safetensors(str(vq_path), tensors)
    record = {
        "path": vq_name,
        "projection": "gate_proj",
        "bytes": vq_path.stat().st_size,
        "sha256": _sha256(vq_path),
        "codes_tensor": f"{prefix}.codes",
        "scales_tensor": f"{prefix}.scales",
    }
    inventory = loader.Dsv4VqManifestInventory(
        backbone_indices=(0,),
        mtp_indices=(),
        file_records={vq_name: record},
        inventory_sha256="1" * 64,
    )
    return resident_dir, resident_manifest, vq_dir, inventory, vq_path


def test_payload_receipt_rejects_same_size_mutation_even_with_restored_mtime(tmp_path):
    resident_dir, resident, vq_dir, inventory, vq_path = _payload_fixture(tmp_path)
    receipt = loader.authenticate_dsv4_payloads(
        resident_dir=resident_dir,
        resident_manifest=resident,
        vq_dir=vq_dir,
        vq_inventory=inventory,
    )
    receipt_sha256 = loader.dsv4_payload_receipt_sha256(receipt)
    before = vq_path.stat()
    payload = bytearray(vq_path.read_bytes())
    payload[-1] ^= 1
    vq_path.write_bytes(payload)
    os.utime(vq_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert vq_path.stat().st_size == before.st_size
    assert vq_path.stat().st_mtime_ns == before.st_mtime_ns

    with pytest.raises(ValueError, match="current file identity"):
        loader.validate_dsv4_payload_receipt(
            resident_dir=resident_dir,
            resident_manifest=resident,
            vq_dir=vq_dir,
            vq_inventory=inventory,
            receipt=receipt,
            expected_receipt_sha256=receipt_sha256,
        )


def test_payload_authentication_rejects_unexpected_routed_safetensors(tmp_path):
    resident_dir, resident, vq_dir, inventory, _ = _payload_fixture(tmp_path)
    mx.save_safetensors(
        str(vq_dir / "unexpected.safetensors"),
        {"unexpected": mx.zeros((1,), dtype=mx.float32)},
    )

    with pytest.raises(ValueError, match="runtime safetensors inventory"):
        loader.authenticate_dsv4_payloads(
            resident_dir=resident_dir,
            resident_manifest=resident,
            vq_dir=vq_dir,
            vq_inventory=inventory,
        )


def test_payload_authentication_rejects_extra_tensor_in_declared_file(tmp_path):
    resident_dir, resident, vq_dir, inventory, _ = _payload_fixture(
        tmp_path, extra_vq_tensor=True
    )

    with pytest.raises(ValueError, match="tensor inventory"):
        loader.authenticate_dsv4_payloads(
            resident_dir=resident_dir,
            resident_manifest=resident,
            vq_dir=vq_dir,
            vq_inventory=inventory,
        )
