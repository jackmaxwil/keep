from __future__ import annotations

import hashlib
import json
import struct

import pytest

from mlx_vq.quality.glm52_s3_artifact_audit import (
    S3ObjectHead,
    audit_s3_artifact_inventory,
    build_s3_artifact_inventory,
    inspect_safetensors_object,
    validate_s3_artifact_inventory,
)


def _safetensors(header: dict[str, object], payload: bytes) -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def _reader(raw: bytes):
    def read_range(start: int, length: int) -> bytes:
        return raw[start : start + length]

    return read_range


def test_safetensors_remote_layout_requires_complete_nonempty_tensor_payloads() -> None:
    raw = _safetensors(
        {"weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}},
        b"\0" * 8,
    )
    report = inspect_safetensors_object(
        key="source/model-00001.safetensors",
        object_size=len(raw),
        read_range=_reader(raw),
    )
    assert report.tensor_count == 1
    assert report.payload_bytes == 8
    assert report.final_payload_offset == 8

    zero_payload = _safetensors(
        {"weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 0]}},
        b"",
    )
    with pytest.raises(ValueError, match="zero-length"):
        inspect_safetensors_object(
            key="source/zero.safetensors",
            object_size=len(zero_payload),
            read_range=_reader(zero_payload),
        )


@pytest.mark.parametrize(
    ("header", "payload", "error"),
    [
        (
            {"weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 6]}},
            b"\0" * 6,
            "dtype/shape byte count",
        ),
        (
            {
                "a": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                "b": {"dtype": "F16", "shape": [1], "data_offsets": [3, 5]},
            },
            b"\0" * 5,
            "noncontiguous",
        ),
        (
            {"weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}},
            b"\0" * 7,
            "outside object",
        ),
    ],
)
def test_safetensors_remote_layout_rejects_malformed_ranges(
    header: dict[str, object],
    payload: bytes,
    error: str,
) -> None:
    raw = _safetensors(header, payload)
    with pytest.raises(ValueError, match=error):
        inspect_safetensors_object(
            key="bad.safetensors",
            object_size=len(raw),
            read_range=_reader(raw),
        )


def test_inventory_audit_binds_s3_size_hash_metadata_and_tensor_layout() -> None:
    run_id = "glm52-sky-20260723"
    raw = _safetensors(
        {"weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}},
        b"\1" * 8,
    )
    sha = hashlib.sha256(raw).hexdigest()
    inventory = build_s3_artifact_inventory(
        run_id=run_id,
        bucket="keep-glm52-us-west-2-246813579024",
        objects=[
            {
                "key": f"campaigns/{run_id}/model.safetensors",
                "size": len(raw),
                "sha256": sha,
                "kind": "source_model",
                "safetensors": True,
                "run_scope": run_id,
                "version_id": "version-1",
            }
        ],
    )

    def head(_key: str) -> S3ObjectHead:
        return S3ObjectHead(
            size=len(raw),
            etag='"single-part"',
            sha256=sha,
            checksum_source="s3_batch_full_object_sha256",
            metadata={"glm52-run-id": run_id},
            version_id="version-1",
        )

    report = audit_s3_artifact_inventory(
        inventory,
        head_object=head,
        read_range=lambda _key, start, length: raw[start : start + length],
        active_multipart_uploads=(),
    )
    assert report["audit_pass"] is True
    assert report["object_count"] == 1
    assert report["safetensors_tensor_count"] == 1

    with pytest.raises(ValueError, match="unfinished multipart"):
        audit_s3_artifact_inventory(
            inventory,
            head_object=head,
            read_range=lambda _key, start, length: raw[start : start + length],
            active_multipart_uploads=(f"campaigns/{run_id}/partial",),
        )

    with pytest.raises(ValueError, match="VersionId mismatch"):
        audit_s3_artifact_inventory(
            inventory,
            head_object=lambda _key: S3ObjectHead(
                size=len(raw),
                etag='"single-part"',
                sha256=sha,
                checksum_source="s3_batch_full_object_sha256",
                metadata={"glm52-run-id": run_id},
                version_id="foreign-version",
            ),
            read_range=lambda _key, start, length: raw[start : start + length],
            active_multipart_uploads=(),
        )


def test_inventory_rejects_foreign_run_or_unknown_fields() -> None:
    item = {
        "key": "campaigns/other/model.safetensors",
        "size": 100,
        "sha256": "a" * 64,
        "kind": "source_model",
        "safetensors": True,
        "run_scope": "other",
    }
    with pytest.raises(ValueError, match="foreign run"):
        build_s3_artifact_inventory(
            run_id="glm52-sky-20260723",
            bucket="bucket",
            objects=[item],
        )
    item["run_scope"] = "shared"
    inventory = build_s3_artifact_inventory(
        run_id="glm52-sky-20260723",
        bucket="bucket",
        objects=[item],
    )
    inventory["surprise"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        audit_s3_artifact_inventory(
            inventory,
            head_object=lambda _key: None,  # type: ignore[arg-type]
            read_range=lambda _key, _start, _length: b"",
            active_multipart_uploads=(),
        )


def test_inventory_accepts_authenticated_qualification_cache_objects() -> None:
    inventory = build_s3_artifact_inventory(
        run_id="glm52-sky-20260724",
        bucket="bucket",
        objects=[
            {
                "key": "qualification-cache/seeds/run/manifest.json",
                "size": 10,
                "sha256": "a" * 64,
                "kind": "qualification_cache",
                "safetensors": False,
                "run_scope": "shared",
            }
        ],
    )
    assert inventory["objects"][0]["kind"] == "qualification_cache"


def test_inventory_version_id_is_opaque_non_null_and_authenticated() -> None:
    item = {
        "key": "source-snapshot/config.json",
        "size": 10,
        "sha256": "a" * 64,
        "kind": "source_model",
        "safetensors": False,
        "run_scope": "shared",
        "version_id": "3Lg_a.b-c",
    }
    inventory = build_s3_artifact_inventory(
        run_id="glm52-sky-20260724",
        bucket="bucket",
        objects=[item],
    )
    assert inventory["objects"][0]["version_id"] == "3Lg_a.b-c"
    assert validate_s3_artifact_inventory(inventory) == inventory

    for invalid in ("", "null", "None"):
        item["version_id"] = invalid
        with pytest.raises(ValueError, match="VersionId"):
            build_s3_artifact_inventory(
                run_id="glm52-sky-20260724",
                bucket="bucket",
                objects=[item],
            )


def test_s3_audit_cli_is_account_guarded_and_has_full_hash_fallback() -> None:
    source = (
        (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "aws/glm52-gpu/scripts/audit_s3_campaign_artifacts.py"
        )
        .read_text()
    )
    assert "assert_rnd_aws_account.sh" in source
    assert 'args.profile == "default"' in source
    assert "--checksum-mode" in source
    assert "FULL_OBJECT" in source
    assert "--checksum-authority" in source
    assert "s3_batch_full_object_sha256" in source
    assert "_stream_sha256" in source
    assert "list-multipart-uploads" in source
