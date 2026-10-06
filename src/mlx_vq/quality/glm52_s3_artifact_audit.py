"""Fail-closed S3 inventory and safetensors layout audit for GLM-5.2.

This module deliberately separates the pure integrity policy from the AWS
transport.  The command-line wrapper supplies S3 HEAD, ranged GET, and full
SHA-256 results; tests can supply in-memory objects.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence


INVENTORY_RECORD_TYPE = "glm52_s3_artifact_inventory_v1"
INVENTORY_SCHEMA_VERSION = 1
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "run_id",
        "bucket",
        "objects",
        "inventory_body_sha256",
    }
)
_UNVERSIONED_OBJECT_FIELDS = frozenset(
    {"key", "size", "sha256", "kind", "safetensors", "run_scope"}
)
_VERSIONED_OBJECT_FIELDS = _UNVERSIONED_OBJECT_FIELDS | {"version_id"}
_KINDS = frozenset(
    {
        "repository_tar",
        "campaign_descriptor",
        "approval",
        "source_model",
        "non_vq_package",
        "training_baseline",
        "prompt_pack",
        "training_configuration",
        "watchdog_code",
        "qualification_cache",
    }
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2FNUZ": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _body_sha256(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = frozenset(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")


def _validate_object(
    value: Mapping[str, object],
    *,
    run_id: str,
) -> dict[str, object]:
    fields = frozenset(value)
    if fields not in {_UNVERSIONED_OBJECT_FIELDS, _VERSIONED_OBJECT_FIELDS}:
        expected = (
            _VERSIONED_OBJECT_FIELDS
            if "version_id" in fields
            else _UNVERSIONED_OBJECT_FIELDS
        )
        _require_exact_fields(value, expected, label="inventory object")
    key = value["key"]
    size = value["size"]
    sha256 = value["sha256"]
    kind = value["kind"]
    safetensors = value["safetensors"]
    run_scope = value["run_scope"]
    version_id = value.get("version_id")
    if not isinstance(key, str) or not key or key.startswith("/") or ".." in key.split("/"):
        raise ValueError("inventory object key must be a safe relative S3 key")
    if type(size) is not int or size <= 0:
        raise ValueError(f"{key!r} size must be a positive integer")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ValueError(f"{key!r} SHA-256 is malformed")
    if kind not in _KINDS:
        raise ValueError(f"{key!r} artifact kind is unsupported")
    if type(safetensors) is not bool:
        raise ValueError(f"{key!r} safetensors flag must be boolean")
    if run_scope not in {run_id, "shared"}:
        raise ValueError(f"{key!r} has foreign run scope {run_scope!r}")
    if run_scope == run_id and not key.startswith(f"campaigns/{run_id}/"):
        raise ValueError(f"{key!r} is outside the immutable campaign prefix")
    if "version_id" in value and (
        not isinstance(version_id, str)
        or not version_id
        or version_id in {"null", "None"}
    ):
        raise ValueError(f"{key!r} VersionId must be opaque and non-null")
    return dict(value)


def build_s3_artifact_inventory(
    *,
    run_id: str,
    bucket: str,
    objects: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be nonempty")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("bucket must be nonempty")
    checked = [_validate_object(item, run_id=run_id) for item in objects]
    if not checked:
        raise ValueError("artifact inventory must not be empty")
    keys = [str(item["key"]) for item in checked]
    if len(keys) != len(set(keys)):
        raise ValueError("artifact inventory contains duplicate keys")
    checked.sort(key=lambda item: str(item["key"]))
    result: dict[str, object] = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "record_type": INVENTORY_RECORD_TYPE,
        "run_id": run_id,
        "bucket": bucket,
        "objects": checked,
    }
    result["inventory_body_sha256"] = _body_sha256(
        result,
        "inventory_body_sha256",
    )
    return result


def validate_s3_artifact_inventory(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("artifact inventory must be an object")
    _require_exact_fields(value, _TOP_FIELDS, label="artifact inventory")
    if value["schema_version"] != INVENTORY_SCHEMA_VERSION:
        raise ValueError("artifact inventory schema version mismatch")
    if value["record_type"] != INVENTORY_RECORD_TYPE:
        raise ValueError("artifact inventory record type mismatch")
    run_id = value["run_id"]
    bucket = value["bucket"]
    objects = value["objects"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("artifact inventory run_id must be nonempty")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("artifact inventory bucket must be nonempty")
    if not isinstance(objects, list) or not objects:
        raise ValueError("artifact inventory objects must be a nonempty list")
    checked = [
        _validate_object(item, run_id=run_id)
        if isinstance(item, dict)
        else (_ for _ in ()).throw(ValueError("inventory object must be an object"))
        for item in objects
    ]
    if checked != sorted(checked, key=lambda item: str(item["key"])):
        raise ValueError("artifact inventory objects must be key-sorted")
    if len({str(item["key"]) for item in checked}) != len(checked):
        raise ValueError("artifact inventory contains duplicate keys")
    expected_sha = _body_sha256(value, "inventory_body_sha256")
    if value["inventory_body_sha256"] != expected_sha:
        raise ValueError("artifact inventory body SHA-256 mismatch")
    return dict(value)


@dataclass(frozen=True)
class SafetensorsObjectReport:
    key: str
    header_bytes: int
    tensor_count: int
    payload_bytes: int
    final_payload_offset: int


def inspect_safetensors_object(
    *,
    key: str,
    object_size: int,
    read_range: Callable[[int, int], bytes],
) -> SafetensorsObjectReport:
    """Authenticate tensor headers against the remote object's exact size."""

    if object_size < 9:
        raise ValueError(f"{key} is too small to be a safetensors object")
    header_size_raw = read_range(0, 8)
    if len(header_size_raw) != 8:
        raise ValueError(f"{key} has a truncated safetensors size prefix")
    header_size = struct.unpack("<Q", header_size_raw)[0]
    if header_size <= 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"{key} has an invalid safetensors header size")
    header_end = 8 + header_size
    if header_end > object_size:
        raise ValueError(f"{key} has a truncated safetensors header")
    header_raw = read_range(8, header_size)
    if len(header_raw) != header_size:
        raise ValueError(f"{key} has a truncated safetensors header")
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{key} has an invalid safetensors header") from error
    if not isinstance(header, dict):
        raise ValueError(f"{key} safetensors header must be an object")

    tensors: list[tuple[int, int, str]] = []
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise ValueError(f"{key} has malformed safetensors metadata")
            continue
        if not isinstance(name, str) or not name or not isinstance(tensor, dict):
            raise ValueError(f"{key} has a malformed tensor entry")
        if set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"{key} tensor {name!r} has malformed fields")
        dtype = tensor["dtype"]
        shape = tensor["shape"]
        offsets = tensor["data_offsets"]
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"{key} tensor {name!r} has unsupported dtype {dtype!r}")
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        ):
            raise ValueError(f"{key} tensor {name!r} has malformed shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise ValueError(f"{key} tensor {name!r} has malformed offsets")
        start, end = offsets
        if start < 0 or end < start:
            raise ValueError(f"{key} tensor {name!r} has invalid payload offsets")
        actual_bytes = end - start
        expected_bytes = math.prod(shape) * _DTYPE_BYTES[str(dtype)]
        if actual_bytes == 0 or expected_bytes == 0:
            raise ValueError(f"{key} tensor {name!r} has a zero-length payload")
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{key} tensor {name!r} dtype/shape byte count does not match offsets"
            )
        if header_end + end > object_size:
            raise ValueError(f"{key} tensor {name!r} payload is outside object")
        tensors.append((start, end, name))

    if not tensors:
        raise ValueError(f"{key} contains no safetensors tensor payloads")
    tensors.sort()
    cursor = 0
    for start, end, name in tensors:
        if start != cursor:
            raise ValueError(
                f"{key} has noncontiguous safetensors payload before {name!r}"
            )
        cursor = end
    if header_end + cursor != object_size:
        raise ValueError(f"{key} safetensors payload does not consume the exact object")
    return SafetensorsObjectReport(
        key=key,
        header_bytes=header_size,
        tensor_count=len(tensors),
        payload_bytes=cursor,
        final_payload_offset=cursor,
    )


@dataclass(frozen=True)
class S3ObjectHead:
    size: int
    etag: str
    sha256: str
    checksum_source: str
    metadata: Mapping[str, str]
    version_id: str | None = None


def audit_s3_artifact_inventory(
    inventory: object,
    *,
    head_object: Callable[[str], S3ObjectHead],
    read_range: Callable[[str, int, int], bytes],
    active_multipart_uploads: Sequence[str],
) -> dict[str, object]:
    checked = validate_s3_artifact_inventory(inventory)
    if active_multipart_uploads:
        raise ValueError(
            "unfinished multipart uploads exist: "
            + ", ".join(sorted(active_multipart_uploads))
        )
    tensor_count = 0
    safetensors_count = 0
    total_bytes = 0
    for raw in checked["objects"]:  # type: ignore[union-attr]
        item = raw
        key = str(item["key"])
        head = head_object(key)
        expected_version = item.get("version_id")
        if expected_version is not None and head.version_id != expected_version:
            raise ValueError(f"{key} S3 VersionId mismatch")
        if head.size != item["size"]:
            raise ValueError(f"{key} S3 size mismatch")
        if head.sha256 != item["sha256"]:
            raise ValueError(f"{key} S3 SHA-256 mismatch")
        if head.checksum_source not in {
            "streamed_sha256",
            "s3_full_object_sha256",
            "s3_batch_full_object_sha256",
        }:
            raise ValueError(f"{key} lacks a full-object SHA-256 authority")
        scope = str(item["run_scope"])
        metadata_scope = head.metadata.get("glm52-run-id")
        if scope != "shared" and metadata_scope not in {None, scope}:
            raise ValueError(f"{key} has foreign-run S3 metadata")
        total_bytes += head.size
        if item["safetensors"]:
            report = inspect_safetensors_object(
                key=key,
                object_size=head.size,
                read_range=lambda start, length, object_key=key: read_range(
                    object_key,
                    start,
                    length,
                ),
            )
            safetensors_count += 1
            tensor_count += report.tensor_count
    return {
        "schema_version": 1,
        "record_type": "glm52_s3_artifact_audit_v1",
        "audit_pass": True,
        "run_id": checked["run_id"],
        "bucket": checked["bucket"],
        "inventory_body_sha256": checked["inventory_body_sha256"],
        "object_count": len(checked["objects"]),  # type: ignore[arg-type]
        "object_bytes": total_bytes,
        "safetensors_object_count": safetensors_count,
        "safetensors_tensor_count": tensor_count,
    }
