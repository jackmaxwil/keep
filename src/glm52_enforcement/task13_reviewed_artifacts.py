"""Closed versioned transport for the remaining Task 13 reviewed artifacts."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Optional

from .canonical import canonical_json_bytes
from .task13_clean_rehearsal import (
    CLEAN_REHEARSAL_KEY_PREFIX,
    clean_rehearsal_artifact_key,
    validate_clean_rehearsal_coordinate,
    validate_clean_rehearsal_evidence,
)
from .task13_fixed_artifacts import (
    activation_artifact_key,
    validate_activation_artifact_key,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
RETAINED_MODELS_BUCKET = "keep-glm52-models-246813579024-us-west-2"

REVIEWED_ARTIFACT_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "TASK11_REVIEW_APPROVAL": "reviews/task11/approval.json",
        "TASK12_REVIEW_APPROVAL": "reviews/task12/approval.json",
        "RETAINED_TEMPLATE": "task13/templates/retained.yaml",
        "FENCE_TEMPLATE": "task13/migration/fence-transfer.json",
        "SUPPORT_TEMPLATE": "task13/templates/support-disabled.yaml",
        "BOOTSTRAP_TEMPLATE": (
            "task13/templates/container-bootstrap-v1.json"
        ),
        "ACCEPTED_BASELINE": "task13/inputs/accepted-baseline.json",
        "PROMPT_PACK": "task13/inputs/prompt-pack.json",
        "TRAINING_CONFIGURATION": (
            "task13/inputs/training-configuration.json"
        ),
        "GPU_SPEND_APPROVAL": "task13/approvals/gpu-spend.json",
        "SUPPORT_APPROVAL": "task13/approvals/support-plane.json",
        "RESIDUAL_LIABILITY_APPROVAL": (
            "task13/approvals/residual-liability.json"
        ),
        "TASK10_WORKER_DESCRIPTOR": (
            "task13/production/task10-worker-descriptor.json"
        ),
        "TASK10_TASK_INPUTS": (
            "task13/production/task10-task-inputs.json"
        ),
    }
)
_TEMPLATE_KINDS = frozenset(
    {
        "RETAINED_TEMPLATE",
        "FENCE_TEMPLATE",
        "SUPPORT_TEMPLATE",
        "BOOTSTRAP_TEMPLATE",
    }
)
_ACTIVATION_SCOPED_REVIEWED_KINDS = frozenset({"PRODUCTION_DESCRIPTOR"})
_COORDINATE_FIELDS = frozenset(
    {
        "artifact_kind",
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_ID = re.compile(
    r"(?!null\Z)(?!None\Z)[\x21-\x7e]{1,1024}\Z"
)


class Task13ReviewedArtifactError(ValueError):
    """A local reviewed source or its versioned S3 truth failed closed."""


@dataclass(frozen=True)
class Task13ReviewedArtifactServices:
    """Typed single-account transport configured for one SDK attempt."""

    sts: object
    s3: object
    total_max_attempts: int


def _fail(message: str) -> None:
    raise Task13ReviewedArtifactError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _checksum(raw: bytes) -> str:
    return base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")


def _expected_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(label + " must be one lowercase SHA-256")
    return value


def _version_id(value: object, label: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or not value.isascii()
        or _VERSION_ID.fullmatch(value) is None
    ):
        _fail(label + " must be one opaque non-null VersionId")
    return value


def _metadata(response: object, operation: str) -> Mapping[str, object]:
    if type(response) is not dict:
        _fail(operation + " returned no exact response object")
    metadata = response.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RetryAttempts")) is not int
        or metadata["RetryAttempts"] != 0
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        _fail(operation + " lacks authenticated zero-retry success")
    return response


def _call(
    client: object,
    method_name: str,
    *,
    operation: str,
    request: Mapping[str, object],
) -> Mapping[str, object]:
    method = getattr(client, method_name, None)
    if not callable(method):
        _fail(operation + " typed client method is absent")
    return _metadata(method(**dict(request)), operation)


def _guard_services(
    services: Task13ReviewedArtifactServices,
    bucket: str,
) -> None:
    if (
        type(services) is not Task13ReviewedArtifactServices
        or services.total_max_attempts != 1
        or bucket != RETAINED_MODELS_BUCKET
    ):
        _fail("Task 13 reviewed-artifact transport is not exact")
    identity = _call(
        services.sts,
        "get_caller_identity",
        operation="GetCallerIdentity",
        request={},
    )
    arn = identity.get("Arn")
    if (
        identity.get("Account") != ACCOUNT_ID
        or type(arn) is not str
        or f"::{ACCOUNT_ID}:" not in arn
        or type(identity.get("UserId")) is not str
        or not identity["UserId"]
    ):
        _fail("Task 13 reviewed-artifact caller account is foreign")
    versioning = _call(
        services.s3,
        "get_bucket_versioning",
        operation="GetBucketVersioning",
        request={
            "Bucket": RETAINED_MODELS_BUCKET,
            "ExpectedBucketOwner": ACCOUNT_ID,
        },
    )
    if versioning.get("Status") != "Enabled":
        _fail("retained models bucket versioning is not enabled")


def _validate_cloudformation_body(
    value: Mapping[str, object],
    artifact_kind: str,
) -> None:
    resources = value.get("Resources")
    if (
        value.get("AWSTemplateFormatVersion") != "2010-09-09"
        or type(resources) is not dict
        or not resources
        or any(
            type(resource) is not dict
            or type(resource.get("Type")) is not str
            or not resource["Type"]
            for resource in resources.values()
        )
    ):
        _fail(artifact_kind + " is not one canonical CloudFormation body")


def _read_source(
    path: Path,
    artifact_kind: str,
) -> tuple[bytes, dict[str, object]]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        _fail("reviewed artifact source must be one regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Task13ReviewedArtifactError(
            "reviewed artifact source is not canonical UTF-8 JSON"
        ) from exc
    if (
        type(value) is not dict
        or not value
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        _fail("reviewed artifact source is not one canonical JSON object plus LF")
    if artifact_kind in _TEMPLATE_KINDS:
        _validate_cloudformation_body(value, artifact_kind)
    if artifact_kind == "CLEAN_REHEARSAL":
        try:
            validate_clean_rehearsal_evidence(value)
        except ValueError as error:
            raise Task13ReviewedArtifactError(
                "CLEAN_REHEARSAL evidence identity drifted"
            ) from error
    return raw, value


def _inventory_versions(
    *,
    s3: object,
    key: str,
) -> tuple[dict[str, object], ...]:
    key_marker: Optional[str] = None
    version_marker: Optional[str] = None
    seen: set[tuple[str, str]] = set()
    exact: list[dict[str, object]] = []
    for _page in range(1024):
        request: dict[str, object] = {
            "Bucket": RETAINED_MODELS_BUCKET,
            "Prefix": key,
            "MaxKeys": 1000,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
            request["VersionIdMarker"] = version_marker
        response = _call(
            s3,
            "list_object_versions",
            operation="ListObjectVersions",
            request=request,
        )
        versions = response.get("Versions", [])
        delete_markers = response.get("DeleteMarkers", [])
        truncated = response.get("IsTruncated")
        if (
            type(versions) is not list
            or type(delete_markers) is not list
            or type(truncated) is not bool
            or any(type(item) is not dict for item in versions)
            or any(type(item) is not dict for item in delete_markers)
        ):
            _fail("reviewed fixed-key S3 version inventory is malformed")
        if any(item.get("Key") == key for item in delete_markers):
            _fail("reviewed fixed-key S3 history contains a delete marker")
        exact.extend(
            dict(item) for item in versions if item.get("Key") == key
        )
        if not truncated:
            return tuple(exact)
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or (next_key, next_version) in seen
        ):
            _fail("reviewed fixed-key S3 version pagination is incomplete")
        seen.add((next_key, next_version))
        key_marker = next_key
        version_marker = next_version
    _fail("reviewed fixed-key S3 version pagination exceeded its bound")


def _read_exact_version(
    *,
    s3: object,
    key: str,
    version_id: str,
    raw: bytes,
    metadata: Mapping[str, str],
) -> None:
    response = _call(
        s3,
        "get_object",
        operation="GetObject",
        request={
            "Bucket": RETAINED_MODELS_BUCKET,
            "Key": key,
            "VersionId": version_id,
            "ExpectedBucketOwner": ACCOUNT_ID,
            "ChecksumMode": "ENABLED",
        },
    )
    body = response.get("Body")
    read = getattr(body, "read", None)
    observed = read() if callable(read) else None
    close = getattr(body, "close", None)
    if callable(close):
        close()
    if (
        type(observed) is not bytes
        or observed != raw
        or response.get("ContentLength") != len(raw)
        or response.get("VersionId") != version_id
        or response.get("ChecksumSHA256") != _checksum(raw)
        or response.get("Metadata") != dict(metadata)
    ):
        _fail("reviewed fixed-key S3 VersionId readback drifted")


def _adopt_exact(
    *,
    s3: object,
    key: str,
    raw: bytes,
    metadata: Mapping[str, str],
) -> Optional[str]:
    versions = _inventory_versions(s3=s3, key=key)
    if not versions:
        return None
    if len(versions) != 1:
        _fail("reviewed fixed-key S3 positive truth is not singular")
    version_id = _version_id(
        versions[0].get("VersionId"),
        "reviewed fixed-key S3 VersionId",
    )
    if versions[0].get("Size") != len(raw):
        _fail("reviewed fixed-key S3 version size is foreign")
    _read_exact_version(
        s3=s3,
        key=key,
        version_id=version_id,
        raw=raw,
        metadata=metadata,
    )
    return version_id


def _publish_once(
    *,
    s3: object,
    artifact_kind: str,
    key: str,
    raw: bytes,
) -> str:
    file_sha = _sha256(raw)
    body_sha = _sha256(raw[:-1])
    metadata = {
        "artifact-kind": artifact_kind,
        "body-sha256": body_sha,
        "file-sha256": file_sha,
        "glm52-run-id": RUN_ID,
    }
    existing = _adopt_exact(
        s3=s3,
        key=key,
        raw=raw,
        metadata=metadata,
    )
    if existing is not None:
        return existing
    request = {
        "Bucket": RETAINED_MODELS_BUCKET,
        "Key": key,
        "Body": raw,
        "ContentType": "application/json",
        "ChecksumAlgorithm": "SHA256",
        "ChecksumSHA256": _checksum(raw),
        "IfNoneMatch": "*",
        "ExpectedBucketOwner": ACCOUNT_ID,
        "Metadata": metadata,
    }
    try:
        response = _call(
            s3,
            "put_object",
            operation="PutObject",
            request=request,
        )
        version_id = _version_id(
            response.get("VersionId"),
            "PutObject VersionId",
        )
        if response.get("ChecksumSHA256") != _checksum(raw):
            _fail("PutObject checksum response drifted")
    except Exception as original_error:
        try:
            adopted = _adopt_exact(
                s3=s3,
                key=key,
                raw=raw,
                metadata=metadata,
            )
        except Exception as readback_error:
            raise Task13ReviewedArtifactError(
                "ambiguous reviewed PutObject has no exact adoption: "
                + str(readback_error)
            ) from original_error
        if adopted is None:
            raise Task13ReviewedArtifactError(
                "ambiguous reviewed PutObject has no durable version"
            ) from original_error
        return adopted
    adopted = _adopt_exact(
        s3=s3,
        key=key,
        raw=raw,
        metadata=metadata,
    )
    if adopted != version_id:
        _fail("PutObject VersionId is not the singular durable version")
    return version_id


def publish_reviewed_artifact(
    *,
    artifact_kind: str,
    source_path: Path,
    expected_file_sha256: str,
    expected_body_sha256: str,
    bucket: str,
    services: Task13ReviewedArtifactServices,
    activation_id: str | None = None,
) -> dict[str, object]:
    """Publish or exactly adopt one kind from the closed reviewed map."""

    if (
        type(artifact_kind) is not str
        or (
            artifact_kind != "CLEAN_REHEARSAL"
            and artifact_kind not in REVIEWED_ARTIFACT_KEYS
            and artifact_kind not in _ACTIVATION_SCOPED_REVIEWED_KINDS
        )
    ):
        _fail("reviewed artifact kind is not in the closed allowlist")
    if artifact_kind in _ACTIVATION_SCOPED_REVIEWED_KINDS:
        try:
            activation_artifact_key(artifact_kind, activation_id)
        except ValueError as error:
            raise Task13ReviewedArtifactError(
                "activation-scoped reviewed artifact activation_id is invalid"
            ) from error
    elif activation_id is not None:
        _fail("fixed reviewed artifact cannot carry activation_id")
    if bucket != RETAINED_MODELS_BUCKET:
        _fail("reviewed artifact bucket is not the retained models bucket")
    raw, value = _read_source(source_path, artifact_kind)
    reviewed_file_sha = _expected_sha256(
        expected_file_sha256,
        "reviewed file SHA-256",
    )
    reviewed_body_sha = _expected_sha256(
        expected_body_sha256,
        "reviewed body SHA-256",
    )
    if (
        _sha256(raw) != reviewed_file_sha
        or _sha256(raw[:-1]) != reviewed_body_sha
    ):
        _fail("reviewed artifact bytes do not match both reviewed SHA-256 pins")
    _guard_services(services, bucket)
    if artifact_kind == "CLEAN_REHEARSAL":
        key = clean_rehearsal_artifact_key(value)
    elif artifact_kind in _ACTIVATION_SCOPED_REVIEWED_KINDS:
        key = activation_artifact_key(artifact_kind, activation_id)
    else:
        key = REVIEWED_ARTIFACT_KEYS[artifact_kind]
    version = _publish_once(
        s3=services.s3,
        artifact_kind=artifact_kind,
        key=key,
        raw=raw,
    )
    return {
        "artifact_kind": artifact_kind,
        "bucket": RETAINED_MODELS_BUCKET,
        "key": key,
        "version_id": version,
        "file_sha256": reviewed_file_sha,
        "body_sha256": reviewed_body_sha,
    }


def write_coordinate_once(
    path: Path,
    coordinate: Mapping[str, object],
) -> None:
    """Write one canonical coordinate to a new private file."""

    if type(coordinate) is not dict or set(coordinate) != _COORDINATE_FIELDS:
        _fail("reviewed artifact coordinate is not exact")
    artifact_kind = coordinate.get("artifact_kind")
    if artifact_kind == "CLEAN_REHEARSAL":
        try:
            validate_clean_rehearsal_coordinate(coordinate)
        except ValueError as error:
            raise Task13ReviewedArtifactError(
                "reviewed artifact coordinate bucket or key is not exact"
            ) from error
    elif artifact_kind in _ACTIVATION_SCOPED_REVIEWED_KINDS:
        try:
            validate_activation_artifact_key(
                coordinate.get("key"),
                artifact_kind=artifact_kind,
            )
        except ValueError as error:
            raise Task13ReviewedArtifactError(
                "reviewed artifact coordinate bucket or key is not exact"
            ) from error
        if coordinate.get("bucket") != RETAINED_MODELS_BUCKET:
            _fail("reviewed artifact coordinate bucket or key is not exact")
    elif (
        type(artifact_kind) is not str
        or artifact_kind not in REVIEWED_ARTIFACT_KEYS
        or coordinate.get("bucket") != RETAINED_MODELS_BUCKET
        or coordinate.get("key") != REVIEWED_ARTIFACT_KEYS[artifact_kind]
    ):
        _fail("reviewed artifact coordinate bucket or key is not exact")
    _version_id(
        coordinate.get("version_id"),
        "reviewed artifact coordinate VersionId",
    )
    _expected_sha256(
        coordinate.get("file_sha256"),
        "reviewed artifact coordinate file SHA-256",
    )
    _expected_sha256(
        coordinate.get("body_sha256"),
        "reviewed artifact coordinate body SHA-256",
    )
    target = Path(path)
    if (
        not target.is_absolute()
        or not target.parent.is_dir()
        or target.parent.is_symlink()
        or target.exists()
        or target.is_symlink()
    ):
        _fail("coordinate output path is not one new absolute file")
    raw = canonical_json_bytes(coordinate) + b"\n"
    descriptor_number = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    descriptor = os.fdopen(descriptor_number, "wb")
    try:
        descriptor.write(raw)
        descriptor.flush()
        os.fsync(descriptor.fileno())
    except BaseException:
        descriptor.close()
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    descriptor.close()


__all__ = [
    "ACCOUNT_ID",
    "CLEAN_REHEARSAL_KEY_PREFIX",
    "PROFILE",
    "REGION",
    "RETAINED_MODELS_BUCKET",
    "REVIEWED_ARTIFACT_KEYS",
    "RUN_ID",
    "Task13ReviewedArtifactError",
    "Task13ReviewedArtifactServices",
    "publish_reviewed_artifact",
    "write_coordinate_once",
]
