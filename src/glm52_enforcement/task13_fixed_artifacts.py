"""Immutable fixed-key Task 13 reviewed-artifact materialization.

This boundary creates no cloud identity. Callers provide exact local source
bytes. The boundary validates those bytes, uses one conditional S3 write, and
then accepts only one byte-identical versioned readback.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Mapping, Optional, Sequence

from .canonical import canonical_json_bytes, canonical_sha256


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
CAMPAIGN_BUCKET = "keep-glm52-models-246813579024-us-west-2"

_SUPPORT_INPUTS_FIXED_KEY = "task13/inputs/support-build-inputs.json"
_ACTIVATION_ARTIFACT_SUFFIXES = {
    "QUALIFICATION_CACHE_SEED_INPUT": (
        "qualification/cache-seed-input.json"
    ),
    "H100_QUALIFICATION_INPUT": "qualification/h100-input.json",
    "PRODUCTION_DESCRIPTOR": "inputs/campaign-descriptor-v2.json",
}
_DRIVER_CONTRACTS = {
    "qualification-cache-seed": (
        "QUALIFICATION_CACHE_SEED_INPUT",
        "aws/glm52-gpu/scripts/submit_sky_campaign.py",
    ),
    "h100-qualification": (
        "H100_QUALIFICATION_INPUT",
        "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh",
    ),
}
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "profile",
        "run_id",
        "activation_id",
        "operation_kind",
        "argv",
        "environment",
        "source_coordinates",
        "canonical_identity_sha256",
    }
)
_SOURCE_FIELDS = frozenset(
    {"role", "path", "size_bytes", "file_sha256"}
)
_DRIVER_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "operation_kind",
        "argv",
        "environment",
        "canonical_identity_sha256",
    }
)
_ARCHIVE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "archive_file_sha256",
        "archive_size_bytes",
        "archive",
        "activation_id",
        "canonical_identity_sha256",
    }
)
_ACTIVATION = re.compile(r"[a-z0-9][a-z0-9-]{2,63}\Z")
_ARCHIVE_MANIFEST_KEY = re.compile(
    r"task13/activations/(?P<activation>[a-z0-9][a-z0-9-]{2,63})/"
    r"archive/repo-tar\.json\Z"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_ID = re.compile(r"(?!null\Z)[\x21-\x7e]{1,1024}\Z")


class Task13FixedArtifactError(ValueError):
    """Local source or versioned S3 truth failed closed."""


@dataclass(frozen=True)
class Task13FixedArtifactServices:
    """One-account, zero-retry transport boundary."""

    sts: object
    s3: object
    total_max_attempts: int
    publisher_s3: object | None = None

class FixedKeyPublicationDisposition(str, Enum):
    """Whether this call created or adopted the singular durable version."""

    WRITTEN = "WRITTEN"
    ADOPTED = "ADOPTED"


@dataclass(frozen=True)
class FixedKeyPublicationOutcome:
    """Truthful outcome of one exact fixed-key create-or-adopt operation."""

    version_id: str
    disposition: FixedKeyPublicationDisposition

    def __post_init__(self) -> None:
        _version(self.version_id, "fixed-key publication VersionId")
        if type(self.disposition) is not FixedKeyPublicationDisposition:
            _fail("fixed-key publication disposition is not exact")

    @property
    def adopted(self) -> bool:
        return self.disposition is FixedKeyPublicationDisposition.ADOPTED


def _fail(message: str) -> None:
    raise Task13FixedArtifactError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _checksum(raw: bytes) -> str:
    return base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")


def activation_artifact_key(
    artifact_kind: object,
    activation_id: object,
) -> str:
    """Return the immutable key for one activation-scoped reviewed artifact."""

    if (
        type(artifact_kind) is not str
        or artifact_kind not in _ACTIVATION_ARTIFACT_SUFFIXES
    ):
        _fail("activation artifact kind is not implemented")
    if (
        type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
    ):
        _fail("activation artifact activation_id is invalid")
    return (
        f"task13/activations/{activation_id}/"
        + _ACTIVATION_ARTIFACT_SUFFIXES[artifact_kind]
    )


def validate_activation_artifact_key(
    key: object,
    *,
    artifact_kind: object,
    expected_activation_id: object | None = None,
) -> str:
    """Validate one activation-scoped key, optionally against its activation."""

    if (
        type(artifact_kind) is not str
        or artifact_kind not in _ACTIVATION_ARTIFACT_SUFFIXES
    ):
        _fail("activation artifact kind is not implemented")
    suffix = "/" + _ACTIVATION_ARTIFACT_SUFFIXES[artifact_kind]
    prefix = "task13/activations/"
    if (
        type(key) is not str
        or not key.startswith(prefix)
        or not key.endswith(suffix)
    ):
        _fail("activation artifact key is invalid")
    activation_id = key[len(prefix) : -len(suffix)]
    if (
        _ACTIVATION.fullmatch(activation_id) is None
        or "/" in activation_id
        or (
            expected_activation_id is not None
            and activation_id != expected_activation_id
        )
    ):
        _fail("activation artifact key activation lineage drifted")
    return key


def _text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or value != value.strip()
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        _fail(label + " must be one nonempty exact ASCII string")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(label + " must be one lowercase SHA-256")
    return value


def _version(value: object, label: str) -> str:
    text = _text(value, label)
    if _VERSION_ID.fullmatch(text) is None:
        _fail(label + " must be one opaque non-null VersionId")
    return text


def _metadata(response: object, operation: str) -> Mapping[str, object]:
    if type(response) is not dict:
        _fail(operation + " returned no exact response object")
    metadata = response.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or metadata.get("RetryAttempts") != 0
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
    services: Task13FixedArtifactServices,
    bucket: str,
) -> None:
    if (
        type(services) is not Task13FixedArtifactServices
        or services.total_max_attempts != 1
        or bucket != CAMPAIGN_BUCKET
    ):
        _fail("Task 13 fixed-artifact transport is not exact")
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
        _fail("Task 13 fixed-artifact caller account is foreign")
    versioning = _call(
        services.s3,
        "get_bucket_versioning",
        operation="GetBucketVersioning",
        request={
            "Bucket": bucket,
            "ExpectedBucketOwner": ACCOUNT_ID,
        },
    )
    if versioning.get("Status") != "Enabled":
        _fail("Task 13 fixed-artifact bucket versioning is not enabled")


def _canonical_mapping(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Task13FixedArtifactError(
            label + " is not canonical ASCII JSON"
        ) from exc
    if (
        type(value) is not dict
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        _fail(label + " is not one canonical JSON object plus LF")
    return value


def _read_canonical_mapping(path: Path, label: str) -> dict[str, object]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        _fail(label + " must be one regular non-symlink file")
    return _canonical_mapping(path.read_bytes(), label)


def _source_coordinate(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _SOURCE_FIELDS:
        _fail("driver source coordinate schema drifted")
    role = _text(value["role"], "driver source role")
    path = Path(_text(value["path"], role + " source path"))
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
    ):
        _fail(role + " source must be one concrete regular absolute file")
    size = value["size_bytes"]
    if type(size) is not int or size < 1:
        _fail(role + " source size must be positive")
    expected_sha = _sha(
        value["file_sha256"],
        role + " source file SHA-256",
    )
    raw = path.read_bytes()
    if len(raw) != size or _sha256(raw) != expected_sha:
        _fail(role + " source bytes drifted")
    return {
        "role": role,
        "path": str(path),
        "size_bytes": size,
        "file_sha256": expected_sha,
    }


def _flag_values(argv: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    index = 3
    while index < len(argv):
        flag = argv[index]
        if not flag.startswith("--") or index + 1 >= len(argv):
            _fail("cache-seed driver argv is not flag/value closed")
        if flag in values:
            _fail("cache-seed driver argv repeats " + flag)
        values[flag] = argv[index + 1]
        index += 2
    return values


def _validate_cache_seed_driver(
    *,
    argv: list[str],
    environment: dict[str, str],
    sources: Mapping[str, Mapping[str, object]],
) -> None:
    if argv[:3] != [
        "aws/glm52-gpu/scripts/submit_sky_campaign.py",
        "cache-seed",
        "acquire-and-launch",
    ]:
        _fail("cache-seed driver entrypoint drifted")
    values = _flag_values(argv)
    expected_flags = {
        "--profile",
        "--descriptor",
        "--approval",
        "--staged-ready",
        "--staged-ready-version-id",
        "--rehearsal-evidence",
        "--task",
        "--config",
        "--sky-bin",
    }
    if set(values) != expected_flags or values["--profile"] != PROFILE:
        _fail("cache-seed driver flag closure drifted")
    if environment != {
        "AWS_PROFILE": PROFILE,
        "AWS_REGION": REGION,
    }:
        _fail("cache-seed driver environment drifted")
    expected_roles = {
        "campaign_descriptor": "--descriptor",
        "approval": "--approval",
        "staged_ready": "--staged-ready",
        "rehearsal_evidence": "--rehearsal-evidence",
        "task": "--task",
        "config": "--config",
    }
    if set(sources) != set(expected_roles):
        _fail("cache-seed driver source roles drifted")
    for role, flag in expected_roles.items():
        if sources[role]["path"] != values[flag]:
            _fail(role + " source is not bound to driver argv")
    _version(
        values["--staged-ready-version-id"],
        "cache-seed staged readiness VersionId",
    )
    sky_bin = Path(values["--sky-bin"])
    if (
        not sky_bin.is_absolute()
        or not sky_bin.is_file()
        or sky_bin.is_symlink()
        or sky_bin.stat().st_mode & 0o111 == 0
    ):
        _fail("cache-seed Sky executable is not concrete")


def _validate_h100_driver(
    *,
    argv: list[str],
    environment: dict[str, str],
    sources: Mapping[str, Mapping[str, object]],
) -> None:
    if argv != [
        "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
    ]:
        _fail("H100 driver entrypoint drifted")
    if (
        set(environment)
        != {"AWS_PROFILE", "AWS_REGION", "CAMPAIGN_DESCRIPTOR"}
        or environment["AWS_PROFILE"] != PROFILE
        or environment["AWS_REGION"] != REGION
        or set(sources) != {"campaign_descriptor"}
        or sources["campaign_descriptor"]["path"]
        != environment["CAMPAIGN_DESCRIPTOR"]
    ):
        _fail("H100 driver source/environment closure drifted")


def parse_driver_materialization_request(
    value: object,
) -> dict[str, object]:
    """Validate caller-supplied concrete local sources for one driver."""

    if type(value) is not dict or set(value) != _REQUEST_FIELDS:
        _fail("driver materialization request schema drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256", None)
    activation_id = value.get("activation_id")
    if (
        value["schema_version"] != 2
        or value["record_type"]
        != "glm52_task13_driver_materialization_request_v2"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["profile"] != PROFILE
        or value["run_id"] != RUN_ID
        or type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
        or identity != canonical_sha256(body)
    ):
        _fail("driver materialization request identity drifted")
    operation_kind = _text(
        value["operation_kind"],
        "driver operation kind",
    )
    if operation_kind not in _DRIVER_CONTRACTS:
        _fail("driver operation kind is not implemented")
    argv = value["argv"]
    environment = value["environment"]
    raw_sources = value["source_coordinates"]
    if (
        type(argv) is not list
        or not argv
        or any(type(item) is not str for item in argv)
        or type(environment) is not dict
        or any(
            type(key) is not str or type(item) is not str
            for key, item in environment.items()
        )
        or type(raw_sources) is not list
        or not raw_sources
    ):
        _fail("driver argv, environment, or sources are malformed")
    exact_argv = [_text(item, "driver argv member") for item in argv]
    exact_environment = {
        _text(key, "driver environment key"): _text(
            item,
            "driver environment value",
        )
        for key, item in environment.items()
    }
    exact_sources = [_source_coordinate(item) for item in raw_sources]
    by_role = {item["role"]: item for item in exact_sources}
    if (
        len(by_role) != len(exact_sources)
        or len({item["path"] for item in exact_sources})
        != len(exact_sources)
    ):
        _fail("driver source coordinates are duplicated")
    validator = (
        _validate_cache_seed_driver
        if operation_kind == "qualification-cache-seed"
        else _validate_h100_driver
    )
    validator(
        argv=exact_argv,
        environment=exact_environment,
        sources=by_role,
    )
    return {
        **body,
        "argv": exact_argv,
        "environment": exact_environment,
        "source_coordinates": exact_sources,
        "canonical_identity_sha256": identity,
    }


def read_driver_materialization_request(
    path: Path,
) -> dict[str, object]:
    return parse_driver_materialization_request(
        _read_canonical_mapping(path, "driver materialization request")
    )


def build_repository_driver_manifest(
    request: object,
) -> dict[str, object]:
    """Build only the exact schema the Task 13 coordinator consumes."""

    exact = parse_driver_materialization_request(request)
    body = {
        "schema_version": 2,
        "record_type": "glm52_task13_repository_driver_v2",
        "activation_id": exact["activation_id"],
        "operation_kind": exact["operation_kind"],
        "argv": exact["argv"],
        "environment": exact["environment"],
    }
    value = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    return validate_repository_driver_manifest(
        value,
        expected_operation_kind=exact["operation_kind"],
        expected_activation_id=exact["activation_id"],
    )


def validate_repository_driver_manifest(
    value: object,
    *,
    expected_operation_kind: str,
    expected_activation_id: str,
) -> dict[str, object]:
    """Validate the exact immutable schema consumed by the coordinator."""

    if type(value) is not dict or set(value) != _DRIVER_FIELDS:
        _fail("repository driver manifest schema drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256", None)
    operation_kind = value.get("operation_kind")
    argv = value.get("argv")
    environment = value.get("environment")
    activation_id = value.get("activation_id")
    if (
        value.get("schema_version") != 2
        or value.get("record_type")
        != "glm52_task13_repository_driver_v2"
        or activation_id != expected_activation_id
        or type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
        or operation_kind != expected_operation_kind
        or operation_kind not in _DRIVER_CONTRACTS
        or identity != canonical_sha256(body)
        or type(argv) is not list
        or not argv
        or any(type(item) is not str for item in argv)
        or type(environment) is not dict
        or any(
            type(key) is not str or type(item) is not str
            for key, item in environment.items()
        )
    ):
        _fail("repository driver manifest identity drifted")
    exact_argv = [_text(item, "driver manifest argv member") for item in argv]
    exact_environment = {
        _text(key, "driver manifest environment key"): _text(
            item,
            "driver manifest environment value",
        )
        for key, item in environment.items()
    }
    if operation_kind == "qualification-cache-seed":
        if exact_argv[:3] != [
            "aws/glm52-gpu/scripts/submit_sky_campaign.py",
            "cache-seed",
            "acquire-and-launch",
        ]:
            _fail("repository cache-seed driver manifest drifted")
        flags = _flag_values(exact_argv)
        if (
            set(flags)
            != {
                "--profile",
                "--descriptor",
                "--approval",
                "--staged-ready",
                "--staged-ready-version-id",
                "--rehearsal-evidence",
                "--task",
                "--config",
                "--sky-bin",
            }
            or flags["--profile"] != PROFILE
            or exact_environment
            != {"AWS_PROFILE": PROFILE, "AWS_REGION": REGION}
        ):
            _fail("repository cache-seed driver closure drifted")
        _version(
            flags["--staged-ready-version-id"],
            "repository cache-seed staged VersionId",
        )
    elif (
        exact_argv
        != ["aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"]
        or set(exact_environment)
        != {"AWS_PROFILE", "AWS_REGION", "CAMPAIGN_DESCRIPTOR"}
        or exact_environment["AWS_PROFILE"] != PROFILE
        or exact_environment["AWS_REGION"] != REGION
        or not Path(exact_environment["CAMPAIGN_DESCRIPTOR"]).is_absolute()
    ):
        _fail("repository H100 driver closure drifted")
    return {
        **body,
        "argv": exact_argv,
        "environment": exact_environment,
        "canonical_identity_sha256": identity,
    }


def repository_archive_manifest_key(activation_id: object) -> str:
    """Return the create-only manifest key for one exact activation."""

    if (
        type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
    ):
        _fail("repository archive activation_id is invalid")
    return (
        f"task13/activations/{activation_id}/archive/repo-tar.json"
    )

def validate_repository_archive_manifest_key(
    key: object,
    *,
    activation_id: Optional[str] = None,
) -> str:
    """Validate one activation-scoped repository manifest key."""

    if type(key) is not str:
        _fail("repository archive manifest key is invalid")
    match = _ARCHIVE_MANIFEST_KEY.fullmatch(key)
    if match is None:
        _fail("repository archive manifest key is invalid")
    if (
        activation_id is not None
        and key != repository_archive_manifest_key(activation_id)
    ):
        _fail("repository archive manifest key activation drifted")
    return key



def validate_repository_archive_manifest(
    value: object,
    *,
    expected_activation_id: str,
    expected_archive_sha256: str,
    expected_archive_size: int,
    expected_bucket: str,
) -> dict[str, object]:
    """Validate one content-addressed archive and its uploaded VersionId."""

    if type(value) is not dict or set(value) != _ARCHIVE_FIELDS:
        _fail("repository archive manifest schema drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256", None)
    archive = value.get("archive")
    expected_sha = _sha(
        expected_archive_sha256,
        "expected repository archive SHA-256",
    )
    if (
        type(expected_archive_size) is not int
        or expected_archive_size < 1
        or expected_bucket != CAMPAIGN_BUCKET
        or value.get("schema_version") != 2
        or value.get("record_type")
        != "glm52_task13_repository_archive_manifest_v2"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or value.get("activation_id") != expected_activation_id
        or repository_archive_manifest_key(value.get("activation_id"))
        != repository_archive_manifest_key(expected_activation_id)
        or value.get("archive_file_sha256") != expected_sha
        or value.get("archive_size_bytes") != expected_archive_size
        or identity != canonical_sha256(body)
        or type(archive) is not dict
        or set(archive)
        != {"bucket", "key", "version_id", "file_sha256"}
        or archive.get("bucket") != expected_bucket
        or archive.get("key")
        != (
            f"campaigns/{RUN_ID}/repository/"
            f"keep-{expected_sha}.tar.gz"
        )
        or archive.get("file_sha256") != expected_sha
    ):
        _fail("repository archive manifest identity drifted")
    _version(archive["version_id"], "repository archive VersionId")
    return json.loads(canonical_json_bytes(value))


def _inventory_versions(
    *,
    s3: object,
    bucket: str,
    key: str,
    reject_foreign_history: bool = False,
) -> tuple[dict[str, object], ...]:
    key_marker: Optional[str] = None
    version_marker: Optional[str] = None
    seen: set[tuple[str, str]] = set()
    exact: list[dict[str, object]] = []
    for _page in range(1024):
        request: dict[str, object] = {
            "Bucket": bucket,
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
            _fail("fixed-key S3 version inventory is malformed")
        if reject_foreign_history and any(
            item.get("Key") != key for item in versions + delete_markers
        ):
            _fail("fixed-key S3 history contains a foreign key")
        if any(item.get("Key") == key for item in delete_markers):
            _fail("fixed-key S3 history contains a delete marker")
        exact.extend(dict(item) for item in versions if item.get("Key") == key)
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
            _fail("fixed-key S3 version pagination is incomplete")
        seen.add((next_key, next_version))
        key_marker = next_key
        version_marker = next_version
    _fail("fixed-key S3 version pagination exceeded its bound")


def _read_exact_version(
    *,
    s3: object,
    bucket: str,
    key: str,
    version_id: str,
    raw: bytes,
    metadata: Mapping[str, str],
    content_type: Optional[str] = None,
    sse_kms_key_id: Optional[str] = None,
    verify_empty_tags: bool = False,
) -> None:
    response = _call(
        s3,
        "get_object",
        operation="GetObject",
        request={
            "Bucket": bucket,
            "Key": key,
            "VersionId": version_id,
            "ExpectedBucketOwner": ACCOUNT_ID,
            "ChecksumMode": "ENABLED",
        },
    )
    body = response.get("Body")
    observed = body.read() if hasattr(body, "read") else None
    if (
        type(observed) is not bytes
        or observed != raw
        or response.get("ContentLength") != len(raw)
        or response.get("VersionId") != version_id
        or response.get("ChecksumSHA256") != _checksum(raw)
        or response.get("Metadata") != dict(metadata)
        or (
            content_type is not None
            and response.get("ContentType") != content_type
        )
        or (
            sse_kms_key_id is not None
            and (
                response.get("ServerSideEncryption") != "aws:kms"
                or response.get("SSEKMSKeyId") != sse_kms_key_id
                or response.get("ObjectLockMode") is not None
                or response.get("ObjectLockRetainUntilDate") is not None
                or response.get("ObjectLockLegalHoldStatus") is not None
            )
        )
    ):
        _fail("fixed-key S3 VersionId readback drifted")
    if verify_empty_tags:
        tagging = _call(
            s3,
            "get_object_tagging",
            operation="GetObjectTagging",
            request={
                "Bucket": bucket,
                "Key": key,
                "VersionId": version_id,
                "ExpectedBucketOwner": ACCOUNT_ID,
            },
        )
        if tagging.get("TagSet") != []:
            _fail("fixed-key S3 VersionId tags are not empty")


def _adopt_exact(
    *,
    s3: object,
    bucket: str,
    key: str,
    raw: bytes,
    metadata: Mapping[str, str],
    content_type: Optional[str] = None,
    sse_kms_key_id: Optional[str] = None,
    reject_foreign_history: bool = False,
    verify_empty_tags: bool = False,
) -> Optional[str]:
    versions = _inventory_versions(
        s3=s3,
        bucket=bucket,
        key=key,
        reject_foreign_history=reject_foreign_history,
    )
    if not versions:
        return None
    if len(versions) != 1:
        _fail("fixed-key S3 positive truth is not singular")
    version_id = _version(
        versions[0].get("VersionId"),
        "fixed-key S3 VersionId",
    )
    if versions[0].get("Size") != len(raw):
        _fail("fixed-key S3 version size is foreign")
    _read_exact_version(
        s3=s3,
        bucket=bucket,
        key=key,
        version_id=version_id,
        raw=raw,
        metadata=metadata,
        content_type=content_type,
        sse_kms_key_id=sse_kms_key_id,
        verify_empty_tags=verify_empty_tags,
    )
    return version_id


def _publish_bytes_with_outcome(
    *,
    s3: object,
    publisher_s3: object | None = None,
    bucket: str,
    key: str,
    raw: bytes,
    record_type: str,
    content_type: str = "application/json",
    metadata_override: Optional[Mapping[str, str]] = None,
    sse_kms_key_id: Optional[str] = None,
    reject_foreign_history: bool = False,
    verify_empty_tags: bool = False,
) -> FixedKeyPublicationOutcome:
    metadata = (
        {
            "glm52-run-id": RUN_ID,
            "record-type": record_type,
            "file-sha256": _sha256(raw),
        }
        if metadata_override is None
        else dict(metadata_override)
    )
    readback_content_type = (
        content_type if sse_kms_key_id is not None else None
    )
    existing = _adopt_exact(
        s3=s3,
        bucket=bucket,
        key=key,
        raw=raw,
        metadata=metadata,
        content_type=readback_content_type,
        sse_kms_key_id=sse_kms_key_id,
        reject_foreign_history=reject_foreign_history,
        verify_empty_tags=verify_empty_tags,
    )
    if existing is not None:
        return FixedKeyPublicationOutcome(
            version_id=existing,
            disposition=FixedKeyPublicationDisposition.ADOPTED,
        )
    request = {
        "Bucket": bucket,
        "Key": key,
        "Body": raw,
        "ContentType": content_type,
        "ChecksumAlgorithm": "SHA256",
        "ChecksumSHA256": _checksum(raw),
        "IfNoneMatch": "*",
        "ExpectedBucketOwner": ACCOUNT_ID,
        "Metadata": metadata,
    }
    if sse_kms_key_id is not None:
        request["ServerSideEncryption"] = "aws:kms"
        request["SSEKMSKeyId"] = sse_kms_key_id
    writer = s3 if publisher_s3 is None else publisher_s3
    try:
        response = _call(
            writer,
            "put_object",
            operation="PutObject",
            request=request,
        )
    except Exception as original_error:
        try:
            adopted = _adopt_exact(
                s3=s3,
                bucket=bucket,
                key=key,
                raw=raw,
                metadata=metadata,
                content_type=readback_content_type,
                sse_kms_key_id=sse_kms_key_id,
                reject_foreign_history=reject_foreign_history,
                verify_empty_tags=verify_empty_tags,
            )
        except Exception as readback_error:
            raise Task13FixedArtifactError(
                "ambiguous fixed-key PutObject has no exact adoption: "
                + str(readback_error)
            ) from original_error
        if adopted is None:
            raise Task13FixedArtifactError(
                "ambiguous fixed-key PutObject has no durable version"
            ) from original_error
        return FixedKeyPublicationOutcome(
            version_id=adopted,
            disposition=FixedKeyPublicationDisposition.WRITTEN,
        )
    version_id = _version(
        response.get("VersionId"),
        "PutObject VersionId",
    )
    if response.get("ChecksumSHA256") != _checksum(raw):
        _fail("PutObject checksum response drifted")
    if sse_kms_key_id is not None and (
        response.get("ServerSideEncryption") != "aws:kms"
        or response.get("SSEKMSKeyId") != sse_kms_key_id
    ):
        _fail("PutObject KMS encryption response drifted")
    adopted = _adopt_exact(
        s3=s3,
        bucket=bucket,
        key=key,
        raw=raw,
        metadata=metadata,
        content_type=readback_content_type,
        sse_kms_key_id=sse_kms_key_id,
        reject_foreign_history=reject_foreign_history,
        verify_empty_tags=verify_empty_tags,
    )
    if adopted != version_id:
        _fail("PutObject VersionId is not the singular durable version")
    return FixedKeyPublicationOutcome(
        version_id=version_id,
        disposition=FixedKeyPublicationDisposition.WRITTEN,
    )


def _publish_bytes(
    *,
    s3: object,
    publisher_s3: object | None = None,
    bucket: str,
    key: str,
    raw: bytes,
    record_type: str,
    content_type: str = "application/json",
    metadata_override: Optional[Mapping[str, str]] = None,
    sse_kms_key_id: Optional[str] = None,
    reject_foreign_history: bool = False,
    verify_empty_tags: bool = False,
) -> str:
    return _publish_bytes_with_outcome(
        s3=s3,
        bucket=bucket,
        key=key,
        raw=raw,
        record_type=record_type,
        content_type=content_type,
        metadata_override=metadata_override,
        sse_kms_key_id=sse_kms_key_id,
        reject_foreign_history=reject_foreign_history,
        verify_empty_tags=verify_empty_tags,
        publisher_s3=publisher_s3,
    ).version_id


def _validate_fixed_key_publication_input(
    *,
    raw: bytes,
    key: str,
    record_type: str,
    sse_kms_key_id: str,
) -> None:
    if (
        type(raw) is not bytes
        or not raw
        or type(key) is not str
        or not key.startswith("campaigns/" + RUN_ID + "/")
        or type(record_type) is not str
        or not record_type
        or type(sse_kms_key_id) is not str
        or not sse_kms_key_id.startswith(
            "arn:aws:kms:" + REGION + ":" + ACCOUNT_ID + ":key/"
        )
    ):
        _fail("fixed-key publication input is not exact")


def publish_fixed_key_bytes_with_outcome(
    *,
    services: Task13FixedArtifactServices,
    bucket: str,
    key: str,
    raw: bytes,
    record_type: str,
    sse_kms_key_id: str,
) -> FixedKeyPublicationOutcome:
    """Create or adopt one exact version and report which path converged."""

    _validate_fixed_key_publication_input(
        raw=raw,
        key=key,
        record_type=record_type,
        sse_kms_key_id=sse_kms_key_id,
    )
    _guard_services(services, bucket)
    return _publish_bytes_with_outcome(
        s3=services.s3,
        bucket=bucket,
        key=key,
        raw=raw,
        record_type=record_type,
        metadata_override={},
        sse_kms_key_id=sse_kms_key_id,
        reject_foreign_history=True,
        verify_empty_tags=True,
        publisher_s3=services.publisher_s3,
    )


def adopt_fixed_key_bytes(
    *,
    services: Task13FixedArtifactServices,
    bucket: str,
    key: str,
    raw: bytes,
    record_type: str,
    sse_kms_key_id: str,
) -> FixedKeyPublicationOutcome:
    """Read and exactly adopt one existing fixed-key version without mutation."""

    _validate_fixed_key_publication_input(
        raw=raw,
        key=key,
        record_type=record_type,
        sse_kms_key_id=sse_kms_key_id,
    )
    _guard_services(services, bucket)
    version_id = _adopt_exact(
        s3=services.s3,
        bucket=bucket,
        key=key,
        raw=raw,
        metadata={},
        content_type="application/json",
        sse_kms_key_id=sse_kms_key_id,
        reject_foreign_history=True,
        verify_empty_tags=True,
    )
    if version_id is None:
        _fail("fixed-key object has no durable version to adopt")
    return FixedKeyPublicationOutcome(
        version_id=version_id,
        disposition=FixedKeyPublicationDisposition.ADOPTED,
    )


def publish_fixed_key_bytes(
    *,
    services: Task13FixedArtifactServices,
    bucket: str,
    key: str,
    raw: bytes,
    record_type: str,
    sse_kms_key_id: str,
) -> str:
    return publish_fixed_key_bytes_with_outcome(
        services=services,
        bucket=bucket,
        key=key,
        raw=raw,
        record_type=record_type,
        sse_kms_key_id=sse_kms_key_id,
    ).version_id


def _coordinate(
    *,
    artifact_kind: str,
    bucket: str,
    key: str,
    version_id: str,
    raw: bytes,
    body_sha256: str,
) -> dict[str, object]:
    return {
        "artifact_kind": artifact_kind,
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "file_sha256": _sha256(raw),
        "body_sha256": _sha(body_sha256, "artifact body SHA-256"),
    }


def validate_repository_archive_payload(path: Path) -> tuple[bytes, str]:
    """Validate and return one safe local repository-tar payload."""
    path = Path(path)
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.name.endswith(".tar.gz")
    ):
        _fail("repository archive must be one regular .tar.gz file")
    raw = path.read_bytes()
    if not raw:
        _fail("repository archive is empty")
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            members = archive.getmembers()
    except (tarfile.TarError, OSError) as exc:
        raise Task13FixedArtifactError(
            "repository archive is not readable gzip tar"
        ) from exc
    if not members:
        _fail("repository archive contains no members")
    for member in members:
        name = PurePosixPath(member.name)
        link = PurePosixPath(member.linkname)
        if (
            member.name.startswith("/")
            or ".." in name.parts
            or member.isdev()
            or (
                (member.issym() or member.islnk())
                and (
                    member.linkname.startswith("/")
                    or ".." in link.parts
                )
            )
        ):
            _fail("repository archive contains an unsafe member")
    return raw, _sha256(raw)


def publish_repository_archive(
    *,
    activation_id: str,
    archive_path: Path,
    bucket: str,
    services: Task13FixedArtifactServices,
) -> dict[str, object]:
    """Publish one archive version, then its activation-scoped manifest."""

    _guard_services(services, bucket)
    manifest_key = repository_archive_manifest_key(activation_id)
    archive_raw, archive_sha = validate_repository_archive_payload(archive_path)
    archive_key = (
        f"campaigns/{RUN_ID}/repository/keep-{archive_sha}.tar.gz"
    )
    archive_version = _publish_bytes(
        s3=services.s3,
        bucket=bucket,
        key=archive_key,
        raw=archive_raw,
        record_type="glm52_task13_repository_archive_payload_v1",
        content_type="application/gzip",
    )
    body = {
        "schema_version": 2,
        "record_type": "glm52_task13_repository_archive_manifest_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "archive_file_sha256": archive_sha,
        "archive_size_bytes": len(archive_raw),
        "archive": {
            "bucket": bucket,
            "key": archive_key,
            "version_id": archive_version,
            "file_sha256": archive_sha,
        },
    }
    manifest = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    manifest = validate_repository_archive_manifest(
        manifest,
        expected_activation_id=activation_id,
        expected_archive_sha256=archive_sha,
        expected_archive_size=len(archive_raw),
        expected_bucket=bucket,
    )
    raw = canonical_json_bytes(manifest) + b"\n"
    version_id = _publish_bytes(
        s3=services.s3,
        bucket=bucket,
        key=manifest_key,
        raw=raw,
        record_type=manifest["record_type"],
    )
    return _coordinate(
        artifact_kind="REPOSITORY_ARCHIVE",
        bucket=bucket,
        key=manifest_key,
        version_id=version_id,
        raw=raw,
        body_sha256=manifest["canonical_identity_sha256"],
    )


def publish_repository_driver(
    *,
    request: object,
    bucket: str,
    services: Task13FixedArtifactServices,
) -> dict[str, object]:
    """Publish one coordinator-exact cache-seed or H100 driver manifest."""

    _guard_services(services, bucket)
    manifest = build_repository_driver_manifest(request)
    operation_kind = manifest["operation_kind"]
    artifact_kind, _executable = _DRIVER_CONTRACTS[operation_kind]
    key = activation_artifact_key(
        artifact_kind,
        manifest["activation_id"],
    )
    raw = canonical_json_bytes(manifest) + b"\n"
    version_id = _publish_bytes(
        s3=services.s3,
        bucket=bucket,
        key=key,
        raw=raw,
        record_type=manifest["record_type"],
    )
    return _coordinate(
        artifact_kind=artifact_kind,
        bucket=bucket,
        key=key,
        version_id=version_id,
        raw=raw,
        body_sha256=manifest["canonical_identity_sha256"],
    )


def publish_support_build_inputs(
    *,
    inputs_path: Path,
    bucket: str,
    services: Task13FixedArtifactServices,
) -> dict[str, object]:
    """Validate and publish the sole exact pre-create SupportBuildInputs."""

    value = _read_canonical_mapping(
        inputs_path,
        "support build inputs",
    )
    from .support_plane import (
        support_build_inputs_from_mapping,
        support_build_inputs_projection,
    )

    try:
        inputs = support_build_inputs_from_mapping(value)
        projection = support_build_inputs_projection(inputs)
    except (TypeError, ValueError) as exc:
        raise Task13FixedArtifactError(
            "support build inputs failed exact parser"
        ) from exc
    raw = Path(inputs_path).read_bytes()
    if projection != value or raw != canonical_json_bytes(projection) + b"\n":
        _fail("support build input projection drifted")
    _guard_services(services, bucket)
    version_id = _publish_bytes(
        s3=services.s3,
        bucket=bucket,
        key=_SUPPORT_INPUTS_FIXED_KEY,
        raw=raw,
        record_type="glm52_h1g_support_build_inputs_v1",
    )
    return _coordinate(
        artifact_kind="SUPPORT_INPUTS",
        bucket=bucket,
        key=_SUPPORT_INPUTS_FIXED_KEY,
        version_id=version_id,
        raw=raw,
        body_sha256=_sha256(raw[:-1]),
    )


def write_coordinate_once(
    path: Path,
    coordinate: Mapping[str, object],
) -> None:
    """Write one exact reviewed-artifact coordinate without replacement."""

    path = Path(path)
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.exists()
        or path.is_symlink()
    ):
        _fail("coordinate output path is not one new absolute file")
    raw = canonical_json_bytes(coordinate) + b"\n"
    descriptor_number = os.open(
        path,
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
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    descriptor.close()


__all__ = [
    "ACCOUNT_ID",
    "CAMPAIGN_BUCKET",
    "PROFILE",
    "REGION",
    "RUN_ID",
    "FixedKeyPublicationDisposition",
    "FixedKeyPublicationOutcome",
    "Task13FixedArtifactError",
    "Task13FixedArtifactServices",
    "adopt_fixed_key_bytes",
    "activation_artifact_key",
    "build_repository_driver_manifest",
    "parse_driver_materialization_request",
    "publish_fixed_key_bytes_with_outcome",
    "publish_fixed_key_bytes",
    "publish_repository_archive",
    "publish_repository_driver",
    "publish_support_build_inputs",
    "read_driver_materialization_request",
    "repository_archive_manifest_key",
    "validate_repository_archive_manifest",
    "validate_repository_archive_manifest_key",
    "validate_repository_archive_payload",
    "validate_activation_artifact_key",
    "validate_repository_driver_manifest",
    "write_coordinate_once",
]
