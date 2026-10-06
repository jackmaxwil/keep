"""Read-only production-fence S3 audit tests.

The simulator models the AWS response boundary closely enough to exercise
opaque two-marker pagination, complete version/delete-marker history, and
exact-version GET plus HEAD.  It never contacts AWS.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "aws/glm52-gpu/scripts/glm52_production_fence_audit.py"
OWNER = "246813579024"
BUCKET = "keep-glm52-us-west-2"
RUN_ID = "glm52-sky-20260724"
KEY = f"campaigns/{RUN_ID}/authorities/fence/FENCE_GENESIS.json"
NOW = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
RAW = b'{"record_type":"test"}\n'
FILE_SHA = hashlib.sha256(RAW).hexdigest()
CHECKSUM = base64.b64encode(hashlib.sha256(RAW).digest()).decode("ascii")
METADATA = (
    ("glm52-body-sha256", "b" * 64),
    ("glm52-campaign-identity-sha256", "c" * 64),
    ("glm52-file-sha256", FILE_SHA),
    ("glm52-record-type", "test"),
    ("glm52-run-id", RUN_ID),
)


class ProcessDeath(BaseException):
    """Injected fatal process loss; ordinary translation must not absorb it."""


class OneShotBody:
    def __init__(
        self,
        raw: bytes,
        *,
        read_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._raw = raw
        self._read_error = read_error
        self._close_error = close_error
        self.read_count = 0
        self.close_count = 0

    def read(self) -> bytes:
        self.read_count += 1
        if self._read_error is not None:
            raise self._read_error
        if self.read_count != 1:
            raise AssertionError("body must be read exactly once")
        return self._raw

    def close(self) -> None:
        self.close_count += 1
        if self._close_error is not None:
            raise self._close_error


@dataclass
class StoredVersion:
    key: str
    version_id: str
    raw: bytes
    is_latest: bool = True
    etag: str = '"0123456789abcdef0123456789abcdef"'
    last_modified: datetime = NOW
    content_type: str = "application/json"
    metadata: tuple[tuple[str, str], ...] = METADATA
    missing_meta: int = 0


@dataclass
class StoredDeleteMarker:
    key: str
    version_id: str
    is_latest: bool = True
    last_modified: datetime = NOW


class VersionedS3Simulator:
    """AWS-shaped read-only S3 simulator with deterministic fault injection."""

    def __init__(self) -> None:
        self.versions: list[StoredVersion] = []
        self.delete_markers: list[StoredDeleteMarker] = []
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.expected_bucket = BUCKET
        self.page_size = 1000
        self.list_mode: str | None = None
        self.list_fault: BaseException | None = None
        self.get_fault: BaseException | None = None
        self.head_fault: BaseException | None = None
        self.get_changes: dict[str, object] = {}
        self.head_changes: dict[str, object] = {}
        self.get_omits: set[str] = set()
        self.head_omits: set[str] = set()
        self.body_read_error: BaseException | None = None
        self.body_close_error: BaseException | None = None
        self.bodies: list[OneShotBody] = []
        self.hidden_on_first_list: set[tuple[str, str]] = set()
        self._markers: dict[tuple[str, str, str], int] = {}
        self._cycle_start: tuple[str, str] | None = None
        self._list_count = 0

    def add_version(
        self,
        *,
        key: str = KEY,
        version_id: str = "version-one",
        raw: bytes = RAW,
        is_latest: bool = True,
        etag: str = '"0123456789abcdef0123456789abcdef"',
        last_modified: datetime = NOW,
        content_type: str = "application/json",
        metadata: tuple[tuple[str, str], ...] = METADATA,
        missing_meta: int = 0,
    ) -> StoredVersion:
        stored = StoredVersion(
            key=key,
            version_id=version_id,
            raw=raw,
            is_latest=is_latest,
            etag=etag,
            last_modified=last_modified,
            content_type=content_type,
            metadata=metadata,
            missing_meta=missing_meta,
        )
        self.versions.append(stored)
        return stored

    def add_delete_marker(
        self,
        *,
        key: str = KEY,
        version_id: str = "delete-one",
        is_latest: bool = True,
        last_modified: datetime = NOW,
    ) -> StoredDeleteMarker:
        stored = StoredDeleteMarker(
            key=key,
            version_id=version_id,
            is_latest=is_latest,
            last_modified=last_modified,
        )
        self.delete_markers.append(stored)
        return stored

    @staticmethod
    def _snapshot(kwargs: dict[str, object]) -> dict[str, object]:
        return deepcopy(kwargs)

    def _record(self, operation: str, kwargs: dict[str, object]) -> None:
        self.calls.append((operation, self._snapshot(kwargs)))

    def _require_common(self, kwargs: dict[str, object]) -> None:
        assert kwargs["Bucket"] == self.expected_bucket
        assert kwargs["ExpectedBucketOwner"] == OWNER

    def list_object_versions(self, **kwargs: object) -> dict[str, object]:
        arguments = dict(kwargs)
        self._record("list_object_versions", arguments)
        self._require_common(arguments)
        self._list_count += 1
        if self.list_fault is not None:
            raise self.list_fault
        prefix = arguments["Prefix"]
        assert isinstance(prefix, str)
        request_key = arguments.get("KeyMarker")
        request_version = arguments.get("VersionIdMarker")
        if (request_key is None) != (request_version is None):
            raise AssertionError("request markers must be paired")
        if request_key is None:
            offset = 0
        else:
            assert isinstance(request_key, str)
            assert isinstance(request_version, str)
            offset = self._markers[(prefix, request_key, request_version)]

        rows: list[tuple[str, StoredVersion | StoredDeleteMarker]] = [
            ("version", item)
            for item in self.versions
            if item.key.startswith(prefix)
            and (
                self._list_count != 1
                or (item.key, item.version_id)
                not in self.hidden_on_first_list
            )
        ]
        rows.extend(
            ("delete", item)
            for item in self.delete_markers
            if item.key.startswith(prefix)
        )
        page = rows[offset : offset + self.page_size]
        next_offset = offset + len(page)
        truncated = next_offset < len(rows)
        response: dict[str, object] = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "Name": self.expected_bucket,
            "Prefix": prefix,
            "KeyMarker": "" if request_key is None else request_key,
            "VersionIdMarker": (
                "" if request_version is None else request_version
            ),
            "IsTruncated": truncated,
            "Versions": [
                {
                    "Key": item.key,
                    "VersionId": item.version_id,
                    "IsLatest": item.is_latest,
                    "ETag": item.etag,
                    "Size": len(item.raw),
                    "LastModified": item.last_modified,
                    "StorageClass": "STANDARD",
                }
                for kind, item in page
                if kind == "version" and isinstance(item, StoredVersion)
            ],
            "DeleteMarkers": [
                {
                    "Key": item.key,
                    "VersionId": item.version_id,
                    "IsLatest": item.is_latest,
                    "LastModified": item.last_modified,
                }
                for kind, item in page
                if kind == "delete" and isinstance(item, StoredDeleteMarker)
            ],
            "CommonPrefixes": [],
        }
        if truncated:
            next_key = f"{prefix}page-{next_offset}"
            next_version = f"marker-{next_offset}"
            self._markers[(prefix, next_key, next_version)] = next_offset
            response["NextKeyMarker"] = next_key
            response["NextVersionIdMarker"] = next_version
        if self.list_mode == "omit-empty-collections":
            if not response["Versions"]:
                response.pop("Versions")
            if not response["DeleteMarkers"]:
                response.pop("DeleteMarkers")
        elif self.list_mode == "partial-next" and truncated:
            response.pop("NextVersionIdMarker")
        elif self.list_mode == "missing-next" and truncated:
            response.pop("NextKeyMarker")
            response.pop("NextVersionIdMarker")
        elif self.list_mode == "repeated-next" and truncated:
            if request_key is None:
                response["NextKeyMarker"] = f"{prefix}page-repeat"
                response["NextVersionIdMarker"] = "marker-repeat"
                self._markers[
                    (prefix, f"{prefix}page-repeat", "marker-repeat")
                ] = next_offset
            else:
                response["NextKeyMarker"] = request_key
                response["NextVersionIdMarker"] = request_version
        elif self.list_mode == "cyclic-next" and truncated:
            if request_key is None:
                self._cycle_start = (
                    str(response["NextKeyMarker"]),
                    str(response["NextVersionIdMarker"]),
                )
            elif self._list_count >= 3:
                assert self._cycle_start is not None
                response["NextKeyMarker"] = self._cycle_start[0]
                response["NextVersionIdMarker"] = self._cycle_start[1]
        elif self.list_mode == "wrong-echo" and request_key is not None:
            response["KeyMarker"] = f"{prefix}different"
        elif self.list_mode == "foreign-next" and truncated:
            response["NextKeyMarker"] = "foreign/prefix"
        elif self.list_mode == "terminal-leftover" and not truncated:
            response["NextKeyMarker"] = f"{prefix}terminal"
            response["NextVersionIdMarker"] = "terminal-marker"
        elif self.list_mode == "nonlist-versions":
            response["Versions"] = {}
        elif self.list_mode == "common-prefix":
            response["CommonPrefixes"] = [{"Prefix": f"{prefix}foreign/"}]
        elif self.list_mode == "float-status":
            response["ResponseMetadata"] = {"HTTPStatusCode": 200.0}
        elif self.list_mode == "nonmapping":
            return []  # type: ignore[return-value]
        elif self.list_mode == "name-drift":
            response["Name"] = "foreign-bucket"
        elif self.list_mode == "prefix-drift":
            response["Prefix"] = f"{prefix}foreign"
        elif self.list_mode == "foreign-row" and response["Versions"]:
            foreign = deepcopy(response["Versions"][0])  # type: ignore[index]
            foreign["Key"] = "foreign/prefix/object.json"
            response["Versions"].append(foreign)  # type: ignore[union-attr]
        elif self.list_mode == "duplicate-row" and response["Versions"]:
            duplicate = deepcopy(response["Versions"][0])  # type: ignore[index]
            response["Versions"].append(duplicate)  # type: ignore[union-attr]
        return response

    def _stored(self, key: str, version_id: str) -> StoredVersion:
        matches = [
            item
            for item in self.versions
            if item.key == key and item.version_id == version_id
        ]
        if len(matches) != 1:
            raise AssertionError("exact version is absent or ambiguous")
        return matches[0]

    @staticmethod
    def _transport(stored: StoredVersion) -> dict[str, object]:
        checksum = base64.b64encode(
            hashlib.sha256(stored.raw).digest()
        ).decode("ascii")
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "VersionId": stored.version_id,
            "ETag": stored.etag,
            "ContentLength": len(stored.raw),
            "LastModified": stored.last_modified,
            "ChecksumSHA256": checksum,
            "ChecksumType": "FULL_OBJECT",
            "ContentType": stored.content_type,
            "Metadata": dict(stored.metadata),
            "MissingMeta": stored.missing_meta,
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        arguments = dict(kwargs)
        self._record("get_object", arguments)
        self._require_common(arguments)
        assert arguments["ChecksumMode"] == "ENABLED"
        if self.get_fault is not None:
            raise self.get_fault
        stored = self._stored(
            str(arguments["Key"]),
            str(arguments["VersionId"]),
        )
        body = OneShotBody(
            stored.raw,
            read_error=self.body_read_error,
            close_error=self.body_close_error,
        )
        self.bodies.append(body)
        response = {**self._transport(stored), "Body": body}
        response.update(self.get_changes)
        for field in self.get_omits:
            response.pop(field, None)
        return response

    def head_object(self, **kwargs: object) -> dict[str, object]:
        arguments = dict(kwargs)
        self._record("head_object", arguments)
        self._require_common(arguments)
        assert arguments["ChecksumMode"] == "ENABLED"
        if self.head_fault is not None:
            raise self.head_fault
        stored = self._stored(
            str(arguments["Key"]),
            str(arguments["VersionId"]),
        )
        response = self._transport(stored)
        response.update(self.head_changes)
        for field in self.head_omits:
            response.pop(field, None)
        return response

    def list_objects_v2(self, **_kwargs: object) -> object:
        raise AssertionError("ListObjectsV2 must never be called")

    def put_object(self, **_kwargs: object) -> object:
        raise AssertionError("write method must never be called")

    def delete_object(self, **_kwargs: object) -> object:
        raise AssertionError("delete method must never be called")

    def delete_objects(self, **_kwargs: object) -> object:
        raise AssertionError("batch delete method must never be called")

    def copy_object(self, **_kwargs: object) -> object:
        raise AssertionError("copy method must never be called")

    def create_multipart_upload(self, **_kwargs: object) -> object:
        raise AssertionError("multipart method must never be called")

    def complete_multipart_upload(self, **_kwargs: object) -> object:
        raise AssertionError("multipart method must never be called")

    def abort_multipart_upload(self, **_kwargs: object) -> object:
        raise AssertionError("multipart abort must never be called")

    def upload_part(self, **_kwargs: object) -> object:
        raise AssertionError("multipart upload must never be called")

    def upload_part_copy(self, **_kwargs: object) -> object:
        raise AssertionError("multipart copy must never be called")


def _module() -> Any:
    name = "_glm52_production_fence_audit_test"
    sys.modules.pop(name, None)
    specification = importlib.util.spec_from_file_location(name, AUDIT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _services(module: Any, s3: object) -> object:
    return module.ProductionFenceAuditServices(s3=s3)


def _audit(module: Any, s3: VersionedS3Simulator, **changes: object) -> object:
    kwargs: dict[str, object] = {
        "services": _services(module, s3),
        "bucket": BUCKET,
        "key": KEY,
        "expected_bucket_owner": OWNER,
    }
    kwargs.update(changes)
    return module.audit_exact_coordinate(**kwargs)


@pytest.fixture(scope="module")
def modeled_controls(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    name = "_h1f_audit_model_fixture"
    sys.modules.pop(name, None)
    specification = importlib.util.spec_from_file_location(
        name,
        ROOT / "tests/test_glm52_sky_production_fence.py",
    )
    assert specification is not None and specification.loader is not None
    model_tests = importlib.util.module_from_spec(specification)
    sys.modules[name] = model_tests
    specification.loader.exec_module(model_tests)
    chain = model_tests.authority_chain.__wrapped__(tmp_path_factory)
    genesis_record = model_tests._build_genesis(chain)
    genesis = model_tests._control_artifact(
        genesis_record,
        version_id="genesis-version-1",
    )
    successor_record = model_tests._build_successor(
        chain,
        (genesis,),
        added_reservations=(model_tests._reservation(2),),
    )
    successor = model_tests._control_artifact(
        successor_record,
        version_id="successor-version-1",
    )
    fork_record = model_tests._build_successor(
        chain,
        (genesis,),
        added_sources=(
            model_tests._added_source(label="audit-fork-branch"),
        ),
        label="audit-fork-branch",
    )
    fork = model_tests._control_artifact(
        fork_record,
        version_id="fork-version-1",
    )
    return {
        "chain": chain,
        "source_kwargs": model_tests._source_kwargs(chain),
        "genesis": genesis,
        "successor": successor,
        "fork": fork,
        "bucket": json.loads(genesis.raw)["bucket"],
    }


def _add_modeled_control(
    s3: VersionedS3Simulator,
    artifact: object,
    *,
    version_id: str | None = None,
    is_latest: bool = True,
) -> StoredVersion:
    last_modified = datetime.strptime(
        artifact.last_modified,
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    return s3.add_version(
        key=artifact.key,
        version_id=version_id or artifact.version_id,
        raw=artifact.raw,
        is_latest=is_latest,
        etag=artifact.etag,
        last_modified=last_modified,
        content_type=artifact.content_type,
        metadata=artifact.metadata,
        missing_meta=artifact.missing_meta,
    )


def _namespace(
    module: Any,
    s3: VersionedS3Simulator,
    modeled_controls: dict[str, object],
    **changes: object,
) -> object:
    source_kwargs = modeled_controls["source_kwargs"]
    assert isinstance(source_kwargs, dict)
    namespace_bucket = modeled_controls["bucket"]
    assert isinstance(namespace_bucket, str)
    s3.expected_bucket = namespace_bucket
    kwargs: dict[str, object] = {
        "services": _services(module, s3),
        "bucket": namespace_bucket,
        "run_id": RUN_ID,
        "expected_bucket_owner": OWNER,
        **source_kwargs,
    }
    kwargs.update(changes)
    return module.audit_fence_namespace(**kwargs)


def test_exact_public_surface_and_frozen_evidence_types() -> None:
    module = _module()
    assert module.__all__ == [
        "AuditedS3Object",
        "ExactCoordinateAudit",
        "FenceNamespaceAudit",
        "ProductionFenceAuditError",
        "ProductionFenceAuditServices",
        "S3VersionRow",
        "audit_exact_coordinate",
        "audit_fence_namespace",
    ]
    assert str(inspect.signature(module.audit_exact_coordinate)) == (
        "(*, services: 'ProductionFenceAuditServices', bucket: 'str', key: 'str', "
        "expected_bucket_owner: \"Literal['246813579024']\", "
        "expected_raw: 'Optional[bytes]' = None, "
        "expected_file_sha256: 'Optional[str]' = None, "
        "expected_content_type: 'Optional[str]' = None, "
        "expected_metadata: 'Optional[tuple[tuple[str, str], ...]]' = None) "
        "-> 'ExactCoordinateAudit'"
    )
    assert str(inspect.signature(module.audit_fence_namespace)) == (
        "(*, services: 'ProductionFenceAuditServices', bucket: 'str', "
        "run_id: 'str', expected_bucket_owner: \"Literal['246813579024']\", "
        "descriptor: 'VersionedJsonArtifact', intent: 'VersionedJsonArtifact', "
        "approval: 'VersionedJsonArtifact', "
        "controller_baseline: 'VersionedJsonArtifact', "
        "must_start_control_plane_ready: 'VersionedJsonArtifact', "
        "submission_acquisition: 'VersionedJsonArtifact', "
        "gpu_spend_snapshot: 'VersionedJsonArtifact') -> 'FenceNamespaceAudit'"
    )
    assert module.ProductionFenceAuditError.__bases__ == (ValueError,)
    assert [item.name for item in fields(module.S3VersionRow)] == [
        "key",
        "version_id",
        "is_latest",
        "is_delete_marker",
        "etag",
        "size",
        "last_modified",
    ]
    assert [item.name for item in fields(module.AuditedS3Object)] == [
        "bucket",
        "key",
        "version_id",
        "raw",
        "file_sha256",
        "content_length",
        "etag",
        "last_modified",
        "checksum_sha256_base64",
        "checksum_type",
        "content_type",
        "metadata",
        "missing_meta",
    ]
    assert [item.name for item in fields(module.ExactCoordinateAudit)] == [
        "bucket",
        "key",
        "state",
        "rows",
        "object",
    ]
    assert [item.name for item in fields(module.FenceNamespaceAudit)] == [
        "bucket",
        "run_id",
        "prefix",
        "rows",
        "modeled_head",
        "observed_zero_coordinate",
        "observation",
    ]
    assert [item.name for item in fields(module.ProductionFenceAuditServices)] == [
        "s3"
    ]
    instance = module.ProductionFenceAuditServices(s3=object())
    with pytest.raises(FrozenInstanceError):
        instance.s3 = object()
    forbidden_fragments = {
        "active",
        "authority",
        "authorized",
        "capability",
        "barrier",
        "lock",
        "receipt",
        "submit",
        "write",
        "consume",
    }
    for dataclass_type in (
        module.S3VersionRow,
        module.AuditedS3Object,
        module.ExactCoordinateAudit,
        module.FenceNamespaceAudit,
        module.ProductionFenceAuditServices,
    ):
        assert dataclass_type.__dataclass_params__.frozen
        assert all(
            not any(fragment in item.name for fragment in forbidden_fragments)
            for item in fields(dataclass_type)
        )


def test_zero_coordinate_uses_complete_version_listing_with_exact_owner() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    result = _audit(module, s3)
    assert result == module.ExactCoordinateAudit(
        bucket=BUCKET,
        key=KEY,
        state="zero",
        rows=(),
        object=None,
    )
    assert s3.calls == [
        (
            "list_object_versions",
            {
                "Bucket": BUCKET,
                "Prefix": KEY,
                "ExpectedBucketOwner": OWNER,
            },
        )
    ]


def test_sole_version_exact_get_and_head_bind_complete_transport() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    result = _audit(
        module,
        s3,
        expected_raw=RAW,
        expected_file_sha256=FILE_SHA,
        expected_content_type="application/json",
        expected_metadata=METADATA,
    )
    assert result.state == "sole-version"
    assert result.rows == (
        module.S3VersionRow(
            key=KEY,
            version_id="version-one",
            is_latest=True,
            is_delete_marker=False,
            etag='"0123456789abcdef0123456789abcdef"',
            size=len(RAW),
            last_modified="2026-07-27T12:00:00Z",
        ),
    )
    assert result.object == module.AuditedS3Object(
        bucket=BUCKET,
        key=KEY,
        version_id="version-one",
        raw=RAW,
        file_sha256=FILE_SHA,
        content_length=len(RAW),
        etag='"0123456789abcdef0123456789abcdef"',
        last_modified="2026-07-27T12:00:00Z",
        checksum_sha256_base64=CHECKSUM,
        checksum_type="FULL_OBJECT",
        content_type="application/json",
        metadata=METADATA,
        missing_meta=0,
    )
    with pytest.raises(FrozenInstanceError):
        result.state = "zero"
    with pytest.raises(FrozenInstanceError):
        result.rows[0].key = "mutated"
    assert result.object is not None
    with pytest.raises(FrozenInstanceError):
        result.object.key = "mutated"
    assert [operation for operation, _ in s3.calls] == [
        "list_object_versions",
        "get_object",
        "head_object",
    ]
    for operation, arguments in s3.calls:
        assert arguments["ExpectedBucketOwner"] == OWNER
        if operation in {"get_object", "head_object"}:
            assert arguments == {
                "Bucket": BUCKET,
                "Key": KEY,
                "VersionId": "version-one",
                "ChecksumMode": "ENABLED",
                "ExpectedBucketOwner": OWNER,
            }
    assert len(s3.bodies) == 1
    assert s3.bodies[0].read_count == 1
    assert s3.bodies[0].close_count == 1


def test_list_versions_exhausts_all_pages_before_rejecting_history() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.page_size = 1
    s3.add_version(version_id="old", is_latest=False)
    s3.add_version(version_id="new", is_latest=True)
    with pytest.raises(module.ProductionFenceAuditError, match="history"):
        _audit(module, s3)
    calls = [
        arguments
        for operation, arguments in s3.calls
        if operation == "list_object_versions"
    ]
    assert len(calls) == 2
    assert "KeyMarker" not in calls[0]
    assert calls[1]["KeyMarker"].startswith(KEY)
    assert calls[1]["VersionIdMarker"] == "marker-1"
    assert not [
        operation
        for operation, _ in s3.calls
        if operation in {"get_object", "head_object"}
    ]


def test_delete_marker_on_later_page_is_not_ignored() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.page_size = 1
    s3.add_version()
    s3.add_delete_marker()
    with pytest.raises(module.ProductionFenceAuditError, match="delete marker"):
        _audit(module, s3)
    assert sum(
        operation == "list_object_versions" for operation, _ in s3.calls
    ) == 2
    assert not [
        operation
        for operation, _ in s3.calls
        if operation in {"get_object", "head_object"}
    ]


@pytest.mark.parametrize(
    "mode",
    [
        "partial-next",
        "missing-next",
        "repeated-next",
        "cyclic-next",
        "wrong-echo",
        "foreign-next",
        "terminal-leftover",
    ],
)
def test_pager_rejects_partial_repeated_foreign_or_nonechoed_markers(
    mode: str,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.page_size = 1
    s3.add_version(version_id="one", is_latest=False)
    s3.add_version(version_id="two", is_latest=True)
    if mode == "repeated-next":
        s3.add_version(version_id="three", is_latest=False)
    elif mode == "cyclic-next":
        s3.add_version(version_id="three", is_latest=False)
        s3.add_version(version_id="four", is_latest=False)
    s3.list_mode = mode
    with pytest.raises(module.ProductionFenceAuditError, match="marker"):
        _audit(module, s3)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("foreign-row", "foreign prefix"),
        ("duplicate-row", "duplicate row"),
    ],
)
def test_pager_rejects_foreign_prefix_and_duplicate_response_rows(
    mode: str,
    message: str,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    s3.list_mode = mode
    with pytest.raises(module.ProductionFenceAuditError, match=message):
        _audit(module, s3)
    assert not [
        operation
        for operation, _ in s3.calls
        if operation in {"get_object", "head_object"}
    ]


def test_pager_accepts_omitted_empty_version_and_delete_collections() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.list_mode = "omit-empty-collections"
    assert _audit(module, s3).state == "zero"


@pytest.mark.parametrize(
    "mode",
    [
        "nonlist-versions",
        "common-prefix",
        "float-status",
        "nonmapping",
        "name-drift",
        "prefix-drift",
    ],
)
def test_pager_rejects_malformed_response_or_identity_drift(mode: str) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.list_mode = mode
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)


@pytest.mark.parametrize("field", ["ResponseMetadata", "IsTruncated"])
def test_pager_rejects_missing_mandatory_page_fields(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    original = s3.list_object_versions

    def omit_field(**kwargs: object) -> object:
        response = original(**kwargs)
        response.pop(field)
        return response

    monkeypatch.setattr(s3, "list_object_versions", omit_field)
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)


@pytest.mark.parametrize(
    "field",
    ["Key", "VersionId", "IsLatest", "ETag", "Size", "LastModified"],
)
def test_pager_rejects_missing_mandatory_version_row_fields(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    original = s3.list_object_versions

    def omit_field(**kwargs: object) -> object:
        response = original(**kwargs)
        response["Versions"][0].pop(field)  # type: ignore[index]
        return response

    monkeypatch.setattr(s3, "list_object_versions", omit_field)
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)


@pytest.mark.parametrize(
    "field",
    ["Key", "VersionId", "IsLatest", "LastModified"],
)
def test_pager_rejects_missing_mandatory_delete_marker_fields(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_delete_marker()
    original = s3.list_object_versions

    def omit_field(**kwargs: object) -> object:
        response = original(**kwargs)
        response["DeleteMarkers"][0].pop(field)  # type: ignore[index]
        return response

    monkeypatch.setattr(s3, "list_object_versions", omit_field)
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version_id", "null"),
        ("version_id", "with space"),
        ("is_latest", 1),
        ("etag", ""),
        ("last_modified", datetime(2026, 7, 27, 12, 0, 0)),
        (
            "last_modified",
            datetime(2026, 7, 27, 12, 0, 0, 1, tzinfo=timezone.utc),
        ),
    ],
)
def test_version_row_requires_exact_opaque_and_typed_fields(
    field: str,
    value: object,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    stored = s3.add_version()
    setattr(stored, field, value)
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)


def test_sibling_key_fails_instead_of_being_filtered() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version(key=f"{KEY}.foreign")
    with pytest.raises(module.ProductionFenceAuditError, match="sibling"):
        _audit(module, s3)


def test_sole_version_must_be_latest() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version(is_latest=False)
    with pytest.raises(module.ProductionFenceAuditError, match="latest"):
        _audit(module, s3)


@pytest.mark.parametrize(
    ("target", "changes"),
    [
        ("get", {"ResponseMetadata": {"HTTPStatusCode": 200.0}}),
        ("head", {"VersionId": "other-version"}),
        ("get", {"DeleteMarker": True}),
        ("head", {"ETag": '"ffffffffffffffffffffffffffffffff"'}),
        ("get", {"ContentLength": len(RAW) + 1}),
        ("head", {"LastModified": NOW + timedelta(seconds=1)}),
        ("get", {"ChecksumSHA256": base64.b64encode(b"x" * 32).decode()}),
        ("head", {"ChecksumType": "COMPOSITE"}),
        ("get", {"ContentType": "text/plain"}),
        ("head", {"Metadata": {"glm52-run-id": RUN_ID}}),
        ("get", {"MissingMeta": 1}),
        ("head", {"MissingMeta": True}),
    ],
)
def test_get_and_head_transport_drift_fails_closed(
    target: str,
    changes: dict[str, object],
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    getattr(s3, f"{target}_changes").update(changes)
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)
    assert len(s3.bodies) == 1
    assert s3.bodies[0].close_count == 1


@pytest.mark.parametrize("target", ["get", "head"])
@pytest.mark.parametrize(
    "field",
    [
        "ResponseMetadata",
        "VersionId",
        "ETag",
        "ContentLength",
        "LastModified",
        "ChecksumSHA256",
        "ChecksumType",
        "ContentType",
        "Metadata",
    ],
)
def test_required_get_and_head_transport_fields_cannot_be_omitted(
    target: str,
    field: str,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    getattr(s3, f"{target}_omits").add(field)
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)
    assert len(s3.bodies) == 1
    assert s3.bodies[0].close_count == 1


@pytest.mark.parametrize("target", ["get", "head"])
def test_exact_get_and_head_reject_nonmapping_responses(
    target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    monkeypatch.setattr(
        s3,
        f"{target}_object",
        lambda **_kwargs: [],
    )
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)


@pytest.mark.parametrize("surface", ["missing", "no-read", "no-close", "nonbytes"])
def test_get_body_surface_is_exact_and_mandatory(surface: str) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()

    class ReadOnlyBody:
        def read(self) -> bytes:
            return RAW

    class CloseOnlyBody:
        def close(self) -> None:
            return None

    class NonBytesBody:
        def read(self) -> bytearray:
            return bytearray(RAW)

        def close(self) -> None:
            return None

    if surface == "missing":
        s3.get_omits.add("Body")
    elif surface == "no-read":
        s3.get_changes["Body"] = CloseOnlyBody()
    elif surface == "no-close":
        s3.get_changes["Body"] = ReadOnlyBody()
    else:
        s3.get_changes["Body"] = NonBytesBody()
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)


def test_absent_missing_meta_normalizes_to_zero_and_must_match_head() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    s3.get_omits.add("MissingMeta")
    s3.head_changes["MissingMeta"] = 0
    result = _audit(module, s3)
    assert result.object.missing_meta == 0


def test_body_read_failure_blocks_result_and_still_closes() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    s3.body_read_error = RuntimeError("read failed")
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)
    assert s3.bodies[0].read_count == 1
    assert s3.bodies[0].close_count == 1


def test_body_close_failure_blocks_result() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    s3.body_close_error = RuntimeError("close failed")
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3)
    assert s3.bodies[0].read_count == 1
    assert s3.bodies[0].close_count == 1


@pytest.mark.parametrize(
    ("location", "fatal"),
    [
        ("list", ProcessDeath("list death")),
        ("get", MemoryError("get death")),
        ("head", KeyboardInterrupt()),
        ("read", SystemExit(19)),
        ("close", ProcessDeath("close death")),
        ("close", MemoryError("close memory death")),
    ],
)
def test_fatal_service_and_body_exceptions_propagate_unchanged(
    location: str,
    fatal: BaseException,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    if location == "list":
        s3.list_fault = fatal
    elif location == "get":
        s3.get_fault = fatal
    elif location == "head":
        s3.head_fault = fatal
    elif location == "read":
        s3.body_read_error = fatal
    else:
        s3.body_close_error = fatal
    with pytest.raises(type(fatal)) as raised:
        _audit(module, s3)
    assert raised.value is fatal


@pytest.mark.parametrize(
    "location",
    ["lookup", "list", "get", "head", "read", "close"],
)
def test_ordinary_service_error_is_sanitized(location: str) -> None:
    module = _module()
    disguised = module.ProductionFenceAuditError("SECRET-CREDENTIAL-MATERIAL")
    if location == "lookup":
        class AttributeFaultS3:
            def __getattribute__(self, name: str) -> object:
                if name == "list_object_versions":
                    raise disguised
                return object.__getattribute__(self, name)

        s3: object = AttributeFaultS3()
    else:
        simulator = VersionedS3Simulator()
        simulator.add_version()
        s3 = simulator
    if location == "list":
        s3.list_fault = disguised
    elif location == "get":
        s3.get_fault = disguised
    elif location == "head":
        s3.head_fault = disguised
    elif location == "read":
        s3.body_read_error = disguised
    elif location == "close":
        s3.body_close_error = disguised
    with pytest.raises(module.ProductionFenceAuditError) as raised:
        _audit(module, s3)
    assert "SECRET-CREDENTIAL-MATERIAL" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ or raised.value.__context__ is None


@pytest.mark.parametrize(
    "location",
    ["list-response", "list-row", "get-response", "response-metadata", "metadata", "head-response"],
)
def test_service_returned_mapping_access_error_is_sanitized(
    location: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    disguised = module.ProductionFenceAuditError("SECRET-CREDENTIAL-MATERIAL")

    class SecretMapping(Mapping[str, object]):
        def __getitem__(self, _key: str) -> object:
            raise disguised

        def __iter__(self) -> object:
            return iter(("secret",))

        def __len__(self) -> int:
            return 1

    s3 = VersionedS3Simulator()
    s3.add_version()
    secret = SecretMapping()
    if location == "list-response":
        monkeypatch.setattr(
            s3,
            "list_object_versions",
            lambda **_kwargs: secret,
        )
    elif location == "list-row":
        original_list = s3.list_object_versions

        def list_with_secret_row(**kwargs: object) -> object:
            response = original_list(**kwargs)
            response["Versions"] = [secret]
            return response

        monkeypatch.setattr(s3, "list_object_versions", list_with_secret_row)
    elif location in {"get-response", "response-metadata", "metadata"}:
        original_get = s3.get_object

        def get_with_secret_mapping(**kwargs: object) -> object:
            response = original_get(**kwargs)
            if location == "get-response":
                return secret
            if location == "response-metadata":
                response["ResponseMetadata"] = secret
            else:
                response["Metadata"] = secret
            return response

        monkeypatch.setattr(s3, "get_object", get_with_secret_mapping)
    else:
        monkeypatch.setattr(
            s3,
            "head_object",
            lambda **_kwargs: secret,
        )

    with pytest.raises(module.ProductionFenceAuditError) as raised:
        _audit(module, s3)
    assert "SECRET-CREDENTIAL-MATERIAL" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ or raised.value.__context__ is None


def test_response_detach_failure_after_body_exposure_still_closes() -> None:
    module = _module()
    disguised = module.ProductionFenceAuditError("SECRET-CREDENTIAL-MATERIAL")
    body = OneShotBody(RAW)

    class BodyThenFaultMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            if key == "Body":
                return body
            raise disguised

        def __iter__(self) -> object:
            return iter(("Body", "secret"))

        def __len__(self) -> int:
            return 2

    s3 = VersionedS3Simulator()
    s3.add_version()
    s3.get_object = lambda **_kwargs: BodyThenFaultMapping()  # type: ignore[method-assign]
    with pytest.raises(module.ProductionFenceAuditError) as raised:
        _audit(module, s3)
    assert "SECRET-CREDENTIAL-MATERIAL" not in str(raised.value)
    assert body.read_count == 0
    assert body.close_count == 1


def test_read_attribute_fault_after_close_capture_still_closes() -> None:
    module = _module()
    disguised = module.ProductionFenceAuditError("SECRET-CREDENTIAL-MATERIAL")

    class ReadLookupFaultBody:
        def __init__(self) -> None:
            self.close_count = 0

        @property
        def read(self) -> object:
            raise disguised

        def close(self) -> None:
            self.close_count += 1

    body = ReadLookupFaultBody()
    s3 = VersionedS3Simulator()
    s3.add_version()
    s3.get_changes["Body"] = body
    with pytest.raises(module.ProductionFenceAuditError) as raised:
        _audit(module, s3)
    assert "SECRET-CREDENTIAL-MATERIAL" not in str(raised.value)
    assert body.close_count == 1


def test_service_datetime_exception_chain_is_suppressed() -> None:
    module = _module()

    class SecretTimezone(tzinfo):
        def utcoffset(self, _value: datetime | None) -> timedelta:
            raise RuntimeError("SECRET-CREDENTIAL-MATERIAL")

        def dst(self, _value: datetime | None) -> timedelta:
            return timedelta(0)

        def tzname(self, _value: datetime | None) -> str:
            return "secret"

    s3 = VersionedS3Simulator()
    stored = s3.add_version()
    stored.last_modified = datetime(
        2026,
        7,
        27,
        12,
        0,
        0,
        tzinfo=SecretTimezone(),
    )
    with pytest.raises(module.ProductionFenceAuditError) as raised:
        _audit(module, s3)
    assert "SECRET-CREDENTIAL-MATERIAL" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ or raised.value.__context__ is None


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_raw": RAW},
        {"expected_file_sha256": FILE_SHA},
        {
            "expected_raw": RAW,
            "expected_file_sha256": "f" * 64,
        },
    ],
)
def test_expected_raw_and_file_sha_are_paired_and_self_consistent(
    changes: dict[str, object],
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3, **changes)
    assert s3.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {
            "expected_raw": RAW,
            "expected_file_sha256": FILE_SHA,
        },
        {"expected_content_type": "application/json"},
        {"expected_metadata": METADATA},
    ],
)
def test_zero_coordinate_cannot_satisfy_expected_object_identity(
    changes: dict[str, object],
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "expected_raw": RAW + b" ",
            "expected_file_sha256": hashlib.sha256(RAW + b" ").hexdigest(),
        },
        {"expected_content_type": "text/plain"},
        {"expected_metadata": (("glm52-run-id", RUN_ID),)},
    ],
)
def test_expected_object_identity_is_strict_when_supplied(
    changes: dict[str, object],
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    s3.add_version()
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3, **changes)


def test_wrong_expected_owner_fails_before_s3() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    with pytest.raises(module.ProductionFenceAuditError):
        _audit(module, s3, expected_bucket_owner="135792468013")
    assert s3.calls == []


def test_services_may_expose_sentinel_write_methods_but_none_are_called() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    assert _audit(module, s3).state == "zero"
    assert [operation for operation, _ in s3.calls] == [
        "list_object_versions"
    ]


def test_namespace_exact_reads_every_control_and_reaches_zero_next_slot(
    modeled_controls: dict[str, object],
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    successor = modeled_controls["successor"]
    genesis = modeled_controls["genesis"]
    namespace_bucket = modeled_controls["bucket"]
    _add_modeled_control(s3, successor)
    _add_modeled_control(s3, genesis)
    result = _namespace(module, s3, modeled_controls)
    assert result.observation == "modeled-chain-with-zero-next-slot"
    assert result.modeled_head.genesis.key == genesis.key
    assert result.modeled_head.head.key == successor.key
    assert result.modeled_head.successors == (result.modeled_head.head,)
    assert result.observed_zero_coordinate.state == "zero"
    assert result.observed_zero_coordinate.key == (
        result.modeled_head.next_successor_key
    )
    with pytest.raises(FrozenInstanceError):
        result.observation = "empty"
    assert [operation for operation, _ in s3.calls] == [
        "list_object_versions",
        "get_object",
        "head_object",
        "get_object",
        "head_object",
        "list_object_versions",
    ]
    assert [
        arguments["Key"]
        for operation, arguments in s3.calls
        if operation in {"get_object", "head_object"}
    ] == [successor.key, successor.key, genesis.key, genesis.key]
    list_calls = [
        arguments
        for operation, arguments in s3.calls
        if operation == "list_object_versions"
    ]
    assert list_calls == [
        {
            "Bucket": namespace_bucket,
            "Prefix": f"campaigns/{RUN_ID}/authorities/fence/",
            "ExpectedBucketOwner": OWNER,
        },
        {
            "Bucket": namespace_bucket,
            "Prefix": result.modeled_head.next_successor_key,
            "ExpectedBucketOwner": OWNER,
        },
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content_type", "text/plain"),
        ("metadata", (("glm52-run-id", RUN_ID),)),
        (
            "metadata",
            METADATA + (("glm52-unexpected", "value"),),
        ),
        ("missing_meta", 1),
    ],
)
def test_namespace_authenticates_complete_control_transport_metadata(
    modeled_controls: dict[str, object],
    field: str,
    value: object,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    stored = _add_modeled_control(s3, modeled_controls["genesis"])
    setattr(stored, field, value)
    with pytest.raises(module.ProductionFenceAuditError):
        _namespace(module, s3, modeled_controls)


@pytest.mark.parametrize("case", ["history", "delete", "alternate"])
def test_namespace_rejects_history_delete_marker_and_alternate_control_key(
    modeled_controls: dict[str, object],
    case: str,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    genesis = modeled_controls["genesis"]
    if case == "history":
        _add_modeled_control(
            s3,
            genesis,
            version_id="genesis-version-old",
            is_latest=False,
        )
        _add_modeled_control(
            s3,
            genesis,
            version_id="genesis-version-new",
        )
    elif case == "delete":
        _add_modeled_control(s3, genesis)
        s3.add_delete_marker(key=genesis.key)
    else:
        _add_modeled_control(s3, genesis)
        s3.add_version(
            key=(
                f"campaigns/{RUN_ID}/authorities/fence/"
                "ALTERNATE_CONTROL.json"
            ),
        )
    with pytest.raises(module.ProductionFenceAuditError):
        _namespace(module, s3, modeled_controls)
    assert not [
        operation
        for operation, _ in s3.calls
        if operation in {"get_object", "head_object"}
    ]


def test_namespace_rejects_an_orphan_modeled_successor(
    modeled_controls: dict[str, object],
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    _add_modeled_control(s3, modeled_controls["successor"])
    with pytest.raises(module.ProductionFenceAuditError):
        _namespace(module, s3, modeled_controls)
    assert [operation for operation, _ in s3.calls] == [
        "list_object_versions",
        "get_object",
        "head_object",
    ]


@pytest.mark.parametrize("case", ["fork", "cycle", "repeated-digest"])
def test_namespace_rejects_fork_cycle_and_repeated_digest_without_filtering(
    modeled_controls: dict[str, object],
    case: str,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    genesis = modeled_controls["genesis"]
    successor = modeled_controls["successor"]
    _add_modeled_control(s3, genesis)
    _add_modeled_control(s3, successor)
    if case == "fork":
        _add_modeled_control(s3, modeled_controls["fork"])
    elif case == "cycle":
        _add_modeled_control(s3, successor)
    else:
        repeated = replace(
            successor,
            key=(
                f"campaigns/{RUN_ID}/authorities/fence/successors/"
                f"{'f' * 64}/FENCE_SUCCESSOR.json"
            ),
            version_id="repeated-digest-version",
        )
        _add_modeled_control(s3, repeated)
    with pytest.raises(module.ProductionFenceAuditError):
        _namespace(module, s3, modeled_controls)
    get_count = sum(
        operation == "get_object" for operation, _ in s3.calls
    )
    if case == "repeated-digest":
        assert get_count == 3
    else:
        assert get_count == 0


def test_namespace_forwards_complete_repeated_digest_tuple_to_fresh_walker(
    modeled_controls: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    genesis = modeled_controls["genesis"]
    successor = modeled_controls["successor"]
    repeated = replace(
        successor,
        key=(
            f"campaigns/{RUN_ID}/authorities/fence/successors/"
            f"{'e' * 64}/FENCE_SUCCESSOR.json"
        ),
        version_id="unfiltered-repeated-digest-version",
    )
    for artifact in (genesis, successor, repeated):
        _add_modeled_control(s3, artifact)
    captured: tuple[object, ...] | None = None
    real_walk = module.walk_modeled_fence_chain

    def capture_then_walk(**kwargs: object) -> object:
        nonlocal captured
        controls = kwargs["controls"]
        assert isinstance(controls, tuple)
        captured = controls
        return real_walk(**kwargs)

    monkeypatch.setattr(
        module,
        "walk_modeled_fence_chain",
        capture_then_walk,
    )
    with pytest.raises(module.ProductionFenceAuditError):
        _namespace(module, s3, modeled_controls)
    assert captured is not None
    assert len(captured) == 3
    assert {item.key for item in captured} == {
        genesis.key,
        successor.key,
        repeated.key,
    }
    assert [
        item.body_sha256 for item in captured
    ].count(successor.body_sha256) == 2


def test_namespace_requires_fresh_zero_at_reached_next_successor_slot(
    modeled_controls: dict[str, object],
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    genesis = modeled_controls["genesis"]
    _add_modeled_control(s3, genesis)
    next_key = (
        f"campaigns/{RUN_ID}/authorities/fence/successors/"
        f"{genesis.body_sha256}/FENCE_SUCCESSOR.json"
    )
    late = s3.add_version(
        key=next_key,
        version_id="late-version",
        raw=b'{"late":true}\n',
        metadata=(("glm52-run-id", RUN_ID),),
    )
    s3.hidden_on_first_list.add((late.key, late.version_id))
    with pytest.raises(module.ProductionFenceAuditError, match="not zero"):
        _namespace(module, s3, modeled_controls)
    assert [
        arguments["Prefix"]
        for operation, arguments in s3.calls
        if operation == "list_object_versions"
    ] == [
        f"campaigns/{RUN_ID}/authorities/fence/",
        next_key,
    ]


def test_namespace_rewalks_from_genesis_on_every_invocation(
    modeled_controls: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    _add_modeled_control(s3, modeled_controls["genesis"])
    calls = 0
    real_walk = module.walk_modeled_fence_chain

    def counted_walk(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_walk(**kwargs)

    monkeypatch.setattr(module, "walk_modeled_fence_chain", counted_walk)
    first = _namespace(module, s3, modeled_controls)
    second = _namespace(module, s3, modeled_controls)
    assert first == second
    assert calls == 2
    assert sum(
        operation == "get_object" for operation, _ in s3.calls
    ) == 2
    assert [
        arguments["Prefix"]
        for operation, arguments in s3.calls
        if operation == "list_object_versions"
    ].count(f"campaigns/{RUN_ID}/authorities/fence/") == 2


def test_namespace_model_fatal_exception_propagates_unchanged(
    modeled_controls: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    _add_modeled_control(s3, modeled_controls["genesis"])
    fatal = MemoryError("model death")

    def die(**_kwargs: object) -> object:
        raise fatal

    monkeypatch.setattr(module, "walk_modeled_fence_chain", die)
    with pytest.raises(MemoryError) as raised:
        _namespace(module, s3, modeled_controls)
    assert raised.value is fatal


def test_empty_namespace_rechecks_the_exact_genesis_coordinate() -> None:
    module = _module()
    s3 = VersionedS3Simulator()
    result = module.audit_fence_namespace(
        services=_services(module, s3),
        bucket=BUCKET,
        run_id=RUN_ID,
        expected_bucket_owner=OWNER,
        descriptor=object(),
        intent=object(),
        approval=object(),
        controller_baseline=object(),
        must_start_control_plane_ready=object(),
        submission_acquisition=object(),
        gpu_spend_snapshot=object(),
    )
    assert result.observation == "empty"
    assert result.modeled_head is None
    assert result.observed_zero_coordinate.state == "zero"
    list_calls = [
        arguments
        for operation, arguments in s3.calls
        if operation == "list_object_versions"
    ]
    assert [item["Prefix"] for item in list_calls] == [
        f"campaigns/{RUN_ID}/authorities/fence/",
        KEY,
    ]


def test_audit_module_ast_has_no_side_effect_or_external_authority_surface() -> None:
    module = _module()
    source = inspect.getsource(module)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {
            "boto3",
            "botocore",
            "http",
            "os",
            "pathlib",
            "random",
            "requests",
            "secrets",
            "shutil",
            "sky",
            "skypilot",
            "socket",
            "subprocess",
            "tempfile",
            "time",
            "urllib",
            "uuid",
        }
    )

    forbidden_calls = {
        "abort_multipart_upload",
        "client",
        "complete_multipart_upload",
        "copy_object",
        "delete_object",
        "delete_objects",
        "eval",
        "exec",
        "getenv",
        "list_objects_v2",
        "makedirs",
        "mkdir",
        "open",
        "put_object",
        "remove",
        "removedirs",
        "rename",
        "replace",
        "rmdir",
        "system",
        "touch",
        "unlink",
        "upload_part",
        "upload_part_copy",
        "write",
        "write_bytes",
        "write_text",
        "writelines",
    }
    forbidden_clock_random_or_environment = {
        "choice",
        "environ",
        "monotonic",
        "now",
        "perf_counter",
        "randbytes",
        "randint",
        "random",
        "time",
        "today",
        "token_bytes",
        "token_hex",
        "token_urlsafe",
        "urandom",
        "utcnow",
        "uuid1",
        "uuid4",
    }
    forbidden_h1e = {
        "decide_production_generation_action",
        "validate_modeled_submit_once",
    }
    observed_call_names: set[str] = set()
    observed_string_literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                observed_call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                observed_call_names.add(node.func.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            observed_string_literals.add(node.value)
    forbidden_names = (
        forbidden_calls
        | forbidden_clock_random_or_environment
        | forbidden_h1e
    )
    assert observed_call_names.isdisjoint(forbidden_names)
    assert observed_string_literals.isdisjoint(forbidden_names)
    assert "glm52_sky_production_generation" not in source

    forbidden_source_fragments = {
        "boto3",
        "botocore",
        "requests",
        "skypilot",
        "socket",
        "subprocess",
        "urllib",
        "os.getenv",
        "datetime.now",
        "os.environ",
        "production_generation_action",
    }
    assert all(
        fragment not in source for fragment in forbidden_source_fragments
    )
    assert not hasattr(module, "main")


def test_no_public_function_accepts_a_modeled_head_as_authority() -> None:
    module = _module()
    for name in module.__all__:
        value = getattr(module, name)
        if not inspect.isfunction(value):
            continue
        signature = inspect.signature(value)
        assert "modeled_head" not in signature.parameters
        assert all(
            "ModeledFenceHead" not in str(parameter.annotation)
            for parameter in signature.parameters.values()
        )
        assert all(
            parameter.kind
            is not inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
