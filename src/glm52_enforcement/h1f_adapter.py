"""Fresh, import-light H.1f S3 namespace and coordinate audit."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Dict, List, Mapping, Optional, Tuple

from .s3_keys import fence_genesis_s3_key, fence_successor_s3_key
from .s3_records import (
    ImmutableJsonCandidate,
    S3ObjectIdentity,
    build_s3_object_identity,
    validate_immutable_json_candidate,
)


_OWNER = "246813579024"
_REGION = "us-west-2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SUCCESSOR = re.compile(
    r"campaigns/(?P<run>[A-Za-z0-9][A-Za-z0-9._-]{0,127})/"
    r"authorities/fence/successors/(?P<predecessor>[0-9a-f]{64})/"
    r"FENCE_SUCCESSOR[.]json\Z"
)
_GENESIS_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "campaign_identity_sha256",
    "genesis_key",
    "enrolled_sources",
    "reserved_generations",
    "live_state",
    "created_at",
    "fence_body_sha256",
}
_SUCCESSOR_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "campaign_identity_sha256",
    "successor_key",
    "predecessor_control_key",
    "predecessor_control_version_id",
    "predecessor_control_file_sha256",
    "predecessor_control_body_sha256",
    "enrolled_sources",
    "reserved_generations",
    "added_sources",
    "added_generation_reservations",
    "live_state",
    "selected_at",
    "fence_body_sha256",
}
_LIVE_STATE_FIELDS = {
    "account_id",
    "region",
    "bucket",
    "bucket_versioning_status",
    "bucket_mfa_delete_status",
    "bucket_policy_sha256",
    "lifecycle_configuration_state",
    "lifecycle_configuration_sha256",
    "replication_configuration_state",
    "replication_configuration_sha256",
    "cloudformation_stack_id",
    "cloudformation_template_sha256",
    "cloudformation_parameters_sha256",
    "executor_identity_sha256",
    "publisher_deny_policy_sha256",
    "h1d_control_plane_ready_body_sha256",
}
_ABSENT_CONFIGURATION_SHA256 = hashlib.sha256(
    b'{"state":"absent"}\n'
).hexdigest()
_ROLLBACK_GUARDED_LIVE_FIELDS = (
    "bucket_policy_sha256",
    "publisher_deny_policy_sha256",
    "cloudformation_template_sha256",
    "cloudformation_parameters_sha256",
    "h1d_control_plane_ready_body_sha256",
)


class H1fAuditError(ValueError):
    """A fresh H.1f observation failed closed."""


@dataclass(frozen=True)
class S3VersionRow:
    key: str
    version_id: str
    is_latest: bool
    is_delete_marker: bool
    etag: Optional[str]
    size: Optional[int]
    last_modified: str


@dataclass(frozen=True)
class H1fAuthoritySnapshot:
    authority_domain: str
    bucket: str
    run_id: str
    activation_id: str
    generation: int
    epoch: int
    execution_arn: str
    barrier_nonce_sha256: str
    closing_revision: int
    active_head_key: str
    active_head_version_id: str
    active_head_file_sha256: str
    active_head_body_sha256: str


@dataclass(frozen=True)
class H1fAuditRequest:
    operation_kind: str
    action_key: str
    candidate: ImmutableJsonCandidate


@dataclass(frozen=True)
class H1fAuditResult:
    authority_domain: str
    operation_kind: str
    action_key: str
    candidate_identity_sha256: str
    activation_id: str
    generation: int
    epoch: int
    execution_arn: str
    barrier_nonce_sha256: str
    closing_revision: int
    expected_authorized_revision: int
    active_head_key: str
    active_head_version_id: str
    active_head_file_sha256: str
    active_head_body_sha256: str
    canonical_body_sha256: str


@dataclass(frozen=True)
class ExactCoordinateResult:
    state: str
    object_identity: Optional[S3ObjectIdentity]


@dataclass(frozen=True)
class _AuditedObject:
    key: str
    version_id: str
    raw: bytes
    file_sha256: str
    etag: str
    content_length: int
    last_modified: str
    checksum_sha256_base64: str
    checksum_type: str
    content_type: str
    metadata: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class _Control:
    object: _AuditedObject
    record: Mapping[str, object]
    body_sha256: str


def _fail(message: str) -> None:
    raise H1fAuditError(message)


def _string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        _fail(f"{name} must be an exact nonempty string")
    return value


def _sha(name: str, value: object) -> str:
    text = _string(name, value)
    if _SHA256.fullmatch(text) is None:
        _fail(f"{name} must be exact lowercase SHA-256")
    return text


def _mapping(name: str, value: object) -> Dict[object, object]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    try:
        return dict(value)
    except Exception as exc:
        raise H1fAuditError(f"{name} cannot be read") from exc


def _status(name: str, value: object) -> Dict[object, object]:
    response = _mapping(f"{name} response", value)
    metadata = _mapping(f"{name} ResponseMetadata", response.get("ResponseMetadata"))
    if type(metadata.get("HTTPStatusCode")) is not int or metadata.get(
        "HTTPStatusCode"
    ) != 200:
        _fail(f"{name} status is not exact 200")
    request_id = metadata.get("RequestId")
    if type(request_id) is not str or not request_id:
        _fail(f"{name} request ID is missing")
    return response


def _opaque(name: str, value: object) -> str:
    text = _string(name, value)
    if (
        text == "null"
        or len(text) > 1024
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in text)
    ):
        _fail(f"{name} is not an opaque VersionId")
    return text


def _time(name: str, value: object) -> str:
    if type(value) is not datetime or value.microsecond != 0:
        _fail(f"{name} must be a timezone-aware whole-second datetime")
    try:
        if value.utcoffset() is None:
            _fail(f"{name} must be timezone-aware")
        normalized = value.astimezone(timezone.utc)
    except Exception as exc:
        raise H1fAuditError(f"{name} cannot normalize") from exc
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _metadata(value: object) -> Tuple[Tuple[str, str], ...]:
    plain = _mapping("S3 metadata", value)
    pairs = []
    for key, item in plain.items():
        if (
            type(key) is not str
            or not key
            or key != key.lower()
            or type(item) is not str
        ):
            _fail("S3 metadata contains an invalid member")
        pairs.append((key, item))
    pairs.sort()
    if len({key for key, _ in pairs}) != len(pairs):
        _fail("S3 metadata contains duplicate keys")
    return tuple(pairs)


def _call(service: object, method_name: str, arguments: Mapping[str, object]) -> object:
    try:
        method = getattr(service, method_name)
    except Exception as exc:
        raise H1fAuditError(f"S3 lacks {method_name}") from exc
    if not callable(method):
        _fail(f"S3 lacks {method_name}")
    try:
        return method(**dict(arguments))
    except Exception as exc:
        raise H1fAuditError(f"S3 {method_name} failed") from exc


def _list_versions(
    *, s3: object, bucket: str, prefix: str
) -> Tuple[S3VersionRow, ...]:
    markers: Optional[Tuple[str, str]] = None
    seen_markers = set()
    seen_rows = set()
    rows: List[S3VersionRow] = []
    while True:
        arguments: Dict[str, object] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "ExpectedBucketOwner": _OWNER,
        }
        if markers is not None:
            arguments["KeyMarker"] = markers[0]
            arguments["VersionIdMarker"] = markers[1]
        response = _status(
            "ListObjectVersions",
            _call(s3, "list_object_versions", arguments),
        )
        if response.get("Name", bucket) != bucket or response.get(
            "Prefix", prefix
        ) != prefix:
            _fail("ListObjectVersions bucket or prefix drifted")
        truncated = response.get("IsTruncated")
        if type(truncated) is not bool:
            _fail("ListObjectVersions IsTruncated must be an exact bool")
        for collection in ("Versions", "DeleteMarkers", "CommonPrefixes"):
            if collection in response and type(response[collection]) is not list:
                _fail(f"ListObjectVersions {collection} must be an exact list")
        if response.get("CommonPrefixes", []):
            _fail("ListObjectVersions returned a common prefix")
        has_key = "KeyMarker" in response
        has_version = "VersionIdMarker" in response
        if has_key != has_version:
            _fail("ListObjectVersions marker echo is partial")
        expected_echo = ("", "") if markers is None else markers
        if has_key and (
            response["KeyMarker"],
            response["VersionIdMarker"],
        ) != expected_echo:
            _fail("ListObjectVersions marker echo drifted")
        if markers is not None and not has_key:
            _fail("ListObjectVersions omitted request marker echo")
        for collection_name, is_delete in (
            ("Versions", False),
            ("DeleteMarkers", True),
        ):
            for raw_row in response.get(collection_name, []):
                row = _mapping("ListObjectVersions row", raw_row)
                key = _string("version key", row.get("Key"))
                if not key.startswith(prefix):
                    _fail("ListObjectVersions row has a foreign prefix")
                version_id = _opaque("version ID", row.get("VersionId"))
                latest = row.get("IsLatest")
                if type(latest) is not bool:
                    _fail("ListObjectVersions IsLatest must be an exact bool")
                modified = _time("ListObjectVersions LastModified", row.get("LastModified"))
                if is_delete:
                    if "ETag" in row or "Size" in row:
                        _fail("delete marker carries object fields")
                    etag = None
                    size = None
                else:
                    etag = _string("version ETag", row.get("ETag"))
                    size = row.get("Size")
                    if type(size) is not int or size < 0:
                        _fail("version size must be an exact nonnegative integer")
                identity = (key, version_id, is_delete)
                if identity in seen_rows:
                    _fail("ListObjectVersions returned a duplicate row")
                seen_rows.add(identity)
                rows.append(
                    S3VersionRow(
                        key=key,
                        version_id=version_id,
                        is_latest=latest,
                        is_delete_marker=is_delete,
                        etag=etag,
                        size=size,
                        last_modified=modified,
                    )
                )
        next_key = "NextKeyMarker" in response
        next_version = "NextVersionIdMarker" in response
        if next_key != next_version:
            _fail("ListObjectVersions next marker pair is partial")
        if truncated:
            if not next_key:
                _fail("truncated ListObjectVersions page lacks markers")
            pair = (
                _string("next key marker", response["NextKeyMarker"]),
                _opaque("next version marker", response["NextVersionIdMarker"]),
            )
            if not pair[0].startswith(prefix):
                _fail("next key marker has a foreign prefix")
            if pair == markers or pair in seen_markers:
                _fail("ListObjectVersions marker pair repeated or cycled")
            seen_markers.add(pair)
            markers = pair
            continue
        if next_key and (
            response["NextKeyMarker"] != ""
            or response["NextVersionIdMarker"] != ""
        ):
            _fail("terminal ListObjectVersions page has leftover markers")
        return tuple(rows)


def _transport(
    response: Mapping[object, object],
    *,
    operation: str,
    version_id: str,
) -> Tuple[str, int, str, str, str, str, Tuple[Tuple[str, str], ...]]:
    if _opaque(f"{operation} VersionId", response.get("VersionId")) != version_id:
        _fail(f"{operation} VersionId drifted")
    if "DeleteMarker" in response and response["DeleteMarker"] is not False:
        _fail(f"{operation} reported a delete marker")
    etag = _string(f"{operation} ETag", response.get("ETag"))
    length = response.get("ContentLength")
    if type(length) is not int or length < 0:
        _fail(f"{operation} ContentLength is invalid")
    modified = _time(f"{operation} LastModified", response.get("LastModified"))
    checksum = _string(f"{operation} checksum", response.get("ChecksumSHA256"))
    checksum_type = _string(
        f"{operation} checksum type", response.get("ChecksumType")
    )
    if checksum_type != "FULL_OBJECT":
        _fail(f"{operation} checksum is not FULL_OBJECT")
    content_type = _string(
        f"{operation} content type", response.get("ContentType")
    )
    metadata = _metadata(response.get("Metadata"))
    missing_meta = response.get("MissingMeta", 0)
    if type(missing_meta) is not int or missing_meta != 0:
        _fail(f"{operation} MissingMeta is not exact zero")
    return (
        etag,
        length,
        modified,
        checksum,
        checksum_type,
        content_type,
        metadata,
    )


def _audit_row(*, s3: object, bucket: str, row: S3VersionRow) -> _AuditedObject:
    arguments = {
        "Bucket": bucket,
        "Key": row.key,
        "VersionId": row.version_id,
        "ChecksumMode": "ENABLED",
        "ExpectedBucketOwner": _OWNER,
    }
    get_response_raw = _call(s3, "get_object", arguments)
    body: object = None
    closer: object = None
    try:
        if not isinstance(get_response_raw, Mapping):
            _fail("GetObject response must be a mapping")
        body = get_response_raw.get("Body")
        closer = getattr(body, "close")
        if not callable(closer):
            _fail("GetObject body lacks callable close")
        reader = getattr(body, "read")
        if not callable(reader):
            _fail("GetObject body lacks callable read")
    except BaseException as primary:
        if callable(closer):
            try:
                closer()
            except BaseException:
                pass
        if isinstance(primary, H1fAuditError):
            raise
        raise H1fAuditError("GetObject body lacks read/close") from primary
    primary: Optional[BaseException] = None
    raw: object = None
    get_identity: object = None
    try:
        get_response = _mapping("GetObject response", get_response_raw)
        normalized_get = _status("GetObject", get_response)
        get_identity = _transport(
            normalized_get,
            operation="GetObject",
            version_id=row.version_id,
        )
        raw = reader()
        if type(raw) is not bytes:
            _fail("GetObject body did not return exact bytes")
    except BaseException as exc:
        primary = exc
    close_error: Optional[BaseException] = None
    try:
        closer()
    except BaseException as exc:
        close_error = exc
    if primary is not None:
        raise primary
    if close_error is not None:
        raise close_error
    head_response = _status(
        "HeadObject",
        _call(s3, "head_object", arguments),
    )
    head_identity = _transport(
        head_response,
        operation="HeadObject",
        version_id=row.version_id,
    )
    if get_identity != head_identity:
        _fail("exact GET and HEAD transport identities disagree")
    assert type(raw) is bytes
    (
        etag,
        length,
        modified,
        checksum,
        checksum_type,
        content_type,
        metadata,
    ) = head_identity
    digest = hashlib.sha256(raw).digest()
    if (
        length != len(raw)
        or checksum != base64.b64encode(digest).decode("ascii")
        or row.etag != etag
        or row.size != length
        or row.last_modified != modified
    ):
        _fail("exact object transport identity disagrees")
    return _AuditedObject(
        key=row.key,
        version_id=row.version_id,
        raw=raw,
        file_sha256=digest.hex(),
        etag=etag,
        content_length=length,
        last_modified=modified,
        checksum_sha256_base64=checksum,
        checksum_type=checksum_type,
        content_type=content_type,
        metadata=metadata,
    )


def _canonical_json_file(raw: bytes) -> Dict[str, object]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("control JSON LF policy is invalid")
    try:
        text = raw[:-1].decode("ascii")
    except UnicodeError as exc:
        raise H1fAuditError("control JSON is not ASCII-safe") from exc

    def unique(pairs: object) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in pairs:  # type: ignore[union-attr]
            if key in result:
                _fail("control JSON contains duplicate members")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=lambda _item: _fail("non-finite JSON is forbidden"),
        )
    except H1fAuditError:
        raise
    except Exception as exc:
        raise H1fAuditError("control JSON is invalid") from exc
    if type(value) is not dict or _canonical(value) + b"\n" != raw:
        _fail("control JSON is not canonical")
    return value


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise H1fAuditError("audit value is not canonical JSON") from exc


def _live_state(
    value: object,
    *,
    bucket: str,
) -> Dict[str, object]:
    if type(value) is not dict or set(value) != _LIVE_STATE_FIELDS:
        _fail("fence control live_state field set is not exact and closed")
    live = dict(value)
    if (
        live.get("account_id") != _OWNER
        or live.get("region") != _REGION
        or live.get("bucket") != bucket
        or live.get("bucket_versioning_status") != "Enabled"
        or live.get("bucket_mfa_delete_status")
        not in {"Enabled", "Disabled", "Unavailable"}
    ):
        _fail("fence control live_state cloud identity drifted")
    if (
        live.get("lifecycle_configuration_state") != "absent"
        or live.get("lifecycle_configuration_sha256")
        != _ABSENT_CONFIGURATION_SHA256
        or live.get("replication_configuration_state") != "absent"
        or live.get("replication_configuration_sha256")
        != _ABSENT_CONFIGURATION_SHA256
    ):
        _fail("fence control lifecycle/replication is not exact absent")
    for field in (
        "bucket_policy_sha256",
        "lifecycle_configuration_sha256",
        "replication_configuration_sha256",
        "cloudformation_template_sha256",
        "cloudformation_parameters_sha256",
        "executor_identity_sha256",
        "publisher_deny_policy_sha256",
        "h1d_control_plane_ready_body_sha256",
    ):
        _sha(f"live_state {field}", live.get(field))
    _string(
        "live_state cloudformation_stack_id",
        live.get("cloudformation_stack_id"),
    )
    return live


def _control(audited: _AuditedObject, *, bucket: str, run_id: str) -> _Control:
    record = _canonical_json_file(audited.raw)
    record_type = record.get("record_type")
    if record_type not in {
        "glm52_sky_production_fence_genesis_v1",
        "glm52_sky_production_fence_successor_v1",
    }:
        _fail("fence record type is unknown")
    expected_fields = (
        _GENESIS_FIELDS
        if record_type == "glm52_sky_production_fence_genesis_v1"
        else _SUCCESSOR_FIELDS
    )
    if set(record) != expected_fields:
        _fail("fence control field set is not exact and closed")
    if (
        record.get("schema_version") != 1
        or record.get("account_id") != _OWNER
        or record.get("region") != _REGION
        or record.get("bucket") != bucket
        or record.get("run_id") != run_id
    ):
        _fail("fence control authority identity drifted")
    _live_state(record.get("live_state"), bucket=bucket)
    sources = _closed_array(record, "enrolled_sources")
    reservations = _closed_array(record, "reserved_generations")
    if (
        sources != sorted(sources, key=_source_sort_key)
        or reservations != sorted(
            reservations, key=_reservation_sort_key
        )
        or len({_canonical(item) for item in sources}) != len(sources)
        or len({_canonical(item) for item in reservations})
        != len(reservations)
    ):
        _fail("fence control retained sets are not closed and ordered")
    if record_type == "glm52_sky_production_fence_genesis_v1":
        if record.get("genesis_key") != fence_genesis_s3_key(run_id=run_id):
            _fail("fence genesis coordinate drifted")
    else:
        _closed_array(record, "added_sources")
        _closed_array(record, "added_generation_reservations")
    body_hash = _sha("fence_body_sha256", record.get("fence_body_sha256"))
    body = dict(record)
    del body["fence_body_sha256"]
    if hashlib.sha256(_canonical(body)).hexdigest() != body_hash:
        _fail("fence control body self-hash drifted")
    campaign = _sha(
        "campaign_identity_sha256", record.get("campaign_identity_sha256")
    )
    expected_metadata = tuple(
        sorted(
            (
                ("glm52-body-sha256", body_hash),
                ("glm52-campaign-identity-sha256", campaign),
                ("glm52-file-sha256", audited.file_sha256),
                ("glm52-record-type", record_type),
                ("glm52-run-id", run_id),
            )
        )
    )
    if (
        audited.content_type != "application/json"
        or audited.metadata != expected_metadata
    ):
        _fail("fence control transport metadata drifted")
    return _Control(audited, record, body_hash)


def _closed_array(
    record: Mapping[str, object], field: str
) -> List[object]:
    value = record.get(field)
    if type(value) is not list:
        _fail(f"fence {field} must be an exact array")
    return value


def _source_sort_key(value: object) -> Tuple[str, str]:
    if type(value) is not dict:
        _fail("fence source enrollment must be an exact object")
    source_kind = value.get("source_kind")
    key = value.get("key")
    if type(source_kind) is not str or not source_kind or type(key) is not str or not key:
        _fail("fence source enrollment identity is invalid")
    return source_kind, key


def _reservation_sort_key(value: object) -> int:
    if type(value) is not dict:
        _fail("fence generation reservation must be an exact object")
    generation = value.get("generation")
    if type(generation) is not int or generation < 1:
        _fail("fence generation reservation identity is invalid")
    return generation


def _validate_successor_history(
    predecessor: _Control,
    successor: _Control,
    *,
    ancestor_live_states: Tuple[Mapping[str, object], ...],
) -> None:
    for field in (
        "account_id",
        "region",
        "bucket",
        "run_id",
        "campaign_identity_sha256",
    ):
        if successor.record.get(field) != predecessor.record.get(field):
            _fail("fence successor current/history identity mismatch")
    prior_sources = _closed_array(predecessor.record, "enrolled_sources")
    prior_reservations = _closed_array(
        predecessor.record, "reserved_generations"
    )
    added_sources = _closed_array(successor.record, "added_sources")
    added_reservations = _closed_array(
        successor.record, "added_generation_reservations"
    )
    if not added_sources and not added_reservations:
        _fail("fence successor is an empty transition")
    expected_sources = sorted(
        prior_sources + added_sources, key=_source_sort_key
    )
    expected_reservations = sorted(
        prior_reservations + added_reservations, key=_reservation_sort_key
    )
    if (
        successor.record.get("enrolled_sources") != expected_sources
        or successor.record.get("reserved_generations")
        != expected_reservations
        or len({_canonical(item) for item in expected_sources})
        != len(expected_sources)
        or len({_canonical(item) for item in expected_reservations})
        != len(expected_reservations)
    ):
        _fail("fence successor current/history set mismatch")
    bucket = _string("fence successor bucket", successor.record.get("bucket"))
    prior_live = _live_state(
        predecessor.record.get("live_state"),
        bucket=bucket,
    )
    current_live = _live_state(
        successor.record.get("live_state"),
        bucket=bucket,
    )
    for field in (
        "account_id",
        "region",
        "bucket",
        "bucket_versioning_status",
        "bucket_mfa_delete_status",
        "lifecycle_configuration_state",
        "lifecycle_configuration_sha256",
        "replication_configuration_state",
        "replication_configuration_sha256",
        "cloudformation_stack_id",
        "executor_identity_sha256",
    ):
        if current_live[field] != prior_live[field]:
            _fail("fence successor changed immutable live_state")
    if (
        current_live["bucket_policy_sha256"]
        == prior_live["bucket_policy_sha256"]
    ):
        _fail("fence successor live-state policy did not advance")
    if bool(added_sources) != (
        current_live["publisher_deny_policy_sha256"]
        != prior_live["publisher_deny_policy_sha256"]
    ):
        _fail("fence successor publisher-deny/source delta drifted")
    readiness_sources = []
    for source in added_sources:
        if type(source) is not dict:
            _fail("fence added source must be an exact object")
        if source.get("source_kind") == "production-control-plane-readiness":
            _sha("readiness source body_sha256", source.get("body_sha256"))
            readiness_sources.append(source)
    readiness_fields = (
        "cloudformation_template_sha256",
        "cloudformation_parameters_sha256",
        "h1d_control_plane_ready_body_sha256",
    )
    if not readiness_sources:
        if any(
            current_live[field] != prior_live[field]
            for field in readiness_fields
        ):
            _fail("fence successor changed readiness state without source")
    elif (
        len(readiness_sources) != 1
        or current_live["h1d_control_plane_ready_body_sha256"]
        != readiness_sources[0]["body_sha256"]
    ):
        _fail("fence successor readiness source binding drifted")
    for field in _ROLLBACK_GUARDED_LIVE_FIELDS:
        before = prior_live[field]
        after = current_live[field]
        if after != before and any(
            after == ancestor.get(field)
            for ancestor in ancestor_live_states[:-1]
        ):
            _fail("fence successor live_state rolled back to an ancestor")


def _walk_namespace(
    *, s3: object, snapshot: H1fAuthoritySnapshot
) -> _Control:
    prefix = f"campaigns/{snapshot.run_id}/authorities/fence/"
    rows = _list_versions(s3=s3, bucket=snapshot.bucket, prefix=prefix)
    if not rows:
        _fail("fresh H.1f namespace is missing genesis")
    if any(row.is_delete_marker for row in rows):
        _fail("fence namespace contains a delete marker")
    by_key: Dict[str, List[S3VersionRow]] = {}
    genesis_key = fence_genesis_s3_key(run_id=snapshot.run_id)
    for row in rows:
        if row.key != genesis_key:
            match = _SUCCESSOR.fullmatch(row.key)
            if match is None or match.group("run") != snapshot.run_id:
                _fail("fence namespace contains an alternate key")
        by_key.setdefault(row.key, []).append(row)
    if any(len(values) != 1 for values in by_key.values()):
        _fail("fence namespace contains version history or siblings")
    if any(not values[0].is_latest for values in by_key.values()):
        _fail("fence namespace contains a stale active version")
    controls = [
        _control(
            _audit_row(s3=s3, bucket=snapshot.bucket, row=row),
            bucket=snapshot.bucket,
            run_id=snapshot.run_id,
        )
        for row in rows
    ]
    genesis = [
        item
        for item in controls
        if item.record.get("record_type")
        == "glm52_sky_production_fence_genesis_v1"
    ]
    if len(genesis) != 1 or genesis[0].object.key != genesis_key:
        _fail("fence namespace genesis is absent or ambiguous")
    successors = [
        item
        for item in controls
        if item.record.get("record_type")
        == "glm52_sky_production_fence_successor_v1"
    ]
    seen_bodies = {genesis[0].body_sha256}
    visited = {genesis[0].object.key}
    current = genesis[0]
    ancestor_live_states = (
        _live_state(
            genesis[0].record.get("live_state"),
            bucket=snapshot.bucket,
        ),
    )
    while True:
        children = [
            item
            for item in successors
            if item.record.get("predecessor_control_body_sha256")
            == current.body_sha256
        ]
        if len(children) > 1:
            _fail("fence namespace contains a fork")
        if not children:
            break
        child = children[0]
        expected_key = fence_successor_s3_key(
            run_id=snapshot.run_id,
            predecessor_body_sha256=current.body_sha256,
        )
        if (
            child.object.key != expected_key
            or child.record.get("successor_key") != expected_key
            or child.record.get("predecessor_control_key") != current.object.key
            or child.record.get("predecessor_control_version_id")
            != current.object.version_id
            or child.record.get("predecessor_control_file_sha256")
            != current.object.file_sha256
            or child.record.get("predecessor_control_body_sha256")
            != current.body_sha256
        ):
            _fail("fence successor predecessor identity drifted")
        _validate_successor_history(
            current,
            child,
            ancestor_live_states=ancestor_live_states,
        )
        if child.body_sha256 in seen_bodies or child.object.key in visited:
            _fail("fence namespace contains a cycle or repeated identity")
        seen_bodies.add(child.body_sha256)
        visited.add(child.object.key)
        current = child
        ancestor_live_states = ancestor_live_states + (
            _live_state(
                child.record.get("live_state"),
                bucket=snapshot.bucket,
            ),
        )
    if len(visited) != len(controls):
        _fail("fence namespace contains orphan history")
    next_key = fence_successor_s3_key(
        run_id=snapshot.run_id,
        predecessor_body_sha256=current.body_sha256,
    )
    if _coordinate_rows(
        s3=s3, bucket=snapshot.bucket, key=next_key
    ):
        _fail("fence head does not have an exact zero-child coordinate")
    return current


def _coordinate_rows(
    *, s3: object, bucket: str, key: str
) -> Tuple[S3VersionRow, ...]:
    rows = _list_versions(s3=s3, bucket=bucket, prefix=key)
    if any(row.key != key for row in rows):
        _fail("exact coordinate prefix contains a sibling key")
    if any(row.is_delete_marker for row in rows):
        _fail("exact coordinate contains a delete marker")
    versions = tuple(row for row in rows if not row.is_delete_marker)
    if len(versions) > 1:
        _fail("exact coordinate contains version history")
    if versions and not versions[0].is_latest:
        _fail("exact coordinate sole version is not current")
    return versions


def reconcile_exact_candidate(
    *,
    s3: object,
    candidate: ImmutableJsonCandidate,
) -> ExactCoordinateResult:
    exact = validate_immutable_json_candidate(candidate)
    versions = _coordinate_rows(
        s3=s3, bucket=exact.bucket, key=exact.key
    )
    if not versions:
        return ExactCoordinateResult("zero", None)
    audited = _audit_row(s3=s3, bucket=exact.bucket, row=versions[0])
    if audited.raw != exact.raw or audited.file_sha256 != exact.file_sha256:
        _fail("exact coordinate bytes do not match candidate")
    identity = build_s3_object_identity(
        candidate=exact,
        version_id=audited.version_id,
        content_length=audited.content_length,
        etag=audited.etag,
        last_modified=audited.last_modified,
        checksum_sha256_base64=audited.checksum_sha256_base64,
        checksum_type=audited.checksum_type,
        content_type=audited.content_type,
        metadata=audited.metadata,
    )
    return ExactCoordinateResult("sole-version", identity)


def _validate_snapshot(value: object) -> H1fAuthoritySnapshot:
    if type(value) is not H1fAuthoritySnapshot:
        raise TypeError("authority reader must return exact H1fAuthoritySnapshot")
    for field in (
        "authority_domain",
        "bucket",
        "run_id",
        "activation_id",
        "execution_arn",
        "active_head_key",
        "active_head_version_id",
    ):
        _string(field, getattr(value, field))
    for field in (
        "barrier_nonce_sha256",
        "active_head_file_sha256",
        "active_head_body_sha256",
    ):
        _sha(field, getattr(value, field))
    for field in ("generation", "epoch", "closing_revision"):
        item = getattr(value, field)
        if type(item) is not int or item < 0:
            _fail(f"{field} must be an exact nonnegative integer")
    if value.generation < 1:
        _fail("generation must be positive")
    if value.authority_domain not in {
        "ACTIVATION",
        "RECOVERY",
        "FINALIZATION",
    }:
        _fail("authority_domain is not a closed H.1f domain")
    return value


def _audit_body(value: H1fAuditResult) -> Mapping[str, object]:
    return {
        "activation_id": value.activation_id,
        "action_key": value.action_key,
        "active_head_body_sha256": value.active_head_body_sha256,
        "active_head_file_sha256": value.active_head_file_sha256,
        "active_head_key": value.active_head_key,
        "active_head_version_id": value.active_head_version_id,
        "authority_domain": value.authority_domain,
        "barrier_nonce_sha256": value.barrier_nonce_sha256,
        "candidate_identity_sha256": value.candidate_identity_sha256,
        "closing_revision": value.closing_revision,
        "epoch": value.epoch,
        "execution_arn": value.execution_arn,
        "expected_authorized_revision": value.expected_authorized_revision,
        "generation": value.generation,
        "operation_kind": value.operation_kind,
        "schema_version": 1,
    }


def validate_h1f_audit_result(value: object) -> H1fAuditResult:
    if type(value) is not H1fAuditResult:
        raise TypeError("audit result must be exact H1fAuditResult")
    for field in (
        "authority_domain",
        "operation_kind",
        "action_key",
        "activation_id",
        "execution_arn",
        "active_head_key",
        "active_head_version_id",
    ):
        _string(field, getattr(value, field))
    for field in (
        "candidate_identity_sha256",
        "barrier_nonce_sha256",
        "active_head_file_sha256",
        "active_head_body_sha256",
        "canonical_body_sha256",
    ):
        _sha(field, getattr(value, field))
    if value.authority_domain not in {
        "ACTIVATION",
        "RECOVERY",
        "FINALIZATION",
    }:
        _fail("audit authority domain is not closed")
    for field in (
        "generation",
        "epoch",
        "closing_revision",
        "expected_authorized_revision",
    ):
        item = getattr(value, field)
        if type(item) is not int or item < 0:
            _fail(f"{field} must be an exact nonnegative integer")
    if (
        value.generation < 1
        or value.expected_authorized_revision != value.closing_revision + 1
    ):
        _fail("audit revision or generation binding is invalid")
    expected = hashlib.sha256(_canonical(_audit_body(value))).hexdigest()
    if value.canonical_body_sha256 != expected:
        _fail("audit canonical body identity drifted")
    return value


class FreshH1fAuditService:
    """Construct a new H.1f observation from current authority every call."""

    def __init__(self, *, authority_reader: object) -> None:
        if not callable(authority_reader):
            raise TypeError("authority_reader must be callable")
        self._authority_reader = authority_reader

    def fresh_audit(
        self,
        *,
        s3: object,
        request: H1fAuditRequest,
    ) -> H1fAuditResult:
        if type(request) is not H1fAuditRequest:
            raise TypeError("request must be exact H1fAuditRequest")
        candidate = validate_immutable_json_candidate(request.candidate)
        operation = _string("operation_kind", request.operation_kind)
        action_key = _string("action_key", request.action_key)
        try:
            snapshot_value = self._authority_reader()
        except Exception as exc:
            raise H1fAuditError("current authority read failed") from exc
        snapshot = _validate_snapshot(snapshot_value)
        if candidate.bucket != snapshot.bucket:
            _fail("candidate bucket does not match current authority")
        candidate_record = _canonical_json_file(candidate.raw)
        if (
            candidate_record.get("run_id") != snapshot.run_id
            or (
                "activation_id" in candidate_record
                and candidate_record.get("activation_id")
                != snapshot.activation_id
            )
            or (
                "generation" in candidate_record
                and candidate_record.get("generation") != snapshot.generation
            )
            or dict(candidate.metadata).get("glm52-activation-id")
            != snapshot.activation_id
            or dict(candidate.metadata).get("glm52-generation-text")
            != f"{snapshot.generation:08d}"
        ):
            _fail("candidate does not match current authority")
        head = _walk_namespace(s3=s3, snapshot=snapshot)
        if (
            head.object.key != snapshot.active_head_key
            or head.object.version_id != snapshot.active_head_version_id
            or head.object.file_sha256 != snapshot.active_head_file_sha256
            or head.body_sha256 != snapshot.active_head_body_sha256
        ):
            _fail("current active-head identity is stale")
        target = reconcile_exact_candidate(s3=s3, candidate=candidate)
        if target.state != "zero":
            _fail("fresh pre-create target coordinate is not exact zero")
        provisional = H1fAuditResult(
            authority_domain=snapshot.authority_domain,
            operation_kind=operation,
            action_key=action_key,
            candidate_identity_sha256=candidate.candidate_identity_sha256,
            activation_id=snapshot.activation_id,
            generation=snapshot.generation,
            epoch=snapshot.epoch,
            execution_arn=snapshot.execution_arn,
            barrier_nonce_sha256=snapshot.barrier_nonce_sha256,
            closing_revision=snapshot.closing_revision,
            expected_authorized_revision=snapshot.closing_revision + 1,
            active_head_key=head.object.key,
            active_head_version_id=head.object.version_id,
            active_head_file_sha256=head.object.file_sha256,
            active_head_body_sha256=head.body_sha256,
            canonical_body_sha256="0" * 64,
        )
        result = H1fAuditResult(
            **{
                **provisional.__dict__,
                "canonical_body_sha256": hashlib.sha256(
                    _canonical(_audit_body(provisional))
                ).hexdigest(),
            }
        )
        return validate_h1f_audit_result(result)


def audit_current_head(
    *,
    s3: object,
    snapshot: H1fAuthoritySnapshot,
    operation_kind: str,
    action_key: str,
    candidate_identity_sha256: str,
) -> H1fAuditResult:
    """Fully walk H.1f and bind an action to the unique current head."""

    exact = _validate_snapshot(snapshot)
    operation = _string("operation_kind", operation_kind)
    action = _string("action_key", action_key)
    candidate_identity = _sha(
        "candidate_identity_sha256",
        candidate_identity_sha256,
    )
    head = _walk_namespace(s3=s3, snapshot=exact)
    if (
        head.object.key != exact.active_head_key
        or head.object.version_id != exact.active_head_version_id
        or head.object.file_sha256 != exact.active_head_file_sha256
        or head.body_sha256 != exact.active_head_body_sha256
    ):
        _fail("current active-head identity is stale")
    provisional = H1fAuditResult(
        authority_domain=exact.authority_domain,
        operation_kind=operation,
        action_key=action,
        candidate_identity_sha256=candidate_identity,
        activation_id=exact.activation_id,
        generation=exact.generation,
        epoch=exact.epoch,
        execution_arn=exact.execution_arn,
        barrier_nonce_sha256=exact.barrier_nonce_sha256,
        closing_revision=exact.closing_revision,
        expected_authorized_revision=exact.closing_revision + 1,
        active_head_key=head.object.key,
        active_head_version_id=head.object.version_id,
        active_head_file_sha256=head.object.file_sha256,
        active_head_body_sha256=head.body_sha256,
        canonical_body_sha256="0" * 64,
    )
    return validate_h1f_audit_result(
        H1fAuditResult(
            **{
                **provisional.__dict__,
                "canonical_body_sha256": hashlib.sha256(
                    _canonical(_audit_body(provisional))
                ).hexdigest(),
            }
        )
    )


__all__ = [
    "ExactCoordinateResult",
    "FreshH1fAuditService",
    "H1fAuditError",
    "H1fAuditRequest",
    "H1fAuditResult",
    "H1fAuthoritySnapshot",
    "S3VersionRow",
    "audit_current_head",
    "reconcile_exact_candidate",
    "validate_h1f_audit_result",
]
