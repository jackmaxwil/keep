"""Injected, read-only exact-version audit for GLM-5.2 fence controls."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from mlx_vq.quality.glm52_sky_production_fence import (
    FenceControlArtifact,
    ModeledFenceHead,
    fence_genesis_s3_key,
    fence_successor_s3_key,
    walk_modeled_fence_chain,
)
from mlx_vq.quality.glm52_sky_production_submission import VersionedJsonArtifact
from mlx_vq.quality.glm52_sky_submission_modes import require_opaque_version_id


class ProductionFenceAuditError(ValueError):
    """The read-only fence observation failed closed."""


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
class AuditedS3Object:
    bucket: str
    key: str
    version_id: str
    raw: bytes
    file_sha256: str
    content_length: int
    etag: str
    last_modified: str
    checksum_sha256_base64: str
    checksum_type: str
    content_type: str
    metadata: tuple[tuple[str, str], ...]
    missing_meta: int


@dataclass(frozen=True)
class ExactCoordinateAudit:
    bucket: str
    key: str
    state: Literal["zero", "sole-version"]
    rows: tuple[S3VersionRow, ...]
    object: Optional[AuditedS3Object]


@dataclass(frozen=True)
class FenceNamespaceAudit:
    bucket: str
    run_id: str
    prefix: str
    rows: tuple[S3VersionRow, ...]
    modeled_head: Optional[ModeledFenceHead]
    observed_zero_coordinate: ExactCoordinateAudit
    observation: Literal["empty", "modeled-chain-with-zero-next-slot"]


@dataclass(frozen=True)
class ProductionFenceAuditServices:
    s3: object


_EXPECTED_OWNER = "246813579024"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GENESIS_RECORD_TYPE = "glm52_sky_production_fence_genesis_v1"
_SUCCESSOR_RECORD_TYPE = "glm52_sky_production_fence_successor_v1"
_SUCCESSOR_KEY = re.compile(
    r"campaigns/(?P<run_id>[A-Za-z0-9][A-Za-z0-9._-]{0,127})/"
    r"authorities/fence/successors/(?P<predecessor>[0-9a-f]{64})/"
    r"FENCE_SUCCESSOR[.]json"
)


def _fail(message: str) -> None:
    raise ProductionFenceAuditError(message)


def _exact_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{field} must be an exact nonempty string")
    return value


def _exact_sha256(value: object, *, field: str) -> str:
    text = _exact_string(value, field=field)
    if _SHA256.fullmatch(text) is None:
        _fail(f"{field} must be exact lowercase SHA-256")
    return text


def _plain_mapping(value: object, *, field: str) -> dict[object, object]:
    try:
        is_mapping = isinstance(value, Mapping)
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(f"{field} access failed") from None
    if not is_mapping:
        _fail(f"{field} must be a mapping")
    try:
        return dict(value)
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(f"{field} access failed") from None


def _opaque_version(value: object, *, field: str) -> str:
    try:
        return require_opaque_version_id(value, field=field)
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(
            f"{field} is not an exact opaque VersionId"
        ) from None


def _canonical_time(value: object, *, field: str) -> str:
    if type(value) is not datetime:
        _fail(f"{field} must be an exact datetime")
    try:
        offset = value.utcoffset()
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(f"{field} timezone is invalid") from None
    if offset is None or value.microsecond != 0:
        _fail(f"{field} must be timezone-aware whole-second time")
    try:
        normalized = value.astimezone(timezone.utc)
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(f"{field} cannot normalize to UTC") from None
    if normalized.microsecond != 0:
        _fail(f"{field} must normalize to a whole second")
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _metadata(value: object, *, field: str) -> tuple[tuple[str, str], ...]:
    plain = _plain_mapping(value, field=field)
    pairs: list[tuple[str, str]] = []
    for key, item in plain.items():
        if (
            type(key) is not str
            or not key
            or key != key.lower()
            or type(item) is not str
        ):
            _fail(f"{field} contains invalid metadata")
        pairs.append((key, item))
    pairs.sort()
    if len({key for key, _ in pairs}) != len(pairs):
        _fail(f"{field} contains duplicate metadata keys")
    return tuple(pairs)


def _expected_metadata(
    value: object,
) -> Optional[tuple[tuple[str, str], ...]]:
    if value is None:
        return None
    if type(value) is not tuple:
        _fail("expected_metadata must be an exact tuple")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            or not item[0]
            or item[0] != item[0].lower()
        ):
            _fail("expected_metadata contains an invalid member")
        pairs.append((item[0], item[1]))
    if pairs != sorted(pairs) or len({key for key, _ in pairs}) != len(pairs):
        _fail("expected_metadata must be sorted and unique")
    return tuple(pairs)


def _response_status(response: object, *, operation: str) -> dict[object, object]:
    normalized = _plain_mapping(
        response,
        field=f"S3 {operation} response",
    )
    metadata = _plain_mapping(
        normalized.get("ResponseMetadata"),
        field=f"S3 {operation} ResponseMetadata",
    )
    if (
        type(metadata.get("HTTPStatusCode")) is not int
        or metadata.get("HTTPStatusCode") != 200
    ):
        _fail(f"S3 {operation} status is not exact 200")
    return normalized


def _call_s3(
    service: object,
    method_name: str,
    arguments: dict[str, object],
    *,
    operation: str,
) -> object:
    try:
        method = getattr(service, method_name, None)
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(
            f"S3 {operation} method lookup failed"
        ) from None
    if not callable(method):
        _fail(f"S3 service lacks {operation}")
    try:
        return method(**arguments)
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(f"S3 {operation} failed") from None


def _list_versions(
    *,
    services: ProductionFenceAuditServices,
    bucket: str,
    prefix: str,
    expected_bucket_owner: str,
) -> tuple[S3VersionRow, ...]:
    service = services.s3
    request_markers: Optional[tuple[str, str]] = None
    seen_pairs: set[tuple[str, str]] = set()
    seen_rows: set[tuple[str, str, bool]] = set()
    rows: list[S3VersionRow] = []
    while True:
        arguments: dict[str, object] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "ExpectedBucketOwner": expected_bucket_owner,
        }
        if request_markers is not None:
            arguments["KeyMarker"] = request_markers[0]
            arguments["VersionIdMarker"] = request_markers[1]
        response = _response_status(
            _call_s3(
                service,
                "list_object_versions",
                arguments,
                operation="ListObjectVersions",
            ),
            operation="ListObjectVersions",
        )
        truncated = response.get("IsTruncated")
        if type(truncated) is not bool:
            _fail("ListObjectVersions IsTruncated must be an exact bool")
        for collection_name in ("Versions", "DeleteMarkers", "CommonPrefixes"):
            if (
                collection_name in response
                and type(response[collection_name]) is not list
            ):
                _fail(f"ListObjectVersions {collection_name} must be an exact list")
        versions = response.get("Versions", [])
        delete_markers = response.get("DeleteMarkers", [])
        common_prefixes = response.get("CommonPrefixes", [])
        if common_prefixes:
            _fail("ListObjectVersions returned a common prefix")
        if "Name" in response and (
            type(response["Name"]) is not str or response["Name"] != bucket
        ):
            _fail("ListObjectVersions bucket identity drifted")
        if "Prefix" in response and (
            type(response["Prefix"]) is not str or response["Prefix"] != prefix
        ):
            _fail("ListObjectVersions prefix identity drifted")

        present_key = "KeyMarker" in response
        present_version = "VersionIdMarker" in response
        if present_key != present_version:
            _fail("ListObjectVersions response marker pair is partial")
        if request_markers is None:
            if present_key and (
                type(response["KeyMarker"]) is not str
                or type(response["VersionIdMarker"]) is not str
                or response["KeyMarker"] != ""
                or response["VersionIdMarker"] != ""
            ):
                _fail("initial ListObjectVersions marker echo is invalid")
        else:
            if (
                not present_key
                or type(response["KeyMarker"]) is not str
                or type(response["VersionIdMarker"]) is not str
                or response["KeyMarker"] != request_markers[0]
                or response["VersionIdMarker"] != request_markers[1]
            ):
                _fail("ListObjectVersions request marker echo is invalid")

        for collection, is_delete_marker in (
            (versions, False),
            (delete_markers, True),
        ):
            for item in collection:
                item = _plain_mapping(
                    item,
                    field="ListObjectVersions row",
                )
                key = _exact_string(
                    item.get("Key"),
                    field="ListObjectVersions row Key",
                )
                if not key.startswith(prefix):
                    _fail("ListObjectVersions row has a foreign prefix")
                version_id = _opaque_version(
                    item.get("VersionId"),
                    field="ListObjectVersions row VersionId",
                )
                is_latest = item.get("IsLatest")
                if type(is_latest) is not bool:
                    _fail("ListObjectVersions IsLatest must be an exact bool")
                last_modified = _canonical_time(
                    item.get("LastModified"),
                    field="ListObjectVersions LastModified",
                )
                if is_delete_marker:
                    if "ETag" in item or "Size" in item:
                        _fail("delete-marker row must omit ETag and Size")
                    etag: Optional[str] = None
                    size: Optional[int] = None
                else:
                    etag = _exact_string(
                        item.get("ETag"),
                        field="ListObjectVersions ETag",
                    )
                    size_value = item.get("Size")
                    if type(size_value) is not int or size_value < 0:
                        _fail("ListObjectVersions Size must be an exact integer")
                    size = size_value
                identity = (key, version_id, is_delete_marker)
                if identity in seen_rows:
                    _fail("ListObjectVersions returned a duplicate row")
                seen_rows.add(identity)
                rows.append(
                    S3VersionRow(
                        key=key,
                        version_id=version_id,
                        is_latest=is_latest,
                        is_delete_marker=is_delete_marker,
                        etag=etag,
                        size=size,
                        last_modified=last_modified,
                    )
                )

        next_key_present = "NextKeyMarker" in response
        next_version_present = "NextVersionIdMarker" in response
        if next_key_present != next_version_present:
            _fail("ListObjectVersions next marker pair is partial")
        if truncated:
            if not next_key_present:
                _fail("truncated ListObjectVersions page lacks markers")
            next_key = _exact_string(
                response["NextKeyMarker"],
                field="ListObjectVersions NextKeyMarker",
            )
            next_version = _opaque_version(
                response["NextVersionIdMarker"],
                field="ListObjectVersions NextVersionIdMarker",
            )
            if not next_key.startswith(prefix):
                _fail("ListObjectVersions next key marker has a foreign prefix")
            pair = (next_key, next_version)
            if pair == request_markers or pair in seen_pairs:
                _fail("ListObjectVersions marker pair repeated or cycled")
            seen_pairs.add(pair)
            request_markers = pair
            continue
        if next_key_present and (
            type(response["NextKeyMarker"]) is not str
            or type(response["NextVersionIdMarker"]) is not str
            or response["NextKeyMarker"] != ""
            or response["NextVersionIdMarker"] != ""
        ):
            _fail("terminal ListObjectVersions page has leftover markers")
        return tuple(rows)


def _delete_marker_absent(response: Mapping[str, object], *, operation: str) -> None:
    if "DeleteMarker" not in response:
        return
    value = response["DeleteMarker"]
    if type(value) is not bool or value:
        _fail(f"S3 {operation} reported a delete marker")


def _missing_meta(response: Mapping[str, object], *, operation: str) -> int:
    value = response.get("MissingMeta", 0)
    if type(value) is not int or value != 0:
        _fail(f"S3 {operation} MissingMeta must be exact zero")
    return value


def _transport_identity(
    response: Mapping[str, object],
    *,
    operation: str,
    expected_version_id: str,
) -> tuple[
    str,
    int,
    str,
    str,
    str,
    str,
    tuple[tuple[str, str], ...],
    int,
]:
    version_id = _opaque_version(
        response.get("VersionId"),
        field=f"S3 {operation} VersionId",
    )
    if version_id != expected_version_id:
        _fail(f"S3 {operation} VersionId drifted")
    _delete_marker_absent(response, operation=operation)
    etag = _exact_string(response.get("ETag"), field=f"S3 {operation} ETag")
    length = response.get("ContentLength")
    if type(length) is not int or length < 0:
        _fail(f"S3 {operation} ContentLength is invalid")
    last_modified = _canonical_time(
        response.get("LastModified"),
        field=f"S3 {operation} LastModified",
    )
    checksum = _exact_string(
        response.get("ChecksumSHA256"),
        field=f"S3 {operation} ChecksumSHA256",
    )
    checksum_type = _exact_string(
        response.get("ChecksumType"),
        field=f"S3 {operation} ChecksumType",
    )
    if checksum_type != "FULL_OBJECT":
        _fail(f"S3 {operation} checksum type is not FULL_OBJECT")
    content_type = _exact_string(
        response.get("ContentType"),
        field=f"S3 {operation} ContentType",
    )
    metadata = _metadata(
        response.get("Metadata"),
        field=f"S3 {operation} Metadata",
    )
    missing_meta = _missing_meta(response, operation=operation)
    return (
        etag,
        length,
        last_modified,
        checksum,
        checksum_type,
        content_type,
        metadata,
        missing_meta,
    )


def _read_get_body(
    response: object,
    *,
    expected_version_id: str,
) -> tuple[
    bytes,
    tuple[
        str,
        int,
        str,
        str,
        str,
        str,
        tuple[tuple[str, str], ...],
        int,
    ],
]:
    try:
        body = response.get("Body") if isinstance(response, Mapping) else None
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(
            "S3 GetObject body access failed"
        ) from None
    try:
        closer = getattr(body, "close", None)
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(
            "S3 GetObject body access failed"
        ) from None
    primary_error: Optional[BaseException] = None
    raw: object = None
    identity: object = None
    try:
        try:
            reader = getattr(body, "read", None)
        except MemoryError:
            raise
        except Exception:
            raise ProductionFenceAuditError(
                "S3 GetObject body read lookup failed"
            ) from None
        detached_response = _plain_mapping(
            response,
            field="S3 GetObject response",
        )
        if detached_response.get("Body") is not body:
            _fail("S3 GetObject Body identity drifted")
        normalized_response = _response_status(
            detached_response,
            operation="GetObject",
        )
        if not callable(reader) or not callable(closer):
            _fail("S3 GetObject body lacks callable read and close")
        identity = _transport_identity(
            normalized_response,
            operation="GetObject",
            expected_version_id=expected_version_id,
        )
        try:
            raw = reader()
        except MemoryError:
            raise
        except Exception:
            raise ProductionFenceAuditError(
                "S3 GetObject body read failed"
            ) from None
        if type(raw) is not bytes:
            _fail("S3 GetObject body did not return exact bytes")
    except BaseException as error:
        primary_error = error
    close_error: Optional[BaseException] = None
    if callable(closer):
        try:
            closer()
        except MemoryError as error:
            close_error = error
        except Exception:
            close_error = ProductionFenceAuditError(
                "S3 GetObject body close failed"
            )
        except BaseException as error:
            close_error = error
    if close_error is not None and (
        isinstance(close_error, MemoryError)
        or not isinstance(close_error, Exception)
        or primary_error is None
    ):
        raise close_error
    if primary_error is not None:
        raise primary_error
    if close_error is not None:
        raise close_error
    assert type(raw) is bytes
    assert isinstance(identity, tuple)
    return raw, identity


def _audit_sole_row(
    *,
    services: ProductionFenceAuditServices,
    bucket: str,
    row: S3VersionRow,
    expected_bucket_owner: str,
) -> AuditedS3Object:
    arguments = {
        "Bucket": bucket,
        "Key": row.key,
        "VersionId": row.version_id,
        "ChecksumMode": "ENABLED",
        "ExpectedBucketOwner": expected_bucket_owner,
    }
    get_response = _call_s3(
        services.s3,
        "get_object",
        dict(arguments),
        operation="GetObject",
    )
    raw, get_identity = _read_get_body(
        get_response,
        expected_version_id=row.version_id,
    )
    head_response = _response_status(
        _call_s3(
            services.s3,
            "head_object",
            dict(arguments),
            operation="HeadObject",
        ),
        operation="HeadObject",
    )
    head_identity = _transport_identity(
        head_response,
        operation="HeadObject",
        expected_version_id=row.version_id,
    )
    if get_identity != head_identity:
        _fail("S3 exact GET and HEAD transport identities disagree")
    (
        etag,
        length,
        last_modified,
        checksum,
        checksum_type,
        content_type,
        metadata,
        missing_meta,
    ) = get_identity
    digest = hashlib.sha256(raw).digest()
    file_sha256 = digest.hex()
    computed_checksum = base64.b64encode(digest).decode("ascii")
    if length != len(raw):
        _fail("S3 exact object content length disagrees with bytes")
    if checksum != computed_checksum:
        _fail("S3 exact object checksum disagrees with bytes")
    if (
        row.etag != etag
        or row.size != length
        or row.last_modified != last_modified
    ):
        _fail("S3 exact object disagrees with selected version row")
    return AuditedS3Object(
        bucket=bucket,
        key=row.key,
        version_id=row.version_id,
        raw=raw,
        file_sha256=file_sha256,
        content_length=length,
        etag=etag,
        last_modified=last_modified,
        checksum_sha256_base64=checksum,
        checksum_type=checksum_type,
        content_type=content_type,
        metadata=metadata,
        missing_meta=missing_meta,
    )


def _validate_common_inputs(
    *,
    services: object,
    bucket: object,
    expected_bucket_owner: object,
) -> tuple[ProductionFenceAuditServices, str, str]:
    if type(services) is not ProductionFenceAuditServices:
        _fail("services must be exact ProductionFenceAuditServices")
    exact_bucket = _exact_string(bucket, field="bucket")
    if (
        type(expected_bucket_owner) is not str
        or expected_bucket_owner != _EXPECTED_OWNER
    ):
        _fail("expected_bucket_owner must equal the approved account")
    return services, exact_bucket, expected_bucket_owner


def _audit_coordinate_operation(
    *,
    services: ProductionFenceAuditServices,
    bucket: str,
    key: str,
    expected_bucket_owner: str,
    expected_raw: Optional[bytes],
    expected_file_sha256: Optional[str],
    expected_content_type: Optional[str],
    expected_metadata: Optional[tuple[tuple[str, str], ...]],
) -> ExactCoordinateAudit:
    exact_key = _exact_string(key, field="key")
    if (expected_raw is None) != (expected_file_sha256 is None):
        _fail("expected_raw and expected_file_sha256 must be supplied together")
    if expected_raw is not None:
        if type(expected_raw) is not bytes:
            _fail("expected_raw must be exact bytes")
        expected_digest = _exact_sha256(
            expected_file_sha256,
            field="expected_file_sha256",
        )
        if hashlib.sha256(expected_raw).hexdigest() != expected_digest:
            _fail("expected_raw and expected_file_sha256 disagree")
    if expected_content_type is not None and (
        type(expected_content_type) is not str or not expected_content_type
    ):
        _fail("expected_content_type must be an exact nonempty string")
    exact_expected_metadata = _expected_metadata(expected_metadata)
    rows = _list_versions(
        services=services,
        bucket=bucket,
        prefix=exact_key,
        expected_bucket_owner=expected_bucket_owner,
    )
    if any(row.key != exact_key for row in rows):
        _fail("exact coordinate prefix contains a sibling key")
    if any(row.is_delete_marker for row in rows):
        _fail("exact coordinate contains a delete marker")
    versions = tuple(row for row in rows if not row.is_delete_marker)
    if len(versions) > 1:
        _fail("exact coordinate contains version history")
    if not versions:
        if (
            expected_raw is not None
            or expected_content_type is not None
            or exact_expected_metadata is not None
        ):
            _fail("zero coordinate cannot satisfy expected object identity")
        return ExactCoordinateAudit(
            bucket=bucket,
            key=exact_key,
            state="zero",
            rows=rows,
            object=None,
        )
    row = versions[0]
    if not row.is_latest:
        _fail("sole coordinate version is not latest")
    audited = _audit_sole_row(
        services=services,
        bucket=bucket,
        row=row,
        expected_bucket_owner=expected_bucket_owner,
    )
    if expected_raw is not None and (
        audited.raw != expected_raw
        or audited.file_sha256 != expected_file_sha256
    ):
        _fail("exact object does not match expected raw identity")
    if (
        expected_content_type is not None
        and audited.content_type != expected_content_type
    ):
        _fail("exact object content type does not match expected value")
    if (
        exact_expected_metadata is not None
        and audited.metadata != exact_expected_metadata
    ):
        _fail("exact object metadata does not match expected value")
    return ExactCoordinateAudit(
        bucket=bucket,
        key=exact_key,
        state="sole-version",
        rows=rows,
        object=audited,
    )


def _json_record(raw: bytes) -> dict[str, object]:
    try:
        text = raw.decode("ascii")
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError(
            "fence control is not exact ASCII JSON"
        ) from None

    def reject_constant(_value: str) -> object:
        _fail("fence control contains a non-finite JSON number")

    def unique_object(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail("fence control contains a duplicate JSON member")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except MemoryError:
        raise
    except ProductionFenceAuditError:
        raise
    except Exception:
        raise ProductionFenceAuditError(
            "fence control is not valid JSON"
        ) from None
    if type(value) is not dict:
        _fail("fence control must be an exact JSON object")
    return value


def _control_artifact(
    *,
    audited: AuditedS3Object,
    bucket: str,
    run_id: str,
) -> FenceControlArtifact:
    record = _json_record(audited.raw)
    record_type = _exact_string(
        record.get("record_type"),
        field="fence control record_type",
    )
    if record_type not in {_GENESIS_RECORD_TYPE, _SUCCESSOR_RECORD_TYPE}:
        _fail("fence control record type is unknown")
    control_run_id = _exact_string(
        record.get("run_id"),
        field="fence control run_id",
    )
    if control_run_id != run_id:
        _fail("fence control run identity drifted")
    control_bucket = _exact_string(
        record.get("bucket"),
        field="fence control bucket",
    )
    if control_bucket != bucket:
        _fail("fence control bucket identity drifted")
    body_sha256 = _exact_sha256(
        record.get("fence_body_sha256"),
        field="fence control body SHA-256",
    )
    campaign_identity_sha256 = _exact_sha256(
        record.get("campaign_identity_sha256"),
        field="fence control campaign identity SHA-256",
    )
    expected_metadata = tuple(
        sorted(
            (
                ("glm52-body-sha256", body_sha256),
                (
                    "glm52-campaign-identity-sha256",
                    campaign_identity_sha256,
                ),
                ("glm52-file-sha256", audited.file_sha256),
                ("glm52-record-type", record_type),
                ("glm52-run-id", run_id),
            )
        )
    )
    if audited.content_type != "application/json":
        _fail("fence control content type is not application/json")
    if audited.metadata != expected_metadata:
        _fail("fence control transport metadata is not exact")
    return FenceControlArtifact(
        key=audited.key,
        raw=audited.raw,
        version_id=audited.version_id,
        file_sha256=audited.file_sha256,
        body_sha256=body_sha256,
        content_length=audited.content_length,
        etag=audited.etag,
        last_modified=audited.last_modified,
        checksum_sha256_base64=audited.checksum_sha256_base64,
        checksum_type=audited.checksum_type,
        content_type=audited.content_type,
        metadata=audited.metadata,
        missing_meta=audited.missing_meta,
    )


def _audit_fence_namespace_operation(
    *,
    services: ProductionFenceAuditServices,
    bucket: str,
    run_id: str,
    expected_bucket_owner: str,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
) -> FenceNamespaceAudit:
    genesis_key = fence_genesis_s3_key(run_id=run_id)
    suffix = "FENCE_GENESIS.json"
    if not genesis_key.endswith(suffix):
        _fail("canonical fence genesis key is malformed")
    prefix = genesis_key[: -len(suffix)]
    rows = _list_versions(
        services=services,
        bucket=bucket,
        prefix=prefix,
        expected_bucket_owner=expected_bucket_owner,
    )
    if not rows:
        observed_zero = audit_exact_coordinate(
            services=services,
            bucket=bucket,
            key=genesis_key,
            expected_bucket_owner="246813579024",
        )
        if observed_zero.state != "zero":
            _fail("empty fence namespace genesis coordinate is not zero")
        return FenceNamespaceAudit(
            bucket=bucket,
            run_id=run_id,
            prefix=prefix,
            rows=rows,
            modeled_head=None,
            observed_zero_coordinate=observed_zero,
            observation="empty",
        )

    versions_by_key: dict[str, list[S3VersionRow]] = {}
    for row in rows:
        if row.is_delete_marker:
            _fail("fence namespace contains a delete marker")
        if row.key != genesis_key:
            match = _SUCCESSOR_KEY.fullmatch(row.key)
            if (
                match is None
                or match.group("run_id") != run_id
                or row.key
                != fence_successor_s3_key(
                    run_id=run_id,
                    predecessor_body_sha256=match.group("predecessor"),
                )
            ):
                _fail("fence namespace contains an alternate control key")
        versions_by_key.setdefault(row.key, []).append(row)
    for key_rows in versions_by_key.values():
        if len(key_rows) != 1:
            _fail("fence control coordinate has version history")
        if not key_rows[0].is_latest:
            _fail("sole fence control version is not latest")

    controls: list[FenceControlArtifact] = []
    for row in rows:
        audited = _audit_sole_row(
            services=services,
            bucket=bucket,
            row=row,
            expected_bucket_owner=expected_bucket_owner,
        )
        controls.append(
            _control_artifact(
                audited=audited,
                bucket=bucket,
                run_id=run_id,
            )
        )

    modeled_head = walk_modeled_fence_chain(
        run_id=run_id,
        controls=tuple(controls),
        descriptor=descriptor,
        intent=intent,
        approval=approval,
        controller_baseline=controller_baseline,
        must_start_control_plane_ready=must_start_control_plane_ready,
        submission_acquisition=submission_acquisition,
        gpu_spend_snapshot=gpu_spend_snapshot,
    )
    observed_zero = audit_exact_coordinate(
        services=services,
        bucket=bucket,
        key=modeled_head.next_successor_key,
        expected_bucket_owner="246813579024",
    )
    if observed_zero.state != "zero":
        _fail("reached fence head next-successor coordinate is not zero")
    return FenceNamespaceAudit(
        bucket=bucket,
        run_id=run_id,
        prefix=prefix,
        rows=rows,
        modeled_head=modeled_head,
        observed_zero_coordinate=observed_zero,
        observation="modeled-chain-with-zero-next-slot",
    )


def audit_exact_coordinate(
    *,
    services: ProductionFenceAuditServices,
    bucket: str,
    key: str,
    expected_bucket_owner: Literal["246813579024"],
    expected_raw: Optional[bytes] = None,
    expected_file_sha256: Optional[str] = None,
    expected_content_type: Optional[str] = None,
    expected_metadata: Optional[tuple[tuple[str, str], ...]] = None,
) -> ExactCoordinateAudit:
    try:
        exact_services, exact_bucket, exact_owner = _validate_common_inputs(
            services=services,
            bucket=bucket,
            expected_bucket_owner=expected_bucket_owner,
        )
        return _audit_coordinate_operation(
            services=exact_services,
            bucket=exact_bucket,
            key=key,
            expected_bucket_owner=exact_owner,
            expected_raw=expected_raw,
            expected_file_sha256=expected_file_sha256,
            expected_content_type=expected_content_type,
            expected_metadata=expected_metadata,
        )
    except ProductionFenceAuditError:
        raise
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError("exact coordinate audit failed") from None


def audit_fence_namespace(
    *,
    services: ProductionFenceAuditServices,
    bucket: str,
    run_id: str,
    expected_bucket_owner: Literal["246813579024"],
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
) -> FenceNamespaceAudit:
    try:
        exact_services, exact_bucket, exact_owner = _validate_common_inputs(
            services=services,
            bucket=bucket,
            expected_bucket_owner=expected_bucket_owner,
        )
        exact_run_id = _exact_string(run_id, field="run_id")
        return _audit_fence_namespace_operation(
            services=exact_services,
            bucket=exact_bucket,
            run_id=exact_run_id,
            expected_bucket_owner=exact_owner,
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            gpu_spend_snapshot=gpu_spend_snapshot,
        )
    except ProductionFenceAuditError:
        raise
    except MemoryError:
        raise
    except Exception:
        raise ProductionFenceAuditError("fence namespace audit failed") from None


__all__ = [
    "AuditedS3Object",
    "ExactCoordinateAudit",
    "FenceNamespaceAudit",
    "ProductionFenceAuditError",
    "ProductionFenceAuditServices",
    "S3VersionRow",
    "audit_exact_coordinate",
    "audit_fence_namespace",
]
