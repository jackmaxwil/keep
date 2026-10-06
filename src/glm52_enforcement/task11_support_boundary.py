"""Closed immutable inputs and wire results for Task 11 support callees."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Mapping, Tuple

from .canonical import canonical_json_bytes, canonical_sha256
from .decision_closure import (
    AUTHORITY_AUDIT_KINDS,
    SourcePublication,
    build_audit_evidence,
    build_source_publication,
)
from .source_authorities import VersionedJsonArtifact


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^[\x21-\x7e]{1,1024}$")
_SOURCE_KINDS = (
    "GPU_SPEND",
    "SUBMISSION_INTENT",
    "CONTROLLER_BASELINE",
    "CONTROL_PLANE_READINESS",
    "SUBMISSION_ACQUISITION",
)
_INPUT_KINDS = tuple(kind + "_SOURCE_REQUEST" for kind in _SOURCE_KINDS)
_DIGEST_FIELDS = {
    "GPU_SPEND": "snapshot_body_sha256",
    "SUBMISSION_INTENT": "intent_body_sha256",
    "CONTROLLER_BASELINE": "baseline_body_sha256",
    "CONTROL_PLANE_READINESS": "control_plane_ready_body_sha256",
    "SUBMISSION_ACQUISITION": "acquisition_body_sha256",
}
_PREDECESSOR_FIELDS = {
    "GPU_SPEND": (),
    "SUBMISSION_INTENT": (
        ("gpu_spend_snapshot", 0),
    ),
    "CONTROLLER_BASELINE": (
        ("intent", 1),
    ),
    "CONTROL_PLANE_READINESS": (
        ("intent", 1),
        ("controller_baseline", 2),
    ),
    "SUBMISSION_ACQUISITION": (
        ("intent", 1),
        ("controller_baseline", 2),
        ("must_start_control_plane_ready", 3),
    ),
}


class Task11SupportBoundaryError(ValueError):
    """An immutable support input or exact callee response drifted."""


@dataclass(frozen=True)
class ExactInputCoordinate:
    input_kind: str
    bucket: str
    key: str
    version_id: str
    file_sha256: str
    body_sha256: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class SourceInputDocument:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    generation: int
    source_kind: str
    action_key: str
    source_template: Mapping[str, object]
    descriptor: Mapping[str, object] | None
    canonical_identity_sha256: str


def _fail(message: str) -> None:
    raise Task11SupportBoundaryError(message)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail(label + " must be a lowercase SHA-256")
    return value


def _coordinate_body(value: ExactInputCoordinate) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def exact_input_coordinate_from_mapping(
    value: object,
    *,
    expected_kind: str,
) -> ExactInputCoordinate:
    if (
        type(value) is not dict
        or set(value) != set(ExactInputCoordinate.__dataclass_fields__)
    ):
        _fail("support input coordinate field set drifted")
    try:
        result = ExactInputCoordinate(**value)
    except TypeError as exc:
        raise Task11SupportBoundaryError(
            "support input coordinate is malformed"
        ) from exc
    if (
        result.input_kind != expected_kind
        or type(result.bucket) is not str
        or not result.bucket
        or type(result.key) is not str
        or not result.key
        or type(result.version_id) is not str
        or _VERSION.fullmatch(result.version_id) is None
        or result.version_id == "null"
    ):
        _fail("support input coordinate identity drifted")
    _sha(result.file_sha256, "support input file identity")
    _sha(result.body_sha256, "support input body identity")
    if result.canonical_identity_sha256 != canonical_sha256(
        _coordinate_body(result)
    ):
        _fail("support input coordinate self-hash drifted")
    return result


def _transport(value: object, operation: str) -> Mapping[str, object]:
    metadata = value.get("ResponseMetadata") if type(value) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        _fail(operation + " response is unauthenticated")
    return value


def load_exact_input(
    *,
    s3: object,
    coordinate: ExactInputCoordinate,
) -> Mapping[str, object]:
    get_object = getattr(s3, "get_object", None)
    if not callable(get_object):
        _fail("support input S3 reader is absent")
    value = _transport(
        get_object(
            Bucket=coordinate.bucket,
            Key=coordinate.key,
            VersionId=coordinate.version_id,
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        "GetObject",
    )
    stream = value.get("Body")
    read = getattr(stream, "read", None)
    if (
        value.get("VersionId") != coordinate.version_id
        or "DeleteMarker" in value
        or type(value.get("ChecksumSHA256")) is not str
        or not callable(read)
    ):
        _fail("support input version response drifted")
    raw = read(2 * 1024 * 1024 + 1)
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    if (
        type(raw) is not bytes
        or len(raw) > 2 * 1024 * 1024
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or hashlib.sha256(raw).hexdigest() != coordinate.file_sha256
        or value["ChecksumSHA256"] != checksum
    ):
        _fail("support input file/checksum identity drifted")
    try:
        parsed = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Task11SupportBoundaryError(
            "support input is invalid JSON"
        ) from exc
    if (
        type(parsed) is not dict
        or canonical_json_bytes(parsed) + b"\n" != raw
        or parsed.get("canonical_identity_sha256")
        != coordinate.body_sha256
    ):
        _fail("support input canonical identity drifted")
    body = dict(parsed)
    identity = body.pop("canonical_identity_sha256")
    if identity != canonical_sha256(body):
        _fail("support input self-hash drifted")
    return parsed


def _artifact(value: object, label: str) -> VersionedJsonArtifact:
    if type(value) is not dict or set(value) != {
        "key",
        "version_id",
        "record",
    }:
        _fail(label + " artifact field set drifted")
    if (
        type(value["key"]) is not str
        or not value["key"]
        or type(value["version_id"]) is not str
        or _VERSION.fullmatch(value["version_id"]) is None
        or value["version_id"] == "null"
        or type(value["record"]) is not dict
    ):
        _fail(label + " artifact identity drifted")
    return VersionedJsonArtifact(
        key=value["key"],
        version_id=value["version_id"],
        raw=canonical_json_bytes(value["record"]) + b"\n",
    )


def source_input_from_mapping(
    value: object,
    *,
    source_kind: str,
    activation_id: str,
    generation: int,
) -> SourceInputDocument:
    if (
        source_kind not in _SOURCE_KINDS
        or type(value) is not dict
        or set(value) != set(SourceInputDocument.__dataclass_fields__)
    ):
        _fail("source input field set drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256")
    if (
        value["schema_version"] != 1
        or value["record_type"] != "glm52_task11_source_input_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
        or value["activation_id"] != activation_id
        or value["generation"] != generation
        or value["source_kind"] != source_kind
        or type(value["action_key"]) is not str
        or "S3_CREATE" not in value["action_key"]
        or type(value["source_template"]) is not dict
        or type(identity) is not str
        or identity != canonical_sha256(body)
    ):
        _fail("source input identity drifted")
    descriptor = value["descriptor"]
    if source_kind == "GPU_SPEND":
        if descriptor is not None:
            _fail("GPU spend input invented a descriptor")
    else:
        _artifact(descriptor, "descriptor")
    for prefix, _ in _PREDECESSOR_FIELDS[source_kind]:
        for suffix in ("key", "file_sha256", "body_sha256", "version_id"):
            field = prefix + "_" + suffix
            if value["source_template"].get(field) != "$" + field:
                _fail("source predecessor placeholder drifted")
    try:
        return SourceInputDocument(**value)
    except TypeError as exc:
        raise Task11SupportBoundaryError(
            "source input is malformed"
        ) from exc


def _publication_artifact(value: SourcePublication) -> VersionedJsonArtifact:
    # The next accepted source validator requires the predecessor bytes.  The
    # exact bytes are carried only invocation-locally by the support response.
    raw = getattr(value, "_task11_raw", None)
    if type(raw) is not bytes:
        _fail("source predecessor bytes left invocation-local custody")
    return VersionedJsonArtifact(
        key=value.key,
        version_id=value.version_id,
        raw=raw,
    )


def materialize_source_input(
    document: SourceInputDocument,
    *,
    prior_publications: Tuple[SourcePublication, ...],
) -> tuple[bytes, Mapping[str, VersionedJsonArtifact]]:
    source_kind = document.source_kind
    index = _SOURCE_KINDS.index(source_kind)
    if (
        len(prior_publications) != index
        or tuple(item.source_kind for item in prior_publications)
        != _SOURCE_KINDS[:index]
    ):
        _fail("source predecessor order drifted")
    record = dict(document.source_template)
    artifacts: dict[str, VersionedJsonArtifact] = {}
    if document.descriptor is not None:
        artifacts["descriptor"] = _artifact(
            document.descriptor,
            "descriptor",
        )
    for prefix, predecessor_index in _PREDECESSOR_FIELDS[source_kind]:
        predecessor = prior_publications[predecessor_index]
        record.update(
            {
                prefix + "_key": predecessor.key,
                prefix + "_file_sha256": predecessor.file_sha256,
                prefix + "_body_sha256": predecessor.body_sha256,
                prefix + "_version_id": predecessor.version_id,
            }
        )
        artifact_name = {
            "gpu_spend_snapshot": "gpu_spend_snapshot",
            "intent": "intent",
            "controller_baseline": "controller_baseline",
            "must_start_control_plane_ready": (
                "must_start_control_plane_ready"
            ),
        }[prefix]
        artifacts[artifact_name] = _publication_artifact(predecessor)
    digest_field = _DIGEST_FIELDS[source_kind]
    record[digest_field] = canonical_sha256(
        {
            field: value
            for field, value in record.items()
            if field != digest_field
        }
    )
    return canonical_json_bytes(record) + b"\n", artifacts


def source_publication_to_payload(
    value: SourcePublication,
    *,
    raw: bytes,
) -> Mapping[str, object]:
    if type(value) is not SourcePublication:
        raise TypeError("source publication must be exact and typed")
    return {
        "schema_version": 1,
        "record_type": "glm52_task11_source_publication_response_v1",
        "publication": asdict(value),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
    }


def source_publication_from_payload(
    value: object,
    *,
    expected_source_kind: str,
    expected_predecessor_version_id: str,
) -> SourcePublication:
    if (
        type(value) is not dict
        or set(value) != {
            "schema_version",
            "record_type",
            "publication",
            "raw_base64",
        }
        or value["schema_version"] != 1
        or value["record_type"]
        != "glm52_task11_source_publication_response_v1"
        or type(value["publication"]) is not dict
        or type(value["raw_base64"]) is not str
    ):
        _fail("source publication response field set drifted")
    publication = value["publication"]
    try:
        audit_value = publication["audit"]
        audit = build_audit_evidence(
            kind=audit_value["kind"],
            audit_identity_sha256=audit_value["audit_identity_sha256"],
            invocation_identity_sha256=(
                audit_value["invocation_identity_sha256"]
            ),
            closing_revision=audit_value["closing_revision"],
        )
        result = build_source_publication(
            source_kind=publication["source_kind"],
            predecessor_version_id=publication[
                "predecessor_version_id"
            ],
            bucket=publication["bucket"],
            key=publication["key"],
            version_id=publication["version_id"],
            file_sha256=publication["file_sha256"],
            body_sha256=publication["body_sha256"],
            etag=publication["etag"],
            checksum_sha256_base64=publication[
                "checksum_sha256_base64"
            ],
            direct_request_id=publication["direct_request_id"],
            direct_server_date=publication["direct_server_date"],
            direct_response_authenticated=publication[
                "direct_response_authenticated"
            ],
            audit=audit,
        )
        raw = base64.b64decode(value["raw_base64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise Task11SupportBoundaryError(
            "source publication response is malformed"
        ) from exc
    if (
        result.source_kind != expected_source_kind
        or result.predecessor_version_id
        != expected_predecessor_version_id
        or result.direct_response_authenticated is not True
        or audit.kind
        != AUTHORITY_AUDIT_KINDS[_SOURCE_KINDS.index(expected_source_kind)]
        or result.canonical_identity_sha256
        != publication.get("canonical_identity_sha256")
        or hashlib.sha256(raw).hexdigest() != result.file_sha256
    ):
        _fail("source publication response identity drifted")
    # SourcePublication is frozen.  An invocation-local raw-byte side table is
    # deliberately attached with object.__setattr__; it is never serialized
    # into the authority identity and exists only to build the next artifact.
    object.__setattr__(result, "_task11_raw", raw)
    return result


__all__ = [
    "ExactInputCoordinate",
    "SourceInputDocument",
    "Task11SupportBoundaryError",
    "exact_input_coordinate_from_mapping",
    "load_exact_input",
    "materialize_source_input",
    "source_input_from_mapping",
    "source_publication_from_payload",
    "source_publication_to_payload",
]
