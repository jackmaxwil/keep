"""Pure immutable-S3 record identities for GLM-5.2."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Mapping, Optional, Tuple

from .s3_keys import (
    h1g_drained_s3_key,
    production_terminal_v2_s3_key,
    sky_job_bound_s3_key,
    sky_post_handoff_s3_key,
    sky_request_correlated_s3_key,
    snapshot_cleanup_authority_audit_s3_key,
    support_plane_finalized_s3_key,
    validate_frozen_key,
)


_OWNER = "246813579024"
_REGION = "us-west-2"
_TASK12_RUN_ID = "glm52-sky-20260724"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_KIND = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ACTIVATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_SELF_HASH_FIELDS = (
    "canonical_body_sha256",
    "fence_body_sha256",
    "descriptor_body_sha256",
    "approval_body_sha256",
    "snapshot_body_sha256",
    "intent_body_sha256",
    "baseline_body_sha256",
    "control_plane_ready_body_sha256",
    "acquisition_body_sha256",
    "generation_claim_body_sha256",
    "start_decision_body_sha256",
    "generation_terminal_body_sha256",
    "sky_post_handoff_body_sha256",
    "handoff_body_sha256",
)
_METADATA_KEYS = (
    "glm52-account-id",
    "glm52-activation-id",
    "glm52-body-sha256",
    "glm52-candidate-identity-sha256",
    "glm52-file-sha256",
    "glm52-generation-text",
    "glm52-record-kind",
    "glm52-region",
    "glm52-run-id",
)
_FROZEN_FAMILY_BY_RECORD_KIND = {
    "gpu-spend-approval": "gpu-spend-approval",
    "gpu-spend-snapshot": "gpu-spend-snapshot",
    "production-submission-intent": "production-submission-intent",
    "production-controller-baseline": "production-controller-baseline",
    "production-control-plane-readiness": (
        "production-control-plane-readiness"
    ),
    "production-submission-acquisition": (
        "production-submission-acquisition"
    ),
    "fence-genesis": "fence-genesis",
    "fence-successor": "fence-successor",
    "generation-claim": "generation-claim",
    "start-decision": "start-decision",
    "generation-terminal": "generation-terminal",
    "terminal-v2": "production-terminal-v2",
    "sky-post-handoff": "sky-post-handoff",
    "recovery-handoff": "sky-post-handoff",
    "sky-request-correlated": "sky-request-correlated",
    "sky-job-bound": "sky-job-bound",
    "bootstrap-ready": "bootstrap-ready",
    "worker-graceful-stop": "worker-graceful-stop",
    "campaign-drained": "campaign-drained",
    "support-plane-finalized": "support-plane-finalized",
    "h1g-drained": "h1g-drained",
    "snapshot-cleanup-authority-audit": (
        "snapshot-cleanup-authority-audit"
    ),
}
_TASK12_S3_RECORD_FAMILY_ROWS = (
    ("sky-post-handoff", "sky-post-handoff"),
    ("recovery-handoff", "sky-post-handoff"),
    ("sky-request-correlated", "sky-request-correlated"),
    ("sky-job-bound", "sky-job-bound"),
    ("terminal-v2", "production-terminal-v2"),
    ("support-plane-finalized", "support-plane-finalized"),
    ("h1g-drained", "h1g-drained"),
    (
        "snapshot-cleanup-authority-audit",
        "snapshot-cleanup-authority-audit",
    ),
)
# OPERATOR_DISPOSITION, SNAPSHOT_DISPOSITION, and
# SNAPSHOT_CLEANUP_TRANSITION are frozen DynamoDB records.  They intentionally
# have no S3 family or generic marker fallback here.
_TASK12_S3_FAMILY_BY_RECORD_KIND = dict(
    _TASK12_S3_RECORD_FAMILY_ROWS
)


def validate_task12_s3_record_family_rows(
    value: object,
) -> Tuple[Tuple[str, str], ...]:
    """Validate the complete architecture-backed Task 12 S3 family table."""

    if type(value) is not tuple:
        raise ValueError("Task 12 S3 record families must be an exact tuple")
    rows = []
    for row in value:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not str
        ):
            raise ValueError("Task 12 S3 record family row is invalid")
        rows.append(row)
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Task 12 S3 record kind is duplicated")
    exact = tuple(rows)
    if exact != _TASK12_S3_RECORD_FAMILY_ROWS:
        raise ValueError("Task 12 S3 record family table is not closed")
    return exact


def task12_s3_record_family(record_kind: object) -> str:
    """Return one physical S3 family; DynamoDB-only kinds are rejected."""

    validate_task12_s3_record_family_rows(
        _TASK12_S3_RECORD_FAMILY_ROWS
    )
    if type(record_kind) is not str:
        raise ValueError("record kind is not an S3 record kind")
    family = _TASK12_S3_FAMILY_BY_RECORD_KIND.get(record_kind)
    if family is None:
        raise ValueError("record kind is not an S3 record kind")
    return family


@dataclass(frozen=True)
class ImmutableJsonCandidate:
    record_kind: str
    bucket: str
    key: str
    raw: bytes
    file_sha256: str
    body_sha256: str
    content_type: str
    metadata: Tuple[Tuple[str, str], ...]
    candidate_identity_sha256: str


@dataclass(frozen=True)
class S3ObjectIdentity:
    bucket: str
    key: str
    version_id: str
    file_sha256: str
    body_sha256: str
    content_length: int
    etag: str
    last_modified: str
    checksum_sha256_base64: str
    checksum_type: str
    content_type: str
    metadata: Tuple[Tuple[str, str], ...]
    canonical_identity_sha256: str


@dataclass(frozen=True)
class TerminalMarkerIdentity:
    key: str
    version_id: str
    body_sha256: str
    canonical_identity_sha256: str


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
        raise ValueError("value is not canonical JSON") from exc


def _parse_canonical_file(raw: object) -> Dict[str, object]:
    if type(raw) is not bytes or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("immutable JSON must end in exactly one LF")
    try:
        text = raw[:-1].decode("ascii")
    except UnicodeError as exc:
        raise ValueError("immutable JSON must be canonical ASCII-safe UTF-8") from exc

    def unique(pairs: object) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in pairs:  # type: ignore[union-attr]
            if type(key) is not str or key in result:
                raise ValueError("JSON object members must be unique strings")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise ValueError("non-finite JSON number is forbidden")

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (TypeError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("immutable JSON is invalid") from exc
    if type(value) is not dict or _canonical(value) + b"\n" != raw:
        raise ValueError("immutable JSON bytes are not canonical")
    return value


def _exact_string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be an exact nonempty string")
    return value


def _safe_key(value: object, *, run_id: str) -> str:
    key = _exact_string("key", value)
    if (
        not key.startswith(f"campaigns/{run_id}/")
        or len(key) > 1024
        or "\\" in key
        or "//" in key
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in key)
        or any(segment in {"", ".", ".."} for segment in key.split("/"))
    ):
        raise ValueError("candidate key is not an exact safe run coordinate")
    return key


def _frozen_record_coordinate(
    *,
    record_kind: object,
    key: str,
    run_id: str,
) -> str:
    if type(record_kind) is not str:
        raise ValueError("record_kind must be an exact string")
    family = _FROZEN_FAMILY_BY_RECORD_KIND.get(record_kind)
    if family is None:
        raise ValueError("record_kind is not a frozen Task 4 family")
    return validate_frozen_key(
        key=key,
        run_id=run_id,
        allowed_families=(family,),
    )


def _task12_exact_record_coordinate(
    *,
    record_kind: str,
    key: str,
    run_id: str,
    activation_id: str,
    generation: int,
    record: Mapping[str, object],
) -> str:
    if record_kind not in _TASK12_S3_FAMILY_BY_RECORD_KIND:
        return key
    if run_id != _TASK12_RUN_ID:
        raise ValueError("Task 12 run_id is not the fixed production run")
    if record_kind in {"sky-post-handoff", "recovery-handoff"}:
        expected = sky_post_handoff_s3_key(
            run_id=run_id,
            generation=generation,
        )
    elif record_kind == "sky-request-correlated":
        expected = sky_request_correlated_s3_key(
            run_id=run_id,
            generation=generation,
        )
    elif record_kind == "sky-job-bound":
        expected = sky_job_bound_s3_key(
            run_id=run_id,
            generation=generation,
        )
    elif record_kind == "terminal-v2":
        expected = production_terminal_v2_s3_key(
            run_id=run_id,
            generation=generation,
        )
    elif record_kind == "support-plane-finalized":
        expected = support_plane_finalized_s3_key(
            run_id=run_id,
            activation_id=activation_id,
        )
    elif record_kind == "h1g-drained":
        expected = h1g_drained_s3_key(
            run_id=run_id,
            activation_id=activation_id,
        )
    else:
        activation_ordinal = record.get("activation_ordinal")
        delete_logical_attempt = record.get(
            "delete_logical_attempt"
        )
        delete_logical_attempt_text = record.get(
            "delete_logical_attempt_text"
        )
        if (
            type(activation_ordinal) is not int
            or activation_ordinal < 1
            or activation_ordinal > 99999999
        ):
            raise ValueError(
                "snapshot cleanup activation ordinal is invalid"
            )
        if (
            type(delete_logical_attempt) is not int
            or delete_logical_attempt < 1
            or delete_logical_attempt > 99999999
            or delete_logical_attempt_text
            != f"{delete_logical_attempt:08d}"
        ):
            raise ValueError(
                "snapshot cleanup logical attempt text drifted"
            )
        expected = snapshot_cleanup_authority_audit_s3_key(
            run_id=run_id,
            activation_ordinal=activation_ordinal,
            delete_logical_attempt=delete_logical_attempt,
        )
    if key != expected:
        raise ValueError("candidate key is not the exact record coordinate")
    return key


def _sha256(name: str, value: object) -> str:
    text = _exact_string(name, value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{name} must be exact lowercase SHA-256")
    return text


def _metadata_tuple(value: object) -> Tuple[Tuple[str, str], ...]:
    if type(value) is not tuple:
        raise ValueError("metadata must be an exact tuple")
    pairs = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            or not item[0]
            or item[0] != item[0].lower()
        ):
            raise ValueError("metadata contains an invalid member")
        pairs.append(item)
    if tuple(sorted(pairs)) != value or len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("metadata must be sorted with unique lower-case keys")
    return value


def _body_hash(record: Mapping[str, object]) -> str:
    present = [field for field in _SELF_HASH_FIELDS if field in record]
    matches = []
    for field in present:
        expected = _sha256(field, record[field])
        body = dict(record)
        del body[field]
        if hashlib.sha256(_canonical(body)).hexdigest() == expected:
            matches.append(expected)
    if len(matches) != 1:
        raise ValueError(
            "record must contain exactly one valid supported self-hash"
        )
    return matches[0]


def _validate_optional_payload_authority(
    record: Mapping[str, object],
) -> None:
    if "account_id" in record and record["account_id"] != _OWNER:
        raise ValueError("payload account_id is not approved")
    if "region" in record and record["region"] != _REGION:
        raise ValueError("payload region is not approved")


def _candidate_preimage(
    *,
    record_kind: str,
    bucket: str,
    key: str,
    file_sha256: str,
    body_sha256: str,
    content_type: str,
    metadata: Tuple[Tuple[str, str], ...],
) -> Mapping[str, object]:
    return {
        "record_kind": record_kind,
        "bucket": bucket,
        "key": key,
        "file_sha256": file_sha256,
        "body_sha256": body_sha256,
        "content_type": content_type,
        "metadata": [
            [name, value]
            for name, value in metadata
            if name != "glm52-candidate-identity-sha256"
        ],
    }


def _candidate_identity(**values: object) -> str:
    return hashlib.sha256(_canonical(_candidate_preimage(**values))).hexdigest()  # type: ignore[arg-type]


def build_immutable_json_candidate(
    *,
    record_kind: str,
    bucket: str,
    key: str,
    raw: bytes,
    activation_id: str,
    generation: int,
) -> ImmutableJsonCandidate:
    if type(record_kind) is not str or _RECORD_KIND.fullmatch(record_kind) is None:
        raise ValueError("record_kind is not closed lower-case kebab text")
    exact_bucket = _exact_string("bucket", bucket)
    exact_key = _exact_string("key", key)
    if type(activation_id) is not str or _ACTIVATION_ID.fullmatch(activation_id) is None:
        raise ValueError("activation_id is not an ASCII-safe identity")
    if type(generation) is not int or generation < 1 or generation > 99999999:
        raise ValueError("generation must be an exact positive eight-digit integer")
    record = _parse_canonical_file(raw)
    _validate_optional_payload_authority(record)
    run_id = record.get("run_id")
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("record run_id is unsafe")
    if (
        "activation_id" in record
        and record.get("activation_id") != activation_id
    ):
        raise ValueError("record activation_id drifted")
    if "generation" in record and record.get("generation") != generation:
        raise ValueError("record generation identity drifted")
    if (
        "generation_text" in record
        and record.get("generation_text") != f"{generation:08d}"
    ):
        raise ValueError("record generation text identity drifted")
    exact_key = _safe_key(exact_key, run_id=run_id)
    exact_key = _frozen_record_coordinate(
        record_kind=record_kind,
        key=exact_key,
        run_id=run_id,
    )
    exact_key = _task12_exact_record_coordinate(
        record_kind=record_kind,
        key=exact_key,
        run_id=run_id,
        activation_id=activation_id,
        generation=generation,
        record=record,
    )
    file_sha256 = hashlib.sha256(raw).hexdigest()
    body_sha256 = _body_hash(record)
    base_metadata = tuple(
        sorted(
            (
                ("glm52-account-id", _OWNER),
                ("glm52-activation-id", activation_id),
                ("glm52-body-sha256", body_sha256),
                ("glm52-file-sha256", file_sha256),
                ("glm52-generation-text", f"{generation:08d}"),
                ("glm52-record-kind", record_kind),
                ("glm52-region", _REGION),
                ("glm52-run-id", run_id),
            )
        )
    )
    identity = _candidate_identity(
        record_kind=record_kind,
        bucket=exact_bucket,
        key=exact_key,
        file_sha256=file_sha256,
        body_sha256=body_sha256,
        content_type="application/json",
        metadata=base_metadata,
    )
    metadata = tuple(
        sorted(
            base_metadata
            + (("glm52-candidate-identity-sha256", identity),)
        )
    )
    candidate = ImmutableJsonCandidate(
        record_kind=record_kind,
        bucket=exact_bucket,
        key=exact_key,
        raw=raw,
        file_sha256=file_sha256,
        body_sha256=body_sha256,
        content_type="application/json",
        metadata=metadata,
        candidate_identity_sha256=identity,
    )
    return validate_immutable_json_candidate(candidate)


def validate_immutable_json_candidate(
    value: object,
) -> ImmutableJsonCandidate:
    if type(value) is not ImmutableJsonCandidate:
        raise TypeError("candidate must be an exact ImmutableJsonCandidate")
    record = _parse_canonical_file(value.raw)
    _validate_optional_payload_authority(record)
    if hashlib.sha256(value.raw).hexdigest() != _sha256(
        "file_sha256", value.file_sha256
    ):
        raise ValueError("candidate file hash drifted")
    if _body_hash(record) != _sha256("body_sha256", value.body_sha256):
        raise ValueError("candidate body hash drifted")
    if value.content_type != "application/json":
        raise ValueError("candidate content type must be application/json")
    metadata = _metadata_tuple(value.metadata)
    if tuple(name for name, _ in metadata) != _METADATA_KEYS:
        raise ValueError("candidate metadata map is not exact and closed")
    metadata_map = dict(metadata)
    run_id = record.get("run_id")
    run_id = record.get("run_id")
    activation_id = metadata_map.get("glm52-activation-id")
    generation_text = metadata_map.get("glm52-generation-text")
    expected = {
        "glm52-account-id": _OWNER,
        "glm52-activation-id": metadata_map.get("glm52-activation-id"),
        "glm52-body-sha256": value.body_sha256,
        "glm52-candidate-identity-sha256": value.candidate_identity_sha256,
        "glm52-file-sha256": value.file_sha256,
        "glm52-generation-text": metadata_map.get("glm52-generation-text"),
        "glm52-record-kind": value.record_kind,
        "glm52-region": _REGION,
        "glm52-run-id": record.get("run_id"),
    }
    if metadata_map != expected:
        raise ValueError("candidate metadata does not match record identity")
    if (
        type(generation_text) is not str
        or re.fullmatch(r"[0-9]{8}", generation_text) is None
        or generation_text == "00000000"
    ):
        raise ValueError("candidate generation identity is invalid")
    generation = int(generation_text)
    if (
        ("activation_id" in record and record.get("activation_id") != activation_id)
        or ("generation" in record and record.get("generation") != generation)
        or (
            "generation_text" in record
            and record.get("generation_text") != generation_text
        )
    ):
        raise ValueError("candidate metadata scope disagrees with source record")
    if (
        type(activation_id) is not str
        or _ACTIVATION_ID.fullmatch(activation_id) is None
        or type(run_id) is not str
        or _RUN_ID.fullmatch(run_id) is None
    ):
        raise ValueError("candidate metadata scope is unsafe")
    _exact_string("bucket", value.bucket)
    _safe_key(value.key, run_id=run_id)
    if type(value.record_kind) is not str or _RECORD_KIND.fullmatch(
        value.record_kind
    ) is None:
        raise ValueError("candidate record kind is invalid")
    _frozen_record_coordinate(
        record_kind=value.record_kind,
        key=value.key,
        run_id=run_id,
    )
    _task12_exact_record_coordinate(
        record_kind=value.record_kind,
        key=value.key,
        run_id=run_id,
        activation_id=activation_id,
        generation=generation,
        record=record,
    )
    expected_identity = _candidate_identity(
        record_kind=value.record_kind,
        bucket=value.bucket,
        key=value.key,
        file_sha256=value.file_sha256,
        body_sha256=value.body_sha256,
        content_type=value.content_type,
        metadata=value.metadata,
    )
    if _sha256(
        "candidate_identity_sha256", value.candidate_identity_sha256
    ) != expected_identity:
        raise ValueError("candidate canonical identity drifted")
    return value


def _object_preimage(value: S3ObjectIdentity) -> Mapping[str, object]:
    return {
        "bucket": value.bucket,
        "key": value.key,
        "version_id": value.version_id,
        "file_sha256": value.file_sha256,
        "body_sha256": value.body_sha256,
        "content_length": value.content_length,
        "etag": value.etag,
        "last_modified": value.last_modified,
        "checksum_sha256_base64": value.checksum_sha256_base64,
        "checksum_type": value.checksum_type,
        "content_type": value.content_type,
        "metadata": [list(item) for item in value.metadata],
    }


def build_s3_object_identity(
    *,
    candidate: ImmutableJsonCandidate,
    version_id: str,
    content_length: int,
    etag: str,
    last_modified: str,
    checksum_sha256_base64: str,
    checksum_type: str,
    content_type: str,
    metadata: Tuple[Tuple[str, str], ...],
) -> S3ObjectIdentity:
    validate_immutable_json_candidate(candidate)
    provisional = S3ObjectIdentity(
        bucket=candidate.bucket,
        key=candidate.key,
        version_id=version_id,
        file_sha256=candidate.file_sha256,
        body_sha256=candidate.body_sha256,
        content_length=content_length,
        etag=etag,
        last_modified=last_modified,
        checksum_sha256_base64=checksum_sha256_base64,
        checksum_type=checksum_type,
        content_type=content_type,
        metadata=metadata,
        canonical_identity_sha256="0" * 64,
    )
    result = S3ObjectIdentity(
        **{
            **provisional.__dict__,
            "canonical_identity_sha256": hashlib.sha256(
                _canonical(_object_preimage(provisional))
            ).hexdigest(),
        }
    )
    return validate_s3_object_identity(result, expected_candidate=candidate)


def validate_s3_object_identity(
    value: object,
    *,
    expected_candidate: Optional[ImmutableJsonCandidate] = None,
) -> S3ObjectIdentity:
    if type(value) is not S3ObjectIdentity:
        raise TypeError("object identity must be an exact S3ObjectIdentity")
    _exact_string("bucket", value.bucket)
    _exact_string("key", value.key)
    _exact_string("version_id", value.version_id)
    _sha256("file_sha256", value.file_sha256)
    _sha256("body_sha256", value.body_sha256)
    if type(value.content_length) is not int or value.content_length < 0:
        raise ValueError("content_length must be an exact nonnegative integer")
    _exact_string("etag", value.etag)
    if (
        type(value.last_modified) is not str
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value.last_modified)
        is None
    ):
        raise ValueError("last_modified must be canonical UTC whole-second text")
    _exact_string("checksum_sha256_base64", value.checksum_sha256_base64)
    try:
        decoded_checksum = base64.b64decode(
            value.checksum_sha256_base64, validate=True
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("checksum is not canonical base64") from exc
    if (
        len(decoded_checksum) != 32
        or decoded_checksum.hex() != value.file_sha256
        or base64.b64encode(decoded_checksum).decode("ascii")
        != value.checksum_sha256_base64
    ):
        raise ValueError("checksum does not bind the full-object file hash")
    if value.checksum_type != "FULL_OBJECT":
        raise ValueError("checksum type must be FULL_OBJECT")
    if value.content_type != "application/json":
        raise ValueError("content type must be application/json")
    metadata = _metadata_tuple(value.metadata)
    if tuple(name for name, _ in metadata) != _METADATA_KEYS:
        raise ValueError("object metadata map is not exact and closed")
    identity = hashlib.sha256(_canonical(_object_preimage(value))).hexdigest()
    if _sha256(
        "canonical_identity_sha256", value.canonical_identity_sha256
    ) != identity:
        raise ValueError("object canonical identity drifted")
    if expected_candidate is not None:
        candidate = validate_immutable_json_candidate(expected_candidate)
        if (
            value.bucket != candidate.bucket
            or value.key != candidate.key
            or value.file_sha256 != candidate.file_sha256
            or value.body_sha256 != candidate.body_sha256
            or value.content_length != len(candidate.raw)
            or value.content_type != candidate.content_type
            or value.metadata != candidate.metadata
        ):
            raise ValueError("object identity does not match candidate")
    return value


def _terminal_projection_preimage(
    *, key: str, version_id: str, body_sha256: str
) -> Mapping[str, object]:
    return {
        "key": key,
        "version_id": version_id,
        "body_sha256": body_sha256,
    }


def terminal_marker_identity(value: S3ObjectIdentity) -> TerminalMarkerIdentity:
    exact = validate_s3_object_identity(value)
    result = TerminalMarkerIdentity(
        key=exact.key,
        version_id=exact.version_id,
        body_sha256=exact.body_sha256,
        canonical_identity_sha256=hashlib.sha256(
            _canonical(
                _terminal_projection_preimage(
                    key=exact.key,
                    version_id=exact.version_id,
                    body_sha256=exact.body_sha256,
                )
            )
        ).hexdigest(),
    )
    return validate_terminal_marker_identity(result)


def validate_terminal_marker_identity(
    value: object,
) -> TerminalMarkerIdentity:
    if type(value) is not TerminalMarkerIdentity:
        raise TypeError("terminal marker must be exact TerminalMarkerIdentity")
    key = _exact_string("key", value.key)
    version_id = _exact_string("version_id", value.version_id)
    body_sha256 = _sha256("body_sha256", value.body_sha256)
    expected = hashlib.sha256(
        _canonical(
            _terminal_projection_preimage(
                key=key,
                version_id=version_id,
                body_sha256=body_sha256,
            )
        )
    ).hexdigest()
    if _sha256(
        "canonical_identity_sha256", value.canonical_identity_sha256
    ) != expected:
        raise ValueError("terminal marker canonical identity drifted")
    return value


__all__ = [
    "ImmutableJsonCandidate",
    "S3ObjectIdentity",
    "TerminalMarkerIdentity",
    "build_immutable_json_candidate",
    "build_s3_object_identity",
    "terminal_marker_identity",
    "task12_s3_record_family",
    "validate_task12_s3_record_family_rows",
    "validate_immutable_json_candidate",
    "validate_s3_object_identity",
    "validate_terminal_marker_identity",
]
