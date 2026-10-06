"""One-shot conditional immutable-JSON S3 create adapter."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from typing import Mapping, Optional

from .dynamodb import TransactionResolution, WriteOutcome
from .h1f_adapter import (
    H1fAuditRequest,
    H1fAuditResult,
    reconcile_exact_candidate,
    validate_h1f_audit_result,
)
from .records import validate_record
from .s3_records import (
    ImmutableJsonCandidate,
    S3ObjectIdentity,
    validate_immutable_json_candidate,
)


_OWNER = "246813579024"


class S3CreateError(ValueError):
    """The conditional create did not establish a closed outcome."""


@dataclass(frozen=True)
class S3PublicationServices:
    s3: object
    actions: object
    h1f: object


@dataclass(frozen=True)
class S3CreateActionRequest:
    authority_domain: str
    operation_kind: str
    action_key: str
    candidate: ImmutableJsonCandidate
    activation_id: str
    generation: int


@dataclass(frozen=True)
class ConditionalCreateResult:
    outcome: str
    object_identity: Optional[S3ObjectIdentity]
    authority_audit_body_sha256: str
    closing_revision: int
    authorized_revision: int
    provenance: str
    direct_request_id: Optional[str] = None
    direct_server_date: Optional[str] = None
    direct_request_started_at: Optional[str] = None
    direct_response_received_at: Optional[str] = None
    direct_response_authenticated: bool = False


@dataclass(frozen=True)
class _DirectResponse:
    version_id: str
    etag: str
    checksum_sha256_base64: str
    request_id: str
    server_date: str
    request_timestamp: str
    response_timestamp: str


def _fail(message: str) -> None:
    raise S3CreateError(message)


def _string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        _fail(f"{name} must be an exact nonempty string")
    return value


def _method(service: object, name: str) -> object:
    try:
        method = getattr(service, name)
    except Exception as exc:
        raise S3CreateError(f"service lacks {name}") from exc
    if not callable(method):
        _fail(f"service lacks {name}")
    return method


def _require_live_resolution(
    value: object, *, operation: str
) -> TransactionResolution:
    if type(value) is not TransactionResolution:
        _fail(f"{operation} did not return exact Task 3 resolution")
    if (
        value.outcome is not WriteOutcome.EXACT_LIVE_OWNER_COMMIT
        or not value.may_issue_external_side_effect
    ):
        _fail(f"{operation} did not prove exact live owner commit")
    return value


def _validate_request(value: object) -> S3CreateActionRequest:
    if type(value) is not S3CreateActionRequest:
        raise TypeError("request must be exact S3CreateActionRequest")
    candidate = validate_immutable_json_candidate(value.candidate)
    for field in (
        "authority_domain",
        "operation_kind",
        "action_key",
        "activation_id",
    ):
        _string(field, getattr(value, field))
    if value.operation_kind != "S3_CREATE":
        _fail("operation_kind must be S3_CREATE")
    if value.authority_domain != "ACTIVATION":
        _fail("Task 4 supports only ACTIVATION S3_CREATE authority")
    if type(value.generation) is not int or value.generation < 1:
        _fail("generation must be an exact positive integer")
    metadata = dict(candidate.metadata)
    if (
        metadata["glm52-activation-id"] != value.activation_id
        or metadata["glm52-generation-text"] != f"{value.generation:08d}"
    ):
        _fail("request and candidate activation/generation drifted")
    return value


def _validate_audit(
    value: object, *, request: S3CreateActionRequest
) -> H1fAuditResult:
    try:
        value = validate_h1f_audit_result(value)
    except (TypeError, ValueError) as exc:
        raise S3CreateError("fresh audit result is not closed") from exc
    candidate = request.candidate
    if (
        value.authority_domain != request.authority_domain
        or value.operation_kind != request.operation_kind
        or value.action_key != request.action_key
        or value.candidate_identity_sha256
        != candidate.candidate_identity_sha256
        or value.activation_id != request.activation_id
        or value.generation != request.generation
        or type(value.closing_revision) is not int
        or value.closing_revision < 0
        or type(value.expected_authorized_revision) is not int
        or value.expected_authorized_revision != value.closing_revision + 1
    ):
        _fail("fresh audit does not bind the exact action request")
    _string("audit canonical body SHA-256", value.canonical_body_sha256)
    if (
        len(value.canonical_body_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value.canonical_body_sha256
        )
    ):
        _fail("audit canonical body SHA-256 is invalid")
    return value


def _validated_control_action_readback(
    resolution: TransactionResolution,
    *,
    request: S3CreateActionRequest,
    operation: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    indexes = [
        record
        for record in resolution.records
        if isinstance(record, Mapping)
        and record.get("record_type")
        == "glm52_production_activation_index"
    ]
    controls = [
        record
        for record in resolution.records
        if isinstance(record, Mapping)
        and record.get("record_type") == "glm52_production_control"
    ]
    actions = [
        record
        for record in resolution.records
        if isinstance(record, Mapping)
        and record.get("record_type") == "glm52_production_action"
    ]
    if (
        len(resolution.records) != 3
        or len(indexes) != 1
        or len(controls) != 1
        or len(actions) != 1
    ):
        _fail(f"{operation} readback is not coherent and complete")
    try:
        index = validate_record(
            "glm52_production_activation_index",
            dict(indexes[0]),
        )
        control = validate_record(
            "glm52_production_control",
            dict(controls[0]),
        )
        action = validate_record(
            "glm52_production_action",
            dict(actions[0]),
            sk=request.action_key,
        )
    except (TypeError, ValueError) as exc:
        raise S3CreateError(
            f"{operation} readback is not a closed Task 2 record pair"
        ) from exc

    metadata = dict(request.candidate.metadata)
    if (
        index.get("run_id") != metadata["glm52-run-id"]
        or control.get("run_id") != metadata["glm52-run-id"]
        or action.get("run_id") != metadata["glm52-run-id"]
        or index.get("campaign_identity_sha256")
        != control.get("campaign_identity_sha256")
        or control.get("campaign_identity_sha256")
        != action.get("campaign_identity_sha256")
        or index.get("current_activation_id") != request.activation_id
        or control.get("activation_id") != request.activation_id
        or action.get("activation_id") != request.activation_id
        or index.get("current_activation_ordinal")
        != control.get("activation_ordinal")
        or control.get("activation_ordinal")
        != action.get("activation_ordinal")
        or control.get("last_sky_post_generation") != request.generation
        or action.get("generation") != request.generation
        or action.get("generation_text") != f"{request.generation:08d}"
        or control.get("last_sky_post_action_key") != request.action_key
        or action.get("action_kind") != request.operation_kind
        or action.get("candidate_key") != request.candidate.key
        or action.get("candidate_file_sha256")
        != request.candidate.file_sha256
        or action.get("candidate_body_sha256")
        != request.candidate.body_sha256
        or action.get("request_body_sha256")
        != request.candidate.file_sha256
    ):
        _fail(f"{operation} readback does not bind the exact request")
    if (
        action.get("owner_epoch") != control.get("active_epoch")
        or action.get("armed_by_epoch") != control.get("active_epoch")
        or action.get("owner_execution_arn")
        != control.get("active_execution_arn")
        or action.get("armed_by_execution_arn")
        != control.get("active_execution_arn")
        or action.get("armed_by_state_machine_version_arn")
        != control.get("active_state_machine_version_arn")
        or action.get("barrier_nonce_sha256")
        != control.get("barrier_nonce_sha256")
    ):
        _fail(f"{operation} readback authority ownership drifted")
    return control, action


def _coherent_arm(
    resolution: TransactionResolution,
    *,
    request: S3CreateActionRequest,
) -> int:
    control, action = _validated_control_action_readback(
        resolution,
        request=request,
        operation="action arm",
    )
    if (
        control.get("last_sky_post_state") != "ARMED"
        or action.get("action_kind") != "S3_CREATE"
        or action.get("state") != "ARMED"
        or type(control.get("revision")) is not int
        or control.get("revision") < 0
    ):
        _fail("action arm does not bind the exact request")
    return control["revision"]  # type: ignore[return-value]


def _coherent_action_owner(
    resolution: TransactionResolution,
    *,
    request: S3CreateActionRequest,
    audit: H1fAuditResult,
) -> None:
    control, action = _validated_control_action_readback(
        resolution,
        request=request,
        operation="action consumption",
    )
    if (
        control.get("revision") != audit.expected_authorized_revision
        or control.get("active_epoch") != audit.epoch
        or control.get("active_execution_arn") != audit.execution_arn
        or control.get("barrier_nonce_sha256")
        != audit.barrier_nonce_sha256
        or control.get("fence_head_version_id")
        != audit.active_head_version_id
        or control.get("fence_head_body_sha256")
        != audit.active_head_body_sha256
        or control.get("last_sky_post_action_key") != request.action_key
        or control.get("last_sky_post_state") != "CONSUMED"
        or action.get("action_kind") != "S3_CREATE"
        or action.get("state") != "CONSUMED"
        or action.get("owner_epoch") != audit.epoch
        or action.get("owner_execution_arn") != audit.execution_arn
        or action.get("barrier_nonce_sha256")
        != audit.barrier_nonce_sha256
        or action.get("authority_audit_body_sha256")
        != audit.canonical_body_sha256
        or action.get("authority_audit_closing_revision")
        != audit.closing_revision
        or action.get("authorized_transition_from_revision")
        != audit.closing_revision
        or action.get("authorized_transition_to_revision")
        != audit.expected_authorized_revision
    ):
        _fail("action consumption audit/revision ownership drifted")
    nonce_hash = action.get("owner_invocation_nonce_sha256")
    if (
        type(nonce_hash) is not str
        or len(nonce_hash) != 64
        or any(character not in "0123456789abcdef" for character in nonce_hash)
    ):
        _fail("action consumption lacks coherent caller-nonce ownership")


def _canonical_time(name: str, value: object) -> str:
    if type(value) is not datetime or value.microsecond != 0:
        _fail(f"{name} must be an exact whole-second datetime")
    try:
        if value.utcoffset() is None:
            _fail(f"{name} must be timezone-aware")
        normalized = value.astimezone(timezone.utc)
    except Exception as exc:
        raise S3CreateError(f"{name} cannot normalize") from exc
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _direct_response(
    value: object,
    *,
    candidate: ImmutableJsonCandidate,
    request_timestamp: datetime,
    response_timestamp: datetime,
) -> _DirectResponse:
    if not isinstance(value, Mapping):
        _fail("PutObject response is malformed")
    response = dict(value)
    metadata_value = response.get("ResponseMetadata")
    if not isinstance(metadata_value, Mapping):
        _fail("PutObject response metadata is malformed")
    metadata = dict(metadata_value)
    if (
        type(metadata.get("HTTPStatusCode")) is not int
        or metadata.get("HTTPStatusCode") != 200
    ):
        _fail("PutObject status is not exact 200")
    request_id = _string("PutObject request ID", metadata.get("RequestId"))
    headers_value = metadata.get("HTTPHeaders")
    if not isinstance(headers_value, Mapping):
        _fail("PutObject HTTP headers are missing")
    headers = dict(headers_value)
    date_text = _string("PutObject server Date", headers.get("date"))
    try:
        date = parsedate_to_datetime(date_text)
    except (TypeError, ValueError) as exc:
        raise S3CreateError("PutObject server Date is invalid") from exc
    if date.tzinfo is None or date.microsecond != 0:
        _fail("PutObject server Date is not canonical")
    server_date = date.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    request_time = _canonical_time(
        "PutObject request timestamp", request_timestamp
    )
    response_time = _canonical_time(
        "PutObject response timestamp", response_timestamp
    )
    if request_time > response_time:
        _fail("PutObject request/response timestamp order is invalid")
    version = _string("PutObject VersionId", response.get("VersionId"))
    if (
        version == "null"
        or len(version) > 1024
        or any(
            ord(character) < 0x21 or ord(character) > 0x7E
            for character in version
        )
    ):
        _fail("PutObject VersionId is not opaque")
    etag = _string("PutObject ETag", response.get("ETag"))
    checksum = _string(
        "PutObject ChecksumSHA256", response.get("ChecksumSHA256")
    )
    expected_checksum = base64.b64encode(
        hashlib.sha256(candidate.raw).digest()
    ).decode("ascii")
    if checksum != expected_checksum:
        _fail("PutObject checksum does not match candidate")
    return _DirectResponse(
        version_id=version,
        etag=etag,
        checksum_sha256_base64=checksum,
        request_id=request_id,
        server_date=server_date,
        request_timestamp=request_time,
        response_timestamp=response_time,
    )


def conditional_create_immutable_json(
    *,
    services: S3PublicationServices,
    request: S3CreateActionRequest,
) -> ConditionalCreateResult:
    if type(services) is not S3PublicationServices:
        raise TypeError("services must be exact S3PublicationServices")
    exact_request = _validate_request(request)
    candidate = exact_request.candidate

    arm = _method(services.actions, "arm")
    try:
        arm_result = arm(request=exact_request)
    except Exception as exc:
        raise S3CreateError("action arm failed") from exc
    armed = _require_live_resolution(arm_result, operation="action arm")
    armed_revision = _coherent_arm(armed, request=exact_request)

    fresh_audit = _method(services.h1f, "fresh_audit")
    try:
        audit_value = fresh_audit(
            s3=services.s3,
            request=H1fAuditRequest(
                operation_kind=exact_request.operation_kind,
                action_key=exact_request.action_key,
                candidate=candidate,
            ),
        )
    except Exception as exc:
        raise S3CreateError("fresh H.1f audit failed") from exc
    audit = _validate_audit(audit_value, request=exact_request)
    if audit.closing_revision != armed_revision:
        _fail("fresh audit did not begin from the just-armed revision")

    consume = _method(services.actions, "consume_audit")
    try:
        consume_value = consume(
            request=exact_request,
            authority_audit_body_sha256=audit.canonical_body_sha256,
            closing_revision=audit.closing_revision,
            authorized_revision=audit.expected_authorized_revision,
        )
    except Exception as exc:
        raise S3CreateError("audit-bound action consume failed") from exc
    consumption = _require_live_resolution(
        consume_value, operation="audit-bound action consume"
    )
    _coherent_action_owner(
        consumption, request=exact_request, audit=audit
    )

    checksum = base64.b64encode(
        hashlib.sha256(candidate.raw).digest()
    ).decode("ascii")
    put = _method(services.s3, "put_object")
    direct: Optional[_DirectResponse] = None
    try:
        request_timestamp = datetime.now(timezone.utc).replace(microsecond=0)
        response = put(
            Bucket=candidate.bucket,
            Key=candidate.key,
            Body=candidate.raw,
            IfNoneMatch="*",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum,
            ContentType="application/json",
            Metadata=dict(candidate.metadata),
            ExpectedBucketOwner=_OWNER,
        )
        response_timestamp = datetime.now(timezone.utc).replace(microsecond=0)
        try:
            direct = _direct_response(
                response,
                candidate=candidate,
                request_timestamp=request_timestamp,
                response_timestamp=response_timestamp,
            )
        except S3CreateError:
            direct = None
    except Exception:
        direct = None

    try:
        reconciliation = reconcile_exact_candidate(
            s3=services.s3, candidate=candidate
        )
    except Exception as exc:
        raise S3CreateError("all-version reconciliation failed closed") from exc

    if reconciliation.state == "zero":
        return ConditionalCreateResult(
            outcome="absent-after-reconciliation",
            object_identity=None,
            authority_audit_body_sha256=audit.canonical_body_sha256,
            closing_revision=audit.closing_revision,
            authorized_revision=audit.expected_authorized_revision,
            provenance="all-version-reconciliation",
        )
    identity = reconciliation.object_identity
    if identity is None:
        _fail("reconciliation result is internally inconsistent")
    if direct is None:
        return ConditionalCreateResult(
            outcome="reconciled-existing",
            object_identity=identity,
            authority_audit_body_sha256=audit.canonical_body_sha256,
            closing_revision=audit.closing_revision,
            authorized_revision=audit.expected_authorized_revision,
            provenance="all-version-reconciliation",
        )
    if (
        identity.version_id != direct.version_id
        or identity.etag != direct.etag
        or identity.checksum_sha256_base64
        != direct.checksum_sha256_base64
    ):
        _fail("direct PutObject identity disagrees with all-version readback")
    return ConditionalCreateResult(
        outcome="created-direct",
        object_identity=identity,
        authority_audit_body_sha256=audit.canonical_body_sha256,
        closing_revision=audit.closing_revision,
        authorized_revision=audit.expected_authorized_revision,
        provenance="direct-response",
        direct_request_id=direct.request_id,
        direct_server_date=direct.server_date,
        direct_request_started_at=direct.request_timestamp,
        direct_response_received_at=direct.response_timestamp,
        direct_response_authenticated=True,
    )


__all__ = [
    "ConditionalCreateResult",
    "S3CreateActionRequest",
    "S3CreateError",
    "S3PublicationServices",
    "conditional_create_immutable_json",
]
