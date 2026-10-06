"""Fixed-domain Task 12 conditional-create writer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
from typing import Mapping, Optional

from .canonical import canonical_json_bytes, canonical_sha256
from .records import ledger_pk, ledger_sk, validate_record
from .s3_keys import (
    h1g_drained_s3_key,
    production_terminal_v2_s3_key,
    sky_post_handoff_s3_key,
    support_plane_finalized_s3_key,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_VERSION = re.compile(r"[\x21-\x7e]{1,1024}\Z")
_RECOVERY_HANDOFF_FIELDS = frozenset(
    """run_id campaign_identity_sha256 generation generation_text
    submit_attempt_id decision_key decision_version_id decision_file_sha256
    decision_body_sha256 sky_post_action_key sky_post_consumed_at
    sky_post_outcome_class expected_sky_job_name task_yaml_sha256
    request_body_sha256 api_server_identity_sha256 sky_request_id
    post_started_at post_completed_or_lost_at binding_state
    handoff_body_sha256""".split()
)
_COMMON_RETAINED_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "campaign_identity_sha256",
        "activation_id",
        "activation_ordinal",
        "writer_function_version_arn",
        "writer_dispatch_identity_sha256",
        "writer_invocation_nonce_sha256",
        "published_at",
        "canonical_body_sha256",
    }
)
_WRITER_SPECS: Mapping[str, Mapping[str, object]] = {
    "TerminalV2": {
        "domains": ("ACTIVATION", "RECOVERY"),
        "action_kind": "TERMINAL_V2",
        "record_type": "glm52_production_terminal_v2",
        "hash_field": "canonical_body_sha256",
    },
    "SupportPlaneFinalized": {
        "domains": ("FINALIZATION",),
        "action_kind": "SUPPORT_PLANE_FINALIZED",
        "record_type": "glm52_production_support_plane_finalized",
        "hash_field": "canonical_body_sha256",
    },
    "H1GDrained": {
        "domains": ("FINALIZATION",),
        "action_kind": "H1G_DRAINED",
        "record_type": "glm52_production_h1g_drained",
        "hash_field": "canonical_body_sha256",
    },
    "RecoveryHandoff": {
        "domains": ("RECOVERY",),
        "action_kind": "RECOVERY_HANDOFF",
        "record_type": None,
        "hash_field": "handoff_body_sha256",
    },
    "OperatorDisposition": {
        "domains": ("OPERATOR_DISPOSITION",),
        "action_kind": "OPERATOR_DISPOSITION",
        "record_type": "glm52_production_operator_disposition",
        "hash_field": "canonical_body_sha256",
    },
}


class Task12WriterError(ValueError):
    """A Task 12 writer lost fixed-domain or immutable-create custody."""


@dataclass(frozen=True)
class RetainedWriteCandidate:
    writer_kind: str
    authority_domain: str
    action_kind: str
    campaign_bucket: str
    activation_id: str
    generation: int
    coordinate: str
    record: Mapping[str, object]
    raw: bytes
    file_sha256: str
    body_sha256: str
    candidate_identity_sha256: str


@dataclass(frozen=True)
class RetainedWriterActionAuthority:
    authority_domain: str
    action_kind: str
    action_key: str
    candidate_identity_sha256: str
    action_identity_sha256: str
    owner_invocation_nonce_sha256: str
    state: str
    authorized_revision: int


@dataclass(frozen=True)
class RetainedWriterAuditAuthority:
    authority_domain: str
    action_kind: str
    action_key: str
    candidate_identity_sha256: str
    action_identity_sha256: str
    audit_kind: str
    closing_revision: int
    authorized_revision: int
    current_revision: int
    observed_at: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class ConditionalCreateResponse:
    classification: str
    candidate_identity_sha256: str
    request_id: str
    response_identity_sha256: str
    version_id: Optional[str]
    authenticated: bool


@dataclass(frozen=True)
class ExactCandidateReconciliation:
    state: str
    candidate_identity_sha256: str
    file_sha256: str
    raw: bytes
    version_id: str
    request_ids: tuple[str, ...]
    authenticated: bool


@dataclass(frozen=True)
class Task12WriterServices:
    boundary: object


@dataclass(frozen=True)
class RetainedWriteResult:
    writer_kind: str
    outcome: str
    coordinate: str
    candidate_identity_sha256: str
    object_version_id: str
    response_request_ids: tuple[str, ...]
    response_authenticated: bool
    canonical_identity_sha256: str


def _fail(message: str) -> None:
    raise Task12WriterError(message)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail(label + " must be exact lowercase SHA-256")
    return value


def _utc(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(label + " must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Task12WriterError(label + " must be canonical UTC") from exc
    if parsed.microsecond:
        _fail(label + " must be whole-second UTC")
    return value


def _record_body_hash(
    writer_kind: str,
    record: Mapping[str, object],
) -> str:
    spec = _WRITER_SPECS[writer_kind]
    hash_field = str(spec["hash_field"])
    expected = record.get(hash_field)
    _sha(expected, writer_kind + " body identity")
    body = dict(record)
    body.pop(hash_field)
    if expected != canonical_sha256(body):
        _fail(writer_kind + " record self-hash drifted")
    return str(expected)


def _validate_recovery_handoff(
    record: Mapping[str, object],
    *,
    generation: int,
) -> None:
    try:
        from .task12_correlation import validate_sky_post_handoff_record

        validate_sky_post_handoff_record(dict(record))
    except (TypeError, ValueError) as exc:
        raise Task12WriterError("RecoveryHandoff record schema drifted") from exc
    if (
        record.get("run_id") != RUN_ID
        or record.get("generation") != generation
        or record.get("generation_text") != f"{generation:08d}"
        or record.get("binding_state") != "reconcile-required"
        or record.get("sky_post_outcome_class")
        not in {
            "PROVED_NOT_SENT_OWNER_DIED",
            "AMBIGUOUS_OWNER_DIED",
        }
        or type(record.get("sky_post_action_key")) is not str
        or "SKY_POST" not in str(record["sky_post_action_key"])
    ):
        _fail("RecoveryHandoff retained classification drifted")
    for field in (
        "campaign_identity_sha256",
        "decision_file_sha256",
        "decision_body_sha256",
        "task_yaml_sha256",
        "request_body_sha256",
        "api_server_identity_sha256",
    ):
        _sha(record.get(field), "RecoveryHandoff " + field)
    for field in ("sky_post_consumed_at", "post_completed_or_lost_at"):
        _utc(record.get(field), "RecoveryHandoff " + field)
    if record.get("post_started_at") is not None:
        _utc(record.get("post_started_at"), "RecoveryHandoff post_started_at")


def _validate_common_retained(
    writer_kind: str,
    record: Mapping[str, object],
    *,
    activation_id: str,
) -> None:
    expected_type = _WRITER_SPECS[writer_kind]["record_type"]
    try:
        validate_record(str(expected_type), record)
    except (TypeError, ValueError) as exc:
        raise Task12WriterError(
            writer_kind + " record schema drifted"
        ) from exc
    if (
        writer_kind != "H1GDrained"
        and set(record) != _COMMON_RETAINED_FIELDS
    ):
        _fail(writer_kind + " record schema drifted")
    if (
        record.get("schema_version") != 1
        or record.get("record_type") != expected_type
        or record.get("account_id") != ACCOUNT_ID
        or record.get("region") != REGION
        or record.get("run_id") != RUN_ID
        or record.get("activation_id") != activation_id
        or type(record.get("activation_ordinal")) is not int
        or record["activation_ordinal"] <= 0
        or type(record.get("writer_function_version_arn")) is not str
        or not record["writer_function_version_arn"]
    ):
        _fail(writer_kind + " retained identity drifted")
    for field in (
        "campaign_identity_sha256",
        "writer_dispatch_identity_sha256",
        "writer_invocation_nonce_sha256",
    ):
        _sha(record.get(field), writer_kind + " " + field)
    _utc(record.get("published_at"), writer_kind + " published_at")


def _coordinate(
    writer_kind: str,
    *,
    campaign_bucket: str,
    activation_id: str,
    generation: int,
) -> str:
    if writer_kind == "TerminalV2":
        key = production_terminal_v2_s3_key(
            run_id=RUN_ID,
            generation=generation,
        )
    elif writer_kind == "SupportPlaneFinalized":
        key = support_plane_finalized_s3_key(
            run_id=RUN_ID,
            activation_id=activation_id,
        )
    elif writer_kind == "H1GDrained":
        key = h1g_drained_s3_key(
            run_id=RUN_ID,
            activation_id=activation_id,
        )
    elif writer_kind == "RecoveryHandoff":
        key = sky_post_handoff_s3_key(
            run_id=RUN_ID,
            generation=generation,
        )
    else:
        return (
            "dynamodb://"
            + ledger_pk(RUN_ID)
            + "/"
            + ledger_sk(
                "glm52_production_operator_disposition",
                activation_id=activation_id,
            )
        )
    return "s3://" + campaign_bucket + "/" + key


def retained_writer_coordinate(
    *,
    writer_kind: str,
    campaign_bucket: str,
    activation_id: str,
    generation: int,
) -> str:
    """Return the one canonical writer coordinate without constructing a body.

    Runtime operation descriptors use this only to bind an S3 reader to the
    same fixed key builder as the retained conditional-create writer.  The
    version and body identity remain live control facts and are never embedded
    in postpublication descriptors.
    """

    if (
        writer_kind not in _WRITER_SPECS
        or type(campaign_bucket) is not str
        or not campaign_bucket
        or type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
        or type(generation) is not int
        or generation < 1
    ):
        _fail("writer coordinate identity is invalid")
    return _coordinate(
        writer_kind,
        campaign_bucket=campaign_bucket,
        activation_id=activation_id,
        generation=generation,
    )


def build_versioned_writer_control(
    *,
    candidate: object,
    result: object,
    published_at: object,
) -> dict[str, object]:
    """Bind a proven retained S3 version into one durable canonical control.

    This is intentionally built only from the authenticated create/adoption
    result.  It is the handoff consumed by later version-pinned readers; no
    caller may preseed a future version into an operation descriptor.
    """

    exact_candidate = _validate_candidate(candidate)
    if exact_candidate.writer_kind not in {
        "TerminalV2",
        "SupportPlaneFinalized",
        "H1GDrained",
        "RecoveryHandoff",
    } or not exact_candidate.coordinate.startswith("s3://"):
        _fail("only retained S3 writers may create a versioned control")
    if type(result) is not RetainedWriteResult:
        _fail("versioned writer control result must be typed")
    if (
        result.writer_kind != exact_candidate.writer_kind
        or result.coordinate != exact_candidate.coordinate
        or result.candidate_identity_sha256
        != exact_candidate.candidate_identity_sha256
        or result.response_authenticated is not True
        or _VERSION.fullmatch(result.object_version_id) is None
    ):
        _fail("versioned writer control result is not proven")
    _utc(published_at, "versioned writer control published_at")
    bucket_key = exact_candidate.coordinate[len("s3://") :].split("/", 1)
    if len(bucket_key) != 2 or not all(bucket_key):
        _fail("versioned writer control coordinate is malformed")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task12_versioned_writer_control_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": exact_candidate.activation_id,
        "generation": exact_candidate.generation,
        "generation_text": f"{exact_candidate.generation:08d}",
        "writer_kind": exact_candidate.writer_kind,
        "campaign_bucket": bucket_key[0],
        "object_key": bucket_key[1],
        "object_version_id": result.object_version_id,
        "file_sha256": exact_candidate.file_sha256,
        "body_sha256": exact_candidate.body_sha256,
        "candidate_identity_sha256": exact_candidate.candidate_identity_sha256,
        "writer_result_identity_sha256": result.canonical_identity_sha256,
        "published_at": published_at,
    }
    body["canonical_body_sha256"] = canonical_sha256(body)
    try:
        return validate_record("glm52_task12_versioned_writer_control_v1", body)
    except (TypeError, ValueError) as exc:
        raise Task12WriterError("versioned writer control is invalid") from exc


def build_retained_writer_candidate(
    *,
    writer_kind: str,
    campaign_bucket: str,
    activation_id: str,
    generation: int,
    authority_domain: str,
    record: object,
) -> RetainedWriteCandidate:
    """Build one exact fixed-coordinate candidate without external effects."""

    if writer_kind not in _WRITER_SPECS:
        _fail("writer kind is not closed")
    spec = _WRITER_SPECS[writer_kind]
    if (
        type(campaign_bucket) is not str
        or not campaign_bucket
        or type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
        or type(generation) is not int
        or generation < 1
        or authority_domain not in spec["domains"]
        or type(record) is not dict
    ):
        _fail("writer candidate identity is invalid")
    if writer_kind == "TerminalV2":
        try:
            exact_record = validate_record(
                "glm52_production_terminal_v2",
                record,
            )
        except (TypeError, ValueError) as exc:
            raise Task12WriterError(
                "TerminalV2 record validation failed"
            ) from exc
        if (
            exact_record["activation_id"] != activation_id
            or exact_record["generation"] != generation
        ):
            _fail("TerminalV2 record coordinate drifted")
    elif writer_kind == "OperatorDisposition":
        try:
            exact_record = validate_record(
                "glm52_production_operator_disposition",
                record,
            )
        except (TypeError, ValueError) as exc:
            raise Task12WriterError(
                "OperatorDisposition record validation failed"
            ) from exc
        if exact_record["activation_id"] != activation_id:
            _fail("OperatorDisposition record coordinate drifted")
    else:
        exact_record = dict(record)
        if writer_kind == "RecoveryHandoff":
            _validate_recovery_handoff(
                exact_record,
                generation=generation,
            )
        else:
            _validate_common_retained(
                writer_kind,
                exact_record,
                activation_id=activation_id,
            )
    body_sha256 = _record_body_hash(writer_kind, exact_record)
    raw = canonical_json_bytes(exact_record) + b"\n"
    file_sha256 = hashlib.sha256(raw).hexdigest()
    coordinate = retained_writer_coordinate(
        writer_kind=writer_kind,
        campaign_bucket=campaign_bucket,
        activation_id=activation_id,
        generation=generation,
    )
    identity_body = {
        "writer_kind": writer_kind,
        "authority_domain": authority_domain,
        "action_kind": spec["action_kind"],
        "coordinate": coordinate,
        "file_sha256": file_sha256,
        "body_sha256": body_sha256,
    }
    return RetainedWriteCandidate(
        writer_kind=writer_kind,
        authority_domain=authority_domain,
        action_kind=str(spec["action_kind"]),
        campaign_bucket=campaign_bucket,
        activation_id=activation_id,
        generation=generation,
        coordinate=coordinate,
        record=exact_record,
        raw=raw,
        file_sha256=file_sha256,
        body_sha256=body_sha256,
        candidate_identity_sha256=canonical_sha256(identity_body),
    )


def _validate_candidate(value: object) -> RetainedWriteCandidate:
    if type(value) is not RetainedWriteCandidate:
        _fail("writer candidate must be typed")
    rebuilt = build_retained_writer_candidate(
        writer_kind=value.writer_kind,
        campaign_bucket=value.campaign_bucket,
        activation_id=value.activation_id,
        generation=value.generation,
        authority_domain=value.authority_domain,
        record=dict(value.record),
    )
    if value != rebuilt:
        _fail("writer candidate identity drifted")
    return value


def _audit_body(
    value: RetainedWriterAuditAuthority,
) -> Mapping[str, object]:
    return {
        "authority_domain": value.authority_domain,
        "action_kind": value.action_kind,
        "action_key": value.action_key,
        "candidate_identity_sha256": value.candidate_identity_sha256,
        "action_identity_sha256": value.action_identity_sha256,
        "audit_kind": value.audit_kind,
        "closing_revision": value.closing_revision,
        "authorized_revision": value.authorized_revision,
        "current_revision": value.current_revision,
        "observed_at": value.observed_at,
    }


def _expected_action_key_fragment(
    candidate: RetainedWriteCandidate,
) -> str:
    if candidate.authority_domain == "OPERATOR_DISPOSITION":
        return (
            "ACTIVATION#"
            + candidate.activation_id
            + "#OPERATOR_DISPOSITION"
        )
    return (
        "ACTIVATION#"
        + candidate.activation_id
        + "#"
        + candidate.authority_domain
        + "_ACTION#"
        + candidate.action_kind
        + "#"
    )


def validate_retained_writer_authority(
    *,
    candidate: RetainedWriteCandidate,
    action: object,
    audit: object,
) -> tuple[RetainedWriterActionAuthority, RetainedWriterAuditAuthority]:
    """Reject stale, relabelled, generic, or foreign action/audit authority."""

    candidate = _validate_candidate(candidate)
    if (
        type(action) is not RetainedWriterActionAuthority
        or type(audit) is not RetainedWriterAuditAuthority
    ):
        _fail("writer action and audit authority must be typed")
    expected_audit_kind = (
        "WRITTEN_APPROVAL_VERSION"
        if candidate.authority_domain == "OPERATOR_DISPOSITION"
        else "H1F_GENESIS_TO_ZERO_CHILD"
    )
    exact_key = _expected_action_key_fragment(candidate)
    key_matches = (
        action.action_key == exact_key
        if candidate.authority_domain == "OPERATOR_DISPOSITION"
        else exact_key in action.action_key
    )
    if (
        action.authority_domain != candidate.authority_domain
        or action.action_kind != candidate.action_kind
        or action.action_kind == "S3_CREATE"
        or not key_matches
        or action.candidate_identity_sha256
        != candidate.candidate_identity_sha256
        or action.state != "CONSUMED"
        or type(action.authorized_revision) is not int
        or action.authorized_revision < 1
    ):
        _fail("writer action does not bind the fixed candidate")
    for field in (
        "action_identity_sha256",
        "owner_invocation_nonce_sha256",
    ):
        _sha(getattr(action, field), "writer " + field)
    if (
        audit.authority_domain != action.authority_domain
        or audit.action_kind != action.action_kind
        or audit.action_key != action.action_key
        or audit.candidate_identity_sha256
        != candidate.candidate_identity_sha256
        or audit.action_identity_sha256 != action.action_identity_sha256
        or audit.audit_kind != expected_audit_kind
        or type(audit.closing_revision) is not int
        or audit.authorized_revision != audit.closing_revision + 1
        or audit.current_revision != audit.authorized_revision
        or audit.authorized_revision != action.authorized_revision
        or audit.canonical_identity_sha256
        != canonical_sha256(_audit_body(audit))
    ):
        _fail("writer audit is stale, relabelled, or foreign")
    _utc(audit.observed_at, "writer audit observed_at")
    return action, audit


def _request_ids(values: object, label: str) -> tuple[str, ...]:
    if (
        type(values) is not tuple
        or not values
        or any(type(value) is not str or not value for value in values)
        or len(set(values)) != len(values)
    ):
        _fail(label + " request IDs are not exact")
    return values


def _result(
    *,
    candidate: RetainedWriteCandidate,
    outcome: str,
    version_id: str,
    request_ids: tuple[str, ...],
) -> RetainedWriteResult:
    body = {
        "writer_kind": candidate.writer_kind,
        "outcome": outcome,
        "coordinate": candidate.coordinate,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "object_version_id": version_id,
        "response_request_ids": request_ids,
        "response_authenticated": True,
    }
    return RetainedWriteResult(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


def write_retained_candidate(
    *,
    services: Task12WriterServices,
    candidate: RetainedWriteCandidate,
    action: object,
    audit: object,
) -> RetainedWriteResult:
    """Issue one conditional create; duplicates are reconcile-only."""

    if type(services) is not Task12WriterServices:
        _fail("writer services must be typed")
    candidate = _validate_candidate(candidate)
    exact_action, exact_audit = validate_retained_writer_authority(
        candidate=candidate,
        action=action,
        audit=audit,
    )
    create = getattr(services.boundary, "conditional_create", None)
    reconcile = getattr(services.boundary, "reconcile_exact", None)
    if not callable(create) or not callable(reconcile):
        _fail("writer boundary is incomplete")
    try:
        response = create(
            candidate=candidate,
            action=exact_action,
            audit=exact_audit,
        )
    except Exception:
        response = None
    if type(response) is ConditionalCreateResponse:
        if (
            response.classification == "CREATED"
            and response.authenticated is True
            and response.candidate_identity_sha256
            == candidate.candidate_identity_sha256
            and type(response.request_id) is str
            and response.request_id
            and type(response.version_id) is str
            and _VERSION.fullmatch(response.version_id) is not None
        ):
            _sha(
                response.response_identity_sha256,
                "create response identity",
            )
            return _result(
                candidate=candidate,
                outcome="created-authenticated",
                version_id=response.version_id,
                request_ids=(response.request_id,),
            )
        if response.classification not in {
            "PRECONDITION_FAILED",
            "AMBIGUOUS",
            "CREATED",
        }:
            _fail("conditional create classification is not closed")
    try:
        reconciliation = reconcile(candidate=candidate)
    except Exception as exc:
        raise Task12WriterError(
            "exact candidate reconciliation failed"
        ) from exc
    if (
        type(reconciliation) is not ExactCandidateReconciliation
        or reconciliation.state != "SOLE_EXACT_CANDIDATE"
        or reconciliation.authenticated is not True
        or reconciliation.candidate_identity_sha256
        != candidate.candidate_identity_sha256
        or reconciliation.file_sha256 != candidate.file_sha256
        or reconciliation.raw != candidate.raw
        or type(reconciliation.version_id) is not str
        or _VERSION.fullmatch(reconciliation.version_id) is None
    ):
        _fail("duplicate adoption did not reconcile exact candidate bytes")
    request_ids = _request_ids(
        reconciliation.request_ids,
        "reconciliation",
    )
    return _result(
        candidate=candidate,
        outcome="reconciled-exact-existing",
        version_id=reconciliation.version_id,
        request_ids=request_ids,
    )


__all__ = [
    "ConditionalCreateResponse",
    "ExactCandidateReconciliation",
    "RetainedWriteCandidate",
    "RetainedWriteResult",
    "RetainedWriterActionAuthority",
    "RetainedWriterAuditAuthority",
    "Task12WriterError",
    "Task12WriterServices",
    "build_retained_writer_candidate",
    "validate_retained_writer_authority",
    "write_retained_candidate",
]
