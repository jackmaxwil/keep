"""Private Task 11 decision-to-POST closure contracts.

This module is deliberately import-light.  It contains the frozen workflow
grammar and the timing authority validator used both by local qualification
tests and by the production support-plane route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import math
import re
import secrets
import time
from typing import Callable, Mapping, Optional, Tuple, Union

from .canonical import canonical_json_bytes, canonical_sha256
from .live_authority import H1dLiveAuthorityResult
from .sky_admission import AdmissionResult, AttestationResult


LAMBDA_CONFIGURED_TIMEOUT_SECONDS = 840
CLOSURE_DEADLINE_SECONDS = 720
STEP_FUNCTION_TASK_TIMEOUT_SECONDS = 1200
MIN_REMAINING_BEFORE_TOKEN_SECONDS = 120

CLOSURE_STEPS = (
    "PROVE_PREAUTHORIZED_BATCH_TEMPLATE",
    "ACQUIRE_CFN_QUIESCENCE",
    "REVALIDATE_RUNTIME_ATTACHMENTS_AND_SEALS",
    "PUBLISH_FIVE_SOURCES_SEQUENTIALLY",
    "CREATE_AND_APPLY_BATCH_SUCCESSOR",
    "SEAL_DECISION_AND_ACQUIRE_BARRIER",
    "ATTEST_SKY_IDENTITY_PRE_DECISION",
    "CREATE_OR_RECOVER_CLAIM",
    "MODEL_CURRENT_LAUNCH_OR_EXPIRE",
    "CREATE_DIRECT_DECISION",
    "AUTHENTICATE_DIRECT_DECISION_RESPONSE",
    "PAGINATE_AND_EXACT_READ_DECISION",
    "VALIDATE_H1E_MODELED_SUBMIT_ONCE",
    "REINSPECT_TASK8_LIVE_AUTHORITY",
    "REATTEST_SKY_IDENTITY",
    "RECHECK_ALL_AUTHORITY",
    "ARM_SKY_ACTION",
    "CONSUME_ADMISSION_RESERVATION",
    "COHERENT_READBACK_AFTER_TRANSACTIONS",
    "INVOKE_EXACT_LAUNCH_ADMISSION_VERSION_ONCE",
    "ADMISSION_OWN_POST_STARTED",
    "ADMISSION_AUDIT_POST_AUTHORITY",
    "ADMISSION_AUTHORIZE_AND_SEND_ONCE",
    "CLASSIFY_POST_WITHOUT_RETRY",
    "AUDIT_HANDOFF_AUTHORITY",
    "PERSIST_CORRELATION_AND_HANDOFF",
    "RELEASE_TO_NUMERIC_BINDING",
    "PROVE_NO_STORED_DECISION_SUBMIT_SURFACE",
)

AUTHORITY_AUDIT_KINDS = (
    "SOURCE_GPU_SPEND",
    "SOURCE_SUBMISSION_INTENT",
    "SOURCE_CONTROLLER_BASELINE",
    "SOURCE_CONTROL_PLANE_READINESS",
    "SOURCE_SUBMISSION_ACQUISITION",
    "BATCH_SUCCESSOR",
    "CHANGE_SET_CREATE",
    "CHANGE_SET_EXECUTE",
    "CLAIM",
    "DECISION",
    "SKY_POST",
)
HANDOFF_AUDIT_KIND = "SKY_POST_HANDOFF"
SOURCE_KINDS = (
    "GPU_SPEND",
    "SUBMISSION_INTENT",
    "CONTROLLER_BASELINE",
    "CONTROL_PLANE_READINESS",
    "SUBMISSION_ACQUISITION",
)
SOURCE_SERVICE_METHODS = (
    "publish_gpu_spend_snapshot",
    "publish_submission_intent",
    "publish_controller_baseline",
    "publish_control_plane_readiness",
    "publish_submission_acquisition",
)
SUCCESSOR_SERVICE_OPERATIONS = (
    ("create_batch_successor", "BATCH_SUCCESSOR"),
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_ADMISSION_VERSION = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:"
    r"keep-glm52-h1g-launch-admission:[1-9][0-9]*$"
)
_NUMERIC_BINDING_VERSION = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:"
    r"keep-glm52-h1g-numeric-binding:[1-9][0-9]*$"
)

# These insertion orders are part of the frozen timing-ledger grammar.
CLOSURE_PHASE_CEILINGS = {
    "TEMPLATE_CHANGE_SET_QUIESCENCE_MAINTENANCE_SEAL_PREFLIGHT": 150,
    "CLIENT_WARMING_AND_IMMUTABLE_CONSTRUCTION": 60,
    "NON_AUTHORITATIVE_SIZING": 90,
    "STABLE_TLS_SKY_IDENTITY_PREFLIGHT": 30,
    "FRESH_AUTHORITY_SUFFIX": 58,
    "POST_ADMISSION_HANDOFF_EVIDENCE": 60,
    "AMBIGUITY_FAILURE_UNWIND": 55,
    "UNUSED_RESERVE": 97,
}

SUFFIX_PHASE_CEILINGS = {
    "FIVE_SOURCE_CYCLES": 7,
    "SUCCESSOR_AND_CHANGE_SET_CYCLES": 8,
    "POLICY_DENY_READBACKS": 13,
    "CLAIM_WALK_AND_TRANSPORT": 6,
    "DECISION_WALK_PUT_READBACK_H1E": 8,
    "LIVE_REINSPECTION_AND_SKY_PROBE": 7,
    "ARM_RESERVE_POST_AUTHORIZATION_RELAY": 7,
    "UNALLOCATED_RESERVE": 2,
}


@dataclass(frozen=True)
class PhaseSpan:
    """One ordered monotonic timing observation."""

    phase_name: str
    started_monotonic_seconds: float
    ended_monotonic_seconds: float


@dataclass(frozen=True)
class TimingAuthority:
    """Validated nested timing evidence for one closure attempt."""

    closure_elapsed_seconds: float
    suffix_elapsed_seconds: float
    first_policy_readback_monotonic_seconds: float
    second_policy_readback_monotonic_seconds: float


class ClosureRouteError(ValueError):
    """The private closure failed closed."""


class StoredDecisionNoPostError(ClosureRouteError):
    """A direct decision exists but can never become POST authority."""

    status = "STORED_DECISION_NO_POST"


@dataclass(frozen=True)
class ClosureRequest:
    schema_version: int
    record_type: str
    activation_id: str
    generation: int
    candidate_identity_sha256: str
    initial_source_predecessor_version_id: str
    admission_version_arn: str
    numeric_binding_version_arn: str
    closure_budget_status: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class AuditEvidence:
    kind: str
    audit_identity_sha256: str
    invocation_identity_sha256: str
    closing_revision: int
    canonical_identity_sha256: str


@dataclass(frozen=True)
class StepReceipt:
    step_name: str
    status: str
    operation_identity_sha256: str
    audits: Tuple[AuditEvidence, ...]
    policy_readback_monotonic_seconds: Optional[Tuple[float, float]]
    nonce_ownership_sha256: Optional[str]
    coherent_readback: bool
    canonical_identity_sha256: str


@dataclass(frozen=True)
class TransactionReadback:
    transaction_name: str
    operation_identity_sha256: str
    index_coherent: bool
    control_coherent: bool
    action_coherent: bool
    canonical_identity_sha256: str


@dataclass(frozen=True)
class SourcePublication:
    source_kind: str
    predecessor_version_id: str
    bucket: str
    key: str
    version_id: str
    file_sha256: str
    body_sha256: str
    etag: str
    checksum_sha256_base64: str
    direct_request_id: str
    direct_server_date: str
    direct_response_authenticated: bool
    audit: AuditEvidence
    canonical_identity_sha256: str


@dataclass(frozen=True)
class SourceBatchReceipt:
    publications: Tuple[SourcePublication, ...]
    canonical_identity_sha256: str


@dataclass(frozen=True)
class DirectDecisionResponse:
    status_code: int
    expected_bucket_owner: str
    version_id: str
    etag: str
    checksum_sha256: str
    request_id: str
    response_date: str
    candidate_identity_sha256: str
    audit: AuditEvidence
    custody_nonce_sha256: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class LaunchOrExpireDecision:
    branch: str
    receipt: StepReceipt
    decision_writer_identity_sha256: Optional[str]
    terminal_v1_identity_sha256: Optional[str]
    canonical_identity_sha256: str


@dataclass(frozen=True)
class AdmissionClosureResult:
    version_arn: str
    decision_nonce_sha256: str
    internal_steps: Tuple[str, ...]
    action_states: Tuple[str, ...]
    post_audit: AuditEvidence
    classification: str
    invocation_count: int
    relay_call_count: int
    retry_count: int
    rearm_count: int
    relay_receipt_sha256: Optional[str]
    task9_result: AdmissionResult
    canonical_identity_sha256: str


@dataclass(frozen=True)
class DecisionClosureResult:
    status: str
    steps: Tuple[str, ...]
    authority_audit_kinds: Tuple[str, ...]
    handoff_audit_kind: str
    classification: str
    request_identity_sha256: str
    decision_response_identity_sha256: str
    admission_identity_sha256: str
    closure_spans: Tuple[PhaseSpan, ...]
    suffix_spans: Tuple[PhaseSpan, ...]
    timing_authority: TimingAuthority
    canonical_identity_sha256: str


@dataclass(frozen=True)
class TerminalDecisionClosureResult:
    status: str
    steps: Tuple[str, ...]
    request_identity_sha256: str
    decision_writer_identity_sha256: str
    terminal_v1_identity_sha256: str
    canonical_identity_sha256: str


REHEARSAL_PATH_PROOFS = (
    "EXACT_PRODUCTION_CLIENTS",
    "FULL_VERSION_PAGINATION",
    "NON_VPC_DECISION_PATH",
    "ISOLATED_ADMISSION_ATTESTATION_PATH",
    "HOST_NAT_PATH",
    "FROZEN_NAMESPACE_SCALE",
    "CONDITIONAL_WRITE",
    "DIRECT_RESPONSE",
    "EXACT_GET_HEAD",
    "H1E_MODELED_VALIDATION",
)
ZERO_PRODUCTION_EFFECTS = (
    ("PRODUCTION_SOURCE", 0),
    ("PRODUCTION_CLAIM", 0),
    ("PRODUCTION_DECISION", 0),
    ("PRODUCTION_ACTION_CONSUME", 0),
    ("SKY_POST", 0),
)
INJECTED_FAILURE_KINDS = (
    "THROTTLING",
    "PAGINATION",
    "NETWORK_AMBIGUITY",
)


@dataclass(frozen=True)
class RehearsalMeasurement:
    rehearsal_id: str
    provenance: str
    lambda_environment_id: str
    cold_start: bool
    path_proofs: Tuple[str, ...]
    canary_bucket: str
    canary_key: str
    canary_can_satisfy_production_authority: bool
    canary_worker_readable: bool
    canary_admission_readable: bool
    effect_counts: Tuple[Tuple[str, int], ...]
    relay_call_count: int
    closure_spans: Tuple[PhaseSpan, ...]
    suffix_spans: Tuple[PhaseSpan, ...]
    first_policy_readback_monotonic_seconds: float
    second_policy_readback_monotonic_seconds: float
    injected_failure_kind: str
    failure_unwind_seconds: float
    canonical_identity_sha256: str


@dataclass(frozen=True)
class RehearsalGate:
    status: str
    local_shape_verified: bool
    measurement_count: int
    cold_environment_count: int
    worst_closure_seconds: float
    worst_suffix_seconds: float
    production_effect_count: int
    failure_kinds: Tuple[str, ...]
    measurements_identity_sha256: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class DeployedGateDocument:
    account_id: str
    region: str
    run_id: str
    activation_id: str
    deployment_identity_sha256: str
    measurements: Tuple[RehearsalMeasurement, ...]
    measurement_count: int
    cold_environment_count: int
    environment_set_sha256: str
    measurements_identity_sha256: str
    canonical_body_sha256: str


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ClosureRouteError(label + " must be a lowercase SHA-256")
    return value


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value or not value.strip():
        raise ClosureRouteError(label + " must be a nonempty exact string")
    return value


def _identity_body(value: object) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def build_closure_request(
    *,
    activation_id: str,
    generation: int,
    candidate_identity_sha256: str,
    initial_source_predecessor_version_id: str,
    admission_version_arn: str,
    numeric_binding_version_arn: str,
    closure_budget_status: str,
) -> ClosureRequest:
    provisional = ClosureRequest(
        schema_version=1,
        record_type="glm52_task11_private_closure_request_v1",
        activation_id=activation_id,
        generation=generation,
        candidate_identity_sha256=candidate_identity_sha256,
        initial_source_predecessor_version_id=(
            initial_source_predecessor_version_id
        ),
        admission_version_arn=admission_version_arn,
        numeric_binding_version_arn=numeric_binding_version_arn,
        closure_budget_status=closure_budget_status,
        canonical_identity_sha256="",
    )
    result = ClosureRequest(
        **{
            **asdict(provisional),
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )
    return validate_closure_request(result)


def validate_closure_request(value: object) -> ClosureRequest:
    if type(value) is not ClosureRequest:
        raise ClosureRouteError("closure request must be exact and typed")
    if (
        value.schema_version != 1
        or value.record_type != "glm52_task11_private_closure_request_v1"
        or type(value.generation) is not int
        or value.generation <= 0
        or value.closure_budget_status != "CLOSURE_BUDGET_PROVEN"
    ):
        raise ClosureRouteError("closure request authority is invalid")
    _nonempty(value.activation_id, "activation id")
    _nonempty(
        value.initial_source_predecessor_version_id,
        "source predecessor VersionId",
    )
    _sha(value.candidate_identity_sha256, "candidate identity")
    if _ADMISSION_VERSION.fullmatch(value.admission_version_arn) is None:
        raise ClosureRouteError(
            "admission target must be the exact numeric published version"
        )
    if (
        _NUMERIC_BINDING_VERSION.fullmatch(
            value.numeric_binding_version_arn
        )
        is None
    ):
        raise ClosureRouteError(
            "numeric binding target must be the exact numeric published version"
        )
    if value.canonical_identity_sha256 != canonical_sha256(
        _identity_body(value)
    ):
        raise ClosureRouteError("closure request identity drifted")
    return value


def build_audit_evidence(
    *,
    kind: str,
    audit_identity_sha256: str,
    invocation_identity_sha256: str,
    closing_revision: int,
) -> AuditEvidence:
    provisional = AuditEvidence(
        kind=kind,
        audit_identity_sha256=audit_identity_sha256,
        invocation_identity_sha256=invocation_identity_sha256,
        closing_revision=closing_revision,
        canonical_identity_sha256="",
    )
    result = AuditEvidence(
        **{
            **asdict(provisional),
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )
    return _validate_audit(result)


def _validate_audit(value: object) -> AuditEvidence:
    if type(value) is not AuditEvidence:
        raise ClosureRouteError("authority audit must be exact and typed")
    if (
        value.kind not in (*AUTHORITY_AUDIT_KINDS, HANDOFF_AUDIT_KIND)
        or type(value.closing_revision) is not int
        or value.closing_revision < 0
    ):
        raise ClosureRouteError("authority audit grammar drifted")
    _sha(value.audit_identity_sha256, "audit identity")
    _sha(value.invocation_identity_sha256, "audit invocation identity")
    if value.canonical_identity_sha256 != canonical_sha256(
        _identity_body(value)
    ):
        raise ClosureRouteError("authority audit identity drifted")
    return value


def build_step_receipt(
    *,
    step_name: str,
    operation_identity_sha256: str,
    audits: Tuple[AuditEvidence, ...] = (),
    policy_readback_monotonic_seconds: Optional[Tuple[float, float]] = None,
    nonce_ownership_sha256: Optional[str] = None,
    coherent_readback: bool = False,
) -> StepReceipt:
    provisional = StepReceipt(
        step_name=step_name,
        status="PROVEN",
        operation_identity_sha256=operation_identity_sha256,
        audits=audits,
        policy_readback_monotonic_seconds=(
            policy_readback_monotonic_seconds
        ),
        nonce_ownership_sha256=nonce_ownership_sha256,
        coherent_readback=coherent_readback,
        canonical_identity_sha256="",
    )
    result = StepReceipt(
        **{
            **asdict(provisional),
            "audits": audits,
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )
    return _validate_step_receipt(result, step_name)


def _validate_step_receipt(
    value: object,
    expected_step: str,
) -> StepReceipt:
    if type(value) is not StepReceipt:
        raise ClosureRouteError(expected_step + " receipt must be exact")
    if (
        value.step_name != expected_step
        or value.status != "PROVEN"
        or type(value.audits) is not tuple
        or type(value.coherent_readback) is not bool
    ):
        raise ClosureRouteError(expected_step + " receipt grammar drifted")
    _sha(value.operation_identity_sha256, "operation identity")
    for audit in value.audits:
        _validate_audit(audit)
    if value.policy_readback_monotonic_seconds is not None:
        readbacks = value.policy_readback_monotonic_seconds
        if (
            type(readbacks) is not tuple
            or len(readbacks) != 2
            or _number(readbacks[1], "second policy readback")
            - _number(readbacks[0], "first policy readback")
            < 10
        ):
            raise ClosureRouteError(
                "policy readbacks must be at least 10 seconds apart"
            )
    if value.nonce_ownership_sha256 is not None:
        _sha(value.nonce_ownership_sha256, "nonce ownership")
    if value.canonical_identity_sha256 != canonical_sha256(
        _identity_body(value)
    ):
        raise ClosureRouteError(expected_step + " receipt identity drifted")
    return value


def build_launch_or_expire_decision(
    *,
    branch: str,
    receipt: StepReceipt,
    decision_writer_identity_sha256: Optional[str] = None,
    terminal_v1_identity_sha256: Optional[str] = None,
) -> LaunchOrExpireDecision:
    """Bind the branch model to its exact launch or terminal effects."""

    _validate_step_receipt(receipt, CLOSURE_STEPS[8])
    if branch == "LAUNCH":
        if (
            decision_writer_identity_sha256 is not None
            or terminal_v1_identity_sha256 is not None
        ):
            raise ClosureRouteError("launch branch invented terminal effects")
    elif branch == "EXPIRE":
        _sha(
            decision_writer_identity_sha256,
            "expire decision writer identity",
        )
        _sha(terminal_v1_identity_sha256, "terminal-v1 identity")
    else:
        raise ClosureRouteError("launch/expire branch is not closed")
    provisional = LaunchOrExpireDecision(
        branch=branch,
        receipt=receipt,
        decision_writer_identity_sha256=decision_writer_identity_sha256,
        terminal_v1_identity_sha256=terminal_v1_identity_sha256,
        canonical_identity_sha256="",
    )
    return LaunchOrExpireDecision(
        **{
            **asdict(provisional),
            "receipt": receipt,
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )


def _validate_launch_or_expire_decision(
    value: object,
) -> LaunchOrExpireDecision:
    if type(value) is not LaunchOrExpireDecision:
        raise ClosureRouteError(
            "launch/expire model must be exact and typed"
        )
    rebuilt = build_launch_or_expire_decision(
        branch=value.branch,
        receipt=value.receipt,
        decision_writer_identity_sha256=(
            value.decision_writer_identity_sha256
        ),
        terminal_v1_identity_sha256=value.terminal_v1_identity_sha256,
    )
    if value != rebuilt:
        raise ClosureRouteError("launch/expire model identity drifted")
    return value


def _terminal_closure_result(
    *,
    request: ClosureRequest,
    model: LaunchOrExpireDecision,
    steps: Tuple[str, ...],
) -> TerminalDecisionClosureResult:
    if (
        model.branch != "EXPIRE"
        or model.decision_writer_identity_sha256 is None
        or model.terminal_v1_identity_sha256 is None
    ):
        raise ClosureRouteError("terminal closure lacks exact effects")
    provisional = TerminalDecisionClosureResult(
        status="TERMINAL_V1_RECONCILIATION",
        steps=steps,
        request_identity_sha256=request.canonical_identity_sha256,
        decision_writer_identity_sha256=(
            model.decision_writer_identity_sha256
        ),
        terminal_v1_identity_sha256=model.terminal_v1_identity_sha256,
        canonical_identity_sha256="",
    )
    return TerminalDecisionClosureResult(
        **{
            **asdict(provisional),
            "steps": steps,
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )


def build_transaction_readback(
    *,
    transaction_name: str,
    operation_identity_sha256: str,
    index_coherent: bool,
    control_coherent: bool,
    action_coherent: bool,
) -> TransactionReadback:
    provisional = TransactionReadback(
        transaction_name=transaction_name,
        operation_identity_sha256=operation_identity_sha256,
        index_coherent=index_coherent,
        control_coherent=control_coherent,
        action_coherent=action_coherent,
        canonical_identity_sha256="",
    )
    return TransactionReadback(
        **{
            **asdict(provisional),
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )


def _validate_transaction_readback(
    value: object,
    *,
    transaction_name: str,
    operation_identity_sha256: str,
) -> TransactionReadback:
    if type(value) is not TransactionReadback:
        raise ClosureRouteError(
            transaction_name + " transaction readback must be exact"
        )
    if (
        value.transaction_name != transaction_name
        or value.operation_identity_sha256
        != operation_identity_sha256
        or value.index_coherent is not True
        or value.control_coherent is not True
        or value.action_coherent is not True
        or value.canonical_identity_sha256
        != canonical_sha256(_identity_body(value))
    ):
        raise ClosureRouteError(
            transaction_name
            + " index/control/action readback is not coherent"
        )
    return value


def build_source_publication(
    *,
    source_kind: str,
    predecessor_version_id: str,
    bucket: str,
    key: str,
    version_id: str,
    file_sha256: str,
    body_sha256: str,
    etag: str,
    checksum_sha256_base64: str,
    direct_request_id: str,
    direct_server_date: str,
    direct_response_authenticated: bool,
    audit: AuditEvidence,
) -> SourcePublication:
    row = {
        "source_kind": source_kind,
        "predecessor_version_id": predecessor_version_id,
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "file_sha256": file_sha256,
        "body_sha256": body_sha256,
        "etag": etag,
        "checksum_sha256_base64": checksum_sha256_base64,
        "direct_request_id": direct_request_id,
        "direct_server_date": direct_server_date,
        "direct_response_authenticated": direct_response_authenticated,
        "audit": audit,
    }
    return build_source_batch_receipt((row,)).publications[0]


def build_source_batch_receipt(
    rows: Tuple[Mapping[str, object], ...],
) -> SourceBatchReceipt:
    if type(rows) is not tuple:
        raise ClosureRouteError("source publications must be an exact tuple")
    publications = []
    for row in rows:
        if type(row) is not dict or set(row) != {
            "source_kind",
            "predecessor_version_id",
            "bucket",
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
            "etag",
            "checksum_sha256_base64",
            "direct_request_id",
            "direct_server_date",
            "direct_response_authenticated",
            "audit",
        }:
            raise ClosureRouteError("source publication field set drifted")
        if (
            type(row["source_kind"]) is not str
            or type(row["predecessor_version_id"]) is not str
            or type(row["bucket"]) is not str
            or type(row["key"]) is not str
            or type(row["version_id"]) is not str
            or type(row["file_sha256"]) is not str
            or type(row["body_sha256"]) is not str
            or type(row["etag"]) is not str
            or type(row["checksum_sha256_base64"]) is not str
            or type(row["direct_request_id"]) is not str
            or type(row["direct_server_date"]) is not str
            or type(row["direct_response_authenticated"]) is not bool
        ):
            raise ClosureRouteError(
                "source publication values must have exact types"
            )
        for field in (
            "source_kind",
            "predecessor_version_id",
            "bucket",
            "key",
            "version_id",
            "etag",
            "checksum_sha256_base64",
            "direct_request_id",
            "direct_server_date",
        ):
            _nonempty(row[field], "source publication " + field)
        _sha(row["file_sha256"], "source publication file identity")
        _sha(row["body_sha256"], "source publication body identity")
        provisional = SourcePublication(
            source_kind=row["source_kind"],
            predecessor_version_id=row["predecessor_version_id"],
            bucket=row["bucket"],
            key=row["key"],
            version_id=row["version_id"],
            file_sha256=row["file_sha256"],
            body_sha256=row["body_sha256"],
            etag=row["etag"],
            checksum_sha256_base64=row[
                "checksum_sha256_base64"
            ],
            direct_request_id=row["direct_request_id"],
            direct_server_date=row["direct_server_date"],
            direct_response_authenticated=row[
                "direct_response_authenticated"
            ],
            audit=_validate_audit(row["audit"]),
            canonical_identity_sha256="",
        )
        publications.append(
            SourcePublication(
                **{
                    **asdict(provisional),
                    "audit": provisional.audit,
                    "canonical_identity_sha256": canonical_sha256(
                        _identity_body(provisional)
                    ),
                }
            )
        )
    provisional_batch = SourceBatchReceipt(
        publications=tuple(publications),
        canonical_identity_sha256="",
    )
    return SourceBatchReceipt(
        publications=tuple(publications),
        canonical_identity_sha256=canonical_sha256(
            _identity_body(provisional_batch)
        ),
    )


def _validate_source_batch(
    value: object,
    *,
    initial_predecessor: str,
) -> SourceBatchReceipt:
    if type(value) is not SourceBatchReceipt:
        raise ClosureRouteError("five-source batch receipt must be exact")
    if (
        type(value.publications) is not tuple
        or tuple(item.source_kind for item in value.publications)
        != SOURCE_KINDS
    ):
        raise ClosureRouteError("five-source order or membership drifted")
    predecessor = initial_predecessor
    version_ids = []
    for index, item in enumerate(value.publications):
        if type(item) is not SourcePublication:
            raise ClosureRouteError("source publication must be exact")
        if (
            item.predecessor_version_id != predecessor
            or item.direct_response_authenticated is not True
            or item.audit.kind != AUTHORITY_AUDIT_KINDS[index]
        ):
            raise ClosureRouteError(
                "source VersionId dependency or direct custody drifted"
            )
        _nonempty(item.version_id, "service-assigned VersionId")
        if item.version_id in version_ids:
            raise ClosureRouteError("source VersionId was replayed")
        _validate_audit(item.audit)
        if item.canonical_identity_sha256 != canonical_sha256(
            _identity_body(item)
        ):
            raise ClosureRouteError("source publication identity drifted")
        version_ids.append(item.version_id)
        predecessor = item.version_id
    if value.canonical_identity_sha256 != canonical_sha256(
        _identity_body(value)
    ):
        raise ClosureRouteError("source batch identity drifted")
    return value


def build_direct_decision_response(
    *,
    status_code: int,
    expected_bucket_owner: str,
    version_id: str,
    etag: str,
    checksum_sha256: str,
    request_id: str,
    response_date: str,
    candidate_identity_sha256: str,
    audit: AuditEvidence,
    custody_nonce_sha256: str,
) -> DirectDecisionResponse:
    provisional = DirectDecisionResponse(
        status_code=status_code,
        expected_bucket_owner=expected_bucket_owner,
        version_id=version_id,
        etag=etag,
        checksum_sha256=checksum_sha256,
        request_id=request_id,
        response_date=response_date,
        candidate_identity_sha256=candidate_identity_sha256,
        audit=_validate_audit(audit),
        custody_nonce_sha256=custody_nonce_sha256,
        canonical_identity_sha256="",
    )
    result = DirectDecisionResponse(
        **{
            **asdict(provisional),
            "audit": provisional.audit,
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )
    return _validate_direct_response(result)


def _validate_direct_response(value: object) -> DirectDecisionResponse:
    if type(value) is not DirectDecisionResponse:
        raise ClosureRouteError("direct decision response must be exact")
    if (
        value.status_code != 200
        or value.expected_bucket_owner != "246813579024"
        or value.audit.kind != "DECISION"
    ):
        raise ClosureRouteError(
            "decision create was not a direct unambiguous exact 200"
        )
    for field in ("version_id", "etag", "request_id", "response_date"):
        _nonempty(getattr(value, field), field)
    for field in (
        "checksum_sha256",
        "candidate_identity_sha256",
        "custody_nonce_sha256",
    ):
        _sha(getattr(value, field), field)
    _validate_audit(value.audit)
    if value.canonical_identity_sha256 != canonical_sha256(
        _identity_body(value)
    ):
        raise ClosureRouteError("direct decision response identity drifted")
    return value


def build_admission_closure_result(
    *,
    version_arn: str,
    decision_nonce_sha256: str,
    internal_steps: Tuple[str, ...],
    action_states: Tuple[str, ...],
    post_audit: AuditEvidence,
    classification: str,
    invocation_count: int,
    relay_call_count: int,
    retry_count: int,
    rearm_count: int,
    relay_receipt_sha256: Optional[str],
    task9_result: AdmissionResult,
) -> AdmissionClosureResult:
    provisional = AdmissionClosureResult(
        version_arn=version_arn,
        decision_nonce_sha256=decision_nonce_sha256,
        internal_steps=internal_steps,
        action_states=action_states,
        post_audit=_validate_audit(post_audit),
        classification=classification,
        invocation_count=invocation_count,
        relay_call_count=relay_call_count,
        retry_count=retry_count,
        rearm_count=rearm_count,
        relay_receipt_sha256=relay_receipt_sha256,
        task9_result=task9_result,
        canonical_identity_sha256="",
    )
    result = AdmissionClosureResult(
        **{
            **asdict(provisional),
            "internal_steps": internal_steps,
            "action_states": action_states,
            "post_audit": provisional.post_audit,
            "task9_result": provisional.task9_result,
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )
    return _validate_admission_result(result, version_arn, decision_nonce_sha256)


def _validate_admission_result(
    value: object,
    version_arn: str,
    decision_nonce_sha256: str,
) -> AdmissionClosureResult:
    if type(value) is not AdmissionClosureResult:
        raise ClosureRouteError("launch-admission result must be exact")
    if (
        value.version_arn != version_arn
        or _ADMISSION_VERSION.fullmatch(value.version_arn) is None
        or value.decision_nonce_sha256 != decision_nonce_sha256
        or value.internal_steps != CLOSURE_STEPS[20:24]
        or value.action_states
        != ("POST_STARTED", "POST_AUTHORIZED", "POST_CLASSIFIED")
        or value.post_audit.kind != "SKY_POST"
        or value.classification
        not in ("ACCEPTED", "KNOWN_REJECTED", "AMBIGUOUS")
        or value.invocation_count != 1
        or value.relay_call_count != 1
        or value.retry_count != 0
        or value.rearm_count != 0
        or type(value.task9_result) is not AdmissionResult
        or value.task9_result.classification != value.classification
        or value.task9_result.durable_state != "POST_CLASSIFIED"
    ):
        raise ClosureRouteError(
            "launch admission was not one version, one wire, and no retry"
        )
    _sha(value.decision_nonce_sha256, "decision nonce")
    _validate_audit(value.post_audit)
    if (
        value.task9_result.response_identity_sha256 is not None
        and _SHA.fullmatch(
            value.task9_result.response_identity_sha256
        )
        is None
    ):
        raise ClosureRouteError("Task 9 response identity is invalid")
    if (
        value.task9_result.request_id is not None
        and (
            type(value.task9_result.request_id) is not str
            or not value.task9_result.request_id
        )
    ):
        raise ClosureRouteError("Task 9 request id is invalid")
    if value.classification == "AMBIGUOUS":
        if value.relay_receipt_sha256 is not None:
            raise ClosureRouteError(
                "ambiguous admission cannot invent a relay receipt"
            )
    else:
        _sha(value.relay_receipt_sha256, "relay receipt")
    if value.canonical_identity_sha256 != canonical_sha256(
        _identity_body(value)
    ):
        raise ClosureRouteError("launch-admission result identity drifted")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(label + " must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(label + " must be a finite number")
    return converted


def _validate_spans(
    *,
    spans: object,
    ceilings: Mapping[str, int],
    label: str,
) -> Tuple[Tuple[PhaseSpan, ...], float]:
    if type(spans) is not tuple:
        raise ValueError(label + " spans must be an exact tuple")
    expected_names = tuple(ceilings)
    if tuple(
        span.phase_name if isinstance(span, PhaseSpan) else None
        for span in spans
    ) != expected_names:
        raise ValueError(label + " phase order or membership drifted")

    previous_end = None
    elapsed = 0.0
    validated = []
    for index, span in enumerate(spans):
        if type(span) is not PhaseSpan:
            raise ValueError(label + " spans must be exact PhaseSpan values")
        start = _number(
            span.started_monotonic_seconds,
            label + " start",
        )
        end = _number(
            span.ended_monotonic_seconds,
            label + " end",
        )
        if start < 0 or end < start:
            raise ValueError(label + " monotonic clock moved backwards")
        if previous_end is not None and start != previous_end:
            raise ValueError(
                label + " spans overlap, contain a gap, or moved backwards"
            )
        duration = end - start
        ceiling = ceilings[expected_names[index]]
        if duration > ceiling:
            raise ValueError(label + " phase exceeded its ceiling")
        elapsed += duration
        previous_end = end
        validated.append(span)
    wall_elapsed = (
        float(validated[-1].ended_monotonic_seconds)
        - float(validated[0].started_monotonic_seconds)
    )
    if not math.isclose(
        elapsed,
        wall_elapsed,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(label + " ledger does not account for all wall time")
    return tuple(validated), elapsed


def validate_timing_ledgers(
    *,
    closure_spans: object,
    suffix_spans: object,
    first_policy_readback_monotonic_seconds: object,
    second_policy_readback_monotonic_seconds: object,
) -> TimingAuthority:
    """Validate the exact closure/suffix ledgers and policy-readback cadence."""

    closure, closure_elapsed = _validate_spans(
        spans=closure_spans,
        ceilings=CLOSURE_PHASE_CEILINGS,
        label="closure",
    )
    suffix, suffix_elapsed = _validate_spans(
        spans=suffix_spans,
        ceilings=SUFFIX_PHASE_CEILINGS,
        label="suffix",
    )
    if closure_elapsed > sum(CLOSURE_PHASE_CEILINGS.values()):
        raise ValueError("closure ledger exceeded 600 seconds")
    if suffix_elapsed > sum(SUFFIX_PHASE_CEILINGS.values()):
        raise ValueError("suffix ledger exceeded 58 seconds")

    nested = closure[4]
    if (
        suffix[0].started_monotonic_seconds
        != nested.started_monotonic_seconds
        or suffix[-1].ended_monotonic_seconds
        != nested.ended_monotonic_seconds
    ):
        raise ValueError("suffix ledger is not the exact nested fresh suffix")

    first = _number(
        first_policy_readback_monotonic_seconds,
        "first policy readback",
    )
    second = _number(
        second_policy_readback_monotonic_seconds,
        "second policy readback",
    )
    policy_phase = suffix[2]
    if (
        first < policy_phase.started_monotonic_seconds
        or second > policy_phase.ended_monotonic_seconds
        or second - first < 10
    ):
        raise ValueError("policy readbacks must be at least 10 seconds apart")

    return TimingAuthority(
        closure_elapsed_seconds=closure_elapsed,
        suffix_elapsed_seconds=suffix_elapsed,
        first_policy_readback_monotonic_seconds=first,
        second_policy_readback_monotonic_seconds=second,
    )


def build_rehearsal_measurement(
    *,
    rehearsal_id: str,
    provenance: str,
    lambda_environment_id: str,
    cold_start: bool,
    path_proofs: Tuple[str, ...],
    canary_bucket: str,
    canary_key: str,
    canary_can_satisfy_production_authority: bool,
    canary_worker_readable: bool,
    canary_admission_readable: bool,
    effect_counts: Tuple[Tuple[str, int], ...],
    relay_call_count: int,
    closure_spans: Tuple[PhaseSpan, ...],
    suffix_spans: Tuple[PhaseSpan, ...],
    first_policy_readback_monotonic_seconds: float,
    second_policy_readback_monotonic_seconds: float,
    injected_failure_kind: str,
    failure_unwind_seconds: float,
) -> RehearsalMeasurement:
    """Build one self-hashed deployed-shaped no-POST observation."""

    provisional = RehearsalMeasurement(
        rehearsal_id=rehearsal_id,
        provenance=provenance,
        lambda_environment_id=lambda_environment_id,
        cold_start=cold_start,
        path_proofs=path_proofs,
        canary_bucket=canary_bucket,
        canary_key=canary_key,
        canary_can_satisfy_production_authority=(
            canary_can_satisfy_production_authority
        ),
        canary_worker_readable=canary_worker_readable,
        canary_admission_readable=canary_admission_readable,
        effect_counts=effect_counts,
        relay_call_count=relay_call_count,
        closure_spans=closure_spans,
        suffix_spans=suffix_spans,
        first_policy_readback_monotonic_seconds=(
            first_policy_readback_monotonic_seconds
        ),
        second_policy_readback_monotonic_seconds=(
            second_policy_readback_monotonic_seconds
        ),
        injected_failure_kind=injected_failure_kind,
        failure_unwind_seconds=failure_unwind_seconds,
        canonical_identity_sha256="",
    )
    result = RehearsalMeasurement(
        **{
            **asdict(provisional),
            "path_proofs": path_proofs,
            "effect_counts": effect_counts,
            "closure_spans": closure_spans,
            "suffix_spans": suffix_spans,
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )
    _validate_rehearsal_measurement(result)
    return result


def _validate_rehearsal_measurement(
    value: object,
) -> Tuple[RehearsalMeasurement, TimingAuthority]:
    if type(value) is not RehearsalMeasurement:
        raise ValueError("rehearsal measurement must be exact and typed")
    _nonempty(value.rehearsal_id, "rehearsal id")
    _nonempty(value.lambda_environment_id, "Lambda environment id")
    if (
        value.provenance not in ("LOCAL_SIMULATOR", "DEPLOYED_REHEARSAL")
        or type(value.cold_start) is not bool
        or value.path_proofs != REHEARSAL_PATH_PROOFS
        or value.canary_bucket != "keep-glm52-h1g-rehearsal"
        or not value.canary_key.startswith("rehearsal/canary/")
        or value.canary_can_satisfy_production_authority is not False
        or value.canary_worker_readable is not False
        or value.canary_admission_readable is not False
        or value.effect_counts != ZERO_PRODUCTION_EFFECTS
        or type(value.relay_call_count) is not int
        or value.relay_call_count != 0
    ):
        raise ValueError(
            "rehearsal path, canary isolation, or zero-effect proof drifted"
        )
    timing = validate_timing_ledgers(
        closure_spans=value.closure_spans,
        suffix_spans=value.suffix_spans,
        first_policy_readback_monotonic_seconds=(
            value.first_policy_readback_monotonic_seconds
        ),
        second_policy_readback_monotonic_seconds=(
            value.second_policy_readback_monotonic_seconds
        ),
    )
    unwind = _number(value.failure_unwind_seconds, "failure unwind")
    if value.injected_failure_kind == "NONE":
        if unwind != 0:
            raise ValueError("non-failure rehearsal invented unwind time")
    elif value.injected_failure_kind in INJECTED_FAILURE_KINDS:
        if unwind <= 0 or unwind > 55:
            raise ValueError("injected failure did not unwind within 55 seconds")
    else:
        raise ValueError("injected failure kind is not frozen")
    if value.canonical_identity_sha256 != canonical_sha256(
        _identity_body(value)
    ):
        raise ValueError("rehearsal measurement identity drifted")
    return value, timing


def rehearsal_measurement_from_mapping(
    value: object,
) -> RehearsalMeasurement:
    """Parse one exact JSON measurement without coercing boundary types."""

    if type(value) is not dict:
        raise ValueError("rehearsal measurement JSON must be an object")
    expected = set(RehearsalMeasurement.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("rehearsal measurement JSON fields drifted")
    closure_raw = value["closure_spans"]
    suffix_raw = value["suffix_spans"]
    if type(closure_raw) is not list or type(suffix_raw) is not list:
        raise ValueError("timing span JSON must use exact arrays")

    def spans(raw: object) -> Tuple[PhaseSpan, ...]:
        if type(raw) is not list:
            raise ValueError("timing span JSON must use an exact array")
        result = []
        for item in raw:
            if (
                type(item) is not dict
                or set(item)
                != {
                    "phase_name",
                    "started_monotonic_seconds",
                    "ended_monotonic_seconds",
                }
            ):
                raise ValueError("timing span JSON fields drifted")
            result.append(PhaseSpan(**item))
        return tuple(result)

    path_proofs = value["path_proofs"]
    effect_counts = value["effect_counts"]
    if (
        type(path_proofs) is not list
        or any(type(item) is not str for item in path_proofs)
        or type(effect_counts) is not list
        or any(
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            for item in effect_counts
        )
    ):
        raise ValueError("rehearsal path/effect JSON types drifted")
    parsed = RehearsalMeasurement(
        **{
            **value,
            "path_proofs": tuple(path_proofs),
            "effect_counts": tuple(
                (item[0], item[1]) for item in effect_counts
            ),
            "closure_spans": spans(closure_raw),
            "suffix_spans": spans(suffix_raw),
        }
    )
    _validate_rehearsal_measurement(parsed)
    return parsed


def build_deployed_gate_document(
    *,
    account_id: str,
    region: str,
    run_id: str,
    activation_id: str,
    deployment_identity_sha256: str,
    measurements: Tuple[RehearsalMeasurement, ...],
) -> bytes:
    """Render one canonical deployed-measurement object without a status."""

    if (
        account_id != "246813579024"
        or region != "us-west-2"
        or run_id != "glm52-sky-20260724"
        or type(activation_id) is not str
        or not activation_id
        or type(measurements) is not tuple
    ):
        raise ValueError("deployed gate coordinates are not frozen")
    _sha(deployment_identity_sha256, "deployment identity")
    gate = verify_no_post_rehearsals(measurements)
    if any(item.provenance != "DEPLOYED_REHEARSAL" for item in measurements):
        raise ValueError("local measurements cannot enter a deployed gate")
    environments = sorted(
        item.lambda_environment_id for item in measurements
    )
    body = {
        "schema_version": 1,
        "record_type": "glm52_task11_deployed_closure_gate_v1",
        "account_id": account_id,
        "region": region,
        "run_id": run_id,
        "activation_id": activation_id,
        "deployment_identity_sha256": deployment_identity_sha256,
        "measurements": [asdict(item) for item in measurements],
        "measurement_count": gate.measurement_count,
        "cold_environment_count": gate.cold_environment_count,
        "environment_set_sha256": canonical_sha256(environments),
        "measurements_identity_sha256": gate.measurements_identity_sha256,
    }
    body["canonical_body_sha256"] = canonical_sha256(body)
    return canonical_json_bytes(body)


def parse_deployed_gate_document(
    raw: object,
    *,
    activation_id: str,
    deployment_identity_sha256: str,
) -> DeployedGateDocument:
    """Authenticate canonical gate bytes and every contained measurement."""

    if type(raw) is not bytes or not raw or len(raw) > 2 * 1024 * 1024:
        raise ValueError("deployed gate bytes are absent or oversized")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("deployed gate is not canonical JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError("deployed gate bytes are not canonical")
    expected = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "deployment_identity_sha256",
        "measurements",
        "measurement_count",
        "cold_environment_count",
        "environment_set_sha256",
        "measurements_identity_sha256",
        "canonical_body_sha256",
    }
    if set(value) != expected:
        raise ValueError("deployed gate field set drifted")
    body = dict(value)
    identity = body.pop("canonical_body_sha256")
    if (
        value["schema_version"] != 1
        or value["record_type"]
        != "glm52_task11_deployed_closure_gate_v1"
        or value["account_id"] != "246813579024"
        or value["region"] != "us-west-2"
        or value["run_id"] != "glm52-sky-20260724"
        or value["activation_id"] != activation_id
        or value["deployment_identity_sha256"]
        != deployment_identity_sha256
        or type(value["measurement_count"]) is not int
        or type(value["cold_environment_count"]) is not int
        or type(value["measurements"]) is not list
        or identity != canonical_sha256(body)
    ):
        raise ValueError("deployed gate identity or coordinates drifted")
    measurements = tuple(
        rehearsal_measurement_from_mapping(item)
        for item in value["measurements"]
    )
    gate = verify_no_post_rehearsals(measurements)
    environments = sorted(
        item.lambda_environment_id for item in measurements
    )
    if (
        any(item.provenance != "DEPLOYED_REHEARSAL" for item in measurements)
        or value["measurement_count"] != gate.measurement_count
        or value["cold_environment_count"] != gate.cold_environment_count
        or value["environment_set_sha256"]
        != canonical_sha256(environments)
        or value["measurements_identity_sha256"]
        != gate.measurements_identity_sha256
    ):
        raise ValueError("deployed measurement set is not authenticated")
    return DeployedGateDocument(
        account_id=value["account_id"],
        region=value["region"],
        run_id=value["run_id"],
        activation_id=value["activation_id"],
        deployment_identity_sha256=value[
            "deployment_identity_sha256"
        ],
        measurements=measurements,
        measurement_count=value["measurement_count"],
        cold_environment_count=value["cold_environment_count"],
        environment_set_sha256=value["environment_set_sha256"],
        measurements_identity_sha256=value[
            "measurements_identity_sha256"
        ],
        canonical_body_sha256=identity,
    )


def verify_no_post_rehearsals(
    measurements: object,
) -> RehearsalGate:
    """Validate the local/deployed-shaped matrix without inventing live proof.

    Task 11 intentionally emits ``CLOSURE_BUDGET_UNPROVEN``.  A later live
    deployment must authenticate the collected measurements before replacing
    that status; self-authored local or serialized records are never enough.
    """

    if type(measurements) is not tuple or len(measurements) < 20:
        raise ValueError("at least 20 rehearsal measurements are required")
    validated = []
    timings = []
    for measurement in measurements:
        item, timing = _validate_rehearsal_measurement(measurement)
        validated.append(item)
        timings.append(timing)
    rehearsal_ids = tuple(item.rehearsal_id for item in validated)
    if len(set(rehearsal_ids)) != len(rehearsal_ids):
        raise ValueError("rehearsal ids must be unique")
    cold_environments = tuple(
        item.lambda_environment_id for item in validated if item.cold_start
    )
    if (
        len(cold_environments) < 5
        or len(set(cold_environments)) != len(cold_environments)
    ):
        raise ValueError("at least five distinct cold environments are required")
    present_failures = {
        item.injected_failure_kind
        for item in validated
        if item.injected_failure_kind != "NONE"
    }
    if present_failures != set(INJECTED_FAILURE_KINDS):
        raise ValueError("the exact injected-failure matrix is incomplete")
    effects = sum(
        count
        for item in validated
        for _kind, count in item.effect_counts
    )
    if effects != 0:
        raise ValueError("rehearsal observed a production effect")
    measurements_identity = canonical_sha256(
        [item.canonical_identity_sha256 for item in validated]
    )
    provisional = RehearsalGate(
        status="CLOSURE_BUDGET_UNPROVEN",
        local_shape_verified=True,
        measurement_count=len(validated),
        cold_environment_count=len(set(cold_environments)),
        worst_closure_seconds=max(
            timing.closure_elapsed_seconds for timing in timings
        ),
        worst_suffix_seconds=max(
            timing.suffix_elapsed_seconds for timing in timings
        ),
        production_effect_count=effects,
        failure_kinds=INJECTED_FAILURE_KINDS,
        measurements_identity_sha256=measurements_identity,
        canonical_identity_sha256="",
    )
    return RehearsalGate(
        **{
            **asdict(provisional),
            "failure_kinds": provisional.failure_kinds,
            "canonical_identity_sha256": canonical_sha256(
                _identity_body(provisional)
            ),
        }
    )


def build_task11_workflow_definition() -> Mapping[str, object]:
    """Build the version-pinned Standard support workflow.

    The synchronous decision Lambda keeps direct-response custody through the
    H.1d suffix.  Its result is then retained beside the original immutable
    closure request so the separately published support-continuation Lambda
    can exact-read the handoff and drain the resulting work.
    """

    definition = {
        "StartAt": "AUTHENTICATE_DEPLOYED_CLOSURE_BUDGET",
        "States": {
            "AUTHENTICATE_DEPLOYED_CLOSURE_BUDGET": {
                "Type": "Task",
                "Resource": {"Ref": "BudgetGateVersion"},
                "Parameters": {
                    "schema_version": 1,
                    "record_type": "glm52_task11_budget_gate_request_v1",
                },
                "ResultPath": "$.trusted_closure_budget",
                "TimeoutSeconds": 120,
                "Next": "VALIDATE_CLOSURE_BUDGET",
            },
            "VALIDATE_CLOSURE_BUDGET": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.trusted_closure_budget.status",
                        "StringEquals": "CLOSURE_BUDGET_PROVEN",
                        "Next": "DECISION_CLOSURE",
                    }
                ],
                "Default": "CLOSURE_BUDGET_UNPROVEN",
            },
            "CLOSURE_BUDGET_UNPROVEN": {
                "Type": "Fail",
                "Error": "CLOSURE_BUDGET_UNPROVEN",
            },
            "DECISION_CLOSURE": {
                "Type": "Task",
                "Resource": {"Ref": "DecisionVersion"},
                "TimeoutSeconds": STEP_FUNCTION_TASK_TIMEOUT_SECONDS,
                "ResultPath": "$.decision_result",
                "Next": "VALIDATE_APPROVED_TASK11_HANDOFF",
            },
        },
    }
    from .task12_support_continuation_adapter import (
        SUPPORT_OPERATION_SEQUENCE,
    )
    from .task12_workflows import (
        support_continuation_operation_input,
    )

    states = definition["States"]
    for index, operation in enumerate(SUPPORT_OPERATION_SEQUENCE):
        operation_input = support_continuation_operation_input(
            index=index,
            previous_result_path=(
                "$.terminal_attempt"
                if operation == "REQUEST_SUPPORT_FINALIZATION"
                else "$.support_last_result"
            ),
        )
        state: dict[str, object] = {
            "Type": "Task",
            "Resource": {"Ref": "SupportDeadlineVersion"},
            "Parameters": {
                "schema_version": 1,
                "record_type": (
                    "glm52_task12_support_continuation_invocation_v1"
                ),
                "operation_kind": operation,
                "operation_input": operation_input,
            },
            "ResultPath": (
                "$.terminal_attempt"
                if operation
                == "CREATE_OR_RECONCILE_PRODUCTION_TERMINAL_V2"
                else "$.support_last_result"
            ),
            "TimeoutSeconds": 540,
        }
        if operation == (
            "CREATE_OR_RECONCILE_PRODUCTION_TERMINAL_V2"
        ):
            state["Next"] = "CHECK_TERMINAL_V2_RESULT"
        elif index == len(SUPPORT_OPERATION_SEQUENCE) - 1:
            state["End"] = True
        else:
            state["Next"] = SUPPORT_OPERATION_SEQUENCE[index + 1]
        states[operation] = state
    states["CHECK_TERMINAL_V2_RESULT"] = {
        "Type": "Choice",
        "Choices": [
            {
                "Variable": "$.terminal_attempt.outcome",
                "StringEquals": "SUCCEEDED",
                "Next": "REQUEST_SUPPORT_FINALIZATION",
            },
            {
                "Variable": "$.terminal_attempt.outcome",
                "StringEquals": "INCOMPLETE",
                "Next": "WAIT_FOR_TERMINAL_V2",
            },
        ],
        "Default": "TERMINAL_V2_RESULT_INVALID",
    }
    states["WAIT_FOR_TERMINAL_V2"] = {
        "Type": "Wait",
        "Seconds": 5,
        "Next": "CREATE_OR_RECONCILE_PRODUCTION_TERMINAL_V2",
    }
    states["TERMINAL_V2_RESULT_INVALID"] = {
        "Type": "Fail",
        "Error": "TERMINAL_V2_RESULT_INVALID",
    }
    return definition


_SERVICE_STEPS = (
    ("prove_preauthorized_batch_template", CLOSURE_STEPS[0]),
    ("acquire_cfn_quiescence", CLOSURE_STEPS[1]),
    (
        "revalidate_runtime_attachments_and_seals",
        CLOSURE_STEPS[2],
    ),
)


def _private_nonce(
    nonce_source: Callable[[], bytes],
    label: str,
) -> Tuple[bytes, str]:
    nonce = nonce_source()
    if type(nonce) is not bytes or len(nonce) < 16:
        raise ClosureRouteError(label + " nonce must be private random bytes")
    return nonce, hashlib.sha256(nonce).hexdigest()


def _method(services: object, name: str) -> Callable[..., object]:
    value = getattr(services, name, None)
    if not callable(value):
        raise ClosureRouteError("closure dependency is absent: " + name)
    return value


def _remaining_milliseconds(services: object, label: str) -> int:
    remaining = _method(services, "remaining_milliseconds")()
    if type(remaining) is not int or remaining < 0:
        raise ClosureRouteError(label + " remaining time is invalid")
    return remaining


def _validate_h1d_live_result(
    value: object,
    *,
    activation_id: str,
) -> H1dLiveAuthorityResult:
    if type(value) is not H1dLiveAuthorityResult:
        raise ClosureRouteError(
            "Task 8 authority must be the live typed result"
        )
    identity_body = asdict(value) if type(value) is H1dLiveAuthorityResult else {}
    if identity_body:
        identity_body.pop("canonical_identity_sha256")
        identity_body["phase_elapsed_seconds"] = format(
            value.phase_elapsed_seconds,
            "f",
        )
    if (
        value.schema_version != 1
        or value.record_type != "glm52_h1d_live_authority_result_v1"
        or value.account_id != "246813579024"
        or value.region != "us-west-2"
        or value.run_id != "glm52-sky-20260724"
        or value.activation_id != activation_id
        or type(value.page_identities) is not tuple
        or type(value.family_identities) is not dict
        or type(value.phase_elapsed_seconds) is not Decimal
        or value.phase_elapsed_seconds < Decimal("0")
        or value.phase_elapsed_seconds > Decimal("7")
        or value.canonical_identity_sha256
        != canonical_sha256(identity_body)
    ):
        raise ClosureRouteError("Task 8 live authority drifted")
    return value


def _validate_attestation_result(
    value: object,
) -> AttestationResult:
    if type(value) is not AttestationResult:
        raise ClosureRouteError(
            "Task 9 attestation must be the live typed result"
        )
    if (
        value.schema_version != 1
        or value.record_type != "glm52_sky_attestation_v1"
        or value.account_id != "246813579024"
        or value.region != "us-west-2"
        or value.run_id != "glm52-sky-20260724"
        or type(value.sky_roles) is not tuple
        or value.canonical_identity_sha256
        != canonical_sha256(_identity_body(value))
    ):
        raise ClosureRouteError("Task 9 attestation drifted")
    for field in (
        "freshness_nonce",
        "direct_response_request_id",
        "sky_user_identity",
        "token_expires_at",
        "observed_at",
    ):
        _nonempty(getattr(value, field), "attestation " + field)
    for field in (
        "tls_peer_certificate_sha256",
        "effective_controller_identity_sha256",
        "sky_identity_contract_sha256",
    ):
        _sha(getattr(value, field), "attestation " + field)
    return value


def assert_no_stored_decision_submit_surface(services: object) -> None:
    """Reject every exported stored-decision or raw/retry effect surface."""

    forbidden = (
        "submit_stored_decision",
        "consume_stored_decision",
        "retry_post",
        "rearm_post",
        "post",
        "send",
        "jobs_launch",
    )
    exposed = [
        name for name in forbidden if callable(getattr(services, name, None))
    ]
    if exposed:
        raise ClosureRouteError(
            "closure boundary exposes forbidden effect surface: "
            + ",".join(exposed)
        )


def transition_abandoned_arm(
    *,
    current_state: str,
    owner_hard_expired: bool,
    owner_execution_terminal: bool,
    zero_side_effect: bool,
) -> str:
    """Authorize the sole ``ARMED -> ABANDONED`` no-effect transition."""

    if (
        current_state != "ARMED"
        or owner_hard_expired is not True
        or owner_execution_terminal is not True
        or zero_side_effect is not True
    ):
        raise ClosureRouteError(
            "arm abandonment requires hard expiry, terminal owner, "
            "and exact zero-side-effect proof"
        )
    return "ABANDONED"


def execute_decision_closure(
    request: ClosureRequest,
    *,
    services: object,
    nonce_source: Optional[Callable[[], bytes]] = None,
    monotonic_clock: Optional[Callable[[], float]] = None,
    sleeper: Optional[Callable[[float], None]] = None,
) -> Union[DecisionClosureResult, TerminalDecisionClosureResult]:
    """Execute one exact private closure with no replay or retry edge.

    All effectful dependencies are injected and named by their single
    authority operation.  The direct decision response stays in this stack
    frame and is passed by object identity to exact-read and H.1e validation.
    """

    request = validate_closure_request(request)
    if nonce_source is None:
        def system_nonce_source() -> bytes:
            return secrets.token_bytes(32)

        nonce_source = system_nonce_source
    if not callable(nonce_source):
        raise ClosureRouteError("private nonce source is absent")
    if monotonic_clock is None:
        monotonic_clock = time.monotonic
    if sleeper is None:
        sleeper = time.sleep
    if not callable(monotonic_clock) or not callable(sleeper):
        raise ClosureRouteError("monotonic clock or bounded sleeper is absent")
    assert_no_stored_decision_submit_surface(services)

    steps = []
    authority_audits = []
    handoff_audit = None
    operation_identities = set()
    direct_create_attempted = False
    action_consumed = False
    _custody_nonce, custody_nonce_sha256 = _private_nonce(
        nonce_source,
        "decision custody",
    )
    sampled = _number(monotonic_clock(), "closure monotonic start")
    if sampled < 0:
        raise ClosureRouteError("closure monotonic start is negative")
    closure_started = sampled
    deadline = closure_started + CLOSURE_DEADLINE_SECONDS

    def sample_clock(label: str) -> float:
        nonlocal sampled
        current = _number(monotonic_clock(), label + " monotonic time")
        if current < sampled:
            raise ClosureRouteError("closure monotonic clock moved backwards")
        sampled = current
        if current > deadline:
            raise ClosureRouteError(
                "independent 720-second closure deadline expired after "
                + label
            )
        return current

    def call_service(name: str, **kwargs: object) -> object:
        result = _method(services, name)(**kwargs)
        sample_clock(name)
        return result

    def record_receipt(
        value: object,
        expected_step: str,
        *,
        expected_audits: Tuple[str, ...] = (),
        expected_nonce: Optional[str] = None,
        coherent: bool = False,
        include_step: bool = True,
    ) -> StepReceipt:
        receipt = _validate_step_receipt(value, expected_step)
        if receipt.operation_identity_sha256 in operation_identities:
            raise ClosureRouteError("cached operation receipt was replayed")
        operation_identities.add(receipt.operation_identity_sha256)
        if tuple(audit.kind for audit in receipt.audits) != expected_audits:
            raise ClosureRouteError(expected_step + " audit assignment drifted")
        if expected_nonce is not None:
            if receipt.nonce_ownership_sha256 != expected_nonce:
                raise ClosureRouteError(
                    expected_step + " nonce ownership drifted"
                )
        elif receipt.nonce_ownership_sha256 is not None:
            raise ClosureRouteError(
                expected_step + " invented nonce ownership"
            )
        if receipt.coherent_readback is not coherent:
            raise ClosureRouteError(
                expected_step + " coherent-readback grammar drifted"
            )
        if include_step:
            steps.append(expected_step)
        for audit in receipt.audits:
            if audit.kind == HANDOFF_AUDIT_KIND:
                nonlocal handoff_audit
                handoff_audit = audit
            else:
                authority_audits.append(audit)
        return receipt

    def readback_transaction(
        *,
        transaction_name: str,
        operation_identity_sha256: str,
    ) -> TransactionReadback:
        return _validate_transaction_readback(
            call_service(
                "coherent_readback_after_transaction",
                request=request,
                transaction_name=transaction_name,
                operation_identity_sha256=operation_identity_sha256,
                custody_nonce_sha256=custody_nonce_sha256,
            ),
            transaction_name=transaction_name,
            operation_identity_sha256=operation_identity_sha256,
        )

    try:
        for method_name, step_name in _SERVICE_STEPS:
            record_receipt(
                call_service(
                    method_name,
                    request=request,
                    custody_nonce_sha256=custody_nonce_sha256,
                ),
                step_name,
            )
        closure_phase_0_end = sampled

        record_receipt(
            call_service(
                "warm_clients_and_construct",
                request=request,
                custody_nonce_sha256=custody_nonce_sha256,
            ),
            "CLIENT_WARMING_AND_IMMUTABLE_CONSTRUCTION",
            include_step=False,
        )
        closure_phase_1_end = sampled
        record_receipt(
            call_service(
                "size_non_authoritative",
                request=request,
                custody_nonce_sha256=custody_nonce_sha256,
            ),
            "NON_AUTHORITATIVE_SIZING",
            include_step=False,
        )
        closure_phase_2_end = sampled
        record_receipt(
            call_service(
                "stable_tls_sky_identity_preflight",
                request=request,
                custody_nonce_sha256=custody_nonce_sha256,
            ),
            "STABLE_TLS_SKY_IDENTITY_PREFLIGHT",
            include_step=False,
        )
        closure_phase_3_end = sampled
        suffix_started = sampled

        predecessor = request.initial_source_predecessor_version_id
        publications = []
        for index, method_name in enumerate(SOURCE_SERVICE_METHODS):
            publication = call_service(
                method_name,
                request=request,
                predecessor_version_id=predecessor,
                prior_publications=tuple(publications),
                custody_nonce_sha256=custody_nonce_sha256,
            )
            if type(publication) is not SourcePublication:
                raise ClosureRouteError(
                    SOURCE_KINDS[index]
                    + " publication must be the exact typed result"
                )
            if (
                publication.source_kind != SOURCE_KINDS[index]
                or publication.predecessor_version_id != predecessor
                or publication.direct_response_authenticated is not True
                or publication.audit.kind
                != AUTHORITY_AUDIT_KINDS[index]
            ):
                raise ClosureRouteError(
                    "source VersionId dependency or direct custody drifted"
                )
            _nonempty(
                publication.version_id,
                "service-assigned VersionId",
            )
            _validate_audit(publication.audit)
            if (
                publication.canonical_identity_sha256
                != canonical_sha256(_identity_body(publication))
            ):
                raise ClosureRouteError(
                    "source publication identity drifted"
                )
            if (
                publication.canonical_identity_sha256
                in operation_identities
            ):
                raise ClosureRouteError("source operation was replayed")
            operation_identities.add(
                publication.canonical_identity_sha256
            )
            readback_transaction(
                transaction_name=method_name,
                operation_identity_sha256=(
                    publication.canonical_identity_sha256
                ),
            )
            publications.append(publication)
            authority_audits.append(publication.audit)
            predecessor = publication.version_id
        source_batch = _validate_source_batch(
            SourceBatchReceipt(
                publications=tuple(publications),
                canonical_identity_sha256=canonical_sha256(
                    {"publications": [asdict(item) for item in publications]}
                ),
            ),
            initial_predecessor=(
                request.initial_source_predecessor_version_id
            ),
        )
        steps.append(CLOSURE_STEPS[3])
        suffix_phase_0_end = sampled

        for method_name, audit_kind in SUCCESSOR_SERVICE_OPERATIONS:
            operation = record_receipt(
                call_service(
                    method_name,
                    request=request,
                    source_batch=source_batch,
                    custody_nonce_sha256=custody_nonce_sha256,
                ),
                method_name.upper(),
                expected_audits=(audit_kind,),
                include_step=False,
            )
            readback_transaction(
                transaction_name=method_name,
                operation_identity_sha256=(
                    operation.operation_identity_sha256
                ),
            )
        fence_invocation_started = sampled
        fence_receipt = record_receipt(
            call_service(
                "invoke_fence_executor_once",
                request=request,
                source_batch=source_batch,
                custody_nonce_sha256=custody_nonce_sha256,
            ),
            "INVOKE_FENCE_EXECUTOR_ONCE",
            expected_audits=(
                "CHANGE_SET_CREATE",
                "CHANGE_SET_EXECUTE",
            ),
            include_step=False,
        )
        for transaction_name in (
            "authorize_change_set_create",
            "authorize_change_set_execute",
        ):
            readback_transaction(
                transaction_name=transaction_name,
                operation_identity_sha256=(
                    fence_receipt.operation_identity_sha256
                ),
            )
        internal_readbacks = (
            fence_receipt.policy_readback_monotonic_seconds
        )
        if internal_readbacks is None:
            raise ClosureRouteError(
                "fence executor omitted internal stabilization evidence"
            )
        first_policy_offset, second_policy_offset = (
            internal_readbacks
        )
        first_policy_readback = (
            fence_invocation_started + first_policy_offset
        )
        second_policy_readback = (
            fence_invocation_started + second_policy_offset
        )
        if sampled < second_policy_readback:
            sleeper(second_policy_readback - sampled)
            sample_clock("fence executor internal stabilization")
        if (
            first_policy_readback < fence_invocation_started
            or second_policy_readback > sampled
        ):
            raise ClosureRouteError(
                "fence executor internal timing evidence escaped invocation"
            )
        steps.append(CLOSURE_STEPS[4])
        suffix_phase_1_end = first_policy_readback
        suffix_phase_2_end = second_policy_readback

        record_receipt(
            call_service(
                "seal_decision_and_acquire_barrier",
                request=request,
                source_batch=source_batch,
            ),
            CLOSURE_STEPS[5],
        )
        pre_decision_attestation = _validate_attestation_result(
            call_service(
                "attest_sky_identity_pre_decision",
                request=request,
            )
        )
        steps.append(CLOSURE_STEPS[6])
        claim_receipt = record_receipt(
            call_service(
                "create_or_recover_claim",
                request=request,
                source_batch=source_batch,
                custody_nonce_sha256=custody_nonce_sha256,
            ),
            CLOSURE_STEPS[7],
            expected_audits=(AUTHORITY_AUDIT_KINDS[8],),
        )
        readback_transaction(
            transaction_name="create_or_recover_claim",
            operation_identity_sha256=(
                claim_receipt.operation_identity_sha256
            ),
        )
        suffix_phase_3_end = sampled

        remaining = call_service("remaining_milliseconds")
        if type(remaining) is not int or remaining <= 0:
            raise ClosureRouteError(
                "closure deadline expired before decision construction"
            )
        launch_model = _validate_launch_or_expire_decision(
            call_service(
                "model_current_launch_or_expire",
                request=request
            )
        )
        record_receipt(launch_model.receipt, CLOSURE_STEPS[8])
        if launch_model.branch == "EXPIRE":
            return _terminal_closure_result(
                request=request,
                model=launch_model,
                steps=tuple(steps),
            )

        remaining = call_service("remaining_milliseconds")
        if type(remaining) is not int or remaining <= 0:
            raise ClosureRouteError(
                "closure deadline expired before decision PUT"
            )
        direct_create_attempted = True
        direct_response = _validate_direct_response(
            call_service(
                "create_direct_decision",
                request=request,
                source_batch=source_batch,
                custody_nonce_sha256=custody_nonce_sha256,
            )
        )
        if (
            direct_response.custody_nonce_sha256
            != custody_nonce_sha256
            or direct_response.candidate_identity_sha256
            != request.candidate_identity_sha256
        ):
            raise ClosureRouteError(
                "direct decision response escaped this invocation's custody"
            )
        steps.extend(CLOSURE_STEPS[9:11])
        authority_audits.append(direct_response.audit)
        readback_transaction(
            transaction_name="create_direct_decision",
            operation_identity_sha256=(
                direct_response.canonical_identity_sha256
            ),
        )

        record_receipt(
            call_service(
                "paginate_and_exact_read_decision",
                request=request,
                direct_response=direct_response,
            ),
            CLOSURE_STEPS[11],
        )
        record_receipt(
            call_service(
                "validate_h1e_modeled_submit_once",
                request=request,
                direct_response=direct_response,
            ),
            CLOSURE_STEPS[12],
        )
        suffix_phase_4_end = sampled

        task8_live = _validate_h1d_live_result(
            call_service(
                "reinspect_task8_live_authority",
                request=request,
            ),
            activation_id=request.activation_id,
        )
        steps.append(CLOSURE_STEPS[13])
        repeated_attestation = _validate_attestation_result(
            call_service(
                "reattest_sky_identity",
                request=request,
            )
        )
        if (
            repeated_attestation.freshness_nonce
            == pre_decision_attestation.freshness_nonce
            or repeated_attestation.canonical_identity_sha256
            == pre_decision_attestation.canonical_identity_sha256
        ):
            raise ClosureRouteError("Sky attestation was replayed")
        steps.append(CLOSURE_STEPS[14])
        record_receipt(
            call_service(
                "recheck_all_authority",
                request=request,
                direct_response=direct_response,
                task8_live=task8_live,
                pre_decision_attestation=pre_decision_attestation,
                repeated_attestation=repeated_attestation,
            ),
            CLOSURE_STEPS[15],
        )
        suffix_phase_5_end = sampled

        owner_nonce, owner_nonce_sha256 = _private_nonce(
            nonce_source,
            "Sky arm owner",
        )
        arm_receipt = record_receipt(
            call_service(
                "arm_sky_action",
                request=request,
                owner_nonce=owner_nonce,
                owner_nonce_sha256=owner_nonce_sha256,
            ),
            CLOSURE_STEPS[16],
            expected_nonce=owner_nonce_sha256,
        )
        readback_transaction(
            transaction_name="arm_sky_action",
            operation_identity_sha256=(
                arm_receipt.operation_identity_sha256
            ),
        )

        remaining = call_service("remaining_milliseconds")
        if (
            type(remaining) is not int
            or remaining
            < MIN_REMAINING_BEFORE_TOKEN_SECONDS * 1000
        ):
            raise StoredDecisionNoPostError(
                "fewer than 120 seconds remain; token was not consumed"
            )
        decision_nonce, decision_nonce_sha256 = _private_nonce(
            nonce_source,
            "decision invocation",
        )
        consume_receipt = record_receipt(
            call_service(
                "consume_admission_reservation",
                request=request,
                owner_nonce=owner_nonce,
                owner_nonce_sha256=owner_nonce_sha256,
                decision_nonce=decision_nonce,
                decision_nonce_sha256=decision_nonce_sha256,
            ),
            CLOSURE_STEPS[17],
            expected_nonce=decision_nonce_sha256,
        )
        action_consumed = True
        readback_transaction(
            transaction_name="consume_admission_reservation",
            operation_identity_sha256=(
                consume_receipt.operation_identity_sha256
            ),
        )
        record_receipt(
            call_service(
                "coherent_readback_after_transactions",
                request=request,
                owner_nonce_sha256=owner_nonce_sha256,
                decision_nonce_sha256=decision_nonce_sha256,
            ),
            CLOSURE_STEPS[18],
            coherent=True,
        )

        admission = _validate_admission_result(
            call_service(
                "invoke_launch_admission_version_once",
                request=request,
                version_arn=request.admission_version_arn,
                decision_nonce=decision_nonce,
                decision_nonce_sha256=decision_nonce_sha256,
                direct_response=direct_response,
                task8_live=task8_live,
                repeated_attestation=repeated_attestation,
            ),
            request.admission_version_arn,
            decision_nonce_sha256,
        )
        steps.extend(CLOSURE_STEPS[19:24])
        authority_audits.append(admission.post_audit)
        readback_transaction(
            transaction_name="invoke_launch_admission_version_once",
            operation_identity_sha256=(
                admission.canonical_identity_sha256
            ),
        )
        suffix_phase_6_end = sampled
        suffix_phase_7_end = sampled
        closure_phase_4_end = sampled

        record_receipt(
            call_service(
                "audit_handoff_authority",
                request=request,
                admission=admission,
            ),
            CLOSURE_STEPS[24],
            expected_audits=(HANDOFF_AUDIT_KIND,),
        )
        persist_receipt = record_receipt(
            call_service(
                "persist_correlation_and_handoff",
                request=request,
                admission=admission,
                handoff_audit=handoff_audit,
            ),
            CLOSURE_STEPS[25],
        )
        readback_transaction(
            transaction_name="persist_correlation_and_handoff",
            operation_identity_sha256=(
                persist_receipt.operation_identity_sha256
            ),
        )
        record_receipt(
            call_service(
                "release_to_numeric_binding",
                request=request,
                version_arn=request.numeric_binding_version_arn,
                admission=admission,
            ),
            CLOSURE_STEPS[26],
        )
        closure_phase_5_end = sampled
        closure_phase_6_end = sampled
        closure_phase_7_end = sampled

        assert_no_stored_decision_submit_surface(services)
        steps.append(CLOSURE_STEPS[27])
        if tuple(steps) != CLOSURE_STEPS:
            raise ClosureRouteError("private closure step order drifted")
        if tuple(audit.kind for audit in authority_audits) != (
            AUTHORITY_AUDIT_KINDS
        ):
            raise ClosureRouteError("eleven-audit assignment drifted")
        audit_ids = tuple(
            audit.audit_identity_sha256 for audit in authority_audits
        )
        invocation_ids = tuple(
            audit.invocation_identity_sha256 for audit in authority_audits
        )
        if (
            len(set(audit_ids)) != 11
            or len(set(invocation_ids)) != 11
            or handoff_audit is None
            or handoff_audit.audit_identity_sha256 in audit_ids
            or handoff_audit.invocation_identity_sha256 in invocation_ids
        ):
            raise ClosureRouteError(
                "authority audit replay or cross-use was detected"
            )

        closure_spans = tuple(
            PhaseSpan(name, start, end)
            for name, start, end in (
                (
                    tuple(CLOSURE_PHASE_CEILINGS)[0],
                    closure_started,
                    closure_phase_0_end,
                ),
                (
                    tuple(CLOSURE_PHASE_CEILINGS)[1],
                    closure_phase_0_end,
                    closure_phase_1_end,
                ),
                (
                    tuple(CLOSURE_PHASE_CEILINGS)[2],
                    closure_phase_1_end,
                    closure_phase_2_end,
                ),
                (
                    tuple(CLOSURE_PHASE_CEILINGS)[3],
                    closure_phase_2_end,
                    closure_phase_3_end,
                ),
                (
                    tuple(CLOSURE_PHASE_CEILINGS)[4],
                    suffix_started,
                    closure_phase_4_end,
                ),
                (
                    tuple(CLOSURE_PHASE_CEILINGS)[5],
                    closure_phase_4_end,
                    closure_phase_5_end,
                ),
                (
                    tuple(CLOSURE_PHASE_CEILINGS)[6],
                    closure_phase_5_end,
                    closure_phase_6_end,
                ),
                (
                    tuple(CLOSURE_PHASE_CEILINGS)[7],
                    closure_phase_6_end,
                    closure_phase_7_end,
                ),
            )
        )
        suffix_spans = tuple(
            PhaseSpan(name, start, end)
            for name, start, end in (
                (
                    tuple(SUFFIX_PHASE_CEILINGS)[0],
                    suffix_started,
                    suffix_phase_0_end,
                ),
                (
                    tuple(SUFFIX_PHASE_CEILINGS)[1],
                    suffix_phase_0_end,
                    suffix_phase_1_end,
                ),
                (
                    tuple(SUFFIX_PHASE_CEILINGS)[2],
                    suffix_phase_1_end,
                    suffix_phase_2_end,
                ),
                (
                    tuple(SUFFIX_PHASE_CEILINGS)[3],
                    suffix_phase_2_end,
                    suffix_phase_3_end,
                ),
                (
                    tuple(SUFFIX_PHASE_CEILINGS)[4],
                    suffix_phase_3_end,
                    suffix_phase_4_end,
                ),
                (
                    tuple(SUFFIX_PHASE_CEILINGS)[5],
                    suffix_phase_4_end,
                    suffix_phase_5_end,
                ),
                (
                    tuple(SUFFIX_PHASE_CEILINGS)[6],
                    suffix_phase_5_end,
                    suffix_phase_6_end,
                ),
                (
                    tuple(SUFFIX_PHASE_CEILINGS)[7],
                    suffix_phase_6_end,
                    suffix_phase_7_end,
                ),
            )
        )
        try:
            timing_authority = validate_timing_ledgers(
                closure_spans=closure_spans,
                suffix_spans=suffix_spans,
                first_policy_readback_monotonic_seconds=(
                    first_policy_readback
                ),
                second_policy_readback_monotonic_seconds=(
                    second_policy_readback
                ),
            )
        except ValueError as exc:
            raise ClosureRouteError(str(exc)) from exc
        provisional_result = DecisionClosureResult(
            status="NUMERIC_BINDING_RECONCILIATION",
            steps=tuple(steps),
            authority_audit_kinds=tuple(
                audit.kind for audit in authority_audits
            ),
            handoff_audit_kind=handoff_audit.kind,
            classification=admission.classification,
            request_identity_sha256=request.canonical_identity_sha256,
            decision_response_identity_sha256=(
                direct_response.canonical_identity_sha256
            ),
            admission_identity_sha256=admission.canonical_identity_sha256,
            closure_spans=closure_spans,
            suffix_spans=suffix_spans,
            timing_authority=timing_authority,
            canonical_identity_sha256="",
        )
        return DecisionClosureResult(
            **{
                **asdict(provisional_result),
                "steps": provisional_result.steps,
                "authority_audit_kinds": (
                    provisional_result.authority_audit_kinds
                ),
                "closure_spans": provisional_result.closure_spans,
                "suffix_spans": provisional_result.suffix_spans,
                "timing_authority": provisional_result.timing_authority,
                "canonical_identity_sha256": canonical_sha256(
                    _identity_body(provisional_result)
                ),
            }
        )
    except StoredDecisionNoPostError:
        raise
    except ClosureRouteError as exc:
        if direct_create_attempted and not action_consumed:
            raise StoredDecisionNoPostError(
                "direct decision is permanently reconcile-only"
            ) from exc
        raise
    except Exception as exc:
        if direct_create_attempted and not action_consumed:
            raise StoredDecisionNoPostError(
                "direct decision is permanently reconcile-only"
            ) from exc
        raise ClosureRouteError(
            "private closure failed; no retry edge exists"
        ) from exc


__all__ = [
    "AUTHORITY_AUDIT_KINDS",
    "AdmissionClosureResult",
    "CLOSURE_DEADLINE_SECONDS",
    "CLOSURE_PHASE_CEILINGS",
    "CLOSURE_STEPS",
    "ClosureRequest",
    "ClosureRouteError",
    "DecisionClosureResult",
    "DeployedGateDocument",
    "DirectDecisionResponse",
    "LaunchOrExpireDecision",
    "HANDOFF_AUDIT_KIND",
    "LAMBDA_CONFIGURED_TIMEOUT_SECONDS",
    "MIN_REMAINING_BEFORE_TOKEN_SECONDS",
    "PhaseSpan",
    "REHEARSAL_PATH_PROOFS",
    "RehearsalGate",
    "RehearsalMeasurement",
    "SOURCE_KINDS",
    "STEP_FUNCTION_TASK_TIMEOUT_SECONDS",
    "StoredDecisionNoPostError",
    "TerminalDecisionClosureResult",
    "SUFFIX_PHASE_CEILINGS",
    "TimingAuthority",
    "assert_no_stored_decision_submit_surface",
    "build_admission_closure_result",
    "build_audit_evidence",
    "build_closure_request",
    "build_direct_decision_response",
    "build_launch_or_expire_decision",
    "build_deployed_gate_document",
    "build_rehearsal_measurement",
    "build_source_batch_receipt",
    "build_step_receipt",
    "build_task11_workflow_definition",
    "execute_decision_closure",
    "parse_deployed_gate_document",
    "rehearsal_measurement_from_mapping",
    "transition_abandoned_arm",
    "validate_closure_request",
    "validate_timing_ledgers",
    "verify_no_post_rehearsals",
]
