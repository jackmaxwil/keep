"""Exact Task 12 worker-drain authority and completion contracts.

Systems Manager accepting a command proves only that AWS accepted one exact
request.  Drain completion additionally requires the Task 10 graceful-stop
observation and Task 9 liability settlement for the same retained candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import re
from typing import Mapping

from .canonical import canonical_sha256
from .spend_authority import (
    AllocationInterval,
    GpuLiabilityReserveResult,
    LiabilitySettlementEvidence,
    SpendAuthorityError,
    canonical_decimal_json_bytes,
    validate_liability_settlement,
)
from .task10_worker import (
    ACCOUNT_ID,
    GRACEFUL_SCRIPT_NAMES,
    GRACEFUL_STOP_DOCUMENT,
    REGION,
    RUN_ID,
    UNIT_NAMES,
    WORKER_DRAIN_ROLE_ARN,
    Task10WorkerError,
    Task12WorkerDrainAuthority as RetainedTask10WorkerDrainAuthority,
    WorkerBootstrapDescriptor,
    WorkerInstanceObservation,
    build_task12_worker_drain_authority as build_retained_task10_authority,
    build_worker_runtime_authority,
    validate_graceful_stop_evidence,
)
from .task12_writers import (
    RetainedWriterActionAuthority,
    RetainedWriterAuditAuthority,
)


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_INSTANCE = re.compile(r"i-(?:[0-9a-f]{8}|[0-9a-f]{17})\Z")
_COMMAND = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_DOCUMENT_VERSION = re.compile(r"[1-9][0-9]*\Z")
_RESERVE_DISPOSITIONS = frozenset(
    {
        "CREATED",
        "EXACT_DUPLICATE",
        "AMBIGUOUS_EXACT_READBACK",
    }
)
_RESERVE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "generation",
        "generation_text",
        "allocation_ordinal",
        "allocation_ordinal_text",
        "ec2_client_token",
        "request_identity_sha256",
        "ledger_predecessor_identity_sha256",
        "gpu_reserve_seconds",
        "gpu_reserve_cost_usd",
        "root_volume_gib",
        "root_volume_tail_usd_max",
        "residual_liability_approval_identity_sha256",
        "epoch",
        "revision",
        "nonce_owner_identity_sha256",
        "state",
        "observed_at",
        "reserve_key",
        "canonical_body_sha256",
    }
)


class Task12WorkerDrainError(ValueError):
    """Worker-drain authority, dispatch, or completion evidence drifted."""


@dataclass(frozen=True)
class WorkerDrainCandidate:
    allocation: AllocationInterval
    liability_reserve: GpuLiabilityReserveResult
    worker_descriptor: WorkerBootstrapDescriptor
    worker_observation: WorkerInstanceObservation
    expected_unit_hashes: Mapping[str, str]
    expected_script_hashes: Mapping[str, str]
    document_name: str
    document_version: str
    worker_drain_role_arn: str
    allocation_identity_sha256: str
    liability_reserve_identity_sha256: str
    runtime_authority_sha256: str
    candidate_identity_sha256: str


@dataclass(frozen=True)
class Task12WorkerDrainAuthority:
    candidate: WorkerDrainCandidate
    action: RetainedWriterActionAuthority
    audit: RetainedWriterAuditAuthority
    worker_drain_role_arn: str
    authority_identity_sha256: str


@dataclass(frozen=True)
class WorkerDrainServices:
    ssm: object


@dataclass(frozen=True)
class WorkerDrainDispatch:
    authority_identity_sha256: str
    candidate_identity_sha256: str
    instance_id: str
    command_id: str
    request_id: str
    response_identity_sha256: str
    retained_authority: RetainedTask10WorkerDrainAuthority
    retained_authority_body_sha256: str
    state: str
    drain_complete: bool
    liability_settled: bool
    dispatch_identity_sha256: str


@dataclass(frozen=True)
class WorkerDrainCompletion:
    authority_identity_sha256: str
    dispatch_identity_sha256: str
    command_id: str
    graceful_stop_identity_sha256: str
    liability_settlement_identity_sha256: str
    state: str
    drain_complete: bool
    liability_settled: bool
    completion_identity_sha256: str


def _fail(message: str) -> None:
    raise Task12WorkerDrainError(message)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail(label + " must be exact lowercase SHA-256")
    return value


def _utc(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(label + " must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError) as exc:
        raise Task12WorkerDrainError(
            label + " must be canonical UTC"
        ) from exc
    if (
        parsed.tzinfo is None
        or parsed.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
        != value
    ):
        _fail(label + " must be canonical UTC")
    return value


def _allocation_identity(value: AllocationInterval) -> str:
    if type(value) is not AllocationInterval:
        _fail("Task 9 allocation must be typed")
    if (
        type(value.instance_id) is not str
        or _INSTANCE.fullmatch(value.instance_id) is None
        or type(value.job_id) is not str
        or not value.job_id
        or value.state != "OPEN"
        or value.ended_at is not None
        or type(value.charged_seconds) is not int
        or value.charged_seconds < 0
        or not isinstance(value.charged_cost_usd, Decimal)
        or not value.charged_cost_usd.is_finite()
        or value.charged_cost_usd < 0
    ):
        _fail("Task 9 allocation is not an exact open allocation")
    _utc(value.started_at, "allocation started_at")
    body = {
        **asdict(value),
        "charged_cost_usd": format(value.charged_cost_usd, "f"),
    }
    return canonical_sha256(body)


def _validate_reserve(
    value: GpuLiabilityReserveResult,
    *,
    descriptor: WorkerBootstrapDescriptor,
    observation: WorkerInstanceObservation,
) -> str:
    if type(value) is not GpuLiabilityReserveResult:
        _fail("Task 9 liability reserve must be typed")
    if (
        value.disposition not in _RESERVE_DISPOSITIONS
        or value.gpu_reserve_seconds != 900
        or value.gpu_reserve_cost_usd != Decimal("13.76")
        or value.root_volume_gib != 300
        or value.root_volume_tail_usd_max != Decimal("0.01")
        or type(value.remaining_gpu_seconds) is not int
        or value.remaining_gpu_seconds < 0
        or not isinstance(value.remaining_gpu_cost_usd, Decimal)
        or not value.remaining_gpu_cost_usd.is_finite()
        or value.remaining_gpu_cost_usd < 0
        or type(value.record) is not dict
        or set(value.record) != _RESERVE_FIELDS
    ):
        _fail("Task 9 liability reserve is not exact")
    record = dict(value.record)
    identity = record.pop("canonical_body_sha256")
    _sha(identity, "liability reserve body")
    if (
        identity
        != hashlib.sha256(canonical_decimal_json_bytes(record)).hexdigest()
        or identity != value.reserve_identity_sha256
        or value.reserve_key != record.get("reserve_key")
        or record.get("schema_version") != 1
        or record.get("record_type")
        != "glm52_gpu_liability_reserve_v1"
        or record.get("account_id") != ACCOUNT_ID
        or record.get("region") != REGION
        or record.get("run_id") != RUN_ID
        or record.get("activation_id") != descriptor.activation_id
        or record.get("generation") != descriptor.generation
        or record.get("generation_text") != descriptor.generation_text
        or record.get("allocation_ordinal")
        != observation.allocation_ordinal
        or record.get("allocation_ordinal_text")
        != observation.allocation_ordinal_text
        or record.get("gpu_reserve_seconds") != value.gpu_reserve_seconds
        or record.get("gpu_reserve_cost_usd")
        != value.gpu_reserve_cost_usd
        or record.get("root_volume_gib") != value.root_volume_gib
        or record.get("root_volume_tail_usd_max")
        != value.root_volume_tail_usd_max
        or record.get("state") != "HELD"
        or type(record.get("epoch")) is not int
        or record["epoch"] < 1
        or type(record.get("revision")) is not int
        or record["revision"] < 0
    ):
        _fail("Task 9 liability reserve provenance drifted")
    expected_key = (
        "GPU_LIABILITY_RESERVE#"
        + descriptor.activation_id
        + "#"
        + descriptor.generation_text
        + "#"
        + observation.allocation_ordinal_text
    )
    if value.reserve_key != expected_key:
        _fail("Task 9 liability reserve coordinate drifted")
    for field in (
        "ec2_client_token",
        "request_identity_sha256",
        "ledger_predecessor_identity_sha256",
        "residual_liability_approval_identity_sha256",
        "nonce_owner_identity_sha256",
    ):
        _sha(record.get(field), "liability reserve " + field)
    _utc(record.get("observed_at"), "liability reserve observed_at")
    return identity


def _validate_hashes(
    value: object,
    *,
    names: tuple[str, ...],
    label: str,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(names):
        _fail(label + " hash set is not closed")
    result = dict(value)
    for name in names:
        _sha(result.get(name), label + " " + name)
    return result


def prepare_worker_drain_candidate(
    *,
    allocation: AllocationInterval,
    liability_reserve: GpuLiabilityReserveResult,
    worker_descriptor: WorkerBootstrapDescriptor,
    worker_observation: WorkerInstanceObservation,
    expected_unit_hashes: Mapping[str, str],
    expected_script_hashes: Mapping[str, str],
    document_version: str,
) -> WorkerDrainCandidate:
    """Cross-bind the live Task 9 and Task 10 facts before any SSM call."""

    try:
        runtime = build_worker_runtime_authority(
            worker_descriptor,
            worker_observation,
        )
    except Task10WorkerError as exc:
        raise Task12WorkerDrainError(
            "Task 10 worker runtime authority failed"
        ) from exc
    allocation_identity = _allocation_identity(allocation)
    if (
        allocation.instance_id != worker_observation.instance_id
        or allocation_identity
        != worker_descriptor.gpu_allocation_sha256
    ):
        _fail("Task 9 allocation does not bind the Task 10 worker")
    reserve_identity = _validate_reserve(
        liability_reserve,
        descriptor=worker_descriptor,
        observation=worker_observation,
    )
    unit_hashes = _validate_hashes(
        expected_unit_hashes,
        names=UNIT_NAMES,
        label="systemd unit",
    )
    script_hashes = _validate_hashes(
        expected_script_hashes,
        names=GRACEFUL_SCRIPT_NAMES,
        label="graceful-stop script",
    )
    if (
        type(document_version) is not str
        or _DOCUMENT_VERSION.fullmatch(document_version) is None
    ):
        _fail("SSM document version must be one exact positive integer")
    identity_body = {
        "allocation_identity_sha256": allocation_identity,
        "liability_reserve_identity_sha256": reserve_identity,
        "worker_descriptor_body_sha256": (
            worker_descriptor.descriptor_body_sha256
        ),
        "worker_observation_body_sha256": (
            worker_observation.observation_body_sha256
        ),
        "runtime_authority_sha256": runtime.runtime_authority_sha256,
        "instance_id": worker_observation.instance_id,
        "document_name": GRACEFUL_STOP_DOCUMENT,
        "document_version": document_version,
        "worker_drain_role_arn": WORKER_DRAIN_ROLE_ARN,
        "expected_unit_hashes": unit_hashes,
        "expected_script_hashes": script_hashes,
    }
    return WorkerDrainCandidate(
        allocation=allocation,
        liability_reserve=liability_reserve,
        worker_descriptor=worker_descriptor,
        worker_observation=worker_observation,
        expected_unit_hashes=unit_hashes,
        expected_script_hashes=script_hashes,
        document_name=GRACEFUL_STOP_DOCUMENT,
        document_version=document_version,
        worker_drain_role_arn=WORKER_DRAIN_ROLE_ARN,
        allocation_identity_sha256=allocation_identity,
        liability_reserve_identity_sha256=reserve_identity,
        runtime_authority_sha256=runtime.runtime_authority_sha256,
        candidate_identity_sha256=canonical_sha256(identity_body),
    )


def _validate_candidate(value: object) -> WorkerDrainCandidate:
    if type(value) is not WorkerDrainCandidate:
        _fail("worker-drain candidate must be typed")
    rebuilt = prepare_worker_drain_candidate(
        allocation=value.allocation,
        liability_reserve=value.liability_reserve,
        worker_descriptor=value.worker_descriptor,
        worker_observation=value.worker_observation,
        expected_unit_hashes=dict(value.expected_unit_hashes),
        expected_script_hashes=dict(value.expected_script_hashes),
        document_version=value.document_version,
    )
    if value != rebuilt:
        _fail("worker-drain candidate identity drifted")
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


def _validate_action_audit(
    *,
    candidate: WorkerDrainCandidate,
    action: object,
    audit: object,
) -> tuple[RetainedWriterActionAuthority, RetainedWriterAuditAuthority]:
    if (
        type(action) is not RetainedWriterActionAuthority
        or type(audit) is not RetainedWriterAuditAuthority
    ):
        _fail("worker-drain action and audit must be typed")
    expected_key = (
        "ACTIVATION#"
        + candidate.worker_descriptor.activation_id
        + "#RECOVERY_ACTION#WORKER_DRAIN_SIGNAL#"
        + candidate.worker_descriptor.generation_text
    )
    if (
        action.authority_domain != "RECOVERY"
        or action.action_kind != "WORKER_DRAIN_SIGNAL"
        or action.action_key != expected_key
        or action.candidate_identity_sha256
        != candidate.candidate_identity_sha256
        or action.state != "CONSUMED"
        or type(action.authorized_revision) is not int
        or action.authorized_revision < 1
    ):
        _fail("worker-drain action is generic, foreign, or unconsumed")
    _sha(action.action_identity_sha256, "worker-drain action identity")
    _sha(
        action.owner_invocation_nonce_sha256,
        "worker-drain invocation nonce",
    )
    if (
        audit.authority_domain != action.authority_domain
        or audit.action_kind != action.action_kind
        or audit.action_key != action.action_key
        or audit.candidate_identity_sha256
        != candidate.candidate_identity_sha256
        or audit.action_identity_sha256 != action.action_identity_sha256
        or audit.audit_kind != "H1F_GENESIS_TO_ZERO_CHILD"
        or type(audit.closing_revision) is not int
        or audit.authorized_revision != audit.closing_revision + 1
        or audit.authorized_revision != action.authorized_revision
        or audit.current_revision != audit.authorized_revision
        or audit.canonical_identity_sha256
        != canonical_sha256(_audit_body(audit))
    ):
        _fail("worker-drain audit is stale, relabelled, or foreign")
    _utc(audit.observed_at, "worker-drain audit observed_at")
    return action, audit


def build_task12_worker_drain_authority(
    *,
    candidate: WorkerDrainCandidate,
    action: object,
    audit: object,
) -> Task12WorkerDrainAuthority:
    """Bind the exact drain candidate to one consumed Task 12 action."""

    candidate = _validate_candidate(candidate)
    exact_action, exact_audit = _validate_action_audit(
        candidate=candidate,
        action=action,
        audit=audit,
    )
    body = {
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "action_identity_sha256": exact_action.action_identity_sha256,
        "audit_identity_sha256": exact_audit.canonical_identity_sha256,
    }
    return Task12WorkerDrainAuthority(
        candidate=candidate,
        action=exact_action,
        audit=exact_audit,
        worker_drain_role_arn=WORKER_DRAIN_ROLE_ARN,
        authority_identity_sha256=canonical_sha256(body),
    )


def _validate_authority(value: object) -> Task12WorkerDrainAuthority:
    if type(value) is not Task12WorkerDrainAuthority:
        _fail("Task 12 worker-drain authority must be typed")
    rebuilt = build_task12_worker_drain_authority(
        candidate=value.candidate,
        action=value.action,
        audit=value.audit,
    )
    if value != rebuilt:
        _fail("Task 12 worker-drain authority identity drifted")
    return value


def _retained_task10_authority(
    *,
    authority: Task12WorkerDrainAuthority,
    command_id: str,
) -> RetainedTask10WorkerDrainAuthority:
    candidate = authority.candidate
    descriptor = candidate.worker_descriptor
    observation = candidate.worker_observation
    try:
        return build_retained_task10_authority(
            worker_descriptor=descriptor,
            schema_version=1,
            record_type="glm52_task12_retained_worker_drain_authority_v1",
            account_id=ACCOUNT_ID,
            region=REGION,
            run_id=RUN_ID,
            campaign_identity_sha256=(
                descriptor.campaign_identity_sha256
            ),
            activation_id=descriptor.activation_id,
            activation_ordinal=descriptor.activation_ordinal,
            generation=descriptor.generation,
            generation_text=descriptor.generation_text,
            allocation_ordinal=observation.allocation_ordinal,
            allocation_ordinal_text=observation.allocation_ordinal_text,
            instance_id=observation.instance_id,
            command_id=command_id,
            document_name=candidate.document_name,
            document_version=candidate.document_version,
            worker_drain_role_arn=authority.worker_drain_role_arn,
            worker_descriptor_body_sha256=(
                descriptor.descriptor_body_sha256
            ),
            worker_observation=asdict(observation),
            expected_unit_hashes=dict(candidate.expected_unit_hashes),
            expected_script_hashes=dict(
                candidate.expected_script_hashes
            ),
            task9_launch_identity_sha256=(
                descriptor.task9_launch_identity_sha256
            ),
            task9_custody_identity_sha256=(
                descriptor.task9_custody_identity_sha256
            ),
            task12_action_identity_sha256=(
                authority.action.action_identity_sha256
            ),
            task12_audit_identity_sha256=(
                authority.audit.canonical_identity_sha256
            ),
        )
    except Task10WorkerError as exc:
        raise Task12WorkerDrainError(
            "retained Task 10 worker-drain authority failed"
        ) from exc


def dispatch_worker_drain(
    *,
    services: WorkerDrainServices,
    authority: Task12WorkerDrainAuthority,
) -> WorkerDrainDispatch:
    """Send one parameterless SSM command without claiming completion."""

    if type(services) is not WorkerDrainServices:
        _fail("worker-drain services must be typed")
    authority = _validate_authority(authority)
    send = getattr(services.ssm, "send_command", None)
    if not callable(send):
        _fail("SSM boundary lacks send_command")
    candidate = authority.candidate
    request = {
        "DocumentName": candidate.document_name,
        "DocumentVersion": candidate.document_version,
        "InstanceIds": [candidate.worker_observation.instance_id],
    }
    try:
        response = send(**request)
    except Exception as exc:
        raise Task12WorkerDrainError(
            "SSM command transport did not authenticate acceptance"
        ) from exc
    if type(response) is not dict:
        _fail("SSM response must be one authenticated object")
    metadata = response.get("ResponseMetadata")
    command = response.get("Command")
    if type(metadata) is not dict or type(command) is not dict:
        _fail("SSM response lacks authenticated metadata or command")
    request_id = metadata.get("RequestId")
    command_id = command.get("CommandId")
    instance_ids = command.get("InstanceIds")
    if (
        metadata.get("HTTPStatusCode") != 200
        or type(request_id) is not str
        or not request_id
        or type(metadata.get("RetryAttempts")) is not int
        or metadata.get("RetryAttempts") != 0
        or type(command_id) is not str
        or _COMMAND.fullmatch(command_id) is None
        or command.get("DocumentName") != candidate.document_name
        or command.get("DocumentVersion") != candidate.document_version
        or instance_ids != [candidate.worker_observation.instance_id]
        or command.get("Status")
        not in {"Pending", "InProgress", "Success"}
    ):
        _fail("SSM response did not authenticate the exact drain request")
    response_body = {
        "request_id": request_id,
        "command_id": command_id,
        "document_name": command["DocumentName"],
        "document_version": command["DocumentVersion"],
        "instance_ids": instance_ids,
        "status": command["Status"],
    }
    response_identity = canonical_sha256(response_body)
    retained = _retained_task10_authority(
        authority=authority,
        command_id=command_id,
    )
    body = {
        "authority_identity_sha256": authority.authority_identity_sha256,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "instance_id": candidate.worker_observation.instance_id,
        "command_id": command_id,
        "request_id": request_id,
        "response_identity_sha256": response_identity,
        "retained_authority_body_sha256": retained.authority_body_sha256,
        "state": "SSM_ACCEPTED_NOT_DRAINED",
        "drain_complete": False,
        "liability_settled": False,
    }
    return WorkerDrainDispatch(
        **body,
        retained_authority=retained,
        dispatch_identity_sha256=canonical_sha256(body),
    )


def _validate_dispatch(
    *,
    authority: Task12WorkerDrainAuthority,
    value: object,
) -> WorkerDrainDispatch:
    if type(value) is not WorkerDrainDispatch:
        _fail("worker-drain dispatch must be typed")
    candidate = authority.candidate
    if (
        value.authority_identity_sha256
        != authority.authority_identity_sha256
        or value.candidate_identity_sha256
        != candidate.candidate_identity_sha256
        or value.instance_id != candidate.worker_observation.instance_id
        or _COMMAND.fullmatch(value.command_id) is None
        or type(value.request_id) is not str
        or not value.request_id
        or value.state != "SSM_ACCEPTED_NOT_DRAINED"
        or value.drain_complete is not False
        or value.liability_settled is not False
    ):
        _fail("worker-drain dispatch identity drifted")
    _sha(value.response_identity_sha256, "SSM response identity")
    retained = _retained_task10_authority(
        authority=authority,
        command_id=value.command_id,
    )
    if value.retained_authority != retained:
        _fail("retained Task 10 worker-drain authority drifted")
    body = {
        "authority_identity_sha256": value.authority_identity_sha256,
        "candidate_identity_sha256": value.candidate_identity_sha256,
        "instance_id": value.instance_id,
        "command_id": value.command_id,
        "request_id": value.request_id,
        "response_identity_sha256": value.response_identity_sha256,
        "retained_authority_body_sha256": (
            value.retained_authority.authority_body_sha256
        ),
        "state": value.state,
        "drain_complete": value.drain_complete,
        "liability_settled": value.liability_settled,
    }
    if value.dispatch_identity_sha256 != canonical_sha256(body):
        _fail("worker-drain dispatch self-hash drifted")
    return value


def complete_worker_drain(
    *,
    authority: Task12WorkerDrainAuthority,
    dispatch: WorkerDrainDispatch,
    graceful_stop_observation: object,
    liability_settlement: object,
) -> WorkerDrainCompletion:
    """Promote to drained only with post-command stop and settlement proof."""

    authority = _validate_authority(authority)
    dispatch = _validate_dispatch(authority=authority, value=dispatch)
    candidate = authority.candidate
    try:
        graceful = validate_graceful_stop_evidence(
            graceful_stop_observation,
            expected_unit_hashes=candidate.expected_unit_hashes,
            expected_script_hashes=candidate.expected_script_hashes,
        )
    except Task10WorkerError as exc:
        raise Task12WorkerDrainError(
            "post-command graceful-stop observation failed"
        ) from exc
    if (
        graceful.get("authority") != "SSM"
        or graceful.get("ssm_command_id") != dispatch.command_id
    ):
        _fail("graceful-stop observation is not post-command evidence")
    if type(liability_settlement) is not LiabilitySettlementEvidence:
        _fail("liability settlement must be typed and present")
    try:
        settlement = validate_liability_settlement(
            candidate.liability_reserve,
            liability_settlement,
        )
    except SpendAuthorityError as exc:
        raise Task12WorkerDrainError(
            "Task 9 liability settlement failed"
        ) from exc
    if (
        settlement.liability_identity_sha256
        != candidate.liability_reserve_identity_sha256
    ):
        _fail("liability settlement is foreign to the drained worker")
    graceful_identity = str(graceful["graceful_stop_body_sha256"])
    settlement_identity = settlement.canonical_body_sha256
    _sha(graceful_identity, "graceful-stop identity")
    _sha(settlement_identity, "liability settlement identity")
    body = {
        "authority_identity_sha256": authority.authority_identity_sha256,
        "dispatch_identity_sha256": dispatch.dispatch_identity_sha256,
        "command_id": dispatch.command_id,
        "graceful_stop_identity_sha256": graceful_identity,
        "liability_settlement_identity_sha256": settlement_identity,
        "state": "DRAINED_AND_LIABILITY_SETTLED",
        "drain_complete": True,
        "liability_settled": True,
    }
    return WorkerDrainCompletion(
        **body,
        completion_identity_sha256=canonical_sha256(body),
    )


__all__ = [
    "Task12WorkerDrainAuthority",
    "Task12WorkerDrainError",
    "WorkerDrainCandidate",
    "WorkerDrainCompletion",
    "WorkerDrainDispatch",
    "WorkerDrainServices",
    "build_task12_worker_drain_authority",
    "complete_worker_drain",
    "dispatch_worker_drain",
    "prepare_worker_drain_candidate",
]
