"""Pure Task 12 runtime closure, cancellation, and Task 9 handoff contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Mapping, Optional, Tuple

from .canonical import canonical_sha256
from .launch_custody import LiabilitySettlementCoordinator
from .task10_worker import (
    Task10WorkerError,
    WorkerBootstrapDescriptor,
    validate_worker_bootstrap_descriptor,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE = re.compile(r"^i-[0-9a-f]{17}$")
_ACTIVATION = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?$")
_READ_DOMAINS = (
    "REQUEST",
    "JOB",
    "CONTROLLER",
    "WORKER",
    "ALLOCATION",
)
_REQUEST_TERMINAL = frozenset({"FAILED", "CANCELLED"})
_JOB_TERMINAL = frozenset({"FAILED", "CANCELLED", "SUCCEEDED"})
_WORKER_TERMINAL = frozenset({"shutting-down", "terminated"})
_ALLOCATION_TERMINAL = frozenset({"CLOSED"})
_DEADLINE_PHASE_RANK = {
    "BEFORE_T_MINUS_60": 0,
    "T_MINUS_60": 1,
    "T_MINUS_50": 2,
    "T_MINUS_30": 3,
}
_DEADLINE_WAKE_REASONS = frozenset(
    {
        "SCHEDULE_ACCELERATOR",
        "STATE_CHANGE_ACCELERATOR",
        "RECOVERY_REPLAY",
    }
)


class Task12RuntimeError(ValueError):
    """Runtime evidence failed the closed Task 12 contract."""


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise Task12RuntimeError(label + " must be a lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise Task12RuntimeError(label + " must be a nonempty exact string")
    return value


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise Task12RuntimeError(label + " must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Task12RuntimeError(label + " must be canonical UTC") from exc
    return parsed.replace(tzinfo=timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _identity(value: object) -> str:
    return canonical_sha256(asdict(value))


@dataclass(frozen=True)
class RuntimeEntity:
    identity: str
    state: str
    observed_at: str
    evidence_identity_sha256: str

    def __post_init__(self) -> None:
        _text(self.identity, "runtime entity identity")
        _text(self.state, "runtime entity state")
        _utc(self.observed_at, "runtime entity observed_at")
        _sha(self.evidence_identity_sha256, "runtime entity evidence")


@dataclass(frozen=True)
class CompleteRuntimeRead:
    domain: str
    members: Tuple[RuntimeEntity, ...]
    observed_at: str
    pagination_complete: bool
    evidence_identity_sha256: str


@dataclass(frozen=True)
class SpendRead:
    state: str
    ledger_head_identity_sha256: str
    remaining_gpu_seconds: int
    remaining_gpu_cost_usd: str
    observed_at: str
    evidence_identity_sha256: str


@dataclass(frozen=True)
class RuntimeScan:
    requests: CompleteRuntimeRead
    jobs: CompleteRuntimeRead
    controller: CompleteRuntimeRead
    workers: CompleteRuntimeRead
    allocations: CompleteRuntimeRead
    spend: SpendRead

    @property
    def observed_at(self) -> str:
        return self.requests.observed_at


@dataclass(frozen=True)
class RuntimeReadBoundaries:
    request_reader: object
    job_reader: object
    controller_reader: object
    worker_reader: object
    allocation_reader: object
    spend_reader: object


def _validate_read(
    value: object,
    *,
    domain: str,
) -> CompleteRuntimeRead:
    if not isinstance(value, CompleteRuntimeRead):
        raise Task12RuntimeError(domain + " read must be typed")
    if (
        value.domain != domain
        or value.pagination_complete is not True
        or type(value.members) is not tuple
    ):
        raise Task12RuntimeError(domain + " read must be complete and exact")
    _utc(value.observed_at, domain + " read observed_at")
    _sha(value.evidence_identity_sha256, domain + " read evidence")
    identities = []
    for member in value.members:
        if not isinstance(member, RuntimeEntity):
            raise Task12RuntimeError(domain + " member must be typed")
        if member.observed_at != value.observed_at:
            raise Task12RuntimeError(domain + " member observation drifted")
        identities.append(member.identity)
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise Task12RuntimeError(domain + " members must be sorted and unique")
    return value


def _validate_spend(value: object) -> SpendRead:
    if not isinstance(value, SpendRead):
        raise Task12RuntimeError("spend read must be typed")
    if (
        value.state not in {"OPEN", "CLOSED"}
        or type(value.remaining_gpu_seconds) is not int
        or value.remaining_gpu_seconds < 0
        or type(value.remaining_gpu_cost_usd) is not str
        or re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{2}", value.remaining_gpu_cost_usd)
        is None
    ):
        raise Task12RuntimeError("spend read is not closed")
    _utc(value.observed_at, "spend read observed_at")
    _sha(value.ledger_head_identity_sha256, "spend ledger head")
    _sha(value.evidence_identity_sha256, "spend read evidence")
    return value


def _validate_scan(value: object) -> RuntimeScan:
    if not isinstance(value, RuntimeScan):
        raise Task12RuntimeError("runtime scan must be typed")
    reads = (
        _validate_read(value.requests, domain="REQUEST"),
        _validate_read(value.jobs, domain="JOB"),
        _validate_read(value.controller, domain="CONTROLLER"),
        _validate_read(value.workers, domain="WORKER"),
        _validate_read(value.allocations, domain="ALLOCATION"),
    )
    spend = _validate_spend(value.spend)
    observed = {item.observed_at for item in reads} | {spend.observed_at}
    if len(observed) != 1:
        raise Task12RuntimeError("runtime scan observation window drifted")
    return value


def collect_runtime_scan(
    *,
    boundaries: RuntimeReadBoundaries,
    correlation_identity_sha256: str,
) -> RuntimeScan:
    """Read all six immutable runtime domains once through closed boundaries."""

    if not isinstance(boundaries, RuntimeReadBoundaries):
        raise Task12RuntimeError("runtime read boundaries must be typed")
    correlation = _sha(
        correlation_identity_sha256,
        "runtime correlation identity",
    )
    calls = (
        (boundaries.request_reader, "read_requests", "REQUEST"),
        (boundaries.job_reader, "read_jobs", "JOB"),
        (boundaries.controller_reader, "read_controller", "CONTROLLER"),
        (boundaries.worker_reader, "read_workers", "WORKER"),
        (boundaries.allocation_reader, "read_allocations", "ALLOCATION"),
    )
    results = []
    for boundary, method_name, domain in calls:
        method = getattr(boundary, method_name, None)
        if not callable(method):
            raise Task12RuntimeError(domain + " read boundary is absent")
        results.append(_validate_read(method(correlation), domain=domain))
    spend_method = getattr(boundaries.spend_reader, "read_spend", None)
    if not callable(spend_method):
        raise Task12RuntimeError("spend read boundary is absent")
    spend = _validate_spend(spend_method(correlation))
    return _validate_scan(
        RuntimeScan(
            requests=results[0],
            jobs=results[1],
            controller=results[2],
            workers=results[3],
            allocations=results[4],
            spend=spend,
        )
    )


@dataclass(frozen=True)
class CancellationCommand:
    kind: str
    target_id: str
    action_key: str
    invocation_nonce_sha256: str
    correlation_identity_sha256: str
    method: str
    path: str
    body: Mapping[str, object]
    total_max_attempts: int
    command_identity_sha256: str


@dataclass(frozen=True)
class CancellationObservation:
    classification: str
    send_count: int
    response_request_id: Optional[str]
    response_identity_sha256: Optional[str]
    observed_target_state: str
    observation_identity_sha256: str


@dataclass(frozen=True)
class CancellationReconciliation:
    command_identity_sha256: str
    state: str
    target_terminal: bool
    may_resend: bool
    observation_identity_sha256: str
    reconciliation_identity_sha256: str


def build_cancellation_command(
    *,
    kind: str,
    target_id: str,
    action_key: str,
    invocation_nonce_sha256: str,
    correlation_identity_sha256: str,
) -> CancellationCommand:
    if kind not in {"REQUEST_CANCEL", "JOB_CANCEL"}:
        raise Task12RuntimeError("cancellation kind is not closed")
    _text(target_id, "cancellation target")
    if kind == "JOB_CANCEL" and (
        re.fullmatch(r"[1-9][0-9]*", target_id) is None
    ):
        raise Task12RuntimeError("job cancellation target must be positive numeric")
    if type(action_key) is not str or kind not in action_key:
        raise Task12RuntimeError("cancellation action key is foreign")
    nonce = _sha(invocation_nonce_sha256, "cancellation invocation nonce")
    correlation = _sha(
        correlation_identity_sha256,
        "cancellation correlation identity",
    )
    path, body = (
        ("/api/cancel", {"request_id": target_id})
        if kind == "REQUEST_CANCEL"
        else ("/jobs/cancel", {"job_id": target_id})
    )
    identity = canonical_sha256(
        {
            "kind": kind,
            "target_id": target_id,
            "action_key": action_key,
            "invocation_nonce_sha256": nonce,
            "correlation_identity_sha256": correlation,
            "method": "POST",
            "path": path,
            "body": body,
            "total_max_attempts": 1,
        }
    )
    return CancellationCommand(
        kind=kind,
        target_id=target_id,
        action_key=action_key,
        invocation_nonce_sha256=nonce,
        correlation_identity_sha256=correlation,
        method="POST",
        path=path,
        body=body,
        total_max_attempts=1,
        command_identity_sha256=identity,
    )


def _validate_cancellation_command(
    value: object,
) -> CancellationCommand:
    if not isinstance(value, CancellationCommand):
        raise Task12RuntimeError("cancellation command must be typed")
    expected_path, expected_body = (
        ("/api/cancel", {"request_id": value.target_id})
        if value.kind == "REQUEST_CANCEL"
        else ("/jobs/cancel", {"job_id": value.target_id})
    ) if value.kind in {"REQUEST_CANCEL", "JOB_CANCEL"} else ("", {})
    if (
        value.kind not in {"REQUEST_CANCEL", "JOB_CANCEL"}
        or value.method != "POST"
        or value.path != expected_path
        or value.body != expected_body
        or value.total_max_attempts != 1
    ):
        raise Task12RuntimeError(
            "cancellation route is not the closed retained route"
        )
    expected_identity = canonical_sha256(
        {
            "kind": value.kind,
            "target_id": value.target_id,
            "action_key": value.action_key,
            "invocation_nonce_sha256": value.invocation_nonce_sha256,
            "correlation_identity_sha256": value.correlation_identity_sha256,
            "method": value.method,
            "path": value.path,
            "body": value.body,
            "total_max_attempts": value.total_max_attempts,
        }
    )
    if value.command_identity_sha256 != expected_identity:
        raise Task12RuntimeError("cancellation command identity drifted")
    return value


def reconcile_cancellation(
    *,
    command: CancellationCommand,
    observation: CancellationObservation,
) -> CancellationReconciliation:
    command = _validate_cancellation_command(command)
    if not isinstance(observation, CancellationObservation):
        raise Task12RuntimeError("cancellation observation must be typed")
    if observation.classification not in {
        "NOT_SENT",
        "DIRECT",
        "AMBIGUOUS",
        "POSITIVE_REJECTION",
    }:
        raise Task12RuntimeError("cancellation classification is not closed")
    if (
        type(observation.send_count) is not int
        or observation.send_count < 0
        or observation.send_count > 1
    ):
        raise Task12RuntimeError("cancellation permits at most one relay request")
    if observation.classification == "NOT_SENT" and observation.send_count != 0:
        raise Task12RuntimeError("not-sent cancellation has a send count")
    if observation.classification != "NOT_SENT" and observation.send_count != 1:
        raise Task12RuntimeError("sent cancellation lacks its single send")
    _text(observation.observed_target_state, "cancel target state")
    _sha(
        observation.observation_identity_sha256,
        "cancellation observation",
    )
    if observation.response_request_id is not None:
        _text(observation.response_request_id, "cancel response request ID")
    if observation.response_identity_sha256 is not None:
        _sha(observation.response_identity_sha256, "cancel response identity")
    if observation.classification == "DIRECT" and (
        observation.response_request_id is None
        or observation.response_identity_sha256 is None
    ):
        raise Task12RuntimeError("direct cancellation response is incomplete")
    terminal_states = (
        _REQUEST_TERMINAL
        if command.kind == "REQUEST_CANCEL"
        else _JOB_TERMINAL
    )
    terminal = observation.observed_target_state in terminal_states
    if terminal:
        state = (
            "REQUEST_CANCEL_TERMINAL"
            if command.kind == "REQUEST_CANCEL"
            else "JOB_CANCEL_TERMINAL"
        )
    elif observation.classification == "AMBIGUOUS":
        state = "AMBIGUOUS_RECONCILE_ONLY"
    elif observation.classification == "NOT_SENT":
        state = "NOT_APPLICABLE"
    else:
        state = "TERMINAL_OBSERVATION_REQUIRED"
    body = {
        "command_identity_sha256": command.command_identity_sha256,
        "state": state,
        "target_terminal": terminal,
        "may_resend": False,
        "observation_identity_sha256": (
            observation.observation_identity_sha256
        ),
    }
    return CancellationReconciliation(
        **body,
        reconciliation_identity_sha256=canonical_sha256(body),
    )


@dataclass(frozen=True)
class RetainedDeadlineAuthority:
    """Immutable T-60/T-50/T-30 edges from authenticated Task 10 authority."""

    worker_descriptor: WorkerBootstrapDescriptor
    descriptor_body_sha256: str
    activation_id: str
    generation: int
    execution_deadline: str
    stop_assignment_at: str
    graceful_stop_at: str
    retained_force_not_before: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class RetainedDeadlineDecision:
    authority: RetainedDeadlineAuthority
    authority_identity_sha256: str
    observed_at: str
    phase: str
    allow_new_assignments: bool
    request_graceful_stop: bool
    allow_controller_quiesced_force: bool
    prior_decision_identity_sha256: Optional[str]
    canonical_identity_sha256: str


def _deadline_authority_body(
    value: RetainedDeadlineAuthority,
) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def _validate_deadline_authority(
    value: object,
) -> RetainedDeadlineAuthority:
    if not isinstance(value, RetainedDeadlineAuthority):
        raise Task12RuntimeError(
            "retained deadline authority must be typed"
        )
    try:
        descriptor = validate_worker_bootstrap_descriptor(
            value.worker_descriptor
        )
    except Task10WorkerError as exc:
        raise Task12RuntimeError(
            "retained deadline descriptor is unauthenticated"
        ) from exc
    deadline = _utc(
        descriptor.execution_deadline,
        "retained execution deadline",
    )
    if (
        value.descriptor_body_sha256
        != descriptor.descriptor_body_sha256
        or value.activation_id != descriptor.activation_id
        or value.generation != descriptor.generation
        or value.execution_deadline != descriptor.execution_deadline
        or value.stop_assignment_at
        != _utc_text(deadline - timedelta(minutes=60))
        or value.graceful_stop_at
        != _utc_text(deadline - timedelta(minutes=50))
        or value.retained_force_not_before
        != _utc_text(deadline - timedelta(minutes=30))
        or value.canonical_identity_sha256
        != canonical_sha256(_deadline_authority_body(value))
    ):
        raise Task12RuntimeError(
            "retained deadline authority identity drifted"
        )
    return value


def build_retained_deadline_authority(
    *,
    worker_descriptor: WorkerBootstrapDescriptor,
    wake: Mapping[str, object],
) -> RetainedDeadlineAuthority:
    """Authenticate a wake as an accelerator and derive time only from Task 10."""

    try:
        descriptor = validate_worker_bootstrap_descriptor(
            worker_descriptor
        )
    except Task10WorkerError as exc:
        raise Task12RuntimeError(
            "retained deadline descriptor is unauthenticated"
        ) from exc
    if (
        type(wake) is not dict
        or set(wake)
        != {
            "schema_version",
            "record_type",
            "activation_id",
            "generation",
            "reason",
        }
    ):
        raise Task12RuntimeError(
            "retained deadline wake schema is not exact"
        )
    if (
        type(wake.get("schema_version")) is not int
        or wake.get("schema_version") != 1
        or wake.get("record_type")
        != "glm52_task12_retained_deadline_wake_v1"
        or wake.get("activation_id") != descriptor.activation_id
        or type(wake.get("generation")) is not int
        or wake.get("generation") != descriptor.generation
        or wake.get("reason") not in _DEADLINE_WAKE_REASONS
    ):
        raise Task12RuntimeError(
            "retained deadline wake authority is foreign"
        )
    deadline = _utc(
        descriptor.execution_deadline,
        "retained execution deadline",
    )
    body = {
        "worker_descriptor": descriptor,
        "descriptor_body_sha256": descriptor.descriptor_body_sha256,
        "activation_id": descriptor.activation_id,
        "generation": descriptor.generation,
        "execution_deadline": descriptor.execution_deadline,
        "stop_assignment_at": _utc_text(
            deadline - timedelta(minutes=60)
        ),
        "graceful_stop_at": _utc_text(
            deadline - timedelta(minutes=50)
        ),
        "retained_force_not_before": _utc_text(
            deadline - timedelta(minutes=30)
        ),
    }
    return _validate_deadline_authority(
        RetainedDeadlineAuthority(
            **body,
            canonical_identity_sha256=canonical_sha256(
                {
                    **body,
                    "worker_descriptor": asdict(descriptor),
                }
            ),
        )
    )


def _deadline_decision_body(
    value: RetainedDeadlineDecision,
) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("authority")
    body.pop("canonical_identity_sha256")
    return body


def _validate_deadline_decision(
    value: object,
) -> RetainedDeadlineDecision:
    if not isinstance(value, RetainedDeadlineDecision):
        raise Task12RuntimeError(
            "retained deadline decision must be typed"
        )
    authority = _validate_deadline_authority(value.authority)
    observed = _utc(value.observed_at, "retained deadline observation")
    if observed < _utc(
        authority.stop_assignment_at,
        "retained T-minus-60",
    ):
        wall_phase = "BEFORE_T_MINUS_60"
    elif observed < _utc(
        authority.graceful_stop_at,
        "retained T-minus-50",
    ):
        wall_phase = "T_MINUS_60"
    elif observed < _utc(
        authority.retained_force_not_before,
        "retained T-minus-30",
    ):
        wall_phase = "T_MINUS_50"
    else:
        wall_phase = "T_MINUS_30"
    if value.phase not in _DEADLINE_PHASE_RANK:
        raise Task12RuntimeError(
            "retained deadline decision phase is not closed"
        )
    effective_phase = value.phase
    if _DEADLINE_PHASE_RANK[effective_phase] < (
        _DEADLINE_PHASE_RANK[wall_phase]
    ):
        raise Task12RuntimeError(
            "retained deadline decision regressed behind wall authority"
        )
    expected_flags = {
        "allow_new_assignments": (
            effective_phase == "BEFORE_T_MINUS_60"
        ),
        "request_graceful_stop": effective_phase
        in {"T_MINUS_50", "T_MINUS_30"},
        "allow_controller_quiesced_force": (
            effective_phase == "T_MINUS_30"
        ),
    }
    if (
        value.authority_identity_sha256
        != authority.canonical_identity_sha256
        or value.allow_new_assignments
        is not expected_flags["allow_new_assignments"]
        or value.request_graceful_stop
        is not expected_flags["request_graceful_stop"]
        or value.allow_controller_quiesced_force
        is not expected_flags["allow_controller_quiesced_force"]
        or (
            value.prior_decision_identity_sha256 is not None
            and _SHA.fullmatch(
                value.prior_decision_identity_sha256
            )
            is None
        )
        or value.canonical_identity_sha256
        != canonical_sha256(_deadline_decision_body(value))
    ):
        raise Task12RuntimeError(
            "retained deadline decision identity drifted"
        )
    return value


def evaluate_retained_deadline(
    *,
    authority: RetainedDeadlineAuthority,
    observed_at: str,
    prior_decision: Optional[RetainedDeadlineDecision] = None,
) -> RetainedDeadlineDecision:
    """Advance a non-regressing retained phase from immutable absolute edges."""

    authority = _validate_deadline_authority(authority)
    observed = _utc(observed_at, "retained deadline observation")
    if observed < _utc(authority.stop_assignment_at, "retained T-minus-60"):
        wall_phase = "BEFORE_T_MINUS_60"
    elif observed < _utc(
        authority.graceful_stop_at,
        "retained T-minus-50",
    ):
        wall_phase = "T_MINUS_60"
    elif observed < _utc(
        authority.retained_force_not_before,
        "retained T-minus-30",
    ):
        wall_phase = "T_MINUS_50"
    else:
        wall_phase = "T_MINUS_30"
    prior_identity: Optional[str] = None
    effective_phase = wall_phase
    if prior_decision is not None:
        prior = _validate_deadline_decision(prior_decision)
        if (
            prior.authority_identity_sha256
            != authority.canonical_identity_sha256
        ):
            raise Task12RuntimeError(
                "retained deadline prior authority is foreign"
            )
        prior_identity = prior.canonical_identity_sha256
        if (
            _DEADLINE_PHASE_RANK[prior.phase]
            > _DEADLINE_PHASE_RANK[effective_phase]
        ):
            effective_phase = prior.phase
    body = {
        "authority_identity_sha256": authority.canonical_identity_sha256,
        "observed_at": observed_at,
        "phase": effective_phase,
        "allow_new_assignments": (
            effective_phase == "BEFORE_T_MINUS_60"
        ),
        "request_graceful_stop": effective_phase
        in {"T_MINUS_50", "T_MINUS_30"},
        "allow_controller_quiesced_force": (
            effective_phase == "T_MINUS_30"
        ),
        "prior_decision_identity_sha256": prior_identity,
    }
    return _validate_deadline_decision(
        RetainedDeadlineDecision(
            authority=authority,
            **body,
            canonical_identity_sha256=canonical_sha256(body),
        )
    )


@dataclass(frozen=True)
class ControllerStopEvidence:
    instance_id: str
    ec2_state: str
    stopped_at: str
    ssm_offline: bool
    controller_processes_unreachable: bool
    start_instances_denied: bool
    replacement_denied: bool
    new_launch_denied: bool
    evidence_identity_sha256: str


@dataclass(frozen=True)
class ControllerQuiescenceProof:
    controller_stop_identity_sha256: str
    first_scan_identity_sha256: str
    second_scan_identity_sha256: str
    minimum_quiet_seconds: int
    quiesced_at: str
    canonical_identity_sha256: str


def _member_ids(read: CompleteRuntimeRead) -> Tuple[str, ...]:
    return tuple(item.identity for item in read.members)


def _states_are(
    read: CompleteRuntimeRead,
    allowed: frozenset[str],
) -> bool:
    return all(item.state in allowed for item in read.members)


def _two_scans(
    *,
    first: RuntimeScan,
    second: RuntimeScan,
    minimum_quiet_seconds: int,
) -> Tuple[RuntimeScan, RuntimeScan]:
    first = _validate_scan(first)
    second = _validate_scan(second)
    if type(minimum_quiet_seconds) is not int or minimum_quiet_seconds <= 0:
        raise Task12RuntimeError("quiet interval must be positive")
    first_at = _utc(first.observed_at, "first scan observed_at")
    second_at = _utc(second.observed_at, "second scan observed_at")
    if second_at - first_at < timedelta(seconds=minimum_quiet_seconds):
        raise Task12RuntimeError("runtime scans are not sufficiently separated")
    return first, second


def prove_controller_quiescence(
    *,
    stop: ControllerStopEvidence,
    first: RuntimeScan,
    second: RuntimeScan,
    minimum_quiet_seconds: int,
) -> ControllerQuiescenceProof:
    if not isinstance(stop, ControllerStopEvidence):
        raise Task12RuntimeError("controller stop evidence must be typed")
    if (
        _INSTANCE.fullmatch(stop.instance_id) is None
        or stop.ec2_state not in {"stopped", "terminated"}
        or stop.ssm_offline is not True
        or stop.controller_processes_unreachable is not True
        or stop.start_instances_denied is not True
        or stop.replacement_denied is not True
        or stop.new_launch_denied is not True
    ):
        raise Task12RuntimeError("controller is not authoritatively stopped")
    stopped_at = _utc(stop.stopped_at, "controller stopped_at")
    _sha(stop.evidence_identity_sha256, "controller stop evidence")
    first, second = _two_scans(
        first=first,
        second=second,
        minimum_quiet_seconds=minimum_quiet_seconds,
    )
    if _utc(first.observed_at, "first scan observed_at") < stopped_at:
        raise Task12RuntimeError("quiescence scan predates controller stop")
    if first.controller.members or second.controller.members:
        raise Task12RuntimeError("controller work remains after stop")
    for before, after, label in (
        (first.requests, second.requests, "request"),
        (first.jobs, second.jobs, "job"),
        (first.workers, second.workers, "worker"),
        (first.allocations, second.allocations, "allocation"),
    ):
        if _member_ids(before) != _member_ids(after):
            if label == "worker":
                raise Task12RuntimeError(
                    "new or nonterminal worker remains after stop"
                )
            raise Task12RuntimeError("new or missing " + label + " after stop")
    body = {
        "controller_stop_identity_sha256": stop.evidence_identity_sha256,
        "first_scan_identity_sha256": _identity(first),
        "second_scan_identity_sha256": _identity(second),
        "minimum_quiet_seconds": minimum_quiet_seconds,
        "quiesced_at": second.observed_at,
    }
    return ControllerQuiescenceProof(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


@dataclass(frozen=True)
class ForcedWorkerTerminationAuthority:
    controller_quiesced_identity_sha256: str
    deadline_decision_identity_sha256: str
    instance_ids: Tuple[str, ...]
    operation: str
    total_max_attempts_per_action: int
    canonical_identity_sha256: str


def _validate_quiescence_proof(
    value: object,
) -> ControllerQuiescenceProof:
    if not isinstance(value, ControllerQuiescenceProof):
        raise Task12RuntimeError(
            "forced worker termination requires controller quiescence"
        )
    body = {
        "controller_stop_identity_sha256": (
            value.controller_stop_identity_sha256
        ),
        "first_scan_identity_sha256": value.first_scan_identity_sha256,
        "second_scan_identity_sha256": value.second_scan_identity_sha256,
        "minimum_quiet_seconds": value.minimum_quiet_seconds,
        "quiesced_at": value.quiesced_at,
    }
    if value.canonical_identity_sha256 != canonical_sha256(body):
        raise Task12RuntimeError("controller quiescence identity drifted")
    return value


def authorize_forced_worker_termination(
    *,
    quiescence: ControllerQuiescenceProof,
    deadline_decision: RetainedDeadlineDecision,
    instance_ids: Tuple[str, ...],
) -> ForcedWorkerTerminationAuthority:
    quiescence = _validate_quiescence_proof(quiescence)
    deadline_decision = _validate_deadline_decision(
        deadline_decision
    )
    if (
        deadline_decision.phase != "T_MINUS_30"
        or deadline_decision.allow_controller_quiesced_force is not True
        or _utc(
            deadline_decision.observed_at,
            "forced worker termination observation",
        )
        < _utc(
            deadline_decision.authority.retained_force_not_before,
            "retained T-minus-30",
        )
    ):
        raise Task12RuntimeError(
            "forced worker termination is before authenticated T-minus-30"
        )
    if (
        type(instance_ids) is not tuple
        or not instance_ids
        or tuple(sorted(instance_ids)) != instance_ids
        or len(instance_ids) != len(set(instance_ids))
        or any(_INSTANCE.fullmatch(item) is None for item in instance_ids)
    ):
        raise Task12RuntimeError("forced worker instance set is not exact")
    body = {
        "controller_quiesced_identity_sha256": (
            quiescence.canonical_identity_sha256
        ),
        "deadline_decision_identity_sha256": (
            deadline_decision.canonical_identity_sha256
        ),
        "instance_ids": instance_ids,
        "operation": "ec2:TerminateInstances",
        "total_max_attempts_per_action": 1,
    }
    return ForcedWorkerTerminationAuthority(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


@dataclass(frozen=True)
class NoJobTerminalProof:
    request_id: str
    request_state: str
    first_scan_identity_sha256: str
    second_scan_identity_sha256: str
    canonical_identity_sha256: str


def prove_no_job_terminal(
    *,
    request: RuntimeEntity,
    first: RuntimeScan,
    second: RuntimeScan,
    minimum_quiet_seconds: int,
) -> NoJobTerminalProof:
    if not isinstance(request, RuntimeEntity) or request.state not in {
        "FAILED",
        "CANCELLED",
    }:
        raise Task12RuntimeError(
            "no-job request must be exact terminal FAILED or CANCELLED"
        )
    first, second = _two_scans(
        first=first,
        second=second,
        minimum_quiet_seconds=minimum_quiet_seconds,
    )
    if _utc(first.observed_at, "first scan observed_at") < _utc(
        request.observed_at,
        "request observed_at",
    ):
        raise Task12RuntimeError("no-job scan predates request terminality")
    for scan in (first, second):
        if (
            scan.requests.members
            or scan.jobs.members
            or scan.controller.members
            or scan.workers.members
            or scan.allocations.members
        ):
            raise Task12RuntimeError(
                "no-job proof requires complete zero runtime scans"
            )
    body = {
        "request_id": request.identity,
        "request_state": request.state,
        "first_scan_identity_sha256": _identity(first),
        "second_scan_identity_sha256": _identity(second),
    }
    return NoJobTerminalProof(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


@dataclass(frozen=True)
class RuntimeTerminalProof:
    first_scan_identity_sha256: str
    second_scan_identity_sha256: str
    final_spend_ledger_head_identity_sha256: str
    final_worker_ids: Tuple[str, ...]
    final_allocation_ids: Tuple[str, ...]
    terminal_at: str
    canonical_identity_sha256: str


def prove_runtime_terminal(
    *,
    first: RuntimeScan,
    second: RuntimeScan,
    minimum_quiet_seconds: int,
) -> RuntimeTerminalProof:
    first, second = _two_scans(
        first=first,
        second=second,
        minimum_quiet_seconds=minimum_quiet_seconds,
    )
    for before, after, label in (
        (first.requests, second.requests, "request"),
        (first.jobs, second.jobs, "job"),
        (first.controller, second.controller, "controller"),
        (first.workers, second.workers, "worker"),
        (first.allocations, second.allocations, "allocation"),
    ):
        if _member_ids(before) != _member_ids(after):
            raise Task12RuntimeError("terminal " + label + " set drifted")
    if (
        not _states_are(first.requests, _REQUEST_TERMINAL)
        or not _states_are(second.requests, _REQUEST_TERMINAL)
        or not _states_are(first.jobs, _JOB_TERMINAL)
        or not _states_are(second.jobs, _JOB_TERMINAL)
        or first.controller.members
        or second.controller.members
        or not _states_are(first.workers, _WORKER_TERMINAL)
        or not _states_are(second.workers, _WORKER_TERMINAL)
        or not _states_are(first.allocations, _ALLOCATION_TERMINAL)
        or not _states_are(second.allocations, _ALLOCATION_TERMINAL)
        or first.spend.state != "CLOSED"
        or second.spend.state != "CLOSED"
        or first.spend.ledger_head_identity_sha256
        != second.spend.ledger_head_identity_sha256
    ):
        raise Task12RuntimeError(
            "runtime is not terminal with workers and allocations closed"
        )
    body = {
        "first_scan_identity_sha256": _identity(first),
        "second_scan_identity_sha256": _identity(second),
        "final_spend_ledger_head_identity_sha256": (
            second.spend.ledger_head_identity_sha256
        ),
        "final_worker_ids": _member_ids(second.workers),
        "final_allocation_ids": _member_ids(second.allocations),
        "terminal_at": second.observed_at,
    }
    return RuntimeTerminalProof(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


@dataclass(frozen=True)
class Task9LiabilitySettlementHandoff:
    activation_id: str
    allocation_ordinal: int
    terminal_v2_identity_sha256: str
    runtime_terminal_identity_sha256: str
    final_spend_ledger_head_identity_sha256: str
    canonical_identity_sha256: str


def build_task9_liability_settlement_handoff(
    *,
    activation_id: str,
    allocation_ordinal: int,
    terminal_v2_identity_sha256: str,
    runtime_terminal: RuntimeTerminalProof,
) -> Task9LiabilitySettlementHandoff:
    if (
        type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
        or type(allocation_ordinal) is not int
        or allocation_ordinal <= 0
        or not isinstance(runtime_terminal, RuntimeTerminalProof)
    ):
        raise Task12RuntimeError("Task 9 settlement handoff is invalid")
    terminal_v2 = _sha(
        terminal_v2_identity_sha256,
        "terminal-v2 identity",
    )
    _sha(runtime_terminal.canonical_identity_sha256, "runtime terminal identity")
    body = {
        "activation_id": activation_id,
        "allocation_ordinal": allocation_ordinal,
        "terminal_v2_identity_sha256": terminal_v2,
        "runtime_terminal_identity_sha256": (
            runtime_terminal.canonical_identity_sha256
        ),
        "final_spend_ledger_head_identity_sha256": (
            runtime_terminal.final_spend_ledger_head_identity_sha256
        ),
    }
    return Task9LiabilitySettlementHandoff(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


def settle_task9_liability(
    *,
    handoff: Task9LiabilitySettlementHandoff,
    coordinator: LiabilitySettlementCoordinator,
) -> Mapping[str, object]:
    """Delegate settlement to Task 9's sole coordinator without new policy."""

    if not isinstance(handoff, Task9LiabilitySettlementHandoff):
        raise Task12RuntimeError("Task 9 settlement handoff must be typed")
    if not isinstance(coordinator, LiabilitySettlementCoordinator):
        raise Task12RuntimeError(
            "settlement must use Task 9 LiabilitySettlementCoordinator"
        )
    result = coordinator.settle(
        handoff.activation_id,
        handoff.allocation_ordinal,
    )
    if (
        not isinstance(result, Mapping)
        or result.get("activation_id") != handoff.activation_id
        or result.get("allocation_ordinal") != handoff.allocation_ordinal
    ):
        raise Task12RuntimeError("Task 9 settlement result is foreign")
    _sha(result.get("canonical_body_sha256"), "Task 9 settlement identity")
    return result


__all__ = [
    "ACCOUNT_ID",
    "REGION",
    "RUN_ID",
    "CancellationCommand",
    "CancellationObservation",
    "CancellationReconciliation",
    "CompleteRuntimeRead",
    "ControllerQuiescenceProof",
    "ControllerStopEvidence",
    "ForcedWorkerTerminationAuthority",
    "NoJobTerminalProof",
    "RuntimeEntity",
    "RuntimeReadBoundaries",
    "RuntimeScan",
    "RuntimeTerminalProof",
    "RetainedDeadlineAuthority",
    "RetainedDeadlineDecision",
    "SpendRead",
    "Task12RuntimeError",
    "Task9LiabilitySettlementHandoff",
    "authorize_forced_worker_termination",
    "build_cancellation_command",
    "build_retained_deadline_authority",
    "build_task9_liability_settlement_handoff",
    "collect_runtime_scan",
    "evaluate_retained_deadline",
    "prove_controller_quiescence",
    "prove_no_job_terminal",
    "prove_runtime_terminal",
    "reconcile_cancellation",
    "settle_task9_liability",
]
