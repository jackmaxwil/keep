"""Current-authority Task 12 drain and TerminalV2 effects.

The retained workflow reaches these operations after recovery ownership has
been acquired.  This module never accepts a caller-projected request.  It
re-reads the exact recovery owner, support execution, complete Task 9 worker
families, spend authority, and EC2 state before constructing any domain
request.  Owner transactions use only the KMS continuation capsule carried by
the retained workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import time
from types import SimpleNamespace
from typing import Mapping

from .canonical import canonical_sha256
from .dynamodb import DynamoLedgerAdapter, ExactCheck, ExactUpdate, LedgerKey
from .records import (
    canonical_record_identity,
    ledger_pk,
    ledger_sk,
    validate_record,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
QUIET_SECONDS = 60
CONTROLLER_POLL_SECONDS = 5
CONTROLLER_MAX_POLLS = 36

QUIESCE = "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED"
TRANSFER = "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES"
WORKER_DRAIN = "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS"
TERMINAL_V2 = "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
ZERO_WORK = "RETAINED_PROVE_ZERO_ACTIVATION_WORK"

SUPPORTED_OPERATIONS = frozenset(
    {QUIESCE, TRANSFER, WORKER_DRAIN, TERMINAL_V2, ZERO_WORK}
)

_EXPECTED_SOURCES = {
    QUIESCE: {"request_job_correlation": "glm52_task12_request_job_correlation_v1"},
    TRANSFER: {"worker_drain_authority": "glm52_production_recovery_control"},
    WORKER_DRAIN: {"worker_drain_authority": "glm52_production_recovery_control"},
    TERMINAL_V2: {
        "recovery_control": "glm52_production_recovery_control",
        "worker_drain_authority": "glm52_production_recovery_control",
    },
    ZERO_WORK: {"control": "glm52_production_control"},
}


class Task12LiveDrainEffectsError(ValueError):
    """A live drain authority, effect, or readback was absent or divergent."""


@dataclass(frozen=True)
class LiabilityTransferProof:
    activation_id: str
    allocation_ordinals: tuple[int, ...]
    before_identity_sha256s: tuple[str, ...]
    after_identity_sha256s: tuple[str, ...]
    complete_family_identity_sha256: str
    owner_nonce_sha256: str
    state: str
    canonical_identity_sha256: str


def _fail(message: str) -> None:
    raise Task12LiveDrainEffectsError(message)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _sleep(seconds: int) -> None:
    time.sleep(seconds)


def _roles(ports: object) -> Mapping[str, object]:
    deployment = getattr(ports, "deployment", None)
    roles = getattr(deployment, "role_coordinates", None)
    if not isinstance(roles, Mapping):
        _fail("drain deployment coordinates are absent")
    return roles


def _metadata(response: object, label: str) -> Mapping[str, object]:
    metadata = response.get("ResponseMetadata") if type(response) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        _fail(label + " metadata must authenticate RetryAttempts == 0")
    return metadata


def _validate_invocation(invocation: object, operation_kind: str) -> None:
    if operation_kind not in SUPPORTED_OPERATIONS:
        _fail("drain effect operation is not supported")
    if getattr(invocation, "operation_kind", None) != operation_kind:
        _fail("drain effect invocation operation drifted")
    activation_id = getattr(invocation, "activation_id", None)
    activation_ordinal = getattr(invocation, "activation_ordinal", None)
    generation = getattr(invocation, "generation", None)
    if (
        type(activation_id) is not str
        or not activation_id
        or type(activation_ordinal) is not int
        or isinstance(activation_ordinal, bool)
        or activation_ordinal < 1
        or type(generation) is not int
        or isinstance(generation, bool)
        or generation < 1
        or getattr(invocation, "generation_text", None) != f"{generation:08d}"
    ):
        _fail("drain effect invocation identity drifted")


def _validate_sources(
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Validate every producer before the first service client is requested."""

    _validate_invocation(invocation, operation_kind)
    expected = _EXPECTED_SOURCES[operation_kind]
    if type(live_sources) is not dict or set(live_sources) != set(expected):
        _fail(operation_kind + " live producer set drifted")
    result: dict[str, dict[str, object]] = {}
    campaign_identity: str | None = None
    for alias, record_type in expected.items():
        try:
            value = validate_record(record_type, live_sources[alias])
        except (TypeError, ValueError) as exc:
            raise Task12LiveDrainEffectsError(
                operation_kind + " missing or stale producer " + alias
            ) from exc
        if (
            value.get("account_id") != ACCOUNT_ID
            or value.get("region") != REGION
            or value.get("run_id") != RUN_ID
            or value.get("activation_id")
            != getattr(invocation, "activation_id")
            or value.get("activation_ordinal")
            != getattr(invocation, "activation_ordinal")
        ):
            _fail(alias + " producer binding drifted")
        identity = value.get("campaign_identity_sha256")
        if type(identity) is not str or len(identity) != 64:
            _fail(alias + " campaign identity is absent")
        if campaign_identity is None:
            campaign_identity = identity
        elif campaign_identity != identity:
            _fail("drain producer campaign identities are swapped")
        result[alias] = value
    if operation_kind == TERMINAL_V2 and (
        result["recovery_control"] != result["worker_drain_authority"]
    ):
        _fail("TerminalV2 recovery producers are swapped or stale")
    return result


def _runtime_view(invocation: object) -> object:
    """Present the same invocation to the already-audited runtime collector."""

    from .task12_live_runtime import PROVE_TERMINAL

    values = {
        name: getattr(invocation, name)
        for name in (
            "activation_id",
            "activation_ordinal",
            "generation",
            "generation_text",
            "dispatch_identity_sha256",
            "state_machine_execution_arn",
            "caller_state_machine_version_arn",
            "operation_input",
        )
        if hasattr(invocation, name)
    }
    return SimpleNamespace(**values, operation_kind=PROVE_TERMINAL)


def _live_state(
    *,
    ports: object,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    from .task12_live_drain_terminal import _live_state as read_state

    try:
        state = read_state(
            ports=ports,
            invocation=invocation,
            live_sources=live_sources,
        )
    except ValueError as exc:
        raise Task12LiveDrainEffectsError(
            "current retained state read failed"
        ) from exc
    recovery = state["recovery_control"]
    if (
        recovery.get("state") != "OWNED"
        or recovery.get("owner_execution_arn")
        != getattr(invocation, "state_machine_execution_arn", None)
        or recovery.get("owner_state_machine_version_arn")
        != getattr(invocation, "caller_state_machine_version_arn", None)
    ):
        _fail("current recovery owner is absent or foreign")
    return state


def _execution(
    *, ports: object, invocation: object, state: Mapping[str, object]
) -> dict[str, object]:
    from .task12_live_runtime import _read_exact_execution

    control = state["control"]
    epoch = control.get("active_epoch")
    if type(epoch) is not int or epoch < 1:
        _fail("current support execution epoch is absent")
    return _read_exact_execution(
        ports=ports,
        execution={
            "activation_id": getattr(invocation, "activation_id"),
            "epoch": epoch,
        },
    )


def _correlation(
    operation_kind: str,
    sources: Mapping[str, Mapping[str, object]],
) -> str:
    return canonical_sha256(
        {
            "operation_kind": operation_kind,
            "sources": {
                alias: canonical_record_identity(
                    _EXPECTED_SOURCES[operation_kind][alias], value
                )
                for alias, value in sorted(sources.items())
            },
        }
    )


def _collect_scan(
    *,
    ports: object,
    invocation: object,
    execution: Mapping[str, object],
    campaign_identity_sha256: str,
    correlation_identity_sha256: str,
    require_execution_terminal: bool,
) -> tuple[object, dict[str, object], str]:
    from .task12_live_runtime import _collect_live_scan

    return _collect_live_scan(
        ports=ports,
        invocation=_runtime_view(invocation),
        execution=execution,
        campaign_identity_sha256=campaign_identity_sha256,
        correlation_identity_sha256=correlation_identity_sha256,
        require_execution_terminal=require_execution_terminal,
    )


def _two_scans(
    *,
    ports: object,
    invocation: object,
    sources: Mapping[str, Mapping[str, object]],
    require_execution_terminal: bool,
) -> tuple[object, object]:
    state = _live_state(
        ports=ports, invocation=invocation, live_sources=sources
    )
    execution = _execution(ports=ports, invocation=invocation, state=state)
    campaign = str(state["recovery_control"]["campaign_identity_sha256"])
    correlation = _correlation(getattr(invocation, "operation_kind"), sources)
    first, exact_execution, first_status = _collect_scan(
        ports=ports,
        invocation=invocation,
        execution=execution,
        campaign_identity_sha256=campaign,
        correlation_identity_sha256=correlation,
        require_execution_terminal=require_execution_terminal,
    )
    _sleep(QUIET_SECONDS)
    second, _execution_after, second_status = _collect_scan(
        ports=ports,
        invocation=invocation,
        execution=exact_execution,
        campaign_identity_sha256=campaign,
        correlation_identity_sha256=correlation,
        require_execution_terminal=require_execution_terminal,
    )
    if (
        first.observed_at == second.observed_at
        or (require_execution_terminal and first_status != second_status)
    ):
        _fail("two distinct stable runtime scans are required")
    return first, second


def _controller_instance(
    *, ports: object, invocation: object, correlation: str
) -> tuple[str, str]:
    from .task12_live_runtime import _numeric_observation

    observation = _numeric_observation(
        ports=ports,
        invocation=_runtime_view(invocation),
        correlation_identity_sha256=correlation,
    )
    members = observation["controller_snapshot"]["members"]
    if len(members) > 1:
        _fail("controller observation is multiple")
    if len(members) == 1:
        instance_id = members[0]["identity"]
    else:
        instance_id = _roles(ports).get("controller_instance_id")
    if (
        type(instance_id) is not str
        or not instance_id.startswith("i-")
        or len(instance_id) not in {10, 19}
    ):
        _fail("exact current controller instance is absent")
    return instance_id, str(observation["observed_at"])


def _stop_controller(
    *, ports: object, invocation: object, sources: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    correlation = _correlation(QUIESCE, sources)
    instance_id, _numeric_observed_at = _controller_instance(
        ports=ports, invocation=invocation, correlation=correlation
    )
    ec2 = ports.client("ec2")
    request_ids: list[str] = []

    def describe() -> str:
        response = ec2.describe_instances(
            InstanceIds=[instance_id],
            DryRun=False,
        )
        request_ids.append(
            str(_metadata(response, "controller DescribeInstances")["RequestId"])
        )
        reservations = response.get("Reservations")
        instances = (
            reservations[0].get("Instances")
            if type(reservations) is list and len(reservations) == 1
            else None
        )
        if type(instances) is not list or len(instances) != 1:
            _fail("controller DescribeInstances is not singular")
        instance = instances[0]
        state = instance.get("State") if type(instance) is dict else None
        name = state.get("Name") if type(state) is dict else None
        if instance.get("InstanceId") != instance_id or name not in {
            "pending",
            "running",
            "stopping",
            "stopped",
            "shutting-down",
            "terminated",
        }:
            _fail("controller EC2 state is foreign")
        return str(name)

    state = describe()
    if state in {"pending", "running"}:
        stopped = ec2.stop_instances(
            InstanceIds=[instance_id],
            Force=False,
            Hibernate=False,
            DryRun=False,
        )
        request_ids.append(
            str(_metadata(stopped, "controller StopInstances")["RequestId"])
        )
    for _attempt in range(CONTROLLER_MAX_POLLS + 1):
        state = describe()
        if state in {"stopped", "terminated"}:
            break
        _sleep(CONTROLLER_POLL_SECONDS)
    else:
        _fail("controller did not stop after bounded polling")
    ssm = ports.client("ssm").describe_instance_information(
        Filters=[{"Key": "InstanceIds", "Values": [instance_id]}],
        MaxResults=5,
    )
    request_ids.append(
        str(_metadata(ssm, "controller SSM readback")["RequestId"])
    )
    rows = ssm.get("InstanceInformationList")
    if type(rows) is not list or any(
        item.get("InstanceId") != instance_id for item in rows
    ):
        _fail("controller SSM readback is malformed")
    if any(item.get("PingStatus") == "Online" for item in rows):
        _fail("controller remains SSM-online after stop")
    body = {
        "instance_id": instance_id,
        "ec2_state": state,
        "stopped_at": _now(),
        "ssm_offline": True,
        "controller_processes_unreachable": True,
        "start_instances_denied": True,
        "replacement_denied": True,
        "new_launch_denied": True,
    }
    return {
        **body,
        "evidence_identity_sha256": canonical_sha256(
            {**body, "request_ids": request_ids}
        ),
    }


def _quiesce_request(
    *, ports: object, invocation: object, sources: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    from .task12_live_drain_terminal import _json_value

    # Recovery ownership and the terminal support execution must be current
    # before the first EC2 mutation.  The later scan path re-reads both; this
    # pre-effect read closes the authority/effect ordering gap.
    state = _live_state(
        ports=ports, invocation=invocation, live_sources=sources
    )
    execution = _execution(
        ports=ports, invocation=invocation, state=state
    )
    if execution["state"] not in {
        "SUCCEEDED",
        "FAILED",
        "TIMED_OUT",
        "ABORTED",
    }:
        _fail("controller quiescence requires terminal support execution")
    stop = _stop_controller(
        ports=ports, invocation=invocation, sources=sources
    )
    first, second = _two_scans(
        ports=ports,
        invocation=invocation,
        sources=sources,
        require_execution_terminal=True,
    )
    if first.controller.members or second.controller.members:
        _fail("controller work remains after the stop effect")
    return {
        "stop": stop,
        "first_runtime_scan": _json_value(first),
        "second_runtime_scan": _json_value(second),
        "minimum_quiet_seconds": QUIET_SECONDS,
    }


def _owner_capsule(
    *, ports: object, invocation: object, recovery: Mapping[str, object]
) -> tuple[dict[str, object], bytes]:
    from .task12_live_drain_terminal import _continuation_capsule
    from .task12_nonce_capsule import decrypt_owner_nonce_capsule

    capsule = _continuation_capsule(
        invocation,
        recovery,
        authority_domain="RECOVERY",
        barrier_field="recovery_barrier_nonce_sha256",
        control_revision_field="support_control_revision_at_seal",
    )
    authority = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": recovery["activation_id"],
        "authority_domain": "RECOVERY",
        "owner_execution_arn": recovery["owner_execution_arn"],
        "owner_state_machine_version_arn": (
            recovery["owner_state_machine_version_arn"]
        ),
        "owner_attempt": recovery["owner_attempt"],
        "barrier_nonce_sha256": recovery["recovery_barrier_nonce_sha256"],
        "control_revision": recovery["support_control_revision_at_seal"],
        "owner_hard_expires_at": recovery["owner_hard_expires_at"],
    }
    raw = decrypt_owner_nonce_capsule(
        ports=ports, capsule=capsule, expected_authority=authority
    )
    return capsule, raw


def _families(
    *, ports: object, invocation: object, campaign_identity_sha256: str
) -> Mapping[str, tuple[dict[str, object], ...]]:
    from .task12_live_runtime import _query_runtime_families

    families, _request_ids = _query_runtime_families(
        ports=ports,
        invocation=_runtime_view(invocation),
        campaign_identity_sha256=campaign_identity_sha256,
    )
    return families


def _family_identity(
    families: Mapping[str, tuple[dict[str, object], ...]],
) -> str:
    return canonical_sha256(
        {
            name: [
                canonical_record_identity(name, row)
                for row in rows
            ]
            for name, rows in sorted(families.items())
        }
    )


def _liability_after(
    before: Mapping[str, object],
    recovery: Mapping[str, object],
    *,
    observed_at: str,
    nonce_sha256: str,
) -> dict[str, object]:
    hard_expiry = str(recovery["owner_hard_expires_at"])
    if hard_expiry <= observed_at:
        _fail("recovery owner expires before liability watch")
    after = {
        **before,
        "state": "WATCHING",
        "watch_started_at": observed_at,
        "watch_not_before": observed_at,
        "watch_not_after": hard_expiry,
        "next_scan_at": observed_at,
        "owner_attempt": recovery["owner_attempt"],
        "owner_execution_arn": recovery["owner_execution_arn"],
        "owner_state_machine_version_arn": (
            recovery["owner_state_machine_version_arn"]
        ),
        "owner_dispatch_identity_sha256": (
            recovery["owner_dispatch_identity_sha256"]
        ),
        "owner_invocation_nonce_sha256": nonce_sha256,
        "owner_hard_expires_at": hard_expiry,
        "revision": int(before["revision"]) + 1,
        "updated_at": observed_at,
    }
    try:
        from .transitions import validate_transition

        return validate_transition(
            "glm52_production_worker_launch_liability", before, after
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveDrainEffectsError(
            "liability owner transfer candidate is invalid"
        ) from exc


def _liability_has_current_owner(
    value: Mapping[str, object],
    recovery: Mapping[str, object],
    *,
    nonce_sha256: str,
) -> bool:
    return all(
        value.get(field) == expected
        for field, expected in (
            ("owner_attempt", recovery["owner_attempt"]),
            ("owner_execution_arn", recovery["owner_execution_arn"]),
            (
                "owner_state_machine_version_arn",
                recovery["owner_state_machine_version_arn"],
            ),
            (
                "owner_dispatch_identity_sha256",
                recovery["owner_dispatch_identity_sha256"],
            ),
            ("owner_invocation_nonce_sha256", nonce_sha256),
            ("owner_hard_expires_at", recovery["owner_hard_expires_at"]),
        )
    )


def _transfer_liabilities(
    *, ports: object, invocation: object, sources: Mapping[str, Mapping[str, object]]
) -> LiabilityTransferProof:
    state = _live_state(
        ports=ports, invocation=invocation, live_sources=sources
    )
    recovery = state["recovery_control"]
    capsule, raw_nonce = _owner_capsule(
        ports=ports, invocation=invocation, recovery=recovery
    )
    if capsule["nonce_sha256"] != recovery[
        "owner_invocation_nonce_sha256"
    ]:
        raw_nonce = b""
        _fail("TerminalV2 recovery owner capsule drifted")
    campaign = str(recovery["campaign_identity_sha256"])
    before_families = _families(
        ports=ports,
        invocation=invocation,
        campaign_identity_sha256=campaign,
    )
    liabilities = before_families[
        "glm52_production_worker_launch_liability"
    ]
    index = state["activation_index"]
    adapter = DynamoLedgerAdapter(
        client=ports.client("dynamodb"),
        table_name=str(_roles(ports)["ledger_table_name"]),
    )
    observed_at = _now()
    before_ids: list[str] = []
    after_ids: list[str] = []
    ordinals: list[int] = []
    try:
        for before in liabilities:
            ordinal = int(before["allocation_ordinal"])
            if before["state"] == "UNOWNED_NOT_ACTIONABLE":
                after = _liability_after(
                    before,
                    recovery,
                    observed_at=observed_at,
                    nonce_sha256=str(capsule["nonce_sha256"]),
                )
                adapter.acquire_liability_owner(
                    index=ExactCheck(
                        LedgerKey(
                            RUN_ID,
                            ledger_sk("glm52_production_activation_index"),
                        ),
                        index,
                    ),
                    liability=ExactUpdate(
                        LedgerKey(
                            RUN_ID,
                            ledger_sk(
                                "glm52_production_worker_launch_liability",
                                activation_id=getattr(
                                    invocation, "activation_id"
                                ),
                                allocation_ordinal=ordinal,
                            ),
                        ),
                        before,
                        after,
                    ),
                    domain="LIABILITY_OWNER_ACQUIRE",
                    operation_identity_sha256=canonical_sha256(
                        {
                            "operation_kind": TRANSFER,
                            "liability_before": canonical_record_identity(
                                "glm52_production_worker_launch_liability",
                                before,
                            ),
                            "recovery_owner": canonical_record_identity(
                                "glm52_production_recovery_control", recovery
                            ),
                        }
                    ),
                    raw_owner_nonce=raw_nonce,
                )
            elif (
                not _liability_has_current_owner(
                    before,
                    recovery,
                    nonce_sha256=str(capsule["nonce_sha256"]),
                )
                or before["state"]
                not in {
                    "WATCHING",
                    "SAME_TOKEN_COMPLETION",
                    "REJECTION_PROVED_AWAITING_TERMINAL_V2",
                    "LATE_INSTANCE_DRAINING",
                }
            ):
                _fail("liability is owned by a stale or foreign recovery")
            ordinals.append(ordinal)
            before_ids.append(
                canonical_record_identity(
                    "glm52_production_worker_launch_liability", before
                )
            )
        after_families = _families(
            ports=ports,
            invocation=invocation,
            campaign_identity_sha256=campaign,
        )
    finally:
        raw_nonce = b""
    after_rows = after_families[
        "glm52_production_worker_launch_liability"
    ]
    if [int(row["allocation_ordinal"]) for row in after_rows] != ordinals:
        _fail("liability family multiplicity changed during transfer")
    for row in after_rows:
        if (
            row["state"] not in {
                "WATCHING",
                "SAME_TOKEN_COMPLETION",
                "REJECTION_PROVED_AWAITING_TERMINAL_V2",
                "LATE_INSTANCE_DRAINING",
            }
            or not _liability_has_current_owner(
                row,
                recovery,
                nonce_sha256=str(capsule["nonce_sha256"]),
            )
        ):
            _fail("liability transfer readback is not current-owner exact")
        after_ids.append(
            canonical_record_identity(
                "glm52_production_worker_launch_liability", row
            )
        )
    family_identity = _family_identity(after_families)
    body = {
        "activation_id": getattr(invocation, "activation_id"),
        "allocation_ordinals": tuple(ordinals),
        "before_identity_sha256s": tuple(before_ids),
        "after_identity_sha256s": tuple(after_ids),
        "complete_family_identity_sha256": family_identity,
        "owner_nonce_sha256": str(capsule["nonce_sha256"]),
        "state": "LIABILITIES_TRANSFERRED",
    }
    return LiabilityTransferProof(
        **body, canonical_identity_sha256=canonical_sha256(body)
    )


def validate_liability_transfer_proof(value: object) -> LiabilityTransferProof:
    if type(value) is not LiabilityTransferProof:
        _fail("liability transfer proof must be typed")
    body = asdict(value)
    identity = body.pop("canonical_identity_sha256")
    ordinals = tuple(body["allocation_ordinals"])
    if (
        body["state"] != "LIABILITIES_TRANSFERRED"
        or ordinals != tuple(sorted(set(ordinals)))
        or any(type(item) is not int or item < 1 for item in ordinals)
        or len(body["before_identity_sha256s"]) != len(ordinals)
        or len(body["after_identity_sha256s"]) != len(ordinals)
        or identity != canonical_sha256(body)
    ):
        _fail("liability transfer proof is stale or incomplete")
    return value


def _transfer_request(
    *, ports: object, invocation: object, sources: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    """Return a current-effect proof; no future TerminalV2 identity is accepted."""

    proof = _transfer_liabilities(
        ports=ports, invocation=invocation, sources=sources
    )
    return {"liability_transfer": asdict(proof)}


def _authority_pair(
    *,
    candidate_identity_sha256: str,
    action_kind: str,
    recovery: Mapping[str, object],
    observed_at: str,
) -> tuple[dict[str, object], dict[str, object]]:
    from .task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
    )

    revision = int(recovery["revision"])
    authorized = revision + 1
    action_key = (
        "ACTIVATION#"
        + str(recovery["activation_id"])
        + "#RECOVERY_ACTION#"
        + action_kind
        + "#"
        + f"{authorized:08d}"
    )
    action_body = {
        "authority_domain": "RECOVERY",
        "action_kind": action_kind,
        "action_key": action_key,
        "candidate_identity_sha256": candidate_identity_sha256,
        "owner_invocation_nonce_sha256": (
            recovery["owner_invocation_nonce_sha256"]
        ),
        "state": "CONSUMED",
        "authorized_revision": authorized,
    }
    action = RetainedWriterActionAuthority(
        **action_body,
        action_identity_sha256=canonical_sha256(action_body),
    )
    audit_body = {
        "authority_domain": "RECOVERY",
        "action_kind": action_kind,
        "action_key": action_key,
        "candidate_identity_sha256": candidate_identity_sha256,
        "action_identity_sha256": action.action_identity_sha256,
        "audit_kind": "H1F_GENESIS_TO_ZERO_CHILD",
        "closing_revision": revision,
        "authorized_revision": authorized,
        "current_revision": authorized,
        "observed_at": observed_at,
    }
    audit = RetainedWriterAuditAuthority(
        **audit_body,
        canonical_identity_sha256=canonical_sha256(audit_body),
    )
    return asdict(action), asdict(audit)


def _exact_s3_mapping(
    *, ports: object, coordinate: object, label: str
) -> dict[str, object]:
    from .task12_live_runtime import _get_exact_object

    if (
        type(coordinate) is not dict
        or set(coordinate)
        != {"bucket", "key", "version_id", "file_sha256", "body_sha256"}
    ):
        _fail(label + " coordinate is absent")
    value, _raw, _request_id = _get_exact_object(
        ports=ports,
        bucket=str(coordinate["bucket"]),
        key=str(coordinate["key"]),
        version_id=str(coordinate["version_id"]),
        expected_file_sha256=str(coordinate["file_sha256"]),
        label=label,
    )
    if value.get(
        "descriptor_body_sha256",
        value.get("canonical_identity_sha256"),
    ) != coordinate["body_sha256"]:
        _fail(label + " body identity drifted")
    return value


def _reserve_rows(
    *, ports: object, invocation: object
) -> tuple[dict[str, object], ...]:
    from .dynamodb import decode_item, encode_item

    activation_id = getattr(invocation, "activation_id")
    prefix = "GPU_LIABILITY_RESERVE#" + activation_id + "#"
    response = ports.client("dynamodb").query(
        TableName=str(_roles(ports)["ledger_table_name"]),
        KeyConditionExpression="#pk = :pk AND begins_with(#sk, :prefix)",
        ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
        ExpressionAttributeValues=encode_item(
            {":pk": ledger_pk(RUN_ID), ":prefix": prefix}
        ),
        ConsistentRead=True,
        ScanIndexForward=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(response, "GPU liability reserve Query")
    if response.get("LastEvaluatedKey") is not None:
        _fail("GPU liability reserve Query is not complete")
    items = response.get("Items")
    if (
        type(items) is not list
        or response.get("Count") != len(items)
        or response.get("ScannedCount") != len(items)
    ):
        _fail("GPU liability reserve Query is malformed")
    rows: list[dict[str, object]] = []
    for physical in items:
        value = decode_item(physical)
        if value.pop("PK", None) != ledger_pk(RUN_ID):
            _fail("GPU liability reserve partition drifted")
        sort_key = value.pop("SK", None)
        body = dict(value)
        identity = body.pop("canonical_body_sha256", None)
        if (
            type(sort_key) is not str
            or not sort_key.startswith(prefix)
            or value.get("record_type") != "glm52_gpu_liability_reserve_v1"
            or value.get("reserve_key") != sort_key
            or identity != hashlib.sha256(
                __import__(
                    "glm52_enforcement.spend_authority",
                    fromlist=("canonical_decimal_json_bytes",),
                ).canonical_decimal_json_bytes(body)
            ).hexdigest()
        ):
            _fail("GPU liability reserve row drifted")
        rows.append(value)
    rows.sort(key=lambda row: int(row["allocation_ordinal"]))
    if len({int(row["allocation_ordinal"]) for row in rows}) != len(rows):
        _fail("GPU liability reserve multiplicity drifted")
    return tuple(rows)


def _worker_request(
    *, ports: object, invocation: object, sources: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    from .spend_authority import (
        AllocationInterval,
        GpuLiabilityReserveResult,
    )
    from .task10_worker import (
        GRACEFUL_SCRIPT_NAMES,
        UNIT_NAMES,
        build_worker_instance_observation,
        render_worker_units,
        worker_bootstrap_descriptor_from_mapping,
        worker_unit_hashes,
    )
    from .task12_live_runtime import _ec2_instances, _spend_authority
    from .task12_worker_drain import prepare_worker_drain_candidate

    state = _live_state(
        ports=ports, invocation=invocation, live_sources=sources
    )
    recovery = state["recovery_control"]
    campaign = str(recovery["campaign_identity_sha256"])
    families = _families(
        ports=ports,
        invocation=invocation,
        campaign_identity_sha256=campaign,
    )
    execution = _execution(ports=ports, invocation=invocation, state=state)
    correlation = _correlation(WORKER_DRAIN, sources)
    scan, _execution_row, _status = _collect_scan(
        ports=ports,
        invocation=invocation,
        execution=execution,
        campaign_identity_sha256=campaign,
        correlation_identity_sha256=correlation,
        require_execution_terminal=True,
    )
    spend, intervals, _chain, _request_ids = _spend_authority(
        ports=ports,
        invocation=_runtime_view(invocation),
        observed_at=scan.observed_at,
        campaign_identity_sha256=campaign,
        families=families,
    )
    ec2, _ec2_requests = _ec2_instances(
        ports=ports, invocation=_runtime_view(invocation)
    )
    open_intervals = [
        row
        for row in intervals
        if row["state"] == "OPEN"
        and ec2.get(str(row["instance_id"]), {}).get("_state_name")
        in {"pending", "running", "stopping"}
    ]
    if len(open_intervals) != 1:
        _fail("worker drain requires exactly one current open worker")
    interval = open_intervals[0]
    instance_id = str(interval["instance_id"])
    launches = [
        row
        for row in families["glm52_production_worker_launch"]
        if instance_id in row["observed_instance_ids"]
    ]
    if len(launches) != 1:
        _fail("worker drain launch multiplicity is not singular")
    launch = launches[0]
    descriptor = worker_bootstrap_descriptor_from_mapping(
        _exact_s3_mapping(
            ports=ports,
            coordinate=_roles(ports).get(
                "task10_worker_descriptor_coordinate"
            ),
            label="Task10 worker descriptor",
        )
    )
    allocation = AllocationInterval(
        instance_id=instance_id,
        job_id=str(interval["job_id"]),
        started_at=str(interval["started_at"]),
        ended_at=None,
        charged_seconds=int(interval["charged_seconds"]),
        charged_cost_usd=Decimal(str(interval["charged_cost_usd"])),
        state="OPEN",
    )
    observation = build_worker_instance_observation(
        schema_version=1,
        record_type="glm52_task10_worker_instance_observation_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        campaign_identity_sha256=campaign,
        activation_id=getattr(invocation, "activation_id"),
        activation_ordinal=getattr(invocation, "activation_ordinal"),
        activation_ordinal_text=(
            f"{getattr(invocation, 'activation_ordinal'):08d}"
        ),
        generation=getattr(invocation, "generation"),
        generation_text=getattr(invocation, "generation_text"),
        allocation_ordinal=int(launch["allocation_ordinal"]),
        allocation_ordinal_text=str(launch["allocation_ordinal_text"]),
        instance_id=instance_id,
        action_key=str(launch["sky_action_key"]),
        sky_job_name=str(launch["sky_job_name"]),
        task_yaml_sha256=str(launch["task_yaml_sha256"]),
        request_body_sha256=str(launch["request_body_sha256"]),
        task9_launch_identity_sha256=(
            descriptor.task9_launch_identity_sha256
        ),
        task9_custody_identity_sha256=(
            descriptor.task9_custody_identity_sha256
        ),
    )
    reserve_rows = [
        row
        for row in _reserve_rows(ports=ports, invocation=invocation)
        if int(row["allocation_ordinal"])
        == int(launch["allocation_ordinal"])
    ]
    if len(reserve_rows) != 1:
        _fail("worker liability reserve multiplicity is not singular")
    reserve_row = reserve_rows[0]
    reserve = GpuLiabilityReserveResult(
        disposition="EXACT_DUPLICATE",
        reserve_key=str(reserve_row["reserve_key"]),
        reserve_identity_sha256=str(
            reserve_row["canonical_body_sha256"]
        ),
        gpu_reserve_seconds=int(reserve_row["gpu_reserve_seconds"]),
        gpu_reserve_cost_usd=Decimal(
            str(reserve_row["gpu_reserve_cost_usd"])
        ),
        root_volume_gib=int(reserve_row["root_volume_gib"]),
        root_volume_tail_usd_max=Decimal(
            str(reserve_row["root_volume_tail_usd_max"])
        ),
        remaining_gpu_seconds=int(spend.remaining_gpu_seconds),
        remaining_gpu_cost_usd=Decimal(spend.remaining_gpu_cost_usd),
        record=reserve_row,
    )
    unit_hashes = worker_unit_hashes(dict(render_worker_units()))
    scripts = _roles(ports).get("worker_script_hashes")
    if (
        type(scripts) is not dict
        or set(scripts) != set(GRACEFUL_SCRIPT_NAMES)
        or any(
            type(value) is not str or len(value) != 64
            for value in scripts.values()
        )
    ):
        _fail("published worker script hashes are absent")
    if set(unit_hashes) != set(UNIT_NAMES):
        _fail("published worker unit hashes are absent")
    candidate = prepare_worker_drain_candidate(
        allocation=allocation,
        liability_reserve=reserve,
        worker_descriptor=descriptor,
        worker_observation=observation,
        expected_unit_hashes=unit_hashes,
        expected_script_hashes=scripts,
        document_version=str(
            _roles(ports).get("worker_drain_document_version")
        ),
    )
    action, audit = _authority_pair(
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        action_kind="WORKER_DRAIN_SIGNAL",
        recovery=recovery,
        observed_at=scan.observed_at,
    )
    return {
        "candidate": asdict(candidate),
        "action": action,
        "audit": audit,
    }


_TERMINAL_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "campaign_identity_sha256",
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "source_family_identity_sha256",
        "handoff",
        "binding",
        "final_sky_state",
        "final_ec2_states",
        "request_cardinality",
        "allocations",
        "worker_launch_evidence",
        "worker_launch_liabilities",
        "final_heartbeat_identity",
        "checkpoint_identity",
        "cache_identity",
        "training_identity",
        "evaluation_identity",
        "drain_identity",
        "request_evidence",
        "post_terminal_quiescence_evidence",
        "prior_terminal_v1_identity",
        "outcome",
        "operator_disposition_required",
        "evidence_created_at",
        "canonical_body_sha256",
    }
)


def _validate_terminal_evidence(
    value: object,
    *,
    invocation: object,
    recovery: Mapping[str, object],
    families: Mapping[str, tuple[dict[str, object], ...]],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _TERMINAL_EVIDENCE_FIELDS:
        _fail("Task9 terminal evidence schema is incomplete")
    body = dict(value)
    supplied = body.pop("canonical_body_sha256")
    evidence_created_at = value.get("evidence_created_at")
    try:
        parsed_evidence_created_at = datetime.strptime(
            evidence_created_at, "%Y-%m-%dT%H:%M:%SZ"
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveDrainEffectsError(
            "Task9 terminal evidence timestamp is not canonical UTC"
        ) from exc
    if (
        value["schema_version"] != 1
        or value["record_type"] != "glm52_task9_terminal_evidence_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
        or value["campaign_identity_sha256"]
        != recovery["campaign_identity_sha256"]
        or value["activation_id"] != getattr(invocation, "activation_id")
        or value["activation_ordinal"]
        != getattr(invocation, "activation_ordinal")
        or value["generation"] != getattr(invocation, "generation")
        or value["generation_text"] != getattr(invocation, "generation_text")
        or value["source_family_identity_sha256"]
        != _family_identity(families)
        or supplied != canonical_sha256(body)
        or parsed_evidence_created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        != evidence_created_at
    ):
        _fail("Task9 terminal evidence identity drifted")
    for field in (
        "allocations",
        "worker_launch_evidence",
        "worker_launch_liabilities",
        "request_evidence",
        "final_ec2_states",
    ):
        if type(value[field]) is not list:
            _fail("Task9 terminal evidence arrays are not exact")
    launches = families["glm52_production_worker_launch"]
    liabilities = families[
        "glm52_production_worker_launch_liability"
    ]
    launch_by_ordinal = {
        int(row["allocation_ordinal"]): row for row in launches
    }
    liability_by_ordinal = {
        int(row["allocation_ordinal"]): row for row in liabilities
    }
    evidence_by_ordinal = {
        entry.get("allocation_ordinal"): entry
        for entry in value["worker_launch_evidence"]
        if type(entry) is dict
    }
    liability_evidence_by_ordinal = {
        entry.get("allocation_ordinal"): entry
        for entry in value["worker_launch_liabilities"]
        if type(entry) is dict
    }
    if (
        set(evidence_by_ordinal) != set(launch_by_ordinal)
        or len(evidence_by_ordinal)
        != len(value["worker_launch_evidence"])
        or set(liability_evidence_by_ordinal)
        != set(liability_by_ordinal)
        or len(liability_evidence_by_ordinal)
        != len(value["worker_launch_liabilities"])
    ):
        _fail("Task9 terminal evidence multiplicity drifted")
    for ordinal, launch in launch_by_ordinal.items():
        entry = evidence_by_ordinal[ordinal]
        owner_history = entry.get("owner_history")
        if (
            entry.get("worker_launch_identity_sha256")
            != canonical_record_identity(
                "glm52_production_worker_launch", launch
            )
            or entry.get("ec2_client_token")
            != launch["ec2_client_token"]
            or entry.get("launch_parameters_sha256")
            != launch["launch_parameters_sha256"]
            or entry.get("expected_worker_tags_sha256")
            != launch["expected_worker_tags_sha256"]
            or entry.get("run_instances_attempt_evidence")
            != launch["run_instances_attempt_evidence"]
            or entry.get("observed_instance_ids")
            != launch["observed_instance_ids"]
            or type(owner_history) is not list
            or len(owner_history) != int(launch["owner_attempt"])
        ):
            _fail("Task9 worker-launch terminal evidence drifted")
        latest_owner = owner_history[-1] if owner_history else None
        if (
            type(latest_owner) is not dict
            or latest_owner.get("owner_attempt") != launch["owner_attempt"]
            or latest_owner.get("owner_principal_arn")
            != launch["owner_principal_arn"]
            or latest_owner.get("owner_function_version_arn")
            != launch["owner_function_version_arn"]
            or latest_owner.get("owner_dispatch_identity_sha256")
            != launch["owner_dispatch_identity_sha256"]
            or latest_owner.get("owner_invocation_nonce_sha256")
            != launch["owner_invocation_nonce_sha256"]
            or latest_owner.get("owner_hard_expires_at")
            != launch["owner_hard_expires_at"]
        ):
            _fail("Task9 worker-launch owner history is incomplete")
    for ordinal, liability in liability_by_ordinal.items():
        entry = liability_evidence_by_ordinal[ordinal]
        launch = launch_by_ordinal.get(ordinal)
        if (
            launch is None
            or entry.get("worker_launch_identity_sha256")
            != canonical_record_identity(
                "glm52_production_worker_launch", launch
            )
            or entry.get("worker_launch_liability_identity_sha256")
            != canonical_record_identity(
                "glm52_production_worker_launch_liability", liability
            )
            or entry.get("ec2_client_token")
            != liability["ec2_client_token"]
            or entry.get("state") != liability["state"]
        ):
            _fail("Task9 liability terminal evidence drifted")
    allocation_keys: set[tuple[int, str]] = set()
    for allocation in value["allocations"]:
        if type(allocation) is not dict:
            _fail("Task9 allocation terminal evidence is incomplete")
        ordinal = allocation.get("allocation_ordinal")
        instance_id = allocation.get("instance_id")
        launch = launch_by_ordinal.get(ordinal)
        if (
            launch is None
            or instance_id not in launch["observed_instance_ids"]
            or allocation.get("instance_tags_sha256")
            != launch["expected_worker_tags_sha256"]
            or allocation.get("allocation_open_identity_sha256")
            != launch["spend_allocation_open_identity_sha256"]
            or allocation.get("allocation_close_identity_sha256")
            != launch["spend_allocation_close_identity_sha256"]
            or allocation.get("instance_terminal_identity_sha256")
            != launch["instance_terminal_identity_sha256"]
        ):
            _fail("Task9 allocation terminal evidence drifted")
        allocation_keys.add((int(ordinal), str(instance_id)))
    if len(allocation_keys) != len(value["allocations"]):
        _fail("Task9 allocation terminal evidence is duplicated")
    return dict(value)


def _read_terminal_evidence(
    *,
    ports: object,
    invocation: object,
    recovery: Mapping[str, object],
    families: Mapping[str, tuple[dict[str, object], ...]],
) -> dict[str, object]:
    from .task12_live_runtime import (
        _get_exact_object,
        _list_current_versions,
    )

    roles = _roles(ports)
    bucket = roles.get("campaign_bucket")
    prefix = roles.get("terminal_evidence_prefix")
    expected_prefix = (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        f"{getattr(invocation, 'generation_text')}/terminal-evidence/"
    )
    if (
        type(bucket) is not str
        or not bucket
        or prefix != expected_prefix
    ):
        _fail("Task9 terminal evidence coordinate is absent")
    expected_key = expected_prefix + "TASK9_TERMINAL_EVIDENCE.json"
    current = _list_current_versions(
        ports=ports,
        bucket=bucket,
        prefix=expected_prefix,
    )
    if len(current) != 1 or current[0][0] != expected_key:
        _fail("Task9 terminal evidence current version is not singular")
    value, _raw, _request_id = _get_exact_object(
        ports=ports,
        bucket=bucket,
        key=expected_key,
        version_id=current[0][1],
        expected_file_sha256=None,
        label="Task9 terminal evidence",
    )
    return _validate_terminal_evidence(
        value,
        invocation=invocation,
        recovery=recovery,
        families=families,
    )


def _terminal_base(
    *,
    invocation: object,
    recovery: Mapping[str, object],
    proof: object,
    second_scan: object,
    families: Mapping[str, tuple[dict[str, object], ...]],
    terminal_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build TerminalV2 from current scans and exact Task9 evidence."""

    launches = families["glm52_production_worker_launch"]
    liabilities = families[
        "glm52_production_worker_launch_liability"
    ]
    created_at = proof.terminal_at
    if launches or liabilities:
        evidence = _validate_terminal_evidence(
            terminal_evidence,
            invocation=invocation,
            recovery=recovery,
            families=families,
        )
    else:
        if terminal_evidence is not None:
            _fail("zero-launch TerminalV2 forbids unrelated Task9 evidence")
        evidence = None
    if evidence is not None and (
        type(evidence["evidence_created_at"]) is not str
        or evidence["evidence_created_at"] > created_at
    ):
        _fail("Task9 terminal evidence postdates the terminal observation")
    allocations = [] if evidence is None else list(evidence["allocations"])
    launch_evidence = (
        [] if evidence is None else list(evidence["worker_launch_evidence"])
    )
    liability_evidence = (
        []
        if evidence is None
        else list(evidence["worker_launch_liabilities"])
    )
    request_evidence = (
        [] if evidence is None else list(evidence["request_evidence"])
    )
    body = {
        "schema_version": 2,
        "record_type": "glm52_production_terminal_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": recovery[
            "campaign_identity_sha256"
        ],
        "activation_id": getattr(invocation, "activation_id"),
        "activation_ordinal": getattr(invocation, "activation_ordinal"),
        "generation": getattr(invocation, "generation"),
        "generation_text": getattr(invocation, "generation_text"),
        "action_key": (
            "ACTIVATION#"
            + getattr(invocation, "activation_id")
            + "#RECOVERY_ACTION#TERMINAL_V2#"
            + getattr(invocation, "generation_text")
        ),
        "action_identity_sha256": canonical_sha256(
            {"operation_kind": TERMINAL_V2, "created_at": created_at}
        ),
        "handoff": None if evidence is None else evidence["handoff"],
        "binding": None if evidence is None else evidence["binding"],
        "final_sky_state": (
            "NOT_LAUNCHED"
            if evidence is None
            else evidence["final_sky_state"]
        ),
        "final_ec2_states": (
            [] if evidence is None else list(evidence["final_ec2_states"])
        ),
        "request_cardinality": (
            "NOT_APPLICABLE"
            if evidence is None
            else evidence["request_cardinality"]
        ),
        "worker_cardinality": (
            "ZERO"
            if not allocations
            else ("ONE" if len(allocations) == 1 else "MULTIPLE")
        ),
        "allocations": allocations,
        "allocations_array_sha256": canonical_sha256(allocations),
        "worker_launch_evidence": launch_evidence,
        "worker_launch_evidence_array_sha256": canonical_sha256(
            launch_evidence
        ),
        "worker_launch_liabilities": liability_evidence,
        "worker_launch_liabilities_array_sha256": canonical_sha256(
            liability_evidence
        ),
        "spend_ledger_head_identity": {
            "identity": proof.final_spend_ledger_head_identity_sha256
        },
        "remaining_approved_gpu_seconds": (
            second_scan.spend.remaining_gpu_seconds
        ),
        "remaining_approved_gpu_usd": (
            second_scan.spend.remaining_gpu_cost_usd
        ),
        "final_heartbeat_identity": (
            None if evidence is None else evidence["final_heartbeat_identity"]
        ),
        "checkpoint_identity": (
            None if evidence is None else evidence["checkpoint_identity"]
        ),
        "cache_identity": (
            None if evidence is None else evidence["cache_identity"]
        ),
        "training_identity": (
            None if evidence is None else evidence["training_identity"]
        ),
        "evaluation_identity": (
            None if evidence is None else evidence["evaluation_identity"]
        ),
        "drain_identity": (
            None if evidence is None else evidence["drain_identity"]
        ),
        "terminal_observation_window": {
            "first": proof.first_scan_identity_sha256,
            "second": proof.second_scan_identity_sha256,
            "identity": proof.canonical_identity_sha256,
        },
        "request_evidence": request_evidence,
        "request_evidence_array_sha256": canonical_sha256(request_evidence),
        "post_terminal_quiescence_evidence": (
            None
            if evidence is None
            else evidence["post_terminal_quiescence_evidence"]
        ),
        "prior_terminal_v1_identity": (
            None
            if evidence is None
            else evidence["prior_terminal_v1_identity"]
        ),
        "outcome": (
            "ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH"
            if evidence is None
            else evidence["outcome"]
        ),
        "operator_disposition_required": (
            True
            if evidence is None
            else evidence["operator_disposition_required"]
        ),
        "writer_function_version_arn": getattr(
            invocation, "invoked_function_version_arn", None
        ),
        "writer_dispatch_identity_sha256": getattr(
            invocation, "dispatch_identity_sha256"
        ),
        "writer_invocation_nonce_sha256": recovery[
            "owner_invocation_nonce_sha256"
        ],
        "created_at": created_at,
    }
    body["canonical_body_sha256"] = canonical_sha256(body)
    return validate_record("glm52_production_terminal_v2", body)


def _terminal_request(
    *, ports: object, invocation: object, sources: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    from .task12_runtime import prove_runtime_terminal
    from .task12_writers import build_retained_writer_candidate

    state = _live_state(
        ports=ports, invocation=invocation, live_sources=sources
    )
    first, second = _two_scans(
        ports=ports,
        invocation=invocation,
        sources=sources,
        require_execution_terminal=True,
    )
    proof = prove_runtime_terminal(
        first=first,
        second=second,
        minimum_quiet_seconds=QUIET_SECONDS,
    )
    families = _families(
        ports=ports,
        invocation=invocation,
        campaign_identity_sha256=str(
            state["recovery_control"]["campaign_identity_sha256"]
        ),
    )
    terminal_evidence = (
        _read_terminal_evidence(
            ports=ports,
            invocation=invocation,
            recovery=state["recovery_control"],
            families=families,
        )
        if (
            families["glm52_production_worker_launch"]
            or families["glm52_production_worker_launch_liability"]
        )
        else None
    )
    record = _terminal_base(
        invocation=invocation,
        recovery=state["recovery_control"],
        proof=proof,
        second_scan=second,
        families=families,
        terminal_evidence=terminal_evidence,
    )
    candidate = build_retained_writer_candidate(
        writer_kind="TerminalV2",
        campaign_bucket=str(_roles(ports)["campaign_bucket"]),
        activation_id=getattr(invocation, "activation_id"),
        generation=getattr(invocation, "generation"),
        authority_domain="RECOVERY",
        record=record,
    )
    action, audit = _authority_pair(
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        action_kind="TERMINAL_V2",
        recovery=state["recovery_control"],
        observed_at=record["created_at"],
    )
    return {
        "writer_kind": "TerminalV2",
        "authority_domain": "RECOVERY",
        "record": record,
        "action": action,
        "audit": audit,
    }


def _zero_request(
    *, ports: object, invocation: object, sources: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    from .task12_live_drain_terminal import _json_value

    first, second = _two_scans(
        ports=ports,
        invocation=invocation,
        sources=sources,
        require_execution_terminal=True,
    )
    return {
        "first_runtime_scan": _json_value(first),
        "second_runtime_scan": _json_value(second),
        "minimum_quiet_seconds": QUIET_SECONDS,
    }


def materialize_live_request(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> Mapping[str, object] | None:
    """Build one live request after validating its complete producer set."""

    if operation_kind not in SUPPORTED_OPERATIONS:
        return None
    sources = _validate_sources(operation_kind, invocation, live_sources)
    if operation_kind == QUIESCE:
        return _quiesce_request(
            ports=ports, invocation=invocation, sources=sources
        )
    if operation_kind == TRANSFER:
        return _transfer_request(
            ports=ports, invocation=invocation, sources=sources
        )
    if operation_kind == WORKER_DRAIN:
        return _worker_request(
            ports=ports, invocation=invocation, sources=sources
        )
    if operation_kind == TERMINAL_V2:
        return _terminal_request(
            ports=ports, invocation=invocation, sources=sources
        )
    return _zero_request(
        ports=ports, invocation=invocation, sources=sources
    )


def persist_live_successors(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    request: Mapping[str, object],
    domain_result: object,
    ports: object,
) -> bool:
    """Persist the one successor not committed by the operation itself.

    Liability acquisition commits during materialization and worker drain
    commits through SSM in the domain dispatcher.  TerminalV2 is different:
    only after the conditional S3 writer and its immutable version-control row
    are both proven may this function atomically advance the exact recovery
    control from ``OWNED`` to ``TERMINAL_V2_PUBLISHED``.
    """

    if operation_kind not in SUPPORTED_OPERATIONS:
        return False
    if operation_kind == TERMINAL_V2:
        _persist_terminal_publication(
            invocation=invocation,
            live_sources=live_sources,
            request=request,
            domain_result=domain_result,
            ports=ports,
        )
    return True


def _persist_terminal_publication(
    *,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    request: Mapping[str, object],
    domain_result: object,
    ports: object,
) -> None:
    """Bind the proven TerminalV2 S3 version into retained recovery state."""

    from .task12_live_drain_terminal import _read_exact_record
    from .task12_retained_state import build_recovery_progress_plan
    from .task12_writers import (
        RetainedWriteResult,
        build_retained_writer_candidate,
        build_versioned_writer_control,
    )

    if type(request) is not dict or set(request) != {
        "writer_kind",
        "authority_domain",
        "record",
        "action",
        "audit",
    }:
        _fail("TerminalV2 publication request is not exact")
    if (
        request["writer_kind"] != "TerminalV2"
        or request["authority_domain"] != "RECOVERY"
        or type(domain_result) is not RetainedWriteResult
        or domain_result.writer_kind != "TerminalV2"
        or domain_result.response_authenticated is not True
    ):
        _fail("TerminalV2 writer result is not authenticated")
    try:
        terminal = validate_record(
            "glm52_production_terminal_v2", request["record"]
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveDrainEffectsError(
            "TerminalV2 publication record is invalid"
        ) from exc
    terminal_identity = canonical_record_identity(
        "glm52_production_terminal_v2", terminal
    )
    candidate = build_retained_writer_candidate(
        writer_kind="TerminalV2",
        campaign_bucket=str(_roles(ports)["campaign_bucket"]),
        activation_id=getattr(invocation, "activation_id"),
        generation=getattr(invocation, "generation"),
        authority_domain="RECOVERY",
        record=terminal,
    )
    state = _live_state(
        ports=ports, invocation=invocation, live_sources=live_sources
    )
    index = state["activation_index"]
    control = state["control"]
    recovery = state["recovery_control"]
    if control["phase"] != "RECOVERY_SEALING" or recovery["state"] != "OWNED":
        _fail("TerminalV2 recovery publication edge is not ready")
    version_control = _read_exact_record(
        ports=ports,
        activation_id=getattr(invocation, "activation_id"),
        record_type="glm52_task12_versioned_writer_control_v1",
        generation=getattr(invocation, "generation"),
        writer_kind="TerminalV2",
    )
    expected_version_control = build_versioned_writer_control(
        candidate=candidate,
        result=domain_result,
        published_at=version_control["published_at"],
    )
    if (
        version_control != expected_version_control
        or version_control["body_sha256"] != terminal_identity
    ):
        _fail("TerminalV2 version-control binding drifted")
    if terminal["allocations"]:
        from .campaign_drained_publication import (
            CampaignDrainedPublicationServices,
            publish_campaign_drained,
        )

        publish_campaign_drained(
            activation_id=getattr(invocation, "activation_id"),
            activation_ordinal=getattr(invocation, "activation_ordinal"),
            generation=getattr(invocation, "generation"),
            generation_text=getattr(invocation, "generation_text"),
            terminal_v2=terminal,
            task10_worker_descriptor_coordinate=_roles(ports).get(
                "task10_worker_descriptor_coordinate"
            ),
            services=CampaignDrainedPublicationServices(
                s3=ports.client("s3"),
                total_max_attempts=1,
            ),
        )
    capsule, raw_nonce = _owner_capsule(
        ports=ports, invocation=invocation, recovery=recovery
    )
    try:
        if capsule["nonce_sha256"] != recovery[
            "owner_invocation_nonce_sha256"
        ]:
            _fail("TerminalV2 recovery owner capsule drifted")
        observed_at = str(version_control["published_at"])
        recovery_after = dict(recovery)
        recovery_after.update(
            state="TERMINAL_V2_PUBLISHED",
            terminal_v2_identity_sha256=terminal_identity,
            revision=int(recovery["revision"]) + 1,
            updated_at=observed_at,
        )
        plan = build_recovery_progress_plan(
            index=ExactCheck(
                LedgerKey(
                    RUN_ID,
                    ledger_sk("glm52_production_activation_index"),
                ),
                index,
            ),
            control=ExactCheck(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_control",
                        activation_id=getattr(invocation, "activation_id"),
                    ),
                ),
                control,
            ),
            recovery_control=ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_recovery_control",
                        activation_id=getattr(invocation, "activation_id"),
                    ),
                ),
                recovery,
                recovery_after,
            ),
        )
        adapter = DynamoLedgerAdapter(
            client=ports.client("dynamodb"),
            table_name=str(_roles(ports)["ledger_table_name"]),
        )
        adapter.commit_recovery_progress(
            plan=plan,
            domain="RECOVERY",
            operation_identity_sha256=canonical_sha256(
                {
                    "operation_kind": TERMINAL_V2,
                    "version_control": version_control[
                        "canonical_body_sha256"
                    ],
                    "recovery_before": canonical_record_identity(
                        "glm52_production_recovery_control", recovery
                    ),
                    "recovery_after": canonical_record_identity(
                        "glm52_production_recovery_control", recovery_after
                    ),
                }
            ),
            raw_owner_nonce=raw_nonce,
        )
    finally:
        raw_nonce = b""


__all__ = [
    "LiabilityTransferProof",
    "SUPPORTED_OPERATIONS",
    "Task12LiveDrainEffectsError",
    "materialize_live_request",
    "persist_live_successors",
    "validate_liability_transfer_proof",
]
