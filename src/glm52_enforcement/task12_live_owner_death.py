"""Production owner-death classification for the retained Task 12 route.

The classifier is deliberately no-launch.  It consumes only the exact current
ledger rows selected by the immutable operation descriptor, proves the
activation-scoped support execution terminal, and prepares one conditional
DynamoDB transition owned by the retained RECOVERY nonce.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Mapping, Union

from .canonical import canonical_sha256
from .dynamodb import ExactCheck, ExactUpdate, LedgerKey
from .records import (
    canonical_record_identity,
    ledger_sk,
    validate_record,
)
from .transitions import validate_transition


RUN_ID = "glm52-sky-20260724"
SUPPORTED_OPERATIONS = frozenset(
    {"RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION"}
)
_TERMINAL_EXECUTION_STATES = frozenset(
    {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
)
_CLASSIFIABLE_STATES = frozenset(
    {"ARMED", "CONSUMED", "POST_STARTED", "POST_AUTHORIZED"}
)
_SOURCE_TYPES = {
    "activation_index": "glm52_production_activation_index",
    "control": "glm52_production_control",
    "execution": "glm52_production_execution",
    "recovery_control": "glm52_production_recovery_control",
    "sky_action": "glm52_production_action",
}


class Task12OwnerDeathError(ValueError):
    """Owner-death sources or transition authority are not exact."""


@dataclass(frozen=True)
class OwnerDeathClassificationPlan:
    index: ExactCheck
    control: Union[ExactCheck, ExactUpdate]
    recovery_control: ExactCheck
    execution: ExactCheck
    action: Union[ExactCheck, ExactUpdate]

    @property
    def transaction_plans(self) -> tuple[object, ...]:
        return (
            self.index,
            self.control,
            self.recovery_control,
            self.execution,
            self.action,
        )


def _fail(message: str) -> None:
    raise Task12OwnerDeathError(message)


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        _fail(label + " is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Task12OwnerDeathError(
            label + " is not canonical UTC"
        ) from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.microsecond != 0
    ):
        _fail(label + " is not normalized UTC")
    return parsed.astimezone(timezone.utc)


def _now(ports: object) -> str:
    boundary = getattr(ports, "now_utc", None)
    now = boundary() if callable(boundary) else datetime.now(timezone.utc)
    if (
        type(now) is not datetime
        or now.tzinfo is None
        or now.utcoffset() is None
        or now.microsecond != 0
    ):
        _fail("owner-death runtime clock is not normalized")
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validated_sources(
    *,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if type(live_sources) is not dict or set(live_sources) != set(
        _SOURCE_TYPES
    ):
        _fail("owner-death live source set drifted")
    activation_id = getattr(invocation, "activation_id", None)
    activation_ordinal = getattr(invocation, "activation_ordinal", None)
    generation = getattr(invocation, "generation", None)
    generation_text = getattr(invocation, "generation_text", None)
    if (
        getattr(invocation, "operation_kind", None)
        != "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION"
        or type(activation_id) is not str
        or not activation_id
        or type(activation_ordinal) is not int
        or activation_ordinal < 1
        or type(generation) is not int
        or generation < 1
        or generation_text != f"{generation:08d}"
    ):
        _fail("owner-death invocation identity drifted")
    exact: dict[str, dict[str, object]] = {}
    campaign_identity: object = None
    for alias, record_type in _SOURCE_TYPES.items():
        try:
            record = validate_record(record_type, live_sources[alias])
        except (TypeError, ValueError) as exc:
            raise Task12OwnerDeathError(
                alias + " owner-death source drifted"
            ) from exc
        if record_type == "glm52_production_activation_index":
            if (
                record.get("current_activation_id") != activation_id
                or record.get("current_activation_ordinal")
                != activation_ordinal
            ):
                _fail("owner-death activation index drifted")
        elif (
            record.get("activation_id") != activation_id
            or record.get("activation_ordinal") != activation_ordinal
        ):
            _fail(alias + " owner-death activation drifted")
        if alias == "activation_index":
            campaign_identity = record.get("campaign_identity_sha256")
        elif record.get("campaign_identity_sha256") != campaign_identity:
            _fail("owner-death campaign identity forked")
        exact[alias] = record
    action = exact["sky_action"]
    if (
        action.get("generation") != generation
        or action.get("generation_text") != generation_text
        or action.get("action_kind") != "SKY_POST"
    ):
        _fail("owner-death Sky action generation drifted")
    return exact


def _continuation_capsule(invocation: object) -> dict[str, object]:
    operation_input = getattr(invocation, "operation_input", None)
    prior = (
        operation_input.get("task12_last_result")
        if isinstance(operation_input, Mapping)
        else None
    )
    result = prior.get("result") if isinstance(prior, Mapping) else None
    capsule = (
        result.get("owner_nonce_capsule")
        if isinstance(result, Mapping)
        else None
    )
    if type(capsule) is not dict:
        _fail("recovery owner nonce continuation is absent")
    body = dict(capsule)
    identity = body.pop("canonical_body_sha256", None)
    if identity != canonical_sha256(body):
        _fail("recovery owner nonce capsule identity drifted")
    return dict(capsule)


def build_owner_death_request(
    *,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    observed_at: str,
) -> dict[str, object]:
    """Build the exact conditional owner-death plan from authenticated rows."""

    sources = _validated_sources(
        invocation=invocation, live_sources=live_sources
    )
    observed = _utc(observed_at, "owner-death observed_at")
    index = sources["activation_index"]
    control = sources["control"]
    recovery = sources["recovery_control"]
    execution = sources["execution"]
    action = sources["sky_action"]
    action_key = ledger_sk(
        "glm52_production_action",
        activation_id=action["activation_id"],
        generation=action["generation"],
        action_kind=action["action_kind"],
        attempt=action["attempt"],
    )
    recovery_expiry = _utc(
        recovery.get("owner_hard_expires_at"),
        "recovery owner hard expiry",
    )
    if (
        control.get("phase") != "RECOVERY_SEALING"
        or control.get("active_epoch") != execution.get("epoch")
        or control.get("active_execution_arn")
        != execution.get("expected_execution_arn")
        or control.get("active_state_machine_version_arn")
        != execution.get("expected_state_machine_version_arn")
        or control.get("last_sky_post_action_key") != action_key
        or control.get("last_sky_post_generation") != action.get("generation")
        or control.get("last_sky_post_state") != action.get("state")
        or recovery.get("state") != "OWNED"
        or recovery.get("owner_execution_arn")
        != getattr(invocation, "state_machine_execution_arn", None)
        or recovery.get("owner_state_machine_version_arn")
        != getattr(invocation, "caller_state_machine_version_arn", None)
        or recovery_expiry <= observed
        or execution.get("state") not in _TERMINAL_EXECUTION_STATES
        or execution.get("terminal_status") != execution.get("state")
        or action.get("owner_epoch") != execution.get("epoch")
        or action.get("owner_execution_arn")
        != execution.get("expected_execution_arn")
        or action.get("armed_by_epoch") != execution.get("epoch")
        or action.get("armed_by_execution_arn")
        != execution.get("expected_execution_arn")
        or action.get("armed_by_state_machine_version_arn")
        != execution.get("expected_state_machine_version_arn")
        or action.get("barrier_nonce_sha256")
        != control.get("barrier_nonce_sha256")
    ):
        _fail("owner-death authority tuple drifted")

    source_state = action["state"]
    index_key = LedgerKey(RUN_ID, ledger_sk("glm52_production_activation_index"))
    control_key = LedgerKey(
        RUN_ID,
        ledger_sk(
            "glm52_production_control",
            activation_id=action["activation_id"],
        ),
    )
    recovery_key = LedgerKey(
        RUN_ID,
        ledger_sk(
            "glm52_production_recovery_control",
            activation_id=action["activation_id"],
        ),
    )
    execution_key = LedgerKey(
        RUN_ID,
        ledger_sk(
            "glm52_production_execution",
            activation_id=action["activation_id"],
            epoch=execution["epoch"],
        ),
    )
    action_ledger_key = LedgerKey(RUN_ID, action_key)

    if source_state == "POST_CLASSIFIED":
        operation_identity = canonical_sha256(
            {
                "schema_version": 1,
                "operation_kind": (
                    "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION"
                ),
                "classification": "ADOPTED_POST_CLASSIFIED",
                "index_identity_sha256": canonical_record_identity(
                    "glm52_production_activation_index", index
                ),
                "control_identity_sha256": canonical_record_identity(
                    "glm52_production_control", control
                ),
                "recovery_identity_sha256": canonical_record_identity(
                    "glm52_production_recovery_control", recovery
                ),
                "execution_identity_sha256": canonical_record_identity(
                    "glm52_production_execution", execution
                ),
                "action_identity_sha256": canonical_record_identity(
                    "glm52_production_action", action
                ),
            }
        )
        plan = OwnerDeathClassificationPlan(
            index=ExactCheck(index_key, index),
            control=ExactCheck(control_key, control),
            recovery_control=ExactCheck(recovery_key, recovery),
            execution=ExactCheck(execution_key, execution),
            action=ExactCheck(action_ledger_key, action),
        )
        return {
            "plan": asdict(plan),
            "domain": "RECOVERY",
            "operation_identity_sha256": operation_identity,
            "owner_nonce_capsule": _continuation_capsule(invocation),
        }

    if source_state not in _CLASSIFIABLE_STATES:
        _fail("Sky action state has no owner-death edge")
    hard_expiry_field = (
        "post_owner_hard_expires_at"
        if source_state in {"POST_STARTED", "POST_AUTHORIZED"}
        else "arming_hard_expires_at"
    )
    hard_expiry = _utc(
        action.get(hard_expiry_field),
        "dead action hard expiry",
    )
    if hard_expiry >= observed:
        _fail("Sky action owner is not yet proved dead")

    operation_preimage = {
        "schema_version": 1,
        "operation_kind": "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION",
        "observed_at": observed_at,
        "source_state": source_state,
        "action_key": action_key,
        "action_revision": action["revision"],
        "index_identity_sha256": canonical_record_identity(
            "glm52_production_activation_index", index
        ),
        "control_identity_sha256": canonical_record_identity(
            "glm52_production_control", control
        ),
        "recovery_identity_sha256": canonical_record_identity(
            "glm52_production_recovery_control", recovery
        ),
        "execution_identity_sha256": canonical_record_identity(
            "glm52_production_execution", execution
        ),
        "action_identity_sha256": canonical_record_identity(
            "glm52_production_action", action
        ),
    }
    operation_identity = canonical_sha256(operation_preimage)
    proof = {
        **operation_preimage,
        "record_type": "glm52_task12_owner_death_proof_v1",
        "run_id": action["run_id"],
        "campaign_identity_sha256": action[
            "campaign_identity_sha256"
        ],
        "activation_id": action["activation_id"],
        "activation_ordinal": action["activation_ordinal"],
        "generation": action["generation"],
        "generation_text": action["generation_text"],
        "action_hard_expiry_field": hard_expiry_field,
        "action_hard_expires_at": action[hard_expiry_field],
        "action_owner_epoch": action["owner_epoch"],
        "action_owner_execution_arn": action["owner_execution_arn"],
        "action_post_owner_dispatch_identity_sha256": action[
            "post_owner_dispatch_identity_sha256"
        ],
        "action_post_authorized_at": action["post_authorized_at"],
        "recovery_owner_execution_arn": recovery["owner_execution_arn"],
        "recovery_owner_state_machine_version_arn": recovery[
            "owner_state_machine_version_arn"
        ],
        "recovery_owner_attempt": recovery["owner_attempt"],
        "recovery_owner_hard_expires_at": recovery[
            "owner_hard_expires_at"
        ],
        "recovery_barrier_nonce_sha256": recovery[
            "recovery_barrier_nonce_sha256"
        ],
        "recovery_revision": recovery["revision"],
        "terminal_execution_arn": execution["expected_execution_arn"],
        "terminal_execution_version_arn": execution[
            "expected_state_machine_version_arn"
        ],
        "terminal_execution_state": execution["state"],
        "terminal_execution_revision": execution["revision"],
        "operation_identity_sha256": operation_identity,
    }
    proof_identity = canonical_sha256(proof)
    after = dict(action)
    if source_state == "ARMED":
        target_state = "ABANDONED"
        after.update(
            state=target_state,
            abandoned_at=observed_at,
            abandonment_proof_sha256=proof_identity,
            revision=action["revision"] + 1,
        )
    else:
        target_state = "POST_CLASSIFIED"
        after.update(
            state=target_state,
            completed_at=observed_at,
            outcome_class=(
                "AMBIGUOUS_OWNER_DIED"
                if source_state == "POST_AUTHORIZED"
                else "PROVED_NOT_SENT_OWNER_DIED"
            ),
            classification_evidence_kind="OWNER_DEATH_PROOF",
            classification_evidence_body_sha256=proof_identity,
            sky_request_id=None,
            response_identity_sha256=None,
            revision=action["revision"] + 1,
        )
    try:
        after = validate_transition(
            "glm52_production_action", action, after
        )
    except (TypeError, ValueError) as exc:
        raise Task12OwnerDeathError(
            "owner-death action transition is not exact"
        ) from exc
    control_after = dict(control)
    control_after.update(
        last_sky_post_state=target_state,
        revision=control["revision"] + 1,
        updated_at=observed_at,
    )
    try:
        control_after = validate_record(
            "glm52_production_control", control_after
        )
    except (TypeError, ValueError) as exc:
        raise Task12OwnerDeathError(
            "owner-death control transition is not exact"
        ) from exc
    plan = OwnerDeathClassificationPlan(
        index=ExactCheck(index_key, index),
        control=ExactUpdate(control_key, control, control_after),
        recovery_control=ExactCheck(recovery_key, recovery),
        execution=ExactCheck(execution_key, execution),
        action=ExactUpdate(action_ledger_key, action, after),
    )
    return {
        "plan": asdict(plan),
        "domain": "RECOVERY",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": _continuation_capsule(invocation),
    }


def materialize_live_request(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> Mapping[str, object] | None:
    if operation_kind not in SUPPORTED_OPERATIONS:
        return None
    return build_owner_death_request(
        invocation=invocation,
        live_sources=live_sources,
        observed_at=_now(ports),
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
    if operation_kind not in SUPPORTED_OPERATIONS:
        return False
    # The conditional transaction is the successor; the adapter reconciles its
    # exact multi-record readback before this acknowledgment is reached.
    del invocation, live_sources, request, domain_result, ports
    return True


__all__ = [
    "OwnerDeathClassificationPlan",
    "SUPPORTED_OPERATIONS",
    "Task12OwnerDeathError",
    "build_owner_death_request",
    "materialize_live_request",
    "persist_live_successors",
]
