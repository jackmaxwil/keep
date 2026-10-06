"""Typed, fail-closed transaction plans for Task 12 retained state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple, Union

from .dynamodb import ExactCheck, ExactPut, ExactUpdate
from .records import canonical_record_identity, validate_record
from .transitions import validate_owner_takeover, validate_transition


_ExactControl = Union[ExactCheck, ExactUpdate]
_ExactRetained = Union[ExactCheck, ExactUpdate]
_IDENTITY_FIELDS = (
    "run_id",
    "campaign_identity_sha256",
    "activation_id",
    "activation_ordinal",
    "rollover_identity_sha256",
)
_TERMINAL_EXECUTION_STATES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "TIMED_OUT",
        "ABORTED",
        "START_POSSIBLY_SENT_UNRESOLVED_INCIDENT",
        "ABANDONED_PROVED_NOT_STARTED",
    }
)
_TERMINAL_CLEANUP_STATES = frozenset(
    {"DELETED", "ALREADY_ABSENT", "CLEANUP_INCIDENT"}
)


@dataclass(frozen=True)
class RecoverySealPlan:
    index: ExactCheck
    support_execution: ExactCheck
    control: ExactUpdate
    recovery_control: ExactUpdate

    @property
    def transaction_plans(self) -> Tuple[object, ...]:
        return (
            self.index,
            self.support_execution,
            self.control,
            self.recovery_control,
        )


@dataclass(frozen=True)
class RecoveryProgressPlan:
    index: ExactCheck
    control: _ExactControl
    recovery_control: ExactUpdate

    @property
    def transaction_plans(self) -> Tuple[object, ...]:
        return (self.index, self.control, self.recovery_control)


@dataclass(frozen=True)
class TeardownSealPlan:
    index: ExactCheck
    recovery_control: ExactCheck
    control: ExactUpdate
    finalization_control: _ExactRetained

    @property
    def transaction_plans(self) -> Tuple[object, ...]:
        return (
            self.index,
            self.recovery_control,
            self.control,
            self.finalization_control,
        )


@dataclass(frozen=True)
class FinalizationProgressPlan:
    index: ExactCheck
    control: ExactCheck
    finalization_control: ExactUpdate

    @property
    def transaction_plans(self) -> Tuple[object, ...]:
        return (self.index, self.control, self.finalization_control)


@dataclass(frozen=True)
class RetainedOwnerTakeoverPlan:
    index: ExactCheck
    control: ExactCheck
    prior_owner_execution: ExactCheck
    action_guards: Tuple[ExactCheck, ...]
    owner: ExactUpdate

    @property
    def transaction_plans(self) -> Tuple[object, ...]:
        return (
            self.index,
            self.control,
            self.prior_owner_execution,
            *self.action_guards,
            self.owner,
        )


@dataclass(frozen=True)
class SnapshotCleanupTransitionPlan:
    index: ExactCheck
    cleanup_chain: ExactUpdate
    authority: ExactCheck
    cleanup_control: ExactUpdate
    cleanup_transition: ExactPut
    cleanup_action: Union[ExactPut, ExactUpdate, None] = None

    @property
    def transaction_plans(self) -> Tuple[object, ...]:
        action = (
            ()
            if self.cleanup_action is None
            else (self.cleanup_action,)
        )
        return (
            self.index,
            self.cleanup_chain,
            self.authority,
            self.cleanup_control,
            *action,
            self.cleanup_transition,
        )


@dataclass(frozen=True)
class SnapshotDeleteActionPlan:
    """Action-only snapshot transaction; retained control is an exact guard."""

    index: ExactCheck
    control: ExactCheck
    action: Union[ExactPut, ExactUpdate]

    @property
    def transaction_plans(self) -> Tuple[object, ...]:
        return (self.index, self.control, self.action)


@dataclass(frozen=True)
class SnapshotCleanupOwnerTakeoverPlan:
    """Expired-owner replacement guarded by sealed production authority."""

    index: ExactCheck
    authority: ExactCheck
    cleanup_control: ExactUpdate

    @property
    def transaction_plans(self) -> Tuple[object, ...]:
        return (self.index, self.authority, self.cleanup_control)


@dataclass(frozen=True)
class SnapshotCleanupReconcileReadPlan:
    """Read-only recovery authority for an already-reconciling attempt."""

    index: ExactCheck
    authority: ExactCheck
    cleanup_control: ExactCheck
    cleanup_action: ExactCheck


def _record(plan: object) -> Mapping[str, object]:
    if type(plan) is ExactCheck:
        return plan.expected
    if type(plan) is ExactUpdate:
        return plan.after
    if type(plan) is ExactPut:
        return plan.item
    raise TypeError("retained transaction member has the wrong exact type")


def _require_type(plan: object, expected: object, label: str) -> None:
    if type(plan) is not expected:
        raise TypeError(label + " has the wrong exact plan type")


def _require_family(plan: object, record_type: str, label: str) -> None:
    if _record(plan).get("record_type") != record_type:
        raise ValueError(label + " has the wrong record family")


def _require_same_activation(*records: Mapping[str, object]) -> None:
    reference = records[0]
    for record in records[1:]:
        for field in _IDENTITY_FIELDS:
            if (
                field in reference
                and field in record
                and record[field] != reference[field]
            ):
                raise ValueError("retained activation identity mismatch")


def _validate_index(
    index: ExactCheck, retained: Mapping[str, object]
) -> Mapping[str, object]:
    _require_type(index, ExactCheck, "activation index")
    _require_family(
        index, "glm52_production_activation_index", "activation index"
    )
    value = validate_record(
        "glm52_production_activation_index", index.expected
    )
    if (
        value["run_id"] != retained["run_id"]
        or value["campaign_identity_sha256"]
        != retained["campaign_identity_sha256"]
        or value["current_activation_ordinal"]
        < retained["activation_ordinal"]
        or (
            value["current_activation_ordinal"]
            == retained["activation_ordinal"]
            and value["current_activation_id"] != retained["activation_id"]
        )
    ):
        raise ValueError("activation index does not authorize retained state")
    return value


def _validate_control_plan(
    control: _ExactControl, *, expected_after_phase: str
) -> Mapping[str, object]:
    if type(control) is ExactCheck:
        _require_family(control, "glm52_production_control", "control")
        value = validate_record(
            "glm52_production_control", control.expected
        )
    elif type(control) is ExactUpdate:
        _require_family(control, "glm52_production_control", "control")
        value = validate_transition(
            "glm52_production_control", control.before, control.after
        )
    else:
        raise TypeError("control has the wrong exact plan type")
    if value["phase"] != expected_after_phase:
        raise ValueError("control phase does not match retained operation")
    return value


def build_recovery_seal_plan(
    *,
    index: ExactCheck,
    support_execution: ExactCheck,
    control: ExactUpdate,
    recovery_control: ExactUpdate,
) -> RecoverySealPlan:
    """Bind OPEN->RECOVERY_SEALING and DORMANT->OWNED in one plan."""
    _require_type(support_execution, ExactCheck, "support execution")
    _require_family(
        support_execution,
        "glm52_production_execution",
        "support execution",
    )
    execution = validate_record(
        "glm52_production_execution", support_execution.expected
    )
    if execution["state"] not in _TERMINAL_EXECUTION_STATES:
        raise ValueError("recovery seal requires terminal support execution")
    _require_type(control, ExactUpdate, "control")
    control_after = _validate_control_plan(
        control, expected_after_phase="RECOVERY_SEALING"
    )
    if control.before.get("phase") != "OPEN":
        raise ValueError("recovery cannot begin after teardown")
    _require_type(recovery_control, ExactUpdate, "recovery control")
    _require_family(
        recovery_control,
        "glm52_production_recovery_control",
        "recovery control",
    )
    recovery_after = validate_transition(
        "glm52_production_recovery_control",
        recovery_control.before,
        recovery_control.after,
    )
    if (
        recovery_control.before["state"] != "DORMANT"
        or recovery_after["state"] != "OWNED"
        or recovery_after["support_control_revision_at_seal"]
        != control_after["revision"]
        or recovery_after["support_execution_identity_sha256"]
        != canonical_record_identity(
            "glm52_production_execution", execution
        )
        or recovery_after["owner_execution_arn"]
        == execution["expected_execution_arn"]
        or recovery_after["owner_state_machine_version_arn"]
        == execution["expected_state_machine_version_arn"]
    ):
        raise ValueError("recovery seal evidence binding mismatch")
    _require_same_activation(
        control_after, execution, recovery_after
    )
    _validate_index(index, recovery_after)
    return RecoverySealPlan(
        index, support_execution, control, recovery_control
    )


def build_recovery_progress_plan(
    *,
    index: ExactCheck,
    control: _ExactControl,
    recovery_control: ExactUpdate,
) -> RecoveryProgressPlan:
    """Build the terminal-v2 or recovery-complete retained transition."""
    _require_type(recovery_control, ExactUpdate, "recovery control")
    _require_family(
        recovery_control,
        "glm52_production_recovery_control",
        "recovery control",
    )
    after = validate_transition(
        "glm52_production_recovery_control",
        recovery_control.before,
        recovery_control.after,
    )
    if after["state"] == "TERMINAL_V2_PUBLISHED":
        control_value = _validate_control_plan(
            control, expected_after_phase="RECOVERY_SEALING"
        )
    elif after["state"] == "RECOVERY_COMPLETE":
        if type(control) is not ExactUpdate:
            raise TypeError(
                "recovery completion requires an exact control update"
            )
        control_value = _validate_control_plan(
            control, expected_after_phase="RECOVERY_COMPLETE"
        )
        if control.before.get("phase") != "RECOVERY_SEALING":
            raise ValueError("recovery completion control edge mismatch")
    else:
        raise ValueError("unsupported recovery progress target")
    _require_same_activation(control_value, after)
    _validate_index(index, after)
    return RecoveryProgressPlan(index, control, recovery_control)


def build_teardown_seal_plan(
    *,
    index: ExactCheck,
    recovery_control: ExactCheck,
    control: ExactUpdate,
    finalization_control: _ExactRetained,
) -> TeardownSealPlan:
    """Build either teardown-sealing or teardown-sealed acquisition."""
    _require_type(recovery_control, ExactCheck, "recovery control")
    _require_family(
        recovery_control,
        "glm52_production_recovery_control",
        "recovery control",
    )
    recovery = validate_record(
        "glm52_production_recovery_control",
        recovery_control.expected,
    )
    if recovery["state"] != "RECOVERY_COMPLETE":
        raise ValueError("teardown requires complete recovery")
    _require_type(control, ExactUpdate, "control")
    if control.after.get("phase") == "TEARDOWN_SEALING":
        control_value = _validate_control_plan(
            control, expected_after_phase="TEARDOWN_SEALING"
        )
        if (
            control.before.get("phase") != "RECOVERY_COMPLETE"
            or type(finalization_control) is not ExactCheck
        ):
            raise ValueError("teardown-sealing transaction shape mismatch")
        _require_family(
            finalization_control,
            "glm52_production_finalization_control",
            "finalization control",
        )
        finalization = validate_record(
            "glm52_production_finalization_control",
            finalization_control.expected,
        )
        if finalization["state"] != "DORMANT":
            raise ValueError("teardown sealing requires dormant finalization")
    elif control.after.get("phase") == "TEARDOWN_SEALED":
        control_value = _validate_control_plan(
            control, expected_after_phase="TEARDOWN_SEALED"
        )
        if (
            control.before.get("phase") != "TEARDOWN_SEALING"
            or type(finalization_control) is not ExactUpdate
        ):
            raise ValueError("teardown-sealed transaction shape mismatch")
        _require_family(
            finalization_control,
            "glm52_production_finalization_control",
            "finalization control",
        )
        finalization = validate_transition(
            "glm52_production_finalization_control",
            finalization_control.before,
            finalization_control.after,
        )
        if (
            finalization_control.before["state"] != "DORMANT"
            or finalization["state"] != "OWNED"
            or finalization["teardown_sealed_control_revision"]
            != control_value["revision"]
            or finalization["terminal_v2_identity_sha256"]
            != recovery["terminal_v2_identity_sha256"]
        ):
            raise ValueError("finalization ownership binding mismatch")
    else:
        raise ValueError("unsupported teardown control edge")
    _require_same_activation(control_value, recovery, finalization)
    _validate_index(index, recovery)
    return TeardownSealPlan(
        index, recovery_control, control, finalization_control
    )


def build_finalization_progress_plan(
    *,
    index: ExactCheck,
    control: ExactCheck,
    finalization_control: ExactUpdate,
) -> FinalizationProgressPlan:
    """Build one ordered finalization evidence transition."""
    _require_type(control, ExactCheck, "control")
    control_value = _validate_control_plan(
        control, expected_after_phase="TEARDOWN_SEALED"
    )
    _require_type(finalization_control, ExactUpdate, "finalization control")
    _require_family(
        finalization_control,
        "glm52_production_finalization_control",
        "finalization control",
    )
    finalization = validate_transition(
        "glm52_production_finalization_control",
        finalization_control.before,
        finalization_control.after,
    )
    if finalization["state"] not in {
        "SUPPORT_FINALIZED",
        "SNAPSHOT_DISPOSITION_RECORDED",
        "DRAINED_PUBLISHED",
    }:
        raise ValueError("unsupported finalization progress target")
    if (
        finalization["teardown_sealed_control_revision"]
        != control_value["revision"]
    ):
        raise ValueError("finalization lost teardown revision binding")
    _require_same_activation(control_value, finalization)
    _validate_index(index, finalization)
    return FinalizationProgressPlan(index, control, finalization_control)


def build_retained_owner_takeover_plan(
    *,
    index: ExactCheck,
    control: ExactCheck,
    prior_owner_execution: ExactCheck,
    action_guards: Tuple[ExactCheck, ...],
    owner: ExactUpdate,
) -> RetainedOwnerTakeoverPlan:
    """Build a state-preserving recovery/finalization/cleanup takeover."""
    _require_type(control, ExactCheck, "control")
    _require_type(
        prior_owner_execution, ExactCheck, "prior owner execution"
    )
    _require_family(
        prior_owner_execution,
        "glm52_production_execution",
        "prior owner execution",
    )
    prior_execution = validate_record(
        "glm52_production_execution",
        prior_owner_execution.expected,
    )
    if type(action_guards) is not tuple:
        raise TypeError("action guards must be an exact tuple")
    for guard in action_guards:
        _require_type(guard, ExactCheck, "action guard")
        state = _record(guard).get("state")
        if state not in {
            "COMPLETED",
            "AMBIGUOUS",
            "ABANDONED",
        }:
            raise ValueError("takeover action is not terminal or reconcilable")
    _require_type(owner, ExactUpdate, "owner")
    family = owner.after.get("record_type")
    allowed_phases = {
        "glm52_production_recovery_control": {
            "RECOVERY_SEALING",
        },
        "glm52_production_finalization_control": {"TEARDOWN_SEALED"},
        "glm52_production_snapshot_cleanup_control": {
            "TEARDOWN_SEALED",
        },
    }
    if family not in allowed_phases:
        raise ValueError("retained takeover family is forbidden")
    control_value = validate_record(
        "glm52_production_control", control.expected
    )
    if control_value["phase"] not in allowed_phases[family]:
        raise ValueError("control phase forbids retained takeover")
    validate_owner_takeover(family, owner.before, owner.after)
    if (
        prior_execution["expected_execution_arn"]
        != owner.before["owner_execution_arn"]
        or prior_execution["expected_state_machine_version_arn"]
        != owner.before["owner_state_machine_version_arn"]
        or (
            prior_execution["state"] not in _TERMINAL_EXECUTION_STATES
            and owner.before["owner_hard_expires_at"]
            >= owner.after["updated_at"]
        )
    ):
        raise ValueError("takeover lacks terminal or expired old-owner proof")
    _require_same_activation(
        control_value, prior_execution, owner.after
    )
    _validate_index(index, owner.after)
    return RetainedOwnerTakeoverPlan(
        index,
        control,
        prior_owner_execution,
        action_guards,
        owner,
    )


def build_snapshot_cleanup_transition_plan(
    *,
    index: ExactCheck,
    cleanup_chain: ExactUpdate,
    authority: ExactCheck,
    cleanup_control: ExactUpdate,
    cleanup_transition: ExactPut,
    cleanup_action: Union[ExactPut, ExactUpdate, None] = None,
) -> SnapshotCleanupTransitionPlan:
    """Bind every snapshot cleanup revision to an immutable chain record."""
    _require_type(cleanup_control, ExactUpdate, "cleanup control")
    _require_family(
        cleanup_control,
        "glm52_production_snapshot_cleanup_control",
        "cleanup control",
    )
    after = validate_transition(
        "glm52_production_snapshot_cleanup_control",
        cleanup_control.before,
        cleanup_control.after,
    )
    _validate_index(index, after)
    _require_type(cleanup_chain, ExactUpdate, "cleanup chain")
    _require_family(
        cleanup_chain,
        "glm52_production_recovery_control",
        "cleanup chain",
    )
    validate_record(
        "glm52_production_recovery_control", cleanup_chain.before
    )
    validate_record(
        "glm52_production_recovery_control", cleanup_chain.after
    )
    cleanup_chain_mutable = {
        "cleanup_transition_chain_head_sha256",
        "cleanup_transition_chain_length",
        "revision",
        "updated_at",
    }
    if (
        cleanup_chain.after["revision"]
        != cleanup_chain.before["revision"] + 1
        or any(
            cleanup_chain.before[field] != cleanup_chain.after[field]
            for field in cleanup_chain.before
            if field not in cleanup_chain_mutable
        )
    ):
        raise ValueError("cleanup chain update changed frozen recovery evidence")
    _require_type(authority, ExactCheck, "cleanup authority")
    authority_record = _record(authority)
    if authority_record.get("record_type") not in {
        "glm52_production_control",
        "glm52_production_finalization_control",
    }:
        raise ValueError("snapshot cleanup authority family is forbidden")
    validate_record(authority_record["record_type"], authority_record)
    _require_same_activation(authority_record, after)
    if after["state"] == "ARMED":
        if (
            authority_record["record_type"]
            != "glm52_production_finalization_control"
            or authority_record["state"]
            not in {"OWNED", "SUPPORT_FINALIZED"}
        ):
            raise ValueError("snapshot arm requires live finalization authority")
    elif (
        authority_record["record_type"]
        != "glm52_production_control"
        or authority_record["phase"] != "TEARDOWN_SEALED"
    ):
        raise ValueError("snapshot cleanup requires sealed control authority")
    _require_type(cleanup_transition, ExactPut, "cleanup transition")
    _require_family(
        cleanup_transition,
        "glm52_production_snapshot_cleanup_transition",
        "cleanup transition",
    )
    transition = validate_record(
        "glm52_production_snapshot_cleanup_transition",
        cleanup_transition.item,
    )
    if (
        transition["from_state"] != cleanup_control.before["state"]
        or transition["to_state"] != after["state"]
        or transition["from_revision"]
        != cleanup_control.before["revision"]
        or transition["to_revision"] != after["revision"]
        or transition["cleanup_control_root_identity_sha256"]
        != cleanup_chain.before["cleanup_control_root_identity_sha256"]
        or transition["prior_transition_sha256"]
        != cleanup_chain.before["cleanup_transition_chain_head_sha256"]
        or cleanup_chain.after["cleanup_transition_chain_length"]
        != cleanup_chain.before["cleanup_transition_chain_length"] + 1
        or cleanup_chain.after["cleanup_transition_chain_head_sha256"]
        != transition["canonical_body_sha256"]
    ):
        raise ValueError("snapshot cleanup transition chain mismatch")
    for field in (
        "owner_attempt",
        "owner_execution_arn",
        "owner_state_machine_version_arn",
        "owner_dispatch_identity_sha256",
        "owner_invocation_nonce_sha256",
    ):
        if transition[field] != after[field]:
            raise ValueError("snapshot cleanup transition owner mismatch")
    if (
        after["state"] not in {"ARMED"}
        and transition["transitioned_at"] < after["delete_not_before"]
    ):
        raise ValueError("snapshot cleanup cannot run before its deadline")
    if after["state"] in _TERMINAL_CLEANUP_STATES and (
        cleanup_control.before["state"] != "DELETE_RECONCILING"
        and not (
            cleanup_control.before["state"] == "OWNED"
            and after["state"] == "ALREADY_ABSENT"
        )
    ):
        raise ValueError("snapshot cleanup terminal edge lacks reconciliation")
    if cleanup_action is not None:
        if type(cleanup_action) not in (ExactPut, ExactUpdate):
            raise TypeError("cleanup action has the wrong exact plan type")
        _require_family(
            cleanup_action,
            "glm52_production_snapshot_cleanup_action",
            "cleanup action",
        )
        if type(cleanup_action) is ExactPut:
            action_after = validate_record(
                "glm52_production_snapshot_cleanup_action",
                cleanup_action.item,
            )
            if (
                after["state"] != "OWNED"
                or action_after["state"] != "ARMED"
                or action_after["attempt"]
                != after["delete_logical_attempt"] + 1
                or transition["action_identity_sha256"] is not None
            ):
                raise ValueError("snapshot delete action arm binding mismatch")
        else:
            action_after = validate_transition(
                "glm52_production_snapshot_cleanup_action",
                cleanup_action.before,
                cleanup_action.after,
            )
            if (
                after["state"] != "DELETE_POSSIBLY_SENT"
                or action_after["state"] != "CONSUMED"
                or transition["action_identity_sha256"]
                != after["latest_delete_action_identity_sha256"]
                or not after["delete_action_identities"]
                or after["delete_action_identities"][-1]
                != transition["action_identity_sha256"]
                or action_after["authority_audit_body_sha256"]
                != transition["authority_audit_identity_sha256"]
                or action_after["authorized_transition_from_revision"]
                != transition["from_revision"]
                or action_after["authorized_transition_to_revision"]
                != transition["to_revision"]
            ):
                raise ValueError("snapshot delete action binding mismatch")
        if (
            action_after["authority_barrier_nonce_sha256"]
            != after["cleanup_barrier_nonce_sha256"]
        ):
            raise ValueError("snapshot action lost cleanup barrier binding")
        for field in (
            "owner_attempt",
            "owner_execution_arn",
            "owner_state_machine_version_arn",
            "owner_dispatch_identity_sha256",
            "owner_invocation_nonce_sha256",
        ):
            if action_after[field] != after[field]:
                raise ValueError("snapshot action owner binding mismatch")
    elif after["state"] == "DELETE_POSSIBLY_SENT":
        raise ValueError("snapshot delete send requires a consumed action")
    _require_same_activation(
        cleanup_chain.after, after, transition
    )
    return SnapshotCleanupTransitionPlan(
        index,
        cleanup_chain,
        authority,
        cleanup_control,
        cleanup_transition,
        cleanup_action,
    )


def build_snapshot_delete_action_plan(
    *,
    index: ExactCheck,
    control: ExactCheck,
    action: Union[ExactPut, ExactUpdate],
) -> SnapshotDeleteActionPlan:
    """Validate an action arm/close without mutating cleanup control."""

    _require_type(control, ExactCheck, "snapshot cleanup control")
    _require_family(
        control,
        "glm52_production_snapshot_cleanup_control",
        "snapshot cleanup control",
    )
    control_record = validate_record(
        "glm52_production_snapshot_cleanup_control", control.expected
    )
    _validate_index(index, control_record)
    if type(action) not in {ExactPut, ExactUpdate}:
        raise TypeError("snapshot delete action has the wrong exact plan type")
    _require_family(
        action,
        "glm52_production_snapshot_cleanup_action",
        "snapshot delete action",
    )
    if type(action) is ExactPut:
        action_before = None
        action_after = validate_record(
            "glm52_production_snapshot_cleanup_action", action.item
        )
        if (
            control_record["state"] not in {"OWNED", "DELETE_RECONCILING"}
            or action_after["state"] != "ARMED"
            or action_after["attempt"]
            != control_record["delete_logical_attempt"] + 1
        ):
            raise ValueError("snapshot delete action arm binding mismatch")
    else:
        action_before = validate_record(
            "glm52_production_snapshot_cleanup_action", action.before
        )
        action_after = validate_transition(
            "glm52_production_snapshot_cleanup_action",
            action.before,
            action.after,
        )
        if (
            control_record["state"] != "DELETE_RECONCILING"
            or action_before["state"] != "CONSUMED"
            or action_after["state"] not in {"COMPLETED", "AMBIGUOUS"}
            or action_after["attempt"]
            != control_record["delete_logical_attempt"]
        ):
            raise ValueError("snapshot delete action close binding mismatch")
    _require_same_activation(control_record, action_after)
    if (
        action_after["authority_barrier_nonce_sha256"]
        != control_record["cleanup_barrier_nonce_sha256"]
    ):
        raise ValueError("snapshot delete action lost cleanup barrier binding")
    owner_fields = (
        "owner_attempt",
        "owner_execution_arn",
        "owner_state_machine_version_arn",
        "owner_dispatch_identity_sha256",
        "owner_invocation_nonce_sha256",
    )
    current_owner_binding = all(
        action_after[field] == control_record[field]
        and (
            action_before is None
            or action_before[field] == control_record[field]
        )
        for field in owner_fields
    )
    takeover_action_binding = (
        action_before is not None
        and all(
            action_before[field] == action_after[field]
            for field in owner_fields
        )
        and action_before["owner_attempt"] < control_record["owner_attempt"]
        and action_before["owner_hard_expires_at"]
        < control_record["updated_at"]
        and canonical_record_identity(
            "glm52_production_snapshot_cleanup_action",
            action_before,
        )
        == control_record["latest_delete_action_identity_sha256"]
    )
    if not current_owner_binding and not takeover_action_binding:
        raise ValueError("snapshot delete action owner binding mismatch")
    return SnapshotDeleteActionPlan(index, control, action)


def build_snapshot_cleanup_owner_takeover_plan(
    *,
    index: ExactCheck,
    authority: ExactCheck,
    cleanup_control: ExactUpdate,
) -> SnapshotCleanupOwnerTakeoverPlan:
    """Replace only an expired cleanup owner without changing cleanup work."""

    _require_type(authority, ExactCheck, "snapshot takeover authority")
    _require_family(
        authority,
        "glm52_production_control",
        "snapshot takeover authority",
    )
    authority_record = validate_record(
        "glm52_production_control", authority.expected
    )
    if authority_record["phase"] != "TEARDOWN_SEALED":
        raise ValueError("snapshot takeover requires sealed authority")
    _require_type(cleanup_control, ExactUpdate, "snapshot cleanup owner")
    _require_family(
        cleanup_control,
        "glm52_production_snapshot_cleanup_control",
        "snapshot cleanup owner",
    )
    before = validate_record(
        "glm52_production_snapshot_cleanup_control",
        cleanup_control.before,
    )
    after = validate_owner_takeover(
        "glm52_production_snapshot_cleanup_control",
        cleanup_control.before,
        cleanup_control.after,
    )
    if (
        before["state"]
        not in {"OWNED", "DELETE_POSSIBLY_SENT", "DELETE_RECONCILING"}
        or before["owner_hard_expires_at"] >= after["updated_at"]
        or before["owner_execution_arn"] == after["owner_execution_arn"]
    ):
        raise ValueError("snapshot takeover lacks an expired foreign owner")
    _require_same_activation(authority_record, after)
    _validate_index(index, after)
    return SnapshotCleanupOwnerTakeoverPlan(
        index, authority, cleanup_control
    )


def build_snapshot_cleanup_reconcile_read_plan(
    *,
    index: ExactCheck,
    authority: ExactCheck,
    cleanup_control: ExactCheck,
    cleanup_action: ExactCheck,
) -> SnapshotCleanupReconcileReadPlan:
    """Authenticate a fresh describe without inventing another state edge."""

    _require_type(authority, ExactCheck, "snapshot readback authority")
    _require_family(
        authority,
        "glm52_production_control",
        "snapshot readback authority",
    )
    authority_record = validate_record(
        "glm52_production_control", authority.expected
    )
    if authority_record["phase"] != "TEARDOWN_SEALED":
        raise ValueError("snapshot readback requires sealed authority")
    _require_type(cleanup_control, ExactCheck, "snapshot readback control")
    _require_family(
        cleanup_control,
        "glm52_production_snapshot_cleanup_control",
        "snapshot readback control",
    )
    control = validate_record(
        "glm52_production_snapshot_cleanup_control",
        cleanup_control.expected,
    )
    if control["state"] != "DELETE_RECONCILING":
        raise ValueError("snapshot readback requires reconciling control")
    index_record = _validate_index(index, control)
    if (
        index_record["current_activation_ordinal"]
        > control["activation_ordinal"] + 1
    ):
        raise ValueError("snapshot readback activation is outside N or N+1")
    _require_same_activation(authority_record, control)
    _require_type(cleanup_action, ExactCheck, "snapshot readback action")
    _require_family(
        cleanup_action,
        "glm52_production_snapshot_cleanup_action",
        "snapshot readback action",
    )
    action = validate_record(
        "glm52_production_snapshot_cleanup_action",
        cleanup_action.expected,
    )
    action_identity = canonical_record_identity(
        "glm52_production_snapshot_cleanup_action", action
    )
    if (
        action["state"] != "CONSUMED"
        or action["attempt"] != control["delete_logical_attempt"]
        or action_identity
        != control["latest_delete_action_identity_sha256"]
        or action["authority_barrier_nonce_sha256"]
        != control["cleanup_barrier_nonce_sha256"]
        or action["owner_attempt"] > control["owner_attempt"]
        or (
            action["owner_attempt"] != control["owner_attempt"]
            and action["owner_hard_expires_at"] >= control["updated_at"]
        )
    ):
        raise ValueError("snapshot readback action lineage drifted")
    _require_same_activation(control, action)
    return SnapshotCleanupReconcileReadPlan(
        index, authority, cleanup_control, cleanup_action
    )
