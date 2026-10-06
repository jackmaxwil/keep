"""Pure state-transition validation for GLM-5.2 durable records."""

from __future__ import annotations

from typing import Dict, FrozenSet, Mapping, Set, Tuple

from .records import (
    RECORD_FIELDS,
    _STANDARD_OWNER,
    _STATE_RULES,
    _WORKER_OWNER,
    RecordValidationError,
    validate_record,
)


class TransitionValidationError(ValueError):
    """A durable state transition violates its closed grammar."""


def _edges(text: str) -> FrozenSet[Tuple[str, str]]:
    return frozenset(tuple(edge.split(">")) for edge in text.split())


_RETAINED_ACTION = _edges(
    "ARMED>CONSUMED CONSUMED>COMPLETED CONSUMED>AMBIGUOUS ARMED>ABANDONED"
)

TRANSITIONS: Mapping[str, FrozenSet[Tuple[str, str]]] = {
    "glm52_production_control": _edges(
        "OPEN>RECOVERY_SEALING RECOVERY_SEALING>RECOVERY_COMPLETE "
        "RECOVERY_COMPLETE>TEARDOWN_SEALING TEARDOWN_SEALING>TEARDOWN_SEALED"
    ),
    "glm52_production_recovery_control": _edges(
        "DORMANT>OWNED OWNED>TERMINAL_V2_PUBLISHED "
        "TERMINAL_V2_PUBLISHED>RECOVERY_COMPLETE"
    ),
    "glm52_production_finalization_control": _edges(
        "DORMANT>OWNED OWNED>SUPPORT_FINALIZED "
        "SUPPORT_FINALIZED>SNAPSHOT_DISPOSITION_RECORDED "
        "SNAPSHOT_DISPOSITION_RECORDED>DRAINED_PUBLISHED"
    ),
    "glm52_production_snapshot_cleanup_control": _edges(
        "DORMANT>ARMED ARMED>OWNED OWNED>DELETE_POSSIBLY_SENT "
        "OWNED>ALREADY_ABSENT DELETE_POSSIBLY_SENT>DELETE_RECONCILING "
        "DELETE_RECONCILING>DELETE_POSSIBLY_SENT DELETE_RECONCILING>DELETED "
        "DELETE_RECONCILING>ALREADY_ABSENT "
        "DELETE_RECONCILING>CLEANUP_INCIDENT"
    ),
    "glm52_production_recovery_action": _RETAINED_ACTION,
    "glm52_production_finalization_action": _RETAINED_ACTION,
    "glm52_production_snapshot_cleanup_action": _RETAINED_ACTION,
    "glm52_production_worker_launch_liability_action": _RETAINED_ACTION,
    "glm52_production_execution": _edges(
        "START_OWNED>START_POSSIBLY_SENT "
        "START_OWNED>ABANDONED_PROVED_NOT_STARTED "
        "START_POSSIBLY_SENT>RUNNING "
        "START_POSSIBLY_SENT>START_POSSIBLY_SENT_UNRESOLVED_INCIDENT "
        "RUNNING>SUCCEEDED RUNNING>FAILED RUNNING>TIMED_OUT RUNNING>ABORTED"
    ),
    "glm52_production_worker_launch": _edges(
        "PREPARED_NOT_SENT>POSSIBLY_SENT "
        "PREPARED_NOT_SENT>ABANDONED_NOT_SENT "
        "POSSIBLY_SENT>INSTANCE_OBSERVED POSSIBLY_SENT>REJECTED_NO_INSTANCE "
        "POSSIBLY_SENT>MULTIPLE_INSTANCE_TOKEN_INCIDENT "
        "POSSIBLY_SENT>UNRESOLVED_LAUNCH_INCIDENT "
        "INSTANCE_OBSERVED>ALLOCATION_OPEN "
        "ALLOCATION_OPEN>INSTANCE_TERMINAL "
        "INSTANCE_TERMINAL>ALLOCATION_CLOSED"
    ),
    "glm52_production_worker_launch_liability": _edges(
        "UNOWNED_NOT_ACTIONABLE>WATCHING WATCHING>SAME_TOKEN_COMPLETION "
        "SAME_TOKEN_COMPLETION>WATCHING "
        "SAME_TOKEN_COMPLETION>REJECTION_PROVED_AWAITING_TERMINAL_V2 "
        "WATCHING>REJECTION_PROVED_AWAITING_TERMINAL_V2 "
        "REJECTION_PROVED_AWAITING_TERMINAL_V2>SETTLED_NO_INSTANCE_REJECTED "
        "WATCHING>LATE_INSTANCE_DRAINING "
        "LIABILITY_INCIDENT>LATE_INSTANCE_DRAINING "
        "LATE_INSTANCE_DRAINING>WATCHING "
        "LATE_INSTANCE_DRAINING>LIABILITY_INCIDENT "
        "WATCHING>SETTLED_INSTANCE_CLOSED "
        "WATCHING>LIABILITY_INCIDENT "
        "SAME_TOKEN_COMPLETION>LIABILITY_INCIDENT "
        "LIABILITY_INCIDENT>REJECTION_PROVED_AWAITING_TERMINAL_V2 "
        "LIABILITY_INCIDENT>SETTLED_INSTANCE_CLOSED"
    ),
    "glm52_production_post_terminal_allocation": _edges(
        "DISCOVERED>ALLOCATION_OPEN ALLOCATION_OPEN>INSTANCE_TERMINAL "
        "INSTANCE_TERMINAL>ALLOCATION_CLOSED"
    ),
    "glm52_production_action": _RETAINED_ACTION
    | _edges(
        "CONSUMED>POST_STARTED CONSUMED>POST_CLASSIFIED "
        "POST_STARTED>POST_AUTHORIZED POST_STARTED>POST_CLASSIFIED "
        "POST_AUTHORIZED>POST_CLASSIFIED"
    ),
}

_TRANSITION_METADATA = frozenset(
    {"state", "phase", "revision", "updated_at", "canonical_body_sha256"}
)
OWNER_FIELDS_BY_RECORD: Mapping[str, FrozenSet[str]] = {
    record_type: frozenset(
        _WORKER_OWNER
        if record_type == "glm52_production_worker_launch"
        else _STANDARD_OWNER
    )
    for record_type in _STATE_RULES
    if any(
        rule["owner"] != "unchanged"
        for rule in _STATE_RULES[record_type].values()
    )
}
APPEND_ONLY_ARRAY_FIELDS: Mapping[str, FrozenSet[str]] = {
    "glm52_production_snapshot_cleanup_control": frozenset(
        {"cleanup_authority_audit_identities", "delete_action_identities"}
    ),
    "glm52_production_worker_launch": frozenset(
        {"run_instances_attempt_evidence", "observed_instance_ids"}
    ),
    "glm52_production_worker_launch_liability": frozenset(
        {
            "observed_instance_ids",
            "late_instance_drain_identities",
            "late_instance_termination_action_identities",
            "late_instance_termination_call_counts",
            "post_terminal_allocation_identities",
            "spend_close_identities",
        }
    ),
}


def _field_modes(rule: Mapping[str, object]) -> Dict[str, object]:
    modes: Dict[str, object] = {}
    for field in rule["null"]:
        modes[field] = ("null",)
    for field in rule["nonnull"]:
        modes[field] = ("nonnull",)
    for field in rule["nonempty"]:
        modes[field] = ("nonempty",)
    for field, value in rule["exact"].items():
        modes[field] = ("exact", repr(value))
    return modes


def _edge_writable_fields(
    record_type: str, edge: Tuple[str, str]
) -> FrozenSet[str]:
    writable: Set[str] = set(_TRANSITION_METADATA)
    if record_type == "glm52_production_control":
        return frozenset(writable)
    before_rule = _STATE_RULES[record_type][edge[0]]
    after_rule = _STATE_RULES[record_type][edge[1]]
    before_modes = _field_modes(before_rule)
    after_modes = _field_modes(after_rule)
    for field in set(before_modes) | set(after_modes):
        if before_modes.get(field, ("unconstrained",)) != after_modes.get(
            field, ("unconstrained",)
        ):
            writable.add(field)
    if before_rule["owner"] != after_rule["owner"]:
        writable.update(OWNER_FIELDS_BY_RECORD.get(record_type, frozenset()))
        writable.add("owner_attempt")
    writable.update(APPEND_ONLY_ARRAY_FIELDS.get(record_type, frozenset()))
    return frozenset(writable)


EDGE_WRITABLE_FIELDS: Mapping[
    str, Mapping[Tuple[str, str], FrozenSet[str]]
] = {
    record_type: {
        edge: _edge_writable_fields(record_type, edge)
        for edge in edges
    }
    for record_type, edges in TRANSITIONS.items()
}
EDGE_IMMUTABLE_FIELDS: Mapping[
    str, Mapping[Tuple[str, str], FrozenSet[str]]
] = {
    record_type: {
        edge: frozenset(RECORD_FIELDS[record_type]) - writable
        for edge, writable in edge_policies.items()
    }
    for record_type, edge_policies in EDGE_WRITABLE_FIELDS.items()
}
RECORD_IMMUTABLE_FIELDS: Mapping[str, FrozenSet[str]] = {
    record_type: frozenset.intersection(
        *(EDGE_IMMUTABLE_FIELDS[record_type][edge] for edge in edges)
    )
    for record_type, edges in TRANSITIONS.items()
}

ACTION_EDGE_DOMAINS: Mapping[Tuple[str, str], str] = {
    ("CONSUMED", "POST_STARTED"): "SKY_POST",
    ("CONSUMED", "POST_CLASSIFIED"): "SKY_POST",
    ("POST_STARTED", "POST_AUTHORIZED"): "SKY_POST",
    ("POST_STARTED", "POST_CLASSIFIED"): "SKY_POST",
    ("POST_AUTHORIZED", "POST_CLASSIFIED"): "SKY_POST",
    ("CONSUMED", "COMPLETED"): "NON_SKY_POST",
    ("CONSUMED", "AMBIGUOUS"): "NON_SKY_POST",
}


def allowed_targets(record_type: str, state: str) -> FrozenSet[str]:
    if record_type not in TRANSITIONS:
        raise TransitionValidationError("record type has no transition grammar")
    return frozenset(
        target for source, target in TRANSITIONS[record_type] if source == state
    )


def validate_transition(
    record_type: str,
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> Dict[str, object]:
    try:
        old = validate_record(record_type, before)
        new = validate_record(record_type, after)
    except RecordValidationError as exc:
        raise TransitionValidationError(str(exc)) from exc
    state_field = "phase" if record_type == "glm52_production_control" else "state"
    edge = (old.get(state_field), new.get(state_field))
    if edge not in TRANSITIONS.get(record_type, frozenset()):
        raise TransitionValidationError("state edge is not allowed")
    if record_type == "glm52_production_action":
        domain = ACTION_EDGE_DOMAINS.get(edge)
        if domain == "SKY_POST" and old["action_kind"] != "SKY_POST":
            raise TransitionValidationError("post edge requires SKY_POST action")
        if domain == "NON_SKY_POST" and old["action_kind"] == "SKY_POST":
            raise TransitionValidationError(
                "non-post terminal edge forbids SKY_POST action"
            )
        if edge in {
            ("CONSUMED", "POST_CLASSIFIED"),
            ("POST_STARTED", "POST_CLASSIFIED"),
        } and (
            new["classification_evidence_kind"] != "OWNER_DEATH_PROOF"
            or new["outcome_class"] != "PROVED_NOT_SENT_OWNER_DIED"
        ):
            raise TransitionValidationError(
                "direct owner-death edge requires proved-not-sent outcome"
            )
    if (
        type(old.get("revision")) is not int
        or new.get("revision") != old["revision"] + 1
    ):
        raise TransitionValidationError("revision must increment by exactly one")
    writable = EDGE_WRITABLE_FIELDS[record_type][edge]
    changed = {field for field in old if old[field] != new[field]}
    forbidden = changed - writable
    if forbidden:
        raise TransitionValidationError(
            "edge changed immutable fields: " + ", ".join(sorted(forbidden))
        )
    for field in APPEND_ONLY_ARRAY_FIELDS.get(record_type, frozenset()):
        old_items = old[field]
        new_items = new[field]
        if (
            len(new_items) < len(old_items)
            or new_items[: len(old_items)] != old_items
        ):
            raise TransitionValidationError(
                "append-only evidence changed or was truncated: " + field
            )
    return new


def validate_owner_takeover(
    record_type: str,
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> Dict[str, object]:
    if record_type == "glm52_production_worker_launch":
        owner_fields = {
            "owner_principal_arn",
            "owner_function_version_arn",
            "owner_dispatch_identity_sha256",
            "owner_invocation_nonce_sha256",
            "owner_hard_expires_at",
        }
    else:
        owner_fields = {
            "owner_execution_arn",
            "owner_state_machine_version_arn",
            "owner_dispatch_identity_sha256",
            "owner_invocation_nonce_sha256",
            "owner_hard_expires_at",
        }
    if any(after.get(field) is None for field in owner_fields):
        raise TransitionValidationError("owner takeover has an ownerless gap")
    try:
        old = validate_record(record_type, before)
        new = validate_record(record_type, after)
    except RecordValidationError as exc:
        raise TransitionValidationError(str(exc)) from exc
    eligible = {
        "glm52_production_recovery_control": {
            "OWNED", "TERMINAL_V2_PUBLISHED"
        },
        "glm52_production_finalization_control": {
            "OWNED", "SUPPORT_FINALIZED", "SNAPSHOT_DISPOSITION_RECORDED"
        },
        "glm52_production_snapshot_cleanup_control": {
            "OWNED", "DELETE_POSSIBLY_SENT", "DELETE_RECONCILING"
        },
        "glm52_production_worker_launch": {
            "PREPARED_NOT_SENT", "POSSIBLY_SENT", "INSTANCE_OBSERVED",
            "ALLOCATION_OPEN", "INSTANCE_TERMINAL",
        },
        "glm52_production_worker_launch_liability": {
            "WATCHING", "SAME_TOKEN_COMPLETION",
            "REJECTION_PROVED_AWAITING_TERMINAL_V2",
            "LATE_INSTANCE_DRAINING", "LIABILITY_INCIDENT",
        },
        "glm52_production_post_terminal_allocation": {
            "DISCOVERED", "ALLOCATION_OPEN", "INSTANCE_TERMINAL"
        },
    }
    if old.get("state") not in eligible.get(record_type, set()):
        raise TransitionValidationError("owner takeover is illegal in this state")
    if new.get("state") != old.get("state"):
        raise TransitionValidationError("owner takeover must preserve state")
    if new.get("owner_attempt") != old.get("owner_attempt") + 1:
        raise TransitionValidationError("owner attempt must increment by exactly one")
    if new.get("revision") != old.get("revision") + 1:
        raise TransitionValidationError("revision must increment by exactly one")
    allowed = owner_fields | {
        "owner_attempt", "revision", "updated_at", "canonical_body_sha256"
    }
    changed = {field for field in old if old[field] != new[field]}
    if not changed <= allowed:
        raise TransitionValidationError("takeover changed non-owner evidence")
    if old["owner_invocation_nonce_sha256"] == new["owner_invocation_nonce_sha256"]:
        raise TransitionValidationError("owner nonce must change")
    return new
