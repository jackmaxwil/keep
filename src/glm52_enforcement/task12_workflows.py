"""Closed, version-pinned Step Functions definitions for Task 12.

These builders deliberately contain no deployment or execution behaviour.  A
caller supplies the published Lambda and state-machine versions that make up a
particular deployment; the builders reject aliases and unqualified ARNs before
they can be rendered into an ASL document.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional


class Task12WorkflowError(ValueError):
    """Raised when a Task 12 workflow is not an exact closed definition."""


_LAMBDA_VERSION_ARN = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:[A-Za-z0-9_-]+:[1-9][0-9]*$"
)
_STATE_MACHINE_VERSION_ARN = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:stateMachine:[A-Za-z0-9_-]+:[1-9][0-9]*$"
)

_SUPPORT_KEYS = (
    "task11_handoff_reader",
    "numeric_binding",
    "runtime_observer",
    "terminal_v2_writer",
    "finalization_requester",
)
_RETAINED_KEYS = (
    "recovery_coordinator",
    "recovery_cancellation",
    "recovery_drain",
    "recovery_terminal_writer",
    "teardown_coordinator",
    "finalizer",
    "orphan_auditor",
    "snapshot_scheduler",
    "h1g_drained_writer",
)
_SNAPSHOT_KEYS = ("snapshot_cleanup_coordinator", "snapshot_cleanup_writer")


def _versioned_lambdas(versions: Mapping[str, str], required: tuple[str, ...]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for key in required:
        value = versions.get(key)
        if not isinstance(value, str) or not _LAMBDA_VERSION_ARN.fullmatch(value):
            raise Task12WorkflowError(
                "{0} must be an exact published Lambda version ARN".format(key)
            )
        selected[key] = value
    if len(set(selected.values())) != len(selected):
        raise Task12WorkflowError(
            "workflow capabilities require distinct published Lambda versions"
        )
    return selected


def _task(resource: str, next_state: Optional[str] = None, timeout: int = 540) -> dict[str, Any]:
    state: dict[str, Any] = {"Type": "Task", "Resource": resource, "TimeoutSeconds": timeout}
    if next_state is None:
        state["End"] = True
    else:
        state["Next"] = next_state
    return state


def support_continuation_operation_input(
    *,
    index: int,
    previous_result_path: str,
) -> dict[str, object]:
    """Project only authenticated workflow results into Task 12 invocations."""

    if type(index) is not int or index < 0:
        raise Task12WorkflowError(
            "support continuation operation index is invalid"
        )
    if index == 0:
        if previous_result_path != "$.support_last_result":
            raise Task12WorkflowError(
                "support continuation first ResultPath is foreign"
            )
        return {
            "task11_context": {
                "closure_request.$": "$.closure_request",
                "decision_result.$": "$.decision_result",
                "task11_execution_arn.$": "$$.Execution.Id",
                "approved_task11_workflow_version_arn.$": (
                    "$$.StateMachine.Id"
                ),
            }
        }
    if previous_result_path not in {
        "$.support_last_result",
        "$.terminal_attempt",
    }:
        raise Task12WorkflowError(
            "support continuation predecessor ResultPath is foreign"
        )
    return {"support_last_result.$": previous_result_path}


def build_support_continuation_workflow(*, versions: Mapping[str, str]) -> dict[str, Any]:
    """Build the support-owned post-decision continuation.

    These are deliberately not aliases for the 32 retained/snapshot Task 12
    operations.  Each state invokes the separate support-continuation contract
    with an exact ResultPath chain.
    """
    lambdas = _versioned_lambdas(versions, _SUPPORT_KEYS)
    names_and_resources = (
        ("VALIDATE_APPROVED_TASK11_HANDOFF", "task11_handoff_reader"),
        ("NUMERIC_BINDING_HANDOFF", "numeric_binding"),
        ("BIND_NUMERIC_JOB_OR_PROVE_NONE", "runtime_observer"),
        ("RUNTIME_OBSERVATION", "runtime_observer"),
        ("DRAIN_ARMED", "runtime_observer"),
        ("DRAIN_REQUESTED", "runtime_observer"),
        ("SUPPORT_REQUEST_CANCEL_RECONCILIATION", "runtime_observer"),
        ("SUPPORT_JOB_CANCEL_RECONCILIATION", "runtime_observer"),
        (
            "SUPPORT_CONTROLLER_QUIESCE_BEFORE_FORCE_IF_REQUIRED",
            "runtime_observer",
        ),
        ("WORKER_TERMINAL_RECONCILIATION", "runtime_observer"),
        ("ALLOCATION_CLOSED", "runtime_observer"),
        (
            "CREATE_OR_RECONCILE_PRODUCTION_TERMINAL_V2",
            "terminal_v2_writer",
        ),
        ("REQUEST_SUPPORT_FINALIZATION", "finalization_requester"),
    )
    states: dict[str, dict[str, Any]] = {}
    for index, (name, resource_key) in enumerate(names_and_resources):
        state = _task(
            lambdas[resource_key],
            (
                names_and_resources[index + 1][0]
                if index + 1 < len(names_and_resources)
                else None
            ),
        )
        state["Parameters"] = {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_support_continuation_invocation_v1"
            ),
            "operation_kind": name,
            "operation_input.$": "$",
        }
        state["ResultPath"] = "$.support_last_result"
        states[name] = state
    return {
        "StartAt": "VALIDATE_APPROVED_TASK11_HANDOFF",
        "TimeoutSeconds": 259200,
        "States": states,
    }


def validate_support_continuation_workflow(
    definition: Mapping[str, Any], *, versions: Mapping[str, str]
) -> None:
    """Reject any retry, extra state, or graph change in the support workflow."""
    expected = build_support_continuation_workflow(versions=versions)
    if not isinstance(definition, Mapping) or dict(definition) != expected:
        raise Task12WorkflowError("support continuation must match the closed Task 12 graph")


def build_retained_lifecycle_workflow(
    *, versions: Mapping[str, str], support_deletion_workflow_version_arn: str
) -> dict[str, Any]:
    """Build retained recovery/finalization, including the sealed delete call."""
    lambdas = _versioned_lambdas(versions, _RETAINED_KEYS)
    if not _STATE_MACHINE_VERSION_ARN.fullmatch(support_deletion_workflow_version_arn):
        raise Task12WorkflowError("support deletion target must be an exact published state-machine version ARN")

    names_and_resources = (
        ("RECONCILE_RETAINED_LIFECYCLE_TRIGGER", "recovery_coordinator"),
        ("RETAINED_ACQUIRE_RECOVERY_SEALING", "recovery_coordinator"),
        ("RETAINED_STOP_OR_OBSERVE_SUPPORT_EXECUTION", "recovery_coordinator"),
        ("RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT", "recovery_coordinator"),
        ("RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION", "recovery_coordinator"),
        ("RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED", "recovery_coordinator"),
        ("RETAINED_CORRELATE_REQUESTS_AND_JOBS", "recovery_cancellation"),
        ("RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION", "recovery_cancellation"),
        ("RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED", "recovery_drain"),
        ("RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES", "recovery_drain"),
        ("RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS", "recovery_drain"),
        ("RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2", "recovery_terminal_writer"),
        ("RETAINED_ENTER_RECOVERY_COMPLETE", "recovery_terminal_writer"),
        ("RETAINED_ACQUIRE_TEARDOWN_SEALING", "teardown_coordinator"),
        ("RETAINED_PROVE_ZERO_ACTIVATION_WORK", "teardown_coordinator"),
        ("RETAINED_ENTER_TEARDOWN_SEALED", "teardown_coordinator"),
        ("RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED", "finalizer"),
    )
    states: dict[str, dict[str, Any]] = {}
    for index, (name, key) in enumerate(names_and_resources):
        next_state = (
            names_and_resources[index + 1][0]
            if index + 1 < len(names_and_resources)
            else "RETAINED_SUPPORT_DELETE_REQUESTED"
        )
        states[name] = _task(lambdas[key], next_state)
    states["RETAINED_SUPPORT_DELETE_REQUESTED"] = {
        "Type": "Task",
        "Resource": "arn:aws:states:::states:startExecution.sync:2",
        "Parameters": {"StateMachineArn": support_deletion_workflow_version_arn},
        "TimeoutSeconds": 900,
        "ResultPath": "$.support_delete_result",
        "Next": "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
    }
    tail = (
        ("RETAINED_RECONCILE_DELETE_UNTIL_ABSENT", "finalizer"),
        ("RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT", "orphan_auditor"),
        ("RETAINED_AUDIT_SUPPORT_ORPHANS", "orphan_auditor"),
        ("RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE", "snapshot_scheduler"),
        ("RETAINED_INVOKE_H1G_DRAINED_WRITER", "h1g_drained_writer"),
    )
    for index, (name, key) in enumerate(tail):
        next_state = tail[index + 1][0] if index + 1 < len(tail) else None
        states[name] = _task(lambdas[key], next_state)
    return {"StartAt": names_and_resources[0][0], "TimeoutSeconds": 3600, "States": states}


def build_snapshot_cleanup_workflow(*, versions: Mapping[str, str]) -> dict[str, Any]:
    """Build the bounded, same-id snapshot deletion reconciliation workflow."""
    lambdas = _versioned_lambdas(versions, _SNAPSHOT_KEYS)
    states = {
        "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE": _task(
            lambdas["snapshot_cleanup_coordinator"], "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
        ),
        "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER": _task(
            lambdas["snapshot_cleanup_coordinator"],
            "SNAPSHOT_CLEANUP_ROUTE_ACQUIRED_CONTROL",
        ),
        "SNAPSHOT_CLEANUP_ROUTE_ACQUIRED_CONTROL": {
            "Type": "Choice",
            "Choices": [
                {
                    "Variable": (
                        "$.task12_last_result.result.domain_result."
                        "cleanup_state"
                    ),
                    "StringEquals": "OWNED",
                    "Next": "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION",
                },
                {
                    "Variable": (
                        "$.task12_last_result.result.domain_result."
                        "cleanup_state"
                    ),
                    "StringEquals": "DELETE_POSSIBLY_SENT",
                    "Next": "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
                },
                {
                    "Variable": (
                        "$.task12_last_result.result.domain_result."
                        "cleanup_state"
                    ),
                    "StringEquals": "DELETE_RECONCILING",
                    "Next": "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
                },
            ],
            "Default": "SNAPSHOT_CLEANUP_ACQUISITION_STATE_INVALID",
        },
        "SNAPSHOT_CLEANUP_ACQUISITION_STATE_INVALID": {
            "Type": "Fail",
            "Error": "SNAPSHOT_CLEANUP_ACQUISITION_STATE_INVALID",
        },
        "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION": _task(
            lambdas["snapshot_cleanup_coordinator"], "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY"
        ),
        "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY": _task(
            lambdas["snapshot_cleanup_coordinator"], "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT"
        ),
        "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT": _task(
            lambdas["snapshot_cleanup_coordinator"], "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
        ),
        "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK": {
            "Type": "Task",
            "Resource": lambdas["snapshot_cleanup_coordinator"],
            "TimeoutSeconds": 540,
            "Next": "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
            "Catch": [
                {
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.delete_ambiguity",
                    "Next": "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
                }
            ],
        },
        "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE": _task(
            lambdas["snapshot_cleanup_coordinator"],
            "SNAPSHOT_CLEANUP_ROUTE_RECONCILIATION",
        ),
        "SNAPSHOT_CLEANUP_ROUTE_RECONCILIATION": {
            "Type": "Choice",
            "Choices": [
                {
                    "Variable": (
                        "$.task12_last_result.result.domain_result."
                        "snapshot_present"
                    ),
                    "BooleanEquals": False,
                    "Next": "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE",
                },
                {
                    "And": [
                        {
                            "Variable": (
                                "$.task12_last_result.result.domain_result."
                                "snapshot_present"
                            ),
                            "BooleanEquals": True,
                        },
                        {
                            "Variable": (
                                "$.task12_last_result.result.domain_result."
                                "delete_logical_attempt"
                            ),
                            "NumericLessThan": 12,
                        },
                    ],
                    "Next": "SNAPSHOT_CLEANUP_PRESERVE_RECONCILIATION_RESULT",
                },
            ],
            "Default": "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE",
        },
        "SNAPSHOT_CLEANUP_PRESERVE_RECONCILIATION_RESULT": {
            "Type": "Pass",
            "InputPath": "$.task12_last_result",
            "ResultPath": "$.snapshot_reconciliation_result",
            "Next": "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT",
        },
        "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE": _task(
            lambdas["snapshot_cleanup_writer"]
        ),
        "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT": _task(
            lambdas["snapshot_cleanup_writer"], "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT"
        ),
        "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT": _task(
            lambdas["snapshot_cleanup_writer"],
            "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY",
        ),
    }
    return {
        "StartAt": "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE",
        "TimeoutSeconds": 3600,
        "States": states,
    }


def validate_retained_lifecycle_workflow(
    definition: Mapping[str, Any],
    *,
    versions: Mapping[str, str],
    support_deletion_workflow_version_arn: str,
) -> None:
    """Reject recovery-mode launch paths, early deletion, and graph drift."""
    expected = build_retained_lifecycle_workflow(
        versions=versions,
        support_deletion_workflow_version_arn=support_deletion_workflow_version_arn,
    )
    if not isinstance(definition, Mapping) or dict(definition) != expected:
        raise Task12WorkflowError("retained lifecycle must match the closed Task 12 graph")


def validate_snapshot_cleanup_workflow(
    definition: Mapping[str, Any], *, versions: Mapping[str, str]
) -> None:
    """Reject retries and any edge outside the bounded same-ID Choice loop."""
    expected = build_snapshot_cleanup_workflow(versions=versions)
    if not isinstance(definition, Mapping) or dict(definition) != expected:
        raise Task12WorkflowError("snapshot cleanup must match the closed Task 12 graph")
