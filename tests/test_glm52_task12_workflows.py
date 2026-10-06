from __future__ import annotations

import copy

import pytest


def _versions() -> dict[str, str]:
    names = (
        "task11_handoff_reader",
        "numeric_binding",
        "runtime_observer",
        "terminal_v2_writer",
        "finalization_requester",
        "recovery_coordinator",
        "recovery_cancellation",
        "recovery_drain",
        "recovery_terminal_writer",
        "teardown_coordinator",
        "finalizer",
        "orphan_auditor",
        "snapshot_scheduler",
        "h1g_drained_writer",
        "snapshot_cleanup_coordinator",
        "snapshot_cleanup_writer",
    )
    return {
        name: (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            f"keep-glm52-{name}:42"
        )
        for name in names
    }


def _assert_no_retry(definition: dict[str, object]) -> None:
    for state in definition["States"].values():
        if isinstance(state, dict) and state.get("Type") == "Task":
            assert "Retry" not in state


def test_support_continuation_is_exact_versioned_and_cannot_enter_retained_mode() -> None:
    from glm52_enforcement.task12_workflows import (
        build_support_continuation_workflow,
    )

    definition = build_support_continuation_workflow(versions=_versions())

    assert definition["StartAt"] == "VALIDATE_APPROVED_TASK11_HANDOFF"
    assert definition["TimeoutSeconds"] == 259200
    assert definition["States"]["NUMERIC_BINDING_HANDOFF"]["Resource"] == (
        _versions()["numeric_binding"]
    )
    assert definition["States"]["REQUEST_SUPPORT_FINALIZATION"] == {
        "Type": "Task",
        "Resource": _versions()["finalization_requester"],
        "Parameters": {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_support_continuation_invocation_v1"
            ),
            "operation_kind": "REQUEST_SUPPORT_FINALIZATION",
            "operation_input.$": "$",
        },
        "ResultPath": "$.support_last_result",
        "TimeoutSeconds": 540,
        "End": True,
    }
    _assert_no_retry(definition)
    assert "RETAINED_ACQUIRE_RECOVERY_SEALING" not in definition["States"]


def test_production_support_projection_accepts_only_authenticated_predecessors() -> None:
    from glm52_enforcement.task12_workflows import (
        Task12WorkflowError,
        support_continuation_operation_input,
    )

    assert support_continuation_operation_input(
        index=0,
        previous_result_path="$.support_last_result",
    ) == {
        "task11_context": {
            "closure_request.$": "$.closure_request",
            "decision_result.$": "$.decision_result",
            "task11_execution_arn.$": "$$.Execution.Id",
            "approved_task11_workflow_version_arn.$": (
                "$$.StateMachine.Id"
            ),
        }
    }
    assert support_continuation_operation_input(
        index=1,
        previous_result_path="$.support_last_result",
    ) == {"support_last_result.$": "$.support_last_result"}
    assert support_continuation_operation_input(
        index=12,
        previous_result_path="$.terminal_attempt",
    ) == {"support_last_result.$": "$.terminal_attempt"}
    with pytest.raises(Task12WorkflowError, match="ResultPath"):
        support_continuation_operation_input(
            index=1,
            previous_result_path="$.caller.execution_deadline",
        )


def test_retained_recovery_finalization_invokes_support_deletion_only_after_seal() -> None:
    from glm52_enforcement.task12_workflows import (
        build_retained_lifecycle_workflow,
    )

    definition = build_retained_lifecycle_workflow(
        versions=_versions(),
        support_deletion_workflow_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained-support-delete:7"
        ),
    )

    states = definition["States"]
    assert definition["TimeoutSeconds"] == 3600
    assert states["RETAINED_SUPPORT_DELETE_REQUESTED"]["Resource"] == (
        "arn:aws:states:::states:startExecution.sync:2"
    )
    assert states["RETAINED_SUPPORT_DELETE_REQUESTED"]["Parameters"][
        "StateMachineArn"
    ] == (
        "arn:aws:states:us-west-2:246813579024:stateMachine:"
        "keep-glm52-h1g-retained-support-delete:7"
    )
    assert states["RETAINED_ENTER_TEARDOWN_SEALED"]["Next"] == (
        "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"
    )
    assert states["RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"]["Next"] == (
        "RETAINED_SUPPORT_DELETE_REQUESTED"
    )
    _assert_no_retry(definition)
    assert "LAUNCH_ADMISSION" not in states
    assert "PUBLISH_SUBMISSION_INTENT" not in states


def test_snapshot_cleanup_has_bounded_choice_loop_without_retry() -> None:
    from glm52_enforcement.task12_workflows import (
        build_snapshot_cleanup_workflow,
    )

    definition = build_snapshot_cleanup_workflow(versions=_versions())
    states = definition["States"]

    assert definition["StartAt"] == "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE"
    assert states["SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"]["Catch"] == [
        {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.delete_ambiguity",
            "Next": "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
        }
    ]
    assert states["SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"]["Next"] == (
        "SNAPSHOT_CLEANUP_ROUTE_RECONCILIATION"
    )
    route = states["SNAPSHOT_CLEANUP_ROUTE_RECONCILIATION"]
    assert route["Type"] == "Choice"
    assert route["Default"] == (
        "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE"
    )
    assert route["Choices"][1]["And"][1] == {
        "Variable": (
            "$.task12_last_result.result.domain_result."
            "delete_logical_attempt"
        ),
        "NumericLessThan": 12,
    }
    assert states["SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE"]["End"] is True
    assert states["SNAPSHOT_CLEANUP_PRESERVE_RECONCILIATION_RESULT"] == {
        "Type": "Pass",
        "InputPath": "$.task12_last_result",
        "ResultPath": "$.snapshot_reconciliation_result",
        "Next": "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT",
    }
    assert states["SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT"]["Next"] == (
        "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY"
    )
    _assert_no_retry(definition)
    forbidden = {
        "RETAINED_ACQUIRE_RECOVERY_SEALING",
        "REQUEST_SUPPORT_FINALIZATION",
        "NUMERIC_BINDING",
    }
    assert not (forbidden & set(states))


@pytest.mark.parametrize(
defect := "mutator",
    [
        lambda definition: definition["States"][
            "NUMERIC_BINDING_HANDOFF"
        ].update(
            {"Retry": [{"ErrorEquals": ["States.ALL"]}]}
        ),
        lambda definition: definition["States"].update(
            {"RETAINED_ACQUIRE_RECOVERY_SEALING": {"Type": "Succeed"}}
        ),
    ],
)
def test_support_workflow_validator_rejects_retry_and_mode_crossover(mutator: object) -> None:
    from glm52_enforcement.task12_workflows import (
        Task12WorkflowError,
        build_support_continuation_workflow,
        validate_support_continuation_workflow,
    )

    definition = copy.deepcopy(build_support_continuation_workflow(versions=_versions()))
    mutator(definition)

    with pytest.raises(Task12WorkflowError):
        validate_support_continuation_workflow(definition, versions=_versions())


def test_workflow_builder_rejects_unqualified_lambda_and_alias_state_machine_arns() -> None:
    from glm52_enforcement.task12_workflows import (
        Task12WorkflowError,
        build_retained_lifecycle_workflow,
        build_support_continuation_workflow,
    )

    bad_versions = _versions()
    bad_versions["numeric_binding"] = (
        "arn:aws:lambda:us-west-2:246813579024:function:keep-glm52-numeric-binding"
    )
    with pytest.raises(Task12WorkflowError, match="published Lambda version"):
        build_support_continuation_workflow(versions=bad_versions)

    with pytest.raises(Task12WorkflowError, match="published state-machine version"):
        build_retained_lifecycle_workflow(
            versions=_versions(),
            support_deletion_workflow_version_arn=(
                "arn:aws:states:us-west-2:246813579024:stateMachine:"
                "keep-glm52-h1g-retained-support-delete:PROD"
            ),
        )


def test_retained_builder_requires_nine_distinct_published_lambda_versions() -> None:
    from glm52_enforcement.task12_workflows import (
        Task12WorkflowError,
        build_retained_lifecycle_workflow,
    )

    versions = _versions()
    versions["recovery_cancellation"] = versions["recovery_coordinator"]

    with pytest.raises(Task12WorkflowError, match="distinct published Lambda versions"):
        build_retained_lifecycle_workflow(
            versions=versions,
            support_deletion_workflow_version_arn=(
                "arn:aws:states:us-west-2:246813579024:stateMachine:"
                "keep-glm52-h1g-retained-support-delete:7"
            ),
        )


@pytest.mark.parametrize(
    defect := "mutator",
    [
        lambda definition: definition["States"][
            "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION"
        ].update({"Parameters": {"method": "POST", "source": "launch"}}),
        lambda definition: definition["States"]["RETAINED_ENTER_TEARDOWN_SEALED"].update(
            {"Next": "RETAINED_SUPPORT_DELETE_REQUESTED"}
        ),
    ],
)
def test_retained_validator_rejects_recovery_capability_and_early_deletion(
    mutator: object,
) -> None:
    from glm52_enforcement.task12_workflows import (
        Task12WorkflowError,
        build_retained_lifecycle_workflow,
        validate_retained_lifecycle_workflow,
    )

    deletion_arn = (
        "arn:aws:states:us-west-2:246813579024:stateMachine:"
        "keep-glm52-h1g-retained-support-delete:7"
    )
    definition = copy.deepcopy(
        build_retained_lifecycle_workflow(
            versions=_versions(), support_deletion_workflow_version_arn=deletion_arn
        )
    )
    mutator(definition)

    with pytest.raises(Task12WorkflowError):
        validate_retained_lifecycle_workflow(
            definition,
            versions=_versions(),
            support_deletion_workflow_version_arn=deletion_arn,
        )


def test_snapshot_validator_rejects_cleanup_back_edge() -> None:
    from glm52_enforcement.task12_workflows import (
        Task12WorkflowError,
        build_snapshot_cleanup_workflow,
        validate_snapshot_cleanup_workflow,
    )

    definition = copy.deepcopy(build_snapshot_cleanup_workflow(versions=_versions()))
    definition["States"]["SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT"].update(
        {"Next": "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK", "End": False}
    )

    with pytest.raises(Task12WorkflowError):
        validate_snapshot_cleanup_workflow(definition, versions=_versions())
