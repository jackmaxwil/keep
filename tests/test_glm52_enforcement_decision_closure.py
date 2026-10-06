"""Task 11 private decision-to-POST closure contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from glm52_enforcement.decision_closure import (
    AUTHORITY_AUDIT_KINDS,
    CLOSURE_DEADLINE_SECONDS,
    CLOSURE_PHASE_CEILINGS,
    CLOSURE_STEPS,
    HANDOFF_AUDIT_KIND,
    LAMBDA_CONFIGURED_TIMEOUT_SECONDS,
    MIN_REMAINING_BEFORE_TOKEN_SECONDS,
    STEP_FUNCTION_TASK_TIMEOUT_SECONDS,
    SUFFIX_PHASE_CEILINGS,
    PhaseSpan,
    build_task11_workflow_definition,
    validate_timing_ledgers,
)
from glm52_enforcement.task12_support_continuation_adapter import (
    SUPPORT_OPERATION_SEQUENCE,
)


def _spans(
    names_and_durations: tuple[tuple[str, int], ...],
    *,
    start: int = 0,
) -> tuple[PhaseSpan, ...]:
    spans = []
    cursor = start
    for name, duration in names_and_durations:
        spans.append(
            PhaseSpan(
                phase_name=name,
                started_monotonic_seconds=cursor,
                ended_monotonic_seconds=cursor + duration,
            )
        )
        cursor += duration
    return tuple(spans)


def _exact_boundary_ledgers() -> tuple[
    tuple[PhaseSpan, ...],
    tuple[PhaseSpan, ...],
]:
    closure = _spans(tuple(CLOSURE_PHASE_CEILINGS.items()))
    suffix_start = closure[4].started_monotonic_seconds
    suffix = _spans(
        tuple(SUFFIX_PHASE_CEILINGS.items()),
        start=suffix_start,
    )
    return closure, suffix


def test_task11_red_exact_28_step_order_and_distinct_audit_grammar() -> None:
    assert CLOSURE_STEPS == (
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
    assert len(CLOSURE_STEPS) == len(set(CLOSURE_STEPS)) == 28
    assert AUTHORITY_AUDIT_KINDS == (
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
    assert len(AUTHORITY_AUDIT_KINDS) == len(
        set(AUTHORITY_AUDIT_KINDS)
    ) == 11
    assert HANDOFF_AUDIT_KIND == "SKY_POST_HANDOFF"
    assert HANDOFF_AUDIT_KIND not in AUTHORITY_AUDIT_KINDS


def test_task11_red_exact_timeout_and_nested_ledger_boundaries() -> None:
    assert LAMBDA_CONFIGURED_TIMEOUT_SECONDS == 840
    assert CLOSURE_DEADLINE_SECONDS == 720
    assert STEP_FUNCTION_TASK_TIMEOUT_SECONDS == 1200
    assert MIN_REMAINING_BEFORE_TOKEN_SECONDS == 120
    assert sum(CLOSURE_PHASE_CEILINGS.values()) == 600
    assert sum(SUFFIX_PHASE_CEILINGS.values()) == 58

    closure, suffix = _exact_boundary_ledgers()
    authority = validate_timing_ledgers(
        closure_spans=closure,
        suffix_spans=suffix,
        first_policy_readback_monotonic_seconds=(
            suffix[2].started_monotonic_seconds
        ),
        second_policy_readback_monotonic_seconds=(
            suffix[2].started_monotonic_seconds + 10
        ),
    )
    assert authority.closure_elapsed_seconds == 600
    assert authority.suffix_elapsed_seconds == 58


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate",
        "reordered",
        "overlap",
        "gap",
        "negative",
        "closure_overrun",
        "suffix_overrun",
        "wall_span_overrun",
        "clock_rollback",
        "policy_readbacks_too_close",
    ),
)
def test_task11_red_timing_mutants_fail_closed(mutation: str) -> None:
    closure, suffix = _exact_boundary_ledgers()
    first_readback = suffix[2].started_monotonic_seconds
    second_readback = first_readback + 10

    if mutation == "duplicate":
        closure = (closure[0], closure[0], *closure[2:])
    elif mutation == "reordered":
        closure = (closure[1], closure[0], *closure[2:])
    elif mutation == "overlap":
        closure = (
            closure[0],
            replace(
                closure[1],
                started_monotonic_seconds=(
                    closure[0].ended_monotonic_seconds - 1
                ),
            ),
            *closure[2:],
        )
    elif mutation == "gap":
        closure = (
            closure[0],
            replace(
                closure[1],
                started_monotonic_seconds=(
                    closure[0].ended_monotonic_seconds + 1
                ),
            ),
            *closure[2:],
        )
    elif mutation == "negative":
        closure = (
            replace(
                closure[0],
                ended_monotonic_seconds=(
                    closure[0].started_monotonic_seconds - 1
                ),
            ),
            *closure[1:],
        )
    elif mutation == "closure_overrun":
        closure = (
            replace(
                closure[0],
                ended_monotonic_seconds=(
                    closure[0].ended_monotonic_seconds + 1
                ),
            ),
            *closure[1:],
        )
    elif mutation == "suffix_overrun":
        suffix = (
            replace(
                suffix[0],
                ended_monotonic_seconds=suffix[0].ended_monotonic_seconds + 1,
            ),
            *suffix[1:],
        )
    elif mutation == "wall_span_overrun":
        closure = (
            closure[0],
            *tuple(
                replace(
                    span,
                    started_monotonic_seconds=(
                        span.started_monotonic_seconds + 1
                    ),
                    ended_monotonic_seconds=(
                        span.ended_monotonic_seconds + 1
                    ),
                )
                for span in closure[1:]
            ),
        )
        suffix = tuple(
            replace(
                span,
                started_monotonic_seconds=(
                    span.started_monotonic_seconds + 1
                ),
                ended_monotonic_seconds=(
                    span.ended_monotonic_seconds + 1
                ),
            )
            for span in suffix
        )
    elif mutation == "clock_rollback":
        suffix = (
            suffix[0],
            replace(
                suffix[1],
                started_monotonic_seconds=(
                    suffix[0].ended_monotonic_seconds - 1
                ),
            ),
            *suffix[2:],
        )
    else:
        second_readback = first_readback + 9

    with pytest.raises(ValueError):
        validate_timing_ledgers(
            closure_spans=closure,
            suffix_spans=suffix,
            first_policy_readback_monotonic_seconds=first_readback,
            second_policy_readback_monotonic_seconds=second_readback,
        )


def test_task11_red_workflow_is_versioned_standard_and_budget_gated() -> None:
    definition = build_task11_workflow_definition()
    assert definition["StartAt"] == "AUTHENTICATE_DEPLOYED_CLOSURE_BUDGET"
    assert tuple(definition["States"]) == (
        "AUTHENTICATE_DEPLOYED_CLOSURE_BUDGET",
        "VALIDATE_CLOSURE_BUDGET",
        "CLOSURE_BUDGET_UNPROVEN",
        "DECISION_CLOSURE",
        *SUPPORT_OPERATION_SEQUENCE,
        "CHECK_TERMINAL_V2_RESULT",
        "WAIT_FOR_TERMINAL_V2",
        "TERMINAL_V2_RESULT_INVALID",
    )
    assert definition["States"][
        "AUTHENTICATE_DEPLOYED_CLOSURE_BUDGET"
    ] == {
        "Type": "Task",
        "Resource": {"Ref": "BudgetGateVersion"},
        "Parameters": {
            "schema_version": 1,
            "record_type": "glm52_task11_budget_gate_request_v1",
        },
        "ResultPath": "$.trusted_closure_budget",
        "TimeoutSeconds": 120,
        "Next": "VALIDATE_CLOSURE_BUDGET",
    }
    gate = definition["States"]["VALIDATE_CLOSURE_BUDGET"]
    assert gate == {
        "Type": "Choice",
        "Choices": [
            {
                "Variable": "$.trusted_closure_budget.status",
                "StringEquals": "CLOSURE_BUDGET_PROVEN",
                "Next": "DECISION_CLOSURE",
            }
        ],
        "Default": "CLOSURE_BUDGET_UNPROVEN",
    }
    assert definition["States"]["CLOSURE_BUDGET_UNPROVEN"] == {
        "Type": "Fail",
        "Error": "CLOSURE_BUDGET_UNPROVEN",
    }
    assert definition["States"]["DECISION_CLOSURE"] == {
        "Type": "Task",
        "Resource": {"Ref": "DecisionVersion"},
        "TimeoutSeconds": 1200,
        "ResultPath": "$.decision_result",
        "Next": "VALIDATE_APPROVED_TASK11_HANDOFF",
    }
    for index, operation in enumerate(SUPPORT_OPERATION_SEQUENCE):
        state = definition["States"][operation]
        assert state["Resource"] == {
            "Ref": "SupportDeadlineVersion"
        }
        assert state["Parameters"]["operation_kind"] == operation
        assert state["ResultPath"] == (
            "$.terminal_attempt"
            if operation
            == "CREATE_OR_RECONCILE_PRODUCTION_TERMINAL_V2"
            else "$.support_last_result"
        )
        if operation == (
            "CREATE_OR_RECONCILE_PRODUCTION_TERMINAL_V2"
        ):
            assert state["Next"] == "CHECK_TERMINAL_V2_RESULT"
        elif index == len(SUPPORT_OPERATION_SEQUENCE) - 1:
            assert state["End"] is True
        else:
            assert state["Next"] == SUPPORT_OPERATION_SEQUENCE[index + 1]
    assert "Retry" not in str(definition)
    assert "closure_budget_authority" not in str(definition)
