from __future__ import annotations

from copy import deepcopy
import json
import re
from types import SimpleNamespace
from typing import Mapping

import pytest


ACTIVATION_ID = "h1g-act-20260729-0001"
ACTIVATION_ORDINAL = 1
GENERATION = 1
GENERATION_TEXT = "00000001"
DISPATCH_IDENTITY = "d" * 64
CALLER_NAMES = {
    "support": "keep-glm52-h1g-support-continuation",
    "retained": "keep-glm52-h1g-retained-lifecycle",
    "snapshot": "keep-glm52-h1g-snapshot-cleanup",
}
STATE_MACHINE_VERSIONS = {
    name: (
        "arn:aws:states:us-west-2:246813579024:stateMachine:"
        + caller
        + ":7"
    )
    for name, caller in CALLER_NAMES.items()
}
STATE_MACHINE_ARNS = {
    name: version.rsplit(":", 1)[0]
    for name, version in STATE_MACHINE_VERSIONS.items()
}
EXECUTION_ARNS = {
    name: (
        "arn:aws:states:us-west-2:246813579024:execution:"
        + caller
        + ":execution-1"
    )
    for name, caller in CALLER_NAMES.items()
}
_INDEX = re.compile(r"^([A-Za-z0-9_]+)\[([0-9]+)\]$")


def _inputs() -> object:
    from glm52_enforcement.task12_support_plane import (
        Task12SupportPlaneInputs,
    )

    return Task12SupportPlaneInputs(
        activation_id=ACTIVATION_ID,
        lambda_code_sha256="a" * 64,
        lambda_code_bucket="keep-glm52-runtime-artifacts",
        lambda_code_key="releases/task12-runtime.zip",
        lambda_code_version="3HL4kqtJlcpXroDTDmJ+rmSpXd3Ly3B/",
        ledger_table_arn=(
            "arn:aws:dynamodb:us-west-2:246813579024:"
            "table/keep-glm52-h1g-ledger-v1"
        ),
        retained_kms_key_arn=(
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-4234-8234-1234567890ab"
        ),
        model_bucket_arn="arn:aws:s3:::keep-glm52-models",
        worker_drain_document_arn=(
            "arn:aws:ssm:us-west-2:246813579024:"
            "document/KeepGlm52GracefulStopV1"
        ),
    )


def _versions() -> dict[str, str]:
    names = (
        "task11_handoff_reader",
        "numeric_binding",
        "runtime_observer",
        "terminal_v2_writer",
        "finalization_requester",
    )
    return {
        name: (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            f"keep-glm52-{name}:42"
        )
        for name in names
    }


def _definitions() -> dict[str, dict[str, object]]:
    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )
    from glm52_enforcement.task12_workflows import (
        build_support_continuation_workflow,
    )

    resources = render_task12_support_plane_fragment(
        inputs=_inputs()
    )["Resources"]
    return {
        "support": build_support_continuation_workflow(
            versions=_versions()
        ),
        "retained": json.loads(
            resources["RetainedLifecycleStateMachine"]["Properties"][
                "DefinitionString"
            ]
        ),
        "snapshot": json.loads(
            resources["SnapshotCleanupStateMachine"]["Properties"][
                "DefinitionString"
            ]
        ),
    }


def _path(value: object, expression: str) -> object:
    if expression == "$":
        return value
    assert expression.startswith("$."), expression
    current = value
    for raw_part in expression[2:].split("."):
        matched = _INDEX.fullmatch(raw_part)
        key = matched.group(1) if matched is not None else raw_part
        assert isinstance(current, Mapping), expression
        current = current[key]
        if matched is not None:
            assert isinstance(current, list), expression
            current = current[int(matched.group(2))]
    return current


def _merge(value: Mapping[str, object], result_path: str, result: object) -> dict[
    str, object
]:
    assert result_path.startswith("$.") and "." not in result_path[2:]
    merged = deepcopy(dict(value))
    merged[result_path[2:]] = deepcopy(result)
    return merged


def _handler_kind(operation_kind: str) -> str:
    from glm52_enforcement import task12_lambda_adapters as adapters

    matches = [
        handler_kind
        for handler_kind, operations in adapters._OPERATIONS.items()
        if operation_kind in operations
    ]
    if len(matches) != 1:
        raise AssertionError(
            operation_kind + " has no unique production wrapper parser"
        )
    return matches[0]


def _deployment(operation_kind: str, workflow: str) -> object:
    from glm52_enforcement import task12_lambda_adapters as adapters
    from glm52_enforcement.task12_lambda_adapters import (
        Task12LambdaDeployment,
        VersionedAuthorityCoordinate,
    )

    handler_kind = _handler_kind(operation_kind)
    spec = adapters._SPECS[handler_kind]
    function_arn = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        + spec.function_name
        + ":42"
    )
    return Task12LambdaDeployment(
        handler_kind=handler_kind,
        mode=spec.mode,
        activation_id=ACTIVATION_ID,
        activation_ordinal=ACTIVATION_ORDINAL,
        generation=GENERATION,
        generation_text=GENERATION_TEXT,
        function_name=spec.function_name,
        function_version="42",
        invoked_function_version_arn=function_arn,
        caller_state_machine_version_arn=STATE_MACHINE_VERSIONS[workflow],
        authority=VersionedAuthorityCoordinate(
            bucket="keep-glm52-retained",
            key="task12/authority.json",
            version_id="authority-version-1",
            file_sha256="a" * 64,
        ),
        role_coordinates={},
        canonical_body_sha256="b" * 64,
    )


def _static_parameter(value: object) -> object:
    replacements: dict[str, object] = {
        "${Task12GenerationText}": GENERATION_TEXT,
        "${Task12DispatchIdentitySha256}": DISPATCH_IDENTITY,
    }
    return replacements.get(value, value)


def _event(
    *,
    state_name: str,
    state: Mapping[str, object],
    state_input: Mapping[str, object],
    workflow: str,
) -> dict[str, object]:
    parameters = state.get("Parameters")
    assert isinstance(parameters, dict), (
        state_name + " is not executable through the Task12 wrapper"
    )
    event: dict[str, object] = {}
    for raw_key, value in parameters.items():
        if raw_key.endswith(".$"):
            key = raw_key[:-2]
            if value == "$":
                event[key] = deepcopy(dict(state_input))
            elif value == "$$.StateMachine.Id":
                event[key] = STATE_MACHINE_ARNS[workflow]
            elif value == "$$.Execution.Id":
                event[key] = EXECUTION_ARNS[workflow]
            elif value == (
                "States.StringToJson('${Task12ActivationOrdinal}')"
            ):
                event[key] = ACTIVATION_ORDINAL
            elif value == "States.StringToJson('${Task12Generation}')":
                event[key] = GENERATION
            elif isinstance(value, str) and value.startswith("$."):
                event[key] = deepcopy(_path(state_input, value))
            else:  # pragma: no cover - a new intrinsic is a contract change
                raise AssertionError("unsupported ASL intrinsic: " + str(value))
        else:
            event[raw_key] = _static_parameter(value)
    return event


def _execute_task(
    *,
    definition: Mapping[str, object],
    state_name: str,
    state_input: Mapping[str, object],
    workflow: str,
    domain_result: object,
) -> tuple[object, dict[str, object], dict[str, object]]:
    from glm52_enforcement import task12_lambda_adapters as adapters

    states = definition["States"]
    assert isinstance(states, dict)
    state = states[state_name]
    assert isinstance(state, dict) and state.get("Type") == "Task"
    deployment = _deployment(state_name, workflow)
    event = _event(
        state_name=state_name,
        state=state,
        state_input=state_input,
        workflow=workflow,
    )
    invocation = adapters._parse_invocation(
        event=event,
        context=SimpleNamespace(
            invoked_function_arn=deployment.invoked_function_version_arn
        ),
        deployment=deployment,
    )
    result = adapters._result(
        handler_kind=deployment.handler_kind,
        outcome="SUCCEEDED",
        operation_kind=invocation.operation_kind,
        result=domain_result,
        operation_input=invocation.operation_input,
    )
    result_path = state.get("ResultPath")
    assert isinstance(result_path, str), (
        state_name + " does not preserve its wrapper result"
    )
    return invocation, _merge(state_input, result_path, result), result


def _capsule() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_task12_owner_nonce_capsule_v1",
        "kms_key_id": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-4234-8234-1234567890ab"
        ),
        "encryption_context": {
            "activation_id": ACTIVATION_ID,
            "authority_domain": "RECOVERY",
        },
        "ciphertext_base64": "Y2lwaGVydGV4dA==",
        "nonce_sha256": "e" * 64,
        "canonical_body_sha256": "f" * 64,
    }


def _assert_no_raw_nonce(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert key not in {
                "raw_owner_nonce",
                "raw_owner_nonce_hex",
                "task12_runtime_authority",
            }
            _assert_no_raw_nonce(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_raw_nonce(item)


def _assert_same_execution(left: object, right: object) -> None:
    fields = (
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "dispatch_identity_sha256",
        "deployment_identity_sha256",
        "invoked_function_version_arn",
        "caller_state_machine_arn",
        "caller_state_machine_version_arn",
        "state_machine_execution_arn",
    )
    assert {field: getattr(left, field) for field in fields} == {
        field: getattr(right, field) for field in fields
    }


@pytest.mark.parametrize(
    ("workflow", "first", "second"),
    [
        (
            "retained",
            "RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
            "RETAINED_ACQUIRE_RECOVERY_SEALING",
        ),
        (
            "snapshot",
            "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE",
            "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        ),
    ],
)
def test_two_consecutive_tasks_compose_exact_wrapper_invocations(
    workflow: str,
    first: str,
    second: str,
) -> None:
    """Break caught: a conceptual ASL edge is not an executable wrapper edge."""

    definition = _definitions()[workflow]
    capsule = _capsule()
    first_invocation, state, first_result = _execute_task(
        definition=definition,
        state_name=first,
        state_input={},
        workflow=workflow,
        domain_result={"owner_nonce_capsule": capsule},
    )
    assert definition["States"][first]["Next"] == second
    second_invocation, _, _ = _execute_task(
        definition=definition,
        state_name=second,
        state_input=state,
        workflow=workflow,
        domain_result={},
    )

    assert first_invocation.operation_kind == first
    assert second_invocation.operation_kind == second
    _assert_same_execution(first_invocation, second_invocation)
    assert second_invocation.operation_input == {
        "task12_last_result": first_result
    }
    assert (
        second_invocation.operation_input["task12_last_result"][
            "operation_kind"
        ]
        == first
    )
    assert "operation_kind" not in {
        key
        for key in second_invocation.operation_input
        if key != "task12_last_result"
    }
    assert second_invocation.operation_input["task12_last_result"]["result"][
        "owner_nonce_capsule"
    ] == capsule
    _assert_no_raw_nonce(second_invocation.operation_input)


def test_support_post_decision_tasks_compose_through_separate_contract() -> None:
    """Break caught: support states are fake aliases in the 32-row registry."""

    from glm52_enforcement.task12_support_continuation_adapter import (
        build_support_result,
        parse_support_invocation,
    )

    definition = _definitions()["support"]
    first_name = "VALIDATE_APPROVED_TASK11_HANDOFF"
    second_name = "NUMERIC_BINDING_HANDOFF"
    first_state = definition["States"][first_name]
    first_event = _event(
        state_name=first_name,
        state=first_state,
        state_input={},
        workflow="support",
    )
    first_invocation = parse_support_invocation(first_event)
    first_result = build_support_result(
        invocation=first_invocation,
        result={"handoff_state": "APPROVED"},
    )
    state_input = _merge(
        {},
        str(first_state["ResultPath"]),
        first_result,
    )
    assert first_state["Next"] == second_name
    second_state = definition["States"][second_name]
    second_event = _event(
        state_name=second_name,
        state=second_state,
        state_input=state_input,
        workflow="support",
    )
    second_invocation = parse_support_invocation(second_event)
    assert second_invocation.operation_kind == second_name
    assert second_invocation.operation_input == {
        "support_last_result": first_result
    }
    assert first_name not in {
        operation
        for operations in __import__(
            "glm52_enforcement.task12_lambda_adapters",
            fromlist=["_OPERATIONS"],
        )._OPERATIONS.values()
        for operation in operations
    }


def _reconcile_input(
    *,
    snapshot_present: bool,
    attempt: int,
    ambiguity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from glm52_enforcement import task12_lambda_adapters as adapters

    result = adapters._result(
        handler_kind="RETAINED_SNAPSHOT_CLEANUP",
        outcome="SUCCEEDED",
        operation_kind="SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
        result={
            "domain_result": {
                "source_state": "DELETE_POSSIBLY_SENT",
                "target_state": "DELETE_RECONCILING",
                "committed": True,
                "snapshot_present": snapshot_present,
                "delete_logical_attempt": attempt,
                "resolution": {
                    "records": [{}, {}, {}, {
                        "delete_logical_attempt": attempt
                    }]
                },
            },
        },
        operation_input={},
    )
    state: dict[str, object] = {"task12_last_result": result}
    if ambiguity is not None:
        state["delete_ambiguity"] = deepcopy(dict(ambiguity))
    return state


def _acquisition_input(cleanup_state: str) -> dict[str, object]:
    from glm52_enforcement import task12_lambda_adapters as adapters

    result = adapters._result(
        handler_kind="RETAINED_SNAPSHOT_CLEANUP",
        outcome="SUCCEEDED",
        operation_kind="SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        result={
            "domain_result": {
                "cleanup_state": cleanup_state,
                "resolution": {
                    "outcome": "EXACT_LIVE_OWNER_COMMIT",
                    "records": [],
                    "request_id": "takeover-1",
                    "error_code": None,
                    "cancellation_reasons": [],
                },
            },
            "owner_nonce_capsule": _capsule(),
        },
        operation_input={},
    )
    return {"task12_last_result": result}


@pytest.mark.parametrize(
    ("cleanup_state", "expected"),
    [
        ("OWNED", "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION"),
        (
            "DELETE_POSSIBLY_SENT",
            "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
        ),
        (
            "DELETE_RECONCILING",
            "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
        ),
    ],
)
def test_snapshot_post_acquire_choice_routes_from_authenticated_state(
    cleanup_state: str,
    expected: str,
) -> None:
    definition = _definitions()["snapshot"]
    choice = definition["States"]["SNAPSHOT_CLEANUP_ROUTE_ACQUIRED_CONTROL"]

    assert _choice_next(choice, _acquisition_input(cleanup_state)) == expected


@pytest.mark.parametrize(
    "cleanup_state",
    ["DELETE_POSSIBLY_SENT", "DELETE_RECONCILING"],
)
def test_snapshot_recovered_owner_executes_fresh_reconcile_without_ambiguity(
    cleanup_state: str,
) -> None:
    definition = _definitions()["snapshot"]
    state_input = _acquisition_input(cleanup_state)
    next_state = _choice_next(
        definition["States"]["SNAPSHOT_CLEANUP_ROUTE_ACQUIRED_CONTROL"],
        state_input,
    )
    invocation, _, _ = _execute_task(
        definition=definition,
        state_name=next_state,
        state_input=state_input,
        workflow="snapshot",
        domain_result={
            "source_state": cleanup_state,
            "target_state": "DELETE_RECONCILING",
            "committed": cleanup_state == "DELETE_POSSIBLY_SENT",
            "snapshot_present": True,
            "delete_logical_attempt": 1,
            "resolution": (
                {
                    "outcome": "EXACT_LIVE_OWNER_COMMIT",
                    "records": [],
                    "request_id": "reconcile-1",
                    "error_code": None,
                    "cancellation_reasons": [],
                }
                if cleanup_state == "DELETE_POSSIBLY_SENT"
                else None
            ),
        },
    )

    assert invocation.operation_kind == (
        "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"
    )
    assert invocation.operation_input == state_input
    assert "delete_ambiguity" not in invocation.operation_input


def _choice_next(
    state: Mapping[str, object], state_input: Mapping[str, object]
) -> str:
    assert state.get("Type") == "Choice"

    def matches(rule: Mapping[str, object]) -> bool:
        if "And" in rule:
            clauses = rule["And"]
            assert isinstance(clauses, list)
            return all(matches(item) for item in clauses)
        value = _path(state_input, str(rule["Variable"]))
        if "BooleanEquals" in rule:
            return value is rule["BooleanEquals"]
        if "StringEquals" in rule:
            return value == rule["StringEquals"]
        if "NumericLessThan" in rule:
            return (
                type(value) in {int, float}
                and value < rule["NumericLessThan"]
            )
        raise AssertionError("unsupported Choice comparison")

    choices = state.get("Choices")
    assert isinstance(choices, list)
    for choice in choices:
        assert isinstance(choice, dict)
        if matches(choice):
            return str(choice["Next"])
    return str(state["Default"])


def _advance_passes(
    *,
    definition: Mapping[str, object],
    state_name: str,
    state_input: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    states = definition["States"]
    assert isinstance(states, dict)
    current_name = state_name
    current_input = deepcopy(dict(state_input))
    while states[current_name].get("Type") == "Pass":
        state = states[current_name]
        selected = _path(current_input, str(state.get("InputPath", "$")))
        current_input = _merge(
            current_input,
            str(state["ResultPath"]),
            selected,
        )
        current_name = str(state["Next"])
    return current_name, current_input


@pytest.mark.parametrize(
    ("snapshot_present", "attempt", "expected"),
    [
        (False, 1, "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE"),
        (True, 11, "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"),
        (True, 12, "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE"),
    ],
)
def test_snapshot_choice_routes_terminal_retry_and_bound(
    snapshot_present: bool,
    attempt: int,
    expected: str,
) -> None:
    """Break caught: terminal evidence enters close, or attempt 12 loops."""

    definition = _definitions()["snapshot"]
    state_input = _reconcile_input(
        snapshot_present=snapshot_present,
        attempt=attempt,
    )
    choice = definition["States"][
        "SNAPSHOT_CLEANUP_ROUTE_RECONCILIATION"
    ]
    next_state, _ = _advance_passes(
        definition=definition,
        state_name=_choice_next(choice, state_input),
        state_input=state_input,
    )

    assert next_state == expected
    if snapshot_present is False:
        assert next_state != "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"
    if attempt == 12:
        assert next_state != "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"


def test_snapshot_terminal_branch_executes_record_then_end() -> None:
    """Break caught: a terminal reconciliation is routed into retry closure."""

    definition = _definitions()["snapshot"]
    state_input = _reconcile_input(snapshot_present=False, attempt=1)
    next_state = _choice_next(
        definition["States"]["SNAPSHOT_CLEANUP_ROUTE_RECONCILIATION"],
        state_input,
    )
    invocation, _, _ = _execute_task(
        definition=definition,
        state_name=next_state,
        state_input=state_input,
        workflow="snapshot",
        domain_result={},
    )

    assert invocation.operation_kind == (
        "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE"
    )
    assert definition["States"][next_state]["End"] is True
    assert next_state != "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"


def test_snapshot_present_branch_executes_close_armnext_audit_atomic_loop() -> None:
    """Break caught: the ASL branch disagrees with wrapper predecessor custody."""

    definition = _definitions()["snapshot"]
    state_input = _reconcile_input(snapshot_present=True, attempt=11)
    close, state_input = _advance_passes(
        definition=definition,
        state_name=_choice_next(
            definition["States"][
                "SNAPSHOT_CLEANUP_ROUTE_RECONCILIATION"
            ],
            state_input,
        ),
        state_input=state_input,
    )
    close_invocation, state_input, _ = _execute_task(
        definition=definition,
        state_name=close,
        state_input=state_input,
        workflow="snapshot",
        domain_result={},
    )
    arm_next = definition["States"][close]["Next"]
    arm_invocation, state_input, _ = _execute_task(
        definition=definition,
        state_name=arm_next,
        state_input=state_input,
        workflow="snapshot",
        domain_result={},
    )
    audit = definition["States"][arm_next]["Next"]
    audit_invocation, state_input, _ = _execute_task(
        definition=definition,
        state_name=audit,
        state_input=state_input,
        workflow="snapshot",
        domain_result={},
    )
    atomic = definition["States"][audit]["Next"]
    atomic_invocation, _, _ = _execute_task(
        definition=definition,
        state_name=atomic,
        state_input=state_input,
        workflow="snapshot",
        domain_result={},
    )

    assert [
        close_invocation.operation_kind,
        arm_invocation.operation_kind,
        audit_invocation.operation_kind,
        atomic_invocation.operation_kind,
    ] == [
        "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT",
        "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT",
        "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY",
        "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
    ]


def test_snapshot_close_accepts_reconcile_as_its_branch_predecessor() -> None:
    """Break caught: the branch parser insists on the unreachable linear prior."""

    definition = _definitions()["snapshot"]
    state_input = _reconcile_input(snapshot_present=True, attempt=11)
    close, state_input = _advance_passes(
        definition=definition,
        state_name=_choice_next(
            definition["States"][
                "SNAPSHOT_CLEANUP_ROUTE_RECONCILIATION"
            ],
            state_input,
        ),
        state_input=state_input,
    )

    invocation, _, _ = _execute_task(
        definition=definition,
        state_name=close,
        state_input=state_input,
        workflow="snapshot",
        domain_result={},
    )

    assert invocation.operation_kind == (
        "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"
    )
    assert invocation.operation_input["snapshot_reconciliation_result"] == (
        invocation.operation_input["task12_last_result"]
    )


def test_snapshot_audit_accepts_armnext_as_its_loop_predecessor() -> None:
    """Break caught: the loop re-enters audit with an authenticated arm-next result."""

    from glm52_enforcement import task12_lambda_adapters as adapters

    definition = _definitions()["snapshot"]
    arm_next_result = adapters._result(
        handler_kind="RETAINED_SNAPSHOT_CLEANUP",
        outcome="SUCCEEDED",
        operation_kind="SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT",
        result={"owner_nonce_capsule": _capsule()},
        operation_input={},
    )

    invocation, _, _ = _execute_task(
        definition=definition,
        state_name="SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY",
        state_input={"task12_last_result": arm_next_result},
        workflow="snapshot",
        domain_result={},
    )

    assert invocation.operation_kind == (
        "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY"
    )
    assert invocation.operation_input == {
        "task12_last_result": arm_next_result
    }


def test_snapshot_catch_preserves_exact_ambiguity_and_atomic_prior() -> None:
    """Break caught: Catch drops the ambiguity or feeds the failed send result."""

    from glm52_enforcement import task12_lambda_adapters as adapters

    definition = _definitions()["snapshot"]
    atomic_result = adapters._result(
        handler_kind="RETAINED_SNAPSHOT_CLEANUP",
        outcome="SUCCEEDED",
        operation_kind=(
            "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT"
        ),
        result={"owner_nonce_capsule": _capsule()},
        operation_input={},
    )
    before_send = {"task12_last_result": atomic_result}
    catch = definition["States"][
        "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
    ]["Catch"][0]
    ambiguity = {
        "Error": "Lambda.ServiceException",
        "Cause": "authenticated timeout after same-ID send",
    }
    caught = _merge(before_send, catch["ResultPath"], ambiguity)
    invocation, _, _ = _execute_task(
        definition=definition,
        state_name=catch["Next"],
        state_input=caught,
        workflow="snapshot",
        domain_result={"snapshot_present": True},
    )

    assert invocation.operation_kind == (
        "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"
    )
    assert invocation.operation_input == {
        "task12_last_result": atomic_result,
        "delete_ambiguity": ambiguity,
    }
    _assert_no_raw_nonce(invocation.operation_input)


@pytest.mark.parametrize(
    "bad_input",
    [
        {},
        {"task12_last_result": None},
        {"task12_last_result": {"operation_kind": "foreign"}},
    ],
)
def test_second_task_rejects_absent_or_malformed_prior_result(
    bad_input: dict[str, object],
) -> None:
    """Break caught: an edge runs without the authenticated predecessor result."""

    definition = _definitions()["retained"]
    with pytest.raises(
        ValueError,
        match="operation input prior ResultPath is foreign",
    ):
        _execute_task(
            definition=definition,
            state_name="RETAINED_ACQUIRE_RECOVERY_SEALING",
            state_input=bad_input,
            workflow="retained",
            domain_result={},
        )
