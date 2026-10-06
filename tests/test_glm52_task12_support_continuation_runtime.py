"""Executable support-owned post-decision continuation proofs."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.task12_correlation import (
    NumericJobBinding,
    RequestCorrelation,
    Task11HandoffAuthority,
)
from glm52_enforcement.task12_support_continuation_adapter import (
    SUPPORT_OPERATION_SEQUENCE,
)
from glm52_enforcement.task12_support_continuation_runtime import (
    SupportEffect,
    ZeroRequestProof,
    execute_support_operation,
)


def _effect(kind: str, **evidence: object) -> SupportEffect:
    body = {
        "effect_kind": kind,
        "outcome": "SUCCEEDED",
        "evidence": evidence,
    }
    return SupportEffect(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


class _FakeAws:
    def __init__(
        self,
        cardinality: str,
        *,
        calls: list[str] | None = None,
    ) -> None:
        self.cardinality = cardinality
        self.calls = [] if calls is None else calls

    def hydrate(self, state: object) -> None:
        assert state.handoff.body_sha256 == "2" * 64

    def read_approved_task11_handoff(
        self, context: object
    ) -> Task11HandoffAuthority:
        self.calls.append("read_handoff")
        assert context["decision_result"]["status"] == "CLOSED"
        return Task11HandoffAuthority(
            bucket="keep-fixture-246813579024-us-west-2",
            key=(
                "campaigns/glm52-sky-20260724/submissions/production/"
                "generations/00000001/handoff/SKY_POST_HANDOFF.json"
            ),
            version_id="version-1",
            file_sha256="1" * 64,
            body={
                "run_id": "glm52-sky-20260724",
                "generation": 1,
                "expected_sky_job_name": "glm52-job",
                "handoff_body_sha256": "2" * 64,
            },
            body_sha256="2" * 64,
            approved_task11_workflow_version_arn=(
                "arn:aws:states:us-west-2:246813579024:"
                "stateMachine:keep-h1g:1"
            ),
            task11_execution_arn=(
                "arn:aws:states:us-west-2:246813579024:"
                "execution:keep-h1g:execution-1"
            ),
        )

    def materialize_correlation_tuple(
        self, authority: Task11HandoffAuthority
    ) -> dict[str, object]:
        self.calls.append("numeric_handoff")
        assert authority.body_sha256 == "2" * 64
        return {"expected_sky_job_name": "glm52-job"}

    def correlate_request(
        self,
        authority: Task11HandoffAuthority,
        correlation_tuple: object,
    ) -> RequestCorrelation:
        self.calls.append("correlate")
        assert correlation_tuple == {"expected_sky_job_name": "glm52-job"}
        if self.cardinality == "ZERO":
            return RequestCorrelation(kind="ZERO", matches=())
        from glm52_enforcement.task12_correlation import CorrelatedRequest

        count = 2 if self.cardinality == "MULTIPLE" else 1
        return RequestCorrelation(
            kind=self.cardinality,
            matches=tuple(
                CorrelatedRequest(
                    request_id=f"request-{index}",
                    state="RUNNING",
                    body={
                        "request_id": f"request-{index}",
                        "state": "RUNNING",
                    },
                )
                for index in range(1, count + 1)
            ),
        )

    def prove_zero_request(
        self,
        authority: Task11HandoffAuthority,
        correlation: RequestCorrelation,
    ) -> ZeroRequestProof:
        self.calls.append("prove_zero_request")
        assert correlation.kind == "ZERO"
        body = {
            "handoff_identity_sha256": authority.body_sha256,
            "pagination_complete": True,
            "first_observed_at": "2026-07-29T00:00:00Z",
            "second_observed_at": "2026-07-29T00:01:00Z",
            "request_ids": (),
            "job_ids": (),
            "worker_instance_ids": (),
            "allocation_ids": (),
            "controller_work_ids": (),
        }
        return ZeroRequestProof(
            **body,
            canonical_identity_sha256=canonical_sha256(body),
        )

    def read_numeric_binding(
        self, request_id: str
    ) -> dict[str, object]:
        self.calls.append("bind_job")
        return {
            "request_id": request_id,
            "numeric_job_id": 42,
            "sky_job_name": "glm52-job",
            "state": "RUNNING",
        }

    def observe_runtime(self, state: object) -> SupportEffect:
        self.calls.append("observe_runtime")
        return _effect(
            "RUNTIME_OBSERVATION",
            request_cardinality=state.request_cardinality,
            numeric_job_id=(
                None if state.binding is None else state.binding.numeric_job_id
            ),
        )

    def arm_drain(self, state: object) -> SupportEffect:
        self.calls.append("arm_drain")
        return _effect("DRAIN_ARMED", phase="DRAIN_ARMED")

    def request_drain(self, state: object) -> SupportEffect:
        self.calls.append("request_drain")
        return _effect("DRAIN_REQUESTED", phase="DRAIN_REQUESTED")

    def reconcile_request_cancel(self, state: object) -> SupportEffect:
        self.calls.append("cancel_request")
        return _effect("REQUEST_CANCEL", request_id="request-1")

    def reconcile_job_cancel(self, state: object) -> SupportEffect:
        self.calls.append("cancel_job")
        return _effect("JOB_CANCEL", numeric_job_id=42)

    def quiesce_controller(self, state: object) -> SupportEffect:
        self.calls.append("quiesce_controller")
        return _effect("CONTROLLER_QUIESCED", ec2_state="stopped")

    def reconcile_workers(self, state: object) -> SupportEffect:
        self.calls.append("reconcile_workers")
        return _effect(
            "WORKER_TERMINAL",
            instances=(
                ()
                if self.cardinality == "ZERO"
                else ("i-0123456789abcdef0",)
            ),
        )

    def close_allocations(self, state: object) -> SupportEffect:
        self.calls.append("close_allocations")
        return _effect(
            "ALLOCATION_CLOSED",
            allocations=(() if self.cardinality == "ZERO" else ("allocation-1",)),
        )

    def commit_allocation_closed(self, state: object) -> None:
        self.calls.append("commit_allocation_closed")
        assert state.phase == "ALLOCATION_CLOSED"
        assert state.effects[-1].effect_kind == "ALLOCATION_CLOSED"

    def create_terminal_v2(self, state: object) -> SupportEffect:
        self.calls.append("terminal_v2")
        return _effect(
            "TERMINAL_V2",
            object_version_id="terminal-version-1",
            request_cardinality=state.request_cardinality,
        )

    def request_finalization(self, state: object) -> SupportEffect:
        self.calls.append("finalization")
        return _effect(
            "FINALIZATION_REQUESTED",
            execution_terminal_required=True,
        )


def _root_event(operation: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_task12_support_continuation_invocation_v1",
        "operation_kind": operation,
        "operation_input": {
            "task11_context": {
                "closure_request": {
                    "activation_id": "activation-1",
                    "generation": 1,
                },
                "decision_result": {
                    "status": "CLOSED",
                    "canonical_identity_sha256": "3" * 64,
                },
                "task11_execution_arn": (
                    "arn:aws:states:us-west-2:246813579024:"
                    "execution:keep-h1g:execution-1"
                ),
            }
        },
    }


def _run_path(services: _FakeAws) -> dict[str, object]:
    result = execute_support_operation(
        _root_event(SUPPORT_OPERATION_SEQUENCE[0]),
        services=services,
    )
    for operation in SUPPORT_OPERATION_SEQUENCE[1:]:
        result = execute_support_operation(
            {
                "schema_version": 1,
                "record_type": (
                    "glm52_task12_support_continuation_invocation_v1"
                ),
                "operation_kind": operation,
                "operation_input": {"support_last_result": result},
            },
            services=services,
        )
    return result


def _run_fresh_path(
    cardinality: str,
) -> tuple[dict[str, object], list[str]]:
    calls: list[str] = []
    result = execute_support_operation(
        _root_event(SUPPORT_OPERATION_SEQUENCE[0]),
        services=_FakeAws(cardinality, calls=calls),
    )
    for operation in SUPPORT_OPERATION_SEQUENCE[1:]:
        result = execute_support_operation(
            {
                "schema_version": 1,
                "record_type": (
                    "glm52_task12_support_continuation_invocation_v1"
                ),
                "operation_kind": operation,
                "operation_input": {"support_last_result": result},
            },
            services=_FakeAws(cardinality, calls=calls),
        )
    return result, calls


@pytest.mark.parametrize("cardinality", ["ZERO", "ONE"])
def test_all_13_support_states_execute_real_effects(cardinality: str) -> None:
    services = _FakeAws(cardinality)
    result = _run_path(services)

    assert result["operation_kind"] == "REQUEST_SUPPORT_FINALIZATION"
    assert result["result"]["continuation_state"]["phase"] == (
        "FINALIZATION_REQUESTED"
    )
    assert services.calls == [
        "read_handoff",
        "numeric_handoff",
        "correlate",
        *(
            ["prove_zero_request"]
            if cardinality == "ZERO"
            else (["bind_job"] if cardinality == "ONE" else [])
        ),
        "observe_runtime",
        "arm_drain",
        "request_drain",
        *(
            []
            if cardinality == "ZERO"
            else ["cancel_request", "cancel_job"]
        ),
        "quiesce_controller",
        "reconcile_workers",
        "close_allocations",
        "commit_allocation_closed",
        "terminal_v2",
        "finalization",
    ]

    state = result["result"]["continuation_state"]
    assert state["request_cardinality"] == cardinality
    assert len(state["effects"]) == (
        10 if cardinality == "ONE" else 8
    )
    if cardinality == "ZERO":
        assert state["binding"] is None
        assert state["zero_request_proof"]["pagination_complete"] is True
    elif cardinality == "ONE":
        assert state["binding"] == asdict(
            NumericJobBinding(
                request_id="request-1",
                numeric_job_id=42,
                sky_job_name="glm52-job",
                state="RUNNING",
            )
        )
        assert state["zero_request_proof"] is None


def test_multiple_request_correlation_fails_before_any_cancel_mutation() -> None:
    services = _FakeAws("MULTIPLE")
    result = execute_support_operation(
        _root_event(SUPPORT_OPERATION_SEQUENCE[0]),
        services=services,
    )
    result = execute_support_operation(
        {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_support_continuation_invocation_v1"
            ),
            "operation_kind": "NUMERIC_BINDING_HANDOFF",
            "operation_input": {"support_last_result": result},
        },
        services=services,
    )

    with pytest.raises(
        ValueError,
        match="multiple request correlation is a sealed incident",
    ):
        execute_support_operation(
            {
                "schema_version": 1,
                "record_type": (
                    "glm52_task12_support_continuation_invocation_v1"
                ),
                "operation_kind": "BIND_NUMERIC_JOB_OR_PROVE_NONE",
                "operation_input": {"support_last_result": result},
            },
            services=services,
        )

    assert services.calls == ["read_handoff", "numeric_handoff", "correlate"]


@pytest.mark.parametrize("cardinality", ["ZERO", "ONE"])
def test_all_13_states_rehydrate_in_fresh_lambda_processes(
    cardinality: str,
) -> None:
    result, calls = _run_fresh_path(cardinality)

    assert result["result"]["continuation_state"]["phase"] == (
        "FINALIZATION_REQUESTED"
    )
    assert calls[0:3] == ["read_handoff", "numeric_handoff", "correlate"]
    assert calls[-2:] == ["terminal_v2", "finalization"]
