from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest


SHA = "a" * 64
SHA_B = "b" * 64


def _entity(
    identity: str,
    state: str,
    *,
    observed_at: str = "2026-07-29T12:00:00Z",
):
    from glm52_enforcement.task12_runtime import RuntimeEntity

    return RuntimeEntity(
        identity=identity,
        state=state,
        observed_at=observed_at,
        evidence_identity_sha256=SHA,
    )


def _read(
    domain: str,
    members=(),
    *,
    observed_at: str = "2026-07-29T12:00:00Z",
    complete: bool = True,
):
    from glm52_enforcement.task12_runtime import CompleteRuntimeRead

    return CompleteRuntimeRead(
        domain=domain,
        members=tuple(members),
        observed_at=observed_at,
        pagination_complete=complete,
        evidence_identity_sha256=SHA,
    )


def _spend(
    *,
    state: str = "CLOSED",
    observed_at: str = "2026-07-29T12:00:00Z",
):
    from glm52_enforcement.task12_runtime import SpendRead

    return SpendRead(
        state=state,
        ledger_head_identity_sha256=SHA,
        remaining_gpu_seconds=0,
        remaining_gpu_cost_usd="0.00",
        observed_at=observed_at,
        evidence_identity_sha256=SHA,
    )


def _scan(*, observed_at: str, worker_state: str = "terminated"):
    from glm52_enforcement.task12_runtime import RuntimeScan

    return RuntimeScan(
        requests=_read(
            "REQUEST",
            (_entity("request-1", "CANCELLED", observed_at=observed_at),),
            observed_at=observed_at,
        ),
        jobs=_read(
            "JOB",
            (_entity("17", "CANCELLED", observed_at=observed_at),),
            observed_at=observed_at,
        ),
        controller=_read("CONTROLLER", observed_at=observed_at),
        workers=_read(
            "WORKER",
            (_entity("i-0123456789abcdef0", worker_state, observed_at=observed_at),),
            observed_at=observed_at,
        ),
        allocations=_read(
            "ALLOCATION",
            (_entity("allocation-1", "CLOSED", observed_at=observed_at),),
            observed_at=observed_at,
        ),
        spend=_spend(observed_at=observed_at),
    )


def _worker_descriptor(
    *,
    execution_deadline: str = "2026-07-31T12:00:00Z",
):
    from glm52_enforcement.task10_worker import (
        build_worker_bootstrap_descriptor,
    )

    return build_worker_bootstrap_descriptor(
        schema_version=1,
        record_type="glm52_task10_worker_bootstrap_descriptor_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        campaign_identity_sha256=SHA,
        activation_id="activation-0001",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        action_key="ACTION#00000001#SKY_POST#00000001",
        sky_job_name="glm52-sky-20260724",
        execution_deadline=execution_deadline,
        gpu_allocation_sha256=SHA_B,
        base_descriptor_s3_uri=(
            "s3://keep-glm52-models-246813579024-us-west-2/"
            "campaigns/glm52-sky-20260724/descriptor.json"
        ),
        base_descriptor_version_id="descriptor-version-1",
        base_descriptor_file_sha256=SHA,
        base_descriptor_body_sha256=SHA_B,
        archive_identity_sha256=SHA,
        repository_archive_version_id="archive-version-1",
        approval_identity_sha256=SHA_B,
        approval_version_id="approval-version-1",
        intent_identity_sha256=SHA,
        intent_version_id="intent-version-1",
        task8_live_h1d_identity_sha256=SHA_B,
        task8_spend_authority_identity_sha256=SHA,
        task9_launch_identity_sha256=SHA_B,
        task9_admission_identity_sha256=SHA,
        task9_custody_identity_sha256=SHA_B,
    )


def _deadline_wake() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_task12_retained_deadline_wake_v1",
        "activation_id": "activation-0001",
        "generation": 1,
        "reason": "SCHEDULE_ACCELERATOR",
    }


class _ReadBoundary:
    def __init__(self, method: str, value: object) -> None:
        setattr(self, method, lambda correlation_identity_sha256: value)


def test_runtime_collection_uses_six_closed_read_protocols() -> None:
    from glm52_enforcement.task12_runtime import (
        RuntimeReadBoundaries,
        collect_runtime_scan,
    )

    scan = _scan(observed_at="2026-07-29T12:00:00Z")
    boundaries = RuntimeReadBoundaries(
        request_reader=_ReadBoundary("read_requests", scan.requests),
        job_reader=_ReadBoundary("read_jobs", scan.jobs),
        controller_reader=_ReadBoundary("read_controller", scan.controller),
        worker_reader=_ReadBoundary("read_workers", scan.workers),
        allocation_reader=_ReadBoundary("read_allocations", scan.allocations),
        spend_reader=_ReadBoundary("read_spend", scan.spend),
    )

    assert collect_runtime_scan(
        boundaries=boundaries,
        correlation_identity_sha256=SHA_B,
    ) == scan


def test_runtime_collection_rejects_incomplete_or_wrong_domain_read() -> None:
    from glm52_enforcement.task12_runtime import (
        RuntimeReadBoundaries,
        Task12RuntimeError,
        collect_runtime_scan,
    )

    scan = _scan(observed_at="2026-07-29T12:00:00Z")
    boundaries = RuntimeReadBoundaries(
        request_reader=_ReadBoundary(
            "read_requests", replace(scan.requests, pagination_complete=False)
        ),
        job_reader=_ReadBoundary("read_jobs", scan.jobs),
        controller_reader=_ReadBoundary("read_controller", scan.controller),
        worker_reader=_ReadBoundary("read_workers", scan.workers),
        allocation_reader=_ReadBoundary("read_allocations", scan.allocations),
        spend_reader=_ReadBoundary("read_spend", scan.spend),
    )

    with pytest.raises(Task12RuntimeError, match="complete"):
        collect_runtime_scan(
            boundaries=boundaries,
            correlation_identity_sha256=SHA_B,
        )


def test_cancellation_command_is_closed_and_ambiguous_send_never_rearms() -> None:
    from glm52_enforcement.task12_runtime import (
        CancellationObservation,
        Task12RuntimeError,
        build_cancellation_command,
        reconcile_cancellation,
    )

    command = build_cancellation_command(
        kind="REQUEST_CANCEL",
        target_id="request-1",
        action_key="RECOVERY_ACTION#REQUEST_CANCEL#request-1",
        invocation_nonce_sha256=SHA,
        correlation_identity_sha256=SHA_B,
    )
    assert command.method == "POST"
    assert command.path == "/api/cancel"
    assert command.body == {"request_id": "request-1"}
    assert command.total_max_attempts == 1

    reconciled = reconcile_cancellation(
        command=command,
        observation=CancellationObservation(
            classification="AMBIGUOUS",
            send_count=1,
            response_request_id=None,
            response_identity_sha256=None,
            observed_target_state="RUNNING",
            observation_identity_sha256=SHA,
        ),
    )
    assert reconciled.state == "AMBIGUOUS_RECONCILE_ONLY"
    assert reconciled.may_resend is False

    with pytest.raises(Task12RuntimeError, match="route"):
        reconcile_cancellation(
            command=replace(command, path="/jobs/launch"),
            observation=CancellationObservation(
                classification="AMBIGUOUS",
                send_count=1,
                response_request_id=None,
                response_identity_sha256=None,
                observed_target_state="RUNNING",
                observation_identity_sha256=SHA,
            ),
        )


def test_cancellation_rejects_launch_route_and_second_send() -> None:
    from glm52_enforcement.task12_runtime import (
        CancellationObservation,
        Task12RuntimeError,
        build_cancellation_command,
        reconcile_cancellation,
    )

    with pytest.raises(Task12RuntimeError, match="kind"):
        build_cancellation_command(
            kind="LAUNCH",
            target_id="17",
            action_key="RECOVERY_ACTION#LAUNCH",
            invocation_nonce_sha256=SHA,
            correlation_identity_sha256=SHA_B,
        )

    command = build_cancellation_command(
        kind="JOB_CANCEL",
        target_id="17",
        action_key="RECOVERY_ACTION#JOB_CANCEL#17",
        invocation_nonce_sha256=SHA,
        correlation_identity_sha256=SHA_B,
    )
    assert command.path == "/jobs/cancel"
    with pytest.raises(Task12RuntimeError, match="at most one"):
        reconcile_cancellation(
            command=command,
            observation=CancellationObservation(
                classification="DIRECT",
                send_count=2,
                response_request_id="cancel-request-1",
                response_identity_sha256=SHA,
                observed_target_state="CANCELLED",
                observation_identity_sha256=SHA_B,
            ),
        )


def test_force_termination_requires_controller_quiescence_and_two_scans() -> None:
    from glm52_enforcement.task12_runtime import (
        ControllerStopEvidence,
        Task12RuntimeError,
        authorize_forced_worker_termination,
        build_retained_deadline_authority,
        evaluate_retained_deadline,
        prove_controller_quiescence,
    )

    stop = ControllerStopEvidence(
        instance_id="i-0fedcba9876543210",
        ec2_state="stopped",
        stopped_at="2026-07-29T12:00:00Z",
        ssm_offline=True,
        controller_processes_unreachable=True,
        start_instances_denied=True,
        replacement_denied=True,
        new_launch_denied=True,
        evidence_identity_sha256=SHA,
    )
    first = _scan(
        observed_at="2026-07-29T12:01:00Z",
        worker_state="running",
    )
    second = _scan(
        observed_at="2026-07-29T12:02:00Z",
        worker_state="running",
    )
    proof = prove_controller_quiescence(
        stop=stop,
        first=first,
        second=second,
        minimum_quiet_seconds=60,
    )
    deadline_authority = build_retained_deadline_authority(
        worker_descriptor=_worker_descriptor(),
        wake=_deadline_wake(),
    )
    deadline_decision = evaluate_retained_deadline(
        authority=deadline_authority,
        observed_at="2026-07-31T11:30:00Z",
    )
    authority = authorize_forced_worker_termination(
        quiescence=proof,
        deadline_decision=deadline_decision,
        instance_ids=("i-0123456789abcdef0",),
    )
    assert authority.instance_ids == ("i-0123456789abcdef0",)
    assert authority.deadline_decision_identity_sha256 == (
        deadline_decision.canonical_identity_sha256
    )
    with pytest.raises(Task12RuntimeError, match="identity"):
        authorize_forced_worker_termination(
            quiescence=replace(proof, canonical_identity_sha256=SHA_B),
            deadline_decision=deadline_decision,
            instance_ids=("i-0123456789abcdef0",),
        )
    with pytest.raises(Task12RuntimeError, match="T-minus-30"):
        authorize_forced_worker_termination(
            quiescence=proof,
            deadline_decision=evaluate_retained_deadline(
                authority=deadline_authority,
                observed_at="2026-07-31T11:29:59Z",
            ),
            instance_ids=("i-0123456789abcdef0",),
        )
    with pytest.raises(Task12RuntimeError, match="T-minus-30"):
        authorize_forced_worker_termination(
            quiescence=proof,
            deadline_decision=evaluate_retained_deadline(
                authority=deadline_authority,
                observed_at="2026-07-31T11:29:59Z",
                prior_decision=deadline_decision,
            ),
            instance_ids=("i-0123456789abcdef0",),
        )

    with pytest.raises(Task12RuntimeError, match="separated"):
        prove_controller_quiescence(
            stop=stop,
            first=first,
            second=_scan(observed_at="2026-07-29T12:01:30Z"),
            minimum_quiet_seconds=60,
        )


def test_retained_deadline_authority_ignores_wake_timing_and_rejects_substitution() -> None:
    from glm52_enforcement.task12_runtime import (
        Task12RuntimeError,
        build_retained_deadline_authority,
    )

    descriptor = _worker_descriptor()
    authority = build_retained_deadline_authority(
        worker_descriptor=descriptor,
        wake=_deadline_wake(),
    )

    assert authority.descriptor_body_sha256 == descriptor.descriptor_body_sha256
    assert authority.execution_deadline == "2026-07-31T12:00:00Z"
    assert authority.stop_assignment_at == "2026-07-31T11:00:00Z"
    assert authority.graceful_stop_at == "2026-07-31T11:10:00Z"
    assert authority.retained_force_not_before == "2026-07-31T11:30:00Z"

    substituted = {
        **_deadline_wake(),
        "execution_deadline": "2026-08-01T12:00:00Z",
    }
    with pytest.raises(Task12RuntimeError, match="wake schema"):
        build_retained_deadline_authority(
            worker_descriptor=descriptor,
            wake=substituted,
        )
    with pytest.raises(Task12RuntimeError, match="wake authority"):
        build_retained_deadline_authority(
            worker_descriptor=descriptor,
            wake={**_deadline_wake(), "activation_id": "activation-0002"},
        )
    with pytest.raises(Task12RuntimeError, match="wake authority"):
        build_retained_deadline_authority(
            worker_descriptor=descriptor,
            wake={**_deadline_wake(), "schema_version": True},
        )


@pytest.mark.parametrize(
    ("observed_at", "phase"),
    [
        ("2026-07-31T10:59:59Z", "BEFORE_T_MINUS_60"),
        ("2026-07-31T11:00:00Z", "T_MINUS_60"),
        ("2026-07-31T11:10:00Z", "T_MINUS_50"),
        ("2026-07-31T11:30:00Z", "T_MINUS_30"),
    ],
)
def test_retained_deadline_decision_uses_only_derived_absolute_edges(
    observed_at: str,
    phase: str,
) -> None:
    from glm52_enforcement.task12_runtime import (
        build_retained_deadline_authority,
        evaluate_retained_deadline,
    )

    authority = build_retained_deadline_authority(
        worker_descriptor=_worker_descriptor(),
        wake=_deadline_wake(),
    )
    decision = evaluate_retained_deadline(
        authority=authority,
        observed_at=observed_at,
    )

    assert decision.phase == phase
    assert decision.allow_new_assignments is (
        phase == "BEFORE_T_MINUS_60"
    )
    assert decision.request_graceful_stop is (
        phase in {"T_MINUS_50", "T_MINUS_30"}
    )
    assert decision.allow_controller_quiesced_force is (
        phase == "T_MINUS_30"
    )


def test_retained_deadline_replay_and_clock_rollback_never_regress() -> None:
    from glm52_enforcement.task12_runtime import (
        Task12RuntimeError,
        build_retained_deadline_authority,
        evaluate_retained_deadline,
    )

    authority = build_retained_deadline_authority(
        worker_descriptor=_worker_descriptor(),
        wake={**_deadline_wake(), "reason": "RECOVERY_REPLAY"},
    )
    at_t50 = evaluate_retained_deadline(
        authority=authority,
        observed_at="2026-07-31T11:10:00Z",
    )
    rolled_back = evaluate_retained_deadline(
        authority=authority,
        observed_at="2026-07-31T10:59:59Z",
        prior_decision=at_t50,
    )
    replay = evaluate_retained_deadline(
        authority=authority,
        observed_at="2026-07-31T10:59:59Z",
        prior_decision=at_t50,
    )

    assert rolled_back.phase == "T_MINUS_50"
    assert rolled_back == replay
    assert rolled_back.allow_new_assignments is False
    assert rolled_back.request_graceful_stop is True
    with pytest.raises(Task12RuntimeError, match="decision identity"):
        evaluate_retained_deadline(
            authority=authority,
            observed_at="2026-07-31T10:59:59Z",
            prior_decision=replace(
                at_t50,
                canonical_identity_sha256=SHA,
            ),
        )


def test_quiescence_rejects_new_worker_and_nonterminal_worker() -> None:
    from glm52_enforcement.task12_runtime import (
        ControllerStopEvidence,
        Task12RuntimeError,
        prove_controller_quiescence,
    )

    stop = ControllerStopEvidence(
        instance_id="i-0fedcba9876543210",
        ec2_state="terminated",
        stopped_at="2026-07-29T12:00:00Z",
        ssm_offline=True,
        controller_processes_unreachable=True,
        start_instances_denied=True,
        replacement_denied=True,
        new_launch_denied=True,
        evidence_identity_sha256=SHA,
    )
    first = _scan(
        observed_at="2026-07-29T12:01:00Z",
        worker_state="running",
    )
    second = _scan(
        observed_at="2026-07-29T12:02:00Z",
        worker_state="running",
    )
    second = replace(
        second,
        workers=_read(
            "WORKER",
            (
                    _entity(
                        "i-0123456789abcdef0",
                        "running",
                        observed_at="2026-07-29T12:02:00Z",
                    ),
                _entity(
                    "i-0aaaaaaaaaaaaaaaa",
                    "running",
                    observed_at="2026-07-29T12:02:00Z",
                ),
            ),
            observed_at="2026-07-29T12:02:00Z",
        ),
    )

    with pytest.raises(Task12RuntimeError, match="new or nonterminal worker"):
        prove_controller_quiescence(
            stop=stop,
            first=first,
            second=second,
            minimum_quiet_seconds=60,
        )


@pytest.mark.parametrize(
    "state",
    ["PENDING", "WAITING", "RUNNING", "UNREADABLE", "SUCCEEDED"],
)
def test_no_job_never_collapses_nonfailure_request_state(state: str) -> None:
    from glm52_enforcement.task12_runtime import (
        Task12RuntimeError,
        prove_no_job_terminal,
    )

    empty_first = replace(
        _scan(observed_at="2026-07-29T12:01:00Z"),
        requests=_read("REQUEST", observed_at="2026-07-29T12:01:00Z"),
        jobs=_read("JOB", observed_at="2026-07-29T12:01:00Z"),
        workers=_read("WORKER", observed_at="2026-07-29T12:01:00Z"),
        allocations=_read("ALLOCATION", observed_at="2026-07-29T12:01:00Z"),
    )
    empty_second = replace(
        _scan(observed_at="2026-07-29T12:02:00Z"),
        requests=_read("REQUEST", observed_at="2026-07-29T12:02:00Z"),
        jobs=_read("JOB", observed_at="2026-07-29T12:02:00Z"),
        workers=_read("WORKER", observed_at="2026-07-29T12:02:00Z"),
        allocations=_read("ALLOCATION", observed_at="2026-07-29T12:02:00Z"),
    )
    with pytest.raises(Task12RuntimeError, match="FAILED or CANCELLED"):
        prove_no_job_terminal(
            request=_entity("request-1", state),
            first=empty_first,
            second=empty_second,
            minimum_quiet_seconds=60,
        )


def test_terminal_runtime_proof_and_typed_task9_settlement_handoff() -> None:
    from glm52_enforcement.launch_custody import LiabilitySettlementCoordinator
    from glm52_enforcement.task12_runtime import (
        build_task9_liability_settlement_handoff,
        prove_runtime_terminal,
        settle_task9_liability,
    )

    first = _scan(observed_at="2026-07-29T12:01:00Z")
    second = _scan(observed_at="2026-07-29T12:02:00Z")
    terminal = prove_runtime_terminal(
        first=first,
        second=second,
        minimum_quiet_seconds=60,
    )
    handoff = build_task9_liability_settlement_handoff(
        activation_id="activation-0001",
        allocation_ordinal=1,
        terminal_v2_identity_sha256=SHA,
        runtime_terminal=terminal,
    )

    class _Coordinator(LiabilitySettlementCoordinator):
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def settle(self, activation_id: str, allocation_ordinal: int):
            self.calls.append((activation_id, allocation_ordinal))
            return {
                "activation_id": activation_id,
                "allocation_ordinal": allocation_ordinal,
                "canonical_body_sha256": SHA_B,
            }

    coordinator = _Coordinator()
    result = settle_task9_liability(
        handoff=handoff,
        coordinator=coordinator,
    )
    assert result["canonical_body_sha256"] == SHA_B
    assert coordinator.calls == [("activation-0001", 1)]


def test_runtime_clock_inputs_are_canonical_utc() -> None:
    from glm52_enforcement.task12_runtime import Task12RuntimeError

    with pytest.raises(Task12RuntimeError, match="canonical UTC"):
        _entity("request-1", "FAILED", observed_at="2026-07-29 12:00:00")
    assert datetime.now(timezone.utc).tzinfo is not None
