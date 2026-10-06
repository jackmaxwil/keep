from __future__ import annotations

from copy import deepcopy

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.dynamodb import (
    ExactCheck,
    ExactPut,
    ExactUpdate,
    LedgerKey,
    TransactionResolution,
    WriteOutcome,
)
from glm52_enforcement.task12_retained_state import (
    SnapshotCleanupTransitionPlan,
)
from glm52_enforcement.task12_snapshot_cleanup import (
    CleanupTransitionProof,
    DeleteAttemptResult,
    ReconciliationResult,
    SnapshotCapture,
    SnapshotCleanupCoordinator,
    SnapshotObservation,
    SnapshotSchedule,
    verify_three_activation_lineage,
)


SHA = "a" * 64
RUN_ID = "glm52-sky-20260724"


def _observation(**overrides: object) -> SnapshotObservation:
    value = {
        "snapshot_id": "snap-0123456789abcdef0",
        "source_volume_id": "vol-0123456789abcdef0",
        "kms_key_arn": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-1234-1234-123456789012"
        ),
        "snapshot_tags_sha256": "1" * 64,
        "encrypted": True,
        "state": "completed",
        "observed_at": "2026-07-28T12:00:00Z",
        "describe_request_id": "request-1",
        "describe_response_sha256": "2" * 64,
    }
    value.update(overrides)
    return SnapshotObservation(**value)


def test_snapshot_capture_selects_one_exact_completed_snapshot() -> None:
    observation = _observation()

    capture = SnapshotCapture.discover(
        observations=(observation,),
        expected_source_volume_id=observation.source_volume_id,
        expected_kms_key_arn=observation.kms_key_arn,
        expected_snapshot_tags_sha256=observation.snapshot_tags_sha256,
    )

    assert capture.snapshot_id == observation.snapshot_id
    assert capture.snapshot_identity_sha256 == canonical_sha256(
        {
            "schema_version": 1,
            "snapshot_id": observation.snapshot_id,
            "source_volume_id": observation.source_volume_id,
            "kms_key_arn": observation.kms_key_arn,
            "snapshot_tags_sha256": observation.snapshot_tags_sha256,
            "encrypted": True,
        }
    )
    assert capture.capture_evidence_sha256 == canonical_sha256(
        {
            "schema_version": 1,
            "snapshot_id": observation.snapshot_id,
            "source_volume_id": observation.source_volume_id,
            "kms_key_arn": observation.kms_key_arn,
            "snapshot_tags_sha256": observation.snapshot_tags_sha256,
            "encrypted": True,
            "state": observation.state,
            "observed_at": observation.observed_at,
            "describe_request_id": observation.describe_request_id,
            "describe_response_sha256": (
                observation.describe_response_sha256
            ),
        }
    )


@pytest.mark.parametrize(
    "mutant",
    [
        "zero",
        "multiple",
        "foreign-volume",
        "foreign-kms",
        "foreign-tags",
        "unencrypted",
        "pending",
    ],
)
def test_snapshot_capture_rejects_discovery_mutants(mutant: str) -> None:
    expected = _observation()
    observations = (expected,)
    if mutant == "zero":
        observations = ()
    elif mutant == "multiple":
        observations = (expected, expected)
    elif mutant == "foreign-volume":
        observations = (
            _observation(source_volume_id="vol-fffffffffffffffff"),
        )
    elif mutant == "foreign-kms":
        observations = (
            _observation(
                kms_key_arn=(
                    "arn:aws:kms:us-west-2:246813579024:key/"
                    "ffffffff-ffff-ffff-ffff-ffffffffffff"
                )
            ),
        )
    elif mutant == "foreign-tags":
        observations = (_observation(snapshot_tags_sha256="3" * 64),)
    elif mutant == "unencrypted":
        observations = (_observation(encrypted=False),)
    elif mutant == "pending":
        observations = (_observation(state="pending"),)

    with pytest.raises(ValueError):
        SnapshotCapture.discover(
            observations=observations,
            expected_source_volume_id=expected.source_volume_id,
            expected_kms_key_arn=expected.kms_key_arn,
            expected_snapshot_tags_sha256=expected.snapshot_tags_sha256,
        )


def test_snapshot_schedule_is_exactly_seven_days_and_parameterless() -> None:
    capture = SnapshotCapture.discover(
        observations=(_observation(),),
        expected_source_volume_id=_observation().source_volume_id,
        expected_kms_key_arn=_observation().kms_key_arn,
        expected_snapshot_tags_sha256=_observation().snapshot_tags_sha256,
    )

    schedule = SnapshotSchedule.create(
        capture=capture,
        schedule_arn=(
            "arn:aws:scheduler:us-west-2:246813579024:"
            "schedule/glm52-cleanup"
        ),
    )

    assert schedule.delete_not_before == "2026-08-04T12:00:00Z"
    assert schedule.schedule_input_sha256 == canonical_sha256({})
    assert schedule.schedule_identity_sha256 == canonical_sha256(
        {
            "schema_version": 1,
            "snapshot_identity_sha256": capture.snapshot_identity_sha256,
            "schedule_arn": schedule.schedule_arn,
            "delete_not_before": schedule.delete_not_before,
            "schedule_input_sha256": schedule.schedule_input_sha256,
        }
    )


def test_snapshot_schedule_rejects_wrong_deadline_or_input() -> None:
    capture = SnapshotCapture.discover(
        observations=(_observation(),),
        expected_source_volume_id=_observation().source_volume_id,
        expected_kms_key_arn=_observation().kms_key_arn,
        expected_snapshot_tags_sha256=_observation().snapshot_tags_sha256,
    )
    with pytest.raises(ValueError):
        SnapshotSchedule.validate(
            capture=capture,
            schedule_arn="arn:schedule",
            delete_not_before="2026-08-03T12:00:00Z",
            schedule_input_sha256=canonical_sha256({}),
        )
    with pytest.raises(ValueError):
        SnapshotSchedule.validate(
            capture=capture,
            schedule_arn="arn:schedule",
            delete_not_before="2026-08-04T12:00:00Z",
            schedule_input_sha256=canonical_sha256({"snapshot_id": "foreign"}),
        )


def _lineage_entry(
    activation_ordinal: int,
    root: str,
    transitions: tuple[str, ...],
) -> dict[str, object]:
    return {
        "activation_ordinal": activation_ordinal,
        "cleanup_control_root_identity_sha256": root,
        "cleanup_transition_chain_head_sha256": (
            root if not transitions else transitions[-1]
        ),
        "cleanup_transition_chain_length": len(transitions),
        "transition_identities": transitions,
    }


def _transition_proof(
    *,
    root: str,
    prior: str,
    revision: int,
    activation_ordinal: int,
) -> CleanupTransitionProof:
    record = {
        "schema_version": 1,
        "record_type": "glm52_production_snapshot_cleanup_transition",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": RUN_ID,
        "campaign_identity_sha256": SHA,
        "activation_id": "activation-" + str(activation_ordinal),
        "activation_ordinal": activation_ordinal,
        "cleanup_control_root_identity_sha256": root,
        "from_state": "DELETE_RECONCILING",
        "to_state": "DELETE_POSSIBLY_SENT",
        "from_revision": revision,
        "to_revision": revision + 1,
        "prior_transition_sha256": prior,
        "owner_attempt": 1,
        "owner_execution_arn": "execution-arn",
        "owner_state_machine_version_arn": "version-arn",
        "owner_dispatch_identity_sha256": "7" * 64,
        "owner_invocation_nonce_sha256": "8" * 64,
        "authority_audit_identity_sha256": "9" * 64,
        "action_identity_sha256": "d" * 64,
        "transitioned_at": "2026-08-04T12:00:00Z",
        "canonical_body_sha256": SHA,
    }
    body = dict(record)
    body.pop("canonical_body_sha256")
    record["canonical_body_sha256"] = canonical_sha256(body)
    return CleanupTransitionProof.from_record(record)


def _proof_chain(
    root: str, count: int, activation_ordinal: int
) -> tuple[CleanupTransitionProof, ...]:
    prior = root
    result = []
    for revision in range(1, count + 1):
        proof = _transition_proof(
            root=root,
            prior=prior,
            revision=revision,
            activation_ordinal=activation_ordinal,
        )
        result.append(proof)
        prior = proof.identity_sha256
    return tuple(result)


def _three_activation_lineage_arguments() -> dict[str, object]:
    first_root = "a" * 64
    second_root = "b" * 64
    first_live = _proof_chain(first_root, 4, 1)
    second_live = _proof_chain(second_root, 2, 2)
    first_at_second = _lineage_entry(
        1,
        first_root,
        tuple(proof.identity_sha256 for proof in first_live[:2]),
    )
    first_at_third = _lineage_entry(
        1,
        first_root,
        tuple(proof.identity_sha256 for proof in first_live[:3]),
    )
    second_at_third = _lineage_entry(
        2,
        second_root,
        (second_live[0].identity_sha256,),
    )
    third = (first_at_third, second_at_third)
    return {
        "second_activation_lineage": (first_at_second,),
        "third_activation_lineage": third,
        "third_lineage_sha256": canonical_sha256(
            [
                {
                    **entry,
                    "transition_identities": list(
                        entry["transition_identities"]
                    ),
                }
                for entry in third
            ]
        ),
        "live_transition_chains": {
            1: first_live,
            2: second_live,
        },
    }


def test_three_activation_lineage_preserves_transitive_descendants() -> None:
    arguments = _three_activation_lineage_arguments()
    verified = verify_three_activation_lineage(**arguments)

    assert verified.recorded_activation_ordinals == (1, 2)
    assert verified.live_chain_heads == (
        (
            1,
            arguments["live_transition_chains"][1][-1].identity_sha256,
        ),
        (
            2,
            arguments["live_transition_chains"][2][-1].identity_sha256,
        ),
    )


@pytest.mark.parametrize(
    "mutant",
    [
        "omitted-ancestor",
        "fork",
        "wrong-hash",
        "live-gap",
        "wrong-root",
        "cycle",
    ],
)
def test_three_activation_lineage_rejects_permanent_mutants(
    mutant: str,
) -> None:
    arguments = _three_activation_lineage_arguments()
    if mutant == "omitted-ancestor":
        arguments["third_activation_lineage"] = (
            arguments["third_activation_lineage"][1],
        )
    elif mutant == "fork":
        third = deepcopy(arguments["third_activation_lineage"])
        transitions = third[0]["transition_identities"]
        third[0]["transition_identities"] = (
            transitions[0],
            "f" * 64,
            transitions[2],
        )
        third[0]["cleanup_transition_chain_head_sha256"] = transitions[2]
        arguments["third_activation_lineage"] = third
    elif mutant == "wrong-hash":
        arguments["third_lineage_sha256"] = "f" * 64
    elif mutant == "live-gap":
        chains = dict(arguments["live_transition_chains"])
        chains[1] = (
            chains[1][0],
            _transition_proof(
                root="a" * 64,
                prior="f" * 64,
                revision=2,
                activation_ordinal=1,
            ),
        )
        arguments["live_transition_chains"] = chains
    elif mutant == "wrong-root":
        chains = dict(arguments["live_transition_chains"])
        chains[2] = _proof_chain("c" * 64, 1, 2)
        arguments["live_transition_chains"] = chains
    elif mutant == "cycle":
        chains = dict(arguments["live_transition_chains"])
        chains[2] = (chains[2][0], chains[2][0])
        arguments["live_transition_chains"] = chains

    with pytest.raises(ValueError):
        verify_three_activation_lineage(**arguments)


class FakeRetainedAdapter:
    def __init__(
        self,
        outcome: WriteOutcome = WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
    ) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def commit_snapshot_cleanup_transition(
        self, **kwargs: object
    ) -> TransactionResolution:
        self.calls.append(kwargs)
        return TransactionResolution(
            self.outcome, (), "ddb-request", None, ()
        )


class FakeSnapshotClient:
    def __init__(
        self,
        *,
        delete_result: object = None,
        describe_result: object = None,
    ) -> None:
        self.delete_result = (
            {
                "outcome": "ACCEPTED",
                "request_id": "delete-request",
                "response_sha256": "d" * 64,
            }
            if delete_result is None
            else delete_result
        )
        self.describe_result = describe_result
        self.calls: list[tuple[str, str]] = []

    def delete_snapshot(self, *, snapshot_id: str) -> object:
        self.calls.append(("delete_snapshot", snapshot_id))
        if isinstance(self.delete_result, BaseException):
            raise self.delete_result
        return self.delete_result

    def describe_snapshot(self, *, snapshot_id: str) -> object:
        self.calls.append(("describe_snapshot", snapshot_id))
        return self.describe_result


def _capture_and_schedule() -> tuple[SnapshotCapture, SnapshotSchedule]:
    capture = SnapshotCapture.discover(
        observations=(_observation(),),
        expected_source_volume_id=_observation().source_volume_id,
        expected_kms_key_arn=_observation().kms_key_arn,
        expected_snapshot_tags_sha256=_observation().snapshot_tags_sha256,
    )
    return capture, SnapshotSchedule.create(
        capture=capture, schedule_arn="arn:schedule"
    )


def _typed_plan(
    *,
    before_state: str,
    after_state: str,
    before_attempt: int = 0,
    after_attempt: int | None = None,
    before_call_count: int = 0,
    after_call_count: int | None = None,
    mutate_after: dict[str, object] | None = None,
) -> SnapshotCleanupTransitionPlan:
    capture, schedule = _capture_and_schedule()
    before = {
        "state": before_state,
        "snapshot_id": capture.snapshot_id,
        "snapshot_identity_sha256": capture.snapshot_identity_sha256,
        "source_volume_id": capture.source_volume_id,
        "snapshot_tags_sha256": capture.snapshot_tags_sha256,
        "delete_not_before": schedule.delete_not_before,
        "schedule_arn": schedule.schedule_arn,
        "schedule_input_sha256": schedule.schedule_input_sha256,
        "delete_logical_attempt": before_attempt,
        "delete_call_count": before_call_count,
    }
    after = dict(before)
    after.update(
        state=after_state,
        delete_logical_attempt=(
            before_attempt if after_attempt is None else after_attempt
        ),
        delete_call_count=(
            before_call_count
            if after_call_count is None
            else after_call_count
        ),
    )
    if mutate_after:
        after.update(mutate_after)
    key = LedgerKey(RUN_ID, "SNAPSHOT_CLEANUP_CONTROL")
    return SnapshotCleanupTransitionPlan(
        index=ExactCheck(LedgerKey(RUN_ID, "INDEX"), {}),
        cleanup_chain=ExactUpdate(
            LedgerKey(RUN_ID, "CHAIN"), {}, {}
        ),
        authority=ExactCheck(LedgerKey(RUN_ID, "AUTHORITY"), {}),
        cleanup_control=ExactUpdate(key, before, after),
        cleanup_transition=ExactPut(
            LedgerKey(RUN_ID, "TRANSITION"), {}
        ),
    )


def _coordinator(
    *,
    outcome: WriteOutcome = WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
    delete_result: object = None,
    describe_result: object = None,
) -> tuple[
    SnapshotCleanupCoordinator,
    FakeRetainedAdapter,
    FakeSnapshotClient,
]:
    adapter = FakeRetainedAdapter(outcome)
    snapshots = FakeSnapshotClient(
        delete_result=delete_result,
        describe_result=describe_result,
    )
    return (
        SnapshotCleanupCoordinator(
            adapter=adapter, snapshot_client=snapshots
        ),
        adapter,
        snapshots,
    )


def _operation() -> dict[str, object]:
    return {
        "domain": "TASK12_SNAPSHOT_CLEANUP",
        "operation_identity_sha256": "9" * 64,
        "raw_owner_nonce": b"n" * 32,
    }


def test_arm_and_acquire_use_typed_plan_and_condition_race() -> None:
    capture, schedule = _capture_and_schedule()
    coordinator, adapter, snapshots = _coordinator(
        outcome=WriteOutcome.CONDITION_REJECTED
    )
    arm = _typed_plan(before_state="DORMANT", after_state="ARMED")

    result = coordinator.arm(
        plan=arm,
        capture=capture,
        schedule=schedule,
        **_operation(),
    )
    acquire = _typed_plan(before_state="ARMED", after_state="OWNED")
    race = coordinator.acquire(
        plan=acquire,
        observed_at=schedule.delete_not_before,
        **_operation(),
    )

    assert result.outcome is WriteOutcome.CONDITION_REJECTED
    assert race.outcome is WriteOutcome.CONDITION_REJECTED
    assert len(adapter.calls) == 2
    assert all(
        type(call["plan"]) is SnapshotCleanupTransitionPlan
        for call in adapter.calls
    )
    assert snapshots.calls == []


def test_acquire_rejects_early_or_foreign_snapshot_without_adapter_call() -> None:
    coordinator, adapter, _ = _coordinator()
    plan = _typed_plan(before_state="ARMED", after_state="OWNED")
    with pytest.raises(ValueError, match="deadline"):
        coordinator.acquire(
            plan=plan,
            observed_at="2026-08-04T11:59:59Z",
            **_operation(),
        )
    foreign = _typed_plan(
        before_state="ARMED",
        after_state="OWNED",
        mutate_after={"snapshot_id": "snap-fffffffffffffffff"},
    )
    with pytest.raises(ValueError, match="snapshot binding"):
        coordinator.acquire(
            plan=foreign,
            observed_at="2026-08-04T12:00:00Z",
            **_operation(),
        )
    assert adapter.calls == []


def test_delete_once_calls_exact_snapshot_once_and_never_retries_ambiguity() -> None:
    coordinator, adapter, snapshots = _coordinator(
        delete_result=TimeoutError("ambiguous")
    )
    plan = _typed_plan(
        before_state="OWNED",
        after_state="DELETE_POSSIBLY_SENT",
        after_attempt=1,
        after_call_count=1,
    )

    result = coordinator.delete_once(
        plan=plan,
        observed_at="2026-08-04T12:00:00Z",
        **_operation(),
    )

    assert type(result) is DeleteAttemptResult
    assert result.outcome == "AMBIGUOUS"
    assert len(adapter.calls) == 1
    assert snapshots.calls == [
        ("delete_snapshot", _observation().snapshot_id)
    ]


def test_delete_once_blocks_unresolved_or_thirteenth_attempt() -> None:
    coordinator, adapter, snapshots = _coordinator()
    unresolved = _typed_plan(
        before_state="DELETE_POSSIBLY_SENT",
        after_state="DELETE_POSSIBLY_SENT",
        before_attempt=1,
        after_attempt=2,
        before_call_count=1,
        after_call_count=2,
    )
    with pytest.raises(ValueError, match="readback"):
        coordinator.delete_once(
            plan=unresolved,
            observed_at="2026-08-04T12:00:00Z",
            **_operation(),
        )
    exhausted = _typed_plan(
        before_state="DELETE_RECONCILING",
        after_state="DELETE_POSSIBLY_SENT",
        before_attempt=12,
        after_attempt=13,
        before_call_count=12,
        after_call_count=13,
    )
    with pytest.raises(ValueError, match="12"):
        coordinator.delete_once(
            plan=exhausted,
            observed_at="2026-08-04T12:00:00Z",
            **_operation(),
        )
    assert adapter.calls == []
    assert snapshots.calls == []


@pytest.mark.parametrize(
    ("before", "present", "attempts", "target"),
    [
        ("OWNED", False, 0, "ALREADY_ABSENT"),
        ("DELETE_POSSIBLY_SENT", False, 1, "DELETE_RECONCILING"),
        ("DELETE_RECONCILING", False, 1, "DELETED"),
        ("DELETE_RECONCILING", True, 12, "CLEANUP_INCIDENT"),
    ],
)
def test_readback_commits_only_closed_reconciliation_target(
    before: str, present: bool, attempts: int, target: str
) -> None:
    observation = _observation() if present else None
    coordinator, adapter, snapshots = _coordinator(
        describe_result=observation
    )
    plan = _typed_plan(
        before_state=before,
        after_state=target,
        before_attempt=attempts,
        after_attempt=attempts,
        before_call_count=attempts,
        after_call_count=attempts,
    )

    result = coordinator.reconcile_readback(
        plan=plan,
        **_operation(),
    )

    assert type(result) is ReconciliationResult
    assert result.target_state == target
    assert len(adapter.calls) == 1
    assert snapshots.calls == [
        ("describe_snapshot", _observation().snapshot_id)
    ]


def test_readback_present_before_limit_requires_new_closed_attempt() -> None:
    coordinator, adapter, snapshots = _coordinator(
        describe_result=_observation()
    )
    result = coordinator.reconcile_readback(
        plan=None,
        current_control=_typed_plan(
            before_state="DELETE_RECONCILING",
            after_state="DELETED",
            before_attempt=2,
        ).cleanup_control.before,
        **_operation(),
    )

    assert result.target_state == "DELETE_RECONCILING"
    assert result.committed is False
    assert adapter.calls == []
    assert len(snapshots.calls) == 1


@pytest.mark.parametrize(
    "observation",
    [
        _observation(snapshot_id="snap-fffffffffffffffff"),
        _observation(
            kms_key_arn=(
                "arn:aws:kms:us-west-2:246813579024:key/"
                "ffffffff-ffff-ffff-ffff-ffffffffffff"
            )
        ),
    ],
)
def test_readback_rejects_foreign_snapshot_without_commit(
    observation: SnapshotObservation,
) -> None:
    coordinator, adapter, _ = _coordinator(
        describe_result=observation
    )
    with pytest.raises(ValueError, match="foreign snapshot"):
        coordinator.reconcile_readback(
            plan=None,
            current_control=_typed_plan(
                before_state="DELETE_RECONCILING",
                after_state="DELETED",
            ).cleanup_control.before,
            **_operation(),
        )
    assert adapter.calls == []
