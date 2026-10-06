from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from glm52_enforcement.dynamodb import (
    DynamoLedgerAdapter,
    ExactCheck,
    ExactPut,
    ExactUpdate,
    LedgerKey,
    WriteOutcome,
    decode_item,
)
from glm52_enforcement.records import canonical_record_identity
from glm52_enforcement.task12_retained_state import (
    FinalizationProgressPlan,
    RecoveryProgressPlan,
    RecoverySealPlan,
    RetainedOwnerTakeoverPlan,
    SnapshotCleanupTransitionPlan,
    TeardownSealPlan,
    build_finalization_progress_plan,
    build_recovery_progress_plan,
    build_recovery_seal_plan,
    build_retained_owner_takeover_plan,
    build_snapshot_cleanup_transition_plan,
    build_teardown_seal_plan,
)

_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_task12_dynamodb_fixtures",
    Path(__file__).with_name("test_glm52_enforcement_dynamodb.py"),
)
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
_FIXTURES = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURES)
AwsShapedSimulator = _FIXTURES.AwsShapedSimulator
SimulatedClientError = _FIXTURES.SimulatedClientError
RUN_ID = _FIXTURES.RUN_ID
_activation_index = _FIXTURES._activation_index
_closed_record = _FIXTURES._closed_record
_control = _FIXTURES._control
_rehash = _FIXTURES._rehash


def test_task12_retained_state_exports_typed_plan_constructors() -> None:
    assert build_recovery_seal_plan.__annotations__["return"] == (
        "RecoverySealPlan"
    )
    assert build_recovery_progress_plan.__annotations__["return"] == (
        "RecoveryProgressPlan"
    )
    assert build_teardown_seal_plan.__annotations__["return"] == (
        "TeardownSealPlan"
    )
    assert build_finalization_progress_plan.__annotations__["return"] == (
        "FinalizationProgressPlan"
    )
    assert build_retained_owner_takeover_plan.__annotations__["return"] == (
        "RetainedOwnerTakeoverPlan"
    )
    assert build_snapshot_cleanup_transition_plan.__annotations__["return"] == (
        "SnapshotCleanupTransitionPlan"
    )
    assert all(
        plan_type.__dataclass_params__.frozen
        for plan_type in (
            RecoverySealPlan,
            RecoveryProgressPlan,
            TeardownSealPlan,
            FinalizationProgressPlan,
            RetainedOwnerTakeoverPlan,
            SnapshotCleanupTransitionPlan,
        )
    )


def _recovery_seal_plan() -> RecoverySealPlan:
    execution = _closed_record(
        "glm52_production_execution",
        activation_id="activation-1",
        state="SUCCEEDED",
    )
    control_before = _control()
    control_after = dict(control_before)
    control_after.update(
        phase="RECOVERY_SEALING",
        revision=2,
        updated_at="2026-07-28T12:00:01Z",
    )
    recovery_before = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
    )
    recovery_after = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
        state="OWNED",
        revision=2,
        owner_execution_arn="retained-recovery-execution",
        owner_state_machine_version_arn="retained-recovery-version",
        support_control_revision_at_seal=2,
        support_execution_identity_sha256=canonical_record_identity(
            "glm52_production_execution", execution
        ),
    )
    return build_recovery_seal_plan(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        support_execution=ExactCheck(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-1#EXECUTION#00000001",
            ),
            execution,
        ),
        control=ExactUpdate(
            LedgerKey(RUN_ID, "ACTIVATION#activation-1#CONTROL"),
            control_before,
            control_after,
        ),
        recovery_control=ExactUpdate(
            LedgerKey(
                RUN_ID, "ACTIVATION#activation-1#RECOVERY_CONTROL"
            ),
            recovery_before,
            recovery_after,
        ),
    )


def test_recovery_seal_is_one_transaction_with_live_nonce_readback() -> None:
    client = AwsShapedSimulator()
    plan = _recovery_seal_plan()
    for item in plan.transaction_plans:
        if type(item) is ExactCheck:
            client.install(item.key.sort_key, dict(item.expected))
        elif type(item) is ExactUpdate:
            client.install(item.key.sort_key, dict(item.before))
    raw_nonce = b"r" * 32

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_recovery_seal(
        plan=plan,
        domain="TASK12_RECOVERY_SEAL",
        operation_identity_sha256="9" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records[-1]["owner_invocation_nonce_sha256"] == (
        hashlib.sha256(raw_nonce).hexdigest()
    )
    assert [name for name, _ in client.calls] == [
        "transact_write_items",
        "transact_get_items",
    ]
    write = client.calls[0][1]
    assert [next(iter(item)) for item in write["TransactItems"]] == [
        "ConditionCheck",
        "ConditionCheck",
        "Update",
        "Update",
    ]
    assert raw_nonce not in repr(write).encode()


def test_recovery_seal_rejects_nonterminal_support_execution() -> None:
    plan = _recovery_seal_plan()
    running = _closed_record(
        "glm52_production_execution",
        activation_id="activation-1",
        state="RUNNING",
    )
    with pytest.raises(ValueError, match="terminal support execution"):
        build_recovery_seal_plan(
            index=plan.index,
            support_execution=ExactCheck(
                plan.support_execution.key, running
            ),
            control=plan.control,
            recovery_control=plan.recovery_control,
        )


def test_recovery_seal_rejects_control_after_teardown() -> None:
    plan = _recovery_seal_plan()
    before = dict(plan.control.before)
    before.update(phase="TEARDOWN_SEALED")
    with pytest.raises(ValueError):
        build_recovery_seal_plan(
            index=plan.index,
            support_execution=plan.support_execution,
            control=ExactUpdate(
                plan.control.key, before, plan.control.after
            ),
            recovery_control=plan.recovery_control,
        )


def test_takeover_rejects_cleared_recovery_evidence() -> None:
    plan = _recovery_seal_plan()
    owner_before = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
        state="TERMINAL_V2_PUBLISHED",
        revision=3,
    )
    owner_after = deepcopy(owner_before)
    owner_after.update(
        owner_attempt=2,
        owner_execution_arn="new-owner",
        owner_state_machine_version_arn="new-version",
        owner_dispatch_identity_sha256="b" * 64,
        owner_invocation_nonce_sha256="c" * 64,
        owner_hard_expires_at="2026-07-28T13:00:00Z",
        terminal_v2_identity_sha256=None,
        revision=4,
        updated_at="2026-07-28T12:00:02Z",
    )
    control = dict(plan.control.after)
    with pytest.raises(ValueError):
        build_retained_owner_takeover_plan(
            index=plan.index,
            control=ExactCheck(plan.control.key, control),
            prior_owner_execution=plan.support_execution,
            action_guards=(),
            owner=ExactUpdate(
                plan.recovery_control.key, owner_before, owner_after
            ),
        )


def _snapshot_arm_plan(
    raw_nonce: bytes,
) -> SnapshotCleanupTransitionPlan:
    index = ExactCheck(
        LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
    )
    cleanup_before = _closed_record(
        "glm52_production_snapshot_cleanup_control",
        activation_id="activation-1",
    )
    cleanup_after = _closed_record(
        "glm52_production_snapshot_cleanup_control",
        activation_id="activation-1",
        state="ARMED",
        revision=2,
    )
    chain_before = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
        state="RECOVERY_COMPLETE",
        revision=4,
        cleanup_control_root_identity_sha256="2" * 64,
        cleanup_transition_chain_head_sha256="1" * 64,
        cleanup_transition_chain_length=1,
    )
    authority = _closed_record(
        "glm52_production_finalization_control",
        activation_id="activation-1",
        state="SUPPORT_FINALIZED",
        revision=3,
        owner_invocation_nonce_sha256=hashlib.sha256(
            raw_nonce
        ).hexdigest(),
    )
    transition = _closed_record(
        "glm52_production_snapshot_cleanup_transition",
        activation_id="activation-1",
        from_state="DORMANT",
        to_state="ARMED",
        from_revision=1,
        to_revision=2,
        cleanup_control_root_identity_sha256="2" * 64,
        prior_transition_sha256="1" * 64,
        owner_attempt=None,
        owner_execution_arn=None,
        owner_state_machine_version_arn=None,
        owner_dispatch_identity_sha256=None,
        owner_invocation_nonce_sha256=None,
        action_identity_sha256=None,
    )
    chain_after = dict(chain_before)
    chain_after.update(
        cleanup_transition_chain_head_sha256=transition[
            "canonical_body_sha256"
        ],
        cleanup_transition_chain_length=2,
        revision=5,
        updated_at="2026-07-28T12:00:01Z",
    )
    return build_snapshot_cleanup_transition_plan(
        index=index,
        cleanup_chain=ExactUpdate(
            LedgerKey(
                RUN_ID, "ACTIVATION#activation-1#RECOVERY_CONTROL"
            ),
            chain_before,
            chain_after,
        ),
        authority=ExactCheck(
            LedgerKey(
                RUN_ID, "ACTIVATION#activation-1#FINALIZATION_CONTROL"
            ),
            authority,
        ),
        cleanup_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_CONTROL",
            ),
            cleanup_before,
            cleanup_after,
        ),
        cleanup_transition=ExactPut(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-1#"
                "SNAPSHOT_CLEANUP_TRANSITION#00000002",
            ),
            transition,
        ),
    )


def test_snapshot_arm_advances_chain_with_immutable_transition_atomically() -> None:
    raw_nonce = b"s" * 32
    plan = _snapshot_arm_plan(raw_nonce)
    client = AwsShapedSimulator()
    for item in plan.transaction_plans:
        if type(item) is ExactCheck:
            client.install(item.key.sort_key, dict(item.expected))
        elif type(item) is ExactUpdate:
            client.install(item.key.sort_key, dict(item.before))

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_snapshot_cleanup_transition(
        plan=plan,
        domain="TASK12_SNAPSHOT_ARM",
        operation_identity_sha256="8" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert [next(iter(item)) for item in client.calls[0][1][
        "TransactItems"
    ]] == [
        "ConditionCheck",
        "Update",
        "ConditionCheck",
        "Update",
        "Put",
    ]
    assert result.records[1]["cleanup_transition_chain_length"] == 2
    assert result.records[-1]["to_state"] == "ARMED"


def test_snapshot_transition_rejects_a_chain_fork_before_writing() -> None:
    plan = _snapshot_arm_plan(b"s" * 32)
    transition = deepcopy(plan.cleanup_transition.item)
    transition["prior_transition_sha256"] = "f" * 64
    _rehash(transition)
    with pytest.raises(ValueError, match="chain mismatch"):
        build_snapshot_cleanup_transition_plan(
            index=plan.index,
            cleanup_chain=plan.cleanup_chain,
            authority=plan.authority,
            cleanup_control=plan.cleanup_control,
            cleanup_transition=ExactPut(
                plan.cleanup_transition.key, transition
            ),
        )


def test_snapshot_cleanup_rejects_ownership_before_delete_deadline() -> None:
    deadline = "2026-08-04T12:00:00Z"
    cleanup_before = _closed_record(
        "glm52_production_snapshot_cleanup_control",
        activation_id="activation-1",
        state="ARMED",
        revision=2,
        delete_not_before=deadline,
    )
    cleanup_after = _closed_record(
        "glm52_production_snapshot_cleanup_control",
        activation_id="activation-1",
        state="OWNED",
        revision=3,
        delete_not_before=deadline,
    )
    transition = _closed_record(
        "glm52_production_snapshot_cleanup_transition",
        activation_id="activation-1",
        from_state="ARMED",
        to_state="OWNED",
        from_revision=2,
        to_revision=3,
        cleanup_control_root_identity_sha256="2" * 64,
        prior_transition_sha256="1" * 64,
        owner_attempt=cleanup_after["owner_attempt"],
        owner_execution_arn=cleanup_after["owner_execution_arn"],
        owner_state_machine_version_arn=cleanup_after[
            "owner_state_machine_version_arn"
        ],
        owner_dispatch_identity_sha256=cleanup_after[
            "owner_dispatch_identity_sha256"
        ],
        owner_invocation_nonce_sha256=cleanup_after[
            "owner_invocation_nonce_sha256"
        ],
        action_identity_sha256=None,
        transitioned_at="2026-07-28T12:00:00Z",
    )
    chain_before = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
        state="RECOVERY_COMPLETE",
        revision=5,
        cleanup_control_root_identity_sha256="2" * 64,
        cleanup_transition_chain_head_sha256="1" * 64,
        cleanup_transition_chain_length=2,
    )
    chain_after = dict(chain_before)
    chain_after.update(
        cleanup_transition_chain_head_sha256=transition[
            "canonical_body_sha256"
        ],
        cleanup_transition_chain_length=3,
        revision=6,
        updated_at="2026-07-28T12:00:01Z",
    )
    with pytest.raises(ValueError, match="before its deadline"):
        build_snapshot_cleanup_transition_plan(
            index=ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
                _activation_index(),
            ),
            cleanup_chain=ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    "ACTIVATION#activation-1#RECOVERY_CONTROL",
                ),
                chain_before,
                chain_after,
            ),
            authority=ExactCheck(
                LedgerKey(
                    RUN_ID, "ACTIVATION#activation-1#CONTROL"
                ),
                _control(phase="TEARDOWN_SEALED", revision=5),
            ),
            cleanup_control=ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_CONTROL",
                ),
                cleanup_before,
                cleanup_after,
            ),
            cleanup_transition=ExactPut(
                LedgerKey(
                    RUN_ID,
                    "ACTIVATION#activation-1#"
                    "SNAPSHOT_CLEANUP_TRANSITION#00000003",
                ),
                transition,
            ),
        )


def test_recovery_seal_malformed_success_requires_coherent_readback() -> None:
    client = AwsShapedSimulator()
    client.malformed_write_response = True
    plan = _recovery_seal_plan()
    for item in plan.transaction_plans:
        if type(item) is ExactCheck:
            client.install(item.key.sort_key, dict(item.expected))
        elif type(item) is ExactUpdate:
            client.install(item.key.sort_key, dict(item.before))

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_recovery_seal(
        plan=plan,
        domain="TASK12_RECOVERY_SEAL",
        operation_identity_sha256="7" * 64,
        raw_owner_nonce=b"m" * 32,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.error_code == "MalformedResponse"
    assert [name for name, _ in client.calls] == [
        "transact_write_items",
        "transact_get_items",
    ]


def test_recovery_seal_condition_rejection_is_not_retried() -> None:
    client = AwsShapedSimulator()
    client.pre_write_fault = SimulatedClientError(
        "TransactionCanceledException"
    )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_recovery_seal(
        plan=_recovery_seal_plan(),
        domain="TASK12_RECOVERY_SEAL",
        operation_identity_sha256="6" * 64,
        raw_owner_nonce=b"c" * 32,
    )

    assert result.outcome is WriteOutcome.CONDITION_REJECTED
    assert [name for name, _ in client.calls] == ["transact_write_items"]


def test_lost_recovery_seal_response_rejects_foreign_nonce_readback() -> None:
    client = AwsShapedSimulator()
    client.post_commit_fault = TimeoutError("lost write response")
    plan = _recovery_seal_plan()
    for item in plan.transaction_plans:
        if type(item) is ExactCheck:
            client.install(item.key.sort_key, dict(item.expected))
        elif type(item) is ExactUpdate:
            client.install(item.key.sort_key, dict(item.before))

    def replace_owner_with_foreign_nonce() -> None:
        key = (
            "RUN#" + RUN_ID,
            plan.recovery_control.key.sort_key,
        )
        physical = decode_item(client.items[key])
        record = dict(physical)
        record.pop("PK")
        record.pop("SK")
        record["owner_invocation_nonce_sha256"] = "f" * 64
        client.install(plan.recovery_control.key.sort_key, record)

    client.before_read_hook = replace_owner_with_foreign_nonce
    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_recovery_seal(
        plan=plan,
        domain="TASK12_RECOVERY_SEAL",
        operation_identity_sha256="5" * 64,
        raw_owner_nonce=b"n" * 32,
    )

    assert result.outcome is WriteOutcome.FOREIGN_NONCE
    assert result.may_issue_external_side_effect is False
    assert [name for name, _ in client.calls] == [
        "transact_write_items",
        "transact_get_items",
    ]


def test_retained_takeover_preserves_progress_and_uses_new_live_nonce() -> None:
    raw_nonce = b"t" * 32
    owner_before = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
        state="TERMINAL_V2_PUBLISHED",
        revision=3,
        owner_execution_arn="old-retained-execution",
        owner_state_machine_version_arn="old-retained-version",
    )
    owner_after = deepcopy(owner_before)
    owner_after.update(
        owner_attempt=2,
        owner_execution_arn="new-retained-execution",
        owner_state_machine_version_arn="new-retained-version",
        owner_dispatch_identity_sha256="b" * 64,
        owner_invocation_nonce_sha256="c" * 64,
        owner_hard_expires_at="2026-07-28T13:00:00Z",
        revision=4,
        updated_at="2026-07-28T12:00:02Z",
    )
    old_execution = _closed_record(
        "glm52_production_execution",
        activation_id="activation-1",
        state="SUCCEEDED",
        expected_execution_arn="old-retained-execution",
        expected_state_machine_version_arn="old-retained-version",
    )
    plan = build_retained_owner_takeover_plan(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        control=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION#activation-1#CONTROL"),
            _control(phase="RECOVERY_SEALING", revision=2),
        ),
        prior_owner_execution=ExactCheck(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-1#EXECUTION#00000001",
            ),
            old_execution,
        ),
        action_guards=(),
        owner=ExactUpdate(
            LedgerKey(
                RUN_ID, "ACTIVATION#activation-1#RECOVERY_CONTROL"
            ),
            owner_before,
            owner_after,
        ),
    )
    client = AwsShapedSimulator()
    for item in plan.transaction_plans:
        if type(item) is ExactCheck:
            client.install(item.key.sort_key, dict(item.expected))
        else:
            client.install(item.key.sort_key, dict(item.before))

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_retained_owner_takeover(
        plan=plan,
        domain="TASK12_RECOVERY_TAKEOVER",
        operation_identity_sha256="4" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records[-1]["terminal_v2_identity_sha256"] == (
        owner_before["terminal_v2_identity_sha256"]
    )
    assert result.records[-1]["owner_invocation_nonce_sha256"] == (
        hashlib.sha256(raw_nonce).hexdigest()
    )


def _finalization_progress_plan(
    raw_nonce: bytes,
) -> FinalizationProgressPlan:
    control = _control(phase="TEARDOWN_SEALED", revision=5)
    before = _closed_record(
        "glm52_production_finalization_control",
        activation_id="activation-1",
        state="OWNED",
        revision=2,
        teardown_sealed_control_revision=5,
        owner_invocation_nonce_sha256=hashlib.sha256(
            raw_nonce
        ).hexdigest(),
    )
    after = _closed_record(
        "glm52_production_finalization_control",
        activation_id="activation-1",
        state="SUPPORT_FINALIZED",
        revision=3,
        teardown_sealed_control_revision=5,
        owner_invocation_nonce_sha256=hashlib.sha256(
            raw_nonce
        ).hexdigest(),
    )
    return build_finalization_progress_plan(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        control=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION#activation-1#CONTROL"),
            control,
        ),
        finalization_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-1#FINALIZATION_CONTROL",
            ),
            before,
            after,
        ),
    )


def test_finalization_progress_is_ordered_and_one_transaction() -> None:
    raw_nonce = b"f" * 32
    plan = _finalization_progress_plan(raw_nonce)
    client = AwsShapedSimulator()
    for item in plan.transaction_plans:
        if type(item) is ExactCheck:
            client.install(item.key.sort_key, dict(item.expected))
        else:
            client.install(item.key.sort_key, dict(item.before))

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_finalization_progress(
        plan=plan,
        domain="TASK12_FINALIZATION_PROGRESS",
        operation_identity_sha256="3" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records[-1]["state"] == "SUPPORT_FINALIZED"
    assert len(client.calls[0][1]["TransactItems"]) == 3


def test_finalization_progress_rejects_skipped_evidence_state() -> None:
    plan = _finalization_progress_plan(b"f" * 32)
    skipped = _closed_record(
        "glm52_production_finalization_control",
        activation_id="activation-1",
        state="SNAPSHOT_DISPOSITION_RECORDED",
        revision=3,
        teardown_sealed_control_revision=5,
        owner_invocation_nonce_sha256=plan.finalization_control.before[
            "owner_invocation_nonce_sha256"
        ],
    )
    with pytest.raises(ValueError):
        build_finalization_progress_plan(
            index=plan.index,
            control=plan.control,
            finalization_control=ExactUpdate(
                plan.finalization_control.key,
                plan.finalization_control.before,
                skipped,
            ),
        )
