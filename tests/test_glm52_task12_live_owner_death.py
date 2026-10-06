from __future__ import annotations

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.dynamodb import (
    DynamoLedgerAdapter,
    ExactCheck,
    ExactUpdate,
    LedgerKey,
    WriteOutcome,
)
from glm52_enforcement.records import ledger_sk
from glm52_enforcement.task12_live_owner_death import (
    Task12OwnerDeathError,
    build_owner_death_request,
)

from test_glm52_enforcement_dynamodb import (
    AwsShapedSimulator,
    SimulatedClientError,
    _activation_index,
    _closed_record,
    _control,
)


RUN_ID = "glm52-sky-20260724"
SHA = "a" * 64
PAST = "2026-07-28T10:00:00Z"
NOW = "2026-07-28T12:00:00Z"
FUTURE = "2026-07-28T14:00:00Z"
ACTIVATION_EXECUTION = (
    "arn:aws:states:us-west-2:246813579024:execution:"
    "keep-glm52-h1g:activation-1"
)
ACTIVATION_VERSION = (
    "arn:aws:states:us-west-2:246813579024:stateMachine:"
    "keep-glm52-h1g:1"
)
RECOVERY_EXECUTION = (
    "arn:aws:states:us-west-2:246813579024:execution:"
    "keep-glm52-h1g-recovery:recovery-1"
)
RECOVERY_VERSION = (
    "arn:aws:states:us-west-2:246813579024:stateMachine:"
    "keep-glm52-h1g-recovery:1"
)


def _capsule() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "ciphertext_base64": "Y2lwaGVydGV4dA==",
        "nonce_sha256": SHA,
        "encryption_context": {},
    }
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def _invocation() -> SimpleNamespace:
    return SimpleNamespace(
        operation_kind="RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION",
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        state_machine_execution_arn=RECOVERY_EXECUTION,
        caller_state_machine_version_arn=RECOVERY_VERSION,
        operation_input={
            "task12_last_result": {
                "result": {"owner_nonce_capsule": _capsule()}
            }
        },
    )


def _sources(
    state: str, *, recovery_nonce_sha256: str = SHA
) -> dict[str, dict[str, object]]:
    action_key = ledger_sk(
        "glm52_production_action",
        activation_id="activation-1",
        generation=1,
        action_kind="SKY_POST",
        attempt=1,
    )
    action_overrides: dict[str, object] = {
        "state": state,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "action_kind": "SKY_POST",
        "attempt": 1,
        "owner_epoch": 1,
        "owner_execution_arn": ACTIVATION_EXECUTION,
        "armed_by_epoch": 1,
        "armed_by_execution_arn": ACTIVATION_EXECUTION,
        "armed_by_state_machine_version_arn": ACTIVATION_VERSION,
        "arming_hard_expires_at": PAST,
    }
    if state in {"POST_STARTED", "POST_AUTHORIZED"}:
        action_overrides["post_owner_hard_expires_at"] = PAST
    if state == "POST_CLASSIFIED":
        action_overrides.update(
            outcome_class="ACCEPTED",
            classification_evidence_kind="RELAY_RESPONSE",
            classification_evidence_body_sha256="b" * 64,
            response_identity_sha256="c" * 64,
            sky_request_id="sky-request-1",
        )
    action = _closed_record(
        "glm52_production_action", **action_overrides
    )
    control = _control(
        phase="RECOVERY_SEALING",
        active_epoch=1,
        active_execution_arn=ACTIVATION_EXECUTION,
        active_state_machine_version_arn=ACTIVATION_VERSION,
        last_sky_post_generation=1,
        last_sky_post_action_key=action_key,
        last_sky_post_state=state,
    )
    recovery = _closed_record(
        "glm52_production_recovery_control",
        state="OWNED",
        activation_id="activation-1",
        activation_ordinal=1,
        owner_execution_arn=RECOVERY_EXECUTION,
        owner_state_machine_version_arn=RECOVERY_VERSION,
        owner_hard_expires_at=FUTURE,
        owner_invocation_nonce_sha256=recovery_nonce_sha256,
        terminal_v2_identity_sha256=None,
    )
    execution = _closed_record(
        "glm52_production_execution",
        state="FAILED",
        activation_id="activation-1",
        activation_ordinal=1,
        epoch=1,
        epoch_text="00000001",
        expected_execution_arn=ACTIVATION_EXECUTION,
        expected_state_machine_version_arn=ACTIVATION_VERSION,
        terminal_status="FAILED",
    )
    return {
        "activation_index": _activation_index(),
        "control": control,
        "execution": execution,
        "recovery_control": recovery,
        "sky_action": action,
    }


@pytest.mark.parametrize(
    ("state", "target", "outcome"),
    (
        ("ARMED", "ABANDONED", None),
        (
            "CONSUMED",
            "POST_CLASSIFIED",
            "PROVED_NOT_SENT_OWNER_DIED",
        ),
        (
            "POST_STARTED",
            "POST_CLASSIFIED",
            "PROVED_NOT_SENT_OWNER_DIED",
        ),
        (
            "POST_AUTHORIZED",
            "POST_CLASSIFIED",
            "AMBIGUOUS_OWNER_DIED",
        ),
    ),
)
def test_builds_every_frozen_owner_death_edge(
    state: str, target: str, outcome: str | None
) -> None:
    request = build_owner_death_request(
        invocation=_invocation(),
        live_sources=_sources(state),
        observed_at=NOW,
    )
    plan = request["plan"]
    assert request["domain"] == "RECOVERY"
    assert isinstance(plan["control"], dict)
    assert plan["control"]["after"]["last_sky_post_state"] == target
    assert plan["action"]["before"]["state"] == state
    assert plan["action"]["after"]["state"] == target
    if outcome is None:
        assert plan["action"]["after"]["abandonment_proof_sha256"]
    else:
        assert plan["action"]["after"]["outcome_class"] == outcome
        assert (
            plan["action"]["after"]["classification_evidence_kind"]
            == "OWNER_DEATH_PROOF"
        )
        assert plan["action"]["after"]["sky_request_id"] is None
        assert plan["action"]["after"]["response_identity_sha256"] is None


def test_adopts_exact_already_classified_without_an_update() -> None:
    request = build_owner_death_request(
        invocation=_invocation(),
        live_sources=_sources("POST_CLASSIFIED"),
        observed_at=NOW,
    )
    plan = request["plan"]
    assert set(plan["control"]) == {"key", "expected"}
    assert set(plan["action"]) == {"key", "expected"}
    assert plan["action"]["expected"]["state"] == "POST_CLASSIFIED"


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value["execution"].update(state="RUNNING"),
        lambda value: value["execution"].update(
            expected_execution_arn=RECOVERY_EXECUTION
        ),
        lambda value: value["recovery_control"].update(
            owner_hard_expires_at=PAST
        ),
        lambda value: value["control"].update(
            last_sky_post_action_key="foreign-action"
        ),
        lambda value: value["sky_action"].update(
            arming_hard_expires_at=FUTURE
        ),
    ),
)
def test_rejects_foreign_live_or_unexpired_authority_before_plan(
    mutator: object,
) -> None:
    sources = deepcopy(_sources("ARMED"))
    mutator(sources)
    with pytest.raises(Task12OwnerDeathError):
        build_owner_death_request(
            invocation=_invocation(),
            live_sources=sources,
            observed_at=NOW,
        )


def test_plan_contains_only_checks_and_the_two_conditional_updates() -> None:
    request = build_owner_death_request(
        invocation=_invocation(),
        live_sources=_sources("CONSUMED"),
        observed_at=NOW,
    )
    plan = request["plan"]
    assert set(plan) == {
        "index",
        "control",
        "recovery_control",
        "execution",
        "action",
    }
    assert set(plan["index"]) == {"key", "expected"}
    assert set(plan["recovery_control"]) == {"key", "expected"}
    assert set(plan["execution"]) == {"key", "expected"}
    assert set(plan["control"]) == {"key", "before", "after"}
    assert set(plan["action"]) == {"key", "before", "after"}


def test_typed_plan_uses_exact_check_or_update_members() -> None:
    from glm52_enforcement.task12_live_owner_death import (
        OwnerDeathClassificationPlan,
    )

    sources = _sources("CONSUMED")
    request = build_owner_death_request(
        invocation=_invocation(),
        live_sources=sources,
        observed_at=NOW,
    )
    plan = request["plan"]
    typed = OwnerDeathClassificationPlan(
        index=ExactCheck(
            key=LedgerKey(**plan["index"]["key"]),
            expected=plan["index"]["expected"],
        ),
        control=ExactUpdate(
            key=LedgerKey(**plan["control"]["key"]),
            before=plan["control"]["before"],
            after=plan["control"]["after"],
        ),
        recovery_control=ExactCheck(
            key=LedgerKey(**plan["recovery_control"]["key"]),
            expected=plan["recovery_control"]["expected"],
        ),
        execution=ExactCheck(
            key=LedgerKey(**plan["execution"]["key"]),
            expected=plan["execution"]["expected"],
        ),
        action=ExactUpdate(
            key=LedgerKey(**plan["action"]["key"]),
            before=plan["action"]["before"],
            after=plan["action"]["after"],
        ),
    )
    assert len(typed.transaction_plans) == 5


def _typed_update_plan(
    value: dict[str, object],
) -> object:
    from glm52_enforcement.task12_live_owner_death import (
        OwnerDeathClassificationPlan,
    )

    return OwnerDeathClassificationPlan(
        index=ExactCheck(
            key=LedgerKey(**value["index"]["key"]),
            expected=value["index"]["expected"],
        ),
        control=ExactUpdate(
            key=LedgerKey(**value["control"]["key"]),
            before=value["control"]["before"],
            after=value["control"]["after"],
        ),
        recovery_control=ExactCheck(
            key=LedgerKey(**value["recovery_control"]["key"]),
            expected=value["recovery_control"]["expected"],
        ),
        execution=ExactCheck(
            key=LedgerKey(**value["execution"]["key"]),
            expected=value["execution"]["expected"],
        ),
        action=ExactUpdate(
            key=LedgerKey(**value["action"]["key"]),
            before=value["action"]["before"],
            after=value["action"]["after"],
        ),
    )


@pytest.mark.parametrize("lost_response", (False, True))
def test_dynamo_owner_death_commit_reconciles_exact_multirow_readback(
    lost_response: bool,
) -> None:
    raw_nonce = b"R" * 32
    sources = _sources(
        "CONSUMED",
        recovery_nonce_sha256=hashlib.sha256(raw_nonce).hexdigest(),
    )
    request = build_owner_death_request(
        invocation=_invocation(),
        live_sources=sources,
        observed_at=NOW,
    )
    plan = _typed_update_plan(request["plan"])
    client = AwsShapedSimulator()
    client.install(plan.index.key.sort_key, plan.index.expected)
    client.install(plan.control.key.sort_key, plan.control.before)
    client.install(
        plan.recovery_control.key.sort_key,
        plan.recovery_control.expected,
    )
    client.install(plan.execution.key.sort_key, plan.execution.expected)
    client.install(plan.action.key.sort_key, plan.action.before)
    if lost_response:
        client.post_commit_fault = SimulatedClientError(
            "InternalServerError"
        )
    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).classify_owner_dead_sky_action(
        plan=plan,
        domain=request["domain"],
        operation_identity_sha256=request[
            "operation_identity_sha256"
        ],
        raw_owner_nonce=raw_nonce,
    )
    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert len(result.records) == 5
    assert (
        result.records[-1]["post_classification_transaction_client_request_token_sha256"]
        is not None
    )
    assert sum(name == "transact_write_items" for name, _ in client.calls) == 1
