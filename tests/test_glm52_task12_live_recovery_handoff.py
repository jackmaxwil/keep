from __future__ import annotations

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.dynamodb import (
    DynamoLedgerAdapter,
    TransactionResolution,
)
from glm52_enforcement.h1f_adapter import H1fAuditResult
from glm52_enforcement.records import ledger_sk
import glm52_enforcement.task12_live_recovery_handoff as recovery_handoff
from glm52_enforcement.task12_live_recovery_handoff import (
    RecoveryHandoffAdoption,
    RecoveryHandoffNotRequired,
    Task12RecoveryHandoffError,
    build_recovery_handoff_record,
    execute_live_request,
    materialize_live_request,
    persist_live_successors,
    validate_normal_task11_handoff,
)
from glm52_enforcement.task12_correlation import (
    build_sky_post_handoff_record,
)
from glm52_enforcement.task12_writers import (
    RetainedWriteResult,
    build_retained_writer_candidate,
)

from test_glm52_enforcement_dynamodb import (
    AwsShapedSimulator,
    _activation_index,
    _closed_record,
    _control,
)


RUN_ID = "glm52-sky-20260724"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
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
BUCKET = "keep-glm52-models-246813579024-us-west-2"


def _invocation() -> SimpleNamespace:
    return SimpleNamespace(
        operation_kind="RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        state_machine_execution_arn=RECOVERY_EXECUTION,
        caller_state_machine_version_arn=RECOVERY_VERSION,
    )


def _sources(outcome: str = "PROVED_NOT_SENT_OWNER_DIED") -> dict[str, dict[str, object]]:
    action_key = ledger_sk(
        "glm52_production_action",
        activation_id="activation-1",
        generation=1,
        action_kind="SKY_POST",
        attempt=1,
    )
    action = _closed_record(
        "glm52_production_action",
        state="POST_CLASSIFIED",
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        action_kind="SKY_POST",
        attempt=1,
        owner_epoch=1,
        owner_execution_arn=ACTIVATION_EXECUTION,
        armed_by_epoch=1,
        armed_by_execution_arn=ACTIVATION_EXECUTION,
        armed_by_state_machine_version_arn=ACTIVATION_VERSION,
        candidate_key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/START_DECISION.json"
        ),
        candidate_file_sha256=SHA_A,
        candidate_body_sha256=SHA_B,
        request_body_sha256=SHA_C,
        outcome_class=outcome,
        classification_evidence_kind="OWNER_DEATH_PROOF",
        classification_evidence_body_sha256=SHA_A,
        response_identity_sha256=None,
        sky_request_id=None,
        relay_envelope_sha256=None,
        post_owner_invocation_nonce_sha256=None,
        post_owner_function_version_arn=None,
        post_owner_dispatch_identity_sha256=None,
        post_owner_hard_expires_at=None,
        post_started_at=None,
        post_authorized_at=None,
        consumed_at="2026-07-28T10:00:00Z",
        completed_at="2026-07-28T12:00:00Z",
    )
    control = _control(
        phase="RECOVERY_SEALING",
        active_epoch=1,
        active_execution_arn=ACTIVATION_EXECUTION,
        active_state_machine_version_arn=ACTIVATION_VERSION,
        last_sky_post_generation=1,
        last_sky_post_action_key=action_key,
        last_sky_post_state="POST_CLASSIFIED",
    )
    return {
        "activation_index": _activation_index(),
        "control": control,
        "execution": _closed_record(
            "glm52_production_execution",
            state="FAILED",
            activation_id="activation-1",
            activation_ordinal=1,
            epoch=1,
            epoch_text="00000001",
            expected_execution_arn=ACTIVATION_EXECUTION,
            expected_state_machine_version_arn=ACTIVATION_VERSION,
            terminal_status="FAILED",
        ),
        "recovery_control": _closed_record(
            "glm52_production_recovery_control",
            state="OWNED",
            activation_id="activation-1",
            activation_ordinal=1,
            owner_execution_arn=RECOVERY_EXECUTION,
            owner_state_machine_version_arn=RECOVERY_VERSION,
            owner_hard_expires_at="2026-07-29T12:00:00Z",
            terminal_v2_identity_sha256=None,
        ),
        "sky_action": action,
    }


def _decision() -> dict[str, object]:
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_production_generation_start_decision_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "run_id": RUN_ID,
        "managed_mode": "production",
        "campaign_identity_sha256": SHA_A,
        "generation": 1,
        "generation_text": "00000001",
        "submit_attempt_id": SHA_C,
        "sky_job_name": RUN_ID,
        "sky_job_identity_sha256": SHA_A,
        "generation_claim_key": (
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/GENERATION_CLAIM.json"
        ),
        "generation_claim_file_sha256": SHA_A,
        "generation_claim_body_sha256": SHA_B,
        "generation_claim_version_id": "claim-version-1",
        "decision": "launch-once",
        "must_start_by": "2026-07-28T13:00:00Z",
        "decided_at": "2026-07-28T11:00:00Z",
    }
    return {
        **body,
        "start_decision_body_sha256": canonical_sha256(body),
    }


def _production_authority() -> dict[str, object]:
    return {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "action_key": ledger_sk(
            "glm52_production_action",
            activation_id="activation-1",
            generation=1,
            action_kind="SKY_POST",
            attempt=1,
        ),
        "sky_job_name": RUN_ID,
        "task_yaml_sha256": SHA_B,
        "request_body_sha256": SHA_C,
        "expected_execution_arn": ACTIVATION_EXECUTION,
        "workflow_version_arn": ACTIVATION_VERSION,
    }


def test_builds_the_frozen_recovery_body_from_authenticated_sources() -> None:
    decision = _decision()
    sources = _sources()
    sources["sky_action"]["candidate_body_sha256"] = decision[
        "start_decision_body_sha256"
    ]
    record = build_recovery_handoff_record(
        invocation=_invocation(),
        live_sources=sources,
        decision=decision,
        decision_version_id="decision-version-1",
        decision_file_sha256=SHA_A,
        production_authority=_production_authority(),
        api_server_identity_sha256=SHA_C,
    )
    assert set(record) == {
        "run_id",
        "campaign_identity_sha256",
        "generation",
        "generation_text",
        "submit_attempt_id",
        "decision_key",
        "decision_version_id",
        "decision_file_sha256",
        "decision_body_sha256",
        "sky_post_action_key",
        "sky_post_consumed_at",
        "sky_post_outcome_class",
        "expected_sky_job_name",
        "task_yaml_sha256",
        "request_body_sha256",
        "api_server_identity_sha256",
        "sky_request_id",
        "post_started_at",
        "post_completed_or_lost_at",
        "binding_state",
        "handoff_body_sha256",
    }
    assert record["sky_request_id"] is None
    assert record["post_started_at"] is None
    body = dict(record)
    assert body.pop("handoff_body_sha256") == canonical_sha256(body)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["sky_action"].update(request_body_sha256=SHA_A),
        lambda value: value["recovery_control"].update(
            owner_execution_arn=ACTIVATION_EXECUTION
        ),
        lambda value: value["execution"].update(state="RUNNING"),
        lambda value: value["control"].update(last_sky_post_state="CONSUMED"),
    ),
)
def test_recovery_body_rejects_a_forked_authority_tuple(
    mutation: object,
) -> None:
    sources = deepcopy(_sources())
    mutation(sources)
    with pytest.raises(Task12RecoveryHandoffError):
        build_recovery_handoff_record(
            invocation=_invocation(),
            live_sources=sources,
            decision=_decision(),
            decision_version_id="decision-version-1",
            decision_file_sha256=SHA_A,
            production_authority=_production_authority(),
            api_server_identity_sha256=SHA_C,
        )


def _normal_handoff() -> dict[str, object]:
    return build_sky_post_handoff_record(
        run_id=RUN_ID,
        campaign_identity_sha256=SHA_A,
        generation=1,
        generation_text="00000001",
        submit_attempt_id="attempt-1",
        decision_key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/decision/START_DECISION.json"
        ),
        decision_version_id="decision-version-1",
        decision_file_sha256=SHA_A,
        decision_body_sha256=SHA_B,
        sky_post_action_key=ledger_sk(
            "glm52_production_action",
            activation_id="activation-1",
            generation=1,
            action_kind="SKY_POST",
            attempt=1,
        ),
        sky_post_consumed_at="2026-07-28T10:00:00Z",
        sky_post_outcome_class="ACCEPTED",
        expected_sky_job_name=RUN_ID,
        task_yaml_sha256=SHA_B,
        request_body_sha256=SHA_C,
        api_server_identity_sha256=SHA_A,
        sky_request_id="sky-request-1",
        post_started_at="2026-07-28T11:00:00Z",
        post_completed_or_lost_at="2026-07-28T12:00:00Z",
        binding_state="reconcile-required",
    )


def test_adopts_the_exact_shared_frozen_task11_schema() -> None:
    action = deepcopy(_sources("PROVED_NOT_SENT_OWNER_DIED")["sky_action"])
    action.update(
        outcome_class="ACCEPTED",
        classification_evidence_kind="RELAY_RESPONSE",
        classification_evidence_body_sha256=SHA_A,
        authority_audit_body_sha256=SHA_B,
        response_identity_sha256=SHA_C,
        sky_request_id="sky-request-1",
        post_started_at="2026-07-28T11:00:00Z",
    )
    assert validate_normal_task11_handoff(
        body=_normal_handoff(),
        invocation=_invocation(),
        action=action,
    ) == _normal_handoff()


def test_normal_task11_adoption_rejects_action_drift() -> None:
    action = deepcopy(_sources()["sky_action"])
    action["outcome_class"] = "ACCEPTED"
    action["classification_evidence_kind"] = "RELAY_RESPONSE"
    action["request_body_sha256"] = SHA_A
    with pytest.raises(Task12RecoveryHandoffError):
        validate_normal_task11_handoff(
            body=_normal_handoff(),
            invocation=_invocation(),
            action=action,
        )


def test_compact_legacy_task11_handoff_is_foreign_bytes() -> None:
    compact = {
        "schema_version": 1,
        "record_type": "glm52_task11_sky_post_handoff_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": RUN_ID,
        "activation_id": "activation-1",
        "generation": 1,
        "admission_identity_sha256": SHA_A,
        "post_audit_body_sha256": SHA_B,
        "relay_receipt_sha256": SHA_C,
        "classification": "POST_ACCEPTED_ONCE",
        "persisted_at": "2026-07-28T12:00:00Z",
    }
    compact["sky_post_handoff_body_sha256"] = canonical_sha256(compact)
    with pytest.raises(Task12RecoveryHandoffError):
        validate_normal_task11_handoff(
            body=compact,
            invocation=_invocation(),
            action=_sources()["sky_action"],
        )


def _record_candidate(
    *,
    sources: dict[str, dict[str, object]] | None = None,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, object],
    object,
]:
    exact_sources = _sources() if sources is None else sources
    decision = _decision()
    exact_sources["sky_action"]["candidate_body_sha256"] = decision[
        "start_decision_body_sha256"
    ]
    record = build_recovery_handoff_record(
        invocation=_invocation(),
        live_sources=exact_sources,
        decision=decision,
        decision_version_id="decision-version-1",
        decision_file_sha256=SHA_A,
        production_authority=_production_authority(),
        api_server_identity_sha256=SHA_C,
    )
    candidate = build_retained_writer_candidate(
        writer_kind="RecoveryHandoff",
        campaign_bucket=BUCKET,
        activation_id="activation-1",
        generation=1,
        authority_domain="RECOVERY",
        record=record,
    )
    return exact_sources, record, candidate


def _ports() -> SimpleNamespace:
    return SimpleNamespace(
        deployment=SimpleNamespace(
            role_coordinates={
                "campaign_bucket": BUCKET,
                "ledger_table_name": "glm52-ledger",
            }
        ),
        client=lambda service: SimpleNamespace(service=service),
    )


def _consumed_recovery_action(
    *,
    sources: dict[str, dict[str, object]],
    candidate: object,
) -> dict[str, object]:
    recovery = recovery_handoff._rehash(
        {
            **sources["recovery_control"],
            "revision": 3,
            "updated_at": "2026-07-28T12:02:00Z",
        }
    )
    sources["recovery_control"] = recovery_handoff.validate_record(
        "glm52_production_recovery_control", recovery
    )
    return _closed_record(
        "glm52_production_recovery_action",
        authority_domain="RECOVERY",
        campaign_identity_sha256=recovery["campaign_identity_sha256"],
        activation_id="activation-1",
        activation_ordinal=1,
        action_kind="RECOVERY_HANDOFF",
        attempt=1,
        candidate_key=recovery_handoff._candidate_key(candidate),
        candidate_body_sha256=candidate.body_sha256,
        request_body_sha256=candidate.candidate_identity_sha256,
        owner_attempt=recovery["owner_attempt"],
        owner_execution_arn=recovery["owner_execution_arn"],
        owner_state_machine_version_arn=(
            recovery["owner_state_machine_version_arn"]
        ),
        owner_dispatch_identity_sha256=(
            recovery["owner_dispatch_identity_sha256"]
        ),
        owner_invocation_nonce_sha256=(
            recovery["owner_invocation_nonce_sha256"]
        ),
        owner_hard_expires_at=recovery["owner_hard_expires_at"],
        authority_barrier_nonce_sha256=(
            recovery["recovery_barrier_nonce_sha256"]
        ),
        authority_audit_body_sha256=SHA_A,
        authority_audit_closing_revision=2,
        authorized_transition_from_revision=2,
        authorized_transition_to_revision=3,
        state="CONSUMED",
        armed_at="2026-07-28T12:01:00Z",
        consumed_at="2026-07-28T12:02:00Z",
        completed_at=None,
        response_identity_sha256=None,
        reconciliation_identity_sha256=None,
        arming_transaction_client_request_token_sha256=SHA_B,
        consume_transaction_client_request_token_sha256=SHA_C,
        revision=2,
        generation=1,
        generation_text="00000001",
        allocation_ordinal=None,
        allocation_ordinal_text=None,
        worker_launch_identity_sha256=None,
        worker_launch_liability_identity_sha256=None,
    )


def test_abandoned_sky_action_is_typed_no_action_without_aws_or_writer() -> None:
    from glm52_enforcement.task12_live_owner_death import (
        build_owner_death_request,
    )
    from test_glm52_task12_live_owner_death import (
        _invocation as owner_death_invocation,
        _sources as owner_death_sources,
    )

    sources = owner_death_sources("ARMED")
    owner_death = build_owner_death_request(
        invocation=owner_death_invocation(),
        live_sources=sources,
        observed_at="2026-07-28T12:00:00Z",
    )
    sources["control"] = owner_death["plan"]["control"]["after"]
    sources["sky_action"] = owner_death["plan"]["action"]["after"]

    class NoAwsOrWriter:
        deployment = SimpleNamespace(role_coordinates={})

        @staticmethod
        def client(service: str) -> object:
            raise AssertionError("no-action path must not touch " + service)

    request = materialize_live_request(
        operation_kind="RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
        invocation=_invocation(),
        live_sources=sources,
        ports=NoAwsOrWriter(),
    )
    assert request is not None
    result = execute_live_request(
        invocation=_invocation(),
        request=request,
        ports=NoAwsOrWriter(),
        writer=lambda **kwargs: pytest.fail(
            "no-action path must not invoke the writer"
        ),
    )
    assert type(result) is RecoveryHandoffNotRequired
    assert result.state == "NOT_REQUIRED_ABANDONED_SKY_ACTION"
    assert persist_live_successors(
        operation_kind="RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
        invocation=_invocation(),
        live_sources=sources,
        request=request,
        domain_result=result,
        ports=NoAwsOrWriter(),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["control"].update(
            last_sky_post_action_key="foreign-action"
        ),
        lambda value: value["execution"].update(state="RUNNING"),
        lambda value: value["recovery_control"].update(
            owner_execution_arn=ACTIVATION_EXECUTION
        ),
        lambda value: value["sky_action"].update(
            generation=2,
            generation_text="00000002",
        ),
    ),
)
def test_abandoned_no_action_rejects_every_authority_fork_before_aws(
    mutation: object,
) -> None:
    from glm52_enforcement.task12_live_owner_death import (
        build_owner_death_request,
    )
    from test_glm52_task12_live_owner_death import (
        _invocation as owner_death_invocation,
        _sources as owner_death_sources,
    )

    sources = owner_death_sources("ARMED")
    owner_death = build_owner_death_request(
        invocation=owner_death_invocation(),
        live_sources=sources,
        observed_at="2026-07-28T12:00:00Z",
    )
    sources["control"] = owner_death["plan"]["control"]["after"]
    sources["sky_action"] = owner_death["plan"]["action"]["after"]
    mutation(sources)

    class NoAws:
        deployment = SimpleNamespace(role_coordinates={})

        @staticmethod
        def client(service: str) -> object:
            raise AssertionError("forked no-action touched " + service)

    with pytest.raises(Task12RecoveryHandoffError):
        materialize_live_request(
            operation_kind=(
                "RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED"
            ),
            invocation=_invocation(),
            live_sources=sources,
            ports=NoAws(),
        )


def test_normal_task11_handoff_adoption_never_invokes_recovery_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_glm52_task12_live_owner_death import (
        _sources as owner_death_sources,
    )

    sources = owner_death_sources("POST_CLASSIFIED")
    identity_body = {
        "state": "ADOPTED_NORMAL_TASK11_HANDOFF",
        "version_id": "normal-task11-version",
        "file_sha256": SHA_A,
        "body_sha256": SHA_B,
    }
    adoption = RecoveryHandoffAdoption(
        **identity_body,
        canonical_identity_sha256=canonical_sha256(identity_body),
    )
    calls: list[str] = []

    def exact_existing(**kwargs: object) -> RecoveryHandoffAdoption:
        calls.append("exact-existing")
        return adoption

    monkeypatch.setattr(
        recovery_handoff, "_existing_handoff", exact_existing
    )
    request = materialize_live_request(
        operation_kind="RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
        invocation=_invocation(),
        live_sources=sources,
        ports=_ports(),
    )
    assert request is not None
    result = execute_live_request(
        invocation=_invocation(),
        request=request,
        ports=_ports(),
        writer=lambda **kwargs: pytest.fail(
            "normal Task11 adoption must not invoke recovery writer"
        ),
    )
    assert result == adoption
    assert calls == ["exact-existing"]


def test_public_materializer_resumes_consumed_action_without_rearming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, record, candidate = _record_candidate()
    durable_action = _consumed_recovery_action(
        sources=sources,
        candidate=candidate,
    )
    monkeypatch.setattr(
        recovery_handoff,
        "_decision_authority",
        lambda **kwargs: (_decision(), "decision-version-1", SHA_A),
    )
    monkeypatch.setattr(
        recovery_handoff,
        "_production_authority",
        lambda **kwargs: _production_authority(),
    )
    monkeypatch.setattr(
        recovery_handoff,
        "_api_server_identity",
        lambda *args: SHA_C,
    )
    monkeypatch.setattr(
        recovery_handoff,
        "_sole_audited_object",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        recovery_handoff,
        "_continuation_capsule",
        lambda invocation: {"opaque": "capsule"},
    )
    monkeypatch.setattr(
        recovery_handoff,
        "_maybe_read_recovery_action",
        lambda **kwargs: durable_action,
    )
    monkeypatch.setattr(
        recovery_handoff,
        "_arm_and_consume",
        lambda **kwargs: pytest.fail(
            "CONSUMED restart must not ARM or repeat H1f"
        ),
    )

    request = materialize_live_request(
        operation_kind="RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
        invocation=_invocation(),
        live_sources=sources,
        ports=_ports(),
    )
    assert request is not None
    assert request["handoff_record"] == {
        "mode": "CREATE",
        "record": record,
    }
    assert (
        request["handoff_action"]["durable_action"] == durable_action
    )


def test_fresh_process_resumes_durable_arm_at_h1f_then_consumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_nonce = b"r" * 32
    sources = _sources()
    sources["recovery_control"]["owner_invocation_nonce_sha256"] = (
        hashlib.sha256(raw_nonce).hexdigest()
    )
    sources, _record, candidate = _record_candidate(sources=sources)
    armed_at = "2026-07-28T12:01:00Z"
    armed_action = recovery_handoff._recovery_action_record(
        invocation=_invocation(),
        recovery=sources["recovery_control"],
        candidate=candidate,
        observed_at=armed_at,
    )
    recovery_after_arm = recovery_handoff._rehash(
        {
            **sources["recovery_control"],
            "revision": sources["recovery_control"]["revision"] + 1,
            "updated_at": armed_at,
        }
    )
    sources["recovery_control"] = recovery_handoff.validate_record(
        "glm52_production_recovery_control",
        recovery_after_arm,
    )
    simulator = AwsShapedSimulator()
    for sort_key, record in (
        (
            ledger_sk("glm52_production_activation_index"),
            sources["activation_index"],
        ),
        (
            ledger_sk(
                "glm52_production_control",
                activation_id="activation-1",
            ),
            sources["control"],
        ),
        (
            ledger_sk(
                "glm52_production_recovery_control",
                activation_id="activation-1",
            ),
            sources["recovery_control"],
        ),
        (
            ledger_sk(
                "glm52_production_recovery_action",
                activation_id="activation-1",
                action_kind="RECOVERY_HANDOFF",
                attempt=1,
            ),
            armed_action,
        ),
    ):
        simulator.install(sort_key, record)
    durable = DynamoLedgerAdapter(
        client=simulator, table_name="glm52-ledger"
    )
    events: list[str] = []

    class ResumeLedger:
        def commit_recovery_handoff_arm(
            self, **kwargs: object
        ) -> TransactionResolution:
            raise AssertionError("fresh process must not ARM twice")

        def commit_recovery_handoff_consume(
            self, **kwargs: object
        ) -> TransactionResolution:
            events.append("consume")
            return durable.commit_recovery_handoff_consume(**kwargs)

    class FreshAudit:
        def __init__(self, *, authority_reader: object) -> None:
            self.authority_reader = authority_reader

        def fresh_audit(
            self, *, s3: object, request: object
        ) -> H1fAuditResult:
            events.append("fresh-h1f")
            self.authority_reader()
            body = {
                "authority_domain": "RECOVERY",
                "operation_kind": "S3_CREATE",
                "action_key": request.action_key,
                "candidate_identity_sha256": (
                    request.candidate.candidate_identity_sha256
                ),
                "activation_id": "activation-1",
                "generation": 1,
                "epoch": 1,
                "execution_arn": RECOVERY_EXECUTION,
                "barrier_nonce_sha256": sources["recovery_control"][
                    "recovery_barrier_nonce_sha256"
                ],
                "closing_revision": sources["recovery_control"][
                    "revision"
                ],
                "expected_authorized_revision": (
                    sources["recovery_control"]["revision"] + 1
                ),
                "active_head_key": "campaigns/fence/FENCE.json",
                "active_head_version_id": "fence-version-1",
                "active_head_file_sha256": SHA_A,
                "active_head_body_sha256": SHA_B,
            }
            return H1fAuditResult(
                **body,
                canonical_body_sha256=canonical_sha256(body),
            )

    monkeypatch.setattr(recovery_handoff, "_ledger", lambda ports: ResumeLedger())
    monkeypatch.setattr(
        recovery_handoff, "_owner_nonce", lambda **kwargs: raw_nonce
    )
    monkeypatch.setattr(
        recovery_handoff,
        "_fresh_authority_reader",
        lambda **kwargs: lambda: object(),
    )
    monkeypatch.setattr(
        recovery_handoff,
        "_now",
        lambda: "2026-07-28T12:02:00Z",
    )
    monkeypatch.setattr(
        "glm52_enforcement.h1f_adapter.FreshH1fAuditService",
        FreshAudit,
    )

    consumed, _audit = recovery_handoff._arm_and_consume(
        ports=_ports(),
        invocation=_invocation(),
        sources=sources,
        candidate=candidate,
        capsule={"opaque": "capsule"},
        existing_armed_action=armed_action,
    )
    assert consumed["state"] == "CONSUMED"
    assert events == ["fresh-h1f", "consume"]


def test_recovery_route_arms_fresh_audits_consumes_writes_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_nonce = b"r" * 32
    sources = _sources()
    sources["recovery_control"]["owner_invocation_nonce_sha256"] = (
        hashlib.sha256(raw_nonce).hexdigest()
    )
    sources, record, candidate = _record_candidate(sources=sources)
    simulator = AwsShapedSimulator()
    simulator.install(
        ledger_sk("glm52_production_activation_index"),
        sources["activation_index"],
    )
    simulator.install(
        ledger_sk(
            "glm52_production_control", activation_id="activation-1"
        ),
        sources["control"],
    )
    simulator.install(
        ledger_sk(
            "glm52_production_recovery_control",
            activation_id="activation-1",
        ),
        sources["recovery_control"],
    )
    durable = DynamoLedgerAdapter(
        client=simulator, table_name="glm52-ledger"
    )
    events: list[str] = []

    class RecordingLedger:
        def read_coherent(self, **kwargs: object) -> object:
            return durable.read_coherent(**kwargs)

        def commit_recovery_handoff_arm(
            self, **kwargs: object
        ) -> TransactionResolution:
            events.append("arm")
            return durable.commit_recovery_handoff_arm(**kwargs)

        def commit_recovery_handoff_consume(
            self, **kwargs: object
        ) -> TransactionResolution:
            events.append("consume")
            return durable.commit_recovery_handoff_consume(**kwargs)

        def commit_recovery_handoff_complete(
            self, **kwargs: object
        ) -> TransactionResolution:
            events.append("complete")
            return durable.commit_recovery_handoff_complete(**kwargs)

    class FreshAudit:
        def __init__(self, *, authority_reader: object) -> None:
            self.authority_reader = authority_reader

        def fresh_audit(
            self, *, s3: object, request: object
        ) -> H1fAuditResult:
            events.append("fresh-authority")
            self.authority_reader()
            events.append("fresh-h1f")
            body = {
                "authority_domain": "RECOVERY",
                "operation_kind": "S3_CREATE",
                "action_key": request.action_key,
                "candidate_identity_sha256": (
                    request.candidate.candidate_identity_sha256
                ),
                "activation_id": "activation-1",
                "generation": 1,
                "epoch": 1,
                "execution_arn": RECOVERY_EXECUTION,
                "barrier_nonce_sha256": sources["recovery_control"][
                    "recovery_barrier_nonce_sha256"
                ],
                "closing_revision": (
                    sources["recovery_control"]["revision"] + 1
                ),
                "expected_authorized_revision": (
                    sources["recovery_control"]["revision"] + 2
                ),
                "active_head_key": "campaigns/fence/FENCE.json",
                "active_head_version_id": "fence-version-1",
                "active_head_file_sha256": SHA_A,
                "active_head_body_sha256": SHA_B,
            }
            return H1fAuditResult(
                **body,
                canonical_body_sha256=canonical_sha256(body),
            )

    timestamps = iter(
        (
            "2026-07-28T12:01:00Z",
            "2026-07-28T12:02:00Z",
            "2026-07-28T12:03:00Z",
        )
    )
    monkeypatch.setattr(recovery_handoff, "_ledger", lambda ports: RecordingLedger())
    monkeypatch.setattr(
        recovery_handoff,
        "_owner_nonce",
        lambda **kwargs: raw_nonce,
    )
    monkeypatch.setattr(recovery_handoff, "_now", lambda: next(timestamps))
    monkeypatch.setattr(
        recovery_handoff,
        "_fresh_authority_reader",
        lambda **kwargs: lambda: object(),
    )
    monkeypatch.setattr(
        "glm52_enforcement.h1f_adapter.FreshH1fAuditService",
        FreshAudit,
    )

    consumed_action, _audit = recovery_handoff._arm_and_consume(
        ports=_ports(),
        invocation=_invocation(),
        sources=sources,
        candidate=candidate,
        capsule={"opaque": "capsule"},
    )
    assert consumed_action["state"] == "CONSUMED"
    assert consumed_action["authority_audit_body_sha256"]
    writer_action, writer_audit = recovery_handoff._writer_authorities(
        candidate=candidate,
        durable_action=consumed_action,
    )
    request = {
        "handoff_record": {"mode": "CREATE", "record": record},
        "handoff_action": {
            "mode": "CREATE",
            "durable_action": consumed_action,
            "writer_action": writer_action,
            "owner_nonce_capsule": {"opaque": "capsule"},
        },
        "audit": {"mode": "CREATE", "writer_audit": writer_audit},
    }

    def writer(**kwargs: object) -> RetainedWriteResult:
        events.append("writer")
        assert kwargs == {
            "record": record,
            "action": writer_action,
            "audit": writer_audit,
        }
        body = {
            "writer_kind": "RecoveryHandoff",
            "outcome": "created",
            "coordinate": candidate.coordinate,
            "candidate_identity_sha256": (
                candidate.candidate_identity_sha256
            ),
            "object_version_id": "handoff-version-1",
            "response_request_ids": ("put-request-1",),
            "response_authenticated": True,
        }
        return RetainedWriteResult(
            **body,
            canonical_identity_sha256=canonical_sha256(body),
        )

    result = execute_live_request(
        invocation=_invocation(),
        request=request,
        ports=_ports(),
        writer=writer,
    )
    assert type(result) is RetainedWriteResult
    forked_request = deepcopy(request)
    forked_request["handoff_action"]["durable_action"][
        "candidate_body_sha256"
    ] = SHA_C
    with pytest.raises(
        Task12RecoveryHandoffError,
        match="request/durable action binding drifted",
    ):
        persist_live_successors(
            operation_kind=(
                "RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED"
            ),
            invocation=_invocation(),
            live_sources=sources,
            request=forked_request,
            domain_result=result,
            ports=_ports(),
        )
    assert "complete" not in events
    assert persist_live_successors(
        operation_kind="RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
        invocation=_invocation(),
        live_sources=sources,
        request=request,
        domain_result=result,
        ports=_ports(),
    )
    action_key = ledger_sk(
        "glm52_production_recovery_action",
        activation_id="activation-1",
        action_kind="RECOVERY_HANDOFF",
        attempt=1,
    )
    completed_action = durable.read_coherent(
        items=(
            (
                recovery_handoff.LedgerKey(RUN_ID, action_key),
                "glm52_production_recovery_action",
            ),
        )
    )[0]
    assert completed_action["state"] == "COMPLETED"
    assert (
        completed_action["response_identity_sha256"]
        == result.canonical_identity_sha256
    )
    writer_control = durable.read_coherent(
        items=(
            (
                recovery_handoff.LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_task12_versioned_writer_control_v1",
                        activation_id="activation-1",
                        generation=1,
                        writer_kind="RecoveryHandoff",
                    ),
                ),
                "glm52_task12_versioned_writer_control_v1",
            ),
        )
    )[0]
    assert writer_control["object_version_id"] == "handoff-version-1"
    completed_recovery = durable.read_coherent(
        items=(
            (
                recovery_handoff.LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_recovery_control",
                        activation_id="activation-1",
                    ),
                ),
                "glm52_production_recovery_control",
            ),
        )
    )[0]
    adoption = recovery_handoff._completed_recovery_adoption(
        ports=_ports(),
        invocation=_invocation(),
        recovery=completed_recovery,
        action=completed_action,
        candidate=candidate,
        version_id="handoff-version-1",
    )
    assert adoption.state == "ADOPTED_COMPLETED_RECOVERY_HANDOFF"
    assert events == [
        "arm",
        "fresh-authority",
        "fresh-h1f",
        "consume",
        "writer",
        "complete",
    ]
