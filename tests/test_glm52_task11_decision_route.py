"""Task 11 executable private decision-to-POST route."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import hashlib

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.decision_closure import (
    AUTHORITY_AUDIT_KINDS,
    CLOSURE_STEPS,
    HANDOFF_AUDIT_KIND,
    ClosureRouteError,
    DirectDecisionResponse,
    StoredDecisionNoPostError,
    build_admission_closure_result,
    build_audit_evidence,
    build_closure_request,
    build_direct_decision_response,
    build_launch_or_expire_decision,
    build_source_batch_receipt,
    build_source_publication,
    build_step_receipt,
    build_transaction_readback,
    build_task11_workflow_definition,
    execute_decision_closure,
    transition_abandoned_arm,
)
from glm52_enforcement.live_authority import H1dLiveAuthorityResult
from glm52_enforcement.sky_admission import AdmissionResult, AttestationResult


SHA = hashlib.sha256(b"task11").hexdigest()
ADMISSION_VERSION = (
    "arn:aws:lambda:us-west-2:246813579024:"
    "function:keep-glm52-h1g-launch-admission:7"
)
NUMERIC_VERSION = (
    "arn:aws:lambda:us-west-2:246813579024:"
    "function:keep-glm52-h1g-numeric-binding:4"
)
SOURCE_KINDS = (
    "GPU_SPEND",
    "SUBMISSION_INTENT",
    "CONTROLLER_BASELINE",
    "CONTROL_PLANE_READINESS",
    "SUBMISSION_ACQUISITION",
)

METHOD_STEPS = {
    "prove_preauthorized_batch_template": CLOSURE_STEPS[0],
    "acquire_cfn_quiescence": CLOSURE_STEPS[1],
    "revalidate_runtime_attachments_and_seals": CLOSURE_STEPS[2],
    "warm_clients_and_construct": "CLIENT_WARMING_AND_IMMUTABLE_CONSTRUCTION",
    "size_non_authoritative": "NON_AUTHORITATIVE_SIZING",
    "stable_tls_sky_identity_preflight": "STABLE_TLS_SKY_IDENTITY_PREFLIGHT",
    "publish_gpu_spend_snapshot": CLOSURE_STEPS[3],
    "publish_submission_intent": CLOSURE_STEPS[3],
    "publish_controller_baseline": CLOSURE_STEPS[3],
    "publish_control_plane_readiness": CLOSURE_STEPS[3],
    "publish_submission_acquisition": CLOSURE_STEPS[3],
    "create_batch_successor": "CREATE_BATCH_SUCCESSOR",
    "authorize_change_set_create": "AUTHORIZE_CHANGE_SET_CREATE",
    "authorize_change_set_execute": "AUTHORIZE_CHANGE_SET_EXECUTE",
    "invoke_fence_executor_once": "INVOKE_FENCE_EXECUTOR_ONCE",
    "policy_deny_readback_one": "POLICY_DENY_READBACK_ONE",
    "policy_deny_readback_two": "POLICY_DENY_READBACK_TWO",
    "seal_decision_and_acquire_barrier": CLOSURE_STEPS[5],
    "attest_sky_identity_pre_decision": CLOSURE_STEPS[6],
    "create_or_recover_claim": CLOSURE_STEPS[7],
    "model_current_launch_or_expire": CLOSURE_STEPS[8],
    "create_direct_decision": CLOSURE_STEPS[9],
    "paginate_and_exact_read_decision": CLOSURE_STEPS[11],
    "validate_h1e_modeled_submit_once": CLOSURE_STEPS[12],
    "reinspect_task8_live_authority": CLOSURE_STEPS[13],
    "reattest_sky_identity": CLOSURE_STEPS[14],
    "recheck_all_authority": CLOSURE_STEPS[15],
    "arm_sky_action": CLOSURE_STEPS[16],
    "consume_admission_reservation": CLOSURE_STEPS[17],
    "coherent_readback_after_transactions": CLOSURE_STEPS[18],
    "invoke_launch_admission_version_once": CLOSURE_STEPS[19],
    "audit_handoff_authority": CLOSURE_STEPS[24],
    "persist_correlation_and_handoff": CLOSURE_STEPS[25],
    "release_to_numeric_binding": CLOSURE_STEPS[26],
}

SOURCE_METHODS = (
    "publish_gpu_spend_snapshot",
    "publish_submission_intent",
    "publish_controller_baseline",
    "publish_control_plane_readiness",
    "publish_submission_acquisition",
)
SUCCESSOR_AUDITS = {
    "create_batch_successor": "BATCH_SUCCESSOR",
}


def _request(**changes: object):
    values = {
        "activation_id": "approved-20260728",
        "generation": 1,
        "candidate_identity_sha256": SHA,
        "initial_source_predecessor_version_id": "predecessor-v1",
        "admission_version_arn": ADMISSION_VERSION,
        "numeric_binding_version_arn": NUMERIC_VERSION,
        "closure_budget_status": "CLOSURE_BUDGET_PROVEN",
    }
    values.update(changes)
    return build_closure_request(**values)


def _self_hashed(value: object) -> str:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return canonical_sha256(body)


def _attestation(label: str) -> AttestationResult:
    provisional = AttestationResult(
        schema_version=1,
        record_type="glm52_sky_attestation_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        freshness_nonce=label + "-freshness",
        direct_response_request_id=label + "-request",
        tls_peer_certificate_sha256=hashlib.sha256(
            (label + "-peer").encode()
        ).hexdigest(),
        sky_user_identity="svc-glm52",
        sky_roles=("glm52-production",),
        token_expires_at="2026-07-29T20:00:00Z",
        effective_controller_identity_sha256=hashlib.sha256(
            (label + "-controller").encode()
        ).hexdigest(),
        observed_at="2026-07-29T19:00:00Z",
        sky_identity_contract_sha256=hashlib.sha256(
            b"sky-contract"
        ).hexdigest(),
        canonical_identity_sha256="",
    )
    return AttestationResult(
        **{
            **asdict(provisional),
            "sky_roles": provisional.sky_roles,
            "canonical_identity_sha256": _self_hashed(provisional),
        }
    )


def _h1d() -> H1dLiveAuthorityResult:
    provisional = H1dLiveAuthorityResult(
        schema_version=1,
        record_type="glm52_h1d_live_authority_result_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        activation_id="approved-20260728",
        caller_identity_sha256=hashlib.sha256(b"caller").hexdigest(),
        expected_state_identity_sha256=hashlib.sha256(
            b"expected"
        ).hexdigest(),
        expected_state_authentication_identity_sha256=hashlib.sha256(
            b"expected-auth"
        ).hexdigest(),
        page_identities=(hashlib.sha256(b"page").hexdigest(),),
        family_identities={
            "cloudformation": hashlib.sha256(b"family").hexdigest()
        },
        spend_authority_identity_sha256=hashlib.sha256(
            b"spend"
        ).hexdigest(),
        sky_probe_identity_sha256=hashlib.sha256(b"probe").hexdigest(),
        phase_started_at="2026-07-29T19:00:00Z",
        phase_ended_at="2026-07-29T19:00:01Z",
        phase_elapsed_seconds=Decimal("1.000000"),
        canonical_identity_sha256="",
    )
    body = asdict(provisional)
    body.pop("canonical_identity_sha256")
    body["phase_elapsed_seconds"] = format(
        provisional.phase_elapsed_seconds,
        "f",
    )
    return H1dLiveAuthorityResult(
        **{
            **asdict(provisional),
            "page_identities": provisional.page_identities,
            "family_identities": provisional.family_identities,
            "phase_elapsed_seconds": provisional.phase_elapsed_seconds,
            "canonical_identity_sha256": canonical_sha256(body),
        }
    )


class RouteClock:
    def __init__(self) -> None:
        self.value = 0.0

    def clock(self) -> float:
        self.value += 0.01
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class ExactClosureServices:
    def __init__(
        self,
        *,
        mutate: str | None = None,
        fail_step: str | None = None,
        remaining: tuple[int, ...] = (300_000, 290_000, 280_000),
        classification: str = "ACCEPTED",
    ) -> None:
        self.calls: list[str] = []
        self.mutate = mutate
        self.fail_step = fail_step
        self.remaining = iter(remaining)
        self.classification = classification
        self.direct_objects: list[DirectDecisionResponse] = []
        self.owner_nonce_sha256 = ""
        self.decision_nonce_sha256 = ""

    def remaining_milliseconds(self) -> int:
        self.calls.append("REMAINING_TIME")
        return next(self.remaining)

    def coherent_readback_after_transaction(
        self,
        *,
        transaction_name: str,
        operation_identity_sha256: str,
        **kwargs: object,
    ):
        del kwargs
        self.calls.append("READBACK:" + transaction_name)
        coherent = not (
            self.mutate == "coherent_readback"
            and transaction_name == "authorize_change_set_execute"
        )
        return build_transaction_readback(
            transaction_name=transaction_name,
            operation_identity_sha256=operation_identity_sha256,
            index_coherent=coherent,
            control_coherent=True,
            action_coherent=True,
        )

    def __getattr__(self, method_name: str):
        if method_name not in METHOD_STEPS:
            raise AttributeError(method_name)
        step = METHOD_STEPS[method_name]

        def invoke(**kwargs: object):
            call_name = (
                method_name
                if method_name in (
                    *SOURCE_METHODS,
                    *SUCCESSOR_AUDITS,
                    "invoke_fence_executor_once",
                    "policy_deny_readback_one",
                    "policy_deny_readback_two",
                    "warm_clients_and_construct",
                    "size_non_authoritative",
                    "stable_tls_sky_identity_preflight",
                )
                else step
            )
            self.calls.append(call_name)
            if self.fail_step in (step, method_name):
                raise RuntimeError("injected " + step)
            audits = ()
            kinds = ()
            coherent = step == CLOSURE_STEPS[18]
            nonce = None
            if method_name in SOURCE_METHODS:
                index = SOURCE_METHODS.index(method_name)
                kind = SOURCE_KINDS[index]
                predecessor = str(kwargs["predecessor_version_id"])
                if self.mutate == "source_chain" and index == 2:
                    predecessor = "caller-version"
                return build_source_publication(
                    source_kind=kind,
                    predecessor_version_id=predecessor,
                    bucket="keep-glm52-campaign",
                    key=(
                        "campaigns/glm52-sky-20260724/task11/source-"
                        + str(index + 1)
                        + ".json"
                    ),
                    version_id="service-version-" + str(index + 1),
                    file_sha256=hashlib.sha256(
                        ("file:" + kind).encode()
                    ).hexdigest(),
                    body_sha256=hashlib.sha256(
                        ("body:" + kind).encode()
                    ).hexdigest(),
                    etag='"etag-' + str(index + 1) + '"',
                    checksum_sha256_base64="checksum-" + str(index + 1),
                    direct_request_id="request-" + str(index + 1),
                    direct_server_date=(
                        "Tue, 28 Jul 2026 18:00:00 GMT"
                    ),
                    direct_response_authenticated=True,
                    audit=build_audit_evidence(
                        kind="SOURCE_" + kind,
                        audit_identity_sha256=hashlib.sha256(
                            ("audit:" + kind).encode()
                        ).hexdigest(),
                        invocation_identity_sha256=hashlib.sha256(
                            ("walk:" + kind).encode()
                        ).hexdigest(),
                        closing_revision=index + 1,
                    ),
                )
            if method_name in SUCCESSOR_AUDITS:
                kind = SUCCESSOR_AUDITS[method_name]
                if (
                    self.mutate == "change_set_audit_replay"
                    and method_name == "authorize_change_set_execute"
                ):
                    kind = "CHANGE_SET_CREATE"
                audits = (
                    build_audit_evidence(
                        kind=kind,
                        audit_identity_sha256=hashlib.sha256(
                            ("audit:" + kind).encode()
                        ).hexdigest(),
                        invocation_identity_sha256=hashlib.sha256(
                            ("walk:" + kind).encode()
                        ).hexdigest(),
                        closing_revision=10,
                    ),
                )
                return build_step_receipt(
                    step_name=method_name.upper(),
                    operation_identity_sha256=hashlib.sha256(
                        method_name.encode()
                    ).hexdigest(),
                    audits=audits,
                )
            if method_name in (
                "invoke_fence_executor_once",
                "policy_deny_readback_one",
                "policy_deny_readback_two",
                "warm_clients_and_construct",
                "size_non_authoritative",
                "stable_tls_sky_identity_preflight",
            ):
                if method_name == "invoke_fence_executor_once":
                    fence_kinds = (
                        "CHANGE_SET_CREATE",
                        (
                            "CHANGE_SET_CREATE"
                            if self.mutate == "change_set_audit_replay"
                            else "CHANGE_SET_EXECUTE"
                        ),
                    )
                    return build_step_receipt(
                        step_name=step,
                        operation_identity_sha256=hashlib.sha256(
                            method_name.encode()
                        ).hexdigest(),
                        audits=tuple(
                            build_audit_evidence(
                                kind=kind,
                                audit_identity_sha256=hashlib.sha256(
                                    (
                                        "audit:"
                                        + kind
                                        + ":"
                                        + str(index)
                                    ).encode()
                                ).hexdigest(),
                                invocation_identity_sha256=hashlib.sha256(
                                    (
                                        "walk:"
                                        + kind
                                        + ":"
                                        + str(index)
                                    ).encode()
                                ).hexdigest(),
                                closing_revision=10 + index,
                            )
                            for index, kind in enumerate(fence_kinds)
                        ),
                        policy_readback_monotonic_seconds=(2.0, 12.0),
                    )
                return build_step_receipt(
                    step_name=step,
                    operation_identity_sha256=hashlib.sha256(
                        method_name.encode()
                    ).hexdigest(),
                )
            if step == CLOSURE_STEPS[7]:
                kinds = (AUTHORITY_AUDIT_KINDS[8],)
            elif step == CLOSURE_STEPS[6]:
                return _attestation("pre")
            elif step == CLOSURE_STEPS[8]:
                receipt = build_step_receipt(
                    step_name=step,
                    operation_identity_sha256=hashlib.sha256(
                        step.encode()
                    ).hexdigest(),
                )
                return build_launch_or_expire_decision(
                    branch="LAUNCH",
                    receipt=receipt,
                )
            elif step == CLOSURE_STEPS[13]:
                return _h1d()
            elif step == CLOSURE_STEPS[14]:
                return _attestation("repeat")
            elif step == CLOSURE_STEPS[9]:
                custody = str(kwargs["custody_nonce_sha256"])
                audit = build_audit_evidence(
                    kind=AUTHORITY_AUDIT_KINDS[9],
                    audit_identity_sha256=hashlib.sha256(
                        b"audit:DECISION"
                    ).hexdigest(),
                    invocation_identity_sha256=hashlib.sha256(
                        b"walk:DECISION"
                    ).hexdigest(),
                    closing_revision=20,
                )
                status = 409 if self.mutate == "decision_409" else 200
                return build_direct_decision_response(
                    status_code=status,
                    expected_bucket_owner="246813579024",
                    version_id="decision-version-1",
                    etag='"decision-etag"',
                    checksum_sha256=SHA,
                    request_id="request-id-1",
                    response_date="Tue, 28 Jul 2026 18:00:00 GMT",
                    candidate_identity_sha256=SHA,
                    audit=audit,
                    custody_nonce_sha256=custody,
                )
            elif step in (CLOSURE_STEPS[11], CLOSURE_STEPS[12]):
                direct = kwargs["direct_response"]
                assert isinstance(direct, DirectDecisionResponse)
                self.direct_objects.append(direct)
            elif step == CLOSURE_STEPS[16]:
                nonce = str(kwargs["owner_nonce_sha256"])
                self.owner_nonce_sha256 = nonce
            elif step == CLOSURE_STEPS[17]:
                nonce = str(kwargs["decision_nonce_sha256"])
                self.decision_nonce_sha256 = nonce
                assert kwargs["owner_nonce_sha256"] == self.owner_nonce_sha256
            elif step == CLOSURE_STEPS[19]:
                assert kwargs["version_arn"] == ADMISSION_VERSION
                result = build_admission_closure_result(
                    version_arn=ADMISSION_VERSION,
                    decision_nonce_sha256=self.decision_nonce_sha256,
                    internal_steps=CLOSURE_STEPS[20:24],
                    action_states=(
                        "POST_STARTED",
                        "POST_AUTHORIZED",
                        "POST_CLASSIFIED",
                    ),
                    post_audit=build_audit_evidence(
                        kind=AUTHORITY_AUDIT_KINDS[10],
                        audit_identity_sha256=hashlib.sha256(
                            b"audit:SKY_POST"
                        ).hexdigest(),
                        invocation_identity_sha256=hashlib.sha256(
                            b"walk:SKY_POST"
                        ).hexdigest(),
                        closing_revision=30,
                    ),
                    classification=self.classification,
                    invocation_count=1,
                    relay_call_count=1,
                    retry_count=(
                        1 if self.mutate == "admission_retry" else 0
                    ),
                    rearm_count=0,
                    relay_receipt_sha256=(
                        None if self.classification == "AMBIGUOUS" else SHA
                    ),
                    task9_result=AdmissionResult(
                        classification=self.classification,
                        durable_state="POST_CLASSIFIED",
                        response_identity_sha256=(
                            None
                            if self.classification == "AMBIGUOUS"
                            else SHA
                        ),
                        request_id=(
                            None
                            if self.classification == "AMBIGUOUS"
                            else "sky-request-1"
                        ),
                    ),
                )
                return result
            elif step == CLOSURE_STEPS[24]:
                kinds = (HANDOFF_AUDIT_KIND,)
            else:
                kinds = ()
            audits = tuple(
                build_audit_evidence(
                    kind=kind,
                    audit_identity_sha256=hashlib.sha256(
                        ("audit:" + kind).encode()
                    ).hexdigest(),
                    invocation_identity_sha256=hashlib.sha256(
                        ("walk:" + kind).encode()
                    ).hexdigest(),
                    closing_revision=40 + index,
                )
                for index, kind in enumerate(kinds)
            )
            return build_step_receipt(
                step_name=step,
                operation_identity_sha256=hashlib.sha256(
                    step.encode()
                ).hexdigest(),
                audits=audits,
                nonce_ownership_sha256=nonce,
                coherent_readback=coherent,
            )

        return invoke


def _nonces():
    values = iter((b"c" * 32, b"a" * 32, b"d" * 32))
    return lambda: next(values)


def _execute(
    services: ExactClosureServices,
    *,
    request: object = None,
):
    clock = RouteClock()
    return execute_decision_closure(
        _request() if request is None else request,
        services=services,
        nonce_source=_nonces(),
        monotonic_clock=clock.clock,
        sleeper=clock.sleep,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_kind", 7),
        ("predecessor_version_id", 8),
        ("version_id", 9),
        ("direct_response_authenticated", "false"),
        ("direct_response_authenticated", 1),
    ),
)
def test_task11_fix1_red_source_receipt_rejects_type_coercion(
    field: str,
    value: object,
) -> None:
    row = {
        "source_kind": "GPU_SPEND",
        "predecessor_version_id": "predecessor-v1",
        "bucket": "keep-glm52-campaign",
        "key": "campaigns/glm52-sky-20260724/source.json",
        "version_id": "service-version-1",
        "file_sha256": hashlib.sha256(b"file").hexdigest(),
        "body_sha256": hashlib.sha256(b"body").hexdigest(),
        "etag": '"etag"',
        "checksum_sha256_base64": "checksum",
        "direct_request_id": "request-1",
        "direct_server_date": "Tue, 28 Jul 2026 18:00:00 GMT",
        "direct_response_authenticated": True,
        "audit": build_audit_evidence(
            kind=AUTHORITY_AUDIT_KINDS[0],
            audit_identity_sha256=hashlib.sha256(b"audit").hexdigest(),
            invocation_identity_sha256=hashlib.sha256(b"walk").hexdigest(),
            closing_revision=1,
        ),
    }
    row[field] = value
    with pytest.raises(ClosureRouteError):
        build_source_batch_receipt((row,))


def test_task11_red_route_executes_exact_sequence_and_keeps_direct_custody() -> None:
    services = ExactClosureServices()
    result = _execute(services)
    assert result.status == "NUMERIC_BINDING_RECONCILIATION"
    assert result.steps == CLOSURE_STEPS
    assert result.authority_audit_kinds == AUTHORITY_AUDIT_KINDS
    assert result.handoff_audit_kind == HANDOFF_AUDIT_KIND
    assert result.classification == "ACCEPTED"
    assert len(services.direct_objects) == 2
    assert services.direct_objects[0] is services.direct_objects[1]
    assert tuple(
        name for name in services.calls if name in SOURCE_METHODS
    ) == SOURCE_METHODS
    assert tuple(
        name for name in services.calls if name in SUCCESSOR_AUDITS
    ) == tuple(SUCCESSOR_AUDITS)
    assert services.calls.count("invoke_fence_executor_once") == 1
    assert services.calls.count("policy_deny_readback_one") == 0
    assert services.calls.count("policy_deny_readback_two") == 0
    assert services.calls.count(CLOSURE_STEPS[19]) == 1
    assert services.calls.count("REMAINING_TIME") == 3


def test_task11_generated_workflow_terminates_after_one_numeric_binding() -> None:
    """Break caught: ASL invokes Numeric Binding again after Decision did it."""

    services = ExactClosureServices()
    definition = build_task11_workflow_definition()
    state_name = definition["StartAt"]
    workflow_input: dict[str, object] = {}
    terminal_output: object = None
    external_numeric_invocations = 0
    support_continuation_invocations = 0
    while True:
        state = definition["States"][state_name]
        if state["Type"] == "Choice":
            if state_name == "CHECK_TERMINAL_V2_RESULT":
                state_name = state["Choices"][0]["Next"]
            else:
                status = workflow_input["trusted_closure_budget"]["status"]
                state_name = (
                    state["Choices"][0]["Next"]
                    if status == "CLOSURE_BUDGET_PROVEN"
                    else state["Default"]
                )
            continue
        assert state["Type"] == "Task"
        resource = state["Resource"]
        if resource == {"Ref": "BudgetGateVersion"}:
            workflow_input["trusted_closure_budget"] = {
                "status": "CLOSURE_BUDGET_PROVEN"
            }
        elif resource == {"Ref": "DecisionVersion"}:
            terminal_output = _execute(services)
        elif resource == {"Ref": "NumericBindingVersion"}:
            external_numeric_invocations += 1
        elif resource == {"Ref": "SupportDeadlineVersion"}:
            support_continuation_invocations += 1
            if state_name == (
                "CREATE_OR_RECONCILE_PRODUCTION_TERMINAL_V2"
            ):
                workflow_input["terminal_attempt"] = {
                    "outcome": "SUCCEEDED"
                }
        else:
            raise AssertionError("workflow invoked a foreign task")
        if state.get("End") is True:
            break
        state_name = state["Next"]

    assert terminal_output is not None
    assert terminal_output.status == "NUMERIC_BINDING_RECONCILIATION"
    assert (
        services.calls.count(CLOSURE_STEPS[26])
        + external_numeric_invocations
        == 1
    )
    assert support_continuation_invocations == 13


@pytest.mark.parametrize(
    "mutation",
    (
        "source_chain",
        "change_set_audit_replay",
        "decision_409",
        "admission_retry",
    ),
)
def test_task11_red_route_rejects_cross_use_direct_and_retry_mutants(
    mutation: str,
) -> None:
    services = ExactClosureServices(mutate=mutation)
    with pytest.raises(ClosureRouteError):
        _execute(services)
    assert services.calls.count(CLOSURE_STEPS[19]) <= 1


@pytest.mark.parametrize(
    "version",
    (
        "keep-glm52-h1g-launch-admission",
        (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:keep-glm52-h1g-launch-admission"
        ),
        (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:keep-glm52-h1g-launch-admission:$LATEST"
        ),
        (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:keep-glm52-h1g-launch-admission:prod"
        ),
        (
            "arn:aws:lambda:us-west-2:000000000000:"
            "function:keep-glm52-h1g-launch-admission:7"
        ),
    ),
)
def test_task11_red_route_rejects_non_numeric_or_foreign_admission(
    version: str,
) -> None:
    with pytest.raises(ClosureRouteError):
        _request(admission_version_arn=version)


def test_task11_red_remaining_time_fails_before_token_consumption() -> None:
    services = ExactClosureServices(remaining=(300_000, 290_000, 119_999))
    with pytest.raises(StoredDecisionNoPostError) as caught:
        _execute(services)
    assert caught.value.status == "STORED_DECISION_NO_POST"
    assert CLOSURE_STEPS[17] not in services.calls
    assert CLOSURE_STEPS[19] not in services.calls


@pytest.mark.parametrize("classification", ("ACCEPTED", "KNOWN_REJECTED", "AMBIGUOUS"))
def test_task11_red_classification_never_resends(
    classification: str,
) -> None:
    services = ExactClosureServices(classification=classification)
    result = _execute(services)
    assert result.classification == classification
    assert services.calls.count(CLOSURE_STEPS[19]) == 1


def test_task11_red_post_decision_failure_is_permanently_no_post() -> None:
    services = ExactClosureServices(fail_step=CLOSURE_STEPS[13])
    with pytest.raises(StoredDecisionNoPostError) as caught:
        _execute(services)
    assert caught.value.status == "STORED_DECISION_NO_POST"
    assert CLOSURE_STEPS[17] not in services.calls
    assert CLOSURE_STEPS[19] not in services.calls


def test_task11_red_boundary_cannot_expose_stored_decision_or_post_retry() -> None:
    services = ExactClosureServices()
    services.submit_stored_decision = lambda: None
    with pytest.raises(ClosureRouteError, match="forbidden"):
        _execute(services)


@pytest.mark.parametrize(
    ("step", "stored_decision", "admission_count"),
    (
        (CLOSURE_STEPS[6], False, 0),
        (CLOSURE_STEPS[15], True, 0),
        (CLOSURE_STEPS[18], False, 0),
        (CLOSURE_STEPS[25], False, 1),
    ),
)
def test_task11_red_crash_boundaries_have_no_replay_edge(
    step: str,
    stored_decision: bool,
    admission_count: int,
) -> None:
    services = ExactClosureServices(fail_step=step)
    error = StoredDecisionNoPostError if stored_decision else ClosureRouteError
    with pytest.raises(error):
        _execute(services)
    assert services.calls.count(CLOSURE_STEPS[19]) == admission_count


@pytest.mark.parametrize(
    "missing",
    ("owner_hard_expired", "owner_execution_terminal", "zero_side_effect"),
)
def test_task11_red_arm_abandonment_requires_all_three_proofs(
    missing: str,
) -> None:
    proofs = {
        "owner_hard_expired": True,
        "owner_execution_terminal": True,
        "zero_side_effect": True,
    }
    proofs[missing] = False
    with pytest.raises(ClosureRouteError):
        transition_abandoned_arm(current_state="ARMED", **proofs)


def test_task11_red_arm_abandonment_exact_transition() -> None:
    assert transition_abandoned_arm(
        current_state="ARMED",
        owner_hard_expired=True,
        owner_execution_terminal=True,
        zero_side_effect=True,
    ) == "ABANDONED"
