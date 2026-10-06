from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement import records as RECORD_CONTRACT


ROOT = Path(__file__).resolve().parent
SHA_A = "a" * 64
SHA_B = "b" * 64
TS = "2026-07-29T12:00:00Z"
EXECUTION = (
    "arn:aws:states:us-west-2:246813579024:execution:"
    "keep-glm52-h1g-retained:activation-1"
)
VERSION = (
    "arn:aws:states:us-west-2:246813579024:stateMachine:"
    "keep-glm52-h1g-retained:1"
)


def _fixture(filename: str) -> object:
    spec = importlib.util.spec_from_file_location(
        "_drain_effects_" + filename.removesuffix(".py"),
        ROOT / filename,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DB = _fixture("test_glm52_enforcement_dynamodb.py")
RECORD_FIXTURES = _fixture("test_glm52_enforcement_records.py")


def _invocation(operation: str) -> object:
    return SimpleNamespace(
        operation_kind=operation,
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        dispatch_identity_sha256=SHA_A,
        state_machine_execution_arn=EXECUTION,
        caller_state_machine_version_arn=VERSION,
        invoked_function_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-terminal-v2-writer:7"
        ),
        operation_input={},
    )


def _recovery(**updates: object) -> dict[str, object]:
    value = DB._closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
        state="OWNED",
    )
    value.update(
        owner_execution_arn=EXECUTION,
        owner_state_machine_version_arn=VERSION,
        owner_invocation_nonce_sha256=SHA_B,
        owner_hard_expires_at="2099-07-29T13:00:00Z",
        **updates,
    )
    return value


def _correlation() -> dict[str, object]:
    value = DB._closed_record(
        "glm52_task12_request_job_correlation_v1",
        activation_id="activation-1",
    )
    value.update(
        request_ids=["request-1"],
        job_ids=[],
        request_states={"request-1": "CANCELLED"},
        job_states={},
    )
    body = dict(value)
    body.pop("canonical_body_sha256")
    value["canonical_body_sha256"] = canonical_sha256(body)
    return value


def _control() -> dict[str, object]:
    return DB._closed_record(
        "glm52_production_control",
        activation_id="activation-1",
    )


def _rehash_nested(value: dict[str, object]) -> None:
    value["canonical_entry_sha256"] = canonical_sha256(
        {
            field: item
            for field, item in value.items()
            if field != "canonical_entry_sha256"
        }
    )


def _terminal_family_and_evidence(
    *,
    outcome: str,
    request_cardinality: str,
) -> tuple[
    dict[str, tuple[dict[str, object], ...]],
    dict[str, object],
]:
    terminal = RECORD_FIXTURES._terminal_for_outcome(
        RECORD_CONTRACT,
        outcome,
        request_cardinality,
    )
    launches: list[dict[str, object]] = []
    liabilities: list[dict[str, object]] = []
    launch_evidence = terminal["worker_launch_evidence"]
    liability_evidence = terminal["worker_launch_liabilities"]
    assert isinstance(launch_evidence, list)
    assert isinstance(liability_evidence, list)
    for nested_launch, nested_liability in zip(
        launch_evidence, liability_evidence, strict=True
    ):
        assert isinstance(nested_launch, dict)
        assert isinstance(nested_liability, dict)
        owner = nested_launch["owner_history"][-1]
        assert isinstance(owner, dict)
        ordinal = int(nested_launch["allocation_ordinal"])
        launch = DB._closed_record(
            "glm52_production_worker_launch",
            activation_id="activation-1",
            allocation_ordinal=ordinal,
            allocation_ordinal_text=f"{ordinal:08d}",
            state="ALLOCATION_CLOSED",
            ec2_client_token=nested_launch["ec2_client_token"],
            launch_parameters_sha256=nested_launch[
                "launch_parameters_sha256"
            ],
            expected_worker_tags_sha256=nested_launch[
                "expected_worker_tags_sha256"
            ],
            observed_instance_ids=nested_launch["observed_instance_ids"],
            run_instances_attempt_evidence=nested_launch[
                "run_instances_attempt_evidence"
            ],
            owner_attempt=owner["owner_attempt"],
            owner_principal_arn=owner["owner_principal_arn"],
            owner_function_version_arn=owner["owner_function_version_arn"],
            owner_dispatch_identity_sha256=owner[
                "owner_dispatch_identity_sha256"
            ],
            owner_invocation_nonce_sha256=owner[
                "owner_invocation_nonce_sha256"
            ],
            owner_hard_expires_at=owner["owner_hard_expires_at"],
        )
        liability = DB._closed_record(
            "glm52_production_worker_launch_liability",
            activation_id="activation-1",
            allocation_ordinal=ordinal,
            state="WATCHING",
            ec2_client_token=launch["ec2_client_token"],
            launch_parameters_sha256=launch["launch_parameters_sha256"],
            expected_worker_tags_sha256=launch[
                "expected_worker_tags_sha256"
            ],
            worker_launch_identity_sha256=(
                RECORD_CONTRACT.canonical_record_identity(
                    "glm52_production_worker_launch", launch
                )
            ),
        )
        nested_launch["worker_launch_identity_sha256"] = (
            RECORD_CONTRACT.canonical_record_identity(
                "glm52_production_worker_launch", launch
            )
        )
        nested_launch["worker_launch_liability_identity_sha256"] = (
            RECORD_CONTRACT.canonical_record_identity(
                "glm52_production_worker_launch_liability", liability
            )
        )
        nested_liability.update(
            activation_id=liability["activation_id"],
            activation_ordinal=liability["activation_ordinal"],
            generation=liability["generation"],
            worker_launch_identity_sha256=(
                nested_launch["worker_launch_identity_sha256"]
            ),
            worker_launch_liability_identity_sha256=(
                nested_launch["worker_launch_liability_identity_sha256"]
            ),
            ec2_client_token=liability["ec2_client_token"],
            launch_parameters_sha256=liability[
                "launch_parameters_sha256"
            ],
            expected_worker_tags_sha256=liability[
                "expected_worker_tags_sha256"
            ],
            state=liability["state"],
        )
        _rehash_nested(nested_launch)
        _rehash_nested(nested_liability)
        launches.append(launch)
        liabilities.append(liability)
    families = {
        "glm52_production_worker_launch": tuple(launches),
        "glm52_production_worker_launch_liability": tuple(liabilities),
        "glm52_production_worker_launch_liability_settlement": (),
        "glm52_production_post_terminal_allocation": (),
    }
    evidence_fields = {
        "handoff",
        "binding",
        "final_sky_state",
        "final_ec2_states",
        "request_cardinality",
        "allocations",
        "worker_launch_evidence",
        "worker_launch_liabilities",
        "final_heartbeat_identity",
        "checkpoint_identity",
        "cache_identity",
        "training_identity",
        "evaluation_identity",
        "drain_identity",
        "request_evidence",
        "post_terminal_quiescence_evidence",
        "prior_terminal_v1_identity",
        "outcome",
        "operator_disposition_required",
    }
    evidence: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task9_terminal_evidence_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": SHA_A,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "source_family_identity_sha256": canonical_sha256(
            {
                name: [
                    RECORD_CONTRACT.canonical_record_identity(name, row)
                    for row in rows
                ]
                for name, rows in sorted(families.items())
            }
        ),
        **{field: terminal[field] for field in evidence_fields},
        "evidence_created_at": "2026-07-29T11:59:00Z",
    }
    evidence["canonical_body_sha256"] = canonical_sha256(evidence)
    return families, evidence


class _NoEffects:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.deployment = SimpleNamespace(role_coordinates={})

    def client(self, service: str) -> object:
        self.calls.append(service)
        raise AssertionError("service effect occurred before producer validation")


def test_supported_operation_set_is_exact() -> None:
    from glm52_enforcement.task12_live_drain_effects import (
        SUPPORTED_OPERATIONS,
    )

    assert SUPPORTED_OPERATIONS == frozenset(
        {
            "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
            "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
            "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
            "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
        }
    )


@pytest.mark.parametrize(
    "operation",
    (
        "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
        "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
        "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
        "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
        "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
    ),
)
def test_missing_producer_fails_before_any_service_effect(operation: str) -> None:
    from glm52_enforcement.task12_live_drain_effects import (
        materialize_live_request,
    )

    ports = _NoEffects()
    with pytest.raises(ValueError, match="producer set"):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={},
            ports=ports,
        )
    assert ports.calls == []


@pytest.mark.parametrize(
    ("operation", "sources", "builder"),
    (
        (
            "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
            {"request_job_correlation": _correlation()},
            "_quiesce_request",
        ),
        (
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
            {"worker_drain_authority": _recovery()},
            "_transfer_request",
        ),
        (
            "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
            {"worker_drain_authority": _recovery()},
            "_worker_request",
        ),
        (
            "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
            {
                "recovery_control": _recovery(),
                "worker_drain_authority": _recovery(),
            },
            "_terminal_request",
        ),
        (
            "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
            {"control": _control()},
            "_zero_request",
        ),
    ),
)
def test_production_dispatch_selects_one_operation_specific_builder(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    sources: dict[str, dict[str, object]],
    builder: str,
) -> None:
    import glm52_enforcement.task12_live_drain_effects as live

    calls: list[tuple[object, object, object]] = []

    def selected(*, ports: object, invocation: object, sources: object):
        calls.append((ports, invocation, sources))
        return {"selected": operation}

    monkeypatch.setattr(live, builder, selected)
    ports = _NoEffects()
    assert live.materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=sources,
        ports=ports,
    ) == {"selected": operation}
    assert len(calls) == 1
    assert ports.calls == []


def test_terminal_rejects_swapped_recovery_producers_before_effect() -> None:
    from glm52_enforcement.task12_live_drain_effects import (
        materialize_live_request,
    )

    operation = "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
    stale = _recovery(revision=2)
    ports = _NoEffects()
    with pytest.raises(ValueError, match="swapped or stale"):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={
                "recovery_control": _recovery(),
                "worker_drain_authority": stale,
            },
            ports=ports,
        )
    assert ports.calls == []


def _proof(**updates: object):
    from glm52_enforcement.task12_live_drain_effects import (
        LiabilityTransferProof,
    )

    body = {
        "activation_id": "activation-1",
        "allocation_ordinals": (1,),
        "before_identity_sha256s": (SHA_A,),
        "after_identity_sha256s": (SHA_B,),
        "complete_family_identity_sha256": SHA_A,
        "owner_nonce_sha256": SHA_B,
        "state": "LIABILITIES_TRANSFERRED",
    }
    body.update(updates)
    return LiabilityTransferProof(
        **body, canonical_identity_sha256=canonical_sha256(body)
    )


def test_liability_transfer_proof_rejects_stale_or_duplicate_multiplicity() -> None:
    from glm52_enforcement.task12_live_drain_effects import (
        LiabilityTransferProof,
        validate_liability_transfer_proof,
    )

    assert validate_liability_transfer_proof(_proof()) == _proof()
    duplicate = _proof(
        allocation_ordinals=(1, 1),
        before_identity_sha256s=(SHA_A, SHA_A),
        after_identity_sha256s=(SHA_B, SHA_B),
    )
    with pytest.raises(ValueError, match="stale or incomplete"):
        validate_liability_transfer_proof(duplicate)
    stale = asdict(_proof())
    stale["state"] = "STALE"
    with pytest.raises(ValueError, match="stale or incomplete"):
        validate_liability_transfer_proof(
            LiabilityTransferProof(**stale)
        )


def test_controller_multiplicity_fails_before_ec2_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import glm52_enforcement.task12_live_runtime as runtime
    from glm52_enforcement.task12_live_drain_effects import (
        _controller_instance,
    )

    monkeypatch.setattr(
        runtime,
        "_numeric_observation",
        lambda **_kwargs: {
            "observed_at": TS,
            "controller_snapshot": {
                "members": [
                    {"identity": "i-0123456789abcdef0"},
                    {"identity": "i-1123456789abcdef0"},
                ]
            },
        },
    )
    ports = _NoEffects()
    with pytest.raises(ValueError, match="multiple"):
        _controller_instance(
            ports=ports,
            invocation=_invocation(
                "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED"
            ),
            correlation=SHA_A,
        )
    assert ports.calls == []


def test_quiesce_revalidates_current_owner_before_controller_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import glm52_enforcement.task12_live_drain_effects as live

    effects: list[str] = []

    def stale(**_kwargs: object) -> object:
        raise live.Task12LiveDrainEffectsError(
            "current recovery owner is absent or foreign"
        )

    monkeypatch.setattr(live, "_live_state", stale)
    monkeypatch.setattr(
        live,
        "_stop_controller",
        lambda **_kwargs: effects.append("stop"),
    )
    with pytest.raises(ValueError, match="current recovery owner"):
        live._quiesce_request(
            ports=object(),
            invocation=_invocation(
                "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED"
            ),
            sources={"request_job_correlation": _correlation()},
        )
    assert effects == []


def test_liability_lost_response_path_sends_one_owner_transaction_then_reads_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import glm52_enforcement.task12_live_drain_effects as live

    before = DB._closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
    )
    recovery = _recovery()
    after = live._liability_after(
        before,
        recovery,
        observed_at=TS,
        nonce_sha256=SHA_B,
    )
    empty = {
        "glm52_production_worker_launch": (),
        "glm52_production_worker_launch_liability_settlement": (),
        "glm52_production_post_terminal_allocation": (),
    }
    family_reads = iter(
        (
            {
                **empty,
                "glm52_production_worker_launch_liability": (before,),
            },
            {
                **empty,
                "glm52_production_worker_launch_liability": (after,),
            },
        )
    )
    sends: list[dict[str, object]] = []

    class Adapter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def acquire_liability_owner(self, **kwargs: object) -> object:
            sends.append(dict(kwargs))
            # Represents DynamoLedgerAdapter's exact-readback adoption after a
            # lost TransactWriteItems response.
            return SimpleNamespace(outcome="EXACT_DURABLE_ADOPTION")

    monkeypatch.setattr(live, "DynamoLedgerAdapter", Adapter)
    monkeypatch.setattr(
        live,
        "_live_state",
        lambda **_kwargs: {
            "recovery_control": recovery,
            "activation_index": DB._activation_index(),
        },
    )
    monkeypatch.setattr(
        live,
        "_owner_capsule",
        lambda **_kwargs: (
            {"nonce_sha256": SHA_B},
            b"r" * 32,
        ),
    )
    monkeypatch.setattr(
        live,
        "_families",
        lambda **_kwargs: next(family_reads),
    )
    monkeypatch.setattr(live, "_now", lambda: TS)
    monkeypatch.setattr(
        live,
        "_roles",
        lambda _ports: {"ledger_table_name": "ledger"},
    )
    ports = SimpleNamespace(client=lambda _service: object())
    proof = live._transfer_liabilities(
        ports=ports,
        invocation=_invocation(
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES"
        ),
        sources={"worker_drain_authority": recovery},
    )
    assert proof.state == "LIABILITIES_TRANSFERRED"
    assert proof.allocation_ordinals == (1,)
    assert len(sends) == 1


def test_worker_request_executes_real_current_authority_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import glm52_enforcement.task10_worker as task10
    import glm52_enforcement.task12_live_drain_effects as live
    import glm52_enforcement.task12_live_runtime as runtime
    import glm52_enforcement.task12_worker_drain as drain

    worker_fixtures = _fixture("test_glm52_task12_worker_drain.py")
    candidate = worker_fixtures._candidate()
    _units, scripts = worker_fixtures._hashes()
    recovery = _recovery()
    instance_id = worker_fixtures.INSTANCE_ID
    launch = {
        "allocation_ordinal": 1,
        "allocation_ordinal_text": "00000001",
        "observed_instance_ids": [instance_id],
        "sky_action_key": candidate.worker_descriptor.action_key,
        "sky_job_name": candidate.worker_descriptor.sky_job_name,
        "task_yaml_sha256": candidate.worker_observation.task_yaml_sha256,
        "request_body_sha256": (
            candidate.worker_observation.request_body_sha256
        ),
    }
    monkeypatch.setattr(
        live,
        "_live_state",
        lambda **_kwargs: {"recovery_control": recovery},
    )
    monkeypatch.setattr(
        live,
        "_families",
        lambda **_kwargs: {
            "glm52_production_worker_launch": (launch,),
        },
    )
    monkeypatch.setattr(live, "_execution", lambda **_kwargs: {"epoch": 1})
    monkeypatch.setattr(
        live,
        "_collect_scan",
        lambda **_kwargs: (
            SimpleNamespace(observed_at=TS),
            {"epoch": 1},
            "SUCCEEDED",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_spend_authority",
        lambda **_kwargs: (
            SimpleNamespace(
                remaining_gpu_seconds=85500,
                remaining_gpu_cost_usd="1307.20",
            ),
            (
                {
                    "state": "OPEN",
                    "instance_id": instance_id,
                    "job_id": "1",
                    "started_at": "2026-07-29T11:00:00Z",
                    "charged_seconds": 3600,
                    "charged_cost_usd": "55.04",
                },
            ),
            (),
            (),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_ec2_instances",
        lambda **_kwargs: (
            {instance_id: {"_state_name": "running"}},
            (),
        ),
    )
    monkeypatch.setattr(
        live,
        "_exact_s3_mapping",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        task10,
        "worker_bootstrap_descriptor_from_mapping",
        lambda _value: candidate.worker_descriptor,
    )
    monkeypatch.setattr(
        task10,
        "build_worker_instance_observation",
        lambda **_kwargs: candidate.worker_observation,
    )
    monkeypatch.setattr(
        live,
        "_reserve_rows",
        lambda **_kwargs: (
            {
                **dict(candidate.liability_reserve.record),
                "canonical_body_sha256": (
                    candidate.liability_reserve.reserve_identity_sha256
                ),
            },
        ),
    )
    monkeypatch.setattr(
        drain,
        "prepare_worker_drain_candidate",
        lambda **_kwargs: candidate,
    )
    monkeypatch.setattr(
        live,
        "_roles",
        lambda _ports: {
            "task10_worker_descriptor_coordinate": {
                "bucket": "keep-glm52-campaign",
                "key": "task13/production/task10-worker-descriptor.json",
                "version_id": "task10-worker-version-1",
                "file_sha256": SHA_A,
                "body_sha256": SHA_B,
            },
            "worker_script_hashes": scripts,
            "worker_drain_document_version": "7",
        },
    )
    monkeypatch.setattr(
        live,
        "_authority_pair",
        lambda **_kwargs: ({"action": "exact"}, {"audit": "exact"}),
    )
    result = live._worker_request(
        ports=object(),
        invocation=_invocation(
            "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS"
        ),
        sources={"worker_drain_authority": recovery},
    )
    assert result == {
        "candidate": asdict(candidate),
        "action": {"action": "exact"},
        "audit": {"audit": "exact"},
    }


def test_zero_launch_terminal_is_assembled_from_current_terminal_proof() -> None:
    from glm52_enforcement.records import validate_record
    from glm52_enforcement.task12_live_drain_effects import _terminal_base

    proof = SimpleNamespace(
        terminal_at=TS,
        first_scan_identity_sha256=SHA_A,
        second_scan_identity_sha256=SHA_B,
        final_spend_ledger_head_identity_sha256=SHA_A,
        canonical_identity_sha256=SHA_B,
    )
    second = SimpleNamespace(
        spend=SimpleNamespace(
            remaining_gpu_seconds=86400,
            remaining_gpu_cost_usd="1320.96",
        )
    )
    families = {
        "glm52_production_worker_launch": (),
        "glm52_production_worker_launch_liability": (),
    }
    terminal = _terminal_base(
        invocation=_invocation(
            "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
        ),
        recovery=_recovery(),
        proof=proof,
        second_scan=second,
        families=families,
    )
    assert terminal["worker_cardinality"] == "ZERO"
    assert terminal["request_cardinality"] == "NOT_APPLICABLE"
    assert validate_record(
        "glm52_production_terminal_v2", terminal
    ) == terminal


@pytest.mark.parametrize(
    ("outcome", "request_cardinality", "expected_cardinality"),
    (
        ("FAILED_TERMINAL_AFTER_ALLOCATION", "ONE", "ONE"),
        ("MULTIPLE_UNBOUND_WORKERS_INCIDENT", "ZERO", "MULTIPLE"),
    ),
)
def test_nonzero_terminal_uses_complete_current_task9_evidence(
    outcome: str,
    request_cardinality: str,
    expected_cardinality: str,
) -> None:
    from glm52_enforcement.records import validate_record
    from glm52_enforcement.task12_live_drain_effects import _terminal_base

    families, evidence = _terminal_family_and_evidence(
        outcome=outcome,
        request_cardinality=request_cardinality,
    )
    terminal = _terminal_base(
        invocation=_invocation(
            "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
        ),
        recovery=_recovery(),
        proof=SimpleNamespace(
            terminal_at=TS,
            first_scan_identity_sha256=SHA_A,
            second_scan_identity_sha256=SHA_B,
            final_spend_ledger_head_identity_sha256=SHA_A,
            canonical_identity_sha256=SHA_B,
        ),
        second_scan=SimpleNamespace(
            spend=SimpleNamespace(
                remaining_gpu_seconds=85500,
                remaining_gpu_cost_usd="1307.20",
            )
        ),
        families=families,
        terminal_evidence=evidence,
    )
    assert terminal["outcome"] == outcome
    assert terminal["worker_cardinality"] == expected_cardinality
    assert validate_record(
        "glm52_production_terminal_v2", terminal
    ) == terminal


def test_late_allocation_terminal_preserves_unbound_incident_evidence() -> None:
    from glm52_enforcement.task12_live_drain_effects import _terminal_base

    families, evidence = _terminal_family_and_evidence(
        outcome="UNBOUND_LATE_WORKER_INCIDENT",
        request_cardinality="ZERO",
    )
    terminal = _terminal_base(
        invocation=_invocation(
            "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
        ),
        recovery=_recovery(),
        proof=SimpleNamespace(
            terminal_at=TS,
            first_scan_identity_sha256=SHA_A,
            second_scan_identity_sha256=SHA_B,
            final_spend_ledger_head_identity_sha256=SHA_A,
            canonical_identity_sha256=SHA_B,
        ),
        second_scan=SimpleNamespace(
            spend=SimpleNamespace(
                remaining_gpu_seconds=85500,
                remaining_gpu_cost_usd="1307.20",
            )
        ),
        families=families,
        terminal_evidence=evidence,
    )
    assert terminal["outcome"] == "UNBOUND_LATE_WORKER_INCIDENT"
    assert terminal["binding"] is None
    assert terminal["request_cardinality"] == "ZERO"
    assert terminal["post_terminal_quiescence_evidence"] is not None


def test_nonzero_terminal_rejects_missing_or_incomplete_task9_producer() -> None:
    from glm52_enforcement.task12_live_drain_effects import _terminal_base

    families, evidence = _terminal_family_and_evidence(
        outcome="FAILED_TERMINAL_AFTER_ALLOCATION",
        request_cardinality="ONE",
    )
    common = {
        "invocation": _invocation(
            "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
        ),
        "recovery": _recovery(),
        "proof": SimpleNamespace(terminal_at=TS),
        "second_scan": object(),
        "families": families,
    }
    with pytest.raises(ValueError, match="schema is incomplete"):
        _terminal_base(**common)
    incomplete = dict(evidence)
    incomplete.pop("request_evidence")
    with pytest.raises(ValueError, match="schema is incomplete"):
        _terminal_base(**common, terminal_evidence=incomplete)


def test_terminal_evidence_read_is_paginated_current_and_version_exact() -> None:
    from glm52_enforcement.task12_live_drain_effects import (
        _read_terminal_evidence,
    )
    from glm52_enforcement.spend_authority import (
        canonical_decimal_json_bytes,
    )

    families, evidence = _terminal_family_and_evidence(
        outcome="FAILED_TERMINAL_AFTER_ALLOCATION",
        request_cardinality="ONE",
    )
    prefix = (
        "campaigns/glm52-sky-20260724/submissions/production/"
        "generations/00000001/terminal-evidence/"
    )
    key = prefix + "TASK9_TERMINAL_EVIDENCE.json"
    raw = canonical_decimal_json_bytes(evidence) + b"\n"
    calls: list[tuple[str, dict[str, object]]] = []

    class Body:
        def read(self) -> bytes:
            return raw

    class S3:
        def list_object_versions(self, **request: object) -> object:
            calls.append(("list", dict(request)))
            if "KeyMarker" not in request:
                return {
                    "Versions": [
                        {
                            "Key": key,
                            "VersionId": "old-version",
                            "IsLatest": False,
                        }
                    ],
                    "DeleteMarkers": [],
                    "IsTruncated": True,
                    "NextKeyMarker": key,
                    "NextVersionIdMarker": "old-version",
                    "ResponseMetadata": {
                        "HTTPStatusCode": 200,
                        "RequestId": "list-page-1",
                        "RetryAttempts": 0,
                    },
                }
            assert request["KeyMarker"] == key
            assert request["VersionIdMarker"] == "old-version"
            return {
                "Versions": [
                    {
                        "Key": key,
                        "VersionId": "current-version",
                        "IsLatest": True,
                    }
                ],
                "DeleteMarkers": [],
                "IsTruncated": False,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "list-page-2",
                    "RetryAttempts": 0,
                },
            }

        def get_object(self, **request: object) -> object:
            calls.append(("get", dict(request)))
            return {
                "Body": Body(),
                "VersionId": "current-version",
                "ETag": '"'
                + hashlib.md5(raw, usedforsecurity=False).hexdigest()
                + '"',
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "get-current-version",
                    "RetryAttempts": 0,
                },
            }

    s3 = S3()
    ports = SimpleNamespace(
        deployment=SimpleNamespace(
            role_coordinates={
                "campaign_bucket": "keep-glm52-campaign",
                "terminal_evidence_prefix": prefix,
            }
        ),
        client=lambda service: s3 if service == "s3" else None,
    )
    assert _read_terminal_evidence(
        ports=ports,
        invocation=_invocation(
            "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
        ),
        recovery=_recovery(),
        families=families,
    ) == evidence
    assert calls[-1] == (
        "get",
        {
            "Bucket": "keep-glm52-campaign",
            "Key": key,
            "VersionId": "current-version",
            "ExpectedBucketOwner": "246813579024",
            "ChecksumMode": "ENABLED",
        },
    )


@pytest.mark.parametrize("publication_fails", [False, True])
def test_terminal_writer_result_atomically_advances_exact_recovery_control(
    monkeypatch: pytest.MonkeyPatch,
    publication_fails: bool,
) -> None:
    import glm52_enforcement.task12_live_drain_effects as live
    from glm52_enforcement.task12_writers import (
        RetainedWriteResult,
        build_retained_writer_candidate,
        build_versioned_writer_control,
    )

    invocation = _invocation(
        "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
    )
    recovery = _recovery()
    control = _control()
    control.update(phase="RECOVERY_SEALING")
    proof = SimpleNamespace(
        terminal_at=TS,
        first_scan_identity_sha256=SHA_A,
        second_scan_identity_sha256=SHA_B,
        final_spend_ledger_head_identity_sha256=SHA_A,
        canonical_identity_sha256=SHA_B,
    )
    families, evidence = _terminal_family_and_evidence(
        outcome="DRAINED_COMPLETED",
        request_cardinality="ONE",
    )
    terminal = live._terminal_base(
        invocation=invocation,
        recovery=recovery,
        proof=proof,
        second_scan=SimpleNamespace(
            spend=SimpleNamespace(
                remaining_gpu_seconds=86400,
                remaining_gpu_cost_usd="1320.96",
            )
        ),
        families=families,
        terminal_evidence=evidence,
    )
    candidate = build_retained_writer_candidate(
        writer_kind="TerminalV2",
        campaign_bucket="campaign-bucket",
        activation_id="activation-1",
        generation=1,
        authority_domain="RECOVERY",
        record=terminal,
    )
    result_body = {
        "writer_kind": "TerminalV2",
        "outcome": "CREATED",
        "coordinate": candidate.coordinate,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "object_version_id": "version-1",
        "response_request_ids": ("request-1",),
        "response_authenticated": True,
    }
    result = RetainedWriteResult(
        **result_body,
        canonical_identity_sha256=canonical_sha256(result_body),
    )
    version_control = build_versioned_writer_control(
        candidate=candidate,
        result=result,
        published_at=TS,
    )
    commits: list[dict[str, object]] = []
    publications: list[dict[str, object]] = []

    class Adapter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def commit_recovery_progress(self, **kwargs: object) -> object:
            commits.append(dict(kwargs))
            return SimpleNamespace(outcome="APPLIED")

    monkeypatch.setattr(live, "DynamoLedgerAdapter", Adapter)
    monkeypatch.setattr(
        live,
        "_live_state",
        lambda **_kwargs: {
            "activation_index": DB._activation_index(),
            "control": control,
            "recovery_control": recovery,
        },
    )
    monkeypatch.setattr(
        "glm52_enforcement.task12_live_drain_terminal._read_exact_record",
        lambda **_kwargs: version_control,
    )
    monkeypatch.setattr(
        live,
        "_owner_capsule",
        lambda **_kwargs: (
            {"nonce_sha256": recovery["owner_invocation_nonce_sha256"]},
            b"r" * 32,
        ),
    )
    def publish(**kwargs: object) -> object:
        if publication_fails:
            raise ValueError("generation publication failed closed")
        publications.append(dict(kwargs))
        return {"outcome": "CREATED"}

    monkeypatch.setattr(
        "glm52_enforcement.campaign_drained_publication."
        "publish_campaign_drained",
        publish,
    )
    ports = SimpleNamespace(
        deployment=SimpleNamespace(
            role_coordinates={
                "campaign_bucket": "campaign-bucket",
                "ledger_table_name": "ledger",
                "task10_worker_descriptor_coordinate": {
                    "bucket": "campaign-bucket",
                    "key": "task13/production/task10-worker-descriptor.json",
                    "version_id": "descriptor-version-1",
                    "file_sha256": SHA_A,
                    "body_sha256": SHA_B,
                },
            }
        ),
        client=lambda service: object(),
    )
    call = lambda: live.persist_live_successors(
        operation_kind=invocation.operation_kind,
        invocation=invocation,
        live_sources={
            "recovery_control": recovery,
            "worker_drain_authority": recovery,
        },
        request={
            "writer_kind": "TerminalV2",
            "authority_domain": "RECOVERY",
            "record": terminal,
            "action": {},
            "audit": {},
        },
        domain_result=result,
        ports=ports,
    )
    if publication_fails:
        with pytest.raises(ValueError, match="failed closed"):
            call()
        assert publications == []
        assert commits == []
        return
    assert call()
    assert len(publications) == 1
    assert publications[0]["generation_text"] == "00000001"
    assert publications[0]["terminal_v2"] == terminal
    assert len(commits) == 1
    plan = commits[0]["plan"]
    assert plan.control.expected == control
    assert plan.recovery_control.before == recovery
    assert (
        plan.recovery_control.after["state"]
        == "TERMINAL_V2_PUBLISHED"
    )
    assert (
        plan.recovery_control.after["terminal_v2_identity_sha256"]
        == terminal["canonical_body_sha256"]
    )
    assert commits[0]["raw_owner_nonce"] == b"r" * 32
