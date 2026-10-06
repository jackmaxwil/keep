from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.dynamodb import encode_item


RUN_ID = "glm52-sky-20260724"
SHA_A = "a" * 64
SHA_B = "b" * 64
KMS_KEY = (
    "arn:aws:kms:us-west-2:246813579024:key/"
    "12345678-1234-4234-8234-1234567890ab"
)
ROOT = Path(__file__).resolve().parent


def _fixture(filename: str) -> object:
    spec = importlib.util.spec_from_file_location(
        "_drain_terminal_" + filename.removesuffix(".py"),
        ROOT / filename,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DB_FIXTURES = _fixture("test_glm52_enforcement_dynamodb.py")
WRITER_FIXTURES = _fixture("test_glm52_task12_writers.py")


def _invocation(operation: str) -> object:
    return SimpleNamespace(
        operation_kind=operation,
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        dispatch_identity_sha256=SHA_A,
        state_machine_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:activation-1"
        ),
        caller_state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained:1"
        ),
        operation_input={},
    )


def _metadata(request_id: str = "request-1", retries: int = 0):
    return {
        "HTTPStatusCode": 200,
        "RequestId": request_id,
        "RetryAttempts": retries,
    }


class _PagedDynamo:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = list(pages)
        self.calls: list[dict[str, object]] = []

    def query(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return self.pages.pop(0)


def test_owned_set_is_exact_but_only_direct_state_transitions_are_advertised() -> None:
    from glm52_enforcement.task12_live_drain_terminal import (
        OWNED_OPERATIONS,
        SUPPORTED_OPERATIONS,
    )

    assert OWNED_OPERATIONS == frozenset(
        {
            "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
            "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
            "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
            "RETAINED_ENTER_RECOVERY_COMPLETE",
            "RETAINED_ACQUIRE_TEARDOWN_SEALING",
            "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
            "RETAINED_ENTER_TEARDOWN_SEALED",
        }
    )
    assert SUPPORTED_OPERATIONS == frozenset(
        {
            "RETAINED_ENTER_RECOVERY_COMPLETE",
            "RETAINED_ACQUIRE_TEARDOWN_SEALING",
            "RETAINED_ENTER_TEARDOWN_SEALED",
        }
    )


def test_live_source_schema_rejects_future_or_precomputed_operation_payload() -> None:
    from glm52_enforcement.task12_live_drain_terminal import (
        build_live_domain_source,
    )

    with pytest.raises(ValueError, match="future operation payload"):
        build_live_domain_source(
            source_kind="TERMINAL_CLOSURE",
            activation_id="activation-1",
            generation=1,
            payload={
                "operation_kind": "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
                "request": {},
            },
        )


def test_plural_query_requires_complete_zero_retry_pages_and_no_duplicates() -> None:
    from glm52_enforcement.task12_live_drain_terminal import (
        query_complete_family,
    )

    record = DB_FIXTURES._closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        allocation_ordinal=1,
    )
    physical = encode_item(
        {
            "PK": "RUN#" + RUN_ID,
            "SK": "ACTIVATION#activation-1#WORKER_LAUNCH#00000001",
            **record,
        }
    )
    cursor = encode_item(
        {
            "PK": "RUN#" + RUN_ID,
            "SK": "ACTIVATION#activation-1#WORKER_LAUNCH#00000001",
        }
    )
    client = _PagedDynamo(
        [
            {
                "Items": [physical],
                "LastEvaluatedKey": cursor,
                "ResponseMetadata": _metadata("page-1"),
            },
            {
                "Items": [],
                "ResponseMetadata": _metadata("page-2"),
            },
        ]
    )

    records, identity = query_complete_family(
        client=client,
        table_name="ledger",
        activation_id="activation-1",
        record_type="glm52_production_worker_launch",
    )

    assert len(records) == 1
    assert len(identity) == 64
    assert len(client.calls) == 2
    assert all(call["ConsistentRead"] is True for call in client.calls)

    with pytest.raises(ValueError, match="zero-retry"):
        query_complete_family(
            client=_PagedDynamo(
                [
                    {
                        "Items": [],
                        "ResponseMetadata": _metadata(retries=1),
                    }
                ]
            ),
            table_name="ledger",
            activation_id="activation-1",
            record_type="glm52_production_worker_launch",
        )

    with pytest.raises(ValueError, match="duplicate"):
        query_complete_family(
            client=_PagedDynamo(
                [
                    {
                        "Items": [physical, physical],
                        "ResponseMetadata": _metadata(),
                    }
                ]
            ),
            table_name="ledger",
            activation_id="activation-1",
            record_type="glm52_production_worker_launch",
        )


class _EmptyDynamo:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query(self, **kwargs: object):
        self.calls.append(dict(kwargs))
        return {"Items": [], "ResponseMetadata": _metadata()}


class _LostPutDynamo:
    def __init__(self, existing: dict[str, object] | None = None) -> None:
        self.existing = existing

    def put_item(self, **kwargs: object):
        if self.existing is None:
            from glm52_enforcement.dynamodb import decode_item

            self.existing = decode_item(kwargs["Item"])
        raise TimeoutError("lost put response")

    def get_item(self, **kwargs: object):
        return {
            "Item": encode_item(self.existing),
            "ResponseMetadata": _metadata("readback"),
        }


def test_lost_source_write_adopts_only_exact_strong_readback() -> None:
    from glm52_enforcement.task12_live_drain_terminal import (
        build_live_domain_source,
        persist_exact_live_source,
    )

    source = build_live_domain_source(
        source_kind="REQUEST_JOB_CORRELATION",
        activation_id="activation-1",
        generation=1,
        payload={
            "request_ids": ["request-1"],
            "job_ids": ["17"],
            "observed_at": "2026-07-29T12:00:00Z",
        },
    )
    persist_exact_live_source(
        client=_LostPutDynamo(),
        table_name="ledger",
        partition_key="RUN#" + RUN_ID,
        sort_key="ACTIVATION#activation-1#TASK12_REQUEST_JOB_CORRELATION",
        source=source,
    )

    foreign = {
        "PK": "RUN#" + RUN_ID,
        "SK": "ACTIVATION#activation-1#TASK12_REQUEST_JOB_CORRELATION",
        **source,
        "canonical_body_sha256": SHA_A,
    }
    with pytest.raises(ValueError, match="foreign bytes"):
        persist_exact_live_source(
            client=_LostPutDynamo(existing=foreign),
            table_name="ledger",
            partition_key="RUN#" + RUN_ID,
            sort_key=foreign["SK"],
            source=source,
        )


class _Kms:
    def __init__(self) -> None:
        self.counter = ord("r")
        self.calls: list[dict[str, object]] = []

    def generate_data_key(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        value = self.counter
        self.counter += 1
        return {
            "KeyId": KMS_KEY,
            "Plaintext": bytes([value]) * 32,
            "CiphertextBlob": b"kms-ciphertext-" + bytes([value]),
            "ResponseMetadata": _metadata("kms-generate-" + str(value)),
        }


class _Ports:
    def __init__(self, dynamodb: object, kms: object | None = None) -> None:
        self._dynamodb = dynamodb
        self._kms = kms or _Kms()
        self.deployment = SimpleNamespace(
            role_coordinates={
                "ledger_table_name": "ledger",
                "campaign_bucket": WRITER_FIXTURES.BUCKET,
                "kms_key_id": KMS_KEY,
            }
        )

    def client(self, service: str) -> object:
        if service == "dynamodb":
            return self._dynamodb
        assert service == "kms"
        return self._kms


def test_unadvertised_drain_operations_have_no_hidden_materializer() -> None:
    from glm52_enforcement.task12_live_drain_terminal import (
        materialize_live_request,
    )

    for operation in sorted(
        {
            "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
            "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
            "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
            "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
        }
    ):
        assert materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={},
            ports=_Ports(_EmptyDynamo()),
        ) is None


def test_recovery_continuation_derives_full_context_from_current_owner() -> None:
    from glm52_enforcement.task12_live_drain_terminal import (
        _continuation_capsule,
    )

    owner = {
        "activation_id": "activation-1",
        "owner_attempt": 1,
        "owner_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:activation-1"
        ),
        "owner_state_machine_version_arn": (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained:1"
        ),
        "owner_invocation_nonce_sha256": SHA_A,
        "owner_hard_expires_at": "2099-07-29T13:00:00Z",
        "recovery_barrier_nonce_sha256": SHA_B,
        "support_control_revision_at_seal": 4,
    }
    context = {
        "account_id": "246813579024",
        "activation_id": "activation-1",
        "authority_domain": "RECOVERY",
        "barrier_nonce_sha256": SHA_A,
        "control_revision": "4",
        "owner_attempt": "1",
        "owner_execution_arn": owner["owner_execution_arn"],
        "owner_hard_expires_at": owner["owner_hard_expires_at"],
        "owner_state_machine_version_arn": owner[
            "owner_state_machine_version_arn"
        ],
        "region": "us-west-2",
        "run_id": RUN_ID,
    }
    invocation = SimpleNamespace(
        operation_input={
            "task12_last_result": {
                "result": {
                    "owner_nonce_capsule": {
                        "nonce_sha256": SHA_A,
                        "encryption_context": context,
                    }
                }
            }
        }
    )

    with pytest.raises(ValueError, match="authority drifted"):
        _continuation_capsule(
            invocation,
            owner,
            authority_domain="RECOVERY",
            barrier_field="recovery_barrier_nonce_sha256",
            control_revision_field="support_control_revision_at_seal",
        )


def test_recovery_complete_requires_actual_terminal_version_source() -> None:
    from glm52_enforcement.task12_live_drain_terminal import (
        materialize_live_request,
    )

    with pytest.raises(ValueError, match="TerminalV2 versioned source"):
        materialize_live_request(
            operation_kind="RETAINED_ENTER_RECOVERY_COMPLETE",
            invocation=_invocation("RETAINED_ENTER_RECOVERY_COMPLETE"),
            live_sources={},
            ports=_Ports(_EmptyDynamo()),
        )


def _terminal_version_control(terminal: dict[str, object]):
    from glm52_enforcement.task12_writers import (
        RetainedWriteResult,
        build_retained_writer_candidate,
        build_versioned_writer_control,
    )

    candidate = build_retained_writer_candidate(
        writer_kind="TerminalV2",
        campaign_bucket=WRITER_FIXTURES.BUCKET,
        activation_id="activation-1",
        generation=1,
        authority_domain="RECOVERY",
        record=terminal,
    )
    result_body = {
        "writer_kind": "TerminalV2",
        "outcome": "created-authenticated",
        "coordinate": candidate.coordinate,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "object_version_id": "terminal-version-1",
        "response_request_ids": ("writer-request-1",),
        "response_authenticated": True,
    }
    return build_versioned_writer_control(
        candidate=candidate,
        result=RetainedWriteResult(
            **result_body,
            canonical_identity_sha256=canonical_sha256(result_body),
        ),
        published_at="2026-07-29T12:00:00Z",
    )


class _StateDynamo(_EmptyDynamo):
    def __init__(self, records: dict[str, dict[str, object]]) -> None:
        super().__init__()
        self.records = records

    def get_item(self, **kwargs: object):
        from glm52_enforcement.dynamodb import decode_item

        key = decode_item(kwargs["Key"])
        record = self.records.get(key["SK"])
        response = {"ResponseMetadata": _metadata("get")}
        if record is not None:
            response["Item"] = encode_item(
                {"PK": key["PK"], "SK": key["SK"], **record}
            )
        return response


def _state_records(*, phase: str, recovery_state: str):
    from glm52_enforcement.records import ledger_sk

    return {
        ledger_sk(
            "glm52_production_activation_index"
        ): DB_FIXTURES._activation_index(),
        ledger_sk(
            "glm52_production_control", activation_id="activation-1"
        ): DB_FIXTURES._control(phase=phase, revision=4),
        ledger_sk(
            "glm52_production_recovery_control",
            activation_id="activation-1",
        ): DB_FIXTURES._closed_record(
            "glm52_production_recovery_control",
            activation_id="activation-1",
            state=recovery_state,
            revision=4,
        ),
        ledger_sk(
            "glm52_production_finalization_control",
            activation_id="activation-1",
        ): DB_FIXTURES._closed_record(
            "glm52_production_finalization_control",
            activation_id="activation-1",
        ),
    }


def test_recovery_complete_binds_actual_terminal_version_and_live_nonce() -> None:
    import hashlib

    from glm52_enforcement.records import (
        canonical_record_identity,
        ledger_sk,
    )
    from glm52_enforcement.task12_live_drain_terminal import (
        materialize_live_request,
    )

    terminal = WRITER_FIXTURES._terminal_v2()
    records = _state_records(
        phase="RECOVERY_SEALING",
        recovery_state="TERMINAL_V2_PUBLISHED",
    )
    recovery_key = ledger_sk(
        "glm52_production_recovery_control",
        activation_id="activation-1",
    )
    recovery = dict(records[recovery_key])
    recovery.update(
        owner_attempt=1,
        owner_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:activation-1"
        ),
        owner_state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:"
            "stateMachine:keep-glm52-h1g-retained:1"
        ),
        owner_dispatch_identity_sha256="d" * 64,
        owner_invocation_nonce_sha256=hashlib.sha256(
            b"r" * 32
        ).hexdigest(),
        owner_hard_expires_at="2099-07-29T13:00:00Z",
        recovery_barrier_nonce_sha256="c" * 64,
        support_control_revision_at_seal=4,
    )
    recovery["terminal_v2_identity_sha256"] = canonical_record_identity(
        "glm52_production_terminal_v2", terminal
    )
    records[recovery_key] = recovery
    version_control = _terminal_version_control(terminal)
    records[
        ledger_sk(
            "glm52_task12_versioned_writer_control_v1",
            activation_id="activation-1",
            generation=1,
            writer_kind="TerminalV2",
        )
    ] = version_control

    ports = _Ports(_StateDynamo(records))
    from glm52_enforcement.task12_nonce_capsule import (
        generate_owner_nonce_capsule,
    )

    capsule, _ = generate_owner_nonce_capsule(
        ports=ports,
        authority={
            "account_id": "246813579024",
            "region": "us-west-2",
            "run_id": RUN_ID,
            "activation_id": "activation-1",
            "authority_domain": "RECOVERY",
            "owner_execution_arn": recovery["owner_execution_arn"],
            "owner_state_machine_version_arn": recovery[
                "owner_state_machine_version_arn"
            ],
            "owner_attempt": recovery["owner_attempt"],
            "barrier_nonce_sha256": recovery[
                "recovery_barrier_nonce_sha256"
            ],
            "control_revision": recovery[
                "support_control_revision_at_seal"
            ],
            "owner_hard_expires_at": recovery["owner_hard_expires_at"],
        },
        now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )
    invocation = _invocation("RETAINED_ENTER_RECOVERY_COMPLETE")
    invocation.operation_input = {
        "task12_last_result": {"result": {"owner_nonce_capsule": capsule}}
    }
    request = materialize_live_request(
        operation_kind="RETAINED_ENTER_RECOVERY_COMPLETE",
        invocation=invocation,
        live_sources={
            "terminal_v2": terminal,
            "recovery_control": recovery,
        },
        ports=ports,
    )

    assert request["plan"]["control"]["after"]["phase"] == (
        "RECOVERY_COMPLETE"
    )
    assert request["plan"]["recovery_control"]["after"]["state"] == (
        "RECOVERY_COMPLETE"
    )
    assert request["plan"]["recovery_control"]["after"][
        "owner_invocation_nonce_sha256"
    ] is None
    assert request["owner_nonce_capsule"] == capsule

    foreign = dict(version_control)
    foreign["body_sha256"] = SHA_A
    foreign_body = dict(foreign)
    foreign_body.pop("canonical_body_sha256")
    foreign["canonical_body_sha256"] = canonical_sha256(foreign_body)
    records[
        ledger_sk(
            "glm52_task12_versioned_writer_control_v1",
            activation_id="activation-1",
            generation=1,
            writer_kind="TerminalV2",
        )
    ] = foreign
    with pytest.raises(ValueError, match="version binding"):
        materialize_live_request(
            operation_kind="RETAINED_ENTER_RECOVERY_COMPLETE",
            invocation=_invocation("RETAINED_ENTER_RECOVERY_COMPLETE"),
            live_sources={
                "terminal_v2": terminal,
                "recovery_control": recovery,
            },
            ports=_Ports(_StateDynamo(records)),
        )


def test_teardown_sealing_uses_exact_state_and_fresh_transaction_nonce() -> None:
    from glm52_enforcement.records import ledger_sk
    from glm52_enforcement.task12_live_drain_terminal import (
        materialize_live_request,
    )

    records = _state_records(
        phase="RECOVERY_COMPLETE",
        recovery_state="RECOVERY_COMPLETE",
    )
    recovery = records[
        ledger_sk(
            "glm52_production_recovery_control",
            activation_id="activation-1",
        )
    ]
    kwargs = {
        "operation_kind": "RETAINED_ACQUIRE_TEARDOWN_SEALING",
        "invocation": _invocation("RETAINED_ACQUIRE_TEARDOWN_SEALING"),
        "live_sources": {
            "recovery_control": recovery,
            "control": records[
                ledger_sk(
                    "glm52_production_control",
                    activation_id="activation-1",
                )
            ],
        },
        "ports": _Ports(_StateDynamo(records)),
    }
    request = materialize_live_request(**kwargs)
    second = materialize_live_request(**kwargs)

    assert request["plan"]["control"]["after"]["phase"] == (
        "TEARDOWN_SEALING"
    )
    assert request["owner_nonce_capsule"] != second["owner_nonce_capsule"]
    assert "raw_owner_nonce_hex" not in request


def test_teardown_owner_nonce_is_fresh_and_not_precomputed() -> None:
    from glm52_enforcement.task12_live_drain_terminal import (
        materialize_live_request,
    )

    records = _state_records(
        phase="TEARDOWN_SEALING",
        recovery_state="RECOVERY_COMPLETE",
    )
    from glm52_enforcement.records import ledger_sk

    kwargs = {
        "operation_kind": "RETAINED_ENTER_TEARDOWN_SEALED",
        "invocation": _invocation("RETAINED_ENTER_TEARDOWN_SEALED"),
        "live_sources": {
            "control": records[
                ledger_sk(
                    "glm52_production_control",
                    activation_id="activation-1",
                )
            ],
            "finalization_control": records[
                ledger_sk(
                    "glm52_production_finalization_control",
                    activation_id="activation-1",
                )
            ],
        },
        "ports": _Ports(_StateDynamo(records)),
    }

    first = materialize_live_request(**kwargs)
    second = materialize_live_request(**kwargs)

    assert first["owner_nonce_capsule"] != second["owner_nonce_capsule"]
    first_hash = first["owner_nonce_capsule"]["nonce_sha256"]
    assert first["plan"]["finalization_control"]["after"][
        "owner_invocation_nonce_sha256"
    ] == first_hash


@pytest.mark.parametrize(
    ("operation", "phase"),
    [
        ("RETAINED_ACQUIRE_TEARDOWN_SEALING", "TEARDOWN_SEALING"),
        ("RETAINED_ENTER_TEARDOWN_SEALED", "RECOVERY_COMPLETE"),
    ],
)
def test_invalid_teardown_state_fails_before_kms(
    operation: str, phase: str
) -> None:
    from glm52_enforcement.records import ledger_sk
    from glm52_enforcement.task12_live_drain_terminal import (
        materialize_live_request,
    )

    records = _state_records(
        phase=phase,
        recovery_state="RECOVERY_COMPLETE",
    )
    kms = _Kms()
    ports = _Ports(_StateDynamo(records), kms)
    live_sources = {
        "control": records[
            ledger_sk(
                "glm52_production_control",
                activation_id="activation-1",
            )
        ],
        "finalization_control": records[
            ledger_sk(
                "glm52_production_finalization_control",
                activation_id="activation-1",
            )
        ],
        "recovery_control": records[
            ledger_sk(
                "glm52_production_recovery_control",
                activation_id="activation-1",
            )
        ],
    }

    with pytest.raises(ValueError):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources=live_sources,
            ports=ports,
        )

    assert kms.calls == []
