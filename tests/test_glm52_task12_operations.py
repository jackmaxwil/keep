from __future__ import annotations

from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
import importlib.util
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_sha256


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _fixture_module(filename: str) -> object:
    spec = importlib.util.spec_from_file_location(
        "_task12_operation_fixture_" + filename.removesuffix(".py"),
        Path(__file__).with_name(filename),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request_payload(operation: str) -> dict[str, object]:
    from glm52_enforcement.task12_operations import OPERATION_SPECS

    if operation == "RECONCILE_RETAINED_LIFECYCLE_TRIGGER":
        runtime = _fixture_module("test_glm52_task12_runtime.py")
        return {
            "reconcile_retained_lifecycle_trigger_request": {
                "correlation_identity_sha256": "a" * 64,
                "runtime_scan": _jsonable(
                    runtime._scan(observed_at="2026-07-29T12:00:00Z")
                ),
            }
        }
    if operation == "RETAINED_ACQUIRE_RECOVERY_SEALING":
        retained = _fixture_module("test_glm52_task12_retained_state.py")
        return {
            "commit_recovery_seal_request": {
                "plan": _jsonable(retained._recovery_seal_plan()),
                "domain": "RECOVERY",
                "operation_identity_sha256": "b" * 64,
                "owner_nonce_capsule": {
                    "record_type": "glm52_task12_owner_nonce_capsule_v1"
                },
            }
        }

    spec = OPERATION_SPECS[operation]
    request: dict[str, object] = {}
    arrays = {
        "observations",
        "request_pages",
        "expected_retained",
        "retained_grant_baseline",
        "direct_grants",
        "service_grants",
    }
    for field in spec.request_fields:
        if field in {"minimum_quiet_seconds", "allocation_ordinal"}:
            request[field] = 1
        elif field == "stop_if_running":
            request[field] = True
        elif field in arrays:
            request[field] = [{"fixture": operation}]
        elif field.endswith(("_sha256", "_arn", "_id", "_at", "_hex")):
            request[field] = "fixture"
        elif field in {
            "activation_id",
            "authority_domain",
            "bucket",
            "domain",
            "expected_action",
            "expected_kms_key_arn",
            "expected_source_volume_id",
            "execution_arn",
            "file_sha256",
            "key",
            "owner_execution_status",
            "settling_deadline",
            "stack_id",
            "version_id",
            "writer_kind",
        }:
            request[field] = "fixture"
        else:
            request[field] = {"fixture": operation}
    return {spec.action_kind + "_request": request}


def test_snapshot_action_source_offsets_are_explicit_and_consumer_bound() -> None:
    from glm52_enforcement.task12_operations import (
        _expected_source_coordinates,
    )

    def offset(operation: str) -> int:
        sources = _expected_source_coordinates(
            operation_kind=operation,
            activation_id="activation-1",
            generation=1,
            campaign_bucket="keep-glm52-models",
        )
        action = next(
            source
            for source in sources
            if source["alias"] == "snapshot_cleanup_action"
        )
        return action["derive"]["integer_offset"]

    assert offset("SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY") == 1
    assert (
        offset("SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT")
        == 1
    )
    assert offset("SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT") == 0


def test_all_32_operations_have_distinct_closed_semantic_contracts() -> None:
    from glm52_enforcement.task12_operations import OPERATION_SPECS

    assert len(OPERATION_SPECS) == 32
    assert len({spec.result_kind for spec in OPERATION_SPECS.values()}) == 32
    assert all(spec.payload_fields for spec in OPERATION_SPECS.values())
    assert all(
        spec.operation_kind == operation
        for operation, spec in OPERATION_SPECS.items()
    )


def test_operation_row_builder_rejects_swapped_operation_payload() -> None:
    from glm52_enforcement.task12_operations import (
        Task12OperationContractError,
        build_operation_row,
    )

    with pytest.raises(
        Task12OperationContractError,
        match="payload fields are not exact",
    ):
        build_operation_row(
            operation_kind="RETAINED_ACQUIRE_RECOVERY_SEALING",
            activation_id="activation-1",
            activation_ordinal=1,
            generation=1,
            payload={
                "correlation_identity_sha256": "a" * 64,
                "runtime_scan": {},
            },
        )


def test_operation_row_builder_emits_exact_hashed_operation_schema() -> None:
    from glm52_enforcement.task12_operations import build_operation_row

    row = build_operation_row(
        operation_kind="RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        payload=_request_payload("RECONCILE_RETAINED_LIFECYCLE_TRIGGER"),
    )

    assert row["record_type"] == (
        "glm52_task12_reconcile_retained_lifecycle_trigger_input_v1"
    )
    assert row["handler_kind"] == "RETAINED_EXECUTION_OBSERVER"
    assert row["operation_kind"] == "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
    assert row["canonical_body_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in row.items()
            if key != "canonical_body_sha256"
        }
    )


def test_exact_ledger_authority_request_accepts_text_partition_and_sort_keys() -> None:
    from glm52_enforcement.task12_operations import (
        validate_operation_request_payload,
    )

    request = {
        "partition_key": "RUN#glm52-sky-20260724",
        "sort_key": (
            "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_ACTION#00000001"
        ),
        "expected_record_identity_sha256": "a" * 64,
    }

    assert validate_operation_request_payload(
        "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY",
        request,
    ) == request


def test_distinct_operations_execute_distinct_semantics_and_consume_predecessor() -> None:
    from glm52_enforcement.task12_operations import (
        build_operation_row,
        execute_operation_semantics,
    )

    first_kind = "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
    first = build_operation_row(
        operation_kind=first_kind,
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        payload=_request_payload(first_kind),
    )
    first_result = execute_operation_semantics(row=first, operation_input={})
    prior_body = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_result_v1",
        "handler_kind": "RETAINED_EXECUTION_OBSERVER",
        "operation_kind": first_kind,
        "operation_input_identity_sha256": canonical_sha256({}),
        "outcome": "SUCCEEDED",
        "result": {"result_kind": first_result.result_kind},
    }
    prior = {**prior_body, "canonical_body_sha256": canonical_sha256(prior_body)}
    second_kind = "RETAINED_ACQUIRE_RECOVERY_SEALING"
    second = build_operation_row(
        operation_kind=second_kind,
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        payload=_request_payload(second_kind),
    )
    second_result = execute_operation_semantics(
        row=second,
        operation_input={"task12_last_result": prior},
    )

    assert first_result.result_kind != second_result.result_kind
    assert first_result.action_kind != second_result.action_kind
    assert first_result.semantic_result != second_result.semantic_result
    assert second_result.predecessor_operation_kind == first_kind
    assert second_result.predecessor_result_identity_sha256 == prior[
        "canonical_body_sha256"
    ]


def test_semantic_executor_rejects_swapped_predecessor() -> None:
    from glm52_enforcement.task12_operations import (
        Task12OperationContractError,
        build_operation_row,
        execute_operation_semantics,
    )

    operation = "RETAINED_ACQUIRE_RECOVERY_SEALING"
    row = build_operation_row(
        operation_kind=operation,
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        payload=_request_payload(operation),
    )
    foreign_body = {
        "operation_kind": "RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
        "outcome": "SUCCEEDED",
    }
    foreign = {
        **foreign_body,
        "canonical_body_sha256": canonical_sha256(foreign_body),
    }

    with pytest.raises(
        Task12OperationContractError,
        match="exact predecessor",
    ):
        execute_operation_semantics(
            row=row,
            operation_input={"task12_last_result": foreign},
        )
