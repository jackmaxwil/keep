from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from glm52_enforcement.task12_lambda_adapters import (
    _materialize_live_operation_request,
    _strict_operation_result,
)

KMS_KEY = (
    "arn:aws:kms:us-west-2:246813579024:key/"
    "12345678-1234-4234-8234-1234567890ab"
)


class _Kms:
    def generate_data_key(self, **kwargs: object) -> dict[str, object]:
        return {
            "KeyId": KMS_KEY,
            "Plaintext": b"n" * 32,
            "CiphertextBlob": b"kms-ciphertext",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "kms-generate",
                "RetryAttempts": 0,
            },
        }


class _Ports:
    def __init__(
        self,
        stepfunctions: object | None = None,
        *,
        now: datetime | None = None,
    ) -> None:
        self._stepfunctions = stepfunctions
        self._now = now
        self.deployment = SimpleNamespace(
            role_coordinates={"kms_key_id": KMS_KEY}
        )

    def client(self, service: str) -> object:
        if service == "kms":
            return _Kms()
        assert service == "stepfunctions"
        assert self._stepfunctions is not None
        return self._stepfunctions

    def now_utc(self) -> datetime:
        assert self._now is not None
        return self._now


def _invocation(operation_kind: str) -> object:
    return SimpleNamespace(operation_kind=operation_kind)


def test_live_stop_request_is_derived_only_from_canonical_execution() -> None:
    request = _materialize_live_operation_request(
        ports=_Ports(),
        invocation=_invocation(
            "RETAINED_STOP_OR_OBSERVE_SUPPORT_EXECUTION"
        ),
        live_sources={
            "execution": {
                "expected_execution_arn": "execution-arn",
                "expected_state_machine_version_arn": "workflow-version",
                "state": "RUNNING",
            }
        },
    )

    assert request == {
        "execution_arn": "execution-arn",
        "expected_state_machine_version_arn": "workflow-version",
        "stop_if_running": True,
    }


def test_live_owner_request_builds_the_real_conditional_classification() -> None:
    from test_glm52_task12_live_owner_death import (
        _invocation as owner_invocation,
        _sources as owner_sources,
    )

    class NoStepFunctionsRead:
        def describe_execution(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(
                "classifier must use the committed terminal EXECUTION source"
            )

    sources = owner_sources("CONSUMED")
    sources["recovery_control"]["owner_hard_expires_at"] = (
        "2026-07-30T12:00:00Z"
    )
    request = _materialize_live_operation_request(
        ports=_Ports(
            NoStepFunctionsRead(),
            now=datetime(2026, 7, 30, 11, 0, tzinfo=timezone.utc),
        ),
        invocation=owner_invocation(),
        live_sources=sources,
    )

    assert request["domain"] == "RECOVERY"
    assert request["plan"]["execution"]["expected"]["state"] == "FAILED"
    assert request["plan"]["action"]["before"]["state"] == "CONSUMED"
    assert (
        request["plan"]["action"]["after"]["outcome_class"]
        == "PROVED_NOT_SENT_OWNER_DIED"
    )


def test_live_snapshot_delete_request_uses_retained_control_snapshot() -> None:
    from test_glm52_task12_live_snapshot_finalization import (
        _invocation as _snapshot_invocation,
        _send_sources,
    )

    request = _materialize_live_operation_request(
        ports=_Ports(),
        invocation=_snapshot_invocation(
            "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
        ),
        live_sources=_send_sources(),
    )

    assert request == {"snapshot_id": "snap-0123456789abcdef0"}


def test_live_recovery_seal_materializes_a_real_transition_plan() -> None:
    from test_glm52_task12_retained_state import _recovery_seal_plan

    expected = _recovery_seal_plan()
    invocation = SimpleNamespace(
        operation_kind="RETAINED_ACQUIRE_RECOVERY_SEALING",
        activation_id="activation-1",
        activation_ordinal=1,
        handler_kind="RETAINED_EXECUTION_OBSERVER",
        state_machine_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained-lifecycle:recovery"
        ),
        caller_state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained-lifecycle:11"
        ),
        dispatch_identity_sha256="d" * 64,
    )

    request = _materialize_live_operation_request(
        ports=_Ports(),
        invocation=invocation,
        live_sources={
            "activation_index": expected.index.expected,
            "control": expected.control.before,
            "execution": expected.support_execution.expected,
            "recovery_control": expected.recovery_control.before,
        },
    )

    assert request["domain"] == "RECOVERY"
    assert request["owner_nonce_capsule"]["nonce_sha256"]
    assert "raw_owner_nonce_hex" not in request
    assert request["plan"]["control"]["after"]["phase"] == "RECOVERY_SEALING"
    assert request["plan"]["recovery_control"]["after"]["state"] == "OWNED"


def test_runtime_projection_materializes_request_but_cannot_replace_domain_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task12_lambda_adapters, task12_runtime
    from glm52_enforcement.task12_operations import build_operation_row
    from test_glm52_task12_runtime import _scan

    operation = "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
    scan = _scan(observed_at="2026-07-29T12:00:00Z")
    request = {
        "correlation_identity_sha256": "a" * 64,
        "runtime_scan": asdict(scan),
    }
    row = build_operation_row(
        operation_kind=operation,
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        payload={"reconcile_retained_lifecycle_trigger_request": request},
    )
    projection = {
        "record_type": "glm52_task12_live_projection_v1",
        "payload": request,
    }
    calls: list[str] = []

    monkeypatch.setattr(
        task12_lambda_adapters,
        "_read_live_operation_source",
        lambda **_kwargs: (row, {"operation_projection": projection}),
    )
    monkeypatch.setattr(
        task12_lambda_adapters,
        "_persist_live_operation_successors",
        lambda **_kwargs: None,
    )

    def collect(**_kwargs: object) -> object:
        calls.append("collect_runtime_scan")
        return scan

    monkeypatch.setattr(task12_runtime, "collect_runtime_scan", collect)
    invocation = SimpleNamespace(
        operation_kind=operation,
        operation_input={},
    )

    result = _strict_operation_result(
        ports=object(),
        invocation=invocation,
        state={
            "record_type": (
                "glm52_task12_reconcile_retained_lifecycle_trigger_descriptor_v1"
            )
        },
    )

    assert calls == ["collect_runtime_scan"]
    assert result is not None
    assert result["domain_result"]["spend"] == asdict(scan)["spend"]
    assert result["domain_result"] != projection
