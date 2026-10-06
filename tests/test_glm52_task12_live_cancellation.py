from __future__ import annotations

import io
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256


SHA_A = "a" * 64
SHA_B = "b" * 64


def _result(record_type: str, **values: object) -> bytes:
    body = {"schema_version": 1, "record_type": record_type, **values}
    return canonical_json_bytes(
        {**body, "canonical_identity_sha256": canonical_sha256(body)}
    )


class _Lambda:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.numeric_calls = 0

    def invoke(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        event = json.loads(kwargs["Payload"])
        function = kwargs["FunctionName"]
        if function.endswith("numeric-binding:7"):
            self.numeric_calls += 1
            state = "CANCELLED"
            payload = _result(
                "glm52_task12_runtime_observation_v1",
                run_id="glm52-sky-20260724",
                activation_id="activation-1",
                correlation_identity_sha256=event[
                    "correlation_identity_sha256"
                ],
                task9_deployed_identity_sha256=SHA_B,
                request_snapshot={
                    "requests": [
                        {
                            "request_id": "request-1",
                            "state": state,
                        }
                    ]
                },
                job_snapshot={"jobs": []},
            )
            version = "7"
        else:
            payload = _result(
                "glm52_task11_retained_cancellation_result_v1",
                run_id="glm52-sky-20260724",
                activation_id="activation-1",
                path=event["path"],
                request_body_sha256=event["request_body_sha256"],
                task9_deployed_identity_sha256=SHA_B,
                status_code=202,
                request_id="cancel-1",
                response_sha256=SHA_A,
                tls_peer_certificate_sha256=SHA_B,
            )
            version = "8"
        return {
            "StatusCode": 200,
            "ExecutedVersion": version,
            "Payload": io.BytesIO(payload),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "invoke-" + version,
                "RetryAttempts": 0,
            },
        }


class _Ports:
    def __init__(self, client: _Lambda) -> None:
        from glm52_enforcement.task11_boundary import (
            build_task11_input_coordinate,
        )

        self._client = client
        self.deployment = SimpleNamespace(
            role_coordinates={
                "numeric_binding_version_arn": (
                    "arn:aws:lambda:us-west-2:246813579024:function:"
                    "keep-glm52-h1g-numeric-binding:7"
                ),
                "retained_cancellation_version_arn": (
                    "arn:aws:lambda:us-west-2:246813579024:function:"
                    "keep-glm52-h1g-retained-cancellation:8"
                ),
                "task9_deployed_identity_coordinate": asdict(
                    build_task11_input_coordinate(
                        input_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
                        bucket="bucket",
                        key="task9-deployed-identity.json",
                        version_id="version-1",
                        file_sha256=SHA_A,
                        body_sha256=SHA_B,
                    )
                ),
                "task9_deployed_identity_sha256": SHA_B,
            }
        )

    def client(self, service: str) -> object:
        assert service == "lambda"
        return self._client


class _CorrelationDynamo:
    def __init__(self) -> None:
        self.item: dict[str, object] | None = None

    def put_item(self, **kwargs: object) -> dict[str, object]:
        from glm52_enforcement.dynamodb import decode_item

        self.item = decode_item(kwargs["Item"])
        return {"ResponseMetadata": {
            "HTTPStatusCode": 200,
            "RequestId": "put-correlation",
            "RetryAttempts": 0,
        }}


class _CorrelationPorts:
    def __init__(self, dynamodb: _CorrelationDynamo) -> None:
        self._dynamodb = dynamodb
        self.deployment = SimpleNamespace(
            role_coordinates={"ledger_table_name": "ledger"}
        )

    def client(self, service: str) -> object:
        assert service == "dynamodb"
        return self._dynamodb


def _invocation() -> object:
    return SimpleNamespace(
        operation_kind="RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION",
        activation_id="activation-1",
        generation=1,
        generation_text="00000001",
        dispatch_identity_sha256=SHA_A,
    )


def test_correlation_observes_runtime_and_conditionally_persists_canonical_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import (
        task12_lambda_adapters,
        task12_live_cancellation,
    )
    from glm52_enforcement.records import canonical_record_identity
    from test_glm52_enforcement_dynamodb import _closed_record

    execution = _closed_record(
        "glm52_production_execution",
        activation_id="activation-1",
    )
    source_identity = canonical_record_identity(
        "glm52_production_execution", execution
    )
    observed_at = "2026-07-29T12:00:00Z"
    observation = {
        "request_snapshot": {
            "observed_at": observed_at,
            "requests": [
                {"request_id": "request-1", "state": "RUNNING"},
            ],
        },
        "job_snapshot": {
            "observed_at": observed_at,
            "jobs": [
                {
                    "job_id": "17",
                    "request_id": "request-1",
                    "state": "RUNNING",
                },
            ],
        },
        "observed_at": observed_at,
        "canonical_identity_sha256": SHA_A,
    }
    monkeypatch.setattr(
        task12_lambda_adapters,
        "_read_exact_ledger_authority",
        lambda **_kwargs: execution,
    )
    monkeypatch.setattr(
        task12_live_cancellation,
        "_numeric_observation",
        lambda **kwargs: (
            observation
            if kwargs["correlation_identity_sha256"] == source_identity
            else pytest.fail("correlation source identity drifted")
        ),
    )
    dynamodb = _CorrelationDynamo()
    invocation = SimpleNamespace(
        operation_kind="RETAINED_CORRELATE_REQUESTS_AND_JOBS",
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
    )

    result = task12_lambda_adapters._correlate_and_persist_request_jobs(
        ports=_CorrelationPorts(dynamodb),
        invocation=invocation,
        request={
            "partition_key": "RUN#glm52-sky-20260724",
            "sort_key": "ignored-by-patched-reader",
            "expected_record_identity_sha256": source_identity,
        },
    )

    assert result["record_type"] == (
        "glm52_task12_request_job_correlation_v1"
    )
    assert result["request_ids"] == ["request-1"]
    assert result["job_ids"] == ["17"]
    assert dynamodb.item == {
        "PK": "RUN#glm52-sky-20260724",
        "SK": "ACTIVATION#activation-1#TASK12_REQUEST_JOB_CORRELATION",
        **result,
    }


def test_cancel_materializer_uses_only_exact_published_lambda_versions() -> None:
    from glm52_enforcement.task12_live_cancellation import (
        materialize_live_request,
    )

    client = _Lambda()
    request = materialize_live_request(
        operation_kind="RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION",
        invocation=_invocation(),
        live_sources={
            "request_job_correlation": {
                "record_type": "glm52_task12_request_job_correlation_v1",
                "request_ids": ["request-1"],
                "job_ids": [],
                "request_states": {"request-1": "RUNNING"},
                "job_states": {},
                "canonical_body_sha256": SHA_B,
            }
        },
        ports=_Ports(client),
    )

    assert request["command"]["path"] == "/api/cancel"
    assert request["observation"]["send_count"] == 1
    assert request["observation"]["observed_target_state"] == "CANCELLED"
    assert [call["FunctionName"].rsplit(":", 1)[1] for call in client.calls] == [
        "8",
        "7",
    ]
    assert all(
        call["InvocationType"] == "RequestResponse" for call in client.calls
    )


def test_cancel_materializer_rejects_unqualified_or_retried_lambda() -> None:
    from glm52_enforcement.task12_live_cancellation import (
        materialize_live_request,
    )

    client = _Lambda()
    ports = _Ports(client)
    ports.deployment.role_coordinates["numeric_binding_version_arn"] = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-h1g-numeric-binding"
    )
    with pytest.raises(ValueError, match="published Lambda version"):
        materialize_live_request(
            operation_kind="RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION",
            invocation=_invocation(),
            live_sources={
                "request_job_correlation": {
                    "record_type": "glm52_task12_request_job_correlation_v1",
                    "request_ids": ["request-1"],
                    "job_ids": [],
                    "request_states": {"request-1": "RUNNING"},
                    "job_states": {},
                    "canonical_body_sha256": SHA_B,
                }
            },
            ports=ports,
        )

    class RetriedLambda(_Lambda):
        def invoke(self, **kwargs: object) -> dict[str, object]:
            response = super().invoke(**kwargs)
            response["ResponseMetadata"]["RetryAttempts"] = 1
            return response

    with pytest.raises(ValueError, match="zero-retry"):
        materialize_live_request(
            operation_kind="RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION",
            invocation=_invocation(),
            live_sources={
                "request_job_correlation": {
                    "record_type": "glm52_task12_request_job_correlation_v1",
                    "request_ids": ["request-1"],
                    "job_ids": [],
                    "request_states": {"request-1": "RUNNING"},
                    "job_states": {},
                    "canonical_body_sha256": SHA_B,
                }
            },
            ports=_Ports(RetriedLambda()),
        )
