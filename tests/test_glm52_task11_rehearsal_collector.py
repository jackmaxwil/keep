"""Executable deployed Task 11 rehearsal collector and finalizer."""

from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.decision_closure import (
    CLOSURE_STEPS,
    build_closure_request,
    build_step_receipt,
)
from glm52_enforcement import support_rehearsal_collector_handler
from glm52_enforcement.task11_rehearsal_collector import (
    COLLECTOR_MEASUREMENT_IDS,
    CollectorConfig,
    CollectorRuntimeIdentity,
    CollectorServices,
    RehearsalCollectorError,
    collect_rehearsal,
    finalize_rehearsal_gate,
)
from glm52_enforcement.task11_production import (
    Task11AcceptedAdapters,
    Task11AwsClients,
    Task11ProductionConfig,
    Task11ProductionServices,
    Task11ReadOnlyPhaseServices,
)


ACCOUNT = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
ACTIVATION = "approved-20260728"
DEPLOYMENT = hashlib.sha256(b"task11-collector-deployment").hexdigest()
BUCKET = "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
VERSION_ARN = (
    "arn:aws:lambda:us-west-2:246813579024:function:"
    "keep-glm52-h1g-rehearsal-collector:19"
)
PROBE_VERSION_ARN = (
    "arn:aws:lambda:us-west-2:246813579024:function:"
    "keep-glm52-h1g-rehearsal-probe:7"
)


class Clock:
    def __init__(self) -> None:
        self.value = 1000.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    def utcnow(self) -> datetime:
        return datetime(2026, 7, 29, 20, 0, tzinfo=timezone.utc)


class Probe:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def inspect(self, request: object) -> object:
        assert type(request) is dict
        self.calls.append(dict(request))
        measurement_id = str(request["measurement_id"])
        suffix = measurement_id.removeprefix("measurement-")
        request_binding = {
            "activation_id": request["activation_id"],
            "measurement_id": measurement_id,
            "candidate_bucket": request["candidate_bucket"],
            "candidate_key": request["candidate_key"],
            "candidate_version_id": request["candidate_version_id"],
            "candidate_body_sha256": request["candidate_body_sha256"],
        }
        body = {
            "schema_version": 1,
            "record_type": "glm52_task11_rehearsal_probe_result_v1",
            "activation_id": ACTIVATION,
            "measurement_id": measurement_id,
            "candidate_bucket": BUCKET,
            "candidate_key": request["candidate_key"],
            "candidate_version_id": request["candidate_version_id"],
            "candidate_body_sha256": request["candidate_body_sha256"],
            "executed_version_arn": PROBE_VERSION_ARN,
            "probe_lambda_request_id": (
                "10000000-0000-4000-8000-" + suffix.zfill(12)
            ),
            "health_path": "/api/health",
            "health_request_identity_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "method": "GET",
                        "path": "/api/health",
                        **request_binding,
                    }
                )
            ).hexdigest(),
            "health_response_request_id": "health-request-" + suffix,
            "health_status_code": 200,
            "health_response_body_sha256": hashlib.sha256(
                ("health-body-" + measurement_id).encode()
            ).hexdigest(),
            "role_path": "/users/role",
            "role_request_identity_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "method": "GET",
                        "path": "/users/role",
                        **request_binding,
                    }
                )
            ).hexdigest(),
            "role_response_request_id": "role-request-" + suffix,
            "role_status_code": 200,
            "role_response_body_sha256": hashlib.sha256(
                ("role-body-" + measurement_id).encode()
            ).hexdigest(),
            "relay_call_count": 0,
            "sky_post_call_count": 0,
        }
        body["canonical_identity_sha256"] = hashlib.sha256(
            canonical_json_bytes(body)
        ).hexdigest()
        return body


class ProductionReadOnlyPhases:
    def __init__(self, clock: Clock, *, advance: bool = True) -> None:
        self.clock = clock
        self.advance = advance
        self.calls: list[tuple[str, object, str]] = []

    def _receipt(
        self,
        method_name: str,
        step_name: str,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        self.calls.append(
            (method_name, request, custody_nonce_sha256)
        )
        if self.advance:
            self.clock.value += 0.25
        return build_step_receipt(
            step_name=step_name,
            operation_identity_sha256=hashlib.sha256(
                (
                    method_name
                    + ":"
                    + custody_nonce_sha256
                    + ":"
                    + str(len(self.calls))
                ).encode("ascii")
            ).hexdigest(),
        )

    def prove_preauthorized_batch_template(self, **kwargs: object) -> object:
        return self._receipt(
            "prove_preauthorized_batch_template",
            CLOSURE_STEPS[0],
            **kwargs,
        )

    def acquire_cfn_quiescence(self, **kwargs: object) -> object:
        return self._receipt(
            "acquire_cfn_quiescence",
            CLOSURE_STEPS[1],
            **kwargs,
        )

    def revalidate_runtime_attachments_and_seals(
        self,
        **kwargs: object,
    ) -> object:
        return self._receipt(
            "revalidate_runtime_attachments_and_seals",
            CLOSURE_STEPS[2],
            **kwargs,
        )

    def warm_clients_and_construct(self, **kwargs: object) -> object:
        return self._receipt(
            "warm_clients_and_construct",
            "CLIENT_WARMING_AND_IMMUTABLE_CONSTRUCTION",
            **kwargs,
        )

    def size_non_authoritative(self, **kwargs: object) -> object:
        return self._receipt(
            "size_non_authoritative",
            "NON_AUTHORITATIVE_SIZING",
            **kwargs,
        )

    def stable_tls_sky_identity_preflight(
        self,
        **kwargs: object,
    ) -> object:
        return self._receipt(
            "stable_tls_sky_identity_preflight",
            "STABLE_TLS_SKY_IDENTITY_PREFLIGHT",
            **kwargs,
        )


class InjectedFaults:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.calls: list[str] = []

    def exercise(self, scenario: str) -> float:
        self.calls.append(scenario)
        if scenario == "NONE":
            return 0.0
        self.clock.value += 1.0
        return 1.0


class VersionedS3:
    def __init__(self, *, page_size: int = 1000) -> None:
        self.page_size = page_size
        self.objects: dict[str, tuple[str, bytes, str]] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.counter = 0

    @staticmethod
    def _checksum(raw: bytes) -> str:
        return base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")

    def put_object(self, **kwargs: object) -> object:
        self.calls.append(("put_object", dict(kwargs)))
        key = str(kwargs["Key"])
        raw = kwargs["Body"]
        assert type(raw) is bytes
        if key in self.objects:
            raise RuntimeError("PreconditionFailed")
        self.counter += 1
        version = f"version-{self.counter:04d}"
        checksum = self._checksum(raw)
        assert kwargs["ChecksumSHA256"] == checksum
        self.objects[key] = (version, raw, checksum)
        return {
            "VersionId": version,
            "ETag": '"' + hashlib.md5(raw).hexdigest() + '"',  # noqa: S324
            "ChecksumSHA256": checksum,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": f"put-{self.counter}",
                "HTTPHeaders": {
                    "date": "Wed, 29 Jul 2026 20:00:00 GMT",
                    "x-amz-version-id": version,
                    "x-amz-checksum-sha256": checksum,
                },
            },
        }

    def _selected(self, prefix: str) -> list[tuple[str, tuple[str, bytes, str]]]:
        return sorted(
            (key, value)
            for key, value in self.objects.items()
            if key.startswith(prefix)
        )

    def list_object_versions(self, **kwargs: object) -> object:
        self.calls.append(("list_object_versions", dict(kwargs)))
        rows = self._selected(str(kwargs["Prefix"]))
        marker = kwargs.get("KeyMarker")
        start = 0
        if marker is not None:
            names = [key for key, _value in rows]
            start = names.index(str(marker)) + 1
        page = rows[start : start + self.page_size]
        truncated = start + self.page_size < len(rows)
        result: dict[str, object] = {
            "Versions": [
                {
                    "Key": key,
                    "VersionId": value[0],
                    "IsLatest": True,
                    "Size": len(value[1]),
                }
                for key, value in page
            ],
            "DeleteMarkers": [],
            "IsTruncated": truncated,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": f"list-{len(self.calls)}",
            },
        }
        if marker is not None:
            result["KeyMarker"] = marker
            result["VersionIdMarker"] = kwargs["VersionIdMarker"]
        if truncated:
            result["NextKeyMarker"] = page[-1][0]
            result["NextVersionIdMarker"] = page[-1][1][0]
        return result

    def get_object(self, **kwargs: object) -> object:
        self.calls.append(("get_object", dict(kwargs)))
        key = str(kwargs["Key"])
        version, raw, checksum = self.objects[key]
        assert kwargs["VersionId"] == version
        return {
            "Body": io.BytesIO(raw),
            "VersionId": version,
            "ContentLength": len(raw),
            "ChecksumSHA256": checksum,
            "ETag": '"' + hashlib.md5(raw).hexdigest() + '"',  # noqa: S324
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": f"get-{len(self.calls)}",
                "HTTPHeaders": {
                    "date": "Wed, 29 Jul 2026 20:00:00 GMT",
                    "x-amz-version-id": version,
                    "x-amz-checksum-sha256": checksum,
                },
            },
        }

    def head_object(self, **kwargs: object) -> object:
        self.calls.append(("head_object", dict(kwargs)))
        key = str(kwargs["Key"])
        version, raw, checksum = self.objects[key]
        assert kwargs["VersionId"] == version
        return {
            "VersionId": version,
            "ContentLength": len(raw),
            "ChecksumSHA256": checksum,
            "ETag": '"' + hashlib.md5(raw).hexdigest() + '"',  # noqa: S324
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": f"head-{len(self.calls)}",
                "HTTPHeaders": {
                    "date": "Wed, 29 Jul 2026 20:00:00 GMT",
                    "x-amz-version-id": version,
                    "x-amz-checksum-sha256": checksum,
                },
            },
        }

    def get_bucket_policy(self, **kwargs: object) -> object:
        self.calls.append(("get_bucket_policy", dict(kwargs)))
        return {
            "Policy": json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Deny",
                            "Principal": "*",
                            "Action": "s3:GetObject",
                            "Resource": f"arn:aws:s3:::{BUCKET}/rehearsal/*",
                            "Condition": {
                                "StringNotEquals": {
                                    "aws:PrincipalAccount": ACCOUNT,
                                }
                            },
                        }
                    ],
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": f"policy-{len(self.calls)}",
            },
        }


def config() -> CollectorConfig:
    return CollectorConfig(
        account_id=ACCOUNT,
        region=REGION,
        run_id=RUN_ID,
        activation_id=ACTIVATION,
        deployment_identity_sha256=DEPLOYMENT,
        bucket=BUCKET,
        expected_bucket_owner=ACCOUNT,
        function_version_arn=VERSION_ARN,
        probe_version_arn=PROBE_VERSION_ARN,
    )


def production_request() -> object:
    return build_closure_request(
        activation_id=ACTIVATION,
        generation=1,
        candidate_identity_sha256=hashlib.sha256(
            b"deployed-task11-candidate"
        ).hexdigest(),
        initial_source_predecessor_version_id="predecessor-version-1",
        admission_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-launch-admission:7"
        ),
        numeric_binding_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-numeric-binding:4"
        ),
        closure_budget_status="CLOSURE_BUDGET_PROVEN",
    )


def services(
    *,
    s3: VersionedS3,
    clock: Clock,
    production_phases: object | None = None,
) -> CollectorServices:
    return CollectorServices(
        s3=s3,
        probe=Probe(),
        production_phases=(
            ProductionReadOnlyPhases(clock)
            if production_phases is None
            else production_phases
        ),
        faults=InjectedFaults(clock),
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
        utcnow=clock.utcnow,
    )


def runtime(index: int, *, cold: bool) -> CollectorRuntimeIdentity:
    return CollectorRuntimeIdentity(
        function_version_arn=VERSION_ARN,
        lambda_request_id=f"00000000-0000-4000-8000-{index:012d}",
        lambda_environment_id=hashlib.sha256(
            f"environment-{index}".encode()
        ).hexdigest(),
        cold_start=cold,
    )


def event(index: int) -> dict[str, object]:
    scenarios = {
        1: "THROTTLING",
        2: "PAGINATION",
        3: "NETWORK_AMBIGUITY",
    }
    return {
        "schema_version": 1,
        "record_type": "glm52_task11_collect_rehearsal_v1",
        "activation_id": ACTIVATION,
        "measurement_id": COLLECTOR_MEASUREMENT_IDS[index - 1],
        "scenario": scenarios.get(index, "NONE"),
        "task11_request": asdict(production_request()),
        "task11_boundary": {
            "bucket": "keep-glm52-campaign",
            "key": (
                "campaigns/glm52-sky-20260724/authorities/task11/"
                + ACTIVATION
                + "/00000001.json"
            ),
            "version_id": "task11-boundary-version-1",
            "file_sha256": hashlib.sha256(
                b"task11-boundary-file"
            ).hexdigest(),
            "body_sha256": hashlib.sha256(
                b"task11-boundary-body"
            ).hexdigest(),
        },
    }


def test_collect_schema_rejects_caller_selected_authority_before_s3() -> None:
    """Break caught: a caller can inject paths, timings, effects, or provenance."""

    s3 = VersionedS3()
    clock = Clock()
    value = event(1)
    value["effect_counts"] = [["SKY_POST", 0]]
    with pytest.raises(RehearsalCollectorError, match="field set"):
        collect_rehearsal(
            value,
            config=config(),
            runtime=runtime(1, cold=True),
            services=services(s3=s3, clock=clock),
        )
    assert s3.calls == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value["task11_request"].__setitem__(
                "canonical_identity_sha256",
                "0" * 64,
            ),
            "Task 11 request identity",
        ),
        (
            lambda value: value["task11_boundary"].__setitem__(
                "version_id",
                "null",
            ),
            "boundary coordinate",
        ),
    ),
)
def test_collect_rejects_substituted_deployed_coordinates_before_s3(
    mutate: object,
    message: str,
) -> None:
    """Break caught: rehearsal measures a caller-substituted Task 11 input."""

    s3 = VersionedS3()
    clock = Clock()
    value = event(1)
    mutate(value)
    with pytest.raises(RehearsalCollectorError, match=message):
        collect_rehearsal(
            value,
            config=config(),
            runtime=runtime(1, cold=True),
            services=services(s3=s3, clock=clock),
        )
    assert s3.calls == []


def test_collect_derives_runtime_and_canary_evidence_then_writes_once() -> None:
    """Break caught: deployed provenance is caller-authored or can reach production."""

    s3 = VersionedS3(page_size=1)
    clock = Clock()
    result = collect_rehearsal(
        event(1),
        config=config(),
        runtime=runtime(1, cold=True),
        services=services(s3=s3, clock=clock),
    )
    assert result["status"] == "DEPLOYED_REHEARSAL_RECORDED"
    assert result["measurement_id"] == "measurement-01"
    assert result["version_id"] == "version-0002"
    measurement_key = str(result["key"])
    assert measurement_key.startswith(
        f"rehearsal/measurements/{ACTIVATION}/{DEPLOYMENT}/"
    )
    raw = s3.objects[measurement_key][1]
    artifact = json.loads(raw)
    assert artifact["measurement"]["provenance"] == "DEPLOYED_REHEARSAL"
    assert artifact["measurement"]["lambda_environment_id"] == (
        runtime(1, cold=True).lambda_environment_id
    )
    assert artifact["measurement"]["cold_start"] is True
    assert artifact["measurement"]["effect_counts"] == [
        ["PRODUCTION_SOURCE", 0],
        ["PRODUCTION_CLAIM", 0],
        ["PRODUCTION_DECISION", 0],
        ["PRODUCTION_ACTION_CONSUME", 0],
        ["SKY_POST", 0],
    ]
    assert artifact["measurement"]["relay_call_count"] == 0
    closure_spans = artifact["measurement"]["closure_spans"][:4]
    assert all(
        span["ended_monotonic_seconds"]
        > span["started_monotonic_seconds"]
        for span in closure_spans
    )
    assert [
        row["operation"]
        for row in artifact["wire_ledger"][:6]
    ] == [
        "prove_preauthorized_batch_template",
        "acquire_cfn_quiescence",
        "revalidate_runtime_attachments_and_seals",
        "warm_clients_and_construct",
        "size_non_authoritative",
        "stable_tls_sky_identity_preflight",
    ]
    assert all(
        set(row)
        == {
            "operation",
            "step_name",
            "operation_identity_sha256",
            "receipt_identity_sha256",
            "phase_name",
        }
        for row in artifact["wire_ledger"][:6]
    )
    assert artifact["wire_ledger"]
    assert all(
        row["bucket"] == BUCKET
        and row["key"].startswith("rehearsal/")
        for row in artifact["wire_ledger"]
        if "bucket" in row
    )
    put_calls = [call for call in s3.calls if call[0] == "put_object"]
    assert len(put_calls) == 2
    assert all(call[1]["IfNoneMatch"] == "*" for call in put_calls)


def test_collect_rejects_zero_duration_production_phase() -> None:
    """Break caught: closure preflight phases are emitted without AWS reads."""

    s3 = VersionedS3()
    clock = Clock()
    with pytest.raises(
        RehearsalCollectorError,
        match="production phase timing did not advance",
    ):
        collect_rehearsal(
            event(4),
            config=config(),
            runtime=runtime(4, cold=True),
            services=services(
                s3=s3,
                clock=clock,
                production_phases=ProductionReadOnlyPhases(
                    clock,
                    advance=False,
                ),
            ),
        )
    assert s3.calls == []


def test_collect_calls_exact_named_production_phase_methods_in_order() -> None:
    """Break caught: closure-path measurements regress to proxy AWS reads."""

    s3 = VersionedS3()
    clock = Clock()
    phases = ProductionReadOnlyPhases(clock)
    collect_rehearsal(
        event(4),
        config=config(),
        runtime=runtime(4, cold=True),
        services=services(
            s3=s3,
            clock=clock,
            production_phases=phases,
        ),
    )
    assert [call[0] for call in phases.calls] == [
        "prove_preauthorized_batch_template",
        "acquire_cfn_quiescence",
        "revalidate_runtime_attachments_and_seals",
        "warm_clients_and_construct",
        "size_non_authoritative",
        "stable_tls_sky_identity_preflight",
    ]
    assert all(
        call[1] == production_request()
        for call in phases.calls
    )
    nonces = {call[2] for call in phases.calls}
    assert len(nonces) == 1
    assert next(iter(nonces)) == hashlib.sha256(
        (
            runtime(4, cold=True).lambda_request_id
            + ":"
            + config().deployment_identity_sha256
            + ":"
            + event(4)["measurement_id"]
        ).encode("ascii")
    ).hexdigest()


def test_collect_rejects_untyped_production_phase_receipt() -> None:
    """Break caught: proxy dictionaries are accepted as closure receipts."""

    class UntypedPhases(ProductionReadOnlyPhases):
        def acquire_cfn_quiescence(self, **kwargs: object) -> object:
            super().acquire_cfn_quiescence(**kwargs)
            return {
                "step_name": CLOSURE_STEPS[1],
                "operation_identity_sha256": "1" * 64,
            }

    s3 = VersionedS3()
    clock = Clock()
    with pytest.raises(
        RehearsalCollectorError,
        match="typed production receipt",
    ):
        collect_rehearsal(
            event(4),
            config=config(),
            runtime=runtime(4, cold=True),
            services=services(
                s3=s3,
                clock=clock,
                production_phases=UntypedPhases(clock),
            ),
        )
    assert s3.calls == []


def _production_phase_service(
    *,
    mutation_client: object | None = None,
    read_only_rehearsal: bool = False,
) -> Task11ProductionServices:
    closure_role_name = (
        "keep-glm52-h1g-closure-session-"
        + hashlib.sha256(ACTIVATION.encode("ascii")).hexdigest()[:16]
    )
    task11_config = Task11ProductionConfig(
        account_id=ACCOUNT,
        region=REGION,
        run_id=RUN_ID,
        activation_id=ACTIVATION,
        ledger_table_name="keep-glm52-ledger",
        campaign_bucket="keep-glm52-campaign",
        model_bucket="keep-glm52-campaign",
        model_prefix="models/glm52/",
        fence_stack_id="keep-glm52-fence",
        support_stack_id="keep-glm52-h1g-support",
        closure_role_arn=(
            f"arn:aws:iam::{ACCOUNT}:role/{closure_role_name}"
        ),
        attestation_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-attestation:3"
        ),
        launch_admission_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-launch-admission:7"
        ),
        numeric_binding_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-numeric-binding:4"
        ),
        source_gpu_spend_version_arn=PROBE_VERSION_ARN,
        source_submission_intent_version_arn=PROBE_VERSION_ARN,
        source_controller_baseline_version_arn=PROBE_VERSION_ARN,
        source_control_plane_readiness_version_arn=PROBE_VERSION_ARN,
        source_submission_acquisition_version_arn=PROBE_VERSION_ARN,
        fence_executor_version_arn=PROBE_VERSION_ARN,
        fence_successor_version_arn=PROBE_VERSION_ARN,
        claim_writer_version_arn=PROBE_VERSION_ARN,
        decision_writer_version_arn=PROBE_VERSION_ARN,
        terminal_v1_writer_version_arn=PROBE_VERSION_ARN,
        closure_handoff_version_arn=PROBE_VERSION_ARN,
        deployment_identity_sha256=DEPLOYMENT,
        decision_function_version_arn=VERSION_ARN,
    )
    caller = {
        "Account": ACCOUNT,
        "Arn": (
            f"arn:aws:sts::{ACCOUNT}:assumed-role/"
            f"{closure_role_name}/h1g-decision-session"
        ),
        "UserId": "AROACLOSURE:h1g-decision-session",
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "RequestId": "caller-request",
        },
    }
    clients = Task11AwsClients(
        s3=mutation_client,
        dynamodb=mutation_client,
        cloudformation=None,
        lambda_client=None,
        sts=None,
        caller_identity=caller,
    )
    return Task11ProductionServices(
        config=task11_config,
        adapters=Task11AcceptedAdapters(
            ledger=None,
            fresh_h1f=None,
            clients=clients,
        ),
        boundary=SimpleNamespace(),
        event={},
        context=SimpleNamespace(
            get_remaining_time_in_millis=lambda: 120_000
        ),
        read_only_rehearsal=read_only_rehearsal,
    )


def test_production_and_rehearsal_use_one_shared_phase_implementation() -> None:
    """Break caught: production and rehearsal phase entrypoints diverge."""

    direct = _production_phase_service()
    shared = _production_phase_service()
    kwargs = {
        "request": production_request(),
        "custody_nonce_sha256": hashlib.sha256(b"custody").hexdigest(),
    }
    assert direct.warm_clients_and_construct(**kwargs) == (
        shared.read_only_phases.warm_clients_and_construct(**kwargs)
    )
    assert Task11ReadOnlyPhaseServices._METHODS == (
        "prove_preauthorized_batch_template",
        "acquire_cfn_quiescence",
        "revalidate_runtime_attachments_and_seals",
        "warm_clients_and_construct",
        "size_non_authoritative",
        "stable_tls_sky_identity_preflight",
    )


def test_read_only_rehearsal_rejects_mutation_capable_client() -> None:
    """Break caught: rehearsal receives a Dynamo or write-capable SDK client."""

    class MutationClient:
        def put_object(self, **_kwargs: object) -> object:
            raise AssertionError("must never be reachable")

    with pytest.raises(
        RuntimeError,
        match="mutation-capable client",
    ):
        _production_phase_service(
            mutation_client=MutationClient(),
            read_only_rehearsal=True,
        )


def test_rehearsal_session_validator_binds_exact_in_graph_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: session evidence is account-root, foreign, or replayed."""

    controller_arn = (
        f"arn:aws:iam::{ACCOUNT}:role/"
        "keep-glm52-h1g-fence-executor"
    )
    executor_arn = (
        f"arn:aws:iam::{ACCOUNT}:role/"
        "keep-glm52-h1g-rehearsal-executor"
    )
    monkeypatch.setenv(
        "GLM52_REHEARSAL_CONTROLLER_ROLE_ARN",
        controller_arn,
    )
    monkeypatch.setenv(
        "GLM52_REHEARSAL_EXECUTOR_ROLE_ARN",
        executor_arn,
    )

    class Iam:
        def get_role(self, *, RoleName: str) -> object:
            if RoleName == "keep-glm52-h1g-fence-executor":
                role_arn = controller_arn
                role_id = "AROACONTROLLER"
                request_id = "controller-request"
            else:
                role_arn = executor_arn
                role_id = "AROAEXECUTOR"
                request_id = "executor-request"
            return {
                "Role": {"Arn": role_arn, "RoleId": role_id},
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": request_id,
                },
            }

    caller = {
        "Account": ACCOUNT,
        "Arn": (
            f"arn:aws:sts::{ACCOUNT}:assumed-role/"
            "keep-glm52-h1g-rehearsal-collector/lambda-session"
        ),
        "UserId": "AROACOLLECTOR:lambda-session",
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "RequestId": "caller-request",
        },
    }
    validator = (
        support_rehearsal_collector_handler._RehearsalSessionValidator(
            iam=Iam()
        )
    )
    binding = validator.bind_runtime_session(
        config=SimpleNamespace(
            account_id=ACCOUNT,
            activation_id=ACTIVATION,
        ),
        clients=SimpleNamespace(caller_identity=caller),
        semantic_authority=SimpleNamespace(
            role_ids_by_arn={
                controller_arn: "AROACONTROLLER",
                executor_arn: "AROAEXECUTOR",
            }
        ),
    )
    assert binding.controller_role_arn == controller_arn
    assert binding.executor_role_arn == executor_arn
    assert binding.executor_session_name == (
        "h1g-rehearsal-"
        + hashlib.sha256(ACTIVATION.encode("ascii")).hexdigest()[:16]
    )
    assert binding.canonical_identity_sha256

    caller["ResponseMetadata"]["RequestId"] = "executor-request"
    with pytest.raises(RuntimeError, match="replayed"):
        validator.bind_runtime_session(
            config=SimpleNamespace(
                account_id=ACCOUNT,
                activation_id=ACTIVATION,
            ),
            clients=SimpleNamespace(caller_identity=caller),
            semantic_authority=SimpleNamespace(
                role_ids_by_arn={
                    controller_arn: "AROACONTROLLER",
                    executor_arn: "AROAEXECUTOR",
                }
            ),
        )


def test_finalize_fully_paginates_exact_twenty_five_cold_and_three_faults() -> None:
    """Break caught: a partial or fixture-shaped set can enable production."""

    s3 = VersionedS3(page_size=4)
    clock = Clock()
    service = services(s3=s3, clock=clock)
    for index in range(1, 21):
        collect_rehearsal(
            event(index),
            config=config(),
            runtime=runtime(((index - 1) % 5) + 1, cold=index <= 5),
            services=service,
        )
    finalized = finalize_rehearsal_gate(
        {
            "schema_version": 1,
            "record_type": "glm52_task11_finalize_rehearsal_gate_v1",
            "activation_id": ACTIVATION,
        },
        config=config(),
        runtime=runtime(99, cold=False),
        services=service,
    )
    assert finalized["status"] == "CLOSURE_BUDGET_PROVEN"
    assert finalized["measurement_count"] == 20
    assert finalized["cold_environment_count"] == 5
    assert finalized["key"] == (
        f"rehearsal/gates/{ACTIVATION}/CLOSURE_BUDGET.json"
    )
    gate = json.loads(s3.objects[str(finalized["key"])][1])
    assert gate["measurement_count"] == 20
    assert gate["cold_environment_count"] == 5
    measurement_lists = [
        call
        for call in s3.calls
        if call[0] == "list_object_versions"
        and str(call[1]["Prefix"]).startswith("rehearsal/measurements/")
    ]
    assert len(measurement_lists) >= 5
    assert any("KeyMarker" in call[1] for call in measurement_lists[1:])


def test_finalize_rejects_reused_cold_environment_without_gate_write() -> None:
    """Break caught: five caller flags are counted as five cold environments."""

    s3 = VersionedS3(page_size=5)
    clock = Clock()
    service = services(s3=s3, clock=clock)
    for index in range(1, 21):
        collect_rehearsal(
            event(index),
            config=config(),
            runtime=runtime(1, cold=index <= 5),
            services=service,
        )
    with pytest.raises(RehearsalCollectorError, match="cold"):
        finalize_rehearsal_gate(
            {
                "schema_version": 1,
                "record_type": "glm52_task11_finalize_rehearsal_gate_v1",
                "activation_id": ACTIVATION,
            },
            config=config(),
            runtime=runtime(99, cold=False),
            services=service,
        )
    assert not any(
        key.startswith(f"rehearsal/gates/{ACTIVATION}/")
        for key in s3.objects
    )


def test_finalize_rejects_replayed_static_probe_evidence() -> None:
    """Break caught: one successful probe result is replayed twenty times."""

    class StaticProbe(Probe):
        def inspect(self, request: object) -> object:
            dynamic = super().inspect(request)
            body = dict(dynamic)
            body.pop("canonical_identity_sha256")
            body["measurement_id"] = "measurement-01"
            body["candidate_key"] = (
                f"rehearsal/canary/{ACTIVATION}/{DEPLOYMENT}/"
                "measurement-01.json"
            )
            body["candidate_version_id"] = "version-0001"
            body["candidate_body_sha256"] = "1" * 64
            body["probe_lambda_request_id"] = (
                "10000000-0000-4000-8000-000000000001"
            )
            body["health_request_identity_sha256"] = "2" * 64
            body["health_response_request_id"] = "health-request-static"
            body["health_response_body_sha256"] = "3" * 64
            body["role_request_identity_sha256"] = "4" * 64
            body["role_response_request_id"] = "role-request-static"
            body["role_response_body_sha256"] = "5" * 64
            body["canonical_identity_sha256"] = hashlib.sha256(
                canonical_json_bytes(body)
            ).hexdigest()
            return body

    s3 = VersionedS3(page_size=4)
    clock = Clock()
    service = services(s3=s3, clock=clock)
    object.__setattr__(service, "probe", StaticProbe())
    with pytest.raises(
        RehearsalCollectorError,
        match="probe request binding",
    ):
        for index in range(1, 21):
            collect_rehearsal(
                event(index),
                config=config(),
                runtime=runtime(
                    ((index - 1) % 5) + 1,
                    cold=index <= 5,
                ),
                services=service,
            )


def test_finalize_rejects_reused_probe_request_ids() -> None:
    """Break caught: request-bound bodies conceal reused transport evidence."""

    class ReusedRequestIdsProbe(Probe):
        def inspect(self, request: object) -> object:
            dynamic = super().inspect(request)
            body = dict(dynamic)
            body.pop("canonical_identity_sha256")
            body["probe_lambda_request_id"] = (
                "10000000-0000-4000-8000-000000000001"
            )
            body["health_response_request_id"] = "health-request-static"
            body["role_response_request_id"] = "role-request-static"
            body["canonical_identity_sha256"] = hashlib.sha256(
                canonical_json_bytes(body)
            ).hexdigest()
            return body

    s3 = VersionedS3(page_size=4)
    clock = Clock()
    service = services(s3=s3, clock=clock)
    object.__setattr__(service, "probe", ReusedRequestIdsProbe())
    for index in range(1, 21):
        collect_rehearsal(
            event(index),
            config=config(),
            runtime=runtime(
                ((index - 1) % 5) + 1,
                cold=index <= 5,
            ),
            services=service,
        )
    with pytest.raises(RehearsalCollectorError, match="probe uniqueness"):
        finalize_rehearsal_gate(
            {
                "schema_version": 1,
                "record_type": "glm52_task11_finalize_rehearsal_gate_v1",
                "activation_id": ACTIVATION,
            },
            config=config(),
            runtime=runtime(99, cold=False),
            services=service,
        )
    assert not any(
        key.startswith(f"rehearsal/gates/{ACTIVATION}/")
        for key in s3.objects
    )


def test_finalize_rejects_reused_production_receipt_identities() -> None:
    """Break caught: one set of phase receipts is replayed twenty times."""

    class ReusedProductionReceipts(ProductionReadOnlyPhases):
        def _receipt(
            self,
            method_name: str,
            step_name: str,
            *,
            request: object,
            custody_nonce_sha256: str,
        ) -> object:
            self.calls.append(
                (method_name, request, custody_nonce_sha256)
            )
            self.clock.value += 0.25
            return build_step_receipt(
                step_name=step_name,
                operation_identity_sha256=hashlib.sha256(
                    method_name.encode("ascii")
                ).hexdigest(),
            )

    s3 = VersionedS3(page_size=4)
    clock = Clock()
    service = services(
        s3=s3,
        clock=clock,
        production_phases=ReusedProductionReceipts(clock),
    )
    for index in range(1, 21):
        collect_rehearsal(
            event(index),
            config=config(),
            runtime=runtime(
                ((index - 1) % 5) + 1,
                cold=index <= 5,
            ),
            services=service,
        )
    with pytest.raises(
        RehearsalCollectorError,
        match="production receipt uniqueness",
    ):
        finalize_rehearsal_gate(
            {
                "schema_version": 1,
                "record_type": "glm52_task11_finalize_rehearsal_gate_v1",
                "activation_id": ACTIVATION,
            },
            config=config(),
            runtime=runtime(99, cold=False),
            services=service,
        )
    assert not any(
        key.startswith(f"rehearsal/gates/{ACTIVATION}/")
        for key in s3.objects
    )


def test_finalize_rejects_extra_measurement_and_delete_markers() -> None:
    """Break caught: an unknown or deleted measurement is ignored."""

    s3 = VersionedS3()
    clock = Clock()
    service = services(s3=s3, clock=clock)
    for index in range(1, 21):
        collect_rehearsal(
            event(index),
            config=config(),
            runtime=runtime(((index - 1) % 5) + 1, cold=index <= 5),
            services=service,
        )
    prefix = f"rehearsal/measurements/{ACTIVATION}/{DEPLOYMENT}/"
    s3.objects[prefix + "foreign.json"] = (
        "foreign-version",
        b"{}\n",
        VersionedS3._checksum(b"{}\n"),
    )
    with pytest.raises(RehearsalCollectorError, match="inventory"):
        finalize_rehearsal_gate(
            {
                "schema_version": 1,
                "record_type": "glm52_task11_finalize_rehearsal_gate_v1",
                "activation_id": ACTIVATION,
            },
            config=config(),
            runtime=runtime(99, cold=False),
            services=service,
        )


def test_handler_derives_environment_identity_and_cold_start_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: callers can author cold-start or environment evidence."""

    values = {
        "GLM52_ACCOUNT_ID": ACCOUNT,
        "AWS_REGION": REGION,
        "GLM52_RUN_ID": RUN_ID,
        "GLM52_ACTIVATION_ID": ACTIVATION,
        "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": DEPLOYMENT,
        "GLM52_REHEARSAL_BUCKET": BUCKET,
        "GLM52_EXPECTED_BUCKET_OWNER": ACCOUNT,
        "AWS_LAMBDA_FUNCTION_NAME": "keep-glm52-h1g-rehearsal-collector",
        "AWS_LAMBDA_FUNCTION_VERSION": "19",
        "GLM52_REHEARSAL_PROBE_VERSION_ARN": (
            PROBE_VERSION_ARN
        ),
        "AWS_LAMBDA_LOG_STREAM_NAME": "2026/07/29/[19]environment",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    observed: list[CollectorRuntimeIdentity] = []

    def fake_collect(
        request: object,
        *,
        config: CollectorConfig,
        runtime: CollectorRuntimeIdentity,
        services: CollectorServices,
    ) -> dict[str, object]:
        assert request == event(1)
        assert config.function_version_arn == VERSION_ARN
        assert type(services) is CollectorServices
        observed.append(runtime)
        return {"status": "OK"}

    class Context:
        aws_request_id = "00000000-0000-4000-8000-000000000001"
        invoked_function_arn = VERSION_ARN

    monkeypatch.setattr(
        support_rehearsal_collector_handler,
        "collect_rehearsal",
        fake_collect,
    )
    monkeypatch.setattr(
        support_rehearsal_collector_handler,
        "_COLD_START",
        True,
    )
    service = services(s3=VersionedS3(), clock=Clock())
    assert support_rehearsal_collector_handler.main(
        event(1),
        Context(),
        services=service,
    ) == {"status": "OK"}
    assert support_rehearsal_collector_handler.main(
        event(1),
        Context(),
        services=service,
    ) == {"status": "OK"}
    assert [item.cold_start for item in observed] == [True, False]
    assert observed[0].lambda_environment_id == observed[1].lambda_environment_id
    assert observed[0].lambda_environment_id == hashlib.sha256(
        canonical_json_bytes(
            {
                "deployment_identity_sha256": DEPLOYMENT,
                "function_version_arn": VERSION_ARN,
                "log_stream_name": "2026/07/29/[19]environment",
            }
        )
    ).hexdigest()
