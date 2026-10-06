"""Version-pinned Lambda route for deployed Task 11 rehearsals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from typing import Optional

from .canonical import canonical_json_bytes, canonical_sha256
from .task11_rehearsal_collector import (
    CollectorConfig,
    CollectorRuntimeIdentity,
    CollectorServices,
    collect_rehearsal,
    finalize_rehearsal_gate,
)
from .task11_boundary import Task11BoundaryCoordinate, load_task11_boundary
from .task11_production import (
    Task11AcceptedAdapters,
    Task11AwsClients,
    Task11ProductionConfig,
    Task11ProductionServices,
)


_COLD_START = True


def _configuration() -> CollectorConfig:
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
    function_version = os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "")
    region = os.environ.get("AWS_REGION", "")
    account_id = os.environ.get("GLM52_ACCOUNT_ID", "")
    function_arn = (
        f"arn:aws:lambda:{region}:{account_id}:function:"
        f"{function_name}:{function_version}"
    )
    return CollectorConfig(
        account_id=account_id,
        region=region,
        run_id=os.environ.get("GLM52_RUN_ID", ""),
        activation_id=os.environ.get("GLM52_ACTIVATION_ID", ""),
        deployment_identity_sha256=os.environ.get(
            "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
            "",
        ),
        bucket=os.environ.get("GLM52_REHEARSAL_BUCKET", ""),
        expected_bucket_owner=os.environ.get(
            "GLM52_EXPECTED_BUCKET_OWNER",
            "",
        ),
        function_version_arn=function_arn,
        probe_version_arn=os.environ.get(
            "GLM52_REHEARSAL_PROBE_VERSION_ARN",
            "",
        ),
    )


def _runtime_identity(
    *,
    config: CollectorConfig,
    context: object,
    cold_start: bool,
) -> CollectorRuntimeIdentity:
    request_id = getattr(context, "aws_request_id", None)
    invoked_arn = getattr(context, "invoked_function_arn", None)
    log_stream = os.environ.get("AWS_LAMBDA_LOG_STREAM_NAME", "")
    if (
        type(request_id) is not str
        or not request_id
        or invoked_arn != config.function_version_arn
        or not log_stream
    ):
        raise ValueError("collector Lambda runtime identity is incomplete")
    environment_id = canonical_sha256(
        {
            "deployment_identity_sha256": (
                config.deployment_identity_sha256
            ),
            "function_version_arn": config.function_version_arn,
            "log_stream_name": log_stream,
        }
    )
    return CollectorRuntimeIdentity(
        function_version_arn=config.function_version_arn,
        lambda_request_id=request_id,
        lambda_environment_id=environment_id,
        cold_start=cold_start,
    )


class _VersionPinnedProbe:
    def __init__(self, client: object, version_arn: str) -> None:
        self._client = client
        self._version_arn = version_arn
        self._qualifier = version_arn.rsplit(":", 1)[-1]

    def inspect(self, request: object) -> object:
        raw = canonical_json_bytes(request)
        response = self._client.invoke(
            FunctionName=self._version_arn,
            InvocationType="RequestResponse",
            Payload=raw,
        )
        if (
            type(response) is not dict
            or response.get("StatusCode") != 200
            or "FunctionError" in response
            or response.get("ExecutedVersion") != self._qualifier
        ):
            raise ValueError("rehearsal probe invocation was not version-pinned")
        payload = response.get("Payload")
        if not hasattr(payload, "read"):
            raise ValueError("rehearsal probe payload was absent")
        body = payload.read()
        if type(body) is not bytes or not body or len(body) > 128 * 1024:
            raise ValueError("rehearsal probe payload size drifted")
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("rehearsal probe payload was not JSON") from exc
        if canonical_json_bytes(value) != body:
            raise ValueError("rehearsal probe payload was not canonical")
        return value


class _ReadOnlyClient:
    """Expose only SDK inspection methods to the shared phase algorithms."""

    _PREFIXES = ("describe_", "get_", "head_", "list_")

    def __init__(self, client: object) -> None:
        self._client = client

    def __getattr__(self, name: str) -> object:
        if not name.startswith(self._PREFIXES):
            raise AttributeError(name)
        value = getattr(self._client, name)
        if not callable(value):
            raise AttributeError(name)
        return value


@dataclass(frozen=True)
class _RehearsalSessionBinding:
    controller_role_arn: str
    controller_role_id: str
    executor_role_arn: str
    executor_role_id: str
    executor_session_name: str
    collector_caller_arn: str
    controller_request_id: str
    executor_request_id: str
    caller_request_id: str
    canonical_identity_sha256: str


class _RehearsalSessionValidator:
    def __init__(self, *, iam: object) -> None:
        self._iam = iam

    @staticmethod
    def _metadata(value: object, label: str) -> str:
        metadata = (
            value.get("ResponseMetadata")
            if type(value) is dict
            else None
        )
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
        ):
            raise RuntimeError(label + " response is unauthenticated")
        return metadata["RequestId"]

    def validate_caller_identity(
        self, *, config: object, clients: object
    ) -> object:
        value = clients.caller_identity
        self._metadata(value, "rehearsal collector caller identity")
        expected = (
            "arn:aws:sts::"
            + config.account_id
            + ":assumed-role/keep-glm52-h1g-rehearsal-collector/"
        )
        if (
            type(value) is not dict
            or value.get("Account") != config.account_id
            or type(value.get("Arn")) is not str
            or not value["Arn"].startswith(expected)
            or type(value.get("UserId")) is not str
            or not value["UserId"]
        ):
            raise RuntimeError(
                "rehearsal collector caller identity drifted"
            )
        return value

    def bind_runtime_session(
        self,
        *,
        config: object,
        clients: object,
        semantic_authority: object,
    ) -> object:
        caller = self.validate_caller_identity(
            config=config,
            clients=clients,
        )
        controller_arn = os.environ.get(
            "GLM52_REHEARSAL_CONTROLLER_ROLE_ARN", ""
        )
        executor_arn = os.environ.get(
            "GLM52_REHEARSAL_EXECUTOR_ROLE_ARN", ""
        )
        expected_controller = (
            f"arn:aws:iam::{config.account_id}:role/"
            "keep-glm52-h1g-fence-executor"
        )
        expected_executor = (
            f"arn:aws:iam::{config.account_id}:role/"
            "keep-glm52-h1g-rehearsal-executor"
        )
        session_name = (
            "h1g-rehearsal-"
            + hashlib.sha256(
                config.activation_id.encode("ascii")
            ).hexdigest()[:16]
        )
        if (
            controller_arn != expected_controller
            or executor_arn != expected_executor
        ):
            raise RuntimeError("rehearsal session role coordinates drifted")
        controller = self._iam.get_role(
            RoleName=controller_arn.rsplit("/", 1)[1]
        )
        executor = self._iam.get_role(
            RoleName=executor_arn.rsplit("/", 1)[1]
        )
        controller_request = self._metadata(
            controller, "rehearsal controller role"
        )
        executor_request = self._metadata(
            executor, "rehearsal executor role"
        )
        controller_role = controller.get("Role")
        executor_role = executor.get("Role")
        role_ids = semantic_authority.role_ids_by_arn
        if (
            type(controller_role) is not dict
            or type(executor_role) is not dict
            or controller_role.get("Arn") != controller_arn
            or executor_role.get("Arn") != executor_arn
            or role_ids.get(controller_arn) != controller_role.get("RoleId")
            or role_ids.get(executor_arn) != executor_role.get("RoleId")
        ):
            raise RuntimeError("rehearsal session graph drifted")
        caller_request = caller["ResponseMetadata"]["RequestId"]
        if len(
            {controller_request, executor_request, caller_request}
        ) != 3:
            raise RuntimeError("rehearsal session evidence was replayed")
        body = {
            "controller_role_arn": controller_arn,
            "controller_role_id": controller_role["RoleId"],
            "executor_role_arn": executor_arn,
            "executor_role_id": executor_role["RoleId"],
            "executor_session_name": session_name,
            "collector_caller_arn": caller["Arn"],
            "controller_request_id": controller_request,
            "executor_request_id": executor_request,
            "caller_request_id": caller_request,
        }
        return _RehearsalSessionBinding(
            **body,
            canonical_identity_sha256=canonical_sha256(body),
        )


class _FaultExercises:
    """Execute the frozen local failure-unwind paths without AWS mutations."""

    def __init__(self, monotonic: object) -> None:
        self._monotonic = monotonic

    def exercise(self, scenario: str) -> float:
        started = self._monotonic()
        if scenario == "NONE":
            return 0.0
        try:
            if scenario == "THROTTLING":
                raise RuntimeError("injected throttling response")
            if scenario == "PAGINATION":
                pages = (
                    {"IsTruncated": True, "NextKeyMarker": "next"},
                    {"IsTruncated": False},
                )
                if (
                    pages[0]["IsTruncated"] is not True
                    or pages[0]["NextKeyMarker"] != "next"
                    or pages[1]["IsTruncated"] is not False
                ):
                    raise RuntimeError("injected pagination response drifted")
                raise RuntimeError("injected pagination boundary")
            if scenario == "NETWORK_AMBIGUITY":
                raise TimeoutError("injected ambiguous network response")
            raise ValueError("injected failure scenario drifted")
        except (RuntimeError, TimeoutError):
            return float(self._monotonic() - started)


def _task11_configuration(config: CollectorConfig) -> Task11ProductionConfig:
    return Task11ProductionConfig(
        account_id=config.account_id,
        region=config.region,
        run_id=config.run_id,
        activation_id=config.activation_id,
        ledger_table_name=os.environ.get("GLM52_LEDGER_TABLE_NAME", ""),
        campaign_bucket=os.environ.get("GLM52_CAMPAIGN_BUCKET", ""),
        model_bucket=os.environ.get("GLM52_MODEL_BUCKET", ""),
        model_prefix=os.environ.get("GLM52_MODEL_PREFIX", ""),
        fence_stack_id=os.environ.get("GLM52_FENCE_STACK_ID", ""),
        support_stack_id=os.environ.get("GLM52_SUPPORT_STACK_ID", ""),
        closure_role_arn=os.environ.get("GLM52_CLOSURE_ROLE_ARN", ""),
        attestation_version_arn=os.environ.get(
            "GLM52_ATTESTATION_VERSION_ARN", ""
        ),
        launch_admission_version_arn=os.environ.get(
            "GLM52_LAUNCH_ADMISSION_VERSION_ARN", ""
        ),
        numeric_binding_version_arn=os.environ.get(
            "GLM52_NUMERIC_BINDING_VERSION_ARN", ""
        ),
        source_gpu_spend_version_arn=os.environ.get(
            "GLM52_SOURCE_GPU_SPEND_VERSION_ARN", ""
        ),
        source_submission_intent_version_arn=os.environ.get(
            "GLM52_SOURCE_SUBMISSION_INTENT_VERSION_ARN", ""
        ),
        source_controller_baseline_version_arn=os.environ.get(
            "GLM52_SOURCE_CONTROLLER_BASELINE_VERSION_ARN", ""
        ),
        source_control_plane_readiness_version_arn=os.environ.get(
            "GLM52_SOURCE_CONTROL_PLANE_READINESS_VERSION_ARN", ""
        ),
        source_submission_acquisition_version_arn=os.environ.get(
            "GLM52_SOURCE_SUBMISSION_ACQUISITION_VERSION_ARN", ""
        ),
        fence_executor_version_arn=os.environ.get(
            "GLM52_FENCE_EXECUTOR_VERSION_ARN", ""
        ),
        fence_successor_version_arn=os.environ.get(
            "GLM52_FENCE_SUCCESSOR_VERSION_ARN", ""
        ),
        claim_writer_version_arn=os.environ.get(
            "GLM52_CLAIM_WRITER_VERSION_ARN", ""
        ),
        decision_writer_version_arn=os.environ.get(
            "GLM52_DECISION_WRITER_VERSION_ARN", ""
        ),
        terminal_v1_writer_version_arn=os.environ.get(
            "GLM52_TERMINAL_V1_WRITER_VERSION_ARN", ""
        ),
        closure_handoff_version_arn=os.environ.get(
            "GLM52_CLOSURE_HANDOFF_VERSION_ARN", ""
        ),
        deployment_identity_sha256=config.deployment_identity_sha256,
        decision_function_version_arn=os.environ.get(
            "GLM52_DECISION_VERSION_ARN", ""
        ),
    )


def _production_services(
    config: CollectorConfig,
    event: object,
    context: object,
) -> CollectorServices:
    """Build raw clients with no hidden SDK retry attempts."""

    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda runtime dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    raw_config = Config(
        connect_timeout=5,
        read_timeout=10,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session = boto3.session.Session(region_name=config.region)
    raw_clients = {
        name: session.client(name, config=raw_config)
        for name in (
            "cloudformation",
            "lambda",
            "iam",
            "events",
            "scheduler",
            "sqs",
            "sns",
            "ec2",
            "stepfunctions",
            "s3",
            "dynamodb",
            "ssm",
            "cloudwatch",
            "logs",
            "sts",
        )
    }
    s3 = raw_clients["s3"]
    lambda_client = raw_clients["lambda"]
    record_type = event.get("record_type") if type(event) is dict else None
    production_phases = None
    if record_type == "glm52_task11_collect_rehearsal_v1":
        if type(event.get("task11_boundary")) is not dict:
            raise RuntimeError(
                "deployed Task 11 boundary coordinate is absent"
            )
        task11_config = _task11_configuration(config)
        boundary = load_task11_boundary(
            s3=_ReadOnlyClient(s3),
            coordinate=Task11BoundaryCoordinate(
                **event["task11_boundary"]
            ),
        )
        h1d_names = {
            "cloudformation": "cloudformation",
            "lambda": "lambda",
            "iam": "iam",
            "eventbridge": "events",
            "scheduler": "scheduler",
            "sqs": "sqs",
            "sns": "sns",
            "ec2": "ec2",
            "stepfunctions": "stepfunctions",
            "s3": "s3",
            "dynamodb": "dynamodb",
            "ssm": "ssm",
            "cloudwatch": "cloudwatch",
            "logs": "logs",
        }
        h1d_clients = {
            family: _ReadOnlyClient(raw_clients[service])
            for family, service in h1d_names.items()
        }
        caller_identity = raw_clients["sts"].get_caller_identity()
        task11_clients = Task11AwsClients(
            s3=h1d_clients["s3"],
            dynamodb=None,
            cloudformation=h1d_clients["cloudformation"],
            lambda_client=h1d_clients["lambda"],
            sts=_ReadOnlyClient(raw_clients["sts"]),
            h1d_clients=h1d_clients,
            caller_identity=caller_identity,
        )
        adapters = Task11AcceptedAdapters(
            ledger=None,
            fresh_h1f=None,
            clients=task11_clients,
        )
        production = Task11ProductionServices(
            config=task11_config,
            adapters=adapters,
            boundary=boundary,
            event=event,
            context=context,
            session_binding_validator=_RehearsalSessionValidator(
                iam=h1d_clients["iam"]
            ),
            read_only_rehearsal=True,
        )
        production_phases = production.read_only_phases
    elif record_type != "glm52_task11_finalize_rehearsal_gate_v1":
        raise RuntimeError("rehearsal collector record type drifted")
    return CollectorServices(
        s3=s3,
        probe=_VersionPinnedProbe(lambda_client, config.probe_version_arn),
        production_phases=production_phases,
        faults=_FaultExercises(time.monotonic),
        monotonic=time.monotonic,
        sleeper=time.sleep,
        utcnow=lambda: datetime.now(timezone.utc),
    )


def main(
    event: object,
    context: object,
    *,
    config: Optional[CollectorConfig] = None,
    services: Optional[CollectorServices] = None,
) -> dict[str, object]:
    """Dispatch one exact COLLECT or FINALIZE request."""

    global _COLD_START
    exact_config = _configuration() if config is None else config
    cold_start = _COLD_START
    _COLD_START = False
    runtime = _runtime_identity(
        config=exact_config,
        context=context,
        cold_start=cold_start,
    )
    exact_services = (
        _production_services(
            exact_config,
            event,
            context,
        )
        if services is None
        else services
    )
    record_type = event.get("record_type") if type(event) is dict else None
    if record_type == "glm52_task11_collect_rehearsal_v1":
        return collect_rehearsal(
            event,
            config=exact_config,
            runtime=runtime,
            services=exact_services,
        )
    if record_type == "glm52_task11_finalize_rehearsal_gate_v1":
        return finalize_rehearsal_gate(
            event,
            config=exact_config,
            runtime=runtime,
            services=exact_services,
        )
    raise ValueError("collector handler record type drifted")


__all__ = ["main"]
