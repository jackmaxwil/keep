from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.dynamodb import encode_item

SHA = "a" * 64
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
CALLER_ARN = (
    "arn:aws:states:us-west-2:246813579024:stateMachine:"
    "keep-glm52-h1g-retained:9"
)
SNAPSHOT_CALLER_ARN = (
    "arn:aws:states:us-west-2:246813579024:stateMachine:"
    "keep-glm52-h1g-snapshot-cleanup:9"
)


def _caller_arn(kind: str) -> str:
    return (
        SNAPSHOT_CALLER_ARN
        if kind == "RETAINED_SNAPSHOT_CLEANUP"
        else CALLER_ARN
    )
LAMBDA_ROOT = (
    Path(__file__).resolve().parents[1] / "aws" / "glm52-gpu" / "lambda"
)
WRAPPERS = (
    (
        "retained_execution_observer_handler.py",
        "RETAINED_EXECUTION_OBSERVER",
        "RETAINED",
        "keep-glm52-h1g-execution-observer",
        (
            "authority",
            "campaign_bucket",
            "spend_runtime_prefix",
            "campaign_descriptor_key",
            "campaign_descriptor_version_id",
            "campaign_descriptor_file_sha256",
            "gpu_spend_approval_key",
            "gpu_spend_approval_version_id",
            "gpu_spend_approval_file_sha256",
            "kms_key_id",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
        ),
    ),
    (
        "retained_terminal_v2_handler.py",
        "RETAINED_TERMINAL_V2",
        "RETAINED",
        "keep-glm52-h1g-terminal-v2-writer",
        (
            "authority",
            "campaign_bucket",
            "spend_runtime_prefix",
            "campaign_descriptor_key",
            "campaign_descriptor_version_id",
            "campaign_descriptor_file_sha256",
            "gpu_spend_approval_key",
            "gpu_spend_approval_version_id",
            "gpu_spend_approval_file_sha256",
            "kms_key_id",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
            "task10_worker_descriptor_coordinate",
            "terminal_evidence_prefix",
        ),
    ),
    (
        "retained_finalizer_handler.py",
        "RETAINED_FINALIZER",
        "RETAINED",
        "keep-glm52-h1g-finalizer",
        ("authority", "campaign_bucket", "ledger_table_name"),
    ),
    (
        "retained_h1g_drained_handler.py",
        "RETAINED_H1G_DRAINED",
        "RETAINED",
        "keep-glm52-h1g-h1g-drained-writer",
        ("authority", "campaign_bucket", "ledger_table_name"),
    ),
    (
        "retained_worker_drain_handler.py",
        "RETAINED_WORKER_DRAIN",
        "RETAINED",
        "keep-glm52-h1g-worker-drain-signal",
        (
            "authority",
            "campaign_bucket",
            "spend_runtime_prefix",
            "campaign_descriptor_key",
            "campaign_descriptor_version_id",
            "campaign_descriptor_file_sha256",
            "gpu_spend_approval_key",
            "gpu_spend_approval_version_id",
            "gpu_spend_approval_file_sha256",
            "kms_key_id",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
            "task10_worker_descriptor_coordinate",
            "worker_script_hashes",
            "worker_drain_document_name",
            "worker_drain_document_version",
        ),
    ),
    (
        "retained_operator_disposition_handler.py",
        "RETAINED_OPERATOR_DISPOSITION",
        "RETAINED",
        "keep-glm52-h1g-operator-disposition-writer",
        (
            "authority",
            "campaign_bucket",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "retained_cancellation_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
        ),
    ),
    (
        "retained_orphan_audit_handler.py",
        "RETAINED_ORPHAN_AUDIT",
        "RETAINED",
        "keep-glm52-h1g-orphan-audit",
        ("authority", "kms_key_id", "ledger_table_name"),
    ),
    (
        "retained_snapshot_cleanup_handler.py",
        "RETAINED_SNAPSHOT_CLEANUP",
        "SNAPSHOT_CLEANUP",
        "keep-glm52-h1g-snapshot-cleanup",
        (
            "authority",
            "kms_key_id",
            "ledger_table_name",
            "snapshot_cleanup_state_machine_arn",
            "snapshot_cleanup_state_machine_version",
            "snapshot_cleanup_schedule_invoke_role_arn",
            "snapshot_cleanup_schedule_group_name",
            "snapshot_cleanup_schedule_name",
        ),
    ),
)

DEFAULT_OPERATIONS = {
    "RETAINED_EXECUTION_OBSERVER": "RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
    "RETAINED_TERMINAL_V2": "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
    "RETAINED_FINALIZER": "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED",
    "RETAINED_H1G_DRAINED": "RETAINED_INVOKE_H1G_DRAINED_WRITER",
    "RETAINED_WORKER_DRAIN": "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
    "RETAINED_OPERATOR_DISPOSITION": (
        "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION"
    ),
    "RETAINED_ORPHAN_AUDIT": "RETAINED_AUDIT_SUPPORT_ORPHANS",
    "RETAINED_SNAPSHOT_CLEANUP": (
        "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
    ),
}


def _load_wrapper(filename: str) -> object:
    path = LAMBDA_ROOT / filename
    spec = importlib.util.spec_from_file_location(
        "_task12_wrapper_" + filename.removesuffix(".py"), path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_drain_wrapper_has_no_generation_publication_capability() -> None:
    source = (
        LAMBDA_ROOT / "retained_worker_drain_handler.py"
    ).read_text()
    assert "campaign_drained_publication" not in source
    assert "publish_campaign_drained" not in source
    assert "make_retained_worker_drain_coordinator" in source


def _role_coordinates(names: tuple[str, ...]) -> dict[str, object]:
    from glm52_enforcement.task11_boundary import (
        build_task11_input_coordinate,
    )

    available: dict[str, object] = {
        "authority": {
            "bucket": "keep-glm52-retained",
            "key": "task12/activation-1/authority.json",
            "version_id": "authority-version-1",
            "file_sha256": SHA,
        },
        "campaign_bucket": "keep-glm52-campaign",
        "spend_runtime_prefix": (
            "campaigns/glm52-sky-20260724/runtime"
        ),
        "campaign_descriptor_key": (
            "campaigns/glm52-sky-20260724/submissions/production/"
            "descriptor.json"
        ),
        "campaign_descriptor_version_id": "descriptor-version-1",
        "campaign_descriptor_file_sha256": "d" * 64,
        "gpu_spend_approval_key": (
            "campaigns/glm52-sky-20260724/authorities/"
            "GPU_SPEND_APPROVAL.json"
        ),
        "gpu_spend_approval_version_id": "approval-version-1",
        "gpu_spend_approval_file_sha256": "e" * 64,
        "ledger_table_name": "keep-glm52-ledger",
        "numeric_binding_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-numeric-binding:17"
        ),
        "retained_cancellation_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-retained-cancellation:19"
        ),
        "task9_deployed_identity_coordinate": asdict(
            build_task11_input_coordinate(
                input_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
                bucket="keep-glm52-campaign",
                key=(
                    "campaigns/glm52-sky-20260724/authorities/task9/"
                    "activation-1/TASK9_DEPLOYED_IDENTITY.json"
                ),
                version_id="task9-version-1",
                file_sha256="b" * 64,
                body_sha256="c" * 64,
            )
        ),
        "task9_deployed_identity_sha256": "c" * 64,
        "task10_worker_descriptor_coordinate": {
            "bucket": "keep-glm52-campaign",
            "key": "task13/production/task10-worker-descriptor.json",
            "version_id": "task10-worker-version-1",
            "file_sha256": "d" * 64,
            "body_sha256": "e" * 64,
        },
        "worker_script_hashes": {
            "glm52_checkpoint_commit.py": "1" * 64,
            "glm52_drain_and_stop.py": "2" * 64,
            "glm52_deadline_guard.py": "3" * 64,
        },
        "terminal_evidence_prefix": (
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/terminal-evidence/"
        ),
        "state_machine_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:activation-1"
        ),
        "worker_drain_document_name": "KeepGlm52GracefulStopV1",
        "worker_drain_document_version": "7",
        "kms_key_id": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "11111111-2222-3333-4444-555555555555"
        ),
        "snapshot_cleanup_state_machine_arn": (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-snapshot-cleanup"
        ),
        "snapshot_cleanup_state_machine_version": "11",
        "snapshot_cleanup_schedule_invoke_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-snapshot-cleanup-schedule-invoke"
        ),
        "snapshot_cleanup_schedule_group_name": "default",
        "snapshot_cleanup_schedule_name": (
            "keep-glm52-h1g-snapshot-cleanup-activation-1"
        ),
    }
    return {name: available[name] for name in names}


def _deployment(
    *,
    kind: str,
    mode: str,
    function_name: str,
    coordinate_names: tuple[str, ...],
) -> tuple[str, str]:
    function_arn = (
        f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{function_name}:42"
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_deployment_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "handler_kind": kind,
        "mode": mode,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "function_name": function_name,
        "function_version": "42",
        "invoked_function_version_arn": function_arn,
        "caller_state_machine_version_arn": _caller_arn(kind),
        "role_coordinates": _role_coordinates(coordinate_names),
    }
    body["canonical_body_sha256"] = canonical_sha256(body)
    return canonical_json_bytes(body).decode("ascii"), function_arn


def _event(
    *,
    kind: str,
    mode: str,
    function_arn: str,
    deployment_identity_sha256: str,
) -> dict[str, object]:
    del mode, function_arn, deployment_identity_sha256
    caller = _caller_arn(kind)
    caller_name = caller.split(":stateMachine:", 1)[1].rsplit(":", 1)[0]
    return {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "dispatch_identity_sha256": SHA,
        "caller_state_machine_arn": caller.rsplit(":", 1)[0],
        "state_machine_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            + caller_name
            + ":activation-1"
        ),
        "operation_kind": DEFAULT_OPERATIONS[kind],
        "operation_input": _operation_input(DEFAULT_OPERATIONS[kind]),
    }


def _operation_input(operation_kind: str) -> dict[str, object]:
    from glm52_enforcement import task12_lambda_adapters as adapters

    predecessor = adapters._OPERATION_PREDECESSOR[operation_kind]
    if predecessor is None:
        return {}
    prior_handler = next(
        kind
        for kind, operations in adapters._OPERATIONS.items()
        if predecessor in operations
    )
    prior_body = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_result_v1",
        "handler_kind": prior_handler,
        "operation_kind": predecessor,
        "operation_input_identity_sha256": canonical_sha256({}),
        "outcome": "SUCCEEDED",
        "result": {},
    }
    return {
        "task12_last_result": {
            **prior_body,
            "canonical_body_sha256": canonical_sha256(prior_body),
        }
    }


class _CountingFactory:
    def __init__(self) -> None:
        self.factory_calls = 0
        self.coordinator_calls = 0

    def __call__(self, deployment: object) -> object:
        self.factory_calls += 1

        def coordinate(invocation: object) -> dict[str, object]:
            self.coordinator_calls += 1
            return {
                "record_type": "glm52_task12_test_result_v1",
                "handler_kind": getattr(deployment, "handler_kind"),
                "activation_id": getattr(invocation, "activation_id"),
                "canonical_body_sha256": SHA,
            }

        return coordinate


def _trusted_execution(
    deployment: object,
    invocation: object,
) -> dict[str, object]:
    return {
        "execution_arn": getattr(invocation, "state_machine_execution_arn"),
        "state_machine_arn": getattr(invocation, "caller_state_machine_arn"),
        "state_machine_version_arn": getattr(
            deployment, "caller_state_machine_version_arn"
        ),
        "status": "RUNNING",
        "request_id": "trusted-test-execution",
    }


@pytest.mark.parametrize(
    ("filename", "kind", "mode", "function_name", "coordinate_names"),
    WRAPPERS,
)
def test_each_wrapper_authenticates_exact_caller_version_before_factory_or_effects(
    filename: str,
    kind: str,
    mode: str,
    function_name: str,
    coordinate_names: tuple[str, ...],
) -> None:
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=json.loads(raw_config)[
            "canonical_body_sha256"
        ],
    )
    aws_calls: list[tuple[str, str]] = []

    class _StepFunctions:
        def describe_execution(self, **request: object) -> object:
            aws_calls.append(("stepfunctions", "describe_execution"))
            return {
                "executionArn": request["executionArn"],
                "stateMachineArn": event["caller_state_machine_arn"],
                    "stateMachineVersionArn": (
                        event["caller_state_machine_arn"] + ":42"
                    ),
                "status": "RUNNING",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "forged-version-read",
                    "RetryAttempts": 0,
                },
            }

    class _Session:
        def client(
            self,
            service: str,
            *,
            region_name: str,
            config: object,
        ) -> object:
            assert service == "stepfunctions"
            assert region_name == REGION
            assert config.retries == {
                "mode": "standard",
                "total_max_attempts": 1,
            }
            return _StepFunctions()

    factory = _CountingFactory()
    result = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _deployment_session=_Session(),
        _coordinator_factory=factory,
    )

    assert result["outcome"] == "REJECTED"
    assert factory.factory_calls == 0
    assert factory.coordinator_calls == 0
    assert aws_calls == [("stepfunctions", "describe_execution")]


@pytest.mark.parametrize(
    ("filename", "kind", "mode", "function_name", "coordinate_names"),
    WRAPPERS,
)
def test_lambda_wrapper_validates_then_invokes_one_named_coordinator(
    filename: str,
    kind: str,
    mode: str,
    function_name: str,
    coordinate_names: tuple[str, ...],
) -> None:
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    deployment_identity = json.loads(raw_config)["canonical_body_sha256"]
    factory = _CountingFactory()

    result = module.main(
        _event(
            kind=kind,
            mode=mode,
            function_arn=function_arn,
            deployment_identity_sha256=deployment_identity,
        ),
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _coordinator_factory=factory,
        _execution_authenticator=_trusted_execution,
    )

    assert module.__all__ == ["main"]
    assert factory.factory_calls == 1
    assert factory.coordinator_calls == 1
    assert result["record_type"] == "glm52_task12_lambda_result_v1"
    assert result["outcome"] == "SUCCEEDED"
    assert result["handler_kind"] == kind
    assert result["result"]["activation_id"] == "activation-1"
    assert result["canonical_body_sha256"] == canonical_sha256(
        {
            name: value
            for name, value in result.items()
            if name != "canonical_body_sha256"
        }
    )


def test_eventbridge_finalization_detail_reaches_packaged_first_retained_lambda(
) -> None:
    """The real rule and ASL Parameters form the closed wrapper envelope."""

    from glm52_enforcement.task12_support_plane import (
        Task12SupportPlaneInputs,
        render_task12_support_plane_fragment,
    )

    inputs = Task12SupportPlaneInputs(
        activation_id="activation-1",
        lambda_code_sha256=SHA,
        lambda_code_bucket="keep-glm52-runtime-artifacts",
        lambda_code_key="releases/task12-runtime.zip",
        lambda_code_version="runtime-version-1",
        ledger_table_arn=(
            "arn:aws:dynamodb:us-west-2:246813579024:table/"
            "keep-glm52-h1g-ledger-v1"
        ),
        retained_kms_key_arn=(
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-4234-8234-1234567890ab"
        ),
        model_bucket_arn="arn:aws:s3:::keep-glm52-models",
        worker_drain_document_arn=(
            "arn:aws:ssm:us-west-2:246813579024:document/"
            "KeepGlm52GracefulStopV1"
        ),
    )
    resources = render_task12_support_plane_fragment(
        inputs=inputs
    )["Resources"]
    rule = resources["RetainedLifecycleEventRule"]["Properties"]
    target = rule["Targets"][0]
    assert target["InputPath"] == "$.detail"
    detail_body = {
        "schema_version": 1,
        "record_type": (
            "glm52_task12_retained_lifecycle_request_v1"
        ),
        "run_id": "glm52-sky-20260724",
        "activation_id": "activation-1",
        "generation": 1,
        "generation_text": "00000001",
        "reason": "SUPPORT_FINALIZATION_REQUESTED",
        "support_state_identity_sha256": SHA,
        "support_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-support:support-1"
        ),
        "support_state_machine_version_arn": (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-support:7"
        ),
        "terminal_v2_identity_sha256": SHA,
    }
    detail = {
        **detail_body,
        "canonical_identity_sha256": canonical_sha256(detail_body),
    }
    eventbridge_event = {
        "source": "keep.glm52.task12",
        "detail-type": "RETAINED_LIFECYCLE_REQUESTED",
        "detail": detail,
    }
    state_input = eventbridge_event[
        target["InputPath"].removeprefix("$.")
    ]
    definition = json.loads(
        resources["RetainedLifecycleStateMachine"]["Properties"][
            "DefinitionString"
        ]
    )
    first_name = definition["StartAt"]
    assert first_name == "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
    parameters = definition["States"][first_name]["Parameters"]
    caller_version = (
        "arn:aws:states:us-west-2:246813579024:stateMachine:"
        "keep-glm52-h1g-retainedlifecycle:9"
    )
    execution_arn = (
        "arn:aws:states:us-west-2:246813579024:execution:"
        "keep-glm52-h1g-retainedlifecycle:finalization-1"
    )
    wrapper_event = {
        "activation_id": parameters["activation_id"],
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "dispatch_identity_sha256": SHA,
        "caller_state_machine_arn": caller_version.rsplit(":", 1)[0],
        "state_machine_execution_arn": execution_arn,
        "operation_kind": parameters["operation_kind"],
        "operation_input": state_input,
    }
    raw_config, function_arn = _deployment(
        kind="RETAINED_EXECUTION_OBSERVER",
        mode="RETAINED",
        function_name="keep-glm52-h1g-execution-observer",
        coordinate_names=WRAPPERS[0][4],
    )
    config = json.loads(raw_config)
    config["caller_state_machine_version_arn"] = caller_version
    config["canonical_body_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in config.items()
            if key != "canonical_body_sha256"
        }
    )
    module = _load_wrapper(
        "retained_execution_observer_handler.py"
    )
    factory = _CountingFactory()

    result = module.main(
        wrapper_event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={
            "GLM52_TASK12_DEPLOYMENT_CONFIG": canonical_json_bytes(
                config
            ).decode("ascii")
        },
        _coordinator_factory=factory,
        _execution_authenticator=_trusted_execution,
    )

    assert result["outcome"] == "SUCCEEDED"
    assert result["operation_kind"] == first_name
    assert result["operation_input_identity_sha256"] == canonical_sha256(
        detail
    )
    assert factory.factory_calls == factory.coordinator_calls == 1


@pytest.mark.parametrize(
    "mutation",
    ("bad_self_hash", "durable_bytes_mismatch"),
)
def test_packaged_eventbridge_finalization_rejects_foreign_detail_before_runtime(
    mutation: str,
) -> None:
    """The packaged wrapper must reach the live durable-trigger boundary."""

    from glm52_enforcement.task12_live_runtime import (
        materialize_live_request,
    )
    from test_glm52_task12_live_runtime import (
        _Ports,
        _finalization_detail,
        _live_sources,
        _sources,
    )

    kind = "RETAINED_EXECUTION_OBSERVER"
    mode = "RETAINED"
    function_name = "keep-glm52-h1g-execution-observer"
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=WRAPPERS[0][4],
    )
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=json.loads(raw_config)[
            "canonical_body_sha256"
        ],
    )
    operation = "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
    detail = _finalization_detail()
    ports = _Ports()
    sources = _sources()
    if mutation == "bad_self_hash":
        detail["canonical_identity_sha256"] = "b" * 64
    else:
        durable = _finalization_detail()
        durable["support_execution_arn"] = (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-support:foreign"
        )
        body = dict(durable)
        body.pop("canonical_identity_sha256")
        durable["canonical_identity_sha256"] = canonical_sha256(body)
        ports.clients["dynamodb"].finalization_record = (
            canonical_json_bytes(durable).decode("ascii")
        )
    event["operation_kind"] = operation
    event["operation_input"] = detail

    class _LiveFactory:
        def __call__(self, _deployment_value: object) -> object:
            def coordinate(invocation: object) -> dict[str, object]:
                return materialize_live_request(
                    operation_kind=operation,
                    invocation=invocation,
                    live_sources=_live_sources(sources, operation),
                    ports=ports,
                )

            return coordinate

    module = _load_wrapper(
        "retained_execution_observer_handler.py"
    )
    result = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _coordinator_factory=_LiveFactory(),
        _execution_authenticator=_trusted_execution,
    )

    assert result["outcome"] == "FAILED"
    assert result["error_code"] == "Task12LiveRuntimeError"
    assert ports.clients["lambda"].calls == []
    assert ports.clients["stepfunctions"].calls == []
    assert ports.clients["s3"].list_calls == []
    assert ports.clients["ec2"].calls == []
    assert ports.clients["dynamodb"].put_calls == []


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_activation",
        "wrong_context_version",
        "wrong_caller",
        "wrong_execution",
        "wrong_config_kind",
        "wrong_config_hash",
        "extra_event",
    ),
)
def test_lambda_wrapper_mutants_return_closed_failure_before_factory(
    mutation: str,
) -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[0]
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    config = json.loads(raw_config)
    deployment_identity = config["canonical_body_sha256"]
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=deployment_identity,
    )
    context_arn = function_arn
    if mutation == "wrong_activation":
        event["activation_id"] = "activation-foreign"
    elif mutation == "wrong_context_version":
        context_arn = function_arn[:-2] + "43"
    elif mutation == "wrong_caller":
        event["caller_state_machine_arn"] = (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-foreign"
        )
    elif mutation == "wrong_execution":
        event["state_machine_execution_arn"] = (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-foreign:activation-1"
        )
    elif mutation == "wrong_config_kind":
        config["handler_kind"] = "RETAINED_FOREIGN"
        config["canonical_body_sha256"] = canonical_sha256(
            {
                name: value
                for name, value in config.items()
                if name != "canonical_body_sha256"
            }
        )
        raw_config = canonical_json_bytes(config).decode("ascii")
    elif mutation == "wrong_config_hash":
        config["canonical_body_sha256"] = "b" * 64
        raw_config = canonical_json_bytes(config).decode("ascii")
    else:
        event["unexpected"] = True
    factory = _CountingFactory()

    result = module.main(
        event,
        SimpleNamespace(invoked_function_arn=context_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _coordinator_factory=factory,
    )

    assert factory.factory_calls == 0
    assert factory.coordinator_calls == 0
    assert set(result) == {
        "schema_version",
        "record_type",
        "handler_kind",
        "outcome",
        "error_code",
        "canonical_body_sha256",
    }
    assert result["outcome"] == "REJECTED"
    assert result["error_code"] != ""
    assert result["canonical_body_sha256"] == canonical_sha256(
        {
            name: value
            for name, value in result.items()
            if name != "canonical_body_sha256"
        }
    )


def test_lambda_wrapper_rejects_noncanonical_config_path_and_symlink(
    tmp_path: Path,
) -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[0]
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    deployment_identity = json.loads(raw_config)["canonical_body_sha256"]
    config_path = tmp_path / "deployment.json"
    config_path.write_text(raw_config + "\n", encoding="ascii")
    symlink = tmp_path / "deployment-link.json"
    symlink.symlink_to(config_path)
    factory = _CountingFactory()
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=deployment_identity,
    )

    noncanonical = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG_PATH": str(config_path)},
        _coordinator_factory=factory,
    )
    symlinked = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG_PATH": str(symlink)},
        _coordinator_factory=factory,
    )

    assert noncanonical["outcome"] == "REJECTED"
    assert symlinked["outcome"] == "REJECTED"
    assert factory.factory_calls == 0
    assert factory.coordinator_calls == 0


@pytest.mark.parametrize(
    "mutation",
    ("extra_field", "noncanonical_inline", "both_sources"),
)
def test_lambda_wrapper_rejects_nonclosed_deployment_config_before_factory(
    mutation: str,
    tmp_path: Path,
) -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[0]
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    config = json.loads(raw_config)
    if mutation == "extra_field":
        config["unexpected"] = True
        config["canonical_body_sha256"] = canonical_sha256(
            {
                name: value
                for name, value in config.items()
                if name != "canonical_body_sha256"
            }
        )
        environ = {
            "GLM52_TASK12_DEPLOYMENT_CONFIG": canonical_json_bytes(
                config
            ).decode("ascii")
        }
    elif mutation == "noncanonical_inline":
        environ = {"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config + "\n"}
    else:
        path = tmp_path / "deployment.json"
        path.write_text(raw_config, encoding="ascii")
        environ = {
            "GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config,
            "GLM52_TASK12_DEPLOYMENT_CONFIG_PATH": str(path),
        }
    factory = _CountingFactory()

    result = module.main(
        _event(
            kind=kind,
            mode=mode,
            function_arn=function_arn,
            deployment_identity_sha256=json.loads(raw_config)[
                "canonical_body_sha256"
            ],
        ),
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ=environ,
        _coordinator_factory=factory,
    )

    assert result["outcome"] == "REJECTED"
    assert factory.factory_calls == 0
    assert factory.coordinator_calls == 0


@pytest.mark.parametrize(
    ("kind", "mode", "function_name", "coordinate_names", "factory_name"),
    tuple(
        (*wrapper[1:], "make_" + wrapper[0].removesuffix("_handler.py") + "_coordinator")
        for wrapper in WRAPPERS
    ),
)
def test_named_production_factory_uses_lazy_zero_retry_versioned_authority_read(
    kind: str,
    mode: str,
    function_name: str,
    coordinate_names: tuple[str, ...],
    factory_name: str,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    deployment = adapters._load_deployment(
        environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        expected_kind=kind,
    )
    invocation = adapters._parse_invocation(
        event=_event(
            kind=kind,
            mode=mode,
            function_arn=function_arn,
            deployment_identity_sha256=json.loads(raw_config)[
                "canonical_body_sha256"
            ],
        ),
        context=SimpleNamespace(invoked_function_arn=function_arn),
        deployment=deployment,
    )

    class _UnavailableS3:
        def get_object(self, **_request: object) -> object:
            raise RuntimeError("expected fake transport stop")

    class _Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def client(
            self,
            service: str,
            *,
            region_name: str,
            config: object,
        ) -> object:
            assert region_name == REGION
            self.calls.append((service, config))
            return _UnavailableS3()

    session = _Session()
    coordinator = getattr(adapters, factory_name)(
        deployment,
        _session=session,
    )

    with pytest.raises(
        adapters.Task12LambdaAdapterError,
        match="versioned coordinator authority read failed",
    ):
        coordinator(invocation)

    assert len(session.calls) == 1
    service, client_config = session.calls[0]
    assert service == "s3"
    assert client_config.retries == {
        "mode": "standard",
        "total_max_attempts": 1,
    }


def test_execution_observer_factory_reads_pinned_authority_and_typed_state() -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters
    from glm52_enforcement.dynamodb import encode_item
    from glm52_enforcement.records import ledger_pk
    from glm52_enforcement.task12_runtime import (
        CompleteRuntimeRead,
        RuntimeScan,
        SpendRead,
    )

    filename, kind, mode, function_name, coordinate_names = WRAPPERS[0]
    del filename
    observed_at = "2026-07-29T12:00:00Z"
    reads = {
        domain: CompleteRuntimeRead(
            domain=domain,
            members=(),
            observed_at=observed_at,
            pagination_complete=True,
            evidence_identity_sha256=SHA,
        )
        for domain in ("REQUEST", "JOB", "CONTROLLER", "WORKER", "ALLOCATION")
    }
    scan = RuntimeScan(
        requests=reads["REQUEST"],
        jobs=reads["JOB"],
        controller=reads["CONTROLLER"],
        workers=reads["WORKER"],
        allocations=reads["ALLOCATION"],
        spend=SpendRead(
            state="CLOSED",
            ledger_head_identity_sha256=SHA,
            remaining_gpu_seconds=0,
            remaining_gpu_cost_usd="0.00",
            observed_at=observed_at,
            evidence_identity_sha256=SHA,
        ),
    )
    state_sort_key = (
        "ACTIVATION#activation-1#TASK12_LAMBDA_INPUT#"
        + kind
        + "#"
        + DEFAULT_OPERATIONS[kind]
        + "#00000001"
    )
    state_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": (
            "glm52_task12_retained_execution_observer_input_v1"
        ),
        "handler_kind": kind,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "operation_kind": DEFAULT_OPERATIONS[kind],
        "correlation_identity_sha256": SHA,
        "runtime_scan": json.loads(json.dumps(asdict(scan))),
    }
    state = {
        **state_body,
        "canonical_body_sha256": canonical_sha256(state_body),
    }
    authority_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_invocation_authority_v1",
        "handler_kind": kind,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "operations": {
            operation: {
                "state_sort_key": (
                    state_sort_key
                    if operation == DEFAULT_OPERATIONS[kind]
                    else (
                        "ACTIVATION#activation-1#TASK12_LAMBDA_INPUT#"
                        + kind
                        + "#"
                        + operation
                        + "#00000001"
                    )
                ),
                "state_body_sha256": (
                    state["canonical_body_sha256"]
                    if operation == DEFAULT_OPERATIONS[kind]
                    else SHA
                ),
            }
            for operation in adapters._OPERATIONS[kind]
        },
    }
    authority = {
        **authority_body,
        "canonical_body_sha256": canonical_sha256(authority_body),
    }
    authority_raw = canonical_json_bytes(authority)

    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    config = json.loads(raw_config)
    config["role_coordinates"]["authority"]["file_sha256"] = (
        hashlib.sha256(authority_raw).hexdigest()
    )
    config["canonical_body_sha256"] = canonical_sha256(
        {
            name: value
            for name, value in config.items()
            if name != "canonical_body_sha256"
        }
    )
    raw_config = canonical_json_bytes(config).decode("ascii")
    deployment = adapters._load_deployment(
        environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        expected_kind=kind,
    )
    invocation = adapters._parse_invocation(
        event=_event(
            kind=kind,
            mode=mode,
            function_arn=function_arn,
            deployment_identity_sha256=config["canonical_body_sha256"],
        ),
        context=SimpleNamespace(invoked_function_arn=function_arn),
        deployment=deployment,
    )

    class _Body:
        def read(self) -> bytes:
            return authority_raw

    class _S3:
        def get_object(self, **request: object) -> object:
            assert request["VersionId"] == "authority-version-1"
            assert request["ExpectedBucketOwner"] == ACCOUNT_ID
            return {
                "Body": _Body(),
                "VersionId": "authority-version-1",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "s3-request-1",
                    "RetryAttempts": 0,
                },
            }

    class _DynamoDB:
        def get_item(self, **request: object) -> object:
            assert request["ConsistentRead"] is True
            return {
                "Item": encode_item(
                    {
                        "PK": ledger_pk("glm52-sky-20260724"),
                        "SK": state_sort_key,
                        **state,
                    }
                ),
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "ddb-request-1",
                    "RetryAttempts": 0,
                },
            }

    class _StepFunctions:
        def describe_execution(self, **request: object) -> object:
            assert request["executionArn"] == (
                "arn:aws:states:us-west-2:246813579024:execution:"
                "keep-glm52-h1g-retained:activation-1"
            )
            return {
                "executionArn": request["executionArn"],
                "stateMachineArn": CALLER_ARN.rsplit(":", 1)[0],
                "stateMachineVersionArn": CALLER_ARN,
                "status": "SUCCEEDED",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "states-request-1",
                    "RetryAttempts": 0,
                },
            }

    class _Session:
        def client(
            self,
            service: str,
            *,
            region_name: str,
            config: object,
        ) -> object:
            assert region_name == REGION
            assert config.retries["total_max_attempts"] == 1
            return {
                "s3": _S3(),
                "dynamodb": _DynamoDB(),
                "stepfunctions": _StepFunctions(),
            }[service]

    coordinator = adapters.make_retained_execution_observer_coordinator(
        deployment,
        _session=_Session(),
    )
    result = coordinator(invocation)

    assert result == scan


def test_lambda_production_loader_reads_context_keyed_deployment_authority() -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[0]
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    config = json.loads(raw_config)
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=config["canonical_body_sha256"],
    )
    partition_key = "RUN#glm52-sky-20260724"
    prefix = "TASK12_LAMBDA_DEPLOYMENT#"
    sort_key = (
        prefix
        + function_arn
        + "#CALLER#"
        + CALLER_ARN.rsplit(":", 1)[0]
        + "#ACTIVATION#activation-1#GENERATION#00000001"
    )

    class _DynamoDB:
        def get_item(self, **request: object) -> object:
            assert request == {
                "TableName": "keep-glm52-ledger",
                "Key": encode_item({"PK": partition_key, "SK": sort_key}),
                "ConsistentRead": True,
                "ReturnConsumedCapacity": "NONE",
            }
            return {
                "Item": encode_item(
                    {
                        "PK": partition_key,
                        "SK": sort_key,
                        **config,
                    }
                ),
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "deployment-authority-read",
                    "RetryAttempts": 0,
                },
            }

    class _Session:
        def client(
            self,
            service: str,
            *,
            region_name: str,
            config: object,
        ) -> object:
            assert service == "dynamodb"
            assert region_name == REGION
            assert config.retries == {
                "mode": "standard",
                "total_max_attempts": 1,
            }
            return _DynamoDB()

    factory = _CountingFactory()
    result = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={
            "GLM52_TASK12_DEPLOYMENT_TABLE_NAME": "keep-glm52-ledger",
            "GLM52_TASK12_DEPLOYMENT_PARTITION_KEY": partition_key,
            "GLM52_TASK12_DEPLOYMENT_SORT_KEY_PREFIX": prefix,
        },
        _deployment_session=_Session(),
        _coordinator_factory=factory,
        _execution_authenticator=_trusted_execution,
    )

    assert result["outcome"] == "SUCCEEDED"
    assert factory.factory_calls == 1
    assert factory.coordinator_calls == 1


def test_lambda_result_serializes_nested_decimal_as_canonical_text() -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[0]
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    deployment_identity = json.loads(raw_config)["canonical_body_sha256"]

    def factory(_deployment: object) -> object:
        def coordinate(_invocation: object) -> object:
            return {
                "liability": {
                    "gpu_reserve_cost_usd": Decimal("13.76"),
                }
            }

        return coordinate

    result = module.main(
        _event(
            kind=kind,
            mode=mode,
            function_arn=function_arn,
            deployment_identity_sha256=deployment_identity,
        ),
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _coordinator_factory=factory,
        _execution_authenticator=_trusted_execution,
    )

    assert result["result"]["liability"]["gpu_reserve_cost_usd"] == "13.76"
    assert json.loads(json.dumps(result)) == result


def test_execution_arn_is_runtime_event_authority_not_deployment_coordinate() -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[0]
    module = _load_wrapper(filename)
    runtime_coordinates = tuple(
        name
        for name in coordinate_names
        if name != "state_machine_execution_arn"
    )
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=runtime_coordinates,
    )
    deployment_identity = json.loads(raw_config)["canonical_body_sha256"]
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=deployment_identity,
    )
    event["state_machine_execution_arn"] = (
        "arn:aws:states:us-west-2:246813579024:execution:"
        "keep-glm52-h1g-retained:activation-1"
    )
    factory = _CountingFactory()

    result = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _coordinator_factory=factory,
        _execution_authenticator=_trusted_execution,
    )

    assert result["outcome"] == "SUCCEEDED"
    assert factory.factory_calls == 1


def test_real_operation_scoped_wrappers_reject_cross_operation_prebuilt_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters
    from glm52_enforcement.dynamodb import TransactionResolution, WriteOutcome

    fixture_spec = importlib.util.spec_from_file_location(
        "_task12_real_chain_fixtures",
        Path(__file__).with_name(
            "test_glm52_task12_lambda_adapter_factories.py"
        ),
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixtures = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixtures)
    wrappers_by_kind = {
        kind: (filename, mode, function_name, coordinate_names)
        for filename, kind, mode, function_name, coordinate_names in WRAPPERS
    }
    cases_by_kind = {
        case[0]: case
        for case in fixtures.CASES
    }

    class _FakeDynamoLedgerAdapter:
        def __init__(self, *, client: object, table_name: str) -> None:
            assert table_name == "keep-glm52-ledger"

        def commit_snapshot_cleanup_transition(
            self, **request: object
        ) -> TransactionResolution:
            return TransactionResolution(
                outcome=WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
                records=(),
                request_id="ddb-real-chain-transition",
                error_code=None,
                cancellation_reasons=(),
            )

    import glm52_enforcement.dynamodb as dynamodb

    monkeypatch.setattr(
        dynamodb,
        "DynamoLedgerAdapter",
        _FakeDynamoLedgerAdapter,
    )
    common_event = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "dispatch_identity_sha256": SHA,
        "caller_state_machine_arn": CALLER_ARN.rsplit(":", 1)[0],
        "state_machine_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:activation-1"
        ),
    }
    operation_to_kind = {
        operation: kind
        for kind, operations in adapters._OPERATIONS.items()
        for operation in operations
    }
    sequence = tuple(
        operation
        for operation in (
            adapters._RETAINED_OPERATION_SEQUENCE
            + adapters._SNAPSHOT_OPERATION_SEQUENCE
        )
        if operation != fixtures.DEFAULT_OPERATIONS[
            operation_to_kind[operation]
        ]
    )
    observed_operations: list[str] = []

    for operation_kind in sequence:
        kind = operation_to_kind[operation_kind]
        filename, mode, function_name, coordinate_names = wrappers_by_kind[
            kind
        ]
        _, _, _, _, _, record_type = cases_by_kind[kind]
        state_payload = (
            fixtures._snapshot_arm_input()
            if operation_kind
            == "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE"
            else fixtures._INPUT_BUILDERS[kind]()
        )
        harness = fixtures._AwsHarness(
            kind=kind,
            state_record_type=record_type,
            state_payload=state_payload,
            mode=mode,
            function_name=function_name,
            coordinate_names=coordinate_names,
            operation_kind=operation_kind,
        )
        monkeypatch.setattr(
            adapters,
            "_zero_retry_client",
            lambda service, session=None, current=harness: current,
        )
        module = _load_wrapper(filename)
        result = module.main(
            {
                **common_event,
                "operation_kind": operation_kind,
                "operation_input": {},
            },
            SimpleNamespace(
                invoked_function_arn=(
                    harness.deployment.invoked_function_version_arn
                )
            ),
            _environ={
                "GLM52_TASK12_DEPLOYMENT_CONFIG": harness.raw_config
            },
        )
        # The harness provides the handler's default direct input body while
        # claiming every other operation on that handler. A scoped runtime
        # must reject that synthetic prebuilt path; a different operation is
        # entered only through its typed canonical live sources.
        assert result["outcome"] == "REJECTED"
        observed_operations.append(operation_kind)

    assert tuple(observed_operations) == sequence


def test_unqualified_context_caller_selects_version_pinned_deployment() -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[0]
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=json.loads(raw_config)[
            "canonical_body_sha256"
        ],
    )
    result = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _coordinator_factory=_CountingFactory(),
        _execution_authenticator=_trusted_execution,
    )

    assert result["outcome"] == "SUCCEEDED"


def test_wrapper_rejects_foreign_prior_resultpath_before_coordinator() -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[1]
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    prior_body = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_result_v1",
        "handler_kind": "RETAINED_ORPHAN_AUDIT",
        "operation_kind": "RETAINED_AUDIT_SUPPORT_ORPHANS",
        "operation_input_identity_sha256": canonical_sha256({}),
        "outcome": "SUCCEEDED",
        "result": {},
    }
    prior = {
        **prior_body,
        "canonical_body_sha256": canonical_sha256(prior_body),
    }
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=json.loads(raw_config)[
            "canonical_body_sha256"
        ],
    )
    event["operation_kind"] = "RETAINED_ENTER_RECOVERY_COMPLETE"
    event["operation_input"] = {"task12_last_result": prior}
    factory = _CountingFactory()

    result = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _coordinator_factory=factory,
        _execution_authenticator=_trusted_execution,
    )

    assert result["outcome"] == "REJECTED"
    assert factory.factory_calls == 0
    assert factory.coordinator_calls == 0


def test_wrapper_rejects_missing_required_prior_resultpath() -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[1]
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=json.loads(raw_config)[
            "canonical_body_sha256"
        ],
    )
    event["operation_kind"] = "RETAINED_ENTER_RECOVERY_COMPLETE"
    event["operation_input"] = {}
    factory = _CountingFactory()

    result = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _coordinator_factory=factory,
        _execution_authenticator=_trusted_execution,
    )

    assert result["outcome"] == "REJECTED"
    assert factory.factory_calls == 0


def test_snapshot_reconcile_accepts_exact_delete_ambiguity_catch_path() -> None:
    filename, kind, mode, function_name, coordinate_names = WRAPPERS[7]
    module = _load_wrapper(filename)
    raw_config, function_arn = _deployment(
        kind=kind,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    event = _event(
        kind=kind,
        mode=mode,
        function_arn=function_arn,
        deployment_identity_sha256=json.loads(raw_config)[
            "canonical_body_sha256"
        ],
    )
    event["operation_kind"] = "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"
    event["operation_input"] = {
        **_operation_input(
            "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
        ),
        "delete_ambiguity": {
            "Error": "States.TaskFailed",
            "Cause": "delete transport became ambiguous",
        },
    }
    factory = _CountingFactory()

    result = module.main(
        event,
        SimpleNamespace(invoked_function_arn=function_arn),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
        _coordinator_factory=factory,
        _execution_authenticator=_trusted_execution,
    )

    assert result["outcome"] == "SUCCEEDED"
    assert factory.factory_calls == 1
