from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import base64
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.dynamodb import decode_item, encode_item
from glm52_enforcement.records import canonical_record_identity, ledger_pk, ledger_sk
from glm52_enforcement.spend_authority import (
    APPROVED_GPU_COST_USD,
    APPROVED_GPU_RUNTIME_SECONDS,
    APPROVED_HOURLY_COST_USD,
    build_gpu_spend_approval,
    canonical_decimal_json_bytes,
)
from glm52_enforcement.task12_live_runtime import (
    SUPPORTED_OPERATIONS,
    Task12LiveRuntimeError,
    materialize_live_request,
    persist_live_successors,
)
from glm52_enforcement.task12_runtime import RuntimeScan


RUN_ID = "glm52-sky-20260724"
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
BUCKET = "keep-glm52-campaign-246813579024-us-west-2"
PREFIX = f"campaigns/{RUN_ID}/runtime"
PK = ledger_pk(RUN_ID)
SHA_A = "a" * 64
SHA_B = "b" * 64
EXECUTION_ARN = (
    "arn:aws:states:us-west-2:246813579024:execution:"
    "keep-glm52-h1g-retainedlifecycle:activation-1"
)
STATE_MACHINE_VERSION_ARN = (
    "arn:aws:states:us-west-2:246813579024:stateMachine:"
    "keep-glm52-h1g-retainedlifecycle:9"
)
SUPPORT_EXECUTION_ARN = (
    "arn:aws:states:us-west-2:246813579024:execution:"
    "keep-glm52-h1g-support:support-1"
)
SUPPORT_STATE_MACHINE_VERSION_ARN = (
    "arn:aws:states:us-west-2:246813579024:stateMachine:"
    "keep-glm52-h1g-support:7"
)
NUMERIC_VERSION_ARN = (
    "arn:aws:lambda:us-west-2:246813579024:function:"
    "keep-glm52-h1g-numeric-binding:7"
)


def _meta(request_id: str, *, retries: int = 0) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": request_id,
        "RetryAttempts": retries,
    }


def _raw(value: object) -> bytes:
    return canonical_decimal_json_bytes(value) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(
    body: dict[str, object], field: str
) -> dict[str, object]:
    return {
        **body,
        field: hashlib.sha256(
            canonical_decimal_json_bytes(body)
        ).hexdigest(),
    }


def _spend_objects() -> tuple[
    dict[str, bytes], dict[str, str], dict[str, str]
]:
    approval = build_gpu_spend_approval(
        ingested_at="2026-07-15T00:00:00Z",
        slack_permalink=None,
    )
    approval_raw = _raw(approval)
    descriptor = _self_hash(
        {
            "schema_version": 2,
            "record_type": "glm52_sky_campaign_descriptor_v2",
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "campaign_identity_sha256": SHA_A,
            "approval_sha256": _sha(approval_raw),
            "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
            "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
            "max_hourly_cost_usd": APPROVED_HOURLY_COST_USD,
        },
        "descriptor_body_sha256",
    )
    descriptor_raw = _raw(descriptor)
    genesis = hashlib.sha256(
        canonical_decimal_json_bytes(
            {
                "record_type": "glm52_gpu_spend_ledger_genesis_v1",
                "run_id": RUN_ID,
                "approval_sha256": _sha(approval_raw),
                "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
                "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
                "hourly_cost_usd": APPROVED_HOURLY_COST_USD,
            }
        )
    ).hexdigest()
    latest = _self_hash(
        {
            "schema_version": 1,
            "record_type": "glm52_gpu_spend_ledger_latest_v1",
            "run_id": RUN_ID,
            "gpu_spend_authority_sha256": genesis,
            "record_count": 0,
            "record_keys": [],
            "latest_record_sha256": genesis,
            "ledger_sha256": hashlib.sha256(b"").hexdigest(),
        },
        "latest_body_sha256",
    )
    keys = {
        "descriptor": f"campaigns/{RUN_ID}/submissions/production/descriptor.json",
        "approval": f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json",
        "latest": f"{PREFIX}/GPU_SPEND_LEDGER_LATEST.json",
        "records": f"{PREFIX}/spend-ledger/records/",
    }
    objects = {
        keys["descriptor"]: descriptor_raw,
        keys["approval"]: approval_raw,
        keys["latest"]: _raw(latest),
    }
    versions = {
        keys["descriptor"]: "descriptor-version-1",
        keys["approval"]: "approval-version-1",
        keys["latest"]: "latest-version-1",
    }
    return objects, versions, keys


def _sources() -> dict[str, dict[str, object]]:
    from test_glm52_task12_retained_state import _recovery_seal_plan

    plan = _recovery_seal_plan()
    recovery = dict(plan.recovery_control.before)
    execution = dict(plan.support_execution.expected)
    execution.update(
        expected_execution_arn=EXECUTION_ARN,
        expected_state_machine_version_arn=STATE_MACHINE_VERSION_ARN,
    )
    control = dict(plan.control.before)
    control.update(
        active_execution_arn=EXECUTION_ARN,
        active_state_machine_version_arn=STATE_MACHINE_VERSION_ARN,
        recovery_control_initial_body_sha256=canonical_record_identity(
            "glm52_production_recovery_control", recovery
        ),
    )
    recovery_owned = dict(plan.recovery_control.after)
    recovery_owned["support_execution_identity_sha256"] = (
        canonical_record_identity("glm52_production_execution", execution)
    )
    return {
        "activation_index": dict(plan.index.expected),
        "control": control,
        "execution": execution,
        "recovery_control": recovery,
        "recovery_control_owned": recovery_owned,
    }


def _finalization_detail() -> dict[str, object]:
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_retained_lifecycle_request_v1",
        "run_id": RUN_ID,
        "activation_id": "activation-1",
        "generation": 1,
        "generation_text": "00000001",
        "reason": "SUPPORT_FINALIZATION_REQUESTED",
        "support_state_identity_sha256": SHA_A,
        "support_execution_arn": SUPPORT_EXECUTION_ARN,
        "support_state_machine_version_arn": (
            SUPPORT_STATE_MACHINE_VERSION_ARN
        ),
        "terminal_v2_identity_sha256": SHA_B,
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def _invocation(
    operation: str,
    *,
    operation_input: dict[str, object] | None = None,
) -> object:
    return SimpleNamespace(
        operation_kind=operation,
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        operation_input=(
            _finalization_detail()
            if operation == "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
            and operation_input is None
            else ({} if operation_input is None else operation_input)
        ),
    )


def _task9_coordinate() -> dict[str, object]:
    from dataclasses import asdict

    from glm52_enforcement.task11_boundary import (
        build_task11_input_coordinate,
    )

    return asdict(
        build_task11_input_coordinate(
            input_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
            bucket=BUCKET,
            key=f"{PREFIX}/task9-deployed-identity.json",
            version_id="task9-version-1",
            file_sha256=SHA_A,
            body_sha256=SHA_B,
        )
    )


def _observation(
    *, correlation: str, observed_at: str
) -> dict[str, object]:
    controller = {
        "members": [],
        "evidence_identity_sha256": canonical_sha256([]),
    }
    request_snapshot = {
        "observed_at": observed_at,
        "pagination_complete": True,
        "requests": [
            {
                "request_id": "request-1",
                "state": "CANCELLED",
                "created_at": "2026-07-29T11:00:00Z",
                "updated_at": observed_at,
                "evidence_identity_sha256": SHA_A,
            }
        ],
        "controller_snapshot": controller,
    }
    job_snapshot = {
        "observed_at": observed_at,
        "pagination_complete": True,
        "jobs": [
            {
                "job_id": "1",
                "request_id": "request-1",
                "state": "SUCCEEDED",
                "evidence_identity_sha256": SHA_B,
            }
        ],
        "controller_snapshot": controller,
    }
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_runtime_observation_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": "activation-1",
        "correlation_identity_sha256": correlation,
        "task9_deployed_identity_sha256": SHA_B,
        "observed_at": observed_at,
        "request_snapshot": request_snapshot,
        "job_snapshot": job_snapshot,
        "controller_snapshot": controller,
        "request_transport": {
            "request_id": "sky-request-1",
            "response_sha256": canonical_sha256(request_snapshot),
            "tls_peer_certificate_sha256": SHA_A,
        },
        "job_transport": {
            "request_id": "sky-job-1",
            "response_sha256": canonical_sha256(job_snapshot),
            "tls_peer_certificate_sha256": SHA_A,
        },
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


class _Lambda:
    def __init__(self, observed: tuple[str, ...]) -> None:
        self.observed = list(observed)
        self.calls: list[dict[str, object]] = []

    def invoke(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        event = json.loads(kwargs["Payload"])
        value = _observation(
            correlation=event["correlation_identity_sha256"],
            observed_at=self.observed.pop(0),
        )
        return {
            "StatusCode": 200,
            "ExecutedVersion": "7",
            "Payload": io.BytesIO(canonical_json_bytes(value)),
            "ResponseMetadata": _meta(f"lambda-{len(self.calls)}"),
        }


class _Dynamo:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, object]]] = {}
        self.recovery: dict[str, object] | None = None
        self.execution: dict[str, object] | None = None
        self.put_mode = "SUCCESS"
        self.query_calls: list[dict[str, object]] = []
        self.put_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self.finalization_record = canonical_json_bytes(
            _finalization_detail()
        ).decode("ascii")

    def query(self, **kwargs: object) -> dict[str, object]:
        self.query_calls.append(dict(kwargs))
        prefix = decode_item(kwargs["ExpressionAttributeValues"])[":prefix"]
        rows = self.rows.get(prefix, [])
        return {
            "Items": [encode_item(row) for row in rows],
            "Count": len(rows),
            "ScannedCount": len(rows),
            "ResponseMetadata": _meta(f"ddb-query-{len(self.query_calls)}"),
        }

    def put_item(self, **kwargs: object) -> dict[str, object]:
        self.put_calls.append(dict(kwargs))
        value = decode_item(kwargs["Item"])
        if value.get("record_type") == "glm52_production_execution":
            if self.execution is None:
                raise RuntimeError("execution fixture is absent")
            expected = decode_item(
                kwargs["ExpressionAttributeValues"]
            )[":before_body"]
            if (
                self.execution.get("canonical_body_sha256") != expected
            ):
                raise RuntimeError("ConditionalCheckFailedException")
            self.execution = value
            if self.put_mode == "LOST_RESPONSE":
                raise TimeoutError("response lost after commit")
            return {"ResponseMetadata": _meta("ddb-execution-put-1")}
        if self.recovery is not None:
            raise RuntimeError("ConditionalCheckFailedException")
        self.recovery = value
        if self.put_mode == "LOST_RESPONSE":
            raise TimeoutError("response lost after commit")
        return {"ResponseMetadata": _meta("ddb-put-1")}

    def update_item(self, **kwargs: object) -> dict[str, object]:
        self.update_calls.append(dict(kwargs))
        if self.execution is None:
            raise RuntimeError("execution fixture is absent")
        values = decode_item(kwargs["ExpressionAttributeValues"])
        expected_state = values[":b_state"]
        expected_revision = values[":b_revision"]
        if (
            self.execution.get("state") != expected_state
            or self.execution.get("revision") != expected_revision
        ):
            raise RuntimeError("ConditionalCheckFailedException")
        names = {
            placeholder: field
            for placeholder, field in kwargs[
                "ExpressionAttributeNames"
            ].items()
        }
        for placeholder, value in values.items():
            if not placeholder.startswith(":a_"):
                continue
            name_placeholder = "#n_" + placeholder.removeprefix(":a_")
            self.execution[names[name_placeholder]] = value
        if self.put_mode == "LOST_RESPONSE":
            raise TimeoutError("response lost after commit")
        return {"ResponseMetadata": _meta("ddb-execution-update-1")}

    def get_item(self, **kwargs: object) -> dict[str, object]:
        self.get_calls.append(dict(kwargs))
        response: dict[str, object] = {
            "ResponseMetadata": _meta(f"ddb-get-{len(self.get_calls)}")
        }
        key = decode_item(kwargs["Key"])
        if str(key["SK"]).endswith("#FINALIZATION_REQUESTED"):
            response["Item"] = encode_item(
                {
                    **key,
                    "record": self.finalization_record,
                }
            )
        elif "#EXECUTION#" in str(key["SK"]) and self.execution is not None:
            response["Item"] = encode_item(self.execution)
        elif self.recovery is not None:
            response["Item"] = encode_item(self.recovery)
        return response


class _StepFunctions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.execution_absent = False
        self.status = "SUCCEEDED"
        self.statuses: list[str] = []

    def describe_execution(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        if self.execution_absent:
            error = RuntimeError("ExecutionDoesNotExist")
            error.response = {
                "Error": {
                    "Code": "ExecutionDoesNotExist",
                    "Message": "Execution Does Not Exist",
                },
                "ResponseMetadata": {
                    "HTTPStatusCode": 400,
                    "RequestId": f"sfn-{len(self.calls)}",
                    "RetryAttempts": 0,
                },
            }
            raise error
        return {
            "executionArn": EXECUTION_ARN,
            "stateMachineArn": STATE_MACHINE_VERSION_ARN.rsplit(":", 1)[0],
            "stateMachineVersionArn": STATE_MACHINE_VERSION_ARN,
            "status": (
                self.statuses.pop(0)
                if self.statuses
                else self.status
            ),
            "ResponseMetadata": _meta(f"sfn-{len(self.calls)}"),
        }


class _S3:
    def __init__(
        self,
        objects: dict[str, bytes],
        versions: dict[str, str],
        keys: dict[str, str],
    ) -> None:
        self.objects = objects
        self.versions = versions
        self.keys = keys
        self.list_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []

    def list_object_versions(self, **kwargs: object) -> dict[str, object]:
        self.list_calls.append(dict(kwargs))
        prefix = kwargs["Prefix"]
        selected = [
            key for key in self.objects
            if key.startswith(prefix)
            and (
                prefix == self.keys["records"]
                or key == self.keys["latest"]
            )
        ]
        return {
            "Versions": [
                {
                    "Key": key,
                    "VersionId": self.versions[key],
                    "IsLatest": True,
                    "ETag": '"' + _sha(self.objects[key]) + '"',
                    "Size": len(self.objects[key]),
                }
                for key in sorted(selected)
            ],
            "DeleteMarkers": [],
            "IsTruncated": False,
            "ResponseMetadata": _meta(f"s3-list-{len(self.list_calls)}"),
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self.get_calls.append(dict(kwargs))
        key = kwargs["Key"]
        raw = self.objects[key]
        assert kwargs["VersionId"] == self.versions[key]
        return {
            "Body": io.BytesIO(raw),
            "VersionId": self.versions[key],
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
            "ETag": '"' + _sha(raw) + '"',
            "ResponseMetadata": _meta(f"s3-get-{len(self.get_calls)}"),
        }


class _EC2:
    def __init__(
        self, instances: list[dict[str, object]] | None = None
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.instances = [] if instances is None else instances

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "Reservations": (
                [] if not self.instances else [{"Instances": self.instances}]
            ),
            "ResponseMetadata": _meta(f"ec2-{len(self.calls)}"),
        }


class _Ports:
    def __init__(
        self,
        *,
        observed: tuple[str, ...] = ("2026-07-29T12:00:00Z",),
    ) -> None:
        objects, versions, keys = _spend_objects()
        self.keys = keys
        self.deployment = SimpleNamespace(
            role_coordinates={
                "ledger_table_name": "keep-glm52-ledger",
                "campaign_bucket": BUCKET,
                "spend_runtime_prefix": PREFIX,
                "campaign_descriptor_key": keys["descriptor"],
                "campaign_descriptor_version_id": versions[keys["descriptor"]],
                "campaign_descriptor_file_sha256": _sha(
                    objects[keys["descriptor"]]
                ),
                "gpu_spend_approval_key": keys["approval"],
                "gpu_spend_approval_version_id": versions[keys["approval"]],
                "gpu_spend_approval_file_sha256": _sha(
                    objects[keys["approval"]]
                ),
                "numeric_binding_version_arn": NUMERIC_VERSION_ARN,
                "task9_deployed_identity_coordinate": _task9_coordinate(),
                "task9_deployed_identity_sha256": SHA_B,
            }
        )
        self.clients = {
            "dynamodb": _Dynamo(),
            "stepfunctions": _StepFunctions(),
            "lambda": _Lambda(observed),
            "s3": _S3(objects, versions, keys),
            "ec2": _EC2(),
        }

    def client(self, service: str) -> object:
        return self.clients[service]


def _live_sources(
    sources: dict[str, dict[str, object]], operation: str
) -> dict[str, dict[str, object]]:
    if operation == "RECONCILE_RETAINED_LIFECYCLE_TRIGGER":
        return {
            name: sources[name]
            for name in ("activation_index", "control", "execution")
        }
    return {
        "execution": sources["execution"],
        "recovery_control": sources["recovery_control_owned"],
    }


def _install_running_execution(
    ports: _Ports,
    sources: dict[str, dict[str, object]],
) -> dict[str, object]:
    from test_glm52_enforcement_dynamodb import _closed_record

    execution = _closed_record(
        "glm52_production_execution",
        activation_id="activation-1",
        state="RUNNING",
        expected_execution_arn=EXECUTION_ARN,
        expected_state_machine_version_arn=STATE_MACHINE_VERSION_ARN,
    )
    sources["execution"] = execution
    recovery = deepcopy(sources["recovery_control_owned"])
    recovery["support_execution_identity_sha256"] = (
        canonical_record_identity(
            "glm52_production_execution", execution
        )
    )
    sources["recovery_control_owned"] = recovery
    ports.clients["dynamodb"].execution = {
        "PK": PK,
        "SK": ledger_sk(
            "glm52_production_execution",
            activation_id="activation-1",
            epoch=execution["epoch"],
        ),
        **execution,
    }
    return execution


def _reconcile_result(
    ports: _Ports, sources: dict[str, dict[str, object]]
) -> tuple[dict[str, object], RuntimeScan]:
    operation = "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=_live_sources(sources, operation),
        ports=ports,
    )
    from glm52_enforcement.task12_lambda_adapters import _materialize_dataclass

    return request, _materialize_dataclass(RuntimeScan, request["runtime_scan"])


def _install_open_allocation(ports: _Ports) -> None:
    from test_glm52_enforcement_dynamodb import _closed_record

    s3 = ports.clients["s3"]
    latest_key = ports.keys["latest"]
    latest = json.loads(s3.objects[latest_key])
    instance_id = "i-00000000000000001"
    start = _self_hash(
        {
            "record_type": "glm52_gpu_spend_event_v1",
            "run_id": RUN_ID,
            "approval_sha256": ports.deployment.role_coordinates[
                "gpu_spend_approval_file_sha256"
            ],
            "event": "allocation_started",
            "job_id": "1",
            "instance_id": instance_id,
            "timestamp": "2026-07-29T11:00:00Z",
            "prior_record_sha256": latest[
                "gpu_spend_authority_sha256"
            ],
        },
        "record_sha256",
    )
    name = (
        "000000-allocation_started-"
        + str(start["record_sha256"])
        + ".json"
    )
    record_key = ports.keys["records"] + name
    record_raw = _raw(start)
    s3.objects[record_key] = record_raw
    s3.versions[record_key] = "record-version-1"
    latest = _self_hash(
        {
            "schema_version": 1,
            "record_type": "glm52_gpu_spend_ledger_latest_v1",
            "run_id": RUN_ID,
            "gpu_spend_authority_sha256": latest[
                "gpu_spend_authority_sha256"
            ],
            "record_count": 1,
            "record_keys": [name],
            "latest_record_sha256": start["record_sha256"],
            "ledger_sha256": _sha(record_raw),
        },
        "latest_body_sha256",
    )
    s3.objects[latest_key] = _raw(latest)
    launch = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="ALLOCATION_OPEN",
        sky_job_id="1",
        observed_instance_ids=[instance_id],
    )
    worker_tags = {
        "Project": "KEEP",
        "Campaign": "GLM-5.2",
        "RunId": RUN_ID,
        "Market": "on-demand",
        "campaign-identity-sha256": launch[
            "campaign_identity_sha256"
        ],
        "activation-id": "activation-1",
        "activation-ordinal-text": "00000001",
        "generation-text": "00000001",
        "allocation-ordinal-text": "00000001",
        "action-key": launch["sky_action_key"],
        "sky-request-id": launch["sky_request_id"],
        "sky-job-name": launch["sky_job_name"],
        "sky-task-name": "glm52-production",
        "task-yaml-sha256": launch["task_yaml_sha256"],
        "request-body-sha256": launch["request_body_sha256"],
    }
    launch["expected_worker_tags_sha256"] = canonical_sha256(worker_tags)
    ports.clients["dynamodb"].rows[
        "ACTIVATION#activation-1#WORKER_LAUNCH#"
    ] = [
        {
            "PK": PK,
            "SK": ledger_sk(
                "glm52_production_worker_launch",
                activation_id="activation-1",
                allocation_ordinal=1,
            ),
            **launch,
        }
    ]
    liability = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        worker_launch_identity_sha256=canonical_record_identity(
            "glm52_production_worker_launch", launch
        ),
        expected_worker_tags_sha256=launch[
            "expected_worker_tags_sha256"
        ],
    )
    ports.clients["dynamodb"].rows[
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#"
    ] = [
        {
            "PK": PK,
            "SK": ledger_sk(
                "glm52_production_worker_launch_liability",
                activation_id="activation-1",
                allocation_ordinal=1,
            ),
            **liability,
        }
    ]
    ports.clients["ec2"].instances = [
        {
            "InstanceId": instance_id,
            "InstanceType": "p5.48xlarge",
            "State": {"Name": "running"},
            "Tags": [
                {"Key": key, "Value": value}
                for key, value in worker_tags.items()
            ],
        }
    ]


def test_supported_runtime_operations_are_exact_and_do_not_claim_teardown() -> None:
    assert SUPPORTED_OPERATIONS == frozenset(
        {
            "RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
            "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT",
        }
    )


def test_reconcile_materializes_live_numeric_spend_and_ec2_without_terminal_v2() -> None:
    ports = _Ports()
    sources = _sources()

    request, _result = _reconcile_result(ports, sources)

    scan = request["runtime_scan"]
    assert [item["identity"] for item in scan["requests"]["members"]] == [
        "request-1"
    ]
    assert [item["identity"] for item in scan["jobs"]["members"]] == ["1"]
    assert scan["controller"]["members"] == []
    assert scan["workers"]["members"] == []
    assert scan["allocations"]["members"] == []
    assert scan["spend"]["state"] == "CLOSED"
    assert scan["spend"]["remaining_gpu_seconds"] == 86400
    assert scan["spend"]["remaining_gpu_cost_usd"] == "1320.96"
    assert len(ports.clients["lambda"].calls) == 1
    assert len(ports.clients["s3"].list_calls) == 2
    assert len(ports.clients["ec2"].calls) == 1
    assert ports.clients["stepfunctions"].calls == [
        {"executionArn": EXECUTION_ARN}
    ]


def test_reconcile_strongly_reads_exact_finalization_detail_before_runtime() -> None:
    ports = _Ports()
    sources = _sources()
    operation = "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"

    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=_live_sources(sources, operation),
        ports=ports,
    )

    first = ports.clients["dynamodb"].get_calls[0]
    assert first == {
        "TableName": "keep-glm52-ledger",
        "Key": encode_item(
            {
                "PK": PK,
                "SK": (
                    "ACTIVATION#activation-1#FINALIZATION_REQUESTED"
                ),
            }
        ),
        "ConsistentRead": True,
        "ReturnConsumedCapacity": "NONE",
    }
    assert request["runtime_scan"]["spend"]["state"] == "CLOSED"


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_detail",
        "foreign_reason",
        "bad_self_hash",
        "foreign_activation",
        "durable_bytes_mismatch",
    ),
)
def test_reconcile_rejects_foreign_finalization_before_runtime_or_mutation(
    mutation: str,
) -> None:
    ports = _Ports()
    sources = _sources()
    operation = "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
    detail = _finalization_detail()
    if mutation == "missing_detail":
        detail = {}
    elif mutation == "foreign_reason":
        detail["reason"] = "SUPPORT_DEADLINE_EXPIRED"
        body = dict(detail)
        body.pop("canonical_identity_sha256")
        detail["canonical_identity_sha256"] = canonical_sha256(body)
    elif mutation == "bad_self_hash":
        detail["canonical_identity_sha256"] = SHA_B
    elif mutation == "foreign_activation":
        detail["activation_id"] = "activation-foreign"
        body = dict(detail)
        body.pop("canonical_identity_sha256")
        detail["canonical_identity_sha256"] = canonical_sha256(body)
    elif mutation == "durable_bytes_mismatch":
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

    with pytest.raises(Task12LiveRuntimeError):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(
                operation,
                operation_input=detail,
            ),
            live_sources=_live_sources(sources, operation),
            ports=ports,
        )

    assert ports.clients["lambda"].calls == []
    assert ports.clients["stepfunctions"].calls == []
    assert ports.clients["s3"].list_calls == []
    assert ports.clients["s3"].get_calls == []
    assert ports.clients["ec2"].calls == []
    assert ports.clients["dynamodb"].put_calls == []


def test_live_open_allocation_is_charged_through_numeric_observation_and_bound_to_ec2() -> None:
    ports = _Ports()
    _install_open_allocation(ports)
    sources = _sources()

    request, _result = _reconcile_result(ports, sources)

    scan = request["runtime_scan"]
    assert scan["workers"]["members"] == [
        {
            "identity": "i-00000000000000001",
            "state": "running",
            "observed_at": "2026-07-29T12:00:00Z",
            "evidence_identity_sha256": (
                scan["workers"]["members"][0][
                    "evidence_identity_sha256"
                ]
            ),
        }
    ]
    assert scan["allocations"]["members"][0]["state"] == "OPEN"
    assert scan["spend"]["state"] == "OPEN"
    assert scan["spend"]["remaining_gpu_seconds"] == 81900
    assert scan["spend"]["remaining_gpu_cost_usd"] == "1252.16"
    assert len(ports.clients["s3"].get_calls) == 4
    assert ports.clients["ec2"].calls[0]["Filters"][0] == {
        "Name": "tag:RunId",
        "Values": [RUN_ID],
    }


def test_live_held_liability_reserve_must_match_its_worker_launch_custody() -> None:
    ports = _Ports()
    _install_open_allocation(ports)
    sources = _sources()
    liability = ports.clients["dynamodb"].rows[
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#"
    ][0]
    liability["gpu_liability_reserve_ledger_identity_sha256"] = "c" * 64

    with pytest.raises(
        Task12LiveRuntimeError,
        match="liability reserve custody",
    ):
        _reconcile_result(ports, sources)


def test_terminal_v2_cannot_substitute_for_missing_numeric_observation() -> None:
    ports = _Ports()
    sources = _sources()
    del ports.clients["lambda"]
    ports.clients["dynamodb"].rows[
        "ACTIVATION#activation-1#TASK12_VERSIONED_WRITER_CONTROL#"
    ] = [{"record_type": "future-terminal-v2-is-not-runtime-truth"}]

    with pytest.raises(
        Task12LiveRuntimeError, match="NumericBinding|Lambda"
    ):
        _reconcile_result(ports, sources)


def test_durable_unresolved_start_incident_accepts_only_authenticated_execution_absence() -> None:
    from test_glm52_enforcement_dynamodb import _closed_record

    ports = _Ports()
    ports.clients["stepfunctions"].execution_absent = True
    sources = _sources()
    execution = _closed_record(
        "glm52_production_execution",
        activation_id="activation-1",
        state="START_POSSIBLY_SENT_UNRESOLVED_INCIDENT",
        expected_execution_arn=EXECUTION_ARN,
        expected_state_machine_version_arn=STATE_MACHINE_VERSION_ARN,
    )
    sources["execution"] = execution

    request, _result = _reconcile_result(ports, sources)

    assert request["runtime_scan"]["controller"]["members"] == []
    assert ports.clients["stepfunctions"].calls == [
        {"executionArn": EXECUTION_ARN}
    ]


def test_terminal_proof_collects_two_distinct_live_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = (
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT"
    )
    ports = _Ports(
        observed=(
            "2026-07-29T12:00:00Z",
            "2026-07-29T12:01:00Z",
        )
    )
    sources = _sources()
    sleeps: list[int] = []
    monkeypatch.setattr(
        "glm52_enforcement.task12_live_runtime._sleep",
        lambda seconds: sleeps.append(seconds),
    )

    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=_live_sources(sources, operation),
        ports=ports,
    )

    assert sleeps == [60]
    assert request["minimum_quiet_seconds"] == 60
    assert request["first_runtime_scan"]["requests"]["observed_at"] == (
        "2026-07-29T12:00:00Z"
    )
    assert request["second_runtime_scan"]["requests"]["observed_at"] == (
        "2026-07-29T12:01:00Z"
    )
    assert len(ports.clients["lambda"].calls) == 2


def test_terminal_proof_rejects_support_execution_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = (
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT"
    )
    ports = _Ports(
        observed=(
            "2026-07-29T12:00:00Z",
            "2026-07-29T12:00:05Z",
            "2026-07-29T12:00:10Z",
        )
    )
    sources = _sources()
    _install_running_execution(ports, sources)
    ports.clients["stepfunctions"].status = "RUNNING"
    monkeypatch.setattr(
        "glm52_enforcement.task12_live_runtime._sleep",
        lambda seconds: sleeps.append(seconds),
    )
    monkeypatch.setattr(
        "glm52_enforcement.task12_live_runtime."
        "EXECUTION_TERMINAL_MAX_POLLS",
        2,
    )
    sleeps: list[int] = []

    with pytest.raises(
        Task12LiveRuntimeError,
        match="support execution is not terminal",
    ):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources=_live_sources(sources, operation),
            ports=ports,
        )

    assert ports.clients["dynamodb"].update_calls == []
    assert sleeps == [5, 5]
    assert len(ports.clients["stepfunctions"].calls) == 3


def test_terminal_proof_polls_exact_execution_until_terminal_then_quiets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = (
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT"
    )
    ports = _Ports(
        observed=(
            "2026-07-29T12:00:00Z",
            "2026-07-29T12:00:05Z",
            "2026-07-29T12:01:05Z",
        )
    )
    sources = _sources()
    _install_running_execution(ports, sources)
    ports.clients["stepfunctions"].statuses = [
        "RUNNING",
        "ABORTED",
        "ABORTED",
    ]
    sleeps: list[int] = []
    monkeypatch.setattr(
        "glm52_enforcement.task12_live_runtime._sleep",
        lambda seconds: sleeps.append(seconds),
    )

    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=_live_sources(sources, operation),
        ports=ports,
    )

    assert sleeps == [5, 60]
    assert len(ports.clients["stepfunctions"].calls) == 3
    assert len(ports.clients["lambda"].calls) == 3
    assert request["first_runtime_scan"]["requests"]["observed_at"] == (
        "2026-07-29T12:00:05Z"
    )
    assert request["second_runtime_scan"]["requests"]["observed_at"] == (
        "2026-07-29T12:01:05Z"
    )


@pytest.mark.parametrize("put_mode", ["SUCCESS", "LOST_RESPONSE"])
def test_terminal_proof_reconciles_live_aborted_execution_before_proving(
    monkeypatch: pytest.MonkeyPatch,
    put_mode: str,
) -> None:
    operation = (
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT"
    )
    ports = _Ports(
        observed=(
            "2026-07-29T12:00:00Z",
            "2026-07-29T12:01:00Z",
        )
    )
    sources = _sources()
    running = _install_running_execution(ports, sources)
    ports.clients["stepfunctions"].status = "ABORTED"
    ports.clients["dynamodb"].put_mode = put_mode
    monkeypatch.setattr(
        "glm52_enforcement.task12_live_runtime._sleep",
        lambda _seconds: None,
    )

    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=_live_sources(sources, operation),
        ports=ports,
    )

    assert request["first_runtime_scan"]["requests"]["observed_at"] == (
        "2026-07-29T12:00:00Z"
    )
    assert request["second_runtime_scan"]["requests"]["observed_at"] == (
        "2026-07-29T12:01:00Z"
    )
    assert len(ports.clients["dynamodb"].update_calls) == 1
    execution_update = ports.clients["dynamodb"].update_calls[0]
    assert execution_update["ConditionExpression"].startswith(
        "#n_account_id = :b_account_id AND "
    )
    assert "#n_state = :b_state" in execution_update[
        "ConditionExpression"
    ]
    assert "#n_revision = :b_revision" in execution_update[
        "ConditionExpression"
    ]
    exact = dict(ports.clients["dynamodb"].execution)
    assert exact.pop("PK") == PK
    assert exact.pop("SK") == ledger_sk(
        "glm52_production_execution",
        activation_id="activation-1",
        epoch=running["epoch"],
    )
    assert exact["state"] == "ABORTED"
    assert exact["terminal_status"] == "ABORTED"
    assert exact["terminal_observed_at"] == "2026-07-29T12:00:00Z"
    assert exact["revision"] == running["revision"] + 1
    assert exact["updated_at"] == running["updated_at"]
    assert exact["describe_execution_request_id"] == "sfn-1"
    assert exact["describe_execution_response_sha256"] != SHA_A
    assert exact["terminal_body_sha256"] != SHA_A


def test_terminal_proof_retry_accepts_only_its_seal_bound_terminal_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = (
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT"
    )
    ports = _Ports(
        observed=(
            "2026-07-29T12:00:00Z",
            "2026-07-29T12:01:00Z",
        )
    )
    sources = _sources()
    _install_running_execution(ports, sources)
    ports.clients["stepfunctions"].status = "ABORTED"
    monkeypatch.setattr(
        "glm52_enforcement.task12_live_runtime._sleep",
        lambda _seconds: None,
    )
    live_sources = _live_sources(sources, operation)

    materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=live_sources,
        ports=ports,
    )
    terminal = dict(ports.clients["dynamodb"].execution)
    terminal.pop("PK")
    terminal.pop("SK")
    retry_sources = {
        **live_sources,
        "execution": terminal,
    }
    ports.clients["lambda"].observed = [
        "2026-07-29T13:00:00Z",
        "2026-07-29T13:01:00Z",
    ]

    retried = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=retry_sources,
        ports=ports,
    )

    assert retried["second_runtime_scan"]["requests"]["observed_at"] == (
        "2026-07-29T13:01:00Z"
    )
    assert len(ports.clients["dynamodb"].update_calls) == 1

    foreign = {
        **terminal,
        "updated_at": terminal["terminal_observed_at"],
    }
    with pytest.raises(
        Task12LiveRuntimeError,
        match="recovery support execution source drifted"
        "|predecessor body identity drifted",
    ):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={**retry_sources, "execution": foreign},
            ports=ports,
        )


@pytest.mark.parametrize(
    "future_state", ["TERMINAL_V2_PUBLISHED", "RECOVERY_COMPLETE"]
)
def test_terminal_proof_rejects_post_terminal_v2_recovery_state_before_live_calls(
    future_state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_glm52_enforcement_dynamodb import _closed_record

    operation = (
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT"
    )
    ports = _Ports(
        observed=(
            "2026-07-29T12:00:00Z",
            "2026-07-29T12:01:00Z",
        )
    )
    sources = _sources()
    execution = sources["execution"]
    sources["recovery_control_owned"] = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
        state=future_state,
        support_execution_identity_sha256=canonical_record_identity(
            "glm52_production_execution", execution
        ),
    )
    monkeypatch.setattr(
        "glm52_enforcement.task12_live_runtime._sleep",
        lambda _seconds: None,
    )

    with pytest.raises(
        Task12LiveRuntimeError,
        match="recovery control source is not sealed",
    ):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources=_live_sources(sources, operation),
            ports=ports,
        )

    assert ports.clients["lambda"].calls == []
    assert ports.clients["stepfunctions"].calls == []
    assert ports.clients["dynamodb"].put_calls == []


@pytest.mark.parametrize("put_mode", ["SUCCESS", "LOST_RESPONSE"])
def test_reconcile_creates_or_adopts_exact_canonical_recovery_control(
    put_mode: str,
) -> None:
    ports = _Ports()
    ports.clients["dynamodb"].put_mode = put_mode
    sources = _sources()
    request, result = _reconcile_result(ports, sources)

    assert persist_live_successors(
        operation_kind="RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
        invocation=_invocation("RECONCILE_RETAINED_LIFECYCLE_TRIGGER"),
        live_sources=_live_sources(
            sources, "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
        ),
        request=request,
        domain_result=result,
        ports=ports,
    ) is True

    ddb = ports.clients["dynamodb"]
    assert len(ddb.put_calls) == 1
    assert ddb.put_calls[0]["ConditionExpression"] == (
        "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
    )
    assert ddb.get_calls[-1]["ConsistentRead"] is True
    stored = dict(ddb.recovery)
    assert stored.pop("PK") == PK
    assert stored.pop("SK") == ledger_sk(
        "glm52_production_recovery_control",
        activation_id="activation-1",
    )
    assert stored == sources["recovery_control"]


def test_reconcile_rejects_foreign_preexisting_recovery_control() -> None:
    ports = _Ports()
    sources = _sources()
    foreign = deepcopy(sources["recovery_control"])
    foreign["updated_at"] = "2026-07-28T12:00:01Z"
    ports.clients["dynamodb"].recovery = {
        "PK": PK,
        "SK": ledger_sk(
            "glm52_production_recovery_control",
            activation_id="activation-1",
        ),
        **foreign,
    }
    request, result = _reconcile_result(ports, sources)

    with pytest.raises(
        Task12LiveRuntimeError, match="foreign recovery control"
    ):
        persist_live_successors(
            operation_kind="RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
            invocation=_invocation("RECONCILE_RETAINED_LIFECYCLE_TRIGGER"),
            live_sources=_live_sources(
                sources, "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
            ),
            request=request,
            domain_result=result,
            ports=ports,
        )


def test_source_drift_is_rejected_before_live_calls() -> None:
    ports = _Ports()
    sources = _sources()
    sources["control"]["activation_ordinal"] = 2

    with pytest.raises(Task12LiveRuntimeError, match="source drift"):
        _reconcile_result(ports, sources)
    assert ports.clients["lambda"].calls == []
    assert ports.clients["s3"].list_calls == []


def test_spend_latest_transport_retry_is_not_adopted() -> None:
    ports = _Ports()
    sources = _sources()
    original = ports.clients["s3"].list_object_versions

    def retried(**kwargs: object) -> dict[str, object]:
        response = original(**kwargs)
        response["ResponseMetadata"]["RetryAttempts"] = 1
        return response

    ports.clients["s3"].list_object_versions = retried
    with pytest.raises(Task12LiveRuntimeError, match="RetryAttempts"):
        _reconcile_result(ports, sources)


def test_terminal_observation_persists_no_fabricated_successor() -> None:
    from glm52_enforcement.task12_runtime import prove_runtime_terminal
    from test_glm52_task12_runtime import _scan

    operation = (
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT"
    )
    first = _scan(observed_at="2026-07-29T12:00:00Z")
    second = _scan(observed_at="2026-07-29T12:01:00Z")
    proof = prove_runtime_terminal(
        first=first,
        second=second,
        minimum_quiet_seconds=60,
    )
    ports = _Ports()
    sources = _sources()

    assert persist_live_successors(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=_live_sources(sources, operation),
        request={
            "first_runtime_scan": json.loads(json.dumps(asdict(first))),
            "second_runtime_scan": json.loads(json.dumps(asdict(second))),
            "minimum_quiet_seconds": 60,
        },
        domain_result=proof,
        ports=ports,
    ) is True
    assert ports.clients["dynamodb"].put_calls == []
