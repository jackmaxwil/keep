"""Postpublication materialization for the retained Task 12 runtime.

This boundary runs only after CloudFormation has published the retained Lambda
and Step Functions versions.  It reads those physical version ARNs back,
authenticates one versioned static source authority, constructs immutable
operation descriptors over canonical retained sources, publishes one authority
manifest per caller edge, and conditionally writes the 32 descriptors plus ten
deployment records.  The tenth edge binds the already-published TerminalV2
version to the exact support workflow and SupportDeadline versions read from
the authenticated support stack.  It never predicts future runtime observations.
Every ambiguous write is reconciled with an authenticated, strongly consistent
read so a crash or lost success response cannot create a second authority.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .canonical import canonical_json_bytes, canonical_sha256
from .dynamodb import decode_item, encode_item
from .task12_operations import (
    OPERATION_SPECS,
    Task12OperationContractError,
    build_operation_descriptor_graph,
    validate_operation_descriptor,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
PARTITION_KEY = f"RUN#{RUN_ID}"
DEPLOYMENT_PREFIX = "TASK12_LAMBDA_DEPLOYMENT#"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ACTIVATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STACK_ID = re.compile(
    r"^arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"[A-Za-z0-9._-]+/[0-9a-fA-F-]{36}$"
)
_LAMBDA_VERSION = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:"
    r"([A-Za-z0-9_-]+):([1-9][0-9]*)$"
)
_STATE_VERSION = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:stateMachine:"
    r"([A-Za-z0-9_-]+):([1-9][0-9]*)$"
)
_RESOURCE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,255}$")
_S3_BUCKET = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)[a-z0-9]"
    r"(?:[a-z0-9.-]{1,61}[a-z0-9])?$"
)
_KMS_ARN = re.compile(
    r"^arn:aws:kms:us-west-2:246813579024:key/[0-9a-fA-F-]{36}$"
)


class Task12PostpublicationError(ValueError):
    """Postpublication AWS truth or immutable materialization drifted."""


@dataclass(frozen=True)
class Task12PostpublicationServices:
    cloudformation: object
    dynamodb: object
    s3: object


@dataclass(frozen=True)
class Task12OperationSourceCoordinate:
    bucket: str
    key: str
    version_id: str
    file_sha256: str


@dataclass(frozen=True)
class Task12PostpublicationInputs:
    retained_stack_id: str
    support_stack_id: str
    activation_id: str
    activation_ordinal: int
    generation: int
    dispatch_identity_sha256: str
    ledger_table_name: str
    authority_bucket: str
    campaign_bucket: str
    kms_key_id: str
    worker_drain_document_name: str
    worker_drain_document_version: str
    operation_source: Task12OperationSourceCoordinate
    campaign_descriptor: Task12OperationSourceCoordinate
    gpu_spend_approval: Task12OperationSourceCoordinate


_OPERATIONS: Mapping[str, tuple[str, ...]] = {
    "ExecutionObserver": (
        "RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
        "RETAINED_ACQUIRE_RECOVERY_SEALING",
        "RETAINED_STOP_OR_OBSERVE_SUPPORT_EXECUTION",
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT",
        "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION",
        "RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
        "RETAINED_ACQUIRE_TEARDOWN_SEALING",
        "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
        "RETAINED_ENTER_TEARDOWN_SEALED",
    ),
    "OperatorDisposition": (
        "RETAINED_CORRELATE_REQUESTS_AND_JOBS",
        "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION",
    ),
    "WorkerDrain": (
        "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
        "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
        "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
    ),
    "TerminalV2": (
        "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
        "RETAINED_ENTER_RECOVERY_COMPLETE",
    ),
    "TerminalV2Support": (
        "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
    ),
    "Finalizer": (
        "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED",
        "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
    ),
    "OrphanAudit": (
        "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT",
        "RETAINED_AUDIT_SUPPORT_ORPHANS",
    ),
    "SnapshotCleanupRetained": (
        "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",
    ),
    "SnapshotCleanupSnapshot": (
        "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE",
        "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION",
        "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY",
        "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
        "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK",
        "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
        "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE",
        "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT",
        "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT",
    ),
    "H1GDrained": ("RETAINED_INVOKE_H1G_DRAINED_WRITER",),
}

_EDGES = (
    ("ExecutionObserver", "RETAINED_EXECUTION_OBSERVER", "RETAINED", "RetainedLifecycle"),
    ("TerminalV2", "RETAINED_TERMINAL_V2", "RETAINED", "RetainedLifecycle"),
    (
        "TerminalV2Support",
        "RETAINED_TERMINAL_V2",
        "SUPPORT_OBSERVER",
        "Support",
    ),
    ("Finalizer", "RETAINED_FINALIZER", "RETAINED", "RetainedLifecycle"),
    ("H1GDrained", "RETAINED_H1G_DRAINED", "RETAINED", "RetainedLifecycle"),
    ("WorkerDrain", "RETAINED_WORKER_DRAIN", "RETAINED", "RetainedLifecycle"),
    ("OperatorDisposition", "RETAINED_OPERATOR_DISPOSITION", "RETAINED", "RetainedLifecycle"),
    ("OrphanAudit", "RETAINED_ORPHAN_AUDIT", "RETAINED", "RetainedLifecycle"),
    ("SnapshotCleanupRetained", "RETAINED_SNAPSHOT_CLEANUP", "SNAPSHOT_CLEANUP", "RetainedLifecycle"),
    ("SnapshotCleanupSnapshot", "RETAINED_SNAPSHOT_CLEANUP", "SNAPSHOT_CLEANUP", "SnapshotCleanup"),
)


def _method(service: object, name: str) -> object:
    value = getattr(service, name, None)
    if not callable(value):
        raise TypeError("Task 12 postpublication service lacks " + name)
    return value


def _authenticated(response: object, operation: str) -> Mapping[str, object]:
    if type(response) is not dict:
        raise Task12PostpublicationError(operation + " returned no object")
    metadata = response.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") not in {200, 201}
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") not in {None, 0}
    ):
        raise Task12PostpublicationError(operation + " response is unauthenticated")
    return response


def _state_key(
    activation_id: str, handler_kind: str, operation: str, generation_text: str
) -> str:
    return (
        f"ACTIVATION#{activation_id}#TASK12_LAMBDA_INPUT#{handler_kind}#"
        f"{operation}#{generation_text}"
    )


def _canonical_mapping(raw: bytes, label: str) -> dict[str, object]:
    import json

    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Task12PostpublicationError(
            label + " is not canonical ASCII JSON"
        ) from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise Task12PostpublicationError(
            label + " is not one canonical JSON object"
        )
    return value


def _read_operation_source(
    inputs: Task12PostpublicationInputs,
    services: Task12PostpublicationServices,
) -> tuple[
    str,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, str],
    str,
]:
    coordinate = inputs.operation_source
    if (
        type(coordinate) is not Task12OperationSourceCoordinate
        or _S3_BUCKET.fullmatch(coordinate.bucket) is None
        or type(coordinate.key) is not str
        or coordinate.key.startswith("/")
        or any(part in {"", ".", ".."} for part in coordinate.key.split("/"))
        or type(coordinate.version_id) is not str
        or not coordinate.version_id
        or coordinate.version_id == "null"
        or _SHA.fullmatch(coordinate.file_sha256) is None
    ):
        raise Task12PostpublicationError(
            "operation source coordinate is not exact and versioned"
        )
    response = _authenticated(
        _method(services.s3, "get_object")(
            Bucket=coordinate.bucket,
            Key=coordinate.key,
            VersionId=coordinate.version_id,
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        "GetOperationSource",
    )
    body = response.get("Body")
    try:
        raw = body.read()
    except Exception as exc:
        raise Task12PostpublicationError(
            "operation source body is unreadable"
        ) from exc
    expected_checksum = base64.b64encode(
        hashlib.sha256(raw).digest()
    ).decode("ascii") if type(raw) is bytes else None
    if (
        type(raw) is not bytes
        or response.get("VersionId") != coordinate.version_id
        or response.get("ChecksumSHA256") != expected_checksum
        or hashlib.sha256(raw).hexdigest() != coordinate.file_sha256
    ):
        raise Task12PostpublicationError(
            "operation source transport identity drifted"
        )
    value = _canonical_mapping(raw, "operation source authority")
    fields = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "dispatch_identity_sha256",
        "ledger_table_name",
        "ledger_partition_key",
        "authority_bucket",
        "campaign_bucket",
        "kms_key_id",
        "worker_drain_document_name",
        "worker_drain_document_version",
        "task9_deployed_identity_coordinate",
        "task9_deployed_identity_sha256",
        "campaign_descriptor_coordinate",
        "gpu_spend_approval_coordinate",
        "task10_worker_descriptor_coordinate",
        "worker_script_hashes",
        "terminal_evidence_prefix",
        "spend_runtime_prefix",
        "descriptor_inventory",
        "canonical_body_sha256",
    }
    body_value = dict(value)
    supplied_hash = body_value.pop("canonical_body_sha256", None)
    expected_inventory = sorted(OPERATION_SPECS)
    try:
        from .task11_support_boundary import (
            exact_input_coordinate_from_mapping,
        )

        task9_coordinate = exact_input_coordinate_from_mapping(
            value.get("task9_deployed_identity_coordinate"),
            expected_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
        )
    except (TypeError, ValueError) as exc:
        raise Task12PostpublicationError(
            "Task 9 deployed identity coordinate is invalid"
        ) from exc
    descriptor_coordinate = value.get("campaign_descriptor_coordinate")
    approval_coordinate = value.get("gpu_spend_approval_coordinate")
    worker_coordinate = value.get("task10_worker_descriptor_coordinate")
    worker_script_hashes = value.get("worker_script_hashes")
    expected_descriptor = {
        "bucket": inputs.campaign_descriptor.bucket,
        "key": inputs.campaign_descriptor.key,
        "version_id": inputs.campaign_descriptor.version_id,
        "file_sha256": inputs.campaign_descriptor.file_sha256,
    }
    expected_approval = {
        "bucket": inputs.gpu_spend_approval.bucket,
        "key": inputs.gpu_spend_approval.key,
        "version_id": inputs.gpu_spend_approval.version_id,
        "file_sha256": inputs.gpu_spend_approval.file_sha256,
    }
    if (
        set(value) != fields
        or value.get("schema_version") != 1
        or value.get("record_type")
        != "glm52_task12_operation_static_authority_v1"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or value.get("activation_id") != inputs.activation_id
        or value.get("activation_ordinal") != inputs.activation_ordinal
        or value.get("generation") != inputs.generation
        or value.get("generation_text") != f"{inputs.generation:08d}"
        or value.get("dispatch_identity_sha256")
        != inputs.dispatch_identity_sha256
        or value.get("ledger_table_name") != inputs.ledger_table_name
        or value.get("ledger_partition_key") != PARTITION_KEY
        or value.get("authority_bucket") != inputs.authority_bucket
        or value.get("campaign_bucket") != inputs.campaign_bucket
        or value.get("kms_key_id") != inputs.kms_key_id
        or value.get("worker_drain_document_name")
        != inputs.worker_drain_document_name
        or value.get("worker_drain_document_version")
        != inputs.worker_drain_document_version
        or task9_coordinate.bucket != inputs.campaign_bucket
        or task9_coordinate.key
        != (
            f"campaigns/{RUN_ID}/authorities/task9/"
            f"{inputs.activation_id}/TASK9_DEPLOYED_IDENTITY.json"
        )
        or value.get("task9_deployed_identity_sha256")
        != task9_coordinate.body_sha256
        or descriptor_coordinate != expected_descriptor
        or approval_coordinate != expected_approval
        or type(worker_coordinate) is not dict
        or set(worker_coordinate)
        != {
            "bucket",
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
        }
        or worker_coordinate.get("bucket") != inputs.campaign_bucket
        or worker_coordinate.get("key")
        != "task13/production/task10-worker-descriptor.json"
        or type(worker_coordinate.get("version_id")) is not str
        or not worker_coordinate["version_id"]
        or worker_coordinate["version_id"] == "null"
        or any(
            type(worker_coordinate.get(field)) is not str
            or _SHA.fullmatch(worker_coordinate[field]) is None
            for field in ("file_sha256", "body_sha256")
        )
        or type(worker_script_hashes) is not dict
        or set(worker_script_hashes)
        != {
            "glm52_checkpoint_commit.py",
            "glm52_drain_and_stop.py",
            "glm52_deadline_guard.py",
        }
        or any(
            type(item) is not str or _SHA.fullmatch(item) is None
            for item in worker_script_hashes.values()
        )
        or value.get("terminal_evidence_prefix")
        != (
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{inputs.generation:08d}/terminal-evidence/"
        )
        or inputs.campaign_descriptor.bucket != inputs.campaign_bucket
        or inputs.gpu_spend_approval.bucket != inputs.campaign_bucket
        or inputs.campaign_descriptor.key
        != (
            f"campaigns/{RUN_ID}/submissions/production/"
            "descriptor.json"
        )
        or inputs.gpu_spend_approval.key
        != f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json"
        or value.get("spend_runtime_prefix")
        != f"campaigns/{RUN_ID}/runtime"
        or value.get("descriptor_inventory") != expected_inventory
        or supplied_hash != canonical_sha256(body_value)
    ):
        raise Task12PostpublicationError(
            "operation source authority is foreign or incomplete"
        )
    return (
        supplied_hash,
        dict(value["task9_deployed_identity_coordinate"]),
        dict(descriptor_coordinate),
        dict(approval_coordinate),
        dict(worker_coordinate),
        dict(worker_script_hashes),
        str(value["terminal_evidence_prefix"]),
    )


def _closed_operation_rows(
    inputs: Task12PostpublicationInputs,
    services: Task12PostpublicationServices,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, str],
    str,
]:
    expected = {operation for operations in _OPERATIONS.values() for operation in operations}
    (
        static_authority_sha256,
        task9_coordinate,
        campaign_descriptor_coordinate,
        gpu_spend_approval_coordinate,
        task10_worker_descriptor_coordinate,
        worker_script_hashes,
        terminal_evidence_prefix,
    ) = _read_operation_source(inputs, services)
    try:
        descriptors = build_operation_descriptor_graph(
            activation_id=inputs.activation_id,
            activation_ordinal=inputs.activation_ordinal,
            generation=inputs.generation,
            ledger_partition_key=PARTITION_KEY,
            campaign_bucket=inputs.campaign_bucket,
            kms_key_id=inputs.kms_key_id,
            static_authority_sha256=static_authority_sha256,
        )
    except Task12OperationContractError as exc:
        raise Task12PostpublicationError(
            "operation descriptor graph is not closed"
        ) from exc
    if set(descriptors) != expected:
        raise Task12PostpublicationError(
            "operation descriptor inventory is not the closed 32-state graph"
        )
    generation_text = f"{inputs.generation:08d}"
    rows: dict[str, dict[str, object]] = {}
    for edge_key, operations in _OPERATIONS.items():
        handler_kind = next(edge[1] for edge in _EDGES if edge[0] == edge_key)
        for operation in operations:
            try:
                body = validate_operation_descriptor(descriptors[operation])
            except Task12OperationContractError as exc:
                raise Task12PostpublicationError(
                    operation + " descriptor schema is not closed"
                ) from exc
            if (
                body.get("handler_kind") != handler_kind
                or body.get("operation_kind") != operation
                or body.get("activation_id") != inputs.activation_id
                or body.get("activation_ordinal") != inputs.activation_ordinal
                or body.get("generation") != inputs.generation
                or body.get("generation_text") != generation_text
            ):
                raise Task12PostpublicationError(operation + " input identity drifted")
            rows[operation] = body
    return (
        rows,
        task9_coordinate,
        campaign_descriptor_coordinate,
        gpu_spend_approval_coordinate,
        task10_worker_descriptor_coordinate,
        worker_script_hashes,
        terminal_evidence_prefix,
    )


def _read_versions(
    inputs: Task12PostpublicationInputs, services: Task12PostpublicationServices
) -> dict[str, str]:
    describe = _method(services.cloudformation, "describe_stacks")
    response = _authenticated(
        describe(StackName=inputs.retained_stack_id), "DescribeStacks"
    )
    stacks = response.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1 or type(stacks[0]) is not dict:
        raise Task12PostpublicationError("retained stack readback is not singular")
    stack = stacks[0]
    parameters = stack.get("Parameters")
    actual_parameters = (
        {
            row["ParameterKey"]: row["ParameterValue"]
            for row in parameters
            if type(row) is dict
            and set(row) == {"ParameterKey", "ParameterValue"}
        }
        if type(parameters) is list
        else {}
    )
    expected_parameters = {
        "Task12ActivationOrdinal": str(inputs.activation_ordinal),
        "Task12Generation": str(inputs.generation),
        "Task12GenerationText": f"{inputs.generation:08d}",
        "Task12DispatchIdentitySha256": inputs.dispatch_identity_sha256,
    }
    if (
        stack.get("StackId") != inputs.retained_stack_id
        or stack.get("StackStatus") not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
        or any(actual_parameters.get(key) != value for key, value in expected_parameters.items())
    ):
        raise Task12PostpublicationError("retained stack parameters are not exact publication truth")
    list_resources = _method(services.cloudformation, "list_stack_resources")
    resource_response = _authenticated(
        list_resources(StackName=inputs.retained_stack_id),
        "ListStackResources",
    )
    if "NextToken" in resource_response:
        raise Task12PostpublicationError("retained stack resource readback is truncated")
    summaries = resource_response.get("StackResourceSummaries")
    if type(summaries) is not list:
        raise Task12PostpublicationError("retained resource readback is absent")
    physical: dict[str, str] = {}
    for row in summaries:
        if (
            type(row) is not dict
            or type(row.get("LogicalResourceId")) is not str
            or type(row.get("PhysicalResourceId")) is not str
            or row["LogicalResourceId"] in physical
        ):
            raise Task12PostpublicationError("retained resource readback is malformed")
        physical[row["LogicalResourceId"]] = row["PhysicalResourceId"]
    required = {
        *(
            logical.removesuffix("Retained").removesuffix("Snapshot")
            + "Version"
            for logical, *_ in _EDGES
            if logical != "TerminalV2Support"
        ),
        "NumericBindingVersion",
        "RetainedCancellationVersion",
        "RetainedLifecycleStateMachineVersion",
        "SnapshotCleanupStateMachineVersion",
        "SnapshotCleanupScheduleInvokeRole",
    }
    if not required.issubset(physical):
        raise Task12PostpublicationError("retained published versions are incomplete")
    for logical_id in required - {"SnapshotCleanupScheduleInvokeRole"}:
        pattern = _STATE_VERSION if "StateMachineVersion" in logical_id else _LAMBDA_VERSION
        if pattern.fullmatch(physical[logical_id]) is None:
            raise Task12PostpublicationError(logical_id + " is not an exact published version ARN")
    if _RESOURCE_NAME.fullmatch(
        physical["SnapshotCleanupScheduleInvokeRole"]
    ) is None:
        raise Task12PostpublicationError(
            "snapshot cleanup schedule role readback is invalid"
        )
    support_response = _authenticated(
        describe(StackName=inputs.support_stack_id),
        "DescribeSupportStack",
    )
    support_stacks = support_response.get("Stacks")
    if (
        type(support_stacks) is not list
        or len(support_stacks) != 1
        or type(support_stacks[0]) is not dict
        or support_stacks[0].get("StackId") != inputs.support_stack_id
        or support_stacks[0].get("StackStatus")
        not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
    ):
        raise Task12PostpublicationError(
            "support stack readback is not exact publication truth"
        )
    support_resources = _authenticated(
        list_resources(StackName=inputs.support_stack_id),
        "ListSupportStackResources",
    )
    if "NextToken" in support_resources:
        raise Task12PostpublicationError(
            "support stack resource readback is truncated"
        )
    support_summaries = support_resources.get("StackResourceSummaries")
    if type(support_summaries) is not list:
        raise Task12PostpublicationError(
            "support resource readback is absent"
        )
    support_physical: dict[str, str] = {}
    for row in support_summaries:
        if (
            type(row) is not dict
            or type(row.get("LogicalResourceId")) is not str
            or type(row.get("PhysicalResourceId")) is not str
            or row["LogicalResourceId"] in support_physical
        ):
            raise Task12PostpublicationError(
                "support resource readback is malformed"
            )
        support_physical[row["LogicalResourceId"]] = row[
            "PhysicalResourceId"
        ]
    for logical_id, pattern in (
        ("SupportDeadlineVersion", _LAMBDA_VERSION),
        ("SupportStateMachineVersion", _STATE_VERSION),
    ):
        value = support_physical.get(logical_id)
        if type(value) is not str or pattern.fullmatch(value) is None:
            raise Task12PostpublicationError(
                logical_id + " is not an exact support published version ARN"
            )
        physical[logical_id] = value
    return physical


def _reconcile_ddb(
    *, services: Task12PostpublicationServices, table: str, key: Mapping[str, str], expected: Mapping[str, object]
) -> bool:
    response = _authenticated(
        _method(services.dynamodb, "get_item")(
            TableName=table,
            Key=encode_item(key),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        ),
        "GetItem",
    )
    item = response.get("Item")
    if type(item) is not dict:
        return False
    value = decode_item(item)
    return value == {**key, **expected}


def _put_immutable_ddb(
    *, services: Task12PostpublicationServices, table: str, key: Mapping[str, str], value: Mapping[str, object]
) -> str:
    try:
        response = _method(services.dynamodb, "put_item")(
            TableName=table,
            Item=encode_item({**key, **value}),
            ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
            ReturnConsumedCapacity="NONE",
        )
        _authenticated(response, "PutItem")
        return "WRITTEN"
    except Exception:
        if _reconcile_ddb(
            services=services, table=table, key=key, expected=value
        ):
            return "RECONCILED"
        raise


def _authority_key(inputs: Task12PostpublicationInputs, edge_key: str, caller_version: str) -> str:
    qualifier = caller_version.rsplit(":", 1)[-1]
    return (
        f"campaigns/{RUN_ID}/task12/deployment/{inputs.activation_id}/"
        f"{inputs.generation:08d}/{edge_key}/{qualifier}/authority.json"
    )


def build_task12_postpublication_iam_policy(
    inputs: Task12PostpublicationInputs,
) -> Mapping[str, object]:
    """Return the exact least-privilege policy required by the CLI boundary."""

    if type(inputs) is not Task12PostpublicationInputs:
        raise TypeError("postpublication IAM policy requires exact typed inputs")
    table_arn = (
        f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/"
        + inputs.ledger_table_name
    )
    authority_prefix = (
        f"campaigns/{RUN_ID}/task12/deployment/{inputs.activation_id}/"
        f"{inputs.generation:08d}/"
    )
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadExactPublishedVersions",
                "Effect": "Allow",
                "Action": [
                    "cloudformation:DescribeStacks",
                    "cloudformation:ListStackResources",
                ],
                "Resource": [
                    inputs.retained_stack_id,
                    inputs.support_stack_id,
                ],
            },
            {
                "Sid": "MaterializeExactTask12Rows",
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:PutItem"],
                "Resource": table_arn,
                "Condition": {
                    "ForAllValues:StringEquals": {
                        "dynamodb:LeadingKeys": [PARTITION_KEY]
                    }
                },
            },
            {
                "Sid": "ReadExactVersionedOperationSource",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:GetObjectVersion"],
                "Resource": (
                    f"arn:aws:s3:::{inputs.operation_source.bucket}/"
                    + inputs.operation_source.key
                ),
                "Condition": {
                    "StringEquals": {
                        "s3:VersionId": inputs.operation_source.version_id
                    }
                },
            },
            {
                "Sid": "ListOnlyTask12AuthorityVersions",
                "Effect": "Allow",
                "Action": "s3:ListBucketVersions",
                "Resource": f"arn:aws:s3:::{inputs.authority_bucket}",
                "Condition": {
                    "StringLike": {"s3:prefix": authority_prefix + "*"}
                },
            },
            {
                "Sid": "WriteAndReconcileOnlyTask12Authority",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": (
                    f"arn:aws:s3:::{inputs.authority_bucket}/"
                    + authority_prefix
                    + "*"
                ),
            },
        ],
    }


def _read_exact_s3_version(
    *, services: Task12PostpublicationServices, bucket: str, key: str, raw: bytes
) -> tuple[str, str] | None:
    response = _authenticated(
        _method(services.s3, "list_object_versions")(
            Bucket=bucket,
            Prefix=key,
            MaxKeys=1000,
            ExpectedBucketOwner=ACCOUNT_ID,
        ),
        "ListObjectVersions",
    )
    if response.get("IsTruncated") is True:
        raise Task12PostpublicationError(
            "authority version reconciliation is truncated"
        )
    exact: list[tuple[str, str]] = []
    for row in response.get("Versions", []):
        if type(row) is not dict or row.get("Key") != key or type(row.get("VersionId")) is not str:
            continue
        version_id = row["VersionId"]
        read = _authenticated(
            _method(services.s3, "get_object")(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
                ExpectedBucketOwner=ACCOUNT_ID,
                ChecksumMode="ENABLED",
            ),
            "GetObject",
        )
        body = read.get("Body")
        candidate = body.read() if hasattr(body, "read") else None
        if (
            candidate == raw
            and read.get("VersionId") == version_id
            and read.get("ChecksumSHA256")
            == base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
        ):
            exact.append((version_id, hashlib.sha256(raw).hexdigest()))
    if len(exact) > 1:
        raise Task12PostpublicationError("authority publication has duplicate exact versions")
    return exact[0] if exact else None


def _put_immutable_authority(
    *, services: Task12PostpublicationServices, bucket: str, key: str, value: Mapping[str, object]
) -> tuple[str, str, str]:
    raw = canonical_json_bytes(value)
    try:
        response = _authenticated(
            _method(services.s3, "put_object")(
                Bucket=bucket,
                Key=key,
                Body=raw,
                ContentType="application/json",
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii"),
                IfNoneMatch="*",
                ExpectedBucketOwner=ACCOUNT_ID,
            ),
            "PutObject",
        )
        version_id = response.get("VersionId")
        if type(version_id) is not str or not version_id or version_id == "null":
            raise Task12PostpublicationError("authority bucket did not return an immutable version")
        return version_id, hashlib.sha256(raw).hexdigest(), "WRITTEN"
    except Exception:
        reconciled = _read_exact_s3_version(
            services=services, bucket=bucket, key=key, raw=raw
        )
        if reconciled is None:
            raise
        return reconciled[0], reconciled[1], "RECONCILED"


def materialize_task12_postpublication(
    *, inputs: Task12PostpublicationInputs, services: Task12PostpublicationServices
) -> Mapping[str, object]:
    """Materialize and reconcile 32 static descriptors and ten caller edges."""

    if (
        type(inputs) is not Task12PostpublicationInputs
        or type(services) is not Task12PostpublicationServices
        or _STACK_ID.fullmatch(inputs.retained_stack_id) is None
        or _STACK_ID.fullmatch(inputs.support_stack_id) is None
        or _ACTIVATION.fullmatch(inputs.activation_id) is None
        or type(inputs.activation_ordinal) is not int
        or inputs.activation_ordinal < 1
        or type(inputs.generation) is not int
        or inputs.generation < 1
        or _SHA.fullmatch(inputs.dispatch_identity_sha256) is None
        or _RESOURCE_NAME.fullmatch(inputs.ledger_table_name) is None
        or _S3_BUCKET.fullmatch(inputs.authority_bucket) is None
        or _S3_BUCKET.fullmatch(inputs.campaign_bucket) is None
        or _KMS_ARN.fullmatch(inputs.kms_key_id) is None
        or _RESOURCE_NAME.fullmatch(inputs.worker_drain_document_name) is None
        or re.fullmatch(
            r"[1-9][0-9]*", inputs.worker_drain_document_version
        )
        is None
    ):
        raise Task12PostpublicationError("postpublication inputs are not exact")
    (
        operation_rows,
        task9_coordinate,
        campaign_descriptor_coordinate,
        gpu_spend_approval_coordinate,
        task10_worker_descriptor_coordinate,
        worker_script_hashes,
        terminal_evidence_prefix,
    ) = _closed_operation_rows(inputs, services)
    physical = _read_versions(inputs, services)
    generation_text = f"{inputs.generation:08d}"
    operation_outcomes: dict[str, str] = {}
    for edge_key, kind, _mode, _caller in _EDGES:
        for operation in _OPERATIONS[edge_key]:
            if operation in operation_outcomes:
                continue
            key = {
                "PK": PARTITION_KEY,
                "SK": _state_key(inputs.activation_id, kind, operation, generation_text),
            }
            operation_outcomes[operation] = _put_immutable_ddb(
                services=services,
                table=inputs.ledger_table_name,
                key=key,
                value=operation_rows[operation],
            )

    deployment_outcomes: dict[str, str] = {}
    authority_outcomes: dict[str, str] = {}
    for edge_key, kind, mode, caller in _EDGES:
        function_logical = (
            "TerminalV2"
            if edge_key == "TerminalV2Support"
            else edge_key.removesuffix("Retained").removesuffix(
                "Snapshot"
            )
        )
        function_arn = physical[function_logical + "Version"]
        function_match = _LAMBDA_VERSION.fullmatch(function_arn)
        assert function_match is not None
        caller_version_arn = physical[caller + "StateMachineVersion"]
        caller_match = _STATE_VERSION.fullmatch(caller_version_arn)
        assert caller_match is not None
        caller_arn = caller_version_arn.rsplit(":", 1)[0]
        operations = {
            operation: {
                "state_sort_key": _state_key(
                    inputs.activation_id, kind, operation, generation_text
                ),
                "state_body_sha256": operation_rows[operation][
                    "canonical_body_sha256"
                ],
            }
            for operation in _OPERATIONS[edge_key]
        }
        authority_body = {
            "schema_version": 1,
            "record_type": "glm52_task12_lambda_invocation_authority_v1",
            "handler_kind": kind,
            "activation_id": inputs.activation_id,
            "activation_ordinal": inputs.activation_ordinal,
            "generation": inputs.generation,
            "generation_text": generation_text,
            "operations": operations,
        }
        authority = {
            **authority_body,
            "canonical_body_sha256": canonical_sha256(authority_body),
        }
        authority_key = _authority_key(inputs, edge_key, caller_version_arn)
        version_id, file_sha256, authority_outcome = _put_immutable_authority(
            services=services,
            bucket=inputs.authority_bucket,
            key=authority_key,
            value=authority,
        )
        authority_outcomes[edge_key] = authority_outcome
        role_coordinates: dict[str, object] = {
            "authority": {
                "bucket": inputs.authority_bucket,
                "key": authority_key,
                "version_id": version_id,
                "file_sha256": file_sha256,
            },
            "ledger_table_name": inputs.ledger_table_name,
        }
        if function_logical in {
            "ExecutionObserver",
            "OperatorDisposition",
            "WorkerDrain",
            "TerminalV2",
        }:
            role_coordinates.update(
                {
                    "numeric_binding_version_arn": physical[
                        "NumericBindingVersion"
                    ],
                    "task9_deployed_identity_coordinate": task9_coordinate,
                    "task9_deployed_identity_sha256": task9_coordinate[
                        "body_sha256"
                    ],
                }
            )
        if function_logical in {
            "ExecutionObserver",
            "WorkerDrain",
            "TerminalV2",
        }:
            role_coordinates.update(
                {
                    "campaign_bucket": inputs.campaign_bucket,
                    "spend_runtime_prefix": (
                        f"campaigns/{RUN_ID}/runtime"
                    ),
                    "campaign_descriptor_key": (
                        campaign_descriptor_coordinate["key"]
                    ),
                    "campaign_descriptor_version_id": (
                        campaign_descriptor_coordinate["version_id"]
                    ),
                    "campaign_descriptor_file_sha256": (
                        campaign_descriptor_coordinate["file_sha256"]
                    ),
                    "gpu_spend_approval_key": (
                        gpu_spend_approval_coordinate["key"]
                    ),
                    "gpu_spend_approval_version_id": (
                        gpu_spend_approval_coordinate["version_id"]
                    ),
                    "gpu_spend_approval_file_sha256": (
                        gpu_spend_approval_coordinate["file_sha256"]
                    ),
                }
            )
        if function_logical == "WorkerDrain":
            role_coordinates.update(
                {
                    "task10_worker_descriptor_coordinate": (
                        task10_worker_descriptor_coordinate
                    ),
                    "worker_script_hashes": worker_script_hashes,
                }
            )
        if function_logical == "TerminalV2":
            role_coordinates.update(
                {
                    "task10_worker_descriptor_coordinate": (
                        task10_worker_descriptor_coordinate
                    ),
                    "terminal_evidence_prefix": terminal_evidence_prefix,
                }
            )
        if edge_key == "TerminalV2Support":
            role_coordinates["support_observer_version_arn"] = physical[
                "SupportDeadlineVersion"
            ]
        if function_logical in {
            "ExecutionObserver",
            "TerminalV2",
            "Finalizer",
            "H1GDrained",
            "WorkerDrain",
            "SnapshotCleanup",
        }:
            role_coordinates["kms_key_id"] = inputs.kms_key_id
        if function_logical == "OperatorDisposition":
            role_coordinates["retained_cancellation_version_arn"] = physical[
                "RetainedCancellationVersion"
            ]
        if function_logical in {"TerminalV2", "Finalizer", "H1GDrained", "OperatorDisposition"}:
            role_coordinates["campaign_bucket"] = inputs.campaign_bucket
        elif function_logical == "WorkerDrain":
            role_coordinates.update(
                {
                    "campaign_bucket": inputs.campaign_bucket,
                    "spend_runtime_prefix": (
                        f"campaigns/{RUN_ID}/runtime"
                    ),
                    "worker_drain_document_name": inputs.worker_drain_document_name,
                    "worker_drain_document_version": inputs.worker_drain_document_version,
                }
            )
        elif function_logical == "OrphanAudit":
            role_coordinates["kms_key_id"] = inputs.kms_key_id
        elif function_logical == "SnapshotCleanup":
            schedule_role_name = physical["SnapshotCleanupScheduleInvokeRole"]
            role_coordinates.update(
                {
                    "snapshot_cleanup_state_machine_arn": physical[
                        "SnapshotCleanupStateMachineVersion"
                    ].rsplit(":", 1)[0],
                    "snapshot_cleanup_state_machine_version": physical[
                        "SnapshotCleanupStateMachineVersion"
                    ].rsplit(":", 1)[-1],
                    "snapshot_cleanup_schedule_invoke_role_arn": (
                        f"arn:aws:iam::{ACCOUNT_ID}:role/{schedule_role_name}"
                    ),
                    "snapshot_cleanup_schedule_group_name": "default",
                    "snapshot_cleanup_schedule_name": (
                        "keep-glm52-h1g-snapshot-cleanup-" + inputs.activation_id
                    ),
                }
            )
        if function_logical in {"H1GDrained", "OrphanAudit"}:
            role_coordinates["support_stack_id"] = "keep-glm52-h1g-support"
        deployment_body = {
            "schema_version": 1,
            "record_type": "glm52_task12_lambda_deployment_v1",
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "handler_kind": kind,
            "mode": mode,
            "activation_id": inputs.activation_id,
            "activation_ordinal": inputs.activation_ordinal,
            "generation": inputs.generation,
            "generation_text": generation_text,
            "function_name": function_match.group(1),
            "function_version": function_match.group(2),
            "invoked_function_version_arn": function_arn,
            "caller_state_machine_version_arn": caller_version_arn,
            "role_coordinates": role_coordinates,
        }
        deployment = {
            **deployment_body,
            "canonical_body_sha256": canonical_sha256(deployment_body),
        }
        deployment_key = {
            "PK": PARTITION_KEY,
            "SK": (
                DEPLOYMENT_PREFIX
                + function_arn
                + "#CALLER#"
                + caller_arn
                + "#ACTIVATION#"
                + inputs.activation_id
                + "#GENERATION#"
                + generation_text
            ),
        }
        deployment_outcomes[edge_key] = _put_immutable_ddb(
            services=services,
            table=inputs.ledger_table_name,
            key=deployment_key,
            value=deployment,
        )
    evidence_body = {
        "schema_version": 1,
        "record_type": "glm52_task12_postpublication_materialization_v1",
        "retained_stack_id": inputs.retained_stack_id,
        "support_stack_id": inputs.support_stack_id,
        "activation_id": inputs.activation_id,
        "generation": inputs.generation,
        "operation_record_count": len(operation_outcomes),
        "deployment_record_count": len(deployment_outcomes),
        "authority_manifest_count": len(authority_outcomes),
        "operation_source": {
            "bucket": inputs.operation_source.bucket,
            "key": inputs.operation_source.key,
            "version_id": inputs.operation_source.version_id,
            "file_sha256": inputs.operation_source.file_sha256,
        },
        "operation_outcomes": operation_outcomes,
        "deployment_outcomes": deployment_outcomes,
        "authority_outcomes": authority_outcomes,
    }
    return {**evidence_body, "canonical_body_sha256": canonical_sha256(evidence_body)}
