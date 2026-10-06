"""Production Lambda adapters for the eight retained Task 12 capabilities.

The Lambda envelope, deployed function version, versioned S3 authority, and
one retained DynamoDB input are all authenticated before a coordinator is
constructed.  AWS clients are created lazily and with one total attempt
(zero SDK retries).  The retained input is then materialized into the typed
contracts owned by the Task 12 runtime modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import base64
import hashlib
import json
import os
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .canonical import canonical_json_bytes, canonical_sha256
from .task12_operations import (
    OPERATION_ALLOWED_PREDECESSORS,
    OPERATION_PREDECESSORS,
    RETAINED_OPERATION_SEQUENCE,
    SNAPSHOT_OPERATION_SEQUENCE,
    Task12OperationContractError,
    build_operation_row,
    execute_operation_semantics,
    operation_spec,
    validate_operation_row,
    validate_operation_descriptor,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ACTIVATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_VERSION = re.compile(r"^[\x21-\x7e]{1,1024}$")
_LAMBDA_ARN = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:"
    r"([A-Za-z0-9_-]+):([1-9][0-9]*)$"
)
_STATE_MACHINE_ARN = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:stateMachine:"
    r"[A-Za-z0-9_-]+:[1-9][0-9]*$"
)
_STATE_MACHINE_UNQUALIFIED_ARN = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:stateMachine:"
    r"[A-Za-z0-9_-]+$"
)
_IAM_ROLE_ARN = re.compile(
    r"^arn:aws:iam::246813579024:role/[A-Za-z0-9_+=,.@/-]+$"
)
_EXECUTION_ARN = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:execution:"
    r"[A-Za-z0-9_-]+:[A-Za-z0-9._-]+$"
)
_KMS_ARN = re.compile(
    r"^arn:aws:kms:us-west-2:246813579024:key/"
    r"[0-9a-fA-F-]{36}$"
)
_RESOURCE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,255}$")
_S3_BUCKET = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)[a-z0-9]"
    r"(?:[a-z0-9.-]{1,61}[a-z0-9])?$"
)
_DEPLOYMENT_SORT_KEY_PREFIX = "TASK12_LAMBDA_DEPLOYMENT#"
_EVENT_FIELDS = frozenset(
    (
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "dispatch_identity_sha256",
        "caller_state_machine_arn",
        "state_machine_execution_arn",
        "operation_kind",
        "operation_input",
    )
)
_DEPLOYMENT_FIELDS = frozenset(
    (
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "handler_kind",
        "mode",
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "function_name",
        "function_version",
        "invoked_function_version_arn",
        "caller_state_machine_version_arn",
        "role_coordinates",
        "canonical_body_sha256",
    )
)
_AUTHORITY_FIELDS = frozenset(
    (
        "schema_version",
        "record_type",
        "handler_kind",
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "operations",
        "canonical_body_sha256",
    )
)
_STATE_COMMON_FIELDS = frozenset(
    (
        "schema_version",
        "record_type",
        "handler_kind",
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "operation_kind",
        "canonical_body_sha256",
    )
)


class Task12LambdaAdapterError(ValueError):
    """The Lambda deployment, authority, invocation, or retained input drifted."""


@dataclass(frozen=True)
class VersionedAuthorityCoordinate:
    bucket: str
    key: str
    version_id: str
    file_sha256: str


@dataclass(frozen=True)
class Task12LambdaDeployment:
    handler_kind: str
    mode: str
    activation_id: str
    activation_ordinal: int
    generation: int
    generation_text: str
    function_name: str
    function_version: str
    invoked_function_version_arn: str
    caller_state_machine_version_arn: str
    authority: VersionedAuthorityCoordinate
    role_coordinates: Mapping[str, object]
    canonical_body_sha256: str


@dataclass(frozen=True)
class Task12LambdaInvocation:
    handler_kind: str
    mode: str
    activation_id: str
    activation_ordinal: int
    generation: int
    generation_text: str
    dispatch_identity_sha256: str
    deployment_identity_sha256: str
    invoked_function_version_arn: str
    caller_state_machine_arn: str
    caller_state_machine_version_arn: str
    state_machine_execution_arn: str
    operation_kind: str
    operation_input: Mapping[str, object]
    authenticated_execution: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class _HandlerSpec:
    kind: str
    mode: str
    function_name: str
    coordinate_fields: Tuple[str, ...]
    state_record_type: str


_SPECS: Mapping[str, _HandlerSpec] = {
    "RETAINED_EXECUTION_OBSERVER": _HandlerSpec(
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
        "glm52_task12_retained_execution_observer_input_v1",
    ),
    "RETAINED_TERMINAL_V2": _HandlerSpec(
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
        "glm52_task12_retained_terminal_v2_input_v1",
    ),
    "RETAINED_FINALIZER": _HandlerSpec(
        "RETAINED_FINALIZER",
        "RETAINED",
        "keep-glm52-h1g-finalizer",
        ("authority", "campaign_bucket", "ledger_table_name"),
        "glm52_task12_retained_finalizer_input_v1",
    ),
    "RETAINED_H1G_DRAINED": _HandlerSpec(
        "RETAINED_H1G_DRAINED",
        "RETAINED",
        "keep-glm52-h1g-h1g-drained-writer",
        ("authority", "campaign_bucket", "ledger_table_name"),
        "glm52_task12_retained_h1g_drained_input_v1",
    ),
    "RETAINED_WORKER_DRAIN": _HandlerSpec(
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
        "glm52_task12_retained_worker_drain_input_v1",
    ),
    "RETAINED_OPERATOR_DISPOSITION": _HandlerSpec(
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
        "glm52_task12_retained_operator_disposition_input_v1",
    ),
    "RETAINED_ORPHAN_AUDIT": _HandlerSpec(
        "RETAINED_ORPHAN_AUDIT",
        "RETAINED",
        "keep-glm52-h1g-orphan-audit",
        ("authority", "kms_key_id", "ledger_table_name"),
        "glm52_task12_retained_orphan_audit_input_v1",
    ),
    "RETAINED_SNAPSHOT_CLEANUP": _HandlerSpec(
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
        "glm52_task12_retained_snapshot_cleanup_input_v1",
    ),
}

_OPERATIONS: Mapping[str, Tuple[str, ...]] = {
    "RETAINED_EXECUTION_OBSERVER": (
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
    "RETAINED_OPERATOR_DISPOSITION": (
        "RETAINED_CORRELATE_REQUESTS_AND_JOBS",
        "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION",
    ),
    "RETAINED_WORKER_DRAIN": (
        "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
        "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
        "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
    ),
    "RETAINED_TERMINAL_V2": (
        "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
        "RETAINED_ENTER_RECOVERY_COMPLETE",
    ),
    "RETAINED_FINALIZER": (
        "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED",
        "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
    ),
    "RETAINED_ORPHAN_AUDIT": (
        "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT",
        "RETAINED_AUDIT_SUPPORT_ORPHANS",
    ),
    "RETAINED_SNAPSHOT_CLEANUP": (
        "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",
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
    "RETAINED_H1G_DRAINED": (
        "RETAINED_INVOKE_H1G_DRAINED_WRITER",
    ),
}

_RETAINED_OPERATION_SEQUENCE = RETAINED_OPERATION_SEQUENCE
_SNAPSHOT_OPERATION_SEQUENCE = SNAPSHOT_OPERATION_SEQUENCE
_OPERATION_PREDECESSOR = OPERATION_PREDECESSORS
_OPERATION_ALLOWED_PREDECESSORS = OPERATION_ALLOWED_PREDECESSORS


def _caller_edge_operations(
    handler_kind: str, caller_state_machine_version_arn: str
) -> Tuple[str, ...]:
    operations = _OPERATIONS[handler_kind]
    if handler_kind != "RETAINED_SNAPSHOT_CLEANUP":
        return operations
    state_machine_name = caller_state_machine_version_arn.split(
        ":stateMachine:", 1
    )[-1].rsplit(":", 1)[0]
    normalized = state_machine_name.replace("-", "").lower()
    if normalized.endswith(("retained", "retainedlifecycle")):
        return ("RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",)
    if normalized.endswith("snapshotcleanup"):
        return SNAPSHOT_OPERATION_SEQUENCE
    _fail("snapshot cleanup caller edge is not closed")
    raise AssertionError("unreachable")


def _fail(message: str) -> None:
    raise Task12LambdaAdapterError(message)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail(label + " must be a lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(label + " must be a nonempty exact string")
    return value


def _authenticated_metadata(value: object) -> bool:
    return (
        type(value) is dict
        and {"HTTPStatusCode", "RequestId", "RetryAttempts"}.issubset(value)
        and type(value.get("HTTPStatusCode")) is int
        and value.get("HTTPStatusCode") == 200
        and type(value.get("RequestId")) is str
        and bool(value.get("RequestId"))
        and type(value.get("RetryAttempts")) is int
        and value.get("RetryAttempts") == 0
    )


def _positive(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail(label + " must be a positive integer")
    return value


def _canonical_mapping(raw: bytes, label: str) -> Dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Task12LambdaAdapterError(label + " is not canonical JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        _fail(label + " must be an exact canonical JSON object")
    return value


def _validate_operation_input(
    value: object,
    *,
    operation_kind: str,
) -> Dict[str, object]:
    if type(value) is not dict:
        _fail("operation input must be one exact JSON object")
    prior = value.get("task12_last_result")
    allowed_predecessors = _OPERATION_ALLOWED_PREDECESSORS.get(
        operation_kind, frozenset()
    )
    if not allowed_predecessors:
        if prior is not None:
            _fail("operation input prior ResultPath is foreign")
        return dict(value)
    ambiguity = value.get("delete_ambiguity")
    if ambiguity is not None:
        if operation_kind != "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE":
            _fail("snapshot delete ambiguity ResultPath is foreign")
        if (
            type(ambiguity) is not dict
            or set(ambiguity) != {"Error", "Cause"}
            or type(ambiguity.get("Error")) is not str
            or not ambiguity.get("Error")
            or type(ambiguity.get("Cause")) is not str
            or not ambiguity.get("Cause")
        ):
            _fail("snapshot delete ambiguity ResultPath is malformed")
    if (
        type(prior) is not dict
        or set(prior)
        != {
            "schema_version",
            "record_type",
            "handler_kind",
            "operation_kind",
            "operation_input_identity_sha256",
            "outcome",
            "result",
            "canonical_body_sha256",
        }
        or prior.get("schema_version") != 1
        or prior.get("record_type") != "glm52_task12_lambda_result_v1"
        or prior.get("outcome") != "SUCCEEDED"
        or prior.get("operation_kind") not in allowed_predecessors
    ):
        _fail("operation input prior ResultPath is foreign")
    if (
        ambiguity is not None
        and prior.get("operation_kind")
        != "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT"
    ):
        _fail("snapshot delete ambiguity predecessor is foreign")
    _self_hash(prior, "operation input prior ResultPath")
    return dict(value)


def _self_hash(value: Mapping[str, object], label: str) -> str:
    expected = _sha(value.get("canonical_body_sha256"), label + " identity")
    body = dict(value)
    body.pop("canonical_body_sha256", None)
    if expected != canonical_sha256(body):
        _fail(label + " self-hash drifted")
    return expected


def _authority_coordinate(value: object) -> VersionedAuthorityCoordinate:
    if type(value) is not dict or set(value) != {
        "bucket",
        "key",
        "version_id",
        "file_sha256",
    }:
        _fail("deployment authority coordinate is not closed")
    bucket = _text(value["bucket"], "authority bucket")
    key = _text(value["key"], "authority key")
    version_id = _text(value["version_id"], "authority version")
    if _VERSION.fullmatch(version_id) is None:
        _fail("authority version is invalid")
    if (
        _S3_BUCKET.fullmatch(bucket) is None
        or key.startswith("/")
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        _fail("authority S3 coordinate is invalid")
    return VersionedAuthorityCoordinate(
        bucket=bucket,
        key=key,
        version_id=version_id,
        file_sha256=_sha(value["file_sha256"], "authority file"),
    )


def _zero_retry_client(
    service: str, *, session: Optional[object] = None
) -> object:
    try:
        from botocore.config import Config

        if session is None:
            import boto3

            exact_session = boto3.session.Session(region_name=REGION)
        else:
            exact_session = session
        return exact_session.client(
            service,
            region_name=REGION,
            config=Config(
                region_name=REGION,
                retries={"mode": "standard", "total_max_attempts": 1},
            ),
        )
    except Exception as exc:
        raise Task12LambdaAdapterError(
            "zero-retry " + service + " client creation failed"
        ) from exc


def _load_deployment(
    *,
    environ: Mapping[str, str],
    expected_kind: str,
    event: Optional[object] = None,
    context: Optional[object] = None,
    session: Optional[object] = None,
    allow_test_config: bool = True,
) -> Task12LambdaDeployment:
    inline = environ.get("GLM52_TASK12_DEPLOYMENT_CONFIG")
    path_text = environ.get("GLM52_TASK12_DEPLOYMENT_CONFIG_PATH")
    if inline is not None or path_text is not None:
        if not allow_test_config or (inline is None) == (path_text is None):
            _fail("injected deployment config source is not exact")
        if inline is not None:
            raw = inline.encode("utf-8")
        else:
            assert path_text is not None
            path = Path(path_text)
            try:
                if path.is_symlink() or not path.is_file():
                    _fail("deployment config path must be a regular non-symlink")
                raw = path.read_bytes()
            except OSError as exc:
                raise Task12LambdaAdapterError(
                    "deployment config path is unreadable"
                ) from exc
    else:
        from .dynamodb import decode_item, encode_item

        if type(event) is not dict or set(event) != _EVENT_FIELDS:
            _fail("deployment lookup requires the exact invocation envelope")
        context_arn = getattr(context, "invoked_function_arn", None)
        activation_id = event.get("activation_id")
        generation_text = event.get("generation_text")
        caller_arn = event.get("caller_state_machine_arn")
        table_name = environ.get("GLM52_TASK12_DEPLOYMENT_TABLE_NAME")
        partition_key = environ.get(
            "GLM52_TASK12_DEPLOYMENT_PARTITION_KEY"
        )
        prefix = environ.get("GLM52_TASK12_DEPLOYMENT_SORT_KEY_PREFIX")
        if (
            type(context_arn) is not str
            or _LAMBDA_ARN.fullmatch(context_arn) is None
            or type(caller_arn) is not str
            or _STATE_MACHINE_UNQUALIFIED_ARN.fullmatch(caller_arn) is None
            or type(activation_id) is not str
            or _ACTIVATION.fullmatch(activation_id) is None
            or type(generation_text) is not str
            or re.fullmatch(r"[0-9]{8}", generation_text) is None
            or type(table_name) is not str
            or _RESOURCE_NAME.fullmatch(table_name) is None
            or type(partition_key) is not str
            or not partition_key
            or prefix != _DEPLOYMENT_SORT_KEY_PREFIX
        ):
            _fail("deployment authority lookup coordinates are invalid")
        sort_key = (
            prefix
            + context_arn
            + "#CALLER#"
            + caller_arn
            + "#ACTIVATION#"
            + activation_id
            + "#GENERATION#"
            + generation_text
        )
        try:
            response = _zero_retry_client(
                "dynamodb", session=session
            ).get_item(
                TableName=table_name,
                Key=encode_item({"PK": partition_key, "SK": sort_key}),
                ConsistentRead=True,
                ReturnConsumedCapacity="NONE",
            )
        except Exception as exc:
            raise Task12LambdaAdapterError(
                "deployment authority read failed"
            ) from exc
        metadata = response.get("ResponseMetadata")
        if (
            not _authenticated_metadata(metadata)
            or type(response.get("Item")) is not dict
        ):
            _fail("deployment authority transport is unauthenticated")
        value = decode_item(response["Item"])
        if (
            value.pop("PK", None) != partition_key
            or value.pop("SK", None) != sort_key
        ):
            _fail("deployment authority key drifted")
        raw = canonical_json_bytes(value)
    value = _canonical_mapping(raw, "deployment config")
    if set(value) != _DEPLOYMENT_FIELDS:
        _fail("deployment config fields are not closed")
    identity = _self_hash(value, "deployment config")
    spec = _SPECS.get(expected_kind)
    if spec is None:
        _fail("handler kind is unknown")
    generation = _positive(value["generation"], "deployment generation")
    activation_ordinal = _positive(
        value["activation_ordinal"], "deployment activation ordinal"
    )
    coordinates = value["role_coordinates"]
    if type(coordinates) is not dict or set(coordinates) != set(
        spec.coordinate_fields
    ):
        _fail("deployment role coordinates are not closed for handler")
    authority = _authority_coordinate(coordinates.get("authority"))
    for name, coordinate in coordinates.items():
        if name == "authority":
            continue
        if name == "task9_deployed_identity_coordinate":
            try:
                from .task11_support_boundary import (
                    exact_input_coordinate_from_mapping,
                )

                exact_task9 = exact_input_coordinate_from_mapping(
                    coordinate,
                    expected_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
                )
            except (TypeError, ValueError) as exc:
                raise Task12LambdaAdapterError(
                    "Task 9 deployed identity coordinate is invalid"
                ) from exc
            if (
                exact_task9.body_sha256
                != coordinates.get("task9_deployed_identity_sha256")
            ):
                _fail("Task 9 deployed identity hash drifted")
            continue
        if name == "task10_worker_descriptor_coordinate":
            if (
                type(coordinate) is not dict
                or set(coordinate)
                != {
                    "bucket",
                    "key",
                    "version_id",
                    "file_sha256",
                    "body_sha256",
                }
                or coordinate.get("bucket")
                != coordinates.get("campaign_bucket")
                or coordinate.get("key")
                != "task13/production/task10-worker-descriptor.json"
                or type(coordinate.get("version_id")) is not str
                or not coordinate["version_id"]
                or coordinate["version_id"] == "null"
                or _VERSION.fullmatch(coordinate["version_id"]) is None
                or type(coordinate.get("file_sha256")) is not str
                or _SHA.fullmatch(coordinate["file_sha256"]) is None
                or type(coordinate.get("body_sha256")) is not str
                or _SHA.fullmatch(coordinate["body_sha256"]) is None
            ):
                _fail("Task 10 worker descriptor coordinate is invalid")
            continue
        if name == "worker_script_hashes":
            if (
                type(coordinate) is not dict
                or set(coordinate)
                != {
                    "glm52_checkpoint_commit.py",
                    "glm52_drain_and_stop.py",
                    "glm52_deadline_guard.py",
                }
                or any(
                    type(item) is not str or _SHA.fullmatch(item) is None
                    for item in coordinate.values()
                )
            ):
                _fail("worker script hashes are invalid")
            continue
        exact = _text(coordinate, "role coordinate " + name)
        if name == "campaign_bucket" and _S3_BUCKET.fullmatch(exact) is None:
            _fail("campaign bucket coordinate is invalid")
        if name == "spend_runtime_prefix" and exact != (
            f"campaigns/{RUN_ID}/runtime"
        ):
            _fail("spend runtime prefix is not the fixed run namespace")
        if name == "campaign_descriptor_key" and exact != (
            f"campaigns/{RUN_ID}/submissions/production/descriptor.json"
        ):
            _fail("campaign descriptor key is not exact")
        if name == "gpu_spend_approval_key" and exact != (
            f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json"
        ):
            _fail("GPU spend approval key is not exact")
        if name == "terminal_evidence_prefix" and exact != (
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{generation:08d}/terminal-evidence/"
        ):
            _fail("TerminalV2 evidence prefix is not exact")
        if name in {
            "campaign_descriptor_version_id",
            "gpu_spend_approval_version_id",
        } and (exact == "null" or _VERSION.fullmatch(exact) is None):
            _fail(name + " is not an exact object version")
        if name in {
            "campaign_descriptor_file_sha256",
            "gpu_spend_approval_file_sha256",
        } and _SHA.fullmatch(exact) is None:
            _fail(name + " is not an exact file SHA-256")
        if name in {"ledger_table_name", "worker_drain_document_name"} and (
            _RESOURCE_NAME.fullmatch(exact) is None
        ):
            _fail(name + " coordinate is invalid")
        if name == "worker_drain_document_version" and (
            re.fullmatch(r"[1-9][0-9]*", exact) is None
        ):
            _fail("worker drain document version is not published")
        if name in {
            "numeric_binding_version_arn",
            "retained_cancellation_version_arn",
        } and _LAMBDA_ARN.fullmatch(exact) is None:
            _fail(name + " is not an exact published Lambda version")
        if name == "state_machine_execution_arn" and (
            _EXECUTION_ARN.fullmatch(exact) is None
        ):
            _fail("state-machine execution coordinate is invalid")
        if name == "kms_key_id" and _KMS_ARN.fullmatch(exact) is None:
            _fail("KMS key coordinate is invalid")
        if name == "snapshot_cleanup_state_machine_arn" and (
            _STATE_MACHINE_UNQUALIFIED_ARN.fullmatch(exact) is None
        ):
            _fail("snapshot cleanup state-machine coordinate is invalid")
        if name == "snapshot_cleanup_state_machine_version" and (
            re.fullmatch(r"[1-9][0-9]*", exact) is None
        ):
            _fail("snapshot cleanup state-machine version is not published")
        if name == "snapshot_cleanup_schedule_invoke_role_arn" and (
            _IAM_ROLE_ARN.fullmatch(exact) is None
        ):
            _fail("snapshot cleanup schedule role is invalid")
        if name in {
            "snapshot_cleanup_schedule_group_name",
            "snapshot_cleanup_schedule_name",
        } and _RESOURCE_NAME.fullmatch(exact) is None:
            _fail(name + " coordinate is invalid")
    arn = value["invoked_function_version_arn"]
    match = _LAMBDA_ARN.fullmatch(arn) if type(arn) is str else None
    if (
        value["schema_version"] != 1
        or value["record_type"] != "glm52_task12_lambda_deployment_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["handler_kind"] != spec.kind
        or value["mode"] != spec.mode
        or value["function_name"] != spec.function_name
        or match is None
        or match.group(1) != value["function_name"]
        or match.group(2) != value["function_version"]
        or value["generation_text"] != "{0:08d}".format(generation)
        or type(value["activation_id"]) is not str
        or _ACTIVATION.fullmatch(value["activation_id"]) is None
        or type(value["caller_state_machine_version_arn"]) is not str
        or _STATE_MACHINE_ARN.fullmatch(
            value["caller_state_machine_version_arn"]
        )
        is None
    ):
        _fail("deployment config identity is invalid")
    return Task12LambdaDeployment(
        handler_kind=spec.kind,
        mode=spec.mode,
        activation_id=value["activation_id"],
        activation_ordinal=activation_ordinal,
        generation=generation,
        generation_text=value["generation_text"],
        function_name=value["function_name"],
        function_version=value["function_version"],
        invoked_function_version_arn=arn,
        caller_state_machine_version_arn=value[
            "caller_state_machine_version_arn"
        ],
        authority=authority,
        role_coordinates=dict(coordinates),
        canonical_body_sha256=identity,
    )


def _parse_invocation(
    *,
    event: object,
    context: object,
    deployment: Task12LambdaDeployment,
) -> Task12LambdaInvocation:
    if type(event) is not dict or set(event) != _EVENT_FIELDS:
        _fail("Lambda event must have the exact closed envelope")
    context_arn = getattr(context, "invoked_function_arn", None)
    execution_arn = event["state_machine_execution_arn"]
    execution_match = (
        _EXECUTION_ARN.fullmatch(execution_arn)
        if type(execution_arn) is str
        else None
    )
    caller_arn = event["caller_state_machine_arn"]
    caller_name = (
        caller_arn.split(":stateMachine:", 1)[1]
        if type(caller_arn) is str
        and _STATE_MACHINE_UNQUALIFIED_ARN.fullmatch(caller_arn) is not None
        else None
    )
    deployed_caller_arn = deployment.caller_state_machine_version_arn
    deployed_caller_base = deployed_caller_arn.rsplit(":", 1)[0]
    operation_kind = event["operation_kind"]
    operation_input = event["operation_input"]
    execution_name = (
        execution_arn.split(":execution:", 1)[1].split(":", 1)[0]
        if execution_match is not None
        else None
    )
    if (
        event["activation_id"] != deployment.activation_id
        or event["activation_ordinal"] != deployment.activation_ordinal
        or event["generation"] != deployment.generation
        or event["generation_text"] != deployment.generation_text
        or context_arn != deployment.invoked_function_version_arn
        or caller_arn != deployed_caller_base
        or execution_match is None
        or execution_name != caller_name
        or operation_kind not in _caller_edge_operations(
            deployment.handler_kind,
            deployment.caller_state_machine_version_arn,
        )
        or type(operation_input) is not dict
    ):
        _fail("Lambda invocation does not match deployed authority")
    return Task12LambdaInvocation(
        handler_kind=deployment.handler_kind,
        mode=deployment.mode,
        activation_id=event["activation_id"],
        activation_ordinal=_positive(
            event["activation_ordinal"], "event activation ordinal"
        ),
        generation=_positive(event["generation"], "event generation"),
        generation_text=event["generation_text"],
        dispatch_identity_sha256=_sha(
            event["dispatch_identity_sha256"], "dispatch identity"
        ),
        deployment_identity_sha256=deployment.canonical_body_sha256,
        invoked_function_version_arn=deployment.invoked_function_version_arn,
        caller_state_machine_arn=caller_arn,
        caller_state_machine_version_arn=deployed_caller_arn,
        state_machine_execution_arn=execution_arn,
        operation_kind=operation_kind,
        operation_input=_validate_operation_input(
            operation_input,
            operation_kind=operation_kind,
        ),
    )


def _result(
    *, handler_kind: str, outcome: str,
    operation_kind: Optional[str] = None, result: Optional[object] = None,
    operation_input: Optional[Mapping[str, object]] = None,
    error_code: Optional[str] = None
) -> Dict[str, object]:
    body: Dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_result_v1",
        "handler_kind": handler_kind,
        "outcome": outcome,
    }
    if operation_kind is not None:
        body["operation_kind"] = operation_kind
    if operation_input is not None:
        body["operation_input_identity_sha256"] = canonical_sha256(
            operation_input
        )
    if outcome == "SUCCEEDED":
        body["result"] = _json_value(result)
    else:
        body["error_code"] = _text(error_code, "Lambda error code")
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def build_lambda_handler(
    *,
    handler_kind: str,
    coordinator_factory: Callable[
        [Task12LambdaDeployment],
        Callable[[Task12LambdaInvocation], object],
    ],
) -> Callable[..., Dict[str, object]]:
    """Build one Lambda-compatible ``(event, context)`` named adapter."""

    if handler_kind not in _SPECS or not callable(coordinator_factory):
        _fail("Lambda adapter declaration is invalid")

    def main(
        event: object,
        context: object,
        *,
        _environ: Optional[Mapping[str, str]] = None,
        _coordinator_factory: Optional[Callable[[Task12LambdaDeployment], object]] = None,
        _deployment_session: Optional[object] = None,
        _execution_authenticator: Optional[
            Callable[
                [Task12LambdaDeployment, Task12LambdaInvocation],
                Mapping[str, object],
            ]
        ] = None,
    ) -> Dict[str, object]:
        invocation: Optional[Task12LambdaInvocation] = None
        try:
            deployment = _load_deployment(
                environ=os.environ if _environ is None else _environ,
                expected_kind=handler_kind,
                event=event,
                context=context,
                session=_deployment_session,
                allow_test_config=_environ is not None,
            )
            invocation = _parse_invocation(
                event=event,
                context=context,
                deployment=deployment,
            )
            if _execution_authenticator is None:
                execution_evidence = _observe_exact_execution(
                    _ports(deployment, _deployment_session),
                    invocation,
                )
            else:
                execution_evidence = _execution_authenticator(
                    deployment,
                    invocation,
                )
            invocation = replace(
                invocation,
                authenticated_execution=_validated_execution_evidence(
                    execution_evidence,
                    deployment=deployment,
                    invocation=invocation,
                ),
            )
        except Exception as exc:
            return _result(
                handler_kind=handler_kind,
                outcome="REJECTED",
                error_code=type(exc).__name__,
            )
        factory = (
            coordinator_factory
            if _coordinator_factory is None
            else _coordinator_factory
        )
        try:
            coordinator = factory(deployment)
            if not callable(coordinator):
                _fail("coordinator factory did not return one callable")
            coordinated = coordinator(invocation)
        except Exception as exc:
            return _result(
                handler_kind=handler_kind,
                outcome="FAILED",
                operation_kind=invocation.operation_kind,
                operation_input=invocation.operation_input,
                error_code=type(exc).__name__,
            )
        return _result(
            handler_kind=handler_kind,
            outcome="SUCCEEDED",
            operation_kind=invocation.operation_kind,
            operation_input=invocation.operation_input,
            result=coordinated,
        )

    return main


def _state_sort_key(invocation: Task12LambdaInvocation) -> str:
    return (
        "ACTIVATION#"
        + invocation.activation_id
        + "#TASK12_LAMBDA_INPUT#"
        + invocation.handler_kind
        + "#"
        + invocation.operation_kind
        + "#"
        + invocation.generation_text
    )


class _AwsPorts:
    def __init__(
        self,
        *,
        deployment: Task12LambdaDeployment,
        session: Optional[object] = None,
    ) -> None:
        self.deployment = deployment
        self._session = session
        self._clients: Dict[str, object] = {}

    def client(self, service: str) -> object:
        if service in self._clients:
            return self._clients[service]
        try:
            client = _zero_retry_client(service, session=self._session)
        except Exception as exc:
            raise Task12LambdaAdapterError(
                "zero-retry " + service + " client creation failed"
            ) from exc
        self._clients[service] = client
        return client

    def retained_input(
        self, invocation: Task12LambdaInvocation
    ) -> Dict[str, object]:
        authority = self._read_authority(invocation)
        return self._read_state(invocation, authority)

    def _read_authority(
        self, invocation: Task12LambdaInvocation
    ) -> Mapping[str, object]:
        coordinate = self.deployment.authority
        try:
            response = self.client("s3").get_object(
                Bucket=coordinate.bucket,
                Key=coordinate.key,
                VersionId=coordinate.version_id,
                ExpectedBucketOwner=ACCOUNT_ID,
            )
            raw = response["Body"].read()
        except Exception as exc:
            raise Task12LambdaAdapterError(
                "versioned coordinator authority read failed"
            ) from exc
        metadata = response.get("ResponseMetadata")
        if (
            type(raw) is not bytes
            or response.get("VersionId") != coordinate.version_id
            or not _authenticated_metadata(metadata)
            or hashlib.sha256(raw).hexdigest() != coordinate.file_sha256
        ):
            _fail("versioned coordinator authority transport drifted")
        value = _canonical_mapping(raw, "coordinator authority")
        if set(value) != _AUTHORITY_FIELDS:
            _fail("coordinator authority fields are not closed")
        _self_hash(value, "coordinator authority")
        expected_key = _state_sort_key(invocation)
        expected_operations = _caller_edge_operations(
            invocation.handler_kind,
            invocation.caller_state_machine_version_arn,
        )
        operations = value.get("operations")
        if (
            value["schema_version"] != 1
            or value["record_type"]
            != "glm52_task12_lambda_invocation_authority_v1"
            or value["handler_kind"] != invocation.handler_kind
            or value["activation_id"] != invocation.activation_id
            or value["activation_ordinal"] != invocation.activation_ordinal
            or value["generation"] != invocation.generation
            or value["generation_text"] != invocation.generation_text
            or type(operations) is not dict
            or set(operations) != set(expected_operations)
        ):
            _fail("coordinator authority is foreign to invocation")
        for operation_kind, entry in operations.items():
            if (
                type(entry) is not dict
                or set(entry) != {"state_sort_key", "state_body_sha256"}
                or entry["state_sort_key"]
                != (
                    "ACTIVATION#"
                    + invocation.activation_id
                    + "#TASK12_LAMBDA_INPUT#"
                    + invocation.handler_kind
                    + "#"
                    + operation_kind
                    + "#"
                    + invocation.generation_text
                )
            ):
                _fail("coordinator authority operation manifest drifted")
            _sha(entry["state_body_sha256"], "retained input identity")
        selected = operations[invocation.operation_kind]
        if selected["state_sort_key"] != expected_key:
            _fail("coordinator authority operation is foreign to invocation")
        return selected

    def _read_state(
        self,
        invocation: Task12LambdaInvocation,
        authority: Mapping[str, object],
    ) -> Dict[str, object]:
        from .dynamodb import decode_item, encode_item
        from .records import ledger_pk

        table_name = _text(
            self.deployment.role_coordinates.get("ledger_table_name"),
            "ledger table",
        )
        key = {
            "PK": ledger_pk(RUN_ID),
            "SK": authority["state_sort_key"],
        }
        try:
            response = self.client("dynamodb").get_item(
                TableName=table_name,
                Key=encode_item(key),
                ConsistentRead=True,
                ReturnConsumedCapacity="NONE",
            )
        except Exception as exc:
            raise Task12LambdaAdapterError(
                "retained coordinator input read failed"
            ) from exc
        metadata = response.get("ResponseMetadata")
        if (
            not _authenticated_metadata(metadata)
            or type(response.get("Item")) is not dict
        ):
            _fail("retained coordinator input is unavailable")
        value = decode_item(response["Item"])
        if value.pop("PK", None) != key["PK"] or value.pop("SK", None) != key["SK"]:
            _fail("retained coordinator input key drifted")
        spec = _SPECS[invocation.handler_kind]
        identity = _self_hash(value, "retained coordinator input")
        operation_contract = operation_spec(invocation.operation_kind)
        is_descriptor = value.get("record_type") == (
            "glm52_task12_"
            + invocation.operation_kind.lower()
            + "_descriptor_v1"
        )
        if is_descriptor:
            try:
                validate_operation_descriptor(value)
            except Task12OperationContractError as exc:
                raise Task12LambdaAdapterError(
                    "retained operation descriptor drifted"
                ) from exc
        is_operation_row = value.get("record_type") == operation_contract.record_type
        if is_operation_row:
            try:
                validate_operation_row(value)
            except Task12OperationContractError as exc:
                raise Task12LambdaAdapterError(
                    "retained operation input schema drifted"
                ) from exc
        if (
            value.get("schema_version") != 1
            or (
                not is_operation_row
                and not is_descriptor
                and value.get("record_type") != spec.state_record_type
            )
            or value.get("handler_kind") != invocation.handler_kind
            or value.get("operation_kind") != invocation.operation_kind
            or value.get("activation_id") != invocation.activation_id
            or value.get("activation_ordinal") != invocation.activation_ordinal
            or value.get("generation") != invocation.generation
            or value.get("generation_text") != invocation.generation_text
            or identity != authority["state_body_sha256"]
        ):
            _fail("retained coordinator input is foreign to invocation")
        return value


T = TypeVar("T")


def _materialize_dataclass(cls: Type[T], value: object) -> T:
    if not is_dataclass(cls) or type(value) is not dict:
        _fail("retained typed value is not an exact dataclass mapping")
    expected = {field.name for field in fields(cls)}
    if set(value) != expected:
        _fail(cls.__name__ + " fields are not closed")
    hints = get_type_hints(cls)
    arguments = {
        field.name: _coerce(hints.get(field.name, Any), value[field.name])
        for field in fields(cls)
    }
    if cls.__name__ == "GpuLiabilityReserveResult":
        arguments["record"] = _restore_decimal_record(arguments["record"])
    try:
        return cls(**arguments)
    except Exception as exc:
        raise Task12LambdaAdapterError(
            cls.__name__ + " materialization failed"
        ) from exc


def _restore_decimal_record(value: object) -> object:
    if type(value) is not dict:
        _fail("liability reserve record is not an exact mapping")
    restored = dict(value)
    for name in (
        "gpu_reserve_cost_usd",
        "root_volume_tail_usd_max",
        "remaining_gpu_cost_usd",
    ):
        if name in restored and type(restored[name]) is str:
            try:
                restored[name] = Decimal(restored[name])
            except InvalidOperation as exc:
                raise Task12LambdaAdapterError(
                    "liability reserve decimal is invalid"
                ) from exc
    return restored


def _coerce(annotation: object, value: object) -> object:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Union:
        if value is None and type(None) in arguments:
            return None
        choices = tuple(item for item in arguments if item is not type(None))
        for choice in choices:
            try:
                return _coerce(choice, value)
            except (Task12LambdaAdapterError, TypeError, ValueError):
                pass
        _fail("retained union value has no exact member")
    if origin in (tuple, Tuple):
        if type(value) not in (list, tuple):
            _fail("retained tuple value is not a sequence")
        item_type = arguments[0] if arguments else Any
        return tuple(_coerce(item_type, item) for item in value)
    if origin in (dict, Dict, Mapping):
        if type(value) is not dict:
            _fail("retained mapping value is not exact")
        key_type, value_type = arguments if len(arguments) == 2 else (Any, Any)
        return {
            _coerce(key_type, key): _coerce(value_type, item)
            for key, item in value.items()
        }
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _materialize_dataclass(annotation, value)
    if annotation in (str, int, bool, bytes):
        if annotation is bytes and type(value) is str:
            try:
                return bytes.fromhex(value)
            except ValueError as exc:
                raise Task12LambdaAdapterError(
                    "retained bytes are not canonical hex"
                ) from exc
        if type(value) is not annotation:
            _fail("retained scalar has the wrong exact type")
    if annotation is Decimal:
        if type(value) is not str:
            _fail("retained decimal must be a canonical string")
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise Task12LambdaAdapterError(
                "retained decimal is invalid"
            ) from exc
    return value


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Decimal):
        if not value.is_finite():
            _fail("Lambda result decimal must be finite")
        return format(value, "f")
    return value


def _closed_state(
    state: Mapping[str, object],
    *,
    extra_fields: Tuple[str, ...],
) -> None:
    if set(state) != _STATE_COMMON_FIELDS | frozenset(extra_fields):
        _fail("retained coordinator input fields are not closed")


class _StaticRuntimeReader:
    def __init__(self, scan: object) -> None:
        self._scan = scan

    def read_requests(self, _identity: str) -> object:
        return self._scan.requests

    def read_jobs(self, _identity: str) -> object:
        return self._scan.jobs

    def read_controller(self, _identity: str) -> object:
        return self._scan.controller

    def read_workers(self, _identity: str) -> object:
        return self._scan.workers

    def read_allocations(self, _identity: str) -> object:
        return self._scan.allocations

    def read_spend(self, _identity: str) -> object:
        return self._scan.spend


class _RetainedWriteBoundary:
    def __init__(self, ports: _AwsPorts) -> None:
        self._ports = ports

    @staticmethod
    def _s3_candidate(candidate: object) -> object:
        """Rebuild the closed immutable-S3 candidate, including metadata."""

        from .s3_records import build_immutable_json_candidate

        coordinate = getattr(candidate, "coordinate", None)
        if type(coordinate) is not str or not coordinate.startswith("s3://"):
            _fail("retained writer S3 coordinate is malformed")
        bucket_key = coordinate[5:].split("/", 1)
        if len(bucket_key) != 2 or not all(bucket_key):
            _fail("retained writer S3 coordinate is malformed")
        record_kind = {
            "TerminalV2": "terminal-v2",
            "SupportPlaneFinalized": "support-plane-finalized",
            "H1GDrained": "h1g-drained",
            "RecoveryHandoff": "recovery-handoff",
        }.get(getattr(candidate, "writer_kind", None))
        if record_kind is None:
            _fail("retained writer has no immutable-S3 record kind")
        return build_immutable_json_candidate(
            record_kind=record_kind,
            bucket=bucket_key[0],
            key=bucket_key[1],
            raw=candidate.raw,
            activation_id=candidate.activation_id,
            generation=candidate.generation,
        )

    def conditional_create(self, *, candidate: object, action: object, audit: object) -> object:
        del action, audit
        from .task12_writers import ConditionalCreateResponse

        coordinate = candidate.coordinate
        if coordinate.startswith("dynamodb://"):
            return self._conditional_dynamodb_create(candidate)
        if not coordinate.startswith("s3://"):
            _fail("retained writer coordinate is not closed")
        exact = self._s3_candidate(candidate)
        checksum = base64.b64encode(
            hashlib.sha256(exact.raw).digest()
        ).decode("ascii")
        try:
            response = self._ports.client("s3").put_object(
                Bucket=exact.bucket,
                Key=exact.key,
                Body=exact.raw,
                IfNoneMatch="*",
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=checksum,
                ContentType=exact.content_type,
                Metadata=dict(exact.metadata),
                ExpectedBucketOwner=ACCOUNT_ID,
            )
        except Exception:
            return None
        metadata = response.get("ResponseMetadata")
        request_id = metadata.get("RequestId") if type(metadata) is dict else None
        version_id = response.get("VersionId")
        authenticated = (
            _authenticated_metadata(metadata)
            and type(request_id) is str
            and bool(request_id)
            and type(version_id) is str
            and bool(version_id)
            and response.get("ChecksumSHA256") in {None, checksum}
        )
        response_body = {
            "request_id": request_id,
            "version_id": version_id,
            "candidate_identity_sha256": candidate.candidate_identity_sha256,
        }
        return ConditionalCreateResponse(
            classification="CREATED" if authenticated else "AMBIGUOUS",
            candidate_identity_sha256=candidate.candidate_identity_sha256,
            request_id=request_id or "unavailable",
            response_identity_sha256=canonical_sha256(response_body),
            version_id=version_id,
            authenticated=authenticated,
        )

    def _conditional_dynamodb_create(self, candidate: object) -> object:
        from .dynamodb import encode_item
        from .task12_writers import ConditionalCreateResponse

        table_name = _text(
            self._ports.deployment.role_coordinates.get("ledger_table_name"),
            "ledger table",
        )
        coordinate = candidate.coordinate[len("dynamodb://") :]
        pieces = coordinate.split("/", 1)
        if len(pieces) != 2 or not all(pieces):
            _fail("retained writer DynamoDB coordinate is malformed")
        item = {"PK": pieces[0], "SK": pieces[1], **dict(candidate.record)}
        try:
            response = self._ports.client("dynamodb").put_item(
                TableName=table_name,
                Item=encode_item(item),
                ConditionExpression=(
                    "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
                ),
                ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
                ReturnConsumedCapacity="NONE",
            )
        except Exception:
            return None
        metadata = response.get("ResponseMetadata")
        request_id = metadata.get("RequestId") if type(metadata) is dict else None
        authenticated = (
            _authenticated_metadata(metadata)
            and type(request_id) is str
            and bool(request_id)
        )
        response_body = {
            "request_id": request_id,
            "candidate_identity_sha256": candidate.candidate_identity_sha256,
        }
        return ConditionalCreateResponse(
            classification="CREATED" if authenticated else "AMBIGUOUS",
            candidate_identity_sha256=candidate.candidate_identity_sha256,
            request_id=request_id or "unavailable",
            response_identity_sha256=canonical_sha256(response_body),
            version_id=request_id if authenticated else None,
            authenticated=authenticated,
        )

    def reconcile_exact(self, *, candidate: object) -> object:
        from .task12_writers import ExactCandidateReconciliation

        if candidate.coordinate.startswith("dynamodb://"):
            return self._reconcile_dynamodb(candidate)
        from .h1f_adapter import reconcile_exact_candidate

        exact = self._s3_candidate(candidate)
        reconciliation = reconcile_exact_candidate(
            s3=self._ports.client("s3"),
            candidate=exact,
        )
        identity = reconciliation.object_identity
        if reconciliation.state != "sole-version" or identity is None:
            _fail("retained writer reconciliation found no sole exact version")
        return ExactCandidateReconciliation(
            state="SOLE_EXACT_CANDIDATE",
            candidate_identity_sha256=candidate.candidate_identity_sha256,
            file_sha256=identity.file_sha256,
            raw=exact.raw,
            version_id=identity.version_id,
            request_ids=(
                "reconciled:" + identity.canonical_identity_sha256,
            ),
            authenticated=True,
        )

    def _reconcile_dynamodb(self, candidate: object) -> object:
        from .dynamodb import decode_item, encode_item
        from .task12_writers import ExactCandidateReconciliation

        table_name = _text(
            self._ports.deployment.role_coordinates.get("ledger_table_name"),
            "ledger table",
        )
        coordinate = candidate.coordinate[len("dynamodb://") :]
        pieces = coordinate.split("/", 1)
        response = self._ports.client("dynamodb").get_item(
            TableName=table_name,
            Key=encode_item({"PK": pieces[0], "SK": pieces[1]}),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        metadata = response.get("ResponseMetadata")
        request_id = metadata.get("RequestId") if type(metadata) is dict else None
        item = decode_item(response.get("Item"))
        if item.pop("PK", None) != pieces[0] or item.pop("SK", None) != pieces[1]:
            _fail("retained DynamoDB reconciliation key drifted")
        raw = canonical_json_bytes(item) + b"\n"
        return ExactCandidateReconciliation(
            state="SOLE_EXACT_CANDIDATE",
            candidate_identity_sha256=candidate.candidate_identity_sha256,
            file_sha256=hashlib.sha256(raw).hexdigest(),
            raw=raw,
            version_id=_text(request_id, "DynamoDB reconciliation request"),
            request_ids=(request_id,),
            authenticated=_authenticated_metadata(metadata),
        )


class _SnapshotClient:
    def __init__(self, ports: _AwsPorts) -> None:
        self._ports = ports

    def delete_snapshot(self, *, snapshot_id: str) -> Mapping[str, object]:
        response = self._ports.client("ec2").delete_snapshot(
            SnapshotId=snapshot_id
        )
        metadata = response.get("ResponseMetadata")
        if not _authenticated_metadata(metadata):
            _fail("snapshot delete response is unauthenticated")
        request_id = _text(metadata.get("RequestId"), "snapshot delete request")
        body = {"snapshot_id": snapshot_id, "request_id": request_id}
        return {
            "outcome": "ACCEPTED",
            "request_id": request_id,
            "response_sha256": canonical_sha256(body),
        }

    def describe_snapshot(self, *, snapshot_id: str) -> object:
        from .task12_snapshot_cleanup import SnapshotObservation

        try:
            response = self._ports.client("ec2").describe_snapshots(
                SnapshotIds=[snapshot_id]
            )
        except Exception as exc:
            error = getattr(exc, "response", None)
            detail = error.get("Error") if type(error) is dict else None
            metadata = (
                error.get("ResponseMetadata")
                if type(error) is dict
                else None
            )
            if (
                type(detail) is dict
                and detail.get("Code") == "InvalidSnapshot.NotFound"
                and snapshot_id in str(detail.get("Message", ""))
                and type(metadata) is dict
                and metadata.get("HTTPStatusCode") == 400
                and type(metadata.get("RequestId")) is str
                and bool(metadata["RequestId"])
                and metadata.get("RetryAttempts") == 0
            ):
                return None
            raise
        metadata = response.get("ResponseMetadata")
        snapshots = response.get("Snapshots")
        if (
            not _authenticated_metadata(metadata)
            or type(snapshots) is not list
        ):
            _fail("snapshot describe response is unauthenticated")
        if not snapshots:
            return None
        if len(snapshots) != 1 or type(snapshots[0]) is not dict:
            _fail("snapshot describe response is not singular")
        item = snapshots[0]
        request_id = _text(metadata.get("RequestId"), "snapshot describe request")
        tags = sorted(
            (
                {"Key": tag.get("Key"), "Value": tag.get("Value")}
                for tag in item.get("Tags", [])
            ),
            key=lambda tag: (str(tag["Key"]), str(tag["Value"])),
        )
        return SnapshotObservation(
            snapshot_id=item.get("SnapshotId"),
            source_volume_id=item.get("VolumeId"),
            kms_key_arn=item.get("KmsKeyId"),
            snapshot_tags_sha256=canonical_sha256(tags),
            encrypted=item.get("Encrypted"),
            state=item.get("State"),
            observed_at=item.get("StartTime").strftime("%Y-%m-%dT%H:%M:%SZ"),
            describe_request_id=request_id,
            describe_response_sha256=canonical_sha256(
                {
                    "snapshot_id": item.get("SnapshotId"),
                    "source_volume_id": item.get("VolumeId"),
                    "kms_key_arn": item.get("KmsKeyId"),
                    "snapshot_tags_sha256": canonical_sha256(tags),
                    "encrypted": item.get("Encrypted"),
                    "state": item.get("State"),
                    "observed_at": item.get("StartTime").strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "request_id": request_id,
                }
            ),
        )


class _AuthenticatedDynamoClient:
    def __init__(self, client: object) -> None:
        self._client = client

    def __getattr__(self, name: str) -> object:
        boundary = getattr(self._client, name)
        if not callable(boundary):
            return boundary

        def authenticated_call(*args: object, **kwargs: object) -> object:
            response = boundary(*args, **kwargs)
            if (
                type(response) is not dict
                or not _authenticated_metadata(
                    response.get("ResponseMetadata")
                )
            ):
                _fail("DynamoDB " + name + " response is unauthenticated")
            return response

        return authenticated_call


class _SchedulerClient:
    def __init__(self, ports: _AwsPorts) -> None:
        self._ports = ports
        coordinates = ports.deployment.role_coordinates
        self._state_machine_arn = _text(
            coordinates.get("snapshot_cleanup_state_machine_arn"),
            "snapshot cleanup state machine",
        )
        self._state_machine_version = _text(
            coordinates.get("snapshot_cleanup_state_machine_version"),
            "snapshot cleanup state-machine version",
        )
        self._target_arn = (
            self._state_machine_arn + ":" + self._state_machine_version
        )
        if _STATE_MACHINE_ARN.fullmatch(self._target_arn) is None:
            _fail("snapshot cleanup target version is not exact")
        self._role_arn = _text(
            coordinates.get("snapshot_cleanup_schedule_invoke_role_arn"),
            "snapshot cleanup schedule role",
        )
        self._group_name = _text(
            coordinates.get("snapshot_cleanup_schedule_group_name"),
            "snapshot cleanup schedule group",
        )
        self._name = _text(
            coordinates.get("snapshot_cleanup_schedule_name"),
            "snapshot cleanup schedule name",
        )
        self._schedule_arn = (
            f"arn:aws:scheduler:{REGION}:{ACCOUNT_ID}:schedule/"
            + self._group_name
            + "/"
            + self._name
        )

    @staticmethod
    def _request_id(response: Mapping[str, object], label: str) -> str:
        metadata = response.get("ResponseMetadata")
        if not _authenticated_metadata(metadata):
            _fail(label + " response is unauthenticated")
        return metadata["RequestId"]

    def _request(self, schedule: object, client_token: str) -> Dict[str, object]:
        from .task12_snapshot_cleanup import SnapshotSchedule

        if type(schedule) is not SnapshotSchedule:
            _fail("snapshot cleanup schedule must be typed")
        if (
            schedule.schedule_arn != self._schedule_arn
            or schedule.schedule_input_sha256 != canonical_sha256({})
        ):
            _fail("snapshot cleanup schedule lost deployed authority")
        try:
            deadline = datetime.strptime(
                schedule.delete_not_before, "%Y-%m-%dT%H:%M:%SZ"
            )
        except ValueError as exc:
            raise Task12LambdaAdapterError(
                "snapshot cleanup deadline is not canonical UTC"
            ) from exc
        return {
            "Name": self._name,
            "GroupName": self._group_name,
            "ScheduleExpression": (
                "at(" + deadline.strftime("%Y-%m-%dT%H:%M:%S") + ")"
            ),
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "ActionAfterCompletion": "DELETE",
            "State": "ENABLED",
            "Target": {
                "Arn": self._target_arn,
                "RoleArn": self._role_arn,
                "Input": "{}",
                "RetryPolicy": {
                    "MaximumEventAgeInSeconds": 60,
                    "MaximumRetryAttempts": 0,
                },
            },
            "ClientToken": _sha(
                client_token, "snapshot cleanup schedule operation"
            ),
        }

    def _reconcile(
        self, expected: Mapping[str, object]
    ) -> Mapping[str, object]:
        response = self._ports.client("scheduler").get_schedule(
            Name=self._name,
            GroupName=self._group_name,
        )
        request_id = self._request_id(response, "scheduler GetSchedule")
        expected_fields = {
            "Arn": self._schedule_arn,
            "Name": self._name,
            "GroupName": self._group_name,
            "ScheduleExpression": expected["ScheduleExpression"],
            "FlexibleTimeWindow": expected["FlexibleTimeWindow"],
            "ActionAfterCompletion": "DELETE",
            "State": "ENABLED",
            "Target": expected["Target"],
        }
        if any(response.get(key) != value for key, value in expected_fields.items()):
            _fail("existing snapshot cleanup schedule is foreign")
        body = {
            "outcome": "RECONCILED",
            "schedule_arn": self._schedule_arn,
            "request_id": request_id,
        }
        return {**body, "identity_sha256": canonical_sha256(body)}

    def arm(
        self, *, schedule: object, operation_identity_sha256: str
    ) -> Mapping[str, object]:
        request = self._request(schedule, operation_identity_sha256)
        try:
            response = self._ports.client("scheduler").create_schedule(
                **request
            )
        except Exception:
            return self._reconcile(request)
        try:
            request_id = self._request_id(
                response, "scheduler CreateSchedule"
            )
            if response.get("ScheduleArn") != self._schedule_arn:
                _fail("created snapshot cleanup schedule ARN drifted")
        except Task12LambdaAdapterError:
            return self._reconcile(request)
        body = {
            "outcome": "CREATED",
            "schedule_arn": self._schedule_arn,
            "request_id": request_id,
        }
        return {**body, "identity_sha256": canonical_sha256(body)}

    def reconcile(
        self, *, schedule: object, operation_identity_sha256: str
    ) -> Mapping[str, object]:
        return self._reconcile(
            self._request(schedule, operation_identity_sha256)
        )

    @staticmethod
    def _not_found(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        error = response.get("Error") if type(response) is dict else None
        return (
            type(error) is dict
            and error.get("Code") == "ResourceNotFoundException"
        )

    def delete(self, *, schedule: object) -> Mapping[str, object]:
        expected = self._request(schedule, "0" * 64)
        try:
            self._reconcile(expected)
        except Exception as exc:
            if self._not_found(exc):
                body = {
                    "outcome": "ABSENT",
                    "schedule_arn": self._schedule_arn,
                }
                return {
                    **body,
                    "identity_sha256": canonical_sha256(body),
                }
            raise
        response = self._ports.client("scheduler").delete_schedule(
            Name=self._name,
            GroupName=self._group_name,
            ClientToken=canonical_sha256(
                {"schedule_arn": self._schedule_arn, "operation": "DELETE"}
            ),
        )
        request_id = self._request_id(
            response, "scheduler DeleteSchedule"
        )
        body = {
            "outcome": "DELETED",
            "schedule_arn": self._schedule_arn,
            "request_id": request_id,
        }
        return {**body, "identity_sha256": canonical_sha256(body)}


def _observe_exact_execution(
    ports: _AwsPorts, invocation: Task12LambdaInvocation
) -> Mapping[str, object]:
    execution_arn = _text(
        invocation.state_machine_execution_arn,
        "state-machine execution",
    )
    try:
        response = ports.client("stepfunctions").describe_execution(
            executionArn=execution_arn
        )
    except Exception as exc:
        raise Task12LambdaAdapterError(
            "live state-machine execution read failed"
        ) from exc
    metadata = response.get("ResponseMetadata")
    if (
        response.get("executionArn") != execution_arn
        or response.get("stateMachineArn")
        != invocation.caller_state_machine_arn
        or response.get("stateMachineVersionArn")
        != ports.deployment.caller_state_machine_version_arn
        or response.get("status")
        not in {
            "RUNNING",
            "SUCCEEDED",
            "FAILED",
            "TIMED_OUT",
            "ABORTED",
            "PENDING_REDRIVE",
        }
        or not _authenticated_metadata(metadata)
    ):
        _fail("live state-machine execution evidence is unauthenticated")
    return _validated_execution_evidence(
        {
            "execution_arn": execution_arn,
            "state_machine_arn": invocation.caller_state_machine_arn,
            "state_machine_version_arn": (
                ports.deployment.caller_state_machine_version_arn
            ),
            "status": response["status"],
            "request_id": metadata["RequestId"],
        },
        deployment=ports.deployment,
        invocation=invocation,
    )


def _validated_execution_evidence(
    value: object,
    *,
    deployment: Task12LambdaDeployment,
    invocation: Task12LambdaInvocation,
) -> Mapping[str, object]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "execution_arn",
            "state_machine_arn",
            "state_machine_version_arn",
            "status",
            "request_id",
        }
        or value["execution_arn"] != invocation.state_machine_execution_arn
        or value["state_machine_arn"] != invocation.caller_state_machine_arn
        or value["state_machine_version_arn"]
        != deployment.caller_state_machine_version_arn
        or value["status"]
        not in {
            "RUNNING",
            "SUCCEEDED",
            "FAILED",
            "TIMED_OUT",
            "ABORTED",
            "PENDING_REDRIVE",
        }
        or type(value["request_id"]) is not str
        or not value["request_id"]
    ):
        _fail("authenticated execution evidence is invalid")
    return dict(value)


def _read_live_kms_grants(ports: _AwsPorts) -> object:
    from .task12_orphan_audit import KmsGrantIdentity, KmsGrantScan

    key_id = _text(
        ports.deployment.role_coordinates.get("kms_key_id"),
        "KMS key",
    )
    grants: list[KmsGrantIdentity] = []
    request_ids: list[str] = []
    marker: Optional[str] = None
    seen_markers: set[str] = set()
    while True:
        request: Dict[str, object] = {"KeyId": key_id, "Limit": 100}
        if marker is not None:
            request["Marker"] = marker
        try:
            response = ports.client("kms").list_grants(**request)
        except Exception as exc:
            raise Task12LambdaAdapterError(
                "live KMS ListGrants read failed"
            ) from exc
        metadata = response.get("ResponseMetadata")
        raw_grants = response.get("Grants")
        truncated = response.get("Truncated")
        if (
            not _authenticated_metadata(metadata)
            or type(raw_grants) is not list
            or type(truncated) is not bool
        ):
            _fail("live KMS ListGrants transport is unauthenticated")
        request_ids.append(metadata["RequestId"])
        for raw_grant in raw_grants:
            if type(raw_grant) is not dict:
                _fail("live KMS grant is malformed")
            operations = raw_grant.get("Operations")
            constraints = raw_grant.get("Constraints", {})
            context = (
                constraints.get("EncryptionContextSubset", {})
                if type(constraints) is dict
                else None
            )
            if (
                type(operations) is not list
                or not operations
                or not all(type(item) is str and item for item in operations)
                or type(context) is not dict
                or any(
                    type(key) is not str
                    or not key
                    or type(item) is not str
                    or not item
                    for key, item in context.items()
                )
            ):
                _fail("live KMS grant authority is malformed")
            context_order = {
                "RunId": 0,
                "ActivationId": 1,
            }
            normalized_context = tuple(
                sorted(
                    context.items(),
                    key=lambda item: (
                        context_order.get(item[0], 2),
                        item[0],
                    ),
                )
            )
            grants.append(
                KmsGrantIdentity(
                    grant_id=_text(raw_grant.get("GrantId"), "KMS grant ID"),
                    grant_name=_text(
                        raw_grant.get("Name"), "KMS grant name"
                    ),
                    grantee_principal=_text(
                        raw_grant.get("GranteePrincipal"),
                        "KMS grant grantee",
                    ),
                    retiring_principal=_text(
                        raw_grant.get("RetiringPrincipal"),
                        "KMS grant retiring principal",
                    ),
                    operations=tuple(sorted(set(operations))),
                    encryption_context_identity_sha256=canonical_sha256(
                        normalized_context if context else {}
                    ),
                )
            )
        if not truncated:
            break
        next_marker = response.get("NextMarker")
        if (
            type(next_marker) is not str
            or not next_marker
            or next_marker in seen_markers
        ):
            _fail("live KMS ListGrants pagination is not exact")
        seen_markers.add(next_marker)
        marker = next_marker
    ordered = tuple(sorted(grants, key=lambda item: item.grant_id))
    if len({item.grant_id for item in ordered}) != len(ordered):
        _fail("live KMS ListGrants contains duplicate grant IDs")
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    identity = canonical_sha256(
        {
            "key_id": key_id,
            "grants": [asdict(item) for item in ordered],
            "observed_at": observed_at,
            "request_ids": request_ids,
        }
    )
    return KmsGrantScan(
        grants=ordered,
        pagination_complete=True,
        observed_at=observed_at,
        list_grants_identity_sha256=identity,
    )


def _ports(
    deployment: Task12LambdaDeployment, session: Optional[object]
) -> _AwsPorts:
    return _AwsPorts(deployment=deployment, session=session)


def _exact_operation_request(
    state: Mapping[str, object],
    invocation: Task12LambdaInvocation,
) -> Dict[str, object]:
    spec = operation_spec(invocation.operation_kind)
    key = spec.action_kind + "_request"
    value = state.get(key)
    if type(value) is not dict:
        _fail(invocation.operation_kind + " request is not one exact object")
    return dict(value)


def _exact_fields(
    value: Mapping[str, object], fields_: Tuple[str, ...], label: str
) -> Dict[str, object]:
    if type(value) is not dict or set(value) != set(fields_):
        _fail(label + " fields are not closed")
    return dict(value)


def _operation_ledger(ports: _AwsPorts) -> object:
    from .dynamodb import DynamoLedgerAdapter

    return DynamoLedgerAdapter(
        client=_AuthenticatedDynamoClient(ports.client("dynamodb")),
        table_name=_text(
            ports.deployment.role_coordinates.get("ledger_table_name"),
            "ledger table",
        ),
    )


def _commit_operation_plan(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    request: Mapping[str, object],
    plan_type: Type[object],
    method_name: str,
) -> object:
    exact = _exact_fields(
        request,
        ("plan", "domain", "operation_identity_sha256", "owner_nonce_capsule"),
        method_name,
    )
    plan = _materialize_dataclass(plan_type, exact["plan"])
    nonce = _decrypt_operation_nonce(
        ports=ports,
        invocation=invocation,
        capsule=exact["owner_nonce_capsule"],
        plan=plan,
        domain=_text(exact["domain"], method_name + " domain"),
    )
    method = getattr(_operation_ledger(ports), method_name, None)
    if not callable(method):
        _fail(method_name + " boundary is unavailable")
    result = method(
        plan=plan,
        domain=_text(exact["domain"], method_name + " domain"),
        operation_identity_sha256=_sha(
            exact["operation_identity_sha256"], method_name + " identity"
        ),
        raw_owner_nonce=nonce,
    )
    from .dynamodb import TransactionResolution

    if type(result) is not TransactionResolution:
        _fail(method_name + " returned an unauthenticated transaction")
    return result


def _closed_evidence_action(
    request: Mapping[str, object], *, action_kind: str
) -> Mapping[str, object]:
    """Narrow boundary for evidence-only operations with no external mutation."""

    exact = _exact_fields(
        request,
        ("action_kind", "evidence", "evidence_identity_sha256"),
        action_kind,
    )
    evidence = exact["evidence"]
    if (
        exact["action_kind"] != action_kind
        or type(evidence) is not dict
        or exact["evidence_identity_sha256"] != canonical_sha256(evidence)
    ):
        _fail(action_kind + " evidence is unauthenticated")
    return {
        "action_kind": action_kind,
        "evidence_identity_sha256": exact["evidence_identity_sha256"],
        "evidence": evidence,
    }


def _read_exact_ledger_authority(
    *,
    ports: _AwsPorts,
    request: Mapping[str, object],
    label: str,
) -> Mapping[str, object]:
    from .dynamodb import decode_item, encode_item

    exact = _exact_fields(
        request,
        ("partition_key", "sort_key", "expected_record_identity_sha256"),
        label,
    )
    partition_key = _text(exact["partition_key"], label + " partition key")
    sort_key = _text(exact["sort_key"], label + " sort key")
    expected = _sha(
        exact["expected_record_identity_sha256"], label + " record identity"
    )
    response = ports.client("dynamodb").get_item(
        TableName=_text(
            ports.deployment.role_coordinates.get("ledger_table_name"),
            "ledger table",
        ),
        Key=encode_item({"PK": partition_key, "SK": sort_key}),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    if (
        not _authenticated_metadata(response.get("ResponseMetadata"))
        or type(response.get("Item")) is not dict
    ):
        _fail(label + " live read is unauthenticated")
    value = decode_item(response["Item"])
    if (
        value.pop("PK", None) != partition_key
        or value.pop("SK", None) != sort_key
        or _self_hash(value, label + " live record") != expected
    ):
        _fail(label + " live record drifted")
    return value


def _correlate_and_persist_request_jobs(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    request: Mapping[str, object],
) -> Mapping[str, object]:
    """Observe exact Sky request/job sets and publish their canonical handoff."""

    from .dynamodb import decode_item, encode_item
    from .records import (
        canonical_record_identity,
        ledger_pk,
        ledger_sk,
        validate_record,
    )
    from .task12_live_cancellation import _numeric_observation

    execution = _read_exact_ledger_authority(
        ports=ports,
        request=request,
        label="request and job correlation source",
    )
    source_identity = canonical_record_identity(
        "glm52_production_execution", execution
    )
    observation = _numeric_observation(
        ports=ports,
        invocation=invocation,
        correlation_identity_sha256=source_identity,
    )
    request_snapshot = observation.get("request_snapshot")
    job_snapshot = observation.get("job_snapshot")
    requests = (
        request_snapshot.get("requests")
        if type(request_snapshot) is dict
        else None
    )
    jobs = (
        job_snapshot.get("jobs")
        if type(job_snapshot) is dict
        else None
    )
    if (
        type(requests) is not list
        or not requests
        or any(type(item) is not dict for item in requests)
        or type(jobs) is not list
        or any(type(item) is not dict for item in jobs)
        or request_snapshot.get("observed_at") != observation.get("observed_at")
        or job_snapshot.get("observed_at") != observation.get("observed_at")
    ):
        _fail("request/job runtime correlation is incomplete")
    request_ids = [item.get("request_id") for item in requests]
    job_ids = [item.get("job_id") for item in jobs]
    if (
        any(type(item) is not str or not item for item in request_ids)
        or request_ids != sorted(request_ids)
        or len(request_ids) != len(set(request_ids))
        or any(
            type(item) is not str
            or re.fullmatch(r"[1-9][0-9]*", item) is None
            for item in job_ids
        )
        or job_ids != sorted(job_ids, key=int)
        or len(job_ids) != len(set(job_ids))
        or any(item.get("request_id") not in request_ids for item in jobs)
    ):
        _fail("request/job runtime correlation cardinality drifted")
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_request_job_correlation_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": execution["campaign_identity_sha256"],
        "activation_id": invocation.activation_id,
        "activation_ordinal": invocation.activation_ordinal,
        "generation": invocation.generation,
        "generation_text": invocation.generation_text,
        "request_ids": request_ids,
        "job_ids": job_ids,
        "request_states": {
            item["request_id"]: _text(item.get("state"), "request state")
            for item in requests
        },
        "job_states": {
            item["job_id"]: _text(item.get("state"), "job state")
            for item in jobs
        },
        "request_evidence_identity_sha256": canonical_sha256(request_snapshot),
        "job_evidence_identity_sha256": canonical_sha256(job_snapshot),
        "correlation_source_identity_sha256": source_identity,
        "observation_identity_sha256": observation[
            "canonical_identity_sha256"
        ],
        "observed_at": observation["observed_at"],
    }
    record = validate_record(
        "glm52_task12_request_job_correlation_v1",
        {**body, "canonical_body_sha256": canonical_sha256(body)},
    )
    partition_key = ledger_pk(RUN_ID)
    sort_key = ledger_sk(
        "glm52_task12_request_job_correlation_v1",
        activation_id=invocation.activation_id,
    )
    table_name = _text(
        ports.deployment.role_coordinates.get("ledger_table_name"),
        "ledger table",
    )
    client = ports.client("dynamodb")
    try:
        response = client.put_item(
            TableName=table_name,
            Item=encode_item(
                {"PK": partition_key, "SK": sort_key, **record}
            ),
            ConditionExpression=(
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            ),
            ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
            ReturnConsumedCapacity="NONE",
        )
        if not _authenticated_metadata(response.get("ResponseMetadata")):
            _fail("request/job correlation write is unauthenticated")
    except Exception:
        readback = client.get_item(
            TableName=table_name,
            Key=encode_item({"PK": partition_key, "SK": sort_key}),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        if (
            not _authenticated_metadata(readback.get("ResponseMetadata"))
            or type(readback.get("Item")) is not dict
            or decode_item(readback["Item"])
            != {"PK": partition_key, "SK": sort_key, **record}
        ):
            _fail("request/job correlation adopted foreign bytes")
    return record


def _observe_support_execution(
    *, ports: _AwsPorts, request: Mapping[str, object]
) -> Mapping[str, object]:
    exact = _exact_fields(
        request,
        (
            "execution_arn",
            "expected_state_machine_version_arn",
            "stop_if_running",
        ),
        "support execution",
    )
    execution_arn = _text(exact["execution_arn"], "support execution ARN")
    expected_version = _text(
        exact["expected_state_machine_version_arn"],
        "support state-machine version",
    )
    if type(exact["stop_if_running"]) is not bool:
        _fail("support stop_if_running must be exact boolean")
    client = ports.client("stepfunctions")
    response = client.describe_execution(executionArn=execution_arn)
    if (
        not _authenticated_metadata(response.get("ResponseMetadata"))
        or response.get("executionArn") != execution_arn
        or response.get("stateMachineVersionArn") != expected_version
        or response.get("status")
        not in {"RUNNING", "SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
    ):
        _fail("support execution readback is unauthenticated")
    stop_request_id: Optional[str] = None
    if response["status"] == "RUNNING" and exact["stop_if_running"]:
        stopped = client.stop_execution(
            executionArn=execution_arn,
            error="Task12RetainedRecovery",
            cause="Task 12 retained recovery owns the terminal path",
        )
        metadata = stopped.get("ResponseMetadata")
        if not _authenticated_metadata(metadata):
            _fail("support execution stop is unauthenticated")
        stop_request_id = metadata["RequestId"]
    body = {
        "execution_arn": execution_arn,
        "state_machine_version_arn": expected_version,
        "observed_status": response["status"],
        "stop_requested": stop_request_id is not None,
        "describe_request_id": response["ResponseMetadata"]["RequestId"],
        "stop_request_id": stop_request_id,
    }
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def _observe_owner_action(
    *, ports: _AwsPorts, request: Mapping[str, object]
) -> Mapping[str, object]:
    exact = _exact_fields(
        request,
        (
            "owner_execution_arn",
            "expected_state_machine_version_arn",
            "expected_action",
        ),
        "owner-dead Sky action",
    )
    execution_arn = _text(
        exact["owner_execution_arn"], "owner execution ARN"
    )
    response = ports.client("stepfunctions").describe_execution(
        executionArn=execution_arn
    )
    status = response.get("status")
    if (
        not _authenticated_metadata(response.get("ResponseMetadata"))
        or response.get("executionArn") != execution_arn
        or response.get("stateMachineVersionArn")
        != exact["expected_state_machine_version_arn"]
        or status
        not in {"RUNNING", "SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
    ):
        _fail("owner execution readback is unauthenticated")
    action = "OBSERVE_OWNER" if status == "RUNNING" else "RECOVER_OWNER_DEAD"
    if exact["expected_action"] != action:
        _fail("owner-dead Sky action classification drifted")
    body = {
        "owner_execution_arn": execution_arn,
        "owner_execution_status": status,
        "action": action,
        "request_id": response["ResponseMetadata"]["RequestId"],
    }
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def _observe_stack_absence(
    *, ports: _AwsPorts, request: Mapping[str, object]
) -> Mapping[str, object]:
    exact = _exact_fields(request, ("stack_id",), "support deletion")
    stack_id = _text(exact["stack_id"], "support stack ID")
    client = ports.client("cloudformation")
    try:
        response = client.describe_stacks(StackName=stack_id)
    except Exception as exc:
        response = getattr(exc, "response", None)
        metadata = response.get("ResponseMetadata") if type(response) is dict else None
        code = (
            response.get("Error", {}).get("Code")
            if type(response) is dict
            else None
        )
        if code != "ValidationError" or not _authenticated_metadata(metadata):
            raise
        body = {
            "stack_id": stack_id,
            "absent": True,
            "status": "ABSENT",
            "request_id": metadata["RequestId"],
        }
        return {**body, "canonical_body_sha256": canonical_sha256(body)}
    if not _authenticated_metadata(response.get("ResponseMetadata")):
        _fail("support stack readback is unauthenticated")
    stacks = response.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1:
        _fail("support stack readback is not singular")
    body = {
        "stack_id": stack_id,
        "absent": False,
        "status": stacks[0].get("StackStatus"),
        "request_id": response["ResponseMetadata"]["RequestId"],
    }
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def _snapshot_coordinator(ports: _AwsPorts) -> object:
    from .task12_snapshot_cleanup import SnapshotCleanupCoordinator

    return SnapshotCleanupCoordinator(
        adapter=_operation_ledger(ports),
        snapshot_client=_SnapshotClient(ports),
    )


def _snapshot_common_request(
    request: Mapping[str, object],
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
) -> Tuple[object, Dict[str, object]]:
    from .task12_retained_state import SnapshotCleanupTransitionPlan

    exact = _exact_fields(
        request,
        ("plan", "domain", "operation_identity_sha256", "owner_nonce_capsule"),
        "snapshot cleanup transition",
    )
    plan = _materialize_dataclass(SnapshotCleanupTransitionPlan, exact["plan"])
    domain = _text(exact["domain"], "snapshot cleanup domain")
    nonce = _decrypt_operation_nonce(
        ports=ports,
        invocation=invocation,
        capsule=exact["owner_nonce_capsule"],
        plan=plan,
        domain=domain,
    )
    return (
        plan,
        {
            "domain": domain,
            "operation_identity_sha256": _sha(
                exact["operation_identity_sha256"],
                "snapshot cleanup operation",
            ),
            "raw_owner_nonce": nonce,
        },
    )


def _decrypt_operation_nonce(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    capsule: object,
    plan: object,
    domain: str,
) -> bytes:
    """Decrypt one ciphertext capsule only for this exact retained execution."""

    from .task12_nonce_capsule import decrypt_owner_nonce_capsule

    if type(capsule) is not dict:
        _fail("owner nonce capsule is absent")
    allowed_domains = {
        "RECOVERY": {"RECOVERY"},
        "TEARDOWN": {"TEARDOWN", "FINALIZATION"},
        "TASK12_SNAPSHOT_CLEANUP": {"SNAPSHOT_CLEANUP"},
        "SNAPSHOT_CLEANUP": {"SNAPSHOT_CLEANUP"},
    }
    owner_records: list[Mapping[str, object]] = []
    for name in (
        "recovery_control",
        "finalization_control",
        "cleanup_control",
        "control",
        "authority",
        "owner",
    ):
        exact = getattr(plan, name, None)
        for side in ("before", "after", "expected"):
            record = getattr(exact, side, None)
            if (
                isinstance(record, Mapping)
                and record.get("owner_invocation_nonce_sha256") is not None
            ):
                owner_records.append(record)

    authorities: list[dict[str, object]] = []
    for record in owner_records:
        record_type = record.get("record_type")
        if record_type == "glm52_production_recovery_control":
            authority_domain = "RECOVERY"
            barrier = record.get("recovery_barrier_nonce_sha256")
            control_revision = record.get(
                "support_control_revision_at_seal"
            )
        elif record_type == "glm52_production_finalization_control":
            authority_domain = "FINALIZATION"
            barrier = record.get("finalization_barrier_nonce_sha256")
            control_revision = record.get(
                "teardown_sealed_control_revision"
            )
        elif (
            record_type
            == "glm52_production_snapshot_cleanup_control"
        ):
            authority_domain = "SNAPSHOT_CLEANUP"
            barrier = record.get("cleanup_barrier_nonce_sha256")
            context = capsule.get("encryption_context")
            try:
                control_revision = int(
                    context.get("control_revision", "")
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise Task12LambdaAdapterError(
                    "snapshot capsule revision is not typed"
                ) from exc
            if (
                type(record.get("revision")) is not int
                or control_revision < 1
                or control_revision > record["revision"]
                or record.get("state")
                not in {
                    "OWNED",
                    "DELETE_POSSIBLY_SENT",
                    "DELETE_RECONCILING",
                }
            ):
                _fail("snapshot capsule revision drifted")
        else:
            continue
        expected = {
            "account_id": record.get("account_id"),
            "region": record.get("region"),
            "run_id": record.get("run_id"),
            "activation_id": record.get("activation_id"),
            "authority_domain": authority_domain,
            "owner_execution_arn": record.get("owner_execution_arn"),
            "owner_state_machine_version_arn": record.get(
                "owner_state_machine_version_arn"
            ),
            "owner_attempt": record.get("owner_attempt"),
            "barrier_nonce_sha256": barrier,
            "control_revision": control_revision,
            "owner_hard_expires_at": record.get(
                "owner_hard_expires_at"
            ),
        }
        if (
            expected["account_id"] != ACCOUNT_ID
            or expected["region"] != REGION
            or expected["run_id"] != RUN_ID
            or expected["activation_id"] != invocation.activation_id
            or expected["owner_execution_arn"]
            != invocation.state_machine_execution_arn
            or expected["owner_state_machine_version_arn"]
            != invocation.caller_state_machine_version_arn
            or expected["authority_domain"]
            not in allowed_domains.get(domain, {domain})
            or capsule.get("nonce_sha256")
            != record.get("owner_invocation_nonce_sha256")
        ):
            continue
        if expected not in authorities:
            authorities.append(expected)
    if len(authorities) == 1:
        expected = authorities[0]
    elif not authorities and domain == "TEARDOWN":
        # TEARDOWN_SEALING is a one-shot transition with no retained owner.
        context = capsule.get("encryption_context")
        try:
            expected = {
                "account_id": ACCOUNT_ID,
                "region": REGION,
                "run_id": RUN_ID,
                "activation_id": invocation.activation_id,
                "authority_domain": "TEARDOWN",
                "owner_execution_arn": invocation.state_machine_execution_arn,
                "owner_state_machine_version_arn": (
                    invocation.caller_state_machine_version_arn
                ),
                "owner_attempt": int(context.get("owner_attempt", "")),
                "barrier_nonce_sha256": context.get(
                    "barrier_nonce_sha256"
                ),
                "control_revision": int(
                    context.get("control_revision", "")
                ),
                "owner_hard_expires_at": context.get(
                    "owner_hard_expires_at"
                ),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise Task12LambdaAdapterError(
                "one-shot teardown nonce context is not typed"
            ) from exc
    else:
        _fail("owner nonce authority is absent or ambiguous")
    nonce = decrypt_owner_nonce_capsule(
        ports=ports,
        capsule=capsule,
        expected_authority=expected,
    )
    nonce_sha256 = hashlib.sha256(nonce).hexdigest()
    if nonce_sha256 != capsule.get("nonce_sha256"):
        _fail("owner nonce capsule plaintext identity drifted")
    return nonce


def _strict_operation_result(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    state: Mapping[str, object],
) -> Optional[object]:
    """Dispatch a strict operation row to its real domain boundary."""

    spec = operation_spec(invocation.operation_kind)
    descriptor_type = (
        "glm52_task12_"
        + invocation.operation_kind.lower()
        + "_descriptor_v1"
    )
    live_sources: Mapping[str, Mapping[str, object]] = {}
    if state.get("record_type") == descriptor_type:
        state, live_sources = _read_live_operation_source(
            ports=ports,
            invocation=invocation,
            descriptor=state,
        )
    if state.get("record_type") != spec.record_type:
        return None
    request = _exact_operation_request(state, invocation)
    operation = invocation.operation_kind
    result: object

    if operation == "RECONCILE_RETAINED_LIFECYCLE_TRIGGER":
        from .task12_runtime import RuntimeReadBoundaries, RuntimeScan, collect_runtime_scan

        exact = _exact_fields(
            request,
            ("correlation_identity_sha256", "runtime_scan"),
            operation,
        )
        scan = _materialize_dataclass(RuntimeScan, exact["runtime_scan"])
        reader = _StaticRuntimeReader(scan)
        result = collect_runtime_scan(
            boundaries=RuntimeReadBoundaries(
                request_reader=reader,
                job_reader=reader,
                controller_reader=reader,
                worker_reader=reader,
                allocation_reader=reader,
                spend_reader=reader,
            ),
            correlation_identity_sha256=_sha(
                exact["correlation_identity_sha256"], "runtime correlation"
            ),
        )
    elif operation == "RETAINED_ACQUIRE_RECOVERY_SEALING":
        from .task12_retained_state import RecoverySealPlan

        result = _commit_operation_plan(
            ports=ports,
            invocation=invocation,
            request=request,
            plan_type=RecoverySealPlan,
            method_name="commit_recovery_seal",
        )
    elif operation == "RETAINED_STOP_OR_OBSERVE_SUPPORT_EXECUTION":
        result = _observe_support_execution(ports=ports, request=request)
    elif operation == "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION":
        from .task12_live_owner_death import OwnerDeathClassificationPlan

        result = _commit_operation_plan(
            ports=ports,
            invocation=invocation,
            request=request,
            plan_type=OwnerDeathClassificationPlan,
            method_name="classify_owner_dead_sky_action",
        )
    elif operation == "RETAINED_CORRELATE_REQUESTS_AND_JOBS":
        result = _correlate_and_persist_request_jobs(
            ports=ports,
            invocation=invocation,
            request=request,
        )
    elif operation == "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT":
        result = _observe_stack_absence(ports=ports, request=request)
    elif operation == "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY":
        result = _read_exact_ledger_authority(
            ports=ports,
            request=request,
            label="snapshot delete authority",
        )
    elif operation in {
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT",
        "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
    }:
        from .task12_runtime import RuntimeScan, prove_runtime_terminal

        exact = _exact_fields(
            request,
            ("first_runtime_scan", "second_runtime_scan", "minimum_quiet_seconds"),
            operation,
        )
        result = prove_runtime_terminal(
            first=_materialize_dataclass(RuntimeScan, exact["first_runtime_scan"]),
            second=_materialize_dataclass(RuntimeScan, exact["second_runtime_scan"]),
            minimum_quiet_seconds=_positive(
                exact["minimum_quiet_seconds"], "runtime quiet interval"
            ),
        )
    elif operation == "RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED":
        from .task12_live_recovery_handoff import execute_live_request

        result = execute_live_request(
            invocation=invocation,
            request=request,
            ports=ports,
            writer=lambda *, record, action, audit: _write_candidate(
                ports=ports,
                invocation=invocation,
                writer_kind="RecoveryHandoff",
                authority_domain="RECOVERY",
                record=record,
                action=action,
                audit=audit,
            ),
        )
    elif operation == "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION":
        from .task12_runtime import (
            CancellationCommand,
            CancellationObservation,
            reconcile_cancellation,
        )

        exact = _exact_fields(request, ("command", "observation"), operation)
        result = reconcile_cancellation(
            command=_materialize_dataclass(CancellationCommand, exact["command"]),
            observation=_materialize_dataclass(
                CancellationObservation, exact["observation"]
            ),
        )
    elif operation == "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED":
        from .task12_runtime import (
            ControllerStopEvidence,
            RuntimeScan,
            prove_controller_quiescence,
        )

        exact = _exact_fields(
            request,
            ("stop", "first_runtime_scan", "second_runtime_scan", "minimum_quiet_seconds"),
            operation,
        )
        result = prove_controller_quiescence(
            stop=_materialize_dataclass(ControllerStopEvidence, exact["stop"]),
            first=_materialize_dataclass(RuntimeScan, exact["first_runtime_scan"]),
            second=_materialize_dataclass(RuntimeScan, exact["second_runtime_scan"]),
            minimum_quiet_seconds=_positive(
                exact["minimum_quiet_seconds"], "controller quiet interval"
            ),
        )
    elif operation == "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES":
        from .task12_live_drain_effects import (
            LiabilityTransferProof,
            validate_liability_transfer_proof,
        )

        exact = _exact_fields(
            request,
            ("liability_transfer",),
            operation,
        )
        result = validate_liability_transfer_proof(
            _materialize_dataclass(
                LiabilityTransferProof,
                exact["liability_transfer"],
            )
        )
    elif operation == "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS":
        from .task12_worker_drain import (
            WorkerDrainCandidate,
            WorkerDrainServices,
            build_task12_worker_drain_authority,
            dispatch_worker_drain,
        )
        from .task12_writers import (
            RetainedWriterActionAuthority,
            RetainedWriterAuditAuthority,
        )

        exact = _exact_fields(request, ("candidate", "action", "audit"), operation)
        candidate = _materialize_dataclass(WorkerDrainCandidate, exact["candidate"])
        authority = build_task12_worker_drain_authority(
            candidate=candidate,
            action=_materialize_dataclass(
                RetainedWriterActionAuthority, exact["action"]
            ),
            audit=_materialize_dataclass(
                RetainedWriterAuditAuthority, exact["audit"]
            ),
        )
        result = dispatch_worker_drain(
            services=WorkerDrainServices(ssm=ports.client("ssm")),
            authority=authority,
        )
    elif operation in {
        "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
        "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED",
        "RETAINED_INVOKE_H1G_DRAINED_WRITER",
    }:
        exact = _exact_fields(
            request,
            ("writer_kind", "authority_domain", "record", "action", "audit"),
            operation,
        )
        result = _write_candidate(
            ports=ports,
            invocation=invocation,
            writer_kind=_text(exact["writer_kind"], operation + " writer"),
            authority_domain=_text(
                exact["authority_domain"], operation + " authority domain"
            ),
            record=exact["record"],
            action=exact["action"],
            audit=exact["audit"],
        )
    elif operation == "RETAINED_ENTER_RECOVERY_COMPLETE":
        from .task12_retained_state import RecoveryProgressPlan

        result = _commit_operation_plan(
            ports=ports,
            invocation=invocation,
            request=request,
            plan_type=RecoveryProgressPlan,
            method_name="commit_recovery_progress",
        )
    elif operation in {
        "RETAINED_ACQUIRE_TEARDOWN_SEALING",
        "RETAINED_ENTER_TEARDOWN_SEALED",
    }:
        from .task12_retained_state import TeardownSealPlan

        result = _commit_operation_plan(
            ports=ports,
            invocation=invocation,
            request=request,
            plan_type=TeardownSealPlan,
            method_name="commit_teardown_seal",
        )
    elif operation == "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT":
        from .task12_snapshot_cleanup import SnapshotCapture, SnapshotObservation

        exact = _exact_fields(
            request,
            (
                "observations",
                "expected_source_volume_id",
                "expected_kms_key_arn",
                "expected_snapshot_tags_sha256",
            ),
            operation,
        )
        if type(exact["observations"]) is not list:
            _fail("snapshot observations must be one exact array")
        result = SnapshotCapture.discover(
            observations=tuple(
                _materialize_dataclass(SnapshotObservation, item)
                for item in exact["observations"]
            ),
            expected_source_volume_id=_text(
                exact["expected_source_volume_id"], "snapshot source volume"
            ),
            expected_kms_key_arn=_text(
                exact["expected_kms_key_arn"], "snapshot KMS key"
            ),
            expected_snapshot_tags_sha256=_sha(
                exact["expected_snapshot_tags_sha256"], "snapshot tags"
            ),
        )
    elif operation == "RETAINED_AUDIT_SUPPORT_ORPHANS":
        result = _strict_orphan_audit(ports=ports, request=request)
    elif operation == "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE":
        result = _strict_snapshot_arm(
            ports=ports,
            invocation=invocation,
            request=request,
        )
    elif operation == "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE":
        from .task12_snapshot_cleanup import SnapshotCapture, SnapshotSchedule

        exact = _exact_fields(
            request,
            ("capture", "schedule"),
            operation,
        )
        capture = _materialize_dataclass(SnapshotCapture, exact["capture"])
        schedule = _materialize_dataclass(SnapshotSchedule, exact["schedule"])
        result = SnapshotSchedule.validate(
            capture=capture,
            schedule_arn=schedule.schedule_arn,
            delete_not_before=schedule.delete_not_before,
            schedule_input_sha256=schedule.schedule_input_sha256,
        )
    elif operation == "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER":
        exact = _exact_fields(
            request,
            (
                "plan",
                "observed_at",
                "domain",
                "operation_identity_sha256",
                "owner_nonce_capsule",
            ),
            operation,
        )
        raw_plan = exact["plan"]
        if (
            type(raw_plan) is dict
            and set(raw_plan)
            == {"index", "authority", "cleanup_control"}
        ):
            from .task12_retained_state import (
                SnapshotCleanupOwnerTakeoverPlan,
                build_snapshot_cleanup_owner_takeover_plan,
            )

            plan = _materialize_dataclass(
                SnapshotCleanupOwnerTakeoverPlan, raw_plan
            )
            plan = build_snapshot_cleanup_owner_takeover_plan(
                index=plan.index,
                authority=plan.authority,
                cleanup_control=plan.cleanup_control,
            )
            observed_at = _text(
                exact["observed_at"], "snapshot takeover observed_at"
            )
            if plan.cleanup_control.after["updated_at"] != observed_at:
                _fail("snapshot takeover observed_at drifted")
            domain = _text(
                exact["domain"], "snapshot takeover domain"
            )
            nonce = _decrypt_operation_nonce(
                ports=ports,
                invocation=invocation,
                capsule=exact["owner_nonce_capsule"],
                plan=plan,
                domain=domain,
            )
            result = _operation_ledger(
                ports
            ).take_over_snapshot_cleanup_owner(
                plan=plan,
                domain=domain,
                operation_identity_sha256=_sha(
                    exact["operation_identity_sha256"],
                    "snapshot takeover identity",
                ),
                raw_owner_nonce=nonce,
            )
            from .dynamodb import TransactionResolution

            if type(result) is not TransactionResolution:
                _fail("snapshot takeover returned an unauthenticated transaction")
        else:
            plan, common = _snapshot_common_request(
                {key: exact[key] for key in exact if key != "observed_at"},
                ports=ports,
                invocation=invocation,
            )
            result = _snapshot_coordinator(ports).acquire(
                plan=plan,
                observed_at=_text(
                    exact["observed_at"], "snapshot observed_at"
                ),
                **common,
            )
        from .dynamodb import TransactionResolution
        from .task12_snapshot_cleanup import (
            SnapshotOwnerAcquisitionResult,
        )

        acquired_control = plan.cleanup_control.after
        if (
            type(result) is not TransactionResolution
            or not any(
                record == acquired_control for record in result.records
            )
        ):
            _fail(
                "snapshot acquisition lacks authenticated committed control"
            )
        result = SnapshotOwnerAcquisitionResult(
            cleanup_state=acquired_control["state"],
            resolution=result,
        )
    elif operation == "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION":
        from .task12_retained_state import (
            SnapshotDeleteActionPlan,
            build_snapshot_delete_action_plan,
        )

        exact = _exact_fields(
            request,
            (
                "plan",
                "partition_key",
                "sort_key",
                "expected_record_identity_sha256",
                "domain",
                "operation_identity_sha256",
                "owner_nonce_capsule",
            ),
            operation,
        )
        plan = _materialize_dataclass(
            SnapshotDeleteActionPlan, exact["plan"]
        )
        validated_plan = build_snapshot_delete_action_plan(
            index=plan.index,
            control=plan.control,
            action=plan.action,
        )
        nonce = _decrypt_operation_nonce(
            ports=ports,
            invocation=invocation,
            capsule=exact["owner_nonce_capsule"],
            plan=validated_plan,
            domain=_text(exact["domain"], "snapshot action arm domain"),
        )
        resolution = _operation_ledger(ports).arm_snapshot_delete_action(
            plan=validated_plan,
            domain=_text(exact["domain"], "snapshot action arm domain"),
            operation_identity_sha256=_sha(
                exact["operation_identity_sha256"],
                "snapshot action arm identity",
            ),
            raw_owner_nonce=nonce,
        )
        from .dynamodb import TransactionResolution

        if type(resolution) is not TransactionResolution:
            _fail("snapshot action arm returned an unauthenticated transaction")
        live_action = _read_exact_ledger_authority(
            ports=ports,
            request={
                key: exact[key]
                for key in (
                    "partition_key",
                    "sort_key",
                    "expected_record_identity_sha256",
                )
            },
            label="snapshot armed delete action",
        )
        result = {"transaction": resolution, "live_action": live_action}
    elif operation in {
        "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
        "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE",
    }:
        from .task12_retained_state import SnapshotCleanupTransitionPlan

        result = _commit_operation_plan(
            ports=ports,
            invocation=invocation,
            request=request,
            plan_type=SnapshotCleanupTransitionPlan,
            method_name="commit_snapshot_cleanup_transition",
        )
    elif operation in {
        "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT",
        "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT",
    }:
        from .task12_retained_state import SnapshotDeleteActionPlan

        result = _commit_operation_plan(
            ports=ports,
            invocation=invocation,
            request=request,
            plan_type=SnapshotDeleteActionPlan,
            method_name=(
                "close_snapshot_delete_action"
                if operation
                == "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"
                else "arm_snapshot_delete_action"
            ),
        )
    elif operation == "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK":
        exact = _exact_fields(request, ("snapshot_id",), operation)
        result = _SnapshotClient(ports).delete_snapshot(
            snapshot_id=_text(exact["snapshot_id"], "snapshot ID")
        )
    elif operation == "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE":
        exact = _exact_fields(
            request,
            ("plan", "domain", "operation_identity_sha256", "owner_nonce_capsule"),
            operation,
        )
        raw_plan = exact["plan"]
        if (
            type(raw_plan) is dict
            and set(raw_plan)
            == {
                "index",
                "authority",
                "cleanup_control",
                "cleanup_action",
            }
        ):
            from .task12_retained_state import (
                SnapshotCleanupReconcileReadPlan,
                build_snapshot_cleanup_reconcile_read_plan,
            )

            plan = _materialize_dataclass(
                SnapshotCleanupReconcileReadPlan, raw_plan
            )
            plan = build_snapshot_cleanup_reconcile_read_plan(
                index=plan.index,
                authority=plan.authority,
                cleanup_control=plan.cleanup_control,
                cleanup_action=plan.cleanup_action,
            )
            domain = _text(
                exact["domain"], "snapshot reconciliation domain"
            )
            nonce = _decrypt_operation_nonce(
                ports=ports,
                invocation=invocation,
                capsule=exact["owner_nonce_capsule"],
                plan=plan,
                domain=domain,
            )
            result = _snapshot_coordinator(ports).reconcile_readback(
                plan=None,
                current_control=plan.cleanup_control.expected,
                domain=domain,
                operation_identity_sha256=_sha(
                    exact["operation_identity_sha256"],
                    "snapshot reconciliation operation",
                ),
                raw_owner_nonce=nonce,
            )
        else:
            plan, common = _snapshot_common_request(
                exact,
                ports=ports,
                invocation=invocation,
            )
            result = _snapshot_coordinator(ports).reconcile_readback(
                plan=plan,
                **common,
            )
    else:  # pragma: no cover - registry/dispatcher import-time invariant
        _fail("strict Task 12 operation lacks a domain boundary")

    _persist_live_operation_successors(
        ports=ports,
        invocation=invocation,
        live_sources=live_sources,
        request=request,
        domain_result=result,
    )
    semantic = execute_operation_semantics(
        row=state,
        operation_input=invocation.operation_input,
    )
    domain_value = _json_value(result)
    domain_identity = canonical_sha256(domain_value)
    owner_nonce_capsule = request.get("owner_nonce_capsule")
    if owner_nonce_capsule is None:
        prior = invocation.operation_input.get("task12_last_result")
        prior_result = prior.get("result") if type(prior) is dict else None
        if type(prior_result) is dict:
            owner_nonce_capsule = prior_result.get("owner_nonce_capsule")
    result_body = {
        "result_kind": semantic.result_kind,
        "action_kind": semantic.action_kind,
        "behavior_kind": semantic.behavior_kind,
        "predecessor_operation_kind": semantic.predecessor_operation_kind,
        "predecessor_result_identity_sha256": (
            semantic.predecessor_result_identity_sha256
        ),
        "operation_payload_identity_sha256": (
            semantic.operation_payload_identity_sha256
        ),
        "domain_result": domain_value,
        "domain_result_identity_sha256": domain_identity,
    }
    if owner_nonce_capsule is not None:
        if type(owner_nonce_capsule) is not dict:
            _fail("owner nonce continuation capsule is not closed")
        result_body["owner_nonce_capsule"] = owner_nonce_capsule
    return {
        **result_body,
        "canonical_body_sha256": canonical_sha256(result_body),
    }


def _read_live_operation_source(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    descriptor: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, Mapping[str, object]]]:
    """Read and authenticate the current operation source after predecessor auth."""

    try:
        exact_descriptor = validate_operation_descriptor(descriptor)
    except Task12OperationContractError as exc:
        raise Task12LambdaAdapterError(
            "live operation descriptor is invalid"
        ) from exc
    # Force every descriptor through the typed canonical reader.  The legacy
    # generic ``live_source`` coordinate is intentionally not a compatibility
    # path: descriptors published by the current graph contain only the three
    # coordinate kinds below.
    live_sources = _read_typed_live_sources(
        ports=ports,
        invocation=invocation,
        descriptor=exact_descriptor,
    )
    request = _materialize_live_operation_request(
        ports=ports,
        invocation=invocation,
        live_sources=live_sources,
    )
    spec = operation_spec(invocation.operation_kind)
    request_key = spec.action_kind + "_request"
    try:
        row = build_operation_row(
            operation_kind=invocation.operation_kind,
            activation_id=invocation.activation_id,
            activation_ordinal=invocation.activation_ordinal,
            generation=invocation.generation,
            payload={request_key: request},
        )
    except Task12OperationContractError as exc:
        raise Task12LambdaAdapterError(
            invocation.operation_kind
            + " live request materialization is not closed"
        ) from exc
    return row, live_sources


def _materialize_live_operation_request(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    live_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Construct the operations whose request is wholly present in live truth.

    This boundary deliberately has named branches.  A new operation cannot
    become executable by accidentally sharing field names with another
    operation's source record.
    """

    operation = invocation.operation_kind
    for builder in _external_live_materializers():
        if operation in builder.SUPPORTED_OPERATIONS:
            try:
                request = builder.materialize_live_request(
                    operation_kind=operation,
                    invocation=invocation,
                    live_sources=live_sources,
                    ports=ports,
                )
            except ValueError as exc:
                raise Task12LambdaAdapterError(
                    operation + " live materializer rejected canonical sources"
                ) from exc
            if type(request) is not dict:
                _fail(operation + " external live builder returned no request")
            return dict(request)
    if operation == "RETAINED_ACQUIRE_RECOVERY_SEALING":
        return _materialize_recovery_seal_request(
            ports=ports,
            invocation=invocation,
            live_sources=live_sources,
        )
    if operation == "RETAINED_STOP_OR_OBSERVE_SUPPORT_EXECUTION":
        execution = live_sources.get("execution")
        if type(execution) is not dict:
            _fail("support execution source is absent")
        return {
            "execution_arn": _text(
                execution.get("expected_execution_arn"),
                "support execution ARN",
            ),
            "expected_state_machine_version_arn": _text(
                execution.get("expected_state_machine_version_arn"),
                "support state-machine version",
            ),
            "stop_if_running": execution.get("state") == "RUNNING",
        }
    if operation == "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION":
        recovery = live_sources.get("recovery_control")
        if type(recovery) is not dict:
            _fail("recovery owner source is absent")
        execution_arn = _text(
            recovery.get("owner_execution_arn"), "recovery owner execution"
        )
        response = ports.client("stepfunctions").describe_execution(
            executionArn=execution_arn
        )
        if (
            not _authenticated_metadata(response.get("ResponseMetadata"))
            or response.get("executionArn") != execution_arn
            or response.get("stateMachineVersionArn")
            != recovery.get("owner_state_machine_version_arn")
            or response.get("status")
            not in {"RUNNING", "SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
        ):
            _fail("recovery owner source observation is unauthenticated")
        return {
            "owner_execution_arn": execution_arn,
            "expected_state_machine_version_arn": _text(
                recovery.get("owner_state_machine_version_arn"),
                "recovery owner state-machine version",
            ),
            "expected_action": (
                "OBSERVE_OWNER"
                if response["status"] == "RUNNING"
                else "RECOVER_OWNER_DEAD"
            ),
        }
    if operation == "RETAINED_CORRELATE_REQUESTS_AND_JOBS":
        from .records import canonical_record_identity, ledger_pk, ledger_sk

        execution = live_sources.get("execution")
        if type(execution) is not dict:
            _fail("request/job execution source is absent")
        epoch = execution.get("epoch")
        if type(epoch) is not int or epoch < 1:
            _fail("request/job execution epoch is not authenticated")
        return {
            "partition_key": ledger_pk(RUN_ID),
            "sort_key": ledger_sk(
                "glm52_production_execution",
                activation_id=invocation.activation_id,
                epoch=epoch,
            ),
            "expected_record_identity_sha256": canonical_record_identity(
                "glm52_production_execution", execution
            ),
        }
    if operation == "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY":
        from .records import canonical_record_identity, ledger_pk, ledger_sk

        action = live_sources.get("snapshot_cleanup_action")
        if type(action) is not dict:
            _fail("snapshot delete action source is absent")
        attempt = action.get("attempt")
        if type(attempt) is not int or attempt < 1:
            _fail("snapshot delete action attempt is not authenticated")
        return {
            "partition_key": ledger_pk(RUN_ID),
            "sort_key": ledger_sk(
                "glm52_production_snapshot_cleanup_action",
                activation_id=invocation.activation_id,
                attempt=attempt,
            ),
            "expected_record_identity_sha256": canonical_record_identity(
                "glm52_production_snapshot_cleanup_action", action
            ),
        }
    if operation == "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK":
        cleanup = live_sources.get("snapshot_cleanup_control")
        if type(cleanup) is not dict:
            _fail("snapshot cleanup control source is absent")
        return {
            "snapshot_id": _text(
                cleanup.get("snapshot_id"), "snapshot cleanup snapshot ID"
            )
        }
    _fail(
        operation
        + " has no canonical live request builder; authenticated sources="
        + ",".join(sorted(live_sources))
    )


def _external_live_materializers() -> tuple[object, ...]:
    """Load independently tested operation-family builders without cycles."""

    modules: list[object] = []
    for name in (
        "task12_live_runtime",
        "task12_live_cancellation",
        "task12_live_drain_effects",
        "task12_live_drain_terminal",
        "task12_live_owner_death",
        "task12_live_recovery_handoff",
        "task12_live_snapshot_finalization",
    ):
        try:
            module = __import__(
                "glm52_enforcement." + name,
                fromlist=("SUPPORTED_OPERATIONS",),
            )
        except ModuleNotFoundError as exc:
            if exc.name != "glm52_enforcement." + name:
                raise
            continue
        if (
            type(getattr(module, "SUPPORTED_OPERATIONS", None)) is not frozenset
            or not callable(getattr(module, "materialize_live_request", None))
            or not callable(getattr(module, "persist_live_successors", None))
        ):
            _fail(name + " live materializer module is not closed")
        modules.append(module)
    return tuple(modules)


def _persist_live_operation_successors(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    live_sources: Mapping[str, Mapping[str, object]],
    request: Mapping[str, object],
    domain_result: object,
) -> None:
    handled = False
    for builder in _external_live_materializers():
        if invocation.operation_kind not in builder.SUPPORTED_OPERATIONS:
            continue
        handled = bool(
            builder.persist_live_successors(
                operation_kind=invocation.operation_kind,
                invocation=invocation,
                live_sources=live_sources,
                request=request,
                domain_result=domain_result,
                ports=ports,
            )
        )
        if not handled:
            _fail(
                invocation.operation_kind
                + " successor persistence was not acknowledged"
            )
        return
    # Local operations either commit their successor in the domain effect
    # (recovery sealing) or are observation-only.
    if invocation.operation_kind in {
        "RETAINED_ACQUIRE_RECOVERY_SEALING",
        "RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
        "RETAINED_STOP_OR_OBSERVE_SUPPORT_EXECUTION",
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT",
        "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION",
        "RETAINED_CORRELATE_REQUESTS_AND_JOBS",
    }:
        return
    if live_sources:
        _fail(invocation.operation_kind + " has no successor persistence owner")


def _materialize_recovery_seal_request(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    live_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build the OPEN/DORMANT recovery acquisition from current ledger rows."""

    from .dynamodb import ExactCheck, ExactUpdate, LedgerKey
    from .records import canonical_record_identity, ledger_sk, validate_record
    from .task12_retained_state import build_recovery_seal_plan
    from .task12_nonce_capsule import generate_owner_nonce_capsule

    index = live_sources.get("activation_index")
    control = live_sources.get("control")
    execution = live_sources.get("execution")
    recovery = live_sources.get("recovery_control")
    if not all(type(value) is dict for value in (index, control, execution, recovery)):
        _fail("recovery seal sources are incomplete")
    index = validate_record("glm52_production_activation_index", index)
    control = validate_record("glm52_production_control", control)
    execution = validate_record("glm52_production_execution", execution)
    recovery = validate_record("glm52_production_recovery_control", recovery)
    if (
        index["current_activation_id"] != invocation.activation_id
        or index["current_activation_ordinal"] != invocation.activation_ordinal
        or control["phase"] != "OPEN"
        or recovery["state"] != "DORMANT"
    ):
        _fail("recovery seal sources are not at the acquisition edge")
    now = datetime.now(timezone.utc)
    observed_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    hard_expiry = (now + timedelta(minutes=15)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    def rehash(value: Mapping[str, object]) -> dict[str, object]:
        body = dict(value)
        carried_hash = body.pop("canonical_body_sha256", None)
        if carried_hash is None:
            return body
        return {**body, "canonical_body_sha256": canonical_sha256(body)}

    control_after = rehash(
        {
            **control,
            "phase": "RECOVERY_SEALING",
            "revision": control["revision"] + 1,
            "updated_at": observed_at,
        }
    )
    recovery_barrier = canonical_sha256(
        {
            "authority_domain": "RECOVERY",
            "activation_id": invocation.activation_id,
            "owner_execution_arn": invocation.state_machine_execution_arn,
            "owner_state_machine_version_arn": (
                invocation.caller_state_machine_version_arn
            ),
            "control_revision": control_after["revision"],
            "owner_hard_expires_at": hard_expiry,
        }
    )
    owner_nonce_capsule, raw_nonce = generate_owner_nonce_capsule(
        ports=ports,
        authority={
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "activation_id": invocation.activation_id,
            "authority_domain": "RECOVERY",
            "owner_execution_arn": invocation.state_machine_execution_arn,
            "owner_state_machine_version_arn": (
                invocation.caller_state_machine_version_arn
            ),
            "owner_attempt": 1,
            "barrier_nonce_sha256": recovery_barrier,
            "control_revision": control_after["revision"],
            "owner_hard_expires_at": hard_expiry,
        },
    )
    nonce_sha256 = hashlib.sha256(raw_nonce).hexdigest()
    recovery_after = rehash(
        {
            **recovery,
            "state": "OWNED",
            "owner_attempt": 1,
            "owner_execution_arn": invocation.state_machine_execution_arn,
            "owner_state_machine_version_arn": (
                invocation.caller_state_machine_version_arn
            ),
            "owner_dispatch_identity_sha256": (
                invocation.dispatch_identity_sha256
            ),
            "owner_invocation_nonce_sha256": nonce_sha256,
            "owner_hard_expires_at": hard_expiry,
            "recovery_barrier_nonce_sha256": recovery_barrier,
            "support_control_revision_at_seal": control_after["revision"],
            "support_execution_identity_sha256": canonical_record_identity(
                "glm52_production_execution", execution
            ),
            "allowed_action_set_sha256": canonical_sha256(
                {"operations": sorted(_OPERATIONS[invocation.handler_kind])}
            ),
            "revision": recovery["revision"] + 1,
            "updated_at": observed_at,
        }
    )
    plan = build_recovery_seal_plan(
        index=ExactCheck(LedgerKey(RUN_ID, "ACTIVATION_INDEX"), index),
        support_execution=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_execution",
                    activation_id=invocation.activation_id,
                    epoch=execution["epoch"],
                ),
            ),
            execution,
        ),
        control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=invocation.activation_id,
                ),
            ),
            control,
            control_after,
        ),
        recovery_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_recovery_control",
                    activation_id=invocation.activation_id,
                ),
            ),
            recovery,
            recovery_after,
        ),
    )
    operation_identity = canonical_sha256(
        {
            "operation_kind": invocation.operation_kind,
            "execution_arn": invocation.state_machine_execution_arn,
            "control_before": canonical_record_identity(
                "glm52_production_control", control
            ),
            "recovery_before": canonical_record_identity(
                "glm52_production_recovery_control", recovery
            ),
        }
    )
    return {
        "plan": _json_value(plan),
        "domain": "RECOVERY",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": owner_nonce_capsule,
    }


def _read_typed_live_sources(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    descriptor: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    """Authenticate the exact DDB/S3 records selected by a typed descriptor."""

    from .dynamodb import decode_item, encode_item
    from .records import canonical_record_identity, ledger_sk, validate_record
    from .task12_writers import retained_writer_coordinate

    coordinates = descriptor["source_coordinates"]
    partition_key = coordinates["ledger_partition_key"]
    sources = coordinates["sources"]
    table_name = _text(
        ports.deployment.role_coordinates.get("ledger_table_name"), "ledger table"
    )

    def read_ddb(source: Mapping[str, object]) -> Mapping[str, object]:
        response = ports.client("dynamodb").get_item(
            TableName=table_name,
            Key=encode_item({"PK": partition_key, "SK": source["sort_key"]}),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        if (
            not _authenticated_metadata(response.get("ResponseMetadata"))
            or type(response.get("Item")) is not dict
        ):
            _fail("typed DDB source read is unauthenticated")
        value = decode_item(response["Item"])
        if value.pop("PK", None) != partition_key or value.pop("SK", None) != source["sort_key"]:
            _fail("typed DDB source key drifted")
        try:
            exact = validate_record(_text(source["record_type"], "source record type"), value)
        except (TypeError, ValueError) as exc:
            raise Task12LambdaAdapterError("typed DDB source record drifted") from exc
        identity = canonical_record_identity(
            _text(source["record_type"], "source record type"),
            exact,
        )
        if (
            "canonical_body_sha256" in exact
            and exact["canonical_body_sha256"] != identity
        ):
            _fail("typed DDB source self identity drifted")
        return exact

    loaded: dict[str, Mapping[str, object]] = {}
    for raw_source in sources:
        if type(raw_source) is not dict:
            _fail("typed source is not an object")
        source = raw_source
        alias = _text(source.get("alias"), "typed source alias")
        kind = source.get("coordinate_kind")
        if kind == "DDB_EXACT":
            loaded[alias] = read_ddb(source)
        elif kind == "DDB_DERIVED":
            control = source.get("control")
            derive = source.get("derive")
            if (
                type(control) is not dict
                or type(derive) is not dict
                or set(derive)
                != {
                    "identity_field",
                    "control_field",
                    "integer_offset",
                }
            ):
                _fail("derived source is not closed")
            control_value = read_ddb(control)
            field = _text(derive.get("control_field"), "derived control field")
            identity_field = _text(derive.get("identity_field"), "derived identity field")
            base_ordinal = control_value.get(field)
            offset = derive.get("integer_offset")
            if (
                type(base_ordinal) is not int
                or type(offset) is not int
                or offset not in {0, 1}
            ):
                _fail("derived source control ordinal is not authenticated")
            ordinal = base_ordinal + offset
            if ordinal < 1:
                _fail("derived source ordinal is outside its closed range")
            record_type = _text(source.get("record_type"), "derived record type")
            try:
                sort_key = ledger_sk(
                    record_type,
                    activation_id=invocation.activation_id,
                    **{identity_field: ordinal},
                )
            except (TypeError, ValueError) as exc:
                raise Task12LambdaAdapterError("derived source key is not canonical") from exc
            loaded[alias] = read_ddb(
                {"record_type": record_type, "sort_key": sort_key}
            )
        elif kind == "DDB_KEY_FROM_CONTROL":
            control = source.get("control")
            if type(control) is not dict:
                _fail("key-from-control source control is not closed")
            control_value = read_ddb(control)
            control_field = _text(
                source.get("control_field"),
                "key-from-control field",
            )
            sort_key = _text(
                control_value.get(control_field),
                "key-from-control sort key",
            )
            record_type = _text(
                source.get("record_type"),
                "key-from-control record type",
            )
            value = read_ddb(
                {"record_type": record_type, "sort_key": sort_key}
            )
            if (
                record_type != "glm52_production_action"
                or value.get("activation_id") != invocation.activation_id
                or value.get("activation_ordinal")
                != invocation.activation_ordinal
                or value.get("generation") != invocation.generation
                or value.get("generation_text")
                != invocation.generation_text
                or value.get("action_kind") != "SKY_POST"
                or control_value.get("last_sky_post_generation")
                != invocation.generation
                or control_value.get("last_sky_post_state")
                != value.get("state")
            ):
                _fail("key-from-control Sky action drifted")
            loaded[alias] = value
        elif kind == "S3_VERSIONED_FROM_CONTROL":
            control = source.get("control")
            if type(control) is not dict:
                _fail("versioned S3 source control is not closed")
            control_value = read_ddb(control)
            writer_kind = _text(source.get("writer_kind"), "S3 writer kind")
            bucket = _text(source.get("campaign_bucket"), "S3 campaign bucket")
            expected_coordinate = retained_writer_coordinate(
                writer_kind=writer_kind,
                campaign_bucket=bucket,
                activation_id=invocation.activation_id,
                generation=invocation.generation,
            )
            if (
                control_value.get("writer_kind") != writer_kind
                or control_value.get("campaign_bucket") != bucket
                or expected_coordinate != "s3://" + bucket + "/" + _text(control_value.get("object_key"), "S3 object key")
            ):
                _fail("versioned S3 source control coordinate drifted")
            response = ports.client("s3").get_object(
                Bucket=bucket,
                Key=control_value["object_key"],
                VersionId=control_value["object_version_id"],
                ExpectedBucketOwner=ACCOUNT_ID,
                ChecksumMode="ENABLED",
            )
            raw = response.get("Body").read() if type(response) is dict and response.get("Body") is not None else None
            expected_checksum = (
                base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
                if type(raw) is bytes
                else None
            )
            if (
                type(raw) is not bytes
                or not _authenticated_metadata(response.get("ResponseMetadata"))
                or response.get("VersionId") != control_value["object_version_id"]
                or response.get("ChecksumSHA256") != expected_checksum
                or hashlib.sha256(raw).hexdigest() != control_value["file_sha256"]
            ):
                _fail("versioned S3 source read is unauthenticated")
            value = _canonical_mapping(raw.rstrip(b"\n"), "versioned S3 source")
            try:
                exact = validate_record(_text(source.get("record_type"), "S3 record type"), value)
            except (TypeError, ValueError) as exc:
                raise Task12LambdaAdapterError("versioned S3 source record drifted") from exc
            if exact["canonical_body_sha256"] != control_value["body_sha256"]:
                _fail("versioned S3 source body identity drifted")
            loaded[alias] = exact
        else:
            _fail("typed source kind is not closed")
    return loaded


def _strict_orphan_audit(
    *, ports: _AwsPorts, request: Mapping[str, object]
) -> object:
    from .task12_orphan_audit import (
        DirectGrantAttribution,
        KmsGrantIdentity,
        KmsGrantScan,
        ResourceInventoryScan,
        RetainedResource,
        ServiceGrantAttribution,
        audit_orphans,
    )

    state = _exact_fields(
        request,
        (
            "expected_retained",
            "inventory",
            "retained_grant_baseline",
            "pre_cleanup_grants",
            "direct_grants",
            "service_grants",
            "settling_deadline",
        ),
        "orphan audit",
    )
    if any(
        type(state[name]) is not list
        for name in (
            "expected_retained",
            "retained_grant_baseline",
            "direct_grants",
            "service_grants",
        )
    ):
        _fail("orphan audit arrays are not exact")
    return audit_orphans(
        expected_retained=tuple(
            _materialize_dataclass(RetainedResource, item)
            for item in state["expected_retained"]
        ),
        inventory=_materialize_dataclass(
            ResourceInventoryScan, state["inventory"]
        ),
        retained_grant_baseline=tuple(
            _materialize_dataclass(KmsGrantIdentity, item)
            for item in state["retained_grant_baseline"]
        ),
        pre_cleanup_grants=_materialize_dataclass(
            KmsGrantScan, state["pre_cleanup_grants"]
        ),
        final_grants=_read_live_kms_grants(ports),
        direct_grants=tuple(
            _materialize_dataclass(DirectGrantAttribution, item)
            for item in state["direct_grants"]
        ),
        service_grants=tuple(
            _materialize_dataclass(ServiceGrantAttribution, item)
            for item in state["service_grants"]
        ),
        settling_deadline=_text(
            state["settling_deadline"], "orphan settling deadline"
        ),
    )


def _strict_snapshot_arm(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    request: Mapping[str, object],
) -> object:
    from .task12_snapshot_cleanup import SnapshotCapture, SnapshotSchedule

    exact = _exact_fields(
        request,
        (
            "plan",
            "capture",
            "schedule",
            "domain",
            "operation_identity_sha256",
            "owner_nonce_capsule",
        ),
        "snapshot arm",
    )
    plan, common = _snapshot_common_request(
        {
            key: exact[key]
            for key in (
                "plan",
                "domain",
                "operation_identity_sha256",
                "owner_nonce_capsule",
            )
        },
        ports=ports,
        invocation=invocation,
    )
    capture = _materialize_dataclass(SnapshotCapture, exact["capture"])
    schedule = _materialize_dataclass(SnapshotSchedule, exact["schedule"])
    resolution = _snapshot_coordinator(ports).arm(
        plan=plan,
        capture=capture,
        schedule=schedule,
        **common,
    )
    scheduler = _SchedulerClient(ports)
    schedule_result = (
        scheduler.arm(
            schedule=schedule,
            operation_identity_sha256=common["operation_identity_sha256"],
        )
        if resolution.may_issue_external_side_effect
        else scheduler.reconcile(
            schedule=schedule,
            operation_identity_sha256=common["operation_identity_sha256"],
        )
    )
    return {"transition": resolution, "schedule": schedule_result}


def make_retained_execution_observer_coordinator(
    deployment: Task12LambdaDeployment, *, _session: Optional[object] = None
) -> Callable[[Task12LambdaInvocation], object]:
    ports = _ports(deployment, _session)

    def coordinate(invocation: Task12LambdaInvocation) -> object:
        from .task12_runtime import (
            RuntimeReadBoundaries,
            RuntimeScan,
            collect_runtime_scan,
        )

        state = ports.retained_input(invocation)
        strict = _strict_operation_result(
            ports=ports, invocation=invocation, state=state
        )
        if strict is not None:
            return strict
        _closed_state(state, extra_fields=("correlation_identity_sha256", "runtime_scan"))
        if invocation.authenticated_execution is None:
            _observe_exact_execution(ports, invocation)
        scan = _materialize_dataclass(RuntimeScan, state["runtime_scan"])
        reader = _StaticRuntimeReader(scan)
        result = collect_runtime_scan(
            boundaries=RuntimeReadBoundaries(
                request_reader=reader,
                job_reader=reader,
                controller_reader=reader,
                worker_reader=reader,
                allocation_reader=reader,
                spend_reader=reader,
            ),
            correlation_identity_sha256=_sha(
                state["correlation_identity_sha256"], "runtime correlation"
            ),
        )
        return result

    return coordinate


def _write_candidate(
    *,
    ports: _AwsPorts,
    invocation: Task12LambdaInvocation,
    writer_kind: str,
    authority_domain: str,
    record: object,
    action: object,
    audit: object,
) -> object:
    from .task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
        Task12WriterServices,
        build_retained_writer_candidate,
        write_retained_candidate,
    )

    candidate = build_retained_writer_candidate(
        writer_kind=writer_kind,
        campaign_bucket=_text(
            ports.deployment.role_coordinates.get("campaign_bucket"),
            "campaign bucket",
        ),
        activation_id=invocation.activation_id,
        generation=invocation.generation,
        authority_domain=authority_domain,
        record=record,
    )
    exact_audit = _materialize_dataclass(RetainedWriterAuditAuthority, audit)
    result = write_retained_candidate(
        services=Task12WriterServices(boundary=_RetainedWriteBoundary(ports)),
        candidate=candidate,
        action=_materialize_dataclass(RetainedWriterActionAuthority, action),
        audit=exact_audit,
    )
    if writer_kind in {"TerminalV2", "SupportPlaneFinalized", "H1GDrained"}:
        _persist_versioned_writer_control(
            ports=ports,
            candidate=candidate,
            result=result,
            published_at=exact_audit.observed_at,
        )
    return result


def _persist_versioned_writer_control(
    *,
    ports: _AwsPorts,
    candidate: object,
    result: object,
    published_at: object,
) -> None:
    """Conditionally persist a proven S3 version, or adopt only exact bytes."""

    from .dynamodb import decode_item, encode_item
    from .records import ledger_pk, ledger_sk
    from .task12_writers import build_versioned_writer_control

    control = build_versioned_writer_control(
        candidate=candidate,
        result=result,
        published_at=published_at,
    )
    partition_key = ledger_pk("glm52-sky-20260724")
    sort_key = ledger_sk(
        "glm52_task12_versioned_writer_control_v1",
        activation_id=control["activation_id"],
        generation=control["generation"],
        writer_kind=control["writer_kind"],
    )
    client = ports.client("dynamodb")
    table_name = _text(
        ports.deployment.role_coordinates.get("ledger_table_name"),
        "ledger table",
    )
    try:
        response = client.put_item(
            TableName=table_name,
            Item=encode_item({"PK": partition_key, "SK": sort_key, **control}),
            ConditionExpression=(
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            ),
            ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
            ReturnConsumedCapacity="NONE",
        )
    except Exception:
        response = None
    if (
        type(response) is dict
        and _authenticated_metadata(response.get("ResponseMetadata"))
    ):
        return
    try:
        readback = client.get_item(
            TableName=table_name,
            Key=encode_item({"PK": partition_key, "SK": sort_key}),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
    except Exception as exc:
        raise Task12LambdaAdapterError(
            "versioned writer control was not durably proven"
        ) from exc
    if (
        type(readback) is not dict
        or not _authenticated_metadata(readback.get("ResponseMetadata"))
        or type(readback.get("Item")) is not dict
    ):
        _fail("versioned writer control readback is unauthenticated")
    actual = decode_item(readback["Item"])
    if actual.pop("PK", None) != partition_key or actual.pop("SK", None) != sort_key:
        _fail("versioned writer control key drifted")
    if actual != control:
        _fail("versioned writer control adopted foreign bytes")


def make_retained_terminal_v2_coordinator(
    deployment: Task12LambdaDeployment, *, _session: Optional[object] = None
) -> Callable[[Task12LambdaInvocation], object]:
    ports = _ports(deployment, _session)

    def coordinate(invocation: Task12LambdaInvocation) -> object:
        from .task12_terminal import assemble_terminal_v2

        state = ports.retained_input(invocation)
        strict = _strict_operation_result(
            ports=ports, invocation=invocation, state=state
        )
        if strict is not None:
            return strict
        _closed_state(
            state,
            extra_fields=(
                "base_record",
                "allocations",
                "worker_launch_evidence",
                "worker_launch_liabilities",
                "request_evidence",
                "late_allocations",
                "action",
                "audit",
            ),
        )
        terminal = assemble_terminal_v2(
            base_record=state["base_record"],
            allocations=state["allocations"],
            worker_launch_evidence=state["worker_launch_evidence"],
            worker_launch_liabilities=state["worker_launch_liabilities"],
            request_evidence=state["request_evidence"],
            late_allocations=state["late_allocations"],
        )
        return _write_candidate(
            ports=ports,
            invocation=invocation,
            writer_kind="TerminalV2",
            authority_domain="RECOVERY",
            record=terminal,
            action=state["action"],
            audit=state["audit"],
        )

    return coordinate


def make_retained_finalizer_coordinator(
    deployment: Task12LambdaDeployment, *, _session: Optional[object] = None
) -> Callable[[Task12LambdaInvocation], object]:
    ports = _ports(deployment, _session)

    def coordinate(invocation: Task12LambdaInvocation) -> object:
        from .task12_runtime import RuntimeScan, prove_runtime_terminal

        state = ports.retained_input(invocation)
        strict = _strict_operation_result(
            ports=ports, invocation=invocation, state=state
        )
        if strict is not None:
            return strict
        _closed_state(
            state,
            extra_fields=(
                "first_runtime_scan",
                "second_runtime_scan",
                "minimum_quiet_seconds",
                "record",
                "action",
                "audit",
            ),
        )
        proof = prove_runtime_terminal(
            first=_materialize_dataclass(
                RuntimeScan, state["first_runtime_scan"]
            ),
            second=_materialize_dataclass(
                RuntimeScan, state["second_runtime_scan"]
            ),
            minimum_quiet_seconds=_positive(
                state["minimum_quiet_seconds"], "runtime quiet interval"
            ),
        )
        write = _write_candidate(
            ports=ports,
            invocation=invocation,
            writer_kind="SupportPlaneFinalized",
            authority_domain="FINALIZATION",
            record=state["record"],
            action=state["action"],
            audit=state["audit"],
        )
        return {"runtime_terminal_proof": proof, "write_result": write}

    return coordinate


def make_retained_h1g_drained_coordinator(
    deployment: Task12LambdaDeployment, *, _session: Optional[object] = None
) -> Callable[[Task12LambdaInvocation], object]:
    ports = _ports(deployment, _session)

    def coordinate(invocation: Task12LambdaInvocation) -> object:
        from .task12_orphan_audit import (
            MarkerLastPrerequisites,
            build_h1g_drained_prerequisites,
        )

        state = ports.retained_input(invocation)
        strict = _strict_operation_result(
            ports=ports, invocation=invocation, state=state
        )
        if strict is not None:
            return strict
        _closed_state(
            state,
            extra_fields=("prerequisites", "record", "action", "audit"),
        )
        proof = build_h1g_drained_prerequisites(
            _materialize_dataclass(
                MarkerLastPrerequisites, state["prerequisites"]
            )
        )
        write = _write_candidate(
            ports=ports,
            invocation=invocation,
            writer_kind="H1GDrained",
            authority_domain="FINALIZATION",
            record=state["record"],
            action=state["action"],
            audit=state["audit"],
        )
        return {"marker_last_proof": proof, "write_result": write}

    return coordinate


def make_retained_worker_drain_coordinator(
    deployment: Task12LambdaDeployment, *, _session: Optional[object] = None
) -> Callable[[Task12LambdaInvocation], object]:
    ports = _ports(deployment, _session)

    def coordinate(invocation: Task12LambdaInvocation) -> object:
        from .task12_worker_drain import (
            WorkerDrainCandidate,
            WorkerDrainServices,
            build_task12_worker_drain_authority,
            dispatch_worker_drain,
        )
        from .task12_writers import (
            RetainedWriterActionAuthority,
            RetainedWriterAuditAuthority,
        )

        state = ports.retained_input(invocation)
        strict = _strict_operation_result(
            ports=ports, invocation=invocation, state=state
        )
        if strict is not None:
            return strict
        _closed_state(state, extra_fields=("candidate", "action", "audit"))
        candidate = _materialize_dataclass(
            WorkerDrainCandidate, state["candidate"]
        )
        if (
            candidate.document_name
            != deployment.role_coordinates["worker_drain_document_name"]
            or candidate.document_version
            != deployment.role_coordinates["worker_drain_document_version"]
        ):
            _fail("worker-drain candidate lost deployed document pin")
        authority = build_task12_worker_drain_authority(
            candidate=candidate,
            action=_materialize_dataclass(
                RetainedWriterActionAuthority, state["action"]
            ),
            audit=_materialize_dataclass(
                RetainedWriterAuditAuthority, state["audit"]
            ),
        )
        return dispatch_worker_drain(
            services=WorkerDrainServices(ssm=ports.client("ssm")),
            authority=authority,
        )

    return coordinate


def make_retained_operator_disposition_coordinator(
    deployment: Task12LambdaDeployment, *, _session: Optional[object] = None
) -> Callable[[Task12LambdaInvocation], object]:
    ports = _ports(deployment, _session)

    def coordinate(invocation: Task12LambdaInvocation) -> object:
        state = ports.retained_input(invocation)
        strict = _strict_operation_result(
            ports=ports, invocation=invocation, state=state
        )
        if strict is not None:
            return strict
        _closed_state(state, extra_fields=("record", "action", "audit"))
        return _write_candidate(
            ports=ports,
            invocation=invocation,
            writer_kind="OperatorDisposition",
            authority_domain="OPERATOR_DISPOSITION",
            record=state["record"],
            action=state["action"],
            audit=state["audit"],
        )

    return coordinate


def make_retained_orphan_audit_coordinator(
    deployment: Task12LambdaDeployment, *, _session: Optional[object] = None
) -> Callable[[Task12LambdaInvocation], object]:
    ports = _ports(deployment, _session)

    def coordinate(invocation: Task12LambdaInvocation) -> object:
        from .task12_orphan_audit import (
            DirectGrantAttribution,
            KmsGrantIdentity,
            KmsGrantScan,
            ResourceInventoryScan,
            RetainedResource,
            ServiceGrantAttribution,
            audit_orphans,
        )

        state = ports.retained_input(invocation)
        strict = _strict_operation_result(
            ports=ports, invocation=invocation, state=state
        )
        if strict is not None:
            return strict
        _closed_state(
            state,
            extra_fields=(
                "expected_retained",
                "inventory",
                "retained_grant_baseline",
                "pre_cleanup_grants",
                "direct_grants",
                "service_grants",
                "settling_deadline",
            ),
        )
        expected = tuple(
            _materialize_dataclass(RetainedResource, item)
            for item in state["expected_retained"]
        )
        baseline = tuple(
            _materialize_dataclass(KmsGrantIdentity, item)
            for item in state["retained_grant_baseline"]
        )
        direct = tuple(
            _materialize_dataclass(DirectGrantAttribution, item)
            for item in state["direct_grants"]
        )
        service = tuple(
            _materialize_dataclass(ServiceGrantAttribution, item)
            for item in state["service_grants"]
        )
        return audit_orphans(
            expected_retained=expected,
            inventory=_materialize_dataclass(
                ResourceInventoryScan, state["inventory"]
            ),
            retained_grant_baseline=baseline,
            pre_cleanup_grants=_materialize_dataclass(
                KmsGrantScan, state["pre_cleanup_grants"]
            ),
            final_grants=_read_live_kms_grants(ports),
            direct_grants=direct,
            service_grants=service,
            settling_deadline=_text(
                state["settling_deadline"], "orphan settling deadline"
            ),
        )

    return coordinate


def make_retained_snapshot_cleanup_coordinator(
    deployment: Task12LambdaDeployment, *, _session: Optional[object] = None
) -> Callable[[Task12LambdaInvocation], object]:
    ports = _ports(deployment, _session)

    def coordinate(invocation: Task12LambdaInvocation) -> object:
        from .dynamodb import DynamoLedgerAdapter
        from .task12_retained_state import SnapshotCleanupTransitionPlan
        from .task12_snapshot_cleanup import (
            SnapshotCapture,
            SnapshotCleanupCoordinator,
            SnapshotSchedule,
        )

        state = ports.retained_input(invocation)
        strict = _strict_operation_result(
            ports=ports, invocation=invocation, state=state
        )
        if strict is not None:
            return strict
        operation = state.get("operation")
        common_fields = (
            "operation",
            "plan",
            "domain",
            "operation_identity_sha256",
            "raw_owner_nonce_hex",
        )
        if operation == "ARM_SCHEDULE":
            extra_fields = common_fields + ("capture", "schedule")
        elif operation == "DELETE_ONCE":
            extra_fields = common_fields + ("observed_at",)
        elif operation == "RECONCILE_READBACK":
            extra_fields = common_fields + ("capture", "schedule")
        else:
            _fail("snapshot cleanup operation is not closed")
        _closed_state(
            state,
            extra_fields=extra_fields,
        )
        plan = _materialize_dataclass(
            SnapshotCleanupTransitionPlan, state["plan"]
        )
        coordinator = SnapshotCleanupCoordinator(
            adapter=DynamoLedgerAdapter(
                client=_AuthenticatedDynamoClient(
                    ports.client("dynamodb")
                ),
                table_name=_text(
                    deployment.role_coordinates.get("ledger_table_name"),
                    "ledger table",
                ),
            ),
            snapshot_client=_SnapshotClient(ports),
        )
        scheduler = _SchedulerClient(ports)
        common = {
            "plan": plan,
            "domain": _text(state["domain"], "snapshot cleanup domain"),
            "operation_identity_sha256": _sha(
                state["operation_identity_sha256"],
                "snapshot cleanup operation",
            ),
            "raw_owner_nonce": bytes.fromhex(
                _text(state["raw_owner_nonce_hex"], "snapshot cleanup nonce")
            ),
        }
        if operation == "ARM_SCHEDULE":
            capture = _materialize_dataclass(
                SnapshotCapture, state["capture"]
            )
            schedule = _materialize_dataclass(
                SnapshotSchedule, state["schedule"]
            )
            resolution = coordinator.arm(
                capture=capture,
                schedule=schedule,
                **common,
            )
            if resolution.may_issue_external_side_effect:
                schedule_result = scheduler.arm(
                    schedule=schedule,
                    operation_identity_sha256=common[
                        "operation_identity_sha256"
                    ],
                )
            else:
                schedule_result = scheduler.reconcile(
                    schedule=schedule,
                    operation_identity_sha256=common[
                        "operation_identity_sha256"
                    ],
                )
            return {
                "transition": resolution,
                "schedule": schedule_result,
            }
        if operation == "DELETE_ONCE":
            return coordinator.delete_once(
                observed_at=_text(
                    state["observed_at"], "snapshot cleanup observation"
                ),
                **common,
            )
        result = coordinator.reconcile_readback(**common)
        if result.target_state in {"ALREADY_ABSENT", "DELETED"} and (
            not result.snapshot_present
        ):
            schedule = _materialize_dataclass(
                SnapshotSchedule, state["schedule"]
            )
            return {
                "reconciliation": result,
                "schedule": scheduler.delete(schedule=schedule),
            }
        return result

    return coordinate


__all__ = [
    "Task12LambdaAdapterError",
    "Task12LambdaDeployment",
    "Task12LambdaInvocation",
    "build_lambda_handler",
    "make_retained_execution_observer_coordinator",
    "make_retained_finalizer_coordinator",
    "make_retained_h1g_drained_coordinator",
    "make_retained_operator_disposition_coordinator",
    "make_retained_orphan_audit_coordinator",
    "make_retained_snapshot_cleanup_coordinator",
    "make_retained_terminal_v2_coordinator",
    "make_retained_worker_drain_coordinator",
]
