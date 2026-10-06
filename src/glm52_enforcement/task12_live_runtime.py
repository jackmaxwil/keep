"""Live Task 12 runtime observations from production authorities.

The execution observer runs before TerminalV2 exists.  Runtime truth therefore
comes from the published NumericBinding observation Lambda, exact retained
DynamoDB records, a full versioned walk of the append-only GPU spend ledger,
EC2 readback, and the pinned Step Functions execution.  No caller-provided
projection or future terminal artifact is accepted.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
import base64
import hashlib
import json
import re
import time
from typing import Mapping

from .canonical import canonical_json_bytes, canonical_sha256
from .dynamodb import decode_item, encode_item
from .records import canonical_record_identity, ledger_pk, ledger_sk, validate_record
from .spend_authority import (
    APPROVED_GPU_COST_USD,
    APPROVED_GPU_RUNTIME_SECONDS,
    GPU_RESERVE_COST_USD,
    GPU_RESERVE_SECONDS,
    SpendAuthorityError,
    _INSTANCE_ID,
    _RECORD_NAME,
    _cost,
    _ledger_genesis,
    _object,
    _parse_time,
    _sha,
    _sha_raw,
    _validate_descriptor,
    validate_gpu_spend_approval,
)
from .task12_runtime import (
    CompleteRuntimeRead,
    RuntimeEntity,
    RuntimeReadBoundaries,
    RuntimeScan,
    RuntimeTerminalProof,
    SpendRead,
    collect_runtime_scan,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
QUIET_SECONDS = 60
EXECUTION_TERMINAL_POLL_SECONDS = 5
EXECUTION_TERMINAL_MAX_POLLS = 36

RECONCILE = "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"
PROVE_TERMINAL = (
    "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT"
)
SUPPORTED_OPERATIONS = frozenset({RECONCILE, PROVE_TERMINAL})

_EXPECTED_SOURCES = {
    RECONCILE: {
        "activation_index": "glm52_production_activation_index",
        "control": "glm52_production_control",
        "execution": "glm52_production_execution",
    },
    PROVE_TERMINAL: {
        "execution": "glm52_production_execution",
        "recovery_control": "glm52_production_recovery_control",
    },
}
_FAMILIES = (
    ("WORKER_LAUNCH#", "glm52_production_worker_launch"),
    (
        "WORKER_LAUNCH_LIABILITY#",
        "glm52_production_worker_launch_liability",
    ),
    (
        "WORKER_LAUNCH_LIABILITY_SETTLEMENT#",
        "glm52_production_worker_launch_liability_settlement",
    ),
    (
        "POST_TERMINAL_ALLOCATION#",
        "glm52_production_post_terminal_allocation",
    ),
)
_TERMINAL_EXECUTION_STATES = frozenset(
    {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
)
_PROOF_EXECUTION_TERMINAL_STATES = (
    _TERMINAL_EXECUTION_STATES
    | {"START_POSSIBLY_SENT_UNRESOLVED_INCIDENT"}
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_SUPPORT_EXECUTION_ARN = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:execution:"
    r"keep-glm52-h1g-support:[A-Za-z0-9_-]+$"
)
_SUPPORT_VERSION_ARN = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:stateMachine:"
    r"keep-glm52-h1g-support:[1-9][0-9]*$"
)
_FINALIZATION_TRIGGER_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "activation_id",
    "generation",
    "generation_text",
    "reason",
    "support_state_identity_sha256",
    "support_execution_arn",
    "support_state_machine_version_arn",
    "terminal_v2_identity_sha256",
    "canonical_identity_sha256",
}


class Task12LiveRuntimeError(ValueError):
    """A production runtime authority is absent, ambiguous, or divergent."""


def _fail(message: str) -> None:
    raise Task12LiveRuntimeError(message)


def _sleep(seconds: int) -> None:
    time.sleep(seconds)


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(label + " must be one exact nonempty string")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail(label + " must be one lowercase SHA-256")
    return value


def _positive(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail(label + " must be a positive integer")
    return value


def _utc(value: object, label: str) -> str:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        _fail(label + " must be canonical whole-second UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Task12LiveRuntimeError(label + " must be canonical UTC") from exc
    parsed.replace(tzinfo=timezone.utc)
    return value


def _metadata(response: object, label: str) -> Mapping[str, object]:
    metadata = response.get("ResponseMetadata") if type(response) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        _fail(label + " metadata must authenticate RetryAttempts == 0")
    return metadata


def _error_metadata(
    response: object,
    *,
    label: str,
    expected_code: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    metadata = response.get("ResponseMetadata") if type(response) is dict else None
    error = response.get("Error") if type(response) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 400
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
        or type(error) is not dict
        or set(error) != {"Code", "Message"}
        or error.get("Code") != expected_code
        or type(error.get("Message")) is not str
        or not error["Message"]
    ):
        _fail(label + " is not one authenticated zero-retry service error")
    return metadata, error


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Decimal):
        return format(value, "f")
    return value


def _roles(ports: object) -> Mapping[str, object]:
    deployment = getattr(ports, "deployment", None)
    roles = getattr(deployment, "role_coordinates", None)
    if not isinstance(roles, Mapping):
        _fail("runtime deployment role coordinates are absent")
    return roles


def _client(ports: object, service: str, label: str) -> object:
    try:
        value = ports.client(service)
    except Exception as exc:
        raise Task12LiveRuntimeError(label + " client is absent") from exc
    if value is None:
        _fail(label + " client is absent")
    return value


def _validate_invocation(
    invocation: object, operation_kind: str
) -> tuple[str, int, int]:
    if operation_kind not in SUPPORTED_OPERATIONS:
        _fail("runtime operation is not supported")
    if getattr(invocation, "operation_kind", None) != operation_kind:
        _fail("runtime invocation operation drifted")
    activation_id = _text(
        getattr(invocation, "activation_id", None), "activation id"
    )
    activation_ordinal = _positive(
        getattr(invocation, "activation_ordinal", None),
        "activation ordinal",
    )
    generation = _positive(
        getattr(invocation, "generation", None), "generation"
    )
    if getattr(invocation, "generation_text", None) != f"{generation:08d}":
        _fail("runtime invocation generation text drifted")
    return activation_id, activation_ordinal, generation


def _validate_finalization_trigger(
    *,
    invocation: object,
    ports: object,
) -> Mapping[str, object]:
    """Authenticate the one deployed EventBridge trigger against its row."""

    activation_id, _activation_ordinal, generation = _validate_invocation(
        invocation, RECONCILE
    )
    detail = getattr(invocation, "operation_input", None)
    if type(detail) is not dict or set(detail) != _FINALIZATION_TRIGGER_FIELDS:
        _fail("retained lifecycle trigger is not the closed finalization shape")
    identity = detail.get("canonical_identity_sha256")
    body = dict(detail)
    body.pop("canonical_identity_sha256", None)
    if (
        detail.get("schema_version") != 1
        or detail.get("record_type")
        != "glm52_task12_retained_lifecycle_request_v1"
        or detail.get("run_id") != RUN_ID
        or detail.get("activation_id") != activation_id
        or detail.get("generation") != generation
        or detail.get("generation_text") != f"{generation:08d}"
        or detail.get("reason") != "SUPPORT_FINALIZATION_REQUESTED"
        or _SHA.fullmatch(
            str(detail.get("support_state_identity_sha256", ""))
        )
        is None
        or _SHA.fullmatch(
            str(detail.get("terminal_v2_identity_sha256", ""))
        )
        is None
        or _SUPPORT_EXECUTION_ARN.fullmatch(
            str(detail.get("support_execution_arn", ""))
        )
        is None
        or _SUPPORT_VERSION_ARN.fullmatch(
            str(detail.get("support_state_machine_version_arn", ""))
        )
        is None
        or _SHA.fullmatch(str(identity or "")) is None
        or identity != canonical_sha256(body)
    ):
        _fail("retained lifecycle finalization trigger is noncanonical")

    table_name = _text(
        _roles(ports).get("ledger_table_name"), "ledger table"
    )
    key = {
        "PK": ledger_pk(RUN_ID),
        "SK": (
            "ACTIVATION#"
            + activation_id
            + "#FINALIZATION_REQUESTED"
        ),
    }
    response = _client(ports, "dynamodb", "DynamoDB").get_item(
        TableName=table_name,
        Key=encode_item(key),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(response, "DynamoDB FINALIZATION_REQUESTED GetItem")
    item = response.get("Item")
    if type(item) is not dict:
        _fail("durable FINALIZATION_REQUESTED row is absent")
    stored = decode_item(item)
    if set(stored) != {"PK", "SK", "record"}:
        _fail("durable FINALIZATION_REQUESTED row shape drifted")
    raw = stored.get("record")
    expected_raw = canonical_json_bytes(detail).decode("ascii")
    if (
        stored.get("PK") != key["PK"]
        or stored.get("SK") != key["SK"]
        or type(raw) is not str
        or raw != expected_raw
    ):
        _fail("durable FINALIZATION_REQUESTED bytes drifted")
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise Task12LiveRuntimeError(
            "durable FINALIZATION_REQUESTED bytes are not JSON"
        ) from exc
    if decoded != detail:
        _fail("durable FINALIZATION_REQUESTED identity drifted")
    return detail


def _reconstruct_seal_bound_running_execution(
    execution: Mapping[str, object],
) -> dict[str, object]:
    if execution.get("state") not in _TERMINAL_EXECUTION_STATES:
        _fail("terminal execution has no reconstructible RUNNING predecessor")
    revision = execution.get("revision")
    if type(revision) is not int or revision < 2:
        _fail("terminal execution revision has no predecessor")
    prior = {
        **execution,
        "state": "RUNNING",
        "terminal_status": None,
        "terminal_observed_at": None,
        "describe_execution_request_id": None,
        "describe_execution_response_sha256": None,
        "terminal_body_sha256": None,
        "revision": revision - 1,
    }
    try:
        exact_prior = validate_record(
            "glm52_production_execution", prior
        )
        from .transitions import validate_transition

        validate_transition(
            "glm52_production_execution", exact_prior, execution
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveRuntimeError(
            "terminal execution predecessor is noncanonical"
        ) from exc
    prior_identity = canonical_record_identity(
        "glm52_production_execution", exact_prior
    )
    terminal_body = {
        "schema_version": 1,
        "record_type": "glm52_production_execution_terminal_observation_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": execution["activation_id"],
        "activation_ordinal": execution["activation_ordinal"],
        "epoch": execution["epoch"],
        "execution_arn": execution["expected_execution_arn"],
        "state_machine_version_arn": (
            execution["expected_state_machine_version_arn"]
        ),
        "terminal_status": execution["terminal_status"],
        "terminal_observed_at": execution["terminal_observed_at"],
        "describe_execution_request_id": (
            execution["describe_execution_request_id"]
        ),
        "describe_execution_response_sha256": (
            execution["describe_execution_response_sha256"]
        ),
        "prior_execution_identity_sha256": prior_identity,
    }
    if execution.get("terminal_body_sha256") != canonical_sha256(
        terminal_body
    ):
        _fail("terminal execution predecessor body identity drifted")
    return exact_prior


def _validate_live_sources(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    activation_id, activation_ordinal, _generation = _validate_invocation(
        invocation, operation_kind
    )
    expected = _EXPECTED_SOURCES[operation_kind]
    if type(live_sources) is not dict or set(live_sources) != set(expected):
        _fail(operation_kind + " live source set drifted")
    exact: dict[str, dict[str, object]] = {}
    campaign_identity: str | None = None
    for alias, record_type in expected.items():
        try:
            value = validate_record(record_type, live_sources.get(alias))
        except (TypeError, ValueError) as exc:
            raise Task12LiveRuntimeError(alias + " source drifted") from exc
        if (
            value.get("run_id") != RUN_ID
            or value.get("account_id") != ACCOUNT_ID
            or value.get("region") != REGION
        ):
            _fail(alias + " source drifted")
        if record_type == "glm52_production_activation_index":
            if (
                value.get("current_activation_id") != activation_id
                or value.get("current_activation_ordinal")
                != activation_ordinal
            ):
                _fail(alias + " source drifted")
        elif (
            value.get("activation_id") != activation_id
            or value.get("activation_ordinal") != activation_ordinal
        ):
            _fail(alias + " source drifted")
        identity = _digest(
            value.get("campaign_identity_sha256"), "campaign identity"
        )
        if campaign_identity is None:
            campaign_identity = identity
        elif identity != campaign_identity:
            _fail(alias + " campaign source drifted")
        exact[alias] = value
    execution = exact["execution"]
    if operation_kind == RECONCILE:
        control = exact["control"]
        if (
            control.get("active_epoch") != execution.get("epoch")
            or control.get("active_execution_arn")
            != execution.get("expected_execution_arn")
            or control.get("active_state_machine_version_arn")
            != execution.get("expected_state_machine_version_arn")
        ):
            _fail("control and execution source drifted")
    else:
        recovery = exact["recovery_control"]
        if recovery.get("state") != "OWNED":
            _fail("recovery control source is not sealed")
        sealed_identity = recovery.get(
            "support_execution_identity_sha256"
        )
        current_identity = canonical_record_identity(
            "glm52_production_execution", execution
        )
        if sealed_identity != current_identity:
            prior_execution = _reconstruct_seal_bound_running_execution(
                execution
            )
            if sealed_identity != canonical_record_identity(
                "glm52_production_execution", prior_execution
            ):
                _fail("recovery support execution source drifted")
    return exact


def _query_family(
    *,
    ports: object,
    table_name: str,
    activation_id: str,
    activation_ordinal: int,
    generation: int,
    campaign_identity_sha256: str,
    suffix: str,
    expected_record_type: str,
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    prefix = "ACTIVATION#" + activation_id + "#" + suffix
    client = _client(ports, "dynamodb", "DynamoDB")
    start_key: object = None
    seen_keys: set[tuple[str, str]] = set()
    seen_pages: set[str] = set()
    records: list[dict[str, object]] = []
    request_ids: list[str] = []
    while True:
        arguments: dict[str, object] = {
            "TableName": table_name,
            "KeyConditionExpression": "#pk = :pk AND begins_with(#sk, :prefix)",
            "ExpressionAttributeNames": {"#pk": "PK", "#sk": "SK"},
            "ExpressionAttributeValues": encode_item(
                {":pk": ledger_pk(RUN_ID), ":prefix": prefix}
            ),
            "ConsistentRead": True,
            "ScanIndexForward": True,
            "ReturnConsumedCapacity": "NONE",
        }
        if start_key is not None:
            arguments["ExclusiveStartKey"] = start_key
        response = client.query(**arguments)
        metadata = _metadata(response, "DynamoDB Query")
        request_ids.append(str(metadata["RequestId"]))
        items = response.get("Items")
        if (
            type(items) is not list
            or response.get("Count") != len(items)
            or response.get("ScannedCount") != len(items)
        ):
            _fail("DynamoDB Query page is incomplete")
        for encoded in items:
            if type(encoded) is not dict:
                _fail("DynamoDB Query item is not encoded")
            value = decode_item(encoded)
            pk = value.pop("PK", None)
            sk = value.pop("SK", None)
            if (
                pk != ledger_pk(RUN_ID)
                or type(sk) is not str
                or not sk.startswith(prefix)
                or (pk, sk) in seen_keys
                or value.get("record_type") != expected_record_type
            ):
                _fail("DynamoDB Query key or family drifted")
            try:
                exact = validate_record(
                    expected_record_type, value, pk=pk, sk=sk
                )
            except (TypeError, ValueError) as exc:
                raise Task12LiveRuntimeError(
                    "DynamoDB Query canonical record drifted"
                ) from exc
            if (
                exact.get("activation_id") != activation_id
                or exact.get("activation_ordinal") != activation_ordinal
                or exact.get("campaign_identity_sha256")
                != campaign_identity_sha256
                or (
                    "generation" in exact
                    and (
                        type(exact.get("generation")) is not int
                        or int(exact["generation"]) < 1
                        or int(exact["generation"]) > generation
                        or (
                            "generation_text" in exact
                            and exact.get("generation_text")
                            != f"{int(exact['generation']):08d}"
                        )
                    )
                )
            ):
                _fail("DynamoDB runtime family binding drifted")
            seen_keys.add((pk, sk))
            records.append(exact)
        continuation = response.get("LastEvaluatedKey")
        if continuation is None:
            break
        if type(continuation) is not dict:
            _fail("DynamoDB Query continuation is not closed")
        decoded = decode_item(continuation)
        if (
            set(decoded) != {"PK", "SK"}
            or decoded["PK"] != ledger_pk(RUN_ID)
            or type(decoded["SK"]) is not str
            or not decoded["SK"].startswith(prefix)
        ):
            _fail("DynamoDB Query continuation drifted")
        identity = canonical_sha256(decoded)
        if identity in seen_pages:
            _fail("DynamoDB Query continuation cycled")
        seen_pages.add(identity)
        start_key = continuation
    return tuple(records), tuple(request_ids)


def _query_runtime_families(
    *,
    ports: object,
    invocation: object,
    campaign_identity_sha256: str,
) -> tuple[dict[str, tuple[dict[str, object], ...]], tuple[str, ...]]:
    roles = _roles(ports)
    table_name = _text(roles.get("ledger_table_name"), "ledger table")
    activation_id, activation_ordinal, generation = _validate_invocation(
        invocation, getattr(invocation, "operation_kind")
    )
    values: dict[str, tuple[dict[str, object], ...]] = {}
    request_ids: list[str] = []
    for suffix, record_type in _FAMILIES:
        rows, ids = _query_family(
            ports=ports,
            table_name=table_name,
            activation_id=activation_id,
            activation_ordinal=activation_ordinal,
            generation=generation,
            campaign_identity_sha256=campaign_identity_sha256,
            suffix=suffix,
            expected_record_type=record_type,
        )
        values[record_type] = rows
        request_ids.extend(ids)
    _validate_family_graph(values)
    return values, tuple(request_ids)


def _validate_family_graph(
    families: Mapping[str, tuple[dict[str, object], ...]],
) -> None:
    launches = families["glm52_production_worker_launch"]
    liabilities = families[
        "glm52_production_worker_launch_liability"
    ]
    settlements = families[
        "glm52_production_worker_launch_liability_settlement"
    ]
    post_terminal = families[
        "glm52_production_post_terminal_allocation"
    ]
    launch_by_ordinal = {
        int(row["allocation_ordinal"]): row for row in launches
    }
    liability_by_ordinal = {
        int(row["allocation_ordinal"]): row for row in liabilities
    }
    settlement_by_ordinal = {
        int(row["allocation_ordinal"]): row for row in settlements
    }
    if (
        len(launch_by_ordinal) != len(launches)
        or len(liability_by_ordinal) != len(liabilities)
        or len(settlement_by_ordinal) != len(settlements)
    ):
        _fail("retained worker family ordinal is duplicated")
    liability_required_states = {
        "POSSIBLY_SENT",
        "INSTANCE_OBSERVED",
        "ALLOCATION_OPEN",
        "INSTANCE_TERMINAL",
        "ALLOCATION_CLOSED",
        "REJECTED_NO_INSTANCE",
        "MULTIPLE_INSTANCE_TOKEN_INCIDENT",
        "UNRESOLVED_LAUNCH_INCIDENT",
    }
    for ordinal, launch in launch_by_ordinal.items():
        if (
            launch.get("state") in liability_required_states
            and ordinal not in liability_by_ordinal
        ):
            _fail("sent worker launch lacks its liability record")
    for ordinal, liability in liability_by_ordinal.items():
        launch = launch_by_ordinal.get(ordinal)
        if (
            launch is None
            or liability.get("worker_launch_identity_sha256")
            != canonical_record_identity(
                "glm52_production_worker_launch", launch
            )
        ):
            _fail("worker launch liability binding drifted")
        for field in (
            "generation",
            "ec2_client_token",
            "launch_parameters_sha256",
            "expected_worker_tags_sha256",
            "gpu_liability_reserve_seconds",
            "gpu_liability_reserve_cost_usd",
            "ebs_liability_reserve_cost_usd",
            "residual_liability_approval_identity_sha256",
            "gpu_liability_reserve_ledger_identity_sha256",
        ):
            if liability.get(field) != launch.get(field):
                _fail("worker launch liability reserve custody drifted")
    for ordinal, settlement in settlement_by_ordinal.items():
        launch = launch_by_ordinal.get(ordinal)
        liability = liability_by_ordinal.get(ordinal)
        if (
            launch is None
            or liability is None
            or settlement.get("worker_launch_identity_sha256")
            != canonical_record_identity(
                "glm52_production_worker_launch", launch
            )
            or settlement.get("worker_launch_liability_identity_sha256")
            != canonical_record_identity(
                "glm52_production_worker_launch_liability", liability
            )
            or liability.get("settlement_identity_sha256")
            != canonical_record_identity(
                "glm52_production_worker_launch_liability_settlement",
                settlement,
            )
        ):
            _fail("worker liability settlement binding drifted")
    for ordinal, liability in liability_by_ordinal.items():
        settled = str(liability.get("state", "")).startswith("SETTLED_")
        if settled != (ordinal in settlement_by_ordinal):
            _fail("worker liability settlement cardinality drifted")
    for record in post_terminal:
        ordinal = int(record["allocation_ordinal"])
        launch = launch_by_ordinal.get(ordinal)
        liability = liability_by_ordinal.get(ordinal)
        if (
            launch is None
            or liability is None
            or record.get("worker_launch_identity_sha256")
            != canonical_record_identity(
                "glm52_production_worker_launch", launch
            )
            or record.get("worker_launch_liability_identity_sha256")
            != canonical_record_identity(
                "glm52_production_worker_launch_liability", liability
            )
        ):
            _fail("post-terminal allocation custody binding drifted")


def _correlation(
    *, operation_kind: str, sources: Mapping[str, Mapping[str, object]]
) -> str:
    return canonical_sha256(
        {
            "operation_kind": operation_kind,
            "sources": {
                alias: canonical_record_identity(
                    str(record["record_type"]), record
                )
                for alias, record in sorted(sources.items())
            },
        }
    )


def _numeric_observation(
    *,
    ports: object,
    invocation: object,
    correlation_identity_sha256: str,
) -> dict[str, object]:
    try:
        from .task12_live_cancellation import _numeric_observation as observe

        value = observe(
            ports=ports,
            invocation=invocation,
            correlation_identity_sha256=correlation_identity_sha256,
        )
    except Exception as exc:
        raise Task12LiveRuntimeError(
            "exact NumericBinding Lambda observation failed"
        ) from exc
    expected = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "correlation_identity_sha256",
        "task9_deployed_identity_sha256",
        "observed_at",
        "request_snapshot",
        "job_snapshot",
        "controller_snapshot",
        "request_transport",
        "job_transport",
        "canonical_identity_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or value.get("schema_version") != 1
        or value.get("record_type")
        != "glm52_task12_runtime_observation_v1"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or value.get("activation_id")
        != getattr(invocation, "activation_id", None)
        or value.get("correlation_identity_sha256")
        != correlation_identity_sha256
    ):
        _fail("NumericBinding observation schema is not closed")
    observed_at = _utc(value["observed_at"], "NumericBinding observed_at")
    controller = value["controller_snapshot"]
    if (
        type(controller) is not dict
        or set(controller) != {"members", "evidence_identity_sha256"}
        or type(controller["members"]) is not list
        or _SHA.fullmatch(controller.get("evidence_identity_sha256", ""))
        is None
    ):
        _fail("NumericBinding controller snapshot is not closed")
    controller_ids: list[str] = []
    for member in controller["members"]:
        if (
            type(member) is not dict
            or set(member)
            != {"identity", "state", "evidence_identity_sha256"}
            or type(member["identity"]) is not str
            or not member["identity"]
            or type(member["state"]) is not str
            or not member["state"]
            or _SHA.fullmatch(member["evidence_identity_sha256"]) is None
        ):
            _fail("NumericBinding controller member is not closed")
        controller_ids.append(member["identity"])
    if (
        controller_ids != sorted(controller_ids)
        or len(controller_ids) != len(set(controller_ids))
    ):
        _fail("NumericBinding controller member set is not exact")
    snapshots = (
        (
            "request_snapshot",
            "requests",
            "request_id",
            {
                "request_id",
                "state",
                "created_at",
                "updated_at",
                "evidence_identity_sha256",
            },
        ),
        (
            "job_snapshot",
            "jobs",
            "job_id",
            {
                "job_id",
                "request_id",
                "state",
                "evidence_identity_sha256",
            },
        ),
    )
    request_ids: set[str] = set()
    for snapshot_name, collection, identity_field, item_fields in snapshots:
        snapshot = value[snapshot_name]
        if (
            type(snapshot) is not dict
            or set(snapshot)
            != {
                "observed_at",
                "pagination_complete",
                collection,
                "controller_snapshot",
            }
            or snapshot["observed_at"] != observed_at
            or snapshot["pagination_complete"] is not True
            or snapshot["controller_snapshot"] != controller
            or type(snapshot[collection]) is not list
        ):
            _fail("NumericBinding " + collection + " snapshot is incomplete")
        identities: list[str] = []
        for item in snapshot[collection]:
            if (
                type(item) is not dict
                or set(item) != item_fields
                or type(item.get(identity_field)) is not str
                or not item[identity_field]
                or type(item.get("state")) is not str
                or not item["state"]
                or _SHA.fullmatch(
                    item.get("evidence_identity_sha256", "")
                )
                is None
            ):
                _fail("NumericBinding " + collection + " member is not closed")
            if collection == "requests":
                _utc(item["created_at"], "request created_at")
                _utc(item["updated_at"], "request updated_at")
                created = datetime.strptime(
                    item["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                )
                updated = datetime.strptime(
                    item["updated_at"], "%Y-%m-%dT%H:%M:%SZ"
                )
                observed = datetime.strptime(
                    observed_at, "%Y-%m-%dT%H:%M:%SZ"
                )
                if created > updated or updated > observed:
                    _fail("NumericBinding request chronology drifted")
                request_ids.add(item["request_id"])
            elif (
                re.fullmatch(r"[1-9][0-9]*", item["job_id"]) is None
                or item["request_id"] not in request_ids
            ):
                _fail("NumericBinding job binding is not exact")
            identities.append(item[identity_field])
        ordered = (
            sorted(identities)
            if collection == "requests"
            else sorted(identities, key=int)
        )
        if identities != ordered or len(identities) != len(set(identities)):
            _fail("NumericBinding " + collection + " set is not exact")
    for transport_name, snapshot_name in (
        ("request_transport", "request_snapshot"),
        ("job_transport", "job_snapshot"),
    ):
        transport = value[transport_name]
        if (
            type(transport) is not dict
            or set(transport)
            != {
                "request_id",
                "response_sha256",
                "tls_peer_certificate_sha256",
            }
            or type(transport["request_id"]) is not str
            or not transport["request_id"]
            or transport["response_sha256"]
            != canonical_sha256(value[snapshot_name])
            or _SHA.fullmatch(
                transport["tls_peer_certificate_sha256"]
            )
            is None
        ):
            _fail("NumericBinding transport evidence drifted")
    if (
        value["request_transport"]["tls_peer_certificate_sha256"]
        != value["job_transport"]["tls_peer_certificate_sha256"]
    ):
        _fail("NumericBinding mTLS peer identity drifted")
    return value


def _read_exact_execution(
    *,
    ports: object,
    execution: Mapping[str, object],
) -> dict[str, object]:
    epoch = _positive(execution.get("epoch"), "support execution epoch")
    pk = ledger_pk(RUN_ID)
    sk = ledger_sk(
        "glm52_production_execution",
        activation_id=_text(
            execution.get("activation_id"), "execution activation id"
        ),
        epoch=epoch,
    )
    response = _client(ports, "dynamodb", "DynamoDB").get_item(
        TableName=_text(
            _roles(ports).get("ledger_table_name"), "ledger table"
        ),
        Key=encode_item({"PK": pk, "SK": sk}),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(response, "DynamoDB execution GetItem")
    item = response.get("Item")
    if type(item) is not dict:
        _fail("DynamoDB execution readback is absent")
    value = decode_item(item)
    if value.pop("PK", None) != pk or value.pop("SK", None) != sk:
        _fail("DynamoDB execution readback key drifted")
    try:
        return validate_record(
            "glm52_production_execution",
            value,
            pk=pk,
            sk=sk,
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveRuntimeError(
            "DynamoDB execution readback is noncanonical"
        ) from exc


def _terminal_execution_candidate(
    *,
    execution: Mapping[str, object],
    status: str,
    observed_at: str,
    describe_request_id: str,
    describe_response_sha256: str,
) -> dict[str, object]:
    before_updated = _utc(
        execution.get("updated_at"), "execution updated_at"
    )
    if datetime.strptime(
        observed_at, "%Y-%m-%dT%H:%M:%SZ"
    ) < datetime.strptime(
        before_updated, "%Y-%m-%dT%H:%M:%SZ"
    ):
        _fail("terminal execution observation predates durable execution")
    terminal_body = {
        "schema_version": 1,
        "record_type": "glm52_production_execution_terminal_observation_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": execution["activation_id"],
        "activation_ordinal": execution["activation_ordinal"],
        "epoch": execution["epoch"],
        "execution_arn": execution["expected_execution_arn"],
        "state_machine_version_arn": (
            execution["expected_state_machine_version_arn"]
        ),
        "terminal_status": status,
        "terminal_observed_at": observed_at,
        "describe_execution_request_id": describe_request_id,
        "describe_execution_response_sha256": (
            describe_response_sha256
        ),
        "prior_execution_identity_sha256": canonical_record_identity(
            "glm52_production_execution", execution
        ),
    }
    candidate = {
        **execution,
        "state": status,
        "terminal_status": status,
        "terminal_observed_at": observed_at,
        "describe_execution_request_id": describe_request_id,
        "describe_execution_response_sha256": (
            describe_response_sha256
        ),
        "terminal_body_sha256": canonical_sha256(terminal_body),
        "revision": int(execution["revision"]) + 1,
        # Preserve the exact seal-time RUNNING preimage.  The authoritative
        # mutation time is terminal_observed_at; keeping updated_at stable
        # makes the prior identity reconstructible for fail-closed retries.
        "updated_at": execution["updated_at"],
    }
    try:
        from .transitions import validate_transition

        return validate_transition(
            "glm52_production_execution", execution, candidate
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveRuntimeError(
            "terminal execution transition is noncanonical"
        ) from exc


def _reconcile_terminal_execution(
    *,
    ports: object,
    execution: Mapping[str, object],
    status: str,
    observed_at: str,
    describe_request_id: str,
    describe_response_sha256: str,
) -> dict[str, object]:
    candidate = _terminal_execution_candidate(
        execution=execution,
        status=status,
        observed_at=observed_at,
        describe_request_id=describe_request_id,
        describe_response_sha256=describe_response_sha256,
    )
    before_fields = tuple(sorted(execution))
    changed_fields = tuple(
        sorted(
            field
            for field in candidate
            if candidate[field] != execution[field]
        )
    )
    names = {
        "#n_" + field: field
        for field in sorted(set(before_fields) | set(changed_fields))
    }
    before_values = {
        ":b_" + field: execution[field] for field in before_fields
    }
    after_values = {
        ":a_" + field: candidate[field] for field in changed_fields
    }
    key = {
        "PK": ledger_pk(RUN_ID),
        "SK": ledger_sk(
            "glm52_production_execution",
            activation_id=str(execution["activation_id"]),
            epoch=int(execution["epoch"]),
        ),
    }
    response_returned = False
    try:
        response = _client(
            ports, "dynamodb", "DynamoDB"
        ).update_item(
            TableName=_text(
                _roles(ports).get("ledger_table_name"), "ledger table"
            ),
            Key=encode_item(key),
            ConditionExpression=" AND ".join(
                "#n_" + field + " = :b_" + field
                for field in before_fields
            ),
            UpdateExpression="SET "
            + ", ".join(
                "#n_" + field + " = :a_" + field
                for field in changed_fields
            ),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=encode_item(
                {**before_values, **after_values}
            ),
            ReturnConsumedCapacity="NONE",
        )
    except Exception:
        # A zero-retry conditional update may have committed before the
        # response was lost.  Never resend it; exact consistent readback is
        # the only adoption path.
        pass
    else:
        _metadata(response, "DynamoDB execution UpdateItem")
        response_returned = True
    observed = _read_exact_execution(
        ports=ports, execution=execution
    )
    if observed == candidate:
        return observed
    if response_returned:
        _fail("terminal execution changed after acknowledged update")
    try:
        from .transitions import validate_transition

        adopted = validate_transition(
            "glm52_production_execution", execution, observed
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveRuntimeError(
            "terminal execution readback adopted foreign state"
        ) from exc
    if (
        adopted.get("state") != status
        or adopted.get("terminal_status") != status
    ):
        _fail("terminal execution readback adopted foreign status")
    return adopted


def _describe_support_execution(
    *,
    ports: object,
    execution: Mapping[str, object],
    observed_at: str,
) -> tuple[str, str, dict[str, object]]:
    execution_arn = _text(
        execution.get("expected_execution_arn"), "support execution ARN"
    )
    expected_version = _text(
        execution.get("expected_state_machine_version_arn"),
        "support state-machine version",
    )
    try:
        response = _client(
            ports, "stepfunctions", "Step Functions"
        ).describe_execution(executionArn=execution_arn)
    except Exception as exc:
        response = getattr(exc, "response", None)
        metadata, error = _error_metadata(
            response,
            label="Step Functions DescribeExecution error",
            expected_code="ExecutionDoesNotExist",
        )
        if (
            execution.get("state")
            != "START_POSSIBLY_SENT_UNRESOLVED_INCIDENT"
            or _SHA.fullmatch(
                str(
                    execution.get(
                        "unresolved_start_incident_body_sha256"
                    )
                )
            )
            is None
        ):
            _fail("Step Functions execution observation failed")
        evidence = canonical_sha256(
            {
                "execution_arn": execution_arn,
                "state_machine_version_arn": expected_version,
                "status": "START_POSSIBLY_SENT_UNRESOLVED_INCIDENT",
                "error_code": error["Code"],
                "request_id": metadata["RequestId"],
                "incident_body_sha256": execution[
                    "unresolved_start_incident_body_sha256"
                ],
            }
        )
        return (
            "START_POSSIBLY_SENT_UNRESOLVED_INCIDENT",
            evidence,
            dict(execution),
        )
    metadata = _metadata(response, "Step Functions DescribeExecution")
    status = response.get("status")
    if (
        response.get("executionArn") != execution_arn
        or response.get("stateMachineArn")
        != expected_version.rsplit(":", 1)[0]
        or response.get("stateMachineVersionArn") != expected_version
        or status
        not in {
            "RUNNING",
            "SUCCEEDED",
            "FAILED",
            "TIMED_OUT",
            "ABORTED",
        }
    ):
        _fail("Step Functions execution observation drifted")
    response_identity = canonical_sha256(
        {
            "execution_arn": execution_arn,
            "state_machine_arn": response["stateMachineArn"],
            "state_machine_version_arn": expected_version,
            "status": status,
            "request_id": metadata["RequestId"],
        }
    )
    durable_state = execution.get("state")
    exact_execution = dict(execution)
    if status == "RUNNING":
        if durable_state != "RUNNING":
            _fail("Step Functions execution and durable source drifted")
    elif status in _TERMINAL_EXECUTION_STATES:
        if durable_state == "RUNNING":
            exact_execution = _reconcile_terminal_execution(
                ports=ports,
                execution=execution,
                status=str(status),
                observed_at=observed_at,
                describe_request_id=str(metadata["RequestId"]),
                describe_response_sha256=response_identity,
            )
        elif (
            durable_state != status
            or execution.get("terminal_status") != status
        ):
            _fail("Step Functions execution and durable source drifted")
    else:  # pragma: no cover - status was closed above
        _fail("Step Functions execution and durable source drifted")
    evidence = canonical_sha256(
        {
            "execution_arn": execution_arn,
            "state_machine_version_arn": expected_version,
            "status": status,
            "request_id": metadata["RequestId"],
            "execution_identity_sha256": canonical_record_identity(
                "glm52_production_execution", exact_execution
            ),
        }
    )
    return str(status), evidence, exact_execution


def _list_current_versions(
    *,
    ports: object,
    bucket: str,
    prefix: str,
) -> tuple[tuple[str, str], ...]:
    client = _client(ports, "s3", "S3")
    key_marker: str | None = None
    version_marker: str | None = None
    seen_pages: set[tuple[str | None, str | None]] = set()
    current: dict[str, str] = {}
    current_deletes: set[str] = set()
    while True:
        args: dict[str, object] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        if key_marker is not None:
            args["KeyMarker"] = key_marker
        if version_marker is not None:
            args["VersionIdMarker"] = version_marker
        response = client.list_object_versions(**args)
        _metadata(response, "S3 ListObjectVersions")
        versions = response.get("Versions", [])
        deletes = response.get("DeleteMarkers", [])
        if type(versions) is not list or type(deletes) is not list:
            _fail("S3 version listing is malformed")
        for entry, is_delete in tuple(
            (item, False) for item in versions
        ) + tuple((item, True) for item in deletes):
            if (
                type(entry) is not dict
                or type(entry.get("Key")) is not str
                or not entry["Key"].startswith(prefix)
                or type(entry.get("VersionId")) is not str
                or not entry["VersionId"]
            ):
                _fail("S3 version listing contains a foreign entry")
            if entry.get("IsLatest") is True:
                key = entry["Key"]
                if key in current or key in current_deletes:
                    _fail("S3 version listing has duplicate current versions")
                if is_delete:
                    current_deletes.add(key)
                else:
                    current[key] = entry["VersionId"]
        truncated = response.get("IsTruncated")
        if truncated is False:
            break
        if truncated is not True:
            _fail("S3 version listing truncation state is absent")
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or (next_key, next_version) in seen_pages
        ):
            _fail("S3 version listing continuation drifted")
        seen_pages.add((next_key, next_version))
        key_marker, version_marker = next_key, next_version
    if current_deletes:
        _fail("S3 current spend namespace contains a delete marker")
    return tuple(sorted(current.items()))


def _get_exact_object(
    *,
    ports: object,
    bucket: str,
    key: str,
    version_id: str,
    expected_file_sha256: str | None,
    label: str,
) -> tuple[dict[str, object], bytes, str]:
    response = _client(ports, "s3", "S3").get_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    metadata = _metadata(response, label + " S3 GetObject")
    body = response.get("Body")
    read = getattr(body, "read", None)
    raw = read() if callable(read) else None
    if type(raw) is not bytes:
        _fail(label + " exact bytes are absent")
    checksum_sha256 = base64.b64encode(
        hashlib.sha256(raw).digest()
    ).decode("ascii")
    etag = response.get("ETag")
    single_part_etag = '"' + hashlib.md5(  # noqa: S324 - S3 identity
        raw, usedforsecurity=False
    ).hexdigest() + '"'
    service_checksum_matches = (
        response.get("ChecksumSHA256") == checksum_sha256
        or etag == single_part_etag
    )
    if (
        response.get("VersionId") != version_id
        or not service_checksum_matches
        or (
            expected_file_sha256 is not None
            and _sha_raw(raw) != expected_file_sha256
        )
    ):
        _fail(label + " exact version or checksum drifted")
    try:
        value = _object(raw, label=label)
    except SpendAuthorityError as exc:
        raise Task12LiveRuntimeError(label + " is not canonical") from exc
    return value, raw, str(metadata["RequestId"])


def _spend_authority(
    *,
    ports: object,
    invocation: object,
    observed_at: str,
    campaign_identity_sha256: str,
    families: Mapping[str, tuple[dict[str, object], ...]],
) -> tuple[
    SpendRead,
    tuple[dict[str, object], ...],
    set[str],
    tuple[str, ...],
]:
    roles = _roles(ports)
    bucket = _text(roles.get("campaign_bucket"), "campaign bucket")
    runtime_prefix = _text(
        roles.get("spend_runtime_prefix"), "spend runtime prefix"
    ).rstrip("/")
    expected_prefix = f"campaigns/{RUN_ID}/runtime"
    if runtime_prefix != expected_prefix:
        _fail("spend runtime prefix drifted")
    descriptor_key = _text(
        roles.get("campaign_descriptor_key"), "campaign descriptor key"
    )
    descriptor_version = _text(
        roles.get("campaign_descriptor_version_id"),
        "campaign descriptor version",
    )
    descriptor_file = _digest(
        roles.get("campaign_descriptor_file_sha256"),
        "campaign descriptor file identity",
    )
    approval_key = _text(
        roles.get("gpu_spend_approval_key"), "GPU spend approval key"
    )
    approval_version = _text(
        roles.get("gpu_spend_approval_version_id"),
        "GPU spend approval version",
    )
    approval_file = _digest(
        roles.get("gpu_spend_approval_file_sha256"),
        "GPU spend approval file identity",
    )
    if (
        not descriptor_key.startswith(f"campaigns/{RUN_ID}/")
        or not approval_key.startswith(f"campaigns/{RUN_ID}/authorities/")
    ):
        _fail("spend descriptor or approval coordinate is foreign")
    descriptor, _descriptor_raw, descriptor_request = _get_exact_object(
        ports=ports,
        bucket=bucket,
        key=descriptor_key,
        version_id=descriptor_version,
        expected_file_sha256=descriptor_file,
        label="campaign descriptor",
    )
    approval, _approval_raw, approval_request = _get_exact_object(
        ports=ports,
        bucket=bucket,
        key=approval_key,
        version_id=approval_version,
        expected_file_sha256=approval_file,
        label="GPU spend approval",
    )
    try:
        validate_gpu_spend_approval(approval)
        _validate_descriptor(
            descriptor,
            raw_sha256=descriptor_file,
            approval_sha256=approval_file,
        )
    except SpendAuthorityError as exc:
        raise Task12LiveRuntimeError(
            "GPU descriptor or approval authority drifted"
        ) from exc
    if descriptor.get("campaign_identity_sha256") != campaign_identity_sha256:
        _fail("campaign descriptor identity drifted from retained control")
    latest_key = runtime_prefix + "/GPU_SPEND_LEDGER_LATEST.json"
    latest_versions = _list_current_versions(
        ports=ports, bucket=bucket, prefix=latest_key
    )
    if len(latest_versions) != 1 or latest_versions[0][0] != latest_key:
        _fail("GPU spend latest marker current version is not singular")
    latest, latest_raw, latest_request = _get_exact_object(
        ports=ports,
        bucket=bucket,
        key=latest_key,
        version_id=latest_versions[0][1],
        expected_file_sha256=None,
        label="GPU spend latest marker",
    )
    latest_fields = {
        "schema_version",
        "record_type",
        "run_id",
        "gpu_spend_authority_sha256",
        "record_count",
        "record_keys",
        "latest_record_sha256",
        "ledger_sha256",
        "latest_body_sha256",
    }
    body = dict(latest)
    latest_identity = body.pop("latest_body_sha256", None)
    genesis = _ledger_genesis(approval_file)
    names = latest.get("record_keys")
    if (
        set(latest) != latest_fields
        or latest.get("schema_version") != 1
        or latest.get("record_type") != "glm52_gpu_spend_ledger_latest_v1"
        or latest.get("run_id") != RUN_ID
        or latest.get("gpu_spend_authority_sha256") != genesis
        or latest_identity != _sha(body)
        or type(names) is not list
        or latest.get("record_count") != len(names)
    ):
        _fail("GPU spend latest marker is invalid")
    records_prefix = runtime_prefix + "/spend-ledger/records/"
    listed = _list_current_versions(
        ports=ports, bucket=bucket, prefix=records_prefix
    )
    expected_keys = tuple(records_prefix + str(name) for name in names)
    if tuple(key for key, _version in listed) != expected_keys:
        _fail("GPU spend record namespace differs from latest inventory")
    prior = genesis
    chain_identities = {genesis}
    ledger = bytearray()
    active: tuple[str, str, str] | None = None
    intervals: list[dict[str, object]] = []
    prior_time: datetime | None = None
    seen_instances: set[str] = set()
    request_ids = [descriptor_request, approval_request, latest_request]
    observation = _parse_time(observed_at, field="runtime observed_at")
    for index, (name, key) in enumerate(zip(names, expected_keys)):
        match = _RECORD_NAME.fullmatch(str(name))
        if match is None or int(match.group("index")) != index:
            _fail("GPU spend immutable record name is invalid")
        record, raw, request_id = _get_exact_object(
            ports=ports,
            bucket=bucket,
            key=key,
            version_id=listed[index][1],
            expected_file_sha256=None,
            label=f"GPU spend immutable record {index}",
        )
        request_ids.append(request_id)
        event = record.get("event")
        expected_fields = {
            "record_type",
            "run_id",
            "approval_sha256",
            "event",
            "instance_id",
            "timestamp",
            "prior_record_sha256",
            "record_sha256",
        } | ({"job_id"} if event == "allocation_started" else set())
        record_body = dict(record)
        record_identity = record_body.pop("record_sha256", None)
        try:
            timestamp = _parse_time(
                record.get("timestamp"), field="record timestamp"
            )
        except SpendAuthorityError as exc:
            raise Task12LiveRuntimeError(
                "GPU spend record timestamp drifted"
            ) from exc
        instance_id = record.get("instance_id")
        if (
            set(record) != expected_fields
            or record.get("record_type") != "glm52_gpu_spend_event_v1"
            or record.get("run_id") != RUN_ID
            or record.get("approval_sha256") != approval_file
            or record.get("prior_record_sha256") != prior
            or record_identity != _sha(record_body)
            or record_identity != match.group("sha")
            or event != match.group("event")
            or type(instance_id) is not str
            or _INSTANCE_ID.fullmatch(instance_id) is None
            or timestamp > observation
            or (prior_time is not None and timestamp < prior_time)
        ):
            _fail("GPU spend immutable record is divergent")
        if event == "allocation_started":
            job_id = record.get("job_id")
            if (
                active is not None
                or type(job_id) is not str
                or not job_id
                or instance_id in seen_instances
            ):
                _fail("GPU allocation start is invalid")
            active = (job_id, instance_id, str(record["timestamp"]))
            seen_instances.add(instance_id)
        elif event == "allocation_ended":
            if active is None or active[1] != instance_id:
                _fail("GPU allocation end is noncontiguous")
            started = _parse_time(active[2], field="allocation start")
            seconds_value = (timestamp - started).total_seconds()
            if seconds_value < 0 or not seconds_value.is_integer():
                _fail("GPU allocation duration is not exact")
            seconds = int(seconds_value)
            intervals.append(
                {
                    "instance_id": instance_id,
                    "job_id": active[0],
                    "started_at": active[2],
                    "ended_at": str(record["timestamp"]),
                    "charged_seconds": seconds,
                    "charged_cost_usd": _cost(seconds),
                    "state": "CLOSED",
                    "ledger_identity_sha256": record_identity,
                }
            )
            active = None
        else:
            _fail("GPU spend event is invalid")
        prior = str(record_identity)
        chain_identities.add(prior)
        prior_time = timestamp
        ledger.extend(raw)
    if active is not None:
        started = _parse_time(active[2], field="allocation start")
        seconds_value = (observation - started).total_seconds()
        if seconds_value < 0 or not seconds_value.is_integer():
            _fail("open GPU allocation duration is not exact")
        seconds = int(seconds_value)
        intervals.append(
            {
                "instance_id": active[1],
                "job_id": active[0],
                "started_at": active[2],
                "ended_at": None,
                "charged_seconds": seconds,
                "charged_cost_usd": _cost(seconds),
                "state": "OPEN",
                "ledger_identity_sha256": prior,
            }
        )
    if (
        prior != latest.get("latest_record_sha256")
        or _sha_raw(bytes(ledger)) != latest.get("ledger_sha256")
    ):
        _fail("GPU spend full chain differs from latest marker")
    used_seconds = sum(
        int(interval["charged_seconds"])
        for interval in intervals
        if interval["state"] == "CLOSED"
    )
    open_seconds = sum(
        int(interval["charged_seconds"])
        for interval in intervals
        if interval["state"] == "OPEN"
    )
    charged_seconds = used_seconds + open_seconds
    charged_cost = _cost(charged_seconds)
    liabilities = families[
        "glm52_production_worker_launch_liability"
    ]
    held = [
        liability
        for liability in liabilities
        if not str(liability.get("state", "")).startswith("SETTLED_")
    ]
    if len(held) > 1:
        _fail("more than one GPU launch liability is held")
    for liability in held:
        if (
            liability.get("gpu_liability_reserve_seconds")
            != GPU_RESERVE_SECONDS
            or Decimal(str(liability.get("gpu_liability_reserve_cost_usd")))
            != GPU_RESERVE_COST_USD
            or _SHA.fullmatch(
                str(
                    liability.get(
                        "gpu_liability_reserve_ledger_identity_sha256"
                    )
                )
            )
            is None
        ):
            _fail("held GPU launch liability reserve drifted")
    reserved_seconds = len(held) * GPU_RESERVE_SECONDS
    reserved_cost = Decimal(len(held)) * GPU_RESERVE_COST_USD
    remaining_seconds = (
        APPROVED_GPU_RUNTIME_SECONDS
        - charged_seconds
        - reserved_seconds
    )
    remaining_cost = (
        APPROVED_GPU_COST_USD - charged_cost - reserved_cost
    )
    if remaining_seconds < 0 or remaining_cost < 0:
        _fail("live GPU spend exceeds the approved authority")
    evidence = canonical_sha256(
        {
            "descriptor_file_sha256": descriptor_file,
            "approval_file_sha256": approval_file,
            "latest_file_sha256": _sha_raw(latest_raw),
            "latest_version_id": latest_versions[0][1],
            "ledger_tip_identity_sha256": prior,
            "ledger_file_sha256": _sha_raw(bytes(ledger)),
            "held_liability_identities": [
                canonical_record_identity(
                    "glm52_production_worker_launch_liability", row
                )
                for row in held
            ],
            "request_ids": request_ids,
            "observed_at": observed_at,
        }
    )
    spend = SpendRead(
        state=(
            "OPEN"
            if active is not None or held
            else "CLOSED"
        ),
        ledger_head_identity_sha256=prior,
        remaining_gpu_seconds=remaining_seconds,
        remaining_gpu_cost_usd=format(remaining_cost, ".2f"),
        observed_at=observed_at,
        evidence_identity_sha256=evidence,
    )
    return spend, tuple(intervals), chain_identities, tuple(request_ids)


def _ec2_instances(
    *,
    ports: object,
    invocation: object,
) -> tuple[dict[str, dict[str, object]], tuple[str, ...]]:
    activation_id = _text(
        getattr(invocation, "activation_id", None), "activation id"
    )
    client = _client(ports, "ec2", "EC2")
    token: str | None = None
    seen_tokens: set[str] = set()
    by_id: dict[str, dict[str, object]] = {}
    request_ids: list[str] = []
    while True:
        arguments: dict[str, object] = {
            "Filters": [
                {"Name": "tag:RunId", "Values": [RUN_ID]},
                {"Name": "tag:activation-id", "Values": [activation_id]},
                {
                    "Name": "instance-state-name",
                    "Values": [
                        "pending",
                        "running",
                        "shutting-down",
                        "terminated",
                        "stopping",
                        "stopped",
                    ],
                },
            ],
            "MaxResults": 1000,
        }
        if token is not None:
            arguments["NextToken"] = token
        response = client.describe_instances(**arguments)
        metadata = _metadata(response, "EC2 DescribeInstances")
        request_ids.append(str(metadata["RequestId"]))
        reservations = response.get("Reservations")
        if type(reservations) is not list:
            _fail("EC2 instance readback is not complete")
        for reservation in reservations:
            instances = (
                reservation.get("Instances")
                if type(reservation) is dict
                else None
            )
            if type(instances) is not list:
                _fail("EC2 reservation is malformed")
            for instance in instances:
                instance_id = (
                    instance.get("InstanceId")
                    if type(instance) is dict
                    else None
                )
                if (
                    type(instance_id) is not str
                    or _INSTANCE_ID.fullmatch(instance_id) is None
                    or instance_id in by_id
                ):
                    _fail("EC2 instance identity is duplicated")
                state = instance.get("State")
                state_name = (
                    state.get("Name") if type(state) is dict else state
                )
                tags_raw = instance.get("Tags", [])
                if type(tags_raw) is dict:
                    tags = tags_raw
                elif type(tags_raw) is list and all(
                    type(item) is dict
                    and set(item) >= {"Key", "Value"}
                    and type(item["Key"]) is str
                    and type(item["Value"]) is str
                    for item in tags_raw
                ):
                    tags = {
                        str(item["Key"]): str(item["Value"])
                        for item in tags_raw
                    }
                else:
                    _fail("EC2 worker tags are malformed")
                if (
                    instance.get("InstanceType") != "p5.48xlarge"
                    or instance.get("InstanceLifecycle") is not None
                    or state_name
                    not in {
                        "pending",
                        "running",
                        "shutting-down",
                        "terminated",
                        "stopping",
                        "stopped",
                    }
                    or tags.get("RunId") != RUN_ID
                    or tags.get("activation-id") != activation_id
                    or tags.get("Market") != "on-demand"
                ):
                    _fail("EC2 worker readback contains drift")
                by_id[instance_id] = {
                    **instance,
                    "_state_name": state_name,
                    "_tags": tags,
                }
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if (
            type(next_token) is not str
            or not next_token
            or next_token in seen_tokens
        ):
            _fail("EC2 instance pagination cycled")
        seen_tokens.add(next_token)
        token = next_token
    return by_id, tuple(request_ids)


def _runtime_members(
    *,
    observation: Mapping[str, object],
    families: Mapping[str, tuple[dict[str, object], ...]],
    intervals: tuple[dict[str, object], ...],
    ec2: Mapping[str, dict[str, object]],
    observed_at: str,
) -> tuple[
    tuple[RuntimeEntity, ...],
    tuple[RuntimeEntity, ...],
    tuple[RuntimeEntity, ...],
    tuple[RuntimeEntity, ...],
    tuple[RuntimeEntity, ...],
]:
    requests = tuple(
        RuntimeEntity(
            identity=_text(item["request_id"], "request identity"),
            state=_text(item["state"], "request state"),
            observed_at=observed_at,
            evidence_identity_sha256=_digest(
                item["evidence_identity_sha256"], "request evidence"
            ),
        )
        for item in observation["request_snapshot"]["requests"]
    )
    jobs = tuple(
        RuntimeEntity(
            identity=_text(item["job_id"], "job identity"),
            state=_text(item["state"], "job state"),
            observed_at=observed_at,
            evidence_identity_sha256=_digest(
                item["evidence_identity_sha256"], "job evidence"
            ),
        )
        for item in observation["job_snapshot"]["jobs"]
    )
    controllers = tuple(
        RuntimeEntity(
            identity=_text(item["identity"], "controller identity"),
            state=_text(item["state"], "controller state"),
            observed_at=observed_at,
            evidence_identity_sha256=_digest(
                item["evidence_identity_sha256"], "controller evidence"
            ),
        )
        for item in observation["controller_snapshot"]["members"]
    )
    interval_by_instance = {
        str(interval["instance_id"]): interval for interval in intervals
    }
    launches = families["glm52_production_worker_launch"]
    launched_instances: dict[str, dict[str, object]] = {}
    workers: list[RuntimeEntity] = []
    allocations: list[RuntimeEntity] = []
    for launch in launches:
        instance_ids = launch.get("observed_instance_ids")
        if type(instance_ids) is not list:
            _fail("worker launch observed instance set is malformed")
        for instance_id in instance_ids:
            if (
                type(instance_id) is not str
                or instance_id in launched_instances
                or instance_id not in interval_by_instance
                or str(launch.get("sky_job_id"))
                != str(interval_by_instance[instance_id]["job_id"])
            ):
                _fail("worker launch and spend allocation binding drifted")
            launched_instances[instance_id] = launch
            interval = interval_by_instance[instance_id]
            live = ec2.get(instance_id)
            if live is None:
                if interval["state"] != "CLOSED":
                    _fail("open worker allocation is absent from EC2")
                state = "terminated"
                ec2_identity: object = None
            else:
                state = str(live["_state_name"])
                expected = (
                    {"pending", "running", "stopping", "shutting-down"}
                    if interval["state"] == "OPEN"
                    else {"stopped", "terminated", "shutting-down"}
                )
                if state not in expected:
                    _fail("EC2 state diverges from spend allocation")
                if (
                    canonical_sha256(live["_tags"])
                    != launch.get("expected_worker_tags_sha256")
                ):
                    _fail("EC2 worker tags drifted from retained launch")
                ec2_identity = {
                    "instance_id": instance_id,
                    "state": state,
                    "tags": live["_tags"],
                }
            evidence = canonical_sha256(
                {
                    "worker_launch_identity_sha256": canonical_record_identity(
                        "glm52_production_worker_launch", launch
                    ),
                    "ledger_identity_sha256": interval[
                        "ledger_identity_sha256"
                    ],
                    "ec2": ec2_identity,
                }
            )
            workers.append(
                RuntimeEntity(
                    identity=instance_id,
                    state=state,
                    observed_at=observed_at,
                    evidence_identity_sha256=evidence,
                )
            )
            allocations.append(
                RuntimeEntity(
                    identity=(
                        f"{int(launch['allocation_ordinal'])}:"
                        + instance_id
                    ),
                    state=str(interval["state"]),
                    observed_at=observed_at,
                    evidence_identity_sha256=evidence,
                )
            )
    open_instances = {
        str(interval["instance_id"])
        for interval in intervals
        if interval["state"] == "OPEN"
    }
    if not open_instances.issubset(launched_instances):
        _fail("open spend allocation lacks retained worker custody")
    if not set(ec2).issubset(set(interval_by_instance)):
        _fail("EC2 returned an instance absent from the spend ledger")
    post_terminal = families[
        "glm52_production_post_terminal_allocation"
    ]
    for record in post_terminal:
        instance_id = str(record["instance_id"])
        if instance_id not in interval_by_instance:
            _fail("post-terminal allocation is absent from spend ledger")
    return (
        tuple(sorted(requests, key=lambda item: item.identity)),
        tuple(sorted(jobs, key=lambda item: item.identity)),
        tuple(sorted(controllers, key=lambda item: item.identity)),
        tuple(sorted(workers, key=lambda item: item.identity)),
        tuple(sorted(allocations, key=lambda item: item.identity)),
    )


def _complete_read(
    *,
    domain: str,
    members: tuple[RuntimeEntity, ...],
    observed_at: str,
    evidence: Mapping[str, object],
) -> CompleteRuntimeRead:
    return CompleteRuntimeRead(
        domain=domain,
        members=members,
        observed_at=observed_at,
        pagination_complete=True,
        evidence_identity_sha256=canonical_sha256(
            {
                "domain": domain,
                "members": [_jsonable(member) for member in members],
                **evidence,
            }
        ),
    )


class _CollectedReader:
    def __init__(
        self,
        *,
        reads: tuple[CompleteRuntimeRead, ...],
        spend: SpendRead,
    ) -> None:
        self.reads = reads
        self.spend = spend

    def read_requests(self, _correlation: str) -> CompleteRuntimeRead:
        return self.reads[0]

    def read_jobs(self, _correlation: str) -> CompleteRuntimeRead:
        return self.reads[1]

    def read_controller(self, _correlation: str) -> CompleteRuntimeRead:
        return self.reads[2]

    def read_workers(self, _correlation: str) -> CompleteRuntimeRead:
        return self.reads[3]

    def read_allocations(self, _correlation: str) -> CompleteRuntimeRead:
        return self.reads[4]

    def read_spend(self, _correlation: str) -> SpendRead:
        return self.spend


def _collect_live_scan(
    *,
    ports: object,
    invocation: object,
    execution: Mapping[str, object],
    campaign_identity_sha256: str,
    correlation_identity_sha256: str,
    require_execution_terminal: bool = False,
) -> tuple[RuntimeScan, dict[str, object], str]:
    observation = _numeric_observation(
        ports=ports,
        invocation=invocation,
        correlation_identity_sha256=correlation_identity_sha256,
    )
    observed_at = str(observation["observed_at"])
    (
        execution_status,
        execution_evidence,
        exact_execution,
    ) = _describe_support_execution(
        ports=ports,
        execution=execution,
        observed_at=observed_at,
    )
    if (
        require_execution_terminal
        and execution_status not in _PROOF_EXECUTION_TERMINAL_STATES
    ):
        for _attempt in range(EXECUTION_TERMINAL_MAX_POLLS):
            _sleep(EXECUTION_TERMINAL_POLL_SECONDS)
            observation = _numeric_observation(
                ports=ports,
                invocation=invocation,
                correlation_identity_sha256=correlation_identity_sha256,
            )
            observed_at = str(observation["observed_at"])
            (
                execution_status,
                execution_evidence,
                exact_execution,
            ) = _describe_support_execution(
                ports=ports,
                execution=exact_execution,
                observed_at=observed_at,
            )
            if (
                execution_status
                in _PROOF_EXECUTION_TERMINAL_STATES
            ):
                break
        else:
            _fail("support execution is not terminal after bounded polling")
    families, ddb_request_ids = _query_runtime_families(
        ports=ports,
        invocation=invocation,
        campaign_identity_sha256=campaign_identity_sha256,
    )
    spend, intervals, _chain, s3_request_ids = _spend_authority(
        ports=ports,
        invocation=invocation,
        observed_at=observed_at,
        campaign_identity_sha256=campaign_identity_sha256,
        families=families,
    )
    ec2, ec2_request_ids = _ec2_instances(
        ports=ports, invocation=invocation
    )
    members = _runtime_members(
        observation=observation,
        families=families,
        intervals=intervals,
        ec2=ec2,
        observed_at=observed_at,
    )
    spend = SpendRead(
        state=spend.state,
        ledger_head_identity_sha256=spend.ledger_head_identity_sha256,
        remaining_gpu_seconds=spend.remaining_gpu_seconds,
        remaining_gpu_cost_usd=spend.remaining_gpu_cost_usd,
        observed_at=spend.observed_at,
        evidence_identity_sha256=canonical_sha256(
            {
                "spend_authority_evidence_sha256": (
                    spend.evidence_identity_sha256
                ),
                "ec2_request_ids": list(ec2_request_ids),
                "ec2_instances": [
                    {
                        "instance_id": instance_id,
                        "state": value["_state_name"],
                        "tags": value["_tags"],
                    }
                    for instance_id, value in sorted(ec2.items())
                ],
            }
        ),
    )
    common_evidence = {
        "numeric_observation_identity_sha256": observation[
            "canonical_identity_sha256"
        ],
        "support_execution_status": execution_status,
        "support_execution_evidence_sha256": execution_evidence,
        "dynamodb_request_ids": list(ddb_request_ids),
        "s3_request_ids": list(s3_request_ids),
        "ec2_request_ids": list(ec2_request_ids),
    }
    reads = tuple(
        _complete_read(
            domain=domain,
            members=domain_members,
            observed_at=observed_at,
            evidence=common_evidence,
        )
        for domain, domain_members in zip(
            ("REQUEST", "JOB", "CONTROLLER", "WORKER", "ALLOCATION"),
            members,
        )
    )
    reader = _CollectedReader(reads=reads, spend=spend)
    return (
        collect_runtime_scan(
            boundaries=RuntimeReadBoundaries(
                request_reader=reader,
                job_reader=reader,
                controller_reader=reader,
                worker_reader=reader,
                allocation_reader=reader,
                spend_reader=reader,
            ),
            correlation_identity_sha256=correlation_identity_sha256,
        ),
        exact_execution,
        execution_status,
    )


def materialize_live_request(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> dict[str, object]:
    """Build a request from current production authorities only."""

    sources = _validate_live_sources(
        operation_kind=operation_kind,
        invocation=invocation,
        live_sources=live_sources,
    )
    correlation = _correlation(
        operation_kind=operation_kind, sources=sources
    )
    campaign_identity = str(
        sources["execution"]["campaign_identity_sha256"]
    )
    if operation_kind == RECONCILE:
        _validate_finalization_trigger(
            invocation=invocation,
            ports=ports,
        )
        scan, _exact_execution, _execution_status = _collect_live_scan(
            ports=ports,
            invocation=invocation,
            execution=sources["execution"],
            campaign_identity_sha256=campaign_identity,
            correlation_identity_sha256=correlation,
        )
        return {
            "correlation_identity_sha256": correlation,
            "runtime_scan": _jsonable(scan),
        }
    first, exact_execution, first_execution_status = _collect_live_scan(
        ports=ports,
        invocation=invocation,
        execution=sources["execution"],
        campaign_identity_sha256=campaign_identity,
        correlation_identity_sha256=correlation,
        require_execution_terminal=True,
    )
    if (
        first_execution_status
        not in _PROOF_EXECUTION_TERMINAL_STATES
    ):
        _fail("support execution is not terminal")
    _sleep(QUIET_SECONDS)
    (
        second,
        _exact_execution,
        second_execution_status,
    ) = _collect_live_scan(
        ports=ports,
        invocation=invocation,
        execution=exact_execution,
        campaign_identity_sha256=campaign_identity,
        correlation_identity_sha256=correlation,
        require_execution_terminal=True,
    )
    if (
        second_execution_status not in _PROOF_EXECUTION_TERMINAL_STATES
        or second_execution_status != first_execution_status
    ):
        _fail("support execution terminal status drifted")
    if first.observed_at == second.observed_at:
        _fail("two distinct runtime observations are required")
    return {
        "first_runtime_scan": _jsonable(first),
        "second_runtime_scan": _jsonable(second),
        "minimum_quiet_seconds": QUIET_SECONDS,
    }


def _recovery_candidate(
    *,
    invocation: object,
    control: Mapping[str, object],
) -> dict[str, object]:
    candidate = {
        "schema_version": 1,
        "record_type": "glm52_production_recovery_control",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": control[
            "campaign_identity_sha256"
        ],
        "activation_id": getattr(invocation, "activation_id"),
        "activation_ordinal": getattr(invocation, "activation_ordinal"),
        "rollover_identity_sha256": control[
            "rollover_identity_sha256"
        ],
        "cleanup_control_root_identity_sha256": control[
            "snapshot_cleanup_control_initial_body_sha256"
        ],
        "cleanup_transition_chain_head_sha256": control[
            "snapshot_cleanup_control_initial_body_sha256"
        ],
        # The initial cleanup-control root is the first authenticated lineage
        # member even though no recovery transition has yet occurred.
        "cleanup_transition_chain_length": 1,
        "state": "DORMANT",
        "owner_attempt": None,
        "owner_execution_arn": None,
        "owner_state_machine_version_arn": None,
        "owner_dispatch_identity_sha256": None,
        "owner_invocation_nonce_sha256": None,
        "owner_hard_expires_at": None,
        "recovery_barrier_nonce_sha256": None,
        "support_control_revision_at_seal": None,
        "support_execution_identity_sha256": None,
        "allowed_action_set_sha256": None,
        "terminal_v2_identity_sha256": None,
        "revision": 1,
        "updated_at": control["updated_at"],
    }
    try:
        exact = validate_record(
            "glm52_production_recovery_control", candidate
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveRuntimeError(
            "canonical recovery control candidate is invalid"
        ) from exc
    if (
        canonical_record_identity(
            "glm52_production_recovery_control", exact
        )
        != control.get("recovery_control_initial_body_sha256")
    ):
        _fail("canonical recovery control initial identity drifted")
    return exact


def _read_recovery_control(
    *,
    ports: object,
    invocation: object,
    table_name: str,
) -> dict[str, object] | None:
    pk = ledger_pk(RUN_ID)
    sk = ledger_sk(
        "glm52_production_recovery_control",
        activation_id=getattr(invocation, "activation_id"),
    )
    response = _client(ports, "dynamodb", "DynamoDB").get_item(
        TableName=table_name,
        Key=encode_item({"PK": pk, "SK": sk}),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(response, "DynamoDB recovery GetItem")
    item = response.get("Item")
    if item is None:
        return None
    if type(item) is not dict:
        _fail("DynamoDB recovery readback is malformed")
    value = decode_item(item)
    if value.pop("PK", None) != pk or value.pop("SK", None) != sk:
        _fail("DynamoDB recovery readback key drifted")
    try:
        return validate_record(
            "glm52_production_recovery_control",
            value,
            pk=pk,
            sk=sk,
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveRuntimeError(
            "DynamoDB recovery readback is noncanonical"
        ) from exc


def _create_or_adopt_recovery_control(
    *,
    ports: object,
    invocation: object,
    candidate: Mapping[str, object],
) -> None:
    table_name = _text(
        _roles(ports).get("ledger_table_name"), "ledger table"
    )
    pk = ledger_pk(RUN_ID)
    sk = ledger_sk(
        "glm52_production_recovery_control",
        activation_id=getattr(invocation, "activation_id"),
    )
    client = _client(ports, "dynamodb", "DynamoDB")
    try:
        response = client.put_item(
            TableName=table_name,
            Item=encode_item({"PK": pk, "SK": sk, **candidate}),
            ConditionExpression=(
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            ),
            ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
            ReturnConsumedCapacity="NONE",
        )
    except Exception:
        # The single conditional send may have committed before its response
        # was lost.  Never resend; exact consistent readback is the sole
        # adoption path.
        pass
    else:
        # A returned response with retry metadata is a contract violation, not
        # an ambiguous transport eligible for adoption.
        _metadata(response, "DynamoDB recovery PutItem")
    observed = _read_recovery_control(
        ports=ports,
        invocation=invocation,
        table_name=table_name,
    )
    if observed != candidate:
        _fail("foreign recovery control preexisted or won the write")


def persist_live_successors(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    request: Mapping[str, object],
    domain_result: object,
    ports: object,
) -> bool:
    """Persist the one canonical producer edge; proof is observation-only."""

    sources = _validate_live_sources(
        operation_kind=operation_kind,
        invocation=invocation,
        live_sources=live_sources,
    )
    if operation_kind == RECONCILE:
        if (
            not isinstance(domain_result, RuntimeScan)
            or type(request) is not dict
            or set(request)
            != {"correlation_identity_sha256", "runtime_scan"}
            or request["correlation_identity_sha256"]
            != _correlation(operation_kind=operation_kind, sources=sources)
            or request["runtime_scan"] != _jsonable(domain_result)
        ):
            _fail("reconcile runtime domain result drifted")
        candidate = _recovery_candidate(
            invocation=invocation, control=sources["control"]
        )
        _create_or_adopt_recovery_control(
            ports=ports,
            invocation=invocation,
            candidate=candidate,
        )
        return True
    if not isinstance(domain_result, RuntimeTerminalProof):
        _fail("runtime terminal proof is not typed")
    if (
        type(request) is not dict
        or set(request)
        != {
            "first_runtime_scan",
            "second_runtime_scan",
            "minimum_quiet_seconds",
        }
        or request["minimum_quiet_seconds"] != QUIET_SECONDS
    ):
        _fail("runtime terminal proof request drifted")
    try:
        from .task12_lambda_adapters import _materialize_dataclass
        from .task12_runtime import prove_runtime_terminal

        expected_proof = prove_runtime_terminal(
            first=_materialize_dataclass(
                RuntimeScan, request["first_runtime_scan"]
            ),
            second=_materialize_dataclass(
                RuntimeScan, request["second_runtime_scan"]
            ),
            minimum_quiet_seconds=QUIET_SECONDS,
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveRuntimeError(
            "runtime terminal proof request drifted"
        ) from exc
    if expected_proof != domain_result:
        _fail("runtime terminal proof identity drifted")
    return True


__all__ = [
    "SUPPORTED_OPERATIONS",
    "Task12LiveRuntimeError",
    "materialize_live_request",
    "persist_live_successors",
]
