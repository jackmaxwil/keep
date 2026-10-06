"""Direct live materializers for Task 12 drain, terminal, and teardown.

This module deliberately has no operation-projection format.  Inputs that are
not registered production records are named, content-addressed *current
observations*.  Their schemas contain domain facts only and explicitly reject
future operation requests.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
import hashlib
import re
from typing import Mapping

from .canonical import canonical_sha256
from .dynamodb import (
    DynamoLedgerAdapter,
    ExactCheck,
    ExactUpdate,
    LedgerKey,
    decode_item,
    encode_item,
)
from .records import (
    canonical_record_identity,
    ledger_pk,
    ledger_sk,
    validate_record,
)


RUN_ID = "glm52-sky-20260724"
_SOURCE_TYPE = "glm52_task12_live_domain_source_v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ACTIVATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

OWNED_OPERATIONS = frozenset(
    {
        "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
        "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
        "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
        "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
        "RETAINED_ENTER_RECOVERY_COMPLETE",
        "RETAINED_ACQUIRE_TEARDOWN_SEALING",
        "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
        "RETAINED_ENTER_TEARDOWN_SEALED",
    }
)
SUPPORTED_OPERATIONS = frozenset(
    {
        "RETAINED_ENTER_RECOVERY_COMPLETE",
        "RETAINED_ACQUIRE_TEARDOWN_SEALING",
        "RETAINED_ENTER_TEARDOWN_SEALED",
    }
)

_SOURCE_FIELDS: Mapping[str, frozenset[str]] = {
    "REQUEST_JOB_CORRELATION": frozenset(
        {"request_ids", "job_ids", "observed_at"}
    ),
    "STATE_AUTHORITY": frozenset(
        {
            "activation_index",
            "control",
            "recovery_control",
            "finalization_control",
        }
    ),
}

_FAMILY_NAMES = {
    "glm52_production_worker_launch": "WORKER_LAUNCH",
    "glm52_production_worker_launch_liability": "WORKER_LAUNCH_LIABILITY",
    "glm52_production_worker_launch_liability_settlement": (
        "WORKER_LAUNCH_LIABILITY_SETTLEMENT"
    ),
    "glm52_production_post_terminal_allocation": "POST_TERMINAL_ALLOCATION",
}


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _metadata(value: object, label: str) -> None:
    if (
        type(value) is not dict
        or value.get("HTTPStatusCode") != 200
        or type(value.get("RequestId")) is not str
        or not value["RequestId"]
        or type(value.get("RetryAttempts")) is not int
        or value["RetryAttempts"] != 0
    ):
        raise ValueError(label + " response is not authenticated zero-retry")


def _contains_future_payload(value: object) -> bool:
    if isinstance(value, Mapping):
        if set(value) & {
            "operation_kind",
            "operation_request",
            "successor_request",
            "future_payload",
        }:
            return True
        if "request" in value and type(value.get("request")) is dict:
            return True
        return any(_contains_future_payload(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_future_payload(item) for item in value)
    return False


def build_live_domain_source(
    *,
    source_kind: str,
    activation_id: str,
    generation: int,
    payload: object,
) -> dict[str, object]:
    fields = _SOURCE_FIELDS.get(source_kind)
    if _contains_future_payload(payload):
        raise ValueError("future operation payload is forbidden")
    if (
        fields is None
        or type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
        or type(generation) is not int
        or isinstance(generation, bool)
        or generation < 1
        or type(payload) is not dict
        or set(payload) != fields
    ):
        raise ValueError("live domain source fields are not closed")
    if source_kind == "REQUEST_JOB_CORRELATION":
        requests = payload["request_ids"]
        jobs = payload["job_ids"]
        if (
            type(requests) is not list
            or requests != sorted(requests)
            or len(requests) != len(set(requests))
            or any(type(item) is not str or not item for item in requests)
            or type(jobs) is not list
            or jobs != sorted(jobs, key=int)
            or len(jobs) != len(set(jobs))
            or any(
                type(item) is not str
                or re.fullmatch(r"[1-9][0-9]*", item) is None
                for item in jobs
            )
        ):
            raise ValueError("request/job correlation member set is not exact")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": _SOURCE_TYPE,
        "source_kind": source_kind,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "generation": generation,
        "generation_text": f"{generation:08d}",
        "payload": _json_value(payload),
    }
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def validate_live_domain_source(
    value: object, *, expected_kind: str
) -> dict[str, object]:
    fields = {
        "schema_version",
        "record_type",
        "source_kind",
        "run_id",
        "activation_id",
        "generation",
        "generation_text",
        "payload",
        "canonical_body_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("live domain source envelope is not closed")
    rebuilt = build_live_domain_source(
        source_kind=value["source_kind"],
        activation_id=value["activation_id"],
        generation=value["generation"],
        payload=value["payload"],
    )
    if value["source_kind"] != expected_kind or rebuilt != value:
        raise ValueError("live domain source identity drifted")
    return dict(value)


def query_complete_family(
    *,
    client: object,
    table_name: str,
    activation_id: str,
    record_type: str,
) -> tuple[tuple[Mapping[str, object], ...], str]:
    family = _FAMILY_NAMES.get(record_type)
    if family is None:
        raise ValueError("plural runtime family is not closed")
    class _ZeroRetryQueryClient:
        def query(self, **kwargs: object) -> object:
            response = client.query(**kwargs)
            _metadata(
                response.get("ResponseMetadata"),
                "plural runtime query",
            )
            return response

    adapter = DynamoLedgerAdapter(
        client=_ZeroRetryQueryClient(), table_name=table_name
    )
    try:
        records = adapter.query_activation_family(
            run_id=RUN_ID,
            sort_key_prefix=f"ACTIVATION#{activation_id}#{family}#",
            record_type=record_type,
        )
    except Exception as exc:
        message = str(exc)
        cause = getattr(exc, "__cause__", None)
        if (
            "RetryAttempts" in message
            or "metadata" in message
            or "zero-retry" in str(cause)
        ):
            message = "plural runtime query is not authenticated zero-retry"
        elif "foreign records" in message:
            message = "plural runtime query returned a duplicate or foreign key"
        raise ValueError(message) from exc
    keys = [
        ledger_sk(
            record_type,
            activation_id=activation_id,
            allocation_ordinal=record["allocation_ordinal"],
            **(
                {"instance_id": record["instance_id"]}
                if record_type
                == "glm52_production_post_terminal_allocation"
                else {}
            ),
        )
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("plural runtime query returned a duplicate key")
    identity = canonical_sha256(
        {
            "record_type": record_type,
            "activation_id": activation_id,
            "pagination_complete": True,
            "records": list(records),
        }
    )
    return records, identity


def persist_exact_live_source(
    *,
    client: object,
    table_name: str,
    partition_key: str,
    sort_key: str,
    source: Mapping[str, object],
) -> None:
    exact = validate_live_domain_source(
        source, expected_kind=source["source_kind"]
    )
    stored = {"PK": partition_key, "SK": sort_key, **exact}
    try:
        response = client.put_item(
            TableName=table_name,
            Item=encode_item(stored),
            ConditionExpression=(
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            ),
            ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
            ReturnConsumedCapacity="NONE",
        )
    except Exception:
        response = None
    if type(response) is dict:
        _metadata(response.get("ResponseMetadata"), "live source write")
        return
    try:
        readback = client.get_item(
            TableName=table_name,
            Key=encode_item({"PK": partition_key, "SK": sort_key}),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
    except Exception as exc:
        raise ValueError("live source exact readback failed") from exc
    _metadata(readback.get("ResponseMetadata"), "live source readback")
    if type(readback.get("Item")) is not dict:
        raise ValueError("live source readback is absent")
    if decode_item(readback["Item"]) != stored:
        raise ValueError("live source adopted foreign bytes")


def _roles(ports: object) -> Mapping[str, object]:
    deployment = getattr(ports, "deployment", None)
    roles = getattr(deployment, "role_coordinates", None)
    if not isinstance(roles, Mapping):
        raise ValueError("drain/terminal deployment coordinates are absent")
    return roles


def _table_name(ports: object) -> str:
    value = _roles(ports).get("ledger_table_name")
    if type(value) is not str or not value:
        raise ValueError("drain/terminal ledger table is absent")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _rehash(record: dict[str, object]) -> None:
    if "canonical_body_sha256" in record:
        body = dict(record)
        body.pop("canonical_body_sha256")
        record["canonical_body_sha256"] = canonical_sha256(body)


def _read_exact_record(
    *,
    ports: object,
    activation_id: str,
    record_type: str,
    generation: int | None = None,
    writer_kind: str | None = None,
) -> dict[str, object]:
    if record_type == "glm52_production_activation_index":
        sort_key = ledger_sk(record_type)
    elif record_type == "glm52_task12_versioned_writer_control_v1":
        sort_key = ledger_sk(
            record_type,
            activation_id=activation_id,
            generation=generation,
            writer_kind=writer_kind,
        )
    else:
        sort_key = ledger_sk(record_type, activation_id=activation_id)
    partition_key = ledger_pk(RUN_ID)
    try:
        response = ports.client("dynamodb").get_item(
            TableName=_table_name(ports),
            Key=encode_item({"PK": partition_key, "SK": sort_key}),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
    except Exception as exc:
        raise ValueError("exact retained state read failed") from exc
    _metadata(response.get("ResponseMetadata"), "exact retained state read")
    if type(response.get("Item")) is not dict:
        raise ValueError("exact retained state record is absent")
    physical = decode_item(response["Item"])
    if (
        physical.pop("PK", None) != partition_key
        or physical.pop("SK", None) != sort_key
    ):
        raise ValueError("exact retained state coordinate drifted")
    try:
        return validate_record(record_type, physical)
    except (TypeError, ValueError) as exc:
        raise ValueError("exact retained state record is invalid") from exc


def _live_state(
    *,
    ports: object,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    activation_id = getattr(invocation, "activation_id")
    types = {
        "activation_index": "glm52_production_activation_index",
        "control": "glm52_production_control",
        "recovery_control": "glm52_production_recovery_control",
        "finalization_control": "glm52_production_finalization_control",
    }
    state = {
        alias: _read_exact_record(
            ports=ports,
            activation_id=activation_id,
            record_type=record_type,
        )
        for alias, record_type in types.items()
    }
    for alias in ("control", "recovery_control", "finalization_control"):
        descriptor_value = live_sources.get(alias)
        if descriptor_value is not None and descriptor_value != state[alias]:
            raise ValueError(alias + " live-state drifted")
    return state


def _continuation_capsule(
    invocation: object,
    owner: Mapping[str, object],
    *,
    authority_domain: str,
    barrier_field: str,
    control_revision_field: str,
) -> dict[str, object]:
    operation_input = getattr(invocation, "operation_input", None)
    prior = (
        operation_input.get("task12_last_result")
        if isinstance(operation_input, Mapping)
        else None
    )
    result = prior.get("result") if isinstance(prior, Mapping) else None
    capsule = (
        result.get("owner_nonce_capsule")
        if isinstance(result, Mapping)
        else None
    )
    if type(capsule) is not dict or capsule.get(
        "nonce_sha256"
    ) != owner.get("owner_invocation_nonce_sha256"):
        raise ValueError("encrypted retained owner nonce authority is absent")
    context = capsule.get("encryption_context")
    expected_context = {
        "account_id": "246813579024",
        "activation_id": str(owner.get("activation_id")),
        "authority_domain": authority_domain,
        "barrier_nonce_sha256": str(owner.get(barrier_field)),
        "control_revision": str(owner.get(control_revision_field)),
        "owner_attempt": str(owner.get("owner_attempt")),
        "owner_execution_arn": str(owner.get("owner_execution_arn")),
        "owner_hard_expires_at": str(owner.get("owner_hard_expires_at")),
        "owner_state_machine_version_arn": str(
            owner.get("owner_state_machine_version_arn")
        ),
        "region": "us-west-2",
        "run_id": RUN_ID,
    }
    if type(context) is not dict or context != expected_context:
        raise ValueError("encrypted retained owner nonce authority drifted")
    return dict(capsule)


def _teardown_sealed_request(
    *,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> dict[str, object]:
    from .task12_retained_state import build_teardown_seal_plan
    from .task12_nonce_capsule import generate_owner_nonce_capsule

    state = _live_state(
        ports=ports, invocation=invocation, live_sources=live_sources
    )
    index = state["activation_index"]
    control_before = state["control"]
    recovery = state["recovery_control"]
    finalization_before = state["finalization_control"]
    observed_at = _now()
    control_after = dict(control_before)
    control_after.update(
        phase="TEARDOWN_SEALED",
        revision=control_before["revision"] + 1,
        updated_at=observed_at,
    )
    _rehash(control_after)
    owner_hard_expires_at = (
        datetime.now(timezone.utc) + timedelta(minutes=15)
    ).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    finalization_barrier = canonical_sha256(
        {
            "authority_domain": "FINALIZATION",
            "activation_id": getattr(invocation, "activation_id"),
            "owner_execution_arn": getattr(
                invocation, "state_machine_execution_arn"
            ),
            "owner_state_machine_version_arn": getattr(
                invocation, "caller_state_machine_version_arn"
            ),
            "control_revision": control_after["revision"],
            "owner_hard_expires_at": owner_hard_expires_at,
        }
    )
    activation_id = getattr(invocation, "activation_id")

    def _plan_for_nonce(nonce_sha: str) -> object:
        finalization_after = dict(finalization_before)
        finalization_after.update(
            state="OWNED",
            owner_attempt=1,
            owner_execution_arn=getattr(
                invocation, "state_machine_execution_arn"
            ),
            owner_state_machine_version_arn=getattr(
                invocation, "caller_state_machine_version_arn"
            ),
            owner_dispatch_identity_sha256=getattr(
                invocation, "dispatch_identity_sha256"
            ),
            owner_invocation_nonce_sha256=nonce_sha,
            owner_hard_expires_at=owner_hard_expires_at,
            finalization_barrier_nonce_sha256=finalization_barrier,
            teardown_sealed_control_revision=control_after["revision"],
            terminal_v2_identity_sha256=recovery[
                "terminal_v2_identity_sha256"
            ],
            revision=finalization_before["revision"] + 1,
            updated_at=observed_at,
        )
        _rehash(finalization_after)
        return build_teardown_seal_plan(
            index=ExactCheck(
                LedgerKey(
                    RUN_ID, ledger_sk("glm52_production_activation_index")
                ),
                index,
            ),
            recovery_control=ExactCheck(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_recovery_control",
                        activation_id=activation_id,
                    ),
                ),
                recovery,
            ),
            control=ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_control",
                        activation_id=activation_id,
                    ),
                ),
                control_before,
                control_after,
            ),
            finalization_control=ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_finalization_control",
                        activation_id=activation_id,
                    ),
                ),
                finalization_before,
                finalization_after,
            ),
        )

    # Validate the complete deterministic successor before the first KMS effect.
    _plan_for_nonce("0" * 64)
    owner_nonce_capsule, raw_nonce = generate_owner_nonce_capsule(
        ports=ports,
        authority={
            "account_id": "246813579024",
            "region": "us-west-2",
            "run_id": RUN_ID,
            "activation_id": activation_id,
            "authority_domain": "FINALIZATION",
            "owner_execution_arn": getattr(
                invocation, "state_machine_execution_arn"
            ),
            "owner_state_machine_version_arn": getattr(
                invocation, "caller_state_machine_version_arn"
            ),
            "owner_attempt": 1,
            "barrier_nonce_sha256": finalization_barrier,
            "control_revision": control_after["revision"],
            "owner_hard_expires_at": owner_hard_expires_at,
        },
    )
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    plan = _plan_for_nonce(nonce_sha)
    operation_identity = canonical_sha256(
        {
            "operation_kind": getattr(invocation, "operation_kind"),
            "control_before": canonical_record_identity(
                "glm52_production_control", control_before
            ),
            "recovery": canonical_record_identity(
                "glm52_production_recovery_control", recovery
            ),
            "nonce_sha256": nonce_sha,
        }
    )
    return {
        "plan": _json_value(plan),
        "domain": "TEARDOWN",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": owner_nonce_capsule,
    }


def _recovery_complete_request(
    *,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> dict[str, object]:
    from .task12_retained_state import build_recovery_progress_plan

    terminal_value = live_sources.get("terminal_v2")
    if type(terminal_value) is not dict:
        raise ValueError(
            "actual TerminalV2 versioned source is required before recovery"
        )
    terminal = validate_record("glm52_production_terminal_v2", terminal_value)
    state = _live_state(
        ports=ports, invocation=invocation, live_sources=live_sources
    )
    version_control = _read_exact_record(
        ports=ports,
        activation_id=getattr(invocation, "activation_id"),
        record_type="glm52_task12_versioned_writer_control_v1",
        generation=getattr(invocation, "generation"),
        writer_kind="TerminalV2",
    )
    terminal_identity = canonical_record_identity(
        "glm52_production_terminal_v2", terminal
    )
    if (
        version_control["writer_kind"] != "TerminalV2"
        or version_control["body_sha256"] != terminal_identity
        or version_control["activation_id"]
        != getattr(invocation, "activation_id")
        or version_control["generation"] != getattr(invocation, "generation")
    ):
        raise ValueError("TerminalV2 exact version binding drifted")
    index = state["activation_index"]
    control_before = state["control"]
    recovery_before = state["recovery_control"]
    if (
        control_before["phase"] != "RECOVERY_SEALING"
        or recovery_before["state"] != "TERMINAL_V2_PUBLISHED"
        or recovery_before["terminal_v2_identity_sha256"]
        != terminal_identity
    ):
        raise ValueError("recovery-complete live state is not ready")
    owner_nonce_capsule = _continuation_capsule(
        invocation,
        recovery_before,
        authority_domain="RECOVERY",
        barrier_field="recovery_barrier_nonce_sha256",
        control_revision_field="support_control_revision_at_seal",
    )
    observed_at = _now()
    control_after = dict(control_before)
    control_after.update(
        phase="RECOVERY_COMPLETE",
        revision=control_before["revision"] + 1,
        updated_at=observed_at,
    )
    _rehash(control_after)
    recovery_after = dict(recovery_before)
    recovery_after.update(
        state="RECOVERY_COMPLETE",
        owner_attempt=None,
        owner_execution_arn=None,
        owner_state_machine_version_arn=None,
        owner_dispatch_identity_sha256=None,
        owner_invocation_nonce_sha256=None,
        owner_hard_expires_at=None,
        revision=recovery_before["revision"] + 1,
        updated_at=observed_at,
    )
    _rehash(recovery_after)
    activation_id = getattr(invocation, "activation_id")
    plan = build_recovery_progress_plan(
        index=ExactCheck(
            LedgerKey(RUN_ID, ledger_sk("glm52_production_activation_index")),
            index,
        ),
        control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=activation_id,
                ),
            ),
            control_before,
            control_after,
        ),
        recovery_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_recovery_control",
                    activation_id=activation_id,
                ),
            ),
            recovery_before,
            recovery_after,
        ),
    )
    return {
        "plan": _json_value(plan),
        "domain": "RECOVERY",
        "operation_identity_sha256": canonical_sha256(
            {
                "operation_kind": getattr(invocation, "operation_kind"),
                "terminal_version_control": version_control[
                    "canonical_body_sha256"
                ],
                "control_before": canonical_record_identity(
                    "glm52_production_control", control_before
                ),
                "recovery_before": canonical_record_identity(
                    "glm52_production_recovery_control", recovery_before
                ),
            }
        ),
        "owner_nonce_capsule": owner_nonce_capsule,
    }


def _teardown_sealing_request(
    *,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> dict[str, object]:
    from .task12_retained_state import build_teardown_seal_plan
    from .task12_nonce_capsule import generate_owner_nonce_capsule

    state = _live_state(
        ports=ports, invocation=invocation, live_sources=live_sources
    )
    index = state["activation_index"]
    control_before = state["control"]
    recovery = state["recovery_control"]
    finalization = state["finalization_control"]
    if (
        control_before["phase"] != "RECOVERY_COMPLETE"
        or recovery["state"] != "RECOVERY_COMPLETE"
        or finalization["state"] != "DORMANT"
    ):
        raise ValueError("teardown-sealing live state is not ready")
    control_after = dict(control_before)
    control_after.update(
        phase="TEARDOWN_SEALING",
        revision=control_before["revision"] + 1,
        updated_at=_now(),
    )
    _rehash(control_after)
    owner_hard_expires_at = (
        datetime.now(timezone.utc) + timedelta(minutes=15)
    ).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    transaction_barrier = canonical_sha256(
        {
            "authority_domain": "TEARDOWN",
            "activation_id": getattr(invocation, "activation_id"),
            "owner_execution_arn": getattr(
                invocation, "state_machine_execution_arn"
            ),
            "owner_state_machine_version_arn": getattr(
                invocation, "caller_state_machine_version_arn"
            ),
            "control_revision": control_after["revision"],
            "owner_hard_expires_at": owner_hard_expires_at,
        }
    )
    activation_id = getattr(invocation, "activation_id")
    plan = build_teardown_seal_plan(
        index=ExactCheck(
            LedgerKey(RUN_ID, ledger_sk("glm52_production_activation_index")),
            index,
        ),
        recovery_control=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_recovery_control",
                    activation_id=activation_id,
                ),
            ),
            recovery,
        ),
        control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=activation_id,
                ),
            ),
            control_before,
            control_after,
        ),
        finalization_control=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_finalization_control",
                    activation_id=activation_id,
                ),
            ),
            finalization,
        ),
    )
    # The one-shot transition is fully validated before the first KMS effect.
    owner_nonce_capsule, _raw_nonce = generate_owner_nonce_capsule(
        ports=ports,
        authority={
            "account_id": "246813579024",
            "region": "us-west-2",
            "run_id": RUN_ID,
            "activation_id": activation_id,
            "authority_domain": "TEARDOWN",
            "owner_execution_arn": getattr(
                invocation, "state_machine_execution_arn"
            ),
            "owner_state_machine_version_arn": getattr(
                invocation, "caller_state_machine_version_arn"
            ),
            "owner_attempt": 1,
            "barrier_nonce_sha256": transaction_barrier,
            "control_revision": control_after["revision"],
            "owner_hard_expires_at": owner_hard_expires_at,
        },
    )
    return {
        "plan": _json_value(plan),
        "domain": "TEARDOWN",
        "operation_identity_sha256": canonical_sha256(
            {
                "operation_kind": getattr(invocation, "operation_kind"),
                "control_before": canonical_record_identity(
                    "glm52_production_control", control_before
                ),
                "recovery": canonical_record_identity(
                    "glm52_production_recovery_control", recovery
                ),
                "finalization": canonical_record_identity(
                    "glm52_production_finalization_control", finalization
                ),
            }
        ),
        "owner_nonce_capsule": owner_nonce_capsule,
    }


def materialize_live_request(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> Mapping[str, object] | None:
    if operation_kind not in SUPPORTED_OPERATIONS:
        return None
    if operation_kind == "RETAINED_ENTER_RECOVERY_COMPLETE":
        return _recovery_complete_request(
            invocation=invocation,
            live_sources=live_sources,
            ports=ports,
        )
    if operation_kind == "RETAINED_ENTER_TEARDOWN_SEALED":
        return _teardown_sealed_request(
            invocation=invocation, live_sources=live_sources, ports=ports
        )
    if operation_kind == "RETAINED_ACQUIRE_TEARDOWN_SEALING":
        return _teardown_sealing_request(
            invocation=invocation,
            live_sources=live_sources,
            ports=ports,
        )
    raise ValueError(operation_kind + " has no direct live materializer")


def persist_live_successors(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    request: Mapping[str, object],
    domain_result: object,
    ports: object,
) -> bool:
    if operation_kind not in SUPPORTED_OPERATIONS:
        return False
    # Writers and retained-state commits persist their canonical source family
    # inside the strict domain effect.  No future operation payload is written.
    del invocation, live_sources, request, domain_result, ports
    return True


__all__ = [
    "SUPPORTED_OPERATIONS",
    "build_live_domain_source",
    "materialize_live_request",
    "persist_exact_live_source",
    "persist_live_successors",
    "query_complete_family",
    "validate_live_domain_source",
]
