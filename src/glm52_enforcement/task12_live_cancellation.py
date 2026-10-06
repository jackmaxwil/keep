"""Authorized live materialization for Task 12 cancellation and drain closure.

The only Sky-facing calls in this module are exact-version Lambda invocations
of the published NumericBinding and RetainedCancellation support functions.
Those functions own the pinned VPC/mTLS relay.  This module never accepts a
future operation payload or a caller-selected HTTP coordinate.
"""

from __future__ import annotations

from dataclasses import asdict
import base64
import json
import re
from typing import Mapping

from .canonical import canonical_json_bytes, canonical_sha256
from .task12_runtime import (
    CancellationObservation,
    build_cancellation_command,
)


_SHA = re.compile(r"^[0-9a-f]{64}$")
_VERSIONED_LAMBDA = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:"
    r"[A-Za-z0-9_-]+:([1-9][0-9]*)$"
)
_RUN_ID = "glm52-sky-20260724"

SUPPORTED_OPERATIONS = frozenset(
    {
        "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION",
    }
)


def _metadata(value: object, label: str) -> Mapping[str, object]:
    if (
        type(value) is not dict
        or value.get("HTTPStatusCode") != 200
        or type(value.get("RequestId")) is not str
        or not value["RequestId"]
        or type(value.get("RetryAttempts")) is not int
        or value["RetryAttempts"] != 0
    ):
        raise ValueError(label + " response is not authenticated zero-retry")
    return value


def _roles(ports: object) -> Mapping[str, object]:
    deployment = getattr(ports, "deployment", None)
    value = getattr(deployment, "role_coordinates", None)
    if not isinstance(value, Mapping):
        raise ValueError("Task 12 deployment role coordinates are absent")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(label + " is not one exact string")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(label + " is not one exact SHA-256")
    return value


def _task9_coordinate(value: object) -> dict[str, object]:
    try:
        from .task11_support_boundary import (
            exact_input_coordinate_from_mapping,
        )

        exact = exact_input_coordinate_from_mapping(
            value,
            expected_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Task 9 deployed identity coordinate is not exact"
        ) from exc
    return asdict(exact)


def _version_arn(value: object, label: str) -> str:
    exact = _text(value, label)
    if _VERSIONED_LAMBDA.fullmatch(exact) is None:
        raise ValueError(label + " is not an exact published Lambda version")
    return exact


def _payload_bytes(value: object) -> bytes:
    read = getattr(value, "read", None)
    if callable(read):
        value = read()
    if type(value) is bytearray:
        value = bytes(value)
    if type(value) is not bytes:
        raise ValueError("exact-version Lambda payload is absent")
    return value


def _invoke_exact(
    *,
    ports: object,
    version_arn: str,
    event: Mapping[str, object],
    expected_record_type: str,
) -> dict[str, object]:
    match = _VERSIONED_LAMBDA.fullmatch(version_arn)
    if match is None:
        raise ValueError("Lambda invocation target is not version-qualified")
    response = ports.client("lambda").invoke(
        FunctionName=version_arn,
        InvocationType="RequestResponse",
        Payload=canonical_json_bytes(event),
    )
    _metadata(response.get("ResponseMetadata"), "exact-version Lambda")
    if (
        response.get("StatusCode") != 200
        or "FunctionError" in response
        or response.get("ExecutedVersion") != match.group(1)
    ):
        raise ValueError("exact-version Lambda invocation was not successful")
    try:
        value = json.loads(_payload_bytes(response.get("Payload")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("exact-version Lambda result is malformed") from exc
    if (
        type(value) is not dict
        or value.get("record_type") != expected_record_type
        or _SHA.fullmatch(value.get("canonical_identity_sha256", ""))
        is None
    ):
        raise ValueError("exact-version Lambda result is not closed")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256")
    if canonical_sha256(body) != identity:
        raise ValueError("exact-version Lambda result identity drifted")
    return dict(value)


def _numeric_observation(
    *,
    ports: object,
    invocation: object,
    correlation_identity_sha256: str,
) -> dict[str, object]:
    roles = _roles(ports)
    event = {
        "schema_version": 1,
        "record_type": "glm52_task12_runtime_observation_request_v1",
        "run_id": _RUN_ID,
        "activation_id": getattr(invocation, "activation_id"),
        "correlation_identity_sha256": correlation_identity_sha256,
        "task9_deployed_identity_coordinate": _task9_coordinate(
            roles.get("task9_deployed_identity_coordinate"),
        ),
        "task9_deployed_identity_sha256": _sha(
            roles.get("task9_deployed_identity_sha256"),
            "Task 9 deployed identity",
        ),
    }
    result = _invoke_exact(
        ports=ports,
        version_arn=_version_arn(
            roles.get("numeric_binding_version_arn"),
            "NumericBinding version ARN",
        ),
        event=event,
        expected_record_type="glm52_task12_runtime_observation_v1",
    )
    if (
        result.get("run_id") != _RUN_ID
        or result.get("activation_id") != event["activation_id"]
        or result.get("correlation_identity_sha256")
        != correlation_identity_sha256
        or result.get("task9_deployed_identity_sha256")
        != event["task9_deployed_identity_sha256"]
    ):
        raise ValueError("numeric observation authority binding drifted")
    return result


def _members(
    observation: Mapping[str, object], collection: str
) -> list[dict[str, object]]:
    snapshot = observation.get(collection + "_snapshot")
    values = snapshot.get(collection + "s") if type(snapshot) is dict else None
    if type(values) is not list or any(type(item) is not dict for item in values):
        raise ValueError(collection + " observation is not one complete set")
    return [dict(item) for item in values]


def _cancel_target(
    observation: Mapping[str, object],
) -> tuple[str, str, str]:
    requests = _members(observation, "request")
    jobs = _members(observation, "job")
    request_ids = [item.get("request_id") for item in requests]
    job_ids = [item.get("job_id") for item in jobs]
    if (
        request_ids != sorted(request_ids)
        or len(request_ids) != len(set(request_ids))
        or job_ids != sorted(job_ids, key=lambda item: int(item))
        or len(job_ids) != len(set(job_ids))
    ):
        raise ValueError("runtime cancellation target set is not exact")
    request_nonterminal = [
        item for item in requests if item.get("state") not in {"FAILED", "CANCELLED"}
    ]
    job_nonterminal = [
        item
        for item in jobs
        if item.get("state") not in {"FAILED", "CANCELLED", "SUCCEEDED"}
    ]
    if len(request_nonterminal) == 1:
        item = request_nonterminal[0]
        return "REQUEST_CANCEL", _text(item.get("request_id"), "request ID"), _text(
            item.get("state"), "request state"
        )
    if not request_nonterminal and len(job_nonterminal) == 1:
        item = job_nonterminal[0]
        return "JOB_CANCEL", _text(item.get("job_id"), "job ID"), _text(
            item.get("state"), "job state"
        )
    terminal = requests or jobs
    if not request_nonterminal and not job_nonterminal and terminal:
        item = terminal[0]
        if "request_id" in item and "job_id" not in item:
            return (
                "REQUEST_CANCEL",
                _text(item.get("request_id"), "request ID"),
                _text(item.get("state"), "request state"),
            )
        return (
            "JOB_CANCEL",
            _text(item.get("job_id"), "job ID"),
            _text(item.get("state"), "job state"),
        )
    raise ValueError("runtime cancellation target cardinality is not closed")


def _observed_state(
    observation: Mapping[str, object], *, kind: str, target_id: str
) -> str:
    collection, key = (
        ("request", "request_id")
        if kind == "REQUEST_CANCEL"
        else ("job", "job_id")
    )
    matches = [
        item
        for item in _members(observation, collection)
        if item.get(key) == target_id
    ]
    if len(matches) != 1:
        raise ValueError("post-cancellation target observation is not exact")
    return _text(matches[0].get("state"), "post-cancellation target state")


def _materialize_cancellation(
    *,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> dict[str, object]:
    correlation = live_sources.get("request_job_correlation")
    if (
        type(correlation) is not dict
        or correlation.get("record_type")
        != "glm52_task12_request_job_correlation_v1"
    ):
        raise ValueError("request/job correlation source is absent")
    correlation_identity = correlation.get("canonical_body_sha256")
    _sha(correlation_identity, "request/job correlation identity")
    request_ids = correlation.get("request_ids")
    job_ids = correlation.get("job_ids")
    request_states = correlation.get("request_states")
    job_states = correlation.get("job_states")
    if (
        type(request_ids) is not list
        or type(job_ids) is not list
        or type(request_states) is not dict
        or type(job_states) is not dict
    ):
        raise ValueError("request/job correlation set is not exact")
    first = {
        "request_snapshot": {
            "requests": [
                {"request_id": item, "state": request_states.get(item)}
                for item in request_ids
            ]
        },
        "job_snapshot": {
            "jobs": [
                {"job_id": item, "state": job_states.get(item)}
                for item in job_ids
            ]
        },
        "canonical_identity_sha256": correlation_identity,
    }
    kind, target_id, state = _cancel_target(first)
    command = build_cancellation_command(
        kind=kind,
        target_id=target_id,
        action_key=(
            "ACTIVATION#"
            + getattr(invocation, "activation_id")
            + "#RECOVERY_ACTION#"
            + kind
            + "#"
            + target_id
        ),
        invocation_nonce_sha256=_sha(
            getattr(invocation, "dispatch_identity_sha256", None),
            "Task 12 dispatch identity",
        ),
        correlation_identity_sha256=correlation_identity,
    )
    terminal = state in (
        {"FAILED", "CANCELLED"}
        if kind == "REQUEST_CANCEL"
        else {"FAILED", "CANCELLED", "SUCCEEDED"}
    )
    cancellation: dict[str, object] | None = None
    if not terminal:
        roles = _roles(ports)
        raw = canonical_json_bytes(command.body)
        event = {
            "schema_version": 1,
            "record_type": "glm52_task11_retained_cancellation_request_v1",
            "run_id": _RUN_ID,
            "activation_id": getattr(invocation, "activation_id"),
            "path": command.path,
            "request_body_base64": base64.b64encode(raw).decode("ascii"),
            "request_body_sha256": canonical_sha256(command.body),
            "task9_deployed_identity_coordinate": _task9_coordinate(
                roles.get("task9_deployed_identity_coordinate"),
            ),
            "task9_deployed_identity_sha256": _sha(
                roles.get("task9_deployed_identity_sha256"),
                "Task 9 deployed identity",
            ),
        }
        cancellation = _invoke_exact(
            ports=ports,
            version_arn=_version_arn(
                roles.get("retained_cancellation_version_arn"),
                "RetainedCancellation version ARN",
            ),
            event=event,
            expected_record_type=(
                "glm52_task11_retained_cancellation_result_v1"
            ),
        )
        if (
            cancellation.get("run_id") != _RUN_ID
            or cancellation.get("activation_id")
            != getattr(invocation, "activation_id")
            or cancellation.get("path") != command.path
            or cancellation.get("request_body_sha256")
            != event["request_body_sha256"]
            or cancellation.get("task9_deployed_identity_sha256")
            != event["task9_deployed_identity_sha256"]
            or cancellation.get("status_code") not in {200, 202}
        ):
            raise ValueError("retained cancellation result binding drifted")
        second = _numeric_observation(
            ports=ports,
            invocation=invocation,
            correlation_identity_sha256=correlation_identity,
        )
        state = _observed_state(second, kind=kind, target_id=target_id)
    observation = CancellationObservation(
        classification="NOT_SENT" if cancellation is None else "DIRECT",
        send_count=0 if cancellation is None else 1,
        response_request_id=(
            None if cancellation is None else cancellation["request_id"]
        ),
        response_identity_sha256=(
            None
            if cancellation is None
            else cancellation["canonical_identity_sha256"]
        ),
        observed_target_state=state,
        observation_identity_sha256=canonical_sha256(
            {
                "first_observation": correlation_identity,
                "cancellation": (
                    None
                    if cancellation is None
                    else cancellation["canonical_identity_sha256"]
                ),
                "observed_target_state": state,
            }
        ),
    )
    return {"command": asdict(command), "observation": asdict(observation)}


def materialize_live_request(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> Mapping[str, object] | None:
    if operation_kind not in SUPPORTED_OPERATIONS:
        return None
    if operation_kind == "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION":
        return _materialize_cancellation(
            invocation=invocation,
            live_sources=live_sources,
            ports=ports,
        )
    raise ValueError(
        operation_kind + " lacks a published direct live observation boundary"
    )


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
    # The authorized support Lambdas and strict domain effects persist their
    # own canonical evidence.  This adapter never writes a future request.
    del invocation, live_sources, request, domain_result, ports
    return True


__all__ = [
    "SUPPORTED_OPERATIONS",
    "materialize_live_request",
    "persist_live_successors",
]
