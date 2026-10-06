"""Closed envelopes for the support-owned post-Task11 continuation.

This graph is intentionally separate from the immutable 32-operation retained
Task 12 registry.  It gives the support Lambdas one exact invocation/result
chain without pretending that support effects are retained recovery effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import canonical_sha256


SUPPORT_OPERATION_SEQUENCE = (
    "VALIDATE_APPROVED_TASK11_HANDOFF",
    "NUMERIC_BINDING_HANDOFF",
    "BIND_NUMERIC_JOB_OR_PROVE_NONE",
    "RUNTIME_OBSERVATION",
    "DRAIN_ARMED",
    "DRAIN_REQUESTED",
    "SUPPORT_REQUEST_CANCEL_RECONCILIATION",
    "SUPPORT_JOB_CANCEL_RECONCILIATION",
    "SUPPORT_CONTROLLER_QUIESCE_BEFORE_FORCE_IF_REQUIRED",
    "WORKER_TERMINAL_RECONCILIATION",
    "ALLOCATION_CLOSED",
    "CREATE_OR_RECONCILE_PRODUCTION_TERMINAL_V2",
    "REQUEST_SUPPORT_FINALIZATION",
)
_PREDECESSORS = {
    operation: SUPPORT_OPERATION_SEQUENCE[index - 1] if index else None
    for index, operation in enumerate(SUPPORT_OPERATION_SEQUENCE)
}


class SupportContinuationContractError(ValueError):
    """A support continuation invocation/result edge was not exact."""


@dataclass(frozen=True)
class SupportContinuationInvocation:
    operation_kind: str
    operation_input: Mapping[str, Any]


def _self_hash(value: Mapping[str, Any], label: str) -> str:
    identity = value.get("canonical_body_sha256")
    if type(identity) is not str:
        raise SupportContinuationContractError(label + " identity is absent")
    body = dict(value)
    body.pop("canonical_body_sha256", None)
    if identity != canonical_sha256(body):
        raise SupportContinuationContractError(label + " identity drifted")
    return identity


def parse_support_invocation(event: object) -> SupportContinuationInvocation:
    """Parse one exact ASL invocation and authenticate its predecessor."""

    if type(event) is not dict or set(event) != {
        "schema_version",
        "record_type",
        "operation_kind",
        "operation_input",
    }:
        raise SupportContinuationContractError(
            "support continuation invocation field set drifted"
        )
    if (
        event["schema_version"] != 1
        or event["record_type"]
        != "glm52_task12_support_continuation_invocation_v1"
        or event["operation_kind"] not in SUPPORT_OPERATION_SEQUENCE
        or type(event["operation_input"]) is not dict
    ):
        raise SupportContinuationContractError(
            "support continuation invocation is not closed"
        )
    operation_kind = event["operation_kind"]
    operation_input = dict(event["operation_input"])
    prior = operation_input.get("support_last_result")
    predecessor = _PREDECESSORS[operation_kind]
    if predecessor is None:
        if prior is not None:
            raise SupportContinuationContractError(
                "support continuation root consumed a predecessor"
            )
    elif (
        type(prior) is not dict
        or set(prior)
        != {
            "schema_version",
            "record_type",
            "operation_kind",
            "outcome",
            "result",
            "canonical_body_sha256",
        }
        or prior.get("schema_version") != 1
        or prior.get("record_type")
        != "glm52_task12_support_continuation_result_v1"
        or prior.get("operation_kind") != predecessor
        or prior.get("outcome") != "SUCCEEDED"
        or type(prior.get("result")) is not dict
    ):
        raise SupportContinuationContractError(
            "support continuation predecessor is foreign"
        )
    else:
        _self_hash(prior, "support continuation predecessor")
    return SupportContinuationInvocation(
        operation_kind=operation_kind,
        operation_input=operation_input,
    )


def build_support_result(
    *,
    invocation: SupportContinuationInvocation,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact ResultPath object consumed by the next support state."""

    if (
        type(invocation) is not SupportContinuationInvocation
        or invocation.operation_kind not in SUPPORT_OPERATION_SEQUENCE
        or type(result) is not dict
    ):
        raise SupportContinuationContractError(
            "support continuation result is not closed"
        )
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_support_continuation_result_v1",
        "operation_kind": invocation.operation_kind,
        "outcome": "SUCCEEDED",
        "result": dict(result),
    }
    return {
        **body,
        "canonical_body_sha256": canonical_sha256(body),
    }


__all__ = [
    "SUPPORT_OPERATION_SEQUENCE",
    "SupportContinuationContractError",
    "SupportContinuationInvocation",
    "build_support_result",
    "parse_support_invocation",
]
