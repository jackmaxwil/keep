"""Executable Task 11 Decision Lambda entrypoint."""

from __future__ import annotations

from dataclasses import asdict
import os
import re
from typing import Mapping, Optional

from .canonical import canonical_sha256
from .decision_closure import (
    build_closure_request,
    execute_decision_closure,
)


_TRUSTED_GATE_FIELDS = {
    "schema_version",
    "record_type",
    "status",
    "account_id",
    "region",
    "run_id",
    "activation_id",
    "bucket",
    "key",
    "version_id",
    "file_sha256",
    "body_sha256",
    "checksum_sha256_base64",
    "deployment_identity_sha256",
    "measurement_count",
    "cold_environment_count",
    "environment_set_sha256",
    "measurements_identity_sha256",
    "canonical_identity_sha256",
}
_REQUEST_FIELDS = {
    "activation_id",
    "generation",
    "candidate_identity_sha256",
    "initial_source_predecessor_version_id",
    "admission_version_arn",
    "numeric_binding_version_arn",
    "task11_boundary",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _validate_trusted_gate(
    value: object,
    *,
    activation_id: str,
) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != _TRUSTED_GATE_FIELDS:
        raise ValueError("trusted budget authority field set drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256")
    if (
        value["schema_version"] != 1
        or value["record_type"]
        != "glm52_task11_trusted_budget_authority_v1"
        or value["status"] != "CLOSURE_BUDGET_PROVEN"
        or value["account_id"] != "246813579024"
        or value["region"] != "us-west-2"
        or value["run_id"] != "glm52-sky-20260724"
        or value["activation_id"] != activation_id
        or value["measurement_count"] != 20
        or value["cold_environment_count"] < 5
        or type(identity) is not str
        or identity != canonical_sha256(body)
    ):
        raise ValueError("trusted budget authority is invalid")
    deployed = os.environ.get(
        "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
    )
    if (
        type(deployed) is not str
        or _SHA.fullmatch(deployed) is None
        or value["deployment_identity_sha256"] != deployed
    ):
        raise ValueError(
            "trusted budget authority deployment identity is absent or foreign"
        )
    return value


def _build_production_services(
    event: Mapping[str, object],
    context: object,
) -> object:
    from .task11_production import build_task11_production_services

    return build_task11_production_services(event=event, context=context)


def main(
    event: object,
    context: object,
    *,
    services: Optional[object] = None,
    monotonic_clock: Optional[object] = None,
    sleeper: Optional[object] = None,
) -> Mapping[str, object]:
    """Parse trusted workflow input and run the one-shot private closure."""

    if type(event) is not dict or set(event) != {
        "schema_version",
        "record_type",
        "closure_request",
        "trusted_closure_budget",
    }:
        raise ValueError("decision handler event field set drifted")
    if (
        event["schema_version"] != 1
        or event["record_type"]
        != "glm52_task11_decision_handler_request_v1"
        or type(event["closure_request"]) is not dict
        or set(event["closure_request"]) != _REQUEST_FIELDS
    ):
        raise ValueError("decision handler event is not closed")
    request_fields = event["closure_request"]
    activation_id = request_fields["activation_id"]
    if type(activation_id) is not str or not activation_id:
        raise ValueError("decision handler activation is invalid")
    trusted = _validate_trusted_gate(
        event["trusted_closure_budget"],
        activation_id=activation_id,
    )
    request = build_closure_request(
        activation_id=activation_id,
        generation=request_fields["generation"],
        candidate_identity_sha256=(
            request_fields["candidate_identity_sha256"]
        ),
        initial_source_predecessor_version_id=(
            request_fields["initial_source_predecessor_version_id"]
        ),
        admission_version_arn=request_fields["admission_version_arn"],
        numeric_binding_version_arn=(
            request_fields["numeric_binding_version_arn"]
        ),
        closure_budget_status=trusted["status"],
    )
    if services is None:
        services = _build_production_services(event, context)
    kwargs = {
        "services": services,
    }
    if monotonic_clock is not None:
        kwargs["monotonic_clock"] = monotonic_clock
    if sleeper is not None:
        kwargs["sleeper"] = sleeper
    result = execute_decision_closure(request, **kwargs)
    return asdict(result)


__all__ = ["main"]
