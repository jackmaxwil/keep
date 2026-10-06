"""Exact-version Task 11 one-wire Sky launch-admission Lambda."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import os
import re
from typing import Callable, Mapping, Optional

from .canonical import canonical_sha256
from .decision_closure import CLOSURE_STEPS
from .h1f_adapter import H1fAuditResult, validate_h1f_audit_result
from .sky_admission import (
    AdmissionRequest,
    AdmissionResult,
    SkyAdmissionService,
)
from .task11_support_boundary import (
    exact_input_coordinate_from_mapping,
    load_exact_input,
)


_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AdmissionHandlerServices:
    s3: object
    admission: SkyAdmissionService
    evidence: Callable[[], Mapping[str, object]]
    task9_deployed_identity_sha256: str


def _configuration() -> Mapping[str, str]:
    values = {
        "account_id": os.environ.get("GLM52_ACCOUNT_ID", ""),
        "region": os.environ.get("AWS_REGION", ""),
        "run_id": os.environ.get("GLM52_RUN_ID", ""),
        "activation_id": os.environ.get("GLM52_ACTIVATION_ID", ""),
        "function_name": os.environ.get("AWS_LAMBDA_FUNCTION_NAME", ""),
        "function_version": os.environ.get(
            "AWS_LAMBDA_FUNCTION_VERSION",
            "",
        ),
    }
    if (
        values["account_id"] != "246813579024"
        or values["region"] != "us-west-2"
        or values["run_id"] != "glm52-sky-20260724"
        or not values["activation_id"]
        or values["function_name"]
        != "keep-glm52-h1g-launch-admission"
        or not values["function_version"].isdigit()
        or values["function_version"].startswith("0")
    ):
        raise RuntimeError("launch-admission Lambda identity drifted")
    return values


def _document(
    value: object,
    *,
    activation_id: str,
    generation: int,
) -> Mapping[str, object]:
    expected = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "generation",
        "action_key",
        "request_body_sha256",
        "relay_envelope_sha256",
        "decision_seal_identity_sha256",
        "canonical_identity_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("launch-admission input field set drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256")
    if (
        value["schema_version"] != 1
        or value["record_type"]
        != "glm52_task11_sky_admission_input_v1"
        or value["account_id"] != "246813579024"
        or value["region"] != "us-west-2"
        or value["run_id"] != "glm52-sky-20260724"
        or value["activation_id"] != activation_id
        or value["generation"] != generation
        or type(value["action_key"]) is not str
        or "SKY_POST" not in value["action_key"]
        or any(
            _SHA.fullmatch(value[field]) is None
            for field in (
                "request_body_sha256",
                "relay_envelope_sha256",
                "decision_seal_identity_sha256",
            )
        )
        or identity != canonical_sha256(body)
    ):
        raise ValueError("launch-admission input identity drifted")
    return value


def main(
    event: object,
    context: object,
    *,
    services: Optional[AdmissionHandlerServices] = None,
    config: Optional[Mapping[str, str]] = None,
) -> Mapping[str, object]:
    """Own one consumed reservation, audit it, and issue at most one POST."""

    del context
    if (
        type(event) is not dict
        or set(event)
        != {
            "schema_version",
            "record_type",
            "run_id",
            "activation_id",
            "generation",
            "input_coordinate",
            "decision_nonce_sha256",
            "decision_nonce_base64",
            "direct_decision_identity_sha256",
            "live_h1d_identity_sha256",
            "attestation_identity_sha256",
            "decision_seal_identity_sha256",
            "barrier_transition_identity_sha256",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
        }
        or event["schema_version"] != 1
        or event["record_type"] != "glm52_task11_admission_request_v1"
        or event["run_id"] != "glm52-sky-20260724"
        or type(event["activation_id"]) is not str
        or not event["activation_id"]
        or type(event["generation"]) is not int
        or event["generation"] <= 0
        or any(
            _SHA.fullmatch(event[field]) is None
            for field in (
                "decision_nonce_sha256",
                "direct_decision_identity_sha256",
                "live_h1d_identity_sha256",
                "attestation_identity_sha256",
                "decision_seal_identity_sha256",
                "barrier_transition_identity_sha256",
                "task9_deployed_identity_sha256",
            )
        )
        or type(event["decision_nonce_base64"]) is not str
    ):
        raise ValueError("launch-admission request is not closed")
    try:
        private_nonce = base64.b64decode(
            event["decision_nonce_base64"],
            validate=True,
        )
    except ValueError as exc:
        raise ValueError("launch-admission nonce encoding drifted") from exc
    if (
        len(private_nonce) < 16
        or hashlib.sha256(private_nonce).hexdigest()
        != event["decision_nonce_sha256"]
    ):
        raise ValueError("launch-admission nonce custody drifted")
    exact = dict(_configuration() if config is None else config)
    if (
        exact.get("run_id") != event["run_id"]
        or exact.get("activation_id") != event["activation_id"]
    ):
        raise ValueError("launch-admission invocation target drifted")
    coordinate = exact_input_coordinate_from_mapping(
        event["input_coordinate"],
        expected_kind="SKY_ADMISSION_REQUEST",
    )
    if services is None:
        from .task11_production import build_admission_handler_services

        services = build_admission_handler_services(
            generation=event["generation"],
            decision_nonce=private_nonce,
            live_h1d_identity_sha256=event[
                "live_h1d_identity_sha256"
            ],
            task9_deployed_identity_coordinate=event[
                "task9_deployed_identity_coordinate"
            ],
            task9_deployed_identity_sha256=event[
                "task9_deployed_identity_sha256"
            ],
        )
    if type(services) is not AdmissionHandlerServices:
        raise TypeError("launch-admission handler services are not exact")
    if (
        services.task9_deployed_identity_sha256
        != event["task9_deployed_identity_sha256"]
    ):
        raise ValueError("launch-admission deployed identity body drifted")
    document = _document(
        load_exact_input(s3=services.s3, coordinate=coordinate),
        activation_id=event["activation_id"],
        generation=event["generation"],
    )
    if (
        document["decision_seal_identity_sha256"]
        != event["decision_seal_identity_sha256"]
    ):
        raise ValueError(
            "launch-admission decision seal substitution detected"
        )
    result = services.admission.submit(
        AdmissionRequest(
            account_id="246813579024",
            region="us-west-2",
            run_id=event["run_id"],
            activation_id=event["activation_id"],
            generation=event["generation"],
            attempt=1,
            action_key=document["action_key"],
            request_body_sha256=document["request_body_sha256"],
            relay_envelope_sha256=document[
                "relay_envelope_sha256"
            ],
            attestation_identity_sha256=event[
                "attestation_identity_sha256"
            ],
            direct_decision_identity_sha256=event[
                "direct_decision_identity_sha256"
            ],
            live_h1d_identity_sha256=event[
                "live_h1d_identity_sha256"
            ],
            decision_seal_identity_sha256=document[
                "decision_seal_identity_sha256"
            ],
            decision_nonce_sha256=event["decision_nonce_sha256"],
            barrier_transition_identity_sha256=event[
                "barrier_transition_identity_sha256"
            ],
            private_nonce=private_nonce,
        )
    )
    evidence = services.evidence()
    if (
        type(result) is not AdmissionResult
        or result.classification == "RECONCILE_ONLY"
        or type(evidence) is not dict
        or set(evidence)
        != {
            "post_audit",
            "action_states",
            "relay_call_count",
        }
        or type(evidence["post_audit"]) is not H1fAuditResult
        or tuple(evidence["action_states"])
        != ("POST_STARTED", "POST_AUTHORIZED", "POST_CLASSIFIED")
        or evidence["relay_call_count"] != 1
    ):
        raise ValueError("launch-admission evidence is incomplete")
    audit = validate_h1f_audit_result(evidence["post_audit"])
    relay_receipt = (
        None
        if result.classification == "AMBIGUOUS"
        else canonical_sha256(
            {
                "request_id": result.request_id,
                "response_identity_sha256": (
                    result.response_identity_sha256
                ),
            }
        )
    )
    post_audit = {
        "kind": "SKY_POST",
        "audit_identity_sha256": audit.canonical_body_sha256,
        "invocation_identity_sha256": canonical_sha256(
            {
                "decision_nonce_sha256": event[
                    "decision_nonce_sha256"
                ],
                "result": asdict(result),
            }
        ),
        "closing_revision": audit.closing_revision,
    }
    return {
        "task9_result": asdict(result),
        "internal_steps": list(CLOSURE_STEPS[20:24]),
        "action_states": list(evidence["action_states"]),
        "post_audit": post_audit,
        "classification": result.classification,
        "invocation_count": 1,
        "relay_call_count": 1,
        "retry_count": 0,
        "rearm_count": 0,
        "relay_receipt_sha256": relay_receipt,
    }


__all__ = ["AdmissionHandlerServices", "main"]
