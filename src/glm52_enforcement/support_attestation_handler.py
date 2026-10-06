"""Exact-version Task 11 Sky identity attestation Lambda."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from typing import Mapping, Optional

from .sky_admission import (
    AttestationRequest,
    AttestationResult,
    SkyAttestationService,
)


_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AttestationHandlerServices:
    attestation: SkyAttestationService
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
        or values["function_name"] != "keep-glm52-h1g-attestation"
        or not values["function_version"].isdigit()
        or values["function_version"].startswith("0")
    ):
        raise RuntimeError("attestation Lambda identity drifted")
    return values


def main(
    event: object,
    context: object,
    *,
    services: Optional[AttestationHandlerServices] = None,
    config: Optional[Mapping[str, str]] = None,
) -> Mapping[str, object]:
    """Consume one caller nonce and return the accepted typed attestation."""

    del context
    if (
        type(event) is not dict
        or set(event)
        != {
            "schema_version",
            "record_type",
            "run_id",
            "activation_id",
            "request_identity_sha256",
            "freshness_nonce",
            "sequence",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
        }
        or event["schema_version"] != 1
        or event["record_type"]
        != "glm52_task11_attestation_request_v1"
        or event["run_id"] != "glm52-sky-20260724"
        or type(event["activation_id"]) is not str
        or not event["activation_id"]
        or _SHA.fullmatch(event["request_identity_sha256"]) is None
        or _SHA.fullmatch(event["freshness_nonce"]) is None
        or type(event["sequence"]) is not int
        or event["sequence"] not in {1, 2, 3}
        or _SHA.fullmatch(
            event["task9_deployed_identity_sha256"]
        )
        is None
    ):
        raise ValueError("attestation request is not closed")
    exact = dict(_configuration() if config is None else config)
    if (
        exact.get("run_id") != event["run_id"]
        or exact.get("activation_id") != event["activation_id"]
    ):
        raise ValueError("attestation invocation target drifted")
    if services is None:
        from .task11_production import build_attestation_handler_services

        services = build_attestation_handler_services(
            task9_deployed_identity_coordinate=event[
                "task9_deployed_identity_coordinate"
            ],
            task9_deployed_identity_sha256=event[
                "task9_deployed_identity_sha256"
            ],
        )
    if type(services) is not AttestationHandlerServices:
        raise TypeError("attestation handler services are not exact")
    if (
        services.task9_deployed_identity_sha256
        != event["task9_deployed_identity_sha256"]
    ):
        raise ValueError("attestation deployed identity body drifted")
    result = services.attestation.attest(
        AttestationRequest(event["freshness_nonce"])
    )
    if type(result) is not AttestationResult:
        raise ValueError("attestation service result is not exact")
    return asdict(result)


__all__ = ["AttestationHandlerServices", "main"]
