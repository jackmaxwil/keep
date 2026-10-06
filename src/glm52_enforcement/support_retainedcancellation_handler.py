"""Exact-version retained Sky cancellation relay Lambda."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Mapping, Optional

from .canonical import canonical_json_bytes, canonical_sha256


_SHA = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class RetainedCancellationServices:
    relay: object
    task9_deployed_identity_sha256: str


def _production_services(
    *,
    coordinate: object,
    body_sha256: str,
    activation_id: str,
    campaign_bucket: str,
) -> RetainedCancellationServices:
    from .task11_relay_runtime import (
        PinnedMtlsRelay,
        aws_relay_runtime_clients,
        load_task9_deployed_identity,
    )

    clients = aws_relay_runtime_clients()
    loaded = load_task9_deployed_identity(
        s3=clients.s3,
        coordinate=coordinate,
        activation_id=activation_id,
        expected_body_sha256=body_sha256,
        expected_bucket=campaign_bucket,
    )
    return RetainedCancellationServices(
        relay=PinnedMtlsRelay(
            secretsmanager=clients.secretsmanager,
            purpose="RETAINED_CANCELLATION",
            identity=loaded.sky_identity,
        ),
        task9_deployed_identity_sha256=loaded.body_sha256,
    )


def main(
    event: object,
    context: object,
    *,
    services: Optional[RetainedCancellationServices] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> Mapping[str, object]:
    """Send one already-authorized cancellation over its pinned mTLS path."""

    del context
    env = dict(os.environ if environment is None else environment)
    if (
        type(event) is not dict
        or set(event)
        != {
            "schema_version",
            "record_type",
            "run_id",
            "activation_id",
            "path",
            "request_body_base64",
            "request_body_sha256",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
        }
        or event["schema_version"] != 1
        or event["record_type"]
        != "glm52_task11_retained_cancellation_request_v1"
        or event["run_id"] != "glm52-sky-20260724"
        or event["activation_id"] != env.get("GLM52_ACTIVATION_ID")
        or event["path"] not in {"/api/cancel", "/jobs/cancel"}
        or _SHA.fullmatch(event["request_body_sha256"]) is None
        or _SHA.fullmatch(event["task9_deployed_identity_sha256"]) is None
        or env.get("GLM52_RUN_ID") != event["run_id"]
        or not env.get("GLM52_CAMPAIGN_BUCKET")
    ):
        raise ValueError("retained cancellation request is not closed")
    try:
        body = base64.b64decode(
            event["request_body_base64"],
            validate=True,
        )
        document = json.loads(body)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("retained cancellation body is malformed") from exc
    if (
        type(document) is not dict
        or body != canonical_json_bytes(document)
        or hashlib.sha256(body).hexdigest()
        != event["request_body_sha256"]
    ):
        raise ValueError("retained cancellation body identity drifted")
    runtime = (
        _production_services(
            coordinate=event["task9_deployed_identity_coordinate"],
            body_sha256=event["task9_deployed_identity_sha256"],
            activation_id=event["activation_id"],
            campaign_bucket=env["GLM52_CAMPAIGN_BUCKET"],
        )
        if services is None
        else services
    )
    if type(runtime) is not RetainedCancellationServices:
        raise TypeError("retained cancellation services are not exact")
    if (
        runtime.task9_deployed_identity_sha256
        != event["task9_deployed_identity_sha256"]
    ):
        raise ValueError("retained cancellation deployed identity body drifted")
    request = getattr(runtime.relay, "request", None)
    if not callable(request):
        raise TypeError("retained cancellation relay is absent")
    response = request(method="POST", path=event["path"], body=body)
    if (
        type(response) is not dict
        or response.get("status_code") not in {200, 202}
        or type(response.get("request_id")) is not str
        or not response["request_id"]
        or _SHA.fullmatch(response.get("response_sha256", "")) is None
        or _SHA.fullmatch(
            response.get("tls_peer_certificate_sha256", "")
        )
        is None
    ):
        raise ValueError("retained cancellation response is unauthenticated")
    result = {
        "schema_version": 1,
        "record_type": "glm52_task11_retained_cancellation_result_v1",
        "run_id": event["run_id"],
        "activation_id": event["activation_id"],
        "path": event["path"],
        "request_body_sha256": event["request_body_sha256"],
        "task9_deployed_identity_sha256": event[
            "task9_deployed_identity_sha256"
        ],
        "status_code": response["status_code"],
        "request_id": response["request_id"],
        "response_sha256": response["response_sha256"],
        "tls_peer_certificate_sha256": response[
            "tls_peer_certificate_sha256"
        ],
    }
    return {**result, "canonical_identity_sha256": canonical_sha256(result)}


__all__ = ["RetainedCancellationServices", "main"]
