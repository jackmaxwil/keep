"""Exact-version Task 11 handoff into numeric-binding reconciliation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

from .canonical import canonical_sha256


_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "activation_id",
    "admission_identity_sha256",
    "task9_deployed_identity_coordinate",
    "task9_deployed_identity_sha256",
}
_TASK12_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "activation_id",
    "correlation_identity_sha256",
    "task9_deployed_identity_coordinate",
    "task9_deployed_identity_sha256",
}


@dataclass(frozen=True)
class NumericBindingServices:
    task9_deployed_identity_sha256: str
    relay: Optional[object] = None


def _snapshot(
    response: object, *, collection: str
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        type(response) is not dict
        or response.get("status_code") != 200
        or type(response.get("request_id")) is not str
        or not response["request_id"]
        or _SHA.fullmatch(response.get("response_sha256", "")) is None
        or _SHA.fullmatch(
            response.get("tls_peer_certificate_sha256", "")
        )
        is None
        or type(response.get("raw")) is not bytes
    ):
        raise ValueError("numeric-binding snapshot transport is unauthenticated")
    try:
        value = json.loads(response["raw"].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("numeric-binding snapshot is malformed") from exc
    fields = {
        "observed_at",
        "pagination_complete",
        collection,
        "controller_snapshot",
    }
    controller = value.get("controller_snapshot") if type(value) is dict else None
    entries = value.get(collection) if type(value) is dict else None
    if (
        type(value) is not dict
        or set(value) != fields
        or value["pagination_complete"] is not True
        or type(entries) is not list
        or type(controller) is not dict
        or set(controller) != {"members", "evidence_identity_sha256"}
        or type(controller["members"]) is not list
        or _SHA.fullmatch(
            controller.get("evidence_identity_sha256", "")
        )
        is None
        or canonical_sha256(value) != response["response_sha256"]
    ):
        raise ValueError("numeric-binding snapshot is not closed")
    item_fields = (
        {
            "request_id",
            "state",
            "created_at",
            "updated_at",
            "evidence_identity_sha256",
        }
        if collection == "requests"
        else {
            "job_id",
            "request_id",
            "state",
            "evidence_identity_sha256",
        }
    )
    identity_field = "request_id" if collection == "requests" else "job_id"
    identities: list[str] = []
    for item in entries:
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
            or (
                collection == "requests"
                and (
                    _UTC.fullmatch(item.get("created_at", "")) is None
                    or _UTC.fullmatch(item.get("updated_at", "")) is None
                )
            )
            or (
                collection == "jobs"
                and (
                    type(item.get("request_id")) is not str
                    or not item["request_id"]
                    or re.fullmatch(r"[1-9][0-9]*", item["job_id"]) is None
                )
            )
        ):
            raise ValueError("numeric-binding snapshot member is not closed")
        identities.append(item[identity_field])
    ordered = (
        sorted(identities)
        if collection == "requests"
        else sorted(identities, key=int)
    )
    if identities != ordered or len(identities) != len(set(identities)):
        raise ValueError("numeric-binding snapshot member set is not exact")
    return dict(value), {
        "request_id": response["request_id"],
        "response_sha256": response["response_sha256"],
        "tls_peer_certificate_sha256": response[
            "tls_peer_certificate_sha256"
        ],
    }


def _configuration() -> Mapping[str, str]:
    values = {
        "account_id": os.environ.get("GLM52_ACCOUNT_ID", ""),
        "region": os.environ.get("AWS_REGION", ""),
        "run_id": os.environ.get("GLM52_RUN_ID", ""),
        "activation_id": os.environ.get("GLM52_ACTIVATION_ID", ""),
        "campaign_bucket": os.environ.get("GLM52_CAMPAIGN_BUCKET", ""),
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
        or not values["campaign_bucket"]
        or values["function_name"]
        != "keep-glm52-h1g-numeric-binding"
        or not values["function_version"].isdigit()
        or values["function_version"].startswith("0")
    ):
        raise RuntimeError("numeric-binding Lambda identity drifted")
    return values


def main(
    event: object,
    context: object,
    *,
    config: Mapping[str, str] | None = None,
    services: NumericBindingServices | None = None,
) -> Mapping[str, object]:
    """Acknowledge reconciliation custody without claiming a numeric binding."""

    del context
    if type(event) is not dict:
        raise ValueError("numeric-binding request is not closed")
    task12 = event.get("record_type") == (
        "glm52_task12_runtime_observation_request_v1"
    )
    if (
        set(event) != (_TASK12_FIELDS if task12 else _FIELDS)
        or event["schema_version"] != 1
        or (
            not task12
            and event["record_type"]
            != "glm52_task11_numeric_binding_request_v1"
        )
        or event["run_id"] != "glm52-sky-20260724"
        or type(event["activation_id"]) is not str
        or not event["activation_id"]
        or (
            task12
            and _SHA.fullmatch(
                event["correlation_identity_sha256"]
            )
            is None
        )
        or (
            not task12
            and _SHA.fullmatch(event["admission_identity_sha256"]) is None
        )
        or _SHA.fullmatch(event["task9_deployed_identity_sha256"]) is None
    ):
        raise ValueError("numeric-binding request is not closed")
    exact = dict(_configuration() if config is None else config)
    if (
        exact.get("account_id") != "246813579024"
        or exact.get("region") != "us-west-2"
        or exact.get("run_id") != event["run_id"]
        or exact.get("activation_id") != event["activation_id"]
    ):
        raise ValueError("numeric-binding invocation target drifted")
    if services is None:
        from .task11_relay_runtime import (
            aws_relay_runtime_clients,
            load_task9_deployed_identity,
        )

        clients = aws_relay_runtime_clients()
        loaded = load_task9_deployed_identity(
            s3=clients.s3,
            coordinate=event["task9_deployed_identity_coordinate"],
            activation_id=event["activation_id"],
            expected_body_sha256=event[
                "task9_deployed_identity_sha256"
            ],
            expected_bucket=exact.get("campaign_bucket", ""),
        )
        relay = None
        if task12:
            from .task11_relay_runtime import PinnedMtlsRelay

            relay = PinnedMtlsRelay(
                secretsmanager=clients.secretsmanager,
                purpose="NUMERIC_BINDING",
                identity=loaded.sky_identity,
            )
        services = NumericBindingServices(
            task9_deployed_identity_sha256=loaded.body_sha256,
            relay=relay,
        )
    if (
        type(services) is not NumericBindingServices
        or services.task9_deployed_identity_sha256
        != event["task9_deployed_identity_sha256"]
    ):
        raise ValueError("numeric-binding deployed identity body drifted")
    if task12:
        request = getattr(services.relay, "request", None)
        if not callable(request):
            raise TypeError("numeric-binding observation relay is absent")
        request_snapshot, request_transport = _snapshot(
            request(method="GET", path="/requests/exact", body=None),
            collection="requests",
        )
        job_snapshot, job_transport = _snapshot(
            request(method="GET", path="/jobs/exact", body=None),
            collection="jobs",
        )
        if (
            request_snapshot["observed_at"] != job_snapshot["observed_at"]
            or request_snapshot["controller_snapshot"]
            != job_snapshot["controller_snapshot"]
            or request_transport["tls_peer_certificate_sha256"]
            != job_transport["tls_peer_certificate_sha256"]
        ):
            raise ValueError("numeric-binding snapshots are not coherent")
        body = {
            "schema_version": 1,
            "record_type": "glm52_task12_runtime_observation_v1",
            "account_id": "246813579024",
            "region": "us-west-2",
            "run_id": event["run_id"],
            "activation_id": event["activation_id"],
            "correlation_identity_sha256": event[
                "correlation_identity_sha256"
            ],
            "task9_deployed_identity_sha256": event[
                "task9_deployed_identity_sha256"
            ],
            "observed_at": request_snapshot["observed_at"],
            "request_snapshot": request_snapshot,
            "job_snapshot": job_snapshot,
            "controller_snapshot": request_snapshot[
                "controller_snapshot"
            ],
            "request_transport": request_transport,
            "job_transport": job_transport,
        }
        return {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
    body = {
        "schema_version": 1,
        "record_type": "glm52_task11_numeric_reconciliation_ack_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": event["run_id"],
        "activation_id": event["activation_id"],
        "admission_identity_sha256": event[
            "admission_identity_sha256"
        ],
        "task9_deployed_identity_sha256": event[
            "task9_deployed_identity_sha256"
        ],
        "binding_state": "RECONCILIATION_REQUIRED",
        "numeric_job_id": None,
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


__all__ = ["NumericBindingServices", "main"]
