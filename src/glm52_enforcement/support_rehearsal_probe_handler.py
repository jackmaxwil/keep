"""Stateless read-only mTLS probe for deployed Task 11 rehearsals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Optional, Tuple

from .canonical import canonical_json_bytes, canonical_sha256
from .sky_admission import RelayIdentity


_FIELDS = {
    "schema_version",
    "record_type",
    "activation_id",
    "measurement_id",
    "candidate_bucket",
    "candidate_key",
    "candidate_version_id",
    "candidate_body_sha256",
    "candidate_checksum_sha256_base64",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_VERSION_ARN = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:"
    r"keep-glm52-h1g-rehearsal-probe:[1-9][0-9]*$"
)
_MEASUREMENT = re.compile(r"^measurement-(?:0[1-9]|1[0-9]|20)$")


class RehearsalProbeError(ValueError):
    """The read-only rehearsal probe failed closed."""


@dataclass(frozen=True)
class RehearsalProbeConfig:
    account_id: str
    region: str
    run_id: str
    activation_id: str
    rehearsal_bucket: str
    function_version_arn: str
    client_certificate_sha256: str
    server_certificate_sha256: str


@dataclass(frozen=True)
class RehearsalProbeRuntime:
    function_version_arn: str
    lambda_request_id: str


@dataclass(frozen=True)
class RehearsalProbeServices:
    relay: object


@dataclass(frozen=True)
class _RelayIdentityEnvelope:
    identities: Tuple[RelayIdentity, ...]


def _validate(
    config: RehearsalProbeConfig,
    runtime: RehearsalProbeRuntime,
) -> None:
    if (
        type(config) is not RehearsalProbeConfig
        or config.account_id != "246813579024"
        or config.region != "us-west-2"
        or config.run_id != "glm52-sky-20260724"
        or type(config.activation_id) is not str
        or not config.activation_id
        or config.rehearsal_bucket
        != "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
        or _VERSION_ARN.fullmatch(config.function_version_arn) is None
        or _SHA.fullmatch(config.client_certificate_sha256) is None
        or _SHA.fullmatch(config.server_certificate_sha256) is None
        or type(runtime) is not RehearsalProbeRuntime
        or runtime.function_version_arn != config.function_version_arn
        or _UUID.fullmatch(runtime.lambda_request_id) is None
    ):
        raise RehearsalProbeError("rehearsal probe identity drifted")


def _response(
    value: object,
    *,
    path: str,
    server_certificate_sha256: str,
) -> tuple[dict[str, object], str, str]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "status_code",
            "request_id",
            "raw",
            "response_sha256",
            "tls_peer_certificate_sha256",
        }
        or value["status_code"] != 200
        or type(value["request_id"]) is not str
        or not value["request_id"]
        or type(value["raw"]) is not bytes
        or not value["raw"]
        or value["response_sha256"]
        != hashlib.sha256(value["raw"]).hexdigest()
        or value["tls_peer_certificate_sha256"]
        != server_certificate_sha256
    ):
        raise RehearsalProbeError(path + " direct response drifted")
    try:
        parsed = json.loads(value["raw"].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RehearsalProbeError(path + " response was not JSON") from exc
    if (
        type(parsed) is not dict
        or canonical_json_bytes(parsed) != value["raw"]
    ):
        raise RehearsalProbeError(path + " response was not canonical")
    return parsed, value["request_id"], value["response_sha256"]


def _request_identity(event: dict[str, object], path: str) -> str:
    return canonical_sha256(
        {
            "method": "GET",
            "path": path,
            "activation_id": event["activation_id"],
            "measurement_id": event["measurement_id"],
            "candidate_bucket": event["candidate_bucket"],
            "candidate_key": event["candidate_key"],
            "candidate_version_id": event["candidate_version_id"],
            "candidate_body_sha256": event["candidate_body_sha256"],
        }
    )


def inspect_rehearsal_probe(
    event: object,
    *,
    config: RehearsalProbeConfig,
    runtime: RehearsalProbeRuntime,
    services: RehearsalProbeServices,
) -> dict[str, object]:
    """Execute exactly two stateless GETs over the read-only relay path."""

    if type(event) is not dict or set(event) != _FIELDS:
        raise RehearsalProbeError("rehearsal probe event field set drifted")
    _validate(config, runtime)
    if (
        event["schema_version"] != 1
        or event["record_type"] != "glm52_task11_rehearsal_probe_v1"
        or event["activation_id"] != config.activation_id
        or type(event["measurement_id"]) is not str
        or _MEASUREMENT.fullmatch(event["measurement_id"]) is None
        or event["candidate_bucket"] != config.rehearsal_bucket
        or type(event["candidate_key"]) is not str
        or not event["candidate_key"].startswith(
            "rehearsal/canary/" + config.activation_id + "/"
        )
        or not event["candidate_key"].endswith(
            "/" + event["measurement_id"] + ".json"
        )
        or type(event["candidate_version_id"]) is not str
        or not event["candidate_version_id"]
        or type(event["candidate_body_sha256"]) is not str
        or _SHA.fullmatch(event["candidate_body_sha256"]) is None
        or type(event["candidate_checksum_sha256_base64"]) is not str
        or not event["candidate_checksum_sha256_base64"]
        or type(services) is not RehearsalProbeServices
        or not callable(getattr(services.relay, "request", None))
    ):
        raise RehearsalProbeError("rehearsal probe request drifted")
    health_raw = services.relay.request(
        method="GET",
        path="/api/health",
        body=None,
    )
    role_raw = services.relay.request(
        method="GET",
        path="/users/role",
        body=None,
    )
    health, health_request_id, health_body_sha256 = _response(
        health_raw,
        path="health",
        server_certificate_sha256=config.server_certificate_sha256,
    )
    role, role_request_id, role_body_sha256 = _response(
        role_raw,
        path="role",
        server_certificate_sha256=config.server_certificate_sha256,
    )
    if (
        set(health)
        != {
            "tls_peer_certificate_sha256",
            "effective_controller_identity_sha256",
            "observed_at",
        }
        or health["tls_peer_certificate_sha256"]
        != config.server_certificate_sha256
        or _SHA.fullmatch(
            str(health["effective_controller_identity_sha256"])
        )
        is None
        or type(health["observed_at"]) is not str
        or not health["observed_at"]
        or set(role)
        != {
            "sky_user_identity",
            "sky_roles",
            "token_expires_at",
        }
        or type(role["sky_user_identity"]) is not str
        or not role["sky_user_identity"]
        or role["sky_roles"] != ["user"]
        or type(role["token_expires_at"]) is not str
        or not role["token_expires_at"]
    ):
        raise RehearsalProbeError("read-only Sky identity response drifted")
    body = {
        "schema_version": 1,
        "record_type": "glm52_task11_rehearsal_probe_result_v1",
        "activation_id": event["activation_id"],
        "measurement_id": event["measurement_id"],
        "candidate_bucket": event["candidate_bucket"],
        "candidate_key": event["candidate_key"],
        "candidate_version_id": event["candidate_version_id"],
        "candidate_body_sha256": event["candidate_body_sha256"],
        "executed_version_arn": runtime.function_version_arn,
        "probe_lambda_request_id": runtime.lambda_request_id,
        "health_path": "/api/health",
        "health_request_identity_sha256": _request_identity(
            event,
            "/api/health",
        ),
        "health_response_request_id": health_request_id,
        "health_status_code": 200,
        "health_response_body_sha256": health_body_sha256,
        "role_path": "/users/role",
        "role_request_identity_sha256": _request_identity(
            event,
            "/users/role",
        ),
        "role_response_request_id": role_request_id,
        "role_status_code": 200,
        "role_response_body_sha256": role_body_sha256,
        "relay_call_count": 0,
        "sky_post_call_count": 0,
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def _configuration() -> RehearsalProbeConfig:
    account_id = os.environ.get("GLM52_ACCOUNT_ID", "")
    region = os.environ.get("AWS_REGION", "")
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
    function_version = os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "")
    return RehearsalProbeConfig(
        account_id=account_id,
        region=region,
        run_id=os.environ.get("GLM52_RUN_ID", ""),
        activation_id=os.environ.get("GLM52_ACTIVATION_ID", ""),
        rehearsal_bucket=os.environ.get("GLM52_REHEARSAL_BUCKET", ""),
        function_version_arn=(
            f"arn:aws:lambda:{region}:{account_id}:function:"
            f"{function_name}:{function_version}"
        ),
        client_certificate_sha256=os.environ.get(
            "GLM52_ATTESTATION_CLIENT_DER_SHA256",
            "",
        ),
        server_certificate_sha256=os.environ.get(
            "GLM52_ATTESTATION_SERVER_DER_SHA256",
            "",
        ),
    )


def _production_services(
    config: RehearsalProbeConfig,
) -> RehearsalProbeServices:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    from .task11_relay_runtime import PinnedMtlsRelay

    secretsmanager = boto3.client(
        "secretsmanager",
        region_name=config.region,
        config=Config(
            connect_timeout=1,
            read_timeout=4,
            max_pool_connections=2,
            retries={"mode": "standard", "total_max_attempts": 1},
        ),
    )
    identity = RelayIdentity(
        purpose="ATTESTATION",
        port=18443,
        principal_arn=(
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-rehearsal-probe"
        ),
        client_certificate_sha256=config.client_certificate_sha256,
        server_certificate_sha256=config.server_certificate_sha256,
        allowed_paths=("/api/health", "/users/role"),
        allow_imds=False,
        allow_shell=False,
        allow_aws_credentials=False,
    )
    return RehearsalProbeServices(
        relay=PinnedMtlsRelay(
            secretsmanager=secretsmanager,
            purpose="ATTESTATION",
            identity=_RelayIdentityEnvelope((identity,)),
        )
    )


def main(
    event: object,
    context: object,
    *,
    config: Optional[RehearsalProbeConfig] = None,
    services: Optional[RehearsalProbeServices] = None,
) -> dict[str, object]:
    exact_config = _configuration() if config is None else config
    request_id = getattr(context, "aws_request_id", None)
    invoked_arn = getattr(context, "invoked_function_arn", None)
    if (
        type(request_id) is not str
        or invoked_arn != exact_config.function_version_arn
    ):
        raise RehearsalProbeError("probe Lambda runtime identity drifted")
    return inspect_rehearsal_probe(
        event,
        config=exact_config,
        runtime=RehearsalProbeRuntime(
            function_version_arn=exact_config.function_version_arn,
            lambda_request_id=request_id,
        ),
        services=(
            _production_services(exact_config)
            if services is None
            else services
        ),
    )


__all__ = [
    "RehearsalProbeConfig",
    "RehearsalProbeError",
    "RehearsalProbeRuntime",
    "RehearsalProbeServices",
    "inspect_rehearsal_probe",
    "main",
]
