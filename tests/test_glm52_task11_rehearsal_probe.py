"""Stateless, read-only deployed Task 11 rehearsal probe."""

from __future__ import annotations

import hashlib
import json

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.support_rehearsal_probe_handler import (
    RehearsalProbeConfig,
    RehearsalProbeError,
    RehearsalProbeRuntime,
    RehearsalProbeServices,
    inspect_rehearsal_probe,
)


ACCOUNT = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
ACTIVATION = "approved-20260728"
VERSION_ARN = (
    "arn:aws:lambda:us-west-2:246813579024:function:"
    "keep-glm52-h1g-rehearsal-probe:7"
)
BUCKET = "keep-glm52-h1g-rehearsal-246813579024-us-west-2"


class Relay:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def request(
        self,
        *,
        method: str,
        path: str,
        body: object,
    ) -> dict[str, object]:
        self.calls.append((method, path, body))
        if path == "/api/health":
            value = {
                "tls_peer_certificate_sha256": "a" * 64,
                "effective_controller_identity_sha256": "b" * 64,
                "observed_at": "2026-07-29T20:00:00Z",
            }
            request_id = "host-health-request"
        else:
            value = {
                "sky_user_identity": "keep-glm52-production",
                "sky_roles": ["user"],
                "token_expires_at": "2026-07-29T21:00:00Z",
            }
            request_id = "host-role-request"
        raw = canonical_json_bytes(value)
        return {
            "status_code": 200,
            "request_id": request_id,
            "raw": raw,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "tls_peer_certificate_sha256": "a" * 64,
        }


def _event() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_task11_rehearsal_probe_v1",
        "activation_id": ACTIVATION,
        "measurement_id": "measurement-01",
        "candidate_bucket": BUCKET,
        "candidate_key": (
            "rehearsal/canary/approved-20260728/"
            + "c" * 64
            + "/measurement-01.json"
        ),
        "candidate_version_id": "version-0001",
        "candidate_body_sha256": "d" * 64,
        "candidate_checksum_sha256_base64": (
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        ),
    }


def _config() -> RehearsalProbeConfig:
    return RehearsalProbeConfig(
        account_id=ACCOUNT,
        region=REGION,
        run_id=RUN_ID,
        activation_id=ACTIVATION,
        rehearsal_bucket=BUCKET,
        function_version_arn=VERSION_ARN,
        client_certificate_sha256="a" * 64,
        server_certificate_sha256="a" * 64,
    )


def test_probe_executes_only_two_gets_and_binds_direct_request_ids() -> None:
    relay = Relay()
    result = inspect_rehearsal_probe(
        _event(),
        config=_config(),
        runtime=RehearsalProbeRuntime(
            function_version_arn=VERSION_ARN,
            lambda_request_id="00000000-0000-4000-8000-000000000007",
        ),
        services=RehearsalProbeServices(relay=relay),
    )
    assert relay.calls == [
        ("GET", "/api/health", None),
        ("GET", "/users/role", None),
    ]
    assert result["executed_version_arn"] == VERSION_ARN
    assert result["probe_lambda_request_id"] == (
        "00000000-0000-4000-8000-000000000007"
    )
    assert result["activation_id"] == ACTIVATION
    assert result["measurement_id"] == "measurement-01"
    assert result["candidate_bucket"] == BUCKET
    assert result["candidate_key"] == _event()["candidate_key"]
    assert result["candidate_version_id"] == "version-0001"
    assert result["candidate_body_sha256"] == "d" * 64
    assert result["health_path"] == "/api/health"
    assert result["role_path"] == "/users/role"
    assert result["health_response_request_id"] == "host-health-request"
    assert result["role_response_request_id"] == "host-role-request"
    assert result["health_status_code"] == 200
    assert result["role_status_code"] == 200
    assert result["health_response_body_sha256"]
    assert result["role_response_body_sha256"]
    assert result["health_request_identity_sha256"]
    assert result["role_request_identity_sha256"]
    assert (
        result["health_request_identity_sha256"]
        != result["role_request_identity_sha256"]
    )
    assert result["relay_call_count"] == 0
    assert result["sky_post_call_count"] == 0
    body = dict(result)
    identity = body.pop("canonical_identity_sha256")
    assert identity == hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def test_probe_rejects_caller_effect_or_post_surface_before_relay() -> None:
    relay = Relay()
    value = _event()
    value["method"] = "POST"
    with pytest.raises(RehearsalProbeError, match="field set"):
        inspect_rehearsal_probe(
            value,
            config=_config(),
            runtime=RehearsalProbeRuntime(
                function_version_arn=VERSION_ARN,
                lambda_request_id=(
                    "00000000-0000-4000-8000-000000000007"
                ),
            ),
            services=RehearsalProbeServices(relay=relay),
        )
    assert relay.calls == []


def test_probe_rejects_noncanonical_host_payload() -> None:
    class NoncanonicalRelay(Relay):
        def request(
            self,
            *,
            method: str,
            path: str,
            body: object,
        ) -> dict[str, object]:
            result = super().request(method=method, path=path, body=body)
            parsed = json.loads(result["raw"])
            raw = json.dumps(parsed, indent=2).encode()
            result["raw"] = raw
            result["response_sha256"] = hashlib.sha256(raw).hexdigest()
            return result

    with pytest.raises(RehearsalProbeError, match="canonical"):
        inspect_rehearsal_probe(
            _event(),
            config=_config(),
            runtime=RehearsalProbeRuntime(
                function_version_arn=VERSION_ARN,
                lambda_request_id=(
                    "00000000-0000-4000-8000-000000000007"
                ),
            ),
            services=RehearsalProbeServices(relay=NoncanonicalRelay()),
        )
