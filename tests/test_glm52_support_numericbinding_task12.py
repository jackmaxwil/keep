from __future__ import annotations

import json

import pytest

from glm52_enforcement.canonical import canonical_sha256


SHA_A = "a" * 64
SHA_B = "b" * 64


class _Relay:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def request(self, *, method: str, path: str, body: object):
        self.calls.append((method, path, body))
        document = {
            "/requests/exact": {
                "observed_at": "2026-07-29T12:00:00Z",
                "pagination_complete": True,
                "requests": [
                    {
                        "request_id": "request-1",
                        "state": "CANCELLED",
                        "created_at": "2026-07-29T10:00:00Z",
                        "updated_at": "2026-07-29T11:59:00Z",
                        "evidence_identity_sha256": SHA_A,
                    }
                ],
                "controller_snapshot": {
                    "members": [],
                    "evidence_identity_sha256": SHA_B,
                },
            },
            "/jobs/exact": {
                "observed_at": "2026-07-29T12:00:00Z",
                "pagination_complete": True,
                "jobs": [
                    {
                        "job_id": "17",
                        "request_id": "request-1",
                        "state": "CANCELLED",
                        "evidence_identity_sha256": SHA_B,
                    }
                ],
                "controller_snapshot": {
                    "members": [],
                    "evidence_identity_sha256": SHA_B,
                },
            },
        }[path]
        raw = json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return {
            "status_code": 200,
            "request_id": path,
            "raw": raw,
            "response_sha256": canonical_sha256(document),
            "tls_peer_certificate_sha256": SHA_A,
        }


def _event() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_task12_runtime_observation_request_v1",
        "run_id": "glm52-sky-20260724",
        "activation_id": "activation-1",
        "correlation_identity_sha256": SHA_A,
        "task9_deployed_identity_coordinate": "s3://bucket/task9.json",
        "task9_deployed_identity_sha256": SHA_B,
    }


def test_task12_observation_uses_only_two_fixed_numeric_binding_gets() -> None:
    from glm52_enforcement.support_numericbinding_handler import (
        NumericBindingServices,
        main,
    )

    relay = _Relay()
    result = main(
        _event(),
        object(),
        config={
            "account_id": "246813579024",
            "region": "us-west-2",
            "run_id": "glm52-sky-20260724",
            "activation_id": "activation-1",
            "campaign_bucket": "bucket",
            "function_name": "keep-glm52-h1g-numeric-binding",
            "function_version": "7",
        },
        services=NumericBindingServices(
            task9_deployed_identity_sha256=SHA_B,
            relay=relay,
        ),
    )

    assert relay.calls == [
        ("GET", "/requests/exact", None),
        ("GET", "/jobs/exact", None),
    ]
    assert result["record_type"] == "glm52_task12_runtime_observation_v1"
    assert result["request_snapshot"]["pagination_complete"] is True
    assert result["job_snapshot"]["pagination_complete"] is True


def test_task12_observation_rejects_incomplete_or_swapped_snapshot() -> None:
    from glm52_enforcement.support_numericbinding_handler import (
        NumericBindingServices,
        main,
    )

    class BadRelay(_Relay):
        def request(self, *, method: str, path: str, body: object):
            response = dict(super().request(method=method, path=path, body=body))
            if path == "/jobs/exact":
                document = json.loads(response["raw"])
                document["pagination_complete"] = False
                response["raw"] = json.dumps(
                    document, sort_keys=True, separators=(",", ":")
                ).encode("ascii")
            return response

    with pytest.raises(ValueError, match="snapshot"):
        main(
            _event(),
            object(),
            config={
                "account_id": "246813579024",
                "region": "us-west-2",
                "run_id": "glm52-sky-20260724",
                "activation_id": "activation-1",
                "campaign_bucket": "bucket",
                "function_name": "keep-glm52-h1g-numeric-binding",
                "function_version": "7",
            },
            services=NumericBindingServices(
                task9_deployed_identity_sha256=SHA_B,
                relay=BadRelay(),
            ),
        )


def test_task11_numeric_reconciliation_contract_is_preserved() -> None:
    from glm52_enforcement.support_numericbinding_handler import (
        NumericBindingServices,
        main,
    )

    result = main(
        {
            "schema_version": 1,
            "record_type": "glm52_task11_numeric_binding_request_v1",
            "run_id": "glm52-sky-20260724",
            "activation_id": "activation-1",
            "admission_identity_sha256": SHA_A,
            "task9_deployed_identity_coordinate": {
                "fixture": "coordinate"
            },
            "task9_deployed_identity_sha256": SHA_B,
        },
        object(),
        config={
            "account_id": "246813579024",
            "region": "us-west-2",
            "run_id": "glm52-sky-20260724",
            "activation_id": "activation-1",
            "campaign_bucket": "bucket",
            "function_name": "keep-glm52-h1g-numeric-binding",
            "function_version": "7",
        },
        services=NumericBindingServices(
            task9_deployed_identity_sha256=SHA_B,
        ),
    )

    assert result["record_type"] == (
        "glm52_task11_numeric_reconciliation_ack_v1"
    )
    assert result["binding_state"] == "RECONCILIATION_REQUIRED"
