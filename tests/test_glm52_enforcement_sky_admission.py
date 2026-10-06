from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.sky_admission import (
    ACCOUNT_ID,
    EXPECTED_SKYPILOT_VERSION,
    REGION,
    RUN_ID,
    AdmissionError,
    AdmissionRequest,
    AttestationError,
    AttestationRequest,
    PinnedSkyIdentity,
    SkyAdmissionService,
    SkyAttestationService,
    Task9SkyRelayProbe,
    build_admission_custody_identity,
    build_default_identity_contract,
    build_pinned_sky_identity,
    validate_pinned_sky_identity,
    validate_sky_identity_contract,
)


SHA = "a" * 64
SHA_B = "b" * 64
NOW = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)


def _pinned() -> PinnedSkyIdentity:
    return build_pinned_sky_identity(
        skypilot_version=EXPECTED_SKYPILOT_VERSION,
        wheel_metadata_sha256="1" * 64,
        dist_record_sha256="2" * 64,
        interpreter_path=(
            "/Users/jack.mazac/.local/share/keep/"
            "skypilot-0.13.0/bin/python"
        ),
        interpreter_sha256="3" * 64,
        dependency_lock_sha256="4" * 64,
        server_config_sha256="5" * 64,
        original_provisioner_sha256=(
            "fc5d2e4b94f97c10babb583859da24a4fb19256807e763e064e1f294d6c442ef"
        ),
        patched_provisioner_sha256="6" * 64,
        jobs_server_sha256=(
            "fe63ab110c28c3e368de56a700d437918a7105cf2db893d814f9627be6331955"
        ),
        jobs_launch_retryable=False,
    )


def _relay_pins() -> dict[str, tuple[str, str]]:
    return {
        "ATTESTATION": ("1" * 64, "5" * 64),
        "LAUNCH_ADMISSION": ("2" * 64, "6" * 64),
        "NUMERIC_BINDING": ("3" * 64, "7" * 64),
        "RETAINED_CANCELLATION": ("4" * 64, "8" * 64),
    }


def test_task9_red_pinned_sky_identity_module_exists() -> None:
    assert validate_pinned_sky_identity(_pinned()) == _pinned()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("skypilot_version", "0.13.1"),
        ("wheel_metadata_sha256", "x" * 64),
        ("interpreter_path", "/usr/bin/python3"),
        ("original_provisioner_sha256", "9" * 64),
        ("jobs_server_sha256", "8" * 64),
        ("jobs_launch_retryable", True),
    ],
)
def test_pinned_sky_identity_rejects_version_source_and_retry_mutants(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_pinned_sky_identity(replace(_pinned(), **{field: value}))


def test_loopback_rbac_effective_controller_and_four_identities_are_closed() -> None:
    contract = build_default_identity_contract(
        pinned=_pinned(),
        relay_certificate_pins=_relay_pins(),
        service_account_user_id="usr-glm52-production",
        combined_host_instance_id="i-0123456789abcdef0",
        combined_host_boot_identity_sha256="7" * 64,
        combined_host_service_identity_sha256="8" * 64,
        consolidation_signal_identity_sha256="9" * 64,
    )
    assert validate_sky_identity_contract(contract) == contract
    assert contract.backend_host == "127.0.0.1"
    assert contract.backend_port not in {item.port for item in contract.identities}
    assert len({item.port for item in contract.identities}) == 4
    assert len({item.principal_arn for item in contract.identities}) == 4
    assert len({item.client_certificate_sha256 for item in contract.identities}) == 4
    assert tuple(item.purpose for item in contract.identities) == (
        "ATTESTATION",
        "LAUNCH_ADMISSION",
        "NUMERIC_BINDING",
        "RETAINED_CANCELLATION",
    )
    assert contract.service_account_role == "user"
    assert contract.controller_resource_pin is None
    assert contract.effective_consolidation is True
    assert contract.separate_controller_count == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_host", "0.0.0.0"),
        ("service_account_role", "admin"),
        ("controller_resource_pin", "c6a.xlarge"),
        ("effective_consolidation", False),
        ("separate_controller_count", 1),
    ],
)
def test_sky_identity_contract_rejects_loopback_rbac_and_controller_mutants(
    field: str, value: object
) -> None:
    good = build_default_identity_contract(
        pinned=_pinned(),
        relay_certificate_pins=_relay_pins(),
        service_account_user_id="usr-glm52-production",
        combined_host_instance_id="i-0123456789abcdef0",
        combined_host_boot_identity_sha256="7" * 64,
        combined_host_service_identity_sha256="8" * 64,
        consolidation_signal_identity_sha256="9" * 64,
    )
    with pytest.raises((TypeError, ValueError)):
        validate_sky_identity_contract(replace(good, **{field: value}))


class _NonceRegistry:
    def __init__(self) -> None:
        self.used: set[str] = set()

    def consume(self, nonce: str) -> bool:
        if nonce in self.used:
            return False
        self.used.add(nonce)
        return True


class _Attestor:
    def inspect(self, request: AttestationRequest) -> dict[str, object]:
        return {
            "freshness_nonce": request.freshness_nonce,
            "direct_response_request_id": "attestation-request-1",
            "tls_peer_certificate_sha256": "5" * 64,
            "sky_user_identity": "usr-glm52-production",
            "sky_roles": ("user",),
            "token_expires_at": "2026-07-29T14:00:00Z",
            "effective_controller_identity_sha256": "9" * 64,
            "observed_at": "2026-07-29T12:00:01Z",
        }


def test_attestation_accepts_only_closed_freshness_nonce_and_rejects_replay() -> None:
    registry = _NonceRegistry()
    service = SkyAttestationService(
        identity=build_default_identity_contract(
            pinned=_pinned(),
            relay_certificate_pins=_relay_pins(),
            service_account_user_id="usr-glm52-production",
            combined_host_instance_id="i-0123456789abcdef0",
            combined_host_boot_identity_sha256="7" * 64,
            combined_host_service_identity_sha256="8" * 64,
            consolidation_signal_identity_sha256="9" * 64,
        ),
        nonce_registry=registry,
        attestor=_Attestor(),
    )
    request = AttestationRequest(freshness_nonce="n" * 64)
    result = service.attest(request)
    assert result.freshness_nonce == request.freshness_nonce
    with pytest.raises(AttestationError, match="fresh|replay"):
        service.attest(request)
    with pytest.raises((TypeError, AttestationError)):
        service.attest({"freshness_nonce": "m" * 64, "command": "id"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tls_peer_certificate_sha256", "6" * 64),
        ("effective_controller_identity_sha256", "8" * 64),
    ],
)
def test_attestation_rejects_wrong_deployed_peer_or_controller(
    field: str,
    value: str,
) -> None:
    class MutantAttestor(_Attestor):
        def inspect(self, request: AttestationRequest) -> dict[str, object]:
            return {**super().inspect(request), field: value}

    service = SkyAttestationService(
        identity=build_default_identity_contract(
            pinned=_pinned(),
            relay_certificate_pins=_relay_pins(),
            service_account_user_id="usr-glm52-production",
            combined_host_instance_id="i-0123456789abcdef0",
            combined_host_boot_identity_sha256="7" * 64,
            combined_host_service_identity_sha256="8" * 64,
            consolidation_signal_identity_sha256="9" * 64,
        ),
        nonce_registry=_NonceRegistry(),
        attestor=MutantAttestor(),
    )
    with pytest.raises(AttestationError, match="peer|controller"):
        service.attest(AttestationRequest(freshness_nonce="x" * 64))


class _AdmissionStore:
    def __init__(self, *, crash: str | None = None) -> None:
        self.state = "CONSUMED"
        self.crash = crash
        self.post_count = 0
        self.private_nonce = None

    def own_post_started(self, *, request, private_nonce_sha256):
        if self.state != "CONSUMED":
            return self.state
        self.state = "POST_STARTED"
        self.private_nonce = private_nonce_sha256
        if self.crash == "after_started":
            raise RuntimeError("process died")
        return self.state

    def authorize_post(self, *, request, audit, authority, relay_envelope_sha256):
        assert len(self.private_nonce) == 64
        assert authority["current_owner"] is True
        assert authority["reserve_held"] is True
        assert authority["action_consumed"] is True
        if self.state != "POST_STARTED":
            return self.state
        self.state = "POST_AUTHORIZED"
        if self.crash == "after_authorized":
            raise RuntimeError("process died")
        return self.state

    def classify(
        self,
        *,
        request,
        classification,
        response_identity_sha256,
        request_id,
    ):
        self.state = "POST_CLASSIFIED"
        return self.state

    def read(self, request):
        return self.state


class _FreshAudit:
    def inspect(self, request):
        return {
            "fresh": True,
            "audit_identity_sha256": "c" * 64,
            "closing_revision": 17,
        }


class _CurrentAuthority:
    def __init__(self) -> None:
        self.calls = 0

    def inspect(self, request):
        self.calls += 1
        return {
            "current_owner": True,
            "reserve_held": True,
            "action_consumed": True,
            "live_h1d_identity_sha256": (
                request.live_h1d_identity_sha256
            ),
            "reserve_identity_sha256": "e" * 64,
            "owner_identity_sha256": (
                build_admission_custody_identity(request)
            ),
        }


class _Relay:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.calls = 0
        self.requests: list[bytes] = []

    def send(self, body: bytes):
        self.calls += 1
        self.requests.append(body)
        if self.outcome == "timeout":
            raise TimeoutError("ambiguous")
        if self.outcome == "connection-loss":
            raise ConnectionError("ambiguous")
        if self.outcome == "rejected":
            return {
                "status_code": 412,
                "request_id": "sky-rejected-1",
                "response_sha256": "1" * 64,
            }
        return {
            "status_code": 202,
            "request_id": "sky-accepted-1",
            "response_sha256": "2" * 64,
        }


def _admission_request() -> AdmissionRequest:
    return AdmissionRequest(
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id="activation-0001",
        generation=1,
        attempt=1,
        action_key="ACTION#00000001#SKY_POST#00000001",
        request_body_sha256="1" * 64,
        relay_envelope_sha256="2" * 64,
        attestation_identity_sha256="3" * 64,
        direct_decision_identity_sha256="4" * 64,
        live_h1d_identity_sha256="5" * 64,
        decision_seal_identity_sha256="6" * 64,
        decision_nonce_sha256="7" * 64,
        barrier_transition_identity_sha256="8" * 64,
        private_nonce=b"admission-caller-private-nonce",
    )


@pytest.mark.parametrize(
    ("relay_outcome", "classification"),
    [
        ("accepted", "ACCEPTED"),
        ("rejected", "KNOWN_REJECTED"),
        ("timeout", "AMBIGUOUS"),
        ("connection-loss", "AMBIGUOUS"),
    ],
)
def test_admission_matrix_issues_at_most_one_post(
    relay_outcome: str, classification: str
) -> None:
    store = _AdmissionStore()
    relay = _Relay(relay_outcome)
    authority = _CurrentAuthority()
    service = SkyAdmissionService(
        store=store,
        fresh_audit=_FreshAudit(),
        current_authority=authority,
        relay=relay,
        nonce_source=lambda: b"private-admission-owner",
    )
    result = service.submit(_admission_request())
    assert result.classification == classification
    assert relay.calls == 1
    assert authority.calls == 1
    replay = service.submit(_admission_request())
    assert replay.classification == "RECONCILE_ONLY"
    assert relay.calls == 1


@pytest.mark.parametrize("crash", ["after_started", "after_authorized"])
def test_admission_process_death_never_permits_second_send(crash: str) -> None:
    store = _AdmissionStore(crash=crash)
    relay = _Relay("accepted")
    service = SkyAdmissionService(
        store=store,
        fresh_audit=_FreshAudit(),
        current_authority=_CurrentAuthority(),
        relay=relay,
        nonce_source=lambda: b"private-admission-owner",
    )
    with pytest.raises(RuntimeError, match="process died"):
        service.submit(_admission_request())
    store.crash = None
    result = service.submit(_admission_request())
    assert result.classification == "RECONCILE_ONLY"
    assert relay.calls == 0


def test_admission_rejects_caller_asserted_probe_success_as_authority() -> None:
    class _RejectAuthority:
        def inspect(self, request):
            raise AdmissionError("fresh Task 8 authority absent")

    service = SkyAdmissionService(
        store=_AdmissionStore(),
        fresh_audit=_FreshAudit(),
        current_authority=_RejectAuthority(),
        relay=_Relay("accepted"),
        nonce_source=lambda: b"private-admission-owner",
    )
    with pytest.raises(AdmissionError, match="Task 8"):
        service.submit(_admission_request(), caller_asserted_probe_success=True)


def test_task9_probe_returns_task8_typed_result_not_caller_boolean() -> None:
    contract = build_default_identity_contract(
        pinned=_pinned(),
        relay_certificate_pins=_relay_pins(),
        service_account_user_id="usr-glm52-production",
        combined_host_instance_id="i-0123456789abcdef0",
        combined_host_boot_identity_sha256="7" * 64,
        combined_host_service_identity_sha256="8" * 64,
        consolidation_signal_identity_sha256="9" * 64,
    )
    probe = Task9SkyRelayProbe(
        attestation=SkyAttestationService(
            identity=contract,
            nonce_registry=_NonceRegistry(),
            attestor=_Attestor(),
        ),
        admission_identity_sha256=contract.canonical_identity_sha256,
        nonce_source=lambda: "q" * 64,
    )
    request = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": "activation-0001",
    }
    result = probe.inspect(request)
    assert result.request_identity_sha256 == canonical_sha256(request)
    assert result.sky_roles == ("user",)
    assert result.sky_user_identity == "usr-glm52-production"


def test_task9_probe_rejects_obsolete_three_field_request_schema() -> None:
    contract = build_default_identity_contract(
        pinned=_pinned(),
        relay_certificate_pins=_relay_pins(),
        service_account_user_id="usr-glm52-production",
        combined_host_instance_id="i-0123456789abcdef0",
        combined_host_boot_identity_sha256="7" * 64,
        combined_host_service_identity_sha256="8" * 64,
        consolidation_signal_identity_sha256="9" * 64,
    )
    probe = Task9SkyRelayProbe(
        attestation=SkyAttestationService(
            identity=contract,
            nonce_registry=_NonceRegistry(),
            attestor=_Attestor(),
        ),
        admission_identity_sha256=contract.canonical_identity_sha256,
        nonce_source=lambda: "q" * 64,
    )
    with pytest.raises(AttestationError, match="closed"):
        probe.inspect(
            {
                "activation_id": "activation-0001",
                "generation": 1,
                "request_kind": "H1D_SKY_PROBE",
            }
        )
