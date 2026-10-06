"""Closed SkyPilot identity, attestation, and launch-admission contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import re
from typing import Callable, Mapping, Optional, Tuple

from .canonical import canonical_json_bytes, canonical_sha256
from .live_authority import SkyRelayProbeResult, build_sky_relay_probe_result


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
EXPECTED_SKYPILOT_VERSION = "0.13.0"
EXPECTED_ORIGINAL_PROVISIONER_SHA256 = (
    "fc5d2e4b94f97c10babb583859da24a4fb19256807e763e064e1f294d6c442ef"
)
EXPECTED_JOBS_SERVER_SHA256 = (
    "fe63ab110c28c3e368de56a700d437918a7105cf2db893d814f9627be6331955"
)
EXPECTED_INTERPRETER_PATH = (
    "/Users/jack.mazac/.local/share/keep/skypilot-0.13.0/bin/python"
)
SKY_API_VERSION = 56
SKY_BACKEND_HOST = "127.0.0.1"
SKY_BACKEND_PORT = 46580
SKY_SERVICE_ACCOUNT_ROLE = "user"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_NONCE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
_INSTANCE = re.compile(r"^i-[0-9a-f]{17}$")


class SkyIdentityError(ValueError):
    """The pinned Sky or closed relay identity is invalid."""


class AttestationError(ValueError):
    """The nonce-only attestation did not authenticate."""


class AdmissionError(ValueError):
    """The closed Sky launch admission failed."""


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise SkyIdentityError(label + " must be a lowercase SHA-256")
    return value


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise SkyIdentityError(label + " must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise SkyIdentityError(label + " must be canonical UTC") from exc
    return parsed.replace(tzinfo=timezone.utc)


def _exact_nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value or not value.strip():
        raise SkyIdentityError(label + " must be a nonempty exact string")
    return value


@dataclass(frozen=True)
class PinnedSkyIdentity:
    schema_version: int
    record_type: str
    skypilot_version: str
    wheel_metadata_sha256: str
    dist_record_sha256: str
    interpreter_path: str
    interpreter_sha256: str
    dependency_lock_sha256: str
    server_config_sha256: str
    original_provisioner_sha256: str
    patched_provisioner_sha256: str
    jobs_server_sha256: str
    jobs_launch_retryable: bool
    canonical_identity_sha256: str


def _pinned_body(value: PinnedSkyIdentity) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def build_pinned_sky_identity(
    *,
    skypilot_version: str,
    wheel_metadata_sha256: str,
    dist_record_sha256: str,
    interpreter_path: str,
    interpreter_sha256: str,
    dependency_lock_sha256: str,
    server_config_sha256: str,
    original_provisioner_sha256: str,
    patched_provisioner_sha256: str,
    jobs_server_sha256: str,
    jobs_launch_retryable: bool,
) -> PinnedSkyIdentity:
    provisional = PinnedSkyIdentity(
        schema_version=1,
        record_type="glm52_pinned_skypilot_identity_v1",
        skypilot_version=skypilot_version,
        wheel_metadata_sha256=wheel_metadata_sha256,
        dist_record_sha256=dist_record_sha256,
        interpreter_path=interpreter_path,
        interpreter_sha256=interpreter_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        server_config_sha256=server_config_sha256,
        original_provisioner_sha256=original_provisioner_sha256,
        patched_provisioner_sha256=patched_provisioner_sha256,
        jobs_server_sha256=jobs_server_sha256,
        jobs_launch_retryable=jobs_launch_retryable,
        canonical_identity_sha256="",
    )
    candidate = PinnedSkyIdentity(
        **{
            **asdict(provisional),
            "canonical_identity_sha256": canonical_sha256(
                _pinned_body(provisional)
            ),
        }
    )
    return validate_pinned_sky_identity(candidate)


def validate_pinned_sky_identity(value: object) -> PinnedSkyIdentity:
    if not isinstance(value, PinnedSkyIdentity):
        raise SkyIdentityError("pinned Sky identity must be typed")
    if (
        value.schema_version != 1
        or value.record_type != "glm52_pinned_skypilot_identity_v1"
        or value.skypilot_version != EXPECTED_SKYPILOT_VERSION
        or value.interpreter_path != EXPECTED_INTERPRETER_PATH
        or value.original_provisioner_sha256
        != EXPECTED_ORIGINAL_PROVISIONER_SHA256
        or value.jobs_server_sha256 != EXPECTED_JOBS_SERVER_SHA256
        or type(value.jobs_launch_retryable) is not bool
        or value.jobs_launch_retryable
    ):
        raise SkyIdentityError("pinned Sky version/source/runtime drifted")
    for field in (
        "wheel_metadata_sha256",
        "dist_record_sha256",
        "interpreter_sha256",
        "dependency_lock_sha256",
        "server_config_sha256",
        "original_provisioner_sha256",
        "patched_provisioner_sha256",
        "jobs_server_sha256",
    ):
        _sha(getattr(value, field), field)
    if (
        value.patched_provisioner_sha256
        == value.original_provisioner_sha256
        or value.canonical_identity_sha256
        != canonical_sha256(_pinned_body(value))
    ):
        raise SkyIdentityError("patched Sky identity is absent or mismatched")
    return value


@dataclass(frozen=True)
class RelayIdentity:
    purpose: str
    port: int
    principal_arn: str
    client_certificate_sha256: str
    server_certificate_sha256: str
    allowed_paths: Tuple[str, ...]
    allow_imds: bool
    allow_shell: bool
    allow_aws_credentials: bool


@dataclass(frozen=True)
class SkyIdentityContract:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    pinned_identity_sha256: str
    backend_host: str
    backend_port: int
    api_version: int
    service_account_user_id: str
    service_account_role: str
    controller_resource_pin: Optional[str]
    combined_host_instance_id: str
    combined_host_boot_identity_sha256: str
    combined_host_service_identity_sha256: str
    effective_consolidation: bool
    consolidation_signal_identity_sha256: str
    separate_controller_count: int
    identities: Tuple[RelayIdentity, ...]
    canonical_identity_sha256: str


def _contract_body(value: SkyIdentityContract) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    body["identities"] = [asdict(item) for item in value.identities]
    return body


def build_default_identity_contract(
    *,
    pinned: PinnedSkyIdentity,
    relay_certificate_pins: Mapping[str, Tuple[str, str]],
    service_account_user_id: str,
    combined_host_instance_id: str,
    combined_host_boot_identity_sha256: str,
    combined_host_service_identity_sha256: str,
    consolidation_signal_identity_sha256: str,
) -> SkyIdentityContract:
    validate_pinned_sky_identity(pinned)
    purposes = (
        ("ATTESTATION", 18443, ("/api/health", "/users/role")),
        ("LAUNCH_ADMISSION", 18444, ("/jobs/launch",)),
        ("NUMERIC_BINDING", 18445, ("/requests/exact", "/jobs/exact")),
        ("RETAINED_CANCELLATION", 18446, ("/api/cancel", "/jobs/cancel")),
    )
    expected_purposes = tuple(item[0] for item in purposes)
    if (
        type(relay_certificate_pins) is not dict
        or tuple(relay_certificate_pins) != expected_purposes
        or any(
            type(relay_certificate_pins[purpose]) is not tuple
            or len(relay_certificate_pins[purpose]) != 2
            for purpose in expected_purposes
        )
    ):
        raise SkyIdentityError(
            "four purpose-specific deployed certificate pin pairs are required"
        )
    identities = tuple(
        RelayIdentity(
            purpose=purpose,
            port=port,
            principal_arn=(
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-" + purpose.lower().replace("_", "-")
            ),
            client_certificate_sha256=_sha(
                relay_certificate_pins[purpose][0],
                purpose + " client certificate",
            ),
            server_certificate_sha256=_sha(
                relay_certificate_pins[purpose][1],
                purpose + " server certificate",
            ),
            allowed_paths=paths,
            allow_imds=False,
            allow_shell=False,
            allow_aws_credentials=False,
        )
        for purpose, port, paths in purposes
    )
    provisional = SkyIdentityContract(
        schema_version=1,
        record_type="glm52_sky_identity_contract_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        pinned_identity_sha256=pinned.canonical_identity_sha256,
        backend_host=SKY_BACKEND_HOST,
        backend_port=SKY_BACKEND_PORT,
        api_version=SKY_API_VERSION,
        service_account_user_id=service_account_user_id,
        service_account_role=SKY_SERVICE_ACCOUNT_ROLE,
        controller_resource_pin=None,
        combined_host_instance_id=combined_host_instance_id,
        combined_host_boot_identity_sha256=(
            combined_host_boot_identity_sha256
        ),
        combined_host_service_identity_sha256=(
            combined_host_service_identity_sha256
        ),
        effective_consolidation=True,
        consolidation_signal_identity_sha256=(
            consolidation_signal_identity_sha256
        ),
        separate_controller_count=0,
        identities=identities,
        canonical_identity_sha256="",
    )
    candidate = SkyIdentityContract(
        **{
            **asdict(provisional),
            "identities": identities,
            "canonical_identity_sha256": canonical_sha256(
                _contract_body(provisional)
            ),
        }
    )
    return validate_sky_identity_contract(candidate)


def validate_sky_identity_contract(value: object) -> SkyIdentityContract:
    if not isinstance(value, SkyIdentityContract):
        raise SkyIdentityError("Sky identity contract must be typed")
    if (
        value.schema_version != 1
        or value.record_type != "glm52_sky_identity_contract_v1"
        or value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
        or value.backend_host != SKY_BACKEND_HOST
        or type(value.backend_port) is not int
        or value.backend_port <= 0
        or value.api_version != SKY_API_VERSION
        or value.service_account_role != SKY_SERVICE_ACCOUNT_ROLE
        or value.controller_resource_pin is not None
        or value.effective_consolidation is not True
        or type(value.separate_controller_count) is not int
        or value.separate_controller_count != 0
        or _INSTANCE.fullmatch(value.combined_host_instance_id) is None
    ):
        raise SkyIdentityError("loopback, RBAC, or consolidation contract drifted")
    _exact_nonempty(value.service_account_user_id, "service account user")
    for field in (
        "pinned_identity_sha256",
        "combined_host_boot_identity_sha256",
        "combined_host_service_identity_sha256",
        "consolidation_signal_identity_sha256",
    ):
        _sha(getattr(value, field), field)
    if (
        type(value.identities) is not tuple
        or tuple(item.purpose for item in value.identities)
        != (
            "ATTESTATION",
            "LAUNCH_ADMISSION",
            "NUMERIC_BINDING",
            "RETAINED_CANCELLATION",
        )
    ):
        raise SkyIdentityError("four closed relay identities are required")
    ports = set()
    principals = set()
    clients = set()
    servers = set()
    expected_paths = {
        "ATTESTATION": ("/api/health", "/users/role"),
        "LAUNCH_ADMISSION": ("/jobs/launch",),
        "NUMERIC_BINDING": ("/requests/exact", "/jobs/exact"),
        "RETAINED_CANCELLATION": ("/api/cancel", "/jobs/cancel"),
    }
    for identity in value.identities:
        if (
            not isinstance(identity, RelayIdentity)
            or type(identity.port) is not int
            or identity.port <= 0
            or identity.port == value.backend_port
            or identity.allowed_paths != expected_paths[identity.purpose]
            or identity.allow_imds
            or identity.allow_shell
            or identity.allow_aws_credentials
        ):
            raise SkyIdentityError("relay identity boundary drifted")
        _exact_nonempty(identity.principal_arn, "relay principal")
        _sha(identity.client_certificate_sha256, "client certificate")
        _sha(identity.server_certificate_sha256, "server certificate")
        ports.add(identity.port)
        principals.add(identity.principal_arn)
        clients.add(identity.client_certificate_sha256)
        servers.add(identity.server_certificate_sha256)
    if not all(len(values) == 4 for values in (ports, principals, clients, servers)):
        raise SkyIdentityError("relay identities are mutually usable")
    if value.canonical_identity_sha256 != canonical_sha256(_contract_body(value)):
        raise SkyIdentityError("Sky identity contract hash drifted")
    return value


@dataclass(frozen=True)
class AttestationRequest:
    freshness_nonce: str


@dataclass(frozen=True)
class AttestationResult:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    freshness_nonce: str
    direct_response_request_id: str
    tls_peer_certificate_sha256: str
    sky_user_identity: str
    sky_roles: Tuple[str, ...]
    token_expires_at: str
    effective_controller_identity_sha256: str
    observed_at: str
    sky_identity_contract_sha256: str
    canonical_identity_sha256: str


def _attestation_body(value: AttestationResult) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


class SkyAttestationService:
    """Verifier-nonce-only closed read service."""

    def __init__(
        self,
        *,
        identity: SkyIdentityContract,
        nonce_registry: object,
        attestor: object,
    ) -> None:
        self._identity = validate_sky_identity_contract(identity)
        self._nonces = nonce_registry
        self._attestor = attestor

    def attest(self, request: AttestationRequest) -> AttestationResult:
        if (
            not isinstance(request, AttestationRequest)
            or type(request.freshness_nonce) is not str
            or _NONCE.fullmatch(request.freshness_nonce) is None
        ):
            raise AttestationError(
                "attestation accepts only a closed freshness nonce"
            )
        consume = getattr(self._nonces, "consume", None)
        inspect = getattr(self._attestor, "inspect", None)
        if not callable(consume) or not callable(inspect):
            raise AttestationError("attestation boundary is incomplete")
        if consume(request.freshness_nonce) is not True:
            raise AttestationError("freshness nonce was replayed")
        raw = inspect(request)
        expected = {
            "freshness_nonce",
            "direct_response_request_id",
            "tls_peer_certificate_sha256",
            "sky_user_identity",
            "sky_roles",
            "token_expires_at",
            "effective_controller_identity_sha256",
            "observed_at",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise AttestationError("attestor response is not closed")
        if (
            raw["freshness_nonce"] != request.freshness_nonce
            or raw["sky_user_identity"]
            != self._identity.service_account_user_id
            or raw["sky_roles"] != (SKY_SERVICE_ACCOUNT_ROLE,)
        ):
            raise AttestationError("attested nonce or RBAC identity mismatched")
        try:
            peer = _sha(raw["tls_peer_certificate_sha256"], "TLS peer")
            effective = _sha(
                raw["effective_controller_identity_sha256"],
                "effective controller",
            )
            observed = _utc(raw["observed_at"], "observed_at")
            expires = _utc(raw["token_expires_at"], "token_expires_at")
            request_id = _exact_nonempty(
                raw["direct_response_request_id"], "request id"
            )
        except SkyIdentityError as exc:
            raise AttestationError(str(exc)) from exc
        attestation_identity = next(
            item
            for item in self._identity.identities
            if item.purpose == "ATTESTATION"
        )
        if peer != attestation_identity.server_certificate_sha256:
            raise AttestationError("attested TLS peer is not the deployed peer")
        if (
            effective
            != self._identity.consolidation_signal_identity_sha256
        ):
            raise AttestationError(
                "effective controller is not the deployed consolidation"
            )
        if expires <= observed:
            raise AttestationError("Sky service token is stale")
        provisional = AttestationResult(
            schema_version=1,
            record_type="glm52_sky_attestation_v1",
            account_id=ACCOUNT_ID,
            region=REGION,
            run_id=RUN_ID,
            freshness_nonce=request.freshness_nonce,
            direct_response_request_id=request_id,
            tls_peer_certificate_sha256=peer,
            sky_user_identity=str(raw["sky_user_identity"]),
            sky_roles=tuple(raw["sky_roles"]),
            token_expires_at=str(raw["token_expires_at"]),
            effective_controller_identity_sha256=effective,
            observed_at=str(raw["observed_at"]),
            sky_identity_contract_sha256=(
                self._identity.canonical_identity_sha256
            ),
            canonical_identity_sha256="",
        )
        return AttestationResult(
            **{
                **asdict(provisional),
                "sky_roles": provisional.sky_roles,
                "canonical_identity_sha256": canonical_sha256(
                    _attestation_body(provisional)
                ),
            }
        )


class Task9SkyRelayProbe:
    """Task 8-compatible typed adapter backed by the closed attestor."""

    def __init__(
        self,
        *,
        attestation: SkyAttestationService,
        admission_identity_sha256: str,
        nonce_source: Callable[[], str],
    ) -> None:
        self._attestation = attestation
        self._admission_identity = _sha(
            admission_identity_sha256, "admission identity"
        )
        self._nonce_source = nonce_source

    def inspect(self, request: Mapping[str, object]) -> SkyRelayProbeResult:
        if type(request) is not dict or set(request) != {
            "account_id",
            "region",
            "run_id",
            "activation_id",
        }:
            raise AttestationError("Task 9 probe request is not closed")
        if (
            request["account_id"] != ACCOUNT_ID
            or request["region"] != REGION
            or request["run_id"] != RUN_ID
            or type(request["activation_id"]) is not str
            or not request["activation_id"]
        ):
            raise AttestationError("Task 9 probe request identity is invalid")
        nonce = self._nonce_source()
        result = self._attestation.attest(AttestationRequest(nonce))
        return build_sky_relay_probe_result(
            account_id=ACCOUNT_ID,
            region=REGION,
            run_id=RUN_ID,
            request_identity_sha256=canonical_sha256(request),
            direct_response_request_id=result.direct_response_request_id,
            tls_peer_certificate_sha256=(
                result.tls_peer_certificate_sha256
            ),
            attestation_identity_sha256=result.canonical_identity_sha256,
            admission_identity_sha256=self._admission_identity,
            sky_user_identity=result.sky_user_identity,
            sky_roles=result.sky_roles,
            token_expires_at=result.token_expires_at,
            effective_controller_identity_sha256=(
                result.effective_controller_identity_sha256
            ),
            observed_at=result.observed_at,
        )


@dataclass(frozen=True)
class AdmissionRequest:
    account_id: str
    region: str
    run_id: str
    activation_id: str
    generation: int
    attempt: int
    action_key: str
    request_body_sha256: str
    relay_envelope_sha256: str
    attestation_identity_sha256: str
    direct_decision_identity_sha256: str
    live_h1d_identity_sha256: str
    decision_seal_identity_sha256: str
    decision_nonce_sha256: str
    barrier_transition_identity_sha256: str
    private_nonce: bytes


@dataclass(frozen=True)
class AdmissionResult:
    classification: str
    durable_state: str
    response_identity_sha256: Optional[str]
    request_id: Optional[str]


def _validate_admission_request(request: object) -> AdmissionRequest:
    if not isinstance(request, AdmissionRequest):
        raise AdmissionError("admission request must be a closed typed request")
    if (
        request.account_id != ACCOUNT_ID
        or request.region != REGION
        or request.run_id != RUN_ID
        or type(request.activation_id) is not str
        or not request.activation_id
        or type(request.generation) is not int
        or request.generation <= 0
        or type(request.attempt) is not int
        or request.attempt != 1
        or type(request.action_key) is not str
        or "SKY_POST" not in request.action_key
        or type(request.private_nonce) is not bytes
        or len(request.private_nonce) < 16
    ):
        raise AdmissionError("admission identity is invalid")
    for field in (
        "request_body_sha256",
        "relay_envelope_sha256",
        "attestation_identity_sha256",
        "direct_decision_identity_sha256",
        "live_h1d_identity_sha256",
        "decision_seal_identity_sha256",
        "decision_nonce_sha256",
        "barrier_transition_identity_sha256",
    ):
        try:
            _sha(getattr(request, field), field)
        except SkyIdentityError as exc:
            raise AdmissionError(str(exc)) from exc
    return request


def _relay_body(request: AdmissionRequest) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "record_type": "glm52_closed_sky_launch_envelope_v1",
            "account_id": request.account_id,
            "region": request.region,
            "run_id": request.run_id,
            "activation_id": request.activation_id,
            "generation": request.generation,
            "attempt": request.attempt,
            "action_key": request.action_key,
            "request_body_sha256": request.request_body_sha256,
            "relay_envelope_sha256": request.relay_envelope_sha256,
            "attestation_identity_sha256": (
                request.attestation_identity_sha256
            ),
            "direct_decision_identity_sha256": (
                request.direct_decision_identity_sha256
            ),
            "live_h1d_identity_sha256": (
                request.live_h1d_identity_sha256
            ),
            "decision_seal_identity_sha256": (
                request.decision_seal_identity_sha256
            ),
            "decision_nonce_sha256": request.decision_nonce_sha256,
            "barrier_transition_identity_sha256": (
                request.barrier_transition_identity_sha256
            ),
        }
    )


def build_admission_custody_identity(
    request: AdmissionRequest,
) -> str:
    return canonical_sha256(
        {
            "domain": "GLM52_TASK11_ADMISSION_CUSTODY_V1",
            "run_id": request.run_id,
            "activation_id": request.activation_id,
            "generation": request.generation,
            "action_key": request.action_key,
            "request_body_sha256": request.request_body_sha256,
            "relay_envelope_sha256": request.relay_envelope_sha256,
            "attestation_identity_sha256": (
                request.attestation_identity_sha256
            ),
            "direct_decision_identity_sha256": (
                request.direct_decision_identity_sha256
            ),
            "live_h1d_identity_sha256": (
                request.live_h1d_identity_sha256
            ),
            "decision_seal_identity_sha256": (
                request.decision_seal_identity_sha256
            ),
            "decision_nonce_sha256": request.decision_nonce_sha256,
            "barrier_transition_identity_sha256": (
                request.barrier_transition_identity_sha256
            ),
            "relay_body_sha256": hashlib.sha256(
                _relay_body(request)
            ).hexdigest(),
        }
    )


class SkyAdmissionService:
    """One-owner, one-audit, at-most-one relay call reference monitor."""

    def __init__(
        self,
        *,
        store: object,
        fresh_audit: object,
        current_authority: object,
        relay: object,
        nonce_source: Callable[[], bytes],
    ) -> None:
        self._store = store
        self._fresh_audit = fresh_audit
        self._current_authority = current_authority
        self._relay = relay
        self._nonce_source = nonce_source

    def submit(
        self,
        request: AdmissionRequest,
        *,
        caller_asserted_probe_success: object = None,
    ) -> AdmissionResult:
        del caller_asserted_probe_success
        request = _validate_admission_request(request)
        read = getattr(self._store, "read", None)
        own = getattr(self._store, "own_post_started", None)
        authorize = getattr(self._store, "authorize_post", None)
        classify = getattr(self._store, "classify", None)
        audit_method = getattr(self._fresh_audit, "inspect", None)
        authority_method = getattr(self._current_authority, "inspect", None)
        relay_method = getattr(self._relay, "send", None)
        if not all(
            callable(item)
            for item in (
                read,
                own,
                authorize,
                classify,
                audit_method,
                authority_method,
                relay_method,
            )
        ):
            raise AdmissionError("admission boundary is incomplete")
        current = read(request)
        if current != "CONSUMED":
            return AdmissionResult("RECONCILE_ONLY", str(current), None, None)
        private_nonce = self._nonce_source()
        if type(private_nonce) is not bytes or len(private_nonce) < 16:
            raise AdmissionError("admission owner nonce is invalid")
        private_nonce_sha256 = hashlib.sha256(private_nonce).hexdigest()
        started = own(
            request=request,
            private_nonce_sha256=private_nonce_sha256,
        )
        if started != "POST_STARTED":
            return AdmissionResult("RECONCILE_ONLY", str(started), None, None)
        try:
            audit = audit_method(request)
            authority = authority_method(request)
        except AdmissionError:
            raise
        except Exception as exc:
            raise AdmissionError("fresh Task 8 authority failed") from exc
        if (
            type(audit) is not dict
            or set(audit)
            != {"fresh", "audit_identity_sha256", "closing_revision"}
            or audit["fresh"] is not True
            or type(authority) is not dict
            or set(authority)
            != {
                "current_owner",
                "reserve_held",
                "action_consumed",
                "live_h1d_identity_sha256",
                "reserve_identity_sha256",
                "owner_identity_sha256",
            }
            or authority["current_owner"] is not True
            or authority["reserve_held"] is not True
            or authority["action_consumed"] is not True
            or authority["live_h1d_identity_sha256"]
            != request.live_h1d_identity_sha256
            or authority["owner_identity_sha256"]
            != build_admission_custody_identity(request)
        ):
            raise AdmissionError(
                "fresh Task 8 authority, reserve, or owner is absent"
            )
        for field in (
            "live_h1d_identity_sha256",
            "reserve_identity_sha256",
            "owner_identity_sha256",
        ):
            try:
                _sha(authority[field], field)
            except SkyIdentityError as exc:
                raise AdmissionError(str(exc)) from exc
        authorized = authorize(
            request=request,
            audit=audit,
            authority=authority,
            relay_envelope_sha256=request.relay_envelope_sha256,
        )
        if authorized != "POST_AUTHORIZED":
            return AdmissionResult(
                "RECONCILE_ONLY", str(authorized), None, None
            )
        request_id: Optional[str] = None
        response_identity: Optional[str] = None
        try:
            response = relay_method(_relay_body(request))
            if (
                type(response) is not dict
                or set(response)
                != {"status_code", "request_id", "response_sha256"}
                or type(response["status_code"]) is not int
            ):
                classification = "AMBIGUOUS"
            else:
                request_id = _exact_nonempty(
                    response["request_id"], "Sky request id"
                )
                response_identity = _sha(
                    response["response_sha256"], "Sky response"
                )
                classification = (
                    "ACCEPTED"
                    if response["status_code"] in {200, 201, 202}
                    else "KNOWN_REJECTED"
                )
        except (TimeoutError, ConnectionError):
            classification = "AMBIGUOUS"
        classified = classify(
            request=request,
            classification=classification,
            response_identity_sha256=response_identity,
            request_id=request_id,
        )
        if classified != "POST_CLASSIFIED":
            raise AdmissionError("admission classification did not persist")
        return AdmissionResult(
            classification,
            classified,
            response_identity,
            request_id,
        )


__all__ = [
    "ACCOUNT_ID",
    "EXPECTED_SKYPILOT_VERSION",
    "REGION",
    "RUN_ID",
    "AdmissionError",
    "AdmissionRequest",
    "AdmissionResult",
    "AttestationError",
    "AttestationRequest",
    "AttestationResult",
    "PinnedSkyIdentity",
    "RelayIdentity",
    "SkyAdmissionService",
    "SkyAttestationService",
    "SkyIdentityContract",
    "SkyIdentityError",
    "Task9SkyRelayProbe",
    "build_admission_custody_identity",
    "build_default_identity_contract",
    "build_pinned_sky_identity",
    "validate_pinned_sky_identity",
    "validate_sky_identity_contract",
]
