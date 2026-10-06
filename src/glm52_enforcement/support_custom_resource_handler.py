"""Executable H.1g Secrets Manager custom-resource Lambda entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import ipaddress
import json
import os
import secrets
from typing import Callable, Mapping, Optional
from urllib.parse import urlparse

from .canonical import canonical_json_bytes
from .support_custom_resources import (
    CustomResourceOperation,
    build_success_callback_data,
    cache_callback,
    complete_custom_resource_create,
    complete_identity_preserving_update,
    begin_custom_resource_operation,
    mark_callback_attempted,
    record_material_write,
    record_direct_grant_evidence,
    record_secret_version_observation,
    record_transport_result,
)


@dataclass(frozen=True)
class CustomResourceHandlerServices:
    """Raw production boundaries used by the exact custom-resource route."""

    secretsmanager: object
    callback_transport: object
    material_issuer: Callable[
        [CustomResourceOperation], Mapping[str, str]
    ]
    tls_projector: Optional[
        Callable[[Mapping[str, str]], Mapping[str, str]]
    ] = None
    direct_grant_producer: Optional[
        Callable[[CustomResourceOperation], Mapping[str, object]]
    ] = None


class _CallbackTransport:
    """One no-redirect HTTPS PUT to the already authenticated CFN URL."""

    def put(
        self,
        *,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> Mapping[str, object]:
        parsed = urlparse(url)
        connection = http.client.HTTPSConnection(
            parsed.hostname,
            timeout=10,
        )
        path = parsed.path + ("?" + parsed.query if parsed.query else "")
        try:
            connection.request(
                "PUT",
                path,
                body=body,
                headers=dict(headers),
            )
            response = connection.getresponse()
            response.read()
            return {
                "status_code": response.status,
                "final_url": url,
                "redirect_count": 0,
            }
        finally:
            connection.close()


def _production_secretsmanager() -> object:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda runtime dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    return boto3.client(
        "secretsmanager",
        region_name="us-west-2",
        config=Config(
            connect_timeout=5,
            read_timeout=10,
            retries={"mode": "standard", "total_max_attempts": 1},
        ),
    )


def _production_client(service: str) -> object:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda runtime dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    return boto3.client(
        service,
        region_name="us-west-2",
        config=Config(
            connect_timeout=5,
            read_timeout=10,
            retries={"mode": "standard", "total_max_attempts": 1},
        ),
    )


def _production_direct_grant_producer(
    operation: CustomResourceOperation,
) -> Mapping[str, object]:
    from .task12_orphan_authority import create_or_adopt_direct_grant

    if operation.resource_kind != "TLS_BUNDLE":
        raise ValueError("direct grant is TLS Create only")
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return create_or_adopt_direct_grant(
        kms=_production_client("kms"),
        dynamodb=_production_client("dynamodb"),
        ledger_table_name=os.environ.get("GLM52_LEDGER_TABLE_NAME", ""),
        activation_id=operation.activation_identity,
        custom_resource_request_id=operation.request_id,
        retained_kms_key_arn=os.environ.get(
            "GLM52_RETAINED_KMS_KEY_ARN", ""
        ),
        observed_at=observed_at,
    )


def _canonical_secret(value: Mapping[str, str]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pem_private_key(key: object) -> str:
    from cryptography.hazmat.primitives import serialization

    private_bytes = getattr(key, "private_bytes", None)
    if not callable(private_bytes):
        raise RuntimeError("issued private key cannot be serialized")
    return private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def _production_material_issuer(
    operation: CustomResourceOperation,
) -> Mapping[str, str]:
    """Issue fresh bootstrap or mutually authenticated TLS material."""

    if operation.resource_kind == "SKY_BOOTSTRAP":
        token = secrets.token_urlsafe(48)
        return {
            "RawSkyTokenSecret": _canonical_secret({"token": token}),
            "SkyBootstrapSecret": _canonical_secret(
                {"token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest()}
            ),
        }
    if operation.resource_kind != "TLS_BUNDLE":
        raise ValueError("custom-resource material kind is not exact")
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as exc:  # pragma: no cover - attached Lambda layer
        raise RuntimeError("cryptography Lambda layer is required") from exc

    host = os.environ.get("GLM52_COMBINED_HOST_PRIVATE_IP", "")
    try:
        host_ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise RuntimeError("combined-host private IP is absent") from exc
    now = datetime.now(timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name(
        [
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                "keep-glm52-h1g-" + operation.activation_identity,
            )
        ]
    )
    ca_certificate = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=80))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), True)
        .sign(ca_key, hashes.SHA256())
    )
    ca_pem = ca_certificate.public_bytes(
        serialization.Encoding.PEM
    ).decode("ascii")

    def leaf(
        common_name: str,
        *,
        server: bool,
    ) -> tuple[str, str]:
        key = ec.generate_private_key(ec.SECP256R1())
        builder = (
            x509.CertificateBuilder()
            .subject_name(
                x509.Name(
                    [x509.NameAttribute(NameOID.COMMON_NAME, common_name)]
                )
            )
            .issuer_name(ca_name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(hours=72))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
            .add_extension(
                x509.ExtendedKeyUsage(
                    [
                        (
                            ExtendedKeyUsageOID.SERVER_AUTH
                            if server
                            else ExtendedKeyUsageOID.CLIENT_AUTH
                        )
                    ]
                ),
                True,
            )
        )
        if server:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.IPAddress(host_ip)]),
                False,
            )
        certificate = builder.sign(ca_key, hashes.SHA256())
        return (
            certificate.public_bytes(serialization.Encoding.PEM).decode(
                "ascii"
            ),
            _pem_private_key(key),
        )

    result = {}
    servers = {}
    for logical_id, purpose in (
        ("AttestationClientTlsSecret", "ATTESTATION"),
        ("LaunchAdmissionClientTlsSecret", "LAUNCH_ADMISSION"),
        ("NumericBindingClientTlsSecret", "NUMERIC_BINDING"),
        (
            "RetainedCancellationClientTlsSecret",
            "RETAINED_CANCELLATION",
        ),
    ):
        certificate, private_key = leaf(logical_id, server=False)
        result[logical_id] = _canonical_secret(
            {
                "ca_certificate_pem": ca_pem,
                "client_certificate_pem": certificate,
                "client_private_key_pem": private_key,
            }
        )
        server_certificate, server_private_key = leaf(
            "CombinedHostTlsSecret-" + purpose,
            server=True,
        )
        servers[purpose] = {
            "server_certificate_pem": server_certificate,
            "server_private_key_pem": server_private_key,
        }
    result["CombinedHostTlsSecret"] = _canonical_secret(
        {
            "ca_certificate_pem": ca_pem,
            "servers": servers,
        }
    )
    result["CaIssuanceSecret"] = _canonical_secret(
        {
            "ca_certificate_pem": ca_pem,
            "ca_private_key_pem": _pem_private_key(ca_key),
        }
    )
    return result


def _project_tls_identity(
    materials: Mapping[str, str],
) -> Mapping[str, str]:
    """Hash the exact DER leaves and validity read back from Secrets Manager."""

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        from cryptography.x509.oid import NameOID
    except ImportError as exc:  # pragma: no cover - attached Lambda layer
        raise RuntimeError("cryptography Lambda layer is required") from exc
    expected = (
        "AttestationClientTlsSecret",
        "LaunchAdmissionClientTlsSecret",
        "NumericBindingClientTlsSecret",
        "RetainedCancellationClientTlsSecret",
        "CombinedHostTlsSecret",
        "CaIssuanceSecret",
    )
    if type(materials) is not dict or tuple(materials) != expected:
        raise ValueError("TLS material readback inventory is not exact")

    def parsed(logical_id: str) -> Mapping[str, object]:
        try:
            value = json.loads(materials[logical_id])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("TLS material readback is not JSON") from exc
        if type(value) is not dict:
            raise ValueError("TLS material readback is not one object")
        return value

    clients = {
        "ATTESTATION": ("Attestation", parsed("AttestationClientTlsSecret")),
        "LAUNCH_ADMISSION": (
            "LaunchAdmission",
            parsed("LaunchAdmissionClientTlsSecret"),
        ),
        "NUMERIC_BINDING": (
            "NumericBinding",
            parsed("NumericBindingClientTlsSecret"),
        ),
        "RETAINED_CANCELLATION": (
            "RetainedCancellation",
            parsed("RetainedCancellationClientTlsSecret"),
        ),
    }
    combined = parsed("CombinedHostTlsSecret")
    issuance = parsed("CaIssuanceSecret")
    if (
        set(combined) != {"ca_certificate_pem", "servers"}
        or type(combined["servers"]) is not dict
        or set(combined["servers"]) != set(clients)
        or set(issuance) != {"ca_certificate_pem", "ca_private_key_pem"}
    ):
        raise ValueError("TLS server or CA material shape drifted")
    ca_pem = combined["ca_certificate_pem"]
    if (
        type(ca_pem) is not str
        or issuance["ca_certificate_pem"] != ca_pem
    ):
        raise ValueError("TLS CA certificate material drifted")

    def certificate(pem: object):
        if type(pem) is not str:
            raise ValueError("TLS leaf certificate is absent")
        try:
            return x509.load_pem_x509_certificate(pem.encode("ascii"))
        except (UnicodeError, ValueError) as exc:
            raise ValueError("TLS leaf certificate is invalid") from exc

    ca = certificate(ca_pem)
    projection = {
        "TlsCaDerSha256": hashlib.sha256(
            ca.public_bytes(serialization.Encoding.DER)
        ).hexdigest(),
    }
    validity = set()
    client_pins = set()
    server_pins = set()
    for purpose, (prefix, client_material) in clients.items():
        if set(client_material) != {
            "ca_certificate_pem",
            "client_certificate_pem",
            "client_private_key_pem",
        } or client_material["ca_certificate_pem"] != ca_pem:
            raise ValueError("TLS client material shape or CA drifted")
        server_material = combined["servers"][purpose]
        if type(server_material) is not dict or set(server_material) != {
            "server_certificate_pem",
            "server_private_key_pem",
        }:
            raise ValueError("TLS purpose-specific server material drifted")
        client = certificate(client_material["client_certificate_pem"])
        server = certificate(server_material["server_certificate_pem"])
        client_names = client.subject.get_attributes_for_oid(
            NameOID.COMMON_NAME
        )
        server_names = server.subject.get_attributes_for_oid(
            NameOID.COMMON_NAME
        )
        if (
            len(client_names) != 1
            or client_names[0].value
            != {
                "ATTESTATION": "AttestationClientTlsSecret",
                "LAUNCH_ADMISSION": "LaunchAdmissionClientTlsSecret",
                "NUMERIC_BINDING": "NumericBindingClientTlsSecret",
                "RETAINED_CANCELLATION": (
                    "RetainedCancellationClientTlsSecret"
                ),
            }[purpose]
            or len(server_names) != 1
            or server_names[0].value
            != "CombinedHostTlsSecret-" + purpose
            or client.issuer != ca.subject
            or server.issuer != ca.subject
        ):
            raise ValueError("TLS purpose/client/server certificate swapped")
        client_pin = hashlib.sha256(
            client.public_bytes(serialization.Encoding.DER)
        ).hexdigest()
        server_pin = hashlib.sha256(
            server.public_bytes(serialization.Encoding.DER)
        ).hexdigest()
        projection[prefix + "ClientDerSha256"] = client_pin
        projection[prefix + "ServerDerSha256"] = server_pin
        client_pins.add(client_pin)
        server_pins.add(server_pin)
        for item in (client, server):
            before = getattr(
                item,
                "not_valid_before_utc",
                item.not_valid_before.replace(tzinfo=timezone.utc),
            )
            after = getattr(
                item,
                "not_valid_after_utc",
                item.not_valid_after.replace(tzinfo=timezone.utc),
            )
            validity.add(
                (
                    before.astimezone(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    after.astimezone(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                )
            )
    if len(client_pins) != 4 or len(server_pins) != 4 or len(validity) != 1:
        raise ValueError("TLS leaves or validity are not purpose-specific")
    not_before, not_after = next(iter(validity))
    projection["TlsNotValidBefore"] = not_before
    projection["TlsNotValidAfter"] = not_after
    return projection


def _production_services() -> CustomResourceHandlerServices:
    return CustomResourceHandlerServices(
        secretsmanager=_production_secretsmanager(),
        callback_transport=_CallbackTransport(),
        material_issuer=_production_material_issuer,
        tls_projector=_project_tls_identity,
        direct_grant_producer=_production_direct_grant_producer,
    )


def _success_response(value: object, label: str) -> Mapping[str, object]:
    if (
        type(value) is not dict
        or type(value.get("ResponseMetadata")) is not dict
        or value["ResponseMetadata"].get("HTTPStatusCode") != 200
        or type(value["ResponseMetadata"].get("RequestId")) is not str
        or not value["ResponseMetadata"]["RequestId"]
    ):
        raise ValueError(label + " response is not authenticated")
    return value


def _observe_version(
    *,
    services: CustomResourceHandlerServices,
    operation: CustomResourceOperation,
    secret_logical_id: str,
    secret_id: str,
    version_id: str,
    version_stage: str,
    expected_material: Optional[str],
) -> tuple[str, str]:
    pages = []
    next_token: Optional[str] = None
    for _ in range(100):
        request: dict[str, object] = {
            "SecretId": secret_id,
            "IncludeDeprecated": True,
        }
        if next_token is not None:
            request["NextToken"] = next_token
        response = _success_response(
            services.secretsmanager.list_secret_version_ids(**request),
            "ListSecretVersionIds",
        )
        versions = response.get("Versions")
        if type(versions) is not list:
            raise ValueError("secret version inventory is absent")
        pages.append(
            {
                "versions": versions,
                "request_id": response["ResponseMetadata"]["RequestId"],
            }
        )
        candidate = response.get("NextToken")
        if candidate is None:
            break
        if type(candidate) is not str or not candidate or candidate == next_token:
            raise ValueError("secret version pagination is malformed")
        next_token = candidate
    else:
        raise ValueError("secret version pagination exceeded exact bound")
    matches = [
        item
        for page in pages
        for item in page["versions"]
        if type(item) is dict
        and item.get("VersionId") == version_id
        and item.get("VersionStages") == [version_stage]
    ]
    stage_owners = [
        item
        for page in pages
        for item in page["versions"]
        if type(item) is dict
        and version_stage in item.get("VersionStages", [])
    ]
    if len(matches) != 1 or len(stage_owners) != 1:
        raise ValueError("staged secret version is ambiguous")
    readback = _success_response(
        services.secretsmanager.get_secret_value(
            SecretId=secret_id,
            VersionId=version_id,
            VersionStage=version_stage,
        ),
        "GetSecretValue",
    )
    material = readback.get("SecretString")
    if (
        type(material) is not str
        or not material
        or readback.get("VersionId") != version_id
        or readback.get("VersionStages") != [version_stage]
        or (expected_material is not None and material != expected_material)
    ):
        raise ValueError("staged secret value readback drifted")
    observation_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "operation_token": operation.operation_token,
                "secret_logical_id": secret_logical_id,
                "version_id": version_id,
                "version_stage": version_stage,
                "pages": pages,
                "get_request_id": readback["ResponseMetadata"]["RequestId"],
                "material_sha256": hashlib.sha256(
                    material.encode("utf-8")
                ).hexdigest(),
            }
        )
    ).hexdigest()
    return material, observation_identity


def _callback(
    operation: CustomResourceOperation,
    *,
    services: CustomResourceHandlerServices,
    tls_identity: Optional[Mapping[str, str]] = None,
) -> CustomResourceOperation:
    observations = tuple(operation.version_observations)
    issuance_identity = hashlib.sha256(
        canonical_json_bytes(
            [
                {
                    "secret_logical_id": item.secret_logical_id,
                    "version_id": item.version_id,
                    "version_stage": item.version_stage,
                    "material_sha256": item.material_sha256,
                    "observation_identity_sha256": (
                        item.list_versions_identity_sha256
                    ),
                }
                for item in observations
            ]
        )
    ).hexdigest()
    bundle_identity = hashlib.sha256(
        canonical_json_bytes(
            [
                [item.secret_logical_id, item.material_sha256]
                for item in observations
            ]
        )
    ).hexdigest()
    data = build_success_callback_data(
        operation,
        issuance_identity_sha256=issuance_identity,
        bundle_sha256=bundle_identity,
        tls_identity=tls_identity,
    )
    operation = cache_callback(
        operation,
        status="SUCCESS",
        data=data,
        reason="H1G_CUSTOM_RESOURCE_COMPLETE",
    )
    operation = mark_callback_attempted(operation)
    body = operation.callback_body
    if type(body) is not bytes:
        raise RuntimeError("custom-resource callback was not cached")
    response = services.callback_transport.put(
        url=operation.response_url,
        body=body,
        headers={
            "content-length": str(len(body)),
            "content-type": "",
        },
    )
    if type(response) is not dict or set(response) != {
        "status_code",
        "final_url",
        "redirect_count",
    }:
        raise ValueError("callback transport result is not exact")
    return record_transport_result(
        operation,
        status_code=response["status_code"],
        final_url=response["final_url"],
        redirect_count=response["redirect_count"],
    )


def main(
    event: object,
    context: object,
    *,
    services: Optional[CustomResourceHandlerServices] = None,
) -> Mapping[str, object]:
    """Write/readback material, persist callback bytes, and PUT exactly once."""

    del context
    operation = begin_custom_resource_operation(event)
    runtime = _production_services() if services is None else services
    if type(runtime) is not CustomResourceHandlerServices:
        raise TypeError("custom-resource services are not exact")
    version_ids = {}
    observed_materials = {}
    if operation.request_type == "Create":
        if operation.resource_kind == "TLS_BUNDLE":
            producer = runtime.direct_grant_producer
            if producer is None:
                raise ValueError("TLS Create lacks direct-grant producer")
            direct_evidence = producer(operation)
            if (
                type(direct_evidence) is not dict
                or direct_evidence.get("SK")
                != (
                    f"ACTIVATION#{operation.activation_identity}"
                    f"#CUSTOM_RESOURCE_GRANT#{operation.request_id}"
                )
                or type(direct_evidence.get("canonical_body_sha256"))
                is not str
            ):
                raise ValueError("direct-grant producer result is not exact")
            operation = record_direct_grant_evidence(
                operation,
                coordinate=direct_evidence["SK"],
                evidence_sha256=direct_evidence[
                    "canonical_body_sha256"
                ],
            )
        materials = runtime.material_issuer(operation)
        expected = tuple(
            target.secret_logical_id for target in operation.secret_targets
        )
        if type(materials) is not dict or tuple(materials) != expected:
            raise ValueError("issued material inventory is not exact")
        for target in operation.secret_targets:
            material = materials[target.secret_logical_id]
            if type(material) is not str or not material:
                raise ValueError("issued material is absent")
            version_id = hashlib.sha256(
                (
                    operation.operation_token
                    + ":"
                    + target.secret_logical_id
                ).encode("ascii")
            ).hexdigest()
            request = {
                "SecretId": target.secret_id,
                "ClientRequestToken": version_id,
                "SecretString": material,
                "VersionStages": [target.version_stage],
            }
            response = _success_response(
                runtime.secretsmanager.put_secret_value(**request),
                "PutSecretValue",
            )
            if (
                response.get("ARN") != target.secret_id
                or response.get("VersionId") != version_id
            ):
                raise ValueError("PutSecretValue response identity drifted")
            operation = record_material_write(
                operation,
                secret_logical_id=target.secret_logical_id,
                version_id=version_id,
                version_stage=target.version_stage,
                material_sha256=hashlib.sha256(
                    material.encode("utf-8")
                ).hexdigest(),
                put_request_identity_sha256=hashlib.sha256(
                    canonical_json_bytes(request)
                ).hexdigest(),
                put_response_identity_sha256=hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "arn": response["ARN"],
                            "version_id": response["VersionId"],
                            "request_id": response["ResponseMetadata"][
                                "RequestId"
                            ],
                        }
                    )
                ).hexdigest(),
            )
            observed, observation_identity = _observe_version(
                services=runtime,
                operation=operation,
                secret_logical_id=target.secret_logical_id,
                secret_id=target.secret_id,
                version_id=version_id,
                version_stage=target.version_stage,
                expected_material=material,
            )
            operation = record_secret_version_observation(
                operation,
                secret_logical_id=target.secret_logical_id,
                version_id=version_id,
                version_stage=target.version_stage,
                material_sha256=hashlib.sha256(
                    observed.encode("utf-8")
                ).hexdigest(),
                list_versions_identity_sha256=observation_identity,
            )
            version_ids[target.secret_logical_id] = version_id
            observed_materials[target.secret_logical_id] = observed
        operation = complete_custom_resource_create(operation)
    elif operation.request_type == "Update":
        for target in operation.secret_targets:
            raise ValueError(
                "identity-preserving Update requires an exact existing VersionId"
            )
        operation = complete_identity_preserving_update(operation)
    else:
        raise ValueError("Delete is owned by retained support teardown")
    tls_identity = None
    if operation.resource_kind == "TLS_BUNDLE":
        projector = runtime.tls_projector or _project_tls_identity
        tls_identity = projector(observed_materials)
    operation = _callback(
        operation,
        services=runtime,
        tls_identity=tls_identity,
    )
    body = operation.callback_body
    if type(body) is not bytes:
        raise RuntimeError("custom-resource callback is absent")
    return {
        "schema_version": 1,
        "record_type": "glm52_h1g_custom_resource_result_v1",
        "status": "SUCCESS",
        "physical_resource_id": operation.physical_resource_id,
        "operation_token": operation.operation_token,
        "callback_body_sha256": hashlib.sha256(body).hexdigest(),
        "secret_version_ids": version_ids,
        "direct_grant_evidence_coordinate": (
            operation.direct_grant_evidence_coordinate
        ),
        "direct_grant_evidence_sha256": (
            operation.direct_grant_evidence_sha256
        ),
    }


__all__ = ["CustomResourceHandlerServices", "main"]
