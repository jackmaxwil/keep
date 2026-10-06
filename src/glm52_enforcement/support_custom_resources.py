"""Pure replay-safe callback/state contracts for H.1g custom resources."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Mapping, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from .canonical import canonical_json_bytes
from .support_plane import (
    ACCOUNT_ID,
    REGION,
    RUN_ID,
    SECRET_LOGICAL_IDS,
    SUPPORT_STACK_NAME,
    assert_no_private_material,
    secret_stage,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_LOGICAL_ID = re.compile(r"[A-Za-z][A-Za-z0-9]{0,254}\Z")
_ACTIVATION = re.compile(r"[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?\Z")
_PHYSICAL_ID = re.compile(r"h1g-[a-z0-9-]{1,48}-[0-9a-f]{32}\Z")
_VERSION_ID = re.compile(r"[A-Za-z0-9-]{32,64}\Z")
_CALLBACK_REASON = re.compile(
    r"H1G_[A-Z0-9_]{1,64}(?::[0-9a-f]{64})?\Z"
)
_HOSTS = {
    (
        "cloudformation-custom-resource-response-uswest2."
        "s3.us-west-2.amazonaws.com"
    ),
    (
        "cloudformation-custom-resource-response-uswest2."
        "s3-us-west-2.amazonaws.com"
    ),
}


@dataclass(frozen=True)
class MaterialWrite:
    secret_logical_id: str
    version_id: str
    version_stage: str
    material_sha256: str
    put_request_identity_sha256: str
    put_response_identity_sha256: str


@dataclass(frozen=True)
class SecretVersionObservation:
    secret_logical_id: str
    version_id: str
    version_stage: str
    material_sha256: str
    list_versions_identity_sha256: str


@dataclass(frozen=True)
class SecretTarget:
    secret_logical_id: str
    secret_id: str
    version_stage: str
    identity_kind: str


@dataclass(frozen=True)
class CustomResourceOperation:
    request_type: str
    response_url: str
    stack_id: str
    request_id: str
    resource_type: str
    logical_resource_id: str
    activation_identity: str
    resource_kind: str
    material_identity_sha256: str
    secret_targets: Tuple[SecretTarget, ...]
    physical_resource_id: str
    event_body_sha256: str
    operation_token: str
    phase: str
    material_writes: Tuple[MaterialWrite, ...] = ()
    version_observations: Tuple[SecretVersionObservation, ...] = ()
    material_deletions: Tuple[Tuple[str, str], ...] = ()
    callback_body: Optional[bytes] = None
    callback_attempted: bool = False
    transport_result: Optional[str] = None
    cloudformation_observation: Optional[str] = None
    direct_grant_evidence_coordinate: Optional[str] = None
    direct_grant_evidence_sha256: Optional[str] = None


def _exact_ascii(value: object, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise ValueError(f"{label} must be one nonempty ASCII string")
    return value


def _validate_stack_id(stack_id: object) -> str:
    value = _exact_ascii(stack_id, "StackId")
    prefix = (
        f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:"
        f"stack/{SUPPORT_STACK_NAME}/"
    )
    suffix = value.removeprefix(prefix)
    if not value.startswith(prefix) or _UUID.fullmatch(suffix) is None:
        raise ValueError("StackId is not the exact support stack identity")
    return value


def validate_response_url(
    response_url: object,
    *,
    stack_id: str,
    request_id: str,
    logical_resource_id: str,
) -> str:
    """Authenticate the one documented commercial-partition CFN response URL."""

    value = _exact_ascii(response_url, "ResponseURL")
    _validate_stack_id(stack_id)
    if _UUID.fullmatch(request_id) is None:
        raise ValueError("RequestId is not exact")
    if _LOGICAL_ID.fullmatch(logical_resource_id) is None:
        raise ValueError("LogicalResourceId is not exact")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.fragment
        or unquote(parsed.path.lstrip("/"))
        != f"{stack_id}|{logical_resource_id}|{request_id}"
    ):
        raise ValueError("ResponseURL destination or object key is foreign")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if set(query) != {
        "X-Amz-Algorithm",
        "X-Amz-Credential",
        "X-Amz-Date",
        "X-Amz-Expires",
        "X-Amz-SignedHeaders",
        "X-Amz-Signature",
    } or any(len(values) != 1 for values in query.values()):
        raise ValueError("ResponseURL signed query is not exact")
    credential = query["X-Amz-Credential"][0].split("/")
    if (
        query["X-Amz-Algorithm"] != ["AWS4-HMAC-SHA256"]
        or len(credential) != 5
        or re.fullmatch(r"[A-Z0-9]{16,128}", credential[0]) is None
        or re.fullmatch(r"20[0-9]{6}", credential[1]) is None
        or credential[2:] != [REGION, "s3", "aws4_request"]
        or re.fullmatch(r"20[0-9]{6}T[0-9]{6}Z", query["X-Amz-Date"][0])
        is None
        or query["X-Amz-Expires"][0] not in {"300", "600", "900", "1800", "3600"}
        or query["X-Amz-SignedHeaders"] != ["host"]
        or re.fullmatch(r"[0-9a-f]{64}", query["X-Amz-Signature"][0]) is None
    ):
        raise ValueError("ResponseURL signature scope is malformed or foreign")
    return value


def _event_fields(request_type: str) -> set[str]:
    common = {
        "RequestType",
        "ResponseURL",
        "StackId",
        "RequestId",
        "ResourceType",
        "LogicalResourceId",
        "ResourceProperties",
    }
    if request_type == "Create":
        return common
    if request_type == "Delete":
        return common | {"PhysicalResourceId"}
    if request_type == "Update":
        return common | {"PhysicalResourceId", "OldResourceProperties"}
    raise ValueError("RequestType must be Create, Update, or Delete")


def _secret_targets(
    value: object,
    *,
    activation_identity: str,
    resource_kind: str,
) -> Tuple[SecretTarget, ...]:
    expected = (
        SECRET_LOGICAL_IDS[2:]
        if resource_kind == "TLS_BUNDLE"
        else SECRET_LOGICAL_IDS[:2]
    )
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("custom-resource secret target families are not exact")
    result = []
    for logical_id in expected:
        target = value[logical_id]
        if type(target) is not dict or set(target) != {
            "SecretId",
            "VersionStage",
        }:
            raise ValueError("custom-resource secret target shape is not exact")
        version_stage = target["VersionStage"]
        expected_stage = secret_stage(activation_identity, logical_id)
        if version_stage != expected_stage:
            raise ValueError("custom-resource secret target stage is foreign")
        secret_id = target["SecretId"]
        if secret_id == {"Ref": logical_id}:
            normalized_secret_id = logical_id
            identity_kind = "CLOUDFORMATION_REF"
        elif type(secret_id) is str:
            purpose = expected_stage.removeprefix(
                f"h1g-{activation_identity}-"
            ).removesuffix("-v1")
            arn_prefix = (
                f"arn:aws:secretsmanager:{REGION}:{ACCOUNT_ID}:secret:"
                f"/keep/glm52/{RUN_ID}/{activation_identity}/{purpose}-"
            )
            if (
                re.fullmatch(
                    re.escape(arn_prefix) + r"[A-Za-z0-9]{6}",
                    secret_id,
                )
                is None
            ):
                raise ValueError(
                    "custom-resource secret target identity is foreign"
                )
            normalized_secret_id = secret_id
            identity_kind = "SECRET_ARN"
        else:
            raise ValueError("custom-resource secret target ref is not exact")
        result.append(
            SecretTarget(
                secret_logical_id=logical_id,
                secret_id=normalized_secret_id,
                version_stage=expected_stage,
                identity_kind=identity_kind,
            )
        )
    return tuple(result)


def _properties(value: object) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != {
        "ServiceToken",
        "ActivationIdentity",
        "ResourceKind",
        "MaterialIdentitySha256",
        "SecretTargets",
    }:
        raise ValueError("custom-resource properties are not exact")
    service_token = _exact_ascii(value["ServiceToken"], "ServiceToken")
    if (
        re.fullmatch(
            rf"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
            r"function:[A-Za-z0-9-_]+:[1-9][0-9]*",
            service_token,
        )
        is None
    ):
        raise ValueError("ServiceToken must name one exact published version")
    activation = _exact_ascii(value["ActivationIdentity"], "activation identity")
    if _ACTIVATION.fullmatch(activation) is None:
        raise ValueError("activation identity has unsafe grammar")
    kind = value["ResourceKind"]
    if kind not in {"TLS_BUNDLE", "SKY_BOOTSTRAP"}:
        raise ValueError("resource kind is not exact")
    material = value["MaterialIdentitySha256"]
    if type(material) is not str or _SHA256.fullmatch(material) is None:
        raise ValueError("material identity is not exact")
    targets = _secret_targets(
        value["SecretTargets"],
        activation_identity=activation,
        resource_kind=kind,
    )
    return {
        "ServiceToken": service_token,
        "ActivationIdentity": activation,
        "ResourceKind": kind,
        "MaterialIdentitySha256": material,
        "SecretTargets": targets,
    }


def _stable_physical_id(
    *, stack_id: str, logical_id: str, resource_kind: str, activation: str
) -> str:
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "stack_id": stack_id,
                "logical_resource_id": logical_id,
                "resource_kind": resource_kind,
                "activation_identity": activation,
            }
        )
    ).hexdigest()
    return f"h1g-{resource_kind.lower().replace('_', '-')}-{identity[:32]}"


def begin_custom_resource_operation(event: object) -> CustomResourceOperation:
    """Validate one event and derive stable replacement and replay identities."""

    if type(event) is not dict or type(event.get("RequestType")) is not str:
        raise ValueError("custom-resource event must be one exact object")
    request_type = event["RequestType"]
    if set(event) != _event_fields(request_type):
        raise ValueError("custom-resource event fields are not exact")
    stack_id = _validate_stack_id(event["StackId"])
    request_id = _exact_ascii(event["RequestId"], "RequestId")
    if _UUID.fullmatch(request_id) is None:
        raise ValueError("RequestId is not exact")
    logical_id = _exact_ascii(event["LogicalResourceId"], "LogicalResourceId")
    if _LOGICAL_ID.fullmatch(logical_id) is None:
        raise ValueError("LogicalResourceId is not exact")
    resource_type = _exact_ascii(event["ResourceType"], "ResourceType")
    if resource_type not in {
        "Custom::H1gTlsBundle",
        "Custom::H1gSkyBootstrap",
    }:
        raise ValueError("custom-resource type is not exact")
    properties = _properties(event["ResourceProperties"])
    expected_kind = (
        "TLS_BUNDLE"
        if resource_type == "Custom::H1gTlsBundle"
        else "SKY_BOOTSTRAP"
    )
    if properties["ResourceKind"] != expected_kind:
        raise ValueError("resource type/kind identity mismatch")
    response_url = validate_response_url(
        event["ResponseURL"],
        stack_id=stack_id,
        request_id=request_id,
        logical_resource_id=logical_id,
    )
    physical_id = _stable_physical_id(
        stack_id=stack_id,
        logical_id=logical_id,
        resource_kind=properties["ResourceKind"],
        activation=properties["ActivationIdentity"],
    )
    if request_type == "Update":
        old = _properties(event["OldResourceProperties"])
        if old != properties:
            raise ValueError("material-changing Update is forbidden before writes")
    if request_type in {"Update", "Delete"}:
        supplied = _exact_ascii(
            event["PhysicalResourceId"], "PhysicalResourceId"
        )
        if _PHYSICAL_ID.fullmatch(supplied) is None or supplied != physical_id:
            raise ValueError("replacement or foreign PhysicalResourceId is forbidden")
    event_sha = hashlib.sha256(canonical_json_bytes(event)).hexdigest()
    operation_token = hashlib.sha256(
        canonical_json_bytes(
            {
                "request_id": request_id,
                "request_type": request_type,
                "event_body_sha256": event_sha,
                "physical_resource_id": physical_id,
            }
        )
    ).hexdigest()
    return CustomResourceOperation(
        request_type=request_type,
        response_url=response_url,
        stack_id=stack_id,
        request_id=request_id,
        resource_type=resource_type,
        logical_resource_id=logical_id,
        activation_identity=properties["ActivationIdentity"],
        resource_kind=properties["ResourceKind"],
        material_identity_sha256=properties["MaterialIdentitySha256"],
        secret_targets=properties["SecretTargets"],
        physical_resource_id=physical_id,
        event_body_sha256=event_sha,
        operation_token=operation_token,
        phase={
            "Create": "CREATE_RECONCILING",
            "Update": "UPDATE_IDENTITY_PRESERVED",
            "Delete": "DELETE_RECONCILING",
        }[request_type],
    )


def reconcile_delivery(
    operation: CustomResourceOperation, event: object
) -> CustomResourceOperation:
    """Accept only byte-identical same-RequestId delivery and reuse all state."""

    candidate = begin_custom_resource_operation(event)
    if (
        candidate.request_id != operation.request_id
        or candidate.event_body_sha256 != operation.event_body_sha256
        or candidate.operation_token != operation.operation_token
    ):
        raise ValueError("same operation replay carries different canonical event bytes")
    return operation


def record_material_write(
    operation: CustomResourceOperation,
    *,
    secret_logical_id: str,
    version_id: str,
    version_stage: str,
    material_sha256: str,
    put_request_identity_sha256: str,
    put_response_identity_sha256: str,
) -> CustomResourceOperation:
    """Append one hash-only write identity or reconcile an identical partial write."""

    expected = (
        set(SECRET_LOGICAL_IDS[2:])
        if operation.resource_kind == "TLS_BUNDLE"
        else set(SECRET_LOGICAL_IDS[:2])
    )
    if (
        operation.request_type != "Create"
        or operation.phase != "CREATE_RECONCILING"
        or operation.callback_body is not None
        or operation.callback_attempted
        or operation.transport_result is not None
        or operation.cloudformation_observation is not None
    ):
        raise ValueError("Update/Delete may not create or change material")
    if (
        secret_logical_id not in expected
        or type(version_id) is not str
        or _VERSION_ID.fullmatch(version_id) is None
        or version_stage
        != secret_stage(operation.activation_identity, secret_logical_id)
        or _SHA256.fullmatch(material_sha256) is None
        or _SHA256.fullmatch(put_request_identity_sha256) is None
        or _SHA256.fullmatch(put_response_identity_sha256) is None
    ):
        raise ValueError("material-write identity is not exact and nonsecret")
    write = MaterialWrite(
        secret_logical_id=secret_logical_id,
        version_id=version_id,
        version_stage=version_stage,
        material_sha256=material_sha256,
        put_request_identity_sha256=put_request_identity_sha256,
        put_response_identity_sha256=put_response_identity_sha256,
    )
    prior = {
        item.secret_logical_id: item for item in operation.material_writes
    }
    if secret_logical_id in prior:
        if prior[secret_logical_id] != write:
            raise ValueError("partial-write reconciliation found different material")
        return operation
    return replace(operation, material_writes=operation.material_writes + (write,))


def record_secret_version_observation(
    operation: CustomResourceOperation,
    *,
    secret_logical_id: str,
    version_id: str,
    version_stage: str,
    material_sha256: str,
    list_versions_identity_sha256: str,
) -> CustomResourceOperation:
    """Record exact ListSecretVersionIds/GetSecretValue proof for one version."""

    expected = (
        set(SECRET_LOGICAL_IDS[2:])
        if operation.resource_kind == "TLS_BUNDLE"
        else set(SECRET_LOGICAL_IDS[:2])
    )
    if (
        operation.request_type not in {"Create", "Update"}
        or operation.phase
        not in {"CREATE_RECONCILING", "UPDATE_IDENTITY_PRESERVED"}
        or operation.callback_body is not None
        or secret_logical_id not in expected
        or _VERSION_ID.fullmatch(version_id) is None
        or version_stage
        != secret_stage(operation.activation_identity, secret_logical_id)
        or _SHA256.fullmatch(material_sha256) is None
        or _SHA256.fullmatch(list_versions_identity_sha256) is None
    ):
        raise ValueError("secret-version observation is not exact")
    if operation.request_type == "Create":
        writes = {
            item.secret_logical_id: item for item in operation.material_writes
        }
        write = writes.get(secret_logical_id)
        if (
            write is None
            or write.version_id != version_id
            or write.version_stage != version_stage
            or write.material_sha256 != material_sha256
        ):
            raise ValueError("version observation is not bound to its material write")
    observation = SecretVersionObservation(
        secret_logical_id=secret_logical_id,
        version_id=version_id,
        version_stage=version_stage,
        material_sha256=material_sha256,
        list_versions_identity_sha256=list_versions_identity_sha256,
    )
    prior = {
        item.secret_logical_id: item
        for item in operation.version_observations
    }
    if secret_logical_id in prior:
        if prior[secret_logical_id] != observation:
            raise ValueError("version observation replay changed identity")
        return operation
    return replace(
        operation,
        version_observations=operation.version_observations + (observation,),
    )


def record_direct_grant_evidence(
    operation: CustomResourceOperation,
    *,
    coordinate: str,
    evidence_sha256: str,
) -> CustomResourceOperation:
    """Bind TLS Create to the retained full-evidence ledger record."""

    expected_coordinate = (
        f"ACTIVATION#{operation.activation_identity}"
        f"#CUSTOM_RESOURCE_GRANT#{operation.request_id}"
    )
    if (
        operation.request_type != "Create"
        or operation.resource_kind != "TLS_BUNDLE"
        or operation.phase != "CREATE_RECONCILING"
        or operation.callback_body is not None
        or coordinate != expected_coordinate
        or _SHA256.fullmatch(evidence_sha256) is None
    ):
        raise ValueError("direct-grant evidence binding is not exact")
    if operation.direct_grant_evidence_coordinate is not None:
        if (
            operation.direct_grant_evidence_coordinate != coordinate
            or operation.direct_grant_evidence_sha256 != evidence_sha256
        ):
            raise ValueError("direct-grant evidence replay changed identity")
        return operation
    return replace(
        operation,
        direct_grant_evidence_coordinate=coordinate,
        direct_grant_evidence_sha256=evidence_sha256,
    )


def complete_custom_resource_create(
    operation: CustomResourceOperation,
) -> CustomResourceOperation:
    """Close Create only after exact writes and one-version readbacks exist."""

    expected = (
        set(SECRET_LOGICAL_IDS[2:])
        if operation.resource_kind == "TLS_BUNDLE"
        else set(SECRET_LOGICAL_IDS[:2])
    )
    writes = {item.secret_logical_id: item for item in operation.material_writes}
    observations = {
        item.secret_logical_id: item
        for item in operation.version_observations
    }
    if (
        operation.request_type != "Create"
        or operation.phase != "CREATE_RECONCILING"
        or operation.callback_body is not None
        or set(writes) != expected
        or set(observations) != expected
        or any(
            (
                writes[logical_id].version_id,
                writes[logical_id].version_stage,
                writes[logical_id].material_sha256,
            )
            != (
                observations[logical_id].version_id,
                observations[logical_id].version_stage,
                observations[logical_id].material_sha256,
            )
            for logical_id in expected
        )
        or (
            operation.resource_kind == "TLS_BUNDLE"
            and (
                operation.direct_grant_evidence_coordinate is None
                or operation.direct_grant_evidence_sha256 is None
            )
        )
        or (
            operation.resource_kind != "TLS_BUNDLE"
            and (
                operation.direct_grant_evidence_coordinate is not None
                or operation.direct_grant_evidence_sha256 is not None
            )
        )
    ):
        raise ValueError("Create lacks its exact write/version inventory")
    return replace(operation, phase="CREATE_RECONCILED")


def complete_identity_preserving_update(
    operation: CustomResourceOperation,
) -> CustomResourceOperation:
    """Close Update only after every existing staged version is re-observed."""

    expected = (
        set(SECRET_LOGICAL_IDS[2:])
        if operation.resource_kind == "TLS_BUNDLE"
        else set(SECRET_LOGICAL_IDS[:2])
    )
    observations = {
        item.secret_logical_id: item
        for item in operation.version_observations
    }
    if (
        operation.request_type != "Update"
        or operation.phase != "UPDATE_IDENTITY_PRESERVED"
        or operation.material_writes
        or operation.callback_body is not None
        or set(observations) != expected
    ):
        raise ValueError("Update lacks exact unchanged version inventory")
    return replace(operation, phase="UPDATE_RECONCILED")


def enter_failed_create_rollback(
    operation: CustomResourceOperation,
) -> CustomResourceOperation:
    if (
        operation.request_type != "Create"
        or not operation.material_writes
        or operation.callback_body is not None
        or operation.phase != "CREATE_RECONCILING"
    ):
        raise ValueError("failed-Create rollback is not reachable from this state")
    return replace(operation, phase="FAILED_CREATE_ROLLBACK")


def record_material_deletion(
    operation: CustomResourceOperation,
    *,
    secret_logical_id: str,
    tombstone_sha256: str,
) -> CustomResourceOperation:
    expected = (
        set(SECRET_LOGICAL_IDS[2:])
        if operation.resource_kind == "TLS_BUNDLE"
        else set(SECRET_LOGICAL_IDS[:2])
    )
    if (
        operation.phase not in {"FAILED_CREATE_ROLLBACK", "DELETE_RECONCILING"}
        or operation.callback_body is not None
        or secret_logical_id not in expected
        or _SHA256.fullmatch(tombstone_sha256) is None
    ):
        raise ValueError("private-material deletion tombstone is not exact")
    deletion = (secret_logical_id, tombstone_sha256)
    prior = dict(operation.material_deletions)
    if secret_logical_id in prior:
        if prior[secret_logical_id] != tombstone_sha256:
            raise ValueError("material deletion replay carries another tombstone")
        return operation
    return replace(
        operation,
        material_deletions=operation.material_deletions + (deletion,),
    )


def complete_failed_create_rollback(
    operation: CustomResourceOperation,
) -> CustomResourceOperation:
    expected = {item.secret_logical_id for item in operation.material_writes}
    observed = {item[0] for item in operation.material_deletions}
    if operation.phase != "FAILED_CREATE_ROLLBACK" or observed != expected:
        raise ValueError("failed Create retains partial private material")
    return replace(operation, phase="FAILED_CREATE_ROLLED_BACK")


def complete_custom_resource_delete(
    operation: CustomResourceOperation,
    *,
    deleted_secret_logical_ids: Tuple[str, ...],
    tombstone_sha256: str,
) -> CustomResourceOperation:
    expected = (
        set(SECRET_LOGICAL_IDS[2:])
        if operation.resource_kind == "TLS_BUNDLE"
        else set(SECRET_LOGICAL_IDS[:2])
    )
    observed = dict(operation.material_deletions)
    if (
        operation.request_type != "Delete"
        or operation.phase != "DELETE_RECONCILING"
        or type(deleted_secret_logical_ids) is not tuple
        or deleted_secret_logical_ids
        != tuple(
            logical_id
            for logical_id in SECRET_LOGICAL_IDS
            if logical_id in expected
        )
        or _SHA256.fullmatch(tombstone_sha256) is None
        or set(observed) != expected
        or any(observed[item] != tombstone_sha256 for item in expected)
    ):
        raise ValueError("Delete reconciliation is incomplete or foreign")
    return replace(operation, phase="DELETE_RECONCILED")


def _version_result(
    observation: SecretVersionObservation,
) -> Mapping[str, str]:
    identity_body = {
        "secret_logical_id": observation.secret_logical_id,
        "version_id": observation.version_id,
        "version_stage": observation.version_stage,
        "material_sha256": observation.material_sha256,
        "list_versions_identity_sha256": (
            observation.list_versions_identity_sha256
        ),
    }
    return {
        "version_id": observation.version_id,
        "version_stage": observation.version_stage,
        "version_identity_sha256": hashlib.sha256(
            canonical_json_bytes(identity_body)
        ).hexdigest(),
    }


def build_success_callback_data(
    operation: CustomResourceOperation,
    *,
    issuance_identity_sha256: str,
    bundle_sha256: str,
    tls_identity: Optional[Mapping[str, str]] = None,
) -> Mapping[str, object]:
    """Project only observed nonsecret version identities into SUCCESS Data."""

    if (
        operation.request_type == "Create"
        and operation.phase == "CREATE_RECONCILED"
    ):
        status = "ISSUED"
    elif (
        operation.request_type == "Update"
        and operation.phase == "UPDATE_RECONCILED"
    ):
        status = "VERIFIED"
    else:
        raise ValueError("version callback data requires reconciled Create/Update")
    if (
        _SHA256.fullmatch(issuance_identity_sha256) is None
        or _SHA256.fullmatch(bundle_sha256) is None
    ):
        raise ValueError("callback aggregate identities are not exact")
    result = {
        "status": status,
        "issuance_id": f"iss-{issuance_identity_sha256}",
        "bundle_sha256": bundle_sha256,
    }
    if tls_identity is not None:
        expected_tls_fields = {
            "TlsCaDerSha256",
            "AttestationClientDerSha256",
            "AttestationServerDerSha256",
            "LaunchAdmissionClientDerSha256",
            "LaunchAdmissionServerDerSha256",
            "NumericBindingClientDerSha256",
            "NumericBindingServerDerSha256",
            "RetainedCancellationClientDerSha256",
            "RetainedCancellationServerDerSha256",
            "TlsNotValidBefore",
            "TlsNotValidAfter",
        }
        if (
            operation.resource_kind != "TLS_BUNDLE"
            or type(tls_identity) is not dict
            or set(tls_identity) != expected_tls_fields
            or any(
                _SHA256.fullmatch(tls_identity[field]) is None
                for field in expected_tls_fields
                if field.endswith("Sha256")
            )
            or any(
                re.fullmatch(
                    r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T"
                    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                    tls_identity[field],
                )
                is None
                for field in ("TlsNotValidBefore", "TlsNotValidAfter")
            )
        ):
            raise ValueError("callback TLS identities are not exact")
        result.update(tls_identity)
    if (
        operation.resource_kind == "TLS_BUNDLE"
        and operation.request_type == "Create"
    ):
        if (
            operation.direct_grant_evidence_coordinate is None
            or operation.direct_grant_evidence_sha256 is None
        ):
            raise ValueError("callback direct-grant evidence binding is absent")
        result["DirectGrantEvidenceCoordinate"] = (
            operation.direct_grant_evidence_coordinate
        )
        result["DirectGrantEvidenceSha256"] = (
            operation.direct_grant_evidence_sha256
        )
    for item in operation.version_observations:
        prefix = item.secret_logical_id
        version = _version_result(item)
        result[prefix + "VersionId"] = version["version_id"]
        result[prefix + "VersionStage"] = version["version_stage"]
        result[prefix + "VersionIdentitySha256"] = version[
            "version_identity_sha256"
        ]
    return result


def build_delete_callback_data(
    operation: CustomResourceOperation,
) -> Mapping[str, object]:
    """Project exact private-material destruction tombstones into SUCCESS Data."""

    expected = tuple(
        logical_id
        for logical_id in SECRET_LOGICAL_IDS
        if logical_id
        in (
            set(SECRET_LOGICAL_IDS[2:])
            if operation.resource_kind == "TLS_BUNDLE"
            else set(SECRET_LOGICAL_IDS[:2])
        )
    )
    tombstones = dict(operation.material_deletions)
    values = set(tombstones.values())
    if (
        operation.request_type != "Delete"
        or operation.phase != "DELETE_RECONCILED"
        or tuple(tombstones) != expected
        or len(values) != 1
    ):
        raise ValueError("Delete callback lacks exact destruction tombstones")
    return {
        "status": "DELETED",
        "deleted_secret_logical_ids": list(expected),
        "tombstone_sha256": next(iter(values)),
    }


def _validate_callback_data(
    operation: CustomResourceOperation, data: object, status: str
) -> Mapping[str, object]:
    if type(data) is not dict:
        raise ValueError("callback Data must be one exact object")
    if status == "FAILED":
        if data != {"status": "FAILED"}:
            raise ValueError("FAILED callback Data is not compact and exact")
        return data
    if operation.request_type == "Delete":
        if data != build_delete_callback_data(operation):
            raise ValueError("Delete callback Data is not the tombstone projection")
        return data
    expected_fields = {
        "status",
        "issuance_id",
        "bundle_sha256",
    }
    for item in operation.version_observations:
        expected_fields.update(
            {
                item.secret_logical_id + "VersionId",
                item.secret_logical_id + "VersionStage",
                item.secret_logical_id + "VersionIdentitySha256",
            }
        )
    tls_fields = {
        "TlsCaDerSha256",
        "AttestationClientDerSha256",
        "AttestationServerDerSha256",
        "LaunchAdmissionClientDerSha256",
        "LaunchAdmissionServerDerSha256",
        "NumericBindingClientDerSha256",
        "NumericBindingServerDerSha256",
        "RetainedCancellationClientDerSha256",
        "RetainedCancellationServerDerSha256",
        "TlsNotValidBefore",
        "TlsNotValidAfter",
    }
    present_tls_fields = set(data) & tls_fields
    if present_tls_fields:
        if operation.resource_kind != "TLS_BUNDLE" or present_tls_fields != tls_fields:
            raise ValueError("SUCCESS callback TLS Data is partial or foreign")
        expected_fields.update(tls_fields)
    if (
        operation.resource_kind == "TLS_BUNDLE"
        and operation.request_type == "Create"
    ):
        expected_fields.update(
            {
                "DirectGrantEvidenceCoordinate",
                "DirectGrantEvidenceSha256",
            }
        )
    if set(data) != expected_fields:
        raise ValueError("SUCCESS callback Data is not compact and exact")
    if (
        data["status"]
        != ("ISSUED" if operation.request_type == "Create" else "VERIFIED")
        or type(data["issuance_id"]) is not str
        or re.fullmatch(r"iss-[0-9a-f]{64}", data["issuance_id"]) is None
        or type(data["bundle_sha256"]) is not str
        or _SHA256.fullmatch(data["bundle_sha256"]) is None
        or (
            operation.resource_kind == "TLS_BUNDLE"
            and operation.request_type == "Create"
            and (
                data.get("DirectGrantEvidenceCoordinate")
                != operation.direct_grant_evidence_coordinate
                or data.get("DirectGrantEvidenceSha256")
                != operation.direct_grant_evidence_sha256
            )
        )
        or any(
            (
                data[item.secret_logical_id + "VersionId"],
                data[item.secret_logical_id + "VersionStage"],
                data[item.secret_logical_id + "VersionIdentitySha256"],
            )
            != (
                item.version_id,
                item.version_stage,
                _version_result(item)["version_identity_sha256"],
            )
            for item in operation.version_observations
        )
    ):
        raise ValueError("callback version identities are malformed or unobserved")
    return data


def serialize_callback(
    operation: CustomResourceOperation,
    *,
    status: str,
    data: object,
    reason: str = "H1G_CUSTOM_RESOURCE_COMPLETE",
) -> bytes:
    """Create one canonical, compact, private-material-free CFN response body."""

    if status not in {"SUCCESS", "FAILED"}:
        raise ValueError("callback status is not exact")
    result = _validate_callback_data(operation, data, status)
    reason_text = assert_no_private_material(reason, "callback reason")
    if _CALLBACK_REASON.fullmatch(reason_text) is None:
        raise ValueError("callback reason must be one closed nonsecret code")
    body = {
        "Status": status,
        "Reason": reason_text,
        "PhysicalResourceId": operation.physical_resource_id,
        "StackId": operation.stack_id,
        "RequestId": operation.request_id,
        "LogicalResourceId": operation.logical_resource_id,
        "NoEcho": False,
        "Data": result,
    }
    encoded = canonical_json_bytes(body)
    forbidden = (
        b"PRIVATE KEY",
        b"BEGIN RSA",
        b"BEGIN EC",
        b"secretstring",
        b"raw_token",
        b"certificate",
    )
    lowered = encoded.lower()
    if any(item.lower() in lowered for item in forbidden) or len(encoded) > 4096:
        raise ValueError("callback contains private material or exceeds 4,096 bytes")
    return encoded


def cache_callback(
    operation: CustomResourceOperation,
    *,
    status: str,
    data: object,
    reason: str,
) -> CustomResourceOperation:
    if status == "SUCCESS":
        if (
            operation.request_type == "Create"
            and operation.phase != "CREATE_RECONCILED"
        ) or (
            operation.request_type == "Update"
            and operation.phase != "UPDATE_RECONCILED"
        ) or (
            operation.request_type == "Delete"
            and operation.phase != "DELETE_RECONCILED"
        ):
            raise ValueError("SUCCESS callback precedes required reconciliation")
    if (
        status == "FAILED"
        and operation.material_writes
        and operation.phase != "FAILED_CREATE_ROLLED_BACK"
    ):
        raise ValueError("FAILED callback cannot precede private-material rollback")
    body = serialize_callback(operation, status=status, data=data, reason=reason)
    if operation.callback_body is not None:
        if operation.callback_body != body:
            raise ValueError("immutable cached callback body cannot change")
        return operation
    return replace(operation, callback_body=body)


def mark_callback_attempted(
    operation: CustomResourceOperation,
) -> CustomResourceOperation:
    if operation.callback_body is None:
        raise ValueError("callback bytes must be persisted before first attempt")
    return replace(operation, callback_attempted=True)


def record_transport_result(
    operation: CustomResourceOperation,
    *,
    status_code: int,
    final_url: str,
    redirect_count: int,
) -> CustomResourceOperation:
    if (
        not operation.callback_attempted
        or type(status_code) is not int
        or type(redirect_count) is not int
        or redirect_count != 0
        or final_url != operation.response_url
    ):
        raise ValueError("callback transport redirected or bypassed persisted state")
    validate_response_url(
        final_url,
        stack_id=operation.stack_id,
        request_id=operation.request_id,
        logical_resource_id=operation.logical_resource_id,
    )
    if status_code != 200:
        raise ValueError("callback transport did not return exact HTTP 200")
    return replace(operation, transport_result="HTTP_200")


def observe_cloudformation(
    operation: CustomResourceOperation, observation: str
) -> CustomResourceOperation:
    if observation not in {
        "SUCCESS_OBSERVED",
        "FAILED_OBSERVED",
        "DELETE_OBSERVED",
    }:
        raise ValueError("CloudFormation observation state is not exact")
    if (
        operation.callback_body is None
        or not operation.callback_attempted
        or operation.transport_result != "HTTP_200"
    ):
        raise ValueError("CloudFormation cannot observe an untransported callback")
    expected = (
        "FAILED_OBSERVED"
        if b'"Status":"FAILED"' in operation.callback_body
        else (
            "DELETE_OBSERVED"
            if operation.request_type == "Delete"
            else "SUCCESS_OBSERVED"
        )
    )
    if observation != expected:
        raise ValueError("CloudFormation observation conflicts with callback status")
    return replace(operation, cloudformation_observation=observation)


def validate_reverse_deletion_dependencies(
    template: Mapping[str, object],
) -> bool:
    """Prove reverse order retains handler/access/secrets through Delete callback."""

    if type(template) is not dict or type(template.get("Resources")) is not dict:
        raise ValueError("dependency validation requires an exact support template")
    resources = template["Resources"]
    required = {
        "SkyBootstrapCustomResource": {
            "TlsHandlerRole",
            "TlsHandlerVersion",
            "TlsHandlerPermission",
            "RawSkyTokenSecret",
            "SkyBootstrapSecret",
        },
        "TlsBundleCustomResource": {
            "TlsHandlerRole",
            "TlsHandlerVersion",
            "TlsHandlerPermission",
            *SECRET_LOGICAL_IDS[2:],
        },
    }
    for logical_id, dependencies in required.items():
        resource = resources.get(logical_id)
        if (
            type(resource) is not dict
            or type(resource.get("DependsOn")) is not list
            or set(resource["DependsOn"]) != dependencies
            or resource.get("Properties", {}).get("ServiceToken")
            != {"Ref": "TlsHandlerVersion"}
        ):
            raise ValueError("custom-resource reverse dependency graph is unsafe")
    host_dependencies = resources.get("CombinedHost", {}).get("DependsOn")
    if host_dependencies != [
        "SkyBootstrapCustomResource",
        "TlsBundleCustomResource",
    ]:
        raise ValueError("host may start before private material is ready")
    function = resources.get("TlsHandlerFunction", {}).get("Properties", {})
    if "VpcConfig" in function:
        raise ValueError("callback handler must remain non-VPC")
    permission = resources.get("TlsHandlerPermission", {}).get("Properties", {})
    if (
        permission.get("FunctionName") != {"Ref": "TlsHandlerVersion"}
        or permission.get("Principal") != "cloudformation.amazonaws.com"
    ):
        raise ValueError("handler permission is not pinned to the exact version")
    return True
