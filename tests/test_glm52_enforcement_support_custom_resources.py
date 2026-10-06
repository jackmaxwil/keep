from __future__ import annotations

import copy
import hashlib
import json
from urllib.parse import quote, urlencode

import pytest


STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "stack/keep-glm52-h1g-support/"
    "11111111-2222-4333-8444-555555555555"
)
LOGICAL_ID = "TlsBundleCustomResource"
REQUEST_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_SECRET_PURPOSES = {
    "RawSkyTokenSecret": "raw-sky-token",
    "SkyBootstrapSecret": "sky-bootstrap-hash",
    "AttestationClientTlsSecret": "attestation-client-tls",
    "LaunchAdmissionClientTlsSecret": "launch-admission-client-tls",
    "NumericBindingClientTlsSecret": "numeric-binding-client-tls",
    "RetainedCancellationClientTlsSecret": "retained-cancellation-client-tls",
    "CombinedHostTlsSecret": "combined-host-server-tls",
    "CaIssuanceSecret": "ca-issuance",
}
_SECRET_FAMILIES = {
    "SKY_BOOTSTRAP": (
        "RawSkyTokenSecret",
        "SkyBootstrapSecret",
    ),
    "TLS_BUNDLE": (
        "AttestationClientTlsSecret",
        "LaunchAdmissionClientTlsSecret",
        "NumericBindingClientTlsSecret",
        "RetainedCancellationClientTlsSecret",
        "CombinedHostTlsSecret",
        "CaIssuanceSecret",
    ),
}


def _response_url(
    *,
    stack_id: str = STACK_ID,
    logical_id: str = LOGICAL_ID,
    request_id: str = REQUEST_ID,
    host: str = (
        "cloudformation-custom-resource-response-uswest2."
        "s3.us-west-2.amazonaws.com"
    ),
) -> str:
    key = quote(f"{stack_id}|{logical_id}|{request_id}", safe="")
    query = urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": (
                "FIXTURECREDENTIAL01/20260728/us-west-2/s3/aws4_request"
            ),
            "X-Amz-Date": "20260728T000000Z",
            "X-Amz-Expires": "3600",
            "X-Amz-SignedHeaders": "host",
            "X-Amz-Signature": "1" * 64,
        }
    )
    return f"https://{host}/{key}?{query}"


def _properties(material: str = "2" * 64) -> dict[str, object]:
    return {
        "ServiceToken": (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:h1g-tls-handler:7"
        ),
        "ActivationIdentity": "act-20260728-0001",
        "ResourceKind": "TLS_BUNDLE",
        "MaterialIdentitySha256": material,
        "SecretTargets": _secret_targets("TLS_BUNDLE"),
    }


def _secret_targets(
    resource_kind: str,
    *,
    resolved: bool = False,
) -> dict[str, object]:
    return {
        logical_id: {
            "SecretId": (
                (
                    "arn:aws:secretsmanager:us-west-2:246813579024:secret:"
                    "/keep/glm52/glm52-sky-20260724/"
                    f"act-20260728-0001/{_SECRET_PURPOSES[logical_id]}-ABC123"
                )
                if resolved
                else {"Ref": logical_id}
            ),
            "VersionStage": (
                "h1g-act-20260728-0001-"
                f"{_SECRET_PURPOSES[logical_id]}-v1"
            ),
        }
        for logical_id in _SECRET_FAMILIES[resource_kind]
    }


def _event(
    request_type: str = "Create",
    *,
    request_id: str = REQUEST_ID,
    material: str = "2" * 64,
) -> dict[str, object]:
    event: dict[str, object] = {
        "RequestType": request_type,
        "ResponseURL": _response_url(request_id=request_id),
        "StackId": STACK_ID,
        "RequestId": request_id,
        "ResourceType": "Custom::H1gTlsBundle",
        "LogicalResourceId": LOGICAL_ID,
        "ResourceProperties": _properties(material),
    }
    if request_type in {"Update", "Delete"}:
        event["PhysicalResourceId"] = (
            "h1g-tls-bundle-855ca1015437010212c8d4726deaf369"
        )
    if request_type == "Update":
        event["OldResourceProperties"] = _properties()
    return event


def test_exact_rendered_secret_targets_are_accepted_and_closed() -> None:
    """Break caught: emitted two/six-target properties are rejected or widened."""

    from glm52_enforcement.support_custom_resources import (
        begin_custom_resource_operation,
    )

    tls = _event()
    tls["ResourceProperties"]["SecretTargets"] = _secret_targets("TLS_BUNDLE")
    tls_operation = begin_custom_resource_operation(tls)
    assert tuple(
        target.secret_logical_id for target in tls_operation.secret_targets
    ) == _SECRET_FAMILIES["TLS_BUNDLE"]

    sky = _sky_event()
    sky["ResourceProperties"]["SecretTargets"] = _secret_targets(
        "SKY_BOOTSTRAP"
    )
    sky_operation = begin_custom_resource_operation(sky)
    assert tuple(
        target.secret_logical_id for target in sky_operation.secret_targets
    ) == _SECRET_FAMILIES["SKY_BOOTSTRAP"]

    resolved = copy.deepcopy(tls)
    resolved["ResourceProperties"]["SecretTargets"] = _secret_targets(
        "TLS_BUNDLE",
        resolved=True,
    )
    assert begin_custom_resource_operation(resolved).secret_targets

    mutants = []
    missing = copy.deepcopy(tls)
    del missing["ResourceProperties"]["SecretTargets"][
        "CombinedHostTlsSecret"
    ]
    mutants.append(missing)
    extra = copy.deepcopy(tls)
    extra["ResourceProperties"]["SecretTargets"]["ForbiddenNinthSecret"] = {
        "SecretId": {"Ref": "ForbiddenNinthSecret"},
        "VersionStage": "h1g-act-20260728-0001-forbidden-v1",
    }
    mutants.append(extra)
    wrong_ref = copy.deepcopy(tls)
    wrong_ref["ResourceProperties"]["SecretTargets"][
        "CombinedHostTlsSecret"
    ]["SecretId"] = {"Ref": "CaIssuanceSecret"}
    mutants.append(wrong_ref)
    wrong_stage = copy.deepcopy(tls)
    wrong_stage["ResourceProperties"]["SecretTargets"][
        "CombinedHostTlsSecret"
    ]["VersionStage"] = "h1g-act-20260728-0001-ca-issuance-v1"
    mutants.append(wrong_stage)
    foreign_identity = copy.deepcopy(resolved)
    foreign_identity["ResourceProperties"]["SecretTargets"][
        "CombinedHostTlsSecret"
    ]["SecretId"] = (
        "arn:aws:secretsmanager:us-east-1:246813579024:secret:"
        "/keep/glm52/glm52-sky-20260724/act-20260728-0001/"
        "combined-host-server-tls-ABC123"
    )
    mutants.append(foreign_identity)
    unknown_property = copy.deepcopy(tls)
    unknown_property["ResourceProperties"]["Arbitrary"] = True
    mutants.append(unknown_property)
    for mutant in mutants:
        with pytest.raises(ValueError):
            begin_custom_resource_operation(mutant)


def _sky_event(
    request_type: str = "Create",
    *,
    request_id: str = "11111111-2222-4333-8444-555555555555",
) -> dict[str, object]:
    logical_id = "SkyBootstrapCustomResource"
    event = _event(request_type, request_id=request_id, material="1" * 64)
    event["ResponseURL"] = _response_url(
        request_id=request_id,
        logical_id=logical_id,
    )
    event["ResourceType"] = "Custom::H1gSkyBootstrap"
    event["LogicalResourceId"] = logical_id
    event["ResourceProperties"]["ResourceKind"] = "SKY_BOOTSTRAP"
    event["ResourceProperties"]["SecretTargets"] = _secret_targets(
        "SKY_BOOTSTRAP"
    )
    if request_type == "Update":
        event["OldResourceProperties"]["ResourceKind"] = "SKY_BOOTSTRAP"
        event["OldResourceProperties"]["MaterialIdentitySha256"] = "1" * 64
        event["OldResourceProperties"]["SecretTargets"] = _secret_targets(
            "SKY_BOOTSTRAP"
        )
    return event


def _completed_tls_operation() -> object:
    from glm52_enforcement.support_custom_resources import (
        begin_custom_resource_operation,
        complete_custom_resource_create,
        record_direct_grant_evidence,
        record_material_write,
        record_secret_version_observation,
    )

    operation = begin_custom_resource_operation(_event())
    operation = record_direct_grant_evidence(
        operation,
        coordinate=(
            "ACTIVATION#act-20260728-0001#CUSTOM_RESOURCE_GRANT#"
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        ),
        evidence_sha256="e" * 64,
    )
    bindings = (
        ("AttestationClientTlsSecret", "attestation-client-tls", "a"),
        ("LaunchAdmissionClientTlsSecret", "launch-admission-client-tls", "b"),
        ("NumericBindingClientTlsSecret", "numeric-binding-client-tls", "c"),
        (
            "RetainedCancellationClientTlsSecret",
            "retained-cancellation-client-tls",
            "d",
        ),
        ("CombinedHostTlsSecret", "combined-host-server-tls", "e"),
        ("CaIssuanceSecret", "ca-issuance", "f"),
    )
    for logical_id, purpose, character in bindings:
        stage = f"h1g-act-20260728-0001-{purpose}-v1"
        operation = record_material_write(
            operation,
            secret_logical_id=logical_id,
            version_id=character * 32,
            version_stage=stage,
            material_sha256="1" * 64,
            put_request_identity_sha256="2" * 64,
            put_response_identity_sha256="3" * 64,
        )
        operation = record_secret_version_observation(
            operation,
            secret_logical_id=logical_id,
            version_id=character * 32,
            version_stage=stage,
            material_sha256="1" * 64,
            list_versions_identity_sha256="4" * 64,
        )
    return complete_custom_resource_create(operation)


def test_success_callback_exposes_exact_top_level_getatt_attributes() -> None:
    """Break caught: nested version data cannot satisfy template Fn::GetAtt."""

    from glm52_enforcement.support_custom_resources import (
        build_success_callback_data,
        serialize_callback,
    )

    operation = _completed_tls_operation()
    data = build_success_callback_data(
        operation,
        issuance_identity_sha256="a" * 64,
        bundle_sha256="b" * 64,
    )
    assert "versions" not in data
    for observation in operation.version_observations:
        prefix = observation.secret_logical_id
        assert data[prefix + "VersionId"] == observation.version_id
        assert data[prefix + "VersionStage"] == observation.version_stage
        assert (
            type(data[prefix + "VersionIdentitySha256"]) is str
            and len(data[prefix + "VersionIdentitySha256"]) == 64
        )
    nested_only = {
        "status": data["status"],
        "issuance_id": data["issuance_id"],
        "bundle_sha256": data["bundle_sha256"],
        "versions": {
            item.secret_logical_id: {
                "version_id": item.version_id,
                "version_stage": item.version_stage,
                "version_identity_sha256": "0" * 64,
            }
            for item in operation.version_observations
        },
    }
    with pytest.raises(ValueError):
        serialize_callback(
            operation,
            status="SUCCESS",
            data=nested_only,
            reason="H1G_CUSTOM_RESOURCE_COMPLETE",
        )


def test_production_handler_create_writes_observes_and_callbacks_once() -> None:
    """Break caught: packaged Create has no executable AWS/callback route."""

    from glm52_enforcement.support_custom_resource_handler import (
        CustomResourceHandlerServices,
        main,
    )

    event = _event()
    event["ResourceProperties"]["SecretTargets"] = _secret_targets(
        "TLS_BUNDLE",
        resolved=True,
    )

    class SecretsManager:
        def __init__(self) -> None:
            self.values: dict[tuple[str, str], str] = {}
            self.stages: dict[tuple[str, str], str] = {}
            self.put_calls: list[dict[str, object]] = []

        def put_secret_value(self, **request: object) -> dict[str, object]:
            self.put_calls.append(dict(request))
            version_id = str(request["ClientRequestToken"])
            key = (str(request["SecretId"]), version_id)
            self.values[key] = str(request["SecretString"])
            self.stages[key] = request["VersionStages"][0]
            return {
                "ARN": request["SecretId"],
                "Name": request["SecretId"],
                "VersionId": version_id,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "put-" + str(len(self.put_calls)),
                },
            }

        def list_secret_version_ids(
            self,
            **request: object,
        ) -> dict[str, object]:
            versions = [
                {
                    "VersionId": version_id,
                    "VersionStages": [stage],
                }
                for (secret_id, version_id), stage in self.stages.items()
                if secret_id == request["SecretId"]
            ]
            return {
                "Versions": versions,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "list-" + str(len(versions)),
                },
            }

        def get_secret_value(
            self,
            **request: object,
        ) -> dict[str, object]:
            key = (str(request["SecretId"]), str(request["VersionId"]))
            return {
                "ARN": request["SecretId"],
                "Name": request["SecretId"],
                "VersionId": request["VersionId"],
                "VersionStages": [request["VersionStage"]],
                "SecretString": self.values[key],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "get-1",
                },
            }

    class Callback:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bytes, dict[str, str]]] = []

        def put(
            self,
            *,
            url: str,
            body: bytes,
            headers: dict[str, str],
        ) -> dict[str, object]:
            self.calls.append((url, body, headers))
            return {
                "status_code": 200,
                "final_url": url,
                "redirect_count": 0,
            }

    secrets = SecretsManager()
    callback = Callback()
    material = {
        logical_id: json.dumps(
            {"logical_id": logical_id, "fixture": "material"},
            sort_keys=True,
            separators=(",", ":"),
        )
        for logical_id in _SECRET_FAMILIES["TLS_BUNDLE"]
    }
    result = main(
        event,
        object(),
        services=CustomResourceHandlerServices(
            secretsmanager=secrets,
            callback_transport=callback,
            material_issuer=lambda operation: material,
            tls_projector=lambda observed: {
                "TlsCaDerSha256": "9" * 64,
                "AttestationClientDerSha256": "1" * 64,
                "AttestationServerDerSha256": "5" * 64,
                "LaunchAdmissionClientDerSha256": "2" * 64,
                "LaunchAdmissionServerDerSha256": "6" * 64,
                "NumericBindingClientDerSha256": "3" * 64,
                "NumericBindingServerDerSha256": "7" * 64,
                "RetainedCancellationClientDerSha256": "4" * 64,
                "RetainedCancellationServerDerSha256": "8" * 64,
                "TlsNotValidBefore": "2026-07-29T12:00:00Z",
                "TlsNotValidAfter": "2026-08-01T12:00:00Z",
            },
            direct_grant_producer=lambda operation: {
                "SK": (
                    f"ACTIVATION#{operation.activation_identity}"
                    f"#CUSTOM_RESOURCE_GRANT#{operation.request_id}"
                ),
                "canonical_body_sha256": "e" * 64,
            },
        ),
    )
    assert result == {
        "schema_version": 1,
        "record_type": "glm52_h1g_custom_resource_result_v1",
        "status": "SUCCESS",
        "physical_resource_id": (
            "h1g-tls-bundle-855ca1015437010212c8d4726deaf369"
        ),
        "operation_token": (
            "665cd2ae18d2e815658de375d0ba30c613c6381496b7ab5b0572d524f9f011ef"
        ),
        "callback_body_sha256": hashlib.sha256(
            callback.calls[0][1]
        ).hexdigest(),
        "secret_version_ids": {
            logical_id: hashlib.sha256(
                (
                    result["operation_token"]
                    + ":"
                    + logical_id
                ).encode("ascii")
            ).hexdigest()
            for logical_id in _SECRET_FAMILIES["TLS_BUNDLE"]
        },
        "direct_grant_evidence_coordinate": (
            "ACTIVATION#act-20260728-0001#CUSTOM_RESOURCE_GRANT#"
            + event["RequestId"]
        ),
        "direct_grant_evidence_sha256": "e" * 64,
    }
    assert len(secrets.put_calls) == 6
    assert len(callback.calls) == 1
    callback_url, callback_body, callback_headers = callback.calls[0]
    assert callback_url == event["ResponseURL"]
    assert callback_headers == {
        "content-length": str(len(callback_body)),
        "content-type": "",
    }
    callback_json = json.loads(callback_body)
    assert callback_json["Status"] == "SUCCESS"
    assert callback_json["PhysicalResourceId"] == result[
        "physical_resource_id"
    ]
    assert "versions" not in callback_json["Data"]
    assert callback_json["Data"]["AttestationClientDerSha256"] == "1" * 64
    assert callback_json["Data"]["AttestationServerDerSha256"] == "5" * 64
    assert callback_json["Data"]["DirectGrantEvidenceCoordinate"] == (
        result["direct_grant_evidence_coordinate"]
    )
    assert callback_json["Data"]["DirectGrantEvidenceSha256"] == "e" * 64
    assert len(callback_body) <= 4096
    assert all(
        material_value not in callback_body.decode("ascii")
        for material_value in material.values()
    )


def test_custom_resource_identity_is_stable_but_operation_token_is_request_bound() -> None:
    """Break caught: RequestId/Type causes replacement or material-changing Update writes."""

    from glm52_enforcement.support_custom_resources import (
        begin_custom_resource_operation,
    )

    create = begin_custom_resource_operation(_event())
    update_event = _event(
        "Update",
        request_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff",
    )
    update_event["PhysicalResourceId"] = create.physical_resource_id
    update = begin_custom_resource_operation(update_event)
    assert create.physical_resource_id == update.physical_resource_id
    assert create.operation_token != update.operation_token
    assert create.material_writes == ()
    assert update.material_writes == ()

    changed = copy.deepcopy(update_event)
    changed["ResourceProperties"]["MaterialIdentitySha256"] = "f" * 64
    with pytest.raises(ValueError):
        begin_custom_resource_operation(changed)


def test_resource_kind_owns_exact_two_plus_six_writes_and_observed_versions() -> None:
    """Break caught: a handler writes another kind or claims an unobserved version."""

    from glm52_enforcement.support_custom_resources import (
        begin_custom_resource_operation,
        complete_custom_resource_create,
        record_material_write,
        record_direct_grant_evidence,
        record_secret_version_observation,
    )

    sky = begin_custom_resource_operation(_sky_event())
    sky_bindings = (
        (
            "RawSkyTokenSecret",
            "h1g-act-20260728-0001-raw-sky-token-v1",
            "a" * 32,
        ),
        (
            "SkyBootstrapSecret",
            "h1g-act-20260728-0001-sky-bootstrap-hash-v1",
            "b" * 32,
        ),
    )
    for logical_id, stage, version_id in sky_bindings:
        sky = record_material_write(
            sky,
            secret_logical_id=logical_id,
            version_id=version_id,
            version_stage=stage,
            material_sha256="3" * 64,
            put_request_identity_sha256="4" * 64,
            put_response_identity_sha256="5" * 64,
        )
        sky = record_secret_version_observation(
            sky,
            secret_logical_id=logical_id,
            version_id=version_id,
            version_stage=stage,
            material_sha256="3" * 64,
            list_versions_identity_sha256="6" * 64,
        )
    assert complete_custom_resource_create(sky).phase == "CREATE_RECONCILED"

    tls = begin_custom_resource_operation(_event())
    tls = record_direct_grant_evidence(
        tls,
        coordinate=(
            "ACTIVATION#act-20260728-0001#CUSTOM_RESOURCE_GRANT#"
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        ),
        evidence_sha256="e" * 64,
    )
    with pytest.raises(ValueError):
        record_material_write(
            tls,
            secret_logical_id="RawSkyTokenSecret",
            version_id="c" * 32,
            version_stage="h1g-act-20260728-0001-raw-sky-token-v1",
            material_sha256="3" * 64,
            put_request_identity_sha256="4" * 64,
            put_response_identity_sha256="5" * 64,
        )
    with pytest.raises(ValueError):
        record_material_write(
            tls,
            secret_logical_id="AttestationClientTlsSecret",
            version_id="c" * 32,
            version_stage="wrong-stage",
            material_sha256="3" * 64,
            put_request_identity_sha256="4" * 64,
            put_response_identity_sha256="5" * 64,
        )
    tls_bindings = (
        ("AttestationClientTlsSecret", "attestation-client-tls", "c"),
        ("LaunchAdmissionClientTlsSecret", "launch-admission-client-tls", "d"),
        ("NumericBindingClientTlsSecret", "numeric-binding-client-tls", "e"),
        (
            "RetainedCancellationClientTlsSecret",
            "retained-cancellation-client-tls",
            "f",
        ),
        ("CombinedHostTlsSecret", "combined-host-server-tls", "0"),
        ("CaIssuanceSecret", "ca-issuance", "1"),
    )
    for logical_id, purpose, version_character in tls_bindings:
        version_id = version_character * 32
        stage = f"h1g-act-20260728-0001-{purpose}-v1"
        tls = record_material_write(
            tls,
            secret_logical_id=logical_id,
            version_id=version_id,
            version_stage=stage,
            material_sha256="7" * 64,
            put_request_identity_sha256="8" * 64,
            put_response_identity_sha256="9" * 64,
        )
        tls = record_secret_version_observation(
            tls,
            secret_logical_id=logical_id,
            version_id=version_id,
            version_stage=stage,
            material_sha256="7" * 64,
            list_versions_identity_sha256="a" * 64,
        )
    assert complete_custom_resource_create(tls).phase == "CREATE_RECONCILED"


def test_custom_resource_blocks_premature_success_late_write_and_unbacked_observation() -> None:
    """Break caught: callback state advances without complete durable evidence."""

    from glm52_enforcement.support_custom_resources import (
        begin_custom_resource_operation,
        cache_callback,
        observe_cloudformation,
        record_material_write,
    )

    operation = begin_custom_resource_operation(_event())
    success_data = {
        "status": "ISSUED",
        "issuance_id": "issuance-" + "a" * 32,
        "bundle_sha256": "b" * 64,
        "version_stage_hashes": {"fixture": "c" * 64},
    }
    with pytest.raises(ValueError):
        cache_callback(
            operation,
            status="SUCCESS",
            data=success_data,
            reason="H1G_CUSTOM_RESOURCE_COMPLETE",
        )
    cached_failure = cache_callback(
        operation,
        status="FAILED",
        data={"status": "FAILED"},
        reason="H1G_OPERATION_REJECTED:" + "d" * 64,
    )
    with pytest.raises(ValueError):
        record_material_write(
            cached_failure,
            secret_logical_id="AttestationClientTlsSecret",
            version_id="e" * 32,
            version_stage=(
                "h1g-act-20260728-0001-attestation-client-tls-v1"
            ),
            material_sha256="f" * 64,
            put_request_identity_sha256="1" * 64,
            put_response_identity_sha256="2" * 64,
        )
    with pytest.raises(ValueError):
        observe_cloudformation(operation, "SUCCESS_OBSERVED")

    delete = begin_custom_resource_operation(
        {
            **_event(
                "Delete",
                request_id="cccccccc-dddd-4eee-8fff-000000000000",
            ),
            "PhysicalResourceId": operation.physical_resource_id,
        }
    )
    with pytest.raises(ValueError):
        cache_callback(
            delete,
            status="SUCCESS",
            data=success_data,
            reason="H1G_CUSTOM_RESOURCE_COMPLETE",
        )


def test_create_update_delete_callbacks_bind_observed_versions_and_tombstones() -> None:
    """Break caught: SUCCESS omits real versions or Delete lacks destruction proof."""

    from glm52_enforcement.support_custom_resources import (
        begin_custom_resource_operation,
        build_delete_callback_data,
        build_success_callback_data,
        cache_callback,
        complete_custom_resource_create,
        complete_custom_resource_delete,
        complete_identity_preserving_update,
        record_material_deletion,
        record_direct_grant_evidence,
        record_material_write,
        record_secret_version_observation,
    )

    bindings = (
        ("AttestationClientTlsSecret", "attestation-client-tls", "a"),
        ("LaunchAdmissionClientTlsSecret", "launch-admission-client-tls", "b"),
        ("NumericBindingClientTlsSecret", "numeric-binding-client-tls", "c"),
        (
            "RetainedCancellationClientTlsSecret",
            "retained-cancellation-client-tls",
            "d",
        ),
        ("CombinedHostTlsSecret", "combined-host-server-tls", "e"),
        ("CaIssuanceSecret", "ca-issuance", "f"),
    )
    create = begin_custom_resource_operation(_event())
    create = record_direct_grant_evidence(
        create,
        coordinate=(
            "ACTIVATION#act-20260728-0001#CUSTOM_RESOURCE_GRANT#"
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        ),
        evidence_sha256="e" * 64,
    )
    for logical_id, purpose, character in bindings:
        stage = f"h1g-act-20260728-0001-{purpose}-v1"
        create = record_material_write(
            create,
            secret_logical_id=logical_id,
            version_id=character * 32,
            version_stage=stage,
            material_sha256="1" * 64,
            put_request_identity_sha256="2" * 64,
            put_response_identity_sha256="3" * 64,
        )
        create = record_secret_version_observation(
            create,
            secret_logical_id=logical_id,
            version_id=character * 32,
            version_stage=stage,
            material_sha256="1" * 64,
            list_versions_identity_sha256="4" * 64,
        )
    create = complete_custom_resource_create(create)
    create_data = build_success_callback_data(
        create,
        issuance_identity_sha256="5" * 64,
        bundle_sha256="6" * 64,
    )
    assert create_data["status"] == "ISSUED"
    for logical_id, purpose, character in bindings:
        assert create_data[logical_id + "VersionId"] == character * 32
        assert create_data[logical_id + "VersionStage"] == (
            f"h1g-act-20260728-0001-{purpose}-v1"
        )
        assert len(create_data[logical_id + "VersionIdentitySha256"]) == 64
    assert cache_callback(
        create,
        status="SUCCESS",
        data=create_data,
        reason="H1G_CUSTOM_RESOURCE_COMPLETE",
    ).callback_body is not None

    update_event = _event(
        "Update",
        request_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff",
    )
    update_event["PhysicalResourceId"] = create.physical_resource_id
    update = begin_custom_resource_operation(update_event)
    for logical_id, purpose, character in bindings:
        update = record_secret_version_observation(
            update,
            secret_logical_id=logical_id,
            version_id=character * 32,
            version_stage=f"h1g-act-20260728-0001-{purpose}-v1",
            material_sha256="1" * 64,
            list_versions_identity_sha256="7" * 64,
        )
    update = complete_identity_preserving_update(update)
    update_data = build_success_callback_data(
        update,
        issuance_identity_sha256="5" * 64,
        bundle_sha256="6" * 64,
    )
    assert update_data["status"] == "VERIFIED"
    assert cache_callback(
        update,
        status="SUCCESS",
        data=update_data,
        reason="H1G_CUSTOM_RESOURCE_COMPLETE",
    ).callback_body is not None

    delete_event = _event(
        "Delete",
        request_id="cccccccc-dddd-4eee-8fff-000000000000",
    )
    delete_event["PhysicalResourceId"] = create.physical_resource_id
    delete = begin_custom_resource_operation(delete_event)
    for logical_id, _purpose, _character in bindings:
        delete = record_material_deletion(
            delete,
            secret_logical_id=logical_id,
            tombstone_sha256="8" * 64,
        )
    delete = complete_custom_resource_delete(
        delete,
        deleted_secret_logical_ids=tuple(item[0] for item in bindings),
        tombstone_sha256="8" * 64,
    )
    delete_data = build_delete_callback_data(delete)
    assert delete_data == {
        "status": "DELETED",
        "deleted_secret_logical_ids": [item[0] for item in bindings],
        "tombstone_sha256": "8" * 64,
    }
    assert cache_callback(
        delete,
        status="SUCCESS",
        data=delete_data,
        reason="H1G_CUSTOM_RESOURCE_COMPLETE",
    ).callback_body is not None


def test_response_url_is_bound_to_partition_region_stack_request_and_logical_id() -> None:
    """Break caught: callback can be redirected to a foreign or malformed host."""

    from glm52_enforcement.support_custom_resources import validate_response_url

    assert (
        validate_response_url(
            _response_url(),
            stack_id=STACK_ID,
            request_id=REQUEST_ID,
            logical_resource_id=LOGICAL_ID,
        )
        == _response_url()
    )
    mutants = (
        _response_url(host="attacker.example.com"),
        _response_url(stack_id=STACK_ID.replace("246813579024", "000000000000")),
        _response_url(request_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"),
        _response_url(logical_id="ForeignLogicalId"),
        _response_url().replace("us-west-2%2Fs3", "us-east-1%2Fs3"),
        _response_url().replace("https://", "http://"),
    )
    for mutant in mutants:
        with pytest.raises(ValueError):
            validate_response_url(
                mutant,
                stack_id=STACK_ID,
                request_id=REQUEST_ID,
                logical_resource_id=LOGICAL_ID,
            )


def test_callback_bodies_are_nonsecret_canonical_and_within_4096_bytes() -> None:
    """Break caught: private material or worst-case callback exceeds CFN's budget."""

    from glm52_enforcement.support_custom_resources import (
        build_success_callback_data,
        serialize_callback,
    )

    operation = _completed_tls_operation()
    data = build_success_callback_data(
        operation,
        issuance_identity_sha256="a" * 64,
        bundle_sha256="b" * 64,
    )
    success = serialize_callback(
        operation,
        status="SUCCESS",
        data=data,
        reason="H1G_CUSTOM_RESOURCE_COMPLETE",
    )
    failed = serialize_callback(
        operation,
        status="FAILED",
        data={"status": "FAILED"},
        reason="H1G_" + "X" * 64 + ":" + "f" * 64,
    )
    assert len(success) <= 4096
    assert len(failed) <= 4096
    assert success == serialize_callback(
        operation,
        status="SUCCESS",
        data=data,
        reason="H1G_CUSTOM_RESOURCE_COMPLETE",
    )
    assert b"PRIVATE KEY" not in success
    assert b"certificate" not in success.lower()
    with pytest.raises(ValueError):
        serialize_callback(
            operation,
            status="SUCCESS",
            data={
                **data,
                "private_key": "forbidden-sensitive-value",
            },
            reason="H1G_CUSTOM_RESOURCE_COMPLETE",
        )


def test_callback_uses_closed_identity_and_reason_grammar() -> None:
    """Break caught: arbitrary callback text smuggles credentials or key material."""

    from glm52_enforcement.support_custom_resources import (
        build_success_callback_data,
        serialize_callback,
    )

    operation = _completed_tls_operation()
    valid_data = build_success_callback_data(
        operation,
        issuance_identity_sha256="a" * 64,
        bundle_sha256="b" * 64,
    )
    assert serialize_callback(
        operation,
        status="SUCCESS",
        data=valid_data,
        reason="H1G_CUSTOM_RESOURCE_COMPLETE",
    )
    for issuance_id in (
        "caller-selected",
        "AKIAABCDEFGHIJKLMNOP",
        "eyJhbGciOiJIUzI1NiJ9.payload.signature",
    ):
        with pytest.raises(ValueError):
            serialize_callback(
                operation,
                status="SUCCESS",
                data={**valid_data, "issuance_id": issuance_id},
                reason="H1G_CUSTOM_RESOURCE_COMPLETE",
            )
    for reason in (
        "arbitrary free-form reason",
        "H1G_FAILURE:AKIAABCDEFGHIJKLMNOP",
        "-----BEGIN PRIVATE KEY-----",
    ):
        with pytest.raises(ValueError):
            serialize_callback(
                operation,
                status="FAILED",
                data={"status": "FAILED"},
                reason=reason,
            )


def test_partial_write_failed_create_delete_and_callback_states_reconcile() -> None:
    """Break caught: replay regenerates material or transport success means CFN observed."""

    from glm52_enforcement.support_custom_resources import (
        begin_custom_resource_operation,
        cache_callback,
        complete_custom_resource_delete,
        complete_failed_create_rollback,
        enter_failed_create_rollback,
        mark_callback_attempted,
        observe_cloudformation,
        record_material_deletion,
        record_material_write,
        record_transport_result,
        reconcile_delivery,
    )

    operation = begin_custom_resource_operation(_event())
    partial = record_material_write(
        operation,
        secret_logical_id="AttestationClientTlsSecret",
        version_id="a" * 32,
        version_stage="h1g-act-20260728-0001-attestation-client-tls-v1",
        material_sha256="3" * 64,
        put_request_identity_sha256="7" * 64,
        put_response_identity_sha256="8" * 64,
    )
    replayed = reconcile_delivery(partial, _event())
    assert replayed == partial
    assert record_material_write(
        replayed,
        secret_logical_id="AttestationClientTlsSecret",
        version_id="a" * 32,
        version_stage="h1g-act-20260728-0001-attestation-client-tls-v1",
        material_sha256="3" * 64,
        put_request_identity_sha256="7" * 64,
        put_response_identity_sha256="8" * 64,
    ) == partial
    with pytest.raises(ValueError):
        record_material_write(
            replayed,
            secret_logical_id="AttestationClientTlsSecret",
            version_id="b" * 32,
            version_stage="h1g-act-20260728-0001-attestation-client-tls-v1",
            material_sha256="4" * 64,
            put_request_identity_sha256="7" * 64,
            put_response_identity_sha256="8" * 64,
        )

    rollback = enter_failed_create_rollback(partial)
    rollback = record_material_deletion(
        rollback,
        secret_logical_id="AttestationClientTlsSecret",
        tombstone_sha256="5" * 64,
    )
    rolled_back = complete_failed_create_rollback(rollback)
    assert rolled_back.phase == "FAILED_CREATE_ROLLED_BACK"
    cached = cache_callback(
        rolled_back,
        status="FAILED",
        data={"status": "FAILED"},
        reason="H1G_CREATE_ROLLED_BACK:" + "9" * 64,
    )
    attempted = mark_callback_attempted(cached)
    transported = record_transport_result(
        attempted,
        status_code=200,
        final_url=_response_url(),
        redirect_count=0,
    )
    assert transported.callback_attempted is True
    assert transported.transport_result == "HTTP_200"
    assert transported.cloudformation_observation is None
    observed = observe_cloudformation(transported, "FAILED_OBSERVED")
    assert observed.cloudformation_observation == "FAILED_OBSERVED"
    assert reconcile_delivery(observed, _event()).callback_body == cached.callback_body

    different = _event()
    different["ResourceProperties"]["ActivationIdentity"] = "act-foreign"
    with pytest.raises(ValueError):
        reconcile_delivery(observed, different)

    delete_event = _event(
        "Delete",
        request_id="cccccccc-dddd-4eee-8fff-000000000000",
    )
    delete_event["PhysicalResourceId"] = operation.physical_resource_id
    deletion = begin_custom_resource_operation(delete_event)
    assert deletion.request_type == "Delete"
    assert deletion.physical_resource_id == operation.physical_resource_id
    assert deletion.material_writes == ()
    for logical_id in (
        "AttestationClientTlsSecret",
        "LaunchAdmissionClientTlsSecret",
        "NumericBindingClientTlsSecret",
        "RetainedCancellationClientTlsSecret",
        "CombinedHostTlsSecret",
        "CaIssuanceSecret",
    ):
        deletion = record_material_deletion(
            deletion,
            secret_logical_id=logical_id,
            tombstone_sha256="6" * 64,
        )
    deleted = complete_custom_resource_delete(
        deletion,
        deleted_secret_logical_ids=(
            "AttestationClientTlsSecret",
            "LaunchAdmissionClientTlsSecret",
            "NumericBindingClientTlsSecret",
            "RetainedCancellationClientTlsSecret",
            "CombinedHostTlsSecret",
            "CaIssuanceSecret",
        ),
        tombstone_sha256="6" * 64,
    )
    assert deleted.phase == "DELETE_RECONCILED"


def test_reverse_deletion_graph_keeps_handler_permission_kms_access_and_secrets() -> None:
    """Break caught: a custom-resource dependency can disappear before Delete callback."""

    from test_glm52_enforcement_support_plane import _bundle
    from glm52_enforcement.support_custom_resources import (
        validate_reverse_deletion_dependencies,
    )

    bundle = _bundle()
    assert validate_reverse_deletion_dependencies(bundle.support_template) is True
    mutant = copy.deepcopy(bundle.support_template)
    mutant["Resources"]["TlsBundleCustomResource"]["DependsOn"].remove(
        "TlsHandlerVersion"
    )
    with pytest.raises(ValueError):
        validate_reverse_deletion_dependencies(mutant)
