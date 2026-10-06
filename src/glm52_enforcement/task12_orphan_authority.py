"""Import-light, fail-closed Task 12 orphan authority production.

This module deliberately separates three moments:

* PRECREATE freezes the retained KMS-grant baseline before support mutation.
* POSTCREATE records creation provenance and the exact active inventory.
* FINALIZE performs guarded cleanup and only then constructs the Task 12 audit.

No function in this module retries an AWS mutation.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from .canonical import canonical_json_bytes, canonical_sha256
from .dynamodb import decode_item, encode_item
from .support_plane import (
    SUPPORT_DELETION_ROLE_ARN,
    TLS_HANDLER_ROLE_ARN,
    SupportInputs,
)
from .task12_orphan_audit import (
    ACCOUNT_ID,
    AUDITED_RESOURCE_TYPES,
    DirectGrantAttribution,
    INTENTIONAL_RETAINED_RESOURCE_TYPES,
    KmsGrantIdentity,
    KmsGrantScan,
    REGION,
    RUN_ID,
    ResourceInventoryScan,
    RetainedResource,
    ServiceGrantAttribution,
    audit_orphans,
)


PRECREATE_RECORD_TYPE = "glm52_task12_orphan_precreate_baseline_v1"
ACTIVATION_RECORD_TYPE = "glm52_task12_activation_orphan_authority_v1"
DIRECT_EVIDENCE_RECORD_TYPE = "glm52_task12_direct_grant_evidence_v1"
COORDINATE_RECORD_TYPE = "glm52_task12_activation_orphan_coordinate_v1"
_DIRECT_OPERATIONS = ("Decrypt", "Encrypt", "GenerateDataKey")
_DIRECT_PROVENANCE = frozenset(
    {"DIRECT_RESPONSE", "LIST_GRANTS_RECONCILIATION"}
)
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(label + " must be a nonempty exact string")
    return value


def _utc(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.strptime(text, _UTC_FORMAT)
    except ValueError as exc:
        raise ValueError(label + " must be canonical UTC") from exc
    return parsed.replace(tzinfo=timezone.utc)


def _response_request_id(value: object, label: str) -> str:
    if type(value) is not dict:
        raise ValueError(label + " response must be exact")
    metadata = value.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or metadata.get("RetryAttempts") != 0
    ):
        raise ValueError(label + " response metadata is not zero-retry success")
    return _text(metadata.get("RequestId"), label + " request ID")


def _context_order(key: str) -> tuple[int, str]:
    if key == "RunId":
        return (0, key)
    if key == "ActivationId":
        return (1, key)
    return (2, key)


def _grant_projection(raw: object) -> Mapping[str, object]:
    if type(raw) is not dict:
        raise ValueError("ListGrants member is malformed")
    required = (
        "GrantId",
        "Name",
        "GranteePrincipal",
        "RetiringPrincipal",
        "Operations",
    )
    if any(type(raw.get(field)) is not str for field in required[:-1]):
        raise ValueError("ListGrants member identity is incomplete")
    operations = raw.get("Operations")
    constraints = raw.get("Constraints")
    if (
        type(operations) is not list
        or not operations
        or any(type(item) is not str or not item for item in operations)
        or len(operations) != len(set(operations))
        or type(constraints) is not dict
        or type(constraints.get("EncryptionContextSubset")) is not dict
    ):
        raise ValueError("ListGrants member scope is incomplete")
    context_mapping = constraints["EncryptionContextSubset"]
    if (
        not context_mapping
        or any(
            type(key) is not str
            or not key
            or type(item) is not str
            or not item
            for key, item in context_mapping.items()
        )
    ):
        raise ValueError("ListGrants encryption context is incomplete")
    context = tuple(
        sorted(context_mapping.items(), key=lambda item: _context_order(item[0]))
    )
    return {
        "grant_id": raw["GrantId"],
        "name": raw["Name"],
        "grantee": raw["GranteePrincipal"],
        "retiring_principal": raw["RetiringPrincipal"],
        "operations": tuple(operations),
        "encryption_context": context,
        # A present retiring principal is the authenticated ListGrants proof
        # that this grant has a retirement path. RevokeGrant remains guarded
        # by the exact key/grant identity at FINALIZE.
        "revocable": True,
    }


def _typed_grant(grant: Mapping[str, object]) -> KmsGrantIdentity:
    return KmsGrantIdentity(
        grant_id=grant["grant_id"],
        grant_name=grant["name"],
        grantee_principal=grant["grantee"],
        retiring_principal=grant["retiring_principal"],
        operations=tuple(sorted(grant["operations"])),
        encryption_context_identity_sha256=canonical_sha256(
            grant["encryption_context"]
        ),
    )


def _scan_kms_grants(
    *,
    kms: object,
    key_arn: str,
    observed_at: str,
) -> Mapping[str, object]:
    _text(key_arn, "retained KMS key ARN")
    _utc(observed_at, "ListGrants observed_at")
    marker: Optional[str] = None
    page_number = 0
    raw_pages = []
    grants = []
    while True:
        request: dict[str, object] = {"KeyId": key_arn, "Limit": 100}
        if marker is not None:
            request["Marker"] = marker
        response = kms.list_grants(**request)
        request_id = _response_request_id(response, "ListGrants")
        members = response.get("Grants")
        truncated = response.get("Truncated")
        if type(members) is not list or type(truncated) is not bool:
            raise ValueError("ListGrants pagination response is malformed")
        page_number += 1
        raw_pages.append(
            {
                "page": page_number,
                "request": request,
                "request_id": request_id,
                "grants": members,
                "truncated": truncated,
                "next_marker": response.get("NextMarker"),
            }
        )
        grants.extend(_grant_projection(member) for member in members)
        if not truncated:
            break
        next_marker = response.get("NextMarker")
        if (
            type(next_marker) is not str
            or not next_marker
            or next_marker == marker
        ):
            raise ValueError("ListGrants pagination is incomplete")
        marker = next_marker
    grants.sort(key=lambda item: item["grant_id"])
    grant_ids = tuple(item["grant_id"] for item in grants)
    if len(grant_ids) != len(set(grant_ids)):
        raise ValueError("ListGrants returned duplicate grant identities")
    identity = canonical_sha256(
        {
            "key_arn": key_arn,
            "observed_at": observed_at,
            "pages": raw_pages,
            "grants": grants,
            "pagination_complete": True,
        }
    )
    return {
        "grants": tuple(grants),
        "typed_grants": tuple(_typed_grant(item) for item in grants),
        "pagination_complete": True,
        "observed_at": observed_at,
        "list_grants_identity_sha256": identity,
        "page_request_ids": tuple(
            page["request_id"] for page in raw_pages
        ),
    }


def _expected_retained(
    value: object,
) -> Tuple[RetainedResource, ...]:
    if (
        type(value) is not tuple
        or any(not isinstance(item, RetainedResource) for item in value)
    ):
        raise ValueError("retained inventory must be a typed exact tuple")
    if tuple(item.resource_type for item in value) != (
        INTENTIONAL_RETAINED_RESOURCE_TYPES
    ):
        raise ValueError(
            "retained inventory is missing, duplicated, or unordered"
        )
    identities = tuple((item.resource_type, item.resource_id) for item in value)
    if len(identities) != len(set(identities)):
        raise ValueError("retained inventory contains a duplicate")
    return value


def _with_identity(body: Mapping[str, object]) -> Mapping[str, object]:
    result = dict(body)
    result["canonical_body_sha256"] = canonical_sha256(body)
    return result


def _validate_identity(value: object, record_type: str) -> Mapping[str, object]:
    if type(value) is not dict or value.get("record_type") != record_type:
        raise ValueError(record_type + " record is absent or malformed")
    identity = value.get("canonical_body_sha256")
    body = dict(value)
    body.pop("canonical_body_sha256", None)
    if identity != canonical_sha256(body):
        raise ValueError(record_type + " canonical identity drifted")
    return value


def capture_precreate_baseline(
    *,
    kms: object,
    activation_id: str,
    retained_kms_key_arn: str,
    expected_retained: Tuple[RetainedResource, ...],
    observed_at: str,
) -> Mapping[str, object]:
    """Freeze the complete KMS baseline before any support mutation."""

    _text(activation_id, "activation ID")
    expected = _expected_retained(expected_retained)
    scan = _scan_kms_grants(
        kms=kms,
        key_arn=retained_kms_key_arn,
        observed_at=observed_at,
    )
    body = {
        "record_type": PRECREATE_RECORD_TYPE,
        "schema_version": 1,
        "phase": "PRECREATE",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "retained_kms_key_arn": retained_kms_key_arn,
        "captured_before_support_mutation": True,
        "expected_retained": tuple(asdict(item) for item in expected),
        "baseline_grants": scan["grants"],
        "baseline_typed_grants": tuple(
            asdict(item) for item in scan["typed_grants"]
        ),
        "pagination_complete": scan["pagination_complete"],
        "list_grants_identity_sha256": (
            scan["list_grants_identity_sha256"]
        ),
        "page_request_ids": scan["page_request_ids"],
        "observed_at": observed_at,
    }
    return _with_identity(body)


def write_o_excl_authority(
    *,
    path: Path,
    value: Mapping[str, object],
) -> Mapping[str, object]:
    """Create canonical PRECREATE authority or adopt byte-identical authority."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("authority path must be an absolute Path")
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("foreign existing PRECREATE authority")
        _validate_identity(value, PRECREATE_RECORD_TYPE)
        return {
            "outcome": "RECONCILED_IDENTICAL",
            "canonical_body_sha256": value["canonical_body_sha256"],
        }
    _validate_identity(value, PRECREATE_RECORD_TYPE)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("foreign existing PRECREATE authority")
        return {
            "outcome": "RECONCILED_IDENTICAL",
            "canonical_body_sha256": value["canonical_body_sha256"],
        }
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    return {
        "outcome": "CREATED",
        "canonical_body_sha256": value["canonical_body_sha256"],
    }


def _direct_request(
    *,
    activation_id: str,
    retained_kms_key_arn: str,
) -> Mapping[str, object]:
    return {
        "KeyId": retained_kms_key_arn,
        "GranteePrincipal": TLS_HANDLER_ROLE_ARN,
        "RetiringPrincipal": SUPPORT_DELETION_ROLE_ARN,
        "Operations": list(_DIRECT_OPERATIONS),
        "Constraints": {
            "EncryptionContextSubset": {
                "RunId": RUN_ID,
                "ActivationId": activation_id,
            }
        },
        "Name": f"h1g-{activation_id}-secrets",
    }


def _direct_projection(
    *,
    activation_id: str,
    retained_kms_key_arn: str,
    grant_id: str,
) -> Mapping[str, object]:
    del retained_kms_key_arn
    return {
        "grant_id": grant_id,
        "name": f"h1g-{activation_id}-secrets",
        "grantee": TLS_HANDLER_ROLE_ARN,
        "retiring_principal": SUPPORT_DELETION_ROLE_ARN,
        "operations": _DIRECT_OPERATIONS,
        "encryption_context": (
            ("RunId", RUN_ID),
            ("ActivationId", activation_id),
        ),
        "revocable": True,
    }


def _matching_direct(
    *,
    grants: Sequence[Mapping[str, object]],
    activation_id: str,
    retained_kms_key_arn: str,
    expected_grant_id: Optional[str] = None,
) -> Tuple[Mapping[str, object], ...]:
    name = f"h1g-{activation_id}-secrets"
    same_name = tuple(item for item in grants if item["name"] == name)
    matches = tuple(
        item
        for item in same_name
        if item
        == _direct_projection(
            activation_id=activation_id,
            retained_kms_key_arn=retained_kms_key_arn,
            grant_id=item["grant_id"],
        )
        and (
            expected_grant_id is None
            or item["grant_id"] == expected_grant_id
        )
    )
    if same_name and (len(same_name) != 1 or len(matches) != 1):
        raise ValueError("direct grant readback is not singular exact")
    return matches


def _evidence_key(
    *,
    activation_id: str,
    custom_resource_request_id: str,
) -> tuple[str, str]:
    return (
        f"RUN#{RUN_ID}",
        (
            f"ACTIVATION#{activation_id}#CUSTOM_RESOURCE_GRANT#"
            f"{custom_resource_request_id}"
        ),
    )


def _ddb_read_evidence(
    *,
    dynamodb: object,
    ledger_table_name: str,
    pk: str,
    sk: str,
) -> Optional[Mapping[str, object]]:
    response = dynamodb.get_item(
        TableName=ledger_table_name,
        Key=encode_item({"PK": pk, "SK": sk}),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _response_request_id(response, "DynamoDB GetItem")
    if "Item" not in response:
        return None
    decoded = decode_item(response["Item"])
    grant = decoded.get("grant")
    if type(grant) is dict:
        grant["operations"] = tuple(grant.get("operations", ()))
        grant["encryption_context"] = tuple(
            tuple(item) for item in grant.get("encryption_context", ())
        )
    if decoded.get("PK") != pk or decoded.get("SK") != sk:
        raise ValueError("foreign direct-grant ledger coordinate")
    return _validate_identity(decoded, DIRECT_EVIDENCE_RECORD_TYPE)


def _ddb_put_or_adopt(
    *,
    dynamodb: object,
    ledger_table_name: str,
    evidence: Mapping[str, object],
) -> Mapping[str, object]:
    try:
        response = dynamodb.put_item(
            TableName=ledger_table_name,
            Item=encode_item(evidence),
            ConditionExpression=(
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            ),
            ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
            ReturnConsumedCapacity="NONE",
        )
        _response_request_id(response, "DynamoDB PutItem")
    except Exception:
        # This is readback reconciliation, never a write retry.
        pass
    adopted = _ddb_read_evidence(
        dynamodb=dynamodb,
        ledger_table_name=ledger_table_name,
        pk=evidence["PK"],
        sk=evidence["SK"],
    )
    if adopted is None:
        raise ValueError("direct-grant evidence write was not durably observed")
    if adopted != evidence:
        raise ValueError("foreign direct-grant evidence occupies coordinate")
    return adopted


def create_or_adopt_direct_grant(
    *,
    kms: object,
    dynamodb: object,
    ledger_table_name: str,
    activation_id: str,
    custom_resource_request_id: str,
    retained_kms_key_arn: str,
    observed_at: str,
) -> Mapping[str, object]:
    """Create at most one direct grant and durably bind its exact provenance."""

    _text(ledger_table_name, "ledger table")
    _text(activation_id, "activation ID")
    _text(custom_resource_request_id, "custom-resource request ID")
    _utc(observed_at, "direct grant observed_at")
    pk, sk = _evidence_key(
        activation_id=activation_id,
        custom_resource_request_id=custom_resource_request_id,
    )
    existing = _ddb_read_evidence(
        dynamodb=dynamodb,
        ledger_table_name=ledger_table_name,
        pk=pk,
        sk=sk,
    )
    if existing is not None:
        replay_scan = _scan_kms_grants(
            kms=kms,
            key_arn=retained_kms_key_arn,
            observed_at=observed_at,
        )
        matches = _matching_direct(
            grants=replay_scan["grants"],
            activation_id=activation_id,
            retained_kms_key_arn=retained_kms_key_arn,
            expected_grant_id=existing["grant_id"],
        )
        if len(matches) != 1 or matches[0] != existing["grant"]:
            raise ValueError("durable direct grant no longer has exact readback")
        return existing

    request = _direct_request(
        activation_id=activation_id,
        retained_kms_key_arn=retained_kms_key_arn,
    )
    pre_scan = _scan_kms_grants(
        kms=kms,
        key_arn=retained_kms_key_arn,
        observed_at=observed_at,
    )
    pre_matches = _matching_direct(
        grants=pre_scan["grants"],
        activation_id=activation_id,
        retained_kms_key_arn=retained_kms_key_arn,
    )
    provenance: str
    grant_token: Optional[str]
    response_identity: Optional[str]
    response_request_id: Optional[str]
    reconciliation_identity: Optional[str]
    if len(pre_matches) == 1:
        grant = pre_matches[0]
        provenance = "LIST_GRANTS_RECONCILIATION"
        grant_token = None
        response_identity = None
        response_request_id = None
        reconciliation_identity = pre_scan[
            "list_grants_identity_sha256"
        ]
        post_scan = pre_scan
    else:
        create_response: Optional[Mapping[str, object]] = None
        try:
            candidate = kms.create_grant(**request)
            if type(candidate) is not dict:
                raise ValueError("CreateGrant response is malformed")
            create_response = candidate
        except Exception:
            # Ambiguous CreateGrant outcome: perform one complete readback and
            # adopt if and only if a singular exact grant now exists.
            create_response = None
        post_scan = _scan_kms_grants(
            kms=kms,
            key_arn=retained_kms_key_arn,
            observed_at=observed_at,
        )
        expected_id = (
            create_response.get("GrantId")
            if create_response is not None
            else None
        )
        post_matches = _matching_direct(
            grants=post_scan["grants"],
            activation_id=activation_id,
            retained_kms_key_arn=retained_kms_key_arn,
            expected_grant_id=expected_id,
        )
        if len(post_matches) != 1:
            raise ValueError("direct grant readback is not singular exact")
        grant = post_matches[0]
        if create_response is None:
            provenance = "LIST_GRANTS_RECONCILIATION"
            grant_token = None
            response_identity = None
            response_request_id = None
            reconciliation_identity = post_scan[
                "list_grants_identity_sha256"
            ]
        else:
            response_request_id = _response_request_id(
                create_response, "CreateGrant"
            )
            grant_token = _text(
                create_response.get("GrantToken"), "CreateGrant token"
            )
            if create_response.get("GrantId") != grant["grant_id"]:
                raise ValueError("CreateGrant response/readback identity drifted")
            provenance = "DIRECT_RESPONSE"
            response_identity = canonical_sha256(create_response)
            reconciliation_identity = None

    if provenance not in _DIRECT_PROVENANCE:
        raise AssertionError("unreachable direct provenance")
    body = {
        "record_type": DIRECT_EVIDENCE_RECORD_TYPE,
        "schema_version": 1,
        "phase": "CUSTOM_RESOURCE_CREATE",
        "PK": pk,
        "SK": sk,
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "custom_resource_request_id": custom_resource_request_id,
        "retained_kms_key_arn": retained_kms_key_arn,
        "provenance": provenance,
        "grant_id": grant["grant_id"],
        "grant_token": grant_token,
        "grant": grant,
        "create_request": request,
        "create_request_identity_sha256": canonical_sha256(request),
        "create_response_identity_sha256": response_identity,
        "create_response_request_id": response_request_id,
        "pre_list_grants_identity_sha256": (
            pre_scan["list_grants_identity_sha256"]
        ),
        "post_list_grants_identity_sha256": (
            post_scan["list_grants_identity_sha256"]
        ),
        "reconciliation_list_grants_identity_sha256": (
            reconciliation_identity
        ),
        "observed_at": observed_at,
        "closed": True,
    }
    return _ddb_put_or_adopt(
        dynamodb=dynamodb,
        ledger_table_name=ledger_table_name,
        evidence=_with_identity(body),
    )


def read_direct_grant_evidence(
    *,
    dynamodb: object,
    ledger_table_name: str,
    activation_id: str,
    custom_resource_request_id: str,
) -> Mapping[str, object]:
    """Strongly read the closed TLS custom-resource grant evidence."""

    pk, sk = _evidence_key(
        activation_id=activation_id,
        custom_resource_request_id=custom_resource_request_id,
    )
    value = _ddb_read_evidence(
        dynamodb=dynamodb,
        ledger_table_name=ledger_table_name,
        pk=pk,
        sk=sk,
    )
    if value is None:
        raise ValueError("direct-grant evidence is absent")
    return value


def _baseline_parts(
    baseline: object,
) -> tuple[Mapping[str, object], Tuple[Mapping[str, object], ...]]:
    value = _validate_identity(baseline, PRECREATE_RECORD_TYPE)
    if (
        value.get("captured_before_support_mutation") is not True
        or value.get("pagination_complete") is not True
    ):
        raise ValueError("PRECREATE baseline is not authoritative")
    grants = value.get("baseline_grants")
    if type(grants) is not tuple:
        raise ValueError("PRECREATE baseline grant inventory is malformed")
    return value, grants


def _cloudtrail_event(value: object) -> Mapping[str, object]:
    if type(value) is not dict or value.get("EventName") != "CreateGrant":
        raise ValueError("CloudTrail CreateGrant event is malformed")
    raw = value.get("CloudTrailEvent")
    if type(raw) is not str:
        raise ValueError("CloudTrail CreateGrant payload is absent")
    try:
        event = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("CloudTrail CreateGrant payload is malformed") from exc
    if (
        type(event) is not dict
        or event.get("eventSource") != "kms.amazonaws.com"
        or event.get("eventName") != "CreateGrant"
        or event.get("awsRegion") != REGION
        or event.get("recipientAccountId") != ACCOUNT_ID
        or event.get("eventID") != value.get("EventId")
    ):
        raise ValueError("CloudTrail CreateGrant event identity drifted")
    _utc(event.get("eventTime"), "CloudTrail CreateGrant eventTime")
    return event


def _originating_resource(
    *,
    inputs: SupportInputs,
    grant: Mapping[str, object],
) -> tuple[str, str]:
    context = dict(grant["encryption_context"])
    candidates = tuple(
        value
        for key, value in context.items()
        if key not in {"RunId", "ActivationId"}
    )
    if len(candidates) != 1:
        raise ValueError("service grant resource correlation is ambiguous")
    physical = candidates[0]
    resources = inputs.support_deletion_inventory[
        "stack_resource_readback"
    ]["resources"]
    matches = tuple(
        row
        for row in resources
        if type(row) is dict
        and physical
        in {
            row.get("PhysicalResourceId"),
            row.get("ResourceArn"),
        }
    )
    if len(matches) != 1:
        raise ValueError("service grant resource correlation is absent")
    logical_id = matches[0].get("LogicalResourceId")
    return _text(logical_id, "originating resource"), physical


def _service_evidence(
    *,
    inputs: SupportInputs,
    grant: Mapping[str, object],
    events: Sequence[object],
    baseline_identity: str,
    active_scan_identity: str,
    settling_window_seconds: int,
) -> Mapping[str, object]:
    matches = []
    for raw_event in events:
        event = _cloudtrail_event(raw_event)
        request = event.get("requestParameters")
        response = event.get("responseElements")
        if (
            type(request) is dict
            and type(response) is dict
            and response.get("grantId") == grant["grant_id"]
        ):
            matches.append((raw_event, event, request, response))
    if len(matches) != 1:
        raise ValueError("unknown or unattributed active KMS grant")
    envelope, event, request, response = matches[0]
    request_projection = {
        "grant_id": response.get("grantId"),
        "name": request.get("name"),
        "grantee": request.get("granteePrincipal"),
        "retiring_principal": request.get("retiringPrincipal"),
        "operations": tuple(request.get("operations", ())),
        "encryption_context": tuple(
            sorted(
                request.get("constraints", {})
                .get("encryptionContextSubset", {})
                .items(),
                key=lambda item: _context_order(item[0]),
            )
        ),
        "revocable": True,
    }
    if (
        request.get("keyId") != inputs.retained_kms_key_arn
        or request_projection != grant
    ):
        raise ValueError("unknown or unattributed active KMS grant")
    originating_resource, originating_identity = _originating_resource(
        inputs=inputs, grant=grant
    )
    service = grant["grantee"]
    identity_context = canonical_sha256(
        {
            "originating_resource": originating_resource,
            "originating_identity": originating_identity,
            "encryption_context": grant["encryption_context"],
        }
    )
    return {
        **grant,
        "baseline_identity_sha256": baseline_identity,
        "diff_identity_sha256": canonical_sha256(
            {
                "baseline_identity_sha256": baseline_identity,
                "grant": grant,
            }
        ),
        "list_grants_identity_sha256": active_scan_identity,
        "cloudtrail_event_identity_sha256": canonical_sha256(envelope),
        "cloudtrail_request_identity_sha256": canonical_sha256(request),
        "cloudtrail_response_identity_sha256": canonical_sha256(response),
        "cloudtrail_event_time": event["eventTime"],
        "originating_resource": originating_resource,
        "originating_resource_identity": originating_identity,
        "originating_service": service,
        "encryption_context_correlation_identity_sha256": identity_context,
        "settling_window_seconds": settling_window_seconds,
    }


def build_activation_orphan_authority(
    *,
    baseline: Mapping[str, object],
    support_inputs: SupportInputs,
    direct_grant_evidence: Mapping[str, object],
    active_kms: object,
    cloudtrail_events: Sequence[object],
    observed_at: str,
    settling_window_seconds: int,
) -> Mapping[str, object]:
    """Build POSTCREATE creation authority; this never claims cleanup."""

    baseline, baseline_grants = _baseline_parts(baseline)
    if type(support_inputs) is not SupportInputs:
        raise ValueError("POSTCREATE requires exact SupportInputs")
    direct = _validate_identity(
        direct_grant_evidence, DIRECT_EVIDENCE_RECORD_TYPE
    )
    if (
        direct.get("activation_id") != support_inputs.activation_id
        or baseline.get("activation_id") != support_inputs.activation_id
        or direct.get("retained_kms_key_arn")
        != support_inputs.retained_kms_key_arn
        or baseline.get("retained_kms_key_arn")
        != support_inputs.retained_kms_key_arn
    ):
        raise ValueError("POSTCREATE activation coordinate drifted")
    if (
        type(settling_window_seconds) is not int
        or not 1 <= settling_window_seconds <= 900
    ):
        raise ValueError("settling window is outside the closed bound")
    observed = _utc(observed_at, "POSTCREATE observed_at")
    scan = _scan_kms_grants(
        kms=active_kms,
        key_arn=support_inputs.retained_kms_key_arn,
        observed_at=observed_at,
    )
    active = scan["grants"]
    direct_projection = direct.get("grant")
    if (
        type(direct_projection) is not dict
        or direct_projection not in active
    ):
        raise ValueError("direct grant evidence is absent from active inventory")
    baseline_ids = {item["grant_id"] for item in baseline_grants}
    active_by_id = {item["grant_id"]: item for item in active}
    if any(
        active_by_id.get(item["grant_id"]) != item
        for item in baseline_grants
    ):
        raise ValueError("retained KMS baseline drifted during POSTCREATE")
    service = []
    for grant in active:
        if (
            grant["grant_id"] in baseline_ids
            or grant["grant_id"] == direct["grant_id"]
        ):
            continue
        service.append(
            _service_evidence(
                inputs=support_inputs,
                grant=grant,
                events=cloudtrail_events,
                baseline_identity=baseline["canonical_body_sha256"],
                active_scan_identity=scan[
                    "list_grants_identity_sha256"
                ],
                settling_window_seconds=settling_window_seconds,
            )
        )
    attributable_ids = (
        baseline_ids
        | {direct["grant_id"]}
        | {item["grant_id"] for item in service}
    )
    if {item["grant_id"] for item in active} != attributable_ids:
        raise ValueError("unknown or unattributed active KMS grant")
    event_deadlines = [
        _utc(item["cloudtrail_event_time"], "service grant event time")
        + timedelta(seconds=settling_window_seconds)
        for item in service
    ]
    if event_deadlines and observed < max(event_deadlines):
        raise ValueError("POSTCREATE scan predates service settling deadline")
    settling_deadline = (
        max(event_deadlines).strftime(_UTC_FORMAT)
        if event_deadlines
        else observed_at
    )
    direct_creation = {
        **direct_projection,
        "provenance": direct["provenance"],
        "create_request_identity_sha256": (
            direct["create_request_identity_sha256"]
        ),
        "create_response_identity_sha256": (
            direct["create_response_identity_sha256"]
        ),
        "create_response_request_id": direct[
            "create_response_request_id"
        ],
        "reconciliation_list_grants_identity_sha256": direct[
            "reconciliation_list_grants_identity_sha256"
        ],
        "post_list_grants_identity_sha256": direct[
            "post_list_grants_identity_sha256"
        ],
    }
    body = {
        "record_type": ACTIVATION_RECORD_TYPE,
        "schema_version": 1,
        "phase": "POSTCREATE",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": support_inputs.activation_id,
        "retained_kms_key_arn": support_inputs.retained_kms_key_arn,
        "precreate_authority_identity_sha256": (
            baseline["canonical_body_sha256"]
        ),
        "predelete_physical_inventory": (
            support_inputs.support_deletion_inventory
        ),
        "predelete_physical_inventory_identity_sha256": canonical_sha256(
            support_inputs.support_deletion_inventory
        ),
        "expected_retained": baseline["expected_retained"],
        "baseline_grants": baseline_grants,
        "baseline_typed_grants": baseline["baseline_typed_grants"],
        "direct_grants": (direct_creation,),
        "service_grants": tuple(service),
        "active_grants": active,
        "active_grant_ids": [item["grant_id"] for item in active],
        "active_list_grants_identity_sha256": (
            scan["list_grants_identity_sha256"]
        ),
        "active_observed_at": observed_at,
        "settling_deadline": settling_deadline,
        "cloudtrail_event_count": len(cloudtrail_events),
        "cleanup_performed": False,
    }
    return _with_identity(body)


def _typed_from_mapping(value: Mapping[str, object]) -> KmsGrantIdentity:
    return _typed_grant(value)


def _cleanup_identity(
    *,
    request: Mapping[str, object],
    response: Optional[Mapping[str, object]],
    final_list_identity: str,
) -> str:
    return canonical_sha256(
        {
            "request": request,
            "response": response,
            "final_list_grants_identity_sha256": final_list_identity,
            "zero_retry": True,
        }
    )


def activation_authority_sort_key(activation_id: str) -> str:
    _text(activation_id, "activation ID")
    return f"ACTIVATION#{activation_id}#TASK12_ORPHAN_AUTHORITY"


def publish_activation_authority(
    *,
    s3: object,
    dynamodb: object,
    ledger_table_name: str,
    bucket: str,
    key: str,
    authority: Mapping[str, object],
    observed_at: str,
) -> Mapping[str, object]:
    """Publish one immutable S3 version and its conditional retained pointer."""

    exact = _validate_identity(authority, ACTIVATION_RECORD_TYPE)
    _text(bucket, "authority bucket")
    _text(key, "authority key")
    _utc(observed_at, "authority publication observed_at")
    payload = canonical_json_bytes(exact)
    versions = _exact_s3_authority_versions(
        s3=s3,
        bucket=bucket,
        key=key,
        payload=payload,
    )
    request_id: Optional[str] = None
    if len(versions) > 1:
        raise ValueError("multiple exact activation authority versions exist")
    if len(versions) == 1:
        version_id = versions[0]
    else:
        response: Optional[Mapping[str, object]]
        try:
            candidate = s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
                ChecksumAlgorithm="SHA256",
                Metadata={
                    "canonical-body-sha256": exact["canonical_body_sha256"],
                    "file-sha256": canonical_sha256(exact),
                },
            )
            if type(candidate) is not dict:
                raise ValueError("S3 PutObject response is malformed")
            response = candidate
        except Exception:
            response = None
        if response is not None:
            request_id = _response_request_id(response, "S3 PutObject")
            version_id = _text(
                response.get("VersionId"), "S3 object version"
            )
        else:
            versions = _exact_s3_authority_versions(
                s3=s3,
                bucket=bucket,
                key=key,
                payload=payload,
            )
            if len(versions) != 1:
                raise ValueError(
                    "ambiguous S3 authority write lacks singular readback"
                )
            version_id = versions[0]
    readback = s3.get_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
        ExpectedBucketOwner=ACCOUNT_ID,
    )
    _response_request_id(readback, "S3 GetObject")
    body = readback.get("Body")
    raw = body.read() if hasattr(body, "read") else None
    if (
        type(raw) is not bytes
        or raw != payload
        or readback.get("VersionId") != version_id
    ):
        raise ValueError("versioned activation authority readback drifted")
    pk = f"RUN#{RUN_ID}"
    sk = activation_authority_sort_key(exact["activation_id"])
    existing_coordinate = _ddb_read_coordinate(
        dynamodb=dynamodb,
        ledger_table_name=ledger_table_name,
        pk=pk,
        sk=sk,
    )
    if existing_coordinate is not None:
        if (
            existing_coordinate.get("bucket") != bucket
            or existing_coordinate.get("key") != key
            or existing_coordinate.get("version_id") != version_id
            or existing_coordinate.get("file_sha256")
            != canonical_sha256(exact)
            or existing_coordinate.get("authority_body_sha256")
            != exact["canonical_body_sha256"]
        ):
            raise ValueError("foreign activation authority coordinate exists")
        return _coordinate_result(existing_coordinate)
    coordinate_body = {
        "record_type": COORDINATE_RECORD_TYPE,
        "schema_version": 1,
        "phase": "POSTCREATE_PUBLISHED",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": exact["activation_id"],
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "file_sha256": canonical_sha256(exact),
        "authority_body_sha256": exact["canonical_body_sha256"],
        "publication_provenance": (
            "DIRECT_RESPONSE"
            if request_id is not None
            else "LIST_OBJECT_VERSIONS_RECONCILIATION"
        ),
        "put_object_request_id": request_id,
        "authority_observed_at": exact["active_observed_at"],
    }
    physical = _with_identity(
        {
            **coordinate_body,
            "PK": pk,
            "SK": sk,
        }
    )
    # Coordinate durability follows the same conditional-put/readback rule as
    # direct evidence. It never retries PutItem after an ambiguous outcome.
    adopted = _ddb_put_or_adopt_coordinate(
        dynamodb=dynamodb,
        ledger_table_name=ledger_table_name,
        coordinate=physical,
    )
    return _coordinate_result(adopted)


def _exact_s3_authority_versions(
    *,
    s3: object,
    bucket: str,
    key: str,
    payload: bytes,
) -> Tuple[str, ...]:
    key_marker: Optional[str] = None
    version_marker: Optional[str] = None
    seen = set()
    exact = []
    while True:
        request: dict[str, object] = {
            "Bucket": bucket,
            "Prefix": key,
            "MaxKeys": 1000,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
            request["VersionIdMarker"] = version_marker
        response = s3.list_object_versions(**request)
        _response_request_id(response, "S3 ListObjectVersions")
        versions = response.get("Versions")
        delete_markers = response.get("DeleteMarkers", [])
        if (
            type(versions) is not list
            or type(delete_markers) is not list
            or any(type(item) is not dict for item in versions + delete_markers)
            or type(response.get("IsTruncated")) is not bool
        ):
            raise ValueError("S3 authority version inventory is malformed")
        for item in versions:
            if item.get("Key") != key:
                continue
            version_id = _text(
                item.get("VersionId"), "S3 authority version ID"
            )
            readback = s3.get_object(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
                ExpectedBucketOwner=ACCOUNT_ID,
            )
            _response_request_id(readback, "S3 GetObject")
            body = readback.get("Body")
            raw = body.read() if hasattr(body, "read") else None
            if type(raw) is not bytes:
                raise ValueError("S3 authority version body is unreadable")
            if raw != payload:
                raise ValueError("foreign S3 authority version occupies key")
            exact.append(version_id)
        if not response["IsTruncated"]:
            break
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        marker = (next_key, next_version)
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or marker in seen
        ):
            raise ValueError("S3 authority version pagination is incomplete")
        seen.add(marker)
        key_marker, version_marker = marker
    return tuple(exact)


def _ddb_put_or_adopt_coordinate(
    *,
    dynamodb: object,
    ledger_table_name: str,
    coordinate: Mapping[str, object],
) -> Mapping[str, object]:
    try:
        response = dynamodb.put_item(
            TableName=ledger_table_name,
            Item=encode_item(coordinate),
            ConditionExpression=(
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            ),
            ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
            ReturnConsumedCapacity="NONE",
        )
        _response_request_id(response, "DynamoDB PutItem")
    except Exception:
        pass
    response = dynamodb.get_item(
        TableName=ledger_table_name,
        Key=encode_item(
            {"PK": coordinate["PK"], "SK": coordinate["SK"]}
        ),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _response_request_id(response, "DynamoDB GetItem")
    item = response.get("Item")
    if type(item) is not dict:
        raise ValueError("activation authority coordinate is not durable")
    adopted = decode_item(item)
    _validate_identity(adopted, COORDINATE_RECORD_TYPE)
    if adopted != coordinate:
        raise ValueError("foreign activation authority coordinate exists")
    return adopted


def _ddb_read_coordinate(
    *,
    dynamodb: object,
    ledger_table_name: str,
    pk: str,
    sk: str,
) -> Optional[Mapping[str, object]]:
    response = dynamodb.get_item(
        TableName=ledger_table_name,
        Key=encode_item({"PK": pk, "SK": sk}),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _response_request_id(response, "DynamoDB GetItem")
    item = response.get("Item")
    if item is None:
        return None
    if type(item) is not dict:
        raise ValueError("activation authority coordinate is malformed")
    value = decode_item(item)
    _validate_identity(value, COORDINATE_RECORD_TYPE)
    if value.get("PK") != pk or value.get("SK") != sk:
        raise ValueError("activation authority coordinate key drifted")
    return value


def _coordinate_result(
    physical: Mapping[str, object],
) -> Mapping[str, object]:
    body = {
        key: value
        for key, value in physical.items()
        if key not in {"PK", "SK", "canonical_body_sha256"}
    }
    result = dict(_with_identity(body))
    result["ledger_pk"] = physical["PK"]
    result["ledger_sk"] = physical["SK"]
    result["ledger_record_identity_sha256"] = physical[
        "canonical_body_sha256"
    ]
    return result


def read_activation_authority_coordinate(
    *,
    dynamodb: object,
    ledger_table_name: str,
    activation_id: str,
) -> Optional[Mapping[str, object]]:
    """Strongly read the immutable activation-authority pointer if present."""

    _text(ledger_table_name, "ledger table")
    _text(activation_id, "activation ID")
    value = _ddb_read_coordinate(
        dynamodb=dynamodb,
        ledger_table_name=ledger_table_name,
        pk=f"RUN#{RUN_ID}",
        sk=activation_authority_sort_key(activation_id),
    )
    return None if value is None else _coordinate_result(value)


def read_activation_authority(
    *,
    s3: object,
    dynamodb: object,
    ledger_table_name: str,
    activation_id: str,
) -> Mapping[str, object]:
    """Strongly read the pointer, then exact-read its authenticated S3 version."""

    pk = f"RUN#{RUN_ID}"
    sk = activation_authority_sort_key(activation_id)
    response = dynamodb.get_item(
        TableName=ledger_table_name,
        Key=encode_item({"PK": pk, "SK": sk}),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _response_request_id(response, "DynamoDB GetItem")
    if type(response.get("Item")) is not dict:
        raise ValueError("activation authority coordinate is absent")
    coordinate = decode_item(response["Item"])
    _validate_identity(coordinate, COORDINATE_RECORD_TYPE)
    if (
        coordinate.get("PK") != pk
        or coordinate.get("SK") != sk
        or coordinate.get("activation_id") != activation_id
    ):
        raise ValueError("activation authority coordinate is foreign")
    readback = s3.get_object(
        Bucket=coordinate["bucket"],
        Key=coordinate["key"],
        VersionId=coordinate["version_id"],
        ExpectedBucketOwner=ACCOUNT_ID,
    )
    _response_request_id(readback, "S3 GetObject")
    body = readback.get("Body")
    raw = body.read() if hasattr(body, "read") else None
    if (
        type(raw) is not bytes
        or readback.get("VersionId") != coordinate["version_id"]
    ):
        raise ValueError("activation authority version readback is malformed")
    try:
        authority = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("activation authority version is not canonical JSON") from exc
    if (
        type(authority) is not dict
        or canonical_json_bytes(authority) != raw
        or canonical_sha256(authority) != coordinate["file_sha256"]
        or authority.get("canonical_body_sha256")
        != coordinate["authority_body_sha256"]
        or authority.get("activation_id") != activation_id
    ):
        raise ValueError("activation authority version identity drifted")
    return _validate_identity(authority, ACTIVATION_RECORD_TYPE)


def cleanup_activation_orphans(
    *,
    authority: Mapping[str, object],
    kms: object,
    resource_inventory: ResourceInventoryScan,
    observed_at: str,
) -> Mapping[str, object]:
    """Guard, clean, prove baseline restoration, and return audit_orphans args."""

    value = _validate_identity(authority, ACTIVATION_RECORD_TYPE)
    current = _scan_kms_grants(
        kms=kms,
        key_arn=value["retained_kms_key_arn"],
        observed_at=observed_at,
    )
    active = value.get("active_grants")
    baseline = tuple(value.get("baseline_grants", ()))
    if (
        type(active) is not tuple
        or (
            current["grants"] != active
            and current["grants"] != baseline
        )
    ):
        raise ValueError(
            "current grant inventory does not equal activation authority"
        )
    already_restored = current["grants"] == baseline
    expected_retained = tuple(
        RetainedResource(**item) for item in value["expected_retained"]
    )
    if not isinstance(resource_inventory, ResourceInventoryScan):
        raise ValueError("resource inventory must be a typed exact scan")
    if (
        resource_inventory.covered_resource_types != AUDITED_RESOURCE_TYPES
        or resource_inventory.resources != expected_retained
        or resource_inventory.unreadable_resource_ids
        or resource_inventory.pagination_complete is not True
    ):
        raise ValueError("resource inventory is not the closed Task 12 scan")
    cleanup_rows = []
    attributed = tuple(value["direct_grants"]) + tuple(
        value["service_grants"]
    )
    for grant in attributed:
        request = {
            "KeyId": value["retained_kms_key_arn"],
            "GrantId": grant["grant_id"],
        }
        response: Optional[Mapping[str, object]]
        if already_restored:
            response = None
        else:
            try:
                candidate = kms.revoke_grant(**request)
                if type(candidate) is not dict:
                    raise ValueError("RevokeGrant response is malformed")
                _response_request_id(candidate, "RevokeGrant")
                response = candidate
            except Exception:
                # Lost response remains ambiguous until the single final scan.
                response = None
        cleanup_rows.append(
            {
                "grant_id": grant["grant_id"],
                "request": request,
                "response": response,
            }
        )
    final_scan = (
        current
        if already_restored
        else _scan_kms_grants(
            kms=kms,
            key_arn=value["retained_kms_key_arn"],
            observed_at=observed_at,
        )
    )
    if final_scan["grants"] != baseline:
        raise ValueError("final KMS grants do not equal frozen baseline")
    cleanup_by_id = {
        row["grant_id"]: _cleanup_identity(
            request=row["request"],
            response=row["response"],
            final_list_identity=final_scan[
                "list_grants_identity_sha256"
            ],
        )
        for row in cleanup_rows
    }
    direct_attributions = []
    for item in value["direct_grants"]:
        direct_attributions.append(
            DirectGrantAttribution(
                grant=_typed_from_mapping(item),
                provenance=item["provenance"],
                create_request_identity_sha256=item[
                    "create_request_identity_sha256"
                ],
                create_response_identity_sha256=item[
                    "create_response_identity_sha256"
                ],
                reconciliation_list_grants_identity_sha256=item[
                    "reconciliation_list_grants_identity_sha256"
                ],
                revoke_or_retire_identity_sha256=cleanup_by_id[
                    item["grant_id"]
                ],
            )
        )
    service_attributions = tuple(
        ServiceGrantAttribution(
            grant=_typed_from_mapping(item),
            baseline_diff_identity_sha256=item[
                "diff_identity_sha256"
            ],
            list_grants_identity_sha256=item[
                "list_grants_identity_sha256"
            ],
            cloudtrail_request_identity_sha256=item[
                "cloudtrail_request_identity_sha256"
            ],
            cloudtrail_response_identity_sha256=item[
                "cloudtrail_response_identity_sha256"
            ],
            originating_resource_identity=item[
                "originating_resource_identity"
            ],
            originating_service=item["originating_service"],
            encryption_context_correlation_identity_sha256=item[
                "encryption_context_correlation_identity_sha256"
            ],
            revoke_or_retire_identity_sha256=cleanup_by_id[
                item["grant_id"]
            ],
        )
        for item in value["service_grants"]
    )
    request = {
        "expected_retained": expected_retained,
        "inventory": resource_inventory,
        "retained_grant_baseline": tuple(
            _typed_from_mapping(item) for item in baseline
        ),
        "pre_cleanup_grants": KmsGrantScan(
            grants=tuple(
                _typed_from_mapping(item) for item in active
            ),
            pagination_complete=True,
            observed_at=value["active_observed_at"],
            list_grants_identity_sha256=value[
                "active_list_grants_identity_sha256"
            ],
        ),
        "final_grants": KmsGrantScan(
            grants=tuple(
                _typed_from_mapping(item) for item in final_scan["grants"]
            ),
            pagination_complete=True,
            observed_at=final_scan["observed_at"],
            list_grants_identity_sha256=final_scan[
                "list_grants_identity_sha256"
            ],
        ),
        "direct_grants": tuple(direct_attributions),
        "service_grants": service_attributions,
        "settling_deadline": value["settling_deadline"],
    }
    proof = audit_orphans(**request)
    return {
        "phase": "FINALIZE",
        "audit_orphans_request": request,
        "orphan_audit": proof,
        "cleanup_effects": tuple(cleanup_rows),
        "final_list_grants_identity_sha256": final_scan[
            "list_grants_identity_sha256"
        ],
    }


__all__ = [
    "ACTIVATION_RECORD_TYPE",
    "DIRECT_EVIDENCE_RECORD_TYPE",
    "COORDINATE_RECORD_TYPE",
    "PRECREATE_RECORD_TYPE",
    "build_activation_orphan_authority",
    "capture_precreate_baseline",
    "cleanup_activation_orphans",
    "create_or_adopt_direct_grant",
    "activation_authority_sort_key",
    "publish_activation_authority",
    "read_activation_authority_coordinate",
    "read_activation_authority",
    "read_direct_grant_evidence",
    "write_o_excl_authority",
]
