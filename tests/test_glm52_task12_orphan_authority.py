from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.dynamodb import decode_item


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
ACTIVATION_ID = "act-20260728-0001"
KMS_KEY_ARN = (
    "arn:aws:kms:us-west-2:246813579024:key/"
    "12345678-1234-4234-8234-1234567890ab"
)
LEDGER_TABLE = "keep-glm52-h1g-ledger-v1"
TLS_ROLE = (
    "arn:aws:iam::246813579024:"
    "role/keep-glm52-h1g-support-tls-handler"
)
DELETION_ROLE = (
    "arn:aws:iam::246813579024:"
    "role/keep-glm52-h1g-support-deletion"
)
REQUEST_ID = "11111111-2222-4333-8444-555555555555"
ROOT = Path(__file__).resolve().parent


def _metadata(request_id: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": request_id,
        "RetryAttempts": 0,
    }


def _grant(
    grant_id: str,
    *,
    name: str | None = None,
    grantee: str = "dynamodb.us-west-2.amazonaws.com",
    retiring_principal: str = (
        "arn:aws:iam::246813579024:role/keep-glm52-retained-kms-cleanup"
    ),
    operations: tuple[str, ...] = ("Decrypt", "Encrypt"),
    context: tuple[tuple[str, str], ...] = (
        ("Purpose", "retained-ledger"),
        ("RunId", RUN_ID),
    ),
) -> dict[str, object]:
    return {
        "GrantId": grant_id,
        "Name": name or "grant-" + grant_id,
        "GranteePrincipal": grantee,
        "RetiringPrincipal": retiring_principal,
        "Operations": list(operations),
        "Constraints": {
            "EncryptionContextSubset": dict(context),
        },
    }


def _direct_grant(grant_id: str = "direct-01") -> dict[str, object]:
    return _grant(
        grant_id,
        name=f"h1g-{ACTIVATION_ID}-secrets",
        grantee=TLS_ROLE,
        retiring_principal=DELETION_ROLE,
        operations=("Decrypt", "Encrypt", "GenerateDataKey"),
        context=(
            ("ActivationId", ACTIVATION_ID),
            ("RunId", RUN_ID),
        ),
    )


def _retained_resources() -> tuple[object, ...]:
    from glm52_enforcement.task12_orphan_audit import RetainedResource

    return (
        RetainedResource("LEDGER", LEDGER_TABLE, "DDB_RETAINED"),
        RetainedResource("KMS_KEY", KMS_KEY_ARN, "KMS_RETAINED"),
        RetainedResource(
            "EVIDENCE_BUCKET",
            "keep-glm52-model-evidence-fixture",
            "S3_RETAINED",
        ),
        RetainedResource(
            "PRODUCTION_FENCE_STACK",
            (
                "arn:aws:cloudformation:us-west-2:246813579024:"
                "stack/keep-glm52-h1g-fence/"
                "22222222-3333-4444-8555-666666666666"
            ),
            "CFN_RETAINED",
        ),
        RetainedResource(
            "LIFECYCLE_RESOURCE",
            "keep-glm52-h1g-retained-lifecycle",
            "STEP_FUNCTIONS_RETAINED",
        ),
        RetainedResource(
            "SOURCE_PUBLISHER_IDENTITY",
            (
                "arn:aws:iam::246813579024:"
                "role/keep-glm52-h1g-source-publisher"
            ),
            "IAM_RETAINED",
        ),
    )


class _PagedKms:
    def __init__(
        self,
        pages: list[dict[str, object]],
        *,
        create_response: object | None = None,
    ) -> None:
        self.pages = list(pages)
        self.create_response = create_response
        self.list_calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []
        self.revoke_calls: list[dict[str, object]] = []

    def list_grants(self, **request: object) -> dict[str, object]:
        self.list_calls.append(dict(request))
        if not self.pages:
            raise AssertionError("unexpected ListGrants call")
        return self.pages.pop(0)

    def create_grant(self, **request: object) -> dict[str, object]:
        self.create_calls.append(dict(request))
        if isinstance(self.create_response, Exception):
            raise self.create_response
        if type(self.create_response) is not dict:
            raise AssertionError("unexpected CreateGrant call")
        return dict(self.create_response)

    def revoke_grant(self, **request: object) -> dict[str, object]:
        self.revoke_calls.append(dict(request))
        return {"ResponseMetadata": _metadata("revoke-" + request["GrantId"])}


def _page(
    grants: list[dict[str, object]],
    *,
    request_id: str,
    truncated: bool = False,
    marker: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "Grants": grants,
        "Truncated": truncated,
        "ResponseMetadata": _metadata(request_id),
    }
    if marker is not None:
        result["NextMarker"] = marker
    return result


class _Dynamo:
    def __init__(self, *, lose_put_response: bool = False) -> None:
        self.items: dict[tuple[str, str], dict[str, object]] = {}
        self.put_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self.lose_put_response = lose_put_response

    def get_item(self, **request: object) -> dict[str, object]:
        self.get_calls.append(dict(request))
        key = decode_item(request["Key"])
        item = self.items.get((key["PK"], key["SK"]))
        result: dict[str, object] = {
            "ResponseMetadata": _metadata("ddb-get"),
        }
        if item is not None:
            result["Item"] = deepcopy(item)
        return result

    def put_item(self, **request: object) -> dict[str, object]:
        self.put_calls.append(dict(request))
        item = deepcopy(request["Item"])
        key = decode_item({"PK": item["PK"], "SK": item["SK"]})
        coordinate = (key["PK"], key["SK"])
        if coordinate in self.items:
            raise RuntimeError("ConditionalCheckFailedException")
        self.items[coordinate] = item
        if self.lose_put_response:
            self.lose_put_response = False
            raise TimeoutError("lost PutItem success response")
        return {"ResponseMetadata": _metadata("ddb-put")}


class _S3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str, str], bytes] = {}
        self.put_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []

    def put_object(self, **request: object) -> dict[str, object]:
        self.put_calls.append(dict(request))
        version = "version-0001"
        self.objects[
            (str(request["Bucket"]), str(request["Key"]), version)
        ] = bytes(request["Body"])
        return {
            "VersionId": version,
            "ResponseMetadata": _metadata("s3-put"),
        }

    def list_object_versions(self, **request: object) -> dict[str, object]:
        versions = [
            {
                "Key": key,
                "VersionId": version,
            }
            for (bucket, key, version), _raw in self.objects.items()
            if bucket == request["Bucket"]
            and key.startswith(str(request["Prefix"]))
        ]
        return {
            "Versions": versions,
            "DeleteMarkers": [],
            "IsTruncated": False,
            "ResponseMetadata": _metadata("s3-list"),
        }

    def get_object(self, **request: object) -> dict[str, object]:
        self.get_calls.append(dict(request))
        version = str(request["VersionId"])
        raw = self.objects[
            (str(request["Bucket"]), str(request["Key"]), version)
        ]
        return {
            "VersionId": version,
            "Body": BytesIO(raw),
            "ResponseMetadata": _metadata("s3-get"),
        }


def test_precreate_baseline_is_complete_sorted_and_o_excl(
    tmp_path: Path,
) -> None:
    """Break caught: a partial/default baseline can replace pre-support truth."""

    from glm52_enforcement.task12_orphan_authority import (
        capture_precreate_baseline,
        write_o_excl_authority,
    )

    kms = _PagedKms(
        [
            _page(
                [_grant("z-retained")],
                request_id="list-1",
                truncated=True,
                marker="page-2",
            ),
            _page([_grant("a-retained")], request_id="list-2"),
        ]
    )
    baseline = capture_precreate_baseline(
        kms=kms,
        activation_id=ACTIVATION_ID,
        retained_kms_key_arn=KMS_KEY_ARN,
        expected_retained=_retained_resources(),
        observed_at="2026-07-28T00:00:00Z",
    )

    assert [grant["grant_id"] for grant in baseline["baseline_grants"]] == [
        "a-retained",
        "z-retained",
    ]
    assert baseline["pagination_complete"] is True
    assert baseline["captured_before_support_mutation"] is True
    assert kms.list_calls == [
        {"KeyId": KMS_KEY_ARN, "Limit": 100},
        {"KeyId": KMS_KEY_ARN, "Limit": 100, "Marker": "page-2"},
    ]
    output = tmp_path / "PRECREATE_ORPHAN_BASELINE.json"
    first = write_o_excl_authority(path=output, value=baseline)
    second = write_o_excl_authority(path=output, value=baseline)
    assert first["outcome"] == "CREATED"
    assert second["outcome"] == "RECONCILED_IDENTICAL"
    assert output.read_bytes() == canonical_json_bytes(baseline)

    mutant = dict(baseline)
    mutant["observed_at"] = "2026-07-28T00:00:01Z"
    with pytest.raises(ValueError, match="foreign existing"):
        write_o_excl_authority(path=output, value=mutant)


def test_precreate_rejects_incomplete_pagination_and_retained_inventory() -> None:
    """Break caught: a truncated ListGrants page or missing retained family passes."""

    from glm52_enforcement.task12_orphan_authority import (
        capture_precreate_baseline,
    )

    kms = _PagedKms(
        [
            _page(
                [],
                request_id="list-truncated",
                truncated=True,
            )
        ]
    )
    with pytest.raises(ValueError, match="pagination"):
        capture_precreate_baseline(
            kms=kms,
            activation_id=ACTIVATION_ID,
            retained_kms_key_arn=KMS_KEY_ARN,
            expected_retained=_retained_resources(),
            observed_at="2026-07-28T00:00:00Z",
        )
    with pytest.raises(ValueError, match="retained inventory"):
        capture_precreate_baseline(
            kms=_PagedKms([_page([], request_id="list")]),
            activation_id=ACTIVATION_ID,
            retained_kms_key_arn=KMS_KEY_ARN,
            expected_retained=_retained_resources()[:-1],
            observed_at="2026-07-28T00:00:00Z",
        )


def test_direct_create_grant_persists_full_evidence_and_adopts_lost_put() -> None:
    """Break caught: CreateGrant succeeds but no durable request/response authority exists."""

    from glm52_enforcement.task12_orphan_authority import (
        create_or_adopt_direct_grant,
    )

    kms = _PagedKms(
        [
            _page([], request_id="pre-list"),
            _page([_direct_grant()], request_id="post-list"),
        ],
        create_response={
            "GrantId": "direct-01",
            "GrantToken": "grant-token-direct-01",
            "ResponseMetadata": _metadata("create-direct"),
        },
    )
    dynamodb = _Dynamo(lose_put_response=True)

    evidence = create_or_adopt_direct_grant(
        kms=kms,
        dynamodb=dynamodb,
        ledger_table_name=LEDGER_TABLE,
        activation_id=ACTIVATION_ID,
        custom_resource_request_id=REQUEST_ID,
        retained_kms_key_arn=KMS_KEY_ARN,
        observed_at="2026-07-28T00:01:00Z",
    )

    assert evidence["provenance"] == "DIRECT_RESPONSE"
    assert evidence["grant_id"] == "direct-01"
    assert evidence["grant_token"] == "grant-token-direct-01"
    assert evidence["create_response_request_id"] == "create-direct"
    assert evidence["post_list_grants_identity_sha256"]
    assert len(dynamodb.put_calls) == 1
    assert len(dynamodb.get_calls) >= 2
    assert kms.create_calls == [
        {
            "KeyId": KMS_KEY_ARN,
            "GranteePrincipal": TLS_ROLE,
            "RetiringPrincipal": DELETION_ROLE,
            "Operations": ["Decrypt", "Encrypt", "GenerateDataKey"],
            "Constraints": {
                "EncryptionContextSubset": {
                    "RunId": RUN_ID,
                    "ActivationId": ACTIVATION_ID,
                }
            },
            "Name": f"h1g-{ACTIVATION_ID}-secrets",
        }
    ]

    replay_kms = _PagedKms(
        [_page([_direct_grant()], request_id="replay-list")]
    )
    replay = create_or_adopt_direct_grant(
        kms=replay_kms,
        dynamodb=dynamodb,
        ledger_table_name=LEDGER_TABLE,
        activation_id=ACTIVATION_ID,
        custom_resource_request_id=REQUEST_ID,
        retained_kms_key_arn=KMS_KEY_ARN,
        observed_at="2026-07-28T00:02:00Z",
    )
    assert replay == evidence
    assert replay_kms.create_calls == []
    assert len(dynamodb.put_calls) == 1


def test_direct_grant_reconciliation_never_invents_grant_token() -> None:
    """Break caught: ListGrants adoption fabricates unavailable CreateGrant response fields."""

    from glm52_enforcement.task12_orphan_authority import (
        create_or_adopt_direct_grant,
    )

    kms = _PagedKms(
        [_page([_direct_grant()], request_id="reconcile-list")]
    )
    evidence = create_or_adopt_direct_grant(
        kms=kms,
        dynamodb=_Dynamo(),
        ledger_table_name=LEDGER_TABLE,
        activation_id=ACTIVATION_ID,
        custom_resource_request_id=REQUEST_ID,
        retained_kms_key_arn=KMS_KEY_ARN,
        observed_at="2026-07-28T00:01:00Z",
    )
    assert evidence["provenance"] == "LIST_GRANTS_RECONCILIATION"
    assert evidence["grant_token"] is None
    assert evidence["create_response_identity_sha256"] is None
    assert evidence["create_response_request_id"] is None
    assert evidence["reconciliation_list_grants_identity_sha256"]
    assert kms.create_calls == []


@pytest.mark.parametrize(
    "grants",
    [
        (),
        (_direct_grant("direct-01"), _direct_grant("direct-02")),
        (
            _grant(
                "foreign",
                name=f"h1g-{ACTIVATION_ID}-secrets",
                grantee="arn:aws:iam::246813579024:role/foreign",
                retiring_principal=DELETION_ROLE,
                operations=("Decrypt", "Encrypt", "GenerateDataKey"),
                context=(
                    ("ActivationId", ACTIVATION_ID),
                    ("RunId", RUN_ID),
                ),
            ),
        ),
    ],
)
def test_ambiguous_direct_grant_readback_fails_without_second_create(
    grants: tuple[dict[str, object], ...],
) -> None:
    """Break caught: zero, duplicate, or foreign post-create readback retries CreateGrant."""

    from glm52_enforcement.task12_orphan_authority import (
        create_or_adopt_direct_grant,
    )

    kms = _PagedKms(
        [
            _page([], request_id="pre-list"),
            _page(list(grants), request_id="post-list"),
        ],
        create_response={
            "GrantId": "direct-01",
            "GrantToken": "grant-token-direct-01",
            "ResponseMetadata": _metadata("create-direct"),
        },
    )
    with pytest.raises(ValueError, match="singular exact"):
        create_or_adopt_direct_grant(
            kms=kms,
            dynamodb=_Dynamo(),
            ledger_table_name=LEDGER_TABLE,
            activation_id=ACTIVATION_ID,
            custom_resource_request_id=REQUEST_ID,
            retained_kms_key_arn=KMS_KEY_ARN,
            observed_at="2026-07-28T00:01:00Z",
        )
    assert len(kms.create_calls) == 1


def _support_inputs() -> object:
    spec = importlib.util.spec_from_file_location(
        "_orphan_support_fixtures",
        ROOT / "test_glm52_enforcement_support_plane.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from glm52_enforcement.support_plane import support_inputs_from_mapping

    return support_inputs_from_mapping(module._support_inputs())


def _baseline_value() -> dict[str, object]:
    from glm52_enforcement.task12_orphan_authority import (
        capture_precreate_baseline,
    )

    return capture_precreate_baseline(
        kms=_PagedKms([_page([_grant("baseline")], request_id="baseline")]),
        activation_id=ACTIVATION_ID,
        retained_kms_key_arn=KMS_KEY_ARN,
        expected_retained=_retained_resources(),
        observed_at="2026-07-28T00:00:00Z",
    )


def _reconciled_direct_evidence() -> dict[str, object]:
    from glm52_enforcement.task12_orphan_authority import (
        create_or_adopt_direct_grant,
    )

    return create_or_adopt_direct_grant(
        kms=_PagedKms(
            [_page([_direct_grant()], request_id="direct-reconcile")]
        ),
        dynamodb=_Dynamo(),
        ledger_table_name=LEDGER_TABLE,
        activation_id=ACTIVATION_ID,
        custom_resource_request_id=REQUEST_ID,
        retained_kms_key_arn=KMS_KEY_ARN,
        observed_at="2026-07-28T00:01:00Z",
    )


def _cloudtrail_service_event(
    *,
    grant_id: str,
    volume_id: str,
) -> dict[str, object]:
    request = {
        "keyId": KMS_KEY_ARN,
        "granteePrincipal": "ec2.us-west-2.amazonaws.com",
        "retiringPrincipal": DELETION_ROLE,
        "operations": [
            "Decrypt",
            "Encrypt",
            "GenerateDataKeyWithoutPlaintext",
        ],
        "name": f"h1g-{ACTIVATION_ID}-ebs-data",
        "constraints": {
            "encryptionContextSubset": {
                "RunId": RUN_ID,
                "ActivationId": ACTIVATION_ID,
                "aws:ebs:id": volume_id,
            }
        },
    }
    response = {"grantId": grant_id}
    event = {
        "eventVersion": "1.10",
        "userIdentity": {
            "type": "AWSService",
            "invokedBy": "ec2.us-west-2.amazonaws.com",
        },
        "eventTime": "2026-07-28T00:02:00Z",
        "eventSource": "kms.amazonaws.com",
        "eventName": "CreateGrant",
        "awsRegion": REGION,
        "requestParameters": request,
        "responseElements": response,
        "requestID": "service-create-request",
        "eventID": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        "readOnly": False,
        "eventType": "AwsApiCall",
        "managementEvent": True,
        "recipientAccountId": ACCOUNT_ID,
        "eventCategory": "Management",
    }
    return {
        "EventId": event["eventID"],
        "EventName": "CreateGrant",
        "EventSource": "kms.amazonaws.com",
        "CloudTrailEvent": json.dumps(event, separators=(",", ":")),
    }


def test_postcreate_unknown_grant_blocks_authority_publication() -> None:
    """Break caught: an active grant without direct or CloudTrail provenance is published."""

    from glm52_enforcement.task12_orphan_authority import (
        build_activation_orphan_authority,
    )

    baseline = _baseline_value()
    active = [
        _grant("baseline"),
        _direct_grant(),
        _grant(
            "unknown",
            name=f"h1g-{ACTIVATION_ID}-unknown",
            grantee="ec2.us-west-2.amazonaws.com",
            retiring_principal=DELETION_ROLE,
            operations=(
                "Decrypt",
                "Encrypt",
                "GenerateDataKeyWithoutPlaintext",
            ),
            context=(
                ("ActivationId", ACTIVATION_ID),
                ("RunId", RUN_ID),
                ("aws:ebs:id", "vol-00000000000000002"),
            ),
        ),
    ]
    with pytest.raises(ValueError, match="unknown or unattributed"):
        build_activation_orphan_authority(
            baseline=baseline,
            support_inputs=_support_inputs(),
            direct_grant_evidence=_reconciled_direct_evidence(),
            active_kms=_PagedKms(
                [_page(active, request_id="active-list")]
            ),
            cloudtrail_events=(),
            observed_at="2026-07-28T00:10:00Z",
            settling_window_seconds=300,
        )


def test_postcreate_binds_service_cloudtrail_and_emits_no_cleanup_identity() -> None:
    """Break caught: service attribution is grammar-only or precomputes revocation."""

    from glm52_enforcement.task12_orphan_authority import (
        build_activation_orphan_authority,
    )

    support_inputs = _support_inputs()
    volume_id = next(
        row["PhysicalResourceId"]
        for row in support_inputs.support_deletion_inventory[
            "stack_resource_readback"
        ]["resources"]
        if row["LogicalResourceId"] == "CombinedHostDataVolume"
    )
    service_grant = _grant(
        "service-data",
        name=f"h1g-{ACTIVATION_ID}-ebs-data",
        grantee="ec2.us-west-2.amazonaws.com",
        retiring_principal=DELETION_ROLE,
        operations=(
            "Decrypt",
            "Encrypt",
            "GenerateDataKeyWithoutPlaintext",
        ),
        context=(
            ("ActivationId", ACTIVATION_ID),
            ("RunId", RUN_ID),
            ("aws:ebs:id", volume_id),
        ),
    )
    authority = build_activation_orphan_authority(
        baseline=_baseline_value(),
        support_inputs=support_inputs,
        direct_grant_evidence=_reconciled_direct_evidence(),
        active_kms=_PagedKms(
            [
                _page(
                    [_grant("baseline"), _direct_grant(), service_grant],
                    request_id="active-list",
                )
            ]
        ),
        cloudtrail_events=(
            _cloudtrail_service_event(
                grant_id="service-data",
                volume_id=volume_id,
            ),
        ),
        observed_at="2026-07-28T00:10:00Z",
        settling_window_seconds=300,
    )

    assert authority["active_grant_ids"] == [
        "baseline",
        "direct-01",
        "service-data",
    ]
    assert authority["service_grants"][0]["originating_resource"] == (
        "CombinedHostDataVolume"
    )
    assert "revoke_or_retire_identity_sha256" not in canonical_json_bytes(
        authority
    ).decode("ascii")


def test_postcreate_publishes_and_task12_exact_reads_one_version() -> None:
    """Break caught: Task 12 consumes an unversioned or caller-supplied body."""

    from glm52_enforcement.task12_orphan_authority import (
        build_activation_orphan_authority,
        publish_activation_authority,
        read_activation_authority,
    )

    authority = build_activation_orphan_authority(
        baseline=_baseline_value(),
        support_inputs=_support_inputs(),
        direct_grant_evidence=_reconciled_direct_evidence(),
        active_kms=_PagedKms(
            [
                _page(
                    [_grant("baseline"), _direct_grant()],
                    request_id="active-list",
                )
            ]
        ),
        cloudtrail_events=(),
        observed_at="2026-07-28T00:10:00Z",
        settling_window_seconds=300,
    )
    s3 = _S3()
    dynamodb = _Dynamo()
    coordinate = publish_activation_authority(
        s3=s3,
        dynamodb=dynamodb,
        ledger_table_name=LEDGER_TABLE,
        bucket="keep-glm52-model-evidence-fixture",
        key=(
            "campaigns/glm52-sky-20260724/task12/orphans/"
            f"{ACTIVATION_ID}/{authority['canonical_body_sha256']}.json"
        ),
        authority=authority,
        observed_at="2026-07-28T00:11:00Z",
    )
    assert coordinate["version_id"] == "version-0001"
    assert coordinate["publication_provenance"] == "DIRECT_RESPONSE"
    assert coordinate["put_object_request_id"] == "s3-put"
    replay_coordinate = publish_activation_authority(
        s3=s3,
        dynamodb=dynamodb,
        ledger_table_name=LEDGER_TABLE,
        bucket="keep-glm52-model-evidence-fixture",
        key=(
            "campaigns/glm52-sky-20260724/task12/orphans/"
            f"{ACTIVATION_ID}/{authority['canonical_body_sha256']}.json"
        ),
        authority=authority,
        observed_at="2026-07-28T00:11:00Z",
    )
    assert replay_coordinate == coordinate
    assert len(s3.put_calls) == 1
    loaded = read_activation_authority(
        s3=s3,
        dynamodb=dynamodb,
        ledger_table_name=LEDGER_TABLE,
        activation_id=ACTIVATION_ID,
    )
    assert canonical_json_bytes(loaded) == canonical_json_bytes(authority)
    assert s3.get_calls[-1]["VersionId"] == "version-0001"


def test_runtime_unknown_grant_causes_zero_cleanup_effects() -> None:
    """Break caught: runtime revokes known grants before discovering an unknown grant."""

    from glm52_enforcement.task12_orphan_authority import (
        cleanup_activation_orphans,
    )

    authority = {
        **_baseline_value(),
        "record_type": "glm52_task12_activation_orphan_authority_v1",
        "direct_grants": [],
        "service_grants": [],
        "active_grants": [],
    }
    authority_body = dict(authority)
    authority_body.pop("canonical_body_sha256")
    authority["canonical_body_sha256"] = canonical_sha256(authority_body)
    kms = _PagedKms(
        [
            _page(
                [_grant("baseline"), _grant("unknown")],
                request_id="runtime-list",
            )
        ]
    )
    with pytest.raises(ValueError, match="current grant inventory"):
        cleanup_activation_orphans(
            authority=authority,
            kms=kms,
            resource_inventory=SimpleNamespace(),
            observed_at="2026-07-28T00:20:00Z",
        )
    assert kms.revoke_calls == []


def test_runtime_cleanup_restores_baseline_and_replay_adopts_without_revoke() -> None:
    """Break caught: lost runtime response causes a second grant mutation."""

    from glm52_enforcement.task12_orphan_audit import (
        AUDITED_RESOURCE_TYPES,
        ResourceInventoryScan,
        RetainedResource,
    )
    from glm52_enforcement.task12_orphan_authority import (
        build_activation_orphan_authority,
        cleanup_activation_orphans,
    )

    authority = build_activation_orphan_authority(
        baseline=_baseline_value(),
        support_inputs=_support_inputs(),
        direct_grant_evidence=_reconciled_direct_evidence(),
        active_kms=_PagedKms(
            [
                _page(
                    [_grant("baseline"), _direct_grant()],
                    request_id="active-list",
                )
            ]
        ),
        cloudtrail_events=(),
        observed_at="2026-07-28T00:10:00Z",
        settling_window_seconds=300,
    )
    expected = tuple(
        RetainedResource(**item)
        for item in authority["expected_retained"]
    )
    inventory = ResourceInventoryScan(
        covered_resource_types=AUDITED_RESOURCE_TYPES,
        resources=expected,
        unreadable_resource_ids=(),
        pagination_complete=True,
        observed_at="2026-07-28T00:20:00Z",
        evidence_identity_sha256="f" * 64,
    )
    first_kms = _PagedKms(
        [
            _page(
                [_grant("baseline"), _direct_grant()],
                request_id="cleanup-pre",
            ),
            _page([_grant("baseline")], request_id="cleanup-final"),
        ]
    )
    first = cleanup_activation_orphans(
        authority=authority,
        kms=first_kms,
        resource_inventory=inventory,
        observed_at="2026-07-28T00:20:00Z",
    )
    assert first["orphan_audit"].final_grants[0].grant_id == "baseline"
    assert [call["GrantId"] for call in first_kms.revoke_calls] == [
        "direct-01"
    ]

    replay_kms = _PagedKms(
        [_page([_grant("baseline")], request_id="replay-final")]
    )
    replay = cleanup_activation_orphans(
        authority=authority,
        kms=replay_kms,
        resource_inventory=inventory,
        observed_at="2026-07-28T00:21:00Z",
    )
    assert replay["orphan_audit"].final_grants[0].grant_id == "baseline"
    assert replay_kms.revoke_calls == []


def test_postcreate_cli_derives_content_addressed_key_internally(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Break caught: the caller must guess a hash only the live scan computes."""

    script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/materialize_task12_orphan_authority.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task12_orphan_cli_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    baseline_path = Path("/tmp/precreate.json")
    inputs_path = Path("/tmp/support-inputs.json")
    manifest_path = Path("/tmp/postcreate.json")
    baseline = {
        "expected_retained": [
            {
                "resource_type": "EVIDENCE_BUCKET",
                "resource_id": "keep-glm52-models",
                "cost_class": "S3_RETAINED",
            }
        ]
    }
    support = SimpleNamespace(
        activation_id=ACTIVATION_ID,
        model_bucket_name="keep-glm52-models",
        ledger_table_name=LEDGER_TABLE,
        run_id=RUN_ID,
    )
    manifest = {
        "record_type": "glm52_h1g_support_postcreate_manifest_v1",
        "activation_id": ACTIVATION_ID,
        "direct_grant_evidence_coordinate": (
            f"ACTIVATION#{ACTIVATION_ID}"
            "#CUSTOM_RESOURCE_GRANT#request-1"
        ),
        "direct_grant_evidence_sha256": "d" * 64,
    }
    manifest["canonical_body_sha256"] = canonical_sha256(manifest)
    documents = {
        baseline_path: baseline,
        inputs_path: {"record_type": "support-inputs"},
        manifest_path: manifest,
    }
    monkeypatch.setattr(module, "_canonical", lambda path: documents[path])
    monkeypatch.setattr(
        module,
        "support_inputs_from_mapping",
        lambda value: support,
    )
    monkeypatch.setattr(
        module,
        "read_activation_authority_coordinate",
        lambda **request: None,
    )
    monkeypatch.setattr(
        module,
        "read_direct_grant_evidence",
        lambda **request: {"canonical_body_sha256": "d" * 64},
    )
    monkeypatch.setattr(module, "_cloudtrail_events", lambda **request: ())
    monkeypatch.setattr(
        module,
        "build_activation_orphan_authority",
        lambda **request: {"canonical_body_sha256": "a" * 64},
    )
    published: dict[str, object] = {}

    def publish(**request: object) -> dict[str, object]:
        published.update(request)
        return {
            "bucket": request["bucket"],
            "key": request["key"],
            "version_id": "version-1",
        }

    monkeypatch.setattr(module, "publish_activation_authority", publish)
    args = SimpleNamespace(
        activation_id=ACTIVATION_ID,
        baseline=baseline_path,
        support_inputs=inputs_path,
        postcreate_manifest=manifest_path,
        cloudtrail_start="2026-07-29T00:00:00Z",
        cloudtrail_end="2026-07-29T00:01:00Z",
        settling_window_seconds=60,
        authority_bucket="keep-glm52-models",
    )
    module._postcreate(
        args,
        {
            "dynamodb": object(),
            "cloudtrail": object(),
            "kms": object(),
            "s3": object(),
        },
    )
    expected_key = (
        f"campaigns/{RUN_ID}/task12/orphans/{ACTIVATION_ID}/"
        + "a" * 64
        + ".json"
    )
    assert published["key"] == expected_key
    assert "--authority-key" not in module._parser().format_help()
    assert json.loads(capsys.readouterr().out)["key"] == expected_key


def test_precreate_cli_requires_reviewed_change_set_before_execution() -> None:
    """Break caught: PRECREATE runs after execution or before review exists."""

    script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/materialize_task12_orphan_authority.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task12_orphan_precreate_cli_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stack_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:stack/"
        "keep-glm52-h1g-support/"
        "11111111-2222-4333-8444-555555555555"
    )

    class CloudFormation:
        def __init__(self, execution_status: str) -> None:
            self.execution_status = execution_status

        def describe_change_set(self, **request: object) -> object:
            return {
                "Status": "CREATE_COMPLETE",
                "ExecutionStatus": self.execution_status,
                "ChangeSetName": "reviewed-support",
                "StackId": stack_id,
                "ResponseMetadata": _metadata("describe-change-set"),
            }

    module._change_set_ready_and_unexecuted(
        cloudformation=CloudFormation("AVAILABLE"),
        stack_name=stack_id,
        change_set_name="reviewed-support",
    )
    with pytest.raises(ValueError, match="before execution"):
        module._change_set_ready_and_unexecuted(
            cloudformation=CloudFormation("EXECUTE_COMPLETE"),
            stack_name=stack_id,
            change_set_name="reviewed-support",
        )
