"""Task 11 real Lambda entrypoints and deterministic support package."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import ModuleType, SimpleNamespace
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
import zipfile

import pytest

from glm52_enforcement import support_budget_gate_handler
from glm52_enforcement import support_decision_handler
from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.decision_closure import (
    CLOSURE_PHASE_CEILINGS,
    SUFFIX_PHASE_CEILINGS,
    PhaseSpan,
    build_deployed_gate_document,
    build_rehearsal_measurement,
)
from glm52_enforcement.support_budget_gate_handler import BudgetGateConfig
from glm52_enforcement.task11_boundary import (
    BOUNDARY_INPUT_KINDS,
    build_task11_boundary_document,
    build_task11_input_coordinate,
)
from glm52_enforcement.dynamodb import encode_item
from glm52_enforcement.task11_production import (
    Task11ReserveReader,
    Task11SpendEc2,
    Task11SpendObjectStore,
)


ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_ID = "246813579024"
RUN_ID = "glm52-sky-20260724"
ACTIVATION_ID = "approved-20260728"
BUCKET = "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
KEY = "rehearsal/gates/approved-20260728/CLOSURE_BUDGET.json"
VERSION_ID = "gate-version-0001"
DEPLOYMENT_SHA = hashlib.sha256(b"deployed-support-version").hexdigest()
PATH_PROOFS = (
    "EXACT_PRODUCTION_CLIENTS",
    "FULL_VERSION_PAGINATION",
    "NON_VPC_DECISION_PATH",
    "ISOLATED_ADMISSION_ATTESTATION_PATH",
    "HOST_NAT_PATH",
    "FROZEN_NAMESPACE_SCALE",
    "CONDITIONAL_WRITE",
    "DIRECT_RESPONSE",
    "EXACT_GET_HEAD",
    "H1E_MODELED_VALIDATION",
)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_content_length",
        "missing_checksum",
        "wrong_checksum",
        "missing_etag",
        "missing_request_id",
        "missing_metadata",
        "nonzero_retry",
        "extra_response_field",
    ),
)
def test_task11_spend_object_rejects_unauthenticated_get_mutants(
    mutation: str,
) -> None:
    raw = b'{"spend":"evidence"}'
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    response = {
        "Body": io.BytesIO(raw),
        "VersionId": "spend-version-1",
        "ContentLength": len(raw),
        "ChecksumSHA256": checksum,
        "ETag": '"service-etag"',
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "RequestId": "spend-get-1",
            "RetryAttempts": 0,
        },
    }
    if mutation == "wrong_checksum":
        response["ChecksumSHA256"] = base64.b64encode(b"x" * 32).decode(
            "ascii"
        )
    elif mutation == "missing_request_id":
        response["ResponseMetadata"].pop("RequestId")
    elif mutation == "missing_metadata":
        response.pop("ResponseMetadata")
    elif mutation == "nonzero_retry":
        response["ResponseMetadata"]["RetryAttempts"] = 1
    elif mutation == "extra_response_field":
        response["Foreign"] = "field"
    else:
        response.pop(
            {
                "missing_content_length": "ContentLength",
                "missing_checksum": "ChecksumSHA256",
                "missing_etag": "ETag",
            }[mutation]
        )

    class S3:
        def get_object(self, **request: object):
            assert request == {
                "Bucket": "campaign-bucket",
                "Key": "spend/key.json",
                "VersionId": "spend-version-1",
                "ExpectedBucketOwner": ACCOUNT_ID,
                "ChecksumMode": "ENABLED",
            }
            return response

    store = Task11SpendObjectStore(
        s3=S3(),
        bucket="campaign-bucket",
        version_by_key={"spend/key.json": "spend-version-1"},
    )
    with pytest.raises(RuntimeError):
        store.get_object(key="spend/key.json")


@pytest.mark.parametrize(
    "mutation",
    (
        "literal_is_truncated",
        "missing_next_marker",
        "marker_cycle",
        "marker_echo",
        "hidden_delete_marker",
        "foreign_version",
        "duplicate_version",
        "extra_page_field",
        "missing_request_id",
        "malformed_row",
    ),
)
def test_task11_spend_namespace_rejects_paginated_history_mutants(
    mutation: str,
) -> None:
    prefix = "campaigns/glm52-sky-20260724/ledger/allocations/"
    marker = (prefix + "marker", "page-marker-1")
    version = {
        "Key": prefix + "000001-allocation.json",
        "VersionId": "allocation-version-1",
        "IsLatest": True,
        "Size": 123,
    }

    def page(
        *,
        versions: list[dict[str, object]],
        deletes: list[dict[str, object]],
        truncated: object,
        request_id: str,
        echo: tuple[str, str] | None = None,
        next_marker: tuple[str, str] | None = None,
    ) -> dict[str, object]:
        result = {
            "Versions": versions,
            "DeleteMarkers": deletes,
            "IsTruncated": truncated,
            "Name": "campaign-bucket",
            "Prefix": prefix,
            "MaxKeys": 1000,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": request_id,
                "RetryAttempts": 0,
            },
        }
        if echo is not None:
            result["KeyMarker"] = echo[0]
            result["VersionIdMarker"] = echo[1]
        if next_marker is not None:
            result["NextKeyMarker"] = next_marker[0]
            result["NextVersionIdMarker"] = next_marker[1]
        return result

    if mutation == "literal_is_truncated":
        pages = [
            page(
                versions=[version],
                deletes=[],
                truncated="false",
                request_id="list-1",
            )
        ]
    elif mutation == "missing_next_marker":
        pages = [
            page(
                versions=[],
                deletes=[],
                truncated=True,
                request_id="list-1",
            )
        ]
    elif mutation in ("marker_cycle", "marker_echo"):
        pages = [
            page(
                versions=[],
                deletes=[],
                truncated=True,
                request_id="list-1",
                next_marker=marker,
            ),
            page(
                versions=[] if mutation == "marker_cycle" else [version],
                deletes=[],
                truncated=mutation == "marker_cycle",
                request_id="list-2",
                echo=(
                    marker
                    if mutation == "marker_cycle"
                    else (marker[0], "foreign-marker")
                ),
                next_marker=marker if mutation == "marker_cycle" else None,
            ),
        ]
    elif mutation == "hidden_delete_marker":
        pages = [
            page(
                versions=[version],
                deletes=[],
                truncated=True,
                request_id="list-1",
                next_marker=marker,
            ),
            page(
                versions=[],
                deletes=[
                    {
                        "Key": version["Key"],
                        "VersionId": "deleted-version",
                        "IsLatest": False,
                    }
                ],
                truncated=False,
                request_id="list-2",
                echo=marker,
            ),
        ]
    else:
        mutant = dict(version)
        if mutation == "foreign_version":
            mutant["Key"] = "foreign/key.json"
        elif mutation == "malformed_row":
            mutant.pop("VersionId")
        pages = [
            page(
                versions=[mutant],
                deletes=[],
                truncated=mutation == "duplicate_version",
                request_id="list-1",
                next_marker=(
                    marker if mutation == "duplicate_version" else None
                ),
            )
        ]
        if mutation == "duplicate_version":
            pages.append(
                page(
                    versions=[version],
                    deletes=[],
                    truncated=False,
                    request_id="list-2",
                    echo=marker,
                )
            )
        elif mutation == "extra_page_field":
            pages[0]["Foreign"] = "field"
        elif mutation == "missing_request_id":
            pages[0]["ResponseMetadata"].pop("RequestId")

    class S3:
        def __init__(self) -> None:
            self.pages = list(pages)

        def list_object_versions(self, **request: object):
            return self.pages.pop(0)

    store = Task11SpendObjectStore(
        s3=S3(),
        bucket="campaign-bucket",
        version_by_key={},
    )
    with pytest.raises(RuntimeError):
        first = store.list_namespace(
            prefix=prefix,
            continuation_token=None,
        )
        if first.next_token is not None:
            store.list_namespace(
                prefix=prefix,
                continuation_token=first.next_token,
            )


def test_task11_spend_store_authenticates_get_and_two_page_history() -> None:
    raw = b'{"spend":"evidence"}'
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    prefix = "campaigns/glm52-sky-20260724/ledger/allocations/"
    marker = (prefix + "marker", "page-marker-1")

    class S3:
        def get_object(self, **request: object):
            return {
                "Body": io.BytesIO(raw),
                "VersionId": "spend-version-1",
                "ContentLength": len(raw),
                "ChecksumSHA256": checksum,
                "ETag": '"service-etag"',
                    "ResponseMetadata": {
                        "HTTPStatusCode": 200,
                        "RequestId": "spend-get-1",
                        "RetryAttempts": 0,
                    },
            }

        def list_object_versions(self, **request: object):
            common = {
                "DeleteMarkers": [],
                "Name": "campaign-bucket",
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if "KeyMarker" not in request:
                return {
                    **common,
                    "Versions": [
                        {
                            "Key": prefix + "000001.json",
                            "VersionId": "allocation-version-1",
                            "IsLatest": True,
                            "Size": 123,
                        }
                    ],
                    "IsTruncated": True,
                    "NextKeyMarker": marker[0],
                    "NextVersionIdMarker": marker[1],
                        "ResponseMetadata": {
                            "HTTPStatusCode": 200,
                            "RequestId": "spend-list-1",
                            "RetryAttempts": 0,
                    },
                }
            assert request["KeyMarker"] == marker[0]
            assert request["VersionIdMarker"] == marker[1]
            return {
                **common,
                "Versions": [
                    {
                        "Key": prefix + "000002.json",
                        "VersionId": "allocation-version-2",
                        "IsLatest": True,
                        "Size": 456,
                    }
                ],
                "IsTruncated": False,
                "KeyMarker": marker[0],
                "VersionIdMarker": marker[1],
                    "ResponseMetadata": {
                        "HTTPStatusCode": 200,
                        "RequestId": "spend-list-2",
                        "RetryAttempts": 0,
                },
            }

    store = Task11SpendObjectStore(
        s3=S3(),
        bucket="campaign-bucket",
        version_by_key={"spend/key.json": "spend-version-1"},
    )
    observed = store.get_object(key="spend/key.json")
    first = store.list_namespace(prefix=prefix, continuation_token=None)
    second = store.list_namespace(
        prefix=prefix,
        continuation_token=first.next_token,
    )

    assert observed.etag == '"service-etag"'
    assert observed.checksum_sha256 == hashlib.sha256(raw).hexdigest()
    assert first.keys == (prefix + "000001.json",)
    assert second.keys == (prefix + "000002.json",)
    assert second.next_token is None


def test_task11_spend_namespace_pagination_is_finitely_bounded() -> None:
    prefix = "campaigns/glm52-sky-20260724/ledger/allocations/"

    class S3:
        def __init__(self) -> None:
            self.calls = 0

        def list_object_versions(self, **request: object):
            current = self.calls
            self.calls += 1
            response = {
                "Versions": [],
                "DeleteMarkers": [],
                "IsTruncated": True,
                "Name": "campaign-bucket",
                "Prefix": prefix,
                "MaxKeys": 1000,
                "NextKeyMarker": prefix + f"marker-{self.calls:03d}",
                "NextVersionIdMarker": f"version-{self.calls:03d}",
                    "ResponseMetadata": {
                        "HTTPStatusCode": 200,
                        "RequestId": f"spend-list-{self.calls}",
                        "RetryAttempts": 0,
                },
            }
            if current:
                response["KeyMarker"] = request["KeyMarker"]
                response["VersionIdMarker"] = request["VersionIdMarker"]
            return response

    s3 = S3()
    store = Task11SpendObjectStore(
        s3=s3,
        bucket="campaign-bucket",
        version_by_key={},
    )
    token = None
    with pytest.raises(RuntimeError, match="marker|bound"):
        for _ in range(64):
            token = store.list_namespace(
                prefix=prefix,
                continuation_token=token,
            ).next_token
    assert s3.calls == 64


def _ec2_instance(
    *,
    instance_id: str = "i-00000000000000001",
    instance_type: str = "p5.48xlarge",
    volume_id: str = "vol-00000000000000001",
    activation_id: str = ACTIVATION_ID,
    generation: int = 1,
    allocation_ordinal: int = 1,
    client_token: str = "e" * 64,
) -> dict[str, object]:
    activation_ordinal_text = "00000001"
    generation_text = f"{generation:08d}"
    allocation_ordinal_text = f"{allocation_ordinal:08d}"
    return {
        "Architecture": "x86_64",
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "AttachTime": datetime(
                        2026, 7, 29, 16, tzinfo=timezone.utc
                    ),
                    "DeleteOnTermination": True,
                    "Status": "attached",
                    "VolumeId": volume_id,
                },
            }
        ],
        "EbsOptimized": True,
        "EnaSupport": True,
        "ImageId": "ami-0123456789abcdef0",
        "InstanceId": instance_id,
        "InstanceType": instance_type,
        "ClientToken": client_token,
        "LaunchTime": datetime(2026, 7, 29, 16, tzinfo=timezone.utc),
        "Placement": {
            "AvailabilityZone": "us-west-2a",
            "AvailabilityZoneId": "usw2-az1",
            "Tenancy": "default",
        },
        "RootDeviceName": "/dev/sda1",
        "RootDeviceType": "ebs",
        "State": {"Code": 16, "Name": "running"},
        "Tags": [
            {"Key": "Project", "Value": "KEEP"},
            {"Key": "Campaign", "Value": "GLM-5.2"},
            {"Key": "RunId", "Value": RUN_ID},
            {"Key": "Market", "Value": "on-demand"},
            {"Key": "campaign-identity-sha256", "Value": "1" * 64},
            {"Key": "activation-id", "Value": activation_id},
            {
                "Key": "activation-ordinal-text",
                "Value": activation_ordinal_text,
            },
            {"Key": "generation-text", "Value": generation_text},
            {
                "Key": "allocation-ordinal-text",
                "Value": allocation_ordinal_text,
            },
            {
                "Key": "action-key",
                "Value": (
                    f"ACTION#{activation_ordinal_text}#SKY_POST#"
                    f"{generation_text}"
                ),
            },
            {"Key": "sky-request-id", "Value": "sky-request-0001"},
            {"Key": "sky-job-name", "Value": RUN_ID},
            {"Key": "sky-task-name", "Value": "glm52-production"},
            {"Key": "task-yaml-sha256", "Value": "2" * 64},
            {"Key": "request-body-sha256", "Value": "3" * 64},
        ],
        "VirtualizationType": "hvm",
    }


def _ec2_page(
    *,
    instances: list[dict[str, object]],
    request_id: str,
    next_token: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "Reservations": [
            {
                "Groups": [],
                "Instances": instances,
                "OwnerId": ACCOUNT_ID,
                "ReservationId": "r-00000000000000001",
            }
        ],
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "RequestId": request_id,
            "RetryAttempts": 0,
        },
    }
    if next_token is not None:
        value["NextToken"] = next_token
    return value


def test_task11_spend_ec2_authenticates_two_page_instance_volume_history() -> None:
    """Break caught: EC2 pagination or root-volume identity is locally invented."""

    pages = [
        _ec2_page(
            instances=[_ec2_instance()],
            request_id="ec2-page-1",
            next_token="service-token-1",
        ),
        _ec2_page(
            instances=[
                _ec2_instance(
                    instance_id="i-00000000000000002",
                    volume_id="vol-00000000000000002",
                )
            ],
            request_id="ec2-page-2",
        ),
    ]

    class Ec2:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def describe_instances(self, **request: object) -> object:
            self.calls.append(dict(request))
            return pages.pop(0)

    raw = Ec2()
    adapter = Task11SpendEc2(raw, activation_id=ACTIVATION_ID)
    first = adapter.describe_allocation_history(
        run_id=RUN_ID,
        next_token=None,
    )
    second = adapter.describe_allocation_history(
        run_id=RUN_ID,
        next_token=first["next_token"],
    )
    exact_filters = [
        {"Name": "tag:RunId", "Values": [RUN_ID]},
        {"Name": "tag:activation-id", "Values": [ACTIVATION_ID]},
        {
            "Name": "instance-state-name",
            "Values": ["pending", "running"],
        },
    ]
    assert raw.calls == [
        {"Filters": exact_filters, "MaxResults": 1000},
        {
            "Filters": exact_filters,
            "MaxResults": 1000,
            "NextToken": "service-token-1",
        },
    ]
    assert first["next_token"] != "service-token-1"
    assert second["next_token"] is None
    assert first["instances"][0]["InstanceId"] == "i-00000000000000001"
    assert second["instances"][0]["InstanceId"] == "i-00000000000000002"
    assert first["instances"][0]["RootVolumeId"] == (
        "vol-00000000000000001"
    )
    assert first["instances"][0]["RequestId"] == "ec2-page-1"
    assert first["instances"][0]["ClientToken"] == "e" * 64
    assert first["instances"][0]["Tags"] == {
        tag["Key"]: tag["Value"] for tag in _ec2_instance()["Tags"]
    }
    assert len(first["instances"][0]["ResponseIdentitySha256"]) == 64


def test_task11_spend_ec2_rejects_a_second_current_instance_for_one_launch() -> None:
    """Break caught: one active retained launch is counted more than once."""

    first_instance = _ec2_instance()
    expected_tags = {
        tag["Key"]: tag["Value"] for tag in first_instance["Tags"]
    }
    pages = [
        _ec2_page(
            instances=[first_instance],
            request_id="ec2-active-1",
            next_token="service-token-1",
        ),
        _ec2_page(
            instances=[
                _ec2_instance(
                    instance_id="i-00000000000000002",
                    volume_id="vol-00000000000000002",
                )
            ],
            request_id="ec2-active-2",
        ),
    ]

    class Ec2:
        def describe_instances(self, **request: object) -> object:
            return pages.pop(0)

    adapter = Task11SpendEc2(
        Ec2(),
        activation_id=ACTIVATION_ID,
        expected_launches={
            (ACTIVATION_ID, 1, 1): (expected_tags, "e" * 64)
        },
    )
    token = adapter.describe_allocation_history(
        run_id=RUN_ID,
        next_token=None,
    )["next_token"]
    with pytest.raises(RuntimeError, match="duplicated"):
        adapter.describe_allocation_history(run_id=RUN_ID, next_token=token)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_request_id",
        "extra_metadata",
        "nonzero_retry",
        "extra_page_field",
        "extra_reservation_field",
        "extra_instance_field",
        "missing_instance_field",
        "foreign_instance",
        "missing_client_token",
        "invalid_client_token",
        "capacity_block",
        "capacity_reservation_id",
        "capacity_reservation_specification",
        "spot_lifecycle",
        "spot_request",
        "dedicated_tenancy",
        "foreign_volume",
        "extra_volume_field",
        "foreign_tag",
        "missing_root_mapping",
    ),
)
def test_task11_spend_ec2_rejects_unauthenticated_transport_mutants(
    mutation: str,
) -> None:
    """Break caught: malformed EC2 service evidence becomes spend authority."""

    page = _ec2_page(
        instances=[_ec2_instance()],
        request_id="ec2-page-1",
    )
    metadata = page["ResponseMetadata"]
    reservation = page["Reservations"][0]
    instance = reservation["Instances"][0]
    if mutation == "missing_request_id":
        metadata.pop("RequestId")
    elif mutation == "extra_metadata":
        metadata["Fabricated"] = "yes"
    elif mutation == "nonzero_retry":
        metadata["RetryAttempts"] = 1
    elif mutation == "extra_page_field":
        page["Fabricated"] = "yes"
    elif mutation == "extra_reservation_field":
        reservation["Fabricated"] = "yes"
    elif mutation == "extra_instance_field":
        instance["Fabricated"] = "yes"
    elif mutation == "missing_instance_field":
        instance.pop("ImageId")
    elif mutation == "foreign_instance":
        instance["InstanceType"] = "g5.48xlarge"
    elif mutation == "missing_client_token":
        instance.pop("ClientToken")
    elif mutation == "invalid_client_token":
        instance["ClientToken"] = "E" * 64
    elif mutation == "capacity_block":
        instance["CapacityBlockId"] = "crb-0123456789abcdef0"
    elif mutation == "capacity_reservation_id":
        instance["CapacityReservationId"] = "cr-0123456789abcdef0"
    elif mutation == "capacity_reservation_specification":
        instance["CapacityReservationSpecification"] = {
            "CapacityReservationPreference": "open"
        }
    elif mutation == "spot_lifecycle":
        instance["InstanceLifecycle"] = "spot"
    elif mutation == "spot_request":
        instance["SpotInstanceRequestId"] = "sir-0123456789abcdef0"
    elif mutation == "dedicated_tenancy":
        instance["Placement"]["Tenancy"] = "dedicated"
    elif mutation == "foreign_volume":
        instance["BlockDeviceMappings"][0]["Ebs"]["VolumeId"] = (
            "vol-foreign"
        )
    elif mutation == "extra_volume_field":
        instance["BlockDeviceMappings"][0]["Ebs"]["Fabricated"] = "yes"
    elif mutation == "foreign_tag":
        instance["Tags"][-1]["Value"] = "foreign-run"
    else:
        instance["BlockDeviceMappings"] = []

    class Ec2:
        def describe_instances(self, **request: object) -> object:
            return page

    with pytest.raises(RuntimeError):
        Task11SpendEc2(
            Ec2(),
            activation_id=ACTIVATION_ID,
        ).describe_allocation_history(
            run_id=RUN_ID,
            next_token=None,
        )


@pytest.mark.parametrize(
    ("tag_key", "bad_value"),
    (
        ("Project", "keep"),
        ("Campaign", "glm-5.2"),
        ("RunId", "foreign-run"),
        ("Market", "spot"),
        ("campaign-identity-sha256", "foreign"),
        ("activation-id", "foreign-activation"),
        ("activation-ordinal-text", "1"),
        ("generation-text", "00000002"),
        ("allocation-ordinal-text", "1"),
        ("action-key", "ACTION#foreign"),
        ("sky-request-id", ""),
        ("sky-job-name", ""),
        ("sky-task-name", ""),
        ("task-yaml-sha256", "2" * 63),
        ("request-body-sha256", "3" * 63),
    ),
)
def test_task11_spend_ec2_rejects_frozen_launch_tag_mutants(
    tag_key: str,
    bad_value: str,
) -> None:
    """Break caught: live EC2 identity diverges from frozen launch custody."""

    instance = _ec2_instance()
    next(
        tag for tag in instance["Tags"] if tag["Key"] == tag_key
    )["Value"] = bad_value

    class Ec2:
        def describe_instances(self, **request: object) -> object:
            return _ec2_page(
                instances=[instance],
                request_id="ec2-tag-mutant",
            )

    with pytest.raises(RuntimeError):
        Task11SpendEc2(
            Ec2(),
            activation_id=ACTIVATION_ID,
        ).describe_allocation_history(
            run_id=RUN_ID,
            next_token=None,
        )


def test_task11_spend_ec2_rejects_pagination_cycle_and_token_substitution() -> None:
    """Break caught: a caller supplies a raw or repeated EC2 service token."""

    pages = [
        _ec2_page(
            instances=[_ec2_instance()],
            request_id="ec2-page-1",
            next_token="service-token-1",
        ),
        _ec2_page(
            instances=[],
            request_id="ec2-page-2",
            next_token="service-token-1",
        ),
    ]

    class Ec2:
        def describe_instances(self, **request: object) -> object:
            return pages.pop(0)

    adapter = Task11SpendEc2(Ec2(), activation_id=ACTIVATION_ID)
    first = adapter.describe_allocation_history(
        run_id=RUN_ID,
        next_token=None,
    )
    with pytest.raises(RuntimeError):
        adapter.describe_allocation_history(
            run_id=RUN_ID,
            next_token="service-token-1",
        )
    with pytest.raises(RuntimeError):
        adapter.describe_allocation_history(
            run_id=RUN_ID,
            next_token=first["next_token"],
        )


def test_task11_spend_ec2_pagination_is_finitely_bounded() -> None:
    """Break caught: a perpetually truncated EC2 walk can outlive closure."""

    class Ec2:
        def __init__(self) -> None:
            self.calls = 0

        def describe_instances(self, **request: object) -> object:
            self.calls += 1
            return _ec2_page(
                instances=[],
                request_id=f"ec2-page-{self.calls}",
                next_token=f"service-token-{self.calls}",
            )

    raw = Ec2()
    adapter = Task11SpendEc2(raw, activation_id=ACTIVATION_ID)
    token = None
    with pytest.raises(RuntimeError, match="progress|bound"):
        for _ in range(64):
            token = adapter.describe_allocation_history(
                run_id=RUN_ID,
                next_token=token,
            )["next_token"]
    assert raw.calls == 64


def _ddb_page(
    *,
    items: list[dict[str, object]],
    request_id: str,
    last_key: dict[str, object] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "Items": items,
        "Count": len(items),
        "ScannedCount": len(items),
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "RequestId": request_id,
            "RetryAttempts": 0,
        },
    }
    if last_key is not None:
        value["LastEvaluatedKey"] = last_key
    return value


def _ddb_reserve_item(
    generation: int = 1,
    *,
    allocation_ordinal: int = 1,
    activation_id: str = ACTIVATION_ID,
    ec2_client_token: str = "e" * 64,
) -> dict[str, object]:
    reserve_key = (
        f"GPU_LIABILITY_RESERVE#{activation_id}#"
        f"{generation:08d}#{allocation_ordinal:08d}"
    )
    body = {
        "schema_version": 1,
        "record_type": "glm52_gpu_liability_reserve_v1",
        "account_id": ACCOUNT_ID,
        "region": "us-west-2",
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "generation": generation,
        "generation_text": f"{generation:08d}",
        "allocation_ordinal": allocation_ordinal,
        "allocation_ordinal_text": f"{allocation_ordinal:08d}",
        "ec2_client_token": ec2_client_token,
        "request_identity_sha256": "a" * 64,
        "ledger_predecessor_identity_sha256": "b" * 64,
        "gpu_reserve_seconds": 900,
        "gpu_reserve_cost_usd": "13.76",
        "root_volume_gib": 300,
        "root_volume_tail_usd_max": "0.01",
        "residual_liability_approval_identity_sha256": "c" * 64,
        "epoch": 1,
        "revision": 1,
        "nonce_owner_identity_sha256": "d" * 64,
        "state": "HELD",
        "observed_at": "2026-07-29T16:00:00Z",
        "reserve_key": reserve_key,
    }
    return encode_item(
        {
            "PK": "RUN#glm52-sky-20260724",
            "SK": reserve_key,
            **body,
            "canonical_body_sha256": canonical_sha256(body),
        }
    )


def _ddb_nonreserve_item() -> dict[str, object]:
    return encode_item(
        {
            "PK": "RUN#glm52-sky-20260724",
            "SK": "ACTIVATION#approved-20260728#CONTROL",
            "record_type": "glm52_production_control",
        }
    )


def test_task11_reserve_reader_authenticates_two_page_query_custody() -> None:
    """Break caught: Dynamo continuation state is accepted without page custody."""

    first_item = _ddb_reserve_item(1)
    second_item = _ddb_reserve_item(
        2,
        activation_id="approved-20260729",
        ec2_client_token="f" * 64,
    )
    last_key = {
        "PK": {"S": "RUN#glm52-sky-20260724"},
        "SK": {
            "S": (
                "GPU_LIABILITY_RESERVE#approved-20260728#"
                "00000001#00000001"
            )
        },
    }
    pages = [
        _ddb_page(
            items=[first_item],
            request_id="ddb-page-1",
            last_key=last_key,
        ),
        _ddb_page(items=[second_item], request_id="ddb-page-2"),
    ]

    class Dynamo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def query(self, **request: object) -> object:
            self.calls.append(dict(request))
            return pages.pop(0)

    raw = Dynamo()
    reader = Task11ReserveReader(
        dynamodb=raw,
        table_name="keep-glm52-ledger",
    )
    first = reader.list_reserves(next_token=None)
    second = reader.list_reserves(next_token=first.next_token)
    base_request = {
        "TableName": "keep-glm52-ledger",
        "KeyConditionExpression": (
            "#pk = :pk AND begins_with(#sk, :sk_prefix)"
        ),
        "ExpressionAttributeNames": {"#pk": "PK", "#sk": "SK"},
        "ExpressionAttributeValues": {
            ":pk": {"S": "RUN#glm52-sky-20260724"},
            ":sk_prefix": {
                "S": "GPU_LIABILITY_RESERVE#"
            },
        },
        "ConsistentRead": True,
        "Limit": 100,
        "ScanIndexForward": True,
        "ReturnConsumedCapacity": "NONE",
    }
    assert raw.calls == [
        base_request,
        {**base_request, "ExclusiveStartKey": last_key},
    ]
    assert first.next_token is not None
    assert second.next_token is None
    assert first.records[0]["reserve_key"].endswith(
        "#00000001#00000001"
    )
    assert second.records[0]["reserve_key"].endswith(
        "#00000002#00000001"
    )
    assert second.records[0]["activation_id"] == "approved-20260729"
    assert first.records[0]["ec2_client_token"] == "e" * 64


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_request_id",
        "extra_metadata",
        "nonzero_retry",
        "extra_page_field",
        "wrong_count",
        "wrong_scanned_count",
        "foreign_partition",
        "malformed_attribute",
        "nonreserve_item",
        "extra_item_field",
        "missing_item_field",
        "foreign_reserve_type",
        "invalid_client_token",
        "foreign_activation_binding",
        "foreign_generation_binding",
        "foreign_last_key",
    ),
)
def test_task11_reserve_reader_rejects_unauthenticated_query_mutants(
    mutation: str,
) -> None:
    """Break caught: malformed Dynamo rows or metadata disappear from evidence."""

    item = _ddb_reserve_item()
    page = _ddb_page(items=[item], request_id="ddb-page-1")
    metadata = page["ResponseMetadata"]
    if mutation == "missing_request_id":
        metadata.pop("RequestId")
    elif mutation == "extra_metadata":
        metadata["Fabricated"] = "yes"
    elif mutation == "nonzero_retry":
        metadata["RetryAttempts"] = 1
    elif mutation == "extra_page_field":
        page["Fabricated"] = "yes"
    elif mutation == "wrong_count":
        page["Count"] = 0
    elif mutation == "wrong_scanned_count":
        page["ScannedCount"] = 2
    elif mutation == "foreign_partition":
        item["PK"] = {"S": "RUN#foreign"}
    elif mutation == "malformed_attribute":
        item["SK"] = {"S": "value", "N": "1"}
    elif mutation == "nonreserve_item":
        item = _ddb_nonreserve_item()
        page["Items"] = [item]
    elif mutation == "extra_item_field":
        item["Fabricated"] = {"S": "yes"}
    elif mutation == "missing_item_field":
        item.pop("state")
    elif mutation == "foreign_reserve_type":
        item["record_type"] = {"S": "glm52_production_control"}
    elif mutation == "invalid_client_token":
        item["ec2_client_token"] = {"S": "reserve-token-1"}
    elif mutation == "foreign_activation_binding":
        item["activation_id"] = {"S": "foreign-activation"}
    elif mutation == "foreign_generation_binding":
        item["generation_text"] = {"S": "00000002"}
    else:
        page["LastEvaluatedKey"] = {
            "PK": {"S": "RUN#foreign"},
            "SK": {"S": "ACTIVATION#approved-20260728#CONTROL"},
        }

    class Dynamo:
        def query(self, **request: object) -> object:
            return page

    with pytest.raises(RuntimeError):
        Task11ReserveReader(
            dynamodb=Dynamo(),
            table_name="keep-glm52-ledger",
        ).list_reserves(next_token=None)


def test_task11_reserve_reader_rejects_token_substitution_and_cycle() -> None:
    """Break caught: raw, replayed, or cyclic LastEvaluatedKey is accepted."""

    item = _ddb_reserve_item()
    last_key = {
        "PK": {"S": "RUN#glm52-sky-20260724"},
        "SK": {
            "S": (
                "GPU_LIABILITY_RESERVE#approved-20260728#"
                "00000001#00000001"
            )
        },
    }
    pages = [
        _ddb_page(
            items=[item],
            request_id="ddb-page-1",
            last_key=last_key,
        ),
        _ddb_page(
            items=[],
            request_id="ddb-page-2",
            last_key=last_key,
        ),
    ]

    class Dynamo:
        def query(self, **request: object) -> object:
            return pages.pop(0)

    reader = Task11ReserveReader(
        dynamodb=Dynamo(),
        table_name="keep-glm52-ledger",
    )
    first = reader.list_reserves(next_token=None)
    raw_token = base64.b64encode(canonical_json_bytes(last_key)).decode("ascii")
    with pytest.raises(RuntimeError):
        reader.list_reserves(next_token=raw_token)
    with pytest.raises(RuntimeError):
        reader.list_reserves(next_token=first.next_token)


def test_task11_spend_transport_binds_same_client_token_across_evidence() -> None:
    """Break caught: EC2 allocation and reserve evidence use different tokens."""

    token_custody: dict[tuple[str, int, int], str] = {}

    class Ec2:
        def describe_instances(self, **request: object) -> object:
            return _ec2_page(
                instances=[_ec2_instance(client_token="e" * 64)],
                request_id="ec2-client-token",
            )

    ec2 = Task11SpendEc2(
        Ec2(),
        activation_id=ACTIVATION_ID,
        token_custody=token_custody,
    )
    observed = ec2.describe_allocation_history(
        run_id=RUN_ID,
        next_token=None,
    )
    assert observed["instances"][0]["ClientToken"] == "e" * 64

    class MatchingDynamo:
        def query(self, **request: object) -> object:
            return _ddb_page(
                items=[
                    _ddb_reserve_item(ec2_client_token="e" * 64)
                ],
                request_id="ddb-matching-client-token",
            )

    matching = Task11ReserveReader(
        dynamodb=MatchingDynamo(),
        table_name="keep-glm52-ledger",
        token_custody=token_custody,
    ).list_reserves(next_token=None)
    assert matching.records[0]["ec2_client_token"] == "e" * 64

    mismatched_custody: dict[tuple[str, int, int], str] = {}
    Task11SpendEc2(
        Ec2(),
        activation_id=ACTIVATION_ID,
        token_custody=mismatched_custody,
    ).describe_allocation_history(run_id=RUN_ID, next_token=None)

    class MismatchedDynamo:
        def query(self, **request: object) -> object:
            return _ddb_page(
                items=[
                    _ddb_reserve_item(ec2_client_token="f" * 64)
                ],
                request_id="ddb-mismatched-client-token",
            )

    with pytest.raises(RuntimeError, match="ClientToken|token"):
        Task11ReserveReader(
            dynamodb=MismatchedDynamo(),
            table_name="keep-glm52-ledger",
            token_custody=mismatched_custody,
        ).list_reserves(next_token=None)


def test_task11_reserve_reader_pagination_is_finitely_bounded() -> None:
    """Break caught: a perpetually truncated Query can outlive closure."""

    class Dynamo:
        def __init__(self) -> None:
            self.calls = 0

        def query(self, **request: object) -> object:
            self.calls += 1
            item = _ddb_reserve_item(
                self.calls,
                ec2_client_token=f"{self.calls:064x}",
            )
            last_key = {
                "PK": {"S": "RUN#glm52-sky-20260724"},
                "SK": {
                    "S": (
                        "GPU_LIABILITY_RESERVE#approved-20260728#"
                        f"{self.calls:08d}#00000001"
                    )
                },
            }
            return _ddb_page(
                items=[item],
                request_id=f"ddb-page-{self.calls}",
                last_key=last_key,
            )

    raw = Dynamo()
    reader = Task11ReserveReader(
        dynamodb=raw,
        table_name="keep-glm52-ledger",
    )
    token = None
    with pytest.raises(RuntimeError, match="progress|bound"):
        for _ in range(64):
            token = reader.list_reserves(next_token=token).next_token
    assert raw.calls == 64
ZERO_EFFECTS = (
    ("PRODUCTION_SOURCE", 0),
    ("PRODUCTION_CLAIM", 0),
    ("PRODUCTION_DECISION", 0),
    ("PRODUCTION_ACTION_CONSUME", 0),
    ("SKY_POST", 0),
)


def _spans(
    names_and_durations: tuple[tuple[str, int], ...],
    *,
    start: int = 0,
) -> tuple[PhaseSpan, ...]:
    result = []
    cursor = start
    for name, duration in names_and_durations:
        result.append(PhaseSpan(name, cursor, cursor + duration))
        cursor += duration
    return tuple(result)


def _deployed_measurements() -> tuple[object, ...]:
    suffix_durations = (1, 1, 10, 1, 1, 1, 1, 1)
    closure_durations = (10, 10, 10, 10, 17, 10, 10, 10)
    closure = _spans(
        tuple(zip(CLOSURE_PHASE_CEILINGS, closure_durations))
    )
    suffix = _spans(
        tuple(zip(SUFFIX_PHASE_CEILINGS, suffix_durations)),
        start=closure[4].started_monotonic_seconds,
    )
    failures = ("THROTTLING", "PAGINATION", "NETWORK_AMBIGUITY")
    return tuple(
        build_rehearsal_measurement(
            rehearsal_id="deployed-" + str(index),
            provenance="DEPLOYED_REHEARSAL",
            lambda_environment_id="cold-env-" + str(index),
            cold_start=index < 5,
            path_proofs=PATH_PROOFS,
            canary_bucket="keep-glm52-h1g-rehearsal",
            canary_key=(
                "rehearsal/canary/" + str(index) + "/decision.json"
            ),
            canary_can_satisfy_production_authority=False,
            canary_worker_readable=False,
            canary_admission_readable=False,
            effect_counts=ZERO_EFFECTS,
            relay_call_count=0,
            closure_spans=closure,
            suffix_spans=suffix,
            first_policy_readback_monotonic_seconds=(
                suffix[2].started_monotonic_seconds
            ),
            second_policy_readback_monotonic_seconds=(
                suffix[2].started_monotonic_seconds + 10
            ),
            injected_failure_kind=(
                failures[index] if index < 3 else "NONE"
            ),
            failure_unwind_seconds=10 if index < 3 else 0,
        )
        for index in range(20)
    )


def _gate_bytes(measurements: tuple[object, ...] | None = None) -> bytes:
    return build_deployed_gate_document(
        account_id=ACCOUNT_ID,
        region="us-west-2",
        run_id=RUN_ID,
        activation_id=ACTIVATION_ID,
        deployment_identity_sha256=DEPLOYMENT_SHA,
        measurements=(
            _deployed_measurements()
            if measurements is None
            else measurements
        ),
    )


class GateS3:
    def __init__(
        self,
        raw: bytes,
        *,
        version_id: str = VERSION_ID,
        checksum: str | None = None,
    ) -> None:
        self.raw = raw
        self.version_id = version_id
        self.checksum = (
            base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
            if checksum is None
            else checksum
        )
        self.calls: list[tuple[str, dict[str, object]]] = []

    def list_object_versions(self, **kwargs: object):
        self.calls.append(("list_object_versions", dict(kwargs)))
        return {
            "Versions": [
                {
                    "Key": KEY,
                    "VersionId": VERSION_ID,
                    "IsLatest": True,
                    "Size": len(self.raw),
                }
            ],
            "DeleteMarkers": [],
            "IsTruncated": False,
            "Name": BUCKET,
            "Prefix": KEY,
            "MaxKeys": 1000,
            "ResponseMetadata": {"HTTPStatusCode": 200, "RequestId": "list-1"},
        }

    def get_object(self, **kwargs: object):
        self.calls.append(("get_object", dict(kwargs)))
        return {
            "Body": io.BytesIO(self.raw),
            "VersionId": self.version_id,
            "ContentLength": len(self.raw),
            "ChecksumSHA256": self.checksum,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "get-1",
                "HTTPHeaders": {
                    "date": "Tue, 28 Jul 2026 18:00:00 GMT",
                    "x-amz-version-id": self.version_id,
                    "x-amz-checksum-sha256": self.checksum,
                },
            },
        }


class ProductionContext:
    aws_request_id = "11111111-2222-4333-8444-555555555555"
    invoked_function_arn = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-h1g-decision:11"
    )

    @staticmethod
    def get_remaining_time_in_millis() -> int:
        return 800_000


class FiniteProductionS3:
    def __init__(self, raw: bytes, version_id: str) -> None:
        self.raw = raw
        self.version_id = version_id
        self.calls: list[str] = []

    def get_object(self, **kwargs: object):
        self.calls.append("get_object")
        return {
            "Body": io.BytesIO(self.raw),
            "VersionId": self.version_id,
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(self.raw).digest()
            ).decode("ascii"),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "boundary-read-1",
            },
        }

    def list_object_versions(self, **kwargs: object):
        self.calls.append("list_object_versions")
        return {
            "Versions": [],
            "DeleteMarkers": [],
            "IsTruncated": False,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "size-read-1",
            },
        }


class FiniteProductionCloudFormation:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_change_sets(self, **kwargs: object):
        self.calls.append("list_change_sets")
        return {
            "Summaries": [],
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "change-set-list-1",
            },
        }


class FiniteProductionLambda:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_function(self, **kwargs: object):
        self.calls.append("get_function")
        return {
            "Configuration": {"FunctionArn": kwargs["FunctionName"]},
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "get-function-1",
            },
        }


def _production_boundary():
    bucket = "keep-glm52-campaign"
    inputs = tuple(
        build_task11_input_coordinate(
            input_kind=kind,
            bucket=bucket,
            key=(
                (
                    "campaigns/"
                    + RUN_ID
                    + "/authorities/task9/"
                    + ACTIVATION_ID
                    + "/TASK9_DEPLOYED_IDENTITY.json"
                )
                if index == 13
                else (
                    "campaigns/"
                    + RUN_ID
                    + "/authorities/task11/"
                    + ACTIVATION_ID
                    + "/00000001/"
                    + f"{index + 1:02d}-"
                    + kind.lower().replace("_", "-")
                    + ".json"
                )
            ),
            version_id="input-version-" + str(index + 1),
            file_sha256=hashlib.sha256(
                ("file-" + kind).encode()
            ).hexdigest(),
            body_sha256=hashlib.sha256(
                ("body-" + kind).encode()
            ).hexdigest(),
        )
        for index, kind in enumerate(BOUNDARY_INPUT_KINDS)
    )
    return build_task11_boundary_document(
        activation_id=ACTIVATION_ID,
        generation=1,
        campaign_identity_sha256=hashlib.sha256(
            b"campaign"
        ).hexdigest(),
        state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-production:7"
        ),
        action_key=(
            "ACTIVATION#"
            + ACTIVATION_ID
            + "#ACTION#SKY_POST#00000001"
        ),
        inputs=inputs,
    )


def _budget_config() -> BudgetGateConfig:
    return BudgetGateConfig(
        account_id=ACCOUNT_ID,
        region="us-west-2",
        run_id=RUN_ID,
        activation_id=ACTIVATION_ID,
        bucket=BUCKET,
        key=KEY,
        expected_bucket_owner=ACCOUNT_ID,
        deployment_identity_sha256=DEPLOYMENT_SHA,
    )


def _route_fixture_module():
    name = "_task11_route_fixture"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "tests/test_glm52_task11_decision_route.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _trusted_gate() -> dict[str, object]:
    return dict(
        support_budget_gate_handler.main(
            {
                "schema_version": 1,
                "record_type": "glm52_task11_budget_gate_request_v1",
            },
            object(),
            services=GateS3(_gate_bytes()),
            config=_budget_config(),
        )
    )


def _decision_event() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_task11_decision_handler_request_v1",
        "closure_request": {
            "activation_id": ACTIVATION_ID,
            "generation": 1,
            "candidate_identity_sha256": hashlib.sha256(
                b"task11"
            ).hexdigest(),
            "initial_source_predecessor_version_id": "predecessor-v1",
            "admission_version_arn": (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-launch-admission:7"
            ),
            "numeric_binding_version_arn": (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-numeric-binding:4"
            ),
            "task11_boundary": {
                "bucket": BUCKET,
                "key": (
                    "campaigns/glm52-sky-20260724/authorities/task11/"
                    + ACTIVATION_ID
                    + "/00000001.json"
                ),
                "version_id": "task11-boundary-version-1",
                "file_sha256": hashlib.sha256(
                    b"task11-boundary-file"
                ).hexdigest(),
                "body_sha256": hashlib.sha256(
                    b"task11-boundary-body"
                ).hexdigest(),
            },
        },
        "trusted_closure_budget": _trusted_gate(),
    }


def test_task11_fix1_red_real_handlers_are_importable_and_not_fixture_only() -> None:
    assert callable(support_budget_gate_handler.main)
    assert callable(support_decision_handler.main)
    with pytest.raises(ValueError, match="event field set"):
        support_decision_handler.main({}, object())


def test_task11_fix1_red_decision_handler_executes_real_injected_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
        DEPLOYMENT_SHA,
    )
    fixtures = _route_fixture_module()
    services = fixtures.ExactClosureServices()
    clock = fixtures.RouteClock()
    result = support_decision_handler.main(
        _decision_event(),
        object(),
        services=services,
        monotonic_clock=clock.clock,
        sleeper=clock.sleep,
    )
    assert result["status"] == "NUMERIC_BINDING_RECONCILIATION"
    assert tuple(result["steps"]) == fixtures.CLOSURE_STEPS
    assert tuple(
        name for name in services.calls if name in fixtures.SOURCE_METHODS
    ) == fixtures.SOURCE_METHODS
    assert services.calls.count(fixtures.CLOSURE_STEPS[19]) == 1


@pytest.mark.parametrize("mutation", ("absent", "empty", "foreign"))
def test_task11_decision_requires_exact_deployment_identity_environment(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    if mutation == "absent":
        monkeypatch.delenv(
            "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
            raising=False,
        )
    elif mutation == "empty":
        monkeypatch.setenv(
            "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
            "",
        )
    else:
        monkeypatch.setenv(
            "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
            hashlib.sha256(b"foreign-deployment").hexdigest(),
        )

    with pytest.raises(
        ValueError,
        match="another deployment|deployment identity",
    ):
        support_decision_handler.main(
            _decision_event(),
            object(),
            services=object(),
        )


def test_task11_fix1_red_default_factory_reaches_method_complete_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import decision_closure
    from glm52_enforcement import task11_production
    from glm52_enforcement.task11_production import (
        Task11AwsClients,
        Task11ProductionServices,
    )

    fixtures = _route_fixture_module()
    boundary = _production_boundary()
    boundary_raw = canonical_json_bytes(asdict(boundary)) + b"\n"
    s3 = FiniteProductionS3(
        boundary_raw,
        "task11-boundary-version-1",
    )
    cloudformation = FiniteProductionCloudFormation()
    lambda_client = FiniteProductionLambda()
    closure_role_name = (
        "keep-glm52-h1g-closure-session-"
        + hashlib.sha256(ACTIVATION_ID.encode("ascii")).hexdigest()[:16]
    )
    assumed_arn = (
        "arn:aws:sts::"
        + ACCOUNT_ID
        + ":assumed-role/"
        + closure_role_name
        + "/h1g-decision-"
        + hashlib.sha256(ACTIVATION_ID.encode("ascii")).hexdigest()[:16]
        + "-"
        + hashlib.sha256(
            ProductionContext.aws_request_id.encode("ascii")
        ).hexdigest()[:16]
    )
    clients = Task11AwsClients(
        s3=s3,
        dynamodb=object(),
        cloudformation=cloudformation,
        lambda_client=lambda_client,
        sts=object(),
        h1d_clients={},
        credential_expiration="2026-07-29T20:15:00Z",
        caller_identity={
            "UserId": "AROATEST:" + assumed_arn.rsplit("/", 1)[1],
            "Account": ACCOUNT_ID,
            "Arn": assumed_arn,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "caller-identity-1",
            },
        },
    )
    for name, value in {
        "GLM52_ACCOUNT_ID": ACCOUNT_ID,
        "AWS_REGION": "us-west-2",
        "GLM52_RUN_ID": RUN_ID,
        "GLM52_ACTIVATION_ID": ACTIVATION_ID,
        "GLM52_LEDGER_TABLE_NAME": "keep-glm52-ledger",
        "GLM52_CAMPAIGN_BUCKET": "keep-glm52-campaign",
        "GLM52_MODEL_BUCKET": "keep-glm52-campaign",
        "GLM52_MODEL_PREFIX": "models/glm52/",
        "GLM52_FENCE_STACK_ID": "keep-glm52-gpu-fence",
        "GLM52_SUPPORT_STACK_ID": "keep-glm52-h1g-support",
        "GLM52_CLOSURE_ROLE_ARN": (
            "arn:aws:iam::246813579024:role/" + closure_role_name
        ),
        "GLM52_ATTESTATION_VERSION_ARN": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-attestation:3"
        ),
        "GLM52_LAUNCH_ADMISSION_VERSION_ARN": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-launch-admission:7"
        ),
        "GLM52_NUMERIC_BINDING_VERSION_ARN": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-numeric-binding:4"
        ),
        "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": DEPLOYMENT_SHA,
        "AWS_LAMBDA_FUNCTION_NAME": "keep-glm52-h1g-decision",
        "AWS_LAMBDA_FUNCTION_VERSION": "11",
    }.items():
        monkeypatch.setenv(name, value)
    for name, function_name in {
        "GLM52_SOURCE_GPU_SPEND_VERSION_ARN": "source-gpu-spend",
        "GLM52_SOURCE_SUBMISSION_INTENT_VERSION_ARN": (
            "source-submission-intent"
        ),
        "GLM52_SOURCE_CONTROLLER_BASELINE_VERSION_ARN": (
            "source-controller-baseline"
        ),
        "GLM52_SOURCE_CONTROL_PLANE_READINESS_VERSION_ARN": (
            "source-control-plane-readiness"
        ),
        "GLM52_SOURCE_SUBMISSION_ACQUISITION_VERSION_ARN": (
            "source-submission-acquisition"
        ),
        "GLM52_FENCE_EXECUTOR_VERSION_ARN": "fence-executor",
        "GLM52_FENCE_SUCCESSOR_VERSION_ARN": "fence-successor",
        "GLM52_CLAIM_WRITER_VERSION_ARN": "claim-writer",
        "GLM52_DECISION_WRITER_VERSION_ARN": "decision-writer",
        "GLM52_TERMINAL_V1_WRITER_VERSION_ARN": "terminal-v1-writer",
        "GLM52_CLOSURE_HANDOFF_VERSION_ARN": "closure-handoff",
    }.items():
        monkeypatch.setenv(
            name,
            "arn:aws:lambda:us-west-2:"
            + ACCOUNT_ID
            + ":function:keep-glm52-h1g-"
            + function_name
            + ":1",
        )
    monkeypatch.setattr(
        task11_production,
        "_aws_clients",
        lambda *, config, context: clients,
    )
    production_services: list[Task11ProductionServices] = []

    def execute_with_factory_service(request, **kwargs):
        services = kwargs.pop("services")
        assert type(services) is Task11ProductionServices
        production_services.append(services)
        assert all(
            name in Task11ProductionServices.__dict__
            for name in fixtures.METHOD_STEPS
            if name not in {
                "authorize_change_set_create",
                "authorize_change_set_execute",
            }
        )
        services.warm_clients_and_construct(
            request=request,
            custody_nonce_sha256=hashlib.sha256(b"custody").hexdigest(),
        )
        services.acquire_cfn_quiescence(
            request=request,
            custody_nonce_sha256=hashlib.sha256(b"custody").hexdigest(),
        )
        services.size_non_authoritative(
            request=request,
            custody_nonce_sha256=hashlib.sha256(b"custody").hexdigest(),
        )
        services.stable_tls_sky_identity_preflight(
            request=request,
            custody_nonce_sha256=hashlib.sha256(b"custody").hexdigest(),
        )
        return decision_closure.execute_decision_closure(
            request,
            services=fixtures.ExactClosureServices(),
            **kwargs,
        )

    monkeypatch.setattr(
        support_decision_handler,
        "execute_decision_closure",
        execute_with_factory_service,
    )
    event = _decision_event()
    event["closure_request"]["task11_boundary"] = {
        "bucket": "keep-glm52-campaign",
        "key": (
            "campaigns/"
            + RUN_ID
            + "/authorities/task11/"
            + ACTIVATION_ID
            + "/00000001.json"
        ),
        "version_id": "task11-boundary-version-1",
        "file_sha256": hashlib.sha256(boundary_raw).hexdigest(),
        "body_sha256": boundary.canonical_identity_sha256,
    }
    clock = fixtures.RouteClock()
    result = support_decision_handler.main(
        event,
        ProductionContext(),
        monotonic_clock=clock.clock,
        sleeper=clock.sleep,
    )
    assert result["status"] == "NUMERIC_BINDING_RECONCILIATION"
    assert len(production_services) == 1
    assert s3.calls == ["get_object", "list_object_versions"]
    assert cloudformation.calls == ["list_change_sets"]
    assert lambda_client.calls == ["get_function"]


def test_task11_fix1_red_support_lambda_package_is_deterministic(
    tmp_path: Path,
) -> None:
    script = ROOT / "aws/glm52-gpu/scripts/package_h1g_support_lambdas.py"
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    for output in (first, second):
        result = subprocess.run(
            [str(script), str(output)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).hexdigest() == (
        hashlib.sha256(second.read_bytes()).hexdigest()
    )
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert "support_budget_gate_handler.py" in names
        assert "support_decision_handler.py" in names
        assert "support_rehearsal_collector_handler.py" in names
        assert "support_rehearsal_probe_handler.py" in names
        assert "glm52_enforcement/decision_closure.py" in names
        assert "glm52_enforcement/source_publishers.py" in names
        assert "glm52_enforcement/fence_executor.py" in names
        assert "glm52_enforcement/live_authority.py" in names
        assert "glm52_enforcement/sky_admission.py" in names
        assert "glm52_enforcement/task11_production.py" in names
        assert all("__pycache__" not in name for name in names)
        assert all(not name.endswith(".pyc") for name in names)
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(first)!r});"
                "import support_decision_handler as h;"
                "\ntry:\n h.main({}, object())\n"
                "except ValueError: pass\n"
                "else: raise SystemExit(2)"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr


def test_task11_fix1_red_trusted_gate_authenticates_current_20_5_object() -> None:
    s3 = GateS3(_gate_bytes())
    result = support_budget_gate_handler.main(
        {
            "schema_version": 1,
            "record_type": "glm52_task11_budget_gate_request_v1",
        },
        object(),
        services=s3,
        config=_budget_config(),
    )
    assert result["status"] == "CLOSURE_BUDGET_PROVEN"
    assert result["version_id"] == VERSION_ID
    assert result["measurement_count"] == 20
    assert result["cold_environment_count"] == 5
    assert result["deployment_identity_sha256"] == DEPLOYMENT_SHA
    assert len(s3.calls) == 2
    assert s3.calls[1] == (
        "get_object",
        {
            "Bucket": BUCKET,
            "Key": KEY,
            "VersionId": VERSION_ID,
            "ExpectedBucketOwner": ACCOUNT_ID,
            "ChecksumMode": "ENABLED",
        },
    )


def test_task11_budget_gate_accepts_one_version_after_bounded_pagination() -> None:
    class PagedGateS3(GateS3):
        def list_object_versions(self, **kwargs: object):
            self.calls.append(("list_object_versions", dict(kwargs)))
            if "KeyMarker" not in kwargs:
                return {
                    "Versions": [],
                    "DeleteMarkers": [],
                    "IsTruncated": True,
                    "Name": BUCKET,
                    "Prefix": KEY,
                    "MaxKeys": 1000,
                    "NextKeyMarker": KEY,
                    "NextVersionIdMarker": "page-marker-1",
                    "ResponseMetadata": {
                        "HTTPStatusCode": 200,
                        "RequestId": "list-page-1",
                    },
                }
            assert kwargs["KeyMarker"] == KEY
            assert kwargs["VersionIdMarker"] == "page-marker-1"
            return {
                "Versions": [
                    {
                        "Key": KEY,
                        "VersionId": VERSION_ID,
                        "IsLatest": True,
                        "Size": len(self.raw),
                    }
                ],
                "DeleteMarkers": [],
                "IsTruncated": False,
                "Name": BUCKET,
                "Prefix": KEY,
                "MaxKeys": 1000,
                "KeyMarker": KEY,
                "VersionIdMarker": "page-marker-1",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "list-page-2",
                },
            }

    s3 = PagedGateS3(_gate_bytes())

    result = support_budget_gate_handler.main(
        {
            "schema_version": 1,
            "record_type": "glm52_task11_budget_gate_request_v1",
        },
        object(),
        services=s3,
        config=_budget_config(),
    )

    assert result["version_id"] == VERSION_ID
    assert [name for name, _ in s3.calls] == [
        "list_object_versions",
        "list_object_versions",
        "get_object",
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "literal_is_truncated",
        "missing_next_marker",
        "marker_cycle",
        "marker_echo",
        "hidden_second_version",
        "hidden_delete_marker",
        "foreign_version",
        "duplicate_version",
        "extra_page_field",
    ),
)
def test_task11_budget_gate_rejects_paginated_history_mutants(
    mutation: str,
) -> None:
    version = {
        "Key": KEY,
        "VersionId": VERSION_ID,
        "IsLatest": True,
        "Size": len(_gate_bytes()),
    }

    def page(
        *,
        versions: list[dict[str, object]],
        delete_markers: list[dict[str, object]],
        truncated: object,
        request_id: str,
        marker: tuple[str, str] | None = None,
        next_marker: tuple[str, str] | None = None,
    ) -> dict[str, object]:
        result = {
            "Versions": versions,
            "DeleteMarkers": delete_markers,
            "IsTruncated": truncated,
            "Name": BUCKET,
            "Prefix": KEY,
            "MaxKeys": 1000,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": request_id,
            },
        }
        if marker is not None:
            result["KeyMarker"] = marker[0]
            result["VersionIdMarker"] = marker[1]
        if next_marker is not None:
            result["NextKeyMarker"] = next_marker[0]
            result["NextVersionIdMarker"] = next_marker[1]
        return result

    marker = (KEY, "page-marker-1")
    if mutation == "literal_is_truncated":
        pages = [
            page(
                versions=[version],
                delete_markers=[],
                truncated="false",
                request_id="list-1",
            )
        ]
    elif mutation == "missing_next_marker":
        pages = [
            page(
                versions=[],
                delete_markers=[],
                truncated=True,
                request_id="list-1",
            )
        ]
    elif mutation == "marker_cycle":
        pages = [
            page(
                versions=[],
                delete_markers=[],
                truncated=True,
                request_id="list-1",
                next_marker=marker,
            ),
            page(
                versions=[],
                delete_markers=[],
                truncated=True,
                request_id="list-2",
                marker=marker,
                next_marker=marker,
            ),
        ]
    elif mutation == "marker_echo":
        pages = [
            page(
                versions=[],
                delete_markers=[],
                truncated=True,
                request_id="list-1",
                next_marker=marker,
            ),
            page(
                versions=[version],
                delete_markers=[],
                truncated=False,
                request_id="list-2",
                marker=(KEY, "foreign-marker"),
            ),
        ]
    elif mutation == "hidden_delete_marker":
        pages = [
            page(
                versions=[version],
                delete_markers=[],
                truncated=True,
                request_id="list-1",
                next_marker=marker,
            ),
            page(
                versions=[],
                delete_markers=[
                    {
                        "Key": KEY,
                        "VersionId": "deleted-version",
                        "IsLatest": False,
                    }
                ],
                truncated=False,
                request_id="list-2",
                marker=marker,
            ),
        ]
    else:
        second = dict(version)
        if mutation == "hidden_second_version":
            second["VersionId"] = "older-version"
            second["IsLatest"] = False
        elif mutation == "foreign_version":
            second["Key"] = KEY + ".foreign"
        pages = [
            page(
                versions=(
                    [second]
                    if mutation == "foreign_version"
                    else [version]
                ),
                delete_markers=[],
                truncated=(
                    False
                    if mutation in ("foreign_version", "extra_page_field")
                    else True
                ),
                request_id="list-1",
                next_marker=(
                    None
                    if mutation in ("foreign_version", "extra_page_field")
                    else marker
                ),
            )
        ]
        if mutation == "extra_page_field":
            pages[0]["Foreign"] = "field"
        elif mutation != "foreign_version":
            pages.append(
                page(
                    versions=[second],
                    delete_markers=[],
                    truncated=False,
                    request_id="list-2",
                    marker=marker,
                )
            )

    class MutantS3(GateS3):
        def __init__(self) -> None:
            super().__init__(_gate_bytes())
            self.pages = list(pages)

        def list_object_versions(self, **kwargs: object):
            self.calls.append(("list_object_versions", dict(kwargs)))
            return self.pages.pop(0)

    s3 = MutantS3()
    with pytest.raises(ValueError):
        support_budget_gate_handler.main(
            {
                "schema_version": 1,
                "record_type": "glm52_task11_budget_gate_request_v1",
            },
            object(),
            services=s3,
            config=_budget_config(),
        )
    assert all(name != "get_object" for name, _ in s3.calls)


@pytest.mark.parametrize(
    "mutation",
    (
        "caller_proven",
        "stale_version",
        "wrong_checksum",
        "wrong_owner",
        "nineteen",
        "four_cold",
        "wrong_deployment",
    ),
)
def test_task11_fix1_red_trusted_gate_rejects_bypass_and_stale_mutants(
    mutation: str,
) -> None:
    event = {
        "schema_version": 1,
        "record_type": "glm52_task11_budget_gate_request_v1",
    }
    config = _budget_config()
    if mutation == "caller_proven":
        event["status"] = "CLOSURE_BUDGET_PROVEN"
    raw = _gate_bytes()
    if mutation in ("nineteen", "four_cold"):
        body = json.loads(raw)
        if mutation == "nineteen":
            body["measurements"] = body["measurements"][:-1]
            body["measurement_count"] = 19
        else:
            body["measurements"][4]["cold_start"] = False
            body["cold_environment_count"] = 4
        body.pop("canonical_body_sha256")
        body["canonical_body_sha256"] = canonical_sha256(body)
        raw = canonical_json_bytes(body)
    if mutation == "wrong_owner":
        body = json.loads(raw)
        body["account_id"] = "000000000000"
        body.pop("canonical_body_sha256")
        body["canonical_body_sha256"] = canonical_sha256(body)
        raw = canonical_json_bytes(body)
    elif mutation == "wrong_deployment":
        config = replace(
            config,
            deployment_identity_sha256=hashlib.sha256(
                b"another-deployment"
            ).hexdigest(),
        )
    s3 = GateS3(
        raw,
        version_id=(
            "stale-version-0000"
            if mutation == "stale_version"
            else VERSION_ID
        ),
        checksum=(
            base64.b64encode(b"x" * 32).decode("ascii")
            if mutation == "wrong_checksum"
            else None
        ),
    )
    with pytest.raises(ValueError):
        support_budget_gate_handler.main(
            event,
            object(),
            services=s3,
            config=config,
        )
    if mutation == "caller_proven":
        assert s3.calls == []


def test_task11_fence_caller_uses_only_the_v2_request_coordinate_boundary() -> None:
    from types import SimpleNamespace

    from glm52_enforcement.task11_production import Task11ProductionServices

    coordinate = build_task11_input_coordinate(
        input_kind="FENCE_EXECUTION_REQUEST",
        bucket="keep-glm52-models-246813579024-us-west-2",
        key=(
            "campaigns/glm52-sky-20260724/authorities/fence/"
            "requests/00000001/BATCH_FIVE_SOURCE_ACTIVATION.json"
        ),
        version_id="request-version/+?",
        file_sha256="a" * 64,
        body_sha256="b" * 64,
    )
    result = {
        "schema_version": 2,
        "record_type": "glm52_fence_execution_result_v2",
        "request_identity_sha256": "1" * 64,
        "manifest_identity_sha256": "2" * 64,
        "entry_identity_sha256": "3" * 64,
        "prepared_identity_sha256": "4" * 64,
        "change_set_inventory_identity_sha256": "5" * 64,
        "change_set_arn": (
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "changeSet/exact-change-set/"
            "12345678-1234-1234-1234-123456789abc"
        ),
        "stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-h1g-fence/"
            "12345678-1234-1234-1234-123456789abc"
        ),
        "slot": "BATCH_FIVE_SOURCE_ACTIVATION",
        "execute_authority_identity_sha256": "6" * 64,
        "execute_authority_class": "SUPPORT_RUNTIME",
        "execute_api_caller_role_arn": (
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-h1g-fence-executor"
        ),
        "execute_api_caller_role_id": "AROASUPPORTEXEC001",
        "expected_poststate_stack_role_arn": (
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-h1g-fence-service"
        ),
        "observed_poststate_stack_role_arn": (
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-h1g-fence-service"
        ),
        "observed_poststate_stack_role_id": "AROAFENCESERVICE01",
        "original_template_body_sha256": "7" * 64,
        "processed_template_body_sha256": "7" * 64,
        "poststate_policy_sha256": "8" * 64,
        "first_stable_snapshot_identity_sha256": "9" * 64,
        "second_stable_snapshot_identity_sha256": "9" * 64,
        "stabilization_first_evidence_sha256": "a" * 64,
        "stabilization_second_evidence_sha256": "b" * 64,
        "stabilization_identity_sha256": "c" * 64,
        "freeze_delta_classification": None,
        "batch_execution_eligible": True,
        "completed_at": "2026-07-31T01:02:03Z",
    }
    result["canonical_identity_sha256"] = canonical_sha256(result)
    service = object.__new__(Task11ProductionServices)
    service._executed_fence_result = None
    service._boundary = SimpleNamespace(inputs=(coordinate,) * 7)
    service._config = SimpleNamespace(
        fence_executor_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:keep-glm52-h1g-fence-executor:1"
        )
    )
    calls: list[dict[str, object]] = []

    def invoke_exact(
        *, version_arn: str, payload: dict[str, object]
    ) -> dict[str, object]:
        calls.append(
            {"version_arn": version_arn, "payload": payload}
        )
        return result

    service._invoke_exact = invoke_exact
    receipt = service.invoke_fence_executor_once(
        request=SimpleNamespace(generation=1),
        custody_nonce_sha256="d" * 64,
    )
    assert receipt.status == "PROVEN"
    assert calls == [
        {
            "version_arn": service._config.fence_executor_version_arn,
            "payload": {
                "request_coordinate": {
                    "bucket": coordinate.bucket,
                    "key": coordinate.key,
                    "version_id": coordinate.version_id,
                    "file_sha256": coordinate.file_sha256,
                    "canonical_identity_sha256": coordinate.body_sha256,
                }
            },
        }
    ]


def test_fence_handler_factory_uses_the_real_executor_authority_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement.fence_artifacts import ExecutorAuthorityClass
    from glm52_enforcement.task11_production import (
        _Task11FenceAwsClient,
        build_fence_handler_services,
    )

    class Events:
        def register(self, *_args: object, **_kwargs: object) -> None:
            return None

    cloudformation = SimpleNamespace(
        meta=SimpleNamespace(events=Events())
    )
    clients = {
        "s3": object(),
        "dynamodb": object(),
        "cloudformation": cloudformation,
        "iam": object(),
    }

    class Session:
        def __init__(self, *, region_name: str) -> None:
            assert region_name == "us-west-2"

        def client(self, name: str, *, config: object) -> object:
            assert config is not None
            return clients[name]

    boto3 = ModuleType("boto3")
    boto3.session = SimpleNamespace(Session=Session)  # type: ignore[attr-defined]
    botocore = ModuleType("botocore")
    botocore.__path__ = []  # type: ignore[attr-defined]
    botocore_config = ModuleType("botocore.config")
    botocore_config.Config = (  # type: ignore[attr-defined]
        lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    environment = {
        "AWS_LAMBDA_FUNCTION_NAME": (
            "keep-glm52-h1g-pre-support-fence-executor"
        ),
        "AWS_LAMBDA_FUNCTION_VERSION": "1",
        "GLM52_LEDGER_TABLE_NAME": "keep-glm52-h1g-ledger-v1",
        "GLM52_FENCE_EXECUTOR_AUTHORITY_CLASS": "RETAINED_PRE_SUPPORT",
        "GLM52_ACCOUNT_ID": ACCOUNT_ID,
        "AWS_REGION": "us-west-2",
        "GLM52_RUN_ID": RUN_ID,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    services = build_fence_handler_services(
        generation=1,
        remaining_time_millis=12_000,
    )

    assert type(services.authority_class) is ExecutorAuthorityClass
    assert (
        services.authority_class
        is ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
    )
    assert type(services.executor._client) is _Task11FenceAwsClient


def test_fence_poststate_uses_two_independent_stable_observations() -> None:
    from glm52_enforcement.task11_production import _Task11FenceAwsClient

    stack_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "stack/keep-glm52-h1g-fence/"
        "12345678-1234-1234-1234-123456789abc"
    )
    role_arn = (
        "arn:aws:iam::246813579024:role/"
        "keep-glm52-h1g-fence-service"
    )
    events: list[str] = []
    poststate_request_ids: list[str] = []

    def response(request_id: str, **body: object) -> dict[str, object]:
        return {
            **body,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": request_id,
            },
        }

    class Events:
        def register(self, *_args: object, **_kwargs: object) -> None:
            return None

    class CloudFormation:
        meta = SimpleNamespace(events=Events())

        def __init__(self) -> None:
            self.stack_reads = 0
            self.stack_policy_reads = 0
            self.template_reads = {"Original": 0, "Processed": 0}

        def execute_change_set(self, **_request: object) -> dict[str, object]:
            events.append("execute-change-set")
            return response("execute-request")

        def describe_change_set(self, **_request: object) -> dict[str, object]:
            events.append("change-set-stabilized")
            return response(
                "change-set-stabilized-request",
                Status="CREATE_COMPLETE",
                ExecutionStatus="EXECUTE_COMPLETE",
            )

        def describe_stacks(self, **_request: object) -> dict[str, object]:
            self.stack_reads += 1
            if self.stack_reads == 1:
                request_id = "execution-stabilized-stack-request"
                events.append("execution-stabilized")
            else:
                ordinal = self.stack_reads - 1
                request_id = f"poststate-stack-request-{ordinal}"
                events.append(f"poststate-observation-{ordinal}")
                poststate_request_ids.append(request_id)
            return response(
                request_id,
                Stacks=[
                    {
                        "StackId": stack_id,
                        "StackStatus": "UPDATE_COMPLETE",
                        "RoleARN": role_arn,
                        "EnableTerminationProtection": True,
                    }
                ],
            )

        def get_stack_policy(self, **_request: object) -> dict[str, object]:
            self.stack_policy_reads += 1
            request_id = (
                f"poststate-stack-policy-request-{self.stack_policy_reads}"
            )
            poststate_request_ids.append(request_id)
            return response(
                request_id,
                StackPolicyBody={
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "Update:Modify",
                            "Principal": "*",
                            "Resource": "*",
                        }
                    ]
                },
            )

        def get_template(
            self, *, TemplateStage: str, **_request: object
        ) -> dict[str, object]:
            self.template_reads[TemplateStage] += 1
            request_id = (
                f"poststate-{TemplateStage.lower()}-template-request-"
                f"{self.template_reads[TemplateStage]}"
            )
            poststate_request_ids.append(request_id)
            return response(
                request_id,
                TemplateBody={
                    "Resources": {
                        "H1gProductionFenceBucketPolicy": {
                            "Type": "AWS::S3::BucketPolicy"
                        }
                    }
                },
            )

    class Iam:
        def __init__(self) -> None:
            self.reads = 0

        def get_role(self, **_request: object) -> dict[str, object]:
            self.reads += 1
            request_id = f"poststate-role-request-{self.reads}"
            poststate_request_ids.append(request_id)
            return response(
                request_id,
                Role={"RoleId": "AROAFENCESERVICE01"},
            )

    class S3:
        def __init__(self) -> None:
            self.reads = 0

        def get_bucket_policy(self, **_request: object) -> dict[str, object]:
            self.reads += 1
            request_id = f"poststate-bucket-policy-request-{self.reads}"
            poststate_request_ids.append(request_id)
            return response(
                request_id,
                Policy=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

    cloudformation = CloudFormation()
    iam = Iam()
    s3 = S3()
    client = _Task11FenceAwsClient(
        cloudformation=cloudformation,
        s3=s3,
        iam=iam,
    )

    client.execute_change_set(
        ChangeSetName=(
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "changeSet/prepare/12345678-1234-1234-1234-123456789abc"
        ),
        StackName=stack_id,
        ClientRequestToken="f" * 64,
    )
    observed = client.observe_fence_poststate(
        StackName=stack_id,
        LogicalResourceId="H1gProductionFenceBucketPolicy",
        ExpectedBucketOwner=ACCOUNT_ID,
    )

    assert events[:3] == [
        "execute-change-set",
        "change-set-stabilized",
        "execution-stabilized",
    ]
    assert events[3:] == [
        "poststate-observation-1",
        "poststate-observation-2",
    ]
    assert cloudformation.stack_reads == 3
    assert cloudformation.stack_policy_reads == 2
    assert cloudformation.template_reads == {
        "Original": 2,
        "Processed": 2,
    }
    assert iam.reads == 2
    assert s3.reads == 2
    assert len(poststate_request_ids) == 12
    assert len(set(poststate_request_ids)) == 12
    assert (
        observed["first_stable_snapshot_identity_sha256"]
        == observed["second_stable_snapshot_identity_sha256"]
    )
    assert (
        observed["stabilization_first_evidence_sha256"]
        != observed["stabilization_second_evidence_sha256"]
    )
