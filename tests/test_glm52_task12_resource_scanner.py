from __future__ import annotations

from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.task12_orphan_audit import (
    AUDITED_RESOURCE_TYPES,
    RetainedResource,
)


ACTIVATION_ID = "act-20260728-0001"
KEY_ARN = (
    "arn:aws:kms:us-west-2:246813579024:key/"
    "12345678-1234-4234-8234-1234567890ab"
)


def _metadata(operation: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": "request-" + operation,
        "RetryAttempts": 0,
    }


def _expected() -> tuple[RetainedResource, ...]:
    return (
        RetainedResource("LEDGER", "keep-glm52-ledger", "DDB_RETAINED"),
        RetainedResource("KMS_KEY", KEY_ARN, "KMS_RETAINED"),
        RetainedResource(
            "EVIDENCE_BUCKET",
            "keep-glm52-evidence",
            "S3_RETAINED",
        ),
        RetainedResource(
            "PRODUCTION_FENCE_STACK",
            (
                "arn:aws:cloudformation:us-west-2:246813579024:"
                "stack/keep-glm52-fence/"
                "11111111-2222-4333-8444-555555555555"
            ),
            "CFN_RETAINED",
        ),
        RetainedResource(
            "LIFECYCLE_RESOURCE",
            (
                "arn:aws:states:us-west-2:246813579024:"
                "stateMachine:keep-glm52-retained"
            ),
            "STEP_FUNCTIONS_RETAINED",
        ),
        RetainedResource(
            "SOURCE_PUBLISHER_IDENTITY",
            (
                "arn:aws:iam::246813579024:"
                "role/keep-glm52-source-publisher"
            ),
            "IAM_RETAINED",
        ),
    )


def _authority() -> dict[str, object]:
    inventory = {
        "stack_resource_readback": {
            "resources": [
                {
                    "LogicalResourceId": "CombinedHostDataVolume",
                    "PhysicalResourceId": "vol-predelete",
                    "ResourceArn": None,
                }
            ]
        }
    }
    return {
        "activation_id": ACTIVATION_ID,
        "retained_kms_key_arn": KEY_ARN,
        "expected_retained": tuple(
            {
                "resource_type": item.resource_type,
                "resource_id": item.resource_id,
                "cost_class": item.cost_class,
            }
            for item in _expected()
        ),
        "predelete_physical_inventory": inventory,
        "predelete_physical_inventory_identity_sha256": canonical_sha256(
            inventory
        ),
    }


class _Clients:
    def __init__(
        self,
        *,
        orphan_volume: bool = False,
        broken_pagination: bool = False,
        paginated_buckets: bool = False,
    ) -> None:
        self.orphan_volume = orphan_volume
        self.broken_pagination = broken_pagination
        self.paginated_buckets = paginated_buckets
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def __call__(self, service: str) -> object:
        return SimpleNamespace(
            **{
                operation: self._method(service, operation)
                for operation in (
                    "describe_instances",
                    "describe_volumes",
                    "describe_snapshots",
                    "describe_addresses",
                    "describe_nat_gateways",
                    "describe_network_interfaces",
                    "describe_security_groups",
                    "describe_subnets",
                    "describe_route_tables",
                    "describe_vpc_endpoints",
                    "list_functions",
                    "list_versions_by_function",
                    "get_policy",
                    "list_roles",
                    "list_instance_profiles",
                    "list_policies",
                    "list_state_machines",
                    "list_state_machine_versions",
                    "list_executions",
                    "list_schedules",
                    "list_rules",
                    "describe_alarms",
                    "describe_log_groups",
                    "list_queues",
                    "list_buckets",
                    "list_secrets",
                    "list_grants",
                    "describe_table",
                    "describe_key",
                    "get_bucket_versioning",
                    "describe_stacks",
                    "describe_state_machine",
                    "get_schedule",
                    "get_role",
                )
            }
        )

    def _method(self, service: str, operation: str):
        def call(**request: object) -> dict[str, object]:
            self.calls.append((service, operation, dict(request)))
            response: dict[str, object] = {
                "ResponseMetadata": _metadata(operation)
            }
            fields = {
                "describe_instances": ("Reservations", []),
                "describe_volumes": ("Volumes", []),
                "describe_snapshots": ("Snapshots", []),
                "describe_addresses": ("Addresses", []),
                "describe_nat_gateways": ("NatGateways", []),
                "describe_network_interfaces": ("NetworkInterfaces", []),
                "describe_security_groups": ("SecurityGroups", []),
                "describe_subnets": ("Subnets", []),
                "describe_route_tables": ("RouteTables", []),
                "describe_vpc_endpoints": ("VpcEndpoints", []),
                "list_functions": ("Functions", []),
                "list_versions_by_function": ("Versions", []),
                "list_roles": ("Roles", []),
                "list_instance_profiles": ("InstanceProfiles", []),
                "list_policies": ("Policies", []),
                "list_state_machines": ("stateMachines", []),
                "list_state_machine_versions": (
                    "stateMachineVersions",
                    [],
                ),
                "list_executions": ("executions", []),
                "list_schedules": ("Schedules", []),
                "list_rules": ("Rules", []),
                "describe_log_groups": ("logGroups", []),
                "list_queues": ("QueueUrls", []),
                "list_buckets": ("Buckets", []),
                "list_secrets": ("SecretList", []),
                "list_grants": ("Grants", []),
            }
            if operation in fields:
                field, value = fields[operation]
                response[field] = value
            if operation == "describe_volumes" and self.orphan_volume:
                response["Volumes"] = [
                    {
                        "VolumeId": "vol-orphan",
                        "Tags": [
                            {
                                "Key": "ActivationId",
                                "Value": ACTIVATION_ID,
                            }
                        ],
                    }
                ]
            if operation == "describe_volumes" and self.broken_pagination:
                response["NextToken"] = "same-token"
            if operation == "describe_alarms":
                response.update(MetricAlarms=[], CompositeAlarms=[])
            if operation == "list_grants":
                response["Truncated"] = False
            if operation == "list_buckets" and self.paginated_buckets:
                if "ContinuationToken" not in request:
                    response["Buckets"] = [{"Name": "foreign-first-page"}]
                    response["ContinuationToken"] = "bucket-page-2"
                else:
                    assert request["ContinuationToken"] == "bucket-page-2"
                    response["Buckets"] = [{"Name": "foreign-second-page"}]
            if operation == "describe_table":
                response["Table"] = {"TableName": "keep-glm52-ledger"}
            if operation == "describe_key":
                response["KeyMetadata"] = {"Arn": KEY_ARN}
            if operation == "get_bucket_versioning":
                response["Status"] = "Enabled"
            if operation == "describe_stacks":
                response["Stacks"] = [
                    {"StackId": _expected()[3].resource_id}
                ]
            if operation == "describe_state_machine":
                response["stateMachineArn"] = _expected()[4].resource_id
            if operation == "get_role":
                response["Role"] = {"Arn": _expected()[5].resource_id}
            return response

        return call


def test_scanner_covers_all_typed_families_and_authenticates_retained() -> None:
    """Break caught: an unsupported family is silently projected empty."""

    from glm52_enforcement.task12_resource_scanner import (
        scan_task12_resources,
    )

    clients = _Clients()
    result = scan_task12_resources(
        clients=clients,
        authority=_authority(),
        observed_at="2026-07-29T12:00:00Z",
    )
    assert result.covered_resource_types == AUDITED_RESOURCE_TYPES
    assert result.resources == _expected()
    assert result.pagination_complete is True


def test_scanner_consumes_every_s3_bucket_page() -> None:
    """Break caught: ListBuckets continuation is rejected instead of consumed."""

    from glm52_enforcement.task12_resource_scanner import (
        scan_task12_resources,
    )

    clients = _Clients(paginated_buckets=True)
    result = scan_task12_resources(
        clients=clients,
        authority=_authority(),
        observed_at="2026-07-29T12:00:00Z",
    )
    requests = [
        request
        for service, operation, request in clients.calls
        if service == "s3" and operation == "list_buckets"
    ]
    assert requests == [
        {"BucketRegion": "us-west-2", "MaxBuckets": 10000},
        {
            "BucketRegion": "us-west-2",
            "MaxBuckets": 10000,
            "ContinuationToken": "bucket-page-2",
        },
    ]
    assert result.pagination_complete is True


def test_scanner_surfaces_orphan_and_rejects_incomplete_pagination() -> None:
    """Break caught: a tagged remaining resource or truncated page is ignored."""

    from glm52_enforcement.task12_resource_scanner import (
        Task12ResourceScanError,
        scan_task12_resources,
    )

    result = scan_task12_resources(
        clients=_Clients(orphan_volume=True),
        authority=_authority(),
        observed_at="2026-07-29T12:00:00Z",
    )
    assert result.resources[-1] == RetainedResource(
        "EBS_VOLUME",
        "vol-orphan",
        "UNEXPECTED_REMAINING_RESOURCE",
    )
    with pytest.raises(Task12ResourceScanError, match="pagination"):
        scan_task12_resources(
            clients=_Clients(broken_pagination=True),
            authority=_authority(),
            observed_at="2026-07-29T12:00:00Z",
        )
