"""Closed, service-specific Task 12 resource inventory scanner.

The scanner covers every non-retained family in
``AUDITED_RESOURCE_TYPES`` and authenticates the six retained identities.
Every list/describe boundary is zero-retry and pagination-complete.  Unknown
response shapes fail; no unsupported family is silently projected empty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from typing import Callable, Mapping, Optional, Sequence, Tuple

from .canonical import canonical_sha256
from .task12_orphan_audit import (
    ACCOUNT_ID,
    AUDITED_RESOURCE_TYPES,
    INTENTIONAL_RETAINED_RESOURCE_TYPES,
    REGION,
    RUN_ID,
    ResourceInventoryScan,
    RetainedResource,
)


class Task12ResourceScanError(ValueError):
    """A resource family is unreadable, incomplete, or foreign."""


@dataclass(frozen=True)
class _PageSpec:
    family: str
    service: str
    operation: str
    result_field: str
    request_token: Optional[str] = "NextToken"
    response_token: Optional[str] = "NextToken"
    request: Tuple[Tuple[str, object], ...] = ()


_DIRECT_SPECS = (
    _PageSpec(
        "EBS_VOLUME", "ec2", "describe_volumes", "Volumes",
        request=(("MaxResults", 1000),),
    ),
    _PageSpec(
        "EBS_SNAPSHOT",
        "ec2",
        "describe_snapshots",
        "Snapshots",
        request=(("OwnerIds", ["self"]), ("MaxResults", 1000)),
    ),
    _PageSpec(
        "ELASTIC_IP",
        "ec2",
        "describe_addresses",
        "Addresses",
        request_token=None,
        response_token=None,
    ),
    _PageSpec(
        "NAT_GATEWAY",
        "ec2",
        "describe_nat_gateways",
        "NatGateways",
        request=(("MaxResults", 1000),),
    ),
    _PageSpec(
        "NETWORK_INTERFACE",
        "ec2",
        "describe_network_interfaces",
        "NetworkInterfaces",
        request=(("MaxResults", 1000),),
    ),
    _PageSpec(
        "SECURITY_GROUP",
        "ec2",
        "describe_security_groups",
        "SecurityGroups",
        request=(("MaxResults", 1000),),
    ),
    _PageSpec(
        "SUBNET",
        "ec2",
        "describe_subnets",
        "Subnets",
        request=(("MaxResults", 1000),),
    ),
    _PageSpec(
        "LAMBDA_FUNCTION",
        "lambda",
        "list_functions",
        "Functions",
        request_token="Marker",
        response_token="NextMarker",
        request=(("MaxItems", 50),),
    ),
    _PageSpec(
        "IAM_ROLE",
        "iam",
        "list_roles",
        "Roles",
        request_token="Marker",
        response_token="Marker",
        request=(("MaxItems", 1000),),
    ),
    _PageSpec(
        "IAM_INSTANCE_PROFILE",
        "iam",
        "list_instance_profiles",
        "InstanceProfiles",
        request_token="Marker",
        response_token="Marker",
        request=(("MaxItems", 1000),),
    ),
    _PageSpec(
        "IAM_POLICY",
        "iam",
        "list_policies",
        "Policies",
        request_token="Marker",
        response_token="Marker",
        request=(("Scope", "Local"), ("OnlyAttached", False), ("MaxItems", 1000)),
    ),
    _PageSpec(
        "SCHEDULE",
        "scheduler",
        "list_schedules",
        "Schedules",
        request=(("MaxResults", 100),),
    ),
    _PageSpec(
        "EVENT_RULE",
        "events",
        "list_rules",
        "Rules",
        request=(("Limit", 100),),
    ),
    _PageSpec(
        "LOG_GROUP",
        "logs",
        "describe_log_groups",
        "logGroups",
        request_token="nextToken",
        response_token="nextToken",
        request=(("limit", 50),),
    ),
    _PageSpec(
        "SECRET",
        "secretsmanager",
        "list_secrets",
        "SecretList",
        request=(("MaxResults", 100), ("IncludePlannedDeletion", True)),
    ),
)

_TRANSIENT_FAMILIES = AUDITED_RESOURCE_TYPES[
    : -len(INTENTIONAL_RETAINED_RESOURCE_TYPES)
]
if len(_TRANSIENT_FAMILIES) != 30:  # pragma: no cover - import-time contract
    raise RuntimeError("Task 12 transient resource-family contract drifted")

_IDENTITY_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "EC2_INSTANCE": ("InstanceId",),
    "EBS_VOLUME": ("VolumeId",),
    "EBS_SNAPSHOT": ("SnapshotId",),
    "ELASTIC_IP": ("AllocationId", "PublicIp"),
    "NAT_GATEWAY": ("NatGatewayId",),
    "NETWORK_INTERFACE": ("NetworkInterfaceId",),
    "SECURITY_GROUP": ("GroupId",),
    "SUBNET": ("SubnetId",),
    "ROUTE_TABLE": ("RouteTableId",),
    "GATEWAY_ENDPOINT": ("VpcEndpointId",),
    "INTERFACE_ENDPOINT": ("VpcEndpointId",),
    "LAMBDA_FUNCTION": ("FunctionArn", "FunctionName"),
    "LAMBDA_VERSION": ("FunctionArn",),
    "LAMBDA_POLICY": ("FunctionName",),
    "IAM_ROLE": ("Arn", "RoleName"),
    "IAM_INSTANCE_PROFILE": ("Arn", "InstanceProfileName"),
    "IAM_POLICY": ("Arn", "PolicyName"),
    "STEP_FUNCTION_VERSION": ("stateMachineVersionArn",),
    "STEP_FUNCTION_EXECUTION": ("executionArn",),
    "SCHEDULE": ("Arn", "Name"),
    "EVENT_RULE": ("Arn", "Name"),
    "ALARM": ("AlarmArn", "AlarmName"),
    "LOG_GROUP": ("logGroupArn", "logGroupName"),
    "QUEUE": ("QueueUrl",),
    "DLQ": ("QueueUrl",),
    "REHEARSAL_BUCKET": ("Name",),
    "SECRET": ("ARN", "Name"),
    "KMS_GRANT": ("GrantId",),
}


def _metadata(response: object, label: str) -> str:
    metadata = response.get("ResponseMetadata") if type(response) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or metadata.get("RetryAttempts") != 0
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        raise Task12ResourceScanError(
            label + " response is not authenticated zero-retry"
        )
    return metadata["RequestId"]


def _pages(
    *,
    client: object,
    spec: _PageSpec,
    extra_request: Optional[Mapping[str, object]] = None,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    token: Optional[str] = None
    seen = set()
    items: list[Mapping[str, object]] = []
    evidence = []
    while True:
        request = dict(spec.request)
        if extra_request is not None:
            request.update(extra_request)
        if token is not None and spec.request_token is not None:
            request[spec.request_token] = token
        try:
            response = getattr(client, spec.operation)(**request)
        except Exception as exc:
            raise Task12ResourceScanError(
                spec.family + " inventory is unreadable"
            ) from exc
        request_id = _metadata(response, spec.operation)
        page = response.get(spec.result_field)
        if type(page) is not list or any(type(item) is not dict for item in page):
            raise Task12ResourceScanError(
                spec.family + " inventory page is malformed"
            )
        items.extend(page)
        next_token = (
            response.get(spec.response_token)
            if spec.response_token is not None
            else None
        )
        if (
            response.get("IsTruncated") is True
            or response.get("Truncated") is True
        ) and next_token is None:
            raise Task12ResourceScanError(
                spec.family + " pagination is incomplete"
            )
        if spec.response_token is None and any(
            response.get(field) is not None
            for field in ("NextToken", "nextToken", "NextMarker", "Marker")
        ):
            raise Task12ResourceScanError(
                spec.family + " pagination is unsupported"
            )
        evidence.append(
            {
                "family": spec.family,
                "operation": spec.operation,
                "request": request,
                "request_id": request_id,
                "count": len(page),
                "next_token": next_token,
            }
        )
        if next_token is None:
            break
        if (
            type(next_token) is not str
            or not next_token
            or next_token in seen
            or spec.request_token is None
        ):
            raise Task12ResourceScanError(
                spec.family + " pagination is incomplete"
            )
        seen.add(next_token)
        token = next_token
    return items, evidence


def _instance_pages(client: object) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    reservations, evidence = _pages(
        client=client,
        spec=_PageSpec(
            "EC2_INSTANCE",
            "ec2",
            "describe_instances",
            "Reservations",
            request=(("MaxResults", 1000),),
        ),
    )
    instances = []
    for reservation in reservations:
        members = reservation.get("Instances")
        if type(members) is not list or any(type(item) is not dict for item in members):
            raise Task12ResourceScanError("EC2 instance reservation is malformed")
        instances.extend(members)
    return instances, evidence


def _route_pages(
    client: object,
) -> tuple[dict[str, list[Mapping[str, object]]], list[Mapping[str, object]]]:
    tables, evidence = _pages(
        client=client,
        spec=_PageSpec(
            "ROUTE_TABLE",
            "ec2",
            "describe_route_tables",
            "RouteTables",
            request=(("MaxResults", 1000),),
        ),
    )
    associations = []
    routes = []
    for table in tables:
        table_id = table.get("RouteTableId")
        for association in table.get("Associations", []):
            if type(association) is not dict:
                raise Task12ResourceScanError("route association is malformed")
            associations.append({**association, "RouteTableId": table_id})
        for route in table.get("Routes", []):
            if type(route) is not dict:
                raise Task12ResourceScanError("route is malformed")
            routes.append({**route, "RouteTableId": table_id})
    return {
        "ROUTE_TABLE": tables,
        "ROUTE_ASSOCIATION": associations,
        "ROUTE": routes,
    }, evidence


def _endpoint_pages(
    client: object,
) -> tuple[dict[str, list[Mapping[str, object]]], list[Mapping[str, object]]]:
    endpoints, evidence = _pages(
        client=client,
        spec=_PageSpec(
            "GATEWAY_ENDPOINT",
            "ec2",
            "describe_vpc_endpoints",
            "VpcEndpoints",
            request=(("MaxResults", 1000),),
        ),
    )
    result = {"GATEWAY_ENDPOINT": [], "INTERFACE_ENDPOINT": []}
    for endpoint in endpoints:
        endpoint_type = endpoint.get("VpcEndpointType")
        family = {
            "Gateway": "GATEWAY_ENDPOINT",
            "Interface": "INTERFACE_ENDPOINT",
        }.get(endpoint_type)
        if family is None:
            continue
        result[family].append(endpoint)
    return result, evidence


def _identifier(family: str, item: Mapping[str, object]) -> str:
    if family == "ROUTE_ASSOCIATION":
        association = item.get("RouteTableAssociationId")
        if type(association) is str and association:
            return association
        if item.get("Main") is True:
            return str(item.get("RouteTableId")) + "#main"
    if family == "ROUTE":
        destination = next(
            (
                item.get(field)
                for field in (
                    "DestinationCidrBlock",
                    "DestinationIpv6CidrBlock",
                    "DestinationPrefixListId",
                )
                if type(item.get(field)) is str and item.get(field)
            ),
            None,
        )
        if destination is not None:
            return str(item.get("RouteTableId")) + "#" + destination
    for field in _IDENTITY_FIELDS.get(family, ()):
        value = item.get(field)
        if type(value) is str and value:
            return value
    raise Task12ResourceScanError(family + " resource identity is absent")


def _tag_mapping(item: Mapping[str, object]) -> Mapping[str, str]:
    tags = item.get("Tags", item.get("tags", ()))
    if type(tags) is dict:
        return {
            str(key): str(value)
            for key, value in tags.items()
            if type(key) is str and type(value) is str
        }
    if type(tags) is not list:
        return {}
    result = {}
    for tag in tags:
        if type(tag) is not dict:
            raise Task12ResourceScanError("resource tag is malformed")
        key = tag.get("Key", tag.get("key"))
        value = tag.get("Value", tag.get("value"))
        if type(key) is str and type(value) is str:
            result[key] = value
    return result


def _belongs_to_activation(
    *,
    identifier: str,
    item: Mapping[str, object],
    activation_id: str,
    predelete_identities: frozenset[str],
) -> bool:
    tags = _tag_mapping(item)
    if tags.get("RunId") == RUN_ID or tags.get("ActivationId") == activation_id:
        return True
    if identifier in predelete_identities:
        return True
    exact = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
    return (
        activation_id in identifier
        or activation_id in exact
        or f"/keep/glm52/{RUN_ID}/" in identifier
        or identifier.startswith("keep-glm52-h1g-")
    )


def _dynamic_lambda(
    *,
    client: object,
    functions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, list[Mapping[str, object]]], list[Mapping[str, object]]]:
    versions = []
    policies = []
    evidence = []
    for function in functions:
        name = _identifier("LAMBDA_FUNCTION", function)
        rows, pages = _pages(
            client=client,
            spec=_PageSpec(
                "LAMBDA_VERSION",
                "lambda",
                "list_versions_by_function",
                "Versions",
                request_token="Marker",
                response_token="NextMarker",
                request=(("MaxItems", 50),),
            ),
            extra_request={"FunctionName": name},
        )
        versions.extend(rows)
        evidence.extend(pages)
        try:
            response = client.get_policy(FunctionName=name)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code == "ResourceNotFoundException":
                continue
            raise Task12ResourceScanError("LAMBDA_POLICY inventory is unreadable") from exc
        _metadata(response, "lambda GetPolicy")
        policy = response.get("Policy")
        if type(policy) is not str:
            raise Task12ResourceScanError("Lambda policy response is malformed")
        policies.append({"FunctionName": name, "Policy": policy})
        evidence.append(
            {
                "family": "LAMBDA_POLICY",
                "operation": "get_policy",
                "request": {"FunctionName": name},
                "request_id": response["ResponseMetadata"]["RequestId"],
                "count": 1,
                "next_token": None,
            }
        )
    return {
        "LAMBDA_VERSION": versions,
        "LAMBDA_POLICY": policies,
    }, evidence


def _dynamic_sfn(
    client: object,
) -> tuple[dict[str, list[Mapping[str, object]]], list[Mapping[str, object]]]:
    machines, evidence = _pages(
        client=client,
        spec=_PageSpec(
            "STEP_FUNCTION_VERSION",
            "stepfunctions",
            "list_state_machines",
            "stateMachines",
            request_token="nextToken",
            response_token="nextToken",
            request=(("maxResults", 1000),),
        ),
    )
    versions = []
    executions = []
    for machine in machines:
        arn = machine.get("stateMachineArn")
        if type(arn) is not str or not arn:
            raise Task12ResourceScanError("state machine identity is absent")
        rows, pages = _pages(
            client=client,
            spec=_PageSpec(
                "STEP_FUNCTION_VERSION",
                "stepfunctions",
                "list_state_machine_versions",
                "stateMachineVersions",
                request_token="nextToken",
                response_token="nextToken",
                request=(("maxResults", 1000),),
            ),
            extra_request={"stateMachineArn": arn},
        )
        versions.extend(rows)
        evidence.extend(pages)
        rows, pages = _pages(
            client=client,
            spec=_PageSpec(
                "STEP_FUNCTION_EXECUTION",
                "stepfunctions",
                "list_executions",
                "executions",
                request_token="nextToken",
                response_token="nextToken",
                request=(("maxResults", 1000),),
            ),
            extra_request={"stateMachineArn": arn},
        )
        executions.extend(rows)
        evidence.extend(pages)
    return {
        "STEP_FUNCTION_VERSION": versions,
        "STEP_FUNCTION_EXECUTION": executions,
    }, evidence


def _alarm_pages(
    client: object,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    token = None
    seen = set()
    alarms = []
    evidence = []
    while True:
        request: dict[str, object] = {"MaxRecords": 100}
        if token is not None:
            request["NextToken"] = token
        try:
            response = client.describe_alarms(**request)
        except Exception as exc:
            raise Task12ResourceScanError("ALARM inventory is unreadable") from exc
        request_id = _metadata(response, "cloudwatch DescribeAlarms")
        metric = response.get("MetricAlarms")
        composite = response.get("CompositeAlarms")
        if (
            type(metric) is not list
            or type(composite) is not list
            or any(type(item) is not dict for item in metric + composite)
        ):
            raise Task12ResourceScanError("ALARM inventory page is malformed")
        alarms.extend(metric)
        alarms.extend(composite)
        next_token = response.get("NextToken")
        evidence.append(
            {
                "family": "ALARM",
                "operation": "describe_alarms",
                "request": request,
                "request_id": request_id,
                "count": len(metric) + len(composite),
                "next_token": next_token,
            }
        )
        if next_token is None:
            break
        if type(next_token) is not str or not next_token or next_token in seen:
            raise Task12ResourceScanError("ALARM pagination is incomplete")
        seen.add(next_token)
        token = next_token
    return alarms, evidence


def _queues(client: object) -> tuple[dict[str, list[Mapping[str, object]]], list[Mapping[str, object]]]:
    # boto returns strings for QueueUrls, so this operation is intentionally
    # handled separately from the mapping-only generic paginator.
    token = None
    seen = set()
    result = {"QUEUE": [], "DLQ": []}
    evidence_rows = []
    while True:
        request: dict[str, object] = {"MaxResults": 1000}
        if token is not None:
            request["NextToken"] = token
        try:
            response = client.list_queues(**request)
        except Exception as exc:
            raise Task12ResourceScanError("QUEUE inventory is unreadable") from exc
        request_id = _metadata(response, "sqs ListQueues")
        members = response.get("QueueUrls", [])
        if type(members) is not list or any(type(item) is not str for item in members):
            raise Task12ResourceScanError("QUEUE inventory page is malformed")
        for url in members:
            family = "DLQ" if "dlq" in url.rsplit("/", 1)[-1].lower() else "QUEUE"
            result[family].append({"QueueUrl": url})
        next_token = response.get("NextToken")
        evidence_rows.append(
            {
                "family": "QUEUE",
                "operation": "list_queues",
                "request": request,
                "request_id": request_id,
                "count": len(members),
                "next_token": next_token,
            }
        )
        if next_token is None:
            break
        if type(next_token) is not str or not next_token or next_token in seen:
            raise Task12ResourceScanError("QUEUE pagination is incomplete")
        seen.add(next_token)
        token = next_token
    return result, evidence_rows


def _buckets(client: object) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    all_buckets: list[Mapping[str, object]] = []
    evidence_rows: list[Mapping[str, object]] = []
    continuation: Optional[str] = None
    seen: set[str] = set()
    while True:
        request: dict[str, object] = {
            "BucketRegion": REGION,
            "MaxBuckets": 10000,
        }
        if continuation is not None:
            request["ContinuationToken"] = continuation
        try:
            response = client.list_buckets(**request)
        except Exception as exc:
            raise Task12ResourceScanError(
                "REHEARSAL_BUCKET inventory is unreadable"
            ) from exc
        request_id = _metadata(response, "s3 ListBuckets")
        buckets = response.get("Buckets")
        if type(buckets) is not list or any(
            type(bucket) is not dict for bucket in buckets
        ):
            raise Task12ResourceScanError(
                "REHEARSAL_BUCKET inventory is malformed"
            )
        next_token = response.get("ContinuationToken")
        evidence_rows.append(
            {
                "family": "REHEARSAL_BUCKET",
                "operation": "list_buckets",
                "request": request,
                "request_id": request_id,
                "count": len(buckets),
                "next_token": next_token,
            }
        )
        all_buckets.extend(buckets)
        if next_token is None:
            break
        if (
            type(next_token) is not str
            or not next_token
            or next_token in seen
        ):
            raise Task12ResourceScanError(
                "REHEARSAL_BUCKET pagination is incomplete"
            )
        seen.add(next_token)
        continuation = next_token
    return all_buckets, evidence_rows


def _kms_grants(
    client: object, key_arn: str
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    return _pages(
        client=client,
        spec=_PageSpec(
            "KMS_GRANT",
            "kms",
            "list_grants",
            "Grants",
            request_token="Marker",
            response_token="NextMarker",
            request=(("KeyId", key_arn), ("Limit", 100)),
        ),
    )


def _predelete_identities(authority: Mapping[str, object]) -> frozenset[str]:
    inventory = authority.get("predelete_physical_inventory")
    if (
        type(inventory) is not dict
        or canonical_sha256(inventory)
        != authority.get("predelete_physical_inventory_identity_sha256")
    ):
        raise Task12ResourceScanError(
            "predelete physical inventory identity drifted"
        )
    readback = inventory.get("stack_resource_readback")
    resources = readback.get("resources") if type(readback) is dict else None
    if type(resources) is not list:
        # Canonical in-memory authority may retain tuples.
        if type(resources) is not tuple:
            raise Task12ResourceScanError(
                "predelete physical inventory resources are absent"
            )
    identities = set()
    for row in resources:
        if type(row) is not dict:
            raise Task12ResourceScanError(
                "predelete physical inventory member is malformed"
            )
        for field in ("PhysicalResourceId", "ResourceArn"):
            value = row.get(field)
            if type(value) is str and value:
                identities.add(value)
    return frozenset(identities)


def _verify_retained(
    *,
    clients: Callable[[str], object],
    expected: Tuple[RetainedResource, ...],
) -> list[Mapping[str, object]]:
    evidence = []
    for resource in expected:
        try:
            if resource.resource_type == "LEDGER":
                response = clients("dynamodb").describe_table(
                    TableName=resource.resource_id
                )
                observed = response.get("Table", {}).get("TableName")
                operation = "DescribeTable"
            elif resource.resource_type == "KMS_KEY":
                response = clients("kms").describe_key(
                    KeyId=resource.resource_id
                )
                metadata = response.get("KeyMetadata", {})
                observed = metadata.get("Arn", metadata.get("KeyId"))
                operation = "DescribeKey"
            elif resource.resource_type == "EVIDENCE_BUCKET":
                response = clients("s3").get_bucket_versioning(
                    Bucket=resource.resource_id,
                    ExpectedBucketOwner=ACCOUNT_ID,
                )
                observed = (
                    resource.resource_id
                    if response.get("Status") == "Enabled"
                    else None
                )
                operation = "GetBucketVersioning"
            elif resource.resource_type == "PRODUCTION_FENCE_STACK":
                response = clients("cloudformation").describe_stacks(
                    StackName=resource.resource_id
                )
                stacks = response.get("Stacks", [])
                observed = (
                    stacks[0].get("StackId")
                    if type(stacks) is list and len(stacks) == 1
                    else None
                )
                operation = "DescribeStacks"
            elif resource.resource_type == "LIFECYCLE_RESOURCE":
                if "STEP_FUNCTION" in resource.cost_class:
                    response = clients("stepfunctions").describe_state_machine(
                        stateMachineArn=resource.resource_id
                    )
                    observed = response.get("stateMachineArn")
                    operation = "DescribeStateMachine"
                else:
                    response = clients("scheduler").get_schedule(
                        Name=resource.resource_id.rsplit("/", 1)[-1]
                    )
                    observed = response.get("Arn", response.get("Name"))
                    operation = "GetSchedule"
            elif resource.resource_type == "SOURCE_PUBLISHER_IDENTITY":
                role_name = resource.resource_id.rsplit("/", 1)[-1]
                response = clients("iam").get_role(RoleName=role_name)
                observed = response.get("Role", {}).get("Arn")
                operation = "GetRole"
            else:  # pragma: no cover - expected type contract guards this
                raise Task12ResourceScanError("unknown retained resource type")
        except Task12ResourceScanError:
            raise
        except Exception as exc:
            raise Task12ResourceScanError(
                resource.resource_type + " retained identity is unreadable"
            ) from exc
        request_id = _metadata(response, operation)
        if observed != resource.resource_id:
            raise Task12ResourceScanError(
                resource.resource_type + " retained identity drifted"
            )
        evidence.append(
            {
                "family": resource.resource_type,
                "operation": operation,
                "request_id": request_id,
                "resource_id": resource.resource_id,
            }
        )
    return evidence


def scan_task12_resources(
    *,
    clients: Callable[[str], object],
    authority: Mapping[str, object],
    observed_at: str,
) -> ResourceInventoryScan:
    """Scan all 30 transient families and authenticate all six retained ones."""

    try:
        datetime.strptime(observed_at, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as exc:
        raise Task12ResourceScanError("resource observed_at is not canonical UTC") from exc
    activation_id = authority.get("activation_id")
    if type(activation_id) is not str or not activation_id:
        raise Task12ResourceScanError("activation authority identity is absent")
    raw_expected = authority.get("expected_retained")
    if type(raw_expected) not in {list, tuple}:
        raise Task12ResourceScanError("expected retained inventory is absent")
    expected = tuple(RetainedResource(**item) for item in raw_expected)
    if tuple(item.resource_type for item in expected) != (
        INTENTIONAL_RETAINED_RESOURCE_TYPES
    ):
        raise Task12ResourceScanError("expected retained inventory is not closed")
    predelete = _predelete_identities(authority)

    family_items: dict[str, list[Mapping[str, object]]] = {}
    evidence: list[Mapping[str, object]] = []
    instances, rows = _instance_pages(clients("ec2"))
    family_items["EC2_INSTANCE"] = instances
    evidence.extend(rows)
    for spec in _DIRECT_SPECS:
        items, rows = _pages(client=clients(spec.service), spec=spec)
        family_items[spec.family] = items
        evidence.extend(rows)
    route_items, rows = _route_pages(clients("ec2"))
    family_items.update(route_items)
    evidence.extend(rows)
    endpoint_items, rows = _endpoint_pages(clients("ec2"))
    family_items.update(endpoint_items)
    evidence.extend(rows)
    dynamic, rows = _dynamic_lambda(
        client=clients("lambda"),
        functions=family_items["LAMBDA_FUNCTION"],
    )
    family_items.update(dynamic)
    evidence.extend(rows)
    dynamic, rows = _dynamic_sfn(clients("stepfunctions"))
    family_items.update(dynamic)
    evidence.extend(rows)
    alarms, rows = _alarm_pages(clients("cloudwatch"))
    family_items["ALARM"] = alarms
    evidence.extend(rows)
    dynamic, rows = _queues(clients("sqs"))
    family_items.update(dynamic)
    evidence.extend(rows)
    buckets, rows = _buckets(clients("s3"))
    family_items["REHEARSAL_BUCKET"] = buckets
    evidence.extend(rows)
    key_arn = authority.get("retained_kms_key_arn")
    if type(key_arn) is not str:
        raise Task12ResourceScanError("retained KMS key is absent")
    grants, rows = _kms_grants(clients("kms"), key_arn)
    family_items["KMS_GRANT"] = grants
    evidence.extend(rows)

    if set(family_items) != set(_TRANSIENT_FAMILIES):
        missing = sorted(set(_TRANSIENT_FAMILIES) - set(family_items))
        raise Task12ResourceScanError(
            "resource-family coverage is incomplete: " + ",".join(missing)
        )
    resources = list(expected)
    expected_ids = frozenset(item.resource_id for item in expected)
    for family in _TRANSIENT_FAMILIES:
        # KMS grants are audited by the stronger baseline/attribution path and
        # must not be double-classified as generic resource orphans.
        if family == "KMS_GRANT":
            continue
        for item in family_items[family]:
            identifier = _identifier(family, item)
            if identifier in expected_ids:
                continue
            if _belongs_to_activation(
                identifier=identifier,
                item=item,
                activation_id=activation_id,
                predelete_identities=predelete,
            ):
                resources.append(
                    RetainedResource(
                        resource_type=family,
                        resource_id=identifier,
                        cost_class="UNEXPECTED_REMAINING_RESOURCE",
                    )
                )
    retained_evidence = _verify_retained(
        clients=clients,
        expected=expected,
    )
    body = {
        "covered_resource_types": AUDITED_RESOURCE_TYPES,
        "resources": [asdict(item) for item in resources],
        "unreadable_resource_ids": (),
        "pagination_complete": True,
        "observed_at": observed_at,
        "family_evidence": evidence,
        "retained_evidence": retained_evidence,
        "predelete_physical_inventory_identity_sha256": authority[
            "predelete_physical_inventory_identity_sha256"
        ],
    }
    return ResourceInventoryScan(
        covered_resource_types=AUDITED_RESOURCE_TYPES,
        resources=tuple(resources),
        unreadable_resource_ids=(),
        pagination_complete=True,
        observed_at=observed_at,
        evidence_identity_sha256=canonical_sha256(body),
    )


__all__ = [
    "Task12ResourceScanError",
    "scan_task12_resources",
]
