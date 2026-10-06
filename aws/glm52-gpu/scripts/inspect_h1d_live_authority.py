#!/usr/bin/env python3
"""Execute the fresh read-only H.1d AWS and spend authority walk."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import (
    FIRST_EXCEPTION,
    Future,
    ThreadPoolExecutor,
    wait,
)
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time
from threading import Event, Lock
from typing import Callable, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from glm52_enforcement.live_authority import (  # noqa: E402
    ACCOUNT_ID,
    PROFILE,
    REGION,
    RUN_ID,
    CallerIdentityObservation,
    H1dLiveAuthorityRequest,
    H1dLiveServices,
    LiveReadPage,
    LiveReadSpec,
    build_expected_state_authentication,
    h1d_expected_state_from_mapping,
    inspect_h1d_live_authority,
    serialize_non_authoritative_evidence,
)
from glm52_enforcement.spend_authority import (  # noqa: E402
    ReserveListPage,
    SpendAuthorityServices,
    SpendListPage,
    SpendObject,
    inspect_spend_authority,
    spend_authority_request_from_mapping,
)
from glm52_enforcement.sky_admission import (  # noqa: E402
    AttestationRequest,
    AttestationResult,
    Task9SkyRelayProbe,
)
from glm52_enforcement.task9_contract import (  # noqa: E402
    build_task9_contract,
    validate_task9_deployed_identity,
)
from glm52_enforcement.task11_support_boundary import (  # noqa: E402
    exact_input_coordinate_from_mapping,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INSTANCE_ID = re.compile(r"i-(?:[0-9a-f]{8}|[0-9a-f]{17})\Z")
_IAM_ROLE_ID = re.compile(r"AROA[A-Z0-9]{16,128}\Z")

# Closed inventory for the only live reader. Every verb is mutation-free.
READ_COMMANDS: Mapping[str, Tuple[Tuple[str, ...], ...]] = {
    "cloudformation": (
        ("cloudformation", "describe-stacks"),
        ("cloudformation", "get-template"),
        ("cloudformation", "list-stack-resources"),
    ),
    "lambda": (
        ("lambda", "list-functions"),
        ("lambda", "get-function"),
        ("lambda", "get-policy"),
        ("lambda", "get-function-concurrency"),
        ("lambda", "list-versions-by-function"),
        ("lambda", "list-event-source-mappings"),
    ),
    "iam": (
        ("iam", "list-roles"),
        ("iam", "get-role"),
        ("iam", "list-instance-profiles-for-role"),
        ("iam", "list-role-policies"),
        ("iam", "get-role-policy"),
        ("iam", "list-attached-role-policies"),
        ("iam", "get-policy"),
        ("iam", "get-policy-version"),
    ),
    "eventbridge": (
        ("events", "list-rules"),
        ("events", "describe-rule"),
        ("events", "list-targets-by-rule"),
    ),
    "scheduler": (
        ("scheduler", "list-schedules"),
        ("scheduler", "get-schedule"),
    ),
    "sqs": (
        ("sqs", "list-queues"),
        ("sqs", "get-queue-attributes"),
    ),
    "sns": (
        ("sns", "list-subscriptions-by-topic"),
        ("sns", "get-subscription-attributes"),
    ),
    "ec2": (
        ("ec2", "describe-instances"),
        ("ec2", "describe-volumes"),
        ("ec2", "describe-images"),
        ("ec2", "describe-instance-attribute"),
    ),
    "s3": (
        ("s3api", "list-buckets"),
        ("s3api", "get-bucket-versioning"),
        ("s3api", "get-bucket-policy"),
        ("s3api", "get-bucket-lifecycle-configuration"),
        ("s3api", "get-bucket-replication"),
    ),
    "dynamodb": (("dynamodb", "query"),),
    "ssm": (("ssm", "describe-instance-information"),),
    "cloudwatch": (
        ("cloudwatch", "describe-alarms"),
        ("cloudwatch", "get-metric-data"),
    ),
    "logs": (("logs", "describe-log-groups"),),
}

_DETAIL_PAGINATION: Mapping[
    tuple[str, int],
    tuple[Tuple[str, ...], Tuple[str, ...], str],
] = {
    ("cloudformation", 2): (
        ("StackResourceSummaries",),
        ("NextToken",),
        "--next-token",
    ),
    ("lambda", 4): (("Versions",), ("NextMarker",), "--marker"),
    ("lambda", 5): (
        ("EventSourceMappings",),
        ("NextMarker",),
        "--marker",
    ),
    ("iam", 2): (("InstanceProfiles",), ("Marker",), "--marker"),
    ("iam", 3): (("PolicyNames",), ("Marker",), "--marker"),
    ("iam", 5): (("AttachedPolicies",), ("Marker",), "--marker"),
    ("eventbridge", 2): (("Targets",), ("NextToken",), "--next-token"),
    ("ec2", 1): (("Volumes",), ("NextToken",), "--next-token"),
    ("ec2", 2): (("Images",), ("NextToken",), "--next-token"),
    ("cloudwatch", 1): (
        ("MetricDataResults",),
        ("NextToken",),
        "--next-token",
    ),
}

_INVENTORY_PAGINATION: Mapping[
    str,
    tuple[Tuple[str, ...], Optional[Tuple[str, ...]], Optional[str]],
] = {
    "cloudformation": (("Stacks",), ("NextToken",), "--next-token"),
    "lambda": (("Functions",), ("NextMarker",), "--marker"),
    "iam": (("Roles",), ("Marker",), "--marker"),
    "eventbridge": (("Rules",), ("NextToken",), "--next-token"),
    "scheduler": (("Schedules",), ("NextToken",), "--next-token"),
    "sqs": (("QueueUrls",), ("NextToken",), "--next-token"),
    "sns": (("Subscriptions",), ("NextToken",), "--next-token"),
    "ec2": (("Instances",), ("NextToken",), "--next-token"),
    "s3": (("Buckets",), None, None),
    "dynamodb": (
        ("Items",),
        ("LastEvaluatedKey",),
        "--exclusive-start-key",
    ),
    "ssm": (
        ("InstanceInformationList",),
        ("NextToken",),
        "--next-token",
    ),
    "cloudwatch": (("MetricAlarms",), ("NextToken",), "--next-token"),
    "logs": (("logGroups",), ("nextToken",), "--next-token"),
}

_INVENTORY_IDENTITY_PATHS: Mapping[str, Tuple[object, ...]] = {
    "cloudformation": ("StackId",),
    "lambda": ("FunctionArn",),
    "iam": ("Arn",),
    "eventbridge": ("Arn",),
    "scheduler": ("Arn",),
    "sqs": (),
    "sns": ("SubscriptionArn",),
    "ec2": ("InstanceId",),
    "s3": ("Name",),
    "dynamodb": (),
    "ssm": ("InstanceId",),
    "cloudwatch": ("AlarmArn",),
    "logs": ("arn",),
}

_TRUSTED_SOURCE_TABLE = "keep-glm52-h1g-ledger-v1"
_TRUSTED_SOURCE_SORT_KEY = "H1D_TRUSTED_SOURCE"
_CAMPAIGN_BUCKET = "keep-glm52-models-246813579024-us-west-2"
_AUTHORITY_SOURCE_NAMES = {
    "task6_manifest",
    "task6_templates",
    "task7_postcreate_manifest",
    "task7_inventory",
}
SDK_MAX_WORKERS = 14

_LAMBDA_CONFIGURATION_FIELDS = frozenset(
    {
        "DeadLetterConfig",
        "Description",
        "Environment",
        "Handler",
        "KmsKeyArn",
        "Layers",
        "MemorySize",
        "Role",
        "Runtime",
        "Timeout",
        "TracingConfig",
        "VpcConfig",
    }
)
_LAMBDA_EVENT_SOURCE_MAPPING_FIELDS = frozenset(
    {
        "BatchSize",
        "BisectBatchOnFunctionError",
        "DestinationConfig",
        "Enabled",
        "EventSourceArn",
        "FilterCriteria",
        "FunctionResponseTypes",
        "MaximumBatchingWindowInSeconds",
        "MaximumRecordAgeInSeconds",
        "MaximumRetryAttempts",
        "ParallelizationFactor",
        "Queues",
        "ScalingConfig",
        "SourceAccessConfigurations",
        "StartingPosition",
        "StartingPositionTimestamp",
        "Topics",
        "TumblingWindowInSeconds",
    }
)

_SDK_METHODS: Mapping[tuple[str, str], tuple[str, str]] = {
    ("sts", "get-caller-identity"): ("sts", "get_caller_identity"),
    ("cloudformation", "describe-stacks"): (
        "cloudformation",
        "describe_stacks",
    ),
    ("cloudformation", "get-template"): (
        "cloudformation",
        "get_template",
    ),
    ("cloudformation", "list-stack-resources"): (
        "cloudformation",
        "list_stack_resources",
    ),
    ("lambda", "list-functions"): ("lambda", "list_functions"),
    ("lambda", "get-function"): ("lambda", "get_function"),
    ("lambda", "get-policy"): ("lambda", "get_policy"),
    ("lambda", "get-function-concurrency"): (
        "lambda",
        "get_function_concurrency",
    ),
    ("lambda", "list-versions-by-function"): (
        "lambda",
        "list_versions_by_function",
    ),
    ("lambda", "list-event-source-mappings"): (
        "lambda",
        "list_event_source_mappings",
    ),
    ("iam", "list-roles"): ("iam", "list_roles"),
    ("iam", "get-role"): ("iam", "get_role"),
    ("iam", "list-instance-profiles-for-role"): (
        "iam",
        "list_instance_profiles_for_role",
    ),
    ("iam", "list-role-policies"): ("iam", "list_role_policies"),
    ("iam", "get-role-policy"): ("iam", "get_role_policy"),
    ("iam", "list-attached-role-policies"): (
        "iam",
        "list_attached_role_policies",
    ),
    ("iam", "get-policy"): ("iam", "get_policy"),
    ("iam", "get-policy-version"): ("iam", "get_policy_version"),
    ("events", "list-rules"): ("events", "list_rules"),
    ("events", "describe-rule"): ("events", "describe_rule"),
    ("events", "list-targets-by-rule"): (
        "events",
        "list_targets_by_rule",
    ),
    ("scheduler", "list-schedules"): ("scheduler", "list_schedules"),
    ("scheduler", "get-schedule"): ("scheduler", "get_schedule"),
    ("sqs", "list-queues"): ("sqs", "list_queues"),
    ("sqs", "get-queue-attributes"): ("sqs", "get_queue_attributes"),
    ("sns", "list-subscriptions-by-topic"): (
        "sns",
        "list_subscriptions_by_topic",
    ),
    ("sns", "get-subscription-attributes"): (
        "sns",
        "get_subscription_attributes",
    ),
    ("ec2", "describe-instances"): ("ec2", "describe_instances"),
    ("ec2", "describe-volumes"): ("ec2", "describe_volumes"),
    ("ec2", "describe-images"): ("ec2", "describe_images"),
    ("ec2", "describe-instance-attribute"): (
        "ec2",
        "describe_instance_attribute",
    ),
    ("s3api", "list-buckets"): ("s3", "list_buckets"),
    ("s3api", "get-bucket-versioning"): (
        "s3",
        "get_bucket_versioning",
    ),
    ("s3api", "get-bucket-policy"): ("s3", "get_bucket_policy"),
    ("s3api", "get-bucket-lifecycle-configuration"): (
        "s3",
        "get_bucket_lifecycle_configuration",
    ),
    ("s3api", "get-bucket-replication"): (
        "s3",
        "get_bucket_replication",
    ),
    ("s3api", "list-object-versions"): (
        "s3",
        "list_object_versions",
    ),
    ("dynamodb", "query"): ("dynamodb", "query"),
    ("dynamodb", "get-item"): ("dynamodb", "get_item"),
    ("ssm", "describe-instance-information"): (
        "ssm",
        "describe_instance_information",
    ),
    ("cloudwatch", "describe-alarms"): (
        "cloudwatch",
        "describe_alarms",
    ),
    ("cloudwatch", "get-metric-data"): (
        "cloudwatch",
        "get_metric_data",
    ),
    ("logs", "describe-log-groups"): ("logs", "describe_log_groups"),
}

_SDK_OPTIONS: Mapping[
    str,
    tuple[str, str],
] = {
    "--stack-name": ("StackName", "string"),
    "--function-name": ("FunctionName", "string"),
    "--role-name": ("RoleName", "string"),
    "--policy-name": ("PolicyName", "string"),
    "--policy-arn": ("PolicyArn", "string"),
    "--version-id": ("VersionId", "string"),
    "--name": ("Name", "string"),
    "--rule": ("Rule", "string"),
    "--group-name": ("GroupName", "string"),
    "--queue-url": ("QueueUrl", "string"),
    "--attribute-names": ("AttributeNames", "strings"),
    "--topic-arn": ("TopicArn", "string"),
    "--subscription-arn": ("SubscriptionArn", "string"),
    "--filters": ("Filters", "filters"),
    "--name-prefix": ("NamePrefix", "string"),
    "--queue-name-prefix": ("QueueNamePrefix", "string"),
    "--alarm-name-prefix": ("AlarmNamePrefix", "string"),
    "--log-group-name-prefix": ("logGroupNamePrefix", "string"),
    "--max-results": ("MaxResults", "integer"),
    "--volume-ids": ("VolumeIds", "strings"),
    "--image-ids": ("ImageIds", "strings"),
    "--instance-id": ("InstanceId", "string"),
    "--attribute": ("Attribute", "string"),
    "--bucket": ("Bucket", "string"),
    "--table-name": ("TableName", "string"),
    "--key": ("Key", "json"),
    "--key-condition-expression": ("KeyConditionExpression", "string"),
    "--expression-attribute-values": (
        "ExpressionAttributeValues",
        "json",
    ),
    "--filter-expression": ("FilterExpression", "string"),
    "--consistent-read": ("ConsistentRead", "boolean"),
    "--return-consumed-capacity": (
        "ReturnConsumedCapacity",
        "string",
    ),
    "--exclusive-start-key": ("ExclusiveStartKey", "json"),
    "--metric-data-queries": ("MetricDataQueries", "json"),
    "--start-time": ("StartTime", "string"),
    "--end-time": ("EndTime", "string"),
    "--scan-by": ("ScanBy", "string"),
    "--prefix": ("Prefix", "string"),
    "--max-keys": ("MaxKeys", "integer"),
    "--key-marker": ("KeyMarker", "string"),
    "--version-id-marker": ("VersionIdMarker", "string"),
}

_SDK_TOKEN_OPTIONS: Mapping[tuple[str, str], Mapping[str, str]] = {
    ("cloudformation", "describe-stacks"): {"--next-token": "NextToken"},
    ("cloudformation", "list-stack-resources"): {
        "--next-token": "NextToken"
    },
    ("lambda", "list-functions"): {"--marker": "Marker"},
    ("lambda", "list-versions-by-function"): {"--marker": "Marker"},
    ("lambda", "list-event-source-mappings"): {"--marker": "Marker"},
    ("iam", "list-roles"): {"--marker": "Marker"},
    ("iam", "list-instance-profiles-for-role"): {"--marker": "Marker"},
    ("iam", "list-role-policies"): {"--marker": "Marker"},
    ("iam", "list-attached-role-policies"): {"--marker": "Marker"},
    ("events", "list-rules"): {"--next-token": "NextToken"},
    ("events", "list-targets-by-rule"): {"--next-token": "NextToken"},
    ("scheduler", "list-schedules"): {"--next-token": "NextToken"},
    ("sqs", "list-queues"): {"--next-token": "NextToken"},
    ("sns", "list-subscriptions-by-topic"): {
        "--next-token": "NextToken"
    },
    ("ec2", "describe-instances"): {"--next-token": "NextToken"},
    ("ec2", "describe-volumes"): {"--next-token": "NextToken"},
    ("ec2", "describe-images"): {"--next-token": "NextToken"},
    ("ssm", "describe-instance-information"): {
        "--next-token": "NextToken"
    },
    ("cloudwatch", "describe-alarms"): {"--next-token": "NextToken"},
    ("cloudwatch", "get-metric-data"): {"--next-token": "NextToken"},
    ("logs", "describe-log-groups"): {"--next-token": "nextToken"},
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _now().isoformat().replace("+00:00", "Z")


def _canonical_utc(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is malformed") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} is not timezone-aware")
    return (
        parsed.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validated_authority_sources(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _AUTHORITY_SOURCE_NAMES:
        raise ValueError("trusted campaign source inventory is not exact")
    copied: dict[str, object] = {}
    for name, coordinate in value.items():
        if (
            type(name) is not str
            or type(coordinate) is not dict
            or set(coordinate) != {"key", "version_id", "file_sha256"}
            or type(coordinate["key"]) is not str
            or not coordinate["key"]
            or type(coordinate["version_id"]) is not str
            or not coordinate["version_id"]
            or type(coordinate["file_sha256"]) is not str
            or _SHA256.fullmatch(coordinate["file_sha256"]) is None
        ):
            raise ValueError("trusted campaign source coordinate is malformed")
        copied[name] = dict(coordinate)
    return copied


def _trusted_source_contract_sha256(
    *,
    task6: Mapping[str, object],
    templates: Mapping[str, object],
    postcreate: Mapping[str, object],
    inventory: Mapping[str, object],
) -> str:
    documents = {
        "task6_manifest": task6,
        "task6_templates": templates,
        "task7_postcreate_manifest": postcreate,
        "task7_inventory": inventory,
    }
    expected_types = {
        "task6_manifest": "glm52_h1g_stack_migration_manifest_v1",
        "task6_templates": "glm52_h1d_task6_task7_template_bundle_v1",
        "task7_postcreate_manifest":
            "glm52_h1g_support_postcreate_manifest_v1",
        "task7_inventory":
            "glm52_h1g_support_postcreate_inventory_v1",
    }
    if any(
        documents[name].get("record_type") != record_type
        for name, record_type in expected_types.items()
    ):
        raise ValueError("source document record type is foreign")
    for document in (task6, postcreate):
        if (
            document.get("account_id") != ACCOUNT_ID
            or document.get("region") != REGION
            or document.get("run_id") != RUN_ID
        ):
            raise ValueError("source document campaign identity is foreign")
    support_template = templates.get("task7_support_template")
    if (
        type(support_template) is not dict
        or postcreate.get("support_template_body_sha256")
        != canonical_sha256(support_template)
        or postcreate.get("support_postcreate_inventory_sha256")
        != inventory.get("canonical_body_sha256")
        or inventory.get("support_template_body_sha256")
        != canonical_sha256(support_template)
        or type(postcreate.get("activation_id")) is not str
        or not postcreate["activation_id"]
        or type(task6.get("bucket_name")) is not str
        or task6["bucket_name"] != _CAMPAIGN_BUCKET
    ):
        raise ValueError("source document semantic contract diverged")
    semantic_templates = dict(templates)
    semantic_templates.pop("canonical_body_sha256", None)
    semantic_templates.pop("h1d_specs_identity_sha256", None)
    return canonical_sha256(
        {
            "task6_manifest": task6,
            "task6_templates_without_caller_h1d_specs": semantic_templates,
            "task7_postcreate_manifest": postcreate,
            "task7_inventory": inventory,
        }
    )


def _expected_read_absence(
    operation: Sequence[str],
    error_code: object,
) -> Optional[dict[str, object]]:
    command = tuple(operation[:2])
    if (
        command == ("s3api", "get-bucket-policy")
        and error_code == "NoSuchBucketPolicy"
    ):
        return {"Policy": None}
    if (
        command
        == ("s3api", "get-bucket-lifecycle-configuration")
        and error_code == "NoSuchLifecycleConfiguration"
    ):
        return {"Rules": None}
    if (
        command == ("s3api", "get-bucket-replication")
        and error_code == "ReplicationConfigurationNotFoundError"
    ):
        return {"ReplicationConfiguration": None}
    if (
        command == ("lambda", "get-policy")
        and error_code == "ResourceNotFoundException"
    ):
        return {"Policy": None}
    return None


class AwsCliCommandRunner:
    """AWS CLI subprocess boundary with one attempt and a hard timeout."""

    def __init__(
        self,
        *,
        profile: str = PROFILE,
        region: str = REGION,
        subprocess_run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        if profile != PROFILE or region != REGION:
            raise ValueError("AWS CLI runner requires the exact profile and region")
        self.profile = profile
        self.region = region
        self._subprocess_run = subprocess_run

    def _environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "AWS_PAGER": "",
            "AWS_MAX_ATTEMPTS": "1",
            "AWS_RETRY_MODE": "standard",
            "AWS_DEFAULT_REGION": REGION,
        }

    def run_json(
        self,
        operation: Tuple[str, ...],
        *,
        timeout_seconds: int,
    ) -> dict[str, object]:
        if (
            type(operation) is not tuple
            or not operation
            or any(type(part) is not str or not part for part in operation)
        ):
            raise ValueError("AWS CLI operation must be a closed tuple")
        command = [
            "aws",
            "--profile",
            self.profile,
            "--region",
            self.region,
            *operation,
            "--no-cli-pager",
            "--output",
            "json",
        ]
        try:
            result = self._subprocess_run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                env=self._environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("AWS CLI read timed out") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            matched = re.search(
                r"An error occurred \(([^)]+)\)",
                detail,
            )
            expected_absence = _expected_read_absence(
                operation,
                None if matched is None else matched.group(1),
            )
            if expected_absence is not None:
                return expected_absence
            raise ValueError(detail or "AWS CLI read failed")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("AWS CLI read returned malformed JSON") from exc
        if type(value) is not dict:
            raise ValueError("AWS CLI read returned a non-object")
        return value

    def get_s3_object(
        self,
        *,
        bucket: str,
        key: str,
        version_id: str,
        timeout_seconds: int,
    ) -> tuple[dict[str, object], bytes]:
        descriptor, temporary = tempfile.mkstemp(prefix=".glm52-h1d-object-")
        os.close(descriptor)
        path = Path(temporary)
        try:
            metadata = self.run_json(
                (
                    "s3api",
                    "get-object",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--version-id",
                    version_id,
                    str(path),
                ),
                timeout_seconds=timeout_seconds,
            )
            return metadata, path.read_bytes()
        finally:
            path.unlink(missing_ok=True)


def _sdk_filter(value: str) -> dict[str, object]:
    if not value.startswith("Name=") or ",Values=" not in value:
        raise ValueError("SDK filter is malformed")
    name, values = value[5:].split(",Values=", 1)
    split_values = values.split(",")
    if not name or not split_values or any(not item for item in split_values):
        raise ValueError("SDK filter is malformed")
    return {"Name": name, "Values": split_values}


def _sdk_json_safe(value: object) -> object:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("SDK response contains a naive timestamp")
        return value.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if type(value) is dict:
        return {
            str(key): _sdk_json_safe(item)
            for key, item in value.items()
        }
    if type(value) in (list, tuple):
        return [_sdk_json_safe(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("SDK response contains a noncanonical value")


class AwsSdkCommandRunner:
    """One-attempt in-process AWS SDK boundary over an exact read allowlist."""

    def __init__(
        self,
        *,
        profile: str = PROFILE,
        region: str = REGION,
        session_factory: Optional[Callable[..., object]] = None,
        config_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        if profile != PROFILE or region != REGION:
            raise ValueError("AWS SDK runner requires the exact profile and region")
        if session_factory is None or config_factory is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 and botocore are required for live H.1d"
                ) from exc
            if session_factory is None:
                session_factory = boto3.Session
            if config_factory is None:
                config_factory = Config
        self.profile = profile
        self.region = region
        self._config = config_factory(
            connect_timeout=1,
            read_timeout=2,
            max_pool_connections=SDK_MAX_WORKERS,
            retries={"total_max_attempts": 1, "mode": "standard"},
        )
        self._session = session_factory(
            profile_name=profile,
            region_name=region,
        )
        self._clients: dict[str, object] = {}
        self._client_lock = Lock()

    def _client(self, service: str) -> object:
        with self._client_lock:
            client = self._clients.get(service)
            if client is None:
                factory = getattr(self._session, "client", None)
                if not callable(factory):
                    raise ValueError("AWS SDK session has no client factory")
                client = factory(service, config=self._config)
                self._clients[service] = client
        return client

    @staticmethod
    def _arguments(
        key: tuple[str, str],
        raw: Sequence[str],
    ) -> tuple[dict[str, object], bool]:
        arguments: dict[str, object] = {}
        has_ec2_query = False
        index = 0
        token_options = _SDK_TOKEN_OPTIONS.get(key, {})
        while index < len(raw):
            flag = raw[index]
            if flag == "--no-paginate":
                index += 1
                continue
            if flag == "--query":
                if (
                    key != ("ec2", "describe-instances")
                    or index + 1 >= len(raw)
                    or raw[index + 1]
                    != (
                        "{Instances: Reservations[].Instances[], "
                        "NextToken: NextToken}"
                    )
                ):
                    raise ValueError("SDK query option is not allowlisted")
                has_ec2_query = True
                index += 2
                continue
            token_name = token_options.get(flag)
            if token_name is not None:
                if index + 1 >= len(raw):
                    raise ValueError("SDK pagination token is absent")
                arguments[token_name] = raw[index + 1]
                index += 2
                continue
            option = _SDK_OPTIONS.get(flag)
            if option is None:
                raise ValueError(f"SDK option {flag!r} is not allowlisted")
            name, kind = option
            if kind == "boolean":
                arguments[name] = True
                index += 1
                continue
            if index + 1 >= len(raw):
                raise ValueError(f"SDK option {flag!r} has no value")
            if kind in {"strings", "filters"}:
                values: list[str] = []
                index += 1
                while index < len(raw) and not raw[index].startswith("--"):
                    values.append(raw[index])
                    index += 1
                if not values:
                    raise ValueError(f"SDK option {flag!r} has no values")
                arguments[name] = (
                    [_sdk_filter(value) for value in values]
                    if kind == "filters"
                    else values
                )
                if key[0] == "ssm" and kind == "filters":
                    arguments[name] = [
                        {
                            "Key": item["Name"],
                            "Values": item["Values"],
                        }
                        for item in arguments[name]
                    ]
                continue
            value = raw[index + 1]
            if kind == "integer":
                parsed: object = int(value)
            elif kind == "json":
                parsed = json.loads(value)
            else:
                parsed = value
            arguments[name] = parsed
            index += 2
        return arguments, has_ec2_query

    def credential_expiration(self) -> str:
        get_credentials = getattr(self._session, "get_credentials", None)
        if not callable(get_credentials):
            raise ValueError("AWS SDK session has no credential provider")
        credentials = get_credentials()
        expiration = getattr(credentials, "_expiry_time", None)
        if not isinstance(expiration, datetime) or expiration.tzinfo is None:
            raise ValueError(
                "authenticated AWS credentials have no fixed expiration"
            )
        return expiration.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    @staticmethod
    def _error_code(exc: BaseException) -> object:
        response = getattr(exc, "response", None)
        if type(response) is not dict:
            return None
        error = response.get("Error")
        return error.get("Code") if type(error) is dict else None

    @staticmethod
    def _response_metadata(value: Mapping[str, object]) -> Mapping[str, object]:
        metadata = value.get("ResponseMetadata")
        if (
            type(metadata) is not dict
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
        ):
            raise ValueError("AWS SDK response lacks a service RequestId")
        return metadata

    def run_json(
        self,
        operation: Tuple[str, ...],
        *,
        timeout_seconds: int,
    ) -> dict[str, object]:
        if (
            type(operation) is not tuple
            or len(operation) < 2
            or any(type(part) is not str or not part for part in operation)
            or type(timeout_seconds) is not int
            or timeout_seconds < 1
            or timeout_seconds > 5
        ):
            raise ValueError("AWS SDK operation is malformed")
        key = (operation[0], operation[1])
        target = _SDK_METHODS.get(key)
        if target is None:
            raise ValueError("AWS SDK operation is outside the read allowlist")
        arguments, has_ec2_query = self._arguments(key, operation[2:])
        client = self._client(target[0])
        method = getattr(client, target[1], None)
        if not callable(method):
            raise ValueError("AWS SDK client lacks an allowlisted operation")
        try:
            response = method(**arguments)
        except BaseException as exc:
            expected_absence = _expected_read_absence(
                operation,
                self._error_code(exc),
            )
            if expected_absence is None:
                raise ValueError("AWS SDK read failed") from exc
            error_response = getattr(exc, "response", {})
            metadata = (
                error_response.get("ResponseMetadata")
                if type(error_response) is dict
                else None
            )
            if type(metadata) is not dict:
                raise ValueError(
                    "expected S3 absence lacks authenticated metadata"
                ) from exc
            response = {**expected_absence, "ResponseMetadata": metadata}
        if type(response) is not dict:
            raise ValueError("AWS SDK read returned a non-object")
        normalized = _sdk_json_safe(response)
        if type(normalized) is not dict:
            raise ValueError("AWS SDK read returned a non-object")
        self._response_metadata(normalized)
        if has_ec2_query:
            reservations = normalized.get("Reservations", [])
            if type(reservations) is not list:
                raise ValueError("EC2 reservation response is malformed")
            instances = [
                instance
                for reservation in reservations
                if type(reservation) is dict
                for instance in reservation.get("Instances", [])
            ]
            normalized = {
                "Instances": instances,
                "NextToken": normalized.get("NextToken"),
                "ResponseMetadata": normalized["ResponseMetadata"],
            }
        return normalized

    def get_s3_object(
        self,
        *,
        bucket: str,
        key: str,
        version_id: str,
        timeout_seconds: int,
    ) -> tuple[dict[str, object], bytes]:
        if (
            type(bucket) is not str
            or not bucket
            or type(key) is not str
            or not key
            or type(version_id) is not str
            or not version_id
            or timeout_seconds < 1
            or timeout_seconds > 5
        ):
            raise ValueError("S3 exact object request is malformed")
        method = getattr(self._client("s3"), "get_object", None)
        if not callable(method):
            raise ValueError("AWS SDK S3 client lacks get_object")
        try:
            response = method(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
                ExpectedBucketOwner=ACCOUNT_ID,
                ChecksumMode="ENABLED",
            )
        except BaseException as exc:
            raise ValueError("AWS SDK S3 exact object read failed") from exc
        if type(response) is not dict:
            raise ValueError("AWS SDK S3 object response is malformed")
        body = response.get("Body")
        read = getattr(body, "read", None)
        if not callable(read):
            raise ValueError("AWS SDK S3 object body is absent")
        raw = read()
        if type(raw) is not bytes:
            raise ValueError("AWS SDK S3 object body is malformed")
        metadata = {
            name: value for name, value in response.items()
            if name != "Body"
        }
        normalized = _sdk_json_safe(metadata)
        if type(normalized) is not dict:
            raise ValueError("AWS SDK S3 metadata is malformed")
        self._response_metadata(normalized)
        if (
            normalized.get("VersionId") != version_id
            or type(normalized.get("ChecksumSHA256")) is not str
            or normalized["ChecksumSHA256"]
            != base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
        ):
            raise ValueError("AWS SDK S3 object version/checksum drifted")
        return normalized, raw

    def invoke_task9_attestation(
        self,
        *,
        function_arn: str,
        payload: bytes,
        timeout_seconds: int,
    ) -> Mapping[str, object]:
        if (
            type(function_arn) is not str
            or not function_arn.startswith(
                f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:"
            )
            or function_arn.endswith(":$LATEST")
            or function_arn.count(":") < 7
            or type(payload) is not bytes
            or not payload
            or timeout_seconds != 5
        ):
            raise ValueError("Task 9 attestation invocation is not exact")
        method = getattr(self._client("lambda"), "invoke", None)
        if not callable(method):
            raise ValueError("AWS SDK Lambda client lacks invoke")
        try:
            response = method(
                FunctionName=function_arn,
                InvocationType="RequestResponse",
                LogType="None",
                Payload=payload,
            )
        except BaseException as exc:
            raise ValueError("Task 9 attestation invocation failed") from exc
        if type(response) is not dict:
            raise ValueError("Task 9 attestation response is malformed")
        body = response.get("Payload")
        read = getattr(body, "read", None)
        if not callable(read):
            raise ValueError("Task 9 attestation payload is absent")
        raw = read(1_048_577)
        metadata = response.get("ResponseMetadata")
        if (
            type(raw) is not bytes
            or len(raw) > 1_048_576
            or response.get("StatusCode") != 200
            or "FunctionError" in response
            or response.get("ExecutedVersion")
            != function_arn.rsplit(":", 1)[-1]
            or type(metadata) is not dict
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
        ):
            raise ValueError("Task 9 attestation invocation did not authenticate")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Task 9 attestation returned malformed JSON") from exc
        if type(value) is not dict:
            raise ValueError("Task 9 attestation returned a non-object")
        return value


class AwsTrustedCampaignSourceReader:
    """Read the independently addressed immutable Task 6/7 source set."""

    def __init__(self, *, runner: AwsCliCommandRunner) -> None:
        self.runner = runner
        self._contract: Optional[dict[str, object]] = None

    def read_contract(self) -> dict[str, object]:
        if self._contract is not None:
            return {
                **self._contract,
                "authority_sources": dict(
                    self._contract["authority_sources"]
                ),
                "cloudformation_deployment_role": dict(
                    self._contract["cloudformation_deployment_role"]
                ),
                "dynamodb_expected_items": tuple(
                    dict(item)
                    for item in self._contract["dynamodb_expected_items"]
                ),
                "cloudformation_expected_items": tuple(
                    dict(item)
                    for item in self._contract[
                        "cloudformation_expected_items"
                    ]
                ),
            }
        key = {
            "PK": {"S": f"RUN#{RUN_ID}"},
            "SK": {"S": _TRUSTED_SOURCE_SORT_KEY},
        }
        raw_response = self.runner.run_json(
            (
                "dynamodb",
                "get-item",
                "--table-name",
                _TRUSTED_SOURCE_TABLE,
                "--key",
                canonical_json_bytes(key).decode("utf-8"),
                "--consistent-read",
                "--return-consumed-capacity",
                "NONE",
            ),
            timeout_seconds=5,
        )
        response, request_id = _service_response(raw_response)
        item = response.get("Item")
        expected_fields = {
            "PK",
            "SK",
            "record_type",
            "state",
            "canonical_body_json",
            "canonical_body_sha256",
        }
        if type(item) is not dict or set(item) != expected_fields:
            raise ValueError("trusted campaign source record is absent")

        def string_attribute(name: str) -> str:
            attribute = item.get(name)
            if (
                type(attribute) is not dict
                or set(attribute) != {"S"}
                or type(attribute["S"]) is not str
                or not attribute["S"]
            ):
                raise ValueError(
                    "trusted campaign source record is malformed"
                )
            return attribute["S"]

        if (
            string_attribute("PK") != f"RUN#{RUN_ID}"
            or string_attribute("SK") != _TRUSTED_SOURCE_SORT_KEY
            or string_attribute("record_type")
            != "glm52_h1d_trusted_campaign_sources_v1"
            or string_attribute("state") != "SEALED"
        ):
            raise ValueError("trusted campaign source record is foreign")
        raw_record = string_attribute("canonical_body_json")
        try:
            record = json.loads(raw_record)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "trusted campaign source record has malformed JSON"
            ) from exc
        record_fields = {
            "schema_version",
            "record_type",
            "account_id",
            "region",
            "run_id",
            "PK",
            "SK",
            "state",
            "authority_sources",
            "source_contract_sha256",
            "cloudformation_deployment_role",
            "dynamodb_expected_items",
            "cloudformation_expected_items",
            "canonical_body_sha256",
        }
        if type(record) is not dict or set(record) != record_fields:
            raise ValueError("trusted campaign source body is malformed")
        body = dict(record)
        identity = body.pop("canonical_body_sha256")
        if (
            canonical_json_bytes(record).decode("utf-8") != raw_record
            or record["schema_version"] != 1
            or record["record_type"]
            != "glm52_h1d_trusted_campaign_sources_v1"
            or record["account_id"] != ACCOUNT_ID
            or record["region"] != REGION
            or record["run_id"] != RUN_ID
            or record["PK"] != f"RUN#{RUN_ID}"
            or record["SK"] != _TRUSTED_SOURCE_SORT_KEY
            or record["state"] != "SEALED"
            or identity != canonical_sha256(body)
            or identity != string_attribute("canonical_body_sha256")
        ):
            raise ValueError("trusted campaign source identity drifted")
        source_contract_sha256 = record["source_contract_sha256"]
        expected_items = record["dynamodb_expected_items"]
        cloudformation_expected_items = record[
            "cloudformation_expected_items"
        ]
        cloudformation_deployment_role = record[
            "cloudformation_deployment_role"
        ]
        if (
            type(source_contract_sha256) is not str
            or _SHA256.fullmatch(source_contract_sha256) is None
            or type(expected_items) is not list
            or not expected_items
            or type(cloudformation_expected_items) is not list
            or len(cloudformation_expected_items) != 3
            or type(cloudformation_deployment_role) is not dict
            or set(cloudformation_deployment_role)
            != {"role_arn", "role_id", "request_id"}
            or type(cloudformation_deployment_role["role_arn"]) is not str
            or not cloudformation_deployment_role["role_arn"].startswith(
                f"arn:aws:iam::{ACCOUNT_ID}:role/"
            )
            or type(cloudformation_deployment_role["role_id"]) is not str
            or _IAM_ROLE_ID.fullmatch(
                cloudformation_deployment_role["role_id"]
            )
            is None
            or type(cloudformation_deployment_role["request_id"]) is not str
            or not cloudformation_deployment_role["request_id"]
            or any(
                type(expected_item) is not dict
                or set(expected_item)
                != {
                    "key",
                    "record_type",
                    "body_sha256",
                    "consistent_read",
                }
                or type(expected_item["key"]) is not str
                or not expected_item["key"].startswith(f"RUN#{RUN_ID}|")
                or type(expected_item["record_type"]) is not str
                or not expected_item["record_type"]
                or type(expected_item["body_sha256"]) is not str
                or _SHA256.fullmatch(expected_item["body_sha256"]) is None
                or expected_item["consistent_read"] is not True
                for expected_item in expected_items
            )
        ):
            raise ValueError("trusted campaign source contract is malformed")
        keys = [str(expected_item["key"]) for expected_item in expected_items]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError(
                "trusted campaign DynamoDB inventory is not canonical"
            )
        cloudformation_fields = {
            "stack_id",
            "status",
            "template_sha256",
            "parameters_sha256",
            "resources_sha256",
            "stack_tags",
            "service_role_arn",
            "termination_protection",
            "manifest_identity_sha256",
        }
        if any(
            type(item) is not dict
            or set(item) != cloudformation_fields
            or type(item["stack_id"]) is not str
            or not item["stack_id"].startswith(
                f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
            )
            or item["status"] != "UPDATE_COMPLETE"
            or any(
                type(item[field]) is not str
                or _SHA256.fullmatch(item[field]) is None
                for field in (
                    "template_sha256",
                    "parameters_sha256",
                    "resources_sha256",
                    "manifest_identity_sha256",
                )
            )
            or type(item["service_role_arn"]) is not str
            or not item["service_role_arn"].startswith(
                f"arn:aws:iam::{ACCOUNT_ID}:role/"
            )
            or _cloudformation_stack_tags(
                item["stack_tags"],
                canonical_shape=True,
            )
            != item["stack_tags"]
            or item["termination_protection"] is not True
            for item in cloudformation_expected_items
        ):
            raise ValueError(
                "trusted campaign CloudFormation inventory is malformed"
            )
        stack_ids = [
            str(item["stack_id"])
            for item in cloudformation_expected_items
        ]
        if (
            stack_ids != sorted(stack_ids)
            or len(stack_ids) != len(set(stack_ids))
            or len(
                {
                    str(item["service_role_arn"])
                    for item in cloudformation_expected_items
                }
            )
            != 1
            or {
                str(item["service_role_arn"])
                for item in cloudformation_expected_items
            }
            != {cloudformation_deployment_role["role_arn"]}
        ):
            raise ValueError(
                "trusted campaign CloudFormation inventory is not canonical"
            )
        contract = {
            "authority_sources": _validated_authority_sources(
                record["authority_sources"]
            ),
            "source_contract_sha256": source_contract_sha256,
            "cloudformation_deployment_role": dict(
                cloudformation_deployment_role
            ),
            "dynamodb_expected_items": tuple(
                dict(expected_item) for expected_item in expected_items
            ),
            "cloudformation_expected_items": tuple(
                dict(expected_item)
                for expected_item in cloudformation_expected_items
            ),
            "request_id": (
                request_id
                if request_id is not None
                else canonical_sha256(response)
            ),
        }
        self._contract = contract
        return {
            **contract,
            "authority_sources": dict(contract["authority_sources"]),
            "cloudformation_deployment_role": dict(
                contract["cloudformation_deployment_role"]
            ),
            "dynamodb_expected_items": tuple(
                dict(expected_item)
                for expected_item in contract["dynamodb_expected_items"]
            ),
            "cloudformation_expected_items": tuple(
                dict(expected_item)
                for expected_item in contract[
                    "cloudformation_expected_items"
                ]
            ),
        }

    def read_sources(self) -> dict[str, object]:
        return dict(self.read_contract()["authority_sources"])


def _path(value: object, path: Sequence[object]) -> object:
    current = value
    for component in path:
        if type(component) is int and type(current) is list:
            try:
                current = current[component]
            except IndexError as exc:
                raise ValueError("CLI projection index is absent") from exc
        elif type(component) is str and type(current) is dict:
            if component not in current:
                raise ValueError("CLI projection field is absent")
            current = current[component]
        else:
            raise ValueError("CLI projection path does not match response")
    return current


def _optional_terminal_path(
    value: object, path: Optional[Sequence[object]]
) -> object:
    if path is None:
        return None
    if not path:
        return value
    parent = _path(value, path[:-1]) if len(path) > 1 else value
    terminal = path[-1]
    if type(terminal) is str and type(parent) is dict:
        return parent.get(terminal)
    if type(terminal) is int and type(parent) is list:
        if terminal >= len(parent):
            return None
        return parent[terminal]
    raise ValueError("CLI terminal path does not match response")


def _encoded_pagination_token(
    value: object,
    *,
    token_argument: str,
) -> str:
    if type(value) is str and value:
        return value
    if (
        token_argument == "--exclusive-start-key"
        and type(value) is dict
        and value
    ):
        return canonical_json_bytes(value).decode("utf-8")
    raise ValueError("AWS pagination token is malformed")


def _exact_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} is absent")
    return value


def _service_response(
    response: object,
) -> tuple[dict[str, object], Optional[str]]:
    if type(response) is not dict:
        raise ValueError("AWS read response is not an object")
    metadata = response.get("ResponseMetadata")
    if metadata is None:
        return dict(response), None
    if (
        type(metadata) is not dict
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        raise ValueError("AWS response metadata has no RequestId")
    payload = {
        key: value for key, value in response.items()
        if key != "ResponseMetadata"
    }
    return payload, metadata["RequestId"]


def _unique_sorted_strings(values: Sequence[object], *, label: str) -> list[str]:
    if any(type(value) is not str or not value for value in values):
        raise ValueError(f"{label} contains a malformed identity")
    strings = [str(value) for value in values]
    if len(strings) != len(set(strings)):
        raise ValueError(f"{label} contains a duplicate identity")
    return sorted(strings)


def _iam_arn(
    physical: object,
    *,
    resource_kind: str,
    path: object = "/",
) -> str:
    value = _exact_string(physical, label=f"IAM {resource_kind} identity")
    expected_prefix = f"arn:aws:iam::{ACCOUNT_ID}:{resource_kind}/"
    if value.startswith("arn:"):
        if not value.startswith(expected_prefix):
            raise ValueError(f"IAM {resource_kind} ARN is out of scope")
        return value
    if type(path) is not str or not path.startswith("/") or not path.endswith("/"):
        raise ValueError(f"IAM {resource_kind} path is malformed")
    normalized_path = path.strip("/")
    path_prefix = f"{normalized_path}/" if normalized_path else ""
    return f"{expected_prefix}{path_prefix}{value}"


def _lambda_configuration_projection(
    configuration: object,
    configuration_fields: object,
) -> dict[str, object]:
    if type(configuration) is not dict:
        raise ValueError("Lambda configuration is malformed")
    if type(configuration_fields) not in (list, tuple):
        raise ValueError("Lambda configuration field contract is malformed")
    fields = tuple(configuration_fields)
    if (
        any(type(field) is not str for field in fields)
        or fields != tuple(sorted(fields))
        or len(fields) != len(set(fields))
        or not set(fields).issubset(_LAMBDA_CONFIGURATION_FIELDS)
    ):
        raise ValueError("Lambda configuration field contract is malformed")
    projected: dict[str, object] = {}
    for field in fields:
        api_field = "KMSKeyArn" if field == "KmsKeyArn" else field
        if api_field not in configuration and field in configuration:
            api_field = field
        if api_field not in configuration:
            raise ValueError(f"Lambda configuration lacks {field}")
        value = configuration[api_field]
        if field == "DeadLetterConfig":
            if (
                type(value) is not dict
                or set(value) != {"TargetArn"}
                or type(value.get("TargetArn")) is not str
            ):
                raise ValueError("Lambda dead-letter configuration is malformed")
            projected[field] = {"TargetArn": value["TargetArn"]}
        elif field == "Environment":
            if (
                type(value) is not dict
                or type(value.get("Variables")) is not dict
                or any(
                    type(key) is not str or type(item) is not str
                    for key, item in value["Variables"].items()
                )
            ):
                raise ValueError("Lambda environment configuration is malformed")
            projected[field] = {"Variables": dict(value["Variables"])}
        elif field == "Layers":
            if type(value) is not list:
                raise ValueError("Lambda layer configuration is malformed")
            layer_arns = []
            for layer in value:
                arn = layer.get("Arn") if type(layer) is dict else layer
                layer_arns.append(
                    _exact_string(arn, label="Lambda layer ARN")
                )
            projected[field] = layer_arns
        elif field == "TracingConfig":
            if (
                type(value) is not dict
                or type(value.get("Mode")) is not str
            ):
                raise ValueError("Lambda tracing configuration is malformed")
            projected[field] = {"Mode": value["Mode"]}
        elif field == "VpcConfig":
            if type(value) is not dict:
                raise ValueError("Lambda VPC configuration is malformed")
            security_groups = value.get("SecurityGroupIds")
            subnets = value.get("SubnetIds")
            if type(security_groups) is not list or type(subnets) is not list:
                raise ValueError("Lambda VPC configuration is incomplete")
            projected[field] = {
                "SecurityGroupIds": _unique_sorted_strings(
                    security_groups,
                    label="Lambda VPC security group",
                ),
                "SubnetIds": _unique_sorted_strings(
                    subnets,
                    label="Lambda VPC subnet",
                ),
            }
        else:
            projected[field] = value
    return projected


def _lambda_policy_principal(principal: object) -> object:
    if type(principal) is dict:
        if set(principal) not in ({"Service"}, {"AWS"}):
            raise ValueError("Lambda permission principal is malformed")
        key = next(iter(principal))
        return {
            key: _exact_string(
                principal[key], label="Lambda permission principal"
            )
        }
    value = _exact_string(principal, label="Lambda permission principal")
    if value == "*":
        return value
    if value.endswith(".amazonaws.com"):
        return {"Service": value}
    if re.fullmatch(r"[0-9]{12}", value):
        return {"AWS": f"arn:aws:iam::{value}:root"}
    if value.startswith("arn:"):
        return {"AWS": value}
    raise ValueError("Lambda permission principal is malformed")


def _lambda_policy_statement(
    *,
    sid: object,
    action: object,
    principal: object,
    resource: object,
    source_arn: object = None,
    source_account: object = None,
) -> dict[str, object]:
    statement: dict[str, object] = {
        "Sid": _exact_string(sid, label="Lambda permission Sid"),
        "Effect": "Allow",
        "Principal": _lambda_policy_principal(principal),
        "Action": _exact_string(action, label="Lambda permission action"),
        "Resource": _exact_string(resource, label="Lambda permission resource"),
    }
    condition: dict[str, object] = {}
    if source_arn is not None:
        condition["ArnLike"] = {
            "AWS:SourceArn": _exact_string(
                source_arn, label="Lambda permission source ARN"
            )
        }
    if source_account is not None:
        condition["StringEquals"] = {
            "AWS:SourceAccount": _exact_string(
                source_account, label="Lambda permission source account"
            )
        }
    if condition:
        statement["Condition"] = condition
    return statement


def _lambda_live_policy_statements(policy: object) -> list[dict[str, object]]:
    if policy is None:
        return []
    if policy == []:
        return []
    if type(policy) is not dict or type(policy.get("Statement")) is not list:
        raise ValueError("Lambda policy is malformed")
    statements: list[dict[str, object]] = []
    for raw in policy["Statement"]:
        if type(raw) is not dict or not set(raw).issubset(
            {"Sid", "Effect", "Principal", "Action", "Resource", "Condition"}
        ):
            raise ValueError("Lambda policy statement is malformed")
        if raw.get("Effect") != "Allow":
            raise ValueError("Lambda policy statement effect is malformed")
        condition = raw.get("Condition", {})
        if type(condition) is not dict or not set(condition).issubset(
            {"ArnLike", "StringEquals"}
        ):
            raise ValueError("Lambda policy condition is malformed")
        arn_like = condition.get("ArnLike", {})
        string_equals = condition.get("StringEquals", {})
        if (
            type(arn_like) is not dict
            or type(string_equals) is not dict
            or not set(arn_like).issubset({"AWS:SourceArn"})
            or not set(string_equals).issubset({"AWS:SourceAccount"})
        ):
            raise ValueError("Lambda policy condition is malformed")
        statements.append(
            _lambda_policy_statement(
                sid=raw.get("Sid"),
                action=raw.get("Action"),
                principal=raw.get("Principal"),
                resource=raw.get("Resource"),
                source_arn=arn_like.get("AWS:SourceArn"),
                source_account=string_equals.get("AWS:SourceAccount"),
            )
        )
    return sorted(statements, key=lambda item: str(item["Sid"]))


def _lambda_event_source_mapping_projection(
    value: object,
    *,
    uuid: object,
    function_arn: object,
    fields: object,
) -> dict[str, object]:
    if (
        type(value) is not dict
        or type(fields) not in (list, tuple)
        or any(type(field) is not str for field in fields)
    ):
        raise ValueError("Lambda event-source mapping is malformed")
    normalized_fields = tuple(fields)
    if (
        normalized_fields != tuple(sorted(normalized_fields))
        or len(normalized_fields) != len(set(normalized_fields))
        or not set(normalized_fields).issubset(
            _LAMBDA_EVENT_SOURCE_MAPPING_FIELDS
        )
    ):
        raise ValueError(
            "Lambda event-source mapping field contract is malformed"
        )
    projected: dict[str, object] = {
        "UUID": _exact_string(
            uuid, label="Lambda event-source mapping UUID"
        ),
        "FunctionArn": _exact_string(
            function_arn, label="Lambda event-source mapping function ARN"
        ),
    }
    for field in normalized_fields:
        if field == "Enabled":
            if "Enabled" in value:
                enabled = value["Enabled"]
                if type(enabled) is not bool:
                    raise ValueError(
                        "Lambda event-source mapping Enabled is malformed"
                    )
            else:
                state = value.get("State")
                if state not in {"Enabled", "Disabled"}:
                    raise ValueError(
                        "Lambda event-source mapping State is malformed"
                    )
                enabled = state == "Enabled"
            projected[field] = enabled
        else:
            if field not in value:
                raise ValueError(
                    f"Lambda event-source mapping lacks {field}"
                )
            projected[field] = value[field]
    return projected


def _iam_role_tags(value: object) -> tuple[dict[str, str], ...]:
    if value is None:
        value = []
    if type(value) is not list:
        raise ValueError("IAM role tags are malformed")
    tags = []
    for tag in value:
        if (
            type(tag) is not dict
            or set(tag) != {"Key", "Value"}
            or type(tag["Key"]) is not str
            or not tag["Key"]
            or type(tag["Value"]) is not str
        ):
            raise ValueError("IAM role tag is malformed")
        tags.append({"Key": tag["Key"], "Value": tag["Value"]})
    tags.sort(key=lambda item: item["Key"])
    if len({tag["Key"] for tag in tags}) != len(tags):
        raise ValueError("IAM role tag key is duplicated")
    return tuple(tags)


def _cloudformation_stack_tags(
    value: object,
    *,
    canonical_shape: bool = False,
) -> list[dict[str, str]]:
    if type(value) is not list:
        raise ValueError("CloudFormation stack tags are malformed")
    key_field = "key" if canonical_shape else "Key"
    value_field = "value" if canonical_shape else "Value"
    expected_fields = {key_field, value_field}
    tags: list[dict[str, str]] = []
    for tag in value:
        if (
            type(tag) is not dict
            or set(tag) != expected_fields
            or type(tag[key_field]) is not str
            or not tag[key_field]
            or type(tag[value_field]) is not str
        ):
            raise ValueError("CloudFormation stack tag is malformed")
        tags.append(
            {
                "key": tag[key_field],
                "value": tag[value_field],
            }
        )
    tags.sort(key=lambda item: item["key"])
    if len({tag["key"] for tag in tags}) != len(tags):
        raise ValueError("CloudFormation stack tag key is duplicated")
    return tags


def _closed_inventory_arguments(spec: object) -> list[str]:
    if spec.family == "ec2":
        return _closed_inventory_argument_sets(spec)[0]
    if spec.family == "sns":
        if len(spec.expected_items) != 1:
            raise ValueError("SNS plan identity inventory is not exact")
        subscription_arn = spec.expected_items[0].get("subscription_arn")
        if (
            type(subscription_arn) is not str
            or subscription_arn.count(":") < 6
        ):
            raise ValueError("SNS plan identity inventory is malformed")
        return ["--topic-arn", subscription_arn.rsplit(":", 1)[0]]
    if spec.family == "dynamodb":
        expression_values = {
            ":pk": {"S": f"RUN#{RUN_ID}"},
            ":trusted_source_sk": {"S": _TRUSTED_SOURCE_SORT_KEY},
        }
        return [
            "--table-name",
            _TRUSTED_SOURCE_TABLE,
            "--key-condition-expression",
            "PK = :pk",
            "--expression-attribute-values",
            canonical_json_bytes(expression_values).decode("utf-8"),
            "--filter-expression",
            "SK <> :trusted_source_sk",
            "--consistent-read",
        ]
    if spec.family == "eventbridge":
        return ["--name-prefix", "keep-glm52-h1g"]
    if spec.family == "scheduler":
        return ["--name-prefix", "keep-glm52-h1g"]
    if spec.family == "sqs":
        return ["--queue-name-prefix", "keep-glm52-h1g"]
    if spec.family == "ssm":
        return [
            "--filters",
            f"Key=tag:campaign-run-id,Values={RUN_ID}",
        ]
    if spec.family == "cloudwatch":
        return ["--alarm-name-prefix", "keep-glm52-h1g"]
    if spec.family == "logs":
        return [
            "--log-group-name-prefix",
            "/aws/lambda/keep-glm52-h1g",
        ]
    return []


def _closed_inventory_argument_sets(spec: object) -> list[list[str]]:
    campaign_prefix_options = {
        "eventbridge": ("--name-prefix", "keep-glm52-h1g", RUN_ID),
        "scheduler": ("--name-prefix", "keep-glm52-h1g", RUN_ID),
        "sqs": ("--queue-name-prefix", "keep-glm52-h1g", RUN_ID),
        "cloudwatch": ("--alarm-name-prefix", "keep-glm52-h1g", RUN_ID),
        "logs": (
            "--log-group-name-prefix",
            "/aws/lambda/keep-glm52-h1g",
            f"/aws/lambda/{RUN_ID}",
        ),
    }
    prefix = campaign_prefix_options.get(spec.family)
    if prefix is not None:
        flag, support_prefix, run_prefix = prefix
        return [
            [flag, support_prefix],
            [flag, run_prefix],
        ]
    if spec.family != "ec2":
        return [_closed_inventory_arguments(spec)]

    def query(filter_value: str) -> list[str]:
        return [
            "--filters",
            filter_value,
            "--max-results",
            "100",
            "--no-paginate",
            "--query",
            "{Instances: Reservations[].Instances[], NextToken: NextToken}",
        ]

    return [
        query(f"Name=tag:campaign-run-id,Values={RUN_ID}"),
        query("Name=instance-type,Values=p5.48xlarge"),
        query(f"Name=tag:Name,Values={RUN_ID}"),
    ]


def _inventory_item_in_scope(spec: object, item: object) -> bool:
    """Keep exact expected identities and all campaign-namespaced drift."""
    family = spec.family
    try:
        identity = AwsCliLiveReader._inventory_identity(family, item)
    except (KeyError, TypeError, ValueError):
        return False
    expected = {
        str(expected_item.get(spec.identity_field))
        for expected_item in spec.expected_items
        if type(expected_item) is dict
        and type(expected_item.get(spec.identity_field)) is str
    }
    if identity in expected:
        return True
    campaign_markers = ("keep-glm52-h1g", RUN_ID)
    candidates = [identity]
    if type(item) is dict:
        candidates.extend(
            str(item.get(field, ""))
            for field in (
                "StackName",
                "FunctionName",
                "RoleName",
                "Name",
            )
        )
    return any(
        marker in candidate
        for marker in campaign_markers
        for candidate in candidates
    )


class _LiveReadGuard:
    """Cooperative cancellation checked around every SDK read boundary."""

    def __init__(self, *, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("live-read guard timeout already elapsed")
        self._deadline = time.monotonic() + timeout_seconds
        self._cancelled = Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def check(self) -> None:
        if self._cancelled.is_set() or time.monotonic() >= self._deadline:
            raise ValueError("SDK live-read deadline elapsed")


class AwsCliLiveReader:
    """Execute authenticated manifest-provided projections over closed reads."""

    def __init__(
        self,
        *,
        runner: AwsCliCommandRunner,
        clock: Callable[[], datetime] = _now,
        read_guard: Optional[_LiveReadGuard] = None,
    ) -> None:
        self.runner = runner
        self.clock = clock
        self.read_guard = read_guard
        self._page_indices: dict[str, int] = {}

    def _command(
        self, family: str, index: int, arguments: Sequence[str]
    ) -> dict[str, object]:
        inventory = READ_COMMANDS.get(family)
        if (
            inventory is None
            or type(index) is not int
            or index < 0
            or index >= len(inventory)
            or type(arguments) not in (list, tuple)
            or any(type(item) is not str or not item for item in arguments)
        ):
            raise ValueError("CLI read plan names an unapproved operation")
        closed_arguments = list(arguments)
        if "--no-paginate" not in closed_arguments:
            closed_arguments.append("--no-paginate")
        if self.read_guard is not None:
            self.read_guard.check()
        response = self.runner.run_json(
            (*inventory[index], *closed_arguments),
            timeout_seconds=5,
        )
        if self.read_guard is not None:
            self.read_guard.check()
        return response

    @staticmethod
    def _inventory_identity(family: str, item: object) -> str:
        if family == "dynamodb":
            if type(item) is not dict:
                raise ValueError("DynamoDB inventory item is malformed")
            partition = _path(item, ("PK", "S"))
            sort = _path(item, ("SK", "S"))
            return (
                f"{_exact_string(partition, label='DynamoDB PK')}|"
                f"{_exact_string(sort, label='DynamoDB SK')}"
            )
        identity = _path(item, _INVENTORY_IDENTITY_PATHS[family])
        return _exact_string(identity, label=f"{family} inventory identity")

    @staticmethod
    def _first_response(
        details: Sequence[Mapping[str, object]],
        index: int,
    ) -> Mapping[str, object]:
        try:
            responses = details[index]["responses"]
            response = responses[0]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("AWS detail response is absent") from exc
        if type(responses) is not list or len(responses) != 1 or type(response) is not dict:
            raise ValueError("AWS detail response cardinality is not exact")
        return response

    @staticmethod
    def _json_document(value: object, *, label: str) -> object:
        if value is None:
            return None
        if type(value) is str:
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} is malformed JSON") from exc
        if type(value) in (dict, list):
            return value
        raise ValueError(f"{label} is not a JSON document")

    @staticmethod
    def _expected_item(
        spec: LiveReadSpec,
        identity: str,
    ) -> Mapping[str, object]:
        matches = [
            item
            for item in spec.expected_items
            if item.get(spec.identity_field) == identity
        ]
        if not matches and spec.family == "sqs":
            queue_name = identity.rsplit("/", 1)[-1]
            matches = [
                item
                for item in spec.expected_items
                if str(item.get("queue_arn", "")).endswith(
                    f":{queue_name}"
                )
            ]
        if len(matches) != 1:
            raise ValueError(
                f"{spec.family} AWS identity is not independently expected"
            )
        return matches[0]

    @classmethod
    def _normalize_aws_item(
        cls,
        *,
        spec: LiveReadSpec,
        identity: str,
        inventory_item: object,
        details: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        """Normalize real AWS response shapes into the canonical live item."""
        family = spec.family
        expected = cls._expected_item(spec, identity)
        if type(inventory_item) is not dict and family != "sqs":
            raise ValueError(f"{family} AWS inventory item is malformed")

        if family == "cloudformation":
            template = cls._first_response(details, 0).get("TemplateBody")
            resources = details[1].get("items")
            if type(resources) is not list:
                raise ValueError("CloudFormation resource inventory is absent")
            normalized_resources = sorted(
                (
                    {
                        "logical_id": row.get("LogicalResourceId"),
                        "resource_type": row.get("ResourceType"),
                        "physical_id": row.get("PhysicalResourceId"),
                        "resource_status": row.get("ResourceStatus"),
                    }
                    for row in resources
                    if type(row) is dict
                ),
                key=lambda row: str(row["logical_id"]),
            )
            if len(normalized_resources) != len(resources):
                raise ValueError("CloudFormation resource inventory is malformed")
            return {
                "stack_id": identity,
                "status": inventory_item.get("StackStatus"),
                "template_sha256": canonical_sha256(template),
                "parameters_sha256": canonical_sha256(
                    inventory_item.get("Parameters", [])
                ),
                "resources_sha256": canonical_sha256(normalized_resources),
                "stack_tags": _cloudformation_stack_tags(
                    inventory_item.get("Tags")
                ),
                "service_role_arn": inventory_item.get("RoleARN"),
                "termination_protection": inventory_item.get(
                    "EnableTerminationProtection"
                ),
                "manifest_identity_sha256": expected[
                    "manifest_identity_sha256"
                ],
            }
        if family == "lambda":
            function = cls._first_response(details, 0)
            configuration = function.get("Configuration")
            code = function.get("Code")
            if type(configuration) is not dict or type(code) is not dict:
                raise ValueError("Lambda function response is malformed")
            code_sha256 = configuration.get("CodeSha256")
            if type(code_sha256) is str:
                try:
                    code_sha256 = base64.b64decode(
                        code_sha256, validate=True
                    ).hex()
                except ValueError as exc:
                    raise ValueError(
                        "Lambda CodeSha256 is malformed"
                    ) from exc
            else:
                code_sha256 = canonical_sha256(code)
            configuration_fields = expected.get("configuration_fields")
            semantic_configuration = _lambda_configuration_projection(
                configuration,
                configuration_fields,
            )
            policy = cls._first_response(details, 1).get("Policy")
            policy_document = cls._json_document(
                policy, label="Lambda policy"
            )
            concurrency = cls._first_response(details, 2)
            mapping_contract = expected.get(
                "event_source_mapping_contract"
            )
            mappings = details[4].get("items")
            if (
                type(mapping_contract) not in (list, tuple)
                or any(type(item) is not dict for item in mapping_contract)
                or type(mappings) is not list
                or any(type(item) is not dict for item in mappings)
            ):
                raise ValueError(
                    "Lambda event-source mapping inventory is malformed"
                )
            contract_by_uuid = {
                item.get("uuid"): item for item in mapping_contract
            }
            mapping_by_uuid = {
                item.get("UUID"): item for item in mappings
            }
            if (
                len(contract_by_uuid) != len(mapping_contract)
                or len(mapping_by_uuid) != len(mappings)
                or set(contract_by_uuid) != set(mapping_by_uuid)
            ):
                raise ValueError(
                    "Lambda event-source mapping inventory is not exact"
                )
            semantic_mappings = sorted(
                (
                    _lambda_event_source_mapping_projection(
                        mapping_by_uuid[uuid],
                        uuid=uuid,
                        function_arn=identity,
                        fields=contract_by_uuid[uuid].get("fields"),
                    )
                    for uuid in contract_by_uuid
                ),
                key=lambda item: str(item["UUID"]),
            )
            return {
                "function_arn": identity,
                "code_sha256": code_sha256,
                "configuration_fields": tuple(configuration_fields),
                "configuration_sha256": canonical_sha256(
                    semantic_configuration
                ),
                "policy_sha256": canonical_sha256(
                    _lambda_live_policy_statements(policy_document)
                ),
                "event_source_mapping_contract": tuple(
                    dict(item) for item in mapping_contract
                ),
                "event_source_mappings_sha256": canonical_sha256(
                    semantic_mappings
                ),
                "reserved_concurrency": concurrency.get(
                    "ReservedConcurrentExecutions"
                ),
                "vpc_attachment_sha256": canonical_sha256(
                    semantic_configuration.get("VpcConfig")
                ),
                "timeout_seconds": configuration.get("Timeout"),
                "memory_mib": configuration.get("MemorySize"),
                "log_group_arn": expected["log_group_arn"],
                "alarm_arns": expected["alarm_arns"],
                "dlq_arn": (
                    configuration.get("DeadLetterConfig", {}).get("TargetArn")
                    if type(configuration.get("DeadLetterConfig")) is dict
                    else None
                ),
            }
        if family == "iam":
            role_response = cls._first_response(details, 0).get("Role")
            if type(role_response) is not dict:
                raise ValueError("IAM role response is malformed")
            permissions_boundary = role_response.get(
                "PermissionsBoundary"
            )
            if permissions_boundary is not None and (
                type(permissions_boundary) is not dict
                or permissions_boundary.get("PermissionsBoundaryType")
                != "Policy"
                or type(
                    permissions_boundary.get("PermissionsBoundaryArn")
                )
                is not str
            ):
                raise ValueError("IAM permissions boundary is malformed")
            profiles = details[1].get("items")
            policy_names = details[2].get("items")
            inline_responses = details[3].get("responses")
            attached = details[4].get("items")
            versions = details[6].get("responses")
            if any(
                type(value) is not list
                for value in (
                    profiles,
                    policy_names,
                    inline_responses,
                    attached,
                    versions,
                )
            ):
                raise ValueError("IAM complete detail inventory is malformed")
            inline = sorted(
                (
                    {
                        "PolicyName": response.get("PolicyName"),
                        "PolicyDocument": cls._json_document(
                            response.get("PolicyDocument"),
                            label="IAM inline policy",
                        ),
                    }
                    for response in inline_responses
                    if type(response) is dict
                ),
                key=lambda value: str(value["PolicyName"]),
            )
            attachments = sorted(
                str(item.get("PolicyArn"))
                for item in attached
                if type(item) is dict and type(item.get("PolicyArn")) is str
            )
            version_documents = sorted(
                (
                    {
                        "PolicyArn": response.get("PolicyArn"),
                        "Version": response.get("PolicyVersion"),
                    }
                    for response in versions
                    if type(response) is dict
                ),
                key=lambda value: str(value["PolicyArn"]),
            )
            statements = [
                statement
                for policy in inline
                for statement in policy["PolicyDocument"].get("Statement", [])
                if type(policy["PolicyDocument"]) is dict
                and type(statement) is dict
            ]
            passrole = tuple(
                sorted(
                    str(resource)
                    for statement in statements
                    if "iam:PassRole" in (
                        [statement.get("Action")]
                        if type(statement.get("Action")) is str
                        else statement.get("Action", [])
                    )
                    for resource in (
                        [statement.get("Resource")]
                        if type(statement.get("Resource")) is str
                        else statement.get("Resource", [])
                    )
                )
            )
            return {
                "role_arn": identity,
                "assume_role_policy_sha256": canonical_sha256(
                    cls._json_document(
                        role_response.get("AssumeRolePolicyDocument"),
                        label="IAM assume-role policy",
                    )
                ),
                "path": role_response.get("Path"),
                "permissions_boundary_arn": (
                    permissions_boundary.get("PermissionsBoundaryArn")
                    if type(permissions_boundary) is dict
                    else None
                ),
                "tags": _iam_role_tags(role_response.get("Tags", [])),
                "instance_profile_arns": tuple(
                    sorted(
                        str(item.get("Arn"))
                        for item in profiles
                        if type(item) is dict and type(item.get("Arn")) is str
                    )
                ),
                "inline_policy_sha256": canonical_sha256(inline),
                "managed_policy_versions_sha256": canonical_sha256(
                    version_documents
                ),
                "attachment_sha256": canonical_sha256(attachments),
                "passrole_targets": passrole,
            }
        if family == "eventbridge":
            described = cls._first_response(details, 0)
            targets = details[1].get("items")
            return {
                "rule_arn": identity,
                "state": described.get("State"),
                "targets_sha256": canonical_sha256(targets),
            }
        if family == "scheduler":
            schedule = cls._first_response(details, 0)
            return {
                "schedule_arn": identity,
                "state": schedule.get("State"),
                "target_sha256": canonical_sha256(schedule.get("Target")),
            }
        if family == "sqs":
            attributes = cls._first_response(details, 0).get("Attributes")
            if type(attributes) is not dict:
                raise ValueError("SQS attributes are absent")
            return {
                "queue_arn": attributes.get("QueueArn"),
                "approximate_messages": int(
                    attributes.get("ApproximateNumberOfMessages", "-1")
                ),
                "redrive_policy_sha256": canonical_sha256(
                    cls._json_document(
                        attributes.get("RedrivePolicy"),
                        label="SQS redrive policy",
                    )
                ),
            }
        if family == "sns":
            attributes = cls._first_response(details, 0).get("Attributes")
            if type(attributes) is not dict:
                raise ValueError("SNS subscription attributes are absent")
            pending = attributes.get("PendingConfirmation")
            return {
                "subscription_arn": identity,
                "protocol": attributes.get("Protocol"),
                "confirmed": pending in (False, "false"),
                "confirmation_authenticated": expected[
                    "confirmation_authenticated"
                ],
            }
        if family == "ec2":
            volumes = [
                item
                for detail in details[0].get("responses", [])
                if type(detail) is dict
                for item in detail.get("Volumes", [])
                if type(item) is dict
            ]
            userdata = cls._first_response(details, 2).get("UserData", {})
            tags = {
                str(tag.get("Key")): tag.get("Value")
                for tag in inventory_item.get("Tags", [])
                if type(tag) is dict and type(tag.get("Key")) is str
            }
            mappings = inventory_item.get("BlockDeviceMappings")
            if (
                type(mappings) is not list
                or len(mappings) != 2
                or len(volumes) != 2
            ):
                raise ValueError(
                    "EC2 attached EBS volume cardinality is not exact"
                )
            mapping_ids = [
                mapping.get("Ebs", {}).get("VolumeId")
                for mapping in mappings
                if type(mapping) is dict
                and type(mapping.get("Ebs")) is dict
            ]
            volume_ids = [volume.get("VolumeId") for volume in volumes]
            if (
                len(mapping_ids) != 2
                or any(
                    type(volume_id) is not str or not volume_id
                    for volume_id in (*mapping_ids, *volume_ids)
                )
                or len(set(volume_ids)) != 2
                or set(mapping_ids) != set(volume_ids)
            ):
                raise ValueError(
                    "EC2 attached EBS volume inventory is inconsistent"
                )
            attachment_rows: list[tuple[str, Mapping[str, object]]] = []
            for volume in volumes:
                attachments = volume.get("Attachments")
                if (
                    type(attachments) is not list
                    or len(attachments) != 1
                    or type(attachments[0]) is not dict
                    or attachments[0].get("InstanceId") != identity
                    or attachments[0].get("State") != "attached"
                    or type(attachments[0].get("Device")) is not str
                    or not attachments[0]["Device"]
                ):
                    raise ValueError(
                        "EC2 volume attachment is not exact and active"
                    )
                attachment_rows.append(
                    (str(attachments[0]["Device"]), volume)
                )
            roots = [
                (device, volume)
                for device, volume in attachment_rows
                if device in {"/dev/sda1", "/dev/xvda"}
            ]
            data_rows = [
                (device, volume)
                for device, volume in attachment_rows
                if device not in {"/dev/sda1", "/dev/xvda"}
            ]
            if len(roots) != 1 or len(data_rows) != 1:
                raise ValueError(
                    "EC2 root/data volume attachment roles are not exact"
                )
            root_device, root = roots[0]
            data_device, data = data_rows[0]
            attachment_contract = sorted(
                (
                    {"device": root_device, "role": "root"},
                    {"device": data_device, "role": "data"},
                ),
                key=lambda item: str(item["device"]),
            )
            network = {
                "SubnetId": inventory_item.get("SubnetId"),
                "SecurityGroupIds": sorted(
                    str(group.get("GroupId"))
                    for group in inventory_item.get("SecurityGroups", [])
                    if type(group) is dict
                    and type(group.get("GroupId")) is str
                ),
                "PrivateIpAddress": inventory_item.get("PrivateIpAddress"),
                "NetworkInterfaces": None,
            }
            return {
                "instance_id": identity,
                "role": tags.get("glm52-role"),
                "instance_type": inventory_item.get("InstanceType"),
                "state": inventory_item.get("State", {}).get("Name"),
                "ami_id": inventory_item.get("ImageId"),
                "user_data_sha256": canonical_sha256(userdata.get("Value")),
                "profile_arn": inventory_item.get(
                    "IamInstanceProfile", {}
                ).get("Arn"),
                "network_sha256": canonical_sha256(network),
                "root_volume_gib": root.get("Size"),
                "root_volume_type": root.get("VolumeType"),
                "root_volume_iops": root.get("Iops"),
                "root_volume_throughput_mibps": root.get("Throughput"),
                "root_volume_encrypted": root.get("Encrypted"),
                "data_volume_gib": data.get("Size"),
                "data_volume_type": data.get("VolumeType"),
                "data_volume_encrypted": data.get("Encrypted"),
                "volume_attachments_sha256": canonical_sha256(
                    attachment_contract
                ),
                "tags": tags,
            }
        if family == "s3":
            versioning = cls._first_response(details, 0).get("Status")
            policy_response = cls._first_response(details, 1)
            lifecycle_response = cls._first_response(details, 2)
            replication_response = cls._first_response(details, 3)
            policy = cls._json_document(
                policy_response.get("Policy"),
                label="S3 bucket policy",
            )
            lifecycle = (
                None
                if lifecycle_response.get("Rules") is None
                else lifecycle_response.get("Rules")
            )
            replication = (
                None
                if replication_response.get("ReplicationConfiguration") is None
                else replication_response.get("ReplicationConfiguration")
            )
            return {
                "bucket": identity,
                "bucket_class": expected["bucket_class"],
                "versioning": versioning,
                "policy_sha256": canonical_sha256(policy),
                "lifecycle": lifecycle,
                "replication": replication,
            }
        if family == "ssm":
            return {
                "instance_id": identity,
                "ping_status": inventory_item.get("PingStatus"),
                "platform_type": inventory_item.get("PlatformType"),
            }
        if family == "cloudwatch":
            metric = details[0].get("items")
            if (
                type(metric) is not list
                or len(metric) != 1
                or type(metric[0]) is not dict
            ):
                raise ValueError(
                    "CloudWatch required metric evidence is not exact"
                )
            status = metric[0].get("StatusCode")
            timestamps = metric[0].get("Timestamps")
            values = metric[0].get("Values")
            if (
                status != "Complete"
                and status != "COMPLETE"
            ) or (
                type(timestamps) is not list
                or type(values) is not list
                or not timestamps
                or len(timestamps) != len(values)
                or len(set(map(str, timestamps))) != len(timestamps)
                or any(
                    type(timestamp) is not str or not timestamp
                    for timestamp in timestamps
                )
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    for value in values
                )
            ):
                raise ValueError(
                    "CloudWatch metric evidence coverage is not aligned "
                    "and nonempty"
                )
            return {
                "alarm_arn": identity,
                "state": inventory_item.get("StateValue"),
                "metric_status": "COMPLETE",
                "metric_evidence": "ALIGNED_NONEMPTY",
                "treat_missing_data": inventory_item.get("TreatMissingData"),
            }
        if family == "logs":
            return {
                "log_group_arn": identity,
                "retention_days": inventory_item.get("retentionInDays"),
                "kms_key_arn": inventory_item.get("kmsKeyId"),
            }
        raise ValueError(f"{family} has no explicit AWS response normalizer")

    def _detail_argument_sets(
        self,
        *,
        family: str,
        command_index: int,
        identity: str,
        inventory_item: object,
        detail_values: Sequence[Mapping[str, object]],
    ) -> list[list[str]]:
        if family == "cloudformation":
            return [["--stack-name", identity]]
        if family == "lambda":
            if type(inventory_item) is not dict:
                raise ValueError("Lambda inventory item is malformed")
            function_name = _exact_string(
                inventory_item.get("FunctionName"),
                label="Lambda FunctionName",
            )
            if command_index in {3, 4}:
                return [["--function-name", function_name]]
            function_arn = _exact_string(
                inventory_item.get("FunctionArn"),
                label="Lambda FunctionArn",
            )
            version = inventory_item.get("Version")
            if (
                type(version) is not str
                or not version.isdigit()
                or int(version) < 1
                or not function_arn.endswith(f":{version}")
            ):
                raise ValueError(
                    "Lambda detail identity is not a published version"
                )
            return [["--function-name", function_arn]]
        if family == "iam":
            if type(inventory_item) is not dict:
                raise ValueError("IAM inventory item is malformed")
            role_name = _exact_string(
                inventory_item.get("RoleName"),
                label="IAM RoleName",
            )
            if command_index in {1, 2, 3, 5}:
                return [["--role-name", role_name]]
            if command_index == 4:
                names = _unique_sorted_strings(
                    detail_values[2].get("items", []),
                    label="IAM inline policy inventory",
                )
                return [
                    [
                        "--role-name",
                        role_name,
                        "--policy-name",
                        name,
                    ]
                    for name in names
                ]
            if command_index == 6:
                attached = detail_values[4].get("items", [])
                if type(attached) is not list:
                    raise ValueError(
                        "IAM attached policy inventory is malformed"
                    )
                arns = _unique_sorted_strings(
                    [
                        item.get("PolicyArn")
                        for item in attached
                        if type(item) is dict
                    ],
                    label="IAM attached policy ARN inventory",
                )
                if len(arns) != len(attached):
                    raise ValueError(
                        "IAM attached policy inventory is malformed"
                    )
                return [["--policy-arn", arn] for arn in arns]
            if command_index == 7:
                responses = detail_values[5].get("responses", [])
                if type(responses) is not list:
                    raise ValueError("IAM managed policy reads are malformed")
                pairs: list[tuple[str, str]] = []
                for response in responses:
                    policy = (
                        response.get("Policy")
                        if type(response) is dict
                        else None
                    )
                    if type(policy) is not dict:
                        raise ValueError(
                            "IAM managed policy response is malformed"
                        )
                    pairs.append(
                        (
                            _exact_string(
                                policy.get("Arn"),
                                label="IAM managed policy ARN",
                            ),
                            _exact_string(
                                policy.get("DefaultVersionId"),
                                label="IAM managed policy version",
                            ),
                        )
                    )
                if len(pairs) != len(set(pairs)):
                    raise ValueError(
                        "IAM managed policy version is duplicated"
                    )
                return [
                    [
                        "--policy-arn",
                        arn,
                        "--version-id",
                        version,
                    ]
                    for arn, version in sorted(pairs)
                ]
        if family == "eventbridge":
            if type(inventory_item) is not dict:
                raise ValueError("EventBridge inventory item is malformed")
            name = _exact_string(
                inventory_item.get("Name"),
                label="EventBridge rule name",
            )
            return [["--name" if command_index == 1 else "--rule", name]]
        if family == "scheduler":
            if type(inventory_item) is not dict:
                raise ValueError("Scheduler inventory item is malformed")
            name = _exact_string(
                inventory_item.get("Name"),
                label="Scheduler schedule name",
            )
            group = _exact_string(
                inventory_item.get("GroupName", "default"),
                label="Scheduler group name",
            )
            return [["--name", name, "--group-name", group]]
        if family == "sqs":
            return [
                [
                    "--queue-url",
                    identity,
                    "--attribute-names",
                    "All",
                ]
            ]
        if family == "sns":
            return [["--subscription-arn", identity]]
        if family == "ec2":
            if type(inventory_item) is not dict:
                raise ValueError("EC2 inventory item is malformed")
            if command_index == 1:
                mappings = inventory_item.get("BlockDeviceMappings")
                if type(mappings) is not list:
                    raise ValueError(
                        "EC2 block-device inventory is malformed"
                    )
                volume_ids = _unique_sorted_strings(
                    [
                        mapping.get("Ebs", {}).get("VolumeId")
                        for mapping in mappings
                        if type(mapping) is dict
                        and type(mapping.get("Ebs")) is dict
                    ],
                    label="EC2 volume inventory",
                )
                if len(volume_ids) != len(mappings):
                    raise ValueError(
                        "EC2 block-device inventory is malformed"
                    )
                return [
                    ["--volume-ids", volume_id]
                    for volume_id in volume_ids
                ]
            if command_index == 2:
                image_id = _exact_string(
                    inventory_item.get("ImageId"),
                    label="EC2 ImageId",
                )
                return [["--image-ids", image_id]]
            if command_index == 3:
                return [
                    [
                        "--instance-id",
                        identity,
                        "--attribute",
                        "userData",
                    ]
                ]
        if family == "s3":
            return [["--bucket", identity]]
        if family == "dynamodb":
            if type(inventory_item) is not dict:
                raise ValueError("DynamoDB inventory item is malformed")
            if command_index == 1:
                key = {
                    "PK": inventory_item.get("PK"),
                    "SK": inventory_item.get("SK"),
                }
                if any(type(value) is not dict for value in key.values()):
                    raise ValueError("DynamoDB item key is malformed")
                return [
                    [
                        "--table-name",
                        _TRUSTED_SOURCE_TABLE,
                        "--key",
                        canonical_json_bytes(key).decode("utf-8"),
                        "--consistent-read",
                    ]
                ]
            return [
                [
                    "--table-name",
                    _TRUSTED_SOURCE_TABLE,
                    "--consistent-read",
                ]
            ]
        if family == "cloudwatch":
            if type(inventory_item) is not dict:
                raise ValueError("CloudWatch alarm inventory is malformed")
            metric = {
                "Namespace": _exact_string(
                    inventory_item.get("Namespace"),
                    label="CloudWatch metric namespace",
                ),
                "MetricName": _exact_string(
                    inventory_item.get("MetricName"),
                    label="CloudWatch metric name",
                ),
                "Dimensions": inventory_item.get("Dimensions", []),
            }
            period = inventory_item.get("Period")
            statistic = inventory_item.get("Statistic")
            if (
                type(period) is not int
                or period <= 0
                or type(statistic) is not str
                or not statistic
            ):
                raise ValueError("CloudWatch alarm metric is malformed")
            end = self.clock().astimezone(timezone.utc)
            start = end.timestamp() - max(period * 2, 60)
            start_time = datetime.fromtimestamp(
                start, timezone.utc
            ).isoformat().replace("+00:00", "Z")
            end_time = end.isoformat().replace("+00:00", "Z")
            query = [
                {
                    "Id": "h1d",
                    "MetricStat": {
                        "Metric": metric,
                        "Period": period,
                        "Stat": statistic,
                    },
                    "ReturnData": True,
                }
            ]
            return [
                [
                    "--metric-data-queries",
                    canonical_json_bytes(query).decode("utf-8"),
                    "--start-time",
                    start_time,
                    "--end-time",
                    end_time,
                    "--scan-by",
                    "TimestampDescending",
                ]
            ]
        raise ValueError(
            f"{family} command {command_index} has no closed argument plan"
        )

    def _read_detail_value(
        self,
        *,
        family: str,
        command_index: int,
        argument_sets: Sequence[Sequence[str]],
        response_identities: list[str],
        service_request_ids: list[str],
    ) -> Mapping[str, object]:
        pagination = _DETAIL_PAGINATION.get((family, command_index))
        responses: list[dict[str, object]] = []
        detail_items: list[object] = []
        for base_arguments in argument_sets:
            detail_token: Optional[str] = None
            seen_detail_tokens: set[str] = set()
            while True:
                detail_arguments = list(base_arguments)
                if detail_token is not None:
                    if pagination is None:
                        raise ValueError(
                            "non-paginated detail returned a token"
                        )
                    detail_arguments.extend((pagination[2], detail_token))
                raw_response = self._command(
                    family,
                    command_index,
                    detail_arguments,
                )
                response, service_request_id = _service_response(raw_response)
                if service_request_id is not None:
                    service_request_ids.append(service_request_id)
                responses.append(response)
                response_identities.append(canonical_sha256(response))
                if pagination is None:
                    break
                page_items = _path(response, pagination[0])
                if type(page_items) is not list:
                    raise ValueError(
                        "CLI detail page items are not an array"
                    )
                detail_items.extend(page_items)
                next_detail_token = _optional_terminal_path(
                    response, pagination[1]
                )
                if next_detail_token is None:
                    break
                encoded_detail_token = _encoded_pagination_token(
                    next_detail_token,
                    token_argument=pagination[2],
                )
                if encoded_detail_token in seen_detail_tokens:
                    raise ValueError(
                        "CLI detail pagination token repeated or malformed"
                    )
                seen_detail_tokens.add(encoded_detail_token)
                detail_token = encoded_detail_token
        detail_value: dict[str, object] = {"responses": responses}
        if pagination is not None:
            detail_value["items"] = detail_items
        return detail_value

    def _read_ec2_inventory(
        self,
        spec: LiveReadSpec,
    ) -> tuple[
        dict[str, object],
        list[str],
        list[str],
    ]:
        by_instance: dict[str, object] = {}
        response_identities: list[str] = []
        service_request_ids: list[str] = []
        for base_arguments in _closed_inventory_argument_sets(spec):
            token: Optional[str] = None
            seen_tokens: set[str] = set()
            while True:
                arguments = list(base_arguments)
                if token is not None:
                    arguments.extend(("--next-token", token))
                raw_response = self._command("ec2", 0, arguments)
                response, request_id = _service_response(raw_response)
                if request_id is not None:
                    service_request_ids.append(request_id)
                response_identities.append(canonical_sha256(response))
                instances = response.get("Instances")
                if type(instances) is not list or any(
                    type(item) is not dict for item in instances
                ):
                    raise ValueError(
                        "EC2 union inventory response is malformed"
                    )
                for item in instances:
                    instance_id = _exact_string(
                        item.get("InstanceId"),
                        label="EC2 union instance ID",
                    )
                    prior = by_instance.get(instance_id)
                    if (
                        prior is not None
                        and canonical_sha256(prior)
                        != canonical_sha256(item)
                    ):
                        raise ValueError(
                            "EC2 union returned inconsistent duplicate state"
                        )
                    by_instance[instance_id] = item
                next_token = _optional_terminal_path(
                    response, ("NextToken",)
                )
                if next_token is None:
                    break
                if (
                    type(next_token) is not str
                    or not next_token
                    or next_token in seen_tokens
                ):
                    raise ValueError(
                        "EC2 union pagination token repeated or malformed"
                    )
                seen_tokens.add(next_token)
                token = next_token
        return (
            {
                "Instances": [
                    by_instance[instance_id]
                    for instance_id in sorted(by_instance)
                ]
            },
            response_identities,
            service_request_ids,
        )

    def _read_scoped_inventory_union(
        self,
        spec: LiveReadSpec,
    ) -> tuple[list[object], list[str], list[str]]:
        items_path, token_path, token_argument = _INVENTORY_PAGINATION[
            spec.family
        ]
        if token_argument is None:
            raise ValueError("scoped AWS inventory cannot paginate exactly")
        by_identity: dict[str, object] = {}
        response_identities: list[str] = []
        service_request_ids: list[str] = []
        for base_arguments in _closed_inventory_argument_sets(spec):
            token: Optional[str] = None
            seen_tokens: set[str] = set()
            while True:
                arguments = list(base_arguments)
                if token is not None:
                    arguments.extend((token_argument, token))
                raw = self._command(spec.family, 0, arguments)
                response, request_id = _service_response(raw)
                response_identities.append(canonical_sha256(response))
                if request_id is not None:
                    service_request_ids.append(request_id)
                page_items = _path(response, items_path)
                if type(page_items) is not list:
                    raise ValueError("scoped AWS inventory page is malformed")
                for item in page_items:
                    identity = self._inventory_identity(spec.family, item)
                    prior = by_identity.get(identity)
                    if (
                        prior is not None
                        and canonical_sha256(prior) != canonical_sha256(item)
                    ):
                        raise ValueError(
                            "scoped AWS inventory union is inconsistent"
                        )
                    by_identity[identity] = item
                next_token = _optional_terminal_path(response, token_path)
                if next_token is None:
                    break
                encoded = _encoded_pagination_token(
                    next_token,
                    token_argument=str(token_argument),
                )
                if encoded in seen_tokens:
                    raise ValueError(
                        "scoped AWS inventory pagination repeated"
                    )
                seen_tokens.add(encoded)
                token = encoded
        return (
            [by_identity[identity] for identity in sorted(by_identity)],
            response_identities,
            service_request_ids,
        )

    def read_page(
        self,
        *,
        spec: LiveReadSpec,
        continuation_token: Optional[str],
    ) -> LiveReadPage:
        prior_index = self._page_indices.get(spec.family)
        if continuation_token is None:
            page_index = 0
            self._page_indices[spec.family] = page_index
        elif prior_index is None:
            raise ValueError("CLI pagination cannot begin from a continuation")
        else:
            page_index = prior_index + 1
            self._page_indices[spec.family] = page_index
        plan = spec.parameters.get("cli_plan")
        if type(plan) is not dict or not (
            (
                set(plan) == {"schema_version", "projections"}
                and plan.get("schema_version") == 2
                and type(plan.get("projections")) is dict
            )
            or (
                set(plan) == {"schema_version", "normalizer"}
                and plan.get("schema_version") == 3
                and plan.get("normalizer")
                == f"{spec.family}.aws_response_v1"
            )
        ):
            raise ValueError(f"{spec.family} has no closed CLI projection plan")
        items_path, token_path, token_argument = _INVENTORY_PAGINATION[
            spec.family
        ]
        scoped_union_families = {
            "eventbridge",
            "scheduler",
            "sqs",
            "cloudwatch",
            "logs",
        }
        if spec.family == "ec2":
            if continuation_token is not None:
                raise ValueError(
                    "EC2 union inventory has no outer continuation"
                )
            (
                inventory,
                response_identities,
                service_request_ids,
            ) = self._read_ec2_inventory(spec)
            inventory_items_override: Optional[list[object]] = None
        elif spec.family in scoped_union_families:
            if continuation_token is not None:
                raise ValueError(
                    "scoped AWS inventory union has no outer continuation"
                )
            (
                inventory_items_override,
                response_identities,
                service_request_ids,
            ) = self._read_scoped_inventory_union(spec)
            inventory = {}
        else:
            inventory_items_override = None
            arguments = _closed_inventory_arguments(spec)
            if continuation_token is not None:
                if token_argument is None:
                    raise ValueError(
                        "CLI pagination token argument is invalid"
                    )
                arguments.extend((token_argument, continuation_token))
            raw_inventory = self._command(spec.family, 0, arguments)
            inventory, inventory_request_id = _service_response(raw_inventory)
            response_identities = [canonical_sha256(inventory)]
            service_request_ids = (
                [] if inventory_request_id is None
                else [inventory_request_id]
            )
        inventory_items = (
            inventory_items_override
            if inventory_items_override is not None
            else _path(inventory, items_path)
        )
        if type(inventory_items) is not list or any(
            type(item) not in (dict, str) for item in inventory_items
        ):
            raise ValueError("CLI inventory response is not an identity array")
        if spec.family in {
            "cloudformation",
            "lambda",
            "iam",
            "eventbridge",
            "scheduler",
            "sqs",
            "s3",
        }:
            inventory_items = [
                item
                for item in inventory_items
                if _inventory_item_in_scope(spec, item)
            ]
        projections = plan.get("projections", {})
        items: list[Mapping[str, object]] = []
        expanded_items: list[
            tuple[object, Mapping[int, Mapping[str, object]]]
        ] = [(item, {}) for item in inventory_items]
        if spec.family == "lambda":
            expanded_items = []
            seen_versions: set[str] = set()
            for base_item in inventory_items:
                base_identity = self._inventory_identity(
                    spec.family, base_item
                )
                arguments = self._detail_argument_sets(
                    family="lambda",
                    command_index=4,
                    identity=base_identity,
                    inventory_item=base_item,
                    detail_values=(),
                )
                versions_value = self._read_detail_value(
                    family="lambda",
                    command_index=4,
                    argument_sets=arguments,
                    response_identities=response_identities,
                    service_request_ids=service_request_ids,
                )
                versions = versions_value.get("items")
                if type(versions) is not list:
                    raise ValueError(
                        "Lambda published-version inventory is malformed"
                    )
                for version_item in versions:
                    if type(version_item) is not dict:
                        raise ValueError(
                            "Lambda published version is malformed"
                        )
                    version = version_item.get("Version")
                    arn = version_item.get("FunctionArn")
                    if version == "$LATEST" and arn == base_identity:
                        continue
                    if (
                        type(version) is not str
                        or not version.isdigit()
                        or int(version) < 1
                        or type(arn) is not str
                        or arn != f"{base_identity}:{version}"
                        or arn in seen_versions
                    ):
                        raise ValueError(
                            "Lambda published version identity is malformed"
                        )
                    seen_versions.add(arn)
                    expanded_items.append(
                        (version_item, {4: versions_value})
                    )
        for inventory_item, precomputed in expanded_items:
            identity = self._inventory_identity(
                spec.family, inventory_item
            )
            if spec.family == "dynamodb":
                if type(inventory_item) is not dict:
                    raise ValueError(
                        "DynamoDB inventory item is malformed"
                    )
                record_type = _path(
                    inventory_item,
                    ("record_type", "S"),
                )
                items.append(
                    {
                        "key": identity,
                        "record_type": _exact_string(
                            record_type,
                            label="DynamoDB record_type",
                        ),
                        "body_sha256": canonical_sha256(inventory_item),
                        "consistent_read": True,
                    }
                )
                continue
            detail_values: list[Mapping[str, object]] = []
            for command_index in range(1, len(READ_COMMANDS[spec.family])):
                detail_value = precomputed.get(command_index)
                if detail_value is None:
                    argument_sets = self._detail_argument_sets(
                        family=spec.family,
                        command_index=command_index,
                        identity=identity,
                        inventory_item=inventory_item,
                        detail_values=detail_values,
                    )
                    detail_value = self._read_detail_value(
                        family=spec.family,
                        command_index=command_index,
                        argument_sets=argument_sets,
                        response_identities=response_identities,
                        service_request_ids=service_request_ids,
                    )
                detail_values.append(detail_value)
            sources: list[object] = [inventory_item, *detail_values]
            if (
                plan.get("schema_version") == 3
                and plan.get("normalizer")
                == f"{spec.family}.aws_response_v1"
            ):
                items.append(
                    self._normalize_aws_item(
                        spec=spec,
                        identity=identity,
                        inventory_item=inventory_item,
                        details=detail_values,
                    )
                )
                continue
            projected: dict[str, object] = {}
            for field, projection in projections.items():
                if (
                    type(field) is not str
                    or type(projection) is not dict
                ):
                    raise ValueError("CLI projection is invalid")
                if set(projection) == {"source", "path", "transform"}:
                    source = projection["source"]
                    if (
                        type(source) is not int
                        or not 0 <= source < len(sources)
                    ):
                        raise ValueError("CLI projection source is invalid")
                    value: object = _path(
                        sources[source],
                        projection["path"],
                    )
                elif set(projection) == {"sources", "transform"}:
                    source_specs = projection["sources"]
                    if type(source_specs) is not list or not source_specs:
                        raise ValueError("CLI projection sources are invalid")
                    values: list[object] = []
                    for source_spec in source_specs:
                        if (
                            type(source_spec) is not dict
                            or set(source_spec) != {"source", "path"}
                            or type(source_spec["source"]) is not int
                            or not 0
                            <= source_spec["source"]
                            < len(sources)
                        ):
                            raise ValueError(
                                "CLI projection source is invalid"
                            )
                        values.append(
                            _path(
                                sources[source_spec["source"]],
                                source_spec["path"],
                            )
                        )
                    value = values
                else:
                    raise ValueError("CLI projection is invalid")
                transform = projection["transform"]
                if transform == "identity":
                    if "sources" in projection:
                        raise ValueError(
                            "identity projection must have one source"
                        )
                    projected[field] = value
                elif transform == "canonical_sha256":
                    projected[field] = canonical_sha256(value)
                else:
                    raise ValueError("CLI projection transform is invalid")
            items.append(projected)
        raw_next_token = (
            None
            if spec.family == "ec2" or spec.family in scoped_union_families
            else _optional_terminal_path(inventory, token_path)
        )
        next_token = (
            None
            if raw_next_token is None
            else _encoded_pagination_token(
                raw_next_token,
                token_argument=str(token_argument),
            )
        )
        observed_at = self.clock().astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        return LiveReadPage(
            family=spec.family,
            operation=spec.operation,
            request_token=continuation_token,
            page_index=page_index,
            items=tuple(items),
            next_token=next_token,
            request_id=(
                service_request_ids[0]
                if service_request_ids
                else hashlib.sha256(
                    canonical_json_bytes(tuple(response_identities))
                ).hexdigest()
            ),
            observed_at=observed_at,
            service_request_ids=tuple(service_request_ids),
        )


class AwsSdkLiveReader:
    """Concurrent, deadline-bounded production walk over in-process clients."""

    def __init__(
        self,
        *,
        runner: AwsSdkCommandRunner,
        clock: Callable[[], datetime] = _now,
        max_workers: int = SDK_MAX_WORKERS,
        executor_factory: Callable[..., object] = ThreadPoolExecutor,
        page_reader_factory: Callable[..., object] = AwsCliLiveReader,
    ) -> None:
        if (
            not isinstance(max_workers, int)
            or max_workers < 1
            or max_workers > SDK_MAX_WORKERS
        ):
            raise ValueError("SDK reader concurrency is outside the fixed bound")
        self.runner = runner
        self.clock = clock
        self.max_workers = max_workers
        self._executor_factory = executor_factory
        self._page_reader_factory = page_reader_factory
        self._pages: dict[str, tuple[LiveReadPage, ...]] = {}
        self._positions: dict[str, int] = {}
        self._prepared = False
        self._guard: Optional[_LiveReadGuard] = None

    def _collect_family(
        self,
        spec: LiveReadSpec,
    ) -> tuple[LiveReadPage, ...]:
        reader = self._page_reader_factory(
            runner=self.runner,
            clock=self.clock,
            read_guard=self._guard,
        )
        pages: list[LiveReadPage] = []
        token: Optional[str] = None
        seen_tokens: set[str] = set()
        while True:
            if self._guard is not None:
                self._guard.check()
            page = reader.read_page(
                spec=spec,
                continuation_token=token,
            )
            if self._guard is not None:
                self._guard.check()
            pages.append(page)
            if page.next_token is None:
                return tuple(pages)
            if (
                type(page.next_token) is not str
                or not page.next_token
                or page.next_token in seen_tokens
            ):
                raise ValueError(
                    "SDK family pagination token repeated or malformed"
                )
            seen_tokens.add(page.next_token)
            token = page.next_token

    def _cancel(
        self,
        executor: object,
        futures: Sequence[Future[object]],
    ) -> None:
        if self._guard is not None:
            self._guard.cancel()
        for future in futures:
            future.cancel()
        shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            shutdown(wait=True, cancel_futures=True)

    def prepare(
        self,
        specs: Tuple[LiveReadSpec, ...],
        *,
        deadline: datetime,
    ) -> None:
        if (
            self._prepared
            or type(specs) is not tuple
            or not specs
            or len({spec.family for spec in specs}) != len(specs)
            or not isinstance(deadline, datetime)
            or deadline.tzinfo is None
        ):
            raise ValueError("SDK concurrent preparation contract is invalid")
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("SDK reader clock is not timezone-aware")
        remaining = (
            deadline.astimezone(timezone.utc)
            - now.astimezone(timezone.utc)
        ).total_seconds()
        if remaining <= 0:
            raise ValueError("SDK live-read deadline already elapsed")
        self._guard = _LiveReadGuard(timeout_seconds=remaining)
        executor = self._executor_factory(
            max_workers=min(self.max_workers, len(specs)),
            thread_name_prefix="glm52-h1d",
        )
        submit = getattr(executor, "submit", None)
        shutdown = getattr(executor, "shutdown", None)
        if not callable(submit) or not callable(shutdown):
            raise ValueError("SDK executor boundary is incomplete")
        futures: dict[Future[object], str] = {
            submit(self._collect_family, spec): spec.family
            for spec in specs
        }
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            self._cancel(executor, tuple(futures))
            raise ValueError("SDK reader clock is not timezone-aware")
        remaining = max(
            0.0,
            (
                deadline.astimezone(timezone.utc)
                - now.astimezone(timezone.utc)
            ).total_seconds(),
        )
        done, pending = wait(
            tuple(futures),
            timeout=remaining,
            return_when=FIRST_EXCEPTION,
        )
        failed = [
            future for future in done
            if future.exception() is not None
        ]
        if pending or failed:
            self._cancel(executor, tuple(pending))
            if failed:
                raise ValueError("SDK concurrent family read failed") from (
                    failed[0].exception()
                )
            raise ValueError("SDK concurrent family read exceeded deadline")
        shutdown(wait=True, cancel_futures=False)
        if self._guard is not None:
            self._guard.check()
        for future, family in futures.items():
            pages = future.result()
            if type(pages) is not tuple or not pages:
                raise ValueError("SDK family read returned no typed pages")
            self._pages[family] = pages
            self._positions[family] = 0
        self._prepared = True

    def read_page(
        self,
        *,
        spec: LiveReadSpec,
        continuation_token: Optional[str],
    ) -> LiveReadPage:
        pages = self._pages.get(spec.family)
        position = self._positions.get(spec.family)
        if (
            not self._prepared
            or pages is None
            or position is None
            or position >= len(pages)
        ):
            raise ValueError("SDK live reader was not prepared")
        page = pages[position]
        if (
            page.family != spec.family
            or page.operation != spec.operation
            or page.request_token != continuation_token
        ):
            raise ValueError("SDK cached page request identity differs")
        self._positions[spec.family] = position + 1
        return page


class AwsCliSpendObjectStore:
    """Exact S3 VersionId reader for the spend namespace and source manifests."""

    def __init__(
        self,
        *,
        runner: AwsCliCommandRunner,
        bucket: str,
        exact_versions: Mapping[str, str],
    ) -> None:
        self.runner = runner
        self.bucket = bucket
        self.exact_versions = dict(exact_versions)

    def get_object(self, *, key: str) -> SpendObject:
        version_id = self.exact_versions.get(key)
        if type(version_id) is not str or not version_id:
            raise ValueError("S3 object is not bound to an exact VersionId")
        metadata, raw = self.runner.get_s3_object(
            bucket=self.bucket,
            key=key,
            version_id=version_id,
            timeout_seconds=5,
        )
        checksum = hashlib.sha256(raw).hexdigest()
        response_version = metadata.get("VersionId")
        if response_version != version_id:
            raise ValueError("S3 get-object returned the wrong VersionId")
        _metadata_payload, service_request_id = _service_response(metadata)
        return SpendObject(
            key=key,
            raw=raw,
            version_id=version_id,
            etag=str(metadata.get("ETag", "")),
            checksum_sha256=checksum,
            request_id=(
                service_request_id
                if service_request_id is not None
                else canonical_sha256(metadata)
            ),
            observed_at=_iso_now(),
        )

    def list_namespace(
        self,
        *,
        prefix: str,
        continuation_token: Optional[str],
    ) -> SpendListPage:
        arguments = [
            "s3api",
            "list-object-versions",
            "--bucket",
            self.bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1000",
            "--no-paginate",
        ]
        if continuation_token is not None:
            token = json.loads(continuation_token)
            if type(token) is not dict or set(token) != {
                "key_marker",
                "version_marker",
            }:
                raise ValueError("S3 continuation token is malformed")
            arguments.extend(("--key-marker", str(token["key_marker"])))
            if token["version_marker"] is not None:
                arguments.extend(
                    ("--version-id-marker", str(token["version_marker"]))
                )
        response = self.runner.run_json(tuple(arguments), timeout_seconds=5)
        response_payload, service_request_id = _service_response(response)
        response = response_payload
        if response.get("DeleteMarkers"):
            raise ValueError("spend namespace contains a delete marker")
        versions = response.get("Versions", [])
        if type(versions) is not list:
            raise ValueError("S3 version inventory is malformed")
        entries = [
            (item.get("Key"), item.get("VersionId"))
            for item in versions
            if type(item) is dict
        ]
        if any(
            type(key) is not str
            or type(version) is not str
            or not version
            for key, version in entries
        ):
            raise ValueError("S3 version identity is malformed")
        for key, version in entries:
            self.exact_versions[str(key)] = str(version)
        next_token: Optional[str] = None
        if response.get("IsTruncated") is True:
            marker = response.get("NextKeyMarker")
            if type(marker) is not str or not marker:
                raise ValueError("truncated S3 page lacks NextKeyMarker")
            next_token = json.dumps(
                {
                    "key_marker": marker,
                    "version_marker": response.get("NextVersionIdMarker"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        return SpendListPage(
            keys=tuple(str(key) for key, _version in entries),
            version_ids=tuple(str(version) for _key, version in entries),
            next_token=next_token,
            request_id=(
                service_request_id
                if service_request_id is not None
                else canonical_sha256(response)
            ),
            observed_at=_iso_now(),
        )


class AwsCliSpendEc2:
    def __init__(self, *, runner: AwsCliCommandRunner) -> None:
        self.runner = runner

    def describe_allocation_history(
        self, *, run_id: str, next_token: Optional[str]
    ) -> dict[str, object]:
        if run_id != RUN_ID:
            raise ValueError("EC2 spend history requested a foreign run")
        arguments = [
            "ec2",
            "describe-instances",
            "--filters",
            f"Name=tag:campaign-run-id,Values={RUN_ID}",
            "Name=instance-type,Values=p5.48xlarge",
            "--max-results",
            "100",
            "--no-paginate",
        ]
        if next_token is not None:
            arguments.extend(("--next-token", next_token))
        response = self.runner.run_json(tuple(arguments), timeout_seconds=5)
        response_payload, service_request_id = _service_response(response)
        response = response_payload
        instances: list[dict[str, object]] = []
        for reservation in response.get("Reservations", []):
            for item in reservation.get("Instances", []):
                tags = {
                    tag["Key"]: tag["Value"]
                    for tag in item.get("Tags", [])
                    if type(tag) is dict
                    and type(tag.get("Key")) is str
                    and type(tag.get("Value")) is str
                }
                placement = item.get("Placement", {})
                state = item.get("State", {})
                instances.append(
                    {
                        "InstanceId": item.get("InstanceId"),
                        "InstanceType": item.get("InstanceType"),
                        "InstanceLifecycle": item.get("InstanceLifecycle"),
                        "State": state.get("Name"),
                        "LaunchTime": _canonical_utc(
                            item.get("LaunchTime"),
                            label="EC2 LaunchTime",
                        ),
                        "AvailabilityZone": placement.get("AvailabilityZone"),
                        "Tags": tags,
                    }
                )
        return {
            "instances": instances,
            "next_token": response.get("NextToken"),
            "request_id": (
                service_request_id
                if service_request_id is not None
                else canonical_sha256(response)
            ),
            "observed_at": _iso_now(),
        }


class AwsCliReserveReader:
    """Enumerate every immutable held-liability record from the fixed run."""

    PREFIX = f"campaigns/{RUN_ID}/spend-ledger/reserves/"

    def __init__(self, *, store: AwsCliSpendObjectStore) -> None:
        self.store = store

    def list_reserves(
        self, *, next_token: Optional[str]
    ) -> ReserveListPage:
        page = self.store.list_namespace(
            prefix=self.PREFIX,
            continuation_token=next_token,
        )
        records: list[Mapping[str, object]] = []
        for key in page.keys:
            if not key.startswith(self.PREFIX) or not key.endswith(".json"):
                raise ValueError("GPU reserve namespace contains a foreign key")
            observed = self.store.get_object(key=key)
            try:
                value = json.loads(observed.raw)
            except json.JSONDecodeError as exc:
                raise ValueError("GPU reserve record is malformed JSON") from exc
            if (
                type(value) is not dict
                or canonical_json_bytes(value) + b"\n" != observed.raw
            ):
                raise ValueError("GPU reserve record is not canonical JSON")
            records.append(value)
        return ReserveListPage(
            records=tuple(records),
            next_token=page.next_token,
            request_id=page.request_id,
            observed_at=page.observed_at,
        )


def build_spend_services(
    *,
    runner: object,
    store: AwsCliSpendObjectStore,
) -> SpendAuthorityServices:
    """Build the production spend boundary with mandatory reserve enumeration."""

    return SpendAuthorityServices(
        object_store=store,
        ec2=AwsCliSpendEc2(runner=runner),
        reserve_reader=AwsCliReserveReader(store=store),
    )


class _SpendInspector:
    def __init__(self, request: object, services: SpendAuthorityServices) -> None:
        self.request = request
        self.services = services

    def inspect(self, request: object) -> object:
        if request is not self.request:
            raise ValueError("live inspection substituted the spend request")
        return inspect_spend_authority(self.request, self.services)


class _LambdaAttestationBridge:
    def __init__(
        self,
        *,
        runner: object,
        function_arn: str,
        expected_sky_identity: Mapping[str, object],
        activation_id: str,
        deployed_identity_coordinate: Mapping[str, object],
        deployed_identity_sha256: str,
    ) -> None:
        self._runner = runner
        self._function_arn = function_arn
        self._activation_id = activation_id
        self._deployed_identity_coordinate = dict(
            deployed_identity_coordinate
        )
        self._deployed_identity_sha256 = deployed_identity_sha256
        attestation_identity = next(
            (
                item
                for item in expected_sky_identity["identities"]
                if item["purpose"] == "ATTESTATION"
            ),
            None,
        )
        if attestation_identity is None:
            raise ValueError("Task 9 attestation relay identity is absent")
        self._expected_live_identity = {
            "tls_peer_certificate_sha256": (
                attestation_identity["server_certificate_sha256"]
            ),
            "sky_user_identity": expected_sky_identity[
                "service_account_user_id"
            ],
            "sky_roles": [expected_sky_identity["service_account_role"]],
            "effective_controller_identity_sha256": expected_sky_identity[
                "consolidation_signal_identity_sha256"
            ],
            "sky_identity_contract_sha256": expected_sky_identity[
                "canonical_identity_sha256"
            ],
        }

    def attest(self, request: AttestationRequest) -> AttestationResult:
        if not isinstance(request, AttestationRequest):
            raise ValueError("Task 9 attestation request is not typed")
        invoke = getattr(self._runner, "invoke_task9_attestation", None)
        if not callable(invoke):
            raise ValueError("Task 9 attestation Lambda bridge is absent")
        raw = invoke(
            function_arn=self._function_arn,
            payload=canonical_json_bytes(
                {
                    "schema_version": 1,
                    "record_type": "glm52_task11_attestation_request_v1",
                    "run_id": RUN_ID,
                    "activation_id": self._activation_id,
                    "request_identity_sha256": canonical_sha256(
                        {
                            "domain": "H1D_TASK9_ATTESTATION_V1",
                            "freshness_nonce": request.freshness_nonce,
                        }
                    ),
                    "freshness_nonce": request.freshness_nonce,
                    "sequence": 1,
                    "task9_deployed_identity_coordinate": (
                        self._deployed_identity_coordinate
                    ),
                    "task9_deployed_identity_sha256": (
                        self._deployed_identity_sha256
                    ),
                }
            ),
            timeout_seconds=5,
        )
        fields = {
            "schema_version",
            "record_type",
            "account_id",
            "region",
            "run_id",
            "freshness_nonce",
            "direct_response_request_id",
            "tls_peer_certificate_sha256",
            "sky_user_identity",
            "sky_roles",
            "token_expires_at",
            "effective_controller_identity_sha256",
            "observed_at",
            "sky_identity_contract_sha256",
            "canonical_identity_sha256",
        }
        if type(raw) is not dict or set(raw) != fields:
            raise ValueError("Task 9 attestation result schema drifted")
        body = dict(raw)
        identity = body.pop("canonical_identity_sha256")
        if (
            raw["schema_version"] != 1
            or raw["record_type"] != "glm52_sky_attestation_v1"
            or raw["account_id"] != ACCOUNT_ID
            or raw["region"] != REGION
            or raw["run_id"] != RUN_ID
            or raw["freshness_nonce"] != request.freshness_nonce
            or identity != canonical_sha256(body)
        ):
            raise ValueError("Task 9 attestation result identity drifted")
        if {
            field: raw[field]
            for field in self._expected_live_identity
        } != self._expected_live_identity:
            raise ValueError(
                "Task 9 attestation authenticated identity drifted"
            )
        try:
            return AttestationResult(
                **{
                    **raw,
                    "sky_roles": tuple(raw["sky_roles"]),
                }
            )
        except TypeError as exc:
            raise ValueError("Task 9 attestation result is malformed") from exc


class ProductionTask9Probe:
    """Default H.1d probe through the exact published attestation Lambda."""

    def __init__(
        self,
        *,
        runner: object,
        expected: object,
        deployed_identity_coordinate: Mapping[str, object],
        campaign_bucket: str,
        deployed_identity: Optional[Mapping[str, object]] = None,
        nonce_source: Callable[[], str] = lambda: secrets.token_urlsafe(48),
    ) -> None:
        self._runner = runner
        self._expected = expected
        self._nonce_source = nonce_source
        self._coordinate = exact_input_coordinate_from_mapping(
            deployed_identity_coordinate,
            expected_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
        )
        if self._coordinate.bucket != campaign_bucket:
            raise ValueError("Task 9 deployed identity bucket is foreign")
        if self._coordinate.key != (
            "campaigns/"
            + RUN_ID
            + "/authorities/task9/"
            + expected.activation_id
            + "/TASK9_DEPLOYED_IDENTITY.json"
        ):
            raise ValueError("Task 9 deployed identity key is foreign")
        self._deployed_identity = deployed_identity

    def _load_deployed_identity(self) -> Mapping[str, object]:
        if self._deployed_identity is not None:
            value = self._deployed_identity
        else:
            get_object = getattr(self._runner, "get_s3_object", None)
            if not callable(get_object):
                raise ValueError("Task 9 deployed identity reader is absent")
            metadata, raw = get_object(
                bucket=self._coordinate.bucket,
                key=self._coordinate.key,
                version_id=self._coordinate.version_id,
                timeout_seconds=5,
            )
            if (
                type(metadata) is not dict
                or metadata.get("VersionId") != self._coordinate.version_id
                or type(raw) is not bytes
                or hashlib.sha256(raw).hexdigest()
                != self._coordinate.file_sha256
            ):
                raise ValueError("Task 9 deployed identity readback drifted")
            try:
                value = json.loads(raw[:-1].decode("ascii"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Task 9 deployed identity is malformed"
                ) from exc
            if (
                not raw.endswith(b"\n")
                or raw.endswith(b"\n\n")
                or canonical_json_bytes(value) + b"\n" != raw
                or value.get("canonical_identity_sha256")
                != self._coordinate.body_sha256
            ):
                raise ValueError("Task 9 deployed identity bytes drifted")
        static = build_task9_contract(ROOT)
        try:
            validate_task9_deployed_identity(
                value,
                static_contract=static,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Task 9 deployed identity drifted") from exc
        if value["activation_id"] != self._expected.activation_id:
            raise ValueError("Task 9 deployed activation drifted")
        if (
            value["canonical_identity_sha256"]
            != self._coordinate.body_sha256
        ):
            raise ValueError("Task 9 deployed body identity drifted")
        return value

    def _build_probe(self) -> Task9SkyRelayProbe:
        function_arns = [
            item["function_arn"]
            for spec in self._expected.specs
            if spec.family == "lambda"
            for item in spec.expected_items
            if (
                "function:keep-glm52-h1g-attestation:"
                in item.get("function_arn", "")
            )
            and not item["function_arn"].endswith(":$LATEST")
        ]
        if len(function_arns) != 1:
            raise ValueError(
                "exact published Task 9 attestation function is absent"
            )
        deployed = self._load_deployed_identity()
        sky_identity = deployed["sky_identity"]
        launch_identity = next(
            item
            for item in sky_identity["identities"]
            if item["purpose"] == "LAUNCH_ADMISSION"
        )
        return Task9SkyRelayProbe(
            attestation=_LambdaAttestationBridge(
                runner=self._runner,
                function_arn=function_arns[0],
                expected_sky_identity=sky_identity,
                activation_id=self._expected.activation_id,
                deployed_identity_coordinate={
                    field: getattr(self._coordinate, field)
                    for field in self._coordinate.__dataclass_fields__
                },
                deployed_identity_sha256=deployed[
                    "canonical_identity_sha256"
                ],
            ),
            admission_identity_sha256=canonical_sha256(launch_identity),
            nonce_source=self._nonce_source,
        )

    def inspect(self, request: Mapping[str, object]) -> object:
        return self._build_probe().inspect(request)


class _IdentityAdapter:
    def __init__(
        self,
        *,
        runner: AwsCliCommandRunner,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.runner = runner
        self.clock = clock

    def get_caller_identity(self) -> CallerIdentityObservation:
        identity = self.runner.run_json(
            ("sts", "get-caller-identity"),
            timeout_seconds=5,
        )
        identity_payload, service_request_id = _service_response(identity)
        identity = identity_payload
        return CallerIdentityObservation(
            account_id=str(identity["Account"]),
            arn=str(identity["Arn"]),
            user_id=str(identity["UserId"]),
            credential_expiration=self.runner.credential_expiration(),
            request_id=(
                service_request_id
                if service_request_id is not None
                else canonical_sha256(identity)
            ),
            observed_at=self.clock().astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        )


class AwsExpectedStateAuthority:
    """Exact-read Task 6/7 source artifacts before any family inspection."""

    def __init__(
        self,
        *,
        store: AwsCliSpendObjectStore,
        sources: Mapping[str, object],
        trusted_sources: Optional[Mapping[str, object]] = None,
        trusted_source_reader: Optional[object] = None,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        if (trusted_sources is None) == (trusted_source_reader is None):
            raise ValueError(
                "exactly one trusted campaign source boundary is required"
            )
        self.store = store
        self.sources = _validated_authority_sources(sources)
        self.trusted_sources = (
            None
            if trusted_sources is None
            else _validated_authority_sources(trusted_sources)
        )
        self.trusted_source_reader = trusted_source_reader
        self.clock = clock

    def _read(self, name: str) -> tuple[dict[str, object], str, str]:
        source = self.sources.get(name)
        if type(source) is not dict or set(source) != {
            "key",
            "version_id",
            "file_sha256",
        }:
            raise ValueError(f"{name} source coordinate is not exact")
        self.store.exact_versions[str(source["key"])] = str(source["version_id"])
        observed = self.store.get_object(key=str(source["key"]))
        file_sha = hashlib.sha256(observed.raw).hexdigest()
        if file_sha != source["file_sha256"]:
            raise ValueError(f"{name} exact bytes drifted")
        try:
            value = json.loads(observed.raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is malformed JSON") from exc
        if type(value) is not dict or canonical_json_bytes(value) + b"\n" != observed.raw:
            raise ValueError(f"{name} is not canonical JSON")
        return value, file_sha, observed.request_id

    @staticmethod
    def _resource_identity(
        row: Mapping[str, object],
    ) -> Optional[tuple[str, str, str]]:
        resource_type = row.get("resource_type")
        physical = row.get("physical_id")
        if type(resource_type) is not str or type(physical) is not str:
            raise ValueError("Task 7 inventory resource row is malformed")
        if resource_type == "AWS::Lambda::Version":
            return "lambda", "function_arn", physical
        if resource_type == "AWS::IAM::Role":
            arn = (
                physical
                if physical.startswith("arn:")
                else f"arn:aws:iam::{ACCOUNT_ID}:role/{physical}"
            )
            return "iam", "role_arn", arn
        if resource_type == "AWS::Events::Rule":
            arn = (
                physical
                if physical.startswith("arn:")
                else f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/{physical}"
            )
            return "eventbridge", "rule_arn", arn
        if resource_type == "AWS::Scheduler::Schedule":
            arn = (
                physical
                if physical.startswith("arn:")
                else (
                    f"arn:aws:scheduler:{REGION}:{ACCOUNT_ID}:"
                    f"schedule/default/{physical.rsplit('/', 1)[-1]}"
                )
            )
            return "scheduler", "schedule_arn", arn
        if resource_type == "AWS::SQS::Queue":
            name = physical.rsplit("/", 1)[-1]
            return (
                "sqs",
                "queue_arn",
                f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:{name}",
            )
        if resource_type == "AWS::SNS::Subscription":
            return "sns", "subscription_arn", physical
        if resource_type == "AWS::EC2::Instance":
            return "ec2", "instance_id", physical
        if resource_type == "AWS::S3::Bucket":
            return "s3", "bucket", physical
        if resource_type == "AWS::CloudWatch::Alarm":
            arn = (
                physical
                if physical.startswith("arn:")
                else f"arn:aws:cloudwatch:{REGION}:{ACCOUNT_ID}:alarm:{physical}"
            )
            return "cloudwatch", "alarm_arn", arn
        if resource_type == "AWS::Logs::LogGroup":
            arn = (
                physical
                if physical.startswith("arn:")
                else (
                    f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:"
                    f"log-group:{physical}"
                )
            )
            return "logs", "log_group_arn", arn
        return None

    @staticmethod
    def _specs_identity(expected: object) -> str:
        return canonical_sha256(
            tuple(
                {
                    "family": spec.family,
                    "operation": spec.operation,
                    "parameters": dict(spec.parameters),
                    "identity_field": spec.identity_field,
                    "expected_items": tuple(
                        dict(item) for item in spec.expected_items
                    ),
                }
                for spec in expected.specs
            )
        )

    @staticmethod
    def _properties(
        resources: Mapping[str, object],
        logical_id: object,
        expected_type: str,
    ) -> Mapping[str, object]:
        resource = resources.get(logical_id)
        if (
            type(logical_id) is not str
            or type(resource) is not dict
            or resource.get("Type") != expected_type
            or type(resource.get("Properties", {})) is not dict
        ):
            raise ValueError(
                f"Task 7 {expected_type} source resource is absent"
            )
        return resource.get("Properties", {})

    @staticmethod
    def _physical_resources(
        rows: Sequence[Mapping[str, object]],
    ) -> tuple[dict[str, Mapping[str, object]], dict[str, str]]:
        row_by_logical: dict[str, Mapping[str, object]] = {}
        physical_by_logical: dict[str, str] = {}
        for row in rows:
            logical = row.get("logical_id")
            physical = row.get("physical_id")
            if (
                type(logical) is not str
                or not logical
                or type(physical) is not str
                or not physical
                or logical in row_by_logical
            ):
                raise ValueError(
                    "Task 7 resource inventory has malformed identities"
                )
            row_by_logical[logical] = row
            physical_by_logical[logical] = physical
        return row_by_logical, physical_by_logical

    @staticmethod
    def _resolve_ref(
        value: object,
        physical_by_logical: Mapping[str, str],
    ) -> object:
        if type(value) is dict and set(value) == {"Ref"}:
            return physical_by_logical.get(str(value["Ref"]), value)
        if (
            type(value) is dict
            and set(value) == {"Fn::GetAtt"}
            and type(value["Fn::GetAtt"]) is list
            and len(value["Fn::GetAtt"]) == 2
        ):
            logical, attribute = value["Fn::GetAtt"]
            physical = physical_by_logical.get(str(logical))
            if physical is None:
                return value
            if attribute == "Arn" and physical.startswith("arn:"):
                return physical
            return {"physical_id": physical, "attribute": attribute}
        if type(value) is dict:
            return {
                str(key): AwsExpectedStateAuthority._resolve_ref(
                    item, physical_by_logical
                )
                for key, item in value.items()
            }
        if type(value) is list:
            return [
                AwsExpectedStateAuthority._resolve_ref(
                    item, physical_by_logical
                )
                for item in value
            ]
        return value

    @classmethod
    def _resolve_iam_resource_arn(
        cls,
        value: object,
        *,
        resources: Mapping[str, object],
        physical_by_logical: Mapping[str, str],
        expected_type: str,
        resource_kind: str,
    ) -> str:
        logical: object = None
        if type(value) is dict and set(value) == {"Ref"}:
            logical = value["Ref"]
        elif (
            type(value) is dict
            and set(value) == {"Fn::GetAtt"}
            and type(value["Fn::GetAtt"]) is list
            and len(value["Fn::GetAtt"]) == 2
            and value["Fn::GetAtt"][1] == "Arn"
        ):
            logical = value["Fn::GetAtt"][0]
        if logical is None:
            return _iam_arn(value, resource_kind=resource_kind)
        props = cls._properties(resources, logical, expected_type)
        physical = physical_by_logical.get(str(logical))
        return _iam_arn(
            physical,
            resource_kind=resource_kind,
            path=props.get("Path", "/"),
        )

    @classmethod
    def _resolve_lambda_function_arn(
        cls,
        value: object,
        *,
        qualifier: object,
        resources: Mapping[str, object],
        physical_by_logical: Mapping[str, str],
    ) -> str:
        logical: object = None
        if type(value) is dict and set(value) == {"Ref"}:
            logical = value["Ref"]
            physical = physical_by_logical.get(str(logical))
        else:
            physical = value
        physical_value = _exact_string(
            physical, label="Lambda permission function"
        )
        if physical_value.startswith("arn:"):
            function_arn = physical_value
        else:
            if logical is not None:
                resource = resources.get(logical)
                if type(resource) is not dict or resource.get("Type") not in {
                    "AWS::Lambda::Function",
                    "AWS::Lambda::Version",
                }:
                    raise ValueError(
                        "Lambda permission function source is malformed"
                    )
            function_arn = (
                f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
                f"function:{physical_value}"
            )
        if qualifier is not None:
            resolved_qualifier = cls._resolve_ref(
                qualifier, physical_by_logical
            )
            qualifier_value = _exact_string(
                resolved_qualifier, label="Lambda permission qualifier"
            )
            unqualified_prefix = (
                f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:"
            )
            suffix = function_arn.removeprefix(unqualified_prefix)
            if ":" in suffix:
                if not function_arn.endswith(f":{qualifier_value}"):
                    raise ValueError(
                        "Lambda permission qualifier conflicts with function"
                    )
            else:
                function_arn = f"{function_arn}:{qualifier_value}"
        return function_arn

    @classmethod
    def _resolve_lambda_permission_source_arn(
        cls,
        value: object,
        *,
        permission_logical: str,
        support_stack_id: object,
        resources: Mapping[str, object],
        row_by_logical: Mapping[str, Mapping[str, object]],
        physical_by_logical: Mapping[str, str],
    ) -> object:
        if value is None:
            return None
        if value == {"Ref": "AWS::StackId"}:
            stack_id = row_by_logical.get(permission_logical, {}).get(
                "stack_id", support_stack_id
            )
            return _exact_string(
                stack_id, label="Lambda permission stack source ARN"
            )
        if (
            type(value) is dict
            and set(value) == {"Fn::GetAtt"}
            and type(value["Fn::GetAtt"]) is list
            and len(value["Fn::GetAtt"]) == 2
            and value["Fn::GetAtt"][1] == "Arn"
        ):
            logical = value["Fn::GetAtt"][0]
            resource = resources.get(logical)
            physical = physical_by_logical.get(str(logical))
            if type(resource) is not dict:
                raise ValueError("Lambda permission source ARN is malformed")
            if resource.get("Type") == "AWS::Events::Rule":
                name = _exact_string(
                    physical, label="EventBridge rule identity"
                )
                if name.startswith("arn:"):
                    return name
                return (
                    f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/{name}"
                )
        resolved = cls._resolve_ref(value, physical_by_logical)
        return _exact_string(
            resolved, label="Lambda permission source ARN"
        )

    @classmethod
    def _derive_expected_items(
        cls,
        *,
        task6: Mapping[str, object],
        task6_file_sha256: str,
        support_template: Mapping[str, object],
        inventory: Mapping[str, object],
        activation_id: str,
        sealed_dynamodb_expected_items: Optional[
            Sequence[Mapping[str, object]]
        ],
        sealed_cloudformation_expected_items: Optional[
            Sequence[Mapping[str, object]]
        ],
        sealed_cloudformation_deployment_role: Optional[
            Mapping[str, object]
        ],
    ) -> Mapping[str, Tuple[Mapping[str, object], ...]]:
        """Derive the complete desired live contract without caller claims."""
        resources = support_template.get("Resources")
        rows = inventory.get("stack_resources")
        if (
            type(resources) is not dict
            or type(rows) is not list
            or any(type(row) is not dict for row in rows)
        ):
            raise ValueError("Task 7 semantic source inventory is absent")
        row_by_logical, physical = cls._physical_resources(rows)
        support_stack_id = inventory.get("support_stack_id")
        if type(support_stack_id) is str and support_stack_id:
            physical["AWS::StackId"] = support_stack_id
        physical["AWS::AccountId"] = ACCOUNT_ID
        physical["AWS::Region"] = REGION
        physical["AWS::Partition"] = "aws"
        source_identities: dict[str, list[tuple[str, str]]] = {
            family: [] for family in READ_COMMANDS
        }
        for logical, row in row_by_logical.items():
            projected = cls._resource_identity(row)
            if projected is not None:
                family, _field, identity = projected
                source_identities[family].append((identity, logical))

        stacks = task6.get("stacks")
        if type(stacks) is not list or any(type(row) is not dict for row in stacks):
            raise ValueError("Task 6 stack inventory is absent")
        if (
            type(sealed_cloudformation_expected_items) not in (list, tuple)
            or len(sealed_cloudformation_expected_items) != 3
            or any(
                type(item) is not dict
                for item in sealed_cloudformation_expected_items
            )
        ):
            raise ValueError(
                "sealed CloudFormation expected inventory is absent"
            )
        cloudformation = [
            dict(item) for item in sealed_cloudformation_expected_items
        ]
        source_stack_by_id = {
            stack.get("stack_id"): stack for stack in stacks
        }
        source_stack_ids = set(source_stack_by_id)
        deployment_role = sealed_cloudformation_deployment_role
        if (
            len(source_stack_by_id) != 3
            or any(
                type(stack.get("tags")) is not dict
                or any(
                    type(key) is not str
                    or not key
                    or type(value) is not str
                    for key, value in stack.get("tags", {}).items()
                )
                for stack in stacks
            )
            or {item.get("stack_id") for item in cloudformation}
            != source_stack_ids
            or type(deployment_role) is not dict
            or set(deployment_role)
            != {"role_arn", "role_id", "request_id"}
            or deployment_role.get("role_id")
            != task6.get("retained_deployment_role_id")
            or type(deployment_role.get("role_id")) is not str
            or _IAM_ROLE_ID.fullmatch(str(deployment_role["role_id"]))
            is None
            or type(deployment_role.get("role_arn")) is not str
            or not str(deployment_role["role_arn"]).startswith(
                f"arn:aws:iam::{ACCOUNT_ID}:role/"
            )
            or type(deployment_role.get("request_id")) is not str
            or not deployment_role["request_id"]
            or any(
                item.get("manifest_identity_sha256")
                != task6_file_sha256
                or item.get("status") != "UPDATE_COMPLETE"
                or item.get("service_role_arn")
                != deployment_role["role_arn"]
                or _cloudformation_stack_tags(
                    item.get("stack_tags"),
                    canonical_shape=True,
                )
                != [
                    {"key": key, "value": value}
                    for key, value in sorted(
                        source_stack_by_id[
                            item.get("stack_id")
                        ].get("tags", {}).items()
                    )
                ]
                for item in cloudformation
            )
        ):
            raise ValueError(
                "sealed CloudFormation expected inventory is foreign"
            )

        all_logs = tuple(
            sorted(identity for identity, _ in source_identities["logs"])
        )
        all_alarms = tuple(
            sorted(identity for identity, _ in source_identities["cloudwatch"])
        )
        all_queues = tuple(
            sorted(identity for identity, _ in source_identities["sqs"])
        )
        lambda_items: list[Mapping[str, object]] = []
        for identity, version_logical in source_identities["lambda"]:
            version_props = cls._properties(
                resources, version_logical, "AWS::Lambda::Version"
            )
            function_ref = version_props.get("FunctionName")
            function_logical = (
                function_ref.get("Ref")
                if type(function_ref) is dict
                else None
            )
            function_props = cls._properties(
                resources, function_logical, "AWS::Lambda::Function"
            )
            configuration_fields = tuple(
                sorted(
                    key
                    for key in function_props
                    if key not in {"Code", "ReservedConcurrentExecutions"}
                )
            )
            unsupported_configuration = (
                set(configuration_fields) - _LAMBDA_CONFIGURATION_FIELDS
            )
            if unsupported_configuration:
                raise ValueError(
                    "Task 7 Lambda configuration contains unsupported "
                    "code-owned fields"
                )
            source_configuration = {
                key: value
                for key, value in function_props.items()
                if key in configuration_fields
            }
            configuration = cls._resolve_ref(
                source_configuration,
                physical,
            )
            if "Role" in source_configuration:
                configuration["Role"] = cls._resolve_iam_resource_arn(
                    source_configuration["Role"],
                    resources=resources,
                    physical_by_logical=physical,
                    expected_type="AWS::IAM::Role",
                    resource_kind="role",
                )
            if (
                "VpcConfig" in source_configuration
                and (
                    type(source_configuration["VpcConfig"]) is not dict
                    or set(source_configuration["VpcConfig"])
                    != {"SecurityGroupIds", "SubnetIds"}
                )
            ):
                raise ValueError(
                    "Task 7 Lambda VPC source contract is unsupported"
                )
            semantic_configuration = _lambda_configuration_projection(
                configuration,
                configuration_fields,
            )
            description = version_props.get("Description")
            described_code_sha = (
                re.search(r"(?:^|;)code=([0-9a-f]{64})(?:;|$)", description)
                if type(description) is str
                else None
            )
            permissions = []
            for permission_logical, resource in resources.items():
                if (
                    type(permission_logical) is not str
                    or type(resource) is not dict
                    or resource.get("Type") != "AWS::Lambda::Permission"
                    or type(resource.get("Properties")) is not dict
                ):
                    continue
                permission_props = resource["Properties"]
                permission_function_arn = (
                    cls._resolve_lambda_function_arn(
                        permission_props.get("FunctionName"),
                        qualifier=permission_props.get("Qualifier"),
                        resources=resources,
                        physical_by_logical=physical,
                    )
                )
                if permission_function_arn != identity:
                    continue
                source_account = permission_props.get("SourceAccount")
                if source_account == {"Ref": "AWS::AccountId"}:
                    source_account = ACCOUNT_ID
                permissions.append(
                    _lambda_policy_statement(
                        sid=physical.get(permission_logical),
                        action=permission_props.get("Action"),
                        principal=permission_props.get("Principal"),
                        resource=permission_function_arn,
                        source_arn=(
                            cls._resolve_lambda_permission_source_arn(
                                permission_props.get("SourceArn"),
                                permission_logical=permission_logical,
                                support_stack_id=inventory.get(
                                    "support_stack_id"
                                ),
                                resources=resources,
                                row_by_logical=row_by_logical,
                                physical_by_logical=physical,
                            )
                        ),
                        source_account=source_account,
                    )
                )
            permissions.sort(key=lambda item: str(item["Sid"]))
            mapping_contract = []
            semantic_mappings = []
            for mapping_logical, resource in resources.items():
                if (
                    type(mapping_logical) is not str
                    or type(resource) is not dict
                    or resource.get("Type")
                    != "AWS::Lambda::EventSourceMapping"
                    or type(resource.get("Properties")) is not dict
                ):
                    continue
                mapping_props = resource["Properties"]
                mapping_function_arn = (
                    cls._resolve_lambda_function_arn(
                        mapping_props.get("FunctionName"),
                        qualifier=None,
                        resources=resources,
                        physical_by_logical=physical,
                    )
                )
                if mapping_function_arn != identity:
                    continue
                mapping_fields = tuple(
                    sorted(
                        (
                            set(mapping_props) - {"FunctionName"}
                        )
                        | {"Enabled"}
                    )
                )
                if not set(mapping_fields).issubset(
                    _LAMBDA_EVENT_SOURCE_MAPPING_FIELDS
                ):
                    raise ValueError(
                        "Task 7 Lambda event-source mapping contains "
                        "unsupported code-owned fields"
                    )
                uuid = physical.get(mapping_logical)
                resolved_mapping = cls._resolve_ref(
                    mapping_props, physical
                )
                if "Enabled" not in resolved_mapping:
                    resolved_mapping["Enabled"] = True
                mapping_contract.append(
                    {"uuid": uuid, "fields": mapping_fields}
                )
                semantic_mappings.append(
                    _lambda_event_source_mapping_projection(
                        resolved_mapping,
                        uuid=uuid,
                        function_arn=mapping_function_arn,
                        fields=mapping_fields,
                    )
                )
            mapping_contract.sort(key=lambda item: str(item["uuid"]))
            semantic_mappings.sort(key=lambda item: str(item["UUID"]))
            lambda_items.append(
                {
                    "function_arn": identity,
                    "code_sha256": canonical_sha256(
                        function_props.get("Code")
                    )
                    if described_code_sha is None
                    else described_code_sha.group(1),
                    "configuration_fields": configuration_fields,
                    "configuration_sha256": canonical_sha256(
                        semantic_configuration
                    ),
                    "policy_sha256": canonical_sha256(permissions),
                    "event_source_mapping_contract": tuple(
                        mapping_contract
                    ),
                    "event_source_mappings_sha256": canonical_sha256(
                        semantic_mappings
                    ),
                    "reserved_concurrency": function_props.get(
                        "ReservedConcurrentExecutions", 1
                    ),
                    "vpc_attachment_sha256": canonical_sha256(
                        semantic_configuration.get("VpcConfig")
                    ),
                    "timeout_seconds": function_props.get("Timeout", 840),
                    "memory_mib": function_props.get("MemorySize", 1024),
                    "log_group_arn": (
                        all_logs[0] if len(all_logs) == 1 else None
                    ),
                    "alarm_arns": all_alarms,
                    "dlq_arn": (
                        semantic_configuration.get(
                            "DeadLetterConfig", {}
                        ).get("TargetArn")
                        if type(
                            semantic_configuration.get("DeadLetterConfig")
                        ) is dict
                        else (
                            all_queues[0] if len(all_queues) == 1 else None
                        )
                    ),
                }
            )

        iam_items: list[Mapping[str, object]] = []
        for identity, logical in source_identities["iam"]:
            props = cls._properties(resources, logical, "AWS::IAM::Role")
            policies = props.get("Policies", [])
            managed = cls._resolve_ref(
                props.get("ManagedPolicyArns", []), physical
            )
            statements = [
                statement
                for policy in policies
                if type(policy) is dict
                for statement in (
                    policy.get("PolicyDocument", {}).get("Statement", [])
                    if type(policy.get("PolicyDocument")) is dict
                    else []
                )
                if type(statement) is dict
            ]
            passrole_targets = tuple(
                sorted(
                    str(resource)
                    for statement in statements
                    if "iam:PassRole" in (
                        [statement.get("Action")]
                        if type(statement.get("Action")) is str
                        else statement.get("Action", [])
                    )
                    for resource in (
                        [statement.get("Resource")]
                        if type(statement.get("Resource")) is str
                        else statement.get("Resource", [])
                    )
                )
            )
            profiles = tuple(
                sorted(
                    cls._resolve_iam_resource_arn(
                        {"Ref": profile_logical},
                        resources=resources,
                        physical_by_logical=physical,
                        expected_type="AWS::IAM::InstanceProfile",
                        resource_kind="instance-profile",
                    )
                    for profile_logical in physical
                    if type(resources.get(profile_logical)) is dict
                    and resources[profile_logical].get("Type")
                    == "AWS::IAM::InstanceProfile"
                    and {"Ref": logical}
                    in resources[profile_logical].get(
                        "Properties", {}
                    ).get("Roles", [])
                )
            )
            iam_items.append(
                {
                    "role_arn": identity,
                    "assume_role_policy_sha256": canonical_sha256(
                        cls._resolve_ref(
                            props.get("AssumeRolePolicyDocument"),
                            physical,
                        )
                    ),
                    "path": props.get("Path", "/"),
                    "permissions_boundary_arn": cls._resolve_ref(
                        props.get("PermissionsBoundary"),
                        physical,
                    ),
                    "tags": _iam_role_tags(props.get("Tags", [])),
                    "instance_profile_arns": profiles,
                    "inline_policy_sha256": canonical_sha256(policies),
                    "managed_policy_versions_sha256": canonical_sha256(
                        managed
                    ),
                    "attachment_sha256": canonical_sha256(managed),
                    "passrole_targets": passrole_targets,
                }
            )

        def props_for(
            family: str, expected_type: str
        ) -> list[tuple[str, Mapping[str, object]]]:
            return [
                (
                    identity,
                    cls._properties(resources, logical, expected_type),
                )
                for identity, logical in source_identities[family]
            ]

        eventbridge = tuple(
            {
                "rule_arn": identity,
                "state": props.get("State", "ENABLED"),
                "targets_sha256": canonical_sha256(
                    cls._resolve_ref(props.get("Targets", []), physical)
                ),
            }
            for identity, props in props_for(
                "eventbridge", "AWS::Events::Rule"
            )
        )
        scheduler_items = []
        for identity, props in props_for(
            "scheduler", "AWS::Scheduler::Schedule"
        ):
            source_target = props.get("Target")
            target = cls._resolve_ref(source_target, physical)
            if (
                type(source_target) is not dict
                or type(target) is not dict
            ):
                raise ValueError("Task 7 Scheduler target is malformed")
            if "RoleArn" in source_target:
                target["RoleArn"] = cls._resolve_iam_resource_arn(
                    source_target["RoleArn"],
                    resources=resources,
                    physical_by_logical=physical,
                    expected_type="AWS::IAM::Role",
                    resource_kind="role",
                )
            scheduler_items.append(
                {
                    "schedule_arn": identity,
                    "state": props.get("State", "ENABLED"),
                    "target_sha256": canonical_sha256(target),
                }
            )
        scheduler = tuple(scheduler_items)
        sqs = tuple(
            {
                "queue_arn": identity,
                "approximate_messages": 0,
                "redrive_policy_sha256": canonical_sha256(
                    cls._resolve_ref(props.get("RedrivePolicy"), physical)
                ),
            }
            for identity, props in props_for("sqs", "AWS::SQS::Queue")
        )
        sns = tuple(
            {
                "subscription_arn": identity,
                "protocol": props.get("Protocol"),
                "confirmed": True,
                "confirmation_authenticated": True,
            }
            for identity, props in props_for(
                "sns", "AWS::SNS::Subscription"
            )
        )

        ec2: list[Mapping[str, object]] = []
        for identity, logical in source_identities["ec2"]:
            props = cls._properties(resources, logical, "AWS::EC2::Instance")
            mappings = props.get("BlockDeviceMappings", [])
            root = (
                mappings[0].get("Ebs", {})
                if type(mappings) is list
                and mappings
                and type(mappings[0]) is dict
                else {}
            )
            data_resources = [
                resource.get("Properties", {})
                for resource in resources.values()
                if type(resource) is dict
                and resource.get("Type") == "AWS::EC2::Volume"
                and type(resource.get("Properties", {})) is dict
            ]
            data = data_resources[0] if len(data_resources) == 1 else {}
            attachment_resources = [
                resource.get("Properties", {})
                for resource in resources.values()
                if type(resource) is dict
                and resource.get("Type") == "AWS::EC2::VolumeAttachment"
                and type(resource.get("Properties", {})) is dict
            ]
            data_device = (
                attachment_resources[0].get("Device")
                if len(attachment_resources) == 1
                else None
            )
            root_device = (
                mappings[0].get("DeviceName")
                if type(mappings) is list
                and len(mappings) == 1
                and type(mappings[0]) is dict
                else None
            )
            attachment_contract = sorted(
                (
                    {"device": root_device, "role": "root"},
                    {"device": data_device, "role": "data"},
                ),
                key=lambda item: str(item["device"]),
            )
            if any(
                type(item["device"]) is not str or not item["device"]
                for item in attachment_contract
            ):
                raise ValueError(
                    "Task 7 EC2 volume attachment source is incomplete"
                )
            tags = {
                str(tag.get("Key")): tag.get("Value")
                for tag in props.get("Tags", [])
                if type(tag) is dict and type(tag.get("Key")) is str
            }
            profile_source = props.get("IamInstanceProfile")
            if profile_source is not None:
                profile = cls._resolve_iam_resource_arn(
                    profile_source,
                    resources=resources,
                    physical_by_logical=physical,
                    expected_type="AWS::IAM::InstanceProfile",
                    resource_kind="instance-profile",
                )
            else:
                matching_profiles = [
                    profile_logical
                    for profile_logical in physical
                    if type(resources.get(profile_logical)) is dict
                    and resources[profile_logical].get("Type")
                    == "AWS::IAM::InstanceProfile"
                ]
                profile = (
                    cls._resolve_iam_resource_arn(
                        {"Ref": matching_profiles[0]},
                        resources=resources,
                        physical_by_logical=physical,
                        expected_type="AWS::IAM::InstanceProfile",
                        resource_kind="instance-profile",
                    )
                    if len(matching_profiles) == 1
                    else None
                )
            interfaces = props.get("NetworkInterfaces")
            primary_interface = (
                interfaces[0]
                if type(interfaces) is list
                and len(interfaces) == 1
                and type(interfaces[0]) is dict
                else {}
            )
            source_security_groups = (
                primary_interface.get("GroupSet")
                if primary_interface
                else props.get("SecurityGroupIds", [])
            )
            resolved_security_groups = cls._resolve_ref(
                source_security_groups, physical
            )
            network = {
                "SubnetId": cls._resolve_ref(
                    (
                        primary_interface.get("SubnetId")
                        if primary_interface
                        else props.get("SubnetId")
                    ),
                    physical,
                ),
                "SecurityGroupIds": sorted(resolved_security_groups),
                "PrivateIpAddress": (
                    primary_interface.get("PrivateIpAddress")
                    if primary_interface
                    else props.get("PrivateIpAddress")
                ),
                "NetworkInterfaces": None,
            }
            user_data = cls._resolve_ref(props.get("UserData"), physical)
            if type(user_data) is dict and set(user_data) == {"Fn::Base64"}:
                user_data = user_data["Fn::Base64"]
            encoded_user_data = (
                base64.b64encode(
                    canonical_json_bytes(user_data)
                    if type(user_data) is not str
                    else user_data.encode("utf-8")
                ).decode("ascii")
                if user_data is not None
                else None
            )
            ec2.append(
                {
                    "instance_id": identity,
                    "role": tags.get("glm52-role", "support-host"),
                    "instance_type": props.get("InstanceType"),
                    "state": "running",
                    "ami_id": props.get("ImageId"),
                    "user_data_sha256": canonical_sha256(
                        encoded_user_data
                    ),
                    "profile_arn": profile,
                    "network_sha256": canonical_sha256(network),
                    "root_volume_gib": root.get("VolumeSize"),
                    "root_volume_type": root.get("VolumeType"),
                    "root_volume_iops": root.get("Iops"),
                    "root_volume_throughput_mibps": root.get("Throughput"),
                    "root_volume_encrypted": root.get("Encrypted"),
                    "data_volume_gib": data.get("Size"),
                    "data_volume_type": data.get("VolumeType"),
                    "data_volume_encrypted": data.get("Encrypted"),
                    "volume_attachments_sha256": canonical_sha256(
                        attachment_contract
                    ),
                    "tags": tags,
                }
            )

        retained_bucket = task6.get("bucket_name")
        if type(retained_bucket) is not str or not retained_bucket:
            raise ValueError("Task 6 retained bucket is absent")
        bucket_policies = [
            resource.get("Properties", {})
            for resource in resources.values()
            if type(resource) is dict
            and resource.get("Type") == "AWS::S3::BucketPolicy"
            and type(resource.get("Properties", {})) is dict
        ]
        retained_policy = task6.get("canonical_policy_sha256")
        if type(retained_policy) is not str:
            retained_policy = canonical_sha256(None)
        s3: list[Mapping[str, object]] = [
            {
                "bucket": retained_bucket,
                "bucket_class": "retained_model_evidence",
                "versioning": "Enabled",
                "policy_sha256": retained_policy,
                "lifecycle": None,
                "replication": None,
            }
        ]
        for identity, logical in source_identities["s3"]:
            props = cls._properties(resources, logical, "AWS::S3::Bucket")
            matching_policy = next(
                (
                    policy.get("PolicyDocument")
                    for policy in bucket_policies
                    if cls._resolve_ref(policy.get("Bucket"), physical)
                    == identity
                ),
                None,
            )
            s3.append(
                {
                    "bucket": identity,
                    "bucket_class": "support_rehearsal",
                    "versioning": (
                        props.get("VersioningConfiguration", {}).get("Status")
                        if type(props.get("VersioningConfiguration")) is dict
                        else None
                    ),
                    "policy_sha256": canonical_sha256(matching_policy),
                    "lifecycle": props.get("LifecycleConfiguration"),
                    "replication": props.get("ReplicationConfiguration"),
                }
            )

        if sealed_dynamodb_expected_items is None:
            raise ValueError(
                "sealed activation inventory is required for DynamoDB "
                "expected state"
            )
        ddb = tuple(
            sorted(
                (dict(item) for item in sealed_dynamodb_expected_items),
                key=lambda item: str(item.get("key")),
            )
        )
        host_ids = tuple(identity for identity, _ in source_identities["ec2"])
        ssm = tuple(
            {
                "instance_id": instance_id,
                "ping_status": "Online",
                "platform_type": "Linux",
            }
            for instance_id in host_ids
        )
        cloudwatch = tuple(
            {
                "alarm_arn": identity,
                "state": "OK",
                "metric_status": "COMPLETE",
                "metric_evidence": "ALIGNED_NONEMPTY",
                "treat_missing_data": props.get("TreatMissingData"),
            }
            for identity, props in props_for(
                "cloudwatch", "AWS::CloudWatch::Alarm"
            )
        )
        logs = tuple(
            {
                "log_group_arn": identity,
                "retention_days": props.get("RetentionInDays"),
                "kms_key_arn": cls._resolve_ref(
                    props.get("KmsKeyId"), physical
                ),
            }
            for identity, props in props_for(
                "logs", "AWS::Logs::LogGroup"
            )
        )
        result: dict[str, Tuple[Mapping[str, object], ...]] = {
            "cloudformation": tuple(cloudformation),
            "lambda": tuple(lambda_items),
            "iam": tuple(iam_items),
            "eventbridge": eventbridge,
            "scheduler": scheduler,
            "sqs": sqs,
            "sns": sns,
            "ec2": tuple(ec2),
            "s3": tuple(s3),
            "dynamodb": ddb,
            "ssm": ssm,
            "cloudwatch": cloudwatch,
            "logs": logs,
        }
        if any(not result[family] for family in READ_COMMANDS):
            raise ValueError(
                "Task 6/7 sources do not bind every live family"
            )
        return {
            family: tuple(
                sorted(
                    items,
                    key=lambda item: str(
                        item[
                            {
                                "cloudformation": "stack_id",
                                "lambda": "function_arn",
                                "iam": "role_arn",
                                "eventbridge": "rule_arn",
                                "scheduler": "schedule_arn",
                                "sqs": "queue_arn",
                                "sns": "subscription_arn",
                                "ec2": "instance_id",
                                "s3": "bucket",
                                "dynamodb": "key",
                                "ssm": "instance_id",
                                "cloudwatch": "alarm_arn",
                                "logs": "log_group_arn",
                            }[family]
                        ]
                    ),
                )
            )
            for family, items in result.items()
        }

    @staticmethod
    def _validate_template_bundle(
        templates: Mapping[str, object],
        task6: Mapping[str, object],
        postcreate: Mapping[str, object],
    ) -> Mapping[str, object]:
        if set(templates) != {
            "schema_version",
            "record_type",
            "task6_templates",
            "task7_support_template",
            "h1d_specs_identity_sha256",
            "canonical_body_sha256",
        }:
            raise ValueError("Task 6/7 template bundle schema mismatch")
        body = dict(templates)
        identity = body.pop("canonical_body_sha256")
        if (
            templates.get("schema_version") != 1
            or templates.get("record_type")
            != "glm52_h1d_task6_task7_template_bundle_v1"
            or identity != canonical_sha256(body)
            or type(templates.get("task6_templates")) is not dict
            or type(templates.get("task7_support_template")) is not dict
            or _SHA256.fullmatch(
                str(templates.get("h1d_specs_identity_sha256"))
            )
            is None
        ):
            raise ValueError("Task 6/7 template bundle identity drifted")
        task6_templates = templates["task6_templates"]
        artifacts = task6.get("artifacts")
        if type(artifacts) is not list:
            raise ValueError("Task 6 manifest artifact inventory is absent")
        if set(task6_templates) != {
            str(artifact.get("stage")) for artifact in artifacts
        }:
            raise ValueError("Task 6 template stage inventory diverged")
        for artifact in artifacts:
            stage = artifact.get("stage")
            template = task6_templates.get(stage)
            if (
                type(stage) is not str
                or type(template) is not dict
                or canonical_sha256(template)
                != artifact.get("template_body_sha256")
            ):
                raise ValueError("Task 6 template body identity diverged")
        support_template = templates["task7_support_template"]
        if canonical_sha256(support_template) != postcreate.get(
            "support_template_body_sha256"
        ):
            raise ValueError("Task 7 support template body identity diverged")
        return support_template

    @staticmethod
    def _expected_inventory_arguments(spec: object) -> list[str]:
        return _closed_inventory_arguments(spec)

    @classmethod
    def _validate_closed_cli_plan(cls, spec: object) -> None:
        family = spec.family
        plan = spec.parameters.get("cli_plan")
        if (
            type(plan) is not dict
            or set(plan) != {"schema_version", "normalizer"}
            or plan.get("schema_version") != 3
            or plan.get("normalizer")
            != f"{family}.aws_response_v1"
        ):
            raise ValueError(
                f"{family.upper()} plan is not code-owned version three"
            )

    @staticmethod
    def _code_owned_cli_plan(
        family: str,
        fields: Sequence[str],
    ) -> Mapping[str, object]:
        del fields
        if family not in READ_COMMANDS:
            raise ValueError("unknown AWS normalizer family")
        return {
            "schema_version": 3,
            "normalizer": f"{family}.aws_response_v1",
        }

    @classmethod
    def _validate_expected_projection(
        cls,
        expected: object,
        *,
        task6: Mapping[str, object],
        task6_file_sha256: str,
        templates: Mapping[str, object],
        postcreate: Mapping[str, object],
        inventory: Mapping[str, object],
        sealed_dynamodb_expected_items: Optional[
            Sequence[Mapping[str, object]]
        ] = None,
        sealed_cloudformation_expected_items: Optional[
            Sequence[Mapping[str, object]]
        ] = None,
        sealed_cloudformation_deployment_role: Optional[
            Mapping[str, object]
        ] = None,
    ) -> None:
        support_template = cls._validate_template_bundle(
            templates, task6, postcreate
        )
        rows = inventory.get("stack_resources")
        if type(rows) is not list or any(type(row) is not dict for row in rows):
            raise ValueError("Task 7 resource inventory is absent")
        expected_by_family = {
            spec.family: spec for spec in expected.specs
        }
        if set(expected_by_family) != set(READ_COMMANDS):
            raise ValueError("expected live family inventory is incomplete")
        for spec in expected_by_family.values():
            cls._validate_closed_cli_plan(spec)
        resources = support_template.get("Resources")
        if type(resources) is not dict:
            raise ValueError("Task 7 support template resources are absent")
        derived_expected_items = cls._derive_expected_items(
            task6=task6,
            task6_file_sha256=task6_file_sha256,
            support_template=support_template,
            inventory=inventory,
            activation_id=expected.activation_id,
            sealed_dynamodb_expected_items=sealed_dynamodb_expected_items,
            sealed_cloudformation_expected_items=(
                sealed_cloudformation_expected_items
            ),
            sealed_cloudformation_deployment_role=(
                sealed_cloudformation_deployment_role
            ),
        )
        for family, spec in expected_by_family.items():
            actual_items = tuple(
                sorted(
                    (dict(item) for item in spec.expected_items),
                    key=lambda item: str(item.get(spec.identity_field)),
                )
            )
            if actual_items != derived_expected_items[family]:
                reason = (
                    "EC2 support shape"
                    if family == "ec2"
                    else (
                        "expected identities or configuration"
                        if family == "iam"
                        else "expected state"
                    )
                )
                raise ValueError(
                    f"{family} {reason} differs from independently "
                    "derived, source-bound Task 6/7 and sealed activation "
                    "truth"
                )
            derived_fields = tuple(derived_expected_items[family][0])
            derived_identity_field = {
                "cloudformation": "stack_id",
                "lambda": "function_arn",
                "iam": "role_arn",
                "eventbridge": "rule_arn",
                "scheduler": "schedule_arn",
                "sqs": "queue_arn",
                "sns": "subscription_arn",
                "ec2": "instance_id",
                "s3": "bucket",
                "dynamodb": "key",
                "ssm": "instance_id",
                "cloudwatch": "alarm_arn",
                "logs": "log_group_arn",
            }[family]
            derived_parameters = {
                "run_id": RUN_ID,
                "cli_plan": cls._code_owned_cli_plan(
                    family, derived_fields
                ),
            }
            if (
                spec.operation != f"{family}.inspect_complete"
                or spec.identity_field != derived_identity_field
                or dict(spec.parameters) != derived_parameters
            ):
                raise ValueError(
                    f"{family} complete expected spec body is not "
                    "code-owned"
                )
        logical_by_identity: dict[tuple[str, str], str] = {}
        for row in rows:
            projection = cls._resource_identity(row)
            logical_id = row.get("logical_id")
            if projection is not None and type(logical_id) is str:
                family, _field, identity = projection
                logical_by_identity[(family, identity)] = logical_id
        iam_item = expected_by_family["iam"].expected_items[0]
        iam_logical = logical_by_identity.get(
            ("iam", str(iam_item.get("role_arn")))
        )
        iam_resource = resources.get(iam_logical, {})
        iam_properties = (
            iam_resource.get("Properties", {})
            if type(iam_resource) is dict
            else {}
        )
        if (
            iam_resource.get("Type") != "AWS::IAM::Role"
            or iam_item.get("inline_policy_sha256")
            != canonical_sha256(iam_properties.get("Policies", []))
        ):
            raise ValueError(
                "expected identities or IAM policy configuration is not "
                "Task 7 source-bound"
            )
        lambda_item = expected_by_family["lambda"].expected_items[0]
        version_logical = logical_by_identity.get(
            ("lambda", str(lambda_item.get("function_arn")))
        )
        version_resource = resources.get(version_logical, {})
        function_ref = (
            version_resource.get("Properties", {}).get("FunctionName", {}).get(
                "Ref"
            )
            if type(version_resource) is dict
            else None
        )
        function_resource = resources.get(function_ref, {})
        function_properties = (
            function_resource.get("Properties", {})
            if type(function_resource) is dict
            else {}
        )
        if (
            version_resource.get("Type") != "AWS::Lambda::Version"
            or function_resource.get("Type") != "AWS::Lambda::Function"
            or lambda_item.get("code_sha256")
            != canonical_sha256(function_properties.get("Code"))
        ):
            raise ValueError(
                "expected Lambda code configuration is not Task 7 source-bound"
            )
        derived: dict[str, tuple[str, set[str]]] = {}
        for row in rows:
            projection = cls._resource_identity(row)
            if projection is None:
                continue
            family, field, identity = projection
            prior = derived.setdefault(family, (field, set()))
            if prior[0] != field or identity in prior[1]:
                raise ValueError("source resource identity is duplicated")
            prior[1].add(identity)
        stacks = task6.get("stacks")
        if type(stacks) is not list or any(type(row) is not dict for row in stacks):
            raise ValueError("Task 6 stack identity inventory is absent")
        derived["cloudformation"] = (
            "stack_id",
            {
                str(row.get("stack_id"))
                for row in stacks
                if type(row.get("stack_id")) is str
            },
        )
        retained_bucket = task6.get("bucket_name")
        s3_source = derived.get("s3")
        if (
            type(retained_bucket) is not str
            or not retained_bucket
            or s3_source is None
            or s3_source[0] != "bucket"
            or retained_bucket in s3_source[1]
        ):
            raise ValueError(
                "Task 6 retained or Task 7 rehearsal bucket is absent"
            )
        s3_source[1].add(retained_bucket)
        host_rows = [
            row for row in rows if row.get("logical_id") == "CombinedHost"
        ]
        if len(host_rows) != 1:
            raise ValueError("Task 7 combined-host identity is absent")
        host_id = host_rows[0]["physical_id"]
        derived["ssm"] = ("instance_id", {str(host_id)})
        ddb_items = expected_by_family["dynamodb"].expected_items
        activation_prefix = (
            f"RUN#{RUN_ID}|ACTIVATION#{expected.activation_id}"
        )
        required_ddb_types = {
            f"RUN#{RUN_ID}|ACTIVATION_INDEX":
                "glm52_production_activation_index",
            f"{activation_prefix}#CONTROL": "glm52_production_control",
        }
        actual_ddb_items = tuple(
            sorted(
                (dict(item) for item in ddb_items),
                key=lambda item: str(item.get("key")),
            )
        )
        if sealed_dynamodb_expected_items is not None:
            sealed_ddb_items = tuple(
                sorted(
                    (
                        dict(item)
                        for item in sealed_dynamodb_expected_items
                    ),
                    key=lambda item: str(item.get("key")),
                )
            )
            if actual_ddb_items != sealed_ddb_items:
                raise ValueError(
                    "DynamoDB expected inventory differs from sealed source"
                )
        ddb_types = {
            str(item.get("key")): item.get("record_type")
            for item in ddb_items
            if type(item) is dict
        }
        if (
            len(ddb_types) != len(ddb_items)
            or any(
                ddb_types.get(key) != record_type
                for key, record_type in required_ddb_types.items()
            )
            or not any(
                re.fullmatch(
                    (
                        f"{re.escape(activation_prefix)}#"
                        r"ACTION#[0-9]{8}#SKY#[0-9]{8}"
                    ),
                    key,
                )
                and record_type == "glm52_production_action"
                for key, record_type in ddb_types.items()
            )
            or any(
                not key.startswith(f"RUN#{RUN_ID}|")
                for key in ddb_types
            )
        ):
            raise ValueError(
                "DynamoDB activation records are not source-bound"
            )
        derived["dynamodb"] = ("key", set(ddb_types))
        for item in expected_by_family["cloudformation"].expected_items:
            if (
                item.get("status") != "UPDATE_COMPLETE"
                or item.get("termination_protection") is not True
            ):
                raise ValueError(
                    "CloudFormation live semantics are not source-bound"
                )
        for family in ("eventbridge", "scheduler"):
            if any(
                item.get("state") != "ENABLED"
                for item in expected_by_family[family].expected_items
            ):
                raise ValueError(
                    f"{family} live semantics are not source-bound"
                )
        retained_items = [
            item for item in expected_by_family["s3"].expected_items
            if item.get("bucket") == retained_bucket
        ]
        rehearsal_items = [
            item for item in expected_by_family["s3"].expected_items
            if item.get("bucket") != retained_bucket
        ]
        if (
            len(retained_items) != 1
            or len(rehearsal_items) != 1
            or retained_items[0].get("bucket_class")
            != "retained_model_evidence"
            or retained_items[0].get("versioning") != "Enabled"
            or retained_items[0].get("lifecycle") is not None
            or retained_items[0].get("replication") is not None
            or rehearsal_items[0].get("bucket_class")
            != "support_rehearsal"
            or rehearsal_items[0].get("versioning") is not None
            or rehearsal_items[0].get("lifecycle") is not None
            or rehearsal_items[0].get("replication") is not None
        ):
            raise ValueError("S3 live semantics are not source-bound")
        for family, spec in expected_by_family.items():
            source = derived.get(family)
            if source is None:
                raise ValueError(
                    f"Task 6/7 sources do not bind {family} identities"
                )
            field, identities = source
            if spec.identity_field != field:
                raise ValueError(f"{family} identity field is not source-bound")
            actual = {
                str(item.get(field))
                for item in spec.expected_items
                if type(item) is dict and type(item.get(field)) is str
            }
            if actual != identities or len(actual) != len(spec.expected_items):
                raise ValueError(
                    f"{family} expected identities differ from Task 6/7 truth"
                )
        host = (
            resources.get("CombinedHost", {}).get("Properties", {})
            if type(resources) is dict
            else {}
        )
        data = (
            resources.get("CombinedHostDataVolume", {}).get("Properties", {})
            if type(resources) is dict
            else {}
        )
        mappings = host.get("BlockDeviceMappings")
        root = (
            mappings[0].get("Ebs", {})
            if type(mappings) is list
            and len(mappings) == 1
            and type(mappings[0]) is dict
            else {}
        )
        ec2_item = expected_by_family["ec2"].expected_items[0]
        immutable_shape = {
            "instance_type": host.get("InstanceType"),
            "ami_id": host.get("ImageId"),
            "root_volume_gib": root.get("VolumeSize"),
            "root_volume_type": root.get("VolumeType"),
            "root_volume_iops": root.get("Iops"),
            "root_volume_throughput_mibps": root.get("Throughput"),
            "root_volume_encrypted": root.get("Encrypted"),
            "data_volume_gib": data.get("Size"),
            "data_volume_type": data.get("VolumeType"),
            "data_volume_encrypted": data.get("Encrypted"),
        }
        if any(
            ec2_item.get(field) != value
            for field, value in immutable_shape.items()
        ):
            raise ValueError(
                "expected EC2 support shape differs from Task 7 template"
            )

    def authenticate(self, expected: object) -> object:
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("expected-state authority clock is invalid")
        try:
            must_start_by = datetime.fromisoformat(
                expected.must_start_by.replace("Z", "+00:00")
            )
            execution_deadline = datetime.fromisoformat(
                expected.execution_deadline.replace("Z", "+00:00")
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("expected-state deadline is malformed") from exc
        if (
            must_start_by.tzinfo is None
            or execution_deadline.tzinfo is None
            or must_start_by <= now
            or must_start_by > now + timedelta(hours=12)
            or execution_deadline <= must_start_by
            or execution_deadline > now + timedelta(hours=24)
        ):
            raise ValueError(
                "expected-state deadline exceeds the code-owned campaign "
                "windows"
            )
        trusted_sources = self.trusted_sources
        trusted_source_contract_sha256: Optional[str] = None
        sealed_dynamodb_expected_items: Optional[
            Sequence[Mapping[str, object]]
        ] = None
        sealed_cloudformation_expected_items: Optional[
            Sequence[Mapping[str, object]]
        ] = None
        sealed_cloudformation_deployment_role: Optional[
            Mapping[str, object]
        ] = None
        trusted_source_request_id: Optional[str] = None
        if self.trusted_source_reader is not None:
            read_contract = getattr(
                self.trusted_source_reader,
                "read_contract",
                None,
            )
            read_sources = getattr(
                self.trusted_source_reader,
                "read_sources",
                None,
            )
            if callable(read_contract):
                trusted_contract = read_contract()
                if (
                    type(trusted_contract) is not dict
                    or set(trusted_contract)
                    != {
                        "authority_sources",
                        "source_contract_sha256",
                        "cloudformation_deployment_role",
                        "dynamodb_expected_items",
                        "cloudformation_expected_items",
                        "request_id",
                    }
                    or type(
                        trusted_contract["source_contract_sha256"]
                    )
                    is not str
                    or _SHA256.fullmatch(
                        trusted_contract["source_contract_sha256"]
                    )
                    is None
                    or type(trusted_contract["dynamodb_expected_items"])
                    is not tuple
                    or type(
                        trusted_contract[
                            "cloudformation_deployment_role"
                        ]
                    )
                    is not dict
                    or type(
                        trusted_contract[
                            "cloudformation_expected_items"
                        ]
                    )
                    is not tuple
                    or type(trusted_contract["request_id"]) is not str
                    or not trusted_contract["request_id"]
                ):
                    raise ValueError(
                        "trusted campaign source contract is malformed"
                    )
                trusted_sources = _validated_authority_sources(
                    trusted_contract["authority_sources"]
                )
                trusted_source_contract_sha256 = trusted_contract[
                    "source_contract_sha256"
                ]
                sealed_dynamodb_expected_items = trusted_contract[
                    "dynamodb_expected_items"
                ]
                sealed_cloudformation_expected_items = trusted_contract[
                    "cloudformation_expected_items"
                ]
                sealed_cloudformation_deployment_role = trusted_contract[
                    "cloudformation_deployment_role"
                ]
                trusted_source_request_id = trusted_contract["request_id"]
            elif not callable(read_sources):
                raise ValueError(
                    "trusted campaign source reader is incomplete"
                )
            else:
                trusted_sources = _validated_authority_sources(read_sources())
        if self.sources != trusted_sources:
            raise ValueError(
                "authority sources differ from trusted campaign source"
            )
        task6, task6_sha, request6 = self._read("task6_manifest")
        templates, templates_sha, request_templates = self._read(
            "task6_templates"
        )
        postcreate, _postcreate_sha, request7 = self._read(
            "task7_postcreate_manifest"
        )
        inventory, _inventory_sha, request_inventory = self._read(
            "task7_inventory"
        )
        for value, label in (
            (postcreate, "Task 7 postcreate manifest"),
            (inventory, "Task 7 inventory"),
        ):
            body = dict(value)
            identity = body.pop("canonical_body_sha256", None)
            if identity != canonical_sha256(body):
                raise ValueError(f"{label} self-hash drifted")
        if (
            task6.get("record_type")
            != "glm52_h1g_stack_migration_manifest_v1"
            or postcreate.get("record_type")
            != "glm52_h1g_support_postcreate_manifest_v1"
            or postcreate.get("canonical_body_sha256")
            != expected.manifest_identity_sha256
            or postcreate.get("support_postcreate_inventory_sha256")
            != inventory.get("canonical_body_sha256")
        ):
            raise ValueError("Task 6/7 expected-state source binding diverged")
        source_contract_sha256 = _trusted_source_contract_sha256(
            task6=task6,
            templates=templates,
            postcreate=postcreate,
            inventory=inventory,
        )
        if (
            trusted_source_contract_sha256 is not None
            and source_contract_sha256 != trusted_source_contract_sha256
        ):
            raise ValueError(
                "trusted campaign semantic source contract drifted"
            )
        self._validate_expected_projection(
            expected,
            task6=task6,
            task6_file_sha256=task6_sha,
            templates=templates,
            postcreate=postcreate,
            inventory=inventory,
            sealed_dynamodb_expected_items=(
                sealed_dynamodb_expected_items
            ),
            sealed_cloudformation_expected_items=(
                sealed_cloudformation_expected_items
            ),
            sealed_cloudformation_deployment_role=(
                sealed_cloudformation_deployment_role
            ),
        )
        direct_request_ids = (
            request6,
            request_templates,
            request7,
            request_inventory,
        )
        if trusted_source_request_id is not None:
            direct_request_ids = (
                trusted_source_request_id,
                *direct_request_ids,
            )
        return build_expected_state_authentication(
            expected_state_identity_sha256=expected.canonical_identity_sha256,
            task6_manifest_identity_sha256=task6_sha,
            task6_templates_identity_sha256=templates_sha,
            task7_postcreate_manifest_identity_sha256=postcreate[
                "canonical_body_sha256"
            ],
            task7_inventory_identity_sha256=inventory[
                "canonical_body_sha256"
            ],
            direct_read_request_ids=direct_request_ids,
            observed_at=_iso_now(),
        )


def _read_input(path: Path, expected_sha256: str) -> dict[str, object]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected-state SHA-256 is invalid")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("expected-state exact bytes differ")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("expected-state input is malformed JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError("expected-state input is not canonical JSON")
    if set(value) != {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "campaign_bucket",
        "h1d_expected_state",
        "spend_request",
        "authority_sources",
        "task9_deployed_identity_coordinate",
    }:
        raise ValueError("expected-state envelope schema mismatch")
    if (
        value["schema_version"] != 1
        or value["record_type"] != "glm52_h1d_live_inspection_input_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
        or value["campaign_bucket"] != _CAMPAIGN_BUCKET
    ):
        raise ValueError("expected-state envelope is foreign")
    return value


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--expected-state", type=Path, required=True)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    runner_factory: Callable[..., object] = AwsSdkCommandRunner,
    trusted_source_factory: Callable[
        ..., object
    ] = AwsTrustedCampaignSourceReader,
    task9_probe_factory: Callable[..., object] = ProductionTask9Probe,
) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.profile != PROFILE:
        raise ValueError("--profile must be exactly keep-gpu")
    if args.region != REGION:
        raise ValueError("--region must be exactly us-west-2")
    runner = runner_factory(profile=args.profile, region=args.region)
    identity = runner.run_json(
        ("sts", "get-caller-identity"), timeout_seconds=5
    )
    if (
        identity.get("Account") != ACCOUNT_ID
        or type(identity.get("Arn")) is not str
        or f"::{ACCOUNT_ID}:" not in identity["Arn"]
        or type(identity.get("UserId")) is not str
        or not identity["UserId"]
    ):
        raise ValueError("AWS caller is not the exact approved account")
    envelope = _read_input(args.expected_state, args.expected_state_sha256)
    trusted_source_reader = trusted_source_factory(runner=runner)
    expected = h1d_expected_state_from_mapping(envelope["h1d_expected_state"])
    spend_request = spend_authority_request_from_mapping(
        envelope["spend_request"]
    )
    exact_versions = {
        spend_request.descriptor_key: spend_request.descriptor_version_id,
        spend_request.approval_key: spend_request.approval_version_id,
        spend_request.latest_key: spend_request.latest_version_id,
        spend_request.snapshot_key: spend_request.snapshot_version_id,
    }
    store = AwsCliSpendObjectStore(
        runner=runner,
        bucket=str(envelope["campaign_bucket"]),
        exact_versions=exact_versions,
    )
    spend_services = build_spend_services(runner=runner, store=store)
    request = H1dLiveAuthorityRequest(
        profile=PROFILE,
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id=expected.activation_id,
        expected_state_identity_sha256=expected.canonical_identity_sha256,
        spend_request=spend_request,
        sky_probe_request={
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "activation_id": expected.activation_id,
            "run_id": RUN_ID,
        },
    )
    services = H1dLiveServices(
        identity=_IdentityAdapter(
            runner=runner,
        ),
        expected_state_authority=AwsExpectedStateAuthority(
            store=store,
            sources=envelope["authority_sources"],
            trusted_source_reader=trusted_source_reader,
            clock=_now,
        ),
        reader=AwsSdkLiveReader(runner=runner),
        spend=_SpendInspector(spend_request, spend_services),
        sky_relay_probe=task9_probe_factory(
            runner=runner,
            expected=expected,
            deployed_identity_coordinate=envelope[
                "task9_deployed_identity_coordinate"
            ],
            campaign_bucket=envelope["campaign_bucket"],
        ),
        clock=_now,
    )
    result = inspect_h1d_live_authority(request, expected, services)
    evidence = serialize_non_authoritative_evidence(result)
    _write_new(args.output, evidence)
    print(
        json.dumps(
            {
                "classification": "NON_AUTHORITATIVE_EVIDENCE",
                "live_result_identity_sha256": (
                    result.canonical_identity_sha256
                ),
                "evidence_body_sha256": evidence["canonical_body_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
