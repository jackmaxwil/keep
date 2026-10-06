#!/usr/bin/env python3
"""Conditionally create the fixed SEALED H.1d trusted-source record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)


PROFILE = "keep-gpu"
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
TABLE = "keep-glm52-h1g-ledger-v1"
CAMPAIGN_BUCKET = "keep-glm52-models-246813579024-us-west-2"
PK = f"RUN#{RUN_ID}"
SK = "H1D_TRUSTED_SOURCE"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_NAMES = {
    "task6_manifest",
    "task6_templates",
    "task7_postcreate_manifest",
    "task7_inventory",
}
_SOURCE_ORDER = (
    "task6_manifest",
    "task6_templates",
    "task7_postcreate_manifest",
    "task7_inventory",
)


class AwsCliCommandRunner:
    """One-attempt AWS CLI boundary."""

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
        environment = {
            **os.environ,
            "AWS_PAGER": "",
            "AWS_MAX_ATTEMPTS": "1",
            "AWS_RETRY_MODE": "standard",
            "AWS_DEFAULT_REGION": REGION,
        }
        try:
            result = self._subprocess_run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("AWS CLI operation timed out") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ValueError(detail or "AWS CLI operation failed")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("AWS CLI operation returned malformed JSON") from exc
        if type(value) is not dict:
            raise ValueError("AWS CLI operation returned a non-object")
        return value

    def get_s3_object(
        self,
        *,
        bucket: str,
        key: str,
        version_id: str,
        timeout_seconds: int,
    ) -> tuple[dict[str, object], bytes]:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".glm52-h1d-source-"
        )
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
    ("cloudformation", "describe-termination-protection"): (
        "cloudformation",
        "describe_termination_protection",
    ),
    ("iam", "get-role"): ("iam", "get_role"),
    ("dynamodb", "query"): ("dynamodb", "query"),
    ("dynamodb", "put-item"): ("dynamodb", "put_item"),
    ("dynamodb", "get-item"): ("dynamodb", "get_item"),
}

_SDK_OPTIONS: Mapping[str, tuple[str, str]] = {
    "--stack-name": ("StackName", "string"),
    "--role-name": ("RoleName", "string"),
    "--next-token": ("NextToken", "string"),
    "--table-name": ("TableName", "string"),
    "--key-condition-expression": (
        "KeyConditionExpression",
        "string",
    ),
    "--expression-attribute-values": (
        "ExpressionAttributeValues",
        "json",
    ),
    "--consistent-read": ("ConsistentRead", "boolean"),
    "--exclusive-start-key": ("ExclusiveStartKey", "json"),
    "--item": ("Item", "json"),
    "--condition-expression": ("ConditionExpression", "string"),
    "--return-consumed-capacity": (
        "ReturnConsumedCapacity",
        "string",
    ),
    "--key": ("Key", "json"),
}


def _sdk_json_safe(value: object) -> object:
    if hasattr(value, "isoformat") and not isinstance(value, str):
        isoformat = getattr(value, "isoformat")
        if callable(isoformat):
            return str(isoformat())
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
    raise ValueError("AWS SDK response contains a noncanonical value")


class AwsSdkCommandRunner:
    """One-attempt native AWS SDK boundary for materialization."""

    def __init__(
        self,
        *,
        profile: str = PROFILE,
        region: str = REGION,
        session_factory: Optional[Callable[..., object]] = None,
        config_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        if profile != PROFILE or region != REGION:
            raise ValueError(
                "AWS SDK runner requires the exact profile and region"
            )
        if session_factory is None or config_factory is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 and botocore are required for materialization"
                ) from exc
            if session_factory is None:
                session_factory = boto3.Session
            if config_factory is None:
                config_factory = Config
        self._config = config_factory(
            connect_timeout=1,
            read_timeout=2,
            retries={
                "total_max_attempts": 1,
                "mode": "standard",
            },
        )
        self._session = session_factory(
            profile_name=profile,
            region_name=region,
        )
        self._clients: dict[str, object] = {}

    def _client(self, service: str) -> object:
        client = self._clients.get(service)
        if client is None:
            factory = getattr(self._session, "client", None)
            if not callable(factory):
                raise ValueError("AWS SDK session has no client factory")
            client = factory(service, config=self._config)
            self._clients[service] = client
        return client

    @staticmethod
    def _arguments(raw: Sequence[str]) -> dict[str, object]:
        arguments: dict[str, object] = {}
        index = 0
        while index < len(raw):
            flag = raw[index]
            if flag == "--no-paginate":
                index += 1
                continue
            option = _SDK_OPTIONS.get(flag)
            if option is None:
                raise ValueError(
                    f"AWS SDK option {flag!r} is not allowlisted"
                )
            name, kind = option
            if kind == "boolean":
                arguments[name] = True
                index += 1
                continue
            if index + 1 >= len(raw):
                raise ValueError(
                    f"AWS SDK option {flag!r} has no value"
                )
            raw_value = raw[index + 1]
            arguments[name] = (
                json.loads(raw_value)
                if kind == "json"
                else raw_value
            )
            index += 2
        return arguments

    @staticmethod
    def _authenticated_response(
        value: object,
        *,
        label: str,
    ) -> dict[str, object]:
        if type(value) is not dict:
            raise ValueError(f"{label} returned a non-object")
        normalized = _sdk_json_safe(value)
        if type(normalized) is not dict:
            raise ValueError(f"{label} returned a non-object")
        metadata = normalized.get("ResponseMetadata")
        if (
            type(metadata) is not dict
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
        ):
            raise ValueError(
                f"{label} lacks an authenticated service RequestId"
            )
        return normalized

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
            or not 1 <= timeout_seconds <= 5
        ):
            raise ValueError("AWS SDK operation is malformed")
        command = (operation[0], operation[1])
        target = _SDK_METHODS.get(command)
        if target is None:
            raise ValueError(
                "AWS SDK operation is outside the materializer allowlist"
            )
        client = self._client(target[0])
        method = getattr(client, target[1], None)
        if not callable(method):
            raise ValueError(
                "AWS SDK client lacks an allowlisted operation"
            )
        arguments = self._arguments(operation[2:])
        try:
            response = method(**arguments)
        except BaseException as exc:
            raise ValueError("AWS SDK materializer operation failed") from exc
        normalized = self._authenticated_response(
            response,
            label="AWS SDK materializer operation",
        )
        if command == ("cloudformation", "get-template"):
            template = normalized.get("TemplateBody")
            if type(template) is str:
                try:
                    normalized["TemplateBody"] = json.loads(template)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "CloudFormation template body is not canonical JSON"
                    ) from exc
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
            or type(timeout_seconds) is not int
            or not 1 <= timeout_seconds <= 5
        ):
            raise ValueError("AWS SDK S3 exact-object request is malformed")
        method = getattr(self._client("s3"), "get_object", None)
        if not callable(method):
            raise ValueError("AWS SDK S3 client lacks get_object")
        try:
            response = method(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
            )
        except BaseException as exc:
            raise ValueError("AWS SDK S3 exact-object read failed") from exc
        if type(response) is not dict:
            raise ValueError("AWS SDK S3 exact-object response is malformed")
        body = response.get("Body")
        read = getattr(body, "read", None)
        if not callable(read):
            raise ValueError("AWS SDK S3 exact-object body is absent")
        raw = read()
        if type(raw) is not bytes:
            raise ValueError("AWS SDK S3 exact-object body is malformed")
        metadata = {
            name: value
            for name, value in response.items()
            if name != "Body"
        }
        return (
            self._authenticated_response(
                metadata,
                label="AWS SDK S3 exact-object read",
            ),
            raw,
        )


_PRODUCTION_RUNNER = AwsSdkCommandRunner


def _validate_sources(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _SOURCE_NAMES:
        raise ValueError("authority source inventory is not exact")
    validated: dict[str, object] = {}
    for name, coordinate in value.items():
        if (
            type(coordinate) is not dict
            or set(coordinate) != {"key", "version_id", "file_sha256"}
            or type(coordinate["key"]) is not str
            or not coordinate["key"]
            or type(coordinate["version_id"]) is not str
            or not coordinate["version_id"]
            or type(coordinate["file_sha256"]) is not str
            or _SHA256.fullmatch(coordinate["file_sha256"]) is None
        ):
            raise ValueError(f"{name} source coordinate is malformed")
        validated[str(name)] = dict(coordinate)
    return validated


def _read_input(path: Path, expected_sha256: str) -> dict[str, object]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("trusted-source input SHA-256 is invalid")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("trusted-source input exact bytes differ")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("trusted-source input is malformed JSON") from exc
    if (
        type(value) is not dict
        or canonical_json_bytes(value) + b"\n" != raw
        or set(value)
        != {
            "schema_version",
            "record_type",
            "account_id",
            "region",
            "run_id",
            "authority_sources",
        }
        or value["schema_version"] != 1
        or value["record_type"] != "glm52_h1d_trusted_source_input_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
    ):
        raise ValueError("trusted-source input is foreign or noncanonical")
    return {
        **value,
        "authority_sources": _validate_sources(value["authority_sources"]),
    }


def _canonical_document(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is malformed JSON") from exc
    if (
        type(value) is not dict
        or canonical_json_bytes(value) + b"\n" != raw
    ):
        raise ValueError(f"{label} is not canonical JSON")
    if "canonical_body_sha256" in value:
        body = dict(value)
        identity = body.pop("canonical_body_sha256")
        if identity != canonical_sha256(body):
            raise ValueError(f"{label} self-hash drifted")
    return value


def _read_source_documents(
    runner: object,
    sources: Mapping[str, object],
) -> dict[str, object]:
    documents: dict[str, object] = {}
    read = getattr(runner, "get_s3_object", None)
    if not callable(read):
        raise ValueError("exact-version source reader is absent")
    for name in _SOURCE_ORDER:
        coordinate = sources[name]
        metadata, raw = read(
            bucket=CAMPAIGN_BUCKET,
            key=coordinate["key"],
            version_id=coordinate["version_id"],
            timeout_seconds=5,
        )
        if metadata.get("VersionId") != coordinate["version_id"]:
            raise ValueError(f"{name} returned the wrong VersionId")
        if hashlib.sha256(raw).hexdigest() != coordinate["file_sha256"]:
            raise ValueError(f"{name} exact bytes drifted")
        documents[name] = _canonical_document(raw, label=name)
    return documents


def _source_contract_sha256(documents: Mapping[str, object]) -> str:
    if set(documents) != _SOURCE_NAMES:
        raise ValueError("source document inventory is incomplete")
    task6 = documents["task6_manifest"]
    templates = documents["task6_templates"]
    postcreate = documents["task7_postcreate_manifest"]
    inventory = documents["task7_inventory"]
    if any(type(item) is not dict for item in documents.values()):
        raise ValueError("source document is not an object")
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
        or task6["bucket_name"] != CAMPAIGN_BUCKET
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


def _cloudformation_template_for_stack(
    documents: Mapping[str, object],
    stack: Mapping[str, object],
) -> Mapping[str, object]:
    templates = documents["task6_templates"]
    if type(templates) is not dict:
        raise ValueError("Task 6 template bundle is malformed")
    kind = stack.get("kind")
    if kind == "support":
        template = templates.get("task7_support_template")
    else:
        stages = {
            "retained": "post-retain",
            "fence": "final-fence",
        }
        stage = stages.get(kind)
        task6_templates = templates.get("task6_templates")
        template = (
            task6_templates.get(stage)
            if type(task6_templates) is dict and stage is not None
            else None
        )
    if (
        type(template) is not dict
        or type(template.get("Resources")) is not dict
    ):
        raise ValueError("Task 6 stack template semantics are absent")
    return template


def _canonical_stack_tags(
    value: object,
    *,
    aws_shape: bool,
) -> list[dict[str, str]]:
    if aws_shape:
        if (
            type(value) is not list
            or any(
                type(item) is not dict
                or set(item) != {"Key", "Value"}
                or type(item["Key"]) is not str
                or not item["Key"]
                or type(item["Value"]) is not str
                for item in value
            )
        ):
            raise ValueError("CloudFormation stack tags are malformed")
        tags = [
            {"key": item["Key"], "value": item["Value"]}
            for item in value
        ]
    else:
        if (
            type(value) is not dict
            or any(
                type(key) is not str
                or not key
                or type(tag_value) is not str
                for key, tag_value in value.items()
            )
        ):
            raise ValueError("Task 6 stack tags are malformed")
        tags = [
            {"key": key, "value": tag_value}
            for key, tag_value in value.items()
        ]
    tags.sort(key=lambda item: item["key"])
    if len({item["key"] for item in tags}) != len(tags):
        raise ValueError("CloudFormation stack tags are duplicated")
    return tags


def _read_cloudformation_expected_items(
    runner: object,
    documents: Mapping[str, object],
    *,
    task6_manifest_file_sha256: Optional[str] = None,
) -> list[dict[str, object]]:
    task6 = documents.get("task6_manifest")
    if type(task6) is not dict or type(task6.get("stacks")) is not list:
        raise ValueError("Task 6 stack inventory is absent")
    stacks = task6["stacks"]
    if (
        len(stacks) != 3
        or any(type(stack) is not dict for stack in stacks)
        or {stack.get("kind") for stack in stacks}
        != {"retained", "fence", "support"}
    ):
        raise ValueError("Task 6 exact three-stack inventory is malformed")
    if task6_manifest_file_sha256 is None:
        task6_manifest_file_sha256 = hashlib.sha256(
            canonical_json_bytes(task6) + b"\n"
        ).hexdigest()
    if _SHA256.fullmatch(str(task6_manifest_file_sha256)) is None:
        raise ValueError("Task 6 manifest file identity is malformed")

    expected_items: list[dict[str, object]] = []
    observed_role_arns: set[str] = set()
    for stack in sorted(stacks, key=lambda item: str(item.get("kind"))):
        stack_id = stack.get("stack_id")
        stack_name = stack.get("name")
        if (
            type(stack_id) is not str
            or not stack_id.startswith(
                f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
            )
            or type(stack_name) is not str
            or f":stack/{stack_name}/" not in stack_id
        ):
            raise ValueError("Task 6 stack identity is malformed")
        desired_template = _cloudformation_template_for_stack(
            documents, stack
        )
        source_tags = _canonical_stack_tags(
            stack.get("tags"),
            aws_shape=False,
        )
        describe = runner.run_json(
            (
                "cloudformation",
                "describe-stacks",
                "--stack-name",
                stack_id,
                "--no-paginate",
            ),
            timeout_seconds=5,
        )
        described_stacks = describe.get("Stacks")
        if (
            type(described_stacks) is not list
            or len(described_stacks) != 1
            or type(described_stacks[0]) is not dict
        ):
            raise ValueError("CloudFormation stack readback is not exact")
        described = described_stacks[0]
        role_arn = described.get("RoleARN")
        if (
            described.get("StackId") != stack_id
            or described.get("StackName") != stack_name
            or described.get("StackStatus") != "UPDATE_COMPLETE"
            or type(described.get("Parameters", [])) is not list
            or type(role_arn) is not str
            or re.fullmatch(
                rf"arn:aws:iam::{ACCOUNT_ID}:role/"
                r"[A-Za-z0-9+=,.@_/-]+",
                role_arn,
            )
            is None
        ):
            raise ValueError("CloudFormation stack role or identity is foreign")
        if (
            _canonical_stack_tags(
                described.get("Tags"),
                aws_shape=True,
            )
            != source_tags
        ):
            raise ValueError("CloudFormation stack tags drifted")
        observed_role_arns.add(role_arn)

        template_response = runner.run_json(
            (
                "cloudformation",
                "get-template",
                "--stack-name",
                stack_id,
                "--no-paginate",
            ),
            timeout_seconds=5,
        )
        observed_template = template_response.get("TemplateBody")
        if (
            type(observed_template) is not dict
            or canonical_sha256(observed_template)
            != canonical_sha256(desired_template)
        ):
            raise ValueError(
                "CloudFormation deployed template semantics drifted"
            )

        rows: list[object] = []
        token: Optional[str] = None
        seen_tokens: set[str] = set()
        while True:
            operation = [
                "cloudformation",
                "list-stack-resources",
                "--stack-name",
                stack_id,
                "--no-paginate",
            ]
            if token is not None:
                operation.extend(("--next-token", token))
            response = runner.run_json(
                tuple(operation),
                timeout_seconds=5,
            )
            page = response.get("StackResourceSummaries")
            if type(page) is not list:
                raise ValueError(
                    "CloudFormation stack resource page is malformed"
                )
            rows.extend(page)
            next_token = response.get("NextToken")
            if next_token is None:
                break
            if (
                type(next_token) is not str
                or not next_token
                or next_token in seen_tokens
            ):
                raise ValueError(
                    "CloudFormation stack resource pagination repeated"
                )
            seen_tokens.add(next_token)
            token = next_token
        normalized_rows = []
        for row in rows:
            if (
                type(row) is not dict
                or type(row.get("LogicalResourceId")) is not str
                or type(row.get("PhysicalResourceId")) is not str
                or not row["PhysicalResourceId"].strip()
                or type(row.get("ResourceType")) is not str
                or row.get("ResourceStatus")
                not in {
                    "CREATE_COMPLETE",
                    "IMPORT_COMPLETE",
                    "UPDATE_COMPLETE",
                }
            ):
                raise ValueError(
                    "CloudFormation stack resource status is not complete"
                )
            normalized_rows.append(
                {
                    "logical_id": row["LogicalResourceId"],
                    "resource_type": row["ResourceType"],
                    "physical_id": row["PhysicalResourceId"],
                    "resource_status": row["ResourceStatus"],
                }
            )
        normalized_rows.sort(key=lambda item: str(item["logical_id"]))
        expected_logical_types = {
            str(logical_id): resource.get("Type")
            for logical_id, resource in desired_template["Resources"].items()
            if type(resource) is dict
        }
        observed_logical_types = {
            str(row["logical_id"]): row["resource_type"]
            for row in normalized_rows
        }
        if (
            len(normalized_rows) != len(rows)
            or len(observed_logical_types) != len(normalized_rows)
            or observed_logical_types != expected_logical_types
        ):
            raise ValueError(
                "CloudFormation stack resource inventory is incomplete"
            )

        termination = runner.run_json(
            (
                "cloudformation",
                "describe-termination-protection",
                "--stack-name",
                stack_id,
                "--no-paginate",
            ),
            timeout_seconds=5,
        )
        if termination.get("EnableTerminationProtection") is not True:
            raise ValueError(
                "CloudFormation termination protection is not enabled"
            )
        expected_items.append(
            {
                "stack_id": stack_id,
                "status": described["StackStatus"],
                "template_sha256": canonical_sha256(desired_template),
                "parameters_sha256": canonical_sha256(
                    described.get("Parameters", [])
                ),
                "resources_sha256": canonical_sha256(normalized_rows),
                "stack_tags": source_tags,
                "service_role_arn": role_arn,
                "termination_protection": True,
                "manifest_identity_sha256": (
                    task6_manifest_file_sha256
                ),
            }
        )
    if len(observed_role_arns) != 1:
        raise ValueError(
            "CloudFormation deployment role is not exact across stacks"
        )
    return sorted(expected_items, key=lambda item: str(item["stack_id"]))


def _read_cloudformation_deployment_role(
    runner: object,
    documents: Mapping[str, object],
    expected_items: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    task6 = documents.get("task6_manifest")
    expected_role_id = (
        task6.get("retained_deployment_role_id")
        if type(task6) is dict
        else None
    )
    role_arns = {
        item.get("service_role_arn") for item in expected_items
    }
    if (
        type(expected_role_id) is not str
        or re.fullmatch(r"AROA[A-Z0-9]{16,128}", expected_role_id)
        is None
        or len(role_arns) != 1
    ):
        raise ValueError(
            "Task 6 deployment role identity is absent"
        )
    role_arn = next(iter(role_arns))
    if type(role_arn) is not str:
        raise ValueError("CloudFormation deployment role ARN is absent")
    role_name = role_arn.rsplit("/", 1)[-1]
    response = runner.run_json(
        (
            "iam",
            "get-role",
            "--role-name",
            role_name,
            "--no-paginate",
        ),
        timeout_seconds=5,
    )
    role = response.get("Role")
    metadata = response.get("ResponseMetadata")
    request_id = (
        metadata.get("RequestId")
        if type(metadata) is dict
        else None
    )
    if (
        type(role) is not dict
        or role.get("Arn") != role_arn
        or role.get("RoleId") != expected_role_id
        or role.get("RoleName") != role_name
        or type(role.get("Path")) is not str
        or not role["Path"].startswith("/")
        or not role["Path"].endswith("/")
        or role_arn
        != (
            f"arn:aws:iam::{ACCOUNT_ID}:role"
            f"{role['Path']}{role_name}"
        )
        or type(request_id) is not str
        or not request_id
    ):
        raise ValueError(
            "CloudFormation deployment role identity is foreign"
        )
    return {
        "role_arn": role_arn,
        "role_id": expected_role_id,
        "request_id": request_id,
    }


def _read_cloudformation_contract(
    runner: object,
    documents: Mapping[str, object],
    *,
    task6_manifest_file_sha256: Optional[str] = None,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    expected_items = _read_cloudformation_expected_items(
        runner,
        documents,
        task6_manifest_file_sha256=task6_manifest_file_sha256,
    )
    deployment_role = _read_cloudformation_deployment_role(
        runner,
        documents,
        expected_items,
    )
    return expected_items, deployment_role


def _ddb_string(item: Mapping[str, object], name: str) -> str:
    attribute = item.get(name)
    if (
        type(attribute) is not dict
        or set(attribute) != {"S"}
        or type(attribute["S"]) is not str
        or not attribute["S"]
    ):
        raise ValueError(f"DynamoDB {name} is malformed")
    return attribute["S"]


def _ddb_expected_items(
    items: object,
    *,
    activation_id: str,
) -> list[dict[str, object]]:
    if type(items) is not list or any(type(item) is not dict for item in items):
        raise ValueError("DynamoDB source inventory is malformed")
    expected: list[dict[str, object]] = []
    records: dict[str, str] = {}
    activation_index: Optional[Mapping[str, object]] = None
    for item in items:
        pk = _ddb_string(item, "PK")
        sk = _ddb_string(item, "SK")
        if pk != PK:
            raise ValueError("DynamoDB source row is foreign")
        if sk == SK:
            continue
        record_type = _ddb_string(item, "record_type")
        if sk in records:
            raise ValueError("DynamoDB source key is duplicated")
        records[sk] = record_type
        if sk == "ACTIVATION_INDEX":
            activation_index = item
        expected.append(
            {
                "key": f"{PK}|{sk}",
                "record_type": record_type,
                "body_sha256": canonical_sha256(item),
                "consistent_read": True,
            }
        )
    if (
        activation_index is None
        or _ddb_string(
            activation_index,
            "current_activation_id",
        )
        != activation_id
        or records.get(f"ACTIVATION#{activation_id}#CONTROL")
        != "glm52_production_control"
        or not any(
            re.fullmatch(
                (
                    f"ACTIVATION#{re.escape(activation_id)}#"
                    r"ACTION#[0-9]{8}#SKY#[0-9]{8}"
                ),
                key,
            )
            and record_type == "glm52_production_action"
            for key, record_type in records.items()
        )
    ):
        raise ValueError(
            "DynamoDB current activation/control/SKY action is absent"
        )
    return sorted(expected, key=lambda item: str(item["key"]))


def _read_ddb_expected_items(
    runner: object,
    *,
    activation_id: str,
) -> list[dict[str, object]]:
    values = canonical_json_bytes({":pk": {"S": PK}}).decode("utf-8")
    items: list[object] = []
    exclusive_start_key: Optional[object] = None
    seen_tokens: set[str] = set()
    while True:
        operation = [
            "dynamodb",
            "query",
            "--table-name",
            TABLE,
            "--key-condition-expression",
            "PK = :pk",
            "--expression-attribute-values",
            values,
            "--consistent-read",
            "--no-paginate",
        ]
        if exclusive_start_key is not None:
            operation.extend(
                (
                    "--exclusive-start-key",
                    canonical_json_bytes(exclusive_start_key).decode("utf-8"),
                )
            )
        response = runner.run_json(tuple(operation), timeout_seconds=5)
        page = response.get("Items")
        if type(page) is not list:
            raise ValueError("DynamoDB source page is malformed")
        items.extend(page)
        exclusive_start_key = response.get("LastEvaluatedKey")
        if exclusive_start_key is None:
            break
        if type(exclusive_start_key) is not dict or not exclusive_start_key:
            raise ValueError("DynamoDB source token is malformed")
        token_identity = canonical_sha256(exclusive_start_key)
        if token_identity in seen_tokens:
            raise ValueError("DynamoDB source token repeated")
        seen_tokens.add(token_identity)
    return _ddb_expected_items(items, activation_id=activation_id)


def _record_item(
    value: Mapping[str, object],
    *,
    source_contract_sha256: str,
    dynamodb_expected_items: Sequence[Mapping[str, object]],
    cloudformation_expected_items: Sequence[Mapping[str, object]],
    cloudformation_deployment_role: Mapping[str, object],
) -> dict[str, object]:
    body = {
        "schema_version": 1,
        "record_type": "glm52_h1d_trusted_campaign_sources_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "PK": PK,
        "SK": SK,
        "state": "SEALED",
        "authority_sources": value["authority_sources"],
        "source_contract_sha256": source_contract_sha256,
        "cloudformation_deployment_role": dict(
            cloudformation_deployment_role
        ),
        "dynamodb_expected_items": list(dynamodb_expected_items),
        "cloudformation_expected_items": list(
            cloudformation_expected_items
        ),
    }
    record = {
        **body,
        "canonical_body_sha256": canonical_sha256(body),
    }
    return {
        "PK": {"S": PK},
        "SK": {"S": SK},
        "record_type": {
            "S": "glm52_h1d_trusted_campaign_sources_v1"
        },
        "state": {"S": "SEALED"},
        "canonical_body_json": {
            "S": canonical_json_bytes(record).decode("utf-8")
        },
        "canonical_body_sha256": {
            "S": record["canonical_body_sha256"]
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    return parser


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    runner_factory: Callable[..., object] = _PRODUCTION_RUNNER,
) -> int:
    args = _parser().parse_args(argv)
    if args.profile != PROFILE:
        raise ValueError("--profile must be exactly keep-gpu")
    if args.region != REGION:
        raise ValueError("--region must be exactly us-west-2")
    runner = runner_factory(profile=args.profile, region=args.region)
    identity = runner.run_json(
        ("sts", "get-caller-identity"),
        timeout_seconds=5,
    )
    if (
        identity.get("Account") != ACCOUNT_ID
        or type(identity.get("Arn")) is not str
        or f"::{ACCOUNT_ID}:" not in identity["Arn"]
        or type(identity.get("UserId")) is not str
        or not identity["UserId"]
    ):
        raise ValueError("AWS caller is not the exact approved account")
    value = _read_input(args.input, args.input_sha256)
    documents = _read_source_documents(
        runner,
        value["authority_sources"],
    )
    source_contract_sha256 = _source_contract_sha256(documents)
    (
        cloudformation_expected_items,
        cloudformation_deployment_role,
    ) = _read_cloudformation_contract(
        runner,
        documents,
        task6_manifest_file_sha256=value["authority_sources"][
            "task6_manifest"
        ]["file_sha256"],
    )
    postcreate = documents["task7_postcreate_manifest"]
    dynamodb_expected_items = _read_ddb_expected_items(
        runner,
        activation_id=str(postcreate["activation_id"]),
    )
    item = _record_item(
        value,
        source_contract_sha256=source_contract_sha256,
        dynamodb_expected_items=dynamodb_expected_items,
        cloudformation_expected_items=cloudformation_expected_items,
        cloudformation_deployment_role=(
            cloudformation_deployment_role
        ),
    )
    item_json = canonical_json_bytes(item).decode("utf-8")
    runner.run_json(
        (
            "dynamodb",
            "put-item",
            "--table-name",
            TABLE,
            "--item",
            item_json,
            "--condition-expression",
            "attribute_not_exists(PK) AND attribute_not_exists(SK)",
            "--return-consumed-capacity",
            "NONE",
        ),
        timeout_seconds=5,
    )
    key = {"PK": {"S": PK}, "SK": {"S": SK}}
    readback = runner.run_json(
        (
            "dynamodb",
            "get-item",
            "--table-name",
            TABLE,
            "--key",
            canonical_json_bytes(key).decode("utf-8"),
            "--consistent-read",
            "--return-consumed-capacity",
            "NONE",
        ),
        timeout_seconds=5,
    )
    if set(readback) - {"Item", "ConsumedCapacity", "ResponseMetadata"}:
        raise ValueError("trusted-source readback schema is not closed")
    if readback.get("Item") != item:
        raise ValueError("trusted-source readback differs from created record")
    print(
        json.dumps(
            {
                "account_id": ACCOUNT_ID,
                "record_identity_sha256": item[
                    "canonical_body_sha256"
                ]["S"],
                "region": REGION,
                "run_id": RUN_ID,
                "status": "SEALED_TRUSTED_SOURCE_CREATED",
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
