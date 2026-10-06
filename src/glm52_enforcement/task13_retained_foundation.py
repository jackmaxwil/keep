"""Guarded Task 13 retained-foundation CloudFormation route."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import yaml
from botocore.exceptions import ConnectionError as BotoCoreConnectionError
from botocore.exceptions import HTTPClientError as BotoCoreHTTPClientError

from .task13_reviewed_artifacts import (
    RETAINED_MODELS_BUCKET,
    Task13ReviewedArtifactError,
    Task13ReviewedArtifactServices,
)
from .task13_reviewed_artifacts import (
    _guard_services as _guard_reviewed_artifact_services,
)
from .task13_reviewed_artifacts import (
    _publish_once as _publish_reviewed_artifact_once,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
STACK_NAME = "keep-glm52-gpu"
CHANGE_SET_NAME = "glm52-task13-retained-foundation-v1"
LEDGER_TABLE_NAME = "keep-glm52-h1g-ledger-v1"
DEPLOYMENT_ROLE_NAME = "keep-glm52-h1g-cloudformation-deployment"
FENCE_ROLE_NAME = "keep-glm52-h1g-fence-service"
FOUNDATION_TEMPLATE_ARTIFACT_KIND = "RETAINED_FOUNDATION_TEMPLATE"
FOUNDATION_TEMPLATE_KEY = "task13/templates/retained-foundation.yaml"

_FOUNDATION_RESOURCE_TYPES = {
    "CampaignKmsKey": "AWS::KMS::Key",
    "H1gLedger": "AWS::DynamoDB::Table",
    "H1gCloudFormationDeploymentRole": "AWS::IAM::Role",
    "H1gFenceServiceRole": "AWS::IAM::Role",
}
_REQUIRED_ORIGINAL_RESOURCE_TYPES = {
    "Vpc": "AWS::EC2::VPC",
    "PublicSubnet": "AWS::EC2::Subnet",
    "S3GatewayEndpoint": "AWS::EC2::VPCEndpoint",
    "ModelBucket": "AWS::S3::Bucket",
    "GpuLaunchTemplate": "AWS::EC2::LaunchTemplate",
}
_NEW_OUTPUTS = {
    "CampaignKmsKeyArn",
    "H1gLedgerArn",
}
_REQUIRED_EXPORTS = {
    "VpcId": "KeepGlm52VpcId",
    "SubnetId": "KeepGlm52PrimaryPublicSubnetId",
    "CampaignKmsKeyArn": "KeepGlm52CampaignKmsKeyArn",
    "H1gLedgerArn": "KeepGlm52H1gLedgerArn",
}
_CHANGE_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "profile",
        "stack_id",
        "stack_name",
        "termination_protection",
        "before_template_sha256",
        "after_template_sha256",
        "template_coordinate",
        "template_url",
        "change_set_id",
        "change_set_name",
        "client_token",
        "role_arn",
        "parameters",
        "capabilities",
        "changes",
    }
)
_COORDINATE_FIELDS = frozenset(
    {
        "artifact_kind",
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
    }
)
_RECOVERY_READBACK_RECORD = "glm52_retained_foundation_recovery_readback_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FoundationError(ValueError):
    """The retained-foundation request or live evidence is not exact."""


@dataclass(frozen=True)
class FoundationServices:
    """The seven AWS clients used by the retained-foundation transaction."""

    sts: object
    cloudformation: object
    cloudtrail: object
    iam: object
    dynamodb: object
    kms: object
    s3: object
    total_max_attempts: int


def apply_retained_foundation(
    services: FoundationServices,
    *,
    output_directory: Path,
    sleep: Callable[[float], None],
    max_polls: int,
) -> dict[str, object]:
    """Publish, inspect, execute, wait, and exact-read one foundation update."""

    paths = _guard_transaction_inputs(
        services=services,
        output_directory=output_directory,
        sleep=sleep,
        max_polls=max_polls,
    )
    identity = _call(services.sts, "get_caller_identity")
    caller_arn = identity.get("Arn")
    if (
        identity.get("Account") != ACCOUNT_ID
        or type(caller_arn) is not str
        or f"::{ACCOUNT_ID}:" not in caller_arn
        or type(identity.get("UserId")) is not str
        or not identity["UserId"]
    ):
        raise FoundationError("authenticated AWS caller identity is foreign")

    initial_stack = _describe_stack(
        services.cloudformation,
        STACK_NAME,
        allowed_statuses={"CREATE_COMPLETE", "UPDATE_COMPLETE"},
    )
    stack_id = initial_stack["StackId"]
    termination_protection = _termination_protection(initial_stack)
    role_arn = _authenticated_existing_stack_role(
        services.iam, initial_stack.get("RoleARN")
    )
    _assert_physical_names_absent(services)
    _assert_required_exports_absent(services.cloudformation)
    _assert_change_set_absent(services.cloudformation, stack_id)

    raw_template_body, before_template = _get_template_source(
        services.cloudformation,
        stack_name=stack_id,
    )
    after_template = compose_retained_foundation_template(before_template)
    before_raw = canonical_json_bytes(before_template) + b"\n"
    after_raw = canonical_json_bytes(after_template) + b"\n"
    execution_raw = _execution_template_bytes(
        raw_template_body=raw_template_body,
        original_template=before_template,
        retained_template=after_template,
    )
    _write_once(paths["before"], before_raw)
    _write_once(paths["after"], after_raw)

    try:
        template_coordinate = _publish_foundation_template(
            raw=execution_raw,
            expected_template=after_template,
            expected_file_sha256=hashlib.sha256(execution_raw).hexdigest(),
            expected_body_sha256=hashlib.sha256(execution_raw[:-1]).hexdigest(),
            bucket=RETAINED_MODELS_BUCKET,
            services=Task13ReviewedArtifactServices(
                sts=services.sts,
                s3=services.s3,
                total_max_attempts=services.total_max_attempts,
            ),
        )
    except Task13ReviewedArtifactError as exc:
        raise FoundationError(
            "retained template immutable publication failed: " + str(exc)
        ) from exc
    template_url = _versioned_template_url(template_coordinate)
    live_parameters = _parameter_readback(initial_stack.get("Parameters", []))
    parameters = _previous_parameters(live_parameters)
    client_token = hashlib.sha256(
        canonical_json_bytes(
            {
                "stack_id": stack_id,
                "change_set_name": CHANGE_SET_NAME,
                "template_coordinate": template_coordinate,
                "parameters": parameters,
                "role_arn": role_arn,
            }
        )
    ).hexdigest()
    create_request: dict[str, object] = {
        "StackName": stack_id,
        "ChangeSetName": CHANGE_SET_NAME,
        "ChangeSetType": "UPDATE",
        "Description": (
            "Task 13 retained Phase-1 KMS, ledger, and separated service roles"
        ),
        "TemplateURL": template_url,
        "Parameters": parameters,
        "Capabilities": ["CAPABILITY_NAMED_IAM"],
        "IncludeNestedStacks": False,
        "ClientToken": client_token,
    }
    if role_arn is not None:
        create_request["RoleARN"] = role_arn

    try:
        created = _call(
            services.cloudformation,
            "create_change_set",
            **create_request,
        )
        if created.get("Id") is None or created.get("StackId") != stack_id:
            raise FoundationError("CreateChangeSet response identity is incomplete")
    except Exception as exc:
        if not _ambiguous_exception(exc):
            if isinstance(exc, FoundationError):
                raise
            raise FoundationError("CreateChangeSet failed before adoption") from exc

    change_set = _wait_for_change_set(
        services.cloudformation,
        stack_id=stack_id,
        sleep=sleep,
        max_polls=max_polls,
    )
    validated_change_set = validate_foundation_change_set(
        change_set,
        stack_id=stack_id,
        change_set_name=CHANGE_SET_NAME,
        expected_parameters=live_parameters,
    )
    effective_parameters = _parameter_readback(validated_change_set.get("Parameters"))
    change_set_id = validated_change_set["ChangeSetId"]
    change_set_raw, change_set_template = _get_template_source(
        services.cloudformation,
        change_set_name=change_set_id,
    )
    if (
        canonical_json_bytes(change_set_template) != after_raw[:-1]
        or _template_source_body_sha256(change_set_raw)
        != template_coordinate["body_sha256"]
    ):
        raise FoundationError(
            "change-set Original template is not the sealed retained template"
        )

    current_stack = _describe_stack(
        services.cloudformation,
        stack_id,
        allowed_statuses={"CREATE_COMPLETE", "UPDATE_COMPLETE"},
    )
    if _stack_guard(current_stack) != _stack_guard(initial_stack):
        raise FoundationError(
            "retained stack identity changed while preparing change set"
        )
    if _termination_protection(current_stack) is not termination_protection:
        raise FoundationError(
            "retained stack termination protection changed before execute"
        )
    current_raw, current_template = _get_template_source(
        services.cloudformation,
        stack_name=stack_id,
    )
    if canonical_json_bytes(current_template) != before_raw[
        :-1
    ] or _template_source_body_sha256(current_raw) != _template_source_body_sha256(
        raw_template_body
    ):
        raise FoundationError("retained stack Original template changed before execute")

    change_evidence = {
        "schema_version": 1,
        "record_type": "glm52_task13_retained_foundation_change_set_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "stack_id": stack_id,
        "stack_name": STACK_NAME,
        "termination_protection": termination_protection,
        "before_template_sha256": _template_source_body_sha256(raw_template_body),
        "after_template_sha256": template_coordinate["body_sha256"],
        "template_coordinate": template_coordinate,
        "template_url": template_url,
        "change_set_id": change_set_id,
        "change_set_name": CHANGE_SET_NAME,
        "client_token": client_token,
        "role_arn": role_arn,
        "parameters": parameters,
        "capabilities": ["CAPABILITY_NAMED_IAM"],
        "changes": _normalized_changes(change_set),
    }
    _write_once(
        paths["change_set"],
        canonical_json_bytes(change_evidence) + b"\n",
    )

    execute_token = hashlib.sha256(
        canonical_json_bytes(
            {
                "change_set_id": change_set_id,
                "client_token": client_token,
                "after_template_sha256": change_evidence["after_template_sha256"],
            }
        )
    ).hexdigest()
    execute_was_ambiguous = False
    try:
        _call(
            services.cloudformation,
            "execute_change_set",
            ChangeSetName=change_set_id,
            StackName=stack_id,
            ClientRequestToken=execute_token,
        )
    except Exception as exc:
        if not _ambiguous_exception(exc):
            raise FoundationError("ExecuteChangeSet failed") from exc
        execute_was_ambiguous = True
        _reconcile_ambiguous_execute(
            services.cloudformation,
            stack_id=stack_id,
            change_set_id=change_set_id,
        )

    final_stack = _wait_for_stack_update(
        services.cloudformation,
        stack_id=stack_id,
        sleep=sleep,
        max_polls=max_polls,
    )
    if _termination_protection(final_stack) is not termination_protection:
        raise FoundationError(
            "retained stack termination protection changed after update"
        )
    final_raw, final_template = _get_template_source(
        services.cloudformation,
        stack_name=stack_id,
    )
    if (
        canonical_json_bytes(final_template) != after_raw[:-1]
        or _template_source_body_sha256(final_raw) != template_coordinate["body_sha256"]
    ):
        raise FoundationError(
            "live retained Original template does not match sealed after template"
        )
    resources = _list_stack_resources(services.cloudformation, stack_id)
    readback = _validate_live_readback(
        services=services,
        initial_stack=initial_stack,
        expected_parameters=effective_parameters,
        final_stack=final_stack,
        final_template=final_template,
        final_template_sha256=_template_source_body_sha256(final_raw),
        resources=resources,
        expected_template=after_template,
        termination_protection=termination_protection,
        template_coordinate=template_coordinate,
        execute_was_ambiguous=execute_was_ambiguous,
    )
    _write_once(
        paths["readback"],
        canonical_json_bytes(readback) + b"\n",
    )
    return {
        "status": final_stack["StackStatus"],
        "stack_id": stack_id,
        "change_set_id": change_set_id,
        "template_version_id": template_coordinate["version_id"],
        "readback_sha256": hashlib.sha256(canonical_json_bytes(readback)).hexdigest(),
    }


def recover_retained_foundation(
    services: FoundationServices,
    *,
    change_set_evidence: Path,
    readback_output: Path,
) -> dict[str, object]:
    """Recover final evidence without replaying the one-shot stack update."""

    source_path, output_path = _guard_recovery_inputs(
        services=services,
        change_set_evidence=change_set_evidence,
        readback_output=readback_output,
    )
    identity = _call(services.sts, "get_caller_identity")
    caller_arn = identity.get("Arn")
    if (
        identity.get("Account") != ACCOUNT_ID
        or type(caller_arn) is not str
        or f"::{ACCOUNT_ID}:" not in caller_arn
        or type(identity.get("UserId")) is not str
        or not identity["UserId"]
    ):
        raise FoundationError("authenticated AWS caller identity is foreign")

    source_raw = source_path.read_bytes()
    change_value = _validated_recovery_change_evidence(source_raw)
    coordinate = change_value["template_coordinate"]
    execute_event = _verify_execute_change_set_event(
        services.cloudtrail,
        change_set_id=str(change_value["change_set_id"]),
        stack_id=str(change_value["stack_id"]),
    )
    assert isinstance(coordinate, dict)
    template_raw = _read_foundation_template_version(
        services.s3,
        coordinate=coordinate,
    )
    expected_template = _template_body(template_raw.decode("utf-8"))

    initial_stack = _describe_stack(
        services.cloudformation,
        str(change_value["stack_id"]),
        allowed_statuses={"UPDATE_COMPLETE"},
    )
    if (
        initial_stack["StackId"] != change_value["stack_id"]
        or initial_stack.get("RoleARN") != change_value["role_arn"]
    ):
        raise FoundationError("recovery change evidence stack identity drifted")
    if (
        _previous_parameters(_parameter_readback(initial_stack.get("Parameters", [])))
        != change_value["parameters"]
    ):
        raise FoundationError(
            "recovery change evidence parameters drifted from live stack"
        )
    _authenticated_existing_stack_role(
        services.iam,
        initial_stack.get("RoleARN"),
    )
    live_raw, live_template = _get_template_source(
        services.cloudformation,
        stack_name=str(change_value["stack_id"]),
    )
    if (
        canonical_json_bytes(live_template) != canonical_json_bytes(expected_template)
        or _template_source_body_sha256(live_raw) != coordinate["body_sha256"]
    ):
        raise FoundationError("recovery live retained template drifted")

    protection_was_enabled = _termination_protection(initial_stack)
    protection_update_was_ambiguous = False
    if not protection_was_enabled:
        try:
            _call(
                services.cloudformation,
                "update_termination_protection",
                StackName=str(change_value["stack_id"]),
                EnableTerminationProtection=True,
            )
        except Exception as exc:
            if not _ambiguous_exception(exc):
                raise FoundationError(
                    "retained termination-protection seal failed"
                ) from exc
            protection_update_was_ambiguous = True

    final_stack = _describe_stack(
        services.cloudformation,
        str(change_value["stack_id"]),
        allowed_statuses={"UPDATE_COMPLETE"},
    )
    if _termination_protection(final_stack) is not True:
        raise FoundationError(
            "retained termination protection is not sealed after recovery"
        )
    final_raw, final_template = _get_template_source(
        services.cloudformation,
        stack_name=str(change_value["stack_id"]),
    )
    if (
        canonical_json_bytes(final_template) != canonical_json_bytes(expected_template)
        or _template_source_body_sha256(final_raw) != coordinate["body_sha256"]
    ):
        raise FoundationError("recovery final retained template drifted")

    resources = _list_stack_resources(
        services.cloudformation,
        str(change_value["stack_id"]),
    )
    readback = _validate_live_readback(
        services=services,
        initial_stack=final_stack,
        expected_parameters=_parameter_readback(final_stack.get("Parameters", [])),
        final_stack=final_stack,
        final_template=final_template,
        final_template_sha256=_template_source_body_sha256(final_raw),
        resources=resources,
        expected_template=expected_template,
        termination_protection=True,
        template_coordinate=coordinate,
        execute_was_ambiguous=False,
    )
    readback["record_type"] = _RECOVERY_READBACK_RECORD
    readback["execute_response_was_ambiguous"] = None
    readback["source_change_set_evidence_sha256"] = hashlib.sha256(
        source_raw[:-1]
    ).hexdigest()
    readback["historical_termination_protection"] = change_value[
        "termination_protection"
    ]
    readback[
        "termination_protection_sealed_during_recovery"
    ] = not protection_was_enabled
    readback["termination_protection_update_was_ambiguous"] = (
        protection_update_was_ambiguous
    )
    readback["execute_cloudtrail_event_id"] = execute_event["event_id"]
    readback["execute_cloudtrail_event_time"] = execute_event["event_time"]
    readback_raw = canonical_json_bytes(readback)
    _write_once(output_path, readback_raw + b"\n")
    return {
        "status": final_stack["StackStatus"],
        "stack_id": final_stack["StackId"],
        "readback_sha256": hashlib.sha256(readback_raw).hexdigest(),
    }


def _guard_recovery_inputs(
    *,
    services: FoundationServices,
    change_set_evidence: Path,
    readback_output: Path,
) -> tuple[Path, Path]:
    if type(services) is not FoundationServices or services.total_max_attempts != 1:
        raise FoundationError("retained-foundation recovery services are not exact")
    source = Path(change_set_evidence)
    output = Path(readback_output)
    if (
        not source.is_absolute()
        or not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
    ):
        raise FoundationError("change evidence path is not one exact source file")
    if (
        not output.is_absolute()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
        or output.parent.resolve(strict=True) != output.parent
        or output.exists()
        or output.is_symlink()
    ):
        raise FoundationError("recovery readback output is not one new exact file")
    return source, output


def _validated_recovery_change_evidence(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw.endswith(b"\n"):
        raise FoundationError("recovery change evidence is not canonical")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FoundationError("recovery change evidence is not JSON") from exc
    if (
        type(value) is not dict
        or set(value) != _CHANGE_EVIDENCE_FIELDS
        or raw != canonical_json_bytes(value) + b"\n"
        or value.get("schema_version") != 1
        or value.get("record_type") != "glm52_task13_retained_foundation_change_set_v1"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("profile") != PROFILE
        or value.get("stack_name") != STACK_NAME
        or type(value.get("stack_id")) is not str
        or re.fullmatch(
            rf"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
            rf"{STACK_NAME}/[0-9a-f-]+",
            value["stack_id"],
        )
        is None
        or type(value.get("termination_protection")) is not bool
        or type(value.get("before_template_sha256")) is not str
        or _SHA256.fullmatch(value["before_template_sha256"]) is None
        or type(value.get("after_template_sha256")) is not str
        or _SHA256.fullmatch(value["after_template_sha256"]) is None
        or value.get("change_set_name") != CHANGE_SET_NAME
        or type(value.get("change_set_id")) is not str
        or re.fullmatch(
            rf"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:changeSet/"
            rf"{CHANGE_SET_NAME}/[0-9a-f-]+",
            value["change_set_id"],
        )
        is None
        or type(value.get("client_token")) is not str
        or _SHA256.fullmatch(value["client_token"]) is None
        or value.get("capabilities") != ["CAPABILITY_NAMED_IAM"]
    ):
        raise FoundationError("recovery change evidence identity is not exact")

    coordinate = value.get("template_coordinate")
    if (
        type(coordinate) is not dict
        or set(coordinate) != _COORDINATE_FIELDS
        or coordinate.get("artifact_kind") != FOUNDATION_TEMPLATE_ARTIFACT_KIND
        or coordinate.get("bucket") != RETAINED_MODELS_BUCKET
        or coordinate.get("key") != FOUNDATION_TEMPLATE_KEY
        or type(coordinate.get("version_id")) is not str
        or not coordinate["version_id"]
        or type(coordinate.get("file_sha256")) is not str
        or _SHA256.fullmatch(coordinate["file_sha256"]) is None
        or type(coordinate.get("body_sha256")) is not str
        or _SHA256.fullmatch(coordinate["body_sha256"]) is None
        or coordinate["body_sha256"] != value["after_template_sha256"]
        or value.get("template_url") != _versioned_template_url(coordinate)
    ):
        raise FoundationError(
            "recovery change evidence template coordinate is not exact"
        )
    parameters = value.get("parameters")
    if (
        type(parameters) is not list
        or parameters
        != sorted(
            parameters,
            key=lambda row: str(row.get("ParameterKey")) if type(row) is dict else "",
        )
        or any(
            type(row) is not dict
            or set(row) != {"ParameterKey", "UsePreviousValue"}
            or type(row.get("ParameterKey")) is not str
            or not row["ParameterKey"]
            or row.get("UsePreviousValue") is not True
            for row in parameters
        )
        or len({str(row["ParameterKey"]) for row in parameters}) != len(parameters)
    ):
        raise FoundationError("recovery change evidence parameters are not exact")
    expected_token = hashlib.sha256(
        canonical_json_bytes(
            {
                "stack_id": value["stack_id"],
                "change_set_name": CHANGE_SET_NAME,
                "template_coordinate": coordinate,
                "parameters": parameters,
                "role_arn": value.get("role_arn"),
            }
        )
    ).hexdigest()
    if value["client_token"] != expected_token:
        raise FoundationError("recovery change evidence client token drifted")
    expected_changes = [
        {
            "action": "Add",
            "logical_id": logical_id,
            "replacement": "None",
            "resource_type": resource_type,
        }
        for logical_id, resource_type in sorted(_FOUNDATION_RESOURCE_TYPES.items())
    ]
    launch_template_change = {
        "action": "Modify",
        "logical_id": "GpuLaunchTemplate",
        "replacement": "False",
        "resource_type": "AWS::EC2::LaunchTemplate",
    }
    observed_changes = value.get("changes")
    if type(observed_changes) is not list:
        raise FoundationError("recovery change evidence delta is not exact")
    with_launch_template = sorted(
        expected_changes + [launch_template_change],
        key=lambda row: (
            str(row["logical_id"]),
            str(row["action"]),
            str(row["resource_type"]),
            str(row["replacement"]),
        ),
    )
    observed_change_bytes = tuple(canonical_json_bytes(row) for row in observed_changes)
    if observed_change_bytes not in {
        tuple(canonical_json_bytes(row) for row in expected_changes),
        tuple(canonical_json_bytes(row) for row in with_launch_template),
    }:
        raise FoundationError("recovery change evidence delta is not exact")
    return deepcopy(value)


def _read_foundation_template_version(
    s3: object,
    *,
    coordinate: Mapping[str, object],
) -> bytes:
    response = _call(
        s3,
        "get_object",
        Bucket=RETAINED_MODELS_BUCKET,
        Key=FOUNDATION_TEMPLATE_KEY,
        VersionId=coordinate["version_id"],
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    body = response.get("Body")
    read = getattr(body, "read", None)
    if not callable(read):
        raise FoundationError("recovery template response body is absent")
    raw = read()
    if type(raw) is not bytes:
        raise FoundationError("recovery template response body is malformed")
    expected_checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    if (
        response.get("VersionId") != coordinate["version_id"]
        or response.get("ContentLength") != len(raw)
        or response.get("ChecksumSHA256") != expected_checksum
        or hashlib.sha256(raw).hexdigest() != coordinate["file_sha256"]
        or not raw.endswith(b"\n")
        or hashlib.sha256(raw[:-1]).hexdigest() != coordinate["body_sha256"]
    ):
        raise FoundationError("recovery template versioned readback drifted")
    return raw


def _template_source_body_sha256(value: object) -> str:
    if type(value) is str:
        raw = value.encode("utf-8")
        body = raw[:-1] if raw.endswith(b"\n") else raw
        return hashlib.sha256(body).hexdigest()
    return hashlib.sha256(canonical_json_bytes(_template_body(value))).hexdigest()


def _verify_execute_change_set_event(
    cloudtrail: object,
    *,
    change_set_id: str,
    stack_id: str,
) -> dict[str, str]:
    matches: dict[str, dict[str, str]] = {}
    next_token: str | None = None
    for _page in range(20):
        request: dict[str, object] = {
            "LookupAttributes": [
                {
                    "AttributeKey": "EventName",
                    "AttributeValue": "ExecuteChangeSet",
                }
            ],
            "MaxResults": 50,
        }
        if next_token is not None:
            request["NextToken"] = next_token
        response = _call(cloudtrail, "lookup_events", **request)
        events = response.get("Events")
        if type(events) is not list:
            raise FoundationError("CloudTrail execute-event page is malformed")
        for wrapper in events:
            if type(wrapper) is not dict:
                raise FoundationError("CloudTrail execute-event entry is malformed")
            encoded = wrapper.get("CloudTrailEvent")
            if type(encoded) is not str:
                raise FoundationError("CloudTrail execute-event payload is absent")
            try:
                event = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise FoundationError(
                    "CloudTrail execute-event payload is malformed"
                ) from exc
            if type(event) is not dict:
                raise FoundationError("CloudTrail execute-event payload is malformed")
            parameters = event.get("requestParameters")
            if (
                type(parameters) is not dict
                or parameters.get("changeSetName") != change_set_id
                or parameters.get("stackName") != stack_id
            ):
                continue
            if (
                event.get("errorCode") is not None
                or event.get("errorMessage") is not None
            ):
                continue
            user_identity = event.get("userIdentity")
            event_id = event.get("eventID")
            event_time = event.get("eventTime")
            if (
                event.get("eventName") != "ExecuteChangeSet"
                or event.get("eventSource") != "cloudformation.amazonaws.com"
                or event.get("recipientAccountId") != ACCOUNT_ID
                or event.get("readOnly") is not False
                or event.get("managementEvent") is not True
                or event.get("responseElements") is not None
                or type(user_identity) is not dict
                or user_identity.get("accountId") != ACCOUNT_ID
                or type(event_id) is not str
                or re.fullmatch(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                    r"[0-9a-f]{4}-[0-9a-f]{12}",
                    event_id,
                )
                is None
                or wrapper.get("EventId") != event_id
                or wrapper.get("EventName") != "ExecuteChangeSet"
                or type(event_time) is not str
                or re.fullmatch(
                    r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T"
                    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                    event_time,
                )
                is None
            ):
                raise FoundationError("CloudTrail execute-event identity is not exact")
            matches[event_id] = {
                "event_id": event_id,
                "event_time": event_time,
            }
        token = response.get("NextToken")
        if token is None:
            break
        if type(token) is not str or not token:
            raise FoundationError("CloudTrail execute-event token is malformed")
        next_token = token
    else:
        raise FoundationError("CloudTrail execute-event lookup is incomplete")
    if len(matches) != 1:
        raise FoundationError(
            "CloudTrail has no unique exact ExecuteChangeSet event; "
            "delivery may lag, so retry within the 90-day lookup window"
        )
    return next(iter(matches.values()))


def canonical_json_bytes(value: object) -> bytes:
    """Return the one ASCII canonical encoding accepted for sealed evidence."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise FoundationError("value is not canonical JSON data") from exc


def _guard_transaction_inputs(
    *,
    services: FoundationServices,
    output_directory: Path,
    sleep: Callable[[float], None],
    max_polls: int,
) -> dict[str, Path]:
    if (
        type(services) is not FoundationServices
        or services.total_max_attempts != 1
        or not callable(sleep)
        or type(max_polls) is not int
        or not 2 <= max_polls <= 720
    ):
        raise FoundationError(
            "retained-foundation services or polling bound is not exact"
        )
    directory = Path(output_directory)
    if (
        not directory.is_absolute()
        or not directory.is_dir()
        or directory.is_symlink()
        or directory.resolve(strict=True) != directory
    ):
        raise FoundationError(
            "output directory must be one existing absolute non-symlink path"
        )
    paths = {
        "before": directory / "retained-before-template.json",
        "after": directory / "retained-after-template.json",
        "change_set": directory / "retained-change-set-evidence.json",
        "readback": directory / "retained-readback-evidence.json",
    }
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FoundationError("retained-foundation output already exists")
    return paths


def _write_once(path: Path, raw: bytes) -> None:
    if type(raw) is not bytes or not raw:
        raise FoundationError("sealed output bytes are empty")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("sealed output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise FoundationError("sealed output path is no longer unused") from exc
    except OSError as exc:
        raise FoundationError("sealed output could not be persisted") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _call(client: object, method_name: str, **request: object) -> dict[str, object]:
    method = getattr(client, method_name, None)
    if not callable(method):
        raise FoundationError(f"AWS client method {method_name} is absent")
    response = method(**request)
    if type(response) is not dict:
        raise FoundationError(f"AWS {method_name} returned a non-object")
    return response


def _error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if type(response) is not dict:
        return None
    detail = response.get("Error")
    if type(detail) is not dict:
        return None
    code = detail.get("Code")
    return code if type(code) is str else None


def _ambiguous_exception(error: BaseException) -> bool:
    if isinstance(
        error,
        (
            TimeoutError,
            ConnectionError,
            BotoCoreConnectionError,
            BotoCoreHTTPClientError,
        ),
    ):
        return True
    return _error_code(error) in {
        "InternalFailure",
        "PriorRequestNotComplete",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
    }


def _describe_stack(
    cloudformation: object,
    identity: str,
    *,
    allowed_statuses: set[str],
) -> dict[str, object]:
    response = _call(
        cloudformation,
        "describe_stacks",
        StackName=identity,
    )
    stacks = response.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1 or type(stacks[0]) is not dict:
        raise FoundationError("retained stack lookup is not singular")
    stack = stacks[0]
    stack_id = stack.get("StackId")
    if (
        stack.get("StackName") != STACK_NAME
        or type(stack_id) is not str
        or re.fullmatch(
            rf"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
            rf"{STACK_NAME}/[0-9a-f-]+",
            stack_id,
        )
        is None
        or stack.get("StackStatus") not in allowed_statuses
        or type(stack.get("Parameters", [])) is not list
        or type(stack.get("Outputs", [])) is not list
        or type(stack.get("Tags", [])) is not list
        or type(stack.get("EnableTerminationProtection")) is not bool
    ):
        raise FoundationError("retained stack identity or state is not exact")
    return deepcopy(stack)


def _termination_protection(
    stack: Mapping[str, object],
) -> bool:
    value = stack.get("EnableTerminationProtection")
    if type(value) is not bool:
        raise FoundationError("retained termination-protection readback is malformed")
    return value


def _authenticated_existing_stack_role(
    iam: object,
    value: object,
) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or re.fullmatch(
            rf"arn:aws:iam::{ACCOUNT_ID}:role/[A-Za-z0-9+=,.@_/-]+",
            value,
        )
        is None
    ):
        raise FoundationError("retained stack RoleARN is malformed")
    role_name = value.partition(":role/")[2]
    if role_name in {DEPLOYMENT_ROLE_NAME, FENCE_ROLE_NAME}:
        raise FoundationError(
            "bootstrap cannot use a role created by the same change set"
        )
    response = _call(iam, "get_role", RoleName=role_name)
    role = response.get("Role")
    if (
        type(role) is not dict
        or role.get("Arn") != value
        or role.get("RoleName") != role_name
        or type(role.get("RoleId")) is not str
        or not role["RoleId"]
    ):
        raise FoundationError(
            "retained stack RoleARN is not authenticated by exact IAM readback"
        )
    return value


def _assert_absent_call(
    client: object,
    method_name: str,
    *,
    absent_codes: set[str],
    label: str,
    **request: object,
) -> None:
    try:
        _call(client, method_name, **request)
    except Exception as exc:
        if _error_code(exc) in absent_codes:
            return
        raise FoundationError(f"{label} absence could not be proved") from exc
    raise FoundationError(f"{label} already exists")


def _assert_physical_names_absent(services: FoundationServices) -> None:
    for role_name in (DEPLOYMENT_ROLE_NAME, FENCE_ROLE_NAME):
        _assert_absent_call(
            services.iam,
            "get_role",
            absent_codes={"NoSuchEntity", "NoSuchEntityException"},
            label="IAM role " + role_name,
            RoleName=role_name,
        )
    _assert_absent_call(
        services.dynamodb,
        "describe_table",
        absent_codes={"ResourceNotFoundException"},
        label="DynamoDB table " + LEDGER_TABLE_NAME,
        TableName=LEDGER_TABLE_NAME,
    )


def _assert_required_exports_absent(cloudformation: object) -> None:
    token: str | None = None
    seen: set[str] = set()
    required = set(_REQUIRED_EXPORTS.values())
    for _page in range(64):
        request: dict[str, object] = {}
        if token is not None:
            request["NextToken"] = token
        response = _call(cloudformation, "list_exports", **request)
        exports = response.get("Exports")
        if type(exports) is not list or any(type(row) is not dict for row in exports):
            raise FoundationError("CloudFormation export inventory is malformed")
        conflict = sorted(
            row["Name"]
            for row in exports
            if type(row.get("Name")) is str and row["Name"] in required
        )
        if conflict:
            raise FoundationError(
                "required retained export already exists: " + ",".join(conflict)
            )
        next_token = response.get("NextToken")
        if next_token is None:
            return
        if type(next_token) is not str or not next_token or next_token in seen:
            raise FoundationError("CloudFormation export pagination is incomplete")
        seen.add(next_token)
        token = next_token
    raise FoundationError("CloudFormation export pagination exceeded its bound")


def _assert_change_set_absent(
    cloudformation: object,
    stack_id: str,
) -> None:
    try:
        _call(
            cloudformation,
            "describe_change_set",
            ChangeSetName=CHANGE_SET_NAME,
            StackName=stack_id,
            IncludePropertyValues=True,
        )
    except Exception as exc:
        code = _error_code(exc)
        if code == "ChangeSetNotFound" or (
            code == "ValidationError" and "not exist" in str(exc).lower()
        ):
            return
        raise FoundationError(
            "deterministic change-set absence could not be proved"
        ) from exc
    raise FoundationError("deterministic change set already exists")


def _template_body(value: object) -> dict[str, object]:
    if type(value) is dict:
        canonical_json_bytes(value)
        return deepcopy(value)
    if type(value) is str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = yaml.load(value, Loader=_CloudFormationLoader)
            except yaml.YAMLError as exc:
                raise FoundationError(
                    "CloudFormation Original template is not valid JSON or YAML"
                ) from exc
        if type(parsed) is not dict:
            raise FoundationError("CloudFormation Original template is not one object")
        canonical_json_bytes(parsed)
        return parsed
    raise FoundationError("CloudFormation Original template body is malformed")


class _CloudFormationLoader(yaml.SafeLoader):
    """Safe YAML loader that preserves CloudFormation short-form intrinsics."""

    def construct_mapping(
        self,
        node: yaml.nodes.MappingNode,
        deep: bool = False,
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_SHORT_INTRINSICS = {
    "And": "Fn::And",
    "Base64": "Fn::Base64",
    "Cidr": "Fn::Cidr",
    "Condition": "Condition",
    "Contains": "Fn::Contains",
    "EachMemberEquals": "Fn::EachMemberEquals",
    "EachMemberIn": "Fn::EachMemberIn",
    "Equals": "Fn::Equals",
    "FindInMap": "Fn::FindInMap",
    "ForEach": "Fn::ForEach",
    "GetAtt": "Fn::GetAtt",
    "GetAZs": "Fn::GetAZs",
    "If": "Fn::If",
    "ImportValue": "Fn::ImportValue",
    "Join": "Fn::Join",
    "Length": "Fn::Length",
    "Not": "Fn::Not",
    "Or": "Fn::Or",
    "Ref": "Ref",
    "Select": "Fn::Select",
    "Split": "Fn::Split",
    "Sub": "Fn::Sub",
    "ToJsonString": "Fn::ToJsonString",
    "Transform": "Fn::Transform",
    "ValueOf": "Fn::ValueOf",
    "ValueOfAll": "Fn::ValueOfAll",
}


def _construct_cloudformation_intrinsic(
    loader: _CloudFormationLoader,
    tag_suffix: str,
    node: yaml.nodes.Node,
) -> dict[str, object]:
    key = _SHORT_INTRINSICS.get(tag_suffix)
    if key is None:
        raise yaml.constructor.ConstructorError(
            None,
            None,
            f"unsupported CloudFormation YAML tag !{tag_suffix}",
            node.start_mark,
        )
    if isinstance(node, yaml.nodes.ScalarNode):
        nested: object = loader.construct_scalar(node)
    elif isinstance(node, yaml.nodes.SequenceNode):
        nested = loader.construct_sequence(node, deep=True)
    elif isinstance(node, yaml.nodes.MappingNode):
        nested = loader.construct_mapping(node, deep=True)
    else:  # pragma: no cover - PyYAML nodes are a closed hierarchy
        raise yaml.constructor.ConstructorError(
            None,
            None,
            f"unsupported CloudFormation YAML node for !{tag_suffix}",
            node.start_mark,
        )
    if tag_suffix == "GetAtt" and type(nested) is str:
        nested = nested.split(".", 1)
    return {key: nested}


_CloudFormationLoader.add_multi_constructor(
    "!",
    _construct_cloudformation_intrinsic,
)


def _execution_template_bytes(
    *,
    raw_template_body: object,
    original_template: Mapping[str, object],
    retained_template: Mapping[str, object],
) -> bytes:
    """Preserve live YAML bytes while inserting the reviewed foundation delta."""

    if canonical_json_bytes(_template_body(raw_template_body)) != canonical_json_bytes(
        original_template
    ):
        raise FoundationError("live Original template bytes and parsed body disagree")
    if canonical_json_bytes(
        compose_retained_foundation_template(original_template)
    ) != canonical_json_bytes(retained_template):
        raise FoundationError("retained template is not the exact reviewed composition")
    if type(raw_template_body) is not str:
        return canonical_json_bytes(retained_template) + b"\n"

    lines = raw_template_body.splitlines(keepends=True)
    resources_rows = [
        index for index, line in enumerate(lines) if line.rstrip("\r\n") == "Resources:"
    ]
    outputs_rows = [
        index for index, line in enumerate(lines) if line.rstrip("\r\n") == "Outputs:"
    ]
    if (
        len(resources_rows) != 1
        or len(outputs_rows) != 1
        or resources_rows[0] >= outputs_rows[0]
    ):
        raise FoundationError(
            "live YAML template does not have singular Resources and Outputs sections"
        )

    original_outputs = original_template.get("Outputs")
    if type(original_outputs) is not dict:
        raise FoundationError("live YAML Outputs section is malformed")
    for output_id in ("SubnetId", "VpcId"):
        output = original_outputs.get(output_id)
        if type(output) is not dict:
            raise FoundationError(f"live output {output_id} is malformed")
        if "Export" in output:
            continue
        outputs_row = next(
            index
            for index, line in enumerate(lines)
            if line.rstrip("\r\n") == "Outputs:"
        )
        key_rows = [
            index
            for index in range(outputs_row + 1, len(lines))
            if re.fullmatch(
                rf"  {re.escape(output_id)}:\s*",
                lines[index].rstrip("\r\n"),
            )
            is not None
        ]
        if len(key_rows) != 1:
            raise FoundationError(f"live YAML output {output_id} is not singular")
        insert_at = len(lines)
        for index in range(key_rows[0] + 1, len(lines)):
            if re.fullmatch(
                r"  [A-Za-z][A-Za-z0-9]*:\s*",
                lines[index].rstrip("\r\n"),
            ):
                insert_at = index
                break
        lines[insert_at:insert_at] = [
            "    Export:\n",
            f"      Name: {_REQUIRED_EXPORTS[output_id]}\n",
        ]

    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += "\n"
    new_outputs = {
        output_id: retained_template["Outputs"][output_id]
        for output_id in ("CampaignKmsKeyArn", "H1gLedgerArn")
    }
    output_yaml = yaml.safe_dump(
        new_outputs,
        sort_keys=False,
        default_flow_style=False,
    )
    lines.extend("  " + line for line in output_yaml.splitlines(keepends=True))

    outputs_row = next(
        index for index, line in enumerate(lines) if line.rstrip("\r\n") == "Outputs:"
    )
    new_resources = {
        logical_id: retained_template["Resources"][logical_id]
        for logical_id in _FOUNDATION_RESOURCE_TYPES
    }
    resource_yaml = yaml.safe_dump(
        new_resources,
        sort_keys=False,
        default_flow_style=False,
    )
    lines[outputs_row:outputs_row] = [
        "  " + line for line in resource_yaml.splitlines(keepends=True)
    ]
    execution = "".join(lines).encode("utf-8")
    if not execution.endswith(b"\n"):
        execution += b"\n"
    if canonical_json_bytes(_template_body(execution.decode("utf-8"))) != (
        canonical_json_bytes(retained_template)
    ):
        raise FoundationError(
            "live YAML splice did not produce the reviewed retained template"
        )
    return execution


def _get_template_source(
    cloudformation: object,
    *,
    stack_name: str | None = None,
    change_set_name: str | None = None,
) -> tuple[object, dict[str, object]]:
    if (stack_name is None) == (change_set_name is None):
        raise FoundationError("one CloudFormation template identity is required")
    request: dict[str, object] = {"TemplateStage": "Original"}
    if stack_name is not None:
        request["StackName"] = stack_name
    else:
        request["ChangeSetName"] = change_set_name
    response = _call(cloudformation, "get_template", **request)
    raw_template_body = response.get("TemplateBody")
    return raw_template_body, _template_body(raw_template_body)


def _get_template(
    cloudformation: object,
    *,
    stack_name: str | None = None,
    change_set_name: str | None = None,
) -> dict[str, object]:
    _raw, parsed = _get_template_source(
        cloudformation,
        stack_name=stack_name,
        change_set_name=change_set_name,
    )
    return parsed


def _previous_parameters(value: object) -> list[dict[str, object]]:
    if type(value) is not list:
        raise FoundationError("retained stack parameter list is malformed")
    names: list[str] = []
    for row in value:
        if (
            type(row) is not dict
            or type(row.get("ParameterKey")) is not str
            or not row["ParameterKey"]
            or row["ParameterKey"] in names
        ):
            raise FoundationError("retained stack parameters are not unique")
        names.append(row["ParameterKey"])
    return [{"ParameterKey": name, "UsePreviousValue": True} for name in sorted(names)]


def _parameter_readback(value: object) -> list[dict[str, object]]:
    if type(value) is not list:
        raise FoundationError("retained stack parameter readback is malformed")
    parameters: list[dict[str, object]] = []
    names: set[str] = set()
    for row in value:
        if type(row) is not dict or set(row) not in (
            {"ParameterKey", "ParameterValue"},
            {"ParameterKey", "ParameterValue", "ResolvedValue"},
        ):
            raise FoundationError("retained stack parameter readback is malformed")
        name = row.get("ParameterKey")
        parameter_value = row.get("ParameterValue")
        resolved_value = row.get("ResolvedValue")
        if (
            type(name) is not str
            or not name
            or name in names
            or type(parameter_value) is not str
            or ("ResolvedValue" in row and type(resolved_value) is not str)
        ):
            raise FoundationError("retained stack parameter readback is malformed")
        names.add(name)
        parameters.append(deepcopy(row))
    return sorted(parameters, key=lambda row: str(row["ParameterKey"]))


def _parameter_readbacks_match(
    actual: object,
    expected: object,
    *,
    allow_gpu_resolution_drift: bool,
) -> bool:
    actual_rows = _parameter_readback(actual)
    expected_rows = _parameter_readback(expected)
    if actual_rows == expected_rows:
        return True
    if not allow_gpu_resolution_drift or len(actual_rows) != len(expected_rows):
        return False
    for actual_row, expected_row in zip(actual_rows, expected_rows):
        if actual_row == expected_row:
            continue
        if (
            actual_row.get("ParameterKey") != "GpuAmiId"
            or set(actual_row) != {"ParameterKey", "ParameterValue", "ResolvedValue"}
            or set(expected_row) != {"ParameterKey", "ParameterValue", "ResolvedValue"}
            or actual_row["ParameterKey"] != expected_row["ParameterKey"]
            or actual_row["ParameterValue"] != expected_row["ParameterValue"]
            or actual_row["ResolvedValue"] == expected_row["ResolvedValue"]
        ):
            return False
    return True


def _active_resource_types(
    template: Mapping[str, object],
    parameters: object,
) -> dict[str, str]:
    resources = template.get("Resources")
    conditions = template.get("Conditions", {})
    if type(resources) is not dict or type(conditions) is not dict:
        raise FoundationError("retained template condition inventory is malformed")

    parameter_rows = _parameter_readback(parameters)
    parameter_values = {
        str(row["ParameterKey"]): row.get("ResolvedValue", row["ParameterValue"])
        for row in parameter_rows
    }
    pseudo_values = {
        "AWS::AccountId": ACCOUNT_ID,
        "AWS::Partition": "aws",
        "AWS::Region": REGION,
        "AWS::StackName": STACK_NAME,
        "AWS::URLSuffix": "amazonaws.com",
    }
    evaluated: dict[str, bool] = {}

    def evaluate_operand(value: object, trail: frozenset[str]) -> object:
        if type(value) in {str, int, float, bool} or value is None:
            return value
        if type(value) is not dict or len(value) != 1:
            raise FoundationError(
                "retained template condition expression is unsupported"
            )
        operator, operand = next(iter(value.items()))
        if operator == "Ref":
            if type(operand) is not str:
                raise FoundationError("retained template condition Ref is malformed")
            if operand in parameter_values:
                return parameter_values[operand]
            if operand in pseudo_values:
                return pseudo_values[operand]
            raise FoundationError("retained template condition Ref is unsupported")
        if operator == "Condition":
            if type(operand) is not str:
                raise FoundationError(
                    "retained template condition reference is malformed"
                )
            return evaluate_condition(operand, trail)
        if operator == "Fn::Equals":
            if type(operand) is not list or len(operand) != 2:
                raise FoundationError("retained template Fn::Equals is malformed")
            left = evaluate_operand(operand[0], trail)
            right = evaluate_operand(operand[1], trail)
            if type(left) is not str or type(right) is not str:
                raise FoundationError(
                    "retained template Fn::Equals operands are unsupported"
                )
            return left == right
        if operator in {"Fn::And", "Fn::Or"}:
            if type(operand) is not list or not 2 <= len(operand) <= 10:
                raise FoundationError(f"retained template {operator} is malformed")
            values = [evaluate_operand(item, trail) for item in operand]
            if any(type(item) is not bool for item in values):
                raise FoundationError(
                    f"retained template {operator} operand is not boolean"
                )
            return all(values) if operator == "Fn::And" else any(values)
        if operator == "Fn::Not":
            if type(operand) is not list or len(operand) != 1:
                raise FoundationError("retained template Fn::Not is malformed")
            evaluated_operand = evaluate_operand(operand[0], trail)
            if type(evaluated_operand) is not bool:
                raise FoundationError(
                    "retained template Fn::Not operand is not boolean"
                )
            return not evaluated_operand
        raise FoundationError("retained template condition expression is unsupported")

    def evaluate_condition(name: str, trail: frozenset[str]) -> bool:
        if name in evaluated:
            return evaluated[name]
        if name in trail or name not in conditions:
            raise FoundationError("retained template condition reference is invalid")
        result = evaluate_operand(conditions[name], trail | {name})
        if type(result) is not bool:
            raise FoundationError("retained template condition is not boolean")
        evaluated[name] = result
        return result

    if any(type(name) is not str or not name for name in conditions):
        raise FoundationError("retained template condition name is malformed")
    for condition_name in sorted(conditions):
        evaluate_condition(condition_name, frozenset())

    active: dict[str, str] = {}
    for logical_id, resource in resources.items():
        if (
            type(logical_id) is not str
            or not logical_id
            or type(resource) is not dict
            or type(resource.get("Type")) is not str
            or not resource["Type"]
        ):
            raise FoundationError("retained template resource inventory is malformed")
        condition_name = resource.get("Condition")
        if condition_name is not None and (
            type(condition_name) is not str
            or not condition_name
            or not evaluate_condition(condition_name, frozenset())
        ):
            if type(condition_name) is not str or not condition_name:
                raise FoundationError("retained resource condition is malformed")
            continue
        active[logical_id] = resource["Type"]
    return active


def _publish_foundation_template(
    *,
    raw: bytes,
    expected_template: Mapping[str, object],
    expected_file_sha256: str,
    expected_body_sha256: str,
    bucket: str,
    services: Task13ReviewedArtifactServices,
) -> dict[str, object]:
    if bucket != RETAINED_MODELS_BUCKET:
        raise Task13ReviewedArtifactError(
            "foundation template bucket is not the retained models bucket"
        )
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or canonical_json_bytes(_template_body(raw.decode("utf-8")))
        != canonical_json_bytes(expected_template)
    ):
        raise Task13ReviewedArtifactError(
            "foundation execution template is not the reviewed semantic template"
        )
    if (
        type(expected_file_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256) is None
        or type(expected_body_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_body_sha256) is None
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
        or hashlib.sha256(raw[:-1]).hexdigest() != expected_body_sha256
    ):
        raise Task13ReviewedArtifactError(
            "foundation template bytes do not match both SHA-256 pins"
        )
    _guard_reviewed_artifact_services(services, bucket)
    version = _publish_reviewed_artifact_once(
        s3=services.s3,
        artifact_kind=FOUNDATION_TEMPLATE_ARTIFACT_KIND,
        key=FOUNDATION_TEMPLATE_KEY,
        raw=raw,
    )
    return {
        "artifact_kind": FOUNDATION_TEMPLATE_ARTIFACT_KIND,
        "bucket": RETAINED_MODELS_BUCKET,
        "key": FOUNDATION_TEMPLATE_KEY,
        "version_id": version,
        "file_sha256": expected_file_sha256,
        "body_sha256": expected_body_sha256,
    }


def _versioned_template_url(
    coordinate: Mapping[str, object],
) -> str:
    if (
        type(coordinate) is not dict
        or coordinate.get("artifact_kind") != FOUNDATION_TEMPLATE_ARTIFACT_KIND
        or coordinate.get("bucket") != RETAINED_MODELS_BUCKET
        or coordinate.get("key") != FOUNDATION_TEMPLATE_KEY
        or type(coordinate.get("version_id")) is not str
        or not coordinate["version_id"]
    ):
        raise FoundationError("retained template coordinate is not exact")
    return (
        f"https://{RETAINED_MODELS_BUCKET}.s3.{REGION}.amazonaws.com/"
        f"{FOUNDATION_TEMPLATE_KEY}?versionId="
        + quote(coordinate["version_id"], safe="")
    )


def _describe_full_change_set(
    cloudformation: object,
    *,
    stack_id: str,
    identity: str,
) -> dict[str, object]:
    token: str | None = None
    seen: set[str] = set()
    combined: dict[str, object] | None = None
    changes: list[object] = []
    for _page in range(64):
        request: dict[str, object] = {
            "ChangeSetName": identity,
            "StackName": stack_id,
            "IncludePropertyValues": True,
        }
        if token is not None:
            request["NextToken"] = token
        response = _call(
            cloudformation,
            "describe_change_set",
            **request,
        )
        page_changes = response.get("Changes", [])
        if type(page_changes) is not list:
            raise FoundationError("change-set change page is malformed")
        changes.extend(deepcopy(page_changes))
        projection = {
            key: value
            for key, value in response.items()
            if key not in {"Changes", "NextToken", "ResponseMetadata"}
        }
        if combined is None:
            combined = deepcopy(projection)
        elif canonical_json_bytes(projection) != canonical_json_bytes(combined):
            raise FoundationError("change-set identity changed across pages")
        next_token = response.get("NextToken")
        if next_token is None:
            combined = {} if combined is None else combined
            combined["Changes"] = changes
            return combined
        if type(next_token) is not str or not next_token or next_token in seen:
            raise FoundationError("change-set pagination is incomplete")
        seen.add(next_token)
        token = next_token
    raise FoundationError("change-set pagination exceeded its bound")


def _wait_for_change_set(
    cloudformation: object,
    *,
    stack_id: str,
    sleep: Callable[[float], None],
    max_polls: int,
) -> dict[str, object]:
    for poll in range(max_polls):
        response = _describe_full_change_set(
            cloudformation,
            stack_id=stack_id,
            identity=CHANGE_SET_NAME,
        )
        status = response.get("Status")
        if status == "CREATE_COMPLETE":
            return response
        if status not in {"CREATE_PENDING", "CREATE_IN_PROGRESS"}:
            raise FoundationError("change set reached a terminal failed state")
        if poll + 1 < max_polls:
            sleep(5.0)
    raise FoundationError("change-set creation polling exceeded its bound")


def _stack_guard(stack: Mapping[str, object]) -> dict[str, object]:
    return {
        "StackId": stack.get("StackId"),
        "StackName": stack.get("StackName"),
        "StackStatus": stack.get("StackStatus"),
        "Parameters": stack.get("Parameters", []),
        "Outputs": stack.get("Outputs", []),
        "RoleARN": stack.get("RoleARN"),
        "Tags": stack.get("Tags", []),
    }


def _normalized_changes(
    change_set: Mapping[str, object],
) -> list[dict[str, str]]:
    changes = change_set.get("Changes")
    if type(changes) is not list:
        raise FoundationError("change-set changes are malformed")
    result = []
    for change in changes:
        resource = change.get("ResourceChange") if type(change) is dict else None
        if type(resource) is not dict:
            raise FoundationError("change-set resource change is malformed")
        result.append(
            {
                "action": str(resource.get("Action")),
                "logical_id": str(resource.get("LogicalResourceId")),
                "replacement": str(resource.get("Replacement")),
                "resource_type": str(resource.get("ResourceType")),
            }
        )
    return sorted(result, key=lambda row: row["logical_id"])


def _reconcile_ambiguous_execute(
    cloudformation: object,
    *,
    stack_id: str,
    change_set_id: str,
) -> None:
    response = _describe_full_change_set(
        cloudformation,
        stack_id=stack_id,
        identity=change_set_id,
    )
    if (
        response.get("ChangeSetId") != change_set_id
        or response.get("StackId") != stack_id
        or response.get("ChangeSetName") != CHANGE_SET_NAME
        or response.get("ExecutionStatus")
        not in {"EXECUTE_IN_PROGRESS", "EXECUTE_COMPLETE"}
    ):
        raise FoundationError(
            "ambiguous ExecuteChangeSet has no exact change-set adoption"
        )
    stack = _describe_stack(
        cloudformation,
        stack_id,
        allowed_statuses={
            "UPDATE_IN_PROGRESS",
            "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
            "UPDATE_COMPLETE",
        },
    )
    if stack["StackStatus"] not in {
        "UPDATE_IN_PROGRESS",
        "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
        "UPDATE_COMPLETE",
    }:
        raise FoundationError("ambiguous ExecuteChangeSet has no exact stack adoption")


def _wait_for_stack_update(
    cloudformation: object,
    *,
    stack_id: str,
    sleep: Callable[[float], None],
    max_polls: int,
) -> dict[str, object]:
    for poll in range(max_polls):
        stack = _describe_stack(
            cloudformation,
            stack_id,
            allowed_statuses={
                "UPDATE_IN_PROGRESS",
                "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
                "UPDATE_COMPLETE",
            },
        )
        status = stack["StackStatus"]
        if status == "UPDATE_COMPLETE":
            return stack
        if poll + 1 < max_polls:
            sleep(10.0)
    raise FoundationError("retained stack update polling exceeded its bound")


def _list_stack_resources(
    cloudformation: object,
    stack_id: str,
) -> list[dict[str, object]]:
    token: str | None = None
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for _page in range(64):
        request: dict[str, object] = {"StackName": stack_id}
        if token is not None:
            request["NextToken"] = token
        response = _call(
            cloudformation,
            "list_stack_resources",
            **request,
        )
        page = response.get("StackResourceSummaries")
        if type(page) is not list or any(type(row) is not dict for row in page):
            raise FoundationError("stack resource page is malformed")
        rows.extend(deepcopy(page))
        next_token = response.get("NextToken")
        if next_token is None:
            return rows
        if type(next_token) is not str or not next_token or next_token in seen:
            raise FoundationError("stack resource pagination is incomplete")
        seen.add(next_token)
        token = next_token
    raise FoundationError("stack resource pagination exceeded its bound")


def _outputs_by_key(value: object) -> dict[str, dict[str, object]]:
    if type(value) is not list:
        raise FoundationError("retained stack outputs are malformed")
    result: dict[str, dict[str, object]] = {}
    for row in value:
        if (
            type(row) is not dict
            or type(row.get("OutputKey")) is not str
            or not row["OutputKey"]
            or row["OutputKey"] in result
            or type(row.get("OutputValue")) is not str
            or not row["OutputValue"]
        ):
            raise FoundationError("retained stack output identity is malformed")
        result[row["OutputKey"]] = deepcopy(row)
    return result


def _resolved_policy_document(
    value: object,
    *,
    get_att_arns: Mapping[tuple[str, str], str],
) -> object:
    if type(value) is dict:
        if set(value) == {"Fn::GetAtt"}:
            target = value["Fn::GetAtt"]
            if (
                type(target) is not list
                or len(target) != 2
                or any(type(part) is not str for part in target)
                or tuple(target) not in get_att_arns
            ):
                raise FoundationError("IAM policy GetAtt is not exact")
            return get_att_arns[(target[0], target[1])]
        if set(value) == {"Fn::Sub"}:
            template = value["Fn::Sub"]
            if type(template) is not str:
                raise FoundationError("IAM policy Sub is not exact")
            resolved = (
                template.replace("${AWS::Partition}", "aws")
                .replace("${AWS::Region}", REGION)
                .replace("${AWS::AccountId}", ACCOUNT_ID)
            )
            if "${" in resolved:
                raise FoundationError("IAM policy Sub contains an unknown variable")
            return resolved
        if any(type(key) is not str for key in value):
            raise FoundationError("IAM policy contains a non-string key")
        if any(key == "Ref" or key.startswith("Fn::") for key in value):
            raise FoundationError("IAM policy contains an unsupported intrinsic")
        return {
            key: _resolved_policy_document(item, get_att_arns=get_att_arns)
            for key, item in value.items()
        }
    if type(value) is list:
        return [
            _resolved_policy_document(item, get_att_arns=get_att_arns) for item in value
        ]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise FoundationError("IAM policy contains noncanonical data")


def _role_readback(
    iam: object,
    *,
    role_name: str,
    expected_resource: Mapping[str, object],
    get_att_arns: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    response = _call(iam, "get_role", RoleName=role_name)
    role = response.get("Role")
    expected_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{role_name}"
    properties = expected_resource.get("Properties")
    if (
        type(role) is not dict
        or role.get("RoleName") != role_name
        or role.get("Arn") != expected_arn
        or type(role.get("RoleId")) is not str
        or not role["RoleId"]
        or type(properties) is not dict
        or role.get("AssumeRolePolicyDocument")
        != properties.get("AssumeRolePolicyDocument")
    ):
        raise FoundationError(f"IAM role {role_name} live readback drifted")
    attached = _call(
        iam,
        "list_attached_role_policies",
        RoleName=role_name,
    )
    if (
        attached.get("AttachedPolicies") != []
        or attached.get("IsTruncated") is not False
    ):
        raise FoundationError(f"IAM role {role_name} has unreviewed managed policies")
    expected_policies = properties.get("Policies")
    if type(expected_policies) is not list or any(
        type(policy) is not dict for policy in expected_policies
    ):
        raise FoundationError(f"IAM role {role_name} template policy is malformed")
    policy_by_name = {
        policy.get("PolicyName"): policy.get("PolicyDocument")
        for policy in expected_policies
    }
    if any(type(name) is not str for name in policy_by_name) or len(
        policy_by_name
    ) != len(expected_policies):
        raise FoundationError(
            f"IAM role {role_name} template policy names are not exact"
        )
    resolved_policy_by_name = {
        name: _resolved_policy_document(document, get_att_arns=get_att_arns)
        for name, document in policy_by_name.items()
    }
    listed = _call(iam, "list_role_policies", RoleName=role_name)
    if (
        listed.get("IsTruncated") is not False
        or type(listed.get("PolicyNames")) is not list
        or sorted(listed["PolicyNames"]) != sorted(policy_by_name)
    ):
        raise FoundationError(f"IAM role {role_name} inline policy inventory drifted")
    policy_hashes: list[str] = []
    for policy_name in sorted(policy_by_name):
        policy_response = _call(
            iam,
            "get_role_policy",
            RoleName=role_name,
            PolicyName=policy_name,
        )
        policy_document = policy_response.get("PolicyDocument")
        if (
            policy_response.get("RoleName") != role_name
            or policy_response.get("PolicyName") != policy_name
            or policy_document != resolved_policy_by_name[policy_name]
        ):
            raise FoundationError(
                f"IAM role {role_name} inline policy readback drifted"
            )
        policy_hashes.append(
            hashlib.sha256(canonical_json_bytes(policy_document)).hexdigest()
        )
    return {
        "role_name": role_name,
        "role_arn": expected_arn,
        "role_id": role["RoleId"],
        "assume_role_policy_sha256": hashlib.sha256(
            canonical_json_bytes(role["AssumeRolePolicyDocument"])
        ).hexdigest(),
        "inline_policy_sha256": policy_hashes[0],
    }


def _validate_live_readback(
    *,
    services: FoundationServices,
    initial_stack: Mapping[str, object],
    expected_parameters: list[dict[str, object]],
    final_stack: Mapping[str, object],
    final_template: Mapping[str, object],
    final_template_sha256: str,
    resources: list[dict[str, object]],
    expected_template: Mapping[str, object],
    termination_protection: bool,
    template_coordinate: Mapping[str, object],
    execute_was_ambiguous: bool,
) -> dict[str, object]:
    if canonical_json_bytes(final_template) != canonical_json_bytes(expected_template):
        raise FoundationError("final retained template semantic readback drifted")
    if (
        final_stack.get("StackId") != initial_stack.get("StackId")
        or final_stack.get("StackName") != STACK_NAME
        or final_stack.get("StackStatus") != "UPDATE_COMPLETE"
        or final_stack.get("RoleARN") != initial_stack.get("RoleARN")
        or _parameter_readback(final_stack.get("Parameters", []))
        != _parameter_readback(expected_parameters)
        or final_stack.get("Tags", []) != initial_stack.get("Tags", [])
    ):
        raise FoundationError("final retained stack identity drifted")

    expected_types = _active_resource_types(expected_template, expected_parameters)
    observed: dict[str, dict[str, object]] = {}
    for row in resources:
        logical_id = row.get("LogicalResourceId")
        if (
            type(logical_id) is not str
            or logical_id in observed
            or type(row.get("PhysicalResourceId")) is not str
            or not row["PhysicalResourceId"]
            or type(row.get("ResourceType")) is not str
            or row.get("ResourceStatus") not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
        ):
            raise FoundationError("final stack resource identity is incomplete")
        observed[logical_id] = row
    if {
        logical_id: row["ResourceType"] for logical_id, row in observed.items()
    } != expected_types:
        raise FoundationError("final stack resource inventory is not complete")
    expected_physical = {
        "H1gLedger": LEDGER_TABLE_NAME,
        "H1gCloudFormationDeploymentRole": DEPLOYMENT_ROLE_NAME,
        "H1gFenceServiceRole": FENCE_ROLE_NAME,
    }
    for logical_id, physical_id in expected_physical.items():
        if observed[logical_id]["PhysicalResourceId"] != physical_id:
            raise FoundationError(f"final {logical_id} physical identity drifted")

    initial_outputs = _outputs_by_key(initial_stack.get("Outputs", []))
    final_outputs = _outputs_by_key(final_stack.get("Outputs", []))
    for output_key, output in initial_outputs.items():
        final = final_outputs.get(output_key)
        if (
            type(final) is not dict
            or final.get("OutputValue") != output.get("OutputValue")
            or (
                output.get("ExportName") is not None
                and final.get("ExportName") != output.get("ExportName")
            )
        ):
            raise FoundationError(
                f"legacy retained output {output_key} was not preserved"
            )
    for output_key, export_name in _REQUIRED_EXPORTS.items():
        output = final_outputs.get(output_key)
        if type(output) is not dict or output.get("ExportName") != export_name:
            raise FoundationError(f"required retained export {export_name} is absent")
    kms_arn = final_outputs["CampaignKmsKeyArn"]["OutputValue"]
    ledger_arn = final_outputs["H1gLedgerArn"]["OutputValue"]
    if (
        type(kms_arn) is not str
        or re.fullmatch(
            rf"arn:aws:kms:{REGION}:{ACCOUNT_ID}:key/[0-9a-f-]+",
            kms_arn,
        )
        is None
        or observed["CampaignKmsKey"]["PhysicalResourceId"]
        != kms_arn.rpartition("/")[2]
        or ledger_arn
        != (f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{LEDGER_TABLE_NAME}")
    ):
        raise FoundationError("retained key or ledger output identity drifted")

    policy_get_att_arns = {
        ("CampaignKmsKey", "Arn"): kms_arn,
        ("H1gLedger", "Arn"): ledger_arn,
        ("ModelBucket", "Arn"): f"arn:aws:s3:::{RETAINED_MODELS_BUCKET}",
    }
    deployment_role = _role_readback(
        services.iam,
        role_name=DEPLOYMENT_ROLE_NAME,
        expected_resource=expected_template["Resources"][
            "H1gCloudFormationDeploymentRole"
        ],
        get_att_arns=policy_get_att_arns,
    )
    fence_role = _role_readback(
        services.iam,
        role_name=FENCE_ROLE_NAME,
        expected_resource=expected_template["Resources"]["H1gFenceServiceRole"],
        get_att_arns=policy_get_att_arns,
    )
    table_response = _call(
        services.dynamodb,
        "describe_table",
        TableName=LEDGER_TABLE_NAME,
    )
    table = table_response.get("Table")
    if (
        type(table) is not dict
        or table.get("TableName") != LEDGER_TABLE_NAME
        or table.get("TableArn") != ledger_arn
        or table.get("TableStatus") != "ACTIVE"
        or table.get("DeletionProtectionEnabled") is not True
        or type(table.get("SSEDescription")) is not dict
        or table["SSEDescription"].get("Status") != "ENABLED"
        or table["SSEDescription"].get("SSEType") != "KMS"
        or table["SSEDescription"].get("KMSMasterKeyArn") != kms_arn
    ):
        raise FoundationError("retained ledger live readback drifted")
    key_response = _call(services.kms, "describe_key", KeyId=kms_arn)
    key = key_response.get("KeyMetadata")
    if (
        type(key) is not dict
        or key.get("AWSAccountId") != ACCOUNT_ID
        or key.get("Arn") != kms_arn
        or key.get("Enabled") is not True
        or key.get("KeyManager") != "CUSTOMER"
        or key.get("KeyState") != "Enabled"
    ):
        raise FoundationError("campaign KMS key live readback drifted")
    return {
        "schema_version": 1,
        "record_type": "glm52_task13_retained_foundation_readback_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "stack_id": final_stack["StackId"],
        "stack_name": STACK_NAME,
        "stack_status": "UPDATE_COMPLETE",
        "termination_protection": termination_protection,
        "role_arn": final_stack.get("RoleARN"),
        "template_sha256": final_template_sha256,
        "template_coordinate": deepcopy(template_coordinate),
        "resource_count": len(resources),
        "foundation_resources": [
            {
                "logical_id": logical_id,
                "physical_id": observed[logical_id]["PhysicalResourceId"],
                "resource_type": observed[logical_id]["ResourceType"],
                "resource_status": observed[logical_id]["ResourceStatus"],
            }
            for logical_id in sorted(_FOUNDATION_RESOURCE_TYPES)
        ],
        "deployment_role": deployment_role,
        "fence_role": fence_role,
        "kms_key_arn": kms_arn,
        "ledger_arn": ledger_arn,
        "required_exports": {
            output_key: export_name
            for output_key, export_name in sorted(_REQUIRED_EXPORTS.items())
        },
        "execute_response_was_ambiguous": execute_was_ambiguous,
    }


def _cloudformation_trust_policy() -> dict[str, object]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CloudFormationOnly",
                "Effect": "Allow",
                "Principal": {"Service": "cloudformation.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }


def _campaign_key() -> dict[str, object]:
    key_actions = [
        "kms:CancelKeyDeletion",
        "kms:CreateGrant",
        "kms:Decrypt",
        "kms:DescribeKey",
        "kms:DisableKey",
        "kms:DisableKeyRotation",
        "kms:EnableKey",
        "kms:EnableKeyRotation",
        "kms:Encrypt",
        "kms:GenerateDataKey",
        "kms:GenerateDataKeyWithoutPlaintext",
        "kms:GetKeyPolicy",
        "kms:GetKeyRotationStatus",
        "kms:ListGrants",
        "kms:ListKeyPolicies",
        "kms:ListResourceTags",
        "kms:PutKeyPolicy",
        "kms:ReEncryptFrom",
        "kms:ReEncryptTo",
        "kms:RetireGrant",
        "kms:RevokeGrant",
        "kms:ScheduleKeyDeletion",
        "kms:TagResource",
        "kms:UntagResource",
        "kms:UpdateKeyDescription",
    ]
    return {
        "Type": "AWS::KMS::Key",
        "DeletionPolicy": "Retain",
        "UpdateReplacePolicy": "Retain",
        "Properties": {
            "Description": "KEEP GLM-5.2 retained H.1g campaign authority",
            "EnableKeyRotation": True,
            "PendingWindowInDays": 30,
            "KeyPolicy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "AccountRootDelegatesReviewedIamPolicies",
                        "Effect": "Allow",
                        "Principal": {
                            "AWS": {
                                "Fn::Sub": (
                                    "arn:${AWS::Partition}:iam::${AWS::AccountId}:root"
                                )
                            }
                        },
                        "Action": key_actions,
                        "Resource": "*",
                    }
                ],
            },
        },
    }


def _ledger() -> dict[str, object]:
    return {
        "Type": "AWS::DynamoDB::Table",
        "DeletionPolicy": "Retain",
        "UpdateReplacePolicy": "Retain",
        "Properties": {
            "TableName": LEDGER_TABLE_NAME,
            "BillingMode": "PAY_PER_REQUEST",
            "AttributeDefinitions": [
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            "KeySchema": [
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            "DeletionProtectionEnabled": True,
            "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
            "SSESpecification": {
                "SSEEnabled": True,
                "SSEType": "KMS",
                "KMSMasterKeyId": {"Fn::GetAtt": ["CampaignKmsKey", "Arn"]},
            },
        },
    }


def _deployment_role() -> dict[str, object]:
    lambda_role_names = (
        "keep-glm52-h1g-execution-observer",
        "keep-glm52-h1g-finalizer",
        "keep-glm52-h1g-h1g-drained-writer",
        "keep-glm52-h1g-operator-disposition-writer",
        "keep-glm52-h1g-orphan-audit",
        "keep-glm52-h1g-retained-support-lifecycle",
        "keep-glm52-h1g-snapshot-cleanup",
        "keep-glm52-h1g-task10-capacity-reconciliation",
        "keep-glm52-h1g-terminal-v2-writer",
        "keep-glm52-h1g-worker-drain-signal",
    )
    state_machine_role_names = (
        "keep-glm52-h1g-production-workflow",
        "keep-glm52-h1g-retained-lifecycle-workflow",
        "keep-glm52-h1g-retainedlifecycle-workflow",
        "keep-glm52-h1g-snapshotcleanup-workflow",
    )

    def sub_arn(value: str) -> dict[str, str]:
        return {
            "Fn::Sub": (
                "arn:${AWS::Partition}:"
                + value.replace("${Region}", "${AWS::Region}").replace(
                    "${Account}",
                    "${AWS::AccountId}",
                )
            )
        }

    def role_arns(names: tuple[str, ...]) -> list[dict[str, str]]:
        return [sub_arn(f"iam::${{Account}}:role/{name}") for name in names]

    lambda_resources = [
        sub_arn("lambda:${Region}:${Account}:function:keep-glm52-h1g-*"),
        sub_arn(
            "lambda:${Region}:${Account}:function:"
            "keep-glm52-gpu-H1gRetainedSupportLifecycleFunction-*"
        ),
    ]
    log_group_resources = [
        sub_arn("logs:${Region}:${Account}:log-group:/aws/lambda/keep-glm52-h1g-*:*")
    ]
    log_group_tag_resources = [
        sub_arn("logs:${Region}:${Account}:log-group:/aws/lambda/keep-glm52-h1g-*")
    ]
    state_machine_resources = [
        sub_arn("states:${Region}:${Account}:stateMachine:keep-glm52-h1g-*")
    ]
    alarm_resources = [
        sub_arn("cloudwatch:${Region}:${Account}:alarm:keep-glm52-h1g-*")
    ]
    return {
        "Type": "AWS::IAM::Role",
        "DeletionPolicy": "Retain",
        "UpdateReplacePolicy": "Retain",
        "Properties": {
            "RoleName": DEPLOYMENT_ROLE_NAME,
            "Description": (
                "CloudFormation service role for the reviewed retained H.1g "
                "foundation and Task10/Task12 lifecycle"
            ),
            "AssumeRolePolicyDocument": _cloudformation_trust_policy(),
            "Policies": [
                {
                    "PolicyName": "retained-h1g-complete-lifecycle",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Sid": "ReadAndMaintainCampaignKey",
                                "Effect": "Allow",
                                "Action": [
                                    "kms:DescribeKey",
                                    "kms:EnableKeyRotation",
                                    "kms:GetKeyPolicy",
                                    "kms:GetKeyRotationStatus",
                                    "kms:ListResourceTags",
                                    "kms:PutKeyPolicy",
                                    "kms:TagResource",
                                    "kms:UntagResource",
                                    "kms:UpdateKeyDescription",
                                ],
                                "Resource": {"Fn::GetAtt": ["CampaignKmsKey", "Arn"]},
                            },
                            {
                                "Sid": "ReadAndMaintainRetainedLedger",
                                "Effect": "Allow",
                                "Action": [
                                    "dynamodb:DescribeContinuousBackups",
                                    "dynamodb:DescribeTable",
                                    "dynamodb:DescribeTimeToLive",
                                    "dynamodb:ListTagsOfResource",
                                    "dynamodb:TagResource",
                                    "dynamodb:UntagResource",
                                    "dynamodb:UpdateContinuousBackups",
                                    "dynamodb:UpdateTable",
                                ],
                                "Resource": {"Fn::GetAtt": ["H1gLedger", "Arn"]},
                            },
                            {
                                "Sid": "MaintainExactRetainedRoles",
                                "Effect": "Allow",
                                "Action": [
                                    "iam:CreateRole",
                                    "iam:DeleteRole",
                                    "iam:DeleteRolePolicy",
                                    "iam:GetRole",
                                    "iam:GetRolePolicy",
                                    "iam:ListRolePolicies",
                                    "iam:ListRoleTags",
                                    "iam:PutRolePolicy",
                                    "iam:TagRole",
                                    "iam:UntagRole",
                                    "iam:UpdateAssumeRolePolicy",
                                    "iam:UpdateRoleDescription",
                                ],
                                "Resource": sub_arn(
                                    "iam::${Account}:role/keep-glm52-h1g-*"
                                ),
                            },
                            {
                                "Sid": "PassRetainedLambdaExecutionRoles",
                                "Effect": "Allow",
                                "Action": "iam:PassRole",
                                "Resource": role_arns(lambda_role_names),
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": ("lambda.amazonaws.com")
                                    }
                                },
                            },
                            {
                                "Sid": "PassRetainedStateMachineRoles",
                                "Effect": "Allow",
                                "Action": "iam:PassRole",
                                "Resource": role_arns(state_machine_role_names),
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": ("states.amazonaws.com")
                                    }
                                },
                            },
                            {
                                "Sid": "PassRetainedEventInvokeRole",
                                "Effect": "Allow",
                                "Action": "iam:PassRole",
                                "Resource": role_arns(
                                    ("keep-glm52-h1g-retained-lifecycle-event-invoke",)
                                ),
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": ("events.amazonaws.com")
                                    }
                                },
                            },
                            {
                                "Sid": "PassRetainedSchedulerInvokeRole",
                                "Effect": "Allow",
                                "Action": "iam:PassRole",
                                "Resource": role_arns(
                                    ("keep-glm52-h1g-retained-support-schedule-invoke",)
                                ),
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": (
                                            "scheduler.amazonaws.com"
                                        )
                                    }
                                },
                            },
                            {
                                "Sid": "MaintainRetainedLambdaFunctions",
                                "Effect": "Allow",
                                "Action": [
                                    "lambda:AddPermission",
                                    "lambda:CreateFunction",
                                    "lambda:DeleteFunction",
                                    "lambda:DeleteFunctionConcurrency",
                                    "lambda:GetFunction",
                                    "lambda:GetFunctionConcurrency",
                                    "lambda:GetPolicy",
                                    "lambda:ListTags",
                                    "lambda:ListVersionsByFunction",
                                    "lambda:PublishVersion",
                                    "lambda:PutFunctionConcurrency",
                                    "lambda:RemovePermission",
                                    "lambda:TagResource",
                                    "lambda:UntagResource",
                                    "lambda:UpdateFunctionCode",
                                    "lambda:UpdateFunctionConfiguration",
                                ],
                                "Resource": lambda_resources,
                            },
                            {
                                "Sid": "MaintainRetainedAlarms",
                                "Effect": "Allow",
                                "Action": [
                                    "cloudwatch:DeleteAlarms",
                                    "cloudwatch:DescribeAlarms",
                                    "cloudwatch:ListTagsForResource",
                                    "cloudwatch:PutMetricAlarm",
                                    "cloudwatch:TagResource",
                                    "cloudwatch:UntagResource",
                                ],
                                "Resource": alarm_resources,
                            },
                            {
                                "Sid": "MaintainRetainedLogGroups",
                                "Effect": "Allow",
                                "Action": [
                                    "logs:CreateLogGroup",
                                    "logs:DeleteLogGroup",
                                    "logs:DeleteRetentionPolicy",
                                    "logs:PutRetentionPolicy",
                                ],
                                "Resource": log_group_resources,
                            },
                            {
                                "Sid": "TagExactRetainedLogGroups",
                                "Effect": "Allow",
                                "Action": [
                                    "logs:ListTagsForResource",
                                    "logs:TagResource",
                                    "logs:UntagResource",
                                ],
                                "Resource": log_group_tag_resources,
                            },
                            {
                                "Sid": "DescribeRetainedLogGroups",
                                "Effect": "Allow",
                                "Action": "logs:DescribeLogGroups",
                                "Resource": "*",
                                "Condition": {
                                    "StringEquals": {"aws:RequestedRegion": REGION}
                                },
                            },
                            {
                                "Sid": "MaintainRetainedEventRule",
                                "Effect": "Allow",
                                "Action": [
                                    "events:DeleteRule",
                                    "events:DescribeRule",
                                    "events:ListTagsForResource",
                                    "events:ListTargetsByRule",
                                    "events:PutRule",
                                    "events:PutTargets",
                                    "events:RemoveTargets",
                                    "events:TagResource",
                                    "events:UntagResource",
                                ],
                                "Resource": sub_arn(
                                    "events:${Region}:${Account}:rule/"
                                    "keep-glm52-gpu-RetainedLifecycleEventRule-*"
                                ),
                            },
                            {
                                "Sid": "MaintainRetainedStateMachines",
                                "Effect": "Allow",
                                "Action": [
                                    "states:CreateStateMachine",
                                    "states:DeleteStateMachine",
                                    "states:DeleteStateMachineVersion",
                                    "states:DescribeStateMachine",
                                    "states:ListStateMachineVersions",
                                    "states:ListTagsForResource",
                                    "states:PublishStateMachineVersion",
                                    "states:TagResource",
                                    "states:UntagResource",
                                    "states:UpdateStateMachine",
                                ],
                                "Resource": state_machine_resources,
                            },
                            {
                                "Sid": "MaintainRetainedSchedules",
                                "Effect": "Allow",
                                "Action": [
                                    "scheduler:CreateSchedule",
                                    "scheduler:DeleteSchedule",
                                    "scheduler:GetSchedule",
                                    "scheduler:UpdateSchedule",
                                ],
                                "Resource": [
                                    sub_arn(
                                        "scheduler:${Region}:${Account}:"
                                        "schedule/default/keep-glm52-gpu-"
                                        "H1gRetainedWorkStopSchedule-*"
                                    ),
                                    sub_arn(
                                        "scheduler:${Region}:${Account}:"
                                        "schedule/default/keep-glm52-gpu-"
                                        "H1gRetainedDeleteRequestSchedule-*"
                                    ),
                                    sub_arn(
                                        "scheduler:${Region}:${Account}:"
                                        "schedule/default/keep-glm52-gpu-"
                                        "H1gRetainedAbsenceDeadlineSchedule-*"
                                    ),
                                    sub_arn(
                                        "scheduler:${Region}:${Account}:"
                                        "schedule/default/keep-glm52-h1g-"
                                        "*-nat-accumulator"
                                    ),
                                ],
                            },
                            {
                                "Sid": "MaintainRetainedOperatorTopic",
                                "Effect": "Allow",
                                "Action": [
                                    "sns:CreateTopic",
                                    "sns:DeleteTopic",
                                    "sns:GetTopicAttributes",
                                    "sns:ListTagsForResource",
                                    "sns:SetTopicAttributes",
                                    "sns:TagResource",
                                    "sns:UntagResource",
                                ],
                                "Resource": sub_arn(
                                    "sns:${Region}:${Account}:"
                                    "keep-glm52-h1g-operator-alerts"
                                ),
                            },
                        ],
                    },
                }
            ],
        },
    }


def _fence_role() -> dict[str, object]:
    return {
        "Type": "AWS::IAM::Role",
        "DeletionPolicy": "Retain",
        "UpdateReplacePolicy": "Retain",
        "Properties": {
            "RoleName": FENCE_ROLE_NAME,
            "Description": (
                "CloudFormation service role for the exact retained model bucket policy"
            ),
            "AssumeRolePolicyDocument": _cloudformation_trust_policy(),
            "Policies": [
                {
                    "PolicyName": "exact-model-bucket-policy",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Sid": "ReadWriteExactModelBucketPolicy",
                                "Effect": "Allow",
                                "Action": [
                                    "s3:GetBucketPolicy",
                                    "s3:PutBucketPolicy",
                                ],
                                "Resource": {"Fn::GetAtt": ["ModelBucket", "Arn"]},
                            }
                        ],
                    },
                }
            ],
        },
    }


def _foundation_resources() -> dict[str, object]:
    return {
        "CampaignKmsKey": _campaign_key(),
        "H1gLedger": _ledger(),
        "H1gCloudFormationDeploymentRole": _deployment_role(),
        "H1gFenceServiceRole": _fence_role(),
    }


def compose_retained_foundation_template(
    original_template: Mapping[str, object],
) -> dict[str, object]:
    """Add the reviewed Phase-1 resources to an exact live Original template."""

    if type(original_template) is not dict:
        raise FoundationError("live original template must be one exact object")
    canonical_json_bytes(original_template)
    resources = original_template.get("Resources")
    outputs = original_template.get("Outputs")
    if type(resources) is not dict or type(outputs) is not dict:
        raise FoundationError(
            "live original template requires Resources and Outputs objects"
        )
    for logical_id, resource_type in _REQUIRED_ORIGINAL_RESOURCE_TYPES.items():
        resource = resources.get(logical_id)
        if type(resource) is not dict or resource.get("Type") != resource_type:
            raise FoundationError(
                f"live original template is missing exact {logical_id}"
            )
    gateway_endpoints = [
        logical_id
        for logical_id, resource in resources.items()
        if type(resource) is dict
        and resource.get("Type") == "AWS::EC2::VPCEndpoint"
        and type(resource.get("Properties")) is dict
        and resource["Properties"].get("VpcEndpointType") == "Gateway"
    ]
    if gateway_endpoints != ["S3GatewayEndpoint"]:
        raise FoundationError(
            "existing S3GatewayEndpoint must remain the sole gateway endpoint"
        )
    conflicts = set(resources).intersection(_FOUNDATION_RESOURCE_TYPES)
    if conflicts:
        raise FoundationError(
            "retained foundation logical IDs already exist: "
            + ",".join(sorted(conflicts))
        )
    output_conflicts = set(outputs).intersection(_NEW_OUTPUTS)
    if output_conflicts:
        raise FoundationError(
            "retained foundation output IDs already exist: "
            + ",".join(sorted(output_conflicts))
        )
    for output_id, output in outputs.items():
        if type(output) is not dict:
            raise FoundationError(f"live output {output_id} is malformed")
        export = output.get("Export")
        if export is None:
            continue
        if type(export) is not dict:
            raise FoundationError(f"live output {output_id} export is malformed")
        export_name = export.get("Name")
        if (
            type(export_name) is str
            and export_name in set(_REQUIRED_EXPORTS.values())
            and output_id
            not in {
                key
                for key, expected in _REQUIRED_EXPORTS.items()
                if expected == export_name
            }
        ):
            raise FoundationError(
                f"required retained export is already owned by {output_id}"
            )
    expected_output_values = {
        "VpcId": {"Ref": "Vpc"},
        "SubnetId": {"Ref": "PublicSubnet"},
    }
    for output_id, expected_value in expected_output_values.items():
        output = outputs.get(output_id)
        if (
            type(output) is not dict
            or output.get("Value") != expected_value
            or (
                "Export" in output
                and output["Export"] != {"Name": _REQUIRED_EXPORTS[output_id]}
            )
        ):
            raise FoundationError(
                f"live output {output_id} cannot bind the required export"
            )

    result = deepcopy(original_template)
    result_resources = result["Resources"]
    result_outputs = result["Outputs"]
    result_resources.update(_foundation_resources())
    result_outputs["VpcId"]["Export"] = {"Name": _REQUIRED_EXPORTS["VpcId"]}
    result_outputs["SubnetId"]["Export"] = {"Name": _REQUIRED_EXPORTS["SubnetId"]}
    result_outputs["CampaignKmsKeyArn"] = {
        "Value": {"Fn::GetAtt": ["CampaignKmsKey", "Arn"]},
        "Export": {"Name": _REQUIRED_EXPORTS["CampaignKmsKeyArn"]},
    }
    result_outputs["H1gLedgerArn"] = {
        "Value": {"Fn::GetAtt": ["H1gLedger", "Arn"]},
        "Export": {"Name": _REQUIRED_EXPORTS["H1gLedgerArn"]},
    }
    canonical_json_bytes(result)
    return result


def _is_safe_launch_template_refresh(change: object) -> bool:
    if type(change) is not dict or change.get("Type") != "Resource":
        return False
    resource_change = change.get("ResourceChange")
    if (
        type(resource_change) is not dict
        or resource_change.get("Action") != "Modify"
        or resource_change.get("LogicalResourceId") != "GpuLaunchTemplate"
        or resource_change.get("ResourceType") != "AWS::EC2::LaunchTemplate"
        or resource_change.get("Replacement") != "False"
    ):
        return False
    details = resource_change.get("Details")
    if type(details) is not list or len(details) != 1 or type(details[0]) is not dict:
        return False
    detail = details[0]
    target = detail.get("Target")
    if (
        type(target) is not dict
        or set(target)
        != {
            "AfterValue",
            "Attribute",
            "AttributeChangeType",
            "BeforeValue",
            "Name",
            "Path",
            "RequiresRecreation",
        }
        or target.get("Attribute") != "Properties"
        or target.get("Name") != "LaunchTemplateData"
        or target.get("Path") != "/Properties/LaunchTemplateData"
        or target.get("RequiresRecreation") != "Never"
        or target.get("AttributeChangeType") != "Modify"
        or detail.get("Evaluation") != "Static"
        or detail.get("ChangeSource") != "DirectModification"
        or set(detail) != {"Target", "Evaluation", "ChangeSource"}
    ):
        return False
    before = target.get("BeforeValue")
    after = target.get("AfterValue")
    return (
        type(before) is str
        and type(after) is str
        and before.startswith("(Truncated-Signature):")
        and after.startswith("(Truncated-Signature):")
        and before != after
    )


def validate_foundation_change_set(
    change_set: Mapping[str, object],
    *,
    stack_id: str,
    change_set_name: str,
    expected_parameters: list[dict[str, object]],
) -> Mapping[str, object]:
    """Accept the reviewed additions and only the exact mutable-AMI refresh."""

    if type(change_set) is not dict:
        raise FoundationError("change set response must be one exact object")
    if (
        change_set.get("StackId") != stack_id
        or change_set.get("StackName") != STACK_NAME
        or change_set.get("ChangeSetName") != change_set_name
        or change_set.get("Status") != "CREATE_COMPLETE"
        or change_set.get("ExecutionStatus") != "AVAILABLE"
        or change_set.get("Capabilities") != ["CAPABILITY_NAMED_IAM"]
    ):
        raise FoundationError("change set identity or availability drifted")
    change_set_id = change_set.get("ChangeSetId")
    if type(change_set_id) is not str or not change_set_id.startswith(
        f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:changeSet/{change_set_name}/"
    ):
        raise FoundationError("change set ARN is not exact")
    changes = change_set.get("Changes")
    if type(changes) is not list or len(changes) not in {
        len(_FOUNDATION_RESOURCE_TYPES),
        len(_FOUNDATION_RESOURCE_TYPES) + 1,
    }:
        raise FoundationError("change set does not contain the exact reviewed delta")
    additions: list[object] = []
    refreshes: list[object] = []
    for change in changes:
        if _is_safe_launch_template_refresh(change):
            refreshes.append(change)
        else:
            additions.append(change)
    if len(refreshes) != len(changes) - len(_FOUNDATION_RESOURCE_TYPES):
        raise FoundationError("change set contains an unsafe resource mutation")
    if not _parameter_readbacks_match(
        change_set.get("Parameters"),
        expected_parameters,
        allow_gpu_resolution_drift=len(refreshes) == 1,
    ):
        raise FoundationError(
            "change set parameter readback does not preserve live parameters"
        )

    observed: dict[str, str] = {}
    for change in additions:
        if type(change) is not dict or change.get("Type") != "Resource":
            raise FoundationError("change set contains a non-resource change")
        resource_change = change.get("ResourceChange")
        if (
            type(resource_change) is not dict
            or resource_change.get("Action") != "Add"
            or resource_change.get("Replacement") not in {None, "False"}
            or type(resource_change.get("LogicalResourceId")) is not str
            or type(resource_change.get("ResourceType")) is not str
        ):
            raise FoundationError(
                "change set contains a replacement, deletion, or unsafe modification"
            )
        logical_id = resource_change["LogicalResourceId"]
        if logical_id in observed:
            raise FoundationError("change set repeats one logical resource")
        observed[logical_id] = resource_change["ResourceType"]
    if observed != _FOUNDATION_RESOURCE_TYPES:
        raise FoundationError("change set additions are not the reviewed set")
    return deepcopy(change_set)
