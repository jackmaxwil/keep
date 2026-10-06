"""Strict production adapters for disabled support and its live runtime.

The module accepts only closed, identity-bound requests.  It renders support
bytes locally from authenticated inputs, and all AWS-derived identity is
collected from fresh, bounded reads.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Mapping
from urllib.parse import unquote

from .canonical import canonical_json_bytes, canonical_sha256
from .support_plane import (
    SupportRuntimeIdentity,
    build_support_precreate_plane,
    build_support_runtime_identity,
    commit_support_runtime_identity,
    load_support_runtime_identity,
    parse_support_runtime_identity,
    publish_support_runtime_identity,
    require_live_support_runtime_identity,
    support_price_card_from_mapping,
)
from .task13_migration_adapter import (
    StackMigrationOperation7EvidenceV2,
    StackMigrationTransferCheckpointV2,
    build_seeded_migration_runtime,
    parse_stack_migration_operation_7_evidence_v2,
)
from .task13_production_operations import ProductionServices
from .task13_staged_deployment import (
    DisabledSupportDeploymentEvidence,
    parse_disabled_support_deployment_evidence,
)
from .task13_support_input_materialization import (
    SupportBuildInputs,
    support_build_inputs_from_mapping,
    support_build_inputs_projection,
    support_build_inputs_identity,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
MODEL_BUCKET_NAME = f"keep-glm52-models-{ACCOUNT_ID}-{REGION}"
SUPPORT_STACK_NAME = "keep-glm52-h1g-support"
LEDGER_TABLE_NAME = "keep-glm52-h1g-ledger-v1"
_SUPPORT_FUNCTION_LOGICAL_ID = "FenceExecutorFunction"
_SUPPORT_VERSION_LOGICAL_ID = "FenceExecutorVersion"
_SUPPORT_ROLE_LOGICAL_ID = "FenceExecutorRole"
_SUPPORT_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-fence-executor"
_MIGRATION_SERVICE_ROLE_ARN = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/"
    "keep-glm52-h1g-cloudformation-deployment"
)
_STACK_TAGS = (
    {"Key": "Project", "Value": "KEEP"},
    {"Key": "Campaign", "Value": "GLM-5.2"},
    {"Key": "RunId", "Value": RUN_ID},
    {"Key": "Environment", "Value": "production"},
    {"Key": "ManagedBy", "Value": "CloudFormation"},
    {"Key": "Authority", "Value": "H1g"},
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_KMS_ARN = re.compile(
    rf"arn:aws:kms:{REGION}:{ACCOUNT_ID}:key/[A-Za-z0-9-]+\Z"
)
_STACK_ID = re.compile(
    rf"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
    rf"{re.escape(SUPPORT_STACK_NAME)}/[A-Za-z0-9-]+\Z"
)
_VERSION_ARN = re.compile(
    rf"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:"
    r"keep-glm52-h1g-fence-executor:[1-9][0-9]*\Z"
)
_ROLE_ID = re.compile(r"AROA[A-Z0-9]{16,}\Z")
_LAUNCH_ACTIONS = frozenset(
    {
        "ec2:RunInstances",
        "ec2:StartInstances",
        "ec2:RequestSpotInstances",
        "ec2:CreateFleet",
        "ec2:CreateLaunchTemplate",
        "ec2:CreateLaunchTemplateVersion",
        "ec2:PurchaseCapacityBlock",
        "autoscaling:CreateAutoScalingGroup",
        "autoscaling:SetDesiredCapacity",
    }
)
_LAUNCH_RESOURCE_TYPES = frozenset(
    {
        "AWS::AutoScaling::AutoScalingGroup",
        "AWS::Batch::ComputeEnvironment",
        "AWS::EC2::EC2Fleet",
        "AWS::EC2::Fleet",
        "AWS::EC2::LaunchTemplate",
        "AWS::EC2::SpotFleet",
    }
)
_CLOUDTRAIL_LAUNCH_EVENTS = (
    "RunInstances",
    "StartInstances",
    "RequestSpotInstances",
    "PurchaseCapacityBlock",
)
_LEDGER_PREFIXES = (
    "WORKER#",
    "EC2_LAUNCH#",
    "SOURCE_ACTION#",
    "LAUNCH_AUTHORITY#",
)


class SupportRuntimeTransitionError(ValueError):
    """A closed request or authenticated production readback failed."""


@dataclass(frozen=True)
class DisabledSupportDeploymentRequest:
    activation_id: str
    generation: int
    bootstrap_manifest_coordinate: Mapping[str, object]
    prepare_entry_identity_sha256: str
    support_build_inputs_identity_sha256: str
    support_build_inputs: SupportBuildInputs
    price_card: Mapping[str, object]
    kms_key_arn: str


@dataclass(frozen=True)
class OperationSevenCompletionRequest:
    activation_id: str
    generation: int
    migration_seed: Mapping[str, object]
    checkpoint_identity_sha256: str
    prepare_execution_identity_sha256: str
    disabled_support_identity_sha256: str


@dataclass(frozen=True)
class SupportRuntimeCommitRequest:
    activation_id: str
    generation: int
    bootstrap_manifest_coordinate: Mapping[str, object]
    operation_7_identity_sha256: str
    disabled_support_identity_sha256: str
    expected_contract: Mapping[str, object]
    kms_key_arn: str
    ledger_table_name: str


@dataclass(frozen=True)
class NoLaunchObservationRequest:
    activation_id: str
    generation: int
    runtime_identity_sha256: str
    ledger_table_name: str
    evidence_window_start: datetime
    evidence_window_end: datetime


def _object(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise SupportRuntimeTransitionError(label + " must be one object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise SupportRuntimeTransitionError(
            f"{label} schema mismatch: missing={missing}, unknown={unknown}"
        )
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SupportRuntimeTransitionError(label + " is not canonical JSON") from exc


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise SupportRuntimeTransitionError(label + " is not one lowercase SHA-256")
    return value


def _activation_generation(activation: object, generation: object) -> tuple[str, int]:
    if (
        type(activation) is not str
        or _ACTIVATION.fullmatch(activation) is None
        or type(generation) is not int
        or generation <= 0
    ):
        raise SupportRuntimeTransitionError("activation/generation coordinate drifted")
    return activation, generation


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise SupportRuntimeTransitionError(label + " must be one object")
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SupportRuntimeTransitionError(label + " is not canonical JSON") from exc


def _manifest_coordinate(
    value: object, *, activation_id: str, generation: int
) -> dict[str, object]:
    from .fence_artifacts import parse_artifact_coordinate

    try:
        coordinate = parse_artifact_coordinate(value)
    except (TypeError, ValueError) as exc:
        raise SupportRuntimeTransitionError(
            "bootstrap manifest coordinate is not exact"
        ) from exc
    expected_key = (
        f"campaigns/{RUN_ID}/authorities/fence/manifests/{activation_id}/"
        f"{generation:08d}/FENCE_BOOTSTRAP_MANIFEST.json"
    )
    if coordinate.bucket != MODEL_BUCKET_NAME or coordinate.key != expected_key:
        raise SupportRuntimeTransitionError("bootstrap manifest coordinate is foreign")
    return coordinate.to_dict()


def parse_disabled_support_deployment_request(
    value: object,
) -> DisabledSupportDeploymentRequest:
    fields = frozenset(
        {
            "schema_version",
            "record_type",
            "activation_id",
            "generation",
            "bootstrap_manifest_coordinate",
            "prepare_entry_identity_sha256",
            "support_build_inputs_identity_sha256",
            "price_card",
            "support_build_inputs",
            "kms_key_arn",
        }
    )
    body = _object(value, fields, "disabled support deployment request")
    if (
        body["schema_version"] != 1
        or body["record_type"] != "glm52_disabled_support_deployment_request_v1"
    ):
        raise SupportRuntimeTransitionError("disabled support request identity drifted")
    activation, generation = _activation_generation(
        body["activation_id"], body["generation"]
    )
    coordinate = _manifest_coordinate(
        body["bootstrap_manifest_coordinate"],
        activation_id=activation,
        generation=generation,
    )
    kms = body["kms_key_arn"]
    if type(kms) is not str or _KMS_ARN.fullmatch(kms) is None:
        raise SupportRuntimeTransitionError("disabled support KMS key is not exact")
    support_inputs = support_build_inputs_from_mapping(
        body["support_build_inputs"]
    )
    return DisabledSupportDeploymentRequest(
        activation_id=activation,
        generation=generation,
        bootstrap_manifest_coordinate=coordinate,
        prepare_entry_identity_sha256=_sha(
            body["prepare_entry_identity_sha256"], "PREPARE entry identity"
        ),
        support_build_inputs=support_inputs,
        support_build_inputs_identity_sha256=_sha(
            body["support_build_inputs_identity_sha256"],
            "support build inputs identity",
        ),
        price_card=_mapping(body["price_card"], "support price card"),
        kms_key_arn=kms,
    )


def parse_operation_seven_completion_request(
    value: object,
) -> OperationSevenCompletionRequest:
    fields = frozenset(
        {
            "schema_version",
            "record_type",
            "activation_id",
            "generation",
            "migration_seed",
            "checkpoint_identity_sha256",
            "prepare_execution_identity_sha256",
            "disabled_support_identity_sha256",
        }
    )
    body = _object(value, fields, "operation seven completion request")
    if (
        body["schema_version"] != 1
        or body["record_type"] != "glm52_support_replacement_request_v1"
    ):
        raise SupportRuntimeTransitionError("operation seven request identity drifted")
    activation, generation = _activation_generation(
        body["activation_id"], body["generation"]
    )
    return OperationSevenCompletionRequest(
        activation_id=activation,
        generation=generation,
        migration_seed=_mapping(body["migration_seed"], "migration seed"),
        checkpoint_identity_sha256=_sha(
            body["checkpoint_identity_sha256"], "checkpoint identity"
        ),
        prepare_execution_identity_sha256=_sha(
            body["prepare_execution_identity_sha256"], "PREPARE execution identity"
        ),
        disabled_support_identity_sha256=_sha(
            body["disabled_support_identity_sha256"], "disabled support identity"
        ),
    )


def parse_support_runtime_commit_request(value: object) -> SupportRuntimeCommitRequest:
    fields = frozenset(
        {
            "schema_version",
            "record_type",
            "activation_id",
            "generation",
            "bootstrap_manifest_coordinate",
            "operation_7_identity_sha256",
            "disabled_support_identity_sha256",
            "expected_contract",
            "kms_key_arn",
            "ledger_table_name",
        }
    )
    body = _object(value, fields, "support runtime commit request")
    if (
        body["schema_version"] != 1
        or body["record_type"] != "glm52_support_runtime_commit_request_v1"
    ):
        raise SupportRuntimeTransitionError("support runtime request identity drifted")
    activation, generation = _activation_generation(
        body["activation_id"], body["generation"]
    )
    coordinate = _manifest_coordinate(
        body["bootstrap_manifest_coordinate"],
        activation_id=activation,
        generation=generation,
    )
    kms = body["kms_key_arn"]
    table = body["ledger_table_name"]
    if type(kms) is not str or _KMS_ARN.fullmatch(kms) is None:
        raise SupportRuntimeTransitionError("support runtime KMS key is not exact")
    if table != LEDGER_TABLE_NAME:
        raise SupportRuntimeTransitionError("support runtime ledger table drifted")
    return SupportRuntimeCommitRequest(
        activation_id=activation,
        generation=generation,
        bootstrap_manifest_coordinate=coordinate,
        operation_7_identity_sha256=_sha(
            body["operation_7_identity_sha256"], "operation seven identity"
        ),
        disabled_support_identity_sha256=_sha(
            body["disabled_support_identity_sha256"], "disabled support identity"
        ),
        expected_contract=_mapping(
            body["expected_contract"], "support runtime expected contract"
        ),
        kms_key_arn=kms,
        ledger_table_name=table,
    )


def _timestamp(value: object, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise SupportRuntimeTransitionError(label + " is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SupportRuntimeTransitionError(label + " is not canonical UTC") from exc
    if parsed.tzinfo != timezone.utc or parsed.isoformat().replace("+00:00", "Z") != value:
        raise SupportRuntimeTransitionError(label + " is not canonical UTC")
    return parsed


def parse_no_launch_observation_request(value: object) -> NoLaunchObservationRequest:
    fields = frozenset(
        {
            "schema_version",
            "record_type",
            "activation_id",
            "generation",
            "runtime_identity_sha256",
            "ledger_table_name",
            "evidence_window_start",
            "evidence_window_end",
        }
    )
    body = _object(value, fields, "no-launch observation request")
    if (
        body["schema_version"] != 1
        or body["record_type"] != "glm52_no_launch_observation_request_v1"
    ):
        raise SupportRuntimeTransitionError("no-launch request identity drifted")
    activation, generation = _activation_generation(
        body["activation_id"], body["generation"]
    )
    start = _timestamp(body["evidence_window_start"], "observation start")
    end = _timestamp(body["evidence_window_end"], "observation end")
    if start >= end or (end - start).total_seconds() > 3600:
        raise SupportRuntimeTransitionError("no-launch observation window is unbounded")
    if body["ledger_table_name"] != LEDGER_TABLE_NAME:
        raise SupportRuntimeTransitionError("no-launch ledger table drifted")
    return NoLaunchObservationRequest(
        activation_id=activation,
        generation=generation,
        runtime_identity_sha256=_sha(
            body["runtime_identity_sha256"], "runtime identity"
        ),
        ledger_table_name=LEDGER_TABLE_NAME,
        evidence_window_start=start,
        evidence_window_end=end,
    )


def _metadata(value: object, operation: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise SupportRuntimeTransitionError(operation + " returned no object")
    metadata = value.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") not in {200, 201}
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        raise SupportRuntimeTransitionError(
            operation + " lacks authenticated zero-retry success"
        )
    return value


def _call(client: object, name: str, operation: str, **request: object) -> Mapping[str, object]:
    method = getattr(client, name, None)
    if not callable(method):
        raise SupportRuntimeTransitionError(operation + " method is unavailable")
    return _metadata(method(**request), operation)


def _error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    detail = response.get("Error") if type(response) is dict else None
    code = detail.get("Code") if type(detail) is dict else None
    return code if type(code) is str else None


def _authenticated_absence(error: BaseException, operation: str) -> bool:
    if _error_code(error) not in {"ResourceNotFoundException", "ResourceNotFound"}:
        return False
    response = getattr(error, "response", None)
    metadata = response.get("ResponseMetadata") if type(response) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") not in {400, 404}
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        raise SupportRuntimeTransitionError(
            operation + " absence is not authenticated"
        ) from error
    return True


def _static_launch_inert(template: Mapping[str, object]) -> Mapping[str, object]:
    parameters = template.get("Parameters", {})
    resources = template.get("Resources")
    if type(parameters) is not dict or parameters or type(resources) is not dict or not resources:
        raise SupportRuntimeTransitionError("support template activation surface drifted")
    resource_hits: list[dict[str, str]] = []
    action_hits: set[str] = set()
    p5_paths: list[str] = []

    def walk(value: object, path: tuple[str, ...]) -> None:
        if value == "p5.48xlarge":
            p5_paths.append(".".join(path))
        if type(value) is dict:
            actions = value.get("Action")
            candidates: list[str]
            if type(actions) is str:
                candidates = [actions]
            elif type(actions) is list and all(type(item) is str for item in actions):
                candidates = actions
            elif actions is None:
                candidates = []
            else:
                raise SupportRuntimeTransitionError("support IAM action shape drifted")
            action_hits.update(
                item for item in candidates if item in _LAUNCH_ACTIONS or item in {"*", "ec2:*"}
            )
            for key, child in value.items():
                if type(key) is not str or not key:
                    raise SupportRuntimeTransitionError("support template key drifted")
                walk(child, (*path, key))
        elif type(value) is list:
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    for logical_id, resource in resources.items():
        if type(logical_id) is not str or type(resource) is not dict:
            raise SupportRuntimeTransitionError("support resource graph drifted")
        resource_type = resource.get("Type")
        if type(resource_type) is not str:
            raise SupportRuntimeTransitionError("support resource type drifted")
        if resource_type in _LAUNCH_RESOURCE_TYPES:
            resource_hits.append({"logical_id": logical_id, "resource_type": resource_type})
        if resource_type == "AWS::EC2::Instance":
            properties = resource.get("Properties")
            if type(properties) is not dict or properties.get("InstanceType") != "c6a.xlarge":
                resource_hits.append({"logical_id": logical_id, "resource_type": resource_type})
        walk(resource, ("Resources", logical_id))
    proof = {
        "launch_capable_actions": sorted(action_hits),
        "launch_capable_resources": sorted(resource_hits, key=lambda row: (row["logical_id"], row["resource_type"])),
        "p5_instance_paths": sorted(p5_paths),
        "template_parameters": sorted(parameters),
        "worker_activation_allowed": False,
        "source_action_allowed": False,
    }
    if action_hits or resource_hits or p5_paths:
        raise SupportRuntimeTransitionError("support template contains launch authority")
    return proof


def _disabled_profile_identity() -> str:
    return canonical_sha256(
        {
            "resource_policy_sha256": canonical_sha256({"absent": True}),
            "event_source_state_sha256": canonical_sha256([]),
            "function_url_state_sha256": canonical_sha256({"absent": True}),
        }
    )


def _disabled_support_key(activation_id: str, generation: int) -> str:
    return (
        f"campaigns/{RUN_ID}/authorities/support/templates/{activation_id}/"
        f"{generation:08d}/DISABLED_SUPPORT_TEMPLATE.json"
    )


def _build_disabled_evidence(
    *,
    parsed: DisabledSupportDeploymentRequest,
    template_raw: bytes,
    template_sha: str,
    version_id: str,
    no_launch_sha: str,
) -> DisabledSupportDeploymentEvidence:
    coordinate = {
        "artifact_kind": "DISABLED_SUPPORT_TEMPLATE",
        "bucket": MODEL_BUCKET_NAME,
        "key": _disabled_support_key(parsed.activation_id, parsed.generation),
        "version_id": version_id,
        "file_sha256": hashlib.sha256(template_raw).hexdigest(),
        "body_sha256": template_sha,
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_disabled_support_deployment_v1",
        "profile_record_type": "disabled_support_profile_v1",
        "bootstrap_manifest_coordinate": dict(parsed.bootstrap_manifest_coordinate),
        "prepare_entry_identity_sha256": parsed.prepare_entry_identity_sha256,
        "support_build_inputs_identity_sha256": parsed.support_build_inputs_identity_sha256,
        "disabled_support_profile_sha256": _disabled_profile_identity(),
        "support_template_coordinate": coordinate,
        "support_template_sha256": template_sha,
        "no_launch_evidence_sha256": no_launch_sha,
        "worker_activation_allowed": False,
        "source_action_allowed": False,
    }
    body["canonical_identity_sha256"] = canonical_sha256(body)
    return parse_disabled_support_deployment_evidence(body)


def _template_readback(
    value: object,
    label: str,
) -> dict[str, object]:
    if type(value) is str:
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SupportRuntimeTransitionError(
                label + " is not JSON"
            ) from exc
    return _mapping(value, label)


def _require_inert_support_anchor(
    *,
    checkpoint: StackMigrationTransferCheckpointV2,
    services: ProductionServices,
) -> None:
    if (
        type(checkpoint) is not StackMigrationTransferCheckpointV2
        or checkpoint.operation_7_status != "NOT_SUBMITTED"
        or checkpoint.support_state != "INERT_ANCHOR"
    ):
        raise SupportRuntimeTransitionError(
            "disabled support requires the exact pre-operation-seven checkpoint"
        )
    response = _call(
        services.cloudformation,
        "describe_stacks",
        "DescribeStacks(SupportAnchor)",
        StackName=checkpoint.support_stack_id,
    )
    stacks = response.get("Stacks")
    if (
        type(stacks) is not list
        or len(stacks) != 1
        or type(stacks[0]) is not dict
    ):
        raise SupportRuntimeTransitionError(
            "support anchor stack is absent or ambiguous"
        )
    stack = stacks[0]
    raw_tags = stack.get("Tags")
    if (
        stack.get("StackId") != checkpoint.support_stack_id
        or stack.get("StackName") != SUPPORT_STACK_NAME
        or stack.get("StackStatus") != "CREATE_COMPLETE"
        or stack.get("RoleARN") != checkpoint.migration_service_role.role_arn
        or stack.get("EnableTerminationProtection") is not True
        or type(raw_tags) is not list
        or any(
            type(row) is not dict
            or set(row) != {"Key", "Value"}
            for row in raw_tags
        )
        or len({row["Key"] for row in raw_tags}) != len(raw_tags)
        or {row["Key"]: row["Value"] for row in raw_tags}
        != {row["Key"]: row["Value"] for row in _STACK_TAGS}
    ):
        raise SupportRuntimeTransitionError(
            "support anchor stack identity drifted"
        )
    for stage in ("Original", "Processed"):
        template_response = _call(
            services.cloudformation,
            "get_template",
            f"GetTemplate{stage}(SupportAnchor)",
            StackName=checkpoint.support_stack_id,
            TemplateStage=stage,
        )
        template = _template_readback(
            template_response.get("TemplateBody"),
            f"support anchor {stage} template",
        )
        resources = template.get("Resources")
        if (
            canonical_sha256(template)
            != checkpoint.support_prestate_template_sha256
            or type(resources) is not dict
            or set(resources) != {"ContainerAnchor"}
            or resources["ContainerAnchor"]
            != {"Type": "AWS::CloudFormation::WaitConditionHandle"}
            or template.get("Parameters", {}) != {}
            or template.get("Outputs", {}) != {}
        ):
            raise SupportRuntimeTransitionError(
                "support anchor template identity drifted"
            )


def deploy_disabled_support_v2(
    *,
    request: object,
    checkpoint: StackMigrationTransferCheckpointV2,
    support_inputs: SupportBuildInputs,
    services: ProductionServices,
) -> DisabledSupportDeploymentEvidence:
    """Render, prove inert, publish/adopt, while the support anchor stays inert."""
    if (
        type(checkpoint) is not StackMigrationTransferCheckpointV2
        or type(support_inputs) is not SupportBuildInputs
        or type(services) is not ProductionServices
    ):
        raise SupportRuntimeTransitionError(
            "disabled support binding inputs are not exact"
        )
    parsed = parse_disabled_support_deployment_request(request)
    if (
        support_build_inputs_projection(support_inputs)
        != support_build_inputs_projection(parsed.support_build_inputs)
    ):
        raise SupportRuntimeTransitionError(
            "disabled support staged snapshot drifted"
        )
    if (
        parsed.activation_id != support_inputs.activation_id
        or parsed.bootstrap_manifest_coordinate
        != dict(support_inputs.bootstrap_manifest_coordinate)
        or parsed.prepare_entry_identity_sha256
        != support_inputs.prepare_entry_identity_sha256
        or parsed.support_build_inputs_identity_sha256
        != support_build_inputs_identity(support_inputs)
    ):
        raise SupportRuntimeTransitionError(
            "disabled support prerequisite identity drifted"
        )
    try:
        price_card = support_price_card_from_mapping(parsed.price_card, inputs=support_inputs)
        bundle = build_support_precreate_plane(inputs=support_inputs, price_card=price_card)
    except (TypeError, ValueError) as exc:
        raise SupportRuntimeTransitionError("disabled support render failed closed") from exc
    template = dict(bundle.support_template)
    proof = _static_launch_inert(template)
    template_raw = canonical_json_bytes(template) + b"\n"
    template_sha = hashlib.sha256(template_raw[:-1]).hexdigest()
    no_launch_sha = canonical_sha256(proof)
    from .task13_fixed_artifacts import (
        Task13FixedArtifactServices,
        publish_fixed_key_bytes,
    )

    version_id = publish_fixed_key_bytes(
        services=Task13FixedArtifactServices(
            sts=services.sts,
            s3=services.s3,
            total_max_attempts=services.total_max_attempts,
        ),
        bucket=MODEL_BUCKET_NAME,
        key=_disabled_support_key(parsed.activation_id, parsed.generation),
        raw=template_raw,
        record_type="glm52_disabled_support_template_v1",
        sse_kms_key_id=parsed.kms_key_arn,
    )
    _require_inert_support_anchor(
        checkpoint=checkpoint,
        services=services,
    )
    return _build_disabled_evidence(
        parsed=parsed,
        template_raw=template_raw,
        template_sha=template_sha,
        version_id=version_id,
        no_launch_sha=no_launch_sha,
    )


def _policy_document(value: object, label: str) -> object:
    if type(value) is str:
        try:
            value = json.loads(unquote(value))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SupportRuntimeTransitionError(label + " is not JSON") from exc
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SupportRuntimeTransitionError(label + " is not canonical") from exc


def _permission_inventory(iam: object, role_name: str) -> Mapping[str, object]:
    inline = _call(
        iam,
        "list_role_policies",
        "ListRolePolicies",
        RoleName=role_name,
        MaxItems=100,
    )
    attached = _call(
        iam,
        "list_attached_role_policies",
        "ListAttachedRolePolicies",
        RoleName=role_name,
        MaxItems=100,
    )
    names = inline.get("PolicyNames")
    rows = attached.get("AttachedPolicies")
    if (
        type(names) is not list
        or any(type(name) is not str or not name for name in names)
        or len(names) != len(set(names))
        or inline.get("IsTruncated") is not False
        or type(rows) is not list
        or attached.get("IsTruncated") is not False
        or len(rows) > 100
    ):
        raise SupportRuntimeTransitionError("role permission inventory is unbounded")
    inventory: dict[str, object] = {"inline": [], "attached": []}
    inline_result = inventory["inline"]
    assert type(inline_result) is list
    for name in sorted(names):
        response = _call(
            iam,
            "get_role_policy",
            "GetRolePolicy",
            RoleName=role_name,
            PolicyName=name,
        )
        if response.get("RoleName") != role_name or response.get("PolicyName") != name:
            raise SupportRuntimeTransitionError("inline role policy identity drifted")
        inline_result.append(
            {"name": name, "document": _policy_document(response.get("PolicyDocument"), "inline role policy")}
        )
    attached_result = inventory["attached"]
    assert type(attached_result) is list
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: str(item.get("PolicyArn")) if type(item) is dict else ""):
        if type(row) is not dict or set(row) != {"PolicyName", "PolicyArn"}:
            raise SupportRuntimeTransitionError("attached role policy row drifted")
        arn = row["PolicyArn"]
        if type(arn) is not str or arn in seen:
            raise SupportRuntimeTransitionError("attached role policy identity drifted")
        seen.add(arn)
        policy_response = _call(iam, "get_policy", "GetPolicy", PolicyArn=arn)
        policy = policy_response.get("Policy")
        if type(policy) is not dict or policy.get("Arn") != arn:
            raise SupportRuntimeTransitionError("attached policy readback drifted")
        version = policy.get("DefaultVersionId")
        if type(version) is not str or not version:
            raise SupportRuntimeTransitionError("attached policy version drifted")
        version_response = _call(
            iam,
            "get_policy_version",
            "GetPolicyVersion",
            PolicyArn=arn,
            VersionId=version,
        )
        version_row = version_response.get("PolicyVersion")
        if type(version_row) is not dict or version_row.get("VersionId") != version:
            raise SupportRuntimeTransitionError("attached policy version readback drifted")
        attached_result.append(
            {
                "arn": arn,
                "version_id": version,
                "document": _policy_document(version_row.get("Document"), "attached role policy"),
            }
        )
    return inventory


def _fresh_role_evidence(services: ProductionServices, expected: object) -> object:
    from .cloudformation_stacks import MigrationRoleEvidenceV2

    if type(expected) is not MigrationRoleEvidenceV2:
        raise SupportRuntimeTransitionError("expected migration role evidence is not exact")
    role_name = expected.role_arn.rsplit("/", 1)[-1]
    response = _call(services.iam, "get_role", "GetRole", RoleName=role_name)
    role = response.get("Role")
    if (
        type(role) is not dict
        or role.get("Arn") != expected.role_arn
        or type(role.get("RoleId")) is not str
    ):
        raise SupportRuntimeTransitionError("migration role live identity drifted")
    observed = MigrationRoleEvidenceV2(
        role_arn=role["Arn"],
        role_id=role["RoleId"],
        trust_policy_sha256=canonical_sha256(
            _policy_document(role.get("AssumeRolePolicyDocument"), "role trust policy")
        ),
        permission_policy_sha256=canonical_sha256(
            _permission_inventory(services.iam, role_name)
        ),
    )
    if observed != expected:
        raise SupportRuntimeTransitionError("migration role live hashes drifted")
    return observed


class _MigrationServiceClient:
    """Route migration CFN calls and its pinned S3 reads without guessing."""

    def __init__(self, *, cloudformation: object, s3: object) -> None:
        self._cloudformation = cloudformation
        self._s3 = s3

    def __getattr__(self, name: str) -> object:
        owner = (
            self._s3
            if name in {"get_bucket_policy", "get_object"}
            else self._cloudformation
        )
        value = getattr(owner, name, None)
        if not callable(value):
            raise AttributeError(name)
        return value


def complete_operation_7_v2(
    *,
    request: object,
    checkpoint: StackMigrationTransferCheckpointV2,
    prepare_result: object,
    disabled_support: DisabledSupportDeploymentEvidence,
    services: ProductionServices,
) -> StackMigrationOperation7EvidenceV2:
    """Perform the sole support-anchor replacement through the public primitive."""
    from .fence_executor import FenceExecutionResult
    from .task13_migration_adapter import execute_stack_migration_operation_7_v2

    if (
        type(checkpoint) is not StackMigrationTransferCheckpointV2
        or type(prepare_result) is not FenceExecutionResult
        or type(disabled_support) is not DisabledSupportDeploymentEvidence
        or type(services) is not ProductionServices
    ):
        raise SupportRuntimeTransitionError("operation seven binding inputs are not exact")
    parsed = parse_operation_seven_completion_request(request)
    manifest = disabled_support.bootstrap_manifest_coordinate
    manifest_parts = manifest.key.split("/")
    if (
        parsed.checkpoint_identity_sha256 != checkpoint.canonical_identity_sha256
        or parsed.prepare_execution_identity_sha256 != prepare_result.canonical_identity_sha256
        or parsed.disabled_support_identity_sha256 != disabled_support.canonical_identity_sha256
        or len(manifest_parts) < 3
        or manifest_parts[-3] != parsed.activation_id
        or manifest_parts[-2] != f"{parsed.generation:08d}"
        or checkpoint.operation_7_status != "NOT_SUBMITTED"
    ):
        raise SupportRuntimeTransitionError("operation seven prerequisite ordering drifted")
    runtime = build_seeded_migration_runtime(
        cloudformation=_MigrationServiceClient(
            cloudformation=services.cloudformation,
            s3=services.s3,
        ),
        dynamodb=services.dynamodb,
        value=parsed.migration_seed,
    )
    raw_authority = parsed.migration_seed.get("authority")
    if (
        type(raw_authority) is not dict
        or runtime.seed.authority.action_identity_sha256
        != raw_authority.get("action_identity_sha256")
    ):
        raise SupportRuntimeTransitionError(
            "operation seven migration seed drifted"
        )
    result = execute_stack_migration_operation_7_v2(
        runtime=runtime,
        checkpoint=checkpoint,
        prepare_result=prepare_result,
        disabled_support=disabled_support,
        observed_api_caller=_fresh_role_evidence(services, checkpoint.api_caller),
        observed_migration_service_role=_fresh_role_evidence(
            services, checkpoint.migration_service_role
        ),
        observed_fence_service_role=_fresh_role_evidence(
            services, checkpoint.fence_service_role
        ),
    )
    if type(result) is not StackMigrationOperation7EvidenceV2:
        raise SupportRuntimeTransitionError("operation seven result type drifted")
    parsed_result = parse_stack_migration_operation_7_evidence_v2(result.to_dict())
    if (
        parsed_result.operation_7_status != "COMPLETE"
        or parsed_result.mutation_scope != "SUPPORT_REPLACEMENT_ONLY"
        or parsed_result.checkpoint_identity_sha256 != checkpoint.canonical_identity_sha256
        or parsed_result.disabled_support_identity_sha256
        != disabled_support.canonical_identity_sha256
    ):
        raise SupportRuntimeTransitionError("operation seven result scope drifted")
    return parsed_result


def read_operation_7_v2(
    *,
    request: object,
    checkpoint: StackMigrationTransferCheckpointV2,
    prepare_result: object,
    disabled_support: DisabledSupportDeploymentEvidence,
    services: ProductionServices,
) -> StackMigrationOperation7EvidenceV2:
    """Read the exact journaled support replacement without resending it."""
    from .fence_executor import FenceExecutionResult
    from .task13_migration_adapter import (
        read_stack_migration_operation_7_effect_v2,
    )

    if (
        type(checkpoint) is not StackMigrationTransferCheckpointV2
        or type(prepare_result) is not FenceExecutionResult
        or type(disabled_support) is not DisabledSupportDeploymentEvidence
        or type(services) is not ProductionServices
    ):
        raise SupportRuntimeTransitionError(
            "operation seven read inputs are not exact"
        )
    parsed = parse_operation_seven_completion_request(request)
    if (
        parsed.checkpoint_identity_sha256
        != checkpoint.canonical_identity_sha256
        or parsed.prepare_execution_identity_sha256
        != prepare_result.canonical_identity_sha256
        or parsed.disabled_support_identity_sha256
        != disabled_support.canonical_identity_sha256
    ):
        raise SupportRuntimeTransitionError(
            "operation seven read prerequisite drifted"
        )
    runtime = build_seeded_migration_runtime(
        cloudformation=_MigrationServiceClient(
            cloudformation=services.cloudformation,
            s3=services.s3,
        ),
        dynamodb=services.dynamodb,
        value=parsed.migration_seed,
    )
    result = read_stack_migration_operation_7_effect_v2(
        runtime=runtime,
        checkpoint=checkpoint,
        prepare_result=prepare_result,
        disabled_support=disabled_support,
        observed_api_caller=_fresh_role_evidence(
            services, checkpoint.api_caller
        ),
        observed_migration_service_role=_fresh_role_evidence(
            services, checkpoint.migration_service_role
        ),
        observed_fence_service_role=_fresh_role_evidence(
            services, checkpoint.fence_service_role
        ),
    )
    if type(result) is not StackMigrationOperation7EvidenceV2:
        raise SupportRuntimeTransitionError(
            "operation seven read result type drifted"
        )
    parsed_result = parse_stack_migration_operation_7_evidence_v2(
        result.to_dict()
    )
    if (
        parsed_result.operation_7_status != "COMPLETE"
        or parsed_result.mutation_scope != "SUPPORT_REPLACEMENT_ONLY"
        or parsed_result.checkpoint_identity_sha256
        != checkpoint.canonical_identity_sha256
        or parsed_result.disabled_support_identity_sha256
        != disabled_support.canonical_identity_sha256
    ):
        raise SupportRuntimeTransitionError(
            "operation seven read result scope drifted"
        )
    return parsed_result


def _stack_resource(services: ProductionServices, stack_id: str, logical_id: str) -> str:
    response = _call(
        services.cloudformation,
        "describe_stack_resource",
        "DescribeStackResource",
        StackName=stack_id,
        LogicalResourceId=logical_id,
    )
    detail = response.get("StackResourceDetail")
    if (
        type(detail) is not dict
        or detail.get("StackId") != stack_id
        or detail.get("LogicalResourceId") != logical_id
        or type(detail.get("PhysicalResourceId")) is not str
        or not detail["PhysicalResourceId"]
    ):
        raise SupportRuntimeTransitionError(logical_id + " physical identity drifted")
    return detail["PhysicalResourceId"]


def _fresh_runtime_readback(
    *, services: ProductionServices, support_stack_id: str
) -> tuple[Mapping[str, object], bool]:
    if type(services) is not ProductionServices or _STACK_ID.fullmatch(support_stack_id) is None:
        raise SupportRuntimeTransitionError("support runtime read coordinate drifted")
    stacks = _call(
        services.cloudformation,
        "describe_stacks",
        "DescribeStacks",
        StackName=support_stack_id,
    ).get("Stacks")
    if (
        type(stacks) is not list
        or len(stacks) != 1
        or type(stacks[0]) is not dict
        or stacks[0].get("StackId") != support_stack_id
        or stacks[0].get("StackName") != SUPPORT_STACK_NAME
        or stacks[0].get("StackStatus") != "UPDATE_COMPLETE"
        or stacks[0].get("EnableTerminationProtection") is not True
    ):
        raise SupportRuntimeTransitionError("support runtime stack readback drifted")
    template_response = _call(
        services.cloudformation,
        "get_template",
        "GetTemplate",
        StackName=support_stack_id,
        TemplateStage="Original",
    )
    template = template_response.get("TemplateBody")
    if type(template) is str:
        try:
            template = json.loads(template)
        except json.JSONDecodeError as exc:
            raise SupportRuntimeTransitionError("support template readback drifted") from exc
    template = _mapping(template, "support template readback")
    static = _static_launch_inert(template)
    function_name = _stack_resource(services, support_stack_id, _SUPPORT_FUNCTION_LOGICAL_ID)
    version_arn = _stack_resource(services, support_stack_id, _SUPPORT_VERSION_LOGICAL_ID)
    role_name = _stack_resource(services, support_stack_id, _SUPPORT_ROLE_LOGICAL_ID)
    if _VERSION_ARN.fullmatch(version_arn) is None or role_name != _SUPPORT_ROLE_ARN.rsplit("/", 1)[-1]:
        raise SupportRuntimeTransitionError("support runtime physical identities drifted")
    configuration = _call(
        services.lambda_client,
        "get_function_configuration",
        "GetFunctionConfiguration",
        FunctionName=version_arn,
    )
    role_arn = configuration.get("Role")
    if (
        configuration.get("FunctionName") != function_name
        or configuration.get("FunctionArn") != version_arn
        or role_arn != _SUPPORT_ROLE_ARN
        or configuration.get("State") != "Active"
        or configuration.get("LastUpdateStatus") != "Successful"
    ):
        raise SupportRuntimeTransitionError("support function live identity drifted")
    environment = configuration.get("Environment")
    variables = environment.get("Variables") if type(environment) is dict else None
    if type(variables) is not dict or set(variables) == set():
        raise SupportRuntimeTransitionError("support function environment is absent")
    role_response = _call(services.iam, "get_role", "GetRole", RoleName=role_name)
    role = role_response.get("Role")
    if (
        type(role) is not dict
        or role.get("Arn") != role_arn
        or type(role.get("RoleId")) is not str
        or _ROLE_ID.fullmatch(role["RoleId"]) is None
    ):
        raise SupportRuntimeTransitionError("support execution role live identity drifted")
    permissions = _permission_inventory(services.iam, role_name)
    try:
        policy_response = _call(
            services.lambda_client,
            "get_policy",
            "GetPolicy",
            FunctionName=version_arn,
        )
    except Exception as exc:
        if not _authenticated_absence(exc, "GetPolicy"):
            raise
        resource_policy: object = {"absent": True}
    else:
        policy = policy_response.get("Policy")
        resource_policy = _policy_document(policy, "Lambda resource policy")
    mappings_response = _call(
        services.lambda_client,
        "list_event_source_mappings",
        "ListEventSourceMappings",
        FunctionName=version_arn,
        MaxItems=16,
    )
    rows = mappings_response.get("EventSourceMappings")
    if type(rows) is not list or len(rows) > 16 or mappings_response.get("NextMarker") is not None:
        raise SupportRuntimeTransitionError("support event-source inventory is unbounded")
    mappings: list[dict[str, object]] = []
    for row in rows:
        if type(row) is not dict:
            raise SupportRuntimeTransitionError("support event-source row drifted")
        projection = {
            key: row.get(key)
            for key in ("UUID", "EventSourceArn", "FunctionArn", "State")
        }
        if (
            type(projection["UUID"]) is not str
            or type(projection["EventSourceArn"]) is not str
            or projection["FunctionArn"] != version_arn
            or type(projection["State"]) is not str
        ):
            raise SupportRuntimeTransitionError("support event-source identity drifted")
        mappings.append(projection)
    mappings.sort(key=lambda row: str(row["UUID"]))
    try:
        url_response = _call(
            services.lambda_client,
            "get_function_url_config",
            "GetFunctionUrlConfig",
            FunctionName=version_arn,
        )
    except Exception as exc:
        if not _authenticated_absence(exc, "GetFunctionUrlConfig"):
            raise
        function_url: object = {"absent": True}
    else:
        function_url = {
            key: url_response.get(key)
            for key in ("AuthType", "InvokeMode", "FunctionUrl")
        }
    try:
        code_sha = base64.b64decode(configuration.get("CodeSha256"), validate=True).hex()
    except (TypeError, ValueError) as exc:
        raise SupportRuntimeTransitionError("support function code identity drifted") from exc
    resource_policy_sha = canonical_sha256(resource_policy)
    event_source_sha = canonical_sha256(mappings)
    function_url_sha = canonical_sha256(function_url)
    disabled_profile_sha = canonical_sha256(
        {
            "resource_policy_sha256": resource_policy_sha,
            "event_source_state_sha256": event_source_sha,
            "function_url_state_sha256": function_url_sha,
        }
    )
    launch_present = bool(
        static["launch_capable_actions"]
        or static["launch_capable_resources"]
        or static["p5_instance_paths"]
        or resource_policy != {"absent": True}
        or mappings
        or function_url != {"absent": True}
    )
    return (
        {
            "function_version_arn": version_arn,
            "function_code_sha256": code_sha,
            "execution_role_arn": role_arn,
            "execution_role_id": role["RoleId"],
            "role_trust_policy_sha256": canonical_sha256(
                _policy_document(role.get("AssumeRolePolicyDocument"), "role trust policy")
            ),
            "role_permission_policy_sha256": canonical_sha256(permissions),
            "stack_id": support_stack_id,
            "stack_template_sha256": canonical_sha256(template),
            "resource_policy_sha256": resource_policy_sha,
            "event_source_state_sha256": event_source_sha,
            "function_url_state_sha256": function_url_sha,
            "expected_attachment_identity_sha256": canonical_sha256([]),
            "observed_attachment_identity_sha256": event_source_sha,
            "disabled_support_profile_sha256": disabled_profile_sha,
        },
        launch_present,
    )


def _read_body(response: Mapping[str, object], label: str) -> bytes:
    body = response.get("Body")
    read_method = getattr(body, "read", None)
    raw = read_method() if callable(read_method) else body
    if type(raw) is not bytes:
        raise SupportRuntimeTransitionError(label + " body is not bytes")
    return raw


def _reread_runtime_artifact(
    *, identity: SupportRuntimeIdentity, coordinate: object, services: ProductionServices
) -> SupportRuntimeIdentity:
    to_dict = getattr(coordinate, "to_dict", None)
    value = to_dict() if callable(to_dict) else coordinate
    coordinate_value = _mapping(value, "runtime artifact coordinate")
    response = _call(
        services.s3,
        "get_object",
        "GetObject",
        Bucket=coordinate_value["bucket"],
        Key=coordinate_value["key"],
        VersionId=coordinate_value["version_id"],
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    raw = _read_body(response, "runtime artifact")
    if (
        hashlib.sha256(raw).hexdigest() != coordinate_value.get("file_sha256")
        or raw != canonical_json_bytes(identity.to_dict()) + b"\n"
    ):
        raise SupportRuntimeTransitionError("runtime artifact immutable bytes drifted")
    try:
        parsed = parse_support_runtime_identity(json.loads(raw.decode("ascii")))
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SupportRuntimeTransitionError("runtime artifact parse failed") from exc
    if parsed.to_dict() != identity.to_dict():
        raise SupportRuntimeTransitionError("runtime artifact identity drifted")
    return parsed


def commit_support_runtime_identity_v2(
    *,
    request: object,
    operation_7: StackMigrationOperation7EvidenceV2,
    disabled_support: DisabledSupportDeploymentEvidence,
    services: ProductionServices,
) -> SupportRuntimeIdentity:
    """Freshly derive, publish, commit, reload, and independently revalidate identity."""
    if (
        type(operation_7) is not StackMigrationOperation7EvidenceV2
        or type(disabled_support) is not DisabledSupportDeploymentEvidence
        or type(services) is not ProductionServices
    ):
        raise SupportRuntimeTransitionError("support runtime binding inputs are not exact")
    parsed = parse_support_runtime_commit_request(request)
    if (
        parsed.operation_7_identity_sha256 != operation_7.canonical_identity_sha256
        or parsed.disabled_support_identity_sha256
        != disabled_support.canonical_identity_sha256
        or operation_7.disabled_support_identity_sha256
        != disabled_support.canonical_identity_sha256
        or parsed.expected_contract.get("disabled_support_profile_sha256")
        != disabled_support.disabled_support_profile_sha256
    ):
        raise SupportRuntimeTransitionError("support runtime prerequisite identity drifted")
    first, launch_present = _fresh_runtime_readback(
        services=services, support_stack_id=operation_7.support_stack_id
    )
    if launch_present:
        raise SupportRuntimeTransitionError("support runtime exposes launch authority")
    try:
        identity = build_support_runtime_identity(
            activation_id=parsed.activation_id,
            generation=parsed.generation,
            bootstrap_manifest_coordinate=parsed.bootstrap_manifest_coordinate,
            expected_contract=parsed.expected_contract,
            live_readback=first,
        )
        from .task13_fixed_artifacts import Task13FixedArtifactServices

        coordinate = publish_support_runtime_identity(
            identity,
            services=Task13FixedArtifactServices(
                sts=services.sts,
                s3=services.s3,
                total_max_attempts=services.total_max_attempts,
            ),
            kms_key_arn=parsed.kms_key_arn,
        )
        commit_support_runtime_identity(
            identity,
            coordinate=coordinate,
            dynamodb=services.dynamodb,
            table_name=parsed.ledger_table_name,
        )
        loaded, loaded_coordinate = load_support_runtime_identity(
            dynamodb=services.dynamodb,
            table_name=parsed.ledger_table_name,
            activation_id=parsed.activation_id,
            generation=parsed.generation,
        )
        loaded = _reread_runtime_artifact(
            identity=loaded, coordinate=loaded_coordinate, services=services
        )
        second, second_launch_present = _fresh_runtime_readback(
            services=services, support_stack_id=operation_7.support_stack_id
        )
        if second_launch_present:
            raise SupportRuntimeTransitionError("support runtime gained launch authority")
        return require_live_support_runtime_identity(
            loaded,
            expected_contract=parsed.expected_contract,
            live_readback=second,
        )
    except SupportRuntimeTransitionError:
        raise
    except (TypeError, ValueError) as exc:
        raise SupportRuntimeTransitionError(
            "support runtime publication/commit failed closed"
        ) from exc


def _query_zero(
    *, services: ProductionServices, table: str, activation_id: str, prefix: str
) -> int:
    from .dynamodb import encode_item

    response = _call(
        services.dynamodb,
        "query",
        "Query",
        TableName=table,
        KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
        ExpressionAttributeValues=encode_item(
            {":pk": RUN_ID, ":prefix": f"ACTIVATION#{activation_id}#{prefix}"}
        ),
        ConsistentRead=True,
        Select="COUNT",
        Limit=1,
        ReturnConsumedCapacity="NONE",
    )
    count = response.get("Count")
    if (
        type(count) is not int
        or count < 0
        or response.get("ScannedCount") != count
        or response.get("LastEvaluatedKey") is not None
    ):
        raise SupportRuntimeTransitionError("ledger query is ambiguous or unbounded")
    return count


def _p5_count(services: ProductionServices) -> int:
    response = _call(
        services.ec2,
        "describe_instances",
        "DescribeInstances",
        Filters=[
            {"Name": "instance-type", "Values": ["p5.48xlarge"]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped"],
            },
        ],
        MaxResults=1000,
    )
    if response.get("NextToken") is not None:
        raise SupportRuntimeTransitionError("EC2 worker inventory is unbounded")
    reservations = response.get("Reservations")
    if type(reservations) is not list:
        raise SupportRuntimeTransitionError("EC2 worker inventory drifted")
    seen: set[str] = set()
    for reservation in reservations:
        instances = reservation.get("Instances") if type(reservation) is dict else None
        if type(instances) is not list:
            raise SupportRuntimeTransitionError("EC2 worker reservation drifted")
        for instance in instances:
            instance_id = instance.get("InstanceId") if type(instance) is dict else None
            instance_type = instance.get("InstanceType") if type(instance) is dict else None
            if (
                type(instance_id) is not str
                or instance_id in seen
                or instance_type != "p5.48xlarge"
            ):
                raise SupportRuntimeTransitionError("EC2 worker identity drifted")
            seen.add(instance_id)
    return len(seen)


def _cloudtrail_count(
    *, services: ProductionServices, event_name: str, start: datetime, end: datetime
) -> int:
    response = _call(
        services.cloudtrail,
        "lookup_events",
        "LookupEvents",
        LookupAttributes=[{"AttributeKey": "EventName", "AttributeValue": event_name}],
        StartTime=start,
        EndTime=end,
        MaxResults=50,
    )
    if response.get("NextToken") is not None:
        raise SupportRuntimeTransitionError("CloudTrail launch history is unbounded")
    events = response.get("Events")
    if type(events) is not list or len(events) > 50:
        raise SupportRuntimeTransitionError("CloudTrail launch history drifted")
    seen: set[str] = set()
    for event in events:
        if type(event) is not dict or not {
            "EventId",
            "EventName",
            "EventTime",
            "EventSource",
            "CloudTrailEvent",
        } <= set(event):
            raise SupportRuntimeTransitionError("CloudTrail launch event is incomplete")
        event_id = event["EventId"]
        event_time = event["EventTime"]
        if (
            type(event_id) is not str
            or not event_id
            or event_id in seen
            or event["EventName"] != event_name
            or event["EventSource"] != "ec2.amazonaws.com"
            or type(event_time) is not datetime
            or event_time.tzinfo is None
            or not (start <= event_time.astimezone(timezone.utc) <= end)
        ):
            raise SupportRuntimeTransitionError("CloudTrail launch event identity drifted")
        try:
            raw = json.loads(event["CloudTrailEvent"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise SupportRuntimeTransitionError("CloudTrail canonical event is malformed") from exc
        if (
            type(raw) is not dict
            or raw.get("eventID") != event_id
            or raw.get("eventName") != event_name
            or raw.get("eventSource") != "ec2.amazonaws.com"
            or raw.get("awsRegion") != REGION
        ):
            raise SupportRuntimeTransitionError("CloudTrail canonical event drifted")
        seen.add(event_id)
    return len(seen)


def prove_no_launch_v2(
    *, request: object, runtime_identity: SupportRuntimeIdentity, services: ProductionServices
) -> Mapping[str, object]:
    """Return the exact seven-field zero proof from bounded fresh reads only."""
    if type(runtime_identity) is not SupportRuntimeIdentity or type(services) is not ProductionServices:
        raise SupportRuntimeTransitionError("no-launch binding inputs are not exact")
    parsed = parse_no_launch_observation_request(request)
    if (
        parsed.activation_id != runtime_identity.activation_id
        or parsed.generation != runtime_identity.generation
        or parsed.runtime_identity_sha256 != runtime_identity.canonical_identity_sha256
    ):
        raise SupportRuntimeTransitionError("no-launch runtime identity drifted")
    fresh, launch_present = _fresh_runtime_readback(
        services=services,
        support_stack_id=str(runtime_identity.live_identity["stack_id"]),
    )
    require_live_support_runtime_identity(
        runtime_identity,
        expected_contract=dict(runtime_identity.expected_contract),
        live_readback=fresh,
    )
    counts = {
        prefix: _query_zero(
            services=services,
            table=parsed.ledger_table_name,
            activation_id=parsed.activation_id,
            prefix=prefix,
        )
        for prefix in _LEDGER_PREFIXES
    }
    worker_count = _p5_count(services)
    cloudtrail_calls = sum(
        _cloudtrail_count(
            services=services,
            event_name=name,
            start=parsed.evidence_window_start,
            end=parsed.evidence_window_end,
        )
        for name in _CLOUDTRAIL_LAUNCH_EVENTS
    )
    raw_launch_calls = counts["EC2_LAUNCH#"] + cloudtrail_calls
    source_calls = counts["SOURCE_ACTION#"]
    authority = bool(counts["LAUNCH_AUTHORITY#"] or launch_present)
    if counts["WORKER#"] or worker_count or raw_launch_calls or source_calls or authority:
        raise SupportRuntimeTransitionError("no-launch proof observed a nonzero category")
    return {
        "schema_version": 2,
        "record_type": "glm52_h1g_no_launch_evidence_v2",
        "worker_count": 0,
        "raw_ec2_launch_calls": 0,
        "source_action_calls": 0,
        "launch_authority_present": False,
        "complete": True,
    }


def read_no_launch_v2(
    *, request: object, runtime_identity: SupportRuntimeIdentity, services: ProductionServices
) -> Mapping[str, object]:
    """Adoption reader for the inherently read-only no-launch effect."""
    return prove_no_launch_v2(
        request=request, runtime_identity=runtime_identity, services=services
    )


def read_support_runtime_identity_v2(
    *, request: object, operation_7: StackMigrationOperation7EvidenceV2, services: ProductionServices
) -> SupportRuntimeIdentity:
    """Load the committed identity and independently prove current equality."""
    if type(operation_7) is not StackMigrationOperation7EvidenceV2 or type(services) is not ProductionServices:
        raise SupportRuntimeTransitionError("runtime identity read inputs are not exact")
    parsed = parse_support_runtime_commit_request(request)
    identity, coordinate = load_support_runtime_identity(
        dynamodb=services.dynamodb,
        table_name=parsed.ledger_table_name,
        activation_id=parsed.activation_id,
        generation=parsed.generation,
    )
    identity = _reread_runtime_artifact(
        identity=identity, coordinate=coordinate, services=services
    )
    fresh, launch_present = _fresh_runtime_readback(
        services=services, support_stack_id=operation_7.support_stack_id
    )
    if launch_present:
        raise SupportRuntimeTransitionError("committed runtime exposes launch authority")
    return require_live_support_runtime_identity(
        identity,
        expected_contract=parsed.expected_contract,
        live_readback=fresh,
    )


def read_disabled_support_deployment_v2(
    *,
    request: object,
    checkpoint: StackMigrationTransferCheckpointV2,
    support_inputs: SupportBuildInputs,
    services: ProductionServices,
) -> DisabledSupportDeploymentEvidence:
    """Adopt one sole exact template version while the anchor remains inert."""
    if (
        type(checkpoint) is not StackMigrationTransferCheckpointV2
        or type(support_inputs) is not SupportBuildInputs
        or type(services) is not ProductionServices
    ):
        raise SupportRuntimeTransitionError(
            "disabled support read inputs are not exact"
        )
    parsed = parse_disabled_support_deployment_request(request)
    if (
        support_build_inputs_projection(support_inputs)
        != support_build_inputs_projection(parsed.support_build_inputs)
    ):
        raise SupportRuntimeTransitionError(
            "disabled support staged snapshot drifted"
        )
    if (
        parsed.support_build_inputs_identity_sha256
        != support_build_inputs_identity(support_inputs)
        or parsed.activation_id != support_inputs.activation_id
        or parsed.bootstrap_manifest_coordinate
        != dict(support_inputs.bootstrap_manifest_coordinate)
        or parsed.prepare_entry_identity_sha256
        != support_inputs.prepare_entry_identity_sha256
    ):
        raise SupportRuntimeTransitionError(
            "disabled support read prerequisite drifted"
        )
    price = support_price_card_from_mapping(
        parsed.price_card,
        inputs=support_inputs,
    )
    bundle = build_support_precreate_plane(
        inputs=support_inputs,
        price_card=price,
    )
    template = dict(bundle.support_template)
    proof = _static_launch_inert(template)
    raw = canonical_json_bytes(template) + b"\n"
    key = _disabled_support_key(parsed.activation_id, parsed.generation)
    versions_response = _call(
        services.s3,
        "list_object_versions",
        "ListObjectVersions",
        Bucket=MODEL_BUCKET_NAME,
        Prefix=key,
        MaxKeys=2,
        ExpectedBucketOwner=ACCOUNT_ID,
    )
    versions = versions_response.get("Versions")
    delete_markers = versions_response.get("DeleteMarkers", [])
    if (
        type(versions) is not list
        or len(versions) != 1
        or type(versions[0]) is not dict
        or versions[0].get("Key") != key
        or versions[0].get("IsLatest") is not True
        or type(versions[0].get("VersionId")) is not str
        or not versions[0]["VersionId"]
        or type(delete_markers) is not list
        or delete_markers
        or versions_response.get("IsTruncated") is not False
        or versions_response.get("NextKeyMarker") is not None
        or versions_response.get("NextVersionIdMarker") is not None
    ):
        raise SupportRuntimeTransitionError(
            "disabled support version history is absent or ambiguous"
        )
    version_id = str(versions[0]["VersionId"])
    response = _call(
        services.s3,
        "get_object",
        "GetObject",
        Bucket=MODEL_BUCKET_NAME,
        Key=key,
        VersionId=version_id,
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    observed = _read_body(response, "disabled support template")
    if observed != raw:
        raise SupportRuntimeTransitionError(
            "disabled support template bytes drifted"
        )
    _require_inert_support_anchor(
        checkpoint=checkpoint,
        services=services,
    )
    return _build_disabled_evidence(
        parsed=parsed,
        template_raw=raw,
        template_sha=hashlib.sha256(raw[:-1]).hexdigest(),
        version_id=version_id,
        no_launch_sha=canonical_sha256(proof),
    )


__all__ = [
    "DisabledSupportDeploymentRequest",
    "NoLaunchObservationRequest",
    "OperationSevenCompletionRequest",
    "SupportRuntimeCommitRequest",
    "SupportRuntimeTransitionError",
    "commit_support_runtime_identity_v2",
    "complete_operation_7_v2",
    "deploy_disabled_support_v2",
    "parse_disabled_support_deployment_request",
    "parse_no_launch_observation_request",
    "parse_operation_seven_completion_request",
    "parse_support_runtime_commit_request",
    "prove_no_launch_v2",
    "read_disabled_support_deployment_v2",
    "read_operation_7_v2",
    "read_no_launch_v2",
    "read_support_runtime_identity_v2",
]
