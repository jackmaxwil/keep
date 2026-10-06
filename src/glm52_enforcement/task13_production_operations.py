"""Concrete production adapter for Task 13 staged infrastructure.

The staged-deployment state machine owns ordering and durable restart
semantics.  This module supplies its real AWS and repository-helper boundary:
all SDK clients are configured for one total attempt, all CloudFormation
mutations are UPDATEs against the inert anchors, and no method exposes
``RunInstances`` or a worker-activation helper.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .canonical import canonical_json_bytes, canonical_sha256
from .cloudformation_stacks import (
    STACK_TAGS,
    MigrationBootstrapResult,
    MigrationEvidence,
    StackIdentity,
    StackKind,
    TemplateCoordinate,
    build_stack_migration,
    validate_bootstrap_template,
)
from .task13_migration_adapter import (
    MIGRATION_TEMPLATE_KEYS,
    SeededMigrationRuntime,
    Task13MigrationAdapterError,
    bootstrap_seeded_migration,
    build_sealed_migration_runtime,
    build_seeded_migration_runtime,
    execute_sealed_migration,
    sealed_migration_projection,
    validate_sealed_migration_projection,
    validate_stack_migration_seed_projection,
)
from .task13_staged_deployment import (
    ACCOUNT_ID,
    FENCE_ROLE_ARN,
    FENCE_STACK_NAME,
    MODEL_BUCKET_NAME,
    PROFILE,
    REGION,
    RETAINED_STACK_NAME,
    SUPPORT_ROLE_ARN,
    SUPPORT_STACK_NAME,
    DeploymentStep,
    StagedDeploymentRequest,
)

RUN_ID = "glm52-sky-20260724"
SUPPORT_INPUTS_KEY = "task13/inputs/support-build-inputs.json"
CRYPTOGRAPHY_LAYER_NAME = "keep-glm52-h1g-cryptography-py312-x86-64"
STAGED_INFRASTRUCTURE_ARTIFACT_KEYS = {
    "BOOTSTRAP_TEMPLATE": "task13/templates/container-bootstrap-v1.json",
    "FENCE_TEMPLATE": "task13/migration/fence-transfer.json",
    "RETAINED_FOUNDATION_TEMPLATE": ("task13/templates/retained-foundation.yaml"),
    "RETAINED_PRE_SUPPORT_TEMPLATE": ("task13/templates/retained-pre-support.yaml"),
    "RETAINED_TEMPLATE": "task13/templates/retained.yaml",
    "SUPPORT_INPUTS": "task13/inputs/support-build-inputs.json",
    "SUPPORT_TEMPLATE": "task13/templates/support-disabled.yaml",
}
_RETAINED_PHASES: Mapping[str, Mapping[str, str]] = {
    "pre-support": {
        "artifact_kind": "RETAINED_PRE_SUPPORT_TEMPLATE",
        "key": "task13/templates/retained-pre-support.yaml",
        "change_set_name": "glm52-task13-retained-pre-support-v1",
    },
    "fence-bootstrap-v9": {
        "artifact_kind": "RETAINED_FENCE_BOOTSTRAP_TEMPLATE_V9",
        "key": "task13/templates/retained-fence-bootstrap-v9.json",
        "change_set_name": "glm52-h1g-retained-fence-bootstrap-v9",
    },
    "fence-runtime": {
        "artifact_kind": "RETAINED_FENCE_RUNTIME_TEMPLATE",
        "key": "task13/templates/retained-fence-runtime-v2.json",
        "change_set_name": "glm52-h1g-retained-fence-runtime-v2",
    },
    "final": {
        "artifact_kind": "RETAINED_TEMPLATE",
        "key": "task13/templates/retained.yaml",
        "change_set_name": "glm52-task13-retained-final-v1",
    },
}
_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"([A-Za-z0-9-]+)/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CHANGE_SET_NAME = re.compile(r"[A-Za-z][-A-Za-z0-9]{0,127}\Z")
_INSTANCE_ID = re.compile(r"i-[0-9a-f]{8,17}\Z")
_VERSION = re.compile(r"(?!null\Z)(?!None\Z)[\x21-\x7e]{1,1024}\Z")
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
_TAGS = [{"Key": key, "Value": value} for key, value in STACK_TAGS]
_PRODUCTION_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "output_directory",
        "retained_stack_id",
        "retained_bootstrap_runtime_deployment",
        "bridge_seed_publication",
        "bridge_seed",
        "migration_operations_1_to_6",
        "bootstrap_fence_publication",
        "retained_fence_runtime_deployment",
        "prepare_execution",
        "support_input_materialization_request",
        "disabled_support_deployment",
        "operation_7",
        "support_runtime_identity",
        "no_launch_evidence",
    }
)


class ProductionOperationError(ValueError):
    """A concrete production helper or AWS readback failed closed."""


@dataclass(frozen=True)
class ProductionServices:
    """All zero-retry clients used by the concrete staged adapter."""

    sts: object
    cloudformation: object
    iam: object
    s3: object
    organizations: object
    ec2: object
    ssm: object
    kms: object
    dynamodb: object
    lambda_client: object
    states: object
    cloudtrail: object
    total_max_attempts: int
    credential_expiration: datetime | None = None
    fixed_artifact_publisher_s3: object | None = None


def _missing_binding(*_args: object, **_kwargs: object) -> object:
    raise ProductionOperationError("production repository helper is absent")


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ProductionOperationBindings:
    """Lazy-bound repository helpers used by the AWS adapter."""

    parse_retained_bootstrap_inputs: Callable[[object], object]
    build_retained_bootstrap_fragment: Callable[[object], Mapping[str, object]]
    parse_retained_fence_runtime_inputs: Callable[[object], object]
    build_retained_fence_runtime_fragment: Callable[[object], Mapping[str, object]]
    invoke_bootstrap_materializer_v2: Callable[..., object]
    parse_pre_support_inputs: Callable[[object], object]
    pre_support_inputs_identity: Callable[[object], str]
    build_pre_support_fragment: Callable[[object], Mapping[str, object]]
    collect_support_build_inputs: Callable[..., object]
    support_build_inputs_identity: Callable[[object], str]
    support_build_inputs_projection: Callable[[object], Mapping[str, object]]
    parse_support_price_card: Callable[[object], object]
    build_support_precreate_plane: Callable[..., object]
    coordinate_support_postcreate: Callable[..., object]
    publish_reviewed_artifact: Callable[..., Mapping[str, object]]
    publish_support_build_inputs: Callable[..., Mapping[str, object]]
    apply_retained_update: Callable[..., Mapping[str, object]]
    capture_precreate_baseline: Callable[..., Mapping[str, object]]
    write_precreate_authority: Callable[..., Mapping[str, object]]
    read_direct_grant_evidence: Callable[..., Mapping[str, object]]
    build_activation_orphan_authority: Callable[..., Mapping[str, object]]
    publish_activation_orphan_authority: Callable[..., Mapping[str, object]]
    read_activation_orphan_coordinate: Callable[..., Mapping[str, object] | None]

    materialize_bridge_seed_establishment_request_v2: Callable[..., object] = (
        _missing_binding
    )
    establish_bridge_seed_v2: Callable[..., object] = _missing_binding
    complete_operations_1_to_6_v2: Callable[..., object] = _missing_binding
    publish_bootstrap_fence_artifacts_v2: Callable[..., object] = _missing_binding
    execute_prepare_v2: Callable[..., object] = _missing_binding
    deploy_disabled_support_v2: Callable[..., object] = _missing_binding
    complete_operation_7_v2: Callable[..., object] = _missing_binding
    commit_support_runtime_identity_v2: Callable[..., object] = _missing_binding
    prove_no_launch_v2: Callable[..., object] = _missing_binding
    reconcile_staged_mutation_v2: Callable[..., object] = _missing_binding
    adopt_staged_evidence_v2: Callable[..., object] = _missing_binding

    @classmethod
    def fake_for_tests(
        cls,
        **overrides: Callable[..., object],
    ) -> ProductionOperationBindings:
        names = cls.__dataclass_fields__
        unknown = set(overrides) - set(names)
        if unknown:
            raise TypeError("unknown fake binding: " + ",".join(sorted(unknown)))
        return cls(**{name: overrides.get(name, _missing_binding) for name in names})


def _execute_prepare_v2_binding(
    *,
    request: object,
    checkpoint: object,
    publication: object,
    services: ProductionServices,
) -> object:
    """Pin and execute PREPARE through Task-3's public v2 custody surface."""

    from .fence_artifacts import FenceSlot, parse_fence_transition_request
    from .fence_executor import (
        DynamoFenceRecordStore,
        FenceExecutor,
        execute_prepare_v2,
        load_pinned_fence_entry_template,
        load_pinned_fence_manifest,
    )
    from .task11_production import _Task11FenceAwsClient
    from .task13_migration_adapter import StackMigrationTransferCheckpointV2
    from .task13_staged_deployment import BootstrapFencePublication

    if (
        type(request) is not dict
        or type(checkpoint) is not StackMigrationTransferCheckpointV2
        or type(publication) is not BootstrapFencePublication
        or type(services) is not ProductionServices
    ):
        raise ProductionOperationError("PREPARE binding inputs are not exact v2")
    try:
        transition = parse_fence_transition_request(request)
        if (
            transition.slot is not FenceSlot.PREPARE_GENESIS_LIVE_STATE
            or transition.to_dict().get("manifest_coordinate")
            != publication.manifest_coordinate.to_dict()
        ):
            raise ProductionOperationError("PREPARE request is not publication-bound")
        manifest = load_pinned_fence_manifest(
            s3=services.s3,
            coordinate=publication.manifest_coordinate,
        )
        entry = manifest.entry(FenceSlot.PREPARE_GENESIS_LIVE_STATE)
        if entry.entry_identity_sha256 != publication.prepare_entry_identity_sha256:
            raise ProductionOperationError("PREPARE entry identity drifted")
        entry = load_pinned_fence_entry_template(s3=services.s3, entry=entry)
        result = execute_prepare_v2(
            executor=FenceExecutor(
                client=_Task11FenceAwsClient(
                    cloudformation=services.cloudformation,
                    s3=services.s3,
                    iam=services.iam,
                ),
                record_store=DynamoFenceRecordStore(
                    client=services.dynamodb,
                    table_name="keep-glm52-h1g-ledger-v1",
                ),
            ),
            request=transition,
            manifest=manifest,
            entry=entry,
        )
    except ProductionOperationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProductionOperationError(
            "public PREPARE execution refused its pinned inputs"
        ) from exc
    if result.stack_id != checkpoint.fence_stack_id:
        raise ProductionOperationError("PREPARE result fence stack drifted")
    return result


def _invoke_bootstrap_materializer_v2_binding(
    *,
    event: object,
    runtime: object,
    services: ProductionServices,
) -> object:
    """Assume the exact invoker and parse one version-pinned Lambda result."""

    if (
        type(event) is not dict
        or type(runtime) is not dict
        or type(services) is not ProductionServices
        or services.total_max_attempts != 1
    ):
        raise ProductionOperationError(
            "bootstrap materializer invocation inputs are not exact"
        )
    invoker_role_arn = runtime.get("invoker_role_arn")
    function_arn = runtime.get("materializer_function_version_arn")
    if (
        invoker_role_arn
        != (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-fence-bootstrap-materializer-invoker"
        )
        or type(function_arn) is not str
        or not function_arn.startswith(
            "arn:aws:lambda:us-west-2:246813579024:function:"
        )
        or function_arn.rsplit(":", 1)[-1] in {"", "$LATEST"}
    ):
        raise ProductionOperationError(
            "bootstrap materializer runtime identity is not exact"
        )
    session_name = "glm52-h1g-bootstrap-invoker"
    assumed = _call(
        services.sts,
        "assume_role",
        operation="AssumeRole(BootstrapInvoker)",
        RoleArn=invoker_role_arn,
        RoleSessionName=session_name,
        DurationSeconds=900,
    )
    credentials = assumed.get("Credentials")
    assumed_user = assumed.get("AssumedRoleUser")
    expected_arn = (
        "arn:aws:sts::246813579024:assumed-role/"
        "keep-glm52-h1g-fence-bootstrap-materializer-invoker/" + session_name
    )
    if (
        type(credentials) is not dict
        or type(assumed_user) is not dict
        or assumed_user.get("Arn") != expected_arn
    ):
        raise ProductionOperationError("bootstrap invoker role assumption is not exact")
    for field in ("AccessKeyId", "SecretAccessKey", "SessionToken"):
        if type(credentials.get(field)) is not str or not credentials[field]:
            raise ProductionOperationError(
                "bootstrap invoker credentials are incomplete"
            )
    try:  # pragma: no cover - deployment host dependency
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - deployment host dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    invoker = boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=REGION,
    ).client(
        "lambda",
        config=Config(
            connect_timeout=5,
            read_timeout=30,
            retries={"mode": "standard", "total_max_attempts": 1},
        ),
    )
    response = _call(
        invoker,
        "invoke",
        operation="Invoke(BootstrapMaterializer)",
        FunctionName=function_arn,
        InvocationType="RequestResponse",
        Payload=canonical_json_bytes(event),
    )
    payload = response.get("Payload")
    read = getattr(payload, "read", None)
    raw = read() if callable(read) else payload
    if (
        response.get("StatusCode") != 200
        or response.get("FunctionError") is not None
        or response.get("ExecutedVersion") != function_arn.rsplit(":", 1)[-1]
        or type(raw) is not bytes
    ):
        raise ProductionOperationError(
            "bootstrap materializer invocation did not succeed exactly"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionOperationError(
            "bootstrap materializer payload is not one JSON result"
        ) from exc
    phase = event.get("phase")
    if phase == "BRIDGE_SEED":
        from .fence_bootstrap_publication import (
            parse_bridge_seed_publication_v2,
        )

        return parse_bridge_seed_publication_v2(value)
    if phase == "BOOTSTRAP_MANIFEST":
        from .task13_staged_deployment import (
            parse_bootstrap_fence_publication,
        )

        return parse_bootstrap_fence_publication(value)
    raise ProductionOperationError(
        "bootstrap materializer invocation phase is not implemented"
    )


def _default_bindings() -> ProductionOperationBindings:
    from .fence_bootstrap_publication import (
        publish_bootstrap_fence_artifacts_v2,
    )
    from .staged_effect_recovery import (
        adopt_staged_evidence_v2,
        reconcile_staged_mutation_v2,
    )
    from .staged_migration_operations import (
        complete_operations_1_to_6_v2,
        establish_bridge_seed_v2,
        materialize_bridge_seed_establishment_request_v2,
    )
    from .support_plane import (
        build_pre_support_retained_runtime_fragment,
        build_retained_fence_bootstrap_fragment,
        build_retained_fence_runtime_fragment,
        build_support_precreate_plane,
        coordinate_support_postcreate,
        pre_support_runtime_inputs_from_mapping,
        pre_support_runtime_inputs_identity,
        retained_fence_bootstrap_inputs_from_mapping,
        retained_fence_runtime_inputs_from_mapping,
        support_price_card_from_mapping,
    )
    from .support_runtime_transition import (
        commit_support_runtime_identity_v2,
        complete_operation_7_v2,
        deploy_disabled_support_v2,
        prove_no_launch_v2,
    )
    from .task12_orphan_authority import (
        build_activation_orphan_authority,
        capture_precreate_baseline,
        publish_activation_authority,
        read_activation_authority_coordinate,
        read_direct_grant_evidence,
        write_o_excl_authority,
    )
    from .task13_fixed_artifacts import publish_support_build_inputs
    from .task13_retained_update import apply_retained_update
    from .task13_reviewed_artifacts import publish_reviewed_artifact
    from .task13_support_input_materialization import (
        collect_support_build_input_snapshot,
        support_build_inputs_identity,
        support_build_inputs_projection,
    )

    return ProductionOperationBindings(
        parse_retained_bootstrap_inputs=(retained_fence_bootstrap_inputs_from_mapping),
        build_retained_bootstrap_fragment=(build_retained_fence_bootstrap_fragment),
        parse_retained_fence_runtime_inputs=(
            retained_fence_runtime_inputs_from_mapping
        ),
        build_retained_fence_runtime_fragment=(build_retained_fence_runtime_fragment),
        invoke_bootstrap_materializer_v2=(_invoke_bootstrap_materializer_v2_binding),
        parse_pre_support_inputs=pre_support_runtime_inputs_from_mapping,
        pre_support_inputs_identity=pre_support_runtime_inputs_identity,
        build_pre_support_fragment=(build_pre_support_retained_runtime_fragment),
        collect_support_build_inputs=collect_support_build_input_snapshot,
        support_build_inputs_identity=support_build_inputs_identity,
        support_build_inputs_projection=support_build_inputs_projection,
        parse_support_price_card=support_price_card_from_mapping,
        build_support_precreate_plane=build_support_precreate_plane,
        coordinate_support_postcreate=coordinate_support_postcreate,
        publish_reviewed_artifact=publish_reviewed_artifact,
        publish_support_build_inputs=publish_support_build_inputs,
        apply_retained_update=apply_retained_update,
        capture_precreate_baseline=capture_precreate_baseline,
        write_precreate_authority=write_o_excl_authority,
        read_direct_grant_evidence=read_direct_grant_evidence,
        build_activation_orphan_authority=(build_activation_orphan_authority),
        publish_activation_orphan_authority=publish_activation_authority,
        read_activation_orphan_coordinate=(read_activation_authority_coordinate),
        materialize_bridge_seed_establishment_request_v2=(
            materialize_bridge_seed_establishment_request_v2
        ),
        establish_bridge_seed_v2=establish_bridge_seed_v2,
        complete_operations_1_to_6_v2=complete_operations_1_to_6_v2,
        publish_bootstrap_fence_artifacts_v2=(publish_bootstrap_fence_artifacts_v2),
        deploy_disabled_support_v2=deploy_disabled_support_v2,
        complete_operation_7_v2=complete_operation_7_v2,
        commit_support_runtime_identity_v2=(commit_support_runtime_identity_v2),
        prove_no_launch_v2=prove_no_launch_v2,
        reconcile_staged_mutation_v2=reconcile_staged_mutation_v2,
        adopt_staged_evidence_v2=adopt_staged_evidence_v2,
        execute_prepare_v2=_execute_prepare_v2_binding,
    )


def _copy(value: object, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionOperationError(label + " is not canonical JSON data") from exc


def _metadata(value: object, operation: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ProductionOperationError(operation + " returned no object")
    metadata = value.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") not in {200, 201}
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        raise ProductionOperationError(
            operation + " lacks authenticated zero-retry success"
        )
    return value


def _call(
    client: object,
    method_name: str,
    *,
    operation: str,
    **request: object,
) -> Mapping[str, object]:
    method = getattr(client, method_name, None)
    if not callable(method):
        raise ProductionOperationError(operation + " SDK method is unavailable")
    return _metadata(method(**request), operation)


def _error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    detail = response.get("Error") if type(response) is dict else None
    code = detail.get("Code") if type(detail) is dict else None
    return code if type(code) is str else None


def _error_metadata(error: BaseException, operation: str) -> None:
    response = getattr(error, "response", None)
    metadata = response.get("ResponseMetadata") if type(response) is dict else None
    if (
        type(metadata) is not dict
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
        or type(metadata.get("HTTPStatusCode")) is not int
    ):
        raise ProductionOperationError(
            operation + " error is not authenticated zero-retry evidence"
        ) from error


def _stack_id(value: object, name: str) -> str:
    match = _STACK_ID.fullmatch(value) if type(value) is str else None
    if match is None or match.group(1) != name:
        raise ProductionOperationError(name + " stack ID is not exact")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ProductionOperationError(label + " is not a lowercase SHA-256")
    return value


def _coordinate(
    value: object,
    *,
    artifact_kind: str,
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != _COORDINATE_FIELDS
        or value.get("artifact_kind") != artifact_kind
        or value.get("bucket") != MODEL_BUCKET_NAME
        or type(value.get("key")) is not str
        or not value["key"]
        or value["key"].startswith("/")
        or any(part in {"", ".", ".."} for part in value["key"].split("/"))
        or type(value.get("version_id")) is not str
        or _VERSION.fullmatch(value["version_id"]) is None
    ):
        raise ProductionOperationError(
            artifact_kind + " coordinate is not immutable and exact"
        )
    _sha(value.get("file_sha256"), artifact_kind + " file identity")
    _sha(value.get("body_sha256"), artifact_kind + " body identity")
    return _copy(value, artifact_kind + " coordinate")


def _template_url(coordinate: Mapping[str, object]) -> str:
    return (
        f"https://{coordinate['bucket']}.s3.{REGION}.amazonaws.com/"
        f"{quote(str(coordinate['key']), safe='/')}"
        f"?versionId={quote(str(coordinate['version_id']), safe='')}"
    )


def _common() -> dict[str, int]:
    return {
        "worker_activation_attempts": 0,
        "raw_ec2_launch_calls": 0,
    }


_WORKER_LAUNCH_RESOURCE_TYPES = frozenset(
    {
        "AWS::AutoScaling::AutoScalingGroup",
        "AWS::Batch::ComputeEnvironment",
        "AWS::EC2::EC2Fleet",
        "AWS::EC2::Fleet",
        "AWS::EC2::LaunchTemplate",
        "AWS::EC2::SpotFleet",
    }
)
_WORKER_LAUNCH_ACTIONS = frozenset(
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


def _worker_activation_evidence(
    template: Mapping[str, object],
) -> dict[str, object]:
    if type(template) is not dict:
        raise ProductionOperationError("support template is not one exact object")
    parameters = template.get("Parameters", {})
    resources = template.get("Resources")
    if (
        type(parameters) is not dict
        or parameters
        or type(resources) is not dict
        or not resources
    ):
        raise ProductionOperationError(
            "support template exposes an activation parameter or has no resource graph"
        )
    launch_resources = []
    p5_paths: list[str] = []
    launch_actions: set[str] = set()

    def walk(value: object, path: tuple[str, ...]) -> None:
        if value == "p5.48xlarge":
            p5_paths.append(".".join(path))
            return
        if type(value) is dict:
            actions = value.get("Action")
            if type(actions) is str:
                candidates = [actions]
            elif type(actions) is list and all(
                type(action) is str for action in actions
            ):
                candidates = actions
            elif actions is None:
                candidates = []
            else:
                raise ProductionOperationError("support IAM action shape is malformed")
            for action in candidates:
                if action in _WORKER_LAUNCH_ACTIONS or action in {"ec2:*", "*"}:
                    launch_actions.add(action)
            for key, child in value.items():
                if type(key) is not str or not key:
                    raise ProductionOperationError(
                        "support template object key is malformed"
                    )
                walk(child, (*path, key))
            return
        if type(value) is list:
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    for logical_id, resource in resources.items():
        if (
            type(logical_id) is not str
            or not logical_id
            or type(resource) is not dict
            or type(resource.get("Type")) is not str
        ):
            raise ProductionOperationError("support resource graph is malformed")
        resource_type = resource["Type"]
        if resource_type in _WORKER_LAUNCH_RESOURCE_TYPES:
            launch_resources.append(
                {
                    "logical_resource_id": logical_id,
                    "resource_type": resource_type,
                }
            )
        if resource_type == "AWS::EC2::Instance":
            properties = resource.get("Properties")
            if (
                type(properties) is not dict
                or properties.get("InstanceType") != "c6a.xlarge"
            ):
                launch_resources.append(
                    {
                        "logical_resource_id": logical_id,
                        "resource_type": resource_type,
                    }
                )
        walk(resource, ("Resources", logical_id))
    evidence = {
        "launch_capable_actions": sorted(launch_actions),
        "launch_capable_resources": sorted(
            launch_resources,
            key=lambda row: (
                str(row["logical_resource_id"]),
                str(row["resource_type"]),
            ),
        ),
        "p5_instance_paths": sorted(p5_paths),
        "template_parameters": sorted(parameters),
        "worker_activation_enabled": bool(
            launch_actions or launch_resources or p5_paths or parameters
        ),
    }
    return evidence


def _canonical_mapping_file(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not one regular file")
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionOperationError(label + " is unavailable") from exc
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ProductionOperationError(
            label + " is not one canonical JSON object plus LF"
        )
    return value, raw


def _template_body(value: object, label: str) -> dict[str, object]:
    if type(value) is dict:
        return _copy(value, label)
    if type(value) is str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProductionOperationError(label + " is not canonical JSON") from exc
        if type(parsed) is dict:
            return _copy(parsed, label)
    raise ProductionOperationError(label + " is not one template object")


def _retained_runtime_live_state(
    *,
    retained_stack_id: str,
    fragment: Mapping[str, object],
    services: ProductionServices,
    phase: str,
) -> tuple[str, str, dict[str, str]]:
    if type(services) is not ProductionServices or services.total_max_attempts != 1:
        raise ProductionOperationError(
            phase + " retained runtime reader requires exact one-attempt services"
        )
    response = _call(
        services.cloudformation,
        "describe_stacks",
        operation="DescribeRetainedRuntimeStack",
        StackName=retained_stack_id,
    )
    stacks = response.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1 or type(stacks[0]) is not dict:
        raise ProductionOperationError(
            phase + " retained runtime stack readback is not singular"
        )
    stack = stacks[0]
    if (
        stack.get("StackId") != retained_stack_id
        or stack.get("StackStatus") != "UPDATE_COMPLETE"
    ):
        raise ProductionOperationError(
            phase + " retained runtime stack is not update-complete"
        )
    template_response = _call(
        services.cloudformation,
        "get_template",
        operation="GetRetainedRuntimeTemplate",
        StackName=retained_stack_id,
        TemplateStage="Original",
    )
    template = _template_body(
        template_response.get("TemplateBody"),
        phase + " retained runtime template",
    )
    fragment_resources = fragment.get("Resources")
    fragment_outputs = fragment.get("Outputs")
    live_resources = template.get("Resources")
    live_outputs = template.get("Outputs")
    if (
        type(fragment_resources) is not dict
        or type(fragment_outputs) is not dict
        or type(live_resources) is not dict
        or type(live_outputs) is not dict
        or any(
            live_resources.get(logical_id) != resource
            for logical_id, resource in fragment_resources.items()
        )
        or any(
            live_outputs.get(output_key) != output
            for output_key, output in fragment_outputs.items()
        )
    ):
        raise ProductionOperationError(
            phase + " retained runtime fragment drifted in live template"
        )
    output_rows = stack.get("Outputs")
    if type(output_rows) is not list:
        raise ProductionOperationError(
            phase + " retained runtime outputs are unavailable"
        )
    outputs: dict[str, str] = {}
    for row in output_rows:
        if (
            type(row) is not dict
            or type(row.get("OutputKey")) is not str
            or type(row.get("OutputValue")) is not str
            or row["OutputKey"] in outputs
        ):
            raise ProductionOperationError(
                phase + " retained runtime outputs are not exact"
            )
        outputs[row["OutputKey"]] = row["OutputValue"]
    template_sha256 = hashlib.sha256(canonical_json_bytes(template)).hexdigest()
    retained_update_identity_sha256 = canonical_sha256(
        {
            "schema_version": 2,
            "record_type": "glm52_h1g_retained_runtime_live_state_v2",
            "phase": phase,
            "retained_stack_id": retained_stack_id,
            "template_sha256": template_sha256,
            "source_fragment_sha256": hashlib.sha256(
                canonical_json_bytes(fragment)
            ).hexdigest(),
        }
    )
    return template_sha256, retained_update_identity_sha256, outputs


def _exact_function_version_arn(value: object, function_name: str) -> str:
    prefix = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{function_name}:"
    suffix = (
        value[len(prefix) :] if type(value) is str and value.startswith(prefix) else ""
    )
    if not suffix.isdigit() or suffix.startswith("0"):
        raise ProductionOperationError(
            function_name + " retained function version ARN is not exact"
        )
    return value


def read_retained_bootstrap_runtime_deployment_v2(
    *,
    request: object,
    services: ProductionServices,
) -> dict[str, object]:
    from .support_plane import (
        build_retained_fence_bootstrap_fragment,
        retained_fence_bootstrap_inputs_from_mapping,
    )

    try:
        inputs = retained_fence_bootstrap_inputs_from_mapping(request)
        fragment = build_retained_fence_bootstrap_fragment(inputs)
    except (TypeError, ValueError) as exc:
        raise ProductionOperationError(
            "retained bootstrap runtime request is not exact"
        ) from exc
    template_sha256, update_identity, outputs = _retained_runtime_live_state(
        retained_stack_id=inputs.retained_stack_id,
        fragment=fragment,
        services=services,
        phase="fence-bootstrap-v9",
    )
    materializer_arn = _exact_function_version_arn(
        outputs.get("FenceBootstrapMaterializerVersionArn"),
        "keep-glm52-h1g-fence-bootstrap-materializer",
    )
    publisher_role_arn = outputs.get("FenceBootstrapPublisherRoleArn")
    invoker_role_arn = outputs.get("FenceBootstrapInvokerRoleArn")
    if publisher_role_arn != (
        f"arn:aws:iam::{ACCOUNT_ID}:role/"
        "keep-glm52-h1g-fence-bootstrap-artifact-publisher"
    ) or invoker_role_arn != (
        f"arn:aws:iam::{ACCOUNT_ID}:role/"
        "keep-glm52-h1g-fence-bootstrap-materializer-invoker"
    ):
        raise ProductionOperationError(
            "retained bootstrap runtime role outputs are not exact"
        )
    unsigned = {
        "schema_version": 2,
        "record_type": "glm52_h1g_retained_bootstrap_runtime_deployment_v2",
        "retained_stack_id": inputs.retained_stack_id,
        "template_sha256": template_sha256,
        "retained_update_identity_sha256": update_identity,
        "materializer_function_version_arn": materializer_arn,
        "publisher_role_arn": publisher_role_arn,
        "invoker_role_arn": invoker_role_arn,
        "worker_activation_allowed": False,
        "source_action_allowed": False,
    }
    return {
        **unsigned,
        "canonical_identity_sha256": canonical_sha256(unsigned),
    }


def read_retained_fence_runtime_deployment_v2(
    *,
    request: object,
    checkpoint: object,
    publication: object,
    services: ProductionServices,
) -> dict[str, object]:
    from .support_plane import (
        build_retained_fence_runtime_fragment,
        retained_fence_runtime_inputs_from_mapping,
    )

    try:
        inputs = retained_fence_runtime_inputs_from_mapping(request)
        manifest_coordinate = publication.manifest_coordinate.to_dict()
        if (
            checkpoint.fence_stack_id != inputs.fence_stack_id
            or manifest_coordinate != dict(inputs.bootstrap_manifest_coordinate)
        ):
            raise ValueError("runtime predecessor identity drifted")
        fragment = build_retained_fence_runtime_fragment(inputs)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProductionOperationError(
            "retained fence runtime request is not predecessor-bound"
        ) from exc
    template_sha256, update_identity, outputs = _retained_runtime_live_state(
        retained_stack_id=inputs.retained_stack_id,
        fragment=fragment,
        services=services,
        phase="fence-runtime",
    )
    executor_arn = _exact_function_version_arn(
        outputs.get("PreSupportFenceExecutorVersionArn"),
        "keep-glm52-h1g-pre-support-fence-executor",
    )
    settlement_arn = _exact_function_version_arn(
        outputs.get("FenceSourceSettlementMaterializerVersionArn"),
        "keep-glm52-h1g-fence-source-settlement-materializer",
    )
    unsigned = {
        "schema_version": 2,
        "record_type": "glm52_h1g_retained_fence_runtime_deployment_v2",
        "retained_stack_id": inputs.retained_stack_id,
        "template_sha256": template_sha256,
        "retained_update_identity_sha256": update_identity,
        "bootstrap_manifest_coordinate": manifest_coordinate,
        "pre_support_executor_function_version_arn": executor_arn,
        "source_settlement_function_version_arn": settlement_arn,
        "worker_activation_allowed": False,
        "source_action_allowed": False,
    }
    return {
        **unsigned,
        "canonical_identity_sha256": canonical_sha256(unsigned),
    }


def _guard_production_request(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _PRODUCTION_REQUEST_FIELDS:
        missing = (
            sorted(_PRODUCTION_REQUEST_FIELDS - set(value))
            if type(value) is dict
            else []
        )
        unknown = (
            sorted(set(value) - _PRODUCTION_REQUEST_FIELDS)
            if type(value) is dict
            else []
        )
        raise ProductionOperationError(
            f"production request v2 schema mismatch: missing={missing}, "
            f"unknown={unknown}"
        )
    if (
        value["schema_version"] != 2
        or value["record_type"] != "glm52_task13_production_operations_v2"
        or type(value["activation_id"]) is not str
        or not value["activation_id"]
    ):
        raise ProductionOperationError("production request identity is not exact v2")
    output = Path(str(value["output_directory"]))
    if (
        not output.is_absolute()
        or not output.is_dir()
        or output.is_symlink()
        or output.resolve(strict=True) != output
    ):
        raise ProductionOperationError("production output directory is not exact")
    _stack_id(value["retained_stack_id"], RETAINED_STACK_NAME)
    for field in (
        "retained_bootstrap_runtime_deployment",
        "bridge_seed_publication",
        "bridge_seed",
        "migration_operations_1_to_6",
        "bootstrap_fence_publication",
        "retained_fence_runtime_deployment",
        "prepare_execution",
        "support_input_materialization_request",
        "disabled_support_deployment",
        "operation_7",
        "support_runtime_identity",
        "no_launch_evidence",
    ):
        if type(value[field]) is not dict:
            raise ProductionOperationError(field + " must be one exact object")
    try:
        from .fence_bootstrap_publication import (
            parse_bootstrap_publication_authority_v2,
            parse_bridge_seed_publication_authority_v2,
        )
        from .support_plane import (
            retained_fence_bootstrap_inputs_from_mapping,
            retained_fence_runtime_authority_from_mapping,
        )

        bootstrap_inputs = retained_fence_bootstrap_inputs_from_mapping(
            value["retained_bootstrap_runtime_deployment"]
        )
        seed_request = parse_bridge_seed_publication_authority_v2(
            value["bridge_seed_publication"]
        )
        parse_bootstrap_publication_authority_v2(
            value["bootstrap_fence_publication"]
        )
        runtime_inputs = retained_fence_runtime_authority_from_mapping(
            value["retained_fence_runtime_deployment"]
        )
    except (TypeError, ValueError) as exc:
        raise ProductionOperationError(
            "retained fence runtime requests are not exact v2"
        ) from exc
    if (
        bootstrap_inputs.activation_id != value["activation_id"]
        or seed_request.activation_id != value["activation_id"]
        or runtime_inputs["activation_id"] != value["activation_id"]
        or bootstrap_inputs.retained_stack_id != value["retained_stack_id"]
        or runtime_inputs["retained_stack_id"] != value["retained_stack_id"]
    ):
        raise ProductionOperationError(
            "retained fence runtime request identity drifted"
        )
    forbidden_copies = {"fence_template_inventory", "task11_writer_bindings"}
    if forbidden_copies & set(value["support_input_materialization_request"]):
        raise ProductionOperationError(
            "support input materialization request is not exact v2: "
            "caller-authored fence inventories"
        )
    try:
        from .task13_support_input_materialization import (
            parse_support_input_materialization_request,
        )

        parsed = parse_support_input_materialization_request(
            value["support_input_materialization_request"]
        )
    except (TypeError, ValueError) as exc:
        raise ProductionOperationError(
            "support input materialization request is not exact v2"
        ) from exc
    if parsed.activation_id != value["activation_id"]:
        raise ProductionOperationError(
            "support input materialization activation identity drifted"
        )
    return _copy(value, "production request v2")


def _validate_retained_transaction_composition(
    *,
    phase: str,
    before: Mapping[str, object],
    after: Mapping[str, object],
    source_fragment: Mapping[str, object],
    changes: object,
) -> str:
    from .support_plane import compose_complete_retained_template
    from .task13_retained_update import (
        pin_retained_bootstrap_v9_template,
        retained_bootstrap_v9_roundtrip_change_projection,
    )

    try:
        expected_after = compose_complete_retained_template(before, source_fragment)
        if phase == "fence-bootstrap-v9":
            expected_after = pin_retained_bootstrap_v9_template(expected_after)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ProductionOperationError(
            phase + " retained source fragment composition is invalid"
        ) from exc
    if canonical_json_bytes(expected_after) != canonical_json_bytes(after):
        raise ProductionOperationError(
            phase + " retained transaction composition is not exact"
        )

    resources = source_fragment.get("Resources")
    if type(resources) is not dict or not resources or type(changes) is not list:
        raise ProductionOperationError(
            phase + " retained source resource additions are not exact"
        )
    expected_types: dict[str, str] = {}
    for logical_id, resource in resources.items():
        if (
            type(logical_id) is not str
            or not logical_id
            or type(resource) is not dict
            or type(resource.get("Type")) is not str
            or not resource["Type"]
            or logical_id in expected_types
        ):
            raise ProductionOperationError(
                phase + " retained source resource additions are not exact"
            )
        expected_types[logical_id] = resource["Type"]

    expected_roundtrips = {
        row["logical_id"]: row
        for row in (
            retained_bootstrap_v9_roundtrip_change_projection()
            if phase == "fence-bootstrap-v9"
            else []
        )
    }
    observed_additions: dict[str, str] = {}
    observed_roundtrips: dict[str, dict[str, object]] = {}
    for row in changes:
        if (
            type(row) is not dict
            or row.get("replacement") not in {"None", "False"}
            or type(row.get("logical_id")) is not str
            or type(row.get("resource_type")) is not str
        ):
            raise ProductionOperationError(
                phase + " retained source resource additions are not exact"
            )
        logical_id = row["logical_id"]
        if logical_id in expected_types:
            if row.get("action") != "Add" or logical_id in observed_additions:
                raise ProductionOperationError(
                    phase + " retained source resource additions are not exact"
                )
            observed_additions[logical_id] = row["resource_type"]
        elif logical_id in expected_roundtrips:
            if (
                row != expected_roundtrips[logical_id]
                or logical_id in observed_roundtrips
            ):
                raise ProductionOperationError(
                    phase + " retained semantic round-trips are not exact"
                )
            observed_roundtrips[logical_id] = row
        else:
            raise ProductionOperationError(
                phase + " retained source resource additions are not exact"
            )
    if (
        observed_additions != expected_types
        or observed_roundtrips != expected_roundtrips
    ):
        raise ProductionOperationError(
            phase + " retained source resource additions are not exact"
        )
    return hashlib.sha256(canonical_json_bytes(source_fragment)).hexdigest()


class Task13ProductionOperations:
    """Concrete implementation of every staged-deployment operation."""

    def __init__(
        self,
        *,
        production_request: Mapping[str, object],
        services: ProductionServices,
        bindings: ProductionOperationBindings,
        sleep: Callable[[float], None],
        now: Callable[[], datetime],
        max_polls: int,
    ) -> None:
        if (
            type(services) is not ProductionServices
            or services.total_max_attempts != 1
            or type(bindings) is not ProductionOperationBindings
            or not callable(sleep)
            or not callable(now)
            or type(max_polls) is not int
            or not 2 <= max_polls <= 720
        ):
            raise ProductionOperationError(
                "production services or polling bound is not exact"
            )
        self.production_request = _guard_production_request(production_request)
        self.services = services
        self.bindings = bindings
        self.sleep = sleep
        self.now = now
        self.max_polls = max_polls
        self._pre_inputs: object | None = None
        self._pre_fragment: Mapping[str, object] | None = None
        self._support_inputs: object | None = None
        self._support_bundle: object | None = None
        self._support_runtime_evidence: Mapping[str, object] | None = None
        self._support_template_coordinate: Mapping[str, object] | None = None
        self._support_inputs_coordinate: Mapping[str, object] | None = None
        self._precreate_baseline: Mapping[str, object] | None = None
        self._precreate_observed_at: str | None = None
        self._postcreate_bundle: object | None = None
        self._orphan_activation_authority: Mapping[str, object] | None = None
        self._postcreate_observed_at: str | None = None
        self._migration_runtime: SeededMigrationRuntime | None = None
        self._migration_bootstrap: MigrationBootstrapResult | None = None
        self._sealed_migration: Mapping[str, object] | None = None

    @property
    def output_directory(self) -> Path:
        return Path(str(self.production_request["output_directory"]))

    def _assert_request(self, request: StagedDeploymentRequest) -> None:
        if (
            type(request) is not StagedDeploymentRequest
            or request.activation_id != self.production_request["activation_id"]
            or request.production_request != self.production_request
        ):
            raise ProductionOperationError(
                "staged and production requests are not identical"
            )

    def _caller(self) -> Mapping[str, object]:
        response = _call(
            self.services.sts,
            "get_caller_identity",
            operation="GetCallerIdentity",
        )
        arn = response.get("Arn")
        if (
            response.get("Account") != ACCOUNT_ID
            or type(arn) is not str
            or f"::{ACCOUNT_ID}:" not in arn
            or type(response.get("UserId")) is not str
            or not response["UserId"]
        ):
            raise ProductionOperationError("AWS caller identity is foreign")
        return response

    def _describe_stack(
        self,
        identity: str,
        *,
        name: str,
        statuses: set[str],
    ) -> Mapping[str, object]:
        response = _call(
            self.services.cloudformation,
            "describe_stacks",
            operation="DescribeStacks",
            StackName=identity,
        )
        stacks = response.get("Stacks")
        if type(stacks) is not list or len(stacks) != 1:
            raise ProductionOperationError("stack lookup is not singular")
        stack = stacks[0]
        if (
            type(stack) is not dict
            or stack.get("StackName") != name
            or _stack_id(stack.get("StackId"), name) != stack["StackId"]
            or stack.get("StackStatus") not in statuses
        ):
            raise ProductionOperationError(name + " stack readback drifted")
        return _copy(stack, name + " stack")

    def _optional_stack(
        self,
        name: str,
        *,
        statuses: set[str],
    ) -> Mapping[str, object] | None:
        try:
            return self._describe_stack(
                name,
                name=name,
                statuses=statuses,
            )
        except Exception as exc:
            if _error_code(exc) != "ValidationError":
                raise
            _error_metadata(exc, "DescribeStacks")
            return None

    def _resources(self, stack_id: str) -> list[Mapping[str, object]]:
        rows: list[Mapping[str, object]] = []
        token: str | None = None
        seen: set[str] = set()
        for _page in range(64):
            request: dict[str, object] = {"StackName": stack_id}
            if token is not None:
                request["NextToken"] = token
            response = _call(
                self.services.cloudformation,
                "list_stack_resources",
                operation="ListStackResources",
                **request,
            )
            page = response.get("StackResourceSummaries")
            if type(page) is not list or any(type(row) is not dict for row in page):
                raise ProductionOperationError("stack resource inventory is malformed")
            rows.extend(_copy(page, "stack resources"))
            next_token = response.get("NextToken")
            if next_token is None:
                return rows
            if type(next_token) is not str or not next_token or next_token in seen:
                raise ProductionOperationError(
                    "stack resource pagination is incomplete"
                )
            seen.add(next_token)
            token = next_token
        raise ProductionOperationError("stack resource pagination exceeded its bound")

    def guard_account_live_baseline(self, request, committed):
        del committed
        self._assert_request(request)
        self._caller()
        observed_at = self.now()
        expiration = self.services.credential_expiration
        if (
            type(observed_at) is not datetime
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or type(expiration) is not datetime
            or expiration.tzinfo is None
            or expiration.utcoffset() is None
        ):
            raise ProductionOperationError("credential expiration is absent")
        seconds_remaining = int(
            (expiration.astimezone(UTC) - observed_at.astimezone(UTC)).total_seconds()
        )
        if seconds_remaining < 3600:
            raise ProductionOperationError(
                "credential expiration is insufficient for mutation"
            )
        _call(
            self.services.s3,
            "head_bucket",
            operation="HeadModelBucket",
            Bucket=MODEL_BUCKET_NAME,
            ExpectedBucketOwner=ACCOUNT_ID,
        )
        versioning = _call(
            self.services.s3,
            "get_bucket_versioning",
            operation="GetModelBucketVersioning",
            Bucket=MODEL_BUCKET_NAME,
            ExpectedBucketOwner=ACCOUNT_ID,
        )
        if versioning.get("Status") != "Enabled":
            raise ProductionOperationError("retained model bucket is not versioned")
        stack = self._describe_stack(
            RETAINED_STACK_NAME,
            name=RETAINED_STACK_NAME,
            statuses={"UPDATE_COMPLETE"},
        )
        if stack["StackId"] != self.production_request["retained_stack_id"]:
            raise ProductionOperationError(
                "retained stack ID differs from the sealed request"
            )
        if (
            stack.get("RoleARN") != SUPPORT_ROLE_ARN
            or stack.get("EnableTerminationProtection") is not True
        ):
            raise ProductionOperationError(
                "retained stack service role or termination protection is not exact"
            )
        return {
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "profile": PROFILE,
            "model_bucket_name": MODEL_BUCKET_NAME,
            "retained_stack_id": stack["StackId"],
            "retained_role_arn": SUPPORT_ROLE_ARN,
            "retained_termination_protection": True,
            "credential_observed_at": observed_at.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "credential_expiration": expiration.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "credential_seconds_remaining": seconds_remaining,
            "exact": True,
            **_common(),
        }

    def _anchor_stack(
        self,
        *,
        name: str,
        role_arn: str,
        allow_updated: bool,
    ) -> Mapping[str, object] | None:
        statuses = (
            {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
            if allow_updated
            else {"CREATE_COMPLETE"}
        )
        stack = self._optional_stack(name, statuses=statuses)
        if stack is None:
            return None
        if stack.get("RoleARN") != role_arn:
            raise ProductionOperationError(
                name + " does not use its exact service role"
            )
        if stack.get("EnableTerminationProtection") is not True:
            raise ProductionOperationError(
                name + " termination protection is not enabled"
            )
        tags = {
            row.get("Key"): row.get("Value")
            for row in stack.get("Tags", [])
            if type(row) is dict
        }
        if tags != dict(STACK_TAGS):
            raise ProductionOperationError(name + " tags are not exact")
        template_response = _call(
            self.services.cloudformation,
            "get_template",
            operation="GetBootstrapContainerTemplate",
            StackName=stack["StackId"],
            TemplateStage="Original",
        )
        template = _template_body(
            template_response.get("TemplateBody"),
            name + " bootstrap template",
        )
        try:
            validate_bootstrap_template(template)
        except (TypeError, ValueError) as exc:
            raise ProductionOperationError(
                name + " is not the exact inert v1 container"
            ) from exc
        resources = self._resources(str(stack["StackId"]))
        anchors = [
            row
            for row in resources
            if row.get("LogicalResourceId") == "ContainerAnchor"
            and row.get("ResourceType") == "AWS::CloudFormation::WaitConditionHandle"
        ]
        if len(anchors) != 1:
            raise ProductionOperationError(name + " does not preserve its inert anchor")
        return stack

    def _wait_anchor(self, name: str, role_arn: str) -> Mapping[str, object]:
        for _poll in range(self.max_polls):
            stack = self._describe_stack(
                name,
                name=name,
                statuses={"CREATE_IN_PROGRESS", "CREATE_COMPLETE"},
            )
            if stack["StackStatus"] == "CREATE_COMPLETE":
                exact = self._anchor_stack(
                    name=name,
                    role_arn=role_arn,
                    allow_updated=False,
                )
                assert exact is not None
                return exact
            self.sleep(2.0)
        raise ProductionOperationError(name + " bootstrap did not stabilize")

    def bootstrap_stack_migration(self, request, committed):
        """Initialize and authenticate the coordinator-owned inert anchors."""

        del committed
        self._assert_request(request)
        self._caller()
        try:
            runtime = build_seeded_migration_runtime(
                cloudformation=self.services.cloudformation,
                dynamodb=self.services.dynamodb,
                value=self.production_request["stack_migration_seed"],
            )
            result = bootstrap_seeded_migration(runtime)
            durable = runtime.backend.read_state(
                action_identity_sha256=(runtime.seed.authority.action_identity_sha256)
            )
        except (TypeError, ValueError, Task13MigrationAdapterError) as exc:
            raise ProductionOperationError(
                "stack migration bootstrap failed closed"
            ) from exc
        self._migration_runtime = runtime
        self._migration_bootstrap = result
        return {
            "fence_stack_id": result.fence.stack_id,
            "support_stack_id": result.support.stack_id,
            "deployment_role_arn": (runtime.seed.authority.deployment_role_arn),
            "fence_anchor_inert": True,
            "support_anchor_inert": True,
            "fence_stack_status": "CREATE_COMPLETE",
            "support_stack_status": "CREATE_COMPLETE",
            "state_revision": result.state_revision,
            "reconciled_creates": [kind.value for kind in result.reconciled_creates],
            "action_identity_sha256": result.action_identity_sha256,
            "durable_state_sha256": durable["StateBodySha256"],
            **_common(),
        }

    def bootstrap_inert_anchors(self, request, committed):
        del committed
        self._assert_request(request)
        self._caller()
        coordinate = _coordinate(
            self.production_request["bootstrap_template"],
            artifact_kind="BOOTSTRAP_TEMPLATE",
        )
        found: dict[str, Mapping[str, object]] = {}
        for name, role in (
            (FENCE_STACK_NAME, FENCE_ROLE_ARN),
            (SUPPORT_STACK_NAME, SUPPORT_ROLE_ARN),
        ):
            stack = self._anchor_stack(
                name=name,
                role_arn=role,
                allow_updated=False,
            )
            if stack is None:
                token = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "activation_id": request.activation_id,
                            "stack_name": name,
                            "template": coordinate,
                            "role_arn": role,
                        }
                    )
                ).hexdigest()
                response = _call(
                    self.services.cloudformation,
                    "create_stack",
                    operation="CreateAnchorStack",
                    StackName=name,
                    TemplateURL=_template_url(coordinate),
                    RoleARN=role,
                    ClientRequestToken=token,
                    EnableTerminationProtection=True,
                    Tags=_TAGS,
                )
                _stack_id(response.get("StackId"), name)
                stack = self._wait_anchor(name, role)
            found[name] = stack
        return {
            "fence_stack_id": found[FENCE_STACK_NAME]["StackId"],
            "support_stack_id": found[SUPPORT_STACK_NAME]["StackId"],
            "fence_role_arn": FENCE_ROLE_ARN,
            "support_role_arn": SUPPORT_ROLE_ARN,
            "fence_anchor_inert": True,
            "support_anchor_inert": True,
            "fence_stack_status": "CREATE_COMPLETE",
            "support_stack_status": "CREATE_COMPLETE",
            **_common(),
        }

    def prove_model_bucket_policy_absent(self, request, committed):
        del committed
        self._assert_request(request)
        try:
            _call(
                self.services.s3,
                "get_bucket_policy",
                operation="GetModelBucketPolicy",
                Bucket=MODEL_BUCKET_NAME,
                ExpectedBucketOwner=ACCOUNT_ID,
            )
        except Exception as exc:
            _error_metadata(exc, "GetModelBucketPolicy")
            if _error_code(exc) != "NoSuchBucketPolicy":
                raise ProductionOperationError(
                    "model bucket policy absence is not exact"
                ) from exc
        else:
            raise ProductionOperationError("model bucket already has a bucket policy")
        return {
            "bucket_name": MODEL_BUCKET_NAME,
            "error_code": "NoSuchBucketPolicy",
            "complete": True,
            **_common(),
        }

    def _change_set_context(
        self,
        label: str,
        *,
        allow_updated: bool,
    ) -> tuple[str, str, str, Mapping[str, object]]:
        if label == "fence":
            stack_name = FENCE_STACK_NAME
            role = FENCE_ROLE_ARN
            change_set = str(self.production_request["fence_change_set_name"])
            coordinate = _coordinate(
                self.production_request["fence_template"],
                artifact_kind="FENCE_TEMPLATE",
            )
        elif label == "support":
            stack_name = SUPPORT_STACK_NAME
            role = SUPPORT_ROLE_ARN
            change_set = str(self.production_request["support_change_set_name"])
            if self._support_template_coordinate is None:
                raise ProductionOperationError(
                    "support template has not been immutably published"
                )
            coordinate = _coordinate(
                self._support_template_coordinate,
                artifact_kind="SUPPORT_TEMPLATE",
            )
        else:
            raise ProductionOperationError("change-set label is unknown")
        stack = self._describe_stack(
            stack_name,
            name=stack_name,
            statuses=(
                {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
                if allow_updated
                else {"CREATE_COMPLETE"}
            ),
        )
        expected_role = role
        if stack.get("RoleARN") != expected_role:
            raise ProductionOperationError(label + " anchor role changed before update")
        return str(stack["StackId"]), change_set, role, coordinate

    def _create_change_set(self, label: str) -> Mapping[str, object]:
        stack_id, name, role, coordinate = self._change_set_context(
            label,
            allow_updated=False,
        )
        token = hashlib.sha256(
            canonical_json_bytes(
                {
                    "stack_id": stack_id,
                    "change_set_name": name,
                    "coordinate": coordinate,
                    "role_arn": role,
                }
            )
        ).hexdigest()
        response = _call(
            self.services.cloudformation,
            "create_change_set",
            operation="Create" + label.title() + "ChangeSet",
            StackName=stack_id,
            ChangeSetName=name,
            ChangeSetType="UPDATE",
            Description=(
                "Task 13 reviewed additive "
                + label
                + " update "
                + str(coordinate["body_sha256"])
            ),
            TemplateURL=_template_url(coordinate),
            Parameters=[],
            Capabilities=["CAPABILITY_NAMED_IAM"],
            RoleARN=role,
            IncludeNestedStacks=False,
            ClientToken=token,
            Tags=[
                *_TAGS,
                {
                    "Key": "Task13ActivationId",
                    "Value": self.production_request["activation_id"],
                },
            ],
        )
        change_set_id = response.get("Id")
        if (
            type(change_set_id) is not str
            or not change_set_id
            or response.get("StackId") != stack_id
        ):
            raise ProductionOperationError(
                label + " CreateChangeSet identity is incomplete"
            )
        return {
            "stack_id": stack_id,
            "change_set_id": change_set_id,
            "change_set_type": "UPDATE",
            **_common(),
        }

    def create_fence_change_set(self, request, committed):
        self._assert_request(request)
        del committed
        raise ProductionOperationError(
            "legacy additive fence route is retired; use stack migration"
        )

    def create_support_change_set(self, request, committed):
        self._assert_request(request)
        del committed
        raise ProductionOperationError(
            "legacy additive support route is retired; use stack migration"
        )

    def _describe_change_set(
        self,
        label: str,
        *,
        require_available: bool,
        require_executed: bool = False,
    ) -> Mapping[str, object]:
        if require_available and require_executed:
            raise ProductionOperationError(
                "change set cannot require two execution states"
            )
        stack_id, name, _role, coordinate = self._change_set_context(
            label,
            allow_updated=True,
        )
        changes: list[Mapping[str, object]] = []
        token: str | None = None
        first: Mapping[str, object] | None = None
        for _page in range(64):
            request: dict[str, object] = {
                "StackName": stack_id,
                "ChangeSetName": name,
                "IncludePropertyValues": True,
            }
            if token is not None:
                request["NextToken"] = token
            response = _call(
                self.services.cloudformation,
                "describe_change_set",
                operation="Describe" + label.title() + "ChangeSet",
                **request,
            )
            if first is None:
                first = response
            elif any(
                response.get(field) != first.get(field)
                for field in (
                    "ChangeSetId",
                    "ChangeSetName",
                    "StackId",
                    "StackName",
                    "ChangeSetType",
                    "Status",
                    "ExecutionStatus",
                )
            ):
                raise ProductionOperationError(label + " change-set pagination drifted")
            page_changes = response.get("Changes")
            if type(page_changes) is not list or any(
                type(row) is not dict for row in page_changes
            ):
                raise ProductionOperationError(
                    label + " change-set changes are malformed"
                )
            changes.extend(_copy(page_changes, label + " changes"))
            next_token = response.get("NextToken")
            if next_token is None:
                break
            if type(next_token) is not str or not next_token:
                raise ProductionOperationError(
                    label + " change-set pagination is incomplete"
                )
            token = next_token
        else:
            raise ProductionOperationError(
                label + " change-set pagination exceeded its bound"
            )
        assert first is not None
        expected_name = FENCE_STACK_NAME if label == "fence" else SUPPORT_STACK_NAME
        if (
            first.get("StackId") != stack_id
            or first.get("StackName") != expected_name
            or first.get("ChangeSetName") != name
            or first.get("ChangeSetType") != "UPDATE"
            or first.get("Status") != "CREATE_COMPLETE"
            or (require_available and first.get("ExecutionStatus") != "AVAILABLE")
            or (
                not require_available
                and not require_executed
                and first.get("ExecutionStatus")
                not in {
                    "AVAILABLE",
                    "EXECUTE_IN_PROGRESS",
                    "EXECUTE_COMPLETE",
                }
            )
            or (require_executed and first.get("ExecutionStatus") != "EXECUTE_COMPLETE")
        ):
            raise ProductionOperationError(
                label + " change set identity or state drifted"
            )
        change_template = _call(
            self.services.cloudformation,
            "get_template",
            operation="Get" + label.title() + "ChangeSetTemplate",
            ChangeSetName=first["ChangeSetId"],
            StackName=stack_id,
            TemplateStage="Original",
        )
        change_body = _template_body(
            change_template.get("TemplateBody"),
            label + " change-set Original template",
        )
        if (
            hashlib.sha256(canonical_json_bytes(change_body)).hexdigest()
            != coordinate["body_sha256"]
        ):
            raise ProductionOperationError(
                label + " change-set template identity drifted"
            )
        worker_evidence: dict[str, object] | None = None
        if label == "support":
            worker_evidence = _worker_activation_evidence(change_body)
            if worker_evidence["worker_activation_enabled"] is not False:
                raise ProductionOperationError(
                    "support template contains a worker activation path"
                )
        normalized = []
        for row in changes:
            resource = row.get("ResourceChange")
            if (
                row.get("Type") != "Resource"
                or type(resource) is not dict
                or resource.get("Action") not in {"Add", "Remove"}
                or resource.get("Replacement") not in {None, "False"}
                or resource.get("Details") not in (None, [])
                or resource.get("Scope") not in (None, [])
                or type(resource.get("LogicalResourceId")) is not str
                or not resource["LogicalResourceId"]
                or type(resource.get("ResourceType")) is not str
                or not resource["ResourceType"]
            ):
                raise ProductionOperationError(
                    label + " change set is not an exact reviewed delta "
                    "without replacement"
                )
            normalized.append(
                {
                    "action": resource["Action"],
                    "logical_resource_id": resource["LogicalResourceId"],
                    "resource_type": resource["ResourceType"],
                }
            )
        normalized.sort(
            key=lambda row: (
                0 if row["action"] == "Remove" else 1,
                str(row["logical_resource_id"]),
            )
        )
        if not normalized or len(
            {row["logical_resource_id"] for row in normalized}
        ) != len(normalized):
            raise ProductionOperationError(
                label + " change set inventory is empty or duplicated"
            )
        if label == "fence" and normalized != [
            {
                "action": "Add",
                "logical_resource_id": "H1gProductionFenceBucketPolicy",
                "resource_type": "AWS::S3::BucketPolicy",
            }
        ]:
            raise ProductionOperationError(
                "fence change set is not the sole bucket-policy Add"
            )
        if label == "support":
            resources = change_body["Resources"]
            expected = [
                {
                    "action": "Remove",
                    "logical_resource_id": "ContainerAnchor",
                    "resource_type": ("AWS::CloudFormation::WaitConditionHandle"),
                },
                *[
                    {
                        "action": "Add",
                        "logical_resource_id": logical_id,
                        "resource_type": resource["Type"],
                    }
                    for logical_id, resource in sorted(resources.items())
                ],
            ]
            if normalized != expected:
                raise ProductionOperationError(
                    "support change set does not exactly replace "
                    "ContainerAnchor with the reviewed resource graph"
                )
        result = {
            "stack_id": stack_id,
            "change_set_id": first["ChangeSetId"],
            "change_set_type": "UPDATE",
            "status": "CREATE_COMPLETE",
            "changes": normalized,
            **(
                {
                    "worker_activation_enabled": False,
                    "worker_activation_evidence_sha256": (
                        hashlib.sha256(
                            canonical_json_bytes(worker_evidence)
                        ).hexdigest()
                    ),
                }
                if label == "support"
                else {}
            ),
            **_common(),
        }
        if label == "fence":
            result.update(
                {
                    "fence_template_bucket": coordinate["bucket"],
                    "fence_template_key": coordinate["key"],
                    "fence_template_version_id": coordinate["version_id"],
                    "fence_template_file_sha256": coordinate["file_sha256"],
                    "fence_template_body_sha256": coordinate["body_sha256"],
                }
            )
        return result

    def inspect_fence_change_set(self, request, committed):
        self._assert_request(request)
        del committed
        raise ProductionOperationError("legacy additive fence inspection is retired")

    def inspect_support_change_set(self, request, committed):
        self._assert_request(request)
        del committed
        raise ProductionOperationError("legacy additive support inspection is retired")

    def _execute_change_set(
        self,
        label: str,
        committed: Mapping[DeploymentStep, Mapping[str, object]],
    ) -> Mapping[str, object]:
        del label, committed
        raise ProductionOperationError(
            "legacy additive change-set execution is retired"
        )

    def execute_fence_update(self, request, committed):
        self._assert_request(request)
        return self._execute_change_set("fence", committed)

    def execute_support_change_set(self, request, committed):
        self._assert_request(request)
        return self._execute_change_set("support", committed)

    def materialize_pre_support(self, request, committed):
        del committed
        self._assert_request(request)
        inputs = self.bindings.parse_pre_support_inputs(
            self.production_request["pre_support_runtime_inputs"]
        )
        fragment = self.bindings.build_pre_support_fragment(inputs)
        if type(fragment) is not dict:
            raise ProductionOperationError(
                "pre-support builder returned no exact fragment"
            )
        identity = self.bindings.pre_support_inputs_identity(inputs)
        self._pre_inputs = inputs
        self._pre_fragment = _copy(fragment, "pre-support fragment")
        return {
            "narrow_inputs": True,
            "support_build_inputs_sha256": _sha(
                identity,
                "pre-support input identity",
            ),
            "retained_fragment_sha256": hashlib.sha256(
                canonical_json_bytes(fragment)
            ).hexdigest(),
            **_common(),
        }

    def _phase_directory(self, phase: str) -> Path:
        path = self.output_directory / ("retained-" + phase)
        path.mkdir(mode=0o700, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ProductionOperationError(
                "retained update evidence directory is unsafe"
            )
        return path

    def _retained_services(self):
        from .task13_retained_update import RetainedUpdateServices

        return RetainedUpdateServices(
            sts=self.services.sts,
            cloudformation=self.services.cloudformation,
            ec2=self.services.ec2,
            ssm=self.services.ssm,
            iam=self.services.iam,
            s3=self.services.s3,
            total_max_attempts=1,
        )

    def _retained_transaction_evidence(
        self,
        *,
        phase: str,
        source_fragment: Mapping[str, object],
        result: Mapping[str, object] | None = None,
        require_live_current: bool = True,
    ) -> Mapping[str, object]:
        contract = _RETAINED_PHASES.get(phase)
        if contract is None:
            raise ProductionOperationError("retained transaction phase is unknown")
        bootstrap_v9 = phase == "fence-bootstrap-v9"
        expected_role_arn = None if bootstrap_v9 else SUPPORT_ROLE_ARN
        directory = self._phase_directory(phase)
        stem = "retained-" + phase
        before, before_raw = _canonical_mapping_file(
            directory / (stem + "-before-template.json"),
            phase + " retained before-template",
        )
        after, after_raw = _canonical_mapping_file(
            directory / (stem + "-after-template.json"),
            phase + " retained after-template",
        )
        change, _change_raw = _canonical_mapping_file(
            directory / (stem + "-change-set.json"),
            phase + " retained change-set evidence",
        )
        readback, readback_raw = _canonical_mapping_file(
            directory / (stem + "-readback.json"),
            phase + " retained readback evidence",
        )
        coordinate = _coordinate(
            change.get("template_coordinate"),
            artifact_kind=str(contract["artifact_kind"]),
        )
        body_sha = hashlib.sha256(after_raw[:-1]).hexdigest()
        file_sha = hashlib.sha256(after_raw).hexdigest()
        change_set_id = change.get("change_set_id")
        if (
            coordinate["key"] != contract["key"]
            or coordinate["file_sha256"] != file_sha
            or coordinate["body_sha256"] != body_sha
            or change.get("phase") != phase
            or change.get("stack_id") != self.production_request["retained_stack_id"]
            or change.get("stack_name") != RETAINED_STACK_NAME
            or change.get("deployment_role_arn") != SUPPORT_ROLE_ARN
            or (
                bootstrap_v9
                and (
                    change.get("initial_role_arn") is not None
                    or change.get("change_set_role_arn") is not None
                )
            )
            or change.get("before_template_sha256")
            != hashlib.sha256(before_raw[:-1]).hexdigest()
            or change.get("after_template_sha256") != body_sha
            or change.get("change_set_name") != contract["change_set_name"]
            or type(change_set_id) is not str
            or not change_set_id
            or type(change.get("changes")) is not list
            or not change["changes"]
        ):
            raise ProductionOperationError(
                phase + " retained change-set evidence is not exact"
            )
        source_fragment_sha256 = _validate_retained_transaction_composition(
            phase=phase,
            before=before,
            after=after,
            source_fragment=source_fragment,
            changes=change["changes"],
        )
        if (
            readback.get("phase") != phase
            or readback.get("stack_id") != self.production_request["retained_stack_id"]
            or readback.get("stack_name") != RETAINED_STACK_NAME
            or readback.get("stack_status") != "UPDATE_COMPLETE"
            or readback.get("role_arn") != expected_role_arn
            or readback.get("template_sha256") != body_sha
            or readback.get("template_coordinate") != coordinate
        ):
            raise ProductionOperationError(
                phase + " retained readback evidence is not exact"
            )
        initial_semantics = change.get("initial_semantics")
        if bootstrap_v9:
            from .task13_retained_update import (
                RetainedUpdateError,
                validate_retained_bootstrap_v9_semantics,
            )

            if (
                type(initial_semantics) is not dict
                or change.get("pre_execute_semantics") != initial_semantics
                or readback.get("initial_semantics") != initial_semantics
            ):
                raise ProductionOperationError(
                    "retained bootstrap semantic evidence is not exact"
                )
            try:
                validate_retained_bootstrap_v9_semantics(
                    initial=initial_semantics,
                    final=readback.get("final_semantics"),
                )
            except RetainedUpdateError as exc:
                raise ProductionOperationError(
                    "retained bootstrap semantic evidence is not exact"
                ) from exc
        if result is not None and (
            result.get("status") != "UPDATE_COMPLETE"
            or result.get("stack_id") != self.production_request["retained_stack_id"]
            or result.get("change_set_id") != change_set_id
            or result.get("phase") != phase
            or result.get("template_key") != contract["key"]
            or result.get("template_version_id") != coordinate["version_id"]
        ):
            raise ProductionOperationError(
                "retained updater returned inexact transaction evidence"
            )

        stack = self._describe_stack(
            str(self.production_request["retained_stack_id"]),
            name=RETAINED_STACK_NAME,
            statuses={"UPDATE_COMPLETE"},
        )
        if stack.get("RoleARN") != expected_role_arn:
            raise ProductionOperationError(
                phase + " retained stack role readback is not exact"
            )
        live_template = _call(
            self.services.cloudformation,
            "get_template",
            operation="GetRetainedOriginalTemplate",
            StackName=stack["StackId"],
            TemplateStage="Original",
        )
        live_body = _template_body(
            live_template.get("TemplateBody"),
            phase + " live retained Original template",
        )
        if require_live_current:
            if (
                canonical_json_bytes(live_body) != canonical_json_bytes(after)
                or hashlib.sha256(canonical_json_bytes(live_body)).hexdigest()
                != body_sha
            ):
                raise ProductionOperationError(
                    phase + " live retained template differs from publication"
                )
        else:
            after_resources = after.get("Resources")
            live_resources = live_body.get("Resources")
            if (
                type(after_resources) is not dict
                or type(live_resources) is not dict
                or any(
                    logical_id not in live_resources
                    or canonical_json_bytes(live_resources[logical_id])
                    != canonical_json_bytes(resource)
                    for logical_id, resource in after_resources.items()
                )
            ):
                raise ProductionOperationError(
                    phase + " retained resources are not preserved in final template"
                )

        if bootstrap_v9:
            from .task13_retained_update import (
                RetainedUpdateError,
                read_retained_bootstrap_semantic_snapshot,
                validate_retained_bootstrap_v9_semantics,
            )

            try:
                live_semantics = read_retained_bootstrap_semantic_snapshot(
                    services=self._retained_services(),
                    template=live_body,
                    stack=stack,
                )
            except RetainedUpdateError as exc:
                raise ProductionOperationError(
                    "retained bootstrap live semantic readback failed"
                ) from exc
            try:
                validate_retained_bootstrap_v9_semantics(
                    initial=initial_semantics,
                    final=live_semantics,
                )
            except RetainedUpdateError as exc:
                raise ProductionOperationError(
                    "retained bootstrap live semantic values changed"
                ) from exc

        observed_changes: list[dict[str, str]] = []
        token: str | None = None
        first: Mapping[str, object] | None = None
        for _page in range(64):
            request: dict[str, object] = {
                "StackName": stack["StackId"],
                "ChangeSetName": change_set_id,
                "IncludePropertyValues": True,
            }
            if token is not None:
                request["NextToken"] = token
            response = _call(
                self.services.cloudformation,
                "describe_change_set",
                operation="DescribeRetainedChangeSet",
                **request,
            )
            projection = {
                key: response.get(key)
                for key in (
                    "ChangeSetId",
                    "ChangeSetName",
                    "StackId",
                    "StackName",
                    "ChangeSetType",
                    "Status",
                    "ExecutionStatus",
                    "RoleARN",
                )
            }
            if first is None:
                first = projection
            elif canonical_json_bytes(first) != canonical_json_bytes(projection):
                raise ProductionOperationError(
                    phase + " retained change-set pagination drifted"
                )
            rows = response.get("Changes")
            if type(rows) is not list:
                raise ProductionOperationError(
                    phase + " retained changes are malformed"
                )
            for row in rows:
                resource = row.get("ResourceChange") if type(row) is dict else None
                if (
                    type(resource) is not dict
                    or row.get("Type") != "Resource"
                    or resource.get("Replacement") not in {None, "False"}
                    or resource.get("Action") not in {"Add", "Modify"}
                    or type(resource.get("LogicalResourceId")) is not str
                    or type(resource.get("ResourceType")) is not str
                ):
                    raise ProductionOperationError(
                        phase + " retained committed change set is not exact"
                    )
                observed_changes.append(
                    {
                        "action": resource["Action"],
                        "logical_id": resource["LogicalResourceId"],
                        "replacement": str(resource.get("Replacement")),
                        "resource_type": resource["ResourceType"],
                    }
                )
            next_token = response.get("NextToken")
            if next_token is None:
                break
            if type(next_token) is not str or not next_token:
                raise ProductionOperationError(
                    phase + " retained change-set pagination is invalid"
                )
            token = next_token
        else:
            raise ProductionOperationError(
                phase + " retained change-set pagination exceeded its bound"
            )
        if first is None or (
            first.get("ChangeSetId") != change_set_id
            or first.get("ChangeSetName") != contract["change_set_name"]
            or first.get("StackId") != stack["StackId"]
            or first.get("StackName") != RETAINED_STACK_NAME
            or first.get("ChangeSetType") != "UPDATE"
            or first.get("Status") != "CREATE_COMPLETE"
            or first.get("ExecutionStatus") != "EXECUTE_COMPLETE"
            or first.get("RoleARN") != expected_role_arn
            or sorted(observed_changes, key=lambda row: row["logical_id"])
            != change["changes"]
        ):
            raise ProductionOperationError(
                phase + " retained committed change set is not exact"
            )
        change_template = _call(
            self.services.cloudformation,
            "get_template",
            operation="GetRetainedChangeSetTemplate",
            ChangeSetName=change_set_id,
            TemplateStage="Original",
        )
        change_body = _template_body(
            change_template.get("TemplateBody"),
            phase + " retained change-set template",
        )
        if canonical_json_bytes(change_body) != canonical_json_bytes(after):
            raise ProductionOperationError(
                phase + " retained change-set template is not exact"
            )
        return {
            "stack_id": stack["StackId"],
            "stack_status": "UPDATE_COMPLETE",
            "change_set_type": "UPDATE",
            "role_arn": expected_role_arn,
            "phase": phase,
            "source_fragment_sha256": source_fragment_sha256,
            "template_bucket": coordinate["bucket"],
            "template_key": coordinate["key"],
            "template_version_id": coordinate["version_id"],
            "template_file_sha256": coordinate["file_sha256"],
            "template_body_sha256": coordinate["body_sha256"],
            "change_set_id": change_set_id,
            "change_set_name": contract["change_set_name"],
            "readback_sha256": hashlib.sha256(readback_raw).hexdigest(),
            **_common(),
        }

    def _apply_retained(
        self,
        *,
        phase: str,
        fragment: Mapping[str, object],
    ) -> Mapping[str, object]:
        result = self.bindings.apply_retained_update(
            self._retained_services(),
            phase=phase,
            fragment=fragment,
            output_directory=self._phase_directory(phase),
            sleep=self.sleep,
            max_polls=self.max_polls,
        )
        if type(result) is not dict:
            raise ProductionOperationError("retained updater returned inexact evidence")
        return self._retained_transaction_evidence(
            phase=phase,
            source_fragment=fragment,
            result=result,
        )

    def _retained_evidence_paths(self, phase: str) -> tuple[Path, ...]:
        directory = self._phase_directory(phase)
        stem = "retained-" + phase
        return tuple(
            directory / (stem + suffix)
            for suffix in (
                "-before-template.json",
                "-after-template.json",
                "-change-set.json",
                "-readback.json",
            )
        )

    def _adopt_retained(
        self,
        *,
        phase: str,
        fragment: Mapping[str, object],
    ) -> Mapping[str, object]:
        if not all(path.exists() for path in self._retained_evidence_paths(phase)):
            raise ProductionOperationError(
                phase + " retained predeployment evidence is incomplete"
            )
        return self._retained_transaction_evidence(
            phase=phase,
            source_fragment=fragment,
        )

    def _adopt_or_apply_retained(
        self,
        *,
        phase: str,
        fragment: Mapping[str, object],
    ) -> Mapping[str, object]:
        evidence_paths = self._retained_evidence_paths(phase)
        if any(path.exists() for path in evidence_paths):
            return self._adopt_retained(phase=phase, fragment=fragment)
        return self._apply_retained(phase=phase, fragment=fragment)

    def _read_retained_update_readback(
        self,
        phase: str,
    ) -> Mapping[str, object]:
        path = self._phase_directory(phase) / f"retained-{phase}-readback.json"
        value, _raw = _canonical_mapping_file(
            path,
            phase + " retained update readback",
        )
        if (
            value.get("record_type") != "glm52_task13_retained_update_readback_v1"
            or value.get("phase") != phase
            or value.get("stack_id") != self.production_request["retained_stack_id"]
            or value.get("stack_status") != "UPDATE_COMPLETE"
            or type(value.get("outputs")) is not dict
        ):
            raise ProductionOperationError(
                phase + " retained update readback is not exact"
            )
        _sha(value.get("template_sha256"), phase + " retained template")
        return value

    def update_retained_pre_support(self, request, committed):
        self._assert_request(request)
        if self._pre_fragment is None:
            self.materialize_pre_support(request, committed)
        assert self._pre_fragment is not None
        return self._apply_retained(
            phase="pre-support",
            fragment=self._pre_fragment,
        )

    def _support_input_services(self):
        from .task13_support_input_materialization import (
            SupportInputServices,
        )

        return SupportInputServices(
            sts=self.services.sts,
            organizations=self.services.organizations,
            cloudformation=self.services.cloudformation,
            ec2=self.services.ec2,
            kms=self.services.kms,
            dynamodb=self.services.dynamodb,
            s3=self.services.s3,
            lambda_client=self.services.lambda_client,
            total_max_attempts=1,
            remaining_time_in_millis=lambda: 540_000,
        )

    def _cryptography_layer_matches(
        self,
        publication: Mapping[str, object],
    ) -> list[Mapping[str, object]]:
        layer = publication["cryptography_layer_archive"]
        assert type(layer) is dict
        expected_description = f"Task13 {RUN_ID} cryptography " + str(
            layer["file_sha256"]
        )
        matches: list[Mapping[str, object]] = []
        marker: str | None = None
        for _page in range(64):
            request: dict[str, object] = {
                "LayerName": CRYPTOGRAPHY_LAYER_NAME,
                "CompatibleRuntime": "python3.12",
                "CompatibleArchitecture": "x86_64",
                "MaxItems": 50,
            }
            if marker is not None:
                request["Marker"] = marker
            response = _call(
                self.services.lambda_client,
                "list_layer_versions",
                operation="ListCryptographyLayerVersions",
                **request,
            )
            rows = response.get("LayerVersions")
            if type(rows) is not list:
                raise ProductionOperationError(
                    "cryptography layer inventory is malformed"
                )
            for row in rows:
                if (
                    type(row) is not dict
                    or row.get("Description") != expected_description
                ):
                    continue
                arn = row.get("LayerVersionArn")
                if type(arn) is not str or not arn:
                    raise ProductionOperationError(
                        "cryptography layer version ARN is malformed"
                    )
                detail = _call(
                    self.services.lambda_client,
                    "get_layer_version_by_arn",
                    operation="GetCryptographyLayerVersion",
                    Arn=arn,
                )
                content = detail.get("Content")
                expected_code = base64.b64encode(
                    bytes.fromhex(str(layer["file_sha256"]))
                ).decode("ascii")
                if (
                    detail.get("LayerVersionArn") != arn
                    or detail.get("Version") != row.get("Version")
                    or detail.get("Description") != expected_description
                    or detail.get("CompatibleRuntimes") != ["python3.12"]
                    or detail.get("CompatibleArchitectures") != ["x86_64"]
                    or type(content) is not dict
                    or content.get("CodeSha256") != expected_code
                    or content.get("CodeSize") != layer["size_bytes"]
                ):
                    raise ProductionOperationError(
                        "cryptography layer readback drifted"
                    )
                matches.append(
                    {
                        "version_arn": arn,
                        "version": detail["Version"],
                        "code_sha256": expected_code,
                    }
                )
            next_marker = response.get("NextMarker")
            if next_marker is None:
                break
            if type(next_marker) is not str or not next_marker:
                raise ProductionOperationError(
                    "cryptography layer pagination is invalid"
                )
            marker = next_marker
        else:
            raise ProductionOperationError(
                "cryptography layer pagination exceeded its bound"
            )
        return matches

    def _support_runtime_coordinates(
        self,
        *,
        allow_publish: bool,
    ) -> Mapping[str, object]:
        from .task13_support_artifacts import (
            validate_support_artifact_publication,
        )

        publication = validate_support_artifact_publication(
            self.production_request["support_artifact_publication"]
        )
        matches = self._cryptography_layer_matches(publication)
        if len(matches) > 1:
            raise ProductionOperationError(
                "cryptography layer has ambiguous duplicate versions"
            )
        if not matches:
            if not allow_publish:
                raise ProductionOperationError(
                    "cryptography layer mutation has no exact readback"
                )
            layer = publication["cryptography_layer_archive"]
            assert type(layer) is dict
            try:
                _call(
                    self.services.lambda_client,
                    "publish_layer_version",
                    operation="PublishCryptographyLayerVersion",
                    LayerName=CRYPTOGRAPHY_LAYER_NAME,
                    Description=(
                        f"Task13 {RUN_ID} cryptography " + str(layer["file_sha256"])
                    ),
                    Content={
                        "S3Bucket": layer["bucket"],
                        "S3Key": layer["key"],
                        "S3ObjectVersion": layer["version_id"],
                    },
                    CompatibleRuntimes=["python3.12"],
                    CompatibleArchitectures=["x86_64"],
                )
            except Exception:
                pass
            matches = self._cryptography_layer_matches(publication)
            if len(matches) != 1:
                raise ProductionOperationError(
                    "ambiguous layer publication lacks singular readback"
                )
        match = matches[0]
        evidence = {
            "support_lambda_archive": publication["support_lambda_archive"],
            "cryptography_layer_archive": publication["cryptography_layer_archive"],
            "cryptography_layer_version_arn": match["version_arn"],
            "cryptography_layer_code_sha256": match["code_sha256"],
        }
        self._support_runtime_evidence = _copy(
            evidence,
            "support runtime artifact evidence",
        )
        return self._support_runtime_evidence

    def _materialize_full_support_inputs(
        self,
        request,
        committed,
        *,
        allow_layer_publish: bool,
    ):
        del committed
        self._assert_request(request)
        runtime = self._support_runtime_coordinates(
            allow_publish=allow_layer_publish,
        )
        materialization_request = _copy(
            self.production_request["support_input_materialization_request"],
            "support input materialization request",
        )
        support_archive = runtime["support_lambda_archive"]
        assert type(support_archive) is dict
        materialization_request.update(
            {
                "cryptography_layer_arn": runtime["cryptography_layer_version_arn"],
                "lambda_code_bucket": support_archive["bucket"],
                "lambda_code_key": support_archive["key"],
                "lambda_code_version_id": support_archive["version_id"],
                "lambda_code_sha256": support_archive["file_sha256"],
            }
        )
        layer_archive = runtime["cryptography_layer_archive"]
        assert type(layer_archive) is dict
        materialization_request["cryptography_layer_sha256"] = layer_archive[
            "file_sha256"
        ]
        inputs = self.bindings.collect_support_build_inputs(
            request=materialization_request,
            services=self._support_input_services(),
        )
        identity = self.bindings.support_build_inputs_identity(inputs)
        exports = getattr(inputs, "retained_export_names", None)
        if (
            type(exports) is not tuple
            or "KeepGlm52Task12TerminalV2VersionArn" not in exports
        ):
            raise ProductionOperationError(
                "full support inputs lack the retained TerminalV2 export"
            )
        price = self.bindings.parse_support_price_card(
            self.production_request["support_price_card"]
        )
        bundle = self.bindings.build_support_precreate_plane(
            inputs=inputs,
            price_card=price,
        )
        if (
            getattr(bundle, "inputs", None) is not inputs
            or type(getattr(bundle, "support_template", None)) is not dict
        ):
            raise ProductionOperationError(
                "support precreate builder returned an inexact bundle"
            )
        self._support_inputs = inputs
        self._support_bundle = bundle
        return {
            "complete": True,
            "support_inputs_sha256": _sha(
                identity,
                "support build input identity",
            ),
            "terminal_v2_export_present": True,
            **runtime,
            **_common(),
        }

    def materialize_full_support_inputs(self, request, committed):
        return self._materialize_full_support_inputs(
            request,
            committed,
            allow_layer_publish=True,
        )

    def _write_or_adopt(self, path: Path, raw: bytes) -> None:
        if path.exists():
            if path.is_symlink() or path.read_bytes() != raw:
                raise ProductionOperationError(
                    "existing materialized source is foreign"
                )
            return
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("materialized source write made no progress")
                view = view[written:]
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _reviewed_services(self):
        from .task13_reviewed_artifacts import (
            Task13ReviewedArtifactServices,
        )

        return Task13ReviewedArtifactServices(
            sts=self.services.sts,
            s3=self.services.s3,
            total_max_attempts=1,
        )

    def _fixed_artifact_services(self):
        from .task13_fixed_artifacts import Task13FixedArtifactServices

        return Task13FixedArtifactServices(
            sts=self.services.sts,
            s3=self.services.s3,
            total_max_attempts=1,
        )

    def _support_publication_evidence(
        self,
        *,
        template: object,
        inputs: object,
    ) -> Mapping[str, object]:
        template_coordinate = _coordinate(
            template,
            artifact_kind="SUPPORT_TEMPLATE",
        )
        inputs_coordinate = _coordinate(
            inputs,
            artifact_kind="SUPPORT_INPUTS",
        )
        if inputs_coordinate["key"] != SUPPORT_INPUTS_KEY:
            raise ProductionOperationError(
                "support inputs were not published at the fixed key"
            )
        self._support_template_coordinate = template_coordinate
        self._support_inputs_coordinate = inputs_coordinate
        return {
            "immutable": True,
            "support_template_version_id": template_coordinate["version_id"],
            "support_inputs_version_id": inputs_coordinate["version_id"],
            "support_template_sha256": template_coordinate["body_sha256"],
            "support_inputs_sha256": inputs_coordinate["body_sha256"],
            "support_template_bucket": template_coordinate["bucket"],
            "support_template_key": template_coordinate["key"],
            "support_template_file_sha256": template_coordinate["file_sha256"],
            "support_template_body_sha256": template_coordinate["body_sha256"],
            "support_inputs_bucket": inputs_coordinate["bucket"],
            "support_inputs_key": inputs_coordinate["key"],
            "support_inputs_file_sha256": inputs_coordinate["file_sha256"],
            "support_inputs_body_sha256": inputs_coordinate["body_sha256"],
            **_common(),
        }

    def _support_artifacts(
        self,
    ) -> tuple[Path, bytes, Path, bytes]:
        if self._support_bundle is None or self._support_inputs is None:
            raise ProductionOperationError(
                "full support inputs have not been materialized"
            )
        template = getattr(self._support_bundle, "support_template", None)
        if type(template) is not dict:
            raise ProductionOperationError("support precreate template is absent")
        template_raw = canonical_json_bytes(template) + b"\n"
        source_path = self.output_directory / "support-disabled.json"
        self._write_or_adopt(source_path, template_raw)
        projection = self.bindings.support_build_inputs_projection(self._support_inputs)
        inputs_raw = canonical_json_bytes(projection) + b"\n"
        inputs_path = self.output_directory / "support-build-inputs.json"
        self._write_or_adopt(inputs_path, inputs_raw)
        return source_path, template_raw, inputs_path, inputs_raw

    def build_publish_support(self, request, committed):
        self._assert_request(request)
        if self._support_bundle is None:
            self.materialize_full_support_inputs(request, committed)
        template_path, template_raw, inputs_path, _inputs_raw = (
            self._support_artifacts()
        )
        services = self._reviewed_services()
        template = self.bindings.publish_reviewed_artifact(
            artifact_kind="SUPPORT_TEMPLATE",
            source_path=template_path,
            expected_file_sha256=hashlib.sha256(template_raw).hexdigest(),
            expected_body_sha256=hashlib.sha256(template_raw[:-1]).hexdigest(),
            bucket=MODEL_BUCKET_NAME,
            services=services,
        )
        inputs = self.bindings.publish_support_build_inputs(
            inputs_path=inputs_path,
            bucket=MODEL_BUCKET_NAME,
            services=self._fixed_artifact_services(),
        )
        return self._support_publication_evidence(
            template=template,
            inputs=inputs,
        )

    @staticmethod
    def _stack_identity(
        *,
        kind: StackKind,
        stack: Mapping[str, object],
    ) -> StackIdentity:
        tags = stack.get("Tags")
        if type(tags) is not list:
            raise ProductionOperationError(kind.value + " stack tags are absent")
        exact_tags = tuple(
            sorted(
                (
                    str(row.get("Key")),
                    str(row.get("Value")),
                )
                for row in tags
                if type(row) is dict
            )
        )
        try:
            return StackIdentity(
                kind=kind,
                name=str(stack.get("StackName")),
                stack_id=str(stack.get("StackId")),
                account_id=ACCOUNT_ID,
                region=REGION,
                termination_protection=stack.get("EnableTerminationProtection"),
                tags=exact_tags,
            )
        except (TypeError, ValueError) as exc:
            raise ProductionOperationError(
                kind.value + " stack identity is not exact"
            ) from exc

    def _original_template(
        self,
        stack_id: str,
        *,
        label: str,
    ) -> dict[str, object]:
        response = _call(
            self.services.cloudformation,
            "get_template",
            operation="Get" + label.title().replace(" ", "") + "Template",
            StackName=stack_id,
            TemplateStage="Original",
        )
        return _template_body(
            response.get("TemplateBody"),
            label + " template",
        )

    def _direct_bucket_policy(self) -> dict[str, object]:
        response = _call(
            self.services.s3,
            "get_bucket_policy",
            operation="GetMigrationBucketPolicy",
            Bucket=MODEL_BUCKET_NAME,
            ExpectedBucketOwner=ACCOUNT_ID,
        )
        policy = response.get("Policy")
        if type(policy) is str:
            try:
                policy = json.loads(policy)
            except json.JSONDecodeError as exc:
                raise ProductionOperationError(
                    "live bucket policy is not JSON"
                ) from exc
        if type(policy) is not dict:
            raise ProductionOperationError("live bucket policy is not one exact object")
        return _copy(policy, "live bucket policy")

    def _live_export_names(self, stack_id: str) -> tuple[str, ...]:
        names: list[str] = []
        token: str | None = None
        seen: set[str] = set()
        for _page in range(64):
            request: dict[str, object] = {}
            if token is not None:
                request["NextToken"] = token
            response = _call(
                self.services.cloudformation,
                "list_exports",
                operation="ListMigrationExports",
                **request,
            )
            rows = response.get("Exports")
            if type(rows) is not list:
                raise ProductionOperationError(
                    "CloudFormation export inventory is malformed"
                )
            for row in rows:
                if (
                    type(row) is dict
                    and row.get("ExportingStackId") == stack_id
                    and type(row.get("Name")) is str
                    and row["Name"]
                ):
                    names.append(row["Name"])
            next_token = response.get("NextToken")
            if next_token is None:
                break
            if type(next_token) is not str or not next_token or next_token in seen:
                raise ProductionOperationError(
                    "CloudFormation export pagination is incomplete"
                )
            seen.add(next_token)
            token = next_token
        if len(set(names)) != len(names):
            raise ProductionOperationError("CloudFormation export names are duplicated")
        return tuple(sorted(names))

    @staticmethod
    def _reviewed_export_names(
        template: Mapping[str, object],
    ) -> tuple[str, ...]:
        outputs = template.get("Outputs", {})
        if type(outputs) is not dict:
            raise ProductionOperationError("reviewed support outputs are malformed")
        names = []
        for output in outputs.values():
            if type(output) is not dict:
                raise ProductionOperationError("reviewed support output is malformed")
            export = output.get("Export")
            if export is None:
                continue
            if (
                type(export) is not dict
                or set(export) != {"Name"}
                or type(export["Name"]) is not str
                or not export["Name"]
            ):
                raise ProductionOperationError(
                    "reviewed support export name is not literal"
                )
            names.append(export["Name"])
        if len(set(names)) != len(names):
            raise ProductionOperationError(
                "reviewed support export names are duplicated"
            )
        return tuple(sorted(names))

    def _migration_evidence(
        self,
        *,
        bootstrap: MigrationBootstrapResult,
        reviewed_support_template: Mapping[str, object],
    ) -> tuple[MigrationEvidence, Mapping[str, object]]:
        retained_stack = self._describe_stack(
            str(self.production_request["retained_stack_id"]),
            name=RETAINED_STACK_NAME,
            statuses={"UPDATE_COMPLETE"},
        )
        fence_stack = self._describe_stack(
            bootstrap.fence.stack_id,
            name=FENCE_STACK_NAME,
            statuses={"CREATE_COMPLETE"},
        )
        support_stack = self._describe_stack(
            bootstrap.support.stack_id,
            name=SUPPORT_STACK_NAME,
            statuses={"CREATE_COMPLETE"},
        )
        retained = self._stack_identity(
            kind=StackKind.RETAINED,
            stack=retained_stack,
        )
        fence = self._stack_identity(
            kind=StackKind.FENCE,
            stack=fence_stack,
        )
        support = self._stack_identity(
            kind=StackKind.SUPPORT,
            stack=support_stack,
        )
        if fence != bootstrap.fence or support != bootstrap.support:
            raise ProductionOperationError(
                "bootstrap stack identities changed before materialization"
            )
        archived_template = self._original_template(
            retained.stack_id,
            label="retained pre-support",
        )
        resources = self._resources(retained.stack_id)
        owners = [
            row
            for row in resources
            if row.get("ResourceType") == "AWS::S3::BucketPolicy"
            and row.get("PhysicalResourceId") == MODEL_BUCKET_NAME
            and type(row.get("LogicalResourceId")) is str
        ]
        if len(owners) != 1:
            raise ProductionOperationError(
                "current bucket-policy owner is not singular"
            )
        role = _call(
            self.services.iam,
            "get_role",
            operation="GetMigrationDeploymentRole",
            RoleName=SUPPORT_ROLE_ARN.rsplit("/", 1)[1],
        ).get("Role")
        seed = validate_stack_migration_seed_projection(
            self.production_request["stack_migration_seed"]
        )
        if (
            type(role) is not dict
            or role.get("Arn") != seed.authority.deployment_role_arn
            or role.get("RoleId") != seed.authority.deployment_role_id
        ):
            raise ProductionOperationError("migration deployment-role identity drifted")
        direct = tuple(self._direct_bucket_policy() for _read in range(3))
        physical_ids = tuple(
            sorted(
                {
                    str(row["PhysicalResourceId"])
                    for row in resources
                    if type(row.get("PhysicalResourceId")) is str
                    and row["PhysicalResourceId"]
                }
            )
        )
        evidence = MigrationEvidence(
            retained=retained,
            fence=fence,
            support=support,
            current_policy_logical_id=str(owners[0]["LogicalResourceId"]),
            current_policy_physical_id=MODEL_BUCKET_NAME,
            current_policy_stack_id=retained.stack_id,
            bucket_name=MODEL_BUCKET_NAME,
            import_identifier=(("Bucket", MODEL_BUCKET_NAME),),
            live_policy=direct[0],
            direct_policy_readbacks=direct,
            retained_deployment_role_id=str(role["RoleId"]),
            retained_resource_physical_ids=physical_ids,
            retained_export_names=self._live_export_names(retained.stack_id),
            support_export_names=self._reviewed_export_names(reviewed_support_template),
        )
        return evidence, archived_template

    def _publish_migration_template(
        self,
        *,
        stage: str,
        raw: bytes,
    ) -> str:
        from .task13_fixed_artifacts import (
            Task13FixedArtifactError,
            _publish_bytes,
        )

        try:
            return _publish_bytes(
                s3=self.services.s3,
                bucket=MODEL_BUCKET_NAME,
                key=MIGRATION_TEMPLATE_KEYS[stage],
                raw=raw,
                record_type=(
                    "glm52_task13_stack_migration_" + stage.replace("-", "_") + "_v1"
                ),
            )
        except (TypeError, ValueError, Task13FixedArtifactError) as exc:
            raise ProductionOperationError(
                stage + " migration template publication failed closed"
            ) from exc

    def materialize_publish_stack_migration(self, request, committed):
        """Seal five immutable templates for the reviewed seven mutations."""

        self._assert_request(request)
        if self._migration_bootstrap is None:
            raise ProductionOperationError(
                "migration bootstrap was not adopted in this process"
            )
        if self._support_bundle is None:
            raise ProductionOperationError(
                "reviewed support template is not materialized"
            )
        support_template = getattr(
            self._support_bundle,
            "support_template",
            None,
        )
        if type(support_template) is not dict:
            raise ProductionOperationError("reviewed support template is absent")
        activation = _worker_activation_evidence(support_template)
        if activation["worker_activation_enabled"] is not False:
            raise ProductionOperationError(
                "reviewed support template can activate a worker"
            )
        bootstrap_template = self._original_template(
            self._migration_bootstrap.fence.stack_id,
            label="migration bootstrap",
        )
        try:
            validate_bootstrap_template(bootstrap_template)
        except (TypeError, ValueError) as exc:
            raise ProductionOperationError(
                "migration bootstrap template drifted"
            ) from exc
        evidence, archived_template = self._migration_evidence(
            bootstrap=self._migration_bootstrap,
            reviewed_support_template=support_template,
        )
        reviewed_inventory = support_template.get("Resources")
        if type(reviewed_inventory) is not dict:
            raise ProductionOperationError("reviewed support inventory is absent")
        inventory_sha = hashlib.sha256(
            canonical_json_bytes(reviewed_inventory)
        ).hexdigest()

        def coordinates(
            versions: Mapping[str, str],
        ) -> tuple[TemplateCoordinate, ...]:
            kinds = {
                "retention-only": StackKind.RETAINED,
                "post-retain": StackKind.RETAINED,
                "fence-import": StackKind.FENCE,
                "fence-transfer": StackKind.FENCE,
                "disabled-support": StackKind.SUPPORT,
            }
            identities = {
                StackKind.RETAINED: evidence.retained,
                StackKind.FENCE: evidence.fence,
                StackKind.SUPPORT: evidence.support,
            }
            return tuple(
                TemplateCoordinate(
                    stage=stage,
                    stack_id=identities[kind].stack_id,
                    template_url=(
                        f"https://{MODEL_BUCKET_NAME}.s3.{REGION}."
                        "amazonaws.com/"
                        f"{quote(MIGRATION_TEMPLATE_KEYS[stage], safe='/')}"
                        "?versionId="
                        f"{quote(versions[stage], safe='')}"
                    ),
                    version_id=versions[stage],
                )
                for stage, kind in kinds.items()
            )

        placeholders = {stage: "prepublication-v1" for stage in MIGRATION_TEMPLATE_KEYS}
        try:
            draft = build_stack_migration(
                archived_template=archived_template,
                bootstrap_template=bootstrap_template,
                bootstrap_result=self._migration_bootstrap,
                evidence=evidence,
                template_coordinates=coordinates(placeholders),
                reviewed_support_inventory=reviewed_inventory,
                reviewed_support_inventory_sha256=inventory_sha,
            )
            versions = {
                artifact.stage: self._publish_migration_template(
                    stage=artifact.stage,
                    raw=artifact.body,
                )
                for artifact in draft.artifacts
            }
            bundle = build_stack_migration(
                archived_template=archived_template,
                bootstrap_template=bootstrap_template,
                bootstrap_result=self._migration_bootstrap,
                evidence=evidence,
                template_coordinates=coordinates(versions),
                reviewed_support_inventory=reviewed_inventory,
                reviewed_support_inventory_sha256=inventory_sha,
            )
        except (TypeError, ValueError) as exc:
            raise ProductionOperationError(
                "stack migration bundle construction failed closed"
            ) from exc
        if {artifact.stage: artifact.body for artifact in draft.artifacts} != {
            artifact.stage: artifact.body for artifact in bundle.artifacts
        }:
            raise ProductionOperationError(
                "published migration bodies changed after coordinate sealing"
            )
        runtime = self._migration_runtime
        if runtime is None:
            raise ProductionOperationError("migration durable runtime is absent")
        durable = runtime.backend.read_state(
            action_identity_sha256=(runtime.seed.authority.action_identity_sha256)
        )
        sealed = sealed_migration_projection(
            authority=runtime.seed.authority,
            evidence=evidence,
            bundle=bundle,
            bootstrap_result=self._migration_bootstrap,
            durable_state=durable["State"],
        )
        self._sealed_migration = sealed
        artifacts = {artifact.stage: artifact for artifact in bundle.artifacts}
        activation_sha = hashlib.sha256(canonical_json_bytes(activation)).hexdigest()
        return {
            "sealed_migration": sealed,
            "sealed_migration_identity_sha256": sealed["canonical_identity_sha256"],
            "action_identity_sha256": (runtime.seed.authority.action_identity_sha256),
            "fence_stack_id": self._migration_bootstrap.fence.stack_id,
            "support_stack_id": self._migration_bootstrap.support.stack_id,
            "final_fence_template_sha256": artifacts[
                "fence-transfer"
            ].template_body_sha256,
            "support_template_sha256": artifacts[
                "disabled-support"
            ].template_body_sha256,
            "worker_activation_enabled": False,
            "worker_activation_evidence_sha256": activation_sha,
            "complete": True,
            **_common(),
        }

    def execute_stack_migration(self, request, committed):
        """Execute or monotonically reconcile the exact seven operations."""

        self._assert_request(request)
        materialized = committed.get(DeploymentStep.MATERIALIZE_PUBLISH_STACK_MIGRATION)
        sealed = (
            materialized.get("sealed_migration")
            if type(materialized) is dict
            else self._sealed_migration
        )
        if type(sealed) is not dict:
            raise ProductionOperationError(
                "sealed migration is absent from committed evidence"
            )
        try:
            validated = validate_sealed_migration_projection(sealed)
            runtime = build_sealed_migration_runtime(
                cloudformation=self.services.cloudformation,
                dynamodb=self.services.dynamodb,
                value=sealed,
            )
            result = execute_sealed_migration(runtime)
        except (TypeError, ValueError, Task13MigrationAdapterError) as exc:
            raise ProductionOperationError(
                "seven-operation stack migration failed closed"
            ) from exc
        return {
            "fence_stack_id": result.fence_stack_id,
            "support_stack_id": result.support_stack_id,
            "import_change_set_id": result.import_change_set_id,
            "operation_count": result.operation_count,
            "reconciled_creates": [kind.value for kind in result.reconciled_creates],
            "direct_policy_sha256": list(result.direct_policy_sha256),
            "action_identity_sha256": (validated.authority.action_identity_sha256),
            "bundle_identity_sha256": sealed["canonical_identity_sha256"],
            "complete": True,
            **_common(),
        }

    def _retained_resources(self) -> tuple[object, ...]:
        from .task12_orphan_audit import RetainedResource

        config = self.production_request["orphan_precreate"]
        assert type(config) is dict
        rows = config.get("expected_retained")
        if type(rows) is not list:
            raise ProductionOperationError("PRECREATE retained inventory is malformed")
        try:
            return tuple(
                RetainedResource(
                    resource_type=row["resource_type"],
                    resource_id=row["resource_id"],
                    cost_class=row["cost_class"],
                )
                for row in rows
                if type(row) is dict
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionOperationError(
                "PRECREATE retained inventory is invalid"
            ) from exc

    def capture_precreate_orphan_authority(self, request, committed):
        del committed
        self._assert_request(request)
        if self._support_inputs is None:
            self.materialize_full_support_inputs(request, {})
        config = self.production_request["orphan_precreate"]
        assert type(config) is dict
        retained_kms_key_arn = getattr(
            self._support_inputs,
            "retained_kms_key_arn",
            None,
        )
        observed = self.now()
        if (
            type(observed) is not datetime
            or observed.tzinfo is None
            or observed.utcoffset() is None
        ):
            raise ProductionOperationError(
                "PRECREATE observation clock is not timezone-aware"
            )
        observed_at = (
            observed.astimezone(UTC)
            .replace(microsecond=0)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        baseline = self.bindings.capture_precreate_baseline(
            kms=self.services.kms,
            activation_id=request.activation_id,
            retained_kms_key_arn=retained_kms_key_arn,
            expected_retained=self._retained_resources(),
            observed_at=observed_at,
        )
        path = Path(str(config.get("path", "")))
        result = self.bindings.write_precreate_authority(
            path=path,
            value=baseline,
        )
        if type(result) is not dict or result.get(
            "canonical_body_sha256"
        ) != baseline.get("canonical_body_sha256"):
            raise ProductionOperationError(
                "PRECREATE authority write did not read back exactly"
            )
        canonical_json_bytes(baseline)
        self._precreate_baseline = baseline
        self._precreate_observed_at = observed_at
        return {
            "phase": "PRECREATE",
            "authority_sha256": baseline["canonical_body_sha256"],
            "complete": True,
            **_common(),
        }

    def _support_materialization_services(self):
        from .support_plane import SupportMaterializationServices

        return SupportMaterializationServices(
            cloudformation=self.services.cloudformation,
            ec2=self.services.ec2,
        )

    def _cloudtrail_create_grants(
        self,
        *,
        start: str,
        end: str,
    ) -> tuple[Mapping[str, object], ...]:
        try:
            start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ProductionOperationError(
                "orphan authority observation times are invalid"
            ) from exc
        events: list[Mapping[str, object]] = []
        token: str | None = None
        seen: set[str] = set()
        for _page in range(64):
            request: dict[str, object] = {
                "LookupAttributes": [
                    {
                        "AttributeKey": "EventName",
                        "AttributeValue": "CreateGrant",
                    }
                ],
                "StartTime": start_time,
                "EndTime": end_time,
                "MaxResults": 50,
            }
            if token is not None:
                request["NextToken"] = token
            response = _call(
                self.services.cloudtrail,
                "lookup_events",
                operation="LookupCreateGrantEvents",
                **request,
            )
            page = response.get("Events")
            if type(page) is not list or any(type(row) is not dict for row in page):
                raise ProductionOperationError(
                    "CloudTrail CreateGrant events are malformed"
                )
            events.extend(_copy(page, "CloudTrail events"))
            next_token = response.get("NextToken")
            if next_token is None:
                return tuple(events)
            if type(next_token) is not str or not next_token or next_token in seen:
                raise ProductionOperationError(
                    "CloudTrail CreateGrant pagination is incomplete"
                )
            seen.add(next_token)
            token = next_token
        raise ProductionOperationError(
            "CloudTrail CreateGrant pagination exceeded its bound"
        )

    def _direct_grant_request_ids(
        self,
        *,
        activation_id: str,
        ledger_table_name: str,
    ) -> tuple[str, ...]:
        from .dynamodb import decode_item

        prefix = f"ACTIVATION#{activation_id}#CUSTOM_RESOURCE_GRANT#"
        request_ids: list[str] = []
        key: Mapping[str, object] | None = None
        for _page in range(64):
            query: dict[str, object] = {
                "TableName": ledger_table_name,
                "KeyConditionExpression": ("PK = :pk AND begins_with(SK, :prefix)"),
                "ExpressionAttributeValues": {
                    ":pk": {"S": f"RUN#{RUN_ID}"},
                    ":prefix": {"S": prefix},
                },
                "ProjectionExpression": "SK",
                "ConsistentRead": True,
            }
            if key is not None:
                query["ExclusiveStartKey"] = key
            response = _call(
                self.services.dynamodb,
                "query",
                operation="QueryDirectGrantEvidence",
                **query,
            )
            items = response.get("Items")
            if type(items) is not list or any(type(item) is not dict for item in items):
                raise ProductionOperationError(
                    "direct-grant evidence inventory is malformed"
                )
            for item in items:
                decoded = decode_item(item)
                sk = decoded.get("SK")
                if (
                    type(sk) is not str
                    or not sk.startswith(prefix)
                    or not sk[len(prefix) :]
                ):
                    raise ProductionOperationError(
                        "direct-grant evidence key is malformed"
                    )
                request_ids.append(sk[len(prefix) :])
            next_key = response.get("LastEvaluatedKey")
            if next_key is None or next_key == {}:
                break
            if type(next_key) is not dict:
                raise ProductionOperationError(
                    "direct-grant evidence pagination is invalid"
                )
            key = next_key
        else:
            raise ProductionOperationError(
                "direct-grant evidence pagination exceeded its bound"
            )
        exact = tuple(sorted(set(request_ids)))
        if len(exact) != 1 or len(request_ids) != 1:
            raise ProductionOperationError(
                "direct-grant evidence is not singular exact"
            )
        return exact

    def materialize_postcreate_fragment(self, request, committed):
        del committed
        self._assert_request(request)
        if self._support_inputs is None or self._support_bundle is None:
            self.materialize_full_support_inputs(request, {})
        stack = self._describe_stack(
            SUPPORT_STACK_NAME,
            name=SUPPORT_STACK_NAME,
            statuses={"UPDATE_COMPLETE"},
        )
        bundle = self.bindings.coordinate_support_postcreate(
            inputs=self._support_inputs,
            template=self._support_bundle.support_template,
            support_stack_id=stack["StackId"],
            services=self._support_materialization_services(),
        )
        fragment = getattr(bundle, "postcreate_retained_fragment", None)
        complete_inputs = getattr(bundle, "inputs", None)
        if type(fragment) is not dict or complete_inputs is None:
            raise ProductionOperationError(
                "POSTCREATE coordinator returned no retained fragment"
            )
        if self._precreate_baseline is None:
            self._precreate_baseline = self._read_precreate_baseline()
        if self._precreate_observed_at is None:
            self._precreate_observed_at = str(
                self._precreate_baseline.get("observed_at", "")
            )
        ledger_table_name = complete_inputs.ledger_table_name
        request_id = self._direct_grant_request_ids(
            activation_id=request.activation_id,
            ledger_table_name=ledger_table_name,
        )[0]
        direct = self.bindings.read_direct_grant_evidence(
            dynamodb=self.services.dynamodb,
            ledger_table_name=ledger_table_name,
            activation_id=request.activation_id,
            custom_resource_request_id=request_id,
        )
        if (
            type(direct) is not dict
            or direct.get("custom_resource_request_id") != request_id
            or direct.get("activation_id") != request.activation_id
            or direct.get("closed") is not True
            or type(direct.get("observed_at")) is not str
        ):
            raise ProductionOperationError("direct-grant evidence is not exact")
        observed_at = direct["observed_at"]
        events = self._cloudtrail_create_grants(
            start=self._precreate_observed_at,
            end=observed_at,
        )
        authority = self.bindings.build_activation_orphan_authority(
            baseline=self._precreate_baseline,
            support_inputs=complete_inputs,
            direct_grant_evidence=direct,
            active_kms=self.services.kms,
            cloudtrail_events=events,
            observed_at=observed_at,
            settling_window_seconds=60,
        )
        self._postcreate_bundle = bundle
        canonical_json_bytes(authority)
        self._orphan_activation_authority = authority
        self._postcreate_observed_at = observed_at
        return {
            "physical_truth_exact": True,
            "retained_fragment_sha256": hashlib.sha256(
                canonical_json_bytes(fragment)
            ).hexdigest(),
            "support_stack_id": stack["StackId"],
            **_common(),
        }

    def update_retained_final(self, request, committed):
        self._assert_request(request)
        if self._postcreate_bundle is None:
            self.materialize_postcreate_fragment(request, committed)
        fragment = getattr(
            self._postcreate_bundle,
            "postcreate_retained_fragment",
            None,
        )
        if type(fragment) is not dict:
            raise ProductionOperationError("final retained fragment is absent")
        return self._apply_retained(phase="final", fragment=fragment)

    def establish_postpublication_authority(self, request, committed):
        self._assert_request(request)
        if self._orphan_activation_authority is None:
            self.materialize_postcreate_fragment(request, committed)
        assert self._orphan_activation_authority is not None
        if self._postcreate_observed_at is None:
            raise ProductionOperationError("POSTCREATE observation time is absent")
        complete_inputs = self._postcreate_bundle.inputs
        authority_body_sha256 = _sha(
            self._orphan_activation_authority.get("canonical_body_sha256"),
            "POSTCREATE orphan authority identity",
        )
        authority_key = (
            f"campaigns/{RUN_ID}/task12/orphans/"
            f"{request.activation_id}/{authority_body_sha256}.json"
        )
        orphan = self.bindings.publish_activation_orphan_authority(
            s3=self.services.s3,
            dynamodb=self.services.dynamodb,
            ledger_table_name=complete_inputs.ledger_table_name,
            bucket=MODEL_BUCKET_NAME,
            key=authority_key,
            authority=self._orphan_activation_authority,
            observed_at=self._postcreate_observed_at,
        )
        if (
            type(orphan) is not dict
            or orphan.get("bucket") != MODEL_BUCKET_NAME
            or orphan.get("key") != authority_key
            or orphan.get("authority_body_sha256") != authority_body_sha256
        ):
            raise ProductionOperationError(
                "POSTCREATE orphan authority publication is not exact"
            )
        authority_identity = self._postpublication_identity(authority_body_sha256)
        return {
            "phase": "POSTPUBLICATION",
            "authority_sha256": authority_identity,
            "complete": True,
            **_common(),
        }

    def _postpublication_identity(
        self,
        authority_body_sha256: str,
    ) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": ("GLM52_TASK13_POSTCREATE_ORPHAN_AUTHORITY_V1"),
                    "orphan_authority_body_sha256": _sha(
                        authority_body_sha256,
                        "orphan authority body identity",
                    ),
                }
            )
        ).hexdigest()

    def _reconcile_postpublication_authority(
        self,
        request: StagedDeploymentRequest,
    ) -> Mapping[str, object]:
        if self._orphan_activation_authority is None:
            raise ProductionOperationError(
                "postpublication orphan authority body is absent"
            )
        body_sha = _sha(
            self._orphan_activation_authority.get("canonical_body_sha256"),
            "POSTCREATE orphan authority identity",
        )
        if self._postcreate_bundle is None:
            raise ProductionOperationError("POSTCREATE support bundle is absent")
        inputs = getattr(self._postcreate_bundle, "inputs", None)
        table = getattr(inputs, "ledger_table_name", None)
        if type(table) is not str or not table:
            raise ProductionOperationError("postpublication ledger table is absent")
        coordinate = self.bindings.read_activation_orphan_coordinate(
            dynamodb=self.services.dynamodb,
            ledger_table_name=table,
            activation_id=request.activation_id,
        )
        expected_key = (
            f"campaigns/{RUN_ID}/task12/orphans/{request.activation_id}/{body_sha}.json"
        )
        if (
            type(coordinate) is not dict
            or coordinate.get("bucket") != MODEL_BUCKET_NAME
            or coordinate.get("key") != expected_key
            or coordinate.get("authority_body_sha256") != body_sha
        ):
            raise ProductionOperationError(
                "postpublication orphan coordinate is not exact"
            )
        return {
            "phase": "POSTPUBLICATION",
            "authority_sha256": self._postpublication_identity(body_sha),
            "complete": True,
            **_common(),
        }

    def _pending_change_sets(self, stack_id: str) -> int:
        pending = 0
        token: str | None = None
        seen: set[str] = set()
        for _page in range(64):
            request: dict[str, object] = {"StackName": stack_id}
            if token is not None:
                request["NextToken"] = token
            response = _call(
                self.services.cloudformation,
                "list_change_sets",
                operation="ListChangeSets",
                **request,
            )
            rows = response.get("Summaries")
            if type(rows) is not list or any(type(row) is not dict for row in rows):
                raise ProductionOperationError("change-set inventory is malformed")
            pending += sum(row.get("ExecutionStatus") == "AVAILABLE" for row in rows)
            next_token = response.get("NextToken")
            if next_token is None:
                return pending
            if type(next_token) is not str or not next_token or next_token in seen:
                raise ProductionOperationError("change-set pagination is incomplete")
            seen.add(next_token)
            token = next_token
        raise ProductionOperationError("change-set pagination exceeded its bound")

    def _stack_template_sha256(self, stack_id: str, label: str) -> str:
        response = _call(
            self.services.cloudformation,
            "get_template",
            operation="GetFinal" + label.title() + "Template",
            StackName=stack_id,
            TemplateStage="Original",
        )
        body = _template_body(
            response.get("TemplateBody"),
            label + " final Original template",
        )
        return hashlib.sha256(canonical_json_bytes(body)).hexdigest()

    def final_exact_readback(self, request, committed):
        self._assert_request(request)
        retained = self._describe_stack(
            str(self.production_request["retained_stack_id"]),
            name=RETAINED_STACK_NAME,
            statuses={"UPDATE_COMPLETE"},
        )
        fence = self._describe_stack(
            FENCE_STACK_NAME,
            name=FENCE_STACK_NAME,
            statuses={"UPDATE_COMPLETE"},
        )
        support = self._describe_stack(
            SUPPORT_STACK_NAME,
            name=SUPPORT_STACK_NAME,
            statuses={"UPDATE_COMPLETE"},
        )
        if (
            retained.get("RoleARN") != SUPPORT_ROLE_ARN
            or fence.get("RoleARN") != SUPPORT_ROLE_ARN
            or support.get("RoleARN") != SUPPORT_ROLE_ARN
            or retained.get("EnableTerminationProtection") is not True
            or fence.get("EnableTerminationProtection") is not True
            or support.get("EnableTerminationProtection") is not True
        ):
            raise ProductionOperationError(
                "final stack roles or termination protection drifted"
            )
        pending = sum(
            self._pending_change_sets(str(stack["StackId"]))
            for stack in (retained, fence, support)
        )
        if pending:
            raise ProductionOperationError(
                "final readback found an executable change set"
            )
        retained_sha = self._stack_template_sha256(
            str(retained["StackId"]),
            "retained",
        )
        fence_sha = self._stack_template_sha256(
            str(fence["StackId"]),
            "fence",
        )
        support_sha = self._stack_template_sha256(
            str(support["StackId"]),
            "support",
        )
        expected_final = committed.get(DeploymentStep.UPDATE_RETAINED_FINAL)
        expected_migration = committed.get(
            DeploymentStep.MATERIALIZE_PUBLISH_STACK_MIGRATION
        )
        if (
            type(expected_final) is not dict
            or retained_sha != expected_final.get("template_body_sha256")
            or type(expected_migration) is not dict
            or fence_sha != expected_migration.get("final_fence_template_sha256")
            or support_sha != expected_migration.get("support_template_sha256")
        ):
            raise ProductionOperationError(
                "final live templates differ from committed publications"
            )
        return {
            "exact": True,
            "retained_stack_status": "UPDATE_COMPLETE",
            "fence_stack_status": "UPDATE_COMPLETE",
            "support_stack_status": "UPDATE_COMPLETE",
            "pending_change_sets": 0,
            "retained_stack_id": retained["StackId"],
            "fence_stack_id": fence["StackId"],
            "support_stack_id": support["StackId"],
            "retained_template_sha256": retained_sha,
            "fence_template_sha256": fence_sha,
            "support_template_sha256": support_sha,
            **_common(),
        }

    def _worker_launch_rows(self, activation_id: str) -> int:
        prefix = f"ACTIVATION#{activation_id}#WORKER_LAUNCH#"
        count = 0
        key: Mapping[str, object] | None = None
        for _page in range(64):
            request: dict[str, object] = {
                "TableName": "keep-glm52-h1g-ledger-v1",
                "KeyConditionExpression": "PK = :pk AND begins_with(SK, :sk)",
                "ExpressionAttributeValues": {
                    ":pk": {"S": f"RUN#{RUN_ID}"},
                    ":sk": {"S": prefix},
                },
                "ConsistentRead": True,
                "Select": "COUNT",
            }
            if key is not None:
                request["ExclusiveStartKey"] = key
            response = _call(
                self.services.dynamodb,
                "query",
                operation="QueryWorkerLaunchRows",
                **request,
            )
            page_count = response.get("Count")
            if type(page_count) is not int or page_count < 0:
                raise ProductionOperationError(
                    "worker-launch ledger count is malformed"
                )
            count += page_count
            next_key = response.get("LastEvaluatedKey")
            if next_key is None or next_key == {}:
                return count
            if type(next_key) is not dict:
                raise ProductionOperationError(
                    "worker-launch ledger pagination is incomplete"
                )
            key = next_key
        raise ProductionOperationError(
            "worker-launch ledger pagination exceeded its bound"
        )

    def _gpu_instance_inventory(
        self,
    ) -> list[Mapping[str, object]]:
        inventory: list[Mapping[str, object]] = []
        ids: set[str] = set()
        token: str | None = None
        seen: set[str] = set()
        for _page in range(64):
            request: dict[str, object] = {
                "Filters": [
                    {
                        "Name": "instance-type",
                        "Values": ["p5.48xlarge"],
                    },
                ]
            }
            if token is not None:
                request["NextToken"] = token
            response = _call(
                self.services.ec2,
                "describe_instances",
                operation="DescribePrepackageGpuInstances",
                **request,
            )
            reservations = response.get("Reservations")
            if type(reservations) is not list:
                raise ProductionOperationError("GPU instance inventory is malformed")
            for reservation in reservations:
                instances = (
                    reservation.get("Instances") if type(reservation) is dict else None
                )
                if type(instances) is not list:
                    raise ProductionOperationError("GPU reservation is malformed")
                for instance in instances:
                    if type(instance) is not dict:
                        raise ProductionOperationError(
                            "GPU instance readback is malformed"
                        )
                    instance_id = instance.get("InstanceId")
                    state = instance.get("State")
                    tags = instance.get("Tags", [])
                    launch_time = instance.get("LaunchTime")
                    if (
                        type(instance_id) is not str
                        or _INSTANCE_ID.fullmatch(instance_id) is None
                        or instance_id in ids
                        or instance.get("InstanceType") != "p5.48xlarge"
                        or type(state) is not dict
                        or type(state.get("Name")) is not str
                        or not state["Name"]
                        or type(tags) is not list
                        or any(
                            type(tag) is not dict
                            or set(tag) != {"Key", "Value"}
                            or type(tag["Key"]) is not str
                            or not tag["Key"]
                            or type(tag["Value"]) is not str
                            for tag in tags
                        )
                        or type(launch_time) is not datetime
                        or launch_time.tzinfo is None
                        or launch_time.utcoffset() is None
                    ):
                        raise ProductionOperationError(
                            "GPU instance inventory is malformed"
                        )
                    ids.add(instance_id)
                    inventory.append(
                        {
                            "instance_id": instance_id,
                            "instance_type": "p5.48xlarge",
                            "state": state["Name"],
                            "launch_time": launch_time.astimezone(UTC)
                            .isoformat()
                            .replace("+00:00", "Z"),
                            "tags": sorted(
                                (
                                    {
                                        "key": tag["Key"],
                                        "value": tag["Value"],
                                    }
                                    for tag in tags
                                ),
                                key=lambda tag: (
                                    str(tag["key"]),
                                    str(tag["value"]),
                                ),
                            ),
                        }
                    )
            next_token = response.get("NextToken")
            if next_token is None:
                return sorted(
                    inventory,
                    key=lambda row: str(row["instance_id"]),
                )
            if type(next_token) is not str or not next_token or next_token in seen:
                raise ProductionOperationError("GPU inventory pagination is incomplete")
            seen.add(next_token)
            token = next_token
        raise ProductionOperationError("GPU inventory pagination exceeded its bound")

    @staticmethod
    def _contains_p5(value: object) -> bool:
        if value == "p5.48xlarge":
            return True
        if type(value) is dict:
            return any(
                Task13ProductionOperations._contains_p5(child)
                for child in value.values()
            )
        if type(value) is list:
            return any(
                Task13ProductionOperations._contains_p5(child) for child in value
            )
        return False

    def _raw_p5_launch_events(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Mapping[str, object]]:
        if (
            type(start_time) is not datetime
            or start_time.tzinfo is None
            or start_time.utcoffset() is None
            or type(end_time) is not datetime
            or end_time.tzinfo is None
            or end_time.utcoffset() is None
            or start_time > end_time
        ):
            raise ProductionOperationError(
                "CloudTrail launch evidence window is not exact"
            )
        normalized: list[Mapping[str, object]] = []
        event_ids: set[str] = set()
        event_names = (
            "RunInstances",
            "StartInstances",
            "RequestSpotInstances",
            "PurchaseCapacityBlock",
        )
        for event_name in event_names:
            token: str | None = None
            seen_tokens: set[str] = set()
            for _page in range(256):
                request: dict[str, object] = {
                    "LookupAttributes": [
                        {
                            "AttributeKey": "EventName",
                            "AttributeValue": event_name,
                        }
                    ],
                    "StartTime": start_time,
                    "EndTime": end_time,
                    "MaxResults": 50,
                }
                if token is not None:
                    request["NextToken"] = token
                response = _call(
                    self.services.cloudtrail,
                    "lookup_events",
                    operation="Lookup" + event_name,
                    **request,
                )
                events = response.get("Events")
                if type(events) is not list or any(
                    type(event) is not dict for event in events
                ):
                    raise ProductionOperationError(
                        "CloudTrail launch event page is malformed"
                    )
                for event in events:
                    event_id = event.get("EventId")
                    event_time = event.get("EventTime")
                    raw = event.get("CloudTrailEvent")
                    if (
                        type(event_id) is not str
                        or not event_id
                        or event_id in event_ids
                        or event.get("EventName") != event_name
                        or event.get("EventSource") != "ec2.amazonaws.com"
                        or type(event_time) is not datetime
                        or event_time.tzinfo is None
                        or event_time.utcoffset() is None
                        or not start_time <= event_time <= end_time
                        or type(raw) is not str
                    ):
                        raise ProductionOperationError(
                            "CloudTrail launch event is malformed"
                        )
                    event_ids.add(event_id)
                    try:
                        detail = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ProductionOperationError(
                            "CloudTrail launch event JSON is malformed"
                        ) from exc
                    if (
                        type(detail) is not dict
                        or detail.get("eventID") != event_id
                        or detail.get("eventName") != event_name
                        or detail.get("eventSource") != "ec2.amazonaws.com"
                        or detail.get("awsRegion") != REGION
                        or detail.get("recipientAccountId") != ACCOUNT_ID
                        or type(detail.get("requestParameters")) is not dict
                    ):
                        raise ProductionOperationError(
                            "CloudTrail launch event identity drifted"
                        )
                    if self._contains_p5(detail):
                        normalized.append(
                            {
                                "event_id": event_id,
                                "event_name": event_name,
                                "event_time": event_time.astimezone(UTC)
                                .isoformat()
                                .replace("+00:00", "Z"),
                                "cloudtrail_event_sha256": (
                                    hashlib.sha256(raw.encode("utf-8")).hexdigest()
                                ),
                            }
                        )
                next_token = response.get("NextToken")
                if next_token is None:
                    break
                if (
                    type(next_token) is not str
                    or not next_token
                    or next_token in seen_tokens
                ):
                    raise ProductionOperationError(
                        "CloudTrail launch pagination is incomplete"
                    )
                seen_tokens.add(next_token)
                token = next_token
            else:
                raise ProductionOperationError(
                    "CloudTrail launch pagination exceeded its bound"
                )
        return sorted(
            normalized,
            key=lambda row: (
                str(row["event_time"]),
                str(row["event_id"]),
            ),
        )

    @staticmethod
    def _coordinate_from_evidence(
        evidence: Mapping[str, object],
        *,
        prefix: str,
        artifact_kind: str,
    ) -> dict[str, object]:
        return _coordinate(
            {
                "artifact_kind": artifact_kind,
                "bucket": evidence.get(prefix + "_bucket"),
                "key": evidence.get(prefix + "_key"),
                "version_id": evidence.get(prefix + "_version_id"),
                "file_sha256": evidence.get(prefix + "_file_sha256"),
                "body_sha256": evidence.get(prefix + "_body_sha256"),
            },
            artifact_kind=artifact_kind,
        )

    def materialize_staged_infrastructure_evidence(
        self,
        request,
        committed,
        *,
        staged_journal_sha256,
    ):
        self._assert_request(request)
        _sha(
            staged_journal_sha256,
            "staged deployment journal identity",
        )
        if tuple(committed) != tuple(DeploymentStep):
            raise ProductionOperationError(
                "aggregate evidence requires all 14 committed steps"
            )
        try:
            if request.journal_path.is_symlink():
                raise OSError("journal is a symlink")
            journal_raw = request.journal_path.read_bytes()
        except OSError as exc:
            raise ProductionOperationError(
                "aggregate journal readback is unavailable"
            ) from exc
        if (
            not journal_raw
            or hashlib.sha256(journal_raw).hexdigest() != staged_journal_sha256
        ):
            raise ProductionOperationError("aggregate journal identity drifted")
        journal_records: list[dict[str, object]] = []
        for line in journal_raw.splitlines(keepends=True):
            try:
                record = json.loads(line.decode("ascii"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ProductionOperationError(
                    "aggregate journal record is malformed"
                ) from exc
            if type(record) is not dict or line != canonical_json_bytes(record) + b"\n":
                raise ProductionOperationError(
                    "aggregate journal record is not canonical"
                )
            journal_records.append(record)
        staged_request = {
            "schema_version": request.schema_version,
            "record_type": request.record_type,
            "activation_id": request.activation_id,
            "journal_path": str(request.journal_path),
            "production_request": request.production_request,
        }
        staged_request_identity = hashlib.sha256(
            canonical_json_bytes(staged_request)
        ).hexdigest()
        if any(
            record.get("request_identity_sha256") != staged_request_identity
            for record in journal_records
        ):
            raise ProductionOperationError("aggregate journal request ancestry drifted")
        from .task13_campaign_package import (
            validate_staged_infrastructure_evidence,
        )

        expected_artifact_kinds = set(STAGED_INFRASTRUCTURE_ARTIFACT_KEYS)
        foundation_coordinate = _coordinate(
            self.production_request["fixed_artifacts"][0],
            artifact_kind="RETAINED_FOUNDATION_TEMPLATE",
        )
        artifacts: dict[str, dict[str, object]] = {
            "RETAINED_FOUNDATION_TEMPLATE": foundation_coordinate
        }

        pre = committed[DeploymentStep.UPDATE_RETAINED_PRE_SUPPORT]
        final = committed[DeploymentStep.UPDATE_RETAINED_FINAL]
        support_publication = committed[DeploymentStep.BUILD_PUBLISH_SUPPORT]
        migration_materialization = committed[
            DeploymentStep.MATERIALIZE_PUBLISH_STACK_MIGRATION
        ]
        migration_execution = committed[DeploymentStep.EXECUTE_STACK_MIGRATION]
        sealed = validate_sealed_migration_projection(
            migration_materialization.get("sealed_migration")
        )
        migration_artifacts = {
            artifact.stage: artifact for artifact in sealed.bundle.artifacts
        }
        fence_transfer_artifact = migration_artifacts["fence-transfer"]
        fence_coordinate = {
            "artifact_kind": "FENCE_TEMPLATE",
            "bucket": MODEL_BUCKET_NAME,
            "key": MIGRATION_TEMPLATE_KEYS["fence-transfer"],
            "version_id": fence_transfer_artifact.version_id,
            "file_sha256": fence_transfer_artifact.sha256,
            "body_sha256": (fence_transfer_artifact.template_body_sha256),
        }
        dynamic = {
            "BOOTSTRAP_TEMPLATE": _coordinate(
                self.production_request["bootstrap_template"],
                artifact_kind="BOOTSTRAP_TEMPLATE",
            ),
            "FENCE_TEMPLATE": fence_coordinate,
            "RETAINED_PRE_SUPPORT_TEMPLATE": (
                self._coordinate_from_evidence(
                    pre,
                    prefix="template",
                    artifact_kind="RETAINED_PRE_SUPPORT_TEMPLATE",
                )
            ),
            "RETAINED_TEMPLATE": self._coordinate_from_evidence(
                final,
                prefix="template",
                artifact_kind="RETAINED_TEMPLATE",
            ),
            "SUPPORT_TEMPLATE": self._coordinate_from_evidence(
                support_publication,
                prefix="support_template",
                artifact_kind="SUPPORT_TEMPLATE",
            ),
            "SUPPORT_INPUTS": self._coordinate_from_evidence(
                support_publication,
                prefix="support_inputs",
                artifact_kind="SUPPORT_INPUTS",
            ),
        }
        artifacts.update(dynamic)
        if set(artifacts) != expected_artifact_kinds:
            raise ProductionOperationError(
                "aggregate fixed-artifact inventory is incomplete"
            )
        fixed = [artifacts[kind] for kind in sorted(artifacts)]

        foundation = _copy(
            self.production_request["retained_foundation_evidence"],
            "retained foundation aggregate evidence",
        )
        aggregate_foundation_coordinate = _coordinate(
            foundation.get("template_coordinate"),
            artifact_kind="RETAINED_FOUNDATION_TEMPLATE",
        )
        if (
            set(foundation)
            != {
                "stack_id",
                "stack_status",
                "change_set_id",
                "template_coordinate",
                "template_body_sha256",
                "readback_sha256",
            }
            or foundation.get("stack_id")
            != self.production_request["retained_stack_id"]
            or foundation.get("stack_status") != "UPDATE_COMPLETE"
            or foundation.get("template_body_sha256")
            != aggregate_foundation_coordinate["body_sha256"]
            or aggregate_foundation_coordinate
            != artifacts["RETAINED_FOUNDATION_TEMPLATE"]
        ):
            raise ProductionOperationError(
                "retained foundation aggregate evidence drifted"
            )
        _sha(
            foundation.get("readback_sha256"),
            "retained foundation readback identity",
        )

        support_materialization = committed[
            DeploymentStep.MATERIALIZE_FULL_SUPPORT_INPUTS
        ]
        postpublication = committed[DeploymentStep.POSTPUBLICATION_AUTHORITY]
        final_readback = committed[DeploymentStep.FINAL_EXACT_READBACK]
        no_workers = committed[DeploymentStep.PROVE_NO_WORKER_ACTIVATION]
        body: dict[str, object] = {
            "schema_version": 1,
            "record_type": ("glm52_task13_staged_infrastructure_evidence_v1"),
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "profile": PROFILE,
            "run_id": RUN_ID,
            "activation_id": request.activation_id,
            "staged_request": staged_request,
            "staged_request_identity_sha256": staged_request_identity,
            "staged_journal_path": str(request.journal_path),
            "staged_journal_size_bytes": len(journal_raw),
            "staged_journal_records": journal_records,
            "staged_journal_sha256": staged_journal_sha256,
            "completed_steps": [step.value for step in DeploymentStep],
            "retained_foundation": foundation,
            "fence": {
                "stack_id": migration_execution["fence_stack_id"],
                "stack_status": "UPDATE_COMPLETE",
                "change_set_type": "IMPORT",
                "policy_logical_id": ("H1gProductionFenceBucketPolicy"),
                "template_coordinate": artifacts["FENCE_TEMPLATE"],
                "template_body_sha256": artifacts["FENCE_TEMPLATE"]["body_sha256"],
                "readback_sha256": hashlib.sha256(
                    canonical_json_bytes(migration_execution)
                ).hexdigest(),
            },
            "pre_support": {
                "stack_id": pre["stack_id"],
                "stack_status": pre["stack_status"],
                "change_set_type": pre["change_set_type"],
                "role_arn": pre["role_arn"],
                "template_coordinate": artifacts["RETAINED_PRE_SUPPORT_TEMPLATE"],
                "retained_fragment_sha256": pre["source_fragment_sha256"],
                "readback_sha256": pre["readback_sha256"],
            },
            "support_stack": {
                "stack_id": migration_execution["support_stack_id"],
                "stack_status": "UPDATE_COMPLETE",
                "change_set_type": "UPDATE",
                "worker_activation_enabled": False,
                "template_coordinate": artifacts["SUPPORT_TEMPLATE"],
                "inputs_coordinate": artifacts["SUPPORT_INPUTS"],
                "support_lambda_archive": support_materialization[
                    "support_lambda_archive"
                ],
                "cryptography_layer_archive": support_materialization[
                    "cryptography_layer_archive"
                ],
                "cryptography_layer_version_arn": (
                    support_materialization["cryptography_layer_version_arn"]
                ),
                "cryptography_layer_code_sha256": (
                    support_materialization["cryptography_layer_code_sha256"]
                ),
                "readback_sha256": hashlib.sha256(
                    canonical_json_bytes(migration_execution)
                ).hexdigest(),
            },
            "postcreate_final": {
                "retained_stack_id": final["stack_id"],
                "support_stack_id": migration_execution["support_stack_id"],
                "stack_status": final["stack_status"],
                "change_set_type": final["change_set_type"],
                "role_arn": final["role_arn"],
                "template_coordinate": artifacts["RETAINED_TEMPLATE"],
                "retained_fragment_sha256": final["source_fragment_sha256"],
                "postpublication_authority_sha256": postpublication["authority_sha256"],
                "readback_sha256": final["readback_sha256"],
            },
            "fixed_artifacts": fixed,
            "fixed_artifacts_identity_sha256": hashlib.sha256(
                canonical_json_bytes(fixed)
            ).hexdigest(),
            "final_readback": {
                "retained_stack_id": final_readback["retained_stack_id"],
                "fence_stack_id": final_readback["fence_stack_id"],
                "support_stack_id": final_readback["support_stack_id"],
                "retained_stack_status": final_readback["retained_stack_status"],
                "fence_stack_status": final_readback["fence_stack_status"],
                "support_stack_status": final_readback["support_stack_status"],
                "pending_change_sets": final_readback["pending_change_sets"],
                "retained_template_sha256": final_readback["retained_template_sha256"],
                "fence_template_sha256": final_readback["fence_template_sha256"],
                "support_template_sha256": final_readback["support_template_sha256"],
                "active_p5_instance_ids": no_workers["gpu_instance_ids"],
                "worker_activation_attempts": no_workers["worker_activation_attempts"],
                "raw_ec2_launch_calls": no_workers["raw_ec2_launch_calls"],
            },
        }
        body["canonical_identity_sha256"] = hashlib.sha256(
            canonical_json_bytes(body)
        ).hexdigest()
        validate_staged_infrastructure_evidence(
            body,
            activation_id=request.activation_id,
            retained_stack_id=str(self.production_request["retained_stack_id"]),
            artifacts=artifacts,
        )
        path = self.output_directory / ("staged-infrastructure-evidence-v1.json")
        self._write_or_adopt(path, canonical_json_bytes(body) + b"\n")
        return {
            "path": str(path),
            "canonical_identity_sha256": body["canonical_identity_sha256"],
        }

    def _read_precreate_baseline(self) -> Mapping[str, object]:
        config = self.production_request["orphan_precreate"]
        assert type(config) is dict
        path = Path(str(config.get("path", "")))
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductionOperationError(
                "PRECREATE orphan baseline is unavailable"
            ) from exc
        if (
            type(value) is not dict
            or value.get("phase") != "PRECREATE"
            or type(value.get("canonical_body_sha256")) is not str
        ):
            raise ProductionOperationError("PRECREATE orphan baseline is malformed")
        if raw != canonical_json_bytes(value):
            raise ProductionOperationError("PRECREATE orphan baseline is not canonical")
        identity_body = dict(value)
        identity = identity_body.pop("canonical_body_sha256")
        if hashlib.sha256(canonical_json_bytes(identity_body)).hexdigest() != identity:
            raise ProductionOperationError("PRECREATE orphan baseline identity drifted")
        normalized = dict(value)
        observed_at = normalized.get("observed_at")
        if type(observed_at) is not str or not observed_at.endswith("Z"):
            raise ProductionOperationError(
                "PRECREATE orphan observation time is malformed"
            )
        self._precreate_observed_at = observed_at
        for field in (
            "expected_retained",
            "baseline_grants",
            "baseline_typed_grants",
        ):
            rows = normalized.get(field)
            if type(rows) is list:
                normalized[field] = tuple(
                    _copy(row, "PRECREATE " + field + " row") for row in rows
                )
        return normalized

    def _find_exact_version(
        self,
        *,
        key: str,
        raw: bytes,
    ) -> str:
        versions: list[str] = []
        key_marker: str | None = None
        version_marker: str | None = None
        for _page in range(64):
            request: dict[str, object] = {
                "Bucket": MODEL_BUCKET_NAME,
                "Prefix": key,
                "ExpectedBucketOwner": ACCOUNT_ID,
            }
            if key_marker is not None:
                request["KeyMarker"] = key_marker
                request["VersionIdMarker"] = version_marker
            response = _call(
                self.services.s3,
                "list_object_versions",
                operation="ListSupportArtifactVersions",
                **request,
            )
            for row in response.get("Versions", []):
                if (
                    type(row) is dict
                    and row.get("Key") == key
                    and type(row.get("VersionId")) is str
                ):
                    candidate = _call(
                        self.services.s3,
                        "get_object",
                        operation="GetSupportArtifactVersion",
                        Bucket=MODEL_BUCKET_NAME,
                        Key=key,
                        VersionId=row["VersionId"],
                        ExpectedBucketOwner=ACCOUNT_ID,
                        ChecksumMode="ENABLED",
                    )
                    body = candidate.get("Body")
                    candidate_raw = body.read() if hasattr(body, "read") else None
                    if candidate_raw == raw:
                        versions.append(row["VersionId"])
            if response.get("IsTruncated") is not True:
                break
            key_marker = response.get("NextKeyMarker")
            version_marker = response.get("NextVersionIdMarker")
            if (
                type(key_marker) is not str
                or not key_marker
                or type(version_marker) is not str
                or not version_marker
            ):
                raise ProductionOperationError(
                    "support artifact pagination is incomplete"
                )
        if len(versions) != 1:
            raise ProductionOperationError(
                "ambiguous support publication lacks singular exact version"
            )
        return versions[0]

    def _reconcile_support_publication(self, request, committed):
        if self._support_bundle is None:
            self._materialize_full_support_inputs(
                request,
                committed,
                allow_layer_publish=False,
            )
        template_path, template_raw, _inputs_path, inputs_raw = (
            self._support_artifacts()
        )
        del template_path
        from .task13_reviewed_artifacts import REVIEWED_ARTIFACT_KEYS

        template_version = self._find_exact_version(
            key=REVIEWED_ARTIFACT_KEYS["SUPPORT_TEMPLATE"],
            raw=template_raw,
        )
        inputs_version = self._find_exact_version(
            key=SUPPORT_INPUTS_KEY,
            raw=inputs_raw,
        )
        template_coordinate = {
            "artifact_kind": "SUPPORT_TEMPLATE",
            "bucket": MODEL_BUCKET_NAME,
            "key": REVIEWED_ARTIFACT_KEYS["SUPPORT_TEMPLATE"],
            "version_id": template_version,
            "file_sha256": hashlib.sha256(template_raw).hexdigest(),
            "body_sha256": hashlib.sha256(template_raw[:-1]).hexdigest(),
        }
        inputs_coordinate = {
            "artifact_kind": "SUPPORT_INPUTS",
            "bucket": MODEL_BUCKET_NAME,
            "key": SUPPORT_INPUTS_KEY,
            "version_id": inputs_version,
            "file_sha256": hashlib.sha256(inputs_raw).hexdigest(),
            "body_sha256": hashlib.sha256(inputs_raw[:-1]).hexdigest(),
        }
        return self._support_publication_evidence(
            template=template_coordinate,
            inputs=inputs_coordinate,
        )

    def deploy_retained_bootstrap_runtime(self, request):
        self._assert_request(request)
        self._caller()
        inputs = self.bindings.parse_retained_bootstrap_inputs(
            self.production_request["retained_bootstrap_runtime_deployment"]
        )
        fragment = self.bindings.build_retained_bootstrap_fragment(inputs)
        self._adopt_or_apply_retained(
            phase="fence-bootstrap-v9",
            fragment=fragment,
        )
        return read_retained_bootstrap_runtime_deployment_v2(
            request=self.production_request["retained_bootstrap_runtime_deployment"],
            services=self.services,
        )

    def publish_bridge_seed(self, request, bootstrap_runtime):
        self._assert_request(request)
        self._caller()
        from .fence_bootstrap_publication import (
            materialize_bridge_seed_publication_request_v2,
        )

        seed_request = materialize_bridge_seed_publication_request_v2(
            authority=self.production_request["bridge_seed_publication"],
            materializer_function_version_arn=bootstrap_runtime[
                "materializer_function_version_arn"
            ],
        )
        return self.bindings.invoke_bootstrap_materializer_v2(
            event={
                "phase": "BRIDGE_SEED",
                "request": seed_request,
            },
            runtime=bootstrap_runtime,
            services=self.services,
        )

    def establish_bridge_seed(self, request, bridge_seed_publication):
        self._assert_request(request)
        self._caller()
        from .fence_bootstrap_publication import (
            BridgeSeedPublicationV2,
        )

        if type(bridge_seed_publication) is not BridgeSeedPublicationV2:
            raise ProductionOperationError(
                "bridge seed establishment publication is not exact"
            )
        materialized_request = (
            self.bindings.materialize_bridge_seed_establishment_request_v2(
                prototype=self.production_request["bridge_seed"],
                bridge_seed=bridge_seed_publication.artifact,
            )
        )
        return self.bindings.establish_bridge_seed_v2(
            request=materialized_request,
            services=self.services,
        )

    def complete_stack_migration_operations_1_to_6(self, request, bridge_seed):
        self._assert_request(request)
        self._caller()
        return self.bindings.complete_operations_1_to_6_v2(
            request=self.production_request["migration_operations_1_to_6"],
            bridge_seed=bridge_seed,
            services=self.services,
        )

    def publish_bootstrap_fence_artifacts(
        self,
        request,
        checkpoint,
        bootstrap_runtime,
    ):
        self._assert_request(request)
        self._caller()
        from .fence_bootstrap_publication import (
            materialize_bootstrap_publication_request_v2,
        )

        publication_request = materialize_bootstrap_publication_request_v2(
            authority=self.production_request["bootstrap_fence_publication"],
            materializer_function_version_arn=bootstrap_runtime[
                "materializer_function_version_arn"
            ],
            checkpoint=checkpoint,
        )
        return self.bindings.invoke_bootstrap_materializer_v2(
            event={
                "phase": "BOOTSTRAP_MANIFEST",
                "request": publication_request,
                "checkpoint": checkpoint.to_dict(),
            },
            runtime=bootstrap_runtime,
            services=self.services,
        )

    def deploy_retained_fence_runtime(
        self,
        request,
        checkpoint,
        publication,
    ):
        self._assert_request(request)
        self._caller()
        from .support_plane import materialize_retained_fence_runtime_inputs
        from .task13_migration_adapter import (
            StackMigrationTransferCheckpointV2,
        )
        from .task13_staged_deployment import BootstrapFencePublication

        if (
            type(checkpoint) is not StackMigrationTransferCheckpointV2
            or type(publication) is not BootstrapFencePublication
        ):
            raise ProductionOperationError(
                "retained fence runtime predecessors are not exact"
            )
        runtime_request = materialize_retained_fence_runtime_inputs(
            authority=self.production_request["retained_fence_runtime_deployment"],
            fence_stack_id=checkpoint.fence_stack_id,
            bootstrap_manifest_coordinate=publication.manifest_coordinate.to_dict(),
        )
        inputs = self.bindings.parse_retained_fence_runtime_inputs(
            runtime_request
        )
        if (
            inputs.fence_stack_id != checkpoint.fence_stack_id
            or dict(inputs.bootstrap_manifest_coordinate)
            != publication.manifest_coordinate.to_dict()
        ):
            raise ProductionOperationError(
                "retained fence runtime is not checkpoint and manifest bound"
            )
        fragment = self.bindings.build_retained_fence_runtime_fragment(inputs)
        self._apply_retained(
            phase="fence-runtime",
            fragment=fragment,
        )
        return read_retained_fence_runtime_deployment_v2(
            request=runtime_request,
            checkpoint=checkpoint,
            publication=publication,
            services=self.services,
        )

    def execute_prepare(self, request, checkpoint, publication):
        self._assert_request(request)
        self._caller()
        return self.bindings.execute_prepare_v2(
            request=self.production_request["prepare_execution"],
            checkpoint=checkpoint,
            publication=publication,
            services=self.services,
        )

    def collect_support_input_snapshot(
        self, request, checkpoint, prepare_result, publication
    ):
        del checkpoint, prepare_result, publication
        self._assert_request(request)
        from .task13_support_input_materialization import SupportInputServices

        return self.bindings.collect_support_build_inputs(
            request=self.production_request["support_input_materialization_request"],
            services=SupportInputServices(
                sts=self.services.sts,
                organizations=self.services.organizations,
                cloudformation=self.services.cloudformation,
                ec2=self.services.ec2,
                kms=self.services.kms,
                dynamodb=self.services.dynamodb,
                s3=self.services.s3,
                lambda_client=self.services.lambda_client,
                total_max_attempts=self.services.total_max_attempts,
            ),
        )

    def deploy_disabled_support(self, request, checkpoint, support_inputs):
        self._assert_request(request)
        self._caller()
        return self.bindings.deploy_disabled_support_v2(
            request=self.production_request["disabled_support_deployment"],
            checkpoint=checkpoint,
            support_inputs=support_inputs,
            services=self.services,
        )

    def complete_stack_migration_operation_7(
        self, request, checkpoint, prepare_result, disabled_support
    ):
        self._assert_request(request)
        self._caller()
        return self.bindings.complete_operation_7_v2(
            request=self.production_request["operation_7"],
            checkpoint=checkpoint,
            prepare_result=prepare_result,
            disabled_support=disabled_support,
            services=self.services,
        )

    def commit_support_runtime_identity(self, request, operation_7, disabled_support):
        self._assert_request(request)
        self._caller()
        return self.bindings.commit_support_runtime_identity_v2(
            request=self.production_request["support_runtime_identity"],
            operation_7=operation_7,
            disabled_support=disabled_support,
            services=self.services,
        )

    def prove_no_worker_activation(self, request, runtime_identity):
        self._assert_request(request)
        self._caller()
        return self.bindings.prove_no_launch_v2(
            request=self.production_request["no_launch_evidence"],
            runtime_identity=runtime_identity,
            services=self.services,
        )

    def reconcile_mutation(
        self,
        step,
        request,
        committed,
        possible_send_evidence,
    ):
        self._assert_request(request)
        self._caller()
        if step is DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED:
            inputs = self.bindings.parse_retained_bootstrap_inputs(
                self.production_request["retained_bootstrap_runtime_deployment"]
            )
            fragment = self.bindings.build_retained_bootstrap_fragment(inputs)
            self._adopt_retained(
                phase="fence-bootstrap-v9",
                fragment=fragment,
            )
        return self.bindings.reconcile_staged_mutation_v2(
            step=step,
            request=self.production_request,
            committed=committed,
            possible_send_evidence=possible_send_evidence,
            services=self.services,
        )

    def adopt_committed(
        self,
        step,
        request,
        evidence,
        committed,
    ):
        self._assert_request(request)
        self._caller()
        return self.bindings.adopt_staged_evidence_v2(
            step=step,
            request=self.production_request,
            evidence=evidence,
            committed=committed,
            services=self.services,
        )


def build_staged_deployment_operations(
    production_request: Mapping[str, object],
    *,
    session_factory: Callable[..., object] | None = None,
    config_factory: Callable[..., object] | None = None,
    services: ProductionServices | None = None,
    bindings: ProductionOperationBindings | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _utc_now,
    max_polls: int = 180,
) -> Task13ProductionOperations:
    """Build the concrete default CLI adapter with one-attempt AWS clients."""

    if services is None:
        if session_factory is None or config_factory is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:  # pragma: no cover - deployment host
                raise RuntimeError("boto3 and botocore are required") from exc
            session_factory = boto3.Session
            config_factory = Config
        config = config_factory(
            region_name=REGION,
            connect_timeout=5,
            read_timeout=30,
            retries={
                "mode": "standard",
                "total_max_attempts": 1,
            },
        )
        session = session_factory(
            profile_name=PROFILE,
            region_name=REGION,
        )
        if getattr(session, "region_name", None) != REGION:
            raise ProductionOperationError("AWS session region drifted")
        client = getattr(session, "client", None)
        if not callable(client):
            raise ProductionOperationError("AWS session client factory is absent")
        get_credentials = getattr(session, "get_credentials", None)
        credentials = get_credentials() if callable(get_credentials) else None
        credential_expiration = getattr(
            credentials,
            "_expiry_time",
            None,
        )
        if (
            type(credential_expiration) is not datetime
            or credential_expiration.tzinfo is None
            or credential_expiration.utcoffset() is None
        ):
            raise ProductionOperationError("AWS credential expiration is absent")
        clients = {
            name: client(name, config=config)
            for name in (
                "sts",
                "cloudformation",
                "iam",
                "s3",
                "organizations",
                "ec2",
                "ssm",
                "kms",
                "dynamodb",
                "lambda",
                "states",
                "cloudtrail",
            )
        }
        services = ProductionServices(
            sts=clients["sts"],
            cloudformation=clients["cloudformation"],
            iam=clients["iam"],
            s3=clients["s3"],
            organizations=clients["organizations"],
            ec2=clients["ec2"],
            ssm=clients["ssm"],
            kms=clients["kms"],
            dynamodb=clients["dynamodb"],
            lambda_client=clients["lambda"],
            states=clients["states"],
            cloudtrail=clients["cloudtrail"],
            total_max_attempts=1,
            credential_expiration=credential_expiration,
        )
    return Task13ProductionOperations(
        production_request=production_request,
        services=services,
        bindings=bindings if bindings is not None else _default_bindings(),
        sleep=sleep,
        now=now,
        max_polls=max_polls,
    )


__all__ = [
    "ACCOUNT_ID",
    "FENCE_ROLE_ARN",
    "MODEL_BUCKET_NAME",
    "PROFILE",
    "REGION",
    "SUPPORT_ROLE_ARN",
    "ProductionOperationBindings",
    "ProductionOperationError",
    "ProductionServices",
    "Task13ProductionOperations",
    "build_staged_deployment_operations",
    "read_retained_bootstrap_runtime_deployment_v2",
    "read_retained_fence_runtime_deployment_v2",
]
