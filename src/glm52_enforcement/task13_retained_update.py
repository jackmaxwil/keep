"""Guarded additive updates for the existing GLM-5.2 retained stack."""

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

from .support_plane import compose_complete_retained_template
from .task13_retained_foundation import FoundationError, _active_resource_types
from .task13_reviewed_artifacts import (
    RETAINED_MODELS_BUCKET,
    REVIEWED_ARTIFACT_KEYS,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RETAINED_STACK_NAME = "keep-glm52-gpu"
DEPLOYMENT_ROLE_NAME = "keep-glm52-h1g-cloudformation-deployment"
DEPLOYMENT_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/{DEPLOYMENT_ROLE_NAME}"

_PHASES: Mapping[str, Mapping[str, str]] = {
    "pre-support": {
        "artifact_kind": "RETAINED_PRE_SUPPORT_TEMPLATE",
        "change_set_name": "glm52-task13-retained-pre-support-v1",
        "key": "task13/templates/retained-pre-support.yaml",
    },
    "intentional-drift-settlement-v1": {
        "artifact_kind": "RETAINED_INTENTIONAL_DRIFT_SETTLEMENT_TEMPLATE_V1",
        "change_set_name": "glm52-task13-retained-drift-settlement-v1",
        "key": "task13/templates/retained-drift-settlement-v1.json",
    },
    "fence-bootstrap-v9": {
        "artifact_kind": "RETAINED_FENCE_BOOTSTRAP_TEMPLATE_V9",
        "change_set_name": "glm52-h1g-retained-fence-bootstrap-v9",
        "key": "task13/templates/retained-fence-bootstrap-v9.json",
    },
    "fence-runtime": {
        "artifact_kind": "RETAINED_FENCE_RUNTIME_TEMPLATE",
        "change_set_name": "glm52-h1g-retained-fence-runtime-v2",
        "key": "task13/templates/retained-fence-runtime-v2.json",
    },
    "final": {
        "artifact_kind": "RETAINED_TEMPLATE",
        "change_set_name": "glm52-task13-retained-final-v1",
        "key": REVIEWED_ARTIFACT_KEYS["RETAINED_TEMPLATE"],
    },
}
_STACK_ID = re.compile(
    rf"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
    rf"{RETAINED_STACK_NAME}/[0-9A-Za-z-]+"
)
_CHANGE_SET_ID = re.compile(
    rf"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:changeSet/"
    r"[A-Za-z0-9][-A-Za-z0-9]*/[0-9A-Za-z-]+"
)
_VERSION_ID = re.compile(r"(?!null\Z)(?!None\Z)[\x21-\x7e]{1,1024}\Z")
_AMBIGUOUS_CODES = frozenset(
    {
        "InternalFailure",
        "PriorRequestNotComplete",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
    }
)
_RETAINED_BOOTSTRAP_PINNED_GPU_AMI_ID = "ami-02b19745d2b303803"


class RetainedUpdateError(ValueError):
    """The additive retained-stack transaction failed closed."""


@dataclass(frozen=True)
class RetainedUpdateServices:
    """Typed AWS clients configured for exactly one SDK attempt."""

    sts: object
    cloudformation: object
    ec2: object
    ssm: object
    iam: object
    s3: object
    total_max_attempts: int


def canonical_json_bytes(value: object) -> bytes:
    """Encode one JSON value in the campaign's stable ASCII form."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RetainedUpdateError("value is not canonical JSON data") from exc


def apply_retained_update(
    services: RetainedUpdateServices,
    *,
    phase: str,
    fragment: Mapping[str, object],
    output_directory: Path,
    sleep: Callable[[float], None],
    max_polls: int,
) -> dict[str, object]:
    """Apply one immutable, additive, parameterless retained-stack fragment."""

    phase_contract, paths = _guard_inputs(
        services=services,
        phase=phase,
        fragment=fragment,
        output_directory=output_directory,
        sleep=sleep,
        max_polls=max_polls,
    )
    _authenticate_caller(services.sts)
    _authenticate_deployment_role(services.iam)
    _guard_versioned_bucket(services.s3)
    bootstrap_v9 = phase == "fence-bootstrap-v9"

    initial_stack = _describe_stack(
        services.cloudformation,
        RETAINED_STACK_NAME,
        allowed_statuses=(
            {"UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE"}
            if bootstrap_v9
            else {"UPDATE_COMPLETE"}
        ),
    )
    stack_id = initial_stack["StackId"]
    initial_role = initial_stack.get("RoleARN")
    if bootstrap_v9 and initial_role is not None:
        raise RetainedUpdateError("retained bootstrap v9 requires absent stack RoleARN")
    if not bootstrap_v9 and initial_role not in {None, DEPLOYMENT_ROLE_ARN}:
        raise RetainedUpdateError(
            "retained stack RoleARN is not absent or the exact deployment role"
        )
    termination_protection = _termination_protection(initial_stack)
    before_template = _get_template(
        services.cloudformation,
        stack_name=stack_id,
    )
    live_parameters = _parameter_readback(initial_stack.get("Parameters", []))
    parameters = _previous_parameters(live_parameters)
    _validate_live_parameter_keys(
        before_template=before_template,
        parameters=parameters,
    )
    before_resources = _list_stack_resources(
        services.cloudformation,
        stack_id,
    )
    before_resource_map = _resource_map(
        before_resources,
        label="before retained resource readback",
    )

    try:
        composed_value = compose_complete_retained_template(
            before_template,
            fragment,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RetainedUpdateError(
            "retained fragment is not one canonical additive template"
        ) from exc
    if type(composed_value) is not dict:
        composed = dict(composed_value)
    else:
        composed = deepcopy(composed_value)
    fragment_resources = _fragment_resources(fragment)
    capabilities = _required_capabilities(fragment_resources)
    _guard_parameterless_composition(
        before_template=before_template,
        fragment=fragment,
        composed=composed,
    )
    _validate_before_resources(
        before_template=before_template,
        before_resources=before_resource_map,
    )
    initial_semantics = (
        _intentional_drift_semantic_snapshot(
            services=services,
            template=before_template,
            stack=initial_stack,
            resources=before_resource_map,
        )
        if bootstrap_v9
        else None
    )
    if bootstrap_v9:
        if (
            type(initial_semantics) is not dict
            or initial_semantics.get("resolved_ami_id")
            != _RETAINED_BOOTSTRAP_PINNED_GPU_AMI_ID
        ):
            raise RetainedUpdateError(
                "retained bootstrap prior GPU AMI resolution is not exact"
            )
        composed = pin_retained_bootstrap_v9_template(composed)

    before_raw = canonical_json_bytes(before_template) + b"\n"
    after_raw = canonical_json_bytes(composed) + b"\n"
    _write_once(paths["before"], before_raw)
    _write_once(paths["after"], after_raw)
    coordinate = _publish_template_once(
        s3=services.s3,
        phase_contract=phase_contract,
        raw=after_raw,
    )
    template_url = _template_url(coordinate)

    change_set_name = phase_contract["change_set_name"]
    change_set_role_arn = None if bootstrap_v9 else DEPLOYMENT_ROLE_ARN
    client_identity = {
        "phase": phase,
        "stack_id": stack_id,
        "change_set_name": change_set_name,
        "template_coordinate": coordinate,
        "role_arn": change_set_role_arn,
        "parameters": parameters,
        "capabilities": capabilities,
    }
    if bootstrap_v9:
        client_identity["initial_semantics"] = initial_semantics
    client_token = hashlib.sha256(canonical_json_bytes(client_identity)).hexdigest()
    create_request: dict[str, object] = {
        "StackName": stack_id,
        "ChangeSetName": change_set_name,
        "ChangeSetType": "UPDATE",
        "Description": (f"Task 13 {phase} additive retained-stack update"),
        "TemplateURL": template_url,
        "Parameters": parameters,
        "Capabilities": capabilities,
        "IncludeNestedStacks": False,
        "ClientToken": client_token,
    }
    if change_set_role_arn is not None:
        create_request["RoleARN"] = change_set_role_arn
    try:
        created = _call(
            services.cloudformation,
            "create_change_set",
            **create_request,
        )
        created_id = created.get("Id")
        if (
            type(created_id) is not str
            or _CHANGE_SET_ID.fullmatch(created_id) is None
            or created.get("StackId") != stack_id
        ):
            raise RetainedUpdateError("CreateChangeSet returned an inexact identity")
    except Exception as exc:
        if isinstance(exc, RetainedUpdateError) or not _ambiguous_exception(exc):
            if isinstance(exc, RetainedUpdateError):
                raise
            raise RetainedUpdateError(
                "CreateChangeSet failed before exact adoption"
            ) from exc

    change_set = _wait_for_change_set(
        services.cloudformation,
        stack_id=stack_id,
        change_set_name=change_set_name,
        sleep=sleep,
        max_polls=max_polls,
    )
    validate_retained_update_change_set(
        change_set,
        phase=phase,
        stack_id=stack_id,
        expected_resources={
            logical_id: resource["Type"]
            for logical_id, resource in fragment_resources.items()
        },
        expected_parameters=live_parameters,
        expected_gpu_ami_resolution=(
            str(initial_semantics["ssm_parameter_value"])
            if bootstrap_v9 and type(initial_semantics) is dict
            else None
        ),
    )
    change_set_id = change_set["ChangeSetId"]
    change_set_template = _get_template(
        services.cloudformation,
        change_set_name=change_set_id,
    )
    if canonical_json_bytes(change_set_template) != after_raw[:-1]:
        raise RetainedUpdateError(
            "change-set Original template differs from immutable S3 bytes"
        )

    current_stack = _describe_stack(
        services.cloudformation,
        stack_id,
        allowed_statuses={str(initial_stack["StackStatus"])},
    )
    if _initial_stack_guard(current_stack) != _initial_stack_guard(initial_stack):
        raise RetainedUpdateError(
            "retained stack identity changed before ExecuteChangeSet"
        )
    if _termination_protection(current_stack) is not termination_protection:
        raise RetainedUpdateError(
            "retained termination protection changed before execute"
        )
    current_template = _get_template(
        services.cloudformation,
        stack_name=stack_id,
    )
    if canonical_json_bytes(current_template) != before_raw[:-1]:
        raise RetainedUpdateError("retained Original template changed before execute")
    current_resources = _resource_map(
        _list_stack_resources(services.cloudformation, stack_id),
        label="pre-execute retained resource readback",
    )
    if current_resources != before_resource_map:
        raise RetainedUpdateError("retained resource identities changed before execute")
    pre_execute_semantics = (
        _intentional_drift_semantic_snapshot(
            services=services,
            template=current_template,
            stack=current_stack,
            resources=current_resources,
        )
        if bootstrap_v9
        else None
    )
    if bootstrap_v9 and pre_execute_semantics != initial_semantics:
        raise RetainedUpdateError(
            "retained bootstrap semantic values changed before execute"
        )

    change_evidence: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task13_retained_update_change_set_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "phase": phase,
        "stack_id": stack_id,
        "stack_name": RETAINED_STACK_NAME,
        "initial_role_arn": initial_role,
        "deployment_role_arn": DEPLOYMENT_ROLE_ARN,
        "termination_protection": termination_protection,
        "before_template_sha256": hashlib.sha256(before_raw[:-1]).hexdigest(),
        "after_template_sha256": hashlib.sha256(after_raw[:-1]).hexdigest(),
        "template_coordinate": coordinate,
        "template_url": template_url,
        "change_set_id": change_set_id,
        "change_set_name": change_set_name,
        "client_token": client_token,
        "parameters": parameters,
        "capabilities": capabilities,
        "changes": _normalized_changes(change_set),
    }
    if bootstrap_v9:
        change_evidence.update(
            {
                "change_set_role_arn": change_set_role_arn,
                "initial_semantics": initial_semantics,
                "pre_execute_semantics": pre_execute_semantics,
            }
        )
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
            raise RetainedUpdateError("ExecuteChangeSet failed") from exc
        execute_was_ambiguous = True
        _adopt_ambiguous_execute(
            services.cloudformation,
            stack_id=stack_id,
            change_set_id=change_set_id,
            change_set_name=change_set_name,
        )

    final_stack = _wait_for_update_complete(
        services.cloudformation,
        stack_id=stack_id,
        sleep=sleep,
        max_polls=max_polls,
    )
    if _termination_protection(final_stack) is not termination_protection:
        raise RetainedUpdateError(
            "retained termination protection changed after update"
        )
    final_template = _get_template(
        services.cloudformation,
        stack_name=stack_id,
    )
    if canonical_json_bytes(final_template) != after_raw[:-1]:
        raise RetainedUpdateError(
            "live retained Original template differs from immutable S3 bytes"
        )
    final_resources = _resource_map(
        _list_stack_resources(services.cloudformation, stack_id),
        label="final retained resource readback",
    )
    _validate_final_stack(
        initial_stack=initial_stack,
        final_stack=final_stack,
        before_resources=before_resource_map,
        final_resources=final_resources,
        fragment_resources=fragment_resources,
        fragment=fragment,
        cloudformation=services.cloudformation,
        stack_id=stack_id,
        expected_role_arn=change_set_role_arn,
        expected_parameters=_parameter_readback(change_set.get("Parameters")),
    )
    final_semantics = (
        _intentional_drift_semantic_snapshot(
            services=services,
            template=final_template,
            stack=final_stack,
            resources=final_resources,
        )
        if bootstrap_v9
        else None
    )
    if bootstrap_v9:
        validate_retained_bootstrap_v9_semantics(
            initial=initial_semantics,
            final=final_semantics,
        )
    readback: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task13_retained_update_readback_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "phase": phase,
        "stack_id": stack_id,
        "stack_name": RETAINED_STACK_NAME,
        "stack_status": "UPDATE_COMPLETE",
        "role_arn": final_stack.get("RoleARN"),
        "termination_protection": termination_protection,
        "template_sha256": hashlib.sha256(after_raw[:-1]).hexdigest(),
        "template_coordinate": coordinate,
        "before_resource_count": len(before_resource_map),
        "final_resource_count": len(final_resources),
        "added_resources": [
            {
                "logical_id": logical_id,
                "physical_id": final_resources[logical_id]["PhysicalResourceId"],
                "resource_type": final_resources[logical_id]["ResourceType"],
                "resource_status": final_resources[logical_id]["ResourceStatus"],
            }
            for logical_id in sorted(fragment_resources)
        ],
        "outputs": _outputs_by_key(final_stack.get("Outputs", [])),
        "exports": _exports_for_stack(
            services.cloudformation,
            stack_id=stack_id,
        ),
        "execute_response_was_ambiguous": execute_was_ambiguous,
    }
    if bootstrap_v9:
        readback.update(
            {
                "initial_semantics": initial_semantics,
                "final_semantics": final_semantics,
            }
        )
    _write_once(
        paths["readback"],
        canonical_json_bytes(readback) + b"\n",
    )
    return {
        "status": "UPDATE_COMPLETE",
        "stack_id": stack_id,
        "change_set_id": change_set_id,
        "phase": phase,
        "template_key": coordinate["key"],
        "template_version_id": coordinate["version_id"],
    }


_RETAINED_ROUNDTRIP_SEMANTIC_NOOPS: Mapping[
    str,
    tuple[str, str, str, str, str],
] = {
    "GpuLaunchTemplate": (
        "AWS::EC2::LaunchTemplate",
        "LaunchTemplateData",
        "/Properties/LaunchTemplateData",
        "8892463b805f7adf3b05042cee62f6a4838675933f7ca5f0f36c233de45b6e3b",
        "de795a4a649013332ccb10da08f3f9a6b4295e20e60bc93205e5e026f47db1a0",
    ),
    "SkyMustStartCancelRole": (
        "AWS::IAM::Role",
        "Policies",
        "/Properties/Policies",
        "1d47a8c5906537dd3baa0729fe9e45b3b79f11df5e07ad85ac80c01c8b45c526",
        "6d69414bef0f62fc13f9c739e0e62047dcabbedb3394cc619350738fbe38c4c0",
    ),
    "SkyPilotControllerRole": (
        "AWS::IAM::Role",
        "Policies",
        "/Properties/Policies",
        "7a77bb24011959c4dc1b961c52c046914a401a1e57435bc40c6508fb2e0139d9",
        "943de9e40066cc77ba2625a0a65fb73835850879a3ea85aab0a321e9c8b0729f",
    ),
}


def retained_bootstrap_v9_roundtrip_change_projection() -> list[dict[str, str]]:
    """Return the exact normalized semantic no-op changes permitted for v8."""

    return [
        {
            "action": "Modify",
            "logical_id": logical_id,
            "replacement": "False",
            "resource_type": expectation[0],
        }
        for logical_id, expectation in sorted(
            _RETAINED_ROUNDTRIP_SEMANTIC_NOOPS.items()
        )
    ]


def pin_retained_bootstrap_v9_template(
    template: Mapping[str, object],
) -> dict[str, object]:
    """Pin v9 to the exact live AMI while the public SSM head advances."""

    if type(template) is not dict:
        raise RetainedUpdateError("retained bootstrap v9 template is malformed")
    result = deepcopy(template)
    resources = result.get("Resources")
    launch_template = (
        resources.get("GpuLaunchTemplate") if type(resources) is dict else None
    )
    properties = (
        launch_template.get("Properties") if type(launch_template) is dict else None
    )
    launch_data = (
        properties.get("LaunchTemplateData") if type(properties) is dict else None
    )
    if (
        type(launch_template) is not dict
        or launch_template.get("Type") != "AWS::EC2::LaunchTemplate"
        or type(launch_data) is not dict
        or launch_data.get("ImageId") != {"Ref": "GpuAmiId"}
    ):
        raise RetainedUpdateError(
            "retained bootstrap v9 cannot pin the exact live GPU AMI"
        )
    launch_data["ImageId"] = _RETAINED_BOOTSTRAP_PINNED_GPU_AMI_ID
    return result


def validate_retained_bootstrap_v9_semantics(
    *,
    initial: object,
    final: object,
) -> Mapping[str, object]:
    """Require stable launch/IAM values while SSM readback advances."""

    if type(initial) is not dict or type(final) is not dict:
        raise RetainedUpdateError(
            "retained bootstrap semantic values changed after execute"
        )
    initial_values = deepcopy(initial)
    final_values = deepcopy(final)
    initial_launch_version = initial_values.pop("launch_template_version", None)
    final_launch_version = final_values.pop("launch_template_version", None)
    initial_resolved_ami = initial_values.pop("resolved_ami_id", None)
    final_resolved_ami = final_values.pop("resolved_ami_id", None)
    initial_matches = initial_values.pop("ssm_matches_resolved_ami", None)
    final_matches = final_values.pop("ssm_matches_resolved_ami", None)
    if (
        initial_resolved_ami != _RETAINED_BOOTSTRAP_PINNED_GPU_AMI_ID
        or type(initial_matches) is not bool
        or final_resolved_ami != initial_values.get("ssm_parameter_value")
        or final_matches is not True
        or initial_values != final_values
        or type(initial_launch_version) is not int
        or type(final_launch_version) is not int
        or final_launch_version
        not in {initial_launch_version, initial_launch_version + 1}
    ):
        raise RetainedUpdateError(
            "retained bootstrap semantic values changed after execute"
        )
    return deepcopy(final)


def _matches_retained_roundtrip_semantic_noop(
    logical_id: str,
    resource: Mapping[str, object],
) -> bool:
    expectation = _RETAINED_ROUNDTRIP_SEMANTIC_NOOPS.get(logical_id)
    if expectation is None:
        return False
    (
        resource_type,
        property_name,
        property_path,
        before_signature,
        after_signature,
    ) = expectation
    return (
        resource.get("Action") == "Modify"
        and resource.get("ResourceType") == resource_type
        and resource.get("Replacement") in {None, "False"}
        and resource.get("Scope") == ["Properties"]
        and resource.get("Details")
        == [
            {
                "Target": {
                    "Attribute": "Properties",
                    "Name": property_name,
                    "RequiresRecreation": "Never",
                    "Path": property_path,
                    "BeforeValue": f"(Truncated-Signature):{before_signature}",
                    "AfterValue": f"(Truncated-Signature):{after_signature}",
                    "AttributeChangeType": "Modify",
                },
                "Evaluation": "Static",
                "ChangeSource": "DirectModification",
            }
        ]
    )


def validate_retained_update_change_set(
    change_set: Mapping[str, object],
    *,
    phase: str,
    stack_id: str,
    expected_resources: Mapping[str, str],
    expected_parameters: list[dict[str, object]],
    expected_gpu_ami_resolution: str | None = None,
) -> Mapping[str, object]:
    """Accept exact additions plus v8's proven semantic round-trips."""

    phase_contract = _phase_contract(phase)
    bootstrap_v9 = phase == "fence-bootstrap-v9"
    if (
        type(change_set) is not dict
        or type(expected_resources) is not dict
        or not expected_resources
        or any(
            type(logical_id) is not str
            or not logical_id
            or type(resource_type) is not str
            or not resource_type
            for logical_id, resource_type in expected_resources.items()
        )
    ):
        raise RetainedUpdateError("retained change-set inputs are malformed")
    role_is_exact = (
        change_set.get("RoleARN") is None
        if bootstrap_v9
        else change_set.get("RoleARN") in {None, DEPLOYMENT_ROLE_ARN}
    )
    change_set_name = phase_contract["change_set_name"]
    change_set_id = change_set.get("ChangeSetId")
    if (
        change_set.get("StackId") != stack_id
        or change_set.get("StackName") != RETAINED_STACK_NAME
        or change_set.get("ChangeSetName") != change_set_name
        or change_set.get("Status") != "CREATE_COMPLETE"
        or change_set.get("ExecutionStatus") != "AVAILABLE"
        or change_set.get("Capabilities", [])
        != _required_capabilities_from_types(expected_resources)
        or not role_is_exact
        or type(change_set_id) is not str
        or _CHANGE_SET_ID.fullmatch(change_set_id) is None
        or not change_set_id.startswith(
            f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:changeSet/{change_set_name}/"
        )
    ):
        raise RetainedUpdateError(
            "retained change set identity or authority is not exact"
        )
    expected_parameter_readback = _parameter_readback(expected_parameters)
    if bootstrap_v9:
        if (
            type(expected_gpu_ami_resolution) is not str
            or re.fullmatch(r"ami-[0-9a-f]{17}", expected_gpu_ami_resolution) is None
        ):
            raise RetainedUpdateError(
                "retained bootstrap GPU AMI resolution is malformed"
            )
        gpu_rows = [
            row
            for row in expected_parameter_readback
            if row.get("ParameterKey") == "GpuAmiId"
        ]
        if (
            len(gpu_rows) != 1
            or gpu_rows[0].get("ResolvedValue") != _RETAINED_BOOTSTRAP_PINNED_GPU_AMI_ID
        ):
            raise RetainedUpdateError(
                "retained bootstrap prior GPU AMI resolution is not exact"
            )
        gpu_rows[0]["ResolvedValue"] = expected_gpu_ami_resolution
    elif expected_gpu_ami_resolution is not None:
        raise RetainedUpdateError(
            "retained bootstrap GPU AMI resolution is out of phase"
        )
    if _parameter_readback(change_set.get("Parameters")) != expected_parameter_readback:
        raise RetainedUpdateError("retained change set parameter readback is not exact")
    changes = change_set.get("Changes")
    expected_change_count = len(expected_resources) + (
        len(_RETAINED_ROUNDTRIP_SEMANTIC_NOOPS) if bootstrap_v9 else 0
    )
    if type(changes) is not list or len(changes) != expected_change_count:
        raise RetainedUpdateError(
            "retained change set does not contain the exact additions"
        )
    additions: list[object] = []
    observed_roundtrips: set[str] = set()
    for change in changes:
        resource = change.get("ResourceChange") if type(change) is dict else None
        logical_id = (
            resource.get("LogicalResourceId") if type(resource) is dict else None
        )
        if bootstrap_v9 and logical_id in _RETAINED_ROUNDTRIP_SEMANTIC_NOOPS:
            if (
                type(change) is not dict
                or change.get("Type") != "Resource"
                or type(logical_id) is not str
                or type(resource) is not dict
                or logical_id in observed_roundtrips
                or not _matches_retained_roundtrip_semantic_noop(
                    logical_id,
                    resource,
                )
            ):
                raise RetainedUpdateError(
                    "retained bootstrap semantic roundtrip is not exact"
                )
            observed_roundtrips.add(logical_id)
            continue
        additions.append(change)
    if bootstrap_v9 and observed_roundtrips != set(_RETAINED_ROUNDTRIP_SEMANTIC_NOOPS):
        raise RetainedUpdateError("retained bootstrap semantic roundtrip is not exact")
    observed: dict[str, str] = {}
    for change in additions:
        resource = change.get("ResourceChange") if type(change) is dict else None
        if (
            type(change) is not dict
            or change.get("Type") != "Resource"
            or type(resource) is not dict
            or resource.get("Action") != "Add"
            or resource.get("Replacement") not in {None, "False"}
            or type(resource.get("LogicalResourceId")) is not str
            or type(resource.get("ResourceType")) is not str
        ):
            raise RetainedUpdateError(
                "retained change set contains modification, removal, import, "
                "or replacement"
            )
        logical_id = resource["LogicalResourceId"]
        if logical_id in observed:
            raise RetainedUpdateError(
                "retained change set repeats one logical resource"
            )
        observed[logical_id] = resource["ResourceType"]
    if observed != dict(expected_resources):
        raise RetainedUpdateError(
            "retained change set additions differ from the fragment"
        )
    return deepcopy(change_set)


_INTENTIONAL_DRIFT_EXPECTATIONS: Mapping[str, Mapping[str, object]] = {
    "SkyMustStartCancelRule": {
        "resource_type": "AWS::Events::Rule",
        "status": "MODIFIED",
        "property_differences": [
            {
                "PropertyPath": "/State",
                "ExpectedValue": "ENABLED",
                "ActualValue": "DISABLED",
                "DifferenceType": "NOT_EQUAL",
            }
        ],
    },
    "SkyMustStartDeadlineSchedule": {
        "resource_type": "AWS::Scheduler::Schedule",
        "status": "DELETED",
        "property_differences": [],
    },
}
_INTENTIONAL_DRIFT_AUTO_DELETE = "SkyMustStartDeadlineScheduleAutoDelete"


def compose_intentional_drift_settlement_template(
    before_template: Mapping[str, object],
    drift_rows: object,
) -> dict[str, object]:
    """Reflect only the two observed, intentional must-start lifecycle drifts."""

    if type(before_template) is not dict or type(drift_rows) is not list:
        raise RetainedUpdateError("intentional drift settlement inputs are malformed")
    observed: dict[str, dict[str, object]] = {}
    for row in drift_rows:
        logical_id = row.get("LogicalResourceId") if type(row) is dict else None
        if (
            type(row) is not dict
            or type(logical_id) is not str
            or logical_id in observed
        ):
            raise RetainedUpdateError(
                "intentional drift settlement contains malformed drift"
            )
        observed[logical_id] = row
    if set(observed) != set(_INTENTIONAL_DRIFT_EXPECTATIONS):
        raise RetainedUpdateError(
            "intentional drift settlement differs from the exact observed lifecycle"
        )
    for logical_id, expectation in _INTENTIONAL_DRIFT_EXPECTATIONS.items():
        row = observed[logical_id]
        if (
            row.get("ResourceType") != expectation["resource_type"]
            or row.get("StackResourceDriftStatus") != expectation["status"]
            or row.get("PropertyDifferences", []) != expectation["property_differences"]
        ):
            raise RetainedUpdateError(
                "intentional drift settlement differs from the exact observed lifecycle"
            )

    after = deepcopy(before_template)
    resources = after.get("Resources")
    if type(resources) is not dict:
        raise RetainedUpdateError("intentional drift template Resources are malformed")
    cancel_rule = resources.get("SkyMustStartCancelRule")
    schedule = resources.get("SkyMustStartDeadlineSchedule")
    auto_delete = resources.get(_INTENTIONAL_DRIFT_AUTO_DELETE)
    if (
        type(cancel_rule) is not dict
        or cancel_rule.get("Type") != "AWS::Events::Rule"
        or type(cancel_rule.get("Properties")) is not dict
        or cancel_rule["Properties"].get("State") != "ENABLED"
        or type(schedule) is not dict
        or schedule.get("Type") != "AWS::Scheduler::Schedule"
        or type(auto_delete) is not dict
        or auto_delete.get("Type") != "Custom::SchedulerActionAfterCompletion"
    ):
        raise RetainedUpdateError(
            "intentional drift template does not contain the exact legacy lifecycle"
        )
    cancel_rule["Properties"]["State"] = "DISABLED"
    del resources["SkyMustStartDeadlineSchedule"]
    del resources[_INTENTIONAL_DRIFT_AUTO_DELETE]
    return after


def _resolve_intentional_semantic_value(
    value: object,
    *,
    refs: Mapping[str, object],
    get_atts: Mapping[tuple[str, str], str],
) -> object:
    if type(value) is list:
        return [
            _resolve_intentional_semantic_value(
                item,
                refs=refs,
                get_atts=get_atts,
            )
            for item in value
        ]
    if type(value) is not dict:
        return value
    if set(value) == {"Ref"}:
        target = value["Ref"]
        if type(target) is not str or target not in refs:
            raise RetainedUpdateError(
                "intentional drift semantic proof contains an unknown Ref"
            )
        return refs[target]
    if set(value) == {"Fn::GetAtt"}:
        target = value["Fn::GetAtt"]
        if (
            type(target) is not list
            or len(target) != 2
            or any(type(item) is not str for item in target)
            or (target[0], target[1]) not in get_atts
        ):
            raise RetainedUpdateError(
                "intentional drift semantic proof contains an unknown Fn::GetAtt"
            )
        return get_atts[(target[0], target[1])]
    if set(value) == {"Fn::Sub"}:
        template = value["Fn::Sub"]
        if type(template) is not str:
            raise RetainedUpdateError(
                "intentional drift semantic proof contains an unsupported Fn::Sub"
            )

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            replacement: object | None
            if "." in name:
                logical_id, attribute = name.split(".", 1)
                replacement = get_atts.get((logical_id, attribute))
            else:
                replacement = refs.get(name)
            if type(replacement) is not str:
                raise RetainedUpdateError(
                    "intentional drift semantic proof contains an unknown "
                    "Fn::Sub variable"
                )
            return replacement

        return re.sub(r"\$\{([^{}]+)\}", replace, template)
    if set(value) == {"Fn::Base64"}:
        resolved = _resolve_intentional_semantic_value(
            value["Fn::Base64"],
            refs=refs,
            get_atts=get_atts,
        )
        if type(resolved) is not str:
            raise RetainedUpdateError(
                "intentional drift semantic proof Base64 input is not text"
            )
        return base64.b64encode(resolved.encode("utf-8")).decode("ascii")
    if any(key == "Ref" or str(key).startswith("Fn::") for key in value):
        raise RetainedUpdateError(
            "intentional drift semantic proof contains an unsupported intrinsic"
        )
    return {
        str(key): _resolve_intentional_semantic_value(
            item,
            refs=refs,
            get_atts=get_atts,
        )
        for key, item in value.items()
    }


def _intentional_semantic_context(
    *,
    template: Mapping[str, object],
    stack: Mapping[str, object],
    resources: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[tuple[str, str], str], dict[str, object]]:
    definitions = template.get("Parameters")
    rows = stack.get("Parameters")
    if type(definitions) is not dict or type(rows) is not list:
        raise RetainedUpdateError(
            "intentional drift semantic proof parameters are malformed"
        )
    row_by_key: dict[str, dict[str, object]] = {}
    refs: dict[str, object] = {
        "AWS::AccountId": ACCOUNT_ID,
        "AWS::Partition": "aws",
        "AWS::Region": REGION,
    }
    for row in rows:
        key = row.get("ParameterKey") if type(row) is dict else None
        definition = definitions.get(key) if type(key) is str else None
        value = row.get("ParameterValue") if type(row) is dict else None
        parameter_type = definition.get("Type") if type(definition) is dict else None
        if (
            type(row) is not dict
            or type(key) is not str
            or not key
            or key in row_by_key
            or type(value) is not str
            or type(parameter_type) is not str
        ):
            raise RetainedUpdateError(
                "intentional drift semantic proof parameters are malformed"
            )
        row_by_key[key] = row
        if parameter_type.startswith("AWS::SSM::Parameter::Value<"):
            resolved = row.get("ResolvedValue")
            if type(resolved) is not str or not resolved:
                raise RetainedUpdateError(
                    "intentional drift semantic proof lacks an SSM resolution"
                )
            refs[key] = resolved
        elif parameter_type == "Number":
            if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value) is None:
                raise RetainedUpdateError(
                    "intentional drift semantic proof Number is not an integer"
                )
            refs[key] = int(value)
        else:
            refs[key] = value
    if set(row_by_key) != set(definitions):
        raise RetainedUpdateError(
            "intentional drift semantic proof parameter keys are not exact"
        )
    for logical_id, row in resources.items():
        physical_id = row.get("PhysicalResourceId")
        if type(physical_id) is not str or not physical_id:
            raise RetainedUpdateError(
                "intentional drift semantic proof resource identity is malformed"
            )
        refs[logical_id] = physical_id

    get_atts: dict[tuple[str, str], str] = {}
    model_bucket = resources.get("ModelBucket")
    if type(model_bucket) is dict:
        get_atts[("ModelBucket", "Arn")] = (
            f"arn:aws:s3:::{model_bucket['PhysicalResourceId']}"
        )
    dead_letter_queue = resources.get("SkyMustStartCancelDeadLetterQueue")
    if type(dead_letter_queue) is dict:
        queue_name = str(dead_letter_queue["PhysicalResourceId"]).rsplit("/", 1)[-1]
        get_atts[("SkyMustStartCancelDeadLetterQueue", "Arn")] = (
            f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:{queue_name}"
        )
    worker_role = resources.get("SkyPilotWorkerRole")
    if type(worker_role) is dict:
        get_atts[("SkyPilotWorkerRole", "Arn")] = (
            f"arn:aws:iam::{ACCOUNT_ID}:role/{worker_role['PhysicalResourceId']}"
        )
    security_group = resources.get("InstanceSecurityGroup")
    if type(security_group) is dict:
        get_atts[("InstanceSecurityGroup", "GroupId")] = str(
            security_group["PhysicalResourceId"]
        )
    instance_profile = resources.get("InstanceProfile")
    if type(instance_profile) is dict:
        get_atts[("InstanceProfile", "Arn")] = (
            f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/"
            f"{instance_profile['PhysicalResourceId']}"
        )
    return refs, get_atts, row_by_key


def _intentional_drift_semantic_snapshot(
    *,
    services: RetainedUpdateServices,
    template: Mapping[str, object],
    stack: Mapping[str, object],
    resources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    refs, get_atts, parameter_rows = _intentional_semantic_context(
        template=template,
        stack=stack,
        resources=resources,
    )
    ami_row = parameter_rows.get("GpuAmiId")
    if type(ami_row) is not dict:
        raise RetainedUpdateError("intentional drift semantic proof lacks GpuAmiId")
    parameter_name = ami_row.get("ParameterValue")
    resolved_ami = ami_row.get("ResolvedValue")
    if (
        type(parameter_name) is not str
        or not parameter_name
        or type(resolved_ami) is not str
        or re.fullmatch(r"ami-[0-9a-f]{17}", resolved_ami) is None
    ):
        raise RetainedUpdateError(
            "intentional drift semantic proof has malformed GpuAmiId"
        )
    ssm_response = _call(
        services.ssm,
        "get_parameter",
        Name=parameter_name,
        WithDecryption=False,
    )
    ssm_parameter = ssm_response.get("Parameter")
    ssm_value = ssm_parameter.get("Value") if type(ssm_parameter) is dict else None
    if (
        type(ssm_parameter) is not dict
        or ssm_parameter.get("Name") != parameter_name
        or ssm_parameter.get("Type") != "String"
        or type(ssm_value) is not str
        or re.fullmatch(r"ami-[0-9a-f]{17}", ssm_value) is None
        or type(ssm_parameter.get("Version")) is not int
        or ssm_parameter["Version"] < 1
    ):
        raise RetainedUpdateError(
            "intentional drift semantic proof SSM resolution changed"
        )

    template_resources = template.get("Resources")
    launch_resource = (
        template_resources.get("GpuLaunchTemplate")
        if type(template_resources) is dict
        else None
    )
    launch_properties = (
        launch_resource.get("Properties") if type(launch_resource) is dict else None
    )
    launch_data = (
        launch_properties.get("LaunchTemplateData")
        if type(launch_properties) is dict
        else None
    )
    launch_identity = resources.get("GpuLaunchTemplate")
    if (
        type(launch_resource) is not dict
        or launch_resource.get("Type") != "AWS::EC2::LaunchTemplate"
        or type(launch_data) is not dict
        or type(launch_identity) is not dict
    ):
        raise RetainedUpdateError(
            "intentional drift semantic proof lacks the launch template"
        )
    expected_launch_data = _resolve_intentional_semantic_value(
        launch_data,
        refs=refs,
        get_atts=get_atts,
    )
    launch_template_id = launch_identity["PhysicalResourceId"]
    launch_response = _call(
        services.ec2,
        "describe_launch_template_versions",
        LaunchTemplateId=launch_template_id,
        Versions=["$Latest"],
    )
    versions = launch_response.get("LaunchTemplateVersions")
    version = versions[0] if type(versions) is list and len(versions) == 1 else None
    if (
        type(expected_launch_data) is not dict
        or type(version) is not dict
        or version.get("LaunchTemplateId") not in {None, launch_template_id}
        or type(version.get("VersionNumber")) is not int
        or version["VersionNumber"] < 1
        or version.get("LaunchTemplateData") != expected_launch_data
    ):
        raise RetainedUpdateError(
            "intentional drift launch template semantic value changed"
        )

    policy_specs = {
        "SkyMustStartCancelRole": "sky-must-start-cancel-only",
        "SkyPilotControllerRole": "sky-controller-one-p5",
    }
    policy_snapshots: dict[str, dict[str, object]] = {}
    for logical_id, policy_name in policy_specs.items():
        role_resource = (
            template_resources.get(logical_id)
            if type(template_resources) is dict
            else None
        )
        role_properties = (
            role_resource.get("Properties") if type(role_resource) is dict else None
        )
        policies = (
            role_properties.get("Policies") if type(role_properties) is dict else None
        )
        role_identity = resources.get(logical_id)
        policy = policies[0] if type(policies) is list and len(policies) == 1 else None
        if (
            type(role_resource) is not dict
            or role_resource.get("Type") != "AWS::IAM::Role"
            or type(policy) is not dict
            or policy.get("PolicyName") != policy_name
            or type(policy.get("PolicyDocument")) is not dict
            or type(role_identity) is not dict
        ):
            raise RetainedUpdateError(
                "intentional drift IAM policy semantic input is malformed"
            )
        expected_policy = _resolve_intentional_semantic_value(
            policy["PolicyDocument"],
            refs=refs,
            get_atts=get_atts,
        )
        role_name = role_identity["PhysicalResourceId"]
        live_policy = _call(
            services.iam,
            "get_role_policy",
            RoleName=role_name,
            PolicyName=policy_name,
        )
        if (
            live_policy.get("RoleName") not in {None, role_name}
            or live_policy.get("PolicyName") not in {None, policy_name}
            or live_policy.get("PolicyDocument") != expected_policy
        ):
            raise RetainedUpdateError(
                "intentional drift IAM policy semantic value changed"
            )
        policy_snapshots[logical_id] = {
            "role_name": role_name,
            "policy_name": policy_name,
            "policy_sha256": hashlib.sha256(
                canonical_json_bytes(expected_policy)
            ).hexdigest(),
        }

    return {
        "ssm_parameter_name": parameter_name,
        "ssm_parameter_version": ssm_parameter["Version"],
        "ssm_parameter_value": ssm_value,
        "ssm_matches_resolved_ami": ssm_value == resolved_ami,
        "resolved_ami_id": resolved_ami,
        "launch_template_id": launch_template_id,
        "launch_template_version": version["VersionNumber"],
        "launch_template_data_sha256": hashlib.sha256(
            canonical_json_bytes(expected_launch_data)
        ).hexdigest(),
        "iam_policies": policy_snapshots,
    }


def read_retained_bootstrap_semantic_snapshot(
    *,
    services: RetainedUpdateServices,
    template: Mapping[str, object],
    stack: Mapping[str, object],
) -> Mapping[str, object]:
    """Read the live semantic values guarded by the v9 round-trip update."""

    stack_id = stack.get("StackId") if type(stack) is dict else None
    if (
        type(services) is not RetainedUpdateServices
        or services.total_max_attempts != 1
        or type(template) is not dict
        or type(stack_id) is not str
        or _STACK_ID.fullmatch(stack_id) is None
    ):
        raise RetainedUpdateError(
            "retained bootstrap semantic snapshot inputs are malformed"
        )
    resources = _resource_map(
        _list_stack_resources(services.cloudformation, stack_id),
        label="retained bootstrap semantic resource readback",
    )
    return _intentional_drift_semantic_snapshot(
        services=services,
        template=template,
        stack=stack,
        resources=resources,
    )


def validate_intentional_drift_settlement_change_set(
    change_set: Mapping[str, object],
    *,
    stack_id: str,
    expected_parameters: list[dict[str, object]],
) -> Mapping[str, object]:
    """Accept the intended lifecycle settlement plus proven semantic no-ops."""

    phase_contract = _phase_contract("intentional-drift-settlement-v1")
    change_set_name = phase_contract["change_set_name"]
    change_set_id = change_set.get("ChangeSetId")
    if (
        type(change_set) is not dict
        or change_set.get("StackId") != stack_id
        or change_set.get("StackName") != RETAINED_STACK_NAME
        or change_set.get("ChangeSetName") != change_set_name
        or change_set.get("Status") != "CREATE_COMPLETE"
        or change_set.get("ExecutionStatus") != "AVAILABLE"
        or change_set.get("Capabilities", []) != ["CAPABILITY_NAMED_IAM"]
        or change_set.get("RoleARN") is not None
        or type(change_set_id) is not str
        or _CHANGE_SET_ID.fullmatch(change_set_id) is None
        or not change_set_id.startswith(
            f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:changeSet/{change_set_name}/"
        )
    ):
        raise RetainedUpdateError(
            "intentional drift change set identity or authority is not exact"
        )
    if _parameter_readback(change_set.get("Parameters")) != _parameter_readback(
        expected_parameters
    ):
        raise RetainedUpdateError(
            "intentional drift change set parameter readback is not exact"
        )
    changes = change_set.get("Changes")
    if type(changes) is not list or len(changes) != 6:
        raise RetainedUpdateError(
            "intentional drift change set does not contain six exact actions"
        )
    observed: dict[str, dict[str, object]] = {}
    for change in changes:
        resource = change.get("ResourceChange") if type(change) is dict else None
        logical_id = (
            resource.get("LogicalResourceId") if type(resource) is dict else None
        )
        if (
            type(change) is not dict
            or change.get("Type") != "Resource"
            or type(resource) is not dict
            or type(logical_id) is not str
            or logical_id in observed
        ):
            raise RetainedUpdateError(
                "intentional drift change set does not contain six exact actions"
            )
        observed[logical_id] = resource
    expected_ids = {
        "SkyMustStartCancelRule",
        "SkyMustStartDeadlineSchedule",
        _INTENTIONAL_DRIFT_AUTO_DELETE,
        *_RETAINED_ROUNDTRIP_SEMANTIC_NOOPS,
    }
    if set(observed) != expected_ids:
        raise RetainedUpdateError(
            "intentional drift change set does not contain six exact actions"
        )
    for logical_id in _RETAINED_ROUNDTRIP_SEMANTIC_NOOPS:
        resource = observed[logical_id]
        if not _matches_retained_roundtrip_semantic_noop(logical_id, resource):
            raise RetainedUpdateError(
                "intentional drift change set does not contain six exact actions"
            )
    cancel = observed["SkyMustStartCancelRule"]
    expected_detail = [
        {
            "Target": {
                "Attribute": "Properties",
                "Name": "State",
                "RequiresRecreation": "Never",
                "Path": "/Properties/State",
                "BeforeValue": "ENABLED",
                "AfterValue": "DISABLED",
                "AttributeChangeType": "Modify",
            },
            "Evaluation": "Static",
            "ChangeSource": "DirectModification",
        }
    ]
    if (
        cancel.get("Action") != "Modify"
        or cancel.get("ResourceType") != "AWS::Events::Rule"
        or cancel.get("Replacement") not in {None, "False"}
        or cancel.get("Scope") != ["Properties"]
        or cancel.get("Details") != expected_detail
    ):
        raise RetainedUpdateError(
            "intentional drift change set does not contain six exact actions"
        )
    removal_types = {
        "SkyMustStartDeadlineSchedule": "AWS::Scheduler::Schedule",
        _INTENTIONAL_DRIFT_AUTO_DELETE: ("Custom::SchedulerActionAfterCompletion"),
    }
    for logical_id, resource_type in removal_types.items():
        resource = observed[logical_id]
        if (
            resource.get("Action") != "Remove"
            or resource.get("ResourceType") != resource_type
            or resource.get("Replacement") not in {None, "False"}
            or resource.get("Scope", []) != []
            or resource.get("Details", []) != []
        ):
            raise RetainedUpdateError(
                "intentional drift change set does not contain six exact actions"
            )
    return deepcopy(change_set)


def apply_intentional_drift_settlement(
    services: RetainedUpdateServices,
    *,
    output_directory: Path,
    sleep: Callable[[float], None],
    max_polls: int,
) -> dict[str, object]:
    """Settle only the disabled rule and expired one-time schedule."""

    phase = "intentional-drift-settlement-v1"
    phase_contract, paths = _guard_intentional_drift_inputs(
        services=services,
        output_directory=output_directory,
        sleep=sleep,
        max_polls=max_polls,
    )
    recovering = paths["change_set"].is_file()
    _authenticate_caller(services.sts)
    _guard_versioned_bucket(services.s3)
    initial_stack = _describe_stack(
        services.cloudformation,
        RETAINED_STACK_NAME,
        allowed_statuses={"UPDATE_COMPLETE"},
    )
    stack_id = initial_stack["StackId"]
    initial_role = initial_stack.get("RoleARN")
    if initial_role is not None:
        raise RetainedUpdateError(
            "intentional drift settlement requires absent stack RoleARN"
        )
    termination_protection = _termination_protection(initial_stack)
    before_template = _get_template(
        services.cloudformation,
        stack_name=stack_id,
    )
    live_parameters = _parameter_readback(initial_stack.get("Parameters", []))
    parameters = _previous_parameters(live_parameters)
    _validate_live_parameter_keys(
        before_template=before_template,
        parameters=parameters,
    )
    before_resources = _resource_map(
        _list_stack_resources(services.cloudformation, stack_id),
        label="before intentional drift settlement resource readback",
    )
    _validate_before_resources(
        before_template=before_template,
        before_resources=before_resources,
    )
    if recovering:
        return _recover_intentional_drift_readback(
            services=services,
            phase_contract=phase_contract,
            paths=paths,
            stack=initial_stack,
            template=before_template,
            resources=before_resources,
            termination_protection=termination_protection,
            sleep=sleep,
            max_polls=max_polls,
        )
    initial_semantics = _intentional_drift_semantic_snapshot(
        services=services,
        template=before_template,
        stack=initial_stack,
        resources=before_resources,
    )
    initial_drifts = _read_intentional_drift_rows(
        services.cloudformation,
        stack_id=stack_id,
        sleep=sleep,
        max_polls=max_polls,
        expected_stack_drift_status="DRIFTED",
    )
    after_template = compose_intentional_drift_settlement_template(
        before_template,
        initial_drifts,
    )
    before_raw = canonical_json_bytes(before_template) + b"\n"
    after_raw = canonical_json_bytes(after_template) + b"\n"
    _write_once(paths["before"], before_raw)
    _write_once(paths["after"], after_raw)
    coordinate = _publish_template_once(
        s3=services.s3,
        phase_contract=phase_contract,
        raw=after_raw,
    )
    template_url = _template_url(coordinate)
    change_set_name = phase_contract["change_set_name"]
    capabilities = ["CAPABILITY_NAMED_IAM"]
    client_token = hashlib.sha256(
        canonical_json_bytes(
            {
                "phase": phase,
                "stack_id": stack_id,
                "change_set_name": change_set_name,
                "template_coordinate": coordinate,
                "role_arn": None,
                "parameters": parameters,
                "capabilities": capabilities,
                "initial_drifts": initial_drifts,
                "initial_semantics": initial_semantics,
            }
        )
    ).hexdigest()
    create_request: dict[str, object] = {
        "StackName": stack_id,
        "ChangeSetName": change_set_name,
        "ChangeSetType": "UPDATE",
        "Description": "Task 13 exact intentional retained drift settlement",
        "TemplateURL": template_url,
        "Parameters": parameters,
        "Capabilities": capabilities,
        "IncludeNestedStacks": False,
        "ClientToken": client_token,
    }
    try:
        created = _call(
            services.cloudformation,
            "create_change_set",
            **create_request,
        )
        created_id = created.get("Id")
        if (
            type(created_id) is not str
            or _CHANGE_SET_ID.fullmatch(created_id) is None
            or created.get("StackId") != stack_id
        ):
            raise RetainedUpdateError("CreateChangeSet returned an inexact identity")
    except Exception as exc:
        if isinstance(exc, RetainedUpdateError) or not _ambiguous_exception(exc):
            if isinstance(exc, RetainedUpdateError):
                raise
            raise RetainedUpdateError(
                "CreateChangeSet failed before exact adoption"
            ) from exc

    change_set = _wait_for_change_set(
        services.cloudformation,
        stack_id=stack_id,
        change_set_name=change_set_name,
        sleep=sleep,
        max_polls=max_polls,
    )
    validate_intentional_drift_settlement_change_set(
        change_set,
        stack_id=stack_id,
        expected_parameters=live_parameters,
    )
    change_set_id = change_set["ChangeSetId"]
    change_set_template = _get_template(
        services.cloudformation,
        change_set_name=change_set_id,
    )
    if canonical_json_bytes(change_set_template) != after_raw[:-1]:
        raise RetainedUpdateError(
            "intentional drift change-set template differs from immutable S3 bytes"
        )
    current_stack = _describe_stack(
        services.cloudformation,
        stack_id,
        allowed_statuses={"UPDATE_COMPLETE"},
    )
    if _initial_stack_guard(current_stack) != _initial_stack_guard(initial_stack):
        raise RetainedUpdateError(
            "retained stack identity changed before drift settlement execute"
        )
    if _termination_protection(current_stack) is not termination_protection:
        raise RetainedUpdateError(
            "retained termination protection changed before drift settlement execute"
        )
    current_template = _get_template(
        services.cloudformation,
        stack_name=stack_id,
    )
    if canonical_json_bytes(current_template) != before_raw[:-1]:
        raise RetainedUpdateError(
            "retained Original template changed before drift settlement execute"
        )
    current_resources = _resource_map(
        _list_stack_resources(services.cloudformation, stack_id),
        label="pre-execute intentional drift resource readback",
    )
    if current_resources != before_resources:
        raise RetainedUpdateError(
            "retained resource identities changed before drift settlement execute"
        )
    current_semantics = _intentional_drift_semantic_snapshot(
        services=services,
        template=current_template,
        stack=current_stack,
        resources=current_resources,
    )
    if current_semantics != initial_semantics:
        raise RetainedUpdateError(
            "intentional drift semantic values changed before settlement execute"
        )
    current_drifts = _read_intentional_drift_rows(
        services.cloudformation,
        stack_id=stack_id,
        sleep=sleep,
        max_polls=max_polls,
        expected_stack_drift_status="DRIFTED",
    )
    if current_drifts != initial_drifts:
        raise RetainedUpdateError("intentional drift changed before settlement execute")
    change_evidence = {
        "schema_version": 1,
        "record_type": "glm52_task13_retained_drift_settlement_change_set_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "phase": phase,
        "stack_id": stack_id,
        "stack_name": RETAINED_STACK_NAME,
        "initial_role_arn": initial_role,
        "change_set_role_arn": None,
        "termination_protection": termination_protection,
        "before_template_sha256": hashlib.sha256(before_raw[:-1]).hexdigest(),
        "after_template_sha256": hashlib.sha256(after_raw[:-1]).hexdigest(),
        "template_coordinate": coordinate,
        "template_url": template_url,
        "change_set_id": change_set_id,
        "change_set_name": change_set_name,
        "client_token": client_token,
        "parameters": parameters,
        "capabilities": capabilities,
        "initial_drifts": initial_drifts,
        "initial_semantics": initial_semantics,
        "pre_execute_semantics": current_semantics,
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
            raise RetainedUpdateError(
                "intentional drift ExecuteChangeSet failed"
            ) from exc
        execute_was_ambiguous = True
        _adopt_ambiguous_execute(
            services.cloudformation,
            stack_id=stack_id,
            change_set_id=change_set_id,
            change_set_name=change_set_name,
        )
    final_stack = _wait_for_update_complete(
        services.cloudformation,
        stack_id=stack_id,
        sleep=sleep,
        max_polls=max_polls,
    )
    if _termination_protection(final_stack) is not termination_protection:
        raise RetainedUpdateError(
            "retained termination protection changed after drift settlement"
        )
    expected_final_guard = _initial_stack_guard(initial_stack)
    if _initial_stack_guard(final_stack) != expected_final_guard:
        raise RetainedUpdateError(
            "retained stack identity changed after drift settlement"
        )
    final_template = _get_template(
        services.cloudformation,
        stack_name=stack_id,
    )
    if canonical_json_bytes(final_template) != after_raw[:-1]:
        raise RetainedUpdateError(
            "live retained template differs from drift settlement S3 bytes"
        )
    final_resources = _resource_map(
        _list_stack_resources(services.cloudformation, stack_id),
        label="final intentional drift resource readback",
    )
    expected_remaining = set(before_resources) - {
        "SkyMustStartDeadlineSchedule",
        _INTENTIONAL_DRIFT_AUTO_DELETE,
    }
    if set(final_resources) != expected_remaining:
        raise RetainedUpdateError(
            "intentional drift settlement changed the retained resource set"
        )
    for logical_id in expected_remaining:
        before_resource = before_resources[logical_id]
        final_resource = final_resources[logical_id]
        if (
            final_resource["PhysicalResourceId"]
            != before_resource["PhysicalResourceId"]
            or final_resource["ResourceType"] != before_resource["ResourceType"]
        ):
            raise RetainedUpdateError(
                "intentional drift settlement changed a retained resource identity"
            )
    final_semantics = _intentional_drift_semantic_snapshot(
        services=services,
        template=final_template,
        stack=final_stack,
        resources=final_resources,
    )
    initial_semantic_values = deepcopy(initial_semantics)
    final_semantic_values = deepcopy(final_semantics)
    initial_launch_version = initial_semantic_values.pop("launch_template_version")
    final_launch_version = final_semantic_values.pop("launch_template_version")
    if (
        initial_semantic_values != final_semantic_values
        or type(initial_launch_version) is not int
        or type(final_launch_version) is not int
        or final_launch_version
        not in {initial_launch_version, initial_launch_version + 1}
    ):
        raise RetainedUpdateError(
            "intentional drift semantic values changed after settlement"
        )
    final_drifts = _read_intentional_drift_rows(
        services.cloudformation,
        stack_id=stack_id,
        sleep=sleep,
        max_polls=max_polls,
        expected_stack_drift_status="IN_SYNC",
    )
    if final_drifts:
        raise RetainedUpdateError("intentional drift remains after exact settlement")
    readback = {
        "schema_version": 1,
        "record_type": "glm52_task13_retained_drift_settlement_readback_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "phase": phase,
        "stack_id": stack_id,
        "stack_name": RETAINED_STACK_NAME,
        "stack_status": "UPDATE_COMPLETE",
        "role_arn": final_stack.get("RoleARN"),
        "termination_protection": termination_protection,
        "template_sha256": hashlib.sha256(after_raw[:-1]).hexdigest(),
        "template_coordinate": coordinate,
        "before_resource_count": len(before_resources),
        "final_resource_count": len(final_resources),
        "removed_resources": [
            "SkyMustStartDeadlineSchedule",
            _INTENTIONAL_DRIFT_AUTO_DELETE,
        ],
        "settled_resource": "SkyMustStartCancelRule",
        "initial_drifts": initial_drifts,
        "final_drifts": final_drifts,
        "initial_semantics": initial_semantics,
        "final_semantics": final_semantics,
        "execute_response_was_ambiguous": execute_was_ambiguous,
    }
    _write_once(
        paths["readback"],
        canonical_json_bytes(readback) + b"\n",
    )
    return {
        "status": "UPDATE_COMPLETE",
        "stack_id": stack_id,
        "change_set_id": change_set_id,
        "phase": phase,
        "template_key": coordinate["key"],
        "template_version_id": coordinate["version_id"],
    }


def _read_intentional_recovery_evidence(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RetainedUpdateError(
            f"intentional drift {label} evidence is unreadable"
        ) from exc
    if not raw.endswith(b"\n") or not 2 <= len(raw) <= 2_000_000:
        raise RetainedUpdateError(
            f"intentional drift {label} evidence bytes are malformed"
        )
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RetainedUpdateError(
            f"intentional drift {label} evidence is not JSON"
        ) from exc
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        raise RetainedUpdateError(
            f"intentional drift {label} evidence is not canonical"
        )
    return value, raw


def _recover_intentional_drift_readback(
    *,
    services: RetainedUpdateServices,
    phase_contract: Mapping[str, str],
    paths: Mapping[str, Path],
    stack: Mapping[str, object],
    template: Mapping[str, object],
    resources: Mapping[str, Mapping[str, object]],
    termination_protection: bool,
    sleep: Callable[[float], None],
    max_polls: int,
) -> dict[str, object]:
    before_template, before_raw = _read_intentional_recovery_evidence(
        paths["before"],
        label="before-template",
    )
    after_template, after_raw = _read_intentional_recovery_evidence(
        paths["after"],
        label="after-template",
    )
    change_evidence, _change_raw = _read_intentional_recovery_evidence(
        paths["change_set"],
        label="change-set",
    )
    initial_drifts = [
        {
            "LogicalResourceId": logical_id,
            "ResourceType": expectation["resource_type"],
            "StackResourceDriftStatus": expectation["status"],
            "PropertyDifferences": deepcopy(expectation["property_differences"]),
        }
        for logical_id, expectation in _INTENTIONAL_DRIFT_EXPECTATIONS.items()
    ]
    if (
        compose_intentional_drift_settlement_template(
            before_template,
            initial_drifts,
        )
        != after_template
        or canonical_json_bytes(template) != after_raw[:-1]
    ):
        raise RetainedUpdateError(
            "intentional drift recovery template lineage is not exact"
        )
    try:
        expected_active_resources = _active_resource_types(
            after_template,
            stack.get("Parameters", []),
        )
    except FoundationError as exc:
        raise RetainedUpdateError(
            "intentional drift recovery resource inventory is not evaluable"
        ) from exc
    actual_active_resources = {
        logical_id: resource["ResourceType"]
        for logical_id, resource in resources.items()
    }
    if actual_active_resources != expected_active_resources:
        raise RetainedUpdateError(
            "intentional drift recovery resource inventory is not exact"
        )

    coordinate = change_evidence.get("template_coordinate")
    body_sha = hashlib.sha256(after_raw[:-1]).hexdigest()
    file_sha = hashlib.sha256(after_raw).hexdigest()
    if (
        type(coordinate) is not dict
        or coordinate.get("bucket") != RETAINED_MODELS_BUCKET
        or coordinate.get("key") != phase_contract["key"]
        or coordinate.get("artifact_kind") != phase_contract["artifact_kind"]
        or coordinate.get("body_sha256") != body_sha
        or coordinate.get("file_sha256") != file_sha
        or type(coordinate.get("version_id")) is not str
        or _VERSION_ID.fullmatch(coordinate["version_id"]) is None
    ):
        raise RetainedUpdateError(
            "intentional drift recovery template coordinate is not exact"
        )
    metadata = {
        "artifact-kind": phase_contract["artifact_kind"],
        "body-sha256": body_sha,
        "file-sha256": file_sha,
        "glm52-run-id": "glm52-sky-20260724",
    }
    _read_exact_template_version(
        services.s3,
        key=phase_contract["key"],
        version_id=coordinate["version_id"],
        raw=after_raw,
        metadata=metadata,
    )

    stack_id = stack.get("StackId")
    change_set_name = phase_contract["change_set_name"]
    change_set_id = change_evidence.get("change_set_id")
    parameters = _previous_parameters(_parameter_readback(stack.get("Parameters", [])))
    initial_semantics = change_evidence.get("initial_semantics")
    capabilities = ["CAPABILITY_NAMED_IAM"]
    expected_changes = [
        {
            "action": "Modify",
            "logical_id": "GpuLaunchTemplate",
            "replacement": "False",
            "resource_type": "AWS::EC2::LaunchTemplate",
        },
        {
            "action": "Modify",
            "logical_id": "SkyMustStartCancelRole",
            "replacement": "False",
            "resource_type": "AWS::IAM::Role",
        },
        {
            "action": "Modify",
            "logical_id": "SkyMustStartCancelRule",
            "replacement": "False",
            "resource_type": "AWS::Events::Rule",
        },
        {
            "action": "Remove",
            "logical_id": "SkyMustStartDeadlineSchedule",
            "replacement": "None",
            "resource_type": "AWS::Scheduler::Schedule",
        },
        {
            "action": "Remove",
            "logical_id": _INTENTIONAL_DRIFT_AUTO_DELETE,
            "replacement": "None",
            "resource_type": "Custom::SchedulerActionAfterCompletion",
        },
        {
            "action": "Modify",
            "logical_id": "SkyPilotControllerRole",
            "replacement": "False",
            "resource_type": "AWS::IAM::Role",
        },
    ]
    expected_client_token = hashlib.sha256(
        canonical_json_bytes(
            {
                "phase": "intentional-drift-settlement-v1",
                "stack_id": stack_id,
                "change_set_name": change_set_name,
                "template_coordinate": coordinate,
                "role_arn": None,
                "parameters": parameters,
                "capabilities": capabilities,
                "initial_drifts": initial_drifts,
                "initial_semantics": initial_semantics,
            }
        )
    ).hexdigest()
    if (
        change_evidence.get("schema_version") != 1
        or change_evidence.get("record_type")
        != "glm52_task13_retained_drift_settlement_change_set_v1"
        or change_evidence.get("account_id") != ACCOUNT_ID
        or change_evidence.get("region") != REGION
        or change_evidence.get("profile") != PROFILE
        or change_evidence.get("phase") != "intentional-drift-settlement-v1"
        or change_evidence.get("stack_id") != stack_id
        or change_evidence.get("stack_name") != RETAINED_STACK_NAME
        or change_evidence.get("initial_role_arn") is not None
        or change_evidence.get("change_set_role_arn") is not None
        or change_evidence.get("termination_protection") is not termination_protection
        or change_evidence.get("before_template_sha256")
        != hashlib.sha256(before_raw[:-1]).hexdigest()
        or change_evidence.get("after_template_sha256") != body_sha
        or change_evidence.get("template_url") != _template_url(coordinate)
        or change_evidence.get("change_set_name") != change_set_name
        or type(change_set_id) is not str
        or _CHANGE_SET_ID.fullmatch(change_set_id) is None
        or not change_set_id.startswith(
            f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:changeSet/{change_set_name}/"
        )
        or change_evidence.get("client_token") != expected_client_token
        or change_evidence.get("parameters") != parameters
        or change_evidence.get("capabilities") != capabilities
        or change_evidence.get("initial_drifts") != initial_drifts
        or type(initial_semantics) is not dict
        or change_evidence.get("pre_execute_semantics") != initial_semantics
        or change_evidence.get("changes") != expected_changes
    ):
        raise RetainedUpdateError(
            "intentional drift recovery change evidence is not exact"
        )

    final_semantics = _intentional_drift_semantic_snapshot(
        services=services,
        template=template,
        stack=stack,
        resources=resources,
    )
    initial_semantic_values = deepcopy(initial_semantics)
    final_semantic_values = deepcopy(final_semantics)
    initial_launch_version = initial_semantic_values.pop("launch_template_version")
    final_launch_version = final_semantic_values.pop("launch_template_version")
    if (
        initial_semantic_values != final_semantic_values
        or type(initial_launch_version) is not int
        or type(final_launch_version) is not int
        or final_launch_version
        not in {initial_launch_version, initial_launch_version + 1}
        or "SkyMustStartDeadlineSchedule" in resources
        or _INTENTIONAL_DRIFT_AUTO_DELETE in resources
    ):
        raise RetainedUpdateError(
            "intentional drift recovery semantic state is not exact"
        )
    final_drifts = _read_intentional_drift_rows(
        services.cloudformation,
        stack_id=stack_id,
        sleep=sleep,
        max_polls=max_polls,
        expected_stack_drift_status="IN_SYNC",
    )
    if final_drifts:
        raise RetainedUpdateError("intentional drift recovery found remaining drift")
    readback = {
        "schema_version": 1,
        "record_type": "glm52_task13_retained_drift_settlement_readback_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "phase": "intentional-drift-settlement-v1",
        "stack_id": stack_id,
        "stack_name": RETAINED_STACK_NAME,
        "stack_status": "UPDATE_COMPLETE",
        "role_arn": stack.get("RoleARN"),
        "termination_protection": termination_protection,
        "template_sha256": body_sha,
        "template_coordinate": coordinate,
        "before_resource_count": len(resources) + 2,
        "final_resource_count": len(resources),
        "removed_resources": [
            "SkyMustStartDeadlineSchedule",
            _INTENTIONAL_DRIFT_AUTO_DELETE,
        ],
        "settled_resource": "SkyMustStartCancelRule",
        "initial_drifts": initial_drifts,
        "final_drifts": final_drifts,
        "initial_semantics": initial_semantics,
        "final_semantics": final_semantics,
        "execute_response_observation": (
            "UNKNOWN_RECOVERED_FROM_EXACT_FINAL_STATE"
        ),
        "recovered_from_exact_final_state": True,
    }
    _write_once(
        paths["readback"],
        canonical_json_bytes(readback) + b"\n",
    )
    return {
        "status": "UPDATE_COMPLETE",
        "stack_id": stack_id,
        "change_set_id": change_set_id,
        "phase": "intentional-drift-settlement-v1",
        "template_key": coordinate["key"],
        "template_version_id": coordinate["version_id"],
    }


def _guard_intentional_drift_inputs(
    *,
    services: RetainedUpdateServices,
    output_directory: Path,
    sleep: Callable[[float], None],
    max_polls: int,
) -> tuple[Mapping[str, str], Mapping[str, Path]]:
    if (
        type(services) is not RetainedUpdateServices
        or services.total_max_attempts != 1
        or not callable(sleep)
        or type(max_polls) is not int
        or not 2 <= max_polls <= 720
    ):
        raise RetainedUpdateError(
            "intentional drift services or polling bound is not exact"
        )
    directory = Path(output_directory)
    if (
        not directory.is_absolute()
        or not directory.is_dir()
        or directory.is_symlink()
        or directory.resolve(strict=True) != directory
    ):
        raise RetainedUpdateError(
            "intentional drift output must be one existing absolute directory"
        )
    stem = "retained-intentional-drift-settlement-v1"
    paths = {
        "before": directory / f"{stem}-before-template.json",
        "after": directory / f"{stem}-after-template.json",
        "change_set": directory / f"{stem}-change-set.json",
        "readback": directory / f"{stem}-readback.json",
    }
    present = {key for key, path in paths.items() if path.exists()}
    if not present:
        if any(directory.iterdir()):
            raise RetainedUpdateError(
                "intentional drift settlement output directory is not empty"
            )
    elif present == {"before", "after", "change_set"}:
        if (
            paths["readback"].is_symlink()
            or any(
                path.is_symlink() or not path.is_file()
                for key, path in paths.items()
                if key != "readback"
            )
            or set(directory.iterdir())
            != {paths["before"], paths["after"], paths["change_set"]}
        ):
            raise RetainedUpdateError(
                "intentional drift recovery evidence is not exact"
            )
    else:
        raise RetainedUpdateError("intentional drift settlement output already exists")
    return _phase_contract("intentional-drift-settlement-v1"), paths


def _read_intentional_drift_rows(
    cloudformation: object,
    *,
    stack_id: str,
    sleep: Callable[[float], None],
    max_polls: int,
    expected_stack_drift_status: str,
) -> list[dict[str, object]]:
    started = _call(
        cloudformation,
        "detect_stack_drift",
        StackName=stack_id,
    )
    detection_id = started.get("StackDriftDetectionId")
    if type(detection_id) is not str or not detection_id:
        raise RetainedUpdateError(
            "intentional drift detection returned an inexact identity"
        )
    terminal: dict[str, object] | None = None
    for poll in range(max_polls):
        status = _call(
            cloudformation,
            "describe_stack_drift_detection_status",
            StackDriftDetectionId=detection_id,
        )
        if status.get("DetectionStatus") == "DETECTION_COMPLETE":
            terminal = status
            break
        if status.get("DetectionStatus") not in {
            "DETECTION_IN_PROGRESS",
            "DETECTION_PENDING",
        }:
            raise RetainedUpdateError(
                "intentional drift detection reached a failed terminal state"
            )
        if poll + 1 < max_polls:
            sleep(2.0)
    if terminal is None:
        raise RetainedUpdateError(
            "intentional drift detection polling exceeded its bound"
        )
    if terminal.get("StackDriftStatus") != expected_stack_drift_status:
        raise RetainedUpdateError(
            "intentional drift detection stack drift status is not exact"
        )
    token: str | None = None
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for _page in range(64):
        request: dict[str, object] = {
            "StackName": stack_id,
            "StackResourceDriftStatusFilters": ["DELETED", "MODIFIED"],
        }
        if token is not None:
            request["NextToken"] = token
        response = _call(
            cloudformation,
            "describe_stack_resource_drifts",
            **request,
        )
        page = response.get("StackResourceDrifts")
        if type(page) is not list or any(type(row) is not dict for row in page):
            raise RetainedUpdateError("intentional drift resource page is malformed")
        for row in page:
            logical_id = row.get("LogicalResourceId")
            resource_type = row.get("ResourceType")
            drift_status = row.get("StackResourceDriftStatus")
            differences = row.get("PropertyDifferences", [])
            if (
                type(logical_id) is not str
                or not logical_id
                or type(resource_type) is not str
                or not resource_type
                or drift_status not in {"DELETED", "MODIFIED"}
                or type(differences) is not list
                or any(type(item) is not dict for item in differences)
            ):
                raise RetainedUpdateError(
                    "intentional drift resource page is malformed"
                )
            rows.append(
                {
                    "LogicalResourceId": logical_id,
                    "ResourceType": resource_type,
                    "StackResourceDriftStatus": drift_status,
                    "PropertyDifferences": deepcopy(differences),
                }
            )
        next_token = response.get("NextToken")
        if next_token is None:
            return sorted(rows, key=lambda row: str(row["LogicalResourceId"]))
        if type(next_token) is not str or not next_token or next_token in seen:
            raise RetainedUpdateError(
                "intentional drift resource pagination is incomplete"
            )
        seen.add(next_token)
        token = next_token
    raise RetainedUpdateError(
        "intentional drift resource pagination exceeded its bound"
    )


def _guard_inputs(
    *,
    services: RetainedUpdateServices,
    phase: str,
    fragment: Mapping[str, object],
    output_directory: Path,
    sleep: Callable[[float], None],
    max_polls: int,
) -> tuple[Mapping[str, str], Mapping[str, Path]]:
    phase_contract = _phase_contract(phase)
    if (
        type(services) is not RetainedUpdateServices
        or services.total_max_attempts != 1
        or not callable(sleep)
        or type(max_polls) is not int
        or not 2 <= max_polls <= 720
    ):
        raise RetainedUpdateError(
            "retained update services or polling bound is not exact"
        )
    _fragment_resources(fragment)
    canonical_json_bytes(fragment)
    if fragment.get("Parameters", {}) != {}:
        raise RetainedUpdateError(
            "retained additive fragment introduces forbidden parameters"
        )
    _assert_no_wildcard_iam_actions(fragment)
    directory = Path(output_directory)
    if (
        not directory.is_absolute()
        or not directory.is_dir()
        or directory.is_symlink()
        or directory.resolve(strict=True) != directory
    ):
        raise RetainedUpdateError(
            "output directory must be one existing absolute non-symlink path"
        )
    stem = "retained-" + phase
    paths = {
        "before": directory / f"{stem}-before-template.json",
        "after": directory / f"{stem}-after-template.json",
        "change_set": directory / f"{stem}-change-set.json",
        "readback": directory / f"{stem}-readback.json",
    }
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise RetainedUpdateError("retained update output already exists")
    return phase_contract, paths


def _phase_contract(phase: object) -> Mapping[str, str]:
    if type(phase) is not str or phase not in _PHASES:
        raise RetainedUpdateError("retained update phase is not implemented")
    return _PHASES[phase]


def _fragment_resources(
    fragment: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    if type(fragment) is not dict:
        raise RetainedUpdateError("retained fragment must be one exact mapping object")
    resources = fragment.get("Resources")
    if (
        fragment.get("AWSTemplateFormatVersion") != "2010-09-09"
        or type(resources) is not dict
        or not resources
    ):
        raise RetainedUpdateError(
            "retained fragment requires one non-empty resource graph"
        )
    result: dict[str, dict[str, object]] = {}
    for logical_id, resource in resources.items():
        if (
            type(logical_id) is not str
            or not logical_id
            or type(resource) is not dict
            or type(resource.get("Type")) is not str
            or not resource["Type"]
        ):
            raise RetainedUpdateError("retained fragment contains a malformed resource")
        result[logical_id] = deepcopy(resource)
    return result


def _guard_parameterless_composition(
    *,
    before_template: Mapping[str, object],
    fragment: Mapping[str, object],
    composed: Mapping[str, object],
) -> None:
    before_parameters = before_template.get("Parameters", {})
    if type(before_parameters) is not dict:
        raise RetainedUpdateError("retained base Parameters are malformed")
    if fragment.get("Parameters", {}) != {}:
        raise RetainedUpdateError("retained fragment is not parameterless")
    if composed.get("Parameters", {}) != before_parameters:
        raise RetainedUpdateError("retained composition changed the base Parameters")


def _previous_parameters(value: object) -> list[dict[str, object]]:
    if type(value) is not list:
        raise RetainedUpdateError("retained stack parameter readback is malformed")
    names: set[str] = set()
    for row in value:
        parameter_key = row.get("ParameterKey") if type(row) is dict else None
        if (
            type(row) is not dict
            or type(parameter_key) is not str
            or not parameter_key
            or parameter_key in names
            or ("ParameterValue" not in row and row.get("UsePreviousValue") is not True)
        ):
            raise RetainedUpdateError(
                "retained stack parameters are not exact and unique"
            )
        names.add(parameter_key)
    return [{"ParameterKey": name, "UsePreviousValue": True} for name in sorted(names)]


def _parameter_readback(value: object) -> list[dict[str, object]]:
    if type(value) is not list:
        raise RetainedUpdateError("retained stack parameter readback is malformed")
    parameters: list[dict[str, object]] = []
    names: set[str] = set()
    for row in value:
        if type(row) is not dict or set(row) not in (
            {"ParameterKey", "ParameterValue"},
            {"ParameterKey", "ParameterValue", "ResolvedValue"},
        ):
            raise RetainedUpdateError("retained stack parameter readback is malformed")
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
            raise RetainedUpdateError("retained stack parameter readback is malformed")
        names.add(name)
        parameters.append(deepcopy(row))
    return sorted(parameters, key=lambda row: str(row["ParameterKey"]))


def _validate_live_parameter_keys(
    *,
    before_template: Mapping[str, object],
    parameters: list[dict[str, object]],
) -> None:
    template_parameters = before_template.get("Parameters", {})
    if type(template_parameters) is not dict or any(
        type(name) is not str or not name for name in template_parameters
    ):
        raise RetainedUpdateError("retained Original template Parameters are malformed")
    observed = {row["ParameterKey"] for row in parameters}
    if observed != set(template_parameters):
        raise RetainedUpdateError(
            "live retained ParameterKeys differ from Original template"
        )


def _required_capabilities(
    resources: Mapping[str, Mapping[str, object]],
) -> list[str]:
    return _required_capabilities_from_types(
        {logical_id: resource["Type"] for logical_id, resource in resources.items()}
    )


def _required_capabilities_from_types(
    resources: Mapping[str, str],
) -> list[str]:
    if any(
        resource_type.startswith("AWS::IAM::") for resource_type in resources.values()
    ):
        return ["CAPABILITY_NAMED_IAM"]
    return []


def _authenticate_caller(sts: object) -> None:
    identity = _call(sts, "get_caller_identity")
    arn = identity.get("Arn")
    if (
        identity.get("Account") != ACCOUNT_ID
        or type(arn) is not str
        or f"::{ACCOUNT_ID}:" not in arn
        or type(identity.get("UserId")) is not str
        or not identity["UserId"]
    ):
        raise RetainedUpdateError("authenticated AWS caller identity is foreign")


def _authenticate_deployment_role(iam: object) -> None:
    response = _call(
        iam,
        "get_role",
        RoleName=DEPLOYMENT_ROLE_NAME,
    )
    role = response.get("Role")
    if (
        type(role) is not dict
        or role.get("Arn") != DEPLOYMENT_ROLE_ARN
        or role.get("RoleName") != DEPLOYMENT_ROLE_NAME
        or type(role.get("RoleId")) is not str
        or not role["RoleId"]
    ):
        raise RetainedUpdateError(
            "deployment RoleARN is not authenticated by exact IAM readback"
        )
    trust = role.get("AssumeRolePolicyDocument")
    statements = trust.get("Statement") if type(trust) is dict else None
    if (
        type(statements) is not list
        or len(statements) != 1
        or type(statements[0]) is not dict
        or statements[0].get("Effect") != "Allow"
        or statements[0].get("Principal") != {"Service": "cloudformation.amazonaws.com"}
        or statements[0].get("Action") != "sts:AssumeRole"
    ):
        raise RetainedUpdateError("deployment role trust is not CloudFormation-only")


def _guard_versioned_bucket(s3: object) -> None:
    response = _call(
        s3,
        "get_bucket_versioning",
        Bucket=RETAINED_MODELS_BUCKET,
        ExpectedBucketOwner=ACCOUNT_ID,
    )
    if response.get("Status") != "Enabled":
        raise RetainedUpdateError("retained models bucket versioning is not enabled")


def _call(client: object, method_name: str, **request: object) -> dict[str, object]:
    method = getattr(client, method_name, None)
    if not callable(method):
        raise RetainedUpdateError(f"AWS client method {method_name} is absent")
    response = method(**request)
    if type(response) is not dict:
        raise RetainedUpdateError(f"AWS {method_name} returned a non-object")
    return response


def _error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    detail = response.get("Error") if type(response) is dict else None
    code = detail.get("Code") if type(detail) is dict else None
    return code if type(code) is str else None


def _ambiguous_exception(error: BaseException) -> bool:
    return isinstance(error, (TimeoutError, ConnectionError)) or (
        _error_code(error) in _AMBIGUOUS_CODES
    )


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
        raise RetainedUpdateError("retained stack lookup is not singular")
    stack = deepcopy(stacks[0])
    if (
        stack.get("StackName") != RETAINED_STACK_NAME
        or type(stack.get("StackId")) is not str
        or _STACK_ID.fullmatch(stack["StackId"]) is None
        or stack.get("StackStatus") not in allowed_statuses
        or type(stack.get("Parameters", [])) is not list
        or type(stack.get("Outputs", [])) is not list
        or type(stack.get("Tags", [])) is not list
        or type(stack.get("EnableTerminationProtection")) is not bool
    ):
        raise RetainedUpdateError("retained stack identity or status is not exact")
    return stack


def _termination_protection(stack: Mapping[str, object]) -> bool:
    value = stack.get("EnableTerminationProtection")
    if type(value) is not bool:
        raise RetainedUpdateError(
            "retained termination-protection readback is malformed"
        )
    return value


class _CloudFormationLoader(yaml.SafeLoader):
    """Safe loader preserving CloudFormation short-form intrinsic tags."""

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
    else:  # pragma: no cover
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


def _template_body(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        parsed_mapping = json.loads(canonical_json_bytes(value))
        if type(parsed_mapping) is not dict:
            raise RetainedUpdateError(
                "CloudFormation Original template is not one object"
            )
        return parsed_mapping
    if type(value) is str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = yaml.load(value, Loader=_CloudFormationLoader)
            except yaml.YAMLError as exc:
                raise RetainedUpdateError(
                    "CloudFormation Original is not valid JSON or YAML"
                ) from exc
        if type(parsed) is not dict:
            raise RetainedUpdateError(
                "CloudFormation Original template is not one object"
            )
        canonical_json_bytes(parsed)
        return parsed
    raise RetainedUpdateError("CloudFormation Original template body is malformed")


def _get_template(
    cloudformation: object,
    *,
    stack_name: str | None = None,
    change_set_name: str | None = None,
) -> dict[str, object]:
    if (stack_name is None) == (change_set_name is None):
        raise RetainedUpdateError(
            "one CloudFormation Original template identity is required"
        )
    request: dict[str, object] = {"TemplateStage": "Original"}
    if stack_name is not None:
        request["StackName"] = stack_name
    else:
        request["ChangeSetName"] = change_set_name
    response = _call(cloudformation, "get_template", **request)
    return _template_body(response.get("TemplateBody"))


def _checksum(raw: bytes) -> str:
    return base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")


def _inventory_versions(
    s3: object,
    *,
    key: str,
) -> tuple[dict[str, object], ...]:
    key_marker: str | None = None
    version_marker: str | None = None
    seen: set[tuple[str, str]] = set()
    exact: list[dict[str, object]] = []
    for _page in range(1024):
        request: dict[str, object] = {
            "Bucket": RETAINED_MODELS_BUCKET,
            "Prefix": key,
            "MaxKeys": 1000,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
            request["VersionIdMarker"] = version_marker
        response = _call(s3, "list_object_versions", **request)
        versions = response.get("Versions", [])
        delete_markers = response.get("DeleteMarkers", [])
        truncated = response.get("IsTruncated")
        if (
            type(versions) is not list
            or type(delete_markers) is not list
            or type(truncated) is not bool
            or any(type(row) is not dict for row in versions)
            or any(type(row) is not dict for row in delete_markers)
        ):
            raise RetainedUpdateError(
                "retained template version inventory is malformed"
            )
        if any(row.get("Key") == key for row in delete_markers):
            raise RetainedUpdateError(
                "retained template fixed key contains a delete marker"
            )
        exact.extend(deepcopy(row) for row in versions if row.get("Key") == key)
        if not truncated:
            return tuple(exact)
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or (next_key, next_version) in seen
        ):
            raise RetainedUpdateError(
                "retained template version pagination is incomplete"
            )
        seen.add((next_key, next_version))
        key_marker = next_key
        version_marker = next_version
    raise RetainedUpdateError("retained template version pagination exceeded its bound")


def _version_id(value: object) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or not value.isascii()
        or _VERSION_ID.fullmatch(value) is None
    ):
        raise RetainedUpdateError("retained template VersionId is malformed")
    return value


def _read_exact_template_version(
    s3: object,
    *,
    key: str,
    version_id: str,
    raw: bytes,
    metadata: Mapping[str, str],
) -> None:
    response = _call(
        s3,
        "get_object",
        Bucket=RETAINED_MODELS_BUCKET,
        Key=key,
        VersionId=version_id,
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    body = response.get("Body")
    read = getattr(body, "read", None)
    observed = read() if callable(read) else None
    close = getattr(body, "close", None)
    if callable(close):
        close()
    if (
        type(observed) is not bytes
        or observed != raw
        or response.get("ContentLength") != len(raw)
        or response.get("VersionId") != version_id
        or response.get("ChecksumSHA256") != _checksum(raw)
        or response.get("Metadata") != dict(metadata)
    ):
        raise RetainedUpdateError("retained template VersionId readback drifted")


def _adopt_exact_template_version(
    s3: object,
    *,
    key: str,
    raw: bytes,
    metadata: Mapping[str, str],
) -> str | None:
    versions = _inventory_versions(s3, key=key)
    if not versions:
        return None
    if len(versions) != 1:
        raise RetainedUpdateError(
            "retained template fixed-key positive truth is not singular"
        )
    version_id = _version_id(versions[0].get("VersionId"))
    if versions[0].get("Size") != len(raw):
        raise RetainedUpdateError("retained template fixed-key version size is foreign")
    _read_exact_template_version(
        s3,
        key=key,
        version_id=version_id,
        raw=raw,
        metadata=metadata,
    )
    return version_id


def _publish_template_once(
    *,
    s3: object,
    phase_contract: Mapping[str, str],
    raw: bytes,
) -> dict[str, object]:
    key = phase_contract["key"]
    artifact_kind = phase_contract["artifact_kind"]
    file_sha = hashlib.sha256(raw).hexdigest()
    body_sha = hashlib.sha256(raw[:-1]).hexdigest()
    metadata = {
        "artifact-kind": artifact_kind,
        "body-sha256": body_sha,
        "file-sha256": file_sha,
        "glm52-run-id": "glm52-sky-20260724",
    }
    existing = _adopt_exact_template_version(
        s3,
        key=key,
        raw=raw,
        metadata=metadata,
    )
    if existing is not None:
        version_id = existing
    else:
        try:
            response = _call(
                s3,
                "put_object",
                Bucket=RETAINED_MODELS_BUCKET,
                Key=key,
                Body=raw,
                ContentType="application/json",
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=_checksum(raw),
                IfNoneMatch="*",
                ExpectedBucketOwner=ACCOUNT_ID,
                Metadata=metadata,
            )
            version_id = _version_id(response.get("VersionId"))
            if response.get("ChecksumSHA256") != _checksum(raw):
                raise RetainedUpdateError(
                    "retained template PutObject checksum drifted"
                )
        except Exception as exc:
            if isinstance(exc, RetainedUpdateError):
                raise
            try:
                adopted = _adopt_exact_template_version(
                    s3,
                    key=key,
                    raw=raw,
                    metadata=metadata,
                )
            except Exception as readback_error:
                raise RetainedUpdateError(
                    "ambiguous retained template PutObject has no exact adoption"
                ) from readback_error
            if adopted is None:
                raise RetainedUpdateError(
                    "ambiguous retained template PutObject has no durable version"
                ) from exc
            version_id = adopted
        adopted = _adopt_exact_template_version(
            s3,
            key=key,
            raw=raw,
            metadata=metadata,
        )
        if adopted != version_id:
            raise RetainedUpdateError(
                "retained template PutObject VersionId is not singular"
            )
    return {
        "artifact_kind": artifact_kind,
        "bucket": RETAINED_MODELS_BUCKET,
        "key": key,
        "version_id": version_id,
        "file_sha256": file_sha,
        "body_sha256": body_sha,
    }


def _template_url(coordinate: Mapping[str, object]) -> str:
    phase_contracts = {(row["artifact_kind"], row["key"]) for row in _PHASES.values()}
    pair = (coordinate.get("artifact_kind"), coordinate.get("key"))
    version_id = coordinate.get("version_id")
    if (
        coordinate.get("bucket") != RETAINED_MODELS_BUCKET
        or pair not in phase_contracts
        or type(version_id) is not str
    ):
        raise RetainedUpdateError("retained template coordinate is not exact")
    _version_id(version_id)
    return (
        f"https://{RETAINED_MODELS_BUCKET}.s3.{REGION}.amazonaws.com/"
        f"{quote(str(coordinate['key']), safe='/')}?versionId="
        f"{quote(version_id, safe='')}"
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
            raise RetainedUpdateError("retained change-set page is malformed")
        changes.extend(deepcopy(page_changes))
        projection = {
            key: value
            for key, value in response.items()
            if key not in {"Changes", "NextToken", "ResponseMetadata"}
        }
        if combined is None:
            combined = deepcopy(projection)
        elif canonical_json_bytes(combined) != canonical_json_bytes(projection):
            raise RetainedUpdateError(
                "retained change-set identity changed across pages"
            )
        next_token = response.get("NextToken")
        if next_token is None:
            result = {} if combined is None else combined
            result["Changes"] = changes
            return result
        if type(next_token) is not str or not next_token or next_token in seen:
            raise RetainedUpdateError("retained change-set pagination is incomplete")
        seen.add(next_token)
        token = next_token
    raise RetainedUpdateError("retained change-set pagination exceeded its bound")


def _wait_for_change_set(
    cloudformation: object,
    *,
    stack_id: str,
    change_set_name: str,
    sleep: Callable[[float], None],
    max_polls: int,
) -> dict[str, object]:
    for poll in range(max_polls):
        response = _describe_full_change_set(
            cloudformation,
            stack_id=stack_id,
            identity=change_set_name,
        )
        if response.get("Status") == "CREATE_COMPLETE":
            return response
        if response.get("Status") not in {"CREATE_PENDING", "CREATE_IN_PROGRESS"}:
            raise RetainedUpdateError(
                "retained change set reached a failed terminal state"
            )
        if poll + 1 < max_polls:
            sleep(5.0)
    raise RetainedUpdateError("retained change-set polling exceeded its bound")


def _normalized_changes(
    change_set: Mapping[str, object],
) -> list[dict[str, str]]:
    changes = change_set.get("Changes")
    if type(changes) is not list:
        raise RetainedUpdateError("retained change-set changes are malformed")
    result: list[dict[str, str]] = []
    for change in changes:
        resource = change.get("ResourceChange") if type(change) is dict else None
        if type(resource) is not dict:
            raise RetainedUpdateError("retained resource change is malformed")
        result.append(
            {
                "action": str(resource.get("Action")),
                "logical_id": str(resource.get("LogicalResourceId")),
                "replacement": str(resource.get("Replacement")),
                "resource_type": str(resource.get("ResourceType")),
            }
        )
    return sorted(result, key=lambda row: row["logical_id"])


def _initial_stack_guard(stack: Mapping[str, object]) -> dict[str, object]:
    parameters = sorted(
        _parameter_readback(stack.get("Parameters", [])),
        key=lambda row: str(row["ParameterKey"]),
    )
    outputs = _outputs_by_key(stack.get("Outputs", []))
    tags = stack.get("Tags", [])
    if (
        type(tags) is not list
        or any(
            type(row) is not dict
            or type(row.get("Key")) is not str
            or type(row.get("Value")) is not str
            for row in tags
        )
        or len({str(row["Key"]) for row in tags}) != len(tags)
    ):
        raise RetainedUpdateError("retained stack tags are malformed")
    return {
        "StackId": stack.get("StackId"),
        "StackName": stack.get("StackName"),
        "StackStatus": stack.get("StackStatus"),
        "Parameters": parameters,
        "Outputs": [outputs[key] for key in sorted(outputs)],
        "RoleARN": stack.get("RoleARN"),
        "Tags": sorted(tags, key=lambda row: str(row["Key"])),
        "EnableTerminationProtection": stack.get("EnableTerminationProtection"),
    }


def _adopt_ambiguous_execute(
    cloudformation: object,
    *,
    stack_id: str,
    change_set_id: str,
    change_set_name: str,
) -> None:
    response = _describe_full_change_set(
        cloudformation,
        stack_id=stack_id,
        identity=change_set_id,
    )
    if (
        response.get("StackId") != stack_id
        or response.get("ChangeSetId") != change_set_id
        or response.get("ChangeSetName") != change_set_name
        or response.get("ExecutionStatus")
        not in {"EXECUTE_IN_PROGRESS", "EXECUTE_COMPLETE"}
    ):
        raise RetainedUpdateError(
            "ambiguous ExecuteChangeSet has no exact describe adoption"
        )
    _describe_stack(
        cloudformation,
        stack_id,
        allowed_statuses={
            "UPDATE_IN_PROGRESS",
            "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
            "UPDATE_COMPLETE",
        },
    )


def _wait_for_update_complete(
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
        if stack["StackStatus"] == "UPDATE_COMPLETE":
            return stack
        if poll + 1 < max_polls:
            sleep(10.0)
    raise RetainedUpdateError("retained stack update polling exceeded its bound")


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
            raise RetainedUpdateError("retained stack resource page is malformed")
        rows.extend(deepcopy(page))
        next_token = response.get("NextToken")
        if next_token is None:
            return rows
        if type(next_token) is not str or not next_token or next_token in seen:
            raise RetainedUpdateError(
                "retained stack resource pagination is incomplete"
            )
        seen.add(next_token)
        token = next_token
    raise RetainedUpdateError("retained stack resource pagination exceeded its bound")


def _resource_map(
    rows: list[dict[str, object]],
    *,
    label: str,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        logical_id = row.get("LogicalResourceId")
        if (
            type(logical_id) is not str
            or not logical_id
            or logical_id in result
            or type(row.get("PhysicalResourceId")) is not str
            or not row["PhysicalResourceId"]
            or type(row.get("ResourceType")) is not str
            or not row["ResourceType"]
            or row.get("ResourceStatus") not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
        ):
            raise RetainedUpdateError(label + " is not exact")
        result[logical_id] = {
            "LogicalResourceId": logical_id,
            "PhysicalResourceId": row["PhysicalResourceId"],
            "ResourceType": row["ResourceType"],
            "ResourceStatus": row["ResourceStatus"],
        }
    return result


def _validate_before_resources(
    *,
    before_template: Mapping[str, object],
    before_resources: Mapping[str, Mapping[str, object]],
) -> None:
    template_resources = before_template.get("Resources")
    if type(template_resources) is not dict:
        raise RetainedUpdateError("retained Original template lacks a resource graph")
    for logical_id, row in before_resources.items():
        template_resource = template_resources.get(logical_id)
        if (
            type(template_resource) is not dict
            or template_resource.get("Type") != row["ResourceType"]
        ):
            raise RetainedUpdateError(
                "live retained resource differs from Original template"
            )


def _outputs_by_key(value: object) -> dict[str, dict[str, object]]:
    if type(value) is not list:
        raise RetainedUpdateError("retained stack outputs are malformed")
    result: dict[str, dict[str, object]] = {}
    for row in value:
        output_key = row.get("OutputKey") if type(row) is dict else None
        if (
            type(row) is not dict
            or type(output_key) is not str
            or not output_key
            or output_key in result
            or type(row.get("OutputValue")) is not str
            or not row["OutputValue"]
            or (
                row.get("ExportName") is not None
                and (type(row.get("ExportName")) is not str or not row["ExportName"])
            )
        ):
            raise RetainedUpdateError("retained stack output identity is not exact")
        result[output_key] = deepcopy(row)
    return result


def _exports_for_stack(
    cloudformation: object,
    *,
    stack_id: str,
) -> dict[str, dict[str, object]]:
    token: str | None = None
    seen: set[str] = set()
    exports: dict[str, dict[str, object]] = {}
    for _page in range(64):
        request: dict[str, object] = {}
        if token is not None:
            request["NextToken"] = token
        response = _call(cloudformation, "list_exports", **request)
        rows = response.get("Exports")
        if type(rows) is not list or any(type(row) is not dict for row in rows):
            raise RetainedUpdateError("CloudFormation export page is malformed")
        for row in rows:
            if row.get("ExportingStackId") != stack_id:
                continue
            name = row.get("Name")
            if (
                type(name) is not str
                or not name
                or name in exports
                or type(row.get("Value")) is not str
                or not row["Value"]
            ):
                raise RetainedUpdateError(
                    "retained CloudFormation export identity is not exact"
                )
            exports[name] = deepcopy(row)
        next_token = response.get("NextToken")
        if next_token is None:
            return exports
        if type(next_token) is not str or not next_token or next_token in seen:
            raise RetainedUpdateError("CloudFormation export pagination is incomplete")
        seen.add(next_token)
        token = next_token
    raise RetainedUpdateError("CloudFormation export pagination exceeded its bound")


def _validate_final_stack(
    *,
    initial_stack: Mapping[str, object],
    final_stack: Mapping[str, object],
    before_resources: Mapping[str, Mapping[str, object]],
    final_resources: Mapping[str, Mapping[str, object]],
    fragment_resources: Mapping[str, Mapping[str, object]],
    fragment: Mapping[str, object],
    cloudformation: object,
    stack_id: str,
    expected_role_arn: str | None,
    expected_parameters: list[dict[str, object]],
) -> None:
    if (
        final_stack.get("StackId") != initial_stack.get("StackId")
        or final_stack.get("StackName") != RETAINED_STACK_NAME
        or final_stack.get("StackStatus") != "UPDATE_COMPLETE"
        or final_stack.get("RoleARN") != expected_role_arn
        or _parameter_readback(final_stack.get("Parameters", [])) != expected_parameters
        or final_stack.get("Tags", []) != initial_stack.get("Tags", [])
    ):
        raise RetainedUpdateError("final retained stack identity or role drifted")
    expected_ids = set(before_resources) | set(fragment_resources)
    if set(final_resources) != expected_ids:
        raise RetainedUpdateError("final retained resource inventory is not exact")
    for logical_id, before in before_resources.items():
        final = final_resources[logical_id]
        if (
            final["PhysicalResourceId"] != before["PhysicalResourceId"]
            or final["ResourceType"] != before["ResourceType"]
        ):
            raise RetainedUpdateError("legacy retained resource identity changed")
    for logical_id, resource in fragment_resources.items():
        final = final_resources[logical_id]
        if final["ResourceType"] != resource["Type"]:
            raise RetainedUpdateError("added retained resource type drifted")

    before_outputs = _outputs_by_key(initial_stack.get("Outputs", []))
    final_outputs = _outputs_by_key(final_stack.get("Outputs", []))
    fragment_output_value = fragment.get("Outputs", {})
    if type(fragment_output_value) is not dict:
        raise RetainedUpdateError("retained fragment Outputs are malformed")
    if set(final_outputs) != set(before_outputs) | set(fragment_output_value):
        raise RetainedUpdateError("final retained output inventory is not exact")
    for output_key, before in before_outputs.items():
        if final_outputs.get(output_key) != before:
            raise RetainedUpdateError("legacy retained output identity changed")
    expected_export_values = {
        row["ExportName"]: row["OutputValue"]
        for row in final_outputs.values()
        if row.get("ExportName") is not None
    }
    observed_exports = _exports_for_stack(
        cloudformation,
        stack_id=stack_id,
    )
    if set(observed_exports) != set(expected_export_values):
        raise RetainedUpdateError(
            "retained CloudFormation export inventory is not exact"
        )
    for name, value in expected_export_values.items():
        if observed_exports[name].get("Value") != value:
            raise RetainedUpdateError("retained CloudFormation export value drifted")
    for output_key, template_output in fragment_output_value.items():
        if type(template_output) is not dict:
            raise RetainedUpdateError("retained fragment output is malformed")
        export = template_output.get("Export")
        if export is None:
            if final_outputs[output_key].get("ExportName") is not None:
                raise RetainedUpdateError("unexported retained output gained an export")
            continue
        if (
            type(export) is not dict
            or type(export.get("Name")) is not str
            or final_outputs[output_key].get("ExportName") != export["Name"]
        ):
            raise RetainedUpdateError("added retained output export name drifted")


def _assert_no_wildcard_iam_actions(value: object) -> None:
    def visit(node: object) -> None:
        if type(node) is dict:
            if "Effect" in node and "Action" in node:
                action = node["Action"]
                actions = [action] if type(action) is str else action
                if (
                    type(actions) is not list
                    or not actions
                    or any(
                        type(item) is not str or not item or "*" in item
                        for item in actions
                    )
                ):
                    raise RetainedUpdateError(
                        "retained IAM statement contains Action:*"
                    )
            for nested in node.values():
                visit(nested)
        elif type(node) is list:
            for nested in node:
                visit(nested)

    visit(value)


def _write_once(path: Path, raw: bytes) -> None:
    if type(raw) is not bytes or not raw:
        raise RetainedUpdateError("sealed retained evidence is empty")
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
                raise OSError("sealed evidence write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise RetainedUpdateError(
            "sealed retained evidence path already exists"
        ) from exc
    except OSError as exc:
        raise RetainedUpdateError(
            "sealed retained evidence could not be written"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


__all__ = [
    "ACCOUNT_ID",
    "DEPLOYMENT_ROLE_ARN",
    "DEPLOYMENT_ROLE_NAME",
    "PROFILE",
    "REGION",
    "RETAINED_MODELS_BUCKET",
    "RETAINED_STACK_NAME",
    "RetainedUpdateError",
    "RetainedUpdateServices",
    "apply_retained_update",
    "canonical_json_bytes",
    "validate_retained_update_change_set",
]
