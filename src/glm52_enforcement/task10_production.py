"""Guarded Task 10 production operator route.

The route accepts only a closed, already action-consumed authority projection
and an injected production workflow boundary.  It contains no SkyPilot
client, HTTP client, subprocess, EC2 client, Systems Manager client, or SDK
retry path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import re
from typing import Mapping, Optional

from .canonical import canonical_json_bytes, canonical_sha256
from .task10_worker import (
    ACCOUNT_ID,
    H100_READY_KEY,
    MountFreeTaskInputs,
    REGION,
    RUN_ID,
    WorkerBootstrapDescriptor,
    build_jobs_launch_body,
    render_mount_free_task,
    validate_worker_bootstrap_descriptor,
    validate_task_inputs,
)


PRODUCTION_ACTIONS = ("validate-only", "start", "reconcile")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ACTIVATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9._~+/=-]{1,1024}$")
_EXECUTION = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:execution:"
    r"keep-glm52-h1g-production:[A-Za-z0-9_-]{1,80}$"
)
_WORKFLOW_VERSION = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:stateMachine:"
    r"keep-glm52-h1g-production:[1-9][0-9]*$"
)
_INSTANCE = re.compile(r"^i-[0-9a-f]{17}$")


class ProductionRouteError(ValueError):
    """The closed production route failed before any new effect."""


@dataclass(frozen=True)
class ProductionAuthority:
    """Exact Task 8/9 and artifact identities consumed by one production run."""

    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    managed_mode: str
    campaign_identity_sha256: str
    activation_id: str
    activation_ordinal: int
    generation: int
    generation_text: str
    sky_job_name: str
    execution_deadline: str
    gpu_allocation_sha256: str
    executor_epoch: int
    action_key: str
    action_state: str
    current_activation: bool
    descriptor_identity_sha256: str
    archive_identity_sha256: str
    repository_archive_version_id: str
    approval_identity_sha256: str
    approval_version_id: str
    intent_identity_sha256: str
    intent_version_id: str
    task11_boundary_bucket: str
    task11_boundary_key: str
    task11_boundary_version_id: str
    task11_boundary_file_sha256: str
    task11_boundary_body_sha256: str
    h100_resume_ready_bucket: str
    h100_resume_ready_key: str
    h100_resume_ready_version_id: str
    h100_resume_ready_file_sha256: str
    h100_resume_ready_body_sha256: str
    capacity_retry_availability_zones: list[str]
    capacity_retry_outcome_rules: list[dict[str, str]]
    maximum_capacity_attempts: int
    same_token_identity_sha256: str
    stop_on_first_worker_success: bool
    automatic_spot_fallback: bool
    capacity_block_allowed: bool
    task8_live_h1d_identity_sha256: str
    task8_spend_authority_identity_sha256: str
    task8_spend_reserve_identity_sha256: str
    task9_launch_identity_sha256: str
    task9_admission_identity_sha256: str
    task9_custody_identity_sha256: str
    attestation_identity_sha256: str
    decision_seal_identity_sha256: str
    relay_envelope_sha256: str
    task_yaml_sha256: str
    request_body_sha256: str
    workflow_version_arn: str
    execution_name: str
    expected_execution_arn: str
    execution_input_sha256: str
    task13_route_binding_identity_sha256: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class ProductionRequest:
    action: str
    profile: str
    authority: ProductionAuthority
    task_inputs: MountFreeTaskInputs
    worker_descriptor: WorkerBootstrapDescriptor


@dataclass(frozen=True)
class ProductionOutcome:
    exit_code: int
    status: str
    detail: Mapping[str, object]


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ProductionRouteError(label + " must be a lowercase SHA-256")
    return value


def _authority_body(value: ProductionAuthority) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def _authority_bindings_sha256(value: ProductionAuthority) -> str:
    body = dict(_authority_body(value))
    body.pop("execution_input_sha256")
    return canonical_sha256(body)


def build_production_authority(
    **values: object,
) -> ProductionAuthority:
    """Build and validate a closed authority from explicit authenticated pins."""

    provisional = ProductionAuthority(
        **{
            **values,
            "canonical_identity_sha256": "",
        }
    )
    return validate_production_authority(
        ProductionAuthority(
            **{
                **asdict(provisional),
                "canonical_identity_sha256": canonical_sha256(
                    _authority_body(provisional)
                ),
            }
        )
    )


def production_authority_from_mapping(
    value: object,
) -> ProductionAuthority:
    if type(value) is not dict:
        raise ProductionRouteError("production authority JSON must be an object")
    expected = set(ProductionAuthority.__dataclass_fields__)
    if set(value) != expected:
        raise ProductionRouteError("production authority field set drifted")
    try:
        result = ProductionAuthority(**value)
    except TypeError as exc:
        raise ProductionRouteError("production authority is malformed") from exc
    return validate_production_authority(result)


def validate_production_authority(
    value: object,
) -> ProductionAuthority:
    if not isinstance(value, ProductionAuthority):
        raise ProductionRouteError("production authority must be typed")
    if (
        value.schema_version != 1
        or value.record_type != "glm52_task10_production_authority_v1"
        or value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
        or value.managed_mode != "production"
        or type(value.activation_id) is not str
        or _ACTIVATION.fullmatch(value.activation_id) is None
        or type(value.activation_ordinal) is not int
        or value.activation_ordinal <= 0
        or type(value.generation) is not int
        or value.generation <= 0
        or value.generation_text != "%08d" % value.generation
        or type(value.sky_job_name) is not str
        or value.sky_job_name != RUN_ID
        or type(value.execution_deadline) is not str
        or type(value.executor_epoch) is not int
        or value.executor_epoch <= 0
        or type(value.action_key) is not str
        or "SKY_POST" not in value.action_key
        or value.action_state != "CONSUMED"
        or value.current_activation is not True
        or value.h100_resume_ready_bucket
        != "keep-glm52-models-246813579024-us-west-2"
        or value.h100_resume_ready_key != H100_READY_KEY
        or type(value.task11_boundary_bucket) is not str
        or not value.task11_boundary_bucket
        or type(value.task11_boundary_key) is not str
        or value.task11_boundary_key
        != (
            "campaigns/"
            + RUN_ID
            + "/authorities/task11/"
            + value.activation_id
            + "/"
            + value.generation_text
            + ".json"
        )
        or type(value.h100_resume_ready_version_id) is not str
        or value.h100_resume_ready_version_id == "null"
        or _VERSION.fullmatch(value.h100_resume_ready_version_id) is None
        or type(value.execution_name) is not str
        or not value.execution_name
        or _WORKFLOW_VERSION.fullmatch(value.workflow_version_arn) is None
        or _EXECUTION.fullmatch(value.expected_execution_arn) is None
        or not value.expected_execution_arn.endswith(
            ":" + value.execution_name
        )
    ):
        raise ProductionRouteError(
            "production activation/action/workflow identity drifted"
        )
    expected_zones = ["us-west-2" + letter for letter in "abcdef"]
    expected_outcome_rules = [
        {
            "availability_zone": zone,
            "on_capacity_failure": (
                "ROTATE_NEXT_AZ"
                if index < len(expected_zones) else "STOP_CAPACITY_EXHAUSTED"
            ),
        }
        for index, zone in enumerate(expected_zones, 1)
    ]
    if (
        value.capacity_retry_availability_zones != expected_zones
        or value.capacity_retry_outcome_rules != expected_outcome_rules
        or value.maximum_capacity_attempts != len(expected_zones)
        or value.stop_on_first_worker_success is not True
        or value.automatic_spot_fallback is not False
        or value.capacity_block_allowed is not False
    ):
        raise ProductionRouteError("production capacity retry contract drifted")
    try:
        datetime.strptime(value.execution_deadline, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ProductionRouteError(
            "production execution deadline drifted"
        ) from exc
    for field in (
        "campaign_identity_sha256",
        "gpu_allocation_sha256",
        "descriptor_identity_sha256",
        "archive_identity_sha256",
        "approval_identity_sha256",
        "intent_identity_sha256",
        "task11_boundary_file_sha256",
        "task11_boundary_body_sha256",
        "h100_resume_ready_file_sha256",
        "h100_resume_ready_body_sha256",
        "same_token_identity_sha256",
        "task8_live_h1d_identity_sha256",
        "task8_spend_authority_identity_sha256",
        "task8_spend_reserve_identity_sha256",
        "task9_launch_identity_sha256",
        "task9_admission_identity_sha256",
        "task9_custody_identity_sha256",
        "attestation_identity_sha256",
        "decision_seal_identity_sha256",
        "relay_envelope_sha256",
        "task_yaml_sha256",
        "request_body_sha256",
        "execution_input_sha256",
        "task13_route_binding_identity_sha256",
    ):
        _sha(getattr(value, field), field)
    for field in (
        "repository_archive_version_id",
        "approval_version_id",
        "intent_version_id",
        "task11_boundary_version_id",
    ):
        version = getattr(value, field)
        if (
            type(version) is not str
            or version == "null"
            or _VERSION.fullmatch(version) is None
        ):
            raise ProductionRouteError(field + " must be an opaque VersionId")
    if value.canonical_identity_sha256 != canonical_sha256(
        _authority_body(value)
    ):
        raise ProductionRouteError("production authority self-hash drifted")
    return value


def _closed_execution_input(
    authority: ProductionAuthority,
    task_inputs: MountFreeTaskInputs,
    worker_descriptor: WorkerBootstrapDescriptor,
) -> Mapping[str, object]:
    authority = validate_production_authority(authority)
    task_inputs = validate_task_inputs(task_inputs)
    worker_descriptor = validate_worker_bootstrap_descriptor(
        worker_descriptor
    )
    worker_descriptor_file_sha256 = hashlib.sha256(
        canonical_json_bytes(asdict(worker_descriptor)) + b"\n"
    ).hexdigest()
    if (
        worker_descriptor_file_sha256
        != task_inputs.descriptor_file_sha256
        or worker_descriptor.campaign_identity_sha256
        != authority.campaign_identity_sha256
        or worker_descriptor.activation_id != authority.activation_id
        or worker_descriptor.activation_ordinal
        != authority.activation_ordinal
        or worker_descriptor.generation != authority.generation
        or worker_descriptor.generation_text != authority.generation_text
        or worker_descriptor.action_key != authority.action_key
        or worker_descriptor.sky_job_name != authority.sky_job_name
        or worker_descriptor.execution_deadline
        != authority.execution_deadline
        or worker_descriptor.gpu_allocation_sha256
        != authority.gpu_allocation_sha256
        or worker_descriptor.descriptor_body_sha256
        != authority.descriptor_identity_sha256
        or worker_descriptor.archive_identity_sha256
        != authority.archive_identity_sha256
        or worker_descriptor.archive_identity_sha256
        != task_inputs.repository_archive_file_sha256
        or worker_descriptor.repository_archive_version_id
        != authority.repository_archive_version_id
        or worker_descriptor.repository_archive_version_id
        != task_inputs.repository_archive_version_id
        or worker_descriptor.approval_identity_sha256
        != authority.approval_identity_sha256
        or worker_descriptor.approval_identity_sha256
        != task_inputs.approval_file_sha256
        or worker_descriptor.approval_version_id
        != authority.approval_version_id
        or worker_descriptor.approval_version_id
        != task_inputs.approval_version_id
        or worker_descriptor.intent_identity_sha256
        != authority.intent_identity_sha256
        or worker_descriptor.intent_identity_sha256
        != task_inputs.intent_body_sha256
        or worker_descriptor.intent_version_id
        != authority.intent_version_id
        or worker_descriptor.intent_version_id
        != task_inputs.intent_version_id
        or worker_descriptor.task8_live_h1d_identity_sha256
        != authority.task8_live_h1d_identity_sha256
        or worker_descriptor.task8_spend_authority_identity_sha256
        != authority.task8_spend_authority_identity_sha256
        or worker_descriptor.task9_launch_identity_sha256
        != authority.task9_launch_identity_sha256
        or worker_descriptor.task9_admission_identity_sha256
        != authority.task9_admission_identity_sha256
        or worker_descriptor.task9_custody_identity_sha256
        != authority.task9_custody_identity_sha256
        or worker_descriptor.sky_job_name != task_inputs.job_name
        or authority.sky_job_name != task_inputs.job_name
    ):
        raise ProductionRouteError(
            "production worker descriptor authority drifted"
        )
    task_yaml = render_mount_free_task(task_inputs)
    body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=task_inputs.job_name,
    )
    if (
        authority.task_yaml_sha256
        != hashlib.sha256(task_yaml.encode("utf-8")).hexdigest()
        or authority.request_body_sha256 != canonical_sha256(body)
    ):
        raise ProductionRouteError("task YAML or one-wire body identity drifted")
    value = {
        "schema_version": 1,
        "record_type": "glm52_task10_execution_input_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": authority.campaign_identity_sha256,
        "activation_id": authority.activation_id,
        "activation_ordinal": authority.activation_ordinal,
        "generation": authority.generation,
        "generation_text": authority.generation_text,
        "sky_job_name": authority.sky_job_name,
        "execution_deadline": authority.execution_deadline,
        "gpu_allocation_sha256": authority.gpu_allocation_sha256,
        "executor_epoch": authority.executor_epoch,
        "action_key": authority.action_key,
        "authority_bindings_sha256": _authority_bindings_sha256(authority),
        "artifact_identities": {
            "descriptor": authority.descriptor_identity_sha256,
            "archive": authority.archive_identity_sha256,
            "approval": authority.approval_identity_sha256,
            "intent": authority.intent_identity_sha256,
            "h100_resume_ready": {
                "bucket": authority.h100_resume_ready_bucket,
                "key": authority.h100_resume_ready_key,
                "version_id": authority.h100_resume_ready_version_id,
                "file_sha256": authority.h100_resume_ready_file_sha256,
                "body_sha256": authority.h100_resume_ready_body_sha256,
            },
        },
        "production_capacity_retry": {
            "availability_zones": authority.capacity_retry_availability_zones,
            "outcome_rules": authority.capacity_retry_outcome_rules,
            "maximum_attempts": authority.maximum_capacity_attempts,
            "same_token_identity_sha256": authority.same_token_identity_sha256,
            "stop_on_first_worker_success": authority.stop_on_first_worker_success,
            "automatic_spot_fallback": authority.automatic_spot_fallback,
            "capacity_block_allowed": authority.capacity_block_allowed,
        },
        "task13_route_binding_identity_sha256": (
            authority.task13_route_binding_identity_sha256
        ),
        "artifact_version_ids": {
            "archive": authority.repository_archive_version_id,
            "approval": authority.approval_version_id,
            "intent": authority.intent_version_id,
        },
        "task11_boundary": {
            "bucket": authority.task11_boundary_bucket,
            "key": authority.task11_boundary_key,
            "version_id": authority.task11_boundary_version_id,
            "file_sha256": authority.task11_boundary_file_sha256,
            "body_sha256": authority.task11_boundary_body_sha256,
        },
        "task8_identities": {
            "live_h1d": authority.task8_live_h1d_identity_sha256,
            "spend_authority": (
                authority.task8_spend_authority_identity_sha256
            ),
            "spend_reserve": authority.task8_spend_reserve_identity_sha256,
        },
        "task9_identities": {
            "launch": authority.task9_launch_identity_sha256,
            "admission": authority.task9_admission_identity_sha256,
            "custody": authority.task9_custody_identity_sha256,
        },
        "attestation_identity_sha256": (
            authority.attestation_identity_sha256
        ),
        "decision_seal_identity_sha256": (
            authority.decision_seal_identity_sha256
        ),
        "relay_envelope_sha256": authority.relay_envelope_sha256,
        "task_yaml_sha256": authority.task_yaml_sha256,
        "request_body_sha256": authority.request_body_sha256,
        "workflow_version_arn": authority.workflow_version_arn,
        "execution_name": authority.execution_name,
        "expected_execution_arn": authority.expected_execution_arn,
        "jobs_launch_body": body,
    }
    if canonical_sha256(value) != authority.execution_input_sha256:
        raise ProductionRouteError("production execution input identity drifted")
    return value


def build_execution_input_identity(
    authority_without_input_hash: ProductionAuthority,
    task_inputs: MountFreeTaskInputs,
) -> str:
    """Derive the execution-input hash while constructing fixtures/builders.

    The supplied authority may carry any syntactically valid placeholder in
    ``execution_input_sha256``.  No runtime route uses this helper to weaken
    validation; it is only a deterministic builder companion.
    """

    authority = validate_production_authority(authority_without_input_hash)
    task_inputs = validate_task_inputs(task_inputs)
    task_yaml = render_mount_free_task(task_inputs)
    body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=task_inputs.job_name,
    )
    value = {
        "schema_version": 1,
        "record_type": "glm52_task10_execution_input_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": authority.campaign_identity_sha256,
        "activation_id": authority.activation_id,
        "activation_ordinal": authority.activation_ordinal,
        "generation": authority.generation,
        "generation_text": authority.generation_text,
        "sky_job_name": authority.sky_job_name,
        "execution_deadline": authority.execution_deadline,
        "gpu_allocation_sha256": authority.gpu_allocation_sha256,
        "executor_epoch": authority.executor_epoch,
        "action_key": authority.action_key,
        "authority_bindings_sha256": _authority_bindings_sha256(authority),
        "artifact_identities": {
            "descriptor": authority.descriptor_identity_sha256,
            "archive": authority.archive_identity_sha256,
            "approval": authority.approval_identity_sha256,
            "intent": authority.intent_identity_sha256,
            "h100_resume_ready": {
                "bucket": authority.h100_resume_ready_bucket,
                "key": authority.h100_resume_ready_key,
                "version_id": authority.h100_resume_ready_version_id,
                "file_sha256": authority.h100_resume_ready_file_sha256,
                "body_sha256": authority.h100_resume_ready_body_sha256,
            },
        },
        "production_capacity_retry": {
            "availability_zones": authority.capacity_retry_availability_zones,
            "outcome_rules": authority.capacity_retry_outcome_rules,
            "maximum_attempts": authority.maximum_capacity_attempts,
            "same_token_identity_sha256": authority.same_token_identity_sha256,
            "stop_on_first_worker_success": authority.stop_on_first_worker_success,
            "automatic_spot_fallback": authority.automatic_spot_fallback,
            "capacity_block_allowed": authority.capacity_block_allowed,
        },
        "task13_route_binding_identity_sha256": (
            authority.task13_route_binding_identity_sha256
        ),
        "artifact_version_ids": {
            "archive": authority.repository_archive_version_id,
            "approval": authority.approval_version_id,
            "intent": authority.intent_version_id,
        },
        "task11_boundary": {
            "bucket": authority.task11_boundary_bucket,
            "key": authority.task11_boundary_key,
            "version_id": authority.task11_boundary_version_id,
            "file_sha256": authority.task11_boundary_file_sha256,
            "body_sha256": authority.task11_boundary_body_sha256,
        },
        "task8_identities": {
            "live_h1d": authority.task8_live_h1d_identity_sha256,
            "spend_authority": (
                authority.task8_spend_authority_identity_sha256
            ),
            "spend_reserve": authority.task8_spend_reserve_identity_sha256,
        },
        "task9_identities": {
            "launch": authority.task9_launch_identity_sha256,
            "admission": authority.task9_admission_identity_sha256,
            "custody": authority.task9_custody_identity_sha256,
        },
        "attestation_identity_sha256": (
            authority.attestation_identity_sha256
        ),
        "decision_seal_identity_sha256": (
            authority.decision_seal_identity_sha256
        ),
        "relay_envelope_sha256": authority.relay_envelope_sha256,
        "task_yaml_sha256": authority.task_yaml_sha256,
        "request_body_sha256": authority.request_body_sha256,
        "workflow_version_arn": authority.workflow_version_arn,
        "execution_name": authority.execution_name,
        "expected_execution_arn": authority.expected_execution_arn,
        "jobs_launch_body": body,
    }
    return canonical_sha256(value)


_INSPECTION_FIELDS = frozenset(
    {
        "current_activation",
        "action_consumed",
        "task8_live_authority",
        "task8_spend_authority",
        "task8_spend_reserve",
        "task9_launch_admission",
        "task9_liability_custody",
        "workflow_version_arn",
        "inspection_identity_sha256",
    }
)


def _inspect_boundary(
    boundary: object,
    authority: ProductionAuthority,
    execution_input: Mapping[str, object],
) -> Mapping[str, object]:
    inspect = getattr(boundary, "inspect", None)
    if not callable(inspect):
        raise ProductionRouteError("production workflow inspection is absent")
    result = inspect(authority, execution_input)
    if type(result) is not dict or set(result) != _INSPECTION_FIELDS:
        raise ProductionRouteError("production workflow inspection is not closed")
    if (
        result["current_activation"] is not True
        or result["action_consumed"] is not True
        or result["task8_live_authority"]
        != authority.task8_live_h1d_identity_sha256
        or result["task8_spend_authority"]
        != authority.task8_spend_authority_identity_sha256
        or result["task8_spend_reserve"]
        != authority.task8_spend_reserve_identity_sha256
        or result["task9_launch_admission"]
        != authority.task9_admission_identity_sha256
        or result["task9_liability_custody"]
        != authority.task9_custody_identity_sha256
        or result["workflow_version_arn"] != authority.workflow_version_arn
    ):
        raise ProductionRouteError("Task 8/9 current authority is absent")
    expected_inspection = canonical_sha256(
        {
            key: result[key]
            for key in sorted(_INSPECTION_FIELDS)
            if key != "inspection_identity_sha256"
        }
    )
    if result["inspection_identity_sha256"] != expected_inspection:
        raise ProductionRouteError("production inspection identity drifted")
    return result


def _result(
    *,
    status: str,
    authority: ProductionAuthority,
    extra: Optional[Mapping[str, object]] = None,
) -> ProductionOutcome:
    detail = {
        "managed_mode": "production",
        "activation_id": authority.activation_id,
        "generation": authority.generation,
        "generation_text": authority.generation_text,
        "authority_identity_sha256": authority.canonical_identity_sha256,
        "workflow_version_arn": authority.workflow_version_arn,
        "expected_execution_arn": authority.expected_execution_arn,
    }
    if extra is not None:
        detail.update(extra)
    exit_code = 75 if status == "reconcile-required" else 0
    return ProductionOutcome(exit_code, status, detail)


def _validate_capacity_reconciliation(
    observation: object,
    authority: ProductionAuthority,
) -> Mapping[str, object]:
    fields = {
        "classification", "execution_arn", "capacity_outcomes", "instance_id",
        "availability_zone", "workflow_output", "same_token_identity_sha256",
        "reserve_identity_sha256", "spend_authority_identity_sha256",
        "action_identity_sha256", "task9_custody_identity_sha256",
        "observation_identity_sha256",
    }
    if type(observation) is not dict or set(observation) != fields:
        raise ProductionRouteError("production reconciliation field set drifted")
    if (
        observation["execution_arn"] != authority.expected_execution_arn
        or observation["classification"] not in {
            "WORKER_ALLOCATED", "CAPACITY_EXHAUSTED", "RUNNING", "FAILED"
        }
        or observation["same_token_identity_sha256"]
        != authority.same_token_identity_sha256
        or observation["reserve_identity_sha256"]
        != authority.task8_spend_reserve_identity_sha256
        or observation["spend_authority_identity_sha256"]
        != authority.task8_spend_authority_identity_sha256
        or observation["action_identity_sha256"] != authority.task9_launch_identity_sha256
        or observation["task9_custody_identity_sha256"]
        != authority.task9_custody_identity_sha256
    ):
        raise ProductionRouteError("production reconciliation authority drifted")
    outcomes = observation["capacity_outcomes"]
    if type(outcomes) is not list or len(outcomes) > authority.maximum_capacity_attempts:
        raise ProductionRouteError("production capacity outcomes drifted")
    zones = authority.capacity_retry_availability_zones
    for index, outcome in enumerate(outcomes, 1):
        if (
            type(outcome) is not dict
            or set(outcome) != {"attempt", "availability_zone", "outcome"}
            or outcome["attempt"] != index
            or outcome["availability_zone"] != zones[index - 1]
            or outcome["outcome"] not in {"CAPACITY_REJECTED", "WORKER_ALLOCATED"}
        ):
            raise ProductionRouteError("production capacity outcome is not ordered")
    output = observation["workflow_output"]
    if (
        type(output) is not dict
        or set(output) != {"bucket", "key", "version_id", "file_sha256", "body_sha256"}
        or output["bucket"] != authority.h100_resume_ready_bucket
        or type(output["key"]) is not str
        or not output["key"]
        or type(output["version_id"]) is not str
        or not output["version_id"]
        or output["version_id"] == "null"
    ):
        raise ProductionRouteError("production workflow output coordinate drifted")
    _sha(output["file_sha256"], "workflow output file")
    _sha(output["body_sha256"], "workflow output body")
    classification = observation["classification"]
    allocated = [item for item in outcomes if item["outcome"] == "WORKER_ALLOCATED"]
    if classification == "WORKER_ALLOCATED":
        if (
            len(allocated) != 1
            or outcomes[-1] != allocated[0]
            or type(observation["instance_id"]) is not str
            or _INSTANCE.fullmatch(observation["instance_id"]) is None
            or observation["availability_zone"] != allocated[0]["availability_zone"]
        ):
            raise ProductionRouteError("worker allocation reconciliation drifted")
    elif classification == "CAPACITY_EXHAUSTED":
        if (
            len(outcomes) != authority.maximum_capacity_attempts
            or allocated
            or observation["instance_id"] is not None
            or observation["availability_zone"] is not None
        ):
            raise ProductionRouteError("capacity exhaustion reconciliation drifted")
    elif (
        allocated
        or observation["instance_id"] is not None
        or observation["availability_zone"] is not None
    ):
        raise ProductionRouteError("nonterminal reconciliation drifted")
    identity = {
        key: observation[key] for key in sorted(fields)
        if key != "observation_identity_sha256"
    }
    if observation["observation_identity_sha256"] != canonical_sha256(identity):
        raise ProductionRouteError("production reconciliation identity drifted")
    return observation


def run_production(
    request: ProductionRequest,
    *,
    boundary: object,
) -> ProductionOutcome:
    """Validate or execute one closed production workflow operation."""

    if not isinstance(request, ProductionRequest):
        raise ProductionRouteError("production request must be typed")
    if request.action not in PRODUCTION_ACTIONS:
        raise ProductionRouteError("production action is invalid")
    if request.profile != "keep-gpu":
        raise ProductionRouteError("production profile is invalid")
    authority = validate_production_authority(request.authority)
    task_inputs = validate_task_inputs(request.task_inputs)
    execution_input = _closed_execution_input(
        authority,
        task_inputs,
        request.worker_descriptor,
    )
    inspection = _inspect_boundary(boundary, authority, execution_input)
    if request.action == "validate-only":
        return _result(
            status="validated",
            authority=authority,
            extra={
                "inspection_identity_sha256": (
                    inspection["inspection_identity_sha256"]
                )
            },
        )
    if request.action == "reconcile":
        reconcile = getattr(boundary, "reconcile", None)
        if not callable(reconcile):
            raise ProductionRouteError(
                "production reconciliation boundary is absent"
            )
        observation = reconcile(authority, execution_input)
        observation = _validate_capacity_reconciliation(observation, authority)
        return _result(
            status=(
                "reconcile-required"
                if observation["classification"] == "RUNNING"
                else "reconciled"
            ),
            authority=authority,
            extra=observation,
        )
    start_once = getattr(boundary, "start_once", None)
    if not callable(start_once):
        raise ProductionRouteError("guarded production start boundary is absent")
    observation = start_once(authority, execution_input)
    if (
        type(observation) is not dict
        or set(observation)
        != {
            "classification",
            "execution_arn",
            "reserve_identity_sha256",
            "task9_custody_identity_sha256",
            "observation_identity_sha256",
        }
        or observation["execution_arn"] != authority.expected_execution_arn
        or observation["reserve_identity_sha256"]
        != authority.task8_spend_reserve_identity_sha256
        or observation["task9_custody_identity_sha256"]
        != authority.task9_custody_identity_sha256
        or observation["classification"]
        not in {"STARTED", "KNOWN_REJECTED", "AMBIGUOUS"}
    ):
        raise ProductionRouteError("guarded production start result drifted")
    identity_body = {
        key: observation[key]
        for key in sorted(observation)
        if key != "observation_identity_sha256"
    }
    if observation["observation_identity_sha256"] != canonical_sha256(
        identity_body
    ):
        raise ProductionRouteError("guarded production result identity drifted")
    return _result(
        status=(
            "reconcile-required"
            if observation["classification"] == "AMBIGUOUS"
            else (
                "started"
                if observation["classification"] == "STARTED"
                else "known-rejected"
            )
        ),
        authority=authority,
        extra=observation,
    )


def assert_no_raw_effect_surface(value: object) -> None:
    """Reject a boundary that exposes legacy/public effect methods."""

    forbidden = (
        "sky",
        "http",
        "post",
        "subprocess",
        "run_instances",
        "send_command",
        "jobs_launch",
    )
    exposed = [
        name
        for name in forbidden
        if callable(getattr(value, name, None))
    ]
    if exposed:
        raise ProductionRouteError(
            "production boundary exposes raw effects: " + ",".join(exposed)
        )


__all__ = [
    "PRODUCTION_ACTIONS",
    "ProductionAuthority",
    "ProductionOutcome",
    "ProductionRequest",
    "ProductionRouteError",
    "assert_no_raw_effect_surface",
    "build_execution_input_identity",
    "build_production_authority",
    "production_authority_from_mapping",
    "run_production",
    "validate_production_authority",
]
