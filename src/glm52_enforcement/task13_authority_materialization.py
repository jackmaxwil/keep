"""Pre-package Task 10 authority and Task 13 route-binding materialization."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from .canonical import canonical_json_bytes, canonical_sha256
from .glm52_h100_qualification import validate_h100_resume_ready
from .task10_durable_s3 import publish_or_adopt_exact
from .task10_production import (
    build_execution_input_identity,
    build_production_authority,
    production_authority_from_mapping,
)
from .task10_worker import (
    MountFreeTaskInputs,
    WorkerBootstrapDescriptor,
    build_jobs_launch_body,
    render_mount_free_task,
    validate_task_inputs,
    validate_worker_bootstrap_descriptor,
    worker_bootstrap_descriptor_from_mapping,
)
from .task11_boundary import (
    BOUNDARY_INPUT_KINDS,
    Task11BoundaryDocument,
    validate_task11_boundary_document,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
AUTHORITY_KEY = "task13/production/task10-production-authority.json"
JOURNAL_SCHEMA = "glm52_task13_authority_materialization_journal_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_ID = re.compile(r"[A-Za-z0-9._-]{3,1024}\Z")
_MUTABLE_VERSION_WORDS = {
    "",
    "null",
    "none",
    "latest",
    "$latest",
    "current",
    "current-version",
}
_PUBLIC_INPUT_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "profile",
    "run_id",
    "production_authority_path",
    "production_authority_file_sha256",
    "production_authority_body_sha256",
    "production_task_inputs_path",
    "production_task_inputs_file_sha256",
    "production_task_inputs_body_sha256",
    "production_descriptor_path",
    "production_descriptor_file_sha256",
    "production_descriptor_body_sha256",
    "h100_resume_ready_coordinate_path",
    "h100_resume_ready_coordinate_file_sha256",
    "h100_resume_ready_coordinate_body_sha256",
    "task13_route_binding_identity_path",
    "task13_route_binding_identity_file_sha256",
    "task13_route_binding_identity_body_sha256",
    "task10_authority_coordinate",
    "task13_route_binding_identity_sha256",
    "canonical_identity_sha256",
}


class AuthorityMaterializationError(ValueError):
    """Authenticated authority sources do not close one production route."""


def _public_raw(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _public_path(value: object, label: str) -> Path:
    if type(value) is not str:
        raise AuthorityMaterializationError(label + " path is not exact")
    path = Path(value)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise AuthorityMaterializationError(
            label + " path is not an absolute regular file"
        )
    return path


def _public_canonical_file(path: Path, label: str) -> tuple[object, bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuthorityMaterializationError(label + " is not JSON") from error
    if raw != _public_raw(value):
        raise AuthorityMaterializationError(
            label + " is not canonical JSON plus LF"
        )
    return value, raw


def _h100_coordinate(value: object) -> dict[str, object]:
    fields = {
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["bucket"] != BUCKET
        or value["key"]
        != (
            "campaigns/%s/qualification/H100_RESUME_READY.json"
            % RUN_ID
        )
        or type(value["version_id"]) is not str
        or value["version_id"].lower() in _MUTABLE_VERSION_WORDS
        or _VERSION_ID.fullmatch(value["version_id"]) is None
    ):
        raise AuthorityMaterializationError(
            "H100 resume-ready coordinate drifted"
        )
    _sha(value["file_sha256"], "H100 resume-ready file")
    _sha(value["body_sha256"], "H100 resume-ready body")
    return dict(value)


def _task10_authority_coordinate(value: object) -> dict[str, object]:
    fields = {
        "artifact_kind",
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["artifact_kind"] != "TASK10_PRODUCTION_AUTHORITY"
        or value["bucket"] != BUCKET
        or value["key"] != AUTHORITY_KEY
        or type(value["version_id"]) is not str
        or value["version_id"].lower() in _MUTABLE_VERSION_WORDS
        or _VERSION_ID.fullmatch(value["version_id"]) is None
    ):
        raise AuthorityMaterializationError(
            "Task10 authority coordinate drifted"
        )
    _sha(value["file_sha256"], "Task10 authority file")
    _sha(value["body_sha256"], "Task10 authority body")
    return dict(value)


def build_public_task10_inputs_manifest(
    *,
    production_authority_path: Path,
    authority: object,
    production_task_inputs_path: Path,
    task_inputs: MountFreeTaskInputs,
    production_descriptor_path: Path,
    worker_descriptor: WorkerBootstrapDescriptor,
    h100_resume_ready_coordinate_path: Path,
    h100_resume_ready_coordinate: Mapping[str, object],
    task13_route_binding_identity_path: Path,
    task10_authority_coordinate: Mapping[str, object],
    task13_route_binding_identity_sha256: str,
) -> dict[str, object]:
    """Describe every local input needed by the public guarded Task 10 route."""

    authority_value = asdict(
        production_authority_from_mapping(asdict(authority))
    )
    inputs = validate_task_inputs(task_inputs)
    descriptor = validate_worker_bootstrap_descriptor(worker_descriptor)
    inputs_value = asdict(inputs)
    descriptor_value = asdict(descriptor)
    h100 = _h100_coordinate(h100_resume_ready_coordinate)
    coordinate = _task10_authority_coordinate(
        task10_authority_coordinate
    )
    route_identity = _sha(
        task13_route_binding_identity_sha256,
        "Task13 route binding",
    )
    paths = (
        Path(production_authority_path),
        Path(production_task_inputs_path),
        Path(production_descriptor_path),
        Path(h100_resume_ready_coordinate_path),
        Path(task13_route_binding_identity_path),
    )
    if (
        len(set(paths)) != 5
        or any(not path.is_absolute() for path in paths)
    ):
        raise AuthorityMaterializationError(
            "public Task10 output paths are not exact"
        )
    authority_raw = _public_raw(authority_value)
    inputs_raw = _public_raw(inputs_value)
    descriptor_raw = _public_raw(descriptor_value)
    h100_raw = _public_raw(h100)
    route_body = {
        "schema_version": 1,
        "record_type": "glm52_task13_route_binding_identity_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "task13_route_binding_identity_sha256": route_identity,
    }
    route_value = {
        **route_body,
        "canonical_identity_sha256": canonical_sha256(route_body),
    }
    route_raw = _public_raw(route_value)
    if (
        coordinate["file_sha256"]
        != hashlib.sha256(authority_raw).hexdigest()
        or coordinate["body_sha256"]
        != authority_value["canonical_identity_sha256"]
        or authority_value["task13_route_binding_identity_sha256"]
        != route_identity
        or inputs.descriptor_file_sha256
        != hashlib.sha256(descriptor_raw).hexdigest()
        or authority_value["descriptor_identity_sha256"]
        != descriptor.descriptor_body_sha256
        or authority_value["h100_resume_ready_bucket"] != h100["bucket"]
        or authority_value["h100_resume_ready_key"] != h100["key"]
        or authority_value["h100_resume_ready_version_id"]
        != h100["version_id"]
        or authority_value["h100_resume_ready_file_sha256"]
        != h100["file_sha256"]
        or authority_value["h100_resume_ready_body_sha256"]
        != h100["body_sha256"]
    ):
        raise AuthorityMaterializationError(
            "public Task10 local inputs disagree"
        )
    body = {
        "schema_version": 1,
        "record_type": "glm52_task13_public_task10_inputs_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "production_authority_path": str(paths[0]),
        "production_authority_file_sha256": hashlib.sha256(
            authority_raw
        ).hexdigest(),
        "production_authority_body_sha256": authority_value[
            "canonical_identity_sha256"
        ],
        "production_task_inputs_path": str(paths[1]),
        "production_task_inputs_file_sha256": hashlib.sha256(
            inputs_raw
        ).hexdigest(),
        "production_task_inputs_body_sha256": canonical_sha256(
            inputs_value
        ),
        "production_descriptor_path": str(paths[2]),
        "production_descriptor_file_sha256": hashlib.sha256(
            descriptor_raw
        ).hexdigest(),
        "production_descriptor_body_sha256": (
            descriptor.descriptor_body_sha256
        ),
        "h100_resume_ready_coordinate_path": str(paths[3]),
        "h100_resume_ready_coordinate_file_sha256": hashlib.sha256(
            h100_raw
        ).hexdigest(),
        "h100_resume_ready_coordinate_body_sha256": canonical_sha256(
            h100
        ),
        "task13_route_binding_identity_path": str(paths[4]),
        "task13_route_binding_identity_file_sha256": hashlib.sha256(
            route_raw
        ).hexdigest(),
        "task13_route_binding_identity_body_sha256": route_value[
            "canonical_identity_sha256"
        ],
        "task10_authority_coordinate": coordinate,
        "task13_route_binding_identity_sha256": route_identity,
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def validate_public_task10_inputs_manifest(
    value: object,
) -> dict[str, object]:
    """Exact-read and cross-bind a public Task 10 local-input bundle."""

    if type(value) is not dict or set(value) != _PUBLIC_INPUT_FIELDS:
        raise AuthorityMaterializationError(
            "public Task10 input manifest fields drifted"
        )
    body = dict(value)
    supplied_identity = body.pop("canonical_identity_sha256")
    if (
        value["schema_version"] != 1
        or value["record_type"]
        != "glm52_task13_public_task10_inputs_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["profile"] != PROFILE
        or value["run_id"] != RUN_ID
        or supplied_identity != canonical_sha256(body)
    ):
        raise AuthorityMaterializationError(
            "public Task10 input manifest identity drifted"
        )
    paths = [
        _public_path(value[field], field)
        for field in (
            "production_authority_path",
            "production_task_inputs_path",
            "production_descriptor_path",
            "h100_resume_ready_coordinate_path",
            "task13_route_binding_identity_path",
        )
    ]
    if len(set(paths)) != 5:
        raise AuthorityMaterializationError(
            "public Task10 local paths overlap"
        )
    loaded = [
        _public_canonical_file(path, label)
        for path, label in zip(
            paths,
            (
                "production authority",
                "production task inputs",
                "production descriptor",
                "H100 resume-ready coordinate",
                "Task13 route-binding identity",
            ),
        )
    ]
    authority_value, authority_raw = loaded[0]
    inputs_value, inputs_raw = loaded[1]
    descriptor_value, descriptor_raw = loaded[2]
    h100_value, h100_raw = loaded[3]
    route_value, route_raw = loaded[4]
    try:
        authority = production_authority_from_mapping(authority_value)
        if (
            type(inputs_value) is not dict
            or set(inputs_value)
            != set(MountFreeTaskInputs.__dataclass_fields__)
        ):
            raise AuthorityMaterializationError(
                "public Task10 task-input fields drifted"
            )
        inputs = validate_task_inputs(MountFreeTaskInputs(**inputs_value))
        descriptor = worker_bootstrap_descriptor_from_mapping(
            descriptor_value
        )
    except (TypeError, ValueError) as error:
        raise AuthorityMaterializationError(
            "public Task10 local input schema drifted"
        ) from error
    h100 = _h100_coordinate(h100_value)
    route_fields = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "profile",
        "run_id",
        "task13_route_binding_identity_sha256",
        "canonical_identity_sha256",
    }
    route_body = (
        {
            key: item
            for key, item in route_value.items()
            if key != "canonical_identity_sha256"
        }
        if type(route_value) is dict
        else {}
    )
    if (
        type(route_value) is not dict
        or set(route_value) != route_fields
        or route_value["schema_version"] != 1
        or route_value["record_type"]
        != "glm52_task13_route_binding_identity_v1"
        or route_value["account_id"] != ACCOUNT_ID
        or route_value["region"] != REGION
        or route_value["profile"] != PROFILE
        or route_value["run_id"] != RUN_ID
        or route_value["task13_route_binding_identity_sha256"]
        != value["task13_route_binding_identity_sha256"]
        or route_value["canonical_identity_sha256"]
        != canonical_sha256(route_body)
    ):
        raise AuthorityMaterializationError(
            "Task13 route-binding identity file drifted"
        )
    coordinate = _task10_authority_coordinate(
        value["task10_authority_coordinate"]
    )
    for raw, file_field in (
        (authority_raw, "production_authority_file_sha256"),
        (inputs_raw, "production_task_inputs_file_sha256"),
        (descriptor_raw, "production_descriptor_file_sha256"),
        (
            h100_raw,
            "h100_resume_ready_coordinate_file_sha256",
        ),
        (
            route_raw,
            "task13_route_binding_identity_file_sha256",
        ),
    ):
        if hashlib.sha256(raw).hexdigest() != value[file_field]:
            raise AuthorityMaterializationError(
                "public Task10 local file identity drifted"
            )
    rebuilt = build_public_task10_inputs_manifest(
        production_authority_path=paths[0],
        authority=authority,
        production_task_inputs_path=paths[1],
        task_inputs=inputs,
        production_descriptor_path=paths[2],
        worker_descriptor=descriptor,
        h100_resume_ready_coordinate_path=paths[3],
        h100_resume_ready_coordinate=h100,
        task13_route_binding_identity_path=paths[4],
        task10_authority_coordinate=coordinate,
        task13_route_binding_identity_sha256=str(
            value["task13_route_binding_identity_sha256"]
        ),
    )
    if rebuilt != value:
        raise AuthorityMaterializationError(
            "public Task10 local input manifest drifted"
        )
    return dict(value)


def _sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        raise AuthorityMaterializationError(label + " must be a SHA-256")
    return value


def _find_unique(value: object, field: str) -> object:
    found = []

    def visit(item: object) -> None:
        if type(item) is dict:
            if field in item:
                found.append(item[field])
            for child in item.values():
                visit(child)
        elif type(item) is list:
            for child in item:
                visit(child)

    visit(value)
    unique = {_canonical_key(item): item for item in found}
    if len(unique) != 1:
        raise AuthorityMaterializationError(
            "authenticated field is absent or ambiguous: " + field
        )
    return next(iter(unique.values()))


def _canonical_key(value: object) -> bytes:
    try:
        return canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise AuthorityMaterializationError(
            "authenticated source contains noncanonical data"
        ) from error


def build_task13_route_binding_identity(
    *,
    task11_boundary_coordinate: Mapping[str, object],
    task11_input_coordinates: Sequence[Mapping[str, object]],
    approval_coordinates: Mapping[str, Mapping[str, object]],
    workflow_inventory: Mapping[str, object],
    runner_file_sha256: str,
    coordinator_file_sha256: str,
) -> str:
    """Build the acyclic route identity which deliberately excludes package hashes."""

    if len(task11_input_coordinates) != len(BOUNDARY_INPUT_KINDS):
        raise AuthorityMaterializationError(
            "route binding requires all 14 Task11 source coordinates"
        )
    if {
        str(value.get("input_kind"))
        for value in task11_input_coordinates
    } != set(BOUNDARY_INPUT_KINDS):
        raise AuthorityMaterializationError(
            "route binding Task11 source kinds drifted"
        )
    if set(approval_coordinates) != {
        "GPU_SPEND_APPROVAL",
        "SUPPORT_APPROVAL",
        "RESIDUAL_LIABILITY_APPROVAL",
    }:
        raise AuthorityMaterializationError(
            "route binding approval set drifted"
        )
    expected_inventory = {
        "state_machine_logical_id": "Task10ProductionStateMachine",
        "state_machine_version_logical_id": (
            "Task10ProductionStateMachineVersion"
        ),
        "state_machine_name": "keep-glm52-h1g-production",
        "reconciliation_function_logical_id": (
            "Task10ProductionReconciliationFunction"
        ),
        "reconciliation_function_version_logical_id": (
            "Task10ProductionReconciliationFunctionVersion"
        ),
        "reconciliation_function_name": (
            "keep-glm52-h1g-task10-capacity-reconciliation"
        ),
        "terminal_writer_state": "RunInternalSixAzSoleSender",
        "writer_handler": (
            "aws/glm52-gpu/lambda/"
            "task10_sole_sender_handler.py"
        ),
    }
    if dict(workflow_inventory) != expected_inventory:
        raise AuthorityMaterializationError(
            "Task10 workflow writer inventory drifted"
        )
    body = {
        "schema_version": 1,
        "record_type": "glm52_task13_route_binding_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "task11_boundary_coordinate": dict(task11_boundary_coordinate),
        "task11_input_coordinates": [
            dict(value) for value in task11_input_coordinates
        ],
        "approval_coordinates": {
            key: dict(value)
            for key, value in sorted(approval_coordinates.items())
        },
        "workflow_inventory": expected_inventory,
        "runner_file_sha256": _sha(
            runner_file_sha256,
            "Task13 runner",
        ),
        "coordinator_file_sha256": _sha(
            coordinator_file_sha256,
            "Task13 coordinator",
        ),
        "journal_schema": JOURNAL_SCHEMA,
    }
    return canonical_sha256(body)


def build_task10_authority_from_sources(
    *,
    boundary: Task11BoundaryDocument,
    source_documents: Mapping[str, Mapping[str, object]],
    task_inputs: MountFreeTaskInputs,
    worker_descriptor: WorkerBootstrapDescriptor,
    task11_boundary_coordinate: Mapping[str, object],
    h100_resume_ready_coordinate: Mapping[str, object],
    h100_resume_ready: Mapping[str, object],
    workflow_readback: Mapping[str, object],
    task13_route_binding_identity_sha256: str,
) -> object:
    """Construct the Task10 authority only from exact authenticated sources."""

    boundary = validate_task11_boundary_document(boundary)
    task_inputs = validate_task_inputs(task_inputs)
    descriptor = validate_worker_bootstrap_descriptor(worker_descriptor)
    if set(source_documents) != set(BOUNDARY_INPUT_KINDS):
        raise AuthorityMaterializationError(
            "authority builder requires all 14 Task11 source documents"
        )
    sources = {
        key: dict(value) for key, value in source_documents.items()
    }
    try:
        h100_marker = validate_h100_resume_ready(h100_resume_ready)
    except ValueError as error:
        raise AuthorityMaterializationError(
            "H100 resume readiness schema drifted"
        ) from error
    if (
        h100_marker["run_id"] != RUN_ID
        or h100_marker["campaign_identity_sha256"]
        != boundary.campaign_identity_sha256
    ):
        raise AuthorityMaterializationError(
            "H100 resume readiness campaign identity drifted"
        )
    if (
        h100_marker["repo_tar_sha256"]
        != task_inputs.repository_archive_file_sha256
    ):
        raise AuthorityMaterializationError(
            "H100 resume readiness archive identity drifted"
        )
    if (
        h100_marker["qualification_cache_manifest_sha256"]
        != _find_unique(
            sources,
            "qualification_cache_manifest_sha256",
        )
    ):
        raise AuthorityMaterializationError(
            "H100 resume readiness qualification cache identity drifted"
        )
    if (
        h100_resume_ready_coordinate.get("body_sha256")
        != h100_marker["ready_body_sha256"]
    ):
        raise AuthorityMaterializationError(
            "H100 resume readiness body identity drifted"
        )
    workflow_arn = boundary.state_machine_version_arn
    if workflow_readback != {
        "state_machine_version_arn": workflow_arn,
        "state_machine_name": "keep-glm52-h1g-production",
        "status": "ACTIVE",
        "terminal_writer_state": "RunInternalSixAzSoleSender",
        "reconciliation_function_name": (
            "keep-glm52-h1g-task10-capacity-reconciliation"
        ),
    }:
        raise AuthorityMaterializationError(
            "retained Task10 workflow version readback drifted"
        )
    activation_ordinal = _find_unique(sources, "activation_ordinal")
    execution_deadline = _find_unique(sources, "execution_deadline")
    gpu_allocation_sha256 = _find_unique(
        sources,
        "gpu_allocation_sha256",
    )
    executor_epoch = _find_unique(sources, "executor_epoch")
    opaque_fields = {
        field: _find_unique(sources, field)
        for field in (
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
        )
    }
    if type(activation_ordinal) is not int or activation_ordinal <= 0:
        raise AuthorityMaterializationError("activation ordinal drifted")
    if type(executor_epoch) is not int or executor_epoch <= 0:
        raise AuthorityMaterializationError("executor epoch drifted")
    for field, value in opaque_fields.items():
        _sha(value, field)
    _sha(gpu_allocation_sha256, "GPU allocation")
    _sha(
        task13_route_binding_identity_sha256,
        "Task13 route binding",
    )
    generation = boundary.generation
    execution_name = (
        boundary.activation_id + "-epoch-" + f"{executor_epoch:08d}"
    )
    expected_execution_arn = (
        "arn:aws:states:us-west-2:246813579024:execution:"
        "keep-glm52-h1g-production:"
        + execution_name
    )
    task_yaml = render_mount_free_task(task_inputs)
    jobs_body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=task_inputs.job_name,
    )
    common = {
        "schema_version": 1,
        "record_type": "glm52_task10_production_authority_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": "production",
        "campaign_identity_sha256": boundary.campaign_identity_sha256,
        "activation_id": boundary.activation_id,
        "activation_ordinal": activation_ordinal,
        "generation": generation,
        "generation_text": boundary.generation_text,
        "sky_job_name": task_inputs.job_name,
        "execution_deadline": execution_deadline,
        "gpu_allocation_sha256": gpu_allocation_sha256,
        "executor_epoch": executor_epoch,
        "action_key": boundary.action_key,
        "action_state": "CONSUMED",
        "current_activation": True,
        "descriptor_identity_sha256": descriptor.descriptor_body_sha256,
        "archive_identity_sha256": (
            task_inputs.repository_archive_file_sha256
        ),
        "repository_archive_version_id": (
            task_inputs.repository_archive_version_id
        ),
        "approval_identity_sha256": task_inputs.approval_file_sha256,
        "approval_version_id": task_inputs.approval_version_id,
        "intent_identity_sha256": task_inputs.intent_body_sha256,
        "intent_version_id": task_inputs.intent_version_id,
        "task11_boundary_bucket": task11_boundary_coordinate["bucket"],
        "task11_boundary_key": task11_boundary_coordinate["key"],
        "task11_boundary_version_id": task11_boundary_coordinate[
            "version_id"
        ],
        "task11_boundary_file_sha256": task11_boundary_coordinate[
            "file_sha256"
        ],
        "task11_boundary_body_sha256": task11_boundary_coordinate[
            "body_sha256"
        ],
        "h100_resume_ready_bucket": h100_resume_ready_coordinate["bucket"],
        "h100_resume_ready_key": h100_resume_ready_coordinate["key"],
        "h100_resume_ready_version_id": h100_resume_ready_coordinate[
            "version_id"
        ],
        "h100_resume_ready_file_sha256": h100_resume_ready_coordinate[
            "file_sha256"
        ],
        "h100_resume_ready_body_sha256": h100_resume_ready_coordinate[
            "body_sha256"
        ],
        "capacity_retry_availability_zones": [
            "us-west-2" + letter for letter in "abcdef"
        ],
        "capacity_retry_outcome_rules": [
            {
                "availability_zone": "us-west-2" + letter,
                "on_capacity_failure": (
                    "ROTATE_NEXT_AZ"
                    if index < 6
                    else "STOP_CAPACITY_EXHAUSTED"
                ),
            }
            for index, letter in enumerate("abcdef", 1)
        ],
        "maximum_capacity_attempts": 6,
        **opaque_fields,
        "stop_on_first_worker_success": True,
        "automatic_spot_fallback": False,
        "capacity_block_allowed": False,
        "task_yaml_sha256": hashlib.sha256(
            task_yaml.encode("utf-8")
        ).hexdigest(),
        "request_body_sha256": canonical_sha256(jobs_body),
        "workflow_version_arn": workflow_arn,
        "execution_name": execution_name,
        "expected_execution_arn": expected_execution_arn,
        "task13_route_binding_identity_sha256": (
            task13_route_binding_identity_sha256
        ),
    }
    placeholder = build_production_authority(
        **common,
        execution_input_sha256="0" * 64,
    )
    return build_production_authority(
        **common,
        execution_input_sha256=build_execution_input_identity(
            placeholder,
            task_inputs,
        ),
    )


def publish_task10_authority(
    client: object,
    *,
    authority: object,
) -> Mapping[str, object]:
    """Publish or adopt the exact authority and return its reviewed coordinate."""

    raw = canonical_json_bytes(asdict(authority)) + b"\n"
    version_id = publish_or_adopt_exact(
        client,
        bucket=BUCKET,
        key=AUTHORITY_KEY,
        raw=raw,
        metadata={
            "record-type": "glm52_task10_production_authority_v1",
            "canonical-identity-sha256": getattr(
                authority,
                "canonical_identity_sha256",
            ),
        },
    )
    return {
        "artifact_kind": "TASK10_PRODUCTION_AUTHORITY",
        "bucket": BUCKET,
        "key": AUTHORITY_KEY,
        "version_id": version_id,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": getattr(authority, "canonical_identity_sha256"),
    }


def attach_task10_authority_coordinate(
    reviewed_artifacts: Sequence[Mapping[str, object]],
    coordinate: Mapping[str, object],
) -> list[dict[str, object]]:
    """Append the authority only after publication; replacement is forbidden."""

    if coordinate.get("artifact_kind") != "TASK10_PRODUCTION_AUTHORITY":
        raise AuthorityMaterializationError(
            "Task10 authority coordinate kind drifted"
        )
    if any(
        item.get("artifact_kind") == "TASK10_PRODUCTION_AUTHORITY"
        for item in reviewed_artifacts
    ):
        raise AuthorityMaterializationError(
            "Task10 authority coordinate already exists"
        )
    result = [dict(item) for item in reviewed_artifacts]
    result.append(dict(coordinate))
    result.sort(key=lambda item: str(item.get("artifact_kind")))
    return result


__all__ = [
    "AUTHORITY_KEY",
    "AuthorityMaterializationError",
    "attach_task10_authority_coordinate",
    "build_public_task10_inputs_manifest",
    "build_task10_authority_from_sources",
    "build_task13_route_binding_identity",
    "publish_task10_authority",
    "validate_public_task10_inputs_manifest",
]
