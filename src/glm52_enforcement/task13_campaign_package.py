"""Deterministic Task 13 disabled deployment and campaign rehearsal package.

This module only constructs and validates inert data.  It never imports an AWS
SDK, executes a command, writes a gate, deploys a stack, qualifies hardware, or
launches an instance.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import re
from dataclasses import asdict
from pathlib import PurePosixPath
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.cloudformation_stacks import STACK_TAGS
from glm52_enforcement.decision_closure import (
    ClosureRequest,
    validate_closure_request,
)
from glm52_enforcement.task11_boundary import Task11BoundaryCoordinate
from glm52_enforcement.task13_fixed_artifacts import (
    activation_artifact_key,
    repository_archive_manifest_key,
    validate_activation_artifact_key,
    validate_repository_archive_manifest_key,
)
from glm52_enforcement.task13_transport_gates import (
    SEMANTIC_GATE_PINS,
    TransportGateError,
    validate_semantic_gate_coordinate,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
CAMPAIGN_BUCKET = "keep-glm52-models-246813579024-us-west-2"
CAMPAIGN_OWNER_APPROVAL_SHA256 = (
    "23ccff3454896d7fb8e9d0722b6c76522ebf2dfa3dfea9856a17a3e48b26ebc1"
)
DEPLOYMENT_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/"
    "keep-glm52-h1g-cloudformation-deployment"
)
FENCE_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-service"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ACTIVATION = re.compile(r"[a-z0-9][a-z0-9-]{2,63}")
_VERSION_ID = re.compile(r"[A-Za-z0-9._-]{3,1024}")
_CHANGE_SET_NAME = re.compile(r"glm52-task13-(?:fence|support)-disabled-[0-9]{4}")
_COLLECTOR_ARN = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:function:"
    r"keep-glm52-h1g-rehearsal-collector:([1-9][0-9]*)"
)
_RETAINED_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"keep-glm52-gpu/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_FENCE_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"keep-glm52-h1g-fence/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_SUPPORT_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"keep-glm52-h1g-support/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_CRYPTOGRAPHY_LAYER_VERSION_ARN = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:layer:"
    r"keep-glm52-h1g-cryptography-py312-x86-64:[1-9][0-9]*"
)

_REQUEST_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "profile",
    "run_id",
    "activation_id",
    "collector_version_arn",
    "task11_request",
    "task11_boundary",
    "retained_stack_id",
    "fence_stack_name",
    "support_stack_name",
    "fence_change_set_name",
    "support_change_set_name",
    "retained_role_arn",
    "fence_role_arn",
    "support_role_arn",
    "monitor_descriptor_path",
    "artifacts",
    "staged_infrastructure_evidence",
}
_ARTIFACT_FIELDS = {
    "artifact_kind",
    "bucket",
    "key",
    "version_id",
    "file_sha256",
    "body_sha256",
}
_ARTIFACT_KEYS = {
    "TASK11_REVIEW_APPROVAL": "reviews/task11/approval.json",
    "TASK12_REVIEW_APPROVAL": "reviews/task12/approval.json",
    "RETAINED_FOUNDATION_TEMPLATE": (
        "task13/templates/retained-foundation.yaml"
    ),
    "RETAINED_PRE_SUPPORT_TEMPLATE": (
        "task13/templates/retained-pre-support.yaml"
    ),
    "RETAINED_TEMPLATE": "task13/templates/retained.yaml",
    "FENCE_TEMPLATE": "task13/migration/fence-transfer.json",
    "SUPPORT_TEMPLATE": "task13/templates/support-disabled.yaml",
    "SUPPORT_INPUTS": "task13/inputs/support-build-inputs.json",
    "BOOTSTRAP_TEMPLATE": "task13/templates/container-bootstrap-v1.json",
    "QUALIFICATION_CACHE_SEED_INPUT": "<activation-scoped>",
    "H100_QUALIFICATION_INPUT": "<activation-scoped>",
    "T01_T25_GATE": "task13/gates/t01-t25.json",
    "TRANSPORT_22_MUTANT_GATE": "task13/gates/transport-22-mutants.json",
    "REPOSITORY_ARCHIVE": "<activation-scoped>",
    "CLEAN_REHEARSAL": "<activation-scoped>",
    "ACCEPTED_BASELINE": "task13/inputs/accepted-baseline.json",
    "PROMPT_PACK": "task13/inputs/prompt-pack.json",
    "TRAINING_CONFIGURATION": "task13/inputs/training-configuration.json",
    "GPU_SPEND_APPROVAL": "task13/approvals/gpu-spend.json",
    "SUPPORT_APPROVAL": "task13/approvals/support-plane.json",
    "RESIDUAL_LIABILITY_APPROVAL": (
        "task13/approvals/residual-liability.json"
    ),
    "PRODUCTION_DESCRIPTOR": "<activation-scoped>",
    "TASK10_PRODUCTION_AUTHORITY": (
        "task13/production/task10-production-authority.json"
    ),
    "TASK10_WORKER_DESCRIPTOR": (
        "task13/production/task10-worker-descriptor.json"
    ),
    "TASK10_TASK_INPUTS": (
        "task13/production/task10-task-inputs.json"
    ),
}
_ACTIVATION_SCOPED_ARTIFACT_KINDS = frozenset(
    {
        "QUALIFICATION_CACHE_SEED_INPUT",
        "H100_QUALIFICATION_INPUT",
        "PRODUCTION_DESCRIPTOR",
    }
)
_PRODUCTION_ONLY_ARTIFACT_KIND_ORDER = (
    "CLEAN_REHEARSAL",
    "TASK10_PRODUCTION_AUTHORITY",
    "TASK10_WORKER_DESCRIPTOR",
    "TASK10_TASK_INPUTS",
)
PRODUCTION_ONLY_ARTIFACT_KINDS = frozenset(
    _PRODUCTION_ONLY_ARTIFACT_KIND_ORDER
)
_BASE_RETRY_IMMUTABLE_INPUT_KINDS = (
    "REPOSITORY_ARCHIVE",
    "ACCEPTED_BASELINE",
    "PROMPT_PACK",
    "TRAINING_CONFIGURATION",
    "GPU_SPEND_APPROVAL",
    "SUPPORT_APPROVAL",
    "RESIDUAL_LIABILITY_APPROVAL",
    "PRODUCTION_DESCRIPTOR",
)
_PRODUCTION_RETRY_IMMUTABLE_INPUT_KINDS = (
    *_BASE_RETRY_IMMUTABLE_INPUT_KINDS,
    *_PRODUCTION_ONLY_ARTIFACT_KIND_ORDER,
)
STAGED_INFRASTRUCTURE_ARTIFACT_KINDS = frozenset(
    {
        "BOOTSTRAP_TEMPLATE",
        "FENCE_TEMPLATE",
        "RETAINED_FOUNDATION_TEMPLATE",
        "RETAINED_PRE_SUPPORT_TEMPLATE",
        "RETAINED_TEMPLATE",
        "SUPPORT_TEMPLATE",
        "SUPPORT_INPUTS",
    }
)
_STAGED_DEPLOYMENT_STEPS = (
    "ACCOUNT_LIVE_BASELINE",
    "BOOTSTRAP_STACK_MIGRATION",
    "MATERIALIZE_PRE_SUPPORT",
    "UPDATE_RETAINED_PRE_SUPPORT",
    "MATERIALIZE_FULL_SUPPORT_INPUTS",
    "BUILD_PUBLISH_SUPPORT",
    "MATERIALIZE_PUBLISH_STACK_MIGRATION",
    "CAPTURE_PRECREATE_ORPHAN_AUTHORITY",
    "EXECUTE_STACK_MIGRATION",
    "MATERIALIZE_POSTCREATE_FRAGMENT",
    "UPDATE_RETAINED_FINAL",
    "POSTPUBLICATION_AUTHORITY",
    "FINAL_EXACT_READBACK",
    "PROVE_NO_WORKER_ACTIVATION",
)
_STAGED_EVIDENCE_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "profile",
    "run_id",
    "activation_id",
    "staged_request",
    "staged_request_identity_sha256",
    "staged_journal_path",
    "staged_journal_size_bytes",
    "staged_journal_sha256",
    "staged_journal_records",
    "completed_steps",
    "retained_foundation",
    "fence",
    "pre_support",
    "support_stack",
    "postcreate_final",
    "fixed_artifacts",
    "fixed_artifacts_identity_sha256",
    "final_readback",
    "canonical_identity_sha256",
}
_STAGED_PRODUCTION_REQUEST_FIELDS = {
    "schema_version",
    "record_type",
    "activation_id",
    "output_directory",
    "retained_stack_id",
    "stack_migration_seed",
    "bootstrap_template",
    "pre_support_runtime_inputs",
    "support_input_materialization_request",
    "support_price_card",
    "orphan_precreate",
    "retained_foundation_evidence",
    "fixed_artifacts",
    "support_artifact_publication",
}
_MUTABLE_VERSION_WORDS = {
    "",
    "null",
    "none",
    "latest",
    "$latest",
    "current",
    "current-version",
}
_REQUIRED_GATES = [
    "TASK11_REVIEW_APPROVED",
    "TASK12_REVIEW_APPROVED",
    "DISABLED_STACK_CHANGE_SETS_REVIEWED",
    "DISABLED_STACK_READBACK_PROVEN",
    "NEGATIVE_IAM_PROBES_PROVEN",
    "TASK11_REHEARSAL_GATE_PROVEN",
    "H100_QUALIFICATION_PROVEN",
    "P5_ZERO_PRELAUNCH_PROVEN",
    "SUPPORT_PLANE_HEALTHY",
    "CREDENTIAL_EXPIRY_SUFFICIENT",
    "T01_T25_PROVEN",
    "TRANSPORT_22_MUTANTS_PROVEN",
    "EXACT_ARCHIVE_PROVEN",
    "CLEAN_REHEARSAL_PROVEN",
]
_COMMAND_OPERATION_ALLOWLIST = [
    "cloudformation:CreateChangeSet",
    "cloudformation:DeleteChangeSet",
    "cloudformation:DescribeChangeSet",
    "cloudformation:DescribeStacks",
    "cloudformation:ExecuteChangeSet",
    "cloudformation:GetTemplate",
    "cloudformation:ListStackResources",
    "iam:GetRole",
    "sts:ExportCredentials",
    "sts:GetCallerIdentity",
]
_PRODUCTION_ROUTE = (
    "aws/glm52-gpu/scripts/submit_sky_campaign.sh",
    "--production",
    "start",
)
_REQUIRED_EXECUTABLE_PATHS = (
    "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py",
    "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh",
    "aws/glm52-gpu/scripts/submit_sky_campaign.py",
    "aws/glm52-gpu/scripts/submit_sky_campaign.sh",
    "src/glm52_enforcement/task10_production.py",
)
_COLLECT_RESULT_FIELDS = {
    "schema_version",
    "record_type",
    "status",
    "account_id",
    "region",
    "run_id",
    "activation_id",
    "measurement_id",
    "collector_function_version_arn",
    "key",
    "version_id",
    "file_sha256",
    "checksum_sha256_base64",
    "canonical_identity_sha256",
}
_FINALIZE_RESULT_FIELDS = {
    "schema_version",
    "record_type",
    "status",
    "account_id",
    "region",
    "run_id",
    "activation_id",
    "collector_function_version_arn",
    "key",
    "version_id",
    "file_sha256",
    "body_sha256",
    "checksum_sha256_base64",
    "measurement_count",
    "cold_environment_count",
    "measurements_identity_sha256",
    "canonical_identity_sha256",
}


class CampaignPackageError(ValueError):
    """The Task 13 package or rehearsal result is not exact."""


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CampaignPackageError("%s must be lowercase SHA-256" % field)
    return value


def _require_version_id(value: object, field: str = "version_id") -> str:
    if (
        type(value) is not str
        or value.lower() in _MUTABLE_VERSION_WORDS
        or _VERSION_ID.fullmatch(value) is None
    ):
        raise CampaignPackageError("%s is not an immutable version" % field)
    return value


def _require_text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise CampaignPackageError("%s is invalid" % field)
    return value


def _copy_json(value: object) -> Any:
    """Copy JSON-compatible input while rejecting custom mapping/list types."""

    try:
        encoded = canonical_json_bytes(value)
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise CampaignPackageError("value is not canonical JSON data") from error


def _validate_task11_bindings(
    request_value: object,
    boundary_value: object,
    *,
    activation_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    request_fields = set(ClosureRequest.__dataclass_fields__)
    if type(request_value) is not dict or set(request_value) != request_fields:
        raise CampaignPackageError("task11_request field set drifted")
    try:
        request = validate_closure_request(ClosureRequest(**request_value))
    except (TypeError, ValueError) as error:
        raise CampaignPackageError("task11_request identity drifted") from error
    if request.activation_id != activation_id:
        raise CampaignPackageError("task11_request activation drifted")

    boundary_fields = set(Task11BoundaryCoordinate.__dataclass_fields__)
    if type(boundary_value) is not dict or set(boundary_value) != boundary_fields:
        raise CampaignPackageError("task11_boundary field set drifted")
    try:
        boundary = Task11BoundaryCoordinate(**boundary_value)
    except TypeError as error:
        raise CampaignPackageError("task11_boundary is malformed") from error
    if (
        boundary.bucket != CAMPAIGN_BUCKET
        or boundary.key
        != (
            "campaigns/%s/authorities/task11/%s/%08d/"
            "TASK11_BOUNDARY.json"
            % (RUN_ID, activation_id, request.generation)
        )
    ):
        raise CampaignPackageError("task11_boundary coordinate drifted")
    _require_version_id(boundary.version_id, "task11_boundary.version_id")
    _require_sha256(boundary.file_sha256, "task11_boundary.file_sha256")
    _require_sha256(boundary.body_sha256, "task11_boundary.body_sha256")
    return asdict(request), asdict(boundary)


def _validate_coordinate(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _ARTIFACT_FIELDS:
        raise CampaignPackageError("artifact coordinate field set drifted")
    kind = _require_text(value["artifact_kind"], "artifact_kind")
    if kind not in _ARTIFACT_KEYS:
        raise CampaignPackageError("artifact kind is not allowed")
    if value["bucket"] != CAMPAIGN_BUCKET:
        raise CampaignPackageError("artifact bucket is not exact")
    if kind in _ACTIVATION_SCOPED_ARTIFACT_KINDS:
        try:
            validate_activation_artifact_key(
                value["key"],
                artifact_kind=kind,
            )
        except ValueError as error:
            raise CampaignPackageError("artifact key is not exact") from error
    elif kind == "REPOSITORY_ARCHIVE":
        try:
            validate_repository_archive_manifest_key(value["key"])
        except ValueError as error:
            raise CampaignPackageError("artifact key is not exact") from error
    elif kind == "CLEAN_REHEARSAL":
        try:
            from .task13_clean_rehearsal import validate_clean_rehearsal_key

            validate_clean_rehearsal_key(value["key"])
        except ValueError as error:
            raise CampaignPackageError("artifact key is not exact") from error
    elif value["key"] != _ARTIFACT_KEYS[kind]:
        raise CampaignPackageError("artifact key is not exact")
    _require_version_id(value["version_id"])
    _require_sha256(value["file_sha256"], "file_sha256")
    _require_sha256(value["body_sha256"], "body_sha256")
    return _copy_json(value)


def validate_campaign_artifact_coordinate(
    value: object,
) -> dict[str, object]:
    """Validate one reviewed artifact against the fixed Task 13 artifact map."""

    coordinate = _validate_coordinate(value)
    if coordinate["artifact_kind"] in SEMANTIC_GATE_PINS:
        try:
            validate_semantic_gate_coordinate(coordinate)
        except TransportGateError as error:
            raise CampaignPackageError(
                "semantic transport gate identity is not exact"
            ) from error
    return coordinate


def _require_exact_stack_id(
    value: object,
    *,
    label: str,
    pattern: re.Pattern[str],
) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise CampaignPackageError(label + " stack identity is not exact")
    return value


def _require_bound_coordinate(
    value: object,
    *,
    kind: str,
    artifacts: Mapping[str, Mapping[str, object]],
    label: str,
) -> dict[str, object]:
    coordinate = validate_campaign_artifact_coordinate(value)
    if coordinate.get("artifact_kind") != kind:
        raise CampaignPackageError(label + " artifact kind drifted")
    if coordinate != artifacts.get(kind):
        raise CampaignPackageError(label + " artifact coordinate drifted")
    return coordinate


def _validate_runtime_archive_coordinate(
    value: object,
    *,
    key_kind: str,
    label: str,
) -> dict[str, object]:
    fields = {
        "bucket",
        "key",
        "version_id",
        "size_bytes",
        "file_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise CampaignPackageError(label + " archive coordinate drifted")
    file_sha = value.get("file_sha256")
    if (
        type(file_sha) is not str
        or _SHA256.fullmatch(file_sha) is None
        or value.get("bucket") != CAMPAIGN_BUCKET
        or value.get("key")
        != f"task13/artifacts/{key_kind}/{file_sha}.zip"
        or type(value.get("version_id")) is not str
        or not value["version_id"]
        or str(value["version_id"]).lower() in _MUTABLE_VERSION_WORDS
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] < 1
    ):
        raise CampaignPackageError(label + " archive coordinate drifted")
    return _copy_json(value)


def _validate_staged_journal_source(
    value: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    path = PurePosixPath(str(value.get("staged_journal_path", "")))
    expected_size = value.get("staged_journal_size_bytes")
    records = value.get("staged_journal_records")
    staged_request = value.get("staged_request")
    staged_request_identity = value.get(
        "staged_request_identity_sha256"
    )
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path.name in {"", ".", ".."}
        or type(expected_size) is not int
        or expected_size < 1
        or type(records) is not list
        or not records
        or any(type(row) is not dict for row in records)
        or type(staged_request) is not dict
        or set(staged_request)
        != {
            "schema_version",
            "record_type",
            "activation_id",
            "journal_path",
            "production_request",
        }
        or staged_request["schema_version"] != 1
        or staged_request["record_type"]
        != "glm52_task13_staged_deployment_request_v1"
        or staged_request["activation_id"] != value.get("activation_id")
        or staged_request["journal_path"] != str(path)
        or type(staged_request["production_request"]) is not dict
    ):
        raise CampaignPackageError(
            "staged deployment journal source drifted"
        )
    expected_request_identity = hashlib.sha256(
        canonical_json_bytes(staged_request)
    ).hexdigest()
    _require_sha256(
        staged_request_identity,
        "staged deployment request",
    )
    if staged_request_identity != expected_request_identity:
        raise CampaignPackageError(
            "staged deployment request identity drifted"
        )
    try:
        raw = b"".join(
            canonical_json_bytes(row) + b"\n" for row in records
        )
    except (TypeError, ValueError) as error:
        raise CampaignPackageError(
            "staged deployment journal source drifted"
        ) from error
    if (
        len(raw) != expected_size
        or hashlib.sha256(raw).hexdigest()
        != value.get("staged_journal_sha256")
    ):
        raise CampaignPackageError(
            "staged deployment journal source drifted"
        )
    previous: Optional[str] = None
    committed_steps: list[str] = []
    committed_evidence: dict[str, dict[str, object]] = {}
    request_identity: Optional[str] = None

    for sequence, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            raise CampaignPackageError(
                "staged deployment journal chain drifted"
            )
        try:
            row = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise CampaignPackageError(
                "staged deployment journal chain drifted"
            ) from error
        if (
            type(row) is not dict
            or set(row)
            != {
                "schema_version",
                "record_type",
                "sequence",
                "request_identity_sha256",
                "step",
                "state",
                "evidence",
                "previous_record_sha256",
                "record_sha256",
            }
            or line != canonical_json_bytes(row) + b"\n"
            or row["schema_version"] != 1
            or row["record_type"]
            != "glm52_task13_staged_deployment_journal_v1"
            or row["sequence"] != sequence
            or row["step"] not in _STAGED_DEPLOYMENT_STEPS
            or row["state"]
            not in {"PREPARED", "POSSIBLY_SENT", "COMMITTED"}
            or type(row["evidence"]) is not dict
            or row["previous_record_sha256"] != previous
        ):
            raise CampaignPackageError(
                "staged deployment journal chain drifted"
            )
        record_identity = row["record_sha256"]
        _require_sha256(
            record_identity,
            "staged deployment journal record",
        )
        unsigned = dict(row)
        del unsigned["record_sha256"]
        if record_identity != hashlib.sha256(
            canonical_json_bytes(unsigned)
        ).hexdigest():
            raise CampaignPackageError(
                "staged deployment journal chain drifted"
            )
        _require_sha256(
            row["request_identity_sha256"],
            "staged deployment journal request",
        )
        if request_identity is None:
            request_identity = row["request_identity_sha256"]
        elif row["request_identity_sha256"] != request_identity:
            raise CampaignPackageError(
                "staged deployment journal chain drifted"
            )
        if row["state"] == "COMMITTED":
            if (
                row["evidence"].get("worker_activation_attempts") != 0
                or row["evidence"].get("raw_ec2_launch_calls") != 0
            ):
                raise CampaignPackageError(
                    "staged deployment journal activated a worker"
                )
            committed_steps.append(str(row["step"]))
            committed_evidence[str(row["step"])] = _copy_json(
                row["evidence"]
            )
        previous = str(record_identity)
    if committed_steps != list(_STAGED_DEPLOYMENT_STEPS):
        raise CampaignPackageError(
            "staged deployment journal completion drifted"
        )
    if request_identity != staged_request_identity:
        raise CampaignPackageError(
            "staged deployment journal ancestry drifted"
        )
    return committed_evidence


def _coordinate_from_staged_evidence(
    evidence: Mapping[str, object],
    *,
    prefix: str,
    artifact_kind: str,
) -> dict[str, object]:
    coordinate = {
        "artifact_kind": artifact_kind,
        "bucket": evidence.get(prefix + "_bucket"),
        "key": evidence.get(prefix + "_key"),
        "version_id": evidence.get(prefix + "_version_id"),
        "file_sha256": evidence.get(prefix + "_file_sha256"),
        "body_sha256": evidence.get(prefix + "_body_sha256"),
    }
    return _validate_coordinate(coordinate)


def _validate_staged_request_ancestry(
    value: Mapping[str, object],
    *,
    committed: Mapping[str, Mapping[str, object]],
    artifacts: Mapping[str, Mapping[str, object]],
    retained_stack_id: str,
) -> dict[str, object]:
    staged_request = value["staged_request"]
    assert type(staged_request) is dict
    production = staged_request["production_request"]
    if (
        type(production) is not dict
        or set(production) != _STAGED_PRODUCTION_REQUEST_FIELDS
        or production.get("schema_version") != 1
        or production.get("record_type")
        != "glm52_task13_production_operations_v1"
        or production.get("activation_id") != value.get("activation_id")
        or production.get("retained_stack_id") != retained_stack_id
    ):
        raise CampaignPackageError(
            "staged production request ancestry drifted"
        )
    output = PurePosixPath(str(production.get("output_directory", "")))
    if (
        not output.is_absolute()
        or ".." in output.parts
        or output.name in {"", ".", ".."}
        or production.get("bootstrap_template")
        != artifacts.get("BOOTSTRAP_TEMPLATE")
        or production.get("retained_foundation_evidence")
        != value.get("retained_foundation")
        or production.get("fixed_artifacts")
        != [artifacts.get("RETAINED_FOUNDATION_TEMPLATE")]
    ):
        raise CampaignPackageError(
            "staged production request ancestry drifted"
        )
    for field in (
        "pre_support_runtime_inputs",
        "support_input_materialization_request",
        "support_price_card",
        "orphan_precreate",
    ):
        if type(production.get(field)) is not dict:
            raise CampaignPackageError(
                "staged production request ancestry drifted"
            )
    try:
        from .task13_migration_adapter import (
            Task13MigrationAdapterError,
            validate_stack_migration_seed_projection,
        )

        validate_stack_migration_seed_projection(
            production.get("stack_migration_seed")
        )
    except (TypeError, ValueError, Task13MigrationAdapterError) as error:
        raise CampaignPackageError(
            "staged migration seed ancestry drifted"
        ) from error
    try:
        from .task13_support_artifacts import (
            Task13SupportArtifactError,
            validate_support_artifact_publication,
        )

        support_publication = validate_support_artifact_publication(
            production.get("support_artifact_publication")
        )
    except Task13SupportArtifactError as error:
        raise CampaignPackageError(
            "staged support publication ancestry drifted"
        ) from error
    support_materialization = committed[
        "MATERIALIZE_FULL_SUPPORT_INPUTS"
    ]
    if (
        support_publication["support_lambda_archive"]
        != support_materialization["support_lambda_archive"]
        or support_publication["cryptography_layer_archive"]
        != support_materialization["cryptography_layer_archive"]
    ):
        raise CampaignPackageError(
            "staged support publication ancestry drifted"
        )
    return _copy_json(production)


def _validate_staged_outer_bindings(
    value: Mapping[str, object],
    *,
    committed: Mapping[str, Mapping[str, object]],
    artifacts: Mapping[str, Mapping[str, object]],
) -> None:
    migration_execution = committed["EXECUTE_STACK_MIGRATION"]
    expected_fence = {
        "stack_id": migration_execution["fence_stack_id"],
        "stack_status": "UPDATE_COMPLETE",
        "change_set_type": "IMPORT",
        "policy_logical_id": "H1gProductionFenceBucketPolicy",
        "template_coordinate": artifacts["FENCE_TEMPLATE"],
        "template_body_sha256": artifacts["FENCE_TEMPLATE"][
            "body_sha256"
        ],
        "readback_sha256": hashlib.sha256(
            canonical_json_bytes(migration_execution)
        ).hexdigest(),
    }
    pre = committed["UPDATE_RETAINED_PRE_SUPPORT"]
    expected_pre = {
        "stack_id": pre["stack_id"],
        "stack_status": pre["stack_status"],
        "change_set_type": pre["change_set_type"],
        "role_arn": pre["role_arn"],
        "template_coordinate": _coordinate_from_staged_evidence(
            pre,
            prefix="template",
            artifact_kind="RETAINED_PRE_SUPPORT_TEMPLATE",
        ),
        "retained_fragment_sha256": pre["source_fragment_sha256"],
        "readback_sha256": pre["readback_sha256"],
    }
    support_materialization = committed[
        "MATERIALIZE_FULL_SUPPORT_INPUTS"
    ]
    support_publication = committed["BUILD_PUBLISH_SUPPORT"]
    expected_support = {
        "stack_id": migration_execution["support_stack_id"],
        "stack_status": "UPDATE_COMPLETE",
        "change_set_type": "UPDATE",
        "worker_activation_enabled": False,
        "template_coordinate": _coordinate_from_staged_evidence(
            support_publication,
            prefix="support_template",
            artifact_kind="SUPPORT_TEMPLATE",
        ),
        "inputs_coordinate": _coordinate_from_staged_evidence(
            support_publication,
            prefix="support_inputs",
            artifact_kind="SUPPORT_INPUTS",
        ),
        "support_lambda_archive": support_materialization[
            "support_lambda_archive"
        ],
        "cryptography_layer_archive": support_materialization[
            "cryptography_layer_archive"
        ],
        "cryptography_layer_version_arn": support_materialization[
            "cryptography_layer_version_arn"
        ],
        "cryptography_layer_code_sha256": support_materialization[
            "cryptography_layer_code_sha256"
        ],
        "readback_sha256": hashlib.sha256(
            canonical_json_bytes(migration_execution)
        ).hexdigest(),
    }
    final = committed["UPDATE_RETAINED_FINAL"]
    expected_final = {
        "retained_stack_id": final["stack_id"],
        "support_stack_id": migration_execution["support_stack_id"],
        "stack_status": final["stack_status"],
        "change_set_type": final["change_set_type"],
        "role_arn": final["role_arn"],
        "template_coordinate": _coordinate_from_staged_evidence(
            final,
            prefix="template",
            artifact_kind="RETAINED_TEMPLATE",
        ),
        "retained_fragment_sha256": final["source_fragment_sha256"],
        "postpublication_authority_sha256": committed[
            "POSTPUBLICATION_AUTHORITY"
        ]["authority_sha256"],
        "readback_sha256": final["readback_sha256"],
    }
    exact_readback = committed["FINAL_EXACT_READBACK"]
    no_workers = committed["PROVE_NO_WORKER_ACTIVATION"]
    expected_readback = {
        "retained_stack_id": exact_readback["retained_stack_id"],
        "fence_stack_id": exact_readback["fence_stack_id"],
        "support_stack_id": exact_readback["support_stack_id"],
        "retained_stack_status": exact_readback[
            "retained_stack_status"
        ],
        "fence_stack_status": exact_readback["fence_stack_status"],
        "support_stack_status": exact_readback[
            "support_stack_status"
        ],
        "retained_template_sha256": exact_readback[
            "retained_template_sha256"
        ],
        "fence_template_sha256": exact_readback[
            "fence_template_sha256"
        ],
        "support_template_sha256": exact_readback[
            "support_template_sha256"
        ],
        "pending_change_sets": exact_readback["pending_change_sets"],
        "active_p5_instance_ids": no_workers["gpu_instance_ids"],
        "worker_activation_attempts": no_workers[
            "worker_activation_attempts"
        ],
        "raw_ec2_launch_calls": no_workers["raw_ec2_launch_calls"],
    }
    if (
        value.get("fence") != expected_fence
        or value.get("pre_support") != expected_pre
        or value.get("support_stack") != expected_support
        or value.get("postcreate_final") != expected_final
        or value.get("final_readback") != expected_readback
    ):
        raise CampaignPackageError(
            "staged infrastructure journal summary drifted"
        )


def validate_staged_infrastructure_evidence(
    value: object,
    *,
    activation_id: str,
    retained_stack_id: str,
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Authenticate the already-committed pre-package infrastructure proof."""

    if type(value) is not dict or set(value) != _STAGED_EVIDENCE_FIELDS:
        raise CampaignPackageError(
            "staged infrastructure evidence field set drifted"
        )
    expected_identity = {
        "schema_version": 1,
        "record_type": (
            "glm52_task13_staged_infrastructure_evidence_v1"
        ),
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": activation_id,
    }
    if any(value.get(field) != expected for field, expected in expected_identity.items()):
        raise CampaignPackageError(
            "staged infrastructure campaign identity drifted"
        )
    identity = value.get("canonical_identity_sha256")
    _require_sha256(identity, "staged infrastructure identity")
    unhashed = dict(value)
    del unhashed["canonical_identity_sha256"]
    if identity != hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest():
        raise CampaignPackageError(
            "staged infrastructure identity mismatch"
        )
    _require_sha256(
        value.get("staged_journal_sha256"),
        "staged deployment journal",
    )
    committed = _validate_staged_journal_source(value)
    if value.get("completed_steps") != list(_STAGED_DEPLOYMENT_STEPS):
        raise CampaignPackageError(
            "staged infrastructure completed steps drifted"
        )

    foundation = value.get("retained_foundation")
    if type(foundation) is not dict or set(foundation) != {
        "stack_id",
        "stack_status",
        "change_set_id",
        "template_coordinate",
        "template_body_sha256",
        "readback_sha256",
    }:
        raise CampaignPackageError("retained foundation evidence drifted")
    _require_exact_stack_id(
        foundation["stack_id"],
        label="retained foundation",
        pattern=_RETAINED_STACK_ID,
    )
    foundation_coordinate = _require_bound_coordinate(
        foundation["template_coordinate"],
        kind="RETAINED_FOUNDATION_TEMPLATE",
        artifacts=artifacts,
        label="retained foundation",
    )
    if (
        foundation["stack_id"] != retained_stack_id
        or foundation["stack_status"] != "UPDATE_COMPLETE"
        or type(foundation["change_set_id"]) is not str
        or not foundation["change_set_id"].startswith(
            "arn:aws:cloudformation:us-west-2:246813579024:changeSet/"
        )
        or foundation["template_body_sha256"]
        != foundation_coordinate["body_sha256"]
    ):
        raise CampaignPackageError("retained foundation evidence drifted")
    _require_sha256(
        foundation["readback_sha256"],
        "retained foundation readback",
    )
    _validate_staged_request_ancestry(
        value,
        committed=committed,
        artifacts=artifacts,
        retained_stack_id=retained_stack_id,
    )

    fence = value.get("fence")
    if type(fence) is not dict or set(fence) != {
        "stack_id",
        "stack_status",
        "change_set_type",
        "policy_logical_id",
        "template_coordinate",
        "template_body_sha256",
        "readback_sha256",
    }:
        raise CampaignPackageError("fence evidence drifted")
    _require_exact_stack_id(
        fence["stack_id"],
        label="fence",
        pattern=_FENCE_STACK_ID,
    )
    fence_coordinate = _require_bound_coordinate(
        fence["template_coordinate"],
        kind="FENCE_TEMPLATE",
        artifacts=artifacts,
        label="fence",
    )
    if (
        fence["stack_status"] != "UPDATE_COMPLETE"
        or fence["change_set_type"] != "IMPORT"
        or fence["policy_logical_id"] != "H1gProductionFenceBucketPolicy"
        or fence["template_body_sha256"] != fence_coordinate["body_sha256"]
    ):
        raise CampaignPackageError("fence evidence drifted")
    _require_sha256(fence["readback_sha256"], "fence readback")

    pre_support = value.get("pre_support")
    if type(pre_support) is not dict or set(pre_support) != {
        "stack_id",
        "stack_status",
        "change_set_type",
        "role_arn",
        "template_coordinate",
        "retained_fragment_sha256",
        "readback_sha256",
    }:
        raise CampaignPackageError("pre-support evidence drifted")
    _require_bound_coordinate(
        pre_support["template_coordinate"],
        kind="RETAINED_PRE_SUPPORT_TEMPLATE",
        artifacts=artifacts,
        label="pre-support",
    )
    if (
        pre_support["stack_id"] != retained_stack_id
        or pre_support["stack_status"] != "UPDATE_COMPLETE"
        or pre_support["change_set_type"] != "UPDATE"
        or pre_support["role_arn"] != DEPLOYMENT_ROLE_ARN
    ):
        raise CampaignPackageError("pre-support evidence drifted")
    _require_sha256(
        pre_support["retained_fragment_sha256"],
        "pre-support retained fragment",
    )
    _require_sha256(pre_support["readback_sha256"], "pre-support readback")

    support = value.get("support_stack")
    if type(support) is not dict or set(support) != {
        "stack_id",
        "stack_status",
        "change_set_type",
        "worker_activation_enabled",
        "template_coordinate",
        "inputs_coordinate",
        "support_lambda_archive",
        "cryptography_layer_archive",
        "cryptography_layer_version_arn",
        "cryptography_layer_code_sha256",
        "readback_sha256",
    }:
        raise CampaignPackageError("support stack evidence drifted")
    _require_exact_stack_id(
        support["stack_id"],
        label="support",
        pattern=_SUPPORT_STACK_ID,
    )
    _require_bound_coordinate(
        support["template_coordinate"],
        kind="SUPPORT_TEMPLATE",
        artifacts=artifacts,
        label="support stack",
    )
    _require_bound_coordinate(
        support["inputs_coordinate"],
        kind="SUPPORT_INPUTS",
        artifacts=artifacts,
        label="support inputs",
    )
    _validate_runtime_archive_coordinate(
        support["support_lambda_archive"],
        key_kind="support-lambda",
        label="support Lambda",
    )
    layer_archive = _validate_runtime_archive_coordinate(
        support["cryptography_layer_archive"],
        key_kind="cryptography-layer-python312-x86_64",
        label="cryptography layer",
    )
    layer_code_sha = support["cryptography_layer_code_sha256"]
    if (
        support["stack_status"] != "UPDATE_COMPLETE"
        or support["change_set_type"] != "UPDATE"
        or support["worker_activation_enabled"] is not False
        or type(support["cryptography_layer_version_arn"]) is not str
        or _CRYPTOGRAPHY_LAYER_VERSION_ARN.fullmatch(
            support["cryptography_layer_version_arn"]
        )
        is None
        or type(layer_code_sha) is not str
        or layer_code_sha
        != base64.b64encode(
            bytes.fromhex(str(layer_archive["file_sha256"]))
        ).decode("ascii")
    ):
        raise CampaignPackageError("support stack evidence drifted")
    _require_sha256(support["readback_sha256"], "support stack readback")

    final = value.get("postcreate_final")
    if type(final) is not dict or set(final) != {
        "retained_stack_id",
        "support_stack_id",
        "stack_status",
        "change_set_type",
        "role_arn",
        "template_coordinate",
        "retained_fragment_sha256",
        "postpublication_authority_sha256",
        "readback_sha256",
    }:
        raise CampaignPackageError("postcreate-final evidence drifted")
    _require_bound_coordinate(
        final["template_coordinate"],
        kind="RETAINED_TEMPLATE",
        artifacts=artifacts,
        label="postcreate-final",
    )
    if (
        final["retained_stack_id"] != retained_stack_id
        or final["support_stack_id"] != support["stack_id"]
        or final["stack_status"] != "UPDATE_COMPLETE"
        or final["change_set_type"] != "UPDATE"
        or final["role_arn"] != DEPLOYMENT_ROLE_ARN
    ):
        raise CampaignPackageError("postcreate-final evidence drifted")
    for field in (
        "retained_fragment_sha256",
        "postpublication_authority_sha256",
        "readback_sha256",
    ):
        _require_sha256(final[field], "postcreate-final " + field)

    fixed = value.get("fixed_artifacts")
    if not STAGED_INFRASTRUCTURE_ARTIFACT_KINDS.issubset(artifacts):
        raise CampaignPackageError(
            "staged fixed-artifact evidence drifted"
        )
    expected_fixed = [
        artifacts[kind]
        for kind in sorted(STAGED_INFRASTRUCTURE_ARTIFACT_KINDS)
    ]
    if fixed != expected_fixed:
        raise CampaignPackageError(
            "staged fixed-artifact evidence drifted"
        )
    fixed_identity = hashlib.sha256(
        canonical_json_bytes(expected_fixed)
    ).hexdigest()
    if value.get("fixed_artifacts_identity_sha256") != fixed_identity:
        raise CampaignPackageError(
            "staged fixed-artifact identity drifted"
        )

    readback = value.get("final_readback")
    if type(readback) is not dict or set(readback) != {
        "retained_stack_id",
        "fence_stack_id",
        "support_stack_id",
        "retained_stack_status",
        "fence_stack_status",
        "support_stack_status",
        "retained_template_sha256",
        "fence_template_sha256",
        "support_template_sha256",
        "pending_change_sets",
        "active_p5_instance_ids",
        "worker_activation_attempts",
        "raw_ec2_launch_calls",
    }:
        raise CampaignPackageError(
            "staged final readback field set drifted"
        )
    if (
        readback["retained_stack_id"] != retained_stack_id
        or readback["fence_stack_id"] != fence["stack_id"]
        or readback["support_stack_id"] != support["stack_id"]
        or readback["retained_stack_status"] != "UPDATE_COMPLETE"
        or readback["fence_stack_status"] != "UPDATE_COMPLETE"
        or readback["support_stack_status"] != "UPDATE_COMPLETE"
        or readback["retained_template_sha256"]
        != final["template_coordinate"]["body_sha256"]
        or readback["fence_template_sha256"]
        != fence["template_coordinate"]["body_sha256"]
        or readback["support_template_sha256"]
        != support["template_coordinate"]["body_sha256"]
        or type(readback["pending_change_sets"]) is not int
        or readback["pending_change_sets"] != 0
        or readback["active_p5_instance_ids"] != []
        or type(readback["worker_activation_attempts"]) is not int
        or readback["worker_activation_attempts"] != 0
        or type(readback["raw_ec2_launch_calls"]) is not int
        or readback["raw_ec2_launch_calls"] != 0
    ):
        raise CampaignPackageError(
            "staged infrastructure attempted worker activation"
        )
    _validate_staged_outer_bindings(
        value,
        committed=committed,
        artifacts=artifacts,
    )

    return _copy_json(value)


def _rebind_staged_fixed_artifacts(
    evidence: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rebound = _copy_json(evidence)
    fixed = sorted(
        (
            _copy_json(row)
            for row in artifacts
            if row["artifact_kind"]
            in STAGED_INFRASTRUCTURE_ARTIFACT_KINDS
        ),
        key=lambda row: str(row["artifact_kind"]),
    )
    if (
        rebound.get("fixed_artifacts") != fixed
        or rebound.get("fixed_artifacts_identity_sha256")
        != hashlib.sha256(canonical_json_bytes(fixed)).hexdigest()
    ):
        raise CampaignPackageError(
            "staged fixed-artifact evidence drifted"
        )
    return rebound


def derive_staged_infrastructure_successor_evidence(
    evidence: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bind unchanged staged truth to an exact successor artifact set."""

    return _rebind_staged_fixed_artifacts(evidence, artifacts)


def _validate_request(
    value: object,
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    str,
    str,
]:
    if type(value) is not dict or set(value) != _REQUEST_FIELDS:
        raise CampaignPackageError("campaign package request field set drifted")
    if value["schema_version"] != 1:
        raise CampaignPackageError("campaign package request schema drifted")
    if value["record_type"] != "glm52_task13_campaign_package_request_v1":
        raise CampaignPackageError("campaign package request record type drifted")
    expected = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
    }
    for field, wanted in expected.items():
        if value[field] != wanted:
            raise CampaignPackageError("%s is not exact" % field)
    activation_id = _require_text(value["activation_id"], "activation_id")
    if _ACTIVATION.fullmatch(activation_id) is None:
        raise CampaignPackageError("activation_id is invalid")
    _validate_task11_bindings(
        value["task11_request"],
        value["task11_boundary"],
        activation_id=activation_id,
    )
    collector_arn = _require_text(
        value["collector_version_arn"], "collector_version_arn"
    )
    collector_match = _COLLECTOR_ARN.fullmatch(collector_arn)
    if collector_match is None:
        raise CampaignPackageError("collector_version_arn is not a pinned version")

    retained_stack_id = _require_text(
        value["retained_stack_id"], "retained_stack_id"
    )
    if _RETAINED_STACK_ID.fullmatch(retained_stack_id) is None:
        raise CampaignPackageError("retained_stack_id is not exact")
    if value["fence_stack_name"] != "keep-glm52-h1g-fence":
        raise CampaignPackageError("fence_stack_name is not exact")
    if value["support_stack_name"] != "keep-glm52-h1g-support":
        raise CampaignPackageError("support_stack_name is not exact")
    for field, expected_fragment in (
        ("fence_change_set_name", "fence"),
        ("support_change_set_name", "support"),
    ):
        name = _require_text(value[field], field)
        if (
            _CHANGE_SET_NAME.fullmatch(name) is None
            or expected_fragment not in name
        ):
            raise CampaignPackageError("%s is not deterministic" % field)
    expected_roles = {
        "retained_role_arn": DEPLOYMENT_ROLE_ARN,
        "fence_role_arn": FENCE_ROLE_ARN,
        "support_role_arn": DEPLOYMENT_ROLE_ARN,
    }
    for field, expected in expected_roles.items():
        if value[field] != expected:
            raise CampaignPackageError(field + " is not exact")
    descriptor = _require_text(
        value["monitor_descriptor_path"], "monitor_descriptor_path"
    )
    descriptor_path = PurePosixPath(descriptor)
    if (
        not descriptor_path.is_absolute()
        or ".." in descriptor_path.parts
        or descriptor_path.name != "campaign-descriptor-v2.json"
    ):
        raise CampaignPackageError("monitor_descriptor_path is not exact")
    raw_artifacts = value["artifacts"]
    if type(raw_artifacts) is not list:
        raise CampaignPackageError("artifact coordinates must be a list")
    artifacts = [
        validate_campaign_artifact_coordinate(row) for row in raw_artifacts
    ]
    by_kind = {str(row["artifact_kind"]): row for row in artifacts}
    if (
        by_kind.get("REPOSITORY_ARCHIVE") is not None
        and by_kind["REPOSITORY_ARCHIVE"]["key"]
        != repository_archive_manifest_key(activation_id)
    ):
        raise CampaignPackageError(
            "repository archive activation lineage drifted"
        )
    for kind in _ACTIVATION_SCOPED_ARTIFACT_KINDS:
        coordinate = by_kind.get(kind)
        if (
            coordinate is not None
            and coordinate["key"]
            != activation_artifact_key(kind, activation_id)
        ):
            raise CampaignPackageError(
                kind + " activation lineage drifted"
            )
    production_kinds = set(_ARTIFACT_KEYS)
    prequalification_kinds = (
        production_kinds - PRODUCTION_ONLY_ARTIFACT_KINDS
    )
    artifact_kinds = frozenset(by_kind)
    if (
        len(artifacts) != len(by_kind)
        or artifact_kinds not in {
            frozenset(prequalification_kinds),
            frozenset(production_kinds),
        }
    ):
        raise CampaignPackageError("artifact kinds are not exact")
    try:
        for kind in SEMANTIC_GATE_PINS:
            if kind not in by_kind:
                raise KeyError(kind)
    except KeyError as error:
        raise CampaignPackageError(
            "semantic transport gate identity is not exact"
        ) from error
    phase = (
        "PRODUCTION"
        if artifact_kinds == production_kinds
        else "PREQUALIFICATION"
    )
    validate_staged_infrastructure_evidence(
        value["staged_infrastructure_evidence"],
        activation_id=activation_id,
        retained_stack_id=retained_stack_id,
        artifacts=by_kind,
    )
    return _copy_json(value), by_kind, collector_match.group(1), phase


def _validate_predecessor(
    *,
    phase: str,
    validated_request: Mapping[str, object],
    artifacts: Mapping[str, Mapping[str, object]],
    predecessor_package: object,
    predecessor_reviewed_artifacts: object,
    clean_rehearsal_evidence: object,
) -> Optional[dict[str, object]]:
    if phase == "PREQUALIFICATION":
        if (
            predecessor_package is not None
            or predecessor_reviewed_artifacts is not None
            or clean_rehearsal_evidence is not None
        ):
            raise CampaignPackageError(
                "prequalification package must not have a predecessor"
            )
        return None
    if (
        type(predecessor_package) is not dict
        or type(predecessor_reviewed_artifacts) is not list
    ):
        raise CampaignPackageError(
            "production package requires an exact predecessor package "
            "and reviewed artifacts"
        )
    try:
        predecessor_raw = canonical_campaign_package_bytes(
            predecessor_package
        )
    except (TypeError, ValueError) as error:
        raise CampaignPackageError(
            "predecessor package identity drifted"
        ) from error
    if (
        predecessor_package.get("package_phase") != "PREQUALIFICATION"
        or predecessor_package.get("predecessor_identity") is not None
        or predecessor_package.get("reviewed_artifacts")
        != predecessor_reviewed_artifacts
    ):
        raise CampaignPackageError(
            "predecessor package or reviewed artifacts are not exact"
        )
    predecessor_by_kind = {
        row.get("artifact_kind"): row
        for row in predecessor_reviewed_artifacts
        if type(row) is dict
    }
    expected_predecessor_kinds = (
        set(_ARTIFACT_KEYS) - PRODUCTION_ONLY_ARTIFACT_KINDS
    )
    if set(predecessor_by_kind) != expected_predecessor_kinds:
        raise CampaignPackageError(
            "predecessor reviewed artifacts are not exact"
        )
    expected_predecessor_request = _copy_json(validated_request)
    expected_predecessor_request["artifacts"] = [
        predecessor_by_kind[row["artifact_kind"]]
        for row in expected_predecessor_request["artifacts"]
        if row["artifact_kind"] not in PRODUCTION_ONLY_ARTIFACT_KINDS
    ]
    expected_predecessor_request["staged_infrastructure_evidence"] = (
        _rebind_staged_fixed_artifacts(
            expected_predecessor_request[
                "staged_infrastructure_evidence"
            ],
            expected_predecessor_request["artifacts"],
        )
    )
    expected_predecessor = build_campaign_package(
        expected_predecessor_request
    )
    if predecessor_package != expected_predecessor:
        raise CampaignPackageError(
            "production predecessor differs by more than the permitted "
            "successor coordinates"
        )
    for field in (
        "account_id",
        "region",
        "profile",
        "run_id",
        "activation_id",
        "task11_request",
        "task11_boundary",
    ):
        if predecessor_package.get(field) != validated_request.get(field):
            raise CampaignPackageError(
                "predecessor campaign identity drifted"
            )
    changed_predecessor_kinds = {
        kind
        for kind in expected_predecessor_kinds
        if artifacts[kind] != predecessor_by_kind[kind]
    }
    added_kinds = set(artifacts) - expected_predecessor_kinds
    if (
        changed_predecessor_kinds
        or added_kinds != PRODUCTION_ONLY_ARTIFACT_KINDS
    ):
        raise CampaignPackageError(
            "production successor must preserve every predecessor "
            "coordinate and add exactly four production coordinates"
        )
    try:
        from .task13_clean_rehearsal import (
            validate_clean_rehearsal_coordinate,
            validate_clean_rehearsal_evidence,
        )

        validated_clean_evidence = validate_clean_rehearsal_evidence(
            clean_rehearsal_evidence
        )
        validate_clean_rehearsal_coordinate(
            artifacts["CLEAN_REHEARSAL"],
            activation_id=str(validated_request["activation_id"]),
            evidence=validated_clean_evidence,
        )
    except ValueError as error:
        raise CampaignPackageError(
            "clean rehearsal successor proof drifted"
        ) from error
    evidence_predecessor = validated_clean_evidence[
        "predecessor_package"
    ]
    evidence_archive = validated_clean_evidence[
        "repository_archive"
    ]
    if (
        type(evidence_predecessor) is not dict
        or type(evidence_archive) is not dict
        or validated_clean_evidence["activation_id"]
        != validated_request["activation_id"]
        or evidence_predecessor["package_identity_sha256"]
        != predecessor_package["canonical_identity_sha256"]
        or evidence_predecessor["package_file_sha256"]
        != hashlib.sha256(predecessor_raw).hexdigest()
        or evidence_predecessor["reviewed_artifacts_identity_sha256"]
        != hashlib.sha256(
            canonical_json_bytes(predecessor_reviewed_artifacts)
        ).hexdigest()
        or evidence_archive["coordinate"]
        != predecessor_by_kind["REPOSITORY_ARCHIVE"]
        or artifacts["REPOSITORY_ARCHIVE"]
        != predecessor_by_kind["REPOSITORY_ARCHIVE"]
    ):
        raise CampaignPackageError(
            "clean rehearsal successor lineage drifted"
        )
    return {
        "package_identity_sha256": _require_sha256(
            predecessor_package.get("canonical_identity_sha256"),
            "predecessor package identity",
        ),
        "reviewed_artifacts_identity_sha256": hashlib.sha256(
            canonical_json_bytes(predecessor_reviewed_artifacts)
        ).hexdigest(),
        "clean_rehearsal_evidence": validated_clean_evidence,
    }


def _command(
    command_id: str,
    operation: str,
    argv: Sequence[str],
    *,
    mutates_aws: bool,
) -> dict[str, object]:
    if operation not in _COMMAND_OPERATION_ALLOWLIST:
        raise AssertionError("unallowlisted Task 13 command operation")
    if not argv or any(type(item) is not str or not item for item in argv):
        raise AssertionError("invalid Task 13 command argv")
    return {
        "command_id": command_id,
        "operation": operation,
        "account_id": ACCOUNT_ID,
        "profile": PROFILE,
        "region": REGION,
        "mutates_aws": mutates_aws,
        "execution_state": (
            "BLOCKED_REQUIRES_CONTROLLER"
            if mutates_aws
            else "READ_ONLY_INSPECTION"
        ),
        "argv": list(argv),
    }


def _build_execution_custody(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    body = {
        "schema_version": 1,
        "record_type": (
            "glm52_task13_reviewed_repository_execution_custody_v1"
        ),
        "execution_root_source": "REVIEWED_REPOSITORY_ARCHIVE",
        "repository_archive_manifest": _copy_json(
            artifacts["REPOSITORY_ARCHIVE"]
        ),
        "archive_payload_exact_version_required": True,
        "archive_payload_file_sha256_required": True,
        "safe_archive_extraction_required": True,
        "live_worktree_execution_allowed": False,
        "production_route": list(_PRODUCTION_ROUTE),
        "required_executable_paths": list(_REQUIRED_EXECUTABLE_PATHS),
        "executable_file_sha256_required": True,
        "repository_driver_manifests": {
            "h100-qualification": _copy_json(
                artifacts["H100_QUALIFICATION_INPUT"]
            ),
            "qualification-cache-seed": _copy_json(
                artifacts["QUALIFICATION_CACHE_SEED_INPUT"]
            ),
        },
    }
    return {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            canonical_json_bytes(body)
        ).hexdigest(),
    }


def validate_reviewed_repository_execution_custody(
    value: object,
    *,
    artifacts: Mapping[str, Mapping[str, object]],
    route: object,
) -> dict[str, object]:
    """Validate that campaign code executes only from the reviewed archive."""

    expected = _build_execution_custody(artifacts)
    if value != expected or route != list(_PRODUCTION_ROUTE):
        raise CampaignPackageError(
            "reviewed repository execution custody drifted"
        )
    return _copy_json(expected)


def _aws(
    service: str,
    operation: str,
    *arguments: str,
) -> list[str]:
    return [
        "aws",
        service,
        operation,
        *arguments,
        "--profile",
        PROFILE,
        "--region",
        REGION,
        "--no-cli-pager",
    ]


def _template_url(coordinate: Mapping[str, object]) -> str:
    return (
        "https://%s.s3.%s.amazonaws.com/%s?versionId=%s"
        % (
            coordinate["bucket"],
            REGION,
            quote(str(coordinate["key"]), safe="/"),
            quote(str(coordinate["version_id"]), safe=""),
        )
    )


def _change_set_commands(
    request: Mapping[str, object],
    artifacts: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    stack_specs = (
        (
            "fence",
            request["fence_stack_name"],
            request["fence_change_set_name"],
            artifacts["FENCE_TEMPLATE"],
            request["fence_role_arn"],
        ),
        (
            "support",
            request["support_stack_name"],
            request["support_change_set_name"],
            artifacts["SUPPORT_TEMPLATE"],
            request["support_role_arn"],
        ),
    )
    create: list[dict[str, object]] = []
    inspect: list[dict[str, object]] = []
    execute: list[dict[str, object]] = []
    for label, stack_id, change_set, template, role_arn in stack_specs:
        create.append(
            _command(
                "create-%s-disabled-change-set" % label,
                "cloudformation:CreateChangeSet",
                _aws(
                    "cloudformation",
                    "create-change-set",
                    "--stack-name",
                    str(stack_id),
                    "--change-set-name",
                    str(change_set),
                    "--change-set-type",
                    "UPDATE",
                    "--template-url",
                    _template_url(template),
                    "--role-arn",
                    str(role_arn),
                    "--capabilities",
                    "CAPABILITY_NAMED_IAM",
                    "--description",
                    "Task13 disabled no-execute %s %s"
                    % (label, template["body_sha256"]),
                    "--tags",
                    *[
                        "Key=%s,Value=%s" % (key, value)
                        for key, value in STACK_TAGS
                    ],
                ),
                mutates_aws=True,
            )
        )
        inspect.extend(
            [
                _command(
                    "describe-%s-disabled-change-set" % label,
                    "cloudformation:DescribeChangeSet",
                    _aws(
                        "cloudformation",
                        "describe-change-set",
                        "--stack-name",
                        str(stack_id),
                        "--change-set-name",
                        str(change_set),
                        "--output",
                        "json",
                    ),
                    mutates_aws=False,
                ),
                _command(
                    "inspect-%s-change-set-template" % label,
                    "cloudformation:GetTemplate",
                    _aws(
                        "cloudformation",
                        "get-template",
                        "--stack-name",
                        str(stack_id),
                        "--change-set-name",
                        str(change_set),
                        "--template-stage",
                        "Processed",
                        "--output",
                        "json",
                    ),
                    mutates_aws=False,
                ),
            ]
        )
        execute.append(
            _command(
                "execute-%s-disabled-change-set" % label,
                "cloudformation:ExecuteChangeSet",
                _aws(
                    "cloudformation",
                    "execute-change-set",
                    "--stack-name",
                    str(stack_id),
                    "--change-set-name",
                    str(change_set),
                ),
                mutates_aws=True,
            )
        )
    return create, inspect, {
        "authorized": False,
        "required_gates": [
            "TASK11_REVIEW_APPROVED",
            "TASK12_REVIEW_APPROVED",
            "DISABLED_STACK_CHANGE_SETS_REVIEWED",
        ],
        "commands": execute,
    }


def _guard_commands(request: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        _command(
            "read-caller-identity",
            "sts:GetCallerIdentity",
            _aws("sts", "get-caller-identity", "--output", "json"),
            mutates_aws=False,
        ),
        {
            **_command(
                "assert-caller-account",
                "sts:GetCallerIdentity",
                _aws(
                    "sts",
                    "get-caller-identity",
                    "--query",
                    "Account",
                    "--output",
                    "text",
                ),
                mutates_aws=False,
            ),
            "expected_stdout": ACCOUNT_ID,
        },
        _command(
            "read-credential-expiry",
            "sts:ExportCredentials",
            [
                "aws",
                "configure",
                "export-credentials",
                "--profile",
                PROFILE,
                "--format",
                "process",
            ],
            mutates_aws=False,
        ),
        *[
            _command(
                "read-%s-cloudformation-role" % label,
                "iam:GetRole",
                _aws(
                    "iam",
                    "get-role",
                    "--role-name",
                    str(role_arn).rsplit("/", 1)[1],
                    "--output",
                    "json",
                ),
                mutates_aws=False,
            )
            for label, role_arn in (
                ("deployment", request["retained_role_arn"]),
                ("fence", request["fence_role_arn"]),
            )
        ],
    ]


def _post_deploy_commands(request: Mapping[str, object]) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = []
    stack_coordinates = (
        ("retained", request["retained_stack_id"]),
        ("fence", request["fence_stack_name"]),
        ("support", request["support_stack_name"]),
    )
    for label, stack_id in stack_coordinates:
        commands.extend(
            [
                _command(
                    "readback-%s-stack" % label,
                    "cloudformation:DescribeStacks",
                    _aws(
                        "cloudformation",
                        "describe-stacks",
                        "--stack-name",
                        str(stack_id),
                        "--output",
                        "json",
                    ),
                    mutates_aws=False,
                ),
                _command(
                    "inventory-%s-resources" % label,
                    "cloudformation:ListStackResources",
                    _aws(
                        "cloudformation",
                        "list-stack-resources",
                        "--stack-name",
                        str(stack_id),
                        "--output",
                        "json",
                    ),
                    mutates_aws=False,
                ),
                _command(
                    "readback-%s-template" % label,
                    "cloudformation:GetTemplate",
                    _aws(
                        "cloudformation",
                        "get-template",
                        "--stack-name",
                        str(stack_id),
                        "--template-stage",
                        "Original",
                        "--output",
                        "json",
                    ),
                    mutates_aws=False,
                ),
            ]
        )
    return commands


def _cleanup_commands(request: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        _command(
            "delete-unexecuted-fence-change-set",
            "cloudformation:DeleteChangeSet",
            _aws(
                "cloudformation",
                "delete-change-set",
                "--stack-name",
                str(request["fence_stack_name"]),
                "--change-set-name",
                str(request["fence_change_set_name"]),
            ),
            mutates_aws=True,
        ),
        _command(
            "delete-unexecuted-support-change-set",
            "cloudformation:DeleteChangeSet",
            _aws(
                "cloudformation",
                "delete-change-set",
                "--stack-name",
                str(request["support_stack_name"]),
                "--change-set-name",
                str(request["support_change_set_name"]),
            ),
            mutates_aws=True,
        ),
        _command(
            "reconcile-retained-stack",
            "cloudformation:DescribeStacks",
            _aws(
                "cloudformation",
                "describe-stacks",
                "--stack-name",
                str(request["retained_stack_id"]),
                "--output",
                "json",
            ),
            mutates_aws=False,
        ),
    ]


def _rehearsal(
    request: Mapping[str, object],
    qualifier: str,
) -> dict[str, object]:
    scenario_by_index = {
        1: "THROTTLING",
        2: "PAGINATION",
        3: "NETWORK_AMBIGUITY",
    }
    invokes = []
    for index in range(1, 21):
        invokes.append(
            {
                "function_version_arn": request["collector_version_arn"],
                "qualifier": qualifier,
                "event": {
                    "schema_version": 1,
                    "record_type": "glm52_task11_collect_rehearsal_v1",
                    "activation_id": request["activation_id"],
                    "measurement_id": "measurement-%02d" % index,
                    "scenario": scenario_by_index.get(index, "NONE"),
                    "task11_request": _copy_json(request["task11_request"]),
                    "task11_boundary": _copy_json(request["task11_boundary"]),
                },
            }
        )
    return {
        "execution_state": "BLOCKED_REQUIRES_DISABLED_DEPLOYMENT",
        "invoke_type": "RequestResponse",
        "collect_invocations": invokes,
        "batches": [
            {
                "batch_id": "cold-start-01",
                "concurrency": 5,
                "measurement_ids": [
                    "measurement-%02d" % index for index in range(1, 6)
                ],
                "required_distinct_cold_environments": 5,
            },
            {
                "batch_id": "remaining-02",
                "concurrency": 1,
                "measurement_ids": [
                    "measurement-%02d" % index for index in range(6, 21)
                ],
                "required_distinct_cold_environments": 0,
            },
        ],
        "finalize_invocation": {
            "function_version_arn": request["collector_version_arn"],
            "qualifier": qualifier,
            "event": {
                "schema_version": 1,
                "record_type": "glm52_task11_finalize_rehearsal_gate_v1",
                "activation_id": request["activation_id"],
            },
        },
        "invoke_result_contract": {
            "outer": {
                "status_code": 200,
                "function_error_absent": True,
                "executed_version": qualifier,
            },
            "collect_record_type": "glm52_task11_collect_rehearsal_result_v1",
            "finalize_record_type": (
                "glm52_task11_finalize_rehearsal_gate_result_v1"
            ),
            "collector_function_version_arn": request[
                "collector_version_arn"
            ],
            "self_hash_required": True,
        },
        "immutable_inspection_contract": {
            "source": "VALIDATED_INVOKE_PAYLOAD_ONLY",
            "bucket": (
                "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
            ),
            "required_coordinate_fields": [
                "key",
                "version_id",
                "file_sha256",
                "checksum_sha256_base64",
            ],
            "required_operations": [
                "s3:HeadObject exact VersionId",
                "s3:GetObject exact VersionId",
            ],
            "current_version_fallback": False,
        },
    }


def _negative_iam_probes(
    request: Mapping[str, object],
) -> list[dict[str, object]]:
    principal = (
        "arn:aws:iam::246813579024:role/"
        "keep-glm52-h1g-disabled-rehearsal"
    )
    return [
        {
            "probe_id": "deny-unqualified-collector-invoke",
            "principal_role_arn": principal,
            "operation": "lambda:InvokeFunction",
            "resource": str(request["collector_version_arn"]).rsplit(":", 1)[0],
            "expected_error_code": "AccessDenied",
            "must_not_retry": True,
        },
        {
            "probe_id": "deny-production-s3-write",
            "principal_role_arn": principal,
            "operation": "s3:PutObject",
            "resource": (
                "arn:aws:s3:::keep-glm52-models-"
                "246813579024-us-west-2/campaigns/"
                + RUN_ID
                + "/production/*"
            ),
            "expected_error_code": "AccessDenied",
            "must_not_retry": True,
        },
        {
            "probe_id": "deny-p5-launch",
            "principal_role_arn": principal,
            "operation": "ec2:RunInstances",
            "resource": "arn:aws:ec2:us-west-2:246813579024:instance/*",
            "expected_error_code": "AccessDenied",
            "must_not_retry": True,
        },
        {
            "probe_id": "deny-pass-production-role",
            "principal_role_arn": principal,
            "operation": "iam:PassRole",
            "resource": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-production-worker"
            ),
            "expected_error_code": "AccessDenied",
            "must_not_retry": True,
        },
    ]


def build_campaign_package(
    request: object,
    *,
    predecessor_package: object = None,
    predecessor_reviewed_artifacts: object = None,
    clean_rehearsal_evidence: object = None,
) -> dict[str, object]:
    """Build one deterministic, disabled, no-execute Task 13 package."""

    validated, artifacts, qualifier, phase = _validate_request(request)
    predecessor_identity = _validate_predecessor(
        phase=phase,
        validated_request=validated,
        artifacts=artifacts,
        predecessor_package=predecessor_package,
        predecessor_reviewed_artifacts=predecessor_reviewed_artifacts,
        clean_rehearsal_evidence=clean_rehearsal_evidence,
    )
    cleanup = _cleanup_commands(validated)
    artifact_rows = [
        artifacts[kind] for kind in sorted(artifacts)
    ]
    staged_evidence = validated["staged_infrastructure_evidence"]
    assert type(staged_evidence) is dict
    final_readback = staged_evidence["final_readback"]
    assert type(final_readback) is dict
    production_route = list(_PRODUCTION_ROUTE)
    execution_custody = _build_execution_custody(artifacts)
    package: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task13_campaign_package_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": validated["activation_id"],
        "package_phase": phase,
        "predecessor_identity": predecessor_identity,
        "predecessor_package": (
            None
            if phase == "PREQUALIFICATION"
            else _copy_json(predecessor_package)
        ),
        "task11_request": validated["task11_request"],
        "task11_boundary": validated["task11_boundary"],
        "execution_authority": {
            "mode": "STAGED_INFRASTRUCTURE_ADOPTED_NO_WORKER",
            "live_aws_executed": True,
            "change_set_execution_authorized": False,
            "change_set_execution_requires_scoped_authority": False,
            "qualification_cache_seed_authorized": False,
            "h100_qualification_authorized": False,
            "production_launch_authorized": False,
        },
        "reviewed_artifacts": artifact_rows,
        "semantic_transport_gates": _copy_json(SEMANTIC_GATE_PINS),
        "required_gates": list(_REQUIRED_GATES),
        "command_operation_allowlist": list(_COMMAND_OPERATION_ALLOWLIST),
        "disabled_deployment": {
            "state": "ADOPTED_DISABLED",
            "staged_infrastructure_evidence": staged_evidence,
            "approved_support_envelope": {
                "support_work_stop_hours": 68,
                "delete_request_deadline_hours": 71,
                "absence_expected_hours": 72,
                "support_host_count": 1,
                "host_root_volume_count": 1,
                "support_data_volume_count": 1,
                "support_data_volume_gib": 50,
                "support_nat_count": 1,
                "support_eip_count": 1,
                "interface_endpoint_eni_count": 2,
                "secret_count": 8,
                "secret_versions_per_secret": 1,
                "support_workflow_count": 1,
                "workflow_history_events_max": 12000,
                "accepted_support_lambda_invocations": 10000,
                "lambda_reserved_concurrency_per_function": 1,
                "transient_log_retention_days": 14,
                "application_log_ingestion_mib_max": 512,
                "forensic_snapshot_count": 1,
                "forensic_snapshot_retention_days": 7,
                "nat_processed_gib_max": 10,
            },
            "retained_template": artifacts["RETAINED_TEMPLATE"],
            "fence_template": artifacts["FENCE_TEMPLATE"],
            "support_template": artifacts["SUPPORT_TEMPLATE"],
            "retained_stack_id": validated["retained_stack_id"],
            "fence_stack_name": validated["fence_stack_name"],
            "support_stack_name": validated["support_stack_name"],
            "fence_stack_id": final_readback["fence_stack_id"],
            "support_stack_id": final_readback["support_stack_id"],
            "retained_role_arn": validated["retained_role_arn"],
            "fence_role_arn": validated["fence_role_arn"],
            "support_role_arn": validated["support_role_arn"],
            "tags": [list(tag) for tag in STACK_TAGS],
            "fence_parameters": [],
            "support_parameters": [],
            "adoption_readback_contract": {
                "operation_kind": "staged-infrastructure-adoption",
                "read_only": True,
                "exact_evidence_required": True,
                "staged_request_identity_sha256": staged_evidence[
                    "staged_request_identity_sha256"
                ],
                "staged_journal_sha256": staged_evidence[
                    "staged_journal_sha256"
                ],
                "fixed_artifacts_identity_sha256": staged_evidence[
                    "fixed_artifacts_identity_sha256"
                ],
                "worker_activation_allowed": False,
            },
            "resource_inventory": {
                "stacks": [
                    "retained",
                    "fence",
                    "support",
                ],
                "support_resources": [
                    "alerts",
                    "dead_letter_queues",
                    "log_groups",
                    "schedules",
                    "workflows",
                    "network_boundaries",
                    "host_roles",
                ],
                "expected_active_p5_instances": 0,
                "cloudformation_cardinality": {
                    "AWS::EC2::Subnet": 3,
                    "AWS::EC2::RouteTable": 3,
                    "AWS::EC2::SubnetRouteTableAssociation": 3,
                    "AWS::EC2::Route": 1,
                    "AWS::EC2::NatGateway": 1,
                    "AWS::EC2::EIP": 1,
                    "AWS::EC2::VPCEndpoint": 3,
                    "AWS::EC2::Instance": 1,
                    "AWS::EC2::Volume": 1,
                    "AWS::SecretsManager::Secret": 8,
                    "AWS::S3::Bucket": 1,
                    "AWS::Lambda::Function": 1,
                    "AWS::Lambda::Version": 1,
                    "AWS::StepFunctions::StateMachine": 1,
                    "AWS::StepFunctions::StateMachineVersion": 1,
                },
                "task10_production_workflow": {
                    "resource_count": 14,
                    "state_machine_logical_id": (
                        "Task10ProductionStateMachine"
                    ),
                    "state_machine_version_logical_id": (
                        "Task10ProductionStateMachineVersion"
                    ),
                    "state_machine_name": (
                        "keep-glm52-h1g-production"
                    ),
                    "workflow_role_logical_id": (
                        "Task10ProductionWorkflowRole"
                    ),
                    "reconciliation_role_logical_id": (
                        "Task10ProductionReconciliationRole"
                    ),
                    "reconciliation_log_group_logical_id": (
                        "Task10ProductionReconciliationLogGroup"
                    ),
                    "reconciliation_function_logical_id": (
                        "Task10ProductionReconciliationFunction"
                    ),
                    "reconciliation_function_version_logical_id": (
                        "Task10ProductionReconciliationFunctionVersion"
                    ),
                    "reconciliation_function_name": (
                        "keep-glm52-h1g-task10-"
                        "capacity-reconciliation"
                    ),
                    "invoke_permission_logical_id": (
                        "Task10ProductionReconciliationInvokePermission"
                    ),
                    "error_alarm_logical_id": (
                        "Task10ProductionReconciliationErrorAlarm"
                    ),
                    "liability_watcher_role_logical_id": (
                        "Task9LiabilityWatcherRole"
                    ),
                    "liability_watcher_log_group_logical_id": (
                        "Task9LiabilityWatcherLogGroup"
                    ),
                    "liability_watcher_function_logical_id": (
                        "Task9LiabilityWatcherFunction"
                    ),
                    "liability_watcher_function_version_logical_id": (
                        "Task9LiabilityWatcherFunctionVersion"
                    ),
                    "liability_watcher_error_alarm_logical_id": (
                        "Task9LiabilityWatcherErrorAlarm"
                    ),
                    "sole_sender_state": (
                        "RunInternalSixAzSoleSender"
                    ),
                    "terminal_writer_state": (
                        "RunInternalSixAzSoleSender"
                    ),
                    "maximum_ec2_calls": 6,
                    "ordered_availability_zones": [
                        "us-west-2%s" % letter for letter in "abcdef"
                    ],
                },
            },
            "cost_inventory": {
                "production_gpu_spend_usd": "0.00",
                "production_gpu_seconds": 0,
                "change_sets_only_before_execution": True,
                "live_price_refresh_required_before_controller_execution": True,
            },
            "readback_checklist": [
                "ALERTS_OK",
                "DLQS_EMPTY",
                "LOG_RETENTION_EXACT",
                "SCHEDULES_DISABLED",
                "WORKFLOWS_DISABLED",
                "NETWORK_ISOLATED",
                "HOST_ACCESS_DENIED",
                "ACTIVE_P5_COUNT_ZERO",
            ],
        },
        "rehearsal": _rehearsal(validated, qualifier),
        "negative_iam_probes": _negative_iam_probes(validated),
        "qualification_cache_seed": {
            "authorized": False,
            "activation_id": validated["activation_id"],
            "execution_state": "BLOCKED_REQUIRES_REHEARSAL_GATE",
            "driver_record_type": (
                "glm52_task13_repository_driver_v2"
            ),
            "driver_operation_kind": "qualification-cache-seed",
            "input": artifacts["QUALIFICATION_CACHE_SEED_INPUT"],
            "execution_custody": _copy_json(execution_custody),
            "instance_type": "p5.48xlarge",
            "capacity_type": "ON_DEMAND",
            "max_active_instances": 1,
            "maximum_gpu_hours": 6,
            "required_teacher_rows": 1,
            "ready_marker": "QUALIFICATION_CACHE_SEED_READY.json",
            "teacher_ready_marker": "TEACHER_CACHE_READY.json",
            "teardown_required": True,
        },
        "h100_qualification": {
            "authorized": False,
            "activation_id": validated["activation_id"],
            "execution_state": "BLOCKED_REQUIRES_REHEARSAL_GATE",
            "driver_record_type": (
                "glm52_task13_repository_driver_v2"
            ),
            "driver_operation_kind": "h100-qualification",
            "input": artifacts["H100_QUALIFICATION_INPUT"],
            "execution_custody": _copy_json(execution_custody),
            "instance_type": "p5.48xlarge",
            "capacity_type": "ON_DEMAND",
            "max_active_instances": 1,
            "required_distinct_instance_ids": 2,
            "required_training_steps": 2,
            "cross_node_resume_required": True,
            "peak_memory_limit_gib_exclusive": 70,
            "production_effects_required": 0,
            "source_ready_marker": "SOURCE_NODE_READY.json",
            "termination_requested_marker": (
                "QUALIFICATION_TERMINATION_REQUESTED.json"
            ),
            "resume_ready_marker": "H100_RESUME_READY.json",
            "source_termination_required": True,
            "replacement_instance_required": True,
            "required_gates": _REQUIRED_GATES[:6],
        },
        "production_retry_plan": {
            "authorized": False,
            "execution_state": "BLOCKED_UNTIL_ALL_GATES",
            "route": production_route,
            "execution_custody": execution_custody,
            "instance_type": "p5.48xlarge",
            "capacity_type": "ON_DEMAND",
            "max_active_instances": 1,
            "availability_zones": [
                "us-west-2%s" % letter for letter in "abcdef"
            ],
            "attempts": [
                {
                    "attempt": index,
                    "availability_zone": "us-west-2%s" % letter,
                    "on_capacity_failure": (
                        "ROTATE_NEXT_AZ"
                        if letter != "f"
                        else "STOP_CAPACITY_EXHAUSTED"
                    ),
                }
                for index, letter in enumerate("abcdef", 1)
            ],
            "required_gates": list(_REQUIRED_GATES),
            "pre_attempt_invariants": [
                "ACTIVE_P5_COUNT_ZERO",
                "NO_PENDING_SAME_TOKEN_ATTEMPT",
                "EXACT_REVIEWED_ARTIFACT_COORDINATES",
                "GUARDED_SUBMITTER_ONLY",
            ],
            "stop_on_first_success": True,
            "same_token_multiplicity_is_liability": True,
            "automatic_spot_fallback": False,
            "workflow_reconciliation_contract": {
                "writer_owner": (
                    "TASK10_VERSIONED_PRODUCTION_WORKFLOW"
                ),
                "writer_handler": (
                    "aws/glm52-gpu/lambda/"
                    "task10_sole_sender_handler.py"
                ),
                "state_machine_logical_id": (
                    "Task10ProductionStateMachine"
                ),
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
                "terminal_writer_state": (
                    "RunInternalSixAzSoleSender"
                ),
                "workflow_invocation_count": 1,
                "bucket": (
                    "keep-glm52-models-"
                    "246813579024-us-west-2"
                ),
                "key": (
                    "campaigns/glm52-sky-20260724/"
                    "submissions/production/generations/"
                    "00000001/workflow/LAUNCH_OUTCOME.json"
                ),
                "schema_version": 1,
                "record_type": (
                    "glm52_task10_capacity_reconciliation_v1"
                ),
                "classifications": [
                    "RUNNING",
                    "WORKER_ALLOCATED",
                    "CAPACITY_EXHAUSTED",
                    "FAILED",
                ],
                "capacity_outcome_fields": [
                    "attempt",
                    "availability_zone",
                    "outcome",
                ],
                "exact_version_id_required": True,
                "canonical_self_hash_required": True,
            },
            "immutable_inputs": [
                artifacts[kind]
                for kind in (
                    _PRODUCTION_RETRY_IMMUTABLE_INPUT_KINDS
                    if phase == "PRODUCTION"
                    else _BASE_RETRY_IMMUTABLE_INPUT_KINDS
                )
            ],
            "production_authority_contract": {
                "minimum_remaining_gpu_seconds_exclusive": 3600,
                "action_kind": "PRODUCTION_SUBMISSION",
                "action_count": 1,
                "required_terminal_markers": [
                    "CAMPAIGN_DRAINED.json",
                    "TERMINAL_VERIFIED.json",
                ],
                "monitor_route": (
                    "aws/glm52-gpu/scripts/"
                    "sky_campaign_break_glass.sh status"
                ),
            },
        },
        "monitor_contract": {
            "descriptor_path": validated["monitor_descriptor_path"],
            "environment": {
                "AWS_PROFILE": PROFILE,
                "CAMPAIGN_DESCRIPTOR": validated[
                    "monitor_descriptor_path"
                ],
                "SKY_BIN": (
                    "/Users/jack.mazac/.local/share/keep/"
                    "skypilot-0.13.0/bin/sky"
                ),
                "SKYPILOT_CONFIG": (
                    "/Users/jack.mazac/.local/share/keep/"
                    "skypilot-0.13.0/server-config.yaml"
                ),
            },
            "argv": [
                "aws/glm52-gpu/scripts/sky_campaign_break_glass.sh",
                "status",
            ],
            "read_only": True,
            "account_id": ACCOUNT_ID,
            "profile": PROFILE,
            "region": REGION,
        },
        "cleanup_and_reconciliation": {
            "evidence_retention_required": True,
            "stack_deletion_forbidden": True,
            "s3_evidence_deletion_forbidden": True,
            "commands": cleanup,
        },
    }
    package["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(package)
    ).hexdigest()
    return package


def canonical_campaign_package_bytes(package: object) -> bytes:
    """Return canonical JSON plus LF after validating the package self hash."""

    if type(package) is not dict:
        raise CampaignPackageError("campaign package must be an object")
    phase = package.get("package_phase")
    predecessor = package.get("predecessor_identity")
    embedded_predecessor = package.get("predecessor_package")
    artifacts = package.get("reviewed_artifacts")
    if type(artifacts) is not list:
        raise CampaignPackageError("campaign reviewed artifacts are missing")
    validated_artifacts = [
        validate_campaign_artifact_coordinate(row) for row in artifacts
    ]
    artifacts_by_kind = {
        str(row["artifact_kind"]): row for row in validated_artifacts
    }
    activation_id = _require_text(
        package.get("activation_id"),
        "activation_id",
    )
    if _ACTIVATION.fullmatch(activation_id) is None:
        raise CampaignPackageError("activation_id is invalid")
    if (
        artifacts_by_kind.get("REPOSITORY_ARCHIVE") is not None
        and artifacts_by_kind["REPOSITORY_ARCHIVE"]["key"]
        != repository_archive_manifest_key(activation_id)
    ):
        raise CampaignPackageError(
            "repository archive activation lineage drifted"
        )
    for kind in _ACTIVATION_SCOPED_ARTIFACT_KINDS:
        coordinate = artifacts_by_kind.get(kind)
        if (
            coordinate is not None
            and coordinate["key"]
            != activation_artifact_key(kind, activation_id)
        ):
            raise CampaignPackageError(
                kind + " activation lineage drifted"
            )
    kinds = {row["artifact_kind"] for row in validated_artifacts}
    if len(kinds) != len(artifacts):
        raise CampaignPackageError("campaign reviewed artifacts drifted")
    if phase == "PREQUALIFICATION":
        if (
            predecessor is not None
            or embedded_predecessor is not None
            or kinds
            != set(_ARTIFACT_KEYS) - PRODUCTION_ONLY_ARTIFACT_KINDS
        ):
            raise CampaignPackageError(
                "prequalification package lineage drifted"
            )
    elif phase == "PRODUCTION":
        required_predecessor_fields = {
            "package_identity_sha256",
            "reviewed_artifacts_identity_sha256",
            "clean_rehearsal_evidence",
        }
        if (
            type(predecessor) is not dict
            or set(predecessor) != required_predecessor_fields
            or kinds != set(_ARTIFACT_KEYS)
        ):
            raise CampaignPackageError("production package lineage drifted")
        _require_sha256(
            predecessor["package_identity_sha256"],
            "predecessor package identity",
        )
        _require_sha256(
            predecessor["reviewed_artifacts_identity_sha256"],
            "predecessor reviewed artifacts identity",
        )
        if type(embedded_predecessor) is not dict:
            raise CampaignPackageError(
                "production predecessor package is missing"
            )
        embedded_predecessor_raw = canonical_campaign_package_bytes(
            embedded_predecessor
        )
        predecessor_artifacts = embedded_predecessor.get(
            "reviewed_artifacts"
        )
        if type(predecessor_artifacts) is not list:
            raise CampaignPackageError(
                "production predecessor artifacts are missing"
            )
        validated_predecessor_artifacts = [
            validate_campaign_artifact_coordinate(row)
            for row in predecessor_artifacts
        ]
        predecessor_by_kind = {
            str(row["artifact_kind"]): row
            for row in validated_predecessor_artifacts
        }
        expected_predecessor_kinds = (
            set(_ARTIFACT_KEYS) - PRODUCTION_ONLY_ARTIFACT_KINDS
        )
        if (
            len(predecessor_by_kind)
            != len(validated_predecessor_artifacts)
            or set(predecessor_by_kind) != expected_predecessor_kinds
        ):
            raise CampaignPackageError(
                "production predecessor artifacts drifted"
            )
        changed_predecessor_kinds = {
            kind
            for kind in expected_predecessor_kinds
            if artifacts_by_kind[kind] != predecessor_by_kind[kind]
        }
        added_kinds = set(artifacts_by_kind) - expected_predecessor_kinds
        if (
            changed_predecessor_kinds
            or added_kinds != PRODUCTION_ONLY_ARTIFACT_KINDS
        ):
            raise CampaignPackageError(
                "production successor must preserve every predecessor "
                "coordinate and add exactly four production coordinates"
            )
        evidence = predecessor["clean_rehearsal_evidence"]
        try:
            from .task13_clean_rehearsal import (
                validate_clean_rehearsal_coordinate,
                validate_clean_rehearsal_evidence,
            )

            validated_evidence = validate_clean_rehearsal_evidence(
                evidence
            )
            validate_clean_rehearsal_coordinate(
                artifacts_by_kind["CLEAN_REHEARSAL"],
                activation_id=str(package.get("activation_id")),
                evidence=validated_evidence,
            )
        except ValueError as error:
            raise CampaignPackageError(
                "production clean rehearsal proof drifted"
            ) from error
        evidence_predecessor = validated_evidence[
            "predecessor_package"
        ]
        evidence_archive = validated_evidence["repository_archive"]
        if (
            type(evidence_predecessor) is not dict
            or type(evidence_archive) is not dict
            or validated_evidence["activation_id"]
            != package.get("activation_id")
            or evidence_predecessor["package_identity_sha256"]
            != embedded_predecessor.get("canonical_identity_sha256")
            or evidence_predecessor["package_file_sha256"]
            != hashlib.sha256(
                embedded_predecessor_raw
            ).hexdigest()
            or evidence_predecessor[
                "reviewed_artifacts_identity_sha256"
            ]
            != hashlib.sha256(
                canonical_json_bytes(predecessor_artifacts)
            ).hexdigest()
            or evidence_archive["coordinate"]
            != predecessor_by_kind["REPOSITORY_ARCHIVE"]
            or artifacts_by_kind["REPOSITORY_ARCHIVE"]
            != predecessor_by_kind["REPOSITORY_ARCHIVE"]
        ):
            raise CampaignPackageError(
                "production clean rehearsal lineage drifted"
            )
        if (
            embedded_predecessor.get("package_phase")
            != "PREQUALIFICATION"
            or embedded_predecessor.get("canonical_identity_sha256")
            != predecessor["package_identity_sha256"]
            or hashlib.sha256(
                canonical_json_bytes(predecessor_artifacts)
            ).hexdigest()
            != predecessor["reviewed_artifacts_identity_sha256"]
            or set(predecessor_by_kind) != expected_predecessor_kinds
            or embedded_predecessor.get("task11_request")
            != package.get("task11_request")
            or embedded_predecessor.get("task11_boundary")
            != package.get("task11_boundary")
        ):
            raise CampaignPackageError(
                "production predecessor package drifted"
            )
        predecessor_deployment = embedded_predecessor.get(
            "disabled_deployment"
        )
        production_deployment = package.get("disabled_deployment")
        if (
            type(predecessor_deployment) is not dict
            or type(production_deployment) is not dict
            or predecessor_deployment.get(
                "staged_infrastructure_evidence"
            )
            != _rebind_staged_fixed_artifacts(
                production_deployment.get(
                    "staged_infrastructure_evidence"
                ),
                validated_predecessor_artifacts,
            )
        ):
            raise CampaignPackageError(
                "production predecessor staged evidence drifted"
            )
        normalized_successor = _copy_json(package)
        normalized_successor.pop("canonical_identity_sha256", None)
        normalized_successor["package_phase"] = "PREQUALIFICATION"
        normalized_successor["predecessor_identity"] = None
        normalized_successor["predecessor_package"] = None
        normalized_successor["reviewed_artifacts"] = (
            validated_predecessor_artifacts
        )
        normalized_deployment = normalized_successor[
            "disabled_deployment"
        ]
        assert type(normalized_deployment) is dict
        normalized_deployment["staged_infrastructure_evidence"] = (
            predecessor_deployment[
                "staged_infrastructure_evidence"
            ]
        )
        normalized_deployment["adoption_readback_contract"] = (
            predecessor_deployment[
                "adoption_readback_contract"
            ]
        )
        normalized_retry = normalized_successor[
            "production_retry_plan"
        ]
        assert type(normalized_retry) is dict
        immutable_inputs = normalized_retry.get("immutable_inputs")
        if (
            type(immutable_inputs) is not list
            or [
                row.get("artifact_kind")
                if type(row) is dict
                else None
                for row in immutable_inputs
            ]
            != list(_PRODUCTION_RETRY_IMMUTABLE_INPUT_KINDS)
            or any(
                type(row) is not dict
                or row
                != artifacts_by_kind.get(str(row.get("artifact_kind")))
                for row in immutable_inputs
            )
        ):
            raise CampaignPackageError(
                "production retry immutable inputs drifted"
            )
        normalized_retry["immutable_inputs"] = [
            predecessor_by_kind[row["artifact_kind"]]
            for row in immutable_inputs
            if row["artifact_kind"]
            not in PRODUCTION_ONLY_ARTIFACT_KINDS
        ]
        predecessor_unsigned = _copy_json(embedded_predecessor)
        predecessor_unsigned.pop("canonical_identity_sha256", None)
        if normalized_successor != predecessor_unsigned:
            raise CampaignPackageError(
                "production successor differs outside the four exact "
                "production additions"
            )
    else:
        raise CampaignPackageError("campaign package phase drifted")
    retry = package.get("production_retry_plan")
    if type(retry) is not dict:
        raise CampaignPackageError(
            "reviewed repository execution custody is missing"
        )
    execution_custody = validate_reviewed_repository_execution_custody(
        retry.get("execution_custody"),
        artifacts=artifacts_by_kind,
        route=retry.get("route"),
    )
    drivers = execution_custody["repository_driver_manifests"]
    assert type(drivers) is dict
    for field, operation_kind in (
        ("qualification_cache_seed", "qualification-cache-seed"),
        ("h100_qualification", "h100-qualification"),
    ):
        plan = package.get(field)
        if (
            type(plan) is not dict
            or plan.get("execution_custody") != execution_custody
            or plan.get("input") != drivers[operation_kind]
        ):
            raise CampaignPackageError(
                "reviewed repository driver execution custody drifted"
            )
    deployment = package.get("disabled_deployment")
    if (
        type(deployment) is not dict
        or deployment.get("state") != "ADOPTED_DISABLED"
    ):
        raise CampaignPackageError(
            "campaign staged deployment evidence is missing"
        )
    validate_staged_infrastructure_evidence(
        deployment.get("staged_infrastructure_evidence"),
        activation_id=_require_text(
            package.get("activation_id"),
            "activation_id",
        ),
        retained_stack_id=_require_text(
            deployment.get("retained_stack_id"),
            "retained_stack_id",
        ),
        artifacts=artifacts_by_kind,
    )
    identity = package.get("canonical_identity_sha256")
    _require_sha256(identity, "canonical_identity_sha256")
    unhashed = dict(package)
    del unhashed["canonical_identity_sha256"]
    expected = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    if identity != expected:
        raise CampaignPackageError("campaign package identity mismatch")
    return canonical_json_bytes(package) + b"\n"


def _payload_bytes(value: object) -> bytes:
    if type(value) is bytes:
        return value
    if type(value) is bytearray:
        return bytes(value)
    if isinstance(value, io.BufferedIOBase) or hasattr(value, "read"):
        raw = value.read()
        if type(raw) is bytes:
            return raw
    raise CampaignPackageError("Lambda Invoke Payload is not bytes")


def _parse_invoke_result(
    result: object,
    *,
    expected_version: str,
) -> dict[str, object]:
    if type(result) is not dict:
        raise CampaignPackageError("Lambda Invoke result is not an object")
    allowed = {
        "StatusCode",
        "ExecutedVersion",
        "Payload",
        "ResponseMetadata",
        "LogResult",
        "FunctionError",
    }
    required = {"StatusCode", "ExecutedVersion", "Payload"}
    if not required.issubset(result) or not set(result).issubset(allowed):
        raise CampaignPackageError("Lambda Invoke result field set drifted")
    if result.get("StatusCode") != 200:
        raise CampaignPackageError("Lambda Invoke StatusCode is not 200")
    if "FunctionError" in result:
        raise CampaignPackageError("Lambda Invoke FunctionError is present")
    if result.get("ExecutedVersion") != expected_version:
        raise CampaignPackageError("Lambda Invoke ExecutedVersion drifted")
    raw = _payload_bytes(result["Payload"])
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignPackageError("Lambda Invoke payload is not JSON") from error
    if type(payload) is not dict:
        raise CampaignPackageError("Lambda Invoke payload is not an object")
    return payload


def _validate_checksum_base64(value: object) -> None:
    if type(value) is not str:
        raise CampaignPackageError("payload checksum is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise CampaignPackageError("payload checksum is invalid") from error
    if len(decoded) != 32:
        raise CampaignPackageError("payload checksum is invalid")


def _validate_payload_common(
    payload: dict[str, object],
    *,
    exact_fields: set[str],
    record_type: str,
    status: str,
    expected_function_version_arn: str,
) -> None:
    if set(payload) != exact_fields:
        raise CampaignPackageError("Lambda Invoke payload field set drifted")
    if (
        payload["schema_version"] != 1
        or payload["record_type"] != record_type
        or payload["status"] != status
        or payload["account_id"] != ACCOUNT_ID
        or payload["region"] != REGION
        or payload["run_id"] != RUN_ID
        or payload["collector_function_version_arn"]
        != expected_function_version_arn
    ):
        raise CampaignPackageError("Lambda Invoke payload identity drifted")
    activation = _require_text(payload["activation_id"], "activation_id")
    if _ACTIVATION.fullmatch(activation) is None:
        raise CampaignPackageError("Lambda Invoke payload activation is invalid")
    _require_version_id(payload["version_id"])
    _require_sha256(payload["file_sha256"], "file_sha256")
    _validate_checksum_base64(payload["checksum_sha256_base64"])
    identity = _require_sha256(
        payload["canonical_identity_sha256"],
        "canonical_identity_sha256",
    )
    unhashed = dict(payload)
    del unhashed["canonical_identity_sha256"]
    if identity != hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest():
        raise CampaignPackageError("Lambda Invoke payload self hash drifted")


def validate_collect_invoke_result(
    result: object,
    *,
    expected_function_version_arn: str,
    expected_version: str,
    expected_measurement_id: str,
) -> dict[str, object]:
    """Validate one exact synchronous Task 11 COLLECT invocation result."""

    if _COLLECTOR_ARN.fullmatch(expected_function_version_arn) is None:
        raise CampaignPackageError("expected collector ARN is not pinned")
    match = _COLLECTOR_ARN.fullmatch(expected_function_version_arn)
    assert match is not None
    if match.group(1) != expected_version:
        raise CampaignPackageError("expected ExecutedVersion does not match ARN")
    payload = _parse_invoke_result(result, expected_version=expected_version)
    _validate_payload_common(
        payload,
        exact_fields=_COLLECT_RESULT_FIELDS,
        record_type="glm52_task11_collect_rehearsal_result_v1",
        status="DEPLOYED_REHEARSAL_RECORDED",
        expected_function_version_arn=expected_function_version_arn,
    )
    if not re.fullmatch(r"measurement-(?:0[1-9]|1[0-9]|20)", expected_measurement_id):
        raise CampaignPackageError("expected measurement_id is invalid")
    if payload["measurement_id"] != expected_measurement_id:
        raise CampaignPackageError("Lambda Invoke payload measurement drifted")
    activation = str(payload["activation_id"])
    key = payload["key"]
    if (
        type(key) is not str
        or re.fullmatch(
            r"rehearsal/measurements/%s/[0-9a-f]{64}/%s[.]json"
            % (re.escape(activation), re.escape(expected_measurement_id)),
            key,
        )
        is None
    ):
        raise CampaignPackageError("Lambda Invoke payload key drifted")
    return payload


def validate_finalize_invoke_result(
    result: object,
    *,
    expected_function_version_arn: str,
    expected_version: str,
) -> dict[str, object]:
    """Validate the exact synchronous Task 11 FINALIZE invocation result."""

    match = _COLLECTOR_ARN.fullmatch(expected_function_version_arn)
    if match is None or match.group(1) != expected_version:
        raise CampaignPackageError("expected collector version is not pinned")
    payload = _parse_invoke_result(result, expected_version=expected_version)
    _validate_payload_common(
        payload,
        exact_fields=_FINALIZE_RESULT_FIELDS,
        record_type="glm52_task11_finalize_rehearsal_gate_result_v1",
        status="CLOSURE_BUDGET_PROVEN",
        expected_function_version_arn=expected_function_version_arn,
    )
    if (
        payload["measurement_count"] != 20
        or payload["cold_environment_count"] != 5
    ):
        raise CampaignPackageError("Lambda Invoke finalize payload counts drifted")
    _require_sha256(payload["body_sha256"], "body_sha256")
    _require_sha256(
        payload["measurements_identity_sha256"],
        "measurements_identity_sha256",
    )
    expected_key = "rehearsal/gates/%s/CLOSURE_BUDGET.json" % payload[
        "activation_id"
    ]
    if payload["key"] != expected_key:
        raise CampaignPackageError("Lambda Invoke finalize payload key drifted")
    return payload


__all__ = [
    "ACCOUNT_ID",
    "REGION",
    "PROFILE",
    "RUN_ID",
    "CampaignPackageError",
    "CAMPAIGN_OWNER_APPROVAL_SHA256",
    "build_campaign_package",
    "canonical_campaign_package_bytes",
    "validate_campaign_artifact_coordinate",
    "validate_collect_invoke_result",
    "validate_finalize_invoke_result",
    "validate_reviewed_repository_execution_custody",
]
