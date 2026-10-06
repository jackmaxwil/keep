"""Pure builders for the closed local inputs consumed by Task 13 CLIs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Mapping, Sequence

from .canonical import canonical_json_bytes, canonical_sha256
from .task13_campaign_package import (
    PRODUCTION_ONLY_ARTIFACT_KINDS,
    _ACTIVATION as _PACKAGE_ACTIVATION,
    _ARTIFACT_KEYS as _PACKAGE_ARTIFACT_KEYS,
    _REQUIRED_EXECUTABLE_PATHS as _PACKAGE_REQUIRED_EXECUTABLE_PATHS,
    build_campaign_package,
    canonical_campaign_package_bytes,
    derive_staged_infrastructure_successor_evidence,
    validate_campaign_artifact_coordinate,
)
from .task13_clean_rehearsal import validate_clean_rehearsal_coordinate
from .task13_fixed_artifacts import (
    activation_artifact_key,
    Task13FixedArtifactError,
    build_repository_driver_manifest,
    parse_driver_materialization_request,
    repository_archive_manifest_key,
    validate_repository_archive_manifest,
    validate_repository_archive_payload,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
CAMPAIGN_BUCKET = (
    "keep-glm52-models-246813579024-us-west-2"
)
RUNNER_RELATIVE_PATH = (
    "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
)
COORDINATOR_RELATIVE_PATH = (
    "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
)
REQUIRED_EXECUTABLE_PATHS = tuple(_PACKAGE_REQUIRED_EXECUTABLE_PATHS)
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
_JOURNAL_FIELDS = {
    "schema_version",
    "record_type",
    "operation_id",
    "operation_kind",
    "stage",
    "sequence",
    "state",
    "request_identity_sha256",
    "result_identity_sha256",
    "canonical_identity_sha256",
}
_JOURNAL_STATES = ("PREPARED", "POSSIBLY_SENT", "COMMITTED")
_WORKFLOW_INVENTORY = {
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
        "aws/glm52-gpu/lambda/task10_sole_sender_handler.py"
    ),
}
_MATERIALIZATION_MANIFEST_FIELDS = {
    "task11_boundary",
    "approval_coordinates",
    "task10_task_inputs",
    "h100_resume_ready",
    "task10_worker_descriptor",
    "runner_relative_path",
    "coordinator_relative_path",
    "runner_file_sha256",
    "coordinator_file_sha256",
    "workflow_inventory",
    "reviewed_artifacts_without_task10_authority",
}


class Task13LiveInputError(ValueError):
    """A local Task 13 source did not close one exact live input."""

@dataclass(frozen=True)
class FreshArchiveActivationDriverInputs:
    """In-memory join of one fresh archive and its two existing drivers."""

    activation_id: str
    repository_archive_path: Path
    repository_archive_size_bytes: int
    repository_archive_file_sha256: str
    coordinator_path: Path
    coordinator_size_bytes: int
    coordinator_file_sha256: str
    required_executable_paths: tuple[str, ...]
    repository_executable_file_sha256: Mapping[str, str]
    h100_driver_request: Mapping[str, object]
    cache_seed_driver_request: Mapping[str, object]


def _prequalification_activation_id(value: object) -> str:
    if type(value) is not str or _PACKAGE_ACTIVATION.fullmatch(value) is None:
        raise Task13LiveInputError("activation_id is invalid")
    return value


def _source(path: Path, role: str) -> dict[str, object]:
    exact = Path(path)
    if (
        not exact.is_absolute()
        or not exact.is_file()
        or exact.is_symlink()
        or exact.resolve(strict=True) != exact
    ):
        raise Task13LiveInputError(
            role + " must be an absolute regular non-symlink file"
        )
    raw = exact.read_bytes()
    if not raw:
        raise Task13LiveInputError(role + " source is empty")
    return {
        "role": role,
        "path": str(exact),
        "size_bytes": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_cache_seed_driver_request(
    *,
    activation_id: str,
    campaign_descriptor: Path,
    approval: Path,
    staged_ready: Path,
    staged_ready_version_id: str,
    rehearsal_evidence: Path,
    task: Path,
    config: Path,
    sky_bin: Path,
) -> dict[str, object]:
    """Bind the exact cache-seed submitter argv to concrete local sources."""

    paths = {
        "campaign_descriptor": Path(campaign_descriptor),
        "approval": Path(approval),
        "staged_ready": Path(staged_ready),
        "rehearsal_evidence": Path(rehearsal_evidence),
        "task": Path(task),
        "config": Path(config),
    }
    sources = [_source(paths[role], role) for role in paths]
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_task13_driver_materialization_request_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "operation_kind": "qualification-cache-seed",
        "argv": [
            "aws/glm52-gpu/scripts/submit_sky_campaign.py",
            "cache-seed",
            "acquire-and-launch",
            "--profile",
            PROFILE,
            "--descriptor",
            str(paths["campaign_descriptor"]),
            "--approval",
            str(paths["approval"]),
            "--staged-ready",
            str(paths["staged_ready"]),
            "--staged-ready-version-id",
            staged_ready_version_id,
            "--rehearsal-evidence",
            str(paths["rehearsal_evidence"]),
            "--task",
            str(paths["task"]),
            "--config",
            str(paths["config"]),
            "--sky-bin",
            str(Path(sky_bin)),
        ],
        "environment": {
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
        },
        "source_coordinates": sources,
    }
    value = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    try:
        return parse_driver_materialization_request(value)
    except ValueError as error:
        raise Task13LiveInputError(str(error)) from error


def build_h100_driver_request(
    *,
    activation_id: str,
    campaign_descriptor: Path,
) -> dict[str, object]:
    """Bind the H100 qualification route to one concrete descriptor."""

    descriptor = Path(campaign_descriptor)
    source = _source(descriptor, "campaign_descriptor")
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_task13_driver_materialization_request_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "operation_kind": "h100-qualification",
        "argv": [
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
        ],
        "environment": {
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
            "CAMPAIGN_DESCRIPTOR": str(descriptor),
        },
        "source_coordinates": [source],
    }
    value = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    try:
        return parse_driver_materialization_request(value)
    except ValueError as error:
        raise Task13LiveInputError(str(error)) from error


def _repository_executable_hashes(
    archive_raw: bytes,
) -> dict[str, str]:
    expected = set(REQUIRED_EXECUTABLE_PATHS)
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(
            fileobj=io.BytesIO(archive_raw),
            mode="r:gz",
        ) as archive:
            for member in archive.getmembers():
                raw_name = member.name
                canonical_name = PurePosixPath(raw_name).as_posix()
                directory_spelling = (
                    member.isdir()
                    and raw_name == canonical_name + "/"
                )
                if (
                    not raw_name
                    or canonical_name == "."
                    or (
                        raw_name != canonical_name
                        and not directory_spelling
                    )
                ):
                    raise Task13LiveInputError(
                        "repository archive member path drifted"
                    )
                name = canonical_name
                if name in seen:
                    raise Task13LiveInputError(
                        "repository archive contains a duplicate member"
                    )
                seen.add(name)
                if member.issym() or member.islnk():
                    raise Task13LiveInputError(
                        "repository archive contains a symlink member"
                    )
                if member.isdir():
                    continue
                if not member.isfile():
                    raise Task13LiveInputError(
                        "repository archive member type drifted"
                    )
                if name not in expected:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise Task13LiveInputError(
                        "repository archive executable is unreadable"
                    )
                raw = source.read()
                if not raw:
                    raise Task13LiveInputError(
                        "repository archive executable is empty"
                    )
                hashes[name] = hashlib.sha256(raw).hexdigest()
    except (tarfile.TarError, OSError) as error:
        raise Task13LiveInputError(
            "repository archive executable inventory is unreadable"
        ) from error
    missing = expected.difference(hashes)
    if missing:
        raise Task13LiveInputError(
            "repository archive is missing required executable: "
            + ", ".join(sorted(missing))
        )
    return {
        path: hashes[path]
        for path in REQUIRED_EXECUTABLE_PATHS
    }


def build_fresh_archive_activation_driver_inputs(
    *,
    activation_id: str,
    repository_archive_path: Path,
    coordinator_path: Path,
    h100_campaign_descriptor: Path,
    cache_seed_campaign_descriptor: Path,
    approval: Path,
    staged_ready: Path,
    staged_ready_version_id: str,
    rehearsal_evidence: Path,
    task: Path,
    config: Path,
    sky_bin: Path,
) -> FreshArchiveActivationDriverInputs:
    """Close a fresh activation over one archive and two existing drivers."""

    exact_activation_id = _prequalification_activation_id(activation_id)
    try:
        archive_raw, archive_sha = validate_repository_archive_payload(
            repository_archive_path
        )
    except Task13FixedArtifactError as error:
        raise Task13LiveInputError(str(error)) from error
    executable_hashes = _repository_executable_hashes(archive_raw)
    coordinator = _source(coordinator_path, "coordinator")
    coordinator_sha = str(coordinator["file_sha256"])
    if executable_hashes[COORDINATOR_RELATIVE_PATH] != coordinator_sha:
        raise Task13LiveInputError(
            "repository archive coordinator does not match local coordinator"
        )
    h100_request = build_h100_driver_request(
        activation_id=exact_activation_id,
        campaign_descriptor=h100_campaign_descriptor,
    )
    cache_seed_request = build_cache_seed_driver_request(
        activation_id=exact_activation_id,
        campaign_descriptor=cache_seed_campaign_descriptor,
        approval=approval,
        staged_ready=staged_ready,
        staged_ready_version_id=staged_ready_version_id,
        rehearsal_evidence=rehearsal_evidence,
        task=task,
        config=config,
        sky_bin=sky_bin,
    )
    try:
        h100_request = parse_driver_materialization_request(h100_request)
        cache_seed_request = parse_driver_materialization_request(
            cache_seed_request
        )
    except ValueError as error:
        raise Task13LiveInputError(str(error)) from error
    return FreshArchiveActivationDriverInputs(
        activation_id=exact_activation_id,
        repository_archive_path=Path(repository_archive_path),
        repository_archive_size_bytes=len(archive_raw),
        repository_archive_file_sha256=archive_sha,
        coordinator_path=Path(coordinator_path),
        coordinator_size_bytes=int(coordinator["size_bytes"]),
        coordinator_file_sha256=coordinator_sha,
        required_executable_paths=REQUIRED_EXECUTABLE_PATHS,
        repository_executable_file_sha256=executable_hashes,
        h100_driver_request=h100_request,
        cache_seed_driver_request=cache_seed_request,
    )


def _copy_json(value: object, label: str) -> object:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise Task13LiveInputError(label + " is not canonical JSON") from error


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise Task13LiveInputError(label + " must be a SHA-256")
    return value


def _version_id(value: object, label: str) -> str:
    if (
        type(value) is not str
        or value.lower() in _MUTABLE_VERSION_WORDS
        or _VERSION_ID.fullmatch(value) is None
    ):
        raise Task13LiveInputError(label + " is not an immutable VersionId")
    return value


def _plain_s3_coordinate(
    value: object,
    *,
    label: str,
    expected_key: str | None = None,
) -> dict[str, object]:
    fields = {
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise Task13LiveInputError(label + " coordinate fields drifted")
    if value["bucket"] != CAMPAIGN_BUCKET:
        raise Task13LiveInputError(label + " bucket is not exact")
    if (
        type(value["key"]) is not str
        or not value["key"]
        or (
            expected_key is not None
            and value["key"] != expected_key
        )
    ):
        raise Task13LiveInputError(label + " key is not exact")
    _version_id(value["version_id"], label)
    _sha256(value["file_sha256"], label + " file")
    _sha256(value["body_sha256"], label + " body")
    return _copy_json(value, label)


def _closed_driver_coordinate(
    value: object,
    *,
    artifact_kind: str,
    operation_kind: str,
    activation_id: str,
    request: Mapping[str, object],
) -> dict[str, object]:
    manifest = build_repository_driver_manifest(request)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    coordinate = validate_campaign_artifact_coordinate(value)
    if (
        coordinate["artifact_kind"] != artifact_kind
        or coordinate["key"]
        != activation_artifact_key(artifact_kind, activation_id)
        or coordinate["file_sha256"]
        != hashlib.sha256(manifest_raw).hexdigest()
        or coordinate["body_sha256"]
        != manifest["canonical_identity_sha256"]
        or manifest["operation_kind"] != operation_kind
        or manifest["activation_id"] != activation_id
    ):
        raise Task13LiveInputError(
            artifact_kind + " coordinate does not bind the fresh driver"
        )
    return coordinate


def build_fresh_archive_prequalification_coordinates(
    *,
    fresh_inputs: FreshArchiveActivationDriverInputs,
    repository_archive_coordinate: Mapping[str, object],
    repository_archive_manifest: Mapping[str, object],
    repository_archive_manifest_bytes: bytes,
    h100_driver_coordinate: Mapping[str, object],
    cache_seed_driver_coordinate: Mapping[str, object],
) -> list[dict[str, object]]:
    """Close the three fresh reviewed coordinates from recomputed bytes."""

    if type(fresh_inputs) is not FreshArchiveActivationDriverInputs:
        raise Task13LiveInputError(
            "fresh inputs must be one concrete recomputed input set"
        )
    activation_id = _prequalification_activation_id(
        fresh_inputs.activation_id
    )
    archive_path = Path(fresh_inputs.repository_archive_path)
    if (
        not archive_path.is_absolute()
        or not archive_path.is_file()
        or archive_path.is_symlink()
        or archive_path.resolve(strict=True) != archive_path
        or type(fresh_inputs.repository_executable_file_sha256) is not dict
        or type(fresh_inputs.h100_driver_request) is not dict
        or type(fresh_inputs.cache_seed_driver_request) is not dict
    ):
        raise Task13LiveInputError(
            "fresh inputs are not concrete absolute source bytes"
        )
    try:
        archive_raw, archive_sha = validate_repository_archive_payload(
            archive_path
        )
    except Task13FixedArtifactError as error:
        raise Task13LiveInputError(str(error)) from error
    archive_size = len(archive_raw)
    executable_hashes = _repository_executable_hashes(archive_raw)
    coordinator = _source(fresh_inputs.coordinator_path, "coordinator")
    if (
        fresh_inputs.repository_archive_size_bytes != archive_size
        or fresh_inputs.repository_archive_file_sha256 != archive_sha
        or fresh_inputs.coordinator_size_bytes != coordinator["size_bytes"]
        or fresh_inputs.coordinator_file_sha256
        != coordinator["file_sha256"]
        or fresh_inputs.required_executable_paths
        != REQUIRED_EXECUTABLE_PATHS
        or fresh_inputs.repository_executable_file_sha256
        != executable_hashes
        or executable_hashes[COORDINATOR_RELATIVE_PATH]
        != coordinator["file_sha256"]
    ):
        raise Task13LiveInputError("fresh archive input bytes drifted")
    if type(repository_archive_manifest_bytes) is not bytes:
        raise Task13LiveInputError(
            "repository archive manifest bytes are not exact"
        )
    try:
        manifest = validate_repository_archive_manifest(
            repository_archive_manifest,
            expected_activation_id=activation_id,
            expected_archive_sha256=archive_sha,
            expected_archive_size=archive_size,
            expected_bucket=CAMPAIGN_BUCKET,
        )
    except ValueError as error:
        raise Task13LiveInputError(str(error)) from error
    expected_manifest_raw = canonical_json_bytes(manifest) + b"\n"
    if repository_archive_manifest_bytes != expected_manifest_raw:
        raise Task13LiveInputError(
            "repository archive manifest bytes are not canonical JSON plus LF"
        )
    try:
        archive_coordinate = validate_campaign_artifact_coordinate(
            repository_archive_coordinate
        )
        if (
            archive_coordinate["artifact_kind"] != "REPOSITORY_ARCHIVE"
            or archive_coordinate["key"]
            != repository_archive_manifest_key(activation_id)
            or archive_coordinate["file_sha256"]
            != hashlib.sha256(repository_archive_manifest_bytes).hexdigest()
            or archive_coordinate["body_sha256"]
            != manifest["canonical_identity_sha256"]
        ):
            raise Task13LiveInputError(
                "repository archive coordinate lineage drifted"
            )
        h100_coordinate = _closed_driver_coordinate(
            h100_driver_coordinate,
            artifact_kind="H100_QUALIFICATION_INPUT",
            operation_kind="h100-qualification",
            request=fresh_inputs.h100_driver_request,
            activation_id=activation_id,
        )
        cache_seed_coordinate = _closed_driver_coordinate(
            cache_seed_driver_coordinate,
            artifact_kind="QUALIFICATION_CACHE_SEED_INPUT",
            operation_kind="qualification-cache-seed",
            request=fresh_inputs.cache_seed_driver_request,
            activation_id=activation_id,
        )
    except ValueError as error:
        if isinstance(error, Task13LiveInputError):
            raise
        raise Task13LiveInputError(str(error)) from error
    coordinates = [
        archive_coordinate,
        h100_coordinate,
        cache_seed_coordinate,
    ]
    if {
        str(row["artifact_kind"]) for row in coordinates
    } != {
        "H100_QUALIFICATION_INPUT",
        "QUALIFICATION_CACHE_SEED_INPUT",
        "REPOSITORY_ARCHIVE",
    }:
        raise Task13LiveInputError(
            "fresh coordinate kinds are not the exact closed set"
        )
    return sorted(
        coordinates,
        key=lambda row: str(row["artifact_kind"]),
    )


def _validate_campaign_journal_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if type(rows) not in {list, tuple}:
        raise Task13LiveInputError("campaign journal must be an exact array")
    chains: dict[str, list[dict[str, object]]] = {}
    for value in rows:
        if type(value) is not dict or set(value) != _JOURNAL_FIELDS:
            raise Task13LiveInputError("campaign journal schema drifted")
        body = {
            key: item
            for key, item in value.items()
            if key != "canonical_identity_sha256"
        }
        if (
            value["schema_version"] != 1
            or value["record_type"]
            != "glm52_task13_runner_journal_v1"
            or value["canonical_identity_sha256"]
            != canonical_sha256(body)
            or type(value["operation_id"]) is not str
            or not value["operation_id"]
            or type(value["sequence"]) is not int
        ):
            raise Task13LiveInputError("campaign journal identity drifted")
        _sha256(
            value["request_identity_sha256"],
            "campaign journal request",
        )
        if value["state"] == "COMMITTED":
            _sha256(
                value["result_identity_sha256"],
                "campaign journal result",
            )
        elif value["result_identity_sha256"] is not None:
            raise Task13LiveInputError(
                "campaign journal noncommitted result drifted"
            )
        chains.setdefault(str(value["operation_id"]), []).append(
            _copy_json(value, "campaign journal row")
        )
    for operation_id, chain in chains.items():
        if len(chain) > 3:
            raise Task13LiveInputError(
                "campaign journal chain has extra states: " + operation_id
            )
        for index, row in enumerate(chain, 1):
            if (
                row["sequence"] != index
                or row["state"] != _JOURNAL_STATES[index - 1]
            ):
                raise Task13LiveInputError(
                    "campaign journal chain drifted: " + operation_id
                )
    stage = chains.get("stage:h100-qualification")
    if (
        stage is None
        or len(stage) != 3
        or stage[-1]["state"] != "COMMITTED"
        or any(
            row["operation_kind"] != "stage-completion"
            or row["stage"] != "h100-qualification"
            for row in stage
        )
    ):
        raise Task13LiveInputError(
            "H100 stage completion is not durably committed"
        )
    return stage[-1]


def _validated_h100_stage_result(
    value: object,
    *,
    package_identity_sha256: str,
    reviewed_artifacts_identity_sha256: str,
    committed_stage_row: Mapping[str, object],
) -> dict[str, object]:
    fields = {
        "status",
        "stage",
        "package_identity_sha256",
        "distinct_instance_count",
        "peak_memory_gib",
        "training_steps",
        "h100_resume_ready",
        "canonical_identity_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise Task13LiveInputError("H100 stage result schema drifted")
    body = {
        key: item
        for key, item in value.items()
        if key != "canonical_identity_sha256"
    }
    marker = value["h100_resume_ready"]
    marker_fields = {
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
        "canonical_identity_sha256",
    }
    if type(marker) is not dict or set(marker) != marker_fields:
        raise Task13LiveInputError("H100 resume-ready marker drifted")
    marker_body = {
        key: item
        for key, item in marker.items()
        if key != "canonical_identity_sha256"
    }
    if (
        value["status"] != "STAGE_COMMITTED"
        or value["stage"] != "h100-qualification"
        or value["package_identity_sha256"]
        != package_identity_sha256
        or value["distinct_instance_count"] != 2
        or type(value["peak_memory_gib"]) not in {int, float}
        or type(value["peak_memory_gib"]) is bool
        or value["peak_memory_gib"] >= 70
        or value["training_steps"] != 2
        or value["canonical_identity_sha256"] != canonical_sha256(body)
        or marker["key"]
        != (
            "campaigns/%s/qualification/H100_RESUME_READY.json"
            % RUN_ID
        )
        or marker["canonical_identity_sha256"]
        != canonical_sha256(marker_body)
    ):
        raise Task13LiveInputError("H100 stage result identity drifted")
    _version_id(marker["version_id"], "H100 resume-ready marker")
    _sha256(marker["file_sha256"], "H100 resume-ready file")
    _sha256(marker["body_sha256"], "H100 resume-ready body")
    expected_request_identity = canonical_sha256(
        {
            "stage": "h100-qualification",
            "package_identity_sha256": package_identity_sha256,
            "reviewed_artifacts_identity_sha256": (
                reviewed_artifacts_identity_sha256
            ),
        }
    )
    summary = {
        key: value[key]
        for key in (
            "distinct_instance_count",
            "peak_memory_gib",
            "training_steps",
            "h100_resume_ready",
        )
    }
    if (
        committed_stage_row["request_identity_sha256"]
        != expected_request_identity
        or committed_stage_row["result_identity_sha256"]
        != canonical_sha256(summary)
    ):
        raise Task13LiveInputError(
            "H100 stage result identity does not match committed journal"
        )
    return _copy_json(value, "H100 stage result")


def validate_task10_materialization_manifest(
    value: object,
    *,
    repository_root: Path,
) -> dict[str, object]:
    """Validate the complete local-to-live Task 10 authority manifest."""

    if type(value) is not dict or set(value) != _MATERIALIZATION_MANIFEST_FIELDS:
        raise Task13LiveInputError(
            "Task10 materialization manifest fields drifted"
        )
    root = Path(repository_root)
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise Task13LiveInputError("repository root is not exact")
    boundary = _plain_s3_coordinate(
        value["task11_boundary"],
        label="Task11 boundary",
    )
    if (
        not str(boundary["key"]).startswith(
            "campaigns/%s/authorities/task11/" % RUN_ID
        )
        or not str(boundary["key"]).endswith("/TASK11_BOUNDARY.json")
    ):
        raise Task13LiveInputError("Task11 boundary key is not exact")
    approvals = value["approval_coordinates"]
    if type(approvals) is not dict or set(approvals) != {
        "GPU_SPEND_APPROVAL",
        "SUPPORT_APPROVAL",
        "RESIDUAL_LIABILITY_APPROVAL",
    }:
        raise Task13LiveInputError("Task10 approval set drifted")
    validated_approvals = {
        kind: validate_campaign_artifact_coordinate(coordinate)
        for kind, coordinate in approvals.items()
    }
    if any(
        coordinate["artifact_kind"] != kind
        for kind, coordinate in validated_approvals.items()
    ):
        raise Task13LiveInputError("Task10 approval kind drifted")
    task_inputs = validate_campaign_artifact_coordinate(
        value["task10_task_inputs"]
    )
    worker = validate_campaign_artifact_coordinate(
        value["task10_worker_descriptor"]
    )
    if (
        task_inputs["artifact_kind"] != "TASK10_TASK_INPUTS"
        or worker["artifact_kind"] != "TASK10_WORKER_DESCRIPTOR"
    ):
        raise Task13LiveInputError("Task10 input coordinate kind drifted")
    h100 = _plain_s3_coordinate(
        value["h100_resume_ready"],
        label="H100 resume-ready",
        expected_key=(
            "campaigns/%s/qualification/H100_RESUME_READY.json"
            % RUN_ID
        ),
    )
    if (
        value["runner_relative_path"] != RUNNER_RELATIVE_PATH
        or value["coordinator_relative_path"]
        != COORDINATOR_RELATIVE_PATH
    ):
        raise Task13LiveInputError("Task13 executable path drifted")
    runner = root / RUNNER_RELATIVE_PATH
    coordinator = root / COORDINATOR_RELATIVE_PATH
    observed_runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    observed_coordinator_sha = hashlib.sha256(
        coordinator.read_bytes()
    ).hexdigest()
    if (
        value["runner_file_sha256"] != observed_runner_sha
        or value["coordinator_file_sha256"]
        != observed_coordinator_sha
    ):
        raise Task13LiveInputError("Task13 executable identity drifted")
    if value["workflow_inventory"] != _WORKFLOW_INVENTORY:
        raise Task13LiveInputError("Task10 workflow inventory drifted")
    reviewed = value["reviewed_artifacts_without_task10_authority"]
    if type(reviewed) is not list:
        raise Task13LiveInputError("prequalification reviewed set drifted")
    validated_reviewed = [
        validate_campaign_artifact_coordinate(item)
        for item in reviewed
    ]
    kinds = [str(item["artifact_kind"]) for item in validated_reviewed]
    if (
        kinds != sorted(kinds)
        or len(kinds) != 21
        or len(set(kinds)) != 21
        or {
            "CLEAN_REHEARSAL",
            "TASK10_PRODUCTION_AUTHORITY",
            "TASK10_WORKER_DESCRIPTOR",
            "TASK10_TASK_INPUTS",
        }.intersection(kinds)
    ):
        raise Task13LiveInputError("prequalification reviewed set drifted")
    return {
        "task11_boundary": boundary,
        "approval_coordinates": {
            key: validated_approvals[key]
            for key in sorted(validated_approvals)
        },
        "task10_task_inputs": task_inputs,
        "h100_resume_ready": h100,
        "task10_worker_descriptor": worker,
        "runner_relative_path": RUNNER_RELATIVE_PATH,
        "coordinator_relative_path": COORDINATOR_RELATIVE_PATH,
        "runner_file_sha256": observed_runner_sha,
        "coordinator_file_sha256": observed_coordinator_sha,
        "workflow_inventory": _copy_json(
            _WORKFLOW_INVENTORY,
            "Task10 workflow inventory",
        ),
        "reviewed_artifacts_without_task10_authority": (
            validated_reviewed
        ),
    }


def build_task10_materialization_manifest(
    *,
    prequalification_package: Mapping[str, object],
    prequalification_reviewed_artifacts: Sequence[Mapping[str, object]],
    task10_task_inputs: Mapping[str, object],
    task10_worker_descriptor: Mapping[str, object],
    h100_stage_result: Mapping[str, object],
    campaign_journal_rows: Sequence[Mapping[str, object]],
    repository_root: Path,
) -> dict[str, object]:
    """Derive the Task 10 materializer input from one committed H100 stage."""

    try:
        canonical_campaign_package_bytes(prequalification_package)
    except ValueError as error:
        raise Task13LiveInputError(
            "prequalification package identity drifted"
        ) from error
    if (
        prequalification_package.get("package_phase") != "PREQUALIFICATION"
        or prequalification_package.get("reviewed_artifacts")
        != prequalification_reviewed_artifacts
    ):
        raise Task13LiveInputError(
            "prequalification reviewed artifacts disagree"
        )
    reviewed_identity = canonical_sha256(
        prequalification_reviewed_artifacts
    )
    committed = _validate_campaign_journal_rows(campaign_journal_rows)
    result = _validated_h100_stage_result(
        h100_stage_result,
        package_identity_sha256=str(
            prequalification_package["canonical_identity_sha256"]
        ),
        reviewed_artifacts_identity_sha256=reviewed_identity,
        committed_stage_row=committed,
    )
    by_kind = {
        str(item["artifact_kind"]): item
        for item in prequalification_reviewed_artifacts
    }
    task_inputs = validate_campaign_artifact_coordinate(
        task10_task_inputs
    )
    worker_descriptor = validate_campaign_artifact_coordinate(
        task10_worker_descriptor
    )
    if (
        task_inputs["artifact_kind"] != "TASK10_TASK_INPUTS"
        or worker_descriptor["artifact_kind"]
        != "TASK10_WORKER_DESCRIPTOR"
        or {
            "TASK10_TASK_INPUTS",
            "TASK10_WORKER_DESCRIPTOR",
            "TASK10_PRODUCTION_AUTHORITY",
        }.intersection(by_kind)
    ):
        raise Task13LiveInputError(
            "post-H100 Task10 coordinate phase drifted"
        )
    reconciliation = prequalification_package.get(
        "production_retry_plan",
        {},
    )
    workflow = (
        reconciliation.get("workflow_reconciliation_contract")
        if type(reconciliation) is dict
        else None
    )
    if type(workflow) is not dict:
        raise Task13LiveInputError("Task10 workflow inventory is missing")
    inventory = {
        key: workflow.get(key) for key in _WORKFLOW_INVENTORY
    }
    marker = result["h100_resume_ready"]
    manifest = {
        "task11_boundary": _copy_json(
            prequalification_package["task11_boundary"],
            "Task11 boundary",
        ),
        "approval_coordinates": {
            kind: _copy_json(by_kind[kind], kind)
            for kind in (
                "GPU_SPEND_APPROVAL",
                "SUPPORT_APPROVAL",
                "RESIDUAL_LIABILITY_APPROVAL",
            )
        },
        "task10_task_inputs": _copy_json(
            task_inputs,
            "Task10 task inputs",
        ),
        "h100_resume_ready": {
            "bucket": CAMPAIGN_BUCKET,
            "key": marker["key"],
            "version_id": marker["version_id"],
            "file_sha256": marker["file_sha256"],
            "body_sha256": marker["body_sha256"],
        },
        "task10_worker_descriptor": _copy_json(
            worker_descriptor,
            "Task10 worker descriptor",
        ),
        "runner_relative_path": RUNNER_RELATIVE_PATH,
        "coordinator_relative_path": COORDINATOR_RELATIVE_PATH,
        "runner_file_sha256": hashlib.sha256(
            (Path(repository_root) / RUNNER_RELATIVE_PATH).read_bytes()
        ).hexdigest(),
        "coordinator_file_sha256": hashlib.sha256(
            (
                Path(repository_root) / COORDINATOR_RELATIVE_PATH
            ).read_bytes()
        ).hexdigest(),
        "workflow_inventory": inventory,
        "reviewed_artifacts_without_task10_authority": _copy_json(
            prequalification_reviewed_artifacts,
            "prequalification reviewed artifacts",
        ),
    }
    return validate_task10_materialization_manifest(
        manifest,
        repository_root=repository_root,
    )


def _production_coordinate(
    value: object,
    *,
    expected_kind: str,
    label: str,
) -> dict[str, object]:
    try:
        coordinate = validate_campaign_artifact_coordinate(value)
    except ValueError as error:
        raise Task13LiveInputError(
            label + " coordinate is not canonical"
        ) from error
    if coordinate["artifact_kind"] != expected_kind:
        raise Task13LiveInputError(label + " artifact kind drifted")
    if expected_kind == "CLEAN_REHEARSAL":
        try:
            clean = validate_clean_rehearsal_coordinate(coordinate)
        except ValueError as error:
            raise Task13LiveInputError(
                "clean rehearsal coordinate is not canonical"
            ) from error
        if clean != coordinate:
            raise Task13LiveInputError(
                "clean rehearsal coordinate normalization drifted"
            )
    return coordinate


def build_production_reviewed_artifacts(
    *,
    prequalification_reviewed_artifacts: object,
    clean_rehearsal_coordinate: object,
    task10_production_authority: object,
    task10_worker_descriptor: object,
    task10_task_inputs: object,
) -> list[dict[str, object]]:
    """Close one exact prequalification list with four production coordinates."""

    if type(prequalification_reviewed_artifacts) is not list:
        raise Task13LiveInputError(
            "prequalification reviewed artifacts must be one exact list"
        )
    try:
        predecessor = [
            validate_campaign_artifact_coordinate(row)
            for row in prequalification_reviewed_artifacts
        ]
    except ValueError as error:
        raise Task13LiveInputError(
            "prequalification reviewed artifact coordinate drifted"
        ) from error
    predecessor_kinds = [
        str(coordinate["artifact_kind"]) for coordinate in predecessor
    ]
    expected_predecessor_kinds = (
        set(_PACKAGE_ARTIFACT_KEYS) - set(PRODUCTION_ONLY_ARTIFACT_KINDS)
    )
    if (
        len(predecessor) != 21
        or predecessor_kinds != sorted(predecessor_kinds)
        or len(set(predecessor_kinds)) != len(predecessor_kinds)
        or set(predecessor_kinds) != expected_predecessor_kinds
    ):
        raise Task13LiveInputError(
            "prequalification reviewed artifact kinds drifted"
        )

    additions = [
        _production_coordinate(
            clean_rehearsal_coordinate,
            expected_kind="CLEAN_REHEARSAL",
            label="clean rehearsal",
        ),
        _production_coordinate(
            task10_production_authority,
            expected_kind="TASK10_PRODUCTION_AUTHORITY",
            label="Task10 production authority",
        ),
        _production_coordinate(
            task10_worker_descriptor,
            expected_kind="TASK10_WORKER_DESCRIPTOR",
            label="Task10 worker descriptor",
        ),
        _production_coordinate(
            task10_task_inputs,
            expected_kind="TASK10_TASK_INPUTS",
            label="Task10 task inputs",
        ),
    ]
    result = sorted(
        [*predecessor, *additions],
        key=lambda coordinate: str(coordinate["artifact_kind"]),
    )
    result_by_kind = {
        str(coordinate["artifact_kind"]): coordinate for coordinate in result
    }
    predecessor_by_kind = {
        str(coordinate["artifact_kind"]): coordinate
        for coordinate in prequalification_reviewed_artifacts
    }
    if (
        len(result) != 25
        or len(result_by_kind) != 25
        or set(result_by_kind)
        != expected_predecessor_kinds | set(PRODUCTION_ONLY_ARTIFACT_KINDS)
        or any(
            result_by_kind[kind] != coordinate
            for kind, coordinate in predecessor_by_kind.items()
        )
    ):
        raise Task13LiveInputError(
            "production reviewed artifact lineage drifted"
        )
    return result


def build_production_successor_request(
    *,
    prequalification_request: Mapping[str, object],
    prequalification_package: Mapping[str, object],
    prequalification_reviewed_artifacts: Sequence[Mapping[str, object]],
    production_reviewed_artifacts: Sequence[Mapping[str, object]],
    clean_rehearsal_evidence: object,
) -> dict[str, object]:
    """Add clean evidence and the three Task 10 production coordinates."""

    try:
        canonical_campaign_package_bytes(prequalification_package)
    except ValueError as error:
        raise Task13LiveInputError(
            "prequalification package identity drifted"
        ) from error
    if (
        prequalification_package.get("package_phase") != "PREQUALIFICATION"
        or prequalification_package.get("reviewed_artifacts")
        != prequalification_reviewed_artifacts
        or build_campaign_package(prequalification_request)
        != prequalification_package
    ):
        raise Task13LiveInputError(
            "prequalification package/request ancestry drifted"
        )
    try:
        predecessor = [
            validate_campaign_artifact_coordinate(item)
            for item in prequalification_reviewed_artifacts
        ]
        successor = [
            validate_campaign_artifact_coordinate(item)
            for item in production_reviewed_artifacts
        ]
    except ValueError as error:
        raise Task13LiveInputError(
            "production successor artifact coordinate drifted"
        ) from error
    predecessor_by_kind = {
        str(item["artifact_kind"]): item for item in predecessor
    }
    successor_by_kind = {
        str(item["artifact_kind"]): item for item in successor
    }
    added_kinds = set(successor_by_kind) - set(predecessor_by_kind)
    changed_predecessor_kinds = {
        kind
        for kind, coordinate in predecessor_by_kind.items()
        if successor_by_kind.get(kind) != coordinate
    }
    if (
        len(predecessor) != 21
        or len(successor) != 25
        or len(predecessor_by_kind) != 21
        or len(successor_by_kind) != 25
        or added_kinds
        != {
            "CLEAN_REHEARSAL",
            "TASK10_PRODUCTION_AUTHORITY",
            "TASK10_WORKER_DESCRIPTOR",
            "TASK10_TASK_INPUTS",
        }
        or changed_predecessor_kinds
    ):
        raise Task13LiveInputError(
            "production successor must preserve all 21 predecessor "
            "coordinates and add exactly CLEAN_REHEARSAL plus the three "
            "post-H100 Task10 coordinates"
        )
    if clean_rehearsal_evidence is None:
        raise Task13LiveInputError(
            "production successor requires clean rehearsal evidence"
        )
    request = {
        **_copy_json(
            prequalification_request,
            "prequalification request",
        ),
        "artifacts": successor,
        "staged_infrastructure_evidence": (
            derive_staged_infrastructure_successor_evidence(
                prequalification_request[
                    "staged_infrastructure_evidence"
                ],
                successor,
            )
        ),
    }
    try:
        package = build_campaign_package(
            request,
            predecessor_package=prequalification_package,
            predecessor_reviewed_artifacts=predecessor,
            clean_rehearsal_evidence=clean_rehearsal_evidence,
        )
    except ValueError as error:
        raise Task13LiveInputError(str(error)) from error
    if package.get("package_phase") != "PRODUCTION":
        raise Task13LiveInputError("production successor did not close")
    return request


def build_prequalification_base_request(
    *,
    activation_id: str,
    collector_version_arn: str,
    task11_request: Mapping[str, object],
    task11_boundary: Mapping[str, object],
    staged_infrastructure_evidence: Mapping[str, object],
    retained_stack_id: str,
    fence_change_set_name: str,
    support_change_set_name: str,
    monitor_descriptor_path: str,
) -> dict[str, object]:
    """Build the fixed-scope request fields before reviewed artifacts join."""

    exact_activation_id = _prequalification_activation_id(activation_id)
    value = {
        "schema_version": 1,
        "record_type": "glm52_task13_campaign_package_request_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": exact_activation_id,
        "collector_version_arn": collector_version_arn,
        "task11_request": _copy_json(task11_request, "Task11 request"),
        "task11_boundary": _copy_json(task11_boundary, "Task11 boundary"),
        "staged_infrastructure_evidence": _copy_json(
            staged_infrastructure_evidence,
            "staged infrastructure evidence",
        ),
        "retained_stack_id": retained_stack_id,
        "fence_stack_name": "keep-glm52-h1g-fence",
        "support_stack_name": "keep-glm52-h1g-support",
        "fence_change_set_name": fence_change_set_name,
        "support_change_set_name": support_change_set_name,
        "retained_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-cloudformation-deployment"
        ),
        "fence_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-fence-service"
        ),
        "support_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-cloudformation-deployment"
        ),
        "monitor_descriptor_path": monitor_descriptor_path,
    }
    return value


def assemble_prequalification_inputs(
    *,
    base_request: Mapping[str, object],
    coordinate_documents: Sequence[object],
) -> dict[str, object]:
    """Flatten reviewed coordinate files into one exact prequalification set."""

    if type(base_request) is not dict or "artifacts" in base_request:
        raise Task13LiveInputError(
            "base request must be one exact object without artifacts"
        )
    rows = []
    for document in coordinate_documents:
        if type(document) is dict:
            rows.append(_copy_json(document, "artifact coordinate"))
        elif type(document) is list:
            rows.extend(
                _copy_json(item, "artifact coordinate")
                for item in document
            )
        else:
            raise Task13LiveInputError(
                "coordinate source must contain an object or array"
            )
    if any(type(row) is not dict for row in rows):
        raise Task13LiveInputError("artifact coordinate must be an object")
    coordinates = sorted(
        rows,
        key=lambda row: str(row.get("artifact_kind")),
    )
    request = {
        **_copy_json(base_request, "base request"),
        "artifacts": coordinates,
    }
    try:
        package = build_campaign_package(request)
    except ValueError as error:
        raise Task13LiveInputError(str(error)) from error
    if package.get("package_phase") != "PREQUALIFICATION":
        raise Task13LiveInputError(
            "coordinate sources did not close prequalification"
        )
    reviewed = package["reviewed_artifacts"]
    return {
        "base_request": _copy_json(base_request, "base request"),
        "coordinates": reviewed,
        "request": {
            **_copy_json(base_request, "base request"),
            "artifacts": reviewed,
        },
        "reviewed_artifacts": reviewed,
    }


__all__ = [
    "REQUIRED_EXECUTABLE_PATHS",
    "FreshArchiveActivationDriverInputs",
    "Task13LiveInputError",
    "assemble_prequalification_inputs",
    "build_cache_seed_driver_request",
    "build_fresh_archive_activation_driver_inputs",
    "build_fresh_archive_prequalification_coordinates",
    "build_h100_driver_request",
    "build_prequalification_base_request",
    "build_production_successor_request",
    "build_task10_materialization_manifest",
    "validate_task10_materialization_manifest",
]
