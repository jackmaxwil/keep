#!/usr/bin/env python3
"""Guarded qualification submission I/O orchestration for SkyPilot."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import base64
import datetime as datetime_module
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

UTC = timezone.utc
if not hasattr(datetime_module, "UTC"):
    datetime_module.UTC = UTC

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlx_vq.quality.glm52_gpu_launch_allowance import (
    build_gpu_launch_allowance,
    gpu_launch_allowance_s3_key,
    validate_gpu_launch_allowance,
)
from mlx_vq.quality.glm52_gpu_spend_snapshot import (
    build_gpu_spend_snapshot,
    validate_gpu_spend_snapshot,
)
from mlx_vq.quality.glm52_qualification_cache_seed import (
    accepted_s3_key,
    validate_qualification_cache_seed_accepted,
)
from mlx_vq.quality.glm52_qualification_submission_ready import (
    gpu_spend_snapshot_s3_key,
    qualification_submission_ready_s3_key,
    rehearsal_evidence_s3_key,
    validate_qualification_submission_ready,
)
from mlx_vq.quality.glm52_s3_artifact_audit import (
    validate_s3_artifact_inventory,
)
from mlx_vq.quality.glm52_sky_cache_seed_launch import (
    CLAIM_DIGEST_FIELD as CACHE_SEED_CLAIM_DIGEST_FIELD,
)
from mlx_vq.quality.glm52_sky_cache_seed_launch import (
    INTENT_DIGEST_FIELD as CACHE_SEED_INTENT_DIGEST_FIELD,
)
from mlx_vq.quality.glm52_sky_cache_seed_launch import (
    MANAGED_MODE as CACHE_SEED_MODE,
)
from mlx_vq.quality.glm52_sky_cache_seed_launch import (
    MAX_LIVE_AGE_SECONDS,
    _prevalidate_cache_seed_launch_claim_references,
    build_cache_seed_launch_claim,
    build_cache_seed_submission_intent,
    cache_seed_launch_claim_s3_key,
    cache_seed_submission_intent_s3_key,
    ec2_tagged_p5_zero_inventory_observation_s3_key,
    s3_spend_ledger_absence_observation_s3_key,
    validate_cache_seed_launch_claim,
    validate_cache_seed_submission_intent,
)
from mlx_vq.quality.glm52_sky_cache_seed_launch import (
    canonical_file_bytes as cache_seed_canonical_file_bytes,
)
from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_ACCOUNT_ID,
    APPROVED_REGION,
    require_approved_aws_identity,
    validate_gpu_spend_approval,
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import (
    build_must_start_controller_observation,
    validate_must_start_controller_observation,
)
from mlx_vq.quality.glm52_sky_must_start import (
    canonical_bytes as must_start_canonical_bytes,
)
from mlx_vq.quality.glm52_sky_must_start_dynamic import (
    DynamicJobBindingValidationError,
    build_dynamic_v2_job_binding,
    dynamic_v2_canonical_bytes,
    dynamic_v2_job_binding_s3_key,
    validate_dynamic_v2_job_binding,
)
from mlx_vq.quality.glm52_sky_submission_acquisition import (
    ImmutableJsonArtifact,
    SubmissionAcquisitionReadResult,
    build_submission_acquired,
    decide_submission_acquisition,
    submission_acquired_file_bytes,
    submission_acquired_s3_key,
    validate_submission_acquired,
)
from mlx_vq.quality.glm52_sky_submission_lifecycle import (
    ACCEPTED_DIGEST_FIELD,
    INTENT_DIGEST_FIELD,
    build_submission_accepted,
    build_submission_intent,
    expected_sky_job_name,
    submission_accepted_s3_key,
    submission_intent_s3_key,
    validate_submission_accepted,
    validate_submission_intent,
)
from mlx_vq.quality.glm52_sky_submission_live_authority import (
    COORDINATOR_MODE,
    build_controller_baseline,
    build_must_start_control_plane_ready,
    canonical_file_bytes,
    controller_baseline_file_bytes,
    controller_baseline_s3_key,
    must_start_control_plane_ready_file_bytes,
    must_start_control_plane_ready_s3_key,
    validate_controller_baseline,
    validate_must_start_control_plane_ready,
)
from mlx_vq.quality.glm52_sky_worker_start_v2_readiness import (
    WorkerStartV2ReadinessArtifact,
    WorkerStartV2ReadinessAuthorities,
    WorkerStartV2SourceSnapshot,
    build_worker_start_v2_readiness,
    validate_worker_start_v2_readiness,
)
from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.task10_production import (
    ProductionOutcome as Task10ProductionOutcome,
    ProductionRequest as Task10ProductionRequest,
    ProductionRouteError as Task10ProductionRouteError,
    assert_no_raw_effect_surface,
    production_authority_from_mapping,
    run_production,
)
from glm52_enforcement.task10_task13_bridge import (
    Task13BridgeError,
    Task13LaunchBoundary,
    task13_launch_bridge_from_mapping,
)
from glm52_enforcement.task10_worker import (
    MountFreeTaskInputs,
    validate_task_inputs,
    worker_bootstrap_descriptor_from_mapping,
)

QualificationAction = Literal[
    "validate-only",
    "prepare-intent",
    "acquire-and-launch",
    "reconcile",
]
CacheSeedAction = Literal["validate-only", "acquire-and-launch"]

EXIT_OK = 0
EXIT_USAGE = 64
EXIT_FAIL_CLOSED = 70
EXIT_RECONCILE_PENDING = 75
EXIT_DEPLOYMENT_REQUIRED = 78

EXACT_PROFILE = "keep-gpu"
QUALIFICATION_MODE = "qualification"
PRODUCTION_MODE = "production"
EXPECTED_WORKSPACE = "default"
EXPECTED_SKYPILOT_VERSION = "0.13.0"
EXPECTED_INSTANCE_TYPE = "p5.48xlarge"
MAX_LIVE_AGE = timedelta(seconds=60)
MAX_STATIC_WINDOW = timedelta(hours=12)
SKY_LAUNCH_TIMEOUT_SECONDS = 120
SKY_VALIDATION_TIMEOUT_SECONDS = 30
_SEALED_TASK_RELATIVE_PATH = "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
_SEALED_CONFIG_RELATIVE_PATH = "config/skypilot-config.yaml"
_SEALED_MOUNT_SOURCES = (
    (
        "aws/glm52-gpu/skypilot/publish_worker_start_v2.py",
        "publisher_raw",
    ),
    (
        "aws/glm52-gpu/skypilot/run_h100_qualification.sh",
        "h100_qualification_worker_raw",
    ),
    (
        "src/glm52_enforcement/glm52_h100_qualification.py",
        "h100_qualification_native_raw",
    ),
    (
        "src/mlx_vq/quality/glm52_h100_qualification.py",
        "h100_qualification_raw",
    ),
    (
        "src/glm52_enforcement/glm52_sky_campaign.py",
        "campaign_policy_native_raw",
    ),
    (
        "src/glm52_enforcement/glm52_sky_must_start.py",
        "must_start_policy_native_raw",
    ),
    (
        "src/mlx_vq/quality/glm52_sky_campaign.py",
        "campaign_policy_raw",
    ),
    (
        "src/mlx_vq/quality/glm52_sky_must_start.py",
        "must_start_policy_raw",
    ),
    (
        "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py",
        "dynamic_policy_raw",
    ),
    (
        "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py",
        "worker_policy_raw",
    ),
    (
        "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py",
        "coordinator_raw",
    ),
)
_SEALED_SOURCE_SNAPSHOT_FIELDS = frozenset(
    {
        "publisher_raw",
        "h100_qualification_worker_raw",
        "h100_qualification_native_raw",
        "h100_qualification_raw",
        "campaign_policy_native_raw",
        "must_start_policy_native_raw",
        "campaign_policy_raw",
        "must_start_policy_raw",
        "dynamic_policy_raw",
        "worker_policy_raw",
        "coordinator_raw",
        "sky_task_raw",
        "bootstrap_raw",
        "managed_entrypoint_raw",
    }
)
_HEX64 = frozenset("0123456789abcdef")
_STAGED_READY_V2_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "run_id",
        "descriptor_key",
        "descriptor_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "bundle_manifest_key",
        "bundle_manifest_file_sha256",
        "bundle_manifest_body_sha256",
        "bundle_manifest_version_id",
        "staged_object_version_ids",
        "artifact_audit_key",
        "artifact_audit_sha256",
        "staged_at",
        "ready_body_sha256",
    }
)
_STAGED_VERSION_ROLES = frozenset(
    {
        "repository_tar",
        "approval",
        "training_config",
        "watchdog",
        "artifact_inventory",
        "artifact_audit",
        "descriptor",
    }
)
_BUNDLE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "run_id",
        "bucket",
        "files",
        "descriptor_body_sha256",
        "bundle_manifest_body_sha256",
    }
)
_BUNDLE_FILE_FIELDS = frozenset(
    {"local_name", "key", "role", "stage_order", "size", "sha256"}
)
_BUNDLE_ROLES = frozenset(
    {
        "repository_tar",
        "approval",
        "training_config",
        "watchdog",
        "artifact_inventory",
        "descriptor",
    }
)
_ARTIFACT_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "audit_pass",
        "run_id",
        "bucket",
        "inventory_body_sha256",
        "object_count",
        "object_bytes",
        "safetensors_object_count",
        "safetensors_tensor_count",
    }
)
_ACTIVE_CONTROLLER_STATUSES = frozenset(
    {
        "PENDING",
        "SUBMITTED",
        "STARTING",
        "RUNNING",
        "WINDING_DOWN",
        "RECOVERING",
        "CANCELLING",
    }
)
_CONTROLLER_STATUSES = frozenset(
    {
        *_ACTIVE_CONTROLLER_STATUSES,
        "SUCCEEDED",
        "FAILED",
        "FAILED_SETUP",
        "FAILED_PRECHECKS",
        "FAILED_NO_RESOURCE",
        "FAILED_CONTROLLER",
        "CANCELLED",
    }
)
_TERMINAL_CONTROLLER_STATUSES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "FAILED_SETUP",
        "FAILED_PRECHECKS",
        "FAILED_NO_RESOURCE",
        "FAILED_CONTROLLER",
        "CANCELLED",
    }
)
_CONTROLLER_STATUS_TRANSITIONS = {
    "PENDING": _CONTROLLER_STATUSES,
    "STARTING": frozenset(
        {
            "STARTING",
            "RUNNING",
            "RECOVERING",
            "CANCELLING",
            *_TERMINAL_CONTROLLER_STATUSES,
        }
    ),
    "RUNNING": frozenset(
        {
            "RUNNING",
            "RECOVERING",
            "CANCELLING",
            *_TERMINAL_CONTROLLER_STATUSES,
        }
    ),
    "RECOVERING": frozenset(
        {
            "RECOVERING",
            "RUNNING",
            "CANCELLING",
            *_TERMINAL_CONTROLLER_STATUSES,
        }
    ),
    "CANCELLING": frozenset(
        {
            "CANCELLING",
            "CANCELLED",
            "FAILED_CONTROLLER",
        }
    ),
    **{status: frozenset({status}) for status in _TERMINAL_CONTROLLER_STATUSES},
}
_SCHEDULE_STATE_TRANSITIONS = {
    "INACTIVE": frozenset(
        {
            "INACTIVE",
            "WAITING",
            "LAUNCHING",
            "ALIVE",
            "ALIVE_BACKOFF",
            "ALIVE_WAITING",
            "DONE",
        }
    ),
    "WAITING": frozenset(
        {
            "WAITING",
            "LAUNCHING",
            "ALIVE",
            "ALIVE_BACKOFF",
            "ALIVE_WAITING",
            "DONE",
        }
    ),
    "LAUNCHING": frozenset(
        {
            "LAUNCHING",
            "ALIVE",
            "ALIVE_BACKOFF",
            "ALIVE_WAITING",
            "DONE",
        }
    ),
    "ALIVE": frozenset(
        {
            "ALIVE",
            "ALIVE_WAITING",
            "LAUNCHING",
            "ALIVE_BACKOFF",
            "DONE",
        }
    ),
    "ALIVE_BACKOFF": frozenset(
        {
            "ALIVE_BACKOFF",
            "ALIVE_WAITING",
            "LAUNCHING",
            "ALIVE",
            "DONE",
        }
    ),
    "ALIVE_WAITING": frozenset(
        {
            "ALIVE_WAITING",
            "LAUNCHING",
            "ALIVE",
            "ALIVE_BACKOFF",
            "DONE",
        }
    ),
    "DONE": frozenset({"DONE"}),
}
_HISTORY_ROW_FIELDS = frozenset(
    {
        "sky_job_id",
        "sky_job_name",
        "workspace",
        "controller_submitted_at",
        "controller_status",
        "controller_identity",
        "schedule_state",
        "start_at",
        "worker_cluster_name",
        "recovery_count",
    }
)
_BASELINE_ROW_FIELDS = (
    "sky_job_id",
    "sky_job_name",
    "workspace",
    "controller_submitted_at",
    "controller_status",
    "controller_identity",
)


class SubmissionIntegrationError(ValueError):
    """A malformed, foreign, stale, or ambiguous submission state."""


class SubmissionUsageError(SubmissionIntegrationError):
    """A local argument, profile, or regular-file authority is invalid."""


class SubmissionDeploymentRequired(SubmissionIntegrationError):
    """The separately deployed dynamic must-start coordinator is absent."""


class SubmissionReconcilePending(SubmissionIntegrationError):
    """The acquired intent is valid but controller acceptance is not visible."""


@dataclass(frozen=True)
class LocalAuthorityPaths:
    descriptor: Path
    seed_descriptor: Path
    approval: Path
    staged_ready: Path
    rehearsal_evidence: Path
    cache_seed_accepted: Path
    gpu_spend_snapshot: Path
    qualification_ready: Path
    task: Path
    config: Path


@dataclass(frozen=True)
class QualificationRequest:
    action: QualificationAction
    profile: str
    local: LocalAuthorityPaths | None = None
    descriptor: Path | None = None
    output_handoff: Path | None = None
    intent_s3_uri: str | None = None
    must_start_ready_s3_uri: str | None = None
    acquisition_s3_uri: str | None = None
    sky_bin: str | None = None


@dataclass(frozen=True)
class CacheSeedLocalAuthorityPaths:
    descriptor: Path
    approval: Path
    staged_ready: Path
    rehearsal_evidence: Path
    task: Path
    config: Path


@dataclass(frozen=True)
class CacheSeedRequest:
    action: CacheSeedAction
    profile: str
    local: CacheSeedLocalAuthorityPaths
    staged_readiness_version_id: str
    sky_bin: str
    waive_launch_claim_protection: bool = False


@dataclass(frozen=True)
class SubmissionServices:
    sts: Any
    s3: Any
    ec2: Any
    ssm: Any
    cloudformation: Any
    sky: Any
    clock: Callable[[], datetime]
    sleep: Callable[[float], None]
    claim_put_s3: Any | None = None


@dataclass(frozen=True)
class SubmissionOutcome:
    exit_code: int
    status: str
    detail: Mapping[str, object]


@dataclass(frozen=True)
class _JsonArtifact:
    key: str
    raw: bytes
    value: dict[str, object]

    @property
    def file_sha256(self) -> str:
        return _sha(self.raw)


@dataclass(frozen=True)
class _LocalAuthorities:
    descriptor: _JsonArtifact
    seed_descriptor: _JsonArtifact
    approval: _JsonArtifact
    staged: _JsonArtifact
    rehearsal: _JsonArtifact
    cache_acceptance: _JsonArtifact
    spend_snapshot: _JsonArtifact
    readiness: _JsonArtifact
    task: Path
    config: Path


@dataclass(frozen=True)
class _CacheSeedLocalAuthorities:
    descriptor: _JsonArtifact
    approval: _JsonArtifact
    staged: _JsonArtifact
    bundle_manifest: _JsonArtifact
    rehearsal: _JsonArtifact
    task: Path
    config: Path
    source_snapshot: WorkerStartV2SourceSnapshot
    config_raw: bytes


@dataclass(frozen=True)
class _CacheSeedClaimState:
    artifact: _JsonArtifact
    version_id: str


@dataclass(frozen=True)
class _SealedLaunchBundle:
    task_raw: bytes
    config_raw: bytes
    mount_sources: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True)
class _MaterializedSealedLaunch:
    root: Path
    repository_root: Path
    task_path: Path
    config_path: Path


@dataclass(frozen=True)
class _ControllerFacts:
    controller_instance_id: str
    controller_instance_type: str
    controller_profile_arn: str
    controller_cluster_name: str
    ssm_ping_status: str
    exact_name_history: list[dict[str, object]]
    detailed_history: list[dict[str, object]]
    active_exact_name_job_ids: list[int]
    active_tagged_p5_instance_ids: list[str]
    observed_at: str


@dataclass(frozen=True)
class _SelectedSubmission:
    descriptor: _JsonArtifact
    staged: _JsonArtifact
    rehearsal: _JsonArtifact
    cache_acceptance: _JsonArtifact
    spend_snapshot: _JsonArtifact
    readiness: _JsonArtifact
    intent: _JsonArtifact
    baseline: _JsonArtifact
    control_ready: _JsonArtifact
    acquisition: _JsonArtifact


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SubmissionIntegrationError(
            "submission value is not canonical finite JSON"
        ) from error


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX64 for character in value)
    )


def _whole_second(value: datetime | str, *, field: str) -> tuple[str, datetime]:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise SubmissionIntegrationError(
                f"{field} is not an ISO timestamp"
            ) from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise SubmissionIntegrationError(f"{field} is not a timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SubmissionIntegrationError(f"{field} must include a timezone")
    parsed = parsed.astimezone(UTC)
    if parsed.microsecond:
        raise SubmissionIntegrationError(f"{field} must use whole-second precision")
    canonical = parsed.isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value != canonical:
        raise SubmissionIntegrationError(f"{field} must use canonical UTC Z form")
    return canonical, parsed


def _controller_time(
    value: object,
    *,
    field: str,
    permit_none: bool = False,
    whole_second: bool = False,
) -> tuple[str | None, datetime | None]:
    """Parse exact Sky controller time, flooring only baseline projections."""

    if value is None and permit_none:
        return None, None
    if not isinstance(value, str):
        raise SubmissionIntegrationError(f"{field} is not an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SubmissionIntegrationError(f"{field} is not an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SubmissionIntegrationError(f"{field} must include a timezone")
    parsed = parsed.astimezone(UTC)
    if whole_second:
        parsed = parsed.replace(microsecond=0)
        return parsed.isoformat(timespec="seconds").replace("+00:00", "Z"), parsed
    canonical = parsed.isoformat().replace("+00:00", "Z")
    if value != canonical:
        raise SubmissionIntegrationError(f"{field} must use canonical UTC Z form")
    return canonical, parsed


def _now(services: SubmissionServices) -> datetime:
    _, value = _whole_second(services.clock(), field="current time")
    return value


def _require_submission_window(
    *,
    descriptor: Mapping[str, object],
    now: datetime,
) -> None:
    _, deadline = _whole_second(
        str(descriptor.get("must_start_by")),
        field="must_start_by",
    )
    if now >= deadline:
        raise SubmissionIntegrationError(
            "must_start_by has expired; build a new immutable descriptor"
        )
    if deadline - now > MAX_STATIC_WINDOW:
        raise SubmissionIntegrationError(
            "must_start_by exceeds the 12-hour submission window"
        )


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SubmissionUsageError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _source_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_source(path: Path, *, label: str) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise SubmissionIntegrationError(
            f"{label} cannot be authenticated without no-follow open support"
        )
    try:
        path_before = os.lstat(path)
    except OSError as error:
        raise SubmissionIntegrationError(f"{label} is missing or unreadable") from error
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_nlink != 1:
        raise SubmissionIntegrationError(
            f"{label} must be one regular non-symlink file with one link"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
        )
        descriptor_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_before.st_mode)
            or descriptor_before.st_nlink != 1
            or _source_file_identity(path_before)
            != _source_file_identity(descriptor_before)
        ):
            raise SubmissionIntegrationError(
                f"{label} path identity changed before source read"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
        try:
            path_after = os.lstat(path)
        except OSError as error:
            raise SubmissionIntegrationError(
                f"{label} path disappeared during source read"
            ) from error
        identity = _source_file_identity(descriptor_before)
        if (
            identity != _source_file_identity(descriptor_after)
            or identity != _source_file_identity(path_after)
            or not stat.S_ISREG(path_after.st_mode)
            or path_after.st_nlink != 1
        ):
            raise SubmissionIntegrationError(
                f"{label} path or metadata changed during source read"
            )
        raw = b"".join(chunks)
        if len(raw) != descriptor_after.st_size:
            raise SubmissionIntegrationError(f"{label} source size changed during read")
        return raw
    except SubmissionIntegrationError:
        raise
    except OSError as error:
        raise SubmissionIntegrationError(
            f"{label} stable no-follow source read failed"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_worker_start_v2_source_snapshot(
    *,
    task: Path,
) -> WorkerStartV2SourceSnapshot:
    source_paths = {
        "publisher_raw": (
            REPO_ROOT / "aws/glm52-gpu/skypilot/publish_worker_start_v2.py"
        ),
        "h100_qualification_worker_raw": (
            REPO_ROOT / "aws/glm52-gpu/skypilot/run_h100_qualification.sh"
        ),
        "h100_qualification_native_raw": (
            REPO_ROOT / "src/glm52_enforcement/glm52_h100_qualification.py"
        ),
        "h100_qualification_raw": (
            REPO_ROOT / "src/mlx_vq/quality/glm52_h100_qualification.py"
        ),
        "campaign_policy_native_raw": (
            REPO_ROOT / "src/glm52_enforcement/glm52_sky_campaign.py"
        ),
        "must_start_policy_native_raw": (
            REPO_ROOT / "src/glm52_enforcement/glm52_sky_must_start.py"
        ),
        "campaign_policy_raw": (REPO_ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py"),
        "must_start_policy_raw": (
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_must_start.py"
        ),
        "dynamic_policy_raw": (
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py"
        ),
        "worker_policy_raw": (
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py"
        ),
        "coordinator_raw": (
            REPO_ROOT / "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py"
        ),
        "bootstrap_raw": (REPO_ROOT / "aws/glm52-gpu/skypilot/bootstrap_campaign.sh"),
        "managed_entrypoint_raw": (
            REPO_ROOT / "aws/glm52-gpu/skypilot/run_managed_campaign.sh"
        ),
    }
    values = {
        field: _read_stable_source(path, label=f"worker-start-v2 {field}")
        for field, path in source_paths.items()
    }
    repository_task = _read_stable_source(
        REPO_ROOT / "aws/glm52-gpu/skypilot/glm52-campaign.yaml",
        label="worker-start-v2 repository Sky task",
    )
    submitted_task = _read_stable_source(
        task,
        label="worker-start-v2 submitted Sky task",
    )
    if submitted_task != repository_task:
        raise SubmissionIntegrationError(
            "submitted Sky task bytes differ from the fixed repository task"
        )
    return WorkerStartV2SourceSnapshot(
        publisher_raw=values["publisher_raw"],
        h100_qualification_worker_raw=values["h100_qualification_worker_raw"],
        h100_qualification_native_raw=values["h100_qualification_native_raw"],
        h100_qualification_raw=values["h100_qualification_raw"],
        campaign_policy_native_raw=values["campaign_policy_native_raw"],
        must_start_policy_native_raw=values["must_start_policy_native_raw"],
        campaign_policy_raw=values["campaign_policy_raw"],
        must_start_policy_raw=values["must_start_policy_raw"],
        dynamic_policy_raw=values["dynamic_policy_raw"],
        worker_policy_raw=values["worker_policy_raw"],
        coordinator_raw=values["coordinator_raw"],
        sky_task_raw=submitted_task,
        bootstrap_raw=values["bootstrap_raw"],
        managed_entrypoint_raw=values["managed_entrypoint_raw"],
    )


def _build_sealed_launch_bundle(
    *,
    source_snapshot: WorkerStartV2SourceSnapshot,
    config_raw: bytes,
) -> _SealedLaunchBundle:
    if (
        type(source_snapshot) is not WorkerStartV2SourceSnapshot
        or set(vars(source_snapshot)) != _SEALED_SOURCE_SNAPSHOT_FIELDS
    ):
        raise SubmissionIntegrationError(
            "sealed launch source snapshot inventory is not exact"
        )
    if type(config_raw) is not bytes:
        raise SubmissionIntegrationError(
            "sealed launch config is not immutable exact bytes"
        )
    for field in _SEALED_SOURCE_SNAPSHOT_FIELDS:
        if type(getattr(source_snapshot, field)) is not bytes:
            raise SubmissionIntegrationError(
                f"sealed launch source {field} is not immutable exact bytes"
            )
    bundle = _SealedLaunchBundle(
        task_raw=source_snapshot.sky_task_raw,
        config_raw=config_raw,
        mount_sources=tuple(
            (relative_path, getattr(source_snapshot, field))
            for relative_path, field in _SEALED_MOUNT_SOURCES
        ),
    )
    _validate_sealed_launch_bundle(bundle)
    return bundle


def _validate_sealed_launch_bundle(bundle: object) -> _SealedLaunchBundle:
    if type(bundle) is not _SealedLaunchBundle or set(vars(bundle)) != {
        "task_raw",
        "config_raw",
        "mount_sources",
    }:
        raise SubmissionIntegrationError("sealed launch bundle inventory is not exact")
    assert isinstance(bundle, _SealedLaunchBundle)
    if type(bundle.task_raw) is not bytes or type(bundle.config_raw) is not bytes:
        raise SubmissionIntegrationError(
            "sealed launch task or config is not immutable exact bytes"
        )
    if type(bundle.mount_sources) is not tuple:
        raise SubmissionIntegrationError("sealed launch mount inventory is not a tuple")
    expected_paths = tuple(path for path, _field in _SEALED_MOUNT_SOURCES)
    actual_paths: list[str] = []
    for member in bundle.mount_sources:
        if (
            type(member) is not tuple
            or len(member) != 2
            or type(member[0]) is not str
            or type(member[1]) is not bytes
        ):
            raise SubmissionIntegrationError("sealed launch mount member is not exact")
        actual_paths.append(member[0])
    if tuple(actual_paths) != expected_paths:
        raise SubmissionIntegrationError(
            "sealed launch mount path inventory or order drifted"
        )
    return bundle


def _private_child(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        type(relative_path) is not str
        or pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise SubmissionIntegrationError("sealed launch private path is unsafe")
    return root.joinpath(*pure.parts)


def _sealed_file_specs(
    *,
    root: Path,
    bundle: _SealedLaunchBundle,
) -> tuple[tuple[str, Path, bytes], ...]:
    repository_root = root / "repo"
    return (
        (
            f"repo/{_SEALED_TASK_RELATIVE_PATH}",
            _private_child(repository_root, _SEALED_TASK_RELATIVE_PATH),
            bundle.task_raw,
        ),
        (
            _SEALED_CONFIG_RELATIVE_PATH,
            _private_child(root, _SEALED_CONFIG_RELATIVE_PATH),
            bundle.config_raw,
        ),
        *tuple(
            (
                f"repo/{relative_path}",
                _private_child(repository_root, relative_path),
                raw,
            )
            for relative_path, raw in bundle.mount_sources
        ),
    )


def _sealed_directory_relatives(
    file_specs: tuple[tuple[str, Path, bytes], ...],
) -> tuple[str, ...]:
    relatives: set[str] = set()
    for relative_path, _path, _raw in file_specs:
        parent = PurePosixPath(relative_path).parent
        while parent != PurePosixPath("."):
            relatives.add(parent.as_posix())
            parent = parent.parent
    return tuple(
        sorted(
            relatives,
            key=lambda value: (len(PurePosixPath(value).parts), value),
        )
    )


def _write_private_file(path: Path, raw: bytes) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise SubmissionIntegrationError(
            "sealed launch files require no-follow create support"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | no_follow
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short sealed launch write")
            written += count
        os.fsync(descriptor)
        closing_descriptor = descriptor
        descriptor = None
        os.close(closing_descriptor)
        os.chmod(path, 0o400, follow_symlinks=False)
    except (OSError, NotImplementedError) as error:
        raise SubmissionIntegrationError(
            f"sealed launch file creation failed: {path.name}"
        ) from error
    finally:
        if descriptor is not None:
            closing_descriptor = descriptor
            descriptor = None
            try:
                os.close(closing_descriptor)
            except OSError as error:
                raise SubmissionIntegrationError(
                    f"sealed launch file close failed: {path.name}"
                ) from error


def _fsync_private_directory(path: Path) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise SubmissionIntegrationError(
            "sealed launch directories require no-follow directory support"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | no_follow | directory_flag | getattr(os, "O_CLOEXEC", 0),
        )
        os.fsync(descriptor)
    except OSError as error:
        raise SubmissionIntegrationError(
            f"sealed launch directory sync failed: {path.name}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_materialized_coordinates(
    materialized: object,
) -> _MaterializedSealedLaunch:
    if type(materialized) is not _MaterializedSealedLaunch or set(
        vars(materialized)
    ) != {"root", "repository_root", "task_path", "config_path"}:
        raise SubmissionIntegrationError(
            "materialized sealed launch coordinates are not exact"
        )
    assert isinstance(materialized, _MaterializedSealedLaunch)
    expected_repository = materialized.root / "repo"
    if (
        materialized.repository_root != expected_repository
        or materialized.task_path
        != _private_child(expected_repository, _SEALED_TASK_RELATIVE_PATH)
        or materialized.config_path
        != _private_child(materialized.root, _SEALED_CONFIG_RELATIVE_PATH)
    ):
        raise SubmissionIntegrationError(
            "materialized sealed launch coordinates drifted"
        )
    return materialized


def _scan_private_tree(
    *,
    root: Path,
    expected_files: frozenset[str],
    expected_directories: frozenset[str],
) -> tuple[set[str], set[str]]:
    actual_files: set[str] = set()
    actual_directories: set[str] = set()

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        try:
            entries = tuple(os.scandir(directory))
        except OSError as error:
            raise SubmissionIntegrationError(
                "sealed launch tree cannot be enumerated"
            ) from error
        for entry in entries:
            relative = (prefix / entry.name).as_posix()
            try:
                identity = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise SubmissionIntegrationError(
                    "sealed launch member cannot be inspected"
                ) from error
            mode = stat.S_IMODE(identity.st_mode)
            if stat.S_ISDIR(identity.st_mode):
                if relative not in expected_directories or mode != 0o700:
                    raise SubmissionIntegrationError(
                        "sealed launch directory is unexpected or unsafe"
                    )
                actual_directories.add(relative)
                visit(Path(entry.path), PurePosixPath(relative))
            elif stat.S_ISREG(identity.st_mode):
                if (
                    relative not in expected_files
                    or mode != 0o400
                    or identity.st_nlink != 1
                ):
                    raise SubmissionIntegrationError(
                        "sealed launch file is unexpected or unsafe"
                    )
                actual_files.add(relative)
            else:
                raise SubmissionIntegrationError(
                    "sealed launch tree contains a non-regular member"
                )

    visit(root, PurePosixPath("."))
    return actual_files, actual_directories


def _verify_materialized_launch(
    *,
    materialized: _MaterializedSealedLaunch,
    bundle: _SealedLaunchBundle,
) -> None:
    exact_bundle = _validate_sealed_launch_bundle(bundle)
    exact_materialized = _validate_materialized_coordinates(materialized)
    try:
        root_identity = os.lstat(exact_materialized.root)
    except OSError as error:
        raise SubmissionIntegrationError(
            "sealed launch root is missing or unreadable"
        ) from error
    if (
        not stat.S_ISDIR(root_identity.st_mode)
        or stat.S_IMODE(root_identity.st_mode) != 0o700
    ):
        raise SubmissionIntegrationError("sealed launch root permissions are unsafe")
    file_specs = _sealed_file_specs(
        root=exact_materialized.root,
        bundle=exact_bundle,
    )
    expected_files = frozenset(relative for relative, _path, _raw in file_specs)
    expected_directories = frozenset(_sealed_directory_relatives(file_specs))
    actual_files, actual_directories = _scan_private_tree(
        root=exact_materialized.root,
        expected_files=expected_files,
        expected_directories=expected_directories,
    )
    if actual_files != expected_files or actual_directories != expected_directories:
        raise SubmissionIntegrationError(
            "sealed launch tree inventory is incomplete or unexpected"
        )
    for relative_path, path, expected_raw in file_specs:
        actual_raw = _read_stable_source(
            path,
            label=f"sealed launch {relative_path}",
        )
        if actual_raw != expected_raw:
            raise SubmissionIntegrationError(
                f"sealed launch bytes drifted: {relative_path}"
            )


def _cleanup_sealed_launch(temporary: object) -> None:
    try:
        temporary.cleanup()
    except Exception:  # noqa: BLE001
        # The selector already forbids a second launch. Cleanup is best-effort
        # and must not replace the Sky outcome or bypass exact reconciliation.
        try:
            name = temporary.name
            root = Path(name)
            if (
                type(name) is str
                and root.is_absolute()
                and root.name.startswith("glm52-sky-sealed-launch-")
                and root.parent.resolve() == Path(tempfile.gettempdir()).resolve()
            ):
                shutil.rmtree(root)
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def _materialize_sealed_launch(
    bundle: _SealedLaunchBundle,
) -> Iterator[_MaterializedSealedLaunch]:
    exact_bundle = _validate_sealed_launch_bundle(bundle)
    temporary = tempfile.TemporaryDirectory(prefix="glm52-sky-sealed-launch-")
    try:
        root = Path(temporary.name)
        os.chmod(root, 0o700)
        file_specs = _sealed_file_specs(root=root, bundle=exact_bundle)
        directory_relatives = _sealed_directory_relatives(file_specs)
        for relative_path in directory_relatives:
            directory = _private_child(root, relative_path)
            try:
                os.mkdir(directory, 0o700)
                os.chmod(directory, 0o700)
            except OSError as error:
                raise SubmissionIntegrationError(
                    f"sealed launch directory creation failed: {relative_path}"
                ) from error
        for _relative_path, path, raw in file_specs:
            _write_private_file(path, raw)
        for relative_path in reversed(directory_relatives):
            _fsync_private_directory(_private_child(root, relative_path))
        _fsync_private_directory(root)
        materialized = _MaterializedSealedLaunch(
            root=root,
            repository_root=root / "repo",
            task_path=_private_child(root / "repo", _SEALED_TASK_RELATIVE_PATH),
            config_path=_private_child(root, _SEALED_CONFIG_RELATIVE_PATH),
        )
        _verify_materialized_launch(
            materialized=materialized,
            bundle=exact_bundle,
        )
        yield materialized
    finally:
        _cleanup_sealed_launch(temporary)


def _decode_canonical_file(path: Path, *, label: str) -> dict[str, object]:
    raw = _read_regular(path, label=label)
    try:
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SubmissionUsageError(f"{label} is not finite JSON") from error
    if not isinstance(value, dict):
        raise SubmissionUsageError(f"{label} must contain one JSON object")
    if raw != _canonical(value) + b"\n":
        raise SubmissionUsageError(
            f"{label} must be compact sorted finite ASCII JSON plus one LF"
        )
    return dict(value)


def _artifact_from_path(
    path: Path,
    *,
    key: str,
    label: str,
) -> _JsonArtifact:
    value = _decode_canonical_file(path, label=label)
    return _JsonArtifact(key=key, raw=path.read_bytes(), value=value)


def _decode_artifact_raw(
    *,
    key: str,
    raw: bytes,
    label: str,
    newline: bool = True,
) -> _JsonArtifact:
    try:
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SubmissionIntegrationError(f"{label} is not finite JSON") from error
    if not isinstance(value, dict):
        raise SubmissionIntegrationError(f"{label} must contain one JSON object")
    expected = _canonical(value) + (b"\n" if newline else b"")
    if raw != expected:
        suffix = "plus one LF" if newline else "without an LF"
        raise SubmissionIntegrationError(
            f"{label} must use exact compact sorted finite ASCII JSON {suffix}"
        )
    return _JsonArtifact(key=key, raw=raw, value=dict(value))


def _fetch_artifact(
    services: SubmissionServices,
    *,
    bucket: str,
    key: str,
    label: str,
    newline: bool = True,
) -> _JsonArtifact:
    raw = _get_exact(services.s3, bucket=bucket, key=key)
    if raw is None:
        raise SubmissionIntegrationError(f"{label} is absent at exact key {key}")
    return _decode_artifact_raw(
        key=key,
        raw=raw,
        label=label,
        newline=newline,
    )


def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def _parse_s3_uri(uri: str, *, label: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise SubmissionUsageError(f"{label} must be an s3:// URI")
    remainder = uri.removeprefix("s3://")
    bucket, separator, key = remainder.partition("/")
    if (
        not separator
        or not bucket
        or not key
        or key.startswith("/")
        or ".." in key.split("/")
    ):
        raise SubmissionUsageError(f"{label} is not a safe exact S3 URI")
    return bucket, key


def _aws_error(error: BaseException) -> tuple[str, int | None]:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return "", None
    error_value = response.get("Error")
    metadata = response.get("ResponseMetadata")
    code = str(error_value.get("Code", "")) if isinstance(error_value, Mapping) else ""
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    return code, status if type(status) is int else None


def _body_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    reader = getattr(value, "read", None)
    if callable(reader):
        raw = reader()
        if isinstance(raw, bytes):
            return raw
    raise SubmissionIntegrationError("S3 response body is not readable bytes")


def _get_exact(
    s3: Any,
    *,
    bucket: str,
    key: str,
    version_id: str | None = None,
) -> bytes | None:
    arguments = {"Bucket": bucket, "Key": key}
    if version_id is not None:
        if not isinstance(version_id, str) or version_id in {"", "null"}:
            raise SubmissionIntegrationError(
                "S3 exact GET requires a non-null VersionId"
            )
        arguments["VersionId"] = version_id
    try:
        response = s3.get_object(**arguments)
    except Exception as error:
        code, status = _aws_error(error)
        if status == 404 and code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise SubmissionIntegrationError(
            f"S3 exact GET failed for s3://{bucket}/{key}: {code or error}"
        ) from error
    if not isinstance(response, Mapping):
        raise SubmissionIntegrationError("S3 exact GET returned a non-object")
    if version_id is not None:
        returned_version = response.get("VersionId")
        if (
            not isinstance(returned_version, str)
            or returned_version in {"", "null"}
            or returned_version != version_id
        ):
            raise SubmissionIntegrationError(
                "S3 exact GET returned a missing or unequal VersionId"
            )
    metadata = response.get("ResponseMetadata")
    if isinstance(metadata, Mapping) and metadata.get("HTTPStatusCode") not in (
        None,
        200,
    ):
        raise SubmissionIntegrationError("S3 exact GET returned a non-200 status")
    raw = _body_bytes(response.get("Body"))
    length = response.get("ContentLength")
    if type(length) is not int or length != len(raw):
        raise SubmissionIntegrationError("S3 exact GET content length mismatch")
    checksum = response.get("ChecksumSHA256")
    if checksum is not None:
        expected = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
        if checksum != expected:
            raise SubmissionIntegrationError("S3 exact GET checksum mismatch")
    return raw


def _list_complete(
    s3: Any,
    *,
    bucket: str,
    prefix: str,
) -> tuple[str, ...]:
    token: str | None = None
    keys: list[str] = []
    while True:
        arguments: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if token is not None:
            arguments["ContinuationToken"] = token
        response = s3.list_objects_v2(**arguments)
        if not isinstance(response, Mapping):
            raise SubmissionIntegrationError("S3 LIST returned a non-object")
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, Mapping) and metadata.get("HTTPStatusCode") not in (
            None,
            200,
        ):
            raise SubmissionIntegrationError("S3 LIST returned a non-200 status")
        contents = response.get("Contents", [])
        if not isinstance(contents, list):
            raise SubmissionIntegrationError("S3 LIST contents are malformed")
        key_count = response.get("KeyCount")
        if type(key_count) is not int or key_count != len(contents):
            raise SubmissionIntegrationError("S3 LIST key count is malformed")
        for item in contents:
            if not isinstance(item, Mapping) or not isinstance(item.get("Key"), str):
                raise SubmissionIntegrationError("S3 LIST member is malformed")
            key = str(item["Key"])
            if not key.startswith(prefix):
                raise SubmissionIntegrationError(
                    "S3 LIST returned a foreign prefix member"
                )
            keys.append(key)
        truncated = response.get("IsTruncated")
        if type(truncated) is not bool:
            raise SubmissionIntegrationError("S3 LIST truncation state is missing")
        if not truncated:
            break
        next_token = response.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token or next_token == token:
            raise SubmissionIntegrationError("S3 LIST pagination token is invalid")
        token = next_token
    if keys != sorted(set(keys)):
        raise SubmissionIntegrationError(
            "S3 LIST members are duplicate or not canonically sorted"
        )
    return tuple(keys)


def _body_digest(value: Mapping[str, object]) -> str:
    digest_fields = {
        "glm52_qualification_cache_seed_accepted_v1": "acceptance_body_sha256",
        "glm52_gpu_spend_snapshot_v1": "snapshot_body_sha256",
        "glm52_staged_control_plane_rehearsal_v2": "rehearsal_body_sha256",
        "glm52_qualification_submission_ready_v1": "readiness_body_sha256",
        "glm52_sky_submission_intent_v2": "intent_body_sha256",
        "glm52_controller_baseline_v1": "baseline_body_sha256",
        "glm52_must_start_control_plane_ready_v1": ("control_plane_ready_body_sha256"),
        "glm52_sky_submission_acquired_v1": "acquisition_body_sha256",
        "glm52_sky_must_start_controller_observation_v1": ("observation_body_sha256"),
        "glm52_sky_must_start_job_binding_v1": "job_binding_body_sha256",
        "glm52_sky_must_start_job_binding_v2": "job_binding_body_sha256",
        "glm52_sky_submission_accepted_v2": "accepted_body_sha256",
    }
    field = digest_fields.get(str(value.get("record_type")))
    if field is None or not _is_sha(value.get(field)):
        raise SubmissionIntegrationError("authority self-digest field is ambiguous")
    return str(value[field])


def _put_immutable(
    services: SubmissionServices,
    *,
    bucket: str,
    artifact: _JsonArtifact,
    run_id: str,
    newline: bool = True,
) -> bool:
    expected = _canonical(artifact.value) + (b"\n" if newline else b"")
    if artifact.raw != expected:
        raise SubmissionIntegrationError(
            f"local authority bytes are noncanonical for {artifact.key}"
        )
    existing = _get_exact(
        services.s3,
        bucket=bucket,
        key=artifact.key,
    )
    if existing is not None:
        if existing != artifact.raw:
            raise SubmissionIntegrationError(
                f"immutable S3 winner is incompatible: {artifact.key}"
            )
        return False
    checksum = base64.b64encode(hashlib.sha256(artifact.raw).digest()).decode("ascii")
    try:
        response = services.s3.put_object(
            Bucket=bucket,
            Key=artifact.key,
            Body=artifact.raw,
            IfNoneMatch="*",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum,
            ContentType="application/json",
            Metadata={
                "glm52-run-id": run_id,
                "glm52-body-sha256": _body_digest(artifact.value),
            },
        )
    except Exception as error:
        code, status = _aws_error(error)
        if not (status == 412 and code in {"412", "PreconditionFailed"}):
            winner = _get_exact(
                services.s3,
                bucket=bucket,
                key=artifact.key,
            )
            if winner == artifact.raw:
                return False
            raise SubmissionIntegrationError(
                f"conditional immutable put failed for {artifact.key}: {code or error}"
            ) from error
        winner = _get_exact(
            services.s3,
            bucket=bucket,
            key=artifact.key,
        )
        if winner != artifact.raw:
            raise SubmissionIntegrationError(
                f"conditional immutable put raced with foreign {artifact.key}"
            ) from error
        return False
    if not isinstance(response, Mapping):
        raise SubmissionIntegrationError(
            "conditional immutable put response is invalid"
        )
    metadata = response.get("ResponseMetadata")
    if not isinstance(metadata, Mapping) or metadata.get("HTTPStatusCode") not in (
        200,
        201,
    ):
        raise SubmissionIntegrationError(
            "conditional immutable put did not return service success"
        )
    winner = _get_exact(
        services.s3,
        bucket=bucket,
        key=artifact.key,
    )
    if winner != artifact.raw:
        raise SubmissionIntegrationError(
            f"immutable S3 winner failed post-put authentication: {artifact.key}"
        )
    return True


def _guard_identity(
    request: QualificationRequest | CacheSeedRequest,
    services: SubmissionServices,
) -> dict[str, object]:
    if request.profile != EXACT_PROFILE:
        raise SubmissionUsageError(f"--profile must be exactly {EXACT_PROFILE}")
    identity = services.sts.get_caller_identity()
    try:
        require_approved_aws_identity(identity)
    except (TypeError, ValueError) as error:
        raise SubmissionIntegrationError(str(error)) from error
    if not isinstance(identity, Mapping):
        raise SubmissionIntegrationError("AWS caller identity is malformed")
    return dict(identity)


def _safe_s3_key(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or value.endswith("/")
        or any(not 0x21 <= ord(character) <= 0x7E for character in value)
        or "\\" in value
        or any(character in value for character in "*?[]")
    ):
        raise SubmissionIntegrationError(f"{field} is not a safe S3 key")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise SubmissionIntegrationError(f"{field} is not a safe S3 key")
    return value


def _version_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value in {"", "null"}:
        raise SubmissionIntegrationError(f"{field} is not a valid S3 VersionId")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha(value):
        raise SubmissionIntegrationError(f"{field} is not a canonical SHA-256")
    return str(value)


def _validate_remote_staged_readiness(
    staged: _JsonArtifact,
    *,
    descriptor: _JsonArtifact,
) -> tuple[dict[str, object], dict[str, str]]:
    value = staged.value
    if (
        frozenset(value) != _STAGED_READY_V2_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 2
        or value.get("record_type") != "glm52_staged_control_plane_ready_v2"
    ):
        raise SubmissionIntegrationError(
            "remote staged readiness is not exact authorizing v2"
        )
    body = dict(value)
    ready_body_sha = _require_sha256(
        body.pop("ready_body_sha256"),
        field="remote staged readiness body SHA-256",
    )
    if _sha(_canonical(body)) != ready_body_sha:
        raise SubmissionIntegrationError(
            "remote staged readiness body SHA-256 mismatch"
        )
    _whole_second(value.get("staged_at"), field="remote staged readiness staged_at")
    descriptor_key = _safe_s3_key(
        value.get("descriptor_key"),
        field="remote staged readiness descriptor key",
    )
    manifest_key = _safe_s3_key(
        value.get("bundle_manifest_key"),
        field="remote staged readiness manifest key",
    )
    audit_key = _safe_s3_key(
        value.get("artifact_audit_key"),
        field="remote staged readiness audit key",
    )
    manifest_file_sha = _require_sha256(
        value.get("bundle_manifest_file_sha256"),
        field="remote staged readiness manifest file SHA-256",
    )
    manifest_body_sha = _require_sha256(
        value.get("bundle_manifest_body_sha256"),
        field="remote staged readiness manifest body SHA-256",
    )
    audit_sha = _require_sha256(
        value.get("artifact_audit_sha256"),
        field="remote staged readiness audit SHA-256",
    )
    for field in (
        "descriptor_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
    ):
        _require_sha256(value.get(field), field=f"remote staged readiness {field}")
    expected_bindings = {
        "run_id": descriptor.value["run_id"],
        "descriptor_key": descriptor.key,
        "descriptor_sha256": descriptor.file_sha256,
        "descriptor_body_sha256": descriptor.value["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor.value["campaign_identity_sha256"],
    }
    for field, expected in expected_bindings.items():
        if value.get(field) != expected:
            raise SubmissionIntegrationError(
                f"remote staged readiness {field} binding mismatch"
            )
    if descriptor_key != descriptor.key:
        raise SubmissionIntegrationError(
            "remote staged readiness descriptor key mismatch"
        )
    expected_audit_key = (
        f"campaigns/{descriptor.value['run_id']}/audits/"
        f"artifact-audit-{audit_sha}.json"
    )
    if audit_key != expected_audit_key:
        raise SubmissionIntegrationError(
            "remote staged readiness audit key binding mismatch"
        )
    versions = value.get("staged_object_version_ids")
    if not isinstance(versions, dict) or frozenset(versions) != _STAGED_VERSION_ROLES:
        raise SubmissionIntegrationError(
            "remote staged readiness VersionId role map mismatch"
        )
    validated_versions = {
        role: _version_id(
            versions[role],
            field=f"remote staged readiness {role} VersionId",
        )
        for role in _STAGED_VERSION_ROLES
    }
    _version_id(
        value.get("bundle_manifest_version_id"),
        field="remote staged readiness manifest VersionId",
    )
    # Values are used again by the manifest validator; validate them here so
    # no partially authenticated authority reaches a versioned read.
    if manifest_file_sha != value["bundle_manifest_file_sha256"] or (
        manifest_body_sha != value["bundle_manifest_body_sha256"]
    ):
        raise SubmissionIntegrationError(
            "remote staged readiness manifest digest mismatch"
        )
    if manifest_key != value["bundle_manifest_key"]:
        raise SubmissionIntegrationError(
            "remote staged readiness manifest key mismatch"
        )
    return dict(value), validated_versions


def _validate_remote_bundle_manifest(
    raw: bytes,
    *,
    ready: Mapping[str, object],
    descriptor: _JsonArtifact,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    artifact = _decode_artifact_raw(
        key=str(ready["bundle_manifest_key"]),
        raw=raw,
        label="remote bundle manifest",
    )
    value = artifact.value
    if (
        frozenset(value) != _BUNDLE_MANIFEST_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("record_type") != "glm52_sky_campaign_bundle_v1"
    ):
        raise SubmissionIntegrationError("remote bundle manifest schema mismatch")
    body = dict(value)
    body_sha = _require_sha256(
        body.pop("bundle_manifest_body_sha256"),
        field="remote bundle manifest body SHA-256",
    )
    if _sha(_canonical(body)) != body_sha:
        raise SubmissionIntegrationError(
            "remote bundle manifest body SHA-256 mismatch"
        )
    if (
        artifact.file_sha256 != ready["bundle_manifest_file_sha256"]
        or body_sha != ready["bundle_manifest_body_sha256"]
        or value.get("run_id") != descriptor.value["run_id"]
        or value.get("bucket") != descriptor.value["bucket"]
        or value.get("descriptor_body_sha256")
        != descriptor.value["descriptor_body_sha256"]
    ):
        raise SubmissionIntegrationError("remote bundle manifest authority mismatch")
    descriptor_prefix = (
        f"campaigns/{descriptor.value['run_id']}/submissions/"
    )
    descriptor_suffix = "/campaign-descriptor-v2.json"
    if not descriptor.key.startswith(descriptor_prefix) or not descriptor.key.endswith(
        descriptor_suffix
    ):
        raise SubmissionIntegrationError(
            "remote bundle manifest descriptor coordinate is invalid"
        )
    submission_id = descriptor.key[
        len(descriptor_prefix) : -len(descriptor_suffix)
    ]
    if not submission_id or "/" in submission_id or submission_id in {".", ".."}:
        raise SubmissionIntegrationError(
            "remote bundle manifest submission ID is unsafe"
        )
    expected_key = (
        f"{descriptor_prefix}{submission_id}/bundle-manifests/"
        f"{body_sha}/bundle-manifest-v1.json"
    )
    if ready["bundle_manifest_key"] != expected_key:
        raise SubmissionIntegrationError(
            "remote bundle manifest key derivation mismatch"
        )
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(_BUNDLE_ROLES):
        raise SubmissionIntegrationError("remote bundle manifest roles are incomplete")
    if any(
        not isinstance(item, dict) or frozenset(item) != _BUNDLE_FILE_FIELDS
        for item in files
    ):
        raise SubmissionIntegrationError("remote bundle manifest file schema mismatch")
    copied = [dict(item) for item in files]
    roles = [item.get("role") for item in copied]
    if frozenset(roles) != _BUNDLE_ROLES or len(set(roles)) != len(roles):
        raise SubmissionIntegrationError(
            "remote bundle manifest roles are duplicated or foreign"
        )
    names: list[str] = []
    keys: list[str] = []
    orders: list[int] = []
    for item in copied:
        name = item.get("local_name")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or PurePosixPath(name).name != name
            or "\\" in name
        ):
            raise SubmissionIntegrationError(
                "remote bundle manifest local name is unsafe"
            )
        try:
            name.encode("ascii")
        except UnicodeEncodeError as error:
            raise SubmissionIntegrationError(
                "remote bundle manifest local name is not ASCII"
            ) from error
        names.append(name)
        keys.append(
            _safe_s3_key(
                item.get("key"),
                field="remote bundle manifest object key",
            )
        )
        size = item.get("size")
        if type(size) is not int or size < 0:
            raise SubmissionIntegrationError(
                "remote bundle manifest object size is invalid"
            )
        _require_sha256(
            item.get("sha256"),
            field="remote bundle manifest object SHA-256",
        )
        order = item.get("stage_order")
        if type(order) is not int or order < 0:
            raise SubmissionIntegrationError(
                "remote bundle manifest stage order is invalid"
            )
        orders.append(order)
    if (
        len(names) != len(set(names))
        or len(keys) != len(set(keys))
        or len(orders) != len(set(orders))
    ):
        raise SubmissionIntegrationError(
            "remote bundle manifest coordinates are duplicated"
        )
    ordered = tuple(sorted(copied, key=lambda item: int(item["stage_order"])))
    if ordered[-1]["role"] != "descriptor":
        raise SubmissionIntegrationError(
            "remote bundle manifest descriptor is not staged last"
        )
    descriptor_items = [item for item in ordered if item["role"] == "descriptor"]
    if len(descriptor_items) != 1:
        raise SubmissionIntegrationError(
            "remote bundle manifest descriptor role mismatch"
        )
    descriptor_item = descriptor_items[0]
    if (
        descriptor_item["key"] != descriptor.key
        or descriptor_item["size"] != len(descriptor.raw)
        or descriptor_item["sha256"] != descriptor.file_sha256
    ):
        raise SubmissionIntegrationError(
            "remote bundle manifest descriptor binding mismatch"
        )
    return dict(value), ordered


def _validate_remote_audit(
    raw: bytes,
    *,
    ready: Mapping[str, object],
    descriptor: _JsonArtifact,
    inventory: Mapping[str, object],
) -> None:
    artifact = _decode_artifact_raw(
        key=str(ready["artifact_audit_key"]),
        raw=raw,
        label="remote artifact audit",
    )
    value = artifact.value
    if (
        frozenset(value) != _ARTIFACT_AUDIT_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("record_type") != "glm52_s3_artifact_audit_v1"
        or value.get("audit_pass") is not True
    ):
        raise SubmissionIntegrationError("remote artifact audit schema mismatch")
    if artifact.file_sha256 != ready["artifact_audit_sha256"]:
        raise SubmissionIntegrationError("remote artifact audit identity mismatch")
    if (
        value.get("run_id") != descriptor.value["run_id"]
        or value.get("bucket") != descriptor.value["bucket"]
        or value.get("inventory_body_sha256")
        != inventory.get("inventory_body_sha256")
    ):
        raise SubmissionIntegrationError("remote artifact audit binding mismatch")
    objects = inventory.get("objects")
    if not isinstance(objects, list):
        raise SubmissionIntegrationError(
            "remote artifact inventory object list is malformed"
        )
    expected = {
        "object_count": len(objects),
        "object_bytes": sum(int(item["size"]) for item in objects),
        "safetensors_object_count": sum(
            item["safetensors"] is True for item in objects
        ),
    }
    for field in (
        "object_count",
        "object_bytes",
        "safetensors_object_count",
        "safetensors_tensor_count",
    ):
        if type(value.get(field)) is not int or int(value[field]) < 0:
            raise SubmissionIntegrationError(
                f"remote artifact audit {field} is invalid"
            )
    if any(value[field] != expected_value for field, expected_value in expected.items()):
        raise SubmissionIntegrationError(
            "remote artifact audit count contract mismatch"
        )
    tensor_count = int(value["safetensors_tensor_count"])
    safetensors_count = expected["safetensors_object_count"]
    if (safetensors_count == 0 and tensor_count != 0) or (
        safetensors_count > 0 and tensor_count < safetensors_count
    ):
        raise SubmissionIntegrationError(
            "remote artifact audit tensor count is inconsistent"
        )


def _authenticate_remote_staged_chain(
    services: SubmissionServices,
    *,
    descriptor: _JsonArtifact,
    staged: _JsonArtifact,
    rehearsal: _JsonArtifact,
    readiness: _JsonArtifact,
    approval: _JsonArtifact | None = None,
) -> None:
    bucket = str(descriptor.value["bucket"])
    current_ready = _get_exact(
        services.s3,
        bucket=bucket,
        key=staged.key,
    )
    if current_ready is None or current_ready != staged.raw:
        raise SubmissionIntegrationError(
            "remote staged readiness differs from the selected authority"
        )
    ready, versions = _validate_remote_staged_readiness(
        staged,
        descriptor=descriptor,
    )
    manifest_raw = _get_exact(
        services.s3,
        bucket=bucket,
        key=str(ready["bundle_manifest_key"]),
        version_id=str(ready["bundle_manifest_version_id"]),
    )
    if manifest_raw is None:
        raise SubmissionIntegrationError(
            "remote bundle manifest pinned version is unavailable"
        )
    _manifest, files = _validate_remote_bundle_manifest(
        manifest_raw,
        ready=ready,
        descriptor=descriptor,
    )
    role_raw: dict[str, bytes] = {}
    role_items: dict[str, dict[str, object]] = {}
    for item in files:
        role = str(item["role"])
        raw = _get_exact(
            services.s3,
            bucket=bucket,
            key=str(item["key"]),
            version_id=versions[role],
        )
        if (
            raw is None
            or len(raw) != int(item["size"])
            or _sha(raw) != item["sha256"]
        ):
            raise SubmissionIntegrationError(
                f"remote staged object exact-version drift: {role}"
            )
        role_raw[role] = raw
        role_items[role] = item
    if role_raw["descriptor"] != descriptor.raw:
        raise SubmissionIntegrationError(
            "remote staged descriptor differs from the selected descriptor"
        )
    approval_item = role_items["approval"]
    if (
        approval_item["key"] != descriptor.value["approval_key"]
        or approval_item["sha256"] != descriptor.value["approval_sha256"]
        or _sha(role_raw["approval"]) != descriptor.value["approval_sha256"]
        or (approval is not None and role_raw["approval"] != approval.raw)
    ):
        raise SubmissionIntegrationError(
            "remote staged approval differs from the selected approval"
        )
    repo_item = role_items["repository_tar"]
    if (
        repo_item["key"] != descriptor.value["repo_tar_key"]
        or repo_item["sha256"] != descriptor.value["repo_tar_sha256"]
    ):
        raise SubmissionIntegrationError(
            "remote staged repository differs from descriptor pins"
        )
    descriptor_artifacts = descriptor.value.get("artifacts")
    if not isinstance(descriptor_artifacts, Mapping):
        raise SubmissionIntegrationError("descriptor artifact pins are malformed")
    training_item = role_items["training_config"]
    if (
        training_item["key"] != descriptor_artifacts["training_config_key"]
        or training_item["sha256"]
        != descriptor_artifacts["training_config_sha256"]
    ):
        raise SubmissionIntegrationError(
            "remote staged training configuration differs from descriptor pins"
        )
    inventory_item = role_items["artifact_inventory"]
    if (
        inventory_item["key"] != descriptor_artifacts["artifact_inventory_key"]
        or inventory_item["sha256"]
        != descriptor_artifacts["artifact_inventory_sha256"]
    ):
        raise SubmissionIntegrationError(
            "remote staged inventory differs from descriptor pins"
        )
    inventory_artifact = _decode_artifact_raw(
        key=str(inventory_item["key"]),
        raw=role_raw["artifact_inventory"],
        label="remote artifact inventory",
    )
    try:
        inventory = validate_s3_artifact_inventory(inventory_artifact.value)
    except ValueError as error:
        raise SubmissionIntegrationError(
            f"remote artifact inventory is invalid: {error}"
        ) from error
    if (
        inventory.get("run_id") != descriptor.value["run_id"]
        or inventory.get("bucket") != bucket
    ):
        raise SubmissionIntegrationError("remote artifact inventory is foreign")
    audit_raw = _get_exact(
        services.s3,
        bucket=bucket,
        key=str(ready["artifact_audit_key"]),
        version_id=versions["artifact_audit"],
    )
    if audit_raw is None:
        raise SubmissionIntegrationError(
            "remote artifact audit pinned version is unavailable"
        )
    _validate_remote_audit(
        audit_raw,
        ready=ready,
        descriptor=descriptor,
        inventory=inventory,
    )
    rehearsal_bindings = {
        "staged_readiness_key": staged.key,
        "staged_readiness_file_sha256": staged.file_sha256,
        "staged_readiness_body_sha256": ready["ready_body_sha256"],
        "artifact_inventory_key": inventory_item["key"],
        "artifact_inventory_file_sha256": inventory_item["sha256"],
        "artifact_audit_key": ready["artifact_audit_key"],
        "artifact_audit_file_sha256": ready["artifact_audit_sha256"],
    }
    for field, expected in rehearsal_bindings.items():
        if rehearsal.value.get(field) != expected:
            raise SubmissionIntegrationError(
                f"rehearsal remote-chain {field} binding mismatch"
            )
    qualification_bindings = {
        "staged_readiness_key": staged.key,
        "staged_readiness_file_sha256": staged.file_sha256,
        "staged_readiness_body_sha256": ready["ready_body_sha256"],
        "rehearsal_evidence_key": rehearsal.key,
        "rehearsal_evidence_sha256": rehearsal.file_sha256,
        "rehearsal_evidence_body_sha256": rehearsal.value[
            "rehearsal_body_sha256"
        ],
    }
    for field, expected in qualification_bindings.items():
        if readiness.value.get(field) != expected:
            raise SubmissionIntegrationError(
                f"qualification remote-chain {field} binding mismatch"
            )


def _load_local(
    request: QualificationRequest,
    services: SubmissionServices,
) -> _LocalAuthorities:
    paths = request.local
    if paths is None:
        raise SubmissionUsageError(
            f"{request.action} requires all local authority paths"
        )
    descriptor_value = _decode_canonical_file(paths.descriptor, label="descriptor")
    descriptor = validate_sky_campaign_descriptor(descriptor_value)
    if (
        descriptor.get("account_id") != APPROVED_ACCOUNT_ID
        or descriptor.get("region") != APPROVED_REGION
    ):
        raise SubmissionIntegrationError("descriptor account or region is foreign")
    if descriptor.get("task_name") != "glm52-campaign":
        raise SubmissionIntegrationError("descriptor Sky task is foreign")
    if descriptor.get("instance_type") != EXPECTED_INSTANCE_TYPE:
        raise SubmissionIntegrationError("descriptor instance type is foreign")
    if descriptor.get("use_spot") is not False or descriptor.get("instance_count") != 1:
        raise SubmissionIntegrationError("descriptor must authorize one on-demand P5")
    _require_submission_window(descriptor=descriptor, now=_now(services))
    run_id = str(descriptor["run_id"])
    descriptor_key = str(descriptor["campaign_descriptor_key"])
    seed_value = validate_sky_campaign_descriptor(
        _decode_canonical_file(paths.seed_descriptor, label="seed descriptor")
    )
    approval_value = validate_gpu_spend_approval(
        _decode_canonical_file(paths.approval, label="approval")
    )
    acceptance_value = validate_qualification_cache_seed_accepted(
        _decode_canonical_file(
            paths.cache_seed_accepted,
            label="cache-seed acceptance",
        )
    )
    snapshot_value = validate_gpu_spend_snapshot(
        _decode_canonical_file(
            paths.gpu_spend_snapshot,
            label="GPU spend snapshot",
        )
    )
    staged_value = _decode_canonical_file(
        paths.staged_ready,
        label="staged readiness",
    )
    rehearsal_value = _decode_canonical_file(
        paths.rehearsal_evidence,
        label="rehearsal evidence",
    )
    readiness_value = _decode_canonical_file(
        paths.qualification_ready,
        label="qualification readiness",
    )
    descriptor_raw = paths.descriptor.read_bytes()
    seed_raw = paths.seed_descriptor.read_bytes()
    approval_raw = paths.approval.read_bytes()
    staged_raw = paths.staged_ready.read_bytes()
    rehearsal_raw = paths.rehearsal_evidence.read_bytes()
    acceptance_raw = paths.cache_seed_accepted.read_bytes()
    snapshot_raw = paths.gpu_spend_snapshot.read_bytes()
    readiness_raw = paths.qualification_ready.read_bytes()
    if _sha(approval_raw) != descriptor["approval_sha256"]:
        raise SubmissionIntegrationError(
            "approval file identity does not match descriptor"
        )
    if readiness_value.get("seed_descriptor") != seed_value:
        raise SubmissionIntegrationError(
            "caller-pinned seed descriptor differs from readiness"
        )
    staged_key = (
        descriptor_key.removesuffix("campaign-descriptor-v2.json")
        + "STAGED_CONTROL_PLANE_READY.json"
    )
    acceptance_key = accepted_s3_key(
        run_id=run_id,
        acceptance_body_sha256=str(acceptance_value["acceptance_body_sha256"]),
    )
    snapshot_key = gpu_spend_snapshot_s3_key(
        run_id=run_id,
        snapshot_body_sha256=str(snapshot_value["snapshot_body_sha256"]),
    )
    rehearsal_key = rehearsal_evidence_s3_key(
        run_id=run_id,
        rehearsal_body_sha256=str(rehearsal_value["rehearsal_body_sha256"]),
    )
    readiness_key = qualification_submission_ready_s3_key(readiness_value)
    now = _now(services)
    validate_qualification_submission_ready(
        readiness_value,
        descriptor=descriptor,
        staged_readiness=staged_value,
        cache_seed_acceptance=acceptance_value,
        gpu_spend_snapshot=snapshot_value,
        rehearsal_evidence=rehearsal_value,
        descriptor_file_sha256=_sha(descriptor_raw),
        staged_readiness_key=staged_key,
        staged_readiness_file_sha256=_sha(staged_raw),
        cache_seed_acceptance_key=acceptance_key,
        cache_seed_acceptance_file_sha256=_sha(acceptance_raw),
        gpu_spend_snapshot_key=snapshot_key,
        gpu_spend_snapshot_sha256=_sha(snapshot_raw),
        rehearsal_evidence_key=rehearsal_key,
        rehearsal_evidence_sha256=_sha(rehearsal_raw),
        now=now,
    )
    if _sha(seed_raw) != readiness_value["seed_descriptor_file_sha256"]:
        raise SubmissionIntegrationError("seed descriptor file identity mismatch")
    local = _LocalAuthorities(
        descriptor=_JsonArtifact(descriptor_key, descriptor_raw, descriptor),
        seed_descriptor=_JsonArtifact(
            str(readiness_value["seed_descriptor_key"]),
            seed_raw,
            seed_value,
        ),
        approval=_JsonArtifact(
            str(descriptor["approval_key"]), approval_raw, approval_value
        ),
        staged=_JsonArtifact(staged_key, staged_raw, staged_value),
        rehearsal=_JsonArtifact(rehearsal_key, rehearsal_raw, rehearsal_value),
        cache_acceptance=_JsonArtifact(
            acceptance_key,
            acceptance_raw,
            acceptance_value,
        ),
        spend_snapshot=_JsonArtifact(snapshot_key, snapshot_raw, snapshot_value),
        readiness=_JsonArtifact(readiness_key, readiness_raw, readiness_value),
        task=paths.task,
        config=paths.config,
    )
    _authenticate_remote_staged_chain(
        services,
        descriptor=local.descriptor,
        staged=local.staged,
        rehearsal=local.rehearsal,
        readiness=local.readiness,
        approval=local.approval,
    )
    task_raw = _read_regular(paths.task, label="Sky task")
    config_raw = _read_regular(paths.config, label="Sky config")
    validation = services.sky.validate_control_plane(
        task=paths.task,
        config=paths.config,
    )
    if not isinstance(validation, Mapping):
        raise SubmissionIntegrationError("Sky control-plane validation is malformed")
    if (
        validation.get("skypilot_version") != EXPECTED_SKYPILOT_VERSION
        or validation.get("task_file_sha256")
        != rehearsal_value.get("skypilot_task_file_sha256")
        or validation.get("config_file_sha256")
        != rehearsal_value.get("skypilot_config_file_sha256")
        or _sha(task_raw) != rehearsal_value.get("skypilot_task_file_sha256")
        or _sha(config_raw) != rehearsal_value.get("skypilot_config_file_sha256")
    ):
        raise SubmissionIntegrationError("Sky task/config authority drift")
    return local


def _worker_start_v2_readiness_authorities(
    *,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
) -> WorkerStartV2ReadinessAuthorities:
    return WorkerStartV2ReadinessAuthorities(
        descriptor=WorkerStartV2ReadinessArtifact(
            key=local.descriptor.key,
            raw=local.descriptor.raw,
        ),
        qualification_submission_ready=WorkerStartV2ReadinessArtifact(
            key=local.readiness.key,
            raw=local.readiness.raw,
        ),
        rehearsal_evidence=WorkerStartV2ReadinessArtifact(
            key=local.rehearsal.key,
            raw=local.rehearsal.raw,
        ),
        intent=WorkerStartV2ReadinessArtifact(
            key=intent.key,
            raw=intent.raw,
        ),
    )


def _require_remote_static(
    services: SubmissionServices,
    *,
    local: _LocalAuthorities,
) -> None:
    descriptor = local.descriptor.value
    bucket = str(descriptor["bucket"])
    exact = (
        local.descriptor,
        local.approval,
        local.staged,
    )
    for artifact in exact:
        remote = _get_exact(services.s3, bucket=bucket, key=artifact.key)
        if remote != artifact.raw:
            raise SubmissionIntegrationError(
                f"remote static authority drift: {artifact.key}"
            )
    repo_key = str(descriptor["repo_tar_key"])
    repo_raw = _get_exact(services.s3, bucket=bucket, key=repo_key)
    if repo_raw is None or _sha(repo_raw) != descriptor["repo_tar_sha256"]:
        raise SubmissionIntegrationError("remote repository identity drift")
    artifacts = descriptor.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SubmissionIntegrationError("descriptor artifact pins are malformed")
    inventory_key = str(artifacts["artifact_inventory_key"])
    inventory_raw = _get_exact(services.s3, bucket=bucket, key=inventory_key)
    if (
        inventory_raw is None
        or _sha(inventory_raw) != artifacts["artifact_inventory_sha256"]
    ):
        raise SubmissionIntegrationError("remote inventory identity drift")
    audit_key = str(local.staged.value["artifact_audit_key"])
    audit_raw = _get_exact(services.s3, bucket=bucket, key=audit_key)
    if (
        audit_raw is None
        or _sha(audit_raw) != local.staged.value["artifact_audit_sha256"]
    ):
        raise SubmissionIntegrationError("remote artifact audit identity drift")
    latest_key = (
        f"campaigns/{descriptor['run_id']}/runtime/GPU_SPEND_LEDGER_LATEST.json"
    )
    latest_raw = _get_exact(services.s3, bucket=bucket, key=latest_key)
    if latest_raw is None:
        raise SubmissionIntegrationError("current GPU spend latest marker is absent")
    try:
        latest = json.loads(latest_raw)
    except json.JSONDecodeError as error:
        raise SubmissionIntegrationError(
            "current GPU spend latest marker is malformed"
        ) from error
    if (
        not isinstance(latest, dict)
        or latest_raw != _canonical(latest) + b"\n"
        or latest.get("record_type") != "glm52_gpu_spend_ledger_latest_v1"
        or latest.get("run_id") != descriptor["run_id"]
    ):
        raise SubmissionIntegrationError(
            "current GPU spend latest marker is noncanonical or foreign"
        )
    latest_body = dict(latest)
    digest = latest_body.pop("latest_body_sha256", None)
    if digest != _sha(_canonical(latest_body)):
        raise SubmissionIntegrationError(
            "current GPU spend latest marker body SHA mismatch"
        )
    record_names = latest.get("record_keys")
    if (
        not isinstance(record_names, list)
        or any(
            not isinstance(name, str) or not name or "/" in name or name in {".", ".."}
            for name in record_names
        )
        or len(record_names) != len(set(record_names))
    ):
        raise SubmissionIntegrationError(
            "current GPU spend immutable record inventory is malformed"
        )
    record_prefix = (
        f"campaigns/{descriptor['run_id']}/runtime/spend-ledger/records/"
    )
    expected_record_keys = tuple(record_prefix + name for name in record_names)
    if len(record_names) != len(expected_record_keys):
        raise SubmissionIntegrationError(
            "current GPU spend immutable record inventory length drifted"
        )
    if _list_complete(
        services.s3,
        bucket=bucket,
        prefix=record_prefix,
    ) != tuple(sorted(expected_record_keys)):
        raise SubmissionIntegrationError(
            "current GPU spend immutable record inventory drifted"
        )
    record_raw_by_name: dict[str, bytes] = {}
    for name, key in zip(record_names, expected_record_keys):
        raw = _get_exact(services.s3, bucket=bucket, key=key)
        if raw is None:
            raise SubmissionIntegrationError(
                f"current GPU spend immutable record is absent: {name}"
            )
        record_raw_by_name[name] = raw
    history_raw = _live_spend_history_raw(
        services,
        run_id=str(descriptor["run_id"]),
    )
    snapshot = local.spend_snapshot.value
    try:
        rebuilt_snapshot = build_gpu_spend_snapshot(
            descriptor_raw=local.descriptor.raw,
            expected_descriptor_sha256=local.descriptor.file_sha256,
            approval_raw=local.approval.raw,
            expected_approval_sha256=local.approval.file_sha256,
            latest_raw=latest_raw,
            expected_latest_sha256=_sha(latest_raw),
            immutable_record_raw_by_key=record_raw_by_name,
            ec2_allocation_history_raw=history_raw,
            expected_ec2_allocation_history_sha256=_sha(history_raw),
            observed_at=str(snapshot.get("observed_at")),
        )
    except Exception as error:
        raise SubmissionIntegrationError(
            f"current cumulative GPU spend reconstruction failed: {error}"
        ) from error
    if (
        rebuilt_snapshot != snapshot
        or _sha(history_raw) != snapshot["ec2_allocation_history_sha256"]
        or _sha(latest_raw) != snapshot["gpu_spend_ledger_latest_sha256"]
        or digest != snapshot["gpu_spend_ledger_latest_body_sha256"]
        or latest.get("latest_record_sha256")
        != snapshot["gpu_spend_ledger_tip_record_sha256"]
        or snapshot.get("open_allocation_count") != 0
    ):
        raise SubmissionIntegrationError("current GPU spend tip drifted")


def _live_spend_history_raw(
    services: SubmissionServices,
    *,
    run_id: str,
) -> bytes:
    """Return the deterministic complete EC2 projection used by spend snapshots."""

    required_tags = {
        "project": "keep-glm52",
        "owner": "jack.mazac",
        "model": "glm-5.2",
        "campaign-run-id": run_id,
        "cost-allocation": "glm52-sky-campaign",
    }
    filters: list[dict[str, object]] = [
        {"Name": f"tag:{key}", "Values": [value]}
        for key, value in required_tags.items()
    ]
    filters.append({"Name": "instance-type", "Values": [EXPECTED_INSTANCE_TYPE]})
    token: str | None = None
    projected: list[dict[str, object]] = []
    while True:
        arguments: dict[str, object] = {"Filters": filters}
        if token is not None:
            arguments["NextToken"] = token
        response = services.ec2.describe_instances(**arguments)
        if not isinstance(response, Mapping):
            raise SubmissionIntegrationError(
                "EC2 cumulative spend history is malformed"
            )
        reservations = response.get("Reservations")
        if not isinstance(reservations, list):
            raise SubmissionIntegrationError(
                "EC2 cumulative spend reservations are malformed"
            )
        for reservation in reservations:
            if (
                not isinstance(reservation, Mapping)
                or reservation.get("OwnerId") != APPROVED_ACCOUNT_ID
                or not isinstance(reservation.get("Instances"), list)
            ):
                raise SubmissionIntegrationError(
                    "EC2 cumulative spend reservation is foreign or malformed"
                )
            for instance in reservation["Instances"]:
                if not isinstance(instance, Mapping):
                    raise SubmissionIntegrationError(
                        "EC2 cumulative spend instance is malformed"
                    )
                instance_id = instance.get("InstanceId")
                launch_time = instance.get("LaunchTime")
                if (
                    not isinstance(instance_id, str)
                    or not instance_id.startswith("i-")
                    or not isinstance(launch_time, (str, datetime))
                ):
                    raise SubmissionIntegrationError(
                        "EC2 cumulative spend instance identity is malformed"
                    )
                if isinstance(launch_time, datetime):
                    if launch_time.tzinfo is None or launch_time.utcoffset() is None:
                        raise SubmissionIntegrationError(
                            "EC2 cumulative spend launch time lacks a timezone"
                        )
                    launch_time_value = (
                        launch_time.astimezone(UTC).isoformat().replace("+00:00", "Z")
                    )
                else:
                    launch_time_value, _ = _controller_time(
                        launch_time,
                        field="EC2 cumulative spend LaunchTime",
                    )
                    assert launch_time_value is not None
                tags = _instance_tags(instance)
                if any(tags.get(key) != value for key, value in required_tags.items()):
                    raise SubmissionIntegrationError(
                        "EC2 cumulative spend instance tags are foreign"
                    )
                state = instance.get("State")
                placement = instance.get("Placement")
                if (
                    not isinstance(state, Mapping)
                    or not isinstance(state.get("Name"), str)
                    or not isinstance(placement, Mapping)
                    or not isinstance(placement.get("AvailabilityZone"), str)
                ):
                    raise SubmissionIntegrationError(
                        "EC2 cumulative spend instance state is malformed"
                    )
                value: dict[str, object] = {
                    "InstanceId": instance_id,
                    "InstanceType": instance.get("InstanceType"),
                    "LaunchTime": launch_time_value,
                    "State": {"Name": state["Name"]},
                    "StateTransitionReason": instance.get(
                        "StateTransitionReason",
                        "",
                    ),
                    "Placement": {
                        "AvailabilityZone": placement["AvailabilityZone"],
                    },
                    "Tags": [{"Key": key, "Value": tags[key]} for key in sorted(tags)],
                }
                if instance.get("InstanceLifecycle") is not None:
                    value["InstanceLifecycle"] = instance["InstanceLifecycle"]
                projected.append(value)
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if not isinstance(next_token, str) or not next_token or next_token == token:
            raise SubmissionIntegrationError(
                "EC2 cumulative spend pagination token is invalid"
            )
        token = next_token
    projected.sort(key=lambda value: str(value["InstanceId"]))
    if len({str(value["InstanceId"]) for value in projected}) != len(projected):
        raise SubmissionIntegrationError(
            "EC2 cumulative spend history contains duplicate instances"
        )
    return (
        _canonical(
            {
                "Reservations": [
                    {
                        "OwnerId": APPROVED_ACCOUNT_ID,
                        "Instances": projected,
                    }
                ]
            }
        )
        + b"\n"
    )


def _describe_instances(
    ec2: Any,
    *,
    filters: list[dict[str, object]] | None = None,
    instance_ids: list[str] | None = None,
) -> list[dict[str, object]]:
    token: str | None = None
    instances: list[dict[str, object]] = []
    while True:
        arguments: dict[str, object] = {}
        if filters is not None:
            arguments["Filters"] = filters
        if instance_ids is not None:
            arguments["InstanceIds"] = instance_ids
        if token is not None:
            arguments["NextToken"] = token
        response = ec2.describe_instances(**arguments)
        if not isinstance(response, Mapping):
            raise SubmissionIntegrationError("EC2 describe-instances is malformed")
        reservations = response.get("Reservations", [])
        if not isinstance(reservations, list):
            raise SubmissionIntegrationError("EC2 reservations are malformed")
        for reservation in reservations:
            if not isinstance(reservation, Mapping):
                raise SubmissionIntegrationError("EC2 reservation is malformed")
            values = reservation.get("Instances", [])
            if not isinstance(values, list):
                raise SubmissionIntegrationError("EC2 instances are malformed")
            for value in values:
                if not isinstance(value, Mapping):
                    raise SubmissionIntegrationError("EC2 instance is malformed")
                instances.append(dict(value))
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if not isinstance(next_token, str) or not next_token or next_token == token:
            raise SubmissionIntegrationError("EC2 pagination token is invalid")
        token = next_token
    return instances


def _instance_tags(instance: Mapping[str, object]) -> dict[str, str]:
    tags = instance.get("Tags", [])
    if not isinstance(tags, list):
        raise SubmissionIntegrationError("EC2 instance tags are malformed")
    result: dict[str, str] = {}
    for item in tags:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("Key"), str)
            or not isinstance(item.get("Value"), str)
            or str(item["Key"]) in result
        ):
            raise SubmissionIntegrationError("EC2 instance tags are malformed")
        result[str(item["Key"])] = str(item["Value"])
    return result


def _history_shim() -> str:
    return r"""
import json
import sys
from datetime import datetime, timezone
from sky.jobs import state

name = sys.argv[1]
workspace = sys.argv[2]
controller_identity = sys.argv[3]

def enum_name(value):
    if value is None:
        return None
    return str(getattr(value, "value", getattr(value, "name", value))).upper()

def iso_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(value, timezone.utc)
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

rows = []
for job_id in sorted(set(state.get_all_job_ids_by_name(name))):
    matches, total = state.get_managed_jobs_with_filters(
        job_ids=[job_id],
        workspace_match=workspace,
    )
    exact = [
        row for row in matches
        if row.get("job_id") == job_id
        and row.get("job_name") == name
        and row.get("workspace") == workspace
    ]
    if total != 1 or len(exact) != 1:
        raise RuntimeError("exact controller row is missing or ambiguous")
    row = exact[0]
    rows.append({
        "sky_job_id": job_id,
        "sky_job_name": name,
        "workspace": workspace,
        "controller_submitted_at": iso_time(row.get("submitted_at")),
        "controller_status": enum_name(row.get("status")),
        "controller_identity": controller_identity,
        "schedule_state": enum_name(row.get("schedule_state")),
        "start_at": iso_time(row.get("start_at")),
        "worker_cluster_name": row.get("current_cluster_name"),
        "recovery_count": row.get("recovery_count"),
    })
result = {
    "schema_version": 1,
    "record_type": "glm52_sky_controller_history_v1",
    "sky_job_name": name,
    "workspace": workspace,
    "rows": rows,
    "observed_at": datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z"),
}
print("GLM52_SUBMISSION_HISTORY=" + json.dumps(
    result, sort_keys=True, separators=(",", ":")
))
""".strip()


def _observe_history(
    services: SubmissionServices,
    *,
    controller_instance_id: str,
    sky_job_name: str,
    controller_identity: str,
) -> tuple[list[dict[str, object]], str]:
    command = (
        "sudo -u ubuntu -- /home/ubuntu/skypilot-runtime/bin/python -c "
        f"{shlex.quote(_history_shim())} "
        f"{shlex.quote(sky_job_name)} {shlex.quote(EXPECTED_WORKSPACE)} "
        f"{shlex.quote(controller_identity)}"
    )
    response = services.ssm.send_command(
        InstanceIds=[controller_instance_id],
        DocumentName="AWS-RunShellScript",
        Comment="KEEP GLM52 observe exact qualification controller history",
        Parameters={"commands": [command]},
        TimeoutSeconds=45,
    )
    if not isinstance(response, Mapping):
        raise SubmissionIntegrationError("SSM send-command response is malformed")
    command_value = response.get("Command")
    if not isinstance(command_value, Mapping) or not isinstance(
        command_value.get("CommandId"), str
    ):
        raise SubmissionIntegrationError("SSM command ID is missing")
    command_id = str(command_value["CommandId"])
    for _ in range(10):
        try:
            invocation = services.ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=controller_instance_id,
            )
        except Exception as error:
            code, _ = _aws_error(error)
            if code == "InvocationDoesNotExist":
                services.sleep(1)
                continue
            raise SubmissionIntegrationError(
                f"SSM controller observation failed: {code or error}"
            ) from error
        if not isinstance(invocation, Mapping):
            raise SubmissionIntegrationError("SSM invocation is malformed")
        status = invocation.get("Status")
        if status in {"Pending", "InProgress", "Delayed"}:
            services.sleep(1)
            continue
        if status != "Success":
            raise SubmissionIntegrationError(
                f"SSM controller observation failed: {status}"
            )
        prefix = "GLM52_SUBMISSION_HISTORY="
        lines = [
            line
            for line in str(invocation.get("StandardOutputContent", "")).splitlines()
            if line.startswith(prefix)
        ]
        if len(lines) != 1:
            raise SubmissionIntegrationError(
                "SSM controller history result is missing or ambiguous"
            )
        try:
            result = json.loads(lines[0][len(prefix) :])
        except json.JSONDecodeError as error:
            raise SubmissionIntegrationError(
                "SSM controller history result is not JSON"
            ) from error
        if (
            not isinstance(result, dict)
            or set(result)
            != {
                "schema_version",
                "record_type",
                "sky_job_name",
                "workspace",
                "rows",
                "observed_at",
            }
            or result.get("schema_version") != 1
            or result.get("record_type") != "glm52_sky_controller_history_v1"
            or result.get("sky_job_name") != sky_job_name
            or result.get("workspace") != EXPECTED_WORKSPACE
        ):
            raise SubmissionIntegrationError(
                "SSM controller history authority mismatch"
            )
        rows = result.get("rows")
        if not isinstance(rows, list):
            raise SubmissionIntegrationError(
                "SSM controller history rows are malformed"
            )
        validated: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != _HISTORY_ROW_FIELDS:
                raise SubmissionIntegrationError(
                    "SSM controller history row schema mismatch"
                )
            job_id = row.get("sky_job_id")
            recovery_count = row.get("recovery_count")
            if (
                type(job_id) is not int
                or job_id <= 0
                or row.get("sky_job_name") != sky_job_name
                or row.get("workspace") != EXPECTED_WORKSPACE
                or row.get("controller_identity") != controller_identity
                or row.get("controller_status") not in _CONTROLLER_STATUSES
                or not isinstance(row.get("schedule_state"), str)
                or not row.get("schedule_state")
                or type(recovery_count) is not int
                or recovery_count < 0
            ):
                raise SubmissionIntegrationError(
                    "SSM controller history row is foreign or malformed"
                )
            submitted_at, _ = _controller_time(
                row.get("controller_submitted_at"),
                field="controller submitted_at",
                permit_none=row.get("controller_status") in {"PENDING", "SUBMITTED"},
            )
            if submitted_at is None and row.get("controller_status") not in {
                "PENDING",
                "SUBMITTED",
            }:
                raise SubmissionIntegrationError(
                    "controller submitted_at is missing after pending"
                )
            start_at = row.get("start_at")
            if start_at is not None:
                _controller_time(start_at, field="controller start_at")
            cluster = row.get("worker_cluster_name")
            if cluster is not None and not isinstance(cluster, str):
                raise SubmissionIntegrationError(
                    "SSM controller worker cluster is malformed"
                )
            validated.append(
                {
                    **dict(row),
                    "controller_submitted_at": submitted_at,
                }
            )
        if [row["sky_job_id"] for row in validated] != sorted(
            {int(row["sky_job_id"]) for row in validated}
        ):
            raise SubmissionIntegrationError(
                "SSM controller history is duplicate or unsorted"
            )
        observed_at, _ = _whole_second(
            str(result.get("observed_at")),
            field="controller history observed_at",
        )
        return validated, observed_at
    raise SubmissionIntegrationError("SSM controller observation timed out")


def _capture_controller_facts(
    services: SubmissionServices,
    *,
    descriptor: Mapping[str, object],
    intent: Mapping[str, object],
) -> _ControllerFacts:
    run_id = str(descriptor["run_id"])
    required_tags = {
        "project": "keep-glm52",
        "owner": "jack.mazac",
        "model": "glm-5.2",
        "campaign-run-id": run_id,
        "cost-allocation": "glm52-sky-campaign",
    }
    controller_filters: list[dict[str, object]] = [
        {"Name": f"tag:{key}", "Values": [value]}
        for key, value in required_tags.items()
    ]
    controller_filters.extend(
        [
            {"Name": "tag:ray-cluster-name", "Values": ["sky-jobs-controller-*"]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )
    controllers = _describe_instances(
        services.ec2,
        filters=controller_filters,
    )
    if len(controllers) != 1:
        raise SubmissionIntegrationError(
            "must resolve exactly one tagged SkyPilot controller"
        )
    controller = controllers[0]
    instance_id = controller.get("InstanceId")
    instance_type = controller.get("InstanceType")
    state = controller.get("State")
    profile = controller.get("IamInstanceProfile")
    placement = controller.get("Placement")
    if (
        not isinstance(instance_id, str)
        or not instance_id.startswith("i-")
        or not isinstance(instance_type, str)
        or not instance_type
        or not isinstance(state, Mapping)
        or state.get("Name") != "running"
        or not isinstance(profile, Mapping)
        or not isinstance(profile.get("Arn"), str)
        or not isinstance(placement, Mapping)
        or not str(placement.get("AvailabilityZone", "")).startswith(
            str(descriptor["region"])
        )
    ):
        raise SubmissionIntegrationError("SkyPilot controller EC2 identity is invalid")
    tags = _instance_tags(controller)
    if any(tags.get(key) != value for key, value in required_tags.items()):
        raise SubmissionIntegrationError("SkyPilot controller tags are foreign")
    cluster_name = tags.get("ray-cluster-name")
    if not isinstance(cluster_name, str) or not cluster_name.startswith(
        "sky-jobs-controller-"
    ):
        raise SubmissionIntegrationError("SkyPilot controller cluster tag is invalid")
    role_arn = str(descriptor["controller_identity"])
    role_prefix = f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
    if not role_arn.startswith(role_prefix):
        raise SubmissionIntegrationError("descriptor controller role is invalid")
    expected_profile = (
        f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:instance-profile/"
        f"{role_arn.removeprefix(role_prefix)}"
    )
    if profile.get("Arn") != expected_profile:
        raise SubmissionIntegrationError("SkyPilot controller profile is foreign")
    information = services.ssm.describe_instance_information(
        Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
    )
    if not isinstance(information, Mapping):
        raise SubmissionIntegrationError("SSM controller identity is malformed")
    entries = information.get("InstanceInformationList")
    if (
        not isinstance(entries, list)
        or len(entries) != 1
        or not isinstance(entries[0], Mapping)
        or entries[0].get("InstanceId") != instance_id
        or entries[0].get("PingStatus") != "Online"
    ):
        raise SubmissionIntegrationError(
            "SkyPilot controller is not exactly SSM Online"
        )
    rows, observed_at = _observe_history(
        services,
        controller_instance_id=instance_id,
        sky_job_name=str(intent["sky_job_name"]),
        controller_identity=role_arn,
    )
    _, observed_time = _whole_second(
        observed_at,
        field="controller history observed_at",
    )
    observed_now = _now(services)
    if observed_time > observed_now or observed_now - observed_time > MAX_LIVE_AGE:
        raise SubmissionIntegrationError(
            "controller history observation is not current"
        )
    active_ids = sorted(
        int(row["sky_job_id"])
        for row in rows
        if row["controller_status"] in _ACTIVE_CONTROLLER_STATUSES
    )
    p5_filters: list[dict[str, object]] = [
        {"Name": f"tag:{key}", "Values": [value]}
        for key, value in required_tags.items()
    ]
    p5_filters.extend(
        [
            {"Name": "instance-type", "Values": [EXPECTED_INSTANCE_TYPE]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping"],
            },
        ]
    )
    p5_values = _describe_instances(services.ec2, filters=p5_filters)
    active_p5_ids: list[str] = []
    for value in p5_values:
        value_id = value.get("InstanceId")
        if not isinstance(value_id, str) or not value_id.startswith("i-"):
            raise SubmissionIntegrationError("active tagged P5 ID is invalid")
        if value.get("InstanceType") != EXPECTED_INSTANCE_TYPE:
            raise SubmissionIntegrationError("active tagged instance is not a P5")
        value_tags = _instance_tags(value)
        if any(value_tags.get(key) != tag for key, tag in required_tags.items()):
            raise SubmissionIntegrationError("active P5 tags are foreign")
        active_p5_ids.append(value_id)
    active_p5_ids.sort()
    baseline_rows: list[dict[str, object]] = []
    for row in rows:
        submitted_at, _ = _controller_time(
            row["controller_submitted_at"],
            field="controller submitted_at",
            permit_none=True,
            whole_second=True,
        )
        baseline_rows.append(
            {
                **{field: row[field] for field in _BASELINE_ROW_FIELDS},
                "controller_submitted_at": submitted_at,
            }
        )
    return _ControllerFacts(
        controller_instance_id=instance_id,
        controller_instance_type=str(instance_type),
        controller_profile_arn=expected_profile,
        controller_cluster_name=cluster_name,
        ssm_ping_status="Online",
        exact_name_history=baseline_rows,
        detailed_history=rows,
        active_exact_name_job_ids=active_ids,
        active_tagged_p5_instance_ids=active_p5_ids,
        observed_at=observed_at,
    )


def _build_intent_artifact(
    local: _LocalAuthorities,
    *,
    intent_at: datetime,
) -> _JsonArtifact:
    descriptor = local.descriptor.value
    readiness = local.readiness.value
    intent = build_submission_intent(
        descriptor=descriptor,
        managed_mode=QUALIFICATION_MODE,
        descriptor_file_sha256=local.descriptor.file_sha256,
        cache_seed_acceptance=local.cache_acceptance.value,
        cache_seed_acceptance_key=local.cache_acceptance.key,
        cache_seed_acceptance_file_sha256=local.cache_acceptance.file_sha256,
        gpu_spend_snapshot=local.spend_snapshot.value,
        gpu_spend_snapshot_key=local.spend_snapshot.key,
        gpu_spend_snapshot_sha256=local.spend_snapshot.file_sha256,
        staged_readiness=local.staged.value,
        rehearsal_evidence=local.rehearsal.value,
        qualification_submission_ready=readiness,
        qualification_submission_ready_key=local.readiness.key,
        qualification_submission_ready_sha256=local.readiness.file_sha256,
        qualification_submission_ready_body_sha256=str(
            readiness["readiness_body_sha256"]
        ),
        sky_job_name=expected_sky_job_name(
            str(descriptor["run_id"]),
            QUALIFICATION_MODE,
        ),
        intent_at=intent_at,
    )
    validate_submission_intent(
        intent,
        descriptor=descriptor,
        cache_seed_acceptance=local.cache_acceptance.value,
        gpu_spend_snapshot=local.spend_snapshot.value,
        staged_readiness=local.staged.value,
        rehearsal_evidence=local.rehearsal.value,
        qualification_submission_ready=readiness,
    )
    key = submission_intent_s3_key(
        run_id=str(intent["run_id"]),
        managed_mode=str(intent["managed_mode"]),
        intent_body_sha256=str(intent[INTENT_DIGEST_FIELD]),
    )
    return _JsonArtifact(key=key, raw=canonical_file_bytes(intent), value=intent)


def _build_baseline_artifact(
    services: SubmissionServices,
    *,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
) -> tuple[_JsonArtifact, _ControllerFacts]:
    facts = _capture_controller_facts(
        services,
        descriptor=local.descriptor.value,
        intent=intent.value,
    )
    if facts.active_exact_name_job_ids or facts.active_tagged_p5_instance_ids:
        raise SubmissionIntegrationError(
            "active exact-name job or tagged P5 exists before acquisition"
        )
    baseline = build_controller_baseline(
        descriptor=local.descriptor.value,
        descriptor_key=local.descriptor.key,
        descriptor_raw=local.descriptor.raw,
        qualification_submission_ready=local.readiness.value,
        qualification_submission_ready_key=local.readiness.key,
        qualification_submission_ready_raw=local.readiness.raw,
        intent=intent.value,
        intent_key=intent.key,
        intent_raw=intent.raw,
        controller_instance_id=facts.controller_instance_id,
        controller_instance_type=facts.controller_instance_type,
        controller_profile_arn=facts.controller_profile_arn,
        controller_cluster_name=facts.controller_cluster_name,
        ssm_ping_status=facts.ssm_ping_status,
        exact_name_history=facts.exact_name_history,
        active_exact_name_job_ids=facts.active_exact_name_job_ids,
        active_tagged_p5_instance_ids=facts.active_tagged_p5_instance_ids,
        observed_at=facts.observed_at,
    )
    baseline_artifact = _JsonArtifact(
        key=controller_baseline_s3_key(baseline),
        raw=controller_baseline_file_bytes(baseline),
        value=baseline,
    )
    return baseline_artifact, facts


def _require_facts_match_baseline(
    *,
    facts: _ControllerFacts,
    baseline: Mapping[str, object],
) -> None:
    expected = {
        "controller_instance_id": facts.controller_instance_id,
        "controller_instance_type": facts.controller_instance_type,
        "controller_profile_arn": facts.controller_profile_arn,
        "controller_cluster_name": facts.controller_cluster_name,
        "ssm_ping_status": facts.ssm_ping_status,
        "exact_name_history": facts.exact_name_history,
        "active_exact_name_job_ids": facts.active_exact_name_job_ids,
        "active_tagged_p5_instance_ids": facts.active_tagged_p5_instance_ids,
    }
    if any(baseline.get(field) != value for field, value in expected.items()):
        raise SubmissionIntegrationError(
            "current controller/P5 facts differ from the reviewed baseline"
        )


def _require_controller_coordinates_match_baseline(
    *,
    facts: _ControllerFacts,
    baseline: Mapping[str, object],
) -> None:
    expected = {
        "controller_instance_id": facts.controller_instance_id,
        "controller_instance_type": facts.controller_instance_type,
        "controller_profile_arn": facts.controller_profile_arn,
        "controller_cluster_name": facts.controller_cluster_name,
        "ssm_ping_status": facts.ssm_ping_status,
    }
    if any(baseline.get(field) != value for field, value in expected.items()):
        raise SubmissionIntegrationError(
            "current controller coordinates differ from the selected baseline"
        )


def _build_control_ready_artifact(
    services: SubmissionServices,
    *,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
    baseline_artifact: _JsonArtifact,
) -> _JsonArtifact:
    baseline = baseline_artifact.value
    deployment = services.cloudformation.inspect_dynamic_must_start(
        run_id=str(intent.value["run_id"]),
        managed_mode=str(intent.value["managed_mode"]),
        intent_body_sha256=str(intent.value[INTENT_DIGEST_FIELD]),
        controller_baseline_body_sha256=str(baseline["baseline_body_sha256"]),
    )
    if deployment is None:
        raise SubmissionDeploymentRequired(
            "dynamic must-start deployment is absent; execute it separately"
        )
    if not isinstance(deployment, Mapping):
        raise SubmissionIntegrationError(
            "must-start deployment inspection is malformed"
        )
    required = {
        "reviewed_deployment_identity",
        "observed_deployment_identity",
        "reconciliation_rule_state",
        "deadline_schedule_state",
        "coordinator_mode",
        "observed_at",
    }
    if set(deployment) != required:
        raise SubmissionIntegrationError(
            "must-start deployment inspection schema mismatch"
        )
    control_ready = build_must_start_control_plane_ready(
        descriptor=local.descriptor.value,
        descriptor_key=local.descriptor.key,
        descriptor_raw=local.descriptor.raw,
        qualification_submission_ready=local.readiness.value,
        qualification_submission_ready_key=local.readiness.key,
        qualification_submission_ready_raw=local.readiness.raw,
        intent=intent.value,
        intent_key=intent.key,
        intent_raw=intent.raw,
        controller_baseline=baseline,
        controller_baseline_key=baseline_artifact.key,
        controller_baseline_raw=baseline_artifact.raw,
        controller_instance_id=str(baseline["controller_instance_id"]),
        controller_instance_type=str(baseline["controller_instance_type"]),
        controller_profile_arn=str(baseline["controller_profile_arn"]),
        controller_cluster_name=str(baseline["controller_cluster_name"]),
        baseline_ssm_ping_status=str(baseline["ssm_ping_status"]),
        baseline_exact_name_history=baseline["exact_name_history"],  # type: ignore[arg-type]
        baseline_active_exact_name_job_ids=baseline[  # type: ignore[arg-type]
            "active_exact_name_job_ids"
        ],
        baseline_active_tagged_p5_instance_ids=baseline[  # type: ignore[arg-type]
            "active_tagged_p5_instance_ids"
        ],
        baseline_observed_at=str(baseline["observed_at"]),
        reviewed_deployment_identity=deployment["reviewed_deployment_identity"],
        observed_deployment_identity=deployment["observed_deployment_identity"],
        reconciliation_rule_state=str(deployment["reconciliation_rule_state"]),
        deadline_schedule_state=str(deployment["deadline_schedule_state"]),
        coordinator_mode=str(deployment["coordinator_mode"]),
        observed_at=deployment["observed_at"],
    )
    if control_ready.get("coordinator_mode") != COORDINATOR_MODE:
        raise SubmissionIntegrationError("must-start coordinator mode is inactive")
    control_artifact = _JsonArtifact(
        key=must_start_control_plane_ready_s3_key(control_ready),
        raw=must_start_control_plane_ready_file_bytes(control_ready),
        value=control_ready,
    )
    return control_artifact


def _build_live_artifacts(
    services: SubmissionServices,
    *,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
) -> tuple[_JsonArtifact, _JsonArtifact, _ControllerFacts]:
    baseline, facts = _build_baseline_artifact(
        services,
        local=local,
        intent=intent,
    )
    control_ready = _build_control_ready_artifact(
        services,
        local=local,
        intent=intent,
        baseline_artifact=baseline,
    )
    return baseline, control_ready, facts


def _as_acquisition_artifact(value: _JsonArtifact) -> ImmutableJsonArtifact:
    return ImmutableJsonArtifact(
        key=value.key,
        raw=value.raw,
        file_sha256=value.file_sha256,
    )


def _require_remote_published_sources(
    services: SubmissionServices,
    *,
    local: _LocalAuthorities,
) -> None:
    bucket = str(local.descriptor.value["bucket"])
    for artifact in (
        local.cache_acceptance,
        local.spend_snapshot,
        local.rehearsal,
        local.readiness,
    ):
        remote = _get_exact(services.s3, bucket=bucket, key=artifact.key)
        if remote != artifact.raw:
            raise SubmissionIntegrationError(
                f"published immutable source drift: {artifact.key}"
            )


def _validate_prepared_baseline(
    *,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
    baseline: _JsonArtifact,
    acquired_at: datetime,
) -> None:
    value = baseline.value
    validate_controller_baseline(
        value,
        descriptor=local.descriptor.value,
        descriptor_key=local.descriptor.key,
        descriptor_raw=local.descriptor.raw,
        qualification_submission_ready=local.readiness.value,
        qualification_submission_ready_key=local.readiness.key,
        qualification_submission_ready_raw=local.readiness.raw,
        intent=intent.value,
        intent_key=intent.key,
        intent_raw=intent.raw,
        controller_baseline_key=baseline.key,
        controller_baseline_raw=baseline.raw,
        controller_instance_id=str(value.get("controller_instance_id")),
        controller_instance_type=str(value.get("controller_instance_type")),
        controller_profile_arn=str(value.get("controller_profile_arn")),
        controller_cluster_name=str(value.get("controller_cluster_name")),
        ssm_ping_status=str(value.get("ssm_ping_status")),
        exact_name_history=value.get("exact_name_history"),  # type: ignore[arg-type]
        active_exact_name_job_ids=value.get(  # type: ignore[arg-type]
            "active_exact_name_job_ids"
        ),
        active_tagged_p5_instance_ids=value.get(  # type: ignore[arg-type]
            "active_tagged_p5_instance_ids"
        ),
        observed_at=str(value.get("observed_at")),
        acquired_at=acquired_at,
    )


def _load_prepared_artifacts(
    request: QualificationRequest,
    services: SubmissionServices,
    *,
    local: _LocalAuthorities,
) -> tuple[_JsonArtifact, _JsonArtifact, _JsonArtifact]:
    if request.intent_s3_uri is None or request.must_start_ready_s3_uri is None:
        raise SubmissionUsageError(
            "acquire-and-launch requires --intent-s3-uri and --must-start-ready-s3-uri"
        )
    bucket = str(local.descriptor.value["bucket"])
    intent_bucket, intent_key = _parse_s3_uri(
        request.intent_s3_uri,
        label="intent S3 URI",
    )
    ready_bucket, ready_key = _parse_s3_uri(
        request.must_start_ready_s3_uri,
        label="must-start readiness S3 URI",
    )
    if intent_bucket != bucket or ready_bucket != bucket:
        raise SubmissionIntegrationError("prepared authority bucket is foreign")
    intent = _fetch_artifact(
        services,
        bucket=bucket,
        key=intent_key,
        label="submission intent",
    )
    expected_intent_key = submission_intent_s3_key(
        run_id=str(intent.value.get("run_id")),
        managed_mode=str(intent.value.get("managed_mode")),
        intent_body_sha256=str(intent.value.get(INTENT_DIGEST_FIELD)),
    )
    if intent.key != expected_intent_key:
        raise SubmissionIntegrationError("submission intent key is foreign")
    validate_submission_intent(
        intent.value,
        descriptor=local.descriptor.value,
        cache_seed_acceptance=local.cache_acceptance.value,
        gpu_spend_snapshot=local.spend_snapshot.value,
        staged_readiness=local.staged.value,
        rehearsal_evidence=local.rehearsal.value,
        qualification_submission_ready=local.readiness.value,
    )
    if intent.raw != canonical_file_bytes(intent.value):
        raise SubmissionIntegrationError("submission intent bytes are noncanonical")
    reviewed_ready = _fetch_artifact(
        services,
        bucket=bucket,
        key=ready_key,
        label="reviewed must-start control-plane readiness",
    )
    expected_ready_key = must_start_control_plane_ready_s3_key(reviewed_ready.value)
    if reviewed_ready.key != expected_ready_key:
        raise SubmissionIntegrationError("reviewed must-start readiness key is foreign")
    if (
        reviewed_ready.value.get("intent_key") != intent.key
        or reviewed_ready.value.get("intent_file_sha256") != intent.file_sha256
        or reviewed_ready.value.get("intent_body_sha256")
        != intent.value[INTENT_DIGEST_FIELD]
    ):
        raise SubmissionIntegrationError(
            "reviewed must-start readiness selects a foreign intent"
        )
    baseline_key = reviewed_ready.value.get("controller_baseline_key")
    if not isinstance(baseline_key, str):
        raise SubmissionIntegrationError(
            "reviewed must-start readiness baseline key is malformed"
        )
    reviewed_baseline = _fetch_artifact(
        services,
        bucket=bucket,
        key=baseline_key,
        label="reviewed controller baseline",
    )
    baseline = reviewed_baseline.value
    _validate_prepared_baseline(
        local=local,
        intent=intent,
        baseline=reviewed_baseline,
        acquired_at=_now(services),
    )
    identity_fields = (
        "stack_id",
        "template_sha256",
        "lambda_function_arn",
        "lambda_code_sha256",
        "iam_policy_sha256",
        "reconciliation_rule_arn",
        "deadline_schedule_arn",
        "dlq_arn",
    )
    deployment_identity = {
        field: reviewed_ready.value.get(field) for field in identity_fields
    }
    validate_must_start_control_plane_ready(
        reviewed_ready.value,
        descriptor=local.descriptor.value,
        descriptor_key=local.descriptor.key,
        descriptor_raw=local.descriptor.raw,
        qualification_submission_ready=local.readiness.value,
        qualification_submission_ready_key=local.readiness.key,
        qualification_submission_ready_raw=local.readiness.raw,
        intent=intent.value,
        intent_key=intent.key,
        intent_raw=intent.raw,
        controller_baseline=baseline,
        controller_baseline_key=reviewed_baseline.key,
        controller_baseline_raw=reviewed_baseline.raw,
        must_start_control_plane_ready_key=reviewed_ready.key,
        must_start_control_plane_ready_raw=reviewed_ready.raw,
        controller_instance_id=str(baseline.get("controller_instance_id")),
        controller_instance_type=str(baseline.get("controller_instance_type")),
        controller_profile_arn=str(baseline.get("controller_profile_arn")),
        controller_cluster_name=str(baseline.get("controller_cluster_name")),
        baseline_ssm_ping_status=str(baseline.get("ssm_ping_status")),
        baseline_exact_name_history=baseline.get(  # type: ignore[arg-type]
            "exact_name_history"
        ),
        baseline_active_exact_name_job_ids=baseline.get(  # type: ignore[arg-type]
            "active_exact_name_job_ids"
        ),
        baseline_active_tagged_p5_instance_ids=baseline.get(  # type: ignore[arg-type]
            "active_tagged_p5_instance_ids"
        ),
        baseline_observed_at=str(baseline.get("observed_at")),
        reviewed_deployment_identity=deployment_identity,
        observed_deployment_identity=deployment_identity,
        reconciliation_rule_state=str(
            reviewed_ready.value.get("reconciliation_rule_state")
        ),
        deadline_schedule_state=str(
            reviewed_ready.value.get("deadline_schedule_state")
        ),
        coordinator_mode=str(reviewed_ready.value.get("coordinator_mode")),
        observed_at=str(reviewed_ready.value.get("observed_at")),
        acquired_at=_now(services),
    )
    return intent, reviewed_baseline, reviewed_ready


def _acquisition_read_result(
    services: SubmissionServices,
    *,
    bucket: str,
    key: str,
    observed_at: datetime,
) -> SubmissionAcquisitionReadResult:
    raw = _get_exact(services.s3, bucket=bucket, key=key)
    prefix = key.rsplit("/", 1)[0] + "/"
    keys = _list_complete(services.s3, bucket=bucket, prefix=prefix)
    observed = _whole_second(observed_at, field="acquisition read time")[0]
    if raw is None:
        return SubmissionAcquisitionReadResult(
            requested_key=key,
            result="exact-404",
            http_status=404,
            artifact=None,
            listed_prefix=prefix,
            prefix_keys=keys,
            listing_complete=True,
            observed_at=observed,
        )
    artifact = _decode_artifact_raw(
        key=key,
        raw=raw,
        label="stored submission acquisition",
    )
    return SubmissionAcquisitionReadResult(
        requested_key=key,
        result="found",
        http_status=200,
        artifact=_as_acquisition_artifact(artifact),
        listed_prefix=prefix,
        prefix_keys=keys,
        listing_complete=True,
        observed_at=observed,
    )


def _deployment_identity_from_ready(
    ready: Mapping[str, object],
) -> dict[str, object]:
    return {
        field: ready[field]
        for field in (
            "stack_id",
            "template_sha256",
            "lambda_function_arn",
            "lambda_code_sha256",
            "iam_policy_sha256",
            "reconciliation_rule_arn",
            "deadline_schedule_arn",
            "dlq_arn",
        )
    }


def _require_final_live_equality(
    services: SubmissionServices,
    *,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
    baseline: _JsonArtifact,
    control_ready: _JsonArtifact,
    original_facts: _ControllerFacts,
) -> None:
    _require_remote_static(services, local=local)
    current = _capture_controller_facts(
        services,
        descriptor=local.descriptor.value,
        intent=intent.value,
    )
    comparable = (
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
        "ssm_ping_status",
        "exact_name_history",
        "active_exact_name_job_ids",
        "active_tagged_p5_instance_ids",
    )
    if any(
        getattr(current, field) != getattr(original_facts, field)
        for field in comparable
    ):
        raise SubmissionIntegrationError(
            "controller/P5 facts changed after acquisition"
        )
    deployment = services.cloudformation.inspect_dynamic_must_start(
        run_id=str(intent.value["run_id"]),
        managed_mode=str(intent.value["managed_mode"]),
        intent_body_sha256=str(intent.value[INTENT_DIGEST_FIELD]),
        controller_baseline_body_sha256=str(baseline.value["baseline_body_sha256"]),
    )
    if (
        not isinstance(deployment, Mapping)
        or deployment.get("reviewed_deployment_identity")
        != _deployment_identity_from_ready(control_ready.value)
        or deployment.get("observed_deployment_identity")
        != _deployment_identity_from_ready(control_ready.value)
        or deployment.get("reconciliation_rule_state") != "ENABLED"
        or deployment.get("deadline_schedule_state") != "ENABLED"
        or deployment.get("coordinator_mode") != COORDINATOR_MODE
    ):
        raise SubmissionIntegrationError(
            "must-start deployment changed after acquisition"
        )
    _, deadline = _whole_second(
        str(intent.value["must_start_by"]),
        field="must_start_by",
    )
    if _now(services) >= deadline:
        raise SubmissionIntegrationError("submission deadline expired before launch")


def _build_acquisition_artifact(
    *,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
    baseline: _JsonArtifact,
    control_ready: _JsonArtifact,
    acquired_at: datetime,
) -> _JsonArtifact:
    acquisition = build_submission_acquired(
        intent=_as_acquisition_artifact(intent),
        qualification_submission_ready=_as_acquisition_artifact(local.readiness),
        controller_baseline=_as_acquisition_artifact(baseline),
        must_start_control_plane_ready=_as_acquisition_artifact(control_ready),
        acquired_at=acquired_at,
    )
    key = submission_acquired_s3_key(
        run_id=str(acquisition["run_id"]),
        managed_mode=str(acquisition["managed_mode"]),
        descriptor_file_sha256=str(acquisition["descriptor_file_sha256"]),
    )
    return _JsonArtifact(
        key=key,
        raw=submission_acquired_file_bytes(acquisition),
        value=acquisition,
    )


def _put_acquisition_once(
    services: SubmissionServices,
    *,
    bucket: str,
    acquisition: _JsonArtifact,
    run_id: str,
) -> bool:
    """Attempt the selector once; only an authenticated success returns True."""

    checksum = base64.b64encode(hashlib.sha256(acquisition.raw).digest()).decode(
        "ascii"
    )
    try:
        response = services.s3.put_object(
            Bucket=bucket,
            Key=acquisition.key,
            Body=acquisition.raw,
            IfNoneMatch="*",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum,
            ContentType="application/json",
            Metadata={
                "glm52-run-id": run_id,
                "glm52-body-sha256": _body_digest(acquisition.value),
            },
        )
    except Exception as error:
        # A 412, timeout, connection loss, or any other ordinary exception is
        # ownership-ambiguous.  Authentication may recover a winner, but the
        # caller is reconciliation-only forever.
        winner = _get_exact(
            services.s3,
            bucket=bucket,
            key=acquisition.key,
        )
        if winner is None:
            code, _ = _aws_error(error)
            raise SubmissionIntegrationError(
                "acquisition put failed without an authenticated durable "
                f"winner: {code or error}"
            ) from error
        _decode_artifact_raw(
            key=acquisition.key,
            raw=winner,
            label="ambiguous acquisition winner",
        )
        return False
    if not isinstance(response, Mapping):
        raise SubmissionIntegrationError(
            "acquisition conditional put response is malformed"
        )
    metadata = response.get("ResponseMetadata")
    if not isinstance(metadata, Mapping) or metadata.get("HTTPStatusCode") not in (
        200,
        201,
    ):
        # A malformed/non-success response is not launch authority even when
        # the service may have committed the object.
        winner = _get_exact(
            services.s3,
            bucket=bucket,
            key=acquisition.key,
        )
        if winner is None:
            raise SubmissionIntegrationError(
                "acquisition put returned no service success and no winner"
            )
        return False
    return True


def _selected_sources_from_acquisition(
    services: SubmissionServices,
    *,
    bucket: str,
    descriptor: _JsonArtifact,
    acquisition: _JsonArtifact,
    now: datetime,
) -> _SelectedSubmission:
    value = acquisition.value
    if (
        value.get("run_id") != descriptor.value.get("run_id")
        or value.get("managed_mode") != QUALIFICATION_MODE
        or value.get("descriptor_key") != descriptor.key
        or value.get("descriptor_file_sha256") != descriptor.file_sha256
        or value.get("descriptor_body_sha256")
        != descriptor.value.get("descriptor_body_sha256")
    ):
        raise SubmissionIntegrationError(
            "stored acquisition selector is foreign to the pinned descriptor"
        )
    fields = {
        "intent": "intent_key",
        "readiness": "qualification_submission_ready_key",
        "baseline": "controller_baseline_key",
        "control_ready": "must_start_control_plane_ready_key",
    }
    fetched: dict[str, _JsonArtifact] = {}
    for label, field in fields.items():
        key = value.get(field)
        if not isinstance(key, str):
            raise SubmissionIntegrationError(f"stored acquisition {field} is malformed")
        fetched[label] = _fetch_artifact(
            services,
            bucket=bucket,
            key=key,
            label=f"selected {label.replace('_', ' ')}",
        )
    intent = fetched["intent"]
    readiness = fetched["readiness"]
    baseline = fetched["baseline"]
    control_ready = fetched["control_ready"]
    acquired_iso, acquired_at = _whole_second(
        str(value.get("acquired_at")),
        field="submission acquisition acquired_at",
    )
    if acquired_at > now:
        raise SubmissionIntegrationError(
            "stored acquisition selector is from the future"
        )
    validate_submission_acquired(
        _as_acquisition_artifact(acquisition),
        intent=_as_acquisition_artifact(intent),
        qualification_submission_ready=_as_acquisition_artifact(readiness),
        controller_baseline=_as_acquisition_artifact(baseline),
        must_start_control_plane_ready=_as_acquisition_artifact(control_ready),
        # Structural/source authentication remains valid after the deadline.
        # Current-time expiry is applied by the acquisition decision and the
        # exact controller reconciler; accepted retries stay idempotent.
        now=acquired_iso,
    )
    upstream_keys = {
        "cache_acceptance": intent.value.get("cache_seed_acceptance_key"),
        "spend_snapshot": intent.value.get("gpu_spend_snapshot_key"),
        "staged": readiness.value.get("staged_readiness_key"),
        "rehearsal": readiness.value.get("rehearsal_evidence_key"),
    }
    upstream: dict[str, _JsonArtifact] = {}
    for label, key in upstream_keys.items():
        if not isinstance(key, str):
            raise SubmissionIntegrationError(
                f"selected {label.replace('_', ' ')} key is malformed"
            )
        upstream[label] = _fetch_artifact(
            services,
            bucket=bucket,
            key=key,
            label=f"selected {label.replace('_', ' ')}",
        )
    validate_submission_intent(
        intent.value,
        descriptor=descriptor.value,
        cache_seed_acceptance=upstream["cache_acceptance"].value,
        gpu_spend_snapshot=upstream["spend_snapshot"].value,
        staged_readiness=upstream["staged"].value,
        rehearsal_evidence=upstream["rehearsal"].value,
        qualification_submission_ready=readiness.value,
    )
    selected = _SelectedSubmission(
        descriptor=descriptor,
        staged=upstream["staged"],
        rehearsal=upstream["rehearsal"],
        cache_acceptance=upstream["cache_acceptance"],
        spend_snapshot=upstream["spend_snapshot"],
        readiness=readiness,
        intent=intent,
        baseline=baseline,
        control_ready=control_ready,
        acquisition=acquisition,
    )
    _authenticate_remote_staged_chain(
        services,
        descriptor=selected.descriptor,
        staged=selected.staged,
        rehearsal=selected.rehearsal,
        readiness=selected.readiness,
    )
    return selected


def _controller_job_from_row(
    *,
    row: Mapping[str, object],
    descriptor: Mapping[str, object],
) -> dict[str, object]:
    return {
        "sky_job_id": row["sky_job_id"],
        "sky_job_name": row["sky_job_name"],
        "workspace": row["workspace"],
        "controller_submitted_at": row["controller_submitted_at"],
        "controller_status": row["controller_status"],
        "controller_identity": descriptor["controller_identity"],
    }


def _reconcile_controller_job(
    services: SubmissionServices,
    *,
    selected: _SelectedSubmission,
) -> tuple[dict[str, object], _ControllerFacts]:
    facts = _capture_controller_facts(
        services,
        descriptor=selected.descriptor.value,
        intent=selected.intent.value,
    )
    _require_controller_coordinates_match_baseline(
        facts=facts,
        baseline=selected.baseline.value,
    )
    baseline_history = selected.baseline.value.get("exact_name_history")
    if not isinstance(baseline_history, list):
        raise SubmissionIntegrationError(
            "selected controller baseline history is malformed"
        )
    baseline_by_id: dict[int, dict[str, object]] = {}
    for raw in baseline_history:
        if not isinstance(raw, Mapping) or type(raw.get("sky_job_id")) is not int:
            raise SubmissionIntegrationError(
                "selected controller baseline history is malformed"
            )
        job_id = int(raw["sky_job_id"])
        if job_id in baseline_by_id:
            raise SubmissionIntegrationError(
                "selected controller baseline history is duplicate"
            )
        baseline_by_id[job_id] = dict(raw)
    current_by_id = {int(row["sky_job_id"]): row for row in facts.exact_name_history}
    for job_id, baseline_row in baseline_by_id.items():
        if current_by_id.get(job_id) != baseline_row:
            raise SubmissionIntegrationError(
                "controller history present at acquisition changed"
            )
    new_rows = [
        row
        for row in facts.detailed_history
        if int(row["sky_job_id"]) not in baseline_by_id
    ]
    if not new_rows:
        _, deadline = _whole_second(
            str(selected.intent.value["must_start_by"]),
            field="must_start_by",
        )
        if _now(services) >= deadline:
            raise SubmissionIntegrationError(
                "no controller acceptance appeared before must_start_by"
            )
        raise SubmissionReconcilePending(
            "exact controller acceptance is not visible yet"
        )
    if len(new_rows) != 1:
        raise SubmissionIntegrationError(
            "controller reconciliation found multiple new exact-name jobs"
        )
    row = new_rows[0]
    _, deadline = _whole_second(
        str(selected.intent.value["must_start_by"]),
        field="must_start_by",
    )
    if (
        row["controller_status"] in {"SUBMITTED", "WINDING_DOWN"}
        or row["controller_submitted_at"] is None
    ):
        if _now(services) >= deadline:
            raise SubmissionIntegrationError(
                "controller row never reached a receipt-ready state before deadline"
            )
        raise SubmissionReconcilePending(
            "exact controller job exists but has not reached receipt-ready state"
        )
    submitted_iso, submitted_at = _controller_time(
        row["controller_submitted_at"],
        field="controller submitted_at",
    )
    assert submitted_iso is not None and submitted_at is not None
    _, intent_at = _whole_second(
        str(selected.intent.value["intent_at"]),
        field="intent_at",
    )
    if submitted_at < intent_at or submitted_at >= deadline:
        raise SubmissionIntegrationError(
            "reconciled controller submission time is outside the intent window"
        )
    return (
        _controller_job_from_row(
            row=row,
            descriptor=selected.descriptor.value,
        ),
        facts,
    )


def _observation_key(
    *,
    intent: Mapping[str, object],
    observation_body_sha256: str,
) -> str:
    return (
        f"campaigns/{intent['run_id']}/monitor/must-start/"
        f"{intent['managed_mode']}/{intent[INTENT_DIGEST_FIELD]}/"
        f"observations/{observation_body_sha256}.json"
    )


def _must_start_intent_prefix(*, intent: Mapping[str, object]) -> str:
    return (
        f"campaigns/{intent['run_id']}/monitor/must-start/"
        f"{intent['managed_mode']}/{intent[INTENT_DIGEST_FIELD]}/"
    )


def _legacy_binding_key(*, intent: Mapping[str, object]) -> str:
    return _must_start_intent_prefix(intent=intent) + "JOB_BINDING.json"


def _accepted_prefix(*, intent: Mapping[str, object]) -> str:
    return (
        f"campaigns/{intent['run_id']}/submissions/{intent['managed_mode']}/accepted/"
    )


def _require_observation_progress_compatible(
    *,
    observation: Mapping[str, object],
    current_row: Mapping[str, object],
) -> None:
    stored_status = observation.get("status")
    current_status = current_row.get("controller_status")
    allowed_statuses = (
        _CONTROLLER_STATUS_TRANSITIONS.get(stored_status)
        if isinstance(stored_status, str)
        else None
    )
    if (
        not isinstance(stored_status, str)
        or not isinstance(current_status, str)
        or allowed_statuses is None
        or current_status not in allowed_statuses
    ):
        raise SubmissionIntegrationError(
            "stored controller observation is not a legal status predecessor"
        )

    stored_schedule = observation.get("schedule_state")
    current_schedule = current_row.get("schedule_state")
    allowed_schedules = (
        _SCHEDULE_STATE_TRANSITIONS.get(stored_schedule)
        if isinstance(stored_schedule, str)
        else None
    )
    if (
        not isinstance(stored_schedule, str)
        or not isinstance(current_schedule, str)
        or allowed_schedules is None
        or current_schedule not in allowed_schedules
    ):
        raise SubmissionIntegrationError(
            "stored controller observation is not a legal schedule predecessor"
        )

    stored_recovery = observation.get("recovery_count")
    current_recovery = current_row.get("recovery_count")
    if (
        type(stored_recovery) is not int
        or type(current_recovery) is not int
        or stored_recovery < 0
        or current_recovery < stored_recovery
    ):
        raise SubmissionIntegrationError(
            "stored controller observation has impossible recovery progress"
        )

    stored_start, _ = _controller_time(
        observation.get("start_at"),
        field="stored controller observation start_at",
        permit_none=True,
    )
    current_start, _ = _controller_time(
        current_row.get("start_at"),
        field="current controller observation start_at",
        permit_none=True,
    )
    if stored_start is not None and stored_start != current_start:
        raise SubmissionIntegrationError(
            "stored controller observation changed an established start_at"
        )

    stored_worker = observation.get("worker_cluster_name")
    current_worker = current_row.get("worker_cluster_name")
    if (
        stored_recovery == current_recovery
        and stored_worker is not None
        and stored_worker != current_worker
    ):
        raise SubmissionIntegrationError(
            "stored controller observation changed worker identity without recovery"
        )


def _controller_job_from_observation(
    *,
    observation: Mapping[str, object],
    descriptor: Mapping[str, object],
) -> dict[str, object]:
    return {
        "sky_job_id": observation["target_job_id"],
        "sky_job_name": observation["sky_job_name"],
        "workspace": observation["workspace"],
        "controller_submitted_at": observation["submitted_at"],
        "controller_status": observation["status"],
        "controller_identity": descriptor["controller_identity"],
    }


def _dynamic_current_history(
    *,
    selected: _SelectedSubmission,
    facts: _ControllerFacts,
) -> list[dict[str, object]]:
    baseline_history = selected.baseline.value.get("exact_name_history")
    if type(baseline_history) is not list:
        raise SubmissionIntegrationError(
            "selected controller baseline history is malformed"
        )
    baseline_ids = {
        int(row["sky_job_id"])
        for row in baseline_history
        if isinstance(row, Mapping) and type(row.get("sky_job_id")) is int
    }
    if len(baseline_ids) != len(baseline_history):
        raise SubmissionIntegrationError(
            "selected controller baseline history is malformed"
        )
    post_baseline = [
        _controller_job_from_row(
            row=row,
            descriptor=selected.descriptor.value,
        )
        for row in facts.detailed_history
        if int(row["sky_job_id"]) not in baseline_ids
    ]
    return [*(dict(row) for row in baseline_history), *post_baseline]


def _require_no_legacy_binding(
    services: SubmissionServices,
    *,
    bucket: str,
    intent: Mapping[str, object],
) -> None:
    legacy_key = _legacy_binding_key(intent=intent)
    members = _list_complete(
        services.s3,
        bucket=bucket,
        prefix=_must_start_intent_prefix(intent=intent),
    )
    if legacy_key in members:
        raise SubmissionIntegrationError(
            "legacy must-start binding exists for dynamic-v2 qualification intent"
        )


def _validate_dynamic_binding_artifact(
    *,
    selected: _SelectedSubmission,
    binding: _JsonArtifact,
    current_history: list[dict[str, object]],
    current_observed_at: str,
) -> _JsonArtifact:
    expected_key = dynamic_v2_job_binding_s3_key(intent=selected.intent.value)
    if binding.key != expected_key:
        raise SubmissionIntegrationError("dynamic-v2 binding key is not exact")
    try:
        _, observed_at = _controller_time(
            current_observed_at,
            field="controller history observed_at",
        )
        _, deadline = _whole_second(
            str(selected.intent.value["must_start_by"]),
            field="must_start_by",
        )
        assert observed_at is not None
        validation_now = (
            current_observed_at
            if observed_at < deadline
            else str(binding.value.get("bound_at"))
        )
        value = validate_dynamic_v2_job_binding(
            binding.value,
            intent=selected.intent.value,
            controller_baseline=selected.baseline.value,
            acquisition=selected.acquisition.value,
            descriptor_controller_identity=str(
                selected.descriptor.value["controller_identity"]
            ),
            current_exact_name_history=current_history,
            now=validation_now,
        )
    except DynamicJobBindingValidationError as error:
        raise SubmissionIntegrationError(
            "dynamic-v2 binding winner failed authentication"
        ) from error
    if binding.raw != dynamic_v2_canonical_bytes(value):
        raise SubmissionIntegrationError(
            "dynamic-v2 binding winner bytes are noncanonical"
        )
    return _JsonArtifact(key=binding.key, raw=binding.raw, value=value)


def _publish_or_load_dynamic_binding(
    services: SubmissionServices,
    *,
    selected: _SelectedSubmission,
    observation: _JsonArtifact,
    accepted_controller_job: Mapping[str, object],
    facts: _ControllerFacts,
) -> _JsonArtifact:
    intent = selected.intent.value
    bucket = str(selected.descriptor.value["bucket"])
    run_id = str(intent["run_id"])
    binding_key = dynamic_v2_job_binding_s3_key(intent=intent)
    _require_no_legacy_binding(
        services,
        bucket=bucket,
        intent=intent,
    )
    current_history = _dynamic_current_history(selected=selected, facts=facts)
    winner_raw = _get_exact(services.s3, bucket=bucket, key=binding_key)
    if winner_raw is None:
        _, observed_at = _controller_time(
            facts.observed_at,
            field="controller history observed_at",
        )
        _, deadline = _whole_second(
            str(intent["must_start_by"]),
            field="must_start_by",
        )
        assert observed_at is not None
        if observed_at >= deadline:
            raise SubmissionIntegrationError(
                "submission deadline passed before a dynamic-v2 binding existed"
            )
        baseline_history = selected.baseline.value.get("exact_name_history")
        if type(baseline_history) is not list:
            raise SubmissionIntegrationError(
                "selected controller baseline history is malformed"
            )
        candidate_value = build_dynamic_v2_job_binding(
            intent=intent,
            controller_baseline=selected.baseline.value,
            acquisition=selected.acquisition.value,
            descriptor_controller_identity=str(
                selected.descriptor.value["controller_identity"]
            ),
            current_exact_name_history=[
                *(dict(row) for row in baseline_history),
                dict(accepted_controller_job),
            ],
            bound_at=str(observation.value["observed_at"]),
            now=facts.observed_at,
        )
        candidate_raw = dynamic_v2_canonical_bytes(candidate_value)
        checksum = base64.b64encode(hashlib.sha256(candidate_raw).digest()).decode(
            "ascii"
        )
        try:
            services.s3.put_object(
                Bucket=bucket,
                Key=binding_key,
                Body=candidate_raw,
                IfNoneMatch="*",
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=checksum,
                ContentType="application/json",
                Metadata={
                    "glm52-run-id": run_id,
                    "glm52-body-sha256": str(
                        candidate_value["job_binding_body_sha256"]
                    ),
                },
            )
        except Exception:
            pass
        winner_raw = _get_exact(services.s3, bucket=bucket, key=binding_key)
        if winner_raw is None:
            raise SubmissionIntegrationError(
                "dynamic-v2 binding conditional create produced no winner"
            )
    winner = _decode_artifact_raw(
        key=binding_key,
        raw=winner_raw,
        label="dynamic-v2 job binding",
        newline=False,
    )
    authenticated = _validate_dynamic_binding_artifact(
        selected=selected,
        binding=winner,
        current_history=current_history,
        current_observed_at=facts.observed_at,
    )
    _require_no_legacy_binding(
        services,
        bucket=bucket,
        intent=intent,
    )
    return authenticated


def _validate_or_publish_receipts(
    services: SubmissionServices,
    *,
    selected: _SelectedSubmission,
    controller_job: Mapping[str, object],
    facts: _ControllerFacts,
) -> tuple[dict[str, object], _JsonArtifact, _JsonArtifact]:
    intent = selected.intent.value
    bucket = str(selected.descriptor.value["bucket"])
    run_id = str(intent["run_id"])
    job_id = int(controller_job["sky_job_id"])
    matching = [row for row in facts.detailed_history if row["sky_job_id"] == job_id]
    if len(matching) != 1:
        raise SubmissionIntegrationError(
            "numeric controller observation is missing or ambiguous"
        )
    row = matching[0]
    observations_prefix = (
        f"campaigns/{run_id}/monitor/must-start/{intent['managed_mode']}/"
        f"{intent[INTENT_DIGEST_FIELD]}/observations/"
    )
    existing_observation_keys = _list_complete(
        services.s3,
        bucket=bucket,
        prefix=observations_prefix,
    )
    if len(existing_observation_keys) > 1:
        raise SubmissionIntegrationError(
            "multiple controller observations exist for selected intent"
        )
    if existing_observation_keys:
        observation = _fetch_artifact(
            services,
            bucket=bucket,
            key=existing_observation_keys[0],
            label="controller observation",
            newline=False,
        )
        validate_must_start_controller_observation(observation.value)
        expected_observation = build_must_start_controller_observation(
            run_id=run_id,
            managed_mode=str(intent["managed_mode"]),
            account_id=str(intent["account_id"]),
            region=str(intent["region"]),
            bucket=bucket,
            descriptor_body_sha256=str(intent["descriptor_body_sha256"]),
            submission_body_sha256=str(intent[INTENT_DIGEST_FIELD]),
            sky_job_name=str(intent["sky_job_name"]),
            must_start_by=str(intent["must_start_by"]),
            target_job_id=job_id,
            workspace=str(row["workspace"]),
            controller_instance_id=facts.controller_instance_id,
            controller_instance_type=facts.controller_instance_type,
            controller_profile_arn=facts.controller_profile_arn,
            controller_cluster_name=facts.controller_cluster_name,
            status=str(observation.value.get("status")),
            schedule_state=str(observation.value.get("schedule_state")),
            submitted_at=str(observation.value.get("submitted_at")),
            start_at=observation.value.get("start_at"),  # type: ignore[arg-type]
            worker_cluster_name=observation.value.get(  # type: ignore[arg-type]
                "worker_cluster_name"
            ),
            recovery_count=int(observation.value.get("recovery_count", -1)),
            observed_at=str(observation.value.get("observed_at")),
        )
        _, stored_observed_at = _controller_time(
            observation.value.get("observed_at"),
            field="stored controller observation observed_at",
        )
        _, current_observed_at = _controller_time(
            facts.observed_at,
            field="current controller observation observed_at",
        )
        if (
            observation.value != expected_observation
            or observation.value.get("submitted_at") != row["controller_submitted_at"]
            or stored_observed_at is None
            or current_observed_at is None
            or stored_observed_at > current_observed_at
            or observation.key
            != _observation_key(
                intent=intent,
                observation_body_sha256=str(
                    observation.value["observation_body_sha256"]
                ),
            )
        ):
            raise SubmissionIntegrationError(
                "stored controller observation is not the fully authenticated winner"
            )
        _require_observation_progress_compatible(
            observation=observation.value,
            current_row=row,
        )
        accepted_controller_job = _controller_job_from_observation(
            observation=observation.value,
            descriptor=selected.descriptor.value,
        )
    else:
        observation_value = build_must_start_controller_observation(
            run_id=run_id,
            managed_mode=str(intent["managed_mode"]),
            account_id=str(intent["account_id"]),
            region=str(intent["region"]),
            bucket=bucket,
            descriptor_body_sha256=str(intent["descriptor_body_sha256"]),
            submission_body_sha256=str(intent[INTENT_DIGEST_FIELD]),
            sky_job_name=str(intent["sky_job_name"]),
            must_start_by=str(intent["must_start_by"]),
            target_job_id=job_id,
            workspace=str(row["workspace"]),
            controller_instance_id=facts.controller_instance_id,
            controller_instance_type=facts.controller_instance_type,
            controller_profile_arn=facts.controller_profile_arn,
            controller_cluster_name=facts.controller_cluster_name,
            status=str(row["controller_status"]),
            schedule_state=str(row["schedule_state"]),
            submitted_at=str(row["controller_submitted_at"]),
            start_at=row["start_at"],  # type: ignore[arg-type]
            worker_cluster_name=row["worker_cluster_name"],  # type: ignore[arg-type]
            recovery_count=int(row["recovery_count"]),
            observed_at=facts.observed_at,
        )
        observation = _JsonArtifact(
            key=_observation_key(
                intent=intent,
                observation_body_sha256=str(
                    observation_value["observation_body_sha256"]
                ),
            ),
            raw=must_start_canonical_bytes(observation_value),
            value=observation_value,
        )
        _put_immutable(
            services,
            bucket=bucket,
            artifact=observation,
            run_id=run_id,
            newline=False,
        )
        accepted_controller_job = dict(controller_job)

    binding = _publish_or_load_dynamic_binding(
        services,
        selected=selected,
        observation=observation,
        accepted_controller_job=accepted_controller_job,
        facts=facts,
    )
    return accepted_controller_job, observation, binding


def _validate_existing_accepted(
    *,
    selected: _SelectedSubmission,
    accepted: _JsonArtifact,
) -> dict[str, object]:
    value = validate_submission_accepted(
        accepted.value,
        intent=selected.intent.value,
        descriptor=selected.descriptor.value,
        cache_seed_acceptance=selected.cache_acceptance.value,
        gpu_spend_snapshot=selected.spend_snapshot.value,
        staged_readiness=selected.staged.value,
        rehearsal_evidence=selected.rehearsal.value,
        qualification_submission_ready=selected.readiness.value,
        controller_baseline=selected.baseline.value,
        submission_acquisition=selected.acquisition.value,
    )
    expected_key = submission_accepted_s3_key(
        run_id=str(selected.intent.value["run_id"]),
        managed_mode=str(selected.intent.value["managed_mode"]),
        accepted_body_sha256=str(value[ACCEPTED_DIGEST_FIELD]),
    )
    if accepted.key != expected_key or accepted.raw != canonical_file_bytes(value):
        raise SubmissionIntegrationError(
            "stored accepted authority key or bytes are invalid"
        )
    return value


def _load_unique_accepted_if_present(
    services: SubmissionServices,
    *,
    selected: _SelectedSubmission,
) -> _JsonArtifact | None:
    bucket = str(selected.descriptor.value["bucket"])
    keys = _list_complete(
        services.s3,
        bucket=bucket,
        prefix=_accepted_prefix(intent=selected.intent.value),
    )
    if len(keys) > 1:
        raise SubmissionIntegrationError(
            "multiple accepted authorities exist for qualification"
        )
    if not keys:
        return None
    accepted = _fetch_artifact(
        services,
        bucket=bucket,
        key=keys[0],
        label="submission accepted",
    )
    _validate_existing_accepted(selected=selected, accepted=accepted)
    return accepted


def _publish_or_load_accepted(
    services: SubmissionServices,
    *,
    selected: _SelectedSubmission,
    controller_job: Mapping[str, object],
    observation: _JsonArtifact,
    binding: _JsonArtifact,
) -> _JsonArtifact:
    bucket = str(selected.descriptor.value["bucket"])
    prefix = _accepted_prefix(intent=selected.intent.value)
    existing = _load_unique_accepted_if_present(
        services,
        selected=selected,
    )
    if existing is not None:
        _require_no_legacy_binding(
            services,
            bucket=bucket,
            intent=selected.intent.value,
        )
        return existing
    value = build_submission_accepted(
        intent=selected.intent.value,
        descriptor=selected.descriptor.value,
        cache_seed_acceptance=selected.cache_acceptance.value,
        gpu_spend_snapshot=selected.spend_snapshot.value,
        staged_readiness=selected.staged.value,
        rehearsal_evidence=selected.rehearsal.value,
        qualification_submission_ready=selected.readiness.value,
        controller_baseline=selected.baseline.value,
        submission_acquisition=selected.acquisition.value,
        controller_job=controller_job,
        controller_observation=observation.value,
        controller_observation_key=observation.key,
        controller_observation_file_sha256=observation.file_sha256,
        job_binding=binding.value,
        job_binding_key=binding.key,
        job_binding_file_sha256=binding.file_sha256,
    )
    accepted = _JsonArtifact(
        key=submission_accepted_s3_key(
            run_id=str(selected.intent.value["run_id"]),
            managed_mode=str(selected.intent.value["managed_mode"]),
            accepted_body_sha256=str(value[ACCEPTED_DIGEST_FIELD]),
        ),
        raw=canonical_file_bytes(value),
        value=value,
    )
    _require_no_legacy_binding(
        services,
        bucket=bucket,
        intent=selected.intent.value,
    )
    _put_immutable(
        services,
        bucket=bucket,
        artifact=accepted,
        run_id=str(selected.intent.value["run_id"]),
    )
    keys = _list_complete(services.s3, bucket=bucket, prefix=prefix)
    if keys != (accepted.key,):
        raise SubmissionIntegrationError(
            "accepted authority prefix is not unique after publication"
        )
    winner = _fetch_artifact(
        services,
        bucket=bucket,
        key=accepted.key,
        label="submission accepted winner",
    )
    _validate_existing_accepted(selected=selected, accepted=winner)
    # S3 cannot atomically exclude a second key while creating this immutable
    # accepted object. Audit again after publication so this process never
    # reports success when a concurrent legacy member is already visible.
    _require_no_legacy_binding(
        services,
        bucket=bucket,
        intent=selected.intent.value,
    )
    return winner


def _reconcile_selected(
    services: SubmissionServices,
    *,
    selected: _SelectedSubmission,
) -> SubmissionOutcome:
    _require_no_legacy_binding(
        services,
        bucket=str(selected.descriptor.value["bucket"]),
        intent=selected.intent.value,
    )
    existing = _load_unique_accepted_if_present(
        services,
        selected=selected,
    )
    if existing is not None:
        _require_no_legacy_binding(
            services,
            bucket=str(selected.descriptor.value["bucket"]),
            intent=selected.intent.value,
        )
        return SubmissionOutcome(
            EXIT_OK,
            "accepted",
            {
                "sky_job_id": existing.value["sky_job_id"],
                "accepted_s3_uri": _s3_uri(
                    str(selected.descriptor.value["bucket"]),
                    existing.key,
                ),
                "accepted_file_sha256": existing.file_sha256,
                "accepted_body_sha256": existing.value[ACCEPTED_DIGEST_FIELD],
            },
        )
    controller_job, facts = _reconcile_controller_job(
        services,
        selected=selected,
    )
    accepted_controller_job, observation, binding = _validate_or_publish_receipts(
        services,
        selected=selected,
        controller_job=controller_job,
        facts=facts,
    )
    accepted = _publish_or_load_accepted(
        services,
        selected=selected,
        controller_job=accepted_controller_job,
        observation=observation,
        binding=binding,
    )
    _require_no_legacy_binding(
        services,
        bucket=str(selected.descriptor.value["bucket"]),
        intent=selected.intent.value,
    )
    return SubmissionOutcome(
        EXIT_OK,
        "accepted",
        {
            "sky_job_id": accepted_controller_job["sky_job_id"],
            "accepted_s3_uri": _s3_uri(
                str(selected.descriptor.value["bucket"]),
                accepted.key,
            ),
            "accepted_file_sha256": accepted.file_sha256,
            "accepted_body_sha256": accepted.value[ACCEPTED_DIGEST_FIELD],
        },
    )


def _launch_arguments(
    *,
    request: QualificationRequest,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
) -> list[str]:
    descriptor = local.descriptor.value
    bucket = str(descriptor["bucket"])
    env_values = (
        (
            "GLM52_CAMPAIGN_DESCRIPTOR_S3_URI",
            _s3_uri(bucket, local.descriptor.key),
        ),
        ("GLM52_CAMPAIGN_DESCRIPTOR_SHA256", local.descriptor.file_sha256),
        (
            "GLM52_GPU_SPEND_APPROVAL_S3_URI",
            _s3_uri(bucket, local.approval.key),
        ),
        ("GLM52_GPU_SPEND_APPROVAL_SHA256", local.approval.file_sha256),
        ("GLM52_SKY_JOB_NAME", str(intent.value["sky_job_name"])),
        ("GLM52_MANAGED_MODE", QUALIFICATION_MODE),
        ("GLM52_SKY_SUBMISSION_INTENT_S3_URI", _s3_uri(bucket, intent.key)),
        ("GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256", intent.file_sha256),
        (
            "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256",
            str(intent.value[INTENT_DIGEST_FIELD]),
        ),
    )
    arguments = [
        "jobs",
        "launch",
        "--async",
        "--config",
        str(local.config),
        "-y",
        "-n",
        str(intent.value["sky_job_name"]),
        "--no-use-spot",
        "--image-id",
        str(descriptor["image_id"]),
    ]
    for key, value in env_values:
        arguments.extend(["--env", f"{key}={value}"])
    arguments.append(str(local.task))
    return arguments


def _sealed_launch_arguments(
    *,
    arguments: list[str],
    local: _LocalAuthorities,
    materialized: _MaterializedSealedLaunch,
) -> list[str]:
    exact_materialized = _validate_materialized_coordinates(materialized)
    if type(arguments) is not list or any(type(item) is not str for item in arguments):
        raise SubmissionIntegrationError("Sky launch arguments are not exact strings")
    try:
        config_index = arguments.index("--config") + 1
    except (ValueError, IndexError) as error:
        raise SubmissionIntegrationError(
            "Sky launch arguments omit the exact config coordinate"
        ) from error
    if (
        arguments.count("--config") != 1
        or config_index >= len(arguments)
        or arguments[config_index] != str(local.config)
        or not arguments
        or arguments[-1] != str(local.task)
    ):
        raise SubmissionIntegrationError(
            "Sky launch path coordinates differ from authenticated originals"
        )
    sealed = list(arguments)
    sealed[config_index] = str(exact_materialized.config_path)
    sealed[-1] = str(exact_materialized.task_path)
    for index, (original, replacement) in enumerate(zip(arguments, sealed)):
        if index not in {config_index, len(arguments) - 1} and original != replacement:
            raise SubmissionIntegrationError(
                "sealed Sky launch changed a non-path argument"
            )
    return sealed


def _acquire_and_launch(
    request: QualificationRequest,
    services: SubmissionServices,
) -> SubmissionOutcome:
    # The descriptor-addressed selector is checked before rebuilding any
    # expiring preparation authority.  Once it exists, this invocation has no
    # acquisition or launch path: it authenticates the selected sources and
    # reconciles only.
    descriptor_pin = _load_descriptor_pin(request, services)
    bucket = str(descriptor_pin.value["bucket"])
    stable_key = submission_acquired_s3_key(
        run_id=str(descriptor_pin.value["run_id"]),
        managed_mode=QUALIFICATION_MODE,
        descriptor_file_sha256=descriptor_pin.file_sha256,
    )
    stored_raw = _get_exact(services.s3, bucket=bucket, key=stable_key)
    if stored_raw is not None:
        stored_prefix = stable_key.rsplit("/", 1)[0] + "/"
        if _list_complete(
            services.s3,
            bucket=bucket,
            prefix=stored_prefix,
        ) != (stable_key,):
            raise SubmissionIntegrationError(
                "stored acquisition selector prefix is not unique"
            )
        stored = _decode_artifact_raw(
            key=stable_key,
            raw=stored_raw,
            label="stored acquisition selector",
        )
        selected = _selected_sources_from_acquisition(
            services,
            bucket=bucket,
            descriptor=descriptor_pin,
            acquisition=stored,
            now=_now(services),
        )
        return _reconcile_selected(services, selected=selected)

    local = _load_local(request, services)
    _require_remote_static(services, local=local)
    _require_remote_published_sources(services, local=local)
    intent, baseline, control_ready = _load_prepared_artifacts(
        request,
        services,
        local=local,
    )
    facts = _capture_controller_facts(
        services,
        descriptor=local.descriptor.value,
        intent=intent.value,
    )
    _require_facts_match_baseline(
        facts=facts,
        baseline=baseline.value,
    )
    _require_final_live_equality(
        services,
        local=local,
        intent=intent,
        baseline=baseline,
        control_ready=control_ready,
        original_facts=facts,
    )
    run_id = str(local.descriptor.value["run_id"])
    # This is deliberately the last static/spend-tip read before selector
    # construction and its exact-read decision.
    _require_remote_static(services, local=local)
    source_readiness_authorities = _worker_start_v2_readiness_authorities(
        local=local,
        intent=intent,
    )
    gate_one_sources = _read_worker_start_v2_source_snapshot(task=local.task)
    source_readiness = build_worker_start_v2_readiness(
        authorities=source_readiness_authorities,
        source_snapshot=gate_one_sources,
    )
    validate_worker_start_v2_readiness(
        source_readiness,
        authorities=source_readiness_authorities,
        source_snapshot=gate_one_sources,
    )
    gate_one_config_raw = _read_stable_source(
        local.config,
        label="worker-start-v2 Gate-1 Sky config",
    )
    if (
        _sha(gate_one_config_raw)
        != local.rehearsal.value["skypilot_config_file_sha256"]
    ):
        raise SubmissionIntegrationError(
            "Gate-1 Sky config differs from the rehearsal authority"
        )
    # These exact immutable byte values are the Gate-1 launch boundary.  The
    # private wrapper is deliberately constructed only after this process wins
    # the selector, preserving the no-construction rule for every nonlaunch
    # branch; it never rereads an original path.
    decision_time = _now(services)
    acquisition = _build_acquisition_artifact(
        local=local,
        intent=intent,
        baseline=baseline,
        control_ready=control_ready,
        acquired_at=decision_time,
    )
    read_result = _acquisition_read_result(
        services,
        bucket=bucket,
        key=acquisition.key,
        observed_at=decision_time,
    )
    selected_sources: _SelectedSubmission | None = None
    selected_kwargs: dict[str, ImmutableJsonArtifact] = {}
    if read_result.artifact is not None:
        stored = _decode_artifact_raw(
            key=acquisition.key,
            raw=read_result.artifact.raw,
            label="stored acquisition selector",
        )
        selected_sources = _selected_sources_from_acquisition(
            services,
            bucket=bucket,
            descriptor=local.descriptor,
            acquisition=stored,
            now=decision_time,
        )
        selected_kwargs = {
            "selected_intent": _as_acquisition_artifact(selected_sources.intent),
            "selected_qualification_submission_ready": (
                _as_acquisition_artifact(selected_sources.readiness)
            ),
            "selected_controller_baseline": _as_acquisition_artifact(
                selected_sources.baseline
            ),
            "selected_must_start_control_plane_ready": (
                _as_acquisition_artifact(selected_sources.control_ready)
            ),
        }
    decision = decide_submission_acquisition(
        candidate=_as_acquisition_artifact(acquisition),
        read_result=read_result,
        intent=_as_acquisition_artifact(intent),
        qualification_submission_ready=_as_acquisition_artifact(local.readiness),
        controller_baseline=_as_acquisition_artifact(baseline),
        must_start_control_plane_ready=_as_acquisition_artifact(control_ready),
        now=decision_time,
        **selected_kwargs,
    )
    if decision.action == "fail-closed":
        raise SubmissionIntegrationError(decision.reason)
    if decision.action == "reconcile-only":
        if selected_sources is None:
            raise SubmissionIntegrationError(
                "acquisition decision omitted its selected sources"
            )
        return _reconcile_selected(services, selected=selected_sources)
    if decision.action != "conditional-create":
        raise SubmissionIntegrationError(
            f"unsupported acquisition decision: {decision.action}"
        )
    put_success = _put_acquisition_once(
        services,
        bucket=bucket,
        acquisition=acquisition,
        run_id=run_id,
    )
    winner_raw = _get_exact(
        services.s3,
        bucket=bucket,
        key=acquisition.key,
    )
    if winner_raw is None:
        raise SubmissionIntegrationError(
            "acquisition selector disappeared after conditional put"
        )
    winner = _decode_artifact_raw(
        key=acquisition.key,
        raw=winner_raw,
        label="acquisition selector winner",
    )
    if _list_complete(
        services.s3,
        bucket=bucket,
        prefix=acquisition.key.rsplit("/", 1)[0] + "/",
    ) != (acquisition.key,):
        raise SubmissionIntegrationError(
            "acquisition selector prefix is not unique after put"
        )
    selected = _selected_sources_from_acquisition(
        services,
        bucket=bucket,
        descriptor=local.descriptor,
        acquisition=winner,
        now=_now(services),
    )
    if not put_success:
        return _reconcile_selected(services, selected=selected)
    if winner.raw != acquisition.raw:
        raise SubmissionIntegrationError(
            "successful selector put did not persist the submitted candidate"
        )
    _require_final_live_equality(
        services,
        local=local,
        intent=intent,
        baseline=baseline,
        control_ready=control_ready,
        original_facts=facts,
    )
    gate_two_sources = _read_worker_start_v2_source_snapshot(task=local.task)
    validate_worker_start_v2_readiness(
        source_readiness,
        authorities=source_readiness_authorities,
        source_snapshot=gate_two_sources,
    )
    gate_two_config_raw = _read_stable_source(
        local.config,
        label="worker-start-v2 Gate-2 Sky config",
    )
    if gate_two_config_raw != gate_one_config_raw:
        raise SubmissionIntegrationError(
            "Sky config bytes drifted between source gates"
        )
    sealed_bundle = _build_sealed_launch_bundle(
        source_snapshot=gate_one_sources,
        config_raw=gate_one_config_raw,
    )
    with _materialize_sealed_launch(sealed_bundle) as materialized:
        launch_arguments = _sealed_launch_arguments(
            arguments=_launch_arguments(
                request=request,
                local=local,
                intent=intent,
            ),
            local=local,
            materialized=materialized,
        )
        _verify_materialized_launch(
            materialized=materialized,
            bundle=sealed_bundle,
        )
        try:
            services.sky.launch(
                launch_arguments,
                cwd=materialized.repository_root,
            )
        except Exception as launch_error:  # noqa: BLE001
            # All ordinary Sky outcomes converge on the exact same reconciler.
            _ = launch_error
    return _reconcile_selected(services, selected=selected)


def _load_descriptor_pin(
    request: QualificationRequest,
    services: SubmissionServices,
) -> _JsonArtifact:
    path = request.descriptor
    if path is None and request.local is not None:
        path = request.local.descriptor
    if path is None:
        raise SubmissionUsageError("reconcile requires --descriptor")
    value = validate_sky_campaign_descriptor(
        _decode_canonical_file(path, label="descriptor")
    )
    if (
        value.get("account_id") != APPROVED_ACCOUNT_ID
        or value.get("region") != APPROVED_REGION
        or value.get("task_name") != "glm52-campaign"
        or value.get("instance_type") != EXPECTED_INSTANCE_TYPE
        or value.get("use_spot") is not False
        or value.get("instance_count") != 1
    ):
        raise SubmissionIntegrationError("reconcile descriptor authority is foreign")
    raw = path.read_bytes()
    artifact = _JsonArtifact(
        key=str(value["campaign_descriptor_key"]),
        raw=raw,
        value=value,
    )
    bucket = str(value["bucket"])
    if _get_exact(services.s3, bucket=bucket, key=artifact.key) != raw:
        raise SubmissionIntegrationError(
            "reconcile descriptor differs from exact remote authority"
        )
    approval_raw = _get_exact(
        services.s3,
        bucket=bucket,
        key=str(value["approval_key"]),
    )
    if approval_raw is None or _sha(approval_raw) != value["approval_sha256"]:
        raise SubmissionIntegrationError("reconcile approval authority drift")
    approval_artifact = _decode_artifact_raw(
        key=str(value["approval_key"]),
        raw=approval_raw,
        label="reconcile approval",
    )
    validate_gpu_spend_approval(approval_artifact.value)
    repo_raw = _get_exact(
        services.s3,
        bucket=bucket,
        key=str(value["repo_tar_key"]),
    )
    if repo_raw is None or _sha(repo_raw) != value["repo_tar_sha256"]:
        raise SubmissionIntegrationError("reconcile repository authority drift")
    return artifact


def _reconcile(
    request: QualificationRequest,
    services: SubmissionServices,
) -> SubmissionOutcome:
    if request.acquisition_s3_uri is None:
        raise SubmissionUsageError("reconcile requires --acquisition-s3-uri")
    descriptor = _load_descriptor_pin(request, services)
    bucket, key = _parse_s3_uri(
        request.acquisition_s3_uri,
        label="acquisition S3 URI",
    )
    if bucket != descriptor.value["bucket"]:
        raise SubmissionIntegrationError("acquisition selector bucket is foreign")
    expected_key = submission_acquired_s3_key(
        run_id=str(descriptor.value["run_id"]),
        managed_mode=QUALIFICATION_MODE,
        descriptor_file_sha256=descriptor.file_sha256,
    )
    if key != expected_key:
        raise SubmissionIntegrationError(
            "acquisition selector URI is not descriptor-addressed"
        )
    acquisition = _fetch_artifact(
        services,
        bucket=bucket,
        key=key,
        label="submission acquisition",
    )
    selected = _selected_sources_from_acquisition(
        services,
        bucket=bucket,
        descriptor=descriptor,
        acquisition=acquisition,
        now=_now(services),
    )
    return _reconcile_selected(services, selected=selected)


def _handoff(
    *,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
    baseline: _JsonArtifact,
    control_ready: _JsonArtifact,
    created_at: datetime,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_submission_handoff_v1",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": local.descriptor.value["bucket"],
        "run_id": local.descriptor.value["run_id"],
        "managed_mode": QUALIFICATION_MODE,
        "descriptor_s3_uri": _s3_uri(
            str(local.descriptor.value["bucket"]),
            local.descriptor.key,
        ),
        "descriptor_file_sha256": local.descriptor.file_sha256,
        "intent_s3_uri": _s3_uri(
            str(local.descriptor.value["bucket"]),
            intent.key,
        ),
        "intent_file_sha256": intent.file_sha256,
        "intent_body_sha256": intent.value[INTENT_DIGEST_FIELD],
        "controller_baseline_s3_uri": _s3_uri(
            str(local.descriptor.value["bucket"]),
            baseline.key,
        ),
        "controller_baseline_file_sha256": baseline.file_sha256,
        "controller_baseline_body_sha256": baseline.value["baseline_body_sha256"],
        "must_start_ready_s3_uri": _s3_uri(
            str(local.descriptor.value["bucket"]),
            control_ready.key,
        ),
        "must_start_ready_file_sha256": control_ready.file_sha256,
        "must_start_ready_body_sha256": control_ready.value[
            "control_plane_ready_body_sha256"
        ],
        "requires_separate_active_deployment": True,
        "created_at": _whole_second(created_at, field="handoff created_at")[0],
    }
    return {**body, "handoff_body_sha256": _sha(_canonical(body))}


def _deployment_handoff(
    *,
    local: _LocalAuthorities,
    intent: _JsonArtifact,
    baseline: _JsonArtifact,
    created_at: datetime,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_submission_deployment_handoff_v1",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": local.descriptor.value["bucket"],
        "run_id": local.descriptor.value["run_id"],
        "managed_mode": QUALIFICATION_MODE,
        "descriptor_s3_uri": _s3_uri(
            str(local.descriptor.value["bucket"]),
            local.descriptor.key,
        ),
        "descriptor_file_sha256": local.descriptor.file_sha256,
        "intent_s3_uri": _s3_uri(
            str(local.descriptor.value["bucket"]),
            intent.key,
        ),
        "intent_file_sha256": intent.file_sha256,
        "intent_body_sha256": intent.value[INTENT_DIGEST_FIELD],
        "controller_baseline_s3_uri": _s3_uri(
            str(local.descriptor.value["bucket"]),
            baseline.key,
        ),
        "controller_baseline_file_sha256": baseline.file_sha256,
        "controller_baseline_body_sha256": baseline.value["baseline_body_sha256"],
        "requires_separate_active_deployment": True,
        "created_at": _whole_second(
            created_at,
            field="deployment handoff created_at",
        )[0],
    }
    return {**body, "handoff_body_sha256": _sha(_canonical(body))}


def _write_atomic(path: Path, value: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise SubmissionUsageError("output handoff must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise SubmissionUsageError("output handoff parent is invalid")
    raw = _canonical(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if path.read_bytes() != raw:
        raise SubmissionIntegrationError("output handoff changed after atomic write")


def _validate_only(
    request: QualificationRequest,
    services: SubmissionServices,
) -> SubmissionOutcome:
    local = _load_local(request, services)
    _require_remote_static(services, local=local)
    intent = _build_intent_artifact(local, intent_at=_now(services))
    _, control_ready, _ = _build_live_artifacts(
        services,
        local=local,
        intent=intent,
    )
    return SubmissionOutcome(
        exit_code=EXIT_OK,
        status="validated",
        detail={
            "intent_s3_uri": _s3_uri(
                str(local.descriptor.value["bucket"]),
                intent.key,
            ),
            "must_start_ready_s3_uri": _s3_uri(
                str(local.descriptor.value["bucket"]),
                control_ready.key,
            ),
        },
    )


def _resume_deployment_handoff(
    request: QualificationRequest,
    services: SubmissionServices,
    *,
    local: _LocalAuthorities,
) -> tuple[_JsonArtifact, _JsonArtifact] | None:
    path = request.output_handoff
    if path is None or not path.exists():
        return None
    value = _decode_canonical_file(path, label="existing output handoff")
    if value.get("record_type") != "glm52_sky_submission_deployment_handoff_v1":
        return None
    expected_fields = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "managed_mode",
        "descriptor_s3_uri",
        "descriptor_file_sha256",
        "intent_s3_uri",
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_s3_uri",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "requires_separate_active_deployment",
        "created_at",
        "handoff_body_sha256",
    }
    body = dict(value)
    digest = body.pop("handoff_body_sha256", None)
    if (
        set(value) != expected_fields
        or value.get("schema_version") != 1
        or digest != _sha(_canonical(body))
        or value.get("account_id") != APPROVED_ACCOUNT_ID
        or value.get("region") != APPROVED_REGION
        or value.get("bucket") != local.descriptor.value["bucket"]
        or value.get("run_id") != local.descriptor.value["run_id"]
        or value.get("managed_mode") != QUALIFICATION_MODE
        or value.get("descriptor_s3_uri")
        != _s3_uri(
            str(local.descriptor.value["bucket"]),
            local.descriptor.key,
        )
        or value.get("descriptor_file_sha256") != local.descriptor.file_sha256
        or value.get("requires_separate_active_deployment") is not True
    ):
        raise SubmissionIntegrationError(
            "existing deployment handoff is malformed or foreign"
        )
    created_at, created_time = _whole_second(
        str(value.get("created_at")),
        field="deployment handoff created_at",
    )
    if created_at != value.get("created_at") or created_time > _now(services):
        raise SubmissionIntegrationError("existing deployment handoff time is invalid")
    bucket = str(local.descriptor.value["bucket"])
    intent_bucket, intent_key = _parse_s3_uri(
        str(value.get("intent_s3_uri")),
        label="deployment handoff intent S3 URI",
    )
    baseline_bucket, baseline_key = _parse_s3_uri(
        str(value.get("controller_baseline_s3_uri")),
        label="deployment handoff baseline S3 URI",
    )
    if intent_bucket != bucket or baseline_bucket != bucket:
        raise SubmissionIntegrationError(
            "existing deployment handoff bucket is foreign"
        )
    intent = _fetch_artifact(
        services,
        bucket=bucket,
        key=intent_key,
        label="deployment handoff intent",
    )
    baseline = _fetch_artifact(
        services,
        bucket=bucket,
        key=baseline_key,
        label="deployment handoff controller baseline",
    )
    if (
        intent.file_sha256 != value.get("intent_file_sha256")
        or intent.value.get(INTENT_DIGEST_FIELD) != value.get("intent_body_sha256")
        or baseline.file_sha256 != value.get("controller_baseline_file_sha256")
        or baseline.value.get("baseline_body_sha256")
        != value.get("controller_baseline_body_sha256")
    ):
        raise SubmissionIntegrationError(
            "existing deployment handoff authority drifted"
        )
    validate_submission_intent(
        intent.value,
        descriptor=local.descriptor.value,
        cache_seed_acceptance=local.cache_acceptance.value,
        gpu_spend_snapshot=local.spend_snapshot.value,
        staged_readiness=local.staged.value,
        rehearsal_evidence=local.rehearsal.value,
        qualification_submission_ready=local.readiness.value,
    )
    _validate_prepared_baseline(
        local=local,
        intent=intent,
        baseline=baseline,
        acquired_at=_now(services),
    )
    return intent, baseline


def _prepare_intent(
    request: QualificationRequest,
    services: SubmissionServices,
) -> SubmissionOutcome:
    if request.output_handoff is None:
        raise SubmissionUsageError("prepare-intent requires --output-handoff")
    local = _load_local(request, services)
    _require_remote_static(services, local=local)
    bucket = str(local.descriptor.value["bucket"])
    run_id = str(local.descriptor.value["run_id"])
    for artifact in (
        local.cache_acceptance,
        local.spend_snapshot,
        local.rehearsal,
        local.readiness,
    ):
        _put_immutable(
            services,
            bucket=bucket,
            artifact=artifact,
            run_id=run_id,
        )
    resumed = _resume_deployment_handoff(
        request,
        services,
        local=local,
    )
    if resumed is None:
        intent = _build_intent_artifact(local, intent_at=_now(services))
        _put_immutable(
            services,
            bucket=bucket,
            artifact=intent,
            run_id=run_id,
        )
        baseline, _ = _build_baseline_artifact(
            services,
            local=local,
            intent=intent,
        )
        _put_immutable(
            services,
            bucket=bucket,
            artifact=baseline,
            run_id=run_id,
        )
    else:
        intent, baseline = resumed
        facts = _capture_controller_facts(
            services,
            descriptor=local.descriptor.value,
            intent=intent.value,
        )
        _require_facts_match_baseline(
            facts=facts,
            baseline=baseline.value,
        )
    try:
        control_ready = _build_control_ready_artifact(
            services,
            local=local,
            intent=intent,
            baseline_artifact=baseline,
        )
    except SubmissionDeploymentRequired:
        _write_atomic(
            request.output_handoff,
            _deployment_handoff(
                local=local,
                intent=intent,
                baseline=baseline,
                created_at=_now(services),
            ),
        )
        raise
    _put_immutable(
        services,
        bucket=bucket,
        artifact=control_ready,
        run_id=run_id,
    )
    handoff = _handoff(
        local=local,
        intent=intent,
        baseline=baseline,
        control_ready=control_ready,
        created_at=_now(services),
    )
    _write_atomic(request.output_handoff, handoff)
    return SubmissionOutcome(
        exit_code=EXIT_OK,
        status="intent-prepared",
        detail=handoff,
    )


def run_qualification(
    request: QualificationRequest,
    *,
    services: SubmissionServices,
) -> SubmissionOutcome:
    """Run one qualification command without broadening launch authority."""

    try:
        _guard_identity(request, services)
        if request.action == "validate-only":
            return _validate_only(request, services)
        if request.action == "prepare-intent":
            return _prepare_intent(request, services)
        if request.action == "acquire-and-launch":
            return _acquire_and_launch(request, services)
        if request.action == "reconcile":
            return _reconcile(request, services)
        raise SubmissionUsageError(
            f"unsupported qualification action: {request.action}"
        )
    except SubmissionDeploymentRequired as error:
        return SubmissionOutcome(
            EXIT_DEPLOYMENT_REQUIRED,
            "deployment-required",
            {"reason": str(error)},
        )
    except SubmissionReconcilePending as error:
        return SubmissionOutcome(
            EXIT_RECONCILE_PENDING,
            "reconcile-pending",
            {"reason": str(error)},
        )
    except SubmissionUsageError as error:
        return SubmissionOutcome(EXIT_USAGE, "usage-error", {"reason": str(error)})
    except Exception as error:  # noqa: BLE001
        return SubmissionOutcome(
            EXIT_FAIL_CLOSED,
            "fail-closed",
            {"reason": str(error)},
        )


def _cache_seed_remote_staged_chain(
    services: SubmissionServices,
    *,
    descriptor: _JsonArtifact,
    approval: _JsonArtifact,
    staged: _JsonArtifact,
    staged_readiness_version_id: str,
) -> _JsonArtifact:
    bucket = str(descriptor.value["bucket"])
    exact_staged_raw = _get_exact(
        services.s3,
        bucket=bucket,
        key=staged.key,
        version_id=staged_readiness_version_id,
    )
    if exact_staged_raw != staged.raw:
        raise SubmissionIntegrationError(
            "remote staged readiness exact VersionId drifted"
        )
    ready, versions = _validate_remote_staged_readiness(
        staged,
        descriptor=descriptor,
    )
    manifest_key = str(ready["bundle_manifest_key"])
    manifest_version = str(ready["bundle_manifest_version_id"])
    manifest_raw = _get_exact(
        services.s3,
        bucket=bucket,
        key=manifest_key,
        version_id=manifest_version,
    )
    if manifest_raw is None:
        raise SubmissionIntegrationError(
            "remote bundle manifest pinned version is unavailable"
        )
    manifest, files = _validate_remote_bundle_manifest(
        manifest_raw,
        ready=ready,
        descriptor=descriptor,
    )
    role_raw: dict[str, bytes] = {}
    role_items: dict[str, dict[str, object]] = {}
    for item in files:
        role = str(item["role"])
        raw = _get_exact(
            services.s3,
            bucket=bucket,
            key=str(item["key"]),
            version_id=versions[role],
        )
        if (
            raw is None
            or len(raw) != int(item["size"])
            or _sha(raw) != item["sha256"]
        ):
            raise SubmissionIntegrationError(
                f"remote staged object exact-version drift: {role}"
            )
        role_raw[role] = raw
        role_items[role] = dict(item)
    if role_raw["descriptor"] != descriptor.raw:
        raise SubmissionIntegrationError(
            "remote staged descriptor differs from selected cache-seed descriptor"
        )
    if role_raw["approval"] != approval.raw:
        raise SubmissionIntegrationError(
            "remote staged approval differs from selected cache-seed approval"
        )
    descriptor_artifacts = descriptor.value.get("artifacts")
    if not isinstance(descriptor_artifacts, Mapping):
        raise SubmissionIntegrationError("descriptor artifact pins are malformed")
    exact_role_bindings = {
        "repository_tar": (
            descriptor.value["repo_tar_key"],
            descriptor.value["repo_tar_sha256"],
        ),
        "approval": (
            descriptor.value["approval_key"],
            descriptor.value["approval_sha256"],
        ),
        "training_config": (
            descriptor_artifacts["training_config_key"],
            descriptor_artifacts["training_config_sha256"],
        ),
        "artifact_inventory": (
            descriptor_artifacts["artifact_inventory_key"],
            descriptor_artifacts["artifact_inventory_sha256"],
        ),
    }
    for role, (key, digest) in exact_role_bindings.items():
        item = role_items[role]
        if item["key"] != key or item["sha256"] != digest:
            raise SubmissionIntegrationError(
                f"remote staged {role} differs from descriptor pins"
            )
    inventory = _decode_artifact_raw(
        key=str(role_items["artifact_inventory"]["key"]),
        raw=role_raw["artifact_inventory"],
        label="remote cache-seed artifact inventory",
    )
    try:
        inventory_value = validate_s3_artifact_inventory(inventory.value)
    except ValueError as error:
        raise SubmissionIntegrationError(
            f"remote cache-seed artifact inventory is invalid: {error}"
        ) from error
    if (
        inventory_value.get("run_id") != descriptor.value["run_id"]
        or inventory_value.get("bucket") != bucket
    ):
        raise SubmissionIntegrationError(
            "remote cache-seed artifact inventory is foreign"
        )
    audit_raw = _get_exact(
        services.s3,
        bucket=bucket,
        key=str(ready["artifact_audit_key"]),
        version_id=versions["artifact_audit"],
    )
    if audit_raw is None:
        raise SubmissionIntegrationError(
            "remote cache-seed artifact audit pinned version is unavailable"
        )
    _validate_remote_audit(
        audit_raw,
        ready=ready,
        descriptor=descriptor,
        inventory=inventory_value,
    )
    return _JsonArtifact(manifest_key, manifest_raw, manifest)


def _load_cache_seed(
    request: CacheSeedRequest,
    services: SubmissionServices,
) -> _CacheSeedLocalAuthorities:
    paths = request.local
    descriptor_value = validate_sky_campaign_descriptor(
        _decode_canonical_file(paths.descriptor, label="cache-seed descriptor")
    )
    approval_value = validate_gpu_spend_approval(
        _decode_canonical_file(paths.approval, label="cache-seed approval")
    )
    staged_value = _decode_canonical_file(
        paths.staged_ready,
        label="cache-seed staged readiness",
    )
    rehearsal_value = _decode_canonical_file(
        paths.rehearsal_evidence,
        label="cache-seed rehearsal evidence",
    )
    descriptor_raw = _read_regular(
        paths.descriptor,
        label="cache-seed descriptor",
    )
    approval_raw = _read_regular(paths.approval, label="cache-seed approval")
    staged_raw = _read_regular(
        paths.staged_ready,
        label="cache-seed staged readiness",
    )
    rehearsal_raw = _read_regular(
        paths.rehearsal_evidence,
        label="cache-seed rehearsal evidence",
    )
    if (
        descriptor_value.get("account_id") != APPROVED_ACCOUNT_ID
        or descriptor_value.get("region") != APPROVED_REGION
        or descriptor_value.get("task_name") != "glm52-campaign"
        or descriptor_value.get("instance_type") != EXPECTED_INSTANCE_TYPE
        or descriptor_value.get("instance_count") != 1
        or descriptor_value.get("use_spot") is not False
        or _sha(approval_raw) != descriptor_value.get("approval_sha256")
    ):
        raise SubmissionIntegrationError(
            "cache-seed descriptor envelope or approval is foreign"
        )
    artifacts = descriptor_value.get("artifacts")
    if not isinstance(artifacts, Mapping) or (
        artifacts.get("qualification_cache_prefix") != "qualification-cache/"
        or artifacts.get("qualification_cache_manifest_sha256") != "0" * 64
    ):
        raise SubmissionIntegrationError(
            "cache-seed descriptor is not the all-zero cache authority"
        )
    _require_submission_window(descriptor=descriptor_value, now=_now(services))
    run_id = str(descriptor_value["run_id"])
    descriptor_key = str(descriptor_value["campaign_descriptor_key"])
    prefix = f"campaigns/{run_id}/submissions/"
    suffix = "/campaign-descriptor-v2.json"
    if not descriptor_key.startswith(prefix) or not descriptor_key.endswith(suffix):
        raise SubmissionIntegrationError(
            "cache-seed descriptor submission coordinate is invalid"
        )
    submission_id = descriptor_key[len(prefix) : -len(suffix)]
    if not submission_id or "/" in submission_id or submission_id in {".", ".."}:
        raise SubmissionIntegrationError(
            "cache-seed descriptor submission ID is unsafe"
        )
    staged_key = descriptor_key.removesuffix(
        "campaign-descriptor-v2.json"
    ) + "STAGED_CONTROL_PLANE_READY.json"
    rehearsal_body = dict(rehearsal_value)
    rehearsal_digest = rehearsal_body.pop("rehearsal_body_sha256", None)
    if (
        not _is_sha(rehearsal_digest)
        or _sha(_canonical(rehearsal_body)) != rehearsal_digest
    ):
        raise SubmissionIntegrationError(
            "cache-seed rehearsal body SHA-256 mismatch"
        )
    rehearsal_key = (
        f"campaigns/{run_id}/qualification/rehearsals/"
        f"{rehearsal_digest}/GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    )
    descriptor_artifact = _JsonArtifact(
        descriptor_key,
        descriptor_raw,
        descriptor_value,
    )
    approval_artifact = _JsonArtifact(
        str(descriptor_value["approval_key"]),
        approval_raw,
        approval_value,
    )
    staged_artifact = _JsonArtifact(staged_key, staged_raw, staged_value)
    rehearsal_artifact = _JsonArtifact(
        rehearsal_key,
        rehearsal_raw,
        rehearsal_value,
    )
    manifest = _cache_seed_remote_staged_chain(
        services,
        descriptor=descriptor_artifact,
        approval=approval_artifact,
        staged=staged_artifact,
        staged_readiness_version_id=request.staged_readiness_version_id,
    )
    if (
        _get_exact(
            services.s3,
            bucket=str(descriptor_value["bucket"]),
            key=rehearsal_key,
        )
        != rehearsal_raw
    ):
        raise SubmissionIntegrationError(
            "remote cache-seed rehearsal evidence drifted"
        )
    source_snapshot = _read_worker_start_v2_source_snapshot(task=paths.task)
    config_raw = _read_stable_source(
        paths.config,
        label="cache-seed Gate-1 Sky config",
    )
    validation = services.sky.validate_control_plane(
        task=paths.task,
        config=paths.config,
    )
    validated_source_snapshot = _read_worker_start_v2_source_snapshot(
        task=paths.task
    )
    validated_config_raw = _read_stable_source(
        paths.config,
        label="cache-seed post-validation Sky config",
    )
    if not isinstance(validation, Mapping) or (
        validation.get("skypilot_version") != EXPECTED_SKYPILOT_VERSION
        or validation.get("task_file_sha256")
        != rehearsal_value.get("skypilot_task_file_sha256")
        or validation.get("config_file_sha256")
        != rehearsal_value.get("skypilot_config_file_sha256")
        or _sha(source_snapshot.sky_task_raw)
        != rehearsal_value.get("skypilot_task_file_sha256")
        or _sha(config_raw) != rehearsal_value.get("skypilot_config_file_sha256")
        or validated_source_snapshot != source_snapshot
        or validated_config_raw != config_raw
    ):
        raise SubmissionIntegrationError(
            "cache-seed Sky task/config authority drift"
        )
    return _CacheSeedLocalAuthorities(
        descriptor=descriptor_artifact,
        approval=approval_artifact,
        staged=staged_artifact,
        bundle_manifest=manifest,
        rehearsal=rehearsal_artifact,
        task=paths.task,
        config=paths.config,
        source_snapshot=source_snapshot,
        config_raw=config_raw,
    )


def _exact_absence_read(
    s3: Any,
    *,
    bucket: str,
    key: str,
    phase: str,
) -> dict[str, object]:
    try:
        s3.get_object(Bucket=bucket, Key=key)
    except Exception as error:
        code, status = _aws_error(error)
        if code != "NoSuchKey" or status != 404:
            raise SubmissionIntegrationError(
                f"{key} absence is ambiguous: {code or error}"
            ) from error
        return {
            "error_code": "NoSuchKey",
            "http_status_code": 404,
            "phase": phase,
        }
    raise SubmissionIntegrationError(f"{key} unexpectedly exists")


def _collect_empty_s3_record_pages(
    s3: Any,
    *,
    bucket: str,
    prefix: str,
) -> list[dict[str, object]]:
    pages: list[dict[str, object]] = []
    request_token: str | None = None
    seen_tokens: set[str] = set()
    aggregate: list[str] = []
    while True:
        kwargs: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if request_token is not None:
            kwargs["ContinuationToken"] = request_token
        response = s3.list_objects_v2(**kwargs)
        if not isinstance(response, Mapping):
            raise SubmissionIntegrationError("S3 records LIST response is malformed")
        metadata = response.get("ResponseMetadata")
        contents = response.get("Contents", [])
        truncated = response.get("IsTruncated")
        key_count = response.get("KeyCount")
        if (
            not isinstance(metadata, Mapping)
            or type(metadata.get("HTTPStatusCode")) is not int
            or metadata.get("HTTPStatusCode") != 200
            or not isinstance(contents, list)
            or type(truncated) is not bool
            or type(key_count) is not int
            or key_count != len(contents)
        ):
            raise SubmissionIntegrationError("S3 records LIST page is malformed")
        keys = [
            item.get("Key")
            for item in contents
            if isinstance(item, Mapping)
        ]
        if (
            len(keys) != len(contents)
            or any(not isinstance(key, str) for key in keys)
            or keys != sorted(keys)
            or len(keys) != len(set(keys))
        ):
            raise SubmissionIntegrationError(
                "S3 records LIST keys are noncanonical"
            )
        aggregate.extend(str(key) for key in keys)
        next_token = response.get("NextContinuationToken")
        if truncated:
            if (
                not isinstance(next_token, str)
                or not next_token
                or next_token in seen_tokens
            ):
                raise SubmissionIntegrationError(
                    "S3 records LIST continuation chain is malformed"
                )
            seen_tokens.add(next_token)
        elif next_token is not None:
            raise SubmissionIntegrationError(
                "S3 records terminal LIST page has a continuation token"
            )
        pages.append(
            {
                "page_index": len(pages),
                "request_continuation_token": request_token,
                "http_status_code": 200,
                "key_count": key_count,
                "keys": keys,
                "is_truncated": truncated,
                "next_continuation_token": next_token,
            }
        )
        if not truncated:
            break
        request_token = str(next_token)
    if aggregate or aggregate != sorted(set(aggregate)):
        raise SubmissionIntegrationError(
            "S3 spend-ledger immutable record inventory is nonempty"
        )
    return pages


def _collect_empty_ec2_pages(
    ec2: Any,
    *,
    filters: list[dict[str, object]],
) -> list[dict[str, object]]:
    pages: list[dict[str, object]] = []
    request_token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        kwargs: dict[str, object] = {"Filters": filters}
        if request_token is not None:
            kwargs["NextToken"] = request_token
        response = ec2.describe_instances(**kwargs)
        if not isinstance(response, Mapping):
            raise SubmissionIntegrationError(
                "EC2 DescribeInstances response is malformed"
            )
        metadata = response.get("ResponseMetadata")
        reservations = response.get("Reservations")
        next_token = response.get("NextToken")
        if (
            not isinstance(metadata, Mapping)
            or type(metadata.get("HTTPStatusCode")) is not int
            or metadata.get("HTTPStatusCode") != 200
            or not isinstance(reservations, list)
            or reservations
        ):
            raise SubmissionIntegrationError(
                "EC2 tagged-P5 inventory is nonempty or malformed"
            )
        if next_token is not None and (
            not isinstance(next_token, str)
            or not next_token
            or next_token in seen_tokens
        ):
            raise SubmissionIntegrationError(
                "EC2 DescribeInstances token chain is malformed"
            )
        pages.append(
            {
                "page_index": len(pages),
                "request_next_token": request_token,
                "http_status_code": 200,
                "reservations": [],
                "next_token": next_token,
            }
        )
        if next_token is None:
            break
        seen_tokens.add(next_token)
        request_token = next_token
    return pages


def _cache_seed_live_artifacts(
    services: SubmissionServices,
    *,
    local: _CacheSeedLocalAuthorities,
    request: CacheSeedRequest,
    caller_arn: str,
) -> tuple[_JsonArtifact, _JsonArtifact, _JsonArtifact, _JsonArtifact]:
    descriptor = local.descriptor.value
    run_id = str(descriptor["run_id"])
    bucket = str(descriptor["bucket"])
    observed_at, _ = _whole_second(
        services.clock(),
        field="cache-seed collection observed_at",
    )
    latest_key = f"campaigns/{run_id}/runtime/GPU_SPEND_LEDGER_LATEST.json"
    legacy_key = f"campaigns/{run_id}/runtime/GPU_SPEND_LEDGER.jsonl"
    records_prefix = f"campaigns/{run_id}/runtime/spend-ledger/records/"
    latest_before = _exact_absence_read(
        services.s3,
        bucket=bucket,
        key=latest_key,
        phase="before-records-list",
    )
    legacy_before = _exact_absence_read(
        services.s3,
        bucket=bucket,
        key=legacy_key,
        phase="before-records-list",
    )
    record_pages = _collect_empty_s3_record_pages(
        services.s3,
        bucket=bucket,
        prefix=records_prefix,
    )
    latest_after = _exact_absence_read(
        services.s3,
        bucket=bucket,
        key=latest_key,
        phase="after-records-list",
    )
    legacy_after = _exact_absence_read(
        services.s3,
        bucket=bucket,
        key=legacy_key,
        phase="after-records-list",
    )
    history_filters = [
        {"Name": "instance-type", "Values": ["p5.48xlarge"]},
        {"Name": "tag:campaign-run-id", "Values": [run_id]},
        {
            "Name": "tag:cost-allocation",
            "Values": ["glm52-sky-campaign"],
        },
        {"Name": "tag:model", "Values": ["glm-5.2"]},
        {"Name": "tag:owner", "Values": ["jack.mazac"]},
        {"Name": "tag:project", "Values": ["keep-glm52"]},
    ]
    active_filters = [
        *history_filters,
        {
            "Name": "instance-state-name",
            "Values": ["pending", "running", "shutting-down", "stopping"],
        },
    ]
    history_pages = _collect_empty_ec2_pages(
        services.ec2,
        filters=history_filters,
    )
    active_pages = _collect_empty_ec2_pages(
        services.ec2,
        filters=active_filters,
    )
    completed_at, _ = _whole_second(
        services.clock(),
        field="cache-seed collection completed_at",
    )
    translated_history = [
        {"name": item["Name"], "values": item["Values"]}
        for item in history_filters
    ]
    translated_active = [
        {"name": item["Name"], "values": item["Values"]}
        for item in active_filters
    ]
    s3_value: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_s3_spend_ledger_absence_observation_v1",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": bucket,
        "run_id": run_id,
        "latest_marker_key": latest_key,
        "latest_marker_reads": [latest_before, latest_after],
        "legacy_ledger_key": legacy_key,
        "legacy_ledger_reads": [legacy_before, legacy_after],
        "spend_records_prefix": records_prefix,
        "spend_records_list_pages": record_pages,
        "observed_at": observed_at,
    }
    ec2_value: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_ec2_tagged_p5_zero_inventory_observation_v1",
        "account_id": APPROVED_ACCOUNT_ID,
        "caller_arn": caller_arn,
        "region": APPROVED_REGION,
        "run_id": run_id,
        "history_request": {"filters": translated_history},
        "history_pages": history_pages,
        "active_request": {"filters": translated_active},
        "active_pages": active_pages,
        "observed_at": observed_at,
    }
    s3_raw = cache_seed_canonical_file_bytes(s3_value)
    ec2_raw = cache_seed_canonical_file_bytes(ec2_value)
    allowance_value = build_gpu_launch_allowance(
        descriptor_raw=local.descriptor.raw,
        expected_descriptor_sha256=local.descriptor.file_sha256,
        approval_raw=local.approval.raw,
        expected_approval_sha256=local.approval.file_sha256,
        s3_spend_ledger_absence_observation_raw=s3_raw,
        expected_s3_spend_ledger_absence_observation_sha256=_sha(s3_raw),
        ec2_tagged_p5_zero_inventory_observation_raw=ec2_raw,
        expected_ec2_tagged_p5_zero_inventory_observation_sha256=_sha(ec2_raw),
        observed_at=observed_at,
    )
    validate_gpu_launch_allowance(allowance_value, now=completed_at)
    allowance_raw = cache_seed_canonical_file_bytes(allowance_value)
    intent_value = build_cache_seed_submission_intent(
        descriptor=descriptor,
        descriptor_file_sha256=local.descriptor.file_sha256,
        approval=local.approval.value,
        approval_file_sha256=local.approval.file_sha256,
        staged_readiness=local.staged.value,
        staged_readiness_file_sha256=local.staged.file_sha256,
        staged_readiness_version_id=request.staged_readiness_version_id,
        bundle_manifest=local.bundle_manifest.value,
        bundle_manifest_file_sha256=local.bundle_manifest.file_sha256,
        rehearsal_evidence=local.rehearsal.value,
        rehearsal_evidence_file_sha256=local.rehearsal.file_sha256,
        s3_spend_ledger_absence_observation_file_sha256=_sha(s3_raw),
        ec2_tagged_p5_zero_inventory_observation_file_sha256=_sha(ec2_raw),
        launch_allowance=allowance_value,
        launch_allowance_file_sha256=_sha(allowance_raw),
        intent_at=observed_at,
    )
    validate_cache_seed_submission_intent(
        intent_value,
        descriptor=descriptor,
        descriptor_file_sha256=local.descriptor.file_sha256,
        approval=local.approval.value,
        approval_file_sha256=local.approval.file_sha256,
        staged_readiness=local.staged.value,
        staged_readiness_file_sha256=local.staged.file_sha256,
        staged_readiness_version_id=request.staged_readiness_version_id,
        bundle_manifest=local.bundle_manifest.value,
        bundle_manifest_file_sha256=local.bundle_manifest.file_sha256,
        rehearsal_evidence=local.rehearsal.value,
        rehearsal_evidence_file_sha256=local.rehearsal.file_sha256,
        s3_spend_ledger_absence_observation_file_sha256=_sha(s3_raw),
        ec2_tagged_p5_zero_inventory_observation_file_sha256=_sha(ec2_raw),
        launch_allowance=allowance_value,
        launch_allowance_file_sha256=_sha(allowance_raw),
    )
    intent_raw = cache_seed_canonical_file_bytes(intent_value)
    return (
        _JsonArtifact(
            s3_spend_ledger_absence_observation_s3_key(
                run_id=run_id,
                observation_file_sha256=_sha(s3_raw),
            ),
            s3_raw,
            s3_value,
        ),
        _JsonArtifact(
            ec2_tagged_p5_zero_inventory_observation_s3_key(
                run_id=run_id,
                observation_file_sha256=_sha(ec2_raw),
            ),
            ec2_raw,
            ec2_value,
        ),
        _JsonArtifact(
            gpu_launch_allowance_s3_key(
                run_id=run_id,
                allowance_body_sha256=str(
                    allowance_value["allowance_body_sha256"]
                ),
            ),
            allowance_raw,
            allowance_value,
        ),
        _JsonArtifact(
            cache_seed_submission_intent_s3_key(
                run_id=run_id,
                intent_body_sha256=str(
                    intent_value[CACHE_SEED_INTENT_DIGEST_FIELD]
                ),
            ),
            intent_raw,
            intent_value,
        ),
    )


def _strict_claim_get(
    s3: Any,
    *,
    bucket: str,
    key: str,
    version_id: str | None = None,
    expected_version_id: str | None = None,
    allow_absent: bool = False,
) -> bytes | None:
    kwargs: dict[str, object] = {"Bucket": bucket, "Key": key}
    if version_id is not None:
        kwargs["VersionId"] = version_id
    try:
        response = s3.get_object(**kwargs)
    except Exception as error:
        code, status = _aws_error(error)
        if allow_absent and version_id is None and code == "NoSuchKey" and status == 404:
            return None
        raise SubmissionIntegrationError(
            f"strict launch-claim GET failed: {code or error}"
        ) from error
    if not isinstance(response, Mapping):
        raise SubmissionIntegrationError(
            "strict launch-claim GET response is malformed"
        )
    metadata = response.get("ResponseMetadata")
    response_version = response.get("VersionId")
    if (
        not isinstance(metadata, Mapping)
        or type(metadata.get("HTTPStatusCode")) is not int
        or metadata.get("HTTPStatusCode") != 200
        or not isinstance(response_version, str)
        or not response_version
        or response_version == "null"
        or response_version != (expected_version_id or version_id)
    ):
        raise SubmissionIntegrationError(
            "strict launch-claim GET status or VersionId is ambiguous"
        )
    raw = _body_bytes(response.get("Body"))
    length = response.get("ContentLength")
    if type(length) is not int or length != len(raw):
        raise SubmissionIntegrationError(
            "strict launch-claim GET content length is invalid"
        )
    checksum = response.get("ChecksumSHA256")
    if checksum is not None and checksum != base64.b64encode(
        hashlib.sha256(raw).digest()
    ).decode("ascii"):
        raise SubmissionIntegrationError(
            "strict launch-claim GET checksum is invalid"
        )
    return raw


def _inspect_cache_seed_claim_history(
    services: SubmissionServices,
    *,
    local: _CacheSeedLocalAuthorities,
) -> _CacheSeedClaimState | None:
    descriptor = local.descriptor.value
    bucket = str(descriptor["bucket"])
    run_id = str(descriptor["run_id"])
    claim_key = cache_seed_launch_claim_s3_key(
        run_id=run_id,
        descriptor_file_sha256=local.descriptor.file_sha256,
    )
    claim_prefix = claim_key.removesuffix("LAUNCH_CLAIM.json")
    versions: list[Mapping[str, object]] = []
    delete_markers: list[Mapping[str, object]] = []
    prior_markers: tuple[str, str] | None = None
    seen_markers: set[tuple[str, str]] = set()
    while True:
        kwargs: dict[str, object] = {
            "Bucket": bucket,
            "Prefix": claim_prefix,
        }
        if prior_markers is not None:
            kwargs.update(
                {
                    "KeyMarker": prior_markers[0],
                    "VersionIdMarker": prior_markers[1],
                }
            )
        response = services.s3.list_object_versions(**kwargs)
        if not isinstance(response, Mapping):
            raise SubmissionIntegrationError(
                "launch-claim version LIST response is malformed"
            )
        metadata = response.get("ResponseMetadata")
        page_versions = response.get("Versions", [])
        page_markers = response.get("DeleteMarkers", [])
        truncated = response.get("IsTruncated")
        if (
            not isinstance(metadata, Mapping)
            or type(metadata.get("HTTPStatusCode")) is not int
            or metadata.get("HTTPStatusCode") != 200
            or not isinstance(page_versions, list)
            or not isinstance(page_markers, list)
            or type(truncated) is not bool
            or any(not isinstance(item, Mapping) for item in page_versions)
            or any(not isinstance(item, Mapping) for item in page_markers)
        ):
            raise SubmissionIntegrationError(
                "launch-claim version LIST page is malformed"
            )
        versions.extend(page_versions)
        delete_markers.extend(page_markers)
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if truncated:
            if (
                not isinstance(next_key, str)
                or not next_key
                or not isinstance(next_version, str)
                or not next_version
                or (next_key, next_version) in seen_markers
            ):
                raise SubmissionIntegrationError(
                    "launch-claim version marker pair is malformed"
                )
            prior_markers = (next_key, next_version)
            seen_markers.add(prior_markers)
            continue
        if next_key is not None or next_version is not None:
            raise SubmissionIntegrationError(
                "terminal launch-claim version page has markers"
            )
        break
    if delete_markers:
        raise SubmissionIntegrationError(
            "launch-claim prefix contains a delete marker"
        )
    if any(item.get("Key") != claim_key for item in versions):
        raise SubmissionIntegrationError(
            "launch-claim prefix contains a sibling key"
        )
    if len(versions) > 1:
        raise SubmissionIntegrationError(
            "launch-claim prefix contains multiple object versions"
        )
    if not versions:
        if (
            _strict_claim_get(
                services.s3,
                bucket=bucket,
                key=claim_key,
                allow_absent=True,
            )
            is not None
        ):
            raise SubmissionIntegrationError(
                "launch-claim version history is empty but current object exists"
            )
        return None
    listed = versions[0]
    version_id = listed.get("VersionId")
    if (
        not isinstance(version_id, str)
        or not version_id
        or version_id == "null"
        or type(listed.get("IsLatest")) is not bool
        or listed.get("IsLatest") is not True
    ):
        raise SubmissionIntegrationError(
            "sole launch-claim version is not an exact latest version"
        )
    exact_raw = _strict_claim_get(
        services.s3,
        bucket=bucket,
        key=claim_key,
        version_id=version_id,
    )
    current_raw = _strict_claim_get(
        services.s3,
        bucket=bucket,
        key=claim_key,
        expected_version_id=version_id,
    )
    if exact_raw is None or current_raw is None or exact_raw != current_raw:
        raise SubmissionIntegrationError(
            "launch-claim exact/current readback diverged"
        )
    artifact = _decode_artifact_raw(
        key=claim_key,
        raw=exact_raw,
        label="launch claim",
    )
    return _CacheSeedClaimState(artifact=artifact, version_id=version_id)


def _validate_cache_seed_capability(
    services: SubmissionServices,
    *,
    local: _CacheSeedLocalAuthorities,
    request: CacheSeedRequest,
) -> dict[str, object]:
    descriptor = local.descriptor.value
    run_id = str(descriptor["run_id"])
    inspector = services.cloudformation
    if inspector is None or not hasattr(
        inspector,
        "inspect_cache_seed_launch_capability",
    ):
        raise SubmissionDeploymentRequired(
            "cache-seed launch capability is not deployed"
        )
    capability = inspector.inspect_cache_seed_launch_capability(
        run_id=run_id,
        managed_mode=CACHE_SEED_MODE,
        descriptor_file_sha256=local.descriptor.file_sha256,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        staged_readiness_version_id=request.staged_readiness_version_id,
    )
    if capability is None:
        raise SubmissionDeploymentRequired(
            "cache-seed launch capability is not deployed"
        )
    expected_fields = {
        "run_id",
        "managed_mode",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "staged_readiness_version_id",
        "worker_runtime_mode",
        "launch_route_state",
        "bucket",
        "launch_claim_prefix",
        "claim_bucket_versioning_state",
        "claim_list_bucket_versions_state",
        "claim_conditional_put_enforcement_state",
        "claim_lifecycle_protection_state",
        "claim_delete_object_deny_state",
        "claim_delete_object_version_deny_state",
        "observed_at",
    }
    if not isinstance(capability, Mapping) or set(capability) != expected_fields:
        raise SubmissionIntegrationError(
            "cache-seed launch capability schema mismatch"
        )
    claim_key = cache_seed_launch_claim_s3_key(
        run_id=run_id,
        descriptor_file_sha256=local.descriptor.file_sha256,
    )
    exact = {
        "run_id": run_id,
        "managed_mode": CACHE_SEED_MODE,
        "descriptor_file_sha256": local.descriptor.file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "staged_readiness_version_id": request.staged_readiness_version_id,
        "worker_runtime_mode": "cache-seed-v1",
        "launch_route_state": "ENABLED",
        "bucket": descriptor["bucket"],
        "launch_claim_prefix": claim_key.removesuffix("LAUNCH_CLAIM.json"),
        "claim_bucket_versioning_state": "ENABLED",
        "claim_list_bucket_versions_state": "ENABLED",
        "claim_conditional_put_enforcement_state": "ENFORCED",
        "claim_lifecycle_protection_state": "ENFORCED",
        "claim_delete_object_deny_state": "ENFORCED",
        "claim_delete_object_version_deny_state": "ENFORCED",
    }
    if any(
        type(capability.get(field)) is not type(expected)
        or capability.get(field) != expected
        for field, expected in exact.items()
    ):
        raise SubmissionIntegrationError(
            "cache-seed launch capability authority drift"
        )
    observed_iso, observed = _whole_second(
        capability.get("observed_at"),
        field="cache-seed capability observed_at",
    )
    del observed_iso
    now = _now(services)
    if observed > now or now - observed >= timedelta(seconds=MAX_LIVE_AGE_SECONDS):
        raise SubmissionIntegrationError(
            "cache-seed launch capability is stale or future-dated"
        )
    return dict(capability)


def _put_cache_seed_immutable(
    services: SubmissionServices,
    *,
    bucket: str,
    run_id: str,
    artifact: _JsonArtifact,
    digest_field: str | None,
) -> None:
    existing = _get_exact(
        services.s3,
        bucket=bucket,
        key=artifact.key,
    )
    if existing is not None:
        if existing != artifact.raw:
            raise SubmissionIntegrationError(
                f"cache-seed immutable winner is foreign: {artifact.key}"
            )
        return
    metadata = {
        "glm52-run-id": run_id,
        "glm52-managed-mode": CACHE_SEED_MODE,
        "glm52-file-sha256": artifact.file_sha256,
    }
    if digest_field is not None:
        body_digest = artifact.value.get(digest_field)
        if not _is_sha(body_digest):
            raise SubmissionIntegrationError(
                f"cache-seed {digest_field} is invalid"
            )
        metadata["glm52-body-sha256"] = str(body_digest)
    try:
        response = services.s3.put_object(
            Bucket=bucket,
            Key=artifact.key,
            Body=artifact.raw,
            IfNoneMatch="*",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=base64.b64encode(
                hashlib.sha256(artifact.raw).digest()
            ).decode("ascii"),
            ContentType="application/json",
            Metadata=metadata,
        )
    except Exception as error:
        winner = _get_exact(
            services.s3,
            bucket=bucket,
            key=artifact.key,
        )
        if winner == artifact.raw:
            return
        code, status = _aws_error(error)
        raise SubmissionIntegrationError(
            f"cache-seed immutable PUT failed: {code or status or error}"
        ) from error
    if (
        not isinstance(response, Mapping)
        or not isinstance(response.get("ResponseMetadata"), Mapping)
        or response["ResponseMetadata"].get("HTTPStatusCode") != 200
    ):
        raise SubmissionIntegrationError(
            "cache-seed immutable PUT response is malformed"
        )
    if (
        _get_exact(
            services.s3,
            bucket=bucket,
            key=artifact.key,
        )
        != artifact.raw
    ):
        raise SubmissionIntegrationError(
            "cache-seed immutable PUT readback drifted"
        )


def _reauthenticate_existing_cache_seed_claim(
    services: SubmissionServices,
    *,
    local: _CacheSeedLocalAuthorities,
    request: CacheSeedRequest,
    claim: _CacheSeedClaimState,
) -> None:
    value = claim.artifact.value
    try:
        references = _prevalidate_cache_seed_launch_claim_references(
            value,
            descriptor=local.descriptor.value,
            descriptor_file_sha256=local.descriptor.file_sha256,
            claim_key=claim.artifact.key,
        )
    except ValueError as error:
        raise SubmissionIntegrationError(
            f"pre-existing launch claim is unauthenticated: {error}"
        ) from error
    bucket = str(local.descriptor.value["bucket"])
    referenced: dict[str, bytes] = {}
    for field, key in references.items():
        raw = _get_exact(
            services.s3,
            bucket=bucket,
            key=key,
        )
        if raw is None:
            raise SubmissionIntegrationError(
                f"pre-existing launch claim reference is absent: {field}"
            )
        referenced[field] = raw
    if referenced["rehearsal_evidence_key"] != local.rehearsal.raw:
        raise SubmissionIntegrationError(
            "pre-existing launch claim rehearsal reference drifted"
        )
    intent = _decode_artifact_raw(
        key=str(value["submission_intent_key"]),
        raw=referenced["submission_intent_key"],
        label="pre-existing cache-seed intent",
    )
    allowance = _decode_artifact_raw(
        key=str(value["launch_allowance_key"]),
        raw=referenced["launch_allowance_key"],
        label="pre-existing launch allowance",
    )
    s3_raw = referenced["s3_spend_ledger_absence_observation_key"]
    ec2_raw = referenced["ec2_tagged_p5_zero_inventory_observation_key"]
    rebuilt = build_gpu_launch_allowance(
        descriptor_raw=local.descriptor.raw,
        expected_descriptor_sha256=local.descriptor.file_sha256,
        approval_raw=local.approval.raw,
        expected_approval_sha256=local.approval.file_sha256,
        s3_spend_ledger_absence_observation_raw=s3_raw,
        expected_s3_spend_ledger_absence_observation_sha256=_sha(s3_raw),
        ec2_tagged_p5_zero_inventory_observation_raw=ec2_raw,
        expected_ec2_tagged_p5_zero_inventory_observation_sha256=_sha(ec2_raw),
        observed_at=str(allowance.value.get("observed_at")),
    )
    if rebuilt != allowance.value:
        raise SubmissionIntegrationError(
            "pre-existing launch allowance cannot be reconstructed"
        )
    validate_cache_seed_submission_intent(
        intent.value,
        descriptor=local.descriptor.value,
        descriptor_file_sha256=local.descriptor.file_sha256,
        approval=local.approval.value,
        approval_file_sha256=local.approval.file_sha256,
        staged_readiness=local.staged.value,
        staged_readiness_file_sha256=local.staged.file_sha256,
        staged_readiness_version_id=request.staged_readiness_version_id,
        bundle_manifest=local.bundle_manifest.value,
        bundle_manifest_file_sha256=local.bundle_manifest.file_sha256,
        rehearsal_evidence=local.rehearsal.value,
        rehearsal_evidence_file_sha256=local.rehearsal.file_sha256,
        s3_spend_ledger_absence_observation_file_sha256=_sha(s3_raw),
        ec2_tagged_p5_zero_inventory_observation_file_sha256=_sha(ec2_raw),
        launch_allowance=allowance.value,
        launch_allowance_file_sha256=allowance.file_sha256,
    )
    validate_cache_seed_launch_claim(
        claim.artifact.value,
        descriptor=local.descriptor.value,
        descriptor_file_sha256=local.descriptor.file_sha256,
        intent=intent.value,
        intent_file_sha256=intent.file_sha256,
        launch_allowance=allowance.value,
        launch_allowance_file_sha256=allowance.file_sha256,
    )
    _cache_seed_remote_staged_chain(
        services,
        descriptor=local.descriptor,
        approval=local.approval,
        staged=local.staged,
        staged_readiness_version_id=request.staged_readiness_version_id,
    )


def _cache_seed_launch_arguments(
    *,
    local: _CacheSeedLocalAuthorities,
    task: Path,
    config: Path,
) -> list[str]:
    descriptor = local.descriptor.value
    run_id = str(descriptor["run_id"])
    bucket = str(descriptor["bucket"])
    arguments = [
        "jobs",
        "launch",
        "--detach-run",
        "--config",
        str(config),
        "-y",
        "-n",
        f"{run_id}-{CACHE_SEED_MODE}",
        "--no-use-spot",
        "--image-id",
        str(descriptor["image_id"]),
    ]
    for key, value in (
        (
            "GLM52_CAMPAIGN_DESCRIPTOR_S3_URI",
            _s3_uri(bucket, local.descriptor.key),
        ),
        ("GLM52_CAMPAIGN_DESCRIPTOR_SHA256", local.descriptor.file_sha256),
        ("GLM52_SKY_JOB_NAME", f"{run_id}-{CACHE_SEED_MODE}"),
        ("GLM52_MANAGED_MODE", CACHE_SEED_MODE),
    ):
        arguments.extend(["--env", f"{key}={value}"])
    arguments.append(str(task))
    return arguments


def run_cache_seed(
    request: CacheSeedRequest,
    *,
    services: SubmissionServices,
) -> SubmissionOutcome:
    """Run one cache-seed command without granting real deployment authority."""

    try:
        identity = _guard_identity(request, services)
        local = _load_cache_seed(request, services)
        caller_arn = identity.get("Arn")
        if not isinstance(caller_arn, str):
            raise SubmissionIntegrationError("AWS caller identity ARN is malformed")
        if request.action == "validate-only":
            artifacts = _cache_seed_live_artifacts(
                services,
                local=local,
                request=request,
                caller_arn=caller_arn,
            )
            return SubmissionOutcome(
                EXIT_OK,
                "cache-seed-valid",
                {
                    "descriptor_file_sha256": local.descriptor.file_sha256,
                    "launch_allowance_file_sha256": artifacts[2].file_sha256,
                    "intent_file_sha256": artifacts[3].file_sha256,
                },
            )
        if request.action != "acquire-and-launch":
            raise SubmissionUsageError(
                f"unsupported cache-seed action: {request.action}"
            )
        initial_claim = _inspect_cache_seed_claim_history(
            services,
            local=local,
        )
        if initial_claim is not None:
            _reauthenticate_existing_cache_seed_claim(
                services,
                local=local,
                request=request,
                claim=initial_claim,
            )
            raise SubmissionReconcilePending(
                "a durable cache-seed launch claim already exists"
            )
        if services.claim_put_s3 is None:
            raise SubmissionIntegrationError(
                "dedicated retry-disabled claim PUT client is absent"
            )
        if not request.waive_launch_claim_protection:
            _validate_cache_seed_capability(
                services,
                local=local,
                request=request,
            )
        s3_observation, ec2_observation, allowance, intent = (
            _cache_seed_live_artifacts(
                services,
                local=local,
                request=request,
                caller_arn=caller_arn,
            )
        )
        descriptor = local.descriptor.value
        bucket = str(descriptor["bucket"])
        run_id = str(descriptor["run_id"])
        for artifact, digest_field in (
            (s3_observation, None),
            (ec2_observation, None),
            (allowance, "allowance_body_sha256"),
            (intent, CACHE_SEED_INTENT_DIGEST_FIELD),
        ):
            _put_cache_seed_immutable(
                services,
                bucket=bucket,
                run_id=run_id,
                artifact=artifact,
                digest_field=digest_field,
            )
        claimed_at = services.clock()
        claim_value = build_cache_seed_launch_claim(
            descriptor=descriptor,
            descriptor_file_sha256=local.descriptor.file_sha256,
            intent=intent.value,
            intent_file_sha256=intent.file_sha256,
            launch_allowance=allowance.value,
            launch_allowance_file_sha256=allowance.file_sha256,
            claimed_at=claimed_at,
        )
        validate_cache_seed_launch_claim(
            claim_value,
            descriptor=descriptor,
            descriptor_file_sha256=local.descriptor.file_sha256,
            intent=intent.value,
            intent_file_sha256=intent.file_sha256,
            launch_allowance=allowance.value,
            launch_allowance_file_sha256=allowance.file_sha256,
        )
        claim_key = cache_seed_launch_claim_s3_key(
            run_id=run_id,
            descriptor_file_sha256=local.descriptor.file_sha256,
        )
        claim_raw = cache_seed_canonical_file_bytes(claim_value)
        claim = _JsonArtifact(claim_key, claim_raw, claim_value)
        claim_metadata = {
            "glm52-run-id": run_id,
            "glm52-managed-mode": CACHE_SEED_MODE,
            "glm52-file-sha256": claim.file_sha256,
            "glm52-body-sha256": str(
                claim.value[CACHE_SEED_CLAIM_DIGEST_FIELD]
            ),
        }
        try:
            response = services.claim_put_s3.put_object(
                Bucket=bucket,
                Key=claim.key,
                Body=claim.raw,
                IfNoneMatch="*",
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=base64.b64encode(
                    hashlib.sha256(claim.raw).digest()
                ).decode("ascii"),
                ContentType="application/json",
                Metadata=claim_metadata,
            )
        except Exception as error:
            winner = _inspect_cache_seed_claim_history(
                services,
                local=local,
            )
            if winner is not None:
                _reauthenticate_existing_cache_seed_claim(
                    services,
                    local=local,
                    request=request,
                    claim=winner,
                )
                return SubmissionOutcome(
                    EXIT_RECONCILE_PENDING,
                    "launch-claim-reconciliation-only",
                    {
                        "reason": (
                            "claim PUT response was ambiguous or raced; "
                            "durable winner requires operator reconciliation"
                        )
                    },
                )
            code, status = _aws_error(error)
            raise SubmissionIntegrationError(
                "single-attempt launch-claim PUT failed without a verifiable "
                f"winner: {code or status or error}"
            ) from error
        if not isinstance(response, Mapping):
            winner = _inspect_cache_seed_claim_history(
                services,
                local=local,
            )
            if winner is not None:
                _reauthenticate_existing_cache_seed_claim(
                    services,
                    local=local,
                    request=request,
                    claim=winner,
                )
                return SubmissionOutcome(
                    EXIT_RECONCILE_PENDING,
                    "launch-claim-reconciliation-only",
                    {
                        "reason": (
                            "claim PUT returned a non-mapping response; "
                            "durable winner requires operator reconciliation"
                        )
                    },
                )
            raise SubmissionIntegrationError(
                "launch-claim PUT returned a non-mapping response "
                "without a verifiable winner"
            )
        response_metadata = response.get("ResponseMetadata")
        response_version = response.get("VersionId")
        if (
            not isinstance(response_metadata, Mapping)
            or type(response_metadata.get("HTTPStatusCode")) is not int
            or response_metadata.get("HTTPStatusCode") != 200
            or not isinstance(response_version, str)
            or not response_version
            or response_version == "null"
        ):
            winner = _inspect_cache_seed_claim_history(
                services,
                local=local,
            )
            if winner is not None:
                _reauthenticate_existing_cache_seed_claim(
                    services,
                    local=local,
                    request=request,
                    claim=winner,
                )
                return SubmissionOutcome(
                    EXIT_RECONCILE_PENDING,
                    "launch-claim-reconciliation-only",
                    {"reason": "claim PUT response did not prove ownership"},
                )
            raise SubmissionIntegrationError(
                "launch-claim PUT response did not prove a VersionId"
            )
        owned = _inspect_cache_seed_claim_history(
            services,
            local=local,
        )
        if (
            owned is None
            or owned.version_id != response_version
            or owned.artifact.raw != claim.raw
        ):
            raise SubmissionIntegrationError(
                "launch-claim PUT VersionId or readback does not own launch"
            )
        gate_two_source_snapshot = _read_worker_start_v2_source_snapshot(
            task=local.task
        )
        gate_two_config_raw = _read_stable_source(
            local.config,
            label="cache-seed Gate-2 Sky config",
        )
        if (
            gate_two_source_snapshot != local.source_snapshot
            or gate_two_config_raw != local.config_raw
        ):
            raise SubmissionIntegrationError(
                "cache-seed task, config, or mount source drifted after validation"
            )
        sealed_bundle = _build_sealed_launch_bundle(
            source_snapshot=local.source_snapshot,
            config_raw=local.config_raw,
        )
        with _materialize_sealed_launch(sealed_bundle) as materialized:
            launch_arguments = _cache_seed_launch_arguments(
                local=local,
                task=materialized.task_path,
                config=materialized.config_path,
            )
            _verify_materialized_launch(
                materialized=materialized,
                bundle=sealed_bundle,
            )
            final_claim = _inspect_cache_seed_claim_history(
                services,
                local=local,
            )
            if (
                final_claim is None
                or final_claim.version_id != owned.version_id
                or final_claim.artifact.raw != claim.raw
            ):
                raise SubmissionIntegrationError(
                    "launch claim drifted before Sky launch"
                )
            if (
                _get_exact(
                    services.s3,
                    bucket=bucket,
                    key=intent.key,
                )
                != intent.raw
            ):
                raise SubmissionIntegrationError(
                    "cache-seed intent drifted before Sky launch"
                )
            _cache_seed_remote_staged_chain(
                services,
                descriptor=local.descriptor,
                approval=local.approval,
                staged=local.staged,
                staged_readiness_version_id=request.staged_readiness_version_id,
            )
            if not request.waive_launch_claim_protection:
                _validate_cache_seed_capability(
                    services,
                    local=local,
                    request=request,
                )
            validate_gpu_launch_allowance(
                allowance.value,
                now=services.clock(),
            )
            launch_result = services.sky.launch(
                launch_arguments,
                cwd=materialized.repository_root,
            )
        if isinstance(launch_result, int):
            return_code = launch_result
        else:
            return_code = getattr(launch_result, "returncode", None)
        if type(return_code) is not int or return_code != 0:
            raise SubmissionIntegrationError(
                "cache-seed Sky launch did not return exact success"
            )
        return SubmissionOutcome(
            EXIT_OK,
            "cache-seed-submitted",
            {
                "descriptor_s3_uri": _s3_uri(bucket, local.descriptor.key),
                "descriptor_file_sha256": local.descriptor.file_sha256,
                "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
                "intent_s3_uri": _s3_uri(bucket, intent.key),
                "launch_claim_protection_waived": (
                    request.waive_launch_claim_protection
                ),
                "intent_file_sha256": intent.file_sha256,
                "intent_body_sha256": intent.value[
                    CACHE_SEED_INTENT_DIGEST_FIELD
                ],
                "allowance_s3_uri": _s3_uri(bucket, allowance.key),
                "allowance_file_sha256": allowance.file_sha256,
                "allowance_body_sha256": allowance.value[
                    "allowance_body_sha256"
                ],
                "claim_s3_uri": _s3_uri(bucket, claim.key),
                "claim_file_sha256": claim.file_sha256,
                "claim_body_sha256": claim.value[
                    CACHE_SEED_CLAIM_DIGEST_FIELD
                ],
            },
        )
    except SubmissionDeploymentRequired as error:
        return SubmissionOutcome(
            EXIT_DEPLOYMENT_REQUIRED,
            "deployment-required",
            {"reason": str(error)},
        )
    except SubmissionReconcilePending as error:
        return SubmissionOutcome(
            EXIT_RECONCILE_PENDING,
            "launch-claim-reconciliation-only",
            {"reason": str(error)},
        )
    except SubmissionUsageError as error:
        return SubmissionOutcome(EXIT_USAGE, "usage-error", {"reason": str(error)})
    except Exception as error:  # noqa: BLE001
        return SubmissionOutcome(
            EXIT_FAIL_CLOSED,
            "fail-closed",
            {"reason": str(error)},
        )


class _Exit64ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _load_task10_canonical_json(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise Task10ProductionRouteError(
            f"{label} is unreadable or invalid"
        ) from exc
    if (
        type(value) is not dict
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise Task10ProductionRouteError(
            f"{label} must be canonical JSON with one trailing newline"
        )
    return value


def _load_task10_request(
    args: argparse.Namespace,
) -> Task10ProductionRequest:
    if (
        args.production_authority is None
        or args.production_task_inputs is None
        or args.output_handoff is None
    ):
        raise Task10ProductionRouteError(
            "production requires --production-authority, "
            "--production-task-inputs, and --output-handoff"
        )
    authority = production_authority_from_mapping(
        _load_task10_canonical_json(
            args.production_authority,
            "production authority",
        )
    )
    task_raw = _load_task10_canonical_json(
        args.production_task_inputs,
        "production task inputs",
    )
    if set(task_raw) != set(MountFreeTaskInputs.__dataclass_fields__):
        raise Task10ProductionRouteError(
            "production task-input field set drifted"
        )
    try:
        task_inputs = validate_task_inputs(MountFreeTaskInputs(**task_raw))
    except (TypeError, ValueError) as exc:
        raise Task10ProductionRouteError(
            "production task inputs are invalid"
        ) from exc
    try:
        descriptor_raw = args.descriptor.read_bytes()
    except OSError as exc:
        raise Task10ProductionRouteError(
            "production descriptor is unreadable"
        ) from exc
    try:
        descriptor_value = json.loads(descriptor_raw)
    except json.JSONDecodeError as exc:
        raise Task10ProductionRouteError(
            "production descriptor is invalid JSON"
        ) from exc
    if (
        type(descriptor_value) is not dict
        or descriptor_raw
        != canonical_json_bytes(descriptor_value) + b"\n"
        or hashlib.sha256(descriptor_raw).hexdigest()
        != task_inputs.descriptor_file_sha256
    ):
        raise Task10ProductionRouteError(
            "canonical production descriptor identity drifted"
        )
    try:
        worker_descriptor = worker_bootstrap_descriptor_from_mapping(
            descriptor_value
        )
    except ValueError as exc:
        raise Task10ProductionRouteError(
            "production worker-bootstrap descriptor is invalid"
        ) from exc
    if (
        worker_descriptor.campaign_identity_sha256
        != authority.campaign_identity_sha256
        or worker_descriptor.activation_id != authority.activation_id
        or worker_descriptor.generation != authority.generation
        or worker_descriptor.descriptor_body_sha256
        != authority.descriptor_identity_sha256
        or worker_descriptor.archive_identity_sha256
        != authority.archive_identity_sha256
        or worker_descriptor.approval_identity_sha256
        != authority.approval_identity_sha256
        or worker_descriptor.intent_identity_sha256
        != authority.intent_identity_sha256
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
    ):
        raise Task10ProductionRouteError(
            "production descriptor Task 8/9/artifact ancestry drifted"
        )
    return Task10ProductionRequest(
        action=args.action,
        profile=args.profile,
        authority=authority,
        task_inputs=task_inputs,
        worker_descriptor=worker_descriptor,
    )


def _load_task13_launch_boundary(
    args: argparse.Namespace,
    request: Task10ProductionRequest,
) -> Task13LaunchBoundary:
    if args.task13_launch_bridge is None:
        raise Task10ProductionRouteError(
            "default production route requires --task13-launch-bridge"
        )
    try:
        bridge = task13_launch_bridge_from_mapping(
            _load_task10_canonical_json(
                args.task13_launch_bridge,
                "Task 13 launch bridge",
            )
        )
        return Task13LaunchBoundary(bridge, repo_root=REPO_ROOT)
    except Task13BridgeError as error:
        raise Task10ProductionRouteError(str(error)) from error


@dataclass
class _ReservedTask10OutcomeSink:
    path: Path
    descriptor: int
    device: int
    inode: int
    committed: bool = False

    def _assert_owned_path(self) -> None:
        try:
            current = self.path.lstat()
        except OSError as exc:
            raise Task10ProductionRouteError(
                "production outcome sink disappeared after reservation"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != self.device
            or current.st_ino != self.inode
        ):
            raise Task10ProductionRouteError(
                "production outcome sink identity changed after reservation"
            )

    def write(self, raw: bytes) -> None:
        self._assert_owned_path()
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        os.ftruncate(self.descriptor, 0)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(self.descriptor, remaining)
            if written <= 0:
                raise OSError("production outcome sink made no write progress")
            remaining = remaining[written:]
        os.fsync(self.descriptor)
        self._assert_owned_path()
        self.committed = True

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        finally:
            if not self.committed:
                try:
                    current = self.path.lstat()
                except OSError:
                    return
                if (
                    stat.S_ISREG(current.st_mode)
                    and current.st_dev == self.device
                    and current.st_ino == self.inode
                ):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass


def _reserve_task10_outcome_sink(path: Path) -> _ReservedTask10OutcomeSink:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise Task10ProductionRouteError(
            "production outcome sink parent must be a real directory"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    sink: _ReservedTask10OutcomeSink | None = None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise Task10ProductionRouteError(
                "production outcome sink must be a unique regular file"
            )
        sink = _ReservedTask10OutcomeSink(
            path=path,
            descriptor=descriptor,
            device=opened.st_dev,
            inode=opened.st_ino,
        )
        sink._assert_owned_path()
        os.write(descriptor, b"\n")
        os.fsync(descriptor)
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.fsync(descriptor)
        sink._assert_owned_path()
        return sink
    except BaseException:
        if sink is None:
            os.close(descriptor)
        else:
            sink.close()
        raise


def _write_task10_outcome(
    sink: _ReservedTask10OutcomeSink,
    outcome: Task10ProductionOutcome,
) -> None:
    payload = {
        "exit_code": outcome.exit_code,
        "status": outcome.status,
        **dict(outcome.detail),
    }
    raw = canonical_json_bytes(payload) + b"\n"
    sink.write(raw)


def _parser() -> argparse.ArgumentParser:
    parser = _Exit64ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["qualification", "cache-seed", "production"],
    )
    parser.add_argument(
        "action",
        choices=[
            "validate-only",
            "prepare-intent",
            "acquire-and-launch",
            "reconcile",
            "start",
        ],
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--seed-descriptor", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--staged-ready", type=Path)
    parser.add_argument("--staged-ready-version-id")
    parser.add_argument("--rehearsal-evidence", type=Path)
    parser.add_argument("--cache-seed-accepted", type=Path)
    parser.add_argument("--gpu-spend-snapshot", type=Path)
    parser.add_argument("--qualification-ready", type=Path)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-handoff", type=Path)
    parser.add_argument("--intent-s3-uri")
    parser.add_argument("--must-start-ready-s3-uri")
    parser.add_argument("--acquisition-s3-uri")
    parser.add_argument("--sky-bin")
    parser.add_argument("--production-authority", type=Path)
    parser.add_argument("--production-task-inputs", type=Path)
    parser.add_argument("--task13-launch-bridge", type=Path)
    parser.add_argument(
        "--waive-cache-seed-launch-capability",
        action="store_true",
        help=(
            "explicitly waive only the deployed cache-seed launch-capability "
            "proof while retaining the conditional one-shot launch claim"
        ),
    )
    return parser


class _SubprocessSky:
    def __init__(self, sky_bin: str):
        self.sky_bin = sky_bin
        self._validated_sky_path: Path | None = None
        self._validated_sky_sha256: str | None = None

    def _inspect_sky_entrypoint(self) -> tuple[Path, str]:
        sky_path = Path(self.sky_bin)
        if not sky_path.is_absolute():
            raise SubmissionIntegrationError(
                "Sky executable path must be absolute"
            )
        try:
            metadata = sky_path.lstat()
            resolved = sky_path.resolve()
        except OSError as exc:
            raise SubmissionIntegrationError(
                "Sky executable is absent or unreadable"
            ) from exc
        if (
            resolved != sky_path
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or not os.access(sky_path, os.X_OK)
        ):
            raise SubmissionIntegrationError(
                "Sky executable must be an exact executable regular file"
            )
        if sky_path.name != "sky":
            raise SubmissionIntegrationError(
                "Sky executable must be the pinned sky entry point"
            )
        python_path = sky_path.with_name("python")
        if not python_path.is_file() or not os.access(python_path, os.X_OK):
            raise SubmissionIntegrationError(
                "Sky entry point has no executable sibling Python"
            )
        try:
            raw = sky_path.read_bytes()
        except OSError as exc:
            raise SubmissionIntegrationError(
                "Sky executable is unreadable"
            ) from exc
        if (
            not raw.startswith(f"#!{python_path}\n".encode())
            or b"from sky.cli import cli" not in raw
            or b"sys.exit(cli())" not in raw
        ):
            raise SubmissionIntegrationError(
                "Sky executable is not a SkyPilot console entry point"
            )
        return sky_path, _sha(raw)

    def validate_control_plane(self, *, task: Path, config: Path) -> dict[str, object]:
        self._validated_sky_path = None
        self._validated_sky_sha256 = None
        sky_path, initial_sha256 = self._inspect_sky_entrypoint()
        try:
            version_process = subprocess.run(
                [str(sky_path), "--version"],
                text=True,
                capture_output=True,
                check=False,
                timeout=SKY_VALIDATION_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SubmissionIntegrationError(
                "SkyPilot version check failed"
            ) from exc
        expected_output = (
            f"skypilot, version {EXPECTED_SKYPILOT_VERSION}"
        )
        reported_output = version_process.stdout.strip()
        if (
            version_process.returncode != 0
            or reported_output != expected_output
        ):
            raise SubmissionIntegrationError(
                "SkyPilot executable must report exactly "
                f"{expected_output!r}"
            )
        observed_version = reported_output.removeprefix(
            "skypilot, version "
        )
        try:
            process = subprocess.run(
                [
                    str(sky_path.with_name("python")),
                    str(
                        REPO_ROOT
                        / "aws/glm52-gpu/scripts/"
                        "validate_skypilot_control_plane.py"
                    ),
                    "--task",
                    str(task),
                    "--config",
                    str(config),
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=SKY_VALIDATION_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SubmissionIntegrationError(
                "SkyPilot parser validation failed"
            ) from exc
        if process.returncode != 0:
            raise SubmissionIntegrationError(
                process.stderr.strip() or process.stdout.strip()
            )
        try:
            parser_validation = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise SubmissionIntegrationError(
                "SkyPilot parser validation returned malformed evidence"
            ) from exc
        if (
            type(parser_validation) is not dict
            or parser_validation.get("record_type")
            != "glm52_skypilot_parser_validation_v1"
            or parser_validation.get("skypilot_version")
            != observed_version
            or parser_validation.get("status") != "passed"
        ):
            raise SubmissionIntegrationError(
                "SkyPilot parser validation evidence is invalid"
            )
        final_path, final_sha256 = self._inspect_sky_entrypoint()
        if final_path != sky_path or final_sha256 != initial_sha256:
            raise SubmissionIntegrationError(
                "SkyPilot executable changed during validation"
            )
        self._validated_sky_path = final_path
        self._validated_sky_sha256 = final_sha256
        return {
            "skypilot_version": observed_version,
            "task_file_sha256": _sha(task.read_bytes()),
            "config_file_sha256": _sha(config.read_bytes()),
        }

    def launch(
        self,
        argv: list[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        if (
            self._validated_sky_path is None
            or self._validated_sky_sha256 is None
        ):
            raise SubmissionIntegrationError(
                "Sky executable must be validated before launch"
            )
        sky_path, sky_sha256 = self._inspect_sky_entrypoint()
        if (
            sky_path != self._validated_sky_path
            or sky_sha256 != self._validated_sky_sha256
        ):
            raise SubmissionIntegrationError(
                "validated Sky executable identity changed before launch"
            )
        return subprocess.run(
            [str(sky_path), *argv],
            text=True,
            capture_output=True,
            check=False,
            timeout=SKY_LAUNCH_TIMEOUT_SECONDS,
            cwd=cwd,
        )


class _BotoCloudFormationInspector:
    """Read-only, fail-closed inspector for the dynamic must-start stack."""

    _REVIEWED_TEMPLATE_CANONICAL_SHA256 = (
        "c6f3d8f72740944a35c3fbc10fb075a532e541febaac8596d8b755519431dae8"
    )
    _ABSENT_SDK_ERROR_CODES = frozenset(
        {
            "AWS.SimpleQueueService.NonExistentQueue",
            "NoSuchEntity",
            "QueueDoesNotExist",
            "ResourceNotFound",
            "ResourceNotFoundException",
            "ValidationError",
        }
    )
    _STACK_NAME = "keep-glm52-gpu"
    _COMPLETE_STACK_STATUSES = frozenset(
        {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
    )
    _COMPLETE_RESOURCE_STATUSES = frozenset(
        {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
    )
    _RESOURCE_TYPES = {
        "ModelBucket": "AWS::S3::Bucket",
        "CampaignAlertTopic": "AWS::SNS::Topic",
        "SkyMustStartCancelDeadLetterQueue": "AWS::SQS::Queue",
        "SkyMustStartCancelRole": "AWS::IAM::Role",
        "SkyMustStartCancelFunction": "AWS::Lambda::Function",
        "SkyMustStartCancelLogGroup": "AWS::Logs::LogGroup",
        "SkyMustStartCancelDeadLetterQueuePolicy": "AWS::SQS::QueuePolicy",
        "SkyMustStartCancelRule": "AWS::Events::Rule",
        "SkyMustStartSchedulerRole": "AWS::IAM::Role",
        "SkyMustStartDeadlineSchedule": "AWS::Scheduler::Schedule",
    }
    _MAPPING_FIELDS = frozenset(
        {
            "run_id",
            "managed_mode",
            "intent_body_sha256",
            "controller_baseline_body_sha256",
            "template_sha256",
            "lambda_function_arn",
            "lambda_code_sha256",
            "iam_policy_sha256",
            "reconciliation_rule_arn",
            "deadline_schedule_arn",
            "dlq_arn",
            "reconciliation_rule_state",
            "deadline_schedule_state",
            "coordinator_mode",
        }
    )

    def __init__(
        self,
        client: Any,
        *,
        lambda_client: Any | None = None,
        iam_client: Any | None = None,
        events_client: Any | None = None,
        scheduler_client: Any | None = None,
        sqs_client: Any | None = None,
        logs_client: Any | None = None,
        ssm_client: Any | None = None,
        stack_name: str = _STACK_NAME,
        clock: Callable[[], datetime] | None = None,
    ):
        self.client = client
        self.lambda_client = lambda_client
        self.iam_client = iam_client
        self.events_client = events_client
        self.scheduler_client = scheduler_client
        self.sqs_client = sqs_client
        self.logs_client = logs_client
        self.ssm_client = ssm_client
        self.stack_name = stack_name
        self.clock = clock or (lambda: datetime.now(UTC).replace(microsecond=0))

    @staticmethod
    def _response_mapping(value: object, *, label: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise SubmissionIntegrationError(f"{label} response is malformed")
        return value

    @staticmethod
    def _canonical_sha256(value: object) -> str:
        try:
            raw = canonical_json_bytes(value)
        except (TypeError, ValueError) as error:
            raise SubmissionIntegrationError(
                "CloudFormation authority is not canonical JSON"
            ) from error
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _load_template_body(value: object, *, stage: str) -> dict[str, object]:
        if isinstance(value, Mapping):
            loaded: object = dict(value)
        elif isinstance(value, str):
            try:
                import yaml

                class _CloudFormationLoader(yaml.SafeLoader):
                    pass

                def _construct_tag(
                    loader: Any,
                    suffix: str,
                    node: Any,
                ) -> object:
                    if isinstance(node, yaml.ScalarNode):
                        payload: object = loader.construct_scalar(node)
                    elif isinstance(node, yaml.SequenceNode):
                        payload = loader.construct_sequence(node, deep=True)
                    elif isinstance(node, yaml.MappingNode):
                        payload = loader.construct_mapping(node, deep=True)
                    else:
                        raise SubmissionIntegrationError(
                            f"{stage} template contains an unknown YAML node"
                        )
                    intrinsic_names = {
                        "And": "Fn::And",
                        "Base64": "Fn::Base64",
                        "Cidr": "Fn::Cidr",
                        "Equals": "Fn::Equals",
                        "FindInMap": "Fn::FindInMap",
                        "ForEach": "Fn::ForEach",
                        "GetAZs": "Fn::GetAZs",
                        "If": "Fn::If",
                        "ImportValue": "Fn::ImportValue",
                        "Join": "Fn::Join",
                        "Not": "Fn::Not",
                        "Or": "Fn::Or",
                        "Select": "Fn::Select",
                        "Split": "Fn::Split",
                        "Sub": "Fn::Sub",
                        "Transform": "Fn::Transform",
                    }
                    if suffix == "Ref":
                        return {"Ref": payload}
                    if suffix == "Condition":
                        return {"Condition": payload}
                    if suffix == "GetAtt":
                        if isinstance(payload, str):
                            payload = payload.split(".", 1)
                        return {"Fn::GetAtt": payload}
                    intrinsic = intrinsic_names.get(suffix)
                    if intrinsic is None:
                        raise SubmissionIntegrationError(
                            f"{stage} template contains unknown tag !{suffix}"
                        )
                    return {intrinsic: payload}

                _CloudFormationLoader.add_multi_constructor(
                    "!",
                    _construct_tag,
                )
                loaded = yaml.load(value, Loader=_CloudFormationLoader)
            except SubmissionIntegrationError:
                raise
            except (TypeError, ValueError, yaml.YAMLError) as error:
                raise SubmissionIntegrationError(
                    f"{stage} template body is malformed"
                ) from error
        else:
            raise SubmissionIntegrationError(
                f"{stage} template body is malformed"
            )
        if not isinstance(loaded, dict):
            raise SubmissionIntegrationError(
                f"{stage} template body is malformed"
            )
        return loaded

    @staticmethod
    def _stack_coordinates(stack_id: object) -> tuple[str, str, str] | None:
        if not isinstance(stack_id, str):
            return None
        pieces = stack_id.split(":", 5)
        if len(pieces) != 6:
            return None
        arn, partition, service, region, account_id, resource = pieces
        if (
            arn != "arn"
            or partition != "aws"
            or service != "cloudformation"
            or not resource.startswith("stack/")
        ):
            return None
        stack_parts = resource.split("/")
        if len(stack_parts) != 3 or not all(stack_parts):
            return None
        return region, account_id, stack_parts[1]

    @staticmethod
    def _parameter_map(stack: Mapping[str, object]) -> dict[str, str]:
        raw = stack.get("Parameters")
        if not isinstance(raw, list):
            raise SubmissionIntegrationError(
                "CloudFormation stack parameters are malformed"
            )
        parameters: dict[str, str] = {}
        for entry in raw:
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"ParameterKey", "ParameterValue"}
                or not isinstance(entry.get("ParameterKey"), str)
                or not isinstance(entry.get("ParameterValue"), str)
                or entry["ParameterKey"] in parameters
            ):
                raise SubmissionIntegrationError(
                    "CloudFormation stack parameters are malformed"
                )
            parameters[str(entry["ParameterKey"])] = str(entry["ParameterValue"])
        return parameters

    @classmethod
    def _resolve_intrinsics(
        cls,
        value: object,
        *,
        refs: Mapping[str, str],
        get_atts: Mapping[tuple[str, str], str],
    ) -> object:
        if isinstance(value, list):
            return [
                cls._resolve_intrinsics(item, refs=refs, get_atts=get_atts)
                for item in value
            ]
        if not isinstance(value, Mapping):
            return value
        if set(value) == {"Ref"}:
            target = value["Ref"]
            if not isinstance(target, str) or target not in refs:
                raise SubmissionIntegrationError(
                    "reviewed policy contains an unknown Ref"
                )
            return refs[target]
        if set(value) == {"Fn::GetAtt"}:
            target = value["Fn::GetAtt"]
            if (
                not isinstance(target, list)
                or len(target) != 2
                or not all(isinstance(item, str) for item in target)
                or (str(target[0]), str(target[1])) not in get_atts
            ):
                raise SubmissionIntegrationError(
                    "reviewed policy contains an unknown Fn::GetAtt"
                )
            return get_atts[(str(target[0]), str(target[1]))]
        if set(value) == {"Fn::Sub"}:
            template = value["Fn::Sub"]
            if not isinstance(template, str):
                raise SubmissionIntegrationError(
                    "reviewed policy contains an unsupported Fn::Sub"
                )

            def replace(match: Any) -> str:
                name = str(match.group(1))
                if "." in name:
                    logical_id, attribute = name.split(".", 1)
                    replacement = get_atts.get((logical_id, attribute))
                else:
                    replacement = refs.get(name)
                if replacement is None:
                    raise SubmissionIntegrationError(
                        "reviewed policy contains an unknown Fn::Sub variable"
                    )
                return replacement

            return re.sub(r"\$\{([^{}]+)\}", replace, template)
        if any(
            key == "Ref" or str(key).startswith("Fn::")
            for key in value
        ):
            raise SubmissionIntegrationError(
                "reviewed policy contains an unsupported intrinsic"
            )
        return {
            str(key): cls._resolve_intrinsics(
                item,
                refs=refs,
                get_atts=get_atts,
            )
            for key, item in value.items()
        }

    def _stack_resource(
        self,
        *,
        stack_id: str,
        logical_id: str,
        resource_type: str,
    ) -> Mapping[str, object] | None:
        response = self._response_mapping(
            self.client.describe_stack_resource(
                StackName=stack_id,
                LogicalResourceId=logical_id,
            ),
            label=f"CloudFormation {logical_id}",
        )
        detail = response.get("StackResourceDetail")
        if not isinstance(detail, Mapping):
            raise SubmissionIntegrationError(
                f"CloudFormation {logical_id} response is malformed"
            )
        required = {
            "StackId": stack_id,
            "LogicalResourceId": logical_id,
            "ResourceType": resource_type,
        }
        if any(detail.get(key) != expected for key, expected in required.items()):
            return None
        if detail.get("ResourceStatus") not in self._COMPLETE_RESOURCE_STATUSES:
            return None
        if not isinstance(detail.get("PhysicalResourceId"), str) or not str(
            detail["PhysicalResourceId"]
        ):
            raise SubmissionIntegrationError(
                f"CloudFormation {logical_id} response is malformed"
            )
        return detail

    @staticmethod
    def _document(value: object, *, label: str) -> dict[str, object]:
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            try:
                from urllib.parse import unquote

                loaded = json.loads(unquote(value))
            except (TypeError, ValueError) as error:
                raise SubmissionIntegrationError(f"{label} is malformed") from error
            if isinstance(loaded, dict):
                return loaded
        raise SubmissionIntegrationError(f"{label} is malformed")

    @staticmethod
    def _lambda_code_sha256(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, TypeError):
            return None
        if len(decoded) != hashlib.sha256().digest_size:
            return None
        return decoded.hex()

    @staticmethod
    def _template_resource(
        resources: Mapping[str, object],
        logical_id: str,
        resource_type: str,
    ) -> Mapping[str, object] | None:
        resource = resources.get(logical_id)
        if (
            not isinstance(resource, Mapping)
            or resource.get("Type") != resource_type
            or not isinstance(resource.get("Properties"), Mapping)
        ):
            return None
        return resource

    @staticmethod
    def _sdk_error_code(error: Exception) -> str | None:
        response = getattr(error, "response", None)
        if not isinstance(response, Mapping):
            return None
        error_detail = response.get("Error")
        if not isinstance(error_detail, Mapping):
            return None
        code = error_detail.get("Code")
        return code if isinstance(code, str) and code else None

    def inspect_dynamic_must_start(
        self,
        *,
        run_id: str,
        managed_mode: str,
        intent_body_sha256: str,
        controller_baseline_body_sha256: str,
    ) -> dict[str, object] | None:
        try:
            return self._inspect_dynamic_must_start(
                run_id=run_id,
                managed_mode=managed_mode,
                intent_body_sha256=intent_body_sha256,
                controller_baseline_body_sha256=(
                    controller_baseline_body_sha256
                ),
            )
        except Exception as error:
            code = self._sdk_error_code(error)
            if code in self._ABSENT_SDK_ERROR_CODES:
                return None
            if code is not None:
                raise SubmissionIntegrationError(
                    f"dynamic must-start AWS inspection failed ({code})"
                ) from error
            raise

    def _inspect_dynamic_must_start(
        self,
        *,
        run_id: str,
        managed_mode: str,
        intent_body_sha256: str,
        controller_baseline_body_sha256: str,
    ) -> dict[str, object] | None:
        clients = (
            self.lambda_client,
            self.iam_client,
            self.events_client,
            self.scheduler_client,
            self.sqs_client,
            self.logs_client,
            self.ssm_client,
        )
        if any(client is None for client in clients):
            return None
        if (
            not isinstance(run_id, str)
            or not run_id
            or managed_mode not in {"production", "qualification", "cache-seed"}
            or not _is_sha(intent_body_sha256)
            or not _is_sha(controller_baseline_body_sha256)
        ):
            return None

        described = self._response_mapping(
            self.client.describe_stacks(StackName=self.stack_name),
            label="CloudFormation describe-stacks",
        )
        stacks = described.get("Stacks")
        if not isinstance(stacks, list) or len(stacks) != 1:
            raise SubmissionIntegrationError(
                "CloudFormation describe-stacks response is malformed"
            )
        stack = self._response_mapping(
            stacks[0],
            label="CloudFormation stack",
        )
        stack_id = stack.get("StackId")
        coordinates = self._stack_coordinates(stack_id)
        if (
            coordinates
            != (APPROVED_REGION, APPROVED_ACCOUNT_ID, self.stack_name)
            or stack.get("StackName") != self.stack_name
            or stack.get("StackStatus") not in self._COMPLETE_STACK_STATUSES
        ):
            return None
        assert isinstance(stack_id, str)
        parameters = self._parameter_map(stack)
        expected_inputs = {
            "EnableSkyPilotSupport": "true",
            "EnableSkyMustStartObserve": "true",
            "EnableSkyMustStartCancel": "true",
            "SkyCampaignRunId": run_id,
            "SkyMustStartManagedMode": managed_mode,
            "SkyMustStartSubmissionBodySha256": intent_body_sha256,
            "SkyMustStartControllerBaselineBodySha256": (
                controller_baseline_body_sha256
            ),
            "SkyMustStartFoundationRevision": "versioned-code-v1",
        }
        if any(parameters.get(key) != value for key, value in expected_inputs.items()):
            return None
        required_sha_parameters = (
            "SkyMustStartCancelCodeSha256",
            "SkyMustStartActivationJobBindingSha256",
            "SkyMustStartActivationObservationSha256",
            "SkyMustStartDescriptorFileSha256",
        )
        if any(not _is_sha(parameters.get(key)) for key in required_sha_parameters):
            return None
        required_parameters = (
            "ProjectTag",
            "SkyMustStartCancelCodeVersionId",
            "SkyMustStartDescriptorRelativeKey",
            "SkyMustStartTargetJobId",
            "SkyMustStartJobName",
            "SkyMustStartBy",
            "SkyMustStartPrimaryWakeAt",
            "SkyMustStartWorkspace",
            "SkyMustStartSubmissionSubmittedAt",
            "SkyMustStartControllerInstanceId",
            "SkyMustStartControllerInstanceType",
            "SkyMustStartControllerProfileArn",
            "SkyMustStartControllerClusterName",
            "SkyMustStartStartingGraceSeconds",
        )
        if any(
            not isinstance(parameters.get(key), str)
            or parameters.get(key) in {"", "disabled"}
            for key in required_parameters
        ):
            return None
        if (
            parameters["SkyMustStartCancelCodeVersionId"] in {"null", "None"}
            or parameters["SkyMustStartTargetJobId"] == "-1"
            or parameters["SkyMustStartWorkspace"] != "default"
        ):
            return None

        original_response = self._response_mapping(
            self.client.get_template(
                StackName=stack_id,
                TemplateStage="Original",
            ),
            label="CloudFormation original template",
        )
        processed_response = self._response_mapping(
            self.client.get_template(
                StackName=stack_id,
                TemplateStage="Processed",
            ),
            label="CloudFormation processed template",
        )
        if "TemplateBody" not in original_response or "TemplateBody" not in (
            processed_response
        ):
            raise SubmissionIntegrationError(
                "CloudFormation get-template response is malformed"
            )
        original_template = self._load_template_body(
            original_response["TemplateBody"],
            stage="original",
        )
        processed_template = self._load_template_body(
            processed_response["TemplateBody"],
            stage="processed",
        )
        template_sha256 = self._canonical_sha256(original_template)
        if template_sha256 != self._canonical_sha256(processed_template):
            return None
        if template_sha256 != self._REVIEWED_TEMPLATE_CANONICAL_SHA256:
            return None
        raw_resources = original_template.get("Resources")
        if not isinstance(raw_resources, Mapping):
            raise SubmissionIntegrationError(
                "CloudFormation template resources are malformed"
            )

        details: dict[str, Mapping[str, object]] = {}
        for logical_id, resource_type in self._RESOURCE_TYPES.items():
            if self._template_resource(
                raw_resources,
                logical_id,
                resource_type,
            ) is None:
                return None
            detail = self._stack_resource(
                stack_id=stack_id,
                logical_id=logical_id,
                resource_type=resource_type,
            )
            if detail is None:
                return None
            details[logical_id] = detail

        physical = {
            logical_id: str(detail["PhysicalResourceId"])
            for logical_id, detail in details.items()
        }
        project_tag = parameters["ProjectTag"]
        function_name = f"{project_tag}-sky-must-start-cancel"
        rule_name = function_name
        schedule_name = f"{project_tag}-sky-must-start-deadline"
        log_group_name = f"/aws/lambda/{function_name}"
        dlq_name = f"{project_tag}-sky-must-start-cancel-dlq"
        dlq_url = (
            f"https://sqs.{APPROVED_REGION}.amazonaws.com/"
            f"{APPROVED_ACCOUNT_ID}/{dlq_name}"
        )
        lambda_arn = (
            f"arn:aws:lambda:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:"
            f"function:{function_name}"
        )
        rule_arn = (
            f"arn:aws:events:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:"
            f"rule/{rule_name}"
        )
        schedule_arn = (
            f"arn:aws:scheduler:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:"
            f"schedule/default/{schedule_name}"
        )
        dlq_arn = (
            f"arn:aws:sqs:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:{dlq_name}"
        )
        cancel_role_arn = (
            f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
            f"{physical['SkyMustStartCancelRole']}"
        )
        scheduler_role_arn = (
            f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
            f"{physical['SkyMustStartSchedulerRole']}"
        )
        expected_physical = {
            "SkyMustStartCancelDeadLetterQueue": dlq_url,
            "SkyMustStartCancelFunction": function_name,
            "SkyMustStartCancelLogGroup": log_group_name,
            "SkyMustStartCancelRule": rule_name,
            "SkyMustStartDeadlineSchedule": schedule_arn,
        }
        if any(physical[key] != value for key, value in expected_physical.items()):
            return None
        bucket_name = physical["ModelBucket"]
        alert_topic_arn = physical["CampaignAlertTopic"]
        if alert_topic_arn != (
            f"arn:aws:sns:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:"
            f"{project_tag}-campaign-alerts"
        ):
            return None

        refs = {
            **parameters,
            "AWS::AccountId": APPROVED_ACCOUNT_ID,
            "AWS::Partition": "aws",
            "AWS::Region": APPROVED_REGION,
            "ModelBucket": bucket_name,
            "CampaignAlertTopic": alert_topic_arn,
            "SkyMustStartCancelDeadLetterQueue": dlq_url,
            "SkyMustStartCancelFunction": function_name,
        }
        get_atts = {
            ("ModelBucket", "Arn"): f"arn:aws:s3:::{bucket_name}",
            ("SkyMustStartCancelDeadLetterQueue", "Arn"): dlq_arn,
            ("SkyMustStartCancelFunction", "Arn"): lambda_arn,
            ("SkyMustStartCancelRole", "Arn"): cancel_role_arn,
            ("SkyMustStartCancelRule", "Arn"): rule_arn,
            ("SkyMustStartSchedulerRole", "Arn"): scheduler_role_arn,
        }

        role = self._template_resource(
            raw_resources,
            "SkyMustStartCancelRole",
            "AWS::IAM::Role",
        )
        assert role is not None
        role_properties = role["Properties"]
        assert isinstance(role_properties, Mapping)
        policies = role_properties.get("Policies")
        if (
            not isinstance(policies, list)
            or len(policies) != 1
            or not isinstance(policies[0], Mapping)
            or policies[0].get("PolicyName") != "sky-must-start-cancel-only"
            or "PolicyDocument" not in policies[0]
        ):
            return None
        resolved_policy = self._resolve_intrinsics(
            policies[0]["PolicyDocument"],
            refs=refs,
            get_atts=get_atts,
        )
        if not isinstance(resolved_policy, dict):
            raise SubmissionIntegrationError(
                "reviewed must-start IAM policy is malformed"
            )
        reviewed_policy_sha256 = self._canonical_sha256(resolved_policy)

        function_resource = self._template_resource(
            raw_resources,
            "SkyMustStartCancelFunction",
            "AWS::Lambda::Function",
        )
        assert function_resource is not None
        function_properties = function_resource["Properties"]
        assert isinstance(function_properties, Mapping)
        expected_environment = {
            "EXPECTED_ACCOUNT_ID": APPROVED_ACCOUNT_ID,
            "CAMPAIGN_BUCKET": bucket_name,
            "CAMPAIGN_DESCRIPTOR_KEY": (
                f"campaigns/{run_id}/"
                f"{parameters['SkyMustStartDescriptorRelativeKey']}"
            ),
            "IMMUTABLE_SUBMISSION_KEY": (
                f"campaigns/{run_id}/monitor/submission-locks/"
                f"{parameters['SkyMustStartDescriptorFileSha256']}-"
                f"{managed_mode}.json"
            ),
            "SUBMISSION_BODY_SHA256": intent_body_sha256,
            "EXPECTED_CONTROLLER_BASELINE_BODY_SHA256": (
                controller_baseline_body_sha256
            ),
            "EXPECTED_TARGET_JOB_ID": parameters["SkyMustStartTargetJobId"],
            "OBSERVE_ONLY": "false",
            "EXPECTED_WORKSPACE": "default",
            "EXPECTED_DESCRIPTOR_FILE_SHA256": parameters[
                "SkyMustStartDescriptorFileSha256"
            ],
            "EXPECTED_SUBMISSION_SUBMITTED_AT": parameters[
                "SkyMustStartSubmissionSubmittedAt"
            ],
            "EXPECTED_CONTROLLER_INSTANCE_ID": parameters[
                "SkyMustStartControllerInstanceId"
            ],
            "EXPECTED_CONTROLLER_INSTANCE_TYPE": parameters[
                "SkyMustStartControllerInstanceType"
            ],
            "EXPECTED_CONTROLLER_PROFILE_ARN": parameters[
                "SkyMustStartControllerProfileArn"
            ],
            "EXPECTED_CONTROLLER_CLUSTER_NAME": parameters[
                "SkyMustStartControllerClusterName"
            ],
            "STARTING_GRACE_SECONDS": parameters[
                "SkyMustStartStartingGraceSeconds"
            ],
            "PRIMARY_WAKE_MAX_WAIT_SECONDS": "120",
            "RECONCILIATION_RULE_NAME": rule_name,
            "EXPECTED_ACTIVATION_JOB_BINDING_SHA256": parameters[
                "SkyMustStartActivationJobBindingSha256"
            ],
            "EXPECTED_ACTIVATION_OBSERVATION_SHA256": parameters[
                "SkyMustStartActivationObservationSha256"
            ],
            "CAMPAIGN_RUN_ID": run_id,
            "MANAGED_MODE": managed_mode,
            "SKY_JOB_NAME": parameters["SkyMustStartJobName"],
            "MUST_START_BY": parameters["SkyMustStartBy"],
            "ALERT_TOPIC_ARN": alert_topic_arn,
        }
        expected_environment_template = {
            "EXPECTED_ACCOUNT_ID": APPROVED_ACCOUNT_ID,
            "CAMPAIGN_BUCKET": {"Ref": "ModelBucket"},
            "CAMPAIGN_DESCRIPTOR_KEY": {
                "Fn::Sub": (
                    "campaigns/${SkyCampaignRunId}/"
                    "${SkyMustStartDescriptorRelativeKey}"
                )
            },
            "IMMUTABLE_SUBMISSION_KEY": {
                "Fn::Sub": (
                    "campaigns/${SkyCampaignRunId}/monitor/submission-locks/"
                    "${SkyMustStartDescriptorFileSha256}-"
                    "${SkyMustStartManagedMode}.json"
                )
            },
            "SUBMISSION_BODY_SHA256": {
                "Ref": "SkyMustStartSubmissionBodySha256"
            },
            "EXPECTED_CONTROLLER_BASELINE_BODY_SHA256": {
                "Ref": "SkyMustStartControllerBaselineBodySha256"
            },
            "EXPECTED_TARGET_JOB_ID": {"Ref": "SkyMustStartTargetJobId"},
            "OBSERVE_ONLY": {
                "Fn::If": ["SkyMustStartCancelEnabled", "false", "true"]
            },
            "EXPECTED_WORKSPACE": {"Ref": "SkyMustStartWorkspace"},
            "EXPECTED_DESCRIPTOR_FILE_SHA256": {
                "Ref": "SkyMustStartDescriptorFileSha256"
            },
            "EXPECTED_SUBMISSION_SUBMITTED_AT": {
                "Ref": "SkyMustStartSubmissionSubmittedAt"
            },
            "EXPECTED_CONTROLLER_INSTANCE_ID": {
                "Ref": "SkyMustStartControllerInstanceId"
            },
            "EXPECTED_CONTROLLER_INSTANCE_TYPE": {
                "Ref": "SkyMustStartControllerInstanceType"
            },
            "EXPECTED_CONTROLLER_PROFILE_ARN": {
                "Ref": "SkyMustStartControllerProfileArn"
            },
            "EXPECTED_CONTROLLER_CLUSTER_NAME": {
                "Ref": "SkyMustStartControllerClusterName"
            },
            "STARTING_GRACE_SECONDS": {
                "Ref": "SkyMustStartStartingGraceSeconds"
            },
            "PRIMARY_WAKE_MAX_WAIT_SECONDS": "120",
            "RECONCILIATION_RULE_NAME": {
                "Fn::Sub": "${ProjectTag}-sky-must-start-cancel"
            },
            "EXPECTED_ACTIVATION_JOB_BINDING_SHA256": {
                "Ref": "SkyMustStartActivationJobBindingSha256"
            },
            "EXPECTED_ACTIVATION_OBSERVATION_SHA256": {
                "Ref": "SkyMustStartActivationObservationSha256"
            },
            "CAMPAIGN_RUN_ID": {"Ref": "SkyCampaignRunId"},
            "MANAGED_MODE": {"Ref": "SkyMustStartManagedMode"},
            "SKY_JOB_NAME": {"Ref": "SkyMustStartJobName"},
            "MUST_START_BY": {"Ref": "SkyMustStartBy"},
            "ALERT_TOPIC_ARN": {"Ref": "CampaignAlertTopic"},
        }
        template_environment = function_properties.get("Environment")
        expected_code = {
            "S3Bucket": {"Ref": "ModelBucket"},
            "S3Key": {
                "Fn::Sub": (
                    "lambda/sky-must-start-cancel/"
                    "${SkyMustStartCancelCodeSha256}.zip"
                )
            },
            "S3ObjectVersion": {"Ref": "SkyMustStartCancelCodeVersionId"},
        }
        if (
            not isinstance(template_environment, Mapping)
            or template_environment.get("Variables")
            != expected_environment_template
            or function_properties.get("FunctionName")
            != {"Fn::Sub": "${ProjectTag}-sky-must-start-cancel"}
            or function_properties.get("Runtime") != "python3.13"
            or function_properties.get("Handler") != "handler.lambda_handler"
            or function_properties.get("Role")
            != {"Fn::GetAtt": ["SkyMustStartCancelRole", "Arn"]}
            or function_properties.get("ReservedConcurrentExecutions") != 1
            or function_properties.get("DeadLetterConfig")
            != {
                "TargetArn": {
                    "Fn::GetAtt": [
                        "SkyMustStartCancelDeadLetterQueue",
                        "Arn",
                    ]
                }
            }
            or function_properties.get("Timeout") != 300
            or function_properties.get("MemorySize") != 256
            or function_properties.get("Code") != expected_code
        ):
            return None

        queue_policy_resource = self._template_resource(
            raw_resources,
            "SkyMustStartCancelDeadLetterQueuePolicy",
            "AWS::SQS::QueuePolicy",
        )
        assert queue_policy_resource is not None
        queue_policy_properties = queue_policy_resource["Properties"]
        assert isinstance(queue_policy_properties, Mapping)
        if queue_policy_properties.get("Queues") != [
            {"Ref": "SkyMustStartCancelDeadLetterQueue"}
        ]:
            return None
        resolved_queue_policy = self._resolve_intrinsics(
            queue_policy_properties.get("PolicyDocument"),
            refs=refs,
            get_atts=get_atts,
        )
        if not isinstance(resolved_queue_policy, dict):
            raise SubmissionIntegrationError(
                "reviewed must-start queue policy is malformed"
            )

        reviewed_identity = {
            "stack_id": stack_id,
            "template_sha256": template_sha256,
            "lambda_function_arn": lambda_arn,
            "lambda_code_sha256": parameters["SkyMustStartCancelCodeSha256"],
            "iam_policy_sha256": reviewed_policy_sha256,
            "reconciliation_rule_arn": rule_arn,
            "deadline_schedule_arn": schedule_arn,
            "dlq_arn": dlq_arn,
        }

        lambda_response = self._response_mapping(
            self.lambda_client.get_function(FunctionName=function_name),
            label="Lambda get-function",
        )
        configuration = lambda_response.get("Configuration")
        code = lambda_response.get("Code")
        if not isinstance(configuration, Mapping) or not isinstance(code, Mapping):
            raise SubmissionIntegrationError(
                "Lambda get-function response is malformed"
            )
        live_code_sha256 = self._lambda_code_sha256(
            configuration.get("CodeSha256")
        )
        lambda_expected = {
            "FunctionName": function_name,
            "FunctionArn": lambda_arn,
            "Runtime": "python3.13",
            "Role": cancel_role_arn,
            "Handler": "handler.lambda_handler",
            "Timeout": 300,
            "MemorySize": 256,
            "PackageType": "Zip",
            "State": "Active",
            "LastUpdateStatus": "Successful",
            "DeadLetterConfig": {"TargetArn": dlq_arn},
            "Environment": {"Variables": expected_environment},
        }
        if (
            live_code_sha256 is None
            or any(
                configuration.get(key) != value
                for key, value in lambda_expected.items()
            )
            or code.get("RepositoryType") != "S3"
        ):
            return None
        concurrency = self._response_mapping(
            self.lambda_client.get_function_concurrency(
                FunctionName=function_name,
            ),
            label="Lambda get-function-concurrency",
        )
        if concurrency.get("ReservedConcurrentExecutions") != 1:
            return None

        live_role_response = self._response_mapping(
            self.iam_client.get_role_policy(
                RoleName=physical["SkyMustStartCancelRole"],
                PolicyName="sky-must-start-cancel-only",
            ),
            label="IAM get-role-policy",
        )
        if (
            live_role_response.get("RoleName")
            != physical["SkyMustStartCancelRole"]
            or live_role_response.get("PolicyName")
            != "sky-must-start-cancel-only"
            or "PolicyDocument" not in live_role_response
        ):
            raise SubmissionIntegrationError(
                "IAM get-role-policy response is malformed"
            )
        live_policy = self._document(
            live_role_response["PolicyDocument"],
            label="live must-start IAM policy",
        )
        live_policy_sha256 = self._canonical_sha256(live_policy)
        if live_policy_sha256 != reviewed_policy_sha256:
            return None

        rule_response = self._response_mapping(
            self.events_client.describe_rule(Name=rule_name),
            label="EventBridge describe-rule",
        )
        expected_rule_description = (
            f"KEEP GLM52 must-start cancellation for {run_id} {managed_mode}"
        )
        if any(
            rule_response.get(key) != value
            for key, value in {
                "Name": rule_name,
                "Arn": rule_arn,
                "State": "ENABLED",
                "ScheduleExpression": "rate(1 minute)",
                "Description": expected_rule_description,
            }.items()
        ):
            return None
        targets_response = self._response_mapping(
            self.events_client.list_targets_by_rule(
                Rule=rule_name,
                Limit=100,
            ),
            label="EventBridge list-targets-by-rule",
        )
        targets = targets_response.get("Targets")
        if (
            not isinstance(targets, list)
            or len(targets) != 1
            or "NextToken" in targets_response
            or not isinstance(targets[0], Mapping)
        ):
            return None
        rule_input = json.dumps(
            {
                "campaign_run_id": run_id,
                "managed_mode": managed_mode,
                "sky_job_name": parameters["SkyMustStartJobName"],
                "must_start_by": parameters["SkyMustStartBy"],
                "trigger": "reconcile",
            },
            separators=(",", ":"),
        )
        expected_target = {
            "Id": "sky-must-start-cancel",
            "Arn": lambda_arn,
            "Input": rule_input,
            "RetryPolicy": {
                "MaximumEventAgeInSeconds": 300,
                "MaximumRetryAttempts": 2,
            },
            "DeadLetterConfig": {"Arn": dlq_arn},
        }
        if dict(targets[0]) != expected_target:
            return None

        schedule_response = self._response_mapping(
            self.scheduler_client.get_schedule(
                Name=schedule_name,
                GroupName="default",
            ),
            label="Scheduler get-schedule",
        )
        deadline_input = json.dumps(
            {
                "campaign_run_id": run_id,
                "managed_mode": managed_mode,
                "sky_job_name": parameters["SkyMustStartJobName"],
                "must_start_by": parameters["SkyMustStartBy"],
                "trigger": "primary-deadline",
            },
            separators=(",", ":"),
        )
        schedule_expression = (
            f"at({parameters['SkyMustStartPrimaryWakeAt'].removesuffix('Z')})"
        )
        expected_schedule_target = {
            "Arn": lambda_arn,
            "RoleArn": scheduler_role_arn,
            "Input": deadline_input,
            "RetryPolicy": {
                "MaximumEventAgeInSeconds": 300,
                "MaximumRetryAttempts": 2,
            },
            "DeadLetterConfig": {"Arn": dlq_arn},
        }
        schedule_expected = {
            "Name": schedule_name,
            "GroupName": "default",
            "Arn": schedule_arn,
            "State": "ENABLED",
            "ActionAfterCompletion": "DELETE",
            "ScheduleExpression": schedule_expression,
            "ScheduleExpressionTimezone": "UTC",
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "Target": expected_schedule_target,
            "Description": (
                f"Exact must-start deadline for {run_id} {managed_mode}"
            ),
        }
        if any(
            schedule_response.get(key) != value
            for key, value in schedule_expected.items()
        ):
            return None

        queue_response = self._response_mapping(
            self.sqs_client.get_queue_attributes(
                QueueUrl=dlq_url,
                AttributeNames=["All"],
            ),
            label="SQS get-queue-attributes",
        )
        attributes = queue_response.get("Attributes")
        if not isinstance(attributes, Mapping):
            raise SubmissionIntegrationError(
                "SQS get-queue-attributes response is malformed"
            )
        if (
            attributes.get("QueueArn") != dlq_arn
            or attributes.get("MessageRetentionPeriod") != "1209600"
            or attributes.get("SqsManagedSseEnabled") != "true"
            or "Policy" not in attributes
        ):
            return None
        live_queue_policy = self._document(
            attributes["Policy"],
            label="live must-start queue policy",
        )
        if self._canonical_sha256(live_queue_policy) != self._canonical_sha256(
            resolved_queue_policy
        ):
            return None

        logs_response = self._response_mapping(
            self.logs_client.describe_log_groups(
                logGroupNamePrefix=log_group_name,
                limit=1,
            ),
            label="CloudWatch Logs describe-log-groups",
        )
        log_groups = logs_response.get("logGroups")
        if (
            not isinstance(log_groups, list)
            or len(log_groups) != 1
            or "nextToken" in logs_response
            or not isinstance(log_groups[0], Mapping)
            or log_groups[0].get("logGroupName") != log_group_name
            or log_groups[0].get("retentionInDays") != 14
        ):
            return None

        controller_instance_id = parameters["SkyMustStartControllerInstanceId"]
        ssm_response = self._response_mapping(
            self.ssm_client.describe_instance_information(
                Filters=[
                    {
                        "Key": "InstanceIds",
                        "Values": [controller_instance_id],
                    }
                ],
                MaxResults=5,
            ),
            label="SSM describe-instance-information",
        )
        instance_information = ssm_response.get("InstanceInformationList")
        if (
            not isinstance(instance_information, list)
            or len(instance_information) != 1
            or "NextToken" in ssm_response
            or not isinstance(instance_information[0], Mapping)
            or instance_information[0].get("InstanceId")
            != controller_instance_id
            or instance_information[0].get("PingStatus") != "Online"
        ):
            return None

        observed_identity = {
            "stack_id": stack_id,
            "template_sha256": template_sha256,
            "lambda_function_arn": configuration["FunctionArn"],
            "lambda_code_sha256": live_code_sha256,
            "iam_policy_sha256": live_policy_sha256,
            "reconciliation_rule_arn": rule_response["Arn"],
            "deadline_schedule_arn": schedule_response["Arn"],
            "dlq_arn": attributes["QueueArn"],
        }
        if reviewed_identity != observed_identity:
            return None
        observed_at, _ = _whole_second(
            self.clock(),
            field="must-start deployment inspection time",
        )
        return {
            "reviewed_deployment_identity": reviewed_identity,
            "observed_deployment_identity": observed_identity,
            "reconciliation_rule_state": "ENABLED",
            "deadline_schedule_state": "ENABLED",
            "coordinator_mode": "dynamic-job-binding-active",
            "observed_at": observed_at,
        }

    def inspect_cache_seed_launch_capability(
        self,
        *,
        run_id: str,
        managed_mode: str,
        descriptor_file_sha256: str,
        descriptor_body_sha256: str,
        staged_readiness_version_id: str,
    ) -> dict[str, object] | None:
        _ = (
            self.client,
            self.clock,
            run_id,
            managed_mode,
            descriptor_file_sha256,
            descriptor_body_sha256,
            staged_readiness_version_id,
        )
        # H.1b deliberately has no legacy CloudFormation-output fallback.
        # Production stays deployment-required until a live inspector can
        # establish every frozen cache-seed launch-capability field.
        return None


def _default_services(
    *,
    profile: str,
    sky_bin: str,
) -> SubmissionServices:
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=profile, region_name=APPROVED_REGION)
    ordinary_s3 = session.client("s3")
    claim_put_s3 = session.client(
        "s3",
        config=Config(
            retries={
                "mode": "standard",
                "total_max_attempts": 1,
            }
        ),
    )
    ssm = session.client("ssm")
    cloudformation = _BotoCloudFormationInspector(
        session.client("cloudformation"),
        lambda_client=session.client("lambda"),
        iam_client=session.client("iam"),
        events_client=session.client("events"),
        scheduler_client=session.client("scheduler"),
        sqs_client=session.client("sqs"),
        logs_client=session.client("logs"),
        ssm_client=ssm,
    )
    return SubmissionServices(
        sts=session.client("sts"),
        s3=ordinary_s3,
        ec2=session.client("ec2"),
        ssm=ssm,
        cloudformation=cloudformation,
        sky=_SubprocessSky(sky_bin),
        clock=lambda: datetime.now(UTC).replace(microsecond=0),
        sleep=time.sleep,
        claim_put_s3=claim_put_s3,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    services_factory: Callable[..., SubmissionServices] | None = None,
    production_boundary_factory: Callable[..., object] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.profile != EXACT_PROFILE:
        parser.error(f"--profile must be exactly {EXACT_PROFILE}")
    if args.mode == PRODUCTION_MODE:
        if args.action not in {"validate-only", "start", "reconcile"}:
            parser.error(
                "production action must be validate-only, start, or reconcile"
            )
        forbidden = {
            "--seed-descriptor": args.seed_descriptor,
            "--approval": args.approval,
            "--staged-ready": args.staged_ready,
            "--staged-ready-version-id": args.staged_ready_version_id,
            "--rehearsal-evidence": args.rehearsal_evidence,
            "--cache-seed-accepted": args.cache_seed_accepted,
            "--gpu-spend-snapshot": args.gpu_spend_snapshot,
            "--qualification-ready": args.qualification_ready,
            "--task": args.task,
            "--config": args.config,
            "--intent-s3-uri": args.intent_s3_uri,
            "--must-start-ready-s3-uri": args.must_start_ready_s3_uri,
            "--acquisition-s3-uri": args.acquisition_s3_uri,
            "--sky-bin": args.sky_bin,
        }
        supplied = [flag for flag, value in forbidden.items() if value is not None]
        if supplied:
            parser.error(
                f"{', '.join(supplied)} are forbidden for production"
            )
        outcome_sink: _ReservedTask10OutcomeSink | None = None
        try:
            request = _load_task10_request(args)
            assert args.output_handoff is not None
            outcome_sink = _reserve_task10_outcome_sink(
                args.output_handoff
            )
            if production_boundary_factory is None:
                boundary = _load_task13_launch_boundary(args, request)
                assert_no_raw_effect_surface(boundary)
                outcome = run_production(request, boundary=boundary)
            else:
                boundary = production_boundary_factory(profile=args.profile)
                assert_no_raw_effect_surface(boundary)
                outcome = run_production(request, boundary=boundary)
            _write_task10_outcome(outcome_sink, outcome)
        except (Task10ProductionRouteError, Task13BridgeError, OSError) as error:
            outcome = Task10ProductionOutcome(
                EXIT_FAIL_CLOSED,
                "fail-closed",
                {
                    "managed_mode": "production",
                    "reason": str(error),
                },
            )
        finally:
            if outcome_sink is not None:
                outcome_sink.close()
        stream = (
            sys.stdout
            if outcome.exit_code in (EXIT_OK, EXIT_RECONCILE_PENDING)
            else sys.stderr
        )
        print(
            json.dumps(
                {
                    "exit_code": outcome.exit_code,
                    "status": outcome.status,
                    **dict(outcome.detail),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
            file=stream,
        )
        return outcome.exit_code
    request: QualificationRequest | CacheSeedRequest
    if args.mode == QUALIFICATION_MODE:
        if args.waive_cache_seed_launch_capability:
            parser.error(
                "--waive-cache-seed-launch-capability is valid only for "
                "cache-seed acquire-and-launch"
            )
        if args.staged_ready_version_id is not None:
            parser.error(
                "--staged-ready-version-id is valid only for cache-seed"
            )
        local_values = (
            args.seed_descriptor,
            args.approval,
            args.staged_ready,
            args.rehearsal_evidence,
            args.cache_seed_accepted,
            args.gpu_spend_snapshot,
            args.qualification_ready,
            args.task,
            args.config,
        )
        local = None
        if all(value is not None for value in local_values):
            local = LocalAuthorityPaths(
                descriptor=args.descriptor,
                seed_descriptor=args.seed_descriptor,
                approval=args.approval,
                staged_ready=args.staged_ready,
                rehearsal_evidence=args.rehearsal_evidence,
                cache_seed_accepted=args.cache_seed_accepted,
                gpu_spend_snapshot=args.gpu_spend_snapshot,
                qualification_ready=args.qualification_ready,
                task=args.task,
                config=args.config,
            )
        if args.action != "reconcile" and local is None:
            parser.error(
                "validate-only, prepare-intent, and acquire-and-launch "
                "require all local pins"
            )
        request = QualificationRequest(
            action=args.action,
            profile=args.profile,
            local=local,
            descriptor=args.descriptor,
            output_handoff=args.output_handoff,
            intent_s3_uri=args.intent_s3_uri,
            must_start_ready_s3_uri=args.must_start_ready_s3_uri,
            acquisition_s3_uri=args.acquisition_s3_uri,
            sky_bin=args.sky_bin,
        )
    else:
        if args.action not in {"validate-only", "acquire-and-launch"}:
            parser.error(
                "cache-seed action must be validate-only or acquire-and-launch"
            )
        if (
            args.waive_cache_seed_launch_capability
            and args.action != "acquire-and-launch"
        ):
            parser.error(
                "--waive-cache-seed-launch-capability requires "
                "cache-seed acquire-and-launch"
            )
        forbidden = {
            "--seed-descriptor": args.seed_descriptor,
            "--cache-seed-accepted": args.cache_seed_accepted,
            "--gpu-spend-snapshot": args.gpu_spend_snapshot,
            "--qualification-ready": args.qualification_ready,
            "--output-handoff": args.output_handoff,
            "--intent-s3-uri": args.intent_s3_uri,
            "--must-start-ready-s3-uri": args.must_start_ready_s3_uri,
            "--acquisition-s3-uri": args.acquisition_s3_uri,
        }
        supplied = [flag for flag, value in forbidden.items() if value is not None]
        if supplied:
            parser.error(
                f"{', '.join(supplied)} are qualification-only flags"
            )
        cache_values = (
            args.approval,
            args.staged_ready,
            args.staged_ready_version_id,
            args.rehearsal_evidence,
            args.task,
            args.config,
        )
        if any(value is None for value in cache_values):
            parser.error("cache-seed requires all local pins")
        if (
            not isinstance(args.staged_ready_version_id, str)
            or not args.staged_ready_version_id
            or args.staged_ready_version_id == "null"
        ):
            parser.error(
                "--staged-ready-version-id must be an opaque non-null VersionId"
            )
        request = CacheSeedRequest(
            action=args.action,
            profile=args.profile,
            local=CacheSeedLocalAuthorityPaths(
                descriptor=args.descriptor,
                approval=args.approval,
                staged_ready=args.staged_ready,
                rehearsal_evidence=args.rehearsal_evidence,
                task=args.task,
                config=args.config,
            ),
            staged_readiness_version_id=args.staged_ready_version_id,
            sky_bin=args.sky_bin,
            waive_launch_claim_protection=(
                args.waive_cache_seed_launch_capability
            ),
        )
    if args.action != "reconcile" and args.sky_bin is None:
        parser.error("--sky-bin is required")
    factory = services_factory or _default_services
    try:
        services = factory(profile=args.profile, sky_bin=args.sky_bin or "sky")
    except Exception as error:  # noqa: BLE001
        outcome = SubmissionOutcome(
            EXIT_FAIL_CLOSED,
            "fail-closed",
            {"reason": str(error)},
        )
    else:
        outcome = (
            run_qualification(request, services=services)
            if isinstance(request, QualificationRequest)
            else run_cache_seed(request, services=services)
        )
    stream = (
        sys.stdout
        if outcome.exit_code in (EXIT_OK, EXIT_RECONCILE_PENDING)
        else sys.stderr
    )
    print(
        json.dumps(
            {
                "exit_code": outcome.exit_code,
                "status": outcome.status,
                **dict(outcome.detail),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        file=stream,
    )
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
