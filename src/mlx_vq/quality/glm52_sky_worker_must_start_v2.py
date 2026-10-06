"""Pure qualification-only authority for one SkyPilot worker start.

The caller supplies already-read immutable records and transport identities.
This module performs no filesystem, clock, controller, SkyPilot, or AWS I/O.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]
UTC = _datetime.UTC

try:
    from glm52_sky_must_start_dynamic import (  # type: ignore[import-not-found]
        DynamicJobBindingValidationError,
        dynamic_v2_canonical_bytes,
        dynamic_v2_job_binding_s3_key,
        validate_dynamic_v2_job_binding,
    )
except ModuleNotFoundError:
    from mlx_vq.quality.glm52_sky_must_start_dynamic import (
        DynamicJobBindingValidationError,
        dynamic_v2_canonical_bytes,
        dynamic_v2_job_binding_s3_key,
        validate_dynamic_v2_job_binding,
    )

WorkerStartAction = Literal[
    "accept-initial",
    "accept-managed-recovery",
    "idempotent-complete",
    "wait-for-authority",
    "fail-closed",
]

APPROVED_ACCOUNT_ID = "246813579024"
APPROVED_REGION = "us-west-2"
APPROVED_INSTANCE_TYPE = "p5.48xlarge"
QUALIFICATION_MODE = "qualification"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUCKET = re.compile(
    r"^(?![0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$)"
    r"[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])?$"
)
_INSTANCE_ID = re.compile(r"^i-[0-9a-f]{17}$")
_AMI_ID = re.compile(r"^ami-[0-9a-f]{8,17}$")
_ROLE_ARN = re.compile(r"^arn:aws:iam::246813579024:role/[A-Za-z0-9+=,.@_/-]+$")
_PROFILE_ARN = re.compile(
    r"^arn:aws:iam::246813579024:instance-profile/[A-Za-z0-9+=,.@_/-]+$"
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_ETAG = re.compile(r'^"[0-9a-f]{32}(?:-[1-9][0-9]*)?"$')
_VERSION_ID = re.compile(r"^[A-Za-z0-9._+=:/-]{1,1024}$")

_CONTROLLER_STATUSES = frozenset(
    {
        "PENDING",
        "STARTING",
        "RUNNING",
        "RECOVERING",
        "CANCELLING",
        "SUCCEEDED",
        "FAILED",
        "FAILED_SETUP",
        "FAILED_PRECHECKS",
        "FAILED_NO_RESOURCE",
        "FAILED_CONTROLLER",
        "CANCELLED",
    }
)
_STATUS_PROGRESSIONS = {
    "PENDING": _CONTROLLER_STATUSES,
    "STARTING": _CONTROLLER_STATUSES - {"PENDING"},
    "RUNNING": _CONTROLLER_STATUSES - {"PENDING", "STARTING"},
    "RECOVERING": _CONTROLLER_STATUSES - {"PENDING", "STARTING"},
    "CANCELLING": frozenset({"CANCELLING", "CANCELLED", "FAILED"}),
    "SUCCEEDED": frozenset({"SUCCEEDED"}),
    "FAILED": frozenset({"FAILED"}),
    "FAILED_SETUP": frozenset({"FAILED_SETUP"}),
    "FAILED_PRECHECKS": frozenset({"FAILED_PRECHECKS"}),
    "FAILED_NO_RESOURCE": frozenset({"FAILED_NO_RESOURCE"}),
    "FAILED_CONTROLLER": frozenset({"FAILED_CONTROLLER"}),
    "CANCELLED": frozenset({"CANCELLED"}),
}
_REALIZED_STATUSES = frozenset({"STARTING", "RUNNING", "RECOVERING"})
_WAITING_STATUSES = frozenset({"PENDING", "STARTING", "RECOVERING"})
_WORKER_STATES = frozenset({"pending", "running"})
_MAX_EC2_EVIDENCE_AGE = timedelta(seconds=60)

_ARTIFACT_FIELDS = frozenset(
    {
        "source_snapshot_prefix",
        "source_snapshot_sha256",
        "non_vq_prefix",
        "non_vq_package_sha256",
        "teich_pack_key",
        "teich_pack_sha256",
        "frozen_prompt_pack_key",
        "frozen_prompt_pack_sha256",
        "training_baseline_prefix",
        "training_baseline_sha256",
        "training_config_key",
        "training_config_sha256",
        "artifact_inventory_key",
        "artifact_inventory_sha256",
        "qualification_cache_prefix",
        "qualification_cache_manifest_sha256",
    }
)
_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "run_id",
        "campaign_identity_sha256",
        "account_id",
        "provider",
        "region",
        "instance_type",
        "instance_count",
        "use_spot",
        "max_hourly_cost_usd",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "must_start_by",
        "skypilot_version",
        "task_name",
        "controller_identity",
        "worker_identity",
        "vpc_name",
        "image_id",
        "bucket",
        "jobs_bucket",
        "repo_tar_key",
        "repo_tar_sha256",
        "campaign_descriptor_key",
        "approval_key",
        "approval_sha256",
        "artifacts",
        "descriptor_body_sha256",
    }
)
_LATCH_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "managed_mode",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "intent_key",
        "intent_file_sha256",
        "intent_body_sha256",
        "sky_job_name",
        "must_start_by",
        "repo_tar_sha256",
        "controller_injected_sky_job_id",
        "instance_id",
        "instance_type",
        "image_id",
        "worker_role_arn",
        "instance_identity_document_sha256",
        "ec2_pending_time",
        "entrypoint_observed_at",
        "worker_latch_body_sha256",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "run_id",
        "managed_mode",
        "account_id",
        "region",
        "bucket",
        "descriptor_body_sha256",
        "submission_body_sha256",
        "sky_job_name",
        "must_start_by",
        "target_job_id",
        "workspace",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
        "status",
        "schedule_state",
        "submitted_at",
        "start_at",
        "worker_cluster_name",
        "recovery_count",
        "observed_at",
        "observation_body_sha256",
    }
)
_SUBMISSION_ACCEPTED_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "intent_body_sha256",
        "sky_job_id",
        "sky_job_name",
        "workspace",
        "controller_submitted_at",
        "controller_status",
        "controller_identity",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
        "controller_observed_at",
        "controller_observation_key",
        "controller_observation_file_sha256",
        "controller_observation_body_sha256",
        "controller_observation",
        "job_binding_key",
        "job_binding_file_sha256",
        "job_binding_body_sha256",
        "job_binding",
        "accepted_at",
        "accepted_body_sha256",
    }
)
_WORKER_INSTANCE_FIELDS = frozenset(
    {
        "worker_cluster_name",
        "instance_id",
        "worker_instance_type",
        "worker_image_id",
        "worker_instance_lifecycle",
        "worker_instance_state",
        "worker_instance_launch_time",
        "worker_ray_cluster_name",
        "worker_skypilot_cluster_name",
        "worker_campaign_tags",
        "ec2_observed_at",
    }
)
_ACCEPTANCE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "intent_key",
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_key",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "submission_acquisition_key",
        "submission_acquisition_file_sha256",
        "submission_acquisition_body_sha256",
        "submission_accepted_key",
        "submission_accepted_file_sha256",
        "submission_accepted_body_sha256",
        "job_binding_key",
        "job_binding_file_sha256",
        "job_binding_body_sha256",
        "worker_controller_observation_key",
        "worker_controller_observation_file_sha256",
        "worker_controller_observation_body_sha256",
        "sky_job_id",
        "sky_job_name",
        "worker_cluster_name",
        "recovery_count",
        "worker_instance_type",
        "worker_image_id",
        "worker_instance_lifecycle",
        "worker_instance_state",
        "worker_instance_launch_time",
        "worker_ray_cluster_name",
        "worker_skypilot_cluster_name",
        "worker_campaign_tags",
        "active_campaign_p5_instance_ids",
        "ec2_observed_at",
        "worker_latch_key",
        "worker_latch_file_sha256",
        "worker_latch_body_sha256",
        "worker_latch_version_id",
        "worker_latch_etag",
        "worker_latch_last_modified",
        "instance_id",
        "acceptance_kind",
        "prior_worker_acceptance_key",
        "prior_worker_acceptance_file_sha256",
        "prior_worker_acceptance_body_sha256",
        "accepted_at",
        "worker_acceptance_body_sha256",
    }
)
_LATCH_EVIDENCE_FIELDS = frozenset(
    {
        "worker_latch",
        "worker_latch_key",
        "worker_latch_file_sha256",
        "worker_latch_version_id",
        "worker_latch_etag",
        "worker_latch_last_modified",
    }
)
_OBSERVATION_EVIDENCE_FIELDS = frozenset(
    {
        "worker_controller_observation",
        "worker_controller_observation_key",
        "worker_controller_observation_file_sha256",
        "worker_controller_observation_body_sha256",
    }
)


class WorkerStartValidationError(ValueError):
    """Worker-start evidence is malformed, foreign, ambiguous, or incomplete."""


class _WaitForAuthority(Exception):
    pass


@dataclass(frozen=True)
class WorkerStartDecision:
    """One pure admission decision. No action performs or authorizes I/O."""

    action: WorkerStartAction
    reason: str


def worker_v2_canonical_bytes(value: object) -> bytes:
    """Return compact sorted finite ASCII JSON without a trailing newline."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise WorkerStartValidationError(
            "worker-start authority is not canonical finite JSON"
        ) from error


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise WorkerStartValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(worker_v2_canonical_bytes(value)).hexdigest()


def _file_sha(value: object, *, newline: bool) -> str:
    raw = worker_v2_canonical_bytes(value) + (b"\n" if newline else b"")
    return hashlib.sha256(raw).hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise WorkerStartValidationError(f"{label} must be a mapping")
    return dict(value)


def _exact_self_hashed(
    value: object,
    *,
    fields: frozenset[str],
    digest_field: str,
    schema_version: int,
    record_type: str,
    label: str,
) -> dict[str, object]:
    record = _mapping(value, label=label)
    if (
        set(record) != fields
        or type(record.get("schema_version")) is not int
        or record.get("schema_version") != schema_version
        or record.get("record_type") != record_type
    ):
        raise WorkerStartValidationError(f"{label} schema mismatch")
    digest = _sha(record.get(digest_field), label=f"{label} {digest_field}")
    body = dict(record)
    body.pop(digest_field)
    if digest != _canonical_sha(body):
        raise WorkerStartValidationError(f"{label} body SHA-256 mismatch")
    return record


def _canonical_time(value: datetime | str, *, label: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, OverflowError) as error:
            raise WorkerStartValidationError(f"{label} must be ISO-8601") from error
    else:
        raise WorkerStartValidationError(f"{label} must be ISO-8601")
    if parsed.tzinfo is None:
        raise WorkerStartValidationError(f"{label} must be timezone-aware")
    try:
        return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    except (ValueError, OverflowError) as error:
        raise WorkerStartValidationError(f"{label} cannot be normalized") from error


def _stored_time(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise WorkerStartValidationError(f"{label} must be canonical ISO-8601")
    canonical = _canonical_time(value, label=label)
    if value != canonical:
        raise WorkerStartValidationError(f"{label} is not canonical UTC")
    return datetime.fromisoformat(canonical.replace("Z", "+00:00"))


def _input_time(value: datetime | str, *, label: str) -> tuple[str, datetime]:
    canonical = _canonical_time(value, label=label)
    if isinstance(value, str) and value != canonical:
        raise WorkerStartValidationError(f"{label} is not canonical UTC")
    return canonical, datetime.fromisoformat(canonical.replace("Z", "+00:00"))


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise WorkerStartValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise WorkerStartValidationError(f"{label} must be a nonnegative integer")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise WorkerStartValidationError(f"{label} must be a finite float")
    return value


def _run_id(value: object) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise WorkerStartValidationError("run_id is invalid")
    return value


def _campaign_key(value: object, *, run_id: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(f"campaigns/{run_id}/")
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or "*" in value
        or "?" in value
        or any(character.isspace() for character in value)
        or any(segment in {"", ".", ".."} for segment in value.split("/"))
    ):
        raise WorkerStartValidationError(
            f"{label} must be an exact campaign-scoped key"
        )
    return value


def _descriptor(value: object) -> dict[str, object]:
    descriptor = _exact_self_hashed(
        value,
        fields=_DESCRIPTOR_FIELDS,
        digest_field="descriptor_body_sha256",
        schema_version=2,
        record_type="glm52_sky_campaign_descriptor_v2",
        label="campaign descriptor",
    )
    run_id = _run_id(descriptor.get("run_id"))
    exact_values = {
        "account_id": APPROVED_ACCOUNT_ID,
        "provider": "aws",
        "region": APPROVED_REGION,
        "instance_type": APPROVED_INSTANCE_TYPE,
        "instance_count": 1,
        "use_spot": False,
        "max_hourly_cost_usd": 55.04,
        "approved_gpu_runtime_seconds": 86400,
        "approved_gpu_cost_usd": 1320.96,
        "skypilot_version": "0.13.0",
        "task_name": "glm52-campaign",
    }
    if any(descriptor.get(name) != expected for name, expected in exact_values.items()):
        raise WorkerStartValidationError("campaign descriptor authority is foreign")
    if (
        type(descriptor.get("instance_count")) is not int
        or descriptor["instance_count"] != 1
        or type(descriptor.get("use_spot")) is not bool
        or descriptor["use_spot"] is not False
    ):
        raise WorkerStartValidationError(
            "campaign descriptor worker count or market type is invalid"
        )
    _finite_float(descriptor.get("max_hourly_cost_usd"), label="hourly cost")
    _finite_float(descriptor.get("approved_gpu_cost_usd"), label="approved cost")
    _positive_int(
        descriptor.get("approved_gpu_runtime_seconds"),
        label="approved runtime",
    )
    _stored_time(descriptor.get("must_start_by"), label="descriptor must_start_by")
    bucket = descriptor.get("bucket")
    if (
        not isinstance(bucket, str)
        or _BUCKET.fullmatch(bucket) is None
        or descriptor.get("jobs_bucket") != bucket
    ):
        raise WorkerStartValidationError("campaign descriptor bucket is invalid")
    for field in ("controller_identity", "worker_identity"):
        if (
            not isinstance(descriptor.get(field), str)
            or _ROLE_ARN.fullmatch(str(descriptor[field])) is None
        ):
            raise WorkerStartValidationError(f"campaign descriptor {field} is invalid")
    if descriptor["controller_identity"] == descriptor["worker_identity"]:
        raise WorkerStartValidationError(
            "campaign controller and worker roles must remain distinct"
        )
    if (
        not isinstance(descriptor.get("image_id"), str)
        or _AMI_ID.fullmatch(str(descriptor["image_id"])) is None
    ):
        raise WorkerStartValidationError("campaign descriptor image_id is invalid")
    for field in (
        "repo_tar_sha256",
        "approval_sha256",
        "campaign_identity_sha256",
    ):
        _sha(descriptor.get(field), label=f"campaign descriptor {field}")
    for field in (
        "repo_tar_key",
        "campaign_descriptor_key",
        "approval_key",
    ):
        _campaign_key(descriptor.get(field), run_id=run_id, label=field)
    if not str(descriptor["repo_tar_key"]).startswith(
        f"campaigns/{run_id}/repository/"
    ) or not str(descriptor["campaign_descriptor_key"]).startswith(
        f"campaigns/{run_id}/submissions/"
    ):
        raise WorkerStartValidationError("campaign descriptor key scope is invalid")
    artifacts = _mapping(descriptor.get("artifacts"), label="descriptor artifacts")
    if set(artifacts) != _ARTIFACT_FIELDS or any(
        not isinstance(item, str) or not item for item in artifacts.values()
    ):
        raise WorkerStartValidationError("campaign descriptor artifacts are invalid")
    identity = {
        name: item
        for name, item in descriptor.items()
        if name
        not in {
            "campaign_identity_sha256",
            "descriptor_body_sha256",
            "must_start_by",
            "campaign_descriptor_key",
        }
    }
    if descriptor["campaign_identity_sha256"] != _canonical_sha(identity):
        raise WorkerStartValidationError("campaign identity SHA-256 mismatch")
    return descriptor


def _intent_and_descriptor(
    *,
    descriptor: object,
    intent: object,
) -> tuple[dict[str, object], dict[str, object], str]:
    descriptor_value = _descriptor(descriptor)
    intent_value = _mapping(intent, label="submission intent")
    try:
        intent_key = dynamic_v2_job_binding_s3_key(intent=intent_value).removesuffix(
            "DYNAMIC_JOB_BINDING.json"
        )
    except (DynamicJobBindingValidationError, TypeError, ValueError) as error:
        raise WorkerStartValidationError("submission intent is invalid") from error
    run_id = _run_id(intent_value.get("run_id"))
    expected_intent_key = (
        f"campaigns/{run_id}/submissions/qualification/intents/"
        f"{intent_value['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    exact = {
        "account_id": descriptor_value["account_id"],
        "region": descriptor_value["region"],
        "run_id": descriptor_value["run_id"],
        "managed_mode": QUALIFICATION_MODE,
        "descriptor_key": descriptor_value["campaign_descriptor_key"],
        "descriptor_file_sha256": _file_sha(descriptor_value, newline=True),
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor_value["campaign_identity_sha256"],
        "repo_tar_sha256": descriptor_value["repo_tar_sha256"],
        "must_start_by": descriptor_value["must_start_by"],
        "sky_job_name": f"{run_id}-qualification",
    }
    if any(intent_value.get(name) != expected for name, expected in exact.items()):
        raise WorkerStartValidationError(
            "submission intent and descriptor authority drifted"
        )
    if intent_key != (
        f"campaigns/{run_id}/monitor/must-start/qualification/"
        f"{intent_value['intent_body_sha256']}/"
    ):
        raise WorkerStartValidationError("submission intent dynamic prefix is invalid")
    return descriptor_value, intent_value, expected_intent_key


def _parse_controller_job_id(value: object) -> int:
    if type(value) is int:
        return _positive_int(value, label="controller-injected Sky job ID")
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise WorkerStartValidationError(
            "controller-injected Sky job ID is not a positive decimal integer"
        )
    try:
        return int(value)
    except (ValueError, OverflowError) as error:
        raise WorkerStartValidationError(
            "controller-injected Sky job ID cannot be represented"
        ) from error


def build_worker_start_latch_v2(
    *,
    descriptor: object,
    intent: object,
    controller_injected_sky_job_id: object,
    instance_id: object,
    instance_type: object,
    image_id: object,
    worker_role_arn: object,
    instance_identity_document_sha256: object,
    ec2_pending_time: datetime | str,
    entrypoint_observed_at: datetime | str,
) -> dict[str, object]:
    """Build one immutable, instance-specific qualification worker claim."""

    descriptor_value, intent_value, intent_key = _intent_and_descriptor(
        descriptor=descriptor,
        intent=intent,
    )
    job_id = _parse_controller_job_id(controller_injected_sky_job_id)
    if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
        raise WorkerStartValidationError("worker instance_id is invalid")
    if instance_type != descriptor_value["instance_type"]:
        raise WorkerStartValidationError("worker instance type is foreign")
    if image_id != descriptor_value["image_id"]:
        raise WorkerStartValidationError("worker image is foreign")
    if worker_role_arn != descriptor_value["worker_identity"]:
        raise WorkerStartValidationError("worker IAM role is foreign")
    _sha(
        instance_identity_document_sha256,
        label="instance identity document SHA-256",
    )
    pending_iso, pending = _input_time(ec2_pending_time, label="ec2_pending_time")
    entrypoint_iso, entrypoint = _input_time(
        entrypoint_observed_at,
        label="entrypoint_observed_at",
    )
    if pending > entrypoint:
        raise WorkerStartValidationError(
            "EC2 pending time follows entrypoint observation"
        )
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_worker_start_latch_v2",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": descriptor_value["bucket"],
        "run_id": intent_value["run_id"],
        "managed_mode": QUALIFICATION_MODE,
        "descriptor_key": intent_value["descriptor_key"],
        "descriptor_file_sha256": intent_value["descriptor_file_sha256"],
        "descriptor_body_sha256": intent_value["descriptor_body_sha256"],
        "intent_key": intent_key,
        "intent_file_sha256": _file_sha(intent_value, newline=True),
        "intent_body_sha256": intent_value["intent_body_sha256"],
        "sky_job_name": intent_value["sky_job_name"],
        "must_start_by": intent_value["must_start_by"],
        "repo_tar_sha256": intent_value["repo_tar_sha256"],
        "controller_injected_sky_job_id": job_id,
        "instance_id": instance_id,
        "instance_type": instance_type,
        "image_id": image_id,
        "worker_role_arn": worker_role_arn,
        "instance_identity_document_sha256": instance_identity_document_sha256,
        "ec2_pending_time": pending_iso,
        "entrypoint_observed_at": entrypoint_iso,
    }
    latch = {**body, "worker_latch_body_sha256": _canonical_sha(body)}
    return validate_worker_start_latch_v2(
        latch,
        descriptor=descriptor_value,
        intent=intent_value,
    )


def validate_worker_start_latch_v2(
    value: object,
    *,
    descriptor: object,
    intent: object,
) -> dict[str, object]:
    """Authenticate an exact worker latch without normalizing it."""

    descriptor_value, intent_value, intent_key = _intent_and_descriptor(
        descriptor=descriptor,
        intent=intent,
    )
    latch = _exact_self_hashed(
        value,
        fields=_LATCH_FIELDS,
        digest_field="worker_latch_body_sha256",
        schema_version=2,
        record_type="glm52_sky_worker_start_latch_v2",
        label="worker latch",
    )
    expected = {
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": descriptor_value["bucket"],
        "run_id": intent_value["run_id"],
        "managed_mode": QUALIFICATION_MODE,
        "descriptor_key": intent_value["descriptor_key"],
        "descriptor_file_sha256": intent_value["descriptor_file_sha256"],
        "descriptor_body_sha256": intent_value["descriptor_body_sha256"],
        "intent_key": intent_key,
        "intent_file_sha256": _file_sha(intent_value, newline=True),
        "intent_body_sha256": intent_value["intent_body_sha256"],
        "sky_job_name": intent_value["sky_job_name"],
        "must_start_by": intent_value["must_start_by"],
        "repo_tar_sha256": intent_value["repo_tar_sha256"],
        "instance_type": descriptor_value["instance_type"],
        "image_id": descriptor_value["image_id"],
        "worker_role_arn": descriptor_value["worker_identity"],
    }
    if any(latch.get(name) != item for name, item in expected.items()):
        raise WorkerStartValidationError("worker latch authority is foreign")
    _positive_int(
        latch.get("controller_injected_sky_job_id"),
        label="controller-injected Sky job ID",
    )
    instance_id = latch.get("instance_id")
    if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
        raise WorkerStartValidationError("worker latch instance_id is invalid")
    _sha(
        latch.get("instance_identity_document_sha256"),
        label="instance identity document SHA-256",
    )
    pending = _stored_time(latch.get("ec2_pending_time"), label="ec2_pending_time")
    entrypoint = _stored_time(
        latch.get("entrypoint_observed_at"),
        label="entrypoint_observed_at",
    )
    if pending > entrypoint:
        raise WorkerStartValidationError(
            "EC2 pending time follows entrypoint observation"
        )
    return dict(latch)


def worker_start_latch_v2_s3_key(*, worker_latch: object) -> str:
    """Derive the one immutable content-addressed latch key."""

    latch = _exact_self_hashed(
        worker_latch,
        fields=_LATCH_FIELDS,
        digest_field="worker_latch_body_sha256",
        schema_version=2,
        record_type="glm52_sky_worker_start_latch_v2",
        label="worker latch",
    )
    run_id = _run_id(latch.get("run_id"))
    if (
        latch.get("managed_mode") != QUALIFICATION_MODE
        or not isinstance(latch.get("instance_id"), str)
        or _INSTANCE_ID.fullmatch(str(latch["instance_id"])) is None
    ):
        raise WorkerStartValidationError("worker latch key identity is invalid")
    _sha(latch.get("intent_body_sha256"), label="worker latch intent body SHA-256")
    _sha(
        latch.get("worker_latch_body_sha256"),
        label="worker latch body SHA-256",
    )
    return (
        f"campaigns/{run_id}/monitor/must-start/qualification/"
        f"{latch['intent_body_sha256']}/worker-latches/{latch['instance_id']}/"
        f"{latch['worker_latch_body_sha256']}.json"
    )


def _observation(
    value: object,
    *,
    descriptor: Mapping[str, object],
    intent: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    observation = _exact_self_hashed(
        value,
        fields=_OBSERVATION_FIELDS,
        digest_field="observation_body_sha256",
        schema_version=1,
        record_type="glm52_sky_must_start_controller_observation_v1",
        label=label,
    )
    role_prefix = f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
    expected_profile = (
        f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:instance-profile/"
        f"{str(descriptor['controller_identity']).removeprefix(role_prefix)}"
    )
    expected = {
        "run_id": intent["run_id"],
        "managed_mode": QUALIFICATION_MODE,
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": descriptor["bucket"],
        "descriptor_body_sha256": intent["descriptor_body_sha256"],
        "submission_body_sha256": intent["intent_body_sha256"],
        "sky_job_name": intent["sky_job_name"],
        "must_start_by": intent["must_start_by"],
        "workspace": "default",
        "controller_profile_arn": expected_profile,
    }
    if any(observation.get(name) != item for name, item in expected.items()):
        raise WorkerStartValidationError(f"{label} authority is foreign")
    for field in ("controller_instance_id",):
        if (
            not isinstance(observation.get(field), str)
            or _INSTANCE_ID.fullmatch(str(observation[field])) is None
        ):
            raise WorkerStartValidationError(f"{label} {field} is invalid")
    if (
        not isinstance(observation.get("controller_instance_type"), str)
        or not observation["controller_instance_type"]
    ):
        raise WorkerStartValidationError(f"{label} controller_instance_type is invalid")
    if (
        not isinstance(observation.get("controller_cluster_name"), str)
        or _SAFE_NAME.fullmatch(str(observation["controller_cluster_name"])) is None
    ):
        raise WorkerStartValidationError(f"{label} controller_cluster_name is invalid")
    _positive_int(observation.get("target_job_id"), label=f"{label} target job ID")
    status = observation.get("status")
    if status not in _CONTROLLER_STATUSES:
        raise WorkerStartValidationError(f"{label} status is invalid")
    if (
        not isinstance(observation.get("schedule_state"), str)
        or not observation["schedule_state"]
    ):
        raise WorkerStartValidationError(f"{label} schedule_state is invalid")
    submitted = _stored_time(
        observation.get("submitted_at"), label=f"{label} submitted_at"
    )
    observed = _stored_time(
        observation.get("observed_at"), label=f"{label} observed_at"
    )
    if submitted > observed:
        raise WorkerStartValidationError(f"{label} predates controller submission")
    start_at = observation.get("start_at")
    if start_at is not None:
        started = _stored_time(start_at, label=f"{label} start_at")
        if not submitted <= started <= observed:
            raise WorkerStartValidationError(f"{label} start_at is inconsistent")
    cluster = observation.get("worker_cluster_name")
    if cluster is not None and (
        not isinstance(cluster, str) or _SAFE_NAME.fullmatch(cluster) is None
    ):
        raise WorkerStartValidationError(f"{label} worker cluster is invalid")
    _nonnegative_int(
        observation.get("recovery_count"),
        label=f"{label} recovery_count",
    )
    return observation


def _controller_row(
    observation: Mapping[str, object],
    *,
    controller_identity: object,
) -> dict[str, object]:
    return {
        "sky_job_id": observation["target_job_id"],
        "sky_job_name": observation["sky_job_name"],
        "workspace": observation["workspace"],
        "controller_submitted_at": observation["submitted_at"],
        "controller_status": observation["status"],
        "controller_identity": controller_identity,
    }


def _status_progressed(prior: object, current: object, *, label: str) -> None:
    if (
        not isinstance(prior, str)
        or not isinstance(current, str)
        or current not in _STATUS_PROGRESSIONS.get(prior, frozenset())
    ):
        raise WorkerStartValidationError(f"{label} status did not progress legally")


def _submission_authority(
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
    controller_baseline_key: object,
    controller_baseline_file_sha256: object,
    submission_acquisition: object,
    submission_acquisition_key: object,
    submission_acquisition_file_sha256: object,
    submission_accepted: object,
    submission_accepted_key: object,
    submission_accepted_file_sha256: object,
    job_binding: object,
    job_binding_key: object,
    job_binding_file_sha256: object,
    submission_controller_observation: object,
    submission_controller_observation_key: object,
    submission_controller_observation_file_sha256: object,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    descriptor_value, intent_value, intent_key = _intent_and_descriptor(
        descriptor=descriptor,
        intent=intent,
    )
    baseline = _mapping(controller_baseline, label="controller baseline")
    acquisition = _mapping(
        submission_acquisition,
        label="submission acquisition",
    )
    binding = _mapping(job_binding, label="dynamic job binding")
    submission_observation = _observation(
        submission_controller_observation,
        descriptor=descriptor_value,
        intent=intent_value,
        label="submission controller observation",
    )
    controller_coordinate_fields = (
        "workspace",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
    )
    if any(
        submission_observation.get(field) != baseline.get(field)
        for field in controller_coordinate_fields
    ):
        raise WorkerStartValidationError(
            "submission observation controller coordinates drifted from baseline"
        )
    accepted = _exact_self_hashed(
        submission_accepted,
        fields=_SUBMISSION_ACCEPTED_FIELDS,
        digest_field="accepted_body_sha256",
        schema_version=2,
        record_type="glm52_sky_submission_accepted_v2",
        label="submission accepted",
    )
    run_id = str(intent_value["run_id"])
    expected_baseline_key = (
        f"campaigns/{run_id}/qualification/controller-baselines/"
        f"{baseline.get('baseline_body_sha256')}/CONTROLLER_BASELINE.json"
    )
    expected_acquisition_key = (
        f"campaigns/{run_id}/submissions/qualification/acquisitions/"
        f"{intent_value['descriptor_file_sha256']}/SUBMISSION_ACQUIRED.json"
    )
    expected_accepted_key = (
        f"campaigns/{run_id}/submissions/qualification/accepted/"
        f"{accepted['accepted_body_sha256']}/SKYPILOT_SUBMISSION_ACCEPTED.json"
    )
    expected_submission_observation_key = (
        f"campaigns/{run_id}/monitor/must-start/qualification/"
        f"{intent_value['intent_body_sha256']}/observations/"
        f"{submission_observation['observation_body_sha256']}.json"
    )
    expected_pointers = {
        "controller baseline key": (
            controller_baseline_key,
            expected_baseline_key,
        ),
        "controller baseline file": (
            controller_baseline_file_sha256,
            _file_sha(baseline, newline=True),
        ),
        "submission acquisition key": (
            submission_acquisition_key,
            expected_acquisition_key,
        ),
        "submission acquisition file": (
            submission_acquisition_file_sha256,
            _file_sha(acquisition, newline=True),
        ),
        "submission accepted key": (
            submission_accepted_key,
            expected_accepted_key,
        ),
        "submission accepted file": (
            submission_accepted_file_sha256,
            _file_sha(accepted, newline=True),
        ),
        "job binding key": (
            job_binding_key,
            dynamic_v2_job_binding_s3_key(intent=intent_value),
        ),
        "job binding file": (
            job_binding_file_sha256,
            hashlib.sha256(dynamic_v2_canonical_bytes(binding)).hexdigest(),
        ),
        "submission observation key": (
            submission_controller_observation_key,
            expected_submission_observation_key,
        ),
        "submission observation file": (
            submission_controller_observation_file_sha256,
            _file_sha(submission_observation, newline=False),
        ),
    }
    if any(actual != expected for actual, expected in expected_pointers.values()):
        raise WorkerStartValidationError(
            "submission authority key or file identity drifted"
        )
    embedded_observation = accepted.get("controller_observation")
    embedded_binding = accepted.get("job_binding")
    if embedded_observation != submission_observation or embedded_binding != binding:
        raise WorkerStartValidationError(
            "submission accepted embedded authority drifted"
        )
    accepted_exact = {
        "intent_body_sha256": intent_value["intent_body_sha256"],
        "controller_observation_key": expected_submission_observation_key,
        "controller_observation_file_sha256": _file_sha(
            submission_observation,
            newline=False,
        ),
        "controller_observation_body_sha256": submission_observation[
            "observation_body_sha256"
        ],
        "job_binding_key": dynamic_v2_job_binding_s3_key(intent=intent_value),
        "job_binding_file_sha256": hashlib.sha256(
            dynamic_v2_canonical_bytes(binding)
        ).hexdigest(),
        "job_binding_body_sha256": binding.get("job_binding_body_sha256"),
        "sky_job_id": binding.get("sky_job_id"),
        "sky_job_name": intent_value["sky_job_name"],
        "workspace": "default",
        "controller_submitted_at": submission_observation["submitted_at"],
        "controller_status": submission_observation["status"],
        "controller_identity": descriptor_value["controller_identity"],
        "controller_instance_id": submission_observation["controller_instance_id"],
        "controller_instance_type": submission_observation["controller_instance_type"],
        "controller_profile_arn": submission_observation["controller_profile_arn"],
        "controller_cluster_name": submission_observation["controller_cluster_name"],
        "controller_observed_at": submission_observation["observed_at"],
    }
    if any(accepted.get(name) != item for name, item in accepted_exact.items()):
        raise WorkerStartValidationError("submission accepted authority is foreign")
    accepted_at = _stored_time(
        accepted.get("accepted_at"),
        label="submission accepted_at",
    )
    deterministic = max(
        _stored_time(binding.get("bound_at"), label="job binding bound_at"),
        _stored_time(
            submission_observation.get("observed_at"),
            label="submission observation observed_at",
        ),
    )
    if accepted_at != deterministic:
        raise WorkerStartValidationError("submission accepted_at is not deterministic")
    if _stored_time(
        binding.get("bound_at"),
        label="job binding bound_at",
    ) > _stored_time(
        submission_observation.get("observed_at"),
        label="submission observation observed_at",
    ):
        raise WorkerStartValidationError(
            "job binding follows the immutable submission observation"
        )
    baseline_history = baseline.get("exact_name_history")
    if type(baseline_history) is not list:
        raise WorkerStartValidationError("controller baseline history is invalid")
    current_history = [
        *baseline_history,
        _controller_row(
            submission_observation,
            controller_identity=descriptor_value["controller_identity"],
        ),
    ]
    try:
        binding_value = validate_dynamic_v2_job_binding(
            binding,
            intent=intent_value,
            controller_baseline=baseline,
            acquisition=acquisition,
            descriptor_controller_identity=str(descriptor_value["controller_identity"]),
            current_exact_name_history=current_history,
            now=accepted["accepted_at"],
        )
    except (DynamicJobBindingValidationError, TypeError, ValueError) as error:
        raise WorkerStartValidationError(
            "dynamic-v2 submission ancestry is invalid"
        ) from error
    return (
        descriptor_value,
        intent_value,
        baseline,
        acquisition,
        accepted,
        binding_value,
        submission_observation,
        {
            "intent_key": intent_key,
            "controller_baseline_key": expected_baseline_key,
            "controller_baseline_file_sha256": _file_sha(baseline, newline=True),
            "submission_acquisition_key": expected_acquisition_key,
            "submission_acquisition_file_sha256": _file_sha(
                acquisition,
                newline=True,
            ),
            "submission_accepted_key": expected_accepted_key,
            "submission_accepted_file_sha256": _file_sha(
                accepted,
                newline=True,
            ),
            "job_binding_key": dynamic_v2_job_binding_s3_key(intent=intent_value),
            "job_binding_file_sha256": hashlib.sha256(
                dynamic_v2_canonical_bytes(binding_value)
            ).hexdigest(),
        },
    )


def _worker_observation(
    value: object,
    *,
    key: object,
    file_sha256: object,
    descriptor: Mapping[str, object],
    intent: Mapping[str, object],
    submission_observation: Mapping[str, object],
    binding: Mapping[str, object],
    instance_id: str,
    label: str,
    permit_wait: bool,
) -> dict[str, object]:
    observation = _observation(
        value,
        descriptor=descriptor,
        intent=intent,
        label=label,
    )
    common_fields = (
        "target_job_id",
        "sky_job_name",
        "workspace",
        "submitted_at",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
    )
    if any(
        observation.get(field) != submission_observation.get(field)
        for field in common_fields
    ):
        raise WorkerStartValidationError(
            f"{label} rewrites immutable submission identity"
        )
    if observation.get("target_job_id") != binding.get("sky_job_id"):
        raise WorkerStartValidationError(f"{label} targets a foreign Sky job")
    _status_progressed(
        binding.get("controller_status"),
        observation.get("status"),
        label=label,
    )
    _status_progressed(
        submission_observation.get("status"),
        observation.get("status"),
        label=label,
    )
    if _stored_time(
        observation.get("observed_at"),
        label=f"{label} observed_at",
    ) < _stored_time(
        submission_observation.get("observed_at"),
        label="submission controller observation observed_at",
    ):
        raise WorkerStartValidationError(
            f"{label} predates the immutable submission observation"
        )
    expected_key = (
        f"campaigns/{intent['run_id']}/monitor/must-start/qualification/"
        f"{intent['intent_body_sha256']}/worker-controller-observations/"
        f"{instance_id}/{observation['observation_body_sha256']}.json"
    )
    if key != expected_key or file_sha256 != _file_sha(
        observation,
        newline=False,
    ):
        raise WorkerStartValidationError(f"{label} key or file identity is invalid")
    cluster = observation.get("worker_cluster_name")
    start_at = observation.get("start_at")
    if cluster is None or start_at is None:
        if (
            permit_wait
            and observation.get("status") in _WAITING_STATUSES
            and cluster is None
            and start_at is None
        ):
            return observation
        raise WorkerStartValidationError(f"{label} has no realized worker")
    if observation.get("status") not in _REALIZED_STATUSES:
        raise WorkerStartValidationError(f"{label} status is not worker-realized")
    return observation


def _transport(
    *,
    latch: Mapping[str, object],
    key: object,
    file_sha256: object,
    version_id: object,
    etag: object,
    last_modified: datetime | str,
    label: str,
) -> tuple[str, datetime]:
    if key != worker_start_latch_v2_s3_key(worker_latch=latch):
        raise WorkerStartValidationError(f"{label} key is invalid")
    if file_sha256 != _file_sha(latch, newline=False):
        raise WorkerStartValidationError(f"{label} file SHA-256 is invalid")
    if not isinstance(version_id, str) or _VERSION_ID.fullmatch(version_id) is None:
        raise WorkerStartValidationError(f"{label} version ID is invalid")
    if not isinstance(etag, str) or _ETAG.fullmatch(etag) is None:
        raise WorkerStartValidationError(f"{label} ETag is invalid")
    canonical, modified = _input_time(last_modified, label=f"{label} LastModified")
    if isinstance(last_modified, str) and last_modified != canonical:
        raise WorkerStartValidationError(f"{label} LastModified is not canonical")
    if modified < _stored_time(
        latch.get("entrypoint_observed_at"),
        label=f"{label} entrypoint_observed_at",
    ):
        raise WorkerStartValidationError(f"{label} predates latch entrypoint")
    return canonical, modified


def _required_tags(run_id: str) -> dict[str, str]:
    return {
        "project": "keep-glm52",
        "owner": "jack.mazac",
        "model": "glm-5.2",
        "campaign-run-id": run_id,
        "cost-allocation": "glm52-sky-campaign",
    }


def _active_ids(value: object, *, instance_id: str) -> list[str]:
    if type(value) is not list or any(
        not isinstance(item, str) or _INSTANCE_ID.fullmatch(item) is None
        for item in value
    ):
        raise WorkerStartValidationError("active campaign P5 IDs are invalid")
    result = list(value)
    if (
        result != sorted(result)
        or len(result) != len(set(result))
        or result != [instance_id]
    ):
        raise WorkerStartValidationError(
            "active campaign P5 IDs do not prove one exact worker"
        )
    return result


def _worker_snapshot(
    value: object,
    *,
    descriptor: Mapping[str, object],
    intent: Mapping[str, object],
    latch: Mapping[str, object],
    observation: Mapping[str, object],
    active_ids: object,
    accepted_at: datetime,
) -> dict[str, object]:
    snapshot = _mapping(value, label="worker EC2 mapping")
    if set(snapshot) != _WORKER_INSTANCE_FIELDS:
        raise WorkerStartValidationError("worker EC2 mapping schema mismatch")
    cluster = observation["worker_cluster_name"]
    expected = {
        "instance_id": latch["instance_id"],
        "worker_instance_type": descriptor["instance_type"],
        "worker_image_id": descriptor["image_id"],
        "worker_instance_lifecycle": None,
    }
    if any(snapshot.get(name) != item for name, item in expected.items()):
        raise WorkerStartValidationError("worker EC2 mapping is foreign")
    cluster_fields = (
        "worker_cluster_name",
        "worker_ray_cluster_name",
        "worker_skypilot_cluster_name",
    )
    if cluster is None:
        cluster_values = [snapshot.get(field) for field in cluster_fields]
        if (
            any(
                not isinstance(item, str) or _SAFE_NAME.fullmatch(item) is None
                for item in cluster_values
            )
            or len(set(cluster_values)) != 1
        ):
            raise WorkerStartValidationError(
                "worker EC2 cluster evidence is malformed or inconsistent"
            )
    elif any(snapshot.get(field) != cluster for field in cluster_fields):
        raise WorkerStartValidationError("worker EC2 cluster mapping is foreign")
    if snapshot.get("worker_instance_state") not in _WORKER_STATES:
        raise WorkerStartValidationError("worker EC2 state is invalid")
    tags = _mapping(snapshot.get("worker_campaign_tags"), label="worker tags")
    if any(
        tags.get(name) != item
        for name, item in _required_tags(str(intent["run_id"])).items()
    ):
        raise WorkerStartValidationError("worker campaign tags are incomplete")
    launch = _stored_time(
        snapshot.get("worker_instance_launch_time"),
        label="worker instance launch time",
    )
    observed = _stored_time(
        snapshot.get("ec2_observed_at"),
        label="EC2 observed_at",
    )
    pending = _stored_time(
        latch.get("ec2_pending_time"),
        label="worker latch ec2_pending_time",
    )
    if (
        launch > pending
        or pending > observed
        or observed > accepted_at
        or accepted_at - observed > _MAX_EC2_EVIDENCE_AGE
    ):
        raise WorkerStartValidationError("worker EC2 timing evidence is invalid")
    _active_ids(active_ids, instance_id=str(latch["instance_id"]))
    return {
        **snapshot,
        "worker_campaign_tags": tags,
        "active_campaign_p5_instance_ids": list(active_ids),  # type: ignore[arg-type]
    }


def _acceptance_record(value: object, *, label: str) -> dict[str, object]:
    return _exact_self_hashed(
        value,
        fields=_ACCEPTANCE_FIELDS,
        digest_field="worker_acceptance_body_sha256",
        schema_version=2,
        record_type="glm52_sky_worker_start_accepted_v2",
        label=label,
    )


def worker_start_accepted_v2_s3_key(
    *,
    worker_start_accepted: object,
) -> str:
    """Derive the deterministic marker-last acceptance key."""

    accepted = _acceptance_record(
        worker_start_accepted,
        label="worker start accepted",
    )
    run_id = _run_id(accepted.get("run_id"))
    instance_id = accepted.get("instance_id")
    if (
        accepted.get("managed_mode") != QUALIFICATION_MODE
        or not isinstance(instance_id, str)
        or _INSTANCE_ID.fullmatch(instance_id) is None
    ):
        raise WorkerStartValidationError("worker acceptance key identity is invalid")
    _sha(
        accepted.get("intent_body_sha256"),
        label="worker acceptance intent body SHA-256",
    )
    _sha(
        accepted.get("worker_latch_body_sha256"),
        label="worker latch body SHA-256",
    )
    return (
        f"campaigns/{run_id}/monitor/must-start/qualification/"
        f"{accepted['intent_body_sha256']}/worker-acceptances/{instance_id}/"
        f"{accepted['worker_latch_body_sha256']}/WORKER_START_ACCEPTED.json"
    )


def _validate_prior_snapshot(
    accepted: Mapping[str, object],
    *,
    latch: Mapping[str, object],
    observation: Mapping[str, object],
    latch_evidence: Mapping[str, object],
    observation_evidence: Mapping[str, object],
    static: Mapping[str, object],
    descriptor: Mapping[str, object],
    intent: Mapping[str, object],
    deadline: datetime,
    expected_kind: str,
    prior_accepted: Mapping[str, object] | None,
    prior_observation: Mapping[str, object] | None,
) -> datetime:
    if accepted.get("acceptance_kind") != expected_kind:
        raise WorkerStartValidationError("prior acceptance kind is noncontiguous")
    fixed_fields = (
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "intent_key",
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_key",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "submission_acquisition_key",
        "submission_acquisition_file_sha256",
        "submission_acquisition_body_sha256",
        "submission_accepted_key",
        "submission_accepted_file_sha256",
        "submission_accepted_body_sha256",
        "job_binding_key",
        "job_binding_file_sha256",
        "job_binding_body_sha256",
        "sky_job_id",
        "sky_job_name",
    )
    if any(accepted.get(field) != static.get(field) for field in fixed_fields):
        raise WorkerStartValidationError("prior acceptance authority drifted")
    if accepted.get("instance_id") != latch.get("instance_id"):
        raise WorkerStartValidationError("prior acceptance latch instance drifted")
    latch_fields = (
        "worker_latch_key",
        "worker_latch_file_sha256",
        "worker_latch_version_id",
        "worker_latch_etag",
        "worker_latch_last_modified",
    )
    if any(
        accepted.get(field) != latch_evidence.get(field) for field in latch_fields
    ) or accepted.get("worker_latch_body_sha256") != latch.get(
        "worker_latch_body_sha256"
    ):
        raise WorkerStartValidationError("prior acceptance latch pointer drifted")
    observation_fields = (
        "worker_controller_observation_key",
        "worker_controller_observation_file_sha256",
        "worker_controller_observation_body_sha256",
    )
    if any(
        accepted.get(field) != observation_evidence.get(field)
        for field in observation_fields
    ):
        raise WorkerStartValidationError(
            "prior acceptance worker observation pointer drifted"
        )
    if accepted.get("worker_cluster_name") != observation.get(
        "worker_cluster_name"
    ) or accepted.get("recovery_count") != observation.get("recovery_count"):
        raise WorkerStartValidationError("prior acceptance worker observation drifted")
    accepted_at = _stored_time(
        accepted.get("accepted_at"),
        label="prior acceptance accepted_at",
    )
    last_modified = _stored_time(
        accepted.get("worker_latch_last_modified"),
        label="prior latch LastModified",
    )
    observed = _stored_time(
        accepted.get("ec2_observed_at"),
        label="prior EC2 observed_at",
    )
    launch = _stored_time(
        accepted.get("worker_instance_launch_time"),
        label="prior worker launch time",
    )
    worker_observed = _stored_time(
        observation.get("observed_at"),
        label="prior worker observation observed_at",
    )
    pending = _stored_time(
        latch.get("ec2_pending_time"),
        label="prior latch ec2_pending_time",
    )
    if (
        accepted_at != max(last_modified, observed, worker_observed)
        or not launch <= pending <= observed
        or accepted_at - observed > _MAX_EC2_EVIDENCE_AGE
    ):
        raise WorkerStartValidationError("prior acceptance timing is invalid")
    if (
        accepted.get("worker_instance_type") != descriptor["instance_type"]
        or accepted.get("worker_image_id") != descriptor["image_id"]
        or accepted.get("worker_instance_lifecycle") is not None
        or accepted.get("worker_instance_state") not in _WORKER_STATES
        or accepted.get("worker_ray_cluster_name")
        != observation.get("worker_cluster_name")
        or accepted.get("worker_skypilot_cluster_name")
        != observation.get("worker_cluster_name")
    ):
        raise WorkerStartValidationError("prior acceptance EC2 identity is invalid")
    tags = _mapping(
        accepted.get("worker_campaign_tags"),
        label="prior worker campaign tags",
    )
    if any(
        tags.get(name) != item
        for name, item in _required_tags(str(intent["run_id"])).items()
    ):
        raise WorkerStartValidationError("prior worker campaign tags are invalid")
    _active_ids(
        accepted.get("active_campaign_p5_instance_ids"),
        instance_id=str(latch["instance_id"]),
    )
    entrypoint = _stored_time(
        latch.get("entrypoint_observed_at"),
        label="prior latch entrypoint_observed_at",
    )
    if expected_kind == "initial":
        if (
            pending > deadline
            or entrypoint > deadline
            or last_modified > deadline
            or _stored_time(
                observation.get("start_at"),
                label="prior initial start_at",
            )
            > deadline
        ):
            raise WorkerStartValidationError(
                "prior initial acceptance missed must_start_by"
            )
        if any(
            accepted.get(field) is not None
            for field in (
                "prior_worker_acceptance_key",
                "prior_worker_acceptance_file_sha256",
                "prior_worker_acceptance_body_sha256",
            )
        ):
            raise WorkerStartValidationError(
                "prior initial acceptance contains a predecessor"
            )
    else:
        if prior_accepted is None or prior_observation is None:
            raise WorkerStartValidationError("prior recovery chain is truncated")
        predecessor_accepted_at = _stored_time(
            prior_accepted.get("accepted_at"),
            label="predecessor worker acceptance accepted_at",
        )
        if accepted_at <= predecessor_accepted_at:
            raise WorkerStartValidationError(
                "prior recovery acceptance time did not advance"
            )
        expected_prior = {
            "prior_worker_acceptance_key": worker_start_accepted_v2_s3_key(
                worker_start_accepted=prior_accepted
            ),
            "prior_worker_acceptance_file_sha256": _file_sha(
                prior_accepted,
                newline=False,
            ),
            "prior_worker_acceptance_body_sha256": prior_accepted[
                "worker_acceptance_body_sha256"
            ],
        }
        if any(accepted.get(name) != item for name, item in expected_prior.items()):
            raise WorkerStartValidationError(
                "prior recovery predecessor pointer is broken"
            )
        if accepted.get("instance_id") == prior_accepted.get(
            "instance_id"
        ) or _nonnegative_int(
            accepted.get("recovery_count"),
            label="prior recovery_count",
        ) <= _nonnegative_int(
            prior_accepted.get("recovery_count"),
            label="predecessor recovery_count",
        ):
            raise WorkerStartValidationError(
                "prior recovery instance or count did not progress"
            )
        _status_progressed(
            prior_observation.get("status"),
            observation.get("status"),
            label="prior recovery observation",
        )
        if worker_observed <= _stored_time(
            prior_observation.get("observed_at"),
            label="predecessor worker observation observed_at",
        ):
            raise WorkerStartValidationError(
                "prior recovery observation did not advance in time"
            )
    return accepted_at


def _prior_chain(
    *,
    prior_worker_acceptance_chain: object,
    prior_worker_latch_chain: object,
    prior_worker_controller_observation_chain: object,
    descriptor: Mapping[str, object],
    intent: Mapping[str, object],
    submission_observation: Mapping[str, object],
    binding: Mapping[str, object],
    static: Mapping[str, object],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[datetime],
]:
    if (
        type(prior_worker_acceptance_chain) is not list
        or type(prior_worker_latch_chain) is not list
        or type(prior_worker_controller_observation_chain) is not list
    ):
        raise WorkerStartValidationError("prior worker chains must be exact lists")
    acceptances = [
        _acceptance_record(item, label="prior worker acceptance")
        for item in prior_worker_acceptance_chain
    ]
    latch_evidence_values = [
        _mapping(item, label="prior worker latch evidence")
        for item in prior_worker_latch_chain
    ]
    observation_evidence_values = [
        _mapping(item, label="prior worker observation evidence")
        for item in prior_worker_controller_observation_chain
    ]
    if not (
        len(acceptances)
        == len(latch_evidence_values)
        == len(observation_evidence_values)
    ):
        raise WorkerStartValidationError("prior worker chains are truncated")
    deadline = _stored_time(intent.get("must_start_by"), label="must_start_by")
    instances: set[str] = set()
    observations: list[dict[str, object]] = []
    latches: list[dict[str, object]] = []
    accepted_times: list[datetime] = []
    for index, accepted in enumerate(acceptances):
        latch_evidence = latch_evidence_values[index]
        observation_evidence = observation_evidence_values[index]
        if set(latch_evidence) != _LATCH_EVIDENCE_FIELDS:
            raise WorkerStartValidationError("prior latch evidence schema mismatch")
        if set(observation_evidence) != _OBSERVATION_EVIDENCE_FIELDS:
            raise WorkerStartValidationError(
                "prior worker observation evidence schema mismatch"
            )
        latch = validate_worker_start_latch_v2(
            latch_evidence["worker_latch"],
            descriptor=descriptor,
            intent=intent,
        )
        if latch.get("controller_injected_sky_job_id") != binding.get("sky_job_id"):
            raise WorkerStartValidationError("prior latch names a foreign Sky job")
        last_modified_iso, _ = _transport(
            latch=latch,
            key=latch_evidence["worker_latch_key"],
            file_sha256=latch_evidence["worker_latch_file_sha256"],
            version_id=latch_evidence["worker_latch_version_id"],
            etag=latch_evidence["worker_latch_etag"],
            last_modified=latch_evidence["worker_latch_last_modified"],  # type: ignore[arg-type]
            label="prior worker latch",
        )
        latch_evidence["worker_latch_last_modified"] = last_modified_iso
        observation = _worker_observation(
            observation_evidence["worker_controller_observation"],
            key=observation_evidence["worker_controller_observation_key"],
            file_sha256=observation_evidence[
                "worker_controller_observation_file_sha256"
            ],
            descriptor=descriptor,
            intent=intent,
            submission_observation=submission_observation,
            binding=binding,
            instance_id=str(latch["instance_id"]),
            label="prior worker controller observation",
            permit_wait=False,
        )
        if (
            observation_evidence["worker_controller_observation_body_sha256"]
            != observation["observation_body_sha256"]
        ):
            raise WorkerStartValidationError(
                "prior worker observation body identity drifted"
            )
        instance_id = str(latch["instance_id"])
        if instance_id in instances:
            raise WorkerStartValidationError("prior worker chain repeats an instance")
        instances.add(instance_id)
        accepted_times.append(
            _validate_prior_snapshot(
                accepted,
                latch=latch,
                observation=observation,
                latch_evidence=latch_evidence,
                observation_evidence=observation_evidence,
                static=static,
                descriptor=descriptor,
                intent=intent,
                deadline=deadline,
                expected_kind="initial" if index == 0 else "managed-recovery",
                prior_accepted=acceptances[index - 1] if index else None,
                prior_observation=observations[index - 1] if index else None,
            )
        )
        latches.append(latch)
        observations.append(observation)
    return latches, observations, accepted_times


def _candidate(
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
    controller_baseline_key: object,
    controller_baseline_file_sha256: object,
    submission_acquisition: object,
    submission_acquisition_key: object,
    submission_acquisition_file_sha256: object,
    submission_accepted: object,
    submission_accepted_key: object,
    submission_accepted_file_sha256: object,
    job_binding: object,
    job_binding_key: object,
    job_binding_file_sha256: object,
    submission_controller_observation: object,
    submission_controller_observation_key: object,
    submission_controller_observation_file_sha256: object,
    worker_controller_observation: object,
    worker_controller_observation_key: object,
    worker_controller_observation_file_sha256: object,
    worker_latch: object,
    worker_latch_key: object,
    worker_latch_file_sha256: object,
    worker_latch_version_id: object,
    worker_latch_etag: object,
    worker_latch_last_modified: datetime | str,
    worker_instance: object,
    active_campaign_p5_instance_ids: object,
    prior_worker_acceptance_chain: object,
    prior_worker_latch_chain: object,
    prior_worker_controller_observation_chain: object,
    accepted_at: datetime | str,
    now: datetime | str,
) -> dict[str, object]:
    (
        descriptor_value,
        intent_value,
        baseline,
        acquisition,
        accepted,
        binding,
        submission_observation,
        pointers,
    ) = _submission_authority(
        descriptor=descriptor,
        intent=intent,
        controller_baseline=controller_baseline,
        controller_baseline_key=controller_baseline_key,
        controller_baseline_file_sha256=controller_baseline_file_sha256,
        submission_acquisition=submission_acquisition,
        submission_acquisition_key=submission_acquisition_key,
        submission_acquisition_file_sha256=(submission_acquisition_file_sha256),
        submission_accepted=submission_accepted,
        submission_accepted_key=submission_accepted_key,
        submission_accepted_file_sha256=submission_accepted_file_sha256,
        job_binding=job_binding,
        job_binding_key=job_binding_key,
        job_binding_file_sha256=job_binding_file_sha256,
        submission_controller_observation=submission_controller_observation,
        submission_controller_observation_key=(submission_controller_observation_key),
        submission_controller_observation_file_sha256=(
            submission_controller_observation_file_sha256
        ),
    )
    latch = validate_worker_start_latch_v2(
        worker_latch,
        descriptor=descriptor_value,
        intent=intent_value,
    )
    if latch.get("controller_injected_sky_job_id") != binding.get("sky_job_id"):
        raise WorkerStartValidationError(
            "worker latch and dynamic binding job IDs disagree"
        )
    last_modified_iso, last_modified = _transport(
        latch=latch,
        key=worker_latch_key,
        file_sha256=worker_latch_file_sha256,
        version_id=worker_latch_version_id,
        etag=worker_latch_etag,
        last_modified=worker_latch_last_modified,
        label="worker latch",
    )
    current_observation = _worker_observation(
        worker_controller_observation,
        key=worker_controller_observation_key,
        file_sha256=worker_controller_observation_file_sha256,
        descriptor=descriptor_value,
        intent=intent_value,
        submission_observation=submission_observation,
        binding=binding,
        instance_id=str(latch["instance_id"]),
        label="worker controller observation",
        permit_wait=True,
    )
    accepted_iso, accepted_time = _input_time(accepted_at, label="accepted_at")
    now_iso, now_time = _input_time(now, label="now")
    del now_iso
    if accepted_time > now_time:
        raise WorkerStartValidationError("worker acceptance is in the future")
    snapshot = _worker_snapshot(
        worker_instance,
        descriptor=descriptor_value,
        intent=intent_value,
        latch=latch,
        observation=current_observation,
        active_ids=active_campaign_p5_instance_ids,
        accepted_at=accepted_time,
    )
    deterministic_accepted_at = max(
        last_modified,
        _stored_time(
            current_observation.get("observed_at"),
            label="worker controller observed_at",
        ),
        _stored_time(
            snapshot.get("ec2_observed_at"),
            label="EC2 observed_at",
        ),
    )
    if accepted_time != deterministic_accepted_at:
        raise WorkerStartValidationError(
            "worker accepted_at is not the deterministic evidence maximum"
        )
    static: dict[str, object] = {
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "run_id": intent_value["run_id"],
        "managed_mode": QUALIFICATION_MODE,
        "intent_key": pointers["intent_key"],
        "intent_file_sha256": _file_sha(intent_value, newline=True),
        "intent_body_sha256": intent_value["intent_body_sha256"],
        "controller_baseline_key": pointers["controller_baseline_key"],
        "controller_baseline_file_sha256": pointers["controller_baseline_file_sha256"],
        "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
        "submission_acquisition_key": pointers["submission_acquisition_key"],
        "submission_acquisition_file_sha256": pointers[
            "submission_acquisition_file_sha256"
        ],
        "submission_acquisition_body_sha256": acquisition["acquisition_body_sha256"],
        "submission_accepted_key": pointers["submission_accepted_key"],
        "submission_accepted_file_sha256": pointers["submission_accepted_file_sha256"],
        "submission_accepted_body_sha256": accepted["accepted_body_sha256"],
        "job_binding_key": pointers["job_binding_key"],
        "job_binding_file_sha256": pointers["job_binding_file_sha256"],
        "job_binding_body_sha256": binding["job_binding_body_sha256"],
        "sky_job_id": binding["sky_job_id"],
        "sky_job_name": intent_value["sky_job_name"],
    }
    prior_latches, prior_observations, prior_accepted_times = _prior_chain(
        prior_worker_acceptance_chain=prior_worker_acceptance_chain,
        prior_worker_latch_chain=prior_worker_latch_chain,
        prior_worker_controller_observation_chain=(
            prior_worker_controller_observation_chain
        ),
        descriptor=descriptor_value,
        intent=intent_value,
        submission_observation=submission_observation,
        binding=binding,
        static=static,
    )
    prior_acceptances = list(prior_worker_acceptance_chain)  # type: ignore[arg-type]
    if any(prior_time > now_time for prior_time in prior_accepted_times):
        raise WorkerStartValidationError(
            "prior worker acceptance is later than current time"
        )
    worker_realized = current_observation.get("worker_cluster_name") is not None
    deadline = _stored_time(intent_value.get("must_start_by"), label="must_start_by")
    if not prior_acceptances:
        if prior_latches or prior_observations or prior_accepted_times:
            raise WorkerStartValidationError("initial prior chains are not empty")
        if (
            _stored_time(
                latch.get("ec2_pending_time"),
                label="ec2_pending_time",
            )
            > deadline
            or _stored_time(
                latch.get("entrypoint_observed_at"),
                label="entrypoint_observed_at",
            )
            > deadline
            or last_modified > deadline
            or (
                worker_realized
                and _stored_time(
                    current_observation.get("start_at"),
                    label="worker start_at",
                )
                > deadline
            )
        ):
            raise WorkerStartValidationError("initial worker missed must_start_by")
        acceptance_kind = "initial"
        prior_pointer = {
            "prior_worker_acceptance_key": None,
            "prior_worker_acceptance_file_sha256": None,
            "prior_worker_acceptance_body_sha256": None,
        }
    else:
        acceptance_kind = "managed-recovery"
        prior = prior_acceptances[-1]
        prior_latch = prior_latches[-1]
        prior_observation = prior_observations[-1]
        if not prior_accepted_times or prior_accepted_times[-1] >= accepted_time:
            raise WorkerStartValidationError(
                "managed recovery acceptance time did not advance"
            )
        if latch["instance_id"] in {
            str(item["instance_id"]) for item in prior_latches
        } or _nonnegative_int(
            current_observation.get("recovery_count"),
            label="current recovery_count",
        ) <= _nonnegative_int(
            prior.get("recovery_count"),
            label="prior recovery_count",
        ):
            raise WorkerStartValidationError(
                "managed recovery instance or count did not progress"
            )
        _status_progressed(
            prior_observation.get("status"),
            current_observation.get("status"),
            label="managed recovery observation",
        )
        if _stored_time(
            current_observation.get("observed_at"),
            label="managed recovery observed_at",
        ) <= _stored_time(
            prior_observation.get("observed_at"),
            label="prior worker observation observed_at",
        ):
            raise WorkerStartValidationError(
                "managed recovery observation did not advance in time"
            )
        if prior_latch.get("controller_injected_sky_job_id") != latch.get(
            "controller_injected_sky_job_id"
        ):
            raise WorkerStartValidationError("managed recovery changed Sky job ID")
        prior_pointer = {
            "prior_worker_acceptance_key": worker_start_accepted_v2_s3_key(
                worker_start_accepted=prior
            ),
            "prior_worker_acceptance_file_sha256": _file_sha(
                prior,
                newline=False,
            ),
            "prior_worker_acceptance_body_sha256": prior[
                "worker_acceptance_body_sha256"
            ],
        }
    if not worker_realized:
        raise _WaitForAuthority
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_worker_start_accepted_v2",
        **static,
        "worker_controller_observation_key": worker_controller_observation_key,
        "worker_controller_observation_file_sha256": (
            worker_controller_observation_file_sha256
        ),
        "worker_controller_observation_body_sha256": current_observation[
            "observation_body_sha256"
        ],
        "worker_cluster_name": current_observation["worker_cluster_name"],
        "recovery_count": current_observation["recovery_count"],
        "worker_instance_type": snapshot["worker_instance_type"],
        "worker_image_id": snapshot["worker_image_id"],
        "worker_instance_lifecycle": snapshot["worker_instance_lifecycle"],
        "worker_instance_state": snapshot["worker_instance_state"],
        "worker_instance_launch_time": snapshot["worker_instance_launch_time"],
        "worker_ray_cluster_name": snapshot["worker_ray_cluster_name"],
        "worker_skypilot_cluster_name": snapshot["worker_skypilot_cluster_name"],
        "worker_campaign_tags": snapshot["worker_campaign_tags"],
        "active_campaign_p5_instance_ids": snapshot["active_campaign_p5_instance_ids"],
        "ec2_observed_at": snapshot["ec2_observed_at"],
        "worker_latch_key": worker_latch_key,
        "worker_latch_file_sha256": worker_latch_file_sha256,
        "worker_latch_body_sha256": latch["worker_latch_body_sha256"],
        "worker_latch_version_id": worker_latch_version_id,
        "worker_latch_etag": worker_latch_etag,
        "worker_latch_last_modified": last_modified_iso,
        "instance_id": latch["instance_id"],
        "acceptance_kind": acceptance_kind,
        **prior_pointer,
        "accepted_at": accepted_iso,
    }
    return {**body, "worker_acceptance_body_sha256": _canonical_sha(body)}


def decide_worker_start_acceptance_v2(
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
    controller_baseline_key: object,
    controller_baseline_file_sha256: object,
    submission_acquisition: object,
    submission_acquisition_key: object,
    submission_acquisition_file_sha256: object,
    submission_accepted: object,
    submission_accepted_key: object,
    submission_accepted_file_sha256: object,
    job_binding: object,
    job_binding_key: object,
    job_binding_file_sha256: object,
    submission_controller_observation: object,
    submission_controller_observation_key: object,
    submission_controller_observation_file_sha256: object,
    worker_controller_observation: object,
    worker_controller_observation_key: object,
    worker_controller_observation_file_sha256: object,
    worker_latch: object,
    worker_latch_key: object,
    worker_latch_file_sha256: object,
    worker_latch_version_id: object,
    worker_latch_etag: object,
    worker_latch_last_modified: datetime | str,
    worker_instance: object,
    active_campaign_p5_instance_ids: object,
    prior_worker_acceptance_chain: object,
    prior_worker_latch_chain: object,
    prior_worker_controller_observation_chain: object,
    accepted_at: datetime | str,
    now: datetime | str,
    existing_worker_acceptance: object | None,
) -> WorkerStartDecision:
    """Classify one fully supplied worker-start authority snapshot."""

    try:
        existing: dict[str, object] | None = None
        if existing_worker_acceptance is not None:
            existing = _acceptance_record(
                existing_worker_acceptance,
                label="existing worker acceptance",
            )
        candidate = _candidate(
            descriptor=descriptor,
            intent=intent,
            controller_baseline=controller_baseline,
            controller_baseline_key=controller_baseline_key,
            controller_baseline_file_sha256=controller_baseline_file_sha256,
            submission_acquisition=submission_acquisition,
            submission_acquisition_key=submission_acquisition_key,
            submission_acquisition_file_sha256=(submission_acquisition_file_sha256),
            submission_accepted=submission_accepted,
            submission_accepted_key=submission_accepted_key,
            submission_accepted_file_sha256=submission_accepted_file_sha256,
            job_binding=job_binding,
            job_binding_key=job_binding_key,
            job_binding_file_sha256=job_binding_file_sha256,
            submission_controller_observation=submission_controller_observation,
            submission_controller_observation_key=(
                submission_controller_observation_key
            ),
            submission_controller_observation_file_sha256=(
                submission_controller_observation_file_sha256
            ),
            worker_controller_observation=worker_controller_observation,
            worker_controller_observation_key=worker_controller_observation_key,
            worker_controller_observation_file_sha256=(
                worker_controller_observation_file_sha256
            ),
            worker_latch=worker_latch,
            worker_latch_key=worker_latch_key,
            worker_latch_file_sha256=worker_latch_file_sha256,
            worker_latch_version_id=worker_latch_version_id,
            worker_latch_etag=worker_latch_etag,
            worker_latch_last_modified=worker_latch_last_modified,
            worker_instance=worker_instance,
            active_campaign_p5_instance_ids=active_campaign_p5_instance_ids,
            prior_worker_acceptance_chain=prior_worker_acceptance_chain,
            prior_worker_latch_chain=prior_worker_latch_chain,
            prior_worker_controller_observation_chain=(
                prior_worker_controller_observation_chain
            ),
            accepted_at=accepted_at,
            now=now,
        )
        if existing is not None:
            if existing == candidate:
                return WorkerStartDecision(
                    "idempotent-complete",
                    "the exact deterministic worker acceptance already exists",
                )
            return WorkerStartDecision(
                "fail-closed",
                "a different worker acceptance occupies the deterministic key",
            )
        action: WorkerStartAction = (
            "accept-initial"
            if candidate["acceptance_kind"] == "initial"
            else "accept-managed-recovery"
        )
        return WorkerStartDecision(
            action,
            "all immutable controller, worker, EC2, and ancestry evidence agrees",
        )
    except _WaitForAuthority:
        if existing is not None:
            return WorkerStartDecision(
                "fail-closed",
                "an existing worker acceptance cannot be authenticated "
                "without a realized current candidate",
            )
        return WorkerStartDecision(
            "wait-for-authority",
            "the authenticated controller has not realized a worker cluster",
        )
    except (
        WorkerStartValidationError,
        DynamicJobBindingValidationError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        return WorkerStartDecision(
            "fail-closed",
            "worker-start authority is invalid, foreign, ambiguous, or incomplete",
        )


def build_worker_start_accepted_v2(
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
    controller_baseline_key: object,
    controller_baseline_file_sha256: object,
    submission_acquisition: object,
    submission_acquisition_key: object,
    submission_acquisition_file_sha256: object,
    submission_accepted: object,
    submission_accepted_key: object,
    submission_accepted_file_sha256: object,
    job_binding: object,
    job_binding_key: object,
    job_binding_file_sha256: object,
    submission_controller_observation: object,
    submission_controller_observation_key: object,
    submission_controller_observation_file_sha256: object,
    worker_controller_observation: object,
    worker_controller_observation_key: object,
    worker_controller_observation_file_sha256: object,
    worker_latch: object,
    worker_latch_key: object,
    worker_latch_file_sha256: object,
    worker_latch_version_id: object,
    worker_latch_etag: object,
    worker_latch_last_modified: datetime | str,
    worker_instance: object,
    active_campaign_p5_instance_ids: object,
    prior_worker_acceptance_chain: object,
    prior_worker_latch_chain: object,
    prior_worker_controller_observation_chain: object,
    accepted_at: datetime | str,
    now: datetime | str,
) -> dict[str, object]:
    """Build one initial or managed-recovery worker acceptance."""

    try:
        candidate = _candidate(
            descriptor=descriptor,
            intent=intent,
            controller_baseline=controller_baseline,
            controller_baseline_key=controller_baseline_key,
            controller_baseline_file_sha256=controller_baseline_file_sha256,
            submission_acquisition=submission_acquisition,
            submission_acquisition_key=submission_acquisition_key,
            submission_acquisition_file_sha256=(submission_acquisition_file_sha256),
            submission_accepted=submission_accepted,
            submission_accepted_key=submission_accepted_key,
            submission_accepted_file_sha256=submission_accepted_file_sha256,
            job_binding=job_binding,
            job_binding_key=job_binding_key,
            job_binding_file_sha256=job_binding_file_sha256,
            submission_controller_observation=submission_controller_observation,
            submission_controller_observation_key=(
                submission_controller_observation_key
            ),
            submission_controller_observation_file_sha256=(
                submission_controller_observation_file_sha256
            ),
            worker_controller_observation=worker_controller_observation,
            worker_controller_observation_key=worker_controller_observation_key,
            worker_controller_observation_file_sha256=(
                worker_controller_observation_file_sha256
            ),
            worker_latch=worker_latch,
            worker_latch_key=worker_latch_key,
            worker_latch_file_sha256=worker_latch_file_sha256,
            worker_latch_version_id=worker_latch_version_id,
            worker_latch_etag=worker_latch_etag,
            worker_latch_last_modified=worker_latch_last_modified,
            worker_instance=worker_instance,
            active_campaign_p5_instance_ids=active_campaign_p5_instance_ids,
            prior_worker_acceptance_chain=prior_worker_acceptance_chain,
            prior_worker_latch_chain=prior_worker_latch_chain,
            prior_worker_controller_observation_chain=(
                prior_worker_controller_observation_chain
            ),
            accepted_at=accepted_at,
            now=now,
        )
    except WorkerStartValidationError:
        raise
    except _WaitForAuthority as error:
        raise WorkerStartValidationError(
            "worker controller authority is not yet realized"
        ) from error
    except (
        DynamicJobBindingValidationError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as error:
        raise WorkerStartValidationError(
            "worker-start authority input is malformed"
        ) from error
    return dict(candidate)


def validate_worker_start_accepted_v2(
    value: object,
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
    controller_baseline_key: object,
    controller_baseline_file_sha256: object,
    submission_acquisition: object,
    submission_acquisition_key: object,
    submission_acquisition_file_sha256: object,
    submission_accepted: object,
    submission_accepted_key: object,
    submission_accepted_file_sha256: object,
    job_binding: object,
    job_binding_key: object,
    job_binding_file_sha256: object,
    submission_controller_observation: object,
    submission_controller_observation_key: object,
    submission_controller_observation_file_sha256: object,
    worker_controller_observation: object,
    worker_controller_observation_key: object,
    worker_controller_observation_file_sha256: object,
    worker_latch: object,
    worker_latch_key: object,
    worker_latch_file_sha256: object,
    worker_latch_version_id: object,
    worker_latch_etag: object,
    worker_latch_last_modified: datetime | str,
    worker_instance: object,
    active_campaign_p5_instance_ids: object,
    prior_worker_acceptance_chain: object,
    prior_worker_latch_chain: object,
    prior_worker_controller_observation_chain: object,
    accepted_at: datetime | str,
    now: datetime | str,
) -> dict[str, object]:
    """Authenticate an exact worker acceptance against every supplied record."""

    accepted = _acceptance_record(value, label="worker start accepted")
    candidate = build_worker_start_accepted_v2(
        descriptor=descriptor,
        intent=intent,
        controller_baseline=controller_baseline,
        controller_baseline_key=controller_baseline_key,
        controller_baseline_file_sha256=controller_baseline_file_sha256,
        submission_acquisition=submission_acquisition,
        submission_acquisition_key=submission_acquisition_key,
        submission_acquisition_file_sha256=submission_acquisition_file_sha256,
        submission_accepted=submission_accepted,
        submission_accepted_key=submission_accepted_key,
        submission_accepted_file_sha256=submission_accepted_file_sha256,
        job_binding=job_binding,
        job_binding_key=job_binding_key,
        job_binding_file_sha256=job_binding_file_sha256,
        submission_controller_observation=submission_controller_observation,
        submission_controller_observation_key=(submission_controller_observation_key),
        submission_controller_observation_file_sha256=(
            submission_controller_observation_file_sha256
        ),
        worker_controller_observation=worker_controller_observation,
        worker_controller_observation_key=worker_controller_observation_key,
        worker_controller_observation_file_sha256=(
            worker_controller_observation_file_sha256
        ),
        worker_latch=worker_latch,
        worker_latch_key=worker_latch_key,
        worker_latch_file_sha256=worker_latch_file_sha256,
        worker_latch_version_id=worker_latch_version_id,
        worker_latch_etag=worker_latch_etag,
        worker_latch_last_modified=worker_latch_last_modified,
        worker_instance=worker_instance,
        active_campaign_p5_instance_ids=active_campaign_p5_instance_ids,
        prior_worker_acceptance_chain=prior_worker_acceptance_chain,
        prior_worker_latch_chain=prior_worker_latch_chain,
        prior_worker_controller_observation_chain=(
            prior_worker_controller_observation_chain
        ),
        accepted_at=accepted_at,
        now=now,
    )
    if accepted != candidate:
        raise WorkerStartValidationError("worker acceptance authority drifted")
    return dict(accepted)


__all__ = [
    "WorkerStartDecision",
    "WorkerStartValidationError",
    "build_worker_start_accepted_v2",
    "build_worker_start_latch_v2",
    "decide_worker_start_acceptance_v2",
    "validate_worker_start_accepted_v2",
    "validate_worker_start_latch_v2",
    "worker_start_accepted_v2_s3_key",
    "worker_start_latch_v2_s3_key",
    "worker_v2_canonical_bytes",
]
