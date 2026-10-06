"""Import-light validation for the five fixed activation source records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import re
from typing import Mapping

from .s3_keys import (
    gpu_spend_snapshot_s3_key,
    production_controller_baseline_s3_key,
    production_must_start_control_plane_ready_s3_key,
    production_submission_acquired_s3_key,
    production_submission_intent_s3_key,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]\Z")
_INSTANCE_ID = re.compile(r"i-(?:[0-9a-f]{8}|[0-9a-f]{17})\Z")
_ROLE_ARN = re.compile(
    r"arn:aws:iam::246813579024:role/[A-Za-z0-9+=,.@_/-]+\Z"
)
_SNAPSHOT_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "campaign_identity_sha256",
    "descriptor_sha256",
    "descriptor_body_sha256",
    "approval_sha256",
    "approval_body_sha256",
    "gpu_spend_ledger_latest_sha256",
    "gpu_spend_ledger_latest_body_sha256",
    "gpu_spend_ledger_genesis_sha256",
    "gpu_spend_ledger_record_count",
    "gpu_spend_ledger_tip_record_sha256",
    "gpu_spend_ledger_file_sha256",
    "ec2_allocation_history_sha256",
    "ec2_allocation_instance_ids",
    "observed_at",
    "approved_gpu_runtime_seconds",
    "approved_gpu_cost_usd",
    "hourly_cost_usd",
    "consumed_gpu_seconds",
    "remaining_gpu_seconds",
    "consumed_gpu_cost_usd",
    "remaining_gpu_cost_usd",
    "qualification_allowance_seconds",
    "qualification_allowance_cost_usd",
    "open_allocation_count",
}
_SNAPSHOT_FIELDS = _SNAPSHOT_BODY_FIELDS | {"snapshot_body_sha256"}
_SNAPSHOT_DIGEST_FIELDS = {
    "campaign_identity_sha256",
    "descriptor_sha256",
    "descriptor_body_sha256",
    "approval_sha256",
    "approval_body_sha256",
    "gpu_spend_ledger_latest_sha256",
    "gpu_spend_ledger_latest_body_sha256",
    "gpu_spend_ledger_genesis_sha256",
    "gpu_spend_ledger_tip_record_sha256",
    "gpu_spend_ledger_file_sha256",
    "ec2_allocation_history_sha256",
    "snapshot_body_sha256",
}
_INTENT_FIELDS = {
    "schema_version",
    "record_type",
    "managed_mode",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "campaign_identity_sha256",
    "sky_job_name",
    "must_start_by",
    "intent_at",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "repo_tar_key",
    "repo_tar_sha256",
    "approval_key",
    "approval_file_sha256",
    "approval_body_sha256",
    "approval_version_id",
    "staged_readiness_key",
    "staged_readiness_file_sha256",
    "staged_readiness_body_sha256",
    "staged_readiness_version_id",
    "staged_at",
    "bundle_manifest_key",
    "bundle_manifest_file_sha256",
    "bundle_manifest_body_sha256",
    "bundle_manifest_version_id",
    "rehearsal_evidence_key",
    "rehearsal_evidence_file_sha256",
    "rehearsal_evidence_body_sha256",
    "rehearsal_evidence_version_id",
    "rehearsal_completed_at",
    "cache_seed_acceptance_key",
    "cache_seed_acceptance_file_sha256",
    "cache_seed_acceptance_body_sha256",
    "cache_seed_acceptance_version_id",
    "qualification_cache_prefix",
    "qualification_cache_manifest_sha256",
    "cache_seed_instance_id",
    "cache_seed_accepted_at",
    "h100_resume_ready_key",
    "h100_resume_ready_file_sha256",
    "h100_resume_ready_body_sha256",
    "h100_resume_ready_version_id",
    "h100_first_instance_id",
    "h100_replacement_instance_id",
    "h100_completed_at",
    "first_h100_allocation_key",
    "first_h100_allocation_file_sha256",
    "first_h100_allocation_body_sha256",
    "first_h100_allocation_record_sha256",
    "first_h100_allocation_version_id",
    "replacement_h100_allocation_key",
    "replacement_h100_allocation_file_sha256",
    "replacement_h100_allocation_body_sha256",
    "replacement_h100_allocation_record_sha256",
    "replacement_h100_allocation_version_id",
    "gpu_spend_snapshot_key",
    "gpu_spend_snapshot_file_sha256",
    "gpu_spend_snapshot_body_sha256",
    "gpu_spend_snapshot_version_id",
    "gpu_spend_ledger_tip_record_sha256",
    "gpu_spend_ledger_genesis_sha256",
    "ec2_allocation_history_sha256",
    "closed_instance_ids",
    "open_allocation_count",
    "approved_gpu_runtime_seconds",
    "approved_gpu_cost_usd",
    "hourly_cost_usd",
    "consumed_gpu_seconds",
    "consumed_gpu_cost_usd",
    "remaining_gpu_seconds",
    "remaining_gpu_cost_usd",
    "spend_snapshot_observed_at",
    "intent_body_sha256",
}
_BASELINE_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "sky_job_name",
    "workspace",
    "controller_instance_id",
    "controller_instance_type",
    "controller_identity",
    "controller_profile_arn",
    "controller_cluster_name",
    "ssm_ping_status",
    "exact_name_history",
    "active_exact_name_job_ids",
    "active_tagged_p5_instance_ids",
    "observed_at",
    "baseline_body_sha256",
}
_READY_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "controller_baseline_key",
    "controller_baseline_file_sha256",
    "controller_baseline_body_sha256",
    "controller_baseline_version_id",
    "stack_id",
    "template_sha256",
    "lambda_function_arn",
    "lambda_code_s3_key",
    "lambda_code_sha256",
    "lambda_code_version_id",
    "iam_policy_sha256",
    "reconciliation_rule_arn",
    "reconciliation_rule_state",
    "deadline_schedule_arn",
    "deadline_schedule_state",
    "dlq_arn",
    "coordinator_mode",
    "activation_capabilities",
    "observed_at",
    "control_plane_ready_body_sha256",
}
_ACQUISITION_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "controller_baseline_key",
    "controller_baseline_file_sha256",
    "controller_baseline_body_sha256",
    "controller_baseline_version_id",
    "must_start_control_plane_ready_key",
    "must_start_control_plane_ready_file_sha256",
    "must_start_control_plane_ready_body_sha256",
    "must_start_control_plane_ready_version_id",
    "sky_job_name",
    "must_start_by",
    "spend_snapshot_observed_at",
    "controller_baseline_observed_at",
    "control_plane_ready_observed_at",
    "acquired_at",
    "acquisition_body_sha256",
}
_HISTORY_FIELDS = {
    "sky_job_id",
    "sky_job_name",
    "workspace",
    "controller_submitted_at",
    "controller_status",
    "controller_identity",
}
_CONTROLLER_STATUSES = {
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
_ACTIVE_CONTROLLER_STATUSES = {
    "PENDING",
    "STARTING",
    "RUNNING",
    "RECOVERING",
    "CANCELLING",
}
_DESCRIPTOR_FIELDS = {
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
_ARTIFACT_FIELDS = {
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
_DEPLOYMENT_FIELDS = {
    "stack_id",
    "template_sha256",
    "lambda_function_arn",
    "lambda_code_s3_key",
    "lambda_code_sha256",
    "lambda_code_version_id",
    "iam_policy_sha256",
    "reconciliation_rule_arn",
    "deadline_schedule_arn",
    "dlq_arn",
    "activation_capabilities",
}
_ACTIVATION_FIELDS = {
    "stack_status",
    "stack_operation_in_progress",
    "foundation_revision",
    "managed_mode",
    "run_id",
    "bucket",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "sky_job_name",
    "must_start_by",
    "primary_wake_at",
    "lambda_state",
    "lambda_last_update_status",
    "lambda_qualified_arn",
    "lambda_execution_role_arn",
    "lambda_code_sha256",
    "lambda_code_version_id",
    "lambda_environment",
    "reconciliation_schedule_expression",
    "reconciliation_target_arn",
    "reconciliation_target_id",
    "reconciliation_input",
    "reconciliation_retry_max_event_age_seconds",
    "reconciliation_retry_max_attempts",
    "reconciliation_dlq_arn",
    "deadline_schedule_expression",
    "deadline_schedule_timezone",
    "deadline_flexible_window_mode",
    "deadline_target_arn",
    "deadline_target_role_arn",
    "deadline_input",
    "deadline_retry_max_event_age_seconds",
    "deadline_retry_max_attempts",
    "deadline_dlq_arn",
}
_ENVIRONMENT_FIELDS = {
    "EXPECTED_ACCOUNT_ID",
    "CAMPAIGN_BUCKET",
    "CAMPAIGN_RUN_ID",
    "MANAGED_MODE",
    "CAMPAIGN_DESCRIPTOR_KEY",
    "CAMPAIGN_DESCRIPTOR_VERSION_ID",
    "IMMUTABLE_SUBMISSION_KEY",
    "IMMUTABLE_SUBMISSION_VERSION_ID",
    "SUBMISSION_BODY_SHA256",
    "SKY_JOB_NAME",
    "MUST_START_BY",
    "COORDINATOR_MODE",
}
_TRIGGER_FIELDS = {
    "campaign_run_id",
    "managed_mode",
    "sky_job_name",
    "must_start_by",
    "descriptor_key",
    "descriptor_version_id",
    "intent_key",
    "intent_version_id",
    "trigger",
}


class SourceAuthorityError(ValueError):
    """A fixed activation source record is malformed or cross-bound wrongly."""


@dataclass(frozen=True)
class VersionedJsonArtifact:
    """Exact canonical source bytes and their service-assigned S3 version."""

    key: str
    raw: bytes
    version_id: str


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SourceAuthorityError("value is not exact canonical JSON") from exc


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SourceAuthorityError(f"{field} must be exact lowercase SHA-256")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SourceAuthorityError(f"{field} must be an exact integer")
    return value


def _money(value: object, *, field: str) -> Decimal:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SourceAuthorityError(f"{field} must be finite numeric money")
    result = Decimal(str(value))
    if result < 0 or result != result.quantize(Decimal("0.01")):
        raise SourceAuthorityError(
            f"{field} must be non-negative two-decimal money"
        )
    return result


def _cost(seconds: int, hourly: Decimal) -> Decimal:
    return (Decimal(seconds) * hourly / Decimal(3600)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def _canonical_utc(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise SourceAuthorityError(f"{field} must be ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError) as exc:
        raise SourceAuthorityError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise SourceAuthorityError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_gpu_spend_snapshot(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the accepted self-hashed cumulative-spend snapshot schema."""

    if set(value) != _SNAPSHOT_FIELDS:
        raise SourceAuthorityError("GPU spend snapshot schema mismatch")
    body = dict(value)
    actual_sha256 = body.pop("snapshot_body_sha256")
    if actual_sha256 != _sha(_canonical(body)):
        raise SourceAuthorityError("GPU spend snapshot body SHA-256 mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("record_type") != "glm52_gpu_spend_snapshot_v1"
    ):
        raise SourceAuthorityError("GPU spend snapshot schema mismatch")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise SourceAuthorityError("GPU spend snapshot run_id is invalid")
    for field in _SNAPSHOT_DIGEST_FIELDS:
        _digest(value.get(field), field=field)
    if _canonical_utc(value.get("observed_at"), field="observed_at") != value.get(
        "observed_at"
    ):
        raise SourceAuthorityError(
            "GPU spend snapshot observed_at is not canonical UTC"
        )
    approved_seconds = _integer(
        value.get("approved_gpu_runtime_seconds"),
        field="approved_gpu_runtime_seconds",
        minimum=1,
    )
    consumed_seconds = _integer(
        value.get("consumed_gpu_seconds"), field="consumed_gpu_seconds"
    )
    remaining_seconds = _integer(
        value.get("remaining_gpu_seconds"), field="remaining_gpu_seconds"
    )
    allowance_seconds = _integer(
        value.get("qualification_allowance_seconds"),
        field="qualification_allowance_seconds",
    )
    record_count = _integer(
        value.get("gpu_spend_ledger_record_count"),
        field="gpu_spend_ledger_record_count",
        minimum=2,
    )
    open_count = _integer(
        value.get("open_allocation_count"), field="open_allocation_count"
    )
    if (
        approved_seconds != 86_400
        or consumed_seconds + remaining_seconds != approved_seconds
        or allowance_seconds != min(14_400, remaining_seconds)
        or record_count % 2
        or open_count != 0
    ):
        raise SourceAuthorityError(
            "GPU spend snapshot runtime accounting is inconsistent"
        )
    approved_cost = _money(
        value.get("approved_gpu_cost_usd"), field="approved_gpu_cost_usd"
    )
    hourly_cost = _money(
        value.get("hourly_cost_usd"), field="hourly_cost_usd"
    )
    consumed_cost = _money(
        value.get("consumed_gpu_cost_usd"), field="consumed_gpu_cost_usd"
    )
    remaining_cost = _money(
        value.get("remaining_gpu_cost_usd"), field="remaining_gpu_cost_usd"
    )
    allowance_cost = _money(
        value.get("qualification_allowance_cost_usd"),
        field="qualification_allowance_cost_usd",
    )
    if (
        approved_cost != Decimal("1320.96")
        or hourly_cost != Decimal("55.04")
        or consumed_cost != _cost(consumed_seconds, hourly_cost)
        or consumed_cost + remaining_cost != approved_cost
        or allowance_cost != _cost(allowance_seconds, hourly_cost)
    ):
        raise SourceAuthorityError(
            "GPU spend snapshot dollar accounting is inconsistent"
        )
    instance_ids = value.get("ec2_allocation_instance_ids")
    if (
        not isinstance(instance_ids, list)
        or len(instance_ids) * 2 != record_count
        or len(set(instance_ids)) != len(instance_ids)
        or any(
            not isinstance(instance_id, str)
            or not instance_id.startswith("i-")
            for instance_id in instance_ids
        )
    ):
        raise SourceAuthorityError(
            "GPU spend snapshot EC2 allocation inventory is inconsistent"
        )
    return dict(value)


def _exact_fields(
    value: object, fields: set[str], *, label: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise SourceAuthorityError(f"{label} schema mismatch")
    _canonical(value)
    return value


def _exact_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise SourceAuthorityError(f"{field} must be an exact nonempty string")
    return value


def _run_id(value: object) -> str:
    result = _exact_string(value, field="run_id")
    if _RUN_ID.fullmatch(result) is None:
        raise SourceAuthorityError("run_id is invalid")
    return result


def _bucket(value: object) -> str:
    result = _exact_string(value, field="bucket")
    if _BUCKET.fullmatch(result) is None:
        raise SourceAuthorityError("bucket is invalid")
    return result


def _safe_key(value: object, *, field: str) -> str:
    key = _exact_string(value, field=field)
    if (
        key.startswith("/")
        or key.endswith("/")
        or "\\" in key
        or any(character in key for character in "*?[]")
        or any(
            ord(character) < 0x21 or ord(character) > 0x7E
            for character in key
        )
        or any(
            segment in {"", ".", ".."} for segment in key.split("/")
        )
    ):
        raise SourceAuthorityError(f"{field} is not a safe campaign key")
    return key


def _version_id(value: object, *, field: str) -> str:
    result = _exact_string(value, field=field)
    if (
        result == "null"
        or len(result) > 1024
        or any(
            ord(character) < 0x21 or ord(character) > 0x7E
            for character in result
        )
    ):
        raise SourceAuthorityError(f"{field} is not an opaque VersionId")
    return result


def _time(value: object, *, field: str) -> tuple[str, datetime]:
    text = _canonical_utc(value, field=field)
    if text != value or not text.endswith("Z"):
        raise SourceAuthorityError(f"{field} is not canonical UTC text")
    parsed = datetime.fromisoformat(f"{text[:-1]}+00:00")
    if parsed.microsecond != 0:
        raise SourceAuthorityError(f"{field} must be whole-second UTC text")
    return text, parsed


def _self_hash(
    value: object,
    *,
    fields: set[str],
    digest_field: str,
    label: str,
) -> dict[str, object]:
    record = _exact_fields(value, fields, label=label)
    digest = _digest(record.get(digest_field), field=digest_field)
    body = dict(record)
    del body[digest_field]
    if digest != _sha(_canonical(body)):
        raise SourceAuthorityError(f"{label} body SHA-256 mismatch")
    return record


def _submission_id(descriptor_key: str, run_id: str) -> str:
    prefix = f"campaigns/{run_id}/submissions/"
    suffix = "/campaign-descriptor-v2.json"
    if not descriptor_key.startswith(prefix) or not descriptor_key.endswith(
        suffix
    ):
        raise SourceAuthorityError("descriptor key is not exact")
    result = descriptor_key[len(prefix) : -len(suffix)]
    if (
        not result
        or "/" in result
        or result in {".", ".."}
        or any(character in result for character in "\\*?[]")
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in result)
    ):
        raise SourceAuthorityError("submission_id is not a safe segment")
    return result


def validate_production_submission_intent(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the accepted closed structural production intent."""

    intent = _self_hash(
        value,
        fields=_INTENT_FIELDS,
        digest_field="intent_body_sha256",
        label="production intent",
    )
    if (
        type(intent.get("schema_version")) is not int
        or intent["schema_version"] != 1
        or intent.get("record_type")
        != "glm52_sky_production_submission_intent_v1"
        or intent.get("managed_mode") != "production"
        or intent.get("account_id") != "246813579024"
        or intent.get("region") != "us-west-2"
    ):
        raise SourceAuthorityError("production intent discriminant mismatch")
    run_id = _run_id(intent.get("run_id"))
    _bucket(intent.get("bucket"))
    if intent.get("sky_job_name") != run_id:
        raise SourceAuthorityError("production intent Sky job name mismatch")
    for field in _INTENT_FIELDS:
        if field.endswith("_sha256"):
            _digest(intent.get(field), field=field)
        elif field.endswith("_version_id"):
            _version_id(intent.get(field), field=field)
        elif field.endswith("_key"):
            _safe_key(intent.get(field), field=field)
    submission_id = _submission_id(str(intent["descriptor_key"]), run_id)
    expected_keys = {
        "staged_readiness_key": (
            f"campaigns/{run_id}/submissions/{submission_id}/"
            "STAGED_CONTROL_PLANE_READY.json"
        ),
        "bundle_manifest_key": (
            f"campaigns/{run_id}/submissions/{submission_id}/bundle-manifests/"
            f"{intent['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
        ),
        "rehearsal_evidence_key": (
            f"campaigns/{run_id}/qualification/rehearsals/"
            f"{intent['rehearsal_evidence_body_sha256']}/"
            "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
        "cache_seed_acceptance_key": (
            f"campaigns/{run_id}/qualification-cache-seed/accepted/"
            f"{intent['cache_seed_acceptance_body_sha256']}/"
            "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
        ),
        "h100_resume_ready_key": (
            f"campaigns/{run_id}/qualification/H100_RESUME_READY.json"
        ),
        "first_h100_allocation_key": (
            f"campaigns/{run_id}/runtime/GPU_RUNTIME_ALLOCATION.json"
        ),
        "replacement_h100_allocation_key": (
            f"campaigns/{run_id}/runtime/GPU_RUNTIME_ALLOCATION.json"
        ),
        "gpu_spend_snapshot_key": gpu_spend_snapshot_s3_key(
            run_id=run_id,
            snapshot_body_sha256=str(
                intent["gpu_spend_snapshot_body_sha256"]
            ),
        ),
    }
    if any(intent.get(field) != expected for field, expected in expected_keys.items()):
        raise SourceAuthorityError("production intent embedded key mismatch")
    if not str(intent["repo_tar_key"]).startswith(
        f"campaigns/{run_id}/repository/"
    ) or not str(intent["approval_key"]).startswith(
        f"campaigns/{run_id}/authorities/"
    ):
        raise SourceAuthorityError("production intent source key mismatch")
    instance_fields = (
        "cache_seed_instance_id",
        "h100_first_instance_id",
        "h100_replacement_instance_id",
    )
    instance_ids = [
        _exact_string(intent.get(field), field=field)
        for field in instance_fields
    ]
    closed_ids = intent.get("closed_instance_ids")
    if (
        type(closed_ids) is not list
        or any(
            type(item) is not str or _INSTANCE_ID.fullmatch(item) is None
            for item in closed_ids
        )
        or len(set(closed_ids)) != len(closed_ids)
        or any(_INSTANCE_ID.fullmatch(item) is None for item in instance_ids)
        or len(set(instance_ids)) != 3
        or not set(instance_ids).issubset(set(closed_ids))
    ):
        raise SourceAuthorityError("production intent closed inventory mismatch")
    for first, second in (
        (
            "first_h100_allocation_version_id",
            "replacement_h100_allocation_version_id",
        ),
        (
            "first_h100_allocation_file_sha256",
            "replacement_h100_allocation_file_sha256",
        ),
        (
            "first_h100_allocation_body_sha256",
            "replacement_h100_allocation_body_sha256",
        ),
        (
            "first_h100_allocation_record_sha256",
            "replacement_h100_allocation_record_sha256",
        ),
    ):
        if intent[first] == intent[second]:
            raise SourceAuthorityError(
                "production allocation identities are not distinct"
            )
    if (
        type(intent.get("open_allocation_count")) is not int
        or intent["open_allocation_count"] != 0
    ):
        raise SourceAuthorityError("production intent is not spend-closed")
    approved_seconds = _integer(
        intent.get("approved_gpu_runtime_seconds"),
        field="approved_gpu_runtime_seconds",
        minimum=1,
    )
    consumed_seconds = _integer(
        intent.get("consumed_gpu_seconds"), field="consumed_gpu_seconds"
    )
    remaining_seconds = _integer(
        intent.get("remaining_gpu_seconds"), field="remaining_gpu_seconds"
    )
    if (
        approved_seconds != 86_400
        or consumed_seconds + remaining_seconds != approved_seconds
        or remaining_seconds <= 3_600
    ):
        raise SourceAuthorityError("production intent runtime budget mismatch")
    money = {}
    for field in (
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "consumed_gpu_cost_usd",
        "remaining_gpu_cost_usd",
    ):
        if type(intent.get(field)) is not float:
            raise SourceAuthorityError(f"{field} must be an exact finite float")
        money[field] = _money(intent[field], field=field)
    if (
        money["approved_gpu_cost_usd"] != Decimal("1320.96")
        or money["hourly_cost_usd"] != Decimal("55.04")
        or money["remaining_gpu_cost_usd"] <= Decimal("55.04")
        or money["consumed_gpu_cost_usd"]
        != _cost(consumed_seconds, money["hourly_cost_usd"])
        or money["consumed_gpu_cost_usd"]
        + money["remaining_gpu_cost_usd"]
        != money["approved_gpu_cost_usd"]
    ):
        raise SourceAuthorityError("production intent dollar budget mismatch")
    prefix = _exact_string(
        intent.get("qualification_cache_prefix"),
        field="qualification_cache_prefix",
    )
    if (
        intent["qualification_cache_manifest_sha256"] == "0" * 64
        or prefix
        != (
            f"qualification-cache/seeds/{run_id}/"
            f"{intent['qualification_cache_manifest_sha256']}/"
        )
    ):
        raise SourceAuthorityError("production intent cache identity is empty")
    times = {
        field: _time(intent.get(field), field=field)
        for field in (
            "cache_seed_accepted_at",
            "staged_at",
            "rehearsal_completed_at",
            "h100_completed_at",
            "spend_snapshot_observed_at",
            "intent_at",
            "must_start_by",
        )
    }
    ordered = [times[field][1] for field in (
        "cache_seed_accepted_at",
        "staged_at",
        "rehearsal_completed_at",
        "h100_completed_at",
        "spend_snapshot_observed_at",
        "intent_at",
        "must_start_by",
    )]
    if (
        not all(first <= second for first, second in zip(ordered, ordered[1:]))
        or ordered[-2] >= ordered[-1]
        or ordered[-2] - ordered[-3] > timedelta(seconds=60)
        or ordered[-1] - ordered[-2] > timedelta(hours=12)
    ):
        raise SourceAuthorityError("production intent timestamp chain mismatch")
    return dict(intent)


def _parse_artifact(
    value: object, *, label: str
) -> tuple[VersionedJsonArtifact, dict[str, object]]:
    if type(value) is not VersionedJsonArtifact:
        raise SourceAuthorityError(
            f"{label} must be an exact VersionedJsonArtifact"
        )
    key = _safe_key(value.key, field=f"{label}.key")
    version = _version_id(value.version_id, field=f"{label}.version_id")
    if type(value.raw) is not bytes or not value.raw.endswith(b"\n"):
        raise SourceAuthorityError(f"{label}.raw must be canonical bytes plus LF")
    try:
        parsed = json.loads(value.raw[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourceAuthorityError(f"{label}.raw is invalid JSON") from exc
    if type(parsed) is not dict or _canonical(parsed) + b"\n" != value.raw:
        raise SourceAuthorityError(f"{label}.raw is not canonical JSON plus LF")
    return VersionedJsonArtifact(key, value.raw, version), parsed


def _source_context(
    *,
    descriptor: object,
    intent: object,
) -> tuple[
    VersionedJsonArtifact,
    dict[str, object],
    VersionedJsonArtifact,
    dict[str, object],
]:
    descriptor_artifact, descriptor_record = _parse_artifact(
        descriptor, label="descriptor"
    )
    intent_artifact, intent_record = _parse_artifact(
        intent, label="production intent"
    )
    validated_intent = validate_production_submission_intent(intent_record)
    if set(descriptor_record) != _DESCRIPTOR_FIELDS or (
        descriptor_record.get("schema_version") != 2
        or descriptor_record.get("record_type")
        != "glm52_sky_campaign_descriptor_v2"
        or descriptor_record.get("account_id") != "246813579024"
        or descriptor_record.get("region") != "us-west-2"
        or descriptor_record.get("provider") != "aws"
        or descriptor_record.get("instance_type") != "p5.48xlarge"
        or type(descriptor_record.get("instance_count")) is not int
        or descriptor_record.get("instance_count") != 1
        or type(descriptor_record.get("use_spot")) is not bool
        or descriptor_record.get("use_spot") is not False
        or type(descriptor_record.get("max_hourly_cost_usd")) is not float
        or descriptor_record.get("max_hourly_cost_usd") != 55.04
        or type(descriptor_record.get("approved_gpu_runtime_seconds")) is not int
        or descriptor_record.get("approved_gpu_runtime_seconds") != 86_400
        or type(descriptor_record.get("approved_gpu_cost_usd")) is not float
        or descriptor_record.get("approved_gpu_cost_usd") != 1_320.96
        or descriptor_record.get("skypilot_version") != "0.13.0"
        or descriptor_record.get("task_name") != "glm52-campaign"
    ):
        raise SourceAuthorityError("campaign descriptor fixed contract mismatch")
    body = dict(descriptor_record)
    descriptor_body_sha = body.pop("descriptor_body_sha256", None)
    if descriptor_body_sha != _sha(_canonical(body)):
        raise SourceAuthorityError("campaign descriptor body hash mismatch")
    run_id = _run_id(descriptor_record.get("run_id"))
    identity = {
        field: item
        for field, item in descriptor_record.items()
        if field
        not in {
            "campaign_identity_sha256",
            "descriptor_body_sha256",
            "must_start_by",
            "campaign_descriptor_key",
        }
    }
    if descriptor_record["campaign_identity_sha256"] != _sha(
        _canonical(identity)
    ):
        raise SourceAuthorityError("campaign identity hash mismatch")
    _time(descriptor_record.get("must_start_by"), field="must_start_by")
    for field in ("controller_identity", "worker_identity"):
        if _ROLE_ARN.fullmatch(str(descriptor_record.get(field))) is None:
            raise SourceAuthorityError(f"campaign descriptor {field} is invalid")
    if not re.fullmatch(
        r"ami-[0-9a-f]{8}(?:[0-9a-f]{9})?",
        str(descriptor_record.get("image_id")),
    ):
        raise SourceAuthorityError("campaign descriptor image_id is invalid")
    for field in ("repo_tar_sha256", "approval_sha256"):
        _digest(descriptor_record.get(field), field=field)
    campaign_prefix = f"campaigns/{run_id}/"
    if (
        not str(descriptor_record["repo_tar_key"]).startswith(
            campaign_prefix + "repository/"
        )
        or not str(descriptor_record["campaign_descriptor_key"]).startswith(
            campaign_prefix + "submissions/"
        )
        or not str(descriptor_record["approval_key"]).startswith(
            campaign_prefix + "authorities/"
        )
        or descriptor_record["jobs_bucket"] != descriptor_record["bucket"]
    ):
        raise SourceAuthorityError("campaign descriptor coordinate mismatch")
    artifacts = descriptor_record.get("artifacts")
    if (
        type(artifacts) is not dict
        or set(artifacts) != _ARTIFACT_FIELDS
        or any(type(item) is not str or not item for item in artifacts.values())
    ):
        raise SourceAuthorityError("campaign artifact inventory is invalid")
    for field in _ARTIFACT_FIELDS:
        if field.endswith("_sha256"):
            _digest(artifacts[field], field=f"artifacts.{field}")
    if (
        artifacts["source_snapshot_prefix"] != "source-snapshot/"
        or artifacts["non_vq_prefix"] != "non-vq-package/"
        or artifacts["training_baseline_prefix"] != "training-baseline/"
        or not artifacts["teich_pack_key"].startswith("teich-pack/")
        or not artifacts["frozen_prompt_pack_key"].startswith("quality/")
        or not artifacts["training_config_key"].startswith(
            campaign_prefix + "authorities/"
        )
        or artifacts["artifact_inventory_key"]
        != (
            campaign_prefix
            + "inventories/artifact-inventory-"
            + artifacts["artifact_inventory_sha256"]
            + ".json"
        )
    ):
        raise SourceAuthorityError("campaign artifact coordinate mismatch")
    cache_digest = artifacts["qualification_cache_manifest_sha256"]
    expected_cache_prefix = (
        "qualification-cache/"
        if cache_digest == "0" * 64
        else f"qualification-cache/seeds/{run_id}/{cache_digest}/"
    )
    if artifacts["qualification_cache_prefix"] != expected_cache_prefix:
        raise SourceAuthorityError("campaign cache prefix mismatch")
    if descriptor_artifact.key != descriptor_record.get(
        "campaign_descriptor_key"
    ):
        raise SourceAuthorityError("campaign descriptor key mismatch")
    if intent_artifact.key != production_submission_intent_s3_key(
        run_id=run_id,
        intent_body_sha256=str(validated_intent["intent_body_sha256"]),
    ):
        raise SourceAuthorityError("production intent key is not canonical")
    shared = {
        "account_id": descriptor_record["account_id"],
        "region": descriptor_record["region"],
        "bucket": descriptor_record["bucket"],
        "run_id": descriptor_record["run_id"],
        "campaign_identity_sha256": descriptor_record[
            "campaign_identity_sha256"
        ],
        "descriptor_key": descriptor_artifact.key,
        "descriptor_file_sha256": _sha(descriptor_artifact.raw),
        "descriptor_body_sha256": descriptor_record[
            "descriptor_body_sha256"
        ],
        "descriptor_version_id": descriptor_artifact.version_id,
        "must_start_by": descriptor_record["must_start_by"],
    }
    if any(
        type(validated_intent.get(field)) is not type(expected)
        or validated_intent[field] != expected
        for field, expected in shared.items()
    ):
        raise SourceAuthorityError("descriptor/production intent identity drift")
    return (
        descriptor_artifact,
        descriptor_record,
        intent_artifact,
        validated_intent,
    )


def _common_record(record: dict[str, object], *, digest_field: str) -> str:
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("managed_mode") != "production"
        or record.get("account_id") != "246813579024"
        or record.get("region") != "us-west-2"
    ):
        raise SourceAuthorityError("production source discriminant mismatch")
    run_id = _run_id(record.get("run_id"))
    _bucket(record.get("bucket"))
    _digest(record.get(digest_field), field=digest_field)
    for field in record:
        if field.endswith("_sha256"):
            _digest(record.get(field), field=field)
        elif field.endswith("_version_id"):
            _version_id(record.get(field), field=field)
        elif field.endswith("_key"):
            _safe_key(record.get(field), field=field)
    return run_id


def _bind_context(
    record: dict[str, object],
    *,
    descriptor_artifact: VersionedJsonArtifact,
    descriptor_record: dict[str, object],
    intent_artifact: VersionedJsonArtifact,
    intent_record: dict[str, object],
) -> None:
    expected = {
        "account_id": descriptor_record["account_id"],
        "region": descriptor_record["region"],
        "bucket": descriptor_record["bucket"],
        "run_id": descriptor_record["run_id"],
        "campaign_identity_sha256": descriptor_record[
            "campaign_identity_sha256"
        ],
        "descriptor_key": descriptor_artifact.key,
        "descriptor_file_sha256": _sha(descriptor_artifact.raw),
        "descriptor_body_sha256": descriptor_record[
            "descriptor_body_sha256"
        ],
        "descriptor_version_id": descriptor_artifact.version_id,
        "intent_key": intent_artifact.key,
        "intent_file_sha256": _sha(intent_artifact.raw),
        "intent_body_sha256": intent_record["intent_body_sha256"],
        "intent_version_id": intent_artifact.version_id,
    }
    if any(
        type(record.get(field)) is not type(value) or record[field] != value
        for field, value in expected.items()
    ):
        raise SourceAuthorityError("production source predecessor identity drift")


def validate_production_controller_baseline(
    value: Mapping[str, object],
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
) -> dict[str, object]:
    record = _self_hash(
        value,
        fields=_BASELINE_FIELDS,
        digest_field="baseline_body_sha256",
        label="production controller baseline",
    )
    if record.get("record_type") != "glm52_production_controller_baseline_v1":
        raise SourceAuthorityError("controller baseline record type mismatch")
    run_id = _common_record(record, digest_field="baseline_body_sha256")
    if (
        record["intent_key"]
        != production_submission_intent_s3_key(
            run_id=run_id,
            intent_body_sha256=str(record["intent_body_sha256"]),
        )
        or record.get("sky_job_name") != run_id
        or record.get("workspace") != "default"
        or record.get("ssm_ping_status") != "Online"
    ):
        raise SourceAuthorityError("controller baseline fixed contract mismatch")
    role = _exact_string(
        record.get("controller_identity"), field="controller_identity"
    )
    if (
        _ROLE_ARN.fullmatch(role) is None
        or record.get("controller_profile_arn")
        != role.replace(":role/", ":instance-profile/", 1)
    ):
        raise SourceAuthorityError("controller baseline identity mismatch")
    instance_id = _exact_string(
        record.get("controller_instance_id"), field="controller_instance_id"
    )
    if _INSTANCE_ID.fullmatch(instance_id) is None:
        raise SourceAuthorityError("controller instance ID is invalid")
    _exact_string(
        record.get("controller_instance_type"),
        field="controller_instance_type",
    )
    cluster = _exact_string(
        record.get("controller_cluster_name"),
        field="controller_cluster_name",
    )
    if not cluster.startswith("sky-jobs-controller-"):
        raise SourceAuthorityError("controller cluster name is invalid")
    _, observed_at = _time(record.get("observed_at"), field="observed_at")
    history = record.get("exact_name_history")
    if type(history) is not list:
        raise SourceAuthorityError("exact_name_history must be a list")
    normalized_history = []
    seen = set()
    for index, item in enumerate(history):
        row = _exact_fields(
            item, _HISTORY_FIELDS, label=f"exact_name_history[{index}]"
        )
        job_id = _integer(
            row.get("sky_job_id"),
            field=f"exact_name_history[{index}].sky_job_id",
            minimum=1,
        )
        _, submitted_at = _time(
            row.get("controller_submitted_at"),
            field=f"exact_name_history[{index}].controller_submitted_at",
        )
        if (
            job_id in seen
            or submitted_at > observed_at
            or row.get("sky_job_name") != run_id
            or row.get("workspace") != "default"
            or row.get("controller_identity") != role
            or row.get("controller_status") not in _CONTROLLER_STATUSES
        ):
            raise SourceAuthorityError("controller history identity mismatch")
        seen.add(job_id)
        normalized_history.append(dict(row))
    if normalized_history != sorted(
        normalized_history,
        key=lambda row: (
            int(row["sky_job_id"]),
            str(row["controller_submitted_at"]),
            _canonical(row),
        ),
    ):
        raise SourceAuthorityError("controller history is not canonical")
    derived_active = sorted(
        int(row["sky_job_id"])
        for row in normalized_history
        if row["controller_status"] in _ACTIVE_CONTROLLER_STATUSES
    )
    if (
        type(record.get("active_exact_name_job_ids")) is not list
        or record["active_exact_name_job_ids"] != derived_active
        or derived_active
        or type(record.get("active_tagged_p5_instance_ids")) is not list
        or record["active_tagged_p5_instance_ids"]
    ):
        raise SourceAuthorityError("controller baseline is not quiescent")
    (
        descriptor_artifact,
        descriptor_record,
        intent_artifact,
        intent_record,
    ) = _source_context(descriptor=descriptor, intent=intent)
    _bind_context(
        record,
        descriptor_artifact=descriptor_artifact,
        descriptor_record=descriptor_record,
        intent_artifact=intent_artifact,
        intent_record=intent_record,
    )
    return dict(record)


def _artifact_binding(
    record: dict[str, object],
    *,
    prefix: str,
    artifact: object,
    validator: object,
    validator_kwargs: Mapping[str, object],
) -> tuple[VersionedJsonArtifact, dict[str, object]]:
    exact, parsed = _parse_artifact(artifact, label=prefix)
    validated = validator(parsed, **dict(validator_kwargs))  # type: ignore[operator]
    digest_field = {
        "controller_baseline": "baseline_body_sha256",
        "must_start_control_plane_ready": "control_plane_ready_body_sha256",
    }[prefix]
    expected = {
        f"{prefix}_key": exact.key,
        f"{prefix}_file_sha256": _sha(exact.raw),
        f"{prefix}_body_sha256": validated[digest_field],
        f"{prefix}_version_id": exact.version_id,
    }
    if any(record.get(field) != value for field, value in expected.items()):
        raise SourceAuthorityError(f"{prefix} predecessor identity drift")
    return exact, validated


def _validate_activation(
    activation: object, *, record: dict[str, object]
) -> None:
    value = _exact_fields(
        activation, _ACTIVATION_FIELDS, label="activation_capabilities"
    )
    if (
        value.get("stack_status") not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
        or type(value.get("stack_operation_in_progress")) is not bool
        or value["stack_operation_in_progress"] is not False
        or value.get("foundation_revision") != record["template_sha256"]
        or value.get("managed_mode") != "production"
        or value.get("lambda_state") != "Active"
        or value.get("lambda_last_update_status") != "Successful"
        or value.get("lambda_execution_role_arn")
        != (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-sky-production-must-start"
        )
        or value.get("lambda_code_sha256") != record["lambda_code_sha256"]
        or value.get("lambda_code_version_id")
        != record["lambda_code_version_id"]
    ):
        raise SourceAuthorityError("activation fixed state mismatch")
    source_fields = {
        "run_id": record["run_id"],
        "bucket": record["bucket"],
        "descriptor_key": record["descriptor_key"],
        "descriptor_file_sha256": record["descriptor_file_sha256"],
        "descriptor_body_sha256": record["descriptor_body_sha256"],
        "descriptor_version_id": record["descriptor_version_id"],
        "intent_key": record["intent_key"],
        "intent_file_sha256": record["intent_file_sha256"],
        "intent_body_sha256": record["intent_body_sha256"],
        "intent_version_id": record["intent_version_id"],
        "sky_job_name": record["run_id"],
    }
    must_start = value.get("must_start_by")
    source_fields.update(
        {"must_start_by": must_start, "primary_wake_at": must_start}
    )
    if any(value.get(field) != expected for field, expected in source_fields.items()):
        raise SourceAuthorityError("activation source field mismatch")
    qualified = _exact_string(
        value.get("lambda_qualified_arn"), field="lambda_qualified_arn"
    )
    if re.fullmatch(
        re.escape(str(record["lambda_function_arn"])) + r":[1-9][0-9]*",
        qualified,
    ) is None:
        raise SourceAuthorityError("activation qualified Lambda mismatch")
    environment = _exact_fields(
        value.get("lambda_environment"),
        _ENVIRONMENT_FIELDS,
        label="lambda_environment",
    )
    expected_environment = {
        "EXPECTED_ACCOUNT_ID": "246813579024",
        "CAMPAIGN_BUCKET": str(record["bucket"]),
        "CAMPAIGN_RUN_ID": str(record["run_id"]),
        "MANAGED_MODE": "production",
        "CAMPAIGN_DESCRIPTOR_KEY": str(record["descriptor_key"]),
        "CAMPAIGN_DESCRIPTOR_VERSION_ID": str(
            record["descriptor_version_id"]
        ),
        "IMMUTABLE_SUBMISSION_KEY": str(record["intent_key"]),
        "IMMUTABLE_SUBMISSION_VERSION_ID": str(record["intent_version_id"]),
        "SUBMISSION_BODY_SHA256": str(record["intent_body_sha256"]),
        "SKY_JOB_NAME": str(record["run_id"]),
        "MUST_START_BY": str(must_start),
        "COORDINATOR_MODE": "production-dynamic-job-binding-active",
    }
    if environment != expected_environment or any(
        type(item) is not str for item in environment.values()
    ):
        raise SourceAuthorityError("activation environment mismatch")
    base_trigger = {
        "campaign_run_id": str(record["run_id"]),
        "managed_mode": "production",
        "sky_job_name": str(record["run_id"]),
        "must_start_by": str(must_start),
        "descriptor_key": str(record["descriptor_key"]),
        "descriptor_version_id": str(record["descriptor_version_id"]),
        "intent_key": str(record["intent_key"]),
        "intent_version_id": str(record["intent_version_id"]),
    }
    for field, trigger in (
        ("reconciliation_input", "reconcile"),
        ("deadline_input", "primary-deadline"),
    ):
        trigger_value = _exact_fields(
            value.get(field), _TRIGGER_FIELDS, label=field
        )
        if trigger_value != {**base_trigger, "trigger": trigger} or any(
            type(item) is not str for item in trigger_value.values()
        ):
            raise SourceAuthorityError("activation trigger mismatch")
    exact = {
        "reconciliation_schedule_expression": "rate(1 minute)",
        "reconciliation_target_arn": qualified,
        "reconciliation_target_id": "sky-production-must-start-reconcile",
        "reconciliation_retry_max_event_age_seconds": 300,
        "reconciliation_retry_max_attempts": 2,
        "reconciliation_dlq_arn": record["dlq_arn"],
        "deadline_schedule_expression": f"at({str(must_start)[:-1]})",
        "deadline_schedule_timezone": "UTC",
        "deadline_flexible_window_mode": "OFF",
        "deadline_target_arn": qualified,
        "deadline_target_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-sky-production-deadline-scheduler"
        ),
        "deadline_retry_max_event_age_seconds": 300,
        "deadline_retry_max_attempts": 2,
        "deadline_dlq_arn": record["dlq_arn"],
    }
    if any(
        type(value.get(field)) is not type(expected)
        or value[field] != expected
        for field, expected in exact.items()
    ):
        raise SourceAuthorityError("activation schedule identity mismatch")


def validate_production_must_start_control_plane_ready(
    value: Mapping[str, object],
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
) -> dict[str, object]:
    record = _self_hash(
        value,
        fields=_READY_FIELDS,
        digest_field="control_plane_ready_body_sha256",
        label="production control-plane readiness",
    )
    if (
        record.get("record_type")
        != "glm52_production_must_start_control_plane_ready_v1"
    ):
        raise SourceAuthorityError("control-plane readiness record type mismatch")
    run_id = _common_record(
        record, digest_field="control_plane_ready_body_sha256"
    )
    if (
        record.get("reconciliation_rule_state") != "ENABLED"
        or record.get("deadline_schedule_state") != "ENABLED"
        or record.get("coordinator_mode")
        != "production-dynamic-job-binding-active"
    ):
        raise SourceAuthorityError("control-plane readiness is not enabled")
    (
        descriptor_artifact,
        descriptor_record,
        intent_artifact,
        intent_record,
    ) = _source_context(descriptor=descriptor, intent=intent)
    _bind_context(
        record,
        descriptor_artifact=descriptor_artifact,
        descriptor_record=descriptor_record,
        intent_artifact=intent_artifact,
        intent_record=intent_record,
    )
    _artifact_binding(
        record,
        prefix="controller_baseline",
        artifact=controller_baseline,
        validator=validate_production_controller_baseline,
        validator_kwargs={"descriptor": descriptor, "intent": intent},
    )
    if record["controller_baseline_key"] != production_controller_baseline_s3_key(
        run_id=run_id,
        baseline_body_sha256=str(record["controller_baseline_body_sha256"]),
    ):
        raise SourceAuthorityError("controller baseline key is not canonical")
    deployment = {
        field: record[field] for field in _DEPLOYMENT_FIELDS
    }
    if (
        re.fullmatch(
            r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
            r"keep-glm52-sky-production-control-plane/[A-Za-z0-9-]+",
            str(deployment["stack_id"]),
        )
        is None
        or deployment["lambda_function_arn"]
        != (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-sky-production-must-start"
        )
        or deployment["reconciliation_rule_arn"]
        != (
            "arn:aws:events:us-west-2:246813579024:rule/"
            "keep-glm52-sky-production-reconcile"
        )
        or deployment["deadline_schedule_arn"]
        != (
            "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
            "keep-glm52-sky-production-deadline"
        )
        or deployment["dlq_arn"]
        != (
            "arn:aws:sqs:us-west-2:246813579024:"
            "keep-glm52-sky-production-must-start-dlq"
        )
    ):
        raise SourceAuthorityError("control-plane deployment ARN mismatch")
    for field in ("template_sha256", "lambda_code_sha256", "iam_policy_sha256"):
        _digest(deployment[field], field=field)
    _version_id(
        deployment["lambda_code_version_id"],
        field="lambda_code_version_id",
    )
    if deployment["lambda_code_s3_key"] != (
        f"campaigns/{run_id}/control-plane/production/lambda-code/"
        f"{deployment['lambda_code_sha256']}/sky-must-start.zip"
    ):
        raise SourceAuthorityError("Lambda code key mismatch")
    _validate_activation(deployment["activation_capabilities"], record=record)
    _, observed = _time(record.get("observed_at"), field="observed_at")
    _, must_start = _time(
        deployment["activation_capabilities"].get("must_start_by"),
        field="must_start_by",
    )
    if observed >= must_start:
        raise SourceAuthorityError("readiness observed after must-start deadline")
    production_must_start_control_plane_ready_s3_key(
        run_id=run_id,
        intent_body_sha256=str(record["intent_body_sha256"]),
        control_plane_ready_body_sha256=str(
            record["control_plane_ready_body_sha256"]
        ),
    )
    return dict(record)


def validate_production_submission_acquired(
    value: Mapping[str, object],
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    now: object,
) -> dict[str, object]:
    record = _self_hash(
        value,
        fields=_ACQUISITION_FIELDS,
        digest_field="acquisition_body_sha256",
        label="production submission acquisition",
    )
    if (
        record.get("record_type")
        != "glm52_sky_production_submission_acquired_v1"
    ):
        raise SourceAuthorityError("submission acquisition record type mismatch")
    run_id = _common_record(record, digest_field="acquisition_body_sha256")
    (
        descriptor_artifact,
        descriptor_record,
        intent_artifact,
        intent_record,
    ) = _source_context(descriptor=descriptor, intent=intent)
    _bind_context(
        record,
        descriptor_artifact=descriptor_artifact,
        descriptor_record=descriptor_record,
        intent_artifact=intent_artifact,
        intent_record=intent_record,
    )
    _artifact_binding(
        record,
        prefix="controller_baseline",
        artifact=controller_baseline,
        validator=validate_production_controller_baseline,
        validator_kwargs={"descriptor": descriptor, "intent": intent},
    )
    _artifact_binding(
        record,
        prefix="must_start_control_plane_ready",
        artifact=must_start_control_plane_ready,
        validator=validate_production_must_start_control_plane_ready,
        validator_kwargs={
            "descriptor": descriptor,
            "intent": intent,
            "controller_baseline": controller_baseline,
        },
    )
    expected_keys = {
        "intent_key": production_submission_intent_s3_key(
            run_id=run_id,
            intent_body_sha256=str(record["intent_body_sha256"]),
        ),
        "controller_baseline_key": production_controller_baseline_s3_key(
            run_id=run_id,
            baseline_body_sha256=str(
                record["controller_baseline_body_sha256"]
            ),
        ),
        "must_start_control_plane_ready_key": (
            production_must_start_control_plane_ready_s3_key(
                run_id=run_id,
                intent_body_sha256=str(record["intent_body_sha256"]),
                control_plane_ready_body_sha256=str(
                    record["must_start_control_plane_ready_body_sha256"]
                ),
            )
        ),
    }
    if any(record[field] != expected for field, expected in expected_keys.items()):
        raise SourceAuthorityError("submission acquisition key mismatch")
    if record.get("sky_job_name") != run_id:
        raise SourceAuthorityError("submission acquisition job mismatch")
    time_fields = (
        "spend_snapshot_observed_at",
        "controller_baseline_observed_at",
        "control_plane_ready_observed_at",
        "acquired_at",
        "must_start_by",
    )
    times = {field: _time(record.get(field), field=field)[1] for field in time_fields}
    if not (
        times["spend_snapshot_observed_at"]
        <= times["controller_baseline_observed_at"]
        <= times["control_plane_ready_observed_at"]
        <= times["acquired_at"]
        < times["must_start_by"]
    ):
        raise SourceAuthorityError("submission acquisition time order mismatch")
    for field in time_fields[:3]:
        if times["acquired_at"] - times[field] > timedelta(seconds=60):
            raise SourceAuthorityError("submission acquisition source is stale")
    if type(now) is datetime:
        if now.tzinfo is None or now.microsecond != 0:
            raise SourceAuthorityError("now must be whole-second aware")
        current = now.astimezone(timezone.utc)
    else:
        current = _time(now, field="now")[1]
    if current < times["acquired_at"] or current >= times["must_start_by"]:
        raise SourceAuthorityError("submission acquisition is not current")
    for field in time_fields[:3]:
        if current - times[field] > timedelta(seconds=60):
            raise SourceAuthorityError("submission acquisition source is stale now")
    production_submission_acquired_s3_key(
        run_id=run_id,
        descriptor_file_sha256=str(record["descriptor_file_sha256"]),
    )
    return dict(record)


__all__ = [
    "SourceAuthorityError",
    "VersionedJsonArtifact",
    "validate_gpu_spend_snapshot",
    "validate_production_submission_intent",
    "validate_production_controller_baseline",
    "validate_production_must_start_control_plane_ready",
    "validate_production_submission_acquired",
    "gpu_spend_snapshot_s3_key",
    "production_submission_intent_s3_key",
    "production_controller_baseline_s3_key",
    "production_must_start_control_plane_ready_s3_key",
    "production_submission_acquired_s3_key",
]
