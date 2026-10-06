"""Pure short-lived authorities for one guarded SkyPilot submission.

The builders and validators in this module perform no AWS, SSM, S3, SkyPilot,
or deployment I/O.  Callers must obtain authenticated observations elsewhere
and pass their normalized values here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_ACCOUNT_ID,
    APPROVED_REGION,
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_submission_lifecycle import (
    EXPECTED_WORKSPACE,
    INTENT_DIGEST_FIELD,
    INTENT_RECORD_TYPE,
    INTENT_SCHEMA_VERSION,
    expected_sky_job_name,
    submission_intent_s3_key,
)

CONTROLLER_BASELINE_RECORD_TYPE = "glm52_controller_baseline_v1"
CONTROLLER_BASELINE_SCHEMA_VERSION = 1
CONTROLLER_BASELINE_DIGEST_FIELD = "baseline_body_sha256"
MUST_START_CONTROL_PLANE_READY_RECORD_TYPE = "glm52_must_start_control_plane_ready_v1"
MUST_START_CONTROL_PLANE_READY_SCHEMA_VERSION = 1
MUST_START_CONTROL_PLANE_READY_DIGEST_FIELD = "control_plane_ready_body_sha256"
QUALIFICATION_MODE = "qualification"
COORDINATOR_MODE = "dynamic-job-binding-active"
MAX_LIVE_AUTHORITY_AGE = timedelta(seconds=60)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUCKET = re.compile(
    r"^(?![0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$)"
    r"[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])?$"
)
_INSTANCE_ID = re.compile(r"^i-[0-9a-f]{17}$")
_INSTANCE_TYPE = re.compile(r"^[a-z0-9][a-z0-9.]*$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_CONTROLLER_HISTORY_FIELDS = frozenset(
    {
        "sky_job_id",
        "sky_job_name",
        "workspace",
        "controller_submitted_at",
        "controller_status",
        "controller_identity",
    }
)
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
_ACTIVE_CONTROLLER_STATUSES = frozenset(
    {"PENDING", "STARTING", "RUNNING", "RECOVERING", "CANCELLING"}
)

_READINESS_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "repo_tar_sha256",
        "seed_descriptor_key",
        "seed_descriptor_file_sha256",
        "seed_descriptor_body_sha256",
        "seed_campaign_identity_sha256",
        "seed_repo_tar_sha256",
        "seed_descriptor",
        "staged_readiness_key",
        "staged_readiness_file_sha256",
        "staged_readiness_body_sha256",
        "cache_seed_acceptance_key",
        "cache_seed_acceptance_file_sha256",
        "cache_seed_acceptance_body_sha256",
        "gpu_spend_snapshot_key",
        "gpu_spend_snapshot_sha256",
        "gpu_spend_snapshot_body_sha256",
        "gpu_spend_ledger_tip_record_sha256",
        "remaining_gpu_seconds",
        "remaining_gpu_cost_usd",
        "qualification_allowance_seconds",
        "qualification_allowance_cost_usd",
        "rehearsal_evidence_key",
        "rehearsal_evidence_sha256",
        "rehearsal_evidence_body_sha256",
        "sky_task_name",
        "sky_job_name",
        "must_start_by",
        "allowed_gpu_seconds",
        "allowed_gpu_cost_usd",
        "built_at",
    }
)
_READINESS_DIGEST_FIELD = "readiness_body_sha256"
_READINESS_FIELDS = _READINESS_BODY_FIELDS | {_READINESS_DIGEST_FIELD}
_READINESS_SHA_FIELDS = frozenset(
    field for field in _READINESS_FIELDS if field.endswith("_sha256")
)

_INTENT_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "repo_tar_sha256",
        "cache_seed_acceptance_key",
        "cache_seed_acceptance_file_sha256",
        "cache_seed_acceptance_body_sha256",
        "gpu_spend_snapshot_key",
        "gpu_spend_snapshot_sha256",
        "gpu_spend_snapshot_body_sha256",
        "gpu_spend_ledger_tip_record_sha256",
        "remaining_gpu_seconds",
        "remaining_gpu_cost_usd",
        "qualification_allowance_seconds",
        "qualification_allowance_cost_usd",
        "open_allocation_count",
        "qualification_submission_ready_key",
        "qualification_submission_ready_sha256",
        "qualification_submission_ready_body_sha256",
        "sky_job_name",
        "must_start_by",
        "intent_at",
    }
)
_INTENT_FIELDS = _INTENT_BODY_FIELDS | {INTENT_DIGEST_FIELD}
_INTENT_SHA_FIELDS = frozenset(
    field for field in _INTENT_FIELDS if field.endswith("_sha256")
)
_INTENT_READINESS_SHARED_FIELDS = (
    "account_id",
    "region",
    "run_id",
    "managed_mode",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "campaign_identity_sha256",
    "approval_sha256",
    "approval_body_sha256",
    "repo_tar_sha256",
    "cache_seed_acceptance_key",
    "cache_seed_acceptance_file_sha256",
    "cache_seed_acceptance_body_sha256",
    "gpu_spend_snapshot_key",
    "gpu_spend_snapshot_sha256",
    "gpu_spend_snapshot_body_sha256",
    "gpu_spend_ledger_tip_record_sha256",
    "remaining_gpu_seconds",
    "remaining_gpu_cost_usd",
    "qualification_allowance_seconds",
    "qualification_allowance_cost_usd",
    "sky_job_name",
    "must_start_by",
)

_BASELINE_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "managed_mode",
        "intent_key",
        "intent_file_sha256",
        "intent_body_sha256",
        "sky_job_name",
        "workspace",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
        "ssm_ping_status",
        "exact_name_history",
        "active_exact_name_job_ids",
        "active_tagged_p5_instance_ids",
        "observed_at",
    }
)
_BASELINE_FIELDS = _BASELINE_BODY_FIELDS | {CONTROLLER_BASELINE_DIGEST_FIELD}

_MUST_START_BODY_FIELDS = frozenset(
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
        "controller_baseline_key",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "stack_id",
        "template_sha256",
        "lambda_function_arn",
        "lambda_code_sha256",
        "iam_policy_sha256",
        "reconciliation_rule_arn",
        "reconciliation_rule_state",
        "deadline_schedule_arn",
        "deadline_schedule_state",
        "dlq_arn",
        "coordinator_mode",
        "observed_at",
    }
)
_MUST_START_FIELDS = _MUST_START_BODY_FIELDS | {
    MUST_START_CONTROL_PLANE_READY_DIGEST_FIELD
}

MUST_START_DEPLOYMENT_IDENTITY_FIELDS = frozenset(
    {
        "stack_id",
        "template_sha256",
        "lambda_function_arn",
        "lambda_code_sha256",
        "iam_policy_sha256",
        "reconciliation_rule_arn",
        "deadline_schedule_arn",
        "dlq_arn",
    }
)


class SubmissionLiveAuthorityValidationError(ValueError):
    """A short-lived submission authority is malformed, stale, or foreign."""


def canonical_bytes(value: object) -> bytes:
    """Return compact, sorted, finite, ASCII JSON without a newline."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SubmissionLiveAuthorityValidationError(
            "submission live authority is not canonical finite JSON"
        ) from error


def canonical_file_bytes(value: object) -> bytes:
    """Return the immutable artifact encoding: canonical JSON plus one LF."""

    return canonical_bytes(value) + b"\n"


def canonical_sha256(value: object) -> str:
    """Return a SHA-256 over canonical JSON without a newline."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_file_sha256(value: object) -> str:
    """Return a SHA-256 over canonical JSON with exactly one trailing LF."""

    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


def _require_actual_artifact(
    *,
    value: object,
    actual_key: object,
    expected_key: str,
    raw: object,
    label: str,
) -> None:
    if actual_key != expected_key:
        raise SubmissionLiveAuthorityValidationError(
            f"{label} key does not match its exact authority key"
        )
    if type(raw) is not bytes or raw != canonical_file_bytes(value):
        raise SubmissionLiveAuthorityValidationError(
            f"{label} bytes are not exact canonical JSON plus one newline"
        )


def _require_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SubmissionLiveAuthorityValidationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise SubmissionLiveAuthorityValidationError(
            f"{label} contains a non-string field"
        )
    result = dict(value)
    canonical_bytes(result)
    return result


def _require_exact_fields(
    value: Mapping[str, object],
    *,
    fields: frozenset[str],
    label: str,
) -> None:
    if set(value) != fields:
        raise SubmissionLiveAuthorityValidationError(f"{label} schema mismatch")


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise SubmissionLiveAuthorityValidationError(
            f"{field} must be a canonical lowercase SHA-256"
        )
    return value


def _require_integer(
    value: object,
    *,
    field: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise SubmissionLiveAuthorityValidationError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubmissionLiveAuthorityValidationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SubmissionLiveAuthorityValidationError(f"{field} must be a finite number")
    return result


def _canonical_time(
    value: datetime | str | object,
    *,
    field: str,
    permit_datetime: bool,
) -> tuple[str, datetime]:
    if isinstance(value, datetime) and permit_datetime:
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise SubmissionLiveAuthorityValidationError(
                f"{field} is not a valid timestamp"
            ) from error
    else:
        raise SubmissionLiveAuthorityValidationError(
            f"{field} must be a canonical UTC timestamp"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SubmissionLiveAuthorityValidationError(
            f"{field} must include a UTC offset"
        )
    parsed = parsed.astimezone(UTC)
    if parsed.microsecond != 0:
        raise SubmissionLiveAuthorityValidationError(
            f"{field} must use whole-second precision"
        )
    canonical = parsed.isoformat(timespec="seconds").replace("+00:00", "Z")
    if isinstance(value, str) and value != canonical:
        raise SubmissionLiveAuthorityValidationError(f"{field} is not canonical UTC")
    return canonical, parsed


def _require_run_id(value: object) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise SubmissionLiveAuthorityValidationError("run_id is invalid")
    if value in {".", ".."}:
        raise SubmissionLiveAuthorityValidationError("run_id is invalid")
    return value


def _require_qualification_mode(value: object) -> str:
    if value != QUALIFICATION_MODE:
        raise SubmissionLiveAuthorityValidationError(
            "managed_mode must be qualification"
        )
    return QUALIFICATION_MODE


def _require_campaign_key(
    value: object,
    *,
    run_id: str,
    field: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise SubmissionLiveAuthorityValidationError(f"{field} is invalid")
    if "\\" in value or value.startswith("/") or "//" in value:
        raise SubmissionLiveAuthorityValidationError(f"{field} is invalid")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SubmissionLiveAuthorityValidationError(f"{field} contains path traversal")
    if not value.startswith(f"campaigns/{run_id}/"):
        raise SubmissionLiveAuthorityValidationError(f"{field} is foreign to run_id")
    return value


def _require_bucket(value: object) -> str:
    if not isinstance(value, str) or _BUCKET.fullmatch(value) is None:
        raise SubmissionLiveAuthorityValidationError("bucket is invalid")
    if ".." in value:
        raise SubmissionLiveAuthorityValidationError("bucket is invalid")
    return value


def _require_safe_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise SubmissionLiveAuthorityValidationError(f"{field} is invalid")
    if value in {".", ".."} or ".." in value:
        raise SubmissionLiveAuthorityValidationError(f"{field} is invalid")
    return value


def _require_self_hashed(
    value: object,
    *,
    fields: frozenset[str],
    digest_field: str,
    schema_version: int,
    record_type: str,
    label: str,
) -> dict[str, object]:
    result = _require_mapping(value, label=label)
    _require_exact_fields(result, fields=fields, label=label)
    if (
        type(result.get("schema_version")) is not int
        or result.get("schema_version") != schema_version
        or result.get("record_type") != record_type
    ):
        raise SubmissionLiveAuthorityValidationError(f"{label} schema mismatch")
    body = dict(result)
    digest = _require_sha(body.pop(digest_field), field=digest_field)
    if canonical_sha256(body) != digest:
        raise SubmissionLiveAuthorityValidationError(f"{label} body SHA-256 mismatch")
    return result


def _validate_readiness_reference(value: object) -> dict[str, object]:
    readiness = _require_self_hashed(
        value,
        fields=_READINESS_FIELDS,
        digest_field=_READINESS_DIGEST_FIELD,
        schema_version=1,
        record_type="glm52_qualification_submission_ready_v1",
        label="qualification submission readiness",
    )
    for field in _READINESS_SHA_FIELDS:
        _require_sha(readiness.get(field), field=field)
    run_id = _require_run_id(readiness.get("run_id"))
    _require_qualification_mode(readiness.get("managed_mode"))
    for field in (
        "descriptor_key",
        "seed_descriptor_key",
        "staged_readiness_key",
        "cache_seed_acceptance_key",
        "gpu_spend_snapshot_key",
        "rehearsal_evidence_key",
    ):
        _require_campaign_key(readiness.get(field), run_id=run_id, field=field)
    expected_keys = {
        "cache_seed_acceptance_key": (
            f"campaigns/{run_id}/qualification-cache-seed/accepted/"
            f"{readiness['cache_seed_acceptance_body_sha256']}/"
            "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
        ),
        "gpu_spend_snapshot_key": (
            f"campaigns/{run_id}/spend-snapshots/"
            f"{readiness['gpu_spend_snapshot_body_sha256']}/"
            "GPU_SPEND_SNAPSHOT.json"
        ),
        "rehearsal_evidence_key": (
            f"campaigns/{run_id}/qualification/rehearsals/"
            f"{readiness['rehearsal_evidence_body_sha256']}/"
            "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
    }
    for field, expected in expected_keys.items():
        if readiness.get(field) != expected:
            raise SubmissionLiveAuthorityValidationError(
                f"{field} does not match its exact content-addressed key"
            )
    _canonical_time(
        readiness.get("built_at"),
        field="qualification readiness built_at",
        permit_datetime=False,
    )
    _canonical_time(
        readiness.get("must_start_by"),
        field="qualification readiness must_start_by",
        permit_datetime=False,
    )
    for field in (
        "remaining_gpu_seconds",
        "qualification_allowance_seconds",
        "allowed_gpu_seconds",
    ):
        _require_integer(readiness.get(field), field=field)
    for field in (
        "remaining_gpu_cost_usd",
        "qualification_allowance_cost_usd",
        "allowed_gpu_cost_usd",
    ):
        _require_finite_number(readiness.get(field), field=field)
    return readiness


def _validate_intent_reference(value: object) -> dict[str, object]:
    intent = _require_self_hashed(
        value,
        fields=_INTENT_FIELDS,
        digest_field=INTENT_DIGEST_FIELD,
        schema_version=INTENT_SCHEMA_VERSION,
        record_type=INTENT_RECORD_TYPE,
        label="submission intent",
    )
    for field in _INTENT_SHA_FIELDS:
        _require_sha(intent.get(field), field=field)
    run_id = _require_run_id(intent.get("run_id"))
    mode = _require_qualification_mode(intent.get("managed_mode"))
    for field in (
        "descriptor_key",
        "cache_seed_acceptance_key",
        "gpu_spend_snapshot_key",
        "qualification_submission_ready_key",
    ):
        _require_campaign_key(intent.get(field), run_id=run_id, field=field)
    if intent.get("account_id") != APPROVED_ACCOUNT_ID:
        raise SubmissionLiveAuthorityValidationError(
            "submission intent account_id is foreign"
        )
    if intent.get("region") != APPROVED_REGION:
        raise SubmissionLiveAuthorityValidationError(
            "submission intent region is foreign"
        )
    if intent.get("sky_job_name") != expected_sky_job_name(run_id, mode):
        raise SubmissionLiveAuthorityValidationError(
            "submission intent sky_job_name is foreign"
        )
    _require_integer(
        intent.get("remaining_gpu_seconds"),
        field="remaining_gpu_seconds",
    )
    _require_integer(
        intent.get("qualification_allowance_seconds"),
        field="qualification_allowance_seconds",
    )
    if (
        _require_integer(
            intent.get("open_allocation_count"),
            field="open_allocation_count",
        )
        != 0
    ):
        raise SubmissionLiveAuthorityValidationError(
            "submission intent has an open allocation"
        )
    _require_finite_number(
        intent.get("remaining_gpu_cost_usd"),
        field="remaining_gpu_cost_usd",
    )
    _require_finite_number(
        intent.get("qualification_allowance_cost_usd"),
        field="qualification_allowance_cost_usd",
    )
    _, intent_at = _canonical_time(
        intent.get("intent_at"),
        field="submission intent intent_at",
        permit_datetime=False,
    )
    _, deadline = _canonical_time(
        intent.get("must_start_by"),
        field="submission intent must_start_by",
        permit_datetime=False,
    )
    if intent_at >= deadline:
        raise SubmissionLiveAuthorityValidationError(
            "submission intent is not before must_start_by"
        )
    submission_intent_s3_key(
        run_id=run_id,
        managed_mode=mode,
        intent_body_sha256=str(intent[INTENT_DIGEST_FIELD]),
    )
    return intent


def _authenticate_authority_chain(
    *,
    descriptor: object,
    descriptor_key: object,
    descriptor_raw: object,
    qualification_submission_ready: object,
    qualification_submission_ready_key: object,
    qualification_submission_ready_raw: object,
    intent: object,
    intent_key: object,
    intent_raw: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    descriptor_value = _require_mapping(descriptor, label="campaign descriptor")
    try:
        descriptor_value = validate_sky_campaign_descriptor(descriptor_value)
    except (KeyError, TypeError, ValueError) as error:
        raise SubmissionLiveAuthorityValidationError(
            f"campaign descriptor is invalid: {error}"
        ) from error
    readiness = _validate_readiness_reference(qualification_submission_ready)
    intent_value = _validate_intent_reference(intent)

    run_id = _require_run_id(descriptor_value.get("run_id"))
    expected_descriptor_key = _require_campaign_key(
        descriptor_value["campaign_descriptor_key"],
        run_id=run_id,
        field="descriptor_key",
    )
    expected_readiness_key = (
        f"campaigns/{run_id}/qualification/submission-ready/"
        f"{readiness[_READINESS_DIGEST_FIELD]}/"
        "QUALIFICATION_SUBMISSION_READY.json"
    )
    expected_intent_key = submission_intent_s3_key(
        run_id=run_id,
        managed_mode=str(intent_value["managed_mode"]),
        intent_body_sha256=str(intent_value[INTENT_DIGEST_FIELD]),
    )
    _require_actual_artifact(
        value=descriptor_value,
        actual_key=descriptor_key,
        expected_key=expected_descriptor_key,
        raw=descriptor_raw,
        label="descriptor",
    )
    _require_actual_artifact(
        value=readiness,
        actual_key=qualification_submission_ready_key,
        expected_key=expected_readiness_key,
        raw=qualification_submission_ready_raw,
        label="qualification submission readiness",
    )
    _require_actual_artifact(
        value=intent_value,
        actual_key=intent_key,
        expected_key=expected_intent_key,
        raw=intent_raw,
        label="submission intent",
    )
    descriptor_expected = {
        "account_id": descriptor_value["account_id"],
        "region": descriptor_value["region"],
        "run_id": run_id,
        "managed_mode": QUALIFICATION_MODE,
        "descriptor_key": descriptor_value["campaign_descriptor_key"],
        "descriptor_file_sha256": canonical_file_sha256(descriptor_value),
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor_value["campaign_identity_sha256"],
        "approval_sha256": descriptor_value["approval_sha256"],
        "repo_tar_sha256": descriptor_value["repo_tar_sha256"],
        "sky_job_name": expected_sky_job_name(run_id, QUALIFICATION_MODE),
        "must_start_by": descriptor_value["must_start_by"],
    }
    for label, record in (
        ("qualification submission readiness", readiness),
        ("submission intent", intent_value),
    ):
        for field, expected in descriptor_expected.items():
            if record.get(field) != expected:
                raise SubmissionLiveAuthorityValidationError(
                    f"{label} {field} does not match the campaign descriptor"
                )

    for field in _INTENT_READINESS_SHARED_FIELDS:
        if intent_value.get(field) != readiness.get(field):
            raise SubmissionLiveAuthorityValidationError(
                f"submission intent/readiness {field} drift"
            )
    if intent_value.get("qualification_submission_ready_key") != expected_readiness_key:
        raise SubmissionLiveAuthorityValidationError(
            "submission intent qualification readiness key mismatch"
        )
    if intent_value.get(
        "qualification_submission_ready_sha256"
    ) != canonical_file_sha256(readiness):
        raise SubmissionLiveAuthorityValidationError(
            "submission intent qualification readiness file SHA-256 mismatch"
        )
    if (
        intent_value.get("qualification_submission_ready_body_sha256")
        != readiness[_READINESS_DIGEST_FIELD]
    ):
        raise SubmissionLiveAuthorityValidationError(
            "submission intent qualification readiness body SHA-256 mismatch"
        )
    _, built_at = _canonical_time(
        readiness["built_at"],
        field="qualification readiness built_at",
        permit_datetime=False,
    )
    _, intent_at = _canonical_time(
        intent_value["intent_at"],
        field="submission intent intent_at",
        permit_datetime=False,
    )
    if intent_at < built_at or intent_at - built_at > timedelta(minutes=5):
        raise SubmissionLiveAuthorityValidationError(
            "submission intent is outside the readiness freshness window"
        )
    return descriptor_value, readiness, intent_value


def _validate_controller_coordinates(
    *,
    descriptor: Mapping[str, object],
    controller_instance_id: object,
    controller_instance_type: object,
    controller_profile_arn: object,
    controller_cluster_name: object,
    ssm_ping_status: object,
) -> tuple[str, str, str, str]:
    if (
        not isinstance(controller_instance_id, str)
        or _INSTANCE_ID.fullmatch(controller_instance_id) is None
    ):
        raise SubmissionLiveAuthorityValidationError(
            "controller_instance_id is invalid"
        )
    if (
        not isinstance(controller_instance_type, str)
        or _INSTANCE_TYPE.fullmatch(controller_instance_type) is None
    ):
        raise SubmissionLiveAuthorityValidationError(
            "controller_instance_type is invalid"
        )
    controller_role = descriptor.get("controller_identity")
    role_prefix = f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
    if not isinstance(controller_role, str) or not controller_role.startswith(
        role_prefix
    ):
        raise SubmissionLiveAuthorityValidationError(
            "descriptor controller identity is foreign"
        )
    expected_profile = (
        f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:instance-profile/"
        f"{controller_role.removeprefix(role_prefix)}"
    )
    if controller_profile_arn != expected_profile:
        raise SubmissionLiveAuthorityValidationError(
            "controller_profile_arn does not match descriptor controller identity"
        )
    cluster = _require_safe_name(
        controller_cluster_name,
        field="controller_cluster_name",
    )
    if ssm_ping_status != "Online":
        raise SubmissionLiveAuthorityValidationError("ssm_ping_status must be Online")
    return (
        controller_instance_id,
        controller_instance_type,
        expected_profile,
        cluster,
    )


def _validate_controller_history(
    value: object,
    *,
    intent: Mapping[str, object],
    descriptor: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[int]]:
    if type(value) is not list:
        raise SubmissionLiveAuthorityValidationError(
            "exact_name_history must be an exact list"
        )
    result: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    active_ids: list[int] = []
    for index, item in enumerate(value):
        row = _require_mapping(item, label=f"exact_name_history[{index}]")
        _require_exact_fields(
            row,
            fields=_CONTROLLER_HISTORY_FIELDS,
            label=f"exact_name_history[{index}]",
        )
        job_id = _require_integer(
            row.get("sky_job_id"),
            field=f"exact_name_history[{index}].sky_job_id",
            minimum=1,
        )
        if job_id in seen_ids:
            raise SubmissionLiveAuthorityValidationError(
                "exact_name_history contains an ambiguous duplicate sky_job_id"
            )
        seen_ids.add(job_id)
        if row.get("sky_job_name") != intent["sky_job_name"]:
            raise SubmissionLiveAuthorityValidationError(
                f"exact_name_history[{index}].sky_job_name is foreign"
            )
        if row.get("workspace") != EXPECTED_WORKSPACE:
            raise SubmissionLiveAuthorityValidationError(
                f"exact_name_history[{index}].workspace is foreign"
            )
        if row.get("controller_identity") != descriptor["controller_identity"]:
            raise SubmissionLiveAuthorityValidationError(
                f"exact_name_history[{index}].controller_identity is foreign"
            )
        submitted_at, _ = _canonical_time(
            row.get("controller_submitted_at"),
            field=f"exact_name_history[{index}].controller_submitted_at",
            permit_datetime=False,
        )
        status = row.get("controller_status")
        if not isinstance(status, str) or status not in _CONTROLLER_STATUSES:
            raise SubmissionLiveAuthorityValidationError(
                f"exact_name_history[{index}].controller_status is invalid"
            )
        normalized = {
            "sky_job_id": job_id,
            "sky_job_name": row["sky_job_name"],
            "workspace": row["workspace"],
            "controller_submitted_at": submitted_at,
            "controller_status": status,
            "controller_identity": row["controller_identity"],
        }
        result.append(normalized)
        if status in _ACTIVE_CONTROLLER_STATUSES:
            active_ids.append(job_id)
    expected = sorted(
        result,
        key=lambda row: (
            int(row["sky_job_id"]),
            str(row["controller_submitted_at"]),
            canonical_bytes(row),
        ),
    )
    if result != expected:
        raise SubmissionLiveAuthorityValidationError(
            "exact_name_history is not in canonical sorted order"
        )
    return result, sorted(active_ids)


def _require_sorted_unique_job_ids(value: object) -> list[int]:
    if type(value) is not list:
        raise SubmissionLiveAuthorityValidationError(
            "active_exact_name_job_ids must be an exact list"
        )
    result = [
        _require_integer(
            item,
            field="active_exact_name_job_ids",
            minimum=1,
        )
        for item in value
    ]
    if result != sorted(result) or len(result) != len(set(result)):
        raise SubmissionLiveAuthorityValidationError(
            "active_exact_name_job_ids must be sorted and unique"
        )
    return result


def _require_sorted_unique_instance_ids(value: object) -> list[str]:
    if type(value) is not list:
        raise SubmissionLiveAuthorityValidationError(
            "active_tagged_p5_instance_ids must be an exact list"
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or _INSTANCE_ID.fullmatch(item) is None:
            raise SubmissionLiveAuthorityValidationError(
                "active_tagged_p5_instance_ids contains an invalid instance ID"
            )
        result.append(item)
    if result != sorted(result) or len(result) != len(set(result)):
        raise SubmissionLiveAuthorityValidationError(
            "active_tagged_p5_instance_ids must be sorted and unique"
        )
    return result


def _controller_baseline_body(
    *,
    descriptor: object,
    descriptor_key: object,
    descriptor_raw: object,
    qualification_submission_ready: object,
    qualification_submission_ready_key: object,
    qualification_submission_ready_raw: object,
    intent: object,
    intent_key: object,
    intent_raw: object,
    controller_instance_id: object,
    controller_instance_type: object,
    controller_profile_arn: object,
    controller_cluster_name: object,
    ssm_ping_status: object,
    exact_name_history: object,
    active_exact_name_job_ids: object,
    active_tagged_p5_instance_ids: object,
    observed_at: datetime | str | object,
) -> tuple[dict[str, object], datetime, datetime]:
    descriptor_value, _, intent_value = _authenticate_authority_chain(
        descriptor=descriptor,
        descriptor_key=descriptor_key,
        descriptor_raw=descriptor_raw,
        qualification_submission_ready=qualification_submission_ready,
        qualification_submission_ready_key=(qualification_submission_ready_key),
        qualification_submission_ready_raw=(qualification_submission_ready_raw),
        intent=intent,
        intent_key=intent_key,
        intent_raw=intent_raw,
    )
    (
        instance_id,
        instance_type,
        profile_arn,
        cluster_name,
    ) = _validate_controller_coordinates(
        descriptor=descriptor_value,
        controller_instance_id=controller_instance_id,
        controller_instance_type=controller_instance_type,
        controller_profile_arn=controller_profile_arn,
        controller_cluster_name=controller_cluster_name,
        ssm_ping_status=ssm_ping_status,
    )
    history, history_active_ids = _validate_controller_history(
        exact_name_history,
        intent=intent_value,
        descriptor=descriptor_value,
    )
    active_job_ids = _require_sorted_unique_job_ids(active_exact_name_job_ids)
    if active_job_ids != history_active_ids:
        raise SubmissionLiveAuthorityValidationError(
            "active_exact_name_job_ids do not match exact_name_history"
        )
    if active_job_ids:
        raise SubmissionLiveAuthorityValidationError(
            "controller baseline contains an active exact-name job"
        )
    active_p5_ids = _require_sorted_unique_instance_ids(active_tagged_p5_instance_ids)
    if active_p5_ids:
        raise SubmissionLiveAuthorityValidationError(
            "controller baseline contains an active tagged P5 instance"
        )
    observed_iso, observed_time = _canonical_time(
        observed_at,
        field="controller baseline observed_at",
        permit_datetime=True,
    )
    for index, row in enumerate(history):
        _, submitted_at = _canonical_time(
            row["controller_submitted_at"],
            field=f"exact_name_history[{index}].controller_submitted_at",
            permit_datetime=False,
        )
        if submitted_at > observed_time:
            raise SubmissionLiveAuthorityValidationError(
                f"exact_name_history[{index}] was submitted after baseline observation"
            )
    _, intent_time = _canonical_time(
        intent_value["intent_at"],
        field="submission intent intent_at",
        permit_datetime=False,
    )
    _, deadline = _canonical_time(
        intent_value["must_start_by"],
        field="submission intent must_start_by",
        permit_datetime=False,
    )
    if observed_time < intent_time:
        raise SubmissionLiveAuthorityValidationError(
            "controller baseline predates submission intent"
        )
    if observed_time >= deadline:
        raise SubmissionLiveAuthorityValidationError(
            "controller baseline was observed at or after must_start_by"
        )
    run_id = str(intent_value["run_id"])
    mode = str(intent_value["managed_mode"])
    intent_key = submission_intent_s3_key(
        run_id=run_id,
        managed_mode=mode,
        intent_body_sha256=str(intent_value[INTENT_DIGEST_FIELD]),
    )
    body = {
        "schema_version": CONTROLLER_BASELINE_SCHEMA_VERSION,
        "record_type": CONTROLLER_BASELINE_RECORD_TYPE,
        "account_id": intent_value["account_id"],
        "region": intent_value["region"],
        "bucket": _require_bucket(descriptor_value["bucket"]),
        "run_id": run_id,
        "managed_mode": mode,
        "intent_key": intent_key,
        "intent_file_sha256": canonical_file_sha256(intent_value),
        "intent_body_sha256": intent_value[INTENT_DIGEST_FIELD],
        "sky_job_name": intent_value["sky_job_name"],
        "workspace": EXPECTED_WORKSPACE,
        "controller_instance_id": instance_id,
        "controller_instance_type": instance_type,
        "controller_profile_arn": profile_arn,
        "controller_cluster_name": cluster_name,
        "ssm_ping_status": "Online",
        "exact_name_history": history,
        "active_exact_name_job_ids": active_job_ids,
        "active_tagged_p5_instance_ids": active_p5_ids,
        "observed_at": observed_iso,
    }
    return body, observed_time, deadline


def build_controller_baseline(
    *,
    descriptor: Mapping[str, object],
    descriptor_key: str,
    descriptor_raw: bytes,
    qualification_submission_ready: Mapping[str, object],
    qualification_submission_ready_key: str,
    qualification_submission_ready_raw: bytes,
    intent: Mapping[str, object],
    intent_key: str,
    intent_raw: bytes,
    controller_instance_id: str,
    controller_instance_type: str,
    controller_profile_arn: str,
    controller_cluster_name: str,
    ssm_ping_status: str,
    exact_name_history: list[Mapping[str, object]],
    active_exact_name_job_ids: list[int],
    active_tagged_p5_instance_ids: list[str],
    observed_at: datetime | str,
) -> dict[str, object]:
    """Build a fresh, empty exact-name/controller-P5 baseline."""

    body, _, _ = _controller_baseline_body(
        descriptor=descriptor,
        descriptor_key=descriptor_key,
        descriptor_raw=descriptor_raw,
        qualification_submission_ready=qualification_submission_ready,
        qualification_submission_ready_key=(qualification_submission_ready_key),
        qualification_submission_ready_raw=(qualification_submission_ready_raw),
        intent=intent,
        intent_key=intent_key,
        intent_raw=intent_raw,
        controller_instance_id=controller_instance_id,
        controller_instance_type=controller_instance_type,
        controller_profile_arn=controller_profile_arn,
        controller_cluster_name=controller_cluster_name,
        ssm_ping_status=ssm_ping_status,
        exact_name_history=exact_name_history,
        active_exact_name_job_ids=active_exact_name_job_ids,
        active_tagged_p5_instance_ids=active_tagged_p5_instance_ids,
        observed_at=observed_at,
    )
    return {
        **body,
        CONTROLLER_BASELINE_DIGEST_FIELD: canonical_sha256(body),
    }


def _require_controller_baseline_record(value: object) -> dict[str, object]:
    return _require_self_hashed(
        value,
        fields=_BASELINE_FIELDS,
        digest_field=CONTROLLER_BASELINE_DIGEST_FIELD,
        schema_version=CONTROLLER_BASELINE_SCHEMA_VERSION,
        record_type=CONTROLLER_BASELINE_RECORD_TYPE,
        label="controller baseline",
    )


def validate_controller_baseline(
    value: object,
    *,
    descriptor: Mapping[str, object],
    descriptor_key: str,
    descriptor_raw: bytes,
    qualification_submission_ready: Mapping[str, object],
    qualification_submission_ready_key: str,
    qualification_submission_ready_raw: bytes,
    intent: Mapping[str, object],
    intent_key: str,
    intent_raw: bytes,
    controller_baseline_key: str,
    controller_baseline_raw: bytes,
    controller_instance_id: str,
    controller_instance_type: str,
    controller_profile_arn: str,
    controller_cluster_name: str,
    ssm_ping_status: str,
    exact_name_history: list[Mapping[str, object]],
    active_exact_name_job_ids: list[int],
    active_tagged_p5_instance_ids: list[str],
    observed_at: datetime | str,
    acquired_at: datetime | str,
) -> dict[str, object]:
    """Authenticate a controller/P5 baseline at acquisition time."""

    baseline = _require_controller_baseline_record(value)
    if baseline.get("ssm_ping_status") != "Online":
        raise SubmissionLiveAuthorityValidationError(
            "controller baseline ssm_ping_status must be Online"
        )
    _canonical_time(
        baseline.get("observed_at"),
        field="controller baseline observed_at",
        permit_datetime=False,
    )
    _require_actual_artifact(
        value=baseline,
        actual_key=controller_baseline_key,
        expected_key=controller_baseline_s3_key(baseline),
        raw=controller_baseline_raw,
        label="controller baseline",
    )
    expected, observed_at, deadline = _controller_baseline_body(
        descriptor=descriptor,
        descriptor_key=descriptor_key,
        descriptor_raw=descriptor_raw,
        qualification_submission_ready=qualification_submission_ready,
        qualification_submission_ready_key=(qualification_submission_ready_key),
        qualification_submission_ready_raw=(qualification_submission_ready_raw),
        intent=intent,
        intent_key=intent_key,
        intent_raw=intent_raw,
        controller_instance_id=controller_instance_id,
        controller_instance_type=controller_instance_type,
        controller_profile_arn=controller_profile_arn,
        controller_cluster_name=controller_cluster_name,
        ssm_ping_status=ssm_ping_status,
        exact_name_history=exact_name_history,
        active_exact_name_job_ids=active_exact_name_job_ids,
        active_tagged_p5_instance_ids=active_tagged_p5_instance_ids,
        observed_at=observed_at,
    )
    actual_body = dict(baseline)
    actual_body.pop(CONTROLLER_BASELINE_DIGEST_FIELD)
    if actual_body != expected:
        raise SubmissionLiveAuthorityValidationError(
            "controller baseline authority drift"
        )
    _, acquired_time = _canonical_time(
        acquired_at,
        field="acquired_at",
        permit_datetime=True,
    )
    if acquired_time < observed_at:
        raise SubmissionLiveAuthorityValidationError(
            "controller baseline observation is after acquisition"
        )
    if acquired_time >= deadline:
        raise SubmissionLiveAuthorityValidationError(
            "acquisition is at or after must_start_by"
        )
    if acquired_time - observed_at > MAX_LIVE_AUTHORITY_AGE:
        raise SubmissionLiveAuthorityValidationError(
            "controller baseline is older than 60 seconds at acquisition"
        )
    controller_baseline_s3_key(baseline)
    return baseline


def controller_baseline_s3_key(
    value: Mapping[str, object] | None = None,
    *,
    run_id: str | None = None,
    baseline_body_sha256: str | None = None,
) -> str:
    """Return the exact content-addressed controller-baseline key."""

    if value is not None:
        if run_id is not None or baseline_body_sha256 is not None:
            raise SubmissionLiveAuthorityValidationError(
                "controller baseline key arguments are mixed"
            )
        record = _require_controller_baseline_record(value)
        run_id = str(record["run_id"])
        baseline_body_sha256 = str(record[CONTROLLER_BASELINE_DIGEST_FIELD])
    if run_id is None or baseline_body_sha256 is None:
        raise SubmissionLiveAuthorityValidationError(
            "controller baseline key requires run_id and body SHA-256"
        )
    exact_run_id = _require_run_id(run_id)
    digest = _require_sha(
        baseline_body_sha256,
        field=CONTROLLER_BASELINE_DIGEST_FIELD,
    )
    return (
        f"campaigns/{exact_run_id}/qualification/controller-baselines/"
        f"{digest}/CONTROLLER_BASELINE.json"
    )


def controller_baseline_file_bytes(value: object) -> bytes:
    """Return authenticated canonical controller-baseline artifact bytes."""

    baseline = _require_controller_baseline_record(value)
    return canonical_file_bytes(baseline)


def _require_approved_arn(
    value: object,
    *,
    field: str,
    service: str,
    resource_prefix: str,
) -> str:
    if not isinstance(value, str):
        raise SubmissionLiveAuthorityValidationError(f"{field} is invalid")
    prefix = (
        f"arn:aws:{service}:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:{resource_prefix}"
    )
    if not value.startswith(prefix) or len(value) == len(prefix):
        raise SubmissionLiveAuthorityValidationError(f"{field} is foreign")
    if any(character.isspace() for character in value) or any(
        token in value for token in ("*", "?", "\\", "..", "//")
    ):
        raise SubmissionLiveAuthorityValidationError(f"{field} is invalid")
    return value


def _validate_deployment_identity(
    value: object,
) -> dict[str, object]:
    identity = _require_mapping(value, label="must-start deployment identity")
    _require_exact_fields(
        identity,
        fields=MUST_START_DEPLOYMENT_IDENTITY_FIELDS,
        label="must-start deployment identity",
    )
    for field in (
        "template_sha256",
        "lambda_code_sha256",
        "iam_policy_sha256",
    ):
        _require_sha(identity.get(field), field=field)
    _require_approved_arn(
        identity.get("stack_id"),
        field="stack_id",
        service="cloudformation",
        resource_prefix="stack/",
    )
    _require_approved_arn(
        identity.get("lambda_function_arn"),
        field="lambda_function_arn",
        service="lambda",
        resource_prefix="function:",
    )
    _require_approved_arn(
        identity.get("reconciliation_rule_arn"),
        field="reconciliation_rule_arn",
        service="events",
        resource_prefix="rule/",
    )
    _require_approved_arn(
        identity.get("deadline_schedule_arn"),
        field="deadline_schedule_arn",
        service="scheduler",
        resource_prefix="schedule/",
    )
    _require_approved_arn(
        identity.get("dlq_arn"),
        field="dlq_arn",
        service="sqs",
        resource_prefix="",
    )
    stack_resource = str(identity["stack_id"]).split(":stack/", 1)[1]
    stack_parts = stack_resource.split("/")
    if len(stack_parts) != 2 or any(
        _SAFE_NAME.fullmatch(part) is None for part in stack_parts
    ):
        raise SubmissionLiveAuthorityValidationError("stack_id is invalid")
    lambda_name = str(identity["lambda_function_arn"]).split(
        ":function:",
        1,
    )[1]
    if _SAFE_NAME.fullmatch(lambda_name) is None:
        raise SubmissionLiveAuthorityValidationError("lambda_function_arn is invalid")
    rule_name = str(identity["reconciliation_rule_arn"]).split(
        ":rule/",
        1,
    )[1]
    if _SAFE_NAME.fullmatch(rule_name) is None:
        raise SubmissionLiveAuthorityValidationError(
            "reconciliation_rule_arn is invalid"
        )
    schedule_resource = str(identity["deadline_schedule_arn"]).split(
        ":schedule/",
        1,
    )[1]
    schedule_parts = schedule_resource.split("/")
    if (
        len(schedule_parts) != 2
        or schedule_parts[0] != "default"
        or _SAFE_NAME.fullmatch(schedule_parts[1]) is None
    ):
        raise SubmissionLiveAuthorityValidationError("deadline_schedule_arn is invalid")
    queue_name = str(identity["dlq_arn"]).rsplit(":", 1)[1]
    if _SAFE_NAME.fullmatch(queue_name) is None:
        raise SubmissionLiveAuthorityValidationError("dlq_arn is invalid")
    return identity


def _must_start_control_plane_ready_body(
    *,
    descriptor: object,
    descriptor_key: object,
    descriptor_raw: object,
    qualification_submission_ready: object,
    qualification_submission_ready_key: object,
    qualification_submission_ready_raw: object,
    intent: object,
    intent_key: object,
    intent_raw: object,
    controller_baseline: object,
    controller_baseline_key: object,
    controller_baseline_raw: object,
    controller_instance_id: object,
    controller_instance_type: object,
    controller_profile_arn: object,
    controller_cluster_name: object,
    baseline_ssm_ping_status: object,
    baseline_exact_name_history: object,
    baseline_active_exact_name_job_ids: object,
    baseline_active_tagged_p5_instance_ids: object,
    baseline_observed_at: datetime | str | object,
    reviewed_deployment_identity: object,
    observed_deployment_identity: object,
    reconciliation_rule_state: object,
    deadline_schedule_state: object,
    coordinator_mode: object,
    observed_at: datetime | str | object,
) -> tuple[dict[str, object], datetime, datetime]:
    descriptor_value, _, intent_value = _authenticate_authority_chain(
        descriptor=descriptor,
        descriptor_key=descriptor_key,
        descriptor_raw=descriptor_raw,
        qualification_submission_ready=qualification_submission_ready,
        qualification_submission_ready_key=(qualification_submission_ready_key),
        qualification_submission_ready_raw=(qualification_submission_ready_raw),
        intent=intent,
        intent_key=intent_key,
        intent_raw=intent_raw,
    )
    observed_iso, observed_time = _canonical_time(
        observed_at,
        field="must-start control-plane observed_at",
        permit_datetime=True,
    )
    _, intent_at = _canonical_time(
        intent_value["intent_at"],
        field="submission intent intent_at",
        permit_datetime=False,
    )
    _, deadline = _canonical_time(
        intent_value["must_start_by"],
        field="submission intent must_start_by",
        permit_datetime=False,
    )
    if observed_time < intent_at:
        raise SubmissionLiveAuthorityValidationError(
            "must-start control-plane readiness predates submission intent"
        )
    if observed_time >= deadline:
        raise SubmissionLiveAuthorityValidationError(
            "must-start control-plane readiness was observed at or after must_start_by"
        )
    baseline = validate_controller_baseline(
        controller_baseline,
        descriptor=descriptor_value,
        descriptor_key=str(descriptor_key),
        descriptor_raw=descriptor_raw,  # type: ignore[arg-type]
        qualification_submission_ready=qualification_submission_ready,  # type: ignore[arg-type]
        qualification_submission_ready_key=str(qualification_submission_ready_key),
        qualification_submission_ready_raw=qualification_submission_ready_raw,  # type: ignore[arg-type]
        intent=intent_value,
        intent_key=str(intent_key),
        intent_raw=intent_raw,  # type: ignore[arg-type]
        controller_baseline_key=str(controller_baseline_key),
        controller_baseline_raw=controller_baseline_raw,  # type: ignore[arg-type]
        controller_instance_id=str(controller_instance_id),
        controller_instance_type=str(controller_instance_type),
        controller_profile_arn=str(controller_profile_arn),
        controller_cluster_name=str(controller_cluster_name),
        ssm_ping_status=str(baseline_ssm_ping_status),
        exact_name_history=baseline_exact_name_history,  # type: ignore[arg-type]
        active_exact_name_job_ids=baseline_active_exact_name_job_ids,  # type: ignore[arg-type]
        active_tagged_p5_instance_ids=baseline_active_tagged_p5_instance_ids,  # type: ignore[arg-type]
        observed_at=baseline_observed_at,  # type: ignore[arg-type]
        acquired_at=observed_iso,
    )
    reviewed_deployment = _validate_deployment_identity(reviewed_deployment_identity)
    observed_deployment = _validate_deployment_identity(observed_deployment_identity)
    if observed_deployment != reviewed_deployment:
        raise SubmissionLiveAuthorityValidationError(
            "observed deployment does not match reviewed deployment identity"
        )
    if reconciliation_rule_state != "ENABLED":
        raise SubmissionLiveAuthorityValidationError(
            "reconciliation_rule_state must be ENABLED"
        )
    if deadline_schedule_state != "ENABLED":
        raise SubmissionLiveAuthorityValidationError(
            "deadline_schedule_state must be ENABLED"
        )
    if coordinator_mode != COORDINATOR_MODE:
        raise SubmissionLiveAuthorityValidationError(
            f"coordinator_mode must be {COORDINATOR_MODE}"
        )
    baseline_observed_at = str(baseline["observed_at"])
    _, baseline_time = _canonical_time(
        baseline_observed_at,
        field="controller baseline observed_at",
        permit_datetime=False,
    )
    if observed_time < baseline_time:
        raise SubmissionLiveAuthorityValidationError(
            "must-start control-plane readiness predates controller baseline"
        )
    run_id = str(intent_value["run_id"])
    mode = str(intent_value["managed_mode"])
    intent_key = submission_intent_s3_key(
        run_id=run_id,
        managed_mode=mode,
        intent_body_sha256=str(intent_value[INTENT_DIGEST_FIELD]),
    )
    body = {
        "schema_version": MUST_START_CONTROL_PLANE_READY_SCHEMA_VERSION,
        "record_type": MUST_START_CONTROL_PLANE_READY_RECORD_TYPE,
        "account_id": intent_value["account_id"],
        "region": intent_value["region"],
        "bucket": _require_bucket(descriptor_value["bucket"]),
        "run_id": run_id,
        "managed_mode": mode,
        "descriptor_key": intent_value["descriptor_key"],
        "descriptor_file_sha256": intent_value["descriptor_file_sha256"],
        "descriptor_body_sha256": intent_value["descriptor_body_sha256"],
        "intent_key": intent_key,
        "intent_file_sha256": canonical_file_sha256(intent_value),
        "intent_body_sha256": intent_value[INTENT_DIGEST_FIELD],
        "controller_baseline_key": controller_baseline_s3_key(baseline),
        "controller_baseline_file_sha256": canonical_file_sha256(baseline),
        "controller_baseline_body_sha256": baseline[CONTROLLER_BASELINE_DIGEST_FIELD],
        **observed_deployment,
        "reconciliation_rule_state": "ENABLED",
        "deadline_schedule_state": "ENABLED",
        "coordinator_mode": COORDINATOR_MODE,
        "observed_at": observed_iso,
    }
    return body, observed_time, deadline


def build_must_start_control_plane_ready(
    *,
    descriptor: Mapping[str, object],
    descriptor_key: str,
    descriptor_raw: bytes,
    qualification_submission_ready: Mapping[str, object],
    qualification_submission_ready_key: str,
    qualification_submission_ready_raw: bytes,
    intent: Mapping[str, object],
    intent_key: str,
    intent_raw: bytes,
    controller_baseline: Mapping[str, object],
    controller_baseline_key: str,
    controller_baseline_raw: bytes,
    controller_instance_id: str,
    controller_instance_type: str,
    controller_profile_arn: str,
    controller_cluster_name: str,
    baseline_ssm_ping_status: str,
    baseline_exact_name_history: list[Mapping[str, object]],
    baseline_active_exact_name_job_ids: list[int],
    baseline_active_tagged_p5_instance_ids: list[str],
    baseline_observed_at: datetime | str,
    reviewed_deployment_identity: Mapping[str, object],
    observed_deployment_identity: Mapping[str, object],
    reconciliation_rule_state: str,
    deadline_schedule_state: str,
    coordinator_mode: str,
    observed_at: datetime | str,
) -> dict[str, object]:
    """Build an inspection receipt for the exact active must-start plane."""

    body, _, _ = _must_start_control_plane_ready_body(
        descriptor=descriptor,
        descriptor_key=descriptor_key,
        descriptor_raw=descriptor_raw,
        qualification_submission_ready=qualification_submission_ready,
        qualification_submission_ready_key=(qualification_submission_ready_key),
        qualification_submission_ready_raw=(qualification_submission_ready_raw),
        intent=intent,
        intent_key=intent_key,
        intent_raw=intent_raw,
        controller_baseline=controller_baseline,
        controller_baseline_key=controller_baseline_key,
        controller_baseline_raw=controller_baseline_raw,
        controller_instance_id=controller_instance_id,
        controller_instance_type=controller_instance_type,
        controller_profile_arn=controller_profile_arn,
        controller_cluster_name=controller_cluster_name,
        baseline_ssm_ping_status=baseline_ssm_ping_status,
        baseline_exact_name_history=baseline_exact_name_history,
        baseline_active_exact_name_job_ids=baseline_active_exact_name_job_ids,
        baseline_active_tagged_p5_instance_ids=(baseline_active_tagged_p5_instance_ids),
        baseline_observed_at=baseline_observed_at,
        reviewed_deployment_identity=reviewed_deployment_identity,
        observed_deployment_identity=observed_deployment_identity,
        reconciliation_rule_state=reconciliation_rule_state,
        deadline_schedule_state=deadline_schedule_state,
        coordinator_mode=coordinator_mode,
        observed_at=observed_at,
    )
    return {
        **body,
        MUST_START_CONTROL_PLANE_READY_DIGEST_FIELD: canonical_sha256(body),
    }


def _require_must_start_control_plane_ready_record(
    value: object,
) -> dict[str, object]:
    return _require_self_hashed(
        value,
        fields=_MUST_START_FIELDS,
        digest_field=MUST_START_CONTROL_PLANE_READY_DIGEST_FIELD,
        schema_version=MUST_START_CONTROL_PLANE_READY_SCHEMA_VERSION,
        record_type=MUST_START_CONTROL_PLANE_READY_RECORD_TYPE,
        label="must-start control-plane readiness",
    )


def validate_must_start_control_plane_ready(
    value: object,
    *,
    descriptor: Mapping[str, object],
    descriptor_key: str,
    descriptor_raw: bytes,
    qualification_submission_ready: Mapping[str, object],
    qualification_submission_ready_key: str,
    qualification_submission_ready_raw: bytes,
    intent: Mapping[str, object],
    intent_key: str,
    intent_raw: bytes,
    controller_baseline: Mapping[str, object],
    controller_baseline_key: str,
    controller_baseline_raw: bytes,
    must_start_control_plane_ready_key: str,
    must_start_control_plane_ready_raw: bytes,
    controller_instance_id: str,
    controller_instance_type: str,
    controller_profile_arn: str,
    controller_cluster_name: str,
    baseline_ssm_ping_status: str,
    baseline_exact_name_history: list[Mapping[str, object]],
    baseline_active_exact_name_job_ids: list[int],
    baseline_active_tagged_p5_instance_ids: list[str],
    baseline_observed_at: datetime | str,
    reviewed_deployment_identity: Mapping[str, object],
    observed_deployment_identity: Mapping[str, object],
    reconciliation_rule_state: str,
    deadline_schedule_state: str,
    coordinator_mode: str,
    observed_at: datetime | str,
    acquired_at: datetime | str,
) -> dict[str, object]:
    """Authenticate exact, fresh must-start deployment readiness."""

    ready = _require_must_start_control_plane_ready_record(value)
    _canonical_time(
        ready.get("observed_at"),
        field="must-start control-plane observed_at",
        permit_datetime=False,
    )
    _require_actual_artifact(
        value=ready,
        actual_key=must_start_control_plane_ready_key,
        expected_key=must_start_control_plane_ready_s3_key(ready),
        raw=must_start_control_plane_ready_raw,
        label="must-start control-plane readiness",
    )
    expected, observed_at, deadline = _must_start_control_plane_ready_body(
        descriptor=descriptor,
        descriptor_key=descriptor_key,
        descriptor_raw=descriptor_raw,
        qualification_submission_ready=qualification_submission_ready,
        qualification_submission_ready_key=(qualification_submission_ready_key),
        qualification_submission_ready_raw=(qualification_submission_ready_raw),
        intent=intent,
        intent_key=intent_key,
        intent_raw=intent_raw,
        controller_baseline=controller_baseline,
        controller_baseline_key=controller_baseline_key,
        controller_baseline_raw=controller_baseline_raw,
        controller_instance_id=controller_instance_id,
        controller_instance_type=controller_instance_type,
        controller_profile_arn=controller_profile_arn,
        controller_cluster_name=controller_cluster_name,
        baseline_ssm_ping_status=baseline_ssm_ping_status,
        baseline_exact_name_history=baseline_exact_name_history,
        baseline_active_exact_name_job_ids=baseline_active_exact_name_job_ids,
        baseline_active_tagged_p5_instance_ids=(baseline_active_tagged_p5_instance_ids),
        baseline_observed_at=baseline_observed_at,
        reviewed_deployment_identity=reviewed_deployment_identity,
        observed_deployment_identity=observed_deployment_identity,
        reconciliation_rule_state=reconciliation_rule_state,
        deadline_schedule_state=deadline_schedule_state,
        coordinator_mode=coordinator_mode,
        observed_at=observed_at,
    )
    actual_body = dict(ready)
    actual_body.pop(MUST_START_CONTROL_PLANE_READY_DIGEST_FIELD)
    if actual_body != expected:
        raise SubmissionLiveAuthorityValidationError(
            "must-start control-plane readiness authority drift"
        )
    _, acquired_time = _canonical_time(
        acquired_at,
        field="acquired_at",
        permit_datetime=True,
    )
    if acquired_time < observed_at:
        raise SubmissionLiveAuthorityValidationError(
            "must-start control-plane observation is after acquisition"
        )
    if acquired_time >= deadline:
        raise SubmissionLiveAuthorityValidationError(
            "acquisition is at or after must_start_by"
        )
    if acquired_time - observed_at > MAX_LIVE_AUTHORITY_AGE:
        raise SubmissionLiveAuthorityValidationError(
            "must-start control-plane readiness is older than 60 seconds at acquisition"
        )
    validate_controller_baseline(
        controller_baseline,
        descriptor=descriptor,
        descriptor_key=descriptor_key,
        descriptor_raw=descriptor_raw,
        qualification_submission_ready=qualification_submission_ready,
        qualification_submission_ready_key=(qualification_submission_ready_key),
        qualification_submission_ready_raw=(qualification_submission_ready_raw),
        intent=intent,
        intent_key=intent_key,
        intent_raw=intent_raw,
        controller_baseline_key=controller_baseline_key,
        controller_baseline_raw=controller_baseline_raw,
        controller_instance_id=controller_instance_id,
        controller_instance_type=controller_instance_type,
        controller_profile_arn=controller_profile_arn,
        controller_cluster_name=controller_cluster_name,
        ssm_ping_status=baseline_ssm_ping_status,
        exact_name_history=baseline_exact_name_history,
        active_exact_name_job_ids=baseline_active_exact_name_job_ids,
        active_tagged_p5_instance_ids=baseline_active_tagged_p5_instance_ids,
        observed_at=baseline_observed_at,
        acquired_at=acquired_at,
    )
    must_start_control_plane_ready_s3_key(ready)
    return ready


def must_start_control_plane_ready_s3_key(
    value: Mapping[str, object] | None = None,
    *,
    run_id: str | None = None,
    managed_mode: str | None = None,
    intent_body_sha256: str | None = None,
    control_plane_ready_body_sha256: str | None = None,
) -> str:
    """Return the exact intent-scoped, content-addressed readiness key."""

    if value is not None:
        if any(
            item is not None
            for item in (
                run_id,
                managed_mode,
                intent_body_sha256,
                control_plane_ready_body_sha256,
            )
        ):
            raise SubmissionLiveAuthorityValidationError(
                "must-start readiness key arguments are mixed"
            )
        record = _require_must_start_control_plane_ready_record(value)
        run_id = str(record["run_id"])
        managed_mode = str(record["managed_mode"])
        intent_body_sha256 = str(record["intent_body_sha256"])
        control_plane_ready_body_sha256 = str(
            record[MUST_START_CONTROL_PLANE_READY_DIGEST_FIELD]
        )
    if (
        run_id is None
        or managed_mode is None
        or intent_body_sha256 is None
        or control_plane_ready_body_sha256 is None
    ):
        raise SubmissionLiveAuthorityValidationError(
            "must-start readiness key requires run, mode, intent, and body identity"
        )
    exact_run_id = _require_run_id(run_id)
    exact_mode = _require_qualification_mode(managed_mode)
    intent_digest = _require_sha(
        intent_body_sha256,
        field="intent_body_sha256",
    )
    ready_digest = _require_sha(
        control_plane_ready_body_sha256,
        field=MUST_START_CONTROL_PLANE_READY_DIGEST_FIELD,
    )
    return (
        f"campaigns/{exact_run_id}/monitor/must-start/{exact_mode}/"
        f"{intent_digest}/control-plane-ready/{ready_digest}/"
        "CONTROL_PLANE_READY.json"
    )


def must_start_control_plane_ready_file_bytes(value: object) -> bytes:
    """Return authenticated canonical must-start readiness artifact bytes."""

    ready = _require_must_start_control_plane_ready_record(value)
    return canonical_file_bytes(ready)
