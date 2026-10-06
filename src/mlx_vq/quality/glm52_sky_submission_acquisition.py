"""Immutable singleton acquisition contract for a SkyPilot submission.

This module is deliberately pure.  It defines and validates the stable,
descriptor-addressed ``SUBMISSION_ACQUIRED.json`` selector, then classifies
durable selector inventories for a future conditional-put reconciler.  It does
not call S3, SkyPilot, AWS, or any other live control plane.

Only the caller that receives the successful ``PutObject(IfNoneMatch="*")``
response may launch.  Reading an identical stored selector is idempotent, but
it is reconciliation-only and never recreates launch authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal, cast

from mlx_vq.quality.glm52_qualification_submission_ready import (
    DIGEST_FIELD as READINESS_DIGEST_FIELD,
)
from mlx_vq.quality.glm52_qualification_submission_ready import (
    RECORD_TYPE as READINESS_RECORD_TYPE,
)
from mlx_vq.quality.glm52_qualification_submission_ready import (
    SCHEMA_VERSION as READINESS_SCHEMA_VERSION,
)
from mlx_vq.quality.glm52_qualification_submission_ready import (
    gpu_spend_snapshot_s3_key,
    qualification_submission_ready_s3_key,
    rehearsal_evidence_s3_key,
)
from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_ACCOUNT_ID,
    APPROVED_REGION,
)
from mlx_vq.quality.glm52_sky_submission_lifecycle import (
    EXPECTED_WORKSPACE,
    INTENT_DIGEST_FIELD,
    INTENT_RECORD_TYPE,
    INTENT_SCHEMA_VERSION,
    expected_sky_job_name,
    submission_intent_s3_key,
)

SubmissionAcquisitionState = Literal["absent", "winner", "conflict", "expired"]
SubmissionAcquisitionAction = Literal[
    "conditional-create",
    "reconcile-only",
    "fail-closed",
]
SubmissionAcquisitionReadState = Literal["exact-404", "found"]

ACQUISITION_RECORD_TYPE = "glm52_sky_submission_acquired_v1"
ACQUISITION_SCHEMA_VERSION = 1
ACQUISITION_DIGEST_FIELD = "acquisition_body_sha256"
CONTROLLER_BASELINE_RECORD_TYPE = "glm52_controller_baseline_v1"
CONTROLLER_BASELINE_SCHEMA_VERSION = 1
CONTROLLER_BASELINE_DIGEST_FIELD = "baseline_body_sha256"
CONTROL_PLANE_READY_RECORD_TYPE = "glm52_must_start_control_plane_ready_v1"
CONTROL_PLANE_READY_SCHEMA_VERSION = 1
CONTROL_PLANE_READY_DIGEST_FIELD = "control_plane_ready_body_sha256"
QUALIFICATION_MODE = "qualification"
ACQUISITION_KEY_TEMPLATE = (
    "campaigns/{run_id}/submissions/{managed_mode}/acquisitions/"
    "{descriptor_file_sha256}/SUBMISSION_ACQUIRED.json"
)

_MAX_BASELINE_AGE = timedelta(seconds=60)
_MODES = frozenset({"production", "qualification", "cache-seed"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUCKET = re.compile(
    r"^(?![0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$)"
    r"[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])?$"
)
_INSTANCE_ID = re.compile(r"^i-[0-9a-f]{17}$")
_INSTANCE_TYPE = re.compile(r"^[a-z0-9][a-z0-9.]*$")

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
    {
        "PENDING",
        "STARTING",
        "RUNNING",
        "RECOVERING",
        "CANCELLING",
    }
)
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
    {
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "repo_tar_sha256",
        "cache_seed_acceptance_file_sha256",
        "cache_seed_acceptance_body_sha256",
        "gpu_spend_snapshot_sha256",
        "gpu_spend_snapshot_body_sha256",
        "gpu_spend_ledger_tip_record_sha256",
        "qualification_submission_ready_sha256",
        "qualification_submission_ready_body_sha256",
        INTENT_DIGEST_FIELD,
    }
)
_INTENT_KEY_FIELDS = frozenset(
    {
        "descriptor_key",
        "cache_seed_acceptance_key",
        "gpu_spend_snapshot_key",
        "qualification_submission_ready_key",
    }
)

_READINESS_FIELDS = frozenset(
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
        READINESS_DIGEST_FIELD,
    }
)
_READINESS_SHA_FIELDS = frozenset(
    {
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "repo_tar_sha256",
        "seed_descriptor_file_sha256",
        "seed_descriptor_body_sha256",
        "seed_campaign_identity_sha256",
        "seed_repo_tar_sha256",
        "staged_readiness_file_sha256",
        "staged_readiness_body_sha256",
        "cache_seed_acceptance_file_sha256",
        "cache_seed_acceptance_body_sha256",
        "gpu_spend_snapshot_sha256",
        "gpu_spend_snapshot_body_sha256",
        "gpu_spend_ledger_tip_record_sha256",
        "rehearsal_evidence_sha256",
        "rehearsal_evidence_body_sha256",
        READINESS_DIGEST_FIELD,
    }
)
_READINESS_KEY_FIELDS = frozenset(
    {
        "descriptor_key",
        "seed_descriptor_key",
        "staged_readiness_key",
        "cache_seed_acceptance_key",
        "gpu_spend_snapshot_key",
        "rehearsal_evidence_key",
    }
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

_CONTROL_PLANE_BODY_FIELDS = frozenset(
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
_CONTROL_PLANE_FIELDS = _CONTROL_PLANE_BODY_FIELDS | {CONTROL_PLANE_READY_DIGEST_FIELD}

_ACQUISITION_BODY_FIELDS = frozenset(
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
        "qualification_submission_ready_key",
        "qualification_submission_ready_sha256",
        "qualification_submission_ready_body_sha256",
        "intent_key",
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_key",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "must_start_control_plane_ready_key",
        "must_start_control_plane_ready_file_sha256",
        "must_start_control_plane_ready_body_sha256",
        "sky_job_name",
        "must_start_by",
        "acquired_at",
    }
)
_ACQUISITION_FIELDS = _ACQUISITION_BODY_FIELDS | {ACQUISITION_DIGEST_FIELD}
_ACQUISITION_SHA_FIELDS = frozenset(
    {
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "qualification_submission_ready_sha256",
        "qualification_submission_ready_body_sha256",
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "must_start_control_plane_ready_file_sha256",
        "must_start_control_plane_ready_body_sha256",
        ACQUISITION_DIGEST_FIELD,
    }
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


class SubmissionAcquisitionValidationError(ValueError):
    """An acquisition selector or one of its pinned sources is invalid."""


@dataclass(frozen=True)
class ImmutableJsonArtifact:
    """One exact immutable JSON object's S3 key, raw bytes, and file digest."""

    key: str
    raw: bytes
    file_sha256: str


@dataclass(frozen=True)
class SubmissionAcquisitionReadResult:
    """Provenance supplied by a future exact GET plus complete prefix listing."""

    requested_key: str
    result: SubmissionAcquisitionReadState
    http_status: int
    artifact: ImmutableJsonArtifact | None
    listed_prefix: str
    prefix_keys: tuple[str, ...]
    listing_complete: bool
    observed_at: str


@dataclass(frozen=True)
class SubmissionAcquisitionDecision:
    """Pure durable-state classification; no result itself authorizes launch."""

    state: SubmissionAcquisitionState
    action: SubmissionAcquisitionAction
    reason: str
    selector: dict[str, object] | None = None


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SubmissionAcquisitionValidationError(
            "submission acquisition value is not canonical finite JSON"
        ) from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_file_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value) + b"\n").hexdigest()


def _decode_artifact(
    artifact: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(artifact, ImmutableJsonArtifact):
        raise SubmissionAcquisitionValidationError(
            f"{label} must carry exact immutable-artifact provenance"
        )
    _safe_key(artifact.key, field=f"{label} key")
    if type(artifact.raw) is not bytes:
        raise SubmissionAcquisitionValidationError(f"{label} raw bytes must be bytes")
    expected_file_sha256 = _require_sha(
        artifact.file_sha256,
        field=f"{label} file SHA-256",
    )
    if hashlib.sha256(artifact.raw).hexdigest() != expected_file_sha256:
        raise SubmissionAcquisitionValidationError(
            f"{label} file SHA-256 does not authenticate raw bytes"
        )
    try:
        value = json.loads(
            artifact.raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SubmissionAcquisitionValidationError(
            f"{label} raw bytes are not finite JSON"
        ) from error
    if not isinstance(value, dict):
        raise SubmissionAcquisitionValidationError(
            f"{label} raw bytes must contain an object"
        )
    if artifact.raw != _canonical_bytes(value) + b"\n":
        raise SubmissionAcquisitionValidationError(
            f"{label} raw bytes must be canonical compact JSON plus one newline"
        )
    return dict(value)


def _require_artifact_pin(
    artifact: ImmutableJsonArtifact,
    *,
    expected_key: str,
    expected_file_sha256: str,
    label: str,
) -> None:
    if artifact.key != expected_key:
        raise SubmissionAcquisitionValidationError(
            f"{label} key does not match exact immutable authority"
        )
    if artifact.file_sha256 != expected_file_sha256:
        raise SubmissionAcquisitionValidationError(
            f"{label} file SHA-256 does not match exact immutable authority"
        )


def _require_exact_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SubmissionAcquisitionValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise SubmissionAcquisitionValidationError(
            f"{label} schema mismatch: missing={missing}, unknown={unknown}"
        )
    result = dict(value)
    _canonical_bytes(result)
    return result


def _require_self_hashed(
    value: object,
    *,
    fields: frozenset[str],
    digest_field: str,
    schema_version: int,
    record_type: str,
    label: str,
) -> dict[str, object]:
    result = _require_exact_mapping(value, fields=fields, label=label)
    if (
        type(result.get("schema_version")) is not int
        or result.get("schema_version") != schema_version
        or result.get("record_type") != record_type
    ):
        raise SubmissionAcquisitionValidationError(f"{label} schema mismatch")
    body = dict(result)
    digest = _require_sha(body.pop(digest_field), field=digest_field)
    if digest != _canonical_sha256(body):
        raise SubmissionAcquisitionValidationError(f"{label} body SHA-256 mismatch")
    return result


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise SubmissionAcquisitionValidationError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _require_shas(
    value: Mapping[str, object],
    *,
    fields: frozenset[str],
) -> None:
    for field in fields:
        _require_sha(value.get(field), field=field)


def _require_run_id(value: object) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise SubmissionAcquisitionValidationError("run_id is invalid")
    return value


def _require_mode(value: object, *, qualification_only: bool) -> str:
    if not isinstance(value, str) or value not in _MODES:
        raise SubmissionAcquisitionValidationError("managed_mode is invalid")
    if qualification_only and value != QUALIFICATION_MODE:
        raise SubmissionAcquisitionValidationError(
            "submission acquisition requires qualification managed_mode"
        )
    return value


def _require_nonempty_string(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 for character in value)
    ):
        raise SubmissionAcquisitionValidationError(f"{field} is invalid")
    return value


def _require_integer(
    value: object,
    *,
    field: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise SubmissionAcquisitionValidationError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _require_float(
    value: object,
    *,
    field: str,
    minimum: float = 0.0,
) -> float:
    if type(value) is not float or not math.isfinite(value) or value < minimum:
        raise SubmissionAcquisitionValidationError(
            f"{field} must be a finite float >= {minimum}"
        )
    return value


def _require_money(value: object, *, field: str) -> Decimal:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SubmissionAcquisitionValidationError(
            f"{field} must be finite numeric money"
        )
    result = Decimal(str(value))
    if result < 0 or result != result.quantize(Decimal("0.01")):
        raise SubmissionAcquisitionValidationError(
            f"{field} must be non-negative two-decimal money"
        )
    return result


def _canonical_time(
    value: object,
    *,
    field: str,
    permit_datetime: bool,
) -> tuple[str, datetime]:
    if isinstance(value, datetime) and permit_datetime:
        if value.tzinfo is None:
            raise SubmissionAcquisitionValidationError(
                f"{field} must be timezone-aware"
            )
        parsed = value.astimezone(UTC)
        return parsed.isoformat().replace("+00:00", "Z"), parsed
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SubmissionAcquisitionValidationError(
            f"{field} must use canonical UTC Z form"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SubmissionAcquisitionValidationError(
            f"{field} must use canonical UTC Z form"
        ) from error
    canonical = parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if parsed.utcoffset() != timedelta(0) or canonical != value:
        raise SubmissionAcquisitionValidationError(
            f"{field} must use canonical UTC Z form"
        )
    return canonical, parsed.astimezone(UTC)


def _safe_key(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or "*" in value
        or "?" in value
        or "[" in value
        or "]" in value
        or any(character.isspace() for character in value)
        or "//" in value
    ):
        raise SubmissionAcquisitionValidationError(f"{field} is not a safe exact key")
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise SubmissionAcquisitionValidationError(f"{field} is not a safe exact key")
    return value


def _campaign_key(value: object, *, run_id: str, field: str) -> str:
    key = _safe_key(value, field=field)
    if not key.startswith(f"campaigns/{run_id}/"):
        raise SubmissionAcquisitionValidationError(
            f"{field} is not an exact campaign-scoped key"
        )
    return key


def _require_expected(value: object, expected: object, *, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise SubmissionAcquisitionValidationError(
            f"{field} does not match exact upstream authority"
        )


def _validate_identity(value: Mapping[str, object], *, label: str) -> tuple[str, str]:
    account_id = _require_nonempty_string(
        value.get("account_id"),
        field=f"{label} account_id",
    )
    region = _require_nonempty_string(
        value.get("region"),
        field=f"{label} region",
    )
    if account_id != APPROVED_ACCOUNT_ID or region != APPROVED_REGION:
        raise SubmissionAcquisitionValidationError(
            f"{label} is outside the approved account and region"
        )
    return account_id, region


def _validate_intent(value: object) -> tuple[dict[str, object], datetime, datetime]:
    intent = _require_self_hashed(
        value,
        fields=_INTENT_FIELDS,
        digest_field=INTENT_DIGEST_FIELD,
        schema_version=INTENT_SCHEMA_VERSION,
        record_type=INTENT_RECORD_TYPE,
        label="submission intent",
    )
    _require_shas(intent, fields=_INTENT_SHA_FIELDS)
    _validate_identity(intent, label="submission intent")
    run_id = _require_run_id(intent.get("run_id"))
    mode = _require_mode(intent.get("managed_mode"), qualification_only=True)
    for field in _INTENT_KEY_FIELDS:
        _campaign_key(intent.get(field), run_id=run_id, field=field)
    expected_name = expected_sky_job_name(run_id, mode)
    _require_expected(intent.get("sky_job_name"), expected_name, field="sky_job_name")
    _, intent_at = _canonical_time(
        intent.get("intent_at"),
        field="intent_at",
        permit_datetime=False,
    )
    _, deadline = _canonical_time(
        intent.get("must_start_by"),
        field="must_start_by",
        permit_datetime=False,
    )
    if intent_at >= deadline:
        raise SubmissionAcquisitionValidationError(
            "submission intent must precede must_start_by"
        )
    remaining = _require_integer(
        intent.get("remaining_gpu_seconds"),
        field="remaining_gpu_seconds",
        minimum=1,
    )
    allowance = _require_integer(
        intent.get("qualification_allowance_seconds"),
        field="qualification_allowance_seconds",
        minimum=1,
    )
    if allowance > remaining:
        raise SubmissionAcquisitionValidationError(
            "qualification allowance exceeds remaining GPU seconds"
        )
    _require_float(
        intent.get("remaining_gpu_cost_usd"),
        field="remaining_gpu_cost_usd",
    )
    _require_float(
        intent.get("qualification_allowance_cost_usd"),
        field="qualification_allowance_cost_usd",
    )
    if (
        _require_integer(
            intent.get("open_allocation_count"),
            field="open_allocation_count",
        )
        != 0
    ):
        raise SubmissionAcquisitionValidationError(
            "submission intent does not pin a closed spend snapshot"
        )
    expected_spend_key = gpu_spend_snapshot_s3_key(
        run_id=run_id,
        snapshot_body_sha256=str(intent["gpu_spend_snapshot_body_sha256"]),
    )
    _require_expected(
        intent.get("gpu_spend_snapshot_key"),
        expected_spend_key,
        field="gpu_spend_snapshot_key",
    )
    expected_readiness_key = (
        f"campaigns/{run_id}/qualification/submission-ready/"
        f"{intent['qualification_submission_ready_body_sha256']}/"
        "QUALIFICATION_SUBMISSION_READY.json"
    )
    _require_expected(
        intent.get("qualification_submission_ready_key"),
        expected_readiness_key,
        field="qualification_submission_ready_key",
    )
    submission_intent_s3_key(
        run_id=run_id,
        managed_mode=mode,
        intent_body_sha256=str(intent[INTENT_DIGEST_FIELD]),
    )
    return intent, intent_at, deadline


def _validate_readiness(
    value: object,
) -> tuple[dict[str, object], datetime, datetime]:
    readiness = _require_self_hashed(
        value,
        fields=_READINESS_FIELDS,
        digest_field=READINESS_DIGEST_FIELD,
        schema_version=READINESS_SCHEMA_VERSION,
        record_type=READINESS_RECORD_TYPE,
        label="qualification submission readiness",
    )
    _require_shas(readiness, fields=_READINESS_SHA_FIELDS)
    _validate_identity(readiness, label="qualification submission readiness")
    run_id = _require_run_id(readiness.get("run_id"))
    mode = _require_mode(readiness.get("managed_mode"), qualification_only=True)
    for field in _READINESS_KEY_FIELDS:
        _campaign_key(readiness.get(field), run_id=run_id, field=field)
    if not isinstance(readiness.get("seed_descriptor"), Mapping):
        raise SubmissionAcquisitionValidationError(
            "embedded seed descriptor must be an object"
        )
    _canonical_bytes(readiness["seed_descriptor"])
    expected_name = expected_sky_job_name(run_id, mode)
    _require_expected(
        readiness.get("sky_job_name"),
        expected_name,
        field="sky_job_name",
    )
    _require_nonempty_string(
        readiness.get("sky_task_name"),
        field="sky_task_name",
    )
    _, built_at = _canonical_time(
        readiness.get("built_at"),
        field="qualification readiness built_at",
        permit_datetime=False,
    )
    _, deadline = _canonical_time(
        readiness.get("must_start_by"),
        field="must_start_by",
        permit_datetime=False,
    )
    if built_at >= deadline:
        raise SubmissionAcquisitionValidationError(
            "qualification readiness must precede must_start_by"
        )
    remaining = _require_integer(
        readiness.get("remaining_gpu_seconds"),
        field="remaining_gpu_seconds",
        minimum=1,
    )
    allowance = _require_integer(
        readiness.get("qualification_allowance_seconds"),
        field="qualification_allowance_seconds",
        minimum=1,
    )
    allowed = _require_integer(
        readiness.get("allowed_gpu_seconds"),
        field="allowed_gpu_seconds",
        minimum=1,
    )
    if allowance > remaining or allowed != allowance:
        raise SubmissionAcquisitionValidationError(
            "qualification readiness GPU allowance is inconsistent"
        )
    remaining_cost = _require_money(
        readiness.get("remaining_gpu_cost_usd"),
        field="remaining_gpu_cost_usd",
    )
    allowance_cost = _require_money(
        readiness.get("qualification_allowance_cost_usd"),
        field="qualification_allowance_cost_usd",
    )
    allowed_cost = _require_money(
        readiness.get("allowed_gpu_cost_usd"),
        field="allowed_gpu_cost_usd",
    )
    if allowance_cost > remaining_cost or allowed_cost != allowance_cost:
        raise SubmissionAcquisitionValidationError(
            "qualification readiness cost allowance is inconsistent"
        )
    expected_spend_key = gpu_spend_snapshot_s3_key(
        run_id=run_id,
        snapshot_body_sha256=str(readiness["gpu_spend_snapshot_body_sha256"]),
    )
    _require_expected(
        readiness.get("gpu_spend_snapshot_key"),
        expected_spend_key,
        field="gpu_spend_snapshot_key",
    )
    expected_rehearsal_key = rehearsal_evidence_s3_key(
        run_id=run_id,
        rehearsal_body_sha256=str(readiness["rehearsal_evidence_body_sha256"]),
    )
    _require_expected(
        readiness.get("rehearsal_evidence_key"),
        expected_rehearsal_key,
        field="rehearsal_evidence_key",
    )
    qualification_submission_ready_s3_key(readiness)
    return readiness, built_at, deadline


def _cross_validate_readiness_and_intent(
    *,
    intent: Mapping[str, object],
    readiness: Mapping[str, object],
) -> None:
    for field in _INTENT_READINESS_SHARED_FIELDS:
        _require_expected(
            readiness.get(field),
            intent.get(field),
            field=f"readiness/intent {field}",
        )
    expected_key = qualification_submission_ready_s3_key(readiness)
    _require_expected(
        intent.get("qualification_submission_ready_key"),
        expected_key,
        field="qualification_submission_ready_key",
    )
    _require_expected(
        intent.get("qualification_submission_ready_body_sha256"),
        readiness.get(READINESS_DIGEST_FIELD),
        field="qualification submission readiness body SHA-256",
    )
    expected_file_sha = _canonical_file_sha256(readiness)
    _require_expected(
        intent.get("qualification_submission_ready_sha256"),
        expected_file_sha,
        field="qualification submission readiness file SHA-256",
    )


def _validate_controller_history(
    value: object,
    *,
    sky_job_name: str,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise SubmissionAcquisitionValidationError(
            "exact_name_history must be an exact list"
        )
    result: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(value):
        row = _require_exact_mapping(
            item,
            fields=_CONTROLLER_HISTORY_FIELDS,
            label=f"exact_name_history[{index}]",
        )
        job_id = _require_integer(
            row.get("sky_job_id"),
            field=f"exact_name_history[{index}].sky_job_id",
            minimum=1,
        )
        if job_id in seen_ids:
            raise SubmissionAcquisitionValidationError(
                "exact_name_history contains a duplicate sky_job_id"
            )
        seen_ids.add(job_id)
        _require_expected(
            row.get("sky_job_name"),
            sky_job_name,
            field=f"exact_name_history[{index}].sky_job_name",
        )
        _require_expected(
            row.get("workspace"),
            EXPECTED_WORKSPACE,
            field=f"exact_name_history[{index}].workspace",
        )
        _canonical_time(
            row.get("controller_submitted_at"),
            field=f"exact_name_history[{index}].controller_submitted_at",
            permit_datetime=False,
        )
        if row.get("controller_status") not in _CONTROLLER_STATUSES:
            raise SubmissionAcquisitionValidationError(
                f"exact_name_history[{index}].controller_status is invalid"
            )
        _require_nonempty_string(
            row.get("controller_identity"),
            field=f"exact_name_history[{index}].controller_identity",
        )
        result.append(row)
    expected = sorted(
        result,
        key=lambda row: (
            int(row["sky_job_id"]),
            str(row["controller_submitted_at"]),
            _canonical_bytes(row),
        ),
    )
    if result != expected:
        raise SubmissionAcquisitionValidationError(
            "exact_name_history is not in canonical sorted order"
        )
    return result


def _require_sorted_unique_ids(
    value: object,
    *,
    field: str,
    instance_ids: bool,
) -> list[object]:
    if not isinstance(value, list):
        raise SubmissionAcquisitionValidationError(f"{field} must be an exact list")
    normalized: list[object] = []
    for item in value:
        if instance_ids:
            if not isinstance(item, str) or _INSTANCE_ID.fullmatch(item) is None:
                raise SubmissionAcquisitionValidationError(
                    f"{field} contains an invalid instance ID"
                )
            normalized.append(item)
        else:
            normalized.append(_require_integer(item, field=field, minimum=1))
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise SubmissionAcquisitionValidationError(f"{field} must be sorted and unique")
    return normalized


def _controller_baseline_s3_key(value: Mapping[str, object]) -> str:
    run_id = _require_run_id(value.get("run_id"))
    body_sha256 = _require_sha(
        value.get(CONTROLLER_BASELINE_DIGEST_FIELD),
        field=CONTROLLER_BASELINE_DIGEST_FIELD,
    )
    return (
        f"campaigns/{run_id}/qualification/controller-baselines/"
        f"{body_sha256}/CONTROLLER_BASELINE.json"
    )


def _validate_controller_baseline(
    value: object,
    *,
    intent: Mapping[str, object],
) -> tuple[dict[str, object], datetime]:
    baseline = _require_self_hashed(
        value,
        fields=_BASELINE_FIELDS,
        digest_field=CONTROLLER_BASELINE_DIGEST_FIELD,
        schema_version=CONTROLLER_BASELINE_SCHEMA_VERSION,
        record_type=CONTROLLER_BASELINE_RECORD_TYPE,
        label="controller baseline",
    )
    _require_shas(
        baseline,
        fields=frozenset(
            {
                "intent_file_sha256",
                "intent_body_sha256",
                CONTROLLER_BASELINE_DIGEST_FIELD,
            }
        ),
    )
    _validate_identity(baseline, label="controller baseline")
    run_id = _require_run_id(baseline.get("run_id"))
    _require_mode(baseline.get("managed_mode"), qualification_only=True)
    _campaign_key(baseline.get("intent_key"), run_id=run_id, field="intent_key")
    _require_nonempty_string(baseline.get("bucket"), field="bucket")
    if _BUCKET.fullmatch(str(baseline["bucket"])) is None:
        raise SubmissionAcquisitionValidationError("bucket is invalid")
    _require_expected(
        baseline.get("workspace"),
        EXPECTED_WORKSPACE,
        field="workspace",
    )
    instance_id = baseline.get("controller_instance_id")
    if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
        raise SubmissionAcquisitionValidationError("controller_instance_id is invalid")
    instance_type = baseline.get("controller_instance_type")
    if (
        not isinstance(instance_type, str)
        or _INSTANCE_TYPE.fullmatch(instance_type) is None
    ):
        raise SubmissionAcquisitionValidationError(
            "controller_instance_type is invalid"
        )
    profile_arn = _require_nonempty_string(
        baseline.get("controller_profile_arn"),
        field="controller_profile_arn",
    )
    if not profile_arn.startswith(
        f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:instance-profile/"
    ):
        raise SubmissionAcquisitionValidationError("controller_profile_arn is foreign")
    _require_nonempty_string(
        baseline.get("controller_cluster_name"),
        field="controller_cluster_name",
    )
    _require_expected(
        baseline.get("ssm_ping_status"),
        "Online",
        field="ssm_ping_status",
    )
    history = _validate_controller_history(
        baseline.get("exact_name_history"),
        sky_job_name=str(intent["sky_job_name"]),
    )
    if any(row["controller_status"] in _ACTIVE_CONTROLLER_STATUSES for row in history):
        raise SubmissionAcquisitionValidationError(
            "controller baseline history contains an active exact-name job"
        )
    active_jobs = _require_sorted_unique_ids(
        baseline.get("active_exact_name_job_ids"),
        field="active_exact_name_job_ids",
        instance_ids=False,
    )
    if active_jobs:
        raise SubmissionAcquisitionValidationError(
            "controller baseline contains an active exact-name job"
        )
    active_p5 = _require_sorted_unique_ids(
        baseline.get("active_tagged_p5_instance_ids"),
        field="active_tagged_p5_instance_ids",
        instance_ids=True,
    )
    if active_p5:
        raise SubmissionAcquisitionValidationError(
            "controller baseline contains an active tagged P5 instance"
        )
    _, observed_at = _canonical_time(
        baseline.get("observed_at"),
        field="controller baseline observed_at",
        permit_datetime=False,
    )
    expected_intent_key = submission_intent_s3_key(
        run_id=str(intent["run_id"]),
        managed_mode=str(intent["managed_mode"]),
        intent_body_sha256=str(intent[INTENT_DIGEST_FIELD]),
    )
    for field, expected in (
        ("account_id", intent["account_id"]),
        ("region", intent["region"]),
        ("run_id", intent["run_id"]),
        ("managed_mode", intent["managed_mode"]),
        ("intent_key", expected_intent_key),
        ("intent_file_sha256", _canonical_file_sha256(intent)),
        ("intent_body_sha256", intent[INTENT_DIGEST_FIELD]),
        ("sky_job_name", intent["sky_job_name"]),
    ):
        _require_expected(
            baseline.get(field),
            expected,
            field=f"controller baseline {field}",
        )
    _controller_baseline_s3_key(baseline)
    return baseline, observed_at


def _control_plane_ready_s3_key(value: Mapping[str, object]) -> str:
    run_id = _require_run_id(value.get("run_id"))
    mode = _require_mode(value.get("managed_mode"), qualification_only=True)
    intent_body_sha256 = _require_sha(
        value.get("intent_body_sha256"),
        field="intent_body_sha256",
    )
    body_sha256 = _require_sha(
        value.get(CONTROL_PLANE_READY_DIGEST_FIELD),
        field=CONTROL_PLANE_READY_DIGEST_FIELD,
    )
    return (
        f"campaigns/{run_id}/monitor/must-start/{mode}/{intent_body_sha256}/"
        f"control-plane-ready/{body_sha256}/CONTROL_PLANE_READY.json"
    )


def _require_arn_prefix(value: object, *, field: str, prefix: str) -> str:
    result = _require_nonempty_string(value, field=field)
    if not result.startswith(prefix):
        raise SubmissionAcquisitionValidationError(f"{field} is foreign")
    return result


def _validate_control_plane_ready(
    value: object,
    *,
    intent: Mapping[str, object],
    baseline: Mapping[str, object],
) -> tuple[dict[str, object], datetime]:
    ready = _require_self_hashed(
        value,
        fields=_CONTROL_PLANE_FIELDS,
        digest_field=CONTROL_PLANE_READY_DIGEST_FIELD,
        schema_version=CONTROL_PLANE_READY_SCHEMA_VERSION,
        record_type=CONTROL_PLANE_READY_RECORD_TYPE,
        label="must-start control-plane readiness",
    )
    _require_shas(
        ready,
        fields=frozenset(
            {
                "descriptor_file_sha256",
                "descriptor_body_sha256",
                "intent_file_sha256",
                "intent_body_sha256",
                "controller_baseline_file_sha256",
                "controller_baseline_body_sha256",
                "template_sha256",
                "lambda_code_sha256",
                "iam_policy_sha256",
                CONTROL_PLANE_READY_DIGEST_FIELD,
            }
        ),
    )
    _validate_identity(ready, label="must-start control-plane readiness")
    run_id = _require_run_id(ready.get("run_id"))
    mode = _require_mode(ready.get("managed_mode"), qualification_only=True)
    for field in ("descriptor_key", "intent_key", "controller_baseline_key"):
        _campaign_key(ready.get(field), run_id=run_id, field=field)
    _require_nonempty_string(ready.get("bucket"), field="bucket")
    if _BUCKET.fullmatch(str(ready["bucket"])) is None:
        raise SubmissionAcquisitionValidationError("bucket is invalid")
    _require_arn_prefix(
        ready.get("stack_id"),
        field="stack_id",
        prefix=(
            f"arn:aws:cloudformation:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:stack/"
        ),
    )
    _require_arn_prefix(
        ready.get("lambda_function_arn"),
        field="lambda_function_arn",
        prefix=f"arn:aws:lambda:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:function:",
    )
    _require_arn_prefix(
        ready.get("reconciliation_rule_arn"),
        field="reconciliation_rule_arn",
        prefix=f"arn:aws:events:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:rule/",
    )
    _require_arn_prefix(
        ready.get("deadline_schedule_arn"),
        field="deadline_schedule_arn",
        prefix=f"arn:aws:scheduler:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:schedule/",
    )
    _require_arn_prefix(
        ready.get("dlq_arn"),
        field="dlq_arn",
        prefix=f"arn:aws:sqs:{APPROVED_REGION}:{APPROVED_ACCOUNT_ID}:",
    )
    _require_expected(
        ready.get("coordinator_mode"),
        "dynamic-job-binding-active",
        field="coordinator_mode",
    )
    for field in ("reconciliation_rule_state", "deadline_schedule_state"):
        if ready.get(field) != "ENABLED":
            raise SubmissionAcquisitionValidationError(f"{field} must be ENABLED")
    _, observed_at = _canonical_time(
        ready.get("observed_at"),
        field="must-start control-plane observed_at",
        permit_datetime=False,
    )
    expected_intent_key = submission_intent_s3_key(
        run_id=str(intent["run_id"]),
        managed_mode=str(intent["managed_mode"]),
        intent_body_sha256=str(intent[INTENT_DIGEST_FIELD]),
    )
    expected_baseline_key = _controller_baseline_s3_key(baseline)
    expected_values = {
        "account_id": intent["account_id"],
        "region": intent["region"],
        "run_id": intent["run_id"],
        "managed_mode": mode,
        "bucket": baseline["bucket"],
        "descriptor_key": intent["descriptor_key"],
        "descriptor_file_sha256": intent["descriptor_file_sha256"],
        "descriptor_body_sha256": intent["descriptor_body_sha256"],
        "intent_key": expected_intent_key,
        "intent_file_sha256": _canonical_file_sha256(intent),
        "intent_body_sha256": intent[INTENT_DIGEST_FIELD],
        "controller_baseline_key": expected_baseline_key,
        "controller_baseline_file_sha256": _canonical_file_sha256(baseline),
        "controller_baseline_body_sha256": baseline[CONTROLLER_BASELINE_DIGEST_FIELD],
    }
    for field, expected in expected_values.items():
        _require_expected(
            ready.get(field),
            expected,
            field=f"must-start control-plane readiness {field}",
        )
    _control_plane_ready_s3_key(ready)
    return ready, observed_at


def _acquisition_body(
    *,
    intent: object,
    qualification_submission_ready: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
    acquired_at: object,
) -> tuple[dict[str, object], datetime, datetime, datetime, datetime]:
    intent_value, intent_time, deadline = _validate_intent(intent)
    acquired_iso, acquired_time = _canonical_time(
        acquired_at,
        field="acquired_at",
        permit_datetime=True,
    )
    if acquired_time >= deadline:
        raise SubmissionAcquisitionValidationError(
            "acquired_at must be before must_start_by"
        )
    readiness, readiness_built_at, readiness_deadline = _validate_readiness(
        qualification_submission_ready
    )
    _cross_validate_readiness_and_intent(
        intent=intent_value,
        readiness=readiness,
    )
    if readiness_deadline != deadline or readiness_built_at > intent_time:
        raise SubmissionAcquisitionValidationError(
            "qualification readiness time authority drifts from submission intent"
        )
    baseline, baseline_observed_at = _validate_controller_baseline(
        controller_baseline,
        intent=intent_value,
    )
    control_plane, control_plane_observed_at = _validate_control_plane_ready(
        must_start_control_plane_ready,
        intent=intent_value,
        baseline=baseline,
    )
    if acquired_time < intent_time:
        raise SubmissionAcquisitionValidationError(
            "acquired_at precedes submission intent"
        )
    if baseline_observed_at < intent_time:
        raise SubmissionAcquisitionValidationError(
            "controller baseline predates submission intent"
        )
    if baseline_observed_at > acquired_time:
        raise SubmissionAcquisitionValidationError(
            "controller baseline follows acquired_at"
        )
    if acquired_time - baseline_observed_at > _MAX_BASELINE_AGE:
        raise SubmissionAcquisitionValidationError(
            "controller baseline is older than 60 seconds at acquisition"
        )
    if control_plane_observed_at < baseline_observed_at:
        raise SubmissionAcquisitionValidationError(
            "must-start control-plane readiness predates controller baseline"
        )
    if control_plane_observed_at > acquired_time:
        raise SubmissionAcquisitionValidationError(
            "must-start control-plane readiness follows acquired_at"
        )
    if acquired_time - control_plane_observed_at > _MAX_BASELINE_AGE:
        raise SubmissionAcquisitionValidationError(
            "must-start control-plane readiness is older than 60 seconds at acquisition"
        )

    run_id = str(intent_value["run_id"])
    mode = str(intent_value["managed_mode"])
    intent_key = submission_intent_s3_key(
        run_id=run_id,
        managed_mode=mode,
        intent_body_sha256=str(intent_value[INTENT_DIGEST_FIELD]),
    )
    return (
        {
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "record_type": ACQUISITION_RECORD_TYPE,
            "account_id": intent_value["account_id"],
            "region": intent_value["region"],
            "run_id": run_id,
            "managed_mode": mode,
            "descriptor_key": intent_value["descriptor_key"],
            "descriptor_file_sha256": intent_value["descriptor_file_sha256"],
            "descriptor_body_sha256": intent_value["descriptor_body_sha256"],
            "qualification_submission_ready_key": (
                qualification_submission_ready_s3_key(readiness)
            ),
            "qualification_submission_ready_sha256": (
                _canonical_file_sha256(readiness)
            ),
            "qualification_submission_ready_body_sha256": readiness[
                READINESS_DIGEST_FIELD
            ],
            "intent_key": intent_key,
            "intent_file_sha256": _canonical_file_sha256(intent_value),
            "intent_body_sha256": intent_value[INTENT_DIGEST_FIELD],
            "controller_baseline_key": _controller_baseline_s3_key(baseline),
            "controller_baseline_file_sha256": _canonical_file_sha256(baseline),
            "controller_baseline_body_sha256": baseline[
                CONTROLLER_BASELINE_DIGEST_FIELD
            ],
            "must_start_control_plane_ready_key": (
                _control_plane_ready_s3_key(control_plane)
            ),
            "must_start_control_plane_ready_file_sha256": (
                _canonical_file_sha256(control_plane)
            ),
            "must_start_control_plane_ready_body_sha256": control_plane[
                CONTROL_PLANE_READY_DIGEST_FIELD
            ],
            "sky_job_name": intent_value["sky_job_name"],
            "must_start_by": intent_value["must_start_by"],
            "acquired_at": acquired_iso,
        },
        acquired_time,
        deadline,
        baseline_observed_at,
        control_plane_observed_at,
    )


def submission_acquired_s3_key(
    *,
    run_id: str,
    managed_mode: str,
    descriptor_file_sha256: str,
) -> str:
    """Return the audit-approved stable descriptor-addressed selector key."""

    run = _require_run_id(run_id)
    mode = _require_mode(managed_mode, qualification_only=False)
    descriptor_digest = _require_sha(
        descriptor_file_sha256,
        field="descriptor_file_sha256",
    )
    return ACQUISITION_KEY_TEMPLATE.format(
        run_id=run,
        managed_mode=mode,
        descriptor_file_sha256=descriptor_digest,
    )


def submission_acquired_file_bytes(value: Mapping[str, object]) -> bytes:
    """Return the only accepted persisted selector bytes: canonical JSON + LF."""

    acquisition = _require_self_hashed(
        value,
        fields=_ACQUISITION_FIELDS,
        digest_field=ACQUISITION_DIGEST_FIELD,
        schema_version=ACQUISITION_SCHEMA_VERSION,
        record_type=ACQUISITION_RECORD_TYPE,
        label="submission acquisition",
    )
    _require_shas(acquisition, fields=_ACQUISITION_SHA_FIELDS)
    return _canonical_bytes(acquisition) + b"\n"


def submission_acquired_file_sha256(value: Mapping[str, object]) -> str:
    """Hash the exact canonical persisted selector bytes."""

    return hashlib.sha256(submission_acquired_file_bytes(value)).hexdigest()


def _source_records_from_artifacts(
    *,
    intent: object,
    qualification_submission_ready: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
) -> tuple[
    ImmutableJsonArtifact,
    ImmutableJsonArtifact,
    ImmutableJsonArtifact,
    ImmutableJsonArtifact,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    artifacts = (
        intent,
        qualification_submission_ready,
        controller_baseline,
        must_start_control_plane_ready,
    )
    labels = (
        "submission intent",
        "qualification submission readiness",
        "controller baseline",
        "must-start control-plane readiness",
    )
    if any(not isinstance(artifact, ImmutableJsonArtifact) for artifact in artifacts):
        raise SubmissionAcquisitionValidationError(
            "all acquisition sources must carry exact immutable-artifact provenance"
        )
    intent_artifact = cast(ImmutableJsonArtifact, intent)
    readiness_artifact = cast(
        ImmutableJsonArtifact,
        qualification_submission_ready,
    )
    baseline_artifact = cast(ImmutableJsonArtifact, controller_baseline)
    control_plane_artifact = cast(
        ImmutableJsonArtifact,
        must_start_control_plane_ready,
    )
    records = tuple(
        _decode_artifact(artifact, label=label)
        for artifact, label in zip(artifacts, labels, strict=True)
    )
    return (
        intent_artifact,
        readiness_artifact,
        baseline_artifact,
        control_plane_artifact,
        records[0],
        records[1],
        records[2],
        records[3],
    )


def _require_source_artifact_pins(
    *,
    body: Mapping[str, object],
    intent: ImmutableJsonArtifact,
    qualification_submission_ready: ImmutableJsonArtifact,
    controller_baseline: ImmutableJsonArtifact,
    must_start_control_plane_ready: ImmutableJsonArtifact,
) -> None:
    _require_artifact_pin(
        intent,
        expected_key=str(body["intent_key"]),
        expected_file_sha256=str(body["intent_file_sha256"]),
        label="submission intent",
    )
    _require_artifact_pin(
        qualification_submission_ready,
        expected_key=str(body["qualification_submission_ready_key"]),
        expected_file_sha256=str(body["qualification_submission_ready_sha256"]),
        label="qualification submission readiness",
    )
    _require_artifact_pin(
        controller_baseline,
        expected_key=str(body["controller_baseline_key"]),
        expected_file_sha256=str(body["controller_baseline_file_sha256"]),
        label="controller baseline",
    )
    _require_artifact_pin(
        must_start_control_plane_ready,
        expected_key=str(body["must_start_control_plane_ready_key"]),
        expected_file_sha256=str(body["must_start_control_plane_ready_file_sha256"]),
        label="must-start control-plane readiness",
    )


def build_submission_acquired(
    *,
    intent: ImmutableJsonArtifact,
    qualification_submission_ready: ImmutableJsonArtifact,
    controller_baseline: ImmutableJsonArtifact,
    must_start_control_plane_ready: ImmutableJsonArtifact,
    acquired_at: datetime | str,
) -> dict[str, object]:
    """Build one deterministic immutable acquisition-selector record."""

    (
        intent_artifact,
        readiness_artifact,
        baseline_artifact,
        control_plane_artifact,
        intent_value,
        readiness_value,
        baseline_value,
        control_plane_value,
    ) = _source_records_from_artifacts(
        intent=intent,
        qualification_submission_ready=qualification_submission_ready,
        controller_baseline=controller_baseline,
        must_start_control_plane_ready=must_start_control_plane_ready,
    )
    body, _, _, _, _ = _acquisition_body(
        intent=intent_value,
        qualification_submission_ready=readiness_value,
        controller_baseline=baseline_value,
        must_start_control_plane_ready=control_plane_value,
        acquired_at=acquired_at,
    )
    _require_source_artifact_pins(
        body=body,
        intent=intent_artifact,
        qualification_submission_ready=readiness_artifact,
        controller_baseline=baseline_artifact,
        must_start_control_plane_ready=control_plane_artifact,
    )
    result = {
        **body,
        ACQUISITION_DIGEST_FIELD: _canonical_sha256(body),
    }
    return _validate_submission_acquired_record(
        result,
        intent=intent_value,
        qualification_submission_ready=readiness_value,
        controller_baseline=baseline_value,
        must_start_control_plane_ready=control_plane_value,
        now=result["acquired_at"],
        reject_expired=True,
    )


def _validate_submission_acquired_record(
    value: object,
    *,
    intent: object,
    qualification_submission_ready: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
    now: object,
    reject_expired: bool,
) -> dict[str, object]:
    acquisition = _require_self_hashed(
        value,
        fields=_ACQUISITION_FIELDS,
        digest_field=ACQUISITION_DIGEST_FIELD,
        schema_version=ACQUISITION_SCHEMA_VERSION,
        record_type=ACQUISITION_RECORD_TYPE,
        label="submission acquisition",
    )
    _require_shas(acquisition, fields=_ACQUISITION_SHA_FIELDS)
    _validate_identity(acquisition, label="submission acquisition")
    run_id = _require_run_id(acquisition.get("run_id"))
    mode = _require_mode(acquisition.get("managed_mode"), qualification_only=True)
    for field in (
        "descriptor_key",
        "qualification_submission_ready_key",
        "intent_key",
        "controller_baseline_key",
        "must_start_control_plane_ready_key",
    ):
        _campaign_key(acquisition.get(field), run_id=run_id, field=field)
    expected, acquired_time, deadline, _, _ = _acquisition_body(
        intent=intent,
        qualification_submission_ready=qualification_submission_ready,
        controller_baseline=controller_baseline,
        must_start_control_plane_ready=must_start_control_plane_ready,
        acquired_at=acquisition["acquired_at"],
    )
    body = dict(acquisition)
    body.pop(ACQUISITION_DIGEST_FIELD)
    if body != expected:
        raise SubmissionAcquisitionValidationError(
            "submission acquisition authority drift"
        )
    submission_acquired_s3_key(
        run_id=run_id,
        managed_mode=mode,
        descriptor_file_sha256=str(acquisition["descriptor_file_sha256"]),
    )
    _, observed_now = _canonical_time(
        now,
        field="now",
        permit_datetime=True,
    )
    if acquired_time > observed_now:
        raise SubmissionAcquisitionValidationError(
            "submission acquisition acquired_at is in the future"
        )
    if reject_expired and observed_now >= deadline:
        raise SubmissionAcquisitionValidationError("submission acquisition is expired")
    return dict(acquisition)


def _validate_acquisition_artifact(
    acquisition: object,
    *,
    intent: object,
    qualification_submission_ready: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
    now: object,
    reject_expired: bool,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    datetime,
]:
    (
        intent_artifact,
        readiness_artifact,
        baseline_artifact,
        control_plane_artifact,
        intent_value,
        readiness_value,
        baseline_value,
        control_plane_value,
    ) = _source_records_from_artifacts(
        intent=intent,
        qualification_submission_ready=qualification_submission_ready,
        controller_baseline=controller_baseline,
        must_start_control_plane_ready=must_start_control_plane_ready,
    )
    acquisition_value = _decode_artifact(
        acquisition,
        label="submission acquisition",
    )
    validated = _validate_submission_acquired_record(
        acquisition_value,
        intent=intent_value,
        qualification_submission_ready=readiness_value,
        controller_baseline=baseline_value,
        must_start_control_plane_ready=control_plane_value,
        now=now,
        reject_expired=reject_expired,
    )
    _require_source_artifact_pins(
        body=validated,
        intent=intent_artifact,
        qualification_submission_ready=readiness_artifact,
        controller_baseline=baseline_artifact,
        must_start_control_plane_ready=control_plane_artifact,
    )
    acquisition_artifact = cast(ImmutableJsonArtifact, acquisition)
    expected_key = submission_acquired_s3_key(
        run_id=str(validated["run_id"]),
        managed_mode=str(validated["managed_mode"]),
        descriptor_file_sha256=str(validated["descriptor_file_sha256"]),
    )
    _require_artifact_pin(
        acquisition_artifact,
        expected_key=expected_key,
        expected_file_sha256=submission_acquired_file_sha256(validated),
        label="submission acquisition",
    )
    _, deadline = _canonical_time(
        validated["must_start_by"],
        field="must_start_by",
        permit_datetime=False,
    )
    return validated, baseline_value, control_plane_value, deadline


def validate_submission_acquired(
    value: ImmutableJsonArtifact,
    *,
    intent: ImmutableJsonArtifact,
    qualification_submission_ready: ImmutableJsonArtifact,
    controller_baseline: ImmutableJsonArtifact,
    must_start_control_plane_ready: ImmutableJsonArtifact,
    now: datetime | str,
) -> dict[str, object]:
    """Validate exact selector/source keys, LF bytes, file hashes, and deadline."""

    validated, _, _, _ = _validate_acquisition_artifact(
        value,
        intent=intent,
        qualification_submission_ready=qualification_submission_ready,
        controller_baseline=controller_baseline,
        must_start_control_plane_ready=must_start_control_plane_ready,
        now=now,
        reject_expired=True,
    )
    return validated


def _decision(
    *,
    state: SubmissionAcquisitionState,
    action: SubmissionAcquisitionAction,
    reason: str,
    selector: dict[str, object] | None = None,
) -> SubmissionAcquisitionDecision:
    return SubmissionAcquisitionDecision(
        state=state,
        action=action,
        reason=reason,
        selector=selector,
    )


def _validate_read_result(
    value: object,
    *,
    expected_key: str,
    now: object,
) -> ImmutableJsonArtifact | None:
    if not isinstance(value, SubmissionAcquisitionReadResult):
        raise SubmissionAcquisitionValidationError(
            "acquisition read provenance is not an exact read result"
        )
    _, observed_now = _canonical_time(
        now,
        field="now",
        permit_datetime=True,
    )
    observed_iso, observed_at = _canonical_time(
        value.observed_at,
        field="acquisition read observed_at",
        permit_datetime=False,
    )
    if observed_at != observed_now or observed_iso != value.observed_at:
        raise SubmissionAcquisitionValidationError(
            "acquisition read provenance is stale relative to decision time"
        )
    if value.requested_key != expected_key:
        raise SubmissionAcquisitionValidationError(
            "acquisition read requested the wrong exact selector key"
        )
    _safe_key(value.requested_key, field="acquisition read requested_key")
    expected_prefix = expected_key.rsplit("/", 1)[0] + "/"
    if value.listed_prefix != expected_prefix:
        raise SubmissionAcquisitionValidationError(
            "acquisition read listed the wrong exact selector prefix"
        )
    _safe_key(value.listed_prefix[:-1], field="acquisition read listed_prefix")
    if type(value.listing_complete) is not bool or not value.listing_complete:
        raise SubmissionAcquisitionValidationError(
            "acquisition read prefix listing is not complete"
        )
    if type(value.http_status) is not int:
        raise SubmissionAcquisitionValidationError(
            "acquisition read HTTP status is invalid"
        )
    if not isinstance(value.prefix_keys, tuple):
        raise SubmissionAcquisitionValidationError(
            "acquisition read prefix members must be an exact tuple"
        )
    for key in value.prefix_keys:
        _safe_key(key, field="acquisition read prefix member")
        if not key.startswith(expected_prefix):
            raise SubmissionAcquisitionValidationError(
                "acquisition read found an unexpected prefix member"
            )
    if value.prefix_keys != tuple(sorted(value.prefix_keys)) or len(
        set(value.prefix_keys)
    ) != len(value.prefix_keys):
        raise SubmissionAcquisitionValidationError(
            "acquisition read prefix has multiple, duplicate, or unsorted members"
        )
    if len(value.prefix_keys) > 1:
        raise SubmissionAcquisitionValidationError(
            "acquisition read found multiple selector prefix members"
        )
    if value.result == "exact-404":
        if value.http_status != 404 or value.artifact is not None or value.prefix_keys:
            raise SubmissionAcquisitionValidationError(
                "acquisition read does not prove an exact 404 and empty prefix"
            )
        return None
    if value.result == "found":
        if (
            value.http_status != 200
            or not isinstance(value.artifact, ImmutableJsonArtifact)
            or value.prefix_keys != (expected_key,)
            or value.artifact.key != expected_key
        ):
            raise SubmissionAcquisitionValidationError(
                "acquisition read does not prove one exact stored selector"
            )
        return value.artifact
    raise SubmissionAcquisitionValidationError(
        "acquisition read result state is invalid"
    )


def _selected_sources(
    *,
    intent: ImmutableJsonArtifact,
    qualification_submission_ready: ImmutableJsonArtifact,
    controller_baseline: ImmutableJsonArtifact,
    must_start_control_plane_ready: ImmutableJsonArtifact,
    selected_intent: ImmutableJsonArtifact | None,
    selected_qualification_submission_ready: ImmutableJsonArtifact | None,
    selected_controller_baseline: ImmutableJsonArtifact | None,
    selected_must_start_control_plane_ready: ImmutableJsonArtifact | None,
) -> tuple[
    ImmutableJsonArtifact,
    ImmutableJsonArtifact,
    ImmutableJsonArtifact,
    ImmutableJsonArtifact,
]:
    selected = (
        selected_intent,
        selected_qualification_submission_ready,
        selected_controller_baseline,
        selected_must_start_control_plane_ready,
    )
    if all(value is None for value in selected):
        return (
            intent,
            qualification_submission_ready,
            controller_baseline,
            must_start_control_plane_ready,
        )
    if any(value is None for value in selected):
        raise SubmissionAcquisitionValidationError(
            "selected winner sources must be supplied together"
        )
    return (
        cast(ImmutableJsonArtifact, selected_intent),
        cast(
            ImmutableJsonArtifact,
            selected_qualification_submission_ready,
        ),
        cast(ImmutableJsonArtifact, selected_controller_baseline),
        cast(
            ImmutableJsonArtifact,
            selected_must_start_control_plane_ready,
        ),
    )


def _require_put_freshness(
    *,
    baseline: Mapping[str, object],
    control_plane_ready: Mapping[str, object],
    now: object,
) -> None:
    _, observed_now = _canonical_time(
        now,
        field="now",
        permit_datetime=True,
    )
    for value, field in (
        (baseline.get("observed_at"), "controller baseline"),
        (control_plane_ready.get("observed_at"), "must-start control-plane readiness"),
    ):
        _, observed_at = _canonical_time(
            value,
            field=f"{field} observed_at",
            permit_datetime=False,
        )
        if observed_at > observed_now or observed_now - observed_at > _MAX_BASELINE_AGE:
            raise SubmissionAcquisitionValidationError(
                f"{field} is not fresh at conditional-put decision time"
            )


def decide_submission_acquisition(
    *,
    candidate: ImmutableJsonArtifact,
    read_result: SubmissionAcquisitionReadResult,
    intent: ImmutableJsonArtifact,
    qualification_submission_ready: ImmutableJsonArtifact,
    controller_baseline: ImmutableJsonArtifact,
    must_start_control_plane_ready: ImmutableJsonArtifact,
    selected_intent: ImmutableJsonArtifact | None = None,
    selected_qualification_submission_ready: ImmutableJsonArtifact | None = None,
    selected_controller_baseline: ImmutableJsonArtifact | None = None,
    selected_must_start_control_plane_ready: ImmutableJsonArtifact | None = None,
    now: datetime | str,
) -> SubmissionAcquisitionDecision:
    """Classify absent, stored-winner, conflict, and expired selector states.

    ``conditional-create`` means only that a complete, current exact-404 read
    permits a future I/O layer to attempt the stable conditional put.
    ``reconcile-only`` means one independently authenticated winner is already
    durable.  Neither result proves a successful put response or authorizes a
    SkyPilot launch.
    """

    try:
        (
            candidate_value,
            candidate_baseline,
            candidate_control_plane,
            candidate_deadline,
        ) = _validate_acquisition_artifact(
            candidate,
            intent=intent,
            qualification_submission_ready=qualification_submission_ready,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            now=now,
            reject_expired=False,
        )
        _, observed_now = _canonical_time(
            now,
            field="now",
            permit_datetime=True,
        )
        expected_key = submission_acquired_s3_key(
            run_id=str(candidate_value["run_id"]),
            managed_mode=str(candidate_value["managed_mode"]),
            descriptor_file_sha256=str(candidate_value["descriptor_file_sha256"]),
        )
    except (SubmissionAcquisitionValidationError, TypeError, ValueError) as error:
        return _decision(
            state="conflict",
            action="fail-closed",
            reason=f"candidate acquisition is invalid: {error}",
        )

    try:
        winner_artifact = _validate_read_result(
            read_result,
            expected_key=expected_key,
            now=now,
        )
    except (SubmissionAcquisitionValidationError, TypeError, ValueError) as error:
        return _decision(
            state="conflict",
            action="fail-closed",
            reason=f"acquisition read provenance is invalid: {error}",
        )
    if winner_artifact is None:
        if observed_now >= candidate_deadline:
            return _decision(
                state="expired",
                action="fail-closed",
                reason="submission acquisition deadline has expired",
            )
        try:
            _require_put_freshness(
                baseline=candidate_baseline,
                control_plane_ready=candidate_control_plane,
                now=now,
            )
        except SubmissionAcquisitionValidationError as error:
            return _decision(
                state="conflict",
                action="fail-closed",
                reason=f"conditional-put live authority is not fresh: {error}",
            )
        return _decision(
            state="absent",
            action="conditional-create",
            reason=(
                "an exact 404 and complete empty prefix are current; a future "
                "I/O layer may attempt one conditional create, but no launch "
                "is yet authorized"
            ),
        )

    try:
        (
            winner_intent,
            winner_readiness,
            winner_baseline,
            winner_control_plane,
        ) = _selected_sources(
            intent=intent,
            qualification_submission_ready=qualification_submission_ready,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            selected_intent=selected_intent,
            selected_qualification_submission_ready=(
                selected_qualification_submission_ready
            ),
            selected_controller_baseline=selected_controller_baseline,
            selected_must_start_control_plane_ready=(
                selected_must_start_control_plane_ready
            ),
        )
        winner, _, _, winner_deadline = _validate_acquisition_artifact(
            winner_artifact,
            intent=winner_intent,
            qualification_submission_ready=winner_readiness,
            controller_baseline=winner_baseline,
            must_start_control_plane_ready=winner_control_plane,
            now=now,
            reject_expired=False,
        )
    except (SubmissionAcquisitionValidationError, TypeError, ValueError) as error:
        return _decision(
            state="conflict",
            action="fail-closed",
            reason=f"stored acquisition selector conflicts with authority: {error}",
        )
    if observed_now >= winner_deadline:
        return _decision(
            state="expired",
            action="fail-closed",
            reason="stored submission acquisition deadline has expired",
        )
    return _decision(
        state="winner",
        action="reconcile-only",
        reason=(
            "one independently authenticated selector is already durable; "
            "retry is reconciliation-only and must never launch"
        ),
        selector=winner,
    )


__all__ = [
    "ACQUISITION_DIGEST_FIELD",
    "ACQUISITION_KEY_TEMPLATE",
    "ACQUISITION_RECORD_TYPE",
    "ACQUISITION_SCHEMA_VERSION",
    "ImmutableJsonArtifact",
    "SubmissionAcquisitionAction",
    "SubmissionAcquisitionDecision",
    "SubmissionAcquisitionReadResult",
    "SubmissionAcquisitionReadState",
    "SubmissionAcquisitionState",
    "SubmissionAcquisitionValidationError",
    "build_submission_acquired",
    "decide_submission_acquisition",
    "submission_acquired_file_bytes",
    "submission_acquired_file_sha256",
    "submission_acquired_s3_key",
    "validate_submission_acquired",
]
