"""Pure fail-closed binding of a dynamic-v2 SkyPilot numeric job ID.

The caller supplies already-read artifacts.  This module authenticates their
canonical self-hashed bodies and cross-identity relationships; callers that
read from storage must authenticate keys, bytes, and transport provenance
before passing them here.  It performs no AWS, controller, or filesystem I/O.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

DynamicJobBindingAction = Literal[
    "wait-for-acquisition",
    "wait-for-one-new-job",
    "bind-exact-job",
    "reconcile-bound-job",
    "fail-closed",
]

_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUCKET = re.compile(
    r"^(?![0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$)"
    r"[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])?$"
)
_INSTANCE_ID = re.compile(r"^i-[0-9a-f]{17}$")
_INSTANCE_TYPE = re.compile(r"^[a-z0-9][a-z0-9.]*$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONTROLLER_ROLE_ARN = re.compile(
    r"^arn:aws:iam::246813579024:role/[A-Za-z0-9+=,.@_/-]+$"
)
_EXPECTED_WORKSPACE = "default"
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
_CONTROLLER_STATUS_PROGRESSIONS = {
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
_ROW_FIELDS = frozenset(
    {
        "sky_job_id",
        "sky_job_name",
        "workspace",
        "controller_submitted_at",
        "controller_status",
        "controller_identity",
    }
)
_INTENT_FIELDS = frozenset(
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
        "intent_body_sha256",
    }
)
_BASELINE_FIELDS = frozenset(
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
        "baseline_body_sha256",
    }
)
_ACQUISITION_FIELDS = frozenset(
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
        "acquisition_body_sha256",
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
_ACQUISITION_KEY_FIELDS = frozenset(
    {
        "descriptor_key",
        "qualification_submission_ready_key",
        "intent_key",
        "controller_baseline_key",
        "must_start_control_plane_ready_key",
    }
)
_BINDING_FIELDS = frozenset(
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
        "intent_key",
        "intent_body_sha256",
        "controller_baseline_body_sha256",
        "acquisition_body_sha256",
        "sky_job_name",
        "must_start_by",
        "workspace",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
        "sky_job_id",
        "controller_submitted_at",
        "controller_status",
        "controller_identity",
        "bound_at",
        "job_binding_body_sha256",
    }
)


class DynamicJobBindingValidationError(ValueError):
    """A dynamic-v2 binding authority or observation is malformed or foreign."""


def dynamic_v2_canonical_bytes(value: object) -> bytes:
    """Encode one dynamic-v2 hash body as canonical no-newline ASCII JSON."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise DynamicJobBindingValidationError(
            "authority is not canonical JSON"
        ) from error


@dataclass(frozen=True)
class DynamicJobBindingDecision:
    """The only pure decision available to a later I/O coordinator."""

    action: DynamicJobBindingAction
    reason: str
    sky_job_id: int | None = None
    job_row: dict[str, object] | None = None


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(dynamic_v2_canonical_bytes(value)).hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DynamicJobBindingValidationError(f"{label} must be a mapping")
    return dict(value)


def _exact_record(
    value: object,
    *,
    fields: frozenset[str],
    digest_field: str,
    version: int,
    record_type: str,
    label: str,
) -> dict[str, object]:
    record = _mapping(value, label=label)
    if set(record) != fields or type(record.get("schema_version")) is not int:
        raise DynamicJobBindingValidationError(f"{label} schema mismatch")
    if record["schema_version"] != version or record.get("record_type") != record_type:
        raise DynamicJobBindingValidationError(f"{label} schema mismatch")
    digest = record.get(digest_field)
    body = dict(record)
    body.pop(digest_field)
    if not isinstance(digest, str) or digest != _canonical_sha256(body):
        raise DynamicJobBindingValidationError(f"{label} body SHA-256 mismatch")
    return record


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise DynamicJobBindingValidationError(f"{label} is invalid")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise DynamicJobBindingValidationError(f"{label} is invalid")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise DynamicJobBindingValidationError(f"{label} is invalid")
    return value


def _nonnegative_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise DynamicJobBindingValidationError(f"{label} is invalid")
    return value


def _time(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise DynamicJobBindingValidationError(f"{label} must be canonical ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError) as error:
        raise DynamicJobBindingValidationError(
            f"{label} must be canonical ISO-8601"
        ) from error
    if parsed.tzinfo is None:
        raise DynamicJobBindingValidationError(f"{label} must be timezone-aware")
    try:
        normalized = parsed.astimezone(UTC)
        canonical = normalized.isoformat().replace("+00:00", "Z")
    except (ValueError, OverflowError) as error:
        raise DynamicJobBindingValidationError(
            f"{label} cannot be normalized to UTC"
        ) from error
    if value != canonical:
        raise DynamicJobBindingValidationError(f"{label} is not canonical")
    return normalized


def _expected_name(run_id: str, mode: str) -> str:
    if _RUN_ID.fullmatch(run_id) is None or mode != "qualification":
        raise DynamicJobBindingValidationError("run ID or managed mode is invalid")
    return f"{run_id}-qualification"


def _intent_key(intent: Mapping[str, object]) -> str:
    return (
        f"campaigns/{intent['run_id']}/submissions/{intent['managed_mode']}/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )


def _canonical_file_sha256(value: object) -> str:
    return hashlib.sha256(dynamic_v2_canonical_bytes(value) + b"\n").hexdigest()


def _campaign_key(value: object, *, run_id: str, label: str) -> str:
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
        or any(segment in {"", ".", ".."} for segment in value.split("/"))
        or not value.startswith(f"campaigns/{run_id}/")
    ):
        raise DynamicJobBindingValidationError(
            f"{label} is not a safe exact campaign-scoped key"
        )
    return value


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise DynamicJobBindingValidationError(
            f"{label} does not match exact upstream authority"
        )


def _controller_baseline_key(baseline: Mapping[str, object]) -> str:
    return (
        f"campaigns/{baseline['run_id']}/qualification/controller-baselines/"
        f"{baseline['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
    )


def _control_plane_ready_key(
    acquisition: Mapping[str, object],
    *,
    intent: Mapping[str, object],
) -> str:
    return (
        f"campaigns/{intent['run_id']}/monitor/must-start/"
        f"{intent['managed_mode']}/{intent['intent_body_sha256']}/"
        "control-plane-ready/"
        f"{acquisition['must_start_control_plane_ready_body_sha256']}/"
        "CONTROL_PLANE_READY.json"
    )


def _row(
    value: object,
    *,
    intent: Mapping[str, object],
    baseline: Mapping[str, object],
    descriptor_controller_identity: str,
    label: str,
) -> dict[str, object]:
    row = _mapping(value, label=label)
    if set(row) != _ROW_FIELDS:
        raise DynamicJobBindingValidationError(f"{label} schema mismatch")
    _positive_int(row.get("sky_job_id"), label=f"{label}.sky_job_id")
    if row.get("sky_job_name") != intent["sky_job_name"]:
        raise DynamicJobBindingValidationError(f"{label}.sky_job_name is foreign")
    if row.get("workspace") != baseline["workspace"]:
        raise DynamicJobBindingValidationError(f"{label}.workspace is foreign")
    if row.get("controller_identity") != descriptor_controller_identity:
        raise DynamicJobBindingValidationError(
            f"{label}.controller_identity is foreign"
        )
    _time(row.get("controller_submitted_at"), label=f"{label}.controller_submitted_at")
    if row.get("controller_status") not in _CONTROLLER_STATUSES:
        raise DynamicJobBindingValidationError(f"{label}.controller_status is invalid")
    return row


def _history(
    value: object,
    *,
    intent: Mapping[str, object],
    baseline: Mapping[str, object],
    descriptor_controller_identity: str,
    label: str,
) -> list[dict[str, object]]:
    if type(value) is not list:
        raise DynamicJobBindingValidationError(f"{label} must be a list")
    rows = [
        _row(
            item,
            intent=intent,
            baseline=baseline,
            descriptor_controller_identity=descriptor_controller_identity,
            label=f"{label}[{index}]",
        )
        for index, item in enumerate(value)
    ]
    if len({int(row["sky_job_id"]) for row in rows}) != len(rows):
        raise DynamicJobBindingValidationError(f"{label} has duplicate job IDs")
    expected = sorted(
        rows,
        key=lambda row: (
            int(row["sky_job_id"]),
            str(row["controller_submitted_at"]),
            _canonical_sha256(row),
        ),
    )
    if rows != expected:
        raise DynamicJobBindingValidationError(f"{label} is reordered")
    return rows


def _intent(value: object) -> tuple[dict[str, object], datetime, datetime]:
    intent = _exact_record(
        value,
        fields=_INTENT_FIELDS,
        digest_field="intent_body_sha256",
        version=2,
        record_type="glm52_sky_submission_intent_v2",
        label="submission intent",
    )
    run_id, mode = intent.get("run_id"), intent.get("managed_mode")
    if (
        not isinstance(run_id, str)
        or not isinstance(mode, str)
        or intent.get("account_id") != "246813579024"
        or intent.get("region") != "us-west-2"
    ):
        raise DynamicJobBindingValidationError("submission intent identity is foreign")
    if intent.get("sky_job_name") != _expected_name(run_id, mode):
        raise DynamicJobBindingValidationError(
            "submission intent Sky job name is foreign"
        )
    for field in (name for name in intent if name.endswith("_sha256")):
        _sha(intent[field], label=f"submission intent {field}")
    for field in _INTENT_KEY_FIELDS:
        _campaign_key(
            intent.get(field),
            run_id=run_id,
            label=f"submission intent {field}",
        )
    descriptor_prefix = f"campaigns/{run_id}/submissions/"
    if not str(intent["descriptor_key"]).startswith(descriptor_prefix):
        raise DynamicJobBindingValidationError(
            "submission intent descriptor_key is not submission-scoped"
        )
    _require_exact(
        intent.get("cache_seed_acceptance_key"),
        (
            f"campaigns/{run_id}/qualification-cache-seed/accepted/"
            f"{intent['cache_seed_acceptance_body_sha256']}/"
            "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
        ),
        label="submission intent cache_seed_acceptance_key",
    )
    _require_exact(
        intent.get("gpu_spend_snapshot_key"),
        (
            f"campaigns/{run_id}/spend-snapshots/"
            f"{intent['gpu_spend_snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
        ),
        label="submission intent gpu_spend_snapshot_key",
    )
    _require_exact(
        intent.get("qualification_submission_ready_key"),
        (
            f"campaigns/{run_id}/qualification/submission-ready/"
            f"{intent['qualification_submission_ready_body_sha256']}/"
            "QUALIFICATION_SUBMISSION_READY.json"
        ),
        label="submission intent qualification_submission_ready_key",
    )
    intent_at = _time(intent.get("intent_at"), label="submission intent intent_at")
    deadline = _time(
        intent.get("must_start_by"), label="submission intent must_start_by"
    )
    if intent_at >= deadline:
        raise DynamicJobBindingValidationError(
            "submission intent deadline is inconsistent"
        )
    remaining_seconds = _positive_int(
        intent.get("remaining_gpu_seconds"),
        label="submission intent remaining_gpu_seconds",
    )
    allowance_seconds = _positive_int(
        intent.get("qualification_allowance_seconds"),
        label="submission intent qualification_allowance_seconds",
    )
    if allowance_seconds > remaining_seconds:
        raise DynamicJobBindingValidationError(
            "submission intent qualification allowance exceeds remaining seconds"
        )
    _nonnegative_float(
        intent.get("remaining_gpu_cost_usd"),
        label="submission intent remaining_gpu_cost_usd",
    )
    _nonnegative_float(
        intent.get("qualification_allowance_cost_usd"),
        label="submission intent qualification_allowance_cost_usd",
    )
    if (
        _nonnegative_int(
            intent.get("open_allocation_count"),
            label="submission intent open_allocation_count",
        )
        != 0
    ):
        raise DynamicJobBindingValidationError(
            "submission intent spend snapshot has open allocations"
        )
    return intent, intent_at, deadline


def _baseline(
    value: object,
    *,
    intent: Mapping[str, object],
    descriptor_controller_identity: str,
) -> tuple[dict[str, object], list[dict[str, object]], datetime]:
    baseline = _exact_record(
        value,
        fields=_BASELINE_FIELDS,
        digest_field="baseline_body_sha256",
        version=1,
        record_type="glm52_controller_baseline_v1",
        label="controller baseline",
    )
    for field in (
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "intent_body_sha256",
        "sky_job_name",
    ):
        if baseline.get(field) != intent.get(field):
            raise DynamicJobBindingValidationError(
                "controller baseline identity is foreign"
            )
    if baseline.get("intent_key") != _intent_key(intent):
        raise DynamicJobBindingValidationError(
            "controller baseline intent key is foreign"
        )
    _sha(
        baseline.get("intent_file_sha256"),
        label="controller baseline intent_file_sha256",
    )
    if baseline.get("intent_file_sha256") != _canonical_file_sha256(intent):
        raise DynamicJobBindingValidationError(
            "controller baseline intent file is foreign"
        )
    if baseline.get("ssm_ping_status") != "Online":
        raise DynamicJobBindingValidationError(
            "controller baseline controller identity is invalid"
        )
    _require_exact(
        baseline.get("workspace"),
        _EXPECTED_WORKSPACE,
        label="controller baseline workspace",
    )
    bucket = baseline.get("bucket")
    if (
        not isinstance(bucket, str)
        or _BUCKET.fullmatch(bucket) is None
        or ".." in bucket
    ):
        raise DynamicJobBindingValidationError("controller baseline bucket is invalid")
    instance_id = baseline.get("controller_instance_id")
    if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
        raise DynamicJobBindingValidationError(
            "controller baseline controller_instance_id is invalid"
        )
    instance_type = baseline.get("controller_instance_type")
    if (
        not isinstance(instance_type, str)
        or _INSTANCE_TYPE.fullmatch(instance_type) is None
    ):
        raise DynamicJobBindingValidationError(
            "controller baseline controller_instance_type is invalid"
        )
    role_prefix = "arn:aws:iam::246813579024:role/"
    expected_profile = (
        "arn:aws:iam::246813579024:instance-profile/"
        f"{descriptor_controller_identity.removeprefix(role_prefix)}"
    )
    _require_exact(
        baseline.get("controller_profile_arn"),
        expected_profile,
        label="controller baseline controller_profile_arn",
    )
    cluster_name = baseline.get("controller_cluster_name")
    if (
        not isinstance(cluster_name, str)
        or _SAFE_NAME.fullmatch(cluster_name) is None
        or cluster_name in {".", ".."}
        or ".." in cluster_name
    ):
        raise DynamicJobBindingValidationError(
            "controller baseline controller_cluster_name is invalid"
        )
    active_job_ids = baseline.get("active_exact_name_job_ids")
    if type(active_job_ids) is not list:
        raise DynamicJobBindingValidationError(
            "controller baseline active job IDs must be a list"
        )
    normalized_job_ids = [
        _positive_int(item, label="controller baseline active job ID")
        for item in active_job_ids
    ]
    if normalized_job_ids != sorted(normalized_job_ids) or len(
        normalized_job_ids
    ) != len(set(normalized_job_ids)):
        raise DynamicJobBindingValidationError(
            "controller baseline active job IDs are not sorted and unique"
        )
    active_instance_ids = baseline.get("active_tagged_p5_instance_ids")
    if type(active_instance_ids) is not list:
        raise DynamicJobBindingValidationError(
            "controller baseline active P5 IDs must be a list"
        )
    if any(
        not isinstance(item, str) or _INSTANCE_ID.fullmatch(item) is None
        for item in active_instance_ids
    ):
        raise DynamicJobBindingValidationError(
            "controller baseline active P5 ID is invalid"
        )
    if (
        active_instance_ids != sorted(active_instance_ids)
        or len(active_instance_ids) != len(set(active_instance_ids))
        or active_instance_ids
    ):
        raise DynamicJobBindingValidationError(
            "controller baseline contains active tagged P5 instances"
        )
    observed_at = _time(
        baseline.get("observed_at"), label="controller baseline observed_at"
    )
    baseline_history = _history(
        baseline.get("exact_name_history"),
        intent=intent,
        baseline=baseline,
        descriptor_controller_identity=descriptor_controller_identity,
        label="controller baseline history",
    )
    derived_active_job_ids = sorted(
        int(row["sky_job_id"])
        for row in baseline_history
        if row["controller_status"] in _ACTIVE_CONTROLLER_STATUSES
    )
    if normalized_job_ids != derived_active_job_ids:
        raise DynamicJobBindingValidationError(
            "controller baseline active job summary disagrees with exact history"
        )
    if derived_active_job_ids:
        raise DynamicJobBindingValidationError(
            "controller baseline contains active exact-name jobs"
        )
    return (
        baseline,
        baseline_history,
        observed_at,
    )


def _acquisition(
    value: object,
    *,
    intent: Mapping[str, object],
    baseline: Mapping[str, object],
    now: datetime,
) -> datetime:
    acquisition = _exact_record(
        value,
        fields=_ACQUISITION_FIELDS,
        digest_field="acquisition_body_sha256",
        version=1,
        record_type="glm52_sky_submission_acquired_v1",
        label="submission acquisition",
    )
    run_id = str(intent["run_id"])
    for field in (name for name in acquisition if name.endswith("_sha256")):
        _sha(acquisition[field], label=f"submission acquisition {field}")
    for field in _ACQUISITION_KEY_FIELDS:
        _campaign_key(
            acquisition.get(field),
            run_id=run_id,
            label=f"submission acquisition {field}",
        )
    for field in (
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
        "intent_body_sha256",
        "sky_job_name",
        "must_start_by",
    ):
        if acquisition.get(field) != intent.get(field):
            raise DynamicJobBindingValidationError(
                "submission acquisition identity is foreign"
            )
    if acquisition.get("intent_key") != _intent_key(intent):
        raise DynamicJobBindingValidationError(
            "submission acquisition intent key is foreign"
        )
    _require_exact(
        acquisition.get("intent_file_sha256"),
        _canonical_file_sha256(intent),
        label="submission acquisition intent_file_sha256",
    )
    _require_exact(
        acquisition.get("controller_baseline_key"),
        _controller_baseline_key(baseline),
        label="submission acquisition controller_baseline_key",
    )
    _require_exact(
        acquisition.get("controller_baseline_file_sha256"),
        _canonical_file_sha256(baseline),
        label="submission acquisition controller_baseline_file_sha256",
    )
    _require_exact(
        acquisition.get("controller_baseline_body_sha256"),
        baseline.get("baseline_body_sha256"),
        label="submission acquisition controller_baseline_body_sha256",
    )
    _require_exact(
        acquisition.get("must_start_control_plane_ready_key"),
        _control_plane_ready_key(acquisition, intent=intent),
        label="submission acquisition must_start_control_plane_ready_key",
    )
    acquired_at = _time(
        acquisition.get("acquired_at"), label="submission acquisition acquired_at"
    )
    baseline_observed_at = _time(
        baseline.get("observed_at"), label="controller baseline observed_at"
    )
    if baseline_observed_at > acquired_at:
        raise DynamicJobBindingValidationError(
            "controller baseline follows submission acquisition"
        )
    if acquired_at - baseline_observed_at > timedelta(seconds=60):
        raise DynamicJobBindingValidationError(
            "controller baseline is older than 60 seconds at acquisition"
        )
    if acquired_at > now or now >= _time(
        intent["must_start_by"], label="submission intent must_start_by"
    ):
        raise DynamicJobBindingValidationError(
            "submission acquisition is expired or inconsistent"
        )
    return acquired_at


def _stored_binding(
    value: object,
    *,
    intent: Mapping[str, object],
    baseline: Mapping[str, object],
    acquisition: Mapping[str, object],
    current: list[dict[str, object]],
    post_baseline: list[dict[str, object]],
    acquired_at: datetime,
    now: datetime,
) -> dict[str, object]:
    binding = _exact_record(
        value,
        fields=_BINDING_FIELDS,
        digest_field="job_binding_body_sha256",
        version=2,
        record_type="glm52_sky_must_start_job_binding_v2",
        label="JOB_BINDING",
    )
    for field in (
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "intent_body_sha256",
        "sky_job_name",
        "must_start_by",
    ):
        if binding.get(field) != intent.get(field):
            raise DynamicJobBindingValidationError("JOB_BINDING identity is foreign")
    if binding.get("intent_key") != _intent_key(intent):
        raise DynamicJobBindingValidationError("JOB_BINDING intent key is foreign")
    if binding.get("controller_baseline_body_sha256") != baseline.get(
        "baseline_body_sha256"
    ) or binding.get("acquisition_body_sha256") != acquisition.get(
        "acquisition_body_sha256"
    ):
        raise DynamicJobBindingValidationError("JOB_BINDING authority is foreign")
    for field in (
        "workspace",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
    ):
        if binding.get(field) != baseline.get(field):
            raise DynamicJobBindingValidationError(
                "JOB_BINDING controller identity is foreign"
            )
    job_id = _positive_int(binding.get("sky_job_id"), label="JOB_BINDING.sky_job_id")
    if len(post_baseline) != 1 or post_baseline[0]["sky_job_id"] != job_id:
        raise DynamicJobBindingValidationError(
            "JOB_BINDING is not the only post-baseline controller row"
        )
    matches = [row for row in current if row["sky_job_id"] == job_id]
    if len(matches) != 1:
        raise DynamicJobBindingValidationError(
            "JOB_BINDING target is absent or ambiguous"
        )
    row = matches[0]
    if binding.get("controller_status") not in _CONTROLLER_STATUSES:
        raise DynamicJobBindingValidationError(
            "JOB_BINDING controller status is invalid"
        )
    for field in ("controller_submitted_at", "controller_identity"):
        if binding.get(field) != row.get(field):
            raise DynamicJobBindingValidationError("JOB_BINDING target row drifted")
    bound_at = _time(binding.get("bound_at"), label="JOB_BINDING bound_at")
    if bound_at < acquired_at or bound_at > now:
        raise DynamicJobBindingValidationError(
            "JOB_BINDING bound_at is outside acquired_at and now"
        )
    return row


def _validate_candidate_submission_window(
    row: Mapping[str, object],
    *,
    intent_at: datetime,
    deadline: datetime,
) -> None:
    submitted_at = _time(
        row.get("controller_submitted_at"),
        label="candidate controller_submitted_at",
    )
    if submitted_at < intent_at or submitted_at >= deadline:
        raise DynamicJobBindingValidationError(
            "candidate submission is outside the intent window"
        )


def resolve_dynamic_v2_job_binding(
    *,
    intent: object,
    controller_baseline: object,
    acquisition: object | None,
    descriptor_controller_identity: str,
    current_exact_name_history: object,
    stored_binding: object | None,
    now: datetime | str,
) -> DynamicJobBindingDecision:
    """Choose one dynamic-v2 binding action; malformed inputs always fail closed."""

    try:
        observed_now = _time(
            now.astimezone(UTC).isoformat().replace("+00:00", "Z")
            if isinstance(now, datetime) and now.tzinfo
            else now,
            label="now",
        )
        intent_value, intent_at, deadline = _intent(intent)
        if (
            not isinstance(descriptor_controller_identity, str)
            or _CONTROLLER_ROLE_ARN.fullmatch(descriptor_controller_identity) is None
        ):
            raise DynamicJobBindingValidationError(
                "descriptor controller identity is invalid"
            )
        baseline, baseline_history, baseline_observed_at = _baseline(
            controller_baseline,
            intent=intent_value,
            descriptor_controller_identity=descriptor_controller_identity,
        )
        if baseline_observed_at < intent_at:
            raise DynamicJobBindingValidationError(
                "controller baseline observation precedes submission intent"
            )
        if any(
            not (
                intent_at
                <= _time(
                    row["controller_submitted_at"],
                    label="baseline controller_submitted_at",
                )
                <= baseline_observed_at
            )
            for row in baseline_history
        ):
            raise DynamicJobBindingValidationError(
                "controller baseline history is outside intent and observation"
            )
        if baseline_observed_at > observed_now:
            raise DynamicJobBindingValidationError(
                "controller baseline observation is in the future"
            )
        current = _history(
            current_exact_name_history,
            intent=intent_value,
            baseline=baseline,
            descriptor_controller_identity=descriptor_controller_identity,
            label="current controller history",
        )
        if any(
            _time(
                row["controller_submitted_at"],
                label="current controller_submitted_at",
            )
            > observed_now
            for row in current
        ):
            raise DynamicJobBindingValidationError(
                "controller history contains a future submission"
            )
        if current[: len(baseline_history)] != baseline_history:
            raise DynamicJobBindingValidationError(
                "controller baseline history mutated, disappeared, or reordered"
            )
        if acquisition is None:
            return DynamicJobBindingDecision(
                "wait-for-acquisition", "no authenticated acquisition selector"
            )
        acquired_at = _acquisition(
            acquisition, intent=intent_value, baseline=baseline, now=observed_now
        )
        new_rows = current[len(baseline_history) :]
        if stored_binding is not None:
            row = _stored_binding(
                stored_binding,
                intent=intent_value,
                baseline=baseline,
                acquisition=_mapping(acquisition, label="submission acquisition"),
                current=current,
                post_baseline=new_rows,
                acquired_at=acquired_at,
                now=observed_now,
            )
            _validate_candidate_submission_window(
                row,
                intent_at=intent_at,
                deadline=deadline,
            )
            return DynamicJobBindingDecision(
                "reconcile-bound-job",
                "authenticated stored binding remains authoritative",
                int(row["sky_job_id"]),
                row,
            )
        if not new_rows:
            return DynamicJobBindingDecision(
                "wait-for-one-new-job", "no new exact-name controller row"
            )
        if len(new_rows) != 1:
            raise DynamicJobBindingValidationError(
                "more than one new exact-name controller row"
            )
        row = new_rows[0]
        _validate_candidate_submission_window(
            row,
            intent_at=intent_at,
            deadline=deadline,
        )
        return DynamicJobBindingDecision(
            "bind-exact-job",
            "one new authenticated in-window controller row",
            int(row["sky_job_id"]),
            row,
        )
    except (DynamicJobBindingValidationError, TypeError, ValueError, OverflowError):
        return DynamicJobBindingDecision(
            "fail-closed", "invalid, foreign, expired, or inconsistent authority"
        )


def dynamic_v2_job_binding_s3_key(*, intent: object) -> str:
    """Return the distinct immutable key for an authenticated qualification intent."""

    intent_value, _, _ = _intent(intent)
    return (
        f"campaigns/{intent_value['run_id']}/monitor/must-start/qualification/"
        f"{intent_value['intent_body_sha256']}/DYNAMIC_JOB_BINDING.json"
    )


def _canonical_explicit_time(value: datetime | str, *, label: str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise DynamicJobBindingValidationError(f"{label} must be timezone-aware")
        try:
            canonical = value.astimezone(UTC).isoformat().replace("+00:00", "Z")
        except (ValueError, OverflowError) as error:
            raise DynamicJobBindingValidationError(
                f"{label} cannot be normalized to UTC"
            ) from error
    elif isinstance(value, str):
        canonical = value
    else:
        raise DynamicJobBindingValidationError(f"{label} must be canonical ISO-8601")
    _time(canonical, label=label)
    return canonical


def build_dynamic_v2_job_binding(
    *,
    intent: object,
    controller_baseline: object,
    acquisition: object,
    descriptor_controller_identity: str,
    current_exact_name_history: object,
    bound_at: datetime | str,
    now: datetime | str,
) -> dict[str, object]:
    """Build the one exact dynamic-v2 numeric binding selected by the resolver."""

    intent_value = _mapping(intent, label="submission intent")
    baseline_value = _mapping(controller_baseline, label="controller baseline")
    history_value = (
        list(current_exact_name_history)
        if type(current_exact_name_history) is list
        else current_exact_name_history
    )
    decision = resolve_dynamic_v2_job_binding(
        intent=intent_value,
        controller_baseline=baseline_value,
        acquisition=acquisition,
        descriptor_controller_identity=descriptor_controller_identity,
        current_exact_name_history=history_value,
        stored_binding=None,
        now=now,
    )
    if (
        decision.action != "bind-exact-job"
        or decision.sky_job_id is None
        or decision.job_row is None
    ):
        raise DynamicJobBindingValidationError(
            "resolver did not select one exact job for binding"
        )

    acquisition_value = _mapping(acquisition, label="submission acquisition")
    canonical_bound_at = _canonical_explicit_time(bound_at, label="bound_at")
    canonical_now = _canonical_explicit_time(now, label="now")
    acquired_at = _time(
        acquisition_value.get("acquired_at"),
        label="submission acquisition acquired_at",
    )
    parsed_bound_at = _time(canonical_bound_at, label="bound_at")
    parsed_now = _time(canonical_now, label="now")
    if not acquired_at <= parsed_bound_at <= parsed_now:
        raise DynamicJobBindingValidationError(
            "bound_at is outside acquired_at and now"
        )

    row = decision.job_row
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_must_start_job_binding_v2",
        "account_id": intent_value["account_id"],
        "region": intent_value["region"],
        "run_id": intent_value["run_id"],
        "managed_mode": intent_value["managed_mode"],
        "descriptor_key": intent_value["descriptor_key"],
        "descriptor_file_sha256": intent_value["descriptor_file_sha256"],
        "descriptor_body_sha256": intent_value["descriptor_body_sha256"],
        "intent_key": _intent_key(intent_value),
        "intent_body_sha256": intent_value["intent_body_sha256"],
        "controller_baseline_body_sha256": baseline_value["baseline_body_sha256"],
        "acquisition_body_sha256": acquisition_value["acquisition_body_sha256"],
        "sky_job_name": intent_value["sky_job_name"],
        "must_start_by": intent_value["must_start_by"],
        "workspace": baseline_value["workspace"],
        "controller_instance_id": baseline_value["controller_instance_id"],
        "controller_instance_type": baseline_value["controller_instance_type"],
        "controller_profile_arn": baseline_value["controller_profile_arn"],
        "controller_cluster_name": baseline_value["controller_cluster_name"],
        "sky_job_id": decision.sky_job_id,
        "controller_submitted_at": row["controller_submitted_at"],
        "controller_status": row["controller_status"],
        "controller_identity": row["controller_identity"],
        "bound_at": canonical_bound_at,
    }
    binding = {
        **body,
        "job_binding_body_sha256": _canonical_sha256(body),
    }
    return validate_dynamic_v2_job_binding(
        binding,
        intent=intent_value,
        controller_baseline=baseline_value,
        acquisition=acquisition_value,
        descriptor_controller_identity=descriptor_controller_identity,
        current_exact_name_history=history_value,
        now=canonical_now,
    )


def validate_dynamic_v2_job_binding(
    value: object,
    *,
    intent: object,
    controller_baseline: object,
    acquisition: object,
    descriptor_controller_identity: str,
    current_exact_name_history: object,
    now: datetime | str,
) -> dict[str, object]:
    """Authenticate one binding against its authorities and current controller row."""

    binding = _mapping(value, label="JOB_BINDING")
    decision = resolve_dynamic_v2_job_binding(
        intent=_mapping(intent, label="submission intent"),
        controller_baseline=_mapping(controller_baseline, label="controller baseline"),
        acquisition=_mapping(acquisition, label="submission acquisition"),
        descriptor_controller_identity=descriptor_controller_identity,
        current_exact_name_history=(
            list(current_exact_name_history)
            if type(current_exact_name_history) is list
            else current_exact_name_history
        ),
        stored_binding=binding,
        now=now,
    )
    if decision.action != "reconcile-bound-job" or decision.job_row is None:
        raise DynamicJobBindingValidationError(
            "resolver did not reconcile the exact stored binding"
        )
    bound_status = binding.get("controller_status")
    current_status = decision.job_row.get("controller_status")
    if (
        not isinstance(bound_status, str)
        or not isinstance(current_status, str)
        or current_status
        not in _CONTROLLER_STATUS_PROGRESSIONS.get(bound_status, frozenset())
    ):
        raise DynamicJobBindingValidationError(
            "JOB_BINDING controller status did not progress legally"
        )
    return dict(binding)
