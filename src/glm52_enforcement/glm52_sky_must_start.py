"""Enforcement-native must-start cancellation authorities and policy."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Mapping

APPROVED_ACCOUNT_ID = "246813579024"
APPROVED_REGION = "us-west-2"
ManagedMode = Literal["production", "qualification", "cache-seed"]
MustStartAction = Literal[
    "wait",
    "timely-started",
    "request-cancel",
    "reconcile",
    "complete",
    "completed",
    "observe",
    "accept",
    "grace",
    "cancel",
    "fail-closed",
]
AlertKind = Literal["requested", "completed"]

_MODES = {"production", "qualification", "cache-seed"}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")

_AUTHORITY_FIELDS = {
    "run_id",
    "managed_mode",
    "account_id",
    "region",
    "bucket",
    "descriptor_body_sha256",
    "submission_body_sha256",
    "sky_job_name",
    "must_start_by",
}
_REQUEST_BODY_FIELDS = _AUTHORITY_FIELDS | {
    "schema_version",
    "record_type",
    "requested_at",
}
_REQUEST_FIELDS = _REQUEST_BODY_FIELDS | {"request_body_sha256"}
_LATCH_BODY_FIELDS = _AUTHORITY_FIELDS | {
    "schema_version",
    "record_type",
    "started_at",
    "published_at",
}
_LATCH_FIELDS = _LATCH_BODY_FIELDS | {"latch_body_sha256"}
_COMPLETION_BODY_FIELDS = _AUTHORITY_FIELDS | {
    "schema_version",
    "record_type",
    "request_body_sha256",
    "completed_at",
    "terminal_status",
}
_COMPLETION_FIELDS = _COMPLETION_BODY_FIELDS | {"completion_body_sha256"}
_ALERT_BODY_FIELDS = _AUTHORITY_FIELDS | {
    "schema_version",
    "record_type",
    "alert_kind",
    "lifecycle_body_sha256",
    "delivered_at",
}
_ALERT_FIELDS = _ALERT_BODY_FIELDS | {"alert_delivery_body_sha256"}
_TARGET_BODY_FIELDS = _AUTHORITY_FIELDS | {
    "schema_version",
    "record_type",
    "request_body_sha256",
    "target_job_id",
    "bound_at",
}
_TARGET_FIELDS = _TARGET_BODY_FIELDS | {"target_binding_body_sha256"}
_CONTROLLER_IDENTITY_FIELDS = {
    "target_job_id",
    "workspace",
    "controller_instance_id",
    "controller_instance_type",
    "controller_profile_arn",
    "controller_cluster_name",
}
_OBSERVATION_BODY_FIELDS = _AUTHORITY_FIELDS | _CONTROLLER_IDENTITY_FIELDS | {
    "schema_version",
    "record_type",
    "status",
    "schedule_state",
    "submitted_at",
    "start_at",
    "worker_cluster_name",
    "recovery_count",
    "observed_at",
}
_OBSERVATION_FIELDS = _OBSERVATION_BODY_FIELDS | {"observation_body_sha256"}
_JOB_BINDING_BODY_FIELDS = _AUTHORITY_FIELDS | _CONTROLLER_IDENTITY_FIELDS | {
    "schema_version",
    "record_type",
    "descriptor_key",
    "descriptor_file_sha256",
    "submission_key",
    "submission_submitted_at",
    "observation_body_sha256",
    "bound_at",
}
_JOB_BINDING_FIELDS = _JOB_BINDING_BODY_FIELDS | {"job_binding_body_sha256"}
_WORKER_LATCH_BODY_FIELDS = _AUTHORITY_FIELDS | {
    "schema_version",
    "record_type",
    "descriptor_key",
    "descriptor_file_sha256",
    "submission_key",
    "submission_submitted_at",
    "repo_tar_sha256",
    "instance_id",
    "instance_type",
    "image_id",
    "worker_role_arn",
    "instance_identity_document_sha256",
    "ec2_pending_time",
    "entrypoint_observed_at",
}
_WORKER_LATCH_FIELDS = _WORKER_LATCH_BODY_FIELDS | {"worker_latch_body_sha256"}
_ACCEPTED_BODY_FIELDS = _AUTHORITY_FIELDS | {
    "schema_version",
    "record_type",
    "source_kind",
    "source_key",
    "source_version_id",
    "source_etag",
    "source_body_sha256",
    "source_last_modified",
    "started_at",
    "accepted_at",
}
_ACCEPTED_FIELDS = _ACCEPTED_BODY_FIELDS | {"accepted_body_sha256"}
_SUPERSEDED_BODY_FIELDS = _AUTHORITY_FIELDS | {
    "schema_version",
    "record_type",
    "request_body_sha256",
    "accepted_body_sha256",
    "superseded_at",
}
_SUPERSEDED_FIELDS = _SUPERSEDED_BODY_FIELDS | {"superseded_body_sha256"}
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
_TERMINAL_STATUSES = {
    "SUCCEEDED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
    "CANCELLED",
}


class MustStartValidationError(ValueError):
    """Raised when must-start lifecycle authority is malformed or foreign."""


@dataclass(frozen=True)
class MustStartDecision:
    action: MustStartAction
    reason: str


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _canonical_time(value: datetime | str, *, field: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise MustStartValidationError(f"{field} must be ISO-8601") from error
    else:
        raise MustStartValidationError(f"{field} must be ISO-8601")
    if parsed.tzinfo is None:
        raise MustStartValidationError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: datetime | str, *, field: str) -> datetime:
    return datetime.fromisoformat(
        _canonical_time(value, field=field).replace("Z", "+00:00")
    )


def expected_sky_job_name(run_id: str, managed_mode: str) -> str:
    if _RUN_ID.fullmatch(run_id) is None:
        raise MustStartValidationError("run_id is invalid")
    if managed_mode not in _MODES:
        raise MustStartValidationError("managed_mode is invalid")
    suffix = {
        "production": "",
        "qualification": "-qualification",
        "cache-seed": "-cache-seed",
    }[managed_mode]
    return f"{run_id}{suffix}"


def _normalized_authority(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
) -> dict[str, object]:
    if account_id != APPROVED_ACCOUNT_ID:
        raise MustStartValidationError("account does not match approved authority")
    if region != APPROVED_REGION:
        raise MustStartValidationError("region does not match approved authority")
    if _BUCKET.fullmatch(bucket) is None:
        raise MustStartValidationError("bucket is invalid")
    for field, digest in (
        ("descriptor_body_sha256", descriptor_body_sha256),
        ("submission_body_sha256", submission_body_sha256),
    ):
        if _HEX64.fullmatch(digest) is None:
            raise MustStartValidationError(f"{field} is invalid")
    if sky_job_name != expected_sky_job_name(run_id, managed_mode):
        raise MustStartValidationError("SkyPilot job name does not match managed mode")
    return {
        "run_id": run_id,
        "managed_mode": managed_mode,
        "account_id": account_id,
        "region": region,
        "bucket": bucket,
        "descriptor_body_sha256": descriptor_body_sha256,
        "submission_body_sha256": submission_body_sha256,
        "sky_job_name": sky_job_name,
        "must_start_by": _canonical_time(must_start_by, field="must_start_by"),
    }


def _validate_canonical_authority(value: Mapping[str, object]) -> None:
    normalized = _normalized_authority(
        run_id=str(value.get("run_id", "")),
        managed_mode=str(value.get("managed_mode", "")),
        account_id=str(value.get("account_id", "")),
        region=str(value.get("region", "")),
        bucket=str(value.get("bucket", "")),
        descriptor_body_sha256=str(value.get("descriptor_body_sha256", "")),
        submission_body_sha256=str(value.get("submission_body_sha256", "")),
        sky_job_name=str(value.get("sky_job_name", "")),
        must_start_by=value.get("must_start_by"),  # type: ignore[arg-type]
    )
    for field in _AUTHORITY_FIELDS:
        if value.get(field) != normalized[field]:
            raise MustStartValidationError(f"{field} is not canonical")


def _normalize_expected_authority(
    value: Mapping[str, object],
) -> dict[str, object]:
    if set(value) != _AUTHORITY_FIELDS:
        raise MustStartValidationError("expected authority schema mismatch")
    return _normalized_authority(
        run_id=str(value.get("run_id", "")),
        managed_mode=str(value.get("managed_mode", "")),
        account_id=str(value.get("account_id", "")),
        region=str(value.get("region", "")),
        bucket=str(value.get("bucket", "")),
        descriptor_body_sha256=str(value.get("descriptor_body_sha256", "")),
        submission_body_sha256=str(value.get("submission_body_sha256", "")),
        sky_job_name=str(value.get("sky_job_name", "")),
        must_start_by=value.get("must_start_by"),  # type: ignore[arg-type]
    )


def _require_expected_authority(
    marker: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    for field in _AUTHORITY_FIELDS:
        if marker.get(field) != expected[field]:
            raise MustStartValidationError(f"{label} authority mismatch")


def _validate_digest(
    value: Mapping[str, object],
    *,
    fields: set[str],
    digest_field: str,
    record_type: str,
) -> dict[str, object]:
    if set(value) != fields:
        raise MustStartValidationError(f"{record_type} schema mismatch")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("record_type") != record_type
    ):
        raise MustStartValidationError(f"{record_type} schema mismatch")
    body = dict(value)
    actual = body.pop(digest_field)
    if not isinstance(actual, str) or actual != canonical_sha256(body):
        raise MustStartValidationError(f"{record_type} body SHA-256 mismatch")
    _validate_canonical_authority(value)
    return dict(value)


def _require_hex(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MustStartValidationError(f"{field} is invalid")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise MustStartValidationError(f"{field} is invalid")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise MustStartValidationError(f"{field} is invalid")
    return value


def _require_scoped_key(value: object, *, run_id: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(f"campaigns/{run_id}/")
        or ".." in value
    ):
        raise MustStartValidationError(f"{field} is invalid")
    return value


def _normalized_controller_identity(
    *,
    target_job_id: int,
    workspace: str,
    controller_instance_id: str,
    controller_instance_type: str,
    controller_profile_arn: str,
    controller_cluster_name: str,
) -> dict[str, object]:
    _require_positive_int(target_job_id, field="target_job_id")
    if workspace != "default":
        raise MustStartValidationError("workspace must be default")
    if not re.fullmatch(r"i-[0-9a-f]{8,32}", controller_instance_id):
        raise MustStartValidationError("controller_instance_id is invalid")
    if not isinstance(controller_instance_type, str) or not controller_instance_type:
        raise MustStartValidationError("controller_instance_type is invalid")
    profile_prefix = f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:instance-profile/"
    if (
        not isinstance(controller_profile_arn, str)
        or not controller_profile_arn.startswith(profile_prefix)
    ):
        raise MustStartValidationError("controller_profile_arn is invalid")
    if (
        not isinstance(controller_cluster_name, str)
        or not controller_cluster_name.startswith("sky-jobs-controller-")
    ):
        raise MustStartValidationError("controller_cluster_name is invalid")
    return {
        "target_job_id": target_job_id,
        "workspace": workspace,
        "controller_instance_id": controller_instance_id,
        "controller_instance_type": controller_instance_type,
        "controller_profile_arn": controller_profile_arn,
        "controller_cluster_name": controller_cluster_name,
    }


def _validate_controller_identity(value: Mapping[str, object]) -> None:
    normalized = _normalized_controller_identity(
        target_job_id=value.get("target_job_id"),  # type: ignore[arg-type]
        workspace=str(value.get("workspace", "")),
        controller_instance_id=str(value.get("controller_instance_id", "")),
        controller_instance_type=str(value.get("controller_instance_type", "")),
        controller_profile_arn=str(value.get("controller_profile_arn", "")),
        controller_cluster_name=str(value.get("controller_cluster_name", "")),
    )
    for field in _CONTROLLER_IDENTITY_FIELDS:
        if value.get(field) != normalized[field]:
            raise MustStartValidationError(f"{field} is not canonical")


def build_must_start_cancel_requested(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    requested_at: datetime | str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    requested = _canonical_time(requested_at, field="requested_at")
    if _parse_time(requested, field="requested_at") < _parse_time(
        str(authority["must_start_by"]), field="must_start_by"
    ):
        raise MustStartValidationError("requested_at precedes must_start_by")
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_cancel_requested_v1",
        **authority,
        "requested_at": requested,
    }
    marker = {**body, "request_body_sha256": canonical_sha256(body)}
    return validate_must_start_cancel_requested(marker)


def validate_must_start_cancel_requested(
    value: Mapping[str, object],
) -> dict[str, object]:
    marker = _validate_digest(
        value,
        fields=_REQUEST_FIELDS,
        digest_field="request_body_sha256",
        record_type="glm52_sky_must_start_cancel_requested_v1",
    )
    requested_at = _canonical_time(marker["requested_at"], field="requested_at")
    if marker["requested_at"] != requested_at:
        raise MustStartValidationError("requested_at is not canonical")
    if _parse_time(requested_at, field="requested_at") < _parse_time(
        str(marker["must_start_by"]), field="must_start_by"
    ):
        raise MustStartValidationError("requested_at precedes must_start_by")
    return marker


def build_timely_start_latch(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    started_at: datetime | str,
    published_at: datetime | str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    started = _canonical_time(started_at, field="started_at")
    published = _canonical_time(published_at, field="published_at")
    if _parse_time(published, field="published_at") < _parse_time(
        started, field="started_at"
    ):
        raise MustStartValidationError("published_at precedes started_at")
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_timely_start_latch_v1",
        **authority,
        "started_at": started,
        "published_at": published,
    }
    latch = {**body, "latch_body_sha256": canonical_sha256(body)}
    return validate_timely_start_latch(latch)


def validate_timely_start_latch(
    value: Mapping[str, object],
) -> dict[str, object]:
    latch = _validate_digest(
        value,
        fields=_LATCH_FIELDS,
        digest_field="latch_body_sha256",
        record_type="glm52_sky_timely_start_latch_v1",
    )
    for field in ("started_at", "published_at"):
        canonical = _canonical_time(latch[field], field=field)
        if latch[field] != canonical:
            raise MustStartValidationError(f"{field} is not canonical")
    if _parse_time(str(latch["published_at"]), field="published_at") < _parse_time(
        str(latch["started_at"]), field="started_at"
    ):
        raise MustStartValidationError("published_at precedes started_at")
    return latch


def build_must_start_cancel_completed(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    request_body_sha256: str,
    completed_at: datetime | str,
    terminal_status: str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    if _HEX64.fullmatch(request_body_sha256) is None:
        raise MustStartValidationError("request body SHA-256 is invalid")
    if terminal_status not in {
        "CANCELLED",
        "SUCCEEDED",
        "FAILED",
        "FAILED_SETUP",
        "FAILED_PRECHECKS",
        "FAILED_NO_RESOURCE",
        "FAILED_CONTROLLER",
    }:
        raise MustStartValidationError("terminal status is invalid")
    completed = _canonical_time(completed_at, field="completed_at")
    if _parse_time(completed, field="completed_at") < _parse_time(
        str(authority["must_start_by"]), field="must_start_by"
    ):
        raise MustStartValidationError("completed_at precedes must_start_by")
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_cancel_completed_v1",
        **authority,
        "request_body_sha256": request_body_sha256,
        "completed_at": completed,
        "terminal_status": terminal_status,
    }
    marker = {**body, "completion_body_sha256": canonical_sha256(body)}
    return validate_must_start_cancel_completed(marker)


def validate_must_start_cancel_completed(
    value: Mapping[str, object],
) -> dict[str, object]:
    marker = _validate_digest(
        value,
        fields=_COMPLETION_FIELDS,
        digest_field="completion_body_sha256",
        record_type="glm52_sky_must_start_cancel_completed_v1",
    )
    if _HEX64.fullmatch(str(marker.get("request_body_sha256"))) is None:
        raise MustStartValidationError("request body SHA-256 is invalid")
    if marker.get("terminal_status") not in {
        "CANCELLED",
        "SUCCEEDED",
        "FAILED",
        "FAILED_SETUP",
        "FAILED_PRECHECKS",
        "FAILED_NO_RESOURCE",
        "FAILED_CONTROLLER",
    }:
        raise MustStartValidationError("terminal status is invalid")
    canonical = _canonical_time(marker["completed_at"], field="completed_at")
    if marker["completed_at"] != canonical:
        raise MustStartValidationError("completed_at is not canonical")
    if _parse_time(canonical, field="completed_at") < _parse_time(
        str(marker["must_start_by"]), field="must_start_by"
    ):
        raise MustStartValidationError("completed_at precedes must_start_by")
    return marker


def build_must_start_alert_delivered(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    alert_kind: AlertKind,
    lifecycle_body_sha256: str,
    delivered_at: datetime | str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    if alert_kind not in {"requested", "completed"}:
        raise MustStartValidationError("alert kind is invalid")
    if _HEX64.fullmatch(lifecycle_body_sha256) is None:
        raise MustStartValidationError("lifecycle body SHA-256 is invalid")
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_alert_delivered_v1",
        **authority,
        "alert_kind": alert_kind,
        "lifecycle_body_sha256": lifecycle_body_sha256,
        "delivered_at": _canonical_time(delivered_at, field="delivered_at"),
    }
    marker = {**body, "alert_delivery_body_sha256": canonical_sha256(body)}
    return validate_must_start_alert_delivered(marker)


def validate_must_start_alert_delivered(
    value: Mapping[str, object],
) -> dict[str, object]:
    marker = _validate_digest(
        value,
        fields=_ALERT_FIELDS,
        digest_field="alert_delivery_body_sha256",
        record_type="glm52_sky_must_start_alert_delivered_v1",
    )
    if marker.get("alert_kind") not in {"requested", "completed"}:
        raise MustStartValidationError("alert kind is invalid")
    if _HEX64.fullmatch(str(marker.get("lifecycle_body_sha256"))) is None:
        raise MustStartValidationError("lifecycle body SHA-256 is invalid")
    canonical = _canonical_time(marker["delivered_at"], field="delivered_at")
    if marker["delivered_at"] != canonical:
        raise MustStartValidationError("delivered_at is not canonical")
    return marker


def build_must_start_target_binding(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    request_body_sha256: str,
    target_job_id: int,
    bound_at: datetime | str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    if _HEX64.fullmatch(request_body_sha256) is None:
        raise MustStartValidationError("request body SHA-256 is invalid")
    if (
        not isinstance(target_job_id, int)
        or isinstance(target_job_id, bool)
        or target_job_id <= 0
    ):
        raise MustStartValidationError("target job ID is invalid")
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_target_binding_v1",
        **authority,
        "request_body_sha256": request_body_sha256,
        "target_job_id": target_job_id,
        "bound_at": _canonical_time(bound_at, field="bound_at"),
    }
    marker = {**body, "target_binding_body_sha256": canonical_sha256(body)}
    return validate_must_start_target_binding(marker)


def validate_must_start_target_binding(
    value: Mapping[str, object],
) -> dict[str, object]:
    marker = _validate_digest(
        value,
        fields=_TARGET_FIELDS,
        digest_field="target_binding_body_sha256",
        record_type="glm52_sky_must_start_target_binding_v1",
    )
    if _HEX64.fullmatch(str(marker.get("request_body_sha256"))) is None:
        raise MustStartValidationError("request body SHA-256 is invalid")
    target = marker.get("target_job_id")
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        raise MustStartValidationError("target job ID is invalid")
    canonical = _canonical_time(marker["bound_at"], field="bound_at")
    if marker["bound_at"] != canonical:
        raise MustStartValidationError("bound_at is not canonical")
    return marker


def build_must_start_controller_observation(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    target_job_id: int,
    workspace: str,
    controller_instance_id: str,
    controller_instance_type: str,
    controller_profile_arn: str,
    controller_cluster_name: str,
    status: str,
    schedule_state: str,
    submitted_at: datetime | str,
    start_at: datetime | str | None,
    worker_cluster_name: str | None,
    recovery_count: int,
    observed_at: datetime | str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    controller = _normalized_controller_identity(
        target_job_id=target_job_id,
        workspace=workspace,
        controller_instance_id=controller_instance_id,
        controller_instance_type=controller_instance_type,
        controller_profile_arn=controller_profile_arn,
        controller_cluster_name=controller_cluster_name,
    )
    if status not in _CONTROLLER_STATUSES:
        raise MustStartValidationError("controller status is invalid")
    if not isinstance(schedule_state, str) or not schedule_state:
        raise MustStartValidationError("schedule_state is invalid")
    if worker_cluster_name is not None and not isinstance(
        worker_cluster_name, str
    ):
        raise MustStartValidationError("worker_cluster_name is invalid")
    _require_nonnegative_int(recovery_count, field="recovery_count")
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_controller_observation_v1",
        **authority,
        **controller,
        "status": status,
        "schedule_state": schedule_state,
        "submitted_at": _canonical_time(submitted_at, field="submitted_at"),
        "start_at": (
            None if start_at is None else _canonical_time(start_at, field="start_at")
        ),
        "worker_cluster_name": worker_cluster_name,
        "recovery_count": recovery_count,
        "observed_at": _canonical_time(observed_at, field="observed_at"),
    }
    marker = {**body, "observation_body_sha256": canonical_sha256(body)}
    return validate_must_start_controller_observation(marker)


def validate_must_start_controller_observation(
    value: Mapping[str, object],
) -> dict[str, object]:
    marker = _validate_digest(
        value,
        fields=_OBSERVATION_FIELDS,
        digest_field="observation_body_sha256",
        record_type="glm52_sky_must_start_controller_observation_v1",
    )
    _validate_controller_identity(marker)
    if marker.get("status") not in _CONTROLLER_STATUSES:
        raise MustStartValidationError("controller status is invalid")
    if not isinstance(marker.get("schedule_state"), str) or not marker.get(
        "schedule_state"
    ):
        raise MustStartValidationError("schedule_state is invalid")
    for field in ("submitted_at", "observed_at"):
        canonical = _canonical_time(marker[field], field=field)
        if marker[field] != canonical:
            raise MustStartValidationError(f"{field} is not canonical")
    if marker.get("start_at") is not None:
        canonical = _canonical_time(marker["start_at"], field="start_at")
        if marker["start_at"] != canonical:
            raise MustStartValidationError("start_at is not canonical")
    if marker.get("worker_cluster_name") is not None and not isinstance(
        marker.get("worker_cluster_name"), str
    ):
        raise MustStartValidationError("worker_cluster_name is invalid")
    _require_nonnegative_int(marker.get("recovery_count"), field="recovery_count")
    return marker


def build_must_start_job_binding(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    descriptor_key: str,
    descriptor_file_sha256: str,
    submission_key: str,
    submission_submitted_at: datetime | str,
    target_job_id: int,
    workspace: str,
    controller_instance_id: str,
    controller_instance_type: str,
    controller_profile_arn: str,
    controller_cluster_name: str,
    observation_body_sha256: str,
    bound_at: datetime | str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    controller = _normalized_controller_identity(
        target_job_id=target_job_id,
        workspace=workspace,
        controller_instance_id=controller_instance_id,
        controller_instance_type=controller_instance_type,
        controller_profile_arn=controller_profile_arn,
        controller_cluster_name=controller_cluster_name,
    )
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_job_binding_v1",
        **authority,
        **controller,
        "descriptor_key": _require_scoped_key(
            descriptor_key, run_id=run_id, field="descriptor_key"
        ),
        "descriptor_file_sha256": _require_hex(
            descriptor_file_sha256, field="descriptor_file_sha256"
        ),
        "submission_key": _require_scoped_key(
            submission_key, run_id=run_id, field="submission_key"
        ),
        "submission_submitted_at": _canonical_time(
            submission_submitted_at, field="submission_submitted_at"
        ),
        "observation_body_sha256": _require_hex(
            observation_body_sha256, field="observation_body_sha256"
        ),
        "bound_at": _canonical_time(bound_at, field="bound_at"),
    }
    marker = {**body, "job_binding_body_sha256": canonical_sha256(body)}
    return validate_must_start_job_binding(marker)


def validate_must_start_job_binding(
    value: Mapping[str, object],
) -> dict[str, object]:
    marker = _validate_digest(
        value,
        fields=_JOB_BINDING_FIELDS,
        digest_field="job_binding_body_sha256",
        record_type="glm52_sky_must_start_job_binding_v1",
    )
    _validate_controller_identity(marker)
    run_id = str(marker["run_id"])
    for field in ("descriptor_key", "submission_key"):
        _require_scoped_key(marker.get(field), run_id=run_id, field=field)
    for field in (
        "descriptor_file_sha256",
        "observation_body_sha256",
    ):
        _require_hex(marker.get(field), field=field)
    for field in ("submission_submitted_at", "bound_at"):
        canonical = _canonical_time(marker[field], field=field)
        if marker[field] != canonical:
            raise MustStartValidationError(f"{field} is not canonical")
    return marker


def build_worker_start_latch(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    descriptor_key: str,
    descriptor_file_sha256: str,
    submission_key: str,
    submission_submitted_at: datetime | str,
    repo_tar_sha256: str,
    instance_id: str,
    instance_type: str,
    image_id: str,
    worker_role_arn: str,
    instance_identity_document_sha256: str,
    ec2_pending_time: datetime | str,
    entrypoint_observed_at: datetime | str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    if not re.fullmatch(r"i-[0-9a-f]{8,32}", instance_id):
        raise MustStartValidationError("instance_id is invalid")
    if not isinstance(instance_type, str) or not instance_type:
        raise MustStartValidationError("instance_type is invalid")
    if not re.fullmatch(r"ami-[0-9a-f]{8,32}", image_id):
        raise MustStartValidationError("image_id is invalid")
    role_prefix = f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
    if not isinstance(worker_role_arn, str) or not worker_role_arn.startswith(
        role_prefix
    ):
        raise MustStartValidationError("worker_role_arn is invalid")
    pending = _canonical_time(ec2_pending_time, field="ec2_pending_time")
    observed = _canonical_time(
        entrypoint_observed_at, field="entrypoint_observed_at"
    )
    if _parse_time(observed, field="entrypoint_observed_at") < _parse_time(
        pending, field="ec2_pending_time"
    ):
        raise MustStartValidationError("entrypoint precedes EC2 pending time")
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_worker_start_latch_v1",
        **authority,
        "descriptor_key": _require_scoped_key(
            descriptor_key, run_id=run_id, field="descriptor_key"
        ),
        "descriptor_file_sha256": _require_hex(
            descriptor_file_sha256, field="descriptor_file_sha256"
        ),
        "submission_key": _require_scoped_key(
            submission_key, run_id=run_id, field="submission_key"
        ),
        "submission_submitted_at": _canonical_time(
            submission_submitted_at, field="submission_submitted_at"
        ),
        "repo_tar_sha256": _require_hex(
            repo_tar_sha256, field="repo_tar_sha256"
        ),
        "instance_id": instance_id,
        "instance_type": instance_type,
        "image_id": image_id,
        "worker_role_arn": worker_role_arn,
        "instance_identity_document_sha256": _require_hex(
            instance_identity_document_sha256,
            field="instance_identity_document_sha256",
        ),
        "ec2_pending_time": pending,
        "entrypoint_observed_at": observed,
    }
    marker = {**body, "worker_latch_body_sha256": canonical_sha256(body)}
    return validate_worker_start_latch(marker)


def validate_worker_start_latch(
    value: Mapping[str, object],
) -> dict[str, object]:
    marker = _validate_digest(
        value,
        fields=_WORKER_LATCH_FIELDS,
        digest_field="worker_latch_body_sha256",
        record_type="glm52_sky_worker_start_latch_v1",
    )
    run_id = str(marker["run_id"])
    for field in ("descriptor_key", "submission_key"):
        _require_scoped_key(marker.get(field), run_id=run_id, field=field)
    for field in (
        "descriptor_file_sha256",
        "repo_tar_sha256",
        "instance_identity_document_sha256",
    ):
        _require_hex(marker.get(field), field=field)
    if not re.fullmatch(r"i-[0-9a-f]{8,32}", str(marker.get("instance_id", ""))):
        raise MustStartValidationError("instance_id is invalid")
    if not isinstance(marker.get("instance_type"), str) or not marker.get(
        "instance_type"
    ):
        raise MustStartValidationError("instance_type is invalid")
    if not re.fullmatch(r"ami-[0-9a-f]{8,32}", str(marker.get("image_id", ""))):
        raise MustStartValidationError("image_id is invalid")
    role_prefix = f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
    if not isinstance(marker.get("worker_role_arn"), str) or not str(
        marker["worker_role_arn"]
    ).startswith(role_prefix):
        raise MustStartValidationError("worker_role_arn is invalid")
    for field in (
        "submission_submitted_at",
        "ec2_pending_time",
        "entrypoint_observed_at",
    ):
        canonical = _canonical_time(marker[field], field=field)
        if marker[field] != canonical:
            raise MustStartValidationError(f"{field} is not canonical")
    if _parse_time(
        str(marker["entrypoint_observed_at"]), field="entrypoint_observed_at"
    ) < _parse_time(str(marker["ec2_pending_time"]), field="ec2_pending_time"):
        raise MustStartValidationError("entrypoint precedes EC2 pending time")
    return marker


def build_timely_start_accepted(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    source_kind: str,
    source_key: str,
    source_version_id: str | None,
    source_etag: str,
    source_body_sha256: str,
    source_last_modified: datetime | str,
    started_at: datetime | str,
    accepted_at: datetime | str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    if source_kind not in {"controller-observation", "worker-latch"}:
        raise MustStartValidationError("source_kind is invalid")
    _require_scoped_key(source_key, run_id=run_id, field="source_key")
    if source_version_id is not None and not isinstance(source_version_id, str):
        raise MustStartValidationError("source_version_id is invalid")
    if not isinstance(source_etag, str) or not source_etag:
        raise MustStartValidationError("source_etag is invalid")
    modified = _canonical_time(
        source_last_modified, field="source_last_modified"
    )
    started = _canonical_time(started_at, field="started_at")
    deadline = str(authority["must_start_by"])
    if _parse_time(started, field="started_at") > _parse_time(
        deadline, field="must_start_by"
    ):
        raise MustStartValidationError("started_at is after must_start_by")
    if source_kind == "worker-latch" and _parse_time(
        modified, field="source_last_modified"
    ) > _parse_time(deadline, field="must_start_by"):
        raise MustStartValidationError("worker latch LastModified is late")
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_timely_start_accepted_v1",
        **authority,
        "source_kind": source_kind,
        "source_key": source_key,
        "source_version_id": source_version_id,
        "source_etag": source_etag,
        "source_body_sha256": _require_hex(
            source_body_sha256, field="source_body_sha256"
        ),
        "source_last_modified": modified,
        "started_at": started,
        "accepted_at": _canonical_time(accepted_at, field="accepted_at"),
    }
    accepted = str(body["accepted_at"])
    if _parse_time(accepted, field="accepted_at") < _parse_time(
        modified, field="source_last_modified"
    ):
        raise MustStartValidationError("accepted_at precedes source LastModified")
    if _parse_time(accepted, field="accepted_at") < _parse_time(
        started, field="started_at"
    ):
        raise MustStartValidationError("accepted_at precedes started_at")
    marker = {**body, "accepted_body_sha256": canonical_sha256(body)}
    return validate_timely_start_accepted(marker)


def validate_timely_start_accepted(
    value: Mapping[str, object],
) -> dict[str, object]:
    marker = _validate_digest(
        value,
        fields=_ACCEPTED_FIELDS,
        digest_field="accepted_body_sha256",
        record_type="glm52_sky_timely_start_accepted_v1",
    )
    source_kind = marker.get("source_kind")
    if source_kind not in {"controller-observation", "worker-latch"}:
        raise MustStartValidationError("source_kind is invalid")
    _require_scoped_key(
        marker.get("source_key"),
        run_id=str(marker["run_id"]),
        field="source_key",
    )
    if marker.get("source_version_id") is not None and not isinstance(
        marker.get("source_version_id"), str
    ):
        raise MustStartValidationError("source_version_id is invalid")
    if not isinstance(marker.get("source_etag"), str) or not marker.get(
        "source_etag"
    ):
        raise MustStartValidationError("source_etag is invalid")
    _require_hex(marker.get("source_body_sha256"), field="source_body_sha256")
    for field in (
        "source_last_modified",
        "started_at",
        "accepted_at",
    ):
        canonical = _canonical_time(marker[field], field=field)
        if marker[field] != canonical:
            raise MustStartValidationError(f"{field} is not canonical")
    deadline = _parse_time(str(marker["must_start_by"]), field="must_start_by")
    started = _parse_time(str(marker["started_at"]), field="started_at")
    modified = _parse_time(
        str(marker["source_last_modified"]), field="source_last_modified"
    )
    accepted = _parse_time(str(marker["accepted_at"]), field="accepted_at")
    if started > deadline:
        raise MustStartValidationError("started_at is after must_start_by")
    if source_kind == "worker-latch" and modified > deadline:
        raise MustStartValidationError("worker latch LastModified is late")
    if accepted < modified:
        raise MustStartValidationError("accepted_at precedes source LastModified")
    if accepted < started:
        raise MustStartValidationError("accepted_at precedes started_at")
    return marker


def build_must_start_cancel_superseded(
    *,
    run_id: str,
    managed_mode: str,
    account_id: str,
    region: str,
    bucket: str,
    descriptor_body_sha256: str,
    submission_body_sha256: str,
    sky_job_name: str,
    must_start_by: datetime | str,
    request_body_sha256: str,
    accepted_body_sha256: str,
    superseded_at: datetime | str,
) -> dict[str, object]:
    authority = _normalized_authority(
        run_id=run_id,
        managed_mode=managed_mode,
        account_id=account_id,
        region=region,
        bucket=bucket,
        descriptor_body_sha256=descriptor_body_sha256,
        submission_body_sha256=submission_body_sha256,
        sky_job_name=sky_job_name,
        must_start_by=must_start_by,
    )
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_cancel_superseded_v1",
        **authority,
        "request_body_sha256": _require_hex(
            request_body_sha256, field="request_body_sha256"
        ),
        "accepted_body_sha256": _require_hex(
            accepted_body_sha256, field="accepted_body_sha256"
        ),
        "superseded_at": _canonical_time(
            superseded_at, field="superseded_at"
        ),
    }
    marker = {**body, "superseded_body_sha256": canonical_sha256(body)}
    return validate_must_start_cancel_superseded(marker)


def validate_must_start_cancel_superseded(
    value: Mapping[str, object],
) -> dict[str, object]:
    marker = _validate_digest(
        value,
        fields=_SUPERSEDED_FIELDS,
        digest_field="superseded_body_sha256",
        record_type="glm52_sky_must_start_cancel_superseded_v1",
    )
    for field in ("request_body_sha256", "accepted_body_sha256"):
        _require_hex(marker.get(field), field=field)
    canonical = _canonical_time(marker["superseded_at"], field="superseded_at")
    if marker["superseded_at"] != canonical:
        raise MustStartValidationError("superseded_at is not canonical")
    return marker


def decide_controller_start_authority(
    *,
    now: datetime,
    must_start_by: datetime,
    status: str,
    start_at: datetime | None,
    starting_grace_seconds: int,
) -> MustStartDecision:
    if now.tzinfo is None or must_start_by.tzinfo is None:
        raise MustStartValidationError("decision times must be timezone-aware")
    if type(starting_grace_seconds) is not int or not (
        15 <= starting_grace_seconds <= 30
    ):
        raise MustStartValidationError(
            "starting_grace_seconds must be between 15 and 30"
        )
    if start_at is not None and start_at.tzinfo is None:
        raise MustStartValidationError("start_at must be timezone-aware")
    status = status.upper()
    if status == "PENDING":
        if now < must_start_by:
            return MustStartDecision("observe", "pending before deadline")
        return MustStartDecision("cancel", "pending at or after deadline")
    if status == "STARTING":
        grace_deadline = must_start_by.timestamp() + starting_grace_seconds
        if now.timestamp() <= grace_deadline:
            return MustStartDecision("grace", "starting within bounded grace")
        return MustStartDecision("cancel", "starting grace expired")
    if status in {"RUNNING", "RECOVERING"} | _TERMINAL_STATUSES:
        if start_at is None:
            return MustStartDecision("fail-closed", "real start time is missing")
        if start_at <= must_start_by:
            return MustStartDecision("accept", "real start was timely")
        return MustStartDecision("cancel", "real start was late")
    if status == "CANCELLING":
        return MustStartDecision("reconcile", "exact cancellation is pending")
    return MustStartDecision("fail-closed", "controller status is unknown")


def decide_must_start_action(
    *,
    now: datetime,
    must_start_by: datetime,
    expected_authority: Mapping[str, object],
    timely_start_latch: Mapping[str, object] | None,
    request_marker: Mapping[str, object] | None,
    completion_marker: Mapping[str, object] | None,
    controller_outcome: str | None,
) -> MustStartDecision:
    if now.tzinfo is None or must_start_by.tzinfo is None:
        raise MustStartValidationError("decision times must be timezone-aware")
    authority = _normalize_expected_authority(expected_authority)
    expected_deadline = _parse_time(
        str(authority["must_start_by"]),
        field="expected authority must_start_by",
    )
    if must_start_by.astimezone(timezone.utc) != expected_deadline:
        raise MustStartValidationError(
            "decision deadline does not match expected authority"
        )

    latch: dict[str, object] | None = None
    if timely_start_latch is not None:
        latch = validate_timely_start_latch(timely_start_latch)
        _require_expected_authority(
            latch,
            authority,
            label="timely-start latch",
        )
    request: dict[str, object] | None = None
    if request_marker is not None:
        request = validate_must_start_cancel_requested(request_marker)
        _require_expected_authority(
            request,
            authority,
            label="request marker",
        )
    completion: dict[str, object] | None = None
    if completion_marker is not None:
        completion = validate_must_start_cancel_completed(completion_marker)
        _require_expected_authority(
            completion,
            authority,
            label="completion marker",
        )
        if (
            request is None
            or completion["request_body_sha256"]
            != request["request_body_sha256"]
        ):
            raise MustStartValidationError(
                "completion marker does not bind exact request"
            )

    if now < must_start_by:
        return MustStartDecision("wait", "must_start_by is still in the future")
    if latch is not None:
        deadline = _parse_time(str(latch["must_start_by"]), field="must_start_by")
        started = _parse_time(str(latch["started_at"]), field="started_at")
        published = _parse_time(str(latch["published_at"]), field="published_at")
        if started <= deadline and published <= deadline:
            return MustStartDecision(
                "timely-started",
                "exact submission published a timely-start latch",
            )
    if completion is not None:
        return MustStartDecision("completed", "completion marker already exists")
    if request is None:
        return MustStartDecision("request-cancel", "deadline reached without latch")
    if controller_outcome in {"terminal", "cancelled"}:
        return MustStartDecision("complete", "exact job is terminal")
    if controller_outcome in {
        None,
        "pending",
        "starting",
        "running",
        "recovering",
        "controller-unknown",
        "not-found",
    }:
        return MustStartDecision("reconcile", "cancellation is not yet complete")
    raise MustStartValidationError("controller outcome is invalid")


__all__ = [
    "APPROVED_ACCOUNT_ID",
    "APPROVED_REGION",
    "MustStartDecision",
    "MustStartValidationError",
    "build_must_start_alert_delivered",
    "build_must_start_cancel_completed",
    "build_must_start_cancel_requested",
    "build_must_start_cancel_superseded",
    "build_must_start_controller_observation",
    "build_must_start_job_binding",
    "build_must_start_target_binding",
    "build_timely_start_accepted",
    "build_timely_start_latch",
    "build_worker_start_latch",
    "canonical_bytes",
    "canonical_sha256",
    "decide_controller_start_authority",
    "decide_must_start_action",
    "expected_sky_job_name",
    "validate_must_start_alert_delivered",
    "validate_must_start_cancel_completed",
    "validate_must_start_cancel_requested",
    "validate_must_start_cancel_superseded",
    "validate_must_start_controller_observation",
    "validate_must_start_job_binding",
    "validate_must_start_target_binding",
    "validate_timely_start_accepted",
    "validate_timely_start_latch",
    "validate_worker_start_latch",
]
