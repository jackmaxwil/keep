"""Pure, fail-closed SkyPilot submission intent lifecycle primitives."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Literal

from mlx_vq.quality.glm52_gpu_spend_snapshot import (
    validate_gpu_spend_snapshot,
)
from mlx_vq.quality.glm52_qualification_cache_seed import (
    accepted_s3_key,
    validate_qualification_cache_seed_accepted,
)
from mlx_vq.quality.glm52_qualification_submission_ready import (
    qualification_submission_ready_s3_key,
    validate_qualification_submission_ready,
)
from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_ACCOUNT_ID,
    APPROVED_INSTANCE_TYPE,
    APPROVED_REGION,
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import (
    validate_must_start_controller_observation,
)
from mlx_vq.quality.glm52_sky_must_start_dynamic import (
    DynamicJobBindingValidationError,
    dynamic_v2_canonical_bytes,
    dynamic_v2_job_binding_s3_key,
    validate_dynamic_v2_job_binding,
)

ManagedMode = Literal["production", "qualification", "cache-seed"]
SubmissionLifecycleAction = Literal[
    "create-intent",
    "reconcile-wait",
    "accept",
    "fail-closed",
    "idempotent-complete",
]

INTENT_RECORD_TYPE = "glm52_sky_submission_intent_v2"
INTENT_SCHEMA_VERSION = 2
INTENT_DIGEST_FIELD = "intent_body_sha256"
ACCEPTED_RECORD_TYPE = "glm52_sky_submission_accepted_v2"
ACCEPTED_SCHEMA_VERSION = 2
ACCEPTED_DIGEST_FIELD = "accepted_body_sha256"
EXPECTED_WORKSPACE = "default"
_MAX_READINESS_AGE = timedelta(minutes=5)

_MODES = frozenset({"production", "qualification", "cache-seed"})
_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
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
_CONTROLLER_JOB_FIELDS = frozenset(
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
_ACCEPTED_BODY_FIELDS = frozenset(
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
    }
)
_ACCEPTED_FIELDS = _ACCEPTED_BODY_FIELDS | {ACCEPTED_DIGEST_FIELD}


class SubmissionLifecycleValidationError(ValueError):
    """Submission authority or controller evidence is malformed or foreign."""


@dataclass(frozen=True)
class SubmissionLifecycleDecision:
    """One pure retry/reconciliation result; no action launches a job."""

    action: SubmissionLifecycleAction
    reason: str
    controller_job: dict[str, object] | None = None


def canonical_bytes(value: object) -> bytes:
    """Return the only JSON encoding used for lifecycle body digests."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SubmissionLifecycleValidationError(
            "submission lifecycle value is not canonical finite JSON"
        ) from error


def canonical_sha256(value: object) -> str:
    """Hash one canonical JSON value."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def expected_sky_job_name(run_id: str, managed_mode: str) -> str:
    """Return the one job name authorized for a run and managed mode."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise SubmissionLifecycleValidationError("run_id is invalid")
    if not isinstance(managed_mode, str) or managed_mode not in _MODES:
        raise SubmissionLifecycleValidationError("managed_mode is invalid")
    suffix = {
        "production": "",
        "qualification": "-qualification",
        "cache-seed": "-cache-seed",
    }[managed_mode]
    return f"{run_id}{suffix}"


def submission_intent_s3_key(
    *,
    run_id: str,
    managed_mode: str,
    intent_body_sha256: str,
) -> str:
    """Return the immutable content-addressed intent artifact key."""

    expected_sky_job_name(run_id, managed_mode)
    digest = _require_sha(intent_body_sha256, field=INTENT_DIGEST_FIELD)
    return (
        f"campaigns/{run_id}/submissions/{managed_mode}/intents/{digest}/"
        "SKYPILOT_SUBMISSION_INTENT.json"
    )


def submission_accepted_s3_key(
    *,
    run_id: str,
    managed_mode: str,
    accepted_body_sha256: str,
) -> str:
    """Return the immutable content-addressed accepted artifact key."""

    expected_sky_job_name(run_id, managed_mode)
    digest = _require_sha(accepted_body_sha256, field=ACCEPTED_DIGEST_FIELD)
    return (
        f"campaigns/{run_id}/submissions/{managed_mode}/accepted/{digest}/"
        "SKYPILOT_SUBMISSION_ACCEPTED.json"
    )


def _canonical_time(value: datetime | str, *, field: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise SubmissionLifecycleValidationError(
                f"{field} must be ISO-8601"
            ) from error
    else:
        raise SubmissionLifecycleValidationError(f"{field} must be ISO-8601")
    if parsed.tzinfo is None:
        raise SubmissionLifecycleValidationError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_time(value: datetime | str, *, field: str) -> datetime:
    return datetime.fromisoformat(_canonical_time(value, field=field))


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise SubmissionLifecycleValidationError(f"{field} must be a lowercase SHA-256")
    return value


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SubmissionLifecycleValidationError(f"{field} is invalid")
    return value


def _require_float(value: object, *, field: str, minimum: float = 0.0) -> float:
    if type(value) is not float or not math.isfinite(value) or value < minimum:
        raise SubmissionLifecycleValidationError(f"{field} is invalid")
    return value


def _require_canonical_file_sha256(
    value: object,
    *,
    record: Mapping[str, object],
    trailing_newline: bool,
    field: str,
) -> str:
    digest = _require_sha(value, field=field)
    raw = canonical_bytes(record) + (b"\n" if trailing_newline else b"")
    if digest != hashlib.sha256(raw).hexdigest():
        raise SubmissionLifecycleValidationError(
            f"{field} does not authenticate the exact canonical record file"
        )
    return digest


def _require_campaign_key(
    value: object,
    *,
    run_id: str,
    field: str,
    body_sha256: str | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise SubmissionLifecycleValidationError(f"{field} is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or not value.startswith(f"campaigns/{run_id}/")
    ):
        raise SubmissionLifecycleValidationError(f"{field} is not campaign-scoped")
    if body_sha256 is not None and body_sha256 not in path.parts:
        raise SubmissionLifecycleValidationError(f"{field} is not content-addressed")
    return value


def _validate_descriptor(
    descriptor: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(descriptor, Mapping):
        raise SubmissionLifecycleValidationError("descriptor is not an object")
    try:
        value = validate_sky_campaign_descriptor(descriptor)
    except ValueError as error:
        raise SubmissionLifecycleValidationError(str(error)) from error
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 2
        or value.get("account_id") != APPROVED_ACCOUNT_ID
        or value.get("provider") != "aws"
        or value.get("region") != APPROVED_REGION
        or value.get("instance_type") != APPROVED_INSTANCE_TYPE
        or type(value.get("instance_count")) is not int
        or value.get("instance_count") != 1
        or type(value.get("use_spot")) is not bool
        or value.get("use_spot") is not False
    ):
        raise SubmissionLifecycleValidationError(
            "descriptor is not the one on-demand p5.48xlarge AWS authority"
        )
    return value


def _validate_cache_acceptance(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
) -> dict[str, object]:
    try:
        acceptance = validate_qualification_cache_seed_accepted(value)
    except ValueError as error:
        raise SubmissionLifecycleValidationError(str(error)) from error
    authority = acceptance["seed_authority"]
    if not isinstance(authority, Mapping):
        raise SubmissionLifecycleValidationError(
            "cache-seed acceptance authority is invalid"
        )
    expected = {
        "account_id": descriptor["account_id"],
        "region": descriptor["region"],
        "bucket": descriptor["bucket"],
        "run_id": descriptor["run_id"],
        "approval_sha256": descriptor["approval_sha256"],
    }
    if any(
        authority.get(field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise SubmissionLifecycleValidationError(
            "cache-seed acceptance is foreign to the shared run authority"
        )
    return acceptance


def _validate_spend_snapshot(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SubmissionLifecycleValidationError("GPU spend snapshot is not an object")
    try:
        snapshot = validate_gpu_spend_snapshot(value)
    except ValueError as error:
        raise SubmissionLifecycleValidationError(str(error)) from error
    if type(snapshot.get("schema_version")) is not int:
        raise SubmissionLifecycleValidationError(
            "GPU spend snapshot schema version is invalid"
        )
    expected = {
        "run_id": descriptor["run_id"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
    }
    if any(
        snapshot.get(field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise SubmissionLifecycleValidationError(
            "GPU spend snapshot is foreign to the descriptor authority"
        )
    return snapshot


def _validate_submission_readiness(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    staged_readiness: Mapping[str, object],
    cache_seed_acceptance: Mapping[str, object],
    cache_seed_acceptance_key: str,
    cache_seed_acceptance_file_sha256: str,
    gpu_spend_snapshot: Mapping[str, object],
    gpu_spend_snapshot_key: str,
    gpu_spend_snapshot_sha256: str,
    rehearsal_evidence: Mapping[str, object],
    qualification_submission_ready_key: str,
    qualification_submission_ready_sha256: str,
    qualification_submission_ready_body_sha256: str,
    now: datetime | str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SubmissionLifecycleValidationError(
            "qualification submission readiness is not an object"
        )
    try:
        ready = validate_qualification_submission_ready(
            value,
            descriptor=descriptor,
            staged_readiness=staged_readiness,
            cache_seed_acceptance=cache_seed_acceptance,
            gpu_spend_snapshot=gpu_spend_snapshot,
            rehearsal_evidence=rehearsal_evidence,
            descriptor_file_sha256=descriptor_file_sha256,
            staged_readiness_key=value.get("staged_readiness_key"),  # type: ignore[arg-type]
            staged_readiness_file_sha256=value.get(  # type: ignore[arg-type]
                "staged_readiness_file_sha256"
            ),
            cache_seed_acceptance_key=cache_seed_acceptance_key,
            cache_seed_acceptance_file_sha256=(cache_seed_acceptance_file_sha256),
            gpu_spend_snapshot_key=gpu_spend_snapshot_key,
            gpu_spend_snapshot_sha256=gpu_spend_snapshot_sha256,
            rehearsal_evidence_key=value.get("rehearsal_evidence_key"),  # type: ignore[arg-type]
            rehearsal_evidence_sha256=value.get(  # type: ignore[arg-type]
                "rehearsal_evidence_sha256"
            ),
            now=now,
        )
        expected_key = qualification_submission_ready_s3_key(ready)
    except (TypeError, ValueError) as error:
        raise SubmissionLifecycleValidationError(str(error)) from error
    if qualification_submission_ready_key != expected_key:
        raise SubmissionLifecycleValidationError(
            "qualification submission readiness key is not exact and content-addressed"
        )
    _require_canonical_file_sha256(
        qualification_submission_ready_sha256,
        record=ready,
        trailing_newline=True,
        field="qualification_submission_ready_sha256",
    )
    body_sha256 = _require_sha(
        ready.get("readiness_body_sha256"),
        field="qualification_submission_ready_body_sha256",
    )
    if qualification_submission_ready_body_sha256 != body_sha256:
        raise SubmissionLifecycleValidationError(
            "qualification submission readiness body SHA-256 mismatch"
        )
    return ready


def _intent_body(
    *,
    descriptor: Mapping[str, object],
    managed_mode: str,
    descriptor_file_sha256: str,
    cache_seed_acceptance: Mapping[str, object],
    cache_seed_acceptance_key: str,
    cache_seed_acceptance_file_sha256: str,
    gpu_spend_snapshot: Mapping[str, object],
    gpu_spend_snapshot_key: str,
    gpu_spend_snapshot_sha256: str,
    staged_readiness: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    qualification_submission_ready: Mapping[str, object],
    qualification_submission_ready_key: str,
    qualification_submission_ready_sha256: str,
    qualification_submission_ready_body_sha256: str,
    sky_job_name: str,
    intent_at: datetime | str,
) -> dict[str, object]:
    descriptor_value = _validate_descriptor(descriptor)
    descriptor_file_digest = _require_canonical_file_sha256(
        descriptor_file_sha256,
        record=descriptor_value,
        trailing_newline=True,
        field="descriptor_file_sha256",
    )
    acceptance = _validate_cache_acceptance(
        cache_seed_acceptance, descriptor=descriptor_value
    )
    snapshot = _validate_spend_snapshot(
        gpu_spend_snapshot,
        descriptor=descriptor_value,
        descriptor_file_sha256=descriptor_file_digest,
    )
    run_id = str(descriptor_value["run_id"])
    expected_name = expected_sky_job_name(run_id, managed_mode)
    if sky_job_name != expected_name:
        raise SubmissionLifecycleValidationError(
            "SkyPilot job name does not match managed mode"
        )

    acceptance_body_sha256 = _require_sha(
        acceptance.get("acceptance_body_sha256"),
        field="cache_seed_acceptance_body_sha256",
    )
    exact_acceptance_key = accepted_s3_key(
        run_id=run_id,
        acceptance_body_sha256=acceptance_body_sha256,
    )
    if cache_seed_acceptance_key != exact_acceptance_key:
        raise SubmissionLifecycleValidationError(
            "cache_seed_acceptance_key is not the immutable accepted key"
        )
    cache_seed_acceptance_file_sha256 = _require_canonical_file_sha256(
        cache_seed_acceptance_file_sha256,
        record=acceptance,
        trailing_newline=True,
        field="cache_seed_acceptance_file_sha256",
    )
    snapshot_body_sha256 = _require_sha(
        snapshot.get("snapshot_body_sha256"),
        field="gpu_spend_snapshot_body_sha256",
    )
    _require_campaign_key(
        gpu_spend_snapshot_key,
        run_id=run_id,
        field="gpu_spend_snapshot_key",
        body_sha256=snapshot_body_sha256,
    )
    gpu_spend_snapshot_sha256 = _require_canonical_file_sha256(
        gpu_spend_snapshot_sha256,
        record=snapshot,
        trailing_newline=True,
        field="gpu_spend_snapshot_sha256",
    )

    intent_time = _canonical_time(intent_at, field="intent_at")
    must_start_by = _canonical_time(
        descriptor_value["must_start_by"], field="must_start_by"
    )
    if _parse_time(intent_time, field="intent_at") >= _parse_time(
        must_start_by, field="must_start_by"
    ):
        raise SubmissionLifecycleValidationError("submission intent deadline is stale")
    readiness = _validate_submission_readiness(
        qualification_submission_ready,
        descriptor=descriptor_value,
        descriptor_file_sha256=descriptor_file_digest,
        staged_readiness=staged_readiness,
        cache_seed_acceptance=acceptance,
        cache_seed_acceptance_key=cache_seed_acceptance_key,
        cache_seed_acceptance_file_sha256=(cache_seed_acceptance_file_sha256),
        gpu_spend_snapshot=snapshot,
        gpu_spend_snapshot_key=gpu_spend_snapshot_key,
        gpu_spend_snapshot_sha256=gpu_spend_snapshot_sha256,
        rehearsal_evidence=rehearsal_evidence,
        qualification_submission_ready_key=(qualification_submission_ready_key),
        qualification_submission_ready_sha256=(qualification_submission_ready_sha256),
        qualification_submission_ready_body_sha256=(
            qualification_submission_ready_body_sha256
        ),
        now=intent_time,
    )
    if (
        readiness.get("managed_mode") != managed_mode
        or readiness.get("sky_job_name") != sky_job_name
    ):
        raise SubmissionLifecycleValidationError(
            "qualification readiness does not authorize the managed job"
        )
    readiness_body_sha256 = _require_sha(
        readiness.get("readiness_body_sha256"),
        field="qualification_submission_ready_body_sha256",
    )
    for source_time, field in (
        (acceptance["accepted_at"], "cache-seed accepted_at"),
        (snapshot["observed_at"], "GPU spend snapshot observed_at"),
        (readiness["built_at"], "qualification readiness built_at"),
    ):
        if _parse_time(source_time, field=field) > _parse_time(
            intent_time, field="intent_at"
        ):
            raise SubmissionLifecycleValidationError(f"{field} follows intent_at")
    if (
        _parse_time(intent_time, field="intent_at")
        - _parse_time(
            readiness["built_at"],
            field="qualification readiness built_at",
        )
        > _MAX_READINESS_AGE
    ):
        raise SubmissionLifecycleValidationError(
            "qualification readiness is older than five minutes at intent_at"
        )

    remaining_seconds = _require_int(
        snapshot.get("remaining_gpu_seconds"),
        field="remaining_gpu_seconds",
    )
    allowance_seconds = _require_int(
        snapshot.get("qualification_allowance_seconds"),
        field="qualification_allowance_seconds",
    )
    if managed_mode == "qualification" and (
        remaining_seconds <= 0 or allowance_seconds <= 0
    ):
        raise SubmissionLifecycleValidationError(
            "qualification requires positive remaining allowance"
        )
    open_count = _require_int(
        snapshot.get("open_allocation_count"),
        field="open_allocation_count",
    )
    if open_count != 0:
        raise SubmissionLifecycleValidationError(
            "submission requires a closed GPU spend snapshot"
        )

    return {
        "schema_version": INTENT_SCHEMA_VERSION,
        "record_type": INTENT_RECORD_TYPE,
        "account_id": descriptor_value["account_id"],
        "region": descriptor_value["region"],
        "run_id": run_id,
        "managed_mode": managed_mode,
        "descriptor_key": descriptor_value["campaign_descriptor_key"],
        "descriptor_file_sha256": descriptor_file_digest,
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor_value["campaign_identity_sha256"],
        "approval_sha256": descriptor_value["approval_sha256"],
        "approval_body_sha256": _require_sha(
            snapshot.get("approval_body_sha256"),
            field="approval_body_sha256",
        ),
        "repo_tar_sha256": descriptor_value["repo_tar_sha256"],
        "cache_seed_acceptance_key": cache_seed_acceptance_key,
        "cache_seed_acceptance_file_sha256": (cache_seed_acceptance_file_sha256),
        "cache_seed_acceptance_body_sha256": (acceptance_body_sha256),
        "gpu_spend_snapshot_key": gpu_spend_snapshot_key,
        "gpu_spend_snapshot_sha256": gpu_spend_snapshot_sha256,
        "gpu_spend_snapshot_body_sha256": snapshot_body_sha256,
        "gpu_spend_ledger_tip_record_sha256": _require_sha(
            snapshot.get("gpu_spend_ledger_tip_record_sha256"),
            field="gpu_spend_ledger_tip_record_sha256",
        ),
        "remaining_gpu_seconds": remaining_seconds,
        "remaining_gpu_cost_usd": _require_float(
            snapshot.get("remaining_gpu_cost_usd"),
            field="remaining_gpu_cost_usd",
        ),
        "qualification_allowance_seconds": allowance_seconds,
        "qualification_allowance_cost_usd": _require_float(
            snapshot.get("qualification_allowance_cost_usd"),
            field="qualification_allowance_cost_usd",
        ),
        "open_allocation_count": open_count,
        "qualification_submission_ready_key": (qualification_submission_ready_key),
        "qualification_submission_ready_sha256": (
            qualification_submission_ready_sha256
        ),
        "qualification_submission_ready_body_sha256": (readiness_body_sha256),
        "sky_job_name": sky_job_name,
        "must_start_by": must_start_by,
        "intent_at": intent_time,
    }


def build_submission_intent(
    *,
    descriptor: Mapping[str, object],
    managed_mode: str,
    descriptor_file_sha256: str,
    cache_seed_acceptance: Mapping[str, object],
    cache_seed_acceptance_key: str,
    cache_seed_acceptance_file_sha256: str,
    gpu_spend_snapshot: Mapping[str, object],
    gpu_spend_snapshot_key: str,
    gpu_spend_snapshot_sha256: str,
    staged_readiness: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    qualification_submission_ready: Mapping[str, object],
    qualification_submission_ready_key: str,
    qualification_submission_ready_sha256: str,
    qualification_submission_ready_body_sha256: str,
    sky_job_name: str,
    intent_at: datetime | str,
) -> dict[str, object]:
    """Build one exact immutable intent before any launch attempt."""

    body = _intent_body(
        descriptor=descriptor,
        managed_mode=managed_mode,
        descriptor_file_sha256=descriptor_file_sha256,
        cache_seed_acceptance=cache_seed_acceptance,
        cache_seed_acceptance_key=cache_seed_acceptance_key,
        cache_seed_acceptance_file_sha256=(cache_seed_acceptance_file_sha256),
        gpu_spend_snapshot=gpu_spend_snapshot,
        gpu_spend_snapshot_key=gpu_spend_snapshot_key,
        gpu_spend_snapshot_sha256=gpu_spend_snapshot_sha256,
        staged_readiness=staged_readiness,
        rehearsal_evidence=rehearsal_evidence,
        qualification_submission_ready=qualification_submission_ready,
        qualification_submission_ready_key=(qualification_submission_ready_key),
        qualification_submission_ready_sha256=(qualification_submission_ready_sha256),
        qualification_submission_ready_body_sha256=(
            qualification_submission_ready_body_sha256
        ),
        sky_job_name=sky_job_name,
        intent_at=intent_at,
    )
    return {**body, INTENT_DIGEST_FIELD: canonical_sha256(body)}


def validate_submission_intent(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
    cache_seed_acceptance: Mapping[str, object],
    gpu_spend_snapshot: Mapping[str, object],
    staged_readiness: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    qualification_submission_ready: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate an intent against its descriptor and source records."""

    if not isinstance(value, Mapping) or set(value) != _INTENT_FIELDS:
        raise SubmissionLifecycleValidationError("submission intent schema mismatch")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != INTENT_SCHEMA_VERSION
        or value.get("record_type") != INTENT_RECORD_TYPE
    ):
        raise SubmissionLifecycleValidationError("submission intent schema mismatch")
    body = dict(value)
    digest = body.pop(INTENT_DIGEST_FIELD)
    _require_sha(digest, field=INTENT_DIGEST_FIELD)
    if digest != canonical_sha256(body):
        raise SubmissionLifecycleValidationError(
            "submission intent body SHA-256 mismatch"
        )
    expected = _intent_body(
        descriptor=descriptor,
        managed_mode=str(value["managed_mode"]),
        descriptor_file_sha256=str(value["descriptor_file_sha256"]),
        cache_seed_acceptance=cache_seed_acceptance,
        cache_seed_acceptance_key=str(value["cache_seed_acceptance_key"]),
        cache_seed_acceptance_file_sha256=str(
            value["cache_seed_acceptance_file_sha256"]
        ),
        gpu_spend_snapshot=gpu_spend_snapshot,
        gpu_spend_snapshot_key=str(value["gpu_spend_snapshot_key"]),
        gpu_spend_snapshot_sha256=str(value["gpu_spend_snapshot_sha256"]),
        staged_readiness=staged_readiness,
        rehearsal_evidence=rehearsal_evidence,
        qualification_submission_ready=qualification_submission_ready,
        qualification_submission_ready_key=str(
            value["qualification_submission_ready_key"]
        ),
        qualification_submission_ready_sha256=str(
            value["qualification_submission_ready_sha256"]
        ),
        qualification_submission_ready_body_sha256=str(
            value["qualification_submission_ready_body_sha256"]
        ),
        sky_job_name=str(value["sky_job_name"]),
        intent_at=value["intent_at"],  # type: ignore[arg-type]
    )
    if body != expected:
        raise SubmissionLifecycleValidationError("submission intent authority drift")
    return dict(value)


def _validate_controller_job(
    value: Mapping[str, object],
    *,
    intent: Mapping[str, object],
    descriptor: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CONTROLLER_JOB_FIELDS:
        raise SubmissionLifecycleValidationError("controller job schema mismatch")
    sky_job_id = _require_int(value.get("sky_job_id"), field="sky_job_id", minimum=1)
    if (
        not isinstance(value.get("sky_job_name"), str)
        or value.get("sky_job_name") != intent["sky_job_name"]
    ):
        raise SubmissionLifecycleValidationError(
            "controller job name does not match intent"
        )
    if (
        not isinstance(value.get("workspace"), str)
        or value.get("workspace") != EXPECTED_WORKSPACE
    ):
        raise SubmissionLifecycleValidationError(
            "controller job workspace does not match exact authority"
        )
    if (
        not isinstance(value.get("controller_identity"), str)
        or value.get("controller_identity") != descriptor["controller_identity"]
    ):
        raise SubmissionLifecycleValidationError(
            "controller job identity does not match descriptor"
        )
    status = value.get("controller_status")
    if not isinstance(status, str) or status not in _CONTROLLER_STATUSES:
        raise SubmissionLifecycleValidationError("controller job status is invalid")
    submitted_at = _canonical_time(
        value.get("controller_submitted_at"),  # type: ignore[arg-type]
        field="controller_submitted_at",
    )
    if value.get("controller_submitted_at") != submitted_at:
        raise SubmissionLifecycleValidationError(
            "controller_submitted_at is not canonical"
        )
    if _parse_time(submitted_at, field="controller_submitted_at") < _parse_time(
        str(intent["intent_at"]), field="intent_at"
    ):
        raise SubmissionLifecycleValidationError(
            "controller job predates submission intent"
        )
    if _parse_time(submitted_at, field="controller_submitted_at") >= _parse_time(
        str(intent["must_start_by"]), field="must_start_by"
    ):
        raise SubmissionLifecycleValidationError(
            "controller job submission missed must_start_by"
        )
    return {
        "sky_job_id": sky_job_id,
        "sky_job_name": value["sky_job_name"],
        "workspace": value["workspace"],
        "controller_submitted_at": submitted_at,
        "controller_status": status,
        "controller_identity": value["controller_identity"],
    }


def _validate_controller_receipts(
    *,
    intent: Mapping[str, object],
    descriptor: Mapping[str, object],
    controller_baseline: Mapping[str, object],
    submission_acquisition: Mapping[str, object],
    controller_observation: Mapping[str, object],
    controller_observation_key: str,
    controller_observation_file_sha256: str,
    job_binding: Mapping[str, object],
    job_binding_key: str,
    job_binding_file_sha256: str,
    accepted_at: str,
) -> dict[str, object]:
    if not isinstance(controller_observation, Mapping):
        raise SubmissionLifecycleValidationError(
            "controller observation is not an object"
        )
    if not isinstance(job_binding, Mapping):
        raise SubmissionLifecycleValidationError(
            "must-start job binding is not an object"
        )
    try:
        observation = validate_must_start_controller_observation(controller_observation)
    except ValueError as error:
        raise SubmissionLifecycleValidationError(str(error)) from error

    expected = {
        "run_id": intent["run_id"],
        "managed_mode": intent["managed_mode"],
        "account_id": intent["account_id"],
        "region": intent["region"],
        "bucket": descriptor["bucket"],
        "descriptor_body_sha256": intent["descriptor_body_sha256"],
        "submission_body_sha256": intent[INTENT_DIGEST_FIELD],
        "sky_job_name": intent["sky_job_name"],
        "must_start_by": intent["must_start_by"],
        "workspace": EXPECTED_WORKSPACE,
    }
    if any(
        observation.get(field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise SubmissionLifecycleValidationError(
            "controller observation is foreign to the submission intent"
        )

    controller_role = str(descriptor["controller_identity"])
    role_prefix = f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
    if not controller_role.startswith(role_prefix):
        raise SubmissionLifecycleValidationError(
            "descriptor controller identity is invalid"
        )
    expected_profile = (
        f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:instance-profile/"
        f"{controller_role.removeprefix(role_prefix)}"
    )
    if observation.get("controller_profile_arn") != expected_profile:
        raise SubmissionLifecycleValidationError(
            "controller observation instance profile is foreign"
        )

    submitted_at = str(observation["submitted_at"])
    observed_at = str(observation["observed_at"])
    if _parse_time(submitted_at, field="controller submitted_at") < _parse_time(
        str(intent["intent_at"]), field="intent_at"
    ):
        raise SubmissionLifecycleValidationError(
            "controller observation submission predates intent creation"
        )
    if _parse_time(submitted_at, field="controller submitted_at") >= _parse_time(
        str(intent["must_start_by"]), field="must_start_by"
    ):
        raise SubmissionLifecycleValidationError(
            "controller observation submission missed must_start_by"
        )
    if _parse_time(observed_at, field="controller observed_at") < _parse_time(
        str(intent["intent_at"]), field="intent_at"
    ):
        raise SubmissionLifecycleValidationError(
            "controller observation predates intent creation"
        )
    if _parse_time(observed_at, field="controller observed_at") < _parse_time(
        submitted_at,
        field="controller_submitted_at",
    ):
        raise SubmissionLifecycleValidationError(
            "controller observation predates controller submission"
        )

    observation_body_sha256 = _require_sha(
        observation.get("observation_body_sha256"),
        field="controller_observation_body_sha256",
    )
    expected_key = (
        f"campaigns/{intent['run_id']}/monitor/must-start/"
        f"{intent['managed_mode']}/{intent[INTENT_DIGEST_FIELD]}/"
        f"observations/{observation_body_sha256}.json"
    )
    if controller_observation_key != expected_key:
        raise SubmissionLifecycleValidationError(
            "controller observation key is not exact and content-addressed"
        )
    file_sha256 = _require_sha(
        controller_observation_file_sha256,
        field="controller_observation_file_sha256",
    )
    expected_file_sha256 = hashlib.sha256(canonical_bytes(observation)).hexdigest()
    if file_sha256 != expected_file_sha256:
        raise SubmissionLifecycleValidationError(
            "controller observation file SHA-256 mismatch"
        )

    if not isinstance(controller_baseline, Mapping):
        raise SubmissionLifecycleValidationError("controller baseline is not an object")
    if not isinstance(submission_acquisition, Mapping):
        raise SubmissionLifecycleValidationError(
            "submission acquisition is not an object"
        )
    baseline_history = controller_baseline.get("exact_name_history")
    if type(baseline_history) is not list:
        raise SubmissionLifecycleValidationError(
            "controller baseline history is not an exact list"
        )
    observation_row = {
        "sky_job_id": observation["target_job_id"],
        "sky_job_name": observation["sky_job_name"],
        "workspace": observation["workspace"],
        "controller_submitted_at": submitted_at,
        "controller_status": observation["status"],
        "controller_identity": descriptor["controller_identity"],
    }
    current_history = [*baseline_history, observation_row]
    try:
        binding = validate_dynamic_v2_job_binding(
            job_binding,
            intent=intent,
            controller_baseline=controller_baseline,
            acquisition=submission_acquisition,
            descriptor_controller_identity=controller_role,
            current_exact_name_history=current_history,
            now=accepted_at,
        )
        expected_binding_key = dynamic_v2_job_binding_s3_key(intent=intent)
    except DynamicJobBindingValidationError as error:
        raise SubmissionLifecycleValidationError(str(error)) from error
    if job_binding_key != expected_binding_key:
        raise SubmissionLifecycleValidationError(
            "dynamic-v2 job binding key is not exact and submission-addressed"
        )
    binding_file_sha256 = _require_sha(
        job_binding_file_sha256,
        field="job_binding_file_sha256",
    )
    expected_binding_file_sha256 = hashlib.sha256(
        dynamic_v2_canonical_bytes(binding)
    ).hexdigest()
    if binding_file_sha256 != expected_binding_file_sha256:
        raise SubmissionLifecycleValidationError(
            "dynamic-v2 job binding file SHA-256 mismatch"
        )
    bound_at = str(binding["bound_at"])
    if _parse_time(bound_at, field="job binding bound_at") > _parse_time(
        observed_at,
        field="controller_observed_at",
    ):
        raise SubmissionLifecycleValidationError(
            "dynamic-v2 job binding follows controller observation"
        )
    expected_controller_coordinates = {
        "controller_instance_id": controller_baseline.get("controller_instance_id"),
        "controller_instance_type": controller_baseline.get("controller_instance_type"),
        "controller_profile_arn": controller_baseline.get("controller_profile_arn"),
        "controller_cluster_name": controller_baseline.get("controller_cluster_name"),
    }
    if any(
        observation.get(field) != expected_value
        for field, expected_value in expected_controller_coordinates.items()
    ):
        raise SubmissionLifecycleValidationError(
            "controller observation coordinates are foreign to the baseline"
        )

    return {
        "sky_job_id": observation["target_job_id"],
        "sky_job_name": observation["sky_job_name"],
        "workspace": observation["workspace"],
        "controller_submitted_at": submitted_at,
        "controller_status": observation["status"],
        "controller_identity": descriptor["controller_identity"],
        "controller_instance_id": observation["controller_instance_id"],
        "controller_instance_type": observation["controller_instance_type"],
        "controller_profile_arn": observation["controller_profile_arn"],
        "controller_cluster_name": observation["controller_cluster_name"],
        "controller_observed_at": observed_at,
        "controller_observation_key": controller_observation_key,
        "controller_observation_file_sha256": file_sha256,
        "controller_observation_body_sha256": observation_body_sha256,
        "controller_observation": dict(observation),
        "job_binding_key": job_binding_key,
        "job_binding_file_sha256": binding_file_sha256,
        "job_binding_body_sha256": binding["job_binding_body_sha256"],
        "job_binding": dict(binding),
    }


def _accepted_body(
    *,
    intent: Mapping[str, object],
    descriptor: Mapping[str, object],
    cache_seed_acceptance: Mapping[str, object],
    gpu_spend_snapshot: Mapping[str, object],
    staged_readiness: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    qualification_submission_ready: Mapping[str, object],
    controller_baseline: Mapping[str, object],
    submission_acquisition: Mapping[str, object],
    controller_job: Mapping[str, object],
    controller_observation: Mapping[str, object],
    controller_observation_key: str,
    controller_observation_file_sha256: str,
    job_binding: Mapping[str, object],
    job_binding_key: str,
    job_binding_file_sha256: str,
    accepted_at: datetime | str,
) -> dict[str, object]:
    intent_value = validate_submission_intent(
        intent,
        descriptor=descriptor,
        cache_seed_acceptance=cache_seed_acceptance,
        gpu_spend_snapshot=gpu_spend_snapshot,
        staged_readiness=staged_readiness,
        rehearsal_evidence=rehearsal_evidence,
        qualification_submission_ready=qualification_submission_ready,
    )
    descriptor_value = _validate_descriptor(descriptor)
    reconciled_job = _validate_controller_job(
        controller_job,
        intent=intent_value,
        descriptor=descriptor_value,
    )
    accepted_time = _canonical_time(accepted_at, field="accepted_at")
    if not isinstance(job_binding, Mapping):
        raise SubmissionLifecycleValidationError(
            "must-start job binding is not an object"
        )
    binding_time = _canonical_time(
        job_binding.get("bound_at"),  # type: ignore[arg-type]
        field="job binding bound_at",
    )
    if not isinstance(controller_observation, Mapping):
        raise SubmissionLifecycleValidationError(
            "controller observation is not an object"
        )
    observation_time = _canonical_time(
        controller_observation.get("observed_at"),  # type: ignore[arg-type]
        field="controller_observed_at",
    )
    deterministic_time = max(
        (binding_time, observation_time),
        key=lambda value: _parse_time(value, field="accepted_at source"),
    )
    if accepted_time != deterministic_time:
        raise SubmissionLifecycleValidationError(
            "accepted_at must equal the deterministic binding/observation maximum"
        )
    receipt = _validate_controller_receipts(
        intent=intent_value,
        descriptor=descriptor_value,
        controller_baseline=controller_baseline,
        submission_acquisition=submission_acquisition,
        controller_observation=controller_observation,
        controller_observation_key=controller_observation_key,
        controller_observation_file_sha256=(controller_observation_file_sha256),
        job_binding=job_binding,
        job_binding_key=job_binding_key,
        job_binding_file_sha256=job_binding_file_sha256,
        accepted_at=accepted_time,
    )
    for field in _CONTROLLER_JOB_FIELDS:
        if receipt[field] != reconciled_job[field]:
            raise SubmissionLifecycleValidationError(
                "controller observation does not match the reconciled controller job"
            )
    if _parse_time(accepted_time, field="accepted_at") < _parse_time(
        str(receipt["controller_submitted_at"]),
        field="controller_submitted_at",
    ):
        raise SubmissionLifecycleValidationError(
            "accepted_at precedes controller submission"
        )
    if _parse_time(accepted_time, field="accepted_at") < _parse_time(
        str(receipt["controller_observed_at"]),
        field="controller_observed_at",
    ):
        raise SubmissionLifecycleValidationError(
            "accepted_at precedes controller observation"
        )
    return {
        "schema_version": ACCEPTED_SCHEMA_VERSION,
        "record_type": ACCEPTED_RECORD_TYPE,
        "intent_body_sha256": intent_value[INTENT_DIGEST_FIELD],
        **receipt,
        "accepted_at": accepted_time,
    }


def build_submission_accepted(
    *,
    intent: Mapping[str, object],
    descriptor: Mapping[str, object],
    cache_seed_acceptance: Mapping[str, object],
    gpu_spend_snapshot: Mapping[str, object],
    staged_readiness: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    qualification_submission_ready: Mapping[str, object],
    controller_baseline: Mapping[str, object],
    submission_acquisition: Mapping[str, object],
    controller_job: Mapping[str, object],
    controller_observation: Mapping[str, object],
    controller_observation_key: str,
    controller_observation_file_sha256: str,
    job_binding: Mapping[str, object],
    job_binding_key: str,
    job_binding_file_sha256: str,
) -> dict[str, object]:
    """Bind one exact reconciled job to durable must-start receipts."""

    if not isinstance(job_binding, Mapping):
        raise SubmissionLifecycleValidationError(
            "must-start job binding is not an object"
        )
    body = _accepted_body(
        intent=intent,
        descriptor=descriptor,
        cache_seed_acceptance=cache_seed_acceptance,
        gpu_spend_snapshot=gpu_spend_snapshot,
        staged_readiness=staged_readiness,
        rehearsal_evidence=rehearsal_evidence,
        qualification_submission_ready=qualification_submission_ready,
        controller_baseline=controller_baseline,
        submission_acquisition=submission_acquisition,
        controller_job=controller_job,
        controller_observation=controller_observation,
        controller_observation_key=controller_observation_key,
        controller_observation_file_sha256=(controller_observation_file_sha256),
        job_binding=job_binding,
        job_binding_key=job_binding_key,
        job_binding_file_sha256=job_binding_file_sha256,
        accepted_at=max(
            (
                _canonical_time(
                    job_binding.get("bound_at"),  # type: ignore[arg-type]
                    field="job binding bound_at",
                ),
                _canonical_time(
                    controller_observation.get("observed_at"),  # type: ignore[arg-type]
                    field="controller_observed_at",
                ),
            ),
            key=lambda value: _parse_time(value, field="accepted_at source"),
        ),
    )
    return {**body, ACCEPTED_DIGEST_FIELD: canonical_sha256(body)}


def validate_submission_accepted(
    value: Mapping[str, object],
    *,
    intent: Mapping[str, object],
    descriptor: Mapping[str, object],
    cache_seed_acceptance: Mapping[str, object],
    gpu_spend_snapshot: Mapping[str, object],
    staged_readiness: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    qualification_submission_ready: Mapping[str, object],
    controller_baseline: Mapping[str, object],
    submission_acquisition: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate self-contained accepted evidence and embedded receipts."""

    if not isinstance(value, Mapping) or set(value) != _ACCEPTED_FIELDS:
        raise SubmissionLifecycleValidationError("submission accepted schema mismatch")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != ACCEPTED_SCHEMA_VERSION
        or value.get("record_type") != ACCEPTED_RECORD_TYPE
    ):
        raise SubmissionLifecycleValidationError("submission accepted schema mismatch")
    body = dict(value)
    digest = body.pop(ACCEPTED_DIGEST_FIELD)
    _require_sha(digest, field=ACCEPTED_DIGEST_FIELD)
    if digest != canonical_sha256(body):
        raise SubmissionLifecycleValidationError(
            "submission accepted body SHA-256 mismatch"
        )
    expected = _accepted_body(
        intent=intent,
        descriptor=descriptor,
        cache_seed_acceptance=cache_seed_acceptance,
        gpu_spend_snapshot=gpu_spend_snapshot,
        staged_readiness=staged_readiness,
        rehearsal_evidence=rehearsal_evidence,
        qualification_submission_ready=qualification_submission_ready,
        controller_baseline=controller_baseline,
        submission_acquisition=submission_acquisition,
        controller_job={field: value.get(field) for field in _CONTROLLER_JOB_FIELDS},
        controller_observation=value.get("controller_observation"),  # type: ignore[arg-type]
        controller_observation_key=value.get("controller_observation_key"),  # type: ignore[arg-type]
        controller_observation_file_sha256=value.get(  # type: ignore[arg-type]
            "controller_observation_file_sha256"
        ),
        job_binding=value.get("job_binding"),  # type: ignore[arg-type]
        job_binding_key=value.get("job_binding_key"),  # type: ignore[arg-type]
        job_binding_file_sha256=value.get(  # type: ignore[arg-type]
            "job_binding_file_sha256"
        ),
        accepted_at=value["accepted_at"],  # type: ignore[arg-type]
    )
    if body != expected:
        raise SubmissionLifecycleValidationError("submission accepted authority drift")
    return dict(value)


def _decision(
    action: SubmissionLifecycleAction,
    reason: str,
    *,
    controller_job: dict[str, object] | None = None,
) -> SubmissionLifecycleDecision:
    return SubmissionLifecycleDecision(
        action=action,
        reason=reason,
        controller_job=controller_job,
    )


def decide_submission_lifecycle(
    *,
    intent: Mapping[str, object] | None,
    accepted: Mapping[str, object] | None,
    descriptor: Mapping[str, object],
    cache_seed_acceptance: Mapping[str, object],
    gpu_spend_snapshot: Mapping[str, object],
    staged_readiness: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    qualification_submission_ready: Mapping[str, object],
    qualification_submission_ready_key: str,
    qualification_submission_ready_sha256: str,
    accepted_controller_observation: Mapping[str, object] | None,
    controller_jobs: object,
    now: datetime | str,
    controller_baseline: Mapping[str, object] | None = None,
    submission_acquisition: Mapping[str, object] | None = None,
) -> SubmissionLifecycleDecision:
    """Choose the only safe retry action without calling SkyPilot or AWS."""

    try:
        if not isinstance(qualification_submission_ready, Mapping):
            raise SubmissionLifecycleValidationError(
                "qualification submission readiness is not an object"
            )
        if not isinstance(gpu_spend_snapshot, Mapping):
            raise SubmissionLifecycleValidationError(
                "GPU spend snapshot is not an object"
            )
        descriptor_value = _validate_descriptor(descriptor)
        _validate_cache_acceptance(cache_seed_acceptance, descriptor=descriptor_value)
        descriptor_file_sha256 = (
            intent.get("descriptor_file_sha256")
            if isinstance(intent, Mapping)
            else gpu_spend_snapshot.get("descriptor_sha256")
        )
        _validate_spend_snapshot(
            gpu_spend_snapshot,
            descriptor=descriptor_value,
            descriptor_file_sha256=_require_sha(
                descriptor_file_sha256,
                field="descriptor_file_sha256",
            ),
        )
        source = (
            intent if isinstance(intent, Mapping) else qualification_submission_ready
        )
        readiness_observed_at = (
            intent.get("intent_at") if isinstance(intent, Mapping) else now
        )
        _validate_submission_readiness(
            qualification_submission_ready,
            descriptor=descriptor_value,
            descriptor_file_sha256=_require_sha(
                descriptor_file_sha256,
                field="descriptor_file_sha256",
            ),
            staged_readiness=staged_readiness,
            cache_seed_acceptance=cache_seed_acceptance,
            cache_seed_acceptance_key=source.get("cache_seed_acceptance_key"),  # type: ignore[arg-type]
            cache_seed_acceptance_file_sha256=source.get(  # type: ignore[arg-type]
                "cache_seed_acceptance_file_sha256"
            ),
            gpu_spend_snapshot=gpu_spend_snapshot,
            gpu_spend_snapshot_key=source.get("gpu_spend_snapshot_key"),  # type: ignore[arg-type]
            gpu_spend_snapshot_sha256=source.get("gpu_spend_snapshot_sha256"),  # type: ignore[arg-type]
            rehearsal_evidence=rehearsal_evidence,
            qualification_submission_ready_key=(qualification_submission_ready_key),
            qualification_submission_ready_sha256=(
                qualification_submission_ready_sha256
            ),
            qualification_submission_ready_body_sha256=_require_sha(
                qualification_submission_ready.get("readiness_body_sha256"),
                field="qualification_submission_ready_body_sha256",
            ),
            now=readiness_observed_at,  # type: ignore[arg-type]
        )
    except (SubmissionLifecycleValidationError, ValueError) as error:
        return _decision(
            "fail-closed",
            f"submission source authority is invalid: {error}",
        )

    if not isinstance(controller_jobs, (list, tuple)):
        return _decision(
            "fail-closed",
            "controller job history must be an exact list or tuple",
        )
    jobs = tuple(controller_jobs)

    if intent is None:
        if accepted is not None:
            return _decision(
                "fail-closed",
                "accepted evidence exists without its submission intent",
            )
        if accepted_controller_observation is not None:
            return _decision(
                "fail-closed",
                "controller observation exists without an accepted record",
            )
        if jobs:
            return _decision(
                "fail-closed",
                "controller history exists before submission intent",
            )
        return _decision(
            "create-intent",
            "no immutable submission intent exists",
        )

    try:
        intent_value = validate_submission_intent(
            intent,
            descriptor=descriptor_value,
            cache_seed_acceptance=cache_seed_acceptance,
            gpu_spend_snapshot=gpu_spend_snapshot,
            staged_readiness=staged_readiness,
            rehearsal_evidence=rehearsal_evidence,
            qualification_submission_ready=qualification_submission_ready,
        )
    except (SubmissionLifecycleValidationError, ValueError) as error:
        return _decision(
            "fail-closed",
            f"submission intent is invalid: {error}",
        )

    if accepted is not None:
        if not isinstance(controller_baseline, Mapping) or not isinstance(
            submission_acquisition, Mapping
        ):
            return _decision(
                "fail-closed",
                "accepted evidence requires exact baseline and acquisition ancestry",
            )
        if not isinstance(accepted, Mapping):
            return _decision(
                "fail-closed",
                "submission accepted evidence is not an object",
            )
        if len(jobs) > 1:
            return _decision(
                "fail-closed",
                "controller history is ambiguous for an accepted submission",
            )
        if len(jobs) == 1:
            accepted_controller_job = jobs[0]
            if not isinstance(accepted_controller_job, Mapping):
                return _decision(
                    "fail-closed",
                    "controller history contains a non-object job",
                )
        else:
            accepted_controller_job = {
                field: accepted.get(field) for field in _CONTROLLER_JOB_FIELDS
            }
        if (
            accepted_controller_observation is not None
            and accepted_controller_observation
            != accepted.get("controller_observation")
        ):
            return _decision(
                "fail-closed",
                "external controller observation conflicts with accepted evidence",
            )
        try:
            validate_submission_accepted(
                accepted,
                intent=intent_value,
                descriptor=descriptor_value,
                cache_seed_acceptance=cache_seed_acceptance,
                gpu_spend_snapshot=gpu_spend_snapshot,
                staged_readiness=staged_readiness,
                rehearsal_evidence=rehearsal_evidence,
                qualification_submission_ready=qualification_submission_ready,
                controller_baseline=controller_baseline,
                submission_acquisition=submission_acquisition,
            )
            normalized_job = _validate_controller_job(
                accepted_controller_job,
                intent=intent_value,
                descriptor=descriptor_value,
            )
            if any(
                normalized_job[field] != accepted.get(field)
                for field in _CONTROLLER_JOB_FIELDS
            ):
                raise SubmissionLifecycleValidationError(
                    "controller history conflicts with accepted evidence"
                )
        except (SubmissionLifecycleValidationError, ValueError) as error:
            return _decision(
                "fail-closed",
                f"submission accepted evidence is invalid: {error}",
            )
        return _decision(
            "idempotent-complete",
            "the exact controller job is already accepted",
        )
    if accepted_controller_observation is not None:
        return _decision(
            "fail-closed",
            "controller observation exists without an accepted record",
        )

    if len(jobs) > 1:
        return _decision(
            "fail-closed",
            "controller history is ambiguous for this submission intent",
        )
    if len(jobs) == 1:
        job = jobs[0]
        if not isinstance(job, Mapping):
            return _decision(
                "fail-closed",
                "controller history contains a non-object job",
            )
        try:
            normalized_job = _validate_controller_job(
                job,
                intent=intent_value,
                descriptor=descriptor_value,
            )
        except (SubmissionLifecycleValidationError, ValueError) as error:
            return _decision(
                "fail-closed",
                f"controller job is foreign to the intent: {error}",
            )
        return _decision(
            "accept",
            "one exact controller job matches the submission intent",
            controller_job=normalized_job,
        )

    try:
        observed_now = _parse_time(now, field="now")
    except SubmissionLifecycleValidationError as error:
        return _decision("fail-closed", f"current time is invalid: {error}")
    if observed_now >= _parse_time(
        str(intent_value["must_start_by"]), field="must_start_by"
    ):
        return _decision(
            "fail-closed",
            "submission deadline passed without an exact controller job",
        )
    return _decision(
        "reconcile-wait",
        "intent exists but no exact controller job is observable; never relaunch",
    )


__all__ = [
    "ACCEPTED_DIGEST_FIELD",
    "ACCEPTED_RECORD_TYPE",
    "ACCEPTED_SCHEMA_VERSION",
    "EXPECTED_WORKSPACE",
    "INTENT_DIGEST_FIELD",
    "INTENT_RECORD_TYPE",
    "INTENT_SCHEMA_VERSION",
    "ManagedMode",
    "SubmissionLifecycleAction",
    "SubmissionLifecycleDecision",
    "SubmissionLifecycleValidationError",
    "build_submission_accepted",
    "build_submission_intent",
    "canonical_bytes",
    "canonical_sha256",
    "decide_submission_lifecycle",
    "expected_sky_job_name",
    "submission_accepted_s3_key",
    "submission_intent_s3_key",
    "validate_submission_accepted",
    "validate_submission_intent",
]
