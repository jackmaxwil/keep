"""Fail-closed authority for a post-seed GLM-5.2 qualification submission.

The record built here is deliberately redundant.  It closes the immutable
cache-seed authority, the post-seed descriptor, staged-control-plane proof,
cumulative spend snapshot, and rehearsal evidence into one short-lived
submission allowance without treating any individual self-hash as external
authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import PurePosixPath
from typing import Mapping

from mlx_vq.quality.glm52_gpu_spend_snapshot import (
    QUALIFICATION_MAX_SECONDS,
    validate_gpu_spend_snapshot,
)
from mlx_vq.quality.glm52_qualification_cache_seed import (
    accepted_s3_key,
    validate_qualification_cache_seed_accepted,
)
from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_ACCOUNT_ID,
    APPROVED_REGION,
    validate_sky_campaign_descriptor,
)

SCHEMA_VERSION = 1
RECORD_TYPE = "glm52_qualification_submission_ready_v1"
MANAGED_MODE = "qualification"
DIGEST_FIELD = "readiness_body_sha256"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_READY_WINDOW = timedelta(hours=12)
_MAX_FRESHNESS = timedelta(minutes=5)
_STAGED_FILENAME = "STAGED_CONTROL_PLANE_READY.json"
_SNAPSHOT_FILENAME = "GPU_SPEND_SNAPSHOT.json"
_REHEARSAL_FILENAME = "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
_READY_FILENAME = "QUALIFICATION_SUBMISSION_READY.json"

_STAGED_FIELDS = frozenset(
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
_STAGED_OBJECT_VERSION_ROLES = frozenset(
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
_REHEARSAL_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "status",
        "run_id",
        "campaign_identity_sha256",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "repo_tar_sha256",
        "bootstrap_receipt_file_sha256",
        "staged_readiness_key",
        "staged_readiness_file_sha256",
        "staged_readiness_body_sha256",
        "artifact_inventory_key",
        "artifact_inventory_file_sha256",
        "artifact_inventory_body_sha256",
        "artifact_audit_key",
        "artifact_audit_file_sha256",
        "extracted_repo_path",
        "production_repo_path",
        "production_resume_root",
        "skypilot_task_path",
        "skypilot_task_file_sha256",
        "skypilot_config_file_sha256",
        "skypilot_validation_file_sha256",
        "skypilot_version",
        "skypilot_task_name",
        "completed_at",
        "rehearsal_body_sha256",
    }
)
_READY_FIELDS = frozenset(
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
        DIGEST_FIELD,
    }
)
_STAGED_DIGEST_FIELDS = frozenset(
    {
        "descriptor_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "bundle_manifest_file_sha256",
        "bundle_manifest_body_sha256",
        "artifact_audit_sha256",
        "ready_body_sha256",
    }
)
_REHEARSAL_DIGEST_FIELDS = frozenset(
    {
        "campaign_identity_sha256",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "repo_tar_sha256",
        "bootstrap_receipt_file_sha256",
        "staged_readiness_file_sha256",
        "staged_readiness_body_sha256",
        "artifact_inventory_file_sha256",
        "artifact_inventory_body_sha256",
        "artifact_audit_file_sha256",
        "skypilot_task_file_sha256",
        "skypilot_config_file_sha256",
        "skypilot_validation_file_sha256",
        "rehearsal_body_sha256",
    }
)
_SCIENCE_ARTIFACT_FIELDS = frozenset(
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
    }
)
_ALLOWED_ARTIFACT_CHANGES = frozenset(
    {
        "artifact_inventory_key",
        "artifact_inventory_sha256",
        "qualification_cache_prefix",
        "qualification_cache_manifest_sha256",
    }
)
_ALLOWED_DESCRIPTOR_CHANGES = frozenset(
    {
        "campaign_identity_sha256",
        "descriptor_body_sha256",
        "must_start_by",
        "campaign_descriptor_key",
        "artifacts",
    }
)


class QualificationSubmissionReadyError(ValueError):
    """A qualification submission is not closed over exact authority."""


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
        raise QualificationSubmissionReadyError(
            "value is not canonical finite JSON"
        ) from error


def _sha_raw(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_object(value: object) -> str:
    return _sha_raw(_canonical_bytes(value))


def _canonical_file_sha256(value: object) -> str:
    return _sha_raw(_canonical_bytes(value) + b"\n")


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise QualificationSubmissionReadyError(f"{label} must be an object")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise QualificationSubmissionReadyError(
            f"{label} schema mismatch: missing={missing}, unknown={unknown}"
        )


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise QualificationSubmissionReadyError(f"{field} must be a lowercase SHA-256")
    return value


def _require_version_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value in {"", "null"}:
        raise QualificationSubmissionReadyError(
            f"{field} must be a non-null S3 VersionId"
        )
    return value


def _require_run_id(value: object) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise QualificationSubmissionReadyError("run_id is invalid")
    return value


def _safe_key(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or value.endswith("/")
        or any(not 0x21 <= ord(character) <= 0x7E for character in value)
        or "\\" in value
        or any(character in value for character in "*?[]")
        or "//" in value
    ):
        raise QualificationSubmissionReadyError(f"{field} is not a safe exact key")
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise QualificationSubmissionReadyError(f"{field} is not a safe exact key")
    return value


def _canonical_time(
    value: object,
    *,
    field: str,
    permit_datetime: bool = False,
) -> tuple[str, datetime]:
    if isinstance(value, datetime) and permit_datetime:
        parsed = value
        if parsed.tzinfo is None:
            raise QualificationSubmissionReadyError(
                f"{field} timestamp must be timezone-aware"
            )
        canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return canonical, parsed.astimezone(timezone.utc)
    if not isinstance(value, str) or not value.endswith("Z"):
        raise QualificationSubmissionReadyError(
            f"{field} timestamp must use canonical UTC Z form"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise QualificationSubmissionReadyError(
            f"{field} timestamp is invalid"
        ) from error
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if parsed.utcoffset() != timedelta(0) or canonical != value:
        raise QualificationSubmissionReadyError(
            f"{field} timestamp is not canonical UTC"
        )
    return canonical, parsed


def _require_integer(
    value: object,
    *,
    field: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise QualificationSubmissionReadyError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _money(value: object, *, field: str) -> Decimal:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise QualificationSubmissionReadyError(f"{field} must be finite numeric money")
    result = Decimal(str(value))
    if result < 0 or result != result.quantize(Decimal("0.01")):
        raise QualificationSubmissionReadyError(
            f"{field} must be non-negative two-decimal money"
        )
    return result


def _cost(seconds: int, hourly: object) -> Decimal:
    return (Decimal(seconds) * Decimal(str(hourly)) / Decimal(3600)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def _validate_file_pin(
    value: object,
    expected_sha256: object,
    *,
    label: str,
) -> str:
    expected = _require_sha(expected_sha256, field=f"{label} file SHA-256")
    actual = _canonical_file_sha256(value)
    if actual != expected:
        raise QualificationSubmissionReadyError(
            f"{label} file SHA-256 does not match canonical object bytes"
        )
    return actual


def gpu_spend_snapshot_s3_key(
    *,
    run_id: str,
    snapshot_body_sha256: str,
) -> str:
    """Return the only accepted content-addressed spend-snapshot key."""

    run = _require_run_id(run_id)
    digest = _require_sha(
        snapshot_body_sha256,
        field="snapshot_body_sha256",
    )
    return f"campaigns/{run}/spend-snapshots/{digest}/{_SNAPSHOT_FILENAME}"


def rehearsal_evidence_s3_key(
    *,
    run_id: str,
    rehearsal_body_sha256: str,
) -> str:
    """Return the only accepted content-addressed rehearsal-evidence key."""

    run = _require_run_id(run_id)
    digest = _require_sha(
        rehearsal_body_sha256,
        field="rehearsal_body_sha256",
    )
    return f"campaigns/{run}/qualification/rehearsals/{digest}/{_REHEARSAL_FILENAME}"


def qualification_submission_ready_s3_key(
    value: Mapping[str, object],
) -> str:
    """Return the content-addressed key for an exact ready-record identity."""

    ready = _require_mapping(value, label="qualification submission readiness")
    run = _require_run_id(ready.get("run_id"))
    digest = _require_sha(
        ready.get(DIGEST_FIELD),
        field=DIGEST_FIELD,
    )
    return f"campaigns/{run}/qualification/submission-ready/{digest}/{_READY_FILENAME}"


def _validate_staged_readiness(
    value: Mapping[str, object],
) -> tuple[dict[str, object], datetime]:
    staged = _require_mapping(value, label="staged readiness")
    _require_exact_fields(staged, _STAGED_FIELDS, label="staged readiness")
    if (
        type(staged.get("schema_version")) is not int
        or staged.get("schema_version") != 2
        or staged.get("record_type") != "glm52_staged_control_plane_ready_v2"
    ):
        raise QualificationSubmissionReadyError("staged readiness schema mismatch")
    run_id = _require_run_id(staged.get("run_id"))
    for field in _STAGED_DIGEST_FIELDS:
        _require_sha(staged.get(field), field=f"staged readiness {field}")
    for field in (
        "descriptor_key",
        "bundle_manifest_key",
        "artifact_audit_key",
    ):
        _safe_key(
            staged.get(field),
            field=f"staged readiness {field.replace('_', ' ')}",
        )
    _require_version_id(
        staged.get("bundle_manifest_version_id"),
        field="staged readiness bundle manifest VersionId",
    )
    versions = _require_mapping(
        staged.get("staged_object_version_ids"),
        label="staged readiness staged_object_version_ids",
    )
    _require_exact_fields(
        versions,
        _STAGED_OBJECT_VERSION_ROLES,
        label="staged readiness VersionId map",
    )
    for role in _STAGED_OBJECT_VERSION_ROLES:
        _require_version_id(
            versions.get(role),
            field=f"staged readiness {role} VersionId",
        )
    descriptor_key = str(staged["descriptor_key"])
    descriptor_suffix = "/campaign-descriptor-v2.json"
    submission_prefix = f"campaigns/{run_id}/submissions/"
    if (
        not descriptor_key.startswith(submission_prefix)
        or not descriptor_key.endswith(descriptor_suffix)
    ):
        raise QualificationSubmissionReadyError(
            "staged readiness descriptor key does not match the run submission prefix"
        )
    submission_id = descriptor_key[
        len(submission_prefix) : -len(descriptor_suffix)
    ]
    if not submission_id or "/" in submission_id:
        raise QualificationSubmissionReadyError(
            "staged readiness descriptor submission ID is unsafe"
        )
    expected_manifest_key = (
        f"{submission_prefix}{submission_id}/bundle-manifests/"
        f"{staged['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
    )
    if staged["bundle_manifest_key"] != expected_manifest_key:
        raise QualificationSubmissionReadyError(
            "staged readiness bundle manifest key binding mismatch"
        )
    body = dict(staged)
    digest = body.pop("ready_body_sha256")
    if digest != _sha_object(body):
        raise QualificationSubmissionReadyError(
            "staged readiness body SHA-256 mismatch"
        )
    _, staged_at = _canonical_time(staged.get("staged_at"), field="staged_at")
    if staged_at.microsecond != 0:
        raise QualificationSubmissionReadyError(
            "staged readiness staged_at timestamp must use whole seconds"
        )
    return dict(staged), staged_at


def _validate_rehearsal_evidence(
    value: Mapping[str, object],
) -> tuple[dict[str, object], datetime]:
    rehearsal = _require_mapping(value, label="rehearsal evidence")
    _require_exact_fields(
        rehearsal,
        _REHEARSAL_FIELDS,
        label="rehearsal evidence",
    )
    if (
        type(rehearsal.get("schema_version")) is not int
        or rehearsal.get("schema_version") != 2
        or rehearsal.get("record_type") != "glm52_staged_control_plane_rehearsal_v2"
        or rehearsal.get("status") != "passed_before_cuda_h100_boundary"
    ):
        raise QualificationSubmissionReadyError(
            "rehearsal evidence schema or status mismatch"
        )
    _require_run_id(rehearsal.get("run_id"))
    for field in _REHEARSAL_DIGEST_FIELDS:
        _require_sha(rehearsal.get(field), field=f"rehearsal {field}")
    for field in (
        "descriptor_key",
        "staged_readiness_key",
        "artifact_inventory_key",
        "artifact_audit_key",
    ):
        _safe_key(rehearsal.get(field), field=f"rehearsal {field}")
    extracted = rehearsal.get("extracted_repo_path")
    if (
        not isinstance(extracted, str)
        or not extracted
        or not PurePosixPath(extracted).is_absolute()
        or ".." in PurePosixPath(extracted).parts
    ):
        raise QualificationSubmissionReadyError(
            "rehearsal extracted_repo_path must be an absolute safe path"
        )
    if (
        rehearsal.get("production_repo_path") != "/opt/keep-campaign/repo"
        or rehearsal.get("production_resume_root") != "/mnt/nvme/glm52-campaign"
        or rehearsal.get("skypilot_task_path")
        != ("/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/glm52-campaign.yaml")
        or rehearsal.get("skypilot_version") != "0.13.0"
        or not isinstance(rehearsal.get("skypilot_task_name"), str)
        or not rehearsal["skypilot_task_name"]
    ):
        raise QualificationSubmissionReadyError(
            "rehearsal production path or SkyPilot identity mismatch"
        )
    body = dict(rehearsal)
    digest = body.pop("rehearsal_body_sha256")
    if digest != _sha_object(body):
        raise QualificationSubmissionReadyError(
            "rehearsal evidence body SHA-256 mismatch"
        )
    _, completed_at = _canonical_time(
        rehearsal.get("completed_at"),
        field="rehearsal completed_at",
    )
    return dict(rehearsal), completed_at


def _validate_seed_post_bridge(
    *,
    seed_descriptor: Mapping[str, object],
    seed_descriptor_file_sha256: str,
    descriptor: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> None:
    authority = _require_mapping(
        acceptance.get("seed_authority"),
        label="cache-seed acceptance seed authority",
    )
    seed_bindings = {
        "account_id": seed_descriptor["account_id"],
        "region": seed_descriptor["region"],
        "bucket": seed_descriptor["bucket"],
        "run_id": seed_descriptor["run_id"],
        "descriptor_key": seed_descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": seed_descriptor_file_sha256,
        "descriptor_body_sha256": seed_descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": seed_descriptor["campaign_identity_sha256"],
        "repo_tar_key": seed_descriptor["repo_tar_key"],
        "repo_tar_sha256": seed_descriptor["repo_tar_sha256"],
        "approval_key": seed_descriptor["approval_key"],
        "approval_sha256": seed_descriptor["approval_sha256"],
    }
    for field, expected in seed_bindings.items():
        if authority.get(field) != expected:
            raise QualificationSubmissionReadyError(
                f"seed descriptor {field} does not match cache-seed acceptance"
            )
    if authority.get("managed_mode") != "cache-seed":
        raise QualificationSubmissionReadyError(
            "cache-seed acceptance managed mode mismatch"
        )

    seed_artifacts = _require_mapping(
        seed_descriptor.get("artifacts"),
        label="seed descriptor artifacts",
    )
    post_artifacts = _require_mapping(
        descriptor.get("artifacts"),
        label="post-seed descriptor artifacts",
    )
    for field in _SCIENCE_ARTIFACT_FIELDS:
        if seed_artifacts.get(field) != post_artifacts.get(field):
            raise QualificationSubmissionReadyError(
                f"seed-to-post science artifact drift: {field}"
            )
    for field in set(seed_artifacts) - _ALLOWED_ARTIFACT_CHANGES:
        if seed_artifacts.get(field) != post_artifacts.get(field):
            raise QualificationSubmissionReadyError(
                f"seed-to-post descriptor artifact drift: {field}"
            )
    for field in set(seed_descriptor) - _ALLOWED_DESCRIPTOR_CHANGES:
        if seed_descriptor.get(field) != descriptor.get(field):
            raise QualificationSubmissionReadyError(
                f"seed-to-post descriptor authority drift: {field}"
            )
    if (
        seed_artifacts.get("qualification_cache_prefix") != "qualification-cache/"
        or seed_artifacts.get("qualification_cache_manifest_sha256") != "0" * 64
    ):
        raise QualificationSubmissionReadyError(
            "seed descriptor cache authority must be the exact placeholder"
        )

    cache_audit = _require_mapping(
        acceptance.get("cache_audit"),
        label="cache-seed acceptance cache audit",
    )
    post_cache = {
        "qualification_cache_prefix": cache_audit.get("cache_prefix"),
        "qualification_cache_manifest_sha256": cache_audit.get("manifest_file_sha256"),
        "teich_pack_key": cache_audit.get("teich_pack_key"),
        "teich_pack_sha256": cache_audit.get("teich_pack_file_sha256"),
        "frozen_prompt_pack_key": cache_audit.get("frozen_prompt_pack_key"),
        "frozen_prompt_pack_sha256": cache_audit.get("frozen_prompt_pack_file_sha256"),
    }
    for field, expected in post_cache.items():
        if post_artifacts.get(field) != expected:
            raise QualificationSubmissionReadyError(
                f"post-seed descriptor cache authority mismatch: {field}"
            )
    if descriptor.get("campaign_descriptor_key") == seed_descriptor.get(
        "campaign_descriptor_key"
    ):
        raise QualificationSubmissionReadyError(
            "post-seed descriptor must be a distinct immutable submission"
        )


def _validate_authorities(
    *,
    descriptor: Mapping[str, object],
    seed_descriptor: Mapping[str, object],
    seed_descriptor_file_sha256: str,
    staged_readiness: Mapping[str, object],
    cache_seed_acceptance: Mapping[str, object],
    gpu_spend_snapshot: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    descriptor_file_sha256: str,
    staged_readiness_key: str,
    staged_readiness_file_sha256: str,
    cache_seed_acceptance_key: str,
    cache_seed_acceptance_file_sha256: str,
    gpu_spend_snapshot_key: str,
    gpu_spend_snapshot_sha256: str,
    rehearsal_evidence_key: str,
    rehearsal_evidence_sha256: str,
) -> dict[str, object]:
    try:
        post = validate_sky_campaign_descriptor(descriptor)
    except (TypeError, ValueError) as error:
        raise QualificationSubmissionReadyError(
            f"post-seed descriptor is invalid: {error}"
        ) from error
    try:
        seed = validate_sky_campaign_descriptor(seed_descriptor)
    except (TypeError, ValueError) as error:
        raise QualificationSubmissionReadyError(
            f"seed descriptor is invalid: {error}"
        ) from error
    descriptor_sha = _validate_file_pin(
        post,
        descriptor_file_sha256,
        label="descriptor",
    )
    seed_sha = _validate_file_pin(
        seed,
        seed_descriptor_file_sha256,
        label="seed descriptor",
    )
    try:
        acceptance = validate_qualification_cache_seed_accepted(cache_seed_acceptance)
    except (TypeError, ValueError) as error:
        raise QualificationSubmissionReadyError(
            f"cache-seed acceptance is invalid: {error}"
        ) from error
    try:
        snapshot = validate_gpu_spend_snapshot(gpu_spend_snapshot)
    except (TypeError, ValueError) as error:
        raise QualificationSubmissionReadyError(
            f"GPU spend snapshot is invalid: {error}"
        ) from error
    staged, staged_at = _validate_staged_readiness(staged_readiness)
    rehearsal, rehearsal_at = _validate_rehearsal_evidence(rehearsal_evidence)

    staged_file_sha = _validate_file_pin(
        staged,
        staged_readiness_file_sha256,
        label="staged readiness",
    )
    acceptance_file_sha = _validate_file_pin(
        acceptance,
        cache_seed_acceptance_file_sha256,
        label="cache-seed acceptance",
    )
    snapshot_file_sha = _validate_file_pin(
        snapshot,
        gpu_spend_snapshot_sha256,
        label="GPU spend snapshot",
    )
    rehearsal_file_sha = _validate_file_pin(
        rehearsal,
        rehearsal_evidence_sha256,
        label="rehearsal evidence",
    )

    run_id = _require_run_id(post.get("run_id"))
    descriptor_key = _safe_key(
        post.get("campaign_descriptor_key"),
        field="descriptor key",
    )
    expected_descriptor_prefix = f"campaigns/{run_id}/submissions/"
    if not descriptor_key.startswith(
        expected_descriptor_prefix
    ) or not descriptor_key.endswith("/campaign-descriptor-v2.json"):
        raise QualificationSubmissionReadyError(
            "descriptor key is outside the exact run submission prefix"
        )
    if post.get("account_id") != APPROVED_ACCOUNT_ID:
        raise QualificationSubmissionReadyError(
            "descriptor account does not match approved account"
        )
    if post.get("region") != APPROVED_REGION:
        raise QualificationSubmissionReadyError(
            "descriptor region does not match approved region"
        )

    expected_staged_key = (
        descriptor_key.removesuffix("campaign-descriptor-v2.json") + _STAGED_FILENAME
    )
    staged_key = _safe_key(
        staged_readiness_key,
        field="staged readiness key",
    )
    if staged_key != expected_staged_key:
        raise QualificationSubmissionReadyError(
            "staged readiness key does not match the descriptor submission"
        )
    staged_bindings = {
        "run_id": run_id,
        "descriptor_key": descriptor_key,
        "descriptor_sha256": descriptor_sha,
        "descriptor_body_sha256": post["descriptor_body_sha256"],
        "campaign_identity_sha256": post["campaign_identity_sha256"],
    }
    for field, expected in staged_bindings.items():
        if staged.get(field) != expected:
            raise QualificationSubmissionReadyError(
                f"staged readiness {field.replace('_', ' ')} mismatch"
            )
    expected_audit_key = (
        f"campaigns/{run_id}/audits/"
        f"artifact-audit-{staged['artifact_audit_sha256']}.json"
    )
    if staged.get("artifact_audit_key") != expected_audit_key:
        raise QualificationSubmissionReadyError(
            "staged readiness artifact audit key mismatch"
        )

    acceptance_digest = _require_sha(
        acceptance.get("acceptance_body_sha256"),
        field="cache-seed acceptance body SHA-256",
    )
    expected_acceptance_key = accepted_s3_key(
        run_id=run_id,
        acceptance_body_sha256=acceptance_digest,
    )
    acceptance_key = _safe_key(
        cache_seed_acceptance_key,
        field="cache-seed acceptance key",
    )
    if acceptance_key != expected_acceptance_key:
        raise QualificationSubmissionReadyError(
            "cache-seed acceptance key does not match its exact body"
        )

    snapshot_digest = _require_sha(
        snapshot.get("snapshot_body_sha256"),
        field="GPU spend snapshot body SHA-256",
    )
    expected_snapshot_key = gpu_spend_snapshot_s3_key(
        run_id=run_id,
        snapshot_body_sha256=snapshot_digest,
    )
    snapshot_key = _safe_key(
        gpu_spend_snapshot_key,
        field="GPU spend snapshot key",
    )
    if snapshot_key != expected_snapshot_key:
        raise QualificationSubmissionReadyError(
            "GPU spend snapshot key does not match its exact body"
        )
    snapshot_bindings = {
        "run_id": run_id,
        "campaign_identity_sha256": post["campaign_identity_sha256"],
        "descriptor_sha256": descriptor_sha,
        "descriptor_body_sha256": post["descriptor_body_sha256"],
        "approval_sha256": post["approval_sha256"],
    }
    for field, expected in snapshot_bindings.items():
        if snapshot.get(field) != expected:
            raise QualificationSubmissionReadyError(
                f"GPU spend snapshot {field} mismatch"
            )

    rehearsal_digest = _require_sha(
        rehearsal.get("rehearsal_body_sha256"),
        field="rehearsal evidence body SHA-256",
    )
    expected_rehearsal_key = rehearsal_evidence_s3_key(
        run_id=run_id,
        rehearsal_body_sha256=rehearsal_digest,
    )
    rehearsal_key = _safe_key(
        rehearsal_evidence_key,
        field="rehearsal evidence key",
    )
    if rehearsal_key != expected_rehearsal_key:
        raise QualificationSubmissionReadyError(
            "rehearsal evidence key does not match its exact body"
        )
    rehearsal_bindings = {
        "run_id": run_id,
        "campaign_identity_sha256": post["campaign_identity_sha256"],
        "descriptor_key": descriptor_key,
        "descriptor_file_sha256": descriptor_sha,
        "descriptor_body_sha256": post["descriptor_body_sha256"],
        "repo_tar_sha256": post["repo_tar_sha256"],
        "staged_readiness_key": staged_key,
        "staged_readiness_file_sha256": staged_file_sha,
        "staged_readiness_body_sha256": staged["ready_body_sha256"],
        "artifact_inventory_key": post["artifacts"]["artifact_inventory_key"],
        "artifact_inventory_file_sha256": post["artifacts"][
            "artifact_inventory_sha256"
        ],
        "artifact_audit_key": staged["artifact_audit_key"],
        "artifact_audit_file_sha256": staged["artifact_audit_sha256"],
        "skypilot_task_name": post["task_name"],
    }
    for field, expected in rehearsal_bindings.items():
        if rehearsal.get(field) != expected:
            raise QualificationSubmissionReadyError(
                f"rehearsal {field.replace('_', ' ')} mismatch"
            )

    _validate_seed_post_bridge(
        seed_descriptor=seed,
        seed_descriptor_file_sha256=seed_sha,
        descriptor=post,
        acceptance=acceptance,
    )
    spend_closure = _require_mapping(
        acceptance.get("spend_closure"),
        label="cache-seed spend closure",
    )
    if spend_closure.get("ledger_tip_record_sha256") != snapshot.get(
        "gpu_spend_ledger_tip_record_sha256"
    ):
        raise QualificationSubmissionReadyError(
            "cache-seed and spend snapshot ledger tip mismatch"
        )

    if staged_at > rehearsal_at:
        raise QualificationSubmissionReadyError(
            "staged readiness timestamp follows rehearsal timestamp"
        )
    _, acceptance_at = _canonical_time(
        acceptance.get("accepted_at"),
        field="cache-seed accepted_at",
    )
    _, snapshot_at = _canonical_time(
        snapshot.get("observed_at"),
        field="GPU spend snapshot observed_at",
    )
    return {
        "descriptor": post,
        "descriptor_file_sha256": descriptor_sha,
        "seed_descriptor": seed,
        "seed_descriptor_file_sha256": seed_sha,
        "staged_readiness": staged,
        "staged_readiness_key": staged_key,
        "staged_readiness_file_sha256": staged_file_sha,
        "staged_at": staged_at,
        "cache_seed_acceptance": acceptance,
        "cache_seed_acceptance_key": acceptance_key,
        "cache_seed_acceptance_file_sha256": acceptance_file_sha,
        "cache_seed_accepted_at": acceptance_at,
        "gpu_spend_snapshot": snapshot,
        "gpu_spend_snapshot_key": snapshot_key,
        "gpu_spend_snapshot_sha256": snapshot_file_sha,
        "gpu_spend_snapshot_at": snapshot_at,
        "rehearsal_evidence": rehearsal,
        "rehearsal_evidence_key": rehearsal_key,
        "rehearsal_evidence_sha256": rehearsal_file_sha,
        "rehearsal_at": rehearsal_at,
    }


def _expected_ready_body(
    authorities: Mapping[str, object],
    *,
    built_at: str,
) -> dict[str, object]:
    descriptor = authorities["descriptor"]
    seed_descriptor = authorities["seed_descriptor"]
    staged = authorities["staged_readiness"]
    acceptance = authorities["cache_seed_acceptance"]
    snapshot = authorities["gpu_spend_snapshot"]
    rehearsal = authorities["rehearsal_evidence"]
    assert isinstance(descriptor, Mapping)
    assert isinstance(seed_descriptor, Mapping)
    assert isinstance(staged, Mapping)
    assert isinstance(acceptance, Mapping)
    assert isinstance(snapshot, Mapping)
    assert isinstance(rehearsal, Mapping)
    run_id = str(descriptor["run_id"])
    allowed_seconds = snapshot["qualification_allowance_seconds"]
    allowed_cost = snapshot["qualification_allowance_cost_usd"]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "run_id": run_id,
        "managed_mode": MANAGED_MODE,
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": authorities["descriptor_file_sha256"],
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
        "approval_body_sha256": snapshot["approval_body_sha256"],
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "seed_descriptor_key": seed_descriptor["campaign_descriptor_key"],
        "seed_descriptor_file_sha256": authorities["seed_descriptor_file_sha256"],
        "seed_descriptor_body_sha256": seed_descriptor["descriptor_body_sha256"],
        "seed_campaign_identity_sha256": seed_descriptor["campaign_identity_sha256"],
        "seed_repo_tar_sha256": seed_descriptor["repo_tar_sha256"],
        "seed_descriptor": dict(seed_descriptor),
        "staged_readiness_key": authorities["staged_readiness_key"],
        "staged_readiness_file_sha256": authorities["staged_readiness_file_sha256"],
        "staged_readiness_body_sha256": staged["ready_body_sha256"],
        "cache_seed_acceptance_key": authorities["cache_seed_acceptance_key"],
        "cache_seed_acceptance_file_sha256": authorities[
            "cache_seed_acceptance_file_sha256"
        ],
        "cache_seed_acceptance_body_sha256": acceptance["acceptance_body_sha256"],
        "gpu_spend_snapshot_key": authorities["gpu_spend_snapshot_key"],
        "gpu_spend_snapshot_sha256": authorities["gpu_spend_snapshot_sha256"],
        "gpu_spend_snapshot_body_sha256": snapshot["snapshot_body_sha256"],
        "gpu_spend_ledger_tip_record_sha256": snapshot[
            "gpu_spend_ledger_tip_record_sha256"
        ],
        "remaining_gpu_seconds": snapshot["remaining_gpu_seconds"],
        "remaining_gpu_cost_usd": snapshot["remaining_gpu_cost_usd"],
        "qualification_allowance_seconds": allowed_seconds,
        "qualification_allowance_cost_usd": allowed_cost,
        "rehearsal_evidence_key": authorities["rehearsal_evidence_key"],
        "rehearsal_evidence_sha256": authorities["rehearsal_evidence_sha256"],
        "rehearsal_evidence_body_sha256": rehearsal["rehearsal_body_sha256"],
        "sky_task_name": descriptor["task_name"],
        "sky_job_name": f"{run_id}-qualification",
        "must_start_by": descriptor["must_start_by"],
        "allowed_gpu_seconds": allowed_seconds,
        "allowed_gpu_cost_usd": allowed_cost,
        "built_at": built_at,
    }


def _validate_allowance_and_window(
    *,
    body: Mapping[str, object],
    authorities: Mapping[str, object],
    now: datetime | str | None,
) -> None:
    descriptor = authorities["descriptor"]
    snapshot = authorities["gpu_spend_snapshot"]
    assert isinstance(descriptor, Mapping)
    assert isinstance(snapshot, Mapping)
    allowed_seconds = _require_integer(
        body.get("allowed_gpu_seconds"),
        field="allowed_gpu_seconds",
        minimum=1,
    )
    qualification_seconds = _require_integer(
        body.get("qualification_allowance_seconds"),
        field="qualification_allowance_seconds",
        minimum=1,
    )
    remaining_seconds = _require_integer(
        body.get("remaining_gpu_seconds"),
        field="remaining_gpu_seconds",
        minimum=1,
    )
    if (
        allowed_seconds != qualification_seconds
        or allowed_seconds != snapshot.get("qualification_allowance_seconds")
        or allowed_seconds > remaining_seconds
        or allowed_seconds > QUALIFICATION_MAX_SECONDS
    ):
        raise QualificationSubmissionReadyError(
            "allowed_gpu_seconds is outside the exact spend allowance"
        )
    allowed_cost = _money(
        body.get("allowed_gpu_cost_usd"),
        field="allowed_gpu_cost_usd",
    )
    qualification_cost = _money(
        body.get("qualification_allowance_cost_usd"),
        field="qualification_allowance_cost_usd",
    )
    remaining_cost = _money(
        body.get("remaining_gpu_cost_usd"),
        field="remaining_gpu_cost_usd",
    )
    expected_cost = _cost(
        allowed_seconds,
        descriptor["max_hourly_cost_usd"],
    )
    if (
        allowed_cost != qualification_cost
        or allowed_cost
        != Decimal(str(snapshot.get("qualification_allowance_cost_usd")))
        or allowed_cost != expected_cost
        or allowed_cost > remaining_cost
    ):
        raise QualificationSubmissionReadyError(
            "allowed_gpu_cost_usd is outside the exact spend allowance"
        )

    _, built = _canonical_time(body.get("built_at"), field="built_at")
    _, deadline = _canonical_time(
        body.get("must_start_by"),
        field="must_start_by",
    )
    authority_times = (
        authorities["cache_seed_accepted_at"],
        authorities["gpu_spend_snapshot_at"],
        authorities["staged_at"],
        authorities["rehearsal_at"],
    )
    if any(
        not isinstance(value, datetime) or built < value for value in authority_times
    ):
        raise QualificationSubmissionReadyError(
            "built_at timestamp precedes upstream authority timestamp"
        )
    snapshot_at = authorities["gpu_spend_snapshot_at"]
    if not isinstance(snapshot_at, datetime) or built - snapshot_at > _MAX_FRESHNESS:
        raise QualificationSubmissionReadyError(
            "GPU spend snapshot is more than 5 minutes older than built_at"
        )
    if deadline <= built:
        raise QualificationSubmissionReadyError(
            "qualification submission authority is expired at built_at"
        )
    if deadline - built > _MAX_READY_WINDOW:
        raise QualificationSubmissionReadyError(
            "qualification submission window exceeds 12 hours"
        )
    if now is not None:
        _, observation = _canonical_time(
            now,
            field="now",
            permit_datetime=True,
        )
        if observation < built:
            raise QualificationSubmissionReadyError("now timestamp precedes built_at")
        if observation >= deadline:
            raise QualificationSubmissionReadyError(
                "qualification submission authority is expired"
            )
        if observation - built > _MAX_FRESHNESS:
            raise QualificationSubmissionReadyError(
                "qualification readiness is more than 5 minutes old"
            )


def build_qualification_submission_ready(
    *,
    descriptor: Mapping[str, object],
    seed_descriptor_raw: bytes,
    seed_descriptor_file_sha256: str,
    staged_readiness: Mapping[str, object],
    cache_seed_acceptance: Mapping[str, object],
    gpu_spend_snapshot: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    descriptor_file_sha256: str,
    staged_readiness_key: str,
    staged_readiness_file_sha256: str,
    cache_seed_acceptance_key: str,
    cache_seed_acceptance_file_sha256: str,
    gpu_spend_snapshot_key: str,
    gpu_spend_snapshot_sha256: str,
    rehearsal_evidence_key: str,
    rehearsal_evidence_sha256: str,
    built_at: datetime | str,
) -> dict[str, object]:
    """Build the exact short-lived authority accepted by qualification submit."""

    if not isinstance(seed_descriptor_raw, bytes):
        raise QualificationSubmissionReadyError(
            "seed descriptor raw value must be bytes"
        )
    seed_sha = _require_sha(
        seed_descriptor_file_sha256,
        field="seed descriptor file SHA-256",
    )
    if _sha_raw(seed_descriptor_raw) != seed_sha:
        raise QualificationSubmissionReadyError("seed descriptor raw SHA-256 mismatch")
    try:
        seed_descriptor = json.loads(
            seed_descriptor_raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise QualificationSubmissionReadyError(
            f"seed descriptor is not valid finite JSON: {error}"
        ) from error
    seed_mapping = _require_mapping(
        seed_descriptor,
        label="seed descriptor",
    )
    if seed_descriptor_raw != _canonical_bytes(seed_mapping) + b"\n":
        raise QualificationSubmissionReadyError(
            "seed descriptor bytes are not exact canonical JSON plus newline"
        )
    built_iso, _ = _canonical_time(
        built_at,
        field="built_at",
        permit_datetime=True,
    )
    authorities = _validate_authorities(
        descriptor=descriptor,
        seed_descriptor=seed_mapping,
        seed_descriptor_file_sha256=seed_sha,
        staged_readiness=staged_readiness,
        cache_seed_acceptance=cache_seed_acceptance,
        gpu_spend_snapshot=gpu_spend_snapshot,
        rehearsal_evidence=rehearsal_evidence,
        descriptor_file_sha256=descriptor_file_sha256,
        staged_readiness_key=staged_readiness_key,
        staged_readiness_file_sha256=staged_readiness_file_sha256,
        cache_seed_acceptance_key=cache_seed_acceptance_key,
        cache_seed_acceptance_file_sha256=cache_seed_acceptance_file_sha256,
        gpu_spend_snapshot_key=gpu_spend_snapshot_key,
        gpu_spend_snapshot_sha256=gpu_spend_snapshot_sha256,
        rehearsal_evidence_key=rehearsal_evidence_key,
        rehearsal_evidence_sha256=rehearsal_evidence_sha256,
    )
    body = _expected_ready_body(authorities, built_at=built_iso)
    _validate_allowance_and_window(
        body=body,
        authorities=authorities,
        now=built_iso,
    )
    result = {**body, DIGEST_FIELD: _sha_object(body)}
    return validate_qualification_submission_ready(
        result,
        descriptor=descriptor,
        staged_readiness=staged_readiness,
        cache_seed_acceptance=cache_seed_acceptance,
        gpu_spend_snapshot=gpu_spend_snapshot,
        rehearsal_evidence=rehearsal_evidence,
        descriptor_file_sha256=descriptor_file_sha256,
        staged_readiness_key=staged_readiness_key,
        staged_readiness_file_sha256=staged_readiness_file_sha256,
        cache_seed_acceptance_key=cache_seed_acceptance_key,
        cache_seed_acceptance_file_sha256=cache_seed_acceptance_file_sha256,
        gpu_spend_snapshot_key=gpu_spend_snapshot_key,
        gpu_spend_snapshot_sha256=gpu_spend_snapshot_sha256,
        rehearsal_evidence_key=rehearsal_evidence_key,
        rehearsal_evidence_sha256=rehearsal_evidence_sha256,
        now=built_iso,
    )


def validate_qualification_submission_ready(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
    staged_readiness: Mapping[str, object],
    cache_seed_acceptance: Mapping[str, object],
    gpu_spend_snapshot: Mapping[str, object],
    rehearsal_evidence: Mapping[str, object],
    descriptor_file_sha256: str,
    staged_readiness_key: str,
    staged_readiness_file_sha256: str,
    cache_seed_acceptance_key: str,
    cache_seed_acceptance_file_sha256: str,
    gpu_spend_snapshot_key: str,
    gpu_spend_snapshot_sha256: str,
    rehearsal_evidence_key: str,
    rehearsal_evidence_sha256: str,
    now: datetime | str | None = None,
) -> dict[str, object]:
    """Validate an exact ready record against every external authority input."""

    ready = _require_mapping(value, label="qualification submission readiness")
    _require_exact_fields(
        ready,
        _READY_FIELDS,
        label="qualification submission readiness",
    )
    _canonical_bytes(ready)
    if (
        type(ready.get("schema_version")) is not int
        or ready.get("schema_version") != SCHEMA_VERSION
        or ready.get("record_type") != RECORD_TYPE
    ):
        raise QualificationSubmissionReadyError(
            "qualification submission readiness schema mismatch"
        )
    body = dict(ready)
    digest = _require_sha(
        body.pop(DIGEST_FIELD),
        field=DIGEST_FIELD,
    )
    if digest != _sha_object(body):
        raise QualificationSubmissionReadyError(
            "qualification submission readiness body SHA-256 mismatch"
        )
    embedded_seed = _require_mapping(
        ready.get("seed_descriptor"),
        label="embedded seed descriptor",
    )
    seed_file_sha = _require_sha(
        ready.get("seed_descriptor_file_sha256"),
        field="seed descriptor file SHA-256",
    )
    authorities = _validate_authorities(
        descriptor=descriptor,
        seed_descriptor=embedded_seed,
        seed_descriptor_file_sha256=seed_file_sha,
        staged_readiness=staged_readiness,
        cache_seed_acceptance=cache_seed_acceptance,
        gpu_spend_snapshot=gpu_spend_snapshot,
        rehearsal_evidence=rehearsal_evidence,
        descriptor_file_sha256=descriptor_file_sha256,
        staged_readiness_key=staged_readiness_key,
        staged_readiness_file_sha256=staged_readiness_file_sha256,
        cache_seed_acceptance_key=cache_seed_acceptance_key,
        cache_seed_acceptance_file_sha256=cache_seed_acceptance_file_sha256,
        gpu_spend_snapshot_key=gpu_spend_snapshot_key,
        gpu_spend_snapshot_sha256=gpu_spend_snapshot_sha256,
        rehearsal_evidence_key=rehearsal_evidence_key,
        rehearsal_evidence_sha256=rehearsal_evidence_sha256,
    )
    expected = _expected_ready_body(
        authorities,
        built_at=str(ready.get("built_at")),
    )
    for field, expected_value in expected.items():
        if ready.get(field) != expected_value:
            raise QualificationSubmissionReadyError(
                f"{field} does not match exact upstream authority"
            )
    _validate_allowance_and_window(
        body=body,
        authorities=authorities,
        now=now,
    )
    qualification_submission_ready_s3_key(ready)
    return dict(ready)


__all__ = [
    "DIGEST_FIELD",
    "MANAGED_MODE",
    "QualificationSubmissionReadyError",
    "RECORD_TYPE",
    "SCHEMA_VERSION",
    "build_qualification_submission_ready",
    "gpu_spend_snapshot_s3_key",
    "qualification_submission_ready_s3_key",
    "rehearsal_evidence_s3_key",
    "validate_qualification_submission_ready",
]
