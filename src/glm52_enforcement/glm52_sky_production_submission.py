"""Enforcement-native authority for a production Sky submission intent."""

# ruff: noqa: UP017

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import PurePosixPath
from typing import Any, Callable, Union


class _FlatModuleUnavailable(Exception):
    pass


def _flat_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        raise _FlatModuleUnavailable from error


try:
    _snapshot_module = _flat_module("glm52_gpu_spend_snapshot")
    _h100_module = _flat_module("glm52_h100_qualification")
    _cache_module = _flat_module("glm52_qualification_cache_seed")
    _campaign_module = _flat_module("glm52_sky_campaign")
    _modes_module = _flat_module("glm52_sky_submission_modes")
except _FlatModuleUnavailable:
    from .glm52_gpu_spend_snapshot import (
        validate_gpu_spend_snapshot,
    )
    from .glm52_h100_qualification import (
        validate_h100_resume_ready,
        validate_h100_runtime_allocation,
    )
    from .glm52_qualification_cache_seed import (
        validate_qualification_cache_seed_accepted,
    )
    from .glm52_sky_campaign import (
        validate_gpu_spend_approval,
        validate_sky_campaign_descriptor,
    )
    from .glm52_sky_submission_modes import (
        SubmissionModeContractError,
        expected_sky_job_name,
        record_contract,
        require_opaque_version_id,
    )
else:
    validate_gpu_spend_snapshot = _snapshot_module.validate_gpu_spend_snapshot
    validate_h100_resume_ready = _h100_module.validate_h100_resume_ready
    validate_h100_runtime_allocation = _h100_module.validate_h100_runtime_allocation
    validate_qualification_cache_seed_accepted = (
        _cache_module.validate_qualification_cache_seed_accepted
    )
    validate_gpu_spend_approval = _campaign_module.validate_gpu_spend_approval
    validate_sky_campaign_descriptor = _campaign_module.validate_sky_campaign_descriptor
    SubmissionModeContractError = _modes_module.SubmissionModeContractError
    expected_sky_job_name = _modes_module.expected_sky_job_name
    record_contract = _modes_module.record_contract
    require_opaque_version_id = _modes_module.require_opaque_version_id


class ProductionSubmissionError(ValueError):
    """Raised when a production intent or source chain fails closed."""


@dataclass(frozen=True)
class VersionedJsonArtifact:
    """Caller-supplied exact bytes and immutable transport identity."""

    key: str
    raw: bytes
    version_id: str


_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_INSTANCE_ID = re.compile(r"i-(?:[0-9a-f]{8}|[0-9a-f]{17})")
_MANIFEST_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "bucket",
    "files",
    "descriptor_body_sha256",
    "bundle_manifest_body_sha256",
}
_MANIFEST_ROW_FIELDS = {
    "local_name",
    "key",
    "role",
    "stage_order",
    "size",
    "sha256",
}
_MANIFEST_ROLE_ORDERS = (
    ("repository_tar", 10),
    ("approval", 20),
    ("training_config", 30),
    ("watchdog", 40),
    ("artifact_inventory", 50),
    ("descriptor", 100),
)
_STAGED_FIELDS = {
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
_STAGED_VERSION_FIELDS = {
    "repository_tar",
    "approval",
    "training_config",
    "watchdog",
    "artifact_inventory",
    "artifact_audit",
    "descriptor",
}
_REHEARSAL_FIELDS = {
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
_INTENT_SHA_FIELDS = {field for field in _INTENT_FIELDS if field.endswith("_sha256")}
_INTENT_VERSION_FIELDS = {
    field for field in _INTENT_FIELDS if field.endswith("_version_id")
}
_INTENT_KEY_FIELDS = {field for field in _INTENT_FIELDS if field.endswith("_key")}


def _translate(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except ProductionSubmissionError:
        raise
    except SubmissionModeContractError as error:
        raise ProductionSubmissionError(str(error)) from error
    except Exception as error:
        raise ProductionSubmissionError(str(error)) from error


def _require_exact_json(value: object, *, field: str) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ProductionSubmissionError(f"{field} must be finite")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _require_exact_json(item, field=f"{field}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ProductionSubmissionError(
                    f"{field} dictionary keys must be exact strings"
                )
            _require_exact_json(item, field=f"{field}.{key}")
        return
    raise ProductionSubmissionError(f"{field} is not an exact JSON value")


def _canonical(value: object) -> bytes:
    _require_exact_json(value, field="value")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha(value: Mapping[str, object], digest_field: str) -> str:
    body = dict(value)
    body.pop(digest_field)
    return _sha(_canonical(body))


def _duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProductionSubmissionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProductionSubmissionError(f"nonfinite JSON number is forbidden: {value}")


def _parse_raw(raw: bytes, *, label: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise ProductionSubmissionError(f"{label} raw must be exact bytes")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_duplicates,
            parse_constant=_reject_constant,
        )
    except ProductionSubmissionError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ProductionSubmissionError(f"{label} is not strict JSON") from error
    if type(value) is not dict:
        raise ProductionSubmissionError(f"{label} root must be an exact object")
    _require_exact_json(value, field=label)
    if raw != _canonical(value) + b"\n":
        raise ProductionSubmissionError(f"{label} bytes are not canonical JSON plus LF")
    return value


def _exact_string(value: object, *, field: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise ProductionSubmissionError(f"{field} must be an exact string")
    return value


def _safe_key(value: object, *, field: str) -> str:
    key = _exact_string(value, field=field)
    if (
        key.startswith("/")
        or key.endswith("/")
        or "\\" in key
        or any(character in key for character in "*?[]")
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in key)
    ):
        raise ProductionSubmissionError(f"{field} is not a safe campaign key")
    segments = key.split("/")
    if any(not segment or segment in {".", ".."} for segment in segments):
        raise ProductionSubmissionError(f"{field} is not a safe campaign key")
    return key


def _digest(value: object, *, field: str) -> str:
    digest = _exact_string(value, field=field)
    if _SHA256.fullmatch(digest) is None:
        raise ProductionSubmissionError(f"{field} must be lowercase SHA-256")
    return digest


def _run_id(value: object, *, field: str = "run_id") -> str:
    run_id = _exact_string(value, field=field)
    if _RUN_ID.fullmatch(run_id) is None:
        raise ProductionSubmissionError(f"{field} is invalid")
    return run_id


def _bucket(value: object) -> str:
    bucket = _exact_string(value, field="bucket")
    if _BUCKET.fullmatch(bucket) is None:
        raise ProductionSubmissionError("bucket is invalid")
    return bucket


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProductionSubmissionError(f"{field} must be an exact integer")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ProductionSubmissionError(f"{field} must be an exact finite float")
    return value


def _canonical_time(value: object, *, field: str) -> tuple[str, datetime]:
    if type(value) is datetime:
        parsed = value
        if parsed.tzinfo is None:
            raise ProductionSubmissionError(f"{field} must be timezone-aware")
        rendered = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return rendered, parsed.astimezone(timezone.utc)
    if type(value) is not str or not value.endswith("Z"):
        raise ProductionSubmissionError(f"{field} must be canonical UTC Z text")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except (ValueError, OverflowError) as error:
        raise ProductionSubmissionError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ProductionSubmissionError(f"{field} must be timezone-aware")
    rendered = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if rendered != value:
        raise ProductionSubmissionError(f"{field} is not canonical UTC text")
    return rendered, parsed.astimezone(timezone.utc)


def _exact_fields(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ProductionSubmissionError(f"{label} schema mismatch")
    _require_exact_json(value, field=label)
    return value


def _upstream_record_contract(
    value: dict[str, object],
    *,
    label: str,
    schema_version: int,
    record_type: str,
) -> None:
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != schema_version
        or type(value.get("record_type")) is not str
        or value.get("record_type") != record_type
    ):
        raise ProductionSubmissionError(f"{label} schema mismatch")


def _self_hash(
    value: dict[str, object],
    *,
    digest_field: str,
    label: str,
) -> None:
    digest = _digest(value.get(digest_field), field=digest_field)
    if digest != _body_sha(value, digest_field):
        raise ProductionSubmissionError(f"{label} body SHA-256 mismatch")


def _artifact(
    value: object,
    *,
    label: str,
) -> tuple[VersionedJsonArtifact, dict[str, object], str]:
    if type(value) is not VersionedJsonArtifact:
        raise ProductionSubmissionError(
            f"{label} must be an exact VersionedJsonArtifact"
        )
    key = _safe_key(value.key, field=f"{label}.key")
    if type(value.raw) is not bytes:
        raise ProductionSubmissionError(f"{label}.raw must be exact bytes")
    version_id = require_opaque_version_id(
        value.version_id,
        field=f"{label}.version_id",
    )
    raw_sha = _sha(value.raw)
    parsed = _parse_raw(value.raw, label=label)
    return (
        VersionedJsonArtifact(key=key, raw=value.raw, version_id=version_id),
        parsed,
        raw_sha,
    )


def _validate_manifest(value: dict[str, object]) -> dict[str, object]:
    manifest = _exact_fields(value, _MANIFEST_FIELDS, label="bundle manifest")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("record_type") != "glm52_sky_campaign_bundle_v1"
    ):
        raise ProductionSubmissionError("bundle manifest schema mismatch")
    _run_id(manifest.get("run_id"))
    _exact_string(manifest.get("bucket"), field="bundle manifest bucket")
    _digest(
        manifest.get("descriptor_body_sha256"),
        field="descriptor_body_sha256",
    )
    rows = manifest.get("files")
    if type(rows) is not list or len(rows) != len(_MANIFEST_ROLE_ORDERS):
        raise ProductionSubmissionError("bundle manifest role inventory mismatch")
    local_names: set[str] = set()
    keys: set[str] = set()
    normalized_roles: list[tuple[str, int]] = []
    for index, item in enumerate(rows):
        row = _exact_fields(
            item,
            _MANIFEST_ROW_FIELDS,
            label=f"bundle manifest row {index}",
        )
        local_name = _exact_string(
            row.get("local_name"),
            field=f"bundle manifest row {index} local_name",
        )
        key = _safe_key(
            row.get("key"),
            field=f"bundle manifest row {index} key",
        )
        role = _exact_string(
            row.get("role"),
            field=f"bundle manifest row {index} role",
        )
        stage_order = _integer(
            row.get("stage_order"),
            field=f"bundle manifest row {index} stage_order",
        )
        _integer(
            row.get("size"),
            field=f"bundle manifest row {index} size",
        )
        _digest(
            row.get("sha256"),
            field=f"bundle manifest row {index} sha256",
        )
        if local_name in local_names or key in keys:
            raise ProductionSubmissionError("bundle manifest coordinates are duplicate")
        local_names.add(local_name)
        keys.add(key)
        normalized_roles.append((role, stage_order))
    if tuple(normalized_roles) != _MANIFEST_ROLE_ORDERS:
        raise ProductionSubmissionError(
            "bundle manifest roles or stage order are not exact"
        )
    _self_hash(
        manifest,
        digest_field="bundle_manifest_body_sha256",
        label="bundle manifest",
    )
    return manifest


def _validate_staged(value: dict[str, object]) -> dict[str, object]:
    staged = _exact_fields(value, _STAGED_FIELDS, label="staged readiness")
    if (
        type(staged.get("schema_version")) is not int
        or staged["schema_version"] != 2
        or staged.get("record_type") != "glm52_staged_control_plane_ready_v2"
    ):
        raise ProductionSubmissionError("staged readiness schema mismatch")
    _run_id(staged.get("run_id"))
    for field in (
        "descriptor_key",
        "bundle_manifest_key",
        "artifact_audit_key",
    ):
        _safe_key(staged.get(field), field=field)
    for field in (
        "descriptor_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "bundle_manifest_file_sha256",
        "bundle_manifest_body_sha256",
        "artifact_audit_sha256",
        "ready_body_sha256",
    ):
        _digest(staged.get(field), field=field)
    versions = _exact_fields(
        staged.get("staged_object_version_ids"),
        _STAGED_VERSION_FIELDS,
        label="staged object VersionIds",
    )
    for role, version_id in versions.items():
        require_opaque_version_id(version_id, field=f"{role} VersionId")
    require_opaque_version_id(
        staged.get("bundle_manifest_version_id"),
        field="bundle_manifest_version_id",
    )
    staged_text, staged_at = _canonical_time(
        staged.get("staged_at"),
        field="staged_at",
    )
    if staged_text != staged["staged_at"] or staged_at.microsecond:
        raise ProductionSubmissionError("staged_at must be canonical whole-second UTC")
    _self_hash(staged, digest_field="ready_body_sha256", label="staged readiness")
    return staged


def _validate_rehearsal(value: dict[str, object]) -> dict[str, object]:
    rehearsal = _exact_fields(value, _REHEARSAL_FIELDS, label="rehearsal")
    if (
        type(rehearsal.get("schema_version")) is not int
        or rehearsal["schema_version"] != 2
        or rehearsal.get("record_type") != "glm52_staged_control_plane_rehearsal_v2"
        or rehearsal.get("status") != "passed_before_cuda_h100_boundary"
    ):
        raise ProductionSubmissionError("rehearsal schema or status mismatch")
    _run_id(rehearsal.get("run_id"))
    for field in (
        "descriptor_key",
        "staged_readiness_key",
        "artifact_inventory_key",
        "artifact_audit_key",
    ):
        _safe_key(rehearsal.get(field), field=field)
    for field in (
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
    ):
        _digest(rehearsal.get(field), field=field)
    fixed = {
        "production_repo_path": "/opt/keep-campaign/repo",
        "production_resume_root": "/mnt/nvme/glm52-campaign",
        "skypilot_task_path": (
            "/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/glm52-campaign.yaml"
        ),
        "skypilot_version": "0.13.0",
        "skypilot_task_name": "glm52-campaign",
    }
    if any(rehearsal.get(field) != expected for field, expected in fixed.items()):
        raise ProductionSubmissionError("rehearsal fixed production contract drifted")
    extracted = _exact_string(
        rehearsal.get("extracted_repo_path"),
        field="extracted_repo_path",
    )
    extracted_path = PurePosixPath(extracted)
    if (
        "\x00" in extracted
        or not extracted_path.is_absolute()
        or ".." in extracted_path.parts
    ):
        raise ProductionSubmissionError("extracted_repo_path is not an exact path")
    completed_text, _completed_at = _canonical_time(
        rehearsal.get("completed_at"),
        field="completed_at",
    )
    if completed_text != rehearsal["completed_at"]:
        raise ProductionSubmissionError("rehearsal completed_at is not canonical")
    _self_hash(
        rehearsal,
        digest_field="rehearsal_body_sha256",
        label="rehearsal",
    )
    return rehearsal


def _submission_id_from_descriptor_key(key: str, run_id: str) -> str:
    prefix = f"campaigns/{run_id}/submissions/"
    suffix = "/campaign-descriptor-v2.json"
    if not key.startswith(prefix) or not key.endswith(suffix):
        raise ProductionSubmissionError("descriptor key is not exact")
    submission_id = key[len(prefix) : -len(suffix)]
    if (
        not submission_id
        or "/" in submission_id
        or submission_id in {".", ".."}
        or any(character in submission_id for character in "\\*?[]")
        or any(
            ord(character) < 0x21 or ord(character) > 0x7E
            for character in submission_id
        )
    ):
        raise ProductionSubmissionError("submission_id is not a safe segment")
    return submission_id


def _manifest_rows(value: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["role"]): row for row in value["files"]}


def _validate_builder(
    *,
    descriptor: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    staged_readiness: VersionedJsonArtifact,
    bundle_manifest: VersionedJsonArtifact,
    rehearsal_evidence: VersionedJsonArtifact,
    cache_seed_acceptance: VersionedJsonArtifact,
    h100_resume_ready: VersionedJsonArtifact,
    first_h100_runtime_allocation: VersionedJsonArtifact,
    replacement_h100_runtime_allocation: VersionedJsonArtifact,
    closed_gpu_spend_snapshot: VersionedJsonArtifact,
    intent_at: Union[datetime, str],
) -> dict[str, object]:
    descriptor_artifact, descriptor_value, descriptor_file_sha = _artifact(
        descriptor,
        label="descriptor",
    )
    approval_artifact, approval_value, approval_file_sha = _artifact(
        approval,
        label="approval",
    )
    staged_artifact, staged_value, staged_file_sha = _artifact(
        staged_readiness,
        label="staged readiness",
    )
    manifest_artifact, manifest_value, manifest_file_sha = _artifact(
        bundle_manifest,
        label="bundle manifest",
    )
    rehearsal_artifact, rehearsal_value, rehearsal_file_sha = _artifact(
        rehearsal_evidence,
        label="rehearsal evidence",
    )
    cache_artifact, cache_value, cache_file_sha = _artifact(
        cache_seed_acceptance,
        label="cache-seed acceptance",
    )
    h100_artifact, h100_value, h100_file_sha = _artifact(
        h100_resume_ready,
        label="H100 resume ready",
    )
    first_artifact, first_value, first_file_sha = _artifact(
        first_h100_runtime_allocation,
        label="first H100 runtime allocation",
    )
    replacement_artifact, replacement_value, replacement_file_sha = _artifact(
        replacement_h100_runtime_allocation,
        label="replacement H100 runtime allocation",
    )
    snapshot_artifact, snapshot_value, snapshot_file_sha = _artifact(
        closed_gpu_spend_snapshot,
        label="closed GPU spend snapshot",
    )

    descriptor_record = validate_sky_campaign_descriptor(descriptor_value)
    approval_record = validate_gpu_spend_approval(approval_value)
    staged_record = _validate_staged(staged_value)
    manifest_record = _validate_manifest(manifest_value)
    rehearsal_record = _validate_rehearsal(rehearsal_value)
    cache_record = validate_qualification_cache_seed_accepted(cache_value)
    h100_record = validate_h100_resume_ready(h100_value)
    first_record = validate_h100_runtime_allocation(first_value)
    replacement_record = validate_h100_runtime_allocation(replacement_value)
    snapshot_record = validate_gpu_spend_snapshot(snapshot_value)

    _upstream_record_contract(
        descriptor_record,
        label="descriptor",
        schema_version=2,
        record_type="glm52_sky_campaign_descriptor_v2",
    )
    _upstream_record_contract(
        approval_record,
        label="approval",
        schema_version=1,
        record_type="glm52_gpu_spend_approval_v1",
    )
    _upstream_record_contract(
        cache_record,
        label="cache acceptance",
        schema_version=1,
        record_type="glm52_qualification_cache_seed_accepted_v1",
    )
    _upstream_record_contract(
        h100_record,
        label="H100 resume ready",
        schema_version=1,
        record_type="glm52_h100_resume_ready_v1",
    )
    _upstream_record_contract(
        snapshot_record,
        label="closed GPU spend snapshot",
        schema_version=1,
        record_type="glm52_gpu_spend_snapshot_v1",
    )

    run_id = _run_id(descriptor_record.get("run_id"))
    if (
        descriptor_record.get("account_id") != "246813579024"
        or descriptor_record.get("provider") != "aws"
        or descriptor_record.get("region") != "us-west-2"
        or descriptor_record.get("instance_type") != "p5.48xlarge"
        or type(descriptor_record.get("instance_count")) is not int
        or descriptor_record.get("instance_count") != 1
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
        raise ProductionSubmissionError("production descriptor authority mismatch")
    submission_id = _submission_id_from_descriptor_key(
        descriptor_artifact.key,
        run_id,
    )
    if descriptor_record.get("campaign_descriptor_key") != descriptor_artifact.key:
        raise ProductionSubmissionError("descriptor envelope key mismatch")
    descriptor_body_sha = _digest(
        descriptor_record.get("descriptor_body_sha256"),
        field="descriptor_body_sha256",
    )
    campaign_identity_sha = _digest(
        descriptor_record.get("campaign_identity_sha256"),
        field="campaign_identity_sha256",
    )
    repo_tar_key = _safe_key(
        descriptor_record.get("repo_tar_key"),
        field="repo_tar_key",
    )
    repo_tar_sha = _digest(
        descriptor_record.get("repo_tar_sha256"),
        field="repo_tar_sha256",
    )
    if (
        descriptor_record.get("approval_key") != approval_artifact.key
        or descriptor_record.get("approval_sha256") != approval_file_sha
        or type(approval_record.get("approved_hourly_usd")) is not float
        or approval_record.get("approved_hourly_usd") != 55.04
        or type(approval_record.get("approved_gpu_hours")) is not int
        or approval_record.get("approved_gpu_hours") != 24
        or type(approval_record.get("approved_gpu_cost_usd")) is not float
        or approval_record.get("approved_gpu_cost_usd") != 1_320.96
        or approval_record.get("region") != "us-west-2"
        or approval_record.get("instance_type") != "p5.48xlarge"
        or approval_record.get("includes_qualification") is not True
        or approval_record.get("includes_recovery_instances") is not True
    ):
        raise ProductionSubmissionError("descriptor/approval authority mismatch")
    approval_body_sha = _digest(
        approval_record.get("approval_body_sha256"),
        field="approval_body_sha256",
    )
    artifacts = _exact_fields(
        descriptor_record.get("artifacts"),
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
        },
        label="descriptor artifacts",
    )
    cache_prefix = _exact_string(
        artifacts.get("qualification_cache_prefix"),
        field="qualification_cache_prefix",
    )
    cache_manifest_sha = _digest(
        artifacts.get("qualification_cache_manifest_sha256"),
        field="qualification_cache_manifest_sha256",
    )
    if cache_manifest_sha == "0" * 64:
        raise ProductionSubmissionError("qualification cache is still empty")

    expected_staged_key = (
        f"campaigns/{run_id}/submissions/{submission_id}/"
        "STAGED_CONTROL_PLANE_READY.json"
    )
    expected_manifest_prefix = (
        f"campaigns/{run_id}/submissions/{submission_id}/bundle-manifests/"
    )
    expected_manifest_key = (
        f"{expected_manifest_prefix}"
        f"{manifest_record['bundle_manifest_body_sha256']}/"
        "bundle-manifest-v1.json"
    )
    staged_versions = staged_record["staged_object_version_ids"]
    if (
        staged_artifact.key != expected_staged_key
        or staged_record.get("run_id") != run_id
        or staged_record.get("descriptor_key") != descriptor_artifact.key
        or staged_record.get("descriptor_sha256") != descriptor_file_sha
        or staged_record.get("descriptor_body_sha256") != descriptor_body_sha
        or staged_record.get("campaign_identity_sha256") != campaign_identity_sha
        or staged_record.get("bundle_manifest_key") != manifest_artifact.key
        or staged_record.get("bundle_manifest_file_sha256") != manifest_file_sha
        or staged_record.get("bundle_manifest_body_sha256")
        != manifest_record["bundle_manifest_body_sha256"]
        or staged_record.get("bundle_manifest_version_id")
        != manifest_artifact.version_id
        or staged_versions["descriptor"] != descriptor_artifact.version_id
        or staged_versions["approval"] != approval_artifact.version_id
        or manifest_artifact.key != expected_manifest_key
        or manifest_record.get("run_id") != run_id
        or manifest_record.get("bucket") != descriptor_record["bucket"]
        or manifest_record.get("descriptor_body_sha256") != descriptor_body_sha
    ):
        raise ProductionSubmissionError("staging source chain mismatch")
    manifest_rows = _manifest_rows(manifest_record)
    row_expectations = {
        "repository_tar": (repo_tar_key, repo_tar_sha),
        "approval": (approval_artifact.key, approval_file_sha),
        "training_config": (
            artifacts["training_config_key"],
            artifacts["training_config_sha256"],
        ),
        "artifact_inventory": (
            artifacts["artifact_inventory_key"],
            artifacts["artifact_inventory_sha256"],
        ),
        "descriptor": (descriptor_artifact.key, descriptor_file_sha),
    }
    for role, (expected_key, expected_sha) in row_expectations.items():
        row = manifest_rows[role]
        if row["key"] != expected_key or row["sha256"] != expected_sha:
            raise ProductionSubmissionError(f"manifest {role} ancestry mismatch")

    expected_rehearsal_key = (
        f"campaigns/{run_id}/qualification/rehearsals/"
        f"{rehearsal_record['rehearsal_body_sha256']}/"
        "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    )
    if (
        rehearsal_artifact.key != expected_rehearsal_key
        or rehearsal_record.get("run_id") != run_id
        or rehearsal_record.get("campaign_identity_sha256") != campaign_identity_sha
        or rehearsal_record.get("descriptor_key") != descriptor_artifact.key
        or rehearsal_record.get("descriptor_file_sha256") != descriptor_file_sha
        or rehearsal_record.get("descriptor_body_sha256") != descriptor_body_sha
        or rehearsal_record.get("repo_tar_sha256") != repo_tar_sha
        or rehearsal_record.get("staged_readiness_key") != staged_artifact.key
        or rehearsal_record.get("staged_readiness_file_sha256") != staged_file_sha
        or rehearsal_record.get("staged_readiness_body_sha256")
        != staged_record["ready_body_sha256"]
        or rehearsal_record.get("artifact_inventory_key")
        != artifacts["artifact_inventory_key"]
        or rehearsal_record.get("artifact_inventory_file_sha256")
        != artifacts["artifact_inventory_sha256"]
        or rehearsal_record.get("artifact_inventory_key")
        != manifest_rows["artifact_inventory"]["key"]
        or rehearsal_record.get("artifact_inventory_file_sha256")
        != manifest_rows["artifact_inventory"]["sha256"]
        or rehearsal_record.get("artifact_audit_key")
        != staged_record["artifact_audit_key"]
        or rehearsal_record.get("artifact_audit_file_sha256")
        != staged_record["artifact_audit_sha256"]
    ):
        raise ProductionSubmissionError("rehearsal source chain mismatch")

    cache_authority = cache_record.get("seed_authority")
    cache_audit = cache_record.get("cache_audit")
    spend_closure = cache_record.get("spend_closure")
    if not all(
        type(item) is dict for item in (cache_authority, cache_audit, spend_closure)
    ):
        raise ProductionSubmissionError("cache acceptance nested schema mismatch")
    expected_cache_key = (
        f"campaigns/{run_id}/qualification-cache-seed/accepted/"
        f"{cache_record['acceptance_body_sha256']}/"
        "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
    )
    if (
        cache_artifact.key != expected_cache_key
        or cache_authority.get("run_id") != run_id
        or cache_authority.get("account_id") != descriptor_record["account_id"]
        or cache_authority.get("region") != descriptor_record["region"]
        or cache_authority.get("bucket") != descriptor_record["bucket"]
        or cache_authority.get("repo_tar_key") != repo_tar_key
        or cache_authority.get("repo_tar_sha256") != repo_tar_sha
        or cache_authority.get("approval_key") != approval_artifact.key
        or cache_authority.get("approval_sha256") != approval_file_sha
        or cache_audit.get("cache_prefix") != cache_prefix
        or cache_audit.get("manifest_file_sha256") != cache_manifest_sha
        or cache_audit.get("semantic_audit_pass") is not True
        or type(cache_audit.get("supervised_position_count")) is not int
        or cache_audit["supervised_position_count"] <= 0
    ):
        raise ProductionSubmissionError("cache acceptance source chain mismatch")
    seed_instance_id = _exact_string(
        spend_closure.get("instance_id"),
        field="cache seed instance_id",
    )

    expected_h100_key = f"campaigns/{run_id}/qualification/H100_RESUME_READY.json"
    first_instance_id = _exact_string(
        h100_record.get("first_instance_id"),
        field="H100 first instance_id",
    )
    replacement_instance_id = _exact_string(
        h100_record.get("replacement_instance_id"),
        field="H100 replacement instance_id",
    )
    if (
        h100_artifact.key != expected_h100_key
        or h100_record.get("run_id") != run_id
        or h100_record.get("campaign_identity_sha256") != campaign_identity_sha
        or h100_record.get("repo_tar_sha256") != repo_tar_sha
        or h100_record.get("qualification_cache_manifest_sha256") != cache_manifest_sha
        or type(h100_record.get("training_sample_steps")) is not int
        or h100_record.get("training_sample_steps") != 2
        or h100_record.get("cross_node_resume") is not True
        or type(h100_record.get("peak_gpu_gib")) is not float
        or not math.isfinite(h100_record["peak_gpu_gib"])
        or not 0 <= h100_record["peak_gpu_gib"] < 70
        or len({seed_instance_id, first_instance_id, replacement_instance_id}) != 3
    ):
        raise ProductionSubmissionError("H100 readiness source chain mismatch")

    allocation_key = f"campaigns/{run_id}/runtime/GPU_RUNTIME_ALLOCATION.json"
    first_body_sha = _digest(
        first_record.get("allocation_body_sha256"),
        field="first allocation_body_sha256",
    )
    replacement_body_sha = _digest(
        replacement_record.get("allocation_body_sha256"),
        field="replacement allocation_body_sha256",
    )
    first_record_sha = _digest(
        first_record.get("gpu_spend_record_sha256"),
        field="first gpu_spend_record_sha256",
    )
    replacement_record_sha = _digest(
        replacement_record.get("gpu_spend_record_sha256"),
        field="replacement gpu_spend_record_sha256",
    )
    if (
        first_artifact.key != allocation_key
        or replacement_artifact.key != allocation_key
        or first_artifact.version_id == replacement_artifact.version_id
        or first_artifact.raw == replacement_artifact.raw
        or first_file_sha == replacement_file_sha
        or first_body_sha == replacement_body_sha
        or first_record_sha == replacement_record_sha
        or first_record.get("instance_id") != first_instance_id
        or replacement_record.get("instance_id") != replacement_instance_id
        or first_record.get("instance_id") == replacement_record.get("instance_id")
        or first_record.get("run_id") != run_id
        or replacement_record.get("run_id") != run_id
        or first_record.get("job_id") != f"{run_id}-qualification"
        or replacement_record.get("job_id") != f"{run_id}-qualification"
        or first_record.get("approval_sha256") != approval_file_sha
        or replacement_record.get("approval_sha256") != approval_file_sha
        or h100_record.get("first_allocation_record_sha256") != first_record_sha
        or h100_record.get("replacement_allocation_record_sha256")
        != replacement_record_sha
    ):
        raise ProductionSubmissionError("runtime allocation source chain mismatch")
    _finite_float(
        first_record.get("estimated_gpu_cost_usd"),
        field="first estimated_gpu_cost_usd",
    )
    _finite_float(
        replacement_record.get("estimated_gpu_cost_usd"),
        field="replacement estimated_gpu_cost_usd",
    )

    snapshot_body_sha = _digest(
        snapshot_record.get("snapshot_body_sha256"),
        field="snapshot_body_sha256",
    )
    expected_snapshot_key = (
        f"campaigns/{run_id}/spend-snapshots/{snapshot_body_sha}/"
        "GPU_SPEND_SNAPSHOT.json"
    )
    spend_genesis_sha = _digest(
        snapshot_record.get("gpu_spend_ledger_genesis_sha256"),
        field="gpu_spend_ledger_genesis_sha256",
    )
    if (
        snapshot_artifact.key != expected_snapshot_key
        or snapshot_record.get("run_id") != run_id
        or snapshot_record.get("descriptor_sha256") != descriptor_file_sha
        or snapshot_record.get("descriptor_body_sha256") != descriptor_body_sha
        or snapshot_record.get("campaign_identity_sha256") != campaign_identity_sha
        or snapshot_record.get("approval_sha256") != approval_file_sha
        or snapshot_record.get("approval_body_sha256") != approval_body_sha
        or first_record.get("gpu_spend_authority_sha256") != spend_genesis_sha
        or replacement_record.get("gpu_spend_authority_sha256") != spend_genesis_sha
    ):
        raise ProductionSubmissionError("closed spend source chain mismatch")
    if type(snapshot_record.get("open_allocation_count")) is not int or (
        snapshot_record["open_allocation_count"] != 0
    ):
        raise ProductionSubmissionError("closed spend has an open allocation")
    if (
        _finite_float(
            snapshot_record.get("hourly_cost_usd"),
            field="hourly_cost_usd",
        )
        != 55.04
    ):
        raise ProductionSubmissionError("closed spend hourly rate mismatch")
    for money_field in (
        "approved_gpu_cost_usd",
        "consumed_gpu_cost_usd",
        "remaining_gpu_cost_usd",
    ):
        _finite_float(snapshot_record.get(money_field), field=money_field)
    remaining_seconds = _integer(
        snapshot_record.get("remaining_gpu_seconds"),
        field="remaining_gpu_seconds",
    )
    if remaining_seconds <= 3_600:
        raise ProductionSubmissionError("closed spend remaining time is insufficient")
    if snapshot_record["remaining_gpu_cost_usd"] <= 55.04:
        raise ProductionSubmissionError("closed spend remaining cost is insufficient")
    closed_ids = snapshot_record.get("ec2_allocation_instance_ids")
    if (
        type(closed_ids) is not list
        or len(set(closed_ids)) != len(closed_ids)
        or any(
            type(item) is not str or _INSTANCE_ID.fullmatch(item) is None
            for item in closed_ids
        )
        or not {seed_instance_id, first_instance_id, replacement_instance_id}.issubset(
            set(closed_ids)
        )
    ):
        raise ProductionSubmissionError("closed instance inventory is incomplete")

    cache_accepted_text, cache_accepted_at = _canonical_time(
        cache_record.get("accepted_at"),
        field="cache_seed_accepted_at",
    )
    staged_text, staged_at_value = _canonical_time(
        staged_record.get("staged_at"),
        field="staged_at",
    )
    rehearsal_text, rehearsal_at_value = _canonical_time(
        rehearsal_record.get("completed_at"),
        field="rehearsal_completed_at",
    )
    h100_text, h100_at_value = _canonical_time(
        h100_record.get("completed_at"),
        field="h100_completed_at",
    )
    snapshot_text, snapshot_at_value = _canonical_time(
        snapshot_record.get("observed_at"),
        field="spend_snapshot_observed_at",
    )
    intent_text, intent_at_value = _canonical_time(intent_at, field="intent_at")
    must_start_text, must_start_at = _canonical_time(
        descriptor_record.get("must_start_by"),
        field="must_start_by",
    )
    if not (
        cache_accepted_at
        <= staged_at_value
        <= rehearsal_at_value
        <= h100_at_value
        <= snapshot_at_value
        <= intent_at_value
        < must_start_at
    ):
        raise ProductionSubmissionError("production source chronology mismatch")
    if intent_at_value - snapshot_at_value > timedelta(
        seconds=60
    ) or must_start_at - intent_at_value > timedelta(hours=12):
        raise ProductionSubmissionError("production intent time window mismatch")

    first_launched_text, first_launched = _canonical_time(
        first_record.get("launched_at"),
        field="first allocation launched_at",
    )
    first_observed_text, first_observed = _canonical_time(
        first_record.get("observed_at"),
        field="first allocation observed_at",
    )
    replacement_launched_text, replacement_launched = _canonical_time(
        replacement_record.get("launched_at"),
        field="replacement allocation launched_at",
    )
    replacement_observed_text, replacement_observed = _canonical_time(
        replacement_record.get("observed_at"),
        field="replacement allocation observed_at",
    )
    if not (
        rehearsal_at_value
        <= first_launched
        <= first_observed
        <= replacement_launched
        <= replacement_observed
        <= h100_at_value
    ):
        raise ProductionSubmissionError("runtime allocation chronology mismatch")
    if (
        first_launched_text != first_record["launched_at"]
        or first_observed_text != first_record["observed_at"]
        or replacement_launched_text != replacement_record["launched_at"]
        or replacement_observed_text != replacement_record["observed_at"]
    ):
        raise ProductionSubmissionError("allocation timestamps are not canonical")

    contract = record_contract(
        managed_mode="production",
        record_kind="submission-intent",
    )
    body: dict[str, object] = {
        "schema_version": contract.schema_version,
        "record_type": contract.record_type,
        "managed_mode": "production",
        "account_id": descriptor_record["account_id"],
        "region": descriptor_record["region"],
        "bucket": descriptor_record["bucket"],
        "run_id": run_id,
        "campaign_identity_sha256": campaign_identity_sha,
        "sky_job_name": expected_sky_job_name(
            run_id=run_id,
            managed_mode="production",
        ),
        "must_start_by": must_start_text,
        "intent_at": intent_text,
        "descriptor_key": descriptor_artifact.key,
        "descriptor_file_sha256": descriptor_file_sha,
        "descriptor_body_sha256": descriptor_body_sha,
        "descriptor_version_id": descriptor_artifact.version_id,
        "repo_tar_key": repo_tar_key,
        "repo_tar_sha256": repo_tar_sha,
        "approval_key": approval_artifact.key,
        "approval_file_sha256": approval_file_sha,
        "approval_body_sha256": approval_body_sha,
        "approval_version_id": approval_artifact.version_id,
        "staged_readiness_key": staged_artifact.key,
        "staged_readiness_file_sha256": staged_file_sha,
        "staged_readiness_body_sha256": staged_record["ready_body_sha256"],
        "staged_readiness_version_id": staged_artifact.version_id,
        "staged_at": staged_text,
        "bundle_manifest_key": manifest_artifact.key,
        "bundle_manifest_file_sha256": manifest_file_sha,
        "bundle_manifest_body_sha256": manifest_record["bundle_manifest_body_sha256"],
        "bundle_manifest_version_id": manifest_artifact.version_id,
        "rehearsal_evidence_key": rehearsal_artifact.key,
        "rehearsal_evidence_file_sha256": rehearsal_file_sha,
        "rehearsal_evidence_body_sha256": rehearsal_record["rehearsal_body_sha256"],
        "rehearsal_evidence_version_id": rehearsal_artifact.version_id,
        "rehearsal_completed_at": rehearsal_text,
        "cache_seed_acceptance_key": cache_artifact.key,
        "cache_seed_acceptance_file_sha256": cache_file_sha,
        "cache_seed_acceptance_body_sha256": cache_record["acceptance_body_sha256"],
        "cache_seed_acceptance_version_id": cache_artifact.version_id,
        "qualification_cache_prefix": cache_prefix,
        "qualification_cache_manifest_sha256": cache_manifest_sha,
        "cache_seed_instance_id": seed_instance_id,
        "cache_seed_accepted_at": cache_accepted_text,
        "h100_resume_ready_key": h100_artifact.key,
        "h100_resume_ready_file_sha256": h100_file_sha,
        "h100_resume_ready_body_sha256": h100_record["ready_body_sha256"],
        "h100_resume_ready_version_id": h100_artifact.version_id,
        "h100_first_instance_id": first_instance_id,
        "h100_replacement_instance_id": replacement_instance_id,
        "h100_completed_at": h100_text,
        "first_h100_allocation_key": first_artifact.key,
        "first_h100_allocation_file_sha256": first_file_sha,
        "first_h100_allocation_body_sha256": first_body_sha,
        "first_h100_allocation_record_sha256": first_record_sha,
        "first_h100_allocation_version_id": first_artifact.version_id,
        "replacement_h100_allocation_key": replacement_artifact.key,
        "replacement_h100_allocation_file_sha256": replacement_file_sha,
        "replacement_h100_allocation_body_sha256": replacement_body_sha,
        "replacement_h100_allocation_record_sha256": replacement_record_sha,
        "replacement_h100_allocation_version_id": replacement_artifact.version_id,
        "gpu_spend_snapshot_key": snapshot_artifact.key,
        "gpu_spend_snapshot_file_sha256": snapshot_file_sha,
        "gpu_spend_snapshot_body_sha256": snapshot_body_sha,
        "gpu_spend_snapshot_version_id": snapshot_artifact.version_id,
        "gpu_spend_ledger_tip_record_sha256": snapshot_record[
            "gpu_spend_ledger_tip_record_sha256"
        ],
        "gpu_spend_ledger_genesis_sha256": spend_genesis_sha,
        "ec2_allocation_history_sha256": snapshot_record[
            "ec2_allocation_history_sha256"
        ],
        "closed_instance_ids": list(closed_ids),
        "open_allocation_count": snapshot_record["open_allocation_count"],
        "approved_gpu_runtime_seconds": snapshot_record["approved_gpu_runtime_seconds"],
        "approved_gpu_cost_usd": snapshot_record["approved_gpu_cost_usd"],
        "hourly_cost_usd": snapshot_record["hourly_cost_usd"],
        "consumed_gpu_seconds": snapshot_record["consumed_gpu_seconds"],
        "consumed_gpu_cost_usd": snapshot_record["consumed_gpu_cost_usd"],
        "remaining_gpu_seconds": remaining_seconds,
        "remaining_gpu_cost_usd": snapshot_record["remaining_gpu_cost_usd"],
        "spend_snapshot_observed_at": snapshot_text,
    }
    intent = {
        **body,
        contract.digest_field: _sha(_canonical(body)),
    }
    return _validate_intent(intent)


def _validate_intent(value: Mapping[str, object]) -> dict[str, object]:
    intent = _exact_fields(value, _INTENT_FIELDS, label="production intent")
    contract = record_contract(
        managed_mode="production",
        record_kind="submission-intent",
    )
    if (
        type(intent.get("schema_version")) is not int
        or intent["schema_version"] != contract.schema_version
        or intent.get("record_type") != contract.record_type
        or intent.get("managed_mode") != "production"
    ):
        raise ProductionSubmissionError("production intent discriminant mismatch")
    _self_hash(
        intent,
        digest_field=contract.digest_field,
        label="production intent",
    )
    run_id = _run_id(intent.get("run_id"))
    if intent.get("sky_job_name") != expected_sky_job_name(
        run_id=run_id,
        managed_mode="production",
    ):
        raise ProductionSubmissionError("production intent Sky job name mismatch")
    if (
        intent.get("account_id") != "246813579024"
        or intent.get("region") != "us-west-2"
    ):
        raise ProductionSubmissionError("production intent cloud identity mismatch")
    _bucket(intent.get("bucket"))
    for field in _INTENT_SHA_FIELDS:
        _digest(intent.get(field), field=field)
    for field in _INTENT_VERSION_FIELDS:
        require_opaque_version_id(intent.get(field), field=field)
    for field in _INTENT_KEY_FIELDS:
        _safe_key(intent.get(field), field=field)

    submission_id = _submission_id_from_descriptor_key(
        str(intent["descriptor_key"]),
        run_id,
    )
    expected_staged_key = (
        f"campaigns/{run_id}/submissions/{submission_id}/"
        "STAGED_CONTROL_PLANE_READY.json"
    )
    expected_manifest_key = (
        f"campaigns/{run_id}/submissions/{submission_id}/bundle-manifests/"
        f"{intent['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
    )
    expected_rehearsal_key = (
        f"campaigns/{run_id}/qualification/rehearsals/"
        f"{intent['rehearsal_evidence_body_sha256']}/"
        "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    )
    expected_cache_key = (
        f"campaigns/{run_id}/qualification-cache-seed/accepted/"
        f"{intent['cache_seed_acceptance_body_sha256']}/"
        "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
    )
    expected_h100_key = f"campaigns/{run_id}/qualification/H100_RESUME_READY.json"
    expected_allocation_key = f"campaigns/{run_id}/runtime/GPU_RUNTIME_ALLOCATION.json"
    expected_snapshot_key = (
        f"campaigns/{run_id}/spend-snapshots/"
        f"{intent['gpu_spend_snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
    )
    key_expectations = {
        "staged_readiness_key": expected_staged_key,
        "bundle_manifest_key": expected_manifest_key,
        "rehearsal_evidence_key": expected_rehearsal_key,
        "cache_seed_acceptance_key": expected_cache_key,
        "h100_resume_ready_key": expected_h100_key,
        "first_h100_allocation_key": expected_allocation_key,
        "replacement_h100_allocation_key": expected_allocation_key,
        "gpu_spend_snapshot_key": expected_snapshot_key,
    }
    if any(
        intent.get(field) != expected for field, expected in key_expectations.items()
    ):
        raise ProductionSubmissionError("production intent embedded key mismatch")
    if not str(intent["repo_tar_key"]).startswith(f"campaigns/{run_id}/repository/"):
        raise ProductionSubmissionError("production repository key mismatch")
    if not str(intent["approval_key"]).startswith(f"campaigns/{run_id}/authorities/"):
        raise ProductionSubmissionError("production approval key mismatch")

    instance_fields = (
        "cache_seed_instance_id",
        "h100_first_instance_id",
        "h100_replacement_instance_id",
    )
    instance_ids = [
        _exact_string(intent.get(field), field=field) for field in instance_fields
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
        raise ProductionSubmissionError("production intent closed inventory mismatch")
    distinct_pairs = (
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
    )
    if any(intent[first] == intent[second] for first, second in distinct_pairs):
        raise ProductionSubmissionError(
            "production allocation identities are not distinct"
        )

    if type(intent.get("open_allocation_count")) is not int or (
        intent["open_allocation_count"] != 0
    ):
        raise ProductionSubmissionError("production intent is not spend-closed")
    approved_seconds = _integer(
        intent.get("approved_gpu_runtime_seconds"),
        field="approved_gpu_runtime_seconds",
        minimum=1,
    )
    consumed_seconds = _integer(
        intent.get("consumed_gpu_seconds"),
        field="consumed_gpu_seconds",
    )
    remaining_seconds = _integer(
        intent.get("remaining_gpu_seconds"),
        field="remaining_gpu_seconds",
    )
    if (
        approved_seconds != 86_400
        or consumed_seconds + remaining_seconds != approved_seconds
        or remaining_seconds <= 3_600
    ):
        raise ProductionSubmissionError("production intent runtime budget mismatch")
    for field in (
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "consumed_gpu_cost_usd",
        "remaining_gpu_cost_usd",
    ):
        _finite_float(intent.get(field), field=field)
    if (
        intent["approved_gpu_cost_usd"] != 1_320.96
        or intent["hourly_cost_usd"] != 55.04
        or intent["remaining_gpu_cost_usd"] <= 55.04
        or Decimal(str(intent["consumed_gpu_cost_usd"]))
        != (
            Decimal(consumed_seconds)
            * Decimal(str(intent["hourly_cost_usd"]))
            / Decimal(3600)
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        or (
            Decimal(str(intent["consumed_gpu_cost_usd"]))
            + Decimal(str(intent["remaining_gpu_cost_usd"]))
            != Decimal(str(intent["approved_gpu_cost_usd"]))
        )
    ):
        raise ProductionSubmissionError("production intent dollar budget mismatch")
    _exact_string(
        intent.get("qualification_cache_prefix"),
        field="qualification_cache_prefix",
    )
    if intent["qualification_cache_manifest_sha256"] == "0" * 64 or intent[
        "qualification_cache_prefix"
    ] != (
        f"qualification-cache/seeds/{run_id}/"
        f"{intent['qualification_cache_manifest_sha256']}/"
    ):
        raise ProductionSubmissionError("production intent cache identity is empty")

    cache_text, cache_time = _canonical_time(
        intent.get("cache_seed_accepted_at"),
        field="cache_seed_accepted_at",
    )
    staged_text, staged_time = _canonical_time(
        intent.get("staged_at"),
        field="staged_at",
    )
    rehearsal_text, rehearsal_time = _canonical_time(
        intent.get("rehearsal_completed_at"),
        field="rehearsal_completed_at",
    )
    h100_text, h100_time = _canonical_time(
        intent.get("h100_completed_at"),
        field="h100_completed_at",
    )
    snapshot_text, snapshot_time = _canonical_time(
        intent.get("spend_snapshot_observed_at"),
        field="spend_snapshot_observed_at",
    )
    intent_text, built_time = _canonical_time(
        intent.get("intent_at"),
        field="intent_at",
    )
    must_start_text, must_start = _canonical_time(
        intent.get("must_start_by"),
        field="must_start_by",
    )
    if (
        cache_text != intent["cache_seed_accepted_at"]
        or staged_text != intent["staged_at"]
        or rehearsal_text != intent["rehearsal_completed_at"]
        or h100_text != intent["h100_completed_at"]
        or snapshot_text != intent["spend_snapshot_observed_at"]
        or intent_text != intent["intent_at"]
        or must_start_text != intent["must_start_by"]
        or staged_time.microsecond != 0
        or not (
            cache_time
            <= staged_time
            <= rehearsal_time
            <= h100_time
            <= snapshot_time
            <= built_time
            < must_start
        )
        or built_time - snapshot_time > timedelta(seconds=60)
        or must_start - built_time > timedelta(hours=12)
    ):
        raise ProductionSubmissionError("production intent timestamp chain mismatch")
    return dict(intent)


def canonical_file_bytes(value: Mapping[str, object]) -> bytes:
    """Return exact sorted compact finite ASCII JSON plus one LF."""

    def operation() -> bytes:
        if type(value) is not dict:
            raise ProductionSubmissionError("value must be an exact object")
        return _canonical(value) + b"\n"

    return _translate(operation)


def build_production_submission_intent(
    *,
    descriptor: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    staged_readiness: VersionedJsonArtifact,
    bundle_manifest: VersionedJsonArtifact,
    rehearsal_evidence: VersionedJsonArtifact,
    cache_seed_acceptance: VersionedJsonArtifact,
    h100_resume_ready: VersionedJsonArtifact,
    first_h100_runtime_allocation: VersionedJsonArtifact,
    replacement_h100_runtime_allocation: VersionedJsonArtifact,
    closed_gpu_spend_snapshot: VersionedJsonArtifact,
    intent_at: Union[datetime, str],
) -> dict[str, object]:
    """Build a self-hashed structural production submission intent."""

    return _translate(
        lambda: _validate_builder(
            descriptor=descriptor,
            approval=approval,
            staged_readiness=staged_readiness,
            bundle_manifest=bundle_manifest,
            rehearsal_evidence=rehearsal_evidence,
            cache_seed_acceptance=cache_seed_acceptance,
            h100_resume_ready=h100_resume_ready,
            first_h100_runtime_allocation=first_h100_runtime_allocation,
            replacement_h100_runtime_allocation=(replacement_h100_runtime_allocation),
            closed_gpu_spend_snapshot=closed_gpu_spend_snapshot,
            intent_at=intent_at,
        )
    )


def validate_production_submission_intent(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate every invariant embedded in a standalone production intent."""

    return _translate(lambda: _validate_intent(value))


def production_submission_intent_s3_key(
    *,
    run_id: str,
    intent_body_sha256: str,
) -> str:
    """Return the exact content-addressed production-intent key."""

    def operation() -> str:
        exact_run_id = _run_id(run_id)
        body_sha = _digest(intent_body_sha256, field="intent_body_sha256")
        return (
            f"campaigns/{exact_run_id}/submissions/production/intents/"
            f"{body_sha}/SKYPILOT_SUBMISSION_INTENT.json"
        )

    return _translate(operation)


def production_submission_intent_file_bytes(
    value: Mapping[str, object],
) -> bytes:
    """Return validated durable production-intent bytes."""

    return _translate(lambda: _canonical(_validate_intent(value)) + b"\n")


def production_submission_intent_file_sha256(
    value: Mapping[str, object],
) -> str:
    """Hash the validated full-record production-intent file bytes."""

    return _translate(lambda: _sha(production_submission_intent_file_bytes(value)))


__all__ = [
    "ProductionSubmissionError",
    "VersionedJsonArtifact",
    "build_production_submission_intent",
    "canonical_file_bytes",
    "production_submission_intent_file_bytes",
    "production_submission_intent_file_sha256",
    "production_submission_intent_s3_key",
    "validate_production_submission_intent",
]
