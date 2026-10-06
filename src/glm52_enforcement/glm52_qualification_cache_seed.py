"""Pure-stdlib runtime validation for accepted qualification-cache seeds.

The offline cache builder and tensor auditor intentionally remain in
``mlx_vq.quality.glm52_qualification_cache_seed``.  Privileged production
Lambdas import only this validation surface, which preserves the exact accepted
record contract without bringing NumPy or the offline ML package into their
dependency closure.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import PurePosixPath

from .glm52_sky_campaign import APPROVED_ACCOUNT_ID, APPROVED_REGION
from .glm52_sky_must_start import validate_must_start_job_binding

MANIFEST_FILENAME = "glm52-teacher-signal-cache-v3-manifest.json"
READY_FILENAME = "TEACHER_CACHE_READY.json"

ACCEPTED_RECORD_TYPE = "glm52_qualification_cache_seed_accepted_v1"
ACCEPTED_SCHEMA_VERSION = 1
ACCEPTED_DIGEST_FIELD = "acceptance_body_sha256"
EXPECTED_PROMPT_ID = "teich_claude_agent-a3622521df8a9137d"
EXPECTED_TOP_K = 2048
EXPECTED_HIDDEN_SIZE = 6144
EXPECTED_SESSION_COUNT = 1
EXPECTED_MANAGED_MODE = "cache-seed"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_INSTANCE_ID = re.compile(r"^i-(?:[0-9a-f]{8}|[0-9a-f]{17})$")

_ACCEPTED_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "seed_authority",
        "cache_object_inventory",
        "cache_object_inventory_sha256",
        "cache_audit",
        "spend_closure",
        "accepted_at",
        ACCEPTED_DIGEST_FIELD,
    }
)
_SEED_AUTHORITY_FIELDS = frozenset(
    {
        "account_id",
        "region",
        "bucket",
        "run_id",
        "managed_mode",
        "target_job_id",
        "sky_job_name",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "repo_tar_key",
        "repo_tar_sha256",
        "approval_key",
        "approval_sha256",
        "submission_key",
        "submission_file_sha256",
        "submission_body_sha256",
        "submission_alias_key",
        "job_binding_key",
        "job_binding",
        "job_binding_file_sha256",
        "job_binding_body_sha256",
        "seed_ready_key",
    }
)
_INVENTORY_FIELDS = frozenset(
    {"key", "size", "sha256", "checksum_type", "etag", "version_id"}
)
_CACHE_AUDIT_FIELDS = frozenset(
    {
        "legacy_seed_ready_file_sha256",
        "legacy_seed_ready_body_sha256",
        "cache_prefix",
        "manifest_key",
        "manifest_file_sha256",
        "manifest_body_sha256",
        "teacher_cache_ready_key",
        "teacher_cache_ready_file_sha256",
        "teacher_cache_ready_body_sha256",
        "prompt_pack_key",
        "prompt_pack_file_sha256",
        "teich_pack_key",
        "teich_pack_file_sha256",
        "frozen_prompt_pack_key",
        "frozen_prompt_pack_file_sha256",
        "prompt_id",
        "session_count",
        "supervised_position_count",
        "top_k",
        "hidden_size",
        "semantic_audit_pass",
    }
)
_SPEND_CLOSURE_FIELDS = frozenset(
    {
        "allocation_key",
        "allocation_file_sha256",
        "allocation_body_sha256",
        "allocation_start_record_sha256",
        "spend_authority_sha256",
        "ledger_key",
        "ledger_file_sha256",
        "ledger_tip_record_sha256",
        "ledger_tip_event",
        "spend_status_key",
        "spend_status_file_sha256",
        "job_status_key",
        "job_status_file_sha256",
        "job_status_body_sha256",
        "job_status",
        "instance_id",
        "sky_job_name",
        "submission_submitted_at",
        "launched_at",
        "must_start_by",
        "ended_at",
        "job_observed_at",
        "consumed_gpu_seconds",
        "remaining_gpu_seconds",
        "estimated_gpu_cost_usd",
    }
)


class QualificationCacheSeedValidationError(ValueError):
    """The accepted cache seed does not close around its exact authority."""


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
        raise QualificationCacheSeedValidationError(
            "value is not canonical finite JSON"
        ) from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise QualificationCacheSeedValidationError(
            f"{label} schema mismatch: missing={missing}, extra={extra}"
        )


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise QualificationCacheSeedValidationError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise QualificationCacheSeedValidationError(
            f"{field} must be a positive integer"
        )
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise QualificationCacheSeedValidationError(
            f"{field} must be a non-negative integer"
        )
    return value


def _canonical_time(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise QualificationCacheSeedValidationError(
            f"{field} must be an ISO-8601 timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise QualificationCacheSeedValidationError(
            f"{field} must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise QualificationCacheSeedValidationError(f"{field} must be timezone-aware")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if value != canonical:
        raise QualificationCacheSeedValidationError(f"{field} is not canonical")
    return canonical


def _safe_key(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationCacheSeedValidationError(f"{field} is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise QualificationCacheSeedValidationError(f"{field} is unsafe")
    return value


def _expected_job_name(run_id: str) -> str:
    return f"{run_id}-cache-seed"


def _submission_alias_key(run_id: str) -> str:
    return f"campaigns/{run_id}/monitor/QUALIFICATION_CACHE_SEED_SUBMITTED.json"


def _job_binding_key(
    run_id: str,
    *,
    managed_mode: str,
    submission_body_sha256: str,
) -> str:
    return (
        f"campaigns/{run_id}/monitor/must-start/{managed_mode}/"
        f"{submission_body_sha256}/JOB_BINDING.json"
    )


def _expected_ready_key(run_id: str) -> str:
    return (
        f"campaigns/{run_id}/qualification-cache-seed/"
        "QUALIFICATION_CACHE_SEED_READY.json"
    )


def validate_qualification_cache_seed_accepted(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact canonical acceptance record for downstream binding."""

    if not isinstance(value, Mapping):
        raise QualificationCacheSeedValidationError(
            "cache-seed acceptance must be an object"
        )
    _require_exact_fields(value, _ACCEPTED_FIELDS, label="cache-seed acceptance")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != ACCEPTED_SCHEMA_VERSION
        or value.get("record_type") != ACCEPTED_RECORD_TYPE
    ):
        raise QualificationCacheSeedValidationError(
            "cache-seed acceptance schema mismatch"
        )
    body = dict(value)
    digest = body.pop(ACCEPTED_DIGEST_FIELD)
    if digest != _canonical_sha256(body):
        raise QualificationCacheSeedValidationError(
            "cache-seed acceptance body SHA-256 mismatch"
        )
    accepted_at = _canonical_time(value.get("accepted_at"), field="accepted_at")

    authority = value.get("seed_authority")
    if not isinstance(authority, Mapping):
        raise QualificationCacheSeedValidationError("seed authority must be an object")
    _require_exact_fields(authority, _SEED_AUTHORITY_FIELDS, label="seed authority")
    if (
        authority.get("account_id") != APPROVED_ACCOUNT_ID
        or authority.get("region") != APPROVED_REGION
        or authority.get("managed_mode") != EXPECTED_MANAGED_MODE
        or not isinstance(authority.get("run_id"), str)
        or _RUN_ID.fullmatch(str(authority["run_id"])) is None
        or authority.get("sky_job_name") != _expected_job_name(str(authority["run_id"]))
    ):
        raise QualificationCacheSeedValidationError("seed authority identity mismatch")
    if (
        not isinstance(authority.get("bucket"), str)
        or _BUCKET.fullmatch(str(authority["bucket"])) is None
    ):
        raise QualificationCacheSeedValidationError("seed authority bucket invalid")
    _require_positive_int(authority.get("target_job_id"), field="target_job_id")
    for field in (
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "repo_tar_sha256",
        "approval_sha256",
        "submission_file_sha256",
        "submission_body_sha256",
        "job_binding_file_sha256",
        "job_binding_body_sha256",
    ):
        _require_sha(authority.get(field), field=field)
    run_id = str(authority["run_id"])
    for field in (
        "descriptor_key",
        "repo_tar_key",
        "approval_key",
        "submission_key",
        "submission_alias_key",
        "job_binding_key",
        "seed_ready_key",
    ):
        _safe_key(authority.get(field), field=field)
    if authority.get("submission_alias_key") != _submission_alias_key(run_id):
        raise QualificationCacheSeedValidationError(
            "seed authority submission alias key mismatch"
        )
    expected_submission_key = (
        f"campaigns/{run_id}/monitor/submission-locks/"
        f"{authority['descriptor_file_sha256']}-{EXPECTED_MANAGED_MODE}.json"
    )
    if authority.get("submission_key") != expected_submission_key:
        raise QualificationCacheSeedValidationError(
            "seed authority immutable submission key mismatch"
        )
    expected_job_binding_key = _job_binding_key(
        run_id,
        managed_mode=EXPECTED_MANAGED_MODE,
        submission_body_sha256=str(authority["submission_body_sha256"]),
    )
    if authority.get("job_binding_key") != expected_job_binding_key:
        raise QualificationCacheSeedValidationError(
            "seed authority JOB_BINDING.json key mismatch"
        )
    job_binding_value = authority.get("job_binding")
    if not isinstance(job_binding_value, Mapping):
        raise QualificationCacheSeedValidationError(
            "seed authority JOB_BINDING.json must be an object"
        )
    try:
        job_binding = validate_must_start_job_binding(job_binding_value)
    except ValueError as error:
        raise QualificationCacheSeedValidationError(str(error)) from error
    if (
        authority.get("job_binding_file_sha256")
        != _sha256(_canonical_bytes(job_binding))
        or authority.get("job_binding_body_sha256")
        != job_binding["job_binding_body_sha256"]
    ):
        raise QualificationCacheSeedValidationError(
            "seed authority JOB_BINDING.json file/body SHA-256 mismatch"
        )
    expected_job_binding_authority = {
        "run_id": run_id,
        "managed_mode": EXPECTED_MANAGED_MODE,
        "account_id": authority["account_id"],
        "region": authority["region"],
        "bucket": authority["bucket"],
        "descriptor_body_sha256": authority["descriptor_body_sha256"],
        "submission_body_sha256": authority["submission_body_sha256"],
        "sky_job_name": authority["sky_job_name"],
        "descriptor_key": authority["descriptor_key"],
        "descriptor_file_sha256": authority["descriptor_file_sha256"],
        "submission_key": authority["submission_key"],
        "target_job_id": authority["target_job_id"],
    }
    for field, expected in expected_job_binding_authority.items():
        if job_binding.get(field) != expected:
            raise QualificationCacheSeedValidationError(
                f"seed authority JOB_BINDING.json {field} mismatch"
            )
    if not str(authority["descriptor_key"]).startswith(
        f"campaigns/{run_id}/submissions/"
    ):
        raise QualificationCacheSeedValidationError(
            "seed authority descriptor key is outside the run submission prefix"
        )
    if not str(authority["repo_tar_key"]).startswith(f"campaigns/{run_id}/repository/"):
        raise QualificationCacheSeedValidationError(
            "seed authority repository key is outside the run prefix"
        )
    if not str(authority["approval_key"]).startswith(
        f"campaigns/{run_id}/authorities/"
    ):
        raise QualificationCacheSeedValidationError(
            "seed authority approval key is outside the run prefix"
        )
    if authority.get("seed_ready_key") != _expected_ready_key(run_id):
        raise QualificationCacheSeedValidationError("seed authority ready key mismatch")

    inventory = value.get("cache_object_inventory")
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise QualificationCacheSeedValidationError(
            "cache object inventory must contain exactly four objects"
        )
    normalized_inventory: list[dict[str, object]] = []
    for item in inventory:
        if not isinstance(item, Mapping):
            raise QualificationCacheSeedValidationError(
                "cache inventory item must be an object"
            )
        _require_exact_fields(item, _INVENTORY_FIELDS, label="cache inventory item")
        key = _safe_key(item.get("key"), field="cache inventory key")
        size = _require_positive_int(item.get("size"), field="cache object size")
        sha = _require_sha(item.get("sha256"), field="cache object SHA-256")
        if item.get("checksum_type") != "FULL_OBJECT":
            raise QualificationCacheSeedValidationError(
                "cache object lacks FULL_OBJECT checksum type"
            )
        etag = item.get("etag")
        if not isinstance(etag, str) or not etag or "-" in etag.strip('"'):
            raise QualificationCacheSeedValidationError(
                "cache object ETag is multipart or invalid"
            )
        version_id = item.get("version_id")
        if version_id is not None and (
            not isinstance(version_id, str) or not version_id
        ):
            raise QualificationCacheSeedValidationError(
                "cache object version_id is invalid"
            )
        normalized_inventory.append(
            {
                "key": key,
                "size": size,
                "sha256": sha,
                "checksum_type": "FULL_OBJECT",
                "etag": etag,
                "version_id": version_id,
            }
        )
    if (
        normalized_inventory
        != sorted(normalized_inventory, key=lambda item: str(item["key"]))
        or len({str(item["key"]) for item in normalized_inventory}) != 4
    ):
        raise QualificationCacheSeedValidationError(
            "cache object inventory must be unique and key-sorted"
        )
    if value.get("cache_object_inventory_sha256") != _canonical_sha256(
        normalized_inventory
    ):
        raise QualificationCacheSeedValidationError(
            "cache object inventory SHA-256 mismatch"
        )

    cache = value.get("cache_audit")
    if not isinstance(cache, Mapping):
        raise QualificationCacheSeedValidationError("cache audit must be an object")
    _require_exact_fields(cache, _CACHE_AUDIT_FIELDS, label="cache audit")
    for field in (
        "legacy_seed_ready_file_sha256",
        "legacy_seed_ready_body_sha256",
        "manifest_file_sha256",
        "manifest_body_sha256",
        "teacher_cache_ready_file_sha256",
        "teacher_cache_ready_body_sha256",
        "prompt_pack_file_sha256",
        "teich_pack_file_sha256",
        "frozen_prompt_pack_file_sha256",
    ):
        _require_sha(cache.get(field), field=field)
    _safe_key(cache.get("teich_pack_key"), field="teich_pack_key")
    _safe_key(cache.get("frozen_prompt_pack_key"), field="frozen_prompt_pack_key")
    if (
        cache.get("prompt_id") != EXPECTED_PROMPT_ID
        or cache.get("session_count") != EXPECTED_SESSION_COUNT
        or type(cache.get("supervised_position_count")) is not int
        or int(cache["supervised_position_count"]) <= 0
        or cache.get("top_k") != EXPECTED_TOP_K
        or cache.get("hidden_size") != EXPECTED_HIDDEN_SIZE
        or cache.get("semantic_audit_pass") is not True
    ):
        raise QualificationCacheSeedValidationError(
            "cache audit prompt/session/top_k/hidden_size authority mismatch"
        )
    expected_prefix = (
        f"qualification-cache/seeds/{run_id}/{cache['manifest_file_sha256']}/"
    )
    if cache.get("cache_prefix") != expected_prefix:
        raise QualificationCacheSeedValidationError(
            "cache audit prefix is not content-addressed"
        )
    expected_cache_keys = {
        f"{expected_prefix}{MANIFEST_FILENAME}",
        f"{expected_prefix}{READY_FILENAME}",
        f"{expected_prefix}prompt-pack.json",
    }
    if (
        cache.get("manifest_key") != f"{expected_prefix}{MANIFEST_FILENAME}"
        or cache.get("teacher_cache_ready_key") != f"{expected_prefix}{READY_FILENAME}"
        or cache.get("prompt_pack_key") != f"{expected_prefix}prompt-pack.json"
    ):
        raise QualificationCacheSeedValidationError(
            "cache audit fixed object keys mismatch"
        )
    inventory_keys = {str(item["key"]) for item in normalized_inventory}
    shard_keys = inventory_keys - expected_cache_keys
    if (
        len(shard_keys) != 1
        or not next(iter(shard_keys)).startswith(f"{expected_prefix}teacher_signal/")
        or not next(iter(shard_keys)).endswith(".safetensors")
        or not expected_cache_keys.issubset(inventory_keys)
    ):
        raise QualificationCacheSeedValidationError(
            "cache audit exact one-row inventory mismatch"
        )
    inventory_by_key = {str(item["key"]): item for item in normalized_inventory}
    fixed_hashes = {
        str(cache["manifest_key"]): cache["manifest_file_sha256"],
        str(cache["teacher_cache_ready_key"]): cache["teacher_cache_ready_file_sha256"],
        str(cache["prompt_pack_key"]): cache["prompt_pack_file_sha256"],
    }
    if any(
        inventory_by_key[key]["sha256"] != expected_sha
        for key, expected_sha in fixed_hashes.items()
    ):
        raise QualificationCacheSeedValidationError(
            "cache audit object identities do not match inventory"
        )

    spend = value.get("spend_closure")
    if not isinstance(spend, Mapping):
        raise QualificationCacheSeedValidationError("spend closure must be an object")
    _require_exact_fields(spend, _SPEND_CLOSURE_FIELDS, label="spend closure")
    for field in (
        "allocation_file_sha256",
        "allocation_body_sha256",
        "allocation_start_record_sha256",
        "spend_authority_sha256",
        "ledger_file_sha256",
        "ledger_tip_record_sha256",
        "spend_status_file_sha256",
        "job_status_file_sha256",
        "job_status_body_sha256",
    ):
        _require_sha(spend.get(field), field=field)
    for field in (
        "allocation_key",
        "ledger_key",
        "spend_status_key",
        "job_status_key",
    ):
        _safe_key(spend.get(field), field=field)
    expected_runtime = f"campaigns/{run_id}/runtime"
    expected_spend_keys = {
        "allocation_key": f"{expected_runtime}/GPU_RUNTIME_ALLOCATION.json",
        "ledger_key": f"{expected_runtime}/GPU_SPEND_LEDGER.jsonl",
        "spend_status_key": f"{expected_runtime}/GPU_SPEND_STATUS.json",
        "job_status_key": f"campaigns/{run_id}/monitor/SKY_JOB_STATUS.json",
    }
    if any(spend.get(field) != key for field, key in expected_spend_keys.items()):
        raise QualificationCacheSeedValidationError(
            "spend closure object keys mismatch"
        )
    if (
        spend.get("ledger_tip_event") != "allocation_ended"
        or spend.get("job_status") != "SUCCEEDED"
        or spend.get("sky_job_name") != authority.get("sky_job_name")
        or not isinstance(spend.get("instance_id"), str)
        or _INSTANCE_ID.fullmatch(str(spend["instance_id"])) is None
    ):
        raise QualificationCacheSeedValidationError(
            "spend closure instance/job/tip identity mismatch"
        )
    submitted = _canonical_time(
        spend.get("submission_submitted_at"),
        field="submission_submitted_at",
    )
    launched = _canonical_time(spend.get("launched_at"), field="launched_at")
    must_start = _canonical_time(spend.get("must_start_by"), field="must_start_by")
    ended = _canonical_time(spend.get("ended_at"), field="ended_at")
    job_observed = _canonical_time(
        spend.get("job_observed_at"),
        field="job_observed_at",
    )
    if (
        submitted != job_binding["submission_submitted_at"]
        or must_start != job_binding["must_start_by"]
    ):
        raise QualificationCacheSeedValidationError(
            "cache-seed timestamp chain does not match JOB_BINDING.json"
        )
    timestamps = [
        datetime.fromisoformat(item.replace("Z", "+00:00"))
        for item in (
            submitted,
            launched,
            must_start,
            ended,
            job_observed,
            accepted_at,
        )
    ]
    (
        submitted_at_value,
        launched_at_value,
        must_start_by_value,
        ended_at_value,
        job_observed_at_value,
        accepted_at_value,
    ) = timestamps
    if not (
        submitted_at_value <= launched_at_value <= must_start_by_value
        and launched_at_value
        <= ended_at_value
        <= job_observed_at_value
        <= accepted_at_value
    ):
        raise QualificationCacheSeedValidationError(
            "cache-seed timestamp chain is not submitted <= launched <= "
            "must_start_by and launched <= ended <= job_observed <= accepted"
        )
    _require_nonnegative_int(
        spend.get("consumed_gpu_seconds"), field="consumed_gpu_seconds"
    )
    _require_nonnegative_int(
        spend.get("remaining_gpu_seconds"), field="remaining_gpu_seconds"
    )
    cost = spend.get("estimated_gpu_cost_usd")
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise QualificationCacheSeedValidationError("estimated_gpu_cost_usd is invalid")
    return dict(value)


__all__ = [
    "ACCEPTED_DIGEST_FIELD",
    "ACCEPTED_RECORD_TYPE",
    "ACCEPTED_SCHEMA_VERSION",
    "EXPECTED_HIDDEN_SIZE",
    "EXPECTED_MANAGED_MODE",
    "EXPECTED_PROMPT_ID",
    "EXPECTED_SESSION_COUNT",
    "EXPECTED_TOP_K",
    "QualificationCacheSeedValidationError",
    "validate_qualification_cache_seed_accepted",
]
