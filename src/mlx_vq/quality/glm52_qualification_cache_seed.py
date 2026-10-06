"""Strict compatibility acceptance for the immutable qualification-cache seed.

The current cache-seed worker publishes a weak v1 readiness marker.  Its
self-hash is useful for corruption detection but is not an authority.  This
module accepts that output only when a caller supplies every immutable seed
identity independently and the S3/runtime/cache evidence closes around those
pins.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from mlx_vq.quality.glm52_campaign_watchdog import (
    validate_skypilot_job_status,
    validate_skypilot_submission_marker,
)
from mlx_vq.quality.glm52_h100_qualification import (
    validate_h100_runtime_allocation,
)
from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_ACCOUNT_ID,
    APPROVED_REGION,
    GpuSpendLedger,
    require_approved_aws_identity,
    validate_gpu_spend_approval,
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import validate_must_start_job_binding
from mlx_vq.quality.glm52_teich_training_cache import (
    MANIFEST_FILENAME,
    READY_FILENAME,
    ROUTER_LAYERS,
    SPLIT_SEED,
    audit_glm52_teich_teacher_cache,
)

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
_MISSING_CODES = {"404", "NoSuchKey", "NotFound"}

_SEED_READY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "run_id",
        "qualification_cache_prefix",
        "qualification_cache_manifest_sha256",
        "teacher_cache_ready_sha256",
        "ready_body_sha256",
    }
)
_SPEND_STATUS_FIELDS = frozenset(
    {
        "record_type",
        "run_id",
        "instance_id",
        "ended_at",
        "gpu_spend_record_sha256",
        "gpu_spend_ledger_sha256",
        "consumed_gpu_seconds",
        "remaining_gpu_seconds",
        "estimated_gpu_cost_usd",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "release_eligible",
        "personal_training_data",
        "prompt_pack_sha256",
        "frozen_66_prompt_pack_sha256",
        "session_count",
        "supervised_position_count",
        "top_k",
        "hidden_size",
        "router_layers",
        "cka_probe_layers",
        "split_policy",
        "frozen_66_overlap_count",
        "shards",
        "manifest_body_sha256",
    }
)
_SPLIT_POLICY_FIELDS = frozenset(
    {
        "record_type",
        "seed",
        "target_fractions",
        "weight",
        "split_supervised_positions",
    }
)
_SHARD_FIELDS = frozenset(
    {
        "row_index",
        "prompt_id",
        "source_session_id",
        "provider",
        "split",
        "tuning_eligible",
        "sparse_positions",
        "supervised_position_count",
        "token_ids_sha256",
        "relative_path",
        "file_sha256",
        "tensors",
    }
)
_TENSOR_FIELDS = frozenset({"shape", "dtype", "sha256"})
_READY_FIELDS = frozenset(
    {
        "record_type",
        "manifest_filename",
        "manifest_sha256",
        "manifest_body_sha256",
        "shard_inventory_sha256",
        "session_count",
        "supervised_position_count",
        "ready_record_sha256",
    }
)
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
    """The current seed output is not authenticated against external pins."""


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


def accepted_s3_key(*, run_id: str, acceptance_body_sha256: str) -> str:
    """Return the future immutable publication key for an accepted record."""

    if _RUN_ID.fullmatch(run_id) is None:
        raise QualificationCacheSeedValidationError("run_id is invalid")
    _require_sha(acceptance_body_sha256, field="acceptance_body_sha256")
    return (
        f"campaigns/{run_id}/qualification-cache-seed/accepted/"
        f"{acceptance_body_sha256}/QUALIFICATION_CACHE_SEED_ACCEPTED.json"
    )


@dataclass(frozen=True)
class QualificationCacheSeedPins:
    """Externally supplied immutable authority for the current Job 3 output."""

    account_id: str
    region: str
    bucket: str
    run_id: str
    managed_mode: str
    target_job_id: int
    descriptor_key: str
    descriptor_file_sha256: str
    descriptor_body_sha256: str
    campaign_identity_sha256: str
    repo_tar_sha256: str
    approval_sha256: str
    submission_key: str
    submission_file_sha256: str
    submission_body_sha256: str
    ready_key: str

    def validate(self) -> QualificationCacheSeedPins:
        if self.account_id != APPROVED_ACCOUNT_ID:
            raise QualificationCacheSeedValidationError(
                "seed account does not match approved authority"
            )
        if self.region != APPROVED_REGION:
            raise QualificationCacheSeedValidationError(
                "seed region does not match approved authority"
            )
        if _BUCKET.fullmatch(self.bucket) is None:
            raise QualificationCacheSeedValidationError("seed bucket is invalid")
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise QualificationCacheSeedValidationError("seed run_id is invalid")
        if self.managed_mode != EXPECTED_MANAGED_MODE:
            raise QualificationCacheSeedValidationError(
                "seed managed mode must be cache-seed"
            )
        _require_positive_int(self.target_job_id, field="target_job_id")
        for field in (
            "descriptor_file_sha256",
            "descriptor_body_sha256",
            "campaign_identity_sha256",
            "repo_tar_sha256",
            "approval_sha256",
            "submission_file_sha256",
            "submission_body_sha256",
        ):
            _require_sha(getattr(self, field), field=field)
        descriptor_key = _safe_key(self.descriptor_key, field="seed descriptor key")
        descriptor_prefix = f"campaigns/{self.run_id}/submissions/"
        if not descriptor_key.startswith(descriptor_prefix):
            raise QualificationCacheSeedValidationError(
                "seed descriptor key is outside the immutable submission prefix"
            )
        submission_key = _safe_key(self.submission_key, field="submission key")
        expected_submission = (
            f"campaigns/{self.run_id}/monitor/submission-locks/"
            f"{self.descriptor_file_sha256}-{self.managed_mode}.json"
        )
        if submission_key != expected_submission:
            raise QualificationCacheSeedValidationError(
                "submission key does not match descriptor file and mode authority"
            )
        if self.ready_key != _expected_ready_key(self.run_id):
            raise QualificationCacheSeedValidationError(
                "seed ready key is not the exact current compatibility marker key"
            )
        return self


def validate_approved_identity(identity: Mapping[str, object]) -> dict[str, object]:
    """Public account guard used by the CLI before it creates an S3 client."""

    try:
        return require_approved_aws_identity(identity)
    except ValueError as error:
        raise QualificationCacheSeedValidationError(str(error)) from error


def _duplicate_key_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationCacheSeedValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise QualificationCacheSeedValidationError(f"non-finite JSON constant {value}")


def _load_json_object(
    raw: bytes,
    *,
    label: str,
    canonical_file: bool = True,
) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise QualificationCacheSeedValidationError(f"{label} is not UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_key_rejector,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, QualificationCacheSeedValidationError) as error:
        if isinstance(error, QualificationCacheSeedValidationError):
            raise
        raise QualificationCacheSeedValidationError(
            f"{label} is malformed JSON"
        ) from error
    if not isinstance(value, dict):
        raise QualificationCacheSeedValidationError(f"{label} must be a JSON object")
    if canonical_file and raw != _canonical_bytes(value) + b"\n":
        raise QualificationCacheSeedValidationError(
            f"{label} is not canonical JSON with one trailing newline"
        )
    return value


def _error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return str(code) if code is not None else None


def _s3_read(label: str, operation):
    try:
        return operation()
    except Exception as error:
        code = _error_code(error)
        if code in _MISSING_CODES:
            raise QualificationCacheSeedValidationError(
                f"required S3 object is missing: {label}"
            ) from error
        suffix = code or type(error).__name__
        raise QualificationCacheSeedValidationError(
            f"S3 transport error reading {label}: {suffix}"
        ) from error


def _get_bytes(s3_client, *, bucket: str, key: str, label: str) -> bytes:
    response = _s3_read(
        f"s3://{bucket}/{key}",
        lambda: s3_client.get_object(Bucket=bucket, Key=key),
    )
    if not isinstance(response, Mapping):
        raise QualificationCacheSeedValidationError(
            f"malformed S3 GET response for {label}"
        )
    body = response.get("Body")
    if body is None or not hasattr(body, "read"):
        raise QualificationCacheSeedValidationError(
            f"malformed S3 GET body for {label}"
        )
    raw = body.read()
    if not isinstance(raw, bytes):
        raise QualificationCacheSeedValidationError(
            f"malformed S3 GET bytes for {label}"
        )
    length = response.get("ContentLength")
    if length is not None and (type(length) is not int or length != len(raw)):
        raise QualificationCacheSeedValidationError(
            f"S3 GET length mismatch for {label}"
        )
    return raw


def _decode_full_sha(response: Mapping[str, object], *, label: str) -> str:
    if response.get("ChecksumType") != "FULL_OBJECT":
        raise QualificationCacheSeedValidationError(
            f"{label} lacks FULL_OBJECT SHA-256 authority"
        )
    checksum = response.get("ChecksumSHA256")
    if not isinstance(checksum, str):
        raise QualificationCacheSeedValidationError(
            f"{label} lacks FULL_OBJECT SHA-256 authority"
        )
    try:
        decoded = base64.b64decode(checksum, validate=True)
    except (binascii.Error, ValueError) as error:
        raise QualificationCacheSeedValidationError(
            f"{label} has malformed FULL_OBJECT SHA-256 authority"
        ) from error
    if len(decoded) != 32:
        raise QualificationCacheSeedValidationError(
            f"{label} has malformed FULL_OBJECT SHA-256 authority"
        )
    return decoded.hex()


def _head_full_object(
    s3_client,
    *,
    bucket: str,
    key: str,
    label: str,
    expected_run_id: str | None = None,
) -> dict[str, object]:
    response = _s3_read(
        f"s3://{bucket}/{key}",
        lambda: s3_client.head_object(
            Bucket=bucket,
            Key=key,
            ChecksumMode="ENABLED",
        ),
    )
    if not isinstance(response, Mapping):
        raise QualificationCacheSeedValidationError(
            f"malformed S3 HEAD response for {label}"
        )
    size = response.get("ContentLength")
    if type(size) is not int or size <= 0:
        raise QualificationCacheSeedValidationError(f"{label} S3 size is invalid")
    etag = response.get("ETag")
    if not isinstance(etag, str) or not etag:
        raise QualificationCacheSeedValidationError(f"{label} ETag is invalid")
    if "-" in etag.strip('"'):
        raise QualificationCacheSeedValidationError(f"{label} is a multipart object")
    sha = _decode_full_sha(response, label=label)
    version_id = response.get("VersionId")
    if version_id is not None and (not isinstance(version_id, str) or not version_id):
        raise QualificationCacheSeedValidationError(f"{label} VersionId is invalid")
    metadata = response.get("Metadata")
    if expected_run_id is not None and (
        not isinstance(metadata, Mapping)
        or dict(metadata) != {"glm52-run-id": expected_run_id}
    ):
        raise QualificationCacheSeedValidationError(
            f"{label} has foreign-run, missing, or non-exact S3 metadata; "
            "exact S3 metadata is required"
        )
    return {
        "key": key,
        "size": size,
        "sha256": sha,
        "checksum_type": "FULL_OBJECT",
        "etag": etag,
        "version_id": version_id,
    }


def _get_full_object(
    s3_client,
    *,
    bucket: str,
    key: str,
    label: str,
    expected_run_id: str | None = None,
) -> tuple[bytes, dict[str, object]]:
    inventory = _head_full_object(
        s3_client,
        bucket=bucket,
        key=key,
        label=label,
        expected_run_id=expected_run_id,
    )
    raw = _get_bytes(s3_client, bucket=bucket, key=key, label=label)
    if len(raw) != inventory["size"] or _sha256(raw) != inventory["sha256"]:
        raise QualificationCacheSeedValidationError(
            f"{label} changed between S3 HEAD and GET"
        )
    return raw, inventory


def _list_keys(s3_client, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    continuation: str | None = None
    while True:
        arguments: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if continuation is not None:
            arguments["ContinuationToken"] = continuation
        response = _s3_read(
            f"s3://{bucket}/{prefix}",
            lambda arguments=arguments: s3_client.list_objects_v2(**arguments),
        )
        if not isinstance(response, Mapping):
            raise QualificationCacheSeedValidationError(
                "malformed S3 object-list response"
            )
        contents = response.get("Contents", [])
        if not isinstance(contents, list):
            raise QualificationCacheSeedValidationError(
                "malformed S3 object-list contents"
            )
        for item in contents:
            if not isinstance(item, Mapping) or not isinstance(item.get("Key"), str):
                raise QualificationCacheSeedValidationError(
                    "malformed S3 object-list item"
                )
            keys.append(str(item["Key"]))
        truncated = response.get("IsTruncated")
        if truncated is not True:
            break
        next_token = response.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token:
            raise QualificationCacheSeedValidationError(
                "truncated S3 object list lacks a continuation token"
            )
        continuation = next_token
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise QualificationCacheSeedValidationError(
            "S3 cache object inventory is unsorted or duplicated"
        )
    return keys


def _require_no_multipart_uploads(
    s3_client,
    *,
    bucket: str,
    prefix: str,
) -> None:
    key_marker: str | None = None
    upload_marker: str | None = None
    uploads: list[str] = []
    while True:
        arguments: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if key_marker is not None:
            arguments["KeyMarker"] = key_marker
        if upload_marker is not None:
            arguments["UploadIdMarker"] = upload_marker
        response = _s3_read(
            f"s3://{bucket}/{prefix} multipart uploads",
            lambda arguments=arguments: s3_client.list_multipart_uploads(**arguments),
        )
        if not isinstance(response, Mapping):
            raise QualificationCacheSeedValidationError(
                "malformed S3 multipart-list response"
            )
        raw_uploads = response.get("Uploads", [])
        if not isinstance(raw_uploads, list):
            raise QualificationCacheSeedValidationError(
                "malformed S3 multipart-list contents"
            )
        for upload in raw_uploads:
            if not isinstance(upload, Mapping) or not isinstance(
                upload.get("Key"), str
            ):
                raise QualificationCacheSeedValidationError(
                    "malformed S3 multipart upload"
                )
            uploads.append(str(upload["Key"]))
        if response.get("IsTruncated") is not True:
            break
        key_marker = response.get("NextKeyMarker")  # type: ignore[assignment]
        upload_marker = response.get("NextUploadIdMarker")  # type: ignore[assignment]
        if not isinstance(key_marker, str) or not isinstance(upload_marker, str):
            raise QualificationCacheSeedValidationError(
                "truncated multipart list lacks continuation markers"
            )
    if uploads:
        raise QualificationCacheSeedValidationError(
            "unfinished multipart uploads exist under cache prefix: "
            + ", ".join(sorted(uploads))
        )


def _prepare_work_dir(path: Path) -> None:
    if path.is_symlink():
        raise QualificationCacheSeedValidationError(
            "acceptance work directory must not be a symlink"
        )
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise QualificationCacheSeedValidationError(
                "acceptance work directory must be an empty directory"
            )
    else:
        path.mkdir(parents=True)


def _write_work_file(root: Path, relative: str, raw: bytes) -> Path:
    logical = _safe_key(relative, field="temporary object path")
    path = root.joinpath(*PurePosixPath(logical).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _validate_seed_ready(
    value: Mapping[str, object],
    *,
    pins: QualificationCacheSeedPins,
) -> dict[str, object]:
    _require_exact_fields(value, _SEED_READY_FIELDS, label="seed ready")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("record_type") != "glm52_qualification_cache_seed_ready_v1"
    ):
        raise QualificationCacheSeedValidationError("seed ready schema mismatch")
    body = dict(value)
    digest = body.pop("ready_body_sha256")
    if digest != _canonical_sha256(body):
        raise QualificationCacheSeedValidationError("seed ready body SHA-256 mismatch")
    if value.get("run_id") != pins.run_id:
        raise QualificationCacheSeedValidationError("seed ready contains a foreign run")
    manifest_sha = _require_sha(
        value.get("qualification_cache_manifest_sha256"),
        field="qualification cache manifest SHA-256",
    )
    _require_sha(
        value.get("teacher_cache_ready_sha256"),
        field="teacher cache ready SHA-256",
    )
    expected_prefix = f"qualification-cache/seeds/{pins.run_id}/{manifest_sha}/"
    if value.get("qualification_cache_prefix") != expected_prefix:
        raise QualificationCacheSeedValidationError(
            "seed ready cache prefix is not bound to run and manifest"
        )
    return dict(value)


def _validate_manifest(
    value: Mapping[str, object],
    *,
    prompt_id: str,
    supervised_positions: int,
) -> dict[str, object]:
    _require_exact_fields(value, _MANIFEST_FIELDS, label="cache manifest")
    body = dict(value)
    digest = body.pop("manifest_body_sha256")
    if digest != _canonical_sha256(body):
        raise QualificationCacheSeedValidationError(
            "cache manifest body SHA-256 mismatch"
        )
    expected_scalars: dict[str, object] = {
        "schema_version": 3,
        "record_type": "glm52_teacher_signal_cache",
        "release_eligible": False,
        "personal_training_data": True,
        "session_count": EXPECTED_SESSION_COUNT,
        "supervised_position_count": supervised_positions,
        "top_k": EXPECTED_TOP_K,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "router_layers": list(ROUTER_LAYERS),
        "cka_probe_layers": [77],
        "frozen_66_overlap_count": 0,
    }
    if any(
        value.get(field) != expected for field, expected in expected_scalars.items()
    ):
        raise QualificationCacheSeedValidationError(
            "cache manifest one-row schema-v3 authority mismatch"
        )
    for field in (
        "prompt_pack_sha256",
        "frozen_66_prompt_pack_sha256",
        "manifest_body_sha256",
    ):
        _require_sha(value.get(field), field=f"cache manifest {field}")
    split = value.get("split_policy")
    if not isinstance(split, Mapping):
        raise QualificationCacheSeedValidationError(
            "cache manifest split policy must be an object"
        )
    _require_exact_fields(split, _SPLIT_POLICY_FIELDS, label="cache split policy")
    expected_split = {
        "record_type": "provider_balanced_session_disjoint_weighted_v1",
        "seed": SPLIT_SEED,
        "target_fractions": {
            "train": 0.8,
            "validation": 0.1,
            "holdout": 0.1,
        },
        "weight": "supervised_positions",
    }
    if any(split.get(field) != expected for field, expected in expected_split.items()):
        raise QualificationCacheSeedValidationError(
            "cache manifest split policy authority mismatch"
        )
    split_counts = split.get("split_supervised_positions")
    if (
        not isinstance(split_counts, Mapping)
        or set(split_counts) != {"train", "validation", "holdout"}
        or any(type(count) is not int or count < 0 for count in split_counts.values())
        or sum(split_counts.values()) != supervised_positions
    ):
        raise QualificationCacheSeedValidationError(
            "cache manifest split accounting mismatch"
        )
    shards = value.get("shards")
    if not isinstance(shards, list) or len(shards) != 1:
        raise QualificationCacheSeedValidationError(
            "cache manifest must contain exactly one row shard"
        )
    shard = shards[0]
    if not isinstance(shard, Mapping):
        raise QualificationCacheSeedValidationError(
            "cache manifest shard must be an object"
        )
    _require_exact_fields(shard, _SHARD_FIELDS, label="cache manifest shard")
    if (
        shard.get("row_index") != 0
        or shard.get("prompt_id") != prompt_id
        or shard.get("supervised_position_count") != supervised_positions
        or shard.get("sparse_positions") is not True
        or not isinstance(shard.get("source_session_id"), str)
        or not shard.get("source_session_id")
        or not isinstance(shard.get("provider"), str)
        or not shard.get("provider")
        or shard.get("split") not in {"train", "validation", "holdout"}
        or shard.get("tuning_eligible") is not (shard.get("split") == "train")
    ):
        raise QualificationCacheSeedValidationError(
            "cache manifest shard identity mismatch"
        )
    relative = _safe_key(shard.get("relative_path"), field="cache shard path")
    if not relative.startswith("teacher_signal/") or not relative.endswith(
        ".safetensors"
    ):
        raise QualificationCacheSeedValidationError(
            "cache manifest shard path is invalid"
        )
    for field in ("token_ids_sha256", "file_sha256"):
        _require_sha(shard.get(field), field=f"cache shard {field}")
    tensors = shard.get("tensors")
    expected_tensors = {
        "positions",
        "target_token_ids",
        "topk_logit_ids",
        "topk_logit_values",
        "logsumexp",
        "tail_mass",
        "layer_77_hidden_probe",
        "router_top8_expert_ids",
        "router_top8_normalized_weights",
    }
    if not isinstance(tensors, Mapping) or set(tensors) != expected_tensors:
        raise QualificationCacheSeedValidationError(
            "cache manifest tensor inventory mismatch"
        )
    for name, metadata in tensors.items():
        if not isinstance(metadata, Mapping):
            raise QualificationCacheSeedValidationError(
                f"cache tensor {name} metadata must be an object"
            )
        _require_exact_fields(metadata, _TENSOR_FIELDS, label=f"cache tensor {name}")
        shape = metadata.get("shape")
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or not isinstance(metadata.get("dtype"), str)
        ):
            raise QualificationCacheSeedValidationError(
                f"cache tensor {name} metadata is invalid"
            )
        _require_sha(metadata.get("sha256"), field=f"cache tensor {name} SHA-256")
    return dict(value)


def _validate_ready(
    value: Mapping[str, object],
    *,
    manifest_sha256: str,
    manifest_body_sha256: str,
    supervised_positions: int,
) -> dict[str, object]:
    _require_exact_fields(value, _READY_FIELDS, label="teacher cache ready")
    body = dict(value)
    digest = body.pop("ready_record_sha256")
    if digest != _canonical_sha256(body):
        raise QualificationCacheSeedValidationError(
            "teacher cache ready body SHA-256 mismatch"
        )
    expected = {
        "record_type": "glm52_teacher_cache_ready_v3",
        "manifest_filename": MANIFEST_FILENAME,
        "manifest_sha256": manifest_sha256,
        "manifest_body_sha256": manifest_body_sha256,
        "session_count": EXPECTED_SESSION_COUNT,
        "supervised_position_count": supervised_positions,
    }
    if any(value.get(field) != wanted for field, wanted in expected.items()):
        raise QualificationCacheSeedValidationError(
            "teacher cache ready authority mismatch"
        )
    _require_sha(
        value.get("shard_inventory_sha256"),
        field="teacher cache shard inventory SHA-256",
    )
    return dict(value)


def _validate_prompt_pack(
    value: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    rows = value.get("prompt_rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SESSION_COUNT:
        raise QualificationCacheSeedValidationError(
            "qualification prompt pack must contain exactly one row"
        )
    row = rows[0]
    if not isinstance(row, Mapping) or row.get("prompt_id") != EXPECTED_PROMPT_ID:
        raise QualificationCacheSeedValidationError(
            "qualification prompt pack contains a foreign prompt"
        )
    positions = row.get("positions")
    targets = row.get("target_token_ids")
    if (
        not isinstance(positions, list)
        or not positions
        or any(type(item) is not int or item < 0 for item in positions)
        or len(set(positions)) != len(positions)
        or not isinstance(targets, list)
        or len(targets) != len(positions)
        or any(type(item) is not int or item < 0 for item in targets)
    ):
        raise QualificationCacheSeedValidationError(
            "qualification prompt row sparse authority is invalid"
        )
    if value.get("prompt_row_count") != 1 or value.get("supervised_tokens") != len(
        positions
    ):
        raise QualificationCacheSeedValidationError(
            "qualification prompt pack one-row accounting mismatch"
        )
    return dict(value), len(positions)


def _expected_teich_prompt_subset(
    value: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    rows = value.get("prompt_rows")
    if not isinstance(rows, list):
        raise QualificationCacheSeedValidationError(
            "descriptor-pinned Teich prompt authority lacks prompt_rows"
        )
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("prompt_id") == EXPECTED_PROMPT_ID
    ]
    if len(selected) != 1:
        raise QualificationCacheSeedValidationError(
            "descriptor-pinned Teich prompt authority must contain "
            "the seed row exactly once"
        )
    positions = selected[0].get("positions")
    if not isinstance(positions, list):
        raise QualificationCacheSeedValidationError(
            "descriptor-pinned Teich prompt authority seed positions are invalid"
        )
    subset = {
        **value,
        "prompt_row_count": 1,
        "supervised_tokens": len(positions),
        "prompt_rows": selected,
    }
    normalized = dict(subset)
    return normalized, _canonical_bytes(normalized) + b"\n"


def _parse_spend_status(
    value: Mapping[str, object],
    *,
    run_id: str,
    instance_id: str,
    ledger: GpuSpendLedger,
    ledger_sha256: str,
) -> dict[str, object]:
    _require_exact_fields(value, _SPEND_STATUS_FIELDS, label="GPU spend status")
    if (
        value.get("record_type") != "glm52_gpu_spend_status_v1"
        or value.get("run_id") != run_id
        or value.get("instance_id") != instance_id
    ):
        raise QualificationCacheSeedValidationError(
            "GPU spend status contains a foreign allocation"
        )
    ended_at = _canonical_time(value.get("ended_at"), field="GPU spend ended_at")
    ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    tip = ledger.records[-1]
    if (
        tip.get("event") != "allocation_ended"
        or tip.get("instance_id") != instance_id
        or tip.get("timestamp") != ended_at
        or value.get("gpu_spend_record_sha256") != tip.get("record_sha256")
    ):
        raise QualificationCacheSeedValidationError(
            "GPU spend status does not authenticate the closed ledger tip"
        )
    if value.get("gpu_spend_ledger_sha256") != ledger_sha256:
        raise QualificationCacheSeedValidationError(
            "GPU spend status ledger SHA-256 mismatch"
        )
    expected_values: dict[str, object] = {
        "consumed_gpu_seconds": int(ledger.consumed_gpu_seconds(now=ended)),
        "remaining_gpu_seconds": int(ledger.remaining_gpu_seconds(now=ended)),
        "estimated_gpu_cost_usd": ledger.estimated_gpu_cost_usd(now=ended),
    }
    for field, expected in expected_values.items():
        actual = value.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise QualificationCacheSeedValidationError(
                f"GPU spend status {field} does not match authenticated ledger"
            )
    return dict(value)


def _validate_exact_ledger_records(raw: bytes) -> None:
    lines = raw.splitlines(keepends=True)
    if not lines or b"".join(lines) != raw:
        raise QualificationCacheSeedValidationError(
            "GPU spend ledger contains an incomplete record"
        )
    common = {
        "record_type",
        "run_id",
        "approval_sha256",
        "event",
        "instance_id",
        "timestamp",
        "prior_record_sha256",
        "record_sha256",
    }
    for index, line in enumerate(lines, start=1):
        value = _load_json_object(
            line,
            label=f"GPU spend ledger record {index}",
        )
        event = value.get("event")
        expected = common | ({"job_id"} if event == "allocation_started" else set())
        if (
            event not in {"allocation_started", "allocation_ended"}
            or set(value) != expected
        ):
            raise QualificationCacheSeedValidationError(
                f"GPU spend ledger record schema mismatch at line {index}"
            )


def _seed_authority_record(
    *,
    pins: QualificationCacheSeedPins,
    descriptor: Mapping[str, object],
    job_binding_key: str,
    job_binding: Mapping[str, object],
    job_binding_file_sha256: str,
    job_binding_body_sha256: str,
) -> dict[str, object]:
    return {
        "account_id": pins.account_id,
        "region": pins.region,
        "bucket": pins.bucket,
        "run_id": pins.run_id,
        "managed_mode": pins.managed_mode,
        "target_job_id": pins.target_job_id,
        "sky_job_name": _expected_job_name(pins.run_id),
        "descriptor_key": pins.descriptor_key,
        "descriptor_file_sha256": pins.descriptor_file_sha256,
        "descriptor_body_sha256": pins.descriptor_body_sha256,
        "campaign_identity_sha256": pins.campaign_identity_sha256,
        "repo_tar_key": descriptor["repo_tar_key"],
        "repo_tar_sha256": pins.repo_tar_sha256,
        "approval_key": descriptor["approval_key"],
        "approval_sha256": pins.approval_sha256,
        "submission_key": pins.submission_key,
        "submission_file_sha256": pins.submission_file_sha256,
        "submission_body_sha256": pins.submission_body_sha256,
        "submission_alias_key": _submission_alias_key(pins.run_id),
        "job_binding_key": job_binding_key,
        "job_binding": dict(job_binding),
        "job_binding_file_sha256": job_binding_file_sha256,
        "job_binding_body_sha256": job_binding_body_sha256,
        "seed_ready_key": pins.ready_key,
    }


def authenticate_qualification_cache_seed(
    *,
    s3_client,
    pins: QualificationCacheSeedPins,
    work_dir: str | Path,
    accepted_at: datetime,
) -> dict[str, object]:
    """Accept one current cache seed without writing to S3.

    ``accepted_at`` is supplied by the caller so tests and orchestration can
    bind the exact local acceptance event.  The work directory is scratch
    space only; the CLI creates it with ``TemporaryDirectory``.
    """

    pins.validate()
    if accepted_at.tzinfo is None:
        raise QualificationCacheSeedValidationError(
            "accepted_at must be timezone-aware"
        )
    accepted_at_value = (
        accepted_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    root = Path(work_dir)
    _prepare_work_dir(root)

    descriptor_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=pins.descriptor_key,
        label="seed descriptor",
    )
    if _sha256(descriptor_raw) != pins.descriptor_file_sha256:
        raise QualificationCacheSeedValidationError(
            "seed descriptor file SHA-256 does not match external authority"
        )
    descriptor_value = _load_json_object(descriptor_raw, label="seed descriptor")
    try:
        descriptor = validate_sky_campaign_descriptor(descriptor_value)
    except ValueError as error:
        raise QualificationCacheSeedValidationError(str(error)) from error
    expected_descriptor = {
        "run_id": pins.run_id,
        "account_id": pins.account_id,
        "region": pins.region,
        "bucket": pins.bucket,
        "campaign_descriptor_key": pins.descriptor_key,
        "descriptor_body_sha256": pins.descriptor_body_sha256,
        "campaign_identity_sha256": pins.campaign_identity_sha256,
        "repo_tar_sha256": pins.repo_tar_sha256,
        "approval_sha256": pins.approval_sha256,
    }
    mismatched_descriptor = [
        field
        for field, expected in expected_descriptor.items()
        if descriptor.get(field) != expected
    ]
    artifacts = descriptor.get("artifacts")
    if mismatched_descriptor:
        raise QualificationCacheSeedValidationError(
            "seed descriptor external identity mismatch: "
            + ", ".join(sorted(mismatched_descriptor))
        )
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get("qualification_cache_manifest_sha256") != "0" * 64
        or artifacts.get("qualification_cache_prefix") != "qualification-cache/"
    ):
        raise QualificationCacheSeedValidationError(
            "seed descriptor is not the externally pinned placeholder descriptor"
        )

    approval_key = str(descriptor["approval_key"])
    approval_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=approval_key,
        label="GPU spend approval",
    )
    if _sha256(approval_raw) != pins.approval_sha256:
        raise QualificationCacheSeedValidationError(
            "GPU spend approval file SHA-256 mismatch"
        )
    approval_value = _load_json_object(approval_raw, label="GPU spend approval")
    try:
        validate_gpu_spend_approval(approval_value)
    except ValueError as error:
        raise QualificationCacheSeedValidationError(str(error)) from error

    repo_key = str(descriptor["repo_tar_key"])
    repo_head = _head_full_object(
        s3_client,
        bucket=pins.bucket,
        key=repo_key,
        label="seed repository tar",
        expected_run_id=pins.run_id,
    )
    if repo_head["sha256"] != pins.repo_tar_sha256:
        raise QualificationCacheSeedValidationError(
            "seed repository tar FULL_OBJECT SHA-256 mismatch"
        )

    submission_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=pins.submission_key,
        label="immutable seed submission",
    )
    if _sha256(submission_raw) != pins.submission_file_sha256:
        raise QualificationCacheSeedValidationError(
            "immutable submission file SHA-256 mismatch"
        )
    submission_value = _load_json_object(
        submission_raw, label="immutable seed submission"
    )
    try:
        submission = validate_skypilot_submission_marker(submission_value)
    except ValueError as error:
        raise QualificationCacheSeedValidationError(str(error)) from error
    expected_submission = {
        "run_id": pins.run_id,
        "descriptor_body_sha256": pins.descriptor_body_sha256,
        "sky_job_name": _expected_job_name(pins.run_id),
        "submission_body_sha256": pins.submission_body_sha256,
        "must_start_by": descriptor["must_start_by"],
    }
    if any(
        submission.get(field) != expected
        for field, expected in expected_submission.items()
    ):
        raise QualificationCacheSeedValidationError(
            "immutable submission does not match external seed authority"
        )
    alias_key = _submission_alias_key(pins.run_id)
    alias_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=alias_key,
        label="mutable seed submission alias",
    )
    if alias_raw != submission_raw:
        raise QualificationCacheSeedValidationError(
            "mutable submission alias drifted from immutable seed submission"
        )

    job_binding_key = _job_binding_key(
        pins.run_id,
        managed_mode=pins.managed_mode,
        submission_body_sha256=pins.submission_body_sha256,
    )
    job_binding_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=job_binding_key,
        label="must-start JOB_BINDING.json",
    )
    job_binding_value = _load_json_object(
        job_binding_raw,
        label="must-start JOB_BINDING.json",
        canonical_file=False,
    )
    if job_binding_raw != _canonical_bytes(job_binding_value):
        raise QualificationCacheSeedValidationError(
            "must-start JOB_BINDING.json is not exact canonical JSON"
        )
    try:
        job_binding = validate_must_start_job_binding(job_binding_value)
    except ValueError as error:
        raise QualificationCacheSeedValidationError(str(error)) from error
    expected_job_binding = {
        "run_id": pins.run_id,
        "managed_mode": pins.managed_mode,
        "account_id": pins.account_id,
        "region": pins.region,
        "bucket": pins.bucket,
        "descriptor_body_sha256": pins.descriptor_body_sha256,
        "submission_body_sha256": pins.submission_body_sha256,
        "sky_job_name": _expected_job_name(pins.run_id),
        "must_start_by": descriptor["must_start_by"],
        "descriptor_key": pins.descriptor_key,
        "descriptor_file_sha256": pins.descriptor_file_sha256,
        "submission_key": pins.submission_key,
        "submission_submitted_at": submission["submitted_at"],
        "target_job_id": pins.target_job_id,
    }
    for field, expected in expected_job_binding.items():
        if job_binding.get(field) != expected:
            raise QualificationCacheSeedValidationError(
                f"must-start JOB_BINDING.json {field} authority mismatch"
            )

    seed_ready_raw, _seed_ready_head = _get_full_object(
        s3_client,
        bucket=pins.bucket,
        key=pins.ready_key,
        label="legacy qualification cache seed ready marker",
        expected_run_id=pins.run_id,
    )
    seed_ready_value = _load_json_object(
        seed_ready_raw,
        label="legacy qualification cache seed ready marker",
    )
    seed_ready = _validate_seed_ready(seed_ready_value, pins=pins)
    cache_prefix = str(seed_ready["qualification_cache_prefix"])

    runtime_prefix = f"campaigns/{pins.run_id}/runtime"
    allocation_key = f"{runtime_prefix}/GPU_RUNTIME_ALLOCATION.json"
    ledger_key = f"{runtime_prefix}/GPU_SPEND_LEDGER.jsonl"
    spend_status_key = f"{runtime_prefix}/GPU_SPEND_STATUS.json"
    job_status_key = f"campaigns/{pins.run_id}/monitor/SKY_JOB_STATUS.json"
    allocation_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=allocation_key,
        label="GPU runtime allocation",
    )
    allocation_value = _load_json_object(allocation_raw, label="GPU runtime allocation")
    ledger_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=ledger_key,
        label="GPU spend ledger",
    )
    if not ledger_raw.endswith(b"\n"):
        raise QualificationCacheSeedValidationError(
            "GPU spend ledger must end with one complete record"
        )
    _validate_exact_ledger_records(ledger_raw)
    ledger_path = _write_work_file(root, "runtime/GPU_SPEND_LEDGER.jsonl", ledger_raw)
    try:
        ledger = GpuSpendLedger(
            ledger_path,
            run_id=pins.run_id,
            approval_sha256=pins.approval_sha256,
            approved_gpu_runtime_seconds=int(
                descriptor["approved_gpu_runtime_seconds"]
            ),
            approved_gpu_cost_usd=float(descriptor["approved_gpu_cost_usd"]),
            hourly_cost_usd=float(descriptor["max_hourly_cost_usd"]),
        )
    except ValueError as error:
        raise QualificationCacheSeedValidationError(str(error)) from error
    if len(ledger.records) < 2 or ledger.records[-1].get("event") != "allocation_ended":
        raise QualificationCacheSeedValidationError(
            "GPU spend ledger tip is not a closed allocation"
        )
    allocation = validate_h100_runtime_allocation(
        allocation_value,
        expected_run_id=pins.run_id,
        expected_job_id=_expected_job_name(pins.run_id),
        expected_approval_sha256=pins.approval_sha256,
        expected_gpu_spend_authority_sha256=ledger.genesis_sha256,
    )
    instance_id = str(allocation["instance_id"])
    if _INSTANCE_ID.fullmatch(instance_id) is None:
        raise QualificationCacheSeedValidationError(
            "seed allocation instance ID is invalid"
        )
    if (
        ledger.records[-2].get("event") != "allocation_started"
        or ledger.records[-2].get("record_sha256")
        != allocation["gpu_spend_record_sha256"]
        or ledger.records[-2].get("job_id") != _expected_job_name(pins.run_id)
        or ledger.records[-2].get("instance_id") != instance_id
        or ledger.records[-1].get("instance_id") != instance_id
    ):
        raise QualificationCacheSeedValidationError(
            "GPU runtime allocation is not the allocation closed at the ledger tip"
        )

    spend_status_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=spend_status_key,
        label="GPU spend status",
    )
    spend_status_value = _load_json_object(spend_status_raw, label="GPU spend status")
    spend_status = _parse_spend_status(
        spend_status_value,
        run_id=pins.run_id,
        instance_id=instance_id,
        ledger=ledger,
        ledger_sha256=_sha256(ledger_raw),
    )
    job_status_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=job_status_key,
        label="SkyPilot job status",
    )
    job_status_value = _load_json_object(job_status_raw, label="SkyPilot job status")
    try:
        job_status = validate_skypilot_job_status(job_status_value)
    except ValueError as error:
        raise QualificationCacheSeedValidationError(str(error)) from error
    expected_job_status = {
        "run_id": pins.run_id,
        "descriptor_body_sha256": pins.descriptor_body_sha256,
        "sky_job_name": _expected_job_name(pins.run_id),
        "status": "SUCCEEDED",
        "instance_id": instance_id,
    }
    if any(
        job_status.get(field) != expected
        for field, expected in expected_job_status.items()
    ):
        raise QualificationCacheSeedValidationError(
            "SkyPilot job status does not prove exact seed success"
        )
    submission_submitted = datetime.fromisoformat(
        str(submission["submitted_at"]).replace("Z", "+00:00")
    )
    allocation_launched = datetime.fromisoformat(
        str(allocation["launched_at"]).replace("Z", "+00:00")
    )
    must_start_by = datetime.fromisoformat(
        str(descriptor["must_start_by"]).replace("Z", "+00:00")
    )
    spend_ended = datetime.fromisoformat(
        str(spend_status["ended_at"]).replace("Z", "+00:00")
    )
    job_observed = datetime.fromisoformat(
        str(job_status["observed_at"]).replace("Z", "+00:00")
    )
    acceptance_observed = datetime.fromisoformat(
        accepted_at_value.replace("Z", "+00:00")
    )
    if not (
        submission_submitted <= allocation_launched <= must_start_by
        and allocation_launched <= spend_ended <= job_observed <= acceptance_observed
    ):
        raise QualificationCacheSeedValidationError(
            "authenticated cache-seed timestamp chain is not "
            "submitted <= launched <= must_start_by and "
            "launched <= ended <= job_observed <= accepted"
        )

    _require_no_multipart_uploads(s3_client, bucket=pins.bucket, prefix=cache_prefix)
    listed_keys = _list_keys(s3_client, bucket=pins.bucket, prefix=cache_prefix)
    manifest_key = f"{cache_prefix}{MANIFEST_FILENAME}"
    teacher_ready_key = f"{cache_prefix}{READY_FILENAME}"
    prompt_pack_key = f"{cache_prefix}prompt-pack.json"
    shard_keys = [
        key
        for key in listed_keys
        if key.startswith(f"{cache_prefix}teacher_signal/")
        and key.endswith(".safetensors")
    ]
    if len(shard_keys) != 1 or set(listed_keys) != {
        manifest_key,
        teacher_ready_key,
        prompt_pack_key,
        shard_keys[0],
    }:
        raise QualificationCacheSeedValidationError(
            "exact cache object inventory requires one manifest, one ready "
            "marker, one prompt pack, and one row shard"
        )

    cache_raw: dict[str, bytes] = {}
    inventory: list[dict[str, object]] = []
    for key in listed_keys:
        raw, item = _get_full_object(
            s3_client,
            bucket=pins.bucket,
            key=key,
            label=f"qualification cache object {key}",
            expected_run_id=pins.run_id,
        )
        cache_raw[key] = raw
        inventory.append(item)
        _write_work_file(root, key.removeprefix(cache_prefix), raw)
    if (
        _sha256(cache_raw[manifest_key])
        != seed_ready["qualification_cache_manifest_sha256"]
        or _sha256(cache_raw[teacher_ready_key])
        != seed_ready["teacher_cache_ready_sha256"]
    ):
        raise QualificationCacheSeedValidationError(
            "legacy seed marker does not identify the exact cache objects"
        )

    teich_key = _safe_key(
        artifacts["teich_pack_key"],
        field="descriptor-pinned Teich pack key",
    )
    teich_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=teich_key,
        label="descriptor-pinned Teich prompt authority",
    )
    teich_sha = _sha256(teich_raw)
    if teich_sha != artifacts["teich_pack_sha256"]:
        raise QualificationCacheSeedValidationError(
            "descriptor-pinned Teich prompt authority SHA-256 mismatch"
        )
    teich_value = _load_json_object(
        teich_raw,
        label="descriptor-pinned Teich prompt authority",
        canonical_file=False,
    )
    _teich_subset, expected_prompt_pack_raw = _expected_teich_prompt_subset(teich_value)
    if cache_raw[prompt_pack_key] != expected_prompt_pack_raw:
        raise QualificationCacheSeedValidationError(
            "qualification cache prompt pack is not the exact Teich prompt "
            "authority subset"
        )
    prompt_pack = _load_json_object(
        cache_raw[prompt_pack_key], label="qualification prompt pack"
    )
    _prompt_pack, supervised_positions = _validate_prompt_pack(prompt_pack)
    manifest_value = _load_json_object(cache_raw[manifest_key], label="cache manifest")
    manifest = _validate_manifest(
        manifest_value,
        prompt_id=EXPECTED_PROMPT_ID,
        supervised_positions=supervised_positions,
    )
    manifest_shard_key = f"{cache_prefix}{manifest['shards'][0]['relative_path']}"  # type: ignore[index]
    if manifest_shard_key != shard_keys[0]:
        raise QualificationCacheSeedValidationError(
            "cache manifest shard key does not match exact S3 inventory"
        )
    ready_value = _load_json_object(
        cache_raw[teacher_ready_key], label="teacher cache ready"
    )
    teacher_ready = _validate_ready(
        ready_value,
        manifest_sha256=_sha256(cache_raw[manifest_key]),
        manifest_body_sha256=str(manifest["manifest_body_sha256"]),
        supervised_positions=supervised_positions,
    )
    if manifest["prompt_pack_sha256"] != _sha256(cache_raw[prompt_pack_key]):
        raise QualificationCacheSeedValidationError(
            "cache manifest prompt-pack SHA-256 mismatch"
        )

    frozen_key = str(artifacts["frozen_prompt_pack_key"])
    frozen_raw = _get_bytes(
        s3_client,
        bucket=pins.bucket,
        key=frozen_key,
        label="frozen prompt pack",
    )
    frozen_sha = _sha256(frozen_raw)
    if (
        frozen_sha != artifacts["frozen_prompt_pack_sha256"]
        or frozen_sha != manifest["frozen_66_prompt_pack_sha256"]
    ):
        raise QualificationCacheSeedValidationError(
            "cache frozen prompt authority SHA-256 mismatch"
        )
    frozen_path = _write_work_file(root, "authorities/frozen-66.json", frozen_raw)
    try:
        semantic = audit_glm52_teich_teacher_cache(
            cache_dir=root,
            prompt_pack_path=root / "prompt-pack.json",
            frozen_prompt_pack_path=frozen_path,
            expected_session_count=EXPECTED_SESSION_COUNT,
            expected_supervised_positions=supervised_positions,
            top_k=EXPECTED_TOP_K,
            hidden_size=EXPECTED_HIDDEN_SIZE,
        )
    except ValueError as error:
        raise QualificationCacheSeedValidationError(
            f"local schema-v3 tensor audit failed: {error}"
        ) from error
    if (
        semantic.manifest_sha256 != _sha256(cache_raw[manifest_key])
        or semantic.manifest_body_sha256 != manifest["manifest_body_sha256"]
        or semantic.session_count != EXPECTED_SESSION_COUNT
        or semantic.supervised_positions != supervised_positions
    ):
        raise QualificationCacheSeedValidationError(
            "local schema-v3 tensor audit returned foreign authority"
        )

    immutable_snapshots = {
        pins.descriptor_key: ("seed descriptor", descriptor_raw),
        approval_key: ("GPU spend approval", approval_raw),
        pins.submission_key: ("immutable seed submission", submission_raw),
        job_binding_key: ("must-start JOB_BINDING.json", job_binding_raw),
        teich_key: ("descriptor-pinned Teich prompt authority", teich_raw),
        frozen_key: ("frozen prompt pack", frozen_raw),
    }
    mutable_snapshots = {
        alias_key: ("mutable seed submission alias", alias_raw),
        allocation_key: ("GPU runtime allocation", allocation_raw),
        ledger_key: ("GPU spend ledger", ledger_raw),
        spend_status_key: ("GPU spend status", spend_status_raw),
        job_status_key: ("SkyPilot job status", job_status_raw),
    }
    for key, (label, original) in {
        **immutable_snapshots,
        **mutable_snapshots,
    }.items():
        current = _get_bytes(
            s3_client,
            bucket=pins.bucket,
            key=key,
            label=f"final {label}",
        )
        if current != original:
            if key == alias_key:
                raise QualificationCacheSeedValidationError(
                    "mutable submission alias drifted during acceptance"
                )
            raise QualificationCacheSeedValidationError(
                f"{label} changed during acceptance"
            )
    final_seed_ready, final_seed_ready_head = _get_full_object(
        s3_client,
        bucket=pins.bucket,
        key=pins.ready_key,
        label="final legacy qualification cache seed ready marker",
        expected_run_id=pins.run_id,
    )
    if final_seed_ready != seed_ready_raw or final_seed_ready_head != _seed_ready_head:
        raise QualificationCacheSeedValidationError(
            "legacy seed ready marker changed during acceptance"
        )
    final_repo_head = _head_full_object(
        s3_client,
        bucket=pins.bucket,
        key=repo_key,
        label="final seed repository tar",
        expected_run_id=pins.run_id,
    )
    if final_repo_head != repo_head:
        raise QualificationCacheSeedValidationError(
            "seed repository tar changed during acceptance"
        )
    _require_no_multipart_uploads(s3_client, bucket=pins.bucket, prefix=cache_prefix)
    if _list_keys(s3_client, bucket=pins.bucket, prefix=cache_prefix) != listed_keys:
        raise QualificationCacheSeedValidationError(
            "exact cache object inventory changed during acceptance"
        )
    final_inventory = [
        _head_full_object(
            s3_client,
            bucket=pins.bucket,
            key=key,
            label=f"final qualification cache object {key}",
            expected_run_id=pins.run_id,
        )
        for key in listed_keys
    ]
    if final_inventory != inventory:
        raise QualificationCacheSeedValidationError(
            "qualification cache object changed during acceptance"
        )

    seed_authority = _seed_authority_record(
        pins=pins,
        descriptor=descriptor,
        job_binding_key=job_binding_key,
        job_binding=job_binding,
        job_binding_file_sha256=_sha256(job_binding_raw),
        job_binding_body_sha256=str(job_binding["job_binding_body_sha256"]),
    )
    cache_object_inventory_sha = _canonical_sha256(inventory)
    cache_audit = {
        "legacy_seed_ready_file_sha256": _sha256(seed_ready_raw),
        "legacy_seed_ready_body_sha256": seed_ready["ready_body_sha256"],
        "cache_prefix": cache_prefix,
        "manifest_key": manifest_key,
        "manifest_file_sha256": _sha256(cache_raw[manifest_key]),
        "manifest_body_sha256": manifest["manifest_body_sha256"],
        "teacher_cache_ready_key": teacher_ready_key,
        "teacher_cache_ready_file_sha256": _sha256(cache_raw[teacher_ready_key]),
        "teacher_cache_ready_body_sha256": teacher_ready["ready_record_sha256"],
        "prompt_pack_key": prompt_pack_key,
        "prompt_pack_file_sha256": _sha256(cache_raw[prompt_pack_key]),
        "teich_pack_key": teich_key,
        "teich_pack_file_sha256": teich_sha,
        "frozen_prompt_pack_key": frozen_key,
        "frozen_prompt_pack_file_sha256": frozen_sha,
        "prompt_id": EXPECTED_PROMPT_ID,
        "session_count": EXPECTED_SESSION_COUNT,
        "supervised_position_count": supervised_positions,
        "top_k": EXPECTED_TOP_K,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "semantic_audit_pass": True,
    }
    spend_closure = {
        "allocation_key": allocation_key,
        "allocation_file_sha256": _sha256(allocation_raw),
        "allocation_body_sha256": allocation["allocation_body_sha256"],
        "allocation_start_record_sha256": allocation["gpu_spend_record_sha256"],
        "spend_authority_sha256": ledger.genesis_sha256,
        "ledger_key": ledger_key,
        "ledger_file_sha256": _sha256(ledger_raw),
        "ledger_tip_record_sha256": ledger.records[-1]["record_sha256"],
        "ledger_tip_event": ledger.records[-1]["event"],
        "spend_status_key": spend_status_key,
        "spend_status_file_sha256": _sha256(spend_status_raw),
        "job_status_key": job_status_key,
        "job_status_file_sha256": _sha256(job_status_raw),
        "job_status_body_sha256": job_status["status_body_sha256"],
        "job_status": job_status["status"],
        "instance_id": instance_id,
        "sky_job_name": _expected_job_name(pins.run_id),
        "submission_submitted_at": submission["submitted_at"],
        "launched_at": allocation["launched_at"],
        "must_start_by": descriptor["must_start_by"],
        "ended_at": spend_status["ended_at"],
        "job_observed_at": job_status["observed_at"],
        "consumed_gpu_seconds": spend_status["consumed_gpu_seconds"],
        "remaining_gpu_seconds": spend_status["remaining_gpu_seconds"],
        "estimated_gpu_cost_usd": spend_status["estimated_gpu_cost_usd"],
    }
    body: dict[str, object] = {
        "schema_version": ACCEPTED_SCHEMA_VERSION,
        "record_type": ACCEPTED_RECORD_TYPE,
        "seed_authority": seed_authority,
        "cache_object_inventory": inventory,
        "cache_object_inventory_sha256": cache_object_inventory_sha,
        "cache_audit": cache_audit,
        "spend_closure": spend_closure,
        "accepted_at": accepted_at_value,
    }
    accepted = {
        **body,
        ACCEPTED_DIGEST_FIELD: _canonical_sha256(body),
    }
    return validate_qualification_cache_seed_accepted(accepted)


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
    _safe_key(
        cache.get("teich_pack_key"),
        field="teich_pack_key",
    )
    _safe_key(
        cache.get("frozen_prompt_pack_key"),
        field="frozen_prompt_pack_key",
    )
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
    "QualificationCacheSeedPins",
    "QualificationCacheSeedValidationError",
    "accepted_s3_key",
    "authenticate_qualification_cache_seed",
    "validate_approved_identity",
    "validate_qualification_cache_seed_accepted",
]
