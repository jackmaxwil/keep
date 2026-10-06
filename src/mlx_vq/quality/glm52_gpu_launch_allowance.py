"""Pure, short-lived authority for the first cache-seed GPU allocation."""

# ruff: noqa: UP017

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

try:
    from glm52_sky_campaign import (
        APPROVED_ACCOUNT_ID,
        APPROVED_GPU_COST_USD,
        APPROVED_GPU_RUNTIME_SECONDS,
        APPROVED_HOURLY_COST_USD,
        APPROVED_INSTANCE_TYPE,
        APPROVED_REGION,
        SkyCampaignValidationError,
        validate_gpu_spend_approval,
        validate_sky_campaign_descriptor,
    )
except ModuleNotFoundError as error:
    if error.name != "glm52_sky_campaign":
        raise
    from mlx_vq.quality.glm52_sky_campaign import (
        APPROVED_ACCOUNT_ID,
        APPROVED_GPU_COST_USD,
        APPROVED_GPU_RUNTIME_SECONDS,
        APPROVED_HOURLY_COST_USD,
        APPROVED_INSTANCE_TYPE,
        APPROVED_REGION,
        SkyCampaignValidationError,
        validate_gpu_spend_approval,
        validate_sky_campaign_descriptor,
    )


SCHEMA_VERSION = 1
RECORD_TYPE = "glm52_gpu_launch_allowance_v1"
MANAGED_MODE = "cache-seed"
DIGEST_FIELD = "allowance_body_sha256"
CACHE_SEED_MAX_SECONDS = 21_600
CACHE_SEED_MAX_COST_USD = 330.24
MAX_ALLOWANCE_AGE_SECONDS = 60
MAX_MUST_START_WINDOW_SECONDS = 43_200

_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_IPV4_BUCKET = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}")
_IAM_ROLE_ARN = re.compile(
    r"arn:aws:iam::([0-9]{12}):role/"
    r"(?:[A-Za-z0-9+=,.@_-]+/)*[A-Za-z0-9+=,.@_-]+"
)
_STS_ASSUMED_ROLE_ARN = re.compile(
    r"arn:aws:sts::([0-9]{12}):assumed-role/"
    r"[A-Za-z0-9+=,.@_-]+/[A-Za-z0-9+=,.@_-]+"
)
_RESERVED_BUCKET_PREFIXES = ("xn--", "sthree-", "amzn-s3-demo-")
_RESERVED_BUCKET_SUFFIXES = (
    "-s3alias",
    "--ol-s3",
    ".mrap",
    "--x-s3",
    "--table-s3",
)
_S3_OBSERVATION_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "latest_marker_key",
    "latest_marker_reads",
    "legacy_ledger_key",
    "legacy_ledger_reads",
    "spend_records_prefix",
    "spend_records_list_pages",
    "observed_at",
}
_ABSENCE_READ_FIELDS = {
    "error_code",
    "http_status_code",
    "phase",
}
_S3_PAGE_FIELDS = {
    "page_index",
    "request_continuation_token",
    "http_status_code",
    "key_count",
    "keys",
    "is_truncated",
    "next_continuation_token",
}
_EXACT_ABSENCE_READS = [
    {
        "error_code": "NoSuchKey",
        "http_status_code": 404,
        "phase": "before-records-list",
    },
    {
        "error_code": "NoSuchKey",
        "http_status_code": 404,
        "phase": "after-records-list",
    },
]
_EC2_OBSERVATION_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "caller_arn",
    "region",
    "run_id",
    "history_request",
    "history_pages",
    "active_request",
    "active_pages",
    "observed_at",
}
_EC2_FILTER_FIELDS = {"name", "values"}
_EC2_PAGE_FIELDS = {
    "page_index",
    "request_next_token",
    "http_status_code",
    "reservations",
    "next_token",
}
_ALLOWANCE_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "managed_mode",
    "sky_job_name",
    "account_id",
    "region",
    "instance_type",
    "instance_count",
    "use_spot",
    "bucket",
    "campaign_identity_sha256",
    "descriptor_sha256",
    "descriptor_body_sha256",
    "descriptor_key",
    "approval_sha256",
    "approval_body_sha256",
    "approval_key",
    "observed_at",
    "must_start_by",
    "not_after",
    "approved_gpu_runtime_seconds",
    "approved_gpu_cost_usd",
    "hourly_cost_usd",
    "consumed_gpu_seconds",
    "remaining_gpu_seconds",
    "consumed_gpu_cost_usd",
    "remaining_gpu_cost_usd",
    "cache_seed_max_seconds",
    "cache_seed_max_cost_usd",
    "allowed_gpu_seconds",
    "allowed_gpu_cost_usd",
    "open_allocation_count",
    "spend_ledger_latest_marker_key",
    "spend_ledger_legacy_key",
    "spend_ledger_records_prefix",
    "s3_spend_ledger_absence_observation_sha256",
    "ec2_tagged_p5_zero_inventory_observation_sha256",
    "ec2_tagged_p5_history_instance_ids",
    "ec2_active_tagged_p5_instance_ids",
}
_ALLOWANCE_FIELDS = _ALLOWANCE_BODY_FIELDS | {DIGEST_FIELD}
_ALLOWANCE_DIGEST_FIELDS = {
    "campaign_identity_sha256",
    "descriptor_sha256",
    "descriptor_body_sha256",
    "approval_sha256",
    "approval_body_sha256",
    "s3_spend_ledger_absence_observation_sha256",
    "ec2_tagged_p5_zero_inventory_observation_sha256",
    DIGEST_FIELD,
}


class GpuLaunchAllowanceError(ValueError):
    """Raised when first-allocation authority is incomplete or divergent."""


class _DuplicateKeyError(ValueError):
    pass


class _NonfiniteJsonError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha_raw(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise GpuLaunchAllowanceError(f"{field} must be a lowercase SHA-256")
    return value


def _authenticate_raw(
    raw: bytes,
    expected_sha256: str,
    *,
    label: str,
) -> str:
    if not isinstance(raw, bytes):
        raise GpuLaunchAllowanceError(f"{label} bytes must be bytes")
    expected = _require_digest(
        expected_sha256,
        field=f"expected {label} SHA-256",
    )
    actual = _sha_raw(raw)
    if actual != expected:
        raise GpuLaunchAllowanceError(f"{label} exact-byte SHA-256 mismatch")
    return actual


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(key)
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> object:
    raise _NonfiniteJsonError(value)


def _object_from_raw(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except _DuplicateKeyError as error:
        raise GpuLaunchAllowanceError(
            f"{label} contains a duplicate object key"
        ) from error
    except _NonfiniteJsonError as error:
        raise GpuLaunchAllowanceError(
            f"{label} contains a nonfinite JSON value"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GpuLaunchAllowanceError(f"{label} is malformed JSON") from error
    if not isinstance(value, dict):
        raise GpuLaunchAllowanceError(f"{label} must contain an object")
    if raw != _canonical(value) + b"\n":
        raise GpuLaunchAllowanceError(f"{label} bytes are not canonical")
    return value


def _require_bucket(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or _BUCKET.fullmatch(value) is None
        or ".." in value
        or _IPV4_BUCKET.fullmatch(value) is not None
        or value.startswith(_RESERVED_BUCKET_PREFIXES)
        or value.endswith(_RESERVED_BUCKET_SUFFIXES)
    ):
        raise GpuLaunchAllowanceError(f"{field} bucket is invalid")
    return value


def _require_safe_key(
    value: object,
    *,
    field: str,
    prefix: bool = False,
) -> str:
    if not isinstance(value, str) or not value:
        raise GpuLaunchAllowanceError(f"{field} S3 key is invalid")
    candidate = value
    if prefix:
        if not candidate.endswith("/") or candidate.endswith("//"):
            raise GpuLaunchAllowanceError(f"{field} S3 prefix is invalid")
        candidate = candidate[:-1]
    elif candidate.startswith("/") or candidate.endswith("/"):
        raise GpuLaunchAllowanceError(f"{field} S3 key is invalid")
    if (
        not candidate
        or any(
            ord(character) < 0x21 or ord(character) > 0x7E for character in candidate
        )
        or "\\" in candidate
        or any(character in candidate for character in "*?[]")
    ):
        raise GpuLaunchAllowanceError(f"{field} S3 key is invalid")
    segments = candidate.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise GpuLaunchAllowanceError(f"{field} S3 key is invalid")
    return value


def _submission_id_from_descriptor_key(
    value: object,
    *,
    run_id: str,
) -> str:
    key = _require_safe_key(value, field="descriptor")
    prefix = f"campaigns/{run_id}/submissions/"
    suffix = "/campaign-descriptor-v2.json"
    if not key.startswith(prefix) or not key.endswith(suffix):
        raise GpuLaunchAllowanceError(
            "descriptor submission attempt coordinate is invalid"
        )
    submission_id = key[len(prefix) : -len(suffix)]
    if not submission_id or submission_id in {".", ".."} or "/" in submission_id:
        raise GpuLaunchAllowanceError(
            "descriptor submission attempt coordinate is invalid"
        )
    return submission_id


def _validate_descriptor_and_approval_binding(
    descriptor: Mapping[str, object],
    approval: Mapping[str, object],
    *,
    approval_sha256: str,
) -> None:
    if descriptor.get("approval_sha256") != approval_sha256:
        raise GpuLaunchAllowanceError(
            "descriptor does not bind the exact GPU spend approval bytes"
        )
    exact_descriptor_values: dict[str, object] = {
        "schema_version": 2,
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "instance_type": APPROVED_INSTANCE_TYPE,
        "instance_count": 1,
        "use_spot": False,
        "max_hourly_cost_usd": APPROVED_HOURLY_COST_USD,
        "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
    }
    exact_descriptor_types = {
        "schema_version": int,
        "account_id": str,
        "region": str,
        "instance_type": str,
        "instance_count": int,
        "use_spot": bool,
        "max_hourly_cost_usd": float,
        "approved_gpu_runtime_seconds": int,
        "approved_gpu_cost_usd": float,
    }
    for field, expected in exact_descriptor_values.items():
        value = descriptor.get(field)
        if type(value) is not exact_descriptor_types[field] or value != expected:
            raise GpuLaunchAllowanceError(
                f"descriptor {field} diverges from approved campaign authority"
            )
    if (
        type(approval.get("schema_version")) is not int
        or approval.get("schema_version") != 1
        or type(approval.get("includes_qualification")) is not bool
        or approval.get("includes_qualification") is not True
        or type(approval.get("includes_recovery_instances")) is not bool
        or approval.get("includes_recovery_instances") is not True
        or type(approval.get("approved_hourly_usd")) is not float
        or approval.get("approved_hourly_usd") != APPROVED_HOURLY_COST_USD
        or type(approval.get("approved_gpu_hours")) is not int
        or approval.get("approved_gpu_hours") * 3600 != APPROVED_GPU_RUNTIME_SECONDS
        or type(approval.get("approved_gpu_cost_usd")) is not float
        or approval.get("approved_gpu_cost_usd") != APPROVED_GPU_COST_USD
        or approval.get("instance_type") != APPROVED_INSTANCE_TYPE
        or approval.get("region") != APPROVED_REGION
    ):
        raise GpuLaunchAllowanceError(
            "descriptor and approval spend authorities diverge"
        )
    bucket = _require_bucket(descriptor.get("bucket"), field="campaign")
    if descriptor.get("jobs_bucket") != bucket:
        raise GpuLaunchAllowanceError(
            "descriptor jobs bucket does not match campaign bucket"
        )
    run_id = descriptor.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise GpuLaunchAllowanceError("descriptor run_id is invalid")
    _submission_id_from_descriptor_key(
        descriptor.get("campaign_descriptor_key"),
        run_id=run_id,
    )
    approval_key = _require_safe_key(
        descriptor.get("approval_key"),
        field="approval",
    )
    if not approval_key.startswith(f"campaigns/{run_id}/authorities/"):
        raise GpuLaunchAllowanceError("descriptor approval key is not campaign-scoped")
    _require_safe_key(descriptor.get("repo_tar_key"), field="repository tar")
    artifacts = descriptor.get("artifacts")
    if not isinstance(artifacts, dict):
        raise GpuLaunchAllowanceError("descriptor artifact inventory is invalid")
    if (
        artifacts.get("qualification_cache_prefix") != "qualification-cache/"
        or artifacts.get("qualification_cache_manifest_sha256") != "0" * 64
    ):
        raise GpuLaunchAllowanceError(
            "descriptor is not the required zero-cache cache-seed authority"
        )
    for field in (
        "source_snapshot_prefix",
        "non_vq_prefix",
        "training_baseline_prefix",
        "qualification_cache_prefix",
    ):
        _require_safe_key(artifacts.get(field), field=field, prefix=True)
    for field in (
        "teich_pack_key",
        "frozen_prompt_pack_key",
        "training_config_key",
        "artifact_inventory_key",
    ):
        _require_safe_key(artifacts.get(field), field=field)


def _validate_absence_reads(value: object, *, field: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(item, dict) or set(item) != _ABSENCE_READ_FIELDS
            for item in value
        )
        or value != _EXACT_ABSENCE_READS
    ):
        raise GpuLaunchAllowanceError(f"S3 {field} absence reads are invalid")


def _validate_s3_record_pages(value: object, *, prefix: str) -> None:
    if not isinstance(value, list) or not value:
        raise GpuLaunchAllowanceError("S3 record listing is incomplete")
    prior_next_token: str | None = None
    seen_tokens: set[str] = set()
    aggregate_keys: list[str] = []
    for expected_index, page in enumerate(value):
        if not isinstance(page, dict) or set(page) != _S3_PAGE_FIELDS:
            raise GpuLaunchAllowanceError("S3 record listing page schema mismatch")
        page_index = page.get("page_index")
        status = page.get("http_status_code")
        key_count = page.get("key_count")
        keys = page.get("keys")
        is_truncated = page.get("is_truncated")
        request_token = page.get("request_continuation_token")
        next_token = page.get("next_continuation_token")
        if type(page_index) is not int or page_index != expected_index:
            raise GpuLaunchAllowanceError("S3 record listing page index is invalid")
        if request_token != prior_next_token or (
            request_token is not None and not isinstance(request_token, str)
        ):
            raise GpuLaunchAllowanceError(
                "S3 record listing pagination request token is invalid"
            )
        if type(status) is not int or status != 200:
            raise GpuLaunchAllowanceError("S3 record listing status is invalid")
        if (
            type(key_count) is not int
            or key_count < 0
            or not isinstance(keys, list)
            or any(not isinstance(key, str) for key in keys)
            or key_count != len(keys)
        ):
            raise GpuLaunchAllowanceError("S3 record listing key inventory is invalid")
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise GpuLaunchAllowanceError(
                "S3 record listing key inventory is not sorted and unique"
            )
        for key in keys:
            _require_safe_key(key, field="spend record")
            if not key.startswith(prefix) or key == prefix:
                raise GpuLaunchAllowanceError(
                    "S3 record listing contains a foreign key"
                )
        if type(is_truncated) is not bool:
            raise GpuLaunchAllowanceError(
                "S3 record listing truncation state is invalid"
            )
        terminal = expected_index == len(value) - 1
        if terminal:
            if is_truncated or next_token is not None:
                raise GpuLaunchAllowanceError(
                    "S3 record listing terminal page is incomplete"
                )
        elif (
            not is_truncated
            or not isinstance(next_token, str)
            or not next_token
            or next_token in seen_tokens
        ):
            raise GpuLaunchAllowanceError(
                "S3 record listing pagination token is invalid"
            )
        if isinstance(next_token, str):
            seen_tokens.add(next_token)
        prior_next_token = next_token
        aggregate_keys.extend(keys)
    if aggregate_keys:
        raise GpuLaunchAllowanceError("S3 spend record inventory must be empty")


def _validate_s3_absence_observation(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
) -> None:
    if set(value) != _S3_OBSERVATION_FIELDS:
        raise GpuLaunchAllowanceError("S3 absence observation schema mismatch")
    run_id = descriptor["run_id"]
    expected_values: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_s3_spend_ledger_absence_observation_v1",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": descriptor["bucket"],
        "run_id": run_id,
        "latest_marker_key": (
            f"campaigns/{run_id}/runtime/GPU_SPEND_LEDGER_LATEST.json"
        ),
        "legacy_ledger_key": (f"campaigns/{run_id}/runtime/GPU_SPEND_LEDGER.jsonl"),
        "spend_records_prefix": (f"campaigns/{run_id}/runtime/spend-ledger/records/"),
    }
    exact_types = {
        "schema_version": int,
        "record_type": str,
        "account_id": str,
        "region": str,
        "bucket": str,
        "run_id": str,
        "latest_marker_key": str,
        "legacy_ledger_key": str,
        "spend_records_prefix": str,
    }
    for field, expected in expected_values.items():
        item = value.get(field)
        if type(item) is not exact_types[field] or item != expected:
            raise GpuLaunchAllowanceError(f"S3 absence observation {field} is invalid")
    _require_bucket(value["bucket"], field="S3 observation")
    _require_safe_key(value["latest_marker_key"], field="latest marker")
    _require_safe_key(value["legacy_ledger_key"], field="legacy ledger")
    prefix = _require_safe_key(
        value["spend_records_prefix"],
        field="spend records",
        prefix=True,
    )
    _validate_absence_reads(
        value.get("latest_marker_reads"),
        field="latest marker",
    )
    _validate_absence_reads(
        value.get("legacy_ledger_reads"),
        field="legacy ledger",
    )
    _validate_s3_record_pages(
        value.get("spend_records_list_pages"),
        prefix=prefix,
    )


def _expected_ec2_filters(run_id: str) -> list[dict[str, object]]:
    return [
        {"name": "instance-type", "values": [APPROVED_INSTANCE_TYPE]},
        {"name": "tag:campaign-run-id", "values": [run_id]},
        {
            "name": "tag:cost-allocation",
            "values": ["glm52-sky-campaign"],
        },
        {"name": "tag:model", "values": ["glm-5.2"]},
        {"name": "tag:owner", "values": ["jack.mazac"]},
        {"name": "tag:project", "values": ["keep-glm52"]},
    ]


def _validate_ec2_request(
    value: object,
    *,
    expected_filters: list[dict[str, object]],
    label: str,
) -> None:
    if not isinstance(value, dict) or set(value) != {"filters"}:
        raise GpuLaunchAllowanceError(f"EC2 {label} filter request schema mismatch")
    filters = value.get("filters")
    if (
        not isinstance(filters, list)
        or any(
            not isinstance(item, dict)
            or set(item) != _EC2_FILTER_FIELDS
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("values"), list)
            or any(
                not isinstance(filter_value, str)
                for filter_value in item.get("values", [])
            )
            for item in filters
        )
        or filters != expected_filters
    ):
        raise GpuLaunchAllowanceError(
            f"EC2 {label} filters do not match exact campaign inventory"
        )


def _validate_ec2_pages(value: object, *, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise GpuLaunchAllowanceError(f"EC2 {label} pages are incomplete")
    prior_next_token: str | None = None
    seen_tokens: set[str] = set()
    for expected_index, page in enumerate(value):
        if not isinstance(page, dict) or set(page) != _EC2_PAGE_FIELDS:
            raise GpuLaunchAllowanceError(f"EC2 {label} pages schema mismatch")
        page_index = page.get("page_index")
        request_token = page.get("request_next_token")
        status = page.get("http_status_code")
        reservations = page.get("reservations")
        next_token = page.get("next_token")
        if type(page_index) is not int or page_index != expected_index:
            raise GpuLaunchAllowanceError(f"EC2 {label} pages index is invalid")
        if request_token != prior_next_token or (
            request_token is not None and not isinstance(request_token, str)
        ):
            raise GpuLaunchAllowanceError(f"EC2 {label} token chain is invalid")
        if type(status) is not int or status != 200:
            raise GpuLaunchAllowanceError(f"EC2 {label} pages status is invalid")
        if not isinstance(reservations, list):
            raise GpuLaunchAllowanceError(f"EC2 {label} pages reservations are invalid")
        if reservations:
            raise GpuLaunchAllowanceError(f"EC2 {label} reservations must be empty")
        terminal = expected_index == len(value) - 1
        if terminal:
            if next_token is not None:
                raise GpuLaunchAllowanceError(
                    f"EC2 {label} pages terminal token is invalid"
                )
        elif (
            not isinstance(next_token, str)
            or not next_token
            or next_token in seen_tokens
        ):
            raise GpuLaunchAllowanceError(f"EC2 {label} token chain is invalid")
        if isinstance(next_token, str):
            seen_tokens.add(next_token)
        prior_next_token = next_token


def _validate_ec2_zero_inventory_observation(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
) -> None:
    if set(value) != _EC2_OBSERVATION_FIELDS:
        raise GpuLaunchAllowanceError("EC2 zero-inventory observation schema mismatch")
    run_id = descriptor["run_id"]
    exact_values: dict[str, object] = {
        "schema_version": 1,
        "record_type": ("glm52_ec2_tagged_p5_zero_inventory_observation_v1"),
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "run_id": run_id,
    }
    exact_types = {
        "schema_version": int,
        "record_type": str,
        "account_id": str,
        "region": str,
        "run_id": str,
    }
    for field, expected in exact_values.items():
        item = value.get(field)
        if type(item) is not exact_types[field] or item != expected:
            raise GpuLaunchAllowanceError(
                f"EC2 zero-inventory observation {field} is invalid"
            )
    caller_arn = value.get("caller_arn")
    iam_match = (
        _IAM_ROLE_ARN.fullmatch(caller_arn) if isinstance(caller_arn, str) else None
    )
    sts_match = (
        _STS_ASSUMED_ROLE_ARN.fullmatch(caller_arn)
        if isinstance(caller_arn, str)
        else None
    )
    match = iam_match or sts_match
    if match is None or match.group(1) != APPROVED_ACCOUNT_ID:
        raise GpuLaunchAllowanceError("EC2 zero-inventory caller ARN is invalid")
    history_filters = _expected_ec2_filters(run_id)
    active_filters = [
        *history_filters,
        {
            "name": "instance-state-name",
            "values": [
                "pending",
                "running",
                "shutting-down",
                "stopping",
            ],
        },
    ]
    _validate_ec2_request(
        value.get("history_request"),
        expected_filters=history_filters,
        label="history",
    )
    _validate_ec2_request(
        value.get("active_request"),
        expected_filters=active_filters,
        label="active",
    )
    _validate_ec2_pages(value.get("history_pages"), label="history")
    _validate_ec2_pages(value.get("active_pages"), label="active")


def _canonical_time(value: datetime | str, *, field: str) -> tuple[datetime, str]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise GpuLaunchAllowanceError(f"{field} must be timezone-aware")
        parsed = value.astimezone(timezone.utc)
    elif isinstance(value, str):
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError as error:
            raise GpuLaunchAllowanceError(
                f"{field} must be canonical whole-second UTC"
            ) from error
    else:
        raise GpuLaunchAllowanceError(f"{field} must be canonical whole-second UTC")
    if parsed.microsecond:
        raise GpuLaunchAllowanceError(f"{field} must be canonical whole-second UTC")
    canonical = parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, str) and value != canonical:
        raise GpuLaunchAllowanceError(f"{field} must be canonical whole-second UTC")
    return parsed, canonical


def _cost(seconds: int, hourly_cost: Decimal) -> Decimal:
    return (Decimal(seconds) * hourly_cost / Decimal(3600)).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


def _exact_integer(
    value: object,
    *,
    field: str,
    expected: int,
) -> int:
    if type(value) is not int or value != expected:
        raise GpuLaunchAllowanceError(f"{field} must be the exact integer {expected}")
    return value


def _money(
    value: object,
    *,
    field: str,
) -> Decimal:
    if type(value) is not float or not math.isfinite(value):
        raise GpuLaunchAllowanceError(f"{field} must be finite JSON float money")
    if value == 0.0 and math.copysign(1.0, value) < 0:
        raise GpuLaunchAllowanceError(f"{field} money must not be negative zero")
    amount = Decimal(str(value))
    if amount < 0 or amount != amount.quantize(Decimal("0.01")):
        raise GpuLaunchAllowanceError(f"{field} must be nonnegative two-decimal money")
    return amount


def build_gpu_launch_allowance(
    *,
    descriptor_raw: bytes,
    expected_descriptor_sha256: str,
    approval_raw: bytes,
    expected_approval_sha256: str,
    s3_spend_ledger_absence_observation_raw: bytes,
    expected_s3_spend_ledger_absence_observation_sha256: str,
    ec2_tagged_p5_zero_inventory_observation_raw: bytes,
    expected_ec2_tagged_p5_zero_inventory_observation_sha256: str,
    observed_at: datetime | str,
) -> dict[str, object]:
    """Authenticate exact pre-seed inputs and build a short-lived allowance."""

    descriptor_sha256 = _authenticate_raw(
        descriptor_raw,
        expected_descriptor_sha256,
        label="Sky campaign descriptor",
    )
    approval_sha256 = _authenticate_raw(
        approval_raw,
        expected_approval_sha256,
        label="GPU spend approval",
    )
    s3_observation_sha256 = _authenticate_raw(
        s3_spend_ledger_absence_observation_raw,
        expected_s3_spend_ledger_absence_observation_sha256,
        label="S3 spend-ledger absence observation",
    )
    ec2_observation_sha256 = _authenticate_raw(
        ec2_tagged_p5_zero_inventory_observation_raw,
        expected_ec2_tagged_p5_zero_inventory_observation_sha256,
        label="EC2 tagged-P5 zero-inventory observation",
    )
    descriptor = _object_from_raw(
        descriptor_raw,
        label="Sky campaign descriptor",
    )
    approval = _object_from_raw(
        approval_raw,
        label="GPU spend approval",
    )
    s3_observation = _object_from_raw(
        s3_spend_ledger_absence_observation_raw,
        label="S3 spend-ledger absence observation",
    )
    ec2_observation = _object_from_raw(
        ec2_tagged_p5_zero_inventory_observation_raw,
        label="EC2 tagged-P5 zero-inventory observation",
    )
    try:
        descriptor = validate_sky_campaign_descriptor(descriptor)
        approval = validate_gpu_spend_approval(approval)
    except SkyCampaignValidationError as error:
        raise GpuLaunchAllowanceError(str(error)) from error
    _validate_descriptor_and_approval_binding(
        descriptor,
        approval,
        approval_sha256=approval_sha256,
    )
    _validate_s3_absence_observation(
        s3_observation,
        descriptor=descriptor,
    )
    _validate_ec2_zero_inventory_observation(
        ec2_observation,
        descriptor=descriptor,
    )

    observation_time, observation_iso = _canonical_time(
        observed_at,
        field="observed_at",
    )
    _s3_observation_time, s3_observation_iso = _canonical_time(
        s3_observation.get("observed_at"),
        field="S3 observation observed_at",
    )
    _ec2_observation_time, ec2_observation_iso = _canonical_time(
        ec2_observation.get("observed_at"),
        field="EC2 observation observed_at",
    )
    if s3_observation_iso != observation_iso or ec2_observation_iso != observation_iso:
        raise GpuLaunchAllowanceError(
            "raw observation time does not match builder observed_at"
        )
    must_start_time, must_start_iso = _canonical_time(
        descriptor["must_start_by"],
        field="must_start_by",
    )
    if observation_time >= must_start_time:
        raise GpuLaunchAllowanceError("observed_at must precede must_start_by")
    if must_start_time - observation_time > timedelta(
        seconds=MAX_MUST_START_WINDOW_SECONDS
    ):
        raise GpuLaunchAllowanceError("must_start_by window exceeds twelve hours")
    not_after_time = min(
        must_start_time,
        observation_time + timedelta(seconds=MAX_ALLOWANCE_AGE_SECONDS),
    )
    run_id = str(descriptor["run_id"])
    hourly_cost = Decimal(str(APPROVED_HOURLY_COST_USD))
    allowed_seconds = CACHE_SEED_MAX_SECONDS
    allowed_cost = _cost(allowed_seconds, hourly_cost)
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "run_id": run_id,
        "managed_mode": MANAGED_MODE,
        "sky_job_name": f"{run_id}-{MANAGED_MODE}",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "instance_type": APPROVED_INSTANCE_TYPE,
        "instance_count": 1,
        "use_spot": False,
        "bucket": descriptor["bucket"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_sha256": descriptor_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "approval_sha256": approval_sha256,
        "approval_body_sha256": approval["approval_body_sha256"],
        "approval_key": descriptor["approval_key"],
        "observed_at": observation_iso,
        "must_start_by": must_start_iso,
        "not_after": not_after_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "approved_gpu_cost_usd": float(Decimal(str(APPROVED_GPU_COST_USD))),
        "hourly_cost_usd": float(hourly_cost),
        "consumed_gpu_seconds": 0,
        "remaining_gpu_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "consumed_gpu_cost_usd": 0.0,
        "remaining_gpu_cost_usd": float(Decimal(str(APPROVED_GPU_COST_USD))),
        "cache_seed_max_seconds": CACHE_SEED_MAX_SECONDS,
        "cache_seed_max_cost_usd": float(Decimal(str(CACHE_SEED_MAX_COST_USD))),
        "allowed_gpu_seconds": allowed_seconds,
        "allowed_gpu_cost_usd": float(allowed_cost),
        "open_allocation_count": 0,
        "spend_ledger_latest_marker_key": s3_observation["latest_marker_key"],
        "spend_ledger_legacy_key": s3_observation["legacy_ledger_key"],
        "spend_ledger_records_prefix": s3_observation["spend_records_prefix"],
        "s3_spend_ledger_absence_observation_sha256": (s3_observation_sha256),
        "ec2_tagged_p5_zero_inventory_observation_sha256": (ec2_observation_sha256),
        "ec2_tagged_p5_history_instance_ids": [],
        "ec2_active_tagged_p5_instance_ids": [],
    }
    allowance = {**body, DIGEST_FIELD: _sha_raw(_canonical(body))}
    return validate_gpu_launch_allowance(allowance)


def validate_gpu_launch_allowance(
    value: Mapping[str, object],
    *,
    now: datetime | str | None = None,
) -> dict[str, object]:
    """Validate a self-contained allowance without consulting external state."""

    if set(value) != _ALLOWANCE_FIELDS:
        raise GpuLaunchAllowanceError("GPU launch allowance schema mismatch")
    body = dict(value)
    claimed_digest = body.pop(DIGEST_FIELD)
    _require_digest(claimed_digest, field=DIGEST_FIELD)
    try:
        actual_digest = _sha_raw(_canonical(body))
    except (TypeError, ValueError) as error:
        raise GpuLaunchAllowanceError(
            "GPU launch allowance body is not finite canonical JSON"
        ) from error
    if claimed_digest != actual_digest:
        raise GpuLaunchAllowanceError("GPU launch allowance body SHA-256 mismatch")
    for field in _ALLOWANCE_DIGEST_FIELDS:
        _require_digest(value.get(field), field=field)
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise GpuLaunchAllowanceError("GPU launch allowance run_id is invalid")
    exact_strings = {
        "record_type": RECORD_TYPE,
        "managed_mode": MANAGED_MODE,
        "sky_job_name": f"{run_id}-{MANAGED_MODE}",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "instance_type": APPROVED_INSTANCE_TYPE,
    }
    for field, expected in exact_strings.items():
        item = value.get(field)
        if not isinstance(item, str) or item != expected:
            raise GpuLaunchAllowanceError(f"GPU launch allowance {field} is invalid")
    _exact_integer(
        value.get("schema_version"),
        field="schema_version",
        expected=SCHEMA_VERSION,
    )
    _exact_integer(
        value.get("instance_count"),
        field="instance_count",
        expected=1,
    )
    if type(value.get("use_spot")) is not bool or value["use_spot"] is not False:
        raise GpuLaunchAllowanceError("GPU launch allowance use_spot must be false")
    integer_values = {
        "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "consumed_gpu_seconds": 0,
        "remaining_gpu_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "cache_seed_max_seconds": CACHE_SEED_MAX_SECONDS,
        "allowed_gpu_seconds": CACHE_SEED_MAX_SECONDS,
        "open_allocation_count": 0,
    }
    for field, expected in integer_values.items():
        _exact_integer(value.get(field), field=field, expected=expected)
    approved_cost = _money(
        value.get("approved_gpu_cost_usd"),
        field="approved_gpu_cost_usd",
    )
    hourly_cost = _money(
        value.get("hourly_cost_usd"),
        field="hourly_cost_usd",
    )
    consumed_cost = _money(
        value.get("consumed_gpu_cost_usd"),
        field="consumed_gpu_cost_usd",
    )
    remaining_cost = _money(
        value.get("remaining_gpu_cost_usd"),
        field="remaining_gpu_cost_usd",
    )
    cache_seed_cost = _money(
        value.get("cache_seed_max_cost_usd"),
        field="cache_seed_max_cost_usd",
    )
    allowed_cost = _money(
        value.get("allowed_gpu_cost_usd"),
        field="allowed_gpu_cost_usd",
    )
    exact_approved_cost = Decimal(str(APPROVED_GPU_COST_USD))
    exact_hourly_cost = Decimal(str(APPROVED_HOURLY_COST_USD))
    exact_cache_seed_cost = Decimal(str(CACHE_SEED_MAX_COST_USD))
    if (
        approved_cost != exact_approved_cost
        or hourly_cost != exact_hourly_cost
        or consumed_cost != _cost(0, hourly_cost)
        or remaining_cost != approved_cost - consumed_cost
        or cache_seed_cost != exact_cache_seed_cost
        or allowed_cost != _cost(CACHE_SEED_MAX_SECONDS, hourly_cost)
        or allowed_cost != cache_seed_cost
    ):
        raise GpuLaunchAllowanceError(
            "GPU launch allowance dollar accounting is inconsistent"
        )
    bucket = _require_bucket(value.get("bucket"), field="allowance")
    del bucket
    _submission_id_from_descriptor_key(
        value.get("descriptor_key"),
        run_id=run_id,
    )
    approval_key = _require_safe_key(
        value.get("approval_key"),
        field="allowance approval",
    )
    if not approval_key.startswith(f"campaigns/{run_id}/authorities/"):
        raise GpuLaunchAllowanceError(
            "GPU launch allowance approval key is not campaign-scoped"
        )
    expected_latest_key = f"campaigns/{run_id}/runtime/GPU_SPEND_LEDGER_LATEST.json"
    expected_legacy_key = f"campaigns/{run_id}/runtime/GPU_SPEND_LEDGER.jsonl"
    expected_records_prefix = f"campaigns/{run_id}/runtime/spend-ledger/records/"
    if value.get("spend_ledger_latest_marker_key") != expected_latest_key:
        raise GpuLaunchAllowanceError(
            "GPU launch allowance latest marker key is invalid"
        )
    if value.get("spend_ledger_legacy_key") != expected_legacy_key:
        raise GpuLaunchAllowanceError(
            "GPU launch allowance legacy ledger key is invalid"
        )
    if value.get("spend_ledger_records_prefix") != expected_records_prefix:
        raise GpuLaunchAllowanceError(
            "GPU launch allowance spend records prefix is invalid"
        )
    _require_safe_key(expected_latest_key, field="allowance latest marker")
    _require_safe_key(expected_legacy_key, field="allowance legacy ledger")
    _require_safe_key(
        expected_records_prefix,
        field="allowance spend records",
        prefix=True,
    )
    for field in (
        "ec2_tagged_p5_history_instance_ids",
        "ec2_active_tagged_p5_instance_ids",
    ):
        if type(value.get(field)) is not list or value[field] != []:
            raise GpuLaunchAllowanceError(
                f"GPU launch allowance {field} must be an empty list"
            )
    observed_time, _observed_iso = _canonical_time(
        value.get("observed_at"),
        field="observed_at",
    )
    must_start_time, _must_start_iso = _canonical_time(
        value.get("must_start_by"),
        field="must_start_by",
    )
    not_after_time, not_after_iso = _canonical_time(
        value.get("not_after"),
        field="not_after",
    )
    if observed_time >= must_start_time:
        raise GpuLaunchAllowanceError("GPU launch allowance must_start_by is expired")
    if must_start_time - observed_time > timedelta(
        seconds=MAX_MUST_START_WINDOW_SECONDS
    ):
        raise GpuLaunchAllowanceError(
            "GPU launch allowance must_start_by window exceeds twelve hours"
        )
    expected_not_after = min(
        must_start_time,
        observed_time + timedelta(seconds=MAX_ALLOWANCE_AGE_SECONDS),
    )
    if not_after_time != expected_not_after:
        raise GpuLaunchAllowanceError("GPU launch allowance not_after is inconsistent")
    if now is not None:
        now_time, _now_iso = _canonical_time(now, field="now")
        if now_time < observed_time or now_time >= not_after_time:
            raise GpuLaunchAllowanceError("GPU launch allowance is not fresh at now")
    del not_after_iso
    return dict(value)


def gpu_launch_allowance_s3_key(
    *,
    run_id: str,
    allowance_body_sha256: str,
) -> str:
    """Return the immutable content-addressed cache-seed allowance key."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise GpuLaunchAllowanceError("allowance key run_id is invalid")
    digest = _require_digest(
        allowance_body_sha256,
        field="allowance_body_sha256",
    )
    key = (
        f"campaigns/{run_id}/submissions/{MANAGED_MODE}/launch-allowances/"
        f"{digest}/GLM52_GPU_LAUNCH_ALLOWANCE.json"
    )
    return _require_safe_key(key, field="GPU launch allowance")


__all__ = [
    "CACHE_SEED_MAX_COST_USD",
    "CACHE_SEED_MAX_SECONDS",
    "DIGEST_FIELD",
    "MANAGED_MODE",
    "MAX_ALLOWANCE_AGE_SECONDS",
    "MAX_MUST_START_WINDOW_SECONDS",
    "RECORD_TYPE",
    "SCHEMA_VERSION",
    "GpuLaunchAllowanceError",
    "build_gpu_launch_allowance",
    "gpu_launch_allowance_s3_key",
    "validate_gpu_launch_allowance",
]
