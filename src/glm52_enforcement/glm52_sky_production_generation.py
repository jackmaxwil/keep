"""Enforcement-native immutable per-generation production authority.

This module validates modeled bytes and provenance only.  It performs no I/O,
does not prove that any caller-supplied S3 view is live or complete, and does
not expose an interface that can consume a receipt or submit a Sky job.
"""

# ruff: noqa: UP017

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Literal, Optional, Union


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
    _campaign_module = _flat_module("glm52_sky_campaign")
    _snapshot_module = _flat_module("glm52_gpu_spend_snapshot")
    _submission_module = _flat_module("glm52_sky_production_submission")
    _acquisition_module = _flat_module("glm52_sky_production_acquisition")
    _modes_module = _flat_module("glm52_sky_submission_modes")
except _FlatModuleUnavailable:
    package = f"{__package__}."
    _campaign_module = importlib.import_module(
        package + "glm52_sky_campaign"
    )
    _snapshot_module = importlib.import_module(
        package + "glm52_gpu_spend_snapshot"
    )
    _submission_module = importlib.import_module(
        package + "glm52_sky_production_submission"
    )
    _acquisition_module = importlib.import_module(
        package + "glm52_sky_production_acquisition"
    )
    _modes_module = importlib.import_module(
        package + "glm52_sky_submission_modes"
    )
if True:
    validate_gpu_spend_approval = _campaign_module.validate_gpu_spend_approval
    validate_sky_campaign_descriptor = _campaign_module.validate_sky_campaign_descriptor
    validate_gpu_spend_snapshot = _snapshot_module.validate_gpu_spend_snapshot
    VersionedJsonArtifact = _submission_module.VersionedJsonArtifact
    production_submission_intent_file_bytes = (
        _submission_module.production_submission_intent_file_bytes
    )
    production_submission_intent_file_sha256 = (
        _submission_module.production_submission_intent_file_sha256
    )
    production_submission_intent_s3_key = (
        _submission_module.production_submission_intent_s3_key
    )
    validate_production_submission_intent = (
        _submission_module.validate_production_submission_intent
    )
    production_controller_baseline_file_bytes = (
        _acquisition_module.production_controller_baseline_file_bytes
    )
    production_controller_baseline_file_sha256 = (
        _acquisition_module.production_controller_baseline_file_sha256
    )
    production_controller_baseline_s3_key = (
        _acquisition_module.production_controller_baseline_s3_key
    )
    production_must_start_control_plane_ready_file_bytes = (
        _acquisition_module.production_must_start_control_plane_ready_file_bytes
    )
    production_must_start_control_plane_ready_file_sha256 = (
        _acquisition_module.production_must_start_control_plane_ready_file_sha256
    )
    production_must_start_control_plane_ready_s3_key = (
        _acquisition_module.production_must_start_control_plane_ready_s3_key
    )
    production_submission_acquired_file_bytes = (
        _acquisition_module.production_submission_acquired_file_bytes
    )
    production_submission_acquired_file_sha256 = (
        _acquisition_module.production_submission_acquired_file_sha256
    )
    production_submission_acquired_s3_key = (
        _acquisition_module.production_submission_acquired_s3_key
    )
    validate_production_controller_baseline = (
        _acquisition_module.validate_production_controller_baseline
    )
    validate_production_must_start_control_plane_ready = (
        _acquisition_module.validate_production_must_start_control_plane_ready
    )
    validate_production_submission_acquired = (
        _acquisition_module.validate_production_submission_acquired
    )
    SubmissionModeContractError = _modes_module.SubmissionModeContractError
    expected_sky_job_name = _modes_module.expected_sky_job_name
    record_contract = _modes_module.record_contract
    require_opaque_version_id = _modes_module.require_opaque_version_id


ProductionGenerationAction = Literal[
    "create-generation-claim",
    "create-launch-once-decision",
    "create-expire-unstarted-decision",
    "create-expired-unstarted-terminal",
    "reconcile-only",
    "advance-generation",
    "fail-closed",
]

ProductionGenerationReason = Literal[
    "empty-inventory-ready-for-generation-one",
    "prior-expired-generation-ready-for-advance",
    "current-claim-ready-for-launch-decision",
    "current-claim-deadline-reached",
    "current-expiry-decision-ready-for-terminal",
    "stored-or-visible-generation-requires-reconciliation",
    "invalid-or-inconsistent-generation-authority",
]


@dataclass(frozen=True)
class GenerationInventoryEntry:
    key: str
    raw: Optional[bytes]
    version_id: str
    is_latest: bool
    is_delete_marker: bool


@dataclass(frozen=True, eq=False)
class UnambiguousStartDecisionCreateReceipt:
    record_kind: Literal["generation-start-decision"]
    account_id: str
    region: str
    bucket: str
    expected_bucket_owner: Literal["246813579024"]
    operation: Literal["PutObject"]
    if_none_match: Literal["*"]
    generation: int
    generation_text: str
    submit_attempt_id: str
    key: str
    content_length: int
    candidate_file_sha256: str
    request_checksum_algorithm: Literal["SHA256"]
    request_checksum_sha256_base64: str
    immutable_metadata: tuple[tuple[str, str], ...]
    version_id: str
    etag: str
    response_checksum_sha256_base64: str
    aws_request_id: str
    server_date: str
    request_started_at: Union[datetime, str]
    response_received_at: Union[datetime, str]
    http_status: int
    outcome: Literal["created"]
    source: Literal["direct-response"]


@dataclass(frozen=True)
class ModeledSubmitOnceValidation:
    validation_result: Literal["modeled-submit-once-valid"]
    generation: int
    submit_attempt_id: str
    sky_job_identity_sha256: str
    claim_key: str
    claim_version_id: str
    start_decision_key: str
    start_decision_version_id: str
    validated_at: str
    conservative_validation_at: str
    must_start_by: str


@dataclass(frozen=True)
class ProductionGenerationDecision:
    action: ProductionGenerationAction
    reason: ProductionGenerationReason
    generation: Optional[int] = None


class ProductionGenerationAuthorityError(ValueError):
    pass


_ACCOUNT_ID = "246813579024"
_REGION = "us-west-2"
_MODE = "production"
_MAX_GENERATION = 99_999_999
_APPROVED_SECONDS = 86_400
_APPROVED_COST = 1_320.96
_HOURLY_COST = 55.04
_MAX_AGE = timedelta(seconds=60)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_ATTEMPT = re.compile(r"[0-9a-f]{64}")
_CHECKSUM_BASE64 = re.compile(r"[A-Za-z0-9+/]{43}=")
_ETAG = re.compile(r'"[0-9a-f]{32}"')
_REQUEST_ID = re.compile(r"[A-Za-z0-9]{16,128}")
_DURABLE_TIME = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})Z"
)
_IMF_TIME = re.compile(
    r"[A-Z][a-z]{2}, [0-9]{2} [A-Z][a-z]{2} [0-9]{4} "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2} GMT"
)

_APPROVAL_FIELDS = {
    "schema_version",
    "record_type",
    "approved_by",
    "approval_request",
    "approval_response",
    "approval_displayed_time",
    "approval_ingested_at",
    "evidence_source",
    "slack_permalink",
    "approved_hourly_usd",
    "approved_gpu_hours",
    "approved_gpu_cost_usd",
    "instance_type",
    "region",
    "includes_qualification",
    "includes_recovery_instances",
    "approval_body_sha256",
}

_CLAIM_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "generation",
    "generation_text",
    "submit_attempt_id",
    "sky_job_name",
    "sky_job_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "controller_baseline_key",
    "controller_baseline_file_sha256",
    "controller_baseline_body_sha256",
    "controller_baseline_version_id",
    "must_start_control_plane_ready_key",
    "must_start_control_plane_ready_file_sha256",
    "must_start_control_plane_ready_body_sha256",
    "must_start_control_plane_ready_version_id",
    "submission_acquisition_key",
    "submission_acquisition_file_sha256",
    "submission_acquisition_body_sha256",
    "submission_acquisition_version_id",
    "approval_key",
    "approval_file_sha256",
    "approval_body_sha256",
    "approval_version_id",
    "gpu_spend_snapshot_key",
    "gpu_spend_snapshot_file_sha256",
    "gpu_spend_snapshot_body_sha256",
    "gpu_spend_snapshot_version_id",
    "gpu_spend_ledger_genesis_sha256",
    "gpu_spend_ledger_record_count",
    "gpu_spend_ledger_tip_record_sha256",
    "ec2_allocation_history_sha256",
    "approved_gpu_runtime_seconds",
    "approved_gpu_cost_usd",
    "hourly_cost_usd",
    "consumed_gpu_seconds",
    "consumed_gpu_cost_usd",
    "remaining_gpu_seconds",
    "remaining_gpu_cost_usd",
    "open_allocation_count",
    "previous_generation",
    "previous_generation_terminal_key",
    "previous_generation_terminal_file_sha256",
    "previous_generation_terminal_body_sha256",
    "previous_generation_terminal_version_id",
    "must_start_by",
    "claim_created_at",
    "generation_claim_body_sha256",
}

_DECISION_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "generation",
    "generation_text",
    "submit_attempt_id",
    "sky_job_name",
    "sky_job_identity_sha256",
    "generation_claim_key",
    "generation_claim_file_sha256",
    "generation_claim_body_sha256",
    "generation_claim_version_id",
    "decision",
    "must_start_by",
    "decided_at",
    "start_decision_body_sha256",
}

_TERMINAL_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "generation",
    "generation_text",
    "submit_attempt_id",
    "sky_job_name",
    "sky_job_identity_sha256",
    "generation_claim_key",
    "generation_claim_file_sha256",
    "generation_claim_body_sha256",
    "generation_claim_version_id",
    "start_decision_key",
    "start_decision_file_sha256",
    "start_decision_body_sha256",
    "start_decision_version_id",
    "terminal_outcome",
    "must_start_by",
    "terminal_at",
    "final_gpu_spend_snapshot_key",
    "final_gpu_spend_snapshot_file_sha256",
    "final_gpu_spend_snapshot_body_sha256",
    "final_gpu_spend_snapshot_version_id",
    "final_gpu_spend_ledger_genesis_sha256",
    "final_gpu_spend_ledger_record_count",
    "final_gpu_spend_ledger_tip_record_sha256",
    "final_ec2_allocation_history_sha256",
    "final_approved_gpu_runtime_seconds",
    "final_approved_gpu_cost_usd",
    "final_hourly_cost_usd",
    "final_consumed_gpu_seconds",
    "final_consumed_gpu_cost_usd",
    "final_remaining_gpu_seconds",
    "final_remaining_gpu_cost_usd",
    "final_open_allocation_count",
    "generation_terminal_body_sha256",
}

_CLAIM_INT_FIELDS = {
    "schema_version",
    "generation",
    "gpu_spend_ledger_record_count",
    "approved_gpu_runtime_seconds",
    "consumed_gpu_seconds",
    "remaining_gpu_seconds",
    "open_allocation_count",
    "previous_generation",
}
_CLAIM_FLOAT_FIELDS = {
    "approved_gpu_cost_usd",
    "hourly_cost_usd",
    "consumed_gpu_cost_usd",
    "remaining_gpu_cost_usd",
}
_CLAIM_OPTIONAL_FIELDS = {
    "previous_generation_terminal_key",
    "previous_generation_terminal_file_sha256",
    "previous_generation_terminal_body_sha256",
    "previous_generation_terminal_version_id",
}
_CLAIM_STRING_FIELDS = _CLAIM_FIELDS - _CLAIM_INT_FIELDS - _CLAIM_FLOAT_FIELDS - _CLAIM_OPTIONAL_FIELDS

_DECISION_INT_FIELDS = {"schema_version", "generation"}
_DECISION_STRING_FIELDS = _DECISION_FIELDS - _DECISION_INT_FIELDS

_TERMINAL_INT_FIELDS = {
    "schema_version",
    "generation",
    "final_gpu_spend_ledger_record_count",
    "final_approved_gpu_runtime_seconds",
    "final_consumed_gpu_seconds",
    "final_remaining_gpu_seconds",
    "final_open_allocation_count",
}
_TERMINAL_FLOAT_FIELDS = {
    "final_approved_gpu_cost_usd",
    "final_hourly_cost_usd",
    "final_consumed_gpu_cost_usd",
    "final_remaining_gpu_cost_usd",
}
_TERMINAL_STRING_FIELDS = _TERMINAL_FIELDS - _TERMINAL_INT_FIELDS - _TERMINAL_FLOAT_FIELDS


def _translate(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except ProductionGenerationAuthorityError:
        raise
    except MemoryError:
        raise
    except SubmissionModeContractError as error:
        raise ProductionGenerationAuthorityError(str(error)) from error
    except Exception as error:
        raise ProductionGenerationAuthorityError(str(error)) from error


def _require_exact_json(value: object, *, field: str) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ProductionGenerationAuthorityError(f"{field} must be finite")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _require_exact_json(item, field=f"{field}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ProductionGenerationAuthorityError(
                    f"{field} keys must be exact strings"
                )
            _require_exact_json(item, field=f"{field}.{key}")
        return
    raise ProductionGenerationAuthorityError(f"{field} is not exact JSON")


def _canonical(value: object) -> bytes:
    _require_exact_json(value, field="value")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha(value: Mapping[str, object], digest_field: str) -> str:
    body = dict(value)
    if digest_field not in body:
        raise ProductionGenerationAuthorityError(f"{digest_field} is missing")
    body.pop(digest_field)
    return _sha(_canonical(body))


def _duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProductionGenerationAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProductionGenerationAuthorityError(
        f"nonfinite JSON number is forbidden: {value}"
    )


def _parse_raw(raw: object, *, label: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise ProductionGenerationAuthorityError(f"{label}.raw must be exact bytes")
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionGenerationAuthorityError(
            f"{label}.raw is malformed JSON"
        ) from error
    if type(value) is not dict:
        raise ProductionGenerationAuthorityError(f"{label}.raw must contain an object")
    _require_exact_json(value, field=label)
    if _canonical(value) + b"\n" != raw:
        raise ProductionGenerationAuthorityError(f"{label}.raw is not canonical")
    return value


def _exact_dict(
    value: Mapping[str, object],
    fields: set[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ProductionGenerationAuthorityError(f"{label} must be an exact object")
    if set(value) != fields:
        raise ProductionGenerationAuthorityError(f"{label} schema mismatch")
    return dict(value)


def _exact_string(value: object, *, field: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise ProductionGenerationAuthorityError(f"{field} must be an exact string")
    return value


def _digest(value: object, *, field: str) -> str:
    text = _exact_string(value, field=field)
    if _SHA256.fullmatch(text) is None:
        raise ProductionGenerationAuthorityError(
            f"{field} must be a lowercase SHA-256"
        )
    return text


def _exact_integer(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise ProductionGenerationAuthorityError(
            f"{field} must be an exact integer >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise ProductionGenerationAuthorityError(
            f"{field} must be <= {maximum}"
        )
    return value


def _exact_float(value: object, *, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ProductionGenerationAuthorityError(
            f"{field} must be an exact finite float"
        )
    return value


def _run_id(value: object) -> str:
    text = _exact_string(value, field="run_id")
    if _RUN_ID.fullmatch(text) is None:
        raise ProductionGenerationAuthorityError("run_id is invalid")
    return text


def _bucket(value: object) -> str:
    text = _exact_string(value, field="bucket")
    if _BUCKET.fullmatch(text) is None:
        raise ProductionGenerationAuthorityError("bucket is invalid")
    return text


def _safe_key(value: object, *, field: str) -> str:
    text = _exact_string(value, field=field)
    if (
        text.startswith("/")
        or "\\" in text
        or any(character in text for character in "*?[]")
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in text)
    ):
        raise ProductionGenerationAuthorityError(f"{field} is unsafe")
    segments = text.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ProductionGenerationAuthorityError(f"{field} is unsafe")
    return text


def _version(value: object, *, field: str) -> str:
    try:
        return require_opaque_version_id(value, field=field)
    except SubmissionModeContractError as error:
        raise ProductionGenerationAuthorityError(str(error)) from error


def _generation(value: object) -> int:
    return _exact_integer(
        value,
        field="generation",
        minimum=1,
        maximum=_MAX_GENERATION,
    )


def _attempt(value: object) -> str:
    text = _exact_string(value, field="submit_attempt_id")
    if _ATTEMPT.fullmatch(text) is None:
        raise ProductionGenerationAuthorityError(
            "submit_attempt_id must be 64 lowercase hexadecimal characters"
        )
    return text


def _parse_durable_time(value: object, *, field: str) -> datetime:
    text = _exact_string(value, field=field)
    match = _DURABLE_TIME.fullmatch(text)
    if match is None:
        raise ProductionGenerationAuthorityError(
            f"{field} must be whole-second canonical UTC"
        )
    try:
        parsed = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            tzinfo=timezone.utc,
        )
    except ValueError as error:
        raise ProductionGenerationAuthorityError(
            f"{field} must be whole-second canonical UTC"
        ) from error
    if _format_durable_time(parsed) != text:
        raise ProductionGenerationAuthorityError(
            f"{field} must be whole-second canonical UTC"
        )
    return parsed


def _format_durable_time(value: datetime) -> str:
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
        f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}Z"
    )


def _ceil_whole_second(value: datetime) -> datetime:
    if value.microsecond == 0:
        return value
    try:
        return value.replace(microsecond=0) + timedelta(seconds=1)
    except OverflowError as error:
        raise ProductionGenerationAuthorityError(
            "time ceiling overflows UTC range"
        ) from error


def _normalize_time_input(
    value: Union[datetime, str],
    *,
    field: str,
) -> tuple[datetime, str]:
    if type(value) is str:
        parsed = _parse_durable_time(value, field=field)
        return parsed, value
    if type(value) is not datetime:
        raise ProductionGenerationAuthorityError(
            f"{field} must be an exact datetime or canonical UTC string"
        )
    zone = value.tzinfo
    if type(zone) is not timezone:
        raise ProductionGenerationAuthorityError(
            f"{field} tzinfo must be an exact datetime.timezone"
        )
    try:
        offset = zone.utcoffset(value)
    except MemoryError:
        raise
    except Exception as error:
        raise ProductionGenerationAuthorityError(
            f"{field} UTC offset lookup failed"
        ) from error
    if (
        type(offset) is not timedelta
        or offset.microseconds != 0
        or offset.total_seconds() % 60 != 0
        or abs(offset) >= timedelta(hours=24)
    ):
        raise ProductionGenerationAuthorityError(
            f"{field} UTC offset must be whole-minute and below 24 hours"
        )
    try:
        local_naive = datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
        )
        utc_naive = local_naive - offset
        exact = utc_naive.replace(tzinfo=timezone.utc)
        ceiling = _ceil_whole_second(exact)
    except (OverflowError, ValueError) as error:
        raise ProductionGenerationAuthorityError(
            f"{field} UTC normalization overflowed"
        ) from error
    return exact, _format_durable_time(ceiling)


def _parse_imf_time(value: object, *, field: str) -> datetime:
    text = _exact_string(value, field=field)
    if _IMF_TIME.fullmatch(text) is None:
        raise ProductionGenerationAuthorityError(
            f"{field} must be exact IMF-fixdate"
        )
    try:
        parsed = datetime.strptime(text, "%a, %d %b %Y %H:%M:%S GMT").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ProductionGenerationAuthorityError(
            f"{field} must be exact IMF-fixdate"
        ) from error
    if parsed.strftime("%a, %d %b %Y %H:%M:%S GMT") != text:
        raise ProductionGenerationAuthorityError(
            f"{field} must be exact IMF-fixdate"
        )
    return parsed


def _artifact(
    value: object,
    *,
    label: str,
) -> tuple[VersionedJsonArtifact, dict[str, object], str]:
    if type(value) is not VersionedJsonArtifact:
        raise ProductionGenerationAuthorityError(
            f"{label} must be an exact VersionedJsonArtifact"
        )
    _safe_key(value.key, field=f"{label}.key")
    _version(value.version_id, field=f"{label}.version_id")
    record = _parse_raw(value.raw, label=label)
    return value, record, _sha(value.raw)


def _approval_key(run_id: str, file_sha256: str) -> str:
    return (
        f"campaigns/{run_id}/authorities/"
        f"GPU_SPEND_APPROVAL-{file_sha256}.json"
    )


def _snapshot_key(run_id: str, body_sha256: str) -> str:
    return (
        f"campaigns/{run_id}/spend-snapshots/"
        f"{body_sha256}/GPU_SPEND_SNAPSHOT.json"
    )


def _authenticate_approval(
    artifact: object,
    *,
    run_id: str,
    expected_key: Optional[object] = None,
    expected_file_sha256: Optional[object] = None,
    expected_body_sha256: Optional[object] = None,
    expected_version_id: Optional[object] = None,
) -> tuple[dict[str, object], str]:
    exact, record, file_sha = _artifact(artifact, label="approval")
    if set(record) != _APPROVAL_FIELDS:
        raise ProductionGenerationAuthorityError("approval schema mismatch")
    int_fields = {"schema_version", "approved_gpu_hours"}
    float_fields = {"approved_hourly_usd", "approved_gpu_cost_usd"}
    bool_fields = {"includes_qualification", "includes_recovery_instances"}
    nullable_fields = {"slack_permalink"}
    for field in _APPROVAL_FIELDS:
        item = record[field]
        if field in int_fields:
            if type(item) is not int:
                raise ProductionGenerationAuthorityError(
                    f"approval.{field} must be an exact integer"
                )
        elif field in float_fields:
            if type(item) is not float or not math.isfinite(item):
                raise ProductionGenerationAuthorityError(
                    f"approval.{field} must be an exact finite float"
                )
        elif field in bool_fields:
            if type(item) is not bool or item is not True:
                raise ProductionGenerationAuthorityError(
                    f"approval.{field} must be exact True"
                )
        elif field in nullable_fields:
            if item is not None and type(item) is not str:
                raise ProductionGenerationAuthorityError(
                    "approval.slack_permalink must be exact None or string"
                )
        elif type(item) is not str:
            raise ProductionGenerationAuthorityError(
                f"approval.{field} must be an exact string"
            )
    _parse_durable_time(
        record["approval_ingested_at"],
        field="approval.approval_ingested_at",
    )
    body_sha = _digest(
        record["approval_body_sha256"],
        field="approval.approval_body_sha256",
    )
    if body_sha != _body_sha(record, "approval_body_sha256"):
        raise ProductionGenerationAuthorityError("approval body SHA-256 mismatch")
    canonical_key = _approval_key(run_id, file_sha)
    if exact.key != canonical_key:
        raise ProductionGenerationAuthorityError(
            "approval key is not the exact content-addressed coordinate"
        )
    if expected_key is not None and exact.key != expected_key:
        raise ProductionGenerationAuthorityError("approval key lineage mismatch")
    if expected_file_sha256 is not None and file_sha != expected_file_sha256:
        raise ProductionGenerationAuthorityError("approval file SHA-256 mismatch")
    if expected_body_sha256 is not None and body_sha != expected_body_sha256:
        raise ProductionGenerationAuthorityError("approval body SHA-256 lineage mismatch")
    if expected_version_id is not None and exact.version_id != expected_version_id:
        raise ProductionGenerationAuthorityError("approval VersionId mismatch")
    validated = validate_gpu_spend_approval(
        _parse_raw(exact.raw, label="approval validation input")
    )
    if type(validated) is not dict or validated != record:
        raise ProductionGenerationAuthorityError("approval public validation mismatch")
    return dict(record), file_sha


def _authenticate_descriptor(
    artifact: object,
) -> tuple[VersionedJsonArtifact, dict[str, object], str]:
    exact, record, file_sha = _artifact(artifact, label="descriptor")
    validated = validate_sky_campaign_descriptor(
        _parse_raw(exact.raw, label="descriptor validation input")
    )
    if type(validated) is not dict or validated != record:
        raise ProductionGenerationAuthorityError("descriptor validation mismatch")
    if (
        type(record.get("schema_version")) is not int
        or record.get("schema_version") != 2
        or type(record.get("record_type")) is not str
        or type(record.get("run_id")) is not str
        or type(record.get("bucket")) is not str
    ):
        raise ProductionGenerationAuthorityError("descriptor exact type mismatch")
    _run_id(record["run_id"])
    _bucket(record["bucket"])
    if exact.key != record.get("campaign_descriptor_key"):
        raise ProductionGenerationAuthorityError("descriptor key mismatch")
    if record.get("account_id") != _ACCOUNT_ID or record.get("region") != _REGION:
        raise ProductionGenerationAuthorityError("descriptor cloud identity mismatch")
    if (
        record.get("instance_type") != "p5.48xlarge"
        or record.get("instance_count") != 1
        or record.get("use_spot") is not False
        or record.get("max_hourly_cost_usd") != _HOURLY_COST
        or record.get("approved_gpu_runtime_seconds") != _APPROVED_SECONDS
        or record.get("approved_gpu_cost_usd") != _APPROVED_COST
    ):
        raise ProductionGenerationAuthorityError("descriptor spend envelope mismatch")
    _digest(record.get("descriptor_body_sha256"), field="descriptor_body_sha256")
    _digest(record.get("campaign_identity_sha256"), field="campaign_identity_sha256")
    _parse_durable_time(record.get("must_start_by"), field="descriptor.must_start_by")
    return exact, dict(record), file_sha


def _authenticate_intent(
    artifact: object,
) -> tuple[VersionedJsonArtifact, dict[str, object], str]:
    exact, record, file_sha = _artifact(artifact, label="intent")
    validated = validate_production_submission_intent(
        _parse_raw(exact.raw, label="intent validation input")
    )
    if type(validated) is not dict or validated != record:
        raise ProductionGenerationAuthorityError("intent validation mismatch")
    run = _run_id(record.get("run_id"))
    expected_key = production_submission_intent_s3_key(
        run_id=run,
        intent_body_sha256=_digest(
            record.get("intent_body_sha256"),
            field="intent.intent_body_sha256",
        ),
    )
    if exact.key != expected_key:
        raise ProductionGenerationAuthorityError("intent key mismatch")
    if production_submission_intent_file_bytes(record) != exact.raw:
        raise ProductionGenerationAuthorityError("intent exact bytes mismatch")
    if production_submission_intent_file_sha256(record) != file_sha:
        raise ProductionGenerationAuthorityError("intent file SHA-256 mismatch")
    return exact, dict(record), file_sha


def _authenticate_snapshot(
    artifact: object,
    *,
    run_id: str,
) -> tuple[VersionedJsonArtifact, dict[str, object], str]:
    exact, record, file_sha = _artifact(artifact, label="gpu_spend_snapshot")
    int_fields = {
        "schema_version",
        "gpu_spend_ledger_record_count",
        "approved_gpu_runtime_seconds",
        "consumed_gpu_seconds",
        "remaining_gpu_seconds",
        "qualification_allowance_seconds",
        "open_allocation_count",
    }
    float_fields = {
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "consumed_gpu_cost_usd",
        "remaining_gpu_cost_usd",
        "qualification_allowance_cost_usd",
    }
    for field in int_fields:
        if type(record.get(field)) is not int:
            raise ProductionGenerationAuthorityError(
                f"gpu_spend_snapshot.{field} must be an exact integer"
            )
    for field in float_fields:
        if type(record.get(field)) is not float or not math.isfinite(record[field]):
            raise ProductionGenerationAuthorityError(
                f"gpu_spend_snapshot.{field} must be an exact finite float"
            )
    validated = validate_gpu_spend_snapshot(
        _parse_raw(exact.raw, label="gpu_spend_snapshot validation input")
    )
    if type(validated) is not dict or validated != record:
        raise ProductionGenerationAuthorityError("spend snapshot validation mismatch")
    body_sha = _digest(
        record.get("snapshot_body_sha256"),
        field="snapshot_body_sha256",
    )
    if body_sha != _body_sha(record, "snapshot_body_sha256"):
        raise ProductionGenerationAuthorityError("snapshot body SHA-256 mismatch")
    if exact.key != _snapshot_key(run_id, body_sha):
        raise ProductionGenerationAuthorityError("snapshot key mismatch")
    _parse_durable_time(record.get("observed_at"), field="snapshot.observed_at")
    return exact, dict(record), file_sha


def _authenticate_h1d(
    *,
    descriptor: object,
    intent: object,
    approval: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
    submission_acquisition: object,
    now: Union[datetime, str],
    gpu_spend_snapshot: Optional[object] = None,
) -> dict[str, object]:
    now_exact, now_text = _normalize_time_input(now, field="authority_time")
    descriptor_artifact, descriptor_record, descriptor_file_sha = (
        _authenticate_descriptor(descriptor)
    )
    intent_artifact, intent_record, intent_file_sha = _authenticate_intent(intent)
    run = _run_id(intent_record["run_id"])
    if (
        descriptor_record["run_id"] != run
        or descriptor_record["bucket"] != intent_record.get("bucket")
        or descriptor_record["campaign_identity_sha256"]
        != intent_record.get("campaign_identity_sha256")
        or descriptor_artifact.key != intent_record.get("descriptor_key")
        or descriptor_file_sha != intent_record.get("descriptor_file_sha256")
        or descriptor_record["descriptor_body_sha256"]
        != intent_record.get("descriptor_body_sha256")
        or descriptor_artifact.version_id != intent_record.get("descriptor_version_id")
    ):
        raise ProductionGenerationAuthorityError("descriptor/intent lineage mismatch")
    approval_record, approval_file_sha = _authenticate_approval(
        approval,
        run_id=run,
        expected_key=descriptor_record.get("approval_key"),
        expected_file_sha256=descriptor_record.get("approval_sha256"),
        expected_body_sha256=intent_record.get("approval_body_sha256"),
        expected_version_id=intent_record.get("approval_version_id"),
    )
    approval_artifact = approval
    if (
        intent_record.get("approval_key") != approval_artifact.key
        or intent_record.get("approval_file_sha256") != approval_file_sha
        or intent_record.get("approval_body_sha256")
        != approval_record["approval_body_sha256"]
    ):
        raise ProductionGenerationAuthorityError("intent approval lineage mismatch")

    baseline_artifact, baseline_record, baseline_file_sha = _artifact(
        controller_baseline,
        label="controller_baseline",
    )
    validated_baseline = validate_production_controller_baseline(
        _parse_raw(
            baseline_artifact.raw,
            label="controller_baseline validation input",
        ),
        descriptor=descriptor_artifact,
        intent=intent_artifact,
    )
    if type(validated_baseline) is not dict or validated_baseline != baseline_record:
        raise ProductionGenerationAuthorityError("baseline validation mismatch")
    baseline_body_sha = _digest(
        baseline_record.get("baseline_body_sha256"),
        field="baseline_body_sha256",
    )
    if baseline_artifact.key != production_controller_baseline_s3_key(
        run_id=run,
        baseline_body_sha256=baseline_body_sha,
    ):
        raise ProductionGenerationAuthorityError("baseline key mismatch")
    if production_controller_baseline_file_bytes(baseline_record) != baseline_artifact.raw:
        raise ProductionGenerationAuthorityError("baseline exact bytes mismatch")
    if (
        production_controller_baseline_file_sha256(baseline_record)
        != baseline_file_sha
    ):
        raise ProductionGenerationAuthorityError("baseline file SHA-256 mismatch")

    ready_artifact, ready_record, ready_file_sha = _artifact(
        must_start_control_plane_ready,
        label="must_start_control_plane_ready",
    )
    validated_ready = validate_production_must_start_control_plane_ready(
        _parse_raw(
            ready_artifact.raw,
            label="must_start_control_plane_ready validation input",
        ),
        descriptor=descriptor_artifact,
        intent=intent_artifact,
        controller_baseline=baseline_artifact,
    )
    if type(validated_ready) is not dict or validated_ready != ready_record:
        raise ProductionGenerationAuthorityError("readiness validation mismatch")
    ready_body_sha = _digest(
        ready_record.get("control_plane_ready_body_sha256"),
        field="control_plane_ready_body_sha256",
    )
    if ready_artifact.key != production_must_start_control_plane_ready_s3_key(
        run_id=run,
        intent_body_sha256=_digest(
            intent_record.get("intent_body_sha256"),
            field="intent_body_sha256",
        ),
        control_plane_ready_body_sha256=ready_body_sha,
    ):
        raise ProductionGenerationAuthorityError("readiness key mismatch")
    if (
        production_must_start_control_plane_ready_file_bytes(ready_record)
        != ready_artifact.raw
        or production_must_start_control_plane_ready_file_sha256(ready_record)
        != ready_file_sha
    ):
        raise ProductionGenerationAuthorityError("readiness file mismatch")

    acquisition_artifact, acquisition_record, acquisition_file_sha = _artifact(
        submission_acquisition,
        label="submission_acquisition",
    )
    validated_acquisition = validate_production_submission_acquired(
        _parse_raw(
            acquisition_artifact.raw,
            label="submission_acquisition validation input",
        ),
        descriptor=descriptor_artifact,
        intent=intent_artifact,
        controller_baseline=baseline_artifact,
        must_start_control_plane_ready=ready_artifact,
        now=now_text,
    )
    if (
        type(validated_acquisition) is not dict
        or validated_acquisition != acquisition_record
    ):
        raise ProductionGenerationAuthorityError("acquisition validation mismatch")
    _digest(
        acquisition_record.get("acquisition_body_sha256"),
        field="acquisition_body_sha256",
    )
    if acquisition_artifact.key != production_submission_acquired_s3_key(
        run_id=run,
        descriptor_file_sha256=descriptor_file_sha,
    ):
        raise ProductionGenerationAuthorityError("acquisition key mismatch")
    if (
        production_submission_acquired_file_bytes(acquisition_record)
        != acquisition_artifact.raw
        or production_submission_acquired_file_sha256(acquisition_record)
        != acquisition_file_sha
    ):
        raise ProductionGenerationAuthorityError("acquisition file mismatch")

    snapshot_artifact: Optional[VersionedJsonArtifact] = None
    snapshot_record: Optional[dict[str, object]] = None
    snapshot_file_sha: Optional[str] = None
    if gpu_spend_snapshot is not None:
        snapshot_artifact, snapshot_record, snapshot_file_sha = _authenticate_snapshot(
            gpu_spend_snapshot,
            run_id=run,
        )
        snapshot_intent_fields = {
            "observed_at": "spend_snapshot_observed_at",
            "gpu_spend_ledger_tip_record_sha256": (
                "gpu_spend_ledger_tip_record_sha256"
            ),
            "gpu_spend_ledger_genesis_sha256": (
                "gpu_spend_ledger_genesis_sha256"
            ),
            "ec2_allocation_history_sha256": "ec2_allocation_history_sha256",
            "open_allocation_count": "open_allocation_count",
            "approved_gpu_runtime_seconds": "approved_gpu_runtime_seconds",
            "approved_gpu_cost_usd": "approved_gpu_cost_usd",
            "hourly_cost_usd": "hourly_cost_usd",
            "consumed_gpu_seconds": "consumed_gpu_seconds",
            "consumed_gpu_cost_usd": "consumed_gpu_cost_usd",
            "remaining_gpu_seconds": "remaining_gpu_seconds",
            "remaining_gpu_cost_usd": "remaining_gpu_cost_usd",
        }
        if (
            intent_record.get("gpu_spend_snapshot_key") != snapshot_artifact.key
            or intent_record.get("gpu_spend_snapshot_file_sha256")
            != snapshot_file_sha
            or intent_record.get("gpu_spend_snapshot_body_sha256")
            != snapshot_record["snapshot_body_sha256"]
            or intent_record.get("gpu_spend_snapshot_version_id")
            != snapshot_artifact.version_id
            or snapshot_record.get("run_id") != run
            or snapshot_record.get("campaign_identity_sha256")
            != intent_record.get("campaign_identity_sha256")
            or snapshot_record.get("descriptor_sha256") != descriptor_file_sha
            or snapshot_record.get("descriptor_body_sha256")
            != descriptor_record["descriptor_body_sha256"]
            or snapshot_record.get("approval_sha256") != approval_file_sha
            or snapshot_record.get("approval_body_sha256")
            != approval_record["approval_body_sha256"]
            or any(
                snapshot_record.get(snapshot_field)
                != intent_record.get(intent_field)
                for snapshot_field, intent_field in snapshot_intent_fields.items()
            )
        ):
            raise ProductionGenerationAuthorityError("snapshot lineage mismatch")

    intent_at = _parse_durable_time(intent_record.get("intent_at"), field="intent_at")
    baseline_at = _parse_durable_time(
        baseline_record.get("observed_at"),
        field="baseline.observed_at",
    )
    ready_at = _parse_durable_time(
        ready_record.get("observed_at"),
        field="ready.observed_at",
    )
    acquired_at = _parse_durable_time(
        acquisition_record.get("acquired_at"),
        field="acquisition.acquired_at",
    )
    must_start = _parse_durable_time(
        intent_record.get("must_start_by"),
        field="must_start_by",
    )
    if not (
        intent_at <= baseline_at <= ready_at <= acquired_at <= now_exact < must_start
    ):
        raise ProductionGenerationAuthorityError("H.1d chronology mismatch")
    return {
        "now_exact": now_exact,
        "now_text": now_text,
        "descriptor_artifact": descriptor_artifact,
        "descriptor": descriptor_record,
        "descriptor_file_sha256": descriptor_file_sha,
        "intent_artifact": intent_artifact,
        "intent": intent_record,
        "intent_file_sha256": intent_file_sha,
        "approval_artifact": approval_artifact,
        "approval": approval_record,
        "approval_file_sha256": approval_file_sha,
        "baseline_artifact": baseline_artifact,
        "baseline": baseline_record,
        "baseline_file_sha256": baseline_file_sha,
        "ready_artifact": ready_artifact,
        "ready": ready_record,
        "ready_file_sha256": ready_file_sha,
        "acquisition_artifact": acquisition_artifact,
        "acquisition": acquisition_record,
        "acquisition_file_sha256": acquisition_file_sha,
        "snapshot_artifact": snapshot_artifact,
        "snapshot": snapshot_record,
        "snapshot_file_sha256": snapshot_file_sha,
    }


def _validate_budget(
    *,
    approved_seconds: int,
    approved_cost: float,
    hourly_cost: float,
    consumed_seconds: int,
    consumed_cost: float,
    remaining_seconds: int,
    remaining_cost: float,
    open_allocations: int,
    label: str,
) -> None:
    if (
        approved_seconds != _APPROVED_SECONDS
        or approved_cost != _APPROVED_COST
        or hourly_cost != _HOURLY_COST
        or consumed_seconds < 0
        or remaining_seconds < 0
        or consumed_seconds + remaining_seconds != approved_seconds
        or consumed_cost < 0.0
        or remaining_cost < 0.0
        or Decimal(str(consumed_cost)) + Decimal(str(remaining_cost))
        != Decimal(str(approved_cost))
        or open_allocations != 0
    ):
        raise ProductionGenerationAuthorityError(f"{label} budget is inconsistent")


def _validate_claim_intrinsic(value: Mapping[str, object]) -> dict[str, object]:
    record = _exact_dict(value, _CLAIM_FIELDS, label="generation claim")
    for field in _CLAIM_INT_FIELDS:
        if type(record[field]) is not int:
            raise ProductionGenerationAuthorityError(
                f"generation claim {field} must be an exact integer"
            )
    for field in _CLAIM_FLOAT_FIELDS:
        _exact_float(record[field], field=f"generation claim {field}")
    for field in _CLAIM_STRING_FIELDS:
        _exact_string(record[field], field=f"generation claim {field}")
    generation_number = _generation(record["generation"])
    if record["generation_text"] != f"{generation_number:08d}":
        raise ProductionGenerationAuthorityError("generation claim text mismatch")
    if (
        record["schema_version"] != 1
        or record["record_type"] != "glm52_sky_production_generation_claim_v1"
        or record["managed_mode"] != _MODE
        or record["account_id"] != _ACCOUNT_ID
        or record["region"] != _REGION
    ):
        raise ProductionGenerationAuthorityError("generation claim discriminant mismatch")
    run = _run_id(record["run_id"])
    _bucket(record["bucket"])
    _attempt(record["submit_attempt_id"])
    if record["sky_job_name"] != expected_sky_job_name(
        run_id=run,
        managed_mode="production",
    ):
        raise ProductionGenerationAuthorityError("generation claim Sky name mismatch")
    digest_fields = {
        field
        for field in _CLAIM_FIELDS
        if field.endswith("_sha256") and field not in _CLAIM_OPTIONAL_FIELDS
    }
    for field in digest_fields:
        _digest(record[field], field=field)
    key_fields = {
        field
        for field in _CLAIM_FIELDS
        if field.endswith("_key") and field not in _CLAIM_OPTIONAL_FIELDS
    }
    for field in key_fields:
        _safe_key(record[field], field=field)
    version_fields = {
        field
        for field in _CLAIM_FIELDS
        if field.endswith("_version_id") and field not in _CLAIM_OPTIONAL_FIELDS
    }
    for field in version_fields:
        _version(record[field], field=field)
    previous = record["previous_generation"]
    if generation_number == 1:
        if previous != 0 or any(record[field] is not None for field in _CLAIM_OPTIONAL_FIELDS):
            raise ProductionGenerationAuthorityError(
                "generation one previous-terminal state mismatch"
            )
    else:
        if previous != generation_number - 1:
            raise ProductionGenerationAuthorityError("previous generation mismatch")
        for field in _CLAIM_OPTIONAL_FIELDS:
            if type(record[field]) is not str:
                raise ProductionGenerationAuthorityError(
                    f"{field} must be an exact string after generation one"
                )
        _safe_key(record["previous_generation_terminal_key"], field="previous_generation_terminal_key")
        _digest(
            record["previous_generation_terminal_file_sha256"],
            field="previous_generation_terminal_file_sha256",
        )
        _digest(
            record["previous_generation_terminal_body_sha256"],
            field="previous_generation_terminal_body_sha256",
        )
        _version(
            record["previous_generation_terminal_version_id"],
            field="previous_generation_terminal_version_id",
        )
        if record["previous_generation_terminal_key"] != production_generation_terminal_s3_key(
            run_id=run,
            generation=previous,
        ):
            raise ProductionGenerationAuthorityError(
                "previous terminal key mismatch"
            )
    if record["approval_key"] != _approval_key(run, record["approval_file_sha256"]):
        raise ProductionGenerationAuthorityError("generation claim approval key mismatch")
    if record["gpu_spend_snapshot_key"] != _snapshot_key(
        run,
        record["gpu_spend_snapshot_body_sha256"],
    ):
        raise ProductionGenerationAuthorityError("generation claim snapshot key mismatch")
    expected_intent_key = production_submission_intent_s3_key(
        run_id=run,
        intent_body_sha256=record["intent_body_sha256"],
    )
    if record["intent_key"] != expected_intent_key:
        raise ProductionGenerationAuthorityError("generation claim intent key mismatch")
    if record["controller_baseline_key"] != production_controller_baseline_s3_key(
        run_id=run,
        baseline_body_sha256=record["controller_baseline_body_sha256"],
    ):
        raise ProductionGenerationAuthorityError("generation claim baseline key mismatch")
    if record["must_start_control_plane_ready_key"] != production_must_start_control_plane_ready_s3_key(
        run_id=run,
        intent_body_sha256=record["intent_body_sha256"],
        control_plane_ready_body_sha256=record[
            "must_start_control_plane_ready_body_sha256"
        ],
    ):
        raise ProductionGenerationAuthorityError("generation claim readiness key mismatch")
    if record["submission_acquisition_key"] != production_submission_acquired_s3_key(
        run_id=run,
        descriptor_file_sha256=record["descriptor_file_sha256"],
    ):
        raise ProductionGenerationAuthorityError("generation claim acquisition key mismatch")
    expected_identity = _sha(
        _canonical(
            {
                "account_id": _ACCOUNT_ID,
                "campaign_identity_sha256": record["campaign_identity_sha256"],
                "generation": generation_number,
                "intent_body_sha256": record["intent_body_sha256"],
                "managed_mode": _MODE,
                "run_id": run,
                "submit_attempt_id": record["submit_attempt_id"],
            }
        )
    )
    if record["sky_job_identity_sha256"] != expected_identity:
        raise ProductionGenerationAuthorityError("Sky correlation digest mismatch")
    must_start = _parse_durable_time(record["must_start_by"], field="must_start_by")
    claim_created = _parse_durable_time(
        record["claim_created_at"],
        field="claim_created_at",
    )
    if claim_created >= must_start:
        raise ProductionGenerationAuthorityError("claim is not before deadline")
    _validate_budget(
        approved_seconds=record["approved_gpu_runtime_seconds"],
        approved_cost=record["approved_gpu_cost_usd"],
        hourly_cost=record["hourly_cost_usd"],
        consumed_seconds=record["consumed_gpu_seconds"],
        consumed_cost=record["consumed_gpu_cost_usd"],
        remaining_seconds=record["remaining_gpu_seconds"],
        remaining_cost=record["remaining_gpu_cost_usd"],
        open_allocations=record["open_allocation_count"],
        label="generation claim",
    )
    if record["generation_claim_body_sha256"] != _body_sha(
        record,
        "generation_claim_body_sha256",
    ):
        raise ProductionGenerationAuthorityError("generation claim body SHA mismatch")
    return dict(record)


def _validate_decision_intrinsic(
    value: Mapping[str, object],
) -> dict[str, object]:
    record = _exact_dict(value, _DECISION_FIELDS, label="start decision")
    for field in _DECISION_INT_FIELDS:
        if type(record[field]) is not int:
            raise ProductionGenerationAuthorityError(
                f"start decision {field} must be an exact integer"
            )
    for field in _DECISION_STRING_FIELDS:
        _exact_string(record[field], field=f"start decision {field}")
    generation_number = _generation(record["generation"])
    if (
        record["schema_version"] != 1
        or record["record_type"]
        != "glm52_sky_production_generation_start_decision_v1"
        or record["managed_mode"] != _MODE
        or record["account_id"] != _ACCOUNT_ID
        or record["region"] != _REGION
        or record["generation_text"] != f"{generation_number:08d}"
        or record["decision"] not in {"launch-once", "expire-unstarted"}
    ):
        raise ProductionGenerationAuthorityError("start decision discriminant mismatch")
    run = _run_id(record["run_id"])
    _bucket(record["bucket"])
    _attempt(record["submit_attempt_id"])
    for field in {
        "campaign_identity_sha256",
        "sky_job_identity_sha256",
        "generation_claim_file_sha256",
        "generation_claim_body_sha256",
        "start_decision_body_sha256",
    }:
        _digest(record[field], field=field)
    _version(record["generation_claim_version_id"], field="generation_claim_version_id")
    if record["generation_claim_key"] != production_generation_claim_s3_key(
        run_id=run,
        generation=generation_number,
    ):
        raise ProductionGenerationAuthorityError("decision claim key mismatch")
    must_start = _parse_durable_time(record["must_start_by"], field="must_start_by")
    decided = _parse_durable_time(record["decided_at"], field="decided_at")
    if (
        record["decision"] == "launch-once"
        and decided >= must_start
    ) or (
        record["decision"] == "expire-unstarted"
        and decided < must_start
    ):
        raise ProductionGenerationAuthorityError("start decision deadline mismatch")
    if record["start_decision_body_sha256"] != _body_sha(
        record,
        "start_decision_body_sha256",
    ):
        raise ProductionGenerationAuthorityError("start decision body SHA mismatch")
    return dict(record)


def _validate_terminal_intrinsic(
    value: Mapping[str, object],
) -> dict[str, object]:
    record = _exact_dict(value, _TERMINAL_FIELDS, label="generation terminal")
    for field in _TERMINAL_INT_FIELDS:
        if type(record[field]) is not int:
            raise ProductionGenerationAuthorityError(
                f"generation terminal {field} must be an exact integer"
            )
    for field in _TERMINAL_FLOAT_FIELDS:
        _exact_float(record[field], field=f"generation terminal {field}")
    for field in _TERMINAL_STRING_FIELDS:
        _exact_string(record[field], field=f"generation terminal {field}")
    generation_number = _generation(record["generation"])
    if (
        record["schema_version"] != 1
        or record["record_type"] != "glm52_sky_production_generation_terminal_v1"
        or record["managed_mode"] != _MODE
        or record["account_id"] != _ACCOUNT_ID
        or record["region"] != _REGION
        or record["generation_text"] != f"{generation_number:08d}"
        or record["terminal_outcome"] != "expired-unstarted"
    ):
        raise ProductionGenerationAuthorityError("generation terminal discriminant mismatch")
    run = _run_id(record["run_id"])
    _bucket(record["bucket"])
    _attempt(record["submit_attempt_id"])
    for field in {
        field for field in _TERMINAL_FIELDS if field.endswith("_sha256")
    }:
        _digest(record[field], field=field)
    for field in {
        "generation_claim_version_id",
        "start_decision_version_id",
        "final_gpu_spend_snapshot_version_id",
    }:
        _version(record[field], field=field)
    if record["generation_claim_key"] != production_generation_claim_s3_key(
        run_id=run,
        generation=generation_number,
    ) or record["start_decision_key"] != production_generation_start_decision_s3_key(
        run_id=run,
        generation=generation_number,
    ):
        raise ProductionGenerationAuthorityError("terminal generation key mismatch")
    if record["final_gpu_spend_snapshot_key"] != _snapshot_key(
        run,
        record["final_gpu_spend_snapshot_body_sha256"],
    ):
        raise ProductionGenerationAuthorityError("terminal snapshot key mismatch")
    must_start = _parse_durable_time(record["must_start_by"], field="must_start_by")
    terminal_at = _parse_durable_time(record["terminal_at"], field="terminal_at")
    if terminal_at < must_start:
        raise ProductionGenerationAuthorityError("terminal precedes deadline")
    _validate_budget(
        approved_seconds=record["final_approved_gpu_runtime_seconds"],
        approved_cost=record["final_approved_gpu_cost_usd"],
        hourly_cost=record["final_hourly_cost_usd"],
        consumed_seconds=record["final_consumed_gpu_seconds"],
        consumed_cost=record["final_consumed_gpu_cost_usd"],
        remaining_seconds=record["final_remaining_gpu_seconds"],
        remaining_cost=record["final_remaining_gpu_cost_usd"],
        open_allocations=record["final_open_allocation_count"],
        label="generation terminal",
    )
    if record["generation_terminal_body_sha256"] != _body_sha(
        record,
        "generation_terminal_body_sha256",
    ):
        raise ProductionGenerationAuthorityError("generation terminal body SHA mismatch")
    return dict(record)


def _claim_edges_match(
    claim: Mapping[str, object],
    authority: Mapping[str, object],
    *,
    require_snapshot: bool,
) -> None:
    expected = {
        "account_id": _ACCOUNT_ID,
        "region": _REGION,
        "bucket": authority["intent"]["bucket"],
        "run_id": authority["intent"]["run_id"],
        "campaign_identity_sha256": authority["intent"][
            "campaign_identity_sha256"
        ],
        "descriptor_key": authority["descriptor_artifact"].key,
        "descriptor_file_sha256": authority["descriptor_file_sha256"],
        "descriptor_body_sha256": authority["descriptor"]["descriptor_body_sha256"],
        "descriptor_version_id": authority["descriptor_artifact"].version_id,
        "intent_key": authority["intent_artifact"].key,
        "intent_file_sha256": authority["intent_file_sha256"],
        "intent_body_sha256": authority["intent"]["intent_body_sha256"],
        "intent_version_id": authority["intent_artifact"].version_id,
        "controller_baseline_key": authority["baseline_artifact"].key,
        "controller_baseline_file_sha256": authority["baseline_file_sha256"],
        "controller_baseline_body_sha256": authority["baseline"][
            "baseline_body_sha256"
        ],
        "controller_baseline_version_id": authority["baseline_artifact"].version_id,
        "must_start_control_plane_ready_key": authority["ready_artifact"].key,
        "must_start_control_plane_ready_file_sha256": authority["ready_file_sha256"],
        "must_start_control_plane_ready_body_sha256": authority["ready"][
            "control_plane_ready_body_sha256"
        ],
        "must_start_control_plane_ready_version_id": authority[
            "ready_artifact"
        ].version_id,
        "submission_acquisition_key": authority["acquisition_artifact"].key,
        "submission_acquisition_file_sha256": authority[
            "acquisition_file_sha256"
        ],
        "submission_acquisition_body_sha256": authority["acquisition"][
            "acquisition_body_sha256"
        ],
        "submission_acquisition_version_id": authority[
            "acquisition_artifact"
        ].version_id,
        "approval_key": authority["approval_artifact"].key,
        "approval_file_sha256": authority["approval_file_sha256"],
        "approval_body_sha256": authority["approval"]["approval_body_sha256"],
        "approval_version_id": authority["approval_artifact"].version_id,
        "gpu_spend_snapshot_key": authority["intent"]["gpu_spend_snapshot_key"],
        "gpu_spend_snapshot_file_sha256": authority["intent"][
            "gpu_spend_snapshot_file_sha256"
        ],
        "gpu_spend_snapshot_body_sha256": authority["intent"][
            "gpu_spend_snapshot_body_sha256"
        ],
        "gpu_spend_snapshot_version_id": authority["intent"][
            "gpu_spend_snapshot_version_id"
        ],
        "gpu_spend_ledger_genesis_sha256": authority["intent"][
            "gpu_spend_ledger_genesis_sha256"
        ],
        "gpu_spend_ledger_tip_record_sha256": authority["intent"][
            "gpu_spend_ledger_tip_record_sha256"
        ],
        "ec2_allocation_history_sha256": authority["intent"][
            "ec2_allocation_history_sha256"
        ],
        "approved_gpu_runtime_seconds": authority["intent"][
            "approved_gpu_runtime_seconds"
        ],
        "approved_gpu_cost_usd": authority["intent"]["approved_gpu_cost_usd"],
        "hourly_cost_usd": authority["intent"]["hourly_cost_usd"],
        "consumed_gpu_seconds": authority["intent"]["consumed_gpu_seconds"],
        "consumed_gpu_cost_usd": authority["intent"]["consumed_gpu_cost_usd"],
        "remaining_gpu_seconds": authority["intent"]["remaining_gpu_seconds"],
        "remaining_gpu_cost_usd": authority["intent"]["remaining_gpu_cost_usd"],
        "open_allocation_count": authority["intent"]["open_allocation_count"],
        "must_start_by": authority["intent"]["must_start_by"],
    }
    if require_snapshot:
        expected.update(
            {
                "gpu_spend_snapshot_key": authority["snapshot_artifact"].key,
                "gpu_spend_snapshot_file_sha256": authority["snapshot_file_sha256"],
                "gpu_spend_snapshot_body_sha256": authority["snapshot"][
                    "snapshot_body_sha256"
                ],
                "gpu_spend_snapshot_version_id": authority[
                    "snapshot_artifact"
                ].version_id,
                "gpu_spend_ledger_genesis_sha256": authority["snapshot"][
                    "gpu_spend_ledger_genesis_sha256"
                ],
                "gpu_spend_ledger_record_count": authority["snapshot"][
                    "gpu_spend_ledger_record_count"
                ],
                "gpu_spend_ledger_tip_record_sha256": authority["snapshot"][
                    "gpu_spend_ledger_tip_record_sha256"
                ],
                "ec2_allocation_history_sha256": authority["snapshot"][
                    "ec2_allocation_history_sha256"
                ],
                "approved_gpu_runtime_seconds": authority["snapshot"][
                    "approved_gpu_runtime_seconds"
                ],
                "approved_gpu_cost_usd": authority["snapshot"][
                    "approved_gpu_cost_usd"
                ],
                "hourly_cost_usd": authority["snapshot"]["hourly_cost_usd"],
                "consumed_gpu_seconds": authority["snapshot"][
                    "consumed_gpu_seconds"
                ],
                "consumed_gpu_cost_usd": authority["snapshot"][
                    "consumed_gpu_cost_usd"
                ],
                "remaining_gpu_seconds": authority["snapshot"][
                    "remaining_gpu_seconds"
                ],
                "remaining_gpu_cost_usd": authority["snapshot"][
                    "remaining_gpu_cost_usd"
                ],
                "open_allocation_count": authority["snapshot"][
                    "open_allocation_count"
                ],
            }
        )
    if any(claim.get(field) != expected_value for field, expected_value in expected.items()):
        raise ProductionGenerationAuthorityError("claim source lineage mismatch")


def _terminal_artifact_record(
    artifact: object,
) -> tuple[VersionedJsonArtifact, dict[str, object], str]:
    exact, record, file_sha = _artifact(artifact, label="previous_generation_terminal")
    validated = _validate_terminal_intrinsic(record)
    expected_key = production_generation_terminal_s3_key(
        run_id=validated["run_id"],
        generation=validated["generation"],
    )
    if exact.key != expected_key:
        raise ProductionGenerationAuthorityError("previous terminal key mismatch")
    return exact, validated, file_sha


def _projection(
    *,
    record_type: str,
    key: str,
    raw: bytes,
    version_id: str,
    file_sha256: str,
    body_sha256: str,
) -> dict[str, object]:
    return {
        "record_type": record_type,
        "key": key,
        "raw": memoryview(raw).tobytes(),
        "version_id": version_id,
        "file_sha256": file_sha256,
        "body_sha256": body_sha256,
    }


def _validate_inventory(
    entries: object,
    *,
    run_id: object,
) -> dict[str, object]:
    run = _run_id(run_id)
    if type(entries) is not list:
        raise ProductionGenerationAuthorityError("inventory must be an exact list")
    for entry in entries:
        if type(entry) is not GenerationInventoryEntry:
            raise ProductionGenerationAuthorityError(
                "inventory members must be exact GenerationInventoryEntry values"
            )
        if (
            type(entry.key) is not str
            or type(entry.version_id) is not str
            or type(entry.is_latest) is not bool
            or type(entry.is_delete_marker) is not bool
        ):
            raise ProductionGenerationAuthorityError("inventory member type mismatch")
        _version(entry.version_id, field="inventory.version_id")
        if entry.is_delete_marker:
            if entry.raw is not None:
                raise ProductionGenerationAuthorityError(
                    "delete marker raw must be exact None"
                )
        elif type(entry.raw) is not bytes:
            raise ProductionGenerationAuthorityError(
                "non-delete inventory raw must be exact bytes"
            )
    ordered = sorted(
        entries,
        key=lambda item: (item.key, item.version_id, item.is_delete_marker),
    )
    if entries != ordered:
        raise ProductionGenerationAuthorityError("inventory is not canonically sorted")
    tuples = [
        (entry.key, entry.version_id, entry.is_delete_marker) for entry in entries
    ]
    if len(tuples) != len(set(tuples)):
        raise ProductionGenerationAuthorityError("duplicate inventory tuple")
    if not entries:
        return {
            "record_type": "glm52_sky_production_generation_inventory_validation_v1",
            "run_id": run,
            "generation_count": 0,
            "highest_generation": None,
            "open_generation": None,
            "generations": [],
        }
    key_pattern = re.compile(
        rf"campaigns/{re.escape(run)}/submissions/production/generations/"
        r"(?P<generation>[0-9]{8})/"
        r"(?P<name>GENERATION_CLAIM\.json|START_DECISION\.json|"
        r"GENERATION_TERMINAL\.json)"
    )
    key_entries: dict[str, GenerationInventoryEntry] = {}
    grouped: dict[int, dict[str, tuple[GenerationInventoryEntry, dict[str, object], str]]] = {}
    for entry in entries:
        if entry.is_delete_marker:
            raise ProductionGenerationAuthorityError("inventory contains a delete marker")
        if entry.key in key_entries:
            raise ProductionGenerationAuthorityError(
                "inventory contains multiple versions for one key"
            )
        key_entries[entry.key] = entry
        if entry.is_latest is not True:
            raise ProductionGenerationAuthorityError(
                "sole inventory version must be latest"
            )
        match = key_pattern.fullmatch(entry.key)
        if match is None:
            raise ProductionGenerationAuthorityError("inventory key is noncanonical")
        generation_number = int(match.group("generation"))
        if (
            generation_number < 1
            or generation_number > _MAX_GENERATION
            or match.group("generation") != f"{generation_number:08d}"
        ):
            raise ProductionGenerationAuthorityError("inventory generation is invalid")
        name = match.group("name")
        raw = entry.raw
        if type(raw) is not bytes:
            raise ProductionGenerationAuthorityError("inventory raw type mismatch")
        record = _parse_raw(raw, label="inventory record")
        if name == "GENERATION_CLAIM.json":
            validated = _validate_claim_intrinsic(record)
            kind = "claim"
            digest_field = "generation_claim_body_sha256"
        elif name == "START_DECISION.json":
            validated = _validate_decision_intrinsic(record)
            kind = "start_decision"
            digest_field = "start_decision_body_sha256"
        else:
            validated = _validate_terminal_intrinsic(record)
            kind = "terminal"
            digest_field = "generation_terminal_body_sha256"
        if (
            validated["generation"] != generation_number
            or validated["run_id"] != run
        ):
            raise ProductionGenerationAuthorityError("inventory record identity mismatch")
        grouped.setdefault(generation_number, {})
        if kind in grouped[generation_number]:
            raise ProductionGenerationAuthorityError("inventory record fork")
        grouped[generation_number][kind] = (
            entry,
            validated,
            _sha(raw),
        )
        if validated[digest_field] != _body_sha(validated, digest_field):
            raise ProductionGenerationAuthorityError("inventory self-hash mismatch")
    generation_numbers = sorted(grouped)
    if generation_numbers != list(range(1, generation_numbers[-1] + 1)):
        raise ProductionGenerationAuthorityError("inventory generation hole")
    normalized_generations: list[dict[str, object]] = []
    open_generation: Optional[int] = None
    previous_terminal: Optional[tuple[GenerationInventoryEntry, dict[str, object], str]] = None
    previous_claim: Optional[dict[str, object]] = None
    for generation_number in generation_numbers:
        records = grouped[generation_number]
        if "claim" not in records:
            raise ProductionGenerationAuthorityError("decision or terminal lacks claim")
        claim_entry, claim, claim_file_sha = records["claim"]
        decision_tuple = records.get("start_decision")
        terminal_tuple = records.get("terminal")
        if terminal_tuple is not None and decision_tuple is None:
            raise ProductionGenerationAuthorityError("terminal lacks decision")
        if decision_tuple is not None:
            decision_entry, decision, decision_file_sha = decision_tuple
            edge_expectations = {
                "account_id": claim["account_id"],
                "region": claim["region"],
                "bucket": claim["bucket"],
                "run_id": claim["run_id"],
                "campaign_identity_sha256": claim["campaign_identity_sha256"],
                "generation": claim["generation"],
                "generation_text": claim["generation_text"],
                "submit_attempt_id": claim["submit_attempt_id"],
                "sky_job_name": claim["sky_job_name"],
                "sky_job_identity_sha256": claim["sky_job_identity_sha256"],
                "generation_claim_key": claim_entry.key,
                "generation_claim_file_sha256": claim_file_sha,
                "generation_claim_body_sha256": claim[
                    "generation_claim_body_sha256"
                ],
                "generation_claim_version_id": claim_entry.version_id,
                "must_start_by": claim["must_start_by"],
            }
            if any(
                decision.get(field) != expected
                for field, expected in edge_expectations.items()
            ):
                raise ProductionGenerationAuthorityError(
                    "inventory claim/decision lineage mismatch"
                )
            claim_created_at = _parse_durable_time(
                claim["claim_created_at"],
                field="inventory claim_created_at",
            )
            decided_at = _parse_durable_time(
                decision["decided_at"],
                field="inventory decided_at",
            )
            if decided_at < claim_created_at:
                raise ProductionGenerationAuthorityError(
                    "inventory decision precedes claim"
                )
        else:
            decision_entry = None
            decision = None
            decision_file_sha = None
            decided_at = None
        if terminal_tuple is not None:
            terminal_entry, terminal, terminal_file_sha = terminal_tuple
            if decision is None or decision.get("decision") != "expire-unstarted":
                raise ProductionGenerationAuthorityError(
                    "only expired decisions may have v1 terminals"
                )
            terminal_expectations = {
                "account_id": claim["account_id"],
                "region": claim["region"],
                "bucket": claim["bucket"],
                "run_id": claim["run_id"],
                "campaign_identity_sha256": claim["campaign_identity_sha256"],
                "generation": claim["generation"],
                "generation_text": claim["generation_text"],
                "submit_attempt_id": claim["submit_attempt_id"],
                "sky_job_name": claim["sky_job_name"],
                "sky_job_identity_sha256": claim["sky_job_identity_sha256"],
                "generation_claim_key": claim_entry.key,
                "generation_claim_file_sha256": claim_file_sha,
                "generation_claim_body_sha256": claim[
                    "generation_claim_body_sha256"
                ],
                "generation_claim_version_id": claim_entry.version_id,
                "start_decision_key": decision_entry.key,
                "start_decision_file_sha256": decision_file_sha,
                "start_decision_body_sha256": decision[
                    "start_decision_body_sha256"
                ],
                "start_decision_version_id": decision_entry.version_id,
                "must_start_by": claim["must_start_by"],
            }
            if any(
                terminal.get(field) != expected
                for field, expected in terminal_expectations.items()
            ):
                raise ProductionGenerationAuthorityError(
                    "inventory terminal lineage mismatch"
                )
            terminal_at = _parse_durable_time(
                terminal["terminal_at"],
                field="inventory terminal_at",
            )
            if decided_at is None or terminal_at < decided_at:
                raise ProductionGenerationAuthorityError(
                    "inventory terminal precedes decision"
                )
            terminal_continuity = {
                "final_gpu_spend_ledger_genesis_sha256": claim[
                    "gpu_spend_ledger_genesis_sha256"
                ],
                "final_gpu_spend_ledger_record_count": claim[
                    "gpu_spend_ledger_record_count"
                ],
                "final_gpu_spend_ledger_tip_record_sha256": claim[
                    "gpu_spend_ledger_tip_record_sha256"
                ],
                "final_ec2_allocation_history_sha256": claim[
                    "ec2_allocation_history_sha256"
                ],
                "final_approved_gpu_runtime_seconds": claim[
                    "approved_gpu_runtime_seconds"
                ],
                "final_approved_gpu_cost_usd": claim[
                    "approved_gpu_cost_usd"
                ],
                "final_hourly_cost_usd": claim["hourly_cost_usd"],
                "final_consumed_gpu_seconds": claim["consumed_gpu_seconds"],
                "final_consumed_gpu_cost_usd": claim[
                    "consumed_gpu_cost_usd"
                ],
                "final_remaining_gpu_seconds": claim[
                    "remaining_gpu_seconds"
                ],
                "final_remaining_gpu_cost_usd": claim[
                    "remaining_gpu_cost_usd"
                ],
                "final_open_allocation_count": claim[
                    "open_allocation_count"
                ],
            }
            if any(
                terminal.get(field) != expected
                for field, expected in terminal_continuity.items()
            ):
                raise ProductionGenerationAuthorityError(
                    "inventory terminal budget continuity mismatch"
                )
        else:
            terminal_entry = None
            terminal = None
            terminal_file_sha = None
        if previous_terminal is not None and previous_claim is not None:
            prior_entry, prior_terminal_record, prior_file_sha = previous_terminal
            cross = {
                "previous_generation": generation_number - 1,
                "previous_generation_terminal_key": prior_entry.key,
                "previous_generation_terminal_file_sha256": prior_file_sha,
                "previous_generation_terminal_body_sha256": prior_terminal_record[
                    "generation_terminal_body_sha256"
                ],
                "previous_generation_terminal_version_id": prior_entry.version_id,
                "account_id": previous_claim["account_id"],
                "region": previous_claim["region"],
                "bucket": previous_claim["bucket"],
                "run_id": previous_claim["run_id"],
                "campaign_identity_sha256": previous_claim[
                    "campaign_identity_sha256"
                ],
                "approval_key": previous_claim["approval_key"],
                "approval_file_sha256": previous_claim["approval_file_sha256"],
                "approval_body_sha256": previous_claim["approval_body_sha256"],
                "approval_version_id": previous_claim["approval_version_id"],
                "approved_gpu_runtime_seconds": prior_terminal_record[
                    "final_approved_gpu_runtime_seconds"
                ],
                "approved_gpu_cost_usd": prior_terminal_record[
                    "final_approved_gpu_cost_usd"
                ],
                "hourly_cost_usd": prior_terminal_record["final_hourly_cost_usd"],
                "gpu_spend_ledger_genesis_sha256": prior_terminal_record[
                    "final_gpu_spend_ledger_genesis_sha256"
                ],
                "gpu_spend_ledger_record_count": prior_terminal_record[
                    "final_gpu_spend_ledger_record_count"
                ],
                "gpu_spend_ledger_tip_record_sha256": prior_terminal_record[
                    "final_gpu_spend_ledger_tip_record_sha256"
                ],
                "ec2_allocation_history_sha256": prior_terminal_record[
                    "final_ec2_allocation_history_sha256"
                ],
                "consumed_gpu_seconds": prior_terminal_record[
                    "final_consumed_gpu_seconds"
                ],
                "consumed_gpu_cost_usd": prior_terminal_record[
                    "final_consumed_gpu_cost_usd"
                ],
                "remaining_gpu_seconds": prior_terminal_record[
                    "final_remaining_gpu_seconds"
                ],
                "remaining_gpu_cost_usd": prior_terminal_record[
                    "final_remaining_gpu_cost_usd"
                ],
                "open_allocation_count": 0,
            }
            if any(claim.get(field) != expected for field, expected in cross.items()):
                raise ProductionGenerationAuthorityError(
                    "adjacent generation continuity mismatch"
                )
        if generation_number < generation_numbers[-1] and terminal is None:
            raise ProductionGenerationAuthorityError(
                "non-highest generation must be terminal"
            )
        if terminal is None:
            if open_generation is not None:
                raise ProductionGenerationAuthorityError("multiple open generations")
            open_generation = generation_number
        normalized_generations.append(
            {
                "generation": generation_number,
                "generation_text": f"{generation_number:08d}",
                "claim": _projection(
                    record_type="glm52_sky_production_generation_claim_v1",
                    key=claim_entry.key,
                    raw=claim_entry.raw,
                    version_id=claim_entry.version_id,
                    file_sha256=claim_file_sha,
                    body_sha256=claim["generation_claim_body_sha256"],
                ),
                "start_decision": (
                    None
                    if decision is None
                    else _projection(
                        record_type=(
                            "glm52_sky_production_generation_start_decision_v1"
                        ),
                        key=decision_entry.key,
                        raw=decision_entry.raw,
                        version_id=decision_entry.version_id,
                        file_sha256=decision_file_sha,
                        body_sha256=decision["start_decision_body_sha256"],
                    )
                ),
                "terminal": (
                    None
                    if terminal is None
                    else _projection(
                        record_type="glm52_sky_production_generation_terminal_v1",
                        key=terminal_entry.key,
                        raw=terminal_entry.raw,
                        version_id=terminal_entry.version_id,
                        file_sha256=terminal_file_sha,
                        body_sha256=terminal["generation_terminal_body_sha256"],
                    )
                ),
            }
        )
        previous_terminal = terminal_tuple
        previous_claim = claim
    return {
        "record_type": "glm52_sky_production_generation_inventory_validation_v1",
        "run_id": run,
        "generation_count": len(generation_numbers),
        "highest_generation": generation_numbers[-1],
        "open_generation": open_generation,
        "generations": normalized_generations,
    }


def production_generation_claim_s3_key(
    *,
    run_id: str,
    generation: int,
) -> str:
    def operation() -> str:
        run = _run_id(run_id)
        number = _generation(generation)
        return (
            f"campaigns/{run}/submissions/production/generations/"
            f"{number:08d}/GENERATION_CLAIM.json"
        )

    return _translate(operation)


def production_generation_start_decision_s3_key(
    *,
    run_id: str,
    generation: int,
) -> str:
    def operation() -> str:
        run = _run_id(run_id)
        number = _generation(generation)
        return (
            f"campaigns/{run}/submissions/production/generations/"
            f"{number:08d}/START_DECISION.json"
        )

    return _translate(operation)


def production_generation_terminal_s3_key(
    *,
    run_id: str,
    generation: int,
) -> str:
    def operation() -> str:
        run = _run_id(run_id)
        number = _generation(generation)
        return (
            f"campaigns/{run}/submissions/production/generations/"
            f"{number:08d}/GENERATION_TERMINAL.json"
        )

    return _translate(operation)


def production_generation_claim_file_bytes(
    value: Mapping[str, object],
) -> bytes:
    return _translate(lambda: _canonical(_validate_claim_intrinsic(value)) + b"\n")


def production_generation_claim_file_sha256(
    value: Mapping[str, object],
) -> str:
    return _translate(lambda: _sha(production_generation_claim_file_bytes(value)))


def production_generation_start_decision_file_bytes(
    value: Mapping[str, object],
) -> bytes:
    return _translate(lambda: _canonical(_validate_decision_intrinsic(value)) + b"\n")


def production_generation_start_decision_file_sha256(
    value: Mapping[str, object],
) -> str:
    return _translate(
        lambda: _sha(production_generation_start_decision_file_bytes(value))
    )


def production_generation_terminal_file_bytes(
    value: Mapping[str, object],
) -> bytes:
    return _translate(lambda: _canonical(_validate_terminal_intrinsic(value)) + b"\n")


def production_generation_terminal_file_sha256(
    value: Mapping[str, object],
) -> str:
    return _translate(lambda: _sha(production_generation_terminal_file_bytes(value)))


def _build_claim(
    *,
    generation: object,
    descriptor: object,
    intent: object,
    approval: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
    submission_acquisition: object,
    gpu_spend_snapshot: object,
    generation_inventory: object,
    previous_generation_terminal: object,
    submit_attempt_id: object,
    claim_created_at: Union[datetime, str],
) -> dict[str, object]:
    generation_number = _generation(generation)
    attempt = _attempt(submit_attempt_id)
    claim_exact, claim_text = _normalize_time_input(
        claim_created_at,
        field="claim_created_at",
    )
    inventory = _validate_inventory(
        generation_inventory,
        run_id=_parse_raw(intent.raw, label="intent preflight").get("run_id")
        if type(intent) is VersionedJsonArtifact
        else "",
    )
    authority = _authenticate_h1d(
        descriptor=descriptor,
        intent=intent,
        approval=approval,
        controller_baseline=controller_baseline,
        must_start_control_plane_ready=must_start_control_plane_ready,
        submission_acquisition=submission_acquisition,
        gpu_spend_snapshot=gpu_spend_snapshot,
        now=claim_created_at,
    )
    intent_record = authority["intent"]
    snapshot = authority["snapshot"]
    if snapshot is None:
        raise ProductionGenerationAuthorityError("claim snapshot is missing")
    if claim_text >= intent_record["must_start_by"]:
        raise ProductionGenerationAuthorityError(
            "conservative claim time reaches deadline"
        )
    if snapshot["open_allocation_count"] != 0:
        raise ProductionGenerationAuthorityError("claim snapshot has open allocation")
    if (
        snapshot["approved_gpu_runtime_seconds"]
        != intent_record["approved_gpu_runtime_seconds"]
        or snapshot["approved_gpu_cost_usd"]
        != intent_record["approved_gpu_cost_usd"]
        or snapshot["hourly_cost_usd"] != intent_record["hourly_cost_usd"]
        or snapshot["consumed_gpu_seconds"] != intent_record["consumed_gpu_seconds"]
        or snapshot["consumed_gpu_cost_usd"]
        != intent_record["consumed_gpu_cost_usd"]
        or snapshot["remaining_gpu_seconds"] != intent_record["remaining_gpu_seconds"]
        or snapshot["remaining_gpu_cost_usd"]
        != intent_record["remaining_gpu_cost_usd"]
    ):
        raise ProductionGenerationAuthorityError("intent/snapshot budget mismatch")
    previous_key: Optional[str]
    previous_file: Optional[str]
    previous_body: Optional[str]
    previous_version: Optional[str]
    if generation_number == 1:
        if (
            inventory["generation_count"] != 0
            or previous_generation_terminal is not None
        ):
            raise ProductionGenerationAuthorityError(
                "generation one requires empty inventory and no terminal"
            )
        previous_generation = 0
        previous_key = None
        previous_file = None
        previous_body = None
        previous_version = None
    else:
        if (
            inventory["highest_generation"] != generation_number - 1
            or inventory["open_generation"] is not None
            or previous_generation_terminal is None
        ):
            raise ProductionGenerationAuthorityError(
                "later generation requires contiguous closed predecessor"
            )
        terminal_artifact, terminal, terminal_file_sha = _terminal_artifact_record(
            previous_generation_terminal
        )
        projection = inventory["generations"][-1]["terminal"]
        if projection is None or (
            projection["key"] != terminal_artifact.key
            or projection["raw"] != terminal_artifact.raw
            or projection["version_id"] != terminal_artifact.version_id
            or projection["file_sha256"] != terminal_file_sha
            or projection["body_sha256"]
            != terminal["generation_terminal_body_sha256"]
        ):
            raise ProductionGenerationAuthorityError(
                "previous terminal artifact does not equal inventory"
            )
        terminal_at = _parse_durable_time(
            terminal["terminal_at"],
            field="previous terminal terminal_at",
        )
        snapshot_at = _parse_durable_time(
            snapshot["observed_at"],
            field="new snapshot observed_at",
        )
        intent_at = _parse_durable_time(intent_record["intent_at"], field="intent_at")
        if not terminal_at <= snapshot_at <= intent_at <= claim_exact:
            raise ProductionGenerationAuthorityError(
                "next-generation chronology mismatch"
            )
        continuity = {
            "gpu_spend_ledger_genesis_sha256": terminal[
                "final_gpu_spend_ledger_genesis_sha256"
            ],
            "gpu_spend_ledger_record_count": terminal[
                "final_gpu_spend_ledger_record_count"
            ],
            "gpu_spend_ledger_tip_record_sha256": terminal[
                "final_gpu_spend_ledger_tip_record_sha256"
            ],
            "ec2_allocation_history_sha256": terminal[
                "final_ec2_allocation_history_sha256"
            ],
            "approved_gpu_runtime_seconds": terminal[
                "final_approved_gpu_runtime_seconds"
            ],
            "approved_gpu_cost_usd": terminal["final_approved_gpu_cost_usd"],
            "hourly_cost_usd": terminal["final_hourly_cost_usd"],
            "consumed_gpu_seconds": terminal["final_consumed_gpu_seconds"],
            "consumed_gpu_cost_usd": terminal["final_consumed_gpu_cost_usd"],
            "remaining_gpu_seconds": terminal["final_remaining_gpu_seconds"],
            "remaining_gpu_cost_usd": terminal["final_remaining_gpu_cost_usd"],
            "open_allocation_count": terminal["final_open_allocation_count"],
        }
        if any(snapshot.get(field) != expected for field, expected in continuity.items()):
            raise ProductionGenerationAuthorityError(
                "next-generation spend continuity mismatch"
            )
        previous_claim = inventory["generations"][-1]["claim"]
        previous_claim_record = _parse_raw(
            previous_claim["raw"],
            label="previous claim",
        )
        adjacent = {
            "account_id": previous_claim_record["account_id"],
            "region": previous_claim_record["region"],
            "bucket": previous_claim_record["bucket"],
            "run_id": previous_claim_record["run_id"],
            "campaign_identity_sha256": previous_claim_record[
                "campaign_identity_sha256"
            ],
            "approval_key": previous_claim_record["approval_key"],
            "approval_file_sha256": previous_claim_record["approval_file_sha256"],
            "approval_body_sha256": previous_claim_record["approval_body_sha256"],
            "approval_version_id": previous_claim_record["approval_version_id"],
        }
        current_adjacent = {
            "account_id": _ACCOUNT_ID,
            "region": _REGION,
            "bucket": intent_record["bucket"],
            "run_id": intent_record["run_id"],
            "campaign_identity_sha256": intent_record["campaign_identity_sha256"],
            "approval_key": authority["approval_artifact"].key,
            "approval_file_sha256": authority["approval_file_sha256"],
            "approval_body_sha256": authority["approval"]["approval_body_sha256"],
            "approval_version_id": authority["approval_artifact"].version_id,
        }
        if current_adjacent != adjacent:
            raise ProductionGenerationAuthorityError(
                "adjacent claim authority mismatch"
            )
        previous_generation = generation_number - 1
        previous_key = terminal_artifact.key
        previous_file = terminal_file_sha
        previous_body = terminal["generation_terminal_body_sha256"]
        previous_version = terminal_artifact.version_id
    run = intent_record["run_id"]
    correlation = {
        "account_id": _ACCOUNT_ID,
        "campaign_identity_sha256": intent_record["campaign_identity_sha256"],
        "generation": generation_number,
        "intent_body_sha256": intent_record["intent_body_sha256"],
        "managed_mode": _MODE,
        "run_id": run,
        "submit_attempt_id": attempt,
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_production_generation_claim_v1",
        "account_id": _ACCOUNT_ID,
        "region": _REGION,
        "bucket": intent_record["bucket"],
        "run_id": run,
        "managed_mode": _MODE,
        "campaign_identity_sha256": intent_record["campaign_identity_sha256"],
        "generation": generation_number,
        "generation_text": f"{generation_number:08d}",
        "submit_attempt_id": attempt,
        "sky_job_name": intent_record["sky_job_name"],
        "sky_job_identity_sha256": _sha(_canonical(correlation)),
        "descriptor_key": authority["descriptor_artifact"].key,
        "descriptor_file_sha256": authority["descriptor_file_sha256"],
        "descriptor_body_sha256": authority["descriptor"][
            "descriptor_body_sha256"
        ],
        "descriptor_version_id": authority["descriptor_artifact"].version_id,
        "intent_key": authority["intent_artifact"].key,
        "intent_file_sha256": authority["intent_file_sha256"],
        "intent_body_sha256": intent_record["intent_body_sha256"],
        "intent_version_id": authority["intent_artifact"].version_id,
        "controller_baseline_key": authority["baseline_artifact"].key,
        "controller_baseline_file_sha256": authority["baseline_file_sha256"],
        "controller_baseline_body_sha256": authority["baseline"][
            "baseline_body_sha256"
        ],
        "controller_baseline_version_id": authority["baseline_artifact"].version_id,
        "must_start_control_plane_ready_key": authority["ready_artifact"].key,
        "must_start_control_plane_ready_file_sha256": authority["ready_file_sha256"],
        "must_start_control_plane_ready_body_sha256": authority["ready"][
            "control_plane_ready_body_sha256"
        ],
        "must_start_control_plane_ready_version_id": authority[
            "ready_artifact"
        ].version_id,
        "submission_acquisition_key": authority["acquisition_artifact"].key,
        "submission_acquisition_file_sha256": authority[
            "acquisition_file_sha256"
        ],
        "submission_acquisition_body_sha256": authority["acquisition"][
            "acquisition_body_sha256"
        ],
        "submission_acquisition_version_id": authority[
            "acquisition_artifact"
        ].version_id,
        "approval_key": authority["approval_artifact"].key,
        "approval_file_sha256": authority["approval_file_sha256"],
        "approval_body_sha256": authority["approval"]["approval_body_sha256"],
        "approval_version_id": authority["approval_artifact"].version_id,
        "gpu_spend_snapshot_key": authority["snapshot_artifact"].key,
        "gpu_spend_snapshot_file_sha256": authority["snapshot_file_sha256"],
        "gpu_spend_snapshot_body_sha256": snapshot["snapshot_body_sha256"],
        "gpu_spend_snapshot_version_id": authority["snapshot_artifact"].version_id,
        "gpu_spend_ledger_genesis_sha256": snapshot[
            "gpu_spend_ledger_genesis_sha256"
        ],
        "gpu_spend_ledger_record_count": snapshot["gpu_spend_ledger_record_count"],
        "gpu_spend_ledger_tip_record_sha256": snapshot[
            "gpu_spend_ledger_tip_record_sha256"
        ],
        "ec2_allocation_history_sha256": snapshot[
            "ec2_allocation_history_sha256"
        ],
        "approved_gpu_runtime_seconds": snapshot["approved_gpu_runtime_seconds"],
        "approved_gpu_cost_usd": snapshot["approved_gpu_cost_usd"],
        "hourly_cost_usd": snapshot["hourly_cost_usd"],
        "consumed_gpu_seconds": snapshot["consumed_gpu_seconds"],
        "consumed_gpu_cost_usd": snapshot["consumed_gpu_cost_usd"],
        "remaining_gpu_seconds": snapshot["remaining_gpu_seconds"],
        "remaining_gpu_cost_usd": snapshot["remaining_gpu_cost_usd"],
        "open_allocation_count": snapshot["open_allocation_count"],
        "previous_generation": previous_generation,
        "previous_generation_terminal_key": previous_key,
        "previous_generation_terminal_file_sha256": previous_file,
        "previous_generation_terminal_body_sha256": previous_body,
        "previous_generation_terminal_version_id": previous_version,
        "must_start_by": intent_record["must_start_by"],
        "claim_created_at": claim_text,
    }
    result = {
        **body,
        "generation_claim_body_sha256": _sha(_canonical(body)),
    }
    return _validate_claim_intrinsic(result)


def build_production_generation_claim(
    *,
    generation: int,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
    generation_inventory: list[GenerationInventoryEntry],
    previous_generation_terminal: Optional[VersionedJsonArtifact],
    submit_attempt_id: str,
    claim_created_at: Union[datetime, str],
) -> dict[str, object]:
    return _translate(
        lambda: _build_claim(
            generation=generation,
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            gpu_spend_snapshot=gpu_spend_snapshot,
            generation_inventory=generation_inventory,
            previous_generation_terminal=previous_generation_terminal,
            submit_attempt_id=submit_attempt_id,
            claim_created_at=claim_created_at,
        )
    )


def validate_production_generation_claim(
    value: Mapping[str, object],
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
    generation_inventory: list[GenerationInventoryEntry],
    previous_generation_terminal: Optional[VersionedJsonArtifact],
) -> dict[str, object]:
    def operation() -> dict[str, object]:
        record = _validate_claim_intrinsic(value)
        expected = _build_claim(
            generation=record["generation"],
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            gpu_spend_snapshot=gpu_spend_snapshot,
            generation_inventory=generation_inventory,
            previous_generation_terminal=previous_generation_terminal,
            submit_attempt_id=record["submit_attempt_id"],
            claim_created_at=record["claim_created_at"],
        )
        if record != expected:
            raise ProductionGenerationAuthorityError("generation claim mismatch")
        return dict(record)

    return _translate(operation)


def _authenticate_claim_for_decision(
    *,
    generation_claim: object,
    descriptor: object,
    intent: object,
    approval: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
    submission_acquisition: object,
    authority_time: Union[datetime, str],
) -> tuple[VersionedJsonArtifact, dict[str, object], str, dict[str, object]]:
    claim_artifact, claim, claim_file_sha = _artifact(
        generation_claim,
        label="generation_claim",
    )
    claim = _validate_claim_intrinsic(claim)
    if claim_artifact.key != production_generation_claim_s3_key(
        run_id=claim["run_id"],
        generation=claim["generation"],
    ):
        raise ProductionGenerationAuthorityError("generation claim key mismatch")
    authority = _authenticate_h1d(
        descriptor=descriptor,
        intent=intent,
        approval=approval,
        controller_baseline=controller_baseline,
        must_start_control_plane_ready=must_start_control_plane_ready,
        submission_acquisition=submission_acquisition,
        now=authority_time,
    )
    _claim_edges_match(claim, authority, require_snapshot=False)
    return claim_artifact, claim, claim_file_sha, authority


def _build_start_decision(
    *,
    generation_claim: object,
    descriptor: object,
    intent: object,
    approval: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
    submission_acquisition: object,
    decision: object,
    decided_at: Union[datetime, str],
) -> dict[str, object]:
    decision_text = _exact_string(decision, field="decision")
    if decision_text not in {"launch-once", "expire-unstarted"}:
        raise ProductionGenerationAuthorityError("decision is unsupported")
    decided_exact, decided_text = _normalize_time_input(
        decided_at,
        field="decided_at",
    )
    preliminary_artifact, preliminary_claim, _ = _artifact(
        generation_claim,
        label="generation_claim preliminary",
    )
    preliminary_claim = _validate_claim_intrinsic(preliminary_claim)
    authority_time: Union[datetime, str]
    if decision_text == "launch-once":
        authority_time = decided_at
    else:
        authority_time = preliminary_claim["claim_created_at"]
    claim_artifact, claim, claim_file_sha, _ = _authenticate_claim_for_decision(
        generation_claim=preliminary_artifact,
        descriptor=descriptor,
        intent=intent,
        approval=approval,
        controller_baseline=controller_baseline,
        must_start_control_plane_ready=must_start_control_plane_ready,
        submission_acquisition=submission_acquisition,
        authority_time=authority_time,
    )
    claim_created = _parse_durable_time(
        claim["claim_created_at"],
        field="claim_created_at",
    )
    must_start = _parse_durable_time(claim["must_start_by"], field="must_start_by")
    if decided_exact < claim_created:
        raise ProductionGenerationAuthorityError("decision precedes claim")
    if decision_text == "launch-once":
        if decided_exact >= must_start or decided_text >= claim["must_start_by"]:
            raise ProductionGenerationAuthorityError(
                "launch decision reaches deadline"
            )
    elif decided_exact < must_start:
        raise ProductionGenerationAuthorityError(
            "expiry decision precedes deadline"
        )
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_production_generation_start_decision_v1",
        "account_id": claim["account_id"],
        "region": claim["region"],
        "bucket": claim["bucket"],
        "run_id": claim["run_id"],
        "managed_mode": claim["managed_mode"],
        "campaign_identity_sha256": claim["campaign_identity_sha256"],
        "generation": claim["generation"],
        "generation_text": claim["generation_text"],
        "submit_attempt_id": claim["submit_attempt_id"],
        "sky_job_name": claim["sky_job_name"],
        "sky_job_identity_sha256": claim["sky_job_identity_sha256"],
        "generation_claim_key": claim_artifact.key,
        "generation_claim_file_sha256": claim_file_sha,
        "generation_claim_body_sha256": claim["generation_claim_body_sha256"],
        "generation_claim_version_id": claim_artifact.version_id,
        "decision": decision_text,
        "must_start_by": claim["must_start_by"],
        "decided_at": decided_text,
    }
    result = {
        **body,
        "start_decision_body_sha256": _sha(_canonical(body)),
    }
    return _validate_decision_intrinsic(result)


def build_production_generation_start_decision(
    *,
    generation_claim: VersionedJsonArtifact,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    decision: Literal["launch-once", "expire-unstarted"],
    decided_at: Union[datetime, str],
) -> dict[str, object]:
    return _translate(
        lambda: _build_start_decision(
            generation_claim=generation_claim,
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            decision=decision,
            decided_at=decided_at,
        )
    )


def validate_production_generation_start_decision(
    value: Mapping[str, object],
    *,
    generation_claim: VersionedJsonArtifact,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    now: Union[datetime, str],
) -> dict[str, object]:
    def operation() -> dict[str, object]:
        record = _validate_decision_intrinsic(value)
        now_exact, _ = _normalize_time_input(now, field="now")
        decided_exact = _parse_durable_time(record["decided_at"], field="decided_at")
        if now_exact < decided_exact:
            raise ProductionGenerationAuthorityError("now precedes decision")
        expected = _build_start_decision(
            generation_claim=generation_claim,
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            decision=record["decision"],
            decided_at=record["decided_at"],
        )
        if record != expected:
            raise ProductionGenerationAuthorityError("start decision mismatch")
        if record["decision"] == "launch-once":
            claim_artifact, claim, _, authority = _authenticate_claim_for_decision(
                generation_claim=generation_claim,
                descriptor=descriptor,
                intent=intent,
                approval=approval,
                controller_baseline=controller_baseline,
                must_start_control_plane_ready=must_start_control_plane_ready,
                submission_acquisition=submission_acquisition,
                authority_time=now,
            )
            if claim_artifact.key != record["generation_claim_key"]:
                raise ProductionGenerationAuthorityError("claim key mismatch")
            _claim_edges_match(claim, authority, require_snapshot=False)
        return dict(record)

    return _translate(operation)


def _authenticate_terminal_sources(
    *,
    generation_claim: object,
    start_decision: object,
    final_gpu_spend_snapshot: object,
) -> tuple[
    VersionedJsonArtifact,
    dict[str, object],
    str,
    VersionedJsonArtifact,
    dict[str, object],
    str,
    VersionedJsonArtifact,
    dict[str, object],
    str,
]:
    claim_artifact, claim, claim_file_sha = _artifact(
        generation_claim,
        label="generation_claim",
    )
    claim = _validate_claim_intrinsic(claim)
    decision_artifact, decision, decision_file_sha = _artifact(
        start_decision,
        label="start_decision",
    )
    decision = _validate_decision_intrinsic(decision)
    if (
        claim_artifact.key
        != production_generation_claim_s3_key(
            run_id=claim["run_id"],
            generation=claim["generation"],
        )
        or decision_artifact.key
        != production_generation_start_decision_s3_key(
            run_id=claim["run_id"],
            generation=claim["generation"],
        )
    ):
        raise ProductionGenerationAuthorityError("terminal source key mismatch")
    decision_expected = {
        "account_id": claim["account_id"],
        "region": claim["region"],
        "bucket": claim["bucket"],
        "run_id": claim["run_id"],
        "campaign_identity_sha256": claim["campaign_identity_sha256"],
        "generation": claim["generation"],
        "generation_text": claim["generation_text"],
        "submit_attempt_id": claim["submit_attempt_id"],
        "sky_job_name": claim["sky_job_name"],
        "sky_job_identity_sha256": claim["sky_job_identity_sha256"],
        "generation_claim_key": claim_artifact.key,
        "generation_claim_file_sha256": claim_file_sha,
        "generation_claim_body_sha256": claim["generation_claim_body_sha256"],
        "generation_claim_version_id": claim_artifact.version_id,
        "must_start_by": claim["must_start_by"],
        "decision": "expire-unstarted",
    }
    if any(decision.get(field) != expected for field, expected in decision_expected.items()):
        raise ProductionGenerationAuthorityError(
            "terminal decision is not exact expired claim decision"
        )
    snapshot_artifact, snapshot, snapshot_file_sha = _authenticate_snapshot(
        final_gpu_spend_snapshot,
        run_id=claim["run_id"],
    )
    snapshot_expected = {
        "run_id": claim["run_id"],
        "campaign_identity_sha256": claim["campaign_identity_sha256"],
        "descriptor_sha256": claim["descriptor_file_sha256"],
        "descriptor_body_sha256": claim["descriptor_body_sha256"],
        "approval_sha256": claim["approval_file_sha256"],
        "approval_body_sha256": claim["approval_body_sha256"],
        "gpu_spend_ledger_genesis_sha256": claim[
            "gpu_spend_ledger_genesis_sha256"
        ],
        "gpu_spend_ledger_record_count": claim["gpu_spend_ledger_record_count"],
        "gpu_spend_ledger_tip_record_sha256": claim[
            "gpu_spend_ledger_tip_record_sha256"
        ],
        "ec2_allocation_history_sha256": claim[
            "ec2_allocation_history_sha256"
        ],
        "approved_gpu_runtime_seconds": claim["approved_gpu_runtime_seconds"],
        "approved_gpu_cost_usd": claim["approved_gpu_cost_usd"],
        "hourly_cost_usd": claim["hourly_cost_usd"],
        "consumed_gpu_seconds": claim["consumed_gpu_seconds"],
        "consumed_gpu_cost_usd": claim["consumed_gpu_cost_usd"],
        "remaining_gpu_seconds": claim["remaining_gpu_seconds"],
        "remaining_gpu_cost_usd": claim["remaining_gpu_cost_usd"],
        "open_allocation_count": 0,
    }
    if any(snapshot.get(field) != expected for field, expected in snapshot_expected.items()):
        raise ProductionGenerationAuthorityError(
            "terminal snapshot continuity mismatch"
        )
    return (
        claim_artifact,
        claim,
        claim_file_sha,
        decision_artifact,
        decision,
        decision_file_sha,
        snapshot_artifact,
        snapshot,
        snapshot_file_sha,
    )


def _build_terminal(
    *,
    generation_claim: object,
    start_decision: object,
    final_gpu_spend_snapshot: object,
    terminal_at: Union[datetime, str],
) -> dict[str, object]:
    terminal_exact, terminal_text = _normalize_time_input(
        terminal_at,
        field="terminal_at",
    )
    (
        claim_artifact,
        claim,
        claim_file_sha,
        decision_artifact,
        decision,
        decision_file_sha,
        snapshot_artifact,
        snapshot,
        snapshot_file_sha,
    ) = _authenticate_terminal_sources(
        generation_claim=generation_claim,
        start_decision=start_decision,
        final_gpu_spend_snapshot=final_gpu_spend_snapshot,
    )
    decided = _parse_durable_time(decision["decided_at"], field="decided_at")
    must_start = _parse_durable_time(claim["must_start_by"], field="must_start_by")
    snapshot_at = _parse_durable_time(
        snapshot["observed_at"],
        field="final snapshot observed_at",
    )
    if (
        not terminal_exact >= snapshot_at >= decided >= must_start
        or terminal_exact - snapshot_at > _MAX_AGE
    ):
        raise ProductionGenerationAuthorityError("terminal chronology mismatch")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_production_generation_terminal_v1",
        "account_id": claim["account_id"],
        "region": claim["region"],
        "bucket": claim["bucket"],
        "run_id": claim["run_id"],
        "managed_mode": claim["managed_mode"],
        "campaign_identity_sha256": claim["campaign_identity_sha256"],
        "generation": claim["generation"],
        "generation_text": claim["generation_text"],
        "submit_attempt_id": claim["submit_attempt_id"],
        "sky_job_name": claim["sky_job_name"],
        "sky_job_identity_sha256": claim["sky_job_identity_sha256"],
        "generation_claim_key": claim_artifact.key,
        "generation_claim_file_sha256": claim_file_sha,
        "generation_claim_body_sha256": claim["generation_claim_body_sha256"],
        "generation_claim_version_id": claim_artifact.version_id,
        "start_decision_key": decision_artifact.key,
        "start_decision_file_sha256": decision_file_sha,
        "start_decision_body_sha256": decision["start_decision_body_sha256"],
        "start_decision_version_id": decision_artifact.version_id,
        "terminal_outcome": "expired-unstarted",
        "must_start_by": claim["must_start_by"],
        "terminal_at": terminal_text,
        "final_gpu_spend_snapshot_key": snapshot_artifact.key,
        "final_gpu_spend_snapshot_file_sha256": snapshot_file_sha,
        "final_gpu_spend_snapshot_body_sha256": snapshot["snapshot_body_sha256"],
        "final_gpu_spend_snapshot_version_id": snapshot_artifact.version_id,
        "final_gpu_spend_ledger_genesis_sha256": snapshot[
            "gpu_spend_ledger_genesis_sha256"
        ],
        "final_gpu_spend_ledger_record_count": snapshot[
            "gpu_spend_ledger_record_count"
        ],
        "final_gpu_spend_ledger_tip_record_sha256": snapshot[
            "gpu_spend_ledger_tip_record_sha256"
        ],
        "final_ec2_allocation_history_sha256": snapshot[
            "ec2_allocation_history_sha256"
        ],
        "final_approved_gpu_runtime_seconds": snapshot[
            "approved_gpu_runtime_seconds"
        ],
        "final_approved_gpu_cost_usd": snapshot["approved_gpu_cost_usd"],
        "final_hourly_cost_usd": snapshot["hourly_cost_usd"],
        "final_consumed_gpu_seconds": snapshot["consumed_gpu_seconds"],
        "final_consumed_gpu_cost_usd": snapshot["consumed_gpu_cost_usd"],
        "final_remaining_gpu_seconds": snapshot["remaining_gpu_seconds"],
        "final_remaining_gpu_cost_usd": snapshot["remaining_gpu_cost_usd"],
        "final_open_allocation_count": snapshot["open_allocation_count"],
    }
    result = {
        **body,
        "generation_terminal_body_sha256": _sha(_canonical(body)),
    }
    return _validate_terminal_intrinsic(result)


def build_production_generation_terminal(
    *,
    generation_claim: VersionedJsonArtifact,
    start_decision: VersionedJsonArtifact,
    final_gpu_spend_snapshot: VersionedJsonArtifact,
    terminal_at: Union[datetime, str],
) -> dict[str, object]:
    return _translate(
        lambda: _build_terminal(
            generation_claim=generation_claim,
            start_decision=start_decision,
            final_gpu_spend_snapshot=final_gpu_spend_snapshot,
            terminal_at=terminal_at,
        )
    )


def validate_production_generation_terminal(
    value: Mapping[str, object],
    *,
    generation_claim: VersionedJsonArtifact,
    start_decision: VersionedJsonArtifact,
    final_gpu_spend_snapshot: VersionedJsonArtifact,
    now: Union[datetime, str],
) -> dict[str, object]:
    def operation() -> dict[str, object]:
        record = _validate_terminal_intrinsic(value)
        now_exact, _ = _normalize_time_input(now, field="now")
        terminal_exact = _parse_durable_time(record["terminal_at"], field="terminal_at")
        if now_exact < terminal_exact:
            raise ProductionGenerationAuthorityError("now precedes terminal")
        expected = _build_terminal(
            generation_claim=generation_claim,
            start_decision=start_decision,
            final_gpu_spend_snapshot=final_gpu_spend_snapshot,
            terminal_at=record["terminal_at"],
        )
        if record != expected:
            raise ProductionGenerationAuthorityError("generation terminal mismatch")
        return dict(record)

    return _translate(operation)


def validate_production_generation_inventory(
    entries: list[GenerationInventoryEntry],
    *,
    run_id: str,
) -> dict[str, object]:
    return _translate(lambda: _validate_inventory(entries, run_id=run_id))


def _receipt_fields(
    receipt: object,
    *,
    claim: Mapping[str, object],
    decision: Mapping[str, object],
    decision_artifact: VersionedJsonArtifact,
) -> tuple[datetime, datetime, datetime]:
    if type(receipt) is not UnambiguousStartDecisionCreateReceipt:
        raise ProductionGenerationAuthorityError(
            "receipt must be an exact UnambiguousStartDecisionCreateReceipt"
        )
    string_fields = {
        "record_kind",
        "account_id",
        "region",
        "bucket",
        "expected_bucket_owner",
        "operation",
        "if_none_match",
        "generation_text",
        "submit_attempt_id",
        "key",
        "candidate_file_sha256",
        "request_checksum_algorithm",
        "request_checksum_sha256_base64",
        "version_id",
        "etag",
        "response_checksum_sha256_base64",
        "aws_request_id",
        "server_date",
        "outcome",
        "source",
    }
    for field in string_fields:
        if type(getattr(receipt, field)) is not str:
            raise ProductionGenerationAuthorityError(
                f"receipt.{field} must be an exact string"
            )
    for field in {"generation", "content_length", "http_status"}:
        if type(getattr(receipt, field)) is not int:
            raise ProductionGenerationAuthorityError(
                f"receipt.{field} must be an exact integer"
            )
    raw = decision_artifact.raw
    digest = hashlib.sha256(raw).digest()
    file_sha = digest.hex()
    checksum = base64.b64encode(digest).decode("ascii")
    expected_metadata = (
        ("glm52-account-id", _ACCOUNT_ID),
        ("glm52-region", _REGION),
        ("glm52-run-id", claim["run_id"]),
        ("glm52-generation", claim["generation_text"]),
        ("glm52-submit-attempt-id", claim["submit_attempt_id"]),
        (
            "glm52-record-type",
            "glm52_sky_production_generation_start_decision_v1",
        ),
        ("glm52-body-sha256", decision["start_decision_body_sha256"]),
        ("glm52-file-sha256", file_sha),
    )
    if (
        receipt.record_kind != "generation-start-decision"
        or receipt.account_id != _ACCOUNT_ID
        or receipt.region != _REGION
        or receipt.bucket != claim["bucket"]
        or receipt.expected_bucket_owner != _ACCOUNT_ID
        or receipt.operation != "PutObject"
        or receipt.if_none_match != "*"
        or receipt.generation != claim["generation"]
        or receipt.generation_text != claim["generation_text"]
        or receipt.submit_attempt_id != claim["submit_attempt_id"]
        or receipt.key != decision_artifact.key
        or receipt.content_length != len(raw)
        or receipt.candidate_file_sha256 != file_sha
        or receipt.request_checksum_algorithm != "SHA256"
        or receipt.request_checksum_sha256_base64 != checksum
        or receipt.response_checksum_sha256_base64 != checksum
        or receipt.version_id != decision_artifact.version_id
        or receipt.http_status != 200
        or receipt.outcome != "created"
        or receipt.source != "direct-response"
    ):
        raise ProductionGenerationAuthorityError("receipt field mismatch")
    if type(receipt.immutable_metadata) is not tuple:
        raise ProductionGenerationAuthorityError(
            "receipt immutable metadata must be an exact tuple"
        )
    if any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(item) is not str for item in pair)
        for pair in receipt.immutable_metadata
    ) or receipt.immutable_metadata != expected_metadata:
        raise ProductionGenerationAuthorityError("receipt metadata mismatch")
    if (
        _CHECKSUM_BASE64.fullmatch(receipt.request_checksum_sha256_base64) is None
        or _CHECKSUM_BASE64.fullmatch(receipt.response_checksum_sha256_base64)
        is None
        or _ETAG.fullmatch(receipt.etag) is None
        or _REQUEST_ID.fullmatch(receipt.aws_request_id) is None
    ):
        raise ProductionGenerationAuthorityError("receipt syntax mismatch")
    _digest(receipt.candidate_file_sha256, field="candidate_file_sha256")
    _version(receipt.version_id, field="receipt.version_id")
    server = _parse_imf_time(receipt.server_date, field="receipt.server_date")
    request_exact, _ = _normalize_time_input(
        receipt.request_started_at,
        field="request_started_at",
    )
    response_exact, _ = _normalize_time_input(
        receipt.response_received_at,
        field="response_received_at",
    )
    return request_exact, response_exact, server


def validate_modeled_submit_once(
    *,
    generation_claim: VersionedJsonArtifact,
    start_decision: VersionedJsonArtifact,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
    start_decision_receipt: UnambiguousStartDecisionCreateReceipt,
    post_create_generation_inventory: list[GenerationInventoryEntry],
    post_create_audit_server_date: str,
    now: Union[datetime, str],
) -> ModeledSubmitOnceValidation:
    """Return replayable non-authoritative validation, never transport authority."""

    def operation() -> ModeledSubmitOnceValidation:
        claim_artifact, claim, claim_file_sha = _artifact(
            generation_claim,
            label="generation_claim",
        )
        claim = _validate_claim_intrinsic(claim)
        decision_artifact, decision, decision_file_sha = _artifact(
            start_decision,
            label="start_decision",
        )
        decision = _validate_decision_intrinsic(decision)
        if decision["decision"] != "launch-once":
            raise ProductionGenerationAuthorityError(
                "modeled submission requires launch-once"
            )
        edge = {
            "generation_claim_key": claim_artifact.key,
            "generation_claim_file_sha256": claim_file_sha,
            "generation_claim_body_sha256": claim["generation_claim_body_sha256"],
            "generation_claim_version_id": claim_artifact.version_id,
            "generation": claim["generation"],
            "generation_text": claim["generation_text"],
            "submit_attempt_id": claim["submit_attempt_id"],
            "sky_job_identity_sha256": claim["sky_job_identity_sha256"],
        }
        if any(decision.get(field) != expected for field, expected in edge.items()):
            raise ProductionGenerationAuthorityError(
                "modeled claim/decision lineage mismatch"
            )
        request_exact, response_exact, put_server = _receipt_fields(
            start_decision_receipt,
            claim=claim,
            decision=decision,
            decision_artifact=decision_artifact,
        )
        audit_server = _parse_imf_time(
            post_create_audit_server_date,
            field="post_create_audit_server_date",
        )
        now_exact, validated_text = _normalize_time_input(now, field="now")
        decided_exact = _parse_durable_time(
            decision["decided_at"],
            field="decided_at",
        )
        if not decided_exact <= request_exact <= response_exact <= now_exact:
            raise ProductionGenerationAuthorityError(
                "modeled receipt/audit chronology mismatch"
            )
        if (
            response_exact - request_exact > timedelta(seconds=5)
            or now_exact - response_exact > timedelta(seconds=15)
            or abs(put_server - response_exact) > timedelta(seconds=5)
            or abs(audit_server - now_exact) > timedelta(seconds=5)
        ):
            raise ProductionGenerationAuthorityError(
                "modeled receipt/audit duration or skew mismatch"
            )
        try:
            conservative_exact = max(
                _ceil_whole_second(now_exact),
                put_server + timedelta(seconds=5),
                audit_server + timedelta(seconds=5),
            )
        except OverflowError as error:
            raise ProductionGenerationAuthorityError(
                "conservative validation time overflowed"
            ) from error
        conservative_text = _format_durable_time(conservative_exact)
        must_start = _parse_durable_time(claim["must_start_by"], field="must_start_by")
        if conservative_exact >= must_start:
            raise ProductionGenerationAuthorityError(
                "conservative validation reaches deadline"
            )
        authority = _authenticate_h1d(
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            gpu_spend_snapshot=gpu_spend_snapshot,
            now=conservative_text,
        )
        _claim_edges_match(claim, authority, require_snapshot=True)
        validated_decision = validate_production_generation_start_decision(
            decision,
            generation_claim=claim_artifact,
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            now=conservative_text,
        )
        if validated_decision != decision:
            raise ProductionGenerationAuthorityError("decision validation mismatch")
        inventory = _validate_inventory(
            post_create_generation_inventory,
            run_id=claim["run_id"],
        )
        if inventory["open_generation"] != claim["generation"]:
            raise ProductionGenerationAuthorityError(
                "post-create inventory has wrong open generation"
            )
        row = inventory["generations"][-1]
        claim_projection = row["claim"]
        decision_projection = row["start_decision"]
        if (
            row["terminal"] is not None
            or claim_projection["key"] != claim_artifact.key
            or claim_projection["raw"] != claim_artifact.raw
            or claim_projection["version_id"] != claim_artifact.version_id
            or decision_projection is None
            or decision_projection["key"] != decision_artifact.key
            or decision_projection["raw"] != decision_artifact.raw
            or decision_projection["version_id"] != decision_artifact.version_id
            or decision_projection["file_sha256"] != decision_file_sha
        ):
            raise ProductionGenerationAuthorityError(
                "post-create inventory does not contain exact sole artifacts"
            )
        return ModeledSubmitOnceValidation(
            validation_result="modeled-submit-once-valid",
            generation=claim["generation"],
            submit_attempt_id=claim["submit_attempt_id"],
            sky_job_identity_sha256=claim["sky_job_identity_sha256"],
            claim_key=claim_artifact.key,
            claim_version_id=claim_artifact.version_id,
            start_decision_key=decision_artifact.key,
            start_decision_version_id=decision_artifact.version_id,
            validated_at=validated_text,
            conservative_validation_at=conservative_text,
            must_start_by=claim["must_start_by"],
        )

    return _translate(operation)


def _fail_decision() -> ProductionGenerationDecision:
    return ProductionGenerationDecision(
        action="fail-closed",
        reason="invalid-or-inconsistent-generation-authority",
        generation=None,
    )


def decide_production_generation_action(
    *,
    generation_inventory: list[GenerationInventoryEntry],
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
    submit_attempt_id: str,
    now: Union[datetime, str],
) -> ProductionGenerationDecision:
    def operation() -> ProductionGenerationDecision:
        try:
            attempt = _attempt(submit_attempt_id)
            now_exact, _ = _normalize_time_input(now, field="now")
            intent_preflight = _parse_raw(intent.raw, label="intent preflight")
            run = _run_id(intent_preflight.get("run_id"))
            approval_preflight = _parse_raw(approval.raw, label="approval preflight")
            _authenticate_approval(
                approval,
                run_id=run,
                expected_key=intent_preflight.get("approval_key"),
                expected_file_sha256=intent_preflight.get("approval_file_sha256"),
                expected_body_sha256=intent_preflight.get("approval_body_sha256"),
                expected_version_id=intent_preflight.get("approval_version_id"),
            )
            if approval_preflight.get("approval_body_sha256") != intent_preflight.get(
                "approval_body_sha256"
            ):
                raise ProductionGenerationAuthorityError(
                    "resolver approval preflight mismatch"
                )
            inventory = _validate_inventory(generation_inventory, run_id=run)
            if inventory["generation_count"] == 0:
                _build_claim(
                    generation=1,
                    descriptor=descriptor,
                    intent=intent,
                    approval=approval,
                    controller_baseline=controller_baseline,
                    must_start_control_plane_ready=must_start_control_plane_ready,
                    submission_acquisition=submission_acquisition,
                    gpu_spend_snapshot=gpu_spend_snapshot,
                    generation_inventory=generation_inventory,
                    previous_generation_terminal=None,
                    submit_attempt_id=attempt,
                    claim_created_at=now,
                )
                return ProductionGenerationDecision(
                    action="create-generation-claim",
                    reason="empty-inventory-ready-for-generation-one",
                    generation=1,
                )
            row = inventory["generations"][-1]
            generation_number = row["generation"]
            claim_projection = row["claim"]
            claim_artifact = VersionedJsonArtifact(
                key=claim_projection["key"],
                raw=claim_projection["raw"],
                version_id=claim_projection["version_id"],
            )
            claim = _parse_raw(claim_artifact.raw, label="resolver claim")
            claim = _validate_claim_intrinsic(claim)
            if attempt != claim["submit_attempt_id"] and row["terminal"] is None:
                raise ProductionGenerationAuthorityError(
                    "resolver attempt does not match open claim"
                )
            if row["terminal"] is not None:
                if generation_number >= _MAX_GENERATION:
                    raise ProductionGenerationAuthorityError(
                        "generation space exhausted"
                    )
                terminal_projection = row["terminal"]
                terminal_artifact = VersionedJsonArtifact(
                    key=terminal_projection["key"],
                    raw=terminal_projection["raw"],
                    version_id=terminal_projection["version_id"],
                )
                _build_claim(
                    generation=generation_number + 1,
                    descriptor=descriptor,
                    intent=intent,
                    approval=approval,
                    controller_baseline=controller_baseline,
                    must_start_control_plane_ready=must_start_control_plane_ready,
                    submission_acquisition=submission_acquisition,
                    gpu_spend_snapshot=gpu_spend_snapshot,
                    generation_inventory=generation_inventory,
                    previous_generation_terminal=terminal_artifact,
                    submit_attempt_id=attempt,
                    claim_created_at=now,
                )
                return ProductionGenerationDecision(
                    action="advance-generation",
                    reason="prior-expired-generation-ready-for-advance",
                    generation=generation_number + 1,
                )
            decision_projection = row["start_decision"]
            if decision_projection is None:
                must_start = _parse_durable_time(
                    claim["must_start_by"],
                    field="must_start_by",
                )
                if now_exact < must_start:
                    _build_start_decision(
                        generation_claim=claim_artifact,
                        descriptor=descriptor,
                        intent=intent,
                        approval=approval,
                        controller_baseline=controller_baseline,
                        must_start_control_plane_ready=must_start_control_plane_ready,
                        submission_acquisition=submission_acquisition,
                        decision="launch-once",
                        decided_at=now,
                    )
                    return ProductionGenerationDecision(
                        action="create-launch-once-decision",
                        reason="current-claim-ready-for-launch-decision",
                        generation=generation_number,
                    )
                _build_start_decision(
                    generation_claim=claim_artifact,
                    descriptor=descriptor,
                    intent=intent,
                    approval=approval,
                    controller_baseline=controller_baseline,
                    must_start_control_plane_ready=must_start_control_plane_ready,
                    submission_acquisition=submission_acquisition,
                    decision="expire-unstarted",
                    decided_at=now,
                )
                return ProductionGenerationDecision(
                    action="create-expire-unstarted-decision",
                    reason="current-claim-deadline-reached",
                    generation=generation_number,
                )
            decision_artifact = VersionedJsonArtifact(
                key=decision_projection["key"],
                raw=decision_projection["raw"],
                version_id=decision_projection["version_id"],
            )
            decision = _parse_raw(decision_artifact.raw, label="resolver decision")
            decision = _validate_decision_intrinsic(decision)
            if decision["decision"] == "launch-once":
                authority = _authenticate_h1d(
                    descriptor=descriptor,
                    intent=intent,
                    approval=approval,
                    controller_baseline=controller_baseline,
                    must_start_control_plane_ready=must_start_control_plane_ready,
                    submission_acquisition=submission_acquisition,
                    gpu_spend_snapshot=gpu_spend_snapshot,
                    now=claim["claim_created_at"],
                )
                _claim_edges_match(claim, authority, require_snapshot=True)
                return ProductionGenerationDecision(
                    action="reconcile-only",
                    reason="stored-or-visible-generation-requires-reconciliation",
                    generation=generation_number,
                )
            authority = _authenticate_h1d(
                descriptor=descriptor,
                intent=intent,
                approval=approval,
                controller_baseline=controller_baseline,
                must_start_control_plane_ready=must_start_control_plane_ready,
                submission_acquisition=submission_acquisition,
                now=claim["claim_created_at"],
            )
            _claim_edges_match(claim, authority, require_snapshot=False)
            _build_terminal(
                generation_claim=claim_artifact,
                start_decision=decision_artifact,
                final_gpu_spend_snapshot=gpu_spend_snapshot,
                terminal_at=now,
            )
            return ProductionGenerationDecision(
                action="create-expired-unstarted-terminal",
                reason="current-expiry-decision-ready-for-terminal",
                generation=generation_number,
            )
        except MemoryError:
            raise
        except Exception:
            return _fail_decision()

    return _translate(operation)


__all__ = [
    "GenerationInventoryEntry",
    "ModeledSubmitOnceValidation",
    "ProductionGenerationAction",
    "ProductionGenerationAuthorityError",
    "ProductionGenerationDecision",
    "ProductionGenerationReason",
    "UnambiguousStartDecisionCreateReceipt",
    "build_production_generation_claim",
    "production_generation_claim_file_bytes",
    "production_generation_claim_file_sha256",
    "production_generation_claim_s3_key",
    "validate_production_generation_claim",
    "build_production_generation_start_decision",
    "production_generation_start_decision_file_bytes",
    "production_generation_start_decision_file_sha256",
    "production_generation_start_decision_s3_key",
    "validate_production_generation_start_decision",
    "build_production_generation_terminal",
    "production_generation_terminal_file_bytes",
    "production_generation_terminal_file_sha256",
    "production_generation_terminal_s3_key",
    "validate_production_generation_terminal",
    "validate_production_generation_inventory",
    "decide_production_generation_action",
    "validate_modeled_submit_once",
]
