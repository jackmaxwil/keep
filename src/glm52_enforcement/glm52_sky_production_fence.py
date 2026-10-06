"""Enforcement-native production-fence model for the frozen H.1f contract.

The values returned here are modeled evidence only.  This module performs no
I/O and exposes no write, consume, submit, lock, lease, or active-authority
capability.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Union

from .glm52_gpu_spend_snapshot import (
    validate_gpu_spend_snapshot,
)
from .glm52_sky_campaign import (
    validate_gpu_spend_approval,
    validate_sky_campaign_descriptor,
)
from .glm52_sky_production_acquisition import (
    production_controller_baseline_file_bytes,
    production_controller_baseline_file_sha256,
    production_controller_baseline_s3_key,
    production_must_start_control_plane_ready_file_bytes,
    production_must_start_control_plane_ready_file_sha256,
    production_must_start_control_plane_ready_s3_key,
    production_submission_acquired_file_bytes,
    production_submission_acquired_file_sha256,
    production_submission_acquired_s3_key,
    validate_production_controller_baseline,
    validate_production_must_start_control_plane_ready,
    validate_production_submission_acquired,
)
from .glm52_sky_production_generation import (
    production_generation_claim_s3_key,
    production_generation_start_decision_s3_key,
    production_generation_terminal_s3_key,
)
from .glm52_sky_production_submission import (
    VersionedJsonArtifact,
    production_submission_intent_file_bytes,
    production_submission_intent_file_sha256,
    production_submission_intent_s3_key,
    validate_production_submission_intent,
)
from .glm52_sky_submission_modes import (
    SubmissionModeContractError,
    require_opaque_version_id,
)


class FenceAuthorityError(ValueError):
    """Raised when modeled fence evidence fails closed."""


@dataclass(frozen=True)
class SourceEnrollment:
    source_kind: str
    key: str
    version_id: str
    file_sha256: str
    body_sha256: str
    content_length: int
    checksum_sha256_base64: str
    checksum_type: str
    content_type: str
    metadata: tuple[tuple[str, str], ...]
    publisher_principal_arn: str
    publisher_principal_id: str
    publisher_policy_sha256: str
    publisher_state: str


@dataclass(frozen=True)
class GenerationReservation:
    generation: int
    generation_text: str
    claim_key: str
    claim_write_rule: str
    decision_key: str
    decision_write_rule: str
    terminal_key: str
    terminal_write_rule: str


@dataclass(frozen=True)
class FenceLiveStateIdentity:
    account_id: str
    region: str
    bucket: str
    bucket_versioning_status: str
    bucket_mfa_delete_status: str
    bucket_policy_sha256: str
    lifecycle_configuration_state: str
    lifecycle_configuration_sha256: str
    replication_configuration_state: str
    replication_configuration_sha256: str
    cloudformation_stack_id: str
    cloudformation_template_sha256: str
    cloudformation_parameters_sha256: str
    executor_identity_sha256: str
    publisher_deny_policy_sha256: str
    h1d_control_plane_ready_body_sha256: str


@dataclass(frozen=True)
class FenceControlArtifact:
    key: str
    raw: bytes
    version_id: str
    file_sha256: str
    body_sha256: str
    content_length: int
    etag: str
    last_modified: str
    checksum_sha256_base64: str
    checksum_type: str
    content_type: str
    metadata: tuple[tuple[str, str], ...]
    missing_meta: int


@dataclass(frozen=True)
class ModeledFenceHead:
    genesis: FenceControlArtifact
    successors: tuple[FenceControlArtifact, ...]
    head: FenceControlArtifact
    enrolled_sources: tuple[SourceEnrollment, ...]
    reserved_generations: tuple[GenerationReservation, ...]
    next_successor_key: str


@dataclass(frozen=True)
class _ExpectedSource:
    source_kind: str
    artifact: VersionedJsonArtifact
    body_sha256: str
    file_sha256: str


@dataclass(frozen=True)
class _AuthenticatedSources:
    run_id: str
    bucket: str
    campaign_identity_sha256: str
    ready_body_sha256: str
    expected_sources: tuple[_ExpectedSource, ...]


_ACCOUNT_ID = "246813579024"
_REGION = "us-west-2"
_MAX_GENERATION = 99_999_999
_WRITE_RULE = "conditional-single-part-put-if-absent"
_GENESIS_TYPE = "glm52_sky_production_fence_genesis_v1"
_SUCCESSOR_TYPE = "glm52_sky_production_fence_successor_v1"
_SOURCE_KINDS = {
    "campaign-descriptor",
    "production-submission-intent",
    "production-controller-baseline",
    "production-control-plane-readiness",
    "production-submission-acquisition",
    "gpu-spend-approval",
    "gpu-spend-snapshot",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SAFE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_DURABLE_TIME = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})Z"
)
_SOURCE_FIELDS = {
    "source_kind",
    "key",
    "version_id",
    "file_sha256",
    "body_sha256",
    "content_length",
    "checksum_sha256_base64",
    "checksum_type",
    "content_type",
    "metadata",
    "publisher_principal_arn",
    "publisher_principal_id",
    "publisher_policy_sha256",
    "publisher_state",
}
_RESERVATION_FIELDS = {
    "generation",
    "generation_text",
    "claim_key",
    "claim_write_rule",
    "decision_key",
    "decision_write_rule",
    "terminal_key",
    "terminal_write_rule",
}
_LIVE_STATE_FIELDS = {
    "account_id",
    "region",
    "bucket",
    "bucket_versioning_status",
    "bucket_mfa_delete_status",
    "bucket_policy_sha256",
    "lifecycle_configuration_state",
    "lifecycle_configuration_sha256",
    "replication_configuration_state",
    "replication_configuration_sha256",
    "cloudformation_stack_id",
    "cloudformation_template_sha256",
    "cloudformation_parameters_sha256",
    "executor_identity_sha256",
    "publisher_deny_policy_sha256",
    "h1d_control_plane_ready_body_sha256",
}
_GENESIS_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "campaign_identity_sha256",
    "genesis_key",
    "enrolled_sources",
    "reserved_generations",
    "live_state",
    "created_at",
    "fence_body_sha256",
}
_SUCCESSOR_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "campaign_identity_sha256",
    "successor_key",
    "predecessor_control_key",
    "predecessor_control_version_id",
    "predecessor_control_file_sha256",
    "predecessor_control_body_sha256",
    "enrolled_sources",
    "reserved_generations",
    "added_sources",
    "added_generation_reservations",
    "live_state",
    "selected_at",
    "fence_body_sha256",
}
_ABSENT_CONFIGURATION_SHA256 = hashlib.sha256(
    b'{"state":"absent"}\n'
).hexdigest()
_ROLLBACK_GUARDED_FIELDS = (
    "bucket_policy_sha256",
    "publisher_deny_policy_sha256",
    "cloudformation_template_sha256",
    "cloudformation_parameters_sha256",
    "h1d_control_plane_ready_body_sha256",
)


def _translate(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except FenceAuthorityError:
        raise
    except MemoryError:
        raise
    except Exception as error:
        raise FenceAuthorityError(str(error)) from error


def _require_exact_json(value: object, *, field: str) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise FenceAuthorityError(f"{field} must be finite")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _require_exact_json(item, field=f"{field}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise FenceAuthorityError(
                    f"{field} dictionary keys must be exact strings"
                )
            _require_exact_json(item, field=f"{field}.{key}")
        return
    raise FenceAuthorityError(f"{field} is not an exact JSON value")


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


def _duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FenceAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FenceAuthorityError(f"nonfinite JSON number is forbidden: {value}")


def _parse_raw(raw: object, *, label: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise FenceAuthorityError(f"{label} raw must be exact bytes")
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except FenceAuthorityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise FenceAuthorityError(f"{label} is not strict JSON") from error
    if type(value) is not dict:
        raise FenceAuthorityError(f"{label} root must be an exact object")
    _require_exact_json(value, field=label)
    if raw != _canonical(value) + b"\n":
        raise FenceAuthorityError(f"{label} bytes are not canonical JSON plus LF")
    return value


def _plain_copy(value: dict[str, object]) -> dict[str, object]:
    copied = json.loads(_canonical(value))
    if type(copied) is not dict:
        raise FenceAuthorityError("internal canonical object copy failed")
    return copied


def _exact_dict(
    value: Mapping[str, object],
    fields: set[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise FenceAuthorityError(f"{label} must be an exact object")
    if set(value) != fields:
        raise FenceAuthorityError(f"{label} schema mismatch")
    _require_exact_json(value, field=label)
    return dict(value)


def _exact_string(
    value: object,
    *,
    field: str,
    nonempty: bool = True,
) -> str:
    if type(value) is not str or (nonempty and not value):
        raise FenceAuthorityError(f"{field} must be an exact string")
    return value


def _digest(value: object, *, field: str) -> str:
    text = _exact_string(value, field=field)
    if _SHA256.fullmatch(text) is None:
        raise FenceAuthorityError(f"{field} must be a lowercase SHA-256")
    return text


def _exact_integer(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise FenceAuthorityError(
            f"{field} must be an exact integer >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise FenceAuthorityError(f"{field} must be <= {maximum}")
    return value


def _run_id(value: object) -> str:
    text = _exact_string(value, field="run_id")
    if _RUN_ID.fullmatch(text) is None:
        raise FenceAuthorityError("run_id is invalid")
    return text


def _safe_key(value: object, *, field: str) -> str:
    text = _exact_string(value, field=field)
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise FenceAuthorityError(f"{field} must be ASCII") from error
    segments = text.split("/")
    if (
        len(encoded) > 1024
        or _SAFE_KEY.fullmatch(text) is None
        or text.endswith("/")
        or "//" in text
        or "\\" in text
        or any(segment in {".", ".."} for segment in segments)
    ):
        raise FenceAuthorityError(f"{field} is unsafe")
    return text


def _source_key(value: object, *, run_id: str, field: str) -> str:
    key = _safe_key(value, field=field)
    run_prefix = f"campaigns/{run_id}/"
    fence_prefix = f"{run_prefix}authorities/fence/"
    if (
        not key.startswith(run_prefix)
        or key == fence_prefix
        or key.startswith(fence_prefix)
    ):
        raise FenceAuthorityError(
            f"{field} must be run-scoped and outside the fence namespace"
        )
    return key


def _version_id(value: object, *, field: str) -> str:
    try:
        return require_opaque_version_id(value, field=field)
    except SubmissionModeContractError as error:
        raise FenceAuthorityError(str(error)) from error


def _format_time(value: datetime) -> str:
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
        f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}Z"
    )


def _parse_time(value: object, *, field: str) -> datetime:
    text = _exact_string(value, field=field)
    match = _DURABLE_TIME.fullmatch(text)
    if match is None:
        raise FenceAuthorityError(f"{field} must be whole-second canonical UTC")
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
        raise FenceAuthorityError(
            f"{field} must be whole-second canonical UTC"
        ) from error
    if _format_time(parsed) != text:
        raise FenceAuthorityError(f"{field} must be whole-second canonical UTC")
    return parsed


def _normalize_time(value: Union[datetime, str], *, field: str) -> str:
    if type(value) is str:
        _parse_time(value, field=field)
        return value
    if type(value) is not datetime:
        raise FenceAuthorityError(
            f"{field} must be an exact datetime or canonical UTC string"
        )
    zone = value.tzinfo
    if type(zone) is not timezone or value.microsecond != 0:
        raise FenceAuthorityError(
            f"{field} must have exact fixed-offset tzinfo and whole seconds"
        )
    try:
        offset = zone.utcoffset(value)
    except MemoryError:
        raise
    except Exception as error:
        raise FenceAuthorityError(f"{field} UTC offset lookup failed") from error
    if (
        type(offset) is not timedelta
        or offset.microseconds != 0
        or offset.total_seconds() % 60 != 0
        or abs(offset) >= timedelta(hours=24)
    ):
        raise FenceAuthorityError(
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
        )
        utc = (local_naive - offset).replace(tzinfo=timezone.utc)
    except (OverflowError, ValueError) as error:
        raise FenceAuthorityError(f"{field} UTC normalization overflowed") from error
    return _format_time(utc)


def _metadata(
    value: object,
    *,
    run_id: str,
    field: str,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise FenceAuthorityError(f"{field} must be an exact tuple")
    pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(value):
        if type(pair) is not tuple or len(pair) != 2:
            raise FenceAuthorityError(
                f"{field}[{index}] must be an exact two-string tuple"
            )
        key, item = pair
        if type(key) is not str or type(item) is not str:
            raise FenceAuthorityError(
                f"{field}[{index}] must be an exact two-string tuple"
            )
        if not key or key != key.lower():
            raise FenceAuthorityError(f"{field}[{index}] key must be lowercase")
        pairs.append((key, item))
    result = tuple(pairs)
    if result != tuple(sorted(result, key=lambda pair: pair[0])):
        raise FenceAuthorityError(f"{field} must be sorted by metadata key")
    if len({key for key, _ in result}) != len(result):
        raise FenceAuthorityError(f"{field} metadata keys must be unique")
    if dict(result).get("glm52-run-id") != run_id:
        raise FenceAuthorityError(f"{field} must bind the exact run_id")
    return result


def _metadata_from_json(
    value: object,
    *,
    run_id: str,
    field: str,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not list:
        raise FenceAuthorityError(f"{field} must be an exact JSON list")
    pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(value):
        if type(pair) is not list or len(pair) != 2:
            raise FenceAuthorityError(
                f"{field}[{index}] must be an exact two-string list"
            )
        key, item = pair
        if type(key) is not str or type(item) is not str:
            raise FenceAuthorityError(
                f"{field}[{index}] must be an exact two-string list"
            )
        pairs.append((key, item))
    return _metadata(tuple(pairs), run_id=run_id, field=field)


def _validate_source(
    value: object,
    *,
    run_id: str,
    field: str,
) -> SourceEnrollment:
    if type(value) is not SourceEnrollment:
        raise FenceAuthorityError(f"{field} must be an exact SourceEnrollment")
    kind = _exact_string(value.source_kind, field=f"{field}.source_kind")
    if kind not in _SOURCE_KINDS:
        raise FenceAuthorityError(f"{field}.source_kind is unsupported")
    _source_key(value.key, run_id=run_id, field=f"{field}.key")
    _version_id(value.version_id, field=f"{field}.version_id")
    file_sha = _digest(value.file_sha256, field=f"{field}.file_sha256")
    _digest(value.body_sha256, field=f"{field}.body_sha256")
    _exact_integer(value.content_length, field=f"{field}.content_length")
    expected_checksum = base64.b64encode(bytes.fromhex(file_sha)).decode("ascii")
    if (
        type(value.checksum_sha256_base64) is not str
        or value.checksum_sha256_base64 != expected_checksum
    ):
        raise FenceAuthorityError(f"{field}.checksum_sha256_base64 mismatch")
    if value.checksum_type != "FULL_OBJECT":
        raise FenceAuthorityError(f"{field}.checksum_type must be FULL_OBJECT")
    if value.content_type != "application/json":
        raise FenceAuthorityError(
            f"{field}.content_type must be application/json"
        )
    metadata = dict(
        _metadata(value.metadata, run_id=run_id, field=f"{field}.metadata")
    )
    if (
        "glm52-body-sha256" in metadata
        and metadata["glm52-body-sha256"] != value.body_sha256
    ):
        raise FenceAuthorityError(f"{field}.metadata body SHA-256 mismatch")
    if (
        "glm52-file-sha256" in metadata
        and metadata["glm52-file-sha256"] != value.file_sha256
    ):
        raise FenceAuthorityError(f"{field}.metadata file SHA-256 mismatch")
    _exact_string(
        value.publisher_principal_arn,
        field=f"{field}.publisher_principal_arn",
    )
    _exact_string(
        value.publisher_principal_id,
        field=f"{field}.publisher_principal_id",
    )
    _digest(
        value.publisher_policy_sha256,
        field=f"{field}.publisher_policy_sha256",
    )
    if value.publisher_state not in {"retired", "explicitly-denied"}:
        raise FenceAuthorityError(f"{field}.publisher_state is invalid")
    return value


def _source_json(value: SourceEnrollment, *, run_id: str) -> dict[str, object]:
    exact = _validate_source(value, run_id=run_id, field="source_enrollment")
    return {
        "source_kind": exact.source_kind,
        "key": exact.key,
        "version_id": exact.version_id,
        "file_sha256": exact.file_sha256,
        "body_sha256": exact.body_sha256,
        "content_length": exact.content_length,
        "checksum_sha256_base64": exact.checksum_sha256_base64,
        "checksum_type": exact.checksum_type,
        "content_type": exact.content_type,
        "metadata": [[key, item] for key, item in exact.metadata],
        "publisher_principal_arn": exact.publisher_principal_arn,
        "publisher_principal_id": exact.publisher_principal_id,
        "publisher_policy_sha256": exact.publisher_policy_sha256,
        "publisher_state": exact.publisher_state,
    }


def _source_from_json(
    value: object,
    *,
    run_id: str,
    field: str,
) -> SourceEnrollment:
    if type(value) is not dict or set(value) != _SOURCE_FIELDS:
        raise FenceAuthorityError(f"{field} schema mismatch")
    metadata = _metadata_from_json(
        value["metadata"],
        run_id=run_id,
        field=f"{field}.metadata",
    )
    result = SourceEnrollment(
        source_kind=value["source_kind"],  # type: ignore[arg-type]
        key=value["key"],  # type: ignore[arg-type]
        version_id=value["version_id"],  # type: ignore[arg-type]
        file_sha256=value["file_sha256"],  # type: ignore[arg-type]
        body_sha256=value["body_sha256"],  # type: ignore[arg-type]
        content_length=value["content_length"],  # type: ignore[arg-type]
        checksum_sha256_base64=value["checksum_sha256_base64"],  # type: ignore[arg-type]
        checksum_type=value["checksum_type"],  # type: ignore[arg-type]
        content_type=value["content_type"],  # type: ignore[arg-type]
        metadata=metadata,
        publisher_principal_arn=value["publisher_principal_arn"],  # type: ignore[arg-type]
        publisher_principal_id=value["publisher_principal_id"],  # type: ignore[arg-type]
        publisher_policy_sha256=value["publisher_policy_sha256"],  # type: ignore[arg-type]
        publisher_state=value["publisher_state"],  # type: ignore[arg-type]
    )
    return _validate_source(result, run_id=run_id, field=field)


def _source_tuple(
    value: object,
    *,
    run_id: str,
    field: str,
    genesis: bool = False,
) -> tuple[SourceEnrollment, ...]:
    if type(value) is not list:
        raise FenceAuthorityError(f"{field} must be an exact list")
    result = tuple(
        _source_from_json(item, run_id=run_id, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if result != tuple(sorted(result, key=lambda item: (item.source_kind, item.key))):
        raise FenceAuthorityError(f"{field} must be sorted")
    keys = [item.key for item in result]
    if len(keys) != len(set(keys)):
        raise FenceAuthorityError(f"{field} source keys must be unique")
    if genesis:
        if (
            len(result) != len(_SOURCE_KINDS)
            or {item.source_kind for item in result} != _SOURCE_KINDS
        ):
            raise FenceAuthorityError(
                "genesis must contain exactly seven mandatory sources"
            )
    return result


def _validate_reservation(
    value: object,
    *,
    run_id: str,
    field: str,
) -> GenerationReservation:
    if type(value) is not GenerationReservation:
        raise FenceAuthorityError(
            f"{field} must be an exact GenerationReservation"
        )
    generation = _exact_integer(
        value.generation,
        field=f"{field}.generation",
        minimum=1,
        maximum=_MAX_GENERATION,
    )
    if value.generation_text != f"{generation:08d}":
        raise FenceAuthorityError(f"{field}.generation_text mismatch")
    expected_keys = (
        production_generation_claim_s3_key(
            run_id=run_id,
            generation=generation,
        ),
        production_generation_start_decision_s3_key(
            run_id=run_id,
            generation=generation,
        ),
        production_generation_terminal_s3_key(
            run_id=run_id,
            generation=generation,
        ),
    )
    if (
        value.claim_key,
        value.decision_key,
        value.terminal_key,
    ) != expected_keys:
        raise FenceAuthorityError(f"{field} generation keys mismatch")
    if (
        value.claim_write_rule,
        value.decision_write_rule,
        value.terminal_write_rule,
    ) != (_WRITE_RULE, _WRITE_RULE, _WRITE_RULE):
        raise FenceAuthorityError(f"{field} write rule mismatch")
    return value


def _reservation_json(
    value: GenerationReservation,
    *,
    run_id: str,
) -> dict[str, object]:
    exact = _validate_reservation(
        value,
        run_id=run_id,
        field="generation_reservation",
    )
    return {
        "generation": exact.generation,
        "generation_text": exact.generation_text,
        "claim_key": exact.claim_key,
        "claim_write_rule": exact.claim_write_rule,
        "decision_key": exact.decision_key,
        "decision_write_rule": exact.decision_write_rule,
        "terminal_key": exact.terminal_key,
        "terminal_write_rule": exact.terminal_write_rule,
    }


def _reservation_from_json(
    value: object,
    *,
    run_id: str,
    field: str,
) -> GenerationReservation:
    if type(value) is not dict or set(value) != _RESERVATION_FIELDS:
        raise FenceAuthorityError(f"{field} schema mismatch")
    result = GenerationReservation(
        generation=value["generation"],  # type: ignore[arg-type]
        generation_text=value["generation_text"],  # type: ignore[arg-type]
        claim_key=value["claim_key"],  # type: ignore[arg-type]
        claim_write_rule=value["claim_write_rule"],  # type: ignore[arg-type]
        decision_key=value["decision_key"],  # type: ignore[arg-type]
        decision_write_rule=value["decision_write_rule"],  # type: ignore[arg-type]
        terminal_key=value["terminal_key"],  # type: ignore[arg-type]
        terminal_write_rule=value["terminal_write_rule"],  # type: ignore[arg-type]
    )
    return _validate_reservation(result, run_id=run_id, field=field)


def _reservation_tuple(
    value: object,
    *,
    run_id: str,
    field: str,
) -> tuple[GenerationReservation, ...]:
    if type(value) is not list:
        raise FenceAuthorityError(f"{field} must be an exact list")
    result = tuple(
        _reservation_from_json(
            item,
            run_id=run_id,
            field=f"{field}[{index}]",
        )
        for index, item in enumerate(value)
    )
    generations = tuple(item.generation for item in result)
    if generations != tuple(range(1, len(result) + 1)):
        raise FenceAuthorityError(
            f"{field} must be unique, sorted, and contiguous from 1"
        )
    return result


def _validate_live_state(
    value: object,
    *,
    bucket: str,
    field: str,
) -> FenceLiveStateIdentity:
    if type(value) is not FenceLiveStateIdentity:
        raise FenceAuthorityError(
            f"{field} must be an exact FenceLiveStateIdentity"
        )
    if (
        value.account_id != _ACCOUNT_ID
        or value.region != _REGION
        or value.bucket != bucket
        or value.bucket_versioning_status != "Enabled"
        or value.bucket_mfa_delete_status
        not in {"Enabled", "Disabled", "Unavailable"}
    ):
        raise FenceAuthorityError(f"{field} cloud identity mismatch")
    if (
        value.lifecycle_configuration_state != "absent"
        or value.lifecycle_configuration_sha256
        != _ABSENT_CONFIGURATION_SHA256
        or value.replication_configuration_state != "absent"
        or value.replication_configuration_sha256
        != _ABSENT_CONFIGURATION_SHA256
    ):
        raise FenceAuthorityError(
            f"{field} lifecycle and replication must be exact absent"
        )
    for digest_field in (
        "bucket_policy_sha256",
        "lifecycle_configuration_sha256",
        "replication_configuration_sha256",
        "cloudformation_template_sha256",
        "cloudformation_parameters_sha256",
        "executor_identity_sha256",
        "publisher_deny_policy_sha256",
        "h1d_control_plane_ready_body_sha256",
    ):
        _digest(getattr(value, digest_field), field=f"{field}.{digest_field}")
    _exact_string(
        value.cloudformation_stack_id,
        field=f"{field}.cloudformation_stack_id",
    )
    return value


def _live_state_json(
    value: FenceLiveStateIdentity,
    *,
    bucket: str,
) -> dict[str, object]:
    exact = _validate_live_state(value, bucket=bucket, field="live_state")
    return {name: getattr(exact, name) for name in _LIVE_STATE_FIELDS}


def _live_state_from_json(
    value: object,
    *,
    bucket: str,
    field: str,
) -> FenceLiveStateIdentity:
    if type(value) is not dict or set(value) != _LIVE_STATE_FIELDS:
        raise FenceAuthorityError(f"{field} schema mismatch")
    result = FenceLiveStateIdentity(
        account_id=value["account_id"],  # type: ignore[arg-type]
        region=value["region"],  # type: ignore[arg-type]
        bucket=value["bucket"],  # type: ignore[arg-type]
        bucket_versioning_status=value["bucket_versioning_status"],  # type: ignore[arg-type]
        bucket_mfa_delete_status=value["bucket_mfa_delete_status"],  # type: ignore[arg-type]
        bucket_policy_sha256=value["bucket_policy_sha256"],  # type: ignore[arg-type]
        lifecycle_configuration_state=value["lifecycle_configuration_state"],  # type: ignore[arg-type]
        lifecycle_configuration_sha256=value["lifecycle_configuration_sha256"],  # type: ignore[arg-type]
        replication_configuration_state=value["replication_configuration_state"],  # type: ignore[arg-type]
        replication_configuration_sha256=value["replication_configuration_sha256"],  # type: ignore[arg-type]
        cloudformation_stack_id=value["cloudformation_stack_id"],  # type: ignore[arg-type]
        cloudformation_template_sha256=value["cloudformation_template_sha256"],  # type: ignore[arg-type]
        cloudformation_parameters_sha256=value["cloudformation_parameters_sha256"],  # type: ignore[arg-type]
        executor_identity_sha256=value["executor_identity_sha256"],  # type: ignore[arg-type]
        publisher_deny_policy_sha256=value["publisher_deny_policy_sha256"],  # type: ignore[arg-type]
        h1d_control_plane_ready_body_sha256=value[
            "h1d_control_plane_ready_body_sha256"
        ],  # type: ignore[arg-type]
    )
    return _validate_live_state(result, bucket=bucket, field=field)


def _record_common(record: dict[str, object], *, record_type: str) -> tuple[str, str]:
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["record_type"] != record_type
        or record["account_id"] != _ACCOUNT_ID
        or record["region"] != _REGION
    ):
        raise FenceAuthorityError("fence record identity mismatch")
    run = _run_id(record["run_id"])
    bucket = _exact_string(record["bucket"], field="bucket")
    _digest(
        record["campaign_identity_sha256"],
        field="campaign_identity_sha256",
    )
    return run, bucket


def _validate_body_sha(record: dict[str, object]) -> None:
    actual = _digest(record["fence_body_sha256"], field="fence_body_sha256")
    body = dict(record)
    body.pop("fence_body_sha256")
    if actual != _sha(_canonical(body)):
        raise FenceAuthorityError("fence body SHA-256 mismatch")


def _validate_genesis_intrinsic(
    value: Mapping[str, object],
) -> dict[str, object]:
    record = _exact_dict(value, _GENESIS_FIELDS, label="fence genesis")
    run, bucket = _record_common(record, record_type=_GENESIS_TYPE)
    if record["genesis_key"] != _fence_genesis_key(run):
        raise FenceAuthorityError("genesis key mismatch")
    sources = _source_tuple(
        record["enrolled_sources"],
        run_id=run,
        field="enrolled_sources",
        genesis=True,
    )
    reservations = _reservation_tuple(
        record["reserved_generations"],
        run_id=run,
        field="reserved_generations",
    )
    if len(reservations) != 1 or reservations[0].generation != 1:
        raise FenceAuthorityError("genesis must reserve only generation 1")
    live_state = _live_state_from_json(
        record["live_state"],
        bucket=bucket,
        field="live_state",
    )
    ready = next(
        source
        for source in sources
        if source.source_kind == "production-control-plane-readiness"
    )
    if live_state.h1d_control_plane_ready_body_sha256 != ready.body_sha256:
        raise FenceAuthorityError(
            "genesis live state must bind authenticated readiness"
        )
    _parse_time(record["created_at"], field="created_at")
    _validate_body_sha(record)
    return _plain_copy(record)


def _validate_successor_intrinsic(
    value: Mapping[str, object],
) -> dict[str, object]:
    record = _exact_dict(value, _SUCCESSOR_FIELDS, label="fence successor")
    run, bucket = _record_common(record, record_type=_SUCCESSOR_TYPE)
    predecessor_body = _digest(
        record["predecessor_control_body_sha256"],
        field="predecessor_control_body_sha256",
    )
    if record["successor_key"] != _fence_successor_key(run, predecessor_body):
        raise FenceAuthorityError("successor key mismatch")
    _safe_key(
        record["predecessor_control_key"],
        field="predecessor_control_key",
    )
    _version_id(
        record["predecessor_control_version_id"],
        field="predecessor_control_version_id",
    )
    _digest(
        record["predecessor_control_file_sha256"],
        field="predecessor_control_file_sha256",
    )
    sources = _source_tuple(
        record["enrolled_sources"],
        run_id=run,
        field="enrolled_sources",
    )
    added_sources = _source_tuple(
        record["added_sources"],
        run_id=run,
        field="added_sources",
    )
    if len({source.source_kind for source in added_sources}) != len(
        added_sources
    ):
        raise FenceAuthorityError(
            "one successor may add at most one source of each kind"
        )
    if any(source not in sources for source in added_sources):
        raise FenceAuthorityError("added source is absent from complete set")
    reservations = _reservation_tuple(
        record["reserved_generations"],
        run_id=run,
        field="reserved_generations",
    )
    added_reservations_value = record["added_generation_reservations"]
    if type(added_reservations_value) is not list:
        raise FenceAuthorityError(
            "added_generation_reservations must be an exact list"
        )
    added_reservations = tuple(
        _reservation_from_json(
            item,
            run_id=run,
            field=f"added_generation_reservations[{index}]",
        )
        for index, item in enumerate(added_reservations_value)
    )
    if len(added_reservations) > 1:
        raise FenceAuthorityError(
            "one successor may add only the next generation reservation"
        )
    if any(item not in reservations for item in added_reservations):
        raise FenceAuthorityError(
            "added reservation is absent from complete set"
        )
    if added_reservations and (
        len(reservations) < 2
        or added_reservations[0] != reservations[-1]
    ):
        raise FenceAuthorityError(
            "added reservation must be the newest contiguous reservation"
        )
    if not added_sources and not added_reservations:
        raise FenceAuthorityError("successor delta must be nonempty")
    _live_state_from_json(record["live_state"], bucket=bucket, field="live_state")
    _parse_time(record["selected_at"], field="selected_at")
    _validate_body_sha(record)
    return _plain_copy(record)


def _validate_record_intrinsic(
    value: Mapping[str, object],
) -> dict[str, object]:
    if type(value) is not dict:
        raise FenceAuthorityError("fence control must be an exact object")
    record_type = value.get("record_type")
    if record_type == _GENESIS_TYPE:
        return _validate_genesis_intrinsic(value)
    if record_type == _SUCCESSOR_TYPE:
        return _validate_successor_intrinsic(value)
    raise FenceAuthorityError("unknown fence control record")


def _artifact(
    value: object,
    *,
    label: str,
) -> tuple[VersionedJsonArtifact, dict[str, object], str]:
    if type(value) is not VersionedJsonArtifact:
        raise FenceAuthorityError(
            f"{label} must be an exact VersionedJsonArtifact"
        )
    _safe_key(value.key, field=f"{label}.key")
    _version_id(value.version_id, field=f"{label}.version_id")
    record = _parse_raw(value.raw, label=label)
    return value, record, _sha(value.raw)


def _public_match(
    validated: object,
    record: dict[str, object],
    *,
    label: str,
) -> None:
    if type(validated) is not dict or validated != record:
        raise FenceAuthorityError(f"{label} public validation mismatch")


def _authenticate_sources(
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
) -> _AuthenticatedSources:
    descriptor_artifact, descriptor_record, descriptor_file_sha = _artifact(
        descriptor,
        label="descriptor",
    )
    _public_match(
        validate_sky_campaign_descriptor(
            _parse_raw(descriptor_artifact.raw, label="descriptor validation"),
        ),
        descriptor_record,
        label="descriptor",
    )
    run = _run_id(descriptor_record.get("run_id"))
    bucket = _exact_string(descriptor_record.get("bucket"), field="descriptor.bucket")
    campaign_identity = _digest(
        descriptor_record.get("campaign_identity_sha256"),
        field="descriptor.campaign_identity_sha256",
    )
    descriptor_body = _digest(
        descriptor_record.get("descriptor_body_sha256"),
        field="descriptor.descriptor_body_sha256",
    )
    if (
        descriptor_artifact.key != descriptor_record.get("campaign_descriptor_key")
        or descriptor_record.get("account_id") != _ACCOUNT_ID
        or descriptor_record.get("region") != _REGION
    ):
        raise FenceAuthorityError("descriptor coordinate mismatch")

    intent_artifact, intent_record, intent_file_sha = _artifact(
        intent,
        label="intent",
    )
    _public_match(
        validate_production_submission_intent(
            _parse_raw(intent_artifact.raw, label="intent validation"),
        ),
        intent_record,
        label="intent",
    )
    intent_body = _digest(
        intent_record.get("intent_body_sha256"),
        field="intent.intent_body_sha256",
    )
    if (
        intent_artifact.key
        != production_submission_intent_s3_key(
            run_id=run,
            intent_body_sha256=intent_body,
        )
        or production_submission_intent_file_bytes(intent_record)
        != intent_artifact.raw
        or production_submission_intent_file_sha256(intent_record)
        != intent_file_sha
        or intent_record.get("run_id") != run
        or intent_record.get("bucket") != bucket
        or intent_record.get("campaign_identity_sha256") != campaign_identity
        or intent_record.get("descriptor_key") != descriptor_artifact.key
        or intent_record.get("descriptor_file_sha256") != descriptor_file_sha
        or intent_record.get("descriptor_body_sha256") != descriptor_body
        or intent_record.get("descriptor_version_id")
        != descriptor_artifact.version_id
    ):
        raise FenceAuthorityError("descriptor/intent lineage mismatch")

    approval_artifact, approval_record, approval_file_sha = _artifact(
        approval,
        label="approval",
    )
    _public_match(
        validate_gpu_spend_approval(
            _parse_raw(approval_artifact.raw, label="approval validation"),
        ),
        approval_record,
        label="approval",
    )
    approval_body = _digest(
        approval_record.get("approval_body_sha256"),
        field="approval.approval_body_sha256",
    )
    approval_key = (
        f"campaigns/{run}/authorities/"
        f"GPU_SPEND_APPROVAL-{approval_file_sha}.json"
    )
    if (
        approval_artifact.key != approval_key
        or descriptor_record.get("approval_key") != approval_artifact.key
        or descriptor_record.get("approval_sha256") != approval_file_sha
        or intent_record.get("approval_key") != approval_artifact.key
        or intent_record.get("approval_file_sha256") != approval_file_sha
        or intent_record.get("approval_body_sha256") != approval_body
        or intent_record.get("approval_version_id") != approval_artifact.version_id
    ):
        raise FenceAuthorityError("approval lineage mismatch")

    baseline_artifact, baseline_record, baseline_file_sha = _artifact(
        controller_baseline,
        label="controller_baseline",
    )
    _public_match(
        validate_production_controller_baseline(
            _parse_raw(
                baseline_artifact.raw,
                label="controller baseline validation",
            ),
            descriptor=descriptor_artifact,
            intent=intent_artifact,
        ),
        baseline_record,
        label="controller baseline",
    )
    baseline_body = _digest(
        baseline_record.get("baseline_body_sha256"),
        field="baseline.baseline_body_sha256",
    )
    if (
        baseline_artifact.key
        != production_controller_baseline_s3_key(
            run_id=run,
            baseline_body_sha256=baseline_body,
        )
        or production_controller_baseline_file_bytes(baseline_record)
        != baseline_artifact.raw
        or production_controller_baseline_file_sha256(baseline_record)
        != baseline_file_sha
    ):
        raise FenceAuthorityError("controller baseline coordinate mismatch")

    ready_artifact, ready_record, ready_file_sha = _artifact(
        must_start_control_plane_ready,
        label="must_start_control_plane_ready",
    )
    _public_match(
        validate_production_must_start_control_plane_ready(
            _parse_raw(
                ready_artifact.raw,
                label="control plane readiness validation",
            ),
            descriptor=descriptor_artifact,
            intent=intent_artifact,
            controller_baseline=baseline_artifact,
        ),
        ready_record,
        label="control plane readiness",
    )
    ready_body = _digest(
        ready_record.get("control_plane_ready_body_sha256"),
        field="ready.control_plane_ready_body_sha256",
    )
    if (
        ready_artifact.key
        != production_must_start_control_plane_ready_s3_key(
            run_id=run,
            intent_body_sha256=intent_body,
            control_plane_ready_body_sha256=ready_body,
        )
        or production_must_start_control_plane_ready_file_bytes(ready_record)
        != ready_artifact.raw
        or production_must_start_control_plane_ready_file_sha256(ready_record)
        != ready_file_sha
    ):
        raise FenceAuthorityError("control plane readiness coordinate mismatch")

    acquisition_artifact, acquisition_record, acquisition_file_sha = _artifact(
        submission_acquisition,
        label="submission_acquisition",
    )
    acquired_at = acquisition_record.get("acquired_at")
    _public_match(
        validate_production_submission_acquired(
            _parse_raw(
                acquisition_artifact.raw,
                label="submission acquisition validation",
            ),
            descriptor=descriptor_artifact,
            intent=intent_artifact,
            controller_baseline=baseline_artifact,
            must_start_control_plane_ready=ready_artifact,
            now=acquired_at,  # type: ignore[arg-type]
        ),
        acquisition_record,
        label="submission acquisition",
    )
    acquisition_body = _digest(
        acquisition_record.get("acquisition_body_sha256"),
        field="acquisition.acquisition_body_sha256",
    )
    if (
        acquisition_artifact.key
        != production_submission_acquired_s3_key(
            run_id=run,
            descriptor_file_sha256=descriptor_file_sha,
        )
        or production_submission_acquired_file_bytes(acquisition_record)
        != acquisition_artifact.raw
        or production_submission_acquired_file_sha256(acquisition_record)
        != acquisition_file_sha
    ):
        raise FenceAuthorityError("submission acquisition coordinate mismatch")

    snapshot_artifact, snapshot_record, snapshot_file_sha = _artifact(
        gpu_spend_snapshot,
        label="gpu_spend_snapshot",
    )
    _public_match(
        validate_gpu_spend_snapshot(
            _parse_raw(snapshot_artifact.raw, label="spend snapshot validation"),
        ),
        snapshot_record,
        label="spend snapshot",
    )
    snapshot_body = _digest(
        snapshot_record.get("snapshot_body_sha256"),
        field="snapshot.snapshot_body_sha256",
    )
    snapshot_key = (
        f"campaigns/{run}/spend-snapshots/{snapshot_body}/"
        "GPU_SPEND_SNAPSHOT.json"
    )
    if (
        snapshot_artifact.key != snapshot_key
        or snapshot_record.get("run_id") != run
        or snapshot_record.get("campaign_identity_sha256") != campaign_identity
        or snapshot_record.get("descriptor_sha256") != descriptor_file_sha
        or snapshot_record.get("descriptor_body_sha256") != descriptor_body
        or snapshot_record.get("approval_sha256") != approval_file_sha
        or snapshot_record.get("approval_body_sha256") != approval_body
        or intent_record.get("gpu_spend_snapshot_key") != snapshot_artifact.key
        or intent_record.get("gpu_spend_snapshot_file_sha256")
        != snapshot_file_sha
        or intent_record.get("gpu_spend_snapshot_body_sha256") != snapshot_body
        or intent_record.get("gpu_spend_snapshot_version_id")
        != snapshot_artifact.version_id
    ):
        raise FenceAuthorityError("spend snapshot lineage mismatch")

    expected = (
        _ExpectedSource(
            "campaign-descriptor",
            descriptor_artifact,
            descriptor_body,
            descriptor_file_sha,
        ),
        _ExpectedSource(
            "production-submission-intent",
            intent_artifact,
            intent_body,
            intent_file_sha,
        ),
        _ExpectedSource(
            "production-controller-baseline",
            baseline_artifact,
            baseline_body,
            baseline_file_sha,
        ),
        _ExpectedSource(
            "production-control-plane-readiness",
            ready_artifact,
            ready_body,
            ready_file_sha,
        ),
        _ExpectedSource(
            "production-submission-acquisition",
            acquisition_artifact,
            acquisition_body,
            acquisition_file_sha,
        ),
        _ExpectedSource(
            "gpu-spend-approval",
            approval_artifact,
            approval_body,
            approval_file_sha,
        ),
        _ExpectedSource(
            "gpu-spend-snapshot",
            snapshot_artifact,
            snapshot_body,
            snapshot_file_sha,
        ),
    )
    return _AuthenticatedSources(
        run_id=run,
        bucket=bucket,
        campaign_identity_sha256=campaign_identity,
        ready_body_sha256=ready_body,
        expected_sources=tuple(
            sorted(expected, key=lambda item: (item.source_kind, item.artifact.key))
        ),
    )


def _require_context_common(
    record: dict[str, object],
    context: _AuthenticatedSources,
) -> None:
    if (
        record["account_id"] != _ACCOUNT_ID
        or record["region"] != _REGION
        or record["bucket"] != context.bucket
        or record["run_id"] != context.run_id
        or record["campaign_identity_sha256"]
        != context.campaign_identity_sha256
    ):
        raise FenceAuthorityError("fence/source campaign identity mismatch")


def _require_genesis_sources(
    sources: tuple[SourceEnrollment, ...],
    context: _AuthenticatedSources,
) -> None:
    if len(sources) != len(context.expected_sources):
        raise FenceAuthorityError(
            "genesis must contain exactly seven mandatory sources"
        )
    for enrollment, expected in zip(sources, context.expected_sources):
        artifact = expected.artifact
        expected_checksum = base64.b64encode(
            bytes.fromhex(expected.file_sha256)
        ).decode("ascii")
        if (
            enrollment.source_kind != expected.source_kind
            or enrollment.key != artifact.key
            or enrollment.version_id != artifact.version_id
            or enrollment.file_sha256 != expected.file_sha256
            or enrollment.body_sha256 != expected.body_sha256
            or enrollment.content_length != len(artifact.raw)
            or enrollment.checksum_sha256_base64 != expected_checksum
        ):
            raise FenceAuthorityError(
                f"{expected.source_kind} enrollment does not bind exact artifact"
            )


def _validate_genesis_external(
    value: Mapping[str, object],
    *,
    context: _AuthenticatedSources,
) -> dict[str, object]:
    record = _validate_genesis_intrinsic(value)
    _require_context_common(record, context)
    sources = _source_tuple(
        record["enrolled_sources"],
        run_id=context.run_id,
        field="enrolled_sources",
        genesis=True,
    )
    _require_genesis_sources(sources, context)
    live_state = _live_state_from_json(
        record["live_state"],
        bucket=context.bucket,
        field="live_state",
    )
    if live_state.h1d_control_plane_ready_body_sha256 != context.ready_body_sha256:
        raise FenceAuthorityError(
            "genesis live state does not bind authenticated H.1d readiness"
        )
    return record


def _validate_control_artifact(
    value: object,
) -> tuple[FenceControlArtifact, dict[str, object]]:
    if type(value) is not FenceControlArtifact:
        raise FenceAuthorityError(
            "control must be an exact FenceControlArtifact"
        )
    _safe_key(value.key, field="control.key")
    _version_id(value.version_id, field="control.version_id")
    record = _parse_raw(value.raw, label="control")
    record = _validate_record_intrinsic(record)
    file_sha = _sha(value.raw)
    body_sha = _digest(
        record["fence_body_sha256"],
        field="control.fence_body_sha256",
    )
    observed_file_sha = _digest(
        value.file_sha256,
        field="control.file_sha256",
    )
    observed_body_sha = _digest(
        value.body_sha256,
        field="control.body_sha256",
    )
    observed_checksum = _exact_string(
        value.checksum_sha256_base64,
        field="control.checksum_sha256_base64",
    )
    observed_checksum_type = _exact_string(
        value.checksum_type,
        field="control.checksum_type",
    )
    observed_content_type = _exact_string(
        value.content_type,
        field="control.content_type",
    )
    expected_checksum = base64.b64encode(bytes.fromhex(file_sha)).decode("ascii")
    _exact_integer(value.content_length, field="control.content_length")
    _exact_string(value.etag, field="control.etag")
    _parse_time(value.last_modified, field="control.last_modified")
    if (
        observed_file_sha != file_sha
        or observed_body_sha != body_sha
        or value.content_length != len(value.raw)
        or observed_checksum != expected_checksum
        or observed_checksum_type != "FULL_OBJECT"
        or observed_content_type != "application/json"
        or type(value.missing_meta) is not int
        or value.missing_meta != 0
    ):
        raise FenceAuthorityError("control transport identity mismatch")
    metadata = _metadata(
        value.metadata,
        run_id=record["run_id"],  # type: ignore[arg-type]
        field="control.metadata",
    )
    expected_metadata = tuple(
        sorted(
            (
                ("glm52-body-sha256", body_sha),
                (
                    "glm52-campaign-identity-sha256",
                    record["campaign_identity_sha256"],
                ),
                ("glm52-file-sha256", file_sha),
                ("glm52-record-type", record["record_type"]),
                ("glm52-run-id", record["run_id"]),
            )
        )
    )
    if metadata != expected_metadata:
        raise FenceAuthorityError("control metadata contract mismatch")
    key_field = "genesis_key" if record["record_type"] == _GENESIS_TYPE else "successor_key"
    if value.key != record[key_field]:
        raise FenceAuthorityError("control key/body mismatch")
    return value, record


def _live_state_from_control(control: FenceControlArtifact) -> FenceLiveStateIdentity:
    record = _parse_raw(control.raw, label="control state")
    bucket = _exact_string(record.get("bucket"), field="control state bucket")
    return _live_state_from_json(
        record.get("live_state"),
        bucket=bucket,
        field="control state live_state",
    )


def _require_live_transition(
    *,
    predecessor: FenceLiveStateIdentity,
    current: FenceLiveStateIdentity,
    added_sources: tuple[SourceEnrollment, ...],
    ancestor_states: tuple[FenceLiveStateIdentity, ...],
) -> None:
    unchanged = (
        "account_id",
        "region",
        "bucket",
        "bucket_versioning_status",
        "bucket_mfa_delete_status",
        "lifecycle_configuration_state",
        "lifecycle_configuration_sha256",
        "replication_configuration_state",
        "replication_configuration_sha256",
        "cloudformation_stack_id",
        "executor_identity_sha256",
    )
    if any(getattr(current, name) != getattr(predecessor, name) for name in unchanged):
        raise FenceAuthorityError("successor changed an immutable live-state field")
    if current.bucket_policy_sha256 == predecessor.bucket_policy_sha256:
        raise FenceAuthorityError("successor must advance bucket policy identity")
    if bool(added_sources) != (
        current.publisher_deny_policy_sha256
        != predecessor.publisher_deny_policy_sha256
    ):
        raise FenceAuthorityError(
            "successor publisher-deny transition does not match source delta"
        )
    ready_additions = tuple(
        source
        for source in added_sources
        if source.source_kind == "production-control-plane-readiness"
    )
    if not ready_additions:
        for name in (
            "cloudformation_template_sha256",
            "cloudformation_parameters_sha256",
            "h1d_control_plane_ready_body_sha256",
        ):
            if getattr(current, name) != getattr(predecessor, name):
                raise FenceAuthorityError(
                    "successor readiness identities changed without readiness source"
                )
    else:
        if (
            len(ready_additions) != 1
            or current.h1d_control_plane_ready_body_sha256
            != ready_additions[0].body_sha256
        ):
            raise FenceAuthorityError(
                "successor readiness transition does not bind added source"
            )
    for name in _ROLLBACK_GUARDED_FIELDS:
        before = getattr(predecessor, name)
        after = getattr(current, name)
        if after != before and any(
            after == getattr(ancestor, name)
            for ancestor in ancestor_states[:-1]
        ):
            raise FenceAuthorityError(
                f"successor {name} rolls back to an ancestor"
            )


def _validate_successor_against_head(
    value: Mapping[str, object],
    *,
    head: ModeledFenceHead,
    context: _AuthenticatedSources,
    ancestor_states: tuple[FenceLiveStateIdentity, ...],
) -> dict[str, object]:
    record = _validate_successor_intrinsic(value)
    _require_context_common(record, context)
    predecessor = head.head
    if (
        record["successor_key"]
        != _fence_successor_key(context.run_id, predecessor.body_sha256)
        or record["predecessor_control_key"] != predecessor.key
        or record["predecessor_control_version_id"] != predecessor.version_id
        or record["predecessor_control_file_sha256"] != predecessor.file_sha256
        or record["predecessor_control_body_sha256"] != predecessor.body_sha256
    ):
        raise FenceAuthorityError("successor predecessor identity mismatch")
    sources = _source_tuple(
        record["enrolled_sources"],
        run_id=context.run_id,
        field="enrolled_sources",
    )
    added_sources = _source_tuple(
        record["added_sources"],
        run_id=context.run_id,
        field="added_sources",
    )
    if any(
        added in head.enrolled_sources
        or any(added.key == existing.key for existing in head.enrolled_sources)
        for added in added_sources
    ):
        raise FenceAuthorityError("successor repeats an enrolled source")
    expected_sources = tuple(
        sorted(
            head.enrolled_sources + added_sources,
            key=lambda item: (item.source_kind, item.key),
        )
    )
    if sources != expected_sources:
        raise FenceAuthorityError(
            "successor did not preserve predecessor sources byte-for-byte"
        )
    reservations = _reservation_tuple(
        record["reserved_generations"],
        run_id=context.run_id,
        field="reserved_generations",
    )
    added_value = record["added_generation_reservations"]
    if type(added_value) is not list:
        raise FenceAuthorityError(
            "added_generation_reservations must be an exact list"
        )
    added_reservations = tuple(
        _reservation_from_json(
            item,
            run_id=context.run_id,
            field=f"added_generation_reservations[{index}]",
        )
        for index, item in enumerate(added_value)
    )
    if added_reservations:
        next_generation = len(head.reserved_generations) + 1
        if (
            len(added_reservations) != 1
            or added_reservations[0].generation != next_generation
        ):
            raise FenceAuthorityError(
                "successor must add only the next contiguous generation"
            )
    if reservations != head.reserved_generations + added_reservations:
        raise FenceAuthorityError(
            "successor did not preserve predecessor reservations byte-for-byte"
        )
    current_live = _live_state_from_json(
        record["live_state"],
        bucket=context.bucket,
        field="live_state",
    )
    predecessor_live = ancestor_states[-1]
    _require_live_transition(
        predecessor=predecessor_live,
        current=current_live,
        added_sources=added_sources,
        ancestor_states=ancestor_states,
    )
    return record


def _fence_genesis_key(run_id: str) -> str:
    return f"campaigns/{run_id}/authorities/fence/FENCE_GENESIS.json"


def _fence_successor_key(run_id: str, predecessor_body_sha256: str) -> str:
    return (
        f"campaigns/{run_id}/authorities/fence/successors/"
        f"{predecessor_body_sha256}/FENCE_SUCCESSOR.json"
    )


def fence_genesis_s3_key(*, run_id: str) -> str:
    """Return the one canonical genesis coordinate for a validated run."""

    return _translate(lambda: _fence_genesis_key(_run_id(run_id)))


def fence_successor_s3_key(
    *,
    run_id: str,
    predecessor_body_sha256: str,
) -> str:
    """Return the singleton successor coordinate derived from a predecessor."""

    return _translate(
        lambda: _fence_successor_key(
            _run_id(run_id),
            _digest(
                predecessor_body_sha256,
                field="predecessor_body_sha256",
            ),
        )
    )


def fence_control_file_bytes(value: Mapping[str, object]) -> bytes:
    """Return strict canonical bytes for one intrinsically valid control."""

    return _translate(lambda: _canonical(_validate_record_intrinsic(value)) + b"\n")


def fence_control_file_sha256(value: Mapping[str, object]) -> str:
    """Hash the complete canonical control file, including its final LF."""

    return _translate(lambda: _sha(fence_control_file_bytes(value)))


def build_fence_genesis(
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
    source_enrollments: tuple[SourceEnrollment, ...],
    generation_one_reservation: GenerationReservation,
    live_state: FenceLiveStateIdentity,
    created_at: Union[datetime, str],
) -> dict[str, object]:
    """Build one closed, self-hashed canonical genesis candidate."""

    def operation() -> dict[str, object]:
        context = _authenticate_sources(
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            gpu_spend_snapshot=gpu_spend_snapshot,
        )
        if type(source_enrollments) is not tuple:
            raise FenceAuthorityError(
                "source_enrollments must be an exact tuple"
            )
        sources = tuple(
            _validate_source(
                source,
                run_id=context.run_id,
                field=f"source_enrollments[{index}]",
            )
            for index, source in enumerate(source_enrollments)
        )
        if sources != tuple(
            sorted(sources, key=lambda item: (item.source_kind, item.key))
        ):
            raise FenceAuthorityError("source_enrollments must be sorted")
        _require_genesis_sources(sources, context)
        reservation = _validate_reservation(
            generation_one_reservation,
            run_id=context.run_id,
            field="generation_one_reservation",
        )
        if reservation.generation != 1:
            raise FenceAuthorityError("genesis must reserve only generation 1")
        exact_live = _validate_live_state(
            live_state,
            bucket=context.bucket,
            field="live_state",
        )
        if (
            exact_live.h1d_control_plane_ready_body_sha256
            != context.ready_body_sha256
        ):
            raise FenceAuthorityError(
                "genesis live state does not bind authenticated readiness"
            )
        body: dict[str, object] = {
            "schema_version": 1,
            "record_type": _GENESIS_TYPE,
            "account_id": _ACCOUNT_ID,
            "region": _REGION,
            "bucket": context.bucket,
            "run_id": context.run_id,
            "campaign_identity_sha256": context.campaign_identity_sha256,
            "genesis_key": _fence_genesis_key(context.run_id),
            "enrolled_sources": [
                _source_json(source, run_id=context.run_id)
                for source in sources
            ],
            "reserved_generations": [
                _reservation_json(reservation, run_id=context.run_id)
            ],
            "live_state": _live_state_json(
                exact_live,
                bucket=context.bucket,
            ),
            "created_at": _normalize_time(created_at, field="created_at"),
        }
        result = {
            **body,
            "fence_body_sha256": _sha(_canonical(body)),
        }
        return _validate_genesis_external(result, context=context)

    return _translate(operation)


def validate_fence_genesis(
    value: Mapping[str, object],
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
) -> dict[str, object]:
    """Validate one genesis against all seven exact source artifacts."""

    def operation() -> dict[str, object]:
        context = _authenticate_sources(
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            gpu_spend_snapshot=gpu_spend_snapshot,
        )
        return _validate_genesis_external(value, context=context)

    return _translate(operation)


def _walk(
    *,
    run_id: str,
    controls: tuple[FenceControlArtifact, ...],
    context: _AuthenticatedSources,
) -> ModeledFenceHead:
    run = _run_id(run_id)
    if run != context.run_id:
        raise FenceAuthorityError("walk run_id does not match source campaign")
    if type(controls) is not tuple or not controls:
        raise FenceAuthorityError(
            "controls must be a nonempty exact tuple containing genesis"
        )
    parsed = tuple(_validate_control_artifact(item) for item in controls)
    keys = [artifact.key for artifact, _ in parsed]
    body_digests = [artifact.body_sha256 for artifact, _ in parsed]
    if len(keys) != len(set(keys)):
        raise FenceAuthorityError("fence controls contain a sibling or duplicate key")
    if len(body_digests) != len(set(body_digests)):
        raise FenceAuthorityError("fence controls contain a repeated body digest")
    genesis_rows = tuple(
        (artifact, record)
        for artifact, record in parsed
        if record["record_type"] == _GENESIS_TYPE
    )
    if len(genesis_rows) != 1:
        raise FenceAuthorityError("modeled chain requires exactly one genesis")
    genesis_artifact, genesis_record = genesis_rows[0]
    if genesis_artifact.key != _fence_genesis_key(run):
        raise FenceAuthorityError("genesis is at an alternate key")
    genesis_record = _validate_genesis_external(
        genesis_record,
        context=context,
    )
    sources = _source_tuple(
        genesis_record["enrolled_sources"],
        run_id=run,
        field="enrolled_sources",
        genesis=True,
    )
    reservations = _reservation_tuple(
        genesis_record["reserved_generations"],
        run_id=run,
        field="reserved_generations",
    )
    head = ModeledFenceHead(
        genesis=genesis_artifact,
        successors=(),
        head=genesis_artifact,
        enrolled_sources=sources,
        reserved_generations=reservations,
        next_successor_key=_fence_successor_key(
            run,
            genesis_artifact.body_sha256,
        ),
    )
    by_key = {artifact.key: (artifact, record) for artifact, record in parsed}
    reached = {genesis_artifact.key}
    ancestor_states = (_live_state_from_control(genesis_artifact),)
    while head.next_successor_key in by_key:
        artifact, candidate = by_key[head.next_successor_key]
        if artifact.key in reached:
            raise FenceAuthorityError("fence chain contains a cycle")
        candidate = _validate_successor_against_head(
            candidate,
            head=head,
            context=context,
            ancestor_states=ancestor_states,
        )
        sources = _source_tuple(
            candidate["enrolled_sources"],
            run_id=run,
            field="enrolled_sources",
        )
        reservations = _reservation_tuple(
            candidate["reserved_generations"],
            run_id=run,
            field="reserved_generations",
        )
        reached.add(artifact.key)
        ancestor_states = ancestor_states + (_live_state_from_control(artifact),)
        head = ModeledFenceHead(
            genesis=genesis_artifact,
            successors=head.successors + (artifact,),
            head=artifact,
            enrolled_sources=sources,
            reserved_generations=reservations,
            next_successor_key=_fence_successor_key(
                run,
                artifact.body_sha256,
            ),
        )
    if reached != set(keys):
        raise FenceAuthorityError(
            "fence controls contain an orphan, alternate key, fork, or unreachable record"
        )
    return head


def walk_modeled_fence_chain(
    *,
    run_id: str,
    controls: tuple[FenceControlArtifact, ...],
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
) -> ModeledFenceHead:
    """Walk every control from canonical genesis without selection heuristics."""

    def operation() -> ModeledFenceHead:
        context = _authenticate_sources(
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            gpu_spend_snapshot=gpu_spend_snapshot,
        )
        return _walk(run_id=run_id, controls=controls, context=context)

    return _translate(operation)


def build_fence_successor(
    *,
    predecessor_controls: tuple[FenceControlArtifact, ...],
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
    added_sources: tuple[SourceEnrollment, ...],
    added_generation_reservations: tuple[GenerationReservation, ...],
    live_state: FenceLiveStateIdentity,
    selected_at: Union[datetime, str],
) -> dict[str, object]:
    """Build one candidate only after rewalking the complete predecessor chain."""

    def operation() -> dict[str, object]:
        context = _authenticate_sources(
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            gpu_spend_snapshot=gpu_spend_snapshot,
        )
        head = _walk(
            run_id=context.run_id,
            controls=predecessor_controls,
            context=context,
        )
        if type(added_sources) is not tuple:
            raise FenceAuthorityError("added_sources must be an exact tuple")
        exact_sources = tuple(
            _validate_source(
                source,
                run_id=context.run_id,
                field=f"added_sources[{index}]",
            )
            for index, source in enumerate(added_sources)
        )
        if exact_sources != tuple(
            sorted(exact_sources, key=lambda item: (item.source_kind, item.key))
        ):
            raise FenceAuthorityError("added_sources must be sorted")
        if len({item.source_kind for item in exact_sources}) != len(exact_sources):
            raise FenceAuthorityError(
                "one successor may add at most one source of each kind"
            )
        if type(added_generation_reservations) is not tuple:
            raise FenceAuthorityError(
                "added_generation_reservations must be an exact tuple"
            )
        exact_reservations = tuple(
            _validate_reservation(
                item,
                run_id=context.run_id,
                field=f"added_generation_reservations[{index}]",
            )
            for index, item in enumerate(added_generation_reservations)
        )
        if not exact_sources and not exact_reservations:
            raise FenceAuthorityError("successor delta must be nonempty")
        complete_sources = tuple(
            sorted(
                head.enrolled_sources + exact_sources,
                key=lambda item: (item.source_kind, item.key),
            )
        )
        complete_reservations = head.reserved_generations + exact_reservations
        predecessor = head.head
        body: dict[str, object] = {
            "schema_version": 1,
            "record_type": _SUCCESSOR_TYPE,
            "account_id": _ACCOUNT_ID,
            "region": _REGION,
            "bucket": context.bucket,
            "run_id": context.run_id,
            "campaign_identity_sha256": context.campaign_identity_sha256,
            "successor_key": _fence_successor_key(
                context.run_id,
                predecessor.body_sha256,
            ),
            "predecessor_control_key": predecessor.key,
            "predecessor_control_version_id": predecessor.version_id,
            "predecessor_control_file_sha256": predecessor.file_sha256,
            "predecessor_control_body_sha256": predecessor.body_sha256,
            "enrolled_sources": [
                _source_json(item, run_id=context.run_id)
                for item in complete_sources
            ],
            "reserved_generations": [
                _reservation_json(item, run_id=context.run_id)
                for item in complete_reservations
            ],
            "added_sources": [
                _source_json(item, run_id=context.run_id)
                for item in exact_sources
            ],
            "added_generation_reservations": [
                _reservation_json(item, run_id=context.run_id)
                for item in exact_reservations
            ],
            "live_state": _live_state_json(live_state, bucket=context.bucket),
            "selected_at": _normalize_time(selected_at, field="selected_at"),
        }
        result = {
            **body,
            "fence_body_sha256": _sha(_canonical(body)),
        }
        states = tuple(
            _live_state_from_control(control)
            for control in (head.genesis,) + head.successors
        )
        return _validate_successor_against_head(
            result,
            head=head,
            context=context,
            ancestor_states=states,
        )

    return _translate(operation)


def validate_fence_successor(
    value: Mapping[str, object],
    *,
    predecessor_controls: tuple[FenceControlArtifact, ...],
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    approval: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    submission_acquisition: VersionedJsonArtifact,
    gpu_spend_snapshot: VersionedJsonArtifact,
) -> dict[str, object]:
    """Validate one candidate after rewalking the complete predecessor chain."""

    def operation() -> dict[str, object]:
        context = _authenticate_sources(
            descriptor=descriptor,
            intent=intent,
            approval=approval,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            submission_acquisition=submission_acquisition,
            gpu_spend_snapshot=gpu_spend_snapshot,
        )
        head = _walk(
            run_id=context.run_id,
            controls=predecessor_controls,
            context=context,
        )
        states = tuple(
            _live_state_from_control(control)
            for control in (head.genesis,) + head.successors
        )
        return _validate_successor_against_head(
            value,
            head=head,
            context=context,
            ancestor_states=states,
        )

    return _translate(operation)


__all__ = [
    "FenceAuthorityError",
    "FenceControlArtifact",
    "FenceLiveStateIdentity",
    "GenerationReservation",
    "ModeledFenceHead",
    "SourceEnrollment",
    "build_fence_genesis",
    "build_fence_successor",
    "fence_control_file_bytes",
    "fence_control_file_sha256",
    "fence_genesis_s3_key",
    "fence_successor_s3_key",
    "validate_fence_genesis",
    "validate_fence_successor",
    "walk_modeled_fence_chain",
]
