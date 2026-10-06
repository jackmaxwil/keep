"""Enforcement-native version-pinned authority for one production Sky job."""

# ruff: noqa: UP017

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
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
    _campaign_module = _flat_module("glm52_sky_campaign")
    _production_module = _flat_module("glm52_sky_production_submission")
    _modes_module = _flat_module("glm52_sky_submission_modes")
except _FlatModuleUnavailable:
    from .glm52_sky_campaign import (
        validate_sky_campaign_descriptor,
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
        expected_sky_job_name,
        record_contract,
        require_opaque_version_id,
    )
else:
    validate_sky_campaign_descriptor = (
        _campaign_module.validate_sky_campaign_descriptor
    )
    VersionedJsonArtifact = _production_module.VersionedJsonArtifact
    production_submission_intent_file_bytes = (
        _production_module.production_submission_intent_file_bytes
    )
    production_submission_intent_file_sha256 = (
        _production_module.production_submission_intent_file_sha256
    )
    production_submission_intent_s3_key = (
        _production_module.production_submission_intent_s3_key
    )
    validate_production_submission_intent = (
        _production_module.validate_production_submission_intent
    )
    SubmissionModeContractError = _modes_module.SubmissionModeContractError
    expected_sky_job_name = _modes_module.expected_sky_job_name
    record_contract = _modes_module.record_contract
    require_opaque_version_id = _modes_module.require_opaque_version_id


class ProductionPrelaunchAuthorityError(ValueError):
    """Raised when pure production prelaunch authority fails closed."""


_ACCOUNT_ID = "246813579024"
_REGION = "us-west-2"
_MODE = "production"
_WORKSPACE = "default"
_COORDINATOR_MODE = "production-dynamic-job-binding-active"
_MAX_AGE = timedelta(seconds=60)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_INSTANCE_ID = re.compile(r"i-(?:[0-9a-f]{8}|[0-9a-f]{17})")
_INSTANCE_TYPE = re.compile(r"[a-z0-9][a-z0-9.]*")
_CONTROLLER_CLUSTER = re.compile(r"sky-jobs-controller-[A-Za-z0-9._-]+")
_ROLE_ARN = re.compile(
    r"arn:aws:iam::246813579024:role/[A-Za-z0-9+=,.@_/-]+"
)

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
_HISTORY_FIELDS = {
    "sky_job_id",
    "sky_job_name",
    "workspace",
    "controller_submitted_at",
    "controller_status",
    "controller_identity",
}

_BASELINE_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "sky_job_name",
    "workspace",
    "controller_instance_id",
    "controller_instance_type",
    "controller_identity",
    "controller_profile_arn",
    "controller_cluster_name",
    "ssm_ping_status",
    "exact_name_history",
    "active_exact_name_job_ids",
    "active_tagged_p5_instance_ids",
    "observed_at",
}
_BASELINE_FIELDS = _BASELINE_BODY_FIELDS | {"baseline_body_sha256"}

_DEPLOYMENT_IDENTITY_FIELDS = {
    "stack_id",
    "template_sha256",
    "lambda_function_arn",
    "lambda_code_s3_key",
    "lambda_code_sha256",
    "lambda_code_version_id",
    "iam_policy_sha256",
    "reconciliation_rule_arn",
    "deadline_schedule_arn",
    "dlq_arn",
    "activation_capabilities",
}
_ACTIVATION_FIELDS = {
    "stack_status",
    "stack_operation_in_progress",
    "foundation_revision",
    "managed_mode",
    "run_id",
    "bucket",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "sky_job_name",
    "must_start_by",
    "primary_wake_at",
    "lambda_state",
    "lambda_last_update_status",
    "lambda_qualified_arn",
    "lambda_execution_role_arn",
    "lambda_code_sha256",
    "lambda_code_version_id",
    "lambda_environment",
    "reconciliation_schedule_expression",
    "reconciliation_target_arn",
    "reconciliation_target_id",
    "reconciliation_input",
    "reconciliation_retry_max_event_age_seconds",
    "reconciliation_retry_max_attempts",
    "reconciliation_dlq_arn",
    "deadline_schedule_expression",
    "deadline_schedule_timezone",
    "deadline_flexible_window_mode",
    "deadline_target_arn",
    "deadline_target_role_arn",
    "deadline_input",
    "deadline_retry_max_event_age_seconds",
    "deadline_retry_max_attempts",
    "deadline_dlq_arn",
}
_ENVIRONMENT_FIELDS = {
    "EXPECTED_ACCOUNT_ID",
    "CAMPAIGN_BUCKET",
    "CAMPAIGN_RUN_ID",
    "MANAGED_MODE",
    "CAMPAIGN_DESCRIPTOR_KEY",
    "CAMPAIGN_DESCRIPTOR_VERSION_ID",
    "IMMUTABLE_SUBMISSION_KEY",
    "IMMUTABLE_SUBMISSION_VERSION_ID",
    "SUBMISSION_BODY_SHA256",
    "SKY_JOB_NAME",
    "MUST_START_BY",
    "COORDINATOR_MODE",
}
_TRIGGER_FIELDS = {
    "campaign_run_id",
    "managed_mode",
    "sky_job_name",
    "must_start_by",
    "descriptor_key",
    "descriptor_version_id",
    "intent_key",
    "intent_version_id",
    "trigger",
}

_READY_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
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
    "stack_id",
    "template_sha256",
    "lambda_function_arn",
    "lambda_code_s3_key",
    "lambda_code_sha256",
    "lambda_code_version_id",
    "iam_policy_sha256",
    "reconciliation_rule_arn",
    "reconciliation_rule_state",
    "deadline_schedule_arn",
    "deadline_schedule_state",
    "dlq_arn",
    "coordinator_mode",
    "activation_capabilities",
    "observed_at",
}
_READY_FIELDS = _READY_BODY_FIELDS | {"control_plane_ready_body_sha256"}

_ACQUISITION_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
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
    "sky_job_name",
    "must_start_by",
    "spend_snapshot_observed_at",
    "controller_baseline_observed_at",
    "control_plane_ready_observed_at",
    "acquired_at",
}
_ACQUISITION_FIELDS = _ACQUISITION_BODY_FIELDS | {"acquisition_body_sha256"}


def _translate(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except ProductionPrelaunchAuthorityError:
        raise
    except SubmissionModeContractError as error:
        raise ProductionPrelaunchAuthorityError(str(error)) from error
    except Exception as error:
        raise ProductionPrelaunchAuthorityError(str(error)) from error


def _require_exact_json(value: object, *, field: str) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ProductionPrelaunchAuthorityError(f"{field} must be finite")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _require_exact_json(item, field=f"{field}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ProductionPrelaunchAuthorityError(
                    f"{field} keys must be exact strings"
                )
            _require_exact_json(item, field=f"{field}.{key}")
        return
    raise ProductionPrelaunchAuthorityError(f"{field} is not exact JSON")


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


def _duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProductionPrelaunchAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProductionPrelaunchAuthorityError(
        f"nonfinite JSON number is forbidden: {value}"
    )


def _parse_raw(raw: object, *, label: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise ProductionPrelaunchAuthorityError(f"{label}.raw must be exact bytes")
    try:
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicates,
            parse_constant=_reject_constant,
        )
    except ProductionPrelaunchAuthorityError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ProductionPrelaunchAuthorityError(
            f"{label} is not strict JSON"
        ) from error
    if type(parsed) is not dict:
        raise ProductionPrelaunchAuthorityError(
            f"{label} root must be an exact object"
        )
    _require_exact_json(parsed, field=label)
    if raw != _canonical(parsed) + b"\n":
        raise ProductionPrelaunchAuthorityError(
            f"{label} bytes are not canonical JSON plus LF"
        )
    return parsed


def _exact_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ProductionPrelaunchAuthorityError(f"{field} must be an exact string")
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
        raise ProductionPrelaunchAuthorityError(f"{field} is not a safe key")
    segments = key.split("/")
    if any(not segment or segment in {".", ".."} for segment in segments):
        raise ProductionPrelaunchAuthorityError(f"{field} is not a safe key")
    return key


def _digest(value: object, *, field: str) -> str:
    digest = _exact_string(value, field=field)
    if _SHA256.fullmatch(digest) is None:
        raise ProductionPrelaunchAuthorityError(
            f"{field} must be lowercase SHA-256"
        )
    return digest


def _run_id(value: object) -> str:
    run_id = _exact_string(value, field="run_id")
    if _RUN_ID.fullmatch(run_id) is None:
        raise ProductionPrelaunchAuthorityError("run_id is invalid")
    return run_id


def _bucket(value: object) -> str:
    bucket = _exact_string(value, field="bucket")
    if _BUCKET.fullmatch(bucket) is None:
        raise ProductionPrelaunchAuthorityError("bucket is invalid")
    return bucket


def _exact_integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProductionPrelaunchAuthorityError(
            f"{field} must be an exact integer"
        )
    return value


def _exact_float(value: object, *, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ProductionPrelaunchAuthorityError(
            f"{field} must be an exact finite float"
        )
    return value


def _time(
    value: object,
    *,
    field: str,
    permit_datetime: bool,
) -> tuple[str, datetime]:
    if permit_datetime and type(value) is datetime:
        if value.tzinfo is None or value.microsecond != 0:
            raise ProductionPrelaunchAuthorityError(
                f"{field} must be whole-second timezone-aware"
            )
        try:
            utc_offset = value.utcoffset()
        except Exception as error:
            raise ProductionPrelaunchAuthorityError(
                f"{field} UTC offset lookup failed"
            ) from error
        if utc_offset is None:
            raise ProductionPrelaunchAuthorityError(
                f"{field} must have a real UTC offset"
            )
        try:
            parsed = (
                value.replace(tzinfo=None) - utc_offset
            ).replace(tzinfo=timezone.utc)
        except Exception as error:
            raise ProductionPrelaunchAuthorityError(
                f"{field} UTC normalization failed"
            ) from error
        if parsed.microsecond != 0:
            raise ProductionPrelaunchAuthorityError(
                f"{field} normalized UTC instant must be whole-second"
            )
        rendered = parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
        return rendered, parsed
    if type(value) is not str or not value.endswith("Z"):
        raise ProductionPrelaunchAuthorityError(
            f"{field} must be canonical UTC Z text"
        )
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except (ValueError, OverflowError) as error:
        raise ProductionPrelaunchAuthorityError(f"{field} is invalid") from error
    rendered = parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if parsed.microsecond != 0 or rendered != value:
        raise ProductionPrelaunchAuthorityError(
            f"{field} must be whole-second canonical UTC text"
        )
    return rendered, parsed.astimezone(timezone.utc)


def _exact_fields(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ProductionPrelaunchAuthorityError(f"{label} schema mismatch")
    _require_exact_json(value, field=label)
    return value


def _body_sha(value: Mapping[str, object], digest_field: str) -> str:
    body = dict(value)
    body.pop(digest_field)
    return _sha(_canonical(body))


def _self_hashed(
    value: object,
    *,
    fields: set[str],
    digest_field: str,
    label: str,
) -> dict[str, object]:
    record = _exact_fields(value, fields=fields, label=label)
    if _digest(record.get(digest_field), field=digest_field) != _body_sha(
        record,
        digest_field,
    ):
        raise ProductionPrelaunchAuthorityError(f"{label} body SHA-256 mismatch")
    return record


def _contract(
    *,
    record_kind: str,
    schema_version: int,
    record_type: str,
    digest_field: str,
    address_kind: str,
) -> None:
    contract = record_contract(
        managed_mode=_MODE,
        record_kind=record_kind,
    )
    if (
        type(contract.schema_version) is not int
        or contract.schema_version != schema_version
        or type(contract.record_type) is not str
        or contract.record_type != record_type
        or type(contract.digest_field) is not str
        or contract.digest_field != digest_field
        or contract.address_kind != address_kind
    ):
        raise ProductionPrelaunchAuthorityError(
            f"production {record_kind} registry contract drift"
        )


def _artifact(
    value: object,
    *,
    label: str,
) -> tuple[VersionedJsonArtifact, dict[str, object], str]:
    if type(value) is not VersionedJsonArtifact:
        raise ProductionPrelaunchAuthorityError(
            f"{label} must be an exact VersionedJsonArtifact"
        )
    key = _safe_key(value.key, field=f"{label}.key")
    version_id = require_opaque_version_id(
        value.version_id,
        field=f"{label}.version_id",
    )
    if type(value.raw) is not bytes:
        raise ProductionPrelaunchAuthorityError(
            f"{label}.raw must be exact bytes"
        )
    raw_sha = _sha(value.raw)
    parsed = _parse_raw(value.raw, label=label)
    return (
        VersionedJsonArtifact(
            key=key,
            raw=value.raw,
            version_id=version_id,
        ),
        parsed,
        raw_sha,
    )


def _descriptor_exactness(value: dict[str, object]) -> None:
    expected: tuple[tuple[str, type, object], ...] = (
        ("schema_version", int, 2),
        ("record_type", str, "glm52_sky_campaign_descriptor_v2"),
        ("account_id", str, _ACCOUNT_ID),
        ("provider", str, "aws"),
        ("region", str, _REGION),
        ("instance_type", str, "p5.48xlarge"),
        ("instance_count", int, 1),
        ("use_spot", bool, False),
        ("max_hourly_cost_usd", float, 55.04),
        ("approved_gpu_runtime_seconds", int, 86_400),
        ("approved_gpu_cost_usd", float, 1_320.96),
        ("skypilot_version", str, "0.13.0"),
        ("task_name", str, "glm52-campaign"),
    )
    for field, expected_type, expected_value in expected:
        if type(value.get(field)) is not expected_type or value[field] != expected_value:
            raise ProductionPrelaunchAuthorityError(
                f"campaign descriptor {field} is not exact"
            )


def _authenticate_sources(
    *,
    descriptor: object,
    intent: object,
) -> tuple[
    VersionedJsonArtifact,
    VersionedJsonArtifact,
    dict[str, object],
    dict[str, object],
]:
    descriptor_artifact, descriptor_value, descriptor_file_sha = _artifact(
        descriptor,
        label="descriptor",
    )
    intent_artifact, intent_value, intent_file_sha = _artifact(
        intent,
        label="production intent",
    )
    try:
        descriptor_value = validate_sky_campaign_descriptor(descriptor_value)
        intent_value = validate_production_submission_intent(intent_value)
    except Exception as error:
        raise ProductionPrelaunchAuthorityError(str(error)) from error
    if type(descriptor_value) is not dict or type(intent_value) is not dict:
        raise ProductionPrelaunchAuthorityError(
            "upstream validators must return exact objects"
        )
    _descriptor_exactness(descriptor_value)
    if (
        type(intent_value.get("schema_version")) is not int
        or intent_value["schema_version"] != 1
        or type(intent_value.get("record_type")) is not str
        or intent_value["record_type"]
        != "glm52_sky_production_submission_intent_v1"
        or type(intent_value.get("managed_mode")) is not str
        or intent_value["managed_mode"] != _MODE
    ):
        raise ProductionPrelaunchAuthorityError("production intent schema mismatch")
    run_id = _run_id(descriptor_value.get("run_id"))
    descriptor_key = _safe_key(
        descriptor_value.get("campaign_descriptor_key"),
        field="campaign_descriptor_key",
    )
    if descriptor_artifact.key != descriptor_key:
        raise ProductionPrelaunchAuthorityError(
            "descriptor key does not equal its canonical top-level coordinate"
        )
    expected_intent_key = production_submission_intent_s3_key(
        run_id=run_id,
        intent_body_sha256=_digest(
            intent_value.get("intent_body_sha256"),
            field="intent_body_sha256",
        ),
    )
    if intent_artifact.key != expected_intent_key:
        raise ProductionPrelaunchAuthorityError(
            "production intent key is not canonical"
        )
    if (
        intent_artifact.raw != production_submission_intent_file_bytes(intent_value)
        or intent_file_sha
        != production_submission_intent_file_sha256(intent_value)
    ):
        raise ProductionPrelaunchAuthorityError(
            "production intent file identity mismatch"
        )
    shared = {
        "account_id": descriptor_value["account_id"],
        "region": descriptor_value["region"],
        "bucket": descriptor_value["bucket"],
        "run_id": descriptor_value["run_id"],
        "campaign_identity_sha256": descriptor_value[
            "campaign_identity_sha256"
        ],
        "descriptor_key": descriptor_key,
        "descriptor_file_sha256": descriptor_file_sha,
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "descriptor_version_id": descriptor_artifact.version_id,
        "must_start_by": descriptor_value["must_start_by"],
    }
    for field, expected in shared.items():
        if type(intent_value.get(field)) is not type(expected) or (
            intent_value[field] != expected
        ):
            raise ProductionPrelaunchAuthorityError(
                f"descriptor/production intent {field} drift"
            )
    if (
        _bucket(intent_value.get("bucket")) != _bucket(descriptor_value["bucket"])
        or intent_value.get("sky_job_name")
        != expected_sky_job_name(run_id=run_id, managed_mode=_MODE)
    ):
        raise ProductionPrelaunchAuthorityError(
            "production intent job or bucket identity mismatch"
        )
    _time(intent_value.get("spend_snapshot_observed_at"), field="spend_snapshot_observed_at", permit_datetime=False)
    _time(intent_value.get("intent_at"), field="intent_at", permit_datetime=False)
    _time(intent_value.get("must_start_by"), field="must_start_by", permit_datetime=False)
    return (
        descriptor_artifact,
        intent_artifact,
        descriptor_value,
        intent_value,
    )


def _controller_profile(controller_identity: object) -> tuple[str, str]:
    role = _exact_string(controller_identity, field="controller_identity")
    if _ROLE_ARN.fullmatch(role) is None:
        raise ProductionPrelaunchAuthorityError(
            "descriptor controller_identity is invalid"
        )
    return role, role.replace(":role/", ":instance-profile/", 1)


def _controller_history(
    value: object,
    *,
    sky_job_name: str,
    controller_identity: str,
    observed_at: datetime,
) -> list[dict[str, object]]:
    if type(value) is not list:
        raise ProductionPrelaunchAuthorityError(
            "exact_name_history must be an exact list"
        )
    result: list[dict[str, object]] = []
    seen: set[int] = set()
    for index, item in enumerate(value):
        row = _exact_fields(
            item,
            fields=_HISTORY_FIELDS,
            label=f"exact_name_history[{index}]",
        )
        job_id = _exact_integer(
            row.get("sky_job_id"),
            field=f"exact_name_history[{index}].sky_job_id",
            minimum=1,
        )
        if job_id in seen:
            raise ProductionPrelaunchAuthorityError(
                "exact_name_history contains a duplicate sky_job_id"
            )
        seen.add(job_id)
        if (
            type(row.get("sky_job_name")) is not str
            or row["sky_job_name"] != sky_job_name
            or type(row.get("workspace")) is not str
            or row["workspace"] != _WORKSPACE
            or type(row.get("controller_identity")) is not str
            or row["controller_identity"] != controller_identity
        ):
            raise ProductionPrelaunchAuthorityError(
                f"exact_name_history[{index}] contains foreign identity"
            )
        submitted_text, submitted_at = _time(
            row.get("controller_submitted_at"),
            field=f"exact_name_history[{index}].controller_submitted_at",
            permit_datetime=False,
        )
        if submitted_at > observed_at:
            raise ProductionPrelaunchAuthorityError(
                f"exact_name_history[{index}] follows baseline observation"
            )
        status = _exact_string(
            row.get("controller_status"),
            field=f"exact_name_history[{index}].controller_status",
        )
        if status not in _CONTROLLER_STATUSES:
            raise ProductionPrelaunchAuthorityError(
                f"exact_name_history[{index}] status is invalid"
            )
        result.append(
            {
                "sky_job_id": job_id,
                "sky_job_name": sky_job_name,
                "workspace": _WORKSPACE,
                "controller_submitted_at": submitted_text,
                "controller_status": status,
                "controller_identity": controller_identity,
            }
        )
    expected = sorted(
        result,
        key=lambda row: (
            int(row["sky_job_id"]),
            str(row["controller_submitted_at"]),
            _canonical(row),
        ),
    )
    if result != expected:
        raise ProductionPrelaunchAuthorityError(
            "exact_name_history is not in canonical sorted order"
        )
    return result


def _empty_active_job_ids(
    value: object,
    *,
    history: list[dict[str, object]],
) -> list[int]:
    if type(value) is not list:
        raise ProductionPrelaunchAuthorityError(
            "active_exact_name_job_ids must be an exact list"
        )
    actual = [
        _exact_integer(item, field="active_exact_name_job_ids", minimum=1)
        for item in value
    ]
    if actual != sorted(actual) or len(actual) != len(set(actual)):
        raise ProductionPrelaunchAuthorityError(
            "active_exact_name_job_ids must be sorted and unique"
        )
    derived = sorted(
        int(row["sky_job_id"])
        for row in history
        if row["controller_status"] in _ACTIVE_CONTROLLER_STATUSES
    )
    if actual != derived:
        raise ProductionPrelaunchAuthorityError(
            "active_exact_name_job_ids do not match active history"
        )
    if actual:
        raise ProductionPrelaunchAuthorityError(
            "controller baseline contains active exact-name jobs"
        )
    return actual


def _empty_p5_ids(value: object) -> list[str]:
    if type(value) is not list:
        raise ProductionPrelaunchAuthorityError(
            "active_tagged_p5_instance_ids must be an exact list"
        )
    result: list[str] = []
    for item in value:
        instance_id = _exact_string(
            item,
            field="active_tagged_p5_instance_ids",
        )
        if _INSTANCE_ID.fullmatch(instance_id) is None:
            raise ProductionPrelaunchAuthorityError(
                "active_tagged_p5_instance_ids contains an invalid instance ID"
            )
        result.append(instance_id)
    if result != sorted(result) or len(result) != len(set(result)):
        raise ProductionPrelaunchAuthorityError(
            "active_tagged_p5_instance_ids must be sorted and unique"
        )
    if result:
        raise ProductionPrelaunchAuthorityError(
            "controller baseline contains active tagged P5 instances"
        )
    return result


def _baseline_body(
    *,
    descriptor: object,
    intent: object,
    controller_instance_id: object,
    controller_instance_type: object,
    controller_profile_arn: object,
    controller_cluster_name: object,
    ssm_ping_status: object,
    exact_name_history: object,
    active_exact_name_job_ids: object,
    active_tagged_p5_instance_ids: object,
    observed_at: object,
) -> dict[str, object]:
    (
        descriptor_artifact,
        intent_artifact,
        descriptor_value,
        intent_value,
    ) = _authenticate_sources(descriptor=descriptor, intent=intent)
    instance_id = _exact_string(
        controller_instance_id,
        field="controller_instance_id",
    )
    if _INSTANCE_ID.fullmatch(instance_id) is None:
        raise ProductionPrelaunchAuthorityError(
            "controller_instance_id is invalid"
        )
    instance_type = _exact_string(
        controller_instance_type,
        field="controller_instance_type",
    )
    if _INSTANCE_TYPE.fullmatch(instance_type) is None:
        raise ProductionPrelaunchAuthorityError(
            "controller_instance_type is invalid"
        )
    controller_identity, expected_profile = _controller_profile(
        descriptor_value.get("controller_identity")
    )
    if (
        type(controller_profile_arn) is not str
        or controller_profile_arn != expected_profile
    ):
        raise ProductionPrelaunchAuthorityError(
            "controller_profile_arn does not match descriptor identity"
        )
    cluster_name = _exact_string(
        controller_cluster_name,
        field="controller_cluster_name",
    )
    if _CONTROLLER_CLUSTER.fullmatch(cluster_name) is None:
        raise ProductionPrelaunchAuthorityError(
            "controller_cluster_name is invalid"
        )
    if type(ssm_ping_status) is not str or ssm_ping_status != "Online":
        raise ProductionPrelaunchAuthorityError("ssm_ping_status must be Online")
    observed_text, observed_time = _time(
        observed_at,
        field="controller baseline observed_at",
        permit_datetime=True,
    )
    _, intent_time = _time(
        intent_value.get("intent_at"),
        field="intent_at",
        permit_datetime=False,
    )
    _, deadline = _time(
        intent_value.get("must_start_by"),
        field="must_start_by",
        permit_datetime=False,
    )
    if not intent_time <= observed_time < deadline:
        raise ProductionPrelaunchAuthorityError(
            "controller baseline observation is outside intent/deadline"
        )
    history = _controller_history(
        exact_name_history,
        sky_job_name=str(intent_value["sky_job_name"]),
        controller_identity=controller_identity,
        observed_at=observed_time,
    )
    active_ids = _empty_active_job_ids(
        active_exact_name_job_ids,
        history=history,
    )
    active_p5_ids = _empty_p5_ids(active_tagged_p5_instance_ids)
    return {
        "schema_version": 1,
        "record_type": "glm52_production_controller_baseline_v1",
        "account_id": intent_value["account_id"],
        "region": intent_value["region"],
        "bucket": _bucket(intent_value["bucket"]),
        "run_id": intent_value["run_id"],
        "managed_mode": _MODE,
        "campaign_identity_sha256": intent_value[
            "campaign_identity_sha256"
        ],
        "descriptor_key": descriptor_artifact.key,
        "descriptor_file_sha256": _sha(descriptor_artifact.raw),
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "descriptor_version_id": descriptor_artifact.version_id,
        "intent_key": intent_artifact.key,
        "intent_file_sha256": _sha(intent_artifact.raw),
        "intent_body_sha256": intent_value["intent_body_sha256"],
        "intent_version_id": intent_artifact.version_id,
        "sky_job_name": intent_value["sky_job_name"],
        "workspace": _WORKSPACE,
        "controller_instance_id": instance_id,
        "controller_instance_type": instance_type,
        "controller_identity": controller_identity,
        "controller_profile_arn": expected_profile,
        "controller_cluster_name": cluster_name,
        "ssm_ping_status": "Online",
        "exact_name_history": history,
        "active_exact_name_job_ids": active_ids,
        "active_tagged_p5_instance_ids": active_p5_ids,
        "observed_at": observed_text,
    }


def _validate_baseline_standalone(value: object) -> dict[str, object]:
    _contract(
        record_kind="controller-baseline",
        schema_version=1,
        record_type="glm52_production_controller_baseline_v1",
        digest_field="baseline_body_sha256",
        address_kind="content-addressed-body",
    )
    record = _self_hashed(
        value,
        fields=_BASELINE_FIELDS,
        digest_field="baseline_body_sha256",
        label="production controller baseline",
    )
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or type(record.get("record_type")) is not str
        or record["record_type"] != "glm52_production_controller_baseline_v1"
        or type(record.get("managed_mode")) is not str
        or record["managed_mode"] != _MODE
        or type(record.get("account_id")) is not str
        or record["account_id"] != _ACCOUNT_ID
        or type(record.get("region")) is not str
        or record["region"] != _REGION
    ):
        raise ProductionPrelaunchAuthorityError(
            "production controller baseline discriminant mismatch"
        )
    run_id = _run_id(record.get("run_id"))
    _bucket(record.get("bucket"))
    for field in (
        "campaign_identity_sha256",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "intent_file_sha256",
        "intent_body_sha256",
        "baseline_body_sha256",
    ):
        _digest(record.get(field), field=field)
    for field in ("descriptor_key", "intent_key"):
        _safe_key(record.get(field), field=field)
    for field in ("descriptor_version_id", "intent_version_id"):
        require_opaque_version_id(record.get(field), field=field)
    if record["intent_key"] != production_submission_intent_s3_key(
        run_id=run_id,
        intent_body_sha256=str(record["intent_body_sha256"]),
    ):
        raise ProductionPrelaunchAuthorityError(
            "production controller baseline intent key is not canonical"
        )
    sky_job_name = expected_sky_job_name(
        run_id=run_id,
        managed_mode=_MODE,
    )
    if (
        type(record.get("sky_job_name")) is not str
        or record["sky_job_name"] != sky_job_name
        or type(record.get("workspace")) is not str
        or record["workspace"] != _WORKSPACE
    ):
        raise ProductionPrelaunchAuthorityError(
            "production controller baseline job/workspace mismatch"
        )
    observed_text, observed_time = _time(
        record.get("observed_at"),
        field="controller baseline observed_at",
        permit_datetime=False,
    )
    controller_identity, expected_profile = _controller_profile(
        record.get("controller_identity")
    )
    instance_id = _exact_string(
        record.get("controller_instance_id"),
        field="controller_instance_id",
    )
    instance_type = _exact_string(
        record.get("controller_instance_type"),
        field="controller_instance_type",
    )
    cluster_name = _exact_string(
        record.get("controller_cluster_name"),
        field="controller_cluster_name",
    )
    if (
        _INSTANCE_ID.fullmatch(instance_id) is None
        or _INSTANCE_TYPE.fullmatch(instance_type) is None
        or _CONTROLLER_CLUSTER.fullmatch(cluster_name) is None
        or type(record.get("controller_profile_arn")) is not str
        or record["controller_profile_arn"] != expected_profile
        or type(record.get("ssm_ping_status")) is not str
        or record["ssm_ping_status"] != "Online"
    ):
        raise ProductionPrelaunchAuthorityError(
            "production controller baseline controller identity mismatch"
        )
    history = _controller_history(
        record.get("exact_name_history"),
        sky_job_name=sky_job_name,
        controller_identity=controller_identity,
        observed_at=observed_time,
    )
    _empty_active_job_ids(
        record.get("active_exact_name_job_ids"),
        history=history,
    )
    _empty_p5_ids(record.get("active_tagged_p5_instance_ids"))
    if observed_text != record["observed_at"]:
        raise ProductionPrelaunchAuthorityError(
            "controller baseline timestamp is not canonical"
        )
    production_controller_baseline_s3_key(
        run_id=run_id,
        baseline_body_sha256=str(record["baseline_body_sha256"]),
    )
    return _copy_dict(record)


def _validate_baseline_sources(
    value: object,
    *,
    descriptor: object,
    intent: object,
) -> dict[str, object]:
    record = _validate_baseline_standalone(value)
    expected = _baseline_body(
        descriptor=descriptor,
        intent=intent,
        controller_instance_id=record["controller_instance_id"],
        controller_instance_type=record["controller_instance_type"],
        controller_profile_arn=record["controller_profile_arn"],
        controller_cluster_name=record["controller_cluster_name"],
        ssm_ping_status=record["ssm_ping_status"],
        exact_name_history=record["exact_name_history"],
        active_exact_name_job_ids=record["active_exact_name_job_ids"],
        active_tagged_p5_instance_ids=record[
            "active_tagged_p5_instance_ids"
        ],
        observed_at=record["observed_at"],
    )
    body = dict(record)
    body.pop("baseline_body_sha256")
    if body != expected:
        raise ProductionPrelaunchAuthorityError(
            "production controller baseline source authority drift"
        )
    return record


def _decode_baseline_artifact(
    value: object,
    *,
    descriptor: object,
    intent: object,
) -> tuple[VersionedJsonArtifact, dict[str, object]]:
    artifact, parsed, file_sha = _artifact(
        value,
        label="production controller baseline",
    )
    record = _validate_baseline_sources(
        parsed,
        descriptor=descriptor,
        intent=intent,
    )
    expected_key = production_controller_baseline_s3_key(
        run_id=str(record["run_id"]),
        baseline_body_sha256=str(record["baseline_body_sha256"]),
    )
    if (
        artifact.key != expected_key
        or file_sha != production_controller_baseline_file_sha256(record)
    ):
        raise ProductionPrelaunchAuthorityError(
            "production controller baseline artifact identity mismatch"
        )
    return artifact, record


def _deployment_expected(
    *,
    descriptor_artifact: VersionedJsonArtifact,
    intent_artifact: VersionedJsonArtifact,
    descriptor: Mapping[str, object],
    intent: Mapping[str, object],
) -> dict[str, object]:
    return {
        "run_id": intent["run_id"],
        "bucket": intent["bucket"],
        "descriptor_key": descriptor_artifact.key,
        "descriptor_file_sha256": _sha(descriptor_artifact.raw),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "descriptor_version_id": descriptor_artifact.version_id,
        "intent_key": intent_artifact.key,
        "intent_file_sha256": _sha(intent_artifact.raw),
        "intent_body_sha256": intent["intent_body_sha256"],
        "intent_version_id": intent_artifact.version_id,
        "sky_job_name": intent["sky_job_name"],
        "must_start_by": intent["must_start_by"],
    }


def _closed_string_map(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> dict[str, str]:
    result = _exact_fields(value, fields=fields, label=label)
    normalized: dict[str, str] = {}
    for field in fields:
        normalized[field] = _exact_string(
            result.get(field),
            field=f"{label}.{field}",
        )
    return normalized


def _validate_activation(
    value: object,
    *,
    expected: Mapping[str, object],
    template_sha256: str,
    lambda_function_arn: str,
    lambda_code_sha256: str,
    lambda_code_version_id: str,
    dlq_arn: str,
) -> dict[str, object]:
    activation = _exact_fields(
        value,
        fields=_ACTIVATION_FIELDS,
        label="activation_capabilities",
    )
    if (
        type(activation.get("stack_status")) is not str
        or activation["stack_status"] not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
        or type(activation.get("stack_operation_in_progress")) is not bool
        or activation["stack_operation_in_progress"] is not False
        or type(activation.get("foundation_revision")) is not str
        or activation["foundation_revision"] != template_sha256
    ):
        raise ProductionPrelaunchAuthorityError(
            "activation stack state/revision mismatch"
        )
    expected_source_fields = {
        "managed_mode": _MODE,
        "run_id": expected["run_id"],
        "bucket": expected["bucket"],
        "descriptor_key": expected["descriptor_key"],
        "descriptor_file_sha256": expected["descriptor_file_sha256"],
        "descriptor_body_sha256": expected["descriptor_body_sha256"],
        "descriptor_version_id": expected["descriptor_version_id"],
        "intent_key": expected["intent_key"],
        "intent_file_sha256": expected["intent_file_sha256"],
        "intent_body_sha256": expected["intent_body_sha256"],
        "intent_version_id": expected["intent_version_id"],
        "sky_job_name": expected["sky_job_name"],
        "must_start_by": expected["must_start_by"],
        "primary_wake_at": expected["must_start_by"],
    }
    for field, expected_value in expected_source_fields.items():
        if (
            type(activation.get(field)) is not type(expected_value)
            or activation[field] != expected_value
        ):
            raise ProductionPrelaunchAuthorityError(
                f"activation source field {field} mismatch"
            )
    qualified_lambda = _exact_string(
        activation.get("lambda_qualified_arn"),
        field="activation lambda_qualified_arn",
    )
    if re.fullmatch(
        re.escape(lambda_function_arn) + r":[1-9][0-9]*",
        qualified_lambda,
    ) is None:
        raise ProductionPrelaunchAuthorityError(
            "activation Lambda qualified ARN mismatch"
        )
    if (
        type(activation.get("lambda_state")) is not str
        or activation["lambda_state"] != "Active"
        or type(activation.get("lambda_last_update_status")) is not str
        or activation["lambda_last_update_status"] != "Successful"
        or type(activation.get("lambda_execution_role_arn")) is not str
        or activation["lambda_execution_role_arn"]
        != (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-sky-production-must-start"
        )
        or type(activation.get("lambda_code_sha256")) is not str
        or activation["lambda_code_sha256"] != lambda_code_sha256
        or type(activation.get("lambda_code_version_id")) is not str
        or activation["lambda_code_version_id"] != lambda_code_version_id
    ):
        raise ProductionPrelaunchAuthorityError(
            "activation Lambda state/identity mismatch"
        )
    environment = _closed_string_map(
        activation.get("lambda_environment"),
        fields=_ENVIRONMENT_FIELDS,
        label="activation lambda_environment",
    )
    expected_environment = {
        "EXPECTED_ACCOUNT_ID": _ACCOUNT_ID,
        "CAMPAIGN_BUCKET": str(expected["bucket"]),
        "CAMPAIGN_RUN_ID": str(expected["run_id"]),
        "MANAGED_MODE": _MODE,
        "CAMPAIGN_DESCRIPTOR_KEY": str(expected["descriptor_key"]),
        "CAMPAIGN_DESCRIPTOR_VERSION_ID": str(
            expected["descriptor_version_id"]
        ),
        "IMMUTABLE_SUBMISSION_KEY": str(expected["intent_key"]),
        "IMMUTABLE_SUBMISSION_VERSION_ID": str(expected["intent_version_id"]),
        "SUBMISSION_BODY_SHA256": str(expected["intent_body_sha256"]),
        "SKY_JOB_NAME": str(expected["sky_job_name"]),
        "MUST_START_BY": str(expected["must_start_by"]),
        "COORDINATOR_MODE": _COORDINATOR_MODE,
    }
    if environment != expected_environment:
        raise ProductionPrelaunchAuthorityError(
            "activation Lambda environment mismatch"
        )
    base_trigger = {
        "campaign_run_id": str(expected["run_id"]),
        "managed_mode": _MODE,
        "sky_job_name": str(expected["sky_job_name"]),
        "must_start_by": str(expected["must_start_by"]),
        "descriptor_key": str(expected["descriptor_key"]),
        "descriptor_version_id": str(expected["descriptor_version_id"]),
        "intent_key": str(expected["intent_key"]),
        "intent_version_id": str(expected["intent_version_id"]),
    }
    reconciliation_input = _closed_string_map(
        activation.get("reconciliation_input"),
        fields=_TRIGGER_FIELDS,
        label="activation reconciliation_input",
    )
    deadline_input = _closed_string_map(
        activation.get("deadline_input"),
        fields=_TRIGGER_FIELDS,
        label="activation deadline_input",
    )
    if (
        reconciliation_input != {**base_trigger, "trigger": "reconcile"}
        or deadline_input != {**base_trigger, "trigger": "primary-deadline"}
    ):
        raise ProductionPrelaunchAuthorityError(
            "activation trigger input mismatch"
        )
    deadline = str(expected["must_start_by"])
    exact_values: dict[str, object] = {
        "reconciliation_schedule_expression": "rate(1 minute)",
        "reconciliation_target_arn": qualified_lambda,
        "reconciliation_target_id": "sky-production-must-start-reconcile",
        "reconciliation_retry_max_event_age_seconds": 300,
        "reconciliation_retry_max_attempts": 2,
        "reconciliation_dlq_arn": dlq_arn,
        "deadline_schedule_expression": f"at({deadline[:-1]})",
        "deadline_schedule_timezone": "UTC",
        "deadline_flexible_window_mode": "OFF",
        "deadline_target_arn": qualified_lambda,
        "deadline_target_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-sky-production-deadline-scheduler"
        ),
        "deadline_retry_max_event_age_seconds": 300,
        "deadline_retry_max_attempts": 2,
        "deadline_dlq_arn": dlq_arn,
    }
    integer_fields = {
        "reconciliation_retry_max_event_age_seconds",
        "reconciliation_retry_max_attempts",
        "deadline_retry_max_event_age_seconds",
        "deadline_retry_max_attempts",
    }
    for field, expected_value in exact_values.items():
        expected_type = int if field in integer_fields else str
        if (
            type(activation.get(field)) is not expected_type
            or activation[field] != expected_value
        ):
            raise ProductionPrelaunchAuthorityError(
                f"activation {field} mismatch"
            )
    return dict(activation)


def _validate_deployment(
    value: object,
    *,
    expected: Mapping[str, object],
) -> dict[str, object]:
    deployment = _exact_fields(
        value,
        fields=_DEPLOYMENT_IDENTITY_FIELDS,
        label="control-plane deployment identity",
    )
    run_id = _run_id(expected.get("run_id"))
    stack_id = _exact_string(deployment.get("stack_id"), field="stack_id")
    lambda_arn = _exact_string(
        deployment.get("lambda_function_arn"),
        field="lambda_function_arn",
    )
    reconciliation_arn = _exact_string(
        deployment.get("reconciliation_rule_arn"),
        field="reconciliation_rule_arn",
    )
    schedule_arn = _exact_string(
        deployment.get("deadline_schedule_arn"),
        field="deadline_schedule_arn",
    )
    dlq_arn = _exact_string(deployment.get("dlq_arn"), field="dlq_arn")
    if (
        re.fullmatch(
            r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
            r"keep-glm52-sky-production-control-plane/[A-Za-z0-9-]+",
            stack_id,
        )
        is None
        or lambda_arn
        != (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-sky-production-must-start"
        )
        or reconciliation_arn
        != (
            "arn:aws:events:us-west-2:246813579024:rule/"
            "keep-glm52-sky-production-reconcile"
        )
        or schedule_arn
        != (
            "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
            "keep-glm52-sky-production-deadline"
        )
        or dlq_arn
        != (
            "arn:aws:sqs:us-west-2:246813579024:"
            "keep-glm52-sky-production-must-start-dlq"
        )
    ):
        raise ProductionPrelaunchAuthorityError(
            "control-plane deployment ARN identity mismatch"
        )
    template_sha = _digest(
        deployment.get("template_sha256"),
        field="template_sha256",
    )
    lambda_code_sha = _digest(
        deployment.get("lambda_code_sha256"),
        field="lambda_code_sha256",
    )
    _digest(
        deployment.get("iam_policy_sha256"),
        field="iam_policy_sha256",
    )
    lambda_code_version = require_opaque_version_id(
        deployment.get("lambda_code_version_id"),
        field="lambda_code_version_id",
    )
    expected_code_key = (
        f"campaigns/{run_id}/control-plane/production/lambda-code/"
        f"{lambda_code_sha}/sky-must-start.zip"
    )
    if (
        _safe_key(
            deployment.get("lambda_code_s3_key"),
            field="lambda_code_s3_key",
        )
        != expected_code_key
    ):
        raise ProductionPrelaunchAuthorityError(
            "Lambda code S3 key is not content-addressed for this campaign"
        )
    _validate_activation(
        deployment.get("activation_capabilities"),
        expected=expected,
        template_sha256=template_sha,
        lambda_function_arn=lambda_arn,
        lambda_code_sha256=lambda_code_sha,
        lambda_code_version_id=lambda_code_version,
        dlq_arn=dlq_arn,
    )
    return _copy_dict(deployment)


def _ready_body(
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
    reviewed_deployment_identity: object,
    observed_deployment_identity: object,
    reconciliation_rule_state: object,
    deadline_schedule_state: object,
    coordinator_mode: object,
    observed_at: object,
) -> dict[str, object]:
    (
        descriptor_artifact,
        intent_artifact,
        descriptor_value,
        intent_value,
    ) = _authenticate_sources(descriptor=descriptor, intent=intent)
    baseline_artifact, baseline = _decode_baseline_artifact(
        controller_baseline,
        descriptor=descriptor,
        intent=intent,
    )
    expected = _deployment_expected(
        descriptor_artifact=descriptor_artifact,
        intent_artifact=intent_artifact,
        descriptor=descriptor_value,
        intent=intent_value,
    )
    reviewed = _validate_deployment(
        reviewed_deployment_identity,
        expected=expected,
    )
    observed = _validate_deployment(
        observed_deployment_identity,
        expected=expected,
    )
    if reviewed != observed:
        raise ProductionPrelaunchAuthorityError(
            "reviewed and observed deployment identity drift"
        )
    if (
        type(reconciliation_rule_state) is not str
        or reconciliation_rule_state != "ENABLED"
        or type(deadline_schedule_state) is not str
        or deadline_schedule_state != "ENABLED"
        or type(coordinator_mode) is not str
        or coordinator_mode != _COORDINATOR_MODE
    ):
        raise ProductionPrelaunchAuthorityError(
            "production control plane is not enabled in the required mode"
        )
    observed_text, ready_time = _time(
        observed_at,
        field="control-plane readiness observed_at",
        permit_datetime=True,
    )
    _, baseline_time = _time(
        baseline.get("observed_at"),
        field="controller baseline observed_at",
        permit_datetime=False,
    )
    _, deadline = _time(
        intent_value.get("must_start_by"),
        field="must_start_by",
        permit_datetime=False,
    )
    if not baseline_time <= ready_time < deadline:
        raise ProductionPrelaunchAuthorityError(
            "control-plane readiness observation is outside baseline/deadline"
        )
    return {
        "schema_version": 1,
        "record_type": "glm52_production_must_start_control_plane_ready_v1",
        "account_id": intent_value["account_id"],
        "region": intent_value["region"],
        "bucket": intent_value["bucket"],
        "run_id": intent_value["run_id"],
        "managed_mode": _MODE,
        "campaign_identity_sha256": intent_value[
            "campaign_identity_sha256"
        ],
        "descriptor_key": descriptor_artifact.key,
        "descriptor_file_sha256": _sha(descriptor_artifact.raw),
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "descriptor_version_id": descriptor_artifact.version_id,
        "intent_key": intent_artifact.key,
        "intent_file_sha256": _sha(intent_artifact.raw),
        "intent_body_sha256": intent_value["intent_body_sha256"],
        "intent_version_id": intent_artifact.version_id,
        "controller_baseline_key": baseline_artifact.key,
        "controller_baseline_file_sha256": _sha(baseline_artifact.raw),
        "controller_baseline_body_sha256": baseline[
            "baseline_body_sha256"
        ],
        "controller_baseline_version_id": baseline_artifact.version_id,
        "stack_id": reviewed["stack_id"],
        "template_sha256": reviewed["template_sha256"],
        "lambda_function_arn": reviewed["lambda_function_arn"],
        "lambda_code_s3_key": reviewed["lambda_code_s3_key"],
        "lambda_code_sha256": reviewed["lambda_code_sha256"],
        "lambda_code_version_id": reviewed["lambda_code_version_id"],
        "iam_policy_sha256": reviewed["iam_policy_sha256"],
        "reconciliation_rule_arn": reviewed["reconciliation_rule_arn"],
        "reconciliation_rule_state": "ENABLED",
        "deadline_schedule_arn": reviewed["deadline_schedule_arn"],
        "deadline_schedule_state": "ENABLED",
        "dlq_arn": reviewed["dlq_arn"],
        "coordinator_mode": _COORDINATOR_MODE,
        "activation_capabilities": reviewed["activation_capabilities"],
        "observed_at": observed_text,
    }


def _deployment_from_ready(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "stack_id": record["stack_id"],
        "template_sha256": record["template_sha256"],
        "lambda_function_arn": record["lambda_function_arn"],
        "lambda_code_s3_key": record["lambda_code_s3_key"],
        "lambda_code_sha256": record["lambda_code_sha256"],
        "lambda_code_version_id": record["lambda_code_version_id"],
        "iam_policy_sha256": record["iam_policy_sha256"],
        "reconciliation_rule_arn": record["reconciliation_rule_arn"],
        "deadline_schedule_arn": record["deadline_schedule_arn"],
        "dlq_arn": record["dlq_arn"],
        "activation_capabilities": record["activation_capabilities"],
    }


def _validate_ready_standalone(value: object) -> dict[str, object]:
    _contract(
        record_kind="control-plane-ready",
        schema_version=1,
        record_type="glm52_production_must_start_control_plane_ready_v1",
        digest_field="control_plane_ready_body_sha256",
        address_kind="intent-control-body",
    )
    record = _self_hashed(
        value,
        fields=_READY_FIELDS,
        digest_field="control_plane_ready_body_sha256",
        label="production must-start control-plane readiness",
    )
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or type(record.get("record_type")) is not str
        or record["record_type"]
        != "glm52_production_must_start_control_plane_ready_v1"
        or type(record.get("managed_mode")) is not str
        or record["managed_mode"] != _MODE
        or type(record.get("account_id")) is not str
        or record["account_id"] != _ACCOUNT_ID
        or type(record.get("region")) is not str
        or record["region"] != _REGION
    ):
        raise ProductionPrelaunchAuthorityError(
            "production control-plane readiness discriminant mismatch"
        )
    run_id = _run_id(record.get("run_id"))
    _bucket(record.get("bucket"))
    digest_fields = (
        "campaign_identity_sha256",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "template_sha256",
        "lambda_code_sha256",
        "iam_policy_sha256",
        "control_plane_ready_body_sha256",
    )
    for field in digest_fields:
        _digest(record.get(field), field=field)
    for field in (
        "descriptor_key",
        "intent_key",
        "controller_baseline_key",
        "lambda_code_s3_key",
    ):
        _safe_key(record.get(field), field=field)
    for field in (
        "descriptor_version_id",
        "intent_version_id",
        "controller_baseline_version_id",
        "lambda_code_version_id",
    ):
        require_opaque_version_id(record.get(field), field=field)
    if (
        record["intent_key"]
        != production_submission_intent_s3_key(
            run_id=run_id,
            intent_body_sha256=str(record["intent_body_sha256"]),
        )
        or record["controller_baseline_key"]
        != production_controller_baseline_s3_key(
            run_id=run_id,
            baseline_body_sha256=str(record["controller_baseline_body_sha256"]),
        )
    ):
        raise ProductionPrelaunchAuthorityError(
            "production control-plane readiness source key mismatch"
        )
    if (
        type(record.get("reconciliation_rule_state")) is not str
        or record["reconciliation_rule_state"] != "ENABLED"
        or type(record.get("deadline_schedule_state")) is not str
        or record["deadline_schedule_state"] != "ENABLED"
        or type(record.get("coordinator_mode")) is not str
        or record["coordinator_mode"] != _COORDINATOR_MODE
    ):
        raise ProductionPrelaunchAuthorityError(
            "production control-plane readiness is not enabled"
        )
    sky_job_name = expected_sky_job_name(
        run_id=run_id,
        managed_mode=_MODE,
    )
    activation = record.get("activation_capabilities")
    if type(activation) is not dict:
        raise ProductionPrelaunchAuthorityError(
            "activation_capabilities must be an exact object"
        )
    if (
        type(activation.get("sky_job_name")) is not str
        or activation["sky_job_name"] != sky_job_name
    ):
        raise ProductionPrelaunchAuthorityError(
            "production control-plane readiness job mismatch"
        )
    must_start_text, must_start_time = _time(
        activation.get("must_start_by"),
        field="activation must_start_by",
        permit_datetime=False,
    )
    expected = {
        "run_id": run_id,
        "bucket": record["bucket"],
        "descriptor_key": record["descriptor_key"],
        "descriptor_file_sha256": record["descriptor_file_sha256"],
        "descriptor_body_sha256": record["descriptor_body_sha256"],
        "descriptor_version_id": record["descriptor_version_id"],
        "intent_key": record["intent_key"],
        "intent_file_sha256": record["intent_file_sha256"],
        "intent_body_sha256": record["intent_body_sha256"],
        "intent_version_id": record["intent_version_id"],
        "sky_job_name": sky_job_name,
        "must_start_by": must_start_text,
    }
    _validate_deployment(_deployment_from_ready(record), expected=expected)
    _, observed_time = _time(
        record.get("observed_at"),
        field="control-plane readiness observed_at",
        permit_datetime=False,
    )
    if observed_time >= must_start_time:
        raise ProductionPrelaunchAuthorityError(
            "control-plane readiness observed_at must be before must_start_by"
        )
    production_must_start_control_plane_ready_s3_key(
        run_id=run_id,
        intent_body_sha256=str(record["intent_body_sha256"]),
        control_plane_ready_body_sha256=str(
            record["control_plane_ready_body_sha256"]
        ),
    )
    return _copy_dict(record)


def _validate_ready_sources(
    value: object,
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
) -> dict[str, object]:
    record = _validate_ready_standalone(value)
    deployment = _deployment_from_ready(record)
    expected = _ready_body(
        descriptor=descriptor,
        intent=intent,
        controller_baseline=controller_baseline,
        reviewed_deployment_identity=deployment,
        observed_deployment_identity=_copy_dict(deployment),
        reconciliation_rule_state=record["reconciliation_rule_state"],
        deadline_schedule_state=record["deadline_schedule_state"],
        coordinator_mode=record["coordinator_mode"],
        observed_at=record["observed_at"],
    )
    body = dict(record)
    body.pop("control_plane_ready_body_sha256")
    if body != expected:
        raise ProductionPrelaunchAuthorityError(
            "production control-plane readiness source authority drift"
        )
    return record


def _copy_dict(value: dict[str, object]) -> dict[str, object]:
    """Copy one exact JSON object without importing mutable-runtime helpers."""

    return json.loads(_canonical(value).decode("ascii"))


def _decode_ready_artifact(
    value: object,
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
) -> tuple[VersionedJsonArtifact, dict[str, object]]:
    artifact, parsed, file_sha = _artifact(
        value,
        label="production must-start control-plane readiness",
    )
    record = _validate_ready_sources(
        parsed,
        descriptor=descriptor,
        intent=intent,
        controller_baseline=controller_baseline,
    )
    expected_key = production_must_start_control_plane_ready_s3_key(
        run_id=str(record["run_id"]),
        intent_body_sha256=str(record["intent_body_sha256"]),
        control_plane_ready_body_sha256=str(
            record["control_plane_ready_body_sha256"]
        ),
    )
    if (
        artifact.key != expected_key
        or file_sha
        != production_must_start_control_plane_ready_file_sha256(record)
    ):
        raise ProductionPrelaunchAuthorityError(
            "production control-plane readiness artifact identity mismatch"
        )
    return artifact, record


def _acquisition_body(
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
    acquired_at: object,
) -> tuple[dict[str, object], datetime]:
    (
        descriptor_artifact,
        intent_artifact,
        descriptor_value,
        intent_value,
    ) = _authenticate_sources(descriptor=descriptor, intent=intent)
    baseline_artifact, baseline = _decode_baseline_artifact(
        controller_baseline,
        descriptor=descriptor,
        intent=intent,
    )
    ready_artifact, ready = _decode_ready_artifact(
        must_start_control_plane_ready,
        descriptor=descriptor,
        intent=intent,
        controller_baseline=controller_baseline,
    )
    acquired_text, acquired_time = _time(
        acquired_at,
        field="acquired_at",
        permit_datetime=True,
    )
    spend_text, spend_time = _time(
        intent_value.get("spend_snapshot_observed_at"),
        field="spend_snapshot_observed_at",
        permit_datetime=False,
    )
    _, intent_time = _time(
        intent_value.get("intent_at"),
        field="intent_at",
        permit_datetime=False,
    )
    baseline_text, baseline_time = _time(
        baseline.get("observed_at"),
        field="controller baseline observed_at",
        permit_datetime=False,
    )
    ready_text, ready_time = _time(
        ready.get("observed_at"),
        field="control-plane readiness observed_at",
        permit_datetime=False,
    )
    deadline_text, deadline = _time(
        intent_value.get("must_start_by"),
        field="must_start_by",
        permit_datetime=False,
    )
    if not (
        spend_time
        <= intent_time
        <= baseline_time
        <= ready_time
        <= acquired_time
        < deadline
    ):
        raise ProductionPrelaunchAuthorityError(
            "production acquisition timestamp ordering mismatch"
        )
    for label, observation in (
        ("spend snapshot", spend_time),
        ("controller baseline", baseline_time),
        ("control-plane readiness", ready_time),
    ):
        if acquired_time - observation > _MAX_AGE:
            raise ProductionPrelaunchAuthorityError(
                f"{label} is older than 60 seconds at acquisition"
            )
    return (
        {
            "schema_version": 1,
            "record_type": "glm52_sky_production_submission_acquired_v1",
            "account_id": intent_value["account_id"],
            "region": intent_value["region"],
            "bucket": intent_value["bucket"],
            "run_id": intent_value["run_id"],
            "managed_mode": _MODE,
            "campaign_identity_sha256": intent_value[
                "campaign_identity_sha256"
            ],
            "descriptor_key": descriptor_artifact.key,
            "descriptor_file_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor_value[
                "descriptor_body_sha256"
            ],
            "descriptor_version_id": descriptor_artifact.version_id,
            "intent_key": intent_artifact.key,
            "intent_file_sha256": _sha(intent_artifact.raw),
            "intent_body_sha256": intent_value["intent_body_sha256"],
            "intent_version_id": intent_artifact.version_id,
            "controller_baseline_key": baseline_artifact.key,
            "controller_baseline_file_sha256": _sha(baseline_artifact.raw),
            "controller_baseline_body_sha256": baseline[
                "baseline_body_sha256"
            ],
            "controller_baseline_version_id": baseline_artifact.version_id,
            "must_start_control_plane_ready_key": ready_artifact.key,
            "must_start_control_plane_ready_file_sha256": _sha(
                ready_artifact.raw
            ),
            "must_start_control_plane_ready_body_sha256": ready[
                "control_plane_ready_body_sha256"
            ],
            "must_start_control_plane_ready_version_id": ready_artifact.version_id,
            "sky_job_name": intent_value["sky_job_name"],
            "must_start_by": deadline_text,
            "spend_snapshot_observed_at": spend_text,
            "controller_baseline_observed_at": baseline_text,
            "control_plane_ready_observed_at": ready_text,
            "acquired_at": acquired_text,
        },
        acquired_time,
    )


def _validate_acquisition_standalone(value: object) -> dict[str, object]:
    _contract(
        record_kind="submission-acquisition",
        schema_version=1,
        record_type="glm52_sky_production_submission_acquired_v1",
        digest_field="acquisition_body_sha256",
        address_kind="descriptor-file-singleton",
    )
    record = _self_hashed(
        value,
        fields=_ACQUISITION_FIELDS,
        digest_field="acquisition_body_sha256",
        label="production submission acquisition",
    )
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or type(record.get("record_type")) is not str
        or record["record_type"]
        != "glm52_sky_production_submission_acquired_v1"
        or type(record.get("managed_mode")) is not str
        or record["managed_mode"] != _MODE
        or type(record.get("account_id")) is not str
        or record["account_id"] != _ACCOUNT_ID
        or type(record.get("region")) is not str
        or record["region"] != _REGION
    ):
        raise ProductionPrelaunchAuthorityError(
            "production submission acquisition discriminant mismatch"
        )
    run_id = _run_id(record.get("run_id"))
    _bucket(record.get("bucket"))
    for field in (
        "campaign_identity_sha256",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "must_start_control_plane_ready_file_sha256",
        "must_start_control_plane_ready_body_sha256",
        "acquisition_body_sha256",
    ):
        _digest(record.get(field), field=field)
    for field in (
        "descriptor_key",
        "intent_key",
        "controller_baseline_key",
        "must_start_control_plane_ready_key",
    ):
        _safe_key(record.get(field), field=field)
    for field in (
        "descriptor_version_id",
        "intent_version_id",
        "controller_baseline_version_id",
        "must_start_control_plane_ready_version_id",
    ):
        require_opaque_version_id(record.get(field), field=field)
    expected_keys = {
        "intent_key": production_submission_intent_s3_key(
            run_id=run_id,
            intent_body_sha256=str(record["intent_body_sha256"]),
        ),
        "controller_baseline_key": production_controller_baseline_s3_key(
            run_id=run_id,
            baseline_body_sha256=str(record["controller_baseline_body_sha256"]),
        ),
        "must_start_control_plane_ready_key": (
            production_must_start_control_plane_ready_s3_key(
                run_id=run_id,
                intent_body_sha256=str(record["intent_body_sha256"]),
                control_plane_ready_body_sha256=str(
                    record["must_start_control_plane_ready_body_sha256"]
                ),
            )
        ),
    }
    for field, expected_key in expected_keys.items():
        if record[field] != expected_key:
            raise ProductionPrelaunchAuthorityError(
                f"production acquisition {field} is not canonical"
            )
    if (
        type(record.get("sky_job_name")) is not str
        or record["sky_job_name"]
        != expected_sky_job_name(run_id=run_id, managed_mode=_MODE)
    ):
        raise ProductionPrelaunchAuthorityError(
            "production acquisition sky_job_name mismatch"
        )
    _, spend_time = _time(
        record.get("spend_snapshot_observed_at"),
        field="spend_snapshot_observed_at",
        permit_datetime=False,
    )
    _, baseline_time = _time(
        record.get("controller_baseline_observed_at"),
        field="controller_baseline_observed_at",
        permit_datetime=False,
    )
    _, ready_time = _time(
        record.get("control_plane_ready_observed_at"),
        field="control_plane_ready_observed_at",
        permit_datetime=False,
    )
    _, acquired_time = _time(
        record.get("acquired_at"),
        field="acquired_at",
        permit_datetime=False,
    )
    _, deadline = _time(
        record.get("must_start_by"),
        field="must_start_by",
        permit_datetime=False,
    )
    if not spend_time <= baseline_time <= ready_time <= acquired_time < deadline:
        raise ProductionPrelaunchAuthorityError(
            "production acquisition standalone timestamp ordering mismatch"
        )
    for label, observation_time in (
        ("spend snapshot", spend_time),
        ("controller baseline", baseline_time),
        ("control-plane readiness", ready_time),
    ):
        if acquired_time - observation_time > _MAX_AGE:
            raise ProductionPrelaunchAuthorityError(
                f"production acquisition {label} is older than 60 seconds "
                "at recorded acquisition"
            )
    production_submission_acquired_s3_key(
        run_id=run_id,
        descriptor_file_sha256=str(record["descriptor_file_sha256"]),
    )
    return _copy_dict(record)


def _validate_acquisition_sources(
    value: object,
    *,
    descriptor: object,
    intent: object,
    controller_baseline: object,
    must_start_control_plane_ready: object,
    now: object,
) -> dict[str, object]:
    record = _validate_acquisition_standalone(value)
    expected, acquired_time = _acquisition_body(
        descriptor=descriptor,
        intent=intent,
        controller_baseline=controller_baseline,
        must_start_control_plane_ready=must_start_control_plane_ready,
        acquired_at=record["acquired_at"],
    )
    body = dict(record)
    body.pop("acquisition_body_sha256")
    if body != expected:
        raise ProductionPrelaunchAuthorityError(
            "production acquisition source authority drift"
        )
    _, observed_now = _time(
        now,
        field="now",
        permit_datetime=True,
    )
    _, deadline = _time(
        record.get("must_start_by"),
        field="must_start_by",
        permit_datetime=False,
    )
    if observed_now < acquired_time:
        raise ProductionPrelaunchAuthorityError(
            "now precedes production acquisition"
        )
    if observed_now >= deadline:
        raise ProductionPrelaunchAuthorityError(
            "production acquisition is expired"
        )
    for field, label in (
        ("spend_snapshot_observed_at", "spend snapshot"),
        ("controller_baseline_observed_at", "controller baseline"),
        ("control_plane_ready_observed_at", "control-plane readiness"),
    ):
        _, observation = _time(
            record.get(field),
            field=field,
            permit_datetime=False,
        )
        if observed_now - observation > _MAX_AGE:
            raise ProductionPrelaunchAuthorityError(
                f"{label} is older than 60 seconds at current validation"
            )
    return record


def build_production_controller_baseline(
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    controller_instance_id: str,
    controller_instance_type: str,
    controller_profile_arn: str,
    controller_cluster_name: str,
    ssm_ping_status: str,
    exact_name_history: list[Mapping[str, object]],
    active_exact_name_job_ids: list[int],
    active_tagged_p5_instance_ids: list[str],
    observed_at: Union[datetime, str],
) -> dict[str, object]:
    """Build one self-hashed production controller baseline candidate."""

    def operation() -> dict[str, object]:
        body = _baseline_body(
            descriptor=descriptor,
            intent=intent,
            controller_instance_id=controller_instance_id,
            controller_instance_type=controller_instance_type,
            controller_profile_arn=controller_profile_arn,
            controller_cluster_name=controller_cluster_name,
            ssm_ping_status=ssm_ping_status,
            exact_name_history=exact_name_history,
            active_exact_name_job_ids=active_exact_name_job_ids,
            active_tagged_p5_instance_ids=active_tagged_p5_instance_ids,
            observed_at=observed_at,
        )
        result = {
            **body,
            "baseline_body_sha256": _sha(_canonical(body)),
        }
        return _validate_baseline_sources(
            result,
            descriptor=descriptor,
            intent=intent,
        )

    return _translate(operation)


def production_controller_baseline_file_bytes(
    value: Mapping[str, object],
) -> bytes:
    """Return validated canonical baseline bytes plus exactly one LF."""

    return _translate(lambda: _canonical(_validate_baseline_standalone(value)) + b"\n")


def production_controller_baseline_file_sha256(
    value: Mapping[str, object],
) -> str:
    """Hash the complete canonical production baseline file."""

    return _translate(lambda: _sha(production_controller_baseline_file_bytes(value)))


def production_controller_baseline_s3_key(
    *,
    run_id: str,
    baseline_body_sha256: str,
) -> str:
    """Return the content-addressed production baseline key."""

    def operation() -> str:
        run = _run_id(run_id)
        digest = _digest(
            baseline_body_sha256,
            field="baseline_body_sha256",
        )
        return (
            f"campaigns/{run}/production/controller-baselines/"
            f"{digest}/CONTROLLER_BASELINE.json"
        )

    return _translate(operation)


def validate_production_controller_baseline(
    value: Mapping[str, object],
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
) -> dict[str, object]:
    """Validate a baseline against exact versioned descriptor and intent."""

    return _translate(
        lambda: _validate_baseline_sources(
            value,
            descriptor=descriptor,
            intent=intent,
        )
    )


def build_production_must_start_control_plane_ready(
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    reviewed_deployment_identity: Mapping[str, object],
    observed_deployment_identity: Mapping[str, object],
    reconciliation_rule_state: str,
    deadline_schedule_state: str,
    coordinator_mode: str,
    observed_at: Union[datetime, str],
) -> dict[str, object]:
    """Build one self-hashed production control-plane readiness candidate."""

    def operation() -> dict[str, object]:
        body = _ready_body(
            descriptor=descriptor,
            intent=intent,
            controller_baseline=controller_baseline,
            reviewed_deployment_identity=reviewed_deployment_identity,
            observed_deployment_identity=observed_deployment_identity,
            reconciliation_rule_state=reconciliation_rule_state,
            deadline_schedule_state=deadline_schedule_state,
            coordinator_mode=coordinator_mode,
            observed_at=observed_at,
        )
        result = {
            **body,
            "control_plane_ready_body_sha256": _sha(_canonical(body)),
        }
        return _validate_ready_sources(
            result,
            descriptor=descriptor,
            intent=intent,
            controller_baseline=controller_baseline,
        )

    return _translate(operation)


def production_must_start_control_plane_ready_file_bytes(
    value: Mapping[str, object],
) -> bytes:
    """Return validated canonical control-plane readiness bytes plus LF."""

    return _translate(lambda: _canonical(_validate_ready_standalone(value)) + b"\n")


def production_must_start_control_plane_ready_file_sha256(
    value: Mapping[str, object],
) -> str:
    """Hash the complete canonical production readiness file."""

    return _translate(
        lambda: _sha(
            production_must_start_control_plane_ready_file_bytes(value)
        )
    )


def production_must_start_control_plane_ready_s3_key(
    *,
    run_id: str,
    intent_body_sha256: str,
    control_plane_ready_body_sha256: str,
) -> str:
    """Return the intent/control-addressed production readiness key."""

    def operation() -> str:
        run = _run_id(run_id)
        intent_digest = _digest(
            intent_body_sha256,
            field="intent_body_sha256",
        )
        ready_digest = _digest(
            control_plane_ready_body_sha256,
            field="control_plane_ready_body_sha256",
        )
        return (
            f"campaigns/{run}/monitor/must-start/production/"
            f"{intent_digest}/control-plane-ready/"
            f"{ready_digest}/CONTROL_PLANE_READY.json"
        )

    return _translate(operation)


def validate_production_must_start_control_plane_ready(
    value: Mapping[str, object],
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
) -> dict[str, object]:
    """Validate readiness against exact versioned source artifacts."""

    return _translate(
        lambda: _validate_ready_sources(
            value,
            descriptor=descriptor,
            intent=intent,
            controller_baseline=controller_baseline,
        )
    )


def build_production_submission_acquired(
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    acquired_at: Union[datetime, str],
) -> dict[str, object]:
    """Build one descriptor-singleton production acquisition candidate."""

    def operation() -> dict[str, object]:
        body, _ = _acquisition_body(
            descriptor=descriptor,
            intent=intent,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            acquired_at=acquired_at,
        )
        result = {
            **body,
            "acquisition_body_sha256": _sha(_canonical(body)),
        }
        return _validate_acquisition_sources(
            result,
            descriptor=descriptor,
            intent=intent,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            now=result["acquired_at"],
        )

    return _translate(operation)


def production_submission_acquired_file_bytes(
    value: Mapping[str, object],
) -> bytes:
    """Return validated canonical acquisition bytes plus exactly one LF."""

    return _translate(
        lambda: _canonical(_validate_acquisition_standalone(value)) + b"\n"
    )


def production_submission_acquired_file_sha256(
    value: Mapping[str, object],
) -> str:
    """Hash the complete canonical production acquisition file."""

    return _translate(lambda: _sha(production_submission_acquired_file_bytes(value)))


def production_submission_acquired_s3_key(
    *,
    run_id: str,
    descriptor_file_sha256: str,
) -> str:
    """Return the descriptor-file-singleton acquisition coordinate."""

    def operation() -> str:
        run = _run_id(run_id)
        digest = _digest(
            descriptor_file_sha256,
            field="descriptor_file_sha256",
        )
        return (
            f"campaigns/{run}/submissions/production/acquisitions/"
            f"{digest}/SUBMISSION_ACQUIRED.json"
        )

    return _translate(operation)


def validate_production_submission_acquired(
    value: Mapping[str, object],
    *,
    descriptor: VersionedJsonArtifact,
    intent: VersionedJsonArtifact,
    controller_baseline: VersionedJsonArtifact,
    must_start_control_plane_ready: VersionedJsonArtifact,
    now: Union[datetime, str],
) -> dict[str, object]:
    """Validate current acquisition authority against every exact source."""

    return _translate(
        lambda: _validate_acquisition_sources(
            value,
            descriptor=descriptor,
            intent=intent,
            controller_baseline=controller_baseline,
            must_start_control_plane_ready=must_start_control_plane_ready,
            now=now,
        )
    )


__all__ = [
    "ProductionPrelaunchAuthorityError",
    "build_production_controller_baseline",
    "production_controller_baseline_file_bytes",
    "production_controller_baseline_file_sha256",
    "production_controller_baseline_s3_key",
    "validate_production_controller_baseline",
    "build_production_must_start_control_plane_ready",
    "production_must_start_control_plane_ready_file_bytes",
    "production_must_start_control_plane_ready_file_sha256",
    "production_must_start_control_plane_ready_s3_key",
    "validate_production_must_start_control_plane_ready",
    "build_production_submission_acquired",
    "production_submission_acquired_file_bytes",
    "production_submission_acquired_file_sha256",
    "production_submission_acquired_s3_key",
    "validate_production_submission_acquired",
]
