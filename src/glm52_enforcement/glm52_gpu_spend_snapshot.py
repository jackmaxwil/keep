"""Enforcement-native post-seed authority for cumulative GLM-5.2 GPU spend."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Mapping

from .glm52_sky_campaign import (
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

QUALIFICATION_MAX_SECONDS = 4 * 60 * 60

_HEX64 = re.compile(r"[0-9a-f]{64}")
_RECORD_NAME = re.compile(
    r"(?P<index>[0-9]{6})-"
    r"(?P<event>allocation_started|allocation_ended)-"
    r"(?P<sha>[0-9a-f]{64})\.json"
)
_TERMINATION = re.compile(
    r"\((?P<timestamp>[0-9]{4}-[0-9]{2}-[0-9]{2} "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}) GMT\)"
)
_LATEST_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "gpu_spend_authority_sha256",
    "record_count",
    "record_keys",
    "latest_record_sha256",
    "ledger_sha256",
    "latest_body_sha256",
}
_START_RECORD_FIELDS = {
    "record_type",
    "run_id",
    "approval_sha256",
    "event",
    "job_id",
    "instance_id",
    "timestamp",
    "prior_record_sha256",
    "record_sha256",
}
_END_RECORD_FIELDS = _START_RECORD_FIELDS - {"job_id"}
_SNAPSHOT_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "campaign_identity_sha256",
    "descriptor_sha256",
    "descriptor_body_sha256",
    "approval_sha256",
    "approval_body_sha256",
    "gpu_spend_ledger_latest_sha256",
    "gpu_spend_ledger_latest_body_sha256",
    "gpu_spend_ledger_genesis_sha256",
    "gpu_spend_ledger_record_count",
    "gpu_spend_ledger_tip_record_sha256",
    "gpu_spend_ledger_file_sha256",
    "ec2_allocation_history_sha256",
    "ec2_allocation_instance_ids",
    "observed_at",
    "approved_gpu_runtime_seconds",
    "approved_gpu_cost_usd",
    "hourly_cost_usd",
    "consumed_gpu_seconds",
    "remaining_gpu_seconds",
    "consumed_gpu_cost_usd",
    "remaining_gpu_cost_usd",
    "qualification_allowance_seconds",
    "qualification_allowance_cost_usd",
    "open_allocation_count",
}
_SNAPSHOT_FIELDS = _SNAPSHOT_BODY_FIELDS | {"snapshot_body_sha256"}
_SNAPSHOT_DIGEST_FIELDS = {
    "campaign_identity_sha256",
    "descriptor_sha256",
    "descriptor_body_sha256",
    "approval_sha256",
    "approval_body_sha256",
    "gpu_spend_ledger_latest_sha256",
    "gpu_spend_ledger_latest_body_sha256",
    "gpu_spend_ledger_genesis_sha256",
    "gpu_spend_ledger_tip_record_sha256",
    "gpu_spend_ledger_file_sha256",
    "ec2_allocation_history_sha256",
    "snapshot_body_sha256",
}
_REQUIRED_TAGS = {
    "project": "keep-glm52",
    "owner": "jack.mazac",
    "model": "glm-5.2",
    "cost-allocation": "glm52-sky-campaign",
}
_CLOSED_INSTANCE_STATES = {
    "terminated",
    "stopped",
}


class GpuSpendSnapshotError(ValueError):
    """Raised when cumulative GPU spend authority is incomplete or divergent."""


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


def _sha_object(value: object) -> str:
    return _sha_raw(_canonical(value))


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise GpuSpendSnapshotError(f"{field} must be a lowercase SHA-256")
    return value


def _authenticate_raw(
    raw: bytes,
    expected_sha256: str,
    *,
    label: str,
) -> str:
    if not isinstance(raw, bytes):
        raise GpuSpendSnapshotError(f"{label} bytes must be bytes")
    expected = _require_digest(
        expected_sha256,
        field=f"expected {label} SHA-256",
    )
    actual = _sha_raw(raw)
    if actual != expected:
        raise GpuSpendSnapshotError(f"{label} exact-byte SHA-256 mismatch")
    return actual


def _object_from_raw(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GpuSpendSnapshotError(f"{label} is malformed JSON") from error
    if not isinstance(value, dict):
        raise GpuSpendSnapshotError(f"{label} must contain an object")
    return value


def _parse_time(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise GpuSpendSnapshotError(f"{field} must be timezone-aware")
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        raise GpuSpendSnapshotError(f"{field} must be ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GpuSpendSnapshotError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise GpuSpendSnapshotError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime | str, *, field: str) -> str:
    parsed = (
        value.astimezone(timezone.utc)
        if isinstance(value, datetime) and value.tzinfo is not None
        else _parse_time(value, field=field)
    )
    if not isinstance(parsed, datetime) or parsed.tzinfo is None:
        raise GpuSpendSnapshotError(f"{field} must be timezone-aware")
    return parsed.isoformat().replace("+00:00", "Z")


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GpuSpendSnapshotError(f"{field} must be an integer >= {minimum}")
    return value


def _money(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GpuSpendSnapshotError(f"{field} must be a finite number")
    if not math.isfinite(float(value)):
        raise GpuSpendSnapshotError(f"{field} must be a finite number")
    amount = Decimal(str(value))
    if amount < 0 or amount != amount.quantize(Decimal("0.01")):
        raise GpuSpendSnapshotError(f"{field} must be a nonnegative two-decimal amount")
    return amount


def _cost(seconds: int, hourly_cost: Decimal) -> Decimal:
    return (Decimal(seconds) * hourly_cost / Decimal(3600)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def _ledger_genesis(
    *,
    run_id: str,
    approval_sha256: str,
    approved_gpu_runtime_seconds: int,
    approved_gpu_cost_usd: float,
    hourly_cost_usd: float,
) -> str:
    return _sha_object(
        {
            "record_type": "glm52_gpu_spend_ledger_genesis_v1",
            "run_id": run_id,
            "approval_sha256": approval_sha256,
            "approved_gpu_runtime_seconds": approved_gpu_runtime_seconds,
            "approved_gpu_cost_usd": approved_gpu_cost_usd,
            "hourly_cost_usd": hourly_cost_usd,
        }
    )


def _validate_latest(
    latest_raw: bytes,
    *,
    expected_sha256: str,
    run_id: str,
    genesis_sha256: str,
) -> tuple[dict[str, object], str]:
    latest_sha256 = _authenticate_raw(
        latest_raw,
        expected_sha256,
        label="GPU spend latest marker",
    )
    latest = _object_from_raw(
        latest_raw,
        label="GPU spend latest marker",
    )
    if (
        set(latest) != _LATEST_FIELDS
        or latest.get("schema_version") != 1
        or latest.get("record_type") != "glm52_gpu_spend_ledger_latest_v1"
    ):
        raise GpuSpendSnapshotError("GPU spend latest marker schema mismatch")
    body = dict(latest)
    body_sha256 = body.pop("latest_body_sha256")
    if body_sha256 != _sha_object(body):
        raise GpuSpendSnapshotError("GPU spend latest marker body SHA-256 mismatch")
    if latest_raw != _canonical(latest) + b"\n":
        raise GpuSpendSnapshotError("GPU spend latest marker bytes are not canonical")
    if latest.get("run_id") != run_id:
        raise GpuSpendSnapshotError("GPU spend latest marker contains a foreign run")
    if latest.get("gpu_spend_authority_sha256") != genesis_sha256:
        raise GpuSpendSnapshotError(
            "GPU spend latest marker contains foreign spend authority"
        )
    count = _integer(
        latest.get("record_count"),
        field="GPU spend latest record_count",
    )
    keys = latest.get("record_keys")
    if (
        not isinstance(keys, list)
        or any(not isinstance(name, str) for name in keys)
        or len(keys) != count
    ):
        raise GpuSpendSnapshotError("GPU spend latest marker record inventory mismatch")
    if count == 0:
        if latest.get("latest_record_sha256") is not None:
            raise GpuSpendSnapshotError("empty GPU spend latest marker has a tip")
    else:
        tip = _require_digest(
            latest.get("latest_record_sha256"),
            field="GPU spend latest tip record SHA-256",
        )
        match = _RECORD_NAME.fullmatch(keys[-1])
        if match is None or match.group("sha") != tip:
            raise GpuSpendSnapshotError("GPU spend latest tip record SHA-256 mismatch")
    _require_digest(
        latest.get("ledger_sha256"),
        field="GPU spend latest ledger SHA-256",
    )
    return latest, latest_sha256


def _reconstruct_allocations(
    latest: Mapping[str, object],
    immutable_record_raw_by_key: Mapping[str, bytes],
    *,
    run_id: str,
    approval_sha256: str,
    genesis_sha256: str,
    observed_at: datetime,
) -> tuple[bytes, list[tuple[str, datetime, datetime]]]:
    declared_keys = list(latest["record_keys"])
    actual_keys = set(immutable_record_raw_by_key)
    declared_set = set(declared_keys)
    if actual_keys != declared_set:
        missing = declared_set - actual_keys
        if missing:
            raise GpuSpendSnapshotError(
                "GPU spend immutable record is missing: " + sorted(missing)[0]
            )
        raise GpuSpendSnapshotError(
            "GPU spend latest is stale or rollback: "
            "immutable record key set contains undeclared records"
        )
    prior_sha256 = genesis_sha256
    prior_timestamp: datetime | None = None
    active: tuple[str, str, datetime] | None = None
    allocations: list[tuple[str, datetime, datetime]] = []
    seen_instances: set[str] = set()
    ledger = bytearray()
    for index, name in enumerate(declared_keys):
        match = _RECORD_NAME.fullmatch(name)
        if match is None or int(match.group("index")) != index:
            raise GpuSpendSnapshotError("GPU spend immutable record name is invalid")
        raw = immutable_record_raw_by_key[name]
        if not isinstance(raw, bytes):
            raise GpuSpendSnapshotError(
                f"GPU spend immutable record bytes are invalid: {name}"
            )
        record = _object_from_raw(
            raw,
            label=f"GPU spend immutable record {name}",
        )
        event = record.get("event")
        expected_fields = (
            _START_RECORD_FIELDS
            if event == "allocation_started"
            else _END_RECORD_FIELDS
            if event == "allocation_ended"
            else set()
        )
        if (
            not expected_fields
            or set(record) != expected_fields
            or record.get("record_type") != "glm52_gpu_spend_event_v1"
        ):
            raise GpuSpendSnapshotError(
                f"GPU spend immutable record schema mismatch: {name}"
            )
        if raw != _canonical(record) + b"\n":
            raise GpuSpendSnapshotError(
                f"GPU spend immutable record bytes are not canonical: {name}"
            )
        body = dict(record)
        record_sha256 = body.pop("record_sha256")
        if (
            record_sha256 != _sha_object(body)
            or record_sha256 != match.group("sha")
            or event != match.group("event")
        ):
            raise GpuSpendSnapshotError(
                f"GPU spend immutable record identity mismatch: {name}"
            )
        if record.get("run_id") != run_id:
            raise GpuSpendSnapshotError(
                "GPU spend immutable record contains a foreign run"
            )
        if record.get("approval_sha256") != approval_sha256:
            raise GpuSpendSnapshotError(
                "GPU spend immutable record contains foreign approval"
            )
        if record.get("prior_record_sha256") != prior_sha256:
            raise GpuSpendSnapshotError(
                "GPU spend immutable record chain is noncontiguous"
            )
        timestamp = _parse_time(
            record.get("timestamp"),
            field="GPU spend record timestamp",
        )
        if prior_timestamp is not None and timestamp < prior_timestamp:
            raise GpuSpendSnapshotError("GPU spend record timestamps are time-skewed")
        if timestamp > observed_at:
            raise GpuSpendSnapshotError(
                "GPU spend observation time precedes a ledger record"
            )
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.startswith("i-"):
            raise GpuSpendSnapshotError(
                "GPU spend immutable record instance_id is invalid"
            )
        if event == "allocation_started":
            if active is not None:
                raise GpuSpendSnapshotError(
                    "GPU spend immutable chain has an open allocation"
                )
            job_id = record.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise GpuSpendSnapshotError(
                    "GPU spend immutable record job_id is invalid"
                )
            if instance_id in seen_instances:
                raise GpuSpendSnapshotError(
                    "GPU spend immutable chain has a duplicate allocation"
                )
            seen_instances.add(instance_id)
            active = (job_id, instance_id, timestamp)
        else:
            if active is None or active[1] != instance_id:
                raise GpuSpendSnapshotError(
                    "GPU spend immutable allocation end is noncontiguous"
                )
            duration = (timestamp - active[2]).total_seconds()
            if duration < 0 or not duration.is_integer():
                raise GpuSpendSnapshotError(
                    "GPU spend allocation duration must be whole seconds"
                )
            allocations.append((instance_id, active[2], timestamp))
            active = None
        prior_sha256 = str(record_sha256)
        prior_timestamp = timestamp
        ledger.extend(raw)
    if active is not None:
        raise GpuSpendSnapshotError("GPU spend snapshot requires no open allocation")
    if not allocations:
        raise GpuSpendSnapshotError(
            "GPU spend snapshot requires at least one closed allocation"
        )
    if prior_sha256 != latest.get("latest_record_sha256"):
        raise GpuSpendSnapshotError(
            "GPU spend immutable chain tip does not match latest marker"
        )
    if _sha_raw(ledger) != latest.get("ledger_sha256"):
        raise GpuSpendSnapshotError("complete GPU spend ledger file SHA-256 mismatch")
    return bytes(ledger), allocations


def _history_instances(
    history_raw: bytes,
    *,
    expected_sha256: str,
    run_id: str,
    allocations: list[tuple[str, datetime, datetime]],
    observed_at: datetime,
) -> tuple[str, list[str]]:
    history_sha256 = _authenticate_raw(
        history_raw,
        expected_sha256,
        label="EC2 allocation history",
    )
    history = _object_from_raw(
        history_raw,
        label="EC2 allocation history",
    )
    if history.get("NextToken") is not None:
        raise GpuSpendSnapshotError(
            "EC2 allocation history is paginated and incomplete"
        )
    reservations = history.get("Reservations")
    if not isinstance(reservations, list):
        raise GpuSpendSnapshotError(
            "EC2 allocation history Reservations must be a list"
        )
    instances: list[dict[str, object]] = []
    for reservation in reservations:
        if not isinstance(reservation, dict):
            raise GpuSpendSnapshotError("EC2 allocation history reservation is invalid")
        owner_id = reservation.get("OwnerId")
        if owner_id != APPROVED_ACCOUNT_ID:
            raise GpuSpendSnapshotError(
                "EC2 allocation history contains a foreign account"
            )
        values = reservation.get("Instances")
        if not isinstance(values, list) or any(
            not isinstance(item, dict) for item in values
        ):
            raise GpuSpendSnapshotError(
                "EC2 allocation history instance inventory is invalid"
            )
        instances.extend(values)
    by_instance: dict[str, dict[str, object]] = {}
    for instance in instances:
        instance_id = instance.get("InstanceId")
        if (
            not isinstance(instance_id, str)
            or not instance_id.startswith("i-")
            or instance_id in by_instance
        ):
            raise GpuSpendSnapshotError(
                "EC2 allocation history has duplicate or invalid instance IDs"
            )
        tags_raw = instance.get("Tags")
        if not isinstance(tags_raw, list):
            raise GpuSpendSnapshotError(
                "EC2 allocation history contains a foreign campaign instance"
            )
        tags: dict[str, str] = {}
        for tag in tags_raw:
            if (
                not isinstance(tag, dict)
                or set(tag) != {"Key", "Value"}
                or not isinstance(tag.get("Key"), str)
                or not isinstance(tag.get("Value"), str)
                or tag["Key"] in tags
            ):
                raise GpuSpendSnapshotError(
                    "EC2 allocation history tag inventory is invalid"
                )
            tags[tag["Key"]] = tag["Value"]
        expected_tags = {**_REQUIRED_TAGS, "campaign-run-id": run_id}
        if (
            instance.get("InstanceType") != APPROVED_INSTANCE_TYPE
            or any(tags.get(key) != value for key, value in expected_tags.items())
            or instance.get("InstanceLifecycle") is not None
        ):
            raise GpuSpendSnapshotError(
                "EC2 allocation history contains a foreign campaign instance"
            )
        placement = instance.get("Placement")
        availability_zone = (
            placement.get("AvailabilityZone") if isinstance(placement, dict) else None
        )
        if (
            not isinstance(availability_zone, str)
            or re.fullmatch(
                rf"{re.escape(APPROVED_REGION)}[a-z]",
                availability_zone,
            )
            is None
        ):
            raise GpuSpendSnapshotError(
                "EC2 allocation history contains a foreign region"
            )
        state = instance.get("State")
        state_name = state.get("Name") if isinstance(state, dict) else None
        if state_name not in _CLOSED_INSTANCE_STATES:
            raise GpuSpendSnapshotError(
                "EC2 allocation history contains an active instance"
            )
        by_instance[instance_id] = instance
    allocation_ids = [item[0] for item in allocations]
    if set(by_instance) != set(allocation_ids):
        raise GpuSpendSnapshotError(
            "EC2 allocation history does not exactly reconcile the ledger"
        )
    for instance_id, launched_at, ended_at in allocations:
        instance = by_instance[instance_id]
        history_launch = _parse_time(
            instance.get("LaunchTime"),
            field="EC2 allocation LaunchTime",
        )
        if history_launch != launched_at:
            raise GpuSpendSnapshotError(
                "EC2 allocation launch does not match the ledger"
            )
        match = _TERMINATION.search(str(instance.get("StateTransitionReason", "")))
        if match is None:
            raise GpuSpendSnapshotError(
                "EC2 allocation history lacks exact termination time"
            )
        history_end = datetime.strptime(
            match.group("timestamp"),
            "%Y-%m-%d %H:%M:%S",
        ).replace(tzinfo=timezone.utc)
        if history_end != ended_at:
            raise GpuSpendSnapshotError("EC2 allocation end does not match the ledger")
        if history_end > observed_at:
            raise GpuSpendSnapshotError(
                "GPU spend observation time precedes EC2 allocation history"
            )
    return history_sha256, allocation_ids


def build_gpu_spend_snapshot(
    *,
    descriptor_raw: bytes,
    expected_descriptor_sha256: str,
    approval_raw: bytes,
    expected_approval_sha256: str,
    latest_raw: bytes,
    expected_latest_sha256: str,
    immutable_record_raw_by_key: Mapping[str, bytes],
    ec2_allocation_history_raw: bytes,
    expected_ec2_allocation_history_sha256: str,
    observed_at: datetime | str,
) -> dict[str, object]:
    """Rebuild and authenticate closed cumulative spend from immutable inputs."""

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
    descriptor = _object_from_raw(
        descriptor_raw,
        label="Sky campaign descriptor",
    )
    approval = _object_from_raw(
        approval_raw,
        label="GPU spend approval",
    )
    try:
        validate_sky_campaign_descriptor(descriptor)
        validate_gpu_spend_approval(approval)
    except SkyCampaignValidationError as error:
        raise GpuSpendSnapshotError(str(error)) from error
    if descriptor.get("approval_sha256") != approval_sha256:
        raise GpuSpendSnapshotError(
            "descriptor does not bind the exact GPU spend approval bytes"
        )
    if (
        descriptor.get("approved_gpu_runtime_seconds")
        != int(approval["approved_gpu_hours"]) * 3600
        or descriptor.get("approved_gpu_cost_usd")
        != approval.get("approved_gpu_cost_usd")
        or descriptor.get("max_hourly_cost_usd") != approval.get("approved_hourly_usd")
    ):
        raise GpuSpendSnapshotError("descriptor and approval spend authorities diverge")
    run_id = str(descriptor["run_id"])
    observation = _parse_time(observed_at, field="observed_at")
    observation_iso = _iso_utc(observation, field="observed_at")
    genesis_sha256 = _ledger_genesis(
        run_id=run_id,
        approval_sha256=approval_sha256,
        approved_gpu_runtime_seconds=int(descriptor["approved_gpu_runtime_seconds"]),
        approved_gpu_cost_usd=float(descriptor["approved_gpu_cost_usd"]),
        hourly_cost_usd=float(descriptor["max_hourly_cost_usd"]),
    )
    latest, latest_sha256 = _validate_latest(
        latest_raw,
        expected_sha256=expected_latest_sha256,
        run_id=run_id,
        genesis_sha256=genesis_sha256,
    )
    ledger_raw, allocations = _reconstruct_allocations(
        latest,
        immutable_record_raw_by_key,
        run_id=run_id,
        approval_sha256=approval_sha256,
        genesis_sha256=genesis_sha256,
        observed_at=observation,
    )
    history_sha256, allocation_ids = _history_instances(
        ec2_allocation_history_raw,
        expected_sha256=expected_ec2_allocation_history_sha256,
        run_id=run_id,
        allocations=allocations,
        observed_at=observation,
    )
    consumed_seconds = sum(
        int((ended_at - launched_at).total_seconds())
        for _instance_id, launched_at, ended_at in allocations
    )
    approved_seconds = int(descriptor["approved_gpu_runtime_seconds"])
    if consumed_seconds > approved_seconds:
        raise GpuSpendSnapshotError("GPU spend ledger exceeds approved runtime")
    remaining_seconds = approved_seconds - consumed_seconds
    hourly_cost = Decimal(str(descriptor["max_hourly_cost_usd"]))
    approved_cost = Decimal(str(descriptor["approved_gpu_cost_usd"]))
    consumed_cost = _cost(consumed_seconds, hourly_cost)
    if consumed_cost > approved_cost:
        raise GpuSpendSnapshotError("GPU spend ledger exceeds approved cost")
    remaining_cost = (approved_cost - consumed_cost).quantize(Decimal("0.01"))
    qualification_seconds = min(
        QUALIFICATION_MAX_SECONDS,
        remaining_seconds,
    )
    qualification_cost = _cost(qualification_seconds, hourly_cost)
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_snapshot_v1",
        "run_id": run_id,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_sha256": descriptor_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "approval_sha256": approval_sha256,
        "approval_body_sha256": approval["approval_body_sha256"],
        "gpu_spend_ledger_latest_sha256": latest_sha256,
        "gpu_spend_ledger_latest_body_sha256": latest["latest_body_sha256"],
        "gpu_spend_ledger_genesis_sha256": genesis_sha256,
        "gpu_spend_ledger_record_count": len(latest["record_keys"]),
        "gpu_spend_ledger_tip_record_sha256": latest["latest_record_sha256"],
        "gpu_spend_ledger_file_sha256": _sha_raw(ledger_raw),
        "ec2_allocation_history_sha256": history_sha256,
        "ec2_allocation_instance_ids": allocation_ids,
        "observed_at": observation_iso,
        "approved_gpu_runtime_seconds": approved_seconds,
        "approved_gpu_cost_usd": float(approved_cost),
        "hourly_cost_usd": float(hourly_cost),
        "consumed_gpu_seconds": consumed_seconds,
        "remaining_gpu_seconds": remaining_seconds,
        "consumed_gpu_cost_usd": float(consumed_cost),
        "remaining_gpu_cost_usd": float(remaining_cost),
        "qualification_allowance_seconds": qualification_seconds,
        "qualification_allowance_cost_usd": float(qualification_cost),
        "open_allocation_count": 0,
    }
    snapshot = {
        **body,
        "snapshot_body_sha256": _sha_object(body),
    }
    return validate_gpu_spend_snapshot(snapshot)


def validate_gpu_spend_snapshot(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact self-hashed cumulative-spend snapshot schema."""

    if set(value) != _SNAPSHOT_FIELDS:
        raise GpuSpendSnapshotError("GPU spend snapshot schema mismatch")
    body = dict(value)
    actual_sha256 = body.pop("snapshot_body_sha256")
    if actual_sha256 != _sha_object(body):
        raise GpuSpendSnapshotError("GPU spend snapshot body SHA-256 mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("record_type") != "glm52_gpu_spend_snapshot_v1"
    ):
        raise GpuSpendSnapshotError("GPU spend snapshot schema mismatch")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise GpuSpendSnapshotError("GPU spend snapshot run_id is invalid")
    for field in _SNAPSHOT_DIGEST_FIELDS:
        _require_digest(value.get(field), field=field)
    if _iso_utc(value.get("observed_at"), field="observed_at") != value.get(
        "observed_at"
    ):
        raise GpuSpendSnapshotError(
            "GPU spend snapshot observed_at is not canonical UTC"
        )
    approved_seconds = _integer(
        value.get("approved_gpu_runtime_seconds"),
        field="approved_gpu_runtime_seconds",
        minimum=1,
    )
    consumed_seconds = _integer(
        value.get("consumed_gpu_seconds"),
        field="consumed_gpu_seconds",
    )
    remaining_seconds = _integer(
        value.get("remaining_gpu_seconds"),
        field="remaining_gpu_seconds",
    )
    allowance_seconds = _integer(
        value.get("qualification_allowance_seconds"),
        field="qualification_allowance_seconds",
    )
    record_count = _integer(
        value.get("gpu_spend_ledger_record_count"),
        field="gpu_spend_ledger_record_count",
        minimum=2,
    )
    open_count = _integer(
        value.get("open_allocation_count"),
        field="open_allocation_count",
    )
    if (
        approved_seconds != APPROVED_GPU_RUNTIME_SECONDS
        or consumed_seconds + remaining_seconds != approved_seconds
        or allowance_seconds != min(QUALIFICATION_MAX_SECONDS, remaining_seconds)
        or record_count % 2
        or open_count != 0
    ):
        raise GpuSpendSnapshotError(
            "GPU spend snapshot runtime accounting is inconsistent"
        )
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
    allowance_cost = _money(
        value.get("qualification_allowance_cost_usd"),
        field="qualification_allowance_cost_usd",
    )
    if (
        approved_cost != Decimal(str(APPROVED_GPU_COST_USD))
        or hourly_cost != Decimal(str(APPROVED_HOURLY_COST_USD))
        or consumed_cost != _cost(consumed_seconds, hourly_cost)
        or consumed_cost + remaining_cost != approved_cost
        or allowance_cost != _cost(allowance_seconds, hourly_cost)
    ):
        raise GpuSpendSnapshotError(
            "GPU spend snapshot dollar accounting is inconsistent"
        )
    instance_ids = value.get("ec2_allocation_instance_ids")
    if (
        not isinstance(instance_ids, list)
        or len(instance_ids) * 2 != record_count
        or len(set(instance_ids)) != len(instance_ids)
        or any(
            not isinstance(instance_id, str) or not instance_id.startswith("i-")
            for instance_id in instance_ids
        )
    ):
        raise GpuSpendSnapshotError(
            "GPU spend snapshot EC2 allocation inventory is inconsistent"
        )
    return dict(value)


__all__ = [
    "GpuSpendSnapshotError",
    "QUALIFICATION_MAX_SECONDS",
    "build_gpu_spend_snapshot",
    "validate_gpu_spend_snapshot",
]
