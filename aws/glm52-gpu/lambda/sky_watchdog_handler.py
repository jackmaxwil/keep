"""AWS-native ten-minute watchdog for the GLM-5.2 SkyPilot campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping

import boto3
from botocore.exceptions import ClientError
from sky_worker_start_v2_coordinator import (
    WorkerStartCoordinatorError,
    WorkerStartCoordinatorOutcome,
    WorkerStartCoordinatorRequest,
    WorkerStartCoordinatorServices,
    coordinate_worker_start_acceptance_v2,
)

from mlx_vq.quality.glm52_campaign_watchdog import (
    CampaignObservation,
    campaign_notification_decision,
    evaluate_campaign_observation,
    validate_campaign_heartbeat,
    validate_skypilot_job_status,
    validate_skypilot_submission_marker,
)
from mlx_vq.quality.glm52_h100_qualification import (
    build_h100_termination_requested,
    validate_h100_resume_ready,
    validate_h100_runtime_allocation,
    validate_h100_source_node_ready,
    validate_h100_termination_requested,
)
from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_GPU_COST_USD,
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import validate_must_start_job_binding
from mlx_vq.quality.glm52_sky_terminal_state import validate_sky_terminal_state

ACTIVE_STATES = {"pending", "running", "stopping", "stopped"}
_PRECONDITION_CODES = {"412", "PreconditionFailed"}
_SOURCE_STATES = {
    "pending",
    "running",
    "stopping",
    "stopped",
    "shutting-down",
    "terminated",
}
_TERMINABLE_SOURCE_STATES = {"pending", "running"}
_REQUIRED_WORKER_TAGS = {
    "project": "keep-glm52",
    "owner": "jack.mazac",
    "model": "glm-5.2",
    "cost-allocation": "glm52-sky-campaign",
}
_DYNAMIC_EVENT_RECORD_TYPE = "glm52_sky_qualification_watchdog_invocation_v1"
_DYNAMIC_EVENT_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "managed_mode",
    "descriptor_key",
    "intent_key",
    "intent_body_sha256",
}
_WORKER_START_EVENT_RECORD_TYPE = "glm52_sky_worker_start_v2_invocation_v1"
_WORKER_START_EVENT_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "descriptor_key",
    "descriptor_file_sha256",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "worker_latch_key",
    "worker_latch_version_id",
}
_WORKER_START_SUCCESS_STATUSES = {
    "accepted-initial",
    "accepted-managed-recovery",
    "idempotent-complete",
}
_V2_INTENT_FIELDS = {
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
    "intent_body_sha256",
}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_WORKER_START_BUCKET = re.compile(
    r"(?![0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$)"
    r"(?!xn--)(?!sthree-)(?!amzn_s3_demo_)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]"
)
_WORKER_START_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_WORKER_START_VERSION_ID = re.compile(r"[A-Za-z0-9._+=:/-]{1,1024}")
_APPROVED_ACCOUNT_ID = "246813579024"
_APPROVED_REGION = "us-west-2"


class WorkerStartAuthorityPending(RuntimeError):
    """Exact worker authority remains retryable rather than acknowledged."""


def _is_worker_start_event(event: object) -> bool:
    return isinstance(event, Mapping) and (
        event.get("record_type") == _WORKER_START_EVENT_RECORD_TYPE
        or "worker_latch_key" in event
        or "worker_latch_version_id" in event
    )


def _required_worker_start_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"worker-start v2 environment pin {name} is missing")
    return value


def _valid_worker_start_descriptor_key(value: str, *, run_id: str) -> bool:
    prefix = f"campaigns/{run_id}/submissions/qualification/"
    if not value.startswith(prefix):
        return False
    tail = value.removeprefix(prefix)
    return (
        tail.endswith(".json")
        and tail.rsplit("/", 1)[-1] != ".json"
        and "\\" not in value
        and "*" not in value
        and "?" not in value
        and not any(character.isspace() for character in value)
        and all(
            _WORKER_START_SAFE_NAME.fullmatch(part) is not None
            for part in tail.split("/")
        )
    )


def _validate_worker_start_event(event: object) -> dict[str, str]:
    if not isinstance(event, Mapping) or set(event) != _WORKER_START_EVENT_FIELDS:
        raise ValueError("worker-start v2 event schema mismatch")
    if (
        type(event.get("schema_version")) is not int
        or event["schema_version"] != 1
        or event.get("record_type") != _WORKER_START_EVENT_RECORD_TYPE
    ):
        raise ValueError("worker-start v2 event schema mismatch")
    request_fields = _WORKER_START_EVENT_FIELDS - {"schema_version", "record_type"}
    if any(type(event.get(field)) is not str for field in request_fields):
        raise ValueError("worker-start v2 request fields must be strings")

    region = _required_worker_start_environment("AWS_REGION")
    pins = {
        "account_id": _required_worker_start_environment(
            "WORKER_START_EXPECTED_ACCOUNT_ID"
        ),
        "bucket": _required_worker_start_environment("WORKER_START_BUCKET"),
        "descriptor_key": _required_worker_start_environment(
            "WORKER_START_DESCRIPTOR_KEY"
        ),
        "descriptor_file_sha256": _required_worker_start_environment(
            "WORKER_START_DESCRIPTOR_FILE_SHA256"
        ),
        "intent_key": _required_worker_start_environment("WORKER_START_INTENT_KEY"),
        "intent_file_sha256": _required_worker_start_environment(
            "WORKER_START_INTENT_FILE_SHA256"
        ),
        "intent_body_sha256": _required_worker_start_environment(
            "WORKER_START_INTENT_BODY_SHA256"
        ),
    }
    if region != _APPROVED_REGION or event["region"] != region:
        raise ValueError("worker-start v2 event region is foreign")
    if (
        pins["account_id"] != _APPROVED_ACCOUNT_ID
        or event["account_id"] != pins["account_id"]
    ):
        raise ValueError("worker-start v2 event account is foreign")
    for field, pinned_value in pins.items():
        if event[field] != pinned_value:
            raise ValueError(f"worker-start v2 event {field} differs from its pin")

    bucket = event["bucket"]
    if (
        _WORKER_START_BUCKET.fullmatch(bucket) is None
        or any(separator in bucket for separator in ("..", ".-", "-."))
        or bucket.endswith(("-s3alias", "--ol-s3", ".mrap", "--x-s3", "--table-s3"))
    ):
        raise ValueError("worker-start v2 bucket is invalid")
    for field in (
        "descriptor_file_sha256",
        "intent_file_sha256",
        "intent_body_sha256",
    ):
        if _HEX64.fullmatch(event[field]) is None:
            raise ValueError(f"worker-start v2 {field} is invalid")

    intent_match = re.fullmatch(
        r"campaigns/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/"
        r"submissions/qualification/intents/([0-9a-f]{64})/"
        r"SKYPILOT_SUBMISSION_INTENT\.json",
        event["intent_key"],
    )
    if intent_match is None or intent_match.group(2) != event["intent_body_sha256"]:
        raise ValueError("worker-start v2 intent key is invalid")
    run_id = intent_match.group(1)
    if not _valid_worker_start_descriptor_key(event["descriptor_key"], run_id=run_id):
        raise ValueError("worker-start v2 descriptor key is invalid")
    latch_match = re.fullmatch(
        rf"campaigns/{re.escape(run_id)}/monitor/must-start/qualification/"
        rf"{re.escape(event['intent_body_sha256'])}/worker-latches/"
        r"i-[0-9a-f]{17}/[0-9a-f]{64}\.json",
        event["worker_latch_key"],
    )
    if latch_match is None:
        raise ValueError("worker-start v2 worker latch key is invalid")
    version_id = event["worker_latch_version_id"]
    if (
        _WORKER_START_VERSION_ID.fullmatch(version_id) is None
        or version_id in {"null", "None"}
        or any(part == ".." for part in version_id.split("/"))
    ):
        raise ValueError("worker-start v2 worker latch VersionId is invalid")
    return {field: event[field] for field in request_fields}


def _worker_start_retry_sleep(
    *,
    context: object,
    services: WorkerStartCoordinatorServices,
    seconds: int,
    reason: str,
) -> None:
    remaining_time = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(remaining_time):
        raise WorkerStartAuthorityPending(
            f"worker-start v2 authority remains pending: {reason}"
        )
    remaining_millis = remaining_time()
    if type(remaining_millis) is not int or remaining_millis < 30_000:
        raise WorkerStartAuthorityPending(
            f"worker-start v2 authority remains pending: {reason}"
        )
    services.sleep(seconds)


def _handle_worker_start_event(
    *,
    event: object,
    context: object,
) -> dict[str, object]:
    request_fields = _validate_worker_start_event(event)
    region = request_fields["region"]
    services = WorkerStartCoordinatorServices(
        sts=boto3.client("sts", region_name=region),
        s3=boto3.client("s3", region_name=region),
        ec2=boto3.client("ec2", region_name=region),
        ssm=boto3.client("ssm", region_name=region),
        clock=lambda: datetime.now(timezone.utc),
        sleep=time.sleep,
    )
    request = WorkerStartCoordinatorRequest(**request_fields)
    for attempt in range(3):
        outcome = coordinate_worker_start_acceptance_v2(
            services=services,
            request=request,
        )
        if type(outcome) is not WorkerStartCoordinatorOutcome:
            raise WorkerStartCoordinatorError(
                "worker-start v2 coordinator outcome is malformed"
            )
        if outcome.status in _WORKER_START_SUCCESS_STATUSES:
            return asdict(outcome)
        if outcome.status != "waiting-worker-authority":
            raise WorkerStartCoordinatorError(
                "worker-start v2 coordinator outcome status is invalid"
            )
        if attempt == 2:
            raise WorkerStartAuthorityPending(
                f"worker-start v2 authority remains pending: {outcome.reason}"
            )
        _worker_start_retry_sleep(
            context=context,
            services=services,
            seconds=attempt + 1,
            reason=outcome.reason,
        )
    raise AssertionError("worker-start v2 retry loop is unreachable")


def _body(s3, bucket: str, key: str) -> bytes | None:  # noqa: ANN001
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise


def _json_body(s3, bucket: str, key: str) -> dict[str, Any] | None:  # noqa: ANN001
    raw = _body(s3, bucket, key)
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"s3://{bucket}/{key} must contain an object")
    return value


def _modified(s3, bucket: str, key: str) -> datetime | None:  # noqa: ANN001
    try:
        value = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise
    return _parse_time(value["LastModified"])


def _systemd_state(ssm, instance_id: str) -> str | None:  # noqa: ANN001
    command = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment="Read-only KEEP GLM-5.2 Sky watchdog",
        Parameters={
            "commands": [
                "systemctl show keep-glm52-campaign.service "
                "--property=ActiveState --value 2>/dev/null || true"
            ]
        },
        TimeoutSeconds=30,
    )
    command_id = command["Command"]["CommandId"]
    for _ in range(10):
        time.sleep(1)
        try:
            invocation = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
        except ClientError:
            continue
        if invocation.get("Status") in {"Pending", "InProgress", "Delayed"}:
            continue
        value = str(invocation.get("StandardOutputContent", "")).strip()
        return value or None
    return None


def _parse_time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("watchdog timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_v2_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"v2 intent {field} is invalid")
    return value


def _canonical_v2_time(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"v2 intent {field} must be canonical ISO-8601")
    parsed = _parse_time(value)
    canonical = parsed.isoformat().replace("+00:00", "Z")
    if value != canonical:
        raise ValueError(f"v2 intent {field} is not canonical")
    return canonical


def _require_v2_scoped_key(value: object, *, run_id: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(f"campaigns/{run_id}/")
        or ".." in value
        or "\\" in value
    ):
        raise ValueError(f"v2 intent {field} is invalid")
    return value


def _validate_dynamic_event(event: object, *, region: str) -> dict[str, object]:
    if not isinstance(event, Mapping) or set(event) != _DYNAMIC_EVENT_FIELDS:
        raise ValueError("qualification watchdog event schema mismatch")
    if (
        type(event.get("schema_version")) is not int
        or event["schema_version"] != 1
        or event.get("record_type") != _DYNAMIC_EVENT_RECORD_TYPE
        or event.get("managed_mode") != "qualification"
    ):
        raise ValueError("qualification watchdog event schema mismatch")
    run_id = event.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("qualification watchdog event run_id is invalid")
    if region != _APPROVED_REGION:
        raise ValueError("qualification watchdog event region is foreign")
    _require_v2_scoped_key(
        event.get("descriptor_key"),
        run_id=run_id,
        field="descriptor_key",
    )
    intent_sha = _require_v2_sha(
        event.get("intent_body_sha256"), field="intent_body_sha256"
    )
    intent_key = _require_v2_scoped_key(
        event.get("intent_key"), run_id=run_id, field="intent_key"
    )
    expected_intent_key = (
        f"campaigns/{run_id}/submissions/qualification/intents/{intent_sha}/"
        "SKYPILOT_SUBMISSION_INTENT.json"
    )
    if intent_key != expected_intent_key:
        raise ValueError("qualification watchdog event intent_key is not exact")
    return dict(event)


def _validate_v2_qualification_intent(
    intent: object,
    *,
    event: Mapping[str, object],
    descriptor: Mapping[str, object],
    descriptor_raw: bytes,
    bucket: str,
) -> dict[str, object]:
    if not isinstance(intent, Mapping) or set(intent) != _V2_INTENT_FIELDS:
        raise ValueError("v2 intent schema mismatch")
    if (
        type(intent.get("schema_version")) is not int
        or intent["schema_version"] != 2
        or intent.get("record_type") != "glm52_sky_submission_intent_v2"
    ):
        raise ValueError("v2 intent schema mismatch")
    body = dict(intent)
    digest = _require_v2_sha(body.pop("intent_body_sha256"), field="body SHA-256")
    if digest != hashlib.sha256(_canonical(body)).hexdigest():
        raise ValueError("v2 intent body SHA-256 mismatch")
    if digest != event["intent_body_sha256"]:
        raise ValueError("v2 intent does not match event digest")
    run_id = event["run_id"]
    if intent.get("run_id") != run_id or intent.get("managed_mode") != "qualification":
        raise ValueError("v2 intent run or mode is foreign")
    if (
        intent.get("account_id") != _APPROVED_ACCOUNT_ID
        or intent.get("region") != _APPROVED_REGION
        or descriptor.get("account_id") != _APPROVED_ACCOUNT_ID
        or descriptor.get("region") != _APPROVED_REGION
        or descriptor.get("bucket") != bucket
    ):
        raise ValueError("v2 intent account, region, or bucket is foreign")
    descriptor_file_sha256 = hashlib.sha256(descriptor_raw).hexdigest()
    expected_descriptor = {
        "descriptor_key": event["descriptor_key"],
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "must_start_by": descriptor["must_start_by"],
    }
    if descriptor.get("campaign_descriptor_key") != event["descriptor_key"] or any(
        intent.get(field) != expected for field, expected in expected_descriptor.items()
    ):
        raise ValueError("v2 intent descriptor authority mismatch")
    for field in (name for name in intent if name.endswith("_sha256")):
        _require_v2_sha(intent[field], field=field)
    for field in (
        "cache_seed_acceptance_key",
        "gpu_spend_snapshot_key",
        "qualification_submission_ready_key",
    ):
        _require_v2_scoped_key(intent.get(field), run_id=str(run_id), field=field)
    for field in (
        "remaining_gpu_seconds",
        "qualification_allowance_seconds",
        "open_allocation_count",
    ):
        if type(intent.get(field)) is not int or int(intent[field]) < 0:
            raise ValueError(f"v2 intent {field} is invalid")
    for field in ("remaining_gpu_cost_usd", "qualification_allowance_cost_usd"):
        value = intent.get(field)
        if type(value) is not float or not math.isfinite(value) or value < 0:
            raise ValueError(f"v2 intent {field} is invalid")
    if (
        int(intent["remaining_gpu_seconds"]) <= 0
        or int(intent["qualification_allowance_seconds"]) <= 0
        or int(intent["open_allocation_count"]) != 0
    ):
        raise ValueError("v2 qualification intent allocation authority is invalid")
    expected_job_name = f"{run_id}-qualification"
    if intent.get("sky_job_name") != expected_job_name:
        raise ValueError("v2 intent SkyPilot job name is foreign")
    intent_at = _canonical_v2_time(intent.get("intent_at"), field="intent_at")
    must_start_by = _canonical_v2_time(
        intent.get("must_start_by"), field="must_start_by"
    )
    if _parse_time(intent_at) >= _parse_time(must_start_by):
        raise ValueError("v2 intent deadline is stale")
    expected_key = (
        f"campaigns/{run_id}/submissions/qualification/intents/{digest}/"
        "SKYPILOT_SUBMISSION_INTENT.json"
    )
    if event["intent_key"] != expected_key:
        raise ValueError("v2 intent key is not exact")
    return dict(intent)


def _validate_v2_job_binding(
    binding: object,
    *,
    bucket: str,
    event: Mapping[str, object],
    intent: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(binding, Mapping):
        raise ValueError("v2 job binding is invalid")
    try:
        value = validate_must_start_job_binding(binding)
    except ValueError as error:
        raise ValueError(f"v2 job binding is invalid: {error}") from error
    expected = {
        "run_id": event["run_id"],
        "managed_mode": "qualification",
        "account_id": _APPROVED_ACCOUNT_ID,
        "region": _APPROVED_REGION,
        "bucket": bucket,
        "descriptor_body_sha256": intent["descriptor_body_sha256"],
        "submission_body_sha256": intent["intent_body_sha256"],
        "sky_job_name": intent["sky_job_name"],
        "must_start_by": intent["must_start_by"],
        "descriptor_key": event["descriptor_key"],
        "descriptor_file_sha256": intent["descriptor_file_sha256"],
        "submission_key": event["intent_key"],
        "submission_submitted_at": intent["intent_at"],
    }
    if any(
        value.get(field) != expected_value for field, expected_value in expected.items()
    ):
        raise ValueError("v2 job binding authority mismatch")
    if _parse_time(value["bound_at"]) < _parse_time(intent["intent_at"]):
        raise ValueError("v2 job binding predates intent")
    return value


def _has_v2_qualification_intent(s3, *, bucket: str, run_id: str) -> bool:  # noqa: ANN001
    prefix = f"campaigns/{run_id}/submissions/qualification/intents/"
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=1_000,
    )
    contents = response.get("Contents", [])
    if not isinstance(contents, list):
        raise ValueError("v2 qualification intent listing is invalid")
    for item in contents:
        if not isinstance(item, Mapping):
            raise ValueError("v2 qualification intent listing is invalid")
        key = item.get("Key")
        if not isinstance(key, str):
            raise ValueError("v2 qualification intent listing is invalid")
        relative = key.removeprefix(prefix)
        parts = relative.split("/")
        if (
            len(parts) == 2
            and _HEX64.fullmatch(parts[0]) is not None
            and parts[1] == "SKYPILOT_SUBMISSION_INTENT.json"
        ):
            return True
    if response.get("IsTruncated") is True:
        raise ValueError("v2 qualification intent listing is truncated")
    return False


def _expected_spend_authority(descriptor: Mapping[str, object]) -> str:
    genesis = {
        "record_type": "glm52_gpu_spend_ledger_genesis_v1",
        "run_id": descriptor["run_id"],
        "approval_sha256": descriptor["approval_sha256"],
        "approved_gpu_runtime_seconds": descriptor["approved_gpu_runtime_seconds"],
        "approved_gpu_cost_usd": descriptor["approved_gpu_cost_usd"],
        "hourly_cost_usd": descriptor["max_hourly_cost_usd"],
    }
    return hashlib.sha256(_canonical(genesis)).hexdigest()


def _flatten_instances(response: Mapping[str, object]) -> list[dict[str, Any]]:
    reservations = response.get("Reservations", [])
    if not isinstance(reservations, list):
        raise ValueError("EC2 describe-instances Reservations is invalid")
    instances: list[dict[str, Any]] = []
    for reservation in reservations:
        if not isinstance(reservation, dict):
            raise ValueError("EC2 describe-instances reservation is invalid")
        rows = reservation.get("Instances", [])
        if not isinstance(rows, list):
            raise ValueError("EC2 describe-instances Instances is invalid")
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("EC2 describe-instances instance is invalid")
        instances.extend(rows)
    return instances


def _validate_qualification_instance(
    instance: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
    expected_instance_id: str | None = None,
) -> dict[str, Any]:
    instance_id = instance.get("InstanceId")
    if expected_instance_id is not None and instance_id != expected_instance_id:
        raise ValueError("qualification EC2 source instance identity mismatch")
    if instance.get("InstanceType") != descriptor["instance_type"]:
        raise ValueError("qualification EC2 instance type mismatch")
    if instance.get("InstanceLifecycle") not in (None, ""):
        raise ValueError("qualification EC2 instance is not on-demand")
    if instance.get("CapacityReservationId") is not None:
        raise ValueError("qualification EC2 instance uses a capacity reservation")
    reservation = instance.get("CapacityReservationSpecification")
    if isinstance(reservation, dict) and reservation.get("CapacityReservationTarget"):
        raise ValueError("qualification EC2 instance uses a capacity reservation")
    state_value = instance.get("State")
    if not isinstance(state_value, dict) or state_value.get("Name") not in (
        _SOURCE_STATES
    ):
        raise ValueError("qualification EC2 instance state is invalid")
    placement = instance.get("Placement")
    availability_zone = (
        placement.get("AvailabilityZone") if isinstance(placement, dict) else None
    )
    if not isinstance(availability_zone, str) or not availability_zone.startswith(
        str(descriptor["region"])
    ):
        raise ValueError("qualification EC2 instance region mismatch")
    raw_tags = instance.get("Tags")
    if not isinstance(raw_tags, list):
        raise ValueError("qualification EC2 instance tags are missing")
    tags: dict[str, str] = {}
    for tag in raw_tags:
        if not isinstance(tag, dict):
            raise ValueError("qualification EC2 instance tag is invalid")
        key = tag.get("Key")
        value = tag.get("Value")
        if not isinstance(key, str) or not isinstance(value, str) or key in tags:
            raise ValueError("qualification EC2 instance tag is invalid")
        tags[key] = value
    required_tags = {
        **_REQUIRED_WORKER_TAGS,
        "campaign-run-id": str(descriptor["run_id"]),
    }
    mismatched = [
        key for key, expected in required_tags.items() if tags.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            "qualification EC2 instance tag mismatch: " + ", ".join(sorted(mismatched))
        )
    return dict(instance)


def _active_qualification_instances(
    ec2,  # noqa: ANN001
    *,
    descriptor: Mapping[str, object],
) -> list[dict[str, Any]]:
    run_id = str(descriptor["run_id"])
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:project", "Values": ["keep-glm52"]},
            {"Name": "tag:owner", "Values": ["jack.mazac"]},
            {"Name": "tag:model", "Values": ["glm-5.2"]},
            {"Name": "tag:campaign-run-id", "Values": [run_id]},
            {
                "Name": "tag:cost-allocation",
                "Values": ["glm52-sky-campaign"],
            },
            {
                "Name": "instance-type",
                "Values": [str(descriptor["instance_type"])],
            },
            {
                "Name": "instance-state-name",
                "Values": sorted(_TERMINABLE_SOURCE_STATES),
            },
        ]
    )
    instances = _flatten_instances(response)
    return [
        _validate_qualification_instance(instance, descriptor=descriptor)
        for instance in instances
    ]


def _exact_source_instance(
    ec2,  # noqa: ANN001
    *,
    descriptor: Mapping[str, object],
    source_instance_id: str,
) -> dict[str, Any] | None:
    try:
        response = ec2.describe_instances(InstanceIds=[source_instance_id])
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) in {
            "InvalidInstanceID.NotFound",
        }:
            return None
        raise
    instances = _flatten_instances(response)
    if not instances:
        return None
    if len(instances) != 1:
        raise ValueError("qualification source EC2 lookup is ambiguous")
    return _validate_qualification_instance(
        instances[0],
        descriptor=descriptor,
        expected_instance_id=source_instance_id,
    )


def _conditional_termination_request(
    s3,  # noqa: ANN001
    *,
    bucket: str,
    key: str,
    marker: Mapping[str, object],
    source_marker: Mapping[str, object],
) -> tuple[dict[str, object], bool]:
    body = _canonical(marker) + b"\n"
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            IfNoneMatch="*",
            Metadata={
                "glm52-run-id": str(marker["run_id"]),
                "glm52-request-body-sha256": str(marker["request_body_sha256"]),
            },
        )
        return dict(marker), True
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) not in (
            _PRECONDITION_CODES
        ):
            raise
    winner = _json_body(s3, bucket, key)
    if winner is None:
        raise ValueError("qualification termination request winner disappeared")
    return (
        validate_h100_termination_requested(
            winner,
            source_marker=source_marker,
        ),
        False,
    )


def coordinate_h100_qualification_termination(
    *,
    s3,  # noqa: ANN001
    ec2,  # noqa: ANN001
    bucket: str,
    prefix: str,
    descriptor: Mapping[str, object],
    submission: Mapping[str, object],
    submission_kind: str,
    now: datetime,
    expected_runtime_job_id: str | None = None,
) -> dict[str, object]:
    """Durably request, then idempotently terminate only the source worker."""

    if submission_kind != "qualification":
        return {
            "status": "not-applicable",
            "source_instance_id": None,
            "request_body_sha256": None,
        }
    descriptor = validate_sky_campaign_descriptor(descriptor)
    if expected_runtime_job_id is None:
        submission = validate_skypilot_submission_marker(submission)
        if (
            submission["run_id"] != descriptor["run_id"]
            or submission["descriptor_body_sha256"]
            != descriptor["descriptor_body_sha256"]
        ):
            raise ValueError("qualification submission authority mismatch")
        expected_runtime_job_id = str(submission["sky_job_name"])
    if now.tzinfo is None:
        raise ValueError("qualification coordinator time must be timezone-aware")
    now = now.astimezone(timezone.utc)

    source_key = f"{prefix}/qualification/SOURCE_NODE_READY.json"
    request_key = f"{prefix}/qualification/QUALIFICATION_TERMINATION_REQUESTED.json"
    ready_key = f"{prefix}/qualification/H100_RESUME_READY.json"
    allocation_key = f"{prefix}/runtime/GPU_RUNTIME_ALLOCATION.json"
    source_raw = _json_body(s3, bucket, source_key)
    request_raw = _json_body(s3, bucket, request_key)
    ready_raw = _json_body(s3, bucket, ready_key)
    if source_raw is None:
        if request_raw is not None or ready_raw is not None:
            raise ValueError(
                "qualification request or readiness exists without source authority"
            )
        return {
            "status": "awaiting-source",
            "source_instance_id": None,
            "request_body_sha256": None,
        }

    source = validate_h100_source_node_ready(
        source_raw,
        expected_run_id=str(descriptor["run_id"]),
        expected_campaign_identity_sha256=str(descriptor["campaign_identity_sha256"]),
        expected_repo_tar_sha256=str(descriptor["repo_tar_sha256"]),
        expected_descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
    )
    if _parse_time(source["published_at"]) > now:
        raise ValueError("qualification source publication is in the future")
    allocation_raw = _json_body(s3, bucket, allocation_key)
    if allocation_raw is None:
        raise ValueError("qualification source lacks a runtime allocation")
    allocation = validate_h100_runtime_allocation(
        allocation_raw,
        expected_run_id=str(descriptor["run_id"]),
        expected_job_id=expected_runtime_job_id,
        expected_approval_sha256=str(descriptor["approval_sha256"]),
        expected_gpu_spend_authority_sha256=_expected_spend_authority(descriptor),
        max_remaining_gpu_seconds=int(descriptor["approved_gpu_runtime_seconds"]),
        max_estimated_gpu_cost_usd=float(descriptor["approved_gpu_cost_usd"]),
    )
    if _parse_time(allocation["observed_at"]) > now:
        raise ValueError("qualification allocation observation is in the future")

    request: dict[str, object] | None = None
    if request_raw is not None:
        request = validate_h100_termination_requested(
            request_raw,
            source_marker=source,
        )
        if _parse_time(request["requested_at"]) > now:
            raise ValueError("qualification termination request is in the future")
    if ready_raw is not None and request is None:
        raise ValueError("qualification readiness exists without a termination request")

    source_id = str(source["instance_id"])
    if ready_raw is not None:
        ready = validate_h100_resume_ready(ready_raw)
        expected_ready = {
            "run_id": descriptor["run_id"],
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "repo_tar_sha256": descriptor["repo_tar_sha256"],
            "qualification_cache_manifest_sha256": descriptor["artifacts"][
                "qualification_cache_manifest_sha256"
            ],
            "first_instance_id": source_id,
            "first_allocation_record_sha256": source["allocation_record_sha256"],
            "source_checkpoint_marker_sha256": source["checkpoint_marker_sha256"],
            "replacement_instance_id": allocation["instance_id"],
            "replacement_allocation_record_sha256": allocation[
                "gpu_spend_record_sha256"
            ],
        }
        mismatched = [
            field
            for field, expected in expected_ready.items()
            if ready[field] != expected
        ]
        if mismatched:
            raise ValueError(
                "qualification readiness authority mismatch: "
                + ", ".join(sorted(mismatched))
            )
        if _parse_time(ready["completed_at"]) < _parse_time(request["requested_at"]):
            raise ValueError("qualification readiness predates the termination request")
        return {
            "status": "ready",
            "source_instance_id": source_id,
            "request_body_sha256": request["request_body_sha256"],
        }

    source_instance = _exact_source_instance(
        ec2,
        descriptor=descriptor,
        source_instance_id=source_id,
    )
    active_instances = _active_qualification_instances(
        ec2,
        descriptor=descriptor,
    )
    active_ids = [str(instance["InstanceId"]) for instance in active_instances]
    if len(active_ids) > 1:
        raise ValueError("qualification active worker set is ambiguous")
    if (
        active_ids
        and active_ids != [source_id]
        and active_ids != [str(allocation["instance_id"])]
    ):
        raise ValueError(
            "qualification replacement does not match the current allocation"
        )

    if source_instance is None:
        if request is None:
            raise ValueError(
                "qualification source disappeared without a termination request"
            )
        return {
            "status": ("replacement-protected" if active_ids else "source-terminated"),
            "source_instance_id": source_id,
            "request_body_sha256": request["request_body_sha256"],
        }
    source_state = str(source_instance["State"]["Name"])
    if source_state in _TERMINABLE_SOURCE_STATES:
        validate_h100_runtime_allocation(
            allocation,
            expected_instance_id=source_id,
            expected_gpu_spend_record_sha256=str(source["allocation_record_sha256"]),
            expected_allocation_body_sha256=str(source["allocation_body_sha256"]),
        )
        if _parse_time(source["published_at"]) < _parse_time(allocation["observed_at"]):
            raise ValueError("qualification source publication predates its allocation")

    if request is None:
        if source_state not in _TERMINABLE_SOURCE_STATES:
            raise ValueError(
                "qualification source stopped without a termination request"
            )
        if active_ids != [source_id]:
            raise ValueError("qualification requires exactly one active source worker")
        candidate = build_h100_termination_requested(
            source_marker=source,
            requested_at=now,
        )
        request, request_created = _conditional_termination_request(
            s3,
            bucket=bucket,
            key=request_key,
            marker=candidate,
            source_marker=source,
        )
        request_status = (
            "termination-requested" if request_created else "termination-race-converged"
        )
        if _parse_time(request["requested_at"]) > now:
            raise ValueError("qualification termination request is in the future")
    else:
        request_status = "termination-retried"

    if source_state in _TERMINABLE_SOURCE_STATES:
        if active_ids != [source_id]:
            raise ValueError(
                "qualification active source set changed before termination"
            )
        # The authenticated marker is durable before this exact one-ID mutation.
        ec2.terminate_instances(InstanceIds=[source_id])
        return {
            "status": request_status,
            "source_instance_id": source_id,
            "request_body_sha256": request["request_body_sha256"],
        }
    if source_state not in _SOURCE_STATES:
        raise ValueError("qualification source EC2 state is invalid")
    if source_id in active_ids:
        raise ValueError("qualification source state conflicts with active set")
    return {
        "status": ("replacement-protected" if active_ids else "source-transitioning"),
        "source_instance_id": source_id,
        "request_body_sha256": request["request_body_sha256"],
    }


def _handle_dynamic_qualification_event(
    *,
    event: Mapping[str, object],
    s3,  # noqa: ANN001
    ec2,  # noqa: ANN001
    bucket: str,
    descriptor_key: str,
) -> dict[str, object]:
    if event["descriptor_key"] != descriptor_key:
        raise ValueError("qualification watchdog event descriptor key is foreign")
    descriptor_raw = _body(s3, bucket, descriptor_key)
    if descriptor_raw is None:
        raise ValueError("qualification watchdog descriptor is missing")
    descriptor_value = json.loads(descriptor_raw)
    if not isinstance(descriptor_value, dict):
        raise ValueError("qualification watchdog descriptor must be an object")
    descriptor = validate_sky_campaign_descriptor(descriptor_value)
    if descriptor["run_id"] != event["run_id"]:
        raise ValueError("qualification watchdog event run_id is foreign")
    intent_raw = _json_body(s3, bucket, str(event["intent_key"]))
    if intent_raw is None:
        raise ValueError("qualification watchdog v2 intent is missing")
    intent = _validate_v2_qualification_intent(
        intent_raw,
        event=event,
        descriptor=descriptor,
        descriptor_raw=descriptor_raw,
        bucket=bucket,
    )
    now = datetime.now(timezone.utc)
    if _parse_time(intent["intent_at"]) > now:
        raise ValueError("qualification watchdog v2 intent_at is in the future")
    binding_key = (
        f"campaigns/{event['run_id']}/monitor/must-start/qualification/"
        f"{event['intent_body_sha256']}/JOB_BINDING.json"
    )
    binding_raw = _json_body(s3, bucket, binding_key)
    if binding_raw is None:
        return {"status": "awaiting-job-binding", "run_id": event["run_id"]}
    binding = _validate_v2_job_binding(
        binding_raw,
        bucket=bucket,
        event=event,
        intent=intent,
    )
    bound_at = _parse_time(binding["bound_at"])
    if bound_at > now:
        raise ValueError("qualification watchdog v2 binding is in the future")
    if bound_at > _parse_time(intent["must_start_by"]):
        raise ValueError("qualification watchdog v2 binding follows must_start_by")
    if now >= _parse_time(intent["must_start_by"]):
        raise ValueError("qualification watchdog v2 intent is stale")
    result = coordinate_h100_qualification_termination(
        s3=s3,
        ec2=ec2,
        bucket=bucket,
        prefix=f"campaigns/{event['run_id']}",
        descriptor=descriptor,
        submission=intent,
        submission_kind="qualification",
        now=now,
        expected_runtime_job_id=str(binding["target_job_id"]),
    )
    return {"run_id": event["run_id"], **result}


def lambda_handler(_event, _context):  # noqa: ANN001
    if _is_worker_start_event(_event):
        return _handle_worker_start_event(event=_event, context=_context)
    region = os.environ.get("AWS_REGION", "us-west-2")
    bucket = os.environ["CAMPAIGN_BUCKET"]
    descriptor_key = os.environ["CAMPAIGN_DESCRIPTOR_KEY"]
    dynamic_event = None
    if isinstance(_event, Mapping) and (
        _event.get("record_type") == _DYNAMIC_EVENT_RECORD_TYPE
        or "schema_version" in _event
        or "record_type" in _event
    ):
        dynamic_event = _validate_dynamic_event(_event, region=region)
    s3 = boto3.client("s3", region_name=region)
    ec2 = boto3.client("ec2", region_name=region)
    if dynamic_event is not None:
        return _handle_dynamic_qualification_event(
            event=dynamic_event,
            s3=s3,
            ec2=ec2,
            bucket=bucket,
            descriptor_key=descriptor_key,
        )
    topic_arn = os.environ["ALERT_TOPIC_ARN"]
    dlq_url = os.environ["WATCHDOG_DLQ_URL"]
    rule_name = os.environ["WATCHDOG_RULE_NAME"]
    ssm = boto3.client("ssm", region_name=region)
    cloudwatch = boto3.client("cloudwatch", region_name=region)
    sqs = boto3.client("sqs", region_name=region)
    sns = boto3.client("sns", region_name=region)
    events = boto3.client("events", region_name=region)

    descriptor = validate_sky_campaign_descriptor(
        _json_body(s3, bucket, descriptor_key) or {}
    )
    run_id = str(descriptor["run_id"])
    prefix = f"campaigns/{run_id}"
    if dynamic_event is None and _has_v2_qualification_intent(
        s3,
        bucket=bucket,
        run_id=run_id,
    ):
        raise ValueError("v2 intent requires an exact qualification watchdog event")
    submission_raw = None
    submission_kind = ""
    for candidate_kind, candidate_name in (
        ("production", "SKYPILOT_SUBMITTED.json"),
        ("qualification", "H100_QUALIFICATION_SUBMITTED.json"),
        ("cache-seed", "QUALIFICATION_CACHE_SEED_SUBMITTED.json"),
    ):
        candidate = _json_body(
            s3,
            bucket,
            f"{prefix}/monitor/{candidate_name}",
        )
        if candidate is not None:
            submission_raw = candidate
            submission_kind = candidate_kind
            break
    if submission_raw is None:
        return {"status": "not-submitted", "run_id": run_id}
    submission = validate_skypilot_submission_marker(submission_raw)
    if (
        submission["run_id"] != run_id
        or submission["descriptor_body_sha256"] != descriptor["descriptor_body_sha256"]
    ):
        raise ValueError("SkyPilot submission marker authority mismatch")
    now = datetime.now(timezone.utc)
    qualification_coordination = coordinate_h100_qualification_termination(
        s3=s3,
        ec2=ec2,
        bucket=bucket,
        prefix=prefix,
        descriptor=descriptor,
        submission=submission,
        submission_kind=submission_kind,
        now=now,
    )

    reservations = ec2.describe_instances(
        Filters=[
            {"Name": "tag:campaign-run-id", "Values": [run_id]},
            {"Name": "instance-type", "Values": ["p5.48xlarge"]},
            {
                "Name": "instance-state-name",
                "Values": sorted(ACTIVE_STATES),
            },
        ]
    ).get("Reservations", [])
    instances = [
        instance
        for reservation in reservations
        for instance in reservation.get("Instances", [])
    ]
    instances.sort(key=lambda item: str(item.get("LaunchTime", "")))
    instance_ids = tuple(str(item["InstanceId"]) for item in instances)
    instance_states = tuple(
        str(item.get("State", {}).get("Name", "")) for item in instances
    )
    running = [
        instance
        for instance in instances
        if instance.get("State", {}).get("Name") == "running"
    ]
    launch_time = (
        min(_parse_time(item["LaunchTime"]) for item in instances)
        if instances
        else None
    )

    ec2_status_ok: bool | None = None
    if running:
        status = ec2.describe_instance_status(
            InstanceIds=[str(item["InstanceId"]) for item in running],
            IncludeAllInstances=True,
        ).get("InstanceStatuses", [])
        pairs = [
            (
                item.get("SystemStatus", {}).get("Status"),
                item.get("InstanceStatus", {}).get("Status"),
            )
            for item in status
        ]
        if any("impaired" in pair for pair in pairs):
            ec2_status_ok = False
        elif len(pairs) == len(running) and all(pair == ("ok", "ok") for pair in pairs):
            ec2_status_ok = True

    ssm_online: bool | None = None
    systemd_state: str | None = None
    if running:
        information = ssm.describe_instance_information(
            Filters=[
                {
                    "Key": "InstanceIds",
                    "Values": [str(item["InstanceId"]) for item in running],
                }
            ]
        ).get("InstanceInformationList", [])
        ssm_online = len(information) == len(running) and all(
            item.get("PingStatus") == "Online" for item in information
        )
        if ssm_online:
            systemd_state = _systemd_state(
                ssm,
                str(running[0]["InstanceId"]),
            )

    heartbeat_value = _json_body(s3, bucket, f"{prefix}/monitor/heartbeat.json")
    heartbeat = None
    alarms: list[str] = []
    if heartbeat_value is not None:
        try:
            heartbeat = validate_campaign_heartbeat(heartbeat_value)
            if heartbeat["run_id"] != run_id:
                raise ValueError("campaign heartbeat run_id mismatch")
        except ValueError:
            alarms.append("INVALID_CAMPAIGN_HEARTBEAT")
    phase = str(heartbeat["phase"]) if heartbeat else None
    progress_at = _parse_time(heartbeat["observed_at"]) if heartbeat else None

    sky_status_raw = _json_body(
        s3,
        bucket,
        f"{prefix}/monitor/SKY_JOB_STATUS.json",
    )
    sky_job_status = None
    if sky_status_raw is not None:
        try:
            sky_status = validate_skypilot_job_status(sky_status_raw)
            if (
                sky_status["run_id"] != run_id
                or sky_status["descriptor_body_sha256"]
                != descriptor["descriptor_body_sha256"]
                or sky_status["sky_job_name"] != submission["sky_job_name"]
            ):
                raise ValueError("SkyPilot job status authority mismatch")
            sky_job_status = str(sky_status["status"])
        except ValueError:
            alarms.append("INVALID_SKYPILOT_JOB_STATUS")
    spend_status = _json_body(
        s3,
        bucket,
        f"{prefix}/runtime/GPU_SPEND_STATUS.json",
    )
    allocation = _json_body(
        s3,
        bucket,
        f"{prefix}/runtime/GPU_RUNTIME_ALLOCATION.json",
    )
    allocation_is_current = allocation is not None and (
        spend_status is None
        or _parse_time(allocation["observed_at"])
        >= _parse_time(spend_status["ended_at"])
    )
    spend_value = allocation if allocation_is_current else (spend_status or {})
    remaining_seconds = spend_value.get("remaining_gpu_seconds")
    estimated_cost = spend_value.get("estimated_gpu_cost_usd")
    if allocation_is_current and allocation is not None:
        observed_at = _parse_time(allocation["observed_at"])
        elapsed = max(0, int((now - observed_at).total_seconds()))
        if isinstance(remaining_seconds, (int, float)):
            remaining_seconds = max(0, int(remaining_seconds) - elapsed)
        if isinstance(estimated_cost, (int, float)):
            estimated_cost = float(estimated_cost) + elapsed * 55.04 / 3600
    remaining_cost = (
        APPROVED_GPU_COST_USD - float(estimated_cost)
        if isinstance(estimated_cost, (int, float))
        else None
    )

    metric_alarms = cloudwatch.describe_alarms(
        AlarmNamePrefix="keep-glm52",
        StateValue="ALARM",
    ).get("MetricAlarms", [])
    alarms.extend(str(item["AlarmName"]) for item in metric_alarms)
    dlq = sqs.get_queue_attributes(
        QueueUrl=dlq_url,
        AttributeNames=["ApproximateNumberOfMessages"],
    )
    dlq_depth = int(dlq.get("Attributes", {}).get("ApproximateNumberOfMessages", "0"))

    progress_keys = {
        "heartbeat": f"{prefix}/monitor/heartbeat.json",
        "teacher_checkpoint": f"{prefix}/teacher-checkpoints/latest.json",
        "training_checkpoint": f"{prefix}/training-checkpoints/latest.json",
        "campaign_ledger": f"{prefix}/ledger/campaign-ledger.jsonl",
        "spend_ledger": f"{prefix}/runtime/GPU_SPEND_LEDGER.jsonl",
        "worker_failure": f"{prefix}/CAMPAIGN_FAILED.json",
    }
    progress = {name: _modified(s3, bucket, key) for name, key in progress_keys.items()}
    failure_at = progress["worker_failure"]
    heartbeat_at = progress["heartbeat"]
    failure = failure_at is not None and (
        heartbeat_at is None or failure_at >= heartbeat_at
    )
    deferred = _body(s3, bucket, f"{prefix}/TRAINING_DEFERRED.json") is not None
    drained = False
    terminal = _json_body(
        s3,
        bucket,
        f"{prefix}/runtime/TERMINAL_VERIFIED.json",
    )
    if terminal is not None:
        drained_raw = _body(s3, bucket, f"{prefix}/CAMPAIGN_DRAINED.json")
        campaign_ledger = _body(
            s3,
            bucket,
            f"{prefix}/ledger/campaign-ledger.jsonl",
        )
        spend_ledger = _body(
            s3,
            bucket,
            f"{prefix}/runtime/GPU_SPEND_LEDGER.jsonl",
        )
        if not all((drained_raw, campaign_ledger, spend_ledger)):
            alarms.append("INCOMPLETE_TERMINAL_AUTHORITIES")
        else:
            result = validate_sky_terminal_state(
                run_id=run_id,
                execution_deadline=str(
                    terminal.get("execution_deadline")
                    or (json.loads(drained_raw)["execution_deadline"])
                ),
                gpu_spend_authority_sha256=str(
                    json.loads(campaign_ledger.splitlines()[0])[
                        "gpu_spend_authority_sha256"
                    ]
                ),
                drained_raw=drained_raw,
                campaign_ledger_raw=campaign_ledger,
                spend_ledger_raw=spend_ledger,
            )
            drained = result["terminal_state_authenticated"] is True

    observation = CampaignObservation(
        now=now,
        launch_deadline=_parse_time(submission["submitted_at"]),
        instance_ids=instance_ids,
        instance_states=instance_states,
        instance_launch_time=launch_time,
        ec2_status_ok=ec2_status_ok,
        ssm_online=ssm_online,
        systemd_state=systemd_state,
        campaign_phase=phase,
        progress_at=progress_at,
        cloudwatch_alarms=tuple(sorted(set(alarms))),
        worker_failure=failure,
        training_deferred=deferred,
        drained=drained,
        campaign_mode="skypilot",
        submitted_at=_parse_time(submission["submitted_at"]),
        must_start_by=_parse_time(submission["must_start_by"]),
        sky_job_status=sky_job_status,
        dlq_depth=dlq_depth,
        remaining_gpu_seconds=(
            int(remaining_seconds)
            if isinstance(remaining_seconds, (int, float))
            else None
        ),
        remaining_gpu_cost_usd=remaining_cost,
    )
    result = evaluate_campaign_observation(observation)
    state_key = f"{prefix}/monitor/WATCHDOG_STATE.json"
    previous = _json_body(s3, bucket, state_key)
    decision = campaign_notification_decision(
        previous=previous,
        status=result.status,
        findings=result.findings,
    )
    deferred_alert = deferred and (
        previous is None or previous.get("training_deferred") != "true"
    )
    decision.state["training_deferred"] = "true" if deferred else "false"
    report = {
        "schema_version": 1,
        "record_type": "glm52_sky_watchdog_report_v1",
        "run_id": run_id,
        "observed_at": now.isoformat().replace("+00:00", "Z"),
        "status": result.status,
        "findings": result.findings,
        "instance_ids": instance_ids,
        "instance_states": instance_states,
        "campaign_phase": phase,
        "submission_kind": submission_kind,
        "qualification_coordination": qualification_coordination,
        "sky_job_status": sky_job_status,
        "remaining_gpu_seconds": observation.remaining_gpu_seconds,
        "remaining_gpu_cost_usd": remaining_cost,
        "dlq_depth": dlq_depth,
        "s3_progress": {
            name: (
                value.isoformat().replace("+00:00", "Z") if value is not None else None
            )
            for name, value in progress.items()
        },
    }
    if decision.event is not None:
        sns.publish(
            TopicArn=topic_arn,
            Subject=f"[GLM52] {decision.event.upper()} Sky campaign",
            Message=json.dumps(report, sort_keys=True, indent=2),
        )
    if deferred_alert:
        sns.publish(
            TopicArn=topic_arn,
            Subject="[GLM52] TRAINING DEFERRED",
            Message=json.dumps(report, sort_keys=True, indent=2),
        )
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/monitor/watchdog-{now.strftime('%Y%m%dT%H%M%SZ')}.json",
        Body=json.dumps(report, sort_keys=True, separators=(",", ":")).encode() + b"\n",
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=bucket,
        Key=state_key,
        Body=json.dumps(decision.state, sort_keys=True, separators=(",", ":")).encode()
        + b"\n",
        ContentType="application/json",
    )
    if drained:
        events.disable_rule(Name=rule_name)
    return report
