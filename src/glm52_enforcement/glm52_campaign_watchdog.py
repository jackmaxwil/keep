"""Enforcement-native health policy for the GLM-5.2 campaign watchdog."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Mapping


WatchdogStatus = Literal["waiting", "healthy", "warning", "alarm", "drained"]
SkyPilotJobStatus = Literal[
    "SUBMITTED",
    "PENDING",
    "RUNNING",
    "RECOVERING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
]
_SKYPILOT_JOB_STATUSES = {
    "SUBMITTED",
    "PENDING",
    "RUNNING",
    "RECOVERING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
}


@dataclass(frozen=True)
class CampaignObservation:
    now: datetime
    launch_deadline: datetime
    instance_ids: tuple[str, ...]
    instance_states: tuple[str, ...]
    instance_launch_time: datetime | None
    ec2_status_ok: bool | None
    ssm_online: bool | None
    systemd_state: str | None
    campaign_phase: str | None
    progress_at: datetime | None
    cloudwatch_alarms: tuple[str, ...]
    worker_failure: bool
    training_deferred: bool
    drained: bool
    campaign_mode: Literal["legacy_capacity_block", "skypilot"] = (
        "legacy_capacity_block"
    )
    submitted_at: datetime | None = None
    must_start_by: datetime | None = None
    sky_job_status: str | None = None
    dlq_depth: int = 0
    remaining_gpu_seconds: int | None = None
    remaining_gpu_cost_usd: float | None = None


@dataclass(frozen=True)
class CampaignWatchdogResult:
    status: WatchdogStatus
    alert: bool
    findings: tuple[str, ...]


@dataclass(frozen=True)
class CampaignNotificationDecision:
    event: Literal["alert", "recovery", "drained"] | None
    state: dict[str, str]


def campaign_notification_decision(
    *,
    previous: Mapping[str, object] | None,
    status: WatchdogStatus,
    findings: tuple[str, ...],
) -> CampaignNotificationDecision:
    """Choose one transition email while suppressing repeated alarm noise."""

    normalized = tuple(
        re.sub(r"stale for \d+ minutes", "stale for <minutes>", finding)
        for finding in findings
    )
    fingerprint = hashlib.sha256(
        _canonical_bytes({"status": status, "findings": normalized})
    ).hexdigest()
    state = {"status": status, "fingerprint": fingerprint}
    prior_status = previous.get("status") if previous else None
    prior_fingerprint = previous.get("fingerprint") if previous else None
    event: Literal["alert", "recovery", "drained"] | None = None
    if status == "drained" and prior_status != "drained":
        event = "drained"
    elif status == "alarm" and prior_fingerprint != fingerprint:
        event = "alert"
    elif prior_status == "alarm" and status != "alarm":
        event = "recovery"
    return CampaignNotificationDecision(event=event, state=state)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def build_campaign_heartbeat(
    *,
    run_id: str,
    phase: str,
    observed_at: datetime,
    ledger_record_sha256: str | None,
) -> dict[str, object]:
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    body: dict[str, object] = {
        "record_type": "glm52_campaign_watchdog_heartbeat_v1",
        "run_id": run_id,
        "phase": phase,
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
        "ledger_record_sha256": ledger_record_sha256,
    }
    return {
        **body,
        "heartbeat_body_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }


def validate_campaign_heartbeat(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("campaign heartbeat must be an object")
    required = {
        "record_type",
        "run_id",
        "phase",
        "observed_at",
        "ledger_record_sha256",
        "heartbeat_body_sha256",
    }
    if set(value) != required or value.get("record_type") != (
        "glm52_campaign_watchdog_heartbeat_v1"
    ):
        raise ValueError("campaign heartbeat schema mismatch")
    body = dict(value)
    actual = body.pop("heartbeat_body_sha256")
    expected = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if actual != expected:
        raise ValueError("campaign heartbeat body SHA-256 mismatch")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise ValueError("campaign heartbeat run_id is invalid")
    if not isinstance(value.get("phase"), str) or not value["phase"]:
        raise ValueError("campaign heartbeat phase is invalid")
    datetime.fromisoformat(str(value["observed_at"]).replace("Z", "+00:00"))
    ledger_sha = value.get("ledger_record_sha256")
    if ledger_sha is not None and (
        not isinstance(ledger_sha, str)
        or len(ledger_sha) != 64
        or any(character not in "0123456789abcdef" for character in ledger_sha)
    ):
        raise ValueError("campaign heartbeat ledger SHA-256 is invalid")
    return dict(value)


def build_skypilot_submission_marker(
    *,
    run_id: str,
    descriptor_body_sha256: str,
    submitted_at: datetime,
    must_start_by: datetime,
    sky_job_name: str,
) -> dict[str, object]:
    if submitted_at.tzinfo is None or must_start_by.tzinfo is None:
        raise ValueError("SkyPilot submission times must be timezone-aware")
    if must_start_by <= submitted_at:
        raise ValueError("must_start_by must follow submitted_at")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_skypilot_submission_v1",
        "run_id": run_id,
        "descriptor_body_sha256": descriptor_body_sha256,
        "submitted_at": submitted_at.isoformat().replace("+00:00", "Z"),
        "must_start_by": must_start_by.isoformat().replace("+00:00", "Z"),
        "sky_job_name": sky_job_name,
    }
    return {
        **body,
        "submission_body_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }


def validate_skypilot_submission_marker(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("SkyPilot submission marker must be an object")
    required = {
        "schema_version",
        "record_type",
        "run_id",
        "descriptor_body_sha256",
        "submitted_at",
        "must_start_by",
        "sky_job_name",
        "submission_body_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != 1
        or value.get("record_type") != "glm52_skypilot_submission_v1"
    ):
        raise ValueError("SkyPilot submission marker schema mismatch")
    body = dict(value)
    actual = body.pop("submission_body_sha256")
    expected = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if actual != expected:
        raise ValueError("SkyPilot submission marker body SHA-256 mismatch")
    for field in ("run_id", "sky_job_name"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"SkyPilot submission marker {field} is invalid")
    digest = value.get("descriptor_body_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("SkyPilot submission descriptor SHA-256 is invalid")
    submitted_at = datetime.fromisoformat(
        str(value["submitted_at"]).replace("Z", "+00:00")
    )
    must_start_by = datetime.fromisoformat(
        str(value["must_start_by"]).replace("Z", "+00:00")
    )
    if submitted_at.tzinfo is None or must_start_by.tzinfo is None:
        raise ValueError("SkyPilot submission times must be timezone-aware")
    if must_start_by <= submitted_at:
        raise ValueError("SkyPilot must_start_by must follow submitted_at")
    return dict(value)


def build_skypilot_job_status(
    *,
    run_id: str,
    descriptor_body_sha256: str,
    sky_job_name: str,
    status: SkyPilotJobStatus,
    instance_id: str | None,
    observed_at: datetime,
) -> dict[str, object]:
    if observed_at.tzinfo is None:
        raise ValueError("SkyPilot job status time must be timezone-aware")
    if status not in _SKYPILOT_JOB_STATUSES:
        raise ValueError("SkyPilot job status is invalid")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_skypilot_job_status_v1",
        "run_id": run_id,
        "descriptor_body_sha256": descriptor_body_sha256,
        "sky_job_name": sky_job_name,
        "status": status,
        "instance_id": instance_id,
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
    }
    return {
        **body,
        "status_body_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }


def validate_skypilot_job_status(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("SkyPilot job status must be an object")
    required = {
        "schema_version",
        "record_type",
        "run_id",
        "descriptor_body_sha256",
        "sky_job_name",
        "status",
        "instance_id",
        "observed_at",
        "status_body_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != 1
        or value.get("record_type") != "glm52_skypilot_job_status_v1"
    ):
        raise ValueError("SkyPilot job status schema mismatch")
    body = dict(value)
    actual = body.pop("status_body_sha256")
    expected = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if actual != expected:
        raise ValueError("SkyPilot job status body SHA-256 mismatch")
    for field in ("run_id", "sky_job_name"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"SkyPilot job status {field} is invalid")
    digest = value.get("descriptor_body_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("SkyPilot job status descriptor SHA-256 is invalid")
    if value.get("status") not in _SKYPILOT_JOB_STATUSES:
        raise ValueError("SkyPilot job status is invalid")
    instance_id = value.get("instance_id")
    if instance_id is not None and (
        not isinstance(instance_id, str) or not instance_id.startswith("i-")
    ):
        raise ValueError("SkyPilot job status instance_id is invalid")
    observed_at = datetime.fromisoformat(
        str(value["observed_at"]).replace("Z", "+00:00")
    )
    if observed_at.tzinfo is None:
        raise ValueError("SkyPilot job status time must be timezone-aware")
    return dict(value)


def evaluate_campaign_observation(
    observation: CampaignObservation,
) -> CampaignWatchdogResult:
    """Classify an observation without performing any AWS mutations."""

    if observation.drained:
        return CampaignWatchdogResult(
            status="drained",
            alert=False,
            findings=("CAMPAIGN_DRAINED.json authenticated",),
        )

    alarms: list[str] = []
    warnings: list[str] = []
    if observation.cloudwatch_alarms:
        alarms.append(
            "CloudWatch alarms active: " + ", ".join(observation.cloudwatch_alarms)
        )
    if observation.worker_failure:
        alarms.append("campaign worker failure marker present")
    if observation.training_deferred:
        alarms.append("training deferred")
    if observation.campaign_mode == "skypilot":
        if observation.sky_job_status in {
            "FAILED",
            "FAILED_SETUP",
            "FAILED_PRECHECKS",
            "CANCELLED",
        }:
            alarms.append("SkyPilot Managed Job failed")
        if observation.dlq_depth > 0:
            suffix = "" if observation.dlq_depth == 1 else "s"
            alarms.append(
                "controller dead-letter queue contains "
                f"{observation.dlq_depth} message{suffix}"
            )
        if (
            observation.remaining_gpu_seconds is not None
            and observation.remaining_gpu_seconds <= 0
        ):
            alarms.append("approved GPU runtime exhausted")
        elif (
            observation.remaining_gpu_seconds is not None
            and observation.remaining_gpu_seconds <= 3_600
        ):
            minutes = max(0, observation.remaining_gpu_seconds // 60)
            warnings.append(
                f"approved GPU runtime has {minutes} minutes remaining"
            )
        if (
            observation.remaining_gpu_cost_usd is not None
            and observation.remaining_gpu_cost_usd <= 0
        ):
            alarms.append("approved GPU cost exhausted")

    if not observation.instance_ids:
        if observation.campaign_mode == "skypilot":
            if (
                observation.must_start_by is not None
                and observation.now >= observation.must_start_by
            ):
                alarms.append(
                    "SkyPilot must-start deadline reached with no running GPU"
                )
            elif (
                observation.submitted_at is not None
                and observation.now
                >= observation.submitted_at + timedelta(minutes=20)
            ):
                alarms.append(
                    "no GPU running 20 minutes after SkyPilot submission"
                )
            elif not alarms:
                finding = (
                    "SkyPilot job pending; no GPU runtime consumed"
                    if observation.sky_job_status == "PENDING"
                    else "waiting for SkyPilot worker provisioning"
                )
                return CampaignWatchdogResult(
                    status="waiting",
                    alert=False,
                    findings=(finding,),
                )
        else:
            if observation.now >= observation.launch_deadline:
                alarms.append("no running campaign instance")
            elif not alarms:
                return CampaignWatchdogResult(
                    status="waiting",
                    alert=False,
                    findings=("waiting for Capacity Block delivery",),
                )
    else:
        if (
            observation.campaign_mode == "legacy_capacity_block"
            and observation.now >= observation.launch_deadline
            and "running" not in observation.instance_states
        ):
            alarms.append("no running campaign instance")
        elif (
            observation.campaign_mode == "skypilot"
            and "running" not in observation.instance_states
            and observation.submitted_at is not None
            and observation.now
            >= observation.submitted_at + timedelta(minutes=20)
        ):
            alarms.append(
                "no GPU running 20 minutes after SkyPilot submission"
            )
        if len(observation.instance_ids) > 1:
            alarms.append("multiple active campaign instances")
        if observation.ec2_status_ok is False:
            alarms.append("EC2 status checks unhealthy")
        elif observation.ec2_status_ok is None:
            warnings.append("EC2 status checks not available yet")

        if observation.ssm_online is False:
            if (
                observation.campaign_mode == "skypilot"
                and observation.instance_launch_time is not None
                and observation.now
                < observation.instance_launch_time + timedelta(minutes=10)
            ):
                warnings.append("SSM agent unavailable during bootstrap grace")
            else:
                alarms.append("SSM agent unavailable")
        elif observation.ssm_online is None:
            warnings.append("SSM status not available yet")

        if observation.systemd_state in {"failed", "inactive", "deactivating"}:
            alarms.append(
                f"systemd campaign service {observation.systemd_state}"
            )
        elif observation.systemd_state not in {"active", None}:
            warnings.append(
                f"systemd campaign service {observation.systemd_state}"
            )

        progress_origin = observation.progress_at or observation.instance_launch_time
        if progress_origin is None:
            warnings.append("campaign heartbeat/progress not published yet")
        else:
            age = observation.now - progress_origin
            if observation.progress_at is None and age <= timedelta(minutes=30):
                warnings.append("campaign heartbeat/progress not published yet")
            elif age >= timedelta(minutes=30):
                minutes = int(age.total_seconds() // 60)
                alarms.append(
                    f"campaign heartbeat/progress stale for {minutes} minutes"
                )

    if alarms:
        return CampaignWatchdogResult("alarm", True, tuple(alarms + warnings))
    if warnings:
        return CampaignWatchdogResult("warning", False, tuple(warnings))
    return CampaignWatchdogResult(
        "healthy",
        False,
        (
            "campaign control plane healthy"
            + (
                f"; phase={observation.campaign_phase}"
                if observation.campaign_phase
                else ""
            ),
        ),
    )


__all__ = [
    "CampaignObservation",
    "CampaignNotificationDecision",
    "CampaignWatchdogResult",
    "build_campaign_heartbeat",
    "build_skypilot_job_status",
    "build_skypilot_submission_marker",
    "campaign_notification_decision",
    "evaluate_campaign_observation",
    "validate_campaign_heartbeat",
    "validate_skypilot_job_status",
    "validate_skypilot_submission_marker",
]
