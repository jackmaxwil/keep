#!/usr/bin/env python3
"""Collect and evaluate the live GLM-5.2 campaign control-plane state."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_quality_module(name: str, filename: str) -> Any:
    path = REPO_ROOT / "src/mlx_vq/quality" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_watchdog = _load_quality_module(
    "_glm52_campaign_watchdog", "glm52_campaign_watchdog.py"
)
_controller = _load_quality_module(
    "_glm52_capacity_block_controller", "glm52_capacity_block_controller.py"
)
CampaignObservation = _watchdog.CampaignObservation
campaign_notification_decision = _watchdog.campaign_notification_decision
evaluate_campaign_observation = _watchdog.evaluate_campaign_observation
validate_campaign_heartbeat = _watchdog.validate_campaign_heartbeat
validate_campaign_descriptor = _controller.validate_campaign_descriptor


ACTIVE_STATES = {"pending", "running", "stopping", "stopped"}


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp is not timezone-aware: {value}")
    return parsed.astimezone(timezone.utc)


def _aws(
    arguments: Sequence[str], *, region: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["aws", *arguments, "--region", region],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"AWS CLI failed ({' '.join(arguments)}): {message}")
    return result


def _aws_json(arguments: Sequence[str], *, region: str) -> dict[str, Any]:
    result = _aws([*arguments, "--output", "json"], region=region)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError(f"AWS CLI returned a non-object for {' '.join(arguments)}")
    return value


def _s3_head(bucket: str, key: str, *, region: str) -> dict[str, Any] | None:
    result = _aws(
        ["s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json"],
        region=region,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = json.loads(result.stdout)
    return value if isinstance(value, dict) else None


def _s3_body(bucket: str, key: str, *, region: str) -> str | None:
    result = _aws(
        ["s3", "cp", f"s3://{bucket}/{key}", "-", "--only-show-errors"],
        region=region,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _latest_prefix_object(
    bucket: str, prefix: str, *, region: str
) -> dict[str, Any] | None:
    value = _aws_json(
        ["s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix],
        region=region,
    )
    objects = [item for item in value.get("Contents", []) if isinstance(item, dict)]
    if not objects:
        return None
    return max(objects, key=lambda item: _parse_time(str(item["LastModified"])))


def _systemd_status(instance_id: str, *, region: str, timeout: int) -> dict[str, Any]:
    command = (
        "printf 'ACTIVE_STATE='; systemctl show keep-glm52-campaign.service "
        "--property=ActiveState --value 2>/dev/null || true; "
        "printf 'SUB_STATE='; systemctl show keep-glm52-campaign.service "
        "--property=SubState --value 2>/dev/null || true; "
        "journalctl -u keep-glm52-campaign.service -n 20 --no-pager 2>/dev/null || true"
    )
    sent = _aws_json(
        [
            "ssm",
            "send-command",
            "--instance-ids",
            instance_id,
            "--document-name",
            "AWS-RunShellScript",
            "--comment",
            "Read-only KEEP GLM-5.2 campaign watchdog",
            "--parameters",
            json.dumps({"commands": [command]}, separators=(",", ":")),
        ],
        region=region,
    )
    command_id = str(sent["Command"]["CommandId"])
    deadline = time.monotonic() + timeout
    invocation: dict[str, Any] = {}
    while time.monotonic() < deadline:
        result = _aws(
            [
                "ssm",
                "get-command-invocation",
                "--command-id",
                command_id,
                "--instance-id",
                instance_id,
                "--output",
                "json",
            ],
            region=region,
            check=False,
        )
        if result.returncode == 0:
            invocation = json.loads(result.stdout)
            if invocation.get("Status") not in {"Pending", "InProgress", "Delayed"}:
                break
        time.sleep(2)
    output = str(invocation.get("StandardOutputContent", ""))
    active_state = None
    sub_state = None
    for line in output.splitlines():
        if line.startswith("ACTIVE_STATE="):
            active_state = line.partition("=")[2].strip() or None
        elif line.startswith("SUB_STATE="):
            sub_state = line.partition("=")[2].strip() or None
    return {
        "command_id": command_id,
        "command_status": invocation.get("Status", "TimedOut"),
        "active_state": active_state,
        "sub_state": sub_state,
        "recent_journal": output.splitlines()[-20:],
        "stderr": str(invocation.get("StandardErrorContent", "")).strip(),
    }


def collect_campaign_state(
    *,
    region: str,
    parameter_name: str,
    now: datetime,
    inspect_systemd: bool,
    ssm_timeout: int,
) -> tuple[CampaignObservation, dict[str, Any]]:
    parameter = _aws_json(
        ["ssm", "get-parameter", "--name", parameter_name], region=region
    )
    descriptor = validate_campaign_descriptor(
        json.loads(str(parameter["Parameter"]["Value"]))
    )
    region = str(descriptor["region"])
    reservation = _aws_json(
        [
            "ec2",
            "describe-capacity-reservations",
            "--capacity-reservation-ids",
            str(descriptor["capacity_reservation_id"]),
        ],
        region=region,
    )["CapacityReservations"][0]
    block_start = _parse_time(str(reservation["StartDate"]))
    launch_deadline = block_start + timedelta(minutes=10)

    raw_instances = _aws_json(
        [
            "ec2",
            "describe-instances",
            "--filters",
            f"Name=tag:campaign-run-id,Values={descriptor['run_id']}",
        ],
        region=region,
    )
    instances = [
        item
        for group in raw_instances.get("Reservations", [])
        for item in group.get("Instances", [])
        if item.get("State", {}).get("Name") in ACTIVE_STATES
    ]
    instances.sort(key=lambda item: str(item.get("LaunchTime", "")))
    instance_ids = tuple(str(item["InstanceId"]) for item in instances)
    instance_states = tuple(str(item["State"]["Name"]) for item in instances)
    running_ids = [
        str(item["InstanceId"])
        for item in instances
        if item.get("State", {}).get("Name") == "running"
    ]
    launch_times = [
        _parse_time(str(item["LaunchTime"])) for item in instances if item.get("LaunchTime")
    ]
    instance_launch_time = min(launch_times) if launch_times else None

    ec2_status_ok: bool | None = None
    status_details: dict[str, Any] = {}
    if running_ids:
        statuses = _aws_json(
            [
                "ec2",
                "describe-instance-status",
                "--include-all-instances",
                "--instance-ids",
                *running_ids,
            ],
            region=region,
        ).get("InstanceStatuses", [])
        status_details = {str(item["InstanceId"]): item for item in statuses}
        values = [
            (
                item.get("SystemStatus", {}).get("Status"),
                item.get("InstanceStatus", {}).get("Status"),
            )
            for item in statuses
        ]
        if any("impaired" in pair for pair in values):
            ec2_status_ok = False
        elif len(values) == len(running_ids) and all(pair == ("ok", "ok") for pair in values):
            ec2_status_ok = True

    ssm_online: bool | None = None
    ssm_details: dict[str, Any] = {}
    if running_ids:
        ssm = _aws_json(
            [
                "ssm",
                "describe-instance-information",
                "--filters",
                "Key=InstanceIds,Values=" + ",".join(running_ids),
            ],
            region=region,
        ).get("InstanceInformationList", [])
        ssm_details = {str(item["InstanceId"]): item for item in ssm}
        ssm_online = len(ssm) == len(running_ids) and all(
            item.get("PingStatus") == "Online" for item in ssm
        )

    systemd: dict[str, Any] | None = None
    if inspect_systemd and running_ids and ssm_online:
        systemd = _systemd_status(running_ids[0], region=region, timeout=ssm_timeout)
    systemd_state = systemd.get("active_state") if systemd else None

    bucket = str(descriptor["bucket"])
    campaign_prefix = f"campaigns/{descriptor['run_id']}"
    keys = {
        "heartbeat": f"{campaign_prefix}/monitor/heartbeat.json",
        "ledger": f"{campaign_prefix}/ledger/campaign-ledger.jsonl",
        "teacher_checkpoint": f"{campaign_prefix}/teacher-checkpoints/latest.json",
        "training_checkpoint": f"{campaign_prefix}/training-checkpoints/latest.json",
        "drained": f"{campaign_prefix}/CAMPAIGN_DRAINED.json",
        "training_deferred": f"{campaign_prefix}/TRAINING_DEFERRED.json",
        "worker_failure": f"{campaign_prefix}/CAMPAIGN_FAILED.json",
    }
    heads = {name: _s3_head(bucket, key, region=region) for name, key in keys.items()}
    heartbeat: dict[str, object] | None = None
    heartbeat_error: str | None = None
    heartbeat_body = _s3_body(bucket, keys["heartbeat"], region=region)
    if heartbeat_body is not None:
        try:
            heartbeat = validate_campaign_heartbeat(json.loads(heartbeat_body))
            if heartbeat["run_id"] != descriptor["run_id"]:
                raise ValueError("campaign heartbeat run_id mismatch")
        except (ValueError, json.JSONDecodeError) as error:
            heartbeat_error = str(error)
    progress_at = (
        _parse_time(str(heartbeat["observed_at"])) if heartbeat is not None else None
    )
    campaign_phase = str(heartbeat["phase"]) if heartbeat is not None else None
    if campaign_phase is None:
        ledger_body = _s3_body(bucket, keys["ledger"], region=region)
        if ledger_body:
            lines = [line for line in ledger_body.splitlines() if line.strip()]
            if lines:
                campaign_phase = str(json.loads(lines[-1]).get("phase") or "") or None

    capture_progress = _latest_prefix_object(
        bucket, f"{campaign_prefix}/teacher-captures/heartbeat-", region=region
    )
    cloudwatch = _aws_json(
        [
            "cloudwatch",
            "describe-alarms",
            "--alarm-name-prefix",
            "keep-glm52",
            "--state-value",
            "ALARM",
        ],
        region=region,
    )
    active_alarms = tuple(
        sorted(str(item["AlarmName"]) for item in cloudwatch.get("MetricAlarms", []))
    )
    if heartbeat_error:
        active_alarms += ("INVALID_CAMPAIGN_HEARTBEAT",)

    heartbeat_head = heads["heartbeat"]
    failure_head = heads["worker_failure"]
    failure_active = failure_head is not None and (
        heartbeat_head is None
        or _parse_time(str(failure_head["LastModified"]))
        >= _parse_time(str(heartbeat_head["LastModified"]))
    )
    drained_body = _s3_body(bucket, keys["drained"], region=region)
    drained = False
    drained_error: str | None = None
    if drained_body is not None:
        try:
            drained_value = json.loads(drained_body)
            drained = (
                drained_value.get("record_type") == "glm52_teich_campaign_drained_v1"
                and drained_value.get("run_id") == descriptor["run_id"]
            )
            if not drained:
                drained_error = "CAMPAIGN_DRAINED.json identity mismatch"
        except json.JSONDecodeError as error:
            drained_error = f"CAMPAIGN_DRAINED.json invalid JSON: {error}"
    if drained_error:
        active_alarms += ("INVALID_CAMPAIGN_DRAINED_MARKER",)

    observation = CampaignObservation(
        now=now,
        launch_deadline=launch_deadline,
        instance_ids=instance_ids,
        instance_states=instance_states,
        instance_launch_time=instance_launch_time,
        ec2_status_ok=ec2_status_ok,
        ssm_online=ssm_online,
        systemd_state=systemd_state,
        campaign_phase=campaign_phase,
        progress_at=progress_at,
        cloudwatch_alarms=active_alarms,
        worker_failure=failure_active,
        training_deferred=heads["training_deferred"] is not None,
        drained=drained,
    )
    details = {
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "capacity_reservation_id": descriptor["capacity_reservation_id"],
        "capacity_reservation_state": reservation.get("State"),
        "capacity_block_start": block_start.isoformat().replace("+00:00", "Z"),
        "capacity_block_end": descriptor["capacity_block_end"],
        "launch_deadline": launch_deadline.isoformat().replace("+00:00", "Z"),
        "instances": [
            {
                "instance_id": item["InstanceId"],
                "state": item["State"]["Name"],
                "launch_time": str(item.get("LaunchTime", "")),
            }
            for item in instances
        ],
        "ec2_status": status_details,
        "ssm": ssm_details,
        "systemd": systemd,
        "campaign_phase": campaign_phase,
        "heartbeat": heartbeat,
        "heartbeat_error": heartbeat_error,
        "s3_progress": {
            name: (value.get("LastModified") if value else None)
            for name, value in heads.items()
        }
        | {
            "teacher_capture_heartbeat": (
                capture_progress.get("LastModified") if capture_progress else None
            ),
            "teacher_capture_heartbeat_key": (
                capture_progress.get("Key") if capture_progress else None
            ),
        },
        "cloudwatch_alarms": active_alarms,
    }
    return observation, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--parameter", default="/keep-glm52/campaign/active")
    parser.add_argument("--now", help="Override current UTC time for policy rehearsal")
    parser.add_argument("--no-systemd", action="store_true")
    parser.add_argument("--ssm-timeout", type=int, default=30)
    parser.add_argument("--sns-topic-arn")
    parser.add_argument("--notification-state", type=Path)
    args = parser.parse_args()
    now = _parse_time(args.now) if args.now else datetime.now(timezone.utc)
    observation, details = collect_campaign_state(
        region=args.region,
        parameter_name=args.parameter,
        now=now,
        inspect_systemd=not args.no_systemd,
        ssm_timeout=args.ssm_timeout,
    )
    result = evaluate_campaign_observation(observation)
    payload = {
        "record_type": "glm52_campaign_watchdog_report_v1",
        "observed_at": now.isoformat().replace("+00:00", "Z"),
        "status": result.status,
        "alert": result.alert,
        "findings": result.findings,
        **details,
    }
    notification: dict[str, Any] | None = None
    if args.sns_topic_arn:
        if args.notification_state is None:
            parser.error("--notification-state is required with --sns-topic-arn")
        previous: dict[str, object] | None = None
        if args.notification_state.is_file():
            value = json.loads(args.notification_state.read_text())
            if isinstance(value, dict):
                previous = value
        decision = campaign_notification_decision(
            previous=previous,
            status=result.status,
            findings=result.findings,
        )
        notification = {"event": decision.event, "published": False}
        if decision.event is not None:
            subject_label = {
                "alert": "ALERT",
                "recovery": "RECOVERED",
                "drained": "DRAINED",
            }[decision.event]
            subject = f"[GLM52] {subject_label} {details.get('campaign_phase') or 'campaign'}"
            message = json.dumps(
                {
                    "campaign": "glm52-teich-20260717",
                    "event": decision.event,
                    "observed_at": payload["observed_at"],
                    "status": result.status,
                    "findings": result.findings,
                    "phase": details.get("campaign_phase"),
                    "instances": details.get("instances"),
                    "s3_progress": details.get("s3_progress"),
                    "cloudwatch_alarms": details.get("cloudwatch_alarms"),
                },
                sort_keys=True,
                indent=2,
                default=str,
            )
            _aws(
                [
                    "sns",
                    "publish",
                    "--topic-arn",
                    args.sns_topic_arn,
                    "--subject",
                    subject,
                    "--message",
                    message,
                ],
                region=args.region,
            )
            notification["published"] = True
        args.notification_state.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.notification_state.with_suffix(
            args.notification_state.suffix + ".tmp"
        )
        temporary.write_text(json.dumps(decision.state, sort_keys=True) + "\n")
        temporary.replace(args.notification_state)
    payload["notification"] = notification
    print(json.dumps(payload, sort_keys=True, indent=2, default=str))
    return 2 if result.alert else 0


if __name__ == "__main__":
    raise SystemExit(main())
