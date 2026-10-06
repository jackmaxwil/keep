"""Pure, dependency-injected logic for the Capacity Block EventBridge Lambda."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping

_HEX64 = re.compile(r"[0-9a-f]{64}")


class CapacityBlockEventError(ValueError):
    pass


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _iso_utc(value: object) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(value, str):
        raise CapacityBlockEventError("event end date must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CapacityBlockEventError("event end date is invalid") from error
    if parsed.tzinfo is None:
        raise CapacityBlockEventError("event end date must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_campaign_descriptor(value: Mapping[str, object]) -> dict[str, object]:
    required = {
        "schema_version",
        "run_id",
        "account_id",
        "region",
        "capacity_reservation_id",
        "capacity_block_end",
        "availability_zone",
        "subnet_id",
        "security_group_id",
        "launch_template_id",
        "launch_template_version",
        "instance_type",
        "instance_count",
        "bucket",
        "repo_tar_key",
        "repo_tar_sha256",
        "campaign_descriptor_key",
        "artifacts",
        "descriptor_body_sha256",
    }
    if set(value) != required or value.get("schema_version") != 1:
        raise CapacityBlockEventError("campaign descriptor schema mismatch")
    body = dict(value)
    digest = body.pop("descriptor_body_sha256")
    if digest != hashlib.sha256(_canonical_bytes(body)).hexdigest():
        raise CapacityBlockEventError("campaign descriptor body SHA-256 mismatch")
    for field in (
        "run_id",
        "account_id",
        "region",
        "capacity_reservation_id",
        "availability_zone",
        "subnet_id",
        "security_group_id",
        "launch_template_id",
        "launch_template_version",
        "instance_type",
        "bucket",
        "repo_tar_key",
        "campaign_descriptor_key",
    ):
        if not isinstance(value.get(field), str) or not value[field]:
            raise CapacityBlockEventError(f"campaign descriptor {field} is invalid")
    if value.get("instance_count") != 1:
        raise CapacityBlockEventError("campaign descriptor permits exactly one instance")
    if _HEX64.fullmatch(str(value.get("repo_tar_sha256"))) is None:
        raise CapacityBlockEventError("campaign descriptor repo tar SHA-256 is invalid")
    artifacts = value.get("artifacts")
    artifact_keys = {
        "source_snapshot_prefix",
        "source_snapshot_sha256",
        "non_vq_prefix",
        "non_vq_package_sha256",
        "teich_pack_key",
        "teich_pack_sha256",
        "frozen_prompt_pack_key",
        "frozen_prompt_pack_sha256",
        "training_baseline_prefix",
        "training_baseline_sha256",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != artifact_keys or any(
        not isinstance(artifacts[key], str) or not artifacts[key]
        for key in artifact_keys
    ):
        raise CapacityBlockEventError("campaign descriptor artifact inventory is invalid")
    artifact_sha_keys = {
        "source_snapshot_sha256",
        "non_vq_package_sha256",
        "teich_pack_sha256",
        "frozen_prompt_pack_sha256",
        "training_baseline_sha256",
    }
    if any(_HEX64.fullmatch(artifacts[key]) is None for key in artifact_sha_keys):
        raise CapacityBlockEventError("campaign descriptor artifact SHA-256 is invalid")
    _iso_utc(value.get("capacity_block_end"))
    return dict(value)


def capacity_block_client_token(run_id: str, *, generation: int) -> str:
    """Return a stable, bounded idempotency token for one launch generation."""

    if generation < 0:
        raise ValueError("generation must be non-negative")
    digest = hashlib.sha256(f"{run_id}\0{generation}".encode()).hexdigest()[:32]
    return f"keep-glm52-capacity-block-{digest}"


def _campaign_instances(ec2: Any, descriptor: Mapping[str, object]) -> list[dict[str, object]]:
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:campaign-run-id", "Values": [descriptor["run_id"]]},
        ]
    )
    instances: list[dict[str, object]] = []
    for reservation in response.get("Reservations", []):
        instances.extend(reservation.get("Instances", []))
    return instances


def _active_instances(instances: list[dict[str, object]]) -> list[dict[str, object]]:
    active_states = {"pending", "running", "stopping", "stopped"}
    return [
        item
        for item in instances
        if item.get("State", {}).get("Name") in active_states
    ]


def _validate_event_authority(
    event: Mapping[str, object], descriptor: Mapping[str, object]
) -> tuple[str, Mapping[str, object]]:
    if event.get("source") != "aws.ec2":
        raise CapacityBlockEventError("event source is not aws.ec2")
    if event.get("account") != descriptor["account_id"]:
        raise CapacityBlockEventError("event account does not match campaign")
    event_region = event.get("region")
    if event_region is not None and event_region != descriptor["region"]:
        raise CapacityBlockEventError("event region does not match campaign")
    detail_type = event.get("detail-type", event.get("detail_type"))
    if not isinstance(detail_type, str):
        raise CapacityBlockEventError("event detail type is missing")
    detail = event.get("detail")
    if not isinstance(detail, dict):
        raise CapacityBlockEventError("event detail must be an object")
    if detail_type in (
        "Capacity Block Reservation Delivered",
        "Capacity Block Reservation Expiration Warning",
    ):
        if detail.get("capacity-reservation-id") != descriptor["capacity_reservation_id"]:
            raise CapacityBlockEventError("event reservation does not match campaign")
        if _iso_utc(detail.get("end-date")) != _iso_utc(
            descriptor["capacity_block_end"]
        ):
            raise CapacityBlockEventError("event Capacity Block end does not match campaign")
    return detail_type, detail


def handle_capacity_block_event(
    event: Mapping[str, object],
    descriptor_value: Mapping[str, object],
    *,
    ec2: Any,
    ssm: Any,
) -> dict[str, object]:
    descriptor = validate_campaign_descriptor(descriptor_value)
    detail_type, detail = _validate_event_authority(event, descriptor)
    campaign_instances = _campaign_instances(ec2, descriptor)
    instances = _active_instances(campaign_instances)
    if detail_type == "Capacity Block Reservation Delivered":
        if instances:
            return {
                "action": "already-running",
                "instance_ids": [item["InstanceId"] for item in instances],
            }
        reservation = ec2.describe_capacity_reservations(
            CapacityReservationIds=[descriptor["capacity_reservation_id"]]
        ).get("CapacityReservations", [])
        if len(reservation) != 1:
            raise CapacityBlockEventError("capacity reservation lookup was not unique")
        actual = reservation[0]
        if (
            actual.get("CapacityReservationId") != descriptor["capacity_reservation_id"]
            or actual.get("InstanceType") != descriptor["instance_type"]
            or actual.get("AvailabilityZone") != descriptor["availability_zone"]
            or actual.get("State") != "active"
            or _iso_utc(actual.get("EndDate")) != _iso_utc(descriptor["capacity_block_end"])
        ):
            raise CapacityBlockEventError("capacity reservation attributes drifted")
        generation = sum(
            item.get("State", {}).get("Name") == "terminated"
            for item in campaign_instances
        )
        response = ec2.run_instances(
            MinCount=1,
            MaxCount=1,
            ClientToken=capacity_block_client_token(
                str(descriptor["run_id"]), generation=generation
            ),
            LaunchTemplate={
                "LaunchTemplateId": descriptor["launch_template_id"],
                "Version": descriptor["launch_template_version"],
            },
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "AssociatePublicIpAddress": True,
                    "SubnetId": descriptor["subnet_id"],
                    "Groups": [descriptor["security_group_id"]],
                }
            ],
            InstanceMarketOptions={"MarketType": "capacity-block"},
            CapacityReservationSpecification={
                "CapacityReservationTarget": {
                    "CapacityReservationId": descriptor["capacity_reservation_id"]
                }
            },
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "campaign-run-id", "Value": descriptor["run_id"]},
                        {
                            "Key": "capacity-reservation-id",
                            "Value": descriptor["capacity_reservation_id"],
                        },
                        {"Key": "project", "Value": "keep-glm52"},
                    ],
                }
            ],
        )
        launched = response.get("Instances", [])
        if len(launched) != 1:
            raise CapacityBlockEventError("RunInstances did not return exactly one instance")
        if generation:
            return {
                "action": "replacement-launched",
                "instance_id": launched[0]["InstanceId"],
                "replacement_generation": generation,
            }
        return {"action": "launched", "instance_id": launched[0]["InstanceId"]}

    if detail_type in (
        "Capacity Block Reservation Expiration Warning",
        "EC2 Capacity Reservation Instance Interruption Warning",
        "EC2 Spot Instance Interruption Warning",
    ):
        if detail_type == "EC2 Capacity Reservation Instance Interruption Warning":
            requested_id = detail.get("instance-id")
            instance_ids = [
                item["InstanceId"] for item in instances if item.get("InstanceId") == requested_id
            ]
            if not instance_ids:
                raise CapacityBlockEventError("interruption warning names a foreign instance")
        else:
            instance_ids = [item["InstanceId"] for item in instances]
        if not instance_ids:
            return {"action": "no-running-instance"}
        response = ssm.send_command(
            InstanceIds=instance_ids,
            DocumentName="AWS-RunShellScript",
            Comment=f"Graceful stop for {descriptor['run_id']}",
            Parameters={
                "commands": [
                    "sudo install -d -m 0755 /run/keep-glm52",
                    "sudo touch /run/keep-glm52/STOP",
                    "sudo systemctl kill --signal=TERM keep-glm52-campaign.service || true",
                ]
            },
            TimeoutSeconds=300,
        )
        return {
            "action": "graceful-stop-requested",
            "instance_ids": instance_ids,
            "command_id": response.get("Command", {}).get("CommandId"),
        }
    raise CapacityBlockEventError(f"unsupported EC2 event type {detail_type!r}")


__all__ = [
    "CapacityBlockEventError",
    "capacity_block_client_token",
    "handle_capacity_block_event",
    "validate_campaign_descriptor",
]
