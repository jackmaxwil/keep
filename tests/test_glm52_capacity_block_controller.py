"""Idempotent Capacity Block delivery/warning controller tests."""

from __future__ import annotations

import hashlib
import json

import pytest

from mlx_vq.quality.glm52_capacity_block_controller import (
    CapacityBlockEventError,
    capacity_block_client_token,
    handle_capacity_block_event,
)


def _descriptor() -> dict[str, object]:
    body = {
        "schema_version": 1,
        "run_id": "glm52-20260717",
        "account_id": "246813579024",
        "region": "us-west-2",
        "capacity_reservation_id": "cr-0123456789abcdef0",
        "capacity_block_end": "2026-07-19T11:30:00Z",
        "availability_zone": "us-west-2a",
        "subnet_id": "subnet-0123",
        "security_group_id": "sg-0123",
        "launch_template_id": "lt-0123",
        "launch_template_version": "7",
        "instance_type": "p5.48xlarge",
        "instance_count": 1,
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "repo_tar_key": "campaign/repo.tar.gz",
        "repo_tar_sha256": "1" * 64,
        "campaign_descriptor_key": "campaign/descriptor.json",
        "artifacts": {
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": "2" * 64,
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": "3" * 64,
            "teich_pack_key": "teich-pack/glm52-coding-agent-initial-v2-20260713.json",
            "teich_pack_sha256": "4" * 64,
            "frozen_prompt_pack_key": "quality/glm52-family-eval-prompts-20260709-v2.json",
            "frozen_prompt_pack_sha256": "5" * 64,
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": "6" * 64,
        },
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**body, "descriptor_body_sha256": digest}


class FakeEC2:
    def __init__(self):
        self.run_calls = []
        self.instances = []

    def describe_capacity_reservations(self, **_kwargs):
        return {
            "CapacityReservations": [
                {
                    "CapacityReservationId": "cr-0123456789abcdef0",
                    "InstanceType": "p5.48xlarge",
                    "AvailabilityZone": "us-west-2a",
                    "State": "active",
                    "EndDate": "2026-07-19T11:30:00Z",
                }
            ]
        }

    def describe_instances(self, **_kwargs):
        return {"Reservations": [{"Instances": list(self.instances)}] if self.instances else []}

    def run_instances(self, **kwargs):
        self.run_calls.append(kwargs)
        instance = {"InstanceId": "i-0123", "State": {"Name": "pending"}}
        self.instances.append(instance)
        return {"Instances": [instance]}


class FakeSSM:
    def __init__(self):
        self.calls = []

    def send_command(self, **kwargs):
        self.calls.append(kwargs)
        return {"Command": {"CommandId": "cmd-1"}}


def _delivery() -> dict[str, object]:
    return {
        "detail-type": "Capacity Block Reservation Delivered",
        "source": "aws.ec2",
        "account": "246813579024",
        "region": "us-west-2",
        "detail": {
            "capacity-reservation-id": "cr-0123456789abcdef0",
            "end-date": "2026-07-19T11:30:00Z",
        },
    }


def test_duplicate_delivery_events_launch_exactly_one_capacity_block_instance() -> None:
    ec2 = FakeEC2()
    ssm = FakeSSM()
    first = handle_capacity_block_event(_delivery(), _descriptor(), ec2=ec2, ssm=ssm)
    second = handle_capacity_block_event(_delivery(), _descriptor(), ec2=ec2, ssm=ssm)

    assert first["action"] == "launched"
    assert second["action"] == "already-running"
    assert len(ec2.run_calls) == 1
    call = ec2.run_calls[0]
    assert call["InstanceMarketOptions"] == {"MarketType": "capacity-block"}
    assert call["CapacityReservationSpecification"]["CapacityReservationTarget"] == {
        "CapacityReservationId": "cr-0123456789abcdef0"
    }
    assert call["ClientToken"] == capacity_block_client_token(
        "glm52-20260717", generation=0
    )
    assert call["MinCount"] == call["MaxCount"] == 1


def test_terminated_instance_can_be_replaced_once_inside_same_capacity_block() -> None:
    ec2 = FakeEC2()
    ec2.instances.append(
        {"InstanceId": "i-dead", "State": {"Name": "terminated"}}
    )
    ssm = FakeSSM()

    first = handle_capacity_block_event(_delivery(), _descriptor(), ec2=ec2, ssm=ssm)
    second = handle_capacity_block_event(_delivery(), _descriptor(), ec2=ec2, ssm=ssm)

    assert first == {
        "action": "replacement-launched",
        "instance_id": "i-0123",
        "replacement_generation": 1,
    }
    assert second["action"] == "already-running"
    assert len(ec2.run_calls) == 1
    assert ec2.run_calls[0]["ClientToken"] == capacity_block_client_token(
        "glm52-20260717", generation=1
    )
    assert ec2.run_calls[0]["ClientToken"] != capacity_block_client_token(
        "glm52-20260717", generation=0
    )


def test_capacity_block_client_tokens_are_deterministic_and_bounded() -> None:
    token = capacity_block_client_token("x" * 200, generation=17)
    assert token == capacity_block_client_token("x" * 200, generation=17)
    assert token != capacity_block_client_token("x" * 200, generation=18)
    assert len(token) <= 64


def test_foreign_delivery_is_rejected_before_run_instances() -> None:
    ec2 = FakeEC2()
    event = _delivery()
    event["detail"]["capacity-reservation-id"] = "cr-foreign"
    with pytest.raises(CapacityBlockEventError, match="reservation"):
        handle_capacity_block_event(event, _descriptor(), ec2=ec2, ssm=FakeSSM())
    assert ec2.run_calls == []


def test_expiration_warning_sends_idempotent_ssm_graceful_stop() -> None:
    ec2 = FakeEC2()
    ec2.instances.append({"InstanceId": "i-0123", "State": {"Name": "running"}})
    ssm = FakeSSM()
    event = {
        **_delivery(),
        "detail-type": "Capacity Block Reservation Expiration Warning",
    }
    result = handle_capacity_block_event(event, _descriptor(), ec2=ec2, ssm=ssm)

    assert result["action"] == "graceful-stop-requested"
    assert len(ssm.calls) == 1
    assert ssm.calls[0]["InstanceIds"] == ["i-0123"]
    commands = ssm.calls[0]["Parameters"]["commands"]
    assert any("STOP" in command for command in commands)
    assert all("start" not in command for command in commands)
