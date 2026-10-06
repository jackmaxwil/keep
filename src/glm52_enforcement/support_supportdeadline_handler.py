"""Executable non-deleting 68/71/72-hour support-deadline Lambda."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Mapping, Optional

from .canonical import canonical_json_bytes, canonical_sha256


_INSTANCE_ID = re.compile(r"i-[0-9a-f]{17}\Z")
_SECURITY_GROUP_ID = re.compile(r"sg-[0-9a-f]{17}\Z")
_HOST_EGRESS_PERMISSION = {
    "IpProtocol": "-1",
    "IpRanges": [
        {
            "CidrIp": "0.0.0.0/0",
            "Description": "host-only bounded NAT egress",
        }
    ],
}


@dataclass(frozen=True)
class SupportDeadlineServices:
    cloudformation: object
    dynamodb: object
    ec2: object
    sns: object


def _production_services() -> SupportDeadlineServices:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    config = Config(
        connect_timeout=2,
        read_timeout=5,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session = boto3.session.Session(region_name="us-west-2")
    return SupportDeadlineServices(
        cloudformation=session.client("cloudformation", config=config),
        dynamodb=session.client("dynamodb", config=config),
        ec2=session.client("ec2", config=config),
        sns=session.client("sns", config=config),
    )


def _success(value: object, label: str) -> Mapping[str, object]:
    if (
        type(value) is not dict
        or type(value.get("ResponseMetadata")) is not dict
        or value["ResponseMetadata"].get("HTTPStatusCode") != 200
        or type(value["ResponseMetadata"].get("RequestId")) is not str
        or not value["ResponseMetadata"]["RequestId"]
    ):
        raise ValueError(label + " response is not authenticated")
    return value


def _event_action(
    event: object,
    *,
    activation_id: str,
) -> tuple[str, str]:
    if type(event) is not dict:
        raise ValueError("support deadline event is not one object")
    if set(event) == {"activation_id", "deadline"}:
        if (
            event["activation_id"] != activation_id
            or event["deadline"]
            not in {"WORK_STOP", "DELETE_REQUEST", "ABSENCE_EXPECTED"}
        ):
            raise ValueError("support deadline schedule event drifted")
        return str(event["deadline"]), str(event["deadline"])
    if (
        event.get("source") != "aws.cloudwatch"
        or event.get("detail-type") != "CloudWatch Alarm State Change"
        or type(event.get("detail")) is not dict
        or event["detail"].get("state", {}).get("value") != "ALARM"
        or type(event["detail"].get("alarmName")) is not str
    ):
        raise ValueError("support alarm event is not exact")
    alarm = event["detail"]["alarmName"]
    prefix = f"keep-glm52-h1g-{activation_id}-"
    if not alarm.startswith(prefix):
        raise ValueError("support alarm identity is foreign")
    if alarm.endswith(("draintalarm", "authorityalarm")):
        return "EGRESS_DRAIN", alarm
    if "drainalarm" in alarm or "authorityalarm" in alarm:
        return "EGRESS_DRAIN", alarm
    if "warningalarm" in alarm:
        return "EGRESS_WARNING", alarm
    raise ValueError("support alarm action is unknown")


def _stop_and_drain(
    services: SupportDeadlineServices,
    *,
    instance_id: str,
    security_group_id: str,
) -> None:
    stopped = _success(
        services.ec2.stop_instances(InstanceIds=[instance_id]),
        "StopInstances",
    )
    transitions = stopped.get("StoppingInstances")
    if (
        type(transitions) is not list
        or len(transitions) != 1
        or transitions[0].get("InstanceId") != instance_id
    ):
        raise ValueError("support host stop response drifted")
    revoked = _success(
        services.ec2.revoke_security_group_egress(
            GroupId=security_group_id,
            IpPermissions=[_HOST_EGRESS_PERMISSION],
        ),
        "RevokeSecurityGroupEgress",
    )
    if revoked.get("Return") is not True:
        raise ValueError("support host egress revoke failed")


def main(
    event: object,
    context: object,
    *,
    services: Optional[SupportDeadlineServices] = None,
    continuation_services: Optional[object] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> Mapping[str, object]:
    """Enforce stop/drain/page outcomes without owning DeleteStack."""

    if (
        type(event) is dict
        and event.get("record_type")
        == "glm52_task12_support_continuation_invocation_v1"
    ):
        from .task12_support_continuation_runtime import (
            main as support_continuation_main,
        )

        return support_continuation_main(
            event,
            context,
            services=continuation_services,
        )
    del context
    env = dict(os.environ if environment is None else environment)
    activation_id = env.get("GLM52_ACTIVATION_ID", "")
    instance_id = env.get("GLM52_SUPPORT_HOST_INSTANCE_ID", "")
    security_group_id = env.get(
        "GLM52_SUPPORT_HOST_SECURITY_GROUP_ID",
        "",
    )
    if (
        not activation_id
        or env.get("GLM52_RUN_ID") != "glm52-sky-20260724"
        or env.get("GLM52_DIRECT_CHILD_DELETE_ACTIONS") != "FORBIDDEN"
        or env.get("GLM52_DELETE_STACK_OWNER")
        != "RETAINED_LIFECYCLE_ONLY"
        or _INSTANCE_ID.fullmatch(instance_id) is None
        or _SECURITY_GROUP_ID.fullmatch(security_group_id) is None
    ):
        raise RuntimeError("support deadline environment drifted")
    action, source_identity = _event_action(
        event,
        activation_id=activation_id,
    )
    runtime = _production_services() if services is None else services
    if type(runtime) is not SupportDeadlineServices:
        raise TypeError("support deadline services are not exact")
    stack = _success(
        runtime.cloudformation.describe_stacks(
            StackName=env.get("GLM52_SUPPORT_STACK_ID", "")
        ),
        "DescribeStacks",
    )
    stacks = stack.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1:
        raise ValueError("support stack readback is not unique")
    drained = action in {"WORK_STOP", "EGRESS_DRAIN", "ABSENCE_EXPECTED"}
    if drained:
        _stop_and_drain(
            runtime,
            instance_id=instance_id,
            security_group_id=security_group_id,
        )
    record = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_deadline_result_v1",
        "run_id": "glm52-sky-20260724",
        "activation_id": activation_id,
        "action": action,
        "source_identity": source_identity,
        "stack_status": stacks[0].get("StackStatus"),
        "host_stopped": drained,
        "egress_disabled": drained,
        "delete_stack_requested": action == "DELETE_REQUEST",
        "delete_stack_owner": "RETAINED_LIFECYCLE_ONLY",
    }
    identity = canonical_sha256(record)
    item = {
        "PK": {"S": "SUPPORT_DEADLINE#" + activation_id},
        "SK": {"S": action + "#" + identity},
        "record": {
            "S": canonical_json_bytes(
                {**record, "canonical_identity_sha256": identity}
            ).decode("ascii")
        },
    }
    _success(
        runtime.dynamodb.put_item(
            TableName=env.get("GLM52_LEDGER_TABLE_NAME", ""),
            Item=item,
            ConditionExpression="attribute_not_exists(PK)",
        ),
        "PutItem",
    )
    if action in {
        "DELETE_REQUEST",
        "ABSENCE_EXPECTED",
        "EGRESS_DRAIN",
        "EGRESS_WARNING",
    }:
        _success(
            runtime.sns.publish(
                TopicArn=env.get("GLM52_OPERATOR_TOPIC_ARN", ""),
                Subject="GLM52 H1g support deadline",
                Message=json.dumps(
                    {**record, "canonical_identity_sha256": identity},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
            "Publish",
        )
    return {**record, "canonical_identity_sha256": identity}


__all__ = ["SupportDeadlineServices", "main"]
