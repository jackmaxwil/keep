"""Executable retained NAT observation path for the frozen lifecycle Lambda."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import re
from typing import Mapping, Optional


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_NAT_ID = re.compile(r"nat-[0-9a-f]{17}\Z")
_INSTANCE_ID = re.compile(r"i-[0-9a-f]{17}\Z")
_SECURITY_GROUP_ID = re.compile(r"sg-[0-9a-f]{17}\Z")
_ACTIVATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_METRICS = (
    "BytesInFromSource",
    "BytesOutToDestination",
    "BytesInFromDestination",
    "BytesOutToSource",
)
_QUERY_IDS = ("nat0", "nat1", "nat2", "nat3")
_MAX_WINDOWS = 72 * 60
_MAX_DATAPOINTS = len(_METRICS) * _MAX_WINDOWS


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{label} is not exact UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is not exact UTC") from exc
    if parsed.microsecond or parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} is not whole-second UTC")
    return parsed


def activation_cumulative_nat_request(
    *,
    activation_started_at: str,
    observed_at: str,
    nat_gateway_id: str,
) -> Mapping[str, object]:
    """Build one bounded four-counter, 60-second CloudWatch read."""

    start = _utc(activation_started_at, "activation start")
    end = _utc(observed_at, "NAT observation")
    if (
        _NAT_ID.fullmatch(nat_gateway_id) is None
        or end <= start
        or (end - start).total_seconds() > 72 * 60 * 60
    ):
        raise ValueError("NAT observation window is outside exact authority")
    return {
        "StartTime": start,
        "EndTime": end,
        "MetricDataQueries": [
            {
                "Id": query_id,
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/NATGateway",
                        "MetricName": metric_name,
                        "Dimensions": [
                            {
                                "Name": "NatGatewayId",
                                "Value": nat_gateway_id,
                            }
                        ],
                    },
                    "Period": 60,
                    "Stat": "Sum",
                    "Unit": "Bytes",
                },
                "ReturnData": True,
            }
            for query_id, metric_name in zip(_QUERY_IDS, _METRICS)
        ],
        "MaxDatapoints": _MAX_DATAPOINTS,
        "ScanBy": "TimestampAscending",
    }


def _completed_activation_window_end(
    *, activation_started_at: str, observed_at: str
) -> str:
    """Return the last completed activation-aligned minute, capped at 72h."""

    start = _utc(activation_started_at, "activation start")
    observed = _utc(observed_at, "NAT wall observation")
    elapsed_seconds = int((observed - start).total_seconds())
    completed_windows = min(elapsed_seconds // 60, _MAX_WINDOWS)
    if completed_windows <= 0:
        raise ValueError("no complete NAT observation window exists")
    return (
        start + timedelta(seconds=completed_windows * 60)
    ).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ActivationCumulativeNatObservation:
    activation_id: str
    nat_gateway_id: str
    observed_at: str
    window_count: int
    cumulative_bytes: int
    metric_state: str


def activation_cumulative_nat_observation(
    *,
    response: object,
    activation_id: str,
    activation_started_at: str,
    nat_gateway_id: str,
    observed_at: str,
) -> ActivationCumulativeNatObservation:
    """Validate a complete unpaginated response and sum all four counters."""

    if (
        type(response) is not dict
        or response.get("NextToken") is not None
        or _ACTIVATION_ID.fullmatch(activation_id) is None
        or _NAT_ID.fullmatch(nat_gateway_id) is None
    ):
        raise ValueError("NAT response is foreign or paginated")
    activation_start = _utc(activation_started_at, "activation start")
    observation_end = _utc(observed_at, "NAT observation")
    elapsed_seconds = int(
        (observation_end - activation_start).total_seconds()
    )
    if (
        elapsed_seconds <= 0
        or elapsed_seconds > 72 * 60 * 60
        or elapsed_seconds % 60
    ):
        raise ValueError("NAT response window is not exact whole minutes")
    expected_timestamps = tuple(
        activation_start + timedelta(seconds=60 * index)
        for index in range(elapsed_seconds // 60)
    )
    metadata = response.get("ResponseMetadata")
    results = response.get("MetricDataResults")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or type(results) is not list
        or len(results) != 4
    ):
        raise ValueError("NAT response is not authenticated success")
    by_id = {
        item.get("Id"): item for item in results if type(item) is dict
    }
    if set(by_id) != set(_QUERY_IDS):
        raise ValueError("NAT response does not contain four exact counters")
    timestamp_sets = []
    total = 0
    for query_id in _QUERY_IDS:
        item = by_id[query_id]
        timestamps = item.get("Timestamps")
        values = item.get("Values")
        messages = item.get("Messages")
        if (
            item.get("StatusCode") != "Complete"
            or messages not in (None, (), [])
            or type(timestamps) is not list
            or type(values) is not list
            or len(timestamps) != len(values)
            or len(timestamps) > _MAX_WINDOWS
        ):
            raise ValueError("NAT counter read is incomplete")
        normalized = []
        for timestamp, value in zip(timestamps, values):
            if isinstance(timestamp, datetime):
                instant = timestamp
            else:
                instant = _utc(timestamp, "NAT counter timestamp")
            if instant.tzinfo != timezone.utc or instant.microsecond:
                raise ValueError("NAT counter timestamp is not whole-second UTC")
            if (
                type(value) not in {int, float}
                or value < 0
                or int(value) != value
            ):
                raise ValueError("NAT counter value is not exact bytes")
            normalized.append(instant)
            total += int(value)
        if normalized != sorted(normalized) or len(set(normalized)) != len(
            normalized
        ):
            raise ValueError("NAT counter timestamps are not ordered")
        if any(
            int((right - left).total_seconds()) != 60
            for left, right in zip(normalized, normalized[1:])
        ):
            raise ValueError("NAT counter windows are not consecutive 60s")
        timestamp_sets.append(tuple(normalized))
    if len(set(timestamp_sets)) != 1:
        raise ValueError("four NAT counters do not cover identical windows")
    returned_timestamps = timestamp_sets[0]
    window_count = len(returned_timestamps)
    if returned_timestamps == expected_timestamps:
        metric_state = "OK"
    elif not returned_timestamps:
        metric_state = "MISSING"
    else:
        metric_state = "DELAYED"
    return ActivationCumulativeNatObservation(
        activation_id=activation_id,
        nat_gateway_id=nat_gateway_id,
        observed_at=observed_at,
        window_count=window_count,
        cumulative_bytes=total if metric_state == "OK" else 0,
        metric_state=metric_state,
    )


@dataclass(frozen=True)
class RetainedLifecycleServices:
    cloudwatch: object
    ec2: object


_HOST_EGRESS_PERMISSION = {
    "IpProtocol": "-1",
    "IpRanges": [
        {
            "CidrIp": "0.0.0.0/0",
            "Description": "host-only bounded NAT egress",
        }
    ],
}


def _method(service: object, name: str) -> object:
    method = getattr(service, name, None)
    if not callable(method):
        raise TypeError(f"retained lifecycle service lacks {name}")
    return method


def _fail_closed_support_work(
    *,
    ec2: object,
    instance_id: str,
    security_group_id: str,
) -> Mapping[str, bool]:
    """Stop the exact host and remove its one template-owned egress rule."""

    describe_instances = _method(ec2, "describe_instances")
    instance_response = describe_instances(InstanceIds=[instance_id])
    reservations = (
        instance_response.get("Reservations")
        if type(instance_response) is dict
        else None
    )
    instances = [
        instance
        for reservation in reservations
        if type(reservation) is dict
        for instance in reservation.get("Instances", [])
        if type(instance) is dict
    ] if type(reservations) is list else []
    if (
        len(instances) != 1
        or instances[0].get("InstanceId") != instance_id
        or type(instances[0].get("State")) is not dict
        or instances[0]["State"].get("Name")
        not in {"pending", "running", "stopping", "stopped"}
    ):
        raise ValueError("fail-closed support host readback is not exact")
    state = instances[0]["State"]["Name"]
    host_stopped = state in {"stopping", "stopped"}
    if not host_stopped:
        stop_response = _method(ec2, "stop_instances")(
            InstanceIds=[instance_id],
        )
        stopping = (
            stop_response.get("StoppingInstances")
            if type(stop_response) is dict
            else None
        )
        if (
            type(stopping) is not list
            or len(stopping) != 1
            or type(stopping[0]) is not dict
            or stopping[0].get("InstanceId") != instance_id
            or type(stopping[0].get("CurrentState")) is not dict
            or stopping[0]["CurrentState"].get("Name")
            not in {"stopping", "stopped"}
        ):
            raise ValueError("fail-closed support host stop was not accepted")
        host_stopped = True

    group_response = _method(ec2, "describe_security_groups")(
        GroupIds=[security_group_id],
    )
    groups = (
        group_response.get("SecurityGroups")
        if type(group_response) is dict
        else None
    )
    if (
        type(groups) is not list
        or len(groups) != 1
        or type(groups[0]) is not dict
        or groups[0].get("GroupId") != security_group_id
        or type(groups[0].get("IpPermissionsEgress")) is not list
    ):
        raise ValueError("fail-closed egress readback is not exact")
    permissions = groups[0]["IpPermissionsEgress"]
    if permissions == []:
        egress_disabled = True
    elif permissions == [_HOST_EGRESS_PERMISSION]:
        revoke_response = _method(ec2, "revoke_security_group_egress")(
            GroupId=security_group_id,
            IpPermissions=[_HOST_EGRESS_PERMISSION],
        )
        if (
            type(revoke_response) is not dict
            or revoke_response.get("Return") is not True
        ):
            raise ValueError("fail-closed support egress revoke failed")
        egress_disabled = True
    else:
        raise ValueError("support host has foreign egress permissions")
    return {
        "host_stopped": host_stopped,
        "egress_disabled": egress_disabled,
    }


def main(
    event: object,
    _context: object,
    *,
    services: Optional[RetainedLifecycleServices] = None,
    environment: Optional[Mapping[str, str]] = None,
    observed_at: Optional[str] = None,
) -> Mapping[str, object]:
    """Read and publish one activation-cumulative observation without state."""

    if type(event) is not dict:
        raise TypeError("retained lifecycle event is not exact")
    env = dict(os.environ if environment is None else environment)
    activation_id = env.get("GLM52_ACTIVATION_ID")
    nat_gateway_id = env.get("GLM52_NAT_GATEWAY_ID")
    activation_started_at = env.get("GLM52_ACTIVATION_STARTED_AT")
    grace_deadline = env.get("GLM52_BOOTSTRAP_GRACE_DEADLINE")
    host_instance_id = env.get("GLM52_SUPPORT_HOST_INSTANCE_ID")
    host_security_group_id = env.get(
        "GLM52_SUPPORT_HOST_SECURITY_GROUP_ID"
    )
    if (
        event
        != {
            "action": "ACCUMULATE_NAT_60S",
            "activation_id": activation_id,
            "activation_started_at": activation_started_at,
            "nat_gateway_id": nat_gateway_id,
        }
        or env.get("GLM52_RUN_ID") != RUN_ID
        or env.get("GLM52_NAT_ACCUMULATOR_MODE")
        != "ACTIVATION_CUMULATIVE_FOUR_COUNTERS_60S"
        or type(activation_id) is not str
        or type(nat_gateway_id) is not str
        or type(activation_started_at) is not str
        or type(grace_deadline) is not str
        or env.get("GLM52_MISSING_METRIC_AFTER_GRACE") != "DRAIN"
        or _INSTANCE_ID.fullmatch(str(host_instance_id)) is None
        or _SECURITY_GROUP_ID.fullmatch(str(host_security_group_id)) is None
    ):
        raise ValueError("retained NAT event/environment authority drifted")
    wall_now = (
        observed_at
        if observed_at is not None
        else datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
    )
    observation_cutoff = _completed_activation_window_end(
        activation_started_at=activation_started_at,
        observed_at=wall_now,
    )
    if type(services) is not RetainedLifecycleServices:
        raise TypeError("retained lifecycle services are not exact")
    runtime = services
    request = activation_cumulative_nat_request(
        activation_started_at=activation_started_at,
        observed_at=observation_cutoff,
        nat_gateway_id=nat_gateway_id,
    )
    response = _method(
        runtime.cloudwatch, "get_metric_data"
    )(**request)
    observation = activation_cumulative_nat_observation(
        response=response,
        activation_id=activation_id,
        activation_started_at=activation_started_at,
        nat_gateway_id=nat_gateway_id,
        observed_at=observation_cutoff,
    )
    drain_result = {
        "host_stopped": False,
        "egress_disabled": False,
    }
    if observation.metric_state == "OK":
        put_response = _method(
            runtime.cloudwatch, "put_metric_data"
        )(
            Namespace="GLM52/H1g",
            MetricData=[
                {
                    "MetricName": (
                        "ActivationCumulativeNatProcessedBytes"
                    ),
                    "Dimensions": [
                        {
                            "Name": "ActivationId",
                            "Value": activation_id,
                        },
                        {
                            "Name": "NatGatewayId",
                            "Value": nat_gateway_id,
                        },
                    ],
                    "Timestamp": _utc(
                        observation_cutoff, "NAT observation"
                    ),
                    "Unit": "Bytes",
                    "Value": observation.cumulative_bytes,
                }
            ],
        )
        if (
            type(put_response) is not dict
            or type(put_response.get("ResponseMetadata")) is not dict
            or put_response["ResponseMetadata"].get("HTTPStatusCode") != 200
        ):
            raise ValueError("cumulative NAT metric publication failed")
    elif _utc(wall_now, "NAT wall observation") >= _utc(
        grace_deadline, "bootstrap grace deadline"
    ):
        drain_result = dict(
            _fail_closed_support_work(
                ec2=runtime.ec2,
                instance_id=str(host_instance_id),
                security_group_id=str(host_security_group_id),
            )
        )
    return {
        "activation_id": observation.activation_id,
        "nat_gateway_id": observation.nat_gateway_id,
        "observed_at": observation.observed_at,
        "window_count": observation.window_count,
        "cumulative_bytes": observation.cumulative_bytes,
        "metric_state": observation.metric_state,
        "host_stopped": drain_result["host_stopped"],
        "egress_disabled": drain_result["egress_disabled"],
    }
