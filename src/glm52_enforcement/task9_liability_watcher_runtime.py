"""Executable retained Task 9 termination-only liability watcher.

The watcher accepts no instance ID from its caller.  It reconstructs the
current worker and liability from retained DynamoDB records, correlates the
worker through its deterministic ClientToken and complete tag set, consumes a
durable one-call termination action, and only then issues one exact
``TerminateInstances`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Mapping

from .canonical import canonical_json_bytes, canonical_sha256
from .dynamodb import decode_item, encode_item
from .records import (
    ledger_pk,
    ledger_sk,
    validate_record,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
FUNCTION_NAME = "keep-glm52-h1g-worker-launch-custody"
_VERSION_ARN = re.compile(
    rf"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
    rf"function:{FUNCTION_NAME}:([1-9][0-9]*)\Z"
)
_INSTANCE = re.compile(r"i-[0-9a-f]{17}\Z")
_UTC_TEXT = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)
_ACTIVE_STATES = frozenset(
    {"pending", "running", "stopping", "stopped"}
)
_LIABILITY_STATES = frozenset(
    {
        "WATCHING",
        "SAME_TOKEN_COMPLETION",
        "REJECTION_PROVED_AWAITING_TERMINAL_V2",
        "LATE_INSTANCE_DRAINING",
        "LIABILITY_INCIDENT",
    }
)


class Task9LiabilityWatcherRuntimeError(ValueError):
    """Live retained authority or one-attempt transport drifted."""


def _metadata(value: object, label: str) -> str:
    metadata = value.get("ResponseMetadata") if type(value) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        raise Task9LiabilityWatcherRuntimeError(
            label + " response is not exact zero-retry"
        )
    return str(metadata["RequestId"])


def _utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise Task9LiabilityWatcherRuntimeError(
            "liability watcher clock is not aware"
        )
    return value.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _validate_utc_text(value: object) -> None:
    if type(value) is not str or _UTC_TEXT.fullmatch(value) is None:
        raise Task9LiabilityWatcherRuntimeError(
            "Task 9 next-scan timestamp drifted"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Task9LiabilityWatcherRuntimeError(
            "Task 9 next-scan timestamp drifted"
        ) from exc


def _worker_tags(worker: Mapping[str, object]) -> dict[str, str]:
    tags = {
        "Project": "KEEP",
        "Campaign": "GLM-5.2",
        "RunId": RUN_ID,
        "Market": "on-demand",
        "campaign-identity-sha256": str(
            worker["campaign_identity_sha256"]
        ),
        "activation-id": str(worker["activation_id"]),
        "activation-ordinal-text": (
            f"{int(worker['activation_ordinal']):08d}"
        ),
        "generation-text": f"{int(worker['generation']):08d}",
        "allocation-ordinal-text": (
            f"{int(worker['allocation_ordinal']):08d}"
        ),
        "action-key": str(worker["sky_action_key"]),
        "sky-request-id": str(worker["sky_request_id"]),
        "sky-job-name": str(worker["sky_job_name"]),
        "sky-task-name": "glm52-production",
        "task-yaml-sha256": str(worker["task_yaml_sha256"]),
        "request-body-sha256": str(worker["request_body_sha256"]),
    }
    if canonical_sha256(tags) != worker["expected_worker_tags_sha256"]:
        raise Task9LiabilityWatcherRuntimeError(
            "retained worker tag authority drifted"
        )
    return tags


def _physical_item(
    raw: object,
    *,
    prefix: str,
    record_type: str,
) -> Mapping[str, object]:
    value = decode_item(raw)
    if (
        value.pop("PK", None) != ledger_pk(RUN_ID)
        or type(value.pop("SK", None)) is not str
    ):
        raise Task9LiabilityWatcherRuntimeError(
            "retained liability physical key drifted"
        )
    sort_key = decode_item(raw)["SK"]
    if not str(sort_key).startswith(prefix):
        raise Task9LiabilityWatcherRuntimeError(
            "retained liability family drifted"
        )
    try:
        return validate_record(record_type, value)
    except (TypeError, ValueError) as exc:
        raise Task9LiabilityWatcherRuntimeError(
            "retained liability record drifted"
        ) from exc


@dataclass(frozen=True)
class Task9LiabilityWatcherClients:
    sts: object
    dynamodb: object
    ec2: object


class AwsTask9LiabilityWatcher:
    """One-attempt AWS implementation of the retained watcher."""

    def __init__(
        self,
        *,
        clients: Task9LiabilityWatcherClients,
        table_name: str,
        remaining_time_in_millis: object,
        clock: object,
    ) -> None:
        if (
            type(clients) is not Task9LiabilityWatcherClients
            or type(table_name) is not str
            or not table_name
            or not callable(remaining_time_in_millis)
            or not callable(clock)
        ):
            raise TypeError("Task 9 liability watcher boundary is incomplete")
        self._clients = clients
        self._table = table_name
        self._remaining = remaining_time_in_millis
        self._clock = clock

    def _time_guard(self, label: str, minimum: int = 10_000) -> None:
        remaining = self._remaining()
        if (
            type(remaining) is not int
            or isinstance(remaining, bool)
            or remaining < minimum
        ):
            raise Task9LiabilityWatcherRuntimeError(
                label + " lacks bounded remaining time"
            )

    def _query(
        self,
        *,
        prefix: str,
        max_records: int,
    ) -> tuple[Mapping[str, object], ...]:
        records: list[Mapping[str, object]] = []
        start_key: object = None
        seen: set[bytes] = set()
        for _page in range(16):
            self._time_guard("Task 9 retained query")
            request: dict[str, object] = {
                "TableName": self._table,
                "KeyConditionExpression": (
                    "PK = :pk AND begins_with(SK, :sk)"
                ),
                "ExpressionAttributeValues": encode_item(
                    {
                        ":pk": ledger_pk(RUN_ID),
                        ":sk": prefix,
                    }
                ),
                "ConsistentRead": True,
                "ReturnConsumedCapacity": "NONE",
            }
            if start_key is not None:
                request["ExclusiveStartKey"] = start_key
            response = self._clients.dynamodb.query(**request)
            _metadata(response, "Task 9 retained query")
            items = response.get("Items")
            if type(items) is not list:
                raise Task9LiabilityWatcherRuntimeError(
                    "Task 9 retained query items drifted"
                )
            records.extend(items)
            if len(records) > max_records:
                raise Task9LiabilityWatcherRuntimeError(
                    "Task 9 retained query record bound exceeded"
                )
            next_key = response.get("LastEvaluatedKey")
            if next_key is None:
                return tuple(records)
            try:
                identity = canonical_json_bytes(next_key)
            except (TypeError, ValueError) as exc:
                raise Task9LiabilityWatcherRuntimeError(
                    "Task 9 retained query token drifted"
                ) from exc
            if identity in seen:
                raise Task9LiabilityWatcherRuntimeError(
                    "Task 9 retained query token repeated"
                )
            seen.add(identity)
            start_key = next_key
        raise Task9LiabilityWatcherRuntimeError(
            "Task 9 retained query page bound exceeded"
        )

    def _activation_index(self) -> Mapping[str, object]:
        self._time_guard("Task 9 activation-index read")
        response = self._clients.dynamodb.get_item(
            TableName=self._table,
            Key=encode_item(
                {
                    "PK": ledger_pk(RUN_ID),
                    "SK": ledger_sk(
                        "glm52_production_activation_index"
                    ),
                }
            ),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        _metadata(response, "Task 9 activation-index read")
        raw = response.get("Item")
        if type(raw) is not dict:
            raise Task9LiabilityWatcherRuntimeError(
                "Task 9 activation index is absent"
            )
        return _physical_item(
            raw,
            prefix="ACTIVATION_INDEX",
            record_type="glm52_production_activation_index",
        )

    def _authority(
        self,
        *,
        invoked_function_arn: str,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        index = self._activation_index()
        activation_id = index["current_activation_id"]
        worker_prefix = (
            f"ACTIVATION#{activation_id}#WORKER_LAUNCH#"
        )
        liability_prefix = (
            f"ACTIVATION#{activation_id}#"
            "WORKER_LAUNCH_LIABILITY#"
        )
        workers = [
            _physical_item(
                raw,
                prefix=worker_prefix,
                record_type="glm52_production_worker_launch",
            )
            for raw in self._query(
                prefix=worker_prefix,
                max_records=32,
            )
        ]
        liabilities = [
            _physical_item(
                raw,
                prefix=liability_prefix,
                record_type=(
                    "glm52_production_worker_launch_liability"
                ),
            )
            for raw in self._query(
                prefix=liability_prefix,
                max_records=32,
            )
        ]
        active_liabilities = [
            item
            for item in liabilities
            if item["state"] in _LIABILITY_STATES
            and item["owner_function_version_arn"]
            == invoked_function_arn
            and item["settlement_identity_sha256"] is None
        ]
        if len(active_liabilities) != 1:
            raise Task9LiabilityWatcherRuntimeError(
                "exact current Task 9 liability owner is absent"
            )
        liability = active_liabilities[0]
        matching = [
            item
            for item in workers
            if item["activation_id"] == liability["activation_id"]
            and item["allocation_ordinal"]
            == liability["allocation_ordinal"]
            and item["ec2_client_token"]
            == liability["ec2_client_token"]
            and item["expected_worker_tags_sha256"]
            == liability["expected_worker_tags_sha256"]
        ]
        if len(matching) != 1:
            raise Task9LiabilityWatcherRuntimeError(
                "Task 9 worker/liability bijection drifted"
            )
        return matching[0], liability

    def _discover(
        self,
        *,
        worker: Mapping[str, object],
    ) -> tuple[Mapping[str, object], ...]:
        tags = _worker_tags(worker)
        self._time_guard("Task 9 worker discovery")
        response = self._clients.ec2.describe_instances(
            Filters=[
                {
                    "Name": "client-token",
                    "Values": [worker["ec2_client_token"]],
                },
                *[
                    {"Name": "tag:" + key, "Values": [value]}
                    for key, value in sorted(tags.items())
                ],
            ],
            MaxResults=1000,
        )
        _metadata(response, "Task 9 worker discovery")
        if response.get("NextToken") is not None:
            raise Task9LiabilityWatcherRuntimeError(
                "Task 9 worker discovery pagination is not closed"
            )
        reservations = response.get("Reservations")
        if type(reservations) is not list:
            raise Task9LiabilityWatcherRuntimeError(
                "Task 9 worker discovery response drifted"
            )
        instances = [
            instance
            for reservation in reservations
            if type(reservation) is dict
            and type(reservation.get("Instances")) is list
            for instance in reservation["Instances"]
        ]
        discovered: list[Mapping[str, object]] = []
        for instance in instances:
            instance_id = (
                instance.get("InstanceId")
                if type(instance) is dict
                else None
            )
            state = (
                instance.get("State")
                if type(instance) is dict
                else None
            )
            observed_tags = (
                instance.get("Tags")
                if type(instance) is dict
                else None
            )
            if (
                type(instance_id) is not str
                or _INSTANCE.fullmatch(instance_id) is None
                or type(state) is not dict
                or state.get("Name")
                not in _ACTIVE_STATES | {"shutting-down", "terminated"}
                or type(observed_tags) is not list
                or any(type(item) is not dict for item in observed_tags)
                or {
                    item.get("Key"): item.get("Value")
                    for item in observed_tags
                }
                != tags
            ):
                raise Task9LiabilityWatcherRuntimeError(
                    "Task 9 discovered a foreign worker"
                )
            discovered.append(
                {
                    "instance_id": instance_id,
                    "state": state["Name"],
                    "tags_sha256": canonical_sha256(tags),
                }
            )
        discovered.sort(key=lambda item: str(item["instance_id"]))
        if len(
            {str(item["instance_id"]) for item in discovered}
        ) != len(discovered):
            raise Task9LiabilityWatcherRuntimeError(
                "Task 9 discovered duplicate worker identity"
            )
        return tuple(discovered)

    def _consume(
        self,
        *,
        worker: Mapping[str, object],
        liability: Mapping[str, object],
        instance: Mapping[str, object],
        invoked_function_arn: str,
        consumed_at: str,
    ) -> Mapping[str, object]:
        prefix = (
            f"ACTIVATION#{liability['activation_id']}#"
            "TASK9_SUPPORT_TERMINATION#"
            f"{int(liability['allocation_ordinal']):08d}#"
            f"{instance['instance_id']}#"
        )
        prior = self._query(prefix=prefix, max_records=6)
        attempt = len(prior) + 1
        if attempt > 6:
            raise Task9LiabilityWatcherRuntimeError(
                "Task 9 termination call window is exhausted"
            )
        authority = {
            "activation_id": liability["activation_id"],
            "allocation_ordinal": liability["allocation_ordinal"],
            "worker_launch_identity_sha256": (
                liability["worker_launch_identity_sha256"]
            ),
            "liability_identity_sha256": canonical_sha256(liability),
            "expected_worker_tags_sha256": (
                liability["expected_worker_tags_sha256"]
            ),
            "instance_id": instance["instance_id"],
            "invoked_function_version_arn": invoked_function_arn,
            "attempt": attempt,
            "consumed_at": consumed_at,
        }
        body = {
            "schema_version": 1,
            "record_type": "glm52_task9_support_termination_action_v1",
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            **authority,
            "state": "CONSUMED",
            "response_identity_sha256": None,
        }
        action = {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
        sort_key = prefix + f"{attempt:08d}"
        physical = {
            "PK": ledger_pk(RUN_ID),
            "SK": sort_key,
            **action,
        }
        self._time_guard("Task 9 termination action consumption")
        response = self._clients.dynamodb.transact_write_items(
            TransactItems=[
                {
                    "ConditionCheck": {
                        "TableName": self._table,
                        "Key": encode_item(
                            {
                                "PK": ledger_pk(RUN_ID),
                                "SK": (
                                    f"ACTIVATION#"
                                    f"{liability['activation_id']}#"
                                    "WORKER_LAUNCH_LIABILITY#"
                                    f"{int(liability['allocation_ordinal']):08d}"
                                ),
                            }
                        ),
                        "ConditionExpression": (
                            "#state = :state AND "
                            "owner_function_version_arn = :owner AND "
                            "attribute_type("
                            "settlement_identity_sha256, :null_type)"
                        ),
                        "ExpressionAttributeNames": {"#state": "state"},
                        "ExpressionAttributeValues": encode_item(
                            {
                                ":state": liability["state"],
                                ":owner": invoked_function_arn,
                                ":null_type": "NULL",
                            }
                        ),
                    }
                },
                {
                    "Put": {
                        "TableName": self._table,
                        "Item": encode_item(physical),
                        "ConditionExpression": (
                            "attribute_not_exists(PK) "
                            "AND attribute_not_exists(SK)"
                        ),
                    }
                },
            ],
            ClientRequestToken=action[
                "canonical_identity_sha256"
            ][:36],
            ReturnConsumedCapacity="NONE",
        )
        _metadata(response, "Task 9 termination action consumption")
        return {
            **action,
            "sort_key": sort_key,
        }

    def _complete(
        self,
        *,
        action: Mapping[str, object],
        response_identity_sha256: str,
    ) -> None:
        response = self._clients.dynamodb.update_item(
            TableName=self._table,
            Key=encode_item(
                {
                    "PK": ledger_pk(RUN_ID),
                    "SK": action["sort_key"],
                }
            ),
            UpdateExpression=(
                "SET #state = :completed, "
                "response_identity_sha256 = :response"
            ),
            ConditionExpression=(
                "#state = :consumed AND "
                "canonical_identity_sha256 = :identity"
            ),
            ExpressionAttributeNames={"#state": "state"},
            ExpressionAttributeValues=encode_item(
                {
                    ":completed": "COMPLETED",
                    ":response": response_identity_sha256,
                    ":consumed": "CONSUMED",
                    ":identity": action["canonical_identity_sha256"],
                }
            ),
            ReturnValues="NONE",
            ReturnConsumedCapacity="NONE",
        )
        _metadata(response, "Task 9 termination action completion")

    def watch(
        self,
        *,
        invoked_function_arn: str,
    ) -> Mapping[str, object]:
        if _VERSION_ARN.fullmatch(invoked_function_arn) is None:
            raise Task9LiabilityWatcherRuntimeError(
                "Task 9 invocation is not an exact numeric version"
            )
        self._time_guard("Task 9 account authentication", 20_000)
        identity = self._clients.sts.get_caller_identity()
        _metadata(identity, "Task 9 account authentication")
        if identity.get("Account") != ACCOUNT_ID:
            raise Task9LiabilityWatcherRuntimeError(
                "Task 9 AWS account drifted"
            )
        now = self._clock()
        now_text = _utc(now)
        worker, liability = self._authority(
            invoked_function_arn=invoked_function_arn,
        )
        instances = self._discover(worker=worker)
        for instance in instances:
            if instance["state"] in _ACTIVE_STATES:
                action = self._consume(
                    worker=worker,
                    liability=liability,
                    instance=instance,
                    invoked_function_arn=invoked_function_arn,
                    consumed_at=now_text,
                )
                self._time_guard("Task 9 exact termination", 5_000)
                response = self._clients.ec2.terminate_instances(
                    InstanceIds=[instance["instance_id"]]
                )
                request_id = _metadata(
                    response, "Task 9 exact termination"
                )
                terminating = response.get("TerminatingInstances")
                if (
                    type(terminating) is not list
                    or len(terminating) != 1
                    or type(terminating[0]) is not dict
                    or terminating[0].get("InstanceId")
                    != instance["instance_id"]
                ):
                    raise Task9LiabilityWatcherRuntimeError(
                        "Task 9 termination readback drifted"
                    )
                response_identity = canonical_sha256(
                    {
                        "request_id": request_id,
                        "response": response,
                        "action_identity_sha256": action[
                            "canonical_identity_sha256"
                        ],
                    }
                )
                self._complete(
                    action=action,
                    response_identity_sha256=response_identity,
                )
        return {
            "settled": False,
            "instance_ids": [
                str(instance["instance_id"]) for instance in instances
            ],
            "next_scan_at": _utc(
                now.astimezone(timezone.utc) + timedelta(seconds=60)
            ),
            "incident": len(instances) > 1,
        }


def run_task9_liability_watch(
    event: object,
    context: object,
    *,
    watcher: object,
) -> Mapping[str, object]:
    if type(event) is not dict or set(event) not in (
        {"mode"},
        {"mode", "event_accelerator"},
    ):
        raise Task9LiabilityWatcherRuntimeError(
            "Task 9 liability-watch event is not closed"
        )
    if event.get("mode") != "LIABILITY_WATCH" or event.get(
        "event_accelerator"
    ) not in {None, "owner-failover", "ec2-state-change", "cloudtrail"}:
        raise Task9LiabilityWatcherRuntimeError(
            "Task 9 liability-watch mode is not closed"
        )
    invoked = getattr(context, "invoked_function_arn", None)
    if type(invoked) is not str or _VERSION_ARN.fullmatch(invoked) is None:
        raise Task9LiabilityWatcherRuntimeError(
            "Task 9 Lambda context is not a numeric published version"
        )
    run = getattr(watcher, "watch", None)
    if not callable(run):
        raise Task9LiabilityWatcherRuntimeError(
            "Task 9 production watcher is absent"
        )
    result = run(invoked_function_arn=invoked)
    instance_ids = (
        result.get("instance_ids") if type(result) is dict else None
    )
    if (
        type(result) is not dict
        or set(result)
        != {"settled", "instance_ids", "next_scan_at", "incident"}
        or type(result.get("settled")) is not bool
        or type(result.get("incident")) is not bool
        or type(instance_ids) is not list
        or len(instance_ids) > 6
        or instance_ids != sorted(set(instance_ids))
        or any(
            type(item) is not str
            or _INSTANCE.fullmatch(item) is None
            for item in instance_ids
        )
    ):
        raise Task9LiabilityWatcherRuntimeError(
            "Task 9 production watcher result drifted"
        )
    _validate_utc_text(result["next_scan_at"])
    return result


def build_aws_task9_liability_watcher(
    *,
    table_name: object,
    region: object,
    remaining_time_in_millis: object,
    clock: object,
) -> AwsTask9LiabilityWatcher:
    if (
        type(table_name) is not str
        or not table_name
        or region != REGION
    ):
        raise Task9LiabilityWatcherRuntimeError(
            "Task 9 runtime environment drifted"
        )
    import boto3
    from botocore.config import Config

    config = Config(
        retries={"mode": "standard", "total_max_attempts": 1},
        connect_timeout=2,
        read_timeout=15,
    )
    return AwsTask9LiabilityWatcher(
        clients=Task9LiabilityWatcherClients(
            sts=boto3.client("sts", region_name=REGION, config=config),
            dynamodb=boto3.client(
                "dynamodb", region_name=REGION, config=config
            ),
            ec2=boto3.client("ec2", region_name=REGION, config=config),
        ),
        table_name=table_name,
        remaining_time_in_millis=remaining_time_in_millis,
        clock=clock,
    )


__all__ = [
    "AwsTask9LiabilityWatcher",
    "Task9LiabilityWatcherClients",
    "Task9LiabilityWatcherRuntimeError",
    "build_aws_task9_liability_watcher",
    "run_task9_liability_watch",
]
