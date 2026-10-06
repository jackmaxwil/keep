from __future__ import annotations

from datetime import UTC, datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement.dynamodb import decode_item, encode_item
from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.task9_liability_watcher_runtime import (
    AwsTask9LiabilityWatcher,
    Task9LiabilityWatcherClients,
    Task9LiabilityWatcherRuntimeError,
    build_aws_task9_liability_watcher,
    run_task9_liability_watch,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION_ARN = (
    "arn:aws:lambda:us-west-2:246813579024:"
    "function:keep-glm52-h1g-worker-launch-custody:7"
)
INSTANCE_ID = "i-0123456789abcdef0"
SECOND_INSTANCE_ID = "i-0123456789abcdef1"


def _metadata(request_id: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": request_id,
        "RetryAttempts": 0,
    }


class _Sts:
    def __init__(self, calls: list[object]) -> None:
        self._calls = calls

    def get_caller_identity(self) -> dict[str, object]:
        self._calls.append("STS")
        return {
            "Account": "246813579024",
            "Arn": "arn:aws:iam::246813579024:role/task9",
            "UserId": "task9",
            "ResponseMetadata": _metadata("sts-1"),
        }


class _Dynamo:
    def __init__(self, calls: list[object]) -> None:
        self._calls = calls

    def update_item(self, **request: object) -> dict[str, object]:
        self._calls.append(("COMPLETE", request))
        return {"ResponseMetadata": _metadata("ddb-update-1")}


class _Ec2:
    def __init__(self, calls: list[object]) -> None:
        self._calls = calls

    def terminate_instances(
        self, **request: object
    ) -> dict[str, object]:
        self._calls.append(("TERMINATE", request))
        return {
            "TerminatingInstances": [
                {"InstanceId": request["InstanceIds"][0]}
            ],
            "ResponseMetadata": _metadata("terminate-1"),
        }


def _watcher(
    calls: list[object],
    *,
    instance_state: str = "running",
) -> AwsTask9LiabilityWatcher:
    watcher = AwsTask9LiabilityWatcher(
        clients=Task9LiabilityWatcherClients(
            sts=_Sts(calls),
            dynamodb=_Dynamo(calls),
            ec2=_Ec2(calls),
        ),
        table_name="glm52-ledger",
        remaining_time_in_millis=lambda: 60_000,
        clock=lambda: datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
    )
    worker = {"identity": "worker"}
    liability = {"identity": "liability"}
    watcher._authority = lambda **kwargs: (  # type: ignore[method-assign]
        calls.append(("AUTHORITY", kwargs)) or (worker, liability)
    )
    watcher._discover = lambda **kwargs: (  # type: ignore[method-assign]
        calls.append(("DISCOVER", kwargs))
        or (
            {
                "instance_id": INSTANCE_ID,
                "state": instance_state,
                "tags_sha256": "1" * 64,
            },
        )
    )
    watcher._consume = lambda **kwargs: (  # type: ignore[method-assign]
        calls.append(("CONSUME", kwargs))
        or {
            "sort_key": "action-key",
            "canonical_identity_sha256": "2" * 64,
        }
    )
    return watcher


def test_numeric_published_wrapper_executes_exact_termination_route() -> None:
    calls: list[object] = []
    watcher = _watcher(calls)
    result = run_task9_liability_watch(
        {"mode": "LIABILITY_WATCH"},
        SimpleNamespace(invoked_function_arn=VERSION_ARN),
        watcher=watcher,
    )

    assert result == {
        "settled": False,
        "instance_ids": [INSTANCE_ID],
        "next_scan_at": "2026-07-29T12:01:00Z",
        "incident": False,
    }
    assert calls[0] == "STS"
    assert [item[0] for item in calls[1:] if isinstance(item, tuple)] == [
        "AUTHORITY",
        "DISCOVER",
        "CONSUME",
        "TERMINATE",
        "COMPLETE",
    ]
    assert next(
        item[1] for item in calls if isinstance(item, tuple)
        and item[0] == "TERMINATE"
    ) == {"InstanceIds": [INSTANCE_ID]}


def test_terminal_readback_never_consumes_or_terminates_again() -> None:
    calls: list[object] = []
    watcher = _watcher(calls, instance_state="terminated")
    result = watcher.watch(invoked_function_arn=VERSION_ARN)

    assert result["instance_ids"] == [INSTANCE_ID]
    assert "CONSUME" not in {
        item[0] for item in calls if isinstance(item, tuple)
    }
    assert "TERMINATE" not in {
        item[0] for item in calls if isinstance(item, tuple)
    }


def test_same_token_multiplicity_drains_every_correlated_active_instance() -> None:
    """Break caught: production watcher aborts instead of draining multiplicity."""

    calls: list[object] = []
    worker = {
        "campaign_identity_sha256": "1" * 64,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "allocation_ordinal": 1,
        "sky_action_key": "action-1",
        "sky_request_id": "request-1",
        "sky_job_name": "job-1",
        "task_yaml_sha256": "2" * 64,
        "request_body_sha256": "3" * 64,
        "ec2_client_token": "4" * 64,
    }
    tags = {
        "Project": "KEEP",
        "Campaign": "GLM-5.2",
        "RunId": "glm52-sky-20260724",
        "Market": "on-demand",
        "campaign-identity-sha256": "1" * 64,
        "activation-id": "activation-1",
        "activation-ordinal-text": "00000001",
        "generation-text": "00000001",
        "allocation-ordinal-text": "00000001",
        "action-key": "action-1",
        "sky-request-id": "request-1",
        "sky-job-name": "job-1",
        "sky-task-name": "glm52-production",
        "task-yaml-sha256": "2" * 64,
        "request-body-sha256": "3" * 64,
    }
    worker["expected_worker_tags_sha256"] = canonical_sha256(tags)

    class MultipleEc2(_Ec2):
        def describe_instances(
            self, **request: object
        ) -> dict[str, object]:
            calls.append(("DISCOVER", request))
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": instance_id,
                                "State": {"Name": "running"},
                                "Tags": [
                                    {"Key": key, "Value": value}
                                    for key, value in tags.items()
                                ],
                            }
                            for instance_id in (
                                SECOND_INSTANCE_ID,
                                INSTANCE_ID,
                            )
                        ]
                    }
                ],
                "ResponseMetadata": _metadata("describe-many-1"),
            }

    watcher = AwsTask9LiabilityWatcher(
        clients=Task9LiabilityWatcherClients(
            sts=_Sts(calls),
            dynamodb=_Dynamo(calls),
            ec2=MultipleEc2(calls),
        ),
        table_name="glm52-ledger",
        remaining_time_in_millis=lambda: 60_000,
        clock=lambda: datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
    )
    watcher._authority = lambda **kwargs: (  # type: ignore[method-assign]
        calls.append(("AUTHORITY", kwargs))
        or (worker, {"identity": "liability"})
    )

    def consume(**kwargs: object) -> dict[str, object]:
        instance_id = kwargs["instance"]["instance_id"]
        calls.append(("CONSUME", instance_id))
        return {
            "sort_key": "action-" + str(instance_id),
            "canonical_identity_sha256": "5" * 64,
        }

    watcher._consume = consume  # type: ignore[method-assign]
    result = watcher.watch(invoked_function_arn=VERSION_ARN)

    assert result == {
        "settled": False,
        "instance_ids": [INSTANCE_ID, SECOND_INSTANCE_ID],
        "next_scan_at": "2026-07-29T12:01:00Z",
        "incident": True,
    }
    assert [
        item for item in calls
        if isinstance(item, tuple) and item[0] == "CONSUME"
    ] == [
        ("CONSUME", INSTANCE_ID),
        ("CONSUME", SECOND_INSTANCE_ID),
    ]
    assert [
        item[1]["InstanceIds"] for item in calls
        if isinstance(item, tuple) and item[0] == "TERMINATE"
    ] == [[INSTANCE_ID], [SECOND_INSTANCE_ID]]
    assert len(
        [
            item for item in calls
            if isinstance(item, tuple) and item[0] == "COMPLETE"
        ]
    ) == 2


@pytest.mark.parametrize(
    "invoked_arn",
    (
        VERSION_ARN.rsplit(":", 1)[0],
        VERSION_ARN.rsplit(":", 1)[0] + ":$LATEST",
        VERSION_ARN.rsplit(":", 1)[0] + ":prod",
        VERSION_ARN.replace(
            "keep-glm52-h1g-worker-launch-custody",
            "foreign-function",
        ),
    ),
)
def test_unqualified_alias_latest_and_foreign_invocation_fail_closed(
    invoked_arn: str,
) -> None:
    calls: list[object] = []
    with pytest.raises(
        Task9LiabilityWatcherRuntimeError,
        match="numeric published version",
    ):
        run_task9_liability_watch(
            {"mode": "LIABILITY_WATCH"},
            SimpleNamespace(invoked_function_arn=invoked_arn),
            watcher=_watcher(calls),
        )
    assert calls == []


def test_event_cannot_supply_instance_or_owner_authority() -> None:
    calls: list[object] = []
    with pytest.raises(
        Task9LiabilityWatcherRuntimeError,
        match="event is not closed",
    ):
        run_task9_liability_watch(
            {
                "mode": "LIABILITY_WATCH",
                "instance_id": INSTANCE_ID,
            },
            SimpleNamespace(invoked_function_arn=VERSION_ARN),
            watcher=_watcher(calls),
        )
    assert calls == []


@pytest.mark.parametrize(
    "result",
    [
        {
            "settled": "false",
            "instance_ids": [],
            "next_scan_at": "2026-07-29T12:01:00Z",
            "incident": False,
        },
        {
            "settled": False,
            "instance_ids": [INSTANCE_ID, INSTANCE_ID],
            "next_scan_at": "2026-07-29T12:01:00Z",
            "incident": False,
        },
        {
            "settled": False,
            "instance_ids": [],
            "next_scan_at": "caller-deadline",
            "incident": False,
        },
    ],
)
def test_wrapper_rejects_malformed_production_readback(
    result: dict[str, object],
) -> None:
    class Watcher:
        def watch(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return result

    with pytest.raises(
        Task9LiabilityWatcherRuntimeError,
        match="result drifted|timestamp drifted",
    ):
        run_task9_liability_watch(
            {"mode": "LIABILITY_WATCH"},
            SimpleNamespace(invoked_function_arn=VERSION_ARN),
            watcher=Watcher(),
        )


def test_activation_index_is_an_exact_consistent_get() -> None:
    calls: list[object] = []
    index = {
        "schema_version": 1,
        "record_type": "glm52_production_activation_index",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": "1" * 64,
        "current_activation_id": "activation-1",
        "current_activation_ordinal": 1,
        "prior_activation_id": None,
        "prior_activation_terminal_v2_identity": None,
        "prior_h1g_drained_identity": None,
        "prior_spend_ledger_head_identity": None,
        "snapshot_cleanup_lineage_sha256": "2" * 64,
        "revision": 1,
        "updated_at": "2026-07-29T12:00:00Z",
    }

    class Dynamo:
        def get_item(self, **request: object) -> dict[str, object]:
            calls.append(request)
            return {
                "Item": encode_item(
                    {
                        "PK": "RUN#glm52-sky-20260724",
                        "SK": "ACTIVATION_INDEX",
                        **index,
                    }
                ),
                "ResponseMetadata": _metadata("get-1"),
            }

    watcher = AwsTask9LiabilityWatcher(
        clients=Task9LiabilityWatcherClients(
            sts=object(),
            dynamodb=Dynamo(),
            ec2=object(),
        ),
        table_name="glm52-ledger",
        remaining_time_in_millis=lambda: 60_000,
        clock=lambda: datetime.now(UTC),
    )
    assert watcher._activation_index() == index
    assert calls == [
        {
            "TableName": "glm52-ledger",
            "Key": encode_item(
                {
                    "PK": "RUN#glm52-sky-20260724",
                    "SK": "ACTIVATION_INDEX",
                }
            ),
            "ConsistentRead": True,
            "ReturnConsumedCapacity": "NONE",
        }
    ]


def test_authority_queries_only_current_activation_families() -> None:
    calls: list[tuple[str, int]] = []
    watcher = AwsTask9LiabilityWatcher(
        clients=Task9LiabilityWatcherClients(
            sts=object(),
            dynamodb=object(),
            ec2=object(),
        ),
        table_name="glm52-ledger",
        remaining_time_in_millis=lambda: 60_000,
        clock=lambda: datetime.now(UTC),
    )
    watcher._activation_index = lambda: {  # type: ignore[method-assign]
        "current_activation_id": "activation-1"
    }
    watcher._query = lambda **kwargs: (  # type: ignore[method-assign]
        calls.append((kwargs["prefix"], kwargs["max_records"])) or ()
    )
    with pytest.raises(
        Task9LiabilityWatcherRuntimeError,
        match="exact current",
    ):
        watcher._authority(invoked_function_arn=VERSION_ARN)
    assert calls == [
        ("ACTIVATION#activation-1#WORKER_LAUNCH#", 32),
        (
            "ACTIVATION#activation-1#"
            "WORKER_LAUNCH_LIABILITY#",
            32,
        ),
    ]


def test_handler_uses_real_context_and_keeps_only_test_injection_seam() -> None:
    path = (
        ROOT
        / "aws/glm52-gpu/lambda/task9_liability_watcher_handler.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task9_liability_watcher_handler_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[object] = []
    result = module.main(
        {"mode": "LIABILITY_WATCH"},
        SimpleNamespace(invoked_function_arn=VERSION_ARN),
        _watcher=_watcher(calls, instance_state="terminated"),
    )
    assert result["instance_ids"] == [INSTANCE_ID]
    assert calls[0] == "STS"


def test_dynamo_null_condition_uses_attribute_type_not_null_equality() -> None:
    calls: list[object] = []

    class Dynamo:
        def query(self, **request: object) -> dict[str, object]:
            calls.append(("QUERY", request))
            return {
                "Items": [],
                "ResponseMetadata": _metadata("query-1"),
            }

        def transact_write_items(
            self, **request: object
        ) -> dict[str, object]:
            calls.append(("TRANSACT", request))
            return {"ResponseMetadata": _metadata("transaction-1")}

    watcher = AwsTask9LiabilityWatcher(
        clients=Task9LiabilityWatcherClients(
            sts=object(),
            dynamodb=Dynamo(),
            ec2=object(),
        ),
        table_name="glm52-ledger",
        remaining_time_in_millis=lambda: 60_000,
        clock=lambda: datetime.now(UTC),
    )
    liability = {
        "activation_id": "activation-1",
        "allocation_ordinal": 1,
        "worker_launch_identity_sha256": "1" * 64,
        "expected_worker_tags_sha256": "2" * 64,
        "state": "WATCHING",
    }
    watcher._consume(
        worker={},
        liability=liability,
        instance={"instance_id": INSTANCE_ID},
        invoked_function_arn=VERSION_ARN,
        consumed_at="2026-07-29T12:00:00Z",
    )
    transaction = next(
        item[1] for item in calls
        if isinstance(item, tuple) and item[0] == "TRANSACT"
    )
    condition = transaction["TransactItems"][0]["ConditionCheck"]
    assert "attribute_type(settlement_identity_sha256, :null_type)" in (
        condition["ConditionExpression"]
    )
    assert decode_item(condition["ExpressionAttributeValues"])[
        ":null_type"
    ] == "NULL"


def test_aws_factory_uses_one_attempt_and_bounded_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boto3

    calls: list[tuple[str, dict[str, object]]] = []

    def client(
        service: str, **kwargs: object
    ) -> SimpleNamespace:
        calls.append((service, kwargs))
        return SimpleNamespace(service=service)

    monkeypatch.setattr(boto3, "client", client)
    watcher = build_aws_task9_liability_watcher(
        table_name="glm52-ledger",
        region="us-west-2",
        remaining_time_in_millis=lambda: 60_000,
        clock=lambda: datetime.now(UTC),
    )
    assert isinstance(watcher, AwsTask9LiabilityWatcher)
    assert [item[0] for item in calls] == ["sts", "dynamodb", "ec2"]
    for _service, kwargs in calls:
        config = kwargs["config"]
        assert kwargs["region_name"] == "us-west-2"
        assert config.retries["total_max_attempts"] == 1
        assert config.connect_timeout == 2
        assert config.read_timeout == 15
