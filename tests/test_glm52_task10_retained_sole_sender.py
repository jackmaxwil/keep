"""Task 10 retained sole-sender production behavior."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import io
import importlib.util
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.launch_custody import (
    CompletionResult,
    LaunchParameterAuthority,
    PositiveRunInstancesRejection,
    SameTokenCompleter,
    build_launch_parameters,
)
from glm52_enforcement.task10_sole_sender import (
    Task10SoleSenderError,
    Task10SoleSenderServices,
    build_task10_sole_sender_authority,
    run_task10_sole_sender,
    task10_sole_sender_authority_from_mapping,
)
from glm52_enforcement.task10_authority_materialization import (
    Task10AuthorityMaterializationError,
    materialize_task10_sole_sender_authority,
)
from glm52_enforcement.dynamodb import decode_item
from glm52_enforcement.task10_capacity_reconciliation import (
    build_capacity_reconciliation,
    publish_capacity_reconciliation,
)


SHA = "a" * 64
ACTIVATION = "activation-0001"
EXECUTION_ARN = (
    "arn:aws:states:us-west-2:246813579024:execution:"
    "keep-glm52-h1g-production:activation-0001-epoch-00000001"
)
ZONES = tuple("us-west-2" + letter for letter in "abcdef")


def _runtime_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/lambda/task10_sole_sender_runtime.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_task10_sole_sender_runtime_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch_parameters(attempt: int):
    authority = LaunchParameterAuthority(
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        campaign_identity_sha256="1" * 64,
        activation_id=ACTIVATION,
        activation_ordinal=1,
        generation=1,
        action_key=(
            f"ACTIVATION#{ACTIVATION}#ACTION#SKY_POST#{attempt:08d}"
        ),
        sky_request_id=f"request-{attempt}",
        sky_job_name="glm52-sky-20260724",
        sky_task_name="glm52-production",
        task_yaml_sha256="2" * 64,
        request_body_sha256="3" * 64,
        approved_ami_id="ami-00000000000000001",
        subnet_id=f"subnet-{attempt:017x}",
        security_group_id="sg-00000000000000001",
        instance_profile_name="keep-glm52-gpu-worker",
        source_identity_sha256="4" * 64,
    )
    return build_launch_parameters(authority, allocation_ordinal=attempt)


def _authority():
    return build_task10_sole_sender_authority(
        schema_version=1,
        record_type="glm52_task10_retained_sole_sender_authority_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        activation_id=ACTIVATION,
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        availability_zones=list(ZONES),
        attempts=[
            {
                "attempt": attempt,
                "availability_zone": zone,
                "subnet_availability_zone": zone,
                "allocation_ordinal": attempt,
                "launch_parameters": asdict(_launch_parameters(attempt)),
            }
            for attempt, zone in enumerate(ZONES, 1)
        ],
        same_token_identity_sha256="5" * 64,
        reserve_identity_sha256="6" * 64,
        spend_authority_identity_sha256="7" * 64,
        action_identity_sha256="8" * 64,
        task9_custody_identity_sha256="9" * 64,
    )


class _AuthorityReader:
    def __init__(self) -> None:
        self.calls = 0

    def read(self, event):
        assert event == {
            "activation_id": ACTIVATION,
            "generation": 1,
            "generation_text": "00000001",
            "execution_arn": EXECUTION_ARN,
        }
        self.calls += 1
        return _authority()


class _CompletionSender:
    def __init__(self, classifications: list[str]) -> None:
        self.classifications = list(classifications)
        self.calls: list[dict[str, object]] = []

    def complete(self, value):
        self.calls.append(dict(value))
        attempt = len(self.calls)
        classification = self.classifications.pop(0)
        instance_ids = (
            (f"i-{attempt:017x}",)
            if classification == "DIRECT_SUCCESS"
            else ()
        )
        return CompletionResult(
            classification=classification,
            attempt=1,
            instance_ids=instance_ids,
            evidence_identity_sha256=f"{attempt:x}" * 64,
        )


class _DescribeReader:
    def __init__(self, instances_by_attempt: dict[int, tuple[str, ...]]) -> None:
        self.instances_by_attempt = instances_by_attempt
        self.calls: list[dict[str, object]] = []

    def describe(self, *, authority, attempt, completion):
        self.calls.append(
            {
                "zone": attempt.availability_zone,
                "classification": completion.classification,
            }
        )
        return {
            "availability_zone": attempt.availability_zone,
            "instance_ids": self.instances_by_attempt.get(attempt.attempt, ()),
            "request_id": f"describe-{attempt.attempt}",
            "response_identity_sha256": f"{attempt.attempt:x}" * 64,
            "observed_at": "2026-07-29T12:00:00Z",
        }


class _Publisher:
    def __init__(self) -> None:
        self.values = []

    def publish(self, value):
        self.values.append(value)
        return {
            "bucket": "keep-glm52-models-246813579024-us-west-2",
            "key": (
                "campaigns/glm52-sky-20260724/submissions/production/"
                "generations/00000001/workflow/LAUNCH_OUTCOME.json"
            ),
            "version_id": "version-1",
            "file_sha256": "e" * 64,
            "body_sha256": value.canonical_identity_sha256,
        }


def _event() -> dict[str, object]:
    return {
        "activation_id": ACTIVATION,
        "generation": 1,
        "generation_text": "00000001",
        "execution_arn": EXECUTION_ARN,
    }


def test_ordered_sole_sender_stops_after_first_authenticated_worker() -> None:
    """Break caught: capacity rotation skips an AZ or sends after success."""

    authority = _AuthorityReader()
    sender = _CompletionSender(["POSITIVE_REJECTION", "DIRECT_SUCCESS"])
    describe = _DescribeReader({2: ("i-00000000000000002",)})
    publisher = _Publisher()

    result = run_task10_sole_sender(
        _event(),
        services=Task10SoleSenderServices(
            authority_reader=authority,
            completion_sender=sender,
            describe_reader=describe,
            terminal_publisher=publisher,
            clock=lambda: datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
        ),
    )

    assert all(set(call) == {"activation_id", "allocation_ordinal"} for call in sender.calls)
    assert [call["allocation_ordinal"] for call in sender.calls] == [1, 2]
    assert [call["zone"] for call in describe.calls] == [
        "us-west-2a",
        "us-west-2b",
    ]
    assert len(publisher.values) == 1
    value = publisher.values[0]
    assert value.classification == "WORKER_ALLOCATED"
    assert value.instance_id == "i-00000000000000002"
    assert value.availability_zone == "us-west-2b"
    assert value.capacity_outcomes == [
        {
            "attempt": 1,
            "availability_zone": "us-west-2a",
            "outcome": "CAPACITY_REJECTED",
        },
        {
            "attempt": 2,
            "availability_zone": "us-west-2b",
            "outcome": "WORKER_ALLOCATED",
        },
    ]
    assert result["classification"] == "WORKER_ALLOCATED"
    assert result["publication"]["version_id"] == "version-1"


def test_ambiguous_send_is_described_and_adopts_exact_instance() -> None:
    """Break caught: a lost RunInstances response causes another AZ send."""

    sender = _CompletionSender(["AMBIGUOUS"])
    describe = _DescribeReader({1: ("i-00000000000000001",)})
    publisher = _Publisher()

    result = run_task10_sole_sender(
        _event(),
        services=Task10SoleSenderServices(
            authority_reader=_AuthorityReader(),
            completion_sender=sender,
            describe_reader=describe,
            terminal_publisher=publisher,
            clock=lambda: datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
        ),
    )

    assert len(sender.calls) == 1
    assert result["classification"] == "WORKER_ALLOCATED"
    assert publisher.values[0].capacity_outcomes == [
        {
            "attempt": 1,
            "availability_zone": "us-west-2a",
            "outcome": "WORKER_ALLOCATED",
        }
    ]


def test_six_authenticated_capacity_rejections_publish_exhaustion_once() -> None:
    """Break caught: capacity exhaustion sends a seventh request or no terminal."""

    sender = _CompletionSender(["POSITIVE_REJECTION"] * 6)
    describe = _DescribeReader({})
    publisher = _Publisher()

    result = run_task10_sole_sender(
        _event(),
        services=Task10SoleSenderServices(
            authority_reader=_AuthorityReader(),
            completion_sender=sender,
            describe_reader=describe,
            terminal_publisher=publisher,
            clock=lambda: datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
        ),
    )

    assert len(sender.calls) == 6
    assert len(describe.calls) == 6
    assert len(publisher.values) == 1
    assert result["classification"] == "CAPACITY_EXHAUSTED"
    assert [row["availability_zone"] for row in result["capacity_outcomes"]] == list(
        ZONES
    )


def test_foreign_describe_readback_fails_before_terminal_publication() -> None:
    """Break caught: an unauthenticated/foreign EC2 readback allocates a worker."""

    class ForeignDescribe(_DescribeReader):
        def describe(self, *, authority, attempt, completion):
            value = super().describe(
                authority=authority,
                attempt=attempt,
                completion=completion,
            )
            value["availability_zone"] = "us-west-2f"
            return value

    publisher = _Publisher()
    with pytest.raises(Task10SoleSenderError, match="describe"):
        run_task10_sole_sender(
            _event(),
            services=Task10SoleSenderServices(
                authority_reader=_AuthorityReader(),
                completion_sender=_CompletionSender(["DIRECT_SUCCESS"]),
                describe_reader=ForeignDescribe(
                    {1: ("i-00000000000000001",)}
                ),
                terminal_publisher=publisher,
                clock=lambda: datetime(
                    2026, 7, 29, 12, 0, tzinfo=timezone.utc
                ),
            ),
        )
    assert publisher.values == []


def test_authority_self_hash_binds_exact_launch_parameters_and_six_zones() -> None:
    """Break caught: retained authority can alter the reviewed EC2 request."""

    authority = _authority()
    assert authority.canonical_identity_sha256 == canonical_sha256(
        {
            key: value
            for key, value in asdict(authority).items()
            if key != "canonical_identity_sha256"
        }
    )
    first = authority.attempts[0].launch_parameters.run_instances
    assert set(first) == {
        "ImageId",
        "InstanceType",
        "MinCount",
        "MaxCount",
        "SubnetId",
        "SecurityGroupIds",
        "IamInstanceProfile",
        "MetadataOptions",
        "BlockDeviceMappings",
        "TagSpecifications",
    }
    assert "InstanceMarketOptions" not in first
    assert "CapacityReservationSpecification" not in first
    assert len(first["TagSpecifications"][0]["Tags"]) == 15
    assert task10_sole_sender_authority_from_mapping(
        asdict(authority)
    ) == authority


def test_lambda_adapter_invokes_real_core_boundary_once() -> None:
    """Break caught: the published handler remains a terminal-writer placeholder."""

    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/lambda/task10_sole_sender_handler.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_task10_sole_sender_handler_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    publisher = _Publisher()
    services = Task10SoleSenderServices(
        authority_reader=_AuthorityReader(),
        completion_sender=_CompletionSender(["POSITIVE_REJECTION"] * 6),
        describe_reader=_DescribeReader({}),
        terminal_publisher=publisher,
        clock=lambda: datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )
    result = module.main(_event(), object(), _services=services)
    assert result["classification"] == "CAPACITY_EXHAUSTED"
    assert len(publisher.values) == 1


def test_runtime_ec2_boundary_authenticates_retry_count_and_full_readback() -> None:
    """Break caught: the SDK retries or a partial EC2 readback is trusted."""

    module = _runtime_module()
    calls = []

    class Sts:
        def get_caller_identity(self):
            calls.append("STS")
            return {
                "Account": "246813579024",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "sts-1",
                    "RetryAttempts": 0,
                },
            }

    authority = _authority()
    attempt = authority.attempts[0]
    expected = attempt.launch_parameters.run_instances
    tags = expected["TagSpecifications"][0]["Tags"]
    instance_id = "i-00000000000000001"
    volume_id = "vol-00000000000000001"

    class Ec2:
        foreign_volume = False

        def run_instances(self, **request):
            calls.append(("RUN", request))
            return {
                "Instances": [{"InstanceId": instance_id}],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "run-1",
                    "RetryAttempts": 0,
                },
            }

        def describe_instances(self, **request):
            calls.append(("DESCRIBE_INSTANCES", request))
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": instance_id,
                                "ClientToken": attempt.ec2_client_token,
                                "ImageId": expected["ImageId"],
                                "InstanceType": "p5.48xlarge",
                                "SubnetId": expected["SubnetId"],
                                "Placement": {
                                    "AvailabilityZone":
                                    attempt.availability_zone
                                },
                                "IamInstanceProfile": {
                                    "Arn": (
                                        "arn:aws:iam::246813579024:"
                                        "instance-profile/"
                                        "keep-glm52-gpu-worker"
                                    )
                                },
                                "MetadataOptions": {
                                    "HttpTokens": "required",
                                    "HttpEndpoint": "enabled",
                                    "HttpPutResponseHopLimit": 1,
                                },
                                "Tags": tags,
                                "SecurityGroups": [
                                    {"GroupId": expected["SecurityGroupIds"][0]}
                                ],
                                "BlockDeviceMappings": [
                                    {
                                        "DeviceName": "/dev/sda1",
                                        "Ebs": {
                                            "VolumeId": volume_id,
                                            "DeleteOnTermination": True,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                ],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "describe-1",
                    "RetryAttempts": 0,
                },
            }

        def describe_volumes(self, **request):
            calls.append(("DESCRIBE_VOLUMES", request))
            return {
                "Volumes": [
                    {
                        "VolumeId": volume_id,
                        "AvailabilityZone": (
                            "us-west-2f"
                            if self.foreign_volume
                            else attempt.availability_zone
                        ),
                        "Encrypted": True,
                        "Size": 300,
                        "VolumeType": "gp3",
                        "Iops": 3000,
                        "Throughput": 125,
                        "Tags": [] if self.foreign_volume else tags,
                        "Attachments": [
                            {
                                "InstanceId": instance_id,
                                "Device": "/dev/sda1",
                                "DeleteOnTermination": True,
                            }
                        ],
                    }
                ],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "volume-1",
                    "RetryAttempts": 0,
                },
            }

    ec2 = Ec2()
    direct = module.AuthenticatedOneAttemptEc2(sts=Sts(), ec2=ec2)
    response = direct.run_instances(ClientToken=attempt.ec2_client_token)
    assert response["Instances"][0]["InstanceId"] == instance_id
    readback = module.AuthenticatedDescribeReader(
        ec2=ec2,
        clock=lambda: datetime(
            2026, 7, 29, 12, 0, tzinfo=timezone.utc
        ),
    ).describe(
        authority=authority,
        attempt=attempt,
        completion=CompletionResult(
            "DIRECT_SUCCESS", 1, (instance_id,), "a" * 64
        ),
    )
    assert readback["instance_ids"] == (instance_id,)
    assert [item[0] for item in calls if isinstance(item, tuple)] == [
        "RUN",
        "DESCRIBE_INSTANCES",
        "DESCRIBE_VOLUMES",
    ]
    ec2.foreign_volume = True
    with pytest.raises(
        module.Task10SoleSenderRuntimeError,
        match="root readback drifted",
    ):
        module.AuthenticatedDescribeReader(
            ec2=ec2,
            clock=lambda: datetime(
                2026, 7, 29, 12, 0, tzinfo=timezone.utc
            ),
        ).describe(
            authority=authority,
            attempt=attempt,
            completion=CompletionResult(
                "DIRECT_SUCCESS", 1, (instance_id,), "a" * 64
            ),
        )

    class RetriedCapacityEc2:
        def run_instances(self, **request):
            error = RuntimeError("capacity")
            error.response = {
                "Error": {"Code": "InsufficientInstanceCapacity"},
                "ResponseMetadata": {
                    "HTTPStatusCode": 400,
                    "RequestId": "capacity-1",
                    "RetryAttempts": 1,
                },
            }
            raise error

    with pytest.raises(
        module.Task10SoleSenderRuntimeError,
        match="outside capacity semantics",
    ):
        module.AuthenticatedOneAttemptEc2(
            sts=Sts(), ec2=RetriedCapacityEc2()
        ).run_instances(ClientToken=attempt.ec2_client_token)

    class MissingRetrySts(Sts):
        def get_caller_identity(self):
            response = super().get_caller_identity()
            response["ResponseMetadata"].pop("RetryAttempts")
            return response

    with pytest.raises(
        module.Task10SoleSenderRuntimeError,
        match="unauthenticated",
    ):
        module.AuthenticatedOneAttemptEc2(
            sts=MissingRetrySts(), ec2=ec2
        ).run_instances(ClientToken=attempt.ec2_client_token)

    class AuthenticatedServerErrorEc2:
        def run_instances(self, **request):
            error = RuntimeError("lost after service failure")
            error.response = {
                "Error": {"Code": "InternalError"},
                "ResponseMetadata": {
                    "HTTPStatusCode": 503,
                    "RequestId": "server-error-1",
                    "RetryAttempts": 0,
                },
            }
            raise error

    with pytest.raises(module.AmbiguousRunInstances):
        module.AuthenticatedOneAttemptEc2(
            sts=Sts(), ec2=AuthenticatedServerErrorEc2()
        ).run_instances(ClientToken=attempt.ec2_client_token)


def test_real_runtime_store_journals_six_allocations_without_key_collision() -> None:
    """Break caught: allocation 2 overwrites allocation 1's attempt-1 journal."""

    module = _runtime_module()
    from glm52_enforcement.dynamodb import encode_item

    authority = _authority()

    class Dynamo:
        def __init__(self):
            self.items = {}
            self.lost_allocation = 3
            for attempt in authority.attempts:
                ordinal = attempt.allocation_ordinal
                self._seed(
                    (
                        f"ACTIVATION#{ACTIVATION}#"
                        f"WORKER_LAUNCH#{ordinal:08d}"
                    ),
                    {
                        "record_type": "glm52_production_worker_launch",
                        "activation_id": ACTIVATION,
                        "allocation_ordinal": ordinal,
                        "state": "POSSIBLY_SENT",
                        "possibly_sent_at": "2026-07-29T11:59:00Z",
                        "ec2_client_token": attempt.ec2_client_token,
                        "launch_parameters_sha256": (
                            attempt.launch_parameters
                            .canonical_identity_sha256
                        ),
                    },
                )
                self._seed(
                    (
                        f"ACTIVATION#{ACTIVATION}#"
                        f"WORKER_LAUNCH_LIABILITY#{ordinal:08d}"
                    ),
                    {
                        "record_type": (
                            "glm52_production_worker_launch_liability"
                        ),
                        "activation_id": ACTIVATION,
                        "allocation_ordinal": ordinal,
                        "state": "WATCHING",
                        "same_token_completion_attempts": 0,
                        "ec2_client_token": attempt.ec2_client_token,
                        "current_owner": True,
                        "gpu_liability_reserve_ledger_identity_sha256": (
                            "6" * 64
                        ),
                        "gpu_liability_reserve_release_identity_sha256": None,
                    },
                )

        def _seed(self, sort_key, body):
            self.items[(authority.run_id, sort_key)] = {
                "PK": authority.run_id,
                "SK": sort_key,
                **body,
            }

        @staticmethod
        def _metadata(label):
            return {
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": label,
                    "RetryAttempts": 0,
                }
            }

        def get_item(self, **request):
            key = decode_item(request["Key"])
            response = self._metadata("get-1")
            item = self.items.get((key["PK"], key["SK"]))
            if item is not None:
                response["Item"] = encode_item(item)
            return response

        def update_item(self, **request):
            key = decode_item(request["Key"])
            values = decode_item(request["ExpressionAttributeValues"])
            item = self.items[(key["PK"], key["SK"])]
            assert item["same_token_completion_attempts"] == values[":current"]
            item["same_token_completion_attempts"] = values[":next"]
            return {
                **self._metadata("update-1"),
                "Attributes": encode_item(item),
            }

        def put_item(self, **request):
            item = decode_item(request["Item"])
            key = (item["PK"], item["SK"])
            assert key not in self.items
            self.items[key] = item
            if (
                f"#{self.lost_allocation:08d}#00000001"
                in item["SK"]
            ):
                self.lost_allocation = 0
                raise TimeoutError("lost Task10 attempt PutItem response")
            return self._metadata("put-1")

    class CapacityOnlyEc2:
        total_max_attempts = 1

        def run_instances(self, **request):
            raise PositiveRunInstancesRejection(
                "InsufficientInstanceCapacity"
            )

    dynamo = Dynamo()
    reader = module.RetainedTask10AuthorityReader(
        dynamodb=dynamo,
        table_name="keep-glm52-h1g-ledger-v1",
    )
    reader.current = authority
    sender = SameTokenCompleter(
        store=module.RetainedTask9CompletionStore(
            dynamodb=dynamo,
            table_name="keep-glm52-h1g-ledger-v1",
            authority_reader=reader,
        ),
        ec2=CapacityOnlyEc2(),
        clock=lambda: datetime(
            2026, 7, 29, 12, 0, tzinfo=timezone.utc
        ),
    )
    results = [
        sender.complete(
            {
                "activation_id": ACTIVATION,
                "allocation_ordinal": attempt.allocation_ordinal,
            }
        )
        for attempt in authority.attempts
    ]
    assert [result.classification for result in results] == [
        "POSITIVE_REJECTION"
    ] * 6
    assert [result.attempt for result in results] == [1] * 6
    attempt_keys = sorted(
        sort_key
        for _partition_key, sort_key in dynamo.items
        if "TASK10_CAPACITY_ATTEMPT#" in sort_key
    )
    assert attempt_keys == [
        (
            f"ACTIVATION#{ACTIVATION}#TASK10_CAPACITY_ATTEMPT#"
            f"{ordinal:08d}#00000001"
        )
        for ordinal in range(1, 7)
    ]


class _DynamoAuthorityWriter:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, object]] = {}
        self.lose_next_response = False

    def put_item(self, **request):
        item = decode_item(request["Item"])
        key = (item["PK"], item["SK"])
        if key in self.items:
            raise RuntimeError("ConditionalCheckFailedException")
        self.items[key] = item
        if self.lose_next_response:
            self.lose_next_response = False
            raise TimeoutError("lost response")
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "put-authority",
                "RetryAttempts": 0,
            }
        }

    def get_item(self, **request):
        key_value = decode_item(request["Key"])
        item = self.items.get((key_value["PK"], key_value["SK"]))
        response = {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "get-authority",
                "RetryAttempts": 0,
            }
        }
        if item is not None:
            from glm52_enforcement.dynamodb import encode_item

            response["Item"] = encode_item(item)
        return response


def test_presend_authority_materializer_is_conditional_and_lost_response_safe() -> None:
    """Break caught: Task13 can start before durable Task8/9 authority exists."""

    client = _DynamoAuthorityWriter()
    client.lose_next_response = True
    first = materialize_task10_sole_sender_authority(
        _authority(),
        dynamodb=client,
        table_name="keep-glm52-h1g-ledger-v1",
    )
    second = materialize_task10_sole_sender_authority(
        _authority(),
        dynamodb=client,
        table_name="keep-glm52-h1g-ledger-v1",
    )
    assert first["outcome"] == "RECONCILED"
    assert second["outcome"] == "RECONCILED"
    assert len(client.items) == 1
    item = next(iter(client.items.values()))
    assert item["PK"] == "glm52-sky-20260724"
    assert item["SK"] == (
        "ACTIVATION#activation-0001#"
        "TASK10_SOLE_SENDER_AUTHORITY#00000001"
    )


def test_presend_authority_materializer_rejects_foreign_existing_row() -> None:
    """Break caught: a stale generation is adopted after conditional failure."""

    client = _DynamoAuthorityWriter()
    materialize_task10_sole_sender_authority(
        _authority(),
        dynamodb=client,
        table_name="keep-glm52-h1g-ledger-v1",
    )
    next(iter(client.items.values()))["action_identity_sha256"] = "f" * 64
    with pytest.raises(
        Task10AuthorityMaterializationError,
        match="foreign",
    ):
        materialize_task10_sole_sender_authority(
            _authority(),
            dynamodb=client,
            table_name="keep-glm52-h1g-ledger-v1",
        )


def test_presend_materializer_requires_explicit_zero_retry_metadata() -> None:
    """Break caught: absent SDK retry evidence is treated as zero retries."""

    class MissingRetryDynamo(_DynamoAuthorityWriter):
        def put_item(self, **request):
            response = super().put_item(**request)
            response["ResponseMetadata"].pop("RetryAttempts")
            return response

        def get_item(self, **request):
            response = super().get_item(**request)
            response["ResponseMetadata"].pop("RetryAttempts")
            return response

    with pytest.raises(
        Task10AuthorityMaterializationError,
        match="unauthenticated",
    ):
        materialize_task10_sole_sender_authority(
            _authority(),
            dynamodb=MissingRetryDynamo(),
            table_name="keep-glm52-h1g-ledger-v1",
        )


def _capacity_terminal():
    authority = _authority()
    return build_capacity_reconciliation(
        schema_version=1,
        record_type="glm52_task10_capacity_reconciliation_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        activation_id=ACTIVATION,
        generation=1,
        generation_text="00000001",
        classification="CAPACITY_EXHAUSTED",
        execution_arn=EXECUTION_ARN,
        capacity_outcomes=[
            {
                "attempt": index,
                "availability_zone": zone,
                "outcome": "CAPACITY_REJECTED",
            }
            for index, zone in enumerate(ZONES, 1)
        ],
        instance_id=None,
        availability_zone=None,
        same_token_identity_sha256=authority.same_token_identity_sha256,
        reserve_identity_sha256=authority.reserve_identity_sha256,
        spend_authority_identity_sha256=(
            authority.spend_authority_identity_sha256
        ),
        action_identity_sha256=authority.action_identity_sha256,
        task9_custody_identity_sha256=(
            authority.task9_custody_identity_sha256
        ),
    )


def test_terminal_writer_rejects_unauthenticated_s3_success_response() -> None:
    """Break caught: a nominal HTTP 200 without request identity becomes truth."""

    class UnauthenticatedS3:
        def put_object(self, **request):
            del request
            return {
                "VersionId": "version-1",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "put-without-retry-evidence",
                },
            }

        def head_object(self, **request):
            del request
            return {
                "VersionId": "version-1",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "head-without-retry-evidence",
                },
            }

        def get_object(self, **request):
            raise AssertionError(request)

    with pytest.raises(ValueError, match="status"):
        publish_capacity_reconciliation(
            UnauthenticatedS3(),
            value=_capacity_terminal(),
        )


def test_terminal_writer_reconciles_lost_response_and_exact_repeat() -> None:
    """Break caught: a lost terminal PutObject response publishes a new version."""

    class S3:
        def __init__(self) -> None:
            self.stored = None
            self.lose = True

        @staticmethod
        def metadata(request_id):
            return {
                "HTTPStatusCode": 200,
                "RequestId": request_id,
                "RetryAttempts": 0,
            }

        def put_object(self, **request):
            if self.stored is not None:
                raise RuntimeError("PreconditionFailed")
            self.stored = dict(request)
            if self.lose:
                self.lose = False
                raise TimeoutError("lost response")
            return {
                "VersionId": "version-1",
                "ResponseMetadata": self.metadata("put"),
            }

        def head_object(self, **request):
            assert self.stored is not None
            return {
                "VersionId": "version-1",
                "ResponseMetadata": self.metadata("head"),
            }

        def get_object(self, **request):
            assert self.stored is not None
            return {
                "VersionId": "version-1",
                "ChecksumSHA256": self.stored["ChecksumSHA256"],
                "Metadata": self.stored["Metadata"],
                "Body": io.BytesIO(self.stored["Body"]),
                "ResponseMetadata": self.metadata("get"),
            }

    client = S3()
    first = publish_capacity_reconciliation(
        client,
        value=_capacity_terminal(),
    )
    second = publish_capacity_reconciliation(
        client,
        value=_capacity_terminal(),
    )
    assert first == second
    assert first["version_id"] == "version-1"
