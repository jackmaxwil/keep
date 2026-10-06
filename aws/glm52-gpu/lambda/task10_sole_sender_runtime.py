"""AWS adapters for the retained Task 10 sole sender.

All clients are one-attempt and are created only inside the Lambda invocation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
import socket
from typing import Callable, Mapping

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.dynamodb import decode_item, encode_item
from glm52_enforcement.launch_custody import (
    ACCOUNT_ID,
    REGION,
    RUN_ID,
    AmbiguousRunInstances,
    AttemptPermit,
    CompletionResult,
    LaunchCustodyError,
    PositiveRunInstancesRejection,
    SameTokenCompleter,
)
from glm52_enforcement.task10_capacity_reconciliation import (
    publish_capacity_reconciliation,
)
from glm52_enforcement.task10_sole_sender import (
    Task10SoleSenderAuthority,
    Task10SoleSenderServices,
    task10_sole_sender_authority_from_mapping,
)


_TABLE = re.compile(r"^[A-Za-z0-9_.-]{3,255}$")
_BUCKET = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)[a-z0-9]"
    r"(?:[a-z0-9.-]{1,61}[a-z0-9])?$"
)
_INSTANCE = re.compile(r"^i-[0-9a-f]{17}$")
_VOLUME = re.compile(r"^vol-[0-9a-f]{17}$")
_CAPACITY_CODES = {
    "InsufficientInstanceCapacity",
    "InsufficientHostCapacity",
    "InsufficientReservedInstanceCapacity",
}


class Task10SoleSenderRuntimeError(LaunchCustodyError):
    """An authenticated AWS boundary or retained row drifted."""


def _metadata(response: object, label: str) -> str:
    if (
        type(response) is not dict
        or type(response.get("ResponseMetadata")) is not dict
        or response["ResponseMetadata"].get("HTTPStatusCode") != 200
        or type(response["ResponseMetadata"].get("RequestId")) is not str
        or not response["ResponseMetadata"]["RequestId"]
        or type(response["ResponseMetadata"].get("RetryAttempts")) is not int
        or response["ResponseMetadata"]["RetryAttempts"] != 0
    ):
        raise Task10SoleSenderRuntimeError(
            label + " response metadata is unauthenticated"
        )
    return response["ResponseMetadata"]["RequestId"]


def _key(sort_key: str) -> dict[str, dict[str, object]]:
    return encode_item({"PK": RUN_ID, "SK": sort_key})


class RetainedTask10AuthorityReader:
    """Strongly read the Task13-materialized Task8/9 authority closure."""

    def __init__(self, *, dynamodb: object, table_name: str) -> None:
        self._dynamodb = dynamodb
        self._table_name = table_name
        self.current: Task10SoleSenderAuthority | None = None

    @staticmethod
    def _sort_key(activation_id: str, generation_text: str) -> str:
        return (
            f"ACTIVATION#{activation_id}#TASK10_SOLE_SENDER_AUTHORITY#"
            f"{generation_text}"
        )

    def read(self, event: Mapping[str, object]) -> Task10SoleSenderAuthority:
        response = self._dynamodb.get_item(
            TableName=self._table_name,
            Key=_key(
                self._sort_key(
                    str(event["activation_id"]),
                    str(event["generation_text"]),
                )
            ),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        _metadata(response, "Task 10 authority GetItem")
        if type(response.get("Item")) is not dict:
            raise Task10SoleSenderRuntimeError(
                "Task 10 retained authority is absent"
            )
        item = decode_item(response["Item"])
        expected_key = self._sort_key(
            str(event["activation_id"]),
            str(event["generation_text"]),
        )
        if item.pop("PK", None) != RUN_ID or item.pop("SK", None) != expected_key:
            raise Task10SoleSenderRuntimeError(
                "Task 10 retained authority key drifted"
            )
        self.current = task10_sole_sender_authority_from_mapping(item)
        return self.current


class RetainedTask9CompletionStore:
    """Consume one attempt from retained Task9 custody for each AZ."""

    def __init__(
        self,
        *,
        dynamodb: object,
        table_name: str,
        authority_reader: RetainedTask10AuthorityReader,
    ) -> None:
        self._dynamodb = dynamodb
        self._table_name = table_name
        self._authority_reader = authority_reader
        self._contexts: dict[str, dict[str, object]] = {}

    @staticmethod
    def _worker_key(activation_id: str, ordinal: int) -> str:
        return (
            f"ACTIVATION#{activation_id}#WORKER_LAUNCH#{ordinal:08d}"
        )

    @staticmethod
    def _liability_key(activation_id: str, ordinal: int) -> str:
        return (
            f"ACTIVATION#{activation_id}#WORKER_LAUNCH_LIABILITY#"
            f"{ordinal:08d}"
        )

    def _read(self, sort_key: str, label: str) -> dict[str, object]:
        response = self._dynamodb.get_item(
            TableName=self._table_name,
            Key=_key(sort_key),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        _metadata(response, label)
        if type(response.get("Item")) is not dict:
            raise Task10SoleSenderRuntimeError(label + " is absent")
        item = decode_item(response["Item"])
        if item.pop("PK", None) != RUN_ID or item.pop("SK", None) != sort_key:
            raise Task10SoleSenderRuntimeError(label + " key drifted")
        return item

    def _attempt(self, ordinal: int):
        authority = self._authority_reader.current
        if authority is None:
            raise Task10SoleSenderRuntimeError(
                "Task 10 authority was not read before sending"
            )
        try:
            return next(
                attempt
                for attempt in authority.attempts
                if attempt.allocation_ordinal == ordinal
            )
        except StopIteration as error:
            raise Task10SoleSenderRuntimeError(
                "Task 10 allocation is not authorized"
            ) from error

    def read_completion_authority(
        self, activation_id: str, allocation_ordinal: int
    ) -> Mapping[str, object]:
        attempt = self._attempt(allocation_ordinal)
        worker = self._read(
            self._worker_key(activation_id, allocation_ordinal),
            "Task 9 worker launch",
        )
        liability = self._read(
            self._liability_key(activation_id, allocation_ordinal),
            "Task 9 launch liability",
        )
        required_worker = {
            "record_type",
            "activation_id",
            "allocation_ordinal",
            "state",
            "possibly_sent_at",
            "ec2_client_token",
            "launch_parameters_sha256",
        }
        required_liability = {
            "record_type",
            "activation_id",
            "allocation_ordinal",
            "state",
            "same_token_completion_attempts",
            "ec2_client_token",
            "current_owner",
            "gpu_liability_reserve_ledger_identity_sha256",
            "gpu_liability_reserve_release_identity_sha256",
        }
        if (
            not required_worker.issubset(worker)
            or not required_liability.issubset(liability)
            or worker["record_type"] != "glm52_production_worker_launch"
            or liability["record_type"]
            != "glm52_production_worker_launch_liability"
            or worker["activation_id"] != activation_id
            or liability["activation_id"] != activation_id
            or worker["allocation_ordinal"] != allocation_ordinal
            or liability["allocation_ordinal"] != allocation_ordinal
            or worker["ec2_client_token"] != attempt.ec2_client_token
            or liability["ec2_client_token"] != attempt.ec2_client_token
            or worker["launch_parameters_sha256"]
            != attempt.launch_parameters.canonical_identity_sha256
        ):
            raise Task10SoleSenderRuntimeError(
                "Task 8/9 completion authority drifted"
            )
        return {
            "activation_id": activation_id,
            "allocation_ordinal": allocation_ordinal,
            "state": liability["state"],
            "same_token_completion_attempts": liability[
                "same_token_completion_attempts"
            ],
            "possibly_sent_at": worker["possibly_sent_at"],
            "ec2_client_token": worker["ec2_client_token"],
            "launch_parameters": attempt.launch_parameters,
            "current_owner": liability["current_owner"],
            "reserve_held": (
                liability[
                    "gpu_liability_reserve_ledger_identity_sha256"
                ]
                is not None
                and liability[
                    "gpu_liability_reserve_release_identity_sha256"
                ]
                is None
            ),
        }

    def consume_attempt(
        self, activation_id: str, allocation_ordinal: int
    ) -> AttemptPermit:
        current = self.read_completion_authority(
            activation_id, allocation_ordinal
        )
        attempt = current["same_token_completion_attempts"] + 1
        response = self._dynamodb.update_item(
            TableName=self._table_name,
            Key=_key(self._liability_key(activation_id, allocation_ordinal)),
            UpdateExpression="SET same_token_completion_attempts = :next",
            ConditionExpression=(
                "#state = :watching AND current_owner = :true "
                "AND same_token_completion_attempts = :current "
                "AND attribute_exists("
                "gpu_liability_reserve_ledger_identity_sha256) "
                "AND attribute_not_exists("
                "gpu_liability_reserve_release_identity_sha256)"
            ),
            ExpressionAttributeNames={"#state": "state"},
            ExpressionAttributeValues=encode_item(
                {
                    ":next": attempt,
                    ":watching": "WATCHING",
                    ":true": True,
                    ":current": attempt - 1,
                }
            ),
            ReturnValues="ALL_NEW",
            ReturnConsumedCapacity="NONE",
        )
        request_id = _metadata(response, "Task 9 attempt UpdateItem")
        body = {
            "activation_id": activation_id,
            "allocation_ordinal": allocation_ordinal,
            "attempt": attempt,
            "request_id": request_id,
        }
        identity = canonical_sha256(body)
        self._contexts[identity] = body
        return AttemptPermit(
            may_send=True,
            attempt=attempt,
            action_identity_sha256=identity,
        )

    def _record(
        self,
        permit: AttemptPermit,
        classification: str,
        evidence: object,
    ) -> None:
        context = self._contexts.pop(
            permit.action_identity_sha256, None
        )
        if type(context) is not dict:
            raise Task10SoleSenderRuntimeError(
                "consumed Task 9 attempt is absent"
            )
        body = {
            "record_type": "glm52_task10_capacity_attempt_v1",
            **context,
            "classification": classification,
            "evidence": evidence,
            "action_identity_sha256": permit.action_identity_sha256,
        }
        item = {
            "PK": RUN_ID,
            "SK": (
                f"ACTIVATION#{context['activation_id']}#"
                "TASK10_CAPACITY_ATTEMPT#"
                f"{context['allocation_ordinal']:08d}#"
                f"{context['attempt']:08d}"
            ),
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
        try:
            response = self._dynamodb.put_item(
                TableName=self._table_name,
                Item=encode_item(item),
                ConditionExpression=(
                    "attribute_not_exists(PK) AND attribute_not_exists(SK)"
                ),
                ReturnConsumedCapacity="NONE",
            )
            _metadata(response, "Task 10 attempt PutItem")
        except Exception:
            readback = self._read(item["SK"], "Task 10 attempt readback")
            expected = dict(item)
            expected.pop("PK")
            expected.pop("SK")
            if readback != expected:
                raise

    def record_direct(
        self, permit: AttemptPermit, response: Mapping[str, object]
    ) -> None:
        self._record(permit, "DIRECT_SUCCESS", dict(response))

    def record_ambiguous(
        self, permit: AttemptPermit, evidence_identity_sha256: str
    ) -> None:
        self._record(permit, "AMBIGUOUS", evidence_identity_sha256)

    def record_rejection(
        self, permit: AttemptPermit, evidence_identity_sha256: str
    ) -> None:
        self._record(
            permit, "POSITIVE_REJECTION", evidence_identity_sha256
        )


class AuthenticatedOneAttemptEc2:
    """One exact EC2 call after an authenticated account read."""

    total_max_attempts = 1

    def __init__(self, *, sts: object, ec2: object) -> None:
        self._sts = sts
        self._ec2 = ec2

    def run_instances(self, **request: object) -> Mapping[str, object]:
        identity = self._sts.get_caller_identity()
        _metadata(identity, "STS")
        if identity.get("Account") != ACCOUNT_ID:
            raise Task10SoleSenderRuntimeError("STS account drifted")
        try:
            response = self._ec2.run_instances(**request)
        except Exception as error:
            raw = getattr(error, "response", None)
            metadata = raw.get("ResponseMetadata") if type(raw) is dict else None
            detail = raw.get("Error") if type(raw) is dict else None
            code = detail.get("Code") if type(detail) is dict else None
            authenticated_error = (
                type(metadata) is dict
                and type(metadata.get("RequestId")) is str
                and bool(metadata.get("RequestId"))
                and type(metadata.get("RetryAttempts")) is int
                and metadata["RetryAttempts"] == 0
                and type(metadata.get("HTTPStatusCode")) is int
            )
            if (
                authenticated_error
                and metadata["HTTPStatusCode"] in range(400, 500)
                and code in _CAPACITY_CODES
            ):
                raise PositiveRunInstancesRejection(code) from error
            if (
                authenticated_error
                and metadata["HTTPStatusCode"] in range(500, 600)
            ):
                raise AmbiguousRunInstances(
                    "authenticated RunInstances server error"
                ) from error
            if isinstance(
                error,
                (TimeoutError, ConnectionError, socket.timeout, OSError),
            ):
                raise AmbiguousRunInstances(type(error).__name__) from error
            raise Task10SoleSenderRuntimeError(
                "RunInstances failed outside capacity semantics"
            ) from error
        _metadata(response, "RunInstances")
        return response


class AuthenticatedDescribeReader:
    """Authenticate instance and root-volume readback against reviewed bytes."""

    def __init__(
        self,
        *,
        ec2: object,
        clock: Callable[[], datetime],
    ) -> None:
        self._ec2 = ec2
        self._clock = clock

    def describe(
        self,
        *,
        authority: Task10SoleSenderAuthority,
        attempt: object,
        completion: CompletionResult,
    ) -> Mapping[str, object]:
        del authority, completion
        parameters = attempt.launch_parameters
        expected = parameters.run_instances
        instance_tags = expected["TagSpecifications"][0]["Tags"]
        response = self._ec2.describe_instances(
            Filters=[
                {
                    "Name": "client-token",
                    "Values": [attempt.ec2_client_token],
                },
                {
                    "Name": "availability-zone",
                    "Values": [attempt.availability_zone],
                },
                *[
                    {
                        "Name": "tag:" + tag["Key"],
                        "Values": [tag["Value"]],
                    }
                    for tag in instance_tags
                ],
            ]
        )
        request_id = _metadata(response, "DescribeInstances")
        if response.get("NextToken") is not None:
            raise Task10SoleSenderRuntimeError(
                "DescribeInstances pagination is not closed"
            )
        reservations = response.get("Reservations")
        if type(reservations) is not list:
            raise Task10SoleSenderRuntimeError(
                "DescribeInstances reservations drifted"
            )
        instances = [
            instance
            for reservation in reservations
            if type(reservation) is dict
            and type(reservation.get("Instances")) is list
            for instance in reservation["Instances"]
        ]
        if len(instances) > 1 or any(type(item) is not dict for item in instances):
            raise Task10SoleSenderRuntimeError(
                "one token resolved to multiple instances"
            )
        volume_response: Mapping[str, object] = {}
        instance_ids: tuple[str, ...] = ()
        if instances:
            instance = instances[0]
            instance_id = instance.get("InstanceId")
            mappings = instance.get("BlockDeviceMappings")
            profile = instance.get("IamInstanceProfile")
            placement = instance.get("Placement")
            metadata = instance.get("MetadataOptions")
            tags = sorted(
                instance.get("Tags", []),
                key=lambda item: (item.get("Key"), item.get("Value")),
            )
            security_groups = instance.get("SecurityGroups")
            if (
                type(instance_id) is not str
                or _INSTANCE.fullmatch(instance_id) is None
                or instance.get("ClientToken") != attempt.ec2_client_token
                or instance.get("ImageId") != expected["ImageId"]
                or instance.get("InstanceType") != "p5.48xlarge"
                or instance.get("SubnetId") != expected["SubnetId"]
                or type(placement) is not dict
                or placement.get("AvailabilityZone")
                != attempt.availability_zone
                or type(profile) is not dict
                or profile.get("Arn")
                != (
                    "arn:aws:iam::246813579024:"
                    "instance-profile/keep-glm52-gpu-worker"
                )
                or type(metadata) is not dict
                or metadata.get("HttpTokens") != "required"
                or metadata.get("HttpEndpoint") != "enabled"
                or metadata.get("HttpPutResponseHopLimit") != 1
                or tags
                != sorted(
                    instance_tags,
                    key=lambda item: (item["Key"], item["Value"]),
                )
                or type(security_groups) is not list
                or [item.get("GroupId") for item in security_groups]
                != expected["SecurityGroupIds"]
                or instance.get("InstanceLifecycle") is not None
                or type(mappings) is not list
                or len(mappings) != 1
                or mappings[0].get("DeviceName") != "/dev/sda1"
                or type(mappings[0].get("Ebs")) is not dict
                or _VOLUME.fullmatch(
                    str(mappings[0]["Ebs"].get("VolumeId"))
                )
                is None
                or mappings[0]["Ebs"].get("DeleteOnTermination") is not True
            ):
                raise Task10SoleSenderRuntimeError(
                    "DescribeInstances launch readback drifted"
                )
            volume_id = mappings[0]["Ebs"]["VolumeId"]
            volume_response = self._ec2.describe_volumes(
                VolumeIds=[volume_id]
            )
            _metadata(volume_response, "DescribeVolumes")
            if volume_response.get("NextToken") is not None:
                raise Task10SoleSenderRuntimeError(
                    "DescribeVolumes pagination is not closed"
                )
            volumes = volume_response.get("Volumes")
            ebs = expected["BlockDeviceMappings"][0]["Ebs"]
            volume_tags = next(
                item["Tags"]
                for item in expected["TagSpecifications"]
                if item["ResourceType"] == "volume"
            )
            if (
                type(volumes) is not list
                or len(volumes) != 1
                or type(volumes[0]) is not dict
                or volumes[0].get("VolumeId") != volume_id
                or volumes[0].get("AvailabilityZone")
                != attempt.availability_zone
                or volumes[0].get("Encrypted") is not True
                or volumes[0].get("Size") != ebs["VolumeSize"]
                or volumes[0].get("VolumeType") != ebs["VolumeType"]
                or volumes[0].get("Iops") != ebs["Iops"]
                or volumes[0].get("Throughput") != ebs["Throughput"]
                or sorted(
                    volumes[0].get("Tags", []),
                    key=lambda item: (item.get("Key"), item.get("Value")),
                )
                != sorted(
                    volume_tags,
                    key=lambda item: (item["Key"], item["Value"]),
                )
                or type(volumes[0].get("Attachments")) is not list
                or len(volumes[0]["Attachments"]) != 1
                or volumes[0]["Attachments"][0].get("InstanceId")
                != instance_id
                or volumes[0]["Attachments"][0].get("Device")
                != "/dev/sda1"
                or volumes[0]["Attachments"][0].get(
                    "DeleteOnTermination"
                )
                is not True
            ):
                raise Task10SoleSenderRuntimeError(
                    "DescribeVolumes root readback drifted"
                )
            instance_ids = (instance_id,)
        observed = self._clock()
        if not isinstance(observed, datetime) or observed.tzinfo is None:
            raise Task10SoleSenderRuntimeError(
                "Task 10 describe clock is not aware"
            )
        return {
            "availability_zone": attempt.availability_zone,
            "instance_ids": instance_ids,
            "request_id": request_id,
            "response_identity_sha256": canonical_sha256(
                {
                    "instances": response,
                    "volumes": volume_response,
                }
            ),
            "observed_at": observed.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }


class TerminalPublisher:
    def __init__(self, *, s3: object) -> None:
        self._s3 = s3

    def publish(self, value: object) -> Mapping[str, object]:
        return publish_capacity_reconciliation(self._s3, value=value)


def build_runtime_services(
    *,
    ledger_table: object,
    campaign_bucket: object,
    region: object,
    clock: Callable[[], datetime],
) -> Task10SoleSenderServices:
    if (
        type(ledger_table) is not str
        or _TABLE.fullmatch(ledger_table) is None
        or type(campaign_bucket) is not str
        or _BUCKET.fullmatch(campaign_bucket) is None
        or region != REGION
        or not callable(clock)
    ):
        raise Task10SoleSenderRuntimeError(
            "Task 10 runtime environment drifted"
        )
    import boto3
    from botocore.config import Config

    config = Config(
        retries={"mode": "standard", "total_max_attempts": 1},
        connect_timeout=2,
        read_timeout=20,
    )
    dynamodb = boto3.client("dynamodb", region_name=REGION, config=config)
    ec2 = boto3.client("ec2", region_name=REGION, config=config)
    sts = boto3.client("sts", region_name=REGION, config=config)
    s3 = boto3.client("s3", region_name=REGION, config=config)
    reader = RetainedTask10AuthorityReader(
        dynamodb=dynamodb,
        table_name=ledger_table,
    )
    store = RetainedTask9CompletionStore(
        dynamodb=dynamodb,
        table_name=ledger_table,
        authority_reader=reader,
    )
    return Task10SoleSenderServices(
        authority_reader=reader,
        completion_sender=SameTokenCompleter(
            store=store,
            ec2=AuthenticatedOneAttemptEc2(sts=sts, ec2=ec2),
            clock=clock,
        ),
        describe_reader=AuthenticatedDescribeReader(ec2=ec2, clock=clock),
        terminal_publisher=TerminalPublisher(s3=s3),
        clock=clock,
    )


__all__ = [
    "AuthenticatedDescribeReader",
    "AuthenticatedOneAttemptEc2",
    "RetainedTask10AuthorityReader",
    "RetainedTask9CompletionStore",
    "Task10SoleSenderRuntimeError",
    "TerminalPublisher",
    "build_runtime_services",
]
