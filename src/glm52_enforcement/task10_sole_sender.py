"""Retained Task 10 sole sender over Task 8/9 launch custody."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
from typing import Callable, Mapping, Tuple

from .canonical import canonical_sha256
from .launch_custody import (
    CompletionResult,
    LaunchParameterAuthority,
    LaunchParameters,
    build_deterministic_client_token,
    validate_launch_parameters,
)
from .task10_capacity_reconciliation import build_capacity_reconciliation


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
ZONES = tuple("us-west-2" + letter for letter in "abcdef")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE = re.compile(r"^i-[0-9a-f]{17}$")
_EXECUTION = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:execution:"
    r"keep-glm52-h1g-production:[A-Za-z0-9_-]{1,80}$"
)


class Task10SoleSenderError(ValueError):
    """Task 10 retained authority or one-attempt reconciliation drifted."""


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise Task10SoleSenderError(label + " must be a lowercase SHA-256")
    return value


def _launch_parameters_from_mapping(value: object) -> LaunchParameters:
    if type(value) is not dict or set(value) != set(
        LaunchParameters.__dataclass_fields__
    ):
        raise Task10SoleSenderError("launch parameters are not closed")
    authority_value = value.get("authority")
    if type(authority_value) is not dict or set(authority_value) != set(
        LaunchParameterAuthority.__dataclass_fields__
    ):
        raise Task10SoleSenderError("launch parameter authority is not closed")
    try:
        authority = LaunchParameterAuthority(**authority_value)
        parameters = LaunchParameters(
            **{
                **value,
                "authority": authority,
            }
        )
    except TypeError as error:
        raise Task10SoleSenderError("launch parameters are malformed") from error
    try:
        return validate_launch_parameters(parameters, authority)
    except ValueError as error:
        raise Task10SoleSenderError("launch parameters drifted") from error


@dataclass(frozen=True)
class Task10CapacityAttemptAuthority:
    attempt: int
    availability_zone: str
    subnet_availability_zone: str
    allocation_ordinal: int
    ec2_client_token: str
    launch_parameters: LaunchParameters


@dataclass(frozen=True)
class Task10SoleSenderAuthority:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    activation_ordinal: int
    generation: int
    generation_text: str
    availability_zones: Tuple[str, ...]
    attempts: Tuple[Task10CapacityAttemptAuthority, ...]
    same_token_identity_sha256: str
    reserve_identity_sha256: str
    spend_authority_identity_sha256: str
    action_identity_sha256: str
    task9_custody_identity_sha256: str
    canonical_identity_sha256: str


def _authority_body(value: Task10SoleSenderAuthority) -> dict[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def validate_task10_sole_sender_authority(
    value: object,
) -> Task10SoleSenderAuthority:
    if not isinstance(value, Task10SoleSenderAuthority):
        raise Task10SoleSenderError("sole-sender authority must be typed")
    if (
        value.schema_version != 1
        or value.record_type
        != "glm52_task10_retained_sole_sender_authority_v1"
        or value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
        or type(value.activation_id) is not str
        or not value.activation_id
        or type(value.activation_ordinal) is not int
        or value.activation_ordinal < 1
        or type(value.generation) is not int
        or value.generation < 1
        or value.generation_text != f"{value.generation:08d}"
        or value.availability_zones != ZONES
        or len(value.attempts) != 6
    ):
        raise Task10SoleSenderError("sole-sender authority scope drifted")
    tokens: list[str] = []
    for expected, attempt in enumerate(value.attempts, 1):
        if (
            not isinstance(attempt, Task10CapacityAttemptAuthority)
            or attempt.attempt != expected
            or attempt.availability_zone != ZONES[expected - 1]
            or attempt.subnet_availability_zone
            != attempt.availability_zone
            or attempt.allocation_ordinal != expected
        ):
            raise Task10SoleSenderError("capacity attempts are not ordered")
        parameters = validate_launch_parameters(
            attempt.launch_parameters,
            attempt.launch_parameters.authority,
        )
        launch_authority = parameters.authority
        expected_token = build_deterministic_client_token(
            launch_authority,
            attempt.allocation_ordinal,
            parameters,
        )
        if (
            attempt.ec2_client_token != expected_token
            or launch_authority.activation_id != value.activation_id
            or launch_authority.activation_ordinal != value.activation_ordinal
            or launch_authority.generation != value.generation
            or parameters.allocation_ordinal != attempt.allocation_ordinal
        ):
            raise Task10SoleSenderError(
                "capacity attempt lost Task 8/9 launch authority"
            )
        tokens.append(expected_token)
    if len(set(tokens)) != 6:
        raise Task10SoleSenderError("each AZ requires one exact client token")
    for field in (
        "same_token_identity_sha256",
        "reserve_identity_sha256",
        "spend_authority_identity_sha256",
        "action_identity_sha256",
        "task9_custody_identity_sha256",
    ):
        _sha(getattr(value, field), field)
    if value.canonical_identity_sha256 != canonical_sha256(
        _authority_body(value)
    ):
        raise Task10SoleSenderError("sole-sender authority self-hash drifted")
    return value


def build_task10_sole_sender_authority(**values: object) -> Task10SoleSenderAuthority:
    raw_attempts = values.get("attempts")
    if type(raw_attempts) is not list or len(raw_attempts) != 6:
        raise Task10SoleSenderError("sole-sender attempts are not exact")
    attempts = []
    for raw in raw_attempts:
        if type(raw) is not dict or set(raw) != {
            "attempt",
            "availability_zone",
            "subnet_availability_zone",
            "allocation_ordinal",
            "launch_parameters",
        }:
            raise Task10SoleSenderError("capacity attempt fields drifted")
        parameters = _launch_parameters_from_mapping(raw["launch_parameters"])
        attempts.append(
            Task10CapacityAttemptAuthority(
                attempt=raw["attempt"],
                availability_zone=raw["availability_zone"],
                subnet_availability_zone=raw[
                    "subnet_availability_zone"
                ],
                allocation_ordinal=raw["allocation_ordinal"],
                ec2_client_token=build_deterministic_client_token(
                    parameters.authority,
                    raw["allocation_ordinal"],
                    parameters,
                ),
                launch_parameters=parameters,
            )
        )
    authority = Task10SoleSenderAuthority(
        **{
            **values,
            "availability_zones": tuple(values.get("availability_zones", ())),
            "attempts": tuple(attempts),
            "canonical_identity_sha256": "",
        }
    )
    body = _authority_body(authority)
    return validate_task10_sole_sender_authority(
        Task10SoleSenderAuthority(
            **{
                **body,
                "availability_zones": authority.availability_zones,
                "attempts": authority.attempts,
                "canonical_identity_sha256": canonical_sha256(body),
            }
        )
    )


def task10_sole_sender_authority_from_mapping(
    value: object,
) -> Task10SoleSenderAuthority:
    if type(value) is not dict or set(value) != set(
        Task10SoleSenderAuthority.__dataclass_fields__
    ):
        raise Task10SoleSenderError("sole-sender authority fields drifted")
    raw_attempts = value.get("attempts")
    if type(raw_attempts) not in {list, tuple}:
        raise Task10SoleSenderError("sole-sender attempts are not exact")
    builder_attempts = []
    supplied_tokens = []
    for raw in raw_attempts:
        if type(raw) is not dict or set(raw) != {
            "attempt",
            "availability_zone",
            "subnet_availability_zone",
            "allocation_ordinal",
            "ec2_client_token",
            "launch_parameters",
        }:
            raise Task10SoleSenderError("capacity attempt fields drifted")
        supplied_tokens.append(raw["ec2_client_token"])
        builder_attempts.append(
            {
                key: raw[key]
                for key in (
                    "attempt",
                    "availability_zone",
                    "subnet_availability_zone",
                    "allocation_ordinal",
                    "launch_parameters",
                )
            }
        )
    body = dict(value)
    supplied_identity = body.pop("canonical_identity_sha256")
    body["availability_zones"] = list(body["availability_zones"])
    body["attempts"] = builder_attempts
    built = build_task10_sole_sender_authority(**body)
    if (
        tuple(supplied_tokens)
        != tuple(attempt.ec2_client_token for attempt in built.attempts)
        or supplied_identity != built.canonical_identity_sha256
    ):
        raise Task10SoleSenderError("sole-sender authority identity drifted")
    return built


@dataclass(frozen=True)
class Task10SoleSenderServices:
    authority_reader: object
    completion_sender: object
    describe_reader: object
    terminal_publisher: object
    clock: Callable[[], datetime]


def _event(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "activation_id",
        "generation",
        "generation_text",
        "execution_arn",
    }:
        raise Task10SoleSenderError("sole-sender event is not closed")
    if (
        type(value["activation_id"]) is not str
        or not value["activation_id"]
        or type(value["generation"]) is not int
        or value["generation"] < 1
        or value["generation_text"] != f"{value['generation']:08d}"
        or type(value["execution_arn"]) is not str
        or _EXECUTION.fullmatch(value["execution_arn"]) is None
    ):
        raise Task10SoleSenderError("sole-sender event coordinates drifted")
    return dict(value)


def _describe(
    value: object,
    *,
    expected_zone: str,
) -> Tuple[str, ...]:
    if type(value) is not dict or set(value) != {
        "availability_zone",
        "instance_ids",
        "request_id",
        "response_identity_sha256",
        "observed_at",
    }:
        raise Task10SoleSenderError("describe readback is not closed")
    instances = value["instance_ids"]
    if (
        value["availability_zone"] != expected_zone
        or type(instances) is not tuple
        or len(instances) > 1
        or any(
            type(instance) is not str
            or _INSTANCE.fullmatch(instance) is None
            for instance in instances
        )
        or type(value["request_id"]) is not str
        or not value["request_id"]
    ):
        raise Task10SoleSenderError("describe readback is foreign")
    _sha(value["response_identity_sha256"], "describe response")
    try:
        observed = datetime.strptime(
            value["observed_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as error:
        raise Task10SoleSenderError("describe readback time drifted") from error
    if observed.tzinfo is None:
        raise Task10SoleSenderError("describe readback time drifted")
    return instances


def run_task10_sole_sender(
    event: object,
    *,
    services: Task10SoleSenderServices,
) -> Mapping[str, object]:
    """Send at most one Task 9-authorized request per ordered AZ."""

    value = _event(event)
    if not isinstance(services, Task10SoleSenderServices):
        raise Task10SoleSenderError("sole-sender services must be typed")
    now = services.clock()
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise Task10SoleSenderError("sole-sender clock must be aware")
    read = getattr(services.authority_reader, "read", None)
    complete = getattr(services.completion_sender, "complete", None)
    describe = getattr(services.describe_reader, "describe", None)
    publish = getattr(services.terminal_publisher, "publish", None)
    if not all(callable(method) for method in (read, complete, describe, publish)):
        raise Task10SoleSenderError("sole-sender boundary is incomplete")
    authority = validate_task10_sole_sender_authority(read(value))
    if (
        authority.activation_id != value["activation_id"]
        or authority.generation != value["generation"]
        or authority.generation_text != value["generation_text"]
    ):
        raise Task10SoleSenderError("sole-sender authority is foreign")
    outcomes = []
    instance_id = None
    availability_zone = None
    for attempt in authority.attempts:
        result = complete(
            {
                "activation_id": authority.activation_id,
                "allocation_ordinal": attempt.allocation_ordinal,
            }
        )
        if (
            not isinstance(result, CompletionResult)
            or result.classification
            not in {"POSITIVE_REJECTION", "DIRECT_SUCCESS", "AMBIGUOUS"}
            or result.attempt != 1
        ):
            raise Task10SoleSenderError("completion response is foreign")
        _sha(result.evidence_identity_sha256, "completion evidence")
        readback = _describe(
            describe(
                authority=authority,
                attempt=attempt,
                completion=result,
            ),
            expected_zone=attempt.availability_zone,
        )
        if (
            result.classification == "DIRECT_SUCCESS"
            and tuple(result.instance_ids) != readback
        ):
            raise Task10SoleSenderError("direct response and describe drifted")
        if readback:
            instance_id = readback[0]
            availability_zone = attempt.availability_zone
            outcome = "WORKER_ALLOCATED"
        else:
            outcome = "CAPACITY_REJECTED"
        outcomes.append(
            {
                "attempt": attempt.attempt,
                "availability_zone": attempt.availability_zone,
                "outcome": outcome,
            }
        )
        if instance_id is not None:
            break
    classification = (
        "WORKER_ALLOCATED"
        if instance_id is not None
        else "CAPACITY_EXHAUSTED"
    )
    reconciliation = build_capacity_reconciliation(
        schema_version=1,
        record_type="glm52_task10_capacity_reconciliation_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id=authority.activation_id,
        generation=authority.generation,
        generation_text=authority.generation_text,
        classification=classification,
        execution_arn=value["execution_arn"],
        capacity_outcomes=outcomes,
        instance_id=instance_id,
        availability_zone=availability_zone,
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
    publication = publish(reconciliation)
    if type(publication) is not dict:
        raise Task10SoleSenderError("terminal publication is not exact")
    return {
        "classification": reconciliation.classification,
        "capacity_outcomes": reconciliation.capacity_outcomes,
        "instance_id": reconciliation.instance_id,
        "availability_zone": reconciliation.availability_zone,
        "reconciliation_identity_sha256": (
            reconciliation.canonical_identity_sha256
        ),
        "publication": publication,
    }


__all__ = [
    "Task10CapacityAttemptAuthority",
    "Task10SoleSenderAuthority",
    "Task10SoleSenderError",
    "Task10SoleSenderServices",
    "build_task10_sole_sender_authority",
    "run_task10_sole_sender",
    "task10_sole_sender_authority_from_mapping",
    "validate_task10_sole_sender_authority",
]
