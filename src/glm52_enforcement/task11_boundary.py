"""Immutable production-boundary manifest for the Task 11 closure.

The Step Functions input carries only the exact coordinate of this manifest.
Every effect-specific input is another versioned immutable object.  No
serialized success receipt, authority audit, direct response, or live-read
result is representable by this contract.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Mapping, Tuple

from .canonical import canonical_json_bytes, canonical_sha256


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
BOUNDARY_INPUT_KINDS = (
    "GPU_SPEND_SOURCE_REQUEST",
    "SUBMISSION_INTENT_SOURCE_REQUEST",
    "CONTROLLER_BASELINE_SOURCE_REQUEST",
    "CONTROL_PLANE_READINESS_SOURCE_REQUEST",
    "SUBMISSION_ACQUISITION_SOURCE_REQUEST",
    "BATCH_SUCCESSOR_REQUEST",
    "FENCE_EXECUTION_REQUEST",
    "CLAIM_CREATE_REQUEST",
    "DECISION_CREATE_REQUEST",
    "TERMINAL_V1_WRITE_REQUEST",
    "H1D_LIVE_REQUEST",
    "SKY_ADMISSION_REQUEST",
    "CORRELATION_HANDOFF_REQUEST",
    "TASK9_DEPLOYED_IDENTITY_COORDINATE",
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_ACTIVATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[\x21-\x7e]{1,1024}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class Task11BoundaryError(ValueError):
    """The immutable Task 11 input boundary failed closed."""


@dataclass(frozen=True)
class Task11InputCoordinate:
    input_kind: str
    bucket: str
    key: str
    version_id: str
    file_sha256: str
    body_sha256: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class Task11BoundaryDocument:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    generation: int
    generation_text: str
    campaign_identity_sha256: str
    state_machine_version_arn: str
    action_key: str
    inputs: Tuple[Task11InputCoordinate, ...]
    canonical_identity_sha256: str


@dataclass(frozen=True)
class Task11BoundaryCoordinate:
    bucket: str
    key: str
    version_id: str
    file_sha256: str
    body_sha256: str


def _fail(message: str) -> None:
    raise Task11BoundaryError(message)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail(label + " must be a lowercase SHA-256")
    return value


def _coordinate_body(value: Task11InputCoordinate) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def _validate_coordinate(
    value: object,
    *,
    index: int,
    activation_id: str,
    generation_text: str,
) -> Task11InputCoordinate:
    if type(value) is not Task11InputCoordinate:
        _fail("Task 11 input coordinate must be exact and typed")
    if (
        value.input_kind != BOUNDARY_INPUT_KINDS[index]
        or _BUCKET.fullmatch(value.bucket) is None
        or value.key
        != (
            (
                "campaigns/"
                + RUN_ID
                + "/authorities/task9/"
                + activation_id
                + "/TASK9_DEPLOYED_IDENTITY.json"
            )
            if index == 13
            else (
                "campaigns/"
                + RUN_ID
                + "/authorities/task11/"
                + activation_id
                + "/"
                + generation_text
                + "/"
                + f"{index + 1:02d}-"
                + value.input_kind.lower().replace("_", "-")
                + ".json"
            )
        )
        or _VERSION.fullmatch(value.version_id) is None
        or value.version_id == "null"
    ):
        _fail("Task 11 input coordinate identity drifted")
    _sha(value.file_sha256, "Task 11 input file identity")
    _sha(value.body_sha256, "Task 11 input body identity")
    if value.canonical_identity_sha256 != canonical_sha256(
        _coordinate_body(value)
    ):
        _fail("Task 11 input coordinate self-hash drifted")
    return value


def build_task11_input_coordinate(
    *,
    input_kind: str,
    bucket: str,
    key: str,
    version_id: str,
    file_sha256: str,
    body_sha256: str,
) -> Task11InputCoordinate:
    provisional = Task11InputCoordinate(
        input_kind=input_kind,
        bucket=bucket,
        key=key,
        version_id=version_id,
        file_sha256=file_sha256,
        body_sha256=body_sha256,
        canonical_identity_sha256="",
    )
    return Task11InputCoordinate(
        **{
            **asdict(provisional),
            "canonical_identity_sha256": canonical_sha256(
                _coordinate_body(provisional)
            ),
        }
    )


def _document_body(value: Task11BoundaryDocument) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def validate_task11_boundary_document(
    value: object,
) -> Task11BoundaryDocument:
    if type(value) is not Task11BoundaryDocument:
        _fail("Task 11 boundary document must be exact and typed")
    if (
        value.schema_version != 1
        or value.record_type != "glm52_task11_production_boundary_v1"
        or value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
        or type(value.activation_id) is not str
        or _ACTIVATION.fullmatch(value.activation_id) is None
        or type(value.generation) is not int
        or value.generation <= 0
        or value.generation_text != f"{value.generation:08d}"
        or type(value.state_machine_version_arn) is not str
        or not value.state_machine_version_arn.startswith(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-production:"
        )
        or type(value.action_key) is not str
        or "SKY_POST" not in value.action_key
        or type(value.inputs) is not tuple
        or len(value.inputs) != len(BOUNDARY_INPUT_KINDS)
    ):
        _fail("Task 11 boundary document identity drifted")
    for field in (
        "campaign_identity_sha256",
    ):
        _sha(getattr(value, field), field)
    coordinates = tuple(
        _validate_coordinate(
            coordinate,
            index=index,
            activation_id=value.activation_id,
            generation_text=value.generation_text,
        )
        for index, coordinate in enumerate(value.inputs)
    )
    if (
        len({item.canonical_identity_sha256 for item in coordinates})
        != len(coordinates)
        or len({(item.bucket, item.key, item.version_id) for item in coordinates})
        != len(coordinates)
        or len({item.bucket for item in coordinates}) != 1
        or value.canonical_identity_sha256
        != canonical_sha256(_document_body(value))
    ):
        _fail("Task 11 boundary coordinate set or self-hash drifted")
    return value


def build_task11_boundary_document(
    *,
    activation_id: str,
    generation: int,
    campaign_identity_sha256: str,
    state_machine_version_arn: str,
    action_key: str,
    inputs: Tuple[Task11InputCoordinate, ...],
) -> Task11BoundaryDocument:
    provisional = Task11BoundaryDocument(
        schema_version=1,
        record_type="glm52_task11_production_boundary_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id=activation_id,
        generation=generation,
        generation_text=f"{generation:08d}",
        campaign_identity_sha256=campaign_identity_sha256,
        state_machine_version_arn=state_machine_version_arn,
        action_key=action_key,
        inputs=inputs,
        canonical_identity_sha256="",
    )
    result = Task11BoundaryDocument(
        **{
            **asdict(provisional),
            "inputs": inputs,
            "canonical_identity_sha256": canonical_sha256(
                _document_body(provisional)
            ),
        }
    )
    return validate_task11_boundary_document(result)


def task11_boundary_from_bytes(raw: object) -> Task11BoundaryDocument:
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or len(raw) > 2 * 1024 * 1024
    ):
        _fail("Task 11 boundary bytes are invalid")
    try:
        value = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Task11BoundaryError(
            "Task 11 boundary bytes are invalid JSON"
        ) from exc
    if (
        type(value) is not dict
        or canonical_json_bytes(value) + b"\n" != raw
    ):
        _fail("Task 11 boundary bytes are not canonical JSON plus LF")
    expected = set(Task11BoundaryDocument.__dataclass_fields__)
    if set(value) != expected or type(value.get("inputs")) is not list:
        _fail("Task 11 boundary field set drifted")
    coordinate_fields = set(Task11InputCoordinate.__dataclass_fields__)
    inputs = []
    for coordinate in value["inputs"]:
        if type(coordinate) is not dict or set(coordinate) != coordinate_fields:
            _fail("Task 11 input coordinate field set drifted")
        inputs.append(Task11InputCoordinate(**coordinate))
    try:
        document = Task11BoundaryDocument(
            **{
                **value,
                "inputs": tuple(inputs),
            }
        )
    except TypeError as exc:
        raise Task11BoundaryError(
            "Task 11 boundary document is malformed"
        ) from exc
    return validate_task11_boundary_document(document)


def _transport_metadata(value: object, operation: str) -> Mapping[str, object]:
    metadata = value.get("ResponseMetadata") if type(value) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        _fail(operation + " response is unauthenticated")
    return metadata


def load_task11_boundary(
    *,
    s3: object,
    coordinate: Task11BoundaryCoordinate,
) -> Task11BoundaryDocument:
    """Load one exact immutable manifest by opaque VersionId."""

    if type(coordinate) is not Task11BoundaryCoordinate:
        raise TypeError("boundary coordinate must be exact and typed")
    if (
        _BUCKET.fullmatch(coordinate.bucket) is None
        or _VERSION.fullmatch(coordinate.version_id) is None
        or coordinate.version_id == "null"
    ):
        _fail("Task 11 boundary coordinate is invalid")
    _sha(coordinate.file_sha256, "Task 11 boundary file identity")
    _sha(coordinate.body_sha256, "Task 11 boundary body identity")
    get = getattr(s3, "get_object", None)
    if not callable(get):
        _fail("Task 11 boundary S3 reader is absent")
    value = get(
        Bucket=coordinate.bucket,
        Key=coordinate.key,
        VersionId=coordinate.version_id,
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    _transport_metadata(value, "GetObject")
    stream = value.get("Body") if type(value) is dict else None
    read = getattr(stream, "read", None)
    checksum = value.get("ChecksumSHA256") if type(value) is dict else None
    if (
        value.get("VersionId") != coordinate.version_id
        or not callable(read)
        or type(checksum) is not str
    ):
        _fail("Task 11 boundary version response drifted")
    raw = read(2 * 1024 * 1024 + 1)
    if (
        type(raw) is not bytes
        or hashlib.sha256(raw).hexdigest() != coordinate.file_sha256
        or checksum
        != base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    ):
        _fail("Task 11 boundary file/checksum identity drifted")
    document = task11_boundary_from_bytes(raw)
    if document.canonical_identity_sha256 != coordinate.body_sha256:
        _fail("Task 11 boundary body identity drifted")
    return document


__all__ = [
    "ACCOUNT_ID",
    "BOUNDARY_INPUT_KINDS",
    "REGION",
    "RUN_ID",
    "Task11BoundaryCoordinate",
    "Task11BoundaryDocument",
    "Task11BoundaryError",
    "Task11InputCoordinate",
    "build_task11_boundary_document",
    "build_task11_input_coordinate",
    "load_task11_boundary",
    "task11_boundary_from_bytes",
    "validate_task11_boundary_document",
]
