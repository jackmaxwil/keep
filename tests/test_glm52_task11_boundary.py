"""Task 11 immutable production-boundary input contract."""

from __future__ import annotations

from dataclasses import asdict, replace
import base64
import hashlib
import io
import json

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.task11_boundary import (
    BOUNDARY_INPUT_KINDS,
    Task11BoundaryCoordinate,
    Task11BoundaryError,
    build_task11_boundary_document,
    build_task11_input_coordinate,
    load_task11_boundary,
    task11_boundary_from_bytes,
)


BUCKET = "keep-glm52-us-west-2-246813579024"
ACTIVATION_ID = "activation-0001"
GENERATION_TEXT = "00000001"
SHA = hashlib.sha256(b"task11-boundary").hexdigest()


def _inputs():
    return tuple(
        build_task11_input_coordinate(
            input_kind=kind,
            bucket=BUCKET,
            key=(
                (
                    "campaigns/glm52-sky-20260724/authorities/task9/"
                    + ACTIVATION_ID
                    + "/TASK9_DEPLOYED_IDENTITY.json"
                )
                if index == 13
                else (
                    "campaigns/glm52-sky-20260724/authorities/task11/"
                    + ACTIVATION_ID
                    + "/"
                    + GENERATION_TEXT
                    + "/"
                    + f"{index + 1:02d}-"
                    + kind.lower().replace("_", "-")
                    + ".json"
                )
            ),
            version_id="opaque-version-" + str(index + 1),
            file_sha256=hashlib.sha256(
                ("file-" + kind).encode()
            ).hexdigest(),
            body_sha256=hashlib.sha256(
                ("body-" + kind).encode()
            ).hexdigest(),
        )
        for index, kind in enumerate(BOUNDARY_INPUT_KINDS)
    )


def _document():
    return build_task11_boundary_document(
        activation_id=ACTIVATION_ID,
        generation=1,
        campaign_identity_sha256=hashlib.sha256(
            b"campaign"
        ).hexdigest(),
        state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-production:7"
        ),
        action_key=(
            "ACTIVATION#activation-0001#ACTION#SKY_POST#00000001"
        ),
        inputs=_inputs(),
    )


def _raw(document=None) -> bytes:
    value = _document() if document is None else document
    return canonical_json_bytes(asdict(value)) + b"\n"


class ExactS3:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.version_id = "task11-boundary-version-1"
        self.calls = []

    def get_object(self, **kwargs: object):
        self.calls.append(dict(kwargs))
        return {
            "Body": io.BytesIO(self.raw),
            "VersionId": self.version_id,
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(self.raw).digest()
            ).decode("ascii"),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "get-boundary-1",
            },
        }


def test_task11_fix1_red_boundary_manifest_is_closed_and_ordered() -> None:
    document = task11_boundary_from_bytes(_raw())
    assert len(BOUNDARY_INPUT_KINDS) == 14
    assert BOUNDARY_INPUT_KINDS[-1] == (
        "TASK9_DEPLOYED_IDENTITY_COORDINATE"
    )
    assert tuple(item.input_kind for item in document.inputs) == (
        BOUNDARY_INPUT_KINDS
    )
    assert document.activation_id == ACTIVATION_ID
    assert document.generation_text == GENERATION_TEXT
    with pytest.raises(Task11BoundaryError):
        task11_boundary_from_bytes(_raw(replace(document, inputs=document.inputs[::-1])))


@pytest.mark.parametrize(
    "mutation",
    (
        "caller_success",
        "duplicate_coordinate",
        "wrong_key",
        "wrong_version",
        "wrong_self_hash",
        "noncanonical",
    ),
)
def test_task11_fix1_red_boundary_rejects_authority_and_coordinate_mutants(
    mutation: str,
) -> None:
    document = _document()
    if mutation == "caller_success":
        value = asdict(document)
        value["closure_budget_status"] = "CLOSURE_BUDGET_PROVEN"
        raw = canonical_json_bytes(value) + b"\n"
    elif mutation == "duplicate_coordinate":
        raw = _raw(
            replace(
                document,
                inputs=(document.inputs[0], *document.inputs[:-1]),
            )
        )
    elif mutation == "wrong_key":
        raw = _raw(
            replace(
                document,
                inputs=(
                    replace(document.inputs[0], key="foreign.json"),
                    *document.inputs[1:],
                ),
            )
        )
    elif mutation == "wrong_version":
        raw = _raw(
            replace(
                document,
                inputs=(
                    replace(document.inputs[0], version_id="null"),
                    *document.inputs[1:],
                ),
            )
        )
    elif mutation == "wrong_self_hash":
        raw = _raw(
            replace(document, canonical_identity_sha256="0" * 64)
        )
    else:
        raw = json.dumps(
            asdict(document),
            ensure_ascii=True,
            indent=2,
        ).encode("ascii") + b"\n"
    with pytest.raises(Task11BoundaryError):
        task11_boundary_from_bytes(raw)


def test_task11_fix1_red_loader_uses_exact_version_checksum_and_owner() -> None:
    raw = _raw()
    s3 = ExactS3(raw)
    coordinate = Task11BoundaryCoordinate(
        bucket=BUCKET,
        key=(
            "campaigns/glm52-sky-20260724/authorities/task11/"
            "activation-0001/00000001.json"
        ),
        version_id=s3.version_id,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=_document().canonical_identity_sha256,
    )
    result = load_task11_boundary(s3=s3, coordinate=coordinate)
    assert result.canonical_identity_sha256 == coordinate.body_sha256
    assert s3.calls == [
        {
            "Bucket": BUCKET,
            "Key": coordinate.key,
            "VersionId": coordinate.version_id,
            "ExpectedBucketOwner": "246813579024",
            "ChecksumMode": "ENABLED",
        }
    ]
    s3.raw += b"x"
    with pytest.raises(Task11BoundaryError):
        load_task11_boundary(s3=s3, coordinate=coordinate)
