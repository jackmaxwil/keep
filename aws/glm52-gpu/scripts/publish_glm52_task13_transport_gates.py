#!/usr/bin/env python3
"""Conditionally publish and reconcile the two observed Task 13 gates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_transport_gates import (  # noqa: E402
    SEMANTIC_GATE_PINS,
    TransportGateError,
    make_pytest_observer,
    validate_t01_t25_gate,
    validate_transport_mutant_gate,
)

ACCOUNT_ID = "246813579024"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
PROFILE = "keep-gpu"
REGION = "us-west-2"
GATES = (
    (
        "T01_T25_GATE",
        "task13/gates/t01-t25.json",
        validate_t01_t25_gate,
    ),
    (
        "TRANSPORT_22_MUTANT_GATE",
        "task13/gates/transport-22-mutants.json",
        validate_transport_mutant_gate,
    ),
)


def _read_gate(path: Path, validator: object) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if raw != canonical_json_bytes(value) + b"\n":
        raise TransportGateError("transport gate is not canonical JSON plus LF")
    validator(
        value,
        repo_root=REPO_ROOT,
        observe=make_pytest_observer(repo_root=REPO_ROOT),
    )
    return value, raw


def _all_versions(client: object, key: str) -> tuple[list[object], list[object]]:
    versions: list[object] = []
    markers: list[object] = []
    request: dict[str, object] = {
        "Bucket": BUCKET,
        "Prefix": key,
        "ExpectedBucketOwner": ACCOUNT_ID,
    }
    while True:
        response = client.list_object_versions(**request)
        versions.extend(response.get("Versions", []))
        markers.extend(response.get("DeleteMarkers", []))
        if response.get("IsTruncated") is not True:
            return versions, markers
        key_marker = response.get("NextKeyMarker")
        version_marker = response.get("NextVersionIdMarker")
        if not key_marker or not version_marker:
            raise TransportGateError("transport gate pagination is incomplete")
        request["KeyMarker"] = key_marker
        request["VersionIdMarker"] = version_marker


def _publish_one(
    client: object,
    *,
    artifact_kind: str,
    key: str,
    raw: bytes,
    reconcilable_errors: tuple[type[BaseException], ...],
) -> dict[str, object]:
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    try:
        client.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=raw,
            IfNoneMatch="*",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum,
            ContentType="application/json",
            ExpectedBucketOwner=ACCOUNT_ID,
        )
    except reconcilable_errors:
        # A transport/status ambiguity authorizes readback, never another PUT.
        versions, markers = _all_versions(client, key)
    else:
        versions, markers = _all_versions(client, key)
    exact = [row for row in versions if type(row) is dict and row.get("Key") == key]
    if markers or len(exact) != 1 or not exact[0].get("VersionId"):
        raise TransportGateError("transport gate did not reconcile to one version")
    version_id = exact[0]["VersionId"]
    response = client.get_object(
        Bucket=BUCKET,
        Key=key,
        VersionId=version_id,
        ChecksumMode="ENABLED",
        ExpectedBucketOwner=ACCOUNT_ID,
    )
    body = response.get("Body")
    if not callable(getattr(body, "read", None)):
        raise TransportGateError("transport gate readback body is missing")
    observed = body.read()
    close = getattr(body, "close", None)
    if callable(close):
        close()
    pin = SEMANTIC_GATE_PINS[artifact_kind]
    if (
        observed != raw
        or response.get("VersionId") != version_id
        or response.get("ChecksumSHA256") != checksum
        or hashlib.sha256(raw).hexdigest() != pin["file_sha256"]
    ):
        raise TransportGateError("transport gate exact readback drifted")
    return {
        "artifact_kind": artifact_kind,
        "bucket": BUCKET,
        "key": key,
        "version_id": version_id,
        "file_sha256": pin["file_sha256"],
        "body_sha256": pin["body_sha256"],
    }


def _write_new(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, canonical_json_bytes(value) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-coordinates", required=True, type=Path)
    args = parser.parse_args()
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError

        session = boto3.Session(profile_name=PROFILE, region_name=REGION)
        caller = session.client("sts").get_caller_identity()
        if caller.get("Account") != ACCOUNT_ID:
            raise TransportGateError("AWS caller account drifted")
        client = session.client(
            "s3",
            config=Config(retries={"max_attempts": 0}),
        )
        coordinates = []
        for artifact_kind, key, validator in GATES:
            _value, raw = _read_gate(REPO_ROOT / key, validator)
            coordinates.append(
                _publish_one(
                    client,
                    artifact_kind=artifact_kind,
                    key=key,
                    raw=raw,
                    reconcilable_errors=(
                        BotoCoreError,
                        ClientError,
                        TimeoutError,
                        ConnectionError,
                    ),
                )
            )
        _write_new(args.output_coordinates, coordinates)
    except (OSError, TransportGateError) as error:
        print(f"Task 13 gate publication refused: {error}", file=sys.stderr)
        return 64
    print(str(args.output_coordinates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
