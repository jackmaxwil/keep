"""Pure Task 13 clean-rehearsal evidence join and coordinate contract."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import re
from typing import Any, Optional

from .canonical import canonical_json_bytes, canonical_sha256
from .decision_closure import parse_deployed_gate_document
from .task13_campaign_package import (
    ACCOUNT_ID,
    CAMPAIGN_BUCKET,
    REGION,
    RUN_ID,
    canonical_campaign_package_bytes,
    validate_campaign_artifact_coordinate,
    validate_finalize_invoke_result,
)
from .task13_fixed_artifacts import (
    validate_repository_archive_manifest,
    validate_repository_archive_manifest_key,
)


RECORD_TYPE = "glm52_task13_clean_rehearsal_evidence_v1"
CLEAN_REHEARSAL_KEY_PREFIX = "task13/gates/clean-rehearsal/"
REHEARSAL_BUCKET = "keep-glm52-h1g-rehearsal-246813579024-us-west-2"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVATION = re.compile(r"[a-z0-9][a-z0-9-]{2,63}\Z")
_VERSION = re.compile(r"(?!null\Z)(?!None\Z)[\x21-\x7e]{1,1024}\Z")
_COLLECTOR_ARN = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:function:"
    r"keep-glm52-h1g-rehearsal-collector:(?P<version>[1-9][0-9]*)\Z"
)
_CLEAN_KEY = re.compile(
    r"task13/gates/clean-rehearsal/"
    r"(?P<activation>[a-z0-9][a-z0-9-]{2,63})/"
    r"(?P<identity>[0-9a-f]{64})\.json\Z"
)
_EVIDENCE_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "run_id",
    "activation_id",
    "predecessor_package",
    "repository_archive",
    "finalizer",
    "immutable_gate",
    "canonical_identity_sha256",
}
_PREDECESSOR_FIELDS = {
    "package_identity_sha256",
    "package_file_sha256",
    "reviewed_artifacts_identity_sha256",
}
_ARCHIVE_FIELDS = {
    "coordinate",
    "manifest_identity_sha256",
    "manifest_file_sha256",
    "archive_file_sha256",
    "archive_size_bytes",
    "archive_payload",
}
_FINALIZER_FIELDS = {
    "collector_function_version_arn",
    "collector_function_version",
    "status",
    "payload_identity_sha256",
    "payload_file_sha256",
    "measurement_count",
    "cold_environment_count",
    "measurements_identity_sha256",
}
_GATE_FIELDS = {
    "bucket",
    "key",
    "version_id",
    "file_sha256",
    "body_sha256",
    "checksum_sha256_base64",
    "deployment_identity_sha256",
    "canonical_body_sha256",
    "measurement_count",
    "cold_environment_count",
    "measurements_identity_sha256",
}
_COORDINATE_FIELDS = {
    "artifact_kind",
    "bucket",
    "key",
    "version_id",
    "file_sha256",
    "body_sha256",
}


class CleanRehearsalEvidenceError(ValueError):
    """Clean rehearsal inputs do not form one exact immutable authority."""


def _fail(message: str) -> None:
    raise CleanRehearsalEvidenceError(message)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(label + " must be one lowercase SHA-256")
    return value


def _version(value: object, label: str) -> str:
    if type(value) is not str or _VERSION.fullmatch(value) is None:
        _fail(label + " is not one immutable VersionId")
    return value


def _copy_json(value: object, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise CleanRehearsalEvidenceError(
            label + " is not canonical JSON data"
        ) from error


def _bytes(value: object, label: str) -> bytes:
    if type(value) is bytes:
        return value
    if type(value) is bytearray:
        return bytes(value)
    if type(value) is memoryview:
        return value.tobytes()
    if isinstance(value, io.BufferedIOBase) or hasattr(value, "read"):
        raw = value.read()
        if type(raw) is bytes:
            return raw
    _fail(label + " must be exact bytes")


def _canonical_value(raw: bytes, value: object, label: str, *, lf: bool) -> None:
    suffix = b"\n" if lf else b""
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CleanRehearsalEvidenceError(label + " is not JSON") from error
    if (
        type(decoded) is not type(value)
        or decoded != value
        or raw != canonical_json_bytes(value) + suffix
    ):
        _fail(label + " canonical bytes/value drifted")


def _finalizer_result(
    result: object,
    *,
    collector_arn: str,
    collector_version: str,
) -> tuple[dict[str, object], bytes]:
    if type(result) is not dict or "Payload" not in result:
        _fail("Task11 FINALIZE invoke result is not exact")
    payload_raw = _bytes(result["Payload"], "Task11 FINALIZE payload")
    normalized = dict(result)
    normalized["Payload"] = payload_raw
    payload = validate_finalize_invoke_result(
        normalized,
        expected_function_version_arn=collector_arn,
        expected_version=collector_version,
    )
    if payload_raw != canonical_json_bytes(payload):
        _fail("Task11 FINALIZE payload canonicality drifted")
    return payload, payload_raw


def _checksum(value: object) -> str:
    if type(value) is not str:
        _fail("immutable gate checksum is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise CleanRehearsalEvidenceError(
            "immutable gate checksum is invalid"
        ) from error
    if len(decoded) != 32:
        _fail("immutable gate checksum is invalid")
    return value


def build_clean_rehearsal_evidence(
    *,
    predecessor_package: object,
    predecessor_package_bytes: object,
    predecessor_reviewed_artifacts: object,
    repository_archive_coordinate: object,
    repository_archive_manifest: object,
    repository_archive_manifest_bytes: object,
    finalize_invoke_result: object,
    gate_readback: object,
    gate_readback_bytes: object,
) -> dict[str, object]:
    """Join exact predecessor, archive, Task11 finalizer, and gate truth."""

    try:
        expected_package_raw = canonical_campaign_package_bytes(
            predecessor_package
        )
    except ValueError as error:
        raise CleanRehearsalEvidenceError(str(error)) from error
    package_raw = _bytes(
        predecessor_package_bytes,
        "predecessor package bytes",
    )
    if package_raw != expected_package_raw:
        _fail("predecessor package canonical bytes/value drifted")
    if (
        type(predecessor_package) is not dict
        or predecessor_package.get("package_phase") != "PREQUALIFICATION"
        or predecessor_package.get("predecessor_identity") is not None
        or predecessor_package.get("predecessor_package") is not None
        or type(predecessor_reviewed_artifacts) is not list
        or predecessor_package.get("reviewed_artifacts")
        != predecessor_reviewed_artifacts
    ):
        _fail("predecessor package or reviewed-artifact list drifted")
    activation_id = predecessor_package.get("activation_id")
    if type(activation_id) is not str or _ACTIVATION.fullmatch(activation_id) is None:
        _fail("predecessor activation identity drifted")

    try:
        archive_coordinate = validate_campaign_artifact_coordinate(
            repository_archive_coordinate
        )
    except ValueError as error:
        raise CleanRehearsalEvidenceError(str(error)) from error
    predecessor_archive = next(
        (
            row
            for row in predecessor_reviewed_artifacts
            if type(row) is dict
            and row.get("artifact_kind") == "REPOSITORY_ARCHIVE"
        ),
        None,
    )
    if archive_coordinate != predecessor_archive:
        _fail("repository archive coordinate drifted from predecessor")
    if type(repository_archive_manifest) is not dict:
        _fail("repository archive manifest value is not exact")
    archive_raw = _bytes(
        repository_archive_manifest_bytes,
        "repository archive manifest bytes",
    )
    _canonical_value(
        archive_raw,
        repository_archive_manifest,
        "repository archive manifest",
        lf=True,
    )
    try:
        archive_manifest = validate_repository_archive_manifest(
            repository_archive_manifest,
            expected_activation_id=activation_id,
            expected_archive_sha256=repository_archive_manifest.get(
                "archive_file_sha256"
            ),
            expected_archive_size=repository_archive_manifest.get(
                "archive_size_bytes"
            ),
            expected_bucket=archive_coordinate["bucket"],
        )
    except ValueError as error:
        raise CleanRehearsalEvidenceError(str(error)) from error
    if (
        archive_coordinate["artifact_kind"] != "REPOSITORY_ARCHIVE"
        or archive_coordinate["bucket"] != CAMPAIGN_BUCKET
        or archive_coordinate["file_sha256"]
        != hashlib.sha256(archive_raw).hexdigest()
        or archive_coordinate["body_sha256"]
        != archive_manifest["canonical_identity_sha256"]
    ):
        _fail("repository archive manifest coordinate identity drifted")

    rehearsal = predecessor_package.get("rehearsal")
    if type(rehearsal) is not dict:
        _fail("predecessor rehearsal contract is missing")
    finalizer_invocation = rehearsal.get("finalize_invocation")
    if type(finalizer_invocation) is not dict:
        _fail("predecessor finalizer contract is missing")
    collector_arn = finalizer_invocation.get("function_version_arn")
    collector_version = finalizer_invocation.get("qualifier")
    event = finalizer_invocation.get("event")
    if (
        type(collector_arn) is not str
        or type(collector_version) is not str
        or type(event) is not dict
        or event.get("activation_id") != activation_id
    ):
        _fail("predecessor finalizer identity drifted")
    try:
        finalizer_payload, finalizer_raw = _finalizer_result(
            finalize_invoke_result,
            collector_arn=collector_arn,
            collector_version=collector_version,
        )
    except ValueError as error:
        raise CleanRehearsalEvidenceError(str(error)) from error
    if finalizer_payload["activation_id"] != activation_id:
        _fail("Task11 FINALIZE activation drifted")

    if type(gate_readback) is not dict:
        _fail("immutable gate readback value is not exact")
    gate_raw = _bytes(gate_readback_bytes, "immutable gate readback bytes")
    _canonical_value(
        gate_raw,
        gate_readback,
        "immutable gate readback",
        lf=False,
    )
    deployment_identity = gate_readback.get("deployment_identity_sha256")
    try:
        gate = parse_deployed_gate_document(
            gate_raw,
            activation_id=activation_id,
            deployment_identity_sha256=_sha(
                deployment_identity,
                "immutable gate deployment identity",
            ),
        )
    except ValueError as error:
        raise CleanRehearsalEvidenceError(str(error)) from error
    if (
        hashlib.sha256(gate_raw).hexdigest()
        != finalizer_payload["file_sha256"]
        or gate.canonical_body_sha256 != finalizer_payload["body_sha256"]
        or gate.measurement_count != finalizer_payload["measurement_count"]
        or gate.cold_environment_count
        != finalizer_payload["cold_environment_count"]
        or gate.measurements_identity_sha256
        != finalizer_payload["measurements_identity_sha256"]
        or finalizer_payload["checksum_sha256_base64"]
        != base64.b64encode(hashlib.sha256(gate_raw).digest()).decode(
            "ascii"
        )
    ):
        _fail("Task11 FINALIZE and immutable gate readback drifted")

    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": RECORD_TYPE,
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "predecessor_package": {
            "package_identity_sha256": predecessor_package[
                "canonical_identity_sha256"
            ],
            "package_file_sha256": hashlib.sha256(package_raw).hexdigest(),
            "reviewed_artifacts_identity_sha256": canonical_sha256(
                predecessor_reviewed_artifacts
            ),
        },
        "repository_archive": {
            "coordinate": archive_coordinate,
            "manifest_identity_sha256": archive_manifest[
                "canonical_identity_sha256"
            ],
            "manifest_file_sha256": hashlib.sha256(archive_raw).hexdigest(),
            "archive_file_sha256": archive_manifest[
                "archive_file_sha256"
            ],
            "archive_size_bytes": archive_manifest["archive_size_bytes"],
            "archive_payload": archive_manifest["archive"],
        },
        "finalizer": {
            "collector_function_version_arn": collector_arn,
            "collector_function_version": collector_version,
            "status": finalizer_payload["status"],
            "payload_identity_sha256": finalizer_payload[
                "canonical_identity_sha256"
            ],
            "payload_file_sha256": hashlib.sha256(finalizer_raw).hexdigest(),
            "measurement_count": finalizer_payload["measurement_count"],
            "cold_environment_count": finalizer_payload[
                "cold_environment_count"
            ],
            "measurements_identity_sha256": finalizer_payload[
                "measurements_identity_sha256"
            ],
        },
        "immutable_gate": {
            "bucket": REHEARSAL_BUCKET,
            "key": finalizer_payload["key"],
            "version_id": finalizer_payload["version_id"],
            "file_sha256": finalizer_payload["file_sha256"],
            "body_sha256": finalizer_payload["body_sha256"],
            "checksum_sha256_base64": _checksum(
                finalizer_payload["checksum_sha256_base64"]
            ),
            "deployment_identity_sha256": gate.deployment_identity_sha256,
            "canonical_body_sha256": gate.canonical_body_sha256,
            "measurement_count": gate.measurement_count,
            "cold_environment_count": gate.cold_environment_count,
            "measurements_identity_sha256": (
                gate.measurements_identity_sha256
            ),
        },
    }
    evidence = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    return validate_clean_rehearsal_evidence(evidence)


def validate_clean_rehearsal_evidence(
    evidence: object,
) -> dict[str, object]:
    """Validate the complete canonical evidence schema without external I/O."""

    if type(evidence) is not dict or set(evidence) != _EVIDENCE_FIELDS:
        _fail("clean rehearsal evidence schema drifted")
    body = dict(evidence)
    identity = body.pop("canonical_identity_sha256", None)
    activation_id = evidence.get("activation_id")
    predecessor = evidence.get("predecessor_package")
    archive = evidence.get("repository_archive")
    finalizer = evidence.get("finalizer")
    gate = evidence.get("immutable_gate")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("record_type") != RECORD_TYPE
        or evidence.get("account_id") != ACCOUNT_ID
        or evidence.get("region") != REGION
        or evidence.get("run_id") != RUN_ID
        or type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
        or type(predecessor) is not dict
        or set(predecessor) != _PREDECESSOR_FIELDS
        or type(archive) is not dict
        or set(archive) != _ARCHIVE_FIELDS
        or type(finalizer) is not dict
        or set(finalizer) != _FINALIZER_FIELDS
        or type(gate) is not dict
        or set(gate) != _GATE_FIELDS
        or identity != canonical_sha256(body)
    ):
        _fail("clean rehearsal evidence identity drifted")
    for field in _PREDECESSOR_FIELDS:
        _sha(predecessor[field], "predecessor " + field)
    for field in (
        "manifest_identity_sha256",
        "manifest_file_sha256",
        "archive_file_sha256",
    ):
        _sha(archive[field], "repository archive " + field)
    if type(archive["archive_size_bytes"]) is not int or archive[
        "archive_size_bytes"
    ] < 1:
        _fail("repository archive size drifted")
    coordinate = archive["coordinate"]
    if type(coordinate) is not dict:
        _fail("repository archive coordinate drifted")
    try:
        validated_coordinate = validate_campaign_artifact_coordinate(coordinate)
        validate_repository_archive_manifest_key(
            validated_coordinate["key"],
            activation_id=activation_id,
        )
    except ValueError as error:
        raise CleanRehearsalEvidenceError(str(error)) from error
    payload = archive["archive_payload"]
    if (
        validated_coordinate["artifact_kind"] != "REPOSITORY_ARCHIVE"
        or validated_coordinate["file_sha256"]
        != archive["manifest_file_sha256"]
        or validated_coordinate["body_sha256"]
        != archive["manifest_identity_sha256"]
        or type(payload) is not dict
        or set(payload) != {"bucket", "key", "version_id", "file_sha256"}
        or payload["bucket"] != CAMPAIGN_BUCKET
        or payload["file_sha256"] != archive["archive_file_sha256"]
        or payload["key"]
        != (
            "campaigns/%s/repository/keep-%s.tar.gz"
            % (RUN_ID, archive["archive_file_sha256"])
        )
    ):
        _fail("repository archive evidence drifted")
    _version(payload["version_id"], "repository archive payload VersionId")
    expected_manifest_body = {
        "schema_version": 2,
        "record_type": "glm52_task13_repository_archive_manifest_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "archive_file_sha256": archive["archive_file_sha256"],
        "archive_size_bytes": archive["archive_size_bytes"],
        "archive": payload,
    }
    expected_manifest_identity = canonical_sha256(expected_manifest_body)
    expected_manifest = {
        **expected_manifest_body,
        "canonical_identity_sha256": expected_manifest_identity,
    }
    expected_manifest_file_sha = hashlib.sha256(
        canonical_json_bytes(expected_manifest) + b"\n"
    ).hexdigest()
    if (
        archive["manifest_identity_sha256"] != expected_manifest_identity
        or archive["manifest_file_sha256"] != expected_manifest_file_sha
    ):
        _fail("repository archive manifest identity drifted")

    for field in (
        "payload_identity_sha256",
        "payload_file_sha256",
        "measurements_identity_sha256",
    ):
        _sha(finalizer[field], "finalizer " + field)
    if (
        type(finalizer["collector_function_version_arn"]) is not str
        or type(finalizer["collector_function_version"]) is not str
        or
        _COLLECTOR_ARN.fullmatch(
            finalizer["collector_function_version_arn"]
        )
        is None
        or not finalizer["collector_function_version_arn"].endswith(
            ":" + finalizer["collector_function_version"]
        )
        or finalizer["status"] != "CLOSURE_BUDGET_PROVEN"
        or finalizer["measurement_count"] != 20
        or finalizer["cold_environment_count"] != 5
    ):
        _fail("finalizer evidence drifted")
    if (
        gate["bucket"] != REHEARSAL_BUCKET
        or gate["key"]
        != "rehearsal/gates/%s/CLOSURE_BUDGET.json" % activation_id
        or gate["measurement_count"] != 20
        or gate["cold_environment_count"] != 5
        or gate["measurement_count"] != finalizer["measurement_count"]
        or gate["cold_environment_count"]
        != finalizer["cold_environment_count"]
        or gate["measurements_identity_sha256"]
        != finalizer["measurements_identity_sha256"]
        or gate["canonical_body_sha256"] != gate["body_sha256"]
    ):
        _fail("immutable gate evidence drifted")
    _version(gate["version_id"], "immutable gate VersionId")
    for field in (
        "file_sha256",
        "body_sha256",
        "deployment_identity_sha256",
        "canonical_body_sha256",
        "measurements_identity_sha256",
    ):
        _sha(gate[field], "immutable gate " + field)
    _checksum(gate["checksum_sha256_base64"])
    return _copy_json(evidence, "clean rehearsal evidence")


def canonical_clean_rehearsal_evidence_bytes(evidence: object) -> bytes:
    """Return validated canonical evidence bytes plus one LF."""

    return canonical_json_bytes(validate_clean_rehearsal_evidence(evidence)) + b"\n"


def clean_rehearsal_artifact_key(evidence: object) -> str:
    """Derive the activation-scoped key from the evidence self identity."""

    validated = validate_clean_rehearsal_evidence(evidence)
    return "%s%s/%s.json" % (
        CLEAN_REHEARSAL_KEY_PREFIX,
        validated["activation_id"],
        validated["canonical_identity_sha256"],
    )


def validate_clean_rehearsal_key(
    key: object,
    *,
    activation_id: Optional[str] = None,
    evidence_identity_sha256: Optional[str] = None,
) -> str:
    """Validate one exact activation/self-hash evidence key."""
    if type(key) is not str:
        _fail("clean rehearsal key is not exact")
    match = _CLEAN_KEY.fullmatch(key)
    if match is None:
        _fail("clean rehearsal key is not exact")
    if activation_id is not None and match.group("activation") != activation_id:
        _fail("clean rehearsal key activation drifted")
    if (
        evidence_identity_sha256 is not None
        and match.group("identity") != evidence_identity_sha256
    ):
        _fail("clean rehearsal key evidence identity drifted")
    return key


def clean_rehearsal_coordinate(
    *,
    evidence: object,
    version_id: object,
) -> dict[str, object]:
    """Build the reviewed six-field coordinate for published evidence bytes."""

    validated = validate_clean_rehearsal_evidence(evidence)
    raw = canonical_json_bytes(validated) + b"\n"
    return {
        "artifact_kind": "CLEAN_REHEARSAL",
        "bucket": CAMPAIGN_BUCKET,
        "key": clean_rehearsal_artifact_key(validated),
        "version_id": _version(version_id, "clean rehearsal VersionId"),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": hashlib.sha256(raw[:-1]).hexdigest(),
    }


def validate_clean_rehearsal_coordinate(
    coordinate: object,
    *,
    activation_id: Optional[str] = None,
    evidence: object = None,
) -> dict[str, object]:
    """Validate a clean evidence coordinate, optionally against exact evidence."""

    if type(coordinate) is not dict or set(coordinate) != _COORDINATE_FIELDS:
        _fail("clean rehearsal coordinate schema drifted")
    if (
        coordinate.get("artifact_kind") != "CLEAN_REHEARSAL"
        or coordinate.get("bucket") != CAMPAIGN_BUCKET
    ):
        _fail("clean rehearsal coordinate route drifted")
    _version(coordinate.get("version_id"), "clean rehearsal VersionId")
    _sha(coordinate.get("file_sha256"), "clean rehearsal file SHA-256")
    _sha(coordinate.get("body_sha256"), "clean rehearsal body SHA-256")
    if evidence is None:
        validate_clean_rehearsal_key(
            coordinate.get("key"),
            activation_id=activation_id,
        )
    else:
        validated = validate_clean_rehearsal_evidence(evidence)
        expected = clean_rehearsal_coordinate(
            evidence=validated,
            version_id=coordinate["version_id"],
        )
        if activation_id is not None and validated["activation_id"] != activation_id:
            _fail("clean rehearsal evidence activation drifted")
        if coordinate != expected:
            _fail("clean rehearsal coordinate evidence identity drifted")
    return _copy_json(coordinate, "clean rehearsal coordinate")


__all__ = [
    "CLEAN_REHEARSAL_KEY_PREFIX",
    "CleanRehearsalEvidenceError",
    "RECORD_TYPE",
    "REHEARSAL_BUCKET",
    "build_clean_rehearsal_evidence",
    "canonical_clean_rehearsal_evidence_bytes",
    "clean_rehearsal_artifact_key",
    "clean_rehearsal_coordinate",
    "validate_clean_rehearsal_coordinate",
    "validate_clean_rehearsal_evidence",
    "validate_clean_rehearsal_key",
]
