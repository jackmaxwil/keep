"""KMS-bound owner nonce continuation without serialized plaintext.

The plaintext data key exists only inside one Lambda invocation.  Task 12
requests and Step Functions history carry only the authenticated ciphertext
capsule.  Decryption is bound to the exact retained owner coordinate through
the KMS encryption context and a closed canonical capsule schema.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import re
from typing import Mapping

from .canonical import canonical_sha256


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_KMS_ARN = re.compile(
    r"^arn:aws:kms:us-west-2:246813579024:key/"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_EXECUTION_ARN = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:execution:"
    r"[A-Za-z0-9_+=,.@-]+:[A-Za-z0-9_+=,.@-]+$"
)
_VERSION_ARN = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:stateMachine:"
    r"[A-Za-z0-9_+=,.@-]+:[1-9][0-9]*$"
)
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_MAX_CIPHERTEXT_BYTES = 6144
_AUTHORITY_FIELDS = frozenset(
    {
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "authority_domain",
        "owner_execution_arn",
        "owner_state_machine_version_arn",
        "owner_attempt",
        "barrier_nonce_sha256",
        "control_revision",
        "owner_hard_expires_at",
    }
)
_CAPSULE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "kms_key_id",
        "encryption_context",
        "ciphertext_base64",
        "nonce_sha256",
        "canonical_body_sha256",
    }
)


class Task12NonceCapsuleError(ValueError):
    """The encrypted nonce continuation is absent, foreign, or unauthenticated."""


def _fail(message: str) -> None:
    raise Task12NonceCapsuleError(message)


def _metadata(response: object, label: str) -> None:
    metadata = response.get("ResponseMetadata") if type(response) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        _fail(label + " response is not authenticated zero-retry")


def _reference_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if (
        type(now) is not datetime
        or now.tzinfo is None
        or now.utcoffset() is None
    ):
        _fail("owner nonce reference time is not timezone-aware")
    return now.astimezone(timezone.utc)


def _authority(
    value: object,
    *,
    now: datetime | None,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _AUTHORITY_FIELDS:
        _fail("owner nonce authority fields are not closed")
    exact = dict(value)
    if (
        exact["account_id"] != ACCOUNT_ID
        or exact["region"] != REGION
        or exact["run_id"] != RUN_ID
        or type(exact["activation_id"]) is not str
        or not exact["activation_id"]
        or exact["authority_domain"]
        not in {"RECOVERY", "TEARDOWN", "FINALIZATION", "SNAPSHOT_CLEANUP"}
        or type(exact["owner_execution_arn"]) is not str
        or _EXECUTION_ARN.fullmatch(exact["owner_execution_arn"]) is None
        or type(exact["owner_state_machine_version_arn"]) is not str
        or _VERSION_ARN.fullmatch(exact["owner_state_machine_version_arn"])
        is None
        or type(exact["owner_attempt"]) is not int
        or isinstance(exact["owner_attempt"], bool)
        or exact["owner_attempt"] < 1
        or type(exact["barrier_nonce_sha256"]) is not str
        or _SHA.fullmatch(exact["barrier_nonce_sha256"]) is None
        or type(exact["control_revision"]) is not int
        or isinstance(exact["control_revision"], bool)
        or exact["control_revision"] < 1
        or type(exact["owner_hard_expires_at"]) is not str
        or _UTC_TIMESTAMP.fullmatch(exact["owner_hard_expires_at"]) is None
    ):
        _fail("owner nonce authority identity drifted")
    try:
        hard_expiry = datetime.strptime(
            exact["owner_hard_expires_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise Task12NonceCapsuleError(
            "owner nonce authority expiry is not canonical UTC"
        ) from exc
    if hard_expiry <= _reference_now(now):
        _fail("owner nonce authority expiry is stale")
    return exact


def _context(authority: Mapping[str, object]) -> dict[str, str]:
    return {key: str(authority[key]) for key in sorted(_AUTHORITY_FIELDS)}


def _kms(ports: object) -> tuple[object, str]:
    deployment = getattr(ports, "deployment", None)
    roles = getattr(deployment, "role_coordinates", None)
    key_id = roles.get("kms_key_id") if isinstance(roles, Mapping) else None
    if type(key_id) is not str or _KMS_ARN.fullmatch(key_id) is None:
        _fail("owner nonce KMS key coordinate is absent")
    client = getattr(ports, "client", None)
    if not callable(client):
        _fail("owner nonce KMS boundary is absent")
    return client("kms"), key_id


def generate_owner_nonce_capsule(
    *,
    ports: object,
    authority: Mapping[str, object],
    now: datetime | None = None,
) -> tuple[dict[str, object], bytes]:
    """Generate one AES-256 data key and return ciphertext plus ephemeral bytes."""

    exact_authority = _authority(authority, now=now)
    encryption_context = _context(exact_authority)
    client, key_id = _kms(ports)
    try:
        response = client.generate_data_key(
            KeyId=key_id,
            KeySpec="AES_256",
            EncryptionContext=encryption_context,
        )
    except Exception:
        raise Task12NonceCapsuleError(
            "KMS GenerateDataKey request failed"
        ) from None
    _metadata(response, "KMS GenerateDataKey")
    plaintext = response.get("Plaintext")
    ciphertext = response.get("CiphertextBlob")
    if (
        response.get("KeyId") != key_id
        or type(plaintext) is not bytes
        or len(plaintext) != 32
        or type(ciphertext) is not bytes
        or not ciphertext
        or len(ciphertext) > _MAX_CIPHERTEXT_BYTES
        or plaintext in ciphertext
    ):
        _fail("KMS GenerateDataKey result drifted")
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_owner_nonce_capsule_v1",
        "kms_key_id": key_id,
        "encryption_context": encryption_context,
        "ciphertext_base64": base64.b64encode(ciphertext).decode("ascii"),
        "nonce_sha256": hashlib.sha256(plaintext).hexdigest(),
    }
    return (
        {**body, "canonical_body_sha256": canonical_sha256(body)},
        plaintext,
    )


def decrypt_owner_nonce_capsule(
    *,
    ports: object,
    capsule: Mapping[str, object],
    expected_authority: Mapping[str, object],
    now: datetime | None = None,
) -> bytes:
    """Decrypt only an exact capsule bound to the current retained owner."""

    exact_authority = _authority(expected_authority, now=now)
    if type(capsule) is not dict or set(capsule) != _CAPSULE_FIELDS:
        _fail("owner nonce capsule fields are not closed")
    body = dict(capsule)
    supplied_hash = body.pop("canonical_body_sha256")
    if (
        capsule.get("schema_version") != 1
        or capsule.get("record_type")
        != "glm52_task12_owner_nonce_capsule_v1"
        or capsule.get("encryption_context") != _context(exact_authority)
        or type(supplied_hash) is not str
        or supplied_hash != canonical_sha256(body)
        or type(capsule.get("nonce_sha256")) is not str
        or _SHA.fullmatch(capsule["nonce_sha256"]) is None
    ):
        _fail("owner nonce capsule identity drifted")
    client, key_id = _kms(ports)
    if capsule.get("kms_key_id") != key_id:
        _fail("owner nonce capsule KMS key drifted")
    encoded_ciphertext = capsule.get("ciphertext_base64")
    if type(encoded_ciphertext) is not str or not encoded_ciphertext:
        _fail("owner nonce capsule ciphertext is malformed")
    try:
        ciphertext = base64.b64decode(encoded_ciphertext, validate=True)
    except (TypeError, ValueError) as exc:
        raise Task12NonceCapsuleError(
            "owner nonce capsule ciphertext is malformed"
        ) from exc
    if (
        not ciphertext
        or len(ciphertext) > _MAX_CIPHERTEXT_BYTES
        or base64.b64encode(ciphertext).decode("ascii") != encoded_ciphertext
    ):
        _fail("owner nonce capsule ciphertext is malformed")
    try:
        response = client.decrypt(
            CiphertextBlob=ciphertext,
            EncryptionContext=_context(exact_authority),
            KeyId=key_id,
            EncryptionAlgorithm="SYMMETRIC_DEFAULT",
        )
    except Exception:
        raise Task12NonceCapsuleError("KMS Decrypt request failed") from None
    _metadata(response, "KMS Decrypt")
    plaintext = response.get("Plaintext")
    algorithm = response.get("EncryptionAlgorithm")
    if (
        response.get("KeyId") != key_id
        or (algorithm is not None and algorithm != "SYMMETRIC_DEFAULT")
        or type(plaintext) is not bytes
        or len(plaintext) != 32
        or hashlib.sha256(plaintext).hexdigest()
        != capsule["nonce_sha256"]
    ):
        _fail("owner nonce capsule plaintext identity drifted")
    if plaintext in ciphertext:
        _fail("owner nonce capsule ciphertext is unsafe")
    return plaintext


__all__ = [
    "Task12NonceCapsuleError",
    "decrypt_owner_nonce_capsule",
    "generate_owner_nonce_capsule",
]
