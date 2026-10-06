from __future__ import annotations

from copy import deepcopy
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import traceback
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256


KMS_KEY = (
    "arn:aws:kms:us-west-2:246813579024:key/"
    "12345678-1234-4234-8234-1234567890ab"
)
OTHER_KMS_KEY = (
    "arn:aws:kms:us-west-2:246813579024:key/"
    "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
)
PLAINTEXT = b"n" * 32
NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


def _authority() -> dict[str, object]:
    return {
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": "activation-1",
        "authority_domain": "SNAPSHOT_CLEANUP",
        "owner_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:activation-1"
        ),
        "owner_state_machine_version_arn": (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained:7"
        ),
        "owner_attempt": 1,
        "barrier_nonce_sha256": "a" * 64,
        "control_revision": 3,
        "owner_hard_expires_at": "2099-07-29T13:00:00Z",
    }


def _metadata(
    request_id: str,
    *,
    status: int = 200,
    retries: int | None = 0,
) -> dict[str, object]:
    value: dict[str, object] = {
        "HTTPStatusCode": status,
        "RequestId": request_id,
    }
    if retries is not None:
        value["RetryAttempts"] = retries
    return value


class _Kms:
    def __init__(self, *, key_id: str = KMS_KEY) -> None:
        self.key_id = key_id
        self.plaintext = PLAINTEXT
        self.ciphertext = b"kms-ciphertext"
        self.generate_metadata = _metadata("generate-1")
        self.decrypt_metadata = _metadata("decrypt-1")
        self.decrypt_algorithm = "SYMMETRIC_DEFAULT"
        self.generate_calls: list[dict[str, object]] = []
        self.decrypt_calls: list[dict[str, object]] = []

    def generate_data_key(self, **kwargs: object) -> dict[str, object]:
        self.generate_calls.append(dict(kwargs))
        return {
            "KeyId": self.key_id,
            "Plaintext": self.plaintext,
            "CiphertextBlob": self.ciphertext,
            "ResponseMetadata": self.generate_metadata,
        }

    def decrypt(self, **kwargs: object) -> dict[str, object]:
        self.decrypt_calls.append(dict(kwargs))
        return {
            "KeyId": self.key_id,
            "Plaintext": self.plaintext,
            "EncryptionAlgorithm": self.decrypt_algorithm,
            "ResponseMetadata": self.decrypt_metadata,
        }


class _Ports:
    def __init__(self, kms: _Kms, *, key_id: str = KMS_KEY) -> None:
        self.kms = kms
        self.deployment = SimpleNamespace(
            role_coordinates={"kms_key_id": key_id}
        )

    def client(self, service: str) -> object:
        assert service == "kms"
        return self.kms


def _generate(
    *, kms: _Kms | None = None, authority: dict[str, object] | None = None
) -> tuple[_Kms, dict[str, object], bytes]:
    from glm52_enforcement.task12_nonce_capsule import (
        generate_owner_nonce_capsule,
    )

    exact_kms = _Kms() if kms is None else kms
    capsule, plaintext = generate_owner_nonce_capsule(
        ports=_Ports(exact_kms),
        authority=_authority() if authority is None else authority,
        now=NOW,
    )
    return exact_kms, capsule, plaintext


def _rehash_capsule(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("canonical_body_sha256", None)
    value["canonical_body_sha256"] = canonical_sha256(body)
    return value


def _assert_error_has_no_raw_nonce(exc: BaseException) -> None:
    rendered = repr(exc) + str(exc)
    assert PLAINTEXT.decode("ascii") not in rendered
    assert PLAINTEXT.hex() not in rendered
    assert base64.b64encode(PLAINTEXT).decode("ascii") not in rendered


def test_capsule_roundtrip_is_zero_retry_and_has_no_serialized_plaintext() -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        decrypt_owner_nonce_capsule,
    )

    kms, capsule, plaintext = _generate()

    assert plaintext == PLAINTEXT
    assert capsule["nonce_sha256"] == hashlib.sha256(PLAINTEXT).hexdigest()
    serialized = json.dumps(capsule, sort_keys=True)
    assert PLAINTEXT.decode("ascii") not in serialized
    assert PLAINTEXT.hex() not in serialized
    assert base64.b64encode(PLAINTEXT).decode("ascii") not in serialized
    assert kms.generate_calls == [
        {
            "KeyId": KMS_KEY,
            "KeySpec": "AES_256",
            "EncryptionContext": {
                key: str(_authority()[key])
                for key in sorted(_authority())
            },
        }
    ]

    assert decrypt_owner_nonce_capsule(
        ports=_Ports(kms),
        capsule=capsule,
        expected_authority=_authority(),
        now=NOW,
    ) == PLAINTEXT
    assert kms.decrypt_calls == [
        {
            "CiphertextBlob": b"kms-ciphertext",
            "EncryptionContext": {
                key: str(_authority()[key])
                for key in sorted(_authority())
            },
            "KeyId": KMS_KEY,
            "EncryptionAlgorithm": "SYMMETRIC_DEFAULT",
        }
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("account_id", "000000000000"),
        ("region", "us-east-1"),
        ("run_id", "glm52-foreign"),
        ("activation_id", "activation-2"),
        ("authority_domain", "RECOVERY"),
        (
            "owner_execution_arn",
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:foreign",
        ),
        (
            "owner_state_machine_version_arn",
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained:8",
        ),
        ("owner_attempt", 2),
        ("barrier_nonce_sha256", "b" * 64),
        ("control_revision", 4),
        ("owner_hard_expires_at", "2099-07-29T13:00:01Z"),
    ],
)
def test_capsule_rejects_foreign_or_replayed_owner_tuple_before_kms(
    field: str, replacement: object
) -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        decrypt_owner_nonce_capsule,
    )

    kms, capsule, _plaintext = _generate()
    drifted = {**_authority(), field: replacement}

    with pytest.raises(ValueError, match="identity drifted") as observed:
        decrypt_owner_nonce_capsule(
            ports=_Ports(kms),
            capsule=capsule,
            expected_authority=drifted,
            now=NOW,
        )
    _assert_error_has_no_raw_nonce(observed.value)
    assert kms.decrypt_calls == []


@pytest.mark.parametrize(
    "authority",
    [
        {
            **_authority(),
            "owner_hard_expires_at": "2026-07-29T12:00:00Z",
        },
        {
            **_authority(),
            "owner_execution_arn": "foreign-execution",
        },
        {
            **_authority(),
            "owner_execution_arn": (
                "arn:aws:states:us-east-1:246813579024:execution:"
                "keep-glm52-h1g-retained:activation-1"
            ),
        },
        {
            **_authority(),
            "owner_execution_arn": (
                "arn:aws:states:us-west-2:000000000000:execution:"
                "keep-glm52-h1g-retained:activation-1"
            ),
        },
        {
            **_authority(),
            "owner_state_machine_version_arn": "owner-version:7",
        },
        {
            **_authority(),
            "owner_state_machine_version_arn": (
                "arn:aws:states:us-west-2:246813579024:stateMachine:"
                "keep-glm52-h1g-retained:alias"
            ),
        },
    ],
)
def test_generate_rejects_stale_or_unqualified_owner_authority_before_kms(
    authority: dict[str, object],
) -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        generate_owner_nonce_capsule,
    )

    kms = _Kms()
    with pytest.raises(ValueError) as observed:
        generate_owner_nonce_capsule(
            ports=_Ports(kms), authority=authority, now=NOW
        )
    _assert_error_has_no_raw_nonce(observed.value)
    assert kms.generate_calls == []


def test_expiry_and_reference_clock_are_fail_closed_and_deterministic() -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        decrypt_owner_nonce_capsule,
        generate_owner_nonce_capsule,
    )

    authority = {
        **_authority(),
        "owner_hard_expires_at": "2026-07-29T12:00:00Z",
    }
    kms = _Kms()
    capsule, _plaintext = generate_owner_nonce_capsule(
        ports=_Ports(kms),
        authority=authority,
        now=NOW - timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="expiry is stale") as observed:
        decrypt_owner_nonce_capsule(
            ports=_Ports(kms),
            capsule=capsule,
            expected_authority=authority,
            now=NOW,
        )
    _assert_error_has_no_raw_nonce(observed.value)
    assert kms.decrypt_calls == []

    with pytest.raises(ValueError, match="timezone-aware"):
        generate_owner_nonce_capsule(
            ports=_Ports(_Kms()),
            authority=_authority(),
            now=datetime(2026, 7, 29, 12, 0),
        )


def test_generate_rejects_kms_key_and_plaintext_identity_drift() -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        generate_owner_nonce_capsule,
    )

    wrong_key = _Kms(key_id=OTHER_KMS_KEY)
    with pytest.raises(ValueError, match="result drifted") as observed:
        generate_owner_nonce_capsule(
            ports=_Ports(wrong_key),
            authority=_authority(),
            now=NOW,
        )
    _assert_error_has_no_raw_nonce(observed.value)

    wrong_size = _Kms()
    wrong_size.plaintext = b"short"
    with pytest.raises(ValueError, match="result drifted") as observed:
        generate_owner_nonce_capsule(
            ports=_Ports(wrong_size),
            authority=_authority(),
            now=NOW,
        )
    _assert_error_has_no_raw_nonce(observed.value)

    cross_region_key = KMS_KEY.replace("us-west-2", "us-east-1")
    absent_coordinate = _Kms()
    with pytest.raises(ValueError, match="coordinate is absent") as observed:
        generate_owner_nonce_capsule(
            ports=_Ports(absent_coordinate, key_id=cross_region_key),
            authority=_authority(),
            now=NOW,
        )
    _assert_error_has_no_raw_nonce(observed.value)
    assert absent_coordinate.generate_calls == []


@pytest.mark.parametrize(
    "ciphertext",
    [
        PLAINTEXT,
        b"prefix-" + PLAINTEXT + b"-suffix",
        b"x" * 6145,
    ],
)
def test_generate_rejects_unsafe_kms_ciphertext_without_serializing_raw_nonce(
    ciphertext: bytes,
) -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        generate_owner_nonce_capsule,
    )

    kms = _Kms()
    kms.ciphertext = ciphertext

    with pytest.raises(ValueError) as observed:
        generate_owner_nonce_capsule(
            ports=_Ports(kms), authority=_authority(), now=NOW
        )
    _assert_error_has_no_raw_nonce(observed.value)


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata("generate", retries=1),
        _metadata("generate", retries=None),
        _metadata("", retries=0),
        _metadata("generate", status=500),
    ],
)
def test_generate_rejects_unauthenticated_or_retried_metadata(
    metadata: dict[str, object],
) -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        generate_owner_nonce_capsule,
    )

    kms = _Kms()
    kms.generate_metadata = metadata
    with pytest.raises(ValueError, match="zero-retry") as observed:
        generate_owner_nonce_capsule(
            ports=_Ports(kms), authority=_authority(), now=NOW
        )
    _assert_error_has_no_raw_nonce(observed.value)


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata("decrypt", retries=1),
        _metadata("decrypt", retries=None),
        _metadata("", retries=0),
        _metadata("decrypt", status=500),
    ],
)
def test_decrypt_rejects_unauthenticated_or_retried_metadata(
    metadata: dict[str, object],
) -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        decrypt_owner_nonce_capsule,
    )

    kms, capsule, _plaintext = _generate()
    kms.decrypt_metadata = metadata
    with pytest.raises(ValueError, match="zero-retry") as observed:
        decrypt_owner_nonce_capsule(
            ports=_Ports(kms),
            capsule=capsule,
            expected_authority=_authority(),
            now=NOW,
        )
    _assert_error_has_no_raw_nonce(observed.value)


def test_kms_boundary_errors_do_not_expose_raw_nonce() -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        decrypt_owner_nonce_capsule,
        generate_owner_nonce_capsule,
    )

    class _LeakyKms(_Kms):
        def generate_data_key(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError(PLAINTEXT.hex())

        def decrypt(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError(base64.b64encode(PLAINTEXT).decode("ascii"))

    observed_errors = []
    with pytest.raises(ValueError) as observed:
        generate_owner_nonce_capsule(
            ports=_Ports(_LeakyKms()),
            authority=_authority(),
            now=NOW,
        )
    observed_errors.append(observed.value)

    _original_kms, capsule, _plaintext = _generate()
    with pytest.raises(ValueError) as observed:
        decrypt_owner_nonce_capsule(
            ports=_Ports(_LeakyKms()),
            capsule=capsule,
            expected_authority=_authority(),
            now=NOW,
        )
    observed_errors.append(observed.value)

    for error in observed_errors:
        rendered = "".join(
            traceback.format_exception(
                type(error),
                error,
                error.__traceback__,
            )
        )
        _assert_error_has_no_raw_nonce(error)
        assert PLAINTEXT.hex() not in rendered
        assert base64.b64encode(PLAINTEXT).decode("ascii") not in rendered
        assert error.__cause__ is None
        assert error.__suppress_context__ is True


def test_decrypt_rejects_wrong_key_algorithm_plaintext_and_nonce_hash() -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        decrypt_owner_nonce_capsule,
    )

    _kms, capsule, _plaintext = _generate()

    wrong_key = _Kms(key_id=OTHER_KMS_KEY)
    with pytest.raises(ValueError, match="KMS key drifted"):
        decrypt_owner_nonce_capsule(
            ports=_Ports(wrong_key, key_id=OTHER_KMS_KEY),
            capsule=capsule,
            expected_authority=_authority(),
            now=NOW,
        )
    assert wrong_key.decrypt_calls == []

    wrong_algorithm = _Kms()
    wrong_algorithm.decrypt_algorithm = "RSAES_OAEP_SHA_256"
    with pytest.raises(ValueError, match="plaintext identity drifted"):
        decrypt_owner_nonce_capsule(
            ports=_Ports(wrong_algorithm),
            capsule=capsule,
            expected_authority=_authority(),
            now=NOW,
        )

    wrong_response_key = _Kms(key_id=OTHER_KMS_KEY)
    with pytest.raises(ValueError, match="plaintext identity drifted"):
        decrypt_owner_nonce_capsule(
            ports=_Ports(wrong_response_key),
            capsule=capsule,
            expected_authority=_authority(),
            now=NOW,
        )

    wrong_size = _Kms()
    wrong_size.plaintext = b"short"
    with pytest.raises(ValueError, match="plaintext identity drifted"):
        decrypt_owner_nonce_capsule(
            ports=_Ports(wrong_size),
            capsule=capsule,
            expected_authority=_authority(),
            now=NOW,
        )

    wrong_hash = deepcopy(capsule)
    wrong_hash["nonce_sha256"] = "f" * 64
    _rehash_capsule(wrong_hash)
    with pytest.raises(ValueError, match="plaintext identity drifted"):
        decrypt_owner_nonce_capsule(
            ports=_Ports(_Kms()),
            capsule=wrong_hash,
            expected_authority=_authority(),
            now=NOW,
        )


@pytest.mark.parametrize(
    "ciphertext_base64",
    [
        "",
        "not base64!",
        base64.b64encode(b"kms-ciphertext").decode("ascii") + "\n",
    ],
)
def test_decrypt_rejects_empty_malformed_or_noncanonical_ciphertext_before_kms(
    ciphertext_base64: str,
) -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        decrypt_owner_nonce_capsule,
    )

    kms, capsule, _plaintext = _generate()
    drifted = deepcopy(capsule)
    drifted["ciphertext_base64"] = ciphertext_base64
    _rehash_capsule(drifted)

    with pytest.raises(ValueError, match="ciphertext") as observed:
        decrypt_owner_nonce_capsule(
            ports=_Ports(kms),
            capsule=drifted,
            expected_authority=_authority(),
            now=NOW,
        )
    _assert_error_has_no_raw_nonce(observed.value)
    assert kms.decrypt_calls == []


@pytest.mark.parametrize(
    "ciphertext",
    [
        PLAINTEXT,
        b"prefix-" + PLAINTEXT + b"-suffix",
        b"x" * 6145,
    ],
)
def test_decrypt_rejects_unsafe_ciphertext_without_exposing_raw_nonce(
    ciphertext: bytes,
) -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        decrypt_owner_nonce_capsule,
    )

    kms, capsule, _plaintext = _generate()
    unsafe = deepcopy(capsule)
    unsafe["ciphertext_base64"] = base64.b64encode(ciphertext).decode("ascii")
    _rehash_capsule(unsafe)

    with pytest.raises(ValueError, match="ciphertext") as observed:
        decrypt_owner_nonce_capsule(
            ports=_Ports(kms),
            capsule=unsafe,
            expected_authority=_authority(),
            now=NOW,
        )
    _assert_error_has_no_raw_nonce(observed.value)
    if len(ciphertext) > 6144:
        assert kms.decrypt_calls == []


def test_capsule_rejects_wrong_context_self_hash_and_extra_raw_field() -> None:
    from glm52_enforcement.task12_nonce_capsule import (
        decrypt_owner_nonce_capsule,
    )

    kms, capsule, _plaintext = _generate()
    cases = []
    wrong_context = deepcopy(capsule)
    wrong_context["encryption_context"]["control_revision"] = "4"
    _rehash_capsule(wrong_context)
    cases.append(wrong_context)
    wrong_hash = deepcopy(capsule)
    wrong_hash["canonical_body_sha256"] = "0" * 64
    cases.append(wrong_hash)
    raw_field = {**capsule, "raw_owner_nonce_hex": PLAINTEXT.hex()}
    cases.append(raw_field)

    for candidate in cases:
        with pytest.raises(ValueError) as observed:
            decrypt_owner_nonce_capsule(
                ports=_Ports(kms),
                capsule=candidate,
                expected_authority=_authority(),
                now=NOW,
            )
        _assert_error_has_no_raw_nonce(observed.value)
    assert kms.decrypt_calls == []
