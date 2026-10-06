#!/usr/bin/env python3
"""Publish and replay qualification-only dynamic-v2 worker authority.

This pre-repository entrypoint intentionally depends on the Python standard
library plus a source-pinned AWS CLI installation.  Scientific repository
artifacts and GPU paths are outside this module's authority.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import datetime as _datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Literal
from urllib.parse import urlsplit


APPROVED_ACCOUNT_ID = "246813579024"
APPROVED_REGION = "us-west-2"
APPROVED_INSTANCE_TYPE = "p5.48xlarge"
QUALIFICATION_MODE = "qualification"
AWS_ENTRYPOINT = Path("/usr/local/bin/aws")
AWS_INSTALL_PREFIX = Path("/usr/local/aws-cli/v2")
AWS_SUBPROCESS_TIMEOUT_SECONDS = 30
IMDS_BASE_URL = "http://169.254.169.254"
IMDS_TIMEOUT_SECONDS = 2.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_ID = re.compile(r"^i-[0-9a-f]{17}$")
_VERSION_ID = re.compile(r"^[A-Za-z0-9._+=:/-]{1,1024}$")
_ETAG = re.compile(r'^"[0-9a-f]{32}"$')
_AWS_VERSION = re.compile(r"^2\.[0-9]+\.[0-9]+$")
_CLI_ERROR = re.compile(
    rb"^An error occurred \(([A-Za-z0-9]+)\) when calling the "
    rb"([A-Za-z0-9]+) operation: [^\r\n]*(?:\n)?$"
)
_FORBIDDEN_LEGACY = (
    "JOB_BINDING.json",
    "TIMELY_START_ACCEPTED.json",
    "TIMELY_START_LATCH.json",
)
_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
_CONFLICT_CODES = frozenset(
    {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}
)
_FROZEN_SOURCE_HASHES = {
    "glm52_sky_campaign_native": (
        "f023eaeddf8fac73fa6546e3c4a0b60dd32e5266f06e1519fe21e0444d445d03"
    ),
    "glm52_sky_must_start_native": (
        "e910d3d03f7b30b7b9b668e214b61ecc7cbd641a53dedabd620b8e14573d33f3"
    ),
    "glm52_sky_campaign": (
        "77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94"
    ),
    "glm52_sky_must_start": (
        "478ac3a16ce8a77a4e844c1e0a3dca99454d50e7233fd7cd67dfa184b486860a"
    ),
    "glm52_sky_must_start_dynamic": (
        "527019a41b54f810e9f98734c2ca99acc5a7b370344adf48c80f46f925396e18"
    ),
    "glm52_sky_worker_must_start_v2": (
        "8dfad6682d8afc9774dc31385cfc9ecc8c8e2f80d9a535e3c8d4cf6cf0fa97df"
    ),
    "sky_worker_start_v2_coordinator": (
        "efa7c48ad259e10c6ce0e4cb1db9ff0a6e6a9fa203274329a7c2eb5203366406"
    ),
}
_SOURCE_ORDER = tuple(_FROZEN_SOURCE_HASHES)
_SANITIZED_AWS_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/root",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "AWS_REGION": APPROVED_REGION,
    "AWS_DEFAULT_REGION": APPROVED_REGION,
    "AWS_PAGER": "",
    "AWS_EC2_METADATA_DISABLED": "false",
    "AWS_METADATA_SERVICE_NUM_ATTEMPTS": "1",
    "AWS_CONFIG_FILE": "/dev/null",
    "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
    "BOTO_CONFIG": "/dev/null",
    "AWS_MAX_ATTEMPTS": "1",
}
_FORBIDDEN_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
        "AWS_CA_BUNDLE",
        "BOTO_CONFIG",
    }
)
_LATCH_TRANSPORT_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "intent_body_sha256",
        "sky_job_id",
        "instance_id",
        "worker_instance_type",
        "worker_image_id",
        "worker_role_arn",
        "ec2_pending_time",
        "instance_identity_document_sha256",
        "worker_latch_key",
        "worker_latch_file_sha256",
        "worker_latch_body_sha256",
        "worker_latch_version_id",
        "worker_latch_etag",
        "worker_latch_last_modified",
        "content_length",
        "checksum_sha256",
        "checksum_type",
        "content_type",
        "metadata",
        "latch_transport_body_sha256",
    }
)
_SOURCE_TRANSPORT_FIELDS = frozenset(
    {
        "key",
        "file_sha256",
        "body_sha256",
        "version_id",
        "etag",
        "last_modified",
        "content_length",
        "checksum_sha256",
        "checksum_type",
        "content_type",
        "metadata",
        "trailing_newline",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "intent_body_sha256",
        "sky_job_id",
        "instance_id",
        "worker_instance_type",
        "worker_image_id",
        "worker_role_arn",
        "ec2_pending_time",
        "instance_identity_document_sha256",
        "worker_latch_key",
        "worker_latch_file_sha256",
        "worker_latch_body_sha256",
        "worker_latch_version_id",
        "worker_latch_etag",
        "worker_latch_last_modified",
        "worker_acceptance_key",
        "worker_acceptance_file_sha256",
        "worker_acceptance_body_sha256",
        "worker_acceptance_version_id",
        "worker_acceptance_etag",
        "worker_acceptance_last_modified",
        "source_transports",
        "legacy_forbidden_basenames",
        "admission_receipt_body_sha256",
    }
)
_TRUSTED_SOURCE_MODULES: dict[
    str,
    tuple[ModuleType, str, tuple[tuple[str, int, int | None, str], ...]],
] = {}


class WorkerStartPublisherError(RuntimeError):
    """Worker-v2 publication, replay, or receipt authority failed closed."""


@dataclass(frozen=True)
class WorkerStartCommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class WorkerStartSourceBundle:
    publisher_file_sha256: str
    campaign_policy_path: Path
    campaign_policy_file_sha256: str
    campaign_policy_native_path: Path
    campaign_policy_native_file_sha256: str
    must_start_policy_path: Path
    must_start_policy_file_sha256: str
    must_start_policy_native_path: Path
    must_start_policy_native_file_sha256: str
    dynamic_policy_path: Path
    dynamic_policy_file_sha256: str
    worker_policy_path: Path
    worker_policy_file_sha256: str
    coordinator_path: Path
    coordinator_file_sha256: str


@dataclass(frozen=True)
class WorkerStartAuthorityInputs:
    descriptor_path: Path
    descriptor_s3_uri: str
    descriptor_file_sha256: str
    intent_path: Path
    intent_s3_uri: str
    intent_file_sha256: str
    intent_body_sha256: str


@dataclass(frozen=True)
class WorkerStartLatchOutputs:
    latch_path: Path
    latch_transport_path: Path


@dataclass(frozen=True)
class WorkerStartAdmissionOutputs:
    accepted_path: Path
    admission_receipt_path: Path


@dataclass(frozen=True)
class WorkerStartPublisherServices:
    command_runner: Callable[
        [tuple[str, ...], Mapping[str, str], int],
        WorkerStartCommandResult,
    ]
    imds_opener: Callable[[urllib.request.Request, float], bytes]
    utc_now: Callable[[], datetime]
    monotonic: Callable[[], float]
    sleep: Callable[[float], None]
    environ: Mapping[str, str]


@dataclass(frozen=True)
class WorkerStartPublisherOutcome:
    status: Literal[
        "latch-published",
        "latch-idempotent",
        "accepted-initial",
        "accepted-managed-recovery",
        "receipt-verified",
    ]
    run_id: str
    instance_id: str
    sky_job_id: int
    worker_latch_key: str
    worker_acceptance_key: str | None
    worker_acceptance_body_sha256: str | None


@dataclass(frozen=True)
class _LoadedSources:
    campaign: ModuleType
    must_start: ModuleType
    dynamic: ModuleType
    worker: ModuleType
    coordinator: ModuleType


@dataclass(frozen=True)
class _LocalAuthority:
    descriptor: dict[str, object]
    intent: dict[str, object]
    bucket: str
    run_id: str
    descriptor_key: str
    intent_key: str


@dataclass(frozen=True)
class _WorkerIdentity:
    account_id: str
    region: str
    instance_id: str
    instance_type: str
    image_id: str
    worker_role_arn: str
    pending_time: str
    identity_document_sha256: str


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    uid: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _AwsCliAttestation:
    executable: Path
    entrypoint_identity: _FileIdentity | None
    target_identity: _FileIdentity | None


@dataclass(frozen=True)
class _Transport:
    version_id: str
    etag: str
    last_modified: str
    content_length: int
    checksum_sha256: str
    checksum_type: str
    content_type: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class _Prepared:
    sources: _LoadedSources
    authority: _LocalAuthority
    job_id: int
    worker: _WorkerIdentity
    cli: _AwsCliAttestation
    sts_identity: dict[str, object]


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": "AWS CLI service error"}}


class _Unavailable(Exception):
    pass


class _AmbiguousTransport(Exception):
    pass


class _NoCalls:
    def __init__(self, label: str) -> None:
        self.label = label

    def __getattr__(self, name: str) -> Callable[..., object]:
        def fail(**_: object) -> object:
            raise WorkerStartPublisherError(
                f"read-only replay attempted forbidden {self.label}.{name}"
            )

        return fail


class _BytesBody:
    def __init__(self, raw: bytes) -> None:
        self._stream = BytesIO(raw)
        self._read = False
        self._closed = False

    def read(self) -> bytes:
        if self._read:
            raise WorkerStartPublisherError("S3 body stream was read twice")
        self._read = True
        return self._stream.read()

    def close(self) -> None:
        if self._closed:
            raise WorkerStartPublisherError("S3 body stream was closed twice")
        self._closed = True
        self._stream.close()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: object, *, newline: bool) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise WorkerStartPublisherError(
            "authority is not canonical finite ASCII JSON"
        ) from error
    return raw + (b"\n" if newline else b"")


def _duplicate_rejecting_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkerStartPublisherError("JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise WorkerStartPublisherError("JSON contains a non-finite number")


def _decode_json(
    raw: bytes,
    *,
    label: str,
    newline: bool | None,
    require_object: bool = True,
) -> dict[str, object]:
    try:
        text = raw.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except WorkerStartPublisherError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise WorkerStartPublisherError(f"{label} is not finite ASCII JSON") from error
    if require_object and (type(value) is not dict or not value):
        raise WorkerStartPublisherError(f"{label} must be a nonempty JSON object")
    assert isinstance(value, dict)
    if newline is not None and raw != _canonical_bytes(value, newline=newline):
        raise WorkerStartPublisherError(f"{label} is not canonical JSON")
    return value


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise WorkerStartPublisherError(f"{label} is not a lowercase SHA-256")
    return value


def _file_identity(value: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
        uid=value.st_uid,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
    )


def _same_directory_identity(
    first: os.stat_result,
    second: os.stat_result,
) -> bool:
    return (
        stat.S_ISDIR(first.st_mode)
        and stat.S_ISDIR(second.st_mode)
        and first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and first.st_mode == second.st_mode
        and first.st_uid == second.st_uid
    )


def _same_persisted_file(
    opened: os.stat_result,
    entry: os.stat_result,
    *,
    expected_size: int,
) -> bool:
    return (
        stat.S_ISREG(opened.st_mode)
        and stat.S_ISREG(entry.st_mode)
        and opened.st_dev == entry.st_dev
        and opened.st_ino == entry.st_ino
        and opened.st_uid == entry.st_uid
        and stat.S_IMODE(opened.st_mode) == 0o600
        and stat.S_IMODE(entry.st_mode) == 0o600
        and opened.st_nlink == 1
        and entry.st_nlink == 1
        and opened.st_size == expected_size
        and entry.st_size == expected_size
    )


def _same_file_observation(
    first: os.stat_result,
    second: os.stat_result,
) -> bool:
    return (
        _file_identity(first) == _file_identity(second)
        and first.st_ctime_ns == second.st_ctime_ns
        and first.st_nlink == second.st_nlink
    )


def _require_regular_path(path: object, *, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise WorkerStartPublisherError(f"{label} must be an absolute Path")
    try:
        details = path.lstat()
    except OSError as error:
        raise WorkerStartPublisherError(f"{label} is missing") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise WorkerStartPublisherError(f"{label} must be a regular non-symlink file")
    return path


def _read_stable_regular(
    path: Path,
    *,
    label: str,
    required_mode: int | None = None,
) -> bytes:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise WorkerStartPublisherError(
                f"{label} must be a regular non-symlink file"
            )
        if required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode:
            raise WorkerStartPublisherError(f"{label} mode is invalid")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise WorkerStartPublisherError(f"{label} changed before being read")
        if required_mode is not None and stat.S_IMODE(opened.st_mode) != required_mode:
            raise WorkerStartPublisherError(f"{label} mode is invalid")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
        after = path.lstat()
    except WorkerStartPublisherError:
        raise
    except OSError as error:
        raise WorkerStartPublisherError(f"{label} cannot be read") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if _file_identity(before) != _file_identity(after_open) or _file_identity(
        before
    ) != _file_identity(after):
        raise WorkerStartPublisherError(f"{label} changed while being read")
    if required_mode is not None and (
        stat.S_IMODE(after_open.st_mode) != required_mode
        or stat.S_IMODE(after.st_mode) != required_mode
    ):
        raise WorkerStartPublisherError(f"{label} mode is invalid")
    return b"".join(chunks)


def _module_path(module: ModuleType) -> Path:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise WorkerStartPublisherError("frozen source module has no __file__")
    try:
        return Path(module_file).resolve(strict=True)
    except OSError as error:
        raise WorkerStartPublisherError("frozen source module path vanished") from error


def _module_runtime_identity(
    module: ModuleType,
) -> tuple[tuple[str, int, int | None, str], ...]:
    entries: list[tuple[str, int, int | None, str]] = []

    def add(name: str, value: object) -> None:
        code_id = id(value.__code__) if isinstance(value, FunctionType) else None
        if isinstance(value, ModuleType):
            representation = value.__name__
        else:
            try:
                representation = repr(value)
            except Exception as error:
                raise WorkerStartPublisherError(
                    "frozen source runtime identity cannot be represented"
                ) from error
        entries.append((name, id(value), code_id, representation))
        if isinstance(value, type):
            for member_name, member in sorted(value.__dict__.items()):
                member_code_id = (
                    id(member.__code__) if isinstance(member, FunctionType) else None
                )
                try:
                    member_repr = repr(member)
                except Exception as error:
                    raise WorkerStartPublisherError(
                        "frozen class runtime identity cannot be represented"
                    ) from error
                entries.append(
                    (
                        f"{name}.{member_name}",
                        id(member),
                        member_code_id,
                        member_repr,
                    )
                )

    for name, value in sorted(module.__dict__.items()):
        if name not in {"__builtins__", "__warningregistry__"}:
            add(name, value)
    return tuple(entries)


def _load_exact_module(
    name: str,
    path: Path,
    digest: str,
    raw: bytes,
) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        if not isinstance(existing, ModuleType):
            raise WorkerStartPublisherError(
                f"foreign preloaded source module blocks {name}"
            )
        if _module_path(existing) != path.resolve(strict=True):
            raise WorkerStartPublisherError(
                f"foreign preloaded source module blocks {name}"
            )
        current_raw = _read_stable_regular(path, label=f"preloaded {name} source")
        if current_raw != raw or _sha256(current_raw) != digest:
            raise WorkerStartPublisherError(f"preloaded {name} source drifted")
        trusted = _TRUSTED_SOURCE_MODULES.get(name)
        if (
            trusted is None
            or trusted[0] is not existing
            or trusted[1] != digest
            or trusted[2] != _module_runtime_identity(existing)
        ):
            raise WorkerStartPublisherError(
                f"preloaded {name} runtime identity is not trusted"
            )
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise WorkerStartPublisherError(f"frozen source {name} cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        code = compile(raw, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception as error:
        sys.modules.pop(name, None)
        raise WorkerStartPublisherError(
            f"frozen source {name} import failed"
        ) from error
    if _module_path(module) != path.resolve(strict=True):
        sys.modules.pop(name, None)
        raise WorkerStartPublisherError(f"frozen source {name} path drifted")
    current_raw = _read_stable_regular(path, label=f"loaded {name} source")
    if current_raw != raw or _sha256(current_raw) != digest:
        sys.modules.pop(name, None)
        raise WorkerStartPublisherError(f"loaded {name} source drifted")
    _TRUSTED_SOURCE_MODULES[name] = (
        module,
        digest,
        _module_runtime_identity(module),
    )
    return module


def _validate_source_bundle(sources: object) -> _LoadedSources:
    if type(sources) is not WorkerStartSourceBundle:
        raise WorkerStartPublisherError("source bundle type is invalid")
    publisher_path = _require_regular_path(
        Path(__file__),
        label="executing publisher",
    )
    publisher_digest = _require_digest(
        sources.publisher_file_sha256,
        label="publisher file SHA-256",
    )
    if (
        _sha256(
            _read_stable_regular(publisher_path, label="executing publisher source")
        )
        != publisher_digest
    ):
        raise WorkerStartPublisherError("publisher source hash drifted")
    entries = (
        (
            "glm52_sky_campaign_native",
            sources.campaign_policy_native_path,
            sources.campaign_policy_native_file_sha256,
        ),
        (
            "glm52_sky_must_start_native",
            sources.must_start_policy_native_path,
            sources.must_start_policy_native_file_sha256,
        ),
        (
            "glm52_sky_campaign",
            sources.campaign_policy_path,
            sources.campaign_policy_file_sha256,
        ),
        (
            "glm52_sky_must_start",
            sources.must_start_policy_path,
            sources.must_start_policy_file_sha256,
        ),
        (
            "glm52_sky_must_start_dynamic",
            sources.dynamic_policy_path,
            sources.dynamic_policy_file_sha256,
        ),
        (
            "glm52_sky_worker_must_start_v2",
            sources.worker_policy_path,
            sources.worker_policy_file_sha256,
        ),
        (
            "sky_worker_start_v2_coordinator",
            sources.coordinator_path,
            sources.coordinator_file_sha256,
        ),
    )
    verified: dict[str, tuple[Path, bytes]] = {}
    for name, raw_path, supplied_digest in entries:
        path = _require_regular_path(raw_path, label=f"{name} source")
        digest = _require_digest(
            supplied_digest,
            label=f"{name} file SHA-256",
        )
        if digest != _FROZEN_SOURCE_HASHES[name]:
            raise WorkerStartPublisherError(f"{name} source identity is not frozen")
        raw = _read_stable_regular(path, label=f"{name} source")
        if _sha256(raw) != digest:
            raise WorkerStartPublisherError(f"{name} source hash drifted")
        verified[name] = (path, raw)
    if not hasattr(_datetime, "UTC"):
        _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]
    if _datetime.UTC != _datetime.timezone.utc:
        raise WorkerStartPublisherError("datetime.UTC compatibility alias drifted")
    loaded = {
        name: _load_exact_module(
            name,
            verified[name][0],
            _FROZEN_SOURCE_HASHES[name],
            verified[name][1],
        )
        for name in _SOURCE_ORDER
    }
    return _LoadedSources(
        campaign=loaded["glm52_sky_campaign"],
        must_start=loaded["glm52_sky_must_start"],
        dynamic=loaded["glm52_sky_must_start_dynamic"],
        worker=loaded["glm52_sky_worker_must_start_v2"],
        coordinator=loaded["sky_worker_start_v2_coordinator"],
    )


def _parse_s3_uri(uri: object, *, label: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.isascii() or "%" in uri:
        raise WorkerStartPublisherError(f"{label} is invalid")
    try:
        parsed = urlsplit(uri)
        has_user = parsed.username is not None
        has_password = parsed.password is not None
        has_port = parsed.port is not None
    except ValueError as error:
        raise WorkerStartPublisherError(f"{label} is invalid") from error
    key = parsed.path[1:] if parsed.path.startswith("/") else ""
    canonical = f"s3://{parsed.netloc}/{key}"
    if (
        uri != canonical
        or parsed.scheme != "s3"
        or not parsed.netloc
        or has_user
        or has_password
        or has_port
        or parsed.query
        or parsed.fragment
        or not key
        or "\\" in key
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise WorkerStartPublisherError(f"{label} is invalid")
    return parsed.netloc, key


def _load_local_authority(
    authority: object,
    loaded: _LoadedSources,
) -> _LocalAuthority:
    if type(authority) is not WorkerStartAuthorityInputs:
        raise WorkerStartPublisherError("authority input type is invalid")
    descriptor_path = _require_regular_path(
        authority.descriptor_path,
        label="descriptor",
    )
    intent_path = _require_regular_path(authority.intent_path, label="intent")
    descriptor_sha = _require_digest(
        authority.descriptor_file_sha256,
        label="descriptor file SHA-256",
    )
    intent_sha = _require_digest(
        authority.intent_file_sha256,
        label="intent file SHA-256",
    )
    intent_body_sha = _require_digest(
        authority.intent_body_sha256,
        label="intent body SHA-256",
    )
    descriptor_raw = _read_stable_regular(descriptor_path, label="descriptor")
    intent_raw = _read_stable_regular(intent_path, label="intent")
    if _sha256(descriptor_raw) != descriptor_sha:
        raise WorkerStartPublisherError("descriptor file SHA-256 drifted")
    if _sha256(intent_raw) != intent_sha:
        raise WorkerStartPublisherError("intent file SHA-256 drifted")
    descriptor = _decode_json(
        descriptor_raw,
        label="descriptor",
        newline=True,
    )
    intent = _decode_json(intent_raw, label="intent", newline=True)
    try:
        descriptor = loaded.campaign.validate_sky_campaign_descriptor(descriptor)
        # Task 3m performs the complete descriptor-intent cross-validation.
        probe = loaded.worker.build_worker_start_latch_v2(
            descriptor=descriptor,
            intent=intent,
            controller_injected_sky_job_id=1,
            instance_id="i-00000000000000000",
            instance_type=APPROVED_INSTANCE_TYPE,
            image_id=descriptor.get("image_id"),
            worker_role_arn=descriptor.get("worker_identity"),
            instance_identity_document_sha256="0" * 64,
            ec2_pending_time="1970-01-01T00:00:00Z",
            entrypoint_observed_at="1970-01-01T00:00:00Z",
        )
        del probe
    except Exception as error:
        raise WorkerStartPublisherError(
            "descriptor or submission intent authority is invalid"
        ) from error
    if intent.get("intent_body_sha256") != intent_body_sha:
        raise WorkerStartPublisherError("intent body SHA-256 drifted")
    bucket = descriptor.get("bucket")
    run_id = descriptor.get("run_id")
    descriptor_key = descriptor.get("campaign_descriptor_key")
    if not all(isinstance(item, str) for item in (bucket, run_id, descriptor_key)):
        raise WorkerStartPublisherError("descriptor scalar authority is invalid")
    expected_intent_key = (
        f"campaigns/{run_id}/submissions/qualification/intents/"
        f"{intent_body_sha}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    descriptor_uri = _parse_s3_uri(
        authority.descriptor_s3_uri,
        label="descriptor S3 URI",
    )
    intent_uri = _parse_s3_uri(authority.intent_s3_uri, label="intent S3 URI")
    if descriptor_uri != (bucket, descriptor_key):
        raise WorkerStartPublisherError("descriptor S3 URI authority drifted")
    if intent_uri != (bucket, expected_intent_key):
        raise WorkerStartPublisherError("intent S3 URI authority drifted")
    return _LocalAuthority(
        descriptor=dict(descriptor),
        intent=dict(intent),
        bucket=str(bucket),
        run_id=str(run_id),
        descriptor_key=str(descriptor_key),
        intent_key=expected_intent_key,
    )


def _job_id(environ: Mapping[str, str]) -> int:
    if not isinstance(environ, Mapping):
        raise WorkerStartPublisherError("service environment type is invalid")
    value = environ.get("SKYPILOT_MANAGED_JOB_ID")
    if (
        not isinstance(value, str)
        or len(value) > 4300
        or re.fullmatch(r"[1-9][0-9]*", value) is None
    ):
        raise WorkerStartPublisherError(
            "SKYPILOT_MANAGED_JOB_ID is not a canonical positive job ID"
        )
    try:
        parsed = int(value)
    except (ValueError, OverflowError) as error:
        raise WorkerStartPublisherError(
            "managed job ID cannot be represented"
        ) from error
    if parsed <= 0 or str(parsed) != value:
        raise WorkerStartPublisherError("managed job ID is not canonical")
    return parsed


def _validate_parent_environment(environ: Mapping[str, str]) -> None:
    for name in environ:
        if not isinstance(name, str):
            raise WorkerStartPublisherError("service environment key is invalid")
        if name in _FORBIDDEN_ENV_NAMES or name.startswith("AWS_ENDPOINT_URL"):
            raise WorkerStartPublisherError(
                "forbidden AWS credential or endpoint environment is present"
            )


def _parse_canonical_time(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise WorkerStartPublisherError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError) as error:
        raise WorkerStartPublisherError(f"{label} is invalid") from error
    if parsed.tzinfo is None:
        raise WorkerStartPublisherError(f"{label} is not timezone-aware")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if value != canonical:
        raise WorkerStartPublisherError(f"{label} is not canonical UTC")
    return canonical


def _read_imds(
    services: WorkerStartPublisherServices,
    local: _LocalAuthority,
) -> tuple[dict[str, object], bytes]:
    token_request = urllib.request.Request(
        f"{IMDS_BASE_URL}/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        token = services.imds_opener(token_request, IMDS_TIMEOUT_SECONDS)
    except Exception as error:
        raise WorkerStartPublisherError("IMDSv2 token lookup failed") from error
    if (
        not isinstance(token, bytes)
        or not token
        or len(token) > 4096
        or any(value < 0x21 or value > 0x7E for value in token)
    ):
        raise WorkerStartPublisherError("IMDSv2 token is invalid")
    identity_request = urllib.request.Request(
        (f"{IMDS_BASE_URL}/latest/dynamic/instance-identity/document"),
        method="GET",
        headers={"X-aws-ec2-metadata-token": token.decode("ascii")},
    )
    try:
        raw = services.imds_opener(identity_request, IMDS_TIMEOUT_SECONDS)
    except Exception as error:
        raise WorkerStartPublisherError("IMDSv2 identity lookup failed") from error
    if not isinstance(raw, bytes) or not raw:
        raise WorkerStartPublisherError("IMDS identity document is empty")
    identity = _decode_json(
        raw,
        label="IMDS identity document",
        newline=None,
    )
    expected = {
        "accountId": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "instanceType": local.descriptor.get("instance_type"),
        "imageId": local.descriptor.get("image_id"),
    }
    if any(identity.get(name) != item for name, item in expected.items()):
        raise WorkerStartPublisherError("IMDS worker authority mismatch")
    instance_id = identity.get("instanceId")
    if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
        raise WorkerStartPublisherError("IMDS instance ID is invalid")
    _parse_canonical_time(identity.get("pendingTime"), label="IMDS pendingTime")
    return identity, raw


def _check_root_owned_immutable(path: Path, details: os.stat_result) -> None:
    if details.st_uid != 0 or (
        not stat.S_ISLNK(details.st_mode)
        and details.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise WorkerStartPublisherError(f"AWS CLI closure member is mutable: {path}")
    if stat.S_ISDIR(details.st_mode) and not details.st_mode & stat.S_IXUSR:
        raise WorkerStartPublisherError(
            f"AWS CLI closure directory is not searchable: {path}"
        )


def _resolve_symlinks(path: Path) -> tuple[Path, tuple[Path, ...]]:
    if not path.is_absolute():
        raise WorkerStartPublisherError("AWS CLI symlink path is not absolute")
    resolved = Path("/")
    pending = list(path.parts[1:])
    visited: set[tuple[int, int]] = set()
    links: list[Path] = []
    traversals = 0
    while pending:
        current = resolved / pending.pop(0)
        try:
            details = current.lstat()
        except OSError as error:
            raise WorkerStartPublisherError(
                "AWS CLI symlink chain is missing"
            ) from error
        if not stat.S_ISLNK(details.st_mode):
            resolved = current
            continue
        identity = (details.st_dev, details.st_ino)
        if identity in visited:
            raise WorkerStartPublisherError("AWS CLI symlink chain loops")
        visited.add(identity)
        traversals += 1
        if traversals > 40:
            raise WorkerStartPublisherError("AWS CLI symlink chain is too deep")
        _check_root_owned_immutable(current, details)
        links.append(current)
        try:
            target = os.readlink(current)
        except OSError as error:
            raise WorkerStartPublisherError("AWS CLI symlink cannot be read") from error
        target_path = Path(target)
        if target_path.is_absolute():
            combined = target_path.joinpath(*pending)
        else:
            combined = current.parent.joinpath(target_path, *pending)
        normalized = Path(os.path.normpath(str(combined)))
        if not normalized.is_absolute():
            raise WorkerStartPublisherError("AWS CLI symlink target is not absolute")
        resolved = Path("/")
        pending = list(normalized.parts[1:])
    return resolved, tuple(links)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _snapshot_cli_closure(version_root: Path) -> dict[Path, _FileIdentity]:
    snapshot: dict[Path, _FileIdentity] = {}

    def walk_error(error: OSError) -> None:
        raise WorkerStartPublisherError(
            "AWS CLI closure changed during inspection"
        ) from error

    for root, directories, files in os.walk(
        version_root,
        followlinks=False,
        onerror=walk_error,
    ):
        members = [Path(root), *(Path(root) / name for name in directories + files)]
        for member in members:
            try:
                details = member.lstat()
            except OSError as error:
                raise WorkerStartPublisherError(
                    "AWS CLI closure changed during inspection"
                ) from error
            _check_root_owned_immutable(member, details)
            if stat.S_ISLNK(details.st_mode):
                resolved, _ = _resolve_symlinks(member)
                if not _is_within(resolved, version_root):
                    raise WorkerStartPublisherError("AWS CLI closure symlink escapes")
            elif not (stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode)):
                raise WorkerStartPublisherError(
                    "AWS CLI closure contains a forbidden file type"
                )
            snapshot[member] = _file_identity(details)
    return snapshot


def _attest_aws_cli() -> _AwsCliAttestation:
    try:
        entry_details = AWS_ENTRYPOINT.lstat()
    except OSError as error:
        raise WorkerStartPublisherError(
            "fixed AWS CLI entrypoint is missing"
        ) from error
    if not (stat.S_ISLNK(entry_details.st_mode) or stat.S_ISREG(entry_details.st_mode)):
        raise WorkerStartPublisherError("fixed AWS CLI entrypoint type is invalid")
    _check_root_owned_immutable(AWS_ENTRYPOINT, entry_details)
    target, links = _resolve_symlinks(AWS_ENTRYPOINT)
    prefix = AWS_INSTALL_PREFIX
    try:
        relative = target.relative_to(prefix)
    except ValueError as error:
        raise WorkerStartPublisherError(
            "resolved AWS CLI is outside the fixed version root"
        ) from error
    if len(relative.parts) < 2 or _AWS_VERSION.fullmatch(relative.parts[0]) is None:
        raise WorkerStartPublisherError("AWS CLI version root is not canonical")
    version_root = prefix / relative.parts[0]
    if target.name != "aws" or not _is_within(target, version_root):
        raise WorkerStartPublisherError("resolved AWS CLI executable is invalid")
    for link in links[1:]:
        if not _is_within(link, prefix):
            raise WorkerStartPublisherError(
                "AWS CLI symlink chain left the fixed installation prefix"
            )
        resolved, _ = _resolve_symlinks(link)
        if not _is_within(resolved, version_root):
            raise WorkerStartPublisherError("AWS CLI closure symlink escapes")
    fixed_root = prefix.parents[1]
    ancestors = {
        fixed_root,
        AWS_ENTRYPOINT.parent,
        prefix.parent,
        prefix,
        version_root,
        *(parent for parent in target.parents if _is_within(parent, fixed_root)),
    }
    for ancestor in sorted(ancestors, key=lambda item: len(item.parts)):
        try:
            details = ancestor.lstat()
        except OSError as error:
            raise WorkerStartPublisherError("AWS CLI ancestor is missing") from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise WorkerStartPublisherError("AWS CLI ancestor type is invalid")
        _check_root_owned_immutable(ancestor, details)
    first_snapshot = _snapshot_cli_closure(version_root)
    second_snapshot = _snapshot_cli_closure(version_root)
    if first_snapshot != second_snapshot:
        raise WorkerStartPublisherError("AWS CLI closure changed during inspection")
    try:
        current_entry_details = AWS_ENTRYPOINT.lstat()
        current_target, _ = _resolve_symlinks(AWS_ENTRYPOINT)
        target_details = target.lstat()
    except OSError as error:
        raise WorkerStartPublisherError("AWS CLI executable vanished") from error
    if (
        _file_identity(current_entry_details) != _file_identity(entry_details)
        or current_target != target
    ):
        raise WorkerStartPublisherError("AWS CLI changed during inspection")
    if (
        not stat.S_ISREG(target_details.st_mode)
        or not target_details.st_mode & stat.S_IXUSR
    ):
        raise WorkerStartPublisherError("AWS CLI executable is not owner-executable")
    return _AwsCliAttestation(
        executable=target,
        entrypoint_identity=_file_identity(entry_details),
        target_identity=_file_identity(target_details),
    )


def _revalidate_aws_cli(attestation: _AwsCliAttestation) -> None:
    if attestation.entrypoint_identity is None or attestation.target_identity is None:
        raise WorkerStartPublisherError("AWS CLI attestation is incomplete")
    try:
        current_entry = _file_identity(AWS_ENTRYPOINT.lstat())
        resolved, _ = _resolve_symlinks(AWS_ENTRYPOINT)
        current_target = _file_identity(resolved.lstat())
    except OSError as error:
        raise WorkerStartPublisherError("AWS CLI changed before execution") from error
    if (
        current_entry != attestation.entrypoint_identity
        or resolved != attestation.executable
        or current_target != attestation.target_identity
    ):
        raise WorkerStartPublisherError("AWS CLI changed before execution")


def _default_command_runner(
    argv: tuple[str, ...],
    environ: Mapping[str, str],
    timeout: int,
) -> WorkerStartCommandResult:
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            env=dict(environ),
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _AmbiguousTransport("AWS CLI process result is ambiguous") from error
    return WorkerStartCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _common_cli_args(attestation: _AwsCliAttestation) -> tuple[str, ...]:
    return (
        "--region",
        APPROVED_REGION,
        "--cli-connect-timeout",
        "3",
        "--cli-read-timeout",
        "10",
        "--no-cli-pager",
        "--output",
        "json",
    )


def _run_aws(
    services: WorkerStartPublisherServices,
    attestation: _AwsCliAttestation,
    operation_args: Sequence[str],
    *,
    trailing_args: Sequence[str] = (),
) -> WorkerStartCommandResult:
    _revalidate_aws_cli(attestation)
    argv = (
        str(attestation.executable),
        *tuple(operation_args),
        *_common_cli_args(attestation),
        *tuple(trailing_args),
    )
    if "--no-sign-request" in argv:
        raise WorkerStartPublisherError("unsigned AWS requests are forbidden")
    try:
        result = services.command_runner(
            argv,
            dict(_SANITIZED_AWS_ENV),
            AWS_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except _AmbiguousTransport:
        raise
    except (OSError, TimeoutError, subprocess.TimeoutExpired) as error:
        raise _AmbiguousTransport("AWS CLI process result is ambiguous") from error
    except Exception as error:
        raise WorkerStartPublisherError("AWS CLI command runner failed") from error
    if (
        type(result) is not WorkerStartCommandResult
        or type(result.returncode) is not int
        or not isinstance(result.stdout, bytes)
        or not isinstance(result.stderr, bytes)
    ):
        raise WorkerStartPublisherError("AWS CLI result is malformed")
    return result


def _aws_json(
    services: WorkerStartPublisherServices,
    attestation: _AwsCliAttestation,
    operation_args: Sequence[str],
    *,
    operation: str,
    permit_ambiguous: bool = False,
) -> dict[str, object]:
    try:
        result = _run_aws(services, attestation, operation_args)
    except _AmbiguousTransport as error:
        if permit_ambiguous:
            raise
        raise WorkerStartPublisherError(
            f"{operation} transport result is ambiguous"
        ) from error
    if result.returncode != 0:
        if result.stdout:
            raise WorkerStartPublisherError(
                f"{operation} failed with a noncanonical service error"
            )
        match = _CLI_ERROR.fullmatch(result.stderr)
        if match is None or match.group(2).decode("ascii") != operation:
            raise WorkerStartPublisherError(
                f"{operation} failed with a noncanonical service error"
            )
        raise _ClientError(match.group(1).decode("ascii"))
    if result.stderr:
        raise WorkerStartPublisherError(f"{operation} wrote unexpected stderr")
    return _decode_json(
        result.stdout,
        label=f"{operation} response",
        newline=None,
    )


def _sts_identity(
    services: WorkerStartPublisherServices,
    cli: _AwsCliAttestation,
    worker_role_arn: object,
) -> dict[str, object]:
    response = _aws_json(
        services,
        cli,
        ("sts", "get-caller-identity"),
        operation="GetCallerIdentity",
    )
    account = response.get("Account")
    arn = response.get("Arn")
    if account != APPROVED_ACCOUNT_ID or not isinstance(arn, str):
        raise WorkerStartPublisherError("STS account authority mismatch")
    if not isinstance(worker_role_arn, str) or not worker_role_arn.startswith(
        f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
    ):
        raise WorkerStartPublisherError("descriptor worker role is invalid")
    role_name = worker_role_arn.rsplit("/", 1)[-1]
    prefix = f"arn:aws:sts::{APPROVED_ACCOUNT_ID}:assumed-role/{role_name}/"
    if not arn.startswith(prefix) or arn == prefix:
        raise WorkerStartPublisherError("STS assumed-role authority mismatch")
    return response


def _prepare_local(
    *,
    sources: object,
    authority: object,
    services: object,
) -> tuple[_LoadedSources, _LocalAuthority, int, WorkerStartPublisherServices]:
    if type(services) is not WorkerStartPublisherServices:
        raise WorkerStartPublisherError("publisher services type is invalid")
    loaded = _validate_source_bundle(sources)
    local = _load_local_authority(authority, loaded)
    job_id = _job_id(services.environ)
    _validate_parent_environment(services.environ)
    return loaded, local, job_id, services


def _prepare_external(
    *,
    loaded: _LoadedSources,
    local: _LocalAuthority,
    job_id: int,
    services: WorkerStartPublisherServices,
) -> _Prepared:
    identity, identity_raw = _read_imds(services, local)
    cli = _attest_aws_cli()
    sts = _sts_identity(
        services,
        cli,
        local.descriptor.get("worker_identity"),
    )
    worker = _WorkerIdentity(
        account_id=APPROVED_ACCOUNT_ID,
        region=APPROVED_REGION,
        instance_id=str(identity["instanceId"]),
        instance_type=str(identity["instanceType"]),
        image_id=str(identity["imageId"]),
        worker_role_arn=str(local.descriptor["worker_identity"]),
        pending_time=_parse_canonical_time(
            identity["pendingTime"],
            label="IMDS pendingTime",
        ),
        identity_document_sha256=_sha256(identity_raw),
    )
    return _Prepared(
        sources=loaded,
        authority=local,
        job_id=job_id,
        worker=worker,
        cli=cli,
        sts_identity=sts,
    )


def _prepare(
    *,
    sources: object,
    authority: object,
    services: object,
) -> _Prepared:
    loaded, local, job_id, exact_services = _prepare_local(
        sources=sources,
        authority=authority,
        services=services,
    )
    return _prepare_external(
        loaded=loaded,
        local=local,
        job_id=job_id,
        services=exact_services,
    )


def _last_modified(value: object, *, label: str) -> tuple[str, datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise WorkerStartPublisherError(
                f"{label} LastModified is not timezone-aware"
            )
        canonical = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return canonical, datetime.fromisoformat(canonical.replace("Z", "+00:00"))
    if not isinstance(value, str):
        raise WorkerStartPublisherError(f"{label} LastModified is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError) as error:
        raise WorkerStartPublisherError(f"{label} LastModified is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise WorkerStartPublisherError(f"{label} LastModified is not UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if value not in {canonical, canonical.replace("Z", "+00:00")}:
        raise WorkerStartPublisherError(
            f"{label} LastModified is not canonical ISO-8601"
        )
    return canonical, datetime.fromisoformat(canonical.replace("Z", "+00:00"))


def _transport(response: object, *, label: str) -> _Transport:
    if type(response) is not dict:
        raise WorkerStartPublisherError(f"{label} response is malformed")
    version = response.get("VersionId")
    etag = response.get("ETag")
    length = response.get("ContentLength")
    checksum = response.get("ChecksumSHA256")
    checksum_type = response.get("ChecksumType")
    content_type = response.get("ContentType")
    metadata = response.get("Metadata")
    delete_marker = response.get("DeleteMarker")
    if (
        (delete_marker is not None and delete_marker is not False)
        or not isinstance(version, str)
        or _VERSION_ID.fullmatch(version) is None
        or not isinstance(etag, str)
        or _ETAG.fullmatch(etag) is None
        or type(length) is not int
        or length <= 0
        or not isinstance(checksum, str)
        or checksum_type != "FULL_OBJECT"
        or content_type != "application/json"
        or type(metadata) is not dict
        or any(
            not isinstance(name, str) or not isinstance(item, str)
            for name, item in metadata.items()
        )
    ):
        raise WorkerStartPublisherError(f"{label} transport is malformed")
    modified, _ = _last_modified(response.get("LastModified"), label=label)
    return _Transport(
        version_id=version,
        etag=etag,
        last_modified=modified,
        content_length=length,
        checksum_sha256=checksum,
        checksum_type=checksum_type,
        content_type=content_type,
        metadata=dict(metadata),
    )


def _cli_response(value: dict[str, object]) -> dict[str, object]:
    response = dict(value)
    if "LastModified" in response:
        canonical, parsed = _last_modified(
            response["LastModified"],
            label="AWS CLI",
        )
        del canonical
        response["LastModified"] = parsed
    contents = response.get("Contents")
    if isinstance(contents, list):
        translated_contents: list[object] = []
        for member in contents:
            if isinstance(member, dict) and "LastModified" in member:
                translated_member = dict(member)
                canonical, parsed = _last_modified(
                    translated_member["LastModified"],
                    label="AWS CLI LIST member",
                )
                del canonical
                translated_member["LastModified"] = parsed
                translated_contents.append(translated_member)
            else:
                translated_contents.append(member)
        response["Contents"] = translated_contents
    response["ResponseMetadata"] = {"HTTPStatusCode": 200}
    return response


class _AwsCliS3:
    def __init__(
        self,
        *,
        services: WorkerStartPublisherServices,
        cli: _AwsCliAttestation,
        bucket: str,
        run_id: str,
        required_acceptance_key: str | None = None,
        acceptance_prefix: str | None = None,
    ) -> None:
        self.services = services
        self.cli = cli
        self.bucket = bucket
        self.run_id = run_id
        self.required_acceptance_key = required_acceptance_key
        self.acceptance_prefix = acceptance_prefix
        self.records: dict[str, dict[str, object]] = {}
        self._pending_gets: dict[tuple[str, str], tuple[bytes, _Transport]] = {}
        self._list_sequences: dict[str, list[str]] = {}

    def _base_args(
        self,
        operation: str,
        *,
        bucket: object,
        key: object | None = None,
    ) -> list[str]:
        if bucket != self.bucket:
            raise WorkerStartPublisherError("S3 bucket authority drifted")
        args = ["s3api", operation, "--bucket", self.bucket]
        if key is not None:
            if not isinstance(key, str) or not key:
                raise WorkerStartPublisherError("S3 key is invalid")
            args.extend(("--key", key))
        args.extend(("--expected-bucket-owner", APPROVED_ACCOUNT_ID))
        return args

    def head_object(self, **kwargs: object) -> dict[str, object]:
        allowed = {
            "Bucket",
            "Key",
            "VersionId",
            "ChecksumMode",
            "ExpectedBucketOwner",
        }
        if (
            set(kwargs) - allowed
            or kwargs.get("ChecksumMode") != "ENABLED"
            or kwargs.get("ExpectedBucketOwner") != APPROVED_ACCOUNT_ID
        ):
            raise WorkerStartPublisherError("HEAD request shape is invalid")
        key = kwargs.get("Key")
        args = self._base_args(
            "head-object",
            bucket=kwargs.get("Bucket"),
            key=key,
        )
        version = kwargs.get("VersionId")
        if version is not None:
            if not isinstance(version, str) or _VERSION_ID.fullmatch(version) is None:
                raise WorkerStartPublisherError("HEAD VersionId is invalid")
            args.extend(("--version-id", version))
        args.extend(("--checksum-mode", "ENABLED"))
        response = _aws_json(
            self.services,
            self.cli,
            args,
            operation="HeadObject",
        )
        translated = _cli_response(response)
        if version is not None:
            parsed = _transport(translated, label=f"HEAD {key}")
            pending = self._pending_gets.get((str(key), version))
            if pending is not None:
                raw, get_transport = pending
                if parsed != get_transport:
                    raise WorkerStartPublisherError(
                        "GET and versioned HEAD transport disagree"
                    )
                self._record(str(key), raw, parsed)
        return translated

    def get_object(self, **kwargs: object) -> dict[str, object]:
        allowed = {
            "Bucket",
            "Key",
            "VersionId",
            "ChecksumMode",
            "ExpectedBucketOwner",
        }
        if (
            set(kwargs) != allowed
            or kwargs.get("ChecksumMode") != "ENABLED"
            or kwargs.get("ExpectedBucketOwner") != APPROVED_ACCOUNT_ID
        ):
            raise WorkerStartPublisherError("GET request shape is invalid")
        key = kwargs.get("Key")
        version = kwargs.get("VersionId")
        if not isinstance(version, str) or _VERSION_ID.fullmatch(version) is None:
            raise WorkerStartPublisherError("GET VersionId is invalid")
        descriptor, output_name = tempfile.mkstemp(prefix="glm52-s3-get-")
        os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        output_path = Path(output_name)
        try:
            args = self._base_args(
                "get-object",
                bucket=kwargs.get("Bucket"),
                key=key,
            )
            args.extend(
                (
                    "--version-id",
                    version,
                    "--checksum-mode",
                    "ENABLED",
                )
            )
            # get-object's one outfile positional remains last.
            try:
                result = _run_aws(
                    self.services,
                    self.cli,
                    args,
                    trailing_args=(str(output_path),),
                )
            except _AmbiguousTransport as error:
                raise WorkerStartPublisherError(
                    "GetObject transport result is ambiguous"
                ) from error
            if result.returncode != 0:
                if result.stdout:
                    raise WorkerStartPublisherError(
                        "GetObject failed with a noncanonical service error"
                    )
                match = _CLI_ERROR.fullmatch(result.stderr)
                if match is None or match.group(2) != b"GetObject":
                    raise WorkerStartPublisherError(
                        "GetObject failed with a noncanonical service error"
                    )
                raise _ClientError(match.group(1).decode("ascii"))
            if result.stderr:
                raise WorkerStartPublisherError("GetObject wrote unexpected stderr")
            response = _decode_json(
                result.stdout,
                label="GetObject response",
                newline=None,
            )
            try:
                output_details = output_path.lstat()
                if (
                    stat.S_ISLNK(output_details.st_mode)
                    or not stat.S_ISREG(output_details.st_mode)
                    or stat.S_IMODE(output_details.st_mode) != 0o600
                ):
                    raise WorkerStartPublisherError("GetObject output file is invalid")
                raw = _read_stable_regular(
                    output_path,
                    label="GetObject output file",
                    required_mode=0o600,
                )
            except WorkerStartPublisherError:
                raise
            except OSError as error:
                raise WorkerStartPublisherError(
                    "GetObject output file was not produced"
                ) from error
        finally:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
        translated = _cli_response(response)
        parsed = _transport(translated, label=f"GET {key}")
        if parsed.version_id != version:
            raise WorkerStartPublisherError("GET returned the wrong VersionId")
        self._pending_gets[(str(key), version)] = (raw, parsed)
        return {**translated, "Body": _BytesBody(raw)}

    def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
        allowed = {
            "Bucket",
            "Prefix",
            "ContinuationToken",
            "ExpectedBucketOwner",
        }
        if (
            set(kwargs) - allowed
            or kwargs.get("ExpectedBucketOwner") != APPROVED_ACCOUNT_ID
            or not isinstance(kwargs.get("Prefix"), str)
        ):
            raise WorkerStartPublisherError("LIST request shape is invalid")
        prefix = str(kwargs["Prefix"])
        args = self._base_args(
            "list-objects-v2",
            bucket=kwargs.get("Bucket"),
        )
        args.extend(("--prefix", prefix, "--no-paginate"))
        token = kwargs.get("ContinuationToken")
        if token is not None:
            if not isinstance(token, str) or not token:
                raise WorkerStartPublisherError("LIST token is invalid")
            args.extend(("--continuation-token", token))
        response = _aws_json(
            self.services,
            self.cli,
            args,
            operation="ListObjectsV2",
        )
        if token is None:
            self._list_sequences[prefix] = []
        sequence = self._list_sequences.setdefault(prefix, [])
        contents = response.get("Contents", [])
        if type(contents) is not list:
            raise WorkerStartPublisherError("LIST contents are malformed")
        for member in contents:
            if type(member) is not dict or not isinstance(member.get("Key"), str):
                raise WorkerStartPublisherError("LIST member is malformed")
            sequence.append(str(member["Key"]))
        if response.get("IsTruncated") is False and prefix == self.acceptance_prefix:
            if sequence.count(str(self.required_acceptance_key)) != 1:
                raise WorkerStartPublisherError(
                    "deterministic current acceptance is absent from inventory"
                )
        return _cli_response(response)

    def put_object(self, **_: object) -> object:
        raise WorkerStartPublisherError("read-only replay attempted S3 PUT")

    def _record(self, key: str, raw: bytes, transport: _Transport) -> None:
        expected_checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode(
            "ascii"
        )
        if (
            transport.content_length != len(raw)
            or transport.checksum_sha256 != expected_checksum
            or transport.metadata
            != {
                "glm52-run-id": self.run_id,
                "glm52-body-sha256": transport.metadata.get("glm52-body-sha256"),
            }
        ):
            raise WorkerStartPublisherError("S3 object transport is inconsistent")
        body_sha = transport.metadata.get("glm52-body-sha256")
        _require_digest(body_sha, label="S3 metadata body SHA-256")
        record = {
            "key": key,
            "file_sha256": _sha256(raw),
            "body_sha256": body_sha,
            "version_id": transport.version_id,
            "etag": transport.etag,
            "last_modified": transport.last_modified,
            "content_length": transport.content_length,
            "checksum_sha256": transport.checksum_sha256,
            "checksum_type": transport.checksum_type,
            "content_type": transport.content_type,
            "metadata": dict(transport.metadata),
            "trailing_newline": raw.endswith(b"\n"),
        }
        existing = self.records.get(key)
        if existing is not None and existing != record:
            raise WorkerStartPublisherError(
                "one S3 key resolved to inconsistent transports"
            )
        self.records[key] = record


class _AwsCliSts:
    def __init__(
        self,
        services: WorkerStartPublisherServices,
        cli: _AwsCliAttestation,
        role: str,
    ) -> None:
        self.services = services
        self.cli = cli
        self.role = role

    def get_caller_identity(self) -> dict[str, object]:
        return _sts_identity(self.services, self.cli, self.role)


def _read_version(
    s3: _AwsCliS3,
    *,
    key: str,
    version_id: str,
) -> tuple[bytes, _Transport]:
    try:
        response = s3.get_object(
            Bucket=s3.bucket,
            Key=key,
            VersionId=version_id,
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=APPROVED_ACCOUNT_ID,
        )
        stream = response.get("Body")
        if not isinstance(stream, _BytesBody):
            raise WorkerStartPublisherError("GET body stream is invalid")
        raw = stream.read()
        stream.close()
        get_transport = _transport(response, label=f"GET {key}")
        head = s3.head_object(
            Bucket=s3.bucket,
            Key=key,
            VersionId=version_id,
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=APPROVED_ACCOUNT_ID,
        )
        head_transport = _transport(head, label=f"HEAD {key}")
    except _ClientError as error:
        raise WorkerStartPublisherError("required S3 object is unavailable") from error
    if get_transport != head_transport:
        raise WorkerStartPublisherError("GET and versioned HEAD disagree")
    if len(raw) != get_transport.content_length:
        raise WorkerStartPublisherError("S3 content length drifted")
    return raw, get_transport


def _resolve_current_version(s3: _AwsCliS3, *, key: str) -> str:
    try:
        response = s3.head_object(
            Bucket=s3.bucket,
            Key=key,
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=APPROVED_ACCOUNT_ID,
        )
    except _ClientError as error:
        if str(error) in _MISSING_CODES:
            raise _Unavailable from error
        raise WorkerStartPublisherError("unversioned HEAD failed") from error
    return _transport(response, label=f"HEAD {key}").version_id


def _list_all(s3: _AwsCliS3, *, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    seen: set[str] = set()
    while True:
        kwargs: dict[str, object] = {
            "Bucket": s3.bucket,
            "Prefix": prefix,
            "ExpectedBucketOwner": APPROVED_ACCOUNT_ID,
        }
        if token is not None:
            kwargs["ContinuationToken"] = token
        try:
            response = s3.list_objects_v2(**kwargs)
        except _ClientError as error:
            raise WorkerStartPublisherError("S3 legacy audit failed") from error
        contents = response.get("Contents", [])
        if type(contents) is not list:
            raise WorkerStartPublisherError("S3 listing is malformed")
        page: list[str] = []
        for member in contents:
            if (
                type(member) is not dict
                or not isinstance(member.get("Key"), str)
                or type(member.get("Size")) is not int
                or member["Size"] <= 0
            ):
                raise WorkerStartPublisherError("S3 listing member is malformed")
            page.append(str(member["Key"]))
        if page != sorted(page) or len(page) != len(set(page)):
            raise WorkerStartPublisherError("S3 listing page is noncanonical")
        if keys and page and page[0] <= keys[-1]:
            raise WorkerStartPublisherError("S3 listing pages overlap")
        keys.extend(page)
        truncated = response.get("IsTruncated")
        if type(truncated) is not bool:
            raise WorkerStartPublisherError("S3 listing truncation is invalid")
        next_token = response.get("NextContinuationToken")
        if not truncated:
            if next_token is not None:
                raise WorkerStartPublisherError("terminal S3 page has a token")
            break
        if not isinstance(next_token, str) or not next_token or next_token in seen:
            raise WorkerStartPublisherError("S3 listing token is invalid")
        seen.add(next_token)
        token = next_token
    return keys


def _dynamic_prefix(local: _LocalAuthority) -> str:
    return (
        f"campaigns/{local.run_id}/monitor/must-start/qualification/"
        f"{local.intent['intent_body_sha256']}/"
    )


def _audit_legacy(s3: _AwsCliS3, local: _LocalAuthority) -> None:
    keys = _list_all(s3, prefix=_dynamic_prefix(local))
    if any(key.rsplit("/", 1)[-1] in _FORBIDDEN_LEGACY for key in keys):
        raise WorkerStartPublisherError("forbidden legacy authority is present")


def _validate_output_target(path: object, *, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise WorkerStartPublisherError(f"{label} must be an absolute Path")
    parent = path.parent
    try:
        parent_details = parent.lstat()
    except OSError as error:
        raise WorkerStartPublisherError(f"{label} parent is missing") from error
    if stat.S_ISLNK(parent_details.st_mode) or not stat.S_ISDIR(parent_details.st_mode):
        raise WorkerStartPublisherError(f"{label} parent is invalid")
    try:
        details = path.lstat()
    except FileNotFoundError:
        return path
    except OSError as error:
        raise WorkerStartPublisherError(f"{label} cannot be inspected") from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
    ):
        raise WorkerStartPublisherError(f"{label} existing path is invalid")
    return path


def _require_distinct_paths(*paths: Path) -> None:
    normalized = [Path(os.path.normpath(str(path))) for path in paths]
    if len(set(normalized)) != len(normalized):
        raise WorkerStartPublisherError("local authority paths must be distinct")
    identities: set[tuple[int, int]] = set()
    for path in paths:
        try:
            details = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise WorkerStartPublisherError(
                "local authority path cannot be inspected"
            ) from error
        identity = (details.st_dev, details.st_ino)
        if identity in identities:
            raise WorkerStartPublisherError("local authority paths must be distinct")
        identities.add(identity)


def _validate_output_pair(first: Path, second: Path) -> tuple[Path, Path]:
    first = _validate_output_target(first, label="first output")
    second = _validate_output_target(second, label="marker output")
    _require_distinct_paths(first, second)
    return first, second


def _read_exact_persisted_observation(
    descriptor: int,
    *,
    parent_fd: int,
    name: str,
    raw: bytes,
    label: str,
    expected_size: int | None,
    content_error: str,
) -> tuple[os.stat_result, os.stat_result]:
    try:
        opened_before = os.fstat(descriptor)
        entry_before = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise WorkerStartPublisherError(
            f"{label} final entry cannot be authenticated"
        ) from error
    stable_size = opened_before.st_size if expected_size is None else expected_size
    if not _same_persisted_file(
        opened_before,
        entry_before,
        expected_size=stable_size,
    ):
        raise WorkerStartPublisherError(
            f"{label} final entry changed during persistence"
        )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
        entry_after = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise WorkerStartPublisherError(
            f"{label} final entry cannot be authenticated"
        ) from error
    if (
        not _same_persisted_file(
            opened_after,
            entry_after,
            expected_size=stable_size,
        )
        or not _same_file_observation(opened_before, opened_after)
        or not _same_file_observation(entry_before, entry_after)
    ):
        raise WorkerStartPublisherError(
            f"{label} final bytes changed during persistence"
        )
    if b"".join(chunks) != raw:
        raise WorkerStartPublisherError(content_error)
    return opened_after, entry_after


def _persist_exact(path: Path, raw: bytes, *, label: str) -> None:
    path = _validate_output_target(path, label=label)
    try:
        existing_details = path.lstat()
    except FileNotFoundError:
        existing_details = None
    except OSError as error:
        raise WorkerStartPublisherError(f"{label} cannot be inspected") from error
    if existing_details is not None:
        if (
            stat.S_ISLNK(existing_details.st_mode)
            or not stat.S_ISREG(existing_details.st_mode)
            or stat.S_IMODE(existing_details.st_mode) != 0o600
        ):
            raise WorkerStartPublisherError(f"{label} existing path is invalid")
    parent_fd: int | None = None
    try:
        try:
            parent_before = path.parent.lstat()
            parent_fd = os.open(
                path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            parent_open = os.fstat(parent_fd)
        except OSError as error:
            raise WorkerStartPublisherError(
                f"{label} parent cannot be opened"
            ) from error
        if not _same_directory_identity(parent_before, parent_open):
            raise WorkerStartPublisherError(
                f"{label} parent changed before persistence"
            )

        def persist_file() -> None:
            descriptor: int | None = None
            try:
                if existing_details is None:
                    descriptor = os.open(
                        path.name,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=parent_fd,
                    )
                    os.fchmod(descriptor, 0o600)
                    written = 0
                    while written < len(raw):
                        count = os.write(descriptor, raw[written:])
                        if count <= 0:
                            raise OSError("short output write")
                        written += count
                    expected_size: int | None = len(raw)
                    content_error = f"{label} final bytes changed during persistence"
                else:
                    descriptor = os.open(
                        path.name,
                        os.O_RDWR | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                    expected_size = None
                    content_error = f"{label} existing content is foreign"
                before_sync_opened, before_sync_entry = (
                    _read_exact_persisted_observation(
                        descriptor,
                        parent_fd=parent_fd,
                        name=path.name,
                        raw=raw,
                        label=label,
                        expected_size=expected_size,
                        content_error=content_error,
                    )
                )
                if existing_details is not None and (
                    not _same_file_observation(
                        existing_details,
                        before_sync_opened,
                    )
                    or not _same_file_observation(
                        existing_details,
                        before_sync_entry,
                    )
                ):
                    raise WorkerStartPublisherError(
                        f"{label} final entry changed during persistence"
                    )
                os.fsync(descriptor)
                after_sync_opened, after_sync_entry = _read_exact_persisted_observation(
                    descriptor,
                    parent_fd=parent_fd,
                    name=path.name,
                    raw=raw,
                    label=label,
                    expected_size=len(raw),
                    content_error=(f"{label} final bytes changed during persistence"),
                )
                if not _same_file_observation(
                    before_sync_opened,
                    after_sync_opened,
                ) or not _same_file_observation(
                    before_sync_entry,
                    after_sync_entry,
                ):
                    raise WorkerStartPublisherError(
                        f"{label} final bytes changed during persistence"
                    )
                closing_descriptor = descriptor
                descriptor = None
                os.close(closing_descriptor)
                os.fsync(parent_fd)
                try:
                    parent_after = path.parent.lstat()
                except OSError as error:
                    raise WorkerStartPublisherError(
                        f"{label} final entry cannot be authenticated"
                    ) from error
                if not _same_directory_identity(parent_before, parent_after):
                    raise WorkerStartPublisherError(
                        f"{label} parent changed during persistence"
                    )
                try:
                    descriptor = os.open(
                        path.name,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                    reopened = os.fstat(descriptor)
                    entry_after = os.stat(
                        path.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise WorkerStartPublisherError(
                        f"{label} final entry cannot be authenticated"
                    ) from error
                if (
                    not _same_persisted_file(
                        after_sync_opened,
                        reopened,
                        expected_size=len(raw),
                    )
                    or not _same_persisted_file(
                        reopened,
                        entry_after,
                        expected_size=len(raw),
                    )
                    or not _same_file_observation(
                        after_sync_opened,
                        reopened,
                    )
                    or not _same_file_observation(
                        after_sync_entry,
                        entry_after,
                    )
                ):
                    raise WorkerStartPublisherError(
                        f"{label} final entry changed during persistence"
                    )
                final_opened, final_entry = _read_exact_persisted_observation(
                    descriptor,
                    parent_fd=parent_fd,
                    name=path.name,
                    raw=raw,
                    label=label,
                    expected_size=len(raw),
                    content_error=(f"{label} final bytes changed during persistence"),
                )
                if not _same_file_observation(
                    after_sync_opened,
                    final_opened,
                ) or not _same_file_observation(
                    after_sync_entry,
                    final_entry,
                ):
                    raise WorkerStartPublisherError(
                        f"{label} final bytes changed during persistence"
                    )
            except FileExistsError as error:
                raise WorkerStartPublisherError(
                    f"{label} appeared concurrently"
                ) from error
            except WorkerStartPublisherError:
                raise
            except OSError as error:
                raise WorkerStartPublisherError(
                    f"{label} persistence failed"
                ) from error
            finally:
                if descriptor is not None:
                    closing_descriptor = descriptor
                    descriptor = None
                    try:
                        os.close(closing_descriptor)
                    except OSError as error:
                        raise WorkerStartPublisherError(
                            f"{label} persistence failed"
                        ) from error

        persist_file()
    finally:
        if parent_fd is not None:
            closing_parent_fd = parent_fd
            parent_fd = None
            try:
                os.close(closing_parent_fd)
            except OSError as error:
                raise WorkerStartPublisherError(
                    f"{label} parent close failed"
                ) from error


def _self_hashed(
    body: Mapping[str, object],
    *,
    digest_field: str,
) -> dict[str, object]:
    value = dict(body)
    value[digest_field] = _sha256(_canonical_bytes(value, newline=False))
    return value


def _validate_self_hashed(
    value: dict[str, object],
    *,
    fields: frozenset[str],
    digest_field: str,
    record_type: str,
    label: str,
) -> dict[str, object]:
    if (
        set(value) != fields
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("record_type") != record_type
    ):
        raise WorkerStartPublisherError(f"{label} schema mismatch")
    digest = _require_digest(value.get(digest_field), label=f"{label} digest")
    body = dict(value)
    body.pop(digest_field)
    if digest != _sha256(_canonical_bytes(body, newline=False)):
        raise WorkerStartPublisherError(f"{label} body SHA-256 mismatch")
    return dict(value)


def _assert_current_worker(
    prepared: _Prepared,
    latch: Mapping[str, object],
    receipt: Mapping[str, object] | None = None,
) -> None:
    expected = {
        "account_id": prepared.worker.account_id,
        "region": prepared.worker.region,
        "instance_id": prepared.worker.instance_id,
        "instance_type": prepared.worker.instance_type,
        "image_id": prepared.worker.image_id,
        "worker_role_arn": prepared.worker.worker_role_arn,
        "ec2_pending_time": prepared.worker.pending_time,
        "instance_identity_document_sha256": (prepared.worker.identity_document_sha256),
        "controller_injected_sky_job_id": prepared.job_id,
    }
    if any(latch.get(name) != item for name, item in expected.items()):
        raise WorkerStartPublisherError(
            "local latch does not describe the current worker"
        )
    if receipt is not None:
        receipt_expected = {
            "account_id": prepared.worker.account_id,
            "region": prepared.worker.region,
            "instance_id": prepared.worker.instance_id,
            "worker_instance_type": prepared.worker.instance_type,
            "worker_image_id": prepared.worker.image_id,
            "worker_role_arn": prepared.worker.worker_role_arn,
            "ec2_pending_time": prepared.worker.pending_time,
            "instance_identity_document_sha256": (
                prepared.worker.identity_document_sha256
            ),
            "sky_job_id": prepared.job_id,
        }
        if any(receipt.get(name) != item for name, item in receipt_expected.items()):
            raise WorkerStartPublisherError(
                "admission receipt does not describe the current worker"
            )


def _validate_local_latch_transport(
    loaded: _LoadedSources,
    local: _LocalAuthority,
    job_id: int,
    *,
    latch_path: Path,
    latch_transport_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    _require_distinct_paths(latch_path, latch_transport_path)
    latch_file = _require_regular_path(latch_path, label="local worker latch")
    transport_file = _require_regular_path(
        latch_transport_path,
        label="local latch transport",
    )
    if (
        stat.S_IMODE(latch_file.lstat().st_mode) != 0o600
        or stat.S_IMODE(transport_file.lstat().st_mode) != 0o600
    ):
        raise WorkerStartPublisherError("local latch output mode drifted")
    latch_raw = _read_stable_regular(
        latch_file,
        label="local worker latch",
        required_mode=0o600,
    )
    transport_raw = _read_stable_regular(
        transport_file,
        label="local latch transport",
        required_mode=0o600,
    )
    latch = _decode_json(latch_raw, label="local worker latch", newline=False)
    transport = _decode_json(
        transport_raw,
        label="local latch transport",
        newline=True,
    )
    try:
        latch = loaded.worker.validate_worker_start_latch_v2(
            latch,
            descriptor=local.descriptor,
            intent=local.intent,
        )
        latch_key = loaded.worker.worker_start_latch_v2_s3_key(worker_latch=latch)
    except Exception as error:
        raise WorkerStartPublisherError("local worker latch is invalid") from error
    _validate_self_hashed(
        transport,
        fields=_LATCH_TRANSPORT_FIELDS,
        digest_field="latch_transport_body_sha256",
        record_type="glm52_sky_worker_start_latch_transport_v1",
        label="latch transport",
    )
    checksum = base64.b64encode(hashlib.sha256(latch_raw).digest()).decode("ascii")
    metadata = {
        "glm52-run-id": local.run_id,
        "glm52-body-sha256": str(latch["worker_latch_body_sha256"]),
    }
    exact = {
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": local.bucket,
        "run_id": local.run_id,
        "intent_body_sha256": local.intent["intent_body_sha256"],
        "sky_job_id": job_id,
        "instance_id": latch["instance_id"],
        "worker_instance_type": latch["instance_type"],
        "worker_image_id": latch["image_id"],
        "worker_role_arn": latch["worker_role_arn"],
        "ec2_pending_time": latch["ec2_pending_time"],
        "instance_identity_document_sha256": (
            latch["instance_identity_document_sha256"]
        ),
        "worker_latch_key": latch_key,
        "worker_latch_file_sha256": _sha256(latch_raw),
        "worker_latch_body_sha256": latch["worker_latch_body_sha256"],
        "content_length": len(latch_raw),
        "checksum_sha256": checksum,
        "checksum_type": "FULL_OBJECT",
        "content_type": "application/json",
        "metadata": metadata,
    }
    if any(transport.get(name) != item for name, item in exact.items()):
        raise WorkerStartPublisherError("latch transport authority drifted")
    if (
        type(transport.get("sky_job_id")) is not int
        or type(transport.get("content_length")) is not int
        or not isinstance(transport.get("worker_latch_version_id"), str)
        or _VERSION_ID.fullmatch(str(transport["worker_latch_version_id"])) is None
        or not isinstance(transport.get("worker_latch_etag"), str)
        or _ETAG.fullmatch(str(transport["worker_latch_etag"])) is None
    ):
        raise WorkerStartPublisherError("latch transport scalar type drifted")
    _parse_canonical_time(
        transport.get("worker_latch_last_modified"),
        label="latch transport LastModified",
    )
    return dict(latch), dict(transport)


def _latch_and_transport(
    prepared: _Prepared,
    *,
    latch_path: Path,
    latch_transport_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    latch, transport = _validate_local_latch_transport(
        prepared.sources,
        prepared.authority,
        prepared.job_id,
        latch_path=latch_path,
        latch_transport_path=latch_transport_path,
    )
    _assert_current_worker(prepared, latch)
    return latch, transport


def publish_worker_start_latch_v2(
    *,
    sources: WorkerStartSourceBundle,
    authority: WorkerStartAuthorityInputs,
    outputs: WorkerStartLatchOutputs,
    services: WorkerStartPublisherServices,
) -> WorkerStartPublisherOutcome:
    if type(outputs) is not WorkerStartLatchOutputs:
        raise WorkerStartPublisherError("latch output type is invalid")
    _validate_output_pair(outputs.latch_path, outputs.latch_transport_path)
    prepared = _prepare(sources=sources, authority=authority, services=services)
    try:
        sampled = services.utc_now()
    except Exception as error:
        raise WorkerStartPublisherError("worker clock failed") from error
    if not isinstance(sampled, datetime) or sampled.tzinfo is None:
        raise WorkerStartPublisherError("worker clock is not timezone-aware")
    try:
        latch = prepared.sources.worker.build_worker_start_latch_v2(
            descriptor=prepared.authority.descriptor,
            intent=prepared.authority.intent,
            controller_injected_sky_job_id=prepared.job_id,
            instance_id=prepared.worker.instance_id,
            instance_type=prepared.worker.instance_type,
            image_id=prepared.worker.image_id,
            worker_role_arn=prepared.worker.worker_role_arn,
            instance_identity_document_sha256=(
                prepared.worker.identity_document_sha256
            ),
            ec2_pending_time=prepared.worker.pending_time,
            entrypoint_observed_at=sampled,
        )
        prepared.sources.worker.validate_worker_start_latch_v2(
            latch,
            descriptor=prepared.authority.descriptor,
            intent=prepared.authority.intent,
        )
        key = prepared.sources.worker.worker_start_latch_v2_s3_key(worker_latch=latch)
        raw = prepared.sources.worker.worker_v2_canonical_bytes(latch)
    except Exception as error:
        raise WorkerStartPublisherError("Task 3m latch authority failed") from error
    s3 = _AwsCliS3(
        services=services,
        cli=prepared.cli,
        bucket=prepared.authority.bucket,
        run_id=prepared.authority.run_id,
    )
    _audit_legacy(s3, prepared.authority)
    descriptor, body_path_name = tempfile.mkstemp(prefix="glm52-worker-latch-")
    os.fchmod(descriptor, 0o600)
    body_path = Path(body_path_name)
    try:
        written = 0
        while written < len(raw):
            count = os.write(descriptor, raw[written:])
            if count <= 0:
                raise WorkerStartPublisherError(
                    "worker latch temporary write was incomplete"
                )
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            body_details = body_path.lstat()
        except OSError as error:
            raise WorkerStartPublisherError(
                "worker latch temporary file vanished"
            ) from error
        if (
            not body_path.is_absolute()
            or stat.S_ISLNK(body_details.st_mode)
            or not stat.S_ISREG(body_details.st_mode)
            or stat.S_IMODE(body_details.st_mode) != 0o600
            or body_details.st_size != len(raw)
        ):
            raise WorkerStartPublisherError("worker latch temporary file is invalid")
        checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
        args = (
            "s3api",
            "put-object",
            "--bucket",
            prepared.authority.bucket,
            "--key",
            key,
            "--body",
            str(body_path),
            "--if-none-match",
            "*",
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            checksum,
            "--expected-bucket-owner",
            APPROVED_ACCOUNT_ID,
            "--content-type",
            "application/json",
            "--metadata",
            (
                f"glm52-run-id={prepared.authority.run_id},"
                f"glm52-body-sha256={latch['worker_latch_body_sha256']}"
            ),
        )
        published = False
        version: str | None = None
        try:
            response = _aws_json(
                services,
                prepared.cli,
                args,
                operation="PutObject",
                permit_ambiguous=True,
            )
            version_value = response.get("VersionId")
            etag_value = response.get("ETag")
            checksum_value = response.get("ChecksumSHA256")
            if (
                not isinstance(version_value, str)
                or _VERSION_ID.fullmatch(version_value) is None
                or not isinstance(etag_value, str)
                or _ETAG.fullmatch(etag_value) is None
                or checksum_value != checksum
            ):
                raise WorkerStartPublisherError(
                    "PutObject response transport is incomplete"
                )
            version = version_value
            published = True
        except _ClientError as error:
            if str(error) not in _CONFLICT_CODES:
                raise WorkerStartPublisherError(
                    "worker latch PUT failed definitively"
                ) from error
        except _AmbiguousTransport:
            pass
        if version is None:
            try:
                version = _resolve_current_version(s3, key=key)
            except _Unavailable as error:
                raise WorkerStartPublisherError(
                    "worker latch winner is unavailable"
                ) from error
        winner_raw, winner_transport = _read_version(
            s3,
            key=key,
            version_id=version,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            body_path.unlink()
        except FileNotFoundError:
            pass
    expected_metadata = {
        "glm52-run-id": prepared.authority.run_id,
        "glm52-body-sha256": str(latch["worker_latch_body_sha256"]),
    }
    if winner_raw != raw or winner_transport.metadata != expected_metadata:
        raise WorkerStartPublisherError("worker latch winner is foreign")
    transport_body = {
        "schema_version": 1,
        "record_type": "glm52_sky_worker_start_latch_transport_v1",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": prepared.authority.bucket,
        "run_id": prepared.authority.run_id,
        "intent_body_sha256": prepared.authority.intent["intent_body_sha256"],
        "sky_job_id": prepared.job_id,
        "instance_id": prepared.worker.instance_id,
        "worker_instance_type": prepared.worker.instance_type,
        "worker_image_id": prepared.worker.image_id,
        "worker_role_arn": prepared.worker.worker_role_arn,
        "ec2_pending_time": prepared.worker.pending_time,
        "instance_identity_document_sha256": (prepared.worker.identity_document_sha256),
        "worker_latch_key": key,
        "worker_latch_file_sha256": _sha256(raw),
        "worker_latch_body_sha256": latch["worker_latch_body_sha256"],
        "worker_latch_version_id": winner_transport.version_id,
        "worker_latch_etag": winner_transport.etag,
        "worker_latch_last_modified": winner_transport.last_modified,
        "content_length": winner_transport.content_length,
        "checksum_sha256": winner_transport.checksum_sha256,
        "checksum_type": winner_transport.checksum_type,
        "content_type": winner_transport.content_type,
        "metadata": winner_transport.metadata,
    }
    transport_record = _self_hashed(
        transport_body,
        digest_field="latch_transport_body_sha256",
    )
    _persist_exact(outputs.latch_path, raw, label="worker latch output")
    _persist_exact(
        outputs.latch_transport_path,
        _canonical_bytes(transport_record, newline=True),
        label="latch transport output",
    )
    return WorkerStartPublisherOutcome(
        status="latch-published" if published else "latch-idempotent",
        run_id=prepared.authority.run_id,
        instance_id=prepared.worker.instance_id,
        sky_job_id=prepared.job_id,
        worker_latch_key=key,
        worker_acceptance_key=None,
        worker_acceptance_body_sha256=None,
    )


def _acceptance_key(
    prepared: _Prepared,
    latch: Mapping[str, object],
) -> str:
    return (
        f"{_dynamic_prefix(prepared.authority)}worker-acceptances/"
        f"{latch['instance_id']}/{latch['worker_latch_body_sha256']}/"
        "WORKER_START_ACCEPTED.json"
    )


def _acceptance_prefix(prepared: _Prepared) -> str:
    return f"{_dynamic_prefix(prepared.authority)}worker-acceptances/"


def _monotonic_sample(services: WorkerStartPublisherServices) -> float:
    try:
        value = services.monotonic()
    except Exception as error:
        raise WorkerStartPublisherError("monotonic clock failed") from error
    if type(value) not in (int, float):
        raise WorkerStartPublisherError("monotonic clock is invalid")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise WorkerStartPublisherError("monotonic clock is invalid") from error
    if not math.isfinite(numeric):
        raise WorkerStartPublisherError("monotonic clock is invalid")
    return numeric


def _require_wait_budget(
    services: WorkerStartPublisherServices,
    *,
    deadline: float,
) -> float:
    current = _monotonic_sample(services)
    if current >= deadline:
        raise WorkerStartPublisherError("worker-start acceptance wait timed out")
    return current


def _poll_acceptance(
    *,
    prepared: _Prepared,
    s3: _AwsCliS3,
    latch: Mapping[str, object],
    latch_transport: Mapping[str, object],
    services: WorkerStartPublisherServices,
    wait_timeout_seconds: int,
    poll_interval_seconds: int,
) -> tuple[str, dict[str, object], bytes, _Transport]:
    key = _acceptance_key(prepared, latch)
    deadline = _monotonic_sample(services) + wait_timeout_seconds
    first_lookup = True
    while True:
        if first_lookup:
            first_lookup = False
        else:
            _require_wait_budget(services, deadline=deadline)
        try:
            version = _resolve_current_version(s3, key=key)
        except _Unavailable:
            current = _monotonic_sample(services)
            remaining = deadline - float(current)
            if remaining <= 0:
                raise WorkerStartPublisherError(
                    "worker-start acceptance wait timed out"
                )
            try:
                services.sleep(min(float(poll_interval_seconds), remaining))
            except Exception as error:
                raise WorkerStartPublisherError(
                    "acceptance poll sleep failed"
                ) from error
            continue
        _require_wait_budget(services, deadline=deadline)
        raw, transport = _read_version(s3, key=key, version_id=version)
        accepted = _decode_json(
            raw,
            label="worker acceptance",
            newline=False,
        )
        try:
            derived = prepared.sources.worker.worker_start_accepted_v2_s3_key(
                worker_start_accepted=accepted
            )
        except Exception as error:
            raise WorkerStartPublisherError(
                "worker acceptance marker is invalid"
            ) from error
        exact = {
            "account_id": APPROVED_ACCOUNT_ID,
            "region": APPROVED_REGION,
            "run_id": prepared.authority.run_id,
            "intent_body_sha256": prepared.authority.intent["intent_body_sha256"],
            "sky_job_id": prepared.job_id,
            "instance_id": latch["instance_id"],
            "worker_latch_key": (
                prepared.sources.worker.worker_start_latch_v2_s3_key(worker_latch=latch)
            ),
            "worker_latch_file_sha256": _sha256(
                prepared.sources.worker.worker_v2_canonical_bytes(latch)
            ),
            "worker_latch_body_sha256": latch["worker_latch_body_sha256"],
            "worker_latch_version_id": latch_transport["worker_latch_version_id"],
            "worker_latch_etag": latch_transport["worker_latch_etag"],
            "worker_latch_last_modified": latch_transport["worker_latch_last_modified"],
            "worker_instance_type": latch["instance_type"],
            "worker_image_id": latch["image_id"],
            "worker_instance_lifecycle": None,
        }
        drifted = [name for name, item in exact.items() if accepted.get(name) != item]
        expected_metadata = {
            "glm52-run-id": prepared.authority.run_id,
            "glm52-body-sha256": str(accepted.get("worker_acceptance_body_sha256")),
        }
        if (
            type(accepted.get("sky_job_id")) is not int
            or accepted.get("acceptance_kind") not in {"initial", "managed-recovery"}
            or transport.metadata != expected_metadata
        ):
            drifted.append("transport-or-scalar")
        if derived != key or drifted:
            raise WorkerStartPublisherError(
                "worker acceptance marker authority drifted"
                + (f": {','.join(drifted)}" if drifted else ": key")
            )
        _require_wait_budget(services, deadline=deadline)
        return key, accepted, raw, transport


def _coordinator_outcome(
    *,
    prepared: _Prepared,
    latch_transport: Mapping[str, object],
    s3: _AwsCliS3,
    services: WorkerStartPublisherServices,
    acceptance_key: str,
) -> object:
    if s3.records:
        raise WorkerStartPublisherError("coordinator replay S3 trace was not fresh")
    coordinator = prepared.sources.coordinator
    request = coordinator.WorkerStartCoordinatorRequest(
        account_id=APPROVED_ACCOUNT_ID,
        region=APPROVED_REGION,
        bucket=prepared.authority.bucket,
        descriptor_key=prepared.authority.descriptor_key,
        descriptor_file_sha256=_sha256(
            _canonical_bytes(prepared.authority.descriptor, newline=True)
        ),
        intent_key=prepared.authority.intent_key,
        intent_file_sha256=_sha256(
            _canonical_bytes(prepared.authority.intent, newline=True)
        ),
        intent_body_sha256=prepared.authority.intent["intent_body_sha256"],
        worker_latch_key=latch_transport["worker_latch_key"],
        worker_latch_version_id=latch_transport["worker_latch_version_id"],
    )
    coordinator_services = coordinator.WorkerStartCoordinatorServices(
        sts=_AwsCliSts(
            services,
            prepared.cli,
            prepared.worker.worker_role_arn,
        ),
        s3=s3,
        ec2=_NoCalls("EC2"),
        ssm=_NoCalls("SSM"),
        clock=services.utc_now,
        sleep=lambda _: (_ for _ in ()).throw(
            WorkerStartPublisherError("read-only replay attempted sleep")
        ),
    )
    try:
        outcome = coordinator.coordinate_worker_start_acceptance_v2(
            services=coordinator_services,
            request=request,
        )
    except WorkerStartPublisherError:
        raise
    except Exception as error:
        raise WorkerStartPublisherError(
            "frozen coordinator replay rejected authority"
        ) from error
    exact = {
        "status": "idempotent-complete",
        "decision_action": "idempotent-complete",
        "published_observation": False,
        "published_acceptance": False,
        "worker_latch_key": latch_transport["worker_latch_key"],
        "worker_acceptance_key": acceptance_key,
    }
    if any(getattr(outcome, name, object()) != item for name, item in exact.items()):
        raise WorkerStartPublisherError(
            "frozen coordinator did not return idempotent-complete"
        )
    required_keys = {
        prepared.authority.descriptor_key,
        prepared.authority.intent_key,
        str(latch_transport["worker_latch_key"]),
        acceptance_key,
    }
    if not required_keys.issubset(s3.records):
        raise WorkerStartPublisherError(
            "frozen coordinator replay omitted core authority reads"
        )
    latch_record = s3.records[str(latch_transport["worker_latch_key"])]
    latch_expected = {
        "file_sha256": latch_transport["worker_latch_file_sha256"],
        "body_sha256": latch_transport["worker_latch_body_sha256"],
        "version_id": latch_transport["worker_latch_version_id"],
        "etag": latch_transport["worker_latch_etag"],
        "last_modified": latch_transport["worker_latch_last_modified"],
        "content_length": latch_transport["content_length"],
        "checksum_sha256": latch_transport["checksum_sha256"],
        "checksum_type": latch_transport["checksum_type"],
        "content_type": latch_transport["content_type"],
        "metadata": latch_transport["metadata"],
        "trailing_newline": False,
    }
    if any(latch_record.get(name) != item for name, item in latch_expected.items()):
        raise WorkerStartPublisherError("frozen coordinator latch trace drifted")
    return outcome


def _post_replay_sweep(s3: _AwsCliS3) -> None:
    for key, record in sorted(s3.records.items()):
        try:
            current = _resolve_current_version(s3, key=key)
        except _Unavailable as error:
            raise WorkerStartPublisherError(
                "source disappeared during post-replay sweep"
            ) from error
        if current != record["version_id"]:
            raise WorkerStartPublisherError(
                "source current version drifted during replay"
            )


def _require_polled_acceptance_trace(
    *,
    s3: _AwsCliS3,
    acceptance_key: str,
    accepted: Mapping[str, object],
    accepted_raw: bytes,
    acceptance_transport: _Transport,
    run_id: str,
) -> None:
    record = s3.records.get(acceptance_key)
    expected = {
        "key": acceptance_key,
        "file_sha256": _sha256(accepted_raw),
        "body_sha256": accepted["worker_acceptance_body_sha256"],
        "version_id": acceptance_transport.version_id,
        "etag": acceptance_transport.etag,
        "last_modified": acceptance_transport.last_modified,
        "content_length": acceptance_transport.content_length,
        "checksum_sha256": acceptance_transport.checksum_sha256,
        "checksum_type": acceptance_transport.checksum_type,
        "content_type": acceptance_transport.content_type,
        "metadata": {
            "glm52-run-id": run_id,
            "glm52-body-sha256": accepted["worker_acceptance_body_sha256"],
        },
        "trailing_newline": False,
    }
    if record != expected:
        raise WorkerStartPublisherError(
            "coordinator acceptance trace drifted from deterministic poll"
        )


def _receipt(
    *,
    prepared: _Prepared,
    latch: Mapping[str, object],
    latch_transport: Mapping[str, object],
    accepted: Mapping[str, object],
    acceptance_key: str,
    acceptance_raw: bytes,
    acceptance_transport: _Transport,
    source_records: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_worker_admission_receipt_v1",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": prepared.authority.bucket,
        "run_id": prepared.authority.run_id,
        "intent_body_sha256": prepared.authority.intent["intent_body_sha256"],
        "sky_job_id": prepared.job_id,
        "instance_id": prepared.worker.instance_id,
        "worker_instance_type": prepared.worker.instance_type,
        "worker_image_id": prepared.worker.image_id,
        "worker_role_arn": prepared.worker.worker_role_arn,
        "ec2_pending_time": prepared.worker.pending_time,
        "instance_identity_document_sha256": (prepared.worker.identity_document_sha256),
        "worker_latch_key": latch_transport["worker_latch_key"],
        "worker_latch_file_sha256": latch_transport["worker_latch_file_sha256"],
        "worker_latch_body_sha256": latch["worker_latch_body_sha256"],
        "worker_latch_version_id": latch_transport["worker_latch_version_id"],
        "worker_latch_etag": latch_transport["worker_latch_etag"],
        "worker_latch_last_modified": latch_transport["worker_latch_last_modified"],
        "worker_acceptance_key": acceptance_key,
        "worker_acceptance_file_sha256": _sha256(acceptance_raw),
        "worker_acceptance_body_sha256": accepted["worker_acceptance_body_sha256"],
        "worker_acceptance_version_id": acceptance_transport.version_id,
        "worker_acceptance_etag": acceptance_transport.etag,
        "worker_acceptance_last_modified": acceptance_transport.last_modified,
        "source_transports": [
            dict(source_records[key]) for key in sorted(source_records)
        ],
        "legacy_forbidden_basenames": list(_FORBIDDEN_LEGACY),
    }
    return _self_hashed(body, digest_field="admission_receipt_body_sha256")


def _wait_values(
    wait_timeout_seconds: object,
    poll_interval_seconds: object,
) -> tuple[int, int]:
    if (
        type(wait_timeout_seconds) is not int
        or wait_timeout_seconds <= 0
        or wait_timeout_seconds > 900
        or type(poll_interval_seconds) is not int
        or poll_interval_seconds <= 0
        or poll_interval_seconds > 30
        or poll_interval_seconds > wait_timeout_seconds
    ):
        raise WorkerStartPublisherError("acceptance wait bounds are invalid")
    return wait_timeout_seconds, poll_interval_seconds


def replay_worker_start_accepted_v2(
    *,
    sources: WorkerStartSourceBundle,
    authority: WorkerStartAuthorityInputs,
    latch_path: Path,
    latch_transport_path: Path,
    outputs: WorkerStartAdmissionOutputs,
    services: WorkerStartPublisherServices,
    wait_timeout_seconds: int = 600,
    poll_interval_seconds: int = 5,
) -> WorkerStartPublisherOutcome:
    wait_timeout_seconds, poll_interval_seconds = _wait_values(
        wait_timeout_seconds,
        poll_interval_seconds,
    )
    if type(outputs) is not WorkerStartAdmissionOutputs:
        raise WorkerStartPublisherError("admission output type is invalid")
    _validate_output_pair(outputs.accepted_path, outputs.admission_receipt_path)
    loaded, local, job_id, exact_services = _prepare_local(
        sources=sources,
        authority=authority,
        services=services,
    )
    latch, latch_transport = _validate_local_latch_transport(
        loaded,
        local,
        job_id,
        latch_path=latch_path,
        latch_transport_path=latch_transport_path,
    )
    _require_distinct_paths(
        latch_path,
        latch_transport_path,
        outputs.accepted_path,
        outputs.admission_receipt_path,
    )
    prepared = _prepare_external(
        loaded=loaded,
        local=local,
        job_id=job_id,
        services=exact_services,
    )
    _assert_current_worker(prepared, latch)
    acceptance_key = _acceptance_key(prepared, latch)
    poll_s3 = _AwsCliS3(
        services=services,
        cli=prepared.cli,
        bucket=prepared.authority.bucket,
        run_id=prepared.authority.run_id,
        required_acceptance_key=acceptance_key,
        acceptance_prefix=_acceptance_prefix(prepared),
    )
    _audit_legacy(poll_s3, prepared.authority)
    (
        acceptance_key,
        accepted,
        accepted_raw,
        accepted_transport,
    ) = _poll_acceptance(
        prepared=prepared,
        s3=poll_s3,
        latch=latch,
        latch_transport=latch_transport,
        services=services,
        wait_timeout_seconds=wait_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    replay_s3 = _AwsCliS3(
        services=services,
        cli=prepared.cli,
        bucket=prepared.authority.bucket,
        run_id=prepared.authority.run_id,
        required_acceptance_key=acceptance_key,
        acceptance_prefix=_acceptance_prefix(prepared),
    )
    outcome = _coordinator_outcome(
        prepared=prepared,
        latch_transport=latch_transport,
        s3=replay_s3,
        services=services,
        acceptance_key=acceptance_key,
    )
    if getattr(outcome, "worker_acceptance_body_sha256", None) != accepted.get(
        "worker_acceptance_body_sha256"
    ):
        raise WorkerStartPublisherError("coordinator acceptance digest drifted")
    _require_polled_acceptance_trace(
        s3=replay_s3,
        acceptance_key=acceptance_key,
        accepted=accepted,
        accepted_raw=accepted_raw,
        acceptance_transport=accepted_transport,
        run_id=prepared.authority.run_id,
    )
    _post_replay_sweep(replay_s3)
    _audit_legacy(replay_s3, prepared.authority)
    receipt = _receipt(
        prepared=prepared,
        latch=latch,
        latch_transport=latch_transport,
        accepted=accepted,
        acceptance_key=acceptance_key,
        acceptance_raw=accepted_raw,
        acceptance_transport=accepted_transport,
        source_records=replay_s3.records,
    )
    kind = accepted.get("acceptance_kind")
    statuses = {
        "initial": "accepted-initial",
        "managed-recovery": "accepted-managed-recovery",
    }
    if kind not in statuses:
        raise WorkerStartPublisherError("worker acceptance kind is invalid")
    _persist_exact(
        outputs.accepted_path,
        accepted_raw,
        label="worker acceptance output",
    )
    _persist_exact(
        outputs.admission_receipt_path,
        _canonical_bytes(receipt, newline=True),
        label="admission receipt output",
    )
    return WorkerStartPublisherOutcome(
        status=statuses[kind],
        run_id=prepared.authority.run_id,
        instance_id=prepared.worker.instance_id,
        sky_job_id=prepared.job_id,
        worker_latch_key=str(latch_transport["worker_latch_key"]),
        worker_acceptance_key=acceptance_key,
        worker_acceptance_body_sha256=str(accepted["worker_acceptance_body_sha256"]),
    )


def _load_receipt(
    path: Path,
) -> tuple[dict[str, object], bytes]:
    receipt_file = _require_regular_path(path, label="admission receipt")
    if stat.S_IMODE(receipt_file.lstat().st_mode) != 0o600:
        raise WorkerStartPublisherError("admission receipt mode drifted")
    raw = _read_stable_regular(
        receipt_file,
        label="admission receipt",
        required_mode=0o600,
    )
    receipt = _decode_json(raw, label="admission receipt", newline=True)
    _validate_self_hashed(
        receipt,
        fields=_RECEIPT_FIELDS,
        digest_field="admission_receipt_body_sha256",
        record_type="glm52_sky_worker_admission_receipt_v1",
        label="admission receipt",
    )
    sources = receipt.get("source_transports")
    if type(sources) is not list or not sources:
        raise WorkerStartPublisherError("receipt source inventory is invalid")
    keys: list[str] = []
    for source in sources:
        if type(source) is not dict or set(source) != _SOURCE_TRANSPORT_FIELDS:
            raise WorkerStartPublisherError("receipt source transport is invalid")
        _validate_source_transport(source, run_id=receipt.get("run_id"))
        key = source.get("key")
        assert isinstance(key, str)
        keys.append(key)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise WorkerStartPublisherError(
            "receipt source inventory is not sorted and unique"
        )
    if receipt.get("legacy_forbidden_basenames") != list(_FORBIDDEN_LEGACY):
        raise WorkerStartPublisherError("receipt legacy exclusion drifted")
    if (
        type(receipt.get("sky_job_id")) is not int
        or not isinstance(receipt.get("worker_acceptance_version_id"), str)
        or _VERSION_ID.fullmatch(str(receipt["worker_acceptance_version_id"])) is None
        or not isinstance(receipt.get("worker_acceptance_etag"), str)
        or _ETAG.fullmatch(str(receipt["worker_acceptance_etag"])) is None
        or not isinstance(receipt.get("worker_latch_version_id"), str)
        or _VERSION_ID.fullmatch(str(receipt["worker_latch_version_id"])) is None
        or not isinstance(receipt.get("worker_latch_etag"), str)
        or _ETAG.fullmatch(str(receipt["worker_latch_etag"])) is None
    ):
        raise WorkerStartPublisherError("receipt scalar authority is invalid")
    for name in (
        "intent_body_sha256",
        "instance_identity_document_sha256",
        "worker_latch_file_sha256",
        "worker_latch_body_sha256",
        "worker_acceptance_file_sha256",
        "worker_acceptance_body_sha256",
    ):
        _require_digest(receipt.get(name), label=f"receipt {name}")
    _parse_canonical_time(
        receipt.get("worker_latch_last_modified"),
        label="receipt latch LastModified",
    )
    _parse_canonical_time(
        receipt.get("worker_acceptance_last_modified"),
        label="receipt acceptance LastModified",
    )
    return receipt, raw


def _validate_source_transport(
    source: Mapping[str, object],
    *,
    run_id: object,
) -> None:
    key = source.get("key")
    file_sha = _require_digest(
        source.get("file_sha256"),
        label="receipt source file SHA-256",
    )
    body_sha = _require_digest(
        source.get("body_sha256"),
        label="receipt source body SHA-256",
    )
    del file_sha
    version = source.get("version_id")
    etag = source.get("etag")
    length = source.get("content_length")
    checksum = source.get("checksum_sha256")
    metadata = source.get("metadata")
    if (
        not isinstance(key, str)
        or not key
        or not isinstance(version, str)
        or _VERSION_ID.fullmatch(version) is None
        or not isinstance(etag, str)
        or _ETAG.fullmatch(etag) is None
        or type(length) is not int
        or length <= 0
        or not isinstance(checksum, str)
        or source.get("checksum_type") != "FULL_OBJECT"
        or source.get("content_type") != "application/json"
        or source.get("trailing_newline") not in (True, False)
        or type(source.get("trailing_newline")) is not bool
        or metadata
        != {
            "glm52-run-id": run_id,
            "glm52-body-sha256": body_sha,
        }
    ):
        raise WorkerStartPublisherError("receipt source transport is invalid")
    try:
        decoded = base64.b64decode(checksum, validate=True)
    except (ValueError, binascii.Error) as error:
        raise WorkerStartPublisherError("receipt source checksum is invalid") from error
    if len(decoded) != 32 or base64.b64encode(decoded).decode("ascii") != checksum:
        raise WorkerStartPublisherError("receipt source checksum is invalid")
    _parse_canonical_time(
        source.get("last_modified"),
        label="receipt source LastModified",
    )


def _load_local_acceptance(
    loaded: _LoadedSources,
    path: Path,
) -> tuple[dict[str, object], bytes, str]:
    accepted_file = _require_regular_path(
        path,
        label="local worker acceptance",
    )
    if stat.S_IMODE(accepted_file.lstat().st_mode) != 0o600:
        raise WorkerStartPublisherError("local worker acceptance mode drifted")
    accepted_raw = _read_stable_regular(
        accepted_file,
        label="local worker acceptance",
        required_mode=0o600,
    )
    accepted = _decode_json(
        accepted_raw,
        label="local worker acceptance",
        newline=False,
    )
    try:
        accepted_key = loaded.worker.worker_start_accepted_v2_s3_key(
            worker_start_accepted=accepted
        )
    except Exception as error:
        raise WorkerStartPublisherError("local worker acceptance is invalid") from error
    return accepted, accepted_raw, accepted_key


def _validate_local_receipt_identity(
    *,
    local: _LocalAuthority,
    job_id: int,
    latch: Mapping[str, object],
    latch_transport: Mapping[str, object],
    accepted: Mapping[str, object],
    accepted_raw: bytes,
    accepted_key: str,
    receipt: Mapping[str, object],
) -> None:
    accepted_exact = {
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "run_id": local.run_id,
        "managed_mode": QUALIFICATION_MODE,
        "intent_body_sha256": local.intent["intent_body_sha256"],
        "sky_job_id": job_id,
        "instance_id": latch["instance_id"],
        "worker_instance_type": latch["instance_type"],
        "worker_image_id": latch["image_id"],
        "worker_instance_lifecycle": None,
        "worker_latch_key": latch_transport["worker_latch_key"],
        "worker_latch_file_sha256": latch_transport["worker_latch_file_sha256"],
        "worker_latch_body_sha256": latch["worker_latch_body_sha256"],
        "worker_latch_version_id": latch_transport["worker_latch_version_id"],
        "worker_latch_etag": latch_transport["worker_latch_etag"],
        "worker_latch_last_modified": latch_transport["worker_latch_last_modified"],
    }
    if (
        type(accepted.get("sky_job_id")) is not int
        or accepted.get("acceptance_kind") not in {"initial", "managed-recovery"}
        or any(accepted.get(name) != item for name, item in accepted_exact.items())
    ):
        raise WorkerStartPublisherError("local acceptance authority drifted")
    receipt_exact = {
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": local.bucket,
        "run_id": local.run_id,
        "intent_body_sha256": local.intent["intent_body_sha256"],
        "sky_job_id": job_id,
        "instance_id": latch["instance_id"],
        "worker_instance_type": latch["instance_type"],
        "worker_image_id": latch["image_id"],
        "worker_role_arn": latch["worker_role_arn"],
        "ec2_pending_time": latch["ec2_pending_time"],
        "instance_identity_document_sha256": (
            latch["instance_identity_document_sha256"]
        ),
        "worker_latch_key": latch_transport["worker_latch_key"],
        "worker_latch_file_sha256": latch_transport["worker_latch_file_sha256"],
        "worker_latch_body_sha256": latch["worker_latch_body_sha256"],
        "worker_latch_version_id": latch_transport["worker_latch_version_id"],
        "worker_latch_etag": latch_transport["worker_latch_etag"],
        "worker_latch_last_modified": latch_transport["worker_latch_last_modified"],
        "worker_acceptance_key": accepted_key,
        "worker_acceptance_file_sha256": _sha256(accepted_raw),
        "worker_acceptance_body_sha256": accepted["worker_acceptance_body_sha256"],
    }
    if type(receipt.get("sky_job_id")) is not int or any(
        receipt.get(name) != item for name, item in receipt_exact.items()
    ):
        raise WorkerStartPublisherError("local receipt authority drifted")
    sources = {
        str(item["key"]): dict(item)
        for item in receipt["source_transports"]
        if isinstance(item, dict)
    }
    latch_source = {
        "key": latch_transport["worker_latch_key"],
        "file_sha256": latch_transport["worker_latch_file_sha256"],
        "body_sha256": latch_transport["worker_latch_body_sha256"],
        "version_id": latch_transport["worker_latch_version_id"],
        "etag": latch_transport["worker_latch_etag"],
        "last_modified": latch_transport["worker_latch_last_modified"],
        "content_length": latch_transport["content_length"],
        "checksum_sha256": latch_transport["checksum_sha256"],
        "checksum_type": latch_transport["checksum_type"],
        "content_type": latch_transport["content_type"],
        "metadata": latch_transport["metadata"],
        "trailing_newline": False,
    }
    if sources.get(str(latch_transport["worker_latch_key"])) != latch_source:
        raise WorkerStartPublisherError("receipt latch source authority drifted")
    accepted_source = sources.get(accepted_key)
    expected_acceptance_source = {
        "key": accepted_key,
        "file_sha256": _sha256(accepted_raw),
        "body_sha256": accepted["worker_acceptance_body_sha256"],
        "version_id": receipt["worker_acceptance_version_id"],
        "etag": receipt["worker_acceptance_etag"],
        "last_modified": receipt["worker_acceptance_last_modified"],
        "content_length": len(accepted_raw),
        "checksum_sha256": base64.b64encode(
            hashlib.sha256(accepted_raw).digest()
        ).decode("ascii"),
        "checksum_type": "FULL_OBJECT",
        "content_type": "application/json",
        "metadata": {
            "glm52-run-id": local.run_id,
            "glm52-body-sha256": accepted["worker_acceptance_body_sha256"],
        },
        "trailing_newline": False,
    }
    if accepted_source != expected_acceptance_source:
        raise WorkerStartPublisherError("receipt acceptance source authority drifted")


def _record_matches(
    expected: Mapping[str, object], actual: Mapping[str, object]
) -> bool:
    return dict(expected) == dict(actual)


def verify_worker_start_receipt_v2(
    *,
    sources: WorkerStartSourceBundle,
    authority: WorkerStartAuthorityInputs,
    latch_path: Path,
    latch_transport_path: Path,
    accepted_path: Path,
    admission_receipt_path: Path,
    services: WorkerStartPublisherServices,
) -> WorkerStartPublisherOutcome:
    loaded, local, job_id, exact_services = _prepare_local(
        sources=sources,
        authority=authority,
        services=services,
    )
    _require_distinct_paths(
        latch_path,
        latch_transport_path,
        accepted_path,
        admission_receipt_path,
    )
    latch, latch_transport = _validate_local_latch_transport(
        loaded,
        local,
        job_id,
        latch_path=latch_path,
        latch_transport_path=latch_transport_path,
    )
    accepted, accepted_raw, accepted_key = _load_local_acceptance(
        loaded,
        accepted_path,
    )
    receipt, receipt_raw = _load_receipt(admission_receipt_path)
    del receipt_raw
    _validate_local_receipt_identity(
        local=local,
        job_id=job_id,
        latch=latch,
        latch_transport=latch_transport,
        accepted=accepted,
        accepted_raw=accepted_raw,
        accepted_key=accepted_key,
        receipt=receipt,
    )
    # External authority starts only after every local object is canonical
    # and cross-bound to the same descriptor, intent, latch, and acceptance.
    prepared = _prepare_external(
        loaded=loaded,
        local=local,
        job_id=job_id,
        services=exact_services,
    )
    cli = prepared.cli
    _assert_current_worker(prepared, latch, receipt)
    verification_s3 = _AwsCliS3(
        services=services,
        cli=cli,
        bucket=local.bucket,
        run_id=local.run_id,
        required_acceptance_key=accepted_key,
        acceptance_prefix=_acceptance_prefix(prepared),
    )
    _audit_legacy(verification_s3, local)
    expected_sources = {
        str(item["key"]): dict(item)
        for item in receipt["source_transports"]
        if isinstance(item, dict)
    }
    for key, expected in sorted(expected_sources.items()):
        try:
            current = _resolve_current_version(verification_s3, key=key)
        except _Unavailable as error:
            raise WorkerStartPublisherError(
                "receipt source current version is unavailable"
            ) from error
        if current != expected["version_id"]:
            raise WorkerStartPublisherError("receipt source current version drifted")
        raw, transport = _read_version(
            verification_s3,
            key=key,
            version_id=str(expected["version_id"]),
        )
        actual = verification_s3.records.get(key)
        if actual is None or not _record_matches(expected, actual):
            raise WorkerStartPublisherError("receipt source transport drifted")
        if _sha256(raw) != expected["file_sha256"] or (
            transport.metadata.get("glm52-body-sha256") != expected["body_sha256"]
        ):
            raise WorkerStartPublisherError("receipt source byte identity drifted")
    if verification_s3.records != expected_sources:
        raise WorkerStartPublisherError("receipt verification source inventory drifted")
    replay_s3 = _AwsCliS3(
        services=services,
        cli=cli,
        bucket=local.bucket,
        run_id=local.run_id,
        required_acceptance_key=accepted_key,
        acceptance_prefix=_acceptance_prefix(prepared),
    )
    outcome = _coordinator_outcome(
        prepared=prepared,
        latch_transport=latch_transport,
        s3=replay_s3,
        services=services,
        acceptance_key=accepted_key,
    )
    del outcome
    if replay_s3.records != expected_sources:
        raise WorkerStartPublisherError(
            "fresh coordinator source inventory drifted from receipt"
        )
    _audit_legacy(replay_s3, local)
    return WorkerStartPublisherOutcome(
        status="receipt-verified",
        run_id=local.run_id,
        instance_id=prepared.worker.instance_id,
        sky_job_id=job_id,
        worker_latch_key=str(latch_transport["worker_latch_key"]),
        worker_acceptance_key=accepted_key,
        worker_acceptance_body_sha256=str(accepted["worker_acceptance_body_sha256"]),
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        del request, file_pointer, code, message, headers, new_url
        return None


def _default_imds_opener(
    request: urllib.request.Request,
    timeout: float,
) -> bytes:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.geturl() != request.full_url:
                raise WorkerStartPublisherError("IMDSv2 response endpoint drifted")
            value = response.read()
    except WorkerStartPublisherError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise WorkerStartPublisherError("IMDSv2 request failed") from error
    if not isinstance(value, bytes):
        raise WorkerStartPublisherError("IMDSv2 response was not bytes")
    return value


def _services() -> WorkerStartPublisherServices:
    return WorkerStartPublisherServices(
        command_runner=_default_command_runner,
        imds_opener=_default_imds_opener,
        utc_now=lambda: datetime.now(timezone.utc),
        monotonic=time.monotonic,
        sleep=time.sleep,
        environ=dict(os.environ),
    )


def _positive_cli_integer(value: str) -> int:
    if len(value) > 4300 or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("must be a canonical positive integer")
    try:
        parsed = int(value)
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("integer cannot be represented") from error
    if str(parsed) != value:
        raise argparse.ArgumentTypeError("integer is not canonical")
    return parsed


class _RedactedArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        del message
        raise WorkerStartPublisherError("worker-start publisher usage is invalid")


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publisher-file-sha256", required=True)
    parser.add_argument("--campaign-policy", required=True, type=Path)
    parser.add_argument("--campaign-policy-file-sha256", required=True)
    parser.add_argument("--campaign-policy-native", required=True, type=Path)
    parser.add_argument("--campaign-policy-native-file-sha256", required=True)
    parser.add_argument("--must-start-policy", required=True, type=Path)
    parser.add_argument("--must-start-policy-file-sha256", required=True)
    parser.add_argument("--must-start-policy-native", required=True, type=Path)
    parser.add_argument("--must-start-policy-native-file-sha256", required=True)
    parser.add_argument("--dynamic-policy", required=True, type=Path)
    parser.add_argument("--dynamic-policy-file-sha256", required=True)
    parser.add_argument("--worker-policy", required=True, type=Path)
    parser.add_argument("--worker-policy-file-sha256", required=True)
    parser.add_argument("--coordinator", required=True, type=Path)
    parser.add_argument("--coordinator-file-sha256", required=True)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--descriptor-s3-uri", required=True)
    parser.add_argument("--descriptor-file-sha256", required=True)
    parser.add_argument("--intent", required=True, type=Path)
    parser.add_argument("--intent-s3-uri", required=True)
    parser.add_argument("--intent-file-sha256", required=True)
    parser.add_argument("--intent-body-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = _RedactedArgumentParser()
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_RedactedArgumentParser,
    )
    publish = commands.add_parser("publish-latch-v2")
    replay = commands.add_parser("replay-accepted-v2")
    verify = commands.add_parser("verify-receipt-v2")
    for child in (publish, replay, verify):
        _common_arguments(child)
    publish.add_argument("--latch-output", required=True, type=Path)
    publish.add_argument("--latch-transport-output", required=True, type=Path)
    replay.add_argument("--latch", required=True, type=Path)
    replay.add_argument("--latch-transport", required=True, type=Path)
    replay.add_argument("--accepted-output", required=True, type=Path)
    replay.add_argument(
        "--admission-receipt-output",
        required=True,
        type=Path,
    )
    replay.add_argument(
        "--wait-timeout-seconds",
        type=_positive_cli_integer,
        default=600,
    )
    replay.add_argument(
        "--poll-interval-seconds",
        type=_positive_cli_integer,
        default=5,
    )
    verify.add_argument("--latch", required=True, type=Path)
    verify.add_argument("--latch-transport", required=True, type=Path)
    verify.add_argument("--accepted", required=True, type=Path)
    verify.add_argument("--admission-receipt", required=True, type=Path)
    return parser


def _cli_objects(
    args: argparse.Namespace,
) -> tuple[WorkerStartSourceBundle, WorkerStartAuthorityInputs]:
    sources = WorkerStartSourceBundle(
        publisher_file_sha256=args.publisher_file_sha256,
        campaign_policy_path=args.campaign_policy,
        campaign_policy_file_sha256=args.campaign_policy_file_sha256,
        campaign_policy_native_path=args.campaign_policy_native,
        campaign_policy_native_file_sha256=(
            args.campaign_policy_native_file_sha256
        ),
        must_start_policy_path=args.must_start_policy,
        must_start_policy_file_sha256=args.must_start_policy_file_sha256,
        must_start_policy_native_path=args.must_start_policy_native,
        must_start_policy_native_file_sha256=(
            args.must_start_policy_native_file_sha256
        ),
        dynamic_policy_path=args.dynamic_policy,
        dynamic_policy_file_sha256=args.dynamic_policy_file_sha256,
        worker_policy_path=args.worker_policy,
        worker_policy_file_sha256=args.worker_policy_file_sha256,
        coordinator_path=args.coordinator,
        coordinator_file_sha256=args.coordinator_file_sha256,
    )
    authority = WorkerStartAuthorityInputs(
        descriptor_path=args.descriptor,
        descriptor_s3_uri=args.descriptor_s3_uri,
        descriptor_file_sha256=args.descriptor_file_sha256,
        intent_path=args.intent,
        intent_s3_uri=args.intent_s3_uri,
        intent_file_sha256=args.intent_file_sha256,
        intent_body_sha256=args.intent_body_sha256,
    )
    return sources, authority


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sources, authority = _cli_objects(args)
    services = _services()
    if args.command == "publish-latch-v2":
        outcome = publish_worker_start_latch_v2(
            sources=sources,
            authority=authority,
            outputs=WorkerStartLatchOutputs(
                latch_path=args.latch_output,
                latch_transport_path=args.latch_transport_output,
            ),
            services=services,
        )
    elif args.command == "replay-accepted-v2":
        outcome = replay_worker_start_accepted_v2(
            sources=sources,
            authority=authority,
            latch_path=args.latch,
            latch_transport_path=args.latch_transport,
            outputs=WorkerStartAdmissionOutputs(
                accepted_path=args.accepted_output,
                admission_receipt_path=args.admission_receipt_output,
            ),
            services=services,
            wait_timeout_seconds=args.wait_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    else:
        outcome = verify_worker_start_receipt_v2(
            sources=sources,
            authority=authority,
            latch_path=args.latch,
            latch_transport_path=args.latch_transport,
            accepted_path=args.accepted,
            admission_receipt_path=args.admission_receipt,
            services=services,
        )
    sys.stdout.buffer.write(_canonical_bytes(asdict(outcome), newline=True))
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except WorkerStartPublisherError as error:
        message = str(error)
        if message == "worker-start acceptance wait timed out":
            sys.stderr.write(message + "\n")
            return 75
        sys.stderr.write(message + "\n")
        return 2
    except Exception:
        sys.stderr.write("worker-start publisher failed closed\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
