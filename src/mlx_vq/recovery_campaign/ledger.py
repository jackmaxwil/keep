"""Durable hash-chained recovery campaign ledger and evidence registry.

A valid bare chain proves the integrity and order of the events that remain in
the file. It cannot intrinsically detect deletion of a clean, newline-complete
tail. Controllers and handoffs that need that guarantee must seal and later
supply ``expected_head_sha256`` and/or ``expected_event_count`` to
``load_ledger``.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from .models import (
    EvidenceRecord,
    FrozenJsonDict,
    FrozenJsonList,
    LedgerEvent,
    VerificationResult,
)


SCHEMA_VERSION = 1
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "timestamp",
        "event_kind",
        "experiment",
        "payload",
        "previous_event_sha256",
        "event_sha256",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "artifact_path",
        "content_sha256",
        "manifest_identity_sha256",
        "candidate_identity_sha256",
        "parent_identity_sha256",
        "baseline_identity_sha256",
        "evidence_class",
        "release_eligible",
        "recovery_levers",
        "argv",
        "verification_results",
    }
)
_VERIFICATION_FIELDS = frozenset(
    {"name", "argv", "exit_code", "stdout_sha256", "stderr_sha256"}
)
_RETRY_AUTHORIZATION_FIELDS = frozenset(
    {
        "campaign",
        "experiment",
        "transition",
        "terminal_event_sha256",
        "exit_code",
        "reason",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_OPEN_BASE_FLAGS = os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
_ASCII_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class LedgerError(ValueError):
    """Raised when a ledger or evidence record fails closed validation."""


class LedgerUncertainCommitError(LedgerError):
    """Raised after publication when directory durability cannot be proven."""


class LedgerPublicationSourceError(LedgerError):
    """Raised before publication when the staged source no longer matches its fd."""


class LedgerCorruptPublicationError(LedgerError):
    """Raised when a public final exists but does not match the verified source."""


class _DuplicateKey(ValueError):
    pass


def _reject_constant(value: str) -> NoReturn:
    raise LedgerError(f"nonfinite JSON value {value!r} is forbidden")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise LedgerError(f"value is not finite canonical JSON: {exc}") from exc


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_thaw_json(item) for item in value]
    return value


def _validate_json_value(value: object, label: str = "payload") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LedgerError(f"{label} contains a nonfinite number")
        return
    if isinstance(value, (list, FrozenJsonList)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{label}[{index}]")
        return
    if isinstance(value, (dict, FrozenJsonDict)):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LedgerError(f"{label} JSON object keys must be strings")
            _validate_json_value(item, f"{label}.{key}")
        return
    raise LedgerError(f"{label} contains a non-JSON value of type {type(value).__name__}")


def _copy_json_object(value: object, label: str = "payload") -> dict[str, object]:
    if not isinstance(value, dict):
        raise LedgerError(f"{label} must be a JSON object")
    _validate_json_value(value, label)
    # Canonical encode/decode both detaches caller-owned containers and enforces
    # the exact JSON data model used for hashing.
    return json.loads(_canonical_json(value))


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise LedgerError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _require_hash(value: object, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LedgerError(f"{label} must be a lowercase SHA-256 hex digest")


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or _RFC3339_UTC_RE.fullmatch(value) is None:
        raise LedgerError("timestamp must be canonical RFC3339 UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise LedgerError("timestamp must be a valid RFC3339 UTC instant") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise LedgerError("timestamp must be UTC")
    return value


def _require_event_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{label} must be a nonempty string")
    if _ASCII_CONTROL_RE.search(value) is not None:
        raise LedgerError(f"{label} must not contain ASCII control characters")
    return value


def _timestamp_value(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise LedgerError("timestamp datetime must be timezone-aware")
        value = value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
    return _validate_timestamp(value)


def _event_body(event: LedgerEvent) -> dict[str, object]:
    return {
        "schema_version": event.schema_version,
        "sequence": event.sequence,
        "timestamp": event.timestamp,
        "event_kind": event.event_kind,
        "experiment": event.experiment,
        "payload": event.payload,
        "previous_event_sha256": event.previous_event_sha256,
    }


def _event_mapping(event: LedgerEvent) -> dict[str, object]:
    return {**_event_body(event), "event_sha256": event.event_sha256}


def _validate_event_payload(
    event_kind: str,
    experiment: str,
    payload: Mapping[str, object],
) -> None:
    if event_kind != "retry_authorized":
        return
    _require_exact_fields(
        payload,
        _RETRY_AUTHORIZATION_FIELDS,
        "retry_authorized payload",
    )
    _require_event_text(payload["campaign"], "retry_authorized campaign")
    payload_experiment = _require_event_text(
        payload["experiment"],
        "retry_authorized experiment",
    )
    if payload_experiment != experiment:
        raise LedgerError(
            "retry_authorized payload experiment must match event experiment"
        )
    _require_event_text(payload["transition"], "retry_authorized transition")
    _require_hash(
        payload["terminal_event_sha256"],
        "retry_authorized terminal_event_sha256",
    )
    if type(payload["exit_code"]) is not int:
        raise LedgerError("retry_authorized exit_code must be an integer")
    _require_event_text(payload["reason"], "retry_authorized reason")


def _validate_standalone_event(event: LedgerEvent) -> None:
    if not isinstance(event, LedgerEvent):
        raise LedgerError("event must be a LedgerEvent")
    if type(event.schema_version) is not int or event.schema_version != SCHEMA_VERSION:
        raise LedgerError(f"schema_version must be exact integer {SCHEMA_VERSION}")
    if type(event.sequence) is not int or event.sequence < 1:
        raise LedgerError("sequence must be a positive integer")
    _validate_timestamp(event.timestamp)
    _require_event_text(event.event_kind, "event_kind")
    _require_event_text(event.experiment, "experiment")
    if not isinstance(event.payload, Mapping):
        raise LedgerError("payload must be a JSON object")
    _validate_json_value(event.payload)
    _validate_event_payload(event.event_kind, event.experiment, event.payload)
    _require_hash(
        event.previous_event_sha256,
        "previous_event_sha256",
        optional=True,
    )
    if event.sequence == 1 and event.previous_event_sha256 is not None:
        raise LedgerError("sequence 1 previous_event_sha256 must be null")
    if event.sequence > 1 and event.previous_event_sha256 is None:
        raise LedgerError("sequence after 1 requires previous_event_sha256")
    _require_hash(event.event_sha256, "event_sha256")
    computed = hashlib.sha256(_canonical_json(_event_body(event))).hexdigest()
    if event.event_sha256 != computed:
        raise LedgerError(
            "event_sha256 body mismatch: "
            f"recorded {event.event_sha256}, computed {computed}"
        )


def _parse_event(
    raw_line: bytes, expected_sequence: int, previous_hash: str | None
) -> LedgerEvent:
    try:
        text = raw_line.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LedgerError("ledger contains invalid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateKey as exc:
        raise LedgerError(str(exc)) from exc
    except (json.JSONDecodeError, LedgerError) as exc:
        if isinstance(exc, LedgerError):
            raise
        raise LedgerError(f"ledger line is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LedgerError("ledger event must be a JSON object")
    _require_exact_fields(value, _EVENT_FIELDS, "ledger event")

    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise LedgerError(f"schema_version must be exact integer {SCHEMA_VERSION}")
    sequence = value["sequence"]
    if type(sequence) is not int:
        raise LedgerError("sequence must be an integer")
    if sequence != expected_sequence:
        raise LedgerError(
            f"sequence continuity break: expected {expected_sequence}, got {sequence}"
        )
    timestamp = _validate_timestamp(value["timestamp"])
    event_kind = _require_event_text(value["event_kind"], "event_kind")
    experiment = _require_event_text(value["experiment"], "experiment")
    payload = _copy_json_object(value["payload"])
    actual_previous = value["previous_event_sha256"]
    _require_hash(actual_previous, "previous_event_sha256", optional=True)
    if actual_previous != previous_hash:
        raise LedgerError(
            "previous_event_sha256 continuity break: "
            f"expected {previous_hash!r}, got {actual_previous!r}"
        )
    event_hash = value["event_sha256"]
    _require_hash(event_hash, "event_sha256")
    body = {key: item for key, item in value.items() if key != "event_sha256"}
    computed = hashlib.sha256(_canonical_json(body)).hexdigest()
    if event_hash != computed:
        raise LedgerError(
            f"event_sha256 body mismatch: recorded {event_hash}, computed {computed}"
        )
    event = LedgerEvent(
        schema_version=SCHEMA_VERSION,
        sequence=sequence,
        timestamp=timestamp,
        event_kind=event_kind,
        experiment=experiment,
        payload=payload,
        previous_event_sha256=actual_previous,
        event_sha256=event_hash,
    )
    _validate_standalone_event(event)
    return event


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable(fd: int) -> bytes:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise LedgerError("ledger must be a regular file")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(fd)
    raw = b"".join(chunks)
    if _stat_identity(before) != _stat_identity(after) or len(raw) != after.st_size:
        raise LedgerError("ledger mutated while it was being read")
    return raw


def _events_from_fd(fd: int) -> tuple[LedgerEvent, ...]:
    raw = _read_stable(fd)
    if not raw:
        return ()
    if not raw.endswith(b"\n"):
        raise LedgerError(
            "nonempty ledger lacks a final newline (partial/truncated append marker)"
        )
    lines = raw[:-1].split(b"\n")
    if any(not line for line in lines):
        raise LedgerError("ledger contains a blank interior line")
    events: list[LedgerEvent] = []
    previous_hash: str | None = None
    for sequence, line in enumerate(lines, start=1):
        event = _parse_event(line, sequence, previous_hash)
        events.append(event)
        previous_hash = event.event_sha256
    return tuple(events)


def _path_parts(path: str | os.PathLike[str]) -> tuple[tuple[str, ...], str, bool]:
    ledger_path = Path(path)
    if not ledger_path.name or ledger_path.name in {".", ".."}:
        raise LedgerError("ledger path must name a file")
    if ".." in ledger_path.parts:
        raise LedgerError("ledger parent path must not contain '..'")
    if not ledger_path.is_absolute():
        ledger_path = Path.cwd() / ledger_path
    return ledger_path.parent.parts, ledger_path.name, True


@dataclass(frozen=True, slots=True)
class _PathChain:
    fds: tuple[int, ...]
    names: tuple[str, ...]
    child_identities: tuple[tuple[int, int, int], ...]

    @property
    def parent_fd(self) -> int:
        return self.fds[-1]

    def close(self) -> None:
        first_error: OSError | None = None
        for fd in reversed(self.fds):
            try:
                os.close(fd)
            except OSError as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _directory_identity(fd: int) -> tuple[int, int, int]:
    value = os.fstat(fd)
    return (value.st_dev, value.st_ino, value.st_mode)


def _snapshot_path_chain(
    fds: tuple[int, ...], names: tuple[str, ...]
) -> _PathChain:
    return _PathChain(
        fds=fds,
        names=names,
        child_identities=tuple(_directory_identity(fd) for fd in fds[1:]),
    )


def _verify_path_chain(chain: _PathChain) -> None:
    # Directory timestamps are not authority: harmless sibling create/remove
    # operations mutate them. Descriptor identity plus current visible linkage
    # rejects material substitution without treating unrelated churn as ABA.
    # A same-inode rename-and-restore with unchanged descendants is accepted:
    # every ledger operation remains descriptor-bound and the final linkage is
    # identical. Any replaced child/file inode or changed ledger bytes still
    # fails the identity/hash checks below and in the locked ledger parser.
    for index, name in enumerate(chain.names):
        parent_fd = chain.fds[index]
        child_fd = chain.fds[index + 1]
        try:
            visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise LedgerError("ledger ancestor identity was replaced after open") from exc
        descriptor = os.fstat(child_fd)
        descriptor_identity = _directory_identity(child_fd)
        if (
            not stat.S_ISDIR(visible.st_mode)
            or not stat.S_ISDIR(descriptor_identity[2])
            or descriptor.st_nlink == 0
            or descriptor_identity != chain.child_identities[index]
            or (visible.st_dev, visible.st_ino)
            != (descriptor_identity[0], descriptor_identity[1])
        ):
            raise LedgerError("ledger ancestor identity was replaced after open")


def _open_parent(path: str | os.PathLike[str]) -> tuple[_PathChain, str]:
    parts, name, _absolute = _path_parts(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    fds: list[int] = []
    names: list[str] = []
    try:
        fds.append(os.open("/", flags))
        for component in parts[1:]:
            if component in {"", "."}:
                continue
            parent_fd = fds[-1]
            try:
                child_fd = os.open(component, flags, dir_fd=parent_fd)
            except OSError as exc:
                raise LedgerError(
                    "ledger parent path must contain only existing real directories"
                ) from exc
            if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                os.close(child_fd)
                raise LedgerError("ledger parent must be a real directory")
            fds.append(child_fd)
            names.append(component)
        chain = _snapshot_path_chain(tuple(fds), tuple(names))
        _verify_path_chain(chain)
        return chain, name
    except BaseException:
        for fd in reversed(fds):
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def _refresh_full_path_identities(chain: _PathChain) -> _PathChain:
    refreshed = _snapshot_path_chain(chain.fds, chain.names)
    if refreshed.child_identities != chain.child_identities:
        raise LedgerError("ledger path identity changed during publication")
    for index, name in enumerate(refreshed.names):
        try:
            visible = os.stat(
                name,
                dir_fd=refreshed.fds[index],
                follow_symlinks=False,
            )
        except OSError as exc:
            raise LedgerError(
                "ledger path identity changed during publication"
            ) from exc
        child = refreshed.child_identities[index]
        if not stat.S_ISDIR(visible.st_mode) or (visible.st_dev, visible.st_ino) != (
            child[0],
            child[1],
        ):
            raise LedgerError("ledger path identity changed during publication")
    return refreshed


@dataclass(frozen=True, slots=True)
class _OpenedLedger:
    chain: _PathChain
    name: str
    final_name: str
    fd: int
    created: bool
    staging_fd: int | None = None
    staging_name: str | None = None
    expected_bytes: bytes | None = None

    @property
    def parent_fd(self) -> int:
        return self.staging_fd if self.staging_fd is not None else self.chain.parent_fd

    @property
    def final_parent_fd(self) -> int:
        return self.chain.parent_fd

    def close(self) -> None:
        try:
            os.close(self.fd)
        finally:
            try:
                if self.staging_fd is not None:
                    os.close(self.staging_fd)
            finally:
                self.chain.close()


def _verify_visible_identity(opened: _OpenedLedger) -> None:
    _verify_path_chain(opened.chain)
    try:
        visible = os.stat(
            opened.name,
            dir_fd=opened.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise LedgerError("ledger path identity was replaced after open") from exc
    descriptor = os.fstat(opened.fd)
    if not stat.S_ISREG(visible.st_mode) or not stat.S_ISREG(descriptor.st_mode):
        raise LedgerError("ledger must be a regular file")
    if descriptor.st_nlink == 0 or (visible.st_dev, visible.st_ino) != (
        descriptor.st_dev,
        descriptor.st_ino,
    ):
        raise LedgerError("ledger path identity was replaced after open")


def _verify_private_identity(opened: _OpenedLedger) -> _PathChain:
    refreshed = _refresh_full_path_identities(opened.chain)
    try:
        visible = os.stat(
            opened.name,
            dir_fd=opened.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise LedgerError("private ledger identity was replaced after open") from exc
    descriptor = os.fstat(opened.fd)
    if not stat.S_ISREG(visible.st_mode) or not stat.S_ISREG(descriptor.st_mode):
        raise LedgerError("private ledger must be a regular file")
    if descriptor.st_nlink == 0 or (visible.st_dev, visible.st_ino) != (
        descriptor.st_dev,
        descriptor.st_ino,
    ):
        raise LedgerError("private ledger identity was replaced after open")
    return _refresh_full_path_identities(refreshed)


def _verify_staging_absent(opened: _OpenedLedger, chain: _PathChain) -> None:
    if opened.staging_name is None:
        raise LedgerError("private ledger has no staging directory")
    _refresh_full_path_identities(chain)
    try:
        os.stat(
            opened.staging_name,
            dir_fd=chain.parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise LedgerError("private staging absence could not be verified") from exc
    raise LedgerError("private staging directory remains visible after cleanup")


def _remove_retained_staging_directory(
    chain: _PathChain, staging_fd: int, staging_name: str
) -> None:
    try:
        visible = os.stat(
            staging_name,
            dir_fd=chain.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise LedgerError("private staging directory is no longer visible") from exc
    descriptor = os.fstat(staging_fd)
    if not stat.S_ISDIR(visible.st_mode) or (visible.st_dev, visible.st_ino) != (
        descriptor.st_dev,
        descriptor.st_ino,
    ):
        raise LedgerError("private staging directory identity was replaced")
    os.rmdir(staging_name, dir_fd=chain.parent_fd)


def _remove_staging_directory(opened: _OpenedLedger) -> None:
    if opened.staging_fd is None or opened.staging_name is None:
        raise LedgerError("private ledger has no retained staging directory")
    _remove_retained_staging_directory(
        opened.chain,
        opened.staging_fd,
        opened.staging_name,
    )


def _cleanup_private_ledger(opened: _OpenedLedger) -> None:
    if not opened.created:
        raise LedgerError("private-ledger cleanup requires a private inode")
    try:
        refreshed = _refresh_full_path_identities(opened.chain)
        private = _OpenedLedger(
            refreshed,
            opened.name,
            opened.final_name,
            opened.fd,
            True,
            opened.staging_fd,
            opened.staging_name,
        )
        refreshed = _verify_private_identity(private)
        private = _OpenedLedger(
            refreshed,
            opened.name,
            opened.final_name,
            opened.fd,
            True,
            opened.staging_fd,
            opened.staging_name,
        )
        os.unlink(private.name, dir_fd=private.parent_fd)
        if os.fstat(opened.fd).st_nlink != 0:
            raise LedgerError("private ledger inode remains linked after cleanup")
        _remove_staging_directory(private)
        cleaned = _refresh_full_path_identities(refreshed)
        os.fsync(cleaned.parent_fd)
        durable = _refresh_full_path_identities(cleaned)
        _verify_staging_absent(private, durable)
    except BaseException as exc:
        raise LedgerError(
            "private ledger cleanup rollback could not be proven durable"
        ) from exc


def _verify_published_identity(opened: _OpenedLedger) -> None:
    refreshed = _refresh_full_path_identities(opened.chain)
    try:
        visible = os.stat(
            opened.final_name,
            dir_fd=refreshed.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise LedgerError("published ledger identity is not visible") from exc
    descriptor = os.fstat(opened.fd)
    if not stat.S_ISREG(visible.st_mode) or not stat.S_ISREG(descriptor.st_mode):
        raise LedgerError("published ledger must be a regular file")
    if descriptor.st_nlink == 0 or (visible.st_dev, visible.st_ino) != (
        descriptor.st_dev,
        descriptor.st_ino,
    ):
        raise LedgerError("published ledger identity does not match private inode")


def _reauthenticate_publication_source(opened: _OpenedLedger) -> None:
    try:
        _verify_private_identity(opened)
        if opened.expected_bytes is None:
            raise LedgerError("publication has no intended canonical bytes")
        if _read_stable(opened.fd) != opened.expected_bytes:
            raise LedgerError(
                "staged content does not match intended canonical event bytes"
            )
    except BaseException as exc:
        raise LedgerPublicationSourceError(
            "publication source authentication failed: staged name or content "
            "no longer matches the verified canonical source"
        ) from exc


def _verify_publication_result(opened: _OpenedLedger) -> None:
    try:
        _verify_published_identity(opened)
        if opened.expected_bytes is None:
            raise LedgerError("publication has no intended canonical bytes")
        if _read_stable(opened.fd) != opened.expected_bytes:
            raise LedgerError(
                "published content does not match intended canonical event bytes"
            )
    except BaseException as exc:
        raise LedgerCorruptPublicationError(
            "corrupt/unverified public ledger: final inode or content bytes do not "
            "match the verified canonical publication source"
        ) from exc


def _after_native_source_reauthentication(_opened: _OpenedLedger) -> None:
    """Test hook modeling mutation in the pathname-only native syscall window."""


def _after_link_source_reauthentication(_opened: _OpenedLedger) -> None:
    """Test hook modeling mutation in the pathname-only link syscall window."""


def _renameatx_np_noreplace(opened: _OpenedLedger) -> bool:
    if sys.platform != "darwin":
        return False
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameatx_np", None)
    if function is None:
        return False
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    _reauthenticate_publication_source(opened)
    # macOS exposes only pathname-based exclusive rename. A hostile same-UID
    # writer can still mutate the source between this check and the kernel
    # syscall; immediate destination reauthentication below prevents trust.
    _after_native_source_reauthentication(opened)
    result = function(
        opened.parent_fd,
        os.fsencode(opened.name),
        opened.final_parent_fd,
        os.fsencode(opened.final_name),
        0x00000004,  # RENAME_EXCL
    )
    if result == 0:
        _verify_publication_result(opened)
        return True
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), opened.final_name)
    if error in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL}:
        return False
    raise OSError(error, os.strerror(error), opened.final_name)


def _renameat2_noreplace(opened: _OpenedLedger) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:
        return False
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    _reauthenticate_publication_source(opened)
    _after_native_source_reauthentication(opened)
    result = function(
        opened.parent_fd,
        os.fsencode(opened.name),
        opened.final_parent_fd,
        os.fsencode(opened.final_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        _verify_publication_result(opened)
        return True
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), opened.final_name)
    if error in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL}:
        return False
    raise OSError(error, os.strerror(error), opened.final_name)


def _publish_private_noreplace(opened: _OpenedLedger) -> None:
    if not opened.created:
        raise LedgerError("only a private ledger can be published")
    _reauthenticate_publication_source(opened)
    if _renameatx_np_noreplace(opened) or _renameat2_noreplace(opened):
        return
    try:
        _reauthenticate_publication_source(opened)
        _after_link_source_reauthentication(opened)
        os.link(
            opened.name,
            opened.final_name,
            src_dir_fd=opened.parent_fd,
            dst_dir_fd=opened.final_parent_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        raise
    except OSError as exc:
        raise LedgerError(f"private ledger publication failed: {exc}") from exc
    _verify_publication_result(opened)
    try:
        os.unlink(opened.name, dir_fd=opened.parent_fd)
    except BaseException as exc:
        _verify_publication_result(opened)
        raise LedgerUncertainCommitError(
            "uncertain committed state: published ledger is valid but private-link "
            "cleanup could not be proven"
        ) from exc


def _before_private_publication(_opened: _OpenedLedger) -> None:
    """Test hook after private verification and before no-replace publication."""


def _open_existing(
    path: str | os.PathLike[str], flags: int
) -> _OpenedLedger | None:
    for _attempt in range(8):
        chain, name = _open_parent(path)
        try:
            fd = os.open(name, flags | _OPEN_BASE_FLAGS, dir_fd=chain.parent_fd)
        except FileNotFoundError:
            try:
                _verify_path_chain(chain)
                os.stat(name, dir_fd=chain.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                chain.close()
                return None
            except BaseException:
                chain.close()
                raise
            chain.close()
            continue
        except OSError as exc:
            chain.close()
            if exc.errno in {errno.ELOOP, errno.ENXIO}:
                raise LedgerError(
                    "ledger must be a non-symlink regular file"
                ) from exc
            raise LedgerError(f"cannot open ledger safely: {exc}") from exc
        opened = _OpenedLedger(chain, name, name, fd, False)
        try:
            _verify_visible_identity(opened)
        except BaseException:
            opened.close()
            raise
        return opened
    raise LedgerError("ledger path kept appearing while it was opened")


def _open_for_append(path: str | os.PathLike[str]) -> _OpenedLedger:
    for _attempt in range(128):
        chain, name = _open_parent(path)
        existing_flags = os.O_RDWR | os.O_APPEND | _OPEN_BASE_FLAGS
        try:
            fd = os.open(name, existing_flags, dir_fd=chain.parent_fd)
        except FileNotFoundError:
            try:
                _verify_path_chain(chain)
            except LedgerError:
                chain.close()
                continue
        except OSError as exc:
            chain.close()
            if exc.errno == errno.ELOOP:
                raise LedgerError("ledger must be a non-symlink regular file") from exc
            raise LedgerError(f"cannot open ledger safely: {exc}") from exc
        else:
            opened = _OpenedLedger(chain, name, name, fd, False)
            try:
                _verify_visible_identity(opened)
            except BaseException:
                opened.close()
                raise
            return opened

        try:
            fcntl.flock(chain.parent_fd, fcntl.LOCK_EX)
        except BaseException as exc:
            chain.close()
            raise LedgerError(f"cannot lock ledger parent for creation: {exc}") from exc
        try:
            chain = _refresh_full_path_identities(chain)
        except LedgerError:
            chain.close()
            continue
        try:
            os.stat(name, dir_fd=chain.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            chain.close()
            raise LedgerError(f"cannot recheck ledger publication race: {exc}") from exc
        else:
            chain.close()
            continue

        private_token = secrets.token_hex(24)
        staging_name = f".recovery-ledger-stage-{private_token}"
        if staging_name == name:
            staging_name = f".recovery-ledger-stage-{private_token}-private"
        try:
            os.mkdir(staging_name, mode=0o700, dir_fd=chain.parent_fd)
        except FileExistsError:
            chain.close()
            continue
        except OSError as exc:
            chain.close()
            raise LedgerError(
                f"cannot create private staging directory: {exc}"
            ) from exc
        directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            staging_fd = os.open(
                staging_name,
                directory_flags,
                dir_fd=chain.parent_fd,
            )
        except BaseException as exc:
            try:
                os.rmdir(staging_name, dir_fd=chain.parent_fd)
            except OSError as cleanup_exc:
                chain.close()
                raise LedgerError(
                    "private staging cleanup rollback could not be proven durable"
                ) from cleanup_exc
            chain.close()
            raise LedgerError(
                f"cannot retain private staging directory: {exc}"
            ) from exc
        staging_stat = os.fstat(staging_fd)
        if stat.S_IMODE(staging_stat.st_mode) & 0o077:
            os.close(staging_fd)
            try:
                os.rmdir(staging_name, dir_fd=chain.parent_fd)
            finally:
                chain.close()
            raise LedgerError("private staging directory permissions are too broad")
        private_name = f"event-{private_token}.tmp"
        create_flags = existing_flags | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(
                private_name,
                create_flags,
                0o600,
                dir_fd=staging_fd,
            )
        except FileExistsError:
            os.close(staging_fd)
            os.rmdir(staging_name, dir_fd=chain.parent_fd)
            chain.close()
            continue
        except OSError as exc:
            try:
                _remove_retained_staging_directory(
                    chain,
                    staging_fd,
                    staging_name,
                )
            finally:
                os.close(staging_fd)
            chain.close()
            raise LedgerError(f"cannot create ledger safely: {exc}") from exc
        try:
            chain = _refresh_full_path_identities(chain)
        except BaseException as exc:
            opened = _OpenedLedger(
                chain,
                private_name,
                name,
                fd,
                True,
                staging_fd,
                staging_name,
            )
            try:
                _cleanup_private_ledger(opened)
            except LedgerError as cleanup_exc:
                opened.close()
                raise cleanup_exc from exc
            opened.close()
            raise LedgerError("private ledger creation failed") from exc
        opened = _OpenedLedger(
            chain,
            private_name,
            name,
            fd,
            True,
            staging_fd,
            staging_name,
        )
        try:
            _verify_private_identity(opened)
        except BaseException as exc:
            try:
                _cleanup_private_ledger(opened)
            except LedgerError as cleanup_exc:
                opened.close()
                raise cleanup_exc from exc
            opened.close()
            if isinstance(exc, LedgerError):
                raise
            raise LedgerError(f"ledger creation failed: {exc}") from exc
        return opened
    raise LedgerError("ledger parent kept mutating while the file was opened")


def _validate_anchors(
    events: tuple[LedgerEvent, ...],
    expected_head_sha256: str | None,
    expected_event_count: int | None,
) -> None:
    if expected_head_sha256 is not None:
        _require_hash(expected_head_sha256, "expected_head_sha256")
        actual_head = events[-1].event_sha256 if events else None
        if actual_head != expected_head_sha256:
            raise LedgerError(
                f"ledger head mismatch: expected {expected_head_sha256}, got {actual_head}"
            )
    if expected_event_count is not None:
        if type(expected_event_count) is not int or expected_event_count < 0:
            raise LedgerError("expected_event_count must be a nonnegative integer")
        if len(events) != expected_event_count:
            raise LedgerError(
                "ledger event count mismatch: "
                f"expected {expected_event_count}, got {len(events)}"
            )


def load_ledger(
    path: str | os.PathLike[str],
    *,
    expected_head_sha256: str | None = None,
    expected_event_count: int | None = None,
) -> tuple[LedgerEvent, ...]:
    """Load and verify the full ledger.

    A missing ledger in an existing real parent is the empty ledger. External
    head/count anchors can distinguish that state, or a clean tail deletion,
    from the sealed state expected by a controller or handoff.
    """

    opened = _open_existing(path, os.O_RDONLY)
    if opened is None:
        events: tuple[LedgerEvent, ...] = ()
        _validate_anchors(events, expected_head_sha256, expected_event_count)
        return events
    try:
        descriptor = os.fstat(opened.fd)
        if not stat.S_ISREG(descriptor.st_mode):
            raise LedgerError("ledger must be a regular file")
        fcntl.flock(opened.fd, fcntl.LOCK_SH)
        _verify_visible_identity(opened)
        events = _events_from_fd(opened.fd)
        _verify_visible_identity(opened)
        _validate_anchors(events, expected_head_sha256, expected_event_count)
        return events
    finally:
        opened.close()


def _event_for_append(
    events: tuple[LedgerEvent, ...],
    *,
    event_kind: str,
    experiment: str,
    payload: dict[str, object],
    timestamp: str,
) -> LedgerEvent:
    previous_hash = events[-1].event_sha256 if events else None
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "sequence": len(events) + 1,
        "timestamp": timestamp,
        "event_kind": event_kind,
        "experiment": experiment,
        "payload": payload,
        "previous_event_sha256": previous_hash,
    }
    return LedgerEvent(
        schema_version=SCHEMA_VERSION,
        sequence=len(events) + 1,
        timestamp=timestamp,
        event_kind=event_kind,
        experiment=experiment,
        payload=payload,
        previous_event_sha256=previous_hash,
        event_sha256=hashlib.sha256(_canonical_json(body)).hexdigest(),
    )


def append_event(
    path: str | os.PathLike[str],
    *,
    event_kind: str,
    experiment: str,
    payload: dict[str, object],
    timestamp: str | datetime | None = None,
    expected_head_sha256: str | None = None,
    expected_event_count: int | None = None,
) -> LedgerEvent:
    """Append exactly one durable event after verifying the locked chain."""

    event_kind = _require_event_text(event_kind, "event_kind")
    experiment = _require_event_text(experiment, "experiment")
    payload_copy = _copy_json_object(payload)
    _validate_event_payload(event_kind, experiment, payload_copy)
    timestamp_text = _timestamp_value(timestamp)
    for _attempt in range(128):
        opened = _open_for_append(path)
        if opened.created:
            cleanup_attempted = False
            try:
                descriptor = os.fstat(opened.fd)
                if not stat.S_ISREG(descriptor.st_mode):
                    raise LedgerError("ledger must be a regular file")
                fcntl.flock(opened.fd, fcntl.LOCK_EX)
                _verify_private_identity(opened)
                events = _events_from_fd(opened.fd)
                if events:
                    raise LedgerError("new private ledger must start empty")
                _validate_anchors(
                    events,
                    expected_head_sha256,
                    expected_event_count,
                )
                event = _event_for_append(
                    events,
                    event_kind=event_kind,
                    experiment=experiment,
                    payload=payload_copy,
                    timestamp=timestamp_text,
                )
                encoded = _canonical_json(_event_mapping(event)) + b"\n"
                written = os.write(opened.fd, encoded)
                if written != len(encoded):
                    raise LedgerError(
                        f"short ledger append: wrote {written} of {len(encoded)} bytes"
                    )
                os.fsync(opened.fd)
                _verify_private_identity(opened)
                if _events_from_fd(opened.fd) != (event,):
                    raise LedgerError(
                        "private ledger verification did not reproduce event"
                    )
                publication = _OpenedLedger(
                    opened.chain,
                    opened.name,
                    opened.final_name,
                    opened.fd,
                    True,
                    opened.staging_fd,
                    opened.staging_name,
                    encoded,
                )
                _before_private_publication(publication)
                try:
                    _publish_private_noreplace(publication)
                except FileExistsError:
                    cleanup_attempted = True
                    _cleanup_private_ledger(opened)
                    continue
                except (
                    LedgerCorruptPublicationError,
                    LedgerPublicationSourceError,
                    LedgerUncertainCommitError,
                ):
                    raise

                try:
                    _remove_staging_directory(publication)
                    refreshed = _refresh_full_path_identities(opened.chain)
                    os.fsync(refreshed.parent_fd)
                except BaseException as exc:
                    _verify_publication_result(publication)
                    raise LedgerUncertainCommitError(
                        "uncertain committed state: complete ledger was published but "
                        "directory durability could not be proven"
                    ) from exc
                published = _OpenedLedger(
                    refreshed,
                    opened.final_name,
                    opened.final_name,
                    opened.fd,
                    False,
                    expected_bytes=encoded,
                )
                _verify_publication_result(published)
                return event
            except (
                LedgerCorruptPublicationError,
                LedgerPublicationSourceError,
                LedgerUncertainCommitError,
            ):
                raise
            except BaseException as exc:
                if not cleanup_attempted:
                    try:
                        _cleanup_private_ledger(opened)
                    except LedgerError as cleanup_exc:
                        raise cleanup_exc from exc
                if isinstance(exc, LedgerError):
                    raise
                raise LedgerError(f"ledger append failed: {exc}") from exc
            finally:
                opened.close()

        original_length: int | None = None
        append_started = False
        try:
            descriptor = os.fstat(opened.fd)
            if not stat.S_ISREG(descriptor.st_mode):
                raise LedgerError("ledger must be a regular file")
            fcntl.flock(opened.fd, fcntl.LOCK_EX)
            _verify_visible_identity(opened)
            events = _events_from_fd(opened.fd)
            _validate_anchors(events, expected_head_sha256, expected_event_count)
            _verify_visible_identity(opened)
            original_length = os.fstat(opened.fd).st_size
            event = _event_for_append(
                events,
                event_kind=event_kind,
                experiment=experiment,
                payload=payload_copy,
                timestamp=timestamp_text,
            )
            encoded = _canonical_json(_event_mapping(event)) + b"\n"
            append_started = True
            written = os.write(opened.fd, encoded)
            if written != len(encoded):
                raise LedgerError(
                    f"short ledger append: wrote {written} of {len(encoded)} bytes"
                )
            os.fsync(opened.fd)
            _verify_visible_identity(opened)
            return event
        except BaseException as exc:
            if append_started and original_length is not None:
                try:
                    os.ftruncate(opened.fd, original_length)
                    os.fsync(opened.fd)
                    _verify_visible_identity(opened)
                except BaseException as rollback_exc:
                    raise LedgerError(
                        "ledger append failed and rollback could not be made durable"
                    ) from rollback_exc
            if isinstance(exc, LedgerError):
                raise
            raise LedgerError(f"ledger append failed: {exc}") from exc
        finally:
            opened.close()
    raise LedgerError("ledger publication kept losing no-replace races")


def _verification_payload(result: VerificationResult) -> dict[str, object]:
    return {
        "name": result.name,
        "argv": list(result.argv),
        "exit_code": result.exit_code,
        "stdout_sha256": result.stdout_sha256,
        "stderr_sha256": result.stderr_sha256,
    }


def _evidence_payload(evidence: EvidenceRecord) -> dict[str, object]:
    return {
        "artifact_path": evidence.artifact_path,
        "content_sha256": evidence.content_sha256,
        "manifest_identity_sha256": evidence.manifest_identity_sha256,
        "candidate_identity_sha256": evidence.candidate_identity_sha256,
        "parent_identity_sha256": evidence.parent_identity_sha256,
        "baseline_identity_sha256": evidence.baseline_identity_sha256,
        "evidence_class": evidence.evidence_class,
        "release_eligible": evidence.release_eligible,
        "recovery_levers": list(evidence.recovery_levers),
        "argv": list(evidence.argv),
        "verification_results": [
            _verification_payload(result) for result in evidence.verification_results
        ],
    }


def append_evidence(
    path: str | os.PathLike[str],
    *,
    experiment: str,
    evidence: EvidenceRecord,
    timestamp: str | datetime | None = None,
    expected_head_sha256: str | None = None,
    expected_event_count: int | None = None,
) -> LedgerEvent:
    """Register supplied evidence identities without reading the artifact."""

    if not isinstance(evidence, EvidenceRecord):
        raise TypeError("evidence must be an EvidenceRecord")
    return append_event(
        path,
        event_kind="evidence_registered",
        experiment=experiment,
        payload=_evidence_payload(evidence),
        timestamp=timestamp,
        expected_head_sha256=expected_head_sha256,
        expected_event_count=expected_event_count,
    )


def evidence_from_event(event: LedgerEvent) -> EvidenceRecord:
    """Strictly reconstruct the immutable evidence record from one event."""

    _validate_standalone_event(event)
    if event.event_kind != "evidence_registered":
        raise LedgerError("event_kind must be evidence_registered")
    payload = event.payload
    _require_exact_fields(payload, _EVIDENCE_FIELDS, "evidence payload")
    results_raw = payload["verification_results"]
    if not isinstance(results_raw, Sequence) or isinstance(
        results_raw, (str, bytes, bytearray)
    ):
        raise LedgerError("verification_results must be an array")
    results: list[VerificationResult] = []
    for index, raw in enumerate(results_raw):
        if not isinstance(raw, Mapping):
            raise LedgerError(f"verification_results[{index}] must be an object")
        _require_exact_fields(raw, _VERIFICATION_FIELDS, "verification result")
        argv = raw["argv"]
        if not isinstance(argv, Sequence) or isinstance(
            argv, (str, bytes, bytearray)
        ):
            raise LedgerError("verification result argv must be an array")
        try:
            results.append(
                VerificationResult(
                    name=raw["name"],  # type: ignore[arg-type]
                    argv=tuple(argv),  # type: ignore[arg-type]
                    exit_code=raw["exit_code"],  # type: ignore[arg-type]
                    stdout_sha256=raw["stdout_sha256"],  # type: ignore[arg-type]
                    stderr_sha256=raw["stderr_sha256"],  # type: ignore[arg-type]
                )
            )
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"invalid verification result: {exc}") from exc
    levers = payload["recovery_levers"]
    argv = payload["argv"]
    if (
        not isinstance(levers, Sequence)
        or isinstance(levers, (str, bytes, bytearray))
        or not isinstance(argv, Sequence)
        or isinstance(argv, (str, bytes, bytearray))
    ):
        raise LedgerError("evidence recovery_levers and argv must be arrays")
    try:
        return EvidenceRecord(
            artifact_path=payload["artifact_path"],  # type: ignore[arg-type]
            content_sha256=payload["content_sha256"],  # type: ignore[arg-type]
            manifest_identity_sha256=payload["manifest_identity_sha256"],  # type: ignore[arg-type]
            candidate_identity_sha256=payload[  # type: ignore[arg-type]
                "candidate_identity_sha256"
            ],
            parent_identity_sha256=payload["parent_identity_sha256"],  # type: ignore[arg-type]
            baseline_identity_sha256=payload["baseline_identity_sha256"],  # type: ignore[arg-type]
            evidence_class=payload["evidence_class"],  # type: ignore[arg-type]
            release_eligible=payload["release_eligible"],  # type: ignore[arg-type]
            recovery_levers=tuple(levers),  # type: ignore[arg-type]
            argv=tuple(argv),  # type: ignore[arg-type]
            verification_results=tuple(results),
        )
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"invalid evidence payload: {exc}") from exc


__all__ = [
    "LedgerCorruptPublicationError",
    "LedgerError",
    "LedgerPublicationSourceError",
    "LedgerUncertainCommitError",
    "SCHEMA_VERSION",
    "append_event",
    "append_evidence",
    "evidence_from_event",
    "load_ledger",
]
