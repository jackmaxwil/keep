"""Pure review records and conservative finding classification."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from mlx_vq.io.authenticated_artifacts import FileIdentity
from .lanes import LaneError, validate_direct_argv


class ReviewRecordError(ValueError):
    """A review record is malformed, stale, or self-authorized."""


_REVIEW_KEYS = {
    "schema_version",
    "name",
    "base_commit",
    "head_commit",
    "reviewed_content_sha256",
    "profile_sha256",
    "owned_paths",
    "frozen_decisions",
    "out_of_scope_paths",
    "threat_model",
    "verification_evidence",
    "resolved_finding_fingerprints",
    "known_diagnostic_fingerprints",
    "reviewer_authority",
    "result_status",
}
_FINDING_KEYS = {"title", "severity", "paths", "threat", "red_argv"}
_RESULTS = {"approved", "changes-required", "blocked"}
_SEVERITIES = {"critical", "important", "minor"}
_CLASSIFICATIONS = {
    "new",
    "duplicate",
    "resolved",
    "out-of-scope",
    "known-diagnostic",
}
_RECORD_SEAL_KEY = secrets.token_bytes(32)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _safe_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReviewRecordError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ReviewRecordError(f"{label} is unsafe: {value!r}")
    return value


def _path_contains(parent: str, child: str) -> bool:
    return child == parent or child.startswith(parent + "/")


def _paths_overlap(first: str, second: str) -> bool:
    return _path_contains(first, second) or _path_contains(second, first)


def _strict_strings(value: object, *, label: str, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ReviewRecordError(f"{label} must be a list of non-empty strings")
    result = tuple(value)
    if nonempty and not result:
        raise ReviewRecordError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ReviewRecordError(f"{label} contains duplicates")
    return result


def _strict_paths(value: object, *, label: str, nonempty: bool = True) -> tuple[str, ...]:
    raw = _strict_strings(value, label=label, nonempty=nonempty)
    return tuple(_safe_path(item, label=label) for item in raw)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewEvidenceIdentity:
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not _is_sha256(
            self.sha256
        ):
            raise ReviewRecordError("review evidence identity is invalid")

    def canonical_body(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    schema_version: int
    name: str
    base_commit: str | None
    head_commit: str | None
    reviewed_content_sha256: str | None
    profile_sha256: str
    owned_paths: tuple[str, ...]
    frozen_decisions: tuple[str, ...]
    out_of_scope_paths: tuple[str, ...]
    threat_model: tuple[str, ...]
    verification_evidence: tuple[ReviewEvidenceIdentity, ...]
    resolved_finding_fingerprints: tuple[str, ...]
    known_diagnostic_fingerprints: tuple[str, ...]
    reviewer_authority: str
    result_status: str
    _validation_seal: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ReviewRecordError(
                "review record schema_version must be exact integer 1"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ReviewRecordError("review record identity is invalid")
        if not _is_sha256(self.profile_sha256):
            raise ReviewRecordError("review profile authority is invalid")
        for name in (
            "owned_paths",
            "frozen_decisions",
            "out_of_scope_paths",
            "threat_model",
            "verification_evidence",
            "resolved_finding_fingerprints",
            "known_diagnostic_fingerprints",
        ):
            if type(getattr(self, name)) is not tuple:
                raise ReviewRecordError(
                    f"review record {name} must be an immutable tuple"
                )
        commit_mode = self.base_commit is not None or self.head_commit is not None
        content_mode = self.reviewed_content_sha256 is not None
        if commit_mode == content_mode:
            raise ReviewRecordError("review record authority mode is invalid")
        if commit_mode:
            if (
                not _is_commit(self.base_commit)
                or not _is_commit(self.head_commit)
                or self.base_commit == "0" * 40
                or self.head_commit == "0" * 40
                or self.base_commit == self.head_commit
            ):
                raise ReviewRecordError("review record commit range is invalid")
        elif not _is_sha256(self.reviewed_content_sha256):
            raise ReviewRecordError("review record content authority is invalid")
        if not self.owned_paths or not self.out_of_scope_paths:
            raise ReviewRecordError("review record path fences must not be empty")
        for label, paths in (
            ("owned_paths", self.owned_paths),
            ("out_of_scope_paths", self.out_of_scope_paths),
        ):
            for path in paths:
                _safe_path(path, label=label)
            if len(set(paths)) != len(paths):
                raise ReviewRecordError(f"review record {label} contains duplicates")
        if any(
            _paths_overlap(left, right)
            for left in self.owned_paths
            for right in self.out_of_scope_paths
        ):
            raise ReviewRecordError("review record path fences overlap")
        for label, values in (
            ("frozen_decisions", self.frozen_decisions),
            ("threat_model", self.threat_model),
        ):
            if (
                not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)
            ):
                raise ReviewRecordError(f"review record {label} is invalid")
        if (
            not self.verification_evidence
            or any(
                not isinstance(value, ReviewEvidenceIdentity)
                for value in self.verification_evidence
            )
            or len({value.name for value in self.verification_evidence})
            != len(self.verification_evidence)
        ):
            raise ReviewRecordError("review record evidence identities are invalid")
        for label, values in (
            ("resolved", self.resolved_finding_fingerprints),
            ("diagnostic", self.known_diagnostic_fingerprints),
        ):
            if (
                any(not _is_sha256(value) for value in values)
                or len(set(values)) != len(values)
            ):
                raise ReviewRecordError(f"review record {label} findings are invalid")
        if set(self.resolved_finding_fingerprints).intersection(
            self.known_diagnostic_fingerprints
        ):
            raise ReviewRecordError("review record finding states contradict")
        if (
            not isinstance(self.reviewer_authority, str)
            or not self.reviewer_authority
            or "self" in self.reviewer_authority.lower()
            or "implementer" in self.reviewer_authority.lower()
        ):
            raise ReviewRecordError("review record reviewer authority is invalid")
        if self.result_status not in _RESULTS:
            raise ReviewRecordError("review record result status is invalid")

    def canonical_body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "base_commit": self.base_commit,
            "head_commit": self.head_commit,
            "reviewed_content_sha256": self.reviewed_content_sha256,
            "profile_sha256": self.profile_sha256,
            "owned_paths": list(self.owned_paths),
            "frozen_decisions": list(self.frozen_decisions),
            "out_of_scope_paths": list(self.out_of_scope_paths),
            "threat_model": list(self.threat_model),
            "verification_evidence": [
                evidence.canonical_body() for evidence in self.verification_evidence
            ],
            "resolved_finding_fingerprints": list(
                self.resolved_finding_fingerprints
            ),
            "known_diagnostic_fingerprints": list(
                self.known_diagnostic_fingerprints
            ),
            "reviewer_authority": self.reviewer_authority,
            "result_status": self.result_status,
        }


def _record_seal(record: ReviewRecord) -> str:
    payload = json.dumps(
        record.canonical_body(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hmac.new(_RECORD_SEAL_KEY, payload, hashlib.sha256).hexdigest()


def _is_parsed_record(record: object) -> bool:
    return isinstance(record, ReviewRecord) and hmac.compare_digest(
        record._validation_seal,
        _record_seal(record),
    )


def _require_parsed_record(record: ReviewRecord) -> None:
    if not _is_parsed_record(record):
        raise ReviewRecordError("review record lacks parsed external authority")


def _parse_evidence(value: object) -> tuple[ReviewEvidenceIdentity, ...]:
    if not isinstance(value, list) or not value:
        raise ReviewRecordError("verification_evidence must not be empty")
    evidence: list[ReviewEvidenceIdentity] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"name", "sha256"}:
            raise ReviewRecordError("verification evidence keys do not match schema")
        name = item["name"]
        digest = item["sha256"]
        if not isinstance(name, str) or not name or not _is_sha256(digest):
            raise ReviewRecordError("verification evidence identity is invalid")
        evidence.append(ReviewEvidenceIdentity(name, digest))
    if len({item.name for item in evidence}) != len(evidence):
        raise ReviewRecordError("verification evidence names must be unique")
    return tuple(evidence)


def parse_review_record(
    payload: Mapping[str, object],
    *,
    expected_profile_sha256: str,
    recognized_reviewer_authorities: Sequence[str],
    expected_base: str | None = None,
    expected_head: str | None = None,
    expected_content_sha256: str | None = None,
) -> ReviewRecord:
    if isinstance(recognized_reviewer_authorities, (str, bytes)) or any(
        not isinstance(authority, str) or not authority
        for authority in recognized_reviewer_authorities
    ):
        raise ReviewRecordError("recognized reviewer authorities must be a sequence")
    if len(set(recognized_reviewer_authorities)) != len(
        recognized_reviewer_authorities
    ):
        raise ReviewRecordError("recognized reviewer authorities contain duplicates")
    if set(payload) != _REVIEW_KEYS:
        raise ReviewRecordError("review record keys do not match the exact schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ReviewRecordError(
            "review record schema_version must be exact integer 1"
        )
    name = payload["name"]
    if not isinstance(name, str) or not name:
        raise ReviewRecordError("review name must be non-empty")
    profile = payload["profile_sha256"]
    if not _is_sha256(profile) or profile != expected_profile_sha256:
        raise ReviewRecordError("review profile authority is stale")
    base = payload["base_commit"]
    head = payload["head_commit"]
    content = payload["reviewed_content_sha256"]
    commit_mode = base is not None or head is not None
    content_mode = content is not None
    if commit_mode == content_mode:
        raise ReviewRecordError(
            "review must declare exactly one commit-range or content authority"
        )
    if commit_mode:
        if not _is_commit(base) or not _is_commit(head):
            raise ReviewRecordError("review commit authority is incomplete")
        if base == "0" * 40 or head == "0" * 40 or base == head:
            raise ReviewRecordError("review commit range is zero or non-forward")
        if (
            not _is_commit(expected_base)
            or not _is_commit(expected_head)
            or base != expected_base
            or head != expected_head
        ):
            raise ReviewRecordError("review commit authority is absent or stale")
        if expected_content_sha256 is not None:
            raise ReviewRecordError("expected content authority conflicts with commit mode")
    else:
        if not _is_sha256(content):
            raise ReviewRecordError("review content authority is invalid")
        if (
            not _is_sha256(expected_content_sha256)
            or content != expected_content_sha256
        ):
            raise ReviewRecordError("review content authority is absent or stale")
        if expected_base is not None or expected_head is not None:
            raise ReviewRecordError("expected head authority conflicts with content mode")
    owned = _strict_paths(payload["owned_paths"], label="owned_paths")
    out_of_scope = _strict_paths(
        payload["out_of_scope_paths"], label="out_of_scope_paths"
    )
    if any(_paths_overlap(left, right) for left in owned for right in out_of_scope):
        raise ReviewRecordError("owned and out-of-scope paths overlap")
    frozen = _strict_strings(payload["frozen_decisions"], label="frozen_decisions")
    threats = _strict_strings(payload["threat_model"], label="threat_model")
    resolved = _strict_strings(
        payload["resolved_finding_fingerprints"],
        label="resolved_finding_fingerprints",
        nonempty=False,
    )
    diagnostic = _strict_strings(
        payload["known_diagnostic_fingerprints"],
        label="known_diagnostic_fingerprints",
        nonempty=False,
    )
    if not all(_is_sha256(value) for value in (*resolved, *diagnostic)):
        raise ReviewRecordError("finding fingerprints must be SHA-256 values")
    if set(resolved).intersection(diagnostic):
        raise ReviewRecordError("finding cannot be both resolved and diagnostic")
    reviewer = payload["reviewer_authority"]
    if (
        not isinstance(reviewer, str)
        or reviewer not in recognized_reviewer_authorities
        or "self" in reviewer.lower()
        or "implementer" in reviewer.lower()
    ):
        raise ReviewRecordError("reviewer authority is self-asserted or unrecognized")
    status = payload["result_status"]
    if status not in _RESULTS:
        raise ReviewRecordError("review result_status is invalid")
    record = ReviewRecord(
        schema_version=1,
        name=name,
        base_commit=base if isinstance(base, str) else None,
        head_commit=head if isinstance(head, str) else None,
        reviewed_content_sha256=content if isinstance(content, str) else None,
        profile_sha256=profile,
        owned_paths=owned,
        frozen_decisions=frozen,
        out_of_scope_paths=out_of_scope,
        threat_model=threats,
        verification_evidence=_parse_evidence(payload["verification_evidence"]),
        resolved_finding_fingerprints=resolved,
        known_diagnostic_fingerprints=diagnostic,
        reviewer_authority=reviewer,
        result_status=status,
    )
    object.__setattr__(record, "_validation_seal", _record_seal(record))
    return record


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewRecordError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReviewRecordError(f"JSON number must be finite: {value}")


def _read_regular_bytes_no_follow(path: Path, *, label: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReviewRecordError(
            f"could not open {label} without following links: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink == 0:
            raise ReviewRecordError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            visible = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ReviewRecordError(f"{label} disappeared while it was read") from error
        if (
            FileIdentity.from_stat(before) != FileIdentity.from_stat(after)
            or FileIdentity.from_stat(after) != FileIdentity.from_stat(visible)
        ):
            raise ReviewRecordError(f"{label} identity changed while it was read")
        return b"".join(chunks)
    except OSError as error:
        raise ReviewRecordError(f"could not read {label}: {error}") from error
    finally:
        os.close(descriptor)


def load_review_record(
    path: str | Path,
    **authority: object,
) -> ReviewRecord:
    try:
        raw = _read_regular_bytes_no_follow(Path(path), label="review record")
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ReviewRecordError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewRecordError(f"review record is not valid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ReviewRecordError("review record must contain a JSON object")
    return parse_review_record(payload, **authority)  # type: ignore[arg-type]


def review_fingerprint(record: ReviewRecord) -> str:
    _require_parsed_record(record)
    return _canonical_sha256(record.canonical_body())


def _normalize_finding(payload: Mapping[str, object]) -> dict[str, object]:
    if set(payload) != _FINDING_KEYS:
        raise ReviewRecordError("finding keys do not match schema")
    title = payload["title"]
    severity = payload["severity"]
    threat = payload["threat"]
    if not isinstance(title, str) or not title or severity not in _SEVERITIES:
        raise ReviewRecordError("finding title or severity is invalid")
    if not isinstance(threat, str) or not threat:
        raise ReviewRecordError("finding threat must be non-empty")
    paths = _strict_paths(payload["paths"], label="finding paths")
    try:
        argv = validate_direct_argv(payload["red_argv"], label="finding RED argv")
    except LaneError as error:
        raise ReviewRecordError(str(error)) from error
    return {
        "title": title,
        "severity": severity,
        "paths": list(paths),
        "threat": threat,
        "red_argv": list(argv),
    }


def finding_fingerprint(payload: Mapping[str, object]) -> str:
    return _canonical_sha256(_normalize_finding(payload))


def classify_finding(
    payload: Mapping[str, object],
    record: ReviewRecord,
    *,
    duplicate_fingerprints: Sequence[str] = (),
) -> str:
    if not _is_parsed_record(record):
        return "new"
    if (
        isinstance(duplicate_fingerprints, (str, bytes))
        or not isinstance(duplicate_fingerprints, (tuple, list))
        or not all(_is_sha256(value) for value in duplicate_fingerprints)
        or len(set(duplicate_fingerprints)) != len(duplicate_fingerprints)
    ):
        return "new"
    try:
        normalized = _normalize_finding(payload)
        fingerprint = _canonical_sha256(normalized)
    except (ReviewRecordError, TypeError, ValueError):
        return "new"
    paths = tuple(normalized["paths"])
    touches_owned = any(
        any(_paths_overlap(boundary, path) for boundary in record.owned_paths)
        for path in paths
    )
    touches_out_of_scope = any(
        any(
            _paths_overlap(boundary, path)
            for boundary in record.out_of_scope_paths
        )
        for path in paths
    )
    if touches_owned and touches_out_of_scope:
        return "new"
    if fingerprint in record.resolved_finding_fingerprints:
        return "resolved"
    if fingerprint in record.known_diagnostic_fingerprints:
        return "known-diagnostic"
    if fingerprint in duplicate_fingerprints:
        return "duplicate"
    if paths and all(
        any(_path_contains(boundary, path) for boundary in record.out_of_scope_paths)
        for path in paths
    ) and not any(
        any(_path_contains(boundary, path) for boundary in record.owned_paths)
        for path in paths
    ):
        return "out-of-scope"
    return "new"


def render_red_test_queue(
    findings: Sequence[Mapping[str, object]],
    record: ReviewRecord,
    *,
    duplicate_fingerprints: Sequence[str] = (),
) -> tuple[dict[str, object], ...]:
    _require_parsed_record(record)
    queue: list[dict[str, object]] = []
    for finding in findings:
        if classify_finding(
            finding,
            record,
            duplicate_fingerprints=duplicate_fingerprints,
        ) != "new":
            continue
        try:
            normalized = _normalize_finding(finding)
        except ReviewRecordError:
            continue
        queue.append(
            {
                "classification": "new",
                "finding_sha256": _canonical_sha256(normalized),
                "title": normalized["title"],
                "paths": normalized["paths"],
                "red_argv": normalized["red_argv"],
            }
        )
    return tuple(sorted(queue, key=lambda item: str(item["finding_sha256"])))


__all__ = [
    "ReviewEvidenceIdentity",
    "ReviewRecord",
    "ReviewRecordError",
    "classify_finding",
    "finding_fingerprint",
    "load_review_record",
    "parse_review_record",
    "render_red_test_queue",
    "review_fingerprint",
]
