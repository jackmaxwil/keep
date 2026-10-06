"""Pure, version-pinned Task 11 handoff and Task 12 correlation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Mapping, Protocol

from .canonical import canonical_sha256


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"[A-Za-z0-9._+=/-]{1,1024}\Z")
_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,255}\Z")
_TASK11_VERSION_ARN = re.compile(
    r"arn:aws:states:us-west-2:246813579024:stateMachine:"
    r"[A-Za-z0-9._-]+:[1-9][0-9]*\Z"
)
_TASK11_EXECUTION_ARN = re.compile(
    r"arn:aws:states:us-west-2:246813579024:execution:"
    r"[A-Za-z0-9._-]+:[A-Za-z0-9._:-]+\Z"
)
_HANDOFF_FIELDS = frozenset(
    """run_id campaign_identity_sha256 generation generation_text submit_attempt_id
    decision_key decision_version_id decision_file_sha256 decision_body_sha256
    sky_post_action_key sky_post_consumed_at sky_post_outcome_class
    expected_sky_job_name task_yaml_sha256 request_body_sha256
    api_server_identity_sha256 sky_request_id post_started_at
    post_completed_or_lost_at binding_state handoff_body_sha256""".split()
)
_HANDOFF_OUTCOMES = frozenset(
    {
        "ACCEPTED",
        "KNOWN_REJECTED",
        "AMBIGUOUS",
        "PROVED_NOT_SENT_OWNER_DIED",
        "AMBIGUOUS_OWNER_DIED",
    }
)
_TUPLE_FIELDS = frozenset(
    """user_id request_kind activation_id generation action_key envelope_sha256
    task_yaml_sha256 request_body_sha256 expected_sky_job_name window_open_at
    window_close_at""".split()
)
_QUIESCENCE_FIELDS = frozenset(
    """observed_at request_ids job_ids worker_instance_ids allocation_ids
    controller_work_ids""".split()
)


class Task12CorrelationError(ValueError):
    """Task 12 correlation input is incomplete, mutable, or ambiguous."""


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise Task12CorrelationError(label + " must be lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise Task12CorrelationError(label + " must be a nonempty string")
    return value


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise Task12CorrelationError(label + " must be exact UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Task12CorrelationError(label + " must be exact UTC") from exc
    if parsed.tzinfo != timezone.utc or parsed.microsecond:
        raise Task12CorrelationError(label + " must be whole-second UTC")
    return parsed


@dataclass(frozen=True)
class Task11HandoffAuthority:
    bucket: str
    key: str
    version_id: str
    file_sha256: str
    body: Mapping[str, object]
    body_sha256: str
    approved_task11_workflow_version_arn: str
    task11_execution_arn: str


@dataclass(frozen=True)
class ImmutableRequestTuple:
    user_id: str
    request_kind: str
    activation_id: str
    generation: int
    action_key: str
    envelope_sha256: str
    task_yaml_sha256: str
    request_body_sha256: str
    expected_sky_job_name: str
    window_open_at: str
    window_close_at: str


@dataclass(frozen=True)
class CorrelatedRequest:
    request_id: str
    state: str
    body: Mapping[str, object]


@dataclass(frozen=True)
class RequestCorrelation:
    kind: str
    matches: tuple[CorrelatedRequest, ...]


@dataclass(frozen=True)
class NoJobProof:
    request_id: str
    terminal_state: str
    first_quiescence_at: str
    second_quiescence_at: str


@dataclass(frozen=True)
class NumericJobBinding:
    request_id: str
    numeric_job_id: int
    sky_job_name: str
    state: str


class RequestCorrelationReader(Protocol):
    def read_exact_request(self, request_id: str) -> object:
        ...

    def scan_requests(
        self, correlation_tuple: object, cursor: str | None
    ) -> object:
        ...


def build_sky_post_handoff_record(**values: object) -> dict[str, object]:
    """Build the one frozen Task 11/Task 12 SKY_POST_HANDOFF body.

    The handoff deliberately has no schema discriminator.  Its exact field
    set and self-hash are the discriminator shared by the normal Task 11
    writer and the retained owner-death recovery writer.
    """

    if set(values) != _HANDOFF_FIELDS - {"handoff_body_sha256"}:
        raise Task12CorrelationError("Task 11 handoff builder fields are not exact")
    body = dict(values)
    body["handoff_body_sha256"] = canonical_sha256(body)
    return validate_sky_post_handoff_record(body)


def validate_sky_post_handoff_record(value: object) -> dict[str, object]:
    """Validate the frozen handoff, including the owner-death null matrix."""

    if type(value) is not dict or set(value) != _HANDOFF_FIELDS:
        raise Task12CorrelationError("Task 11 handoff schema is not exact")
    body = dict(value)
    if body["run_id"] != RUN_ID:
        raise Task12CorrelationError("Task 11 handoff run identity drifted")
    generation = body["generation"]
    if (
        type(generation) is not int
        or isinstance(generation, bool)
        or generation < 1
        or body["generation_text"] != f"{generation:08d}"
    ):
        raise Task12CorrelationError("Task 11 handoff generation drifted")
    for field in (
        "campaign_identity_sha256",
        "decision_file_sha256",
        "decision_body_sha256",
        "task_yaml_sha256",
        "request_body_sha256",
        "api_server_identity_sha256",
        "handoff_body_sha256",
    ):
        _sha(body[field], "Task 11 handoff " + field)
    canonical_body = {
        name: item
        for name, item in body.items()
        if name != "handoff_body_sha256"
    }
    if body["handoff_body_sha256"] != canonical_sha256(canonical_body):
        raise Task12CorrelationError("Task 11 handoff self-hash drifted")
    for field in (
        "sky_post_action_key",
        "submit_attempt_id",
        "decision_key",
        "decision_version_id",
        "expected_sky_job_name",
    ):
        _text(body[field], "Task 11 handoff " + field)
    consumed = _utc(body["sky_post_consumed_at"], "Task 11 handoff consumed_at")
    completed = _utc(
        body["post_completed_or_lost_at"],
        "Task 11 handoff post completion",
    )
    outcome = body["sky_post_outcome_class"]
    if outcome not in _HANDOFF_OUTCOMES:
        raise Task12CorrelationError("Task 11 handoff outcome drifted")
    started_value = body["post_started_at"]
    if started_value is None:
        if outcome != "PROVED_NOT_SENT_OWNER_DIED":
            raise Task12CorrelationError(
                "Task 11 handoff post-start nullability drifted"
            )
        started = None
    else:
        started = _utc(started_value, "Task 11 handoff post start")
    if completed < consumed or (
        started is not None and (started < consumed or completed < started)
    ):
        raise Task12CorrelationError("Task 11 handoff POST chronology drifted")
    request_id = body["sky_request_id"]
    if request_id is not None and (
        type(request_id) is not str or _REQUEST_ID.fullmatch(request_id) is None
    ):
        raise Task12CorrelationError("Task 11 handoff request ID is invalid")
    if outcome.endswith("_OWNER_DIED") and request_id is not None:
        raise Task12CorrelationError(
            "owner-death handoff cannot assert a request ID"
        )
    if body["binding_state"] != "reconcile-required":
        raise Task12CorrelationError("Task 11 handoff binding state drifted")
    return body


def build_task11_handoff_authority(
    *,
    bucket: object,
    key: object,
    version_id: object,
    file_sha256: object,
    body: object,
    approved_task11_workflow_version_arn: object,
    task11_execution_arn: object,
) -> Task11HandoffAuthority:
    """Close Task 12 over one immutable, approved Task 11 handoff object."""

    exact_bucket = _text(bucket, "handoff bucket")
    if not exact_bucket.endswith("-246813579024-us-west-2"):
        raise Task12CorrelationError("handoff bucket is not the production owner bucket")
    body = validate_sky_post_handoff_record(body)
    generation = body["generation"]
    generation_text = f"{generation:08d}"
    expected_key = (
        f"campaigns/{RUN_ID}/submissions/production/generations/{generation_text}/"
        "handoff/SKY_POST_HANDOFF.json"
    )
    if key != expected_key:
        raise Task12CorrelationError("Task 11 handoff coordinate is not exact")
    exact_version = _text(version_id, "handoff VersionId")
    if _VERSION.fullmatch(exact_version) is None:
        raise Task12CorrelationError("handoff VersionId is not opaque")
    exact_file_sha = _sha(file_sha256, "handoff file hash")
    if (
        type(approved_task11_workflow_version_arn) is not str
        or _TASK11_VERSION_ARN.fullmatch(approved_task11_workflow_version_arn) is None
    ):
        raise Task12CorrelationError("approved Task 11 workflow version is required")
    if (
        type(task11_execution_arn) is not str
        or _TASK11_EXECUTION_ARN.fullmatch(task11_execution_arn) is None
    ):
        raise Task12CorrelationError("Task 11 execution identity is not exact")
    return Task11HandoffAuthority(
        bucket=exact_bucket,
        key=expected_key,
        version_id=exact_version,
        file_sha256=exact_file_sha,
        body=dict(body),
        body_sha256=str(body["handoff_body_sha256"]),
        approved_task11_workflow_version_arn=approved_task11_workflow_version_arn,
        task11_execution_arn=task11_execution_arn,
    )


def _parse_immutable_tuple(value: object) -> ImmutableRequestTuple:
    if type(value) is not dict or set(value) != _TUPLE_FIELDS:
        raise Task12CorrelationError("immutable request tuple is not closed")
    generation = value["generation"]
    if type(generation) is not int or isinstance(generation, bool) or generation < 1:
        raise Task12CorrelationError("request tuple generation is invalid")
    for field in ("user_id", "request_kind", "activation_id", "action_key", "expected_sky_job_name"):
        _text(value[field], "request tuple " + field)
    for field in ("envelope_sha256", "task_yaml_sha256", "request_body_sha256"):
        _sha(value[field], "request tuple " + field)
    open_at = _utc(value["window_open_at"], "request tuple window open")
    close_at = _utc(value["window_close_at"], "request tuple window close")
    if close_at <= open_at:
        raise Task12CorrelationError("request tuple window is not positive")
    return ImmutableRequestTuple(
        user_id=str(value["user_id"]),
        request_kind=str(value["request_kind"]),
        activation_id=str(value["activation_id"]),
        generation=generation,
        action_key=str(value["action_key"]),
        envelope_sha256=str(value["envelope_sha256"]),
        task_yaml_sha256=str(value["task_yaml_sha256"]),
        request_body_sha256=str(value["request_body_sha256"]),
        expected_sky_job_name=str(value["expected_sky_job_name"]),
        window_open_at=str(value["window_open_at"]),
        window_close_at=str(value["window_close_at"]),
    )


def _tuple_from_mapping(
    value: object, authority: Task11HandoffAuthority
) -> ImmutableRequestTuple:
    parsed = _parse_immutable_tuple(value)
    body = authority.body
    expected = {
        "generation": body["generation"],
        "action_key": body["sky_post_action_key"],
        "task_yaml_sha256": body["task_yaml_sha256"],
        "request_body_sha256": body["request_body_sha256"],
        "expected_sky_job_name": body["expected_sky_job_name"],
    }
    if any(getattr(parsed, name) != expected_value for name, expected_value in expected.items()):
        raise Task12CorrelationError("request tuple drifted from Task 11 handoff")
    return parsed


def _request_from_mapping(value: object, expected: ImmutableRequestTuple) -> CorrelatedRequest:
    if type(value) is not dict:
        raise Task12CorrelationError("correlated request is not an object")
    required = {"request_id", "state"} | _TUPLE_FIELDS
    if set(value) != required:
        raise Task12CorrelationError("correlated request schema is not closed")
    request_id = value["request_id"]
    if type(request_id) is not str or _REQUEST_ID.fullmatch(request_id) is None:
        raise Task12CorrelationError("correlated request ID is invalid")
    if type(value["state"]) is not str or not value["state"]:
        raise Task12CorrelationError("correlated request state is invalid")
    actual = _parse_immutable_tuple(
        {name: value[name] for name in _TUPLE_FIELDS}
    )
    if actual != expected:
        raise Task12CorrelationError("correlated request immutable tuple drifted")
    return CorrelatedRequest(request_id=request_id, state=str(value["state"]), body=dict(value))


def correlate_request(
    *,
    authority: Task11HandoffAuthority,
    correlation_tuple: object,
    reader: RequestCorrelationReader,
    maximum_pages: int = 256,
) -> RequestCorrelation:
    """Correlate only the exact UUID or a complete immutable-tuple scan."""

    if type(authority) is not Task11HandoffAuthority:
        raise Task12CorrelationError("Task 11 handoff authority must be typed")
    expected = _tuple_from_mapping(correlation_tuple, authority)
    if type(maximum_pages) is not int or isinstance(maximum_pages, bool) or maximum_pages < 1:
        raise Task12CorrelationError("maximum pages is invalid")
    request_id = authority.body["sky_request_id"]
    if request_id is not None:
        read = getattr(reader, "read_exact_request", None)
        if not callable(read):
            raise Task12CorrelationError("reader lacks exact request read")
        result = _request_from_mapping(read(request_id), expected)
        if result.request_id != request_id:
            raise Task12CorrelationError("exact request read returned a foreign request")
        return RequestCorrelation(kind="ONE", matches=(result,))

    scan = getattr(reader, "scan_requests", None)
    if not callable(scan):
        raise Task12CorrelationError("reader lacks tuple scan")
    cursor: str | None = None
    seen_cursors: set[str] = set()
    matches: list[CorrelatedRequest] = []
    for _ in range(maximum_pages):
        response = scan(correlation_tuple, cursor)
        if type(response) is not dict or set(response) != {"matches", "next_cursor"}:
            raise Task12CorrelationError("tuple scan page is not closed")
        page_matches = response["matches"]
        next_cursor = response["next_cursor"]
        if type(page_matches) is not list:
            raise Task12CorrelationError("tuple scan matches are invalid")
        if next_cursor is not None and (
            type(next_cursor) is not str or not next_cursor
        ):
            raise Task12CorrelationError("tuple scan cursor is invalid")
        matches.extend(_request_from_mapping(item, expected) for item in page_matches)
        if next_cursor is None:
            break
        if next_cursor in seen_cursors or next_cursor == cursor:
            raise Task12CorrelationError("tuple scan cursor did not progress")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    else:
        raise Task12CorrelationError("tuple scan exceeded bounded pages")
    request_ids = [item.request_id for item in matches]
    if request_ids != sorted(set(request_ids)):
        raise Task12CorrelationError("tuple scan matches are not unique and sorted")
    kind = "ZERO" if not matches else ("ONE" if len(matches) == 1 else "MULTIPLE")
    return RequestCorrelation(kind=kind, matches=tuple(matches))


def _correlation_from_mapping(value: object) -> RequestCorrelation:
    if type(value) is RequestCorrelation:
        return value
    if type(value) is not dict or set(value) != {"kind", "matches"}:
        raise Task12CorrelationError("request correlation is not closed")
    kind = value["kind"]
    matches = value["matches"]
    if type(matches) is not list or kind not in {"ZERO", "ONE", "MULTIPLE"}:
        raise Task12CorrelationError("request correlation is invalid")
    if (kind == "ZERO" and matches) or (kind == "ONE" and len(matches) != 1) or (kind == "MULTIPLE" and len(matches) < 2):
        raise Task12CorrelationError("request correlation cardinality drifted")
    parsed = []
    for item in matches:
        if type(item) is not dict or type(item.get("request_id")) is not str or type(item.get("state")) is not str:
            raise Task12CorrelationError("request correlation match is invalid")
        parsed.append(CorrelatedRequest(item["request_id"], item["state"], dict(item)))
    return RequestCorrelation(kind=kind, matches=tuple(parsed))


def build_no_job_proof(
    *,
    correlation: object,
    quiescence_snapshots: object,
    minimum_quiescence_seconds: object,
) -> NoJobProof:
    """Require a terminal request followed by two complete zero-work scans."""

    result = _correlation_from_mapping(correlation)
    if result.kind != "ONE":
        raise Task12CorrelationError("no-job proof requires one exact request")
    request = result.matches[0]
    if request.state not in {"FAILED", "CANCELLED"}:
        raise Task12CorrelationError("no-job proof requires terminal FAILED or CANCELLED request")
    if (
        type(minimum_quiescence_seconds) is not int
        or isinstance(minimum_quiescence_seconds, bool)
        or minimum_quiescence_seconds < 1
    ):
        raise Task12CorrelationError("minimum quiescence interval is invalid")
    if type(quiescence_snapshots) is not tuple or len(quiescence_snapshots) != 2:
        raise Task12CorrelationError("no-job proof requires two quiescence snapshots")
    parsed_times: list[datetime] = []
    for snapshot in quiescence_snapshots:
        if type(snapshot) is not dict or set(snapshot) != _QUIESCENCE_FIELDS:
            raise Task12CorrelationError("quiescence snapshot is not complete")
        observed_at = _utc(snapshot["observed_at"], "quiescence observed_at")
        parsed_times.append(observed_at)
        for field in _QUIESCENCE_FIELDS - {"observed_at"}:
            if type(snapshot[field]) is not list or snapshot[field] != []:
                raise Task12CorrelationError("quiescence snapshot has live work")
    if parsed_times[1] <= parsed_times[0] or (
        parsed_times[1] - parsed_times[0]
    ).total_seconds() < minimum_quiescence_seconds:
        raise Task12CorrelationError("quiescence snapshots are not sufficiently separated")
    return NoJobProof(
        request_id=request.request_id,
        terminal_state=request.state,
        first_quiescence_at=parsed_times[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        second_quiescence_at=parsed_times[1].strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def resolve_numeric_binding(
    *,
    correlation: object,
    expected_sky_job_name: object,
    reader: object,
) -> NumericJobBinding:
    """Bind one positive numeric job without treating binding as terminality."""

    result = _correlation_from_mapping(correlation)
    if result.kind != "ONE":
        raise Task12CorrelationError("numeric binding requires one exact request")
    expected_name = _text(expected_sky_job_name, "expected Sky job name")
    if not callable(reader):
        raise Task12CorrelationError("numeric binding reader is not callable")
    value = reader(result.matches[0].request_id)
    if type(value) is not dict or set(value) != {
        "request_id", "numeric_job_id", "sky_job_name", "state"
    }:
        raise Task12CorrelationError("numeric binding response is not closed")
    if value["request_id"] != result.matches[0].request_id:
        raise Task12CorrelationError("numeric binding response is for a foreign request")
    job_id = value["numeric_job_id"]
    if type(job_id) is not int or isinstance(job_id, bool) or job_id <= 0:
        raise Task12CorrelationError("numeric job ID must be positive")
    if value["sky_job_name"] != expected_name:
        raise Task12CorrelationError("numeric binding job name drifted")
    if type(value["state"]) is not str or not value["state"]:
        raise Task12CorrelationError("numeric binding state is invalid")
    return NumericJobBinding(
        request_id=result.matches[0].request_id,
        numeric_job_id=job_id,
        sky_job_name=expected_name,
        state=value["state"],
    )


__all__ = [
    "CorrelatedRequest",
    "ImmutableRequestTuple",
    "NoJobProof",
    "NumericJobBinding",
    "RequestCorrelation",
    "RequestCorrelationReader",
    "Task11HandoffAuthority",
    "Task12CorrelationError",
    "build_no_job_proof",
    "build_task11_handoff_authority",
    "correlate_request",
    "resolve_numeric_binding",
]
