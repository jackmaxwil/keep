from __future__ import annotations

import pytest

from glm52_enforcement.canonical import canonical_sha256


SHA = "a" * 64


def _handoff_body(*, request_id: str | None = "request-1") -> dict[str, object]:
    body: dict[str, object] = {
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": SHA,
        "generation": 1,
        "generation_text": "00000001",
        "submit_attempt_id": "attempt-1",
        "decision_key": (
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/START_DECISION.json"
        ),
        "decision_version_id": "decision-version-1",
        "decision_file_sha256": SHA,
        "decision_body_sha256": SHA,
        "sky_post_action_key": "action-1",
        "sky_post_consumed_at": "2026-07-29T12:00:00Z",
        "sky_post_outcome_class": "ACCEPTED",
        "expected_sky_job_name": "glm52-sky-20260724",
        "task_yaml_sha256": SHA,
        "request_body_sha256": SHA,
        "api_server_identity_sha256": SHA,
        "sky_request_id": request_id,
        "post_started_at": "2026-07-29T12:00:01Z",
        "post_completed_or_lost_at": "2026-07-29T12:00:02Z",
        "binding_state": "reconcile-required",
        "handoff_body_sha256": "",
    }
    body["handoff_body_sha256"] = canonical_sha256(
        {key: value for key, value in body.items() if key != "handoff_body_sha256"}
    )
    return body


def _tuple() -> dict[str, object]:
    return {
        "user_id": "keep-production",
        "request_kind": "jobs.launch",
        "activation_id": "activation-1",
        "generation": 1,
        "action_key": "action-1",
        "envelope_sha256": SHA,
        "task_yaml_sha256": SHA,
        "request_body_sha256": SHA,
        "expected_sky_job_name": "glm52-sky-20260724",
        "window_open_at": "2026-07-29T12:00:00Z",
        "window_close_at": "2026-07-29T12:10:00Z",
    }


def _request(request_id: str) -> dict[str, object]:
    return {
        "request_id": request_id,
        **_tuple(),
        "state": "FAILED",
    }


class _Reader:
    def __init__(self, *, exact: object = None, pages: list[dict[str, object]] | None = None) -> None:
        self.exact = exact
        self.pages = list(pages or [])
        self.exact_ids: list[str] = []
        self.cursors: list[str | None] = []

    def read_exact_request(self, request_id: str) -> object:
        self.exact_ids.append(request_id)
        return self.exact

    def scan_requests(self, correlation_tuple: object, cursor: str | None) -> object:
        self.cursors.append(cursor)
        if not self.pages:
            raise AssertionError("unexpected extra page read")
        return self.pages.pop(0)


def test_task11_handoff_requires_immutable_coordinate_and_approved_version() -> None:
    from glm52_enforcement.task12_correlation import (
        Task12CorrelationError,
        build_task11_handoff_authority,
    )

    body = _handoff_body()
    with pytest.raises(Task12CorrelationError, match="approved Task 11 workflow"):
        build_task11_handoff_authority(
            bucket="keep-glm52-models-246813579024-us-west-2",
            key=(
                "campaigns/glm52-sky-20260724/submissions/production/"
                "generations/00000001/handoff/SKY_POST_HANDOFF.json"
            ),
            version_id="handoff-version-1",
            file_sha256=SHA,
            body=body,
            approved_task11_workflow_version_arn=(
                "arn:aws:states:us-west-2:246813579024:stateMachine:"
                "keep-glm52-h1g-support"
            ),
            task11_execution_arn=(
                "arn:aws:states:us-west-2:246813579024:execution:"
                "keep-glm52-h1g-support:execution-1"
            ),
        )


def test_uuid_correlation_uses_only_exact_read_and_rejects_tuple_drift() -> None:
    from glm52_enforcement.task12_correlation import (
        build_task11_handoff_authority,
        correlate_request,
    )

    body = _handoff_body(request_id="request-1")
    authority = build_task11_handoff_authority(
        bucket="keep-glm52-models-246813579024-us-west-2",
        key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/handoff/SKY_POST_HANDOFF.json"
        ),
        version_id="handoff-version-1",
        file_sha256=SHA,
        body=body,
        approved_task11_workflow_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-support:42"
        ),
        task11_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-support:execution-1"
        ),
    )
    reader = _Reader(exact=_request("request-1"))

    result = correlate_request(
        authority=authority,
        correlation_tuple=_tuple(),
        reader=reader,
    )

    assert result.kind == "ONE"
    assert tuple(item.request_id for item in result.matches) == ("request-1",)
    assert reader.exact_ids == ["request-1"]
    assert reader.cursors == []


def test_no_uuid_scan_returns_zero_one_or_multiple_without_selecting_one() -> None:
    from glm52_enforcement.task12_correlation import (
        build_task11_handoff_authority,
        correlate_request,
    )

    body = _handoff_body(request_id=None)
    authority = build_task11_handoff_authority(
        bucket="keep-glm52-models-246813579024-us-west-2",
        key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/handoff/SKY_POST_HANDOFF.json"
        ),
        version_id="handoff-version-1",
        file_sha256=SHA,
        body=body,
        approved_task11_workflow_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-support:42"
        ),
        task11_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-support:execution-1"
        ),
    )
    reader = _Reader(
        pages=[
            {"matches": [_request("request-2")], "next_cursor": "page-2"},
            {"matches": [_request("request-3")], "next_cursor": None},
        ]
    )

    result = correlate_request(
        authority=authority,
        correlation_tuple=_tuple(),
        reader=reader,
    )

    assert result.kind == "MULTIPLE"
    assert tuple(item.request_id for item in result.matches) == (
        "request-2",
        "request-3",
    )
    assert reader.cursors == [None, "page-2"]


def test_no_uuid_scan_returns_zero_only_after_a_complete_final_page() -> None:
    from glm52_enforcement.task12_correlation import (
        build_task11_handoff_authority,
        correlate_request,
    )

    authority = build_task11_handoff_authority(
        bucket="keep-glm52-models-246813579024-us-west-2",
        key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/handoff/SKY_POST_HANDOFF.json"
        ),
        version_id="handoff-version-1",
        file_sha256=SHA,
        body=_handoff_body(request_id=None),
        approved_task11_workflow_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-support:42"
        ),
        task11_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-support:execution-1"
        ),
    )
    reader = _Reader(pages=[{"matches": [], "next_cursor": None}])

    result = correlate_request(
        authority=authority,
        correlation_tuple=_tuple(),
        reader=reader,
    )

    assert result.kind == "ZERO"
    assert result.matches == ()


def test_no_uuid_scan_rejects_repeated_cursor_before_claiming_zero() -> None:
    from glm52_enforcement.task12_correlation import (
        Task12CorrelationError,
        build_task11_handoff_authority,
        correlate_request,
    )

    authority = build_task11_handoff_authority(
        bucket="keep-glm52-models-246813579024-us-west-2",
        key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/handoff/SKY_POST_HANDOFF.json"
        ),
        version_id="handoff-version-1",
        file_sha256=SHA,
        body=_handoff_body(request_id=None),
        approved_task11_workflow_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-support:42"
        ),
        task11_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-support:execution-1"
        ),
    )
    reader = _Reader(
        pages=[
            {"matches": [], "next_cursor": "page-1"},
            {"matches": [], "next_cursor": "page-1"},
        ]
    )

    with pytest.raises(Task12CorrelationError, match="cursor"):
        correlate_request(
            authority=authority,
            correlation_tuple=_tuple(),
            reader=reader,
        )


def test_no_job_requires_terminal_request_and_two_complete_separated_scans() -> None:
    from glm52_enforcement.task12_correlation import (
        Task12CorrelationError,
        build_no_job_proof,
    )

    correlation = {
        "kind": "ONE",
        "matches": [_request("request-1")],
    }
    complete = {
        "request_ids": [],
        "job_ids": [],
        "worker_instance_ids": [],
        "allocation_ids": [],
        "controller_work_ids": [],
    }

    proof = build_no_job_proof(
        correlation=correlation,
        quiescence_snapshots=(
            {**complete, "observed_at": "2026-07-29T12:01:00Z"},
            {**complete, "observed_at": "2026-07-29T12:02:00Z"},
        ),
        minimum_quiescence_seconds=60,
    )
    assert proof.request_id == "request-1"

    with pytest.raises(Task12CorrelationError, match="terminal"):
        build_no_job_proof(
            correlation={
                "kind": "ONE",
                "matches": [{**_request("request-1"), "state": "RUNNING"}],
            },
            quiescence_snapshots=(
                {**complete, "observed_at": "2026-07-29T12:01:00Z"},
                {**complete, "observed_at": "2026-07-29T12:02:00Z"},
            ),
            minimum_quiescence_seconds=60,
        )


def test_numeric_binding_accepts_one_exact_positive_job_only() -> None:
    from glm52_enforcement.task12_correlation import (
        Task12CorrelationError,
        resolve_numeric_binding,
    )

    correlation = {"kind": "ONE", "matches": [_request("request-1")]}
    binding = resolve_numeric_binding(
        correlation=correlation,
        expected_sky_job_name="glm52-sky-20260724",
        reader=lambda request_id: {
            "request_id": request_id,
            "numeric_job_id": 17,
            "sky_job_name": "glm52-sky-20260724",
            "state": "RUNNING",
        },
    )
    assert binding.numeric_job_id == 17

    with pytest.raises(Task12CorrelationError, match="positive"):
        resolve_numeric_binding(
            correlation=correlation,
            expected_sky_job_name="glm52-sky-20260724",
            reader=lambda request_id: {
                "request_id": request_id,
                "numeric_job_id": 0,
                "sky_job_name": "glm52-sky-20260724",
                "state": "RUNNING",
            },
        )
