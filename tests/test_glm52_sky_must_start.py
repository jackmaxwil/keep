from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mlx_vq.quality.glm52_sky_must_start import (
    MustStartValidationError,
    build_must_start_alert_delivered,
    build_must_start_cancel_completed,
    build_must_start_cancel_requested,
    build_must_start_cancel_superseded,
    build_must_start_controller_observation,
    build_must_start_job_binding,
    build_must_start_target_binding,
    build_timely_start_accepted,
    build_timely_start_latch,
    build_worker_start_latch,
    canonical_sha256,
    decide_controller_start_authority,
    decide_must_start_action,
    expected_sky_job_name,
    validate_must_start_alert_delivered,
    validate_must_start_cancel_completed,
    validate_must_start_cancel_requested,
    validate_must_start_cancel_superseded,
    validate_must_start_controller_observation,
    validate_must_start_job_binding,
    validate_must_start_target_binding,
    validate_timely_start_accepted,
    validate_timely_start_latch,
    validate_worker_start_latch,
)

UTC = timezone.utc
RUN_ID = "glm52-sky-20260724"
DEADLINE = datetime(2026, 7, 26, 14, 27, 42, tzinfo=UTC)
DESCRIPTOR_SHA = "5" * 64
SUBMISSION_SHA = "8" * 64
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
BUCKET = "keep-glm52-models-246813579024-us-west-2"


def _authority(mode: str = "cache-seed") -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "managed_mode": mode,
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "bucket": BUCKET,
        "descriptor_body_sha256": DESCRIPTOR_SHA,
        "submission_body_sha256": SUBMISSION_SHA,
        "sky_job_name": expected_sky_job_name(RUN_ID, mode),
        "must_start_by": DEADLINE,
    }


def test_mode_maps_to_one_exact_sky_job_name() -> None:
    assert expected_sky_job_name(RUN_ID, "production") == RUN_ID
    assert expected_sky_job_name(RUN_ID, "qualification") == (
        f"{RUN_ID}-qualification"
    )
    assert expected_sky_job_name(RUN_ID, "cache-seed") == f"{RUN_ID}-cache-seed"
    with pytest.raises(MustStartValidationError, match="managed_mode"):
        expected_sky_job_name(RUN_ID, "training")


def test_requested_marker_is_exact_hash_authenticated_and_normalized() -> None:
    marker = build_must_start_cancel_requested(
        **_authority(),
        requested_at=DEADLINE.astimezone(timezone(timedelta(hours=-7))),
    )

    assert marker == validate_must_start_cancel_requested(marker)
    assert marker["record_type"] == "glm52_sky_must_start_cancel_requested_v1"
    assert marker["must_start_by"] == "2026-07-26T14:27:42Z"
    assert marker["requested_at"] == "2026-07-26T14:27:42Z"
    assert len(str(marker["request_body_sha256"])) == 64

    with pytest.raises(MustStartValidationError, match="schema"):
        validate_must_start_cancel_requested({**marker, "unknown": True})
    with pytest.raises(MustStartValidationError, match="SHA-256"):
        validate_must_start_cancel_requested(
            {**marker, "request_body_sha256": "0" * 64}
        )


def test_requested_marker_rejects_foreign_authority_and_wrong_name() -> None:
    with pytest.raises(MustStartValidationError, match="account"):
        build_must_start_cancel_requested(
            **{**_authority(), "account_id": "135792468013"},
            requested_at=DEADLINE,
        )
    with pytest.raises(MustStartValidationError, match="region"):
        build_must_start_cancel_requested(
            **{**_authority(), "region": "us-east-1"},
            requested_at=DEADLINE,
        )
    with pytest.raises(MustStartValidationError, match="job name"):
        build_must_start_cancel_requested(
            **{**_authority(), "sky_job_name": RUN_ID},
            requested_at=DEADLINE,
        )


def test_timely_start_latch_suppresses_only_exact_predeadline_start() -> None:
    latch = build_timely_start_latch(
        **_authority(),
        started_at=DEADLINE - timedelta(seconds=1),
        published_at=DEADLINE - timedelta(seconds=1),
    )
    assert validate_timely_start_latch(latch) == latch
    assert (
        decide_must_start_action(
            now=DEADLINE,
            must_start_by=DEADLINE,
            expected_authority=_authority(),
            timely_start_latch=latch,
            request_marker=None,
            completion_marker=None,
            controller_outcome=None,
        ).action
        == "timely-started"
    )

    late = build_timely_start_latch(
        **_authority(),
        started_at=DEADLINE + timedelta(seconds=1),
        published_at=DEADLINE + timedelta(seconds=1),
    )
    assert validate_timely_start_latch(late) == late
    assert (
        decide_must_start_action(
            now=DEADLINE + timedelta(seconds=1),
            must_start_by=DEADLINE,
            expected_authority=_authority(),
            timely_start_latch=late,
            request_marker=None,
            completion_marker=None,
            controller_outcome=None,
        ).action
        == "request-cancel"
    )


def test_policy_does_nothing_before_deadline_and_requests_at_boundary() -> None:
    before = decide_must_start_action(
        now=DEADLINE - timedelta(microseconds=1),
        must_start_by=DEADLINE,
        expected_authority=_authority(),
        timely_start_latch=None,
        request_marker=None,
        completion_marker=None,
        controller_outcome=None,
    )
    boundary = decide_must_start_action(
        now=DEADLINE,
        must_start_by=DEADLINE,
        expected_authority=_authority(),
        timely_start_latch=None,
        request_marker=None,
        completion_marker=None,
        controller_outcome=None,
    )
    assert before.action == "wait"
    assert boundary.action == "request-cancel"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("pending", "reconcile"),
        ("starting", "reconcile"),
        ("running", "reconcile"),
        ("recovering", "reconcile"),
        ("controller-unknown", "reconcile"),
        ("not-found", "reconcile"),
        ("terminal", "complete"),
        ("cancelled", "complete"),
    ],
)
def test_requested_policy_reconciles_until_exact_terminal_confirmation(
    outcome: str,
    expected: str,
) -> None:
    request = build_must_start_cancel_requested(
        **_authority(),
        requested_at=DEADLINE,
    )
    decision = decide_must_start_action(
        now=DEADLINE + timedelta(minutes=1),
        must_start_by=DEADLINE,
        expected_authority=_authority(),
        timely_start_latch=None,
        request_marker=request,
        completion_marker=None,
        controller_outcome=outcome,
    )
    assert decision.action == expected


def test_completion_binds_request_and_is_strictly_idempotent() -> None:
    request = build_must_start_cancel_requested(
        **_authority(),
        requested_at=DEADLINE,
    )
    completed = build_must_start_cancel_completed(
        **_authority(),
        request_body_sha256=str(request["request_body_sha256"]),
        completed_at=DEADLINE + timedelta(minutes=2),
        terminal_status="CANCELLED",
    )
    assert validate_must_start_cancel_completed(completed) == completed
    assert completed["record_type"] == "glm52_sky_must_start_cancel_completed_v1"
    assert completed["request_body_sha256"] == request["request_body_sha256"]
    assert (
        decide_must_start_action(
            now=DEADLINE + timedelta(minutes=3),
            must_start_by=DEADLINE,
            expected_authority=_authority(),
            timely_start_latch=None,
            request_marker=request,
            completion_marker=completed,
            controller_outcome="cancelled",
        ).action
        == "completed"
    )

    with pytest.raises(MustStartValidationError, match="terminal"):
        build_must_start_cancel_completed(
            **_authority(),
            request_body_sha256=str(request["request_body_sha256"]),
            completed_at=DEADLINE + timedelta(minutes=2),
            terminal_status="RUNNING",
        )
    with pytest.raises(MustStartValidationError, match="completed_at"):
        build_must_start_cancel_completed(
            **_authority(),
            request_body_sha256=str(request["request_body_sha256"]),
            completed_at=DEADLINE - timedelta(seconds=1),
            terminal_status="CANCELLED",
        )


def test_timely_latch_has_precedence_over_existing_completion() -> None:
    request = build_must_start_cancel_requested(
        **_authority(),
        requested_at=DEADLINE,
    )
    completed = build_must_start_cancel_completed(
        **_authority(),
        request_body_sha256=str(request["request_body_sha256"]),
        completed_at=DEADLINE + timedelta(minutes=1),
        terminal_status="CANCELLED",
    )
    latch = build_timely_start_latch(
        **_authority(),
        started_at=DEADLINE - timedelta(seconds=1),
        published_at=DEADLINE - timedelta(seconds=1),
    )
    decision = decide_must_start_action(
        now=DEADLINE + timedelta(minutes=2),
        must_start_by=DEADLINE,
        expected_authority=_authority(),
        timely_start_latch=latch,
        request_marker=request,
        completion_marker=completed,
        controller_outcome="terminal",
    )
    assert decision.action == "timely-started"


def test_policy_rejects_valid_latch_from_another_run_and_deadline() -> None:
    foreign_deadline = DEADLINE + timedelta(minutes=5)
    foreign_authority = {
        **_authority(),
        "run_id": "glm52-sky-foreign",
        "sky_job_name": "glm52-sky-foreign-cache-seed",
        "must_start_by": foreign_deadline,
    }
    foreign_latch = build_timely_start_latch(
        **foreign_authority,
        started_at=foreign_deadline - timedelta(seconds=2),
        published_at=foreign_deadline - timedelta(seconds=1),
    )

    with pytest.raises(MustStartValidationError, match="authority"):
        decide_must_start_action(
            now=foreign_deadline,
            must_start_by=DEADLINE,
            expected_authority=_authority(),
            timely_start_latch=foreign_latch,
            request_marker=None,
            completion_marker=None,
            controller_outcome=None,
        )


def test_policy_requires_completion_to_bind_supplied_exact_request() -> None:
    request = build_must_start_cancel_requested(
        **_authority(),
        requested_at=DEADLINE,
    )
    different_request = build_must_start_cancel_requested(
        **_authority(),
        requested_at=DEADLINE + timedelta(seconds=1),
    )
    completion = build_must_start_cancel_completed(
        **_authority(),
        request_body_sha256=str(different_request["request_body_sha256"]),
        completed_at=DEADLINE + timedelta(minutes=1),
        terminal_status="CANCELLED",
    )

    with pytest.raises(MustStartValidationError, match="exact request"):
        decide_must_start_action(
            now=DEADLINE + timedelta(minutes=2),
            must_start_by=DEADLINE,
            expected_authority=_authority(),
            timely_start_latch=None,
            request_marker=request,
            completion_marker=completion,
            controller_outcome="cancelled",
        )


def test_target_binding_rejects_zero_job_id() -> None:
    request = build_must_start_cancel_requested(
        **_authority(),
        requested_at=DEADLINE,
    )
    with pytest.raises(MustStartValidationError, match="target job ID"):
        build_must_start_target_binding(
            **_authority(),
            request_body_sha256=str(request["request_body_sha256"]),
            target_job_id=0,
            bound_at=DEADLINE,
        )


@pytest.mark.parametrize(
    "marker_kind",
    ["request", "latch", "completion", "target", "alert"],
)
def test_lifecycle_markers_reject_boolean_schema_version(
    marker_kind: str,
) -> None:
    request = build_must_start_cancel_requested(
        **_authority(),
        requested_at=DEADLINE,
    )
    if marker_kind == "request":
        marker = request
        digest_field = "request_body_sha256"
        validator = validate_must_start_cancel_requested
    elif marker_kind == "latch":
        marker = build_timely_start_latch(
            **_authority(),
            started_at=DEADLINE - timedelta(seconds=1),
            published_at=DEADLINE - timedelta(seconds=1),
        )
        digest_field = "latch_body_sha256"
        validator = validate_timely_start_latch
    elif marker_kind == "completion":
        marker = build_must_start_cancel_completed(
            **_authority(),
            request_body_sha256=str(request["request_body_sha256"]),
            completed_at=DEADLINE,
            terminal_status="CANCELLED",
        )
        digest_field = "completion_body_sha256"
        validator = validate_must_start_cancel_completed
    elif marker_kind == "target":
        marker = build_must_start_target_binding(
            **_authority(),
            request_body_sha256=str(request["request_body_sha256"]),
            target_job_id=3,
            bound_at=DEADLINE,
        )
        digest_field = "target_binding_body_sha256"
        validator = validate_must_start_target_binding
    else:
        marker = build_must_start_alert_delivered(
            **_authority(),
            alert_kind="requested",
            lifecycle_body_sha256=str(request["request_body_sha256"]),
            delivered_at=DEADLINE,
        )
        digest_field = "alert_delivery_body_sha256"
        validator = validate_must_start_alert_delivered

    forged = dict(marker)
    forged.pop(digest_field)
    forged["schema_version"] = True
    forged[digest_field] = canonical_sha256(forged)
    with pytest.raises(MustStartValidationError, match="schema"):
        validator(forged)


CURRENT_JOB_ID = 3
CURRENT_WORKSPACE = "default"
CURRENT_CONTROLLER = "i-0511af4e31aa5406a"
TEST_CONTROLLER_TYPE = "m6i.2xlarge"
TEST_CONTROLLER_PROFILE = (
    "arn:aws:iam::246813579024:instance-profile/"
    "keep-glm52-skypilot-controller"
)
TEST_CONTROLLER_CLUSTER = "sky-jobs-controller-current"
CURRENT_ROW_SUBMITTED_AT = datetime(
    2026, 7, 26, 3, 52, 49, 354000, tzinfo=UTC
)


def _controller_observation(
    *,
    status: str = "PENDING",
    schedule_state: str = "LAUNCHING",
    start_at: datetime | None = None,
) -> dict[str, object]:
    return build_must_start_controller_observation(
        **_authority(),
        target_job_id=CURRENT_JOB_ID,
        workspace=CURRENT_WORKSPACE,
        controller_instance_id=CURRENT_CONTROLLER,
        controller_instance_type=TEST_CONTROLLER_TYPE,
        controller_profile_arn=TEST_CONTROLLER_PROFILE,
        controller_cluster_name=TEST_CONTROLLER_CLUSTER,
        status=status,
        schedule_state=schedule_state,
        submitted_at=CURRENT_ROW_SUBMITTED_AT,
        start_at=start_at,
        worker_cluster_name=None,
        recovery_count=0,
        observed_at=DEADLINE,
    )


def test_current_job_binding_and_observation_are_strictly_authenticated() -> None:
    observation = _controller_observation()
    assert validate_must_start_controller_observation(observation) == observation
    binding = build_must_start_job_binding(
        **_authority(),
        descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/seed/"
            "campaign-descriptor-v2.json"
        ),
        descriptor_file_sha256="9" * 64,
        submission_key=(
            f"campaigns/{RUN_ID}/monitor/submission-locks/"
            f"{'9' * 64}-cache-seed.json"
        ),
        submission_submitted_at=datetime(
            2026, 7, 26, 3, 51, 5, 495390, tzinfo=UTC
        ),
        target_job_id=CURRENT_JOB_ID,
        workspace=CURRENT_WORKSPACE,
        controller_instance_id=CURRENT_CONTROLLER,
        controller_instance_type=TEST_CONTROLLER_TYPE,
        controller_profile_arn=TEST_CONTROLLER_PROFILE,
        controller_cluster_name=TEST_CONTROLLER_CLUSTER,
        observation_body_sha256=str(observation["observation_body_sha256"]),
        bound_at=DEADLINE,
    )
    assert validate_must_start_job_binding(binding) == binding
    assert binding["target_job_id"] == 3
    assert binding["workspace"] == "default"

    with pytest.raises(MustStartValidationError, match="schema"):
        validate_must_start_job_binding({**binding, "unknown": True})
    with pytest.raises(MustStartValidationError, match="SHA-256"):
        validate_must_start_controller_observation(
            {**observation, "observation_body_sha256": "0" * 64}
        )


@pytest.mark.parametrize(
    ("status", "start_delta", "now_delta", "expected"),
    [
        ("PENDING", None, -1, "observe"),
        ("PENDING", None, 0, "cancel"),
        ("STARTING", None, 10, "grace"),
        ("STARTING", None, 21, "cancel"),
        ("RUNNING", -1, 30, "accept"),
        ("RECOVERING", -1, 30, "accept"),
        ("SUCCEEDED", -1, 30, "accept"),
        ("RUNNING", 1, 30, "cancel"),
        ("FAILED", None, 30, "fail-closed"),
        ("UNKNOWN", None, 30, "fail-closed"),
    ],
)
def test_controller_start_at_decision_table(
    status: str,
    start_delta: int | None,
    now_delta: int,
    expected: str,
) -> None:
    start_at = (
        None if start_delta is None else DEADLINE + timedelta(seconds=start_delta)
    )
    decision = decide_controller_start_authority(
        now=DEADLINE + timedelta(seconds=now_delta),
        must_start_by=DEADLINE,
        status=status,
        start_at=start_at,
        starting_grace_seconds=20,
    )
    assert decision.action == expected


def test_accepted_start_and_superseded_request_bind_exact_source() -> None:
    observation = _controller_observation(
        status="RUNNING",
        schedule_state="ALIVE",
        start_at=DEADLINE - timedelta(seconds=1),
    )
    accepted = build_timely_start_accepted(
        **_authority(),
        source_kind="controller-observation",
        source_key=(
            f"campaigns/{RUN_ID}/monitor/must-start/cache-seed/"
            f"{SUBMISSION_SHA}/observations/"
            f"{observation['observation_body_sha256']}.json"
        ),
        source_version_id="version-1",
        source_etag='"etag-1"',
        source_body_sha256=str(observation["observation_body_sha256"]),
        source_last_modified=DEADLINE - timedelta(seconds=1),
        started_at=DEADLINE - timedelta(seconds=1),
        accepted_at=DEADLINE,
    )
    assert validate_timely_start_accepted(accepted) == accepted
    request = build_must_start_cancel_requested(
        **_authority(),
        requested_at=DEADLINE,
    )
    superseded = build_must_start_cancel_superseded(
        **_authority(),
        request_body_sha256=str(request["request_body_sha256"]),
        accepted_body_sha256=str(accepted["accepted_body_sha256"]),
        superseded_at=DEADLINE,
    )
    assert validate_must_start_cancel_superseded(superseded) == superseded


def test_worker_latch_binds_instance_role_imds_repo_and_submission() -> None:
    latch = build_worker_start_latch(
        **_authority(),
        descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/seed/"
            "campaign-descriptor-v2.json"
        ),
        descriptor_file_sha256="9" * 64,
        submission_key=(
            f"campaigns/{RUN_ID}/monitor/submission-locks/"
            f"{'9' * 64}-cache-seed.json"
        ),
        submission_submitted_at=datetime(
            2026, 7, 26, 3, 51, 5, 495390, tzinfo=UTC
        ),
        repo_tar_sha256="1" * 64,
        instance_id="i-0123456789abcdef0",
        instance_type="p5.48xlarge",
        image_id="ami-0123456789abcdef0",
        worker_role_arn=(
            "arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"
        ),
        instance_identity_document_sha256="2" * 64,
        ec2_pending_time=DEADLINE - timedelta(seconds=5),
        entrypoint_observed_at=DEADLINE - timedelta(seconds=1),
    )
    assert validate_worker_start_latch(latch) == latch
    assert latch["repo_tar_sha256"] == "1" * 64
    with pytest.raises(MustStartValidationError, match="schema"):
        validate_worker_start_latch({**latch, "unknown": True})
