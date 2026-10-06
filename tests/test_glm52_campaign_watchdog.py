"""Campaign watchdog policy tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mlx_vq.quality.glm52_campaign_watchdog import (
    CampaignObservation,
    build_campaign_heartbeat,
    build_skypilot_job_status,
    build_skypilot_submission_marker,
    campaign_notification_decision,
    evaluate_campaign_observation,
    validate_campaign_heartbeat,
    validate_skypilot_job_status,
    validate_skypilot_submission_marker,
)


UTC = timezone.utc
START = datetime(2026, 7, 17, 11, 30, tzinfo=UTC)
LAUNCH_DEADLINE = START + timedelta(minutes=10)


def _observation(**overrides: object) -> CampaignObservation:
    values: dict[str, object] = {
        "now": START + timedelta(minutes=12),
        "launch_deadline": LAUNCH_DEADLINE,
        "instance_ids": ("i-0123",),
        "instance_states": ("running",),
        "instance_launch_time": START,
        "ec2_status_ok": True,
        "ssm_online": True,
        "systemd_state": "active",
        "campaign_phase": "TEACHER",
        "progress_at": START + timedelta(minutes=10),
        "cloudwatch_alarms": (),
        "worker_failure": False,
        "training_deferred": False,
        "drained": False,
    }
    values.update(overrides)
    return CampaignObservation(**values)


def test_waits_before_launch_deadline_without_false_alarm() -> None:
    result = evaluate_campaign_observation(
        _observation(
            now=START - timedelta(minutes=5),
            instance_ids=(),
            instance_states=(),
            instance_launch_time=None,
            ec2_status_ok=None,
            ssm_online=None,
            systemd_state=None,
            progress_at=None,
            campaign_phase=None,
        )
    )
    assert result.status == "waiting"
    assert result.alert is False


def test_alerts_when_no_instance_is_running_by_launch_deadline() -> None:
    result = evaluate_campaign_observation(
        _observation(
            now=LAUNCH_DEADLINE,
            instance_ids=(),
            instance_states=(),
            instance_launch_time=None,
            ec2_status_ok=None,
            ssm_online=None,
            systemd_state=None,
            progress_at=None,
            campaign_phase=None,
        )
    )
    assert result.status == "alarm"
    assert result.alert is True
    assert "no running campaign instance" in result.findings


def test_alerts_on_each_requested_control_plane_failure() -> None:
    cases = [
        ({"ec2_status_ok": False}, "EC2 status checks unhealthy"),
        ({"ssm_online": False}, "SSM agent unavailable"),
        ({"systemd_state": "failed"}, "systemd campaign service failed"),
        (
            {"cloudwatch_alarms": ("keep-glm52-controller-errors",)},
            "CloudWatch alarms active: keep-glm52-controller-errors",
        ),
        ({"worker_failure": True}, "campaign worker failure marker present"),
        ({"training_deferred": True}, "training deferred"),
    ]
    for changes, expected in cases:
        result = evaluate_campaign_observation(_observation(**changes))
        assert result.status == "alarm"
        assert expected in result.findings


def test_alerts_when_progress_heartbeat_is_stale_for_30_minutes() -> None:
    result = evaluate_campaign_observation(
        _observation(
            now=START + timedelta(minutes=41),
            progress_at=START + timedelta(minutes=10),
        )
    )
    assert result.status == "alarm"
    assert "campaign heartbeat/progress stale for 31 minutes" in result.findings


def test_new_instance_gets_30_minutes_to_publish_first_heartbeat() -> None:
    result = evaluate_campaign_observation(
        _observation(
            now=START + timedelta(minutes=20),
            instance_launch_time=START,
            progress_at=None,
        )
    )
    assert result.status == "warning"
    assert result.alert is False
    assert "campaign heartbeat/progress not published yet" in result.findings


def test_skypilot_worker_gets_ten_minute_ssm_bootstrap_grace() -> None:
    result = evaluate_campaign_observation(
        _observation(
            now=START + timedelta(minutes=5),
            instance_launch_time=START,
            campaign_mode="skypilot",
            ssm_online=False,
        )
    )
    assert result.status == "warning"
    assert result.alert is False
    assert "SSM agent unavailable during bootstrap grace" in result.findings


def test_drained_marker_is_terminal_even_after_instance_terminates() -> None:
    result = evaluate_campaign_observation(
        _observation(
            drained=True,
            instance_ids=(),
            instance_states=(),
            instance_launch_time=None,
            ec2_status_ok=None,
            ssm_online=None,
            systemd_state=None,
            progress_at=START + timedelta(minutes=30),
            campaign_phase="DRAINED",
        )
    )
    assert result.status == "drained"
    assert result.alert is False


def test_pending_instance_is_not_treated_as_running_after_launch_deadline() -> None:
    result = evaluate_campaign_observation(
        _observation(
            now=LAUNCH_DEADLINE,
            instance_states=("pending",),
            ec2_status_ok=None,
            ssm_online=None,
            systemd_state=None,
            progress_at=None,
        )
    )
    assert result.status == "alarm"
    assert "no running campaign instance" in result.findings


def test_heartbeat_identity_detects_tampering() -> None:
    payload = build_campaign_heartbeat(
        run_id="glm52-20260717",
        phase="TEACHER",
        observed_at=START,
        ledger_record_sha256="a" * 64,
    )
    assert validate_campaign_heartbeat(payload) == payload

    tampered = {**payload, "phase": "TRAINING"}
    try:
        validate_campaign_heartbeat(tampered)
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("tampered heartbeat was accepted")


def test_campaign_email_notifications_are_stateful_and_deduplicated() -> None:
    first = campaign_notification_decision(
        previous=None,
        status="alarm",
        findings=("campaign heartbeat/progress stale for 31 minutes",),
    )
    assert first.event == "alert"

    repeated = campaign_notification_decision(
        previous=first.state,
        status="alarm",
        findings=("campaign heartbeat/progress stale for 41 minutes",),
    )
    assert repeated.event is None

    recovered = campaign_notification_decision(
        previous=first.state,
        status="healthy",
        findings=("campaign control plane healthy",),
    )
    assert recovered.event == "recovery"

    drained = campaign_notification_decision(
        previous=recovered.state,
        status="drained",
        findings=("CAMPAIGN_DRAINED.json authenticated",),
    )
    assert drained.event == "drained"
    assert campaign_notification_decision(
        previous=drained.state,
        status="drained",
        findings=("CAMPAIGN_DRAINED.json authenticated",),
    ).event is None


def test_skypilot_alerts_after_twenty_minutes_and_at_must_start_deadline() -> None:
    submitted = START
    result = evaluate_campaign_observation(
        _observation(
            now=submitted + timedelta(minutes=20),
            instance_ids=(),
            instance_states=(),
            instance_launch_time=None,
            ec2_status_ok=None,
            ssm_online=None,
            systemd_state=None,
            progress_at=None,
            campaign_phase=None,
            campaign_mode="skypilot",
            submitted_at=submitted,
            must_start_by=submitted + timedelta(hours=12),
            sky_job_status="PENDING",
        )
    )
    assert result.status == "alarm"
    assert "no GPU running 20 minutes after SkyPilot submission" in result.findings

    expired = evaluate_campaign_observation(
        _observation(
            now=submitted + timedelta(hours=12),
            instance_ids=(),
            instance_states=(),
            instance_launch_time=None,
            ec2_status_ok=None,
            ssm_online=None,
            systemd_state=None,
            progress_at=None,
            campaign_phase=None,
            campaign_mode="skypilot",
            submitted_at=submitted,
            must_start_by=submitted + timedelta(hours=12),
            sky_job_status="PENDING",
        )
    )
    assert "SkyPilot must-start deadline reached with no running GPU" in expired.findings


def test_skypilot_pending_is_low_noise_before_twenty_minutes() -> None:
    result = evaluate_campaign_observation(
        _observation(
            now=START + timedelta(minutes=10),
            instance_ids=(),
            instance_states=(),
            instance_launch_time=None,
            ec2_status_ok=None,
            ssm_online=None,
            systemd_state=None,
            progress_at=None,
            campaign_phase=None,
            campaign_mode="skypilot",
            submitted_at=START,
            must_start_by=START + timedelta(hours=12),
            sky_job_status="PENDING",
        )
    )
    assert result.status == "waiting"
    assert result.alert is False
    assert result.findings == ("SkyPilot job pending; no GPU runtime consumed",)


def test_skypilot_job_dlq_and_spend_authority_failures_alert() -> None:
    cases = (
        ({"sky_job_status": "FAILED"}, "SkyPilot Managed Job failed"),
        ({"dlq_depth": 1}, "controller dead-letter queue contains 1 message"),
        ({"remaining_gpu_seconds": 0}, "approved GPU runtime exhausted"),
        ({"remaining_gpu_cost_usd": -0.01}, "approved GPU cost exhausted"),
    )
    for change, expected in cases:
        result = evaluate_campaign_observation(
            _observation(campaign_mode="skypilot", **change)
        )
        assert result.status == "alarm"
        assert expected in result.findings


def test_skypilot_spend_warning_is_visible_without_false_failure() -> None:
    result = evaluate_campaign_observation(
        _observation(
            campaign_mode="skypilot",
            remaining_gpu_seconds=3_600,
            remaining_gpu_cost_usd=55.04,
        )
    )
    assert result.status == "warning"
    assert "approved GPU runtime has 60 minutes remaining" in result.findings


def test_skypilot_submission_marker_is_hash_authenticated() -> None:
    marker = build_skypilot_submission_marker(
        run_id="glm52-sky-20260723",
        descriptor_body_sha256="a" * 64,
        submitted_at=START,
        must_start_by=START + timedelta(hours=12),
        sky_job_name="glm52-sky-20260723",
    )
    assert validate_skypilot_submission_marker(marker) == marker
    try:
        validate_skypilot_submission_marker({**marker, "sky_job_name": "foreign"})
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("tampered submission marker was accepted")


def test_skypilot_job_status_marker_is_strict_and_hash_authenticated() -> None:
    marker = build_skypilot_job_status(
        run_id="glm52-sky-20260723",
        descriptor_body_sha256="a" * 64,
        sky_job_name="glm52-sky-20260723",
        status="RECOVERING",
        instance_id="i-replacement",
        observed_at=START,
    )
    assert validate_skypilot_job_status(marker) == marker

    tampered = {**marker, "status": "SUCCEEDED"}
    try:
        validate_skypilot_job_status(tampered)
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("tampered SkyPilot job status was accepted")

    unknown = {**marker, "unexpected": True}
    try:
        validate_skypilot_job_status(unknown)
    except ValueError as error:
        assert "schema" in str(error)
    else:
        raise AssertionError("unknown SkyPilot job status field was accepted")
