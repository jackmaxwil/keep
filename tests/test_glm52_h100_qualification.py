from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mlx_vq.quality.glm52_h100_qualification import (
    H100QualificationValidationError,
    build_h100_resume_ready,
    build_h100_source_node_ready,
    build_h100_termination_requested,
    validate_h100_resume_ready,
    validate_h100_runtime_allocation,
    validate_h100_source_node_ready,
    validate_h100_termination_requested,
)

RUN_ID = "glm52-sky-20260723"
CAMPAIGN_SHA = "a" * 64
REPO_SHA = "b" * 64
DESCRIPTOR_SHA = "c" * 64
ALLOCATION_RECORD_SHA = "d" * 64
ALLOCATION_BODY_SHA = "e" * 64
CHECKPOINT_SHA = "f" * 64
SOURCE_INSTANCE_ID = "i-0123456789abcdef0"
REPLACEMENT_INSTANCE_ID = "i-11111111111111111"
PUBLISHED_AT = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
REQUESTED_AT = datetime(2026, 7, 24, 10, 1, tzinfo=UTC)
APPROVAL_SHA = "4" * 64
SPEND_AUTHORITY_SHA = "5" * 64
SPEND_LEDGER_SHA = "6" * 64
JOB_ID = f"{RUN_ID}-qualification"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _source_marker() -> dict[str, object]:
    return build_h100_source_node_ready(
        run_id=RUN_ID,
        campaign_identity_sha256=CAMPAIGN_SHA,
        repo_tar_sha256=REPO_SHA,
        descriptor_body_sha256=DESCRIPTOR_SHA,
        instance_id=SOURCE_INSTANCE_ID,
        allocation_record_sha256=ALLOCATION_RECORD_SHA,
        allocation_body_sha256=ALLOCATION_BODY_SHA,
        checkpoint_marker_sha256=CHECKPOINT_SHA,
        published_at=PUBLISHED_AT,
    )


def _allocation() -> dict[str, object]:
    body: dict[str, object] = {
        "record_type": "glm52_gpu_runtime_allocation_v1",
        "run_id": RUN_ID,
        "job_id": JOB_ID,
        "instance_id": SOURCE_INSTANCE_ID,
        "launched_at": "2026-07-24T09:55:00Z",
        "observed_at": "2026-07-24T10:00:00Z",
        "execution_deadline": "2026-07-25T10:00:00Z",
        "approval_sha256": APPROVAL_SHA,
        "gpu_spend_authority_sha256": SPEND_AUTHORITY_SHA,
        "gpu_spend_record_sha256": ALLOCATION_RECORD_SHA,
        "gpu_spend_ledger_sha256": SPEND_LEDGER_SHA,
        "remaining_gpu_seconds": 86_400,
        "estimated_gpu_cost_usd": 0.08,
    }
    return {
        **body,
        "allocation_body_sha256": _canonical_sha256(body),
    }


def _marker() -> dict[str, object]:
    return build_h100_resume_ready(
        run_id=RUN_ID,
        campaign_identity_sha256=CAMPAIGN_SHA,
        repo_tar_sha256=REPO_SHA,
        qualification_cache_manifest_sha256=DESCRIPTOR_SHA,
        first_instance_id=SOURCE_INSTANCE_ID,
        replacement_instance_id=REPLACEMENT_INSTANCE_ID,
        first_allocation_record_sha256="d" * 64,
        replacement_allocation_record_sha256="e" * 64,
        source_checkpoint_marker_sha256="f" * 64,
        parity_report_sha256="1" * 64,
        training_smoke_sha256="2" * 64,
        resumed_capture_sha256="3" * 64,
        peak_gpu_gib=69.5,
        completed_at=datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
    )


def test_runtime_allocation_is_exact_and_binds_approved_authority() -> None:
    allocation = _allocation()

    assert (
        validate_h100_runtime_allocation(
            allocation,
            expected_run_id=RUN_ID,
            expected_job_id=JOB_ID,
            expected_instance_id=SOURCE_INSTANCE_ID,
            expected_approval_sha256=APPROVAL_SHA,
            expected_gpu_spend_authority_sha256=SPEND_AUTHORITY_SHA,
            max_remaining_gpu_seconds=86_400,
            max_estimated_gpu_cost_usd=1_320.96,
        )
        == allocation
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unknown": True}, "schema"),
        ({"record_type": "glm52_gpu_spend_status_v1"}, "schema"),
        ({"instance_id": "i-source"}, "instance"),
        ({"observed_at": "2026-07-24T03:00:00-07:00"}, "canonical"),
        ({"execution_deadline": "2026-07-24T09:59:59Z"}, "deadline"),
        ({"remaining_gpu_seconds": True}, "remaining"),
        ({"estimated_gpu_cost_usd": True}, "cost"),
    ],
)
def test_runtime_allocation_rejects_malformed_or_mixed_authority(
    mutation: dict[str, object],
    message: str,
) -> None:
    allocation = _allocation()
    allocation.update(mutation)
    body = dict(allocation)
    body.pop("allocation_body_sha256")
    allocation["allocation_body_sha256"] = _canonical_sha256(body)

    with pytest.raises(H100QualificationValidationError, match=message):
        validate_h100_runtime_allocation(allocation)


@pytest.mark.parametrize(
    ("expected_name", "expected_value"),
    [
        ("expected_run_id", "foreign-run"),
        ("expected_job_id", "foreign-job"),
        ("expected_instance_id", REPLACEMENT_INSTANCE_ID),
        ("expected_approval_sha256", "0" * 64),
        ("expected_gpu_spend_authority_sha256", "0" * 64),
        ("max_remaining_gpu_seconds", 86_399),
        ("max_estimated_gpu_cost_usd", 0.07),
    ],
)
def test_runtime_allocation_rejects_expected_identity_or_budget_drift(
    expected_name: str,
    expected_value: object,
) -> None:
    expected: dict[str, object] = {
        "expected_run_id": RUN_ID,
        "expected_job_id": JOB_ID,
        "expected_instance_id": SOURCE_INSTANCE_ID,
        "expected_approval_sha256": APPROVAL_SHA,
        "expected_gpu_spend_authority_sha256": SPEND_AUTHORITY_SHA,
        "max_remaining_gpu_seconds": 86_400,
        "max_estimated_gpu_cost_usd": 1_320.96,
    }
    expected[expected_name] = expected_value

    with pytest.raises(H100QualificationValidationError, match="mismatch|exceeds"):
        validate_h100_runtime_allocation(_allocation(), **expected)


def test_qualification_contract_round_trips_with_system_python_used_by_shells() -> None:
    system_python = Path("/usr/bin/python3")
    if not system_python.exists():
        pytest.skip("system python3 is not present on this platform")
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/mlx_vq/quality/glm52_h100_qualification.py"
    )
    completed = subprocess.run(
        [
            str(system_python),
            "-c",
            (
                "import importlib.util, pathlib, sys;"
                "from datetime import datetime, timezone;"
                "p=pathlib.Path(sys.argv[1]);"
                "s=importlib.util.spec_from_file_location('_qual_contract',p);"
                "m=importlib.util.module_from_spec(s);"
                "s.loader.exec_module(m);"
                "now=datetime(2026,7,24,10,0,tzinfo=timezone.utc);"
                "source=m.build_h100_source_node_ready("
                "run_id='glm52-sky-20260723',"
                "campaign_identity_sha256='a'*64,"
                "repo_tar_sha256='b'*64,"
                "descriptor_body_sha256='c'*64,"
                "instance_id='i-0123456789abcdef0',"
                "allocation_record_sha256='d'*64,"
                "allocation_body_sha256='e'*64,"
                "checkpoint_marker_sha256='f'*64,"
                "published_at=now);"
                "request=m.build_h100_termination_requested("
                "source_marker=source,requested_at=now);"
                "m.validate_h100_termination_requested("
                "request,source_marker=source);"
                "ready=m.build_h100_resume_ready("
                "run_id='glm52-sky-20260723',"
                "campaign_identity_sha256='a'*64,"
                "repo_tar_sha256='b'*64,"
                "qualification_cache_manifest_sha256='c'*64,"
                "first_instance_id='i-0123456789abcdef0',"
                "replacement_instance_id='i-11111111111111111',"
                "first_allocation_record_sha256='d'*64,"
                "replacement_allocation_record_sha256='e'*64,"
                "source_checkpoint_marker_sha256='f'*64,"
                "parity_report_sha256='1'*64,"
                "training_smoke_sha256='2'*64,"
                "resumed_capture_sha256='3'*64,"
                "peak_gpu_gib=69.5,completed_at=now);"
                "m.validate_h100_resume_ready(ready)"
            ),
            str(module_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_h100_worker_loads_its_policy_with_isolated_system_python() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "aws/glm52-gpu/skypilot/run_h100_qualification.sh").read_text()

    assert worker.count("/usr/bin/python3 -I -") >= 5
    assert "python3 - \"$DESCRIPTOR\" \"$ALLOCATION\" \"$REPO\"" not in worker


def test_source_node_ready_is_exact_canonical_and_self_authenticated() -> None:
    marker = _source_marker()

    assert marker == validate_h100_source_node_ready(
        marker,
        expected_run_id=RUN_ID,
        expected_campaign_identity_sha256=CAMPAIGN_SHA,
        expected_repo_tar_sha256=REPO_SHA,
        expected_descriptor_body_sha256=DESCRIPTOR_SHA,
        expected_instance_id=SOURCE_INSTANCE_ID,
        expected_allocation_record_sha256=ALLOCATION_RECORD_SHA,
        expected_allocation_body_sha256=ALLOCATION_BODY_SHA,
        expected_checkpoint_marker_sha256=CHECKPOINT_SHA,
        expected_published_at=PUBLISHED_AT,
        expected_source_body_sha256=str(marker["source_body_sha256"]),
    )
    assert marker == {
        "schema_version": 1,
        "record_type": "glm52_h100_qualification_source_v1",
        "run_id": RUN_ID,
        "campaign_identity_sha256": CAMPAIGN_SHA,
        "repo_tar_sha256": REPO_SHA,
        "descriptor_body_sha256": DESCRIPTOR_SHA,
        "instance_id": SOURCE_INSTANCE_ID,
        "allocation_record_sha256": ALLOCATION_RECORD_SHA,
        "allocation_body_sha256": ALLOCATION_BODY_SHA,
        "checkpoint_marker_sha256": CHECKPOINT_SHA,
        "published_at": "2026-07-24T10:00:00Z",
        "source_body_sha256": marker["source_body_sha256"],
    }
    body = dict(marker)
    digest = body.pop("source_body_sha256")
    assert digest == _canonical_sha256(body)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unknown": True}, "schema"),
        ({"record_type": "glm52_h100_qualification_termination_request_v1"}, "schema"),
        ({"run_id": "not valid!"}, "run_id"),
        ({"instance_id": "i-first"}, "instance"),
        ({"repo_tar_sha256": "A" * 64}, "repo_tar_sha256"),
        ({"published_at": "2026-07-24T03:00:00-07:00"}, "canonical"),
    ],
)
def test_source_node_ready_rejects_unknown_mixed_or_malformed_fields(
    mutation: dict[str, object],
    message: str,
) -> None:
    marker = _source_marker()
    marker.update(mutation)
    body = dict(marker)
    body.pop("source_body_sha256")
    marker["source_body_sha256"] = _canonical_sha256(body)

    with pytest.raises(H100QualificationValidationError, match=message):
        validate_h100_source_node_ready(marker)


def test_source_node_ready_rejects_missing_or_tampered_body_identity() -> None:
    marker = _source_marker()
    missing = dict(marker)
    missing.pop("allocation_body_sha256")
    with pytest.raises(H100QualificationValidationError, match="schema"):
        validate_h100_source_node_ready(missing)
    with pytest.raises(H100QualificationValidationError, match="body SHA-256"):
        validate_h100_source_node_ready({**marker, "source_body_sha256": "0" * 64})


@pytest.mark.parametrize(
    ("expected_name", "expected_value"),
    [
        ("expected_run_id", "foreign-run"),
        ("expected_campaign_identity_sha256", "0" * 64),
        ("expected_repo_tar_sha256", "0" * 64),
        ("expected_descriptor_body_sha256", "0" * 64),
        ("expected_instance_id", "i-22222222222222222"),
        ("expected_allocation_record_sha256", "0" * 64),
        ("expected_allocation_body_sha256", "0" * 64),
        ("expected_checkpoint_marker_sha256", "0" * 64),
        ("expected_published_at", "2026-07-24T10:00:01Z"),
        ("expected_source_body_sha256", "0" * 64),
    ],
)
def test_source_node_ready_rejects_expected_identity_mismatch(
    expected_name: str,
    expected_value: str,
) -> None:
    expected: dict[str, str] = {
        "expected_run_id": RUN_ID,
        "expected_campaign_identity_sha256": CAMPAIGN_SHA,
        "expected_repo_tar_sha256": REPO_SHA,
        "expected_descriptor_body_sha256": DESCRIPTOR_SHA,
        "expected_instance_id": SOURCE_INSTANCE_ID,
        "expected_allocation_record_sha256": ALLOCATION_RECORD_SHA,
        "expected_allocation_body_sha256": ALLOCATION_BODY_SHA,
        "expected_checkpoint_marker_sha256": CHECKPOINT_SHA,
        "expected_published_at": "2026-07-24T10:00:00Z",
        "expected_source_body_sha256": str(_source_marker()["source_body_sha256"]),
    }
    expected[expected_name] = expected_value

    with pytest.raises(H100QualificationValidationError, match="identity mismatch"):
        validate_h100_source_node_ready(_source_marker(), **expected)


def test_termination_request_binds_the_complete_source_authority() -> None:
    source = _source_marker()
    request = build_h100_termination_requested(
        source_marker=source,
        requested_at=REQUESTED_AT,
    )

    assert (
        validate_h100_termination_requested(
            request,
            source_marker=source,
        )
        == request
    )
    assert request == {
        "schema_version": 1,
        "record_type": "glm52_h100_qualification_termination_request_v1",
        "run_id": RUN_ID,
        "campaign_identity_sha256": CAMPAIGN_SHA,
        "repo_tar_sha256": REPO_SHA,
        "descriptor_body_sha256": DESCRIPTOR_SHA,
        "instance_id": SOURCE_INSTANCE_ID,
        "allocation_record_sha256": ALLOCATION_RECORD_SHA,
        "allocation_body_sha256": ALLOCATION_BODY_SHA,
        "checkpoint_marker_sha256": CHECKPOINT_SHA,
        "source_body_sha256": source["source_body_sha256"],
        "requested_at": "2026-07-24T10:01:00Z",
        "request_body_sha256": request["request_body_sha256"],
    }
    body = dict(request)
    digest = body.pop("request_body_sha256")
    assert digest == _canonical_sha256(body)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unknown": True}, "schema"),
        ({"record_type": "glm52_h100_qualification_source_v1"}, "schema"),
        ({"instance_id": "i-first"}, "instance"),
        ({"allocation_body_sha256": "x" * 64}, "allocation_body_sha256"),
        ({"requested_at": "2026-07-24T03:01:00-07:00"}, "canonical"),
    ],
)
def test_termination_request_rejects_unknown_mixed_or_malformed_fields(
    mutation: dict[str, object],
    message: str,
) -> None:
    source = _source_marker()
    request = build_h100_termination_requested(
        source_marker=source,
        requested_at=REQUESTED_AT,
    )
    request.update(mutation)
    body = dict(request)
    body.pop("request_body_sha256")
    request["request_body_sha256"] = _canonical_sha256(body)

    with pytest.raises(H100QualificationValidationError, match=message):
        validate_h100_termination_requested(request, source_marker=source)


def test_termination_request_rejects_tampering_or_source_identity_drift() -> None:
    source = _source_marker()
    request = build_h100_termination_requested(
        source_marker=source,
        requested_at=REQUESTED_AT,
    )
    with pytest.raises(H100QualificationValidationError, match="body SHA-256"):
        validate_h100_termination_requested(
            {**request, "request_body_sha256": "0" * 64},
            source_marker=source,
        )

    foreign = dict(request)
    foreign["checkpoint_marker_sha256"] = "0" * 64
    foreign_body = dict(foreign)
    foreign_body.pop("request_body_sha256")
    foreign["request_body_sha256"] = _canonical_sha256(foreign_body)
    with pytest.raises(H100QualificationValidationError, match="source identity"):
        validate_h100_termination_requested(foreign, source_marker=source)


def test_termination_request_cannot_be_validated_without_source_authority() -> None:
    request = build_h100_termination_requested(
        source_marker=_source_marker(),
        requested_at=REQUESTED_AT,
    )

    with pytest.raises(TypeError, match="source_marker"):
        validate_h100_termination_requested(request)  # type: ignore[call-arg]


@pytest.mark.parametrize("record", ["source", "request"])
def test_qualification_markers_reject_boolean_schema_version(record: str) -> None:
    source = _source_marker()
    marker = (
        source
        if record == "source"
        else build_h100_termination_requested(
            source_marker=source,
            requested_at=REQUESTED_AT,
        )
    )
    marker["schema_version"] = True
    digest_field = "source_body_sha256" if record == "source" else "request_body_sha256"
    body = dict(marker)
    body.pop(digest_field)
    marker[digest_field] = _canonical_sha256(body)

    with pytest.raises(H100QualificationValidationError, match="schema"):
        if record == "source":
            validate_h100_source_node_ready(marker)
        else:
            validate_h100_termination_requested(marker, source_marker=source)


def test_termination_request_cannot_predate_source_publication() -> None:
    with pytest.raises(H100QualificationValidationError, match="precedes"):
        build_h100_termination_requested(
            source_marker=_source_marker(),
            requested_at=datetime(2026, 7, 24, 9, 59, tzinfo=UTC),
        )


def test_h100_ready_requires_cross_node_resume_and_two_training_steps() -> None:
    marker = validate_h100_resume_ready(_marker())
    assert marker["cross_node_resume"] is True
    assert marker["training_sample_steps"] == 2
    assert marker["peak_gpu_gib"] < 70
    assert marker["first_instance_id"] != marker["replacement_instance_id"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("replacement_instance_id", SOURCE_INSTANCE_ID, "distinct"),
        ("training_sample_steps", 20, "two"),
        ("peak_gpu_gib", 70.0, "70 GiB"),
        ("cross_node_resume", False, "cross-node"),
    ],
)
def test_h100_ready_rejects_false_or_over_memory_gate(
    field: str,
    value: object,
    message: str,
) -> None:
    marker = _marker()
    marker[field] = value
    marker.pop("ready_body_sha256")
    with pytest.raises(ValueError, match=message):
        validate_h100_resume_ready(marker, verify_body_sha=False)


def test_qualification_worker_uses_shared_source_marker_contract_then_waits() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "aws/glm52-gpu/skypilot/run_h100_qualification.sh").read_text()
    assert "build_h100_source_node_ready" in worker
    assert "validate_h100_source_node_ready" in worker
    assert "allocation_body_sha256" in worker
    assert "descriptor_body_sha256" in worker
    assert "validate_h100_runtime_allocation" in worker
    assert "--if-none-match '*'" in worker
    assert "PreconditionFailed" in worker
    assert "aws s3api put-object \\" in worker
    assert '--body "$SOURCE_MARKER"' in worker
    assert 'mv "$winner" "$SOURCE_MARKER"' in worker
    assert 'aws s3 cp "$SOURCE_MARKER" "$PREFIX/SOURCE_NODE_READY.json"' not in worker
    assert worker.index("\nvalidate_current_allocation\n") < worker.index(
        '"$REPO/aws/glm52-gpu/scripts/run_spike.sh"'
    )
    assert 'test "$INSTANCE_ID" = "$ACTUAL_INSTANCE_ID"' in worker
    assert worker.index("latest/meta-data/instance-id") < worker.index(
        '"$REPO/aws/glm52-gpu/scripts/run_spike.sh"'
    )
    assert worker.index("--if-none-match '*'") < worker.index("while :")


def test_qualification_worker_requires_real_replacement_and_two_step_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "aws/glm52-gpu/skypilot/run_h100_qualification.sh").read_text()
    spike = (root / "aws/glm52-gpu/scripts/run_spike.sh").read_text()
    submit = (root / "aws/glm52-gpu/scripts/submit_sky_campaign.sh").read_text()
    assert "SOURCE_NODE_READY.json" in worker
    assert "distinct instance" in worker
    assert "--preflight-steps 2" in worker
    assert "H100_RESUME_READY.json" in worker
    assert "GLM52_QUALIFICATION_STAGE" in spike
    assert "source|replacement" in spike
    assert "H100_RESUME_READY.json" in submit
    assert "--qualification" in submit


def test_mac_qualification_supervisor_is_guarded_originator_and_observer() -> None:
    root = Path(__file__).resolve().parents[1]
    supervisor = (
        root / "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
    ).read_text()
    assert "assert_rnd_aws_account.sh" in supervisor
    assert "SOURCE_NODE_READY.json" in supervisor
    assert "QUALIFICATION_TERMINATION_REQUESTED.json" in supervisor
    assert "H100_RESUME_READY.json" in supervisor
    assert "validate_h100_source_node_ready" in supervisor
    assert "validate_h100_termination_requested" in supervisor
    assert "validate_h100_resume_ready" in supervisor
    assert "describe-instances" in supervisor
    assert "head-object" in supervisor
    assert "terminate-instances" not in supervisor
    assert "put-object" not in supervisor
    assert "submit_sky_campaign.sh" in supervisor
    assert "run_h100_qualification_route.py" in supervisor
    assert "run-instances" not in supervisor
    assert "capacity-block" not in supervisor.lower()
    assert "request-spot" not in supervisor
