from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from mlx_vq.quality.glm52_campaign_watchdog import (
    build_skypilot_submission_marker,
)
from mlx_vq.quality.glm52_h100_qualification import (
    build_h100_resume_ready,
    build_h100_source_node_ready,
    build_h100_termination_requested,
    validate_h100_termination_requested,
)
from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import build_must_start_job_binding

ROOT = Path(__file__).resolve().parents[1]
HANDLER = ROOT / "aws/glm52-gpu/lambda/sky_watchdog_handler.py"
PACKAGE = ROOT / "aws/glm52-gpu/scripts/package_sky_watchdog_lambda.sh"
LAMBDA_DIR = HANDLER.parent
UTC = timezone.utc
NOW = datetime(2026, 7, 26, 10, 10, tzinfo=UTC)
RUN_ID = "glm52-sky-20260726"
JOB_ID = f"{RUN_ID}-qualification"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
PREFIX = f"campaigns/{RUN_ID}"
SOURCE_ID = "i-0123456789abcdef0"
REPLACEMENT_ID = "i-11111111111111111"


def _load_handler():
    spec = importlib.util.spec_from_file_location("_sky_watchdog_handler", HANDLER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(LAMBDA_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(LAMBDA_DIR))
    return module


def _sha(character: str) -> str:
    return character * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _descriptor(
    *,
    descriptor_key: str | None = None,
) -> dict[str, object]:
    approval = build_gpu_spend_approval(
        ingested_at=NOW - timedelta(days=1),
        slack_permalink=None,
    )
    return build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=NOW + timedelta(hours=12),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity="arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=f"{PREFIX}/repository/repo.tar.gz",
        repo_tar_sha256=_sha("1"),
        campaign_descriptor_key=(
            descriptor_key or f"{PREFIX}/submissions/first/descriptor.json"
        ),
        approval_key=f"{PREFIX}/authorities/GPU_SPEND_APPROVAL.json",
        approval_sha256=str(approval["approval_body_sha256"]),
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": _sha("2"),
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": _sha("3"),
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_sha256": _sha("4"),
            "frozen_prompt_pack_key": "quality/frozen.json",
            "frozen_prompt_pack_sha256": _sha("5"),
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": _sha("6"),
            "training_config_key": f"{PREFIX}/authorities/training.json",
            "training_config_sha256": _sha("7"),
            "artifact_inventory_key": (
                f"{PREFIX}/inventories/artifact-inventory-{_sha('8')}.json"
            ),
            "artifact_inventory_sha256": _sha("8"),
            "qualification_cache_prefix": (
                f"qualification-cache/seeds/{RUN_ID}/{_sha('9')}/"
            ),
            "qualification_cache_manifest_sha256": _sha("9"),
        },
    )


def _submission(descriptor: dict[str, object]) -> dict[str, object]:
    return build_skypilot_submission_marker(
        run_id=RUN_ID,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submitted_at=NOW - timedelta(minutes=40),
        must_start_by=NOW + timedelta(hours=11, minutes=20),
        sky_job_name=JOB_ID,
    )


def _spend_authority(descriptor: dict[str, object]) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "record_type": "glm52_gpu_spend_ledger_genesis_v1",
                "run_id": RUN_ID,
                "approval_sha256": descriptor["approval_sha256"],
                "approved_gpu_runtime_seconds": descriptor[
                    "approved_gpu_runtime_seconds"
                ],
                "approved_gpu_cost_usd": descriptor["approved_gpu_cost_usd"],
                "hourly_cost_usd": descriptor["max_hourly_cost_usd"],
            }
        )
    ).hexdigest()


def _allocation(
    descriptor: dict[str, object],
    *,
    instance_id: str = SOURCE_ID,
    record_sha: str = _sha("a"),
    job_id: str = JOB_ID,
) -> dict[str, object]:
    body: dict[str, object] = {
        "record_type": "glm52_gpu_runtime_allocation_v1",
        "run_id": RUN_ID,
        "job_id": job_id,
        "instance_id": instance_id,
        "launched_at": "2026-07-26T09:50:00Z",
        "observed_at": "2026-07-26T10:00:00Z",
        "execution_deadline": "2026-07-27T10:00:00Z",
        "approval_sha256": descriptor["approval_sha256"],
        "gpu_spend_authority_sha256": _spend_authority(descriptor),
        "gpu_spend_record_sha256": record_sha,
        "gpu_spend_ledger_sha256": _sha("b"),
        "remaining_gpu_seconds": 86_400,
        "estimated_gpu_cost_usd": 9.17,
    }
    return {
        **body,
        "allocation_body_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _source(
    descriptor: dict[str, object],
    allocation: dict[str, object],
) -> dict[str, object]:
    return build_h100_source_node_ready(
        run_id=RUN_ID,
        campaign_identity_sha256=str(descriptor["campaign_identity_sha256"]),
        repo_tar_sha256=str(descriptor["repo_tar_sha256"]),
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        instance_id=str(allocation["instance_id"]),
        allocation_record_sha256=str(allocation["gpu_spend_record_sha256"]),
        allocation_body_sha256=str(allocation["allocation_body_sha256"]),
        checkpoint_marker_sha256=_sha("c"),
        published_at=NOW - timedelta(minutes=5),
    )


def _instance(
    instance_id: str,
    *,
    state: str = "running",
    instance_type: str = "p5.48xlarge",
    tag_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    tags = {
        "project": "keep-glm52",
        "owner": "jack.mazac",
        "model": "glm-5.2",
        "campaign-run-id": RUN_ID,
        "cost-allocation": "glm52-sky-campaign",
    }
    tags.update(tag_overrides or {})
    return {
        "InstanceId": instance_id,
        "InstanceType": instance_type,
        "State": {"Name": state},
        "LaunchTime": NOW - timedelta(minutes=20),
        "Placement": {"AvailabilityZone": "us-west-2a"},
        "Tags": [{"Key": key, "Value": value} for key, value in tags.items()],
    }


class FakeS3:
    def __init__(
        self,
        values: dict[str, dict[str, object]],
        events: list[str],
    ):
        self.values = dict(values)
        self.events = events
        self.race_winner: dict[str, object] | None = None

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
        assert Bucket == BUCKET
        if Key not in self.values:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        return {"Body": io.BytesIO(_canonical(self.values[Key]) + b"\n")}

    def put_object(self, **kwargs):
        assert kwargs["Bucket"] == BUCKET
        assert kwargs["IfNoneMatch"] == "*"
        key = str(kwargs["Key"])
        self.events.append(f"s3:{key.rsplit('/', 1)[-1]}")
        if self.race_winner is not None:
            self.values[key] = self.race_winner
            self.race_winner = None
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "race"}},
                "PutObject",
            )
        if key in self.values:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "exists"}},
                "PutObject",
            )
        self.values[key] = json.loads(kwargs["Body"])
        return {"ETag": '"etag"'}

    def list_objects_v2(self, *, Bucket: str, Prefix: str, MaxKeys: int):  # noqa: N803
        assert Bucket == BUCKET
        keys = sorted(key for key in self.values if key.startswith(Prefix))[:MaxKeys]
        return {
            "Contents": [{"Key": key} for key in keys],
            "KeyCount": len(keys),
            "IsTruncated": False,
        }


class FakeEC2:
    def __init__(
        self,
        *,
        source: dict[str, object] | None,
        active: list[dict[str, object]],
        events: list[str],
    ):
        self.source = source
        self.active = list(active)
        self.events = events
        self.terminate_calls: list[dict[str, object]] = []

    def describe_instances(self, **kwargs):
        if "InstanceIds" in kwargs:
            assert kwargs["InstanceIds"] == [SOURCE_ID]
            instances = [] if self.source is None else [self.source]
        else:
            filters = {item["Name"]: item["Values"] for item in kwargs["Filters"]}
            assert filters["tag:project"] == ["keep-glm52"]
            assert filters["tag:owner"] == ["jack.mazac"]
            assert filters["tag:model"] == ["glm-5.2"]
            assert filters["tag:campaign-run-id"] == [RUN_ID]
            assert filters["tag:cost-allocation"] == ["glm52-sky-campaign"]
            assert filters["instance-type"] == ["p5.48xlarge"]
            assert set(filters["instance-state-name"]) == {"pending", "running"}
            instances = self.active
        return {"Reservations": [{"Instances": instances}]}

    def terminate_instances(self, **kwargs):
        assert kwargs == {"InstanceIds": [SOURCE_ID]}
        self.events.append(f"ec2:terminate:{SOURCE_ID}")
        self.terminate_calls.append(kwargs)
        return {
            "TerminatingInstances": [
                {
                    "InstanceId": SOURCE_ID,
                    "CurrentState": {"Name": "shutting-down"},
                    "PreviousState": {"Name": "running"},
                }
            ]
        }


def _values(
    descriptor: dict[str, object],
    allocation: dict[str, object],
    source: dict[str, object] | None,
    request: dict[str, object] | None = None,
    ready: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    values = {f"{PREFIX}/runtime/GPU_RUNTIME_ALLOCATION.json": allocation}
    if source is not None:
        values[f"{PREFIX}/qualification/SOURCE_NODE_READY.json"] = source
    if request is not None:
        values[f"{PREFIX}/qualification/QUALIFICATION_TERMINATION_REQUESTED.json"] = (
            request
        )
    if ready is not None:
        values[f"{PREFIX}/qualification/H100_RESUME_READY.json"] = ready
    return values


def _coordinate(
    *,
    s3: FakeS3,
    ec2: FakeEC2,
    descriptor: dict[str, object],
    submission_kind: str = "qualification",
):
    return _load_handler().coordinate_h100_qualification_termination(
        s3=s3,
        ec2=ec2,
        bucket=BUCKET,
        prefix=PREFIX,
        descriptor=descriptor,
        submission=_submission(descriptor),
        submission_kind=submission_kind,
        now=NOW,
    )


def test_qualification_coordinator_waits_without_source_authority() -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, allocation, None), events)
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    result = _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)

    assert result == {
        "status": "awaiting-source",
        "source_instance_id": None,
        "request_body_sha256": None,
    }
    assert events == []
    assert ec2.terminate_calls == []


def test_qualification_coordinator_publishes_marker_before_exact_termination() -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    source = _source(descriptor, allocation)
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, allocation, source), events)
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    result = _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)

    request = s3.values[
        f"{PREFIX}/qualification/QUALIFICATION_TERMINATION_REQUESTED.json"
    ]
    validate_h100_termination_requested(request, source_marker=source)
    assert result["status"] == "termination-requested"
    assert result["source_instance_id"] == SOURCE_ID
    assert result["request_body_sha256"] == request["request_body_sha256"]
    assert events == [
        "s3:QUALIFICATION_TERMINATION_REQUESTED.json",
        f"ec2:terminate:{SOURCE_ID}",
    ]
    assert ec2.terminate_calls == [{"InstanceIds": [SOURCE_ID]}]


def test_qualification_coordinator_converges_after_conditional_put_race() -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    source = _source(descriptor, allocation)
    winner = build_h100_termination_requested(
        source_marker=source,
        requested_at=NOW - timedelta(minutes=1),
    )
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, allocation, source), events)
    s3.race_winner = winner
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    result = _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)

    assert result["status"] == "termination-race-converged"
    assert result["request_body_sha256"] == winner["request_body_sha256"]
    assert events == [
        "s3:QUALIFICATION_TERMINATION_REQUESTED.json",
        f"ec2:terminate:{SOURCE_ID}",
    ]


def test_qualification_coordinator_rejects_foreign_race_winner() -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    source = _source(descriptor, allocation)
    replacement_allocation = _allocation(
        descriptor,
        instance_id=REPLACEMENT_ID,
        record_sha=_sha("d"),
    )
    foreign_source = _source(descriptor, replacement_allocation)
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, allocation, source), events)
    s3.race_winner = build_h100_termination_requested(
        source_marker=foreign_source,
        requested_at=NOW,
    )
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="source identity|foreign"):
        _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)

    assert events == ["s3:QUALIFICATION_TERMINATION_REQUESTED.json"]
    assert ec2.terminate_calls == []


def test_qualification_coordinator_retries_existing_request_idempotently() -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    source = _source(descriptor, allocation)
    request = build_h100_termination_requested(
        source_marker=source,
        requested_at=NOW - timedelta(minutes=1),
    )
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, allocation, source, request), events)
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    result = _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)

    assert result["status"] == "termination-retried"
    assert events == [f"ec2:terminate:{SOURCE_ID}"]
    assert ec2.terminate_calls == [{"InstanceIds": [SOURCE_ID]}]


def test_existing_request_still_rejects_active_source_allocation_drift() -> None:
    descriptor = _descriptor()
    source_allocation = _allocation(descriptor)
    source = _source(descriptor, source_allocation)
    request = build_h100_termination_requested(
        source_marker=source,
        requested_at=NOW - timedelta(minutes=1),
    )
    replacement_allocation = _allocation(
        descriptor,
        instance_id=REPLACEMENT_ID,
        record_sha=_sha("d"),
    )
    events: list[str] = []
    s3 = FakeS3(
        _values(descriptor, replacement_allocation, source, request),
        events,
    )
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="allocation|identity mismatch"):
        _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)
    assert events == []
    assert ec2.terminate_calls == []


def test_qualification_coordinator_never_terminates_replacement() -> None:
    descriptor = _descriptor()
    source_allocation = _allocation(descriptor)
    source = _source(descriptor, source_allocation)
    request = build_h100_termination_requested(
        source_marker=source,
        requested_at=NOW - timedelta(minutes=1),
    )
    replacement_allocation = _allocation(
        descriptor,
        instance_id=REPLACEMENT_ID,
        record_sha=_sha("d"),
    )
    events: list[str] = []
    s3 = FakeS3(
        _values(descriptor, replacement_allocation, source, request),
        events,
    )
    ec2 = FakeEC2(
        source=None,
        active=[_instance(REPLACEMENT_ID)],
        events=events,
    )

    result = _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)

    assert result["status"] == "replacement-protected"
    assert result["source_instance_id"] == SOURCE_ID
    assert events == []
    assert ec2.terminate_calls == []


def test_qualification_coordinator_binds_replacement_to_current_allocation() -> None:
    descriptor = _descriptor()
    source_allocation = _allocation(descriptor)
    source = _source(descriptor, source_allocation)
    request = build_h100_termination_requested(
        source_marker=source,
        requested_at=NOW - timedelta(minutes=1),
    )
    replacement_allocation = _allocation(
        descriptor,
        instance_id=REPLACEMENT_ID,
        record_sha=_sha("d"),
    )
    foreign_replacement_id = "i-22222222222222222"
    events: list[str] = []
    s3 = FakeS3(
        _values(descriptor, replacement_allocation, source, request),
        events,
    )
    ec2 = FakeEC2(
        source=None,
        active=[_instance(foreign_replacement_id)],
        events=events,
    )

    with pytest.raises(ValueError, match="replacement.*allocation"):
        _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)
    assert events == []
    assert ec2.terminate_calls == []


def test_still_described_terminal_source_protects_bound_replacement() -> None:
    descriptor = _descriptor()
    source_allocation = _allocation(descriptor)
    source = _source(descriptor, source_allocation)
    request = build_h100_termination_requested(
        source_marker=source,
        requested_at=NOW - timedelta(minutes=1),
    )
    replacement_allocation = _allocation(
        descriptor,
        instance_id=REPLACEMENT_ID,
        record_sha=_sha("d"),
    )
    events: list[str] = []
    s3 = FakeS3(
        _values(descriptor, replacement_allocation, source, request),
        events,
    )
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID, state="terminated"),
        active=[_instance(REPLACEMENT_ID)],
        events=events,
    )

    result = _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)

    assert result["status"] == "replacement-protected"
    assert events == []
    assert ec2.terminate_calls == []


def test_qualification_coordinator_rejects_allocation_or_instance_drift() -> None:
    descriptor = _descriptor()
    source_allocation = _allocation(descriptor)
    source = _source(descriptor, source_allocation)
    foreign_allocation = _allocation(
        descriptor,
        instance_id=REPLACEMENT_ID,
        record_sha=_sha("d"),
    )
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, foreign_allocation, source), events)
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="allocation|source"):
        _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)
    assert events == []
    assert ec2.terminate_calls == []

    s3 = FakeS3(_values(descriptor, source_allocation, source), events)
    ec2 = FakeEC2(
        source=_instance(
            SOURCE_ID,
            tag_overrides={"cost-allocation": "foreign"},
        ),
        active=[_instance(SOURCE_ID)],
        events=events,
    )
    with pytest.raises(ValueError, match="tag"):
        _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)
    assert ec2.terminate_calls == []


def test_qualification_coordinator_rejects_foreign_source_authority() -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    foreign_source = build_h100_source_node_ready(
        run_id=RUN_ID,
        campaign_identity_sha256=str(descriptor["campaign_identity_sha256"]),
        repo_tar_sha256=_sha("0"),
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        instance_id=SOURCE_ID,
        allocation_record_sha256=str(allocation["gpu_spend_record_sha256"]),
        allocation_body_sha256=str(allocation["allocation_body_sha256"]),
        checkpoint_marker_sha256=_sha("c"),
        published_at=NOW - timedelta(minutes=5),
    )
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, allocation, foreign_source), events)
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)
    assert events == []
    assert ec2.terminate_calls == []


@pytest.mark.parametrize(
    ("instance_mutation", "message"),
    [
        ({"InstanceType": "g5.48xlarge"}, "type"),
        ({"InstanceLifecycle": "spot"}, "on-demand"),
        ({"CapacityReservationId": "cr-0123456789abcdef0"}, "capacity"),
    ],
)
def test_qualification_coordinator_rejects_market_or_type_drift(
    instance_mutation: dict[str, object],
    message: str,
) -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    source = _source(descriptor, allocation)
    source_instance = _instance(SOURCE_ID)
    source_instance.update(instance_mutation)
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, allocation, source), events)
    ec2 = FakeEC2(
        source=source_instance,
        active=[source_instance],
        events=events,
    )

    with pytest.raises(ValueError, match=message):
        _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)
    assert events == []
    assert ec2.terminate_calls == []


def test_qualification_coordinator_rejects_ambiguous_active_workers() -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    source = _source(descriptor, allocation)
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, allocation, source), events)
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID), _instance(REPLACEMENT_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="exactly one|ambiguous"):
        _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)
    assert events == []
    assert ec2.terminate_calls == []


@pytest.mark.parametrize("submission_kind", ["production", "cache-seed"])
def test_nonqualification_submissions_cannot_trigger_termination(
    submission_kind: str,
) -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    source = _source(descriptor, allocation)
    events: list[str] = []
    s3 = FakeS3(_values(descriptor, allocation, source), events)
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    result = _coordinate(
        s3=s3,
        ec2=ec2,
        descriptor=descriptor,
        submission_kind=submission_kind,
    )

    assert result["status"] == "not-applicable"
    assert events == []
    assert ec2.terminate_calls == []


def test_valid_ready_chain_stops_qualification_coordination() -> None:
    descriptor = _descriptor()
    source_allocation = _allocation(descriptor)
    source = _source(descriptor, source_allocation)
    request = build_h100_termination_requested(
        source_marker=source,
        requested_at=NOW - timedelta(minutes=1),
    )
    replacement_allocation = _allocation(
        descriptor,
        instance_id=REPLACEMENT_ID,
        record_sha=_sha("d"),
    )
    ready = build_h100_resume_ready(
        run_id=RUN_ID,
        campaign_identity_sha256=str(descriptor["campaign_identity_sha256"]),
        repo_tar_sha256=str(descriptor["repo_tar_sha256"]),
        qualification_cache_manifest_sha256=str(
            descriptor["artifacts"]["qualification_cache_manifest_sha256"]
        ),
        first_instance_id=SOURCE_ID,
        replacement_instance_id=REPLACEMENT_ID,
        first_allocation_record_sha256=str(
            source_allocation["gpu_spend_record_sha256"]
        ),
        replacement_allocation_record_sha256=str(
            replacement_allocation["gpu_spend_record_sha256"]
        ),
        source_checkpoint_marker_sha256=str(source["checkpoint_marker_sha256"]),
        parity_report_sha256=_sha("e"),
        training_smoke_sha256=_sha("f"),
        resumed_capture_sha256=_sha("0"),
        peak_gpu_gib=69.0,
        completed_at=NOW,
    )
    events: list[str] = []
    s3 = FakeS3(
        _values(descriptor, replacement_allocation, source, request, ready),
        events,
    )
    ec2 = FakeEC2(
        source=None,
        active=[_instance(REPLACEMENT_ID)],
        events=events,
    )

    result = _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)

    assert result["status"] == "ready"
    assert result["request_body_sha256"] == request["request_body_sha256"]
    assert events == []
    assert ec2.terminate_calls == []


def test_ready_without_request_cannot_create_termination_authority() -> None:
    descriptor = _descriptor()
    allocation = _allocation(descriptor)
    source = _source(descriptor, allocation)
    ready = build_h100_resume_ready(
        run_id=RUN_ID,
        campaign_identity_sha256=str(descriptor["campaign_identity_sha256"]),
        repo_tar_sha256=str(descriptor["repo_tar_sha256"]),
        qualification_cache_manifest_sha256=str(
            descriptor["artifacts"]["qualification_cache_manifest_sha256"]
        ),
        first_instance_id=SOURCE_ID,
        replacement_instance_id=REPLACEMENT_ID,
        first_allocation_record_sha256=str(allocation["gpu_spend_record_sha256"]),
        replacement_allocation_record_sha256=_sha("d"),
        source_checkpoint_marker_sha256=str(source["checkpoint_marker_sha256"]),
        parity_report_sha256=_sha("e"),
        training_smoke_sha256=_sha("f"),
        resumed_capture_sha256=_sha("0"),
        peak_gpu_gib=69.0,
        completed_at=NOW,
    )
    events: list[str] = []
    s3 = FakeS3(
        _values(descriptor, allocation, source, ready=ready),
        events,
    )
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="readiness.*request"):
        _coordinate(s3=s3, ec2=ec2, descriptor=descriptor)
    assert events == []
    assert ec2.terminate_calls == []


def _dynamic_descriptor() -> dict[str, object]:
    return _descriptor(
        descriptor_key=(
            f"{PREFIX}/submissions/qualification/campaign-descriptor-v2.json"
        )
    )


def _worker_start_event() -> dict[str, object]:
    intent_body_sha256 = _sha("c")
    return {
        "schema_version": 1,
        "record_type": "glm52_sky_worker_start_v2_invocation_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": BUCKET,
        "descriptor_key": (
            f"{PREFIX}/submissions/qualification/campaign-descriptor-v2.json"
        ),
        "descriptor_file_sha256": _sha("a"),
        "intent_key": (
            f"{PREFIX}/submissions/qualification/intents/{intent_body_sha256}/"
            "SKYPILOT_SUBMISSION_INTENT.json"
        ),
        "intent_file_sha256": _sha("b"),
        "intent_body_sha256": intent_body_sha256,
        "worker_latch_key": (
            f"{PREFIX}/monitor/must-start/qualification/{intent_body_sha256}/"
            f"worker-latches/{REPLACEMENT_ID}/{_sha('d')}.json"
        ),
        "worker_latch_version_id": "worker-version-1+abc/def==",
    }


def _worker_start_environment(event: dict[str, object]) -> dict[str, str]:
    return {
        "AWS_REGION": "us-west-2",
        "WORKER_START_EXPECTED_ACCOUNT_ID": "246813579024",
        "WORKER_START_BUCKET": str(event["bucket"]),
        "WORKER_START_DESCRIPTOR_KEY": str(event["descriptor_key"]),
        "WORKER_START_DESCRIPTOR_FILE_SHA256": str(event["descriptor_file_sha256"]),
        "WORKER_START_INTENT_KEY": str(event["intent_key"]),
        "WORKER_START_INTENT_FILE_SHA256": str(event["intent_file_sha256"]),
        "WORKER_START_INTENT_BODY_SHA256": str(event["intent_body_sha256"]),
    }


class _WorkerStartContext:
    def __init__(self, remaining_times: list[int] | None = None) -> None:
        self.remaining_times = list(remaining_times or [60_000, 60_000])
        self.calls = 0

    def get_remaining_time_in_millis(self) -> int:
        value = self.remaining_times[min(self.calls, len(self.remaining_times) - 1)]
        self.calls += 1
        return value


def _configure_worker_start_handler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    event: dict[str, object],
):
    module = _load_handler()

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr(module, "datetime", FixedDatetime)
    clients = {name: object() for name in ("sts", "s3", "ec2", "ssm")}
    client_calls: list[tuple[str, str | None]] = []

    def client(name: str, **kwargs: object) -> object:
        client_calls.append((name, kwargs.get("region_name")))
        return clients[name]

    sleep_calls: list[float] = []

    def sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(module.boto3, "client", client)
    monkeypatch.setattr(module.time, "sleep", sleep)
    for legacy_name in (
        "CAMPAIGN_BUCKET",
        "CAMPAIGN_DESCRIPTOR_KEY",
        "ALERT_TOPIC_ARN",
        "WATCHDOG_DLQ_URL",
        "WATCHDOG_RULE_NAME",
    ):
        monkeypatch.delenv(legacy_name, raising=False)
    for name, value in _worker_start_environment(event).items():
        monkeypatch.setenv(name, value)
    return module, clients, client_calls, sleep_calls


def _worker_start_outcome(
    module,
    *,
    status: str,
    reason: str = "exact worker authority accepted",
):
    return module.WorkerStartCoordinatorOutcome(
        status=status,
        decision_action="publish-initial-acceptance",
        reason=reason,
        run_id=RUN_ID,
        instance_id=REPLACEMENT_ID,
        worker_latch_key=str(_worker_start_event()["worker_latch_key"]),
        worker_controller_observation_key=(
            f"{PREFIX}/monitor/must-start/qualification/{_sha('c')}/"
            f"worker-controller-observations/{REPLACEMENT_ID}/{_sha('e')}.json"
        ),
        worker_acceptance_key=(
            f"{PREFIX}/monitor/must-start/qualification/{_sha('c')}/"
            f"worker-acceptances/{REPLACEMENT_ID}/0/"
            "WORKER_START_ACCEPTED.json"
        ),
        worker_acceptance_body_sha256=_sha("f"),
        prior_worker_acceptance_count=0,
        published_observation=True,
        published_acceptance=True,
    )


@pytest.mark.parametrize(
    "status",
    [
        "accepted-initial",
        "accepted-managed-recovery",
        "idempotent-complete",
    ],
)
def test_worker_start_v2_projects_exact_request_services_and_success(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    event = _worker_start_event()
    module, clients, client_calls, sleep_calls = _configure_worker_start_handler(
        monkeypatch,
        event=event,
    )
    observed: dict[str, object] = {}

    def coordinate(*, services, request):
        observed["services"] = services
        observed["request"] = request
        observed["clock"] = services.clock()
        return _worker_start_outcome(module, status=status)

    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )

    result = module.lambda_handler(event, _WorkerStartContext())

    assert vars(observed["request"]) == {
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": BUCKET,
        "descriptor_key": (
            f"{PREFIX}/submissions/qualification/campaign-descriptor-v2.json"
        ),
        "descriptor_file_sha256": _sha("a"),
        "intent_key": (
            f"{PREFIX}/submissions/qualification/intents/{_sha('c')}/"
            "SKYPILOT_SUBMISSION_INTENT.json"
        ),
        "intent_file_sha256": _sha("b"),
        "intent_body_sha256": _sha("c"),
        "worker_latch_key": (
            f"{PREFIX}/monitor/must-start/qualification/{_sha('c')}/"
            f"worker-latches/{REPLACEMENT_ID}/{_sha('d')}.json"
        ),
        "worker_latch_version_id": "worker-version-1+abc/def==",
    }
    services = observed["services"]
    assert services.sts is clients["sts"]
    assert services.s3 is clients["s3"]
    assert services.ec2 is clients["ec2"]
    assert services.ssm is clients["ssm"]
    assert services.sleep is module.time.sleep
    assert observed["clock"] == NOW
    assert observed["clock"].tzinfo is UTC
    assert client_calls == [
        ("sts", "us-west-2"),
        ("s3", "us-west-2"),
        ("ec2", "us-west-2"),
        ("ssm", "us-west-2"),
    ]
    assert sleep_calls == []
    assert result == {
        "status": status,
        "decision_action": "publish-initial-acceptance",
        "reason": "exact worker authority accepted",
        "run_id": RUN_ID,
        "instance_id": REPLACEMENT_ID,
        "worker_latch_key": (
            f"{PREFIX}/monitor/must-start/qualification/{_sha('c')}/"
            f"worker-latches/{REPLACEMENT_ID}/{_sha('d')}.json"
        ),
        "worker_controller_observation_key": (
            f"{PREFIX}/monitor/must-start/qualification/{_sha('c')}/"
            f"worker-controller-observations/{REPLACEMENT_ID}/{_sha('e')}.json"
        ),
        "worker_acceptance_key": (
            f"{PREFIX}/monitor/must-start/qualification/{_sha('c')}/"
            f"worker-acceptances/{REPLACEMENT_ID}/0/"
            "WORKER_START_ACCEPTED.json"
        ),
        "worker_acceptance_body_sha256": _sha("f"),
        "prior_worker_acceptance_count": 0,
        "published_observation": True,
        "published_acceptance": True,
    }


def test_worker_start_v2_waits_locally_then_raises_for_async_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _worker_start_event()
    module, _clients, client_calls, sleep_calls = _configure_worker_start_handler(
        monkeypatch,
        event=event,
    )
    coordinator_calls = 0

    def coordinate(*, services, request):
        nonlocal coordinator_calls
        coordinator_calls += 1
        return _worker_start_outcome(
            module,
            status="waiting-worker-authority",
            reason="controller has not observed exact worker authority",
        )

    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )
    context = _WorkerStartContext()

    with pytest.raises(
        module.WorkerStartAuthorityPending,
        match="controller has not observed exact worker authority",
    ):
        module.lambda_handler(event, context)

    assert coordinator_calls == 3
    assert context.calls == 2
    assert sleep_calls == [1, 2]
    assert client_calls == [
        ("sts", "us-west-2"),
        ("s3", "us-west-2"),
        ("ec2", "us-west-2"),
        ("ssm", "us-west-2"),
    ]


@pytest.mark.parametrize(
    "context",
    [
        _WorkerStartContext([29_999]),
        object(),
        None,
    ],
    ids=["low-time", "missing-method", "missing-context"],
)
def test_worker_start_v2_pending_requires_time_for_another_attempt(
    monkeypatch: pytest.MonkeyPatch,
    context: object,
) -> None:
    event = _worker_start_event()
    module, _clients, _client_calls, sleep_calls = _configure_worker_start_handler(
        monkeypatch,
        event=event,
    )
    coordinator_calls = 0

    def coordinate(*, services, request):
        nonlocal coordinator_calls
        coordinator_calls += 1
        return _worker_start_outcome(
            module,
            status="waiting-worker-authority",
            reason="worker authority is still pending",
        )

    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )

    with pytest.raises(module.WorkerStartAuthorityPending):
        module.lambda_handler(event, context)

    assert coordinator_calls == 1
    assert sleep_calls == []


def test_worker_start_v2_coordinator_error_escapes_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _worker_start_event()
    module, _clients, _client_calls, sleep_calls = _configure_worker_start_handler(
        monkeypatch,
        event=event,
    )
    error = module.WorkerStartCoordinatorError("coordinator failed closed")
    coordinator_calls = 0

    def coordinate(*, services, request):
        nonlocal coordinator_calls
        coordinator_calls += 1
        raise error

    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )

    with pytest.raises(module.WorkerStartCoordinatorError) as caught:
        module.lambda_handler(event, _WorkerStartContext())

    assert caught.value is error
    assert coordinator_calls == 1
    assert sleep_calls == []


def test_worker_start_v2_client_error_escapes_before_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _worker_start_event()
    module, _clients, _client_calls, sleep_calls = _configure_worker_start_handler(
        monkeypatch,
        event=event,
    )
    error = RuntimeError("client construction failed")
    coordinator_calls = 0

    def client(name: str, **kwargs: object) -> object:
        raise error

    def coordinate(*, services, request):
        nonlocal coordinator_calls
        coordinator_calls += 1
        raise AssertionError("coordinator must not run")

    monkeypatch.setattr(module.boto3, "client", client)
    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )

    with pytest.raises(RuntimeError) as caught:
        module.lambda_handler(event, _WorkerStartContext())

    assert caught.value is error
    assert coordinator_calls == 0
    assert sleep_calls == []


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "mixed",
        "foreign",
        "malformed-latch",
        "versionless",
        "coordinator-incompatible-version",
        "boolean-schema",
    ],
)
def test_worker_start_v2_rejects_malformed_events_before_clients(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    event = _worker_start_event()
    if case == "missing":
        event.pop("intent_file_sha256")
    elif case == "extra":
        event["source"] = "aws.s3"
    elif case == "mixed":
        event["record_type"] = "glm52_sky_qualification_watchdog_invocation_v1"
    elif case == "foreign":
        event["region"] = "us-east-1"
    elif case == "malformed-latch":
        event["worker_latch_key"] = (
            f"{PREFIX}/monitor/must-start/qualification/{_sha('c')}/"
            f"worker-latches/not-an-instance/{_sha('d')}.json"
        )
    elif case == "versionless":
        event["worker_latch_version_id"] = ""
    elif case == "coordinator-incompatible-version":
        event["worker_latch_version_id"] = "worker~version"
    elif case == "boolean-schema":
        event["schema_version"] = True
    module, _clients, client_calls, _sleep_calls = _configure_worker_start_handler(
        monkeypatch,
        event=_worker_start_event(),
    )
    coordinator_calls = 0

    def coordinate(*, services, request):
        nonlocal coordinator_calls
        coordinator_calls += 1
        raise AssertionError("coordinator must not run")

    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )

    with pytest.raises(ValueError):
        module.lambda_handler(event, _WorkerStartContext())

    assert client_calls == []
    assert coordinator_calls == 0


def test_worker_start_v2_accepts_coordinator_safe_colon_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _worker_start_event()
    event["worker_latch_version_id"] = "worker:version-1"
    module, _clients, client_calls, _sleep_calls = _configure_worker_start_handler(
        monkeypatch,
        event=event,
    )
    observed_version = None

    def coordinate(*, services, request):
        nonlocal observed_version
        observed_version = request.worker_latch_version_id
        return _worker_start_outcome(module, status="accepted-initial")

    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )

    result = module.lambda_handler(event, _WorkerStartContext())

    assert observed_version == "worker:version-1"
    assert len(client_calls) == 4
    assert result["status"] == "accepted-initial"


@pytest.mark.parametrize(
    "missing_pin",
    [
        "AWS_REGION",
        "WORKER_START_EXPECTED_ACCOUNT_ID",
        "WORKER_START_BUCKET",
        "WORKER_START_DESCRIPTOR_KEY",
        "WORKER_START_DESCRIPTOR_FILE_SHA256",
        "WORKER_START_INTENT_KEY",
        "WORKER_START_INTENT_FILE_SHA256",
        "WORKER_START_INTENT_BODY_SHA256",
    ],
)
def test_worker_start_v2_rejects_missing_environment_pin_before_clients(
    monkeypatch: pytest.MonkeyPatch,
    missing_pin: str,
) -> None:
    event = _worker_start_event()
    module, _clients, client_calls, _sleep_calls = _configure_worker_start_handler(
        monkeypatch,
        event=event,
    )
    monkeypatch.delenv(missing_pin)
    coordinator_calls = 0

    def coordinate(*, services, request):
        nonlocal coordinator_calls
        coordinator_calls += 1
        raise AssertionError("coordinator must not run")

    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )

    with pytest.raises(ValueError):
        module.lambda_handler(event, _WorkerStartContext())

    assert client_calls == []
    assert coordinator_calls == 0


def test_worker_start_v2_rejects_static_pin_mismatch_before_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _worker_start_event()
    module, _clients, client_calls, _sleep_calls = _configure_worker_start_handler(
        monkeypatch,
        event=event,
    )
    monkeypatch.setenv("WORKER_START_BUCKET", "foreign-campaign-bucket")
    coordinator_calls = 0

    def coordinate(*, services, request):
        nonlocal coordinator_calls
        coordinator_calls += 1
        raise AssertionError("coordinator must not run")

    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )

    with pytest.raises(ValueError):
        module.lambda_handler(event, _WorkerStartContext())

    assert client_calls == []
    assert coordinator_calls == 0


def _v2_intent(descriptor: dict[str, object]) -> dict[str, object]:
    descriptor_file_sha256 = hashlib.sha256(_canonical(descriptor) + b"\n").hexdigest()
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_submission_intent_v2",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": RUN_ID,
        "managed_mode": "qualification",
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
        "approval_body_sha256": _sha("a"),
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "cache_seed_acceptance_key": (
            f"{PREFIX}/qualification/cache-seed/accepted/{_sha('b')}.json"
        ),
        "cache_seed_acceptance_file_sha256": _sha("c"),
        "cache_seed_acceptance_body_sha256": _sha("d"),
        "gpu_spend_snapshot_key": f"{PREFIX}/runtime/snapshot-{_sha('e')}.json",
        "gpu_spend_snapshot_sha256": _sha("e"),
        "gpu_spend_snapshot_body_sha256": _sha("f"),
        "gpu_spend_ledger_tip_record_sha256": _sha("0"),
        "remaining_gpu_seconds": 86_400,
        "remaining_gpu_cost_usd": 9.17,
        "qualification_allowance_seconds": 3_600,
        "qualification_allowance_cost_usd": 9.17,
        "open_allocation_count": 0,
        "qualification_submission_ready_key": (
            f"{PREFIX}/qualification/readiness/{_sha('1')}.json"
        ),
        "qualification_submission_ready_sha256": _sha("1"),
        "qualification_submission_ready_body_sha256": _sha("2"),
        "sky_job_name": JOB_ID,
        "must_start_by": descriptor["must_start_by"],
        "intent_at": "2026-07-26T10:05:00Z",
    }
    return {
        **body,
        "intent_body_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _v2_event(
    descriptor: dict[str, object], intent: dict[str, object]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_sky_qualification_watchdog_invocation_v1",
        "run_id": RUN_ID,
        "managed_mode": "qualification",
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "intent_key": (
            f"{PREFIX}/submissions/qualification/intents/"
            f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
        ),
        "intent_body_sha256": intent["intent_body_sha256"],
    }


def _v2_binding(
    descriptor: dict[str, object],
    intent: dict[str, object],
    *,
    target_job_id: int = 71,
    bound_at: str = "2026-07-26T10:06:00Z",
) -> dict[str, object]:
    return build_must_start_job_binding(
        run_id=RUN_ID,
        managed_mode="qualification",
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256=str(intent["intent_body_sha256"]),
        sky_job_name=JOB_ID,
        must_start_by=str(intent["must_start_by"]),
        descriptor_key=str(descriptor["campaign_descriptor_key"]),
        descriptor_file_sha256=str(intent["descriptor_file_sha256"]),
        submission_key=(
            f"{PREFIX}/submissions/qualification/intents/"
            f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
        ),
        submission_submitted_at=str(intent["intent_at"]),
        target_job_id=target_job_id,
        workspace="default",
        controller_instance_id="i-0123456789abcdef0",
        controller_instance_type="c6a.xlarge",
        controller_profile_arn=(
            "arn:aws:iam::246813579024:instance-profile/keep-glm52-skypilot-controller"
        ),
        controller_cluster_name="sky-jobs-controller-12345678",
        observation_body_sha256=_sha("3"),
        bound_at=bound_at,
    )


def _invoke_dynamic_handler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    event: dict[str, object],
    values: dict[str, dict[str, object]],
    ec2: FakeEC2,
    events: list[str],
    now: datetime = NOW,
):
    module = _load_handler()

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(module, "datetime", FixedDatetime)
    s3 = FakeS3(values, events)
    clients = {"s3": s3, "ec2": ec2}
    monkeypatch.setattr(
        module.boto3,
        "client",
        lambda name, **_kwargs: clients.get(name, object()),
    )
    for key, value in {
        "AWS_REGION": "us-west-2",
        "CAMPAIGN_BUCKET": BUCKET,
        "CAMPAIGN_DESCRIPTOR_KEY": str(
            event.get(
                "descriptor_key",
                f"{PREFIX}/submissions/qualification/campaign-descriptor-v2.json",
            )
        ),
        "ALERT_TOPIC_ARN": "arn:aws:sns:us-west-2:246813579024:keep-glm52",
        "WATCHDOG_DLQ_URL": "https://sqs.us-west-2.amazonaws.com/246813579024/dlq",
        "WATCHDOG_RULE_NAME": "keep-glm52-watchdog",
    }.items():
        monkeypatch.setenv(key, value)
    return module.lambda_handler(event, None), s3


def _intent_with_time(
    descriptor: dict[str, object],
    *,
    intent_at: str,
) -> dict[str, object]:
    intent = _v2_intent(descriptor)
    body = dict(intent)
    body.pop("intent_body_sha256")
    body["intent_at"] = intent_at
    return {
        **body,
        "intent_body_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _dynamic_values(
    descriptor: dict[str, object],
    intent: dict[str, object],
    binding: dict[str, object] | None,
    *,
    allocation_job_id: str = "71",
) -> dict[str, dict[str, object]]:
    allocation = _allocation(descriptor, job_id=allocation_job_id)
    values = _values(descriptor, allocation, _source(descriptor, allocation))
    values[str(descriptor["campaign_descriptor_key"])] = descriptor
    values[
        f"{PREFIX}/submissions/qualification/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    ] = intent
    if binding is not None:
        values[
            f"{PREFIX}/monitor/must-start/qualification/"
            f"{intent['intent_body_sha256']}/JOB_BINDING.json"
        ] = binding
    return values


def test_untyped_event_cannot_use_legacy_marker_when_v2_intent_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _v2_intent(descriptor)
    values = _dynamic_values(
        descriptor,
        intent,
        _v2_binding(descriptor, intent),
        allocation_job_id=JOB_ID,
    )
    values[f"{PREFIX}/monitor/H100_QUALIFICATION_SUBMITTED.json"] = _submission(
        descriptor
    )
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="v2 intent"):
        _invoke_dynamic_handler(
            monkeypatch,
            event={},
            values=values,
            ec2=ec2,
            events=events,
        )
    assert events == []
    assert ec2.terminate_calls == []


def test_dynamic_v2_event_uses_numeric_binding_id_for_durable_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _v2_intent(descriptor)
    binding = _v2_binding(descriptor, intent)
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    result, _s3 = _invoke_dynamic_handler(
        monkeypatch,
        event=_v2_event(descriptor, intent),
        values=_dynamic_values(descriptor, intent, binding),
        ec2=ec2,
        events=events,
    )

    assert result["status"] == "termination-requested"
    assert result["run_id"] == RUN_ID
    assert events == [
        "s3:QUALIFICATION_TERMINATION_REQUESTED.json",
        f"ec2:terminate:{SOURCE_ID}",
    ]


def test_dynamic_v2_event_rejects_job_name_when_allocation_needs_numeric_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _v2_intent(descriptor)
    binding = _v2_binding(descriptor, intent)
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="job_id"):
        _invoke_dynamic_handler(
            monkeypatch,
            event=_v2_event(descriptor, intent),
            values=_dynamic_values(
                descriptor,
                intent,
                binding,
                allocation_job_id=JOB_ID,
            ),
            ec2=ec2,
            events=events,
        )
    assert events == []
    assert ec2.terminate_calls == []


def test_dynamic_v2_event_without_binding_only_awaits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _v2_intent(descriptor)
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    result, _s3 = _invoke_dynamic_handler(
        monkeypatch,
        event=_v2_event(descriptor, intent),
        values=_dynamic_values(descriptor, intent, None),
        ec2=ec2,
        events=events,
    )

    assert result == {"status": "awaiting-job-binding", "run_id": RUN_ID}
    assert events == []
    assert ec2.terminate_calls == []


def test_dynamic_v2_event_rejects_future_intent_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _intent_with_time(descriptor, intent_at="2026-07-26T10:11:00Z")
    binding = _v2_binding(
        descriptor,
        intent,
        bound_at="2026-07-26T10:12:00Z",
    )
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="intent_at.*future"):
        _invoke_dynamic_handler(
            monkeypatch,
            event=_v2_event(descriptor, intent),
            values=_dynamic_values(descriptor, intent, binding),
            ec2=ec2,
            events=events,
        )
    assert events == []
    assert ec2.terminate_calls == []


def test_dynamic_v2_event_rejects_future_binding_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _v2_intent(descriptor)
    binding = _v2_binding(
        descriptor,
        intent,
        bound_at="2026-07-26T10:11:00Z",
    )
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="binding.*future"):
        _invoke_dynamic_handler(
            monkeypatch,
            event=_v2_event(descriptor, intent),
            values=_dynamic_values(descriptor, intent, binding),
            ec2=ec2,
            events=events,
        )
    assert events == []
    assert ec2.terminate_calls == []


def test_dynamic_v2_event_rejects_binding_after_deadline_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _v2_intent(descriptor)
    binding = _v2_binding(
        descriptor,
        intent,
        bound_at="2026-07-26T22:11:00Z",
    )
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID),
        active=[_instance(SOURCE_ID)],
        events=events,
    )

    with pytest.raises(ValueError, match="binding.*must_start_by"):
        _invoke_dynamic_handler(
            monkeypatch,
            event=_v2_event(descriptor, intent),
            values=_dynamic_values(descriptor, intent, binding),
            ec2=ec2,
            events=events,
            now=datetime(2026, 7, 26, 22, 12, tzinfo=UTC),
        )
    assert events == []
    assert ec2.terminate_calls == []


@pytest.mark.parametrize(
    "bad_event",
    [
        {"schema_version": 1},
        {
            "schema_version": 1,
            "record_type": "glm52_sky_qualification_watchdog_invocation_v1",
            "run_id": RUN_ID,
            "managed_mode": "production",
            "descriptor_key": "campaigns/x/submissions/descriptor.json",
            "intent_key": "campaigns/x/intents/a/SKYPILOT_SUBMISSION_INTENT.json",
            "intent_body_sha256": _sha("a"),
        },
        {
            "schema_version": 1,
            "record_type": "glm52_sky_qualification_watchdog_invocation_v1",
            "run_id": RUN_ID,
            "managed_mode": "cache-seed",
            "descriptor_key": "campaigns/x/submissions/descriptor.json",
            "intent_key": "campaigns/x/intents/a/SKYPILOT_SUBMISSION_INTENT.json",
            "intent_body_sha256": _sha("a"),
        },
        {
            "schema_version": True,
            "record_type": "glm52_sky_qualification_watchdog_invocation_v1",
            "run_id": RUN_ID,
            "managed_mode": "qualification",
            "descriptor_key": "campaigns/x/submissions/descriptor.json",
            "intent_key": "campaigns/x/intents/a/SKYPILOT_SUBMISSION_INTENT.json",
            "intent_body_sha256": _sha("a"),
        },
    ],
)
def test_dynamic_v2_event_rejects_malformed_or_nonqualification_event_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    bad_event: dict[str, object],
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _v2_intent(descriptor)
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID), active=[_instance(SOURCE_ID)], events=events
    )

    with pytest.raises(ValueError):
        _invoke_dynamic_handler(
            monkeypatch,
            event=bad_event,
            values=_dynamic_values(descriptor, intent, _v2_binding(descriptor, intent)),
            ec2=ec2,
            events=events,
        )
    assert events == []
    assert ec2.terminate_calls == []


def test_dynamic_v2_event_rejects_corrupt_binding_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _v2_intent(descriptor)
    binding = _v2_binding(descriptor, intent)
    binding["target_job_id"] = 72
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID), active=[_instance(SOURCE_ID)], events=events
    )

    with pytest.raises(ValueError, match="binding"):
        _invoke_dynamic_handler(
            monkeypatch,
            event=_v2_event(descriptor, intent),
            values=_dynamic_values(descriptor, intent, binding),
            ec2=ec2,
            events=events,
        )
    assert events == []
    assert ec2.terminate_calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda intent: intent.__setitem__("schema_version", 1),
        lambda intent: intent.__setitem__("unexpected", "field"),
    ],
)
def test_dynamic_v2_event_rejects_old_or_mixed_intent_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    descriptor = _dynamic_descriptor()
    intent = _v2_intent(descriptor)
    mutate(intent)
    events: list[str] = []
    ec2 = FakeEC2(
        source=_instance(SOURCE_ID), active=[_instance(SOURCE_ID)], events=events
    )

    with pytest.raises(ValueError, match="v2 intent"):
        _invoke_dynamic_handler(
            monkeypatch,
            event=_v2_event(descriptor, intent),
            values=_dynamic_values(
                descriptor, intent, _v2_binding(descriptor, _v2_intent(descriptor))
            ),
            ec2=ec2,
            events=events,
        )
    assert events == []
    assert ec2.terminate_calls == []


def test_sky_watchdog_checks_full_control_plane_and_stops_after_drain() -> None:
    source = HANDLER.read_text()
    for required in (
        "describe_instances",
        "describe_instance_status",
        "describe_instance_information",
        "send_command",
        "keep-glm52-campaign.service",
        "validate_campaign_heartbeat",
        "GPU_SPEND_STATUS.json",
        "GPU_RUNTIME_ALLOCATION.json",
        "CAMPAIGN_FAILED.json",
        "TRAINING_DEFERRED.json",
        "TERMINAL_VERIFIED.json",
        "validate_sky_terminal_state",
        "ApproximateNumberOfMessages",
        "describe_alarms",
        "campaign_notification_decision",
        "TRAINING DEFERRED",
        "H100_QUALIFICATION_SUBMITTED.json",
        "QUALIFICATION_CACHE_SEED_SUBMITTED.json",
        "validate_skypilot_job_status",
        "coordinate_h100_qualification_termination",
        "terminate_instances",
        'IfNoneMatch="*"',
        "disable_rule",
    ):
        assert required in source


def test_scheduled_empty_event_preserves_legacy_not_submitted_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor()
    descriptor_key = str(descriptor["campaign_descriptor_key"])
    s3 = FakeS3({descriptor_key: descriptor}, [])
    module = _load_handler()
    client_calls: list[tuple[str, str | None]] = []

    def client(name: str, **kwargs: object) -> object:
        client_calls.append((name, kwargs.get("region_name")))
        return s3 if name == "s3" else object()

    def coordinate(*, services, request):
        raise AssertionError("worker-start coordinator must not run")

    monkeypatch.setattr(module.boto3, "client", client)
    monkeypatch.setattr(
        module,
        "coordinate_worker_start_acceptance_v2",
        coordinate,
        raising=False,
    )
    for name, value in {
        "AWS_REGION": "us-west-2",
        "CAMPAIGN_BUCKET": BUCKET,
        "CAMPAIGN_DESCRIPTOR_KEY": descriptor_key,
        "ALERT_TOPIC_ARN": "arn:aws:sns:us-west-2:246813579024:keep-glm52",
        "WATCHDOG_DLQ_URL": (
            "https://sqs.us-west-2.amazonaws.com/246813579024/watchdog-dlq"
        ),
        "WATCHDOG_RULE_NAME": "keep-glm52-watchdog",
    }.items():
        monkeypatch.setenv(name, value)

    assert module.lambda_handler({}, None) == {
        "status": "not-submitted",
        "run_id": RUN_ID,
    }
    assert client_calls == [
        (name, "us-west-2")
        for name in (
            "s3",
            "ec2",
            "ssm",
            "cloudwatch",
            "sqs",
            "sns",
            "events",
        )
    ]


def test_sky_watchdog_lambda_package_is_import_complete(tmp_path) -> None:
    output = tmp_path / "sky-watchdog.zip"
    with zipfile.ZipFile(output, "w") as stale_archive:
        stale_archive.writestr("stale_archive_member.py", "must be removed\n")
    result = subprocess.run(
        [str(PACKAGE), str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(output) as archive:
        names = {
            member.filename for member in archive.infolist() if not member.is_dir()
        }
        assert names == {
            "glm52_enforcement/__init__.py",
            "glm52_enforcement/glm52_campaign_watchdog.py",
            "glm52_enforcement/glm52_h100_qualification.py",
            "glm52_enforcement/glm52_sky_campaign.py",
            "glm52_enforcement/glm52_sky_must_start.py",
            "glm52_enforcement/glm52_teich_campaign.py",
            "handler.py",
            "sky_worker_start_v2_coordinator.py",
            "mlx_vq/__init__.py",
            "mlx_vq/quality/__init__.py",
            "mlx_vq/quality/glm52_campaign_watchdog.py",
            "mlx_vq/quality/glm52_h100_qualification.py",
            "mlx_vq/quality/glm52_sky_campaign.py",
            "mlx_vq/quality/glm52_sky_must_start.py",
            "mlx_vq/quality/glm52_sky_must_start_dynamic.py",
            "mlx_vq/quality/glm52_sky_terminal_state.py",
            "mlx_vq/quality/glm52_sky_worker_must_start_v2.py",
            "mlx_vq/quality/glm52_teich_campaign.py",
        }
        assert archive.read("mlx_vq/__init__.py") == b""
        assert archive.read("mlx_vq/quality/__init__.py") == b""
    import_result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import json, sys, types; "
                "boto3 = types.ModuleType('boto3'); "
                "boto3.client = lambda *args, **kwargs: None; "
                "exceptions = types.ModuleType('botocore.exceptions'); "
                "exceptions.ClientError = type('ClientError', (Exception,), {}); "
                "botocore = types.ModuleType('botocore'); "
                "botocore.exceptions = exceptions; "
                "sys.modules.update({'boto3': boto3, 'botocore': botocore, "
                "'botocore.exceptions': exceptions}); "
                "import handler; "
                "import sky_worker_start_v2_coordinator; "
                "print(json.dumps({"
                "'forbidden': sorted(name for name in ('sky', 'mlx', 'numpy') "
                "if name in sys.modules), "
                "'loaded': sorted(name for name in "
                "('handler', 'sky_worker_start_v2_coordinator') "
                "if name in sys.modules)"
                "}, sort_keys=True))"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(output)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert import_result.returncode == 0, import_result.stderr
    assert json.loads(import_result.stdout) == {
        "forbidden": [],
        "loaded": ["handler", "sky_worker_start_v2_coordinator"],
    }


def test_sky_watchdog_lambda_package_honors_relative_output_path(
    tmp_path,
) -> None:
    output = Path("dist") / "sky-watchdog.zip"
    result = subprocess.run(
        [str(PACKAGE), str(output)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / output).is_file()
    assert result.stdout.strip() == str(output)


def test_sky_watchdog_lambda_package_rejects_suffixless_output_without_mutation(
    tmp_path,
) -> None:
    requested = tmp_path / "sky-watchdog"
    auto_appended = tmp_path / "sky-watchdog.zip"
    with zipfile.ZipFile(auto_appended, "w") as stale_archive:
        stale_archive.writestr("stale.py", "must remain untouched\n")
    stale = auto_appended.read_bytes()

    result = subprocess.run(
        [str(PACKAGE), str(requested)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert not requested.exists()
    assert auto_appended.read_bytes() == stale


def test_sky_watchdog_lambda_package_is_byte_deterministic(tmp_path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    for index, (umask, tz, zipopt, output) in enumerate(
        (
            ("022", "UTC", "", first),
            ("077", "Asia/Tokyo", "-0", second),
        )
    ):
        result = subprocess.run(
            [
                "/bin/sh",
                "-c",
                'umask "$1"; shift; exec "$@"',
                "package-test",
                umask,
                str(PACKAGE),
                str(output),
            ],
            cwd=ROOT,
            env={**os.environ, "TZ": tz, "ZIPOPT": zipopt},
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        if index == 0:
            time.sleep(2)
    assert first.read_bytes() == second.read_bytes()
