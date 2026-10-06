from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import types
import zipfile
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from mlx_vq.quality.glm52_campaign_watchdog import (
    build_skypilot_submission_marker,
)
from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import (
    build_must_start_cancel_requested,
    build_timely_start_latch,
)

ROOT = Path(__file__).resolve().parents[1]
HANDLER_PATH = ROOT / "aws/glm52-gpu/lambda/sky_must_start_cancel_handler.py"
PACKAGE = ROOT / "aws/glm52-gpu/scripts/package_sky_must_start_cancel_lambda.sh"
UTC = timezone.utc
NOW = datetime(2026, 7, 26, 14, 28, 42, tzinfo=UTC)
DEADLINE = datetime(2026, 7, 26, 14, 27, 42, tzinfo=UTC)
SUBMITTED_AT = DEADLINE - timedelta(hours=1)
RUN_ID = "glm52-sky-20260724"
MODE = "cache-seed"
JOB_NAME = f"{RUN_ID}-cache-seed"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
DESCRIPTOR_KEY = (
    f"campaigns/{RUN_ID}/submissions/seed/campaign-descriptor-v2.json"
)


def _load_handler():
    spec = importlib.util.spec_from_file_location("_must_start_handler", HANDLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(character: str) -> str:
    return character * 64


def _descriptor() -> dict[str, object]:
    approval = build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 25, tzinfo=UTC),
        slack_permalink=None,
    )
    return build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=DEADLINE,
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity="arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=f"campaigns/{RUN_ID}/repository/repo.tar.gz",
        repo_tar_sha256=_sha("1"),
        campaign_descriptor_key=DESCRIPTOR_KEY,
        approval_key=f"campaigns/{RUN_ID}/authorities/approval.json",
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
            "training_config_key": f"campaigns/{RUN_ID}/authorities/training.json",
            "training_config_sha256": _sha("7"),
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/"
                f"artifact-inventory-{_sha('8')}.json"
            ),
            "artifact_inventory_sha256": _sha("8"),
            "qualification_cache_prefix": (
                f"qualification-cache/seeds/{RUN_ID}/{_sha('9')}/"
            ),
            "qualification_cache_manifest_sha256": _sha("9"),
        },
    )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


class FakeS3:
    def __init__(self, values: dict[str, dict[str, object]], events: list[str]):
        self.values = dict(values)
        self.events = events
        self.race_winner: dict[str, object] | None = None
        self.after_put = None
        self.last_modified: dict[str, datetime] = {
            key: NOW - timedelta(minutes=5) for key in values
        }

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
        assert Bucket == BUCKET
        if Key not in self.values:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        return {"Body": io.BytesIO(_canonical(self.values[Key]))}

    def put_object(self, **kwargs):
        assert kwargs["Bucket"] == BUCKET
        assert kwargs["IfNoneMatch"] == "*"
        key = kwargs["Key"]
        value = json.loads(kwargs["Body"])
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
        self.values[key] = value
        self.last_modified[key] = NOW
        if self.after_put is not None:
            self.after_put(key)
        return {"ETag": '"etag"'}

    def head_object(self, *, Bucket: str, Key: str):  # noqa: N803
        assert Bucket == BUCKET
        if Key not in self.values:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "HeadObject",
            )
        return {
            "ETag": '"etag"',
            "LastModified": self.last_modified[Key],
            "VersionId": "version-1",
        }


class FakeEC2:
    def __init__(
        self,
        count: int = 1,
        *,
        instance_id: str = "i-0511af4e31aa5406a",
        instance_type: str = "c6a.xlarge",
        profile_arn: str = (
            "arn:aws:iam::246813579024:instance-profile/"
            "keep-glm52-skypilot-controller"
        ),
        cluster_name: str = "sky-jobs-controller-9d9f31a9-9d9f31a9",
    ):
        self.count = count
        self.instance_id = instance_id
        self.instance_type = instance_type
        self.profile_arn = profile_arn
        self.cluster_name = cluster_name

    def describe_instances(self, **kwargs):
        filters = kwargs.get("Filters")
        if filters is not None:
            assert {item["Name"] for item in filters} >= {
                "tag:campaign-run-id",
                "tag:ray-cluster-name",
                "instance-state-name",
            }
        if "InstanceIds" in kwargs:
            assert kwargs["InstanceIds"] == [self.instance_id]
        instances = [
            {
                "InstanceId": (
                    self.instance_id
                    if self.count == 1
                    else f"i-controller-{index}"
                ),
                "InstanceType": self.instance_type,
                "State": {"Name": "running"},
                "IamInstanceProfile": {"Arn": self.profile_arn},
                "Placement": {"AvailabilityZone": "us-west-2a"},
                "Tags": [
                    {"Key": "campaign-run-id", "Value": RUN_ID},
                    {"Key": "ray-cluster-name", "Value": self.cluster_name},
                ],
            }
            for index in range(self.count)
        ]
        return {"Reservations": [{"Instances": instances}]}


class FakeSSM:
    def __init__(self, events: list[str], results: list[object]):
        self.events = events
        self.results = list(results)
        self.commands: list[dict[str, object]] = []
        self.after_result = None
        self.ping_status = "Online"

    def describe_instance_information(self, **kwargs):
        assert kwargs["Filters"][0]["Key"] == "InstanceIds"
        return {
            "InstanceInformationList": [
                {
                    "InstanceId": kwargs["Filters"][0]["Values"][0],
                    "PingStatus": self.ping_status,
                }
            ]
        }

    def send_command(self, **kwargs):
        self.events.append("ssm:send")
        self.commands.append(kwargs)
        return {"Command": {"CommandId": f"command-{len(self.commands)}"}}

    def get_command_invocation(self, **_kwargs):
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, dict)
        if self.after_result is not None:
            self.after_result()
        return {
            "Status": "Success",
            "StandardOutputContent": (
                "GLM52_MUST_START_RESULT="
                + json.dumps(result, sort_keys=True, separators=(",", ":"))
                + "\n"
            ),
            "StandardErrorContent": "",
        }


class FakeSNS:
    def __init__(self, failures: int = 0):
        self.messages: list[dict[str, object]] = []
        self.failures = failures

    def publish(self, **kwargs):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transient SNS failure")
        self.messages.append(kwargs)
        return {"MessageId": str(len(self.messages))}


class FakeEvents:
    def __init__(self) -> None:
        self.disabled: list[dict[str, str]] = []

    def disable_rule(self, **kwargs):
        self.disabled.append(kwargs)
        return {}


def _result(outcome: str, status: str) -> dict[str, object]:
    nonterminal = [3] if outcome == "nonterminal" else []
    target = None if outcome == "not_found" else 3
    return {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_cancel_result_v1",
        "sky_job_name": JOB_NAME,
        "outcome": outcome,
        "before_nonterminal_job_ids": (
            [3] if outcome in {"nonterminal", "terminal"} else []
        ),
        "after_nonterminal_job_ids": nonterminal,
        "all_job_ids": [3] if outcome != "not_found" else [],
        "target_job_id": target,
        "target_status": status if target is not None else None,
        "observed_at": NOW.isoformat().replace("+00:00", "Z"),
    }


def _setup(*, before_deadline: bool = False, controller_count: int = 1):
    module = _load_handler()
    descriptor = _descriptor()
    raw_descriptor = _canonical(descriptor)
    descriptor_file_sha = __import__("hashlib").sha256(raw_descriptor).hexdigest()
    submission_key = (
        f"campaigns/{RUN_ID}/monitor/submission-locks/"
        f"{descriptor_file_sha}-{MODE}.json"
    )
    submission = build_skypilot_submission_marker(
        run_id=RUN_ID,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submitted_at=SUBMITTED_AT,
        must_start_by=DEADLINE,
        sky_job_name=JOB_NAME,
    )
    events: list[str] = []
    s3 = FakeS3(
        {
            DESCRIPTOR_KEY: descriptor,
            submission_key: submission,
        },
        events,
    )
    config = module.MustStartCoordinatorConfig(
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_key=DESCRIPTOR_KEY,
        submission_key=submission_key,
        submission_body_sha256=str(submission["submission_body_sha256"]),
        run_id=RUN_ID,
        managed_mode=MODE,
        sky_job_name=JOB_NAME,
        must_start_by=DEADLINE.isoformat().replace("+00:00", "Z"),
        alert_topic_arn=(
            "arn:aws:sns:us-west-2:246813579024:keep-glm52-campaign-alerts"
        ),
    )
    return (
        module,
        config,
        s3,
        FakeEC2(controller_count),
        events,
        deadline_now(before_deadline),
    )


def deadline_now(before: bool) -> datetime:
    return DEADLINE - timedelta(seconds=1) if before else NOW


def _timely_latch(config, s3) -> dict[str, object]:
    descriptor = s3.values[DESCRIPTOR_KEY]
    submission = s3.values[config.submission_key]
    return build_timely_start_latch(
        run_id=RUN_ID,
        managed_mode=MODE,
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256=str(submission["submission_body_sha256"]),
        sky_job_name=JOB_NAME,
        must_start_by=DEADLINE,
        started_at=DEADLINE - timedelta(seconds=1),
        published_at=DEADLINE - timedelta(seconds=1),
    )


def test_before_deadline_has_no_request_ssm_or_completion() -> None:
    module, config, s3, ec2, events, now = _setup(before_deadline=True)
    ssm = FakeSSM(events, [])
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "waiting"
    assert events == []


@pytest.mark.parametrize("state", ["PENDING", "STARTING", "RUNNING", "RECOVERING"])
def test_deadline_requests_before_ssm_and_targets_exact_job(state: str) -> None:
    module, config, s3, ec2, events, now = _setup()
    ssm = FakeSSM(events, [_result("nonterminal", state)])
    sns = FakeSNS()
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=sns,
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "requested"
    assert events[:3] == [
        "s3:CANCEL_REQUESTED.json",
        "s3:REQUEST_ALERTED.json",
        "ssm:send",
    ]
    command = str(ssm.commands[0]["Parameters"]["commands"][0])
    assert "/home/ubuntu/skypilot-runtime/bin/python" in command
    assert JOB_NAME in command
    assert "terminate-instances" not in command
    assert [message["Subject"] for message in sns.messages] == [
        "KEEP GLM52 must-start cancellation requested"
    ]


def test_not_found_remains_requested_and_retries_without_completion() -> None:
    module, config, s3, ec2, events, now = _setup()
    ssm = FakeSSM(events, [_result("not_found", "")])
    first = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    ssm.results.append(_result("not_found", ""))
    second = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now + timedelta(minutes=1),
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert first["status"] == second["status"] == "requested"
    assert sum(key.endswith("CANCEL_REQUESTED.json") for key in s3.values) == 1
    assert not any(key.endswith("CANCEL_COMPLETED.json") for key in s3.values)
    assert len(ssm.commands) == 2


def test_terminal_confirmation_publishes_one_completion_and_duplicates_converge(
) -> None:
    module, config, s3, ec2, events, now = _setup()
    ssm = FakeSSM(events, [_result("terminal", "CANCELLED")])
    sns = FakeSNS()
    first = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=sns,
        sleep=lambda _seconds: None,
    )
    second = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now + timedelta(minutes=1),
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(events, []),
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert first["status"] == second["status"] == "completed"
    assert sum(key.endswith("CANCEL_COMPLETED.json") for key in s3.values) == 1
    assert [message["Subject"] for message in sns.messages] == [
        "KEEP GLM52 must-start cancellation requested",
        "KEEP GLM52 must-start cancellation completed",
    ]


def test_conditional_request_race_accepts_authenticated_actual_time_winner() -> None:
    module, config, s3, ec2, events, now = _setup()
    descriptor = s3.values[DESCRIPTOR_KEY]
    submission = s3.values[config.submission_key]
    expected = build_must_start_cancel_requested(
        run_id=RUN_ID,
        managed_mode=MODE,
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256=str(submission["submission_body_sha256"]),
        sky_job_name=JOB_NAME,
        must_start_by=DEADLINE,
        requested_at=DEADLINE + timedelta(seconds=1),
    )
    s3.race_winner = expected
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(events, [_result("nonterminal", "PENDING")]),
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "requested"
    assert "ssm:send" in events
    winner = s3.values[config.request_key]
    assert winner == expected
    assert winner["requested_at"] == (
        DEADLINE + timedelta(seconds=1)
    ).isoformat().replace("+00:00", "Z")
    assert winner["requested_at"] != now.isoformat().replace("+00:00", "Z")
    assert result["request_body_sha256"] == winner["request_body_sha256"]


def test_foreign_conditional_request_winner_fails_closed_without_ssm() -> None:
    module, config, s3, ec2, events, now = _setup()
    descriptor = s3.values[DESCRIPTOR_KEY]
    s3.race_winner = build_must_start_cancel_requested(
        run_id=RUN_ID,
        managed_mode=MODE,
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256="a" * 64,
        sky_job_name=JOB_NAME,
        must_start_by=DEADLINE,
        requested_at=DEADLINE + timedelta(seconds=1),
    )
    with pytest.raises(ValueError, match="winner"):
        module.coordinate_must_start(
            config=config,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=now,
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(events, []),
            sns=FakeSNS(),
            sleep=lambda _seconds: None,
        )
    assert "ssm:send" not in events


def test_valid_latch_suppresses_but_late_or_foreign_latch_does_not() -> None:
    module, config, s3, ec2, events, now = _setup()
    descriptor = s3.values[DESCRIPTOR_KEY]
    submission = s3.values[config.submission_key]
    authority = {
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": BUCKET,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "submission_body_sha256": submission["submission_body_sha256"],
        "sky_job_name": JOB_NAME,
        "must_start_by": DEADLINE,
    }
    s3.values[config.latch_key] = build_timely_start_latch(
        **authority,
        started_at=DEADLINE - timedelta(seconds=1),
        published_at=DEADLINE - timedelta(seconds=1),
    )
    suppressed = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(events, []),
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert suppressed["status"] == "timely-started"
    assert events == []

    s3.values[config.latch_key] = build_timely_start_latch(
        **authority,
        started_at=DEADLINE + timedelta(seconds=1),
        published_at=DEADLINE + timedelta(seconds=1),
    )
    late = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(events, [_result("nonterminal", "RUNNING")]),
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert late["status"] == "requested"


def test_foreign_valid_latch_does_not_suppress_cancellation() -> None:
    module, config, s3, ec2, events, now = _setup()
    descriptor = s3.values[DESCRIPTOR_KEY]
    s3.values[config.latch_key] = build_timely_start_latch(
        run_id=RUN_ID,
        managed_mode=MODE,
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256="a" * 64,
        sky_job_name=JOB_NAME,
        must_start_by=DEADLINE,
        started_at=DEADLINE - timedelta(seconds=1),
        published_at=DEADLINE - timedelta(seconds=1),
    )
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(events, [_result("nonterminal", "RUNNING")]),
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "requested"
    assert events[:3] == [
        "s3:CANCEL_REQUESTED.json",
        "s3:REQUEST_ALERTED.json",
        "ssm:send",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_account_id", "135792468013"),
        ("runtime_region", "us-east-1"),
    ],
)
def test_wrong_runtime_authority_performs_no_mutation(field: str, value: str) -> None:
    module, config, s3, ec2, events, now = _setup()
    kwargs = {
        "runtime_account_id": "246813579024",
        "runtime_region": "us-west-2",
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        module.coordinate_must_start(
            config=config,
            now=now,
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(events, []),
            sns=FakeSNS(),
            sleep=lambda _seconds: None,
            **kwargs,
        )
    assert events == []


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ({"bucket": "foreign-authority-bucket"}, None),
        (
            {
                "descriptor_key": (
                    f"campaigns/{RUN_ID}/submissions/foreign/"
                    "campaign-descriptor-v2.json"
                )
            },
            "missing",
        ),
        (
            {
                "submission_key": (
                    f"campaigns/{RUN_ID}/monitor/submission-locks/"
                    f"{'0' * 64}-{MODE}.json"
                )
            },
            "immutable submission key",
        ),
        (
            {
                "submission_body_sha256": "0" * 64,
            },
            "submission body",
        ),
        (
            {
                "managed_mode": "production",
                "sky_job_name": RUN_ID,
            },
            "immutable submission key",
        ),
    ],
)
def test_wrong_campaign_authority_performs_no_mutation(
    replacement: dict[str, str],
    match: str | None,
) -> None:
    module, config, s3, ec2, events, now = _setup()
    foreign = replace(config, **replacement)
    with pytest.raises((AssertionError, ValueError), match=match):
        module.coordinate_must_start(
            config=foreign,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=now,
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(events, []),
            sns=FakeSNS(),
            sleep=lambda _seconds: None,
        )
    assert events == []


def test_wrong_configured_job_name_is_rejected_before_mutation() -> None:
    module, config, _s3, _ec2, events, _now = _setup()
    with pytest.raises(ValueError, match="job name"):
        replace(config, sky_job_name="wrong-job")
    assert events == []


def test_config_rejects_zero_job_id() -> None:
    _module, config, _s3, _ec2, events, _now = _setup()
    with pytest.raises(ValueError, match="target job ID"):
        replace(config, expected_target_job_id=0)
    assert events == []


def test_config_rejects_fractional_job_id_before_any_mutation() -> None:
    _module, config, _s3, _ec2, events, _now = _setup()
    with pytest.raises(ValueError, match="target job ID"):
        replace(config, expected_target_job_id=1.5)
    assert events == []


def test_multiple_matching_controllers_fail_closed_after_durable_request() -> None:
    module, config, s3, ec2, events, now = _setup(controller_count=2)
    with pytest.raises(ValueError, match="exactly one"):
        module.coordinate_must_start(
            config=config,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=now,
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(events, []),
            sns=FakeSNS(),
            sleep=lambda _seconds: None,
        )
    assert events == [
        "s3:CANCEL_REQUESTED.json",
        "s3:REQUEST_ALERTED.json",
    ]


def test_marker_prefix_is_scoped_to_immutable_submission_identity() -> None:
    _module, config, _s3, _ec2, _events, _now = _setup()
    assert config.marker_prefix.endswith(
        f"/{MODE}/{config.submission_body_sha256}"
    )


def test_absent_evidence_marker_is_none_but_foreign_prefix_remains_denied() -> None:
    module, config, _s3, _ec2, _events, _now = _setup()

    class PrefixScopedMissingS3:
        def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
            assert Bucket == BUCKET
            if Key.startswith(f"{config.marker_prefix}/"):
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}},
                "GetObject",
            )

    s3 = PrefixScopedMissingS3()
    assert (
        module._read_json(
            s3,
            bucket=BUCKET,
            key=config.accepted_key,
        )
        is None
    )
    with pytest.raises(ClientError) as denied:
        module._read_json(
            s3,
            bucket=BUCKET,
            key=f"campaigns/{RUN_ID}/monitor/foreign/marker.json",
        )
    assert denied.value.response["Error"]["Code"] == "AccessDenied"


def test_read_json_closes_stream_after_success_and_read_failure() -> None:
    module, _config, _s3, _ec2, _events, _now = _setup()

    class ClosingBody:
        def __init__(
            self,
            *,
            payload: bytes = b'{"status":"ok"}',
            failure: Exception | None = None,
        ) -> None:
            self.payload = payload
            self.failure = failure
            self.close_calls = 0

        def read(self) -> bytes:
            if self.failure is not None:
                raise self.failure
            return self.payload

        def close(self) -> None:
            self.close_calls += 1

    class BodyS3:
        def __init__(self, body: ClosingBody) -> None:
            self.body = body

        def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
            assert Bucket == BUCKET
            assert Key == "marker.json"
            return {"Body": self.body}

    successful_body = ClosingBody()
    assert module._read_json(
        BodyS3(successful_body),
        bucket=BUCKET,
        key="marker.json",
    ) == ({"status": "ok"}, b'{"status":"ok"}')
    assert successful_body.close_calls == 1

    failed_body = ClosingBody(failure=RuntimeError("stream read failed"))
    with pytest.raises(RuntimeError, match="stream read failed"):
        module._read_json(
            BodyS3(failed_body),
            bucket=BUCKET,
            key="marker.json",
        )
    assert failed_body.close_calls == 1

    invalid_json_body = ClosingBody(payload=b"{")
    with pytest.raises(ValueError, match="is not JSON"):
        module._read_json(
            BodyS3(invalid_json_body),
            bucket=BUCKET,
            key="marker.json",
        )
    assert invalid_json_body.close_calls == 1


def test_timely_latch_after_request_suppresses_ssm() -> None:
    module, config, s3, ec2, events, now = _setup()
    latch = _timely_latch(config, s3)

    def after_put(key: str) -> None:
        if key == config.request_key:
            s3.values[config.latch_key] = latch

    s3.after_put = after_put
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(events, []),
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "timely-started"
    assert "ssm:send" not in events
    assert not any(key.endswith("CANCEL_COMPLETED.json") for key in s3.values)


def test_timely_latch_after_ssm_suppresses_completion() -> None:
    module, config, s3, ec2, events, now = _setup()
    ssm = FakeSSM(events, [_result("terminal", "CANCELLED")])
    ssm.after_result = lambda: s3.values.__setitem__(
        config.latch_key, _timely_latch(config, s3)
    )
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "timely-started"
    assert not any(key.endswith("CANCEL_COMPLETED.json") for key in s3.values)


def test_request_alert_delivery_retries_without_duplicate_request() -> None:
    module, config, s3, ec2, events, now = _setup()
    with pytest.raises(RuntimeError, match="SNS"):
        module.coordinate_must_start(
            config=config,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=now,
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(events, []),
            sns=FakeSNS(failures=1),
            sleep=lambda _seconds: None,
        )
    sns = FakeSNS()
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now + timedelta(seconds=1),
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(events, [_result("nonterminal", "PENDING")]),
        sns=sns,
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "requested"
    assert sum(key.endswith("CANCEL_REQUESTED.json") for key in s3.values) == 1
    assert sum(key.endswith("REQUEST_ALERTED.json") for key in s3.values) == 1
    assert len(sns.messages) == 1


def test_ssm_invocation_not_yet_visible_is_retried_without_resend() -> None:
    module = _load_handler()
    missing = ClientError(
        {"Error": {"Code": "InvocationDoesNotExist", "Message": "not visible"}},
        "GetCommandInvocation",
    )
    events: list[str] = []
    ssm = FakeSSM(events, [missing, _result("nonterminal", "PENDING")])
    result = module._run_controller_cancel(
        ssm,
        instance_id="i-controller-0",
        sky_job_name=JOB_NAME,
        sleep=lambda _seconds: None,
    )
    assert result["outcome"] == "nonterminal"
    assert len(ssm.commands) == 1


def test_terminal_result_rejects_historical_same_name_job_ids() -> None:
    module = _load_handler()
    value = _result("terminal", "CANCELLED")
    value["before_nonterminal_job_ids"] = []
    value["all_job_ids"] = [1, 2, 3]
    with pytest.raises(ValueError, match="terminal cancellation result"):
        module._validate_result(value, sky_job_name=JOB_NAME)
    assert module._validate_result(
        value,
        sky_job_name=JOB_NAME,
        expected_target_job_id=3,
    )["target_job_id"] == 3


@pytest.mark.parametrize(
    ("outcome", "field", "boolean_value"),
    [
        ("terminal", "before_nonterminal_job_ids", [True]),
        ("nonterminal", "after_nonterminal_job_ids", [True]),
        ("terminal", "all_job_ids", [True]),
        ("terminal", "target_job_id", True),
    ],
)
def test_controller_result_rejects_boolean_job_ids(
    outcome: str,
    field: str,
    boolean_value: object,
) -> None:
    module = _load_handler()
    value = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_cancel_result_v1",
        "sky_job_name": JOB_NAME,
        "outcome": outcome,
        "before_nonterminal_job_ids": [1],
        "after_nonterminal_job_ids": [1] if outcome == "nonterminal" else [],
        "all_job_ids": [1],
        "target_job_id": 1,
        "target_status": "PENDING" if outcome == "nonterminal" else "CANCELLED",
        "observed_at": NOW.isoformat().replace("+00:00", "Z"),
    }
    value[field] = boolean_value
    with pytest.raises(ValueError, match="job ID|job_ids"):
        module._validate_result(
            value,
            sky_job_name=JOB_NAME,
            expected_target_job_id=1,
        )


def test_controller_result_rejects_zero_job_ids() -> None:
    module = _load_handler()
    value = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_cancel_result_v1",
        "sky_job_name": JOB_NAME,
        "outcome": "terminal",
        "before_nonterminal_job_ids": [],
        "after_nonterminal_job_ids": [],
        "all_job_ids": [0],
        "target_job_id": 0,
        "target_status": "CANCELLED",
        "observed_at": NOW.isoformat().replace("+00:00", "Z"),
    }
    with pytest.raises(ValueError, match="job ID|job_ids"):
        module._validate_result(
            value,
            sky_job_name=JOB_NAME,
            expected_target_job_id=0,
        )


@pytest.mark.parametrize(
    "outcome",
    ["not_found", "nonterminal", "terminal", "ambiguous"],
)
def test_bound_result_rejects_every_mismatched_target_outcome(
    outcome: str,
) -> None:
    module = _load_handler()
    target = None if outcome == "not_found" else 4
    active = [4] if outcome in {"nonterminal", "ambiguous"} else []
    value = {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_cancel_result_v1",
        "sky_job_name": JOB_NAME,
        "outcome": outcome,
        "before_nonterminal_job_ids": active,
        "after_nonterminal_job_ids": active,
        "all_job_ids": [] if target is None else [4],
        "target_job_id": target,
        "target_status": (
            None
            if target is None
            else ("CANCELLED" if outcome == "terminal" else "RUNNING")
        ),
        "observed_at": NOW.isoformat().replace("+00:00", "Z"),
    }
    with pytest.raises(ValueError, match="target job ID|terminal"):
        module._validate_result(
            value,
            sky_job_name=JOB_NAME,
            expected_target_job_id=3,
        )


def test_controller_shim_binds_unique_state_job_and_current_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_handler()

    class Status:
        def __init__(self, value: str):
            self.value = value

    class State:
        calls = 0
        status_calls = 0

        @staticmethod
        def get_all_job_ids_by_name(name):
            assert name == JOB_NAME
            return [3]

        @staticmethod
        def get_status(job_id):
            assert job_id == 3
            State.status_calls += 1
            return Status("PENDING" if State.status_calls == 1 else "CANCELLED")

    cancel_calls: list[tuple[str, str]] = []
    jobs_module = types.ModuleType("sky.jobs")
    jobs_module.state = State
    jobs_module.utils = types.SimpleNamespace(
        cancel_jobs_by_id=lambda ids, current_workspace, graceful: cancel_calls.append(
            (ids, current_workspace, graceful)
        )
    )
    sky_module = types.ModuleType("sky")
    sky_module.jobs = jobs_module
    monkeypatch.setitem(sys.modules, "sky", sky_module)
    monkeypatch.setitem(sys.modules, "sky.jobs", jobs_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shim",
            JOB_NAME,
            "3",
        ],
    )
    output = io.StringIO()
    with redirect_stdout(output):
        exec(module._controller_shim(), {})  # noqa: S102
    value = json.loads(output.getvalue().split("=", 1)[1])
    assert value["target_job_id"] == 3
    assert value["outcome"] == "terminal"
    assert cancel_calls == [([3], "default", False)]


def test_controller_shim_discovers_unique_target_without_cancelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_handler()

    class Status:
        value = "PENDING"

    class State:
        @staticmethod
        def get_nonterminal_job_ids_by_name(_name, all_users=False):
            assert all_users is False
            return [3]

        @staticmethod
        def get_all_job_ids_by_name(_name):
            return [1, 2, 3]

        @staticmethod
        def get_status(job_id):
            assert job_id == 3
            return Status()

    cancel_calls: list[object] = []
    jobs_module = types.ModuleType("sky.jobs")
    jobs_module.state = State
    jobs_module.utils = types.SimpleNamespace(
        cancel_jobs_by_id=lambda *args, **kwargs: cancel_calls.append(
            (args, kwargs)
        )
    )
    sky_module = types.ModuleType("sky")
    sky_module.jobs = jobs_module
    monkeypatch.setitem(sys.modules, "sky", sky_module)
    monkeypatch.setitem(sys.modules, "sky.jobs", jobs_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shim",
            JOB_NAME,
            "-",
        ],
    )
    output = io.StringIO()
    with redirect_stdout(output):
        exec(module._controller_shim(), {})  # noqa: S102
    value = json.loads(output.getvalue().split("=", 1)[1])
    assert value["target_job_id"] == 3
    assert value["outcome"] == "nonterminal"
    assert cancel_calls == []


def test_durable_target_binding_reconciles_with_historical_same_name_jobs() -> None:
    module, config, s3, ec2, events, now = _setup()
    discovered = _result("nonterminal", "PENDING")
    discovered["all_job_ids"] = [1, 2, 3]
    first_ssm = FakeSSM(events, [discovered])
    first = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=first_ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert first["status"] == "requested"
    assert config.target_binding_key in s3.values
    terminal = _result("terminal", "CANCELLED")
    terminal["before_nonterminal_job_ids"] = []
    terminal["all_job_ids"] = [1, 2, 3]
    second_ssm = FakeSSM(events, [terminal])
    second = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now + timedelta(minutes=1),
        s3=s3,
        ec2=ec2,
        ssm=second_ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert second["status"] == "completed"
    assert f"{JOB_NAME} 3" in str(
        second_ssm.commands[0]["Parameters"]["commands"][0]
    )


def test_explicit_observed_target_is_bound_before_current_job_cancel() -> None:
    module, config, s3, ec2, events, now = _setup()
    config = replace(config, expected_target_job_id=3)
    terminal = _result("terminal", "CANCELLED")
    terminal["before_nonterminal_job_ids"] = []
    terminal["all_job_ids"] = [1, 2, 3]
    ssm = FakeSSM(events, [terminal])
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "completed"
    assert events.index("s3:TARGET_BOUND.json") < events.index("ssm:send")
    assert f"{JOB_NAME} 3" in str(ssm.commands[0]["Parameters"]["commands"][0])


def test_controller_shim_multiple_nonterminal_exact_name_jobs_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_handler()

    class State:
        @staticmethod
        def get_nonterminal_job_ids_by_name(_name, all_users=False):
            assert all_users is False
            return [3, 4]

        @staticmethod
        def get_all_job_ids_by_name(_name):
            return [4, 3]

    cancel_calls: list[object] = []
    jobs_module = types.ModuleType("sky.jobs")
    jobs_module.state = State
    jobs_module.utils = types.SimpleNamespace(
        cancel_jobs_by_id=lambda *args, **kwargs: cancel_calls.append(
            (args, kwargs)
        )
    )
    sky_module = types.ModuleType("sky")
    sky_module.jobs = jobs_module
    monkeypatch.setitem(sys.modules, "sky", sky_module)
    monkeypatch.setitem(sys.modules, "sky.jobs", jobs_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shim",
            JOB_NAME,
            "-",
        ],
    )
    output = io.StringIO()
    with redirect_stdout(output):
        exec(module._controller_shim(), {})  # noqa: S102
    value = json.loads(output.getvalue().split("=", 1)[1])
    assert value["outcome"] == "ambiguous"
    assert value["target_job_id"] is None
    assert cancel_calls == []


def test_legacy_non_strict_observe_only_fails_before_ssm_or_mutation() -> None:
    module, base, s3, ec2, events, now = _setup()
    config = replace(base, observe_only=True)

    with pytest.raises(
        ValueError,
        match="observe-only requires complete exact controller authority",
    ):
        module.coordinate_must_start(
            config=config,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=now,
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(events, [_result("terminal", "CANCELLED")]),
            sns=FakeSNS(),
            events=FakeEvents(),
            sleep=lambda _seconds: None,
        )

    assert events == []
    assert not any(key.endswith("/CANCEL_REQUESTED.json") for key in s3.values)
    assert not any(key.endswith("/CANCEL_COMPLETED.json") for key in s3.values)


def _strict_config(config, s3, *, observe_only: bool):
    descriptor_file_sha = __import__("hashlib").sha256(
        _canonical(s3.values[DESCRIPTOR_KEY])
    ).hexdigest()
    submission = s3.values[config.submission_key]
    return replace(
        config,
        observe_only=observe_only,
        expected_target_job_id=3,
        expected_workspace="default",
        expected_descriptor_file_sha256=descriptor_file_sha,
        expected_submission_submitted_at=str(submission["submitted_at"]),
        expected_controller_instance_id="i-0511af4e31aa5406a",
        expected_controller_instance_type="c6a.xlarge",
        expected_controller_profile_arn=(
            "arn:aws:iam::246813579024:instance-profile/"
            "keep-glm52-skypilot-controller"
        ),
        expected_controller_cluster_name="sky-jobs-controller-9d9f31a9-9d9f31a9",
        starting_grace_seconds=20,
    )


def _strict_lifecycle_config(config, s3, *, observe_only: bool):
    return replace(
        _strict_config(config, s3, observe_only=observe_only),
        reconciliation_rule_name="keep-glm52-sky-must-start-cancel",
        primary_wake_max_wait_seconds=120,
    )


def _strict_result(
    *,
    operation: str,
    status: str,
    start_at: datetime | None,
    observed_at: datetime = NOW,
    cancel_called: bool = False,
    post_status: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_sky_must_start_controller_result_v1",
        "operation": operation,
        "sky_job_name": JOB_NAME,
        "workspace": "default",
        "all_job_ids": [1, 2, 3],
        "target_job_id": 3,
        "target_job_name": JOB_NAME,
        "target_workspace": "default",
        "status": status,
        "schedule_state": "LAUNCHING" if status != "RUNNING" else "ALIVE",
        "submitted_at": "2026-07-26T03:52:49.354Z",
        "start_at": (
            None if start_at is None else start_at.isoformat().replace("+00:00", "Z")
        ),
        "worker_cluster_name": None,
        "recovery_count": 0,
        "cancel_called": cancel_called,
        "post_status": post_status,
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
    }


def test_observe_only_binds_exact_job_3_and_never_cancels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, base, s3, ec2, events, now = _setup()
    config = _strict_config(base, s3, observe_only=True)
    ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status="PENDING",
                start_at=None,
            )
        ],
    )
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "observed"
    assert result["target_job_id"] == 3
    assert any(key.endswith("/JOB_BINDING.json") for key in s3.values)
    assert any("/observations/" in key for key in s3.values)
    assert not any(key.endswith("/CANCEL_REQUESTED.json") for key in s3.values)
    command = str(ssm.commands[0]["Parameters"]["commands"][0])
    assert " observe " in command
    assert " cancel " not in command
    assert [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ] == [
        {
            "component": "glm52-sky-must-start",
            "stage": "post-observation",
        },
        {
            "component": "glm52-sky-must-start",
            "stage": "before-binding",
        },
        {
            "component": "glm52-sky-must-start",
            "stage": "after-binding",
        },
    ]


def test_primary_deadline_event_waits_until_exact_deadline_before_job_3_read() -> None:
    module, base, s3, ec2, events, _now = _setup()
    config = _strict_lifecycle_config(base, s3, observe_only=False)
    ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status="PENDING",
                start_at=None,
                observed_at=DEADLINE,
            ),
            _strict_result(
                operation="cancel",
                status="PENDING",
                start_at=None,
                observed_at=DEADLINE + timedelta(seconds=1),
                cancel_called=True,
                post_status="CANCELLED",
            ),
        ],
    )
    clock = DEADLINE - timedelta(seconds=60)

    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=clock,
        trigger="primary-deadline",
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        events=FakeEvents(),
        sleep=lambda seconds: events.append(f"sleep:{seconds:g}"),
    )

    assert result["status"] == "completed"
    assert events[0] == "sleep:60"
    assert events[1] == "ssm:send"


def test_primary_deadline_event_rejects_unbounded_early_wake() -> None:
    module, base, s3, ec2, events, _now = _setup()
    config = _strict_lifecycle_config(base, s3, observe_only=False)
    eventbridge = FakeEvents()

    with pytest.raises(ValueError, match="primary deadline wake is too early"):
        module.coordinate_must_start(
            config=config,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=DEADLINE - timedelta(seconds=121),
            trigger="primary-deadline",
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(events, []),
            sns=FakeSNS(),
            events=eventbridge,
            sleep=lambda seconds: events.append(f"sleep:{seconds:g}"),
        )

    assert events == []
    assert eventbridge.disabled == []


def test_active_pending_job_3_rereads_then_cancels_only_exact_id() -> None:
    module, base, s3, ec2, events, now = _setup()
    config = _strict_config(base, s3, observe_only=False)
    ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status="PENDING",
                start_at=None,
            ),
            _strict_result(
                operation="cancel",
                status="PENDING",
                start_at=None,
                cancel_called=True,
                post_status="CANCELLED",
            ),
        ],
    )
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "completed"
    assert len(ssm.commands) == 2
    cancel_command = str(ssm.commands[1]["Parameters"]["commands"][0])
    assert " cancel " in cancel_command
    assert f"{JOB_NAME} default 3" in cancel_command
    assert "terminate-instances" not in cancel_command


def test_authenticated_completion_disables_only_exact_reconciliation_rule() -> None:
    module, base, s3, ec2, events, now = _setup()
    config = _strict_lifecycle_config(base, s3, observe_only=False)
    eventbridge = FakeEvents()
    ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status="PENDING",
                start_at=None,
            ),
            _strict_result(
                operation="cancel",
                status="PENDING",
                start_at=None,
                cancel_called=True,
                post_status="CANCELLED",
            ),
        ],
    )

    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        trigger="reconcile",
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        events=eventbridge,
        sleep=lambda _seconds: None,
    )

    assert result["status"] == "completed"
    assert eventbridge.disabled == [
        {"Name": "keep-glm52-sky-must-start-cancel"}
    ]


@pytest.mark.parametrize("status", ["RUNNING", "RECOVERING", "SUCCEEDED"])
def test_timely_controller_start_is_accepted_without_cancellation(status: str) -> None:
    module, base, s3, ec2, events, now = _setup()
    config = _strict_config(base, s3, observe_only=False)
    ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status=status,
                start_at=DEADLINE - timedelta(seconds=1),
            )
        ],
    )
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "timely-started"
    assert len(ssm.commands) == 1
    assert any(key.endswith("/TIMELY_START_ACCEPTED.json") for key in s3.values)
    assert not any(key.endswith("/CANCEL_REQUESTED.json") for key in s3.values)


def test_authenticated_acceptance_disables_exact_rule_idempotently() -> None:
    module, base, s3, ec2, events, now = _setup()
    config = _strict_lifecycle_config(base, s3, observe_only=False)
    eventbridge = FakeEvents()
    first = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        trigger="reconcile",
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(
            events,
            [
                _strict_result(
                    operation="observe",
                    status="RUNNING",
                    start_at=DEADLINE - timedelta(seconds=1),
                )
            ],
        ),
        sns=FakeSNS(),
        events=eventbridge,
        sleep=lambda _seconds: None,
    )
    second = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now + timedelta(seconds=1),
        trigger="reconcile",
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(
            events,
            [
                _strict_result(
                    operation="observe",
                    status="RUNNING",
                    start_at=DEADLINE - timedelta(seconds=1),
                    observed_at=now + timedelta(seconds=1),
                )
            ],
        ),
        sns=FakeSNS(),
        events=eventbridge,
        sleep=lambda _seconds: None,
    )

    assert first["status"] == second["status"] == "timely-started"
    assert eventbridge.disabled == [
        {"Name": "keep-glm52-sky-must-start-cancel"},
        {"Name": "keep-glm52-sky-must-start-cancel"},
    ]


def test_existing_acceptance_requires_its_exact_source_before_rule_disable() -> None:
    module, base, s3, ec2, events, now = _setup()
    observe_config = _strict_lifecycle_config(base, s3, observe_only=True)
    observed = module.coordinate_must_start(
        config=observe_config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        trigger="reconcile",
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(
            events,
            [
                _strict_result(
                    operation="observe",
                    status="PENDING",
                    start_at=None,
                )
            ],
        ),
        sns=FakeSNS(),
        events=FakeEvents(),
        sleep=lambda _seconds: None,
    )
    assert observed["status"] == "observed"
    binding = s3.values[observe_config.job_binding_key]
    active_config = replace(
        observe_config,
        observe_only=False,
        expected_activation_job_binding_sha256=str(
            binding["job_binding_body_sha256"]
        ),
        expected_activation_observation_sha256=str(
            binding["observation_body_sha256"]
        ),
    )
    nonexistent_source_sha = "f" * 64
    s3.values[active_config.accepted_key] = module.build_timely_start_accepted(
        run_id=RUN_ID,
        managed_mode=MODE,
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=str(
            s3.values[DESCRIPTOR_KEY]["descriptor_body_sha256"]
        ),
        submission_body_sha256=str(
            s3.values[base.submission_key]["submission_body_sha256"]
        ),
        sky_job_name=JOB_NAME,
        must_start_by=DEADLINE,
        source_kind="controller-observation",
        source_key=(
            f"{active_config.marker_prefix}/observations/"
            f"{nonexistent_source_sha}.json"
        ),
        source_version_id="missing-version",
        source_etag='"forged-etag"',
        source_body_sha256=nonexistent_source_sha,
        source_last_modified=DEADLINE - timedelta(seconds=2),
        started_at=DEADLINE - timedelta(seconds=1),
        accepted_at=now,
    )
    eventbridge = FakeEvents()

    with pytest.raises(
        ValueError,
        match="accepted start source observation is missing",
    ):
        module.coordinate_must_start(
            config=active_config,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=now + timedelta(seconds=1),
            trigger="reconcile",
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(events, []),
            sns=FakeSNS(),
            events=eventbridge,
            sleep=lambda _seconds: None,
        )

    assert eventbridge.disabled == []


def test_existing_completion_requires_fresh_exact_terminal_state_before_disable() -> None:
    module, base, s3, ec2, events, now = _setup()
    observe_config = _strict_lifecycle_config(base, s3, observe_only=True)
    observed = module.coordinate_must_start(
        config=observe_config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        trigger="reconcile",
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(
            events,
            [
                _strict_result(
                    operation="observe",
                    status="PENDING",
                    start_at=None,
                )
            ],
        ),
        sns=FakeSNS(),
        events=FakeEvents(),
        sleep=lambda _seconds: None,
    )
    assert observed["status"] == "observed"
    binding = s3.values[observe_config.job_binding_key]
    active_config = replace(
        observe_config,
        observe_only=False,
        expected_activation_job_binding_sha256=str(
            binding["job_binding_body_sha256"]
        ),
        expected_activation_observation_sha256=str(
            binding["observation_body_sha256"]
        ),
    )
    authority = {
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": BUCKET,
        "descriptor_body_sha256": str(
            s3.values[DESCRIPTOR_KEY]["descriptor_body_sha256"]
        ),
        "submission_body_sha256": str(
            s3.values[base.submission_key]["submission_body_sha256"]
        ),
        "sky_job_name": JOB_NAME,
        "must_start_by": DEADLINE,
    }
    request = module.build_must_start_cancel_requested(
        **authority,
        requested_at=now,
    )
    completion = module.build_must_start_cancel_completed(
        **authority,
        request_body_sha256=str(request["request_body_sha256"]),
        completed_at=now,
        terminal_status="CANCELLED",
    )
    s3.values[active_config.request_key] = request
    s3.values[active_config.completion_key] = completion
    eventbridge = FakeEvents()

    with pytest.raises(
        ValueError,
        match="completion terminal controller evidence mismatch",
    ):
        module.coordinate_must_start(
            config=active_config,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=now + timedelta(seconds=1),
            trigger="reconcile",
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(
                events,
                [
                    _strict_result(
                        operation="observe",
                        status="PENDING",
                        start_at=None,
                        observed_at=now + timedelta(seconds=1),
                    )
                ],
            ),
            sns=FakeSNS(),
            events=eventbridge,
            sleep=lambda _seconds: None,
        )

    assert eventbridge.disabled == []


def test_strict_observe_existing_completion_has_no_cancel_side_effects() -> None:
    module, base, s3, ec2, events, now = _setup()
    config = _strict_lifecycle_config(base, s3, observe_only=True)
    authority = {
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": BUCKET,
        "descriptor_body_sha256": str(
            s3.values[DESCRIPTOR_KEY]["descriptor_body_sha256"]
        ),
        "submission_body_sha256": str(
            s3.values[base.submission_key]["submission_body_sha256"]
        ),
        "sky_job_name": JOB_NAME,
        "must_start_by": DEADLINE,
    }
    request = module.build_must_start_cancel_requested(
        **authority,
        requested_at=now,
    )
    completion = module.build_must_start_cancel_completed(
        **authority,
        request_body_sha256=str(request["request_body_sha256"]),
        completed_at=now,
        terminal_status="CANCELLED",
    )
    s3.values[config.request_key] = request
    s3.values[config.completion_key] = completion
    request_before = dict(request)
    completion_before = dict(completion)
    sns = FakeSNS()
    eventbridge = FakeEvents()
    ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status="CANCELLED",
                start_at=DEADLINE + timedelta(seconds=1),
                observed_at=now + timedelta(seconds=1),
            )
        ],
    )

    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now + timedelta(seconds=1),
        trigger="reconcile",
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=sns,
        events=eventbridge,
        sleep=lambda _seconds: None,
    )

    assert sns.messages == []
    assert s3.values[config.request_key] == request_before
    assert s3.values[config.completion_key] == completion_before
    assert config.request_alerted_key not in s3.values
    assert config.completion_alerted_key not in s3.values
    assert not {
        "s3:CANCEL_REQUESTED.json",
        "s3:CANCEL_COMPLETED.json",
        "s3:REQUEST_ALERTED.json",
        "s3:COMPLETION_ALERTED.json",
    }.intersection(events)
    assert eventbridge.disabled == []
    assert result == {
        "status": "observed",
        "run_id": RUN_ID,
        "target_job_id": 3,
        "controller_status": "CANCELLED",
        "decision": "completed",
    }
    assert len(ssm.commands) == 1
    command = str(ssm.commands[0]["Parameters"]["commands"][0])
    assert " observe " in command
    assert " cancel " not in command


def test_observation_without_authenticated_lifecycle_does_not_disable_rule() -> None:
    module, base, s3, ec2, events, now = _setup()
    config = _strict_lifecycle_config(base, s3, observe_only=True)
    eventbridge = FakeEvents()

    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        trigger="reconcile",
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(
            events,
            [
                _strict_result(
                    operation="observe",
                    status="PENDING",
                    start_at=None,
                )
            ],
        ),
        sns=FakeSNS(),
        events=eventbridge,
        sleep=lambda _seconds: None,
    )

    assert result["status"] == "observed"
    assert eventbridge.disabled == []


def test_active_mode_requires_exact_prior_binding_and_observation_before_ssm() -> None:
    module, base, s3, ec2, events, now = _setup()
    config = replace(
        _strict_lifecycle_config(base, s3, observe_only=False),
        expected_activation_job_binding_sha256="a" * 64,
        expected_activation_observation_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="activation JOB_BINDING.json is missing"):
        module.coordinate_must_start(
            config=config,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=now,
            trigger="reconcile",
            s3=s3,
            ec2=ec2,
            ssm=FakeSSM(events, []),
            sns=FakeSNS(),
            events=FakeEvents(),
            sleep=lambda _seconds: None,
        )

    assert events == []


def test_active_mode_authenticates_exact_prior_observation_proof() -> None:
    module, base, s3, ec2, events, now = _setup()
    observe_config = _strict_lifecycle_config(base, s3, observe_only=True)
    observed = module.coordinate_must_start(
        config=observe_config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        trigger="reconcile",
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(
            events,
            [
                _strict_result(
                    operation="observe",
                    status="PENDING",
                    start_at=None,
                )
            ],
        ),
        sns=FakeSNS(),
        events=FakeEvents(),
        sleep=lambda _seconds: None,
    )
    assert observed["status"] == "observed"
    binding = s3.values[observe_config.job_binding_key]
    observation_sha = str(binding["observation_body_sha256"])
    active_config = replace(
        observe_config,
        observe_only=False,
        expected_activation_job_binding_sha256=str(
            binding["job_binding_body_sha256"]
        ),
        expected_activation_observation_sha256=observation_sha,
    )
    eventbridge = FakeEvents()

    result = module.coordinate_must_start(
        config=active_config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now + timedelta(seconds=1),
        trigger="reconcile",
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(
            events,
            [
                _strict_result(
                    operation="observe",
                    status="PENDING",
                    start_at=None,
                    observed_at=now + timedelta(seconds=1),
                ),
                _strict_result(
                    operation="cancel",
                    status="PENDING",
                    start_at=None,
                    observed_at=now + timedelta(seconds=2),
                    cancel_called=True,
                    post_status="CANCELLED",
                ),
            ],
        ),
        sns=FakeSNS(),
        events=eventbridge,
        sleep=lambda _seconds: None,
    )

    assert result["status"] == "completed"
    assert eventbridge.disabled == [
        {"Name": "keep-glm52-sky-must-start-cancel"}
    ]


def test_starting_grace_reobserves_and_accepts_timely_transition() -> None:
    module, base, s3, ec2, events, _now = _setup()
    config = _strict_config(base, s3, observe_only=False)
    first_observed = DEADLINE + timedelta(seconds=5)
    ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status="STARTING",
                start_at=None,
                observed_at=first_observed,
            ),
            _strict_result(
                operation="observe",
                status="RUNNING",
                start_at=DEADLINE - timedelta(microseconds=1),
                observed_at=DEADLINE + timedelta(seconds=20),
            ),
        ],
    )
    sleeps: list[float] = []
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=first_observed,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=sleeps.append,
    )
    assert result["status"] == "timely-started"
    assert len(ssm.commands) == 2
    assert sleeps == [15]


def test_starting_grace_reobserves_and_cancels_late_transition() -> None:
    module, base, s3, ec2, events, _now = _setup()
    config = _strict_config(base, s3, observe_only=False)
    first_observed = DEADLINE + timedelta(seconds=5)
    ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status="STARTING",
                start_at=None,
                observed_at=first_observed,
            ),
            _strict_result(
                operation="observe",
                status="RUNNING",
                start_at=DEADLINE + timedelta(microseconds=1),
                observed_at=DEADLINE + timedelta(seconds=20),
            ),
            _strict_result(
                operation="cancel",
                status="RUNNING",
                start_at=DEADLINE + timedelta(microseconds=1),
                observed_at=DEADLINE + timedelta(seconds=21),
                cancel_called=True,
                post_status="CANCELLED",
            ),
        ],
    )
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=first_observed,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "completed"
    assert len(ssm.commands) == 3
    assert " cancel " in str(ssm.commands[-1]["Parameters"]["commands"][0])


def test_existing_request_is_superseded_by_exact_timely_start() -> None:
    module, base, s3, ec2, events, now = _setup()
    config = _strict_config(base, s3, observe_only=False)
    descriptor = s3.values[DESCRIPTOR_KEY]
    submission = s3.values[config.submission_key]
    s3.values[config.request_key] = build_must_start_cancel_requested(
        run_id=RUN_ID,
        managed_mode=MODE,
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256=str(submission["submission_body_sha256"]),
        sky_job_name=JOB_NAME,
        must_start_by=DEADLINE,
        requested_at=DEADLINE,
    )
    s3.last_modified[config.request_key] = DEADLINE
    ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status="RUNNING",
                start_at=DEADLINE - timedelta(microseconds=1),
            )
        ],
    )
    result = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "timely-started"
    assert any(key.endswith("/CANCEL_SUPERSEDED.json") for key in s3.values)
    assert len(ssm.commands) == 1


def test_duplicate_exact_invocation_converges_on_accepted_start() -> None:
    module, base, s3, ec2, events, now = _setup()
    config = _strict_config(base, s3, observe_only=False)
    first = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now,
        s3=s3,
        ec2=ec2,
        ssm=FakeSSM(
            events,
            [
                _strict_result(
                    operation="observe",
                    status="RUNNING",
                    start_at=DEADLINE - timedelta(seconds=1),
                )
            ],
        ),
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    second_ssm = FakeSSM(
        events,
        [
            _strict_result(
                operation="observe",
                status="RUNNING",
                start_at=DEADLINE - timedelta(seconds=1),
                observed_at=now + timedelta(seconds=1),
            )
        ],
    )
    second = module.coordinate_must_start(
        config=config,
        runtime_account_id="246813579024",
        runtime_region="us-west-2",
        now=now + timedelta(seconds=1),
        s3=s3,
        ec2=ec2,
        ssm=second_ssm,
        sns=FakeSNS(),
        sleep=lambda _seconds: None,
    )
    assert first["status"] == second["status"] == "timely-started"
    assert len(second_ssm.commands) == 1
    assert sum(key.endswith("/TIMELY_START_ACCEPTED.json") for key in s3.values) == 1


def test_wrong_exact_controller_identity_mutates_no_markers() -> None:
    module, base, s3, _ec2, events, now = _setup()
    config = _strict_config(base, s3, observe_only=True)
    foreign = FakeEC2(instance_type="m6i.2xlarge")
    with pytest.raises(ValueError, match="instance type"):
        module.coordinate_must_start(
            config=config,
            runtime_account_id="246813579024",
            runtime_region="us-west-2",
            now=now,
            s3=s3,
            ec2=foreign,
            ssm=FakeSSM(events, []),
            sns=FakeSNS(),
            sleep=lambda _seconds: None,
        )
    assert events == []


def test_strict_controller_shim_rereads_then_cancels_exact_job_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_handler()

    class EnumValue:
        def __init__(self, value: str):
            self.value = value

    row = {
        "job_id": 3,
        "job_name": JOB_NAME,
        "task_name": JOB_NAME,
        "workspace": "default",
        "status": EnumValue("PENDING"),
        "schedule_state": EnumValue("LAUNCHING"),
        "submitted_at": SUBMITTED_AT.timestamp(),
        "start_at": None,
        "current_cluster_name": None,
        "recovery_count": 0,
    }

    class State:
        @staticmethod
        def get_managed_jobs_with_filters(**kwargs):
            assert kwargs == {
                "job_ids": [3],
                "workspace_match": "default",
            }
            return [row], 1

        @staticmethod
        def get_all_job_ids_by_name(name):
            assert name == JOB_NAME
            return [1, 2, 3]

        @staticmethod
        def get_status(job_id):
            assert job_id == 3
            return EnumValue("CANCELLED")

    cancel_calls: list[tuple[object, str, bool]] = []
    jobs_module = types.ModuleType("sky.jobs")
    jobs_module.state = State
    jobs_module.utils = types.SimpleNamespace(
        cancel_jobs_by_id=lambda ids, current_workspace, graceful: cancel_calls.append(
            (ids, current_workspace, graceful)
        ),
        generate_managed_job_cluster_name=lambda task_name, job_id: (
            f"{task_name}-{job_id}"
        ),
    )
    sky_module = types.ModuleType("sky")
    sky_module.jobs = jobs_module
    monkeypatch.setitem(sys.modules, "sky", sky_module)
    monkeypatch.setitem(sys.modules, "sky.jobs", jobs_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shim",
            JOB_NAME,
            "default",
            "3",
            "cancel",
            str(DEADLINE.timestamp()),
            str(NOW.timestamp()),
        ],
    )
    output = io.StringIO()
    with redirect_stdout(output):
        exec(module._strict_controller_shim(), {})  # noqa: S102
    value = json.loads(output.getvalue().split("=", 1)[1])
    assert value["all_job_ids"] == [1, 2, 3]
    assert value["target_job_id"] == 3
    assert value["cancel_called"] is True
    assert value["post_status"] == "CANCELLED"
    assert cancel_calls == [([3], "default", False)]


def test_environment_builds_observe_only_exact_controller_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, base, s3, _ec2, _events, _now = _setup()
    strict = _strict_lifecycle_config(base, s3, observe_only=True)
    values = {
        "EXPECTED_ACCOUNT_ID": strict.account_id,
        "AWS_REGION": strict.region,
        "CAMPAIGN_BUCKET": strict.bucket,
        "CAMPAIGN_DESCRIPTOR_KEY": strict.descriptor_key,
        "IMMUTABLE_SUBMISSION_KEY": strict.submission_key,
        "SUBMISSION_BODY_SHA256": strict.submission_body_sha256,
        "CAMPAIGN_RUN_ID": strict.run_id,
        "MANAGED_MODE": strict.managed_mode,
        "SKY_JOB_NAME": strict.sky_job_name,
        "MUST_START_BY": strict.must_start_by,
        "ALERT_TOPIC_ARN": strict.alert_topic_arn,
        "EXPECTED_TARGET_JOB_ID": "3",
        "OBSERVE_ONLY": "true",
        "EXPECTED_WORKSPACE": "default",
        "EXPECTED_DESCRIPTOR_FILE_SHA256": str(
            strict.expected_descriptor_file_sha256
        ),
        "EXPECTED_SUBMISSION_SUBMITTED_AT": str(
            strict.expected_submission_submitted_at
        ),
        "EXPECTED_CONTROLLER_INSTANCE_ID": str(
            strict.expected_controller_instance_id
        ),
        "EXPECTED_CONTROLLER_INSTANCE_TYPE": str(
            strict.expected_controller_instance_type
        ),
        "EXPECTED_CONTROLLER_PROFILE_ARN": str(
            strict.expected_controller_profile_arn
        ),
        "EXPECTED_CONTROLLER_CLUSTER_NAME": str(
            strict.expected_controller_cluster_name
        ),
        "STARTING_GRACE_SECONDS": "20",
        "RECONCILIATION_RULE_NAME": str(strict.reconciliation_rule_name),
        "PRIMARY_WAKE_MAX_WAIT_SECONDS": "120",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    loaded = module._config_from_environment()
    assert loaded == strict
    assert loaded.observe_only is True
    assert loaded.strict_controller_authority is True


def test_environment_rejects_raw_observe_only_defaults_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, base, _s3, _ec2, _events, _now = _setup()
    values = {
        "EXPECTED_ACCOUNT_ID": base.account_id,
        "AWS_REGION": base.region,
        "CAMPAIGN_BUCKET": base.bucket,
        "CAMPAIGN_DESCRIPTOR_KEY": base.descriptor_key,
        "IMMUTABLE_SUBMISSION_KEY": base.submission_key,
        "SUBMISSION_BODY_SHA256": base.submission_body_sha256,
        "CAMPAIGN_RUN_ID": base.run_id,
        "MANAGED_MODE": base.managed_mode,
        "SKY_JOB_NAME": base.sky_job_name,
        "MUST_START_BY": base.must_start_by,
        "ALERT_TOPIC_ARN": base.alert_topic_arn,
        "EXPECTED_TARGET_JOB_ID": "3",
        "OBSERVE_ONLY": "true",
        "EXPECTED_WORKSPACE": "default",
        "EXPECTED_DESCRIPTOR_FILE_SHA256": "disabled",
        "EXPECTED_SUBMISSION_SUBMITTED_AT": "1970-01-01T00:00:00Z",
        "EXPECTED_CONTROLLER_INSTANCE_ID": "disabled",
        "EXPECTED_CONTROLLER_INSTANCE_TYPE": "disabled",
        "EXPECTED_CONTROLLER_PROFILE_ARN": "disabled",
        "EXPECTED_CONTROLLER_CLUSTER_NAME": "disabled",
        "STARTING_GRACE_SECONDS": "20",
        "RECONCILIATION_RULE_NAME": "keep-glm52-sky-must-start-cancel",
        "PRIMARY_WAKE_MAX_WAIT_SECONDS": "120",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValueError, match="complete exact controller authority"):
        module._config_from_environment()


def test_lambda_handler_reuses_one_session_and_client_set_across_warm_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, _s3, _ec2, _events, _now = _setup()
    session_regions: list[str] = []
    client_services: list[str] = []
    client_configs: list[object | None] = []
    clients: dict[str, object] = {}
    identity_calls: list[str] = []

    class FakeSTS:
        @staticmethod
        def get_caller_identity() -> dict[str, str]:
            identity_calls.append("sts")
            return {"Account": "246813579024"}

    class FakeSession:
        def __init__(self, *, region_name: str) -> None:
            session_regions.append(region_name)
            self.region_name = region_name

        def client(self, service: str, *, config=None) -> object:
            client_services.append(service)
            client_configs.append(config)
            client = FakeSTS() if service == "sts" else object()
            clients[service] = client
            return client

    class FrozenDateTime:
        @staticmethod
        def now(_timezone: timezone) -> datetime:
            return NOW

    coordinate_calls: list[dict[str, object]] = []

    def fake_coordinate(**kwargs):
        coordinate_calls.append(kwargs)
        return {"status": "observed", "run_id": RUN_ID}

    monkeypatch.setattr(module, "_config_from_environment", lambda: config)
    monkeypatch.setattr(module.boto3.session, "Session", FakeSession)
    monkeypatch.setattr(module, "datetime", FrozenDateTime)
    monkeypatch.setattr(module, "coordinate_must_start", fake_coordinate)
    event = {
        "campaign_run_id": RUN_ID,
        "managed_mode": MODE,
        "sky_job_name": JOB_NAME,
        "must_start_by": config.must_start_by,
        "trigger": "reconcile",
    }

    first = module.lambda_handler(event, None)
    second = module.lambda_handler(event, None)

    assert first == second == {"status": "observed", "run_id": RUN_ID}
    assert session_regions == ["us-west-2"]
    assert client_services == ["sts", "sns", "s3", "ec2", "ssm", "events"]
    assert len(client_configs) == 6
    runtime_config = client_configs[0]
    assert runtime_config is not None
    assert all(config is runtime_config for config in client_configs)
    assert runtime_config.connect_timeout == 1
    assert runtime_config.read_timeout == 2
    assert runtime_config.retries == {
        "mode": "standard",
        "total_max_attempts": 1,
    }
    assert runtime_config.tcp_keepalive is True
    assert identity_calls == ["sts", "sts"]
    assert len(coordinate_calls) == 2
    for call in coordinate_calls:
        assert call == {
            "config": config,
            "runtime_account_id": "246813579024",
            "runtime_region": "us-west-2",
            "now": NOW,
            "trigger": "reconcile",
            "s3": clients["s3"],
            "ec2": clients["ec2"],
            "ssm": clients["ssm"],
            "sns": clients["sns"],
            "events": clients["events"],
        }


def test_package_is_import_complete_content_addressed_and_relative_out_safe(
    tmp_path: Path,
) -> None:
    relative_output = Path("nested/must-start.zip")
    output = tmp_path / relative_output
    output.parent.mkdir(parents=True)
    with zipfile.ZipFile(output, "w") as stale:
        stale.writestr("stale-member.txt", "must not survive clean packaging")
    result = subprocess.run(
        [str(PACKAGE), str(relative_output)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest == {
        "path": str(output),
        "requires_s3_object_version": True,
        "s3_key": f"lambda/sky-must-start-cancel/{digest}.zip",
        "sha256": digest,
    }
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert names == {
        "handler.py",
        "mlx_vq/",
        "mlx_vq/__init__.py",
        "mlx_vq/quality/",
        "mlx_vq/quality/__init__.py",
        "mlx_vq/quality/glm52_campaign_watchdog.py",
        "mlx_vq/quality/glm52_sky_campaign.py",
        "mlx_vq/quality/glm52_sky_must_start.py",
    }
    imported = subprocess.run(
        [sys.executable, "-c", "import handler; print(handler.__name__)"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(output)},
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "handler"
