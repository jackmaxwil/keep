from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mlx_vq.quality.glm52_campaign_watchdog import (
    build_skypilot_submission_marker,
)
from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import (
    build_must_start_job_binding,
    build_timely_start_accepted,
    canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / "aws/glm52-gpu/skypilot/publish_must_start_latch.py"
POLICY = ROOT / "src/mlx_vq/quality/glm52_sky_must_start.py"
UTC = timezone.utc
RUN_ID = "glm52-sky-future"
MODE = "cache-seed"
JOB_NAME = f"{RUN_ID}-cache-seed"
ACCOUNT = "246813579024"
REGION = "us-west-2"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
DEADLINE = datetime(2026, 7, 27, 12, tzinfo=UTC)
DESCRIPTOR_KEY = f"campaigns/{RUN_ID}/submissions/seed/campaign.json"


def _load_publisher():
    spec = importlib.util.spec_from_file_location("_worker_must_start", PUBLISHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(character: str) -> str:
    return character * 64


def _authorities() -> tuple[bytes, bytes, str, str]:
    approval = build_gpu_spend_approval(
        ingested_at=DEADLINE - timedelta(days=1),
        slack_permalink=None,
    )
    descriptor = build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=DEADLINE,
        controller_identity=(
            f"arn:aws:iam::{ACCOUNT}:role/keep-glm52-skypilot-controller"
        ),
        worker_identity=f"arn:aws:iam::{ACCOUNT}:role/keep-glm52-gpu-worker",
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
            "training_config_key": (
                f"campaigns/{RUN_ID}/authorities/training.json"
            ),
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
    descriptor_bytes = (
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    descriptor_file_sha = __import__("hashlib").sha256(
        descriptor_bytes
    ).hexdigest()
    submission = build_skypilot_submission_marker(
        run_id=RUN_ID,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submitted_at=DEADLINE - timedelta(hours=1),
        must_start_by=DEADLINE,
        sky_job_name=JOB_NAME,
    )
    submission_bytes = (
        json.dumps(submission, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    submission_key = (
        f"campaigns/{RUN_ID}/monitor/submission-locks/"
        f"{descriptor_file_sha}-{MODE}.json"
    )
    return descriptor_bytes, submission_bytes, descriptor_file_sha, submission_key


def _identity_document(*, pending: datetime) -> bytes:
    return json.dumps(
        {
            "accountId": ACCOUNT,
            "region": REGION,
            "instanceId": "i-0123456789abcdef0",
            "instanceType": "p5.48xlarge",
            "imageId": "ami-0123456789abcdef0",
            "pendingTime": pending.isoformat().replace("+00:00", "Z"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_worker_builds_content_addressed_latch_from_exact_sts_imds_authority() -> None:
    module = _load_publisher()
    descriptor, submission, descriptor_sha, submission_key = _authorities()
    observed = DEADLINE - timedelta(seconds=2)
    latch, key = module.build_authenticated_worker_latch(
        policy_path=POLICY,
        descriptor_bytes=descriptor,
        descriptor_s3_uri=f"s3://{BUCKET}/{DESCRIPTOR_KEY}",
        descriptor_file_sha256=descriptor_sha,
        submission_bytes=submission,
        submission_s3_uri=f"s3://{BUCKET}/{submission_key}",
        submission_body_sha256=json.loads(submission)["submission_body_sha256"],
        managed_mode=MODE,
        sts_identity={
            "Account": ACCOUNT,
            "Arn": (
                f"arn:aws:sts::{ACCOUNT}:assumed-role/"
                "keep-glm52-gpu-worker/i-0123456789abcdef0"
            ),
        },
        identity_document_bytes=_identity_document(
            pending=DEADLINE - timedelta(seconds=10)
        ),
        entrypoint_observed_at=observed,
    )
    assert latch["instance_id"] == "i-0123456789abcdef0"
    assert latch["worker_role_arn"].endswith(":role/keep-glm52-gpu-worker")
    assert latch["repo_tar_sha256"] == _sha("1")
    assert key == (
        f"campaigns/{RUN_ID}/monitor/must-start/{MODE}/"
        f"{json.loads(submission)['submission_body_sha256']}/worker-latches/"
        f"i-0123456789abcdef0/{latch['worker_latch_body_sha256']}.json"
    )


@pytest.mark.parametrize(
    ("sts_account", "identity_type", "match"),
    [
        ("135792468013", "p5.48xlarge", "STS account"),
        (ACCOUNT, "p4d.24xlarge", "instance type"),
    ],
)
def test_worker_rejects_foreign_sts_or_imds_before_latch(
    sts_account: str,
    identity_type: str,
    match: str,
) -> None:
    module = _load_publisher()
    descriptor, submission, descriptor_sha, submission_key = _authorities()
    document = json.loads(
        _identity_document(pending=DEADLINE - timedelta(seconds=10))
    )
    document["instanceType"] = identity_type
    with pytest.raises(ValueError, match=match):
        module.build_authenticated_worker_latch(
            policy_path=POLICY,
            descriptor_bytes=descriptor,
            descriptor_s3_uri=f"s3://{BUCKET}/{DESCRIPTOR_KEY}",
            descriptor_file_sha256=descriptor_sha,
            submission_bytes=submission,
            submission_s3_uri=f"s3://{BUCKET}/{submission_key}",
            submission_body_sha256=json.loads(submission)[
                "submission_body_sha256"
            ],
            managed_mode=MODE,
            sts_identity={
                "Account": sts_account,
                "Arn": (
                    f"arn:aws:sts::{sts_account}:assumed-role/"
                    "keep-glm52-gpu-worker/session"
                ),
            },
            identity_document_bytes=json.dumps(document).encode(),
            entrypoint_observed_at=DEADLINE - timedelta(seconds=1),
        )


def test_worker_acceptance_rejects_late_s3_last_modified() -> None:
    module = _load_publisher()
    descriptor, submission, descriptor_sha, submission_key = _authorities()
    latch, key = module.build_authenticated_worker_latch(
        policy_path=POLICY,
        descriptor_bytes=descriptor,
        descriptor_s3_uri=f"s3://{BUCKET}/{DESCRIPTOR_KEY}",
        descriptor_file_sha256=descriptor_sha,
        submission_bytes=submission,
        submission_s3_uri=f"s3://{BUCKET}/{submission_key}",
        submission_body_sha256=json.loads(submission)["submission_body_sha256"],
        managed_mode=MODE,
        sts_identity={
            "Account": ACCOUNT,
            "Arn": (
                f"arn:aws:sts::{ACCOUNT}:assumed-role/"
                "keep-glm52-gpu-worker/session"
            ),
        },
        identity_document_bytes=_identity_document(
            pending=DEADLINE - timedelta(seconds=10)
        ),
        entrypoint_observed_at=DEADLINE - timedelta(seconds=1),
    )
    accepted = build_timely_start_accepted(
        **{
            field: latch[field]
            for field in (
                "run_id",
                "managed_mode",
                "account_id",
                "region",
                "bucket",
                "descriptor_body_sha256",
                "submission_body_sha256",
                "sky_job_name",
                "must_start_by",
            )
        },
        source_kind="worker-latch",
        source_key=key,
        source_version_id="v1",
        source_etag='"etag"',
        source_body_sha256=canonical_sha256(latch),
        source_last_modified=DEADLINE,
        started_at=latch["entrypoint_observed_at"],
        accepted_at=DEADLINE,
    )
    module.validate_accepted_for_worker(
        policy_path=POLICY,
        accepted=accepted,
        latch=latch,
        latch_key=key,
    )
    forged = dict(accepted)
    forged["source_last_modified"] = (
        DEADLINE + timedelta(microseconds=1)
    ).isoformat().replace("+00:00", "Z")
    body = dict(forged)
    body.pop("accepted_body_sha256")
    forged["accepted_body_sha256"] = canonical_sha256(body)
    with pytest.raises(ValueError, match="LastModified"):
        module.validate_accepted_for_worker(
            policy_path=POLICY,
            accepted=forged,
            latch=latch,
            latch_key=key,
        )


def test_recovery_accepts_only_same_submission_authority() -> None:
    module = _load_publisher()
    descriptor, submission, descriptor_sha, submission_key = _authorities()
    latch, key = module.build_authenticated_worker_latch(
        policy_path=POLICY,
        descriptor_bytes=descriptor,
        descriptor_s3_uri=f"s3://{BUCKET}/{DESCRIPTOR_KEY}",
        descriptor_file_sha256=descriptor_sha,
        submission_bytes=submission,
        submission_s3_uri=f"s3://{BUCKET}/{submission_key}",
        submission_body_sha256=json.loads(submission)["submission_body_sha256"],
        managed_mode=MODE,
        sts_identity={
            "Account": ACCOUNT,
            "Arn": (
                f"arn:aws:sts::{ACCOUNT}:assumed-role/"
                "keep-glm52-gpu-worker/recovery"
            ),
        },
        identity_document_bytes=_identity_document(
            pending=DEADLINE + timedelta(minutes=5)
        ),
        entrypoint_observed_at=DEADLINE + timedelta(minutes=5),
    )
    accepted = build_timely_start_accepted(
        **{
            field: latch[field]
            for field in (
                "run_id",
                "managed_mode",
                "account_id",
                "region",
                "bucket",
                "descriptor_body_sha256",
                "submission_body_sha256",
                "sky_job_name",
                "must_start_by",
            )
        },
        source_kind="worker-latch",
        source_key=key,
        source_version_id=None,
        source_etag='"etag"',
        source_body_sha256=canonical_sha256(latch),
        source_last_modified=DEADLINE,
        started_at=DEADLINE - timedelta(seconds=1),
        accepted_at=DEADLINE,
    )
    module.validate_accepted_for_worker(
        policy_path=POLICY,
        accepted=accepted,
        latch=latch,
        latch_key=None,
        allow_same_submission_recovery=True,
    )
    foreign = dict(latch)
    foreign["submission_body_sha256"] = _sha("f")
    with pytest.raises(ValueError, match="submission"):
        module.validate_accepted_for_worker(
            policy_path=POLICY,
            accepted=accepted,
            latch=foreign,
            latch_key=None,
            allow_same_submission_recovery=True,
        )


def test_worker_derives_numeric_controller_id_only_from_authenticated_binding() -> None:
    module = _load_publisher()
    descriptor, submission, descriptor_sha, submission_key = _authorities()
    submission_value = json.loads(submission)
    latch, key = module.build_authenticated_worker_latch(
        policy_path=POLICY,
        descriptor_bytes=descriptor,
        descriptor_s3_uri=f"s3://{BUCKET}/{DESCRIPTOR_KEY}",
        descriptor_file_sha256=descriptor_sha,
        submission_bytes=submission,
        submission_s3_uri=f"s3://{BUCKET}/{submission_key}",
        submission_body_sha256=submission_value["submission_body_sha256"],
        managed_mode=MODE,
        sts_identity={
            "Account": ACCOUNT,
            "Arn": (
                f"arn:aws:sts::{ACCOUNT}:assumed-role/"
                "keep-glm52-gpu-worker/session"
            ),
        },
        identity_document_bytes=_identity_document(
            pending=DEADLINE - timedelta(seconds=10)
        ),
        entrypoint_observed_at=DEADLINE - timedelta(seconds=2),
    )
    accepted = build_timely_start_accepted(
        **{
            field: latch[field]
            for field in (
                "run_id",
                "managed_mode",
                "account_id",
                "region",
                "bucket",
                "descriptor_body_sha256",
                "submission_body_sha256",
                "sky_job_name",
                "must_start_by",
            )
        },
        source_kind="worker-latch",
        source_key=key,
        source_version_id="version-1",
        source_etag='"etag"',
        source_body_sha256=canonical_sha256(latch),
        source_last_modified=DEADLINE - timedelta(seconds=1),
        started_at=latch["entrypoint_observed_at"],
        accepted_at=DEADLINE,
    )
    binding = build_must_start_job_binding(
        **{
            field: latch[field]
            for field in (
                "run_id",
                "managed_mode",
                "account_id",
                "region",
                "bucket",
                "descriptor_body_sha256",
                "submission_body_sha256",
                "sky_job_name",
                "must_start_by",
            )
        },
        descriptor_key=DESCRIPTOR_KEY,
        descriptor_file_sha256=descriptor_sha,
        submission_key=submission_key,
        submission_submitted_at=submission_value["submitted_at"],
        target_job_id=17,
        workspace="default",
        controller_instance_id="i-0123456789abcdef1",
        controller_instance_type="c6a.xlarge",
        controller_profile_arn=(
            f"arn:aws:iam::{ACCOUNT}:instance-profile/"
            "keep-glm52-skypilot-controller"
        ),
        controller_cluster_name="sky-jobs-controller-12345678",
        observation_body_sha256=_sha("a"),
        bound_at=DEADLINE - timedelta(seconds=1),
    )

    authenticated = module.validate_binding_for_worker(
        policy_path=POLICY,
        binding=binding,
        accepted=accepted,
        latch=latch,
    )

    assert authenticated["target_job_id"] == 17
    foreign = dict(binding)
    foreign["submission_body_sha256"] = _sha("f")
    with pytest.raises(ValueError, match="submission"):
        module.validate_binding_for_worker(
            policy_path=POLICY,
            binding=foreign,
            accepted=accepted,
            latch=latch,
        )
