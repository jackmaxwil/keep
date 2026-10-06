"""Operator-facing SkyPilot campaign script tests."""

from __future__ import annotations

import json
import os
import subprocess
import hashlib
from pathlib import Path

from glm52_enforcement.canonical import canonical_json_bytes
from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_ACCOUNT_ID,
    validate_gpu_spend_approval,
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_s3_artifact_audit import (
    validate_s3_artifact_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "aws/glm52-gpu/scripts"


def _run(script: str, *arguments: str, input_text: str | None = None):
    return subprocess.run(
        [str(SCRIPTS / script), *arguments],
        cwd=ROOT,
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def _run_without_pythonpath(
    script: str,
    *arguments: str,
    input_text: str | None = None,
):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(SCRIPTS / script), *arguments],
        cwd=ROOT,
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
        env=environment,
    )


def _normalize_sky_queue(
    tmp_path: Path,
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> subprocess.CompletedProcess[str]:
    script = SCRIPTS / "normalize_sky_jobs_queue.py"
    assert script.is_file(), "SkyPilot queue normalizer is missing"
    raw_stdout = tmp_path / "queue.stdout"
    raw_stderr = tmp_path / "queue.stderr"
    output = tmp_path / "queue.json"
    raw_stdout.write_text(stdout)
    raw_stderr.write_text(stderr)
    return subprocess.run(
        [
            str(script),
            "--exit-code",
            str(exit_code),
            "--stdout",
            str(raw_stdout),
            "--stderr",
            str(raw_stderr),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_empty_skypilot_jobs_queue_normalizes_to_an_empty_json_list(
    tmp_path: Path,
) -> None:
    result = _normalize_sky_queue(
        tmp_path,
        exit_code=1,
        stdout="",
        stderr=(
            "sky.exceptions.ClusterNotUpError: "
            "No in-progress managed jobs.\n"
        ),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "queue.json").read_text()) == []


def test_approval_builder_writes_authenticated_kon_authority(tmp_path) -> None:
    output = tmp_path / "GPU_SPEND_APPROVAL.json"
    result = _run(
        "build_gpu_spend_approval.py",
        "--ingested-at",
        "2026-07-24T01:05:00Z",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    approval = validate_gpu_spend_approval(json.loads(output.read_text()))
    assert approval["approved_by"] == "Alex Approver"
    assert approval["approved_gpu_hours"] == 24


def test_approval_builder_writes_enforcement_canonical_utf8_json(
    tmp_path: Path,
) -> None:
    """Break caught: non-ASCII approval text is serialized as JSON escapes."""

    output = tmp_path / "GPU_SPEND_APPROVAL.json"
    result = _run(
        "build_gpu_spend_approval.py",
        "--ingested-at",
        "2026-07-24T01:05:00Z",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    raw = output.read_bytes()
    parsed = json.loads(raw)
    approval_body_sha256 = parsed["approval_body_sha256"]
    assert raw == canonical_json_bytes(parsed) + b"\n"
    assert "—".encode() in raw
    assert b"\\u2014" not in raw
    assert validate_gpu_spend_approval(parsed)[
        "approval_body_sha256"
    ] == approval_body_sha256


def test_account_identity_filter_accepts_only_rnd_account() -> None:
    accepted = _run(
        "assert_rnd_aws_account.py",
        input_text=json.dumps(
            {
                "Account": APPROVED_ACCOUNT_ID,
                "Arn": (
                    "arn:aws:sts::246813579024:assumed-role/"
                    "AWSReservedSSO_AdministratorAccess/user"
                ),
            }
        ),
    )
    assert accepted.returncode == 0
    assert accepted.stdout.strip() == APPROVED_ACCOUNT_ID

    refused = _run(
        "assert_rnd_aws_account.py",
        input_text=json.dumps(
            {
                "Account": "135792468013",
                "Arn": "arn:aws:sts::135792468013:assumed-role/wrong/user",
            }
        ),
    )
    assert refused.returncode == 2
    assert "refusing AWS operation" in refused.stderr


def test_account_identity_filter_runs_from_repo_without_pythonpath() -> None:
    accepted = _run_without_pythonpath(
        "assert_rnd_aws_account.py",
        input_text=json.dumps(
            {
                "Account": APPROVED_ACCOUNT_ID,
                "Arn": (
                    "arn:aws:sts::246813579024:assumed-role/"
                    "AWSReservedSSO_AdministratorAccess/user"
                ),
            }
        ),
    )

    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == APPROVED_ACCOUNT_ID

    refused = _run_without_pythonpath(
        "assert_rnd_aws_account.py",
        input_text=json.dumps(
            {
                "Account": "135792468013",
                "Arn": (
                    "arn:aws:sts::135792468013:"
                    "assumed-role/wrong/user"
                ),
            }
        ),
    )

    assert refused.returncode == 2
    assert "refusing AWS operation" in refused.stderr


def test_replacement_uses_exact_prior_termination_when_aws_reports_it() -> None:
    response = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-prior",
                        "State": {"Name": "terminated"},
                        "StateTransitionReason": (
                            "User initiated (2026-07-24 10:15:00 GMT)"
                        ),
                    }
                ]
            }
        ]
    }
    exact = _run(
        "resolve_previous_allocation_end.py",
        "--instance-id",
        "i-prior",
        "--previous-launch",
        "2026-07-24T08:00:00Z",
        "--replacement-launch",
        "2026-07-24T11:00:00Z",
        input_text=json.dumps(response),
    )
    assert exact.returncode == 0
    assert exact.stdout.strip() == "2026-07-24T10:15:00Z"

    conservative = _run(
        "resolve_previous_allocation_end.py",
        "--instance-id",
        "i-prior",
        "--previous-launch",
        "2026-07-24T08:00:00Z",
        "--replacement-launch",
        "2026-07-24T11:00:00Z",
        input_text="{}",
    )
    assert conservative.returncode == 0
    assert conservative.stdout.strip() == "2026-07-24T11:00:00Z"


def test_missing_prior_worker_is_reconciled_from_exact_tagged_ec2_history(
    tmp_path,
) -> None:
    ledger = tmp_path / "GPU_SPEND_LEDGER.jsonl"
    response = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-prior",
                        "InstanceType": "p5.48xlarge",
                        "LaunchTime": "2026-07-24T08:00:00Z",
                        "State": {"Name": "terminated"},
                        "StateTransitionReason": (
                            "User initiated (2026-07-24 10:15:00 GMT)"
                        ),
                        "Tags": [
                            {"Key": "project", "Value": "keep-glm52"},
                            {"Key": "owner", "Value": "jack.mazac"},
                            {"Key": "model", "Value": "glm-5.2"},
                            {
                                "Key": "cost-allocation",
                                "Value": "glm52-sky-campaign",
                            },
                            {
                                "Key": "campaign-run-id",
                                "Value": "glm52-sky-20260723",
                            },
                        ],
                    }
                ]
            }
        ]
    }
    found = _run(
        "find_unrecorded_gpu_allocations.py",
        "--run-id",
        "glm52-sky-20260723",
        "--ledger",
        str(ledger),
        "--current-instance-id",
        "i-replacement",
        "--replacement-launch",
        "2026-07-24T11:00:00Z",
        input_text=json.dumps(response),
    )
    assert found.returncode == 0, found.stderr
    assert found.stdout.strip() == (
        "i-prior\t2026-07-24T08:00:00Z\t2026-07-24T10:15:00Z"
    )


def test_shell_account_guard_requires_nondefault_named_profile() -> None:
    guard = SCRIPTS / "assert_rnd_aws_account.sh"
    source = guard.read_text()
    assert 'AWS_PROFILE:?set AWS_PROFILE' in source
    assert '"$AWS_PROFILE" = "default"' in source
    assert "aws sts get-caller-identity" in source
    assert "--profile \"$AWS_PROFILE\"" in source


def test_sns_guard_requires_jacks_confirmed_subscription_arn() -> None:
    guard = SCRIPTS / "assert_sns_email_confirmed.sh"
    source = guard.read_text()
    assert "assert_rnd_aws_account.sh" in source
    assert "operator@example.com" in source
    assert "list-subscriptions-by-topic" in source
    assert "PendingConfirmation" in source
    assert 'arn.startswith("arn:")' in source


def test_sky_stack_deploy_lets_cloudformation_upload_large_template() -> None:
    source = (SCRIPTS / "deploy_sky_control_plane.sh").read_text()
    assert "aws cloudformation deploy" in source
    assert '--template-file "$TEMPLATE"' in source
    assert '--s3-bucket "$TEMPLATE_UPLOAD_BUCKET"' in source
    assert "validate-template --template-body" not in source


def test_missing_source_restore_is_pinned_hash_checked_and_marker_safe() -> None:
    source = (SCRIPTS / "restore_missing_hf_shards_to_s3.py").read_text()
    assert "assert_rnd_aws_account.sh" in source
    assert "lfs_sha256" in source
    assert "lfs_size" in source
    assert "create_multipart_upload" in source
    assert "abort_multipart_upload" in source
    assert "complete_multipart_upload" in source
    assert source.index("digest.hexdigest()") < source.index(
        "complete_multipart_upload"
    )
    assert 'headers["Range"] = f"bytes={byte_count}-"' in source
    assert "Content-Range" in source
    assert "urllib.error.HTTPError" in source
    assert "Retry-After" in source
    assert "HTTP_RETRY_ATTEMPTS = 6" in source
    assert "429" in source
    assert "IfNoneMatch" not in source
    assert "glm52-source-revision" in source


def test_server_side_checksum_job_is_exact_account_guarded_and_full_object() -> None:
    source = (SCRIPTS / "run_s3_batch_checksum_audit.py").read_text()
    assert "assert_rnd_aws_account.sh" in source
    assert "validate_s3_artifact_inventory" in source
    assert '"ChecksumAlgorithm": "SHA256"' in source
    assert '"ChecksumType": "FULL_OBJECT"' in source
    assert "ConfirmationRequired=False" in source
    assert "batchoperations.s3.amazonaws.com" not in source
    assert "campaign-audits/" in source
    assert '["Bucket", "Key", "VersionId"]' in source
    assert 'item["version_id"]' in source


def test_checksum_report_collector_authenticates_complete_successful_job() -> None:
    source = (SCRIPTS / "collect_s3_batch_checksum_report.py").read_text()
    assert "assert_rnd_aws_account.sh" in source
    assert "describe_job" in source
    assert 'status != "Complete"' in source
    assert 'task_status != "succeeded"' in source
    assert "if len(row) != 7:" in source
    assert '"checksumType"' in source
    assert "FULL_OBJECT" in source
    assert "authority_body_sha256" in source
    assert "expected_versions" in source
    assert "row_version_id" in source


def test_artifact_audit_binds_batch_authority_to_exact_shared_inventory() -> None:
    source = (SCRIPTS / "audit_s3_campaign_artifacts.py").read_text()
    assert "shared_inventory = AUDIT.build_s3_artifact_inventory" in source
    assert 'item["run_scope"] == "shared"' in source
    assert 'authority["inventory_body_sha256"]' in source
    assert "set(batch_checksums) != shared_keys" in source


def test_stale_source_multipart_cleanup_requires_published_checksum_authority() -> None:
    source = (SCRIPTS / "cleanup_stale_source_multipart_uploads.py").read_text()
    assert "assert_rnd_aws_account.sh" in source
    assert "validate_s3_artifact_inventory" in source
    assert "glm52_s3_batch_checksum_authority_v1" in source
    assert "head_object" in source
    assert "abort_multipart_upload" in source
    assert "--initiated-before" in source
    assert "--apply" in source


def test_production_inventory_is_derived_from_pinned_artifact_authorities() -> None:
    source = (SCRIPTS / "build_production_s3_inventory.py").read_text()
    assert "assert_rnd_aws_account.sh" in source
    assert "lfs_sha256" in source
    assert "non-vq-manifest.json" in source
    assert "conversion-manifest.json" in source
    assert "ROUTED_BASELINE_READY.json" in source
    assert "validation/" in source
    assert "--qualification-cache-prefix" in source
    assert "build_s3_artifact_inventory" in source
    assert "head_object" in source


def test_sky_job_status_publisher_is_schema_backed_and_aws_free() -> None:
    source = (SCRIPTS / "publish_sky_job_status.py").read_text()
    assert "build_skypilot_job_status" in source
    assert "validate_sky_campaign_descriptor" in source
    assert "aws " not in source


def test_descriptor_builder_hashes_exact_tar_and_approval_files(tmp_path) -> None:
    approval_path = tmp_path / "GPU_SPEND_APPROVAL.json"
    assert (
        _run(
            "build_gpu_spend_approval.py",
            "--ingested-at",
            "2026-07-24T01:05:00Z",
            "--output",
            str(approval_path),
        ).returncode
        == 0
    )
    repo_tar = tmp_path / "repo.tar.gz"
    repo_tar.write_bytes(b"immutable-repository-tar")
    output = tmp_path / "campaign-descriptor-v2.json"
    result = _run(
        "build_sky_campaign_descriptor.py",
        "--run-id",
        "glm52-sky-20260723",
        "--must-start-by",
        "2026-07-24T13:05:00Z",
        "--controller-identity",
        "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller",
        "--worker-identity",
        "arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        "--vpc-name",
        "keep-glm52-vpc",
        "--image-id",
        "ami-0123456789abcdef0",
        "--bucket",
        "keep-glm52-us-west-2-246813579024",
        "--repo-tar",
        str(repo_tar),
        "--repo-tar-key",
        "campaigns/glm52-sky-20260723/repository/repo.tar.gz",
        "--campaign-descriptor-key",
        "campaigns/glm52-sky-20260723/submissions/first/campaign-descriptor-v2.json",
        "--approval",
        str(approval_path),
        "--approval-key",
        "campaigns/glm52-sky-20260723/authorities/GPU_SPEND_APPROVAL.json",
        "--source-snapshot-prefix",
        "source-snapshot/",
        "--source-snapshot-sha256",
        "2" * 64,
        "--non-vq-prefix",
        "non-vq-package/",
        "--non-vq-package-sha256",
        "3" * 64,
        "--teich-pack-key",
        "teich-pack/pack.json",
        "--teich-pack-sha256",
        "4" * 64,
        "--frozen-prompt-pack-key",
        "quality/frozen.json",
        "--frozen-prompt-pack-sha256",
        "5" * 64,
        "--training-baseline-prefix",
        "training-baseline/",
        "--training-baseline-sha256",
        "6" * 64,
        "--training-config-key",
        "campaigns/glm52-sky-20260723/authorities/training.json",
        "--training-config-sha256",
        "7" * 64,
        "--artifact-inventory-key",
        (
            "campaigns/glm52-sky-20260723/inventories/"
            + "artifact-inventory-"
            + "8" * 64
            + ".json"
        ),
        "--artifact-inventory-sha256",
        "8" * 64,
        "--qualification-cache-prefix",
        "qualification-cache/seeds/glm52-sky-20260723/" + "9" * 64 + "/",
        "--qualification-cache-manifest-sha256",
        "9" * 64,
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    descriptor = validate_sky_campaign_descriptor(json.loads(output.read_text()))
    assert descriptor["repo_tar_sha256"] == hashlib.sha256(
        repo_tar.read_bytes()
    ).hexdigest()
    assert descriptor["approval_sha256"] == hashlib.sha256(
        approval_path.read_bytes()
    ).hexdigest()
    assert descriptor["jobs_bucket"] == descriptor["bucket"]


def test_artifact_inventory_builder_is_atomic_strict_and_key_sorted(
    tmp_path,
) -> None:
    objects = [
        {
            "key": "shared/prompt.json",
            "size": 12,
            "sha256": "b" * 64,
            "kind": "prompt_pack",
            "safetensors": False,
            "run_scope": "shared",
        },
        {
            "key": "campaigns/glm52-sky-20260723/repo.tar.gz",
            "size": 20,
            "sha256": "a" * 64,
            "kind": "repository_tar",
            "safetensors": False,
            "run_scope": "glm52-sky-20260723",
        },
    ]
    source = tmp_path / "objects.json"
    source.write_text(json.dumps(objects))
    output = tmp_path / "artifact-inventory-v1.json"
    result = _run(
        "build_s3_artifact_inventory.py",
        "--run-id",
        "glm52-sky-20260723",
        "--bucket",
        "keep-glm52-us-west-2-246813579024",
        "--objects-json",
        str(source),
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    value = validate_s3_artifact_inventory(json.loads(output.read_text()))
    assert [item["key"] for item in value["objects"]] == sorted(
        item["key"] for item in objects
    )
    assert not list(tmp_path.glob(".*.tmp"))


def test_gpu_spend_cli_materializes_deadline_and_preserves_replacement_budget(
    tmp_path,
) -> None:
    approval_path = tmp_path / "GPU_SPEND_APPROVAL.json"
    assert (
        _run(
            "build_gpu_spend_approval.py",
            "--ingested-at",
            "2026-07-24T01:05:00Z",
            "--output",
            str(approval_path),
        ).returncode
        == 0
    )
    descriptor = {
        "run_id": "glm52-sky-20260723",
        "approval_sha256": hashlib.sha256(approval_path.read_bytes()).hexdigest(),
        "approved_gpu_runtime_seconds": 86_400,
        "approved_gpu_cost_usd": 1_320.96,
        "max_hourly_cost_usd": 55.04,
    }
    descriptor_path = tmp_path / "descriptor-fragment.json"
    descriptor_path.write_text(json.dumps(descriptor))
    root = tmp_path / "campaign-root"

    started = _run(
        "manage_gpu_spend.py",
        "start",
        "--descriptor",
        str(descriptor_path),
        "--approval",
        str(approval_path),
        "--root",
        str(root),
        "--job-id",
        "sky-job-1",
        "--instance-id",
        "i-first",
        "--launched-at",
        "2026-07-24T08:00:00Z",
        "--observed-at",
        "2026-07-24T08:00:00Z",
    )
    assert started.returncode == 0, started.stderr
    environment = (root / "runtime/campaign.env").read_text()
    assert "GLM52_EXECUTION_DEADLINE=2026-07-25T08:00:00Z" in environment
    allocation = json.loads(
        (root / "runtime/GPU_RUNTIME_ALLOCATION.json").read_text()
    )
    assert allocation["remaining_gpu_seconds"] == 86_400

    ended = _run(
        "manage_gpu_spend.py",
        "end",
        "--descriptor",
        str(descriptor_path),
        "--approval",
        str(approval_path),
        "--root",
        str(root),
        "--instance-id",
        "i-first",
        "--ended-at",
        "2026-07-24T10:00:00Z",
    )
    assert ended.returncode == 0, ended.stderr

    replacement = _run(
        "manage_gpu_spend.py",
        "start",
        "--descriptor",
        str(descriptor_path),
        "--approval",
        str(approval_path),
        "--root",
        str(root),
        "--job-id",
        "sky-job-1",
        "--instance-id",
        "i-replacement",
        "--launched-at",
        "2026-07-24T11:00:00Z",
        "--observed-at",
        "2026-07-24T11:00:00Z",
    )
    assert replacement.returncode == 0, replacement.stderr
    replacement_allocation = json.loads(
        (root / "runtime/GPU_RUNTIME_ALLOCATION.json").read_text()
    )
    assert replacement_allocation["remaining_gpu_seconds"] == 79_200
    assert replacement_allocation["execution_deadline"] == "2026-07-25T09:00:00Z"
    records = sorted((root / "runtime/spend-ledger/records").glob("*.json"))
    assert len(records) == 3
    assert records[0].name.startswith("000000-allocation_started-")
    assert records[1].name.startswith("000001-allocation_ended-")
    assert records[2].name.startswith("000002-allocation_started-")
    latest = json.loads(
        (root / "runtime/spend-ledger/latest.json").read_text()
    )
    assert latest["record_count"] == 3
    assert latest["record_keys"] == [record.name for record in records]
    assert latest["latest_record_sha256"] == replacement_allocation[
        "gpu_spend_record_sha256"
    ]
    reconstructed = tmp_path / "reconstructed-spend-ledger.jsonl"
    restored = _run(
        "restore_gpu_spend_ledger.py",
        "--latest",
        str(root / "runtime/spend-ledger/latest.json"),
        "--records-dir",
        str(root / "runtime/spend-ledger/records"),
        "--output",
        str(reconstructed),
    )
    assert restored.returncode == 0, restored.stderr
    assert reconstructed.read_bytes() == (
        root / "runtime/GPU_SPEND_LEDGER.jsonl"
    ).read_bytes()


def test_gpu_spend_refusal_does_not_append_an_unusable_allocation(tmp_path) -> None:
    approval_path = tmp_path / "GPU_SPEND_APPROVAL.json"
    assert (
        _run(
            "build_gpu_spend_approval.py",
            "--ingested-at",
            "2026-07-24T01:05:00Z",
            "--output",
            str(approval_path),
        ).returncode
        == 0
    )
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "run_id": "glm52-sky-budget-refusal",
                "approval_sha256": hashlib.sha256(
                    approval_path.read_bytes()
                ).hexdigest(),
                "approved_gpu_runtime_seconds": 86_400,
                "approved_gpu_cost_usd": 1_320.96,
                "max_hourly_cost_usd": 55.04,
            }
        )
    )
    root = tmp_path / "root"
    common = (
        "--descriptor",
        str(descriptor_path),
        "--approval",
        str(approval_path),
        "--root",
        str(root),
    )
    assert _run(
        "manage_gpu_spend.py",
        "start",
        *common,
        "--job-id",
        "job-1",
        "--instance-id",
        "i-first",
        "--launched-at",
        "2026-07-24T00:00:00Z",
        "--observed-at",
        "2026-07-24T00:00:00Z",
    ).returncode == 0
    assert _run(
        "manage_gpu_spend.py",
        "end",
        *common,
        "--instance-id",
        "i-first",
        "--ended-at",
        "2026-07-24T23:30:00Z",
    ).returncode == 0
    ledger = root / "runtime/GPU_SPEND_LEDGER.jsonl"
    before = ledger.read_bytes()
    refused = _run(
        "manage_gpu_spend.py",
        "start",
        *common,
        "--job-id",
        "job-1",
        "--instance-id",
        "i-second",
        "--launched-at",
        "2026-07-25T00:00:00Z",
        "--observed-at",
        "2026-07-25T00:00:00Z",
    )
    assert refused.returncode == 64
    assert "one hour or less remaining" in refused.stderr
    assert ledger.read_bytes() == before


def test_gpu_spend_s3_sync_publishes_immutable_records_before_latest_marker() -> None:
    source = (SCRIPTS / "sync_gpu_spend_ledger.sh").read_text()
    assert "--if-none-match '*'" in source
    assert "GPU_SPEND_LEDGER.jsonl" in source
    assert "GPU_SPEND_LEDGER_LATEST.json" in source
    assert source.index("--if-none-match '*'") < source.index(
        "GPU_SPEND_LEDGER_LATEST.json"
    )
    restore = (SCRIPTS / "restore_gpu_spend_ledger.sh").read_text()
    assert "GPU_SPEND_LEDGER_LATEST.json" in restore
    assert "restore_gpu_spend_ledger.py" in restore
