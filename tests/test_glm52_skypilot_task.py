"""SkyPilot task, configuration, and managed wrapper policy tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)


ROOT = Path(__file__).resolve().parents[1]
SKY = ROOT / "aws/glm52-gpu/skypilot"
SCRIPTS = ROOT / "aws/glm52-gpu/scripts"
WORKER_V2_ROOT = "/tmp/glm52-worker-start-v2"
WORKER_V2_SHA256 = {
    "publish_worker_start_v2.py": (
        "4d1d47aef6dd211c20b52f05ede9bd6d32c35433e305f2b1bd56b48c83b9051e"
    ),
    "glm52_sky_campaign_native.py": (
        "f023eaeddf8fac73fa6546e3c4a0b60dd32e5266f06e1519fe21e0444d445d03"
    ),
    "glm52_sky_must_start_native.py": (
        "e910d3d03f7b30b7b9b668e214b61ecc7cbd641a53dedabd620b8e14573d33f3"
    ),
    "glm52_sky_campaign.py": (
        "77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94"
    ),
    "glm52_sky_must_start.py": (
        "478ac3a16ce8a77a4e844c1e0a3dca99454d50e7233fd7cd67dfa184b486860a"
    ),
    "glm52_sky_must_start_dynamic.py": (
        "527019a41b54f810e9f98734c2ca99acc5a7b370344adf48c80f46f925396e18"
    ),
    "glm52_sky_worker_must_start_v2.py": (
        "8dfad6682d8afc9774dc31385cfc9ecc8c8e2f80d9a535e3c8d4cf6cf0fa97df"
    ),
    "sky_worker_start_v2_coordinator.py": (
        "efa7c48ad259e10c6ce0e4cb1db9ff0a6e6a9fa203274329a7c2eb5203366406"
    ),
}
WORKER_V2_MOUNTS = {
    f"{WORKER_V2_ROOT}/publish_worker_start_v2.py": (
        "aws/glm52-gpu/skypilot/publish_worker_start_v2.py"
    ),
    f"{WORKER_V2_ROOT}/glm52_sky_campaign_native.py": (
        "src/glm52_enforcement/glm52_sky_campaign.py"
    ),
    f"{WORKER_V2_ROOT}/glm52_sky_must_start_native.py": (
        "src/glm52_enforcement/glm52_sky_must_start.py"
    ),
    f"{WORKER_V2_ROOT}/glm52_sky_campaign.py": (
        "src/mlx_vq/quality/glm52_sky_campaign.py"
    ),
    f"{WORKER_V2_ROOT}/glm52_sky_must_start.py": (
        "src/mlx_vq/quality/glm52_sky_must_start.py"
    ),
    f"{WORKER_V2_ROOT}/glm52_sky_must_start_dynamic.py": (
        "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py"
    ),
    f"{WORKER_V2_ROOT}/glm52_sky_worker_must_start_v2.py": (
        "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py"
    ),
    f"{WORKER_V2_ROOT}/sky_worker_start_v2_coordinator.py": (
        "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py"
    ),
    f"{WORKER_V2_ROOT}/run_h100_qualification.sh": (
        "aws/glm52-gpu/skypilot/run_h100_qualification.sh"
    ),
    f"{WORKER_V2_ROOT}/glm52_h100_qualification.py": (
        "src/mlx_vq/quality/glm52_h100_qualification.py"
    ),
    f"{WORKER_V2_ROOT}/glm52_h100_qualification_native.py": (
        "src/glm52_enforcement/glm52_h100_qualification.py"
    ),
}
WORKER_V2_FLAGS = {
    "--publisher-file-sha256": WORKER_V2_SHA256["publish_worker_start_v2.py"],
    "--campaign-policy-file-sha256": WORKER_V2_SHA256["glm52_sky_campaign.py"],
    "--campaign-policy-native-file-sha256": (
        WORKER_V2_SHA256["glm52_sky_campaign_native.py"]
    ),
    "--must-start-policy-file-sha256": (WORKER_V2_SHA256["glm52_sky_must_start.py"]),
    "--must-start-policy-native-file-sha256": (
        WORKER_V2_SHA256["glm52_sky_must_start_native.py"]
    ),
    "--dynamic-policy-file-sha256": (
        WORKER_V2_SHA256["glm52_sky_must_start_dynamic.py"]
    ),
    "--worker-policy-file-sha256": (
        WORKER_V2_SHA256["glm52_sky_worker_must_start_v2.py"]
    ),
    "--coordinator-file-sha256": (
        WORKER_V2_SHA256["sky_worker_start_v2_coordinator.py"]
    ),
}
FORBIDDEN_WORKER_PATH_TOKENS = (
    "publish_must_start_latch.py",
    "GLM52_IMMUTABLE_SUBMISSION_",
    "IMMUTABLE_SUBMISSION.json",
    "TIMELY_START_ACCEPTED.json",
    "TIMELY_START_LATCH.json",
    "JOB_BINDING.json",
    "verify-accepted",
)
FINAL_WORKER_PATH_SHA256 = {
    "glm52-campaign.yaml": (
        "cc0a56a34684d297e55827cf017802f4e8b16bd8dd0bb18fa1e1c0a197d787e5"
    ),
    "bootstrap_campaign.sh": (
        "8073ec6e55184073eece202d1726cbb1e7fe83c6f95abee927114b713f4366fc"
    ),
    "run_managed_campaign.sh": (
        "ac2400441f2b6ecac6bf6ad9cc46fba5abb0386e1b2cf6046527f8c3c3ba9a9c"
    ),
}


def test_managed_job_is_one_aws_on_demand_p5_with_bounded_app_retries() -> None:
    task = yaml.safe_load((SKY / "glm52-campaign.yaml").read_text())
    resources = task["resources"]
    assert resources["infra"] == "aws/us-west-2"
    assert resources["instance_type"] == "p5.48xlarge"
    assert resources["use_spot"] is False
    assert resources["max_hourly_cost"] == 55.04
    assert resources["job_recovery"]["strategy"] == "FAILOVER"
    assert resources["job_recovery"]["max_restarts_on_errors"] == 0
    assert "recover_on_exit_codes" not in resources["job_recovery"]
    assert task["num_nodes"] == 1
    assert task["api_server_access"] is False
    assert "workdir" not in task
    assert task["file_mounts"] == WORKER_V2_MOUNTS
    source = (SKY / "glm52-campaign.yaml").read_text().lower()
    assert "capacity-block" not in source
    assert "capacity_reservation" not in source
    assert "137" not in source


def test_final_worker_path_identities_are_frozen_for_next_slice() -> None:
    assert (
        hashlib.sha256((SKY / "publish_worker_start_v2.py").read_bytes()).hexdigest()
        == WORKER_V2_SHA256["publish_worker_start_v2.py"]
    )
    assert {
        name: hashlib.sha256((SKY / name).read_bytes()).hexdigest()
        for name in FINAL_WORKER_PATH_SHA256
    } == FINAL_WORKER_PATH_SHA256


def test_sky_config_is_aws_only_ssm_and_uses_a_four_vcpu_controller() -> None:
    config = yaml.safe_load((SKY / "skypilot-config.yaml.in").read_text())
    assert config["allowed_clouds"] == ["aws"]
    assert config["aws"]["use_ssm"] is True
    assert config["aws"]["vpc_name"] == "keep-glm52-vpc"
    assert config["aws"]["security_group_name"] == "keep-glm52-gpu-sg"
    assert config["aws"]["labels"] == {
        "project": "keep-glm52",
        "owner": "jack.mazac",
        "model": "glm-5.2",
        "campaign-run-id": "__RUN_ID__",
        "cost-allocation": "glm52-sky-campaign",
    }
    identities = config["aws"]["remote_identity"]
    assert identities[-1] == {"*": "keep-glm52-gpu-worker"}
    assert (
        config["jobs"]["controller"]["resources"]["instance_type"] == "c6a.xlarge"
    )
    assert config["jobs"]["controller"]["resources"]["infra"] == "aws/us-west-2"
    assert config["jobs"]["bucket"] == "s3://__JOBS_BUCKET__/"


def test_submission_and_worker_wrappers_are_guarded_and_capacity_block_free() -> None:
    submit = (SCRIPTS / "submit_sky_campaign.sh").read_text()
    managed = (SKY / "run_managed_campaign.sh").read_text()
    bootstrap = (SKY / "bootstrap_campaign.sh").read_text()
    service = (SKY / "keep-glm52-campaign.service").read_text()
    spike = (SCRIPTS / "run_spike.sh").read_text()
    env_setup = (SCRIPTS / "gpu_env_setup.sh").read_text()

    assert "assert_rnd_aws_account.sh" in submit
    assert "assert_sns_email_confirmed.sh" in submit
    assert '"$SKY_BIN" jobs launch' in submit
    assert '--image-id "$IMAGE_ID"' in submit
    assert "--dry-run" in submit
    assert "validate_skypilot_control_plane.py" in submit
    assert "COMMAND+=(--dryrun)" not in submit
    assert "0.13.0" in submit
    assert "$REPO/.venv/bin/python" not in submit
    assert '$(dirname -- "$SKY_BIN")/python' in submit
    assert "ARTIFACT_INVENTORY" in submit
    assert "ARTIFACT_AUDIT" in submit
    assert "SKYPILOT_SUBMITTED.json" in submit
    assert "--if-none-match" in submit
    assert "jobs queue" in submit
    assert "publish_sky_job_status.py" in submit
    assert "use_spot" not in submit
    assert "capacity" not in submit.lower()
    assert "GLM52_SKY_SUBMISSION_INTENT_S3_URI" in submit
    assert "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256" in submit
    assert "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256" in submit
    assert "GLM52_IMMUTABLE_SUBMISSION_S3_URI" not in submit
    assert "GLM52_IMMUTABLE_SUBMISSION_BODY_SHA256" not in submit
    assert "GLM52_IMMUTABLE_SUBMISSION_FILE_SHA256" not in submit

    task_source = (SKY / "glm52-campaign.yaml").read_text()
    publish = task_source.index("publish_worker_start_v2.py publish-latch-v2")
    replay = task_source.index("publish_worker_start_v2.py replay-accepted-v2")
    repo_download = task_source.index('aws s3 cp "$REPO_URI"')
    assert publish < replay < repo_download
    assert "WORKER_START_ADMISSION_RECEIPT.json" in task_source
    assert "GLM52_SKY_JOB_NAME" in task_source
    assert "GLM52_SKY_JOB_ID" not in task_source
    assert "GLM52_SKY_SUBMISSION_INTENT_S3_URI" in task_source
    assert "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256" in task_source
    assert "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256" in task_source
    assert "cmp -s" in task_source

    assert "manage_gpu_spend.py" in managed
    assert "publish_worker_start_v2.py" in managed
    verify = managed.index("verify-receipt-v2")
    spend_start = managed.index("manage_gpu_spend.py\" start")
    assert verify < spend_start
    assert "WORKER_START_ADMISSION_RECEIPT.json" in managed
    assert "GLM52_SKY_CONTROLLER_JOB_ID" in managed
    assert "GLM52_SKY_JOB_NAME" in managed
    assert "GLM52_SKY_JOB_ID" not in managed
    assert "MAX_TRANSIENT_RESTARTS=2" in managed
    assert "run_direct_child_with_retries" in managed
    assert "wait_for_direct_child" in managed
    assert 'if ! kill -0 "$child_pid"' in managed
    assert "wait_for_campaign_service" in managed
    assert "sleep 15 || true" in managed
    assert '[ "$status" -eq 75 ]' in managed
    assert "exit 70" in managed
    assert "systemctl" in managed
    assert "resolve_previous_allocation_end.py" in managed
    assert "sync_gpu_spend_ledger.sh" in managed
    assert "restore_gpu_spend_ledger.sh" in managed
    assert "find_unrecorded_gpu_allocations.py" in managed
    assert "reconcile" in managed
    assert "StateTransitionReason" in (
        SCRIPTS / "resolve_previous_allocation_end.py"
    ).read_text()
    assert "AUTHORIZATION_EXHAUSTED" in managed
    assert "PHASE_DEADLINE_EXHAUSTED" in managed
    assert "QUALIFICATION_MAX_SECONDS=14400" in managed
    assert "CAMPAIGN_DRAINED.json" in managed
    assert "H100_RESUME_READY.json" in managed
    assert "GLM52_MANAGED_MODE" in managed
    assert "run_h100_qualification.sh" in managed
    assert "publish_sky_job_status.py" in managed
    assert "RECOVERING" in managed
    campaign = (ROOT / "benchmarks/run_glm52_campaign.py").read_text()
    assert "_run_managed_child" in campaign
    assert "self._children.add(child)" in campaign

    assert "GPU_SPEND_APPROVAL.json" in bootstrap
    assert "WORKER_START_ADMISSION_RECEIPT.json" in bootstrap
    assert "verify-receipt-v2" in bootstrap
    assert "descriptor_body_sha256" in bootstrap
    assert "prepare_nvme_storage.sh" in bootstrap
    assert "gpu_env_setup.sh" in bootstrap
    assert "GLM_MLX_WIRED_LIMIT_GB" in bootstrap
    assert "--execution-deadline" in spike
    assert "--capacity-block-end" in spike
    assert "--cache-seed" in submit
    seed = (SKY / "run_qualification_cache_seed.sh").read_text()
    assert "produce_glm52_teich_teacher_cache.py" in seed
    assert "finalize_glm52_teich_teacher_cache.py" in seed
    assert "glm52-teacher-signal-cache-v3-manifest.json" in seed
    assert "QUALIFICATION_CACHE_SEED_READY.json" in seed
    assert "qualification-cache/seeds/" in seed
    assert "--execution-deadline" in seed
    assert "sync_checkpoint_tree.sh" in seed
    assert "PRODUCER_PID" in seed
    assert "wait_for_producer" in seed
    assert 'kill -TERM "$PRODUCER_PID"' in seed
    assert "put_immutable" in seed
    assert "--if-none-match" in seed
    assert '--checksum-algorithm SHA256' in seed
    assert 'aws s3 sync "$OUTPUT/teacher_signal/"' not in seed
    assert "cache-seed" in env_setup
    assert "package_set_sha256" in env_setup
    assert (
        "jq -er .package_set_sha256 "
        '"$ROOT/non-vq-package/non-vq-manifest.json"'
    ) in env_setup
    qualification = (SKY / "run_h100_qualification.sh").read_text()
    assert "$ROOT/qualification-cache/prompt-pack.json" in qualification
    finalizer = (ROOT / "benchmarks/finalize_glm52_teich_teacher_cache.py").read_text()
    assert 'parser.add_argument("--hidden-size", type=int, default=6144)' in finalizer

    assert "Restart=no" in service
    assert "Environment=GLM52_ENV_PREPARED=1" in service
    assert "EnvironmentFile=/mnt/nvme/glm52-campaign/runtime/campaign.env" in service
    assert (
        "ExecStart=/opt/keep-campaign/repo/aws/glm52-gpu/scripts/run_campaign.sh"
        in service
    )
    for source in (managed, bootstrap, service):
        assert "capacity-block" not in source.lower()
        assert "purchase-capacity" not in source.lower()


def _assert_worker_v2_source_arguments(source: str, *, root: str) -> None:
    for flag, digest in WORKER_V2_FLAGS.items():
        assert f"{flag} {digest}" in source
    if root == WORKER_V2_ROOT:
        expected_paths = {
            "--campaign-policy": f"{root}/glm52_sky_campaign.py",
            "--campaign-policy-native": f"{root}/glm52_sky_campaign_native.py",
            "--must-start-policy": f"{root}/glm52_sky_must_start.py",
            "--must-start-policy-native": f"{root}/glm52_sky_must_start_native.py",
            "--dynamic-policy": f"{root}/glm52_sky_must_start_dynamic.py",
            "--worker-policy": f"{root}/glm52_sky_worker_must_start_v2.py",
            "--coordinator": f"{root}/sky_worker_start_v2_coordinator.py",
        }
    else:
        assert root == "$REPO"
        expected_paths = {
            "--campaign-policy": '"$REPO/src/mlx_vq/quality/glm52_sky_campaign.py"',
            "--campaign-policy-native": (
                '"$REPO/src/glm52_enforcement/glm52_sky_campaign.py"'
            ),
            "--must-start-policy": (
                '"$REPO/src/mlx_vq/quality/glm52_sky_must_start.py"'
            ),
            "--must-start-policy-native": (
                '"$REPO/src/glm52_enforcement/glm52_sky_must_start.py"'
            ),
            "--dynamic-policy": (
                '"$REPO/src/mlx_vq/quality/glm52_sky_must_start_dynamic.py"'
            ),
            "--worker-policy": (
                '"$REPO/src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py"'
            ),
            "--coordinator": (
                '"$REPO/aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py"'
            ),
        }
    for flag, path in expected_paths.items():
        assert f"{flag} {path}" in source


def test_worker_v2_task_has_exact_inputs_order_and_source_closure() -> None:
    task = yaml.safe_load((SKY / "glm52-campaign.yaml").read_text())
    assert task["file_mounts"] == WORKER_V2_MOUNTS
    assert task["envs"] == {
        "GLM52_CAMPAIGN_DESCRIPTOR_S3_URI": "REQUIRED_AT_SUBMISSION",
        "GLM52_CAMPAIGN_DESCRIPTOR_SHA256": "REQUIRED_AT_SUBMISSION",
        "GLM52_GPU_SPEND_APPROVAL_S3_URI": "REQUIRED_AT_SUBMISSION",
        "GLM52_GPU_SPEND_APPROVAL_SHA256": "REQUIRED_AT_SUBMISSION",
        "GLM52_SKY_JOB_NAME": "REQUIRED_AT_SUBMISSION",
        "GLM52_MANAGED_MODE": "qualification",
        "GLM52_SKY_SUBMISSION_INTENT_S3_URI": "REQUIRED_AT_SUBMISSION",
        "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256": "REQUIRED_AT_SUBMISSION",
        "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256": "REQUIRED_AT_SUBMISSION",
    }
    setup = task["setup"]
    assert isinstance(setup, str)
    publish_at = setup.index("publish_worker_start_v2.py publish-latch-v2")
    replay_at = setup.index("publish_worker_start_v2.py replay-accepted-v2")
    approval_at = setup.index('aws s3 cp "$GLM52_GPU_SPEND_APPROVAL_S3_URI"')
    repo_at = setup.index('aws s3 cp "$REPO_URI"')
    extract_at = setup.index("tar -xzf /tmp/keep-campaign.tar.gz")
    compare_at = setup.index(f"cmp -s {WORKER_V2_ROOT}/publish_worker_start_v2.py")
    install_at = setup.index(
        f"install -m 0644 {WORKER_V2_ROOT}/campaign.json /etc/keep-glm52/campaign.json"
    )
    bootstrap_at = setup.index(
        "/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/bootstrap_campaign.sh"
    )
    assert (
        publish_at
        < replay_at
        < approval_at
        < repo_at
        < extract_at
        < compare_at
        < install_at
        < bootstrap_at
    )
    pre_admission = setup[:publish_at]
    assert pre_admission.count("aws s3 cp") == 2
    assert "$GLM52_CAMPAIGN_DESCRIPTOR_S3_URI" in pre_admission
    assert "$GLM52_SKY_SUBMISSION_INTENT_S3_URI" in pre_admission
    assert "$GLM52_GPU_SPEND_APPROVAL_S3_URI" not in pre_admission
    assert "$REPO_URI" not in pre_admission
    publish_block = setup[publish_at:replay_at]
    replay_block = setup[replay_at:approval_at]
    _assert_worker_v2_source_arguments(
        publish_block,
        root=WORKER_V2_ROOT,
    )
    _assert_worker_v2_source_arguments(
        replay_block,
        root=WORKER_V2_ROOT,
    )
    assert '--wait-timeout-seconds "$WAIT_TIMEOUT_SECONDS"' in replay_block
    assert "--poll-interval-seconds 5" in replay_block
    assert "must_start_by" in setup[:replay_at]
    assert "min(remaining, 600)" in setup[:replay_at]


def test_task_compares_eight_mounted_sources_and_installs_exact_admission_set() -> None:
    setup = yaml.safe_load((SKY / "glm52-campaign.yaml").read_text())["setup"]
    extracted = {
        "publish_worker_start_v2.py": (
            "/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/publish_worker_start_v2.py"
        ),
        "glm52_sky_campaign_native.py": (
            "/opt/keep-campaign/repo/src/glm52_enforcement/glm52_sky_campaign.py"
        ),
        "glm52_sky_must_start_native.py": (
            "/opt/keep-campaign/repo/src/glm52_enforcement/glm52_sky_must_start.py"
        ),
        "glm52_sky_campaign.py": (
            "/opt/keep-campaign/repo/src/mlx_vq/quality/glm52_sky_campaign.py"
        ),
        "glm52_sky_must_start.py": (
            "/opt/keep-campaign/repo/src/mlx_vq/quality/glm52_sky_must_start.py"
        ),
        "glm52_sky_must_start_dynamic.py": (
            "/opt/keep-campaign/repo/src/mlx_vq/quality/glm52_sky_must_start_dynamic.py"
        ),
        "glm52_sky_worker_must_start_v2.py": (
            "/opt/keep-campaign/repo/src/mlx_vq/quality/"
            "glm52_sky_worker_must_start_v2.py"
        ),
        "sky_worker_start_v2_coordinator.py": (
            "/opt/keep-campaign/repo/aws/glm52-gpu/lambda/"
            "sky_worker_start_v2_coordinator.py"
        ),
    }
    assert setup.count("cmp -s") == 8
    for name, extracted_path in extracted.items():
        assert f"cmp -s {WORKER_V2_ROOT}/{name} \\\n    {extracted_path}" in setup
    installed = {
        "campaign.json",
        "GPU_SPEND_APPROVAL.json",
        "SKYPILOT_SUBMISSION_INTENT.json",
        "WORKER_START_LATCH.json",
        "WORKER_START_LATCH_TRANSPORT.json",
        "WORKER_START_ACCEPTED.json",
        "WORKER_START_ADMISSION_RECEIPT.json",
    }
    for name in installed:
        assert f"/etc/keep-glm52/{name}" in setup
    for name in (
        "campaign.json",
        "GPU_SPEND_APPROVAL.json",
        "SKYPILOT_SUBMISSION_INTENT.json",
    ):
        assert f"install -m 0644 {WORKER_V2_ROOT}/{name}" in setup
    for name in (
        "WORKER_START_LATCH.json",
        "WORKER_START_LATCH_TRANSPORT.json",
        "WORKER_START_ACCEPTED.json",
        "WORKER_START_ADMISSION_RECEIPT.json",
    ):
        assert f"install -m 0600 {WORKER_V2_ROOT}/{name}" in setup


def test_bootstrap_and_managed_verification_precede_gpu_or_spend_boundaries() -> None:
    bootstrap = (SKY / "bootstrap_campaign.sh").read_text()
    managed = (SKY / "run_managed_campaign.sh").read_text()
    bootstrap_verify = bootstrap.index("verify-receipt-v2")
    _assert_worker_v2_source_arguments(
        bootstrap[bootstrap_verify:],
        root="$REPO",
    )
    for boundary in (
        "prepare_nvme_storage.sh",
        "gpu_env_setup.sh",
    ):
        assert bootstrap_verify < bootstrap.index(boundary)
    managed_verify = managed.index("verify-receipt-v2")
    _assert_worker_v2_source_arguments(
        managed[managed_verify:],
        root="$REPO",
    )
    for boundary in (
        'mkdir -p "$ROOT/runtime"',
        "restore_gpu_spend_ledger.sh",
        "latest/api/token",
        "describe-instances",
        'manage_gpu_spend.py" start',
        "sync_gpu_spend_ledger.sh",
        "publish_sky_job_status.py",
    ):
        assert managed_verify < managed.index(boundary)


def test_bootstrap_rehearsal_still_authenticates_all_frozen_sources() -> None:
    bootstrap = (SKY / "bootstrap_campaign.sh").read_text()
    rehearsal_branch = bootstrap.index('if [ "$LOCAL_REHEARSAL" -eq 1 ]')
    for name, digest in WORKER_V2_SHA256.items():
        assert name.replace("_native", "") in bootstrap[:rehearsal_branch]
        assert digest in bootstrap[:rehearsal_branch]
    assert "src/glm52_enforcement/glm52_sky_campaign.py" in bootstrap[
        :rehearsal_branch
    ]
    assert "src/glm52_enforcement/glm52_sky_must_start.py" in bootstrap[
        :rehearsal_branch
    ]
    assert "validate_sky_campaign_descriptor" in bootstrap[:rehearsal_branch]
    assert "validate_gpu_spend_approval" in bootstrap[:rehearsal_branch]


def test_admission_boundary_hashes_before_import_and_uses_system_python() -> None:
    setup = yaml.safe_load((SKY / "glm52-campaign.yaml").read_text())["setup"]
    bootstrap = (SKY / "bootstrap_campaign.sh").read_text()
    managed = (SKY / "run_managed_campaign.sh").read_text()

    assert (
        f"/usr/bin/python3 -I {WORKER_V2_ROOT}/publish_worker_start_v2.py publish-latch-v2"
    ) in setup
    assert setup.index("submission intent job name mismatch") < setup.index(
        "publish_worker_start_v2.py publish-latch-v2"
    )
    assert (
        "/usr/bin/python3 -I "
        f"{WORKER_V2_ROOT}/publish_worker_start_v2.py replay-accepted-v2"
    ) in setup
    assert 'WAIT_TIMEOUT_SECONDS=$(/usr/bin/python3 - "$INTENT"' in setup
    source_check = bootstrap.index("for path, expected in frozen_sources.items():")
    source_import = bootstrap.index("spec.loader.exec_module(module)")
    assert source_check < source_import
    assert source_check < bootstrap.index("aws sts get-caller-identity")
    assert (
        '/usr/bin/python3 -I "$REPO/aws/glm52-gpu/skypilot/publish_worker_start_v2.py"'
    ) in bootstrap
    assert (
        "/usr/bin/python3 \\\n"
        '    "$REPO/aws/glm52-gpu/scripts/assert_rnd_aws_account.py"'
    ) in bootstrap
    assert (
        '/usr/bin/python3 -I "$REPO/aws/glm52-gpu/skypilot/publish_worker_start_v2.py"'
    ) in managed
    assert (
        "/usr/bin/python3 \\\n"
        '    "$REPO/aws/glm52-gpu/scripts/assert_rnd_aws_account.py"'
    ) in managed
    assert "GLM52_SKY_CONTROLLER_JOB_ID=$(/usr/bin/python3 -" in managed


def test_mounted_native_policy_publishers_use_isolated_system_python() -> None:
    task = (SKY / "glm52-campaign.yaml").read_text()
    bootstrap = (SKY / "bootstrap_campaign.sh").read_text()
    managed = (SKY / "run_managed_campaign.sh").read_text()

    publisher = "publish_worker_start_v2.py"
    assert task.count(f"/usr/bin/python3 -I {WORKER_V2_ROOT}/{publisher}") == 2
    assert f'/usr/bin/python3 -I "$REPO/aws/glm52-gpu/skypilot/{publisher}"' in bootstrap
    assert f'/usr/bin/python3 -I "$REPO/aws/glm52-gpu/skypilot/{publisher}"' in managed
    assert '"$RECEIPT" "$RECEIPT_FINGERPRINT"' in managed
    assert '"$INTENT" "$INTENT_FINGERPRINT"' in managed
    assert '"$JOB_NAME" "$MANAGED_MODE"' in managed


def test_live_task_scrubs_environment_and_managed_paths_are_literal() -> None:
    task = yaml.safe_load((SKY / "glm52-campaign.yaml").read_text())
    setup = task["setup"]
    run = task["run"]
    bootstrap = (SKY / "bootstrap_campaign.sh").read_text()
    managed = (SKY / "run_managed_campaign.sh").read_text()
    assert "sudo -E" not in setup
    assert "sudo -E" not in run
    assert "/usr/bin/sudo /usr/bin/env -i" in setup
    assert "/usr/bin/sudo /usr/bin/env -i" in run
    assert "GLM52_LIVE_SKY_WORKER=1" in setup
    assert "GLM52_LIVE_SKY_WORKER=1" in run
    assert 'SKYPILOT_MANAGED_JOB_ID="$SKYPILOT_MANAGED_JOB_ID"' in setup
    assert 'SKYPILOT_MANAGED_JOB_ID="$SKYPILOT_MANAGED_JOB_ID"' in run
    assert "GLM52_BOOTSTRAP_REHEARSAL" not in setup
    assert "GLM52_BOOTSTRAP_REHEARSAL" not in run
    assert "sha256sum -c" not in setup
    assert "${SKYPILOT_MANAGED_JOB_ID+x}" not in bootstrap
    for forbidden_override in (
        "${CAMPAIGN_DESCRIPTOR:-",
        "${GPU_SPEND_APPROVAL:-",
        "${SKY_SUBMISSION_INTENT:-",
        "${WORKER_START_LATCH:-",
        "${WORKER_START_LATCH_TRANSPORT:-",
        "${WORKER_START_ACCEPTED:-",
        "${WORKER_START_ADMISSION_RECEIPT:-",
        "${ROOT:-",
        "${KEEP_REPO_DIR:-",
    ):
        assert forbidden_override not in managed


def test_active_worker_path_contains_no_legacy_tokens() -> None:
    for name in (
        "glm52-campaign.yaml",
        "bootstrap_campaign.sh",
        "run_managed_campaign.sh",
    ):
        source = (SKY / name).read_text()
        for token in FORBIDDEN_WORKER_PATH_TOKENS:
            assert token not in source, f"{name} retains forbidden {token}"


def test_worker_path_shell_scripts_parse() -> None:
    for name in (
        "bootstrap_campaign.sh",
        "run_managed_campaign.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", str(SKY / name)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"


@pytest.mark.parametrize("managed_mode", ["production", "cache-seed"])
def test_task_rejects_nonqualification_mode_before_any_aws_call(
    tmp_path: Path,
    managed_mode: str,
) -> None:
    task = yaml.safe_load((SKY / "glm52-campaign.yaml").read_text())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws_called = tmp_path / "aws-called"
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        f"#!/usr/bin/env bash\nprintf called > {aws_called!s}\nexit 97\n"
    )
    fake_aws.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "GLM52_CAMPAIGN_DESCRIPTOR_S3_URI": "s3://fixture/descriptor.json",
        "GLM52_CAMPAIGN_DESCRIPTOR_SHA256": "1" * 64,
        "GLM52_GPU_SPEND_APPROVAL_S3_URI": "s3://fixture/approval.json",
        "GLM52_GPU_SPEND_APPROVAL_SHA256": "2" * 64,
        "GLM52_SKY_JOB_NAME": "glm52-sky-20260726-qualification",
        "GLM52_MANAGED_MODE": managed_mode,
        "GLM52_SKY_SUBMISSION_INTENT_S3_URI": "s3://fixture/intent.json",
        "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256": "3" * 64,
        "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256": "4" * 64,
    }
    result = subprocess.run(
        ["bash", "-c", task["setup"]],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert result.returncode != 0
    assert not aws_called.exists()


@pytest.mark.parametrize(
    "digest_name",
    [
        "GLM52_CAMPAIGN_DESCRIPTOR_SHA256",
        "GLM52_GPU_SPEND_APPROVAL_SHA256",
        "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256",
        "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256",
    ],
)
def test_task_rejects_noncanonical_digest_before_any_aws_call(
    tmp_path: Path,
    digest_name: str,
) -> None:
    task = yaml.safe_load((SKY / "glm52-campaign.yaml").read_text())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws_called = tmp_path / "aws-called"
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        f"#!/usr/bin/env bash\nprintf called > {aws_called!s}\nexit 97\n"
    )
    fake_aws.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "GLM52_CAMPAIGN_DESCRIPTOR_S3_URI": "s3://fixture/descriptor.json",
        "GLM52_CAMPAIGN_DESCRIPTOR_SHA256": "1" * 64,
        "GLM52_GPU_SPEND_APPROVAL_S3_URI": "s3://fixture/approval.json",
        "GLM52_GPU_SPEND_APPROVAL_SHA256": "2" * 64,
        "GLM52_SKY_JOB_NAME": "glm52-sky-20260726-qualification",
        "GLM52_MANAGED_MODE": "qualification",
        "SKYPILOT_MANAGED_JOB_ID": "17",
        "GLM52_SKY_SUBMISSION_INTENT_S3_URI": "s3://fixture/intent.json",
        "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256": "3" * 64,
        "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256": "4" * 64,
    }
    environment[digest_name] = f"{'0' * 64}\n{'0' * 64}  /dev/null"
    result = subprocess.run(
        ["bash", "-c", task["setup"]],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert result.returncode != 0
    assert not aws_called.exists()


def test_task_rejects_foreign_job_name_before_publisher(
    tmp_path: Path,
) -> None:
    worker_root = Path(WORKER_V2_ROOT)
    assert not worker_root.exists()
    worker_root.mkdir(mode=0o700)
    publisher_called = tmp_path / "publisher-called"
    try:
        descriptor = tmp_path / "campaign.json"
        descriptor.write_bytes(b"{}\n")
        intent = tmp_path / "intent.json"
        intent.write_bytes(
            _canonical_json(
                {
                    "managed_mode": "qualification",
                    "sky_job_name": "glm52-sky-20260726-qualification",
                }
            )
        )
        fake_publisher = worker_root / "publish_worker_start_v2.py"
        fake_publisher.write_text(
            "from pathlib import Path\n"
            f"Path({str(publisher_called)!r}).write_text('called')\n"
            "raise SystemExit(97)\n"
        )
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_aws = fake_bin / "aws"
        fake_aws.write_text(
            "#!/usr/bin/python3\n"
            "import os, shutil, sys\n"
            "args = sys.argv[1:]\n"
            "if args[:2] == ['sts', 'get-caller-identity']:\n"
            "    print('246813579024')\n"
            "elif args[:2] == ['s3', 'cp']:\n"
            "    sources = {\n"
            "        's3://fixture/descriptor.json': os.environ['FAKE_DESCRIPTOR'],\n"
            "        's3://fixture/intent.json': os.environ['FAKE_INTENT'],\n"
            "    }\n"
            "    shutil.copyfile(sources[args[2]], args[3])\n"
            "else:\n"
            "    raise SystemExit(96)\n"
        )
        fake_aws.chmod(0o755)
        fake_sudo = fake_bin / "sudo"
        fake_sudo.write_text("#!/bin/sh\nexit 0\n")
        fake_sudo.chmod(0o755)
        environment = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DESCRIPTOR": str(descriptor),
            "FAKE_INTENT": str(intent),
            "GLM52_CAMPAIGN_DESCRIPTOR_S3_URI": ("s3://fixture/descriptor.json"),
            "GLM52_CAMPAIGN_DESCRIPTOR_SHA256": hashlib.sha256(
                descriptor.read_bytes()
            ).hexdigest(),
            "GLM52_GPU_SPEND_APPROVAL_S3_URI": ("s3://fixture/approval.json"),
            "GLM52_GPU_SPEND_APPROVAL_SHA256": "2" * 64,
            "GLM52_SKY_JOB_NAME": "glm52-sky-foreign-qualification",
            "GLM52_MANAGED_MODE": "qualification",
            "SKYPILOT_MANAGED_JOB_ID": "17",
            "GLM52_SKY_SUBMISSION_INTENT_S3_URI": ("s3://fixture/intent.json"),
            "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256": hashlib.sha256(
                intent.read_bytes()
            ).hexdigest(),
            "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256": "4" * 64,
        }
        task = yaml.safe_load((SKY / "glm52-campaign.yaml").read_text())
        result = subprocess.run(
            ["bash", "-c", task["setup"]],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        assert result.returncode != 0
        assert not publisher_called.exists()
    finally:
        shutil.rmtree(worker_root)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _run_setup_acceptance_harness(
    tmp_path: Path,
    *,
    must_start_by: datetime,
) -> tuple[
    subprocess.CompletedProcess[str],
    list[list[str]],
    list[list[str]],
]:
    worker_root = Path(WORKER_V2_ROOT)
    assert not worker_root.exists()
    worker_root.mkdir(mode=0o700)
    try:
        descriptor = tmp_path / "campaign.json"
        descriptor.write_bytes(b"{}\n")
        intent = tmp_path / "intent.json"
        intent.write_bytes(
            _canonical_json(
                {
                    "managed_mode": "qualification",
                    "must_start_by": (
                        must_start_by.astimezone(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z")
                    ),
                    "sky_job_name": "glm52-sky-20260726-qualification",
                }
            )
        )
        publisher_events = tmp_path / "publisher-events.jsonl"
        publisher = worker_root / "publish_worker_start_v2.py"
        publisher.write_text(
            "import json, os, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['PUBLISHER_EVENTS'], 'a') as stream:\n"
            "    stream.write(json.dumps(args) + '\\n')\n"
            "def argument(name):\n"
            "    return args[args.index(name) + 1]\n"
            "if args[0] == 'publish-latch-v2':\n"
            "    for flag in ('--latch-output', '--latch-transport-output'):\n"
            "        path = pathlib.Path(argument(flag))\n"
            "        path.write_bytes(b'{}\\n')\n"
            "        path.chmod(0o600)\n"
            "    raise SystemExit(0)\n"
            "if args[0] == 'replay-accepted-v2':\n"
            "    raise SystemExit(75)\n"
            "raise SystemExit(96)\n"
        )
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        aws_events = tmp_path / "aws-events.jsonl"
        fake_aws = fake_bin / "aws"
        fake_aws.write_text(
            "#!/usr/bin/python3\n"
            "import json, os, shutil, sys\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['AWS_EVENTS'], 'a') as stream:\n"
            "    stream.write(json.dumps(args) + '\\n')\n"
            "if args[:2] == ['sts', 'get-caller-identity']:\n"
            "    print('246813579024')\n"
            "elif args[:2] == ['s3', 'cp']:\n"
            "    sources = {\n"
            "        's3://fixture/descriptor.json': os.environ['FAKE_DESCRIPTOR'],\n"
            "        's3://fixture/intent.json': os.environ['FAKE_INTENT'],\n"
            "    }\n"
            "    shutil.copyfile(sources[args[2]], args[3])\n"
            "else:\n"
            "    raise SystemExit(96)\n"
        )
        fake_aws.chmod(0o755)
        fake_sudo = fake_bin / "sudo"
        fake_sudo.write_text("#!/bin/sh\nexit 0\n")
        fake_sudo.chmod(0o755)
        environment = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "AWS_EVENTS": str(aws_events),
            "PUBLISHER_EVENTS": str(publisher_events),
            "FAKE_DESCRIPTOR": str(descriptor),
            "FAKE_INTENT": str(intent),
            "GLM52_CAMPAIGN_DESCRIPTOR_S3_URI": ("s3://fixture/descriptor.json"),
            "GLM52_CAMPAIGN_DESCRIPTOR_SHA256": hashlib.sha256(
                descriptor.read_bytes()
            ).hexdigest(),
            "GLM52_GPU_SPEND_APPROVAL_S3_URI": ("s3://fixture/approval.json"),
            "GLM52_GPU_SPEND_APPROVAL_SHA256": "2" * 64,
            "GLM52_SKY_JOB_NAME": "glm52-sky-20260726-qualification",
            "GLM52_MANAGED_MODE": "qualification",
            "SKYPILOT_MANAGED_JOB_ID": "17",
            "GLM52_SKY_SUBMISSION_INTENT_S3_URI": ("s3://fixture/intent.json"),
            "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256": hashlib.sha256(
                intent.read_bytes()
            ).hexdigest(),
            "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256": "4" * 64,
        }
        setup = yaml.safe_load((SKY / "glm52-campaign.yaml").read_text())["setup"]
        result = subprocess.run(
            ["bash", "-c", setup],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        publisher_log = (
            [json.loads(line) for line in publisher_events.read_text().splitlines()]
            if publisher_events.exists()
            else []
        )
        aws_log = (
            [json.loads(line) for line in aws_events.read_text().splitlines()]
            if aws_events.exists()
            else []
        )
        return result, publisher_log, aws_log
    finally:
        shutil.rmtree(worker_root)


def test_setup_replay_exit_75_propagates_before_approval_or_repository(
    tmp_path: Path,
) -> None:
    result, publisher_events, aws_events = _run_setup_acceptance_harness(
        tmp_path,
        must_start_by=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert result.returncode == 75
    assert [event[0] for event in publisher_events] == [
        "publish-latch-v2",
        "replay-accepted-v2",
    ]
    replay = publisher_events[1]
    assert replay[replay.index("--wait-timeout-seconds") + 1] == "600"
    assert replay[replay.index("--poll-interval-seconds") + 1] == "5"
    assert [event[:2] for event in aws_events] == [
        ["sts", "get-caller-identity"],
        ["s3", "cp"],
        ["s3", "cp"],
    ]
    assert all("approval" not in item for event in aws_events for item in event)


def test_setup_expired_intent_fails_after_publish_before_replay(
    tmp_path: Path,
) -> None:
    result, publisher_events, aws_events = _run_setup_acceptance_harness(
        tmp_path,
        must_start_by=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    assert result.returncode != 0
    assert [event[0] for event in publisher_events] == ["publish-latch-v2"]
    assert [event[:2] for event in aws_events] == [
        ["sts", "get-caller-identity"],
        ["s3", "cp"],
        ["s3", "cp"],
    ]
    assert all("approval" not in item for event in aws_events for item in event)


def test_setup_replay_timeout_uses_positive_whole_seconds_below_cap(
    tmp_path: Path,
) -> None:
    result, publisher_events, _aws_events = _run_setup_acceptance_harness(
        tmp_path,
        must_start_by=datetime.now(timezone.utc) + timedelta(seconds=90),
    )
    assert result.returncode == 75
    replay = publisher_events[1]
    timeout = int(replay[replay.index("--wait-timeout-seconds") + 1])
    assert 1 <= timeout <= 90
    assert timeout != 600
    assert replay[replay.index("--poll-interval-seconds") + 1] == "5"


def _write_bootstrap_authority(tmp_path: Path) -> tuple[Path, Path]:
    now = datetime.now(timezone.utc)
    approval = build_gpu_spend_approval(
        ingested_at=now - timedelta(minutes=1),
        slack_permalink=None,
    )
    approval_path = tmp_path / "GPU_SPEND_APPROVAL.json"
    approval_path.write_bytes(_canonical_json(approval))
    approval_sha256 = hashlib.sha256(approval_path.read_bytes()).hexdigest()
    artifacts = {
        "source_snapshot_prefix": "source-snapshot/",
        "source_snapshot_sha256": "2" * 64,
        "non_vq_prefix": "non-vq-package/",
        "non_vq_package_sha256": "3" * 64,
        "teich_pack_key": "teich-pack/pack.json",
        "teich_pack_sha256": "4" * 64,
        "frozen_prompt_pack_key": "quality/frozen.json",
        "frozen_prompt_pack_sha256": "5" * 64,
        "training_baseline_prefix": "training-baseline/",
        "training_baseline_sha256": "6" * 64,
        "training_config_key": (
            "campaigns/glm52-sky-20260726/authorities/training.json"
        ),
        "training_config_sha256": "7" * 64,
        "artifact_inventory_key": (
            "campaigns/glm52-sky-20260726/inventories/"
            f"artifact-inventory-{'8' * 64}.json"
        ),
        "artifact_inventory_sha256": "8" * 64,
        "qualification_cache_prefix": (
            f"qualification-cache/seeds/glm52-sky-20260726/{'9' * 64}/"
        ),
        "qualification_cache_manifest_sha256": "9" * 64,
    }
    descriptor = build_sky_campaign_descriptor(
        run_id="glm52-sky-20260726",
        must_start_by=now + timedelta(hours=1),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity="arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket="keep-glm52-us-west-2-246813579024",
        jobs_bucket="keep-glm52-us-west-2-246813579024",
        repo_tar_key="campaigns/glm52-sky-20260726/repository/repo.tar.gz",
        repo_tar_sha256="1" * 64,
        campaign_descriptor_key=(
            "campaigns/glm52-sky-20260726/submissions/qualification/"
            "campaign-descriptor-v2.json"
        ),
        approval_key=(
            "campaigns/glm52-sky-20260726/authorities/GPU_SPEND_APPROVAL.json"
        ),
        approval_sha256=approval_sha256,
        artifacts=artifacts,
    )
    descriptor_path = tmp_path / "campaign.json"
    descriptor_path.write_bytes(_canonical_json(descriptor))
    return descriptor_path, approval_path


def _bootstrap_environment(
    tmp_path: Path,
    *,
    rehearsal: str,
    managed_job_id: str | None,
) -> tuple[dict[str, str], Path]:
    descriptor, approval = _write_bootstrap_authority(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    aws_log = tmp_path / "aws-called"
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {aws_log!s}\n"
        "printf '%s\\n' "
        '\'{"Account":"246813579024",'
        '"Arn":"arn:aws:sts::246813579024:assumed-role/'
        "keep-glm52-gpu-worker/test\"}'\n"
    )
    fake_aws.chmod(0o755)
    missing = tmp_path / "missing-admission-artifact.json"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAMPAIGN_DESCRIPTOR": str(descriptor),
        "GPU_SPEND_APPROVAL": str(approval),
        "KEEP_REPO_DIR": str(ROOT),
        "ROOT": str(tmp_path / "nvme"),
        "GLM52_MANAGED_MODE": "qualification",
        "GLM52_BOOTSTRAP_REHEARSAL": rehearsal,
        "SKY_SUBMISSION_INTENT": str(missing),
        "WORKER_START_LATCH": str(missing),
        "WORKER_START_LATCH_TRANSPORT": str(missing),
        "WORKER_START_ACCEPTED": str(missing),
        "WORKER_START_ADMISSION_RECEIPT": str(missing),
        "GLM52_CAMPAIGN_DESCRIPTOR_S3_URI": "s3://fixture/descriptor.json",
        "GLM52_CAMPAIGN_DESCRIPTOR_SHA256": hashlib.sha256(
            descriptor.read_bytes()
        ).hexdigest(),
        "GLM52_SKY_SUBMISSION_INTENT_S3_URI": "s3://fixture/intent.json",
        "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256": "3" * 64,
        "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256": "4" * 64,
    }
    if managed_job_id is not None:
        environment["SKYPILOT_MANAGED_JOB_ID"] = managed_job_id
    else:
        environment.pop("SKYPILOT_MANAGED_JOB_ID", None)
    environment.pop("GLM52_LIVE_SKY_WORKER", None)
    return environment, aws_log


def test_exact_local_rehearsal_bypasses_live_receipt_but_checks_frozen_tree(
    tmp_path: Path,
) -> None:
    environment, aws_log = _bootstrap_environment(
        tmp_path,
        rehearsal="1",
        managed_job_id=None,
    )
    result = subprocess.run(
        ["bash", str(SKY / "bootstrap_campaign.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "SKYPILOT-BOOTSTRAP-REHEARSAL-OK" in result.stdout
    assert "GPU-ENV-REHEARSAL-OK" in result.stdout
    assert aws_log.is_file()


@pytest.mark.parametrize("rehearsal", ["", "0", "true", "01", "qualification"])
def test_missing_receipt_cannot_bypass_bootstrap_for_nonexact_rehearsal_value(
    tmp_path: Path,
    rehearsal: str,
) -> None:
    environment, _aws_log = _bootstrap_environment(
        tmp_path,
        rehearsal=rehearsal,
        managed_job_id=None,
    )
    result = subprocess.run(
        ["bash", str(SKY / "bootstrap_campaign.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert result.returncode != 0
    assert "SKYPILOT-BOOTSTRAP-REHEARSAL-OK" not in result.stdout
    assert "GPU-ENV-REHEARSAL-OK" not in result.stdout


def test_nonrehearsal_bootstrap_rejects_path_overrides_before_account(
    tmp_path: Path,
) -> None:
    environment, aws_log = _bootstrap_environment(
        tmp_path,
        rehearsal="0",
        managed_job_id=None,
    )
    result = subprocess.run(
        ["bash", str(SKY / "bootstrap_campaign.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert result.returncode != 0
    assert not aws_log.exists()


@pytest.mark.parametrize("managed_job_id", ["", "0", "bad", "17"])
def test_rehearsal_flag_cannot_bypass_receipt_on_a_managed_worker(
    tmp_path: Path,
    managed_job_id: str,
) -> None:
    environment, _aws_log = _bootstrap_environment(
        tmp_path,
        rehearsal="1",
        managed_job_id=managed_job_id,
    )
    environment["GLM52_LIVE_SKY_WORKER"] = "1"
    result = subprocess.run(
        ["bash", str(SKY / "bootstrap_campaign.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert result.returncode != 0
    assert "SKYPILOT-BOOTSTRAP-REHEARSAL-OK" not in result.stdout
    assert "GPU-ENV-REHEARSAL-OK" not in result.stdout


def _write_managed_worker_harness(
    tmp_path: Path,
    *,
    receipt: object,
    managed_mode: str = "qualification",
    intent_job_name: str = "glm52-sky-20260726-qualification",
    replacement_receipt: object | None = None,
    ambient_job_id: str = "17",
) -> tuple[dict[str, str], Path, Path]:
    repo = tmp_path / "repo"
    (repo / "aws/glm52-gpu/skypilot").mkdir(parents=True)
    (repo / "aws/glm52-gpu/scripts").mkdir(parents=True)
    event_log = tmp_path / "events.jsonl"
    publisher = repo / "aws/glm52-gpu/skypilot/publish_worker_start_v2.py"
    publisher.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "with open(os.environ['WORKER_EVENT_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(['publisher', *sys.argv[1:]]) + '\\n')\n"
        "replacement = os.environ.get('REPLACEMENT_RECEIPT')\n"
        "if replacement:\n"
        "    args = sys.argv[1:]\n"
        "    path = pathlib.Path(args[args.index('--admission-receipt') + 1])\n"
        "    path.write_text(replacement)\n"
        "    path.chmod(0o600)\n"
    )
    publisher.chmod(0o755)
    account_guard = repo / "aws/glm52-gpu/scripts/assert_rnd_aws_account.py"
    account_guard.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "value = json.load(sys.stdin)\n"
        "if value.get('Account') != '246813579024':\n"
        "    raise SystemExit(70)\n"
    )
    account_guard.chmod(0o755)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['WORKER_EVENT_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(['aws', *sys.argv[1:]]) + '\\n')\n"
        "print(json.dumps({\n"
        "    'Account': '246813579024',\n"
        "    'Arn': ('arn:aws:sts::246813579024:assumed-role/'\n"
        "            'keep-glm52-gpu-worker/test'),\n"
        "}))\n"
    )
    fake_aws.chmod(0o755)
    fake_mkdir = fake_bin / "mkdir"
    fake_mkdir.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['WORKER_EVENT_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps([\n"
        "        'mkdir', os.environ.get('GLM52_SKY_CONTROLLER_JOB_ID'),\n"
        "        *sys.argv[1:],\n"
        "    ]) + '\\n')\n"
        "raise SystemExit(91)\n"
    )
    fake_mkdir.chmod(0o755)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = {
        "CAMPAIGN_DESCRIPTOR": inputs / "campaign.json",
        "GPU_SPEND_APPROVAL": inputs / "GPU_SPEND_APPROVAL.json",
        "SKY_SUBMISSION_INTENT": inputs / "SKYPILOT_SUBMISSION_INTENT.json",
        "WORKER_START_LATCH": inputs / "WORKER_START_LATCH.json",
        "WORKER_START_LATCH_TRANSPORT": (inputs / "WORKER_START_LATCH_TRANSPORT.json"),
        "WORKER_START_ACCEPTED": inputs / "WORKER_START_ACCEPTED.json",
        "WORKER_START_ADMISSION_RECEIPT": (
            inputs / "WORKER_START_ADMISSION_RECEIPT.json"
        ),
    }
    for name, path in paths.items():
        if name == "WORKER_START_ADMISSION_RECEIPT":
            path.write_bytes(_canonical_json(receipt))
        elif name == "SKY_SUBMISSION_INTENT":
            path.write_bytes(
                _canonical_json(
                    {
                        "managed_mode": "qualification",
                        "sky_job_name": intent_job_name,
                    }
                )
            )
        else:
            path.write_bytes(b"{}\n")
        if name.startswith("WORKER_START_"):
            path.chmod(0o600)
    managed_source = (SKY / "run_managed_campaign.sh").read_text()
    replacements = {
        "DESCRIPTOR=/etc/keep-glm52/campaign.json": (
            'DESCRIPTOR="${CAMPAIGN_DESCRIPTOR:?}"'
        ),
        "APPROVAL=/etc/keep-glm52/GPU_SPEND_APPROVAL.json": (
            'APPROVAL="${GPU_SPEND_APPROVAL:?}"'
        ),
        "INTENT=/etc/keep-glm52/SKYPILOT_SUBMISSION_INTENT.json": (
            'INTENT="${SKY_SUBMISSION_INTENT:?}"'
        ),
        "LATCH=/etc/keep-glm52/WORKER_START_LATCH.json": (
            'LATCH="${WORKER_START_LATCH:?}"'
        ),
        (
            "LATCH_TRANSPORT=/etc/keep-glm52/WORKER_START_LATCH_TRANSPORT.json"
        ): 'LATCH_TRANSPORT="${WORKER_START_LATCH_TRANSPORT:?}"',
        "ACCEPTED=/etc/keep-glm52/WORKER_START_ACCEPTED.json": (
            'ACCEPTED="${WORKER_START_ACCEPTED:?}"'
        ),
        (
            "RECEIPT=/etc/keep-glm52/WORKER_START_ADMISSION_RECEIPT.json"
        ): 'RECEIPT="${WORKER_START_ADMISSION_RECEIPT:?}"',
        "ROOT=/mnt/nvme/glm52-campaign": 'ROOT="${ROOT:?}"',
        "REPO=/opt/keep-campaign/repo": 'REPO="${KEEP_REPO_DIR:?}"',
        "EXPECTED_ROOT_UID = 0": f"EXPECTED_ROOT_UID = {os.getuid()}",
    }
    for old, new in replacements.items():
        assert old in managed_source
        managed_source = managed_source.replace(old, new)
    harness_script = tmp_path / "run_managed_campaign_harness.sh"
    harness_script.write_text(managed_source)
    harness_script.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "WORKER_EVENT_LOG": str(event_log),
        "KEEP_REPO_DIR": str(repo),
        "ROOT": str(tmp_path / "nvme"),
        "GLM52_SKY_JOB_NAME": "glm52-sky-20260726-qualification",
        "GLM52_MANAGED_MODE": managed_mode,
        "GLM52_LIVE_SKY_WORKER": "1",
        "SKYPILOT_MANAGED_JOB_ID": ambient_job_id,
        "GLM52_CAMPAIGN_DESCRIPTOR_S3_URI": "s3://fixture/descriptor.json",
        "GLM52_CAMPAIGN_DESCRIPTOR_SHA256": "1" * 64,
        "GLM52_SKY_SUBMISSION_INTENT_S3_URI": "s3://fixture/intent.json",
        "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256": "2" * 64,
        "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256": "3" * 64,
        **{name: str(path) for name, path in paths.items()},
    }
    if replacement_receipt is not None:
        environment["REPLACEMENT_RECEIPT"] = _canonical_json(
            replacement_receipt
        ).decode("ascii")
    return environment, event_log, harness_script


def _run_managed_harness(
    tmp_path: Path,
    *,
    receipt: object,
    managed_mode: str = "qualification",
    intent_job_name: str = "glm52-sky-20260726-qualification",
    replacement_receipt: object | None = None,
    ambient_job_id: str = "17",
) -> tuple[subprocess.CompletedProcess[str], list[list[object]]]:
    environment, event_log, harness_script = _write_managed_worker_harness(
        tmp_path,
        receipt=receipt,
        managed_mode=managed_mode,
        intent_job_name=intent_job_name,
        replacement_receipt=replacement_receipt,
        ambient_job_id=ambient_job_id,
    )
    result = subprocess.run(
        ["bash", str(harness_script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    events = (
        [json.loads(line) for line in event_log.read_text().splitlines()]
        if event_log.exists()
        else []
    )
    return result, events


@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {"record_type": "foreign", "sky_job_id": 17},
        {
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": True,
        },
        {
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": "17",
        },
        {
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": "0017",
        },
        {
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": 0,
        },
        {
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": -1,
        },
        {
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": 17.0,
        },
    ],
)
def test_managed_receipt_requires_exact_positive_nonboolean_integer_before_runtime(
    tmp_path: Path,
    receipt: object,
) -> None:
    result, events = _run_managed_harness(tmp_path, receipt=receipt)
    assert result.returncode != 0
    assert [event[0] for event in events] == ["aws", "publisher"]
    assert "verify-receipt-v2" in events[1]


def test_managed_job_id_comes_only_from_verified_receipt_before_runtime(
    tmp_path: Path,
) -> None:
    result, events = _run_managed_harness(
        tmp_path,
        receipt={
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": 17,
        },
    )
    assert result.returncode == 91
    assert [event[0] for event in events] == ["aws", "publisher", "mkdir"]
    assert "verify-receipt-v2" in events[1]
    assert events[2][1] == "17"


def test_managed_rejects_ambient_job_id_mismatch_after_verify_before_runtime(
    tmp_path: Path,
) -> None:
    result, events = _run_managed_harness(
        tmp_path,
        receipt={
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": 17,
        },
        ambient_job_id="999",
    )
    assert result.returncode != 0
    assert [event[0] for event in events] == ["aws", "publisher"]


def test_managed_rejects_foreign_job_name_after_verify_before_runtime(
    tmp_path: Path,
) -> None:
    result, events = _run_managed_harness(
        tmp_path,
        receipt={
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": 17,
        },
        intent_job_name="glm52-sky-foreign-qualification",
    )
    assert result.returncode != 0
    assert [event[0] for event in events] == ["aws", "publisher"]


def test_managed_rejects_receipt_swap_after_verify_before_runtime(
    tmp_path: Path,
) -> None:
    result, events = _run_managed_harness(
        tmp_path,
        receipt={
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": 17,
        },
        replacement_receipt={
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": 23,
        },
    )
    assert result.returncode != 0
    assert [event[0] for event in events] == ["aws", "publisher"]


@pytest.mark.parametrize("managed_mode", ["production", "cache-seed"])
def test_managed_wrapper_rejects_nonqualification_before_remote_authority(
    tmp_path: Path,
    managed_mode: str,
) -> None:
    result, events = _run_managed_harness(
        tmp_path,
        receipt={
            "record_type": "glm52_sky_worker_admission_receipt_v1",
            "sky_job_id": 17,
        },
        managed_mode=managed_mode,
    )
    assert result.returncode != 0
    assert events == []


def test_nvme_preflight_is_idempotent_and_refuses_root_disk_fallback() -> None:
    source = (SKY / "prepare_nvme_storage.sh").read_text()
    assert "mountpoint -q /opt/dlami/nvme" in source
    assert "mount --bind" in source
    assert "Amazon EC2 NVMe Instance Storage" in source
    assert "lsblk" in source
    assert "MIN_AVAILABLE_BYTES" in source
    assert "findmnt -n -o SOURCE /" in source
    assert "campaign storage resolved to the root filesystem" in source
    assert "GLM52-NVME-READY" in source


def test_pinned_control_plane_installer_and_validator_are_isolated() -> None:
    installer = (SCRIPTS / "install_skypilot_control_plane.sh").read_text()
    validator = (SCRIPTS / "validate_skypilot_control_plane.py").read_text()

    assert "skypilot[aws]==0.13.0" in installer
    assert 'SKYPILOT_QUEUE_MANAGER_PORT:-50012' in installer
    assert '"botocore[crt]==$BOTOCore_VERSION"' not in installer
    assert '"botocore[crt]==$BOTOCORE_VERSION"' in installer
    assert "DEFAULT_QUEUE_MANAGER_PORT = 50011" in installer
    assert "DEFAULT_QUEUE_MANAGER_PORT = {queue_port}" in installer
    assert "control-environment.json" in installer
    assert "assert_rnd_aws_account.sh" in installer
    assert '"account_id": account_id' in installer
    assert '"aws_profile": aws_profile' in installer
    assert '"api_server_port": api_server_port' in installer
    assert '"python_executable":' in installer
    assert '"sky_executable":' in installer
    assert "uv venv" in installer
    assert "$REPO/.venv" not in installer
    assert "sky.Task.from_yaml" in validator
    assert "parse_and_validate_config_file" in validator
    assert "0.13.0" in validator


def test_pinned_skypilot_parser_accepts_task_and_secret_free_config() -> None:
    pinned_python = Path(
        "/Users/jack.mazac/.local/share/keep/skypilot-0.13.0/bin/python"
    )
    assert pinned_python.is_file()
    result = subprocess.run(
        [
            str(pinned_python),
            str(SCRIPTS / "validate_skypilot_control_plane.py"),
            "--task",
            str(SKY / "glm52-campaign.yaml"),
            "--config",
            str(SKY / "skypilot-config.yaml.in"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "SKYPILOT_DISABLE_USAGE_COLLECTION": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload == {
        "instance_type": "p5.48xlarge",
        "record_type": "glm52_skypilot_parser_validation_v1",
        "region": "us-west-2",
        "skypilot_version": "0.13.0",
        "status": "passed",
        "task": "glm52-campaign",
    }


def test_operator_docs_pin_a_dedicated_aws_only_skypilot_server_config() -> None:
    account_docs = (
        ROOT / "aws/glm52-gpu/AWS_ACCOUNT_PROFILES.md"
    ).read_text()
    gpu_readme = (ROOT / "aws/glm52-gpu/README.md").read_text()
    for source in (account_docs, gpu_readme):
        assert "SKYPILOT_GLOBAL_CONFIG" in source
        assert "SKYPILOT_API_SERVER_ENDPOINT=http://127.0.0.1:46580" in source
        assert "skypilot-config.yaml.in" in source
        assert "allowed_clouds" in source
        assert "account=246813579024" in source
