"""Task 13 deterministic local source-bundle tests."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)
from glm52_enforcement.task10_worker import (
    MountFreeTaskInputs,
    build_worker_bootstrap_descriptor,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
REPO_ROOT = Path(__file__).resolve().parents[1]
OWNER_APPROVAL = (
    REPO_ROOT / "docs/superpowers/approvals/"
    "2026-07-28-glm52-campaign-owner-approvals.md"
)
OWNER_APPROVAL_SHA256 = (
    "23ccff3454896d7fb8e9d0722b6c76522ebf2dfa3dfea9856a17a3e48b26ebc1"
)


def _api():
    from glm52_enforcement import task13_source_bundle

    return task13_source_bundle


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical(path: Path, value: object) -> tuple[str, str]:
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw[:-1]).hexdigest()


def _source(api, path: Path, value: object):
    file_sha256, body_sha256 = _canonical(path, value)
    return api.PinnedJsonSource(
        path=path,
        expected_file_sha256=file_sha256,
        expected_body_sha256=body_sha256,
    )


def _live_sources(tmp_path: Path):
    api = _api()
    sources = tmp_path / "sources"
    sources.mkdir(parents=True)

    task11 = _source(
        api,
        sources / "task11-review.json",
        {"review": "approved", "task": 11},
    )
    task12 = _source(
        api,
        sources / "task12-review.json",
        {"review": "approved", "task": 12},
    )
    gpu = _source(
        api,
        sources / "gpu-spend.json",
        build_gpu_spend_approval(
            ingested_at=datetime(2026, 7, 23, 18, 5, tzinfo=UTC),
            slack_permalink=None,
        ),
    )
    campaign_value = build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=datetime(2026, 7, 31, 6, 5, tzinfo=UTC),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity=("arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"),
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket="keep-glm52-models-246813579024-us-west-2",
        jobs_bucket="keep-glm52-models-246813579024-us-west-2",
        repo_tar_key=("campaigns/glm52-sky-20260724/repository/repo.tar.gz"),
        repo_tar_sha256=_sha("repo"),
        campaign_descriptor_key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "campaign-descriptor-v2.json"
        ),
        approval_key=(
            "campaigns/glm52-sky-20260724/authorities/GPU_SPEND_APPROVAL.json"
        ),
        approval_sha256=_sha("gpu-approval"),
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": _sha("source"),
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": _sha("non-vq"),
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_sha256": _sha("teich"),
            "frozen_prompt_pack_key": (
                "quality/glm52-family-eval-prompts-20260709-v2.json"
            ),
            "frozen_prompt_pack_sha256": _sha("prompt"),
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": _sha("baseline"),
            "training_config_key": (
                "campaigns/glm52-sky-20260724/authorities/training.json"
            ),
            "training_config_sha256": _sha("training"),
            "artifact_inventory_key": (
                "campaigns/glm52-sky-20260724/inventories/"
                f"artifact-inventory-{_sha('inventory')}.json"
            ),
            "artifact_inventory_sha256": _sha("inventory"),
            "qualification_cache_prefix": (
                f"qualification-cache/seeds/glm52-sky-20260724/{_sha('cache')}/"
            ),
            "qualification_cache_manifest_sha256": _sha("cache"),
        },
    )
    campaign = _source(
        api,
        sources / "campaign-descriptor.json",
        campaign_value,
    )
    campaign_raw = campaign.path.read_bytes()
    campaign_file_sha256 = hashlib.sha256(campaign_raw).hexdigest()

    approval_file_sha256 = _sha("approval-file")
    intent_body_sha256 = _sha("intent-body")
    archive_file_sha256 = _sha("archive-file")
    worker_value = asdict(
        build_worker_bootstrap_descriptor(
            schema_version=1,
            record_type="glm52_task10_worker_bootstrap_descriptor_v1",
            account_id=ACCOUNT_ID,
            region=REGION,
            run_id=RUN_ID,
            campaign_identity_sha256=str(campaign_value["campaign_identity_sha256"]),
            activation_id="activation-0001",
            activation_ordinal=1,
            generation=1,
            generation_text="00000001",
            action_key="ACTIVATION#activation-0001#ACTION#SKY_POST#00000001",
            sky_job_name=RUN_ID,
            execution_deadline="2026-07-31T12:00:00Z",
            gpu_allocation_sha256=_sha("allocation"),
            base_descriptor_s3_uri=(
                "s3://keep-glm52-models-246813579024-us-west-2/"
                "campaigns/glm52-sky-20260724/submissions/production/"
                "campaign-descriptor-v2.json"
            ),
            base_descriptor_version_id="campaign-descriptor-version-1",
            base_descriptor_file_sha256=campaign_file_sha256,
            base_descriptor_body_sha256=str(campaign_value["descriptor_body_sha256"]),
            archive_identity_sha256=archive_file_sha256,
            repository_archive_version_id="archive-version-1",
            approval_identity_sha256=approval_file_sha256,
            approval_version_id="approval-version-1",
            intent_identity_sha256=intent_body_sha256,
            intent_version_id="intent-version-1",
            task8_live_h1d_identity_sha256=_sha("task8-live"),
            task8_spend_authority_identity_sha256=_sha("task8-spend"),
            task9_launch_identity_sha256=_sha("task9-launch"),
            task9_admission_identity_sha256=_sha("task9-admission"),
            task9_custody_identity_sha256=_sha("task9-custody"),
        )
    )
    worker = _source(
        api,
        sources / "task10-worker.json",
        worker_value,
    )
    worker_file_sha256 = hashlib.sha256(worker.path.read_bytes()).hexdigest()
    task_inputs_value = asdict(
        MountFreeTaskInputs(
            job_name=RUN_ID,
            descriptor_s3_uri=(
                "s3://keep-glm52-models-246813579024-us-west-2/"
                "task13/production/task10-worker-descriptor.json"
            ),
            descriptor_version_id="worker-descriptor-version-1",
            descriptor_file_sha256=worker_file_sha256,
            approval_s3_uri=(
                "s3://keep-glm52-models-246813579024-us-west-2/"
                "task13/approvals/gpu-spend.json"
            ),
            approval_version_id="approval-version-1",
            approval_file_sha256=approval_file_sha256,
            intent_s3_uri=(
                "s3://keep-glm52-models-246813579024-us-west-2/"
                "task13/production/intent.json"
            ),
            intent_version_id="intent-version-1",
            intent_file_sha256=_sha("intent-file"),
            intent_body_sha256=intent_body_sha256,
            repository_archive_s3_uri=(
                "s3://keep-glm52-models-246813579024-us-west-2/"
                "task13/archive/repo.tar.gz"
            ),
            repository_archive_version_id="archive-version-1",
            repository_archive_file_sha256=archive_file_sha256,
        )
    )
    task_inputs = _source(
        api,
        sources / "task10-task-inputs.json",
        task_inputs_value,
    )
    return {
        "TASK11_REVIEW_APPROVAL": task11,
        "TASK12_REVIEW_APPROVAL": task12,
        "GPU_SPEND_APPROVAL": gpu,
        "CAMPAIGN_DESCRIPTOR": campaign,
        "TASK10_WORKER_DESCRIPTOR": worker,
        "TASK10_TASK_INPUTS": task_inputs,
    }


def _build(tmp_path: Path):
    api = _api()
    output = tmp_path / "bundle"
    manifest = api.build_task13_source_bundle(
        output_directory=output,
        owner_approval_source=api.PinnedFileSource(
            path=OWNER_APPROVAL,
            expected_file_sha256=OWNER_APPROVAL_SHA256,
        ),
        live_sources=_live_sources(tmp_path),
    )
    return api, output, manifest


def test_builds_exact_canonical_write_once_source_bundle(tmp_path: Path) -> None:
    api, output, manifest = _build(tmp_path)

    assert manifest["account_id"] == ACCOUNT_ID
    assert manifest["region"] == REGION
    assert manifest["run_id"] == RUN_ID
    assert [row["artifact_kind"] for row in manifest["artifacts"]] == list(
        api.ARTIFACT_KINDS
    )
    assert manifest["dynamic_outputs"] == [
        {
            "artifact_kind": kind,
            "classification": "DYNAMIC_DEPLOYMENT_OR_REHEARSAL_OUTPUT",
            **(
                {
                    "key_prefix": (
                        "task13/gates/clean-rehearsal/"
                    ),
                    "materialized_by_source_bundle": False,
                }
                if kind == "CLEAN_REHEARSAL"
                else {}
            ),
        }
        for kind in api.DYNAMIC_OUTPUT_KINDS
    ]
    clean = next(
        row
        for row in manifest["dynamic_outputs"]
        if row["artifact_kind"] == "CLEAN_REHEARSAL"
    )
    assert clean["key_prefix"] == api.DYNAMIC_OUTPUT_KEY_PREFIXES[
        "CLEAN_REHEARSAL"
    ]
    assert "relative_path" not in clean
    assert len(list(output.iterdir())) == len(api.ARTIFACT_KINDS) + 1
    for row in manifest["artifacts"]:
        artifact = output / row["relative_path"]
        raw = artifact.read_bytes()
        assert raw == canonical_json_bytes(json.loads(raw)) + b"\n"
        assert hashlib.sha256(raw).hexdigest() == row["file_sha256"]
        assert hashlib.sha256(raw[:-1]).hexdigest() == row["body_sha256"]
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o600

    manifest_path = output / api.MANIFEST_NAME
    raw = manifest_path.read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw)) + b"\n"
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    by_kind = {
        row["artifact_kind"]: json.loads((output / row["relative_path"]).read_bytes())
        for row in manifest["artifacts"]
    }
    assert by_kind["TRAINING_CONFIGURATION"]["automatic_promotion"] is False
    assert by_kind["TRAINING_CONFIGURATION"]["model_upload_authorized"] is False
    assert by_kind["SUPPORT_APPROVAL"]["work_deadline_hours"] == 68
    assert by_kind["SUPPORT_APPROVAL"]["delete_request_deadline_hours"] == 71
    assert by_kind["SUPPORT_APPROVAL"]["cost_incident_deadline_hours"] == 72
    assert by_kind["RESIDUAL_LIABILITY_APPROVAL"]["gpu_reserve_seconds"] == 900
    assert by_kind["RESIDUAL_LIABILITY_APPROVAL"]["gpu_reserve_usd"] == "13.76"
    assert (
        by_kind["RESIDUAL_LIABILITY_APPROVAL"][
            "no_hard_post_acceptance_aws_billing_cap_accepted"
        ]
        is True
    )
    rows = {row["artifact_kind"]: row for row in manifest["artifacts"]}
    assert (
        rows["CAMPAIGN_DESCRIPTOR"]["reviewed_artifact_kind"]
        == "PRODUCTION_DESCRIPTOR"
    )
    assert (
        by_kind["TASK10_TASK_INPUTS"]["descriptor_file_sha256"]
        == rows["TASK10_WORKER_DESCRIPTOR"]["file_sha256"]
    )
    assert (
        by_kind["TASK10_WORKER_DESCRIPTOR"]["base_descriptor_file_sha256"]
        == rows["CAMPAIGN_DESCRIPTOR"]["file_sha256"]
    )


def test_rejects_noncanonical_live_source_and_symlink(tmp_path: Path) -> None:
    api = _api()
    live = _live_sources(tmp_path)
    review = live["TASK11_REVIEW_APPROVAL"]
    review.path.write_text('{\n  "review": "approved", "task": 11\n}\n')
    raw = review.path.read_bytes()
    live["TASK11_REVIEW_APPROVAL"] = api.PinnedJsonSource(
        path=review.path,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        expected_body_sha256=hashlib.sha256(
            canonical_json_bytes(json.loads(raw))
        ).hexdigest(),
    )
    with pytest.raises(api.Task13SourceBundleError, match="canonical JSON"):
        api.build_task13_source_bundle(
            output_directory=tmp_path / "bundle",
            owner_approval_source=api.PinnedFileSource(
                path=OWNER_APPROVAL,
                expected_file_sha256=OWNER_APPROVAL_SHA256,
            ),
            live_sources=live,
        )

    symlink = tmp_path / "owner-link"
    symlink.symlink_to(OWNER_APPROVAL)
    with pytest.raises(api.Task13SourceBundleError, match="non-symlink"):
        api.build_task13_source_bundle(
            output_directory=tmp_path / "bundle-2",
            owner_approval_source=api.PinnedFileSource(
                path=symlink,
                expected_file_sha256=OWNER_APPROVAL_SHA256,
            ),
            live_sources=_live_sources(tmp_path / "again"),
        )


def test_rejects_semantically_coherent_descriptor_substitution(
    tmp_path: Path,
) -> None:
    api = _api()
    live = _live_sources(tmp_path)
    source = live["CAMPAIGN_DESCRIPTOR"]
    value = json.loads(source.path.read_bytes())
    value["run_id"] = "glm52-sky-foreign"
    body = dict(value)
    body.pop("descriptor_body_sha256")
    value["descriptor_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    live["CAMPAIGN_DESCRIPTOR"] = _source(api, source.path, value)

    with pytest.raises(api.Task13SourceBundleError, match="run_id"):
        api.build_task13_source_bundle(
            output_directory=tmp_path / "bundle",
            owner_approval_source=api.PinnedFileSource(
                path=OWNER_APPROVAL,
                expected_file_sha256=OWNER_APPROVAL_SHA256,
            ),
            live_sources=live,
        )


def test_rejects_task10_cross_binding_drift(tmp_path: Path) -> None:
    api = _api()
    live = _live_sources(tmp_path)
    source = live["TASK10_TASK_INPUTS"]
    value = json.loads(source.path.read_bytes())
    value["descriptor_file_sha256"] = _sha("foreign-descriptor")
    live["TASK10_TASK_INPUTS"] = _source(api, source.path, value)

    with pytest.raises(api.Task13SourceBundleError, match="descriptor"):
        api.build_task13_source_bundle(
            output_directory=tmp_path / "bundle",
            owner_approval_source=api.PinnedFileSource(
                path=OWNER_APPROVAL,
                expected_file_sha256=OWNER_APPROVAL_SHA256,
            ),
            live_sources=live,
        )


def test_rejects_output_overwrite(tmp_path: Path) -> None:
    api, output, _manifest = _build(tmp_path)
    with pytest.raises(api.Task13SourceBundleError, match="new absolute"):
        api.build_task13_source_bundle(
            output_directory=output,
            owner_approval_source=api.PinnedFileSource(
                path=OWNER_APPROVAL,
                expected_file_sha256=OWNER_APPROVAL_SHA256,
            ),
            live_sources=_live_sources(tmp_path / "second"),
        )


def test_cli_is_executable_and_exposes_all_live_identity_flags() -> None:
    script = REPO_ROOT / "aws/glm52-gpu/scripts/build_glm52_task13_source_bundle.py"
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for kind in (
        "task11-review-approval",
        "task12-review-approval",
        "gpu-spend-approval",
        "campaign-descriptor",
        "task10-worker-descriptor",
        "task10-task-inputs",
    ):
        assert f"--{kind}-source" in result.stdout
        assert f"--{kind}-file-sha256" in result.stdout
        assert f"--{kind}-body-sha256" in result.stdout
    assert "--owner-approval-source" in result.stdout
    assert "--owner-approval-file-sha256" in result.stdout
    text = script.read_text(encoding="utf-8")
    assert "boto3" not in text
    assert "subprocess" not in text


def test_cli_materializes_the_validated_bundle(tmp_path: Path) -> None:
    api = _api()
    live = _live_sources(tmp_path)
    script = (
        REPO_ROOT
        / "aws/glm52-gpu/scripts/build_glm52_task13_source_bundle.py"
    )
    output = tmp_path / "cli-bundle"
    args = [
        sys.executable,
        str(script),
        "--output-directory",
        str(output),
        "--owner-approval-source",
        str(OWNER_APPROVAL),
        "--owner-approval-file-sha256",
        OWNER_APPROVAL_SHA256,
    ]
    flag_by_kind = {
        "TASK11_REVIEW_APPROVAL": "task11-review-approval",
        "TASK12_REVIEW_APPROVAL": "task12-review-approval",
        "GPU_SPEND_APPROVAL": "gpu-spend-approval",
        "CAMPAIGN_DESCRIPTOR": "campaign-descriptor",
        "TASK10_WORKER_DESCRIPTOR": "task10-worker-descriptor",
        "TASK10_TASK_INPUTS": "task10-task-inputs",
    }
    for kind, flag in flag_by_kind.items():
        source = live[kind]
        args.extend(
            [
                f"--{flag}-source",
                str(source.path),
                f"--{flag}-file-sha256",
                source.expected_file_sha256,
                f"--{flag}-body-sha256",
                source.expected_body_sha256,
            ]
        )
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(output / api.MANIFEST_NAME)
    assert json.loads((output / api.MANIFEST_NAME).read_bytes())[
        "canonical_identity_sha256"
    ]
