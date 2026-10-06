from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aws/glm52-gpu/scripts/rehearse_staged_control_plane.sh"
HEX = {"campaign": "c" * 64}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _body_sha(value: dict[str, object], field: str) -> str:
    return hashlib.sha256(
        _canonical({key: item for key, item in value.items() if key != field})
    ).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    run_id = "glm52-sky-test"
    descriptor_key = (
        f"campaigns/{run_id}/submissions/test/campaign-descriptor-v2.json"
    )
    inventory_key = (
        f"campaigns/{run_id}/inventories/artifact-inventory-test.json"
    )
    audit_key = f"campaigns/{run_id}/audits/artifact-audit-test.json"
    readiness_key = (
        f"campaigns/{run_id}/submissions/test/STAGED_CONTROL_PLANE_READY.json"
    )

    repo_tar = tmp_path / "repo.tar.gz"
    repo_tar.write_bytes(b"exact immutable repository tar fixture")
    inventory_path = tmp_path / "artifact-inventory-v1.json"
    inventory: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_s3_artifact_inventory_v1",
        "run_id": run_id,
        "bucket": "fixture-bucket",
        "objects": [],
    }
    inventory["inventory_body_sha256"] = _body_sha(
        inventory, "inventory_body_sha256"
    )
    _write(inventory_path, inventory)

    descriptor_path = tmp_path / "campaign-descriptor-v2.json"
    descriptor = {
        "schema_version": 2,
        "record_type": "glm52_sky_campaign_descriptor_v2",
        "run_id": run_id,
        "campaign_identity_sha256": HEX["campaign"],
        "campaign_descriptor_key": descriptor_key,
        "skypilot_version": "0.13.0",
        "task_name": "glm52-durable-campaign",
        "instance_type": "p5.48xlarge",
        "region": "us-west-2",
        "repo_tar_sha256": _sha(repo_tar),
        "artifacts": {
            "artifact_inventory_key": inventory_key,
            "artifact_inventory_sha256": _sha(inventory_path),
        },
    }
    descriptor["descriptor_body_sha256"] = _body_sha(
        descriptor, "descriptor_body_sha256"
    )
    _write(descriptor_path, descriptor)

    audit_path = tmp_path / "artifact-audit-v1.json"
    _write(
        audit_path,
        {
            "schema_version": 1,
            "record_type": "glm52_s3_artifact_audit_v1",
            "audit_pass": True,
            "run_id": run_id,
            "bucket": "fixture-bucket",
            "inventory_body_sha256": inventory["inventory_body_sha256"],
            "object_count": 0,
            "object_bytes": 0,
            "safetensors_object_count": 0,
            "safetensors_tensor_count": 0,
        },
    )

    readiness_path = tmp_path / "STAGED_CONTROL_PLANE_READY.json"
    manifest_body_sha = "b" * 64
    readiness: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_ready_v2",
        "run_id": run_id,
        "descriptor_key": descriptor_key,
        "descriptor_sha256": _sha(descriptor_path),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": HEX["campaign"],
        "bundle_manifest_key": (
            f"campaigns/{run_id}/submissions/test/bundle-manifests/"
            f"{manifest_body_sha}/bundle-manifest-v1.json"
        ),
        "bundle_manifest_file_sha256": "d" * 64,
        "bundle_manifest_body_sha256": manifest_body_sha,
        "bundle_manifest_version_id": "manifest-version-1",
        "staged_object_version_ids": {
            role: f"{role}-version-1"
            for role in (
                "repository_tar",
                "approval",
                "training_config",
                "watchdog",
                "artifact_inventory",
                "artifact_audit",
                "descriptor",
            )
        },
        "artifact_audit_key": audit_key,
        "artifact_audit_sha256": _sha(audit_path),
        "staged_at": "2026-07-25T20:00:00Z",
    }
    readiness["ready_body_sha256"] = _body_sha(readiness, "ready_body_sha256")
    _write(readiness_path, readiness)

    extracted_repo = tmp_path / "extracted" / "repo"
    task_path = (
        extracted_repo / "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
    )
    task_path.parent.mkdir(parents=True)
    task_path.write_text("name: glm52-durable-campaign\n")
    config_path = tmp_path / "skypilot-config.yaml"
    config_path.write_text("allowed_clouds: [aws]\n")
    validator_path = tmp_path / "validate_skypilot_fixture.py"
    validator_path.write_text(
        "\n".join(
            (
                "import argparse, json, pathlib",
                "p = argparse.ArgumentParser()",
                "p.add_argument('--task', type=pathlib.Path, required=True)",
                "p.add_argument('--config', type=pathlib.Path, required=True)",
                "a = p.parse_args()",
                "task = a.task.read_text()",
                "config = a.config.read_text()",
                "if 'name: glm52-durable-campaign' not in task:",
                "    p.error('task identity mismatch')",
                "if 'allowed_clouds: [aws]' not in config:",
                "    p.error('config identity mismatch')",
                "print(json.dumps({",
                "    'record_type': 'glm52_skypilot_parser_validation_v1',",
                "    'skypilot_version': '0.13.0',",
                "    'task': 'glm52-durable-campaign',",
                "    'instance_type': 'p5.48xlarge',",
                "    'region': 'us-west-2',",
                "    'status': 'passed',",
                "}, sort_keys=True))",
                "",
            )
        )
    )
    validation_path = tmp_path / "skypilot-validation.json"
    with validation_path.open("wb") as validation_handle:
        subprocess.run(
            [
                sys.executable,
                str(validator_path),
                "--task",
                str(task_path),
                "--config",
                str(config_path),
            ],
            stdout=validation_handle,
            check=True,
        )
    bootstrap_receipt = tmp_path / "bootstrap-rehearsal.log"
    bootstrap_receipt.write_text(
        "GPU-ENV-REHEARSAL-OK next=MLX-CUDA/H100\n"
        "SKYPILOT-BOOTSTRAP-REHEARSAL-OK\n"
    )
    return {
        "run_id": run_id,
        "descriptor_key": descriptor_key,
        "descriptor": descriptor_path,
        "repo_tar": repo_tar,
        "readiness_key": readiness_key,
        "readiness": readiness_path,
        "inventory": inventory_path,
        "audit": audit_path,
        "extracted_repo": extracted_repo,
        "task": task_path,
        "config": config_path,
        "validator": validator_path,
        "validation": validation_path,
        "bootstrap_receipt": bootstrap_receipt,
    }


def _write_args(fixture: dict[str, Path | str], output: Path) -> list[str]:
    return [
        "--output",
        str(output),
        "--bootstrap-receipt",
        str(fixture["bootstrap_receipt"]),
        "--descriptor",
        str(fixture["descriptor"]),
        "--descriptor-key",
        str(fixture["descriptor_key"]),
        "--repo-tar",
        str(fixture["repo_tar"]),
        "--readiness",
        str(fixture["readiness"]),
        "--readiness-key",
        str(fixture["readiness_key"]),
        "--inventory",
        str(fixture["inventory"]),
        "--audit",
        str(fixture["audit"]),
        "--extracted-repo",
        str(fixture["extracted_repo"]),
        "--skypilot-task",
        str(fixture["task"]),
        "--skypilot-config",
        str(fixture["config"]),
        "--skypilot-validation",
        str(fixture["validation"]),
        "--skypilot-python",
        sys.executable,
        "--skypilot-validator",
        str(fixture["validator"]),
        "--completed-at",
        "2026-07-25T21:02:03Z",
    ]


def _write_command(fixture: dict[str, Path | str], output: Path) -> list[str]:
    return ["bash", str(SCRIPT), "--write-evidence", *_write_args(fixture, output)]


def test_writer_atomically_persists_and_validates_canonical_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "evidence" / "rehearsal.json"
    result = subprocess.run(
        _write_command(fixture, output),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_bytes())
    assert value == {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_rehearsal_v2",
        "status": "passed_before_cuda_h100_boundary",
        "run_id": fixture["run_id"],
        "campaign_identity_sha256": HEX["campaign"],
        "descriptor_key": fixture["descriptor_key"],
        "descriptor_file_sha256": _sha(Path(fixture["descriptor"])),
        "descriptor_body_sha256": json.loads(
            Path(fixture["descriptor"]).read_bytes()
        )["descriptor_body_sha256"],
        "repo_tar_sha256": _sha(Path(fixture["repo_tar"])),
        "bootstrap_receipt_file_sha256": _sha(
            Path(fixture["bootstrap_receipt"])
        ),
        "staged_readiness_key": fixture["readiness_key"],
        "staged_readiness_file_sha256": _sha(Path(fixture["readiness"])),
        "staged_readiness_body_sha256": json.loads(
            Path(fixture["readiness"]).read_bytes()
        )["ready_body_sha256"],
        "artifact_inventory_key": json.loads(
            Path(fixture["descriptor"]).read_bytes()
        )["artifacts"]["artifact_inventory_key"],
        "artifact_inventory_file_sha256": _sha(Path(fixture["inventory"])),
        "artifact_inventory_body_sha256": json.loads(
            Path(fixture["inventory"]).read_bytes()
        )["inventory_body_sha256"],
        "artifact_audit_key": json.loads(
            Path(fixture["readiness"]).read_bytes()
        )["artifact_audit_key"],
        "artifact_audit_file_sha256": _sha(Path(fixture["audit"])),
        "extracted_repo_path": str(Path(fixture["extracted_repo"]).resolve()),
        "production_repo_path": "/opt/keep-campaign/repo",
        "production_resume_root": "/mnt/nvme/glm52-campaign",
        "skypilot_task_path": (
            "/opt/keep-campaign/repo/"
            "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
        ),
        "skypilot_task_file_sha256": _sha(Path(fixture["task"])),
        "skypilot_config_file_sha256": _sha(Path(fixture["config"])),
        "skypilot_validation_file_sha256": _sha(Path(fixture["validation"])),
        "skypilot_version": "0.13.0",
        "skypilot_task_name": "glm52-durable-campaign",
        "completed_at": "2026-07-25T21:02:03Z",
        "rehearsal_body_sha256": value["rehearsal_body_sha256"],
    }
    assert value["rehearsal_body_sha256"] == _body_sha(
        value, "rehearsal_body_sha256"
    )
    assert output.read_bytes() == _canonical(value) + b"\n"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    assert f"REHEARSAL_EVIDENCE_PATH={output.resolve()}" in result.stdout
    assert (
        f"REHEARSAL_EVIDENCE_SHA256={_sha(output)}" in result.stdout
    )

    validate = subprocess.run(
        ["bash", str(SCRIPT), "--validate-evidence", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    assert value["rehearsal_body_sha256"] in validate.stdout


def test_writer_fails_closed_without_output_for_foreign_or_tampered_input(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    readiness_path = Path(fixture["readiness"])
    readiness = json.loads(readiness_path.read_bytes())
    readiness["descriptor_sha256"] = "f" * 64
    readiness["ready_body_sha256"] = _body_sha(
        readiness, "ready_body_sha256"
    )
    _write(readiness_path, readiness)
    output = tmp_path / "evidence" / "rehearsal.json"

    result = subprocess.run(
        _write_command(fixture, output),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "readiness descriptor file identity mismatch" in result.stderr
    assert not output.exists()
    assert not output.parent.exists() or not list(
        output.parent.glob(f".{output.name}.*.tmp")
    )


def test_writer_rejects_historical_staged_readiness_v1_without_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    readiness_path = Path(fixture["readiness"])
    readiness = json.loads(readiness_path.read_bytes())
    for field in (
        "bundle_manifest_key",
        "bundle_manifest_file_sha256",
        "bundle_manifest_version_id",
        "staged_object_version_ids",
    ):
        readiness.pop(field)
    readiness["schema_version"] = 1
    readiness["record_type"] = "glm52_staged_control_plane_ready_v1"
    readiness["ready_body_sha256"] = _body_sha(
        readiness, "ready_body_sha256"
    )
    _write(readiness_path, readiness)
    output = tmp_path / "evidence" / "rehearsal.json"

    result = subprocess.run(
        _write_command(fixture, output),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "staged readiness" in result.stderr
    assert not output.exists()


def test_writer_reruns_validation_and_rejects_task_or_config_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    Path(fixture["task"]).write_text(
        "name: changed-after-validation\n"
    )
    Path(fixture["config"]).write_text("allowed_clouds: [gcp]\n")
    output = tmp_path / "rehearsal.json"

    result = subprocess.run(
        _write_command(fixture, output),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "exact SkyPilot task/config revalidation failed" in result.stderr
    assert not output.exists()


def test_bootstrap_sequence_is_behavioral_and_preserves_explicit_work_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    success = tmp_path / "bootstrap-success.sh"
    success.write_text(
        "#!/bin/sh\n"
        "echo GPU-ENV-REHEARSAL-OK next=MLX-CUDA/H100\n"
        "echo SKYPILOT-BOOTSTRAP-REHEARSAL-OK\n"
    )
    success.chmod(0o755)
    failure = tmp_path / "bootstrap-failure.sh"
    failure.write_text("#!/bin/sh\nexit 75\n")
    failure.chmod(0o755)

    failed_work = tmp_path / "failed-rehearsal"
    failed_output = failed_work / "evidence.json"
    failed_env = {
        **os.environ,
        "WORK": str(failed_work),
        "REHEARSAL_EVIDENCE_OUTPUT": str(failed_output),
        "KEEP_REHEARSAL_WORK": "0",
    }
    failed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--exercise-finalization",
            "--bootstrap-command",
            str(failure),
            "--",
            *_write_args(fixture, failed_output),
        ],
        env=failed_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert not failed_output.exists()

    succeeded_work = tmp_path / "successful-rehearsal"
    succeeded_output = succeeded_work / "evidence.json"
    succeeded_env = {
        **os.environ,
        "WORK": str(succeeded_work),
        "REHEARSAL_EVIDENCE_OUTPUT": str(succeeded_output),
        "KEEP_REHEARSAL_WORK": "0",
    }
    succeeded = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--exercise-finalization",
            "--bootstrap-command",
            str(success),
            "--",
            *_write_args(fixture, succeeded_output),
        ],
        env=succeeded_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert succeeded.returncode == 0, succeeded.stderr
    assert succeeded_output.is_file()
    value = json.loads(succeeded_output.read_bytes())
    receipt = succeeded_work / "bootstrap-rehearsal.log"
    assert value["bootstrap_receipt_file_sha256"] == _sha(receipt)
    assert "SKYPILOT-BOOTSTRAP-REHEARSAL-OK" in receipt.read_text()

    finalization_work = tmp_path / "failed-finalization"
    finalization_output = finalization_work / "evidence.json"
    Path(fixture["config"]).write_text("allowed_clouds: [gcp]\n")
    finalization = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--exercise-finalization",
            "--bootstrap-command",
            str(success),
            "--",
            *_write_args(fixture, finalization_output),
        ],
        env={
            **os.environ,
            "WORK": str(finalization_work),
            "REHEARSAL_EVIDENCE_OUTPUT": str(finalization_output),
            "KEEP_REHEARSAL_WORK": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert finalization.returncode != 0
    assert not finalization_output.exists()
    assert not list(finalization_work.glob(".evidence.json.*.tmp"))


def test_validator_rejects_unknown_fields_and_self_hash_tampering(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "rehearsal.json"
    subprocess.run(_write_command(fixture, output), check=True)
    value = json.loads(output.read_bytes())
    value["unexpected"] = True
    _write(output, value)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--validate-evidence", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "field inventory" in result.stderr

    del value["unexpected"]
    value["completed_at"] = "2026-07-25T21:02:04Z"
    _write(output, value)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--validate-evidence", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "body SHA-256 mismatch" in result.stderr


def test_main_rehearsal_writes_evidence_only_after_bootstrap_boundary() -> None:
    source = SCRIPT.read_text()
    assert "SKYPILOT-BOOTSTRAP-REHEARSAL-OK" in source
    assert source.rindex(
        "SKYPILOT-BOOTSTRAP-REHEARSAL-OK"
    ) < source.rindex(
        '"$SCRIPT_PATH" --write-evidence'
    )
    assert "REHEARSAL_EVIDENCE_OUTPUT" in source
    assert "GLM52_BOOTSTRAP_REHEARSAL=1" in source


def test_rehearsal_script_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
